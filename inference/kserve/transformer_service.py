import argparse
import http
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import joblib
import kserve
import numpy as np
import pandas as pd
from fastapi import HTTPException
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferResponse
from kserve.protocol.infer_type import InferInput, InferRequest, InferResponse


class CreditRiskTransformer(kserve.Model):
    PREPROCESSOR_PATH = "/mnt/models/preprocessor.joblib"

    def __init__(self, name: str, predictor_host: str, protocol: str):
        super().__init__(name)
        self.preprocessor = None
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.load()
        self.target_names: List[str] = json.loads(os.environ["TARGET_NAMES"])
        self.ready = True

    def load(self):
        logging.info(
            f"Loading preprocessor pipeline from {CreditRiskTransformer.PREPROCESSOR_PATH}"
        )
        self.preprocessor = joblib.load(CreditRiskTransformer.PREPROCESSOR_PATH)

    def preprocess(
        self, inputs: Dict, headers: Optional[Dict[str, str]] = None
    ) -> InferRequest:
        """Preprocesses the incomming json"""

        if not isinstance(inputs, Dict):
            raise HTTPException(
                status_code=http.HTTPStatus.BAD_REQUEST,
                detail="The provided input must be a json dictionary"
            )

        try:
            logging.debug(f"Preprocessing request with batch size {len(inputs)}")
            X = self.preprocessor.transform(pd.DataFrame(inputs)).astype(np.float32)
        except ValueError as e:
            logging.exception(e)
            raise HTTPException(
                status_code=http.HTTPStatus.BAD_REQUEST,
                detail=f"Invalid request: {str(e)}",
            ) from None

        logging.debug("Building inference request...")
        input_0 = InferInput(name="input_1", shape=X.shape, datatype="FP32", data=X)
        request = InferRequest(
            model_name=self.name, infer_inputs=[input_0], request_id=None
        )
        return request

    def postprocess(
        self,
        response: Union[Dict, ModelInferResponse, InferResponse],
        headers: Dict[str, str],
    ) -> Dict[Any, Any]:
        """Convert model outputs from the prediction to the JSON response"""

        if isinstance(response, ModelInferResponse):
            response = InferResponse.from_grpc(response)
        elif not isinstance(response, InferResponse):
            raise NotImplementedError(f"Unsupported response type {type(response)}")

        # Predictor responses are class label ints (1=YES, 0=No), not scores
        tensor = response.outputs[0].as_numpy()
        payload = {"predictions": [self.target_names[int(score)] for score in tensor]}

        return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    # This arg is automatically provided by the inferenceservice
    # It contains the service host name for the predictor service.
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function", required=True
    )

    # This arg is automatically provided by the inferenceservice
    # it contains the name of the inference service resource.
    parser.add_argument(
        "--model_name", help="The name that the model is served under.", required=True
    )

    # Deterimes whether REST or GRPC will be used to communicate with the predictor
    # The default is REST/JSON format. The value provided must match the interface
    # that is exposed by the predictor.
    parser.add_argument(
        "--protocol", help="The protocol for the predictor", default="v2"
    )

    args, _ = parser.parse_known_args()

    model = CreditRiskTransformer(
        name=args.model_name, predictor_host=args.predictor_host, protocol=args.protocol
    )
    kserve.ModelServer().start([model])