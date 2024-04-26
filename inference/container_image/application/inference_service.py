import logging
from typing import Dict
import numpy as np
import onnxruntime as ort
import flask
import dill
import os
import time

from sklearn_pandas import DataFrameMapper
import pandas as pd
from flask import Flask, request, jsonify

class FraudDatasetTransformer:
    def __init__(self): ...

    def transform(self, dataset: pd.DataFrame, mapper: DataFrameMapper):
        """
        
        dropped columns:
            - mcc
            - zip
            - merchant state

        Args:
            dataset (pd.DataFrame): _description_
            mapper (DataFrameMapper): _description_

        Returns:
            _type_: _description_
        """
        tdf = dataset.copy()
        tdf["merchant name"] = tdf["merchant name"].astype(str)
        tdf.drop(["mcc", "zip", "merchant state"], axis=1, inplace=True)
        tdf.sort_values(by=["user", "card"], inplace=True)
        tdf.reset_index(inplace=True, drop=True)

        tdf = mapper.transform(tdf)
        return tdf


def get_df_mapper():
    with open(os.path.join("encoders", "data", "mapper.pkl"), "rb") as f:
        t_mapper = dill.load(f)
        return t_mapper


class FraudDetectionPredictor:
    MODEL_PATH = 'model/model.onnx'
    
    def __init__(self, name: str) -> None:
        self.name = name
        
    def load(self):
        logging.info(
            f"Starting a CPU inference session using the model {FraudDetectionPredictor.MODEL_PATH}"
        )
        
        self.INFERENCE_SESSION = ort.InferenceSession(
            FraudDetectionPredictor.MODEL_PATH, providers=["CPUExecutionProvider"]
        )
    
    def predict(self, vdf: pd.DataFrame) -> pd.DataFrame:

        # Data preparation
        x = vdf.drop(vdf.columns.values[0], axis=1).to_numpy().astype(np.float32)
        y = np.array([vdf[vdf.columns.values[0]].iloc[0]])

        # Get input name and shape
        input_name = self.INFERENCE_SESSION.get_inputs()[0].name
        sequence_length = self.INFERENCE_SESSION.get_inputs()[0].shape[1]  # Model's expected sequence length
        num_features = self.INFERENCE_SESSION.get_inputs()[0].shape[2]   # Model's expected number of features per sequence element

        # Check if the original features match the required total features
        original_features = x.shape[1]
        if original_features < num_features:
            # If fewer, we may need to pad or adjust the data; this is situational and may not be exactly correct without more context
            # For now, let's assume padding with zeros is acceptable
            x_padded = np.pad(
                x,
                ((0, 0), (0, num_features - original_features)),
                mode="constant",
                constant_values=0,
            )
        else:
            # If it matches or exceeds, truncate or reshape accordingly (though unusual for a single data point)
            x_padded = x[:, :num_features]

        # Reshape to [1, sequence_length, num_features], replicating the single data point across the new sequence length
        x_reshaped = np.tile(x_padded, (sequence_length, 1)).reshape(
            1, sequence_length, num_features
        )

        # Run the model
        outputs = self.INFERENCE_SESSION.run(None, {input_name: x_reshaped})

        # Handle response
        pred = outputs[0][0][0]
        logging.info(f"Actual ({y[0]}) vs. Prediction ({round(pred, 3)} => {int(round(pred, 0))})")

        return pred
    
    
# setup Flask App
app = Flask(__name__)
predictor = FraudDetectionPredictor('fraud_detection')
dataset_transfomer = FraudDatasetTransformer()
mapper = get_df_mapper()
predictor.load()


def do_predict(test_data: Dict):
    ret = {}
    start = time.time()
    
    # setup transformer
    test = pd.DataFrame([test_data])
    vdf = dataset_transfomer.transform(test, mapper)
    
    # run inference
    result = predictor.predict(vdf)
    end = time.time()
    
    # prepare output object
    ret['result'] = result
    total_time = (end - start) * 1000 # ms
    ret['time'] = round(total_time, 3)
    
    return ret
    

@app.route('/api/model/predict', methods=['POST'])
def predict_endpoint():
    try:
        input_data = request.get_json()
        if not input_data:
            raise ValueError('No input data provided')
        result = do_predict(input_data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

if __name__ == '__main__':
    SERVICE_PORT = os.getenv("SERVICE_PORT", default="5000")
    DEBUG_MODE = eval(os.getenv("DEBUG_MODE", default="True"))
    app.run(
        host="0.0.0.0", port=SERVICE_PORT, debug=DEBUG_MODE
    )