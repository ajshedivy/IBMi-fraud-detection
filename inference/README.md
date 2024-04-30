# Deploy standalone Inference Application using ONNX

### Setp 1: Create a Project (optional)

Create a new OpenShift project where your application will reside, or you can use an existing one.

```bash
oc new-project your-project-name
```

or switch to existing project:

```bash
oc project existing-project-name
```

### Step 2: Prepare YAML file

Copy the contents in `deploy-onnx-predictor-knative.yaml` to a single file on openshift

### Step 3: Apply the YAML

```bash
oc apply -f deploy-onnx-predictor-knative.yaml
```

### Step 4: Verify deployment:
Check the status of the deployed service.

```bash
oc get ksvc demo-application-onnx
```

### Step 5: Access the application Endpoint:

Once the service is ready, find the URL to access your application using:

```bash
oc get ksvc demo-application-onnx -o=jsonpath='{.status.url}'
```

## Test model endpoint

To test the model enpoint, create a test container in openshift with curl installed (optional)

curl test command:

```bash
curl  -s -k -X POST https://demo-application-onnx-user-example-com.apps.b2s001.pbm.ihost.com/api/model/predict -H "Content-Type: application/json" -d '{
    "index": 1,
    "user": 2,
    "card": 4,
    "year": 2022,
    "month": 9,
    "day": 2,
    "time": "14:09",
    "amount": "$149345.84",
    "use chip": "Online Transaction",
    "merchant name": 3452760747765970571,
    "merchant city": "ONLINE",
    "merchant state": "",
    "zip": 0,
    "mcc": 3174,
    "errors?": "",
    "is fraud?": "Yes"
}'

```

### Performance Test from IBM i (same cluster):

```bash
{
  "result": 0.24492061138153076,
  "time": 51.978
}
```

### Run Performance showcase

To Run the performance showcase, download the `test_onnx_endpoint.py` and `test_data.json`. Ensure Python is installed.

```bash
wget https://raw.githubusercontent.com/ajshedivy/IBMi-fraud-detection/test/latency/inference/test_onnx_endpoint.py -O test_onnx_endpoint.py
wget https://raw.githubusercontent.com/ajshedivy/IBMi-fraud-detection/test/latency/inference/test_data.json -O test_data.json
```
Usage:
```bash
bash-5.1$ python test_onnx_endpoint.py -h
usage: test_onnx_endpoint.py [-h] -d DATA -m MODEL [-t]

Test ONNX endpoint

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Path to data file
  -m MODEL, --model MODEL
                        Model inference endpoint
  -t, --transformed     flag for using transformed data
```

Run the script:

```bash
bash-5.1$ python test_onnx_endpoint.py -d test_data.json -m 'https://demo-application-onnx-user-example-com.apps.b2s001.pbm.ihost.com' 
{
  "result": 0.0007616877555847168,
  "time": 53.289
}

{
  "result": 0.24492061138153076,
  "time": 50.209
}

{
  "result": 0.004086315631866455,
  "time": 49.845
}

{
  "result": 0.0003190934658050537,
  "time": 51.333
}

{
  "result": 0.9969544410705566,
  "time": 50.149
}

Average Request Time: 50.965 milliseconds
```

### Run Test Showcase with preprocessed Inputs

There is also an option to run the tests with preprocessed transactions. First download the `transformed_data.json` file:
```bash
wget https://raw.githubusercontent.com/ajshedivy/IBMi-fraud-detection/test/latency/inference/transformed_data.json -O transformed_data.json
```

Then run the test script by passing in the new json file, with the additional `-t` flag:

```bash
bash-5.1$ python test_onnx_endpoint.py -t -d transformed_data.json -m https://demo-application-onnx-test-fraud-model.apps.b2s001.pbm.ihost.com
```
Outputs:

```bash
{
  "result": 0.0007616877555847168,
  "time": 4.603
}

{
  "result": 0.24492061138153076,
  "time": 4.102
}

{
  "result": 0.004086315631866455,
  "time": 4.287
}

{
  "result": 0.0003190934658050537,
  "time": 2.758
}

{
  "result": 0.9969544410705566,
  "time": 2.741
}

Average Request Time: 3.6982 milliseconds
```

