import json
import subprocess
import re
import sys
import argparse


onnx_endpoint = 'https://demo-application-onnx-user-example-com.apps.b2s001.pbm.ihost.com'

def load_payloads(file_path):
    """Load JSON payloads from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def run_curl_command(payload, model_endpoint):
    """Run curl command with a given JSON payload."""
    curl_command = [
        'curl', '-s',
        '-k', '-X', 'POST',
        f'{model_endpoint}/api/model/predict',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps(payload)
    ]
    result = subprocess.run(curl_command, capture_output=True, text=True)
    return result.stdout

def parse_output(output):
    """Parse the curl output to extract the JSON response and parse the time."""
    try:
        # Parse the JSON output to extract the 'time' field
        json_output = json.loads(output)
        return json_output['time']
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")
        return None
    except KeyError:
        print("No 'time' key found in the JSON response.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test ONNX endpoint")
    parser.add_argument('-d', '--data', required=True, help='Path to data file')
    parser.add_argument('-m', '--model', required=True, help='Model inference endpoint')
    
    args = parser.parse_args()

    file_path = args.data
    model_endpoint = args.model
    payloads = load_payloads(file_path)
    times = []

    for payload in payloads:
        output = run_curl_command(payload, model_endpoint)
        print(output)
        time_taken = parse_output(output)
        if time_taken is not None:
            times.append(float(time_taken))

    if times:
        average_time = sum(times) / len(times)
        print(f"Average Request Time: {average_time} milliseconds")
    else:
        print("No valid times were collected.")

if __name__ == '__main__':
    main()