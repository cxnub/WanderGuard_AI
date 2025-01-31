import json
import boto3
import os

endpoint_name = os.environ["SAGEMAKER_ENDPOINT_NAME"]

runtime_client = boto3.client('sagemaker-runtime')

test_data = {
    "heart_rate": 75,
    "speed": 2.5,
    "distance_from_safe_zone": 50,
    "timestamp": "2024-02-01T10:30:00"
}

response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(test_data)
)

prediction = json.loads(response["Body"].read().decode())
print(f"Prediction: {prediction}")
