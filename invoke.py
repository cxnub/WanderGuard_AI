import json
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

endpoint_name = "wanderguard-ai-predictor"

runtime_client = boto3.client(
    'sagemaker-runtime',
    region_name='us-east-1',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    aws_session_token=os.environ['AWS_SESSION_TOKEN']
)

sagemaker = boto3.client("sagemaker", region_name="us-east-1")



test_data = {
    "heart_rate": 75,
    "speed": 2.5,
    "distance_from_safe_zone": 50,
    "timestamp": "2024-02-01T10:30:00"
}

response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(test_data),
    InferenceComponentName="wanderguard-predictor-20250131-145747",
)

prediction = json.loads(response["Body"].read().decode())
print(f"Prediction: {prediction}")
