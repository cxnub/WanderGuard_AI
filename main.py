import boto3
import sagemaker
import subprocess

# Setup
client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")
boto_session = boto3.session.Session()
s3 = boto_session.resource("s3")
region = boto_session.region_name
print(region)
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::716513171636:role/LabRole"

# Build tar file with model data + inference code
bashCommand = "tar -cvpzf model.tar.gz model.pkl preprocessor.pkl inference.py"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Bucket for model artifacts
default_bucket = sagemaker_session.default_bucket()
print(default_bucket)

# Upload tar.gz to bucket
model_artifacts = f"s3://{default_bucket}/model.tar.gz"
response = s3.meta.client.upload_file("model.tar.gz", default_bucket, "model.tar.gz")

# retrieve sklearn image
image_uri = sagemaker.image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.t3.medium",
)

# Step 1: Model Creation
model_name = "wanderguard-predictor"
print("Model name: " + model_name)
create_model_response = client.create_model(
    ModelName=model_name,
    Containers=[
        {
            "Image": image_uri,
            "Mode": "SingleModel",
            "ModelDataUrl": model_artifacts,
            "Environment": {
                "SAGEMAKER_SUBMIT_DIRECTORY": model_artifacts,
                "SAGEMAKER_PROGRAM": "inference.py",
            },
        }
    ],
    ExecutionRoleArn=role,
)
print("Model Arn: " + create_model_response["ModelArn"])