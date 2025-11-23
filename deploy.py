"""
Deployment script for creating a SageMaker Serverless Inference Endpoint.

This script creates a SageMaker serverless endpoint for ImageNet classification.
"""

import argparse
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig
import time


def deploy_serverless_endpoint(
    role_arn,
    model_name="resnet50",
    endpoint_name=None,
    memory_size=4096,
    max_concurrency=10,
    region="us-east-1"
):
    """
    Deploy a PyTorch model to SageMaker Serverless Inference.
    
    Args:
        role_arn (str): IAM role ARN with SageMaker permissions
        model_name (str): Model architecture (resnet50, resnet18, efficientnet_b0, densenet121)
        endpoint_name (str): Name for the endpoint (default: auto-generated)
        memory_size (int): Memory size in MB (1024, 2048, 3072, 4096, 5120, 6144)
        max_concurrency (int): Maximum concurrent invocations (1-200)
        region (str): AWS region
        
    Returns:
        dict: Deployment information including endpoint name
    """
    print(f"Starting deployment of {model_name} to SageMaker Serverless...")
    
    # Initialize SageMaker session
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Generate endpoint name if not provided
    if endpoint_name is None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        endpoint_name = f"imagenet-{model_name}-{timestamp}"
    
    print(f"Endpoint name: {endpoint_name}")
    
    # Create PyTorch Model
    # Note: For this tutorial, we use pretrained models loaded in inference.py
    # No model artifacts are needed as we load from torchvision
    pytorch_model = PyTorchModel(
        model_data=None,  # Not needed for pretrained models
        role=role_arn,
        entry_point="inference.py",
        framework_version="1.13.1",
        py_version="py39",
        env={
            "MODEL_NAME": model_name
        },
        sagemaker_session=sagemaker_session
    )
    
    # Configure serverless inference
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=memory_size,
        max_concurrency=max_concurrency,
    )
    
    print(f"Deploying with memory: {memory_size}MB, max concurrency: {max_concurrency}")
    print("This may take several minutes...")
    
    # Deploy the model
    predictor = pytorch_model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=endpoint_name
    )
    
    print(f"\n{'='*60}")
    print("Deployment successful!")
    print(f"{'='*60}")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Model: {model_name}")
    print(f"Region: {region}")
    print(f"\nYou can now invoke the endpoint using the test script:")
    print(f"python test_endpoint.py --endpoint-name {endpoint_name} --image-url <presigned_url>")
    print(f"{'='*60}\n")
    
    return {
        "endpoint_name": endpoint_name,
        "predictor": predictor,
        "model_name": model_name,
        "region": region
    }


def main():
    """Main function to parse arguments and deploy."""
    parser = argparse.ArgumentParser(
        description="Deploy ImageNet model to SageMaker Serverless Inference"
    )
    
    parser.add_argument(
        "--role-arn",
        type=str,
        required=True,
        help="IAM role ARN with SageMaker permissions"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet18", "efficientnet_b0", "densenet121"],
        help="Model architecture to deploy (default: resnet50)"
    )
    
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=None,
        help="Custom endpoint name (default: auto-generated)"
    )
    
    parser.add_argument(
        "--memory-size",
        type=int,
        default=4096,
        choices=[1024, 2048, 3072, 4096, 5120, 6144],
        help="Memory size in MB (default: 4096)"
    )
    
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum concurrent invocations (1-200, default: 10)"
    )
    
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    
    args = parser.parse_args()
    
    # Deploy the endpoint
    deployment_info = deploy_serverless_endpoint(
        role_arn=args.role_arn,
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        memory_size=args.memory_size,
        max_concurrency=args.max_concurrency,
        region=args.region
    )
    
    return deployment_info


if __name__ == "__main__":
    main()
