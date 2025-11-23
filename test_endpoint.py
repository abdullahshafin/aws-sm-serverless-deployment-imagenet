"""
Test script for invoking the SageMaker Serverless Inference Endpoint.

This script demonstrates how to invoke the deployed endpoint with a presigned URL.
"""

import argparse
import json
import boto3
from botocore.exceptions import ClientError


def invoke_endpoint(endpoint_name, image_url, region="us-east-1"):
    """
    Invoke the SageMaker endpoint with a presigned URL.
    
    Args:
        endpoint_name (str): Name of the SageMaker endpoint
        image_url (str): Presigned URL or public URL of the image
        region (str): AWS region
        
    Returns:
        dict: Prediction results
    """
    print(f"Invoking endpoint: {endpoint_name}")
    print(f"Image URL: {image_url}")
    
    # Create SageMaker runtime client
    runtime_client = boto3.client("sagemaker-runtime", region_name=region)
    
    # Prepare input payload
    payload = {
        "url": image_url
    }
    
    try:
        # Invoke endpoint
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response["Body"].read().decode())
        
        print(f"\n{'='*60}")
        print("Prediction Results:")
        print(f"{'='*60}")
        
        predictions = result.get("predictions", [])
        for i, pred in enumerate(predictions, 1):
            class_name = pred["class"]
            probability = pred["probability"]
            print(f"{i}. {class_name:30s} {probability*100:6.2f}%")
        
        print(f"{'='*60}\n")
        
        return result
        
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        print(f"\nError invoking endpoint:")
        print(f"Code: {error_code}")
        print(f"Message: {error_message}\n")
        raise
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}\n")
        raise


def generate_presigned_url(bucket, key, region="us-east-1", expiration=3600):
    """
    Generate a presigned URL for an S3 object.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        region (str): AWS region
        expiration (int): URL expiration time in seconds (default: 3600)
        
    Returns:
        str: Presigned URL
    """
    s3_client = boto3.client("s3", region_name=region)
    
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")
        raise


def main():
    """Main function to parse arguments and test endpoint."""
    parser = argparse.ArgumentParser(
        description="Test SageMaker Serverless Inference Endpoint"
    )
    
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="Name of the SageMaker endpoint"
    )
    
    parser.add_argument(
        "--image-url",
        type=str,
        help="Presigned URL or public URL of the image"
    )
    
    parser.add_argument(
        "--s3-bucket",
        type=str,
        help="S3 bucket name (alternative to --image-url)"
    )
    
    parser.add_argument(
        "--s3-key",
        type=str,
        help="S3 object key (alternative to --image-url)"
    )
    
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    
    args = parser.parse_args()
    
    # Determine the image URL
    if args.image_url:
        image_url = args.image_url
    elif args.s3_bucket and args.s3_key:
        print(f"Generating presigned URL for s3://{args.s3_bucket}/{args.s3_key}")
        image_url = generate_presigned_url(
            args.s3_bucket, 
            args.s3_key, 
            region=args.region
        )
        print(f"Presigned URL generated (expires in 1 hour)")
    else:
        parser.error("Either --image-url or both --s3-bucket and --s3-key must be provided")
    
    # Invoke the endpoint
    result = invoke_endpoint(
        endpoint_name=args.endpoint_name,
        image_url=image_url,
        region=args.region
    )
    
    return result


if __name__ == "__main__":
    main()
