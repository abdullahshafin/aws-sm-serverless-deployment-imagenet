# AWS SageMaker Serverless Deployment - ImageNet Classification

A simple, minimal tutorial demonstrating how to deploy ImageNet classification models (ResNet, EfficientNet, DenseNet) to AWS SageMaker Serverless Inference using PyTorch. The deployment accepts presigned URLs as input and returns top-5 prediction classes with probabilities.

## Overview

This tutorial shows you how to:
- Deploy pretrained ImageNet models to SageMaker Serverless Inference
- Accept image URLs (including S3 presigned URLs) as input
- Return top-5 classification predictions with probabilities
- Use PyTorch as the deep learning framework

### Features

- **Simple & Minimal**: Clean, tutorial-focused code with minimal complexity
- **Multiple Models**: Support for ResNet, EfficientNet, DenseNet architectures
- **Serverless**: Pay only for inference time, auto-scales with traffic
- **URL-based Input**: Accepts presigned URLs for easy integration
- **Production-Ready**: Proper error handling and logging

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured with credentials
- Python 3.9+
- IAM role with SageMaker permissions

### Required AWS Permissions

Your IAM role needs the following permissions:
- `AmazonSageMakerFullAccess` (or equivalent)
- S3 access for model artifacts (if needed)

## Installation

### Option 1: Quick Start Script (Recommended)

Run the automated setup script:
```bash
git clone https://github.com/abdullahshafin/aws-sm-serverless-deployment-imagenet.git
cd aws-sm-serverless-deployment-imagenet
./quickstart.sh
```

The script will:
- Check prerequisites (Python, AWS CLI, credentials)
- Install required dependencies
- Guide you through deployment

### Option 2: Manual Installation

1. Clone this repository:
```bash
git clone https://github.com/abdullahshafin/aws-sm-serverless-deployment-imagenet.git
cd aws-sm-serverless-deployment-imagenet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install SageMaker SDK:
```bash
pip install sagemaker boto3
```

## Project Structure

```
.
├── inference.py        # SageMaker inference script with model loading and prediction
├── deploy.py          # Deployment script for creating serverless endpoint
├── test_endpoint.py   # Test script for invoking the endpoint
├── requirements.txt   # Python dependencies for inference
├── quickstart.sh      # Automated setup and deployment script
├── tutorial.ipynb     # Interactive Jupyter notebook tutorial
├── EXAMPLES.md        # Input/output format examples and usage patterns
├── CONTRIBUTING.md    # Contribution guidelines
└── README.md         # This file
```

## Quick Start

### Step 1: Deploy the Model

Deploy a ResNet50 model to SageMaker Serverless:

```bash
python deploy.py \
    --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE \
    --model-name resnet50 \
    --region us-east-1
```

**Available Models:**
- `resnet50` (default) - ResNet-50
- `resnet18` - ResNet-18 (lighter, faster)
- `efficientnet_b0` - EfficientNet-B0
- `densenet121` - DenseNet-121

**Deployment Options:**
- `--memory-size`: Memory in MB (1024, 2048, 3072, 4096, 5120, 6144) - default: 4096
- `--max-concurrency`: Max concurrent invocations (1-200) - default: 10
- `--endpoint-name`: Custom endpoint name (optional)

The deployment takes several minutes. Once complete, you'll see the endpoint name.

### Step 2: Test the Endpoint

#### Option A: Using a Public Image URL

```bash
python test_endpoint.py \
    --endpoint-name YOUR_ENDPOINT_NAME \
    --image-url "https://example.com/image.jpg" \
    --region us-east-1
```

#### Option B: Using S3 Presigned URL (Recommended)

First, upload an image to S3, then:

```bash
python test_endpoint.py \
    --endpoint-name YOUR_ENDPOINT_NAME \
    --s3-bucket your-bucket-name \
    --s3-key path/to/image.jpg \
    --region us-east-1
```

The script automatically generates a presigned URL and invokes the endpoint.

### Step 3: View Results

The endpoint returns the top-5 predictions with probabilities:

```
==============================================================
Prediction Results:
==============================================================
1. golden retriever               85.23%
2. Labrador retriever             8.45%
3. cocker spaniel                 2.31%
4. Irish setter                   1.89%
5. English setter                 0.76%
==============================================================
```

## How It Works

### Inference Script (`inference.py`)

The inference script implements four key functions required by SageMaker:

1. **`model_fn(model_dir)`**: Loads the PyTorch model
   - Supports multiple architectures via environment variable
   - Uses pretrained ImageNet weights from torchvision
   - Configures image preprocessing transforms
   - Loads ImageNet class labels

2. **`input_fn(request_body, content_type)`**: Deserializes input
   - Accepts JSON payload with `url` field
   - Validates input format

3. **`predict_fn(input_data, model)`**: Runs inference
   - Downloads image from presigned URL
   - Preprocesses image (resize, normalize)
   - Runs model inference
   - Returns top-5 predictions with probabilities

4. **`output_fn(prediction, accept)`**: Serializes output
   - Formats predictions as JSON
   - Returns class names and probabilities

### Deployment Script (`deploy.py`)

The deployment script:
- Creates a PyTorchModel with the inference script
- Configures serverless inference settings (memory, concurrency)
- Deploys to a SageMaker serverless endpoint
- Supports multiple model architectures

### Test Script (`test_endpoint.py`)

The test script:
- Generates presigned URLs for S3 objects (optional)
- Invokes the SageMaker endpoint with image URL
- Displays formatted prediction results

## Cost Considerations

SageMaker Serverless Inference charges based on:
- **Compute**: Per millisecond of inference time
- **Memory**: Based on configured memory size

You only pay when the endpoint is processing requests - no charges when idle.

**Example costs** (us-east-1, as of 2024):
- 4GB memory: ~$0.0000133 per second
- No charges for idle time

For occasional inference (< 100 requests/day), serverless is typically more cost-effective than dedicated instances.

## Advanced Usage

### Using Different Models

To deploy a different model:

```bash
python deploy.py \
    --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE \
    --model-name efficientnet_b0 \
    --memory-size 3072 \
    --region us-east-1
```

### Adjusting Serverless Configuration

For higher throughput:

```bash
python deploy.py \
    --role-arn YOUR_ROLE_ARN \
    --memory-size 6144 \
    --max-concurrency 50
```

### Programmatic Invocation

```python
import boto3
import json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

payload = {
    "url": "https://your-presigned-url"
}

response = runtime.invoke_endpoint(
    EndpointName="your-endpoint-name",
    ContentType="application/json",
    Body=json.dumps(payload)
)

result = json.loads(response["Body"].read())
print(result["predictions"])
```

## Cleanup

To delete the endpoint and avoid charges:

```python
import boto3

client = boto3.client("sagemaker", region_name="us-east-1")
client.delete_endpoint(EndpointName="your-endpoint-name")
client.delete_endpoint_config(EndpointConfigName="your-endpoint-name")
client.delete_model(ModelName="your-model-name")
```

Or use AWS Console:
1. Navigate to SageMaker → Endpoints
2. Select your endpoint
3. Actions → Delete

## Troubleshooting

### Common Issues

**1. Deployment fails with "Role not found"**
- Ensure your IAM role ARN is correct
- Verify the role has SageMaker permissions

**2. Endpoint invocation times out**
- Increase memory size in deployment config
- Check image URL is accessible

**3. "Model loading error"**
- Verify requirements.txt includes all dependencies
- Check PyTorch/torchvision versions are compatible

**4. Image download fails**
- Ensure presigned URL is not expired (default: 1 hour)
- Verify network connectivity from SageMaker

### Logs and Debugging

View endpoint logs in CloudWatch:
1. AWS Console → CloudWatch → Log groups
2. Search for `/aws/sagemaker/Endpoints/[endpoint-name]`

## Supported Models

| Model | Description | Memory | Speed |
|-------|-------------|--------|-------|
| resnet18 | ResNet-18 | Lowest | Fastest |
| resnet50 | ResNet-50 | Medium | Fast |
| efficientnet_b0 | EfficientNet-B0 | Medium | Fast |
| densenet121 | DenseNet-121 | Medium | Medium |

## Limitations

- Input images must be accessible via HTTP/HTTPS URL
- Maximum image size: Limited by memory configuration
- Cold start: First request after idle period takes longer (~10-60s)
- Models use pretrained ImageNet weights (fixed, not customizable in this tutorial)

## Next Steps

- **Custom Models**: Modify `inference.py` to load custom trained models
- **Batch Processing**: Implement batch prediction for multiple images
- **Custom Classes**: Train on custom datasets and update class labels
- **Monitoring**: Set up CloudWatch alarms for endpoint metrics
- **CI/CD**: Automate deployment with AWS CodePipeline

## Resources

- [AWS SageMaker Serverless Inference Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
- [PyTorch on SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [ImageNet Dataset](https://www.image-net.org/)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the terms included in the LICENSE file.

## Acknowledgments

- PyTorch team for pretrained ImageNet models
- AWS SageMaker team for serverless inference capabilities
- ImageNet dataset maintainers
