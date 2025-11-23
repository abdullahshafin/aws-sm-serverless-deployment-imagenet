#!/bin/bash

# Quick start script for SageMaker Serverless ImageNet Deployment
# This script helps verify prerequisites and guides users through initial setup

set -e

echo "========================================================"
echo "SageMaker Serverless ImageNet Deployment - Quick Start"
echo "========================================================"
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python 3 found: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

# Check AWS CLI
echo ""
echo "Checking AWS CLI..."
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version 2>&1 | awk '{print $1}')
    echo "✓ AWS CLI found: $AWS_VERSION"
    
    # Check AWS credentials
    if aws sts get-caller-identity &> /dev/null; then
        AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
        AWS_USER=$(aws sts get-caller-identity --query Arn --output text)
        echo "✓ AWS credentials configured"
        echo "  Account: $AWS_ACCOUNT"
        echo "  User: $AWS_USER"
    else
        echo "✗ AWS credentials not configured. Run 'aws configure' to set up."
        exit 1
    fi
else
    echo "✗ AWS CLI not found. Please install AWS CLI."
    exit 1
fi

# Check if pip is available
echo ""
echo "Checking pip..."
if command -v pip3 &> /dev/null; then
    echo "✓ pip3 found"
else
    echo "✗ pip3 not found. Please install pip."
    exit 1
fi

# Install dependencies
echo ""
echo "========================================================"
echo "Installing Python dependencies..."
echo "========================================================"
echo ""
echo "Recommendation: Consider using a virtual environment:"
echo "  python3 -m venv venv"
echo "  source venv/bin/activate"
echo ""
read -p "Install globally or skip? (y/n): " install_global

if [ "$install_global" = "y" ]; then
    pip3 install -q sagemaker boto3
    echo "✓ SageMaker SDK and boto3 installed"
else
    echo "Skipping installation. Install manually with:"
    echo "  pip3 install sagemaker boto3"
fi

# Check for IAM role
echo ""
echo "========================================================"
echo "IAM Role Setup"
echo "========================================================"
echo ""
echo "You need an IAM role with SageMaker permissions."
echo ""
echo "To create a role:"
echo "1. Go to AWS Console → IAM → Roles"
echo "2. Create role → SageMaker → SageMaker - Execution"
echo "3. Add policies: AmazonSageMakerFullAccess"
echo "4. Name your role (e.g., 'SageMakerExecutionRole')"
echo ""
echo "Once created, copy the Role ARN. It looks like:"
echo "  arn:aws:iam::123456789012:role/SageMakerExecutionRole"
echo ""

# Prompt for role ARN
read -p "Enter your SageMaker IAM Role ARN (or press Enter to skip): " ROLE_ARN

if [ -z "$ROLE_ARN" ]; then
    echo ""
    echo "Skipping deployment. You can deploy later with:"
    echo "  python deploy.py --role-arn YOUR_ROLE_ARN"
else
    echo ""
    echo "========================================================"
    echo "Deploying Model"
    echo "========================================================"
    echo ""
    echo "Starting deployment with Role ARN: $ROLE_ARN"
    echo "This will take several minutes..."
    echo ""
    
    python3 deploy.py \
        --role-arn "$ROLE_ARN" \
        --model-name resnet50 \
        --memory-size 4096 \
        --max-concurrency 10
    
    echo ""
    echo "✓ Deployment complete!"
fi

echo ""
echo "========================================================"
echo "Next Steps"
echo "========================================================"
echo ""
echo "1. Test the endpoint:"
echo "   python test_endpoint.py \\"
echo "     --endpoint-name YOUR_ENDPOINT_NAME \\"
echo "     --image-url https://example.com/image.jpg"
echo ""
echo "2. Or use the Jupyter notebook:"
echo "   jupyter notebook tutorial.ipynb"
echo ""
echo "3. Check the README.md for more examples and documentation"
echo ""
echo "4. Don't forget to delete the endpoint when done to avoid charges:"
echo "   aws sagemaker delete-endpoint --endpoint-name YOUR_ENDPOINT_NAME"
echo ""
echo "========================================================"
