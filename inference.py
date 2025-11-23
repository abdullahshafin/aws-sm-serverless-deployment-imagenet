"""
SageMaker inference script for ImageNet classification using PyTorch.

This script handles model loading and inference for serverless deployment.
It accepts presigned URLs as input and returns top-5 predictions.
"""

import json
import os
import sys
import logging
import requests
from io import BytesIO
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ImageNet class labels URL
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

# Global variables for model and preprocessing
model = None
transform = None
class_labels = None


def model_fn(model_dir):
    """
    Load the PyTorch model from the model directory.
    
    Args:
        model_dir (str): Path to the model artifacts
        
    Returns:
        torch.nn.Module: Loaded PyTorch model
    """
    global model, transform, class_labels
    
    logger.info("Loading model...")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model architecture (default to ResNet50)
    # For tutorial purposes, we use pretrained ImageNet weights
    model_name = os.environ.get("MODEL_NAME", "resnet50")
    
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        logger.warning(f"Unknown model name {model_name}, defaulting to ResNet50")
        model = models.resnet50(pretrained=True)
    
    model = model.to(device)
    model.eval()
    
    # Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load ImageNet class labels
    try:
        response = requests.get(IMAGENET_LABELS_URL, timeout=10)
        class_labels = response.json()
        logger.info(f"Loaded {len(class_labels)} ImageNet class labels")
    except Exception as e:
        logger.error(f"Failed to load class labels: {e}")
        # Fallback to indices if labels can't be loaded
        class_labels = [f"class_{i}" for i in range(1000)]
    
    logger.info(f"Model {model_name} loaded successfully")
    return model


def input_fn(request_body, content_type="application/json"):
    """
    Deserialize and prepare the input data.
    
    Args:
        request_body: The request payload
        content_type: The content type of the request
        
    Returns:
        dict: Parsed input containing presigned URL or image data
    """
    logger.info(f"Processing input with content type: {content_type}")
    
    if content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Perform prediction on the input data.
    
    Args:
        input_data: Dictionary containing 'url' key with presigned URL
        model: The loaded PyTorch model
        
    Returns:
        dict: Top-5 predictions with class names and probabilities
    """
    logger.info("Starting prediction...")
    
    # Get presigned URL from input
    presigned_url = input_data.get("url")
    if not presigned_url:
        raise ValueError("Input must contain 'url' key with presigned URL")
    
    # Download image from presigned URL
    logger.info(f"Downloading image from URL...")
    try:
        response = requests.get(presigned_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        logger.info(f"Image downloaded successfully, size: {image.size}")
    except Exception as e:
        logger.error(f"Failed to download or open image: {e}")
        raise ValueError(f"Failed to download image from URL: {str(e)}")
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)
    
    # Move to appropriate device
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Get top-5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    # Prepare results
    predictions = []
    for i in range(5):
        predictions.append({
            "class": class_labels[top5_indices[i].item()],
            "probability": float(top5_prob[i].item())
        })
    
    logger.info("Prediction completed successfully")
    return {"predictions": predictions}


def output_fn(prediction, accept="application/json"):
    """
    Serialize the prediction output.
    
    Args:
        prediction: The prediction results
        accept: The desired response content type
        
    Returns:
        tuple: (serialized_output, content_type)
    """
    logger.info(f"Formatting output with accept type: {accept}")
    
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
