# Example Input/Output for SageMaker Serverless ImageNet Endpoint

## Input Format

The endpoint accepts JSON payloads with a presigned URL:

```json
{
  "url": "https://your-s3-bucket.s3.amazonaws.com/image.jpg?AWSAccessKeyId=...&Signature=...&Expires=..."
}
```

## Output Format

The endpoint returns JSON with top-5 predictions:

```json
{
  "predictions": [
    {
      "class": "golden retriever",
      "probability": 0.8523
    },
    {
      "class": "Labrador retriever",
      "probability": 0.0845
    },
    {
      "class": "cocker spaniel",
      "probability": 0.0231
    },
    {
      "class": "Irish setter",
      "probability": 0.0189
    },
    {
      "class": "English setter",
      "probability": 0.0076
    }
  ]
}
```

## Example Using cURL

```bash
# Generate presigned URL for your S3 image
PRESIGNED_URL=$(aws s3 presign s3://your-bucket/your-image.jpg)

# Invoke endpoint
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name your-endpoint-name \
    --content-type application/json \
    --body "{\"url\": \"$PRESIGNED_URL\"}" \
    response.json

# View results
cat response.json | jq
```

## Example Using Python

```python
import boto3
import json

# Create SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Prepare input
payload = {
    "url": "https://your-presigned-url"
}

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps(payload)
)

# Parse results
result = json.loads(response['Body'].read())
predictions = result['predictions']

# Display top prediction
top_prediction = predictions[0]
print(f"Class: {top_prediction['class']}")
print(f"Confidence: {top_prediction['probability'] * 100:.2f}%")
```

## Example Using JavaScript/Node.js

```javascript
const AWS = require('aws-sdk');

const sagemakerRuntime = new AWS.SageMakerRuntime({
    region: 'us-east-1'
});

const payload = {
    url: 'https://your-presigned-url'
};

const params = {
    EndpointName: 'your-endpoint-name',
    ContentType: 'application/json',
    Body: JSON.stringify(payload)
};

sagemakerRuntime.invokeEndpoint(params, (err, data) => {
    if (err) {
        console.error(err);
    } else {
        const result = JSON.parse(data.Body.toString());
        console.log('Top prediction:', result.predictions[0]);
    }
});
```

## Generating S3 Presigned URLs

### Using AWS CLI
```bash
aws s3 presign s3://your-bucket/your-image.jpg --expires-in 3600
```

### Using Python Boto3
```python
import boto3

s3 = boto3.client('s3')
url = s3.generate_presigned_url(
    'get_object',
    Params={
        'Bucket': 'your-bucket',
        'Key': 'your-image.jpg'
    },
    ExpiresIn=3600  # 1 hour
)
print(url)
```

### Using AWS SDK for JavaScript
```javascript
const AWS = require('aws-sdk');
const s3 = new AWS.S3();

const params = {
    Bucket: 'your-bucket',
    Key: 'your-image.jpg',
    Expires: 3600  // 1 hour
};

const url = s3.getSignedUrl('getObject', params);
console.log(url);
```
