# Contributing to SageMaker Serverless ImageNet Deployment

Thank you for your interest in contributing! This is a tutorial project aimed at helping users learn SageMaker Serverless deployment.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:
1. Check if the issue already exists
2. Create a new issue with a clear description
3. Include steps to reproduce (for bugs)
4. Include your environment details (Python version, AWS region, etc.)

### Submitting Changes

1. Fork the repository
2. Create a new branch for your feature/fix
3. Make your changes
4. Test your changes thoroughly
5. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Keep code simple and tutorial-focused
- Add comments for complex logic
- Update documentation when changing functionality

### Testing Changes

Before submitting:
1. Run syntax checks: `python -m py_compile *.py`
2. Test deployment (if applicable)
3. Verify documentation updates

### Documentation

- Update README.md for significant changes
- Add examples to EXAMPLES.md if relevant
- Keep documentation clear and beginner-friendly

### Pull Request Guidelines

- Provide a clear description of changes
- Reference related issues
- Keep changes focused and minimal
- Ensure backwards compatibility when possible

## Project Structure

```
.
├── inference.py       # SageMaker inference handler
├── deploy.py         # Deployment script
├── test_endpoint.py  # Testing utility
├── requirements.txt  # Python dependencies
├── README.md        # Main documentation
├── EXAMPLES.md      # Usage examples
├── tutorial.ipynb   # Jupyter notebook tutorial
└── quickstart.sh    # Quick start script
```

## Adding New Models

To add support for a new model architecture:

1. Update `inference.py`:
   - Add model loading logic in `model_fn()`
   - Use appropriate weights enum from torchvision

2. Update `deploy.py`:
   - Add model name to choices in argument parser

3. Update documentation:
   - Add model to README.md model list
   - Update EXAMPLES.md if needed

Example:
```python
elif model_name == "mobilenet_v3":
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
```

## Questions?

Feel free to open an issue for questions or discussions.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
