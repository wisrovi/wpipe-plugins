#!/bin/bash
# Script to run unit tests inside a Docker container
# Enforced by user preferences

IMAGE_NAME="python:3.10-slim"

echo "Building and running tests in Docker..."

docker run --rm -v $(pwd):/app -w /app $IMAGE_NAME bash -c "
    pip install -e .[dev]
    pytest tests/ --cov=src/wpipe_plugins --cov-report=term-missing
"
