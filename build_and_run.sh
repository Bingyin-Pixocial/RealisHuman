#!/bin/bash

# Build and Run Script for RealisHuman Docker Container

set -e

echo "Building RealisHuman Docker image..."
docker build -t realishuman:latest .

echo "Creating necessary directories..."
mkdir -p data outputs checkpoint pretrained_models

echo "Starting RealisHuman container..."
docker-compose up -d

echo "Container is running!"
echo "To access the container:"
echo "  docker exec -it realishuman bash"
echo ""
echo "To run inference:"
echo "  docker exec -it realishuman conda run -n hamer python inference_stage1.py --config configs/stage1-hand.yaml --output outputs/test --ckpt checkpoint/stage1_hand/checkpoint-stage1-hand.ckpt"
echo ""
echo "To start Jupyter notebook:"
echo "  docker-compose --profile jupyter up jupyter"
echo ""
echo "To stop the container:"
echo "  docker-compose down" 