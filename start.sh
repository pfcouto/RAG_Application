#!/bin/bash

echo "Building the Docker image..."
if docker build -t ai-test .; then
    echo "Docker image built successfully."
else
    echo "Docker image build failed. Exiting."
    exit 1
fi

echo "Running the Docker container..."
if docker run -d -p 8000:8000 --name ai-test-container ai-test; then
    echo "Docker container started successfully."
else
    echo "Failed to start the Docker container. Exiting."
    exit 1
fi

# Wait for the container to be ready
echo "Waiting for the container to be ready..."
until curl -s http://localhost:8000/ask > /dev/null; do
    sleep 2
    echo "Still waiting for the container to be ready..."
done

echo "Container is running and ready to receive requests at http://localhost:8000"
