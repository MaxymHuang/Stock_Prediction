# Use the official NVIDIA CUDA 11.8 runtime image as the base
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set a working directory
WORKDIR /app

# Install Python 3 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as the default python command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy the requirements file and install dependencies (if any)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run your Python script
CMD ["bash"]
