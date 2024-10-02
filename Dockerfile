# Use the official NVIDIA CUDA 11.8 runtime image as the base
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set a working directory
WORKDIR /app

# Install Python 3 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    make \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget -O ta-lib-0.4.0-src.tar.gz "https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download"
RUN tar -xvzf ta-lib-0.4.0-src.tar.gz

WORKDIR /tmp/ta-lib/
RUN ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz

# Set Python 3 as the default python command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy the requirements file and install dependencies (if any)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyter
RUN pip install jupytext
RUN rm -rf /tmp/ta-lib
# Copy the application code
COPY src/ ./src

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Command to run Jupyter Notebook without a token
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

