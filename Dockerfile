# Dockerfile for LifeMate (Zalo AI Challenge 2026)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (Python, ffmpeg, audio libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY src/ ./src/
COPY inference.py .

# Create result directory
RUN mkdir -p /app/result

# Set offline mode for Hugging Face (strictly no network at inference)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Entrypoint: Organisers mount private test folder to /data
# Pipeline runs on /data and outputs to /app/result/submission.jsonl
CMD ["python", "inference.py", "--data_dir", "/data", "--split", "private_test", "--output", "result/submission.jsonl"]
