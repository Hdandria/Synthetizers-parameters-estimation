# Multi-stage Dockerfile for GPU-optimized training environment
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    build-essential \
    pkg-config \
    libsndfile1-dev \
    libsndfile1 \
    ffmpeg \
    libasound2-dev \
    alsa-utils \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    wget \
    curl \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install pip and uv for faster package management
RUN python -m pip install --upgrade pip
RUN pip install uv

# Create non-root user
RUN useradd -m -s /bin/bash -u 1000 trainer && \
    usermod -aG audio trainer

# Set working directory
WORKDIR /workspace

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system -e .

# Copy source code and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY presets/ ./presets/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY surge_params.csv ./
COPY Makefile ./

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/logs /workspace/vsts /workspace/checkpoints && \
    chown -R trainer:trainer /workspace

# Switch to non-root user
USER trainer

# Set default command
CMD ["/bin/bash"]

