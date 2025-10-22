# Use NVIDIA CUDA runtime with cuDNN for GPU support
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PROJECT_ROOT=/workspace
ENV HYDRA_FULL_ERROR=1

# Create non-root user
RUN groupadd -g 1000 trainer && \
    useradd -u 1000 -g trainer -m -s /bin/bash trainer

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    portaudio19-dev \
    awscli \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Set working directory and give ownership to trainer
WORKDIR /workspace
RUN chown -R trainer:trainer /workspace

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install remaining dependencies from pyproject.toml (without PyTorch)
RUN uv pip install --system -r pyproject.toml && \
    uv cache clean && \
    rm -rf /tmp/* /var/tmp/*

# Copy project code
COPY --chown=trainer:trainer . .

# Create directories with proper permissions
RUN mkdir -p /workspace/outputs /workspace/.config /workspace/.cache /workspace/datasets && \
    chmod -R 777 /workspace

# Run as root for simplicity in AI Training environment
USER root

# Set matplotlib config directory to a writable location
ENV MPLCONFIGDIR=/tmp/matplotlib

# Expose TensorBoard port
EXPOSE 6006

# Set default command to bash for interactive use
CMD ["/bin/bash"]
