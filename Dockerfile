# Use NVIDIA CUDA runtime with cuDNN for GPU support
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PROJECT_ROOT=/workspace
ENV HYDRA_FULL_ERROR=1

# Create OVHcloud user with UID 42420 as required by OVH AI Training
RUN groupadd -g 42420 ovhcloud && \
    useradd -u 42420 -g ovhcloud -m -s /bin/bash ovhcloud

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
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv awscli

# Set working directory and give ownership to ovhcloud user
WORKDIR /workspace
RUN chown -R ovhcloud:ovhcloud /workspace

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install remaining dependencies from pyproject.toml (without PyTorch)
RUN uv pip install --system -r pyproject.toml && \
    uv cache clean && \
    rm -rf /tmp/* /var/tmp/*

# Copy project code
COPY --chown=ovhcloud:ovhcloud . .

# Create directories with proper permissions for ovhcloud user (UID 42420)
RUN mkdir -p /workspace/outputs /workspace/.config /workspace/.cache /workspace/datasets && \
    chown -R ovhcloud:ovhcloud /workspace && \
    chmod -R 755 /workspace

# Switch to ovhcloud user as required by OVH AI Training
USER ovhcloud

# Set matplotlib config directory to a writable location
ENV MPLCONFIGDIR=/tmp/matplotlib

# Expose TensorBoard port
EXPOSE 6006

# Set default command to bash for interactive use
CMD ["/bin/bash"]
