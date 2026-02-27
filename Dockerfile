# Base image with Python 3.9 and Linux
ARG BASE_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG GPU_SERIES=40
ARG PYTORCH_VERSION=2.4.1
ARG TORCHVISION_VERSION=0.19.1
ARG PYTORCH_CUDA_VERSION=12.1
ARG INSTALL_OPTIONAL=0
ARG MIMICGEN_REPO=https://github.com/NVlabs/mimicgen_environments.git
ARG MIMICGEN_REF=main

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    MUJOCO_GL=osmesa

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    cmake \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libglfw3-dev \
    patchelf && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Accept Anaconda Terms of Service for non-interactive Docker builds
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate robomimic conda environment with Python 3.9
RUN /opt/conda/bin/conda create -n robomimic_venv python=3.9 -y

# Install PyTorch and torchvision
# Supported GPU_SERIES:
# 40 -> conda: pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1
# 50 -> pip  : torch torchvision torchaudio from cu128 index
RUN if [ "${GPU_SERIES}" = "40" ]; then \
        /opt/conda/bin/conda run -n robomimic_venv conda install -y \
            pytorch==${PYTORCH_VERSION} \
            torchvision==${TORCHVISION_VERSION} \
            pytorch-cuda=${PYTORCH_CUDA_VERSION} \
            -c pytorch -c nvidia; \
    elif [ "${GPU_SERIES}" = "50" ]; then \
        /opt/conda/bin/conda run -n robomimic_venv pip install --no-cache-dir \
            torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu128; \
    else \
        echo "Unsupported GPU_SERIES=${GPU_SERIES} (use 40 or 50)"; \
        exit 1; \
    fi


# Install robosuite at a pinned commit
WORKDIR /opt
RUN git clone https://github.com/ARISE-Initiative/robosuite.git && \
    cd robosuite && \
    git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa && \
    /opt/conda/bin/conda run -n robomimic_venv pip install --no-cache-dir --retries 10 --timeout 120 --prefer-binary --only-binary=mujoco -r requirements.txt && \
    /opt/conda/bin/conda run -n robomimic_venv pip install --no-cache-dir -e . --no-deps

# Optional: install MimicGen environments for additional task envs
RUN if [ "${INSTALL_OPTIONAL}" = "1" ]; then \
        git clone "${MIMICGEN_REPO}" /opt/mimicgen_environments && \
        cd /opt/mimicgen_environments && \
        git checkout "${MIMICGEN_REF}" && \
        /opt/conda/bin/conda run -n robomimic_venv pip install --no-cache-dir --retries 10 --timeout 120 -e .; \
    else \
        echo "Skipping MimicGen installation (INSTALL_OPTIONAL=${INSTALL_OPTIONAL})"; \
    fi

# Install local robomimic package from this build context
WORKDIR /workspace
COPY . /workspace
RUN /opt/conda/bin/conda run -n robomimic_venv pip install -e .

# # Optional: Install robomimic documentation dependencies
# WORKDIR /opt/robomimic
# RUN /opt/conda/bin/conda run -n robomimic_venv pip install -r requirements-docs.txt

# Activate Conda environment and start bash when container starts
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && bash"]
