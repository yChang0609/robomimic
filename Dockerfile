# Base image with Python 3.9 and Linux
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

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

# Install PyTorch and torchvision (fallback to pip on transient conda download failures)
RUN /opt/conda/bin/conda run -n robomimic_venv conda install -y pytorch==2.4.1 torchvision==0.19.1 cpuonly -c pytorch


# Install robosuite at a pinned commit
WORKDIR /opt
RUN git clone https://github.com/ARISE-Initiative/robosuite.git && \
    cd robosuite && \
    git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa && \
    /opt/conda/bin/conda run -n robomimic_venv pip install --no-cache-dir --retries 10 --timeout 120 --prefer-binary --only-binary=mujoco -r requirements.txt && \
    /opt/conda/bin/conda run -n robomimic_venv pip install --no-cache-dir -e . --no-deps

# Install local robomimic package from this build context
WORKDIR /workspace
COPY . /workspace
RUN /opt/conda/bin/conda run -n robomimic_venv pip install -e .

# # Optional: Install robomimic documentation dependencies
# WORKDIR /opt/robomimic
# RUN /opt/conda/bin/conda run -n robomimic_venv pip install -r requirements-docs.txt

# Activate Conda environment and start bash when container starts
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && bash"]
