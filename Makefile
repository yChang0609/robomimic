IMAGE_NAME ?= robomimic
GPU_SERIES ?= 40
INSTALL_OPTIONAL ?= 0

# Tag is auto-generated from the two control vars requested by user.
IMAGE_TAG = gpu$(GPU_SERIES)-opt$(INSTALL_OPTIONAL)
IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)
RUN_IMAGE ?= $(IMAGE)
CONTAINER_NAME ?= robomimic
DATASETS_DIR ?= $(HOME)/datasets

DOCKERFILE ?= Dockerfile
CONTEXT ?= .

BASE_IMAGE ?= nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04
PYTORCH_VERSION ?= 2.4.1
TORCHVISION_VERSION ?= 0.19.1
PYTORCH_CUDA_VERSION ?= 12.1

MIMICGEN_REPO ?= https://github.com/NVlabs/mimicgen_environments.git
MIMICGEN_REF ?= main

BUILD_ARGS = \
	--build-arg BASE_IMAGE=$(BASE_IMAGE) \
	--build-arg GPU_SERIES=$(GPU_SERIES) \
	--build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) \
	--build-arg TORCHVISION_VERSION=$(TORCHVISION_VERSION) \
	--build-arg PYTORCH_CUDA_VERSION=$(PYTORCH_CUDA_VERSION) \
	--build-arg INSTALL_OPTIONAL=$(INSTALL_OPTIONAL) \
	--build-arg MIMICGEN_REPO=$(MIMICGEN_REPO) \
	--build-arg MIMICGEN_REF=$(MIMICGEN_REF)

.PHONY: help validate print-config build run run-tail exec

help:
	@echo "Usage:"
	@echo "  make build GPU_SERIES=40 INSTALL_OPTIONAL=0"
	@echo "  make build GPU_SERIES=50 INSTALL_OPTIONAL=1"
	@echo "  make run GPU_SERIES=50 INSTALL_OPTIONAL=1 DATASETS_DIR=\$$HOME/datasets"
	@echo "  make run-tail GPU_SERIES=50 INSTALL_OPTIONAL=1 DATASETS_DIR=\$$HOME/datasets"
	@echo ""
	@echo "Vars:"
	@echo "  GPU_SERIES=40|50"
	@echo "  INSTALL_OPTIONAL=0|1   # 1 installs mimicgen optional envs"
	@echo "  DATASETS_DIR=<host path>   # mounted to /datasets in container"
	@echo "  CONTAINER_NAME=<name>"
	@echo "  RUN_IMAGE=<image:tag>      # default uses built IMAGE"
	@echo ""
	@echo "Auto tag format:"
	@echo "  $(IMAGE_NAME):gpu<GPU_SERIES>-opt<INSTALL_OPTIONAL>"

validate:
	@if [ "$(GPU_SERIES)" != "40" ] && [ "$(GPU_SERIES)" != "50" ]; then \
		echo "ERROR: GPU_SERIES must be 40 or 50 (got $(GPU_SERIES))"; \
		exit 1; \
	fi
	@if [ "$(INSTALL_OPTIONAL)" != "0" ] && [ "$(INSTALL_OPTIONAL)" != "1" ]; then \
		echo "ERROR: INSTALL_OPTIONAL must be 0 or 1 (got $(INSTALL_OPTIONAL))"; \
		exit 1; \
	fi

print-config:
	@echo "IMAGE=$(IMAGE)"
	@echo "RUN_IMAGE=$(RUN_IMAGE)"
	@echo "CONTAINER_NAME=$(CONTAINER_NAME)"
	@echo "DATASETS_DIR=$(DATASETS_DIR)"
	@echo "DOCKERFILE=$(DOCKERFILE)"
	@echo "BASE_IMAGE=$(BASE_IMAGE)"
	@echo "GPU_SERIES=$(GPU_SERIES)"
	@echo "INSTALL_OPTIONAL=$(INSTALL_OPTIONAL)"
	@echo "PYTORCH_VERSION=$(PYTORCH_VERSION)"
	@echo "TORCHVISION_VERSION=$(TORCHVISION_VERSION)"
	@echo "PYTORCH_CUDA_VERSION=$(PYTORCH_CUDA_VERSION)"
	@echo "MIMICGEN_REPO=$(MIMICGEN_REPO)"
	@echo "MIMICGEN_REF=$(MIMICGEN_REF)"

build: validate print-config
	docker build -f $(DOCKERFILE) $(BUILD_ARGS) -t $(IMAGE) $(CONTEXT)

run: validate print-config
	docker run -it --gpus all --name $(CONTAINER_NAME) \
		-v "$(CURDIR)":/workspace \
		-v "$(DATASETS_DIR)":/datasets \
		--ipc=host \
		$(RUN_IMAGE)

run-tail: validate print-config
	docker run -it --gpus all --name $(CONTAINER_NAME) \
		-v "$(CURDIR)":/workspace \
		-v "$(DATASETS_DIR)":/datasets \
		--ipc=host \
		$(RUN_IMAGE) tail -f /dev/null

exec: validate print-config
	docker exec -it $(CONTAINER_NAME) /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && bash"
