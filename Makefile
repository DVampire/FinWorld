SHELL=/usr/bin/env bash

# Conda environment name and Python version
ENV_NAME = finworld
PYTHON_VERSION = 3.11

# Default goal
.DEFAULT_GOAL := help

# 🛠️ Remove Conda environment
.PHONY: clean
clean:
	conda remove -y --name $(ENV_NAME) --all

# 🛠️ Install dependencies using Poetry
.PHONY: install

install-base:
	@echo "Installing base dependencies"
	pip install poetry

	# install torch 2.6.0 cuda 12.4
	pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
	pip install vllm==0.8.5
	pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

	# install markitdown
	pip install 'markitdown[all]'

	# install dependencies
	pip install -r requirements.txt

install-browser:
	@echo "Installing browser dependencies"
	pip install "browser-use[memory]"==0.1.48

install-playwright:
	# install playwright
	pip install playwright
	playwright install chromium --with-deps --no-shell

install-verl:
	@echo "Installing verl dependencies"
	cd libs
	git clone git@github.com:volcengine/verl.git
	pip install -e .
	cd ../..

# 🛠️ Update dependencies using Poetry
.PHONY: update
update:
	poetry update

# 🛠️ Show available Makefile commands
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  make create      - Create Conda environment and install Poetry"
	@echo "  make activate    - Show activation command"
	@echo "  make clean       - Remove Conda environment"
	@echo "  make install     - Install dependencies using Poetry"
	@echo "  make update      - Update dependencies using Poetry"
