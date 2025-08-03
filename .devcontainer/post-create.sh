#!/bin/bash

set -e

echo "Setting up HE-Graph-Embeddings development environment..."

# Update package lists
apt-get update

# Install system dependencies
apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    cppcheck \
    doxygen \
    graphviz \
    pkg-config \
    libssl-dev \
    libboost-all-dev \
    libeigen3-dev \
    libgmp-dev \
    libntl-dev \
    postgresql-client \
    redis-tools \
    jq \
    htop \
    nvtop

# Install Python development dependencies
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]" || pip install \
    torch \
    numpy \
    scipy \
    networkx \
    pandas \
    matplotlib \
    seaborn \
    jupyterlab \
    pytest \
    pytest-cov \
    pytest-benchmark \
    black \
    isort \
    ruff \
    mypy \
    pylint \
    sphinx \
    sphinx-rtd-theme \
    pybind11 \
    tqdm \
    wandb \
    tensorboard

# Install pre-commit hooks
pip install pre-commit
pre-commit install || true

# Setup CUDA environment
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

# Create build directory
mkdir -p build

# Configure ccache
ccache --max-size=10G
echo "export PATH=/usr/lib/ccache:\$PATH" >> ~/.bashrc

# Set up git config
git config --global --add safe.directory /workspace

echo "Development environment setup complete!"
echo "To build the project, run:"
echo "  cd build && cmake .. && make -j\$(nproc)"