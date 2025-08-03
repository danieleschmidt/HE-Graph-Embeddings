# Multi-stage Dockerfile for HE-Graph-Embeddings
FROM nvidia/cuda:12.2-devel-ubuntu22.04 as cuda-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    software-properties-common \
    pkg-config \
    libomp-dev \
    libssl-dev \
    libffi-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Build stage
FROM cuda-base as builder

WORKDIR /app

# Copy build files
COPY CMakeLists.txt ./
COPY src/ckks/ ./src/ckks/
COPY src/cuda/ ./src/cuda/
COPY setup.py ./

# Install Python build dependencies
RUN pip install --no-cache-dir \
    cmake \
    ninja \
    pybind11 \
    numpy

# Build CUDA kernels and C++ components
RUN mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86" \
    -DBUILD_TESTS=OFF \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DUSE_CUDA=ON && \
    make -j$(nproc)

# Install Python package
RUN pip install -e .

# Production stage
FROM cuda-base as production

WORKDIR /app

# Copy built artifacts from builder
COPY --from=builder /app/build/ ./build/
COPY --from=builder /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/

# Copy source code
COPY src/ ./src/
COPY requirements.txt ./
COPY requirements-api.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-api.txt

# Create non-root user
RUN groupadd -r hegraph && \
    useradd -r -g hegraph -s /bin/bash -c "HE-Graph user" hegraph && \
    mkdir -p /app/data /app/logs && \
    chown -R hegraph:hegraph /app

# Switch to non-root user
USER hegraph

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM production as development

USER root

# Install development dependencies
COPY requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements-test.txt

# Install debugging tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    strace \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER hegraph

# Development command
CMD ["python", "-m", "uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Testing stage
FROM development as testing

USER root

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini ./
COPY .coveragerc ./

# Run tests as part of build
RUN python -m pytest tests/ --cov=src --cov-report=term --cov-report=html

USER hegraph

# Security scanning stage
FROM production as security

USER root

# Install security scanning tools
RUN pip install --no-cache-dir \
    bandit \
    safety \
    semgrep

# Run security scans
RUN bandit -r src/ -f json -o security-report.json || true
RUN safety check --json --output security-deps.json || true

USER hegraph

# Minimal runtime
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 as minimal

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy only necessary files from production stage
COPY --from=production /app/src/ ./src/
COPY --from=production /app/build/ ./build/
COPY --from=production /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/

# Create non-root user
RUN groupadd -r hegraph && \
    useradd -r -g hegraph -s /bin/bash -c "HE-Graph user" hegraph && \
    chown -R hegraph:hegraph /app

USER hegraph

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]