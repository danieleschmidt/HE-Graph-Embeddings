#!/bin/bash

# Quantum Task Planner API Startup Script
# Production-grade startup with quantum optimizations and monitoring

set -euo pipefail

# Configuration
QUANTUM_OPTIMIZATION_LEVEL=${QUANTUM_OPTIMIZATION_LEVEL:-3}
QUANTUM_GPU_ENABLED=${QUANTUM_GPU_ENABLED:-true}
API_HOST=${API_HOST:-0.0.0.0}
API_PORT=${API_PORT:-8000}
WORKERS=${WORKERS:-auto}
LOG_LEVEL=${LOG_LEVEL:-info}
ENVIRONMENT=${ENVIRONMENT:-production}

# Quantum-specific settings
QUANTUM_COHERENCE_TIME=${QUANTUM_COHERENCE_TIME:-300}
QUANTUM_ENTANGLEMENT_DEPTH=${QUANTUM_ENTANGLEMENT_DEPTH:-5}
HE_SECURITY_LEVEL=${HE_SECURITY_LEVEL:-128}

# Logging setup
LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"
STARTUP_LOG="$LOG_DIR/startup.log"

# Logging function
log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] [STARTUP] $1" | tee -a "$STARTUP_LOG"
}

log "Starting Quantum Task Planner API..."
log "Environment: $ENVIRONMENT"
log "Quantum Optimization Level: $QUANTUM_OPTIMIZATION_LEVEL"
log "GPU Acceleration: $QUANTUM_GPU_ENABLED"

# Pre-flight checks
log "Running pre-flight checks..."

# Check Python environment
if ! python3 -c "import sys; print(f'Python {sys.version}')" >> "$STARTUP_LOG" 2>&1; then
    log "ERROR: Python environment check failed"
    exit 1
fi

# Check required packages
REQUIRED_PACKAGES=("torch" "numpy" "fastapi" "uvicorn")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" >> "$STARTUP_LOG" 2>&1; then
        log "ERROR: Required package $package not found"
        exit 1
    fi
done

# GPU checks if enabled
if [ "$QUANTUM_GPU_ENABLED" = "true" ]; then
    log "Checking GPU availability..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        log "Found $GPU_COUNT GPU(s)"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> "$STARTUP_LOG"
        
        # Test CUDA availability in Python
        if python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" >> "$STARTUP_LOG" 2>&1; then
            log "CUDA is available for PyTorch"
        else
            log "WARNING: CUDA not available for PyTorch"
        fi
    else
        log "WARNING: nvidia-smi not found, GPU features may not work"
    fi
fi

# Check quantum libraries
log "Checking quantum computing libraries..."
QUANTUM_LIBS=("qiskit" "cirq" "pennylane")
for lib in "${QUANTUM_LIBS[@]}"; do
    if python3 -c "import $lib; print(f'$lib version: {$lib.__version__}')" >> "$STARTUP_LOG" 2>&1; then
        log "$lib is available"
    else
        log "WARNING: $lib not available, some quantum features may be limited"
    fi
done

# Database connectivity check
log "Checking database connectivity..."
if [ -n "${DATABASE_URL:-}" ]; then
    if python3 -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" >> "$STARTUP_LOG" 2>&1; then
        log "Database connectivity verified"
    else
        log "ERROR: Database connectivity check failed"
        exit 1
    fi
else
    log "No database URL configured, using SQLite"
fi

# Redis connectivity check
if [ -n "${REDIS_URL:-}" ]; then
    log "Checking Redis connectivity..."
    if python3 -c "
import redis
import os
try:
    r = redis.from_url(os.environ['REDIS_URL'])
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
    exit(1)
" >> "$STARTUP_LOG" 2>&1; then
        log "Redis connectivity verified"
    else
        log "WARNING: Redis connectivity check failed, some features may be limited"
    fi
else
    log "No Redis URL configured"
fi

# System resource checks
log "Checking system resources..."

# Memory check
AVAILABLE_MEMORY=$(free -m | awk 'NR==2{printf "%.1f", $7*100/$2 }')
log "Available memory: ${AVAILABLE_MEMORY}%"

if (( $(echo "$AVAILABLE_MEMORY < 20.0" | bc -l) )); then
    log "WARNING: Low available memory (${AVAILABLE_MEMORY}%)"
fi

# CPU check
CPU_CORES=$(nproc)
log "CPU cores: $CPU_CORES"

# Disk space check
DISK_USAGE=$(df -h /app | awk 'NR==2{print $5}' | sed 's/%//')
log "Disk usage: ${DISK_USAGE}%"

if [ "$DISK_USAGE" -gt 90 ]; then
    log "WARNING: High disk usage (${DISK_USAGE}%)"
fi

# Network interface check
if ip route get 8.8.8.8 >/dev/null 2>&1; then
    log "Network connectivity verified"
else
    log "WARNING: Network connectivity issues detected"
fi

# Quantum system initialization
log "Initializing quantum systems..."

# Set quantum environment variables
export QUANTUM_OPTIMIZATION_LEVEL
export QUANTUM_GPU_ENABLED
export QUANTUM_COHERENCE_TIME
export QUANTUM_ENTANGLEMENT_DEPTH
export HE_SECURITY_LEVEL

# Initialize homomorphic encryption context
log "Initializing homomorphic encryption..."
if python3 -c "
from src.python.he_graph import CKKSContext, HEConfig
import os

config = HEConfig(
    poly_modulus_degree=int(os.environ.get('HE_POLY_DEGREE', '32768')),
    security_level=int(os.environ.get('HE_SECURITY_LEVEL', '128'))
)

context = CKKSContext(config)
context.generate_keys()
print('HE context initialized successfully')
" >> "$STARTUP_LOG" 2>&1; then
    log "Homomorphic encryption initialized"
else
    log "WARNING: HE initialization failed, privacy features may be limited"
fi

# Quantum optimization application
log "Applying quantum optimizations (level $QUANTUM_OPTIMIZATION_LEVEL)..."

if [ "$QUANTUM_OPTIMIZATION_LEVEL" -ge 2 ]; then
    # CPU performance optimization
    log "Applying CPU performance optimizations..."
    
    # Set CPU affinity for better performance
    if [ -n "${CPU_AFFINITY:-}" ]; then
        taskset -c "$CPU_AFFINITY" $$ 2>/dev/null || log "WARNING: CPU affinity setting failed"
    fi
    
    # Set process priority
    renice -n -10 $$ 2>/dev/null || log "WARNING: Process priority adjustment failed"
fi

if [ "$QUANTUM_OPTIMIZATION_LEVEL" -ge 3 ]; then
    # Memory optimization
    log "Applying memory optimizations..."
    
    # Set memory allocation behavior
    export MALLOC_ARENA_MAX=4
    export MALLOC_MMAP_THRESHOLD_=1048576
    
    # Enable memory prefetching
    export OMP_NUM_THREADS=$CPU_CORES
    export MKL_NUM_THREADS=$CPU_CORES
    export OPENBLAS_NUM_THREADS=$CPU_CORES
fi

if [ "$QUANTUM_OPTIMIZATION_LEVEL" -ge 4 ]; then
    # Network optimization
    log "Applying network optimizations..."
    
    # Increase network buffer sizes
    ulimit -n 65536 2>/dev/null || log "WARNING: File descriptor limit adjustment failed"
fi

# Determine worker count
if [ "$WORKERS" = "auto" ]; then
    if [ "$CPU_CORES" -le 2 ]; then
        WORKERS=2
    elif [ "$CPU_CORES" -le 4 ]; then
        WORKERS=4
    else
        WORKERS=$((CPU_CORES))
    fi
fi

log "Using $WORKERS workers"

# Create necessary directories
mkdir -p "$LOG_DIR" /app/data /app/tmp
chmod 755 "$LOG_DIR" /app/data /app/tmp

# Set up logging configuration
cat > /app/config/logging.json << EOF
{
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "quantum": {
            "format": "%(asctime)s [%(name)s] %(levelname)s [%(quantum_context)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "standard": {
            "format": "%(asctime)s [%(name)s] %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "quantum",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "quantum",
            "filename": "$LOG_DIR/api.log",
            "maxBytes": 10485760,
            "backupCount": 5
        },
        "quantum_file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "quantum",
            "filename": "$LOG_DIR/quantum.log",
            "maxBytes": 10485760,
            "backupCount": 5
        }
    },
    "loggers": {
        "quantum": {
            "level": "DEBUG",
            "handlers": ["quantum_file", "console"],
            "propagate": false
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": false
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["file"],
            "propagate": false
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
EOF

# Set up monitoring and health checks
log "Setting up monitoring and health checks..."

# Start quantum metrics collector
if [ "${ENABLE_MONITORING:-true}" = "true" ]; then
    python3 -c "
from src.quantum.quantum_resource_manager import QuantumResourceManager
import asyncio
import threading
import time

def start_monitoring():
    try:
        manager = QuantumResourceManager(monitoring_interval=10.0)
        asyncio.run(manager.initialize_quantum_resources())
        print('Quantum monitoring started')
    except Exception as e:
        print(f'Monitoring startup failed: {e}')

monitor_thread = threading.Thread(target=start_monitoring, daemon=True)
monitor_thread.start()
print('Quantum monitoring thread started')
" >> "$STARTUP_LOG" 2>&1 &
    
    log "Quantum monitoring started"
fi

# Export final environment
export PYTHONPATH="/app/src:$PYTHONPATH"

# Performance tuning based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    UVICORN_ARGS=(
        --host "$API_HOST"
        --port "$API_PORT"
        --workers "$WORKERS"
        --worker-class uvicorn.workers.UvicornWorker
        --log-level "$LOG_LEVEL"
        --access-log
        --use-colors
        --loop uvloop
        --http httptools
        --interface asgi3
    )
elif [ "$ENVIRONMENT" = "development" ]; then
    UVICORN_ARGS=(
        --host "$API_HOST"
        --port "$API_PORT"
        --reload
        --log-level debug
        --use-colors
        --loop uvloop
    )
else
    # Default configuration
    UVICORN_ARGS=(
        --host "$API_HOST"
        --port "$API_PORT"
        --workers "$WORKERS"
        --log-level "$LOG_LEVEL"
    )
fi

# Final startup message
log "Starting API server with the following configuration:"
log "  Host: $API_HOST"
log "  Port: $API_PORT"
log "  Workers: $WORKERS"
log "  Log Level: $LOG_LEVEL"
log "  Environment: $ENVIRONMENT"
log "  Quantum Optimization: Level $QUANTUM_OPTIMIZATION_LEVEL"
log "  GPU Acceleration: $QUANTUM_GPU_ENABLED"
log "  HE Security Level: $HE_SECURITY_LEVEL bits"

# Start the application
log "Launching Quantum Task Planner API..."

# Use exec to replace the shell process
exec python3 -m uvicorn src.api.main:app "${UVICORN_ARGS[@]}"