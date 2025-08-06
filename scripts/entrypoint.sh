#!/bin/bash
# Production entrypoint script for HE-Graph-Embeddings
# Handles initialization, health checks, and graceful startup

set -euo pipefail

# Color codes for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" >&1
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}" >&2
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG: $1${NC}" >&1
    fi
}

# Signal handlers for graceful shutdown
cleanup() {
    log_info "Received shutdown signal, cleaning up..."
    
    # Kill background processes
    if [[ -n "${HEALTH_CHECK_PID:-}" ]]; then
        kill $HEALTH_CHECK_PID 2>/dev/null || true
    fi
    
    if [[ -n "${MONITOR_PID:-}" ]]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    
    # Clean up temporary files
    rm -rf /tmp/he_graph_* 2>/dev/null || true
    
    log_info "Cleanup completed"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

# Validate environment
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Check required environment variables
    local required_vars=(
        "PYTHONPATH"
        "HE_GRAPH_REGION"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate region
    local valid_regions=("us-east-1" "us-west-2" "ca-central-1" "eu-west-1" "eu-central-1" "eu-north-1" "ap-northeast-1" "ap-southeast-1" "ap-south-1" "sa-east-1")
    if [[ ! " ${valid_regions[@]} " =~ " ${HE_GRAPH_REGION} " ]]; then
        log_error "Invalid region: ${HE_GRAPH_REGION}. Must be one of: ${valid_regions[*]}"
        exit 1
    fi
    
    # Check GPU availability if required
    if [[ "${REQUIRE_GPU:-true}" == "true" ]]; then
        if ! nvidia-smi >/dev/null 2>&1; then
            log_error "GPU not available but required"
            exit 1
        fi
        
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        log_info "Detected $gpu_count GPU(s)"
    fi
    
    log_info "Environment validation completed"
}

# Initialize directories and permissions
initialize_directories() {
    log_info "Initializing directories..."
    
    local directories=(
        "/app/logs"
        "/app/data"
        "/app/security-reports"
        "/tmp/cuda-cache"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_debug "Created directory: $dir"
        fi
    done
    
    # Set proper permissions
    chmod 755 /app/logs /app/data /app/security-reports
    chmod 700 /tmp/cuda-cache
    
    log_info "Directory initialization completed"
}

# Pre-flight system checks
run_preflight_checks() {
    log_info "Running pre-flight system checks..."
    
    # Check Python environment
    if ! python3 -c "import torch; import he_graph" >/dev/null 2>&1; then
        log_error "Python environment validation failed"
        exit 1
    fi
    
    # Check CUDA if available
    if [[ "${REQUIRE_GPU:-true}" == "true" ]]; then
        if ! python3 -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
            log_error "CUDA not available in Python environment"
            exit 1
        fi
    fi
    
    # Test basic HE operations
    if ! python3 -c "
import sys
sys.path.append('/app/src')
from python.he_graph import CKKSContext
context = CKKSContext(poly_modulus_degree=8192, scale=2**30, gpu_id=0 if __import__('torch').cuda.is_available() else None)
print('HE context initialization successful')
" >/dev/null 2>&1; then
        log_error "HE-Graph initialization test failed"
        exit 1
    fi
    
    # Check external dependencies
    local external_deps=("redis" "prometheus")
    for dep in "${external_deps[@]}"; do
        local host="${dep^^}_HOST"
        local port="${dep^^}_PORT"
        
        # Use default values if not set
        case $dep in
            "redis")
                host="${!host:-redis}"
                port="${!port:-6379}"
                ;;
            "prometheus")
                host="${!host:-prometheus}"
                port="${!port:-9090}"
                ;;
        esac
        
        if [[ "${CHECK_EXTERNAL_DEPS:-true}" == "true" ]]; then
            if ! nc -z "$host" "$port" 2>/dev/null; then
                log_warn "$dep not available at $host:$port - continuing anyway"
            else
                log_info "$dep connection verified at $host:$port"
            fi
        fi
    done
    
    log_info "Pre-flight checks completed successfully"
}

# Load configuration
load_configuration() {
    log_info "Loading configuration..."
    
    local config_file="/app/config/production.yaml"
    if [[ -f "$config_file" ]]; then
        log_info "Using configuration file: $config_file"
        export HE_GRAPH_CONFIG_FILE="$config_file"
    else
        log_warn "Configuration file not found, using environment variables"
    fi
    
    # Set default values for optional configuration
    export HE_GRAPH_LOG_LEVEL="${HE_GRAPH_LOG_LEVEL:-INFO}"
    export HE_GRAPH_ENABLE_MONITORING="${HE_GRAPH_ENABLE_MONITORING:-true}"
    export HE_GRAPH_ENABLE_CACHING="${HE_GRAPH_ENABLE_CACHING:-true}"
    export HE_GRAPH_MAX_WORKERS="${HE_GRAPH_MAX_WORKERS:-4}"
    export HE_GRAPH_COMPLIANCE_FRAMEWORKS="${HE_GRAPH_COMPLIANCE_FRAMEWORKS:-GDPR,CCPA,SOC2}"
    
    log_info "Configuration loaded successfully"
}

# Start background monitoring
start_monitoring() {
    if [[ "${HE_GRAPH_ENABLE_MONITORING:-true}" == "true" ]]; then
        log_info "Starting background monitoring..."
        
        # Start resource monitoring
        python3 -c "
import sys, time, psutil, os
sys.path.append('/app/src')
from utils.monitoring import HealthCheckManager

health_manager = HealthCheckManager()
while True:
    try:
        health_status = health_manager.get_comprehensive_health()
        print(f'Health Status: {health_status[\"status\"]}')
        time.sleep(30)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f'Monitoring error: {e}')
        time.sleep(10)
" &
        MONITOR_PID=$!
        log_debug "Monitoring process started with PID: $MONITOR_PID"
    fi
}

# Warm up caches and models
warmup() {
    log_info "Performing system warmup..."
    
    # Pre-compile CUDA kernels
    if [[ "${REQUIRE_GPU:-true}" == "true" ]]; then
        log_info "Pre-compiling CUDA kernels..."
        python3 -c "
import sys
sys.path.append('/app/src')
import torch
from python.he_graph import CKKSContext

if torch.cuda.is_available():
    context = CKKSContext(poly_modulus_degree=8192, scale=2**30, gpu_id=0)
    # Perform dummy operations to compile kernels
    dummy_data = torch.randn(100, 64, device='cuda')
    encrypted = context.encrypt(dummy_data)
    result = context.add(encrypted, encrypted)
    print('CUDA kernels pre-compiled successfully')
" || log_warn "CUDA kernel pre-compilation failed"
    fi
    
    # Initialize caches
    if [[ "${HE_GRAPH_ENABLE_CACHING:-true}" == "true" ]]; then
        log_info "Initializing caches..."
        python3 -c "
import sys
sys.path.append('/app/src')
from utils.caching import CacheManager

cache_manager = CacheManager()
cache_manager.initialize()
print('Caches initialized successfully')
" || log_warn "Cache initialization failed"
    fi
    
    log_info "System warmup completed"
}

# Health check endpoint
start_health_check() {
    log_info "Starting health check service..."
    
    python3 /app/scripts/healthcheck.py --daemon &
    HEALTH_CHECK_PID=$!
    log_debug "Health check service started with PID: $HEALTH_CHECK_PID"
}

# Main execution
main() {
    log_info "Starting HE-Graph-Embeddings production service..."
    log_info "Version: 2.0.0"
    log_info "Environment: ${ENV:-development}"
    log_info "Region: ${HE_GRAPH_REGION:-unknown}"
    
    # Initialization sequence
    validate_environment
    initialize_directories
    load_configuration
    run_preflight_checks
    warmup
    start_monitoring
    start_health_check
    
    # Log final configuration
    log_info "Configuration summary:"
    log_info "  - Region: ${HE_GRAPH_REGION}"
    log_info "  - Compliance: ${HE_GRAPH_COMPLIANCE_FRAMEWORKS}"
    log_info "  - Workers: ${HE_GRAPH_MAX_WORKERS}"
    log_info "  - Monitoring: ${HE_GRAPH_ENABLE_MONITORING}"
    log_info "  - Caching: ${HE_GRAPH_ENABLE_CACHING}"
    log_info "  - Log Level: ${HE_GRAPH_LOG_LEVEL}"
    
    log_info "Initialization completed successfully"
    log_info "Starting main application: $*"
    
    # Execute the main command
    exec "$@"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi