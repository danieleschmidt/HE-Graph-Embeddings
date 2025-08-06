#!/bin/bash

# Quantum Task Planner - Health Check Script
# Comprehensive health monitoring for quantum-optimized API

set -euo pipefail

# Configuration
API_HOST=${API_HOST:-localhost}
API_PORT=${API_PORT:-8000}
TIMEOUT=${HEALTH_CHECK_TIMEOUT:-10}
QUANTUM_CHECK=${QUANTUM_HEALTH_CHECK:-true}

# Health check endpoints
HEALTH_ENDPOINT="http://${API_HOST}:${API_PORT}/health"
READY_ENDPOINT="http://${API_HOST}:${API_PORT}/ready"
QUANTUM_ENDPOINT="http://${API_HOST}:${API_PORT}/quantum/status"
METRICS_ENDPOINT="http://${API_HOST}:${API_PORT}/metrics"

# Exit codes
EXIT_SUCCESS=0
EXIT_FAILURE=1
EXIT_WARNING=2

# Logging
log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] [HEALTH] $1" >&2
}

# Check if service is responsive
check_http_endpoint() {
    local endpoint=$1
    local description=$2
    local required=${3:-true}
    
    log "Checking $description at $endpoint"
    
    if response=$(curl -s --max-time "$TIMEOUT" --fail "$endpoint" 2>/dev/null); then
        log "✓ $description check passed"
        return 0
    else
        if [ "$required" = "true" ]; then
            log "✗ $description check failed (required)"
            return 1
        else
            log "⚠ $description check failed (optional)"
            return 2
        fi
    fi
}

# Check basic health endpoint
check_basic_health() {
    log "Running basic health checks..."
    
    if ! check_http_endpoint "$HEALTH_ENDPOINT" "Basic health"; then
        log "Basic health check failed - service may be starting or unhealthy"
        return $EXIT_FAILURE
    fi
    
    # Parse health response for detailed status
    if health_data=$(curl -s --max-time "$TIMEOUT" "$HEALTH_ENDPOINT" 2>/dev/null); then
        if echo "$health_data" | jq -e '.status == "healthy"' >/dev/null 2>&1; then
            log "Service reports healthy status"
        else
            log "Service reports unhealthy status: $(echo "$health_data" | jq -r '.status // "unknown"')"
            return $EXIT_FAILURE
        fi
    fi
    
    return $EXIT_SUCCESS
}

# Check readiness endpoint
check_readiness() {
    log "Checking service readiness..."
    
    if ! check_http_endpoint "$READY_ENDPOINT" "Readiness"; then
        log "Readiness check failed - service not ready to accept traffic"
        return $EXIT_FAILURE
    fi
    
    # Parse readiness response
    if ready_data=$(curl -s --max-time "$TIMEOUT" "$READY_ENDPOINT" 2>/dev/null); then
        if echo "$ready_data" | jq -e '.ready == true' >/dev/null 2>&1; then
            log "Service is ready to accept traffic"
        else
            log "Service is not ready: $(echo "$ready_data" | jq -r '.reason // "unknown"')"
            return $EXIT_FAILURE
        fi
    fi
    
    return $EXIT_SUCCESS
}

# Check quantum systems
check_quantum_systems() {
    if [ "$QUANTUM_CHECK" != "true" ]; then
        log "Quantum health checks disabled, skipping..."
        return $EXIT_SUCCESS
    fi
    
    log "Checking quantum systems health..."
    
    if ! check_http_endpoint "$QUANTUM_ENDPOINT" "Quantum systems" false; then
        log "Quantum systems check failed, but continuing..."
        return $EXIT_WARNING
    fi
    
    # Parse quantum status
    if quantum_data=$(curl -s --max-time "$TIMEOUT" "$QUANTUM_ENDPOINT" 2>/dev/null); then
        # Check quantum coherence
        coherence=$(echo "$quantum_data" | jq -r '.quantum_coherence_remaining // 0')
        if (( $(echo "$coherence < 0.1" | bc -l 2>/dev/null || echo 0) )); then
            log "⚠ Low quantum coherence: ${coherence}"
        else
            log "✓ Quantum coherence OK: ${coherence}"
        fi
        
        # Check entanglement efficiency
        entanglement=$(echo "$quantum_data" | jq -r '.entanglement_efficiency // 0')
        if (( $(echo "$entanglement < 0.5" | bc -l 2>/dev/null || echo 0) )); then
            log "⚠ Low entanglement efficiency: ${entanglement}"
        else
            log "✓ Entanglement efficiency OK: ${entanglement}"
        fi
        
        # Check quantum speedup
        speedup=$(echo "$quantum_data" | jq -r '.quantum_speedup_average // 1')
        if (( $(echo "$speedup < 1.2" | bc -l 2>/dev/null || echo 0) )); then
            log "⚠ Low quantum speedup: ${speedup}x"
        else
            log "✓ Quantum speedup OK: ${speedup}x"
        fi
    fi
    
    return $EXIT_SUCCESS
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # Memory check
    if command -v free >/dev/null 2>&1; then
        memory_usage=$(free | grep Mem: | awk '{printf "%.1f", $3*100/$2}')
        log "Memory usage: ${memory_usage}%"
        
        if (( $(echo "$memory_usage > 90.0" | bc -l 2>/dev/null || echo 0) )); then
            log "⚠ High memory usage: ${memory_usage}%"
        fi
    fi
    
    # CPU check
    if command -v top >/dev/null 2>&1; then
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
        log "CPU usage: ${cpu_usage}%"
        
        if (( $(echo "$cpu_usage > 95.0" | bc -l 2>/dev/null || echo 0) )); then
            log "⚠ High CPU usage: ${cpu_usage}%"
        fi
    fi
    
    # Disk space check
    if command -v df >/dev/null 2>&1; then
        disk_usage=$(df -h /app 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//' || echo "0")
        log "Disk usage: ${disk_usage}%"
        
        if [ "$disk_usage" -gt 85 ] 2>/dev/null; then
            log "⚠ High disk usage: ${disk_usage}%"
        fi
    fi
    
    return $EXIT_SUCCESS
}

# Check GPU health (if available)
check_gpu_health() {
    if [ "${QUANTUM_GPU_ENABLED:-false}" != "true" ]; then
        log "GPU checks disabled, skipping..."
        return $EXIT_SUCCESS
    fi
    
    log "Checking GPU health..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Check GPU availability
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
        if [ "$gpu_count" -gt 0 ]; then
            log "✓ Found $gpu_count GPU(s)"
            
            # Check GPU utilization
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
            log "GPU utilization: ${gpu_util}%"
            
            # Check GPU memory
            gpu_memory=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
            log "GPU memory utilization: ${gpu_memory}%"
            
            # Check GPU temperature
            gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
            log "GPU temperature: ${gpu_temp}°C"
            
            if [ "$gpu_temp" -gt 80 ] 2>/dev/null; then
                log "⚠ High GPU temperature: ${gpu_temp}°C"
            fi
        else
            log "⚠ No GPUs found"
            return $EXIT_WARNING
        fi
    else
        log "⚠ nvidia-smi not available"
        return $EXIT_WARNING
    fi
    
    return $EXIT_SUCCESS
}

# Check database connectivity
check_database_connectivity() {
    if [ -z "${DATABASE_URL:-}" ]; then
        log "No database URL configured, skipping database check"
        return $EXIT_SUCCESS
    fi
    
    log "Checking database connectivity..."
    
    if python3 -c "
import psycopg2
import os
import sys

try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'], connect_timeout=5)
    cursor = conn.cursor()
    cursor.execute('SELECT 1')
    cursor.fetchone()
    cursor.close()
    conn.close()
    print('✓ Database connection successful')
except Exception as e:
    print(f'✗ Database connection failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log "✓ Database connectivity OK"
    else
        log "✗ Database connectivity failed"
        return $EXIT_FAILURE
    fi
    
    return $EXIT_SUCCESS
}

# Check Redis connectivity
check_redis_connectivity() {
    if [ -z "${REDIS_URL:-}" ]; then
        log "No Redis URL configured, skipping Redis check"
        return $EXIT_SUCCESS
    fi
    
    log "Checking Redis connectivity..."
    
    if python3 -c "
import redis
import os
import sys

try:
    r = redis.from_url(os.environ['REDIS_URL'], socket_connect_timeout=5)
    r.ping()
    print('✓ Redis connection successful')
except Exception as e:
    print(f'✗ Redis connection failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log "✓ Redis connectivity OK"
    else
        log "✗ Redis connectivity failed"
        return $EXIT_FAILURE
    fi
    
    return $EXIT_SUCCESS
}

# Check application metrics
check_application_metrics() {
    log "Checking application metrics..."
    
    if ! check_http_endpoint "$METRICS_ENDPOINT" "Metrics endpoint" false; then
        log "Metrics endpoint unavailable, but continuing..."
        return $EXIT_SUCCESS
    fi
    
    # Parse metrics for anomalies
    if metrics_data=$(curl -s --max-time "$TIMEOUT" "$METRICS_ENDPOINT" 2>/dev/null); then
        # Check request rate
        if echo "$metrics_data" | grep -q "http_requests_total"; then
            log "✓ HTTP metrics available"
        fi
        
        # Check error rate
        if echo "$metrics_data" | grep -q "http_requests_failed_total"; then
            log "✓ Error metrics available"
        fi
        
        # Check response time
        if echo "$metrics_data" | grep -q "http_request_duration_seconds"; then
            log "✓ Response time metrics available"
        fi
    fi
    
    return $EXIT_SUCCESS
}

# Main health check function
main() {
    log "Starting comprehensive health check..."
    
    local overall_status=$EXIT_SUCCESS
    local checks_passed=0
    local checks_failed=0
    local checks_warned=0
    
    # Run all health checks
    local checks=(
        "check_basic_health"
        "check_readiness"
        "check_quantum_systems"
        "check_system_resources"
        "check_gpu_health"
        "check_database_connectivity"
        "check_redis_connectivity"
        "check_application_metrics"
    )
    
    for check in "${checks[@]}"; do
        if $check; then
            case $? in
                $EXIT_SUCCESS)
                    ((checks_passed++))
                    ;;
                $EXIT_WARNING)
                    ((checks_warned++))
                    if [ $overall_status -eq $EXIT_SUCCESS ]; then
                        overall_status=$EXIT_WARNING
                    fi
                    ;;
                $EXIT_FAILURE)
                    ((checks_failed++))
                    overall_status=$EXIT_FAILURE
                    ;;
            esac
        else
            case $? in
                $EXIT_WARNING)
                    ((checks_warned++))
                    if [ $overall_status -eq $EXIT_SUCCESS ]; then
                        overall_status=$EXIT_WARNING
                    fi
                    ;;
                *)
                    ((checks_failed++))
                    overall_status=$EXIT_FAILURE
                    ;;
            esac
        fi
    done
    
    # Summary
    log "Health check summary:"
    log "  ✓ Passed: $checks_passed"
    log "  ⚠ Warnings: $checks_warned"
    log "  ✗ Failed: $checks_failed"
    
    case $overall_status in
        $EXIT_SUCCESS)
            log "Overall health status: HEALTHY"
            ;;
        $EXIT_WARNING)
            log "Overall health status: HEALTHY (with warnings)"
            ;;
        $EXIT_FAILURE)
            log "Overall health status: UNHEALTHY"
            ;;
    esac
    
    exit $overall_status
}

# Run health check
main "$@"