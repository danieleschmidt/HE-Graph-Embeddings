#!/bin/bash

# Quantum Task Planner - EKS Node User Data Script
# Optimizes nodes for quantum-inspired algorithms and high-performance computing

set -euo pipefail

# Variables
QUANTUM_OPTIMIZATION_LEVEL=${quantum_optimization_level}
ENABLE_GPU_SUPPORT=${enable_gpu_support}
CLUSTER_ENDPOINT="${cluster_endpoint}"
CLUSTER_CA="${cluster_ca}"
CLUSTER_NAME="${cluster_name}"

# Logging setup
exec > >(tee /var/log/quantum-node-init.log)
exec 2>&1

echo "Starting Quantum Task Planner node initialization..."
echo "Quantum optimization level: $QUANTUM_OPTIMIZATION_LEVEL"
echo "GPU support: $ENABLE_GPU_SUPPORT"

# Update system
yum update -y
yum install -y aws-cli jq htop iotop

# Install required packages for quantum computing
yum groupinstall -y "Development Tools"
yum install -y \
    cmake3 \
    gcc-c++ \
    python3-devel \
    python3-pip \
    cuda-toolkit-11-8 \
    openmpi-devel \
    hwloc-devel \
    numactl-devel

# Create symlinks for CUDA
if [ "$ENABLE_GPU_SUPPORT" = "true" ]; then
    echo "Setting up CUDA environment..."
    ln -sf /usr/local/cuda-11.8 /usr/local/cuda
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
    chmod +x /etc/profile.d/cuda.sh
fi

# Install Python packages for quantum computing
pip3 install --upgrade pip setuptools wheel
pip3 install \
    numpy==1.24.3 \
    scipy==1.10.1 \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install quantum computing libraries
pip3 install \
    qiskit==0.44.1 \
    cirq==1.2.0 \
    pennylane==0.32.0 \
    networkx==3.1 \
    matplotlib==3.7.2

# Optimize system for quantum workloads
echo "Optimizing system for quantum workloads (level $QUANTUM_OPTIMIZATION_LEVEL)..."

if [ "$QUANTUM_OPTIMIZATION_LEVEL" -ge 2 ]; then
    # CPU performance optimization
    echo "Applying CPU optimizations..."
    
    # Set CPU governor to performance
    echo 'GOVERNOR=performance' > /etc/default/cpufrequtils
    
    # Disable CPU throttling
    echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
    
    # Optimize CPU affinity and NUMA
    echo 'kernel.numa_balancing=0' >> /etc/sysctl.conf
    echo 'vm.zone_reclaim_mode=0' >> /etc/sysctl.conf
    
    # Increase network buffer sizes for quantum communication
    echo 'net.core.rmem_max=134217728' >> /etc/sysctl.conf
    echo 'net.core.wmem_max=134217728' >> /etc/sysctl.conf
    echo 'net.ipv4.tcp_rmem=4096 87380 134217728' >> /etc/sysctl.conf
    echo 'net.ipv4.tcp_wmem=4096 65536 134217728' >> /etc/sysctl.conf
fi

if [ "$QUANTUM_OPTIMIZATION_LEVEL" -ge 3 ]; then
    # Memory optimization for large quantum state vectors
    echo "Applying memory optimizations..."
    
    # Increase shared memory limits
    echo 'kernel.shmmax=68719476736' >> /etc/sysctl.conf  # 64GB
    echo 'kernel.shmall=4294967296' >> /etc/sysctl.conf   # 16TB
    
    # Optimize memory allocation
    echo 'vm.overcommit_memory=1' >> /etc/sysctl.conf
    echo 'vm.swappiness=10' >> /etc/sysctl.conf
    
    # Enable transparent huge pages for better memory performance
    echo always > /sys/kernel/mm/transparent_hugepage/enabled
    echo always > /sys/kernel/mm/transparent_hugepage/defrag
fi

if [ "$QUANTUM_OPTIMIZATION_LEVEL" -ge 4 ]; then
    # Advanced I/O optimization for quantum data processing
    echo "Applying advanced I/O optimizations..."
    
    # Optimize block device settings
    echo noop > /sys/block/nvme0n1/queue/scheduler 2>/dev/null || true
    echo 0 > /sys/block/nvme0n1/queue/rotational 2>/dev/null || true
    echo 2 > /sys/block/nvme0n1/queue/rq_affinity 2>/dev/null || true
    
    # Increase I/O limits
    echo '* soft nofile 1048576' >> /etc/security/limits.conf
    echo '* hard nofile 1048576' >> /etc/security/limits.conf
    echo '* soft nproc 1048576' >> /etc/security/limits.conf
    echo '* hard nproc 1048576' >> /etc/security/limits.conf
fi

if [ "$QUANTUM_OPTIMIZATION_LEVEL" -eq 5 ]; then
    # Maximum optimization - experimental features
    echo "Applying maximum optimization (experimental)..."
    
    # Real-time scheduling for quantum operations
    echo 'kernel.sched_rt_runtime_us=950000' >> /etc/sysctl.conf
    echo 'kernel.sched_rt_period_us=1000000' >> /etc/sysctl.conf
    
    # Disable power management features that can cause latency
    echo 1 > /sys/devices/system/cpu/intel_pstate/disable 2>/dev/null || true
    
    # Optimize interrupt handling
    echo 'net.core.busy_poll=50' >> /etc/sysctl.conf
    echo 'net.core.busy_read=50' >> /etc/sysctl.conf
fi

# Apply sysctl changes
sysctl -p

# Configure Docker for quantum workloads
echo "Configuring Docker for quantum workloads..."
mkdir -p /etc/docker

cat > /etc/docker/daemon.json << 'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "5"
    },
    "storage-driver": "overlay2",
    "default-runtime": "runc",
    "default-shm-size": "2g",
    "default-ulimits": {
        "nofile": {
            "name": "nofile",
            "hard": 1048576,
            "soft": 1048576
        },
        "nproc": {
            "name": "nproc",
            "hard": 1048576,
            "soft": 1048576
        }
    }
}
EOF

# Add GPU runtime if GPU support is enabled
if [ "$ENABLE_GPU_SUPPORT" = "true" ]; then
    echo "Configuring GPU runtime for Docker..."
    
    # Install NVIDIA Docker runtime
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu18.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update && apt-get install -y nvidia-docker2
    
    # Update Docker daemon configuration for GPU
    jq '.runtimes += {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}}' /etc/docker/daemon.json > /tmp/daemon.json
    mv /tmp/daemon.json /etc/docker/daemon.json
fi

# Restart Docker
systemctl restart docker

# Configure kubelet for quantum workloads
echo "Configuring kubelet for quantum optimization..."

# Create kubelet config directory
mkdir -p /etc/kubernetes/kubelet

# Configure kubelet with quantum-optimized settings
cat > /etc/kubernetes/kubelet/kubelet-config.json << EOF
{
    "kind": "KubeletConfiguration",
    "apiVersion": "kubelet.config.k8s.io/v1beta1",
    "address": "0.0.0.0",
    "port": 10250,
    "readOnlyPort": 0,
    "cgroupDriver": "systemd",
    "hairpinMode": "hairpin-veth",
    "serializeImagePulls": false,
    "featureGates": {
        "RotateKubeletServerCertificate": true,
        "CPUManager": true,
        "TopologyManager": true
    },
    "cpuManagerPolicy": "static",
    "cpuManagerReconcilePeriod": "10s",
    "topologyManagerPolicy": "single-numa-node",
    "systemReserved": {
        "cpu": "200m",
        "memory": "512Mi",
        "ephemeral-storage": "2Gi"
    },
    "kubeReserved": {
        "cpu": "200m",
        "memory": "512Mi",
        "ephemeral-storage": "2Gi"
    },
    "evictionHard": {
        "memory.available": "200Mi",
        "nodefs.available": "10%",
        "nodefs.inodesFree": "5%",
        "imagefs.available": "15%"
    },
    "maxPods": 110,
    "authentication": {
        "anonymous": {
            "enabled": false
        },
        "webhook": {
            "enabled": true,
            "cacheTTL": "2m0s"
        }
    },
    "authorization": {
        "mode": "Webhook",
        "webhook": {
            "cacheAuthorizedTTL": "5m0s",
            "cacheUnauthorizedTTL": "30s"
        }
    },
    "eventRecordQPS": 0,
    "protectKernelDefaults": true,
    "streamingConnectionIdleTimeout": "30m",
    "makeIPTablesUtilChains": true,
    "iptablesMasqueradeBit": 14,
    "iptablesDropBit": 15,
    "containerLogMaxSize": "50Mi",
    "containerLogMaxFiles": 10
}
EOF

# Set up EKS node bootstrapping
echo "Bootstrapping EKS node..."

# Download and install the EKS optimized AMI bootstrap script
curl -o /tmp/install-worker.sh https://amazon-eks.s3-us-west-2.amazonaws.com/1.21.2/2021-07-05/bin/linux/amd64/install-worker.sh
chmod +x /tmp/install-worker.sh

# Set up the kubelet arguments for quantum optimization
KUBELET_EXTRA_ARGS=""
if [ "$QUANTUM_OPTIMIZATION_LEVEL" -ge 3 ]; then
    KUBELET_EXTRA_ARGS="--cpu-manager-policy=static --topology-manager-policy=single-numa-node"
fi

if [ "$ENABLE_GPU_SUPPORT" = "true" ]; then
    KUBELET_EXTRA_ARGS="$KUBELET_EXTRA_ARGS --container-runtime=docker --runtime-request-timeout=15m"
fi

# Bootstrap the node
/etc/eks/bootstrap.sh $CLUSTER_NAME \
    --apiserver-endpoint $CLUSTER_ENDPOINT \
    --b64-cluster-ca $CLUSTER_CA \
    --kubelet-extra-args "$KUBELET_EXTRA_ARGS" \
    --enable-docker-bridge true

# Install quantum performance monitoring tools
echo "Installing quantum performance monitoring tools..."

# Install custom monitoring agents
cat > /usr/local/bin/quantum-monitor.sh << 'EOF'
#!/bin/bash
# Quantum Task Planner - Performance Monitoring Script

while true; do
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Collect CPU metrics
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    
    # Collect memory metrics
    memory_info=$(free | grep Mem:)
    memory_total=$(echo $memory_info | awk '{print $2}')
    memory_used=$(echo $memory_info | awk '{print $3}')
    memory_usage=$(echo "scale=2; ($memory_used * 100) / $memory_total" | bc)
    
    # Collect GPU metrics if available
    if command -v nvidia-smi &> /dev/null; then
        gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        gpu_memory=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits | head -1)
    else
        gpu_usage="N/A"
        gpu_memory="N/A"
    fi
    
    # Collect network metrics
    network_rx=$(cat /proc/net/dev | grep eth0 | awk '{print $2}')
    network_tx=$(cat /proc/net/dev | grep eth0 | awk '{print $10}')
    
    # Log metrics
    echo "{\"timestamp\":\"$timestamp\",\"cpu_usage\":$cpu_usage,\"memory_usage\":$memory_usage,\"gpu_usage\":\"$gpu_usage\",\"gpu_memory\":\"$gpu_memory\",\"network_rx\":$network_rx,\"network_tx\":$network_tx}" >> /var/log/quantum-metrics.log
    
    sleep 30
done
EOF

chmod +x /usr/local/bin/quantum-monitor.sh

# Create systemd service for quantum monitoring
cat > /etc/systemd/system/quantum-monitor.service << 'EOF'
[Unit]
Description=Quantum Task Planner Performance Monitor
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/quantum-monitor.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl enable quantum-monitor.service
systemctl start quantum-monitor.service

# Configure log rotation
cat > /etc/logrotate.d/quantum-logs << 'EOF'
/var/log/quantum-*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    postrotate
        systemctl reload quantum-monitor.service
    endscript
}
EOF

# Set up quantum-specific environment variables
echo "Setting up quantum environment variables..."

cat > /etc/profile.d/quantum-env.sh << EOF
# Quantum Task Planner Environment Variables
export QUANTUM_OPTIMIZATION_LEVEL=$QUANTUM_OPTIMIZATION_LEVEL
export QUANTUM_GPU_ENABLED=$ENABLE_GPU_SUPPORT
export QUANTUM_NODE_TYPE="eks-optimized"

# CUDA environment (if GPU enabled)
if [ "$ENABLE_GPU_SUPPORT" = "true" ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH=\$PATH:\$CUDA_HOME/bin
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CUDA_HOME/lib64
fi

# Optimization flags
export OMP_NUM_THREADS=\$(nproc)
export MKL_NUM_THREADS=\$(nproc)
export OPENBLAS_NUM_THREADS=\$(nproc)

# Memory optimization
export MALLOC_ARENA_MAX=4
export MALLOC_MMAP_THRESHOLD_=1048576

# Network optimization for quantum communication
export QUANTUM_NETWORK_BUFFER_SIZE=134217728
export QUANTUM_MAX_CONNECTIONS=10000
EOF

chmod +x /etc/profile.d/quantum-env.sh

# Install quantum-specific Kubernetes resources
echo "Installing quantum-specific Kubernetes resources..."

# Create directory for quantum configurations
mkdir -p /etc/kubernetes/quantum

# Install quantum device plugin (if GPU enabled)
if [ "$ENABLE_GPU_SUPPORT" = "true" ]; then
    cat > /etc/kubernetes/quantum/nvidia-device-plugin.yaml << 'EOF'
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  updateStrategy:
    type: RollingUpdate
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      priorityClassName: "system-node-critical"
      containers:
      - image: nvidia/k8s-device-plugin:v0.12.3
        name: nvidia-device-plugin-ctr
        args: ["--fail-on-init-error=false"]
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
        volumeMounts:
          - name: device-plugin
            mountPath: /var/lib/kubelet/device-plugins
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins
      nodeSelector:
        kubernetes.io/os: linux
EOF
fi

# Validate installation
echo "Validating quantum node setup..."

# Check if all required packages are installed
REQUIRED_PACKAGES=("docker" "kubelet" "aws-cli" "htop" "cmake3")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! command -v $package &> /dev/null; then
        echo "ERROR: Required package $package is not installed"
        exit 1
    fi
done

# Check if quantum monitoring is running
if ! systemctl is-active --quiet quantum-monitor.service; then
    echo "ERROR: Quantum monitoring service is not running"
    exit 1
fi

# Check GPU setup if enabled
if [ "$ENABLE_GPU_SUPPORT" = "true" ]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: GPU support enabled but nvidia-smi not found"
    else
        echo "GPU Status:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    fi
fi

# Final system information
echo "Quantum node initialization completed successfully!"
echo "Node specifications:"
echo "- CPU cores: $(nproc)"
echo "- Memory: $(free -h | grep Mem: | awk '{print $2}')"
echo "- Disk space: $(df -h / | tail -1 | awk '{print $4}') available"
echo "- Quantum optimization level: $QUANTUM_OPTIMIZATION_LEVEL"
echo "- GPU support: $ENABLE_GPU_SUPPORT"
echo "- Kernel: $(uname -r)"
echo "- Docker: $(docker --version)"
echo "- Kubelet: $(kubelet --version)"

# Create completion marker
touch /opt/quantum-node-init-complete

echo "Quantum Task Planner node is ready for quantum workloads!"