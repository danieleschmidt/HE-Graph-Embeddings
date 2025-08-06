# HE-Graph-Embeddings Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying HE-Graph-Embeddings in production environments using Terraform and Kubernetes. The system supports multi-region deployment with auto-scaling, monitoring, and compliance features.

## Prerequisites

### Required Tools

```bash
# Install Terraform
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install terraform

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update && sudo apt-get install helm
```

### AWS Setup

1. **Configure AWS Credentials**:
```bash
aws configure
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region name: us-east-1
# Default output format: json
```

2. **Create IAM Role for Deployment**:
```bash
aws iam create-role --role-name HEGraphDeploymentRole --assume-role-policy-document file://deployment/policies/trust-policy.json
aws iam attach-role-policy --role-name HEGraphDeploymentRole --policy-arn arn:aws:iam::aws:policy/PowerUserAccess
```

## Configuration

### Environment Variables

Create a `.env` file with deployment configuration:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_SECONDARY_REGIONS=eu-west-1,ap-southeast-1
AWS_ACCOUNT_ID=123456789012

# Project Configuration
PROJECT_NAME=he-graph-embeddings
ENVIRONMENT=production
DOMAIN_NAME=he-graph.terragon.ai

# Scaling Configuration
MIN_REPLICAS=2
MAX_REPLICAS=50
TARGET_CPU_UTILIZATION=70
ENABLE_GPU_INSTANCES=true

# Security Configuration
ENABLE_WAF=true
ENABLE_SHIELD=true
CERTIFICATE_ARN=arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012

# Monitoring
ENABLE_CLOUDWATCH=true
LOG_RETENTION_DAYS=30
ENABLE_XRAY=true
```

### Terraform Variables

Create `terraform/terraform.tfvars`:

```hcl
# Project Configuration
project_name = "he-graph-embeddings"
environment  = "production"
regions      = ["us-east-1", "eu-west-1", "ap-southeast-1"]

# Network Configuration
vpc_cidr = "10.0.0.0/16"
availability_zones = {
  "us-east-1"    = ["us-east-1a", "us-east-1b", "us-east-1c"]
  "eu-west-1"    = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]
  "ap-southeast-1" = ["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"]
}

# Application Configuration
domain_name = "he-graph.terragon.ai"
certificate_arn = "arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"

# Scaling Configuration
min_replicas = 2
max_replicas = 50
target_cpu_utilization = 70
enable_gpu_instances = true

# Security Configuration
enable_waf = true
enable_shield_advanced = true
enable_guardduty = true

# Monitoring Configuration
log_retention_days = 30
enable_detailed_monitoring = true
```

## Infrastructure Deployment

### Step 1: Initialize Terraform

```bash
cd deployment/terraform
terraform init
terraform workspace new production
terraform workspace select production
```

### Step 2: Plan Infrastructure

```bash
# Review the planned infrastructure changes
terraform plan -var-file="terraform.tfvars" -out=tfplan

# Review the plan output carefully
terraform show tfplan
```

### Step 3: Deploy Base Infrastructure

```bash
# Deploy networking and security components
terraform apply -target=module.vpc -target=module.security -var-file="terraform.tfvars"

# Deploy EKS clusters
terraform apply -target=module.eks -var-file="terraform.tfvars"

# Deploy remaining infrastructure
terraform apply -var-file="terraform.tfvars"
```

### Step 4: Configure kubectl

```bash
# Update kubeconfig for each region
aws eks update-kubeconfig --region us-east-1 --name he-graph-embeddings-us-east-1
aws eks update-kubeconfig --region eu-west-1 --name he-graph-embeddings-eu-west-1
aws eks update-kubeconfig --region ap-southeast-1 --name he-graph-embeddings-ap-southeast-1

# Verify cluster connectivity
kubectl cluster-info
```

## Application Deployment

### Step 1: Container Registry Setup

```bash
# Create ECR repositories
aws ecr create-repository --repository-name he-graph-embeddings --region us-east-1

# Build and push Docker image
docker build -t he-graph-embeddings:latest .
docker tag he-graph-embeddings:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/he-graph-embeddings:latest

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Push image
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/he-graph-embeddings:latest
```

### Step 2: Kubernetes Manifests

Create `k8s/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: he-graph
  labels:
    name: he-graph
    istio-injection: enabled
```

Create `k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: he-graph-config
  namespace: he-graph
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  ENABLE_GPU: "true"
  POLY_MODULUS_DEGREE: "16384"
  COEFF_MODULUS_BITS: "60,40,40,60"
  PRECISION_BITS: "30"
  MAX_BATCH_SIZE: "32"
  WORKER_POOL_SIZE: "10"
```

Create `k8s/secret.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: he-graph-secrets
  namespace: he-graph
type: Opaque
data:
  API_SECRET_KEY: <base64-encoded-secret>
  DATABASE_URL: <base64-encoded-db-url>
  ENCRYPTION_KEY: <base64-encoded-encryption-key>
```

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: he-graph-api
  namespace: he-graph
  labels:
    app: he-graph-api
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: he-graph-api
  template:
    metadata:
      labels:
        app: he-graph-api
        version: v1
    spec:
      serviceAccountName: he-graph-service-account
      containers:
      - name: api
        image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/he-graph-embeddings:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: PORT
          value: "8000"
        envFrom:
        - configMapRef:
            name: he-graph-config
        - secretRef:
            name: he-graph-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 10000
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: tmp
        emptyDir: {}
      - name: logs
        emptyDir: {}
      nodeSelector:
        node-type: compute
      tolerations:
      - key: "compute"
        operator: "Equal"
        value: "intensive"
        effect: "NoSchedule"
```

### Step 3: Deploy to Kubernetes

```bash
# Apply base manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Deploy GPU workloads (if enabled)
kubectl apply -f k8s/gpu-deployment.yaml

# Verify deployment
kubectl get pods -n he-graph
kubectl get services -n he-graph
```

## Monitoring Setup

### Step 1: Install Prometheus and Grafana

```bash
# Add Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=secure-password \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi

# Install custom dashboards
kubectl apply -f monitoring/grafana-dashboards.yaml
```

### Step 2: Configure Logging

```bash
# Install Fluent Bit for log collection
helm repo add fluent https://fluent.github.io/helm-charts
helm install fluent-bit fluent/fluent-bit \
  --namespace logging \
  --create-namespace \
  --set config.outputs="[OUTPUT]\n    Name cloudwatch_logs\n    Match *\n    region us-east-1\n    log_group_name /aws/eks/he-graph/application\n    auto_create_group true"
```

### Step 3: Set up Alerting

Create `monitoring/alerts.yaml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: he-graph-alerts
  namespace: he-graph
spec:
  groups:
  - name: he-graph.rules
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} per second"
    
    - alert: NoisebudgetLow
      expr: hegraph_noise_budget < 20
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "Encryption noise budget running low"
        description: "Noise budget is {{ $value }} bits"
    
    - alert: GPUMemoryHigh
      expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9
      for: 3m
      labels:
        severity: warning
      annotations:
        summary: "GPU memory usage high"
        description: "GPU memory usage is {{ $value | humanizePercentage }}"
```

## Security Configuration

### Step 1: Network Policies

Create `k8s/network-policy.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: he-graph-network-policy
  namespace: he-graph
spec:
  podSelector:
    matchLabels:
      app: he-graph-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    - podSelector:
        matchLabels:
          app: istio-proxy
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS outbound
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### Step 2: Pod Security Policies

Create `k8s/pod-security-policy.yaml`:

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: he-graph-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### Step 3: Service Mesh (Istio)

```bash
# Install Istio
curl -L https://istio.io/downloadIstio | sh -
cd istio-*
export PATH=$PWD/bin:$PATH
istioctl install --set values.defaultRevision=default

# Enable automatic sidecar injection
kubectl label namespace he-graph istio-injection=enabled

# Apply Istio configuration
kubectl apply -f k8s/istio/virtual-service.yaml
kubectl apply -f k8s/istio/destination-rule.yaml
kubectl apply -f k8s/istio/gateway.yaml
```

## Auto-scaling Configuration

### Step 1: Horizontal Pod Autoscaler

Create `k8s/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: he-graph-api-hpa
  namespace: he-graph
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: he-graph-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: hegraph_active_contexts
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
```

### Step 2: Vertical Pod Autoscaler

```bash
# Install VPA
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/download/vertical-pod-autoscaler-0.13.0/vpa-release-0.13.0-yaml

# Create VPA resource
kubectl apply -f k8s/vpa.yaml
```

## Disaster Recovery

### Step 1: Backup Strategy

Create `scripts/backup.sh`:

```bash
#!/bin/bash
set -e

# Backup Kubernetes configurations
kubectl get all --all-namespaces -o yaml > backups/k8s-resources-$(date +%Y%m%d).yaml

# Backup Terraform state
aws s3 cp terraform.tfstate s3://he-graph-terraform-state-backup/terraform-$(date +%Y%m%d).tfstate

# Backup application data
kubectl exec -n he-graph deployment/he-graph-api -- /app/scripts/backup-data.sh

# Backup encryption keys (handled by KMS automatically)
echo "Encryption keys backed up via AWS KMS"

# Test backup integrity
echo "Testing backup integrity..."
kubectl apply --dry-run=server -f backups/k8s-resources-$(date +%Y%m%d).yaml
```

### Step 2: Disaster Recovery Plan

Create `docs/DISASTER_RECOVERY.md` with detailed procedures for:

1. **Complete region failure**: Failover to secondary region
2. **Data center outage**: Multi-AZ resilience
3. **Application failure**: Rolling deployment and rollback
4. **Security breach**: Incident response procedures

## Performance Tuning

### CPU and Memory Optimization

```yaml
# Resource requests and limits tuning
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
    ephemeral-storage: "2Gi"
  limits:
    memory: "16Gi" 
    cpu: "8000m"
    ephemeral-storage: "10Gi"

# JVM tuning for Java components
env:
- name: JAVA_OPTS
  value: "-Xms2g -Xmx8g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

### GPU Optimization

```yaml
# GPU node pool configuration
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1

# GPU memory sharing (if supported)
env:
- name: CUDA_MPS_PIPE_DIRECTORY
  value: "/tmp/nvidia-mps"
- name: CUDA_MPS_LOG_DIRECTORY  
  value: "/tmp/nvidia-log"
```

## Maintenance and Updates

### Rolling Updates

```bash
# Update application image
kubectl set image deployment/he-graph-api api=123456789012.dkr.ecr.us-east-1.amazonaws.com/he-graph-embeddings:v2.0.0 -n he-graph

# Monitor rollout
kubectl rollout status deployment/he-graph-api -n he-graph

# Rollback if needed
kubectl rollout undo deployment/he-graph-api -n he-graph
```

### Node Updates

```bash
# Drain node for maintenance
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Cordon node to prevent scheduling
kubectl cordon <node-name>

# Uncordon after maintenance
kubectl uncordon <node-name>
```

## Troubleshooting

### Common Issues

1. **Pod OOMKilled**: Increase memory limits
2. **GPU not available**: Check node labels and taints
3. **Noise budget exhausted**: Optimize encryption parameters
4. **High latency**: Check network policies and service mesh configuration

### Debug Commands

```bash
# Check pod logs
kubectl logs -f deployment/he-graph-api -n he-graph

# Execute into pod
kubectl exec -it deployment/he-graph-api -n he-graph -- /bin/bash

# Check resource usage
kubectl top nodes
kubectl top pods -n he-graph

# Describe resources
kubectl describe pod <pod-name> -n he-graph
kubectl describe node <node-name>
```

This deployment guide provides a comprehensive foundation for running HE-Graph-Embeddings in production with high availability, security, and scalability.