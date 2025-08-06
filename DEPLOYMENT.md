# Production Deployment Guide

This guide covers production deployment of HE-Graph-Embeddings v2.0.0 with TERRAGON SDLC implementation.

## 🚀 Quick Start

### Single-Region Production Deployment

```bash
# Clone repository
git clone https://github.com/danieleschmidt/HE-Graph-Embeddings.git
cd HE-Graph-Embeddings

# Set environment variables
export HE_GRAPH_REGION="us-east-1"
export COMPLIANCE_FRAMEWORKS="GDPR,CCPA,SOC2"
export GRAFANA_PASSWORD="your-secure-password"
export REDIS_PASSWORD="your-redis-password"

# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
curl http://localhost:8000/health
```

### Multi-Region Deployment with Terraform

```bash
# Configure Terraform
cd deployment/terraform
terraform init

# Plan deployment
terraform plan -var="regions=[\"us-east-1\",\"eu-west-1\",\"ap-northeast-1\"]"

# Apply configuration
terraform apply -auto-approve

# Verify health across regions
for region in us-east-1 eu-west-1 ap-northeast-1; do
    curl https://he-graph-${region}.terragon.ai/health
done
```

## 🏗️ Architecture Overview

### Production Stack

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │  Region Router  │    │ Compliance Mgr  │
│     (Nginx)     │────│   (FastAPI)     │────│   (GDPR/CCPA)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   HE-Graph API  │    │   Data Storage  │
│ (Grafana/Prom)  │    │   (4 workers)   │    │  (Redis/Logs)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Log Aggreg.   │    │   CUDA Kernels  │    │  Security Scan  │
│ (ELK/Fluentd)   │    │  (Multi-GPU)    │    │  (Background)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Dependencies

- **Core API**: FastAPI with uvicorn workers
- **GPU Processing**: CUDA kernels for HE operations
- **Caching**: Redis for session and computation cache
- **Monitoring**: Prometheus + Grafana dashboards
- **Logging**: Fluentd + Elasticsearch + Kibana
- **Load Balancing**: Nginx with SSL termination
- **Health Checks**: Automated health monitoring
- **Security**: Background vulnerability scanning

## 🔧 Configuration

### Environment Variables

#### Required Variables

```bash
# Core configuration
export HE_GRAPH_REGION="us-east-1"              # Deployment region
export COMPLIANCE_FRAMEWORKS="GDPR,CCPA,SOC2"  # Compliance requirements
export PYTHONPATH="/app/src"                    # Python module path

# Security
export REDIS_PASSWORD="your-secure-redis-password"
export GRAFANA_PASSWORD="your-grafana-password"

# Optional optimization
export HE_GRAPH_MAX_WORKERS="8"                 # API worker count
export HE_GRAPH_ENABLE_CACHING="true"           # Enable caching
export HE_GRAPH_ENABLE_MONITORING="true"        # Enable monitoring
export HE_GRAPH_LOG_LEVEL="INFO"                # Logging level
```

#### Advanced Configuration

```bash
# GPU configuration  
export CUDA_VISIBLE_DEVICES="0,1,2,3"           # Available GPUs
export REQUIRE_GPU="true"                       # Require GPU for startup

# Performance tuning
export HE_GRAPH_BATCH_SIZE="1024"               # Processing batch size
export HE_GRAPH_CACHE_SIZE_GB="16"              # Cache memory limit
export HE_GRAPH_WORKER_TIMEOUT="300"            # Request timeout

# Multi-region settings
export HE_GRAPH_FAILOVER_REGIONS="us-west-2,ca-central-1"
export HE_GRAPH_ROUTING_STRATEGY="latency_optimized"
export HE_GRAPH_DATA_RESIDENCY="true"           # Enforce data residency

# External services
export REDIS_HOST="redis"                       # Redis hostname
export REDIS_PORT="6379"                        # Redis port
export PROMETHEUS_HOST="prometheus"             # Prometheus hostname
export PROMETHEUS_PORT="9090"                   # Prometheus port

# Backup configuration
export BACKUP_S3_BUCKET="he-graph-backups"      # S3 backup bucket
export AWS_ACCESS_KEY_ID="your-key"             # AWS credentials
export AWS_SECRET_ACCESS_KEY="your-secret"      # AWS secret key
```

### Configuration Files

#### production.yaml

```yaml
# /app/config/production.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 8
  timeout: 300
  
encryption:
  poly_modulus_degree: 32768
  coeff_modulus_bits: [60, 40, 40, 40, 40, 60]
  scale: 1099511627776  # 2^40
  security_level: 128
  
caching:
  enabled: true
  redis_url: "redis://redis:6379"
  ttl_seconds: 3600
  max_memory_gb: 16
  
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30
  
logging:
  level: "INFO"
  format: "json"
  correlation_id: true
  audit_enabled: true
  
compliance:
  frameworks: ["GDPR", "CCPA", "SOC2"]
  data_retention_days: 2555
  audit_logging: true
  consent_required: true
  
security:
  scan_interval: 86400
  vulnerability_threshold: "medium"
  auto_remediation: false
  
performance:
  batch_size: 1024
  concurrent_requests: 100
  gpu_memory_fraction: 0.8
```

## 🌍 Multi-Region Deployment

### Supported Regions

| Region | Location | Compliance | GPU Support | Latency Tier |
|--------|----------|------------|-------------|--------------|
| us-east-1 | N. Virginia | CCPA, HIPAA, SOC2 | ✅ V100, A100 | 1 |
| us-west-2 | Oregon | CCPA, HIPAA, SOC2 | ✅ V100, A100 | 2 |
| ca-central-1 | Canada | PIPEDA, PHIPA | ✅ V100, A100 | 2 |
| eu-west-1 | Ireland | GDPR, ISO27001 | ✅ V100, A100 | 1 |
| eu-central-1 | Frankfurt | GDPR, BDSG, ISO27001 | ✅ V100, A100 | 1 |
| eu-north-1 | Stockholm | GDPR, ISO27001 | ❌ CPU Only | 2 |
| ap-northeast-1 | Tokyo | APPI, ISO27001 | ✅ V100, A100 | 1 |
| ap-southeast-1 | Singapore | PDPA, ISO27001 | ✅ V100, A100 | 2 |
| ap-south-1 | Mumbai | IT_ACT, ISO27001 | ✅ V100, A100 | 3 |
| sa-east-1 | São Paulo | LGPD, ISO27001 | ✅ V100, A100 | 4 |

### Region Selection Logic

```python
# Automatic region selection based on:
# 1. User location (lowest latency)
# 2. Compliance requirements (data residency)  
# 3. GPU availability (if required)
# 4. Load balancing (current capacity)
# 5. Failover requirements (backup regions)

from src.deployment import RegionalRouter, RoutingRequest

router = RegionalRouter(multi_region_manager, data_residency_manager)

request = RoutingRequest(
    user_id="user123",
    user_location="DE",                    # Germany
    compliance_requirements=["GDPR"],       # GDPR compliance required
    preferred_strategy="compliance_first",  # Prioritize compliance
    require_gpu=True                       # GPU processing needed
)

response = await router.route_request(request)
# Result: eu-central-1 (Frankfurt) - GDPR compliant, GPU available, lowest latency from Germany
```

### Failover Configuration

```yaml
# Region failover matrix
failover:
  us-east-1:
    primary: ["us-west-2", "ca-central-1"]
    secondary: ["eu-west-1"]
  
  eu-west-1:
    primary: ["eu-central-1", "eu-north-1"] 
    secondary: ["us-east-1"]
    
  ap-northeast-1:
    primary: ["ap-southeast-1", "ap-south-1"]
    secondary: ["us-west-2"]

# Automatic failover triggers
failover_triggers:
  - health_check_failures: 3
  - response_time_ms: 5000
  - error_rate_percent: 10
  - gpu_utilization_percent: 95
```

## 🔐 Security & Compliance

### Compliance Framework Support

#### GDPR (General Data Protection Regulation)
```python
# GDPR compliance features:
# - Explicit consent management
# - Data subject rights (access, deletion, portability)
# - Data residency in EU regions
# - 72-hour breach notification
# - Privacy impact assessments

from src.compliance import ComplianceManager, ComplianceFramework

compliance_manager = ComplianceManager()

# Register data subject
data_subject = compliance_manager.register_data_subject(
    subject_id="user123",
    region="DE", 
    frameworks=[ComplianceFramework.GDPR],
    data_categories={DataCategory.PERSONAL_IDENTIFIERS},
    is_minor=False
)

# Handle data subject request
response = await compliance_manager.handle_data_subject_request(
    subject_id="user123",
    request_type="access"  # Right to access under GDPR Art. 15
)
```

#### HIPAA (Health Insurance Portability and Accountability Act)
```python
# HIPAA compliance features:
# - Business associate agreements
# - Audit logging for PHI access
# - Encryption at rest and in transit
# - Access controls and authentication
# - 60-day breach notification

# Configure HIPAA-compliant processing
hipaa_config = {
    "encryption_required": True,
    "audit_logging": True, 
    "access_controls": "role_based",
    "data_retention_years": 6,
    "cross_border_transfer": False
}
```

### Security Features

#### Vulnerability Scanning
```bash
# Automated security scanning
docker-compose -f docker-compose.prod.yml --profile security up security-scanner

# Manual security scan
python security/security_scanner.py --target src/ --output security_report.html

# Security policy enforcement
python security/policy_enforcer.py --config security/security_config.yaml
```

#### Encryption Standards
- **Data at Rest**: AES-256-GCM
- **Data in Transit**: TLS 1.3
- **Homomorphic**: CKKS with 128-bit security
- **Key Management**: Hardware Security Module (HSM) support
- **Certificate Management**: Automated Let's Encrypt

## 📊 Monitoring & Observability

### Health Monitoring

```bash
# Comprehensive health check
python scripts/healthcheck.py --output json

# Continuous health monitoring
python scripts/healthcheck.py --daemon --interval 30

# Health check via API
curl http://localhost:8000/health | jq .
```

### Metrics Dashboard

Access Grafana dashboard at `http://localhost:3000`:
- **Username**: admin
- **Password**: (value of `GRAFANA_PASSWORD`)

#### Key Metrics Tracked
- **Performance**: Request latency, throughput, error rates
- **Resources**: CPU, memory, GPU utilization
- **Encryption**: HE operation times, noise budget levels
- **Compliance**: Data processing activities, consent status
- **Security**: Vulnerability scan results, access patterns

### Log Aggregation

```bash
# View application logs
docker-compose -f docker-compose.prod.yml logs -f he-graph-api

# Search logs with Elasticsearch
curl "http://localhost:9200/logs-*/_search?q=ERROR&size=10&pretty"

# Fluentd log processing status
docker-compose -f docker-compose.prod.yml logs -f fluentd
```

## 🚀 Performance Optimization

### GPU Optimization

```python
# Multi-GPU configuration
import torch
from src.python.he_graph import CKKSContext

# Enable multi-GPU processing
contexts = []
for gpu_id in range(torch.cuda.device_count()):
    context = CKKSContext(
        poly_modulus_degree=32768,
        scale=2**40,
        gpu_id=gpu_id,
        enable_caching=True,
        memory_pool_gb=10  # Per-GPU memory pool
    )
    contexts.append(context)

# Distributed processing
from src.utils.performance import ResourcePool
resource_pool = ResourcePool(contexts)
```

### Caching Strategy

```python
# Multi-tier caching configuration
from src.utils.caching import CacheManager

cache_manager = CacheManager(
    l1_cache_size_gb=4,      # In-memory cache
    l2_cache_size_gb=16,     # Redis cache  
    l3_cache_enabled=True,   # Disk cache
    eviction_policy="adaptive",
    prefetch_enabled=True
)

# Cache hit rate optimization
cache_stats = cache_manager.get_statistics()
print(f"L1 hit rate: {cache_stats['l1_hit_rate']:.2%}")
print(f"L2 hit rate: {cache_stats['l2_hit_rate']:.2%}")
```

### Load Balancing

```bash
# Configure Nginx load balancing
# /etc/nginx/conf.d/he-graph.conf

upstream he-graph-backend {
    least_conn;
    server he-graph-api-1:8000 max_fails=3 fail_timeout=30s;
    server he-graph-api-2:8000 max_fails=3 fail_timeout=30s;
    server he-graph-api-3:8000 max_fails=3 fail_timeout=30s;
    server he-graph-api-4:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name api.he-graph.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        proxy_pass http://he-graph-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

## 🔄 Backup & Recovery

### Automated Backups

```bash
# Configure backup service
export BACKUP_S3_BUCKET="he-graph-backups-prod"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Run backup manually
docker-compose -f docker-compose.prod.yml --profile backup up backup

# Schedule automated backups (crontab)
0 2 * * * docker-compose -f /app/docker-compose.prod.yml --profile backup up backup
```

### Disaster Recovery

```bash
# 1. Restore from backup
aws s3 cp s3://he-graph-backups-prod/he-graph-backup-20250806.tar.gz ./backup.tar.gz
tar -xzf backup.tar.gz -C ./restore/

# 2. Restore Redis data
docker-compose -f docker-compose.prod.yml stop redis
cp restore/redis/* ./redis-data/
docker-compose -f docker-compose.prod.yml start redis

# 3. Restore configuration  
cp restore/config/* ./config/

# 4. Restart services
docker-compose -f docker-compose.prod.yml up -d

# 5. Verify health
python scripts/healthcheck.py --fail-on-degraded
```

## 📈 Scaling Guidelines

### Horizontal Scaling

```yaml
# Kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: he-graph-api
spec:
  replicas: 8  # Scale based on load
  selector:
    matchLabels:
      app: he-graph-api
  template:
    metadata:
      labels:
        app: he-graph-api
    spec:
      containers:
      - name: he-graph-api
        image: ghcr.io/danieleschmidt/he-graph-embeddings:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi" 
            cpu: "8"
            nvidia.com/gpu: 1
        env:
        - name: HE_GRAPH_REGION
          value: "us-east-1"
        - name: HE_GRAPH_MAX_WORKERS
          value: "4"
```

### Auto-scaling Configuration

```python
# Auto-scaling based on metrics
from src.utils.monitoring import AutoScaler

auto_scaler = AutoScaler(
    min_replicas=2,
    max_replicas=16,
    target_cpu_percent=70,
    target_memory_percent=80,
    target_gpu_percent=85,
    scale_up_cooldown=300,    # 5 minutes
    scale_down_cooldown=600   # 10 minutes
)

# Custom scaling metrics
auto_scaler.add_metric(
    name="he_operations_per_second", 
    threshold=100,
    action="scale_up"
)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. GPU Memory Issues
```bash
# Check GPU memory usage
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size
export HE_GRAPH_BATCH_SIZE="512"  # Default: 1024
```

#### 2. High Latency
```bash
# Check cache hit rates
curl http://localhost:8000/metrics | grep cache_hit_rate

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Check Redis performance
redis-cli info stats
```

#### 3. Compliance Violations  
```python
# Check compliance status
from src.compliance import compliance_manager

report = compliance_manager.generate_compliance_report(ComplianceFramework.GDPR)
print(f"Compliance score: {report['summary']['compliance_score']}")
print(f"Violations: {report['summary']['violations']}")
```

#### 4. Service Health Issues
```bash
# Comprehensive health check
python scripts/healthcheck.py --output text --fail-on-degraded

# Check individual service health
curl http://localhost:8000/health/detailed
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3000/api/health # Grafana
```

### Debug Mode

```bash
# Enable debug logging
export HE_GRAPH_LOG_LEVEL="DEBUG"
export DEBUG="true"

# Start with debug output
docker-compose -f docker-compose.prod.yml up he-graph-api
```

### Log Analysis

```bash
# Search for errors in logs
docker-compose -f docker-compose.prod.yml logs he-graph-api | grep ERROR

# Monitor real-time logs
docker-compose -f docker-compose.prod.yml logs -f --tail=100 he-graph-api

# Elasticsearch log analysis
curl -X GET "localhost:9200/logs-*/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": [
        {"term": {"level": "ERROR"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  },
  "size": 10
}'
```

## 📞 Support

### Production Support Channels

- **Critical Issues**: Create GitHub issue with `critical` label
- **Security Issues**: Email security@terragon.ai
- **Performance Issues**: Include performance metrics and logs
- **Compliance Questions**: Tag compliance team in issues

### Health Check Endpoints

- **Basic Health**: `GET /health`
- **Detailed Health**: `GET /health/detailed` 
- **Readiness**: `GET /ready`
- **Liveness**: `GET /alive`
- **Metrics**: `GET /metrics`

### Documentation

- **API Documentation**: `http://localhost:8000/docs`
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`
- **Monitoring**: `http://localhost:3000` (Grafana)
- **Metrics**: `http://localhost:9090` (Prometheus)

---

*This deployment guide covers the complete TERRAGON SDLC v2.0.0 production deployment. For development deployment, see `docker-compose.yml`. For advanced configurations, see the `config/` directory.*