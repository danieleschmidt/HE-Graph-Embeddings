# HE-Graph-Embeddings API Reference

## Overview

The HE-Graph-Embeddings API provides RESTful endpoints for privacy-preserving graph neural network operations. All graph data is processed using homomorphic encryption to ensure privacy and security.

## Base URL

```
Production: https://api.he-graph.terragon.ai/v1
Staging: https://staging-api.he-graph.terragon.ai/v1
Local: http://localhost:8000/v1
```

## Authentication

All API requests require authentication using API keys or JWT tokens.

### API Key Authentication

Include your API key in the request headers:

```http
Authorization: Bearer your_api_key_here
Content-Type: application/json
```

### JWT Token Authentication

For user-based authentication, include JWT token:

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

## Core Endpoints

### Health Check

#### GET /health

Returns system health status and metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "encryption_context": {
      "status": "healthy",
      "noise_budget": 85.2,
      "last_check": "2024-01-15T10:29:45Z"
    },
    "database": {
      "status": "healthy",
      "connections": 15,
      "response_time_ms": 12
    },
    "gpu_acceleration": {
      "status": "healthy",
      "devices": 2,
      "memory_usage_percent": 34.5
    }
  },
  "metrics": {
    "requests_per_second": 245.7,
    "average_response_time": 150,
    "error_rate_percent": 0.01
  }
}
```

#### GET /health/ready

Kubernetes readiness probe endpoint.

**Response:**
```json
{
  "ready": true,
  "checks": {
    "encryption_ready": true,
    "model_loaded": true,
    "database_connected": true
  }
}
```

#### GET /health/live

Kubernetes liveness probe endpoint.

**Response:**
```json
{
  "alive": true,
  "uptime_seconds": 3600
}
```

### Encryption Context Management

#### POST /context

Create a new CKKS encryption context.

**Request Body:**
```json
{
  "config": {
    "poly_modulus_degree": 16384,
    "coeff_modulus_bits": [60, 40, 40, 60],
    "scale": 1099511627776,
    "precision_bits": 30
  },
  "context_name": "my_analysis_context",
  "ttl_seconds": 3600
}
```

**Response:**
```json
{
  "context_id": "ctx_1234567890abcdef",
  "context_name": "my_analysis_context",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2024-01-15T11:30:00Z",
  "config": {
    "poly_modulus_degree": 16384,
    "coeff_modulus_bits": [60, 40, 40, 60],
    "scale": 1099511627776,
    "precision_bits": 30,
    "security_level": 128
  },
  "noise_budget": 120.5
}
```

#### GET /context/{context_id}

Retrieve encryption context information.

**Response:**
```json
{
  "context_id": "ctx_1234567890abcdef",
  "context_name": "my_analysis_context",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2024-01-15T11:30:00Z",
  "noise_budget": 98.3,
  "usage_stats": {
    "operations_count": 15,
    "encryption_operations": 5,
    "computation_operations": 8,
    "decryption_operations": 2
  }
}
```

#### DELETE /context/{context_id}

Delete an encryption context and all associated data.

**Response:**
```json
{
  "message": "Context deleted successfully",
  "context_id": "ctx_1234567890abcdef",
  "deleted_at": "2024-01-15T10:35:00Z"
}
```

### Graph Processing

#### POST /graph/encrypt

Encrypt graph data using specified context.

**Request Body:**
```json
{
  "context_id": "ctx_1234567890abcdef",
  "graph_data": {
    "features": [
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
      [0.9, 1.0, 1.1, 1.2]
    ],
    "edge_index": [
      [0, 1, 2],
      [1, 2, 0]
    ],
    "edge_attributes": [
      [0.1, 0.2],
      [0.3, 0.4],
      [0.5, 0.6]
    ]
  },
  "metadata": {
    "dataset_name": "karate_club",
    "num_nodes": 3,
    "num_features": 4
  }
}
```

**Response:**
```json
{
  "encrypted_data_id": "enc_abcdef1234567890",
  "context_id": "ctx_1234567890abcdef",
  "encrypted_at": "2024-01-15T10:30:00Z",
  "data_stats": {
    "num_nodes": 3,
    "num_features": 4,
    "num_edges": 3,
    "encryption_time_ms": 45
  },
  "noise_budget_remaining": 115.2
}
```

#### POST /graph/process/graphsage

Process encrypted graph data using GraphSAGE model.

**Request Body:**
```json
{
  "encrypted_data_id": "enc_abcdef1234567890",
  "model_config": {
    "hidden_channels": [16, 8],
    "output_channels": 4,
    "num_layers": 2,
    "aggregation": "mean",
    "activation": "relu"
  },
  "processing_options": {
    "batch_size": 32,
    "use_gpu": true,
    "optimization_level": "standard"
  }
}
```

**Response:**
```json
{
  "task_id": "task_9876543210fedcba",
  "status": "processing",
  "started_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:32:00Z",
  "model_type": "graphsage",
  "progress": {
    "current_layer": 1,
    "total_layers": 2,
    "completion_percent": 45
  }
}
```

#### POST /graph/process/gat

Process encrypted graph data using Graph Attention Network.

**Request Body:**
```json
{
  "encrypted_data_id": "enc_abcdef1234567890",
  "model_config": {
    "hidden_channels": 16,
    "output_channels": 8,
    "num_heads": 4,
    "attention_type": "additive",
    "dropout": 0.1,
    "edge_dim": 2
  },
  "processing_options": {
    "use_gpu": true,
    "memory_efficient": true
  }
}
```

**Response:**
```json
{
  "task_id": "task_abcd1234efgh5678",
  "status": "processing",
  "started_at": "2024-01-15T10:30:00Z",
  "model_type": "gat",
  "attention_heads": 4,
  "progress": {
    "attention_computed": true,
    "aggregation_complete": false,
    "completion_percent": 70
  }
}
```

### Task Management

#### GET /task/{task_id}

Get processing task status and results.

**Response:**
```json
{
  "task_id": "task_9876543210fedcba",
  "status": "completed",
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:31:45Z",
  "model_type": "graphsage",
  "results": {
    "encrypted_output_id": "out_1234abcd5678efgh",
    "output_shape": [3, 4],
    "processing_time_ms": 105000,
    "noise_budget_consumed": 22.3,
    "performance_metrics": {
      "throughput_nodes_per_second": 285.7,
      "memory_usage_mb": 1024,
      "gpu_utilization_percent": 85
    }
  }
}
```

#### POST /task/{task_id}/cancel

Cancel a running processing task.

**Response:**
```json
{
  "message": "Task cancelled successfully",
  "task_id": "task_9876543210fedcba",
  "cancelled_at": "2024-01-15T10:31:00Z",
  "partial_results": {
    "completed_layers": 1,
    "encrypted_partial_output_id": "out_partial_abcd1234"
  }
}
```

### Result Decryption

#### POST /decrypt/{encrypted_output_id}

Decrypt processing results.

**Request Body:**
```json
{
  "context_id": "ctx_1234567890abcdef",
  "output_format": "json",
  "include_metadata": true
}
```

**Response:**
```json
{
  "decrypted_data": [
    [0.15, 0.23, 0.34, 0.42],
    [0.51, 0.68, 0.75, 0.81],
    [0.92, 1.05, 1.18, 1.24]
  ],
  "metadata": {
    "original_shape": [3, 4],
    "decryption_time_ms": 25,
    "noise_budget_at_decryption": 92.1,
    "data_integrity": "verified"
  },
  "statistics": {
    "min_value": 0.15,
    "max_value": 1.24,
    "mean": 0.695,
    "std_dev": 0.385
  }
}
```

## Batch Processing

### POST /batch/process

Process multiple graphs in a single request.

**Request Body:**
```json
{
  "context_id": "ctx_1234567890abcdef",
  "graphs": [
    {
      "graph_id": "graph_001",
      "features": [[0.1, 0.2], [0.3, 0.4]],
      "edge_index": [[0], [1]]
    },
    {
      "graph_id": "graph_002",
      "features": [[0.5, 0.6], [0.7, 0.8]],
      "edge_index": [[0], [1]]
    }
  ],
  "model_config": {
    "type": "graphsage",
    "hidden_channels": 8,
    "output_channels": 4
  },
  "batch_options": {
    "parallel_processing": true,
    "max_batch_size": 10,
    "timeout_seconds": 300
  }
}
```

**Response:**
```json
{
  "batch_id": "batch_xyz789abc123def",
  "status": "processing",
  "total_graphs": 2,
  "started_at": "2024-01-15T10:30:00Z",
  "individual_tasks": [
    {
      "graph_id": "graph_001",
      "task_id": "task_001_abc123",
      "status": "processing"
    },
    {
      "graph_id": "graph_002", 
      "task_id": "task_002_def456",
      "status": "queued"
    }
  ],
  "progress": {
    "completed": 0,
    "processing": 1,
    "queued": 1,
    "completion_percent": 0
  }
}
```

## Monitoring and Analytics

### GET /metrics

Get system performance metrics.

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "performance": {
    "requests_per_second": 145.7,
    "average_response_time_ms": 250,
    "p95_response_time_ms": 580,
    "p99_response_time_ms": 1200
  },
  "encryption": {
    "active_contexts": 23,
    "average_noise_budget": 87.5,
    "encryption_operations_per_second": 45.2,
    "decryption_operations_per_second": 38.1
  },
  "resources": {
    "cpu_utilization_percent": 65.4,
    "memory_usage_percent": 72.1,
    "gpu_utilization_percent": 45.8,
    "disk_usage_percent": 34.2
  },
  "errors": {
    "error_rate_percent": 0.15,
    "common_errors": {
      "validation_errors": 12,
      "timeout_errors": 3,
      "memory_errors": 1
    }
  }
}
```

### GET /analytics/usage

Get usage analytics and billing information.

**Response:**
```json
{
  "billing_period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "usage_summary": {
    "total_requests": 15420,
    "successful_requests": 15397,
    "failed_requests": 23,
    "total_compute_hours": 245.7,
    "encryption_operations": 8932,
    "decryption_operations": 8875
  },
  "resource_consumption": {
    "cpu_hours": 189.4,
    "gpu_hours": 56.3,
    "memory_gb_hours": 1205.8,
    "storage_gb_hours": 45.2
  },
  "cost_breakdown": {
    "compute_cost": 123.45,
    "storage_cost": 15.67,
    "network_cost": 8.90,
    "total_cost": 148.02,
    "currency": "USD"
  }
}
```

## Error Handling

All API endpoints return standard HTTP status codes and structured error responses.

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid graph data format",
    "details": {
      "field": "features",
      "issue": "Array dimensions must be consistent",
      "provided_shape": [3, 4, 2],
      "expected_shape": "[N, feature_dim]"
    },
    "request_id": "req_abc123def456",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `INVALID_API_KEY` | 401 | API key is missing or invalid |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `CONTEXT_NOT_FOUND` | 404 | Encryption context doesn't exist |
| `VALIDATION_ERROR` | 400 | Request data validation failed |
| `NOISE_BUDGET_EXHAUSTED` | 422 | Insufficient noise budget for operation |
| `MEMORY_LIMIT_EXCEEDED` | 413 | Graph data exceeds memory limits |
| `PROCESSING_TIMEOUT` | 408 | Operation exceeded time limit |
| `ENCRYPTION_FAILED` | 500 | Homomorphic encryption operation failed |
| `GPU_UNAVAILABLE` | 503 | GPU resources temporarily unavailable |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests per time window |

## Rate Limiting

API requests are rate-limited based on your subscription plan:

- **Free Tier**: 100 requests/hour, 10 concurrent contexts
- **Pro Tier**: 1,000 requests/hour, 50 concurrent contexts  
- **Enterprise**: Custom limits based on agreement

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1642248600
X-RateLimit-Retry-After: 3600
```

## SDKs and Libraries

Official SDKs are available for multiple programming languages:

- **Python**: `pip install hegraph-client`
- **JavaScript/Node.js**: `npm install hegraph-client`
- **Java**: Maven/Gradle dependency available
- **Go**: `go get github.com/terragon/hegraph-go`
- **Rust**: `cargo add hegraph-client`

## Support

For API support and questions:

- **Documentation**: https://docs.he-graph.terragon.ai
- **Support Email**: api-support@terragon.ai
- **Community Forum**: https://community.terragon.ai
- **Status Page**: https://status.he-graph.terragon.ai