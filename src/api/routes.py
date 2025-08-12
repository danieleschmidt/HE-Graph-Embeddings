"""
FastAPI routes for HE-Graph-Embeddings API server
"""


from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import torch
import asyncio
import logging
from datetime import datetime
import uvicorn

from ..python.he_graph import CKKSContext, HEConfig, HEGraphSAGE, HEGAT
from .middleware import AuthMiddleware, ValidationMiddleware, ErrorHandlerMiddleware
from .validators import GraphDataValidator, ModelConfigValidator
from .models import GraphRequest, ModelTrainingRequest, InferenceRequest, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HE-Graph-Embeddings API",
    description="Privacy-preserving graph neural networks using homomorphic encryption",
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middlewares
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(ValidationMiddleware)
app.add_middleware(AuthMiddleware)

# Global context manager
contexts: Dict[str, CKKSContext] = {}
models: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting HE-Graph-Embeddings API server")

    # Initialize default CKKS context
    default_config = HEConfig()
    contexts["default"] = CKKSContext(default_config)
    contexts["default"].generate_keys()

    logger.info("Default CKKS context initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down HE-Graph-Embeddings API server")
    contexts.clear()
    models.clear()

# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.3.0",
        gpu_available=torch.cuda.is_available(),
        active_contexts=len(contexts),
        active_models=len(models)
    )

@app.get("/health/detailed")
async def detailed_health():
    """Detailed health information"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_cached": torch.cuda.memory_reserved()
        }

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "gpu_info": gpu_info,
        "contexts": {name: {
            "poly_degree": ctx.config.poly_modulus_degree,
            "security_level": ctx.config.security_level
        } for name, ctx in contexts.items()},
        "models": list(models.keys())
    }

# Context management endpoints
@app.post("/contexts/{context_name}")
async def create_context(context_name: str, config: HEConfig):
    """Create new CKKS context"""
    try:
        context = CKKSContext(config)
        context.generate_keys()
        contexts[context_name] = context

        logger.info(f"Created context: {context_name}")
        return {"message": f"Context {context_name} created successfully"}

    except Exception as e:
        logger.error(f"Failed to create context {context_name}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/contexts")
async def list_contexts():
    """List all available contexts"""
    return {
        "contexts": {
            name: {
                "poly_degree": ctx.config.poly_modulus_degree,
                "security_level": ctx.config.security_level,
                "scale": ctx.config.scale
            }
            for name, ctx in contexts.items()
        }
    }

@app.delete("/contexts/{context_name}")
async def delete_context(context_name: str):
    """Delete a context"""
    if context_name == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default context")

    if context_name not in contexts:
        raise HTTPException(status_code=404, detail="Context not found")

    del contexts[context_name]
    logger.info(f"Deleted context: {context_name}")
    return {"message": f"Context {context_name} deleted"}

# Model management endpoints
@app.post("/models/{model_name}")
async def create_model(
    model_name: str,
    request: ModelTrainingRequest,
    context_name: str = "default"
):
    """Create and train a new model"""
    if context_name not in contexts:
        raise HTTPException(status_code=404, detail="Context not found")

    context = contexts[context_name]

    try:
        if request.model_type == "graphsage":
            model = HEGraphSAGE(
                in_channels=request.in_channels,
                hidden_channels=request.hidden_channels,
                out_channels=request.out_channels,
                num_layers=request.num_layers,
                aggregator=request.aggregator or "mean",
                context=context
            )
        elif request.model_type == "gat":
            model = HEGAT(
                in_channels=request.in_channels,
                out_channels=request.out_channels,
                heads=request.heads or 1,
                attention_type=request.attention_type or "additive",
                context=context
            )
        else:
            raise ValueError(f"Unsupported model type: {request.model_type}")

        models[model_name] = {
            "model": model,
            "context_name": context_name,
            "config": request,
            "created_at": datetime.utcnow()
        }

        logger.info(f"Created model: {model_name}")
        return {"message": f"Model {model_name} created successfully"}

    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
async def list_models():
    """List all available models"""
    return {
        "models": {
            name: {
                "model_type": info["config"].model_type,
                "context_name": info["context_name"],
                "created_at": info["created_at"]
            }
            for name, info in models.items()
        }
    }

@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    del models[model_name]
    logger.info(f"Deleted model: {model_name}")
    return {"message": f"Model {model_name} deleted"}

# Encryption endpoints
@app.post("/encrypt")
async def encrypt_data(
    request: GraphRequest,
    context_name: str = "default"
):
    """Encrypt graph data"""
    if context_name not in contexts:
        raise HTTPException(status_code=404, detail="Context not found")

    context = contexts[context_name]

    try:
        # Convert input to tensors
        features = torch.tensor(request.node_features, dtype=torch.float32)
        edge_index = torch.tensor(request.edge_index, dtype=torch.long)

        # Encrypt features
        encrypted_features = context.encrypt(features)

        # Store encrypted data (in production, use proper storage)
        encryption_id = f"enc_{datetime.utcnow().isoformat()}"

        return {
            "encryption_id": encryption_id,
            "encrypted_shape": features.shape,
            "num_edges": edge_index.shape[1],
            "scale": encrypted_features.scale,
            "noise_budget": encrypted_features.noise_budget
        }

    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Inference endpoints
@app.post("/inference/{model_name}")
async def run_inference(
    model_name: str,
    request: InferenceRequest,
    background_tasks: BackgroundTasks
):
    """Run inference on encrypted data"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    model_info = models[model_name]
    model = model_info["model"]
    context = contexts[model_info["context_name"]]

    try:
        # Convert input data
        features = torch.tensor(request.node_features, dtype=torch.float32)
        edge_index = torch.tensor(request.edge_index, dtype=torch.long)

        # Encrypt input
        encrypted_features = context.encrypt(features)

        # Run inference
        with torch.no_grad():
            encrypted_output = model(encrypted_features, edge_index)

        # Store results (in production, use proper storage)
        inference_id = f"inf_{datetime.utcnow().isoformat()}"

        # Background task for cleanup
        background_tasks.add_task(cleanup_inference_data, inference_id)

        return {
            "inference_id": inference_id,
            "output_shape": encrypted_output.c0.shape,
            "noise_budget": encrypted_output.noise_budget,
            "processing_time": "computed_in_background"
        }

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/decrypt/{encryption_id}")
async def decrypt_data(
    encryption_id: str,
    context_name: str = "default"
):
    """Decrypt data (for testing/verification only)"""
    if context_name not in contexts:
        raise HTTPException(status_code=404, detail="Context not found")

    # In production, this would retrieve stored encrypted data
    # This is a simplified example
    return {
        "message": "Decryption endpoint - implement with proper data storage",
        "encryption_id": encryption_id,
        "warning": "This endpoint should only be used for testing"
    }

# Batch processing endpoints
@app.post("/batch/encrypt")
async def encrypt_batch(
    requests: List[GraphRequest],
    context_name: str = "default"
):
    """Encrypt multiple graphs in batch"""
    if context_name not in contexts:
        raise HTTPException(status_code=404, detail="Context not found")

    context = contexts[context_name]
    results = []

    try:
        for i, request in enumerate(requests):
            features = torch.tensor(request.node_features, dtype=torch.float32)
            encrypted_features = context.encrypt(features)

            results.append({
                "batch_index": i,
                "encryption_id": f"batch_enc_{i}_{datetime.utcnow().isoformat()}",
                "encrypted_shape": features.shape,
                "noise_budget": encrypted_features.noise_budget
            })

        return {"batch_results": results}

    except Exception as e:
        logger.error(f"Batch encryption failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Utility endpoints
@app.get("/security/estimate")
async def estimate_security(config: HEConfig):
    """Estimate security level for given parameters"""
    from ..python.he_graph import SecurityEstimator

    try:
        params = {
            "poly_degree": config.poly_modulus_degree,
            "coeff_modulus_bits": config.coeff_modulus_bits
        }

        security_bits = SecurityEstimator.estimate(params)

        return {
            "security_bits": security_bits,
            "parameters": params,
            "recommendation": "acceptable" if security_bits >= 128 else "insufficient"
        }

    except Exception as e:
        logger.error(f"Error in operation: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/performance/benchmark")
async def run_benchmark():
    """Run performance benchmark"""
    try:
        # Simple benchmark
        config = HEConfig(poly_modulus_degree=16384)  # Smaller for quick test
        context = CKKSContext(config)
        context.generate_keys()

        # Benchmark encryption

        import time
        data = torch.randn(100, 64)

        start_time = time.time()
        encrypted = context.encrypt(data)
        encrypt_time = time.time() - start_time

        # Benchmark addition
        start_time = time.time()
        result = context.add(encrypted, encrypted)
        add_time = time.time() - start_time

        return {
            "benchmark_results": {
                "data_size": data.shape,
                "encryption_time_ms": encrypt_time * 1000,
                "addition_time_ms": add_time * 1000,
                "noise_budget": result.noise_budget
            },
            "configuration": {
                "poly_degree": config.poly_modulus_degree,
                "security_level": config.security_level
            }
        }

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def cleanup_inference_data(inference_id: str):
    """Cleanup inference data after processing"""
    # In production, implement proper cleanup
    logger.info(f"Cleaning up inference data: {inference_id}")
    await asyncio.sleep(1)  # Simulate cleanup work

# Main server entry point
if __name__ == "__main__":
    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )