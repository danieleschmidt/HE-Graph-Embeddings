"""
Custom middleware for HE-Graph-Embeddings API
"""

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse
import time
import logging
import json
from typing import Callable
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware"""
    
    PROTECTED_PATHS = [
        "/models",
        "/contexts", 
        "/decrypt",
        "/batch"
    ]
    
    EXCLUDED_PATHS = [
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json"
    ]
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip auth for excluded paths
        if any(request.url.path.startswith(path) for path in self.EXCLUDED_PATHS):
            return await call_next(request)
        
        # For now, implement basic API key authentication
        # In production, use proper JWT/OAuth
        api_key = request.headers.get("X-API-Key")
        
        if any(request.url.path.startswith(path) for path in self.PROTECTED_PATHS):
            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "AuthenticationError",
                        "message": "API key required for this endpoint",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            # Validate API key (simplified - use proper validation in production)
            if api_key != "dev-key-12345":  # Development key
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "AuthenticationError", 
                        "message": "Invalid API key",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
        
        # Add user context to request if authenticated
        request.state.user_id = "dev-user" if api_key else "anonymous"
        request.state.authenticated = bool(api_key)
        
        return await call_next(request)

class ValidationMiddleware(BaseHTTPMiddleware):
    """Input validation and sanitization middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length:
            content_length = int(content_length)
            max_size = 100 * 1024 * 1024  # 100MB max request size
            
            if content_length > max_size:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "PayloadTooLarge",
                        "message": f"Request size {content_length} exceeds maximum {max_size} bytes",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                if not request.url.path.startswith("/docs"):
                    return JSONResponse(
                        status_code=415,
                        content={
                            "error": "UnsupportedMediaType",
                            "message": "Content-Type must be application/json",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
        
        # Process request
        response = await call_next(request)
        
        # Add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        logger.info(f"Response: {response.status_code} ({process_time:.3f}s)")
        
        return response

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        try:
            return await call_next(request)
        
        except HTTPException as e:
            # Let FastAPI handle HTTP exceptions
            raise e
        
        except ValueError as e:
            logger.error(f"ValueError in {request.url.path}: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "ValueError",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        except MemoryError as e:
            logger.error(f"MemoryError in {request.url.path}: {e}")
            return JSONResponse(
                status_code=507,
                content={
                    "error": "InsufficientStorage",
                    "message": "Insufficient memory to process request",
                    "details": {"suggestion": "Try reducing batch size or model complexity"},
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        except Exception as e:
            # Log full traceback for debugging
            logger.error(f"Unhandled exception in {request.url.path}: {e}\n{traceback.format_exc()}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "details": {
                        "error_type": type(e).__name__,
                        "error_message": str(e) if str(e) else "No error message available"
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}  # In production, use Redis or similar
        self.window_size = 60  # 1 minute window
    
    async def dispatch(self, request: Request, call_next: Callable):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - self.window_size
        self.request_counts = {
            ip: [(ts, path) for ts, path in requests if ts > cutoff_time]
            for ip, requests in self.request_counts.items()
            if any(ts > cutoff_time for ts, path in requests)
        }
        
        # Count requests from this IP
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        recent_requests = len(self.request_counts[client_ip])
        
        if recent_requests >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Rate limit exceeded: {recent_requests}/{self.requests_per_minute} requests per minute",
                    "retry_after": 60,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={"Retry-After": "60"}
            )
        
        # Record this request
        self.request_counts[client_ip].append((current_time, request.url.path))
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - recent_requests - 1)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_size))
        
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Remove server information
        if "server" in response.headers:
            del response.headers["server"]
        
        return response

class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with additional security"""
    
    def __init__(self, app, allowed_origins: list = None, allowed_methods: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["http://localhost:3000", "http://localhost:8080"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    async def dispatch(self, request: Request, call_next: Callable):
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = PlainTextResponse("OK", status_code=200)
        else:
            response = await call_next(request)
        
        # Add CORS headers
        if origin in self.allowed_origins or "*" in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Max-Age"] = "3600"
        
        return response

class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect metrics for monitoring"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.request_duration_sum = 0.0
        self.error_count = 0
        
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            self.request_count += 1
            self.request_duration_sum += duration
            
            if response.status_code >= 400:
                self.error_count += 1
            
            # Add metrics headers for debugging
            response.headers["X-Request-Count"] = str(self.request_count)
            response.headers["X-Average-Duration"] = f"{self.request_duration_sum / self.request_count:.3f}"
            response.headers["X-Error-Rate"] = f"{self.error_count / self.request_count:.3f}"
            
            return response
            
        except Exception as e:
            self.error_count += 1
            raise e
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        if self.request_count == 0:
            return {
                "request_count": 0,
                "average_duration": 0,
                "error_rate": 0
            }
        
        return {
            "request_count": self.request_count,
            "average_duration": self.request_duration_sum / self.request_count,
            "error_rate": self.error_count / self.request_count,
            "total_errors": self.error_count
        }