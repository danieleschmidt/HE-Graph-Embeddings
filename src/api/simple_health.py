"""
Simple health check functions that don't require external dependencies.
"""

from datetime import datetime

def health_check():
    """
    Standalone health check function.
    
    Returns:
        dict: Basic health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "he-graph-embeddings",
        "version": "1.0.0"
    }

class BaseModel:
    """Simple BaseModel fallback"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)