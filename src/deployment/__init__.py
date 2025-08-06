"""
Multi-region deployment package for HE-Graph-Embeddings

Provides global deployment capabilities with intelligent routing, 
compliance management, and failover functionality.
"""

from .multi_region import (
    Region,
    RegionConfig, 
    MultiRegionManager,
    DataResidencyManager,
    multi_region_manager,
    data_residency_manager
)

from .region_router import (
    RoutingStrategy,
    RoutingRequest,
    RoutingResponse,
    RegionalRouter,
    RegionFailoverManager
)

__all__ = [
    'Region',
    'RegionConfig',
    'MultiRegionManager', 
    'DataResidencyManager',
    'multi_region_manager',
    'data_residency_manager',
    'RoutingStrategy',
    'RoutingRequest', 
    'RoutingResponse',
    'RegionalRouter',
    'RegionFailoverManager'
]