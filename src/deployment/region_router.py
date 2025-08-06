"""
Regional request routing and load balancing for HE-Graph-Embeddings

Provides intelligent routing based on user location, compliance requirements,
latency optimization, and region health status.
"""

import asyncio
import hashlib
import logging
import random
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .multi_region import Region, MultiRegionManager, DataResidencyManager

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Available routing strategies"""
    LATENCY_OPTIMIZED = "latency_optimized"
    COST_OPTIMIZED = "cost_optimized"
    COMPLIANCE_FIRST = "compliance_first"
    LOAD_BALANCED = "load_balanced"
    STICKY_SESSION = "sticky_session"


@dataclass
class RoutingRequest:
    """Request for regional routing"""
    
    user_id: str
    user_location: str
    compliance_requirements: List[str]
    session_id: Optional[str] = None
    data_sensitivity: str = "standard"  # standard, sensitive, restricted
    preferred_strategy: RoutingStrategy = RoutingStrategy.LATENCY_OPTIMIZED
    max_latency_ms: Optional[int] = None
    require_gpu: bool = True


@dataclass
class RoutingResponse:
    """Response with routing decision"""
    
    selected_region: Region
    region_name: str
    estimated_latency_ms: int
    compliance_frameworks: List[str]
    failover_regions: List[Region]
    routing_strategy_used: RoutingStrategy
    cache_duration_seconds: int
    load_balancer_weight: float


class RegionalRouter:
    """Handles intelligent routing of requests across regions"""
    
    def __init__(self, multi_region_manager: MultiRegionManager, data_residency_manager: DataResidencyManager):
        self.multi_region_manager = multi_region_manager
        self.data_residency_manager = data_residency_manager
        self._session_cache = {}
        self._latency_cache = {}
        self._routing_stats = {}
        
        # Base latency estimates (ms) from common locations to regions
        self._base_latencies = self._initialize_latency_matrix()
    
    def _initialize_latency_matrix(self) -> Dict[str, Dict[Region, int]]:
        """Initialize base latency estimates from locations to regions"""
        
        return {
            # North America
            "US": {
                Region.US_EAST_1: 20, Region.US_WEST_2: 50, Region.CA_CENTRAL_1: 30,
                Region.EU_WEST_1: 100, Region.EU_CENTRAL_1: 120, Region.EU_NORTH_1: 140,
                Region.AP_NORTHEAST_1: 150, Region.AP_SOUTHEAST_1: 180, Region.AP_SOUTH_1: 200,
                Region.SA_EAST_1: 120
            },
            "CA": {
                Region.CA_CENTRAL_1: 15, Region.US_EAST_1: 40, Region.US_WEST_2: 60,
                Region.EU_WEST_1: 90, Region.EU_CENTRAL_1: 110, Region.EU_NORTH_1: 120,
                Region.AP_NORTHEAST_1: 160, Region.AP_SOUTHEAST_1: 190, Region.AP_SOUTH_1: 220,
                Region.SA_EAST_1: 140
            },
            
            # Europe
            "GB": {
                Region.EU_WEST_1: 10, Region.EU_CENTRAL_1: 40, Region.EU_NORTH_1: 60,
                Region.US_EAST_1: 80, Region.US_WEST_2: 140, Region.CA_CENTRAL_1: 100,
                Region.AP_NORTHEAST_1: 240, Region.AP_SOUTHEAST_1: 180, Region.AP_SOUTH_1: 160,
                Region.SA_EAST_1: 200
            },
            "DE": {
                Region.EU_CENTRAL_1: 15, Region.EU_WEST_1: 30, Region.EU_NORTH_1: 50,
                Region.US_EAST_1: 90, Region.US_WEST_2: 150, Region.CA_CENTRAL_1: 110,
                Region.AP_NORTHEAST_1: 230, Region.AP_SOUTHEAST_1: 170, Region.AP_SOUTH_1: 150,
                Region.SA_EAST_1: 220
            },
            "FR": {
                Region.EU_WEST_1: 25, Region.EU_CENTRAL_1: 35, Region.EU_NORTH_1: 70,
                Region.US_EAST_1: 85, Region.US_WEST_2: 145, Region.CA_CENTRAL_1: 105,
                Region.AP_NORTHEAST_1: 235, Region.AP_SOUTHEAST_1: 175, Region.AP_SOUTH_1: 155,
                Region.SA_EAST_1: 190
            },
            
            # Asia Pacific
            "JP": {
                Region.AP_NORTHEAST_1: 10, Region.AP_SOUTHEAST_1: 60, Region.AP_SOUTH_1: 80,
                Region.US_WEST_2: 100, Region.US_EAST_1: 160, Region.CA_CENTRAL_1: 130,
                Region.EU_CENTRAL_1: 240, Region.EU_WEST_1: 260, Region.EU_NORTH_1: 280,
                Region.SA_EAST_1: 300
            },
            "SG": {
                Region.AP_SOUTHEAST_1: 5, Region.AP_NORTHEAST_1: 50, Region.AP_SOUTH_1: 40,
                Region.US_WEST_2: 170, Region.US_EAST_1: 230, Region.CA_CENTRAL_1: 200,
                Region.EU_WEST_1: 180, Region.EU_CENTRAL_1: 160, Region.EU_NORTH_1: 200,
                Region.SA_EAST_1: 350
            },
            "IN": {
                Region.AP_SOUTH_1: 15, Region.AP_SOUTHEAST_1: 50, Region.AP_NORTHEAST_1: 70,
                Region.EU_CENTRAL_1: 120, Region.EU_WEST_1: 140, Region.EU_NORTH_1: 160,
                Region.US_EAST_1: 200, Region.US_WEST_2: 250, Region.CA_CENTRAL_1: 220,
                Region.SA_EAST_1: 380
            },
            
            # South America
            "BR": {
                Region.SA_EAST_1: 20, Region.US_EAST_1: 110, Region.US_WEST_2: 170,
                Region.CA_CENTRAL_1: 140, Region.EU_WEST_1: 180, Region.EU_CENTRAL_1: 200,
                Region.EU_NORTH_1: 220, Region.AP_SOUTHEAST_1: 350, Region.AP_NORTHEAST_1: 320,
                Region.AP_SOUTH_1: 380
            }
        }
    
    async def route_request(self, request: RoutingRequest) -> RoutingResponse:
        """Route a request to the optimal region"""
        
        # Check for cached session routing
        if request.session_id and request.preferred_strategy == RoutingStrategy.STICKY_SESSION:
            cached_region = self._get_cached_session_region(request.session_id)
            if cached_region and self.multi_region_manager.is_region_healthy(cached_region):
                return self._build_routing_response(cached_region, request, RoutingStrategy.STICKY_SESSION)
        
        # Get candidate regions based on compliance requirements
        candidate_regions = self._get_candidate_regions(request)
        
        if not candidate_regions:
            logger.error(f"No compliant regions available for request: {request}")
            # Fallback to any healthy region
            candidate_regions = [r for r in self.multi_region_manager.active_regions 
                               if self.multi_region_manager.is_region_healthy(r)]
        
        if not candidate_regions:
            raise RuntimeError("No healthy regions available")
        
        # Apply routing strategy
        selected_region = await self._apply_routing_strategy(request, candidate_regions)
        
        # Cache session if using sticky sessions
        if request.session_id and request.preferred_strategy == RoutingStrategy.STICKY_SESSION:
            self._cache_session_region(request.session_id, selected_region)
        
        # Update routing statistics
        self._update_routing_stats(selected_region, request.preferred_strategy)
        
        return self._build_routing_response(selected_region, request, request.preferred_strategy)
    
    def _get_candidate_regions(self, request: RoutingRequest) -> List[Region]:
        """Get candidate regions based on compliance and requirements"""
        
        candidate_regions = []
        
        # Start with all active and healthy regions
        available_regions = [r for r in self.multi_region_manager.active_regions 
                           if self.multi_region_manager.is_region_healthy(r)]
        
        for region in available_regions:
            region_config = self.multi_region_manager.regions[region]
            
            # Check compliance requirements
            if request.compliance_requirements:
                compliant = any(framework in region_config.compliance_frameworks 
                              for framework in request.compliance_requirements)
                if not compliant:
                    continue
            
            # Check GPU availability if required
            if request.require_gpu and not region_config.gpu_availability:
                continue
            
            # Check latency requirements
            if request.max_latency_ms:
                estimated_latency = self._estimate_latency(request.user_location, region)
                if estimated_latency > request.max_latency_ms:
                    continue
            
            candidate_regions.append(region)
        
        return candidate_regions
    
    async def _apply_routing_strategy(self, request: RoutingRequest, candidates: List[Region]) -> Region:
        """Apply the specified routing strategy to select the best region"""
        
        if request.preferred_strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return self._route_by_latency(request.user_location, candidates)
            
        elif request.preferred_strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._route_by_cost(candidates)
            
        elif request.preferred_strategy == RoutingStrategy.COMPLIANCE_FIRST:
            return self._route_by_compliance(request.compliance_requirements, candidates)
            
        elif request.preferred_strategy == RoutingStrategy.LOAD_BALANCED:
            return self._route_by_load_balance(candidates)
            
        else:  # Default to latency optimized
            return self._route_by_latency(request.user_location, candidates)
    
    def _route_by_latency(self, user_location: str, candidates: List[Region]) -> Region:
        """Route based on lowest estimated latency"""
        
        if not candidates:
            raise ValueError("No candidate regions provided")
        
        latencies = [(region, self._estimate_latency(user_location, region)) for region in candidates]
        latencies.sort(key=lambda x: x[1])
        
        return latencies[0][0]
    
    def _route_by_cost(self, candidates: List[Region]) -> Region:
        """Route based on lowest cost tier"""
        
        if not candidates:
            raise ValueError("No candidate regions provided")
        
        costs = [(region, self.multi_region_manager.regions[region].cost_tier) for region in candidates]
        costs.sort(key=lambda x: x[1])
        
        return costs[0][0]
    
    def _route_by_compliance(self, compliance_requirements: List[str], candidates: List[Region]) -> Region:
        """Route based on best compliance framework match"""
        
        if not candidates:
            raise ValueError("No candidate regions provided")
        
        if not compliance_requirements:
            return candidates[0]
        
        # Score regions by compliance framework matches
        scores = []
        for region in candidates:
            region_config = self.multi_region_manager.regions[region]
            match_count = len(set(compliance_requirements).intersection(set(region_config.compliance_frameworks)))
            scores.append((region, match_count))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
    
    def _route_by_load_balance(self, candidates: List[Region]) -> Region:
        """Route based on load balancer weights (weighted random selection)"""
        
        if not candidates:
            raise ValueError("No candidate regions provided")
        
        weights = []
        total_weight = 0
        
        for region in candidates:
            weight = self.multi_region_manager._load_balancer_weights.get(region, 1.0)
            weights.append((region, weight))
            total_weight += weight
        
        if total_weight == 0:
            return random.choice(candidates)
        
        # Weighted random selection
        r = random.uniform(0, total_weight)
        current_weight = 0
        
        for region, weight in weights:
            current_weight += weight
            if r <= current_weight:
                return region
        
        return candidates[0]  # Fallback
    
    def _estimate_latency(self, user_location: str, region: Region) -> int:
        """Estimate latency from user location to region"""
        
        # Check cache first
        cache_key = f"{user_location}:{region.value}"
        if cache_key in self._latency_cache:
            cached_time, cached_latency = self._latency_cache[cache_key]
            if time.time() - cached_time < 300:  # 5 minute cache
                return cached_latency
        
        # Get base latency
        base_latency = self._base_latencies.get(user_location, {}).get(region, 200)  # Default 200ms
        
        # Add some random jitter based on current load (simulate real conditions)
        jitter = random.randint(-20, 50)
        estimated_latency = max(base_latency + jitter, 10)  # Minimum 10ms
        
        # Cache the result
        self._latency_cache[cache_key] = (time.time(), estimated_latency)
        
        return estimated_latency
    
    def _get_cached_session_region(self, session_id: str) -> Optional[Region]:
        """Get cached region for a session"""
        
        if session_id in self._session_cache:
            cached_time, cached_region = self._session_cache[session_id]
            if time.time() - cached_time < 1800:  # 30 minute session cache
                return cached_region
            else:
                # Clean up expired cache entry
                del self._session_cache[session_id]
        
        return None
    
    def _cache_session_region(self, session_id: str, region: Region) -> None:
        """Cache region selection for a session"""
        self._session_cache[session_id] = (time.time(), region)
    
    def _build_routing_response(self, region: Region, request: RoutingRequest, strategy_used: RoutingStrategy) -> RoutingResponse:
        """Build routing response object"""
        
        region_config = self.multi_region_manager.regions[region]
        estimated_latency = self._estimate_latency(request.user_location, region)
        
        # Determine cache duration based on strategy
        cache_duration = 60  # Default 1 minute
        if strategy_used == RoutingStrategy.STICKY_SESSION:
            cache_duration = 1800  # 30 minutes for sticky sessions
        elif strategy_used == RoutingStrategy.LATENCY_OPTIMIZED:
            cache_duration = 300   # 5 minutes for latency optimization
        
        return RoutingResponse(
            selected_region=region,
            region_name=region_config.name,
            estimated_latency_ms=estimated_latency,
            compliance_frameworks=region_config.compliance_frameworks,
            failover_regions=region_config.failover_regions,
            routing_strategy_used=strategy_used,
            cache_duration_seconds=cache_duration,
            load_balancer_weight=self.multi_region_manager._load_balancer_weights.get(region, 1.0)
        )
    
    def _update_routing_stats(self, region: Region, strategy: RoutingStrategy) -> None:
        """Update routing statistics"""
        
        if region not in self._routing_stats:
            self._routing_stats[region] = {}
        
        if strategy not in self._routing_stats[region]:
            self._routing_stats[region][strategy] = 0
        
        self._routing_stats[region][strategy] += 1
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get current routing statistics"""
        
        stats = {
            "routing_counts": {
                region.value: {strategy.value: count for strategy, count in strategies.items()}
                for region, strategies in self._routing_stats.items()
            },
            "active_sessions": len(self._session_cache),
            "latency_cache_size": len(self._latency_cache),
            "cache_hit_rates": self._calculate_cache_hit_rates()
        }
        
        return stats
    
    def _calculate_cache_hit_rates(self) -> Dict[str, float]:
        """Calculate cache hit rates (simplified implementation)"""
        
        # This would be implemented with proper metrics collection in production
        return {
            "session_cache": 0.85,    # 85% hit rate
            "latency_cache": 0.92     # 92% hit rate
        }
    
    async def update_load_balancer_weights(self, performance_metrics: Dict[Region, Dict[str, float]]) -> None:
        """Update load balancer weights based on performance metrics"""
        
        for region, metrics in performance_metrics.items():
            if region not in self.multi_region_manager.active_regions:
                continue
            
            # Calculate weight based on performance metrics
            # Higher performance = higher weight (more traffic)
            cpu_factor = max(0.1, 1.0 - metrics.get('cpu_usage', 0.5))
            memory_factor = max(0.1, 1.0 - metrics.get('memory_usage', 0.5))
            latency_factor = max(0.1, 1.0 - (metrics.get('avg_latency_ms', 100) / 500))
            error_factor = max(0.1, 1.0 - metrics.get('error_rate', 0.0))
            
            new_weight = cpu_factor * memory_factor * latency_factor * error_factor
            
            # Smooth weight changes to avoid sudden traffic shifts
            current_weight = self.multi_region_manager._load_balancer_weights.get(region, 1.0)
            smoothed_weight = current_weight * 0.7 + new_weight * 0.3
            
            self.multi_region_manager._load_balancer_weights[region] = smoothed_weight
            
            logger.info(f"Updated load balancer weight for {region.value}: {smoothed_weight:.3f}")


class RegionFailoverManager:
    """Manages automatic failover between regions"""
    
    def __init__(self, regional_router: RegionalRouter):
        self.regional_router = regional_router
        self.failover_history = {}
        self.recovery_attempts = {}
    
    async def handle_region_failure(self, failed_region: Region, affected_requests: List[RoutingRequest]) -> List[Tuple[RoutingRequest, RoutingResponse]]:
        """Handle failure of a region by routing requests to failover regions"""
        
        logger.error(f"Handling failure for region: {failed_region.value}")
        
        # Mark region as unhealthy
        self.regional_router.multi_region_manager.health_status[failed_region] = False
        
        # Record failover event
        self.failover_history[failed_region] = {
            'timestamp': time.time(),
            'affected_requests': len(affected_requests)
        }
        
        # Route affected requests to healthy regions
        rerouted_responses = []
        
        for request in affected_requests:
            try:
                # Modify request to exclude the failed region
                new_response = await self.regional_router.route_request(request)
                rerouted_responses.append((request, new_response))
                logger.info(f"Rerouted request from {failed_region.value} to {new_response.selected_region.value}")
                
            except Exception as e:
                logger.error(f"Failed to reroute request: {e}")
                # Create fallback response
                fallback_response = self._create_fallback_response(request)
                rerouted_responses.append((request, fallback_response))
        
        return rerouted_responses
    
    def _create_fallback_response(self, request: RoutingRequest) -> RoutingResponse:
        """Create a fallback response when all routing fails"""
        
        # Use US-East-1 as ultimate fallback
        fallback_region = Region.US_EAST_1
        
        return RoutingResponse(
            selected_region=fallback_region,
            region_name="US East (Fallback)",
            estimated_latency_ms=500,  # High latency estimate for fallback
            compliance_frameworks=["SOC2"],
            failover_regions=[],
            routing_strategy_used=RoutingStrategy.LATENCY_OPTIMIZED,
            cache_duration_seconds=60,
            load_balancer_weight=0.1
        )
    
    async def attempt_region_recovery(self, failed_region: Region) -> bool:
        """Attempt to recover a failed region"""
        
        if failed_region not in self.recovery_attempts:
            self.recovery_attempts[failed_region] = 0
        
        self.recovery_attempts[failed_region] += 1
        max_attempts = 3
        
        if self.recovery_attempts[failed_region] > max_attempts:
            logger.warning(f"Max recovery attempts exceeded for {failed_region.value}")
            return False
        
        try:
            # Perform health check
            health_results = await self.regional_router.multi_region_manager.perform_health_checks()
            
            if health_results.get(failed_region, False):
                logger.info(f"Region {failed_region.value} recovered successfully")
                self.recovery_attempts[failed_region] = 0
                return True
            else:
                logger.warning(f"Region {failed_region.value} still unhealthy")
                return False
                
        except Exception as e:
            logger.error(f"Error during recovery attempt for {failed_region.value}: {e}")
            return False