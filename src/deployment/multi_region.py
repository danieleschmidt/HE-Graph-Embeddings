"""
Multi-region deployment configuration and management for HE-Graph-Embeddings

Provides region-aware deployment configurations, data residency compliance,
latency optimization, and cross-region failover capabilities.
"""


import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp
import torch

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported deployment regions with compliance mappings"""

    # North America
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    CA_CENTRAL_1 = "ca-central-1"

    # Europe (GDPR compliant)
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    EU_NORTH_1 = "eu-north-1"

    # Asia Pacific
    AP_NORTHEAST_1 = "ap-northeast-1"  # Tokyo
    AP_SOUTHEAST_1 = "ap-southeast-1"  # Singapore
    AP_SOUTH_1 = "ap-south-1"         # Mumbai

    # Other regions
    SA_EAST_1 = "sa-east-1"           # São Paulo


@dataclass
class RegionConfig:
    """Configuration for a specific deployment region"""

    region: Region
    name: str
    compliance_frameworks: List[str]
    data_residency: bool
    encryption_required: bool
    gpu_availability: bool
    latency_tier: int  # 1=lowest latency, 5=highest latency
    cost_tier: int     # 1=lowest cost, 5=highest cost
    supported_languages: List[str]
    health_check_endpoint: str
    failover_regions: List[Region]
    max_nodes: int
    storage_encryption: str
    network_encryption: str


class MultiRegionManager:
    """Manages multi-region deployment configurations and routing"""

    def __init__(self):
        """  Init  ."""
        self.regions = self._initialize_regions()
        self.active_regions = set()
        self.health_status = {}
        self._load_balancer_weights = {}

    def _initialize_regions(self) -> Dict[Region, RegionConfig]:
        """Initialize all supported regions with their configurations"""

        return {
            Region.US_EAST_1: RegionConfig(
                region=Region.US_EAST_1,
                name="US East (N. Virginia)",
                compliance_frameworks=["CCPA", "HIPAA", "SOC2"],
                data_residency=False,
                encryption_required=True,
                gpu_availability=True,
                latency_tier=1,
                cost_tier=2,
                supported_languages=["en", "es", "fr"],
                health_check_endpoint="/health/us-east-1",
                failover_regions=[Region.US_WEST_2, Region.CA_CENTRAL_1],
                max_nodes=50,
                storage_encryption="AES-256-GCM",
                network_encryption="TLS-1.3"
            ),

            Region.US_WEST_2: RegionConfig(
                region=Region.US_WEST_2,
                name="US West (Oregon)",
                compliance_frameworks=["CCPA", "HIPAA", "SOC2"],
                data_residency=False,
                encryption_required=True,
                gpu_availability=True,
                latency_tier=2,
                cost_tier=2,
                supported_languages=["en", "es", "fr"],
                health_check_endpoint="/health/us-west-2",
                failover_regions=[Region.US_EAST_1, Region.CA_CENTRAL_1],
                max_nodes=30,
                storage_encryption="AES-256-GCM",
                network_encryption="TLS-1.3"
            ),

            Region.CA_CENTRAL_1: RegionConfig(
                region=Region.CA_CENTRAL_1,
                name="Canada (Central)",
                compliance_frameworks=["PIPEDA", "PHIPA"],
                data_residency=True,
                encryption_required=True,
                gpu_availability=True,
                latency_tier=2,
                cost_tier=3,
                supported_languages=["en", "fr"],
                health_check_endpoint="/health/ca-central-1",
                failover_regions=[Region.US_EAST_1, Region.US_WEST_2],
                max_nodes=20,
                storage_encryption="AES-256-GCM",
                network_encryption="TLS-1.3"
            ),

            Region.EU_WEST_1: RegionConfig(
                region=Region.EU_WEST_1,
                name="Europe (Ireland)",
                compliance_frameworks=["GDPR", "ISO27001"],
                data_residency=True,
                encryption_required=True,
                gpu_availability=True,
                latency_tier=1,
                cost_tier=3,
                supported_languages=["en", "de", "fr", "es", "it"],
                health_check_endpoint="/health/eu-west-1",
                failover_regions=[Region.EU_CENTRAL_1, Region.EU_NORTH_1],
                max_nodes=40,
                storage_encryption="AES-256-GCM",
                network_encryption="TLS-1.3"
            ),

            Region.EU_CENTRAL_1: RegionConfig(
                region=Region.EU_CENTRAL_1,
                name="Europe (Frankfurt)",
                compliance_frameworks=["GDPR", "BDSG", "ISO27001"],
                data_residency=True,
                encryption_required=True,
                gpu_availability=True,
                latency_tier=1,
                cost_tier=3,
                supported_languages=["de", "en", "fr"],
                health_check_endpoint="/health/eu-central-1",
                failover_regions=[Region.EU_WEST_1, Region.EU_NORTH_1],
                max_nodes=35,
                storage_encryption="AES-256-GCM",
                network_encryption="TLS-1.3"
            ),

            Region.EU_NORTH_1: RegionConfig(
                region=Region.EU_NORTH_1,
                name="Europe (Stockholm)",
                compliance_frameworks=["GDPR", "ISO27001"],
                data_residency=True,
                encryption_required=True,
                gpu_availability=False,
                latency_tier=2,
                cost_tier=4,
                supported_languages=["en", "de", "fr"],
                health_check_endpoint="/health/eu-north-1",
                failover_regions=[Region.EU_WEST_1, Region.EU_CENTRAL_1],
                max_nodes=15,
                storage_encryption="AES-256-GCM",
                network_encryption="TLS-1.3"
            ),

            Region.AP_NORTHEAST_1: RegionConfig(
                region=Region.AP_NORTHEAST_1,
                name="Asia Pacific (Tokyo)",
                compliance_frameworks=["APPI", "ISO27001"],
                data_residency=True,
                encryption_required=True,
                gpu_availability=True,
                latency_tier=1,
                cost_tier=4,
                supported_languages=["ja", "en", "zh"],
                health_check_endpoint="/health/ap-northeast-1",
                failover_regions=[Region.AP_SOUTHEAST_1, Region.AP_SOUTH_1],
                max_nodes=25,
                storage_encryption="AES-256-GCM",
                network_encryption="TLS-1.3"
            ),

            Region.AP_SOUTHEAST_1: RegionConfig(
                region=Region.AP_SOUTHEAST_1,
                name="Asia Pacific (Singapore)",
                compliance_frameworks=["PDPA", "ISO27001"],
                data_residency=True,
                encryption_required=True,
                gpu_availability=True,
                latency_tier=2,
                cost_tier=3,
                supported_languages=["en", "zh", "ja"],
                health_check_endpoint="/health/ap-southeast-1",
                failover_regions=[Region.AP_NORTHEAST_1, Region.AP_SOUTH_1],
                max_nodes=20,
                storage_encryption="AES-256-GCM",
                network_encryption="TLS-1.3"
            ),

            Region.AP_SOUTH_1: RegionConfig(
                region=Region.AP_SOUTH_1,
                name="Asia Pacific (Mumbai)",
                compliance_frameworks=["IT_ACT", "ISO27001"],
                data_residency=True,
                encryption_required=True,
                gpu_availability=True,
                latency_tier=3,
                cost_tier=2,
                supported_languages=["en", "hi"],
                health_check_endpoint="/health/ap-south-1",
                failover_regions=[Region.AP_SOUTHEAST_1, Region.AP_NORTHEAST_1],
                max_nodes=15,
                storage_encryption="AES-256-GCM",
                network_encryption="TLS-1.3"
            ),

            Region.SA_EAST_1: RegionConfig(
                region=Region.SA_EAST_1,
                name="South America (São Paulo)",
                compliance_frameworks=["LGPD", "ISO27001"],
                data_residency=True,
                encryption_required=True,
                gpu_availability=True,
                latency_tier=4,
                cost_tier=4,
                supported_languages=["pt", "es", "en"],
                health_check_endpoint="/health/sa-east-1",
                failover_regions=[Region.US_EAST_1, Region.US_WEST_2],
                max_nodes=10,
                storage_encryption="AES-256-GCM",
                network_encryption="TLS-1.3"
            )
        }

    def get_optimal_region(self, user_location: str, compliance_requirements: List[str] = None) -> RegionConfig:
        """Determine the optimal region based on user location and compliance needs"""

        # Simple geolocation to region mapping
        location_preferences = {
            # North America
            "US": [Region.US_EAST_1, Region.US_WEST_2],
            "CA": [Region.CA_CENTRAL_1, Region.US_EAST_1],
            "MX": [Region.US_WEST_2, Region.US_EAST_1],

            # Europe
            "GB": [Region.EU_WEST_1, Region.EU_CENTRAL_1],
            "DE": [Region.EU_CENTRAL_1, Region.EU_WEST_1],
            "FR": [Region.EU_WEST_1, Region.EU_CENTRAL_1],
            "IT": [Region.EU_WEST_1, Region.EU_CENTRAL_1],
            "ES": [Region.EU_WEST_1, Region.EU_CENTRAL_1],
            "SE": [Region.EU_NORTH_1, Region.EU_CENTRAL_1],

            # Asia Pacific
            "JP": [Region.AP_NORTHEAST_1, Region.AP_SOUTHEAST_1],
            "SG": [Region.AP_SOUTHEAST_1, Region.AP_NORTHEAST_1],
            "IN": [Region.AP_SOUTH_1, Region.AP_SOUTHEAST_1],
            "CN": [Region.AP_SOUTHEAST_1, Region.AP_NORTHEAST_1],
            "KR": [Region.AP_NORTHEAST_1, Region.AP_SOUTHEAST_1],

            # South America
            "BR": [Region.SA_EAST_1, Region.US_EAST_1],
            "AR": [Region.SA_EAST_1, Region.US_EAST_1],
        }

        preferred_regions = location_preferences.get(user_location, [Region.US_EAST_1])

        # Filter by compliance requirements if specified
        if compliance_requirements:
            compliant_regions = []
            for region_enum in preferred_regions:
                region_config = self.regions[region_enum]
                if any(req in region_config.compliance_frameworks for req in compliance_requirements):
                    compliant_regions.append(region_enum)

            if compliant_regions:
                preferred_regions = compliant_regions

        # Filter by active and healthy regions
        available_regions = [r for r in preferred_regions if r in self.active_regions and self.is_region_healthy(r)]

        if available_regions:
            # Return the region with best latency tier among available options
            best_region = min(available_regions, key=lambda r: self.regions[r].latency_tier)
            return self.regions[best_region]

        # Fallback to any healthy region
        for region_enum, region_config in self.regions.items():
            if self.is_region_healthy(region_enum):
                return region_config

        # Ultimate fallback
        return self.regions[Region.US_EAST_1]

    def is_region_healthy(self, region: Region) -> bool:
        """Check if a region is healthy and available"""
        return self.health_status.get(region, False)

    async def perform_health_checks(self) -> Dict[Region, bool]:
        """Perform health checks on all active regions"""

        health_results = {}

        async with aiohttp.ClientSession() as session:
            tasks = []

            for region in self.active_regions:
                task = self._check_region_health(session, region)
                tasks.append(task)

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    region = list(self.active_regions)[i]
                    if isinstance(result, Exception):
                        health_results[region] = False
                        logger.error(f"Health check failed for {region.value}: {result}")
                    else:
                        health_results[region] = result

        self.health_status.update(health_results)
        return health_results

    async def _check_region_health(self, session: aiohttp.ClientSession, region: Region) -> bool:
        """Check health of a specific region"""

        region_config = self.regions[region]

        try:
            # Construct health check URL (this would be your actual service URL)
            base_url = f"https://he-graph-{region.value}.terragon.ai"
            health_url = f"{base_url}{region_config.health_check_endpoint}"

            async with session.get(health_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    # Check various health indicators
                    gpu_healthy = data.get('gpu_available', True) if region_config.gpu_availability else True
                    memory_healthy = data.get('memory_usage_percent', 0) < 90
                    cpu_healthy = data.get('cpu_usage_percent', 0) < 80

                    return gpu_healthy and memory_healthy and cpu_healthy
                else:
                    return False

        except Exception as e:
            logger.error(f"Health check error for {region.value}: {e}")
            return False

    def activate_region(self, region: Region) -> bool:
        """Activate a region for deployment"""

        if region not in self.regions:
            logger.error(f"Unknown region: {region}")
            return False

        self.active_regions.add(region)
        self._load_balancer_weights[region] = 1.0

        logger.info(f"Activated region: {region.value}")
        return True

    def deactivate_region(self, region: Region) -> bool:
        """Deactivate a region"""

        if region in self.active_regions:
            self.active_regions.remove(region)
            self._load_balancer_weights.pop(region, None)
            self.health_status.pop(region, None)

            logger.info(f"Deactivated region: {region.value}")
            return True

        return False

    def get_failover_region(self, failed_region: Region) -> Optional[RegionConfig]:
        """Get the best failover region for a failed region"""

        if failed_region not in self.regions:
            return None

        failed_config = self.regions[failed_region]

        # Try failover regions in order of preference
        for failover_region in failed_config.failover_regions:
            if failover_region in self.active_regions and self.is_region_healthy(failover_region):
                return self.regions[failover_region]

        # If no designated failover regions are available, find any healthy region
        # with similar compliance requirements
        for region_enum, region_config in self.regions.items():
            if (region_enum in self.active_regions and
                self.is_region_healthy(region_enum) and
                set(region_config.compliance_frameworks).intersection(set(failed_config.compliance_frameworks))):
                return region_config

        return None

    def get_deployment_config(self, region: Region) -> Dict[str, Any]:
        """Get deployment configuration for a specific region"""

        if region not in self.regions:
            raise ValueError(f"Unknown region: {region}")

        region_config = self.regions[region]

        return {
            "region": region_config.region.value,
            "name": region_config.name,
            "compliance": {
                "frameworks": region_config.compliance_frameworks,
                "data_residency": region_config.data_residency,
                "encryption_required": region_config.encryption_required
            },
            "infrastructure": {
                "gpu_availability": region_config.gpu_availability,
                "max_nodes": region_config.max_nodes,
                "storage_encryption": region_config.storage_encryption,
                "network_encryption": region_config.network_encryption
            },
            "networking": {
                "latency_tier": region_config.latency_tier,
                "health_check_endpoint": region_config.health_check_endpoint,
                "failover_regions": [r.value for r in region_config.failover_regions]
            },
            "localization": {
                "supported_languages": region_config.supported_languages
            },
            "costs": {
                "cost_tier": region_config.cost_tier
            }
        }

    def export_global_config(self) -> Dict[str, Any]:
        """Export complete multi-region configuration"""

        return {
            "regions": {
                region.value: self.get_deployment_config(region)
                for region in self.regions.keys()
            },
            "active_regions": [r.value for r in self.active_regions],
            "health_status": {r.value: status for r, status in self.health_status.items()},
            "load_balancer_weights": {r.value: weight for r, weight in self._load_balancer_weights.items()},
            "last_updated": datetime.utcnow().isoformat()
        }


class DataResidencyManager:
    """Manages data residency and cross-border data transfer compliance"""

    def __init__(self, multi_region_manager: MultiRegionManager):
        """  Init  ."""
        self.multi_region_manager = multi_region_manager
        self.restricted_transfers = self._initialize_restricted_transfers()

    def _initialize_restricted_transfers(self) -> Dict[str, List[str]]:
        """Initialize restricted data transfer mappings"""

        return {
            "GDPR": ["EU-27", "EEA", "UK"],  # EU data must stay in approved regions
            "CCPA": ["US"],                  # California data preferences
            "PIPEDA": ["CA"],               # Canadian data residency
            "LGPD": ["BR"],                 # Brazilian data residency
            "PIPL": ["CN"],                 # Chinese data residency
            "APPI": ["JP"],                 # Japanese data protection
            "PDPA": ["SG", "TH", "MY"]      # Southeast Asian data protection
        }

    def can_transfer_data(self, from_region: Region, to_region: Region, compliance_frameworks: List[str]) -> bool:
        """Check if data can be transferred between regions under compliance frameworks"""

        from_config = self.multi_region_manager.regions[from_region]
        to_config = self.multi_region_manager.regions[to_region]

        # Check each compliance framework
        for framework in compliance_frameworks:
            if framework in self.restricted_transfers:
                # Check if both regions are compliant with this framework
                if (framework in from_config.compliance_frameworks and
                    framework in to_config.compliance_frameworks):
                    continue
                else:
                    logger.warning(f"Data transfer blocked by {framework}: {from_region.value} -> {to_region.value}")
                    return False

        return True

    def get_compliant_regions(self, compliance_frameworks: List[str]) -> List[Region]:
        """Get all regions that comply with the specified frameworks"""

        compliant_regions = []

        for region, config in self.multi_region_manager.regions.items():
            if all(framework in config.compliance_frameworks for framework in compliance_frameworks):
                compliant_regions.append(region)

        return compliant_regions


# Global instance for easy access
multi_region_manager = MultiRegionManager()
data_residency_manager = DataResidencyManager(multi_region_manager)