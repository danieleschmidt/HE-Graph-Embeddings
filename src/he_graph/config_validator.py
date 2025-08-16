"""
🛡️ GENERATION 2: Enhanced Configuration Validation

This module provides comprehensive validation for HE-Graph-Embeddings configurations,
ensuring security, performance, and compatibility across different deployment scenarios.

Key Features:
- Multi-level configuration validation (security, performance, compatibility)
- Environment-specific recommendations
- Automatic parameter optimization
- Compatibility checking across different hardware/software configurations
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import json

from .robust_error_handling import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SecurityLevel(Enum):
    """Security level requirements"""
    MINIMAL = "minimal"      # 80-bit security
    STANDARD = "standard"    # 128-bit security
    HIGH = "high"           # 192-bit security
    MAXIMUM = "maximum"     # 256-bit security


class PerformanceProfile(Enum):
    """Performance optimization profiles"""
    MEMORY_OPTIMIZED = "memory_optimized"
    SPEED_OPTIMIZED = "speed_optimized"
    BALANCED = "balanced"
    PRECISION_OPTIMIZED = "precision_optimized"


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    optimized_config: Optional[Dict[str, Any]] = None


@dataclass
class HEConfigProfile:
    """Configuration profile for different scenarios"""
    poly_modulus_degree: int
    coeff_modulus_bits: List[int]
    scale_bits: int
    security_level: int
    max_depth: int
    precision_bits: int
    
    @property
    def scale(self) -> float:
        return 2.0 ** self.scale_bits
    
    @property
    def total_modulus_bits(self) -> int:
        return sum(self.coeff_modulus_bits)


class EnhancedConfigValidator:
    """
    Comprehensive configuration validator with security, performance,
    and compatibility checking
    """
    
    # Predefined secure configurations
    SECURE_PROFILES = {
        SecurityLevel.MINIMAL: HEConfigProfile(
            poly_modulus_degree=8192,
            coeff_modulus_bits=[40, 30, 30, 40],
            scale_bits=30,
            security_level=80,
            max_depth=5,
            precision_bits=25
        ),
        SecurityLevel.STANDARD: HEConfigProfile(
            poly_modulus_degree=16384,
            coeff_modulus_bits=[50, 40, 40, 40, 50],
            scale_bits=40,
            security_level=128,
            max_depth=8,
            precision_bits=30
        ),
        SecurityLevel.HIGH: HEConfigProfile(
            poly_modulus_degree=32768,
            coeff_modulus_bits=[60, 50, 50, 50, 50, 60],
            scale_bits=50,
            security_level=192,
            max_depth=12,
            precision_bits=35
        ),
        SecurityLevel.MAXIMUM: HEConfigProfile(
            poly_modulus_degree=65536,
            coeff_modulus_bits=[60, 60, 60, 60, 60, 60, 60],
            scale_bits=60,
            security_level=256,
            max_depth=15,
            precision_bits=40
        )
    }
    
    # Environment-specific requirements
    ENVIRONMENT_REQUIREMENTS = {
        DeploymentEnvironment.DEVELOPMENT: {
            "min_security_level": 80,
            "max_computation_time": 60.0,  # seconds
            "memory_limit_gb": 16
        },
        DeploymentEnvironment.TESTING: {
            "min_security_level": 128,
            "max_computation_time": 30.0,
            "memory_limit_gb": 32
        },
        DeploymentEnvironment.STAGING: {
            "min_security_level": 128,
            "max_computation_time": 10.0,
            "memory_limit_gb": 64
        },
        DeploymentEnvironment.PRODUCTION: {
            "min_security_level": 128,
            "max_computation_time": 5.0,
            "memory_limit_gb": 128
        }
    }
    
    def __init__(self):
        self.validation_cache = {}  # Cache validation results
        logger.info("Enhanced config validator initialized")
    
    def validate_he_config(self, 
                          config: Dict[str, Any],
                          environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION,
                          performance_profile: PerformanceProfile = PerformanceProfile.BALANCED) -> ValidationResult:
        """
        Comprehensive validation of HE configuration
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Basic parameter validation
            self._validate_basic_parameters(config, result)
            
            # Security validation
            self._validate_security_parameters(config, environment, result)
            
            # Performance validation
            self._validate_performance_parameters(config, performance_profile, result)
            
            # Compatibility validation
            self._validate_compatibility(config, result)
            
            # Generate optimization recommendations
            self._generate_optimization_recommendations(config, environment, performance_profile, result)
            
            # Generate optimized configuration if requested
            if result.warnings or result.recommendations:
                result.optimized_config = self._optimize_configuration(
                    config, environment, performance_profile
                )
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation failed with unexpected error: {e}")
            logger.error(f"Config validation error: {e}")
        
        return result
    
    def _validate_basic_parameters(self, config: Dict[str, Any], result: ValidationResult):
        """Validate basic parameter structure and types"""
        required_params = [
            'poly_modulus_degree', 'coeff_modulus_bits', 'scale', 'security_level'
        ]
        
        for param in required_params:
            if param not in config:
                result.errors.append(f"Missing required parameter: {param}")
                result.is_valid = False
                continue
        
        if not result.is_valid:
            return  # Skip further validation if basic structure is wrong
        
        # Validate polynomial degree
        poly_degree = config['poly_modulus_degree']
        if not isinstance(poly_degree, int) or poly_degree <= 0:
            result.errors.append("poly_modulus_degree must be a positive integer")
            result.is_valid = False
        elif poly_degree & (poly_degree - 1) != 0:
            result.errors.append("poly_modulus_degree must be a power of 2")
            result.is_valid = False
        elif poly_degree < 1024:
            result.errors.append("poly_modulus_degree too small (minimum 1024)")
            result.is_valid = False
        elif poly_degree > 131072:
            result.warnings.append("Very large poly_modulus_degree may cause performance issues")
        
        # Validate coefficient modulus
        coeff_bits = config['coeff_modulus_bits']
        if not isinstance(coeff_bits, list) or len(coeff_bits) < 2:
            result.errors.append("coeff_modulus_bits must be a list with at least 2 elements")
            result.is_valid = False
        else:
            for i, bits in enumerate(coeff_bits):
                if not isinstance(bits, int) or bits <= 0:
                    result.errors.append(f"coeff_modulus_bits[{i}] must be a positive integer")
                    result.is_valid = False
                elif bits < 20 or bits > 60:
                    result.warnings.append(f"coeff_modulus_bits[{i}] = {bits} is outside recommended range [20, 60]")
        
        # Validate scale
        scale = config['scale']
        if isinstance(scale, (int, float)) and scale <= 0:
            result.errors.append("scale must be positive")
            result.is_valid = False
        elif isinstance(scale, (int, float)):
            scale_bits = math.log2(scale)
            if scale_bits != int(scale_bits):
                result.warnings.append("scale should be a power of 2 for optimal performance")
            if scale_bits < 20 or scale_bits > 60:
                result.warnings.append(f"scale bits ({scale_bits:.1f}) outside recommended range [20, 60]")
    
    def _validate_security_parameters(self, config: Dict[str, Any], 
                                    environment: DeploymentEnvironment, 
                                    result: ValidationResult):
        """Validate security-related parameters"""
        if not result.is_valid:
            return  # Skip if basic validation failed
        
        security_level = config.get('security_level', 128)
        env_requirements = self.ENVIRONMENT_REQUIREMENTS[environment]
        min_security = env_requirements['min_security_level']
        
        if security_level < min_security:
            result.errors.append(
                f"Security level {security_level} below minimum {min_security} "
                f"for {environment.value} environment"
            )
            result.is_valid = False
        
        # Estimate actual security level based on parameters
        estimated_security = self._estimate_security_level(config)
        
        if estimated_security < security_level:
            result.warnings.append(
                f"Parameters provide only ~{estimated_security}-bit security, "
                f"less than claimed {security_level}-bit"
            )
        
        if estimated_security < min_security:
            result.errors.append(
                f"Parameter combination provides insufficient security "
                f"({estimated_security}-bit < {min_security}-bit required)"
            )
            result.is_valid = False
    
    def _validate_performance_parameters(self, config: Dict[str, Any],
                                       profile: PerformanceProfile,
                                       result: ValidationResult):
        """Validate performance-related parameters"""
        if not result.is_valid:
            return
        
        poly_degree = config['poly_modulus_degree']
        coeff_bits = config['coeff_modulus_bits']
        total_bits = sum(coeff_bits)
        
        # Estimate computational complexity
        complexity_score = self._estimate_complexity(config)
        
        if profile == PerformanceProfile.SPEED_OPTIMIZED:
            if poly_degree > 16384:
                result.warnings.append(
                    "Large polynomial degree may slow speed-optimized operations"
                )
            if total_bits > 400:
                result.warnings.append(
                    "High total modulus bits may impact speed optimization"
                )
        
        elif profile == PerformanceProfile.MEMORY_OPTIMIZED:
            if poly_degree > 32768:
                result.warnings.append(
                    "Very large polynomial degree increases memory usage significantly"
                )
            memory_estimate = self._estimate_memory_usage(config)
            if memory_estimate > 8.0:  # GB
                result.warnings.append(
                    f"Estimated memory usage ({memory_estimate:.1f} GB) may be too high "
                    "for memory-optimized profile"
                )
        
        elif profile == PerformanceProfile.PRECISION_OPTIMIZED:
            scale_bits = math.log2(config['scale']) if config['scale'] > 0 else 0
            if scale_bits < 40:
                result.recommendations.append(
                    "Consider increasing scale for better precision in precision-optimized profile"
                )
    
    def _validate_compatibility(self, config: Dict[str, Any], result: ValidationResult):
        """Validate compatibility with different hardware/software configurations"""
        if not result.is_valid:
            return
        
        poly_degree = config['poly_modulus_degree']
        
        # Check GPU compatibility
        if poly_degree > 32768:
            result.warnings.append(
                "Very large polynomial degrees may not be supported on all GPU configurations"
            )
        
        # Check for known problematic parameter combinations
        coeff_bits = config['coeff_modulus_bits']
        if len(coeff_bits) > 10:
            result.warnings.append(
                "Large number of coefficient modulus primes may cause compatibility issues"
            )
        
        # Check total modulus size
        total_bits = sum(coeff_bits)
        max_safe_bits = self._get_max_safe_modulus_bits(poly_degree)
        
        if total_bits > max_safe_bits:
            result.errors.append(
                f"Total modulus bits ({total_bits}) exceeds safe limit ({max_safe_bits}) "
                f"for polynomial degree {poly_degree}"
            )
            result.is_valid = False
    
    def _generate_optimization_recommendations(self, config: Dict[str, Any],
                                             environment: DeploymentEnvironment,
                                             profile: PerformanceProfile,
                                             result: ValidationResult):
        """Generate optimization recommendations"""
        if not result.is_valid:
            return
        
        recommendations = []
        
        # Security optimizations
        estimated_security = self._estimate_security_level(config)
        target_security = self.ENVIRONMENT_REQUIREMENTS[environment]['min_security_level']
        
        if estimated_security > target_security + 50:
            recommendations.append(
                "Parameters provide excessive security - consider reducing for better performance"
            )
        
        # Performance optimizations
        if profile == PerformanceProfile.SPEED_OPTIMIZED:
            complexity = self._estimate_complexity(config)
            if complexity > 1000:  # Arbitrary threshold
                recommendations.append(
                    "Consider reducing polynomial degree or modulus size for speed optimization"
                )
        
        # Profile-specific recommendations
        if profile == PerformanceProfile.BALANCED:
            poly_degree = config['poly_modulus_degree']
            if poly_degree not in [8192, 16384, 32768]:
                recommendations.append(
                    "Consider using standard polynomial degrees (8192, 16384, 32768) "
                    "for better library support"
                )
        
        result.recommendations.extend(recommendations)
    
    def _optimize_configuration(self, config: Dict[str, Any],
                              environment: DeploymentEnvironment,
                              profile: PerformanceProfile) -> Dict[str, Any]:
        """Generate optimized configuration"""
        # Start with the closest secure profile
        target_security = self.ENVIRONMENT_REQUIREMENTS[environment]['min_security_level']
        
        if target_security >= 256:
            base_profile = self.SECURE_PROFILES[SecurityLevel.MAXIMUM]
        elif target_security >= 192:
            base_profile = self.SECURE_PROFILES[SecurityLevel.HIGH]
        elif target_security >= 128:
            base_profile = self.SECURE_PROFILES[SecurityLevel.STANDARD]
        else:
            base_profile = self.SECURE_PROFILES[SecurityLevel.MINIMAL]
        
        optimized = {
            'poly_modulus_degree': base_profile.poly_modulus_degree,
            'coeff_modulus_bits': base_profile.coeff_modulus_bits.copy(),
            'scale': base_profile.scale,
            'security_level': base_profile.security_level,
            'precision_bits': base_profile.precision_bits
        }
        
        # Apply profile-specific optimizations
        if profile == PerformanceProfile.SPEED_OPTIMIZED:
            # Prefer smaller parameters
            if optimized['poly_modulus_degree'] > 16384:
                optimized['poly_modulus_degree'] = 16384
                optimized['coeff_modulus_bits'] = [50, 40, 40, 50]
        
        elif profile == PerformanceProfile.MEMORY_OPTIMIZED:
            # Minimize memory usage
            optimized['poly_modulus_degree'] = min(optimized['poly_modulus_degree'], 16384)
        
        elif profile == PerformanceProfile.PRECISION_OPTIMIZED:
            # Maximize precision
            optimized['scale'] = 2.0 ** 50
            optimized['precision_bits'] = 40
        
        return optimized
    
    def _estimate_security_level(self, config: Dict[str, Any]) -> int:
        """Estimate actual security level from parameters"""
        poly_degree = config['poly_modulus_degree']
        total_bits = sum(config['coeff_modulus_bits'])
        
        # Simplified security estimation (real implementation would be more complex)
        if poly_degree >= 32768 and total_bits <= 600:
            return 128
        elif poly_degree >= 16384 and total_bits <= 400:
            return 128
        elif poly_degree >= 8192 and total_bits <= 300:
            return 80
        else:
            return max(64, 80 - (total_bits - 300) // 20)
    
    def _estimate_complexity(self, config: Dict[str, Any]) -> float:
        """Estimate computational complexity"""
        poly_degree = config['poly_modulus_degree']
        num_primes = len(config['coeff_modulus_bits'])
        
        # Simplified complexity estimation
        return poly_degree * math.log2(poly_degree) * num_primes
    
    def _estimate_memory_usage(self, config: Dict[str, Any]) -> float:
        """Estimate memory usage in GB"""
        poly_degree = config['poly_modulus_degree']
        num_primes = len(config['coeff_modulus_bits'])
        
        # Simplified memory estimation (bytes per ciphertext * expected ciphertexts)
        bytes_per_ciphertext = poly_degree * num_primes * 8  # 8 bytes per coefficient
        expected_ciphertexts = 100  # Estimate
        
        return (bytes_per_ciphertext * expected_ciphertexts) / (1024**3)  # Convert to GB
    
    def _get_max_safe_modulus_bits(self, poly_degree: int) -> int:
        """Get maximum safe total modulus bits for given polynomial degree"""
        # Conservative estimates based on security literature
        if poly_degree >= 32768:
            return 600
        elif poly_degree >= 16384:
            return 400
        elif poly_degree >= 8192:
            return 300
        else:
            return 200
    
    def get_recommended_config(self, 
                             environment: DeploymentEnvironment,
                             performance_profile: PerformanceProfile = PerformanceProfile.BALANCED,
                             security_level: SecurityLevel = SecurityLevel.STANDARD) -> Dict[str, Any]:
        """Get recommended configuration for specific requirements"""
        
        base_profile = self.SECURE_PROFILES[security_level]
        
        config = {
            'poly_modulus_degree': base_profile.poly_modulus_degree,
            'coeff_modulus_bits': base_profile.coeff_modulus_bits.copy(),
            'scale': base_profile.scale,
            'security_level': base_profile.security_level,
            'precision_bits': base_profile.precision_bits,
            'bootstrap_threshold': 10,
            'auto_mod_switch': True,
            'environment': environment.value,
            'performance_profile': performance_profile.value
        }
        
        # Validate and optimize
        validation_result = self.validate_he_config(config, environment, performance_profile)
        
        if validation_result.optimized_config:
            config.update(validation_result.optimized_config)
        
        return config


# Testing function
def test_config_validation():
    """Test configuration validation functionality"""
    logger.info("Testing configuration validation...")
    
    validator = EnhancedConfigValidator()
    
    # Test valid configuration
    valid_config = {
        'poly_modulus_degree': 16384,
        'coeff_modulus_bits': [50, 40, 40, 50],
        'scale': 2**40,
        'security_level': 128
    }
    
    result = validator.validate_he_config(
        valid_config, 
        DeploymentEnvironment.PRODUCTION,
        PerformanceProfile.BALANCED
    )
    
    logger.info(f"✅ Valid config validation: {'PASSED' if result.is_valid else 'FAILED'}")
    if result.warnings:
        logger.info(f"⚠️  Warnings: {len(result.warnings)}")
    if result.recommendations:
        logger.info(f"💡 Recommendations: {len(result.recommendations)}")
    
    # Test invalid configuration
    invalid_config = {
        'poly_modulus_degree': 1000,  # Not power of 2
        'coeff_modulus_bits': [80],   # Too few primes
        'scale': -1,                  # Negative scale
        'security_level': 64          # Too low for production
    }
    
    result = validator.validate_he_config(
        invalid_config,
        DeploymentEnvironment.PRODUCTION
    )
    
    logger.info(f"❌ Invalid config validation: {'FAILED' if not result.is_valid else 'UNEXPECTED PASS'}")
    logger.info(f"📋 Errors found: {len(result.errors)}")
    
    # Test recommended configurations
    for env in DeploymentEnvironment:
        for sec in SecurityLevel:
            try:
                recommended = validator.get_recommended_config(env, security_level=sec)
                logger.info(f"✅ {env.value} + {sec.value}: poly_degree={recommended['poly_modulus_degree']}")
            except Exception as e:
                logger.error(f"❌ Failed to generate config for {env.value} + {sec.value}: {e}")
    
    logger.info("🛡️ Configuration validation test complete!")


if __name__ == "__main__":
    test_config_validation()