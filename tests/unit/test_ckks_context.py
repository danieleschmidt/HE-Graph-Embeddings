"""
Unit tests for CKKS context and homomorphic operations
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import warnings

from src.python.he_graph import CKKSContext, HEConfig, EncryptedTensor, SecurityLevel
from src.python.he_graph import SecurityEstimator, NoiseTracker

class TestHEConfig:
    """Test HE configuration validation"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = HEConfig()
        assert config.poly_modulus_degree == 32768
        assert config.coeff_modulus_bits == [60, 40, 40, 40, 40, 60]
        assert config.scale == 2**40
        assert config.security_level == 128
        assert config.precision_bits == 30
    
    def test_config_validation_valid_params(self):
        """Test validation with valid parameters"""
        config = HEConfig(
            poly_modulus_degree=16384,
            coeff_modulus_bits=[60, 40, 40, 60],
            scale=2**35,
            security_level=128
        )
        assert config.validate() == True
    
    def test_config_validation_invalid_poly_degree(self):
        """Test validation fails for non-power-of-2 polynomial degree"""
        config = HEConfig(poly_modulus_degree=12345)
        with pytest.raises(ValueError, match="power of 2"):
            config.validate()
    
    def test_config_validation_security_warning(self):
        """Test security warning for large coefficient modulus"""
        config = HEConfig(
            coeff_modulus_bits=[60, 60, 60, 60, 60, 60, 60, 60]  # Total > 438 bits
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config.validate()
            assert len(w) == 1
            assert "security limit" in str(w[0].message)

class TestCKKSContext:
    """Test CKKS context operations"""
    
    @pytest.fixture
    def context(self):
        """Create test CKKS context"""
        config = HEConfig(
            poly_modulus_degree=8192,  # Smaller for faster tests
            coeff_modulus_bits=[60, 40, 40, 60],
            scale=2**35
        )
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    @pytest.fixture
    def gpu_context(self):
        """Create GPU-enabled test context"""
        config = HEConfig(poly_modulus_degree=8192)
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value.total_memory = 16 * 1024**3  # 16GB
            ctx = CKKSContext(config, gpu_id=0)
            ctx.generate_keys()
            return ctx
    
    def test_context_initialization(self, context):
        """Test context initialization"""
        assert context.config.poly_modulus_degree == 8192
        assert context.device.type == 'cpu'
        assert context._secret_key is not None
        assert context._public_key is not None
    
    def test_key_generation(self, context):
        """Test key generation"""
        assert context._secret_key is not None
        assert context._public_key is not None
        assert context._relin_keys is not None
        assert context._galois_keys is not None
        
        # Check key dimensions
        assert len(context._secret_key) == context.config.poly_modulus_degree
        assert len(context._public_key) == 2
    
    def test_encryption_decryption(self, context):
        """Test basic encryption/decryption"""
        # Test data
        plaintext = torch.randn(10, 5)
        
        # Encrypt
        encrypted = context.encrypt(plaintext)
        assert isinstance(encrypted, EncryptedTensor)
        assert encrypted.scale == context.config.scale
        
        # Decrypt
        decrypted = context.decrypt(encrypted)
        
        # Check approximate equality (due to noise)
        torch.testing.assert_close(plaintext, decrypted, rtol=1e-2, atol=1e-2)
    
    def test_homomorphic_addition(self, context):
        """Test homomorphic addition"""
        a = torch.randn(5, 3)
        b = torch.randn(5, 3)
        
        # Encrypt
        enc_a = context.encrypt(a)
        enc_b = context.encrypt(b)
        
        # Homomorphic addition
        enc_result = context.add(enc_a, enc_b)
        
        # Decrypt and verify
        result = context.decrypt(enc_result)
        expected = a + b
        
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
    
    def test_homomorphic_multiplication(self, context):
        """Test homomorphic multiplication"""
        a = torch.randn(3, 3) * 0.1  # Small values to avoid overflow
        b = torch.randn(3, 3) * 0.1
        
        # Encrypt
        enc_a = context.encrypt(a)
        enc_b = context.encrypt(b)
        
        # Homomorphic multiplication
        enc_result = context.multiply(enc_a, enc_b)
        
        # Decrypt and verify
        result = context.decrypt(enc_result)
        expected = a * b
        
        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)
    
    def test_rotation(self, context):
        """Test homomorphic rotation"""
        data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        
        # Encrypt
        encrypted = context.encrypt(data)
        
        # Rotate
        rotated = context.rotate(encrypted, 2)
        
        # Decrypt
        result = context.decrypt(rotated)
        
        # Check that rotation occurred (exact verification depends on implementation)
        assert result.shape == data.shape
        assert not torch.equal(result, data)
    
    def test_rescaling(self, context):
        """Test rescaling operation"""
        data = torch.randn(3, 3)
        encrypted = context.encrypt(data)
        
        original_scale = encrypted.scale
        rescaled = context.rescale(encrypted)
        
        # Scale should be reduced
        assert rescaled.scale < original_scale
        
        # Data should still decrypt correctly (approximately)
        decrypted = context.decrypt(rescaled)
        torch.testing.assert_close(data, decrypted, rtol=1e-1, atol=1e-1)
    
    def test_noise_budget_tracking(self, context):
        """Test noise budget tracking"""
        data = torch.randn(5, 5)
        encrypted = context.encrypt(data)
        
        initial_noise = encrypted.noise_budget
        assert initial_noise > 0
        
        # Multiplication reduces noise budget
        multiplied = context.multiply(encrypted, encrypted)
        assert multiplied.noise_budget < initial_noise
    
    def test_batch_encryption(self, context):
        """Test batch encryption"""
        batch_data = [torch.randn(3, 2) for _ in range(5)]
        
        # Batch encrypt (simulate batch encryption)
        encrypted_batch = [context.encrypt(data) for data in batch_data]
        
        assert len(encrypted_batch) == 5
        for enc in encrypted_batch:
            assert isinstance(enc, EncryptedTensor)
    
    def test_context_from_config(self):
        """Test creating context from configuration"""
        config = HEConfig(poly_modulus_degree=4096)
        
        with patch('torch.cuda.is_available', return_value=False):
            context = CKKSContext.from_config(config)
            assert context.config.poly_modulus_degree == 4096
    
    def test_invalid_operations(self, context):
        """Test invalid operations raise appropriate errors"""
        a = torch.randn(3, 3)
        b = torch.randn(3, 3)
        
        enc_a = context.encrypt(a)
        enc_b = context.encrypt(b)
        
        # Modify scale to create mismatch
        enc_b.scale = enc_a.scale * 2
        
        # Addition with mismatched scales should fail
        with pytest.raises(ValueError, match="Scales must match"):
            context.add(enc_a, enc_b)
    
    def test_gpu_memory_setup(self, gpu_context):
        """Test GPU memory pool setup"""
        assert gpu_context.device.type == 'cuda'
        # Memory setup is mocked, just verify it doesn't crash

class TestEncryptedTensor:
    """Test EncryptedTensor operations"""
    
    @pytest.fixture
    def context(self):
        """Create test context"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    def test_encrypted_tensor_creation(self, context):
        """Test EncryptedTensor creation"""
        data = torch.randn(5, 3)
        encrypted = context.encrypt(data)
        
        assert encrypted.c0.shape == data.shape
        assert encrypted.c1.shape == data.shape
        assert encrypted.scale == context.config.scale
        assert encrypted.context == context
    
    def test_encrypted_tensor_operations(self, context):
        """Test EncryptedTensor operator overloading"""
        a = torch.randn(3, 3) * 0.1
        b = torch.randn(3, 3) * 0.1
        
        enc_a = context.encrypt(a)
        enc_b = context.encrypt(b)
        
        # Test __add__
        enc_sum = enc_a + enc_b
        result_sum = context.decrypt(enc_sum)
        torch.testing.assert_close(result_sum, a + b, rtol=1e-2, atol=1e-2)
        
        # Test __mul__
        enc_prod = enc_a * enc_b
        result_prod = context.decrypt(enc_prod)
        torch.testing.assert_close(result_prod, a * b, rtol=1e-1, atol=1e-1)
    
    def test_noise_budget_property(self, context):
        """Test noise budget property"""
        data = torch.randn(3, 3)
        encrypted = context.encrypt(data)
        
        noise_budget = encrypted.noise_budget
        assert isinstance(noise_budget, float)
        assert noise_budget > 0
    
    def test_rescale_method(self, context):
        """Test rescale method"""
        data = torch.randn(3, 3)
        encrypted = context.encrypt(data)
        
        original_scale = encrypted.scale
        rescaled = encrypted.rescale()
        
        assert rescaled.scale < original_scale
        assert rescaled.context == context

class TestSecurityEstimator:
    """Test security estimation utilities"""
    
    def test_security_estimation_128bit(self):
        """Test 128-bit security estimation"""
        params = {
            'poly_degree': 32768,
            'coeff_modulus_bits': [60, 40, 40, 40, 40, 60]
        }
        
        security_bits = SecurityEstimator.estimate(params)
        assert security_bits == 128
    
    def test_security_estimation_insufficient(self):
        """Test insufficient security detection"""
        params = {
            'poly_degree': 1024,  # Too small
            'coeff_modulus_bits': [60, 60, 60, 60, 60, 60, 60]  # Too large
        }
        
        security_bits = SecurityEstimator.estimate(params)
        assert security_bits == 0  # Insufficient
    
    def test_parameter_recommendation(self):
        """Test parameter recommendation"""
        config = SecurityEstimator.recommend(
            security_bits=128,
            multiplicative_depth=5,
            precision_bits=30
        )
        
        assert isinstance(config, HEConfig)
        assert config.security_level == 128
        assert config.poly_modulus_degree >= 16384
    
    def test_unsupported_security_level(self):
        """Test unsupported security level raises error"""
        with pytest.raises(ValueError, match="Unsupported security level"):
            SecurityEstimator.recommend(security_bits=64)

class TestNoiseTracker:
    """Test noise budget tracking"""
    
    @pytest.fixture
    def context(self):
        """Create test context"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    def test_noise_tracker_context_manager(self, context):
        """Test NoiseTracker as context manager"""
        data = torch.randn(3, 3)
        encrypted = context.encrypt(data)
        
        with NoiseTracker() as tracker:
            tracker.update(encrypted)
            noise_budget = tracker.get_noise_budget()
            
            assert isinstance(noise_budget, float)
            assert noise_budget > 0
    
    def test_noise_history_tracking(self, context):
        """Test noise history tracking"""
        tracker = NoiseTracker()
        
        data = torch.randn(3, 3)
        encrypted = context.encrypt(data)
        
        tracker.update(encrypted)
        initial_noise = tracker.get_noise_budget()
        
        # Perform operation that reduces noise
        multiplied = context.multiply(encrypted, encrypted)
        tracker.update(multiplied)
        
        final_noise = tracker.get_noise_budget()
        
        assert len(tracker.noise_history) == 2
        assert final_noise < initial_noise

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_encryption_without_keys(self):
        """Test encryption fails without keys"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            context = CKKSContext(config)
            # Don't generate keys
            
            data = torch.randn(3, 3)
            with pytest.raises(RuntimeError, match="Public key not generated"):
                context.encrypt(data)
    
    def test_decryption_without_secret_key(self):
        """Test decryption fails without secret key"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            context = CKKSContext(config)
            context.generate_keys()
            
            # Mock missing secret key
            context._secret_key = None
            
            data = torch.randn(3, 3)
            encrypted = EncryptedTensor(
                torch.randn_like(data), torch.randn_like(data),
                context.config.scale, context
            )
            
            with pytest.raises(RuntimeError, match="Secret key not available"):
                context.decrypt(encrypted)
    
    def test_cuda_unavailable_warning(self):
        """Test warning when CUDA is unavailable"""
        config = HEConfig()
        
        with patch('torch.cuda.is_available', return_value=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                context = CKKSContext(config)
                
                assert len(w) == 1
                assert "CUDA not available" in str(w[0].message)
                assert context.device.type == 'cpu'
    
    def test_invalid_rotation_steps(self, context):
        """Test rotation with invalid steps"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            context = CKKSContext(config)
            context.generate_keys()
            
            data = torch.randn(3, 3)
            encrypted = context.encrypt(data)
            
            # Use steps not in galois keys
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = context.rotate(encrypted, 7)  # Not a power of 2
                
                assert len(w) == 1
                assert "Using rotation by" in str(w[0].message)

@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios"""
    
    @pytest.fixture
    def context(self):
        """Create integration test context"""
        config = HEConfig(poly_modulus_degree=8192)
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    def test_matrix_multiplication_workflow(self, context):
        """Test matrix multiplication workflow"""
        # Simulate matrix multiplication A * B
        A = torch.randn(5, 3) * 0.1
        B = torch.randn(3, 4) * 0.1
        
        # Encrypt matrices
        enc_A = context.encrypt(A)
        enc_B = context.encrypt(B)
        
        # Simplified matrix multiplication (element-wise for test)
        # Real implementation would need proper matrix operations
        enc_result = enc_A * enc_B[:3, :3]  # Adjust shapes for test
        
        # Decrypt result
        result = context.decrypt(enc_result)
        
        # Verify operation completed without errors
        assert result.shape == (5, 3)
    
    def test_deep_computation_chain(self, context):
        """Test chain of homomorphic operations"""
        data = torch.randn(3, 3) * 0.01  # Very small values
        encrypted = context.encrypt(data)
        
        # Chain of operations
        result = encrypted
        for i in range(3):  # Limited depth to avoid noise exhaustion
            result = result + encrypted
            if result.noise_budget < 5:  # Bootstrap if needed
                result = context.bootstrap(result)
        
        # Should complete without errors
        final_result = context.decrypt(result)
        assert final_result.shape == data.shape
    
    def test_batch_processing_workflow(self, context):
        """Test batch processing workflow"""
        batch_size = 5
        data_batch = [torch.randn(2, 3) * 0.1 for _ in range(batch_size)]
        
        # Encrypt batch
        encrypted_batch = [context.encrypt(data) for data in data_batch]
        
        # Process batch (simple addition)
        processed_batch = []
        for i in range(batch_size):
            if i == 0:
                processed_batch.append(encrypted_batch[i])
            else:
                processed_batch.append(encrypted_batch[i] + encrypted_batch[0])
        
        # Decrypt results
        results = [context.decrypt(enc) for enc in processed_batch]
        
        assert len(results) == batch_size
        for result in results:
            assert result.shape == (2, 3)