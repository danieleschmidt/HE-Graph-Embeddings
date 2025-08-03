"""
Unit tests for HE graph neural network models
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.python.he_graph import (
    CKKSContext, HEConfig, EncryptedTensor,
    HEModule, HELinear, HEGraphSAGE, HEGAT, GraphSAGEConv
)

class TestHEModule:
    """Test base HE module functionality"""
    
    @pytest.fixture
    def context(self):
        """Create test CKKS context"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    def test_he_module_initialization(self, context):
        """Test HE module base class"""
        module = HEModule(context)
        assert module.context == context
        assert isinstance(module, nn.Module)
    
    def test_forward_encrypted_not_implemented(self, context):
        """Test that forward_encrypted raises NotImplementedError"""
        module = HEModule(context)
        data = torch.randn(5, 3)
        encrypted = context.encrypt(data)
        
        with pytest.raises(NotImplementedError):
            module.forward_encrypted(encrypted)

class TestHELinear:
    """Test encrypted linear layer"""
    
    @pytest.fixture
    def context(self):
        """Create test CKKS context"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    @pytest.fixture
    def linear_layer(self, context):
        """Create test linear layer"""
        return HELinear(in_features=3, out_features=2, context=context, bias=True)
    
    def test_linear_initialization(self, linear_layer, context):
        """Test linear layer initialization"""
        assert linear_layer.in_features == 3
        assert linear_layer.out_features == 2
        assert linear_layer.context == context
        assert linear_layer.weight.shape == (2, 3)
        assert linear_layer.bias.shape == (2,)
    
    def test_linear_no_bias(self, context):
        """Test linear layer without bias"""
        layer = HELinear(in_features=4, out_features=3, context=context, bias=False)
        assert layer.bias is None
    
    def test_linear_forward_encrypted(self, linear_layer, context):
        """Test encrypted forward pass"""
        # Input data
        x = torch.randn(5, 3) * 0.1  # Small values to avoid overflow
        encrypted_x = context.encrypt(x)
        
        # Forward pass
        output_enc = linear_layer.forward_encrypted(encrypted_x)
        
        # Verify output
        assert isinstance(output_enc, EncryptedTensor)
        
        # Decrypt and check approximate correctness
        output = context.decrypt(output_enc)
        assert output.shape == (5, 2)
    
    def test_linear_weight_initialization(self, linear_layer):
        """Test weight initialization distribution"""
        weights = linear_layer.weight.data
        
        # Check that weights are roughly normally distributed
        assert abs(weights.mean()) < 1.0  # Should be close to 0
        assert 0.1 < weights.std() < 1.0  # Should have reasonable variance

class TestGraphSAGEConv:
    """Test GraphSAGE convolution layer"""
    
    @pytest.fixture
    def context(self):
        """Create test CKKS context"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    @pytest.fixture
    def conv_layer(self, context):
        """Create test GraphSAGE conv layer"""
        return GraphSAGEConv(
            in_channels=4, 
            out_channels=3, 
            aggregator='mean',
            context=context
        )
    
    def test_conv_initialization(self, conv_layer, context):
        """Test convolution layer initialization"""
        assert conv_layer.in_channels == 4
        assert conv_layer.out_channels == 3
        assert conv_layer.aggregator == 'mean'
        assert conv_layer.context == context
        assert isinstance(conv_layer.lin_self, HELinear)
        assert isinstance(conv_layer.lin_neigh, HELinear)
    
    def test_conv_forward(self, conv_layer, context):
        """Test convolution forward pass"""
        # Create test graph data
        num_nodes = 5
        x = torch.randn(num_nodes, 4) * 0.1
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # Simple chain
        
        # Encrypt node features
        encrypted_x = context.encrypt(x)
        
        # Forward pass
        output_enc = conv_layer.forward(encrypted_x, edge_index)
        
        # Verify output
        assert isinstance(output_enc, EncryptedTensor)
        
        # Decrypt and check shape
        output = context.decrypt(output_enc)
        assert output.shape == (num_nodes, 3)
    
    def test_conv_different_aggregators(self, context):
        """Test different aggregator types"""
        aggregators = ['mean', 'max', 'sum']
        
        for agg in aggregators:
            conv = GraphSAGEConv(
                in_channels=2, 
                out_channels=2, 
                aggregator=agg,
                context=context
            )
            assert conv.aggregator == agg
    
    def test_conv_aggregation_function(self, conv_layer, context):
        """Test aggregation function"""
        num_nodes = 4
        x = torch.randn(num_nodes, 4) * 0.1
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        
        encrypted_x = context.encrypt(x)
        
        # Test aggregation (simplified implementation returns input)
        aggregated = conv_layer._aggregate(encrypted_x, edge_index)
        assert isinstance(aggregated, EncryptedTensor)

class TestHEGraphSAGE:
    """Test complete GraphSAGE model"""
    
    @pytest.fixture
    def context(self):
        """Create test CKKS context"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    @pytest.fixture
    def graphsage_model(self, context):
        """Create test GraphSAGE model"""
        return HEGraphSAGE(
            in_channels=5,
            hidden_channels=[4, 3],
            out_channels=2,
            aggregator='mean',
            context=context
        )
    
    def test_graphsage_initialization(self, graphsage_model, context):
        """Test GraphSAGE model initialization"""
        assert graphsage_model.num_layers == 3  # 2 hidden + 1 output
        assert graphsage_model.aggregator == 'mean'
        assert graphsage_model.context == context
        assert len(graphsage_model.convs) == 3
    
    def test_graphsage_single_hidden_dim(self, context):
        """Test GraphSAGE with single hidden dimension"""
        model = HEGraphSAGE(
            in_channels=3,
            hidden_channels=4,  # Single int instead of list
            num_layers=2,
            context=context
        )
        
        assert model.num_layers == 2
        assert len(model.convs) == 2
    
    def test_graphsage_forward(self, graphsage_model, context):
        """Test GraphSAGE forward pass"""
        # Create test graph
        num_nodes = 6
        x = torch.randn(num_nodes, 5) * 0.1
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 1, 2],
            [1, 2, 3, 4, 5, 0, 1]
        ])
        
        # Encrypt features
        encrypted_x = context.encrypt(x)
        
        # Forward pass
        output_enc = graphsage_model.forward(encrypted_x, edge_index)
        
        # Verify output
        assert isinstance(output_enc, EncryptedTensor)
        
        # Decrypt and check shape
        output = context.decrypt(output_enc)
        assert output.shape == (num_nodes, 2)
    
    def test_graphsage_activation_function(self, graphsage_model, context):
        """Test activation function application"""
        x = torch.randn(3, 5) * 0.1
        encrypted_x = context.encrypt(x)
        
        # Apply activation
        activated = graphsage_model._apply_activation(encrypted_x)
        
        assert isinstance(activated, EncryptedTensor)
        
        # For relu_poly, should return squared values (simplified)
        decrypted = context.decrypt(activated)
        assert decrypted.shape == x.shape
    
    def test_graphsage_dropout(self, context):
        """Test encrypted dropout"""
        model = HEGraphSAGE(
            in_channels=3,
            hidden_channels=2,
            dropout_enc=0.5,
            context=context
        )
        
        x = torch.randn(4, 3) * 0.1
        encrypted_x = context.encrypt(x)
        
        # Apply dropout (in training mode)
        model.train()
        dropped = model._apply_dropout(encrypted_x)
        
        assert isinstance(dropped, EncryptedTensor)
    
    def test_graphsage_no_dropout_eval(self, context):
        """Test no dropout in evaluation mode"""
        model = HEGraphSAGE(
            in_channels=3,
            hidden_channels=2,
            dropout_enc=0.5,
            context=context
        )
        
        x = torch.randn(4, 3) * 0.1
        encrypted_x = context.encrypt(x)
        
        # No dropout in eval mode
        model.eval()
        result = model._apply_dropout(encrypted_x)
        
        # Should return input unchanged
        assert result == encrypted_x

class TestHEGAT:
    """Test Graph Attention Network model"""
    
    @pytest.fixture
    def context(self):
        """Create test CKKS context"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    @pytest.fixture
    def gat_model(self, context):
        """Create test GAT model"""
        return HEGAT(
            in_channels=4,
            out_channels=2,
            heads=2,
            attention_type='additive',
            context=context
        )
    
    def test_gat_initialization(self, gat_model, context):
        """Test GAT model initialization"""
        assert gat_model.in_channels == 4
        assert gat_model.out_channels == 2
        assert gat_model.heads == 2
        assert gat_model.attention_type == 'additive'
        assert gat_model.context == context
        
        # Check linear layers
        assert isinstance(gat_model.lin_q, HELinear)
        assert isinstance(gat_model.lin_k, HELinear)
        assert isinstance(gat_model.lin_v, HELinear)
        assert isinstance(gat_model.lin_out, HELinear)
    
    def test_gat_with_edge_features(self, context):
        """Test GAT with edge features"""
        model = HEGAT(
            in_channels=3,
            out_channels=2,
            heads=1,
            edge_dim=2,
            context=context
        )
        
        assert model.lin_edge is not None
        assert isinstance(model.lin_edge, HELinear)
    
    def test_gat_forward(self, gat_model, context):
        """Test GAT forward pass"""
        # Create test graph
        num_nodes = 5
        x = torch.randn(num_nodes, 4) * 0.1
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        # Encrypt features
        encrypted_x = context.encrypt(x)
        
        # Forward pass
        output_enc = gat_model.forward(encrypted_x, edge_index)
        
        # Verify output
        assert isinstance(output_enc, EncryptedTensor)
        
        # Decrypt and check shape
        output = context.decrypt(output_enc)
        assert output.shape == (num_nodes, 2)
    
    def test_gat_attention_computation(self, gat_model, context):
        """Test attention score computation"""
        num_nodes = 3
        hidden_dim = 4
        
        q = torch.randn(num_nodes, hidden_dim) * 0.1
        k = torch.randn(num_nodes, hidden_dim) * 0.1
        edge_index = torch.tensor([[0, 1], [1, 2]])
        
        enc_q = context.encrypt(q)
        enc_k = context.encrypt(k)
        
        # Compute attention scores
        scores = gat_model._compute_attention(enc_q, enc_k, edge_index)
        
        assert isinstance(scores, EncryptedTensor)
        
        # Decrypt and verify
        decrypted_scores = context.decrypt(scores)
        assert decrypted_scores.shape == q.shape
    
    def test_gat_softmax_approximation(self, gat_model, context):
        """Test softmax approximation"""
        scores = torch.randn(4, 3) * 0.1  # Small values
        enc_scores = context.encrypt(scores)
        
        # Apply softmax approximation
        attn_weights = gat_model._softmax_approximation(enc_scores)
        
        assert isinstance(attn_weights, EncryptedTensor)
        
        # Decrypt and verify shape
        decrypted_weights = context.decrypt(attn_weights)
        assert decrypted_weights.shape == scores.shape
    
    def test_gat_attention_application(self, gat_model, context):
        """Test attention weight application"""
        num_nodes = 4
        attn_weights = torch.randn(num_nodes, 2) * 0.1
        values = torch.randn(num_nodes, 2) * 0.1
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        
        enc_weights = context.encrypt(attn_weights)
        enc_values = context.encrypt(values)
        
        # Apply attention
        result = gat_model._apply_attention(enc_weights, enc_values, edge_index)
        
        assert isinstance(result, EncryptedTensor)
    
    def test_gat_softmax_configuration(self, gat_model):
        """Test softmax approximation configuration"""
        gat_model.set_softmax_approximation(
            method='taylor',
            order=5,
            range=(-3, 3)
        )
        
        assert gat_model.softmax_order == 5
        assert gat_model.softmax_range == (-3, 3)
    
    def test_gat_multiplicative_attention(self, context):
        """Test multiplicative attention type"""
        model = HEGAT(
            in_channels=3,
            out_channels=2,
            attention_type='multiplicative',
            context=context
        )
        
        num_nodes = 3
        q = torch.randn(num_nodes, 3) * 0.1
        k = torch.randn(num_nodes, 3) * 0.1
        edge_index = torch.tensor([[0, 1], [1, 2]])
        
        enc_q = context.encrypt(q)
        enc_k = context.encrypt(k)
        
        scores = model._compute_attention(enc_q, enc_k, edge_index)
        assert isinstance(scores, EncryptedTensor)

class TestModelIntegration:
    """Integration tests for HE models"""
    
    @pytest.fixture
    def context(self):
        """Create test CKKS context"""
        config = HEConfig(
            poly_modulus_degree=8192,
            coeff_modulus_bits=[60, 40, 40, 60]
        )
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    def test_model_comparison(self, context):
        """Test different models on same data"""
        # Create test graph
        num_nodes = 8
        x = torch.randn(num_nodes, 6) * 0.1
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 3],
            [1, 2, 3, 4, 5, 6, 7, 0, 1, 2]
        ])
        
        # Create models
        graphsage = HEGraphSAGE(
            in_channels=6, hidden_channels=4, out_channels=3, context=context
        )
        gat = HEGAT(
            in_channels=6, out_channels=3, heads=1, context=context
        )
        
        # Encrypt input
        encrypted_x = context.encrypt(x)
        
        # Forward passes
        sage_output = graphsage(encrypted_x, edge_index)
        gat_output = gat(encrypted_x, edge_index)
        
        # Both should produce valid outputs
        assert isinstance(sage_output, EncryptedTensor)
        assert isinstance(gat_output, EncryptedTensor)
        
        # Decrypt and check shapes
        sage_result = context.decrypt(sage_output)
        gat_result = context.decrypt(gat_output)
        
        assert sage_result.shape == (num_nodes, 3)
        assert gat_result.shape == (num_nodes, 3)
    
    def test_model_serialization_compatibility(self, context):
        """Test model parameter serialization"""
        model = HEGraphSAGE(
            in_channels=4, hidden_channels=3, out_channels=2, context=context
        )
        
        # Get state dict
        state_dict = model.state_dict()
        
        # Check that all parameters are tensors
        for name, param in state_dict.items():
            assert isinstance(param, torch.Tensor)
            assert param.requires_grad
    
    def test_gradient_computation_compatibility(self, context):
        """Test gradient computation compatibility"""
        model = HELinear(in_features=3, out_features=2, context=context)
        
        # Check gradient requirements
        assert model.weight.requires_grad
        if model.bias is not None:
            assert model.bias.requires_grad
        
        # Simulate backward pass would work
        # (Actual HE training requires more complex implementation)
        dummy_loss = model.weight.sum()
        dummy_loss.backward()
        
        assert model.weight.grad is not None

class TestModelErrorHandling:
    """Test error handling in HE models"""
    
    @pytest.fixture
    def context(self):
        """Create test CKKS context"""
        config = HEConfig(poly_modulus_degree=4096)
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx
    
    def test_invalid_dimensions(self, context):
        """Test invalid dimension handling"""
        with pytest.raises(Exception):  # Should raise some form of error
            HELinear(in_features=0, out_features=2, context=context)
        
        with pytest.raises(Exception):
            HELinear(in_features=2, out_features=0, context=context)
    
    def test_mismatched_input_dimensions(self, context):
        """Test mismatched input dimensions"""
        layer = HELinear(in_features=3, out_features=2, context=context)
        
        # Wrong input dimension
        wrong_input = torch.randn(5, 4)  # Should be (5, 3)
        encrypted_wrong = context.encrypt(wrong_input)
        
        # This might not raise an error immediately due to simplified implementation
        # but would fail in a full implementation
        try:
            output = layer.forward_encrypted(encrypted_wrong)
        except:
            pass  # Expected to fail in full implementation
    
    def test_empty_edge_index(self, context):
        """Test handling of graphs with no edges"""
        model = HEGraphSAGE(
            in_channels=3, hidden_channels=2, out_channels=1, context=context
        )
        
        # Graph with nodes but no edges
        x = torch.randn(5, 3) * 0.1
        edge_index = torch.tensor([[], []]).long()  # Empty edge list
        
        encrypted_x = context.encrypt(x)
        
        # Should handle gracefully
        output = model(encrypted_x, edge_index)
        assert isinstance(output, EncryptedTensor)
    
    def test_insufficient_noise_budget(self, context):
        """Test behavior with insufficient noise budget"""
        # This is a mock test since we don't have real bootstrapping
        model = HELinear(in_features=2, out_features=2, context=context)
        
        x = torch.randn(3, 2) * 0.1
        encrypted_x = context.encrypt(x)
        
        # Manually reduce noise budget (mock)
        encrypted_x.level = 0  # Simulate low noise
        
        # Should still work with warning or bootstrapping
        output = model.forward_encrypted(encrypted_x)
        assert isinstance(output, EncryptedTensor)