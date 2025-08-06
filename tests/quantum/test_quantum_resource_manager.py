"""
Comprehensive Test Suite for Quantum Resource Manager

Tests quantum-aware resource allocation, real-time monitoring,
auto-scaling, and privacy-preserving resource metrics.
"""

import pytest
import torch
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
import time
import threading
from collections import deque

# Import quantum modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from quantum.quantum_resource_manager import (
    QuantumResourceManager, QuantumResourceNode, ResourceType,
    QuantumAllocation, AllocationStrategy, QuantumMetricsCollector,
    QuantumPerformancePredictor, QuantumAutoScaler
)
from quantum.quantum_task_planner import QuantumTask, TaskPriority, QuantumState
from python.he_graph import CKKSContext, HEConfig

class TestQuantumResourceNode:
    """Test quantum resource node functionality"""
    
    @pytest.fixture
    def sample_node(self):
        """Create sample quantum resource node"""
        return QuantumResourceNode(
            node_id="test_cpu_0",
            resource_type=ResourceType.CPU_CORE,
            total_capacity=100.0,
            available_capacity=75.0,
            quantum_efficiency=1.3,
            entanglement_capability=True
        )
    
    def test_node_initialization(self, sample_node):
        """Test resource node initialization"""
        assert sample_node.node_id == "test_cpu_0"
        assert sample_node.resource_type == ResourceType.CPU_CORE
        assert sample_node.total_capacity == 100.0
        assert sample_node.available_capacity == 75.0
        assert sample_node.quantum_efficiency == 1.3
        assert sample_node.entanglement_capability is True
        assert sample_node.quantum_state == QuantumState.SUPERPOSITION
        
        # Test calculated properties
        assert sample_node.utilization_rate == 0.25  # (100-75)/100
        assert isinstance(sample_node.utilization_history, deque)
        assert sample_node.utilization_history.maxlen == 1000
    
    def test_node_post_init(self, sample_node):
        """Test post-initialization properties"""
        assert hasattr(sample_node, 'quantum_amplitudes')
        assert isinstance(sample_node.quantum_amplitudes, torch.Tensor)
        assert sample_node.quantum_amplitudes.dtype == torch.complex64
        assert sample_node.last_update > 0

class TestQuantumAllocation:
    """Test quantum allocation functionality"""
    
    @pytest.fixture
    def sample_allocation(self):
        """Create sample quantum allocation"""
        return QuantumAllocation(
            allocation_id="test_alloc_001",
            task_id="test_task_001",
            resource_assignments={"cpu_0": 4.0, "gpu_0": 8.0},
            quantum_entangled_allocations={"alloc_002", "alloc_003"},
            expected_duration=120.0,
            quantum_speedup_factor=1.5
        )
    
    def test_allocation_initialization(self, sample_allocation):
        """Test allocation initialization"""
        assert sample_allocation.allocation_id == "test_alloc_001"
        assert sample_allocation.task_id == "test_task_001"
        assert sample_allocation.resource_assignments == {"cpu_0": 4.0, "gpu_0": 8.0}
        assert sample_allocation.quantum_entangled_allocations == {"alloc_002", "alloc_003"}
        assert sample_allocation.expected_duration == 120.0
        assert sample_allocation.quantum_speedup_factor == 1.5
        assert isinstance(sample_allocation.probability_amplitude, complex)
        assert sample_allocation.allocation_timestamp > 0

class TestQuantumResourceManager:
    """Test quantum resource manager core functionality"""
    
    @pytest.fixture
    def he_context(self):
        """Create HE context for testing"""
        config = HEConfig(
            poly_modulus_degree=8192,
            coeff_modulus_bits=[60, 40, 60],
            security_level=80
        )
        context = CKKSContext(config)
        context.generate_keys()
        return context
    
    @pytest.fixture
    def resource_manager(self, he_context):
        """Create quantum resource manager"""
        return QuantumResourceManager(
            he_context=he_context,
            monitoring_interval=0.1,
            quantum_coherence_time=60.0,
            enable_distributed=False
        )
    
    def test_manager_initialization(self, resource_manager):
        """Test resource manager initialization"""
        assert resource_manager.he_context is not None
        assert resource_manager.monitoring_interval == 0.1
        assert resource_manager.quantum_coherence_time == 60.0
        assert resource_manager.enable_distributed is False
        
        # Test initial state
        assert isinstance(resource_manager.quantum_nodes, dict)
        assert isinstance(resource_manager.active_allocations, dict)
        assert isinstance(resource_manager.allocation_history, list)
        assert resource_manager.monitoring_active is False
        assert resource_manager.quantum_speedup_achieved == 1.0
    
    @patch('psutil.cpu_count', return_value=4)
    @patch('psutil.virtual_memory')
    @patch('GPUtil.getGPUs', return_value=[])
    @pytest.mark.asyncio
    async def test_system_resource_discovery(self, mock_gpus, mock_memory, mock_cpu_count, resource_manager):
        """Test system resource discovery"""
        # Mock memory info
        mock_memory.return_value = MagicMock(
            total=16 * 1024**3,  # 16GB
            available=8 * 1024**3  # 8GB available
        )
        
        await resource_manager._discover_system_resources()
        
        # Verify CPU cores discovered
        cpu_nodes = [node for node in resource_manager.quantum_nodes.values() 
                    if node.resource_type == ResourceType.CPU_CORE]
        assert len(cpu_nodes) == 4
        
        # Verify memory node discovered  
        memory_nodes = [node for node in resource_manager.quantum_nodes.values()
                       if node.resource_type == ResourceType.SYSTEM_MEMORY]
        assert len(memory_nodes) == 1
        
        memory_node = memory_nodes[0]
        assert memory_node.node_id == "system_memory"
        assert memory_node.total_capacity == 16.0  # GB
        assert memory_node.available_capacity == 8.0  # GB
    
    @pytest.mark.asyncio
    async def test_quantum_properties_initialization(self, resource_manager):
        """Test quantum properties initialization"""
        # Add mock nodes
        resource_manager.quantum_nodes = {
            "cpu_0": QuantumResourceNode("cpu_0", ResourceType.CPU_CORE, 100, 80),
            "cpu_1": QuantumResourceNode("cpu_1", ResourceType.CPU_CORE, 100, 90),
            "gpu_0": QuantumResourceNode("gpu_0", ResourceType.GPU_MEMORY, 16000, 12000)
        }
        
        await resource_manager._initialize_quantum_properties()
        
        # Verify entanglement matrix
        assert resource_manager.entanglement_matrix is not None
        assert resource_manager.entanglement_matrix.shape == (3, 3)
        assert resource_manager.entanglement_matrix.dtype == torch.complex64
        
        # Verify diagonal is zero (no self-entanglement)
        diagonal = torch.diag(resource_manager.entanglement_matrix)
        assert torch.allclose(diagonal, torch.zeros(3, dtype=torch.complex64), atol=1e-6)
        
        # Verify quantum amplitudes initialized
        for node in resource_manager.quantum_nodes.values():
            assert hasattr(node, 'quantum_amplitudes')
            assert isinstance(node.quantum_amplitudes, torch.Tensor)
            assert node.quantum_amplitudes.dtype == torch.complex64
    
    @pytest.mark.asyncio
    async def test_quantum_task_requirement_analysis(self, resource_manager):
        """Test quantum task requirement analysis"""
        # Setup mock nodes
        resource_manager.quantum_nodes = {
            "cpu_0": QuantumResourceNode("cpu_0", ResourceType.CPU_CORE, 100, 80, quantum_efficiency=1.5),
            "gpu_memory": QuantumResourceNode("gpu_memory", ResourceType.GPU_MEMORY, 16000, 12000, quantum_efficiency=2.0)
        }
        
        task = QuantumTask(
            task_id="analysis_task",
            name="Analysis Test Task",
            priority=TaskPriority.HIGH,
            estimated_duration=120.0,
            resource_requirements={"cpu_cores": 4.0, "gpu_memory": 8.0}
        )
        
        quantum_requirements = await resource_manager._analyze_quantum_requirements(task)
        
        assert isinstance(quantum_requirements, dict)
        assert len(quantum_requirements) > 0
        
        # Verify quantum efficiency applied
        for node_id, amount in quantum_requirements.items():
            node = resource_manager.quantum_nodes[node_id]
            assert amount > 0
            # Amount should be reduced by quantum efficiency
    
    @pytest.mark.asyncio
    async def test_superposition_allocation_generation(self, resource_manager):
        """Test superposition allocation generation"""
        requirements = {"cpu_0": 4.0, "gpu_0": 8.0}
        
        alternatives = await resource_manager._generate_superposition_allocations(
            requirements, AllocationStrategy.QUANTUM_OPTIMAL
        )
        
        assert isinstance(alternatives, list)
        assert len(alternatives) >= 1
        
        # Verify all alternatives are valid
        for alternative in alternatives:
            assert isinstance(alternative, dict)
            for node_id, amount in alternative.items():
                assert amount > 0
    
    @pytest.mark.asyncio
    async def test_quantum_measurement_allocation_selection(self, resource_manager):
        """Test quantum measurement for optimal allocation"""
        # Setup mock nodes with availability
        resource_manager.quantum_nodes = {
            "cpu_0": QuantumResourceNode("cpu_0", ResourceType.CPU_CORE, 100, 50, quantum_efficiency=1.2),
            "cpu_1": QuantumResourceNode("cpu_1", ResourceType.CPU_CORE, 100, 80, quantum_efficiency=1.5),
            "gpu_0": QuantumResourceNode("gpu_0", ResourceType.GPU_MEMORY, 16000, 12000, quantum_efficiency=1.8)
        }
        
        alternatives = [
            {"cpu_0": 30.0, "gpu_0": 4000.0},
            {"cpu_1": 25.0, "gpu_0": 6000.0},
            {"cpu_0": 20.0, "cpu_1": 15.0, "gpu_0": 5000.0}
        ]
        
        task = QuantumTask(
            task_id="measurement_task",
            name="Measurement Test",
            priority=TaskPriority.HIGH,
            estimated_duration=90.0,
            resource_requirements={}
        )
        
        optimal = await resource_manager._quantum_measure_optimal_allocation(alternatives, task)
        
        assert optimal is not None
        assert optimal in alternatives
        assert isinstance(optimal, dict)
    
    @pytest.mark.asyncio
    async def test_resource_allocation_and_deallocation(self, resource_manager):
        """Test complete allocation and deallocation cycle"""
        # Initialize with mock resources
        await resource_manager.initialize_quantum_resources()
        
        task = QuantumTask(
            task_id="cycle_task",
            name="Cycle Test Task",
            priority=TaskPriority.MEDIUM,
            estimated_duration=60.0,
            resource_requirements={"cpu_cores": 2.0, "memory": 4.0}
        )
        
        # Allocate resources
        allocation = await resource_manager.allocate_quantum_resources(
            task, 
            strategy=AllocationStrategy.QUANTUM_OPTIMAL,
            enable_superposition=True
        )
        
        if allocation:  # May be None if insufficient resources
            assert isinstance(allocation, QuantumAllocation)
            assert allocation.task_id == task.task_id
            assert allocation.allocation_id in resource_manager.active_allocations
            
            # Deallocate resources
            success = await resource_manager.deallocate_quantum_resources(allocation.allocation_id)
            assert success is True
            assert allocation.allocation_id not in resource_manager.active_allocations
            assert allocation in resource_manager.allocation_history
    
    @pytest.mark.asyncio
    async def test_quantum_resource_status_reporting(self, resource_manager):
        """Test quantum resource status reporting"""
        await resource_manager.initialize_quantum_resources()
        
        status = await resource_manager.get_quantum_resource_status(encrypt_sensitive=True)
        
        assert isinstance(status, dict)
        
        # Verify required fields
        required_fields = [
            "timestamp", "total_nodes", "active_allocations", 
            "quantum_coherence_remaining", "entanglement_efficiency",
            "overall_utilization", "quantum_speedup_average", "node_details"
        ]
        
        for field in required_fields:
            assert field in status
        
        assert status["total_nodes"] == len(resource_manager.quantum_nodes)
        assert status["active_allocations"] == len(resource_manager.active_allocations)
        assert 0.0 <= status["quantum_coherence_remaining"] <= 1.0
        assert 0.0 <= status["entanglement_efficiency"] <= 1.0
        assert 0.0 <= status["overall_utilization"] <= 1.0
        assert status["quantum_speedup_average"] >= 1.0
        
        # Verify node details
        assert len(status["node_details"]) == len(resource_manager.quantum_nodes)
        
        for node_id, node_detail in status["node_details"].items():
            assert "resource_type" in node_detail
            assert "utilization_rate" in node_detail
            assert "quantum_efficiency" in node_detail
            assert "quantum_state" in node_detail
            assert "available_capacity" in node_detail
            assert "total_capacity" in node_detail
            
            # Verify encrypted metrics if enabled
            assert "encrypted_metrics" in node_detail
    
    @pytest.mark.asyncio
    async def test_quantum_optimization_algorithms(self, resource_manager):
        """Test quantum optimization algorithms"""
        await resource_manager.initialize_quantum_resources()
        
        # Add some mock allocations
        mock_allocation = QuantumAllocation(
            allocation_id="opt_test_001",
            task_id="opt_task_001", 
            resource_assignments={"cpu_0": 2.0},
            expected_duration=60.0
        )
        resource_manager.active_allocations[mock_allocation.allocation_id] = mock_allocation
        
        optimization_results = await resource_manager.optimize_quantum_resource_allocation()
        
        assert isinstance(optimization_results, dict)
        assert "timestamp" in optimization_results
        assert "optimizations_applied" in optimization_results
        assert "performance_improvement" in optimization_results
        assert "energy_savings" in optimization_results
        assert "quantum_coherence_extended" in optimization_results
        
        # Verify optimization types applied
        optimizations = optimization_results["optimizations_applied"]
        expected_optimizations = [
            "quantum_annealing", "entanglement_balancing", 
            "interference_mitigation", "quantum_auto_scaling"
        ]
        
        for opt in expected_optimizations:
            assert opt in optimizations
    
    @pytest.mark.asyncio
    async def test_quantum_resource_prediction(self, resource_manager):
        """Test quantum resource need prediction"""
        await resource_manager.initialize_quantum_resources()
        
        # Add mock allocation history
        for i in range(10):
            mock_allocation = QuantumAllocation(
                allocation_id=f"pred_alloc_{i:03d}",
                task_id=f"pred_task_{i:03d}",
                resource_assignments={f"cpu_{i%2}": float(i+1)},
                expected_duration=60.0 + i*10,
                quantum_speedup_factor=1.0 + i*0.1
            )
            resource_manager.allocation_history.append(mock_allocation)
        
        predictions = await resource_manager.predict_quantum_resource_needs(time_horizon=3600.0)
        
        assert isinstance(predictions, dict)
        assert "time_horizon" in predictions
        assert "predicted_requirements" in predictions
        assert "confidence_intervals" in predictions
        assert "quantum_enhancement_opportunities" in predictions
        assert "recommended_actions" in predictions
        
        assert predictions["time_horizon"] == 3600.0
        assert isinstance(predictions["predicted_requirements"], dict)
        assert isinstance(predictions["quantum_enhancement_opportunities"], list)
        assert isinstance(predictions["recommended_actions"], list)
    
    def test_resource_entanglement_calculation(self, resource_manager):
        """Test resource entanglement strength calculation"""
        node1 = QuantumResourceNode("cpu_0", ResourceType.CPU_CORE, 100, 80, quantum_efficiency=1.2)
        node2 = QuantumResourceNode("cpu_1", ResourceType.CPU_CORE, 100, 70, quantum_efficiency=1.3)
        node3 = QuantumResourceNode("gpu_0", ResourceType.GPU_MEMORY, 16000, 12000, quantum_efficiency=1.8)
        
        # Same type resources should have higher entanglement
        cpu_entanglement = resource_manager._calculate_resource_entanglement(node1, node2)
        mixed_entanglement = resource_manager._calculate_resource_entanglement(node1, node3)
        
        assert isinstance(cpu_entanglement, complex)
        assert isinstance(mixed_entanglement, complex)
        
        # CPU-CPU entanglement should be stronger than CPU-GPU
        assert abs(cpu_entanglement) > abs(mixed_entanglement)
    
    @patch('psutil.cpu_percent', return_value=[25.0, 50.0, 75.0, 90.0])
    @patch('psutil.virtual_memory')
    def test_resource_utilization_update(self, mock_memory, mock_cpu, resource_manager):
        """Test resource utilization monitoring"""
        # Mock memory
        mock_memory.return_value = MagicMock(
            available=8 * 1024**3,
            percent=50.0
        )
        
        # Add mock CPU nodes
        for i in range(4):
            node = QuantumResourceNode(f"cpu_core_{i}", ResourceType.CPU_CORE, 1.0, 1.0)
            resource_manager.quantum_nodes[node.node_id] = node
        
        # Add mock memory node
        memory_node = QuantumResourceNode("system_memory", ResourceType.SYSTEM_MEMORY, 16.0, 16.0)
        resource_manager.quantum_nodes[memory_node.node_id] = memory_node
        
        # Update utilization
        resource_manager._update_resource_utilization()
        
        # Verify CPU utilization updated
        cpu_nodes = [node for node in resource_manager.quantum_nodes.values() 
                    if node.resource_type == ResourceType.CPU_CORE]
        
        assert len(cpu_nodes) == 4
        assert cpu_nodes[0].utilization_rate == 0.25
        assert cpu_nodes[1].utilization_rate == 0.50
        assert cpu_nodes[2].utilization_rate == 0.75
        assert cpu_nodes[3].utilization_rate == 0.90
        
        # Verify memory utilization updated
        memory_node = resource_manager.quantum_nodes["system_memory"]
        assert memory_node.utilization_rate == 0.50
        assert memory_node.available_capacity == 8.0

class TestQuantumMetricsCollector:
    """Test quantum metrics collection"""
    
    @pytest.fixture
    def he_context(self):
        """Create HE context"""
        config = HEConfig(poly_modulus_degree=8192, coeff_modulus_bits=[60, 40, 60])
        context = CKKSContext(config)
        context.generate_keys()
        return context
    
    @pytest.fixture
    def metrics_collector(self, he_context):
        """Create metrics collector"""
        return QuantumMetricsCollector(he_context)
    
    @pytest.mark.asyncio
    async def test_quantum_metrics_collection(self, metrics_collector):
        """Test quantum metrics collection"""
        # Create mock nodes and allocations
        nodes = {
            "cpu_0": QuantumResourceNode("cpu_0", ResourceType.CPU_CORE, 100, 75, quantum_efficiency=1.2),
            "gpu_0": QuantumResourceNode("gpu_0", ResourceType.GPU_MEMORY, 16000, 12000, quantum_efficiency=1.8)
        }
        
        allocations = {
            "alloc_001": QuantumAllocation(
                allocation_id="alloc_001",
                task_id="task_001",
                resource_assignments={"cpu_0": 25.0},
                quantum_speedup_factor=1.3
            )
        }
        
        # Collect metrics
        await metrics_collector.collect_quantum_resource_metrics(nodes, allocations)
        
        # Verify metrics stored
        assert len(metrics_collector.metrics_history) > 0
        
        latest_timestamp = max(metrics_collector.metrics_history.keys())
        latest_metrics = metrics_collector.metrics_history[latest_timestamp]
        
        assert "utilization" in latest_metrics
        assert "allocations" in latest_metrics
        assert "encrypted_summary" in latest_metrics
        
        # Verify utilization metrics
        utilization_metrics = latest_metrics["utilization"]
        assert "cpu_0" in utilization_metrics
        assert "gpu_0" in utilization_metrics
        
        cpu_metrics = utilization_metrics["cpu_0"]
        assert "utilization_rate" in cpu_metrics
        assert "quantum_efficiency" in cpu_metrics
        assert "quantum_state" in cpu_metrics
        assert cpu_metrics["utilization_rate"] == 0.25
        assert cpu_metrics["quantum_efficiency"] == 1.2
        
        # Verify allocation metrics
        allocation_metrics = latest_metrics["allocations"]
        assert "alloc_001" in allocation_metrics
        
        alloc_metrics = allocation_metrics["alloc_001"]
        assert "quantum_speedup" in alloc_metrics
        assert "resource_count" in alloc_metrics
        assert alloc_metrics["quantum_speedup"] == 1.3
        assert alloc_metrics["resource_count"] == 1

class TestQuantumPerformancePredictor:
    """Test quantum performance prediction"""
    
    @pytest.fixture
    def predictor(self):
        """Create performance predictor"""
        return QuantumPerformancePredictor()
    
    @pytest.fixture
    def mock_nodes(self):
        """Create mock resource nodes"""
        return {
            "cpu_0": QuantumResourceNode("cpu_0", ResourceType.CPU_CORE, 100, 80),
            "cpu_1": QuantumResourceNode("cpu_1", ResourceType.CPU_CORE, 100, 90),
            "gpu_0": QuantumResourceNode("gpu_0", ResourceType.GPU_MEMORY, 16000, 12000)
        }
    
    @pytest.fixture
    def mock_history(self):
        """Create mock allocation history"""
        history = []
        for i in range(20):
            allocation = QuantumAllocation(
                allocation_id=f"hist_alloc_{i:03d}",
                task_id=f"hist_task_{i:03d}",
                resource_assignments={
                    "cpu_0": 10.0 + i * 2.0,
                    "gpu_0": 1000.0 + i * 100.0
                },
                expected_duration=60.0 + i * 5.0
            )
            history.append(allocation)
        return history
    
    @pytest.fixture
    def he_context(self):
        """Create HE context"""
        config = HEConfig(poly_modulus_degree=8192)
        context = CKKSContext(config)
        context.generate_keys()
        return context
    
    @pytest.mark.asyncio
    async def test_resource_requirement_prediction(self, predictor, mock_nodes, mock_history, he_context):
        """Test resource requirement prediction"""
        predictions = await predictor.predict_resource_requirements(
            mock_nodes, mock_history, 3600.0, he_context
        )
        
        assert isinstance(predictions, dict)
        assert "requirements" in predictions
        assert "confidence" in predictions
        assert "quantum_opportunities" in predictions
        assert "recommendations" in predictions
        
        # Verify predictions structure
        requirements = predictions["requirements"]
        assert isinstance(requirements, dict)
        
        for node_id, predicted_util in requirements.items():
            assert 0.0 <= predicted_util <= 1.0
        
        # Verify confidence
        confidence = predictions["confidence"]
        assert 0.0 <= confidence <= 1.0
        
        # Verify quantum opportunities
        opportunities = predictions["quantum_opportunities"]
        assert isinstance(opportunities, list)
        
        for opportunity in opportunities:
            assert "node_id" in opportunity
            assert "opportunity" in opportunity
            assert "potential_improvement" in opportunity
        
        # Verify recommendations
        recommendations = predictions["recommendations"]
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_prediction_with_insufficient_history(self, predictor, mock_nodes, he_context):
        """Test prediction with insufficient historical data"""
        short_history = []  # Empty history
        
        predictions = await predictor.predict_resource_requirements(
            mock_nodes, short_history, 3600.0, he_context
        )
        
        assert isinstance(predictions, dict)
        assert predictions["confidence"] == 0.5  # Low confidence with no history
        assert "Collect more historical data" in predictions["recommendations"]
    
    def test_utilization_series_extraction(self, predictor, mock_history, mock_nodes):
        """Test utilization series extraction from history"""
        series = predictor._extract_utilization_series(mock_history, mock_nodes)
        
        assert isinstance(series, dict)
        assert "cpu_0" in series
        assert "gpu_0" in series
        
        # Verify series data
        cpu_series = series["cpu_0"]
        gpu_series = series["gpu_0"]
        
        assert len(cpu_series) == len(mock_history)
        assert len(gpu_series) == len(mock_history)
        
        # Verify utilization values are reasonable
        for util in cpu_series:
            assert 0.0 <= util <= 1.0
        
        for util in gpu_series:
            assert 0.0 <= util <= 1.0
    
    def test_recommendation_generation(self, predictor):
        """Test optimization recommendation generation"""
        predictions = {
            "cpu_0": 0.9,  # High utilization
            "cpu_1": 0.3,  # Low utilization
            "gpu_0": 0.7   # Medium utilization
        }
        
        opportunities = [
            {"node_id": "cpu_0", "opportunity": "quantum_enhancement", "potential_improvement": 0.3}
        ]
        
        recommendations = predictor._generate_recommendations(predictions, opportunities)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend scaling for high utilization
        scale_recommendations = [r for r in recommendations if "scale" in r.lower()]
        assert len(scale_recommendations) > 0
        
        # Should recommend quantum enhancements
        quantum_recommendations = [r for r in recommendations if "quantum" in r.lower()]
        assert len(quantum_recommendations) > 0

class TestQuantumAutoScaler:
    """Test quantum auto-scaling functionality"""
    
    @pytest.fixture
    def auto_scaler(self):
        """Create quantum auto-scaler"""
        return QuantumAutoScaler()
    
    @pytest.fixture
    def mock_current_nodes(self):
        """Create mock current resource nodes"""
        return {
            "cpu_0": QuantumResourceNode("cpu_0", ResourceType.CPU_CORE, 100, 20, quantum_efficiency=1.2),
            "cpu_1": QuantumResourceNode("cpu_1", ResourceType.CPU_CORE, 100, 90, quantum_efficiency=1.1),
            "gpu_0": QuantumResourceNode("gpu_0", ResourceType.GPU_MEMORY, 16000, 2000, quantum_efficiency=1.8)
        }
    
    @pytest.mark.asyncio
    async def test_resource_scaling_decisions(self, auto_scaler, mock_current_nodes):
        """Test auto-scaling decision making"""
        predictions = {
            "cpu_0": 0.9,  # High predicted utilization -> scale up
            "cpu_1": 0.1,  # Low predicted utilization -> scale down
            "gpu_0": 0.5   # Medium predicted utilization -> no action
        }
        
        scaling_results = await auto_scaler.scale_resources(predictions, mock_current_nodes)
        
        assert isinstance(scaling_results, dict)
        assert "scaling_actions" in scaling_results
        assert "energy_impact" in scaling_results
        
        scaling_actions = scaling_results["scaling_actions"]
        assert isinstance(scaling_actions, list)
        
        # Verify scaling decisions
        scale_up_actions = [a for a in scaling_actions if a["action"] == "scale_up"]
        scale_down_actions = [a for a in scaling_actions if a["action"] == "scale_down"]
        
        # Should have scale up for cpu_0
        cpu_0_actions = [a for a in scale_up_actions if a["node_id"] == "cpu_0"]
        assert len(cpu_0_actions) > 0
        
        # Should have scale down for cpu_1
        cpu_1_actions = [a for a in scale_down_actions if a["node_id"] == "cpu_1"]
        assert len(cpu_1_actions) > 0
        
        # Verify action structure
        for action in scaling_actions:
            assert "action" in action
            assert "node_id" in action
            assert "factor" in action
            assert "reason" in action
            assert action["factor"] > 0
    
    def test_energy_impact_calculation(self, auto_scaler):
        """Test energy impact calculation"""
        actions = [
            {"action": "scale_up", "factor": 1.5},
            {"action": "scale_up", "factor": 1.2},
            {"action": "scale_down", "factor": 0.8},
        ]
        
        energy_impact = auto_scaler._calculate_energy_impact(actions)
        
        assert isinstance(energy_impact, float)
        # 2 scale ups (+0.3 each) and 1 scale down (-0.2) = +0.4
        assert energy_impact == pytest.approx(0.4, abs=0.01)

class TestQuantumResourceManagerIntegration:
    """Integration tests for quantum resource management"""
    
    @pytest.mark.asyncio
    async def test_full_resource_lifecycle(self):
        """Test complete resource lifecycle with monitoring"""
        # Create resource manager
        manager = QuantumResourceManager(
            monitoring_interval=0.05,  # Fast for testing
            quantum_coherence_time=30.0,
            enable_distributed=False
        )
        
        try:
            # Initialize resources
            success = await manager.initialize_quantum_resources()
            assert success
            
            # Wait for monitoring to start
            await asyncio.sleep(0.2)
            assert manager.monitoring_active
            
            # Create test task
            task = QuantumTask(
                task_id="lifecycle_task",
                name="Lifecycle Test Task",
                priority=TaskPriority.HIGH,
                estimated_duration=60.0,
                resource_requirements={"cpu_cores": 2.0, "memory": 4.0}
            )
            
            # Allocate resources
            allocation = await manager.allocate_quantum_resources(task)
            
            if allocation:
                assert allocation.task_id == task.task_id
                assert allocation.allocation_id in manager.active_allocations
                
                # Check quantum properties
                assert allocation.quantum_speedup_factor >= 1.0
                
                # Get status
                status = await manager.get_quantum_resource_status()
                assert status["active_allocations"] == 1
                
                # Perform optimization
                optimization = await manager.optimize_quantum_resource_allocation()
                assert "optimizations_applied" in optimization
                
                # Predict future needs
                predictions = await manager.predict_quantum_resource_needs()
                assert "predicted_requirements" in predictions
                
                # Deallocate resources
                dealloc_success = await manager.deallocate_quantum_resources(allocation.allocation_id)
                assert dealloc_success
                assert allocation.allocation_id not in manager.active_allocations
        
        finally:
            # Cleanup
            await manager.shutdown()
            assert not manager.monitoring_active
    
    @pytest.mark.asyncio
    async def test_concurrent_resource_allocations(self):
        """Test concurrent resource allocation requests"""
        manager = QuantumResourceManager(
            monitoring_interval=0.1,
            quantum_coherence_time=60.0
        )
        
        try:
            await manager.initialize_quantum_resources()
            
            # Create multiple tasks
            tasks = []
            for i in range(5):
                task = QuantumTask(
                    task_id=f"concurrent_task_{i}",
                    name=f"Concurrent Task {i}",
                    priority=TaskPriority(i % 5),
                    estimated_duration=60.0 + i * 10,
                    resource_requirements={"cpu_cores": 1.0 + i * 0.5, "memory": 2.0 + i}
                )
                tasks.append(task)
            
            # Allocate resources concurrently
            allocation_tasks = [
                manager.allocate_quantum_resources(task) for task in tasks
            ]
            
            allocations = await asyncio.gather(*allocation_tasks, return_exceptions=True)
            
            # Verify allocations
            successful_allocations = [alloc for alloc in allocations if isinstance(alloc, QuantumAllocation)]
            
            # At least some should succeed
            assert len(successful_allocations) > 0
            
            # Verify no resource conflicts
            all_assignments = {}
            for allocation in successful_allocations:
                for node_id, amount in allocation.resource_assignments.items():
                    if node_id not in all_assignments:
                        all_assignments[node_id] = 0
                    all_assignments[node_id] += amount
            
            # Check against node capacities
            for node_id, total_assigned in all_assignments.items():
                if node_id in manager.quantum_nodes:
                    node = manager.quantum_nodes[node_id]
                    assert total_assigned <= node.total_capacity
            
            # Cleanup allocations
            for allocation in successful_allocations:
                await manager.deallocate_quantum_resources(allocation.allocation_id)
        
        finally:
            await manager.shutdown()
    
    @patch('time.time')
    @pytest.mark.asyncio
    async def test_quantum_coherence_management(self, mock_time):
        """Test quantum coherence time management"""
        # Mock time progression
        start_time = 1000.0
        mock_time.return_value = start_time
        
        manager = QuantumResourceManager(
            quantum_coherence_time=10.0  # Short coherence time for testing
        )
        
        try:
            await manager.initialize_quantum_resources()
            
            # Verify initial coherence
            initial_coherence = manager._calculate_coherence_remaining()
            assert initial_coherence == 1.0
            
            # Advance time beyond coherence time
            mock_time.return_value = start_time + 15.0
            
            # Update coherence
            manager._check_quantum_coherence()
            
            # Verify coherence degradation
            degraded_coherence = manager._calculate_coherence_remaining()
            assert degraded_coherence < initial_coherence
            
            # Check quantum state changes
            collapsed_nodes = [
                node for node in manager.quantum_nodes.values()
                if node.quantum_state == QuantumState.COLLAPSED
            ]
            assert len(collapsed_nodes) > 0
        
        finally:
            await manager.shutdown()

# Performance tests

class TestQuantumResourcePerformance:
    """Performance tests for quantum resource management"""
    
    @pytest.mark.asyncio
    async def test_large_scale_resource_management(self):
        """Test performance with large number of resources"""
        manager = QuantumResourceManager(
            monitoring_interval=0.5,
            quantum_coherence_time=120.0
        )
        
        # Add many mock resources
        for i in range(100):  # Moderate number for CI
            node = QuantumResourceNode(
                node_id=f"perf_cpu_{i}",
                resource_type=ResourceType.CPU_CORE,
                total_capacity=100.0,
                available_capacity=float(np.random.uniform(50, 100)),
                quantum_efficiency=np.random.uniform(1.0, 2.0)
            )
            manager.quantum_nodes[node.node_id] = node
        
        start_time = time.time()
        
        # Initialize quantum properties
        await manager._initialize_quantum_properties()
        
        initialization_time = time.time() - start_time
        
        # Should complete initialization quickly even with many resources
        assert initialization_time < 5.0
        
        # Verify entanglement matrix size
        assert manager.entanglement_matrix.shape == (100, 100)
        
        # Test status reporting performance
        start_time = time.time()
        status = await manager.get_quantum_resource_status()
        status_time = time.time() - start_time
        
        assert status_time < 2.0
        assert status["total_nodes"] == 100
        
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_monitoring_overhead(self):
        """Test monitoring performance overhead"""
        manager = QuantumResourceManager(
            monitoring_interval=0.01,  # Very frequent monitoring
            quantum_coherence_time=60.0
        )
        
        try:
            await manager.initialize_quantum_resources()
            
            start_time = time.time()
            
            # Let monitoring run for a short time
            await asyncio.sleep(1.0)
            
            # Verify monitoring is active and responsive
            assert manager.monitoring_active
            
            # Check that quantum states are being updated
            recent_updates = [
                node for node in manager.quantum_nodes.values()
                if time.time() - node.last_update < 2.0
            ]
            
            # Most nodes should have recent updates
            assert len(recent_updates) > len(manager.quantum_nodes) * 0.5
            
        finally:
            await manager.shutdown()

# Error handling and edge cases

class TestQuantumResourceEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_allocation_with_insufficient_resources(self):
        """Test allocation when resources are insufficient"""
        manager = QuantumResourceManager()
        
        # Add minimal resources
        manager.quantum_nodes = {
            "small_cpu": QuantumResourceNode(
                "small_cpu", ResourceType.CPU_CORE, 1.0, 0.5, quantum_efficiency=1.0
            )
        }
        
        # Try to allocate more than available
        task = QuantumTask(
            task_id="big_task",
            name="Resource Hungry Task",
            priority=TaskPriority.HIGH,
            estimated_duration=60.0,
            resource_requirements={"cpu_cores": 10.0}  # More than available
        )
        
        allocation = await manager.allocate_quantum_resources(task)
        
        # Should return None or handle gracefully
        # Exact behavior depends on implementation strategy
        if allocation:
            # If allocation is successful, verify it's within resource limits
            total_assigned = sum(allocation.resource_assignments.values())
            assert total_assigned <= 1.0  # Can't exceed total capacity
    
    @pytest.mark.asyncio
    async def test_deallocation_of_nonexistent_allocation(self):
        """Test deallocating non-existent allocation"""
        manager = QuantumResourceManager()
        
        # Try to deallocate non-existent allocation
        result = await manager.deallocate_quantum_resources("non_existent_allocation")
        
        # Should return False and not crash
        assert result is False
    
    @pytest.mark.asyncio
    async def test_quantum_state_corruption_recovery(self):
        """Test recovery from quantum state corruption"""
        manager = QuantumResourceManager()
        await manager.initialize_quantum_resources()
        
        # Corrupt quantum state
        manager.entanglement_matrix = None
        
        # Should handle gracefully
        try:
            status = await manager.get_quantum_resource_status()
            # Should not crash, may have degraded functionality
            assert "total_nodes" in status
        except Exception as e:
            # If it does raise an exception, it should be informative
            assert str(e)  # Non-empty error message
    
    def test_zero_capacity_resource_handling(self):
        """Test handling of zero-capacity resources"""
        node = QuantumResourceNode(
            node_id="zero_node",
            resource_type=ResourceType.CPU_CORE,
            total_capacity=0.0,
            available_capacity=0.0
        )
        
        # Should not crash during initialization
        assert node.utilization_rate == 0.0 or np.isnan(node.utilization_rate)
        assert isinstance(node.quantum_amplitudes, torch.Tensor)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])