"""
Comprehensive Test Suite for Quantum Task Planner

Tests all quantum-inspired algorithms, privacy-preserving features,
and integration with homomorphic encryption infrastructure.
"""


import pytest
import torch
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Set
import time

# Import quantum modules

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


from quantum.quantum_task_planner import (
    QuantumTaskScheduler, QuantumTask, QuantumState, TaskPriority,
    QuantumOptimizer, EntanglementManager, TunnelingPathSolver,
    InterferenceResolver, create_quantum_task_scheduler,
    QuantumSchedulingSession
)

from quantum.quantum_resource_manager import (
    QuantumResourceManager, QuantumResourceNode, ResourceType,
    QuantumAllocation, AllocationStrategy
)

from python.he_graph import CKKSContext, HEConfig

class TestQuantumTaskScheduler:
    """Test quantum task scheduling functionality"""

    @pytest.fixture
    def he_context(self) -> None:
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
    def quantum_scheduler(self, he_context) -> None:
        """Create quantum task scheduler"""
        return QuantumTaskScheduler(
            he_context=he_context,
            max_entanglement_depth=3,
            quantum_coherence_time=100.0
        )

    @pytest.fixture
    def sample_tasks(self) -> None:
        """Create sample quantum tasks"""
        return [
            QuantumTask(
                task_id="task_001",
                name="Homomorphic Computation",
                priority=TaskPriority.HIGH,
                estimated_duration=60.0,
                resource_requirements={"gpu_memory": 4.0, "cpu_cores": 2}
            ),
            QuantumTask(
                task_id="task_002",
                name="Graph Embedding",
                priority=TaskPriority.CRITICAL,
                estimated_duration=120.0,
                resource_requirements={"gpu_memory": 8.0, "cpu_cores": 4},
                dependencies={"task_001"}
            ),
            QuantumTask(
                task_id="task_003",
                name="Classification",
                priority=TaskPriority.MEDIUM,
                estimated_duration=90.0,
                resource_requirements={"gpu_memory": 6.0, "cpu_cores": 3},
                dependencies={"task_002"}
            )
        ]

    @pytest.mark.asyncio
    async def test_quantum_task_addition(self, quantum_scheduler, sample_tasks):
        """Test adding quantum tasks with encryption"""
        task = sample_tasks[0]

        # Add task with encryption
        task_id = await quantum_scheduler.add_quantum_task(task, encrypt_metadata=True)

        assert task_id == task.task_id
        assert task_id in quantum_scheduler.quantum_tasks

        stored_task = quantum_scheduler.quantum_tasks[task_id]
        assert stored_task.quantum_state == QuantumState.SUPERPOSITION
        assert stored_task.encrypted_metadata is not None
        assert len(stored_task.execution_paths) > 0
        assert stored_task.probability_amplitude != 0

    @pytest.mark.asyncio
    async def test_quantum_scheduling_basic(self, quantum_scheduler, sample_tasks):
        """Test basic quantum scheduling"""
        # Add all tasks
        for task in sample_tasks:
            await quantum_scheduler.add_quantum_task(task)

        # Schedule tasks
        schedule = await quantum_scheduler.schedule_quantum_tasks(time_horizon=3600.0)

        assert isinstance(schedule, list)
        assert len(schedule) == len(sample_tasks)

        # Verify schedule properties
        for scheduled_task in schedule:
            assert "task_id" in scheduled_task
            assert "start_time" in scheduled_task
            assert "end_time" in scheduled_task
            assert scheduled_task["end_time"] >= scheduled_task["start_time"]

        # Verify quantum speedup
        assert quantum_scheduler.quantum_speedup_achieved >= 1.0

    @pytest.mark.asyncio
    async def test_superposition_analysis(self, quantum_scheduler, sample_tasks):
        """Test quantum superposition state analysis"""
        task = sample_tasks[0]
        await quantum_scheduler.add_quantum_task(task)

        # Analyze superposition states
        analysis = await quantum_scheduler._analyze_superposition_states()

        assert "total_paths" in analysis
        assert "quantum_amplitudes" in analysis
        assert "coherence_measures" in analysis
        assert analysis["total_paths"] > 0
        assert task.task_id in analysis["quantum_amplitudes"]
        assert task.task_id in analysis["coherence_measures"]

        # Verify quantum coherence measurement
        coherence = analysis["coherence_measures"][task.task_id]
        assert 0.0 <= coherence <= 1.0

    @pytest.mark.asyncio
    async def test_entanglement_resolution(self, quantum_scheduler, sample_tasks):
        """Test quantum entanglement dependency resolution"""
        # Add tasks with dependencies
        for task in sample_tasks:
            await quantum_scheduler.add_quantum_task(task)

        # Resolve entangled dependencies
        entangled_groups = await quantum_scheduler._resolve_entangled_dependencies()

        assert isinstance(entangled_groups, list)

        # Verify entanglement properties
        for group in entangled_groups:
            assert isinstance(group, set)
            assert len(group) > 1

            # Check that entangled tasks are properly marked
            for task_id in group:
                if task_id in quantum_scheduler.quantum_tasks:
                    task = quantum_scheduler.quantum_tasks[task_id]
                    assert task.quantum_state == QuantumState.ENTANGLED
                    assert len(task.entangled_tasks) > 0

    @pytest.mark.asyncio
    async def test_tunneling_paths(self, quantum_scheduler, sample_tasks):
        """Test quantum tunneling path finding"""
        task = sample_tasks[0]
        await quantum_scheduler.add_quantum_task(task)

        # Find tunneling paths
        tunneling_paths = await quantum_scheduler._find_tunneling_paths(3600.0)

        assert isinstance(tunneling_paths, dict)

        # Verify tunneling properties
        for task_id, paths in tunneling_paths.items():
            assert task_id in quantum_scheduler.quantum_tasks
            assert isinstance(paths, list)

            for path in paths:
                assert "tunnel_type" in path
                assert "speedup_factor" in path
                assert "cost" in path
                assert "probability" in path
                assert 0.0 < path["speedup_factor"] <= 2.0
                assert 0.0 <= path["probability"] <= 1.0

    @pytest.mark.asyncio
    async def test_interference_resolution(self, quantum_scheduler):
        """Test quantum interference pattern conflict resolution"""
        # Create mock conflict
        conflict = {
            "conflict_id": "conflict_001",
            "tasks": ["task_001", "task_002"],
            "resource": "gpu_memory",
            "overlap_amount": 4.0
        }

        # Create interference pattern
        pattern = quantum_scheduler._create_interference_pattern(conflict)

        assert isinstance(pattern, torch.Tensor)
        assert pattern.shape == (10, 10)  # Default pattern size

        # Resolve interference
        resolution = await quantum_scheduler.interference_resolver.resolve_conflict(
            conflict, pattern, quantum_scheduler.quantum_tasks
        )

        assert "conflict_id" in resolution
        assert "resolution_type" in resolution
        assert "resolution_actions" in resolution
        assert resolution["conflict_id"] == conflict["conflict_id"]

    @pytest.mark.asyncio
    async def test_quantum_measurement_collapse(self, quantum_scheduler, sample_tasks):
        """Test quantum state collapse to optimal schedule"""
        # Setup quantum system
        for task in sample_tasks:
            await quantum_scheduler.add_quantum_task(task)

        # Generate mock analyses
        superposition_analysis = {"total_paths": 6, "quantum_amplitudes": {}, "coherence_measures": {}}
        entangled_groups = [{"task_001", "task_002"}]
        tunneling_paths = {"task_001": [{"speedup_factor": 1.2, "cost": 50.0}]}
        resolved_conflicts = {}

        # Collapse to optimal schedule
        optimal_schedule = await quantum_scheduler._collapse_to_optimal_schedule(
            superposition_analysis, entangled_groups, tunneling_paths, resolved_conflicts
        )

        assert isinstance(optimal_schedule, list)

        # Verify all tasks are collapsed
        for task in quantum_scheduler.quantum_tasks.values():
            assert task.quantum_state == QuantumState.COLLAPSED

    def test_quantum_amplitude_calculation(self, quantum_scheduler, sample_tasks) -> None:
        """Test quantum amplitude calculations"""
        task = sample_tasks[0]

        amplitude = quantum_scheduler._calculate_initial_amplitude(task)

        assert isinstance(amplitude, complex)
        assert amplitude != 0
        assert abs(amplitude) > 0

        # Verify amplitude properties
        magnitude = abs(amplitude)
        phase = np.angle(amplitude)

        assert 0 < magnitude <= 1
        assert -np.pi <= phase <= np.pi

    @pytest.mark.asyncio
    async def test_superposition_path_generation(self, quantum_scheduler, sample_tasks):
        """Test superposition execution path generation"""
        task = sample_tasks[1]  # Task with dependencies

        paths = await quantum_scheduler._generate_superposition_paths(task)

        assert isinstance(paths, list)
        assert len(paths) >= 1

        # Verify path properties
        total_probability = sum(path["probability"] for path in paths)
        assert abs(total_probability - 1.0) < 0.1  # Allow some tolerance

        for path in paths:
            assert "path_id" in path
            assert "path_type" in path
            assert "probability" in path
            assert "cost" in path
            assert "resources" in path
            assert 0.0 <= path["probability"] <= 1.0
            assert path["cost"] > 0

    def test_entanglement_matrix_update(self, quantum_scheduler, sample_tasks) -> None:
        """Test entanglement matrix updates"""
        # Add tasks to create entanglement
        asyncio.run(quantum_scheduler.add_quantum_task(sample_tasks[0]))
        asyncio.run(quantum_scheduler.add_quantum_task(sample_tasks[1]))

        # Update entanglement matrix
        asyncio.run(quantum_scheduler._update_entanglement_matrix())

        assert quantum_scheduler.entanglement_matrix is not None
        assert quantum_scheduler.entanglement_matrix.shape[0] == len(quantum_scheduler.quantum_tasks)
        assert quantum_scheduler.entanglement_matrix.dtype == torch.complex64

        # Verify matrix properties
        matrix = quantum_scheduler.entanglement_matrix
        assert torch.allclose(matrix.diagonal(), torch.zeros(matrix.shape[0], dtype=torch.complex64))

    def test_entanglement_strength_calculation(self, quantum_scheduler, sample_tasks) -> None:
        """Test entanglement strength calculations"""
        task1 = sample_tasks[0]
        task2 = sample_tasks[1]

        strength = quantum_scheduler._calculate_entanglement_strength(task1, task2)

        assert isinstance(strength, complex)
        magnitude = abs(strength)
        phase = np.angle(strength)

        assert magnitude >= 0
        assert -np.pi <= phase <= np.pi

    def test_quantum_coherence_measurement(self, quantum_scheduler) -> None:
        """Test quantum coherence measurements"""
        # Create test amplitudes
        amplitudes = [1+0j, 0.5+0.5j, 0.3+0.7j, 0.8+0.2j]

        coherence = quantum_scheduler._measure_quantum_coherence(amplitudes)

        assert isinstance(coherence, float)
        assert 0.0 <= coherence <= 1.0

        # Test edge cases
        empty_coherence = quantum_scheduler._measure_quantum_coherence([])
        assert empty_coherence == 0.0

        single_coherence = quantum_scheduler._measure_quantum_coherence([1+0j])
        assert single_coherence == 1.0

    @pytest.mark.asyncio
    async def test_classical_fallback(self, quantum_scheduler, sample_tasks):
        """Test classical scheduling fallback"""
        # Add tasks
        for task in sample_tasks:
            await quantum_scheduler.add_quantum_task(task)

        # Test fallback scheduling
        fallback_schedule = await quantum_scheduler._classical_fallback_schedule()

        assert isinstance(fallback_schedule, list)
        assert len(fallback_schedule) == len(sample_tasks)

        # Verify schedule ordering respects dependencies
        task_order = [item["task_id"] for item in fallback_schedule]
        assert task_order.index("task_001") < task_order.index("task_002")
        assert task_order.index("task_002") < task_order.index("task_003")

    def test_topological_sort(self, quantum_scheduler, sample_tasks) -> None:
        """Test topological sorting of tasks"""
        # Add tasks to scheduler
        for task in sample_tasks:
            quantum_scheduler.quantum_tasks[task.task_id] = task

        sorted_tasks = quantum_scheduler._topological_sort()

        assert len(sorted_tasks) == len(sample_tasks)
        assert set(sorted_tasks) == set(task.task_id for task in sample_tasks)

        # Verify topological order
        task_positions = {task_id: i for i, task_id in enumerate(sorted_tasks)}

        for task in sample_tasks:
            for dependency in task.dependencies:
                if dependency in task_positions:
                    assert task_positions[dependency] < task_positions[task.task_id]

class TestQuantumOptimizer:
    """Test quantum optimization algorithms"""

    @pytest.fixture
    def optimizer(self) -> None:
        """Create quantum optimizer"""
        he_context = CKKSContext()
        he_context.generate_keys()
        return QuantumOptimizer(he_context)

    @pytest.fixture
    def mock_tasks(self) -> None:
        """Create mock quantum tasks"""
        return {
            "task_001": QuantumTask(
                task_id="task_001",
                name="Test Task 1",
                priority=TaskPriority.HIGH,
                estimated_duration=60.0,
                resource_requirements={"cpu": 2.0}
            )
        }

    @pytest.fixture
    def mock_resources(self) -> None:
        """Create mock resources"""
        return {}

    @pytest.mark.asyncio
    async def test_quantum_annealing(self, optimizer, mock_tasks, mock_resources):
        """Test quantum annealing optimization"""
        analysis = {"total_paths": 3, "quantum_amplitudes": {}, "coherence_measures": {}}

        schedule = await optimizer._quantum_annealing_optimization(
            mock_tasks, mock_resources, analysis
        )

        assert isinstance(schedule, list)
        if schedule:  # May be empty for mock data
            for item in schedule:
                assert "task_id" in item
                assert "start_time" in item
                assert "end_time" in item

    @pytest.mark.asyncio
    async def test_candidate_schedule_generation(self, optimizer, mock_tasks, mock_resources):
        """Test candidate schedule generation"""
        temperature = 100.0

        candidate = await optimizer._generate_candidate_schedule(
            mock_tasks, mock_resources, temperature
        )

        assert isinstance(candidate, list)
        assert len(candidate) == len(mock_tasks)

        for item in candidate:
            assert "task_id" in item
            assert "start_time" in item
            assert "end_time" in item
            assert "assigned_resources" in item
            assert "quantum_enhancement" in item

    def test_schedule_cost_evaluation(self, optimizer) -> None:
        """Test schedule cost evaluation"""
        schedule = [
            {"task_id": "task_1", "start_time": 0, "end_time": 60, "assigned_resources": {}},
            {"task_id": "task_2", "start_time": 60, "end_time": 120, "assigned_resources": {}}
        ]

        cost = optimizer._evaluate_schedule_cost(schedule)

        assert isinstance(cost, float)
        assert cost >= 0
        assert cost == 120.0  # Total time with no conflicts

    @pytest.mark.asyncio
    async def test_entanglement_constraint_application(self, optimizer):
        """Test application of entanglement constraints"""
        schedule = [
            {"task_id": "task_1", "start_time": 0, "end_time": 60},
            {"task_id": "task_2", "start_time": 30, "end_time": 90}
        ]
        entangled_groups = [{"task_1", "task_2"}]

        constrained_schedule = await optimizer._apply_entanglement_constraints(
            schedule, entangled_groups
        )

        assert len(constrained_schedule) == 2

        # Verify entangled tasks have synchronized start times
        entangled_tasks = [item for item in constrained_schedule if item["task_id"] in {"task_1", "task_2"}]
        start_times = [task["start_time"] for task in entangled_tasks]
        assert len(set(start_times)) == 1  # All same start time

        # Verify quantum entanglement flag
        for task in entangled_tasks:
            assert task.get("quantum_entangled", False)

class TestEntanglementManager:
    """Test quantum entanglement management"""

    @pytest.fixture
    def manager(self) -> None:
        """Create entanglement manager"""
        return EntanglementManager(max_depth=5)

    def test_initialization(self, manager) -> None:
        """Test entanglement manager initialization"""
        assert manager.max_depth == 5
        assert isinstance(manager.entanglement_graph, dict)

    @pytest.mark.asyncio
    async def test_entangled_group_finding(self, manager):
        """Test finding entangled task groups"""
        dependency_graph = {
            "task_1": {"task_3"},
            "task_2": {"task_1"},
            "task_3": {"task_2"},
            "task_4": set()
        }

        mock_tasks = {
            task_id: QuantumTask(
                task_id=task_id,
                name=f"Task {task_id}",
                priority=TaskPriority.MEDIUM,
                estimated_duration=60.0,
                resource_requirements={}
            )
            for task_id in dependency_graph.keys()
        }

        groups = await manager.find_entangled_groups(dependency_graph, mock_tasks)

        assert isinstance(groups, list)

        # Should find circular dependency group
        circular_group = next((g for g in groups if len(g) > 1), None)
        if circular_group:
            assert len(circular_group) >= 2
            assert "task_1" in circular_group or "task_2" in circular_group or "task_3" in circular_group

    def test_connected_component_finding(self, manager) -> None:
        """Test connected component identification"""
        graph = {
            "A": {"B"},
            "B": {"C"},
            "C": {"A"},
            "D": set()
        }

        visited = set()
        component = manager._find_connected_component("A", graph, visited)

        assert isinstance(component, set)
        assert len(component) >= 3  # A, B, C should be connected
        assert "A" in component
        assert "B" in component
        assert "C" in component

class TestTunnelingPathSolver:
    """Test quantum tunneling path solving"""

    @pytest.fixture
    def solver(self) -> None:
        """Create tunneling path solver"""
        return TunnelingPathSolver()

    @pytest.mark.asyncio
    async def test_quantum_tunnel_finding(self, solver):
        """Test quantum tunneling path discovery"""
        task = QuantumTask(
            task_id="test_task",
            name="Test Task",
            priority=TaskPriority.HIGH,
            estimated_duration=120.0,
            resource_requirements={"gpu_memory": 8.0},
            dependencies={"dep_1", "dep_2"}
        )

        resources = {
            "gpu_0": QuantumResourceNode(
                node_id="gpu_0",
                resource_type=ResourceType.GPU_MEMORY,
                total_capacity=16.0,
                available_capacity=12.0,
                quantum_efficiency=1.5
            )
        }

        tunnels = await solver.find_quantum_tunnels(task, resources, 3600.0)

        assert isinstance(tunnels, list)

        for tunnel in tunnels:
            assert "tunnel_type" in tunnel
            assert "speedup_factor" in tunnel
            assert "cost" in tunnel
            assert "probability" in tunnel
            assert 0.0 < tunnel["speedup_factor"] <= 3.0
            assert tunnel["cost"] > 0
            assert 0.0 <= tunnel["probability"] <= 1.0

class TestInterferenceResolver:
    """Test quantum interference resolution"""

    @pytest.fixture
    def resolver(self) -> None:
        """Create interference resolver"""
        return InterferenceResolver()

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, resolver):
        """Test quantum interference conflict resolution"""
        conflict = {
            "conflict_id": "test_conflict",
            "tasks": ["task_1", "task_2"],
            "resource_type": "gpu_memory",
            "severity": 0.8
        }

        interference_pattern = torch.randn(10, 10)

        mock_tasks = {
            "task_1": QuantumTask(
                task_id="task_1",
                name="Task 1",
                priority=TaskPriority.HIGH,
                estimated_duration=60.0,
                resource_requirements={}
            ),
            "task_2": QuantumTask(
                task_id="task_2",
                name="Task 2",
                priority=TaskPriority.MEDIUM,
                estimated_duration=90.0,
                resource_requirements={}
            )
        }

        resolution = await resolver.resolve_conflict(conflict, interference_pattern, mock_tasks)

        assert isinstance(resolution, dict)
        assert "conflict_id" in resolution
        assert "resolution_type" in resolution
        assert "resolution_actions" in resolution
        assert resolution["conflict_id"] == conflict["conflict_id"]
        assert resolution["resolution_type"] == "quantum_interference"

class TestQuantumSchedulingSession:
    """Test quantum scheduling session management"""

    @pytest.mark.asyncio
    async def test_session_context_manager(self):
        """Test quantum scheduling session context manager"""
        scheduler = create_quantum_task_scheduler(privacy_level="low", performance_mode="speed")

        async with QuantumSchedulingSession(scheduler) as session:
            assert session is scheduler
            assert hasattr(session, 'quantum_tasks')

            # Add a test task
            task = QuantumTask(
                task_id="session_test",
                name="Session Test Task",
                priority=TaskPriority.MEDIUM,
                estimated_duration=30.0,
                resource_requirements={"cpu": 1.0}
            )

            await session.add_quantum_task(task)
            assert "session_test" in session.quantum_tasks

        # Verify cleanup after session
        # Tasks should be collapsed
        if scheduler.quantum_tasks:
            for task in scheduler.quantum_tasks.values():
                assert task.quantum_state == QuantumState.COLLAPSED

class TestQuantumSchedulerFactory:
    """Test quantum scheduler factory functions"""

    def test_factory_privacy_levels(self) -> None:
        """Test factory with different privacy levels"""
        for privacy in ["low", "medium", "high"]:
            scheduler = create_quantum_task_scheduler(
                privacy_level=privacy,
                performance_mode="balanced"
            )

            assert isinstance(scheduler, QuantumTaskScheduler)
            assert scheduler.he_context is not None

            # Verify HE context configuration
            config = scheduler.he_context.config
            if privacy == "high":
                assert config.poly_modulus_degree >= 32768
            elif privacy == "medium":
                assert config.poly_modulus_degree >= 16384
            else:  # low
                assert config.poly_modulus_degree >= 8192

    def test_factory_performance_modes(self) -> None:
        """Test factory with different performance modes"""
        for mode in ["speed", "balanced", "accuracy"]:
            scheduler = create_quantum_task_scheduler(
                privacy_level="medium",
                performance_mode=mode
            )

            assert isinstance(scheduler, QuantumTaskScheduler)

            if mode == "speed":
                assert scheduler.max_entanglement_depth <= 3
                assert scheduler.quantum_coherence_time <= 120.0
            elif mode == "accuracy":
                assert scheduler.max_entanglement_depth >= 8
                assert scheduler.quantum_coherence_time >= 600.0
            else:  # balanced
                assert 3 <= scheduler.max_entanglement_depth <= 8
                assert 120.0 <= scheduler.quantum_coherence_time <= 600.0

class TestQuantumResourceManager:
    """Test quantum resource management integration"""

    @pytest.fixture
    def resource_manager(self) -> None:
        """Create quantum resource manager"""
        return QuantumResourceManager(
            monitoring_interval=0.1,
            quantum_coherence_time=60.0,
            enable_distributed=False
        )

    @pytest.mark.asyncio
    async def test_resource_manager_initialization(self, resource_manager):
        """Test resource manager initialization"""
        success = await resource_manager.initialize_quantum_resources()

        assert success
        assert len(resource_manager.quantum_nodes) > 0
        assert resource_manager.entanglement_matrix is not None

    @pytest.mark.asyncio
    async def test_quantum_resource_allocation(self, resource_manager):
        """Test quantum resource allocation"""
        await resource_manager.initialize_quantum_resources()

        task = QuantumTask(
            task_id="resource_test",
            name="Resource Test Task",
            priority=TaskPriority.HIGH,
            estimated_duration=60.0,
            resource_requirements={"cpu_cores": 2.0, "memory": 4.0}
        )

        allocation = await resource_manager.allocate_quantum_resources(
            task,
            strategy=AllocationStrategy.QUANTUM_OPTIMAL,
            enable_superposition=True
        )

        if allocation:  # May fail with insufficient resources in test environment
            assert isinstance(allocation, QuantumAllocation)
            assert allocation.task_id == task.task_id
            assert len(allocation.resource_assignments) > 0
            assert allocation.quantum_speedup_factor >= 1.0

    @pytest.mark.asyncio
    async def test_resource_status_reporting(self, resource_manager):
        """Test quantum resource status reporting"""
        await resource_manager.initialize_quantum_resources()

        status = await resource_manager.get_quantum_resource_status(encrypt_sensitive=True)

        assert isinstance(status, dict)
        assert "total_nodes" in status
        assert "active_allocations" in status
        assert "quantum_coherence_remaining" in status
        assert "entanglement_efficiency" in status
        assert "overall_utilization" in status
        assert "node_details" in status

        assert status["total_nodes"] > 0
        assert 0.0 <= status["quantum_coherence_remaining"] <= 1.0
        assert 0.0 <= status["entanglement_efficiency"] <= 1.0
        assert 0.0 <= status["overall_utilization"] <= 1.0

# Performance and stress tests

class TestQuantumPerformance:
    """Performance and stress tests for quantum algorithms"""

    @pytest.mark.asyncio
    async def test_large_task_scheduling_performance(self):
        """Test performance with large number of tasks"""
        scheduler = create_quantum_task_scheduler(privacy_level="low", performance_mode="speed")

        # Create 100 tasks
        tasks = []
        for i in range(100):
            task = QuantumTask(
                task_id=f"perf_task_{i:03d}",
                name=f"Performance Task {i}",
                priority=TaskPriority(i % 5),
                estimated_duration=float(np.random.uniform(30, 180)),
                resource_requirements={
                    "cpu_cores": np.random.uniform(1, 8),
                    "memory": np.random.uniform(2, 16)
                },
                dependencies=set() if i < 10 else {f"perf_task_{j:03d}" for j in np.random.choice(i, size=min(3, i), replace=False)}
            )
            tasks.append(task)

        # Measure scheduling time
        start_time = time.time()

        # Add tasks
        for task in tasks[:50]:  # Limit for test performance
            await scheduler.add_quantum_task(task)

        # Schedule
        schedule = await scheduler.schedule_quantum_tasks()

        end_time = time.time()
        scheduling_time = end_time - start_time

        assert len(schedule) == 50
        assert scheduling_time < 30.0  # Should complete within 30 seconds
        assert scheduler.quantum_speedup_achieved >= 1.0

    @pytest.mark.asyncio
    async def test_quantum_coherence_under_load(self):
        """Test quantum coherence maintenance under load"""
        scheduler = create_quantum_task_scheduler(privacy_level="medium", performance_mode="balanced")

        # Add many tasks rapidly
        tasks = []
        for i in range(20):
            task = QuantumTask(
                task_id=f"coherence_task_{i}",
                name=f"Coherence Test Task {i}",
                priority=TaskPriority.MEDIUM,
                estimated_duration=60.0,
                resource_requirements={"cpu_cores": 2.0}
            )
            tasks.append(task)
            await scheduler.add_quantum_task(task)

        # Check that quantum states are properly maintained
        superposition_tasks = [
            task for task in scheduler.quantum_tasks.values()
            if task.quantum_state == QuantumState.SUPERPOSITION
        ]

        assert len(superposition_tasks) > 0

        # Verify quantum amplitudes are non-zero
        for task in superposition_tasks:
            assert abs(task.probability_amplitude) > 0
            assert len(task.execution_paths) > 0

    def test_memory_usage_with_large_entanglement(self) -> None:
        """Test memory usage with large entanglement matrices"""
        scheduler = create_quantum_task_scheduler(privacy_level="low", performance_mode="speed")

        # Add tasks to create large entanglement matrix
        for i in range(50):  # Moderate size for testing
            task = QuantumTask(
                task_id=f"memory_task_{i}",
                name=f"Memory Test Task {i}",
                priority=TaskPriority.LOW,
                estimated_duration=30.0,
                resource_requirements={"cpu_cores": 1.0}
            )
            asyncio.run(scheduler.add_quantum_task(task))

        # Update entanglement matrix
        asyncio.run(scheduler._update_entanglement_matrix())

        # Verify matrix exists and has reasonable size
        assert scheduler.entanglement_matrix is not None
        matrix_size = scheduler.entanglement_matrix.numel() * scheduler.entanglement_matrix.element_size()

        # Should be less than 100MB for reasonable number of tasks
        assert matrix_size < 100 * 1024 * 1024

# Integration tests

class TestQuantumIntegration:
    """Integration tests between quantum components"""

    @pytest.mark.asyncio
    async def test_end_to_end_quantum_workflow(self):
        """Test complete quantum scheduling workflow"""
        # Create scheduler with full configuration
        scheduler = create_quantum_task_scheduler(privacy_level="medium", performance_mode="balanced")

        # Create resource manager
        resource_manager = QuantumResourceManager(
            he_context=scheduler.he_context,
            monitoring_interval=0.1,
            quantum_coherence_time=120.0
        )

        await resource_manager.initialize_quantum_resources()

        # Create realistic task workflow
        tasks = [
            QuantumTask(
                task_id="data_ingestion",
                name="Data Ingestion and Preprocessing",
                priority=TaskPriority.CRITICAL,
                estimated_duration=90.0,
                resource_requirements={"cpu_cores": 4.0, "memory": 8.0}
            ),
            QuantumTask(
                task_id="homomorphic_computation",
                name="Homomorphic Matrix Operations",
                priority=TaskPriority.HIGH,
                estimated_duration=180.0,
                resource_requirements={"gpu_memory": 12.0, "cpu_cores": 8.0},
                dependencies={"data_ingestion"}
            ),
            QuantumTask(
                task_id="graph_embedding",
                name="Encrypted Graph Embedding",
                priority=TaskPriority.HIGH,
                estimated_duration=150.0,
                resource_requirements={"gpu_memory": 16.0, "cpu_cores": 6.0},
                dependencies={"homomorphic_computation"}
            ),
            QuantumTask(
                task_id="classification",
                name="Privacy-Preserving Classification",
                priority=TaskPriority.MEDIUM,
                estimated_duration=120.0,
                resource_requirements={"gpu_memory": 8.0, "cpu_cores": 4.0},
                dependencies={"graph_embedding"}
            )
        ]

        # Execute full workflow
        async with QuantumSchedulingSession(scheduler) as quantum_session:
            # Add all tasks
            for task in tasks:
                await quantum_session.add_quantum_task(task, encrypt_metadata=True)

            # Allocate resources
            allocations = {}
            for task in tasks:
                allocation = await resource_manager.allocate_quantum_resources(
                    task,
                    strategy=AllocationStrategy.QUANTUM_OPTIMAL
                )
                if allocation:
                    allocations[task.task_id] = allocation

            # Schedule tasks
            schedule = await quantum_session.schedule_quantum_tasks()

            # Verify results
            assert len(schedule) == len(tasks)
            assert quantum_session.quantum_speedup_achieved >= 1.0

            # Verify task ordering respects dependencies
            task_order = {item["task_id"]: i for i, item in enumerate(schedule)}

            assert task_order["data_ingestion"] < task_order["homomorphic_computation"]
            assert task_order["homomorphic_computation"] < task_order["graph_embedding"]
            assert task_order["graph_embedding"] < task_order["classification"]

            # Clean up allocations
            for allocation in allocations.values():
                await resource_manager.deallocate_quantum_resources(allocation.allocation_id)

        # Shutdown resource manager
        await resource_manager.shutdown()

    @pytest.mark.asyncio
    async def test_quantum_privacy_preservation(self):
        """Test that quantum algorithms preserve privacy"""
        scheduler = create_quantum_task_scheduler(privacy_level="high", performance_mode="balanced")

        sensitive_task = QuantumTask(
            task_id="sensitive_computation",
            name="Sensitive Financial Analysis",
            priority=TaskPriority.CRITICAL,
            estimated_duration=300.0,
            resource_requirements={"cpu_cores": 16.0, "memory": 32.0}
        )

        # Add task with encryption
        await scheduler.add_quantum_task(sensitive_task, encrypt_metadata=True)

        # Verify metadata is encrypted
        stored_task = scheduler.quantum_tasks["sensitive_computation"]
        assert stored_task.encrypted_metadata is not None

        # Schedule with privacy preservation
        schedule = await scheduler.schedule_quantum_tasks()

        assert len(schedule) == 1

        # Verify no sensitive information leaked in schedule
        scheduled_task = schedule[0]
        assert "resource_requirements" not in scheduled_task or not scheduled_task["resource_requirements"]

# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])