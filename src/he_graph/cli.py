#!/usr/bin/env python3
"""
HE-Graph-Embeddings Command Line Interface
Provides easy access to homomorphic graph neural network operations
"""


import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Import our HE graph library
try:
    from ..python.he_graph import (
        CKKSContext, HEConfig, HEGraphSAGE, HEGAT,
        SecurityEstimator, NoiseTracker
    )
    from .quantum_he_softmax import QuantumEnhancedHESoftmax, HEGraphAttention
except ImportError as e:
    print(f"Import error: {e}")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from python.he_graph import (
        CKKSContext, HEConfig, HEGraphSAGE, HEGAT,
        SecurityEstimator, NoiseTracker
    )
    from quantum_he_softmax import QuantumEnhancedHESoftmax, HEGraphAttention
    from robust_error_handling import get_system_health, reset_health_metrics
    from he_health_monitor import get_he_health_status, get_he_recommendations
    from config_validator import EnhancedConfigValidator, DeploymentEnvironment, PerformanceProfile


import numpy as np

def create_sample_graph(num_nodes: int = 34, feature_dim: int = 128) -> tuple:
    """Create a sample graph for testing (Karate Club-style)"""
    print(f"Creating sample graph with {num_nodes} nodes, {feature_dim} features")

    # Create simple ring graph + some random connections
    edges = []
    for i in range(num_nodes):
        edges.append([i, (i + 1) % num_nodes])  # Ring
        if i % 3 == 0 and i + 3 < num_nodes:
            edges.append([i, i + 3])  # Some shortcuts

    # Convert to numpy array format
    edge_index = np.array(edges).T

    # Generate random node features
    features = np.random.randn(num_nodes, feature_dim).astype(np.float32)

    print(f"Graph created: {len(edges)} edges, feature shape: {features.shape}")
    return edge_index, features

def benchmark_mode(args):
    """Run performance benchmarks"""
    print("🚀 Running HE-Graph-Embeddings Benchmarks")
    print("=" * 50)

    # Test different configurations
    configs = [
        {"name": "Small (128-bit)", "nodes": 100, "features": 32, "security": 128},
        {"name": "Medium (128-bit)", "nodes": 500, "features": 64, "security": 128},
        {"name": "Large (128-bit)", "nodes": 1000, "features": 128, "security": 128},
    ]

    results = []

    for config in configs:
        print(f"\n📊 Testing {config['name']}:")
        print(f"   Nodes: {config['nodes']}, Features: {config['features']}")

        # Create test data
        edge_index, features = create_sample_graph(config['nodes'], config['features'])

        # Setup HE context
        he_config = HEConfig(
            poly_modulus_degree=16384 if config['nodes'] < 500 else 32768,
            security_level=config['security']
        )

        context = CKKSContext(he_config)
        context.generate_keys()

        # Create model
        model = HEGraphSAGE(
            in_channels=config['features'],
            hidden_channels=32,
            out_channels=16,
            num_layers=2,
            context=context
        )

        # Benchmark encryption
        start_time = time.time()
        enc_features = context.encrypt(features)
        encrypt_time = time.time() - start_time

        # Benchmark forward pass
        start_time = time.time()
        try:
            enc_output = model(enc_features, edge_index)
            forward_time = time.time() - start_time
            success = True
        except Exception as e:
            forward_time = 0
            success = False
            print(f"   ❌ Forward pass failed: {e}")

        # Benchmark decryption
        if success:
            start_time = time.time()
            output = context.decrypt(enc_output)
            decrypt_time = time.time() - start_time
        else:
            decrypt_time = 0

        result = {
            'config': config['name'],
            'nodes': config['nodes'],
            'features': config['features'],
            'encrypt_time': encrypt_time,
            'forward_time': forward_time,
            'decrypt_time': decrypt_time,
            'total_time': encrypt_time + forward_time + decrypt_time,
            'success': success
        }
        results.append(result)

        if success:
            print(f"   ✅ Encryption: {encrypt_time:.3f}s")
            print(f"   ✅ Forward pass: {forward_time:.3f}s")
            print(f"   ✅ Decryption: {decrypt_time:.3f}s")
            print(f"   🎯 Total time: {result['total_time']:.3f}s")

    # Summary
    print(f"\n📈 Benchmark Summary:")
    print("-" * 50)
    for result in results:
        if result['success']:
            print(f"{result['config']:20} | {result['total_time']:8.3f}s | ✅")
        else:
            print(f"{result['config']:20} | {'failed':8} | ❌")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

def demo_mode(args):
    """Run interactive demonstration"""
    print("🎭 HE-Graph-Embeddings Demo Mode")
    print("=" * 40)

    # Create sample data
    print("\n1. Creating sample graph...")
    edge_index, features = create_sample_graph(args.nodes, args.features)

    # Setup encryption
    print("\n2. Setting up homomorphic encryption...")
    he_config = HEConfig(
        poly_modulus_degree=16384,
        security_level=128,
        scale=2**40
    )

    # Validate security
    estimator = SecurityEstimator()
    params = {
        'poly_degree': he_config.poly_modulus_degree,
        'coeff_modulus_bits': he_config.coeff_modulus_bits,
        'scale': he_config.scale
    }
    security_bits = estimator.estimate(params)
    print(f"   🔒 Estimated security level: {security_bits} bits")

    context = CKKSContext(he_config)
    print("   🔑 Generating encryption keys...")
    context.generate_keys()
    print("   ✅ Encryption context ready")

    # Create model
    print(f"\n3. Creating {args.model.upper()} model...")
    if args.model == 'graphsage':
        model = HEGraphSAGE(
            in_channels=args.features,
            hidden_channels=64,
            out_channels=32,
            num_layers=2,
            aggregator='mean',
            context=context
        )
    elif args.model == 'gat':
        model = HEGAT(
            in_channels=args.features,
            out_channels=32,
            heads=4,
            context=context
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"   ✅ {args.model.upper()} model created")

    # Encrypt features
    print("\n4. Encrypting node features...")
    with NoiseTracker() as tracker:
        enc_features = context.encrypt(features)
        print(f"   📊 Initial noise budget: {tracker.get_noise_budget():.2f} bits")

    # Forward pass
    print("\n5. Running encrypted forward pass...")
    start_time = time.time()
    with NoiseTracker() as tracker:
        enc_output = model(enc_features, edge_index)
        forward_time = time.time() - start_time
        final_noise = tracker.get_noise_budget()

    print(f"   ⚡ Forward pass completed in {forward_time:.3f}s")
    print(f"   📊 Final noise budget: {final_noise:.2f} bits")

    # Decrypt result
    print("\n6. Decrypting results...")
    output = context.decrypt(enc_output)
    print(f"   ✅ Output shape: {output.shape}")
    print(f"   📈 Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("\n🎉 Demo completed successfully!")

    if args.save_output:
        output_data = {
            'model_type': args.model,
            'graph_nodes': args.nodes,
            'feature_dim': args.features,
            'output_shape': list(output.shape),
            'output_stats': {
                'mean': float(output.mean()),
                'std': float(output.std()),
                'min': float(output.min()),
                'max': float(output.max())
            },
            'performance': {
                'forward_time': forward_time,
                'final_noise_budget': final_noise
            }
        }

        with open(args.save_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 Results saved to {args.save_output}")

def security_mode(args):
    """Analyze security parameters"""
    print("🔒 HE-Graph-Embeddings Security Analysis")
    print("=" * 45)

    estimator = SecurityEstimator()

    # Test different parameter sets
    test_configs = [
        {"name": "Conservative", "poly": 16384, "coeff_bits": [60, 40, 40, 60]},
        {"name": "Standard", "poly": 32768, "coeff_bits": [60, 40, 40, 40, 40, 60]},
        {"name": "High-Depth", "poly": 32768, "coeff_bits": [60, 40, 40, 40, 40, 40, 40, 60]},
    ]

    print("\n📋 Security Level Analysis:")
    print("-" * 60)
    print(f"{'Config':15} {'Poly Degree':12} {'Total Bits':12} {'Security':10}")
    print("-" * 60)

    for config in test_configs:
        params = {
            'poly_degree': config['poly'],
            'coeff_modulus_bits': config['coeff_bits']
        }

        security_bits = estimator.estimate(params)
        total_bits = sum(config['coeff_bits'])

        print(f"{config['name']:15} {config['poly']:12} {total_bits:12} {security_bits:10}")

    # Recommendation
    print(f"\n💡 Parameter Recommendations:")
    for depth in [5, 10, 15]:
        try:
            recommended = estimator.recommend(
                security_bits=128,
                multiplicative_depth=depth,
                precision_bits=30
            )
            print(f"   Depth {depth:2d}: poly_degree={recommended.poly_modulus_degree}, "
                    f"modulus_count={len(recommended.coeff_modulus_bits)}")
        except Exception as e:
            print(f"   Depth {depth:2d}: {e}")


def quantum_test_mode(args):
    """Test quantum-enhanced homomorphic operations"""
    print("🚀 Quantum-Enhanced HE Test Mode")
    print("=" * 50)
    
    print(f"\n1. Testing Quantum-Enhanced Softmax (Order {args.order})...")
    
    # Initialize quantum softmax
    quantum_softmax = QuantumEnhancedHESoftmax(
        approximation_order=args.order,
        quantum_enhancement=True
    )
    
    # Test with various input patterns
    test_cases = [
        ("Random inputs", torch.randn(args.nodes, 10)),
        ("Large values", torch.randn(args.nodes, 10) * 5),
        ("Small values", torch.randn(args.nodes, 10) * 0.1),
        ("Uniform inputs", torch.ones(args.nodes, 10)),
    ]
    
    print("   Testing approximation quality:")
    total_error = 0
    
    for name, test_input in test_cases:
        error = quantum_softmax.get_approximation_error(test_input)
        total_error += error
        print(f"   • {name}: MSE = {error:.6f}")
    
    avg_error = total_error / len(test_cases)
    if avg_error < 0.01:
        quality = "✅ EXCELLENT"
    elif avg_error < 0.05:
        quality = "✅ GOOD"
    elif avg_error < 0.1:
        quality = "⚠️  ACCEPTABLE"
    else:
        quality = "❌ POOR"
    
    print(f"   Average approximation error: {avg_error:.6f} {quality}")
    
    print(f"\n2. Testing HE Graph Attention...")
    
    # Create test graph
    import torch
    torch.manual_seed(42)  # For reproducible results
    
    # Generate random graph
    num_nodes = args.nodes
    features_dim = 64
    x = torch.randn(num_nodes, features_dim)
    
    # Random edges (ensuring connectivity)
    num_edges = min(num_nodes * 2, 100)  # Reasonable number of edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Initialize HE Graph Attention
    he_attention = HEGraphAttention(
        in_features=features_dim,
        out_features=features_dim,
        heads=4,
        use_quantum_softmax=True
    )
    
    print(f"   Graph: {num_nodes} nodes, {num_edges} edges, {features_dim} features")
    
    # Forward pass timing
    start_time = time.time()
    try:
        output = he_attention(x, edge_index)
        forward_time = time.time() - start_time
        
        print(f"   ✅ Forward pass successful: {forward_time:.3f}s")
        print(f"   Output shape: {output.shape}")
        
        # Basic sanity checks
        assert output.shape == x.shape, "Output shape should match input"
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert torch.isfinite(output).all(), "Output should be finite"
        
        print("   ✅ All sanity checks passed")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return
    
    print(f"\n3. Performance Summary:")
    print(f"   • Quantum softmax quality: {quality}")
    print(f"   • Graph attention latency: {forward_time:.3f}s")
    print(f"   • Nodes processed: {num_nodes}")
    print(f"   • Features per node: {features_dim}")
    
    print(f"\n🎯 Generation 1 Test Complete!")
    print("   This demonstrates basic quantum-enhanced HE functionality.")
    print("   Ready for Generation 2 robustness enhancements!")


def health_monitoring_mode(args):
    """System health monitoring and diagnostics"""
    print("🛡️ HE-Graph System Health Monitor")
    print("=" * 50)
    
    if args.reset:
        reset_health_metrics()
        print("✅ Health metrics reset successfully")
        return
    
    # Get overall system health
    system_health = get_system_health()
    print(f"\n📊 System Health Status: {system_health['status'].upper()}")
    print(f"   • Uptime: {system_health['uptime_seconds']:.1f} seconds")
    print(f"   • Success rate: {system_health['success_rate']:.2%}")
    print(f"   • Total operations: {system_health['total_operations']}")
    print(f"   • Total errors: {system_health['total_errors']}")
    
    # Get HE-specific health
    try:
        he_health = get_he_health_status()
        print(f"\n🔐 HE Operations Health:")
        print(f"   • Noise status: {he_health.get('noise_status', 'unknown').upper()}")
        if he_health.get('noise_budget_remaining'):
            print(f"   • Noise budget: {he_health['noise_budget_remaining']:.2f} bits")
        print(f"   • Operations since refresh: {he_health.get('operations_since_refresh', 0)}")
        print(f"   • Average operation time: {he_health.get('average_recent_duration', 0):.3f}s")
        
        # Get recommendations
        recommendations = get_he_recommendations()
        if recommendations:
            print(f"\n💡 Optimization Recommendations ({len(recommendations)}):")
            for rec in recommendations:
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec['priority'], "ℹ️")
                print(f"   {priority_icon} {rec['priority'].upper()}: {rec['message']}")
        else:
            print("\n✅ No optimization recommendations at this time")
        
    except ImportError:
        print("\n⚠️  HE health monitoring not available (missing dependencies)")
    except Exception as e:
        print(f"\n❌ Error getting HE health status: {e}")
    
    # Export metrics if requested
    if args.export:
        try:
            import json
            from he_health_monitor import export_he_metrics
            
            metrics = export_he_metrics()
            with open(args.export, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"\n📁 Health metrics exported to {args.export}")
            
        except Exception as e:
            print(f"\n❌ Failed to export metrics: {e}")


def config_validation_mode(args):
    """Configuration validation and optimization"""
    print("🛡️ HE Configuration Validation")
    print("=" * 50)
    
    try:
        validator = EnhancedConfigValidator()
        
        # Load configuration
        if args.config_file:
            try:
                import json
                with open(args.config_file, 'r') as f:
                    config = json.load(f)
                print(f"📁 Loaded configuration from {args.config_file}")
            except Exception as e:
                print(f"❌ Failed to load config file: {e}")
                return
        else:
            # Use default configuration for validation
            config = {
                'poly_modulus_degree': 16384,
                'coeff_modulus_bits': [50, 40, 40, 50],
                'scale': 2**40,
                'security_level': 128
            }
            print("📋 Using default configuration for validation")
        
        # Parse environment and profile
        env_map = {
            'development': DeploymentEnvironment.DEVELOPMENT,
            'testing': DeploymentEnvironment.TESTING,
            'staging': DeploymentEnvironment.STAGING,
            'production': DeploymentEnvironment.PRODUCTION
        }
        
        profile_map = {
            'memory_optimized': PerformanceProfile.MEMORY_OPTIMIZED,
            'speed_optimized': PerformanceProfile.SPEED_OPTIMIZED,
            'balanced': PerformanceProfile.BALANCED,
            'precision_optimized': PerformanceProfile.PRECISION_OPTIMIZED
        }
        
        environment = env_map[args.environment]
        profile = profile_map[args.profile]
        
        print(f"🎯 Target: {args.environment} environment, {args.profile} profile")
        
        # Validate configuration
        result = validator.validate_he_config(config, environment, profile)
        
        # Display results
        print(f"\n📊 Validation Results:")
        if result.is_valid:
            print("   ✅ Configuration is VALID")
        else:
            print("   ❌ Configuration is INVALID")
        
        if result.errors:
            print(f"\n🚨 Errors ({len(result.errors)}):")
            for error in result.errors:
                print(f"   • {error}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"   • {warning}")
        
        if result.recommendations:
            print(f"\n💡 Recommendations ({len(result.recommendations)}):")
            for rec in result.recommendations:
                print(f"   • {rec}")
        
        # Show optimized configuration if available
        if result.optimized_config:
            print(f"\n🎯 Optimized Configuration:")
            for key, value in result.optimized_config.items():
                if key in config and config[key] != value:
                    print(f"   • {key}: {config[key]} → {value}")
                else:
                    print(f"   • {key}: {value}")
        
        # Show recommended configuration for this environment/profile
        print(f"\n🏆 Recommended Configuration for {args.environment} + {args.profile}:")
        recommended = validator.get_recommended_config(environment, profile)
        for key, value in recommended.items():
            if key not in ['environment', 'performance_profile']:
                print(f"   • {key}: {value}")
        
    except ImportError as e:
        print(f"❌ Configuration validation not available: {e}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")


async def comprehensive_test_mode(args):
    """Run comprehensive test suite"""
    print("🔬 HE-Graph Comprehensive Testing Suite")
    print("=" * 50)
    
    try:
        from comprehensive_testing import (
            ComprehensiveTestRunner, TestCategory,
            create_integration_test_suite, create_performance_test_suite, create_security_test_suite
        )
        
        # Parse categories
        categories = None
        if args.categories:
            categories = [TestCategory(cat) for cat in args.categories]
            print(f"🎯 Running categories: {', '.join(args.categories)}")
        else:
            print("🎯 Running all test categories")
        
        # Create test runner
        runner = ComprehensiveTestRunner(
            max_parallel_tests=args.max_workers,
            enable_monitoring=True
        )
        
        # Register test suites
        print(f"\n📋 Registering test suites...")
        runner.register_test_suite(create_integration_test_suite())
        runner.register_test_suite(create_performance_test_suite()) 
        runner.register_test_suite(create_security_test_suite())
        
        print(f"✅ Registered {len(runner.test_suites)} test suites")
        
        # Run tests
        print(f"\n🚀 Starting test execution...")
        start_time = time.time()
        
        report = await runner.run_all_tests(
            categories=categories,
            parallel=args.parallel
        )
        
        execution_time = time.time() - start_time
        
        # Display results
        print(f"\n" + "="*60)
        print("📊 TEST EXECUTION SUMMARY")
        print("="*60)
        
        summary = report['summary']
        print(f"⏱️  Execution Time: {execution_time:.2f}s")
        print(f"📈 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"🎯 Success Rate: {summary['success_rate']:.2%}")
        print(f"⚡ Avg Duration: {summary['average_duration']:.3f}s per test")
        
        # Category breakdown
        if report['results_by_category']:
            print(f"\n📋 Results by Category:")
            for category, count in report['results_by_category'].items():
                print(f"   • {category.title()}: {count} tests")
        
        # Severity breakdown
        if report['results_by_severity']:
            print(f"\n⚠️  Results by Severity:")
            severity_icons = {
                'pass': '✅', 'warning': '⚠️', 'fail': '❌', 'critical': '🚨'
            }
            for severity, count in report['results_by_severity'].items():
                icon = severity_icons.get(severity, '❓')
                print(f"   {icon} {severity.title()}: {count}")
        
        # Failed test details
        failed_results = [
            r for r in report['detailed_results'] 
            if r['severity'] in ['fail', 'critical']
        ]
        
        if failed_results:
            print(f"\n🚨 Failed Test Details:")
            for result in failed_results:
                print(f"   • {result['test_name']}: {result['message']}")
                if result.get('details'):
                    for key, value in result['details'].items():
                        print(f"     - {key}: {value}")
        
        # Performance insights
        if 'performance' in report['results_by_category']:
            perf_results = [
                r for r in report['detailed_results'] 
                if r['category'] == 'performance'
            ]
            if perf_results:
                print(f"\n⚡ Performance Insights:")
                for result in perf_results:
                    if result['severity'] == 'pass':
                        print(f"   ✅ {result['test_name']}: {result['message']}")
                    else:
                        print(f"   ⚠️  {result['test_name']}: {result['message']}")
        
        # Overall assessment
        overall_status = "PASS" if summary['success_rate'] >= 0.8 else "FAIL"
        status_icon = "✅" if overall_status == "PASS" else "❌"
        
        print(f"\n{status_icon} OVERALL STATUS: {overall_status}")
        
        if summary['success_rate'] >= 0.95:
            print("🏆 Excellent! All systems operating optimally.")
        elif summary['success_rate'] >= 0.8:
            print("👍 Good! Most systems functioning correctly.")
        elif summary['success_rate'] >= 0.6:
            print("⚠️  Warning! Some systems need attention.")
        else:
            print("🚨 Critical! Multiple system failures detected.")
        
        # Save results if requested
        if args.output:
            try:
                import json
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"\n📁 Results saved to {args.output}")
            except Exception as e:
                print(f"\n❌ Failed to save results: {e}")
        
        print("="*60)
        
        # Return non-zero exit code if tests failed
        if overall_status == "FAIL":
            import sys
            sys.exit(1)
        
    except ImportError as e:
        print(f"❌ Comprehensive testing not available: {e}")
        print("   Make sure all dependencies are installed.")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="HE-Graph-Embeddings: Privacy-preserving graph neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    he-graph demo --nodes 100 --features 64 --model graphsage
    he-graph benchmark --output results.json
    he-graph security
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demonstration')
    demo_parser.add_argument('--nodes', type=int, default=34,
                            help='Number of nodes in test graph (default: 34)')
    demo_parser.add_argument('--features', type=int, default=128,
                            help='Feature dimension (default: 128)')
    demo_parser.add_argument('--model', choices=['graphsage', 'gat'], default='graphsage',
                            help='Model type to use (default: graphsage)')
    demo_parser.add_argument('--save-output', type=str,
                            help='Save demo results to JSON file')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--output', '-o', type=str, default='benchmark_results.json',
                                help='Output file for results (default: benchmark_results.json)')

    # Security command
    security_parser = subparsers.add_parser('security', help='Analyze security parameters')
    
    # Quantum test command (Generation 1 addition)
    quantum_parser = subparsers.add_parser('quantum-test', help='Test quantum-enhanced features')
    quantum_parser.add_argument('--order', type=int, default=3,
                               help='Polynomial approximation order (default: 3)')
    quantum_parser.add_argument('--nodes', type=int, default=50,
                               help='Number of nodes for test graph (default: 50)')
    
    # Health monitoring command (Generation 2 addition)
    health_parser = subparsers.add_parser('health', help='System health monitoring and diagnostics')
    health_parser.add_argument('--export', type=str,
                              help='Export health metrics to JSON file')
    health_parser.add_argument('--reset', action='store_true',
                              help='Reset health monitoring metrics')
    
    # Config validation command (Generation 2 addition)
    config_parser = subparsers.add_parser('validate-config', help='Validate HE configuration')
    config_parser.add_argument('--config-file', type=str,
                              help='JSON file containing HE configuration')
    config_parser.add_argument('--environment', choices=['development', 'testing', 'staging', 'production'],
                              default='production', help='Deployment environment')
    config_parser.add_argument('--profile', choices=['memory_optimized', 'speed_optimized', 'balanced', 'precision_optimized'],
                              default='balanced', help='Performance profile')
    
    # Comprehensive testing command (Generation 3 addition)
    test_parser = subparsers.add_parser('test', help='Run comprehensive test suite')
    test_parser.add_argument('--categories', nargs='+', 
                            choices=['unit', 'integration', 'performance', 'security', 'stress', 'compatibility'],
                            help='Test categories to run (default: all)')
    test_parser.add_argument('--parallel', action='store_true', default=True,
                            help='Run tests in parallel (default: True)')
    test_parser.add_argument('--output', type=str,
                            help='Output file for test results JSON')
    test_parser.add_argument('--max-workers', type=int, default=4,
                            help='Maximum parallel test workers')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'demo':
            demo_mode(args)
        elif args.command == 'benchmark':
            benchmark_mode(args)
        elif args.command == 'security':
            security_mode(args)
        elif args.command == 'quantum-test':
            quantum_test_mode(args)
        elif args.command == 'health':
            health_monitoring_mode(args)
        elif args.command == 'validate-config':
            config_validation_mode(args)
        elif args.command == 'test':
            import asyncio
            asyncio.run(comprehensive_test_mode(args))
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()