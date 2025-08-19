#!/usr/bin/env python3
"""
Performance optimization validation script for HE-Graph-Embeddings

Validates performance configurations and optimization settings for production deployment.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


class PerformanceValidator:
    """Validates performance optimization configurations and benchmarks"""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.benchmark_dir = self.repo_root / "benchmark_results"
        self.src_dir = self.repo_root / "src"
        self.validation_results = {}
        
    def validate_benchmark_results(self) -> Dict[str, Any]:
        """Validate benchmark results and performance metrics"""
        print("📊 Validating benchmark results...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "benchmarks_found": [],
            "performance_metrics": {}
        }
        
        # Check for benchmark files
        benchmark_files = [
            self.benchmark_dir / "benchmark_analysis.json",
            self.benchmark_dir / "benchmark_results.json",
            self.benchmark_dir / "benchmark_specifications.json"
        ]
        
        for benchmark_file in benchmark_files:
            if benchmark_file.exists():
                try:
                    with open(benchmark_file, 'r') as f:
                        data = json.load(f)
                        
                    results["benchmarks_found"].append(benchmark_file.name)
                    
                    # Validate benchmark analysis
                    if benchmark_file.name == "benchmark_analysis.json":
                        self._validate_benchmark_analysis(data, results)
                        
                except Exception as e:
                    results["errors"].append(f"Error reading {benchmark_file.name}: {str(e)}")
                    results["status"] = "FAIL"
            else:
                results["warnings"].append(f"Benchmark file not found: {benchmark_file.name}")
                
        return results
        
    def _validate_benchmark_analysis(self, data: Dict[str, Any], results: Dict[str, Any]):
        """Validate benchmark analysis data"""
        
        # Check overall performance metrics
        overall_stats = data.get("overall_stats", {})
        if overall_stats:
            avg_improvement = overall_stats.get("avg_improvement", 0)
            best_method = overall_stats.get("best_method", "")
            total_benchmarks = overall_stats.get("total_benchmarks", 0)
            
            results["performance_metrics"] = {
                "avg_improvement": avg_improvement,
                "best_method": best_method,
                "total_benchmarks": total_benchmarks
            }
            
            # Validate performance thresholds
            if avg_improvement < 0.5:  # 50% improvement threshold
                results["warnings"].append(f"Average improvement ({avg_improvement:.2%}) below 50% threshold")
            else:
                print(f"  ✅ Average improvement: {avg_improvement:.2%}")
                
            if total_benchmarks < 10:
                results["warnings"].append(f"Limited benchmark coverage: {total_benchmarks} benchmarks")
            else:
                print(f"  ✅ Comprehensive benchmark coverage: {total_benchmarks} benchmarks")
                
        # Check method performance
        method_stats = data.get("method_stats", {})
        if method_stats and "graph_aware_adaptive" in method_stats:
            adaptive_stats = method_stats["graph_aware_adaptive"]
            wins = adaptive_stats.get("wins", 0)
            sample_size = adaptive_stats.get("sample_size", 0)
            
            if wins / sample_size >= 0.7:  # 70% win rate threshold
                print(f"  ✅ Graph-aware adaptive method win rate: {wins}/{sample_size}")
            else:
                results["warnings"].append(f"Graph-aware adaptive win rate below 70%: {wins}/{sample_size}")
                
    def validate_optimization_configurations(self) -> Dict[str, Any]:
        """Validate optimization engine configurations"""
        print("⚙️ Validating optimization configurations...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "optimization_features": []
        }
        
        # Check optimization engine files
        optimization_files = [
            self.src_dir / "optimization" / "production_optimization_engine.py",
            self.src_dir / "quantum" / "breakthrough_optimization_engine.py",
            self.src_dir / "utils" / "performance.py",
            self.src_dir / "utils" / "auto_scaling.py"
        ]
        
        for opt_file in optimization_files:
            if opt_file.exists():
                try:
                    with open(opt_file, 'r') as f:
                        content = f.read()
                        
                    # Check for key optimization features
                    optimization_features = [
                        "OptimizationTarget",
                        "ScalingPolicy", 
                        "auto_scaling",
                        "performance",
                        "throughput",
                        "latency",
                        "gpu_utilization",
                        "memory_efficiency"
                    ]
                    
                    found_features = []
                    for feature in optimization_features:
                        if feature.lower() in content.lower():
                            found_features.append(feature)
                            
                    if found_features:
                        results["optimization_features"].extend(found_features)
                        print(f"  ✅ {opt_file.name}: {len(found_features)} optimization features")
                    else:
                        results["warnings"].append(f"{opt_file.name}: No optimization features found")
                        
                except Exception as e:
                    results["errors"].append(f"Error reading {opt_file.name}: {str(e)}")
                    results["status"] = "FAIL"
            else:
                results["warnings"].append(f"Optimization file not found: {opt_file.name}")
                
        return results
        
    def validate_gpu_configurations(self) -> Dict[str, Any]:
        """Validate GPU optimization configurations"""
        print("🖥️ Validating GPU configurations...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "gpu_features": []
        }
        
        # Check CUDA kernel files
        cuda_files = [
            self.src_dir / "cuda" / "ckks_kernels.cu",
            self.src_dir / "cuda" / "ckks_kernels.h"
        ]
        
        gpu_features = [
            "CUDA",
            "GPU",
            "nvidia",
            "cuBLAS",
            "cuDNN",
            "memory_pool",
            "kernel",
            "device"
        ]
        
        all_content = ""
        found_cuda_files = 0
        
        for cuda_file in cuda_files:
            if cuda_file.exists():
                found_cuda_files += 1
                try:
                    with open(cuda_file, 'r') as f:
                        all_content += f.read().lower()
                except Exception as e:
                    results["errors"].append(f"Error reading {cuda_file.name}: {str(e)}")
                    
        if found_cuda_files > 0:
            print(f"  ✅ Found {found_cuda_files} CUDA kernel files")
            
            for feature in gpu_features:
                if feature.lower() in all_content:
                    results["gpu_features"].append(feature)
                    
            if len(results["gpu_features"]) >= 4:
                print(f"  ✅ GPU optimization features: {len(results['gpu_features'])}")
            else:
                results["warnings"].append("Limited GPU optimization features")
        else:
            results["warnings"].append("No CUDA kernel files found")
            
        # Check Dockerfile for GPU support
        dockerfile_prod = self.repo_root / "Dockerfile.prod"
        if dockerfile_prod.exists():
            with open(dockerfile_prod, 'r') as f:
                content = f.read()
                
            if "nvidia/cuda" in content:
                print("  ✅ GPU support in production Docker configuration")
            else:
                results["warnings"].append("No GPU support in production Docker configuration")
                
        return results
        
    def validate_caching_configurations(self) -> Dict[str, Any]:
        """Validate caching and memory optimization"""
        print("💾 Validating caching configurations...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "caching_features": []
        }
        
        # Check caching implementation
        caching_files = [
            self.src_dir / "utils" / "caching.py",
            self.repo_root / "docker-compose.prod.yml",
            self.repo_root / "DEPLOYMENT.md"
        ]
        
        caching_features = [
            "redis",
            "cache",
            "memory",
            "lru",
            "ttl",
            "eviction",
            "multi-tier",
            "l1_cache",
            "l2_cache"
        ]
        
        all_content = ""
        for cache_file in caching_files:
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        all_content += f.read().lower()
                except Exception as e:
                    results["errors"].append(f"Error reading {cache_file.name}: {str(e)}")
                    
        for feature in caching_features:
            if feature in all_content:
                results["caching_features"].append(feature)
                
        if len(results["caching_features"]) >= 4:
            print(f"  ✅ Caching features found: {len(results['caching_features'])}")
        else:
            results["warnings"].append("Limited caching configuration")
            
        return results
        
    def validate_scaling_configurations(self) -> Dict[str, Any]:
        """Validate auto-scaling configurations"""
        print("📈 Validating scaling configurations...")
        
        results = {
            "status": "PASS", 
            "errors": [],
            "warnings": [],
            "scaling_features": []
        }
        
        # Check auto-scaling configurations
        scaling_files = [
            self.src_dir / "utils" / "auto_scaling.py",
            self.repo_root / "deployment" / "terraform" / "main.tf",
            self.repo_root / "deployment" / "terraform" / "kubernetes.tf"
        ]
        
        scaling_features = [
            "auto_scaling",
            "horizontal",
            "vertical", 
            "min_replicas",
            "max_replicas",
            "scaling_config",
            "target_cpu",
            "target_memory",
            "load_balancer"
        ]
        
        all_content = ""
        for scale_file in scaling_files:
            if scale_file.exists():
                try:
                    with open(scale_file, 'r') as f:
                        all_content += f.read().lower()
                except Exception as e:
                    results["errors"].append(f"Error reading {scale_file.name}: {str(e)}")
                    
        for feature in scaling_features:
            if feature in all_content:
                results["scaling_features"].append(feature)
                
        if len(results["scaling_features"]) >= 5:
            print(f"  ✅ Auto-scaling features: {len(results['scaling_features'])}")
        else:
            results["warnings"].append("Limited auto-scaling configuration")
            
        return results
        
    def validate_performance_thresholds(self) -> Dict[str, Any]:
        """Validate performance thresholds and SLAs"""
        print("🎯 Validating performance thresholds...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "thresholds_defined": []
        }
        
        # Check for performance threshold definitions
        config_files = [
            self.repo_root / "DEPLOYMENT.md",
            self.src_dir / "optimization" / "production_optimization_engine.py"
        ]
        
        performance_thresholds = [
            "latency",
            "throughput",
            "response_time",
            "sla",
            "target_",
            "threshold",
            "performance",
            "benchmark"
        ]
        
        all_content = ""
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        all_content += f.read().lower()
                except Exception as e:
                    results["errors"].append(f"Error reading {config_file.name}: {str(e)}")
                    
        for threshold in performance_thresholds:
            if threshold in all_content:
                results["thresholds_defined"].append(threshold)
                
        if len(results["thresholds_defined"]) >= 4:
            print(f"  ✅ Performance thresholds defined: {len(results['thresholds_defined'])}")
        else:
            results["warnings"].append("Limited performance threshold definitions")
            
        return results
        
    def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive performance optimization validation"""
        print("🚀 Starting Performance Optimization Validation")
        print("=" * 60)
        
        validation_suite = [
            ("benchmark_results", self.validate_benchmark_results),
            ("optimization_configurations", self.validate_optimization_configurations),
            ("gpu_configurations", self.validate_gpu_configurations),
            ("caching_configurations", self.validate_caching_configurations),
            ("scaling_configurations", self.validate_scaling_configurations),
            ("performance_thresholds", self.validate_performance_thresholds)
        ]
        
        overall_status = "PASS"
        total_errors = 0
        total_warnings = 0
        
        for test_name, test_func in validation_suite:
            try:
                result = test_func()
                self.validation_results[test_name] = result
                
                if result["status"] == "FAIL":
                    overall_status = "FAIL"
                    
                total_errors += len(result.get("errors", []))
                total_warnings += len(result.get("warnings", []))
                
                # Print test result
                status_symbol = "❌" if result["status"] == "FAIL" else "✅"
                print(f"{status_symbol} {test_name.replace('_', ' ').title()}: {result['status']}")
                
                if result.get("errors"):
                    for error in result["errors"][:2]:  # Limit errors shown
                        print(f"   ❌ {error}")
                        
                if result.get("warnings"):
                    for warning in result["warnings"][:2]:  # Limit warnings shown
                        print(f"   ⚠️  {warning}")
                        
            except Exception as e:
                print(f"❌ {test_name} validation failed: {str(e)}")
                overall_status = "FAIL"
                total_errors += 1
                
        print("\n" + "=" * 60)
        print("📋 PERFORMANCE OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        status_symbol = "✅" if overall_status == "PASS" else "❌"
        print(f"{status_symbol} Overall Status: {overall_status}")
        print(f"📊 Total Errors: {total_errors}")
        print(f"⚠️  Total Warnings: {total_warnings}")
        
        # Performance metrics summary
        benchmark_result = self.validation_results.get("benchmark_results", {})
        performance_metrics = benchmark_result.get("performance_metrics", {})
        
        if performance_metrics:
            print(f"\n📊 Performance Metrics:")
            avg_improvement = performance_metrics.get("avg_improvement", 0)
            best_method = performance_metrics.get("best_method", "N/A")
            total_benchmarks = performance_metrics.get("total_benchmarks", 0)
            
            print(f"   📈 Average Improvement: {avg_improvement:.2%}")
            print(f"   🏆 Best Method: {best_method}")
            print(f"   🔍 Total Benchmarks: {total_benchmarks}")
            
        # Feature summary
        optimization_result = self.validation_results.get("optimization_configurations", {})
        gpu_result = self.validation_results.get("gpu_configurations", {})
        caching_result = self.validation_results.get("caching_configurations", {})
        scaling_result = self.validation_results.get("scaling_configurations", {})
        
        print(f"\n🔧 Optimization Features:")
        if optimization_result.get("optimization_features"):
            print(f"   ⚙️ Optimization: {len(set(optimization_result['optimization_features']))} features")
        if gpu_result.get("gpu_features"):
            print(f"   🖥️ GPU: {len(set(gpu_result['gpu_features']))} features")
        if caching_result.get("caching_features"):
            print(f"   💾 Caching: {len(set(caching_result['caching_features']))} features")
        if scaling_result.get("scaling_features"):
            print(f"   📈 Scaling: {len(set(scaling_result['scaling_features']))} features")
            
        print(f"\n💡 Recommendations:")
        if total_warnings > 0:
            print(f"   • Review {total_warnings} warnings for optimal performance")
        if total_errors == 0:
            print(f"   • Performance optimization appears ready for production")
            print(f"   • Consider monitoring performance metrics in production")
        else:
            print(f"   • Address {total_errors} errors before production deployment")
            
        return {
            "overall_status": overall_status,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "performance_metrics": performance_metrics,
            "detailed_results": self.validation_results
        }


def main():
    """Main validation entry point"""
    validator = PerformanceValidator()
    results = validator.run_validation()
    
    # Save results to file
    results_file = Path("performance_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()