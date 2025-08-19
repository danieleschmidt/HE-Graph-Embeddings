#!/usr/bin/env python3
"""
Multi-region deployment validation script for HE-Graph-Embeddings

Validates the global deployment infrastructure configuration for production readiness.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any


class MultiRegionValidator:
    """Validates multi-region deployment configuration and infrastructure"""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.deployment_dir = self.repo_root / "deployment"
        self.terraform_dir = self.deployment_dir / "terraform"
        self.validation_results = {}
        
    def validate_terraform_configs(self) -> Dict[str, Any]:
        """Validate Terraform configuration files"""
        print("🔍 Validating Terraform configurations...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "validated_files": []
        }
        
        terraform_files = [
            self.terraform_dir / "main.tf",
            self.terraform_dir / "kubernetes.tf",
            self.deployment_dir / "quantum-planner" / "terraform" / "global-deployment.tf"
        ]
        
        for tf_file in terraform_files:
            if tf_file.exists():
                try:
                    # Check basic syntax and structure
                    with open(tf_file, 'r') as f:
                        content = f.read()
                        
                    # Validate key components
                    required_components = [
                        "terraform {",
                        "provider \"aws\"",
                        "resource \"aws_",
                        "variable ",
                        "output "
                    ]
                    
                    missing_components = []
                    for component in required_components:
                        if component not in content:
                            missing_components.append(component)
                    
                    if missing_components:
                        results["errors"].append(f"{tf_file.name}: Missing components: {missing_components}")
                        results["status"] = "FAIL"
                    else:
                        results["validated_files"].append(str(tf_file))
                        
                except Exception as e:
                    results["errors"].append(f"Error reading {tf_file.name}: {str(e)}")
                    results["status"] = "FAIL"
            else:
                results["errors"].append(f"Missing Terraform file: {tf_file}")
                results["status"] = "FAIL"
        
        return results
        
    def validate_region_configurations(self) -> Dict[str, Any]:
        """Validate region-specific configurations"""
        print("🌍 Validating region configurations...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "regions_validated": []
        }
        
        # Expected regions for production deployment
        expected_regions = [
            "us-east-1",
            "eu-west-1", 
            "ap-northeast-1",
            "us-west-2"
        ]
        
        # Read main terraform config to check regions
        main_tf = self.terraform_dir / "main.tf"
        if main_tf.exists():
            with open(main_tf, 'r') as f:
                content = f.read()
                
            # Check if all expected regions are configured
            for region in expected_regions:
                if region in content:
                    results["regions_validated"].append(region)
                else:
                    results["warnings"].append(f"Region {region} not found in configuration")
                    
            # Check for multi-region provider configurations
            if "provider \"aws\"" in content and "alias" in content:
                print("  ✅ Multi-region provider configuration found")
            else:
                results["errors"].append("Missing multi-region provider configuration")
                results["status"] = "FAIL"
                
        return results
        
    def validate_compliance_frameworks(self) -> Dict[str, Any]:
        """Validate compliance framework configurations"""
        print("🔒 Validating compliance frameworks...")
        
        results = {
            "status": "PASS", 
            "errors": [],
            "warnings": [],
            "frameworks_validated": []
        }
        
        # Expected compliance frameworks
        expected_frameworks = ["GDPR", "CCPA", "HIPAA", "SOC2"]
        
        # Check main terraform config
        main_tf = self.terraform_dir / "main.tf"
        if main_tf.exists():
            with open(main_tf, 'r') as f:
                content = f.read()
                
            for framework in expected_frameworks:
                if framework in content:
                    results["frameworks_validated"].append(framework)
                else:
                    results["warnings"].append(f"Compliance framework {framework} not explicitly configured")
                    
            # Check for compliance-related configurations
            compliance_indicators = [
                "compliance_frameworks",
                "data_retention",
                "audit_logging",
                "encryption_required"
            ]
            
            missing_indicators = []
            for indicator in compliance_indicators:
                if indicator not in content:
                    missing_indicators.append(indicator)
                    
            if missing_indicators:
                results["warnings"].append(f"Missing compliance configurations: {missing_indicators}")
                
        return results
        
    def validate_kubernetes_configs(self) -> Dict[str, Any]:
        """Validate Kubernetes deployment configurations"""
        print("⚙️ Validating Kubernetes configurations...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "validated_components": []
        }
        
        k8s_tf = self.terraform_dir / "kubernetes.tf" 
        if k8s_tf.exists():
            with open(k8s_tf, 'r') as f:
                content = f.read()
                
            # Check for essential Kubernetes components
            required_components = [
                "aws_eks_cluster",
                "aws_eks_node_group", 
                "aws_iam_role",
                "aws_security_group",
                "encryption_config"
            ]
            
            for component in required_components:
                if component in content:
                    results["validated_components"].append(component)
                else:
                    results["errors"].append(f"Missing Kubernetes component: {component}")
                    results["status"] = "FAIL"
                    
            # Check for GPU support configuration
            if "enable_gpu_instances" in content and "gpu" in content.lower():
                print("  ✅ GPU support configuration found")
            else:
                results["warnings"].append("GPU support configuration not found")
                
        else:
            results["errors"].append("kubernetes.tf file not found")
            results["status"] = "FAIL"
            
        return results
        
    def validate_security_configurations(self) -> Dict[str, Any]:
        """Validate security configurations"""
        print("🛡️ Validating security configurations...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "security_features": []
        }
        
        # Check Terraform files for security configurations
        terraform_files = [
            self.terraform_dir / "main.tf",
            self.terraform_dir / "kubernetes.tf"
        ]
        
        security_features = [
            "kms",
            "encryption",
            "security_group",
            "ssl_certificate",
            "vpc",
            "private_subnet"
        ]
        
        all_content = ""
        for tf_file in terraform_files:
            if tf_file.exists():
                with open(tf_file, 'r') as f:
                    all_content += f.read()
                    
        for feature in security_features:
            if feature in all_content.lower():
                results["security_features"].append(feature)
            else:
                results["warnings"].append(f"Security feature not found: {feature}")
                
        # Check for specific security requirements
        if "enable_deletion_protection" in all_content:
            print("  ✅ Deletion protection enabled")
        else:
            results["warnings"].append("Deletion protection not configured")
            
        return results
        
    def validate_monitoring_setup(self) -> Dict[str, Any]:
        """Validate monitoring and observability setup"""
        print("📊 Validating monitoring setup...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "monitoring_components": []
        }
        
        # Check for monitoring configurations in deployment files
        monitoring_files = [
            self.repo_root / "docker-compose.prod.yml",
            self.repo_root / "DEPLOYMENT.md"
        ]
        
        monitoring_components = [
            "prometheus",
            "grafana", 
            "cloudwatch",
            "health_check",
            "metrics",
            "logging"
        ]
        
        all_content = ""
        for mon_file in monitoring_files:
            if mon_file.exists():
                with open(mon_file, 'r') as f:
                    all_content += f.read().lower()
                    
        for component in monitoring_components:
            if component in all_content:
                results["monitoring_components"].append(component)
            else:
                results["warnings"].append(f"Monitoring component not found: {component}")
                
        return results
        
    def validate_docker_production_config(self) -> Dict[str, Any]:
        """Validate Docker production configuration"""
        print("🐳 Validating Docker production configuration...")
        
        results = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "validated_features": []
        }
        
        dockerfile_prod = self.repo_root / "Dockerfile.prod"
        docker_compose_prod = self.repo_root / "docker-compose.prod.yml"
        
        # Validate Dockerfile.prod
        if dockerfile_prod.exists():
            with open(dockerfile_prod, 'r') as f:
                content = f.read()
                
            production_features = [
                "multi-stage",
                "non-root user",
                "HEALTHCHECK",
                "CUDA",
                "security hardening"
            ]
            
            feature_checks = {
                "multi-stage": "FROM" in content and "as" in content,
                "non-root user": "USER" in content and "hegraph" in content,
                "HEALTHCHECK": "HEALTHCHECK" in content,
                "CUDA": "nvidia/cuda" in content,
                "security hardening": "chmod" in content or "chown" in content
            }
            
            for feature, check in feature_checks.items():
                if check:
                    results["validated_features"].append(feature)
                else:
                    results["warnings"].append(f"Production feature not found: {feature}")
                    
        else:
            results["errors"].append("Dockerfile.prod not found")
            results["status"] = "FAIL"
            
        # Validate docker-compose.prod.yml
        if docker_compose_prod.exists():
            print("  ✅ Production Docker Compose configuration found")
        else:
            results["warnings"].append("docker-compose.prod.yml not found")
            
        return results
        
    def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive multi-region deployment validation"""
        print("🚀 Starting Multi-Region Deployment Validation")
        print("=" * 60)
        
        validation_suite = [
            ("terraform_configs", self.validate_terraform_configs),
            ("region_configurations", self.validate_region_configurations),
            ("compliance_frameworks", self.validate_compliance_frameworks),
            ("kubernetes_configs", self.validate_kubernetes_configs),
            ("security_configurations", self.validate_security_configurations),
            ("monitoring_setup", self.validate_monitoring_setup),
            ("docker_production", self.validate_docker_production_config)
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
                    for error in result["errors"]:
                        print(f"   ❌ {error}")
                        
                if result.get("warnings"):
                    for warning in result["warnings"][:3]:  # Limit warnings shown
                        print(f"   ⚠️  {warning}")
                        
            except Exception as e:
                print(f"❌ {test_name} validation failed: {str(e)}")
                overall_status = "FAIL"
                total_errors += 1
                
        print("\n" + "=" * 60)
        print("📋 MULTI-REGION VALIDATION SUMMARY")
        print("=" * 60)
        
        status_symbol = "✅" if overall_status == "PASS" else "❌"
        print(f"{status_symbol} Overall Status: {overall_status}")
        print(f"📊 Total Errors: {total_errors}")
        print(f"⚠️  Total Warnings: {total_warnings}")
        
        # Summary of key findings
        print(f"\n🔍 Key Validation Results:")
        
        # Terraform validation
        tf_result = self.validation_results.get("terraform_configs", {})
        if tf_result.get("validated_files"):
            print(f"   ✅ {len(tf_result['validated_files'])} Terraform files validated")
            
        # Region validation  
        region_result = self.validation_results.get("region_configurations", {})
        if region_result.get("regions_validated"):
            print(f"   🌍 {len(region_result['regions_validated'])} regions configured")
            
        # Compliance validation
        compliance_result = self.validation_results.get("compliance_frameworks", {})
        if compliance_result.get("frameworks_validated"):
            print(f"   🔒 {len(compliance_result['frameworks_validated'])} compliance frameworks configured")
            
        # Security validation
        security_result = self.validation_results.get("security_configurations", {})
        if security_result.get("security_features"):
            print(f"   🛡️ {len(security_result['security_features'])} security features enabled")
            
        print(f"\n💡 Recommendations:")
        if total_warnings > 0:
            print(f"   • Review {total_warnings} warnings for optimal configuration")
        if total_errors == 0:
            print(f"   • Multi-region deployment appears ready for production")
            print(f"   • Consider running 'terraform validate' for additional checks")
        else:
            print(f"   • Address {total_errors} errors before production deployment")
            
        return {
            "overall_status": overall_status,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "detailed_results": self.validation_results
        }


def main():
    """Main validation entry point"""
    validator = MultiRegionValidator()
    results = validator.run_validation()
    
    # Save results to file
    results_file = Path("multiregion_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()