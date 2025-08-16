"""
Security Policy Enforcer for HE-Graph-Embeddings
Production-grade security policy validation and enforcement
"""


import yaml
import logging
import os
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import requests

from .security_scanner import SecurityReport, SecurityLevel, run_security_scan

logger = logging.getLogger(__name__)

@dataclass
class PolicyViolation:
    """Security policy violation"""
    policy_name: str
    violation_type: str
    description: str
    severity: str
    remediation: str
    finding_ids: List[str]

@dataclass
class ComplianceStatus:
    """Compliance validation status"""
    compliant: bool
    violations: List[PolicyViolation]
    score: float  # 0-100 compliance score
    report: Dict[str, Any]

class SecurityPolicyEnforcer:
    """Enforce security policies and compliance requirements"""

    def __init__(self, config_path: str = None):
        """  Init  ."""
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "security_config.yaml"
        )
        self.config = self._load_config()
        self.violations = []

    def _load_config(self) -> Dict[str, Any]:
        """Load security policy configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Apply environment-specific overrides
            environment = os.getenv('ENVIRONMENT', 'development')
            if environment in config.get('environment_overrides', {}):
                overrides = config['environment_overrides'][environment]
                config = self._merge_config(config, overrides)

            return config

        except FileNotFoundError:
            logger.error(f"Security config file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing security config: {e}")
            return self._get_default_config()

    def _merge_config(self, base: Dict, overrides: Dict) -> Dict:
        """Merge configuration with environment overrides"""
        result = base.copy()
        for key, value in overrides.items():
            if isinstance(value, dict) and key in result:
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default security configuration"""
        return {
            'max_findings': {
                'critical': 0,
                'high': 2,
                'medium': 5,
                'low': 10,
                'info': 50
            },
            'blocked_vulnerabilities': [
                'CWE-89', 'CWE-78', 'CWE-22', 'CWE-798', 'CWE-502'
            ],
            'encryption': {
                'min_security_bits': 128,
                'min_poly_degree': 8192
            },
            'api': {
                'require_authentication': True,
                'require_https': True
            }
        }

    def validate_security_report(self, report: SecurityReport) -> ComplianceStatus:
        """Validate security scan report against policies"""
        violations = []

        # Check finding count limits
        max_findings = self.config.get('max_findings', {})
        for severity, max_count in max_findings.items():
            actual_count = report.summary.get(severity, 0)
            if actual_count > max_count:
                violations.append(PolicyViolation(
                    policy_name="max_findings",
                    violation_type="finding_count_exceeded",
                    description=f"Too many {severity} findings: {actual_count} > {max_count}",
                    severity="high" if severity in ["critical", "high"] else "medium",
                    remediation=f"Reduce {severity} findings to {max_count} or below",
                    finding_ids=[f.id for f in report.findings if f.severity.value == severity]
                ))

        # Check for blocked vulnerability types
        blocked_vulns = self.config.get('blocked_vulnerabilities', [])
        for finding in report.findings:
            if finding.cwe_id in blocked_vulns:
                violations.append(PolicyViolation(
                    policy_name="blocked_vulnerabilities",
                    violation_type="blocked_vulnerability_found",
                    description=f"Blocked vulnerability type {finding.cwe_id}: {finding.title}",
                    severity="critical",
                    remediation=f"Fix or mitigate {finding.cwe_id} vulnerability in {finding.file_path}",
                    finding_ids=[finding.id]
                ))

        # Check critical findings policy
        critical_findings = [f for f in report.findings if f.severity == SecurityLevel.CRITICAL]
        if critical_findings and not self.config.get('allow_critical', False):
            violations.append(PolicyViolation(
                policy_name="critical_findings",
                violation_type="critical_findings_not_allowed",
                description=f"{len(critical_findings)} critical findings found",
                severity="critical",
                remediation="Fix all critical security findings",
                finding_ids=[f.id for f in critical_findings]
            ))

        # Calculate compliance score
        score = self._calculate_compliance_score(report, violations)

        return ComplianceStatus(
            compliant=len(violations) == 0,
            violations=violations,
            score=score,
            report={
                'total_violations': len(violations),
                'critical_violations': len([v for v in violations if v.severity == 'critical']),
                'scan_timestamp': report.timestamp.isoformat(),
                'scan_id': report.scan_id
            }
        )

    def _calculate_compliance_score(self, report: SecurityReport, violations: List[PolicyViolation]) -> float:
        """Calculate compliance score (0-100)"""
        base_score = 100.0

        # Deduct points for violations
        for violation in violations:
            if violation.severity == 'critical':
                base_score -= 25
            elif violation.severity == 'high':
                base_score -= 10
            elif violation.severity == 'medium':
                base_score -= 5
            else:
                base_score -= 1

        # Deduct points for findings
        for severity, count in report.summary.items():
            if severity == 'critical':
                base_score -= count * 5
            elif severity == 'high':
                base_score -= count * 2
            elif severity == 'medium':
                base_score -= count * 1
            elif severity == 'low':
                base_score -= count * 0.5

        return max(0.0, min(100.0, base_score))

    def validate_encryption_config(self, config: Dict[str, Any]) -> List[PolicyViolation]:
        """Validate encryption configuration against policy"""
        violations = []
        encryption_policy = self.config.get('encryption', {})

        # Check minimum security bits
        min_security_bits = encryption_policy.get('min_security_bits', 128)
        if config.get('security_level', 0) < min_security_bits:
            violations.append(PolicyViolation(
                policy_name="encryption_security",
                violation_type="insufficient_security_bits",
                description=f"Security level {config.get('security_level')} < {min_security_bits} bits",
                severity="high",
                remediation=f"Use at least {min_security_bits}-bit security level",
                finding_ids=[]
            ))

        # Check polynomial degree
        min_poly_degree = encryption_policy.get('min_poly_degree', 8192)
        if config.get('poly_modulus_degree', 0) < min_poly_degree:
            violations.append(PolicyViolation(
                policy_name="encryption_parameters",
                violation_type="insufficient_poly_degree",
                description=f"Polynomial degree {config.get('poly_modulus_degree')} < {min_poly_degree}",
                severity="medium",
                remediation=f"Use polynomial degree of at least {min_poly_degree}",
                finding_ids=[]
            ))

        # Check coefficient modulus size
        max_coeff_bits = encryption_policy.get('max_coeff_modulus_bits', 438)
        coeff_bits = sum(config.get('coeff_modulus_bits', []))
        if coeff_bits > max_coeff_bits:
            violations.append(PolicyViolation(
                policy_name="encryption_parameters",
                violation_type="excessive_coeff_modulus",
                description=f"Coefficient modulus {coeff_bits} bits > {max_coeff_bits} bits",
                severity="medium",
                remediation=f"Reduce coefficient modulus to {max_coeff_bits} bits or less",
                finding_ids=[]
            ))

        return violations

    def validate_api_security(self, app_config: Dict[str, Any]) -> List[PolicyViolation]:
        """Validate API security configuration"""
        violations = []
        api_policy = self.config.get('api', {})

        # Check HTTPS requirement
        if api_policy.get('require_https', True) and not app_config.get('use_https', False):
            violations.append(PolicyViolation(
                policy_name="api_security",
                violation_type="https_required",
                description="HTTPS is required but not enabled",
                severity="high",
                remediation="Enable HTTPS/TLS for all API endpoints",
                finding_ids=[]
            ))

        # Check authentication requirement
        if api_policy.get('require_authentication', True) and not app_config.get('auth_enabled', False):
            violations.append(PolicyViolation(
                policy_name="api_security",
                violation_type="authentication_required",
                description="Authentication is required but not enabled",
                severity="critical",
                remediation="Enable authentication middleware",
                finding_ids=[]
            ))

        # Check rate limiting
        max_rpm = api_policy.get('max_requests_per_minute', 60)
        if app_config.get('rate_limit', float('inf')) > max_rpm:
            violations.append(PolicyViolation(
                policy_name="api_security",
                violation_type="rate_limit_exceeded",
                description=f"Rate limit {app_config.get('rate_limit')} > {max_rpm} RPM",
                severity="medium",
                remediation=f"Set rate limit to {max_rpm} requests per minute or less",
                finding_ids=[]
            ))

        return violations

    def validate_compliance(self, project_root: str) -> ComplianceStatus:
        """Run comprehensive compliance validation"""
        violations = []

        # Run security scan
        logger.info("Running security scan for compliance validation")
        report = run_security_scan(project_root)

        # Validate scan results
        scan_compliance = self.validate_security_report(report)
        violations.extend(scan_compliance.violations)

        # Check for required security files
        required_files = [
            'security/security_config.yaml',
            'security/security_scanner.py',
            'requirements.txt'
        ]

        for file_path in required_files:
            full_path = os.path.join(project_root, file_path)
            if not os.path.exists(full_path):
                violations.append(PolicyViolation(
                    policy_name="required_files",
                    violation_type="missing_security_file",
                    description=f"Required security file missing: {file_path}",
                    severity="medium",
                    remediation=f"Create required file: {file_path}",
                    finding_ids=[]
                ))

        # Validate Git security
        git_violations = self._validate_git_security(project_root)
        violations.extend(git_violations)

        # Check dependency vulnerabilities
        dep_violations = self._validate_dependencies(project_root)
        violations.extend(dep_violations)

        # Calculate overall compliance score
        total_findings = report.summary.get('total', 0)
        compliance_score = self._calculate_overall_compliance_score(violations, total_findings)

        return ComplianceStatus(
            compliant=len([v for v in violations if v.severity in ['critical', 'high']]) == 0,
            violations=violations,
            score=compliance_score,
            report={
                'total_violations': len(violations),
                'security_findings': total_findings,
                'scan_report': report.to_dict(),
                'validation_timestamp': datetime.utcnow().isoformat()
            }
        )

    def _validate_git_security(self, project_root: str) -> List[PolicyViolation]:
        """Validate Git repository security"""
        violations = []

        try:
            # Check for .gitignore
            gitignore_path = os.path.join(project_root, '.gitignore')
            if not os.path.exists(gitignore_path):
                violations.append(PolicyViolation(
                    policy_name="git_security",
                    violation_type="missing_gitignore",
                    description="No .gitignore file found",
                    severity="medium",
                    remediation="Create .gitignore file to exclude sensitive files",
                    finding_ids=[]
                ))
            else:
                # Check .gitignore contents
                with open(gitignore_path, 'r') as f:
                    gitignore_content = f.read()

                required_patterns = ['.env', '*.key', '*.pem', '__pycache__']
                missing_patterns = [p for p in required_patterns if p not in gitignore_content]

                if missing_patterns:
                    violations.append(PolicyViolation(
                        policy_name="git_security",
                        violation_type="incomplete_gitignore",
                        description=f"Missing patterns in .gitignore: {missing_patterns}",
                        severity="low",
                        remediation=f"Add missing patterns to .gitignore: {missing_patterns}",
                        finding_ids=[]
                    ))

            # Check for secrets in recent commits
            result = subprocess.run(
                ['git', 'log', '--oneline', '-10', '--grep=password|secret|key'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                violations.append(PolicyViolation(
                    policy_name="git_security",
                    violation_type="potential_secrets_in_commits",
                    description="Found potential secrets in recent commit messages",
                    severity="medium",
                    remediation="Review and clean commit history of sensitive information",
                    finding_ids=[]
                ))

        except subprocess.TimeoutExpired:
            logger.warning("Git security validation timed out")
        except Exception as e:
            logger.warning(f"Git security validation failed: {e}")

        return violations

    def _validate_dependencies(self, project_root: str) -> List[PolicyViolation]:
        """Validate dependency security"""
        violations = []

        # Check requirements.txt for known vulnerable packages
        requirements_path = os.path.join(project_root, 'requirements.txt')
        if os.path.exists(requirements_path):
            try:
                with open(requirements_path, 'r') as f:
                    requirements = f.read()

                # Known vulnerable package patterns
                vulnerable_patterns = {
                    'pillow': '< 8.2.0',
                    'requests': '< 2.20.0',
                    'jinja2': '< 2.11.3',
                    'pyyaml': '< 5.4.0',
                    'urllib3': '< 1.26.5'
                }

                for package, version_constraint in vulnerable_patterns.items():
                    if package in requirements.lower():
                        violations.append(PolicyViolation(
                            policy_name="dependency_security",
                            violation_type="potentially_vulnerable_dependency",
                            description=f"Package {package} may be vulnerable (check version {version_constraint})",
                            severity="medium",
                            remediation=f"Update {package} to latest secure version",
                            finding_ids=[]
                        ))

            except Exception as e:
                logger.warning(f"Could not validate requirements.txt: {e}")

        return violations

    def _calculate_overall_compliance_score(self, violations: List[PolicyViolation], 
                                            total_findings: int) -> float:
        """ Calculate Overall Compliance Score."""
        """Calculate overall compliance score"""
        base_score = 100.0

        # Deduct for violations
        for violation in violations:
            if violation.severity == 'critical':
                base_score -= 20
            elif violation.severity == 'high':
                base_score -= 10
            elif violation.severity == 'medium':
                base_score -= 5
            else:
                base_score -= 2

        # Deduct for total findings
        base_score -= min(total_findings * 0.5, 20)

        return max(0.0, base_score)

    def send_alert(self, compliance_status: ComplianceStatus) -> bool:
        """Send security alerts based on compliance status"""
        if compliance_status.compliant:
            return True

        alert_config = self.config.get('alerts', {})

        # Check if we should send immediate alerts
        critical_violations = [v for v in compliance_status.violations if v.severity == 'critical']
        immediate_severities = alert_config.get('immediate_alert_severity', ['critical'])

        should_alert = any(v.severity in immediate_severities for v in compliance_status.violations)

        if not should_alert:
            return True

        # Send webhook alert
        webhook_url = alert_config.get('webhook_url')
        if webhook_url:
            try:
                payload = {
                    'compliance_status': compliance_status.compliant,
                    'score': compliance_status.score,
                    'violations': len(compliance_status.violations),
                    'critical_violations': len(critical_violations),
                    'timestamp': datetime.utcnow().isoformat(),
                    'details': [
                        {
                            'policy': v.policy_name,
                            'type': v.violation_type,
                            'severity': v.severity,
                            'description': v.description
                        } for v in compliance_status.violations[:10]  # Limit to first 10
                    ]
                }

                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                logger.info("Security alert sent successfully")
                return True

            except requests.RequestException as e:
                logger.error(f"Failed to send security alert: {e}")
                return False

        return True

    def generate_compliance_report(self, compliance_status: ComplianceStatus) -> None:,
        """Generate Compliance Report."""
                                output_path: str) -> bool:
        """Generate detailed compliance report"""
        try:
            report_data = {
                'compliance_summary': {
                    'compliant': compliance_status.compliant,
                    'score': compliance_status.score,
                    'total_violations': len(compliance_status.violations),
                    'critical_violations': len([v for v in compliance_status.violations if v.severity == 'critical']),
                    'timestamp': datetime.utcnow().isoformat()
                },
                'policy_configuration': self.config,
                'violations': [
                    {
                        'policy_name': v.policy_name,
                        'violation_type': v.violation_type,
                        'description': v.description,
                        'severity': v.severity,
                        'remediation': v.remediation,
                        'finding_count': len(v.finding_ids)
                    } for v in compliance_status.violations
                ],
                'recommendations': self._generate_recommendations(compliance_status)
            }

            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"Compliance report generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return False

    def _generate_recommendations(self, compliance_status: ComplianceStatus) -> List[Dict[str, str]]:
        """Generate remediation recommendations"""
        recommendations = []

        # Group violations by severity
        critical_violations = [v for v in compliance_status.violations if v.severity == 'critical']
        high_violations = [v for v in compliance_status.violations if v.severity == 'high']

        if critical_violations:
            recommendations.append({
                'priority': 'immediate',
                'action': 'Fix critical security violations',
                'description': f"Address {len(critical_violations)} critical security violations immediately"
            })

        if high_violations:
            recommendations.append({
                'priority': 'urgent',
                'action': 'Fix high-severity violations',
                'description': f"Address {len(high_violations)} high-severity violations within 24 hours"
            })

        if compliance_status.score < 80:
            recommendations.append({
                'priority': 'important',
                'action': 'Improve overall security posture',
                'description': f"Compliance score of {compliance_status.score:.1f}% needs improvement"
            })

        return recommendations

def enforce_security_policy(project_root: str, config_path: str = None) -> ComplianceStatus:
    """Main entry point for security policy enforcement"""
    enforcer = SecurityPolicyEnforcer(config_path)
    compliance_status = enforcer.validate_compliance(project_root)

    # Send alerts if needed
    enforcer.send_alert(compliance_status)

    # Generate compliance report
    report_dir = os.path.join(project_root, 'security_reports')
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, f"compliance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    enforcer.generate_compliance_report(compliance_status, report_path)

    return compliance_status

if __name__ == "__main__":

    import sys

    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    config_path = sys.argv[2] if len(sys.argv) > 2 else None

    logging.basicConfig(level=logging.INFO)

    compliance_status = enforce_security_policy(project_root, config_path)

    print(f"Compliance Status: {'COMPLIANT' if compliance_status.compliant else 'NON-COMPLIANT'}")
    print(f"Compliance Score: {compliance_status.score:.1f}%")
    print(f"Violations: {len(compliance_status.violations)}")

    if compliance_status.violations:
        print("\nTop Violations:")
        for violation in compliance_status.violations[:5]:
            print(f"  - [{violation.severity.upper()}] {violation.description}")

    # Exit with error code if not compliant
    sys.exit(0 if compliance_status.compliant else 1)