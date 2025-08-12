"""
Comprehensive security scanner for HE-Graph-Embeddings
"""


import os
import re
import ast
import json
import hashlib
import subprocess
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import secrets
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security finding severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class SecurityFinding:
    """Individual security finding"""
    id: str
    title: str
    description: str
    severity: SecurityLevel
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """To Dict."""
        return {
            **asdict(self),
            'severity': self.severity.value
        }

@dataclass
class SecurityReport:
    """Complete security scan report"""
    scan_id: str
    timestamp: datetime
    findings: List[SecurityFinding]
    summary: Dict[str, int]
    scan_duration: float
    scanned_files: int

    def to_dict(self) -> Dict[str, Any]:
        """To Dict."""
        return {
            'scan_id': self.scan_id,
            'timestamp': self.timestamp.isoformat(),
            'findings': [f.to_dict() for f in self.findings],
            'summary': self.summary,
            'scan_duration': self.scan_duration,
            'scanned_files': self.scanned_files
        }

class SecurityScanner:
    """Main security scanner class"""

    def __init__(self, project_root: str):
        """  Init  ."""
        self.project_root = Path(project_root)
        self.findings: List[SecurityFinding] = []
        self.scanned_files = 0

        # Initialize security rules
        self._init_security_rules()

    def _init_security_rules(self) -> None:
        """Initialize security scanning rules"""
        self.secret_patterns = {
            'api_key': re.compile(r'(?i)api[_-]?key["\']?\s*[:=]\s*["\'][a-zA-Z0-9_-]{16,}["\']'),
            'password': re.compile(r'(?i)password["\']?\s*[:=]\s*["\'][^"\']{8,}["\']'),
            'jwt_token': re.compile(r'(?i)jwt[_-]?token["\']?\s*[:=]\s*["\'][a-zA-Z0-9_.-]{20,}["\']'),
            'private_key': re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----'),
            'aws_secret': re.compile(r'(?i)aws[_-]?secret[_-]?access[_-]?key["\']?\s*[:=]\s*["\'][a-zA-Z0-9/+=]{40}["\']'),
            'database_url': re.compile(r'(?i)(?:postgres|mysql|mongodb)://[^\\s"\']+'),
        }

        self.vulnerability_patterns = {
            'sql_injection': re.compile(r'(?i)\.execute\([^)]*%[sf]'),
            'command_injection': re.compile(r'(?i)(?:subprocess|os\.system|eval|exec)\([^)]*\+'),
            'path_traversal': re.compile(r'(?i)open\([^)]*\.\.[/\\]'),
            'weak_crypto': re.compile(r'(?i)(?:md5|sha1)\('),
            'hardcoded_secret': re.compile(r'["\'][a-zA-Z0-9]{32,}["\']'),
            'unsafe_deserialize': re.compile(r'(?i)pickle\.loads?\('),
        }

        self.insecure_functions = {
            'eval', 'exec', 'compile', 'input', '__import__',
            'open', 'file', 'execfile', 'reload'
        }

        self.dangerous_imports = {
            'pickle': 'Can lead to arbitrary code execution',
            'subprocess': 'Potential command injection if user input used',
            'os': 'OS operations can be dangerous',
            'tempfile': 'Insecure temporary file handling',
        }

    def scan(self) -> SecurityReport:
        """Run comprehensive security scan"""
        start_time = datetime.now()
        scan_id = f"scan_{int(start_time.timestamp())}"

        logger.info(f"Starting security scan {scan_id}")

        # Scan different file types
        self._scan_python_files()
        self._scan_config_files()
        self._scan_dockerfile()
        self._scan_dependencies()
        self._scan_git_secrets()

        end_time = datetime.now()
        scan_duration = (end_time - start_time).total_seconds()

        # Generate summary
        summary = self._generate_summary()

        report = SecurityReport(
            scan_id=scan_id,
            timestamp=start_time,
            findings=self.findings,
            summary=summary,
            scan_duration=scan_duration,
            scanned_files=self.scanned_files
        )

        logger.info(f"Security scan completed: {len(self.findings)} findings")
        return report

    def _scan_python_files(self) -> None:
        """Scan Python source files"""
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            self.scanned_files += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for secrets
                self._check_secrets(file_path, content)

                # Check for vulnerabilities
                self._check_vulnerabilities(file_path, content)

                # Parse AST for deeper analysis
                try:
                    tree = ast.parse(content)
                    self._analyze_ast(file_path, tree, content)
                except SyntaxError as e:
                    logger.error(f"Error in operation: {e}")
                    self._add_finding(
                        f"syntax_error_{file_path.name}",
                        "Python Syntax Error",
                        f"File contains syntax errors: {e}",
                        SecurityLevel.MEDIUM,
                        str(file_path),
                        getattr(e, 'lineno', 1),
                        "",
                        "Fix syntax errors to enable proper security analysis"
                    )

            except Exception as e:
                logger.warning(f"Error scanning {file_path}: {e}")

    def _scan_config_files(self) -> None:
        """Scan configuration files"""
        config_patterns = ["*.yml", "*.yaml", "*.json", "*.env", "*.cfg", "*.ini"]

        for pattern in config_patterns:
            for file_path in self.project_root.rglob(pattern):
                if self._should_skip_file(file_path):
                    continue

                self.scanned_files += 1

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    self._check_config_security(file_path, content)

                except Exception as e:
                    logger.warning(f"Error scanning config {file_path}: {e}")

    def _scan_dockerfile(self) -> None:
        """Scan Dockerfile for security issues"""
        dockerfile_paths = list(self.project_root.rglob("*[Dd]ockerfile*"))

        for file_path in dockerfile_paths:
            self.scanned_files += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                self._check_dockerfile_security(file_path, content)

            except Exception as e:
                logger.warning(f"Error scanning Dockerfile {file_path}: {e}")

    def _scan_dependencies(self) -> None:
        """Scan dependencies for known vulnerabilities"""
        # Check requirements.txt
        req_files = list(self.project_root.rglob("requirements*.txt"))

        for req_file in req_files:
            self.scanned_files += 1

            try:
                with open(req_file, 'r') as f:
                    content = f.read()

                self._check_dependency_security(req_file, content)

            except Exception as e:
                logger.warning(f"Error scanning requirements {req_file}: {e}")

    def _scan_git_secrets(self) -> None:
        """Scan for secrets in git history"""
        if not (self.project_root / '.git').exists():
            return

        try:
            # Check current staged/unstaged changes
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                changed_files = result.stdout.strip().split('\n')
                for file_path in changed_files:
                    if file_path and not self._should_skip_file(Path(file_path)):
                        full_path = self.project_root / file_path
                        if full_path.exists():
                            self._check_git_secrets(full_path)

        except Exception as e:
            logger.warning(f"Error scanning git secrets: {e}")

    def _check_secrets(self, file_path -> None: Path, content: str):
        """Check for hardcoded secrets"""
        lines = content.split('\n')

        for pattern_name, pattern in self.secret_patterns.items():
            for line_num, line in enumerate(lines, 1):
                if pattern.search(line):
                    self._add_finding(
                        f"secret_{pattern_name}_{file_path.name}_{line_num}",
                        f"Hardcoded {pattern_name.replace('_', ' ').title()}",
                        f"Found potential hardcoded {pattern_name} in source code",
                        SecurityLevel.CRITICAL,
                        str(file_path),
                        line_num,
                        line.strip(),
                        f"Remove hardcoded {pattern_name} and use environment variables or secure credential management",
                        "CWE-798"
                    )

    def _check_vulnerabilities(self, file_path -> None: Path, content: str):
        """Check for vulnerability patterns"""
        lines = content.split('\n')

        for vuln_name, pattern in self.vulnerability_patterns.items():
            for line_num, line in enumerate(lines, 1):
                if pattern.search(line):
                    severity = SecurityLevel.HIGH
                    if vuln_name == 'weak_crypto':
                        severity = SecurityLevel.MEDIUM
                    elif vuln_name == 'hardcoded_secret':
                        severity = SecurityLevel.CRITICAL

                    self._add_finding(
                        f"vuln_{vuln_name}_{file_path.name}_{line_num}",
                        f"Potential {vuln_name.replace('_', ' ').title()}",
                        f"Code pattern suggests potential {vuln_name} vulnerability",
                        severity,
                        str(file_path),
                        line_num,
                        line.strip(),
                        self._get_vulnerability_recommendation(vuln_name),
                        self._get_cwe_id(vuln_name)
                    )

    def _analyze_ast(self, file_path -> None: Path, tree: ast.AST, content: str):
        """Analyze Python AST for security issues"""
        lines = content.split('\n')

        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.insecure_functions:
                        line_num = getattr(node, 'lineno', 1)
                        line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                        self._add_finding(
                            f"unsafe_func_{node.func.id}_{file_path.name}_{line_num}",
                            f"Unsafe Function: {node.func.id}",
                            f"Use of potentially unsafe function '{node.func.id}'",
                            SecurityLevel.HIGH,
                            str(file_path),
                            line_num,
                            line_content.strip(),
                            f"Avoid using '{node.func.id}' or ensure input validation",
                            "CWE-94"
                        )

            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.dangerous_imports:
                        line_num = getattr(node, 'lineno', 1)
                        line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                        self._add_finding(
                            f"dangerous_import_{alias.name}_{file_path.name}_{line_num}",
                            f"Dangerous Import: {alias.name}",
                            self.dangerous_imports[alias.name],
                            SecurityLevel.MEDIUM,
                            str(file_path),
                            line_num,
                            line_content.strip(),
                            "Review usage and apply proper security measures",
                            "CWE-676"
                        )

    def _check_config_security(self, file_path -> None: Path, content: str):
        """Check configuration files for security issues"""
        lines = content.split('\n')

        # Check for secrets in config files
        for line_num, line in enumerate(lines, 1):
            for pattern_name, pattern in self.secret_patterns.items():
                if pattern.search(line):
                    self._add_finding(
                        f"config_secret_{pattern_name}_{file_path.name}_{line_num}",
                        f"Secret in Config: {pattern_name}",
                        f"Configuration file contains potential {pattern_name}",
                        SecurityLevel.CRITICAL,
                        str(file_path),
                        line_num,
                        line.strip(),
                        "Use environment variables or secure configuration management",
                        "CWE-798"
                    )

        # Check for insecure configurations
        insecure_configs = [
            (r'debug\s*[:=]\s*true', "Debug mode enabled in production"),
            (r'ssl\s*[:=]\s*false', "SSL/TLS disabled"),
            (r'verify\s*[:=]\s*false', "Certificate verification disabled"),
            (r'localhost', "Hardcoded localhost reference"),
            (r'127\.0\.0\.1', "Hardcoded loopback IP address"),
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern, description in insecure_configs:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_finding(
                        f"insecure_config_{file_path.name}_{line_num}",
                        "Insecure Configuration",
                        description,
                        SecurityLevel.MEDIUM,
                        str(file_path),
                        line_num,
                        line.strip(),
                        "Review and secure configuration settings",
                        "CWE-16"
                    )

    def _check_dockerfile_security(self, file_path -> None: Path, content: str):
        """Check Dockerfile for security issues"""
        lines = content.split('\n')

        dockerfile_issues = [
            (r'FROM\s+.*:latest', "Using 'latest' tag", SecurityLevel.MEDIUM),
            (r'USER\s+root', "Running as root user", SecurityLevel.HIGH),
            (r'ADD\s+http', "Using ADD with URL", SecurityLevel.MEDIUM),
            (r'COPY\s+.*\s+/', "Copying to root directory", SecurityLevel.MEDIUM),
            (r'--no-check-certificate', "Disabling certificate checks", SecurityLevel.HIGH),
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern, description, severity in dockerfile_issues:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_finding(
                        f"dockerfile_{file_path.name}_{line_num}",
                        f"Dockerfile Security Issue",
                        description,
                        severity,
                        str(file_path),
                        line_num,
                        line.strip(),
                        "Follow Docker security best practices",
                        "CWE-16"
                    )

    def _check_dependency_security(self, file_path -> None: Path, content: str):
        """Check dependencies for known vulnerabilities"""
        lines = content.split('\n')

        # Known vulnerable packages (simplified - in production, use vulnerability DB)
        vulnerable_packages = {
            'pillow': ('< 8.2.0', 'Multiple vulnerabilities in older versions'),
            'requests': ('< 2.20.0', 'Certificate verification bypass'),
            'jinja2': ('< 2.11.3', 'SSTI vulnerability'),
            'pyyaml': ('< 5.4.0', 'Arbitrary code execution'),
        }

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse package specification
            package_spec = line.split('==')[0].split('>=')[0].split('<=')[0].strip()

            if package_spec.lower() in vulnerable_packages:
                vuln_version, description = vulnerable_packages[package_spec.lower()]

                self._add_finding(
                    f"vulnerable_dep_{package_spec}_{line_num}",
                    f"Vulnerable Dependency: {package_spec}",
                    f"Package {package_spec} {vuln_version}: {description}",
                    SecurityLevel.HIGH,
                    str(file_path),
                    line_num,
                    line,
                    f"Update {package_spec} to the latest secure version",
                    "CWE-937"
                )

    def _check_git_secrets(self, file_path -> None: Path):
        """Check git-tracked files for secrets"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            self._check_secrets(file_path, content)

        except Exception as e:
            logger.warning(f"Error checking git secrets in {file_path}: {e}")

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during scanning"""
        skip_patterns = [
            '*/venv/*', '*/env/*', '*/.git/*', '*/node_modules/*',
            '*/__pycache__/*', '*.pyc', '*.pyo', '*.egg-info/*',
            '*/htmlcov/*', '*/.pytest_cache/*', '*/logs/*'
        ]

        file_str = str(file_path)

        for pattern in skip_patterns:
            if Path(file_str).match(pattern):
                return True

        return False

    def _add_finding(self, finding_id -> None: str, title: str, description: str,
        """ Add Finding."""
                    severity: SecurityLevel, file_path: str, line_number: int,
                    code_snippet: str, recommendation: str, cwe_id: str = None):
        """Add a security finding"""
        finding = SecurityFinding(
            id=finding_id,
            title=title,
            description=description,
            severity=severity,
            file_path=file_path,
            line_number=line_number,
            code_snippet=code_snippet,
            recommendation=recommendation,
            cwe_id=cwe_id
        )

        self.findings.append(finding)

    def _generate_summary(self) -> Dict[str, int]:
        """Generate findings summary"""
        summary = {level.value: 0 for level in SecurityLevel}

        for finding in self.findings:
            summary[finding.severity.value] += 1

        summary['total'] = len(self.findings)
        return summary

    def _get_vulnerability_recommendation(self, vuln_type: str) -> str:
        """Get specific recommendation for vulnerability type"""
        recommendations = {
            'sql_injection': 'Use parameterized queries or ORM instead of string formatting',
            'command_injection': 'Validate and sanitize all inputs before using in commands',
            'path_traversal': 'Validate file paths and use safe path operations',
            'weak_crypto': 'Use strong cryptographic algorithms (SHA-256, SHA-3, etc.)',
            'hardcoded_secret': 'Use environment variables or secure credential management',
            'unsafe_deserialize': 'Avoid pickle for untrusted data, use JSON or validate inputs',
        }

        return recommendations.get(vuln_type, 'Review code for security implications')

    def _get_cwe_id(self, vuln_type: str) -> str:
        """Get CWE ID for vulnerability type"""
        cwe_mapping = {
            'sql_injection': 'CWE-89',
            'command_injection': 'CWE-78',
            'path_traversal': 'CWE-22',
            'weak_crypto': 'CWE-327',
            'hardcoded_secret': 'CWE-798',
            'unsafe_deserialize': 'CWE-502',
        }

        return cwe_mapping.get(vuln_type, 'CWE-1000')

class SecurityPolicyEnforcer:
    """Enforce security policies"""

    def __init__(self, policy_config: Dict[str, Any]):
        """  Init  ."""
        self.config = policy_config

    def validate_report(self, report: SecurityReport) -> Tuple[bool, List[str]]:
        """Validate security report against policies"""
        violations = []

        # Check maximum allowed findings per severity
        max_allowed = self.config.get('max_findings', {})

        for severity, count in report.summary.items():
            if severity in max_allowed and count > max_allowed[severity]:
                violations.append(
                    f"Too many {severity} findings: {count} > {max_allowed[severity]}"
                )

        # Check for blocked vulnerability types
        blocked_types = self.config.get('blocked_vulnerabilities', [])

        for finding in report.findings:
            if finding.cwe_id in blocked_types:
                violations.append(
                    f"Blocked vulnerability type {finding.cwe_id} found: {finding.title}"
                )

        # Check for critical findings
        critical_findings = [f for f in report.findings if f.severity == SecurityLevel.CRITICAL]
        if critical_findings and not self.config.get('allow_critical', False):
            violations.append(f"Critical security findings not allowed: {len(critical_findings)} found")

        return len(violations) == 0, violations

def generate_security_report_html(report: SecurityReport) -> str:
    """Generate HTML security report"""
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .summary-card {{
            background: white; border: 1px solid #ddd;
            padding: 15px; border-radius: 5px; flex: 1; text-align: center;
        }}
        .critical {{ background: #ffebee; border-color: #f44336; }}
        .high {{ background: #fff3e0; border-color: #ff9800; }}
        .medium {{ background: #f3e5f5; border-color: #9c27b0; }}
        .low {{ background: #e8f5e8; border-color: #4caf50; }}
        .finding {{
            border: 1px solid #ddd; margin: 10px 0;
            padding: 15px; border-radius: 5px;
        }}
        .finding-title {{ font-weight: bold; margin-bottom: 10px; }}
        .code-snippet {{
            background: #f5f5f5; padding: 10px;
            border-radius: 3px; font-family: monospace;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Scan Report</h1>
        <p><strong>Scan ID:</strong> {{scan_id}}</p>
        <p><strong>Timestamp:</strong> {{timestamp}}</p>
        <p><strong>Duration:</strong> {{duration:.2f}} seconds</p>
        <p><strong>Files Scanned:</strong> {{scanned_files}}</p>
    </div>

    <div class="summary">
        <div class="summary-card critical">
            <h3>Critical</h3>
            <div style="font-size: 24px;">{{critical_count}}</div>
        </div>
        <div class="summary-card high">
            <h3>High</h3>
            <div style="font-size: 24px;">{{high_count}}</div>
        </div>
        <div class="summary-card medium">
            <h3>Medium</h3>
            <div style="font-size: 24px;">{{medium_count}}</div>
        </div>
        <div class="summary-card low">
            <h3>Low</h3>
            <div style="font-size: 24px;">{{low_count}}</div>
        </div>
    </div>

    <h2>Findings</h2>
    {{findings_html}}
</body>
</html>"""

    # Generate findings HTML
    findings_html = ""
    for finding in sorted(report.findings, key=lambda x: (x.severity.value, x.file_path)):
        findings_html += f"""<div class="finding {finding.severity.value}">
    <div class="finding-title">{finding.title}</div>
    <p><strong>File:</strong> {finding.file_path}:{finding.line_number}</p>
    <p><strong>Description:</strong> {finding.description}</p>
    {f'<p><strong>CWE:</strong> {finding.cwe_id}</p>' if finding.cwe_id else ''}
    {f'<div class="code-snippet">{finding.code_snippet}</div>' if finding.code_snippet else ''}
    <p><strong>Recommendation:</strong> {finding.recommendation}</p>
</div>"""

    return html_template.format(
        scan_id=report.scan_id,
        timestamp=report.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        duration=report.scan_duration,
        scanned_files=report.scanned_files,
        critical_count=report.summary.get('critical', 0),
        high_count=report.summary.get('high', 0),
        medium_count=report.summary.get('medium', 0),
        low_count=report.summary.get('low', 0),
        findings_html=findings_html
    )

def run_security_scan(project_root: str, output_dir: str = None) -> SecurityReport:
    """Run complete security scan and generate reports"""
    scanner = SecurityScanner(project_root)
    report = scanner.scan()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate JSON report
        json_path = output_path / f"security_report_{report.scan_id}.json"
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        # Generate HTML report
        html_path = output_path / f"security_report_{report.scan_id}.html"
        with open(html_path, 'w') as f:
            f.write(generate_security_report_html(report))

        logger.info(f"Security reports generated: {json_path}, {html_path}")

    return report

if __name__ == "__main__":

    import sys

    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./security_reports"

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run security scan
    report = run_security_scan(project_root, output_dir)

    print(f"Security scan completed: {report.summary['total']} findings")
    print(f"  Critical: {report.summary.get('critical', 0)}")
    print(f"  High: {report.summary.get('high', 0)}")
    print(f"  Medium: {report.summary.get('medium', 0)}")
    print(f"  Low: {report.summary.get('low', 0)}")

    # Exit with error code if critical or high findings
    if report.summary.get('critical', 0) > 0 or report.summary.get('high', 0) > 0:
        sys.exit(1)