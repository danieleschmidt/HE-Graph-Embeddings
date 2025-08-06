# Security Documentation

## Security Model Overview

HE-Graph-Embeddings implements a comprehensive security model based on homomorphic encryption, ensuring that sensitive graph data remains encrypted throughout the entire computation pipeline. This document outlines the security architecture, threat model, and compliance frameworks.

## Threat Model

### Assets Protected

1. **Graph Data**: Node features, edge relationships, and graph structure
2. **Model Parameters**: Neural network weights and biases  
3. **Computation Results**: Encrypted embeddings and predictions
4. **System Infrastructure**: Encryption keys, authentication tokens, and system configurations

### Threat Actors

- **External Attackers**: Malicious actors attempting to access encrypted data
- **Insider Threats**: Privileged users with legitimate access attempting unauthorized operations
- **Cloud Service Providers**: Infrastructure providers with potential access to compute resources
- **Malicious Researchers**: Adversaries attempting model extraction or inference attacks

### Attack Vectors

- **Data Interception**: Network-based attacks on data in transit
- **Memory Exploitation**: Attacks targeting data in memory during computation
- **Side-channel Attacks**: Timing, power, or electromagnetic analysis
- **Model Extraction**: Attempts to reverse-engineer model parameters
- **Inference Attacks**: Extracting information about training data

## Cryptographic Foundation

### CKKS Homomorphic Encryption

**Mathematical Foundation**:
- **Ring**: R = Z[X]/(X^N + 1) where N is a power of 2
- **Ciphertext Space**: R_q^2 for polynomial ring modulus q
- **Security**: Based on Ring Learning With Errors (RLWE) problem
- **Parameters**: Configured for 128-bit security level

**Implementation Details**:
```python
# Security parameters
POLY_MODULUS_DEGREE = 16384  # N = 2^14
COEFF_MODULUS_BITS = [60, 40, 40, 40, 40, 40, 60]  # q ≈ 2^340
SCALE = 2^40  # Precision scaling factor
NOISE_BUDGET = 120  # Initial noise budget in bits
```

**Security Properties**:
- **IND-CPA Security**: Indistinguishable under chosen-plaintext attacks
- **Semantic Security**: Ciphertexts reveal no information about plaintexts
- **Circuit Privacy**: Homomorphic operations don't leak intermediate values

### Key Management

**Key Hierarchy**:
```
Master Key (KMS)
├── Context Encryption Key
│   ├── Public Key (encryption)
│   ├── Secret Key (decryption) 
│   └── Relinearization Keys (multiplication)
└── System Keys
    ├── Authentication Keys
    ├── API Keys
    └── TLS Certificates
```

**Key Lifecycle**:
1. **Generation**: Secure random generation using hardware entropy
2. **Distribution**: Encrypted transmission using TLS 1.3
3. **Storage**: AWS KMS with hardware security modules (HSMs)
4. **Rotation**: Automatic rotation every 90 days
5. **Destruction**: Secure deletion with cryptographic erasure

### Noise Management

**Noise Budget Tracking**:
```python
class NoiseTracker:
    def __init__(self, initial_budget=120):
        self.budget = initial_budget
        self.operations = []
    
    def consume_noise(self, operation, cost):
        if self.budget - cost < MIN_NOISE_THRESHOLD:
            raise NoiseExhaustionError()
        self.budget -= cost
        self.operations.append((operation, cost))
```

**Noise Budget Allocation**:
- **Addition Operations**: ~0.1 bits per operation
- **Multiplication Operations**: ~20-40 bits per operation  
- **Activation Functions**: ~15-25 bits per approximation
- **Reserve Buffer**: 10 bits minimum for decryption

## Authentication and Authorization

### API Authentication

**JWT Token Structure**:
```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user_12345",
    "iss": "he-graph.terragon.ai", 
    "aud": "api.he-graph.terragon.ai",
    "exp": 1642276800,
    "iat": 1642190400,
    "scope": ["graph:read", "graph:process", "context:create"],
    "role": "researcher",
    "org": "university_abc"
  }
}
```

**API Key Management**:
- **Generation**: Cryptographically secure random generation (256-bit)
- **Hashing**: PBKDF2 with SHA-256, 100,000 iterations
- **Storage**: Encrypted at rest with separate encryption keys
- **Expiration**: Configurable expiration (default: 1 year)

### Role-Based Access Control (RBAC)

**Role Definitions**:

| Role | Permissions | Description |
|------|-------------|-------------|
| `viewer` | `graph:read`, `context:view` | Read-only access to graphs and contexts |
| `researcher` | `graph:*`, `context:*` | Full graph processing capabilities |
| `admin` | `system:*`, `user:*` | System administration and user management |
| `auditor` | `logs:read`, `metrics:read` | Monitoring and compliance access |

**Permission Granularity**:
- **Resource-level**: Control access to specific graphs or contexts
- **Operation-level**: Restrict specific operations (encrypt, decrypt, process)
- **Data-level**: Limit access based on data sensitivity classification

### Multi-Factor Authentication

**Supported Factors**:
1. **Password**: Minimum 12 characters, complexity requirements
2. **TOTP**: Time-based one-time passwords (RFC 6238)
3. **Hardware Keys**: FIDO2/WebAuthn security keys
4. **Biometric**: Touch ID, Face ID (client-side verification)

## Network Security

### TLS Configuration

**TLS 1.3 Settings**:
```nginx
ssl_protocols TLSv1.3;
ssl_ciphers TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256;
ssl_prefer_server_ciphers off;
ssl_session_timeout 1d;
ssl_session_cache shared:MozTLS:10m;
ssl_stapling on;
ssl_stapling_verify on;
```

**Certificate Management**:
- **Issuer**: AWS Certificate Manager with automatic renewal
- **Validation**: DNS validation for domain ownership
- **Algorithm**: RSA-2048 or ECDSA P-256
- **Chain**: Full certificate chain validation

### Network Segmentation

**VPC Architecture**:
```
Internet Gateway
    ├── Public Subnets (Load Balancers)
    │   └── Application Load Balancer
    ├── Private Subnets (Application Tier)
    │   ├── EKS Worker Nodes
    │   └── API Gateway
    └── Isolated Subnets (Data Tier)
        ├── Database Instances
        └── Encryption Key Storage
```

**Security Groups**:
- **Web Tier**: Ports 80, 443 from internet
- **Application Tier**: Port 8000 from web tier only
- **Database Tier**: Port 5432 from application tier only
- **Management**: SSH (port 22) from bastion hosts only

### DDoS Protection

**AWS Shield Standard**:
- **SYN/UDP Floods**: Automatic detection and mitigation
- **Reflection Attacks**: DNS, NTP, SSDP reflection protection
- **Layer 3/4 Attacks**: Network and transport layer protection

**AWS Shield Advanced** (Enterprise):
- **Application Layer**: HTTP/HTTPS flood protection
- **Real-time Metrics**: Attack visibility and reporting
- **24/7 Support**: DDoS Response Team access
- **Cost Protection**: Credit for scaling costs during attacks

## Data Protection

### Encryption at Rest

**Database Encryption**:
- **Algorithm**: AES-256-GCM with AWS KMS keys
- **Key Rotation**: Automatic rotation every 90 days
- **Backup Encryption**: All backups encrypted with separate keys
- **Performance**: Transparent encryption with minimal overhead

**File System Encryption**:
- **EBS Volumes**: AES-256 encryption for all persistent storage
- **Ephemeral Storage**: Instance store encryption where supported
- **Container Images**: Encrypted container registry (ECR)

### Encryption in Transit

**API Communications**:
- **Client ↔ Load Balancer**: TLS 1.3 with perfect forward secrecy
- **Load Balancer ↔ Application**: TLS 1.2+ with client certificates
- **Service ↔ Database**: TLS 1.2+ with certificate validation
- **Inter-service**: Istio service mesh with mTLS

### Data Classification

**Sensitivity Levels**:

| Level | Description | Encryption | Access Control |
|-------|-------------|------------|----------------|
| **Public** | Non-sensitive operational data | Standard TLS | Basic authentication |
| **Internal** | Business operational data | TLS + AES-256 | Role-based access |
| **Confidential** | Sensitive research data | HE + AES-256 | Multi-factor auth |
| **Restricted** | Regulated/compliance data | HE + AES-256 + HSM | Strict RBAC + audit |

## Compliance Framework

### GDPR Compliance

**Data Protection Principles**:

1. **Lawfulness**: Consent-based processing with clear purpose
2. **Data Minimization**: Only necessary features processed
3. **Purpose Limitation**: Encryption contexts scoped to specific tasks
4. **Accuracy**: Data validation and integrity checks
5. **Storage Limitation**: Automatic deletion of expired contexts
6. **Security**: Encryption by design and by default
7. **Accountability**: Comprehensive audit trails

**Individual Rights Implementation**:

| Right | Implementation | Technical Mechanism |
|-------|----------------|-------------------|
| Access | API endpoints for data retrieval | Encrypted data export |
| Rectification | Data update mechanisms | Context refresh procedures |
| Erasure | Cryptographic deletion | Key destruction |
| Portability | Standardized export formats | JSON/CSV with metadata |
| Object | Processing restriction flags | Context suspension |
| Automated Decision Making | Human oversight requirements | Manual review workflows |

### HIPAA Compliance

**Administrative Safeguards**:
- **Security Officer**: Designated HIPAA Security Officer
- **Workforce Training**: Annual security awareness training
- **Access Management**: Minimum necessary access principle
- **Incident Response**: Documented breach response procedures

**Physical Safeguards**:
- **Data Centers**: SOC 2 Type II certified facilities
- **Workstation Security**: Encrypted endpoints with device management
- **Media Controls**: Secure disposal and sanitization procedures

**Technical Safeguards**:
- **Access Control**: Unique user identification and authentication
- **Audit Controls**: Comprehensive logging and monitoring
- **Integrity Controls**: Cryptographic integrity verification
- **Transmission Security**: End-to-end encryption for all communications

### SOC 2 Compliance

**Trust Service Criteria**:

1. **Security**: Infrastructure protection and access controls
2. **Availability**: System uptime and disaster recovery
3. **Processing Integrity**: Accurate and complete processing
4. **Confidentiality**: Information protection and privacy
5. **Privacy**: Personal information collection and use

## Security Monitoring

### Security Information and Event Management (SIEM)

**Log Sources**:
- **Application Logs**: Authentication, authorization, API access
- **System Logs**: OS events, network connections, process execution
- **Infrastructure Logs**: Load balancer, database, encryption service logs
- **Security Logs**: WAF events, intrusion detection, vulnerability scans

**Detection Rules**:
```yaml
# Example: Suspicious authentication pattern
rule_id: "AUTH_001"
name: "Multiple Failed Logins"
description: "Detect brute force authentication attempts"
condition: |
  count(auth_failed) > 5 
  within time_window(5m) 
  group_by(source_ip)
severity: "MEDIUM"
response: "temporary_ip_block"
```

### Intrusion Detection

**Network-based Detection**:
- **Suricata IDS**: Deep packet inspection and protocol analysis
- **AWS GuardDuty**: Machine learning-based threat detection
- **VPC Flow Logs**: Network traffic analysis and anomaly detection

**Host-based Detection**:
- **OSSEC**: File integrity monitoring and log analysis  
- **AWS Systems Manager**: Compliance monitoring and patch management
- **Container Security**: Runtime security monitoring with Falco

### Vulnerability Management

**Scanning Schedule**:
- **Infrastructure**: Weekly vulnerability scans with Nessus
- **Applications**: Daily SAST/DAST scans in CI/CD pipeline
- **Dependencies**: Continuous monitoring with Snyk/Dependabot
- **Container Images**: Build-time and runtime vulnerability scanning

**Patch Management**:
- **Critical Vulnerabilities**: Emergency patching within 24 hours
- **High Vulnerabilities**: Patching within 7 days  
- **Medium/Low Vulnerabilities**: Monthly patch cycles
- **Zero-day Vulnerabilities**: Coordinated disclosure and emergency response

## Incident Response

### Security Incident Classification

| Severity | Criteria | Response Time | Escalation |
|----------|----------|---------------|------------|
| **P1 - Critical** | Active breach, data exposure | 15 minutes | CTO, Legal, PR |
| **P2 - High** | Potential breach, system compromise | 1 hour | Security team, Management |
| **P3 - Medium** | Security control failure, anomalous activity | 4 hours | Security team |
| **P4 - Low** | Policy violation, suspicious activity | 24 hours | Security analyst |

### Response Procedures

**Incident Response Team**:
- **Incident Commander**: Overall response coordination
- **Technical Lead**: Technical investigation and remediation
- **Communications Lead**: Internal and external communications
- **Legal Counsel**: Regulatory and legal considerations

**Response Phases**:
1. **Detection**: Automated alerts and manual reporting
2. **Analysis**: Threat assessment and impact evaluation
3. **Containment**: Isolate affected systems and prevent spread
4. **Eradication**: Remove threat and address vulnerabilities  
5. **Recovery**: Restore normal operations with enhanced monitoring
6. **Lessons Learned**: Post-incident review and process improvement

### Forensic Capabilities

**Evidence Collection**:
- **Memory Dumps**: Volatile memory capture for analysis
- **Disk Images**: Bit-for-bit copies of storage devices
- **Network Captures**: Packet-level network traffic analysis
- **Log Preservation**: Immutable log storage with chain of custody

**Analysis Tools**:
- **Volatility**: Memory analysis framework
- **Autopsy**: Digital forensics platform
- **Wireshark**: Network protocol analyzer
- **YARA**: Malware identification and classification

## Security Testing

### Penetration Testing

**Testing Schedule**:
- **External Testing**: Quarterly assessments by third-party vendors
- **Internal Testing**: Monthly assessments by security team
- **Application Testing**: Continuous testing in CI/CD pipeline
- **Social Engineering**: Annual phishing and social engineering tests

**Testing Methodology**:
- **OWASP Testing Guide**: Web application security testing
- **PTES**: Penetration Testing Execution Standard
- **NIST SP 800-115**: Technical Guide to Information Security Testing

### Security Code Review

**Static Analysis**:
- **SonarQube**: Code quality and security vulnerability detection
- **Bandit**: Python security linter for common vulnerabilities
- **Semgrep**: Pattern-based static analysis for custom rules
- **CodeQL**: Semantic code analysis for complex vulnerabilities

**Manual Review Process**:
- **Threat Modeling**: Architecture-level security analysis
- **Code Review**: Line-by-line security review for critical components
- **Cryptographic Review**: Expert review of encryption implementations
- **Configuration Review**: Security configuration validation

## Business Continuity

### Backup and Recovery

**Backup Strategy**:
- **Database**: Point-in-time recovery with 7-day retention
- **Application Data**: Daily encrypted backups to cross-region storage
- **Configuration**: Infrastructure as code with version control
- **Encryption Keys**: Hardware security module backup and recovery

**Recovery Objectives**:
- **RTO (Recovery Time)**: 4 hours for critical systems
- **RPO (Recovery Point)**: 1 hour maximum data loss
- **Cross-region Failover**: Automated failover with DNS routing
- **Data Integrity**: Cryptographic verification of restored data

### Disaster Recovery Testing

**Testing Schedule**:
- **Quarterly**: Full disaster recovery testing
- **Monthly**: Backup restoration testing  
- **Weekly**: Failover procedure validation
- **Daily**: Backup integrity verification

## Security Contacts

### Reporting Security Issues

**Security Email**: security@terragon.ai  
**PGP Key**: Available at https://terragon.ai/.well-known/pgp-key.asc  
**Bug Bounty**: https://terragon.ai/security/bug-bounty

### 24/7 Security Operations Center

**Phone**: +1-555-SECURITY (1-555-732-8748)  
**Email**: soc@terragon.ai  
**Slack**: #security-incidents (internal)

This security documentation provides a comprehensive overview of the HE-Graph-Embeddings security architecture and operational procedures. Regular reviews and updates ensure continued effectiveness against evolving threats.