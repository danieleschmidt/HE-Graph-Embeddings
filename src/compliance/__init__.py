"""
Compliance framework package for HE-Graph-Embeddings

Provides comprehensive privacy and data protection compliance management
across multiple regulatory frameworks including GDPR, CCPA, HIPAA, and others.
"""

from .compliance_manager import (
    ComplianceFramework,
    DataCategory,
    ConsentType,
    ProcessingPurpose,
    DataSubject,
    CompliancePolicy,
    DataProcessingRecord,
    ComplianceManager,
    compliance_manager
)

__all__ = [
    'ComplianceFramework',
    'DataCategory', 
    'ConsentType',
    'ProcessingPurpose',
    'DataSubject',
    'CompliancePolicy',
    'DataProcessingRecord',
    'ComplianceManager',
    'compliance_manager'
]