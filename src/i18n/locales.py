"""
Internationalization (i18n) support for HE-Graph-Embeddings
Multi-language support with compliance-aware messaging
"""


import json
import os
from typing import Dict, Optional, Any
from enum import Enum
from datetime import datetime

class SupportedLocale(Enum):
    """Supported locales with region codes"""
    EN_US = "en-US"      # English (United States)
    EN_GB = "en-GB"      # English (United Kingdom)
    EN_CA = "en-CA"      # English (Canada)
    DE_DE = "de-DE"      # German (Germany)
    FR_FR = "fr-FR"      # French (France)
    ES_ES = "es-ES"      # Spanish (Spain)
    IT_IT = "it-IT"      # Italian (Italy)
    JA_JP = "ja-JP"      # Japanese (Japan)
    KO_KR = "ko-KR"      # Korean (South Korea)
    ZH_CN = "zh-CN"      # Chinese (Simplified, China)
    ZH_TW = "zh-TW"      # Chinese (Traditional, Taiwan)
    PT_BR = "pt-BR"      # Portuguese (Brazil)
    RU_RU = "ru-RU"      # Russian (Russia)
    AR_SA = "ar-SA"      # Arabic (Saudi Arabia)

class ComplianceRegion(Enum):
    """Compliance regions with specific requirements"""
    GDPR = "gdpr"        # EU GDPR
    CCPA = "ccpa"        # California CCPA
    PIPEDA = "pipeda"    # Canadian PIPEDA
    LGPD = "lgpd"        # Brazilian LGPD
    PDPA_SG = "pdpa-sg"  # Singapore PDPA
    PDPA_TH = "pdpa-th"  # Thailand PDPA
    PIPL = "pipl"        # Chinese PIPL
    UK_DPA = "uk-dpa"    # UK Data Protection Act
    HIPAA = "hipaa"      # US Healthcare
    SOX = "sox"          # US Sarbanes-Oxley

# Locale to compliance region mapping
LOCALE_COMPLIANCE_MAP = {
    SupportedLocale.EN_US: [ComplianceRegion.CCPA, ComplianceRegion.HIPAA, ComplianceRegion.SOX],
    SupportedLocale.EN_GB: [ComplianceRegion.GDPR, ComplianceRegion.UK_DPA],
    SupportedLocale.EN_CA: [ComplianceRegion.PIPEDA],
    SupportedLocale.DE_DE: [ComplianceRegion.GDPR],
    SupportedLocale.FR_FR: [ComplianceRegion.GDPR],
    SupportedLocale.ES_ES: [ComplianceRegion.GDPR],
    SupportedLocale.IT_IT: [ComplianceRegion.GDPR],
    SupportedLocale.JA_JP: [],
    SupportedLocale.KO_KR: [],
    SupportedLocale.ZH_CN: [ComplianceRegion.PIPL],
    SupportedLocale.ZH_TW: [],
    SupportedLocale.PT_BR: [ComplianceRegion.LGPD],
    SupportedLocale.RU_RU: [],
    SupportedLocale.AR_SA: []
}

# Base translations for common messages
TRANSLATIONS = {
    SupportedLocale.EN_US: {
        # API Messages
        "auth_required": "Authentication required to access this endpoint",
        "invalid_credentials": "Invalid authentication credentials provided",
        "rate_limit_exceeded": "Rate limit exceeded. Please try again later",
        "request_too_large": "Request payload exceeds maximum allowed size",
        "invalid_content_type": "Invalid content type. Expected application/json",
        "internal_error": "An internal server error occurred",
        "insufficient_memory": "Insufficient memory to process request",

        # Encryption Messages
        "encryption_failed": "Failed to encrypt data",
        "decryption_failed": "Failed to decrypt data",
        "key_generation_failed": "Failed to generate encryption keys",
        "invalid_key_format": "Invalid encryption key format",
        "noise_budget_exhausted": "Encryption noise budget exhausted",
        "security_level_insufficient": "Security level insufficient for operation",

        # Graph Processing
        "graph_too_large": "Graph exceeds maximum processing size",
        "invalid_graph_format": "Invalid graph data format",
        "model_training_failed": "Graph model training failed",
        "inference_failed": "Graph inference failed",

        # Compliance & Privacy
        "data_processing_consent": "Your data will be processed according to our privacy policy",
        "data_retention_notice": "Data will be retained according to local regulations",
        "right_to_deletion": "You have the right to request deletion of your data",
        "data_portability": "You have the right to export your data",
        "contact_dpo": "Contact our Data Protection Officer for privacy concerns",

        # Security
        "security_scan_complete": "Security scan completed successfully",
        "vulnerabilities_found": "Security vulnerabilities detected",
        "policy_violation": "Security policy violation detected",
        "access_denied": "Access denied for security reasons",

        # Performance
        "operation_timeout": "Operation timed out",
        "resource_exhausted": "System resources exhausted",
        "performance_degraded": "System performance degraded",

        # General
        "operation_successful": "Operation completed successfully",
        "operation_failed": "Operation failed",
        "validation_error": "Input validation failed",
        "configuration_error": "Configuration error detected"
    },

    SupportedLocale.DE_DE: {
        # API Messages
        "auth_required": "Authentifizierung erforderlich für Zugriff auf diesen Endpunkt",
        "invalid_credentials": "Ungültige Authentifizierungsdaten bereitgestellt",
        "rate_limit_exceeded": "Ratenlimit überschritten. Bitte versuchen Sie es später erneut",
        "request_too_large": "Anfrage-Payload überschreitet die maximal zulässige Größe",
        "invalid_content_type": "Ungültiger Inhaltstyp. Erwartet application/json",
        "internal_error": "Ein interner Serverfehler ist aufgetreten",
        "insufficient_memory": "Nicht genügend Arbeitsspeicher zur Verarbeitung der Anfrage",

        # Compliance & Privacy (GDPR-specific)
        "data_processing_consent": "Ihre Daten werden gemäß unserer Datenschutzrichtlinie verarbeitet",
        "data_retention_notice": "Daten werden gemäß den örtlichen Vorschriften aufbewahrt",
        "right_to_deletion": "Sie haben das Recht, die Löschung Ihrer Daten zu beantragen",
        "data_portability": "Sie haben das Recht, Ihre Daten zu exportieren",
        "contact_dpo": "Kontaktieren Sie unseren Datenschutzbeauftragten bei Datenschutzbedenken",

        # General
        "operation_successful": "Operation erfolgreich abgeschlossen",
        "operation_failed": "Operation fehlgeschlagen",
        "validation_error": "Eingabevalidierung fehlgeschlagen",
        "configuration_error": "Konfigurationsfehler erkannt"
    },

    SupportedLocale.FR_FR: {
        # API Messages
        "auth_required": "Authentification requise pour accéder à ce point de terminaison",
        "invalid_credentials": "Identifiants d'authentification non valides fournis",
        "rate_limit_exceeded": "Limite de débit dépassée. Veuillez réessayer plus tard",
        "internal_error": "Une erreur de serveur interne s'est produite",

        # Compliance & Privacy (GDPR-specific)
        "data_processing_consent": "Vos données seront traitées selon notre politique de confidentialité",
        "data_retention_notice": "Les données seront conservées selon les réglementations locales",
        "right_to_deletion": "Vous avez le droit de demander la suppression de vos données",
        "data_portability": "Vous avez le droit d'exporter vos données",
        "contact_dpo": "Contactez notre délégué à la protection des données pour les préoccupations de confidentialité",

        # General
        "operation_successful": "Opération terminée avec succès",
        "operation_failed": "Opération échouée"
    },

    SupportedLocale.ZH_CN: {
        # API Messages (Simplified Chinese)
        "auth_required": "访问此端点需要身份验证",
        "invalid_credentials": "提供的身份验证凭据无效",
        "rate_limit_exceeded": "超出速率限制，请稍后再试",
        "internal_error": "发生内部服务器错误",

        # Compliance & Privacy (PIPL-specific)
        "data_processing_consent": "您的数据将根据我们的隐私政策进行处理",
        "data_retention_notice": "数据将根据当地法规进行保留",
        "right_to_deletion": "您有权要求删除您的数据",
        "data_portability": "您有权导出您的数据",

        # General
        "operation_successful": "操作成功完成",
        "operation_failed": "操作失败"
    },

    SupportedLocale.JA_JP: {
        # API Messages (Japanese)
        "auth_required": "このエンドポイントへのアクセスには認証が必要です",
        "invalid_credentials": "提供された認証資格情報が無効です",
        "rate_limit_exceeded": "レート制限を超過しました。後でもう一度お試しください",
        "internal_error": "内部サーバーエラーが発生しました",

        # General
        "operation_successful": "操作が正常に完了しました",
        "operation_failed": "操作が失敗しました"
    }
}

class LocalizationManager:
    """Manage localization and compliance-aware messaging"""

    def __init__(self, default_locale: SupportedLocale = SupportedLocale.EN_US):
        """  Init  ."""
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations = TRANSLATIONS

    def set_locale(self, locale: SupportedLocale) -> None:):
        """Set current locale"""
        if locale in SupportedLocale:
            self.current_locale = locale
        else:
            self.current_locale = self.default_locale

    def get_message(self, key: str, locale: Optional[SupportedLocale] = None, **kwargs) -> str:
        """Get localized message"""
        target_locale = locale or self.current_locale

        # Try target locale first
        if target_locale in self.translations:
            if key in self.translations[target_locale]:
                message = self.translations[target_locale][key]
                return message.format(**kwargs) if kwargs else message

        # Fall back to default locale
        if self.default_locale in self.translations:
            if key in self.translations[self.default_locale]:
                message = self.translations[self.default_locale][key]
                return message.format(**kwargs) if kwargs else message

        # Final fallback
        return f"[{key}]"

    def get_compliance_regions(self, locale: Optional[SupportedLocale] = None) -> list[ComplianceRegion]:
        """Get compliance regions for locale"""
        target_locale = locale or self.current_locale
        return LOCALE_COMPLIANCE_MAP.get(target_locale, [])

    def is_gdpr_region(self, locale: Optional[SupportedLocale] = None) -> bool:
        """Check if locale requires GDPR compliance"""
        return ComplianceRegion.GDPR in self.get_compliance_regions(locale)

    def get_privacy_notice(self, locale: Optional[SupportedLocale] = None) -> Dict[str, str]:
        """Get privacy notice for locale"""
        target_locale = locale or self.current_locale
        compliance_regions = self.get_compliance_regions(target_locale)

        notice = {
            "processing_consent": self.get_message("data_processing_consent", target_locale),
            "retention_notice": self.get_message("data_retention_notice", target_locale)
        }

        # Add region-specific rights
        if ComplianceRegion.GDPR in compliance_regions:
            notice.update({
                "right_to_deletion": self.get_message("right_to_deletion", target_locale),
                "data_portability": self.get_message("data_portability", target_locale),
                "contact_dpo": self.get_message("contact_dpo", target_locale)
            })

        if ComplianceRegion.CCPA in compliance_regions:
            notice["right_to_know"] = "You have the right to know what personal information is collected"
            notice["right_to_opt_out"] = "You have the right to opt-out of the sale of personal information"

        return notice

    def format_date(self, dt: datetime, locale: Optional[SupportedLocale] = None) -> str:
        """Format date according to locale"""
        target_locale = locale or self.current_locale

        # Locale-specific date formats
        formats = {
            SupportedLocale.EN_US: "%m/%d/%Y %I:%M %p",
            SupportedLocale.EN_GB: "%d/%m/%Y %H:%M",
            SupportedLocale.DE_DE: "%d.%m.%Y %H:%M",
            SupportedLocale.FR_FR: "%d/%m/%Y %H:%M",
            SupportedLocale.ZH_CN: "%Y年%m月%d日 %H:%M",
            SupportedLocale.JA_JP: "%Y年%m月%d日 %H:%M"
        }

        format_str = formats.get(target_locale, "%Y-%m-%d %H:%M UTC")
        return dt.strftime(format_str)

    def get_currency_format(self, amount: float, locale: Optional[SupportedLocale] = None) -> str:
        """Format currency according to locale"""
        target_locale = locale or self.current_locale

        # Simplified currency formatting
        currency_formats = {
            SupportedLocale.EN_US: "${:.2f}",
            SupportedLocale.EN_GB: "£{:.2f}",
            SupportedLocale.DE_DE: "{:.2f} €",
            SupportedLocale.FR_FR: "{:.2f} €",
            SupportedLocale.ZH_CN: "¥{:.2f}",
            SupportedLocale.JA_JP: "¥{:.0f}"
        }

        format_str = currency_formats.get(target_locale, "${:.2f}")
        return format_str.format(amount)

    def get_supported_locales(self) -> list[SupportedLocale]:
        """Get list of supported locales"""
        return list(SupportedLocale)

    def detect_locale_from_request(self, accept_language: str) -> SupportedLocale:
        """Detect locale from HTTP Accept-Language header"""
        if not accept_language:
            return self.default_locale

        # Parse Accept-Language header (simplified)
        languages = []
        for lang_range in accept_language.split(','):
            parts = lang_range.strip().split(';')
            lang = parts[0].strip()
            # Handle quality factor if present
            quality = 1.0
            if len(parts) > 1:
                q_part = parts[1].strip()
                if q_part.startswith('q='):
                    try:
                        quality = float(q_part[2:])
                    except ValueError:
                        logger.error(f"Error in operation: {e}")
                        quality = 1.0
            languages.append((lang, quality))

        # Sort by quality factor
        languages.sort(key=lambda x: x[1], reverse=True)

        # Match with supported locales
        for lang, _ in languages:
            # Try exact match first
            for locale in SupportedLocale:
                if locale.value.lower() == lang.lower():
                    return locale

            # Try partial match (language only)
            lang_code = lang.split('-')[0].lower()
            for locale in SupportedLocale:
                if locale.value.split('-')[0].lower() == lang_code:
                    return locale

        return self.default_locale

# Global localization manager instance
_localization_manager = LocalizationManager()

def get_localization_manager() -> LocalizationManager:
    """Get global localization manager instance"""
    return _localization_manager

def set_global_locale(locale: SupportedLocale):
    """Set global locale"""
    _localization_manager.set_locale(locale)

def translate(key: str, locale: Optional[SupportedLocale] = None, **kwargs) -> str:
    """Convenience function for translation"""
    return _localization_manager.get_message(key, locale, **kwargs)

def get_privacy_notice(locale: Optional[SupportedLocale] = None) -> Dict[str, str]:
    """Convenience function for privacy notice"""
    return _localization_manager.get_privacy_notice(locale)