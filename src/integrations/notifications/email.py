"""
Email notification service for HE-Graph-Embeddings
"""


import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
import aiosmtplib
from jinja2 import Environment, FileSystemLoader, Template
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    """Email service configuration"""
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30

    @classmethod
    def from_env(cls) -> 'EmailConfig':
        """Create config from environment variables"""
        return cls(
            smtp_host=os.getenv('SMTP_HOST', 'localhost'),
            smtp_port=int(os.getenv('SMTP_PORT', '587')),
            username=os.getenv('SMTP_USERNAME', ''),
            password=os.getenv('SMTP_PASSWORD', ''),
            use_tls=os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
            use_ssl=os.getenv('SMTP_USE_SSL', 'false').lower() == 'true',
            timeout=int(os.getenv('SMTP_TIMEOUT', '30'))
        )

class EmailTemplate:
    """Email template management"""

    def __init__(self, template_dir: str = "src/integrations/notifications/templates"):
        """  Init  ."""
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True
        )

    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render email template with context"""
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {e}")
            raise EmailTemplateError(f"Template rendering failed: {e}")

    def get_default_context(self) -> Dict[str, Any]:
        """Get default template context"""
        return {
            "app_name": "HE-Graph-Embeddings",
            "app_url": os.getenv('APP_URL', 'http://localhost:8000'),
            "timestamp": datetime.now().isoformat(),
            "support_email": os.getenv('SUPPORT_EMAIL', 'support@hegraph.com')
        }

class EmailService:
    """Asynchronous email service"""

    def __init__(self, config: EmailConfig, template_manager: Optional[EmailTemplate] = None):
        """  Init  ."""
        self.config = config
        self.template_manager = template_manager or EmailTemplate()

    async def send_email(self,
                        to_emails: Union[str, List[str]],
                        subject: str,
                        body: str,
                        from_email: Optional[str] = None,
                        html_body: Optional[str] = None,
                        attachments: Optional[List[Dict[str, Any]]] = None,
                        cc_emails: Optional[List[str]] = None,
                        bcc_emails: Optional[List[str]] = None) -> bool:
        """Send email using async SMTP"""
        try:
            # Normalize email lists
            if isinstance(to_emails, str):
                to_emails = [to_emails]

            from_email = from_email or self.config.username

            # Create message
            message = MIMEMultipart('alternative')
            message['From'] = from_email
            message['To'] = ', '.join(to_emails)
            message['Subject'] = subject

            if cc_emails:
                message['Cc'] = ', '.join(cc_emails)

            # Add text part
            text_part = MIMEText(body, 'plain', 'utf-8')
            message.attach(text_part)

            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, 'html', 'utf-8')
                message.attach(html_part)

            # Add attachments
            if attachments:
                for attachment in attachments:
                    await self._add_attachment(message, attachment)

            # Send email
            all_recipients = to_emails + (cc_emails or []) + (bcc_emails or [])

            await aiosmtplib.send(
                message,
                hostname=self.config.smtp_host,
                port=self.config.smtp_port,
                username=self.config.username,
                password=self.config.password,
                use_tls=self.config.use_tls,
                timeout=self.config.timeout,
                recipients=all_recipients
            )

            logger.info(f"Email sent successfully to {len(all_recipients)} recipients")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    async def _add_attachment(self, message: MIMEMultipart, attachment: Dict[str, Any]):
        """Add attachment to email message"""
        try:
            if 'file_path' in attachment:
                # File attachment
                file_path = Path(attachment['file_path'])
                if not file_path.exists():
                    logger.warning(f"Attachment file not found: {file_path}")
                    return

                with open(file_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())

                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment.get("filename", file_path.name)}'
                )

            elif 'content' in attachment:
                # Content attachment
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment['content'])
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment.get("filename", "attachment.txt")}'
                )

            message.attach(part)

        except Exception as e:
            logger.error(f"Failed to add attachment: {e}")

    async def send_template_email(self,
                                to_emails: Union[str, List[str]],
                                template_name: str,
                                context: Dict[str, Any],
                                subject: Optional[str] = None,
                                from_email: Optional[str] = None) -> bool:
        """Send email using template"""
        try:
            # Merge with default context
            full_context = self.template_manager.get_default_context()
            full_context.update(context)

            # Render HTML body
            html_body = self.template_manager.render_template(template_name, full_context)

            # Extract subject from template if not provided
            if not subject and 'subject' in full_context:
                subject = full_context['subject']
            elif not subject:
                subject = f"Notification from {full_context['app_name']}"

            # Create plain text version (simplified)

            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = True
            text_body = h.handle(html_body)

            return await self.send_email(
                to_emails=to_emails,
                subject=subject,
                body=text_body,
                html_body=html_body,
                from_email=from_email
            )

        except Exception as e:
            logger.error(f"Failed to send template email: {e}")
            return False

class HEGraphEmailNotifications:
    """HE-Graph-Embeddings specific email notifications"""

    def __init__(self, email_service: EmailService):
        """  Init  ."""
        self.email_service = email_service

    async def send_experiment_completed(self,
                                        to_emails: Union[str, List[str]],
                                        experiment_results: Dict[str, Any]) -> bool:
        """Send experiment completion notification"""
        context = {
            'subject': f"Experiment Completed: {experiment_results.get('model_type', 'Unknown')}",
            'experiment_name': experiment_results.get('experiment_name', 'Unnamed Experiment'),
            'model_type': experiment_results.get('model_type', 'Unknown'),
            'accuracy': experiment_results.get('accuracy', 0),
            'encryption_time': experiment_results.get('encryption_time_ms', 0),
            'inference_time': experiment_results.get('inference_time_ms', 0),
            'noise_budget': experiment_results.get('final_noise_budget', 0),
            'security_level': experiment_results.get('security_level', 128),
            'status': 'success' if experiment_results.get('success', False) else 'failed',
            'error_message': experiment_results.get('error_message', ''),
            'duration': experiment_results.get('total_duration_minutes', 0),
            'dataset_size': experiment_results.get('dataset_size', 0)
        }

        return await self.email_service.send_template_email(
            to_emails, 'experiment_completed.html', context
        )

    async def send_training_status(self,
                                    to_emails: Union[str, List[str]],
                                    model_name: str,
                                    epoch: int,
                                    total_epochs: int,
                                    loss: float,
                                    noise_budget: float) -> bool:
        """Send training progress notification"""
        context = {
            'subject': f"Training Update: {model_name}",
            'model_name': model_name,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'progress_percent': (epoch / total_epochs) * 100,
            'loss': loss,
            'noise_budget': noise_budget,
            'eta_minutes': ((total_epochs - epoch) * 2) if epoch > 0 else 0  # Rough estimate
        }

        return await self.email_service.send_template_email(
            to_emails, 'training_status.html', context
        )

    async def send_security_alert(self,
                                to_emails: Union[str, List[str]],
                                alert_type: str,
                                details: Dict[str, Any]) -> bool:
        """Send security alert notification"""
        context = {
            'subject': f"Security Alert: {alert_type}",
            'alert_type': alert_type,
            'severity': details.get('severity', 'medium'),
            'description': details.get('description', ''),
            'affected_system': details.get('system', 'Unknown'),
            'remediation': details.get('remediation', 'Please investigate immediately'),
            'timestamp': details.get('timestamp', datetime.now().isoformat()),
            'source_ip': details.get('source_ip', 'Unknown'),
            'user_agent': details.get('user_agent', 'Unknown')
        }

        return await self.email_service.send_template_email(
            to_emails, 'security_alert.html', context
        )

    async def send_performance_report(self,
                                    to_emails: Union[str, List[str]],
                                    report_data: Dict[str, Any]) -> bool:
        """Send performance report"""
        context = {
            'subject': f"Performance Report - {report_data.get('period', 'Daily')}",
            'period': report_data.get('period', 'Daily'),
            'total_requests': report_data.get('total_requests', 0),
            'avg_response_time': report_data.get('avg_response_time_ms', 0),
            'success_rate': report_data.get('success_rate_percent', 0),
            'error_count': report_data.get('error_count', 0),
            'gpu_utilization': report_data.get('gpu_utilization_percent', 0),
            'memory_usage': report_data.get('memory_usage_mb', 0),
            'top_endpoints': report_data.get('top_endpoints', []),
            'slowest_operations': report_data.get('slowest_operations', [])
        }

        return await self.email_service.send_template_email(
            to_emails, 'performance_report.html', context
        )

    async def send_system_health_alert(self,
                                        to_emails: Union[str, List[str]],
                                        health_status: Dict[str, Any]) -> bool:
        """Send system health alert"""
        context = {
            'subject': f"System Health Alert: {health_status.get('status', 'Unknown')}",
            'overall_status': health_status.get('status', 'unknown'),
            'cpu_usage': health_status.get('cpu_usage_percent', 0),
            'memory_usage': health_status.get('memory_usage_percent', 0),
            'disk_usage': health_status.get('disk_usage_percent', 0),
            'gpu_status': health_status.get('gpu_status', 'unknown'),
            'active_contexts': health_status.get('active_contexts', 0),
            'active_models': health_status.get('active_models', 0),
            'uptime_hours': health_status.get('uptime_hours', 0),
            'issues': health_status.get('issues', [])
        }

        return await self.email_service.send_template_email(
            to_emails, 'system_health_alert.html', context
        )

    async def send_backup_notification(self,
                                        to_emails: Union[str, List[str]],
                                        backup_info: Dict[str, Any]) -> bool:
        """Send backup completion notification"""
        context = {
            'subject': f"Backup {'Completed' if backup_info.get('success') else 'Failed'}",
            'success': backup_info.get('success', False),
            'backup_type': backup_info.get('type', 'full'),
            'backup_size_mb': backup_info.get('size_mb', 0),
            'duration_minutes': backup_info.get('duration_minutes', 0),
            'backup_location': backup_info.get('location', 'Unknown'),
            'files_backed_up': backup_info.get('files_count', 0),
            'error_message': backup_info.get('error_message', ''),
            'next_backup': backup_info.get('next_backup_time', '')
        }

        return await self.email_service.send_template_email(
            to_emails, 'backup_notification.html', context
        )

# Email notification scheduler
class EmailNotificationScheduler:
    """Schedule and manage email notifications"""

    def __init__(self, email_service: EmailService):
        """  Init  ."""
        self.email_service = email_service
        self.he_notifications = HEGraphEmailNotifications(email_service)
        self.scheduled_tasks = []

    async def schedule_daily_report(self, to_emails: List[str], hour: int = 9):
        """Schedule daily performance report"""
        # This would integrate with a task scheduler like Celery
        # For now, it's a placeholder for the interface
        logger.info(f"Scheduled daily reports for {len(to_emails)} recipients at {hour}:00")

    async def schedule_weekly_summary(self, to_emails: List[str], day_of_week: int = 0):
        """Schedule weekly summary report"""
        logger.info(f"Scheduled weekly summaries for {len(to_emails)} recipients on day {day_of_week}")

    def add_alert_subscription(self, email -> None: str, alert_types: List[str]):
        """Add email to alert subscriptions"""
        # This would integrate with a database to store subscriptions
        logger.info(f"Added {email} to alerts: {alert_types}")

    def remove_alert_subscription(self, email -> None: str, alert_types: List[str]):
        """Remove email from alert subscriptions"""
        logger.info(f"Removed {email} from alerts: {alert_types}")

# Custom exceptions
class EmailServiceError(Exception):
    """Base email service error"""
    pass

class EmailTemplateError(EmailServiceError):
    """Email template error"""
    pass

class EmailConfigError(EmailServiceError):
    """Email configuration error"""
    pass

# Utility functions
def validate_email(email: str) -> bool:
    """Validate email address format"""

    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def sanitize_email_content(content: str) -> str:
    """Sanitize email content to prevent injection"""

    import html
    return html.escape(content)

# Testing utilities
class MockEmailService(EmailService):
    """Mock email service for testing"""

    def __init__(self):
        """  Init  ."""
        self.sent_emails = []

    async def send_email(self, *args, **kwargs) -> bool:
        """Mock send email - store in memory"""
        self.sent_emails.append({
            'args': args,
            'kwargs': kwargs,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def get_sent_emails(self) -> List[Dict[str, Any]]:
        """Get list of sent emails"""
        return self.sent_emails

    def clear_sent_emails(self) -> None:
        """Clear sent emails list"""
        self.sent_emails.clear()