"""
Anomaly Detection and Alert System for Container Analytics

Provides alerting system that:
- Detects unusual dwell times
- Identifies traffic congestion  
- Monitors throughput deviations
- Sends alert notifications
- Uses statistical methods for anomaly detection
"""

import logging
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import statistics

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc

from ..database.models import (
    session_scope, Image, Detection, Container, Metric
)
from .metrics import ContainerMetrics
from .aggregator import DataAggregator

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    DWELL_TIME = "dwell_time"
    THROUGHPUT = "throughput" 
    CONGESTION = "congestion"
    SYSTEM = "system"
    DATA_QUALITY = "data_quality"


@dataclass
class Alert:
    """Container for alert information."""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    camera_id: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'camera_id': self.camera_id,
            'value': self.value,
            'threshold': self.threshold,
            'metadata': self.metadata
        }


class DwellTimeAlert:
    """Dwell time anomaly detection."""
    
    def __init__(self, threshold_multiplier: float = 2.0):
        """
        Initialize dwell time alert detector.
        
        Args:
            threshold_multiplier: Number of standard deviations for anomaly detection
        """
        self.threshold_multiplier = threshold_multiplier
        self.logger = logging.getLogger(__name__)
    
    def detect_anomalies(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None
    ) -> List[Alert]:
        """
        Detect dwell time anomalies.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            camera_id: Specific camera to analyze
            
        Returns:
            List of dwell time alerts
        """
        alerts = []
        
        with session_scope() as session:
            # Get containers with dwell times
            query = session.query(Container).filter(
                Container.last_seen >= start_date,
                Container.last_seen <= end_date,
                Container.dwell_time.isnot(None)
            )
            
            if camera_id:
                query = query.filter(Container.camera_id == camera_id)
            
            containers = query.all()
            
            if len(containers) < 10:  # Need minimum data for statistical analysis
                return alerts
            
            # Calculate statistical thresholds
            dwell_times = [c.dwell_time for c in containers]
            mean_dwell = statistics.mean(dwell_times)
            std_dwell = statistics.stdev(dwell_times) if len(dwell_times) > 1 else 0
            
            # Thresholds
            upper_threshold = mean_dwell + (self.threshold_multiplier * std_dwell)
            lower_threshold = max(0, mean_dwell - (self.threshold_multiplier * std_dwell))
            
            # Critical thresholds (3 sigma)
            critical_upper = mean_dwell + (3 * std_dwell)
            critical_lower = max(0, mean_dwell - (3 * std_dwell))
            
            # Check each container
            for container in containers:
                dwell_time = container.dwell_time
                
                # Excessive dwell time alerts
                if dwell_time > critical_upper:
                    alerts.append(Alert(
                        alert_type=AlertType.DWELL_TIME,
                        severity=AlertSeverity.CRITICAL,
                        title="Critical: Excessive Container Dwell Time",
                        message=f"Container {container.container_number} has extremely high dwell time: {dwell_time:.1f} hours (threshold: {critical_upper:.1f})",
                        timestamp=container.last_seen,
                        camera_id=container.camera_id,
                        value=dwell_time,
                        threshold=critical_upper,
                        metadata={
                            'container_id': container.id,
                            'container_number': container.container_number,
                            'first_seen': container.first_seen.isoformat(),
                            'mean_dwell': mean_dwell,
                            'std_dwell': std_dwell
                        }
                    ))
                elif dwell_time > upper_threshold:
                    alerts.append(Alert(
                        alert_type=AlertType.DWELL_TIME,
                        severity=AlertSeverity.HIGH,
                        title="High Container Dwell Time",
                        message=f"Container {container.container_number} has high dwell time: {dwell_time:.1f} hours (threshold: {upper_threshold:.1f})",
                        timestamp=container.last_seen,
                        camera_id=container.camera_id,
                        value=dwell_time,
                        threshold=upper_threshold,
                        metadata={
                            'container_id': container.id,
                            'container_number': container.container_number,
                            'first_seen': container.first_seen.isoformat()
                        }
                    ))
                
                # Unusually short dwell time (potential data quality issue)
                elif dwell_time < critical_lower and critical_lower > 0.5:  # Only alert if threshold is meaningful
                    alerts.append(Alert(
                        alert_type=AlertType.DATA_QUALITY,
                        severity=AlertSeverity.MEDIUM,
                        title="Unusually Short Dwell Time",
                        message=f"Container {container.container_number} has unusually short dwell time: {dwell_time:.1f} hours (threshold: {critical_lower:.1f})",
                        timestamp=container.last_seen,
                        camera_id=container.camera_id,
                        value=dwell_time,
                        threshold=critical_lower,
                        metadata={
                            'container_id': container.id,
                            'container_number': container.container_number,
                            'potential_issue': 'tracking_error'
                        }
                    ))
        
        return alerts


class ThroughputAlert:
    """Throughput anomaly detection."""
    
    def __init__(self, min_throughput_threshold: float = 0.5, deviation_threshold: float = 0.3):
        """
        Initialize throughput alert detector.
        
        Args:
            min_throughput_threshold: Minimum expected throughput ratio
            deviation_threshold: Threshold for throughput deviation (30% by default)
        """
        self.min_throughput_threshold = min_throughput_threshold
        self.deviation_threshold = deviation_threshold
        self.logger = logging.getLogger(__name__)
    
    def detect_anomalies(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None
    ) -> List[Alert]:
        """
        Detect throughput anomalies.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            camera_id: Specific camera to analyze
            
        Returns:
            List of throughput alerts
        """
        alerts = []
        
        # Get baseline data (previous period of same length)
        analysis_duration = end_date - start_date
        baseline_start = start_date - analysis_duration
        baseline_end = start_date
        
        metrics = ContainerMetrics()
        
        # Current period throughput
        current_throughput = metrics.calculate_throughput(
            start_date, end_date, camera_id, granularity='daily'
        )
        
        # Baseline period throughput
        baseline_throughput = metrics.calculate_throughput(
            baseline_start, baseline_end, camera_id, granularity='daily'
        )
        
        # Compare throughput
        current_avg = current_throughput['avg_throughput_per_period']
        baseline_avg = baseline_throughput['avg_throughput_per_period']
        
        if baseline_avg == 0:
            # No baseline data available
            if current_avg == 0:
                alerts.append(Alert(
                    alert_type=AlertType.THROUGHPUT,
                    severity=AlertSeverity.HIGH,
                    title="Zero Throughput Detected",
                    message="No container throughput detected in the analysis period",
                    timestamp=datetime.now(),
                    camera_id=camera_id,
                    value=current_avg,
                    threshold=self.min_throughput_threshold,
                    metadata={'analysis_period_days': analysis_duration.days}
                ))
            return alerts
        
        # Calculate deviation
        deviation = abs(current_avg - baseline_avg) / baseline_avg
        
        # Low throughput alert
        if current_avg < baseline_avg * self.min_throughput_threshold:
            severity = AlertSeverity.CRITICAL if deviation > 0.7 else AlertSeverity.HIGH
            alerts.append(Alert(
                alert_type=AlertType.THROUGHPUT,
                severity=severity,
                title="Low Throughput Alert",
                message=f"Current throughput ({current_avg:.1f}/day) is {deviation:.1%} below baseline ({baseline_avg:.1f}/day)",
                timestamp=datetime.now(),
                camera_id=camera_id,
                value=current_avg,
                threshold=baseline_avg * self.min_throughput_threshold,
                metadata={
                    'baseline_avg': baseline_avg,
                    'deviation_percent': deviation * 100,
                    'analysis_period_days': analysis_duration.days
                }
            ))
        
        # Significant deviation alert (up or down)
        elif deviation > self.deviation_threshold:
            direction = "above" if current_avg > baseline_avg else "below"
            severity = AlertSeverity.MEDIUM if deviation < 0.5 else AlertSeverity.HIGH
            
            alerts.append(Alert(
                alert_type=AlertType.THROUGHPUT,
                severity=severity,
                title="Throughput Deviation Alert",
                message=f"Current throughput ({current_avg:.1f}/day) is {deviation:.1%} {direction} baseline ({baseline_avg:.1f}/day)",
                timestamp=datetime.now(),
                camera_id=camera_id,
                value=current_avg,
                threshold=baseline_avg,
                metadata={
                    'baseline_avg': baseline_avg,
                    'deviation_percent': deviation * 100,
                    'direction': direction,
                    'analysis_period_days': analysis_duration.days
                }
            ))
        
        return alerts


class CongestionAlert:
    """Traffic congestion detection."""
    
    def __init__(self, congestion_threshold: int = 20, sustained_duration: int = 2):
        """
        Initialize congestion alert detector.
        
        Args:
            congestion_threshold: Number of containers to consider congestion
            sustained_duration: Hours of sustained activity to trigger alert
        """
        self.congestion_threshold = congestion_threshold
        self.sustained_duration = sustained_duration
        self.logger = logging.getLogger(__name__)
    
    def detect_anomalies(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None
    ) -> List[Alert]:
        """
        Detect traffic congestion anomalies.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            camera_id: Specific camera to analyze
            
        Returns:
            List of congestion alerts
        """
        alerts = []
        
        aggregator = DataAggregator()
        hourly_data = aggregator.aggregate_hourly_data(
            start_date, end_date, camera_id, ['detections']
        )
        
        if not hourly_data.data:
            return alerts
        
        # Convert to time series for congestion detection
        time_series = []
        for hour_str, data in hourly_data.data.items():
            hour_dt = datetime.fromisoformat(hour_str)
            time_series.append((hour_dt, data['detections']))
        
        # Sort by time
        time_series.sort(key=lambda x: x[0])
        
        # Detect sustained high activity periods
        congestion_periods = []
        current_period_start = None
        current_period_detections = []
        
        for hour_dt, detections in time_series:
            if detections >= self.congestion_threshold:
                if current_period_start is None:
                    current_period_start = hour_dt
                    current_period_detections = [detections]
                else:
                    # Check if this hour is consecutive to the previous
                    last_hour = current_period_start + timedelta(hours=len(current_period_detections)-1)
                    if hour_dt == last_hour + timedelta(hours=1):
                        current_period_detections.append(detections)
                    else:
                        # Gap in congestion - end current period and start new one
                        if len(current_period_detections) >= self.sustained_duration:
                            congestion_periods.append((
                                current_period_start,
                                last_hour,
                                current_period_detections
                            ))
                        current_period_start = hour_dt
                        current_period_detections = [detections]
            else:
                # End of congestion period
                if current_period_start and len(current_period_detections) >= self.sustained_duration:
                    last_hour = current_period_start + timedelta(hours=len(current_period_detections)-1)
                    congestion_periods.append((
                        current_period_start,
                        last_hour,
                        current_period_detections
                    ))
                current_period_start = None
                current_period_detections = []
        
        # Check final period
        if current_period_start and len(current_period_detections) >= self.sustained_duration:
            last_hour = current_period_start + timedelta(hours=len(current_period_detections)-1)
            congestion_periods.append((
                current_period_start,
                last_hour,
                current_period_detections
            ))
        
        # Generate alerts for congestion periods
        for start_time, end_time, detections_list in congestion_periods:
            duration_hours = len(detections_list)
            max_detections = max(detections_list)
            avg_detections = sum(detections_list) / len(detections_list)
            
            # Determine severity based on intensity and duration
            if max_detections > self.congestion_threshold * 2 or duration_hours > 6:
                severity = AlertSeverity.HIGH
            elif max_detections > self.congestion_threshold * 1.5 or duration_hours > 4:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            alerts.append(Alert(
                alert_type=AlertType.CONGESTION,
                severity=severity,
                title="Traffic Congestion Detected",
                message=f"Sustained high activity detected from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} on {start_time.strftime('%Y-%m-%d')}. Peak: {max_detections} detections/hour, Duration: {duration_hours} hours",
                timestamp=start_time,
                camera_id=camera_id,
                value=max_detections,
                threshold=self.congestion_threshold,
                metadata={
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_hours': duration_hours,
                    'max_detections': max_detections,
                    'avg_detections': avg_detections,
                    'detections_timeline': detections_list
                }
            ))
        
        return alerts


class AlertSystem:
    """Main alert system coordinator."""
    
    def __init__(
        self,
        email_config: Optional[Dict[str, str]] = None,
        enable_email: bool = False
    ):
        """
        Initialize the alert system.
        
        Args:
            email_config: Email configuration for notifications
            enable_email: Whether to enable email notifications
        """
        self.email_config = email_config or {}
        self.enable_email = enable_email
        self.logger = logging.getLogger(__name__)
        
        # Initialize alert detectors
        self.dwell_time_detector = DwellTimeAlert()
        self.throughput_detector = ThroughputAlert()
        self.congestion_detector = CongestionAlert()
    
    def detect_all_anomalies(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None
    ) -> Dict[str, List[Alert]]:
        """
        Run all anomaly detectors and return results.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            camera_id: Specific camera to analyze
            
        Returns:
            Dictionary of alerts by type
        """
        all_alerts = {
            'dwell_time': [],
            'throughput': [],
            'congestion': [],
            'system': []
        }
        
        try:
            # Dwell time anomalies
            all_alerts['dwell_time'] = self.dwell_time_detector.detect_anomalies(
                start_date, end_date, camera_id
            )
            
            # Throughput anomalies  
            all_alerts['throughput'] = self.throughput_detector.detect_anomalies(
                start_date, end_date, camera_id
            )
            
            # Congestion anomalies
            all_alerts['congestion'] = self.congestion_detector.detect_anomalies(
                start_date, end_date, camera_id
            )
            
        except Exception as e:
            self.logger.error(f"Error during anomaly detection: {e}")
            all_alerts['system'].append(Alert(
                alert_type=AlertType.SYSTEM,
                severity=AlertSeverity.HIGH,
                title="Anomaly Detection System Error",
                message=f"Error occurred during anomaly detection: {str(e)}",
                timestamp=datetime.now(),
                camera_id=camera_id,
                metadata={'error': str(e)}
            ))
        
        return all_alerts
    
    def send_alert_notification(self, alert: Alert) -> bool:
        """
        Send alert notification via configured channels.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if notification sent successfully
        """
        try:
            # Log alert
            self.logger.warning(f"ALERT: {alert.title} - {alert.message}")
            
            # Send email if configured
            if self.enable_email and self.email_config:
                return self._send_email_alert(alert)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending alert notification: {e}")
            return False
    
    def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('from_email')
            msg['To'] = self.email_config.get('to_email')
            msg['Subject'] = f"Container Analytics Alert: {alert.title}"
            
            # Email body
            body = f"""
            Alert Details:
            - Type: {alert.alert_type.value}
            - Severity: {alert.severity.value}
            - Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            - Camera: {alert.camera_id or 'All'}
            
            Message:
            {alert.message}
            
            Additional Information:
            - Value: {alert.value}
            - Threshold: {alert.threshold}
            - Metadata: {alert.metadata}
            
            This is an automated alert from the Container Analytics system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(
                self.email_config.get('smtp_server', 'smtp.gmail.com'),
                self.email_config.get('smtp_port', 587)
            )
            server.starttls()
            server.login(
                self.email_config.get('username'),
                self.email_config.get('password')
            )
            
            text = msg.as_string()
            server.sendmail(
                self.email_config.get('from_email'),
                self.email_config.get('to_email'),
                text
            )
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
            return False
    
    def get_alert_summary(
        self,
        alerts: Dict[str, List[Alert]]
    ) -> Dict[str, Any]:
        """
        Generate summary of alerts.
        
        Args:
            alerts: Dictionary of alerts by type
            
        Returns:
            Dictionary with alert summary statistics
        """
        total_alerts = sum(len(alert_list) for alert_list in alerts.values())
        
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        type_counts = {alert_type: len(alert_list) for alert_type, alert_list in alerts.items()}
        
        # Count by severity
        for alert_list in alerts.values():
            for alert in alert_list:
                severity_counts[alert.severity.value] += 1
        
        # Get most recent alerts
        all_alerts_flat = []
        for alert_list in alerts.values():
            all_alerts_flat.extend(alert_list)
        
        recent_alerts = sorted(all_alerts_flat, key=lambda x: x.timestamp, reverse=True)[:5]
        
        return {
            'total_alerts': total_alerts,
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'recent_alerts': [alert.to_dict() for alert in recent_alerts],
            'has_critical_alerts': severity_counts[AlertSeverity.CRITICAL.value] > 0,
            'has_high_priority_alerts': (
                severity_counts[AlertSeverity.CRITICAL.value] + 
                severity_counts[AlertSeverity.HIGH.value]
            ) > 0
        }


# Convenience functions for direct usage
def detect_anomalies(
    start_date: datetime,
    end_date: datetime,
    camera_id: Optional[str] = None,
    email_config: Optional[Dict[str, str]] = None
) -> Dict[str, List[Alert]]:
    """Detect all types of anomalies."""
    alert_system = AlertSystem(email_config, enable_email=bool(email_config))
    return alert_system.detect_all_anomalies(start_date, end_date, camera_id)


def send_alert_notification(
    alert: Alert,
    email_config: Optional[Dict[str, str]] = None
) -> bool:
    """Send alert notification."""
    alert_system = AlertSystem(email_config, enable_email=bool(email_config))
    return alert_system.send_alert_notification(alert)