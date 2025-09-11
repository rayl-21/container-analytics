"""
Analytics Engine Module for Container Analytics

This module provides analytics capabilities including:
- KPI calculations (metrics.py)
- Data aggregation (aggregator.py) 
- Anomaly detection and alerting (alerts.py)
"""

from .metrics import (
    ContainerMetrics,
    calculate_dwell_time,
    calculate_throughput,
    calculate_gate_efficiency,
    analyze_peak_hours,
    get_container_type_distribution
)

from .aggregator import (
    DataAggregator,
    aggregate_hourly_data,
    aggregate_daily_data,
    calculate_rolling_averages,
    generate_summary_statistics
)

from .alerts import (
    AlertSystem,
    DwellTimeAlert,
    ThroughputAlert,
    CongestionAlert,
    detect_anomalies,
    send_alert_notification
)

__all__ = [
    # Metrics
    'ContainerMetrics',
    'calculate_dwell_time',
    'calculate_throughput', 
    'calculate_gate_efficiency',
    'analyze_peak_hours',
    'get_container_type_distribution',
    
    # Aggregator
    'DataAggregator',
    'aggregate_hourly_data',
    'aggregate_daily_data', 
    'calculate_rolling_averages',
    'generate_summary_statistics',
    
    # Alerts
    'AlertSystem',
    'DwellTimeAlert',
    'ThroughputAlert', 
    'CongestionAlert',
    'detect_anomalies',
    'send_alert_notification',
]