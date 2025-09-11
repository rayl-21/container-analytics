"""
Container Analytics Metrics Module

Provides KPI calculations including:
- Container dwell time calculations
- Terminal throughput metrics  
- Gate efficiency calculations
- Peak hour analysis
- Container type distribution
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..database.models import (
    session_scope, Image, Detection, Container, Metric
)

logger = logging.getLogger(__name__)


@dataclass
class KPIResult:
    """Container for KPI calculation results."""
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


class ContainerMetrics:
    """Main class for calculating container analytics KPIs."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_dwell_time(
        self,
        container_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate container dwell times.
        
        Args:
            container_id: Specific container to analyze
            start_date: Start date for analysis
            end_date: End date for analysis  
            camera_id: Specific camera to analyze
            
        Returns:
            Dictionary with dwell time statistics
        """
        with session_scope() as session:
            query = session.query(Container)
            
            # Apply filters
            if container_id:
                query = query.filter(Container.id == container_id)
            if start_date:
                query = query.filter(Container.first_seen >= start_date)
            if end_date:
                query = query.filter(Container.last_seen <= end_date)
            if camera_id:
                query = query.filter(Container.camera_id == camera_id)
                
            containers = query.all()
            
            if not containers:
                return {
                    'total_containers': 0,
                    'avg_dwell_time': 0.0,
                    'median_dwell_time': 0.0,
                    'min_dwell_time': 0.0,
                    'max_dwell_time': 0.0,
                    'dwell_times': []
                }
            
            # Calculate dwell times
            dwell_times = []
            for container in containers:
                if container.dwell_time is not None:
                    dwell_times.append(container.dwell_time)
                else:
                    # Calculate if not already calculated
                    dwell_time = container.calculate_dwell_time()
                    if dwell_time:
                        dwell_times.append(dwell_time)
            
            if not dwell_times:
                return {
                    'total_containers': len(containers),
                    'avg_dwell_time': 0.0,
                    'median_dwell_time': 0.0,
                    'min_dwell_time': 0.0,
                    'max_dwell_time': 0.0,
                    'dwell_times': []
                }
            
            df = pd.Series(dwell_times)
            
            return {
                'total_containers': len(containers),
                'avg_dwell_time': float(df.mean()),
                'median_dwell_time': float(df.median()),
                'min_dwell_time': float(df.min()),
                'max_dwell_time': float(df.max()),
                'std_dwell_time': float(df.std()),
                'dwell_times': dwell_times,
                'percentile_25': float(df.quantile(0.25)),
                'percentile_75': float(df.quantile(0.75)),
                'percentile_95': float(df.quantile(0.95))
            }
    
    def calculate_throughput(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None,
        granularity: str = 'hourly'
    ) -> Dict[str, Any]:
        """
        Calculate terminal throughput metrics.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            camera_id: Specific camera to analyze
            granularity: 'hourly', 'daily', or 'weekly'
            
        Returns:
            Dictionary with throughput statistics
        """
        with session_scope() as session:
            # Query container movements (departures)
            query = session.query(Container).filter(
                Container.status == 'departed',
                Container.last_seen >= start_date,
                Container.last_seen <= end_date
            )
            
            if camera_id:
                query = query.filter(Container.camera_id == camera_id)
            
            containers = query.all()
            
            if not containers:
                return {
                    'total_throughput': 0,
                    'avg_throughput_per_period': 0.0,
                    'peak_throughput': 0,
                    'throughput_by_period': {},
                    'total_periods': 0
                }
            
            # Group by time periods
            throughput_data = defaultdict(int)
            
            for container in containers:
                if granularity == 'hourly':
                    period_key = container.last_seen.replace(minute=0, second=0, microsecond=0)
                elif granularity == 'daily':
                    period_key = container.last_seen.replace(hour=0, minute=0, second=0, microsecond=0)
                elif granularity == 'weekly':
                    # Get start of week (Monday)
                    days_since_monday = container.last_seen.weekday()
                    period_key = (container.last_seen - timedelta(days=days_since_monday)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                else:
                    raise ValueError(f"Invalid granularity: {granularity}")
                
                throughput_data[period_key] += 1
            
            throughput_values = list(throughput_data.values())
            total_throughput = sum(throughput_values)
            
            return {
                'total_throughput': total_throughput,
                'avg_throughput_per_period': float(sum(throughput_values) / len(throughput_values)) if throughput_values else 0.0,
                'peak_throughput': max(throughput_values) if throughput_values else 0,
                'min_throughput': min(throughput_values) if throughput_values else 0,
                'throughput_by_period': {k.isoformat(): v for k, v in throughput_data.items()},
                'total_periods': len(throughput_data),
                'granularity': granularity
            }
    
    def calculate_gate_efficiency(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate gate efficiency metrics.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            camera_id: Specific camera to analyze
            
        Returns:
            Dictionary with efficiency statistics
        """
        with session_scope() as session:
            # Get metrics data for the period
            query = session.query(Metric).filter(
                Metric.date >= start_date,
                Metric.date <= end_date
            )
            
            if camera_id:
                query = query.filter(Metric.camera_id == camera_id)
            
            metrics = query.all()
            
            if not metrics:
                return {
                    'avg_processing_time': 0.0,
                    'gate_utilization': 0.0,
                    'efficiency_score': 0.0,
                    'total_processed': 0,
                    'peak_efficiency_hour': None
                }
            
            # Calculate efficiency metrics
            total_throughput = sum(m.throughput for m in metrics if m.throughput)
            total_hours = len(metrics)
            avg_dwell_times = [m.avg_dwell_time for m in metrics if m.avg_dwell_time is not None]
            
            # Gate utilization (percentage of time with activity)
            active_periods = sum(1 for m in metrics if m.total_detections > 0)
            gate_utilization = (active_periods / total_hours * 100) if total_hours > 0 else 0.0
            
            # Average processing time (inverse of throughput)
            avg_processing_time = sum(avg_dwell_times) / len(avg_dwell_times) if avg_dwell_times else 0.0
            
            # Efficiency score (throughput per hour / average dwell time)
            avg_throughput_per_hour = total_throughput / total_hours if total_hours > 0 else 0.0
            efficiency_score = (avg_throughput_per_hour / avg_processing_time) if avg_processing_time > 0 else 0.0
            
            # Find peak efficiency hour
            peak_efficiency_hour = None
            max_efficiency = 0.0
            for metric in metrics:
                if metric.throughput and metric.avg_dwell_time:
                    efficiency = metric.throughput / metric.avg_dwell_time
                    if efficiency > max_efficiency:
                        max_efficiency = efficiency
                        peak_efficiency_hour = f"{metric.date.strftime('%Y-%m-%d')} {metric.hour}:00"
            
            return {
                'avg_processing_time': avg_processing_time,
                'gate_utilization': gate_utilization,
                'efficiency_score': efficiency_score,
                'total_processed': total_throughput,
                'avg_throughput_per_hour': avg_throughput_per_hour,
                'peak_efficiency_hour': peak_efficiency_hour,
                'total_analysis_hours': total_hours
            }
    
    def analyze_peak_hours(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze peak hours for container activity.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            camera_id: Specific camera to analyze
            top_n: Number of top peak hours to return
            
        Returns:
            Dictionary with peak hour analysis
        """
        with session_scope() as session:
            query = session.query(Metric).filter(
                Metric.date >= start_date,
                Metric.date <= end_date
            )
            
            if camera_id:
                query = query.filter(Metric.camera_id == camera_id)
            
            metrics = query.all()
            
            if not metrics:
                return {
                    'peak_hours': [],
                    'hourly_averages': {},
                    'busiest_day_hour': None,
                    'quietest_day_hour': None
                }
            
            # Group by hour of day
            hourly_data = defaultdict(list)
            for metric in metrics:
                hourly_data[metric.hour].append(metric.total_detections or 0)
            
            # Calculate hourly averages
            hourly_averages = {}
            for hour, detections_list in hourly_data.items():
                hourly_averages[hour] = sum(detections_list) / len(detections_list)
            
            # Sort hours by average activity
            sorted_hours = sorted(hourly_averages.items(), key=lambda x: x[1], reverse=True)
            
            # Get peak hours
            peak_hours = []
            for hour, avg_activity in sorted_hours[:top_n]:
                peak_hours.append({
                    'hour': hour,
                    'avg_activity': avg_activity,
                    'time_range': f"{hour:02d}:00 - {hour+1:02d}:00"
                })
            
            # Find busiest and quietest specific day-hours
            busiest_metric = max(metrics, key=lambda m: m.total_detections or 0)
            quietest_metric = min(metrics, key=lambda m: m.total_detections or 0)
            
            return {
                'peak_hours': peak_hours,
                'hourly_averages': {f"{h:02d}:00": round(avg, 2) for h, avg in hourly_averages.items()},
                'busiest_day_hour': {
                    'date': busiest_metric.date.strftime('%Y-%m-%d'),
                    'hour': busiest_metric.hour,
                    'activity': busiest_metric.total_detections
                },
                'quietest_day_hour': {
                    'date': quietest_metric.date.strftime('%Y-%m-%d'),
                    'hour': quietest_metric.hour,
                    'activity': quietest_metric.total_detections
                }
            }
    
    def get_container_type_distribution(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get distribution of container types detected.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            camera_id: Specific camera to analyze
            
        Returns:
            Dictionary with container type distribution
        """
        with session_scope() as session:
            # Join detections with images to get timestamp filtering
            query = session.query(Detection).join(Image).filter(
                Image.timestamp >= start_date,
                Image.timestamp <= end_date,
                Detection.object_type == 'container'  # Only container detections
            )
            
            if camera_id:
                query = query.filter(Image.camera_id == camera_id)
            
            detections = query.all()
            
            if not detections:
                return {
                    'total_detections': 0,
                    'type_distribution': {},
                    'confidence_stats': {},
                    'size_distribution': {}
                }
            
            # Analyze container sizes based on bounding box dimensions
            size_categories = defaultdict(int)
            confidence_values = []
            
            for detection in detections:
                confidence_values.append(detection.confidence)
                
                # Categorize by bounding box size (rough estimation)
                bbox_area = detection.bbox_width * detection.bbox_height
                if bbox_area < 0.1:  # Small container
                    size_categories['small'] += 1
                elif bbox_area < 0.3:  # Medium container  
                    size_categories['medium'] += 1
                else:  # Large container
                    size_categories['large'] += 1
            
            # Calculate confidence statistics
            conf_df = pd.Series(confidence_values)
            
            return {
                'total_detections': len(detections),
                'type_distribution': {
                    'containers': len(detections)  # Could be expanded with more types
                },
                'confidence_stats': {
                    'avg_confidence': float(conf_df.mean()),
                    'min_confidence': float(conf_df.min()),
                    'max_confidence': float(conf_df.max()),
                    'std_confidence': float(conf_df.std())
                },
                'size_distribution': dict(size_categories),
                'size_percentages': {
                    size: (count / len(detections) * 100) 
                    for size, count in size_categories.items()
                }
            }


# Convenience functions for direct usage
def calculate_dwell_time(
    container_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    camera_id: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate container dwell times."""
    metrics = ContainerMetrics()
    return metrics.calculate_dwell_time(container_id, start_date, end_date, camera_id)


def calculate_throughput(
    start_date: datetime,
    end_date: datetime,
    camera_id: Optional[str] = None,
    granularity: str = 'hourly'
) -> Dict[str, Any]:
    """Calculate terminal throughput metrics."""
    metrics = ContainerMetrics()
    return metrics.calculate_throughput(start_date, end_date, camera_id, granularity)


def calculate_gate_efficiency(
    start_date: datetime,
    end_date: datetime,
    camera_id: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate gate efficiency metrics."""
    metrics = ContainerMetrics()
    return metrics.calculate_gate_efficiency(start_date, end_date, camera_id)


def analyze_peak_hours(
    start_date: datetime,
    end_date: datetime,
    camera_id: Optional[str] = None,
    top_n: int = 5
) -> Dict[str, Any]:
    """Analyze peak hours for container activity."""
    metrics = ContainerMetrics()
    return metrics.analyze_peak_hours(start_date, end_date, camera_id, top_n)


def get_container_type_distribution(
    start_date: datetime,
    end_date: datetime,
    camera_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get distribution of container types detected."""
    metrics = ContainerMetrics()
    return metrics.get_container_type_distribution(start_date, end_date, camera_id)