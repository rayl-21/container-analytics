"""
Data Aggregation Module for Container Analytics

Provides data aggregation pipeline that:
- Aggregates detection data by time periods (hourly, daily, weekly)
- Calculates rolling averages
- Generates summary statistics  
- Handles data from database with efficient queries
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc

from ..database.models import (
    session_scope, Image, Detection, Container, Metric
)

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Container for aggregation results."""
    data: Dict[str, Any]
    period: str
    start_date: datetime
    end_date: datetime
    total_records: int


class DataAggregator:
    """Main class for aggregating container analytics data."""
    
    def __init__(self):
        """Initialize the data aggregator."""
        self.logger = logging.getLogger(__name__)
    
    def aggregate_hourly_data(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None,
        metrics: List[str] = None
    ) -> AggregationResult:
        """
        Aggregate data by hourly periods.
        
        Args:
            start_date: Start date for aggregation
            end_date: End date for aggregation  
            camera_id: Specific camera to analyze
            metrics: List of metrics to calculate ['detections', 'throughput', 'dwell_time']
            
        Returns:
            AggregationResult with hourly aggregated data
        """
        if metrics is None:
            metrics = ['detections', 'throughput', 'dwell_time']
        
        hourly_data = defaultdict(lambda: {
            'detections': 0,
            'throughput': 0, 
            'dwell_time': [],
            'containers': 0,
            'avg_confidence': 0.0,
            'confidence_values': []
        })
        
        with session_scope() as session:
            # Get images in the time range
            image_query = session.query(Image).filter(
                Image.timestamp >= start_date,
                Image.timestamp <= end_date
            )
            
            if camera_id:
                image_query = image_query.filter(Image.camera_id == camera_id)
            
            images = image_query.all()
            
            # Process each image
            for image in images:
                hour_key = image.timestamp.replace(minute=0, second=0, microsecond=0)
                
                # Count detections for this image
                if 'detections' in metrics:
                    detection_count = session.query(Detection).filter(
                        Detection.image_id == image.id
                    ).count()
                    
                    hourly_data[hour_key]['detections'] += detection_count
                    
                    # Get confidence values
                    detections = session.query(Detection).filter(
                        Detection.image_id == image.id
                    ).all()
                    
                    for detection in detections:
                        hourly_data[hour_key]['confidence_values'].append(detection.confidence)
            
            # Get container data for throughput and dwell time
            if 'throughput' in metrics or 'dwell_time' in metrics:
                container_query = session.query(Container).filter(
                    Container.last_seen >= start_date,
                    Container.last_seen <= end_date
                )
                
                if camera_id:
                    container_query = container_query.filter(Container.camera_id == camera_id)
                
                containers = container_query.all()
                
                for container in containers:
                    hour_key = container.last_seen.replace(minute=0, second=0, microsecond=0)
                    
                    if 'throughput' in metrics and container.status == 'departed':
                        hourly_data[hour_key]['throughput'] += 1
                    
                    if 'dwell_time' in metrics and container.dwell_time is not None:
                        hourly_data[hour_key]['dwell_time'].append(container.dwell_time)
                    
                    hourly_data[hour_key]['containers'] += 1
        
        # Calculate averages and clean up data
        processed_data = {}
        for hour_key, data in hourly_data.items():
            processed_data[hour_key.isoformat()] = {
                'detections': data['detections'],
                'throughput': data['throughput'], 
                'containers': data['containers'],
                'avg_dwell_time': np.mean(data['dwell_time']) if data['dwell_time'] else 0.0,
                'median_dwell_time': np.median(data['dwell_time']) if data['dwell_time'] else 0.0,
                'avg_confidence': np.mean(data['confidence_values']) if data['confidence_values'] else 0.0,
                'total_confidence_samples': len(data['confidence_values'])
            }
        
        return AggregationResult(
            data=processed_data,
            period='hourly',
            start_date=start_date,
            end_date=end_date,
            total_records=len(processed_data)
        )
    
    def aggregate_daily_data(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None,
        metrics: List[str] = None
    ) -> AggregationResult:
        """
        Aggregate data by daily periods.
        
        Args:
            start_date: Start date for aggregation
            end_date: End date for aggregation
            camera_id: Specific camera to analyze  
            metrics: List of metrics to calculate
            
        Returns:
            AggregationResult with daily aggregated data
        """
        if metrics is None:
            metrics = ['detections', 'throughput', 'dwell_time']
        
        daily_data = defaultdict(lambda: {
            'detections': 0,
            'throughput': 0,
            'dwell_time': [],
            'containers': 0,
            'confidence_values': [],
            'peak_hour': {'hour': 0, 'activity': 0},
            'hourly_breakdown': defaultdict(int)
        })
        
        with session_scope() as session:
            # Use existing hourly aggregation and roll up to daily
            hourly_result = self.aggregate_hourly_data(start_date, end_date, camera_id, metrics)
            
            for hour_iso, hour_data in hourly_result.data.items():
                hour_dt = datetime.fromisoformat(hour_iso)
                day_key = hour_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                
                daily_data[day_key]['detections'] += hour_data['detections']
                daily_data[day_key]['throughput'] += hour_data['throughput']
                daily_data[day_key]['containers'] += hour_data['containers']
                
                # Track hourly breakdown for peak analysis
                daily_data[day_key]['hourly_breakdown'][hour_dt.hour] += hour_data['detections']
                
                # Update peak hour if this hour has more activity
                if hour_data['detections'] > daily_data[day_key]['peak_hour']['activity']:
                    daily_data[day_key]['peak_hour'] = {
                        'hour': hour_dt.hour,
                        'activity': hour_data['detections']
                    }
                
                # Collect confidence values (approximate)
                if hour_data['total_confidence_samples'] > 0:
                    # Approximate confidence values based on average
                    avg_conf = hour_data['avg_confidence']
                    sample_count = hour_data['total_confidence_samples']
                    daily_data[day_key]['confidence_values'].extend([avg_conf] * sample_count)
            
            # Get dwell time data directly for more accurate daily aggregation
            if 'dwell_time' in metrics:
                container_query = session.query(Container).filter(
                    Container.last_seen >= start_date,
                    Container.last_seen <= end_date,
                    Container.dwell_time.isnot(None)
                )
                
                if camera_id:
                    container_query = container_query.filter(Container.camera_id == camera_id)
                
                containers = container_query.all()
                
                for container in containers:
                    day_key = container.last_seen.replace(hour=0, minute=0, second=0, microsecond=0)
                    daily_data[day_key]['dwell_time'].append(container.dwell_time)
        
        # Calculate daily statistics
        processed_data = {}
        for day_key, data in daily_data.items():
            processed_data[day_key.isoformat()] = {
                'detections': data['detections'],
                'throughput': data['throughput'],
                'containers': data['containers'],
                'avg_dwell_time': np.mean(data['dwell_time']) if data['dwell_time'] else 0.0,
                'median_dwell_time': np.median(data['dwell_time']) if data['dwell_time'] else 0.0,
                'std_dwell_time': np.std(data['dwell_time']) if data['dwell_time'] else 0.0,
                'avg_confidence': np.mean(data['confidence_values']) if data['confidence_values'] else 0.0,
                'peak_hour': data['peak_hour'],
                'hourly_distribution': dict(data['hourly_breakdown']),
                'total_dwell_samples': len(data['dwell_time']),
                'total_confidence_samples': len(data['confidence_values'])
            }
        
        return AggregationResult(
            data=processed_data,
            period='daily', 
            start_date=start_date,
            end_date=end_date,
            total_records=len(processed_data)
        )
    
    def aggregate_weekly_data(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None,
        metrics: List[str] = None
    ) -> AggregationResult:
        """
        Aggregate data by weekly periods.
        
        Args:
            start_date: Start date for aggregation
            end_date: End date for aggregation
            camera_id: Specific camera to analyze
            metrics: List of metrics to calculate
            
        Returns:
            AggregationResult with weekly aggregated data
        """
        if metrics is None:
            metrics = ['detections', 'throughput', 'dwell_time']
        
        # Get daily data first
        daily_result = self.aggregate_daily_data(start_date, end_date, camera_id, metrics)
        
        weekly_data = defaultdict(lambda: {
            'detections': 0,
            'throughput': 0,
            'containers': 0,
            'dwell_times': [],
            'confidence_values': [],
            'daily_breakdown': {},
            'peak_day': {'date': None, 'activity': 0},
            'busiest_hour': {'hour': 0, 'activity': 0}
        })
        
        for day_iso, day_data in daily_result.data.items():
            day_dt = datetime.fromisoformat(day_iso)
            
            # Get start of week (Monday)
            days_since_monday = day_dt.weekday()
            week_start = (day_dt - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            
            weekly_data[week_start]['detections'] += day_data['detections']
            weekly_data[week_start]['throughput'] += day_data['throughput']
            weekly_data[week_start]['containers'] += day_data['containers']
            
            # Store daily breakdown
            weekly_data[week_start]['daily_breakdown'][day_dt.strftime('%A')] = day_data['detections']
            
            # Track peak day
            if day_data['detections'] > weekly_data[week_start]['peak_day']['activity']:
                weekly_data[week_start]['peak_day'] = {
                    'date': day_dt.strftime('%Y-%m-%d'),
                    'day_name': day_dt.strftime('%A'),
                    'activity': day_data['detections']
                }
            
            # Track busiest hour across the week
            if day_data['peak_hour']['activity'] > weekly_data[week_start]['busiest_hour']['activity']:
                weekly_data[week_start]['busiest_hour'] = {
                    'hour': day_data['peak_hour']['hour'],
                    'activity': day_data['peak_hour']['activity'],
                    'date': day_dt.strftime('%Y-%m-%d')
                }
            
            # Approximate dwell times and confidence (for aggregation)
            if day_data['total_dwell_samples'] > 0:
                # Add approximate dwell times
                avg_dwell = day_data['avg_dwell_time']
                weekly_data[week_start]['dwell_times'].extend([avg_dwell] * day_data['total_dwell_samples'])
            
            if day_data['total_confidence_samples'] > 0:
                avg_conf = day_data['avg_confidence'] 
                weekly_data[week_start]['confidence_values'].extend([avg_conf] * day_data['total_confidence_samples'])
        
        # Calculate weekly statistics
        processed_data = {}
        for week_start, data in weekly_data.items():
            week_end = week_start + timedelta(days=6)
            
            processed_data[week_start.isoformat()] = {
                'week_start': week_start.strftime('%Y-%m-%d'),
                'week_end': week_end.strftime('%Y-%m-%d'),
                'detections': data['detections'],
                'throughput': data['throughput'],
                'containers': data['containers'],
                'avg_dwell_time': np.mean(data['dwell_times']) if data['dwell_times'] else 0.0,
                'median_dwell_time': np.median(data['dwell_times']) if data['dwell_times'] else 0.0,
                'avg_confidence': np.mean(data['confidence_values']) if data['confidence_values'] else 0.0,
                'daily_breakdown': data['daily_breakdown'],
                'peak_day': data['peak_day'],
                'busiest_hour': data['busiest_hour'],
                'avg_daily_detections': data['detections'] / 7.0,
                'avg_daily_throughput': data['throughput'] / 7.0
            }
        
        return AggregationResult(
            data=processed_data,
            period='weekly',
            start_date=start_date, 
            end_date=end_date,
            total_records=len(processed_data)
        )
    
    def calculate_rolling_averages(
        self,
        data: Dict[str, Any],
        window_size: int = 7,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate rolling averages for time series data.
        
        Args:
            data: Time series data dictionary
            window_size: Rolling window size (e.g., 7 for weekly rolling average)
            metrics: List of metrics to calculate rolling averages for
            
        Returns:
            Dictionary with rolling averages
        """
        if metrics is None:
            metrics = ['detections', 'throughput', 'avg_dwell_time']
        
        # Convert to DataFrame for easier rolling calculations
        df_data = []
        for timestamp, values in data.items():
            row = {'timestamp': timestamp}
            row.update(values)
            df_data.append(row)
        
        if not df_data:
            return {}
        
        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        rolling_data = {}
        
        for metric in metrics:
            if metric in df.columns:
                # Calculate rolling average
                rolling_avg = df[metric].rolling(window=window_size, min_periods=1).mean()
                
                # Calculate rolling standard deviation
                rolling_std = df[metric].rolling(window=window_size, min_periods=1).std()
                
                # Store results
                for i, timestamp in enumerate(df['timestamp']):
                    timestamp_str = timestamp.isoformat()
                    
                    if timestamp_str not in rolling_data:
                        rolling_data[timestamp_str] = {}
                    
                    rolling_data[timestamp_str][f'{metric}_rolling_avg'] = rolling_avg.iloc[i]
                    rolling_data[timestamp_str][f'{metric}_rolling_std'] = rolling_std.iloc[i]
                    rolling_data[timestamp_str][f'{metric}_original'] = df[metric].iloc[i]
        
        return rolling_data
    
    def generate_summary_statistics(
        self,
        start_date: datetime,
        end_date: datetime,
        camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            camera_id: Specific camera to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'duration_days': (end_date - start_date).days,
                'camera_id': camera_id
            },
            'totals': {},
            'averages': {},
            'trends': {},
            'distributions': {}
        }
        
        with session_scope() as session:
            # Total counts
            image_query = session.query(Image).filter(
                Image.timestamp >= start_date,
                Image.timestamp <= end_date
            )
            if camera_id:
                image_query = image_query.filter(Image.camera_id == camera_id)
            
            total_images = image_query.count()
            
            detection_query = session.query(Detection).join(Image).filter(
                Image.timestamp >= start_date,
                Image.timestamp <= end_date
            )
            if camera_id:
                detection_query = detection_query.filter(Image.camera_id == camera_id)
            
            total_detections = detection_query.count()
            
            container_query = session.query(Container).filter(
                Container.last_seen >= start_date,
                Container.last_seen <= end_date
            )
            if camera_id:
                container_query = container_query.filter(Container.camera_id == camera_id)
            
            total_containers = container_query.count()
            departed_containers = container_query.filter(Container.status == 'departed').count()
            
            summary['totals'] = {
                'images': total_images,
                'detections': total_detections,
                'containers': total_containers,
                'departed_containers': departed_containers,
                'detections_per_image': total_detections / total_images if total_images > 0 else 0.0
            }
            
            # Average calculations
            avg_confidence = session.query(func.avg(Detection.confidence)).join(Image).filter(
                Image.timestamp >= start_date,
                Image.timestamp <= end_date
            )
            if camera_id:
                avg_confidence = avg_confidence.filter(Image.camera_id == camera_id)
            
            avg_confidence_result = avg_confidence.scalar()
            
            avg_dwell = session.query(func.avg(Container.dwell_time)).filter(
                Container.last_seen >= start_date,
                Container.last_seen <= end_date,
                Container.dwell_time.isnot(None)
            )
            if camera_id:
                avg_dwell = avg_dwell.filter(Container.camera_id == camera_id)
            
            avg_dwell_result = avg_dwell.scalar()
            
            duration_days = (end_date - start_date).days or 1
            
            summary['averages'] = {
                'confidence': float(avg_confidence_result) if avg_confidence_result else 0.0,
                'dwell_time_hours': float(avg_dwell_result) if avg_dwell_result else 0.0,
                'images_per_day': total_images / duration_days,
                'detections_per_day': total_detections / duration_days,
                'containers_per_day': total_containers / duration_days,
                'throughput_per_day': departed_containers / duration_days
            }
            
            # Get daily aggregation for trend analysis
            daily_data = self.aggregate_daily_data(start_date, end_date, camera_id)
            
            if daily_data.data:
                daily_detections = [data['detections'] for data in daily_data.data.values()]
                daily_throughput = [data['throughput'] for data in daily_data.data.values()]
                
                # Calculate trends (simple linear regression slope)
                if len(daily_detections) > 1:
                    days = list(range(len(daily_detections)))
                    detection_trend = np.polyfit(days, daily_detections, 1)[0]
                    throughput_trend = np.polyfit(days, daily_throughput, 1)[0] if daily_throughput else 0.0
                else:
                    detection_trend = 0.0
                    throughput_trend = 0.0
                
                summary['trends'] = {
                    'detection_trend': float(detection_trend),
                    'throughput_trend': float(throughput_trend),
                    'trend_interpretation': {
                        'detection': 'increasing' if detection_trend > 0 else 'decreasing' if detection_trend < 0 else 'stable',
                        'throughput': 'increasing' if throughput_trend > 0 else 'decreasing' if throughput_trend < 0 else 'stable'
                    }
                }
                
                # Distribution statistics
                summary['distributions'] = {
                    'daily_detections': {
                        'min': min(daily_detections),
                        'max': max(daily_detections), 
                        'std': float(np.std(daily_detections)),
                        'median': float(np.median(daily_detections))
                    },
                    'daily_throughput': {
                        'min': min(daily_throughput) if daily_throughput else 0,
                        'max': max(daily_throughput) if daily_throughput else 0,
                        'std': float(np.std(daily_throughput)) if daily_throughput else 0.0,
                        'median': float(np.median(daily_throughput)) if daily_throughput else 0.0
                    }
                }
        
        return summary


# Convenience functions for direct usage
def aggregate_hourly_data(
    start_date: datetime,
    end_date: datetime,
    camera_id: Optional[str] = None,
    metrics: List[str] = None
) -> AggregationResult:
    """Aggregate data by hourly periods."""
    aggregator = DataAggregator()
    return aggregator.aggregate_hourly_data(start_date, end_date, camera_id, metrics)


def aggregate_daily_data(
    start_date: datetime,
    end_date: datetime,
    camera_id: Optional[str] = None,
    metrics: List[str] = None
) -> AggregationResult:
    """Aggregate data by daily periods."""
    aggregator = DataAggregator()
    return aggregator.aggregate_daily_data(start_date, end_date, camera_id, metrics)


def aggregate_weekly_data(
    start_date: datetime,
    end_date: datetime,
    camera_id: Optional[str] = None,
    metrics: List[str] = None
) -> AggregationResult:
    """Aggregate data by weekly periods."""
    aggregator = DataAggregator()
    return aggregator.aggregate_weekly_data(start_date, end_date, camera_id, metrics)


def calculate_rolling_averages(
    data: Dict[str, Any],
    window_size: int = 7,
    metrics: List[str] = None
) -> Dict[str, Any]:
    """Calculate rolling averages for time series data."""
    aggregator = DataAggregator()
    return aggregator.calculate_rolling_averages(data, window_size, metrics)


def generate_summary_statistics(
    start_date: datetime,
    end_date: datetime,
    camera_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate comprehensive summary statistics."""
    aggregator = DataAggregator()
    return aggregator.generate_summary_statistics(start_date, end_date, camera_id)