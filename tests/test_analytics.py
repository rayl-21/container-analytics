"""
Comprehensive tests for the analytics module.

Tests cover:
- Metric calculations (dwell time, throughput, gate efficiency)
- Data aggregation (hourly, daily, weekly)
- Alert detection (dwell time, throughput, congestion)
- Peak hour analysis
- Container type distribution
- Rolling averages
- Summary statistics
- Error handling and edge cases
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from collections import defaultdict

from modules.analytics.metrics import (
    ContainerMetrics, KPIResult, 
    calculate_dwell_time, calculate_throughput, calculate_gate_efficiency,
    analyze_peak_hours, get_container_type_distribution
)

from modules.analytics.aggregator import (
    DataAggregator, AggregationResult,
    aggregate_hourly_data, aggregate_daily_data, aggregate_weekly_data,
    calculate_rolling_averages, generate_summary_statistics
)

from modules.analytics.alerts import (
    AlertSystem, DwellTimeAlert, ThroughputAlert, CongestionAlert,
    Alert, AlertType, AlertSeverity,
    detect_anomalies, send_alert_notification
)


# =============================================================================
# Test Metrics Module
# =============================================================================

class TestKPIResult:
    """Test KPIResult dataclass."""
    
    def test_kpi_result_creation(self):
        """Test KPIResult creation with all fields."""
        timestamp = datetime.utcnow()
        metadata = {'source': 'test', 'version': '1.0'}
        
        result = KPIResult(
            value=42.5,
            unit="hours",
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert result.value == 42.5
        assert result.unit == "hours"
        assert result.timestamp == timestamp
        assert result.metadata == metadata
    
    def test_kpi_result_minimal(self):
        """Test KPIResult creation with minimal fields."""
        timestamp = datetime.utcnow()
        
        result = KPIResult(
            value=10.0,
            unit="count",
            timestamp=timestamp
        )
        
        assert result.value == 10.0
        assert result.unit == "count"
        assert result.timestamp == timestamp
        assert result.metadata is None


class TestContainerMetrics:
    """Test ContainerMetrics class methods."""
    
    def test_init(self):
        """Test ContainerMetrics initialization."""
        metrics = ContainerMetrics()
        assert metrics.logger is not None
    
    @patch('modules.analytics.metrics.session_scope')
    def test_calculate_dwell_time_no_containers(self, mock_session_scope):
        """Test dwell time calculation with no containers."""
        mock_session = Mock()
        mock_session.query.return_value.all.return_value = []
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        result = metrics.calculate_dwell_time()
        
        expected = {
            'total_containers': 0,
            'avg_dwell_time': 0.0,
            'median_dwell_time': 0.0,
            'min_dwell_time': 0.0,
            'max_dwell_time': 0.0,
            'dwell_times': []
        }
        assert result == expected
    
    @patch('modules.analytics.metrics.session_scope')
    def test_calculate_dwell_time_with_data(self, mock_session_scope, sample_container_data):
        """Test dwell time calculation with container data."""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value = mock_session.query.return_value
        mock_session.query.return_value.all.return_value = sample_container_data[:3]
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        result = metrics.calculate_dwell_time()
        
        assert result['total_containers'] == 3
        assert result['avg_dwell_time'] > 0
        assert result['min_dwell_time'] >= 0
        assert result['max_dwell_time'] >= result['min_dwell_time']
        assert len(result['dwell_times']) == 3
        assert 'std_dwell_time' in result
        assert 'percentile_25' in result
        assert 'percentile_75' in result
        assert 'percentile_95' in result
    
    @patch('modules.analytics.metrics.session_scope')
    def test_calculate_dwell_time_with_filters(self, mock_session_scope):
        """Test dwell time calculation with various filters."""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        metrics.calculate_dwell_time(
            container_id=123,
            start_date=start_date,
            end_date=end_date,
            camera_id="camera_1"
        )
        
        # Verify filters were applied
        assert mock_query.filter.call_count == 4
    
    @patch('modules.analytics.metrics.session_scope')
    def test_calculate_throughput_no_containers(self, mock_session_scope):
        """Test throughput calculation with no data."""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = metrics.calculate_throughput(start_date, end_date)
        
        expected = {
            'total_throughput': 0,
            'avg_throughput_per_period': 0.0,
            'peak_throughput': 0,
            'throughput_by_period': {},
            'total_periods': 0
        }
        assert result == expected
    
    @patch('modules.analytics.metrics.session_scope')
    def test_calculate_throughput_hourly(self, mock_session_scope, sample_container_data):
        """Test hourly throughput calculation."""
        departed_containers = [c for c in sample_container_data if c.status == "departed"]
        
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = departed_containers
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = metrics.calculate_throughput(start_date, end_date, granularity='hourly')
        
        assert result['total_throughput'] == len(departed_containers)
        assert result['avg_throughput_per_period'] > 0
        assert result['peak_throughput'] > 0
        assert result['granularity'] == 'hourly'
        assert len(result['throughput_by_period']) > 0
    
    @patch('modules.analytics.metrics.session_scope')
    def test_calculate_throughput_daily(self, mock_session_scope, sample_container_data):
        """Test daily throughput calculation."""
        departed_containers = [c for c in sample_container_data if c.status == "departed"]
        
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = departed_containers
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        result = metrics.calculate_throughput(start_date, end_date, granularity='daily')
        
        assert result['granularity'] == 'daily'
        assert result['total_throughput'] == len(departed_containers)
    
    @patch('modules.analytics.metrics.session_scope') 
    def test_calculate_throughput_weekly(self, mock_session_scope, sample_container_data):
        """Test weekly throughput calculation."""
        departed_containers = [c for c in sample_container_data if c.status == "departed"]
        
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = departed_containers
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        result = metrics.calculate_throughput(start_date, end_date, granularity='weekly')
        
        assert result['granularity'] == 'weekly'
        assert result['total_throughput'] == len(departed_containers)
    
    @patch('modules.analytics.metrics.session_scope')
    def test_calculate_throughput_invalid_granularity(self, mock_session_scope):
        """Test throughput calculation with invalid granularity."""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        
        # Create a mock container with invalid granularity to trigger error
        mock_container = Mock()
        mock_container.last_seen = datetime.utcnow()
        mock_container.status = 'departed'
        mock_query.all.return_value = [mock_container]
        
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        with pytest.raises(ValueError, match="Invalid granularity"):
            metrics.calculate_throughput(start_date, end_date, granularity='invalid')
    
    @patch('modules.analytics.metrics.session_scope')
    def test_calculate_gate_efficiency_no_data(self, mock_session_scope):
        """Test gate efficiency calculation with no data."""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.return_value = []
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = metrics.calculate_gate_efficiency(start_date, end_date)
        
        expected = {
            'avg_processing_time': 0.0,
            'gate_utilization': 0.0,
            'efficiency_score': 0.0,
            'total_processed': 0,
            'peak_efficiency_hour': None
        }
        assert result == expected
    
    @patch('modules.analytics.metrics.session_scope')
    def test_calculate_gate_efficiency_with_data(self, mock_session_scope, sample_metrics_data):
        """Test gate efficiency calculation with metrics data."""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = sample_metrics_data[:10]
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = metrics.calculate_gate_efficiency(start_date, end_date)
        
        assert result['avg_processing_time'] > 0
        assert result['gate_utilization'] > 0
        assert result['efficiency_score'] >= 0
        assert result['total_processed'] > 0
        assert result['avg_throughput_per_hour'] >= 0
        assert 'peak_efficiency_hour' in result
        assert 'total_analysis_hours' in result
    
    @patch('modules.analytics.metrics.session_scope')
    def test_analyze_peak_hours_no_data(self, mock_session_scope):
        """Test peak hours analysis with no data."""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.return_value = []
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = metrics.analyze_peak_hours(start_date, end_date)
        
        expected = {
            'peak_hours': [],
            'hourly_averages': {},
            'busiest_day_hour': None,
            'quietest_day_hour': None
        }
        assert result == expected
    
    @patch('modules.analytics.metrics.session_scope')
    def test_analyze_peak_hours_with_data(self, mock_session_scope, sample_metrics_data):
        """Test peak hours analysis with data."""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = sample_metrics_data
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=2)
        end_date = datetime.utcnow()
        
        result = metrics.analyze_peak_hours(start_date, end_date, top_n=3)
        
        assert len(result['peak_hours']) <= 3
        assert len(result['hourly_averages']) > 0
        assert result['busiest_day_hour'] is not None
        assert result['quietest_day_hour'] is not None
        
        for peak_hour in result['peak_hours']:
            assert 'hour' in peak_hour
            assert 'avg_activity' in peak_hour
            assert 'time_range' in peak_hour
            assert 0 <= peak_hour['hour'] <= 23
    
    @patch('modules.analytics.metrics.session_scope')
    def test_get_container_type_distribution_no_data(self, mock_session_scope):
        """Test container type distribution with no data."""
        mock_session = Mock()
        mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = []
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = metrics.get_container_type_distribution(start_date, end_date)
        
        expected = {
            'total_detections': 0,
            'type_distribution': {},
            'confidence_stats': {},
            'size_distribution': {}
        }
        assert result == expected
    
    @patch('modules.analytics.metrics.session_scope')
    def test_get_container_type_distribution_with_data(self, mock_session_scope, sample_detection_data):
        """Test container type distribution with detection data."""
        container_detections = [d for d in sample_detection_data if d.object_type == "container"]
        
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value.filter.return_value.all.return_value = container_detections
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = metrics.get_container_type_distribution(start_date, end_date)
        
        assert result['total_detections'] == len(container_detections)
        assert 'type_distribution' in result
        assert 'confidence_stats' in result
        assert 'size_distribution' in result
        assert 'size_percentages' in result
        
        conf_stats = result['confidence_stats']
        assert 'avg_confidence' in conf_stats
        assert 'min_confidence' in conf_stats
        assert 'max_confidence' in conf_stats
        assert 'std_confidence' in conf_stats
        
        size_percentages = result['size_percentages']
        if size_percentages:
            total_percentage = sum(size_percentages.values())
            assert abs(total_percentage - 100.0) < 0.01


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    @patch('modules.analytics.metrics.ContainerMetrics.calculate_dwell_time')
    def test_calculate_dwell_time_function(self, mock_method):
        """Test calculate_dwell_time convenience function."""
        mock_method.return_value = {'test': 'result'}
        
        result = calculate_dwell_time(
            container_id=123,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow(),
            camera_id="camera_1"
        )
        
        assert result == {'test': 'result'}
        mock_method.assert_called_once()
    
    @patch('modules.analytics.metrics.ContainerMetrics.calculate_throughput')
    def test_calculate_throughput_function(self, mock_method):
        """Test calculate_throughput convenience function."""
        mock_method.return_value = {'throughput': 10}
        
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = calculate_throughput(start_date, end_date, camera_id="camera_1", granularity="daily")
        
        assert result == {'throughput': 10}
        mock_method.assert_called_once_with(start_date, end_date, "camera_1", "daily")
    
    @patch('modules.analytics.metrics.ContainerMetrics.calculate_gate_efficiency')
    def test_calculate_gate_efficiency_function(self, mock_method):
        """Test calculate_gate_efficiency convenience function."""
        mock_method.return_value = {'efficiency': 0.8}
        
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = calculate_gate_efficiency(start_date, end_date, camera_id="camera_1")
        
        assert result == {'efficiency': 0.8}
        mock_method.assert_called_once_with(start_date, end_date, "camera_1")
    
    @patch('modules.analytics.metrics.ContainerMetrics.analyze_peak_hours')
    def test_analyze_peak_hours_function(self, mock_method):
        """Test analyze_peak_hours convenience function."""
        mock_method.return_value = {'peak_hours': []}
        
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = analyze_peak_hours(start_date, end_date, camera_id="camera_1", top_n=5)
        
        assert result == {'peak_hours': []}
        mock_method.assert_called_once_with(start_date, end_date, "camera_1", 5)
    
    @patch('modules.analytics.metrics.ContainerMetrics.get_container_type_distribution')
    def test_get_container_type_distribution_function(self, mock_method):
        """Test get_container_type_distribution convenience function."""
        mock_method.return_value = {'types': {}}
        
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = get_container_type_distribution(start_date, end_date, camera_id="camera_1")
        
        assert result == {'types': {}}
        mock_method.assert_called_once_with(start_date, end_date, "camera_1")


# =============================================================================
# Test Aggregator Module
# =============================================================================

class TestAggregationResult:
    """Test AggregationResult dataclass."""
    
    def test_aggregation_result_creation(self):
        """Test AggregationResult creation."""
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        data = {'hour1': {'detections': 10}}
        
        result = AggregationResult(
            data=data,
            period='hourly',
            start_date=start_date,
            end_date=end_date,
            total_records=1
        )
        
        assert result.data == data
        assert result.period == 'hourly'
        assert result.start_date == start_date
        assert result.end_date == end_date
        assert result.total_records == 1


class TestDataAggregator:
    """Test DataAggregator class methods."""
    
    def test_init(self):
        """Test DataAggregator initialization."""
        aggregator = DataAggregator()
        assert aggregator.logger is not None
    
    @patch('modules.analytics.aggregator.session_scope')
    def test_aggregate_hourly_data(self, mock_session_scope, sample_image_data, sample_detection_data):
        """Test hourly data aggregation."""
        mock_session = Mock()
        
        # Mock image query
        image_query = Mock()
        image_query.filter.return_value = image_query
        image_query.all.return_value = sample_image_data
        
        # Mock detection query
        detection_query = Mock()
        detection_query.filter.return_value = detection_query
        detection_query.all.return_value = sample_detection_data[:2]
        detection_query.count.return_value = 2
        
        # Mock container query
        container_query = Mock()
        container_query.filter.return_value = container_query
        container_query.all.return_value = []
        
        def query_side_effect(model):
            from modules.database.models import Image, Detection, Container
            if model == Image:
                return image_query
            elif model == Detection:
                return detection_query
            elif model == Container:
                return container_query
            return Mock()
        
        mock_session.query.side_effect = query_side_effect
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        aggregator = DataAggregator()
        start_date = datetime.utcnow() - timedelta(hours=2)
        end_date = datetime.utcnow()
        
        result = aggregator.aggregate_hourly_data(start_date, end_date)
        
        assert isinstance(result, AggregationResult)
        assert result.period == 'hourly'
        assert result.start_date == start_date
        assert result.end_date == end_date
    
    @patch('modules.analytics.aggregator.DataAggregator.aggregate_hourly_data')
    @patch('modules.analytics.aggregator.session_scope')
    def test_aggregate_daily_data(self, mock_session_scope, mock_hourly):
        """Test daily data aggregation."""
        # Mock hourly data
        hourly_data = {
            datetime.utcnow().replace(hour=10, minute=0, second=0, microsecond=0).isoformat(): {
                'detections': 50,
                'throughput': 5,
                'containers': 10,
                'avg_confidence': 0.85,
                'total_confidence_samples': 50
            },
            datetime.utcnow().replace(hour=11, minute=0, second=0, microsecond=0).isoformat(): {
                'detections': 60,
                'throughput': 6,
                'containers': 12,
                'avg_confidence': 0.82,
                'total_confidence_samples': 60
            }
        }
        
        mock_hourly.return_value = AggregationResult(
            data=hourly_data,
            period='hourly',
            start_date=datetime.utcnow() - timedelta(days=1),
            end_date=datetime.utcnow(),
            total_records=2
        )
        
        # Mock container query
        mock_session = Mock()
        container_query = Mock()
        container_query.filter.return_value = container_query
        container_query.all.return_value = []
        mock_session.query.return_value = container_query
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        aggregator = DataAggregator()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = aggregator.aggregate_daily_data(start_date, end_date)
        
        assert isinstance(result, AggregationResult)
        assert result.period == 'daily'
        assert result.total_records > 0
    
    @patch('modules.analytics.aggregator.DataAggregator.aggregate_daily_data')
    def test_aggregate_weekly_data(self, mock_daily):
        """Test weekly data aggregation."""
        # Mock daily data
        base_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        daily_data = {}
        
        for i in range(7):
            date = base_date - timedelta(days=i)
            daily_data[date.isoformat()] = {
                'detections': 100 + i * 10,
                'throughput': 10 + i,
                'containers': 20 + i * 2,
                'avg_confidence': 0.8,
                'peak_hour': {'hour': 14, 'activity': 20},
                'total_dwell_samples': 5,
                'total_confidence_samples': 100,
                'avg_dwell_time': 2.5
            }
        
        mock_daily.return_value = AggregationResult(
            data=daily_data,
            period='daily',
            start_date=base_date - timedelta(days=7),
            end_date=base_date,
            total_records=7
        )
        
        aggregator = DataAggregator()
        start_date = base_date - timedelta(days=7)
        end_date = base_date
        
        result = aggregator.aggregate_weekly_data(start_date, end_date)
        
        assert isinstance(result, AggregationResult)
        assert result.period == 'weekly'
        assert result.total_records > 0
        
        # Check weekly data structure
        for week_data in result.data.values():
            assert 'week_start' in week_data
            assert 'week_end' in week_data
            assert 'daily_breakdown' in week_data
            assert 'peak_day' in week_data
            assert 'busiest_hour' in week_data
            assert 'avg_daily_detections' in week_data
    
    def test_calculate_rolling_averages_empty_data(self):
        """Test rolling averages with empty data."""
        aggregator = DataAggregator()
        result = aggregator.calculate_rolling_averages({}, window_size=7)
        assert result == {}
    
    def test_calculate_rolling_averages_with_data(self):
        """Test rolling averages calculation."""
        # Create sample time series data
        base_date = datetime.utcnow()
        data = {}
        for i in range(10):
            timestamp = base_date - timedelta(days=i)
            data[timestamp.isoformat()] = {
                'detections': 50 + i * 5,
                'throughput': 10 + i,
                'avg_dwell_time': 2.0 + i * 0.1
            }
        
        aggregator = DataAggregator()
        result = aggregator.calculate_rolling_averages(data, window_size=3)
        
        assert len(result) > 0
        # Check that rolling averages are calculated
        for timestamp, values in result.items():
            assert 'detections_rolling_avg' in values
            assert 'detections_rolling_std' in values
            assert 'detections_original' in values
    
    @patch('modules.analytics.aggregator.DataAggregator.aggregate_daily_data')
    @patch('modules.analytics.aggregator.session_scope')
    def test_generate_summary_statistics(self, mock_session_scope, mock_daily):
        """Test summary statistics generation."""
        # Mock database queries
        mock_session = Mock()
        
        # Create a mock query object that returns appropriate counts
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.count.return_value = 100
        mock_query.scalar.return_value = 0.85
        
        # Configure specific query returns for different chains
        def query_side_effect(*args):
            return mock_query
        
        mock_session.query = Mock(side_effect=query_side_effect)
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        # Mock daily aggregation
        daily_data = {
            datetime.utcnow().isoformat(): {
                'detections': 100,
                'throughput': 10
            },
            (datetime.utcnow() - timedelta(days=1)).isoformat(): {
                'detections': 90,
                'throughput': 8
            }
        }
        
        mock_daily.return_value = AggregationResult(
            data=daily_data,
            period='daily',
            start_date=datetime.utcnow() - timedelta(days=2),
            end_date=datetime.utcnow(),
            total_records=2
        )
        
        aggregator = DataAggregator()
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        result = aggregator.generate_summary_statistics(start_date, end_date)
        
        assert 'analysis_period' in result
        assert 'totals' in result
        assert 'averages' in result
        assert 'trends' in result
        assert 'distributions' in result
        
        # Check totals
        assert 'images' in result['totals']
        assert 'detections' in result['totals']
        assert 'containers' in result['totals']
        
        # Check averages
        assert 'confidence' in result['averages']
        assert 'dwell_time_hours' in result['averages']
        assert 'images_per_day' in result['averages']
    
    @patch('modules.analytics.aggregator.DataAggregator.aggregate_hourly_data')
    def test_aggregate_convenience_functions(self, mock_hourly):
        """Test aggregator convenience functions."""
        mock_result = AggregationResult(
            data={},
            period='hourly',
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow(),
            total_records=0
        )
        mock_hourly.return_value = mock_result
        
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow()
        
        result = aggregate_hourly_data(start_date, end_date)
        assert isinstance(result, AggregationResult)


# =============================================================================
# Test Alerts Module
# =============================================================================

class TestAlertClasses:
    """Test Alert-related classes and enums."""
    
    def test_alert_severity_enum(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"
    
    def test_alert_type_enum(self):
        """Test AlertType enum values."""
        assert AlertType.DWELL_TIME.value == "dwell_time"
        assert AlertType.THROUGHPUT.value == "throughput"
        assert AlertType.CONGESTION.value == "congestion"
        assert AlertType.SYSTEM.value == "system"
        assert AlertType.DATA_QUALITY.value == "data_quality"
    
    def test_alert_creation(self):
        """Test Alert creation and to_dict method."""
        timestamp = datetime.utcnow()
        alert = Alert(
            alert_type=AlertType.DWELL_TIME,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            timestamp=timestamp,
            camera_id="camera_1",
            value=10.5,
            threshold=8.0,
            metadata={'test': 'data'}
        )
        
        assert alert.alert_type == AlertType.DWELL_TIME
        assert alert.severity == AlertSeverity.HIGH
        assert alert.title == "Test Alert"
        
        # Test to_dict conversion
        alert_dict = alert.to_dict()
        assert alert_dict['alert_type'] == 'dwell_time'
        assert alert_dict['severity'] == 'high'
        assert alert_dict['value'] == 10.5
        assert alert_dict['metadata'] == {'test': 'data'}


class TestDwellTimeAlert:
    """Test DwellTimeAlert class."""
    
    def test_init(self):
        """Test DwellTimeAlert initialization."""
        detector = DwellTimeAlert(threshold_multiplier=2.5)
        assert detector.threshold_multiplier == 2.5
        assert detector.logger is not None
    
    @patch('modules.analytics.alerts.session_scope')
    def test_detect_anomalies_insufficient_data(self, mock_session_scope):
        """Test anomaly detection with insufficient data."""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []  # No containers
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        detector = DwellTimeAlert()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        alerts = detector.detect_anomalies(start_date, end_date)
        assert alerts == []
    
    @patch('modules.analytics.alerts.session_scope')
    def test_detect_anomalies_with_outliers(self, mock_session_scope):
        """Test anomaly detection with outlier containers."""
        # Create mock containers with varying dwell times
        containers = []
        for i in range(15):
            container = Mock()
            container.id = i
            container.container_number = f"CONT{1000 + i}"
            container.camera_id = "camera_1"
            container.last_seen = datetime.utcnow()
            container.first_seen = datetime.utcnow() - timedelta(hours=2)
            
            # Most containers have normal dwell time (2-4 hours)
            if i < 10:
                container.dwell_time = 3.0 + (i * 0.1)
            elif i == 10:
                # One with very high dwell time
                container.dwell_time = 15.0
            elif i == 11:
                # One with critical high dwell time  
                container.dwell_time = 25.0
            else:
                # Some with very low dwell time
                container.dwell_time = 0.1
            
            containers.append(container)
        
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = containers
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        detector = DwellTimeAlert(threshold_multiplier=2.0)
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        alerts = detector.detect_anomalies(start_date, end_date)
        
        assert len(alerts) > 0
        # Check that we have different severity levels
        severities = {alert.severity for alert in alerts}
        assert AlertSeverity.HIGH in severities or AlertSeverity.CRITICAL in severities


class TestThroughputAlert:
    """Test ThroughputAlert class."""
    
    def test_init(self):
        """Test ThroughputAlert initialization."""
        detector = ThroughputAlert(min_throughput_threshold=0.6, deviation_threshold=0.4)
        assert detector.min_throughput_threshold == 0.6
        assert detector.deviation_threshold == 0.4
        assert detector.logger is not None
    
    @patch('modules.analytics.alerts.ContainerMetrics')
    def test_detect_anomalies_zero_throughput(self, mock_metrics_class):
        """Test anomaly detection with zero throughput."""
        mock_metrics = Mock()
        mock_metrics.calculate_throughput.return_value = {
            'avg_throughput_per_period': 0,
            'total_throughput': 0
        }
        mock_metrics_class.return_value = mock_metrics
        
        detector = ThroughputAlert()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        alerts = detector.detect_anomalies(start_date, end_date)
        
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.THROUGHPUT
        assert alerts[0].severity == AlertSeverity.HIGH
        assert "Zero Throughput" in alerts[0].title
    
    @patch('modules.analytics.alerts.ContainerMetrics')
    def test_detect_anomalies_low_throughput(self, mock_metrics_class):
        """Test anomaly detection with low throughput."""
        mock_metrics = Mock()
        # Current period has low throughput
        mock_metrics.calculate_throughput.side_effect = [
            {'avg_throughput_per_period': 5.0, 'total_throughput': 35},  # Current
            {'avg_throughput_per_period': 20.0, 'total_throughput': 140}  # Baseline
        ]
        mock_metrics_class.return_value = mock_metrics
        
        detector = ThroughputAlert(min_throughput_threshold=0.5)
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        alerts = detector.detect_anomalies(start_date, end_date)
        
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.THROUGHPUT
        assert "Low Throughput" in alerts[0].title
    
    @patch('modules.analytics.alerts.ContainerMetrics')
    def test_detect_anomalies_significant_deviation(self, mock_metrics_class):
        """Test anomaly detection with significant throughput deviation."""
        mock_metrics = Mock()
        # Current period has higher throughput (40% increase)
        mock_metrics.calculate_throughput.side_effect = [
            {'avg_throughput_per_period': 28.0, 'total_throughput': 196},  # Current
            {'avg_throughput_per_period': 20.0, 'total_throughput': 140}  # Baseline
        ]
        mock_metrics_class.return_value = mock_metrics
        
        detector = ThroughputAlert(deviation_threshold=0.3)
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        alerts = detector.detect_anomalies(start_date, end_date)
        
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.THROUGHPUT
        assert "Deviation" in alerts[0].title


class TestCongestionAlert:
    """Test CongestionAlert class."""
    
    def test_init(self):
        """Test CongestionAlert initialization."""
        detector = CongestionAlert(congestion_threshold=25, sustained_duration=3)
        assert detector.congestion_threshold == 25
        assert detector.sustained_duration == 3
        assert detector.logger is not None
    
    @patch('modules.analytics.alerts.DataAggregator')
    def test_detect_anomalies_no_congestion(self, mock_aggregator_class):
        """Test anomaly detection with no congestion."""
        mock_aggregator = Mock()
        
        # Create hourly data with low activity
        hourly_data = {}
        base_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        for i in range(24):
            hour_time = base_time - timedelta(hours=i)
            hourly_data[hour_time.isoformat()] = {
                'detections': 5 + (i % 3)  # Low activity
            }
        
        mock_aggregator.aggregate_hourly_data.return_value = AggregationResult(
            data=hourly_data,
            period='hourly',
            start_date=base_time - timedelta(days=1),
            end_date=base_time,
            total_records=24
        )
        mock_aggregator_class.return_value = mock_aggregator
        
        detector = CongestionAlert(congestion_threshold=20)
        start_date = base_time - timedelta(days=1)
        end_date = base_time
        
        alerts = detector.detect_anomalies(start_date, end_date)
        assert alerts == []
    
    @patch('modules.analytics.alerts.DataAggregator')
    def test_detect_anomalies_with_congestion(self, mock_aggregator_class):
        """Test anomaly detection with sustained congestion."""
        mock_aggregator = Mock()
        
        # Create hourly data with congestion period
        hourly_data = {}
        base_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        
        for i in range(24):
            hour_time = base_time - timedelta(hours=23 - i)
            # Create sustained high activity from hours 10-14
            if 10 <= i <= 14:
                detections = 25 + (i * 2)  # High activity
            else:
                detections = 5 + (i % 3)  # Normal activity
            
            hourly_data[hour_time.isoformat()] = {
                'detections': detections
            }
        
        mock_aggregator.aggregate_hourly_data.return_value = AggregationResult(
            data=hourly_data,
            period='hourly',
            start_date=base_time - timedelta(days=1),
            end_date=base_time,
            total_records=24
        )
        mock_aggregator_class.return_value = mock_aggregator
        
        detector = CongestionAlert(congestion_threshold=20, sustained_duration=2)
        start_date = base_time - timedelta(days=1)
        end_date = base_time
        
        alerts = detector.detect_anomalies(start_date, end_date)
        
        assert len(alerts) > 0
        assert alerts[0].alert_type == AlertType.CONGESTION
        assert "Congestion" in alerts[0].title
        
        # Check metadata
        metadata = alerts[0].metadata
        assert 'duration_hours' in metadata
        assert metadata['duration_hours'] >= detector.sustained_duration


class TestAlertSystem:
    """Test AlertSystem class."""
    
    def test_init(self):
        """Test AlertSystem initialization."""
        email_config = {'smtp_server': 'test.com', 'from_email': 'test@test.com'}
        system = AlertSystem(email_config=email_config, enable_email=True)
        
        assert system.email_config == email_config
        assert system.enable_email is True
        assert system.dwell_time_detector is not None
        assert system.throughput_detector is not None
        assert system.congestion_detector is not None
    
    @patch('modules.analytics.alerts.DwellTimeAlert.detect_anomalies')
    @patch('modules.analytics.alerts.ThroughputAlert.detect_anomalies')
    @patch('modules.analytics.alerts.CongestionAlert.detect_anomalies')
    def test_detect_all_anomalies(self, mock_congestion, mock_throughput, mock_dwell):
        """Test detection of all anomaly types."""
        # Create sample alerts
        dwell_alert = Alert(
            alert_type=AlertType.DWELL_TIME,
            severity=AlertSeverity.HIGH,
            title="Dwell Alert",
            message="Test",
            timestamp=datetime.utcnow()
        )
        
        throughput_alert = Alert(
            alert_type=AlertType.THROUGHPUT,
            severity=AlertSeverity.MEDIUM,
            title="Throughput Alert",
            message="Test",
            timestamp=datetime.utcnow()
        )
        
        congestion_alert = Alert(
            alert_type=AlertType.CONGESTION,
            severity=AlertSeverity.LOW,
            title="Congestion Alert",
            message="Test",
            timestamp=datetime.utcnow()
        )
        
        mock_dwell.return_value = [dwell_alert]
        mock_throughput.return_value = [throughput_alert]
        mock_congestion.return_value = [congestion_alert]
        
        system = AlertSystem()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        alerts = system.detect_all_anomalies(start_date, end_date)
        
        assert len(alerts['dwell_time']) == 1
        assert len(alerts['throughput']) == 1
        assert len(alerts['congestion']) == 1
        assert len(alerts['system']) == 0
    
    @patch('modules.analytics.alerts.DwellTimeAlert.detect_anomalies')
    def test_detect_all_anomalies_with_error(self, mock_dwell):
        """Test anomaly detection with error handling."""
        mock_dwell.side_effect = Exception("Database error")
        
        system = AlertSystem()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        alerts = system.detect_all_anomalies(start_date, end_date)
        
        assert len(alerts['system']) == 1
        assert alerts['system'][0].alert_type == AlertType.SYSTEM
        assert alerts['system'][0].severity == AlertSeverity.HIGH
        assert "error" in alerts['system'][0].metadata
    
    def test_send_alert_notification_no_email(self):
        """Test sending alert without email configuration."""
        system = AlertSystem(enable_email=False)
        
        alert = Alert(
            alert_type=AlertType.DWELL_TIME,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            timestamp=datetime.utcnow()
        )
        
        result = system.send_alert_notification(alert)
        assert result is True
    
    @patch('smtplib.SMTP')
    def test_send_alert_notification_with_email(self, mock_smtp):
        """Test sending alert with email configuration."""
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        email_config = {
            'smtp_server': 'smtp.test.com',
            'smtp_port': 587,
            'from_email': 'alerts@test.com',
            'to_email': 'admin@test.com',
            'username': 'user',
            'password': 'pass'
        }
        
        system = AlertSystem(email_config=email_config, enable_email=True)
        
        alert = Alert(
            alert_type=AlertType.DWELL_TIME,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            timestamp=datetime.utcnow(),
            value=10.5,
            threshold=8.0
        )
        
        result = system.send_alert_notification(alert)
        
        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with('user', 'pass')
        mock_server.sendmail.assert_called_once()
    
    def test_get_alert_summary(self):
        """Test alert summary generation."""
        # Create sample alerts
        alerts = {
            'dwell_time': [
                Alert(AlertType.DWELL_TIME, AlertSeverity.HIGH, "Alert1", "Msg1", datetime.utcnow()),
                Alert(AlertType.DWELL_TIME, AlertSeverity.CRITICAL, "Alert2", "Msg2", datetime.utcnow())
            ],
            'throughput': [
                Alert(AlertType.THROUGHPUT, AlertSeverity.MEDIUM, "Alert3", "Msg3", datetime.utcnow())
            ],
            'congestion': [],
            'system': []
        }
        
        system = AlertSystem()
        summary = system.get_alert_summary(alerts)
        
        assert summary['total_alerts'] == 3
        assert summary['severity_breakdown']['high'] == 1
        assert summary['severity_breakdown']['critical'] == 1
        assert summary['severity_breakdown']['medium'] == 1
        assert summary['type_breakdown']['dwell_time'] == 2
        assert summary['type_breakdown']['throughput'] == 1
        assert summary['has_critical_alerts'] is True
        assert summary['has_high_priority_alerts'] is True
        assert len(summary['recent_alerts']) == 3
    
    def test_detect_anomalies_convenience_function(self):
        """Test detect_anomalies convenience function."""
        with patch('modules.analytics.alerts.AlertSystem.detect_all_anomalies') as mock_method:
            mock_method.return_value = {'test': []}
            
            start_date = datetime.utcnow() - timedelta(days=1)
            end_date = datetime.utcnow()
            
            result = detect_anomalies(start_date, end_date)
            assert result == {'test': []}
    
    def test_send_alert_notification_convenience_function(self):
        """Test send_alert_notification convenience function."""
        with patch('modules.analytics.alerts.AlertSystem.send_alert_notification') as mock_method:
            mock_method.return_value = True
            
            alert = Alert(
                alert_type=AlertType.DWELL_TIME,
                severity=AlertSeverity.HIGH,
                title="Test",
                message="Test",
                timestamp=datetime.utcnow()
            )
            
            result = send_alert_notification(alert)
            assert result is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestAnalyticsIntegration:
    """Integration tests using real database session."""
    
    @pytest.mark.database
    def test_full_analytics_workflow(self, test_db_session, sample_container_data, 
                                   sample_metrics_data, sample_detection_data):
        """Test full analytics workflow with real database session."""
        with patch('modules.analytics.metrics.session_scope') as mock_scope:
            mock_scope.return_value.__enter__.return_value = test_db_session
            
            metrics = ContainerMetrics()
            start_date = datetime.utcnow() - timedelta(days=1)
            end_date = datetime.utcnow()
            
            # Test dwell time calculation
            dwell_result = metrics.calculate_dwell_time(start_date=start_date, end_date=end_date)
            assert isinstance(dwell_result, dict)
            
            # Test throughput calculation
            throughput_result = metrics.calculate_throughput(start_date, end_date)
            assert isinstance(throughput_result, dict)
            
            # Test efficiency calculation
            efficiency_result = metrics.calculate_gate_efficiency(start_date, end_date)
            assert isinstance(efficiency_result, dict)
    
    @pytest.mark.database
    def test_analytics_with_empty_database(self, test_db_session):
        """Test analytics functions with empty database."""
        with patch('modules.analytics.metrics.session_scope') as mock_scope:
            mock_scope.return_value.__enter__.return_value = test_db_session
            
            metrics = ContainerMetrics()
            start_date = datetime.utcnow() - timedelta(days=1)
            end_date = datetime.utcnow()
            
            # All functions should handle empty database gracefully
            dwell_result = metrics.calculate_dwell_time()
            assert dwell_result['total_containers'] == 0
            
            throughput_result = metrics.calculate_throughput(start_date, end_date)
            assert throughput_result['total_throughput'] == 0
            
            efficiency_result = metrics.calculate_gate_efficiency(start_date, end_date)
            assert efficiency_result['total_processed'] == 0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestAnalyticsErrorHandling:
    """Test error handling in analytics functions."""
    
    @patch('modules.analytics.metrics.session_scope')
    def test_database_connection_error(self, mock_session_scope):
        """Test handling of database connection errors."""
        mock_session_scope.side_effect = Exception("Database connection failed")
        
        metrics = ContainerMetrics()
        
        with pytest.raises(Exception, match="Database connection failed"):
            metrics.calculate_dwell_time()
    
    @patch('modules.analytics.metrics.session_scope')
    def test_invalid_date_range(self, mock_session_scope):
        """Test handling of invalid date ranges."""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.return_value = []
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        
        # Test with end_date before start_date
        start_date = datetime.utcnow()
        end_date = datetime.utcnow() - timedelta(days=1)
        
        # Should handle gracefully and return empty results
        result = metrics.calculate_throughput(start_date, end_date)
        assert result['total_throughput'] == 0
    
    def test_pandas_operations_with_empty_data(self):
        """Test pandas operations with empty data."""
        metrics = ContainerMetrics()
        
        # Test with empty list
        with patch('modules.analytics.metrics.session_scope') as mock_scope:
            mock_session = Mock()
            mock_session.query.return_value.all.return_value = []
            mock_scope.return_value.__enter__.return_value = mock_session
            
            result = metrics.calculate_dwell_time()
            assert result['dwell_times'] == []
    
    @patch('modules.analytics.metrics.session_scope')
    def test_malformed_data_handling(self, mock_session_scope):
        """Test handling of malformed data."""
        # Create mock containers with None values
        mock_container = Mock()
        mock_container.dwell_time = None
        mock_container.calculate_dwell_time.return_value = None
        
        mock_session = Mock()
        mock_session.query.return_value.all.return_value = [mock_container]
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        result = metrics.calculate_dwell_time()
        
        # Should handle None values gracefully
        assert result['total_containers'] == 1
        assert result['dwell_times'] == []


# =============================================================================
# Performance Tests
# =============================================================================

class TestAnalyticsPerformance:
    """Performance tests for analytics calculations."""
    
    @pytest.mark.slow
    @patch('modules.analytics.metrics.session_scope')
    def test_large_dataset_performance(self, mock_session_scope):
        """Test analytics performance with large dataset."""
        import time
        
        # Create large mock dataset
        large_container_list = []
        for i in range(1000):
            container = Mock()
            container.dwell_time = float(i % 24 + 1)
            large_container_list.append(container)
        
        mock_session = Mock()
        mock_session.query.return_value.all.return_value = large_container_list
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        
        start_time = time.time()
        result = metrics.calculate_dwell_time()
        end_time = time.time()
        
        # Should process large dataset reasonably fast
        assert end_time - start_time < 2.0
        assert result['total_containers'] == 1000
        assert len(result['dwell_times']) == 1000
    
    @pytest.mark.slow
    @patch('modules.analytics.metrics.session_scope')
    def test_complex_aggregation_performance(self, mock_session_scope, sample_metrics_data):
        """Test performance of complex aggregations."""
        import time
        
        # Use a large sample of metrics data
        large_metrics_data = sample_metrics_data * 10
        
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = large_metrics_data
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        metrics = ContainerMetrics()
        start_date = datetime.utcnow() - timedelta(days=10)
        end_date = datetime.utcnow()
        
        start_time = time.time()
        result = metrics.analyze_peak_hours(start_date, end_date)
        end_time = time.time()
        
        # Should handle complex aggregation reasonably fast
        assert end_time - start_time < 1.0
        assert len(result['peak_hours']) > 0