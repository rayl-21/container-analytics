"""
Container Analytics Database Module

This module provides database models and utilities for the Container Analytics application.
Includes SQLAlchemy models for images, detections, containers, and metrics data.
"""

from .models import (
    Base,
    Image,
    Detection,
    Container,
    Metric,
    init_database,
    get_session,
    create_tables,
    get_engine
)

from .queries import (
    insert_image,
    insert_detection,
    update_container_tracking,
    get_metrics_by_date_range,
    get_unprocessed_images,
    get_container_statistics,
    get_throughput_data,
    get_dwell_time_data,
    aggregate_hourly_metrics,
    cleanup_old_data
)

__all__ = [
    # Models
    'Base',
    'Image',
    'Detection', 
    'Container',
    'Metric',
    'init_database',
    'get_session',
    'create_tables',
    'get_engine',
    
    # Queries
    'insert_image',
    'insert_detection',
    'update_container_tracking',
    'get_metrics_by_date_range',
    'get_unprocessed_images',
    'get_container_statistics',
    'get_throughput_data',
    'get_dwell_time_data',
    'aggregate_hourly_metrics',
    'cleanup_old_data'
]