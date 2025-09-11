"""
Tracking Analytics Module for Container Analytics

This module provides analytics calculations specifically designed for container
tracking, including dwell time analysis, throughput metrics, movement patterns,
and container lifecycle analytics.

Features:
- Container dwell time analysis
- Throughput and capacity metrics
- Movement pattern analysis
- Peak traffic identification
- Container lifecycle tracking
- Real-time analytics dashboard support
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..database.models import Container, ContainerMovement, Detection, Image, session_scope
from ..database.queries import DatabaseQueries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DwellTimeMetrics:
    """Container dwell time analytics."""
    total_containers: int
    active_containers: int
    departed_containers: int
    avg_dwell_time_hours: float
    median_dwell_time_hours: float
    min_dwell_time_hours: float
    max_dwell_time_hours: float
    std_dwell_time_hours: float
    percentile_95_hours: float
    time_window: str


@dataclass
class ThroughputMetrics:
    """Container throughput analytics."""
    time_period: str
    total_entries: int
    total_exits: int
    net_containers: int
    entries_per_hour: float
    exits_per_hour: float
    peak_entry_hour: Optional[int]
    peak_exit_hour: Optional[int]
    peak_entry_count: int
    peak_exit_count: int


@dataclass
class MovementMetrics:
    """Container movement analytics."""
    total_movements: int
    unique_containers_moved: int
    avg_movements_per_container: float
    most_active_camera: str
    camera_activity: Dict[str, int]
    movement_types: Dict[str, int]
    avg_movement_duration_minutes: float


@dataclass
class CapacityMetrics:
    """Container yard capacity analytics."""
    current_occupancy: int
    max_observed_occupancy: int
    avg_occupancy_24h: float
    occupancy_trend: str  # 'increasing', 'decreasing', 'stable'
    capacity_utilization: float  # If max capacity is known


class ContainerTrackingAnalytics:
    """
    Advanced analytics engine for container tracking data.
    
    Provides comprehensive analytics for container movements, dwell times,
    throughput metrics, and operational insights.
    """
    
    def __init__(self, max_capacity: Optional[int] = None):
        """
        Initialize the analytics engine.
        
        Args:
            max_capacity: Maximum container capacity for utilization calculations
        """
        self.max_capacity = max_capacity
        self.db_queries = DatabaseQueries()
        
    def get_dwell_time_analytics(
        self,
        time_window_hours: int = 24,
        camera_id: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> DwellTimeMetrics:
        """
        Calculate comprehensive dwell time analytics.
        
        Args:
            time_window_hours: Time window for analysis
            camera_id: Optional camera filter
            status_filter: Optional status filter ('active', 'departed')
            
        Returns:
            DwellTimeMetrics object with analytics
        """
        try:
            with session_scope() as session:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=time_window_hours)
                
                # Build query
                query = session.query(Container).filter(
                    Container.first_seen >= start_time
                )
                
                if camera_id:
                    query = query.filter(
                        or_(
                            Container.entry_camera_id == camera_id,
                            Container.current_camera_id == camera_id
                        )
                    )
                
                if status_filter:
                    query = query.filter(Container.status == status_filter)
                
                containers = query.all()
                
                if not containers:
                    return DwellTimeMetrics(
                        total_containers=0,
                        active_containers=0,
                        departed_containers=0,
                        avg_dwell_time_hours=0.0,
                        median_dwell_time_hours=0.0,
                        min_dwell_time_hours=0.0,
                        max_dwell_time_hours=0.0,
                        std_dwell_time_hours=0.0,
                        percentile_95_hours=0.0,
                        time_window=f"{time_window_hours}h"
                    )
                
                # Calculate dwell times (ensure they're calculated)
                dwell_times = []
                active_count = 0
                departed_count = 0
                
                for container in containers:
                    if container.dwell_time is None:
                        container.calculate_dwell_time()
                    
                    if container.dwell_time is not None:
                        dwell_times.append(container.dwell_time)
                    
                    if container.status == 'active':
                        active_count += 1
                    elif container.status == 'departed':
                        departed_count += 1
                
                # Calculate statistics
                if dwell_times:
                    dwell_array = np.array(dwell_times)
                    metrics = DwellTimeMetrics(
                        total_containers=len(containers),
                        active_containers=active_count,
                        departed_containers=departed_count,
                        avg_dwell_time_hours=float(np.mean(dwell_array)),
                        median_dwell_time_hours=float(np.median(dwell_array)),
                        min_dwell_time_hours=float(np.min(dwell_array)),
                        max_dwell_time_hours=float(np.max(dwell_array)),
                        std_dwell_time_hours=float(np.std(dwell_array)),
                        percentile_95_hours=float(np.percentile(dwell_array, 95)),
                        time_window=f"{time_window_hours}h"
                    )
                else:
                    metrics = DwellTimeMetrics(
                        total_containers=len(containers),
                        active_containers=active_count,
                        departed_containers=departed_count,
                        avg_dwell_time_hours=0.0,
                        median_dwell_time_hours=0.0,
                        min_dwell_time_hours=0.0,
                        max_dwell_time_hours=0.0,
                        std_dwell_time_hours=0.0,
                        percentile_95_hours=0.0,
                        time_window=f"{time_window_hours}h"
                    )
                
                logger.info(f"Calculated dwell time analytics for {len(containers)} containers")
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating dwell time analytics: {e}")
            raise
    
    def get_throughput_analytics(
        self,
        time_window_hours: int = 24,
        camera_id: Optional[str] = None
    ) -> ThroughputMetrics:
        """
        Calculate container throughput analytics.
        
        Args:
            time_window_hours: Time window for analysis
            camera_id: Optional camera filter
            
        Returns:
            ThroughputMetrics object with analytics
        """
        try:
            with session_scope() as session:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=time_window_hours)
                
                # Query movements in time window
                query = session.query(ContainerMovement).filter(
                    ContainerMovement.timestamp >= start_time
                )
                
                if camera_id:
                    query = query.filter(ContainerMovement.to_camera_id == camera_id)
                
                movements = query.all()
                
                # Categorize movements
                entries = [m for m in movements if m.movement_type == 'entry']
                exits = [m for m in movements if m.movement_type == 'exit']
                
                # Calculate hourly distributions
                entry_hours = defaultdict(int)
                exit_hours = defaultdict(int)
                
                for entry in entries:
                    hour = entry.timestamp.hour
                    entry_hours[hour] += 1
                
                for exit in exits:
                    hour = exit.timestamp.hour
                    exit_hours[hour] += 1
                
                # Find peak hours
                peak_entry_hour = max(entry_hours.keys(), key=lambda h: entry_hours[h]) if entry_hours else None
                peak_exit_hour = max(exit_hours.keys(), key=lambda h: exit_hours[h]) if exit_hours else None
                
                metrics = ThroughputMetrics(
                    time_period=f"{time_window_hours}h",
                    total_entries=len(entries),
                    total_exits=len(exits),
                    net_containers=len(entries) - len(exits),
                    entries_per_hour=len(entries) / time_window_hours,
                    exits_per_hour=len(exits) / time_window_hours,
                    peak_entry_hour=peak_entry_hour,
                    peak_exit_hour=peak_exit_hour,
                    peak_entry_count=entry_hours[peak_entry_hour] if peak_entry_hour else 0,
                    peak_exit_count=exit_hours[peak_exit_hour] if peak_exit_hour else 0
                )
                
                logger.info(f"Calculated throughput analytics: {len(entries)} entries, {len(exits)} exits")
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating throughput analytics: {e}")
            raise
    
    def get_movement_analytics(
        self,
        time_window_hours: int = 24
    ) -> MovementMetrics:
        """
        Calculate container movement pattern analytics.
        
        Args:
            time_window_hours: Time window for analysis
            
        Returns:
            MovementMetrics object with analytics
        """
        try:
            with session_scope() as session:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=time_window_hours)
                
                # Query movements
                movements = session.query(ContainerMovement).filter(
                    ContainerMovement.timestamp >= start_time
                ).all()
                
                if not movements:
                    return MovementMetrics(
                        total_movements=0,
                        unique_containers_moved=0,
                        avg_movements_per_container=0.0,
                        most_active_camera="",
                        camera_activity={},
                        movement_types={},
                        avg_movement_duration_minutes=0.0
                    )
                
                # Analyze movements
                unique_containers = set(m.container_id for m in movements)
                camera_activity = defaultdict(int)
                movement_types = defaultdict(int)
                durations = []
                
                for movement in movements:
                    camera_activity[movement.to_camera_id] += 1
                    movement_types[movement.movement_type] += 1
                    
                    if movement.duration_from_last:
                        durations.append(movement.duration_from_last)
                
                most_active_camera = max(camera_activity.keys(), 
                                       key=lambda c: camera_activity[c]) if camera_activity else ""
                
                avg_duration = np.mean(durations) if durations else 0.0
                
                metrics = MovementMetrics(
                    total_movements=len(movements),
                    unique_containers_moved=len(unique_containers),
                    avg_movements_per_container=len(movements) / len(unique_containers) if unique_containers else 0.0,
                    most_active_camera=most_active_camera,
                    camera_activity=dict(camera_activity),
                    movement_types=dict(movement_types),
                    avg_movement_duration_minutes=float(avg_duration)
                )
                
                logger.info(f"Calculated movement analytics for {len(movements)} movements")
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating movement analytics: {e}")
            raise
    
    def get_capacity_analytics(self) -> CapacityMetrics:
        """
        Calculate container yard capacity analytics.
        
        Returns:
            CapacityMetrics object with capacity analysis
        """
        try:
            with session_scope() as session:
                # Current occupancy (active containers)
                current_occupancy = session.query(Container).filter(
                    Container.status == 'active'
                ).count()
                
                # Historical occupancy analysis (last 24 hours)
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)
                
                # Calculate hourly occupancy changes
                entries = session.query(ContainerMovement).filter(
                    ContainerMovement.movement_type == 'entry',
                    ContainerMovement.timestamp >= start_time
                ).all()
                
                exits = session.query(ContainerMovement).filter(
                    ContainerMovement.movement_type == 'exit',
                    ContainerMovement.timestamp >= start_time
                ).all()
                
                # Simulate hourly occupancy
                hourly_occupancy = []
                base_occupancy = current_occupancy
                
                for hour in range(24):
                    hour_start = start_time + timedelta(hours=hour)
                    hour_end = hour_start + timedelta(hours=1)
                    
                    hour_entries = len([e for e in entries if hour_start <= e.timestamp < hour_end])
                    hour_exits = len([e for e in exits if hour_start <= e.timestamp < hour_end])
                    
                    # Approximate occupancy (this is simplified)
                    base_occupancy += hour_entries - hour_exits
                    hourly_occupancy.append(base_occupancy)
                
                max_occupancy = max(hourly_occupancy) if hourly_occupancy else current_occupancy
                avg_occupancy = np.mean(hourly_occupancy) if hourly_occupancy else current_occupancy
                
                # Determine trend (last 6 hours vs previous 6 hours)
                if len(hourly_occupancy) >= 12:
                    recent_avg = np.mean(hourly_occupancy[-6:])
                    previous_avg = np.mean(hourly_occupancy[-12:-6])
                    
                    if recent_avg > previous_avg * 1.1:
                        trend = "increasing"
                    elif recent_avg < previous_avg * 0.9:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                else:
                    trend = "stable"
                
                # Calculate utilization if max capacity is known
                utilization = (current_occupancy / self.max_capacity * 100) if self.max_capacity else 0.0
                
                metrics = CapacityMetrics(
                    current_occupancy=current_occupancy,
                    max_observed_occupancy=max_occupancy,
                    avg_occupancy_24h=float(avg_occupancy),
                    occupancy_trend=trend,
                    capacity_utilization=utilization
                )
                
                logger.info(f"Calculated capacity analytics: {current_occupancy} current, {max_occupancy} max")
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating capacity analytics: {e}")
            raise
    
    def get_container_lifecycle_summary(
        self,
        container_number: str
    ) -> Dict:
        """
        Get detailed lifecycle summary for a specific container.
        
        Args:
            container_number: Container number to analyze
            
        Returns:
            Dictionary with container lifecycle information
        """
        try:
            with session_scope() as session:
                container = session.query(Container).filter(
                    Container.container_number == container_number
                ).first()
                
                if not container:
                    return {"error": f"Container {container_number} not found"}
                
                # Get all movements
                movements = session.query(ContainerMovement).filter(
                    ContainerMovement.container_id == container.id
                ).order_by(ContainerMovement.timestamp).all()
                
                # Build movement timeline
                timeline = []
                for movement in movements:
                    timeline.append({
                        "timestamp": movement.timestamp,
                        "type": movement.movement_type,
                        "from_camera": movement.from_camera_id,
                        "to_camera": movement.to_camera_id,
                        "duration_from_last": movement.duration_from_last
                    })
                
                # Calculate summary statistics
                camera_visits = {}
                for movement in movements:
                    camera = movement.to_camera_id
                    if camera not in camera_visits:
                        camera_visits[camera] = []
                    camera_visits[camera].append(movement.timestamp)
                
                summary = {
                    "container_number": container_number,
                    "status": container.status,
                    "first_seen": container.first_seen,
                    "last_seen": container.last_seen,
                    "dwell_time_hours": container.dwell_time,
                    "total_detections": container.total_detections,
                    "entry_camera": container.entry_camera_id,
                    "exit_camera": container.exit_camera_id,
                    "current_camera": container.current_camera_id,
                    "total_movements": len(movements),
                    "cameras_visited": list(camera_visits.keys()),
                    "movement_timeline": timeline,
                    "track_ids": container.get_track_ids_list(),
                    "ocr_confidence": container.ocr_confidence
                }
                
                logger.info(f"Generated lifecycle summary for container {container_number}")
                return summary
                
        except Exception as e:
            logger.error(f"Error generating container lifecycle summary: {e}")
            return {"error": str(e)}
    
    def get_real_time_dashboard_data(self) -> Dict:
        """
        Get consolidated data for real-time dashboard display.
        
        Returns:
            Dictionary with all key metrics for dashboard
        """
        try:
            # Get analytics for different time windows
            dwell_24h = self.get_dwell_time_analytics(24)
            throughput_24h = self.get_throughput_analytics(24)
            throughput_1h = self.get_throughput_analytics(1)
            movement_24h = self.get_movement_analytics(24)
            capacity = self.get_capacity_analytics()
            
            dashboard_data = {
                "timestamp": datetime.now(),
                "dwell_time_24h": {
                    "avg_hours": dwell_24h.avg_dwell_time_hours,
                    "median_hours": dwell_24h.median_dwell_time_hours,
                    "total_containers": dwell_24h.total_containers,
                    "active_containers": dwell_24h.active_containers
                },
                "throughput_24h": {
                    "entries": throughput_24h.total_entries,
                    "exits": throughput_24h.total_exits,
                    "net_containers": throughput_24h.net_containers,
                    "entries_per_hour": throughput_24h.entries_per_hour,
                    "peak_entry_hour": throughput_24h.peak_entry_hour
                },
                "throughput_1h": {
                    "entries": throughput_1h.total_entries,
                    "exits": throughput_1h.total_exits,
                    "net_containers": throughput_1h.net_containers
                },
                "movements_24h": {
                    "total_movements": movement_24h.total_movements,
                    "unique_containers": movement_24h.unique_containers_moved,
                    "most_active_camera": movement_24h.most_active_camera,
                    "camera_activity": movement_24h.camera_activity
                },
                "capacity": {
                    "current_occupancy": capacity.current_occupancy,
                    "max_observed": capacity.max_observed_occupancy,
                    "trend": capacity.occupancy_trend,
                    "utilization_percent": capacity.capacity_utilization
                }
            }
            
            logger.info("Generated real-time dashboard data")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {"error": str(e)}
    
    def export_analytics_report(
        self,
        time_window_hours: int = 24,
        include_container_details: bool = False
    ) -> Dict:
        """
        Export comprehensive analytics report.
        
        Args:
            time_window_hours: Time window for analysis
            include_container_details: Whether to include individual container details
            
        Returns:
            Dictionary with complete analytics report
        """
        try:
            report = {
                "report_generated": datetime.now(),
                "time_window_hours": time_window_hours,
                "dwell_time_analytics": self.get_dwell_time_analytics(time_window_hours),
                "throughput_analytics": self.get_throughput_analytics(time_window_hours),
                "movement_analytics": self.get_movement_analytics(time_window_hours),
                "capacity_analytics": self.get_capacity_analytics()
            }
            
            if include_container_details:
                with session_scope() as session:
                    end_time = datetime.now()
                    start_time = end_time - timedelta(hours=time_window_hours)
                    
                    containers = session.query(Container).filter(
                        Container.first_seen >= start_time
                    ).all()
                    
                    container_details = []
                    for container in containers:
                        container_details.append({
                            "container_number": container.container_number,
                            "status": container.status,
                            "dwell_time_hours": container.dwell_time,
                            "entry_camera": container.entry_camera_id,
                            "current_camera": container.current_camera_id,
                            "total_movements": len(container.movements)
                        })
                    
                    report["container_details"] = container_details
            
            logger.info(f"Generated analytics report for {time_window_hours}h window")
            return report
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            return {"error": str(e)}


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Container Tracking Analytics")
    parser.add_argument("--hours", type=int, default=24, help="Time window in hours")
    parser.add_argument("--container", type=str, help="Specific container number to analyze")
    parser.add_argument("--camera", type=str, help="Filter by camera ID")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--dashboard", action="store_true", help="Get dashboard data")
    
    args = parser.parse_args()
    
    analytics = ContainerTrackingAnalytics(max_capacity=1000)  # Example capacity
    
    try:
        if args.container:
            # Analyze specific container
            result = analytics.get_container_lifecycle_summary(args.container)
            print(f"Container {args.container} Lifecycle:")
            print(json.dumps(result, indent=2, default=str))
            
        elif args.dashboard:
            # Get dashboard data
            result = analytics.get_real_time_dashboard_data()
            print("Real-time Dashboard Data:")
            print(json.dumps(result, indent=2, default=str))
            
        elif args.report:
            # Generate full report
            result = analytics.export_analytics_report(
                time_window_hours=args.hours,
                include_container_details=True
            )
            print(f"Analytics Report ({args.hours}h):")
            print(json.dumps(result, indent=2, default=str))
            
        else:
            # Show summary analytics
            print(f"Container Analytics Summary ({args.hours}h):")
            
            dwell = analytics.get_dwell_time_analytics(args.hours, args.camera)
            print(f"\nDwell Time Analytics:")
            print(f"  Total containers: {dwell.total_containers}")
            print(f"  Active containers: {dwell.active_containers}")
            print(f"  Average dwell time: {dwell.avg_dwell_time_hours:.2f} hours")
            print(f"  Median dwell time: {dwell.median_dwell_time_hours:.2f} hours")
            
            throughput = analytics.get_throughput_analytics(args.hours, args.camera)
            print(f"\nThroughput Analytics:")
            print(f"  Entries: {throughput.total_entries}")
            print(f"  Exits: {throughput.total_exits}")
            print(f"  Net containers: {throughput.net_containers}")
            print(f"  Entries per hour: {throughput.entries_per_hour:.1f}")
            
            movements = analytics.get_movement_analytics(args.hours)
            print(f"\nMovement Analytics:")
            print(f"  Total movements: {movements.total_movements}")
            print(f"  Unique containers moved: {movements.unique_containers_moved}")
            print(f"  Most active camera: {movements.most_active_camera}")
            
            capacity = analytics.get_capacity_analytics()
            print(f"\nCapacity Analytics:")
            print(f"  Current occupancy: {capacity.current_occupancy}")
            print(f"  Max observed: {capacity.max_observed_occupancy}")
            print(f"  Trend: {capacity.occupancy_trend}")
    
    except Exception as e:
        print(f"Error: {e}")