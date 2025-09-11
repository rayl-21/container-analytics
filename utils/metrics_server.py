#!/usr/bin/env python3
"""
Prometheus metrics server for Container Analytics.
Provides monitoring metrics for all services.
"""

import time
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Dict, Any

from prometheus_client import (
    start_http_server, 
    Gauge, 
    Counter, 
    Histogram,
    Info
)

# Define metrics
IMAGES_DOWNLOADED_TOTAL = Counter(
    'container_analytics_images_downloaded_total',
    'Total number of images downloaded',
    ['stream_name']
)

IMAGES_PROCESSED_TOTAL = Counter(
    'container_analytics_images_processed_total', 
    'Total number of images processed by detector',
    ['status']
)

DETECTIONS_TOTAL = Counter(
    'container_analytics_detections_total',
    'Total number of container detections',
    ['stream_name']
)

PROCESSING_DURATION = Histogram(
    'container_analytics_processing_duration_seconds',
    'Time spent processing images',
    ['operation']
)

QUEUE_SIZE = Gauge(
    'container_analytics_queue_size',
    'Number of images waiting to be processed'
)

DATABASE_SIZE = Gauge(
    'container_analytics_database_size_bytes',
    'Size of the SQLite database in bytes'
)

DISK_USAGE = Gauge(
    'container_analytics_disk_usage_bytes',
    'Disk usage in data directory',
    ['directory']
)

SERVICE_STATUS = Gauge(
    'container_analytics_service_status',
    'Service health status (1=healthy, 0=unhealthy)',
    ['service_name']
)

LAST_DOWNLOAD_TIME = Gauge(
    'container_analytics_last_download_timestamp',
    'Timestamp of last successful image download',
    ['stream_name']
)

DETECTION_ACCURACY = Gauge(
    'container_analytics_detection_accuracy',
    'Detection accuracy percentage',
    ['model_version']
)

APPLICATION_INFO = Info(
    'container_analytics_application',
    'Application information'
)


class MetricsCollector:
    """Collects and exposes metrics for Container Analytics."""
    
    def __init__(self, data_dir: Path = Path("/data")):
        self.data_dir = data_dir
        self.db_path = data_dir / "database.db"
        self.running = True
        
        # Set application info
        APPLICATION_INFO.info({
            'version': '1.0.0',
            'build_date': datetime.now().isoformat(),
            'python_version': '3.10'
        })
    
    def collect_database_metrics(self):
        """Collect metrics from the database."""
        try:
            if not self.db_path.exists():
                return
                
            # Database size
            DATABASE_SIZE.set(self.db_path.stat().st_size)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Count images by stream
            cursor.execute("""
                SELECT stream_name, COUNT(*) 
                FROM images 
                GROUP BY stream_name
            """)
            
            for stream_name, count in cursor.fetchall():
                IMAGES_DOWNLOADED_TOTAL.labels(stream_name=stream_name)._value.set(count)
            
            # Count detections by stream
            cursor.execute("""
                SELECT i.stream_name, COUNT(d.id)
                FROM images i
                LEFT JOIN detections d ON i.id = d.image_id
                GROUP BY i.stream_name
            """)
            
            for stream_name, count in cursor.fetchall():
                if count:
                    DETECTIONS_TOTAL.labels(stream_name=stream_name)._value.set(count)
            
            # Last download times
            cursor.execute("""
                SELECT stream_name, MAX(created_at) as last_download
                FROM images
                GROUP BY stream_name
            """)
            
            for stream_name, last_download in cursor.fetchall():
                if last_download:
                    timestamp = datetime.fromisoformat(last_download).timestamp()
                    LAST_DOWNLOAD_TIME.labels(stream_name=stream_name).set(timestamp)
            
            # Queue size (images without detections)
            cursor.execute("""
                SELECT COUNT(*)
                FROM images i
                LEFT JOIN detections d ON i.id = d.image_id
                WHERE d.id IS NULL
                AND i.created_at > datetime('now', '-1 hour')
            """)
            
            queue_size = cursor.fetchone()[0]
            QUEUE_SIZE.set(queue_size)
            
            conn.close()
            
        except Exception as e:
            print(f"Error collecting database metrics: {e}")
    
    def collect_filesystem_metrics(self):
        """Collect filesystem-related metrics."""
        try:
            # Disk usage for different directories
            directories = ["images", "models", "logs"]
            
            for dir_name in directories:
                dir_path = self.data_dir / dir_name
                if dir_path.exists():
                    total_size = sum(
                        f.stat().st_size 
                        for f in dir_path.rglob('*') 
                        if f.is_file()
                    )
                    DISK_USAGE.labels(directory=dir_name).set(total_size)
                    
        except Exception as e:
            print(f"Error collecting filesystem metrics: {e}")
    
    def collect_service_health_metrics(self):
        """Collect service health status metrics."""
        try:
            health_file = self.data_dir / ".health_status"
            if health_file.exists():
                health_data = json.loads(health_file.read_text())
                
                # Overall service health
                overall_healthy = health_data.get("overall_healthy", False)
                SERVICE_STATUS.labels(service_name="overall").set(1 if overall_healthy else 0)
                
                # Individual service health
                checks = health_data.get("checks", {})
                for service_name, check_result in checks.items():
                    healthy = check_result.get("healthy", False)
                    SERVICE_STATUS.labels(service_name=service_name).set(1 if healthy else 0)
                    
        except Exception as e:
            print(f"Error collecting service health metrics: {e}")
    
    def collect_scheduler_metrics(self):
        """Collect scheduler-specific metrics."""
        try:
            scheduler_health_file = self.data_dir / ".scheduler_health"
            if scheduler_health_file.exists():
                health_data = json.loads(scheduler_health_file.read_text())
                
                # Processing duration metrics from scheduler logs
                last_duration = health_data.get("last_processing_duration", 0)
                if last_duration:
                    PROCESSING_DURATION.labels(operation="download").observe(last_duration)
                    
        except Exception as e:
            print(f"Error collecting scheduler metrics: {e}")
    
    def collect_detection_metrics(self):
        """Collect detection service metrics."""
        try:
            detector_health_file = self.data_dir / ".detector_health"
            if detector_health_file.exists():
                health_data = json.loads(detector_health_file.read_text())
                
                # Detection accuracy
                accuracy = health_data.get("accuracy", 0)
                model_version = health_data.get("model_version", "unknown")
                
                if accuracy:
                    DETECTION_ACCURACY.labels(model_version=model_version).set(accuracy)
                    
                # Processing duration
                processing_duration = health_data.get("avg_processing_time", 0)
                if processing_duration:
                    PROCESSING_DURATION.labels(operation="detection").observe(processing_duration)
                    
        except Exception as e:
            print(f"Error collecting detection metrics: {e}")
    
    def collect_all_metrics(self):
        """Collect all metrics."""
        self.collect_database_metrics()
        self.collect_filesystem_metrics()
        self.collect_service_health_metrics()
        self.collect_scheduler_metrics()
        self.collect_detection_metrics()
    
    def run_metrics_collection(self):
        """Run metrics collection in a loop."""
        while self.running:
            try:
                self.collect_all_metrics()
                time.sleep(30)  # Collect metrics every 30 seconds
            except Exception as e:
                print(f"Error in metrics collection loop: {e}")
                time.sleep(10)
    
    def start(self, port: int = 9090):
        """Start the metrics server."""
        # Start Prometheus metrics server
        start_http_server(port)
        print(f"Metrics server started on port {port}")
        
        # Start metrics collection thread
        metrics_thread = Thread(target=self.run_metrics_collection, daemon=True)
        metrics_thread.start()
        
        return metrics_thread
    
    def stop(self):
        """Stop the metrics collection."""
        self.running = False


def main():
    """Main function to run the metrics server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Container Analytics Metrics Server")
    parser.add_argument("--port", type=int, default=9090, help="Port to serve metrics on")
    parser.add_argument("--data-dir", type=str, default="/data", help="Data directory path")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    collector = MetricsCollector(data_dir)
    metrics_thread = collector.start(args.port)
    
    try:
        print("Metrics server running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down metrics server...")
        collector.stop()
        print("Metrics server stopped.")


if __name__ == "__main__":
    main()