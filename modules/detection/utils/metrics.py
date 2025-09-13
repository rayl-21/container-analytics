"""
Performance tracking and metrics for YOLO detection.

This module provides functionality to track and analyze performance metrics
for the YOLO detection system, including processing times, throughput, and
detection statistics.
"""

import time
import numpy as np
from typing import Dict, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks and analyzes performance metrics for YOLO detection.

    This class maintains detection timing data, calculates performance statistics,
    and provides methods for performance analysis and monitoring.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the performance tracker.

        Args:
            max_history: Maximum number of timing measurements to keep in memory
        """
        self.max_history = max_history
        self.detection_times = deque(maxlen=max_history)
        self.batch_times = deque(maxlen=max_history)
        self.detection_counts = deque(maxlen=max_history)
        self.start_time = None
        self.total_images_processed = 0
        self.total_detections_found = 0

    def record_detection_time(self, processing_time: float, num_detections: int = 0):
        """
        Record a detection processing time.

        Args:
            processing_time: Time taken to process the detection in seconds
            num_detections: Number of detections found in the image
        """
        self.detection_times.append(processing_time)
        self.detection_counts.append(num_detections)
        self.total_images_processed += 1
        self.total_detections_found += num_detections

    def record_batch_time(self, batch_time: float, batch_size: int):
        """
        Record a batch processing time.

        Args:
            batch_time: Time taken to process the batch in seconds
            batch_size: Number of images in the batch
        """
        self.batch_times.append(batch_time)

    def get_performance_stats(self) -> Dict:
        """
        Get comprehensive performance statistics.

        Returns:
            Dictionary containing performance metrics
        """
        if not self.detection_times:
            return {"message": "No detections performed yet"}

        times = np.array(self.detection_times)
        detection_counts = np.array(self.detection_counts)

        stats = {
            "timing_stats": {
                "total_detections": len(times),
                "mean_time": float(np.mean(times)),
                "median_time": float(np.median(times)),
                "min_time": float(np.min(times)),
                "max_time": float(np.max(times)),
                "std_time": float(np.std(times)),
                "p95_time": float(np.percentile(times, 95)),
                "p99_time": float(np.percentile(times, 99))
            },
            "throughput_stats": {
                "fps_mean": 1.0 / float(np.mean(times)) if np.mean(times) > 0 else 0,
                "fps_median": 1.0 / float(np.median(times)) if np.median(times) > 0 else 0,
                "images_per_minute": 60.0 / float(np.mean(times)) if np.mean(times) > 0 else 0
            },
            "detection_stats": {
                "total_images_processed": self.total_images_processed,
                "total_detections_found": self.total_detections_found,
                "mean_detections_per_image": float(np.mean(detection_counts)),
                "max_detections_per_image": int(np.max(detection_counts)) if len(detection_counts) > 0 else 0
            }
        }

        # Add batch statistics if available
        if self.batch_times:
            batch_times = np.array(self.batch_times)
            stats["batch_stats"] = {
                "total_batches": len(batch_times),
                "mean_batch_time": float(np.mean(batch_times)),
                "median_batch_time": float(np.median(batch_times))
            }

        return stats

    def get_recent_performance(self, last_n: int = 10) -> Dict:
        """
        Get performance statistics for the most recent detections.

        Args:
            last_n: Number of recent detections to analyze

        Returns:
            Dictionary containing recent performance metrics
        """
        if not self.detection_times:
            return {"message": "No detections performed yet"}

        # Get the last N detection times
        recent_times = list(self.detection_times)[-last_n:]
        recent_counts = list(self.detection_counts)[-last_n:]

        if not recent_times:
            return {"message": "No recent detections available"}

        times = np.array(recent_times)
        counts = np.array(recent_counts)

        return {
            "sample_size": len(times),
            "recent_mean_time": float(np.mean(times)),
            "recent_fps": 1.0 / float(np.mean(times)) if np.mean(times) > 0 else 0,
            "recent_mean_detections": float(np.mean(counts)),
            "trend": self._calculate_trend(times)
        }

    def _calculate_trend(self, times: np.ndarray) -> str:
        """
        Calculate performance trend based on recent timing data.

        Args:
            times: Array of recent processing times

        Returns:
            String describing the performance trend
        """
        if len(times) < 5:
            return "insufficient_data"

        # Split into first and second half
        mid = len(times) // 2
        first_half = times[:mid]
        second_half = times[mid:]

        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)

        # Calculate percentage change
        if first_mean > 0:
            change_percent = ((second_mean - first_mean) / first_mean) * 100

            if change_percent > 10:
                return "degrading"
            elif change_percent < -10:
                return "improving"
            else:
                return "stable"
        else:
            return "unknown"

    def get_performance_summary(self) -> str:
        """
        Get a human-readable performance summary.

        Returns:
            String containing performance summary
        """
        if not self.detection_times:
            return "No performance data available"

        stats = self.get_performance_stats()
        timing = stats["timing_stats"]
        throughput = stats["throughput_stats"]
        detection = stats["detection_stats"]

        summary = f"""Performance Summary:
- Total Images Processed: {detection['total_images_processed']}
- Total Detections Found: {detection['total_detections_found']}
- Average Processing Time: {timing['mean_time']:.3f}s
- Average FPS: {throughput['fps_mean']:.1f}
- Average Detections per Image: {detection['mean_detections_per_image']:.1f}
- P95 Processing Time: {timing['p95_time']:.3f}s
- Time Range: {timing['min_time']:.3f}s - {timing['max_time']:.3f}s"""

        return summary

    def clear_metrics(self):
        """Clear all performance metrics and reset counters."""
        self.detection_times.clear()
        self.batch_times.clear()
        self.detection_counts.clear()
        self.total_images_processed = 0
        self.total_detections_found = 0
        logger.info("Performance metrics cleared")

    def export_metrics_data(self) -> Dict:
        """
        Export raw metrics data for external analysis.

        Returns:
            Dictionary containing all raw timing data
        """
        return {
            "detection_times": list(self.detection_times),
            "batch_times": list(self.batch_times),
            "detection_counts": list(self.detection_counts),
            "total_images_processed": self.total_images_processed,
            "total_detections_found": self.total_detections_found,
            "max_history": self.max_history
        }

    def is_performance_degrading(self, threshold_percent: float = 20.0) -> bool:
        """
        Check if performance is degrading based on recent trends.

        Args:
            threshold_percent: Percentage degradation threshold

        Returns:
            True if performance is degrading beyond threshold
        """
        if len(self.detection_times) < 20:
            return False

        recent_stats = self.get_recent_performance(last_n=10)
        if "recent_mean_time" not in recent_stats:
            return False

        # Compare recent performance to overall average
        overall_stats = self.get_performance_stats()
        overall_mean = overall_stats["timing_stats"]["mean_time"]
        recent_mean = recent_stats["recent_mean_time"]

        if overall_mean > 0:
            degradation_percent = ((recent_mean - overall_mean) / overall_mean) * 100
            return degradation_percent > threshold_percent

        return False

    def get_memory_usage_stats(self) -> Dict:
        """
        Get statistics about memory usage of the metrics tracking.

        Returns:
            Dictionary containing memory usage information
        """
        return {
            "detection_times_count": len(self.detection_times),
            "batch_times_count": len(self.batch_times),
            "detection_counts_count": len(self.detection_counts),
            "max_history": self.max_history,
            "memory_usage_estimate_kb": (
                len(self.detection_times) * 8 +  # 8 bytes per float
                len(self.batch_times) * 8 +
                len(self.detection_counts) * 4    # 4 bytes per int
            ) / 1024
        }