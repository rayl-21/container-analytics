"""
Detection Module for Container Analytics

This module provides computer vision capabilities for container detection,
tracking, and OCR functionality with a modular, refactored architecture.

Main Components:
- YOLODetector: YOLOv12-based object detection for containers and vehicles
- ContainerTracker: ByteTrack-based object tracking with dwell time calculation
- ContainerOCR: OCR for extracting container numbers from detected containers

Processing Pipeline:
- DetectionPipeline: Full detection pipeline with database integration
- BatchProcessor: Efficient batch processing capabilities
- ImageProcessingQueue: Thread-safe task management

Watch Mode:
- YOLOWatchMode: Continuous processing of new images
- create_watch_mode: Convenience function for watch mode setup

Utilities:
- DetectionConfig: Configuration management for detection parameters
- DatabaseOperations: Database integration utilities
- CONTAINER_CLASSES: Predefined container and vehicle class mappings

Usage Examples:
    # Basic detection
    from modules.detection import YOLODetector
    detector = YOLODetector()

    # Full pipeline with database
    from modules.detection import DetectionPipeline
    pipeline = DetectionPipeline(detector)

    # Watch mode for continuous processing
    from modules.detection import YOLOWatchMode, create_watch_mode
    watch_mode = create_watch_mode(detector, "data/images")

    # CLI interface
    from modules.detection.cli import main
    # Or run: python -m modules.detection.cli --help
"""

# Core detection classes
from .yolo_detector import YOLODetector
from .tracker import ContainerTracker
from .ocr import ContainerOCR

# Processing pipeline components
from .processing import DetectionPipeline, BatchProcessor, ImageProcessingQueue

# Watch mode components
from .watch import YOLOWatchMode, create_watch_mode

# Utility classes and constants
from .utils import (
    DetectionConfig,
    DatabaseOperations,
    CONTAINER_CLASSES,
    ImageAnnotator,
    PerformanceTracker
)

# Main public API
__all__ = [
    # Core detection
    "YOLODetector",
    "ContainerTracker",
    "ContainerOCR",

    # Processing pipeline
    "DetectionPipeline",
    "BatchProcessor",
    "ImageProcessingQueue",

    # Watch mode
    "YOLOWatchMode",
    "create_watch_mode",

    # Utilities
    "DetectionConfig",
    "DatabaseOperations",
    "CONTAINER_CLASSES",
    "ImageAnnotator",
    "PerformanceTracker"
]

# Version information
__version__ = "1.0.0"
__author__ = "Container Analytics Team"

# Module-level convenience functions
def create_detector(model_path="yolov12x.pt", **kwargs):
    """
    Convenience function to create a YOLODetector with default settings.

    Args:
        model_path: Path to YOLO model weights
        **kwargs: Additional detector configuration

    Returns:
        YOLODetector: Configured detector instance
    """
    return YOLODetector(model_path=model_path, **kwargs)


def create_pipeline(detector=None, **kwargs):
    """
    Convenience function to create a detection pipeline.

    Args:
        detector: YOLODetector instance (creates default if None)
        **kwargs: Additional pipeline configuration

    Returns:
        DetectionPipeline: Configured pipeline instance
    """
    if detector is None:
        detector = create_detector()
    return DetectionPipeline(detector=detector, **kwargs)