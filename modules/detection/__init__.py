"""
Detection Module for Container Analytics

This module provides computer vision capabilities for container detection,
tracking, and OCR functionality.

Components:
- YOLODetector: YOLOv8-based object detection for containers and vehicles
- ContainerTracker: ByteTrack-based object tracking with dwell time calculation
- ContainerOCR: OCR for extracting container numbers from detected containers

Usage:
    from modules.detection import YOLODetector, ContainerTracker, ContainerOCR
    
    detector = YOLODetector()
    tracker = ContainerTracker()
    ocr = ContainerOCR()
"""

from .yolo_detector import YOLODetector
from .tracker import ContainerTracker
from .ocr import ContainerOCR

__all__ = ["YOLODetector", "ContainerTracker", "ContainerOCR"]

# Version information
__version__ = "1.0.0"
__author__ = "Container Analytics Team"