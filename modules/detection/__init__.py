"""
Detection Module for Container Analytics

This module provides computer vision capabilities for container detection.

Components:
- YOLODetector: YOLOv12-based object detection for containers and vehicles

Usage:
    from modules.detection import YOLODetector

    detector = YOLODetector()
"""

from .yolo_detector import YOLODetector

__all__ = ["YOLODetector"]

# Version information
__version__ = "1.0.0"
__author__ = "Container Analytics Team"