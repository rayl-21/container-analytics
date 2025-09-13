"""
Processing package for YOLO detection pipeline.

This package provides modular components for batch processing, queue management,
and pipeline integration for the YOLO detection system.

Components:
- queue: ImageProcessingQueue for thread-safe task management
- batch: BatchProcessor for efficient batch operations
- pipeline: DetectionPipeline for full pipeline integration
"""

from .queue import ImageProcessingQueue
from .batch import BatchProcessor
from .pipeline import DetectionPipeline

__all__ = [
    'ImageProcessingQueue',
    'BatchProcessor',
    'DetectionPipeline'
]