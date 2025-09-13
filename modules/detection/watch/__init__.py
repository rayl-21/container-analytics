"""
YOLO Watch Mode Module

This module provides components for continuous image processing in YOLO detection
pipelines. It includes file system monitoring, worker pool management, and
processing orchestration.

Components:
- ImageFileHandler: File system event handler for new image detection
- ProcessingWorker: Worker thread for image processing
- WorkerPool: Manager for multiple processing workers
- YOLOWatchMode: Main orchestrator for watch mode functionality

Usage:
    from modules.detection.watch import YOLOWatchMode

    # Initialize with detector
    watch_mode = YOLOWatchMode(detector, watch_directory="data/images")

    # Start processing
    with watch_mode.running_context():
        # Watch mode is now running
        pass

    # Or manage manually
    watch_mode.start()
    try:
        # Do other work
        time.sleep(60)
    finally:
        watch_mode.stop()
"""

from .handler import ImageFileHandler
from .worker import ProcessingWorker, WorkerPool
from .monitor import YOLOWatchMode

__all__ = [
    'ImageFileHandler',
    'ProcessingWorker',
    'WorkerPool',
    'YOLOWatchMode'
]

# Version information
__version__ = '1.0.0'
__author__ = 'Container Analytics Team'

# Module-level configuration
DEFAULT_WATCH_CONFIG = {
    'batch_size': 4,
    'max_workers': 2,
    'queue_size': 100,
    'file_write_delay': 0.5,
    'max_retries': 2,
    'process_existing': False
}

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp', '.gif'
}


def create_watch_mode(detector, watch_directory="data/images", **kwargs):
    """
    Convenience function to create a YOLOWatchMode instance with default configuration.

    Args:
        detector: YOLO detector instance
        watch_directory: Directory to monitor for images
        **kwargs: Additional configuration options

    Returns:
        YOLOWatchMode: Configured watch mode instance
    """
    config = DEFAULT_WATCH_CONFIG.copy()
    config.update(kwargs)

    return YOLOWatchMode(
        detector=detector,
        watch_directory=watch_directory,
        **config
    )