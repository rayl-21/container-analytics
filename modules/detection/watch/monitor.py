"""
YOLO Watch Mode Monitor

This module provides the YOLOWatchMode class that orchestrates continuous image
processing by monitoring a directory for new images and coordinating workers
to process them.

Features:
- Directory monitoring with file system events
- Worker pool management
- Processing statistics and monitoring
- Lifecycle management with context manager support
- Configurable batch processing and worker settings
"""

import logging
import time
from pathlib import Path
from typing import Union, Dict, Optional, TYPE_CHECKING
from contextlib import contextmanager
from watchdog.observers import Observer

# Import local modules
from .handler import ImageFileHandler
from .worker import WorkerPool
from ..processing import ImageProcessingQueue

if TYPE_CHECKING:
    from ..yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class YOLOWatchMode:
    """
    Watch mode functionality for continuous image processing.

    Monitors a directory for new images and processes them automatically
    using YOLO detection with database persistence. Coordinates file system
    monitoring, worker pool management, and processing statistics.
    """

    def __init__(
        self,
        detector: 'YOLODetector',
        watch_directory: Union[str, Path] = "data/images",
        batch_size: int = 4,
        max_workers: int = 2,
        queue_size: int = 100,
        process_existing: bool = False,
        file_write_delay: float = 0.5,
        max_retries: int = 2
    ):
        """
        Initialize YOLOWatchMode.

        Args:
            detector: YOLO detector instance for processing images
            watch_directory: Directory to monitor for new images
            batch_size: Number of images to process in each batch
            max_workers: Maximum number of worker threads
            queue_size: Maximum size of the processing queue
            process_existing: Whether to process existing images on startup
            file_write_delay: Delay to wait after file creation
            max_retries: Maximum retry attempts for failed images
        """
        self.detector = detector
        self.watch_directory = Path(watch_directory)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.process_existing = process_existing
        self.file_write_delay = file_write_delay
        self.max_retries = max_retries

        # Initialize processing queue
        self.processing_queue = ImageProcessingQueue(maxsize=queue_size)

        # Initialize file system observer and handler
        self.observer = Observer()
        self.file_handler = ImageFileHandler(
            processing_queue=self.processing_queue,
            file_write_delay=file_write_delay
        )

        # Initialize worker pool
        self.worker_pool = WorkerPool(
            detector=detector,
            processing_queue=self.processing_queue,
            max_workers=max_workers,
            batch_size=batch_size,
            max_retries=max_retries
        )

        # Monitoring state
        self.is_running = False

        # Processing statistics
        self.stats = {
            'start_time': None,
            'last_activity': None,
            'files_detected': 0,
            'queue_peak_size': 0
        }

        logger.info(f"YOLO Watch Mode initialized for directory: {self.watch_directory}")
        logger.info(f"Configuration: {max_workers} workers, batch_size={batch_size}, "
                   f"queue_size={queue_size}")

    def _ensure_directory_exists(self):
        """Ensure the watch directory exists."""
        try:
            self.watch_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Watch directory ready: {self.watch_directory}")
        except Exception as e:
            logger.error(f"Failed to create watch directory {self.watch_directory}: {e}")
            raise

    def _process_existing_images(self):
        """Process existing images in the watch directory."""
        if not self.process_existing:
            logger.info("Skipping existing images (process_existing=False)")
            return

        logger.info("Processing existing images in watch directory...")

        existing_images = []
        supported_extensions = self.file_handler.get_supported_extensions()

        # Find all existing images
        for ext in supported_extensions:
            # Search in current directory
            existing_images.extend(self.watch_directory.glob(f"*{ext}"))
            # Search recursively
            existing_images.extend(self.watch_directory.glob(f"**/*{ext}"))

        # Remove duplicates and sort by modification time
        existing_images = list(set(existing_images))
        existing_images.sort(key=lambda p: p.stat().st_mtime)

        logger.info(f"Found {len(existing_images)} existing images")

        # Add to processing queue
        queued_count = 0
        for image_path in existing_images:
            if self.processing_queue.add_image(image_path):
                queued_count += 1
                logger.debug(f"Queued existing image: {image_path.name}")

        logger.info(f"Queued {queued_count} existing images for processing")

    def _update_stats(self):
        """Update monitoring statistics."""
        current_queue_size = self.processing_queue.qsize()
        if current_queue_size > self.stats['queue_peak_size']:
            self.stats['queue_peak_size'] = current_queue_size

        self.stats['last_activity'] = time.time()

    def start(self, process_existing: Optional[bool] = None):
        """
        Start watch mode processing.

        Args:
            process_existing: Override for processing existing images
        """
        if self.is_running:
            logger.warning("Watch mode is already running")
            return

        # Ensure watch directory exists
        self._ensure_directory_exists()

        logger.info(f"Starting YOLO watch mode on {self.watch_directory}")

        # Initialize statistics
        self.stats['start_time'] = time.time()
        self.stats['last_activity'] = time.time()
        self.stats['files_detected'] = 0
        self.stats['queue_peak_size'] = 0

        # Set running flag
        self.is_running = True

        try:
            # Process existing images if requested
            if process_existing or (process_existing is None and self.process_existing):
                self._process_existing_images()

            # Start worker pool
            self.worker_pool.start()

            # Start file system observer
            self.observer.schedule(
                self.file_handler,
                str(self.watch_directory),
                recursive=True
            )
            self.observer.start()

            logger.info(
                f"Watch mode started with {self.max_workers} workers monitoring "
                f"{self.watch_directory} (recursive)"
            )

        except Exception as e:
            logger.error(f"Failed to start watch mode: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop watch mode processing."""
        if not self.is_running:
            logger.warning("Watch mode is not running")
            return

        logger.info("Stopping YOLO watch mode...")

        try:
            # Stop file observer
            if self.observer.is_alive():
                self.observer.stop()
                self.observer.join(timeout=5.0)

            # Stop worker pool
            self.worker_pool.stop(timeout=5.0)

            # Set running flag to False
            self.is_running = False

            # Log final statistics
            self._log_final_stats()

        except Exception as e:
            logger.error(f"Error during watch mode shutdown: {e}")

        logger.info("Watch mode stopped")

    def _log_final_stats(self):
        """Log final processing statistics."""
        if not self.stats['start_time']:
            return

        runtime = time.time() - self.stats['start_time']
        worker_stats = self.worker_pool.get_stats()

        logger.info(
            f"Watch mode final stats: {worker_stats['total_images_processed']} processed, "
            f"{worker_stats['total_images_failed']} failed, "
            f"{worker_stats['total_detections']} total detections, "
            f"queue peak: {self.stats['queue_peak_size']}, "
            f"runtime: {runtime:.1f}s"
        )

    def get_stats(self) -> Dict:
        """
        Get current processing statistics.

        Returns:
            Dictionary containing comprehensive processing statistics
        """
        runtime = 0
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']

        # Get worker pool statistics
        worker_stats = self.worker_pool.get_stats()

        # Combine with monitor-level statistics
        combined_stats = {
            'monitor': {
                'is_running': self.is_running,
                'runtime_seconds': runtime,
                'files_detected': self.stats['files_detected'],
                'queue_peak_size': self.stats['queue_peak_size'],
                'last_activity': self.stats['last_activity'],
                'watch_directory': str(self.watch_directory),
                'configuration': {
                    'batch_size': self.batch_size,
                    'max_workers': self.max_workers,
                    'process_existing': self.process_existing,
                    'file_write_delay': self.file_write_delay,
                    'max_retries': self.max_retries
                }
            },
            'queue': {
                'current_size': self.processing_queue.qsize(),
                'max_size': self.processing_queue.maxsize,
                'utilization_percent': (
                    self.processing_queue.qsize() / self.processing_queue.maxsize * 100
                    if self.processing_queue.maxsize > 0 else 0
                )
            },
            'workers': worker_stats,
            'file_handler': {
                'supported_extensions': list(self.file_handler.get_supported_extensions())
            }
        }

        return combined_stats

    def add_supported_extension(self, extension: str):
        """
        Add a supported file extension to the file handler.

        Args:
            extension: File extension to add
        """
        self.file_handler.add_supported_extension(extension)

    def remove_supported_extension(self, extension: str):
        """
        Remove a supported file extension from the file handler.

        Args:
            extension: File extension to remove
        """
        self.file_handler.remove_supported_extension(extension)

    def update_file_write_delay(self, delay: float):
        """
        Update the file write delay.

        Args:
            delay: New delay in seconds
        """
        self.file_handler.update_file_write_delay(delay)
        self.file_write_delay = delay

    def get_queue_info(self) -> Dict:
        """
        Get detailed queue information.

        Returns:
            Dictionary containing queue status and statistics
        """
        return {
            'size': self.processing_queue.qsize(),
            'max_size': self.processing_queue.maxsize,
            'empty': self.processing_queue.empty(),
            'full': self.processing_queue.full(),
            'utilization_percent': (
                self.processing_queue.qsize() / self.processing_queue.maxsize * 100
                if self.processing_queue.maxsize > 0 else 0
            )
        }

    def is_healthy(self) -> bool:
        """
        Check if the watch mode is healthy and functioning.

        Returns:
            True if watch mode is running and healthy, False otherwise
        """
        if not self.is_running:
            return False

        # Check if observer is alive
        if not self.observer.is_alive():
            logger.warning("File system observer is not alive")
            return False

        # Check if workers are active
        worker_stats = self.worker_pool.get_stats()
        if worker_stats['active_workers'] == 0:
            logger.warning("No active workers")
            return False

        return True

    @contextmanager
    def running_context(self, process_existing: Optional[bool] = None):
        """
        Context manager for watch mode.

        Args:
            process_existing: Whether to process existing images

        Yields:
            The YOLOWatchMode instance
        """
        try:
            self.start(process_existing=process_existing)
            yield self
        finally:
            self.stop()

    def __enter__(self):
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.stop()
        return False