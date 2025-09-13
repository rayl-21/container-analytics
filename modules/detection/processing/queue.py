"""
Image Processing Queue Management

This module provides thread-safe queue management for image processing tasks
in the YOLO detection pipeline. It ensures proper task distribution and
prevents duplicate processing of images.

Classes:
    ImageProcessingQueue: Thread-safe queue for managing image processing tasks
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)


class ImageProcessingQueue:
    """
    Thread-safe queue for managing image processing tasks.

    This class provides a robust queue system for handling image processing
    tasks with duplicate detection and thread-safe operations. It maintains
    a record of processed files to prevent reprocessing and provides
    controlled access to the processing queue.

    Attributes:
        queue: Internal queue for storing image paths
        processed_files: Set of already processed file paths
        maxsize: Maximum queue size
    """

    def __init__(self, maxsize: int = 100):
        """
        Initialize the image processing queue.

        Args:
            maxsize: Maximum number of items that can be queued
        """
        self.queue = queue.Queue(maxsize=maxsize)
        self.processed_files: Set[str] = set()
        self._lock = threading.Lock()
        self.maxsize = maxsize

        logger.info(f"ImageProcessingQueue initialized with maxsize={maxsize}")

    def add_image(self, image_path: Path) -> bool:
        """
        Add image to processing queue if not already processed.

        Args:
            image_path: Path to the image file to be processed

        Returns:
            True if image was added to queue, False if already processed or queue full
        """
        with self._lock:
            image_str = str(image_path)
            if image_str in self.processed_files:
                logger.debug(f"Image {image_path.name} already processed, skipping")
                return False

            try:
                self.queue.put_nowait(image_path)
                logger.debug(f"Added {image_path.name} to processing queue")
                return True
            except queue.Full:
                logger.warning(f"Processing queue is full (maxsize={self.maxsize}), skipping image {image_path.name}")
                return False

    def get_image(self, timeout: Optional[float] = None) -> Optional[Path]:
        """
        Get next image from queue.

        Args:
            timeout: Maximum time to wait for an item (None for blocking)

        Returns:
            Path to next image to process, or None if timeout/empty
        """
        try:
            image_path = self.queue.get(timeout=timeout)
            logger.debug(f"Retrieved {image_path.name} from processing queue")
            return image_path
        except queue.Empty:
            return None

    def mark_processed(self, image_path: Path) -> None:
        """
        Mark image as processed and complete the queue task.

        Args:
            image_path: Path to the processed image
        """
        with self._lock:
            self.processed_files.add(str(image_path))
            self.queue.task_done()
            logger.debug(f"Marked {image_path.name} as processed")

    def clear_processed_history(self) -> None:
        """
        Clear the processed files history to free memory.

        This method should be called periodically in long-running processes
        to prevent memory buildup from the processed files set.
        """
        with self._lock:
            processed_count = len(self.processed_files)
            self.processed_files.clear()
            logger.info(f"Cleared {processed_count} processed files from history")

    def qsize(self) -> int:
        """
        Get current queue size.

        Returns:
            Number of items currently in the queue
        """
        return self.queue.qsize()

    def is_processed(self, image_path: Path) -> bool:
        """
        Check if an image has already been processed.

        Args:
            image_path: Path to check

        Returns:
            True if image has been processed, False otherwise
        """
        with self._lock:
            return str(image_path) in self.processed_files

    def get_stats(self) -> dict:
        """
        Get queue statistics.

        Returns:
            Dictionary containing queue size and processed files count
        """
        with self._lock:
            return {
                'queue_size': self.queue.qsize(),
                'processed_files_count': len(self.processed_files),
                'maxsize': self.maxsize,
                'queue_full': self.queue.qsize() >= self.maxsize
            }

    def join(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all tasks in the queue to be completed.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if all tasks completed, False if timeout
        """
        try:
            if timeout is None:
                self.queue.join()
                return True
            else:
                # Python's queue.join() doesn't support timeout, so we implement it
                import time
                start_time = time.time()
                while self.queue.unfinished_tasks > 0:
                    if time.time() - start_time > timeout:
                        return False
                    time.sleep(0.1)
                return True
        except Exception as e:
            logger.error(f"Error waiting for queue completion: {e}")
            return False