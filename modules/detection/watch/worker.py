"""
Processing Worker for YOLO Watch Mode

This module provides the ProcessingWorker class that handles the actual image
processing in separate threads, including batch processing and error recovery.

Features:
- Multi-threaded image processing
- Batch processing for efficiency
- Error handling and recovery
- Processing statistics tracking
- Graceful shutdown handling
"""

import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING

# Import processing queue
from ..processing import ImageProcessingQueue

if TYPE_CHECKING:
    from ..yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class ProcessingWorker:
    """
    Processing worker for handling image detection in separate threads.

    This class manages the worker thread that processes images from the queue,
    including batch processing, error handling, and statistics tracking.
    """

    def __init__(
        self,
        detector: 'YOLODetector',
        processing_queue: ImageProcessingQueue,
        batch_size: int = 4,
        worker_id: Optional[str] = None,
        max_retries: int = 2
    ):
        """
        Initialize the ProcessingWorker.

        Args:
            detector: YOLO detector instance for processing images
            processing_queue: Queue containing images to process
            batch_size: Maximum number of images to process in a batch
            worker_id: Optional worker identifier (auto-generated if None)
            max_retries: Maximum number of retry attempts for failed images
        """
        self.detector = detector
        self.processing_queue = processing_queue
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Worker identification
        self.worker_id = worker_id or f"Worker-{id(self)}"
        self.thread: Optional[threading.Thread] = None

        # Worker state
        self.is_running = False
        self.should_stop = False

        # Statistics
        self.stats = {
            'images_processed': 0,
            'images_failed': 0,
            'batches_processed': 0,
            'total_detections': 0,
            'start_time': None,
            'last_processed': None,
            'current_batch_size': 0
        }

        logger.info(f"Processing worker {self.worker_id} initialized with batch_size={batch_size}")

    def start(self, daemon: bool = True):
        """
        Start the worker thread.

        Args:
            daemon: Whether to run as daemon thread
        """
        if self.is_running:
            logger.warning(f"Worker {self.worker_id} is already running")
            return

        self.should_stop = False
        self.stats['start_time'] = time.time()
        self.stats['images_processed'] = 0
        self.stats['images_failed'] = 0
        self.stats['batches_processed'] = 0
        self.stats['total_detections'] = 0

        self.thread = threading.Thread(
            target=self._worker_loop,
            name=f"YOLO-{self.worker_id}",
            daemon=daemon
        )
        self.thread.start()
        self.is_running = True

        logger.info(f"Processing worker {self.worker_id} started")

    def stop(self, timeout: float = 5.0):
        """
        Stop the worker thread.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self.is_running:
            logger.warning(f"Worker {self.worker_id} is not running")
            return

        logger.info(f"Stopping processing worker {self.worker_id}...")
        self.should_stop = True

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning(f"Worker {self.worker_id} did not stop gracefully within {timeout}s")

        self.is_running = False
        self._log_final_stats()

    def _worker_loop(self):
        """Main worker thread loop."""
        logger.info(f"Processing worker {self.worker_id} started main loop")

        batch_images = []

        try:
            while not self.should_stop:
                try:
                    # Get image from queue with timeout
                    image_path = self.processing_queue.get_image(timeout=1.0)

                    if image_path is None:
                        # Process any remaining batch images
                        if batch_images:
                            self._process_batch(batch_images)
                            batch_images = []
                        continue

                    batch_images.append(image_path)
                    self.stats['current_batch_size'] = len(batch_images)

                    # Process batch when full or no more images in queue
                    if (len(batch_images) >= self.batch_size or
                        self.processing_queue.qsize() == 0):
                        self._process_batch(batch_images)
                        batch_images = []
                        self.stats['current_batch_size'] = 0

                except Exception as e:
                    logger.error(f"Error in processing worker {self.worker_id} main loop: {e}")
                    # Mark any images in current batch as processed to avoid blocking
                    self._cleanup_batch(batch_images)
                    batch_images = []
                    self.stats['current_batch_size'] = 0
                    time.sleep(1.0)  # Brief pause before retrying

        finally:
            # Process any remaining images
            if batch_images:
                self._process_batch(batch_images)

        logger.info(f"Processing worker {self.worker_id} stopped main loop")

    def _process_batch(self, image_paths: List[Path]):
        """
        Process a batch of images.

        Args:
            image_paths: List of image file paths to process
        """
        if not image_paths:
            return

        batch_size = len(image_paths)
        logger.info(f"Worker {self.worker_id} processing batch of {batch_size} images")

        batch_start_time = time.time()
        batch_detections = 0

        for image_path in image_paths:
            try:
                # Process image with retry logic
                result = self.detector.detect_with_retry(
                    image_path,
                    max_retries=self.max_retries,
                    save_to_db=True
                )

                if result and 'metadata' in result:
                    num_detections = result['metadata'].get('num_detections', 0)
                    processing_time = result['metadata'].get('processing_time', 0)

                    self.stats['images_processed'] += 1
                    self.stats['total_detections'] += num_detections
                    self.stats['last_processed'] = time.time()
                    batch_detections += num_detections

                    logger.info(
                        f"Worker {self.worker_id} processed {image_path.name}: "
                        f"{num_detections} detections in {processing_time:.2f}s"
                    )
                else:
                    self.stats['images_failed'] += 1
                    logger.error(f"Worker {self.worker_id} failed to process {image_path.name}")

            except Exception as e:
                self.stats['images_failed'] += 1
                logger.error(f"Worker {self.worker_id} error processing {image_path.name}: {e}")

            finally:
                # Mark image as processed
                self.processing_queue.mark_processed(image_path)

        # Update batch statistics
        self.stats['batches_processed'] += 1
        batch_time = time.time() - batch_start_time

        logger.info(
            f"Worker {self.worker_id} completed batch: {batch_size} images, "
            f"{batch_detections} detections in {batch_time:.2f}s"
        )

    def _cleanup_batch(self, image_paths: List[Path]):
        """
        Clean up a batch of images by marking them as processed.

        Args:
            image_paths: List of image file paths to clean up
        """
        for image_path in image_paths:
            try:
                self.processing_queue.mark_processed(image_path)
                logger.debug(f"Cleaned up {image_path.name}")
            except Exception as e:
                logger.error(f"Error cleaning up {image_path.name}: {e}")

    def _log_final_stats(self):
        """Log final processing statistics."""
        runtime = 0
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']

        logger.info(
            f"Worker {self.worker_id} final stats: {self.stats['images_processed']} processed, "
            f"{self.stats['images_failed']} failed, {self.stats['batches_processed']} batches, "
            f"{self.stats['total_detections']} total detections in {runtime:.1f}s"
        )

    def get_stats(self) -> Dict:
        """
        Get current processing statistics.

        Returns:
            Dictionary containing current worker statistics
        """
        runtime = 0
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']

        return {
            'worker_id': self.worker_id,
            'is_running': self.is_running,
            'should_stop': self.should_stop,
            'runtime_seconds': runtime,
            'images_processed': self.stats['images_processed'],
            'images_failed': self.stats['images_failed'],
            'batches_processed': self.stats['batches_processed'],
            'total_detections': self.stats['total_detections'],
            'current_batch_size': self.stats['current_batch_size'],
            'success_rate': (
                self.stats['images_processed'] /
                max(1, self.stats['images_processed'] + self.stats['images_failed'])
            ) * 100,
            'processing_rate': (
                self.stats['images_processed'] / max(1, runtime / 60)
            ) if runtime > 0 else 0,
            'last_processed': self.stats['last_processed']
        }

    def is_alive(self) -> bool:
        """
        Check if the worker thread is alive.

        Returns:
            True if the worker thread is running, False otherwise
        """
        return self.thread is not None and self.thread.is_alive()


class WorkerPool:
    """
    Manager for multiple processing workers.

    This class manages a pool of ProcessingWorker instances for parallel
    image processing.
    """

    def __init__(
        self,
        detector: 'YOLODetector',
        processing_queue: ImageProcessingQueue,
        max_workers: int = 2,
        batch_size: int = 4,
        max_retries: int = 2
    ):
        """
        Initialize the WorkerPool.

        Args:
            detector: YOLO detector instance
            processing_queue: Shared processing queue
            max_workers: Maximum number of worker threads
            batch_size: Batch size for each worker
            max_retries: Maximum retry attempts per image
        """
        self.detector = detector
        self.processing_queue = processing_queue
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_retries = max_retries

        self.workers: List[ProcessingWorker] = []
        self.is_running = False

        logger.info(f"WorkerPool initialized with {max_workers} workers")

    def start(self):
        """Start all workers in the pool."""
        if self.is_running:
            logger.warning("WorkerPool is already running")
            return

        self.workers.clear()

        for i in range(self.max_workers):
            worker = ProcessingWorker(
                detector=self.detector,
                processing_queue=self.processing_queue,
                batch_size=self.batch_size,
                worker_id=f"Worker-{i+1}",
                max_retries=self.max_retries
            )
            worker.start(daemon=True)
            self.workers.append(worker)

        self.is_running = True
        logger.info(f"WorkerPool started with {len(self.workers)} workers")

    def stop(self, timeout: float = 5.0):
        """
        Stop all workers in the pool.

        Args:
            timeout: Maximum time to wait for each worker to stop
        """
        if not self.is_running:
            logger.warning("WorkerPool is not running")
            return

        logger.info("Stopping WorkerPool...")

        for worker in self.workers:
            worker.stop(timeout=timeout)

        self.workers.clear()
        self.is_running = False

        logger.info("WorkerPool stopped")

    def get_stats(self) -> Dict:
        """
        Get aggregated statistics from all workers.

        Returns:
            Dictionary containing aggregated worker statistics
        """
        if not self.workers:
            return {
                'total_workers': 0,
                'active_workers': 0,
                'total_images_processed': 0,
                'total_images_failed': 0,
                'total_batches_processed': 0,
                'total_detections': 0
            }

        total_images_processed = sum(w.stats['images_processed'] for w in self.workers)
        total_images_failed = sum(w.stats['images_failed'] for w in self.workers)
        total_batches_processed = sum(w.stats['batches_processed'] for w in self.workers)
        total_detections = sum(w.stats['total_detections'] for w in self.workers)
        active_workers = sum(1 for w in self.workers if w.is_alive())

        return {
            'total_workers': len(self.workers),
            'active_workers': active_workers,
            'total_images_processed': total_images_processed,
            'total_images_failed': total_images_failed,
            'total_batches_processed': total_batches_processed,
            'total_detections': total_detections,
            'overall_success_rate': (
                total_images_processed /
                max(1, total_images_processed + total_images_failed)
            ) * 100,
            'workers': [worker.get_stats() for worker in self.workers]
        }