"""
Retry operations for YOLO detection.

This module provides retry logic and error handling for detection operations.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Union, Callable

logger = logging.getLogger(__name__)


class RetryOperations:
    """
    Handles retry logic for detection operations.

    This class provides methods for performing detection with retry logic,
    error handling, and exponential backoff.
    """

    def __init__(self, detector):
        """
        Initialize retry operations.

        Args:
            detector: The YOLODetector instance to use for retries
        """
        self.detector = detector

    def detect_with_retry(
        self,
        image_path: Union[str, Path],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        save_to_db: bool = True
    ) -> Optional[Dict]:
        """
        Detect objects with retry logic and error handling.

        Args:
            image_path: Path to the image file
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            save_to_db: Whether to save results to database

        Returns:
            Detection results dictionary or None if failed
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Try to detect objects
                result = self.detector.detect_single_image(image_path)

                # Save to database if requested
                if save_to_db and result:
                    image_id = self.detector.save_detection_to_database(
                        image_path,
                        result['detections'],
                        result['metadata']['processing_time']
                    )
                    result['metadata']['image_id'] = image_id

                logger.info(f"Successfully processed {Path(image_path).name} on attempt {attempt + 1}")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Detection attempt {attempt + 1} failed for {image_path}: {e}")

                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    logger.info(f"Retrying detection for {image_path} in {delay:.1f}s...")

        logger.error(f"All {max_retries + 1} detection attempts failed for {image_path}: {last_error}")
        return None

    def batch_detect_with_retry(
        self,
        image_paths: list,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        save_to_db: bool = True
    ) -> list:
        """
        Perform batch detection with retry logic for failed images.

        Args:
            image_paths: List of image paths to process
            max_retries: Maximum number of retry attempts per image
            retry_delay: Delay between retries in seconds
            save_to_db: Whether to save results to database

        Returns:
            List of detection results (None for failed images)
        """
        results = []
        failed_images = []

        try:
            # First attempt: batch processing
            batch_results = self.detector.detect_batch(image_paths, return_annotated=False)
            results.extend(batch_results)

            # Save to database if requested
            if save_to_db:
                for result in batch_results:
                    if result:
                        self.detector.save_detection_to_database(
                            result['metadata']['image_path'],
                            result['detections'],
                            result['metadata'].get('processing_time', 0)
                        )

        except Exception as e:
            logger.warning(f"Batch processing failed: {e}")
            # Fall back to individual processing with retry
            failed_images.extend(image_paths)

        # Retry failed images individually
        if failed_images:
            logger.info(f"Retrying {len(failed_images)} failed images individually")
            for image_path in failed_images:
                result = self.detect_with_retry(
                    image_path, max_retries, retry_delay, save_to_db
                )
                results.append(result)

        return results

    def detect_with_timeout(
        self,
        image_path: Union[str, Path],
        timeout_seconds: float = 30.0,
        save_to_db: bool = True
    ) -> Optional[Dict]:
        """
        Detect objects with timeout protection.

        Args:
            image_path: Path to the image file
            timeout_seconds: Maximum time to wait for detection
            save_to_db: Whether to save results to database

        Returns:
            Detection results dictionary or None if timeout/failed
        """
        import threading
        import queue

        result_queue = queue.Queue()
        error_queue = queue.Queue()

        def detection_worker():
            try:
                result = self.detector.detect_single_image(image_path)
                if save_to_db and result:
                    image_id = self.detector.save_detection_to_database(
                        image_path,
                        result['detections'],
                        result['metadata']['processing_time']
                    )
                    result['metadata']['image_id'] = image_id
                result_queue.put(result)
            except Exception as e:
                error_queue.put(e)

        # Start detection in separate thread
        worker_thread = threading.Thread(target=detection_worker, daemon=True)
        worker_thread.start()

        # Wait for result or timeout
        worker_thread.join(timeout=timeout_seconds)

        if worker_thread.is_alive():
            logger.error(f"Detection timed out after {timeout_seconds}s for {image_path}")
            return None

        # Check for results or errors
        if not result_queue.empty():
            return result_queue.get()
        elif not error_queue.empty():
            error = error_queue.get()
            logger.error(f"Detection failed for {image_path}: {error}")
            return None
        else:
            logger.error(f"Unknown error during detection for {image_path}")
            return None