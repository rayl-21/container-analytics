"""
Batch Processing Module for YOLO Detection

This module provides efficient batch processing capabilities for the YOLO detection
system. It handles batch size optimization, progress tracking, and efficient
memory management during large-scale image processing operations.

Classes:
    BatchProcessor: Handles efficient batch processing of images
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Efficient batch processor for YOLO detection operations.

    This class handles batch processing optimization, progress tracking,
    and memory management for large-scale image detection tasks.

    Attributes:
        default_batch_size: Default batch size for processing
        progress_callback: Optional callback for progress updates
    """

    def __init__(
        self,
        default_batch_size: int = 8,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Initialize the batch processor.

        Args:
            default_batch_size: Default batch size for processing operations
            progress_callback: Optional callback function for progress updates
                             Called with (current_batch, total_batches)
        """
        self.default_batch_size = default_batch_size
        self.progress_callback = progress_callback

        logger.info(f"BatchProcessor initialized with default_batch_size={default_batch_size}")

    def create_batches(
        self,
        items: List[Any],
        batch_size: Optional[int] = None
    ) -> List[List[Any]]:
        """
        Create batches from a list of items.

        Args:
            items: List of items to batch
            batch_size: Batch size (uses default if None)

        Returns:
            List of batches, where each batch is a list of items
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)

        logger.debug(f"Created {len(batches)} batches from {len(items)} items (batch_size={batch_size})")
        return batches

    def process_batches(
        self,
        items: List[Any],
        process_function: Callable[[List[Any]], List[Dict]],
        batch_size: Optional[int] = None,
        progress_update_interval: int = 1
    ) -> List[Dict]:
        """
        Process items in batches using the provided function.

        Args:
            items: List of items to process
            process_function: Function that processes a batch and returns results
            batch_size: Batch size (uses default if None)
            progress_update_interval: How often to call progress callback (in batches)

        Returns:
            List of results from all batches
        """
        if not items:
            logger.warning("No items provided for batch processing")
            return []

        if batch_size is None:
            batch_size = self.default_batch_size

        logger.info(f"Starting batch processing: {len(items)} items, batch_size={batch_size}")
        start_time = time.time()

        # Create batches
        batches = self.create_batches(items, batch_size)
        all_results = []

        # Process each batch
        for batch_idx, batch in enumerate(batches):
            batch_start_time = time.time()

            try:
                # Process the batch
                batch_results = process_function(batch)
                all_results.extend(batch_results)

                # Log batch completion
                batch_time = time.time() - batch_start_time
                logger.debug(
                    f"Completed batch {batch_idx + 1}/{len(batches)} "
                    f"({len(batch)} items) in {batch_time:.2f}s"
                )

                # Call progress callback
                if (self.progress_callback and
                    (batch_idx + 1) % progress_update_interval == 0):
                    self.progress_callback(batch_idx + 1, len(batches))

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                # Continue with other batches
                continue

        total_time = time.time() - start_time
        logger.info(
            f"Batch processing completed: {len(all_results)} results from "
            f"{len(batches)} batches in {total_time:.2f}s "
            f"({total_time/len(items):.3f}s per item)"
        )

        return all_results

    def optimize_batch_size(
        self,
        test_items: List[Any],
        process_function: Callable[[List[Any]], List[Dict]],
        test_sizes: List[int] = None,
        memory_threshold_mb: float = 1024.0
    ) -> int:
        """
        Find optimal batch size through performance testing.

        Args:
            test_items: Sample items for testing (should be representative)
            process_function: Function to test with different batch sizes
            test_sizes: List of batch sizes to test (default: [1, 2, 4, 8, 16, 32])
            memory_threshold_mb: Maximum memory usage threshold in MB

        Returns:
            Optimal batch size
        """
        if test_sizes is None:
            test_sizes = [1, 2, 4, 8, 16, 32]

        if not test_items:
            logger.warning("No test items provided for batch size optimization")
            return self.default_batch_size

        # Limit test items to avoid long optimization
        max_test_items = min(len(test_items), 20)
        test_sample = test_items[:max_test_items]

        logger.info(f"Optimizing batch size with {len(test_sample)} test items")

        best_batch_size = self.default_batch_size
        best_throughput = 0.0

        for batch_size in test_sizes:
            if batch_size > len(test_sample):
                continue

            try:
                logger.debug(f"Testing batch size {batch_size}")
                start_time = time.time()

                # Test processing with this batch size
                batches = self.create_batches(test_sample, batch_size)
                total_processed = 0

                for batch in batches:
                    batch_results = process_function(batch)
                    total_processed += len(batch_results)

                processing_time = time.time() - start_time
                throughput = total_processed / processing_time if processing_time > 0 else 0

                logger.debug(
                    f"Batch size {batch_size}: {throughput:.2f} items/sec "
                    f"({processing_time:.2f}s for {total_processed} items)"
                )

                # Update best if this is better
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size

            except Exception as e:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
                continue

        logger.info(f"Optimal batch size: {best_batch_size} ({best_throughput:.2f} items/sec)")
        return best_batch_size

    def get_processing_stats(
        self,
        results: List[Dict],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Generate processing statistics from results.

        Args:
            results: List of processing results
            start_time: Processing start time

        Returns:
            Dictionary containing processing statistics
        """
        total_time = time.time() - start_time
        total_items = len(results)

        stats = {
            'total_items': total_items,
            'total_time': total_time,
            'avg_time_per_item': total_time / max(1, total_items),
            'throughput_items_per_second': total_items / max(0.001, total_time),
            'batch_size_used': self.default_batch_size
        }

        # Calculate additional stats if results contain timing information
        if results and isinstance(results[0], dict) and 'processing_time' in results[0]:
            processing_times = [r.get('processing_time', 0) for r in results]
            if processing_times:
                stats.update({
                    'avg_processing_time': np.mean(processing_times),
                    'min_processing_time': np.min(processing_times),
                    'max_processing_time': np.max(processing_times),
                    'std_processing_time': np.std(processing_times)
                })

        return stats

    def validate_batch_results(
        self,
        batch_items: List[Any],
        batch_results: List[Dict],
        require_one_to_one: bool = True
    ) -> bool:
        """
        Validate that batch results match expected format and count.

        Args:
            batch_items: Original batch items
            batch_results: Results from processing
            require_one_to_one: Whether to require one result per input item

        Returns:
            True if validation passes, False otherwise
        """
        if not isinstance(batch_results, list):
            logger.error("Batch results must be a list")
            return False

        if require_one_to_one and len(batch_results) != len(batch_items):
            logger.error(
                f"Result count mismatch: {len(batch_results)} results "
                f"for {len(batch_items)} input items"
            )
            return False

        # Validate result structure
        for i, result in enumerate(batch_results):
            if not isinstance(result, dict):
                logger.error(f"Result {i} is not a dictionary")
                return False

        return True

    def update_batch_size(self, new_batch_size: int) -> None:
        """
        Update the default batch size.

        Args:
            new_batch_size: New batch size to use
        """
        if new_batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {new_batch_size}")

        old_size = self.default_batch_size
        self.default_batch_size = new_batch_size
        logger.info(f"Updated batch size from {old_size} to {new_batch_size}")

    def set_progress_callback(self, callback: Optional[Callable[[int, int], None]]) -> None:
        """
        Set or update the progress callback function.

        Args:
            callback: New callback function or None to disable
        """
        self.progress_callback = callback
        logger.debug(f"Progress callback {'set' if callback else 'cleared'}")