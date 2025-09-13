"""
Detection Pipeline Module

This module provides the complete detection pipeline integration, including
batch processing of images from the database, database persistence, and
container tracking updates.

Classes:
    DetectionPipeline: Full pipeline integration for YOLO detection
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..yolo_detector import YOLODetector

from .batch import BatchProcessor
from .queue import ImageProcessingQueue

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Complete detection pipeline for processing images with YOLO detection.

    This class implements the full detection pipeline including:
    1. Loading unprocessed images from database
    2. Running YOLO detection in batches
    3. Saving detection records to database
    4. Updating container tracking data
    5. Generating detection metrics and statistics

    Attributes:
        detector: YOLO detector instance
        batch_processor: Batch processing handler
        processing_queue: Image processing queue
    """

    def __init__(
        self,
        detector: 'YOLODetector',
        default_batch_size: int = 8,
        queue_size: int = 100
    ):
        """
        Initialize the detection pipeline.

        Args:
            detector: YOLO detector instance
            default_batch_size: Default batch size for processing
            queue_size: Maximum queue size for processing
        """
        self.detector = detector
        self.batch_processor = BatchProcessor(default_batch_size=default_batch_size)
        self.processing_queue = ImageProcessingQueue(maxsize=queue_size)

        logger.info(f"DetectionPipeline initialized with batch_size={default_batch_size}, queue_size={queue_size}")

    def batch_process_images(
        self,
        limit: int = 100,
        batch_size: int = 8,
        save_to_database: bool = True,
        update_container_tracking: bool = True,
        enable_caching: bool = True
    ) -> Dict[str, Any]:
        """
        Process unprocessed images from database in batches with full pipeline integration.

        This method implements the complete detection pipeline:
        1. Load unprocessed images from database using queries.py functions
        2. Run YOLO detection in batches for efficiency
        3. Save Detection records with bbox and confidence to database
        4. Update Container tracking data (if OCR available)
        5. Mark images as processed in database
        6. Generate detection metrics and statistics
        7. Cache results for performance

        Args:
            limit: Maximum number of images to process
            batch_size: Number of images to process simultaneously
            save_to_database: Whether to persist results to database
            update_container_tracking: Whether to update container tracking records
            enable_caching: Whether to cache detection results

        Returns:
            Dictionary containing processing statistics and results
        """
        from modules.database.queries import (
            get_unprocessed_images,
            insert_detection,
            mark_image_processed,
            update_container_tracking as update_container
        )

        logger.info(f"Starting batch processing: limit={limit}, batch_size={batch_size}")
        start_time = time.time()

        # Initialize statistics
        stats = {
            'total_images_requested': limit,
            'total_images_found': 0,
            'total_images_processed': 0,
            'total_detections': 0,
            'processing_time': 0,
            'avg_time_per_image': 0,
            'detection_breakdown': {},
            'errors': [],
            'cached_results': 0,
            'database_saves': 0
        }

        # Cache for detection results (if enabled)
        detection_cache = {} if enable_caching else None

        try:
            # 1. Load unprocessed images from database
            unprocessed_images = get_unprocessed_images(limit=limit)
            stats['total_images_found'] = len(unprocessed_images)

            if not unprocessed_images:
                logger.info("No unprocessed images found in database")
                return stats

            logger.info(f"Found {len(unprocessed_images)} unprocessed images")

            # Group images for batch processing
            image_batches = self.batch_processor.create_batches(unprocessed_images, batch_size)

            # Process each batch
            for batch_idx, image_batch in enumerate(image_batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(image_batches)} ({len(image_batch)} images)")

                try:
                    batch_stats = self._process_image_batch(
                        image_batch,
                        stats,
                        detection_cache,
                        save_to_database,
                        update_container_tracking,
                        enable_caching
                    )

                    # Update overall stats
                    for key, value in batch_stats.items():
                        if key in ['total_images_processed', 'total_detections', 'cached_results', 'database_saves']:
                            stats[key] += value
                        elif key == 'detection_breakdown':
                            for obj_type, count in value.items():
                                stats['detection_breakdown'][obj_type] = (
                                    stats['detection_breakdown'].get(obj_type, 0) + count
                                )
                        elif key == 'errors':
                            stats['errors'].extend(value)

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                    stats['errors'].append(f"Batch {batch_idx + 1}: {e}")
                    continue

            # Calculate final statistics
            stats['processing_time'] = time.time() - start_time
            if stats['total_images_processed'] > 0:
                stats['avg_time_per_image'] = stats['processing_time'] / stats['total_images_processed']

            # Log completion summary
            logger.info(
                f"Batch processing completed: {stats['total_images_processed']}/{stats['total_images_found']} images, "
                f"{stats['total_detections']} detections, {stats['processing_time']:.2f}s total "
                f"({stats['avg_time_per_image']:.3f}s per image)"
            )

            # Log detection breakdown
            if stats['detection_breakdown']:
                breakdown_str = ", ".join([f"{obj}: {count}" for obj, count in stats['detection_breakdown'].items()])
                logger.info(f"Detection breakdown: {breakdown_str}")

            if stats['errors']:
                logger.warning(f"Processing completed with {len(stats['errors'])} errors")

            return stats

        except Exception as e:
            logger.error(f"Fatal error in batch processing: {e}")
            stats['errors'].append(f"Fatal error: {e}")
            stats['processing_time'] = time.time() - start_time
            raise

    def _process_image_batch(
        self,
        image_batch: List[Dict],
        stats: Dict[str, Any],
        detection_cache: Optional[Dict],
        save_to_database: bool,
        update_container_tracking: bool,
        enable_caching: bool
    ) -> Dict[str, Any]:
        """
        Process a single batch of images.

        Args:
            image_batch: Batch of image data from database
            stats: Current statistics dictionary
            detection_cache: Cache for detection results
            save_to_database: Whether to save to database
            update_container_tracking: Whether to update container tracking
            enable_caching: Whether to use caching

        Returns:
            Batch processing statistics
        """
        from modules.database.queries import (
            insert_detection,
            mark_image_processed,
            update_container_tracking as update_container
        )

        batch_stats = {
            'total_images_processed': 0,
            'total_detections': 0,
            'cached_results': 0,
            'database_saves': 0,
            'detection_breakdown': {},
            'errors': []
        }

        # Prepare batch data
        batch_paths = []
        batch_metadata = []
        valid_images = []

        for img_data in image_batch:
            image_path = Path(img_data['filepath'])

            # Check cache first
            if enable_caching and detection_cache and str(image_path) in detection_cache:
                logger.debug(f"Using cached result for {image_path.name}")
                batch_stats['cached_results'] += 1
                continue

            # Verify image exists and is readable
            if image_path.exists():
                batch_paths.append(image_path)
                batch_metadata.append(img_data)
                valid_images.append(img_data)
            else:
                logger.warning(f"Image file not found: {image_path}")
                batch_stats['errors'].append(f"File not found: {image_path}")

        if not batch_paths:
            return batch_stats

        # 2. Run YOLO detection on batch
        batch_results = self.detector.detect_batch(
            image_paths=batch_paths,
            batch_size=len(batch_paths),
            return_annotated=False
        )

        # 3. Process results for each image
        for result, img_data in zip(batch_results, valid_images):
            try:
                detections = result['detections']
                metadata = result['metadata']
                image_id = img_data['id']
                image_path = Path(img_data['filepath'])

                # Cache result if enabled
                if enable_caching and detection_cache is not None:
                    detection_cache[str(image_path)] = {
                        'detections': detections,
                        'metadata': metadata,
                        'timestamp': time.time()
                    }

                # 4. Save detection records to database
                if save_to_database and len(detections) > 0:
                    detection_ids = []
                    for i in range(len(detections)):
                        bbox = detections.xyxy[i]
                        confidence = detections.confidence[i]
                        class_id = detections.class_id[i]

                        # Map class ID to object type
                        object_type = self.detector.CONTAINER_CLASSES.get(int(class_id), 'unknown')

                        # Prepare bbox dictionary
                        bbox_dict = {
                            'x': float(bbox[0]),
                            'y': float(bbox[1]),
                            'width': float(bbox[2] - bbox[0]),
                            'height': float(bbox[3] - bbox[1])
                        }

                        # Save detection to database
                        detection_id = insert_detection(
                            image_id=image_id,
                            object_type=object_type,
                            confidence=float(confidence),
                            bbox=bbox_dict,
                            tracking_id=getattr(detections, 'tracker_id', [None])[i] if hasattr(detections, 'tracker_id') else None
                        )
                        detection_ids.append(detection_id)
                        batch_stats['database_saves'] += 1

                    logger.debug(f"Saved {len(detection_ids)} detections for image {image_id}")

                # 5. Update container tracking (if OCR module available)
                if update_container_tracking and len(detections) > 0:
                    try:
                        # This is where OCR integration would happen
                        # For now, we'll skip container tracking without OCR
                        # In a full implementation, you'd call:
                        # container_numbers = extract_container_numbers(image_path, detections)
                        # for container_number in container_numbers:
                        #     update_container(container_number, img_data['timestamp'],
                        #                    img_data['camera_id'], max_confidence)
                        pass
                    except Exception as e:
                        logger.warning(f"Container tracking update failed: {e}")
                        batch_stats['errors'].append(f"Container tracking error: {e}")

                # 6. Mark image as processed
                if save_to_database:
                    mark_image_processed(image_id)

                # Update statistics
                batch_stats['total_images_processed'] += 1
                batch_stats['total_detections'] += len(detections)

                # Track detection breakdown
                for class_id in detections.class_id:
                    object_type = self.detector.CONTAINER_CLASSES.get(int(class_id), 'unknown')
                    batch_stats['detection_breakdown'][object_type] = (
                        batch_stats['detection_breakdown'].get(object_type, 0) + 1
                    )

            except Exception as e:
                logger.error(f"Error processing image {img_data['id']}: {e}")
                batch_stats['errors'].append(f"Image {img_data['id']}: {e}")
                continue

        return batch_stats

    def get_detection_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate detection statistics and aggregation data.

        Args:
            days: Number of days to look back for statistics

        Returns:
            Dictionary containing comprehensive detection statistics
        """
        from modules.database.queries import (
            get_detection_summary,
            get_container_statistics,
            get_throughput_data,
            get_recent_detections
        )
        from datetime import datetime, timedelta

        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            stats = {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days
                },
                'detection_summary': {},
                'container_stats': {},
                'throughput_data': {},
                'recent_activity': {},
                'performance_metrics': {
                    'avg_detection_time': 0,
                    'total_processing_time': sum(self.detector.detection_times),
                    'images_processed': len(self.detector.detection_times),
                    'detections_per_second': 0
                }
            }

            # Get detection summary
            try:
                stats['detection_summary'] = get_detection_summary(start_date, end_date)
            except Exception as e:
                logger.warning(f"Could not get detection summary: {e}")
                stats['detection_summary'] = {}

            # Get container statistics
            try:
                stats['container_stats'] = get_container_statistics(start_date, end_date)
            except Exception as e:
                logger.warning(f"Could not get container statistics: {e}")
                stats['container_stats'] = {}

            # Get throughput data
            try:
                stats['throughput_data'] = get_throughput_data(start_date, end_date)
            except Exception as e:
                logger.warning(f"Could not get throughput data: {e}")
                stats['throughput_data'] = {}

            # Get recent activity
            try:
                stats['recent_activity'] = get_recent_detections(limit=50)
            except Exception as e:
                logger.warning(f"Could not get recent detections: {e}")
                stats['recent_activity'] = {}

            # Calculate performance metrics
            if self.detector.detection_times:
                stats['performance_metrics']['avg_detection_time'] = (
                    sum(self.detector.detection_times) / len(self.detector.detection_times)
                )
                if stats['performance_metrics']['total_processing_time'] > 0:
                    stats['performance_metrics']['detections_per_second'] = (
                        len(self.detector.detection_times) / stats['performance_metrics']['total_processing_time']
                    )

            return stats

        except Exception as e:
            logger.error(f"Error generating detection statistics: {e}")
            return {'error': str(e)}

    def clear_detection_cache(self) -> None:
        """Clear any cached detection results to free memory."""
        # This would be implemented if we had a persistent cache
        # For now, just reset detection times if they get too large
        if len(self.detector.detection_times) > 1000:
            # Keep only the last 100 measurements for performance tracking
            self.detector.detection_times = self.detector.detection_times[-100:]
            logger.info("Cleared old detection time measurements")

        # Clear queue processed history
        self.processing_queue.clear_processed_history()

    def update_batch_size(self, new_batch_size: int) -> None:
        """
        Update the batch size for processing.

        Args:
            new_batch_size: New batch size to use
        """
        self.batch_processor.update_batch_size(new_batch_size)
        logger.info(f"Updated pipeline batch size to {new_batch_size}")

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get current pipeline statistics.

        Returns:
            Dictionary with pipeline status and statistics
        """
        return {
            'batch_size': self.batch_processor.default_batch_size,
            'queue_stats': self.processing_queue.get_stats(),
            'detector_performance': self.detector.get_performance_stats()
        }