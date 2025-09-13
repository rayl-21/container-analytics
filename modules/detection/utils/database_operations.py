"""
Database operations for YOLO detection results.

This module provides functionality for saving detection results to the database,
managing image records, and performing batch database operations.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import supervision as sv

# Import database models and session management
from modules.database.models import session_scope, Image as ImageModel, Detection as DetectionModel
from modules.database.queries import (
    get_unprocessed_images,
    insert_detection,
    mark_image_processed,
    get_detection_summary,
    get_container_statistics,
    get_throughput_data,
    get_recent_detections
)
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DatabaseOperations:
    """
    Handles database operations for YOLO detection results.

    This class provides methods for saving detection results, managing batch
    processing operations, and retrieving detection statistics from the database.
    """

    def __init__(self, container_classes: Dict[int, str]):
        """
        Initialize database operations.

        Args:
            container_classes: Dictionary mapping class IDs to class names
        """
        self.container_classes = container_classes

    def save_detection_to_database(
        self,
        image_path: Union[str, Path],
        detections: sv.Detections,
        processing_time: float
    ) -> Optional[int]:
        """
        Save detection results to database.

        Args:
            image_path: Path to the processed image
            detections: Detection results from YOLO
            processing_time: Time taken to process the image

        Returns:
            Image ID if successfully saved, None otherwise
        """
        try:
            image_path = Path(image_path)

            with session_scope() as session:
                # Check if image already exists in database
                existing_image = session.query(ImageModel).filter(
                    ImageModel.filepath == str(image_path)
                ).first()

                if existing_image:
                    image_record = existing_image
                    logger.info(f"Using existing image record: {image_record.id}")
                else:
                    # Create new image record
                    file_size = image_path.stat().st_size if image_path.exists() else 0
                    camera_id = self._extract_camera_id_from_path(image_path)

                    image_record = ImageModel(
                        filepath=str(image_path),
                        camera_id=camera_id,
                        processed=True,
                        file_size=file_size
                    )
                    session.add(image_record)
                    session.flush()  # Get the ID
                    logger.info(f"Created new image record: {image_record.id}")

                # Save detection records
                detection_count = 0
                for i in range(len(detections)):
                    bbox = detections.xyxy[i]
                    confidence = detections.confidence[i]
                    class_id = detections.class_id[i]

                    # Map class ID to object type
                    object_type = self.container_classes.get(int(class_id), 'unknown')

                    detection_record = DetectionModel(
                        image_id=image_record.id,
                        object_type=object_type,
                        confidence=float(confidence),
                        bbox_x=float(bbox[0]),
                        bbox_y=float(bbox[1]),
                        bbox_width=float(bbox[2] - bbox[0]),
                        bbox_height=float(bbox[3] - bbox[1])
                    )
                    session.add(detection_record)
                    detection_count += 1

                logger.info(f"Saved {detection_count} detections for image {image_record.id}")
                return image_record.id

        except Exception as e:
            logger.error(f"Failed to save detection to database: {e}")
            return None

    def _extract_camera_id_from_path(self, image_path: Path) -> str:
        """Extract camera ID from image path."""
        # Look for common camera ID patterns in the filename
        filename = image_path.name.lower()

        if 'in_gate' in filename or 'in-gate' in filename:
            return 'in_gate'
        elif 'gate' in filename:
            return 'gate'
        else:
            # Default to unknown or use parent directory name
            return image_path.parent.name if image_path.parent.name != 'images' else 'unknown'

    def get_detection_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate detection statistics and aggregation data.

        Args:
            days: Number of days to look back for statistics

        Returns:
            Dictionary containing comprehensive detection statistics
        """
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
                'recent_activity': {}
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

            return stats

        except Exception as e:
            logger.error(f"Error generating detection statistics: {e}")
            return {'error': str(e)}

    def batch_process_from_database(
        self,
        detector_func,
        limit: int = 100,
        batch_size: int = 8,
        save_to_database: bool = True,
        update_container_tracking: bool = True,
        enable_caching: bool = True
    ) -> Dict[str, Any]:
        """
        Process unprocessed images from database in batches.

        Args:
            detector_func: Function to perform detection (should be detector.detect_batch)
            limit: Maximum number of images to process
            batch_size: Number of images to process simultaneously
            save_to_database: Whether to persist results to database
            update_container_tracking: Whether to update container tracking records
            enable_caching: Whether to cache detection results

        Returns:
            Dictionary containing processing statistics and results
        """
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
            # Load unprocessed images from database
            unprocessed_images = get_unprocessed_images(limit=limit)
            stats['total_images_found'] = len(unprocessed_images)

            if not unprocessed_images:
                logger.info("No unprocessed images found in database")
                return stats

            logger.info(f"Found {len(unprocessed_images)} unprocessed images")

            # Group images for batch processing
            image_batches = []
            for i in range(0, len(unprocessed_images), batch_size):
                batch = unprocessed_images[i:i + batch_size]
                image_batches.append(batch)

            # Process each batch
            for batch_idx, image_batch in enumerate(image_batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(image_batches)} ({len(image_batch)} images)")

                try:
                    # Prepare batch data
                    batch_paths = []
                    batch_metadata = []
                    valid_images = []

                    for img_data in image_batch:
                        image_path = Path(img_data['filepath'])

                        # Check cache first
                        if enable_caching and str(image_path) in detection_cache:
                            logger.debug(f"Using cached result for {image_path.name}")
                            stats['cached_results'] += 1
                            continue

                        # Verify image exists and is readable
                        if image_path.exists():
                            batch_paths.append(image_path)
                            batch_metadata.append(img_data)
                            valid_images.append(img_data)
                        else:
                            logger.warning(f"Image file not found: {image_path}")
                            stats['errors'].append(f"File not found: {image_path}")

                    if not batch_paths:
                        continue

                    # Run detection on batch
                    batch_results = detector_func(
                        image_paths=batch_paths,
                        batch_size=len(batch_paths),
                        return_annotated=False
                    )

                    # Process results for each image
                    for result, img_data in zip(batch_results, valid_images):
                        try:
                            detections = result['detections']
                            metadata = result['metadata']
                            image_id = img_data['id']
                            image_path = Path(img_data['filepath'])

                            # Cache result if enabled
                            if enable_caching:
                                detection_cache[str(image_path)] = {
                                    'detections': detections,
                                    'metadata': metadata,
                                    'timestamp': time.time()
                                }

                            # Save detection records to database
                            if save_to_database and len(detections) > 0:
                                detection_ids = []
                                for i in range(len(detections)):
                                    bbox = detections.xyxy[i]
                                    confidence = detections.confidence[i]
                                    class_id = detections.class_id[i]

                                    # Map class ID to object type
                                    object_type = self.container_classes.get(int(class_id), 'unknown')

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
                                    stats['database_saves'] += 1

                                logger.debug(f"Saved {len(detection_ids)} detections for image {image_id}")

                            # Mark image as processed
                            if save_to_database:
                                mark_image_processed(image_id)

                            # Update statistics
                            stats['total_images_processed'] += 1
                            stats['total_detections'] += len(detections)

                            # Track detection breakdown
                            for class_id in detections.class_id:
                                object_type = self.container_classes.get(int(class_id), 'unknown')
                                stats['detection_breakdown'][object_type] = stats['detection_breakdown'].get(object_type, 0) + 1

                        except Exception as e:
                            logger.error(f"Error processing image {img_data['id']}: {e}")
                            stats['errors'].append(f"Image {img_data['id']}: {e}")
                            continue

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

            return stats

        except Exception as e:
            logger.error(f"Fatal error in batch processing: {e}")
            stats['errors'].append(f"Fatal error: {e}")
            stats['processing_time'] = time.time() - start_time
            raise