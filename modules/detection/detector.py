"""
Core YOLODetector class for container detection.

This module provides the main YOLODetector class with core detection functionality
including single image detection, batch processing, and model management.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
import torch

# Import utility modules
from .utils import ImageAnnotator, PerformanceTracker, DetectionConfig, DatabaseOperations, RetryOperations, CONTAINER_CLASSES
from .patches import apply_yolov12_patch

# Configure logging
logger = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLOv12-based object detector for containers and vehicles.

    This class wraps the ultralytics YOLO model (YOLOv12) and provides methods for
    detecting objects in single images or batch processing multiple images.
    Optimized specifically for YOLOv12 models for container detection.
    """

    def __init__(
        self,
        model_path: str = "yolov12x.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the YOLOv12 detector.

        Args:
            model_path: Path to YOLOv12 model weights
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            verbose: Whether to display verbose output
        """
        # Initialize configuration
        self.config = DetectionConfig(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device,
            verbose=verbose
        )

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing YOLOv12 detector on device: {self.device}")

        # Initialize utility classes
        self.annotator = ImageAnnotator(CONTAINER_CLASSES)
        self.performance_tracker = PerformanceTracker()
        self.database_ops = DatabaseOperations(CONTAINER_CLASSES)
        self.retry_ops = RetryOperations(self)

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model and configure it."""
        try:
            # Detect model version from filename
            model_name = Path(self.config.model_path).name
            is_yolov12 = False

            if 'yolov12' in model_name.lower():
                model_version = "YOLOv12"
                is_yolov12 = True
            elif 'yolo' in model_name.lower():
                # Extract version number if present
                import re
                match = re.search(r'yolov?(\d+)', model_name.lower())
                if match and match.group(1) == '12':
                    model_version = "YOLOv12"
                    is_yolov12 = True
                else:
                    model_version = f"YOLOv{match.group(1)}" if match else "YOLO"
            else:
                model_version = "YOLO"

            # Apply YOLOv12 patch if needed
            if is_yolov12:
                logger.info("Detected YOLOv12 model, applying AAttn compatibility patch...")
                apply_yolov12_patch()

            # Load the model
            self.model = YOLO(self.config.model_path)
            self.model.to(self.device)

            # Configure model parameters
            self.model.conf = self.config.confidence_threshold
            self.model.iou = self.config.iou_threshold

            logger.info(f"Successfully loaded {model_version} model: {self.config.model_path}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model from {self.config.model_path}: {e}")
            raise

    def detect_single_image(
        self,
        image_path: Union[str, Path],
        return_annotated: bool = False
    ) -> Dict:
        """
        Detect objects in a single image.

        Args:
            image_path: Path to the image file
            return_annotated: Whether to return annotated image

        Returns:
            Dictionary containing detection results with keys:
            - detections: supervision Detections object
            - metadata: image metadata and processing info
            - annotated_image: PIL Image with annotations (if requested)
        """
        start_time = time.time()

        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Run inference
            results = self.model(image, verbose=self.config.verbose)

            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results[0])

            # Filter for container-relevant classes only
            container_mask = np.isin(detections.class_id, list(CONTAINER_CLASSES.keys()))
            detections = detections[container_mask]

            # Calculate processing time
            processing_time = time.time() - start_time
            self.performance_tracker.record_detection_time(processing_time, len(detections))

            # Prepare metadata
            metadata = {
                "image_path": str(image_path),
                "processing_time": processing_time,
                "num_detections": len(detections),
                "image_shape": image.shape,
                "model_confidence": self.config.confidence_threshold,
                "device": self.device
            }

            result = {
                "detections": detections,
                "metadata": metadata
            }

            # Add annotated image if requested
            if return_annotated:
                result["annotated_image"] = self.annotator.annotate_image(image, detections)

            logger.info(
                f"Detected {len(detections)} objects in {processing_time:.3f}s "
                f"from {Path(image_path).name}"
            )

            return result

        except Exception as e:
            logger.error(f"Error detecting objects in {image_path}: {e}")
            raise

    def detect_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = None,
        return_annotated: bool = False
    ) -> List[Dict]:
        """
        Detect objects in multiple images using batch processing.

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process simultaneously
            return_annotated: Whether to return annotated images

        Returns:
            List of detection result dictionaries
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        logger.info(f"Starting batch detection for {len(image_paths)} images")
        start_time = time.time()

        results = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            try:
                # Load batch images
                batch_images = []
                valid_paths = []

                for path in batch_paths:
                    image = cv2.imread(str(path))
                    if image is not None:
                        batch_images.append(image)
                        valid_paths.append(path)
                    else:
                        logger.warning(f"Could not load image: {path}")

                if not batch_images:
                    continue

                # Run batch inference
                batch_results = self.model(batch_images, verbose=self.config.verbose)

                # Process each result
                for j, (result, path, image) in enumerate(zip(batch_results, valid_paths, batch_images)):
                    # Convert to supervision format
                    detections = sv.Detections.from_ultralytics(result)

                    # Filter for container-relevant classes
                    container_mask = np.isin(detections.class_id, list(CONTAINER_CLASSES.keys()))
                    detections = detections[container_mask]

                    # Prepare metadata
                    metadata = {
                        "image_path": str(path),
                        "batch_index": i + j,
                        "num_detections": len(detections),
                        "image_shape": image.shape,
                        "model_confidence": self.config.confidence_threshold,
                        "device": self.device
                    }

                    batch_result = {
                        "detections": detections,
                        "metadata": metadata
                    }

                    # Add annotated image if requested
                    if return_annotated:
                        batch_result["annotated_image"] = self.annotator.annotate_image(image, detections)

                    results.append(batch_result)

            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue

        # Record batch processing time
        total_time = time.time() - start_time
        self.performance_tracker.record_batch_time(total_time, len(results))

        logger.info(
            f"Completed batch detection: {len(results)} images in {total_time:.2f}s "
            f"({total_time/len(results):.3f}s per image)"
        )

        return results

    def update_thresholds(
        self,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> None:
        """
        Update detection thresholds.

        Args:
            confidence_threshold: New confidence threshold
            iou_threshold: New IoU threshold
        """
        self.config.update_thresholds(confidence_threshold, iou_threshold)

        # Update model parameters
        if confidence_threshold is not None:
            self.model.conf = confidence_threshold

        if iou_threshold is not None:
            self.model.iou = iou_threshold

    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics for the detector.

        Returns:
            Dictionary with performance metrics
        """
        stats = self.performance_tracker.get_performance_stats()

        # Add detector-specific information
        if "message" not in stats:
            stats.update({
                "device": self.device,
                "model_path": self.config.model_path,
                "confidence_threshold": self.config.confidence_threshold,
                "iou_threshold": self.config.iou_threshold
            })

        return stats

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
        return self.database_ops.save_detection_to_database(image_path, detections, processing_time)

    def get_detection_statistics(self, days: int = 7) -> Dict:
        """Generate detection statistics from database and performance tracker."""
        stats = self.database_ops.get_detection_statistics(days)

        # Add performance metrics from tracker
        performance_stats = self.performance_tracker.get_performance_stats()
        if "message" not in performance_stats:
            stats['performance_metrics'] = {
                'avg_detection_time': performance_stats.get('timing_stats', {}).get('mean_time', 0),
                'total_processing_time': sum(self.performance_tracker.detection_times),
                'images_processed': len(self.performance_tracker.detection_times),
                'detections_per_second': performance_stats.get('throughput_stats', {}).get('fps_mean', 0)
            }
        return stats

    def batch_process_from_database(
        self,
        limit: int = 100,
        batch_size: int = None,
        save_to_database: bool = True,
        update_container_tracking: bool = True,
        enable_caching: bool = True
    ) -> Dict[str, Any]:
        """Process unprocessed images from database in batches."""
        if batch_size is None:
            batch_size = self.config.batch_size

        return self.database_ops.batch_process_from_database(
            detector_func=self.detect_batch,
            limit=limit,
            batch_size=batch_size,
            save_to_database=save_to_database,
            update_container_tracking=update_container_tracking,
            enable_caching=enable_caching
        )

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
        return self.retry_ops.detect_with_retry(image_path, max_retries, retry_delay, save_to_db)

    def clear_cache(self):
        """Clear any cached detection results and performance history."""
        self.performance_tracker.clear_metrics()
        logger.info("Cleared detector cache and performance history")

    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded and ready for inference.

        Returns:
            True if model is loaded
        """
        return hasattr(self, 'model') and self.model is not None