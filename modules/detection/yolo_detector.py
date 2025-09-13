"""
YOLOv12 Object Detection Module for Container Analytics

This module provides YOLOv12-based object detection specifically optimized for
detecting containers and vehicles in port gate camera images.

Features:
- YOLOv12 model support using standard ultralytics package
- Pre-trained models with custom container/vehicle classes
- Batch processing for efficiency
- Configurable confidence thresholds
- GPU acceleration support
- Comprehensive logging and error handling
- Database integration with full pipeline support
- Watch mode for continuous processing
- Performance tracking and statistics
"""

import logging
import time
import queue
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
import numpy as np
import cv2
from PIL import Image
import supervision as sv
from ultralytics import YOLO
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from contextlib import contextmanager

# Import database models and session management
from modules.database.models import session_scope, Image as ImageModel, Detection as DetectionModel, Container
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def patch_yolov12_aattn():
    """
    Patch the ultralytics library to fix YOLOv12 AAttn attribute error.
    
    YOLOv12 uses separate qk and v convolutions instead of combined qkv.
    This patch fixes the 'AAttn' object has no attribute 'qkv' error.
    """
    try:
        import torch
        import torch.nn as nn
        from ultralytics.nn.modules import Conv
        import ultralytics.nn.modules.block as block_module
        
        # Check if flash attention is available
        USE_FLASH_ATTN = False
        try:
            from flash_attn import flash_attn_func
            USE_FLASH_ATTN = True
        except ImportError:
            pass
        
        class AAttnV12(nn.Module):
            """YOLOv12 Area-attention module with fixed qk/v split."""
            
            def __init__(self, dim, num_heads, area=1):
                """Initialize the YOLOv12 area-attention module."""
                super().__init__()
                self.area = area
                self.num_heads = num_heads
                self.head_dim = head_dim = dim // num_heads
                all_head_dim = head_dim * self.num_heads
                
                # YOLOv12 uses separate qk and v instead of combined qkv
                self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
                self.v = Conv(dim, all_head_dim, 1, act=False)
                self.proj = Conv(all_head_dim, dim, 1, act=False)
                
                # Positional encoding with group convolution
                self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)
            
            def forward(self, x):
                """Forward pass of the YOLOv12 area attention module."""
                B, C, H, W = x.shape
                N = H * W
                
                # Get qk and v using separate convolutions
                qk = self.qk(x).flatten(2).transpose(1, 2)
                v = self.v(x)
                pp = self.pe(v)  # Positional encoding
                v = v.flatten(2).transpose(1, 2)
                
                # Handle area-based attention
                if self.area > 1:
                    qk = qk.reshape(B * self.area, N // self.area, C * 2)
                    v = v.reshape(B * self.area, N // self.area, C)
                    B, N, _ = qk.shape
                
                # Split q and k
                q, k = qk.split([C, C], dim=2)
                
                # Check if CUDA and flash attention available
                if x.is_cuda and USE_FLASH_ATTN:
                    q = q.view(B, N, self.num_heads, self.head_dim)
                    k = k.view(B, N, self.num_heads, self.head_dim)
                    v = v.view(B, N, self.num_heads, self.head_dim)
                    
                    x = flash_attn_func(
                        q.contiguous().half(),
                        k.contiguous().half(),
                        v.contiguous().half()
                    ).to(q.dtype)
                else:
                    # Standard attention computation
                    q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
                    k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
                    v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
                    
                    # Compute attention with numerical stability
                    attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
                    max_attn = attn.max(dim=-1, keepdim=True).values
                    exp_attn = torch.exp(attn - max_attn)
                    attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
                    x = (v @ attn.transpose(-2, -1))
                    
                    x = x.permute(0, 3, 1, 2)
                
                # Reshape back if using area attention
                if self.area > 1:
                    x = x.reshape(B // self.area, N * self.area, C)
                    B, N, _ = x.shape
                
                # Reshape to spatial dimensions
                x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
                
                # Apply projection and add positional encoding
                return self.proj(x + pp)
        
        # Replace the AAttn class with our fixed version
        block_module.AAttn = AAttnV12
        logger.info("Successfully patched ultralytics AAttn for YOLOv12 compatibility")
        return True
        
    except Exception as e:
        logger.warning(f"Could not patch ultralytics AAttn: {e}")
        return False

class YOLODetector:
    """
    YOLOv12-based object detector for containers and vehicles.

    This class wraps the ultralytics YOLO model (YOLOv12) and provides methods for
    detecting objects in single images or batch processing multiple images.
    Optimized specifically for YOLOv12 models for container detection.
    """
    
    # Define class mappings for container-relevant objects
    CONTAINER_CLASSES = {
        2: 'car',           # Cars/vehicles
        3: 'motorcycle',    # Motorcycles
        5: 'bus',          # Buses
        7: 'truck',        # Trucks (most relevant for containers)
        8: 'boat',         # Boats (for port context)
    }
    
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
            model_path: Path to YOLOv12 model weights (default: yolov12x.pt)
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            verbose: Whether to display verbose output
        """
        # Ensure model path points to data/models directory
        from pathlib import Path
        model_path_obj = Path(model_path)
        
        # If it's just a filename (no directory), prepend data/models/
        if not model_path_obj.parent.name or model_path_obj.parent == Path('.'):
            self.model_path = str(Path("data/models") / model_path_obj.name)
            # Create the directory if it doesn't exist
            Path("data/models").mkdir(parents=True, exist_ok=True)
        else:
            self.model_path = model_path
            
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing YOLOv12 detector on device: {self.device}")
        
        # Load model
        self._load_model()
        
        # Performance tracking
        self.detection_times = []
        
    def _load_model(self) -> None:
        """Load the YOLO model and configure it."""
        try:
            # Detect model version from filename
            model_name = Path(self.model_path).name
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
                patch_yolov12_aattn()
            
            # Load the model
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Configure model parameters
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            
            logger.info(f"Successfully loaded {model_version} model: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {self.model_path}: {e}")
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
            results = self.model(image, verbose=self.verbose)
            
            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Filter for container-relevant classes only
            container_mask = np.isin(detections.class_id, list(self.CONTAINER_CLASSES.keys()))
            detections = detections[container_mask]
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.detection_times.append(processing_time)
            
            # Prepare metadata
            metadata = {
                "image_path": str(image_path),
                "processing_time": processing_time,
                "num_detections": len(detections),
                "image_shape": image.shape,
                "model_confidence": self.confidence_threshold,
                "device": self.device
            }
            
            result = {
                "detections": detections,
                "metadata": metadata
            }
            
            # Add annotated image if requested
            if return_annotated:
                result["annotated_image"] = self._annotate_image(image, detections)
            
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
        batch_size: int = 8,
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
                batch_results = self.model(batch_images, verbose=self.verbose)
                
                # Process each result
                for j, (result, path, image) in enumerate(zip(batch_results, valid_paths, batch_images)):
                    # Convert to supervision format
                    detections = sv.Detections.from_ultralytics(result)
                    
                    # Filter for container-relevant classes
                    container_mask = np.isin(detections.class_id, list(self.CONTAINER_CLASSES.keys()))
                    detections = detections[container_mask]
                    
                    # Prepare metadata
                    metadata = {
                        "image_path": str(path),
                        "batch_index": i + j,
                        "num_detections": len(detections),
                        "image_shape": image.shape,
                        "model_confidence": self.confidence_threshold,
                        "device": self.device
                    }
                    
                    batch_result = {
                        "detections": detections,
                        "metadata": metadata
                    }
                    
                    # Add annotated image if requested
                    if return_annotated:
                        batch_result["annotated_image"] = self._annotate_image(image, detections)
                    
                    results.append(batch_result)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        total_time = time.time() - start_time
        logger.info(
            f"Completed batch detection: {len(results)} images in {total_time:.2f}s "
            f"({total_time/len(results):.3f}s per image)"
        )
        
        return results

    def batch_process_images(
        self,
        limit: int = 100,
        batch_size: int = 8,
        save_to_database: bool = True,
        update_container_tracking: bool = True,
        enable_caching: bool = True
    ) -> Dict[str, any]:
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
                    
                    # 2. Run YOLO detection on batch
                    batch_results = self.detect_batch(
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
                            if enable_caching:
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
                                    object_type = self.CONTAINER_CLASSES.get(int(class_id), 'unknown')
                                    
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
                                    stats['errors'].append(f"Container tracking error: {e}")
                            
                            # 6. Mark image as processed
                            if save_to_database:
                                mark_image_processed(image_id)
                            
                            # Update statistics
                            stats['total_images_processed'] += 1
                            stats['total_detections'] += len(detections)
                            
                            # Track detection breakdown
                            for class_id in detections.class_id:
                                object_type = self.CONTAINER_CLASSES.get(int(class_id), 'unknown')
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
    
    def get_detection_statistics(self, days: int = 7) -> Dict[str, any]:
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
                    'total_processing_time': sum(self.detection_times),
                    'images_processed': len(self.detection_times),
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
            if self.detection_times:
                stats['performance_metrics']['avg_detection_time'] = sum(self.detection_times) / len(self.detection_times)
                if stats['performance_metrics']['total_processing_time'] > 0:
                    stats['performance_metrics']['detections_per_second'] = len(self.detection_times) / stats['performance_metrics']['total_processing_time']
            
            return stats
        
        except Exception as e:
            logger.error(f"Error generating detection statistics: {e}")
            return {'error': str(e)}

    def clear_detection_cache(self):
        """Clear any cached detection results to free memory."""
        # This would be implemented if we had a persistent cache
        # For now, just reset detection times if they get too large
        if len(self.detection_times) > 1000:
            # Keep only the last 100 measurements for performance tracking
            self.detection_times = self.detection_times[-100:]
            logger.info("Cleared old detection time measurements")
    
    def _annotate_image(self, image: np.ndarray, detections: sv.Detections) -> Image.Image:
        """
        Annotate image with detection bounding boxes and labels.
        
        Args:
            image: Input image as numpy array
            detections: Detection results
            
        Returns:
            PIL Image with annotations
        """
        # Create annotators
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Generate labels
        labels = [
            f"{self.CONTAINER_CLASSES.get(class_id, 'unknown')} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        
        # Apply annotations
        annotated_image = box_annotator.annotate(image.copy(), detections)
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)
        
        # Convert to PIL Image
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_image_rgb)
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics for the detector.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.detection_times:
            return {"message": "No detections performed yet"}
        
        times = np.array(self.detection_times)
        
        return {
            "total_detections": len(times),
            "mean_time": float(np.mean(times)),
            "median_time": float(np.median(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
            "fps_mean": 1.0 / float(np.mean(times)) if np.mean(times) > 0 else 0,
            "device": self.device,
            "model_path": self.model_path
        }
    
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
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            self.model.conf = confidence_threshold
            logger.info(f"Updated confidence threshold to {confidence_threshold}")
        
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            self.model.iou = iou_threshold
            logger.info(f"Updated IoU threshold to {iou_threshold}")
    
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
                    object_type = self.CONTAINER_CLASSES.get(int(class_id), 'unknown')
                    
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
                result = self.detect_single_image(image_path)
                
                # Save to database if requested
                if save_to_db and result:
                    image_id = self.save_detection_to_database(
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
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    logger.info(f"Retrying detection for {image_path} in {retry_delay * (2 ** attempt):.1f}s...")
        
        logger.error(f"All {max_retries + 1} detection attempts failed for {image_path}: {last_error}")
        return None


class ImageProcessingQueue:
    """Thread-safe queue for managing image processing tasks."""
    
    def __init__(self, maxsize: int = 100):
        self.queue = queue.Queue(maxsize=maxsize)
        self.processed_files = set()
        self._lock = threading.Lock()
    
    def add_image(self, image_path: Path) -> bool:
        """Add image to processing queue if not already processed."""
        with self._lock:
            image_str = str(image_path)
            if image_str in self.processed_files:
                return False
            
            try:
                self.queue.put_nowait(image_path)
                logger.debug(f"Added {image_path.name} to processing queue")
                return True
            except queue.Full:
                logger.warning("Processing queue is full, skipping image")
                return False
    
    def get_image(self, timeout: Optional[float] = None) -> Optional[Path]:
        """Get next image from queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def mark_processed(self, image_path: Path):
        """Mark image as processed."""
        with self._lock:
            self.processed_files.add(str(image_path))
            self.queue.task_done()
    
    def clear_processed_history(self):
        """Clear the processed files history."""
        with self._lock:
            self.processed_files.clear()
    
    def qsize(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()


class ImageFileHandler(FileSystemEventHandler):
    """File system event handler for new images."""
    
    def __init__(self, processing_queue: ImageProcessingQueue, supported_extensions: set = None):
        self.processing_queue = processing_queue
        self.supported_extensions = supported_extensions or {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        logger.info(f"Image file handler initialized with extensions: {self.supported_extensions}")
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Check if it's a supported image file
        if file_path.suffix.lower() in self.supported_extensions:
            logger.info(f"New image detected: {file_path.name}")
            
            # Wait a bit to ensure file is fully written
            time.sleep(0.5)
            
            # Add to processing queue
            if self.processing_queue.add_image(file_path):
                logger.info(f"Queued {file_path.name} for processing")
            else:
                logger.debug(f"Image {file_path.name} already processed or queue full")


class YOLOWatchMode:
    """
    Watch mode functionality for continuous image processing.
    
    Monitors a directory for new images and processes them automatically
    using YOLO detection with database persistence.
    """
    
    def __init__(
        self,
        detector: YOLODetector,
        watch_directory: Union[str, Path] = "data/images",
        batch_size: int = 4,
        max_workers: int = 2,
        queue_size: int = 100,
        process_existing: bool = False
    ):
        self.detector = detector
        self.watch_directory = Path(watch_directory)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.process_existing = process_existing
        
        # Initialize processing queue
        self.processing_queue = ImageProcessingQueue(maxsize=queue_size)
        
        # Initialize file system observer
        self.observer = Observer()
        self.file_handler = ImageFileHandler(self.processing_queue)
        
        # Processing control
        self.is_running = False
        self.worker_threads = []
        self.stats = {
            'images_processed': 0,
            'images_failed': 0,
            'total_detections': 0,
            'start_time': None,
            'last_processed': None
        }
        
        logger.info(f"YOLO Watch Mode initialized for directory: {self.watch_directory}")
    
    def _process_worker(self):
        """Worker thread for processing images from queue."""
        worker_id = threading.current_thread().name
        logger.info(f"Processing worker {worker_id} started")
        
        batch_images = []
        
        while self.is_running:
            try:
                # Get image from queue with timeout
                image_path = self.processing_queue.get_image(timeout=1.0)
                
                if image_path is None:
                    # Process any remaining batch images
                    if batch_images:
                        self._process_batch(batch_images, worker_id)
                        batch_images = []
                    continue
                
                batch_images.append(image_path)
                
                # Process batch when full or no more images in queue
                if (len(batch_images) >= self.batch_size or 
                    self.processing_queue.qsize() == 0):
                    self._process_batch(batch_images, worker_id)
                    batch_images = []
                
            except Exception as e:
                logger.error(f"Error in processing worker {worker_id}: {e}")
                # Mark any images in current batch as processed
                for img_path in batch_images:
                    self.processing_queue.mark_processed(img_path)
                batch_images = []
        
        # Process any remaining images
        if batch_images:
            self._process_batch(batch_images, worker_id)
        
        logger.info(f"Processing worker {worker_id} stopped")
    
    def _process_batch(self, image_paths: List[Path], worker_id: str):
        """Process a batch of images."""
        if not image_paths:
            return
        
        logger.info(f"Worker {worker_id} processing batch of {len(image_paths)} images")
        
        for image_path in image_paths:
            try:
                # Process image with retry logic
                result = self.detector.detect_with_retry(
                    image_path, 
                    max_retries=2, 
                    save_to_db=True
                )
                
                if result:
                    num_detections = result['metadata']['num_detections']
                    self.stats['images_processed'] += 1
                    self.stats['total_detections'] += num_detections
                    self.stats['last_processed'] = time.time()
                    
                    logger.info(
                        f"Worker {worker_id} processed {image_path.name}: "
                        f"{num_detections} detections in {result['metadata']['processing_time']:.2f}s"
                    )
                else:
                    self.stats['images_failed'] += 1
                    logger.error(f"Worker {worker_id} failed to process {image_path.name}")
                
            except Exception as e:
                self.stats['images_failed'] += 1
                logger.error(f"Worker {worker_id} error processing {image_path.name}: {e}")
            
            finally:
                # Mark image as processed
                self.processing_queue.mark_processed(image_path)
    
    def _process_existing_images(self):
        """Process existing images in the watch directory."""
        if not self.process_existing:
            return
        
        logger.info("Processing existing images in watch directory...")
        
        existing_images = []
        for ext in self.file_handler.supported_extensions:
            existing_images.extend(self.watch_directory.glob(f"*{ext}"))
            existing_images.extend(self.watch_directory.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(existing_images)} existing images")
        
        for image_path in existing_images:
            if self.processing_queue.add_image(image_path):
                logger.debug(f"Queued existing image: {image_path.name}")
    
    def start(self, process_existing: bool = None):
        """Start watch mode processing."""
        if self.is_running:
            logger.warning("Watch mode is already running")
            return
        
        # Ensure watch directory exists
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting YOLO watch mode on {self.watch_directory}")
        
        # Initialize stats
        self.stats['start_time'] = time.time()
        self.stats['images_processed'] = 0
        self.stats['images_failed'] = 0
        self.stats['total_detections'] = 0
        
        # Set running flag
        self.is_running = True
        
        # Process existing images if requested
        if process_existing or (process_existing is None and self.process_existing):
            self._process_existing_images()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._process_worker, 
                name=f"YOLOWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Start file system observer
        self.observer.schedule(
            self.file_handler, 
            str(self.watch_directory), 
            recursive=True
        )
        self.observer.start()
        
        logger.info(f"Watch mode started with {self.max_workers} workers monitoring {self.watch_directory}")
    
    def stop(self):
        """Stop watch mode processing."""
        if not self.is_running:
            logger.warning("Watch mode is not running")
            return
        
        logger.info("Stopping YOLO watch mode...")
        
        # Stop file observer
        self.observer.stop()
        
        # Set running flag to False
        self.is_running = False
        
        # Wait for worker threads to finish
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        # Stop observer
        self.observer.join(timeout=5.0)
        
        # Clear worker threads
        self.worker_threads.clear()
        
        # Print final stats
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        logger.info(
            f"Watch mode stopped. Stats: {self.stats['images_processed']} processed, "
            f"{self.stats['images_failed']} failed, "
            f"{self.stats['total_detections']} total detections "
            f"in {runtime:.1f}s"
        )
    
    def get_stats(self) -> Dict:
        """Get current processing statistics."""
        runtime = 0
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
        
        return {
            'is_running': self.is_running,
            'runtime_seconds': runtime,
            'images_processed': self.stats['images_processed'],
            'images_failed': self.stats['images_failed'],
            'total_detections': self.stats['total_detections'],
            'queue_size': self.processing_queue.qsize(),
            'success_rate': (
                self.stats['images_processed'] / 
                max(1, self.stats['images_processed'] + self.stats['images_failed'])
            ) * 100,
            'processing_rate': (
                self.stats['images_processed'] / max(1, runtime / 60)
            ) if runtime > 0 else 0,
            'last_processed': self.stats['last_processed']
        }
    
    @contextmanager
    def running_context(self, process_existing: bool = None):
        """Context manager for watch mode."""
        try:
            self.start(process_existing=process_existing)
            yield self
        finally:
            self.stop()


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="YOLO Container Detector")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--batch", type=str, help="Path to directory with images")
    parser.add_argument("--model", type=str, default="yolov12x.pt", help="Model path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, help="Device (cpu/cuda)")
    parser.add_argument("--output", type=str, help="Output directory for annotated images")
    parser.add_argument("--watch", action="store_true", help="Watch mode for continuous detection")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device
    )
    
    if args.image:
        # Single image detection
        result = detector.detect_single_image(args.image, return_annotated=bool(args.output))
        
        print(f"Detections: {len(result['detections'])}")
        print(f"Processing time: {result['metadata']['processing_time']:.3f}s")
        
        # Save annotated image if output specified
        if args.output and "annotated_image" in result:
            output_path = Path(args.output) / f"annotated_{Path(args.image).name}"
            result["annotated_image"].save(output_path)
            print(f"Saved annotated image: {output_path}")
    
    elif args.batch:
        # Batch processing
        image_dir = Path(args.batch)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            sys.exit(1)
        
        results = detector.detect_batch(image_paths, return_annotated=bool(args.output))
        
        print(f"Processed {len(results)} images")
        print("Performance stats:", detector.get_performance_stats())
        
        # Save annotated images if output specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            
            for result in results:
                if "annotated_image" in result:
                    image_name = Path(result["metadata"]["image_path"]).name
                    output_path = output_dir / f"annotated_{image_name}"
                    result["annotated_image"].save(output_path)
            
            print(f"Saved annotated images to {output_dir}")
    
    elif args.watch:
        # Watch mode - continuous processing
        watch_dir = Path("data/images")
        if not watch_dir.exists():
            watch_dir.mkdir(parents=True)
            print(f"Created watch directory: {watch_dir}")
        
        print(f"Starting watch mode on directory: {watch_dir}")
        print("Press Ctrl+C to stop...")
        
        # Initialize watch mode
        watch_mode = YOLOWatchMode(
            detector=detector,
            watch_directory=watch_dir,
            batch_size=4,
            max_workers=2,
            process_existing=True
        )
        
        try:
            with watch_mode.running_context(process_existing=True):
                # Print periodic stats
                while True:
                    try:
                        time.sleep(10)  # Update every 10 seconds
                        stats = watch_mode.get_stats()
                        print(f"\n--- Watch Mode Stats ---")
                        print(f"Running: {stats['is_running']}")
                        print(f"Runtime: {stats['runtime_seconds']:.1f}s")
                        print(f"Images Processed: {stats['images_processed']}")
                        print(f"Images Failed: {stats['images_failed']}")
                        print(f"Total Detections: {stats['total_detections']}")
                        print(f"Queue Size: {stats['queue_size']}")
                        print(f"Success Rate: {stats['success_rate']:.1f}%")
                        print(f"Processing Rate: {stats['processing_rate']:.1f} images/min")
                        print("Press Ctrl+C to stop...")
                    except KeyboardInterrupt:
                        break
        
        except KeyboardInterrupt:
            print("\nShutting down watch mode...")
        
        final_stats = watch_mode.get_stats()
        print(f"\nFinal stats: {final_stats['images_processed']} processed, "
              f"{final_stats['total_detections']} total detections")
    
    else:
        parser.print_help()