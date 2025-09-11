"""
YOLOv12 Object Detection Module for Container Analytics

This module provides YOLOv12-based object detection specifically optimized for
detecting containers and vehicles in port gate camera images.

Features:
- Pre-trained YOLOv12 model with custom container/vehicle classes
- Batch processing for efficiency
- Configurable confidence thresholds
- GPU acceleration support
- Comprehensive logging and error handling
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
from modules.database.models import session_scope, Image as ImageModel, Detection as DetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLOv12-based object detector for containers and vehicles.
    
    This class wraps the ultralytics YOLO model and provides methods for
    detecting objects in single images or batch processing multiple images.
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
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights (default: yolov12x.pt)
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
        """Load the YOLOv12 model and configure it."""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Configure model parameters
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            
            logger.info(f"Successfully loaded YOLOv12 model: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv12 model: {e}")
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
        elif 'out_gate' in filename or 'out-gate' in filename:
            return 'out_gate'
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