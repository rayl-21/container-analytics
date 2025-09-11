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
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
import supervision as sv
from ultralytics import YOLO
import torch

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
        # Watch mode - placeholder for continuous processing
        print("Watch mode - would monitor directory for new images")
        print("This would be integrated with the scheduler module")
    
    else:
        parser.print_help()