"""
Native YOLOv12 detector implementation using sunsmarterjie's YOLOv12.

This implementation uses the YOLOv12 model with the correct AAttn implementation
that has separate qk and v layers instead of combined qkv.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict, Any, Optional, Tuple
import time

# Add YOLOv12 implementation to path BEFORE any ultralytics import
yolov12_path = Path(__file__).parent / "yolov12_impl"
if str(yolov12_path) not in sys.path:
    sys.path.insert(0, str(yolov12_path))

# Remove system ultralytics from path to ensure we use YOLOv12's version
sys.modules.pop('ultralytics', None)
sys.modules.pop('ultralytics.nn', None)
sys.modules.pop('ultralytics.nn.modules', None)
sys.modules.pop('ultralytics.nn.modules.block', None)

# Now import from YOLOv12's modified ultralytics
from ultralytics import YOLO
from ultralytics.utils import ops


class YOLOv12NativeDetector:
    """Native YOLOv12 detector using sunsmarterjie implementation."""
    
    def __init__(
        self,
        model_path: str = "data/models/yolov12x.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        verbose: bool = False
    ):
        """
        Initialize YOLOv12 detector.
        
        Args:
            model_path: Path to YOLOv12 model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run model on ('cpu' or 'cuda')
            verbose: Whether to print verbose output
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.verbose = verbose
        
        # Load model using YOLOv12's ultralytics
        if self.verbose:
            print(f"Loading YOLOv12 model from {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        
        # COCO class names
        self.class_names = self.model.names if hasattr(self.model, 'names') else {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            # Add more COCO classes as needed
        }
        
        # Classes we're interested in for container/truck detection
        self.target_classes = ['truck', 'container', 'car', 'bus', 'trailer']
        
    def detect_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Run detection on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # Load image
        image_path = Path(image_path)
        if not image_path.exists():
            return {
                'success': False,
                'error': f"Image not found: {image_path}",
                'detections': [],
                'processing_time': 0
            }
        
        try:
            # Run inference
            results = self.model(
                str(image_path),
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Process results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get box coordinates
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        # Get confidence and class
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        # Get class name
                        class_name = self.class_names.get(cls, f"class_{cls}")
                        
                        # Only keep detections of interest
                        if any(target in class_name.lower() for target in self.target_classes):
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': conf,
                                'class': cls,
                                'class_name': class_name,
                                'object_type': class_name
                            })
            
            processing_time = time.time() - start_time
            
            if self.verbose:
                print(f"Processed {image_path.name}: {len(detections)} detections in {processing_time:.2f}s")
            
            return {
                'success': True,
                'image_path': str(image_path),
                'detections': detections,
                'num_detections': len(detections),
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing {image_path}: {str(e)}"
            if self.verbose:
                print(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'detections': [],
                'processing_time': processing_time
            }
    
    def detect_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run detection on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of detection results
        """
        results = []
        for image_path in image_paths:
            result = self.detect_single_image(image_path)
            results.append(result)
        return results
    
    def save_detection_to_database(
        self,
        image_path: str,
        save_visualization: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Detect objects and save to database.
        
        Args:
            image_path: Path to image
            save_visualization: Whether to save visualization image
            
        Returns:
            Detection result dictionary
        """
        # Run detection
        result = self.detect_single_image(image_path)
        
        if not result['success']:
            return None
        
        # Import database models
        from modules.database.models import session_scope, Detection, Image as DBImage
        from datetime import datetime
        
        # Save to database
        with session_scope() as session:
            # Find image in database
            image_record = session.query(DBImage).filter(
                DBImage.filepath == str(image_path)
            ).first()
            
            if not image_record:
                # Try with image_path field
                image_record = session.query(DBImage).filter(
                    DBImage.image_path == str(image_path)
                ).first()
            
            if image_record:
                # Save each detection
                for det in result['detections']:
                    bbox = det['bbox']
                    detection = Detection(
                        image_id=image_record.id,
                        x1=bbox[0],
                        y1=bbox[1],
                        x2=bbox[2],
                        y2=bbox[3],
                        confidence=det['confidence'],
                        object_type=det['object_type'],
                        timestamp=datetime.now()
                    )
                    session.add(detection)
                
                session.commit()
                
                if self.verbose:
                    print(f"Saved {len(result['detections'])} detections for {Path(image_path).name}")
        
        return result


def main():
    """Test the YOLOv12 native detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv12 Native Detector')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Directory of images')
    parser.add_argument('--model', type=str, default='data/models/yolov12x.pt', help='Model path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create detector
    detector = YOLOv12NativeDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        verbose=args.verbose
    )
    
    # Run detection
    if args.image:
        result = detector.detect_single_image(args.image)
        print(f"Detections: {result['num_detections']}")
        for det in result['detections']:
            print(f"  - {det['object_type']}: {det['confidence']:.2f}")
    
    elif args.batch:
        batch_dir = Path(args.batch)
        if batch_dir.is_dir():
            image_files = list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.jpeg"))
            print(f"Processing {len(image_files)} images...")
            
            results = detector.detect_batch([str(f) for f in image_files])
            
            total_detections = sum(r['num_detections'] for r in results if r['success'])
            successful = sum(1 for r in results if r['success'])
            
            print(f"Processed {successful}/{len(image_files)} images")
            print(f"Total detections: {total_detections}")
    
    else:
        print("Please specify --image or --batch")


if __name__ == "__main__":
    main()