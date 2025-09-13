"""
Simplified Detection System for Container Analytics

This module provides YOLO-based container detection with database integration.

Features:
- Container detection using YOLOv12
- Database integration for persistence
- Performance monitoring and analytics
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from sqlalchemy.orm import Session
from dataclasses import dataclass

from .yolo_detector import YOLODetector
from ..database.models import Image, Detection, Container, session_scope
from ..database.queries import DatabaseQueries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection result from YOLO processing."""
    timestamp: datetime
    image_path: str
    camera_id: str
    detections: sv.Detections
    processing_time: float


class SimplifiedContainerDetector:
    """
    Simplified container detection system using YOLO.

    This class provides container detection using YOLOv12 with database integration
    for storage and analytics.
    """
    
    def __init__(
        self,
        yolo_model_path: str = "data/models/yolov12x.pt",
        yolo_confidence: float = 0.25,
        use_gpu: bool = True
    ):
        """
        Initialize the simplified detection system.

        Args:
            yolo_model_path: Path to YOLO model weights
            yolo_confidence: Minimum confidence for YOLO detections
            use_gpu: Whether to use GPU acceleration
        """
        self.yolo_confidence = yolo_confidence
        self.use_gpu = use_gpu

        # Initialize YOLO detector
        logger.info("Initializing YOLO detector...")
        self.yolo_detector = YOLODetector(
            model_path=yolo_model_path,
            confidence_threshold=yolo_confidence,
            device="cuda" if use_gpu else "cpu"
        )

        # Database queries helper
        self.db_queries = DatabaseQueries()

        # Performance tracking
        self.processing_times = []
        self.total_detections = 0

        logger.info("Simplified container detector initialized successfully")
    
    def process_image(
        self,
        image_path: Union[str, Path],
        camera_id: str,
        timestamp: Optional[datetime] = None,
        save_to_db: bool = True
    ) -> DetectionResult:
        """
        Process a single image through YOLO detection.

        Args:
            image_path: Path to input image
            camera_id: Identifier for the camera/stream
            timestamp: Image timestamp (default: current time)
            save_to_db: Whether to save results to database

        Returns:
            DetectionResult with detection information
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = datetime.now()
        
        image_path = str(image_path)
        logger.debug(f"Processing image: {Path(image_path).name} from camera: {camera_id}")
        
        try:
            # YOLO Detection
            yolo_result = self.yolo_detector.detect_single_image(image_path)
            detections = yolo_result["detections"]

            # Save to database if requested
            if save_to_db:
                self._save_to_database(
                    image_path, camera_id, timestamp, detections
                )

            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Update statistics
            self.total_detections += len(detections)

            logger.info(f"Processed {Path(image_path).name}: "
                       f"{len(detections)} detections "
                       f"in {processing_time:.3f}s")

            return DetectionResult(
                timestamp=timestamp,
                image_path=image_path,
                camera_id=camera_id,
                detections=detections,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    
    def _save_to_database(
        self,
        image_path: str,
        camera_id: str,
        timestamp: datetime,
        detections: sv.Detections
    ) -> None:
        """
        Save detection results to database.

        Args:
            image_path: Path to processed image
            camera_id: Camera identifier
            timestamp: Processing timestamp
            detections: YOLO detections
        """
        try:
            with session_scope() as session:
                # Save image record
                image_record = Image(
                    image_path=image_path,
                    camera_id=camera_id,
                    timestamp=timestamp,
                    detection_count=len(detections) if detections else 0,
                    processed=True
                )
                session.add(image_record)
                session.flush()  # Get the image ID
                
                # Save detections
                if len(detections) > 0:
                    for i in range(len(detections)):
                        bbox = detections.xyxy[i]
                        confidence = float(detections.confidence[i]) if detections.confidence is not None else 1.0
                        class_id = int(detections.class_id[i]) if detections.class_id is not None else 0

                        detection_record = Detection(
                            image_id=image_record.id,
                            x1=float(bbox[0]),
                            y1=float(bbox[1]),
                            x2=float(bbox[2]),
                            y2=float(bbox[3]),
                            confidence=confidence,
                            class_id=class_id,
                            track_id=None,  # No tracking in simplified version
                            timestamp=timestamp
                        )
                        session.add(detection_record)
                
                # Note: Container records would be created by separate OCR processing
                # This simplified version only stores raw detections
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            raise
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        camera_id: str,
        timestamps: Optional[List[datetime]] = None,
        save_to_db: bool = True
    ) -> List[DetectionResult]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths
            camera_id: Camera identifier
            timestamps: Optional timestamps for each image
            save_to_db: Whether to save results to database
            
        Returns:
            List of detection results
        """
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        start_time = time.time()
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            timestamp = timestamps[i] if timestamps else None
            
            try:
                result = self.process_image(image_path, camera_id, timestamp, save_to_db)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                # Create empty result for failed processing
                results.append(DetectionResult(
                    timestamp=timestamp or datetime.now(),
                    image_path=str(image_path),
                    camera_id=camera_id,
                    detections=sv.Detections.empty(),
                    processing_time=0.0
                ))
        
        total_time = time.time() - start_time
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        
        return results
    
    def get_container_analytics(
        self,
        time_window_hours: int = 24,
        camera_id: Optional[str] = None
    ) -> Dict:
        """
        Get analytics for container activity.
        
        Args:
            time_window_hours: Time window for analytics in hours
            camera_id: Optional camera filter
            
        Returns:
            Dictionary with analytics data
        """
        try:
            with session_scope() as session:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=time_window_hours)
                
                query = session.query(Container).filter(
                    Container.first_seen >= start_time
                )
                
                if camera_id:
                    query = query.filter(Container.camera_id == camera_id)
                
                containers = query.all()
                
                if not containers:
                    return {"message": "No containers found in time window"}
                
                # Calculate metrics
                dwell_times = [c.dwell_time for c in containers if c.dwell_time]
                active_containers = [c for c in containers if c.status == "active"]
                departed_containers = [c for c in containers if c.status == "departed"]
                
                analytics = {
                    "time_window_hours": time_window_hours,
                    "camera_id": camera_id,
                    "total_containers": len(containers),
                    "active_containers": len(active_containers),
                    "departed_containers": len(departed_containers),
                    "throughput_per_hour": len(containers) / time_window_hours,
                }
                
                if dwell_times:
                    analytics.update({
                        "avg_dwell_time_hours": np.mean(dwell_times),
                        "median_dwell_time_hours": np.median(dwell_times),
                        "min_dwell_time_hours": np.min(dwell_times),
                        "max_dwell_time_hours": np.max(dwell_times),
                    })
                
                return analytics
                
        except Exception as e:
            logger.error(f"Error calculating analytics: {e}")
            return {"error": str(e)}
    
    def get_performance_stats(self) -> Dict:
        """
        Get system performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        yolo_stats = self.yolo_detector.get_performance_stats() if hasattr(self.yolo_detector, 'get_performance_stats') else {}

        simplified_stats = {
            "total_images_processed": len(self.processing_times),
            "total_detections": self.total_detections,
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "processing_fps": 1.0 / np.mean(self.processing_times) if self.processing_times else 0,
        }

        return {
            "simplified": simplified_stats,
            "yolo": yolo_stats
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.processing_times.clear()
        self.total_detections = 0
        logger.info("Statistics reset completed")
    
    def shutdown(self) -> None:
        """Cleanup and shutdown the detection system."""
        logger.info("Shutting down simplified detection system...")
        # Cleanup if needed
        logger.info("Shutdown completed")


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Container Detection Test")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--directory", type=str, help="Path to directory with images")
    parser.add_argument("--camera-id", type=str, default="test_camera", help="Camera ID")
    parser.add_argument("--model", type=str, default="data/models/yolov12x.pt", help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--no-db", action="store_true", help="Don't save to database")
    
    args = parser.parse_args()
    
    if not (args.image or args.directory):
        print("Please provide either --image or --directory argument")
        exit(1)
    
    # Initialize detector
    detector = SimplifiedContainerDetector(
        yolo_model_path=args.model,
        yolo_confidence=args.confidence
    )
    
    try:
        if args.image:
            # Process single image
            result = detector.process_image(
                args.image, 
                args.camera_id, 
                save_to_db=not args.no_db
            )
            
            print(f"Processed {Path(args.image).name}:")
            print(f"  Detections: {len(result.detections)}")
            print(f"  Processing time: {result.processing_time:.3f}s")
        
        elif args.directory:
            # Process directory
            image_dir = Path(args.directory)
            image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
            
            results = detector.process_batch(
                image_paths[:10],  # Limit for demo
                args.camera_id,
                save_to_db=not args.no_db
            )
            
            print(f"Processed {len(results)} images:")
            
            total_detections = 0
            for i, result in enumerate(results):
                total_detections += len(result.detections)
                print(f"  {Path(result.image_path).name}: "
                      f"{len(result.detections)} detections")

            print(f"Total detections: {total_detections}")
        
        # Print performance stats
        print("\nPerformance Statistics:")
        stats = detector.get_performance_stats()
        for category, data in stats.items():
            print(f"  {category.upper()}:")
            for key, value in data.items():
                print(f"    {key}: {value}")
        
        # Print analytics if using database
        if not args.no_db:
            print("\nContainer Analytics:")
            analytics = detector.get_container_analytics(time_window_hours=24)
            for key, value in analytics.items():
                print(f"  {key}: {value}")
    
    finally:
        detector.shutdown()