"""
Integrated Detection System for Container Analytics

This module combines YOLO detection, OCR text recognition, and multi-object tracking
to provide a complete container identification and tracking solution.

Features:
- Container detection using YOLOv12
- Container number extraction using OCR
- Multi-object tracking with ByteTrack
- Movement detection for IN/OUT gates
- Container lifecycle management
- Database integration for persistence
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
from .ocr import ContainerOCR
from .tracker import ContainerTracker, TrackInfo
from ..database.models import Image, Detection, Container, session_scope
from ..database.queries import DatabaseQueries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContainerEvent:
    """Container movement event data."""
    container_number: str
    track_id: int
    event_type: str  # 'entry', 'exit', 'movement'
    camera_id: str
    timestamp: datetime
    confidence: float
    bbox: Tuple[float, float, float, float]
    ocr_confidence: float


@dataclass
class DetectionResult:
    """Complete detection result including tracking and OCR."""
    timestamp: datetime
    image_path: str
    camera_id: str
    detections: sv.Detections
    tracked_detections: sv.Detections
    ocr_results: List[Dict]
    container_events: List[ContainerEvent]
    processing_time: float


class IntegratedContainerDetector:
    """
    Integrated container detection system combining YOLO, OCR, and tracking.
    
    This class provides a complete solution for container identification,
    tracking, and lifecycle management across multiple camera feeds.
    """
    
    def __init__(
        self,
        yolo_model_path: str = "data/models/yolov12x.pt",
        yolo_confidence: float = 0.25,
        ocr_confidence: float = 0.5,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        use_gpu: bool = True,
        max_track_age: int = 600  # 10 minutes
    ):
        """
        Initialize the integrated detection system.
        
        Args:
            yolo_model_path: Path to YOLO model weights
            yolo_confidence: Minimum confidence for YOLO detections
            ocr_confidence: Minimum confidence for OCR results
            track_thresh: Threshold for track activation
            track_buffer: Buffer size for lost tracks
            match_thresh: Matching threshold for track association
            use_gpu: Whether to use GPU acceleration
            max_track_age: Maximum age of inactive tracks in seconds
        """
        self.yolo_confidence = yolo_confidence
        self.ocr_confidence = ocr_confidence
        self.use_gpu = use_gpu
        self.max_track_age = max_track_age
        
        # Initialize components
        logger.info("Initializing YOLO detector...")
        self.yolo_detector = YOLODetector(
            model_path=yolo_model_path,
            confidence_threshold=yolo_confidence,
            device="cuda" if use_gpu else "cpu"
        )
        
        logger.info("Initializing OCR engine...")
        self.ocr_engine = ContainerOCR(
            use_easyocr=True,
            use_tesseract=True,
            easyocr_gpu=use_gpu
        )
        
        logger.info("Initializing container tracker...")
        self.tracker = ContainerTracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            max_track_age=max_track_age
        )
        
        # Database queries helper
        self.db_queries = DatabaseQueries()
        
        # Container management
        self.known_containers: Dict[str, Dict] = {}  # container_number -> info
        self.track_to_container: Dict[int, str] = {}  # track_id -> container_number
        
        # Performance tracking
        self.processing_times = []
        self.total_detections = 0
        self.total_containers_identified = 0
        
        logger.info("Integrated container detector initialized successfully")
    
    def process_image(
        self,
        image_path: Union[str, Path],
        camera_id: str,
        timestamp: Optional[datetime] = None,
        save_to_db: bool = True
    ) -> DetectionResult:
        """
        Process a single image through the complete detection pipeline.
        
        Args:
            image_path: Path to input image
            camera_id: Identifier for the camera/stream
            timestamp: Image timestamp (default: current time)
            save_to_db: Whether to save results to database
            
        Returns:
            DetectionResult with all processing information
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = datetime.now()
        
        image_path = str(image_path)
        logger.debug(f"Processing image: {Path(image_path).name} from camera: {camera_id}")
        
        try:
            # 1. YOLO Detection
            yolo_result = self.yolo_detector.detect_single_image(image_path)
            detections = yolo_result["detections"]
            
            # 2. Multi-object tracking
            tracked_detections = self.tracker.update(detections, timestamp)
            
            # 3. OCR on tracked containers
            ocr_results = []
            if len(tracked_detections) > 0:
                ocr_results = self.ocr_engine.extract_container_numbers(
                    image_path,
                    tracked_detections,
                    min_confidence=self.ocr_confidence
                )
            
            # 4. Process container events and lifecycle
            container_events = self._process_container_events(
                tracked_detections, ocr_results, camera_id, timestamp
            )
            
            # 5. Save to database if requested
            if save_to_db:
                self._save_to_database(
                    image_path, camera_id, timestamp,
                    detections, tracked_detections, ocr_results, container_events
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update statistics
            self.total_detections += len(detections)
            self.total_containers_identified += len([
                event for event in container_events 
                if event.event_type == 'entry'
            ])
            
            logger.info(f"Processed {Path(image_path).name}: "
                       f"{len(detections)} detections, {len(tracked_detections)} tracked, "
                       f"{len(ocr_results)} OCR results, {len(container_events)} events "
                       f"in {processing_time:.3f}s")
            
            return DetectionResult(
                timestamp=timestamp,
                image_path=image_path,
                camera_id=camera_id,
                detections=detections,
                tracked_detections=tracked_detections,
                ocr_results=ocr_results,
                container_events=container_events,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def _process_container_events(
        self,
        tracked_detections: sv.Detections,
        ocr_results: List[Dict],
        camera_id: str,
        timestamp: datetime
    ) -> List[ContainerEvent]:
        """
        Process container events based on tracking and OCR results.
        
        Args:
            tracked_detections: Detections with track IDs
            ocr_results: OCR results from detection regions
            camera_id: Camera identifier
            timestamp: Current timestamp
            
        Returns:
            List of container events
        """
        events = []
        
        # Map OCR results to track IDs
        track_ocr_map = {}
        for ocr_result in ocr_results:
            if "detection_index" in ocr_result and ocr_result.get("is_container_number", False):
                detection_idx = ocr_result["detection_index"]
                if detection_idx < len(tracked_detections.tracker_id):
                    track_id = tracked_detections.tracker_id[detection_idx]
                    if track_id != -1:  # Valid track ID
                        container_number = ocr_result["formatted_number"]
                        track_ocr_map[track_id] = {
                            "container_number": container_number,
                            "confidence": ocr_result["confidence"]
                        }
        
        # Process each tracked detection
        if tracked_detections.tracker_id is not None:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                if track_id == -1:
                    continue
                
                bbox = tuple(tracked_detections.xyxy[i])
                confidence = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 1.0
                
                # Check if we have OCR for this track
                ocr_info = track_ocr_map.get(track_id)
                
                if ocr_info:
                    container_number = ocr_info["container_number"]
                    ocr_confidence = ocr_info["confidence"]
                    
                    # Check if this is a new container
                    if track_id not in self.track_to_container:
                        # New container detected
                        self.track_to_container[track_id] = container_number
                        
                        if container_number not in self.known_containers:
                            # First time seeing this container
                            self.known_containers[container_number] = {
                                "first_seen": timestamp,
                                "last_seen": timestamp,
                                "cameras": {camera_id},
                                "track_ids": {track_id},
                                "status": "active"
                            }
                            
                            event_type = self._determine_event_type(camera_id, "entry")
                            events.append(ContainerEvent(
                                container_number=container_number,
                                track_id=track_id,
                                event_type=event_type,
                                camera_id=camera_id,
                                timestamp=timestamp,
                                confidence=confidence,
                                bbox=bbox,
                                ocr_confidence=ocr_confidence
                            ))
                        else:
                            # Container seen before, update info
                            container_info = self.known_containers[container_number]
                            container_info["last_seen"] = timestamp
                            container_info["cameras"].add(camera_id)
                            container_info["track_ids"].add(track_id)
                            
                            # Check for movement between cameras
                            if len(container_info["cameras"]) > 1:
                                event_type = self._determine_event_type(camera_id, "movement")
                                events.append(ContainerEvent(
                                    container_number=container_number,
                                    track_id=track_id,
                                    event_type=event_type,
                                    camera_id=camera_id,
                                    timestamp=timestamp,
                                    confidence=confidence,
                                    bbox=bbox,
                                    ocr_confidence=ocr_confidence
                                ))
                    else:
                        # Update existing tracking
                        container_number = self.track_to_container[track_id]
                        if container_number in self.known_containers:
                            container_info = self.known_containers[container_number]
                            container_info["last_seen"] = timestamp
                            container_info["cameras"].add(camera_id)
        
        return events
    
    def _determine_event_type(self, camera_id: str, default: str) -> str:
        """
        Determine the type of container event based on camera ID.
        
        Args:
            camera_id: Camera identifier
            default: Default event type
            
        Returns:
            Event type string
        """
        camera_lower = camera_id.lower()
        
        if "in" in camera_lower or "entry" in camera_lower or "gate_in" in camera_lower:
            return "entry"
        elif "out" in camera_lower or "exit" in camera_lower or "gate_out" in camera_lower:
            return "exit"
        else:
            return default
    
    def _save_to_database(
        self,
        image_path: str,
        camera_id: str,
        timestamp: datetime,
        detections: sv.Detections,
        tracked_detections: sv.Detections,
        ocr_results: List[Dict],
        container_events: List[ContainerEvent]
    ) -> None:
        """
        Save detection results to database.
        
        Args:
            image_path: Path to processed image
            camera_id: Camera identifier
            timestamp: Processing timestamp
            detections: Original YOLO detections
            tracked_detections: Tracked detections
            ocr_results: OCR results
            container_events: Container events
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
                if len(tracked_detections) > 0:
                    for i in range(len(tracked_detections)):
                        bbox = tracked_detections.xyxy[i]
                        confidence = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 1.0
                        class_id = int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 0
                        track_id = int(tracked_detections.tracker_id[i]) if tracked_detections.tracker_id is not None else None
                        
                        if track_id == -1:
                            track_id = None
                        
                        detection_record = Detection(
                            image_id=image_record.id,
                            x1=float(bbox[0]),
                            y1=float(bbox[1]),
                            x2=float(bbox[2]),
                            y2=float(bbox[3]),
                            confidence=confidence,
                            class_id=class_id,
                            track_id=track_id,
                            timestamp=timestamp
                        )
                        session.add(detection_record)
                
                # Save/update containers
                for event in container_events:
                    if event.event_type == "entry":
                        # Create or update container record
                        container = session.query(Container).filter_by(
                            container_number=event.container_number
                        ).first()
                        
                        if container:
                            # Update existing container
                            container.last_seen = timestamp
                            container.total_detections += 1
                            container.calculate_dwell_time()
                            if event.camera_id and not container.camera_id:
                                container.camera_id = event.camera_id
                        else:
                            # Create new container record
                            container = Container(
                                container_number=event.container_number,
                                first_seen=timestamp,
                                last_seen=timestamp,
                                total_detections=1,
                                avg_confidence=event.ocr_confidence,
                                status="active",
                                camera_id=event.camera_id
                            )
                            container.calculate_dwell_time()
                            session.add(container)
                    
                    elif event.event_type == "exit":
                        # Mark container as departed
                        container = session.query(Container).filter_by(
                            container_number=event.container_number
                        ).first()
                        
                        if container:
                            container.last_seen = timestamp
                            container.status = "departed"
                            container.calculate_dwell_time()
                
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
                    tracked_detections=sv.Detections.empty(),
                    ocr_results=[],
                    container_events=[],
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
        ocr_stats = self.ocr_engine.get_performance_stats()
        tracker_stats = self.tracker.get_performance_stats()
        
        integrated_stats = {
            "total_images_processed": len(self.processing_times),
            "total_detections": self.total_detections,
            "total_containers_identified": self.total_containers_identified,
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "processing_fps": 1.0 / np.mean(self.processing_times) if self.processing_times else 0,
        }
        
        return {
            "integrated": integrated_stats,
            "yolo": yolo_stats,
            "ocr": ocr_stats,
            "tracker": tracker_stats
        }
    
    def reset_tracker(self) -> None:
        """Reset the tracking system."""
        self.tracker.reset()
        self.track_to_container.clear()
        logger.info("Tracker reset completed")
    
    def shutdown(self) -> None:
        """Cleanup and shutdown the detection system."""
        logger.info("Shutting down integrated detection system...")
        # Cleanup if needed
        logger.info("Shutdown completed")


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated Container Detection Test")
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
    detector = IntegratedContainerDetector(
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
            print(f"  Tracked objects: {len(result.tracked_detections)}")
            print(f"  OCR results: {len(result.ocr_results)}")
            print(f"  Container events: {len(result.container_events)}")
            print(f"  Processing time: {result.processing_time:.3f}s")
            
            for event in result.container_events:
                print(f"  Event: {event.event_type} - {event.container_number} "
                      f"(confidence: {event.ocr_confidence:.2f})")
        
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
            
            total_containers = 0
            for i, result in enumerate(results):
                total_containers += len(result.container_events)
                print(f"  {Path(result.image_path).name}: "
                      f"{len(result.detections)} detections, "
                      f"{len(result.container_events)} events")
            
            print(f"Total container events: {total_containers}")
        
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