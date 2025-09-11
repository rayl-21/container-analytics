"""
Object Tracking Module for Container Analytics

This module provides object tracking capabilities using ByteTrack to maintain
unique IDs for containers across multiple frames and calculate dwell time.

Features:
- ByteTrack-based multi-object tracking
- Unique ID assignment and persistence
- Dwell time calculation for containers
- Track lifecycle management
- Performance optimization for real-time processing
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import supervision as sv
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    """Information about a tracked object."""
    track_id: int
    class_id: int
    first_seen: datetime
    last_seen: datetime
    positions: List[Tuple[float, float, float, float]] = field(default_factory=list)  # x1, y1, x2, y2
    confidences: List[float] = field(default_factory=list)
    is_active: bool = True
    
    @property
    def dwell_time(self) -> float:
        """Calculate dwell time in seconds."""
        return (self.last_seen - self.first_seen).total_seconds()
    
    @property
    def center_positions(self) -> List[Tuple[float, float]]:
        """Get center positions of all bounding boxes."""
        return [
            ((x1 + x2) / 2, (y1 + y2) / 2)
            for x1, y1, x2, y2 in self.positions
        ]
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score."""
        return np.mean(self.confidences) if self.confidences else 0.0


class ContainerTracker:
    """
    Multi-object tracker for containers using ByteTrack algorithm.
    
    This class maintains unique IDs for detected objects across frames,
    calculates dwell times, and manages track lifecycles.
    """
    
    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
        max_track_age: int = 300  # seconds
    ):
        """
        Initialize the container tracker.
        
        Args:
            track_thresh: Threshold for track activation
            track_buffer: Buffer size for lost tracks
            match_thresh: Matching threshold for track association
            frame_rate: Expected frame rate (for time calculations)
            max_track_age: Maximum age of inactive tracks in seconds
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.max_track_age = max_track_age
        
        # Initialize ByteTrack tracker with correct parameter names
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )
        
        # Track management
        self.tracks: Dict[int, TrackInfo] = {}
        self.frame_count = 0
        self.processing_times = []
        
        # Statistics
        self.total_tracks_created = 0
        self.active_tracks_count = 0
        
        logger.info(f"Initialized ContainerTracker with parameters: "
                   f"track_thresh={track_thresh}, track_buffer={track_buffer}, "
                   f"match_thresh={match_thresh}, frame_rate={frame_rate}")
    
    def update(self, detections: sv.Detections, timestamp: Optional[datetime] = None) -> sv.Detections:
        """
        Update tracker with new detections.
        
        Args:
            detections: Supervision Detections object
            timestamp: Timestamp for this frame (default: current time)
            
        Returns:
            Updated detections with track IDs
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = datetime.now()
        
        self.frame_count += 1
        
        try:
            # Update tracker with new detections
            tracked_detections = self.tracker.update_with_detections(detections)
            
            # Update track information
            self._update_track_info(tracked_detections, timestamp)
            
            # Clean up old tracks
            self._cleanup_old_tracks(timestamp)
            
            # Update statistics
            self.active_tracks_count = len([t for t in self.tracks.values() if t.is_active])
            
            # Record processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.debug(f"Frame {self.frame_count}: {len(tracked_detections)} tracked objects, "
                        f"{self.active_tracks_count} active tracks")
            
            return tracked_detections
            
        except Exception as e:
            logger.error(f"Error updating tracker: {e}")
            raise
    
    def _update_track_info(self, detections: sv.Detections, timestamp: datetime) -> None:
        """
        Update internal track information with new detections.
        
        Args:
            detections: Tracked detections with IDs
            timestamp: Current timestamp
        """
        # Get current track IDs
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            current_track_ids = set(detections.tracker_id)
            track_ids = detections.tracker_id
        else:
            current_track_ids = set()
            track_ids = []
        
        # Update existing tracks and create new ones
        for i, track_id in enumerate(track_ids):
            if track_id == -1:  # Skip untracked detections
                continue
                
            bbox = detections.xyxy[i]
            confidence = detections.confidence[i] if detections.confidence is not None else 1.0
            class_id = detections.class_id[i] if detections.class_id is not None else 0
            
            if track_id in self.tracks:
                # Update existing track
                track = self.tracks[track_id]
                track.last_seen = timestamp
                track.positions.append(tuple(bbox))
                track.confidences.append(float(confidence))
                track.is_active = True
            else:
                # Create new track
                new_track = TrackInfo(
                    track_id=int(track_id),
                    class_id=int(class_id),
                    first_seen=timestamp,
                    last_seen=timestamp,
                    positions=[tuple(bbox)],
                    confidences=[float(confidence)],
                    is_active=True
                )
                self.tracks[track_id] = new_track
                self.total_tracks_created += 1
                
                logger.info(f"Created new track ID: {track_id}")
        
        # Mark missing tracks as inactive
        for track_id, track in self.tracks.items():
            if track.is_active and track_id not in current_track_ids:
                track.is_active = False
                logger.debug(f"Track {track_id} became inactive")
    
    def _cleanup_old_tracks(self, current_time: datetime) -> None:
        """
        Remove tracks that are too old.
        
        Args:
            current_time: Current timestamp
        """
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            time_since_last_seen = (current_time - track.last_seen).total_seconds()
            
            if time_since_last_seen > self.max_track_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            logger.info(f"Removing old track {track_id} (age: "
                       f"{(current_time - self.tracks[track_id].last_seen).total_seconds():.1f}s)")
            del self.tracks[track_id]
    
    def get_track_info(self, track_id: int) -> Optional[TrackInfo]:
        """
        Get information about a specific track.
        
        Args:
            track_id: Track ID to query
            
        Returns:
            TrackInfo object or None if track doesn't exist
        """
        return self.tracks.get(track_id)
    
    def get_active_tracks(self) -> Dict[int, TrackInfo]:
        """
        Get all currently active tracks.
        
        Returns:
            Dictionary of active tracks
        """
        return {
            track_id: track
            for track_id, track in self.tracks.items()
            if track.is_active
        }
    
    def get_tracks_by_class(self, class_id: int) -> Dict[int, TrackInfo]:
        """
        Get all tracks for a specific class.
        
        Args:
            class_id: Class ID to filter by
            
        Returns:
            Dictionary of tracks matching the class
        """
        return {
            track_id: track
            for track_id, track in self.tracks.items()
            if track.class_id == class_id
        }
    
    def get_dwell_time_statistics(self, class_id: Optional[int] = None) -> Dict:
        """
        Get dwell time statistics for all tracks or a specific class.
        
        Args:
            class_id: Optional class ID to filter by
            
        Returns:
            Dictionary with dwell time statistics
        """
        # Filter tracks
        if class_id is not None:
            relevant_tracks = self.get_tracks_by_class(class_id)
        else:
            relevant_tracks = self.tracks
        
        if not relevant_tracks:
            return {"message": "No tracks available"}
        
        # Calculate dwell times
        dwell_times = [track.dwell_time for track in relevant_tracks.values()]
        
        return {
            "total_tracks": len(relevant_tracks),
            "active_tracks": len([t for t in relevant_tracks.values() if t.is_active]),
            "mean_dwell_time": float(np.mean(dwell_times)),
            "median_dwell_time": float(np.median(dwell_times)),
            "min_dwell_time": float(np.min(dwell_times)),
            "max_dwell_time": float(np.max(dwell_times)),
            "std_dwell_time": float(np.std(dwell_times)),
            "class_id": class_id
        }
    
    def get_track_trajectories(self, track_id: Optional[int] = None) -> Dict:
        """
        Get movement trajectories for tracks.
        
        Args:
            track_id: Specific track ID, or None for all active tracks
            
        Returns:
            Dictionary with trajectory data
        """
        if track_id is not None:
            if track_id not in self.tracks:
                return {"error": f"Track {track_id} not found"}
            tracks_to_analyze = {track_id: self.tracks[track_id]}
        else:
            tracks_to_analyze = self.get_active_tracks()
        
        trajectories = {}
        
        for tid, track in tracks_to_analyze.items():
            center_positions = track.center_positions
            
            # Calculate movement statistics if enough positions
            if len(center_positions) >= 2:
                # Calculate distances between consecutive positions
                distances = []
                for i in range(1, len(center_positions)):
                    x1, y1 = center_positions[i-1]
                    x2, y2 = center_positions[i]
                    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    distances.append(dist)
                
                total_distance = sum(distances)
                avg_speed = total_distance / track.dwell_time if track.dwell_time > 0 else 0
                
                trajectories[tid] = {
                    "track_id": tid,
                    "positions": center_positions,
                    "total_distance": float(total_distance),
                    "average_speed": float(avg_speed),
                    "dwell_time": track.dwell_time,
                    "position_count": len(center_positions)
                }
            else:
                trajectories[tid] = {
                    "track_id": tid,
                    "positions": center_positions,
                    "total_distance": 0.0,
                    "average_speed": 0.0,
                    "dwell_time": track.dwell_time,
                    "position_count": len(center_positions)
                }
        
        return trajectories
    
    def get_performance_stats(self) -> Dict:
        """
        Get tracker performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.processing_times:
            return {"message": "No tracking operations performed yet"}
        
        times = np.array(self.processing_times)
        
        return {
            "total_frames_processed": self.frame_count,
            "total_tracks_created": self.total_tracks_created,
            "active_tracks": self.active_tracks_count,
            "total_tracks": len(self.tracks),
            "mean_processing_time": float(np.mean(times)),
            "median_processing_time": float(np.median(times)),
            "max_processing_time": float(np.max(times)),
            "fps_mean": 1.0 / float(np.mean(times)) if np.mean(times) > 0 else 0,
            "configuration": {
                "track_thresh": self.track_thresh,
                "track_buffer": self.track_buffer,
                "match_thresh": self.match_thresh,
                "frame_rate": self.frame_rate,
                "max_track_age": self.max_track_age
            }
        }
    
    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=self.frame_rate
        )
        
        self.tracks.clear()
        self.frame_count = 0
        self.processing_times.clear()
        self.total_tracks_created = 0
        self.active_tracks_count = 0
        
        logger.info("Tracker reset to initial state")


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import cv2
    from pathlib import Path
    from .yolo_detector import YOLODetector
    
    parser = argparse.ArgumentParser(description="Container Tracker Test")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--images", type=str, help="Path to directory with sequential images")
    parser.add_argument("--output", type=str, help="Output directory for annotated frames")
    parser.add_argument("--model", type=str, default="yolov12x.pt", help="YOLO model path")
    
    args = parser.parse_args()
    
    if not (args.video or args.images):
        print("Please provide either --video or --images argument")
        exit(1)
    
    # Initialize detector and tracker
    detector = YOLODetector(model_path=args.model, confidence_threshold=0.3)
    tracker = ContainerTracker()
    
    # Process video or image sequence
    if args.video:
        cap = cv2.VideoCapture(args.video)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            temp_path = f"/tmp/frame_{frame_count}.jpg"
            cv2.imwrite(temp_path, frame)
            
            result = detector.detect_single_image(temp_path)
            tracked_detections = tracker.update(result["detections"])
            
            print(f"Frame {frame_count}: {len(tracked_detections)} tracked objects")
            frame_count += 1
            
            # Limit processing for demo
            if frame_count > 100:
                break
        
        cap.release()
    
    elif args.images:
        image_dir = Path(args.images)
        image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        
        for i, image_path in enumerate(image_paths[:50]):  # Limit for demo
            result = detector.detect_single_image(image_path)
            tracked_detections = tracker.update(result["detections"])
            
            print(f"Image {i}: {len(tracked_detections)} tracked objects")
    
    # Print final statistics
    print("\nTracker Performance:", tracker.get_performance_stats())
    print("\nDwell Time Statistics:", tracker.get_dwell_time_statistics())
    print("\nActive Tracks:", len(tracker.get_active_tracks()))