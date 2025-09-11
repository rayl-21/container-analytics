#!/usr/bin/env python3
"""
Integration demo script for container tracking system.

This script demonstrates the complete container tracking pipeline:
1. OCR container number extraction
2. Multi-object tracking 
3. Container lifecycle management
4. Analytics calculations

Run with: python test_integration_demo.py
"""

import numpy as np
import cv2
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock
import json

from modules.detection.ocr import ContainerOCR
from modules.detection.tracker import ContainerTracker
from modules.detection.integrated_detector import IntegratedContainerDetector
from modules.analytics.tracking_analytics import ContainerTrackingAnalytics
import supervision as sv


def create_test_image_with_container(container_number: str, save_path: str = None) -> str:
    """Create a test image with a container number."""
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    
    # Add container-like background
    cv2.rectangle(img, (50, 50), (550, 150), (200, 200, 200), -1)
    cv2.rectangle(img, (50, 50), (550, 150), (0, 0, 0), 2)
    
    # Add container number text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    color = (0, 0, 0)
    thickness = 2
    
    text_size = cv2.getTextSize(container_number, font, font_scale, thickness)[0]
    text_x = (600 - text_size[0]) // 2
    text_y = (200 + text_size[1]) // 2
    
    cv2.putText(img, container_number, (text_x, text_y), font, font_scale, color, thickness)
    
    if save_path is None:
        save_path = f"/tmp/container_{container_number}.jpg"
    
    cv2.imwrite(save_path, img)
    return save_path


def demo_ocr_system():
    """Demonstrate OCR container number extraction."""
    print("\n=== OCR System Demo ===")
    
    ocr = ContainerOCR(use_easyocr=False, use_tesseract=True)  # Use only Tesseract for demo
    
    # Test container numbers
    test_containers = ["MSCU1234567", "APLU9876543", "CGMU1111111"]
    
    for container in test_containers:
        # Create test image
        image_path = create_test_image_with_container(container)
        
        try:
            # Mock OCR to avoid actual image processing in demo
            with patch.object(ocr, '_extract_text_from_region') as mock_ocr:
                mock_ocr.return_value = [{
                    'text': container,
                    'confidence': 0.95,
                    'engine': 'mock'
                }]
                
                results = ocr.extract_container_numbers(image_path)
                container_results = [r for r in results if r.get('is_container_number', False)]
                
                print(f"Image: {Path(image_path).name}")
                if container_results:
                    result = container_results[0]
                    print(f"  Found: {result['formatted_number']} (confidence: {result['confidence']:.2f})")
                    print(f"  Valid: {ocr._is_valid_container_number(result['text'])}")
                else:
                    print(f"  No container numbers found")
                
        finally:
            Path(image_path).unlink(missing_ok=True)
    
    # Test performance stats
    stats = ocr.get_performance_stats()
    print(f"\nOCR Performance: {stats['total_operations']} operations")


def demo_tracking_system():
    """Demonstrate multi-object tracking."""
    print("\n=== Tracking System Demo ===")
    
    tracker = ContainerTracker()
    
    # Simulate 5 frames of tracking
    for frame_num in range(5):
        timestamp = datetime.now() + timedelta(seconds=frame_num)
        
        # Create mock detections (containers moving)
        if frame_num < 3:
            # Two containers detected
            xyxy = np.array([
                [100 + frame_num * 10, 100, 200 + frame_num * 10, 200],  # Container 1 moving right
                [300, 150 + frame_num * 5, 400, 250 + frame_num * 5],   # Container 2 moving down
            ], dtype=np.float32)
            confidence = np.array([0.9, 0.8], dtype=np.float32)
            class_id = np.array([0, 0], dtype=int)
        else:
            # One container leaves, one new one appears
            xyxy = np.array([
                [300, 150 + frame_num * 5, 400, 250 + frame_num * 5],   # Container 2 continues
                [500, 100, 600, 200],                                   # New container 3
            ], dtype=np.float32)
            confidence = np.array([0.8, 0.85], dtype=np.float32)
            class_id = np.array([0, 0], dtype=int)
        
        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked_detections = tracker.update(detections, timestamp)
        
        active_tracks = tracker.get_active_tracks()
        print(f"Frame {frame_num}: {len(tracked_detections)} detections, {len(active_tracks)} active tracks")
        
        # Show track details
        for track_id, track_info in active_tracks.items():
            print(f"  Track {track_id}: {len(track_info.positions)} positions, dwell: {track_info.dwell_time:.1f}s")
    
    # Show performance stats
    perf_stats = tracker.get_performance_stats()
    print(f"\nTracking Performance:")
    print(f"  Total tracks created: {perf_stats['total_tracks_created']}")
    print(f"  Processing FPS: {perf_stats['fps_mean']:.1f}")


def demo_integrated_detection():
    """Demonstrate integrated detection system."""
    print("\n=== Integrated Detection Demo ===")
    
    # Mock the YOLO detector to avoid model loading
    with patch('modules.detection.integrated_detector.YOLODetector') as mock_yolo, \
         patch('modules.detection.integrated_detector.ContainerOCR') as mock_ocr, \
         patch('modules.detection.integrated_detector.ContainerTracker') as mock_tracker:
        
        # Configure YOLO mock
        mock_yolo_instance = Mock()
        xyxy = np.array([[100, 100, 300, 200]], dtype=np.float32)
        confidence = np.array([0.9], dtype=np.float32)
        class_id = np.array([0], dtype=int)
        mock_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        mock_yolo_instance.detect_single_image.return_value = {'detections': mock_detections}
        mock_yolo.return_value = mock_yolo_instance
        
        # Configure OCR mock
        mock_ocr_instance = Mock()
        mock_ocr_instance.extract_container_numbers.return_value = [{
            'detection_index': 0,
            'text': 'MSCU1234567',
            'confidence': 0.95,
            'is_container_number': True,
            'formatted_number': 'MSCU1234567'
        }]
        mock_ocr.return_value = mock_ocr_instance
        
        # Configure tracker mock
        mock_tracker_instance = Mock()
        tracked_detections = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id,
            tracker_id=np.array([1], dtype=int)
        )
        mock_tracker_instance.update.return_value = tracked_detections
        mock_tracker.return_value = mock_tracker_instance
        
        # Create integrated detector
        detector = IntegratedContainerDetector()
        detector.yolo_detector = mock_yolo_instance
        detector.ocr_engine = mock_ocr_instance
        detector.tracker = mock_tracker_instance
        
        # Create test image
        test_image = create_test_image_with_container("MSCU1234567")
        
        try:
            # Process image
            result = detector.process_image(
                test_image,
                camera_id="in_gate",
                save_to_db=False
            )
            
            print(f"Processed: {Path(result.image_path).name}")
            print(f"  Detections: {len(result.detections)}")
            print(f"  Tracked objects: {len(result.tracked_detections)}")
            print(f"  OCR results: {len(result.ocr_results)}")
            print(f"  Container events: {len(result.container_events)}")
            print(f"  Processing time: {result.processing_time:.3f}s")
            
            # Show container events
            for event in result.container_events:
                print(f"  Event: {event.event_type} - {event.container_number} at {event.camera_id}")
            
        finally:
            Path(test_image).unlink(missing_ok=True)


def demo_analytics():
    """Demonstrate analytics system."""
    print("\n=== Analytics System Demo ===")
    
    # Mock database operations for demo
    analytics = ContainerTrackingAnalytics(max_capacity=1000)
    
    # Mock analytics methods to return sample data
    from modules.analytics.tracking_analytics import DwellTimeMetrics, ThroughputMetrics, CapacityMetrics
    
    with patch.object(analytics, 'get_dwell_time_analytics') as mock_dwell, \
         patch.object(analytics, 'get_throughput_analytics') as mock_throughput, \
         patch.object(analytics, 'get_capacity_analytics') as mock_capacity:
        
        mock_dwell.return_value = DwellTimeMetrics(
            total_containers=25,
            active_containers=15,
            departed_containers=10,
            avg_dwell_time_hours=4.5,
            median_dwell_time_hours=3.2,
            min_dwell_time_hours=0.5,
            max_dwell_time_hours=12.0,
            std_dwell_time_hours=2.8,
            percentile_95_hours=10.5,
            time_window="24h"
        )
        
        mock_throughput.return_value = ThroughputMetrics(
            time_period="24h",
            total_entries=30,
            total_exits=22,
            net_containers=8,
            entries_per_hour=1.25,
            exits_per_hour=0.92,
            peak_entry_hour=10,
            peak_exit_hour=15,
            peak_entry_count=5,
            peak_exit_count=4
        )
        
        mock_capacity.return_value = CapacityMetrics(
            current_occupancy=45,
            max_observed_occupancy=62,
            avg_occupancy_24h=48.3,
            occupancy_trend="stable",
            capacity_utilization=4.5
        )
        
        # Get analytics
        dwell_metrics = analytics.get_dwell_time_analytics(24)
        throughput_metrics = analytics.get_throughput_analytics(24)
        capacity_metrics = analytics.get_capacity_analytics()
        
        print("Dwell Time Analytics (24h):")
        print(f"  Total containers: {dwell_metrics.total_containers}")
        print(f"  Average dwell time: {dwell_metrics.avg_dwell_time_hours:.1f} hours")
        print(f"  Median dwell time: {dwell_metrics.median_dwell_time_hours:.1f} hours")
        
        print("\nThroughput Analytics (24h):")
        print(f"  Entries: {throughput_metrics.total_entries}")
        print(f"  Exits: {throughput_metrics.total_exits}")
        print(f"  Net containers: {throughput_metrics.net_containers}")
        print(f"  Peak entry hour: {throughput_metrics.peak_entry_hour}:00 ({throughput_metrics.peak_entry_count} containers)")
        
        print("\nCapacity Analytics:")
        print(f"  Current occupancy: {capacity_metrics.current_occupancy}")
        print(f"  Max observed: {capacity_metrics.max_observed_occupancy}")
        print(f"  Trend: {capacity_metrics.occupancy_trend}")
        print(f"  Utilization: {capacity_metrics.capacity_utilization:.1f}%")


def main():
    """Run the complete integration demo."""
    print("🚢 Container Analytics - Tracking System Integration Demo")
    print("=" * 60)
    
    try:
        demo_ocr_system()
        demo_tracking_system()
        demo_integrated_detection()
        demo_analytics()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("\nThe container tracking system includes:")
        print("  • OCR for container number extraction")
        print("  • Multi-object tracking with ByteTrack")
        print("  • Integrated detection pipeline")
        print("  • Container lifecycle management")
        print("  • Advanced analytics and reporting")
        print("  • Database integration for persistence")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()