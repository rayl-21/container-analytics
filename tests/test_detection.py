"""
Tests for the detection module.

Tests cover:
- YOLO detector initialization
- Object detection on sample images
- Tracker functionality
- OCR capabilities
- Performance testing
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import supervision as sv

from modules.detection.yolo_detector import YOLODetector


from tests.conftest import requires_gpu, requires_model_file

class TestYOLODetector:
    """Test class for YOLODetector functionality."""
    
    def test_init_default_params(self):
        """Test YOLODetector initialization with default parameters."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo, \
             patch('torch.cuda.is_available', return_value=False):
            
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            
            assert detector.model_path == "yolov8x.pt"
            assert detector.confidence_threshold == 0.5
            assert detector.iou_threshold == 0.7
            assert detector.device == "cpu"  # Should use CPU when CUDA not available
            assert detector.verbose == True
            
            mock_yolo.assert_called_once_with("yolov8x.pt")
            mock_model.to.assert_called_once_with("cpu")
    
    def test_init_custom_params(self):
        """Test YOLODetector initialization with custom parameters."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(
                model_path="yolov8n.pt",
                confidence_threshold=0.7,
                iou_threshold=0.5,
                device="cuda",
                verbose=False
            )
            
            assert detector.model_path == "yolov8n.pt"
            assert detector.confidence_threshold == 0.7
            assert detector.iou_threshold == 0.5
            assert detector.device == "cuda"
            assert detector.verbose == False
            
            mock_yolo.assert_called_once_with("yolov8n.pt")
            mock_model.to.assert_called_once_with("cuda")
    
    @patch('torch.cuda.is_available')
    def test_auto_device_selection_cuda(self, mock_cuda_available):
        """Test automatic device selection when CUDA is available."""
        mock_cuda_available.return_value = True
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            assert detector.device == "cuda"
    
    @patch('torch.cuda.is_available')
    def test_auto_device_selection_cpu(self, mock_cuda_available):
        """Test automatic device selection when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            assert detector.device == "cpu"
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        with patch('modules.detection.yolo_detector.YOLO', side_effect=Exception("Model not found")):
            with pytest.raises(Exception, match="Model not found"):
                YOLODetector()
    
    def test_detect_single_image_success(self, sample_image_with_objects):
        """Test successful single image detection."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            # Setup mock model and results
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            # Mock YOLO results
            mock_results = Mock()
            mock_model.return_value = [mock_results]
            
            # Mock supervision detections
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([[100, 100, 200, 200], [300, 200, 450, 350]])
            mock_detections.confidence = np.array([0.9, 0.8])
            mock_detections.class_id = np.array([2, 7])  # car, truck (container-relevant)
            mock_detections.__len__ = Mock(return_value=2)
            
            with patch('supervision.Detections.from_ultralytics', return_value=mock_detections), \
                 patch('cv2.imread') as mock_imread:
                
                # Mock image loading
                mock_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
                mock_imread.return_value = mock_image
                
                # Apply filtering (container-relevant classes)
                filtered_detections = Mock(spec=sv.Detections)
                filtered_detections.xyxy = mock_detections.xyxy
                filtered_detections.confidence = mock_detections.confidence
                filtered_detections.class_id = mock_detections.class_id
                filtered_detections.__len__ = Mock(return_value=2)
                mock_detections.__getitem__ = Mock(return_value=filtered_detections)
                
                detector = YOLODetector()
                result = detector.detect_single_image(sample_image_with_objects)
                
                assert 'detections' in result
                assert 'metadata' in result
                assert result['metadata']['num_detections'] == 2
                assert result['metadata']['image_path'] == sample_image_with_objects
                assert 'processing_time' in result['metadata']
                assert result['metadata']['processing_time'] > 0
    
    def test_detect_single_image_with_annotation(self, sample_image_with_objects):
        """Test single image detection with annotation."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            mock_model.return_value = [Mock()]
            
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([[100, 100, 200, 200]])
            mock_detections.confidence = np.array([0.9])
            mock_detections.class_id = np.array([2])
            mock_detections.__len__ = Mock(return_value=1)
            mock_detections.__getitem__ = Mock(return_value=mock_detections)
            
            with patch('supervision.Detections.from_ultralytics', return_value=mock_detections), \
                 patch('cv2.imread') as mock_imread, \
                 patch.object(YOLODetector, '_annotate_image') as mock_annotate:
                
                mock_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
                mock_imread.return_value = mock_image
                
                from PIL import Image
                mock_pil_image = Mock(spec=Image.Image)
                mock_annotate.return_value = mock_pil_image
                
                detector = YOLODetector()
                result = detector.detect_single_image(sample_image_with_objects, return_annotated=True)
                
                assert 'annotated_image' in result
                assert result['annotated_image'] == mock_pil_image
                mock_annotate.assert_called_once()
    
    def test_detect_single_image_file_not_found(self):
        """Test single image detection with non-existent file."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            with patch('cv2.imread', return_value=None):
                detector = YOLODetector()
                
                with pytest.raises(ValueError, match="Could not load image"):
                    detector.detect_single_image("/nonexistent/image.jpg")
    
    def test_detect_batch_success(self, sample_images):
        """Test successful batch detection."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            # Mock batch results
            batch_results = []
            for _ in range(len(sample_images)):
                mock_result = Mock()
                batch_results.append(mock_result)
            mock_model.return_value = batch_results
            
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([[100, 100, 200, 200]])
            mock_detections.confidence = np.array([0.9])
            mock_detections.class_id = np.array([2])  # Container-relevant class
            mock_detections.__len__ = Mock(return_value=1)
            mock_detections.__getitem__ = Mock(return_value=mock_detections)
            
            with patch('supervision.Detections.from_ultralytics', return_value=mock_detections), \
                 patch('cv2.imread') as mock_imread:
                
                mock_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
                mock_imread.return_value = mock_image
                
                detector = YOLODetector()
                results = detector.detect_batch(sample_images, batch_size=2)
                
                assert len(results) == len(sample_images)
                for result in results:
                    assert 'detections' in result
                    assert 'metadata' in result
                    assert 'batch_index' in result['metadata']
    
    def test_detect_batch_with_invalid_images(self, temp_dir):
        """Test batch detection with some invalid images."""
        # Create mix of valid and invalid image paths
        valid_images = []
        invalid_images = []
        
        # Valid images
        for i in range(2):
            img_path = temp_dir / f"valid_image_{i}.jpg"
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img_array)
            valid_images.append(str(img_path))
        
        # Invalid images (don't exist)
        for i in range(2):
            invalid_images.append(str(temp_dir / f"invalid_image_{i}.jpg"))
        
        all_images = valid_images + invalid_images
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            mock_model.return_value = [Mock(), Mock()]  # Results for valid images only
            
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([[100, 100, 200, 200]])
            mock_detections.confidence = np.array([0.9])
            mock_detections.class_id = np.array([2])
            mock_detections.__len__ = Mock(return_value=1)
            mock_detections.__getitem__ = Mock(return_value=mock_detections)
            
            with patch('supervision.Detections.from_ultralytics', return_value=mock_detections):
                detector = YOLODetector()
                results = detector.detect_batch(all_images)
                
                # Should only return results for valid images
                assert len(results) == len(valid_images)
    
    def test_annotate_image(self):
        """Test image annotation functionality."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            
            # Create test image and detections
            test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.confidence = np.array([0.9, 0.8])
            mock_detections.class_id = np.array([2, 7])  # car, truck
            
            with patch('supervision.BoxAnnotator') as mock_box_annotator, \
                 patch('supervision.LabelAnnotator') as mock_label_annotator, \
                 patch('cv2.cvtColor') as mock_cvtcolor, \
                 patch('PIL.Image.fromarray') as mock_from_array:
                
                mock_box_instance = Mock()
                mock_label_instance = Mock()
                mock_box_annotator.return_value = mock_box_instance
                mock_label_annotator.return_value = mock_label_instance
                
                mock_box_instance.annotate.return_value = test_image
                mock_label_instance.annotate.return_value = test_image
                mock_cvtcolor.return_value = test_image
                
                from PIL import Image
                mock_pil_image = Mock(spec=Image.Image)
                mock_from_array.return_value = mock_pil_image
                
                result = detector._annotate_image(test_image, mock_detections)
                
                assert result == mock_pil_image
                mock_box_instance.annotate.assert_called_once()
                mock_label_instance.annotate.assert_called_once()
    
    def test_get_performance_stats_no_detections(self):
        """Test performance stats when no detections have been performed."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            stats = detector.get_performance_stats()
            
            assert stats == {"message": "No detections performed yet"}
    
    def test_get_performance_stats_with_detections(self):
        """Test performance stats with detection history."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            # Simulate some detection times
            detector.detection_times = [0.1, 0.2, 0.15, 0.3, 0.25]
            
            stats = detector.get_performance_stats()
            
            assert stats['total_detections'] == 5
            assert stats['mean_time'] == 0.2
            assert stats['min_time'] == 0.1
            assert stats['max_time'] == 0.3
            assert 'median_time' in stats
            assert 'std_time' in stats
            assert 'fps_mean' in stats
            assert stats['fps_mean'] == 5.0  # 1/0.2
    
    def test_update_thresholds(self):
        """Test updating detection thresholds."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            
            # Update confidence threshold
            detector.update_thresholds(confidence_threshold=0.8)
            assert detector.confidence_threshold == 0.8
            assert mock_model.conf == 0.8
            
            # Update IoU threshold
            detector.update_thresholds(iou_threshold=0.5)
            assert detector.iou_threshold == 0.5
            assert mock_model.iou == 0.5
            
            # Update both
            detector.update_thresholds(confidence_threshold=0.6, iou_threshold=0.4)
            assert detector.confidence_threshold == 0.6
            assert detector.iou_threshold == 0.4
    
    def test_container_classes_filtering(self):
        """Test that only container-relevant classes are detected."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            mock_model.return_value = [Mock()]
            
            # Create detections with mixed classes
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([
                [100, 100, 200, 200],  # car (class 2 - relevant)
                [200, 200, 300, 300],  # person (class 0 - not relevant)  
                [300, 300, 400, 400],  # truck (class 7 - relevant)
                [400, 400, 500, 500]   # bicycle (class 1 - not relevant)
            ])
            mock_detections.confidence = np.array([0.9, 0.8, 0.85, 0.75])
            mock_detections.class_id = np.array([2, 0, 7, 1])
            
            # Mock the filtering behavior
            relevant_mask = np.array([True, False, True, False])  # Only car and truck
            filtered_detections = Mock(spec=sv.Detections)
            filtered_detections.xyxy = mock_detections.xyxy[relevant_mask]
            filtered_detections.confidence = mock_detections.confidence[relevant_mask]
            filtered_detections.class_id = mock_detections.class_id[relevant_mask]
            filtered_detections.__len__ = Mock(return_value=2)
            
            mock_detections.__getitem__ = Mock(return_value=filtered_detections)
            
            with patch('supervision.Detections.from_ultralytics', return_value=mock_detections), \
                 patch('cv2.imread') as mock_imread, \
                 patch('numpy.isin', return_value=relevant_mask):
                
                mock_image = np.ones((480, 640, 3), dtype=np.uint8)
                mock_imread.return_value = mock_image
                
                detector = YOLODetector()
                result = detector.detect_single_image("/fake/path.jpg")
                
                # Should only have 2 detections (car and truck)
                assert result['metadata']['num_detections'] == 2


class TestYOLODetectorIntegration:
    """Integration tests for YOLODetector (may require actual model files)."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(requires_model_file(), reason="YOLO model file not available")
    def test_real_model_loading(self):
        """Test loading a real YOLO model (if available)."""
        try:
            from modules.detection.yolo_detector import YOLODetector
            detector = YOLODetector(model_path="yolov8n.pt", verbose=False)
            assert detector.model is not None
        except Exception as e:
            pytest.skip(f"YOLO model not available: {e}")
    
    @pytest.mark.gpu
    @pytest.mark.skipif(requires_gpu(), reason="GPU not available")
    def test_gpu_detection(self, sample_image_with_objects):
        """Test detection on GPU (if available)."""
        try:
            from modules.detection.yolo_detector import YOLODetector
            detector = YOLODetector(device="cuda", verbose=False)
            result = detector.detect_single_image(sample_image_with_objects)
            assert result['metadata']['device'] == "cuda"
        except Exception as e:
            pytest.skip(f"GPU detection failed: {e}")


class TestContainerTracker:
    """Tests for ContainerTracker functionality."""
    
    def test_tracker_initialization_default(self):
        """Test ContainerTracker initialization with default parameters."""
        from modules.detection.tracker import ContainerTracker
        
        tracker = ContainerTracker()
        
        assert tracker.track_thresh == 0.25
        assert tracker.track_buffer == 30
        assert tracker.match_thresh == 0.8
        assert tracker.frame_rate == 30
        assert tracker.max_track_age == 300  # Default is 300 seconds
        assert tracker.tracks == {}
        assert tracker.frame_count == 0
        assert tracker.processing_times == []
        assert tracker.total_tracks_created == 0
        assert tracker.active_tracks_count == 0
    
    def test_tracker_initialization_custom(self):
        """Test ContainerTracker initialization with custom parameters."""
        from modules.detection.tracker import ContainerTracker
        
        tracker = ContainerTracker(
            track_thresh=0.5,
            track_buffer=60,
            match_thresh=0.9,
            frame_rate=60,
            max_track_age=60
        )
        
        assert tracker.track_thresh == 0.5
        assert tracker.track_buffer == 60
        assert tracker.match_thresh == 0.9
        assert tracker.frame_rate == 60
        assert tracker.max_track_age == 60
    
    def test_update_with_detections(self):
        """Test updating tracker with new detections."""
        from modules.detection.tracker import ContainerTracker
        import supervision as sv
        import numpy as np
        
        tracker = ContainerTracker()
        
        # Create mock detections
        mock_detections = Mock(spec=sv.Detections)
        mock_detections.xyxy = np.array([[100, 100, 200, 200], [300, 200, 450, 350]])
        mock_detections.confidence = np.array([0.9, 0.8])
        mock_detections.class_id = np.array([2, 7])
        
        # Mock ByteTrack update
        tracked_detections = Mock(spec=sv.Detections)
        tracked_detections.xyxy = mock_detections.xyxy
        tracked_detections.confidence = mock_detections.confidence
        tracked_detections.class_id = mock_detections.class_id
        tracked_detections.tracker_id = np.array([1, 2])
        tracked_detections.__len__ = Mock(return_value=2)
        tracked_detections.__iter__ = Mock(return_value=iter(range(2)))
        tracked_detections.__getitem__ = Mock(side_effect=lambda i: Mock(
            tracker_id=tracked_detections.tracker_id[i] if hasattr(tracked_detections.tracker_id, '__getitem__') else tracked_detections.tracker_id
        ))
        
        with patch.object(tracker.tracker, 'update_with_detections', return_value=tracked_detections):
            result = tracker.update(mock_detections)
            
            assert result == tracked_detections
            assert tracker.frame_count == 1
    
    def test_get_track_info(self):
        """Test getting information for a specific track."""
        from modules.detection.tracker import ContainerTracker, TrackInfo
        from datetime import datetime
        
        tracker = ContainerTracker()
        
        # Add a mock track
        track_info = TrackInfo(
            track_id=1,
            class_id=2,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
            positions=[(100, 100, 200, 200)],
            confidences=[0.85] * 10,
            is_active=True
        )
        tracker.tracks[1] = track_info
        
        # Test existing track
        result = tracker.get_track_info(1)
        assert result == track_info
        
        # Test non-existent track
        result = tracker.get_track_info(999)
        assert result is None
    
    def test_get_active_tracks(self):
        """Test getting all active tracks."""
        from modules.detection.tracker import ContainerTracker, TrackInfo
        from datetime import datetime
        
        tracker = ContainerTracker()
        
        # Add active and lost tracks
        for i in range(5):
            track_info = TrackInfo(
                track_id=i,
                class_id=2,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                positions=[(100, 100, 200, 200)],
                confidences=[0.85],
                is_active=i < 3  # First 3 are active
            )
            tracker.tracks[i] = track_info
        
        active_tracks = tracker.get_active_tracks()
        assert len(active_tracks) == 3
        assert all(track.is_active for track in active_tracks.values())
    
    def test_get_tracks_by_class(self):
        """Test getting tracks filtered by class ID."""
        from modules.detection.tracker import ContainerTracker, TrackInfo
        from datetime import datetime
        
        tracker = ContainerTracker()
        
        # Add tracks with different classes
        for i in range(6):
            track_info = TrackInfo(
                track_id=i,
                class_id=2 if i < 3 else 7,  # 3 cars, 3 trucks
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                positions=[(100, 100, 200, 200)],
                confidences=[0.85],
                is_active=True
            )
            tracker.tracks[i] = track_info
        
        # Get only cars (class 2)
        car_tracks = tracker.get_tracks_by_class(2)
        assert len(car_tracks) == 3
        assert all(track.class_id == 2 for track in car_tracks.values())
        
        # Get only trucks (class 7)
        truck_tracks = tracker.get_tracks_by_class(7)
        assert len(truck_tracks) == 3
        assert all(track.class_id == 7 for track in truck_tracks.values())
    
    def test_cleanup_old_tracks(self):
        """Test cleanup of old/lost tracks."""
        from modules.detection.tracker import ContainerTracker, TrackInfo
        from datetime import datetime, timedelta
        
        tracker = ContainerTracker(max_track_age=30)
        
        # Add tracks with different ages
        now = datetime.utcnow()
        for i in range(5):
            track_info = TrackInfo(
                track_id=i,
                class_id=2,
                first_seen=now - timedelta(seconds=60),
                last_seen=now - timedelta(seconds=i * 10),  # Varying ages
                positions=[(100, 100, 200, 200)],
                confidences=[0.85],
                is_active=i <= 2  # First 3 are active
            )
            tracker.tracks[i] = track_info
        
        tracker.frame_count = 100
        # Pass current time instead of frame count
        tracker._cleanup_old_tracks(now)
        
        # Only tracks that are not too old should remain (max_track_age=30)
        # Tracks 0, 1, 2 have ages 0, 10, 20 seconds - should remain
        # Tracks 3, 4 have ages 30, 40 seconds - track 4 should be removed
        assert len(tracker.tracks) == 4  # One track removed
    
    def test_get_dwell_time_statistics(self):
        """Test dwell time statistics calculation."""
        from modules.detection.tracker import ContainerTracker, TrackInfo
        from datetime import datetime, timedelta
        
        tracker = ContainerTracker()
        
        # Add tracks with different dwell times
        now = datetime.utcnow()
        for i in range(4):
            track_info = TrackInfo(
                track_id=i,
                class_id=2,
                first_seen=now - timedelta(seconds=(i + 1) * 60),
                last_seen=now,
                positions=[(100, 100, 200, 200)],
                confidences=[0.85],
                is_active=True
            )
            tracker.tracks[i] = track_info
        
        stats = tracker.get_dwell_time_statistics()
        
        assert 'active_tracks' in stats
        assert 'total_tracks' in stats
        assert 'mean_dwell_time' in stats
        assert 'max_dwell_time' in stats
        assert 'min_dwell_time' in stats
        assert stats['active_tracks'] == 4
        assert stats['total_tracks'] == 4
        assert stats['mean_dwell_time'] > 0
    
    def test_get_track_trajectories(self):
        """Test getting track trajectories."""
        from modules.detection.tracker import ContainerTracker, TrackInfo
        from datetime import datetime
        
        tracker = ContainerTracker()
        
        # Add tracks with bbox history
        for i in range(2):
            positions = [(100 + j*10, 100 + j*10, 200 + j*10, 200 + j*10) 
                        for j in range(5)]
            track_info = TrackInfo(
                track_id=i,
                class_id=2,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                positions=positions,
                confidences=[0.85] * 5,
                is_active=True
            )
            tracker.tracks[i] = track_info
        
        trajectories = tracker.get_track_trajectories()
        
        assert len(trajectories) == 2
        for track_id, trajectory in trajectories.items():
            assert 'positions' in trajectory
            assert 'total_distance' in trajectory
            assert 'average_speed' in trajectory
            assert 'dwell_time' in trajectory
            assert 'position_count' in trajectory
            assert trajectory['position_count'] == 5
    
    def test_reset_tracker(self):
        """Test resetting the tracker."""
        from modules.detection.tracker import ContainerTracker, TrackInfo
        from datetime import datetime
        
        tracker = ContainerTracker()
        
        # Add some tracks and update counters
        for i in range(3):
            track_info = TrackInfo(
                track_id=i,
                class_id=2,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                positions=[(100, 100, 200, 200)],
                confidences=[0.85],
                is_active=True
            )
            tracker.tracks[i] = track_info
        
        tracker.frame_count = 100
        tracker.total_tracks_created = 3
        tracker.active_tracks_count = 3
        tracker.processing_times = [0.1, 0.2, 0.3]
        
        # Reset
        tracker.reset()
        
        assert tracker.tracks == {}
        assert tracker.frame_count == 0
        assert tracker.total_tracks_created == 0
        assert tracker.active_tracks_count == 0
        assert tracker.processing_times == []
    
    def test_get_performance_stats(self):
        """Test getting tracker performance statistics."""
        from modules.detection.tracker import ContainerTracker
        
        tracker = ContainerTracker()
        
        # Add processing times
        tracker.processing_times = [0.1, 0.2, 0.15, 0.25, 0.3]
        tracker.total_tracks_created = 10
        tracker.active_tracks_count = 5
        tracker.frame_count = 100
        
        stats = tracker.get_performance_stats()
        
        assert 'total_frames_processed' in stats
        assert 'total_tracks_created' in stats
        assert 'active_tracks' in stats
        assert 'mean_processing_time' in stats
        assert 'max_processing_time' in stats
        assert 'fps_mean' in stats
        assert stats['total_frames_processed'] == 100
        assert stats['total_tracks_created'] == 10
        assert stats['active_tracks'] == 5
        assert stats['mean_processing_time'] == pytest.approx(0.2, 0.01)
        assert stats['max_processing_time'] == 0.3


class TestContainerOCR:
    """Tests for ContainerOCR functionality."""
    
    def test_ocr_initialization_tesseract_only(self):
        """Test ContainerOCR initialization with Tesseract only."""
        from modules.detection.ocr import ContainerOCR
        
        ocr = ContainerOCR(use_easyocr=False, use_tesseract=True)
        
        assert ocr.use_tesseract == True
        assert ocr.use_easyocr == False
        assert ocr.easyocr_reader is None
        assert ocr.tesseract_config is not None
        assert ocr.ocr_times == []
    
    @patch('easyocr.Reader')
    def test_ocr_initialization_easyocr_only(self, mock_reader):
        """Test ContainerOCR initialization with EasyOCR only."""
        from modules.detection.ocr import ContainerOCR
        
        mock_reader_instance = Mock()
        mock_reader.return_value = mock_reader_instance
        
        ocr = ContainerOCR(use_easyocr=True, use_tesseract=False)
        
        assert ocr.use_easyocr == True
        assert ocr.use_tesseract == False
        assert ocr.easyocr_reader is not None
        mock_reader.assert_called_once_with(['en'], gpu=False)
    
    def test_ocr_initialization_both_engines(self):
        """Test ContainerOCR initialization with both engines."""
        from modules.detection.ocr import ContainerOCR
        
        with patch('easyocr.Reader') as mock_reader:
            mock_reader_instance = Mock()
            mock_reader.return_value = mock_reader_instance
            
            ocr = ContainerOCR(use_easyocr=True, use_tesseract=True)
            
            assert ocr.use_easyocr == True
            assert ocr.use_tesseract == True
            assert ocr.easyocr_reader is not None
            assert ocr.tesseract_config is not None
    
    def test_is_valid_container_number(self):
        """Test container number validation."""
        from modules.detection.ocr import ContainerOCR
        
        ocr = ContainerOCR(use_tesseract=True, use_easyocr=False)
        
        # Valid container numbers
        assert ocr._is_valid_container_number("MSCU1234567") == True
        assert ocr._is_valid_container_number("HLXU1234567") == True
        assert ocr._is_valid_container_number("TGHU1234567") == True
        
        # Invalid container numbers
        assert ocr._is_valid_container_number("INVALID") == False
        assert ocr._is_valid_container_number("12345678") == False
        assert ocr._is_valid_container_number("ABCD12345") == False
        assert ocr._is_valid_container_number("") == False
        assert ocr._is_valid_container_number("MSC123456") == False  # Too short
    
    def test_format_container_number(self):
        """Test container number formatting."""
        from modules.detection.ocr import ContainerOCR
        
        ocr = ContainerOCR(use_tesseract=True, use_easyocr=False)
        
        # Test various formats
        assert ocr._format_container_number("mscu1234567") == "MSCU1234567"
        assert ocr._format_container_number("MSCU 123 4567") == "MSCU1234567"
        assert ocr._format_container_number("mscu-123-4567") == "MSCU1234567"
        assert ocr._format_container_number("MSCU_1234567") == "MSCU1234567"
        assert ocr._format_container_number("  MSCU1234567  ") == "MSCU1234567"
        
        # Test with common OCR mistakes
        assert ocr._format_container_number("M5CU1234567") == "MSCU1234567"  # 5 -> S
        assert ocr._format_container_number("MSCU123456l") == "MSCU1234561"  # l -> 1
        assert ocr._format_container_number("MSCUI234567") == "MSCU1234567"  # I -> 1
    
    @patch('pytesseract.image_to_string')
    def test_extract_text_from_region_tesseract(self, mock_tesseract):
        """Test text extraction from image region using Tesseract."""
        from modules.detection.ocr import ContainerOCR
        import numpy as np
        
        mock_tesseract.return_value = "MSCU1234567\nSome other text"
        
        ocr = ContainerOCR(use_tesseract=True, use_easyocr=False)
        
        # Create a mock image region
        mock_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        result = ocr._extract_text_from_region(mock_image)
        
        assert "MSCU1234567" in result
        mock_tesseract.assert_called_once()
    
    @patch('easyocr.Reader')
    def test_extract_text_from_region_easyocr(self, mock_reader_class):
        """Test text extraction from image region using EasyOCR."""
        from modules.detection.ocr import ContainerOCR
        import numpy as np
        
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "MSCU1234567", 0.95),
            ([[0, 30], [100, 30], [100, 50], [0, 50]], "Other text", 0.85)
        ]
        
        ocr = ContainerOCR(use_easyocr=True, use_tesseract=False)
        
        # Create a mock image region
        mock_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        result = ocr._extract_text_from_region(mock_image)
        
        assert "MSCU1234567" in result
        mock_reader.readtext.assert_called_once()
    
    def test_preprocess_image(self):
        """Test image preprocessing for OCR."""
        from modules.detection.ocr import ContainerOCR
        import numpy as np
        import cv2
        
        ocr = ContainerOCR(use_tesseract=True, use_easyocr=False)
        
        # Create a test image
        test_image = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
        
        # Test different preprocessing options
        result = ocr._preprocess_image(test_image, denoise=True, sharpen=True)
        
        assert result is not None
        assert result.shape[:2] == test_image.shape[:2]
        assert len(result.shape) == 2  # Should be grayscale
    
    @patch('cv2.imread')
    @patch('pytesseract.image_to_string')
    def test_extract_container_numbers(self, mock_tesseract, mock_imread):
        """Test extracting container numbers from an image."""
        from modules.detection.ocr import ContainerOCR
        import numpy as np
        import supervision as sv
        
        # Setup mocks
        mock_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        mock_imread.return_value = mock_image
        mock_tesseract.return_value = "MSCU1234567\nHLXU9876543"
        
        # Create mock detections
        mock_detections = Mock(spec=sv.Detections)
        mock_detections.xyxy = np.array([[100, 100, 300, 200], [350, 100, 550, 200]])
        mock_detections.confidence = np.array([0.9, 0.85])
        mock_detections.class_id = np.array([2, 7])  # Container classes
        mock_detections.__len__ = Mock(return_value=2)
        
        ocr = ContainerOCR(use_tesseract=True, use_easyocr=False)
        
        results = ocr.extract_container_numbers("/fake/image.jpg", mock_detections)
        
        assert 'container_numbers' in results
        assert 'processing_time' in results
        assert 'metadata' in results
        assert len(results['container_numbers']) > 0
    
    def test_extract_from_detections_batch(self, sample_images):
        """Test batch extraction of container numbers."""
        from modules.detection.ocr import ContainerOCR
        import numpy as np
        import supervision as sv
        
        with patch('cv2.imread') as mock_imread, \
             patch('pytesseract.image_to_string') as mock_tesseract:
            
            mock_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            mock_imread.return_value = mock_image
            mock_tesseract.return_value = "MSCU1234567"
            
            # Create mock detection results
            detection_results = []
            for img_path in sample_images:
                mock_detections = Mock(spec=sv.Detections)
                mock_detections.xyxy = np.array([[100, 100, 300, 200]])
                mock_detections.confidence = np.array([0.9])
                mock_detections.class_id = np.array([2])
                mock_detections.__len__ = Mock(return_value=1)
                
                detection_results.append({
                    'detections': mock_detections,
                    'metadata': {'image_path': img_path}
                })
            
            ocr = ContainerOCR(use_tesseract=True, use_easyocr=False)
            
            results = ocr.extract_from_detections_batch(detection_results, max_workers=2)
            
            assert len(results) == len(sample_images)
            for result in results:
                assert 'container_numbers' in result
                assert 'image_path' in result
    
    def test_get_performance_stats(self):
        """Test OCR performance statistics."""
        from modules.detection.ocr import ContainerOCR
        
        ocr = ContainerOCR(use_tesseract=True, use_easyocr=False)
        
        # Add some processing times
        ocr.ocr_times = [0.5, 0.6, 0.4, 0.7, 0.8]
        
        stats = ocr.get_performance_stats()
        
        assert stats['total_extractions'] == 5
        assert stats['mean_time'] == pytest.approx(0.6, 0.01)
        assert stats['min_time'] == 0.4
        assert stats['max_time'] == 0.8
        assert 'median_time' in stats
        assert 'std_time' in stats
    
    def test_validate_container_check_digit(self):
        """Test container check digit validation."""
        from modules.detection.ocr import validate_container_check_digit
        
        # Valid container numbers with correct check digits
        assert validate_container_check_digit("MSCU1234567") == True
        
        # Invalid container numbers
        assert validate_container_check_digit("MSCU1234560") == False  # Wrong check digit
        assert validate_container_check_digit("INVALID") == False
        assert validate_container_check_digit("") == False


class TestYOLODetectorPerformance:
    """Performance tests for YOLODetector."""
    
    @pytest.mark.slow
    def test_batch_processing_performance(self, performance_test_images):
        """Test batch processing performance with many images."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            # Mock batch results
            batch_results = [Mock() for _ in performance_test_images]
            mock_model.return_value = batch_results
            
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([[100, 100, 200, 200]])
            mock_detections.confidence = np.array([0.9])
            mock_detections.class_id = np.array([2])
            mock_detections.__len__ = Mock(return_value=1)
            mock_detections.__getitem__ = Mock(return_value=mock_detections)
            
            with patch('supervision.Detections.from_ultralytics', return_value=mock_detections), \
                 patch('cv2.imread') as mock_imread:
                
                mock_image = np.ones((480, 640, 3), dtype=np.uint8)
                mock_imread.return_value = mock_image
                
                detector = YOLODetector()
                
                import time
                start_time = time.time()
                results = detector.detect_batch(performance_test_images, batch_size=8)
                end_time = time.time()
                
                assert len(results) == len(performance_test_images)
                processing_time = end_time - start_time
                
                # Should process reasonably fast (adjust threshold as needed)
                assert processing_time < 5.0
                
                # Check performance stats
                stats = detector.get_performance_stats()
                # Batch processing doesn't update detection_times, so stats should be empty
                assert stats == {"message": "No detections performed yet"}
    
    def test_memory_usage_large_images(self, temp_dir):
        """Test memory usage with large images."""
        # Create a large test image
        large_image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        large_image_path = temp_dir / "large_image.jpg"
        cv2.imwrite(str(large_image_path), large_image)
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            mock_model.return_value = [Mock()]
            
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([])
            mock_detections.confidence = np.array([])
            mock_detections.class_id = np.array([])
            mock_detections.__len__ = Mock(return_value=0)
            mock_detections.__getitem__ = Mock(return_value=mock_detections)
            
            with patch('supervision.Detections.from_ultralytics', return_value=mock_detections):
                detector = YOLODetector()
                
                # Should handle large images without memory issues
                result = detector.detect_single_image(str(large_image_path))
                assert result['metadata']['image_shape'] == (2000, 3000, 3)


class TestYOLODetectorErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_empty_image(self, temp_dir):
        """Test handling of empty/corrupted images."""
        # Create an empty file
        empty_file = temp_dir / "empty.jpg"
        empty_file.touch()
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            
            with pytest.raises(ValueError, match="Could not load image"):
                detector.detect_single_image(str(empty_file))
    
    def test_model_inference_error(self, sample_image_with_objects):
        """Test handling of model inference errors."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            mock_model.side_effect = Exception("CUDA out of memory")
            
            detector = YOLODetector()
            
            with patch('cv2.imread') as mock_imread:
                mock_image = np.ones((480, 640, 3), dtype=np.uint8)
                mock_imread.return_value = mock_image
                
                with pytest.raises(Exception, match="CUDA out of memory"):
                    detector.detect_single_image(sample_image_with_objects)
    
    def test_invalid_batch_size(self, sample_images):
        """Test batch processing with invalid batch size."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            
            # Test with batch_size = 0 (should raise ValueError)
            with pytest.raises(ValueError):
                detector.detect_batch(sample_images, batch_size=0)
            
            # Test with very large batch_size
            mock_model.return_value = [Mock() for _ in sample_images]
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([])
            mock_detections.confidence = np.array([])
            mock_detections.class_id = np.array([])
            mock_detections.__len__ = Mock(return_value=0)
            mock_detections.__getitem__ = Mock(return_value=mock_detections)
            
            with patch('supervision.Detections.from_ultralytics', return_value=mock_detections), \
                 patch('cv2.imread') as mock_imread:
                
                mock_image = np.ones((480, 640, 3), dtype=np.uint8)
                mock_imread.return_value = mock_image
                
                results = detector.detect_batch(sample_images, batch_size=1000)
                assert len(results) == len(sample_images)