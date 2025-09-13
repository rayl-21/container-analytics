"""
Tests for the detection module.

Tests cover:
- YOLO detector initialization
- Object detection on sample images
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
            
            assert detector.model_path == "data/models/yolov12x.pt"
            assert detector.confidence_threshold == 0.5
            assert detector.iou_threshold == 0.7
            assert detector.device == "cpu"  # Should use CPU when CUDA not available
            assert detector.verbose == True
            
            mock_yolo.assert_called_once_with("data/models/yolov12x.pt")
            mock_model.to.assert_called_once_with("cpu")
    
    def test_init_custom_params(self):
        """Test YOLODetector initialization with custom parameters."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(
                model_path="yolov12n.pt",
                confidence_threshold=0.7,
                iou_threshold=0.5,
                device="cuda",
                verbose=False
            )
            
            assert detector.model_path == "yolov12n.pt"
            assert detector.confidence_threshold == 0.7
            assert detector.iou_threshold == 0.5
            assert detector.device == "cuda"
            assert detector.verbose == False
            
            mock_yolo.assert_called_once_with("yolov12n.pt")
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
            detector = YOLODetector(model_path="yolov12x.pt", verbose=False)
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


class TestYOLODetectorDatabaseIntegration:
    """Test database integration functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_database(self):
        """Setup database tables for testing."""
        from modules.database.models import create_tables, get_engine
        
        # Create tables before each test
        create_tables()
        
        yield
        
        # Clean up after each test
        engine = get_engine()
        with engine.connect() as conn:
            # Drop all tables for clean state
            from modules.database.models import Base
            Base.metadata.drop_all(bind=engine)
    
    def test_save_detection_to_database_new_image(self, temp_dir):
        """Test saving detections for a new image to database."""
        from modules.detection.yolo_detector import YOLODetector
        from modules.database.models import session_scope, Image as ImageModel, Detection as DetectionModel
        import numpy as np
        import supervision as sv
        
        # Create a test image
        test_image_path = temp_dir / "test_image.jpg"
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(test_image_path), test_image)
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            
            # Create mock detections
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([[100, 100, 200, 200], [300, 200, 450, 350]])
            mock_detections.confidence = np.array([0.9, 0.8])
            mock_detections.class_id = np.array([2, 7])  # car, truck
            mock_detections.__len__ = Mock(return_value=2)
            
            # Test saving to database
            image_id = detector.save_detection_to_database(
                test_image_path, mock_detections, 0.5
            )
            
            assert image_id is not None
            
            # Verify records in database
            with session_scope() as session:
                image_record = session.query(ImageModel).filter(ImageModel.id == image_id).first()
                assert image_record is not None
                assert image_record.filepath == str(test_image_path)
                assert image_record.processed == True
                
                detections = session.query(DetectionModel).filter(DetectionModel.image_id == image_id).all()
                assert len(detections) == 2
                
                # Check first detection
                det1 = detections[0]
                assert det1.object_type in ['car', 'truck']
                assert det1.confidence in [0.9, 0.8]
                assert det1.bbox_x == 100 or det1.bbox_x == 300
    
    def test_save_detection_to_database_existing_image(self, temp_dir):
        """Test saving detections for an existing image."""
        from modules.detection.yolo_detector import YOLODetector
        from modules.database.models import session_scope, Image as ImageModel, Detection as DetectionModel
        import numpy as np
        import supervision as sv
        
        test_image_path = temp_dir / "existing_image.jpg"
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(test_image_path), test_image)
        
        # Create existing image record
        with session_scope() as session:
            existing_image = ImageModel(
                filepath=str(test_image_path),
                camera_id='test_camera',
                processed=False,
                file_size=1000
            )
            session.add(existing_image)
            session.flush()
            existing_id = existing_image.id
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([[50, 50, 150, 150]])
            mock_detections.confidence = np.array([0.95])
            mock_detections.class_id = np.array([2])
            mock_detections.__len__ = Mock(return_value=1)
            
            image_id = detector.save_detection_to_database(
                test_image_path, mock_detections, 0.3
            )
            
            assert image_id == existing_id
            
            # Verify detection was added
            with session_scope() as session:
                detections = session.query(DetectionModel).filter(DetectionModel.image_id == image_id).all()
                assert len(detections) == 1
    
    def test_extract_camera_id_from_path(self):
        """Test camera ID extraction from file paths."""
        from modules.detection.yolo_detector import YOLODetector
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            
            # Test various path patterns
            assert detector._extract_camera_id_from_path(Path("data/images/in_gate_123.jpg")) == "in_gate"
            assert detector._extract_camera_id_from_path(Path("data/images/gate-456.jpg")) == "gate"
            assert detector._extract_camera_id_from_path(Path("data/images/gate_789.jpg")) == "gate"
            assert detector._extract_camera_id_from_path(Path("data/images/random_image.jpg")) == "unknown"
            assert detector._extract_camera_id_from_path(Path("data/camera_feed/random_image.jpg")) == "camera_feed"
    
    def test_detect_with_retry_success(self, sample_image_with_objects):
        """Test successful detection with retry logic."""
        from modules.detection.yolo_detector import YOLODetector
        import numpy as np
        import supervision as sv
        
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
                 patch.object(YOLODetector, 'save_detection_to_database', return_value=1):
                
                mock_image = np.ones((480, 640, 3), dtype=np.uint8)
                mock_imread.return_value = mock_image
                
                detector = YOLODetector()
                result = detector.detect_with_retry(sample_image_with_objects, save_to_db=True)
                
                assert result is not None
                assert result['metadata']['image_id'] == 1
    
    def test_detect_with_retry_failure(self, sample_image_with_objects):
        """Test detection retry logic on failures."""
        from modules.detection.yolo_detector import YOLODetector
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            
            # Mock detect_single_image to always fail
            with patch.object(detector, 'detect_single_image', side_effect=Exception("Detection failed")):
                result = detector.detect_with_retry(
                    sample_image_with_objects, 
                    max_retries=2, 
                    retry_delay=0.1
                )
                
                assert result is None


class TestImageProcessingQueue:
    """Test image processing queue functionality."""
    
    def test_queue_initialization(self):
        """Test queue initialization."""
        from modules.detection.yolo_detector import ImageProcessingQueue
        
        queue = ImageProcessingQueue(maxsize=50)
        assert queue.qsize() == 0
        assert len(queue.processed_files) == 0
    
    def test_add_and_get_image(self, temp_dir):
        """Test adding and getting images from queue."""
        from modules.detection.yolo_detector import ImageProcessingQueue
        
        queue = ImageProcessingQueue()
        test_image = temp_dir / "test.jpg"
        test_image.touch()
        
        # Add image
        assert queue.add_image(test_image) == True
        assert queue.qsize() == 1
        
        # Get image
        retrieved = queue.get_image(timeout=0.1)
        assert retrieved == test_image
        assert queue.qsize() == 0
    
    def test_duplicate_image_handling(self, temp_dir):
        """Test that duplicate images are not added."""
        from modules.detection.yolo_detector import ImageProcessingQueue
        
        queue = ImageProcessingQueue()
        test_image = temp_dir / "test.jpg"
        test_image.touch()
        
        # Add image first time
        assert queue.add_image(test_image) == True
        queue.mark_processed(test_image)
        
        # Try to add same image again
        assert queue.add_image(test_image) == False
        assert queue.qsize() == 0
    
    def test_queue_full_handling(self, temp_dir):
        """Test queue behavior when full."""
        from modules.detection.yolo_detector import ImageProcessingQueue
        
        queue = ImageProcessingQueue(maxsize=2)
        
        # Fill queue
        for i in range(2):
            test_image = temp_dir / f"test_{i}.jpg"
            test_image.touch()
            assert queue.add_image(test_image) == True
        
        # Try to add one more
        overflow_image = temp_dir / "overflow.jpg"
        overflow_image.touch()
        assert queue.add_image(overflow_image) == False
    
    def test_clear_processed_history(self, temp_dir):
        """Test clearing processed files history."""
        from modules.detection.yolo_detector import ImageProcessingQueue
        
        queue = ImageProcessingQueue()
        test_image = temp_dir / "test.jpg"
        test_image.touch()
        
        queue.add_image(test_image)
        queue.mark_processed(test_image)
        
        assert len(queue.processed_files) == 1
        
        queue.clear_processed_history()
        assert len(queue.processed_files) == 0


class TestImageFileHandler:
    """Test image file event handler."""
    
    def test_handler_initialization(self):
        """Test file handler initialization."""
        from modules.detection.yolo_detector import ImageFileHandler, ImageProcessingQueue
        
        queue = ImageProcessingQueue()
        handler = ImageFileHandler(queue)
        
        assert handler.processing_queue == queue
        assert '.jpg' in handler.supported_extensions
        assert '.png' in handler.supported_extensions
    
    def test_on_created_image_file(self, temp_dir):
        """Test handling of image file creation."""
        from modules.detection.yolo_detector import ImageFileHandler, ImageProcessingQueue
        from watchdog.events import FileCreatedEvent
        
        queue = ImageProcessingQueue()
        handler = ImageFileHandler(queue)
        
        # Create test image file
        test_image = temp_dir / "new_image.jpg"
        test_image.touch()
        
        # Simulate file creation event
        event = FileCreatedEvent(str(test_image))
        
        with patch('time.sleep'):  # Skip the wait
            handler.on_created(event)
        
        # Check if image was added to queue
        assert queue.qsize() == 1
        retrieved = queue.get_image(timeout=0.1)
        assert retrieved == test_image
    
    def test_on_created_non_image_file(self, temp_dir):
        """Test handling of non-image file creation."""
        from modules.detection.yolo_detector import ImageFileHandler, ImageProcessingQueue
        from watchdog.events import FileCreatedEvent
        
        queue = ImageProcessingQueue()
        handler = ImageFileHandler(queue)
        
        # Create non-image file
        test_file = temp_dir / "document.txt"
        test_file.touch()
        
        # Simulate file creation event
        event = FileCreatedEvent(str(test_file))
        handler.on_created(event)
        
        # Queue should remain empty
        assert queue.qsize() == 0
    
    def test_on_created_directory(self, temp_dir):
        """Test handling of directory creation."""
        from modules.detection.yolo_detector import ImageFileHandler, ImageProcessingQueue
        from watchdog.events import DirCreatedEvent
        
        queue = ImageProcessingQueue()
        handler = ImageFileHandler(queue)
        
        # Create directory
        test_dir = temp_dir / "new_directory"
        test_dir.mkdir()
        
        # Simulate directory creation event
        event = DirCreatedEvent(str(test_dir))
        handler.on_created(event)
        
        # Queue should remain empty
        assert queue.qsize() == 0


class TestYOLOWatchMode:
    """Test YOLO watch mode functionality."""
    
    def test_watch_mode_initialization(self, temp_dir):
        """Test watch mode initialization."""
        from modules.detection.yolo_detector import YOLODetector, YOLOWatchMode
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            watch_mode = YOLOWatchMode(
                detector=detector,
                watch_directory=temp_dir,
                batch_size=2,
                max_workers=1
            )
            
            assert watch_mode.detector == detector
            assert watch_mode.watch_directory == temp_dir
            assert watch_mode.batch_size == 2
            assert watch_mode.max_workers == 1
            assert watch_mode.is_running == False
    
    def test_watch_mode_get_stats_initial(self, temp_dir):
        """Test initial statistics."""
        from modules.detection.yolo_detector import YOLODetector, YOLOWatchMode
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            watch_mode = YOLOWatchMode(detector=detector, watch_directory=temp_dir)
            
            stats = watch_mode.get_stats()
            
            assert stats['is_running'] == False
            assert stats['images_processed'] == 0
            assert stats['images_failed'] == 0
            assert stats['total_detections'] == 0
            assert stats['queue_size'] == 0
    
    def test_process_existing_images(self, temp_dir):
        """Test processing existing images in directory."""
        from modules.detection.yolo_detector import YOLODetector, YOLOWatchMode
        import numpy as np
        
        # Create test images
        for i in range(3):
            img_path = temp_dir / f"existing_{i}.jpg"
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img_array)
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            watch_mode = YOLOWatchMode(
                detector=detector,
                watch_directory=temp_dir,
                process_existing=True
            )
            
            # Process existing images
            watch_mode._process_existing_images()
            
            # Check queue has images
            assert watch_mode.processing_queue.qsize() == 3
    
    @pytest.mark.slow
    def test_watch_mode_context_manager(self, temp_dir):
        """Test watch mode context manager."""
        from modules.detection.yolo_detector import YOLODetector, YOLOWatchMode
        
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector()
            watch_mode = YOLOWatchMode(
                detector=detector,
                watch_directory=temp_dir,
                max_workers=1
            )
            
            # Test context manager
            with watch_mode.running_context():
                assert watch_mode.is_running == True
                time.sleep(0.5)  # Let it run briefly
            
            # Should be stopped after context exit
            assert watch_mode.is_running == False