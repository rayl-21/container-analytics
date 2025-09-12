"""
Integration tests for the enhanced YOLO detection pipeline with batch processing.

This module tests the complete detection pipeline including:
- Batch processing of unprocessed images from database
- Database persistence for all detections
- Container tracking integration
- Detection result caching
- Statistics aggregation
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import shutil
import supervision as sv

from modules.detection.yolo_detector import YOLODetector
from modules.database.models import session_scope, Image as ImageModel, Detection as DetectionModel, Container
from modules.database.queries import insert_image, get_unprocessed_images, mark_image_processed


class TestBatchProcessingIntegration:
    """Test the enhanced batch processing pipeline."""
    
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
            from modules.database.models import Base
            Base.metadata.drop_all(bind=engine)
    
    @pytest.fixture
    def temp_image_dir(self):
        """Create temporary directory with test images."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test images
        test_images = []
        for i in range(5):
            image_path = temp_dir / f"test_image_{i}.jpg"
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), test_image)
            test_images.append(image_path)
        
        yield temp_dir, test_images
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_yolo_detector(self):
        """Create a mocked YOLO detector."""
        with patch('modules.detection.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            # Mock detection results
            mock_result = Mock()
            mock_model.return_value = [mock_result]
            
            # Create mock detections
            mock_detections = Mock(spec=sv.Detections)
            mock_detections.xyxy = np.array([[100, 100, 200, 200], [300, 200, 450, 350]])
            mock_detections.confidence = np.array([0.9, 0.8])
            mock_detections.class_id = np.array([2, 7])  # car, truck
            mock_detections.__len__ = Mock(return_value=2)
            mock_detections.__getitem__ = Mock(return_value=mock_detections)
            
            with patch('supervision.Detections.from_ultralytics', return_value=mock_detections):
                detector = YOLODetector()
                yield detector, mock_detections
    
    def test_batch_process_images_basic_functionality(self, temp_image_dir, mock_yolo_detector):
        """Test basic batch processing functionality."""
        temp_dir, test_images = temp_image_dir
        detector, mock_detections = mock_yolo_detector
        
        # Insert test images into database as unprocessed
        image_ids = []
        for image_path in test_images:
            image_id = insert_image(
                filepath=str(image_path),
                camera_id="test_camera",
                file_size=image_path.stat().st_size,
                processed=False
            )
            image_ids.append(image_id)
        
        # Mock cv2.imread to return valid images
        with patch('cv2.imread') as mock_imread:
            mock_image = np.ones((480, 640, 3), dtype=np.uint8)
            mock_imread.return_value = mock_image
            
            # Run batch processing
            stats = detector.batch_process_images(
                limit=5,
                batch_size=2,
                save_to_database=True,
                update_container_tracking=False
            )
        
        # Verify statistics
        assert stats['total_images_found'] == 5
        assert stats['total_images_processed'] == 5
        assert stats['total_detections'] == 10  # 2 detections per image * 5 images
        assert stats['database_saves'] == 10
        assert len(stats['errors']) == 0
        
        # Verify database persistence
        with session_scope() as session:
            # Check that all images are marked as processed
            processed_images = session.query(ImageModel).filter(ImageModel.processed == True).all()
            assert len(processed_images) == 5
            
            # Check that detections were saved
            detections = session.query(DetectionModel).all()
            assert len(detections) == 10
            
            # Verify detection data
            for detection in detections:
                assert detection.object_type in ['car', 'truck']
                assert detection.confidence in [0.9, 0.8]
                assert detection.bbox_x >= 0
                assert detection.bbox_y >= 0
    
    def test_batch_process_images_with_caching(self, temp_image_dir, mock_yolo_detector):
        """Test batch processing with result caching enabled."""
        temp_dir, test_images = temp_image_dir
        detector, mock_detections = mock_yolo_detector
        
        # Insert one test image
        image_id = insert_image(
            filepath=str(test_images[0]),
            camera_id="test_camera", 
            file_size=test_images[0].stat().st_size,
            processed=False
        )
        
        with patch('cv2.imread') as mock_imread:
            mock_image = np.ones((480, 640, 3), dtype=np.uint8)
            mock_imread.return_value = mock_image
            
            # First run - should process image
            stats1 = detector.batch_process_images(
                limit=1,
                batch_size=1,
                enable_caching=True
            )
            
            # Reset image to unprocessed for second run
            mark_image_processed(image_id)
            with session_scope() as session:
                image = session.query(ImageModel).filter(ImageModel.id == image_id).first()
                image.processed = False
            
            # Second run - should use cache (though in this test it won't since we reset)
            stats2 = detector.batch_process_images(
                limit=1,
                batch_size=1,
                enable_caching=True
            )
        
        assert stats1['total_images_processed'] == 1
        assert stats2['total_images_processed'] == 1
    
    def test_batch_process_images_error_handling(self, temp_image_dir, mock_yolo_detector):
        """Test error handling in batch processing."""
        temp_dir, test_images = temp_image_dir
        detector, mock_detections = mock_yolo_detector
        
        # Insert test images, including one with invalid path
        image_ids = []
        for i, image_path in enumerate(test_images[:3]):
            if i == 1:  # Make middle image have invalid path
                image_id = insert_image(
                    filepath="/invalid/path/image.jpg",
                    camera_id="test_camera",
                    file_size=1000,
                    processed=False
                )
            else:
                image_id = insert_image(
                    filepath=str(image_path),
                    camera_id="test_camera",
                    file_size=image_path.stat().st_size,
                    processed=False
                )
            image_ids.append(image_id)
        
        with patch('cv2.imread') as mock_imread:
            def side_effect(path):
                if "/invalid/path/" in str(path):
                    return None  # Simulate failed image load
                return np.ones((480, 640, 3), dtype=np.uint8)
            
            mock_imread.side_effect = side_effect
            
            # Run batch processing
            stats = detector.batch_process_images(
                limit=3,
                batch_size=2,
                save_to_database=True
            )
        
        # Should process 2 valid images, skip 1 invalid
        assert stats['total_images_found'] == 3
        assert stats['total_images_processed'] == 2
        assert len(stats['errors']) >= 1  # Should have error for invalid image
    
    def test_batch_process_images_no_unprocessed_images(self, mock_yolo_detector):
        """Test behavior when no unprocessed images exist."""
        detector, mock_detections = mock_yolo_detector
        
        # Run batch processing with no images in database
        stats = detector.batch_process_images(limit=10)
        
        assert stats['total_images_found'] == 0
        assert stats['total_images_processed'] == 0
        assert stats['total_detections'] == 0
    
    def test_batch_process_images_detection_breakdown(self, temp_image_dir, mock_yolo_detector):
        """Test detection breakdown statistics."""
        temp_dir, test_images = temp_image_dir
        detector, mock_detections = mock_yolo_detector
        
        # Insert test image
        insert_image(
            filepath=str(test_images[0]),
            camera_id="test_camera",
            file_size=test_images[0].stat().st_size,
            processed=False
        )
        
        with patch('cv2.imread') as mock_imread:
            mock_image = np.ones((480, 640, 3), dtype=np.uint8)
            mock_imread.return_value = mock_image
            
            stats = detector.batch_process_images(limit=1, batch_size=1)
        
        # Verify detection breakdown
        assert 'detection_breakdown' in stats
        breakdown = stats['detection_breakdown']
        assert 'car' in breakdown or 'truck' in breakdown
        assert sum(breakdown.values()) == 2  # 2 detections total


class TestDetectionStatistics:
    """Test detection statistics aggregation functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_database(self):
        """Setup database tables for testing."""
        from modules.database.models import create_tables, get_engine
        
        create_tables()
        yield
        
        engine = get_engine()
        with engine.connect() as conn:
            from modules.database.models import Base
            Base.metadata.drop_all(bind=engine)
    
    def test_get_detection_statistics_basic(self, mock_yolo_detector):
        """Test basic detection statistics generation."""
        detector, mock_detections = mock_yolo_detector
        
        # Add some detection times for performance metrics
        detector.detection_times = [0.1, 0.2, 0.15, 0.3, 0.25]
        
        # Mock the database query functions
        with patch('modules.database.queries.get_detection_summary') as mock_summary, \
             patch('modules.database.queries.get_container_statistics') as mock_container, \
             patch('modules.database.queries.get_throughput_data') as mock_throughput, \
             patch('modules.database.queries.get_recent_detections') as mock_recent:
            
            mock_summary.return_value = {'total_detections': 100, 'avg_confidence': 0.85}
            mock_container.return_value = {'total_containers': 50}
            mock_throughput.return_value = {'hourly_avg': 10.5}
            mock_recent.return_value = {'recent_count': 25}
            
            stats = detector.get_detection_statistics(days=7)
        
        # Verify structure
        assert 'period' in stats
        assert 'detection_summary' in stats
        assert 'container_stats' in stats
        assert 'throughput_data' in stats
        assert 'recent_activity' in stats
        assert 'performance_metrics' in stats
        
        # Verify performance metrics calculation
        perf = stats['performance_metrics']
        assert perf['avg_detection_time'] == 0.2  # (0.1+0.2+0.15+0.3+0.25)/5
        assert perf['images_processed'] == 5
        assert perf['total_processing_time'] == 1.0  # sum of detection times
    
    def test_get_detection_statistics_error_handling(self, mock_yolo_detector):
        """Test error handling in statistics generation."""
        detector, mock_detections = mock_yolo_detector
        
        # Mock database queries to raise exceptions
        with patch('modules.database.queries.get_detection_summary', side_effect=Exception("DB Error")), \
             patch('modules.database.queries.get_container_statistics', side_effect=Exception("DB Error")), \
             patch('modules.database.queries.get_throughput_data', side_effect=Exception("DB Error")), \
             patch('modules.database.queries.get_recent_detections', side_effect=Exception("DB Error")):
            
            stats = detector.get_detection_statistics(days=7)
        
        # Should still return a valid structure with empty data
        assert 'detection_summary' in stats
        assert 'container_stats' in stats
        assert 'throughput_data' in stats
        assert 'recent_activity' in stats
        assert stats['detection_summary'] == {}  # Empty due to error


class TestContainerTrackingIntegration:
    """Test container tracking integration (placeholder for OCR integration)."""
    
    @pytest.fixture(autouse=True)
    def setup_database(self):
        """Setup database tables for testing."""
        from modules.database.models import create_tables, get_engine
        
        create_tables()
        yield
        
        engine = get_engine()
        with engine.connect() as conn:
            from modules.database.models import Base
            Base.metadata.drop_all(bind=engine)
    
    def test_container_tracking_integration_placeholder(self, mock_yolo_detector):
        """Test that container tracking integration is ready for OCR module."""
        detector, mock_detections = mock_yolo_detector
        
        # This test verifies that the batch processing method has the
        # infrastructure in place for container tracking, even though
        # OCR is not yet implemented
        
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(tmp_file.name, test_image)
            
            # Insert image into database
            image_id = insert_image(
                filepath=tmp_file.name,
                camera_id="test_camera",
                file_size=1000,
                processed=False
            )
        
        with patch('cv2.imread') as mock_imread:
            mock_image = np.ones((480, 640, 3), dtype=np.uint8)
            mock_imread.return_value = mock_image
            
            # Run batch processing with container tracking enabled
            stats = detector.batch_process_images(
                limit=1,
                batch_size=1,
                update_container_tracking=True  # This should not fail
            )
        
        # Should process successfully even without OCR
        assert stats['total_images_processed'] == 1
        assert len(stats['errors']) == 0 or not any('Container tracking' in str(error) for error in stats['errors'])
        
        # Clean up temp file
        Path(tmp_file.name).unlink()


class TestCacheManagement:
    """Test detection result caching functionality."""
    
    def test_clear_detection_cache(self, mock_yolo_detector):
        """Test cache clearing functionality."""
        detector, mock_detections = mock_yolo_detector
        
        # Simulate many detection times
        detector.detection_times = list(range(1200))  # > 1000 measurements
        
        # Clear cache should reduce the list
        detector.clear_detection_cache()
        
        # Should keep only last 100 measurements
        assert len(detector.detection_times) == 100
        assert detector.detection_times[0] == 1100  # Should start from 1100
        assert detector.detection_times[-1] == 1199  # Should end at 1199
    
    def test_clear_detection_cache_small_list(self, mock_yolo_detector):
        """Test cache clearing with small detection list."""
        detector, mock_detections = mock_yolo_detector
        
        # Small list should not be affected
        detector.detection_times = [0.1, 0.2, 0.3]
        original_length = len(detector.detection_times)
        
        detector.clear_detection_cache()
        
        # Should remain unchanged
        assert len(detector.detection_times) == original_length


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Test the complete detection pipeline integration."""
    
    @pytest.fixture(autouse=True)
    def setup_database(self):
        """Setup database tables for testing."""
        from modules.database.models import create_tables, get_engine
        
        create_tables()
        yield
        
        engine = get_engine()
        with engine.connect() as conn:
            from modules.database.models import Base
            Base.metadata.drop_all(bind=engine)
    
    def test_end_to_end_batch_processing_pipeline(self, mock_yolo_detector):
        """Test complete end-to-end pipeline."""
        detector, mock_detections = mock_yolo_detector
        
        # Create temporary images and insert into database
        temp_dir = Path(tempfile.mkdtemp())
        try:
            image_paths = []
            for i in range(3):
                image_path = temp_dir / f"e2e_test_{i}.jpg"
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(str(image_path), test_image)
                
                insert_image(
                    filepath=str(image_path),
                    camera_id=f"camera_{i % 2}",  # Alternate cameras
                    file_size=image_path.stat().st_size,
                    processed=False
                )
                image_paths.append(image_path)
            
            with patch('cv2.imread') as mock_imread:
                mock_image = np.ones((480, 640, 3), dtype=np.uint8)
                mock_imread.return_value = mock_image
                
                # Run complete pipeline
                stats = detector.batch_process_images(
                    limit=10,
                    batch_size=2,
                    save_to_database=True,
                    update_container_tracking=True,
                    enable_caching=True
                )
            
            # Verify end-to-end results
            assert stats['total_images_found'] == 3
            assert stats['total_images_processed'] == 3
            assert stats['total_detections'] == 6  # 2 per image
            assert stats['processing_time'] > 0
            assert stats['avg_time_per_image'] > 0
            
            # Verify database state
            with session_scope() as session:
                processed_images = session.query(ImageModel).filter(ImageModel.processed == True).all()
                assert len(processed_images) == 3
                
                all_detections = session.query(DetectionModel).all()
                assert len(all_detections) == 6
            
            # Test statistics generation
            stats_result = detector.get_detection_statistics(days=1)
            assert 'performance_metrics' in stats_result
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)