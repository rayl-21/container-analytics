"""
pytest configuration and fixtures for Container Analytics tests.

This file contains shared fixtures and configuration used across all test modules.
"""

import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

import pytest
import numpy as np
import cv2
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import supervision as sv

from modules.database.models import Base, Image as ImageModel, Detection, Container, Metric
from modules.downloader.selenium_client import DrayDogDownloader
from modules.detection.yolo_detector import YOLODetector
from modules.analytics.metrics import ContainerMetrics


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def test_db_engine():
    """Create an in-memory SQLite database engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="function")
def test_db_session(test_db_engine):
    """Create a database session for testing."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def sample_image_data(test_db_session):
    """Create sample image records in the test database."""
    images = []
    for i in range(5):
        image = ImageModel(
            timestamp=datetime.utcnow() - timedelta(hours=i),
            filepath=f"/test/images/image_{i}.jpg",
            camera_id=f"camera_{i % 2}",  # Alternate between camera_0 and camera_1
            processed=i % 2 == 0,  # Half processed, half not
            file_size=1024 * (i + 1)
        )
        test_db_session.add(image)
        images.append(image)
    
    test_db_session.commit()
    return images


@pytest.fixture(scope="function")
def sample_detection_data(test_db_session, sample_image_data):
    """Create sample detection records in the test database."""
    detections = []
    for i, image in enumerate(sample_image_data):
        # Create 2-3 detections per image
        for j in range(2 + i % 2):
            detection = Detection(
                image_id=image.id,
                object_type=["container", "truck", "car"][j % 3],
                confidence=0.7 + (j * 0.1),
                bbox_x=0.1 + (j * 0.2),
                bbox_y=0.1 + (j * 0.15),
                bbox_width=0.3 + (j * 0.1),
                bbox_height=0.2 + (j * 0.05),
                tracking_id=100 + j
            )
            test_db_session.add(detection)
            detections.append(detection)
    
    test_db_session.commit()
    return detections


@pytest.fixture(scope="function")
def sample_container_data(test_db_session):
    """Create sample container records in the test database."""
    containers = []
    base_time = datetime.utcnow() - timedelta(days=1)
    
    for i in range(10):
        first_seen = base_time + timedelta(hours=i)
        last_seen = first_seen + timedelta(hours=2 + i)  # Variable dwell times
        
        container = Container(
            container_number=f"CONT{1000 + i}",
            first_seen=first_seen,
            last_seen=last_seen,
            total_detections=10 + i,
            avg_confidence=0.8 + (i * 0.02),
            status="departed" if i < 7 else "active",
            camera_id=f"camera_{i % 2}"
        )
        container.calculate_dwell_time()
        test_db_session.add(container)
        containers.append(container)
    
    test_db_session.commit()
    return containers


@pytest.fixture(scope="function") 
def sample_metrics_data(test_db_session):
    """Create sample metrics records in the test database."""
    metrics = []
    base_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Create 48 hours of metrics (2 days)
    for day in range(2):
        for hour in range(24):
            metric_date = base_date - timedelta(days=day)
            metric = Metric(
                date=metric_date,
                hour=hour,
                throughput=5 + (hour % 8),  # Simulate peak hours
                avg_dwell_time=2.0 + (hour * 0.1),
                container_count=10 + (hour % 5),
                total_detections=50 + (hour * 2),
                avg_confidence=0.8 + (hour * 0.005),
                camera_id=f"camera_{day % 2}"
            )
            test_db_session.add(metric)
            metrics.append(metric)
    
    test_db_session.commit()
    return metrics


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_images(temp_dir):
    """Create sample test images."""
    images = []
    for i in range(3):
        # Create a simple test image
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = temp_dir / f"test_image_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        images.append(str(img_path))
    
    return images


@pytest.fixture(scope="function")
def sample_image_with_objects(temp_dir):
    """Create a sample image with mock objects for detection testing."""
    # Create a simple image with some rectangles to simulate objects
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
    
    # Draw some rectangles that could represent containers/trucks
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(img, (300, 200), (450, 350), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(img, (500, 50), (600, 150), (0, 0, 255), -1)   # Red rectangle
    
    img_path = temp_dir / "image_with_objects.jpg"
    cv2.imwrite(str(img_path), img)
    return str(img_path)


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_selenium_driver():
    """Mock Selenium WebDriver for downloader tests."""
    driver = Mock()
    driver.get = Mock()
    driver.find_element = Mock()
    driver.find_elements = Mock(return_value=[])
    driver.execute_script = Mock(return_value=[])
    driver.quit = Mock()
    return driver


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for detection tests."""
    model = Mock()
    model.conf = 0.5
    model.iou = 0.7
    model.to = Mock()
    
    # Mock detection results
    mock_results = Mock()
    mock_results.boxes = Mock()
    mock_results.boxes.data = np.array([
        [100, 100, 200, 200, 0.9, 2],  # [x1, y1, x2, y2, conf, class]
        [300, 200, 450, 350, 0.8, 7],
    ])
    
    model.return_value = [mock_results]
    return model


@pytest.fixture
def mock_supervision_detections():
    """Mock supervision Detections object."""
    detections = Mock(spec=sv.Detections)
    detections.xyxy = np.array([[100, 100, 200, 200], [300, 200, 450, 350]])
    detections.confidence = np.array([0.9, 0.8])
    detections.class_id = np.array([2, 7])  # car, truck
    detections.__len__ = Mock(return_value=2)
    detections.__getitem__ = Mock(return_value=detections)
    return detections


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for image downloading."""
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_content = Mock(return_value=[b'fake_image_data' * 100])
    response.headers = {'content-length': '1024'}
    return response


# ============================================================================
# Component Fixtures
# ============================================================================

@pytest.fixture
def downloader_config():
    """Configuration for DrayDogDownloader tests."""
    return {
        'download_dir': None,  # Will be set to temp_dir in tests
        'headless': True,
        'max_retries': 2,
        'retry_delay': 0.1,  # Fast retries for tests
        'timeout': 5
    }


@pytest.fixture
def detector_config():
    """Configuration for YOLODetector tests."""
    return {
        'model_path': 'yolov12n.pt',  # Use nano model for faster tests
        'confidence_threshold': 0.5,
        'iou_threshold': 0.7,
        'device': 'cpu',  # Force CPU for consistent test environment
        'verbose': False
    }


@pytest.fixture
def sample_image_metadata():
    """Sample image metadata for downloader tests."""
    return [
        {
            'url': 'https://cdn.draydog.com/test/image1.jpg',
            'filename': '2025-01-15T10-30-00-in_gate.jpg',
            'timestamp': '2025-01-15T10:30:00',
            'captureTime': '10:30:00',
            'streamName': 'in_gate',
            'index': 0
        },
        {
            'url': 'https://cdn.draydog.com/test/image2.jpg', 
            'filename': '2025-01-15T10-35-00-in_gate.jpg',
            'timestamp': '2025-01-15T10:35:00',
            'captureTime': '10:35:00',
            'streamName': 'in_gate',
            'index': 1
        }
    ]


@pytest.fixture
def sample_detection_results():
    """Sample detection results for testing."""
    return {
        'detections': mock_supervision_detections(),
        'metadata': {
            'image_path': '/test/image.jpg',
            'processing_time': 0.5,
            'num_detections': 2,
            'image_shape': (480, 640, 3),
            'model_confidence': 0.5,
            'device': 'cpu'
        }
    }


# ============================================================================
# Date/Time Fixtures
# ============================================================================

@pytest.fixture
def date_ranges():
    """Provide various date ranges for testing."""
    now = datetime.utcnow()
    return {
        'today': (now.replace(hour=0, minute=0, second=0, microsecond=0), now),
        'yesterday': (now - timedelta(days=1), now - timedelta(days=1) + timedelta(hours=23, minutes=59)),
        'last_week': (now - timedelta(days=7), now),
        'last_month': (now - timedelta(days=30), now),
        'custom_range': (datetime(2025, 1, 1), datetime(2025, 1, 31))
    }


# ============================================================================
# Performance Test Fixtures
# ============================================================================

@pytest.fixture
def performance_test_images(temp_dir):
    """Create a larger set of test images for performance testing."""
    images = []
    for i in range(20):  # Create 20 test images
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = temp_dir / f"perf_test_image_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img_array)
        images.append(str(img_path))
    
    return images


# ============================================================================
# Environment Setup/Teardown
# ============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_dir):
    """Set up test environment variables."""
    # Set test database URL
    monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')
    
    # Set test data directories
    monkeypatch.setenv('TEST_DATA_DIR', str(temp_dir))
    
    # Disable logging during tests (unless explicitly testing logging)
    import logging
    logging.disable(logging.CRITICAL)
    
    yield
    
    # Re-enable logging
    logging.disable(logging.NOTSET)


# ============================================================================
# Skip Conditions
# ============================================================================

def requires_gpu():
    """Skip test if GPU is not available."""
    try:
        import torch
        return not torch.cuda.is_available()
    except ImportError:
        return True


def requires_model_file():
    """Skip test if YOLO model file is not available."""
    from pathlib import Path
    model_path = Path("data/models/yolov12x.pt")
    return not model_path.exists()


# ============================================================================
# Pytest Markers
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "selenium: mark test as requiring Selenium/WebDriver")
    config.addinivalue_line("markers", "database: mark test as requiring database")
    config.addinivalue_line("markers", "network: mark test as requiring network access")


# ============================================================================
# Custom Assertions
# ============================================================================

def assert_detection_valid(detection):
    """Assert that a detection object has valid properties."""
    assert hasattr(detection, 'bbox_x')
    assert hasattr(detection, 'bbox_y')
    assert hasattr(detection, 'bbox_width')
    assert hasattr(detection, 'bbox_height')
    assert hasattr(detection, 'confidence')
    assert hasattr(detection, 'object_type')
    
    # Validate ranges
    assert 0 <= detection.confidence <= 1
    assert detection.bbox_width > 0
    assert detection.bbox_height > 0


def assert_image_file_valid(image_path):
    """Assert that an image file exists and is valid."""
    path = Path(image_path)
    assert path.exists(), f"Image file does not exist: {image_path}"
    assert path.suffix.lower() in ['.jpg', '.jpeg', '.png'], f"Invalid image format: {image_path}"
    
    # Try to load the image
    img = cv2.imread(str(path))
    assert img is not None, f"Could not load image: {image_path}"
    assert len(img.shape) == 3, f"Image is not 3-channel: {image_path}"