"""
Container Analytics Test Suite

This package contains comprehensive tests for the Container Analytics project.

Test modules:
- test_downloader.py: Tests for image downloading from Dray Dog
- test_detection.py: Tests for YOLOv12 object detection
- test_database.py: Tests for database models and operations
- test_analytics.py: Tests for metrics and analytics calculations

Usage:
    # Run all tests
    pytest tests/
    
    # Run specific test module
    pytest tests/test_detection.py
    
    # Run with coverage
    pytest --cov=modules tests/
    
    # Run in verbose mode
    pytest -v tests/
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "data"
TEST_IMAGES_DIR = TEST_DATA_DIR / "images"
TEST_DATABASE_URL = "sqlite:///:memory:"  # In-memory database for tests

# Ensure test data directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_IMAGES_DIR.mkdir(exist_ok=True)