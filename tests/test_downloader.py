"""
Tests for the downloader module.

Tests cover:
- DrayDogDownloader initialization
- Direct navigation to camera pages (no authentication required)
- Image downloading logic
- Scheduler functionality
- Error handling and retry logic
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

from modules.downloader.selenium_client import DrayDogDownloader


class TestDrayDogDownloader:
    """Test class for DrayDogDownloader functionality."""

    def test_init_default_params(self, temp_dir):
        """Test DrayDogDownloader initialization with default parameters."""
        download_dir = str(temp_dir / "downloads")

        downloader = DrayDogDownloader(download_dir=download_dir)

        assert downloader.download_dir == download_dir
        assert downloader.headless == True  # Default
        assert downloader.max_retries == 3
        assert downloader.retry_delay == 1.0
        assert downloader.timeout == 30
        assert downloader.driver is None

        # Check that download directory was created
        assert Path(download_dir).exists()

    def test_init_custom_params(self, temp_dir):
        """Test DrayDogDownloader initialization with custom parameters."""
        download_dir = str(temp_dir / "custom_downloads")

        downloader = DrayDogDownloader(
            download_dir=download_dir,
            headless=False,
            max_retries=5,
            retry_delay=0.5,
            timeout=60,
        )

        assert downloader.download_dir == download_dir
        assert downloader.headless == False
        assert downloader.max_retries == 5
        assert downloader.retry_delay == 0.5
        assert downloader.timeout == 60

    @patch("modules.downloader.selenium_client.webdriver.Chrome")
    def test_setup_driver_success(self, mock_chrome_class, temp_dir):
        """Test successful Chrome WebDriver setup."""
        mock_driver = Mock()
        mock_chrome_class.return_value = mock_driver

        downloader = DrayDogDownloader(download_dir=str(temp_dir))
        driver = downloader._setup_driver()

        assert driver == mock_driver
        mock_driver.execute_script.assert_called_once()
        mock_chrome_class.assert_called_once()

    @patch("modules.downloader.selenium_client.webdriver.Chrome")
    def test_setup_driver_failure(self, mock_chrome_class, temp_dir):
        """Test Chrome WebDriver setup failure."""
        mock_chrome_class.side_effect = Exception("WebDriver failed to start")

        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        with pytest.raises(Exception, match="WebDriver failed to start"):
            downloader._setup_driver()

    def test_retry_operation_success(self, temp_dir):
        """Test retry operation with successful execution."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        def successful_operation(x, y):
            return x + y

        result = downloader._retry_operation(successful_operation, 2, 3)
        assert result == 5

    def test_retry_operation_success_after_retries(self, temp_dir):
        """Test retry operation that succeeds after initial failures."""
        downloader = DrayDogDownloader(
            download_dir=str(temp_dir),
            max_retries=3,
            retry_delay=0.01,  # Fast retry for testing
        )

        call_count = 0

        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Failure {call_count}")
            return "success"

        result = downloader._retry_operation(flaky_operation)
        assert result == "success"
        assert call_count == 3

    def test_retry_operation_max_retries_exceeded(self, temp_dir):
        """Test retry operation that fails after max retries."""
        downloader = DrayDogDownloader(
            download_dir=str(temp_dir), max_retries=2, retry_delay=0.01
        )

        def failing_operation():
            raise Exception("Always fails")

        with pytest.raises(Exception, match="Always fails"):
            downloader._retry_operation(failing_operation)

    @patch("modules.downloader.selenium_client.WebDriverWait")
    def test_no_authentication_required(
        self, mock_wait_class, mock_selenium_driver, temp_dir
    ):
        """Test that authentication is not required for accessing images."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        # Verify download_images_for_date doesn't require username/password
        import inspect

        sig = inspect.signature(downloader.download_images_for_date)
        params = list(sig.parameters.keys())

        assert "username" not in params, "Should not have username parameter"
        assert "password" not in params, "Should not have password parameter"
        assert "date_str" in params, "Should have date_str parameter"
        assert "stream_name" in params, "Should have stream_name parameter"

    @patch("modules.downloader.selenium_client.webdriver.Chrome")
    @patch("modules.downloader.selenium_client.WebDriverWait")
    def test_direct_navigation_no_login(
        self, mock_wait_class, mock_chrome_class, temp_dir
    ):
        """Test that navigation happens directly without login."""
        mock_driver = Mock()
        mock_chrome_class.return_value = mock_driver
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_wait.until.return_value = Mock()

        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        # Navigate to camera history - should work without authentication
        result = downloader.navigate_to_camera_history("test_stream")

        # Verify URL contains direct access path, not login
        called_url = mock_driver.get.call_args[0][0]
        assert "terminal_cameras/apm" in called_url
        assert "login" not in called_url
        assert result == True

    @patch("modules.downloader.selenium_client.WebDriverWait")
    def test_navigate_to_camera_history_success(
        self, mock_wait_class, mock_selenium_driver, temp_dir
    ):
        """Test successful navigation to camera history."""
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_element = Mock()
        mock_wait.until.return_value = mock_element

        downloader = DrayDogDownloader(download_dir=str(temp_dir))
        downloader.driver = mock_selenium_driver

        result = downloader.navigate_to_camera_history("in_gate")

        assert result == True
        expected_url = "https://app.draydog.com/terminal_cameras/apm?streamName=in_gate#camera-history"
        mock_selenium_driver.get.assert_called_with(expected_url)

    @patch("modules.downloader.selenium_client.webdriver.Chrome")
    @patch("modules.downloader.selenium_client.WebDriverWait")
    def test_navigate_to_camera_history_initializes_driver(
        self, mock_wait_class, mock_chrome_class, temp_dir
    ):
        """Test navigation initializes driver if not already initialized."""
        mock_driver = Mock()
        mock_chrome_class.return_value = mock_driver
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_wait.until.return_value = Mock()

        downloader = DrayDogDownloader(download_dir=str(temp_dir))
        assert downloader.driver is None  # Initially no driver

        result = downloader.navigate_to_camera_history("in_gate")

        assert result == True
        assert mock_chrome_class.called
        assert downloader.driver is not None

    def test_extract_image_urls(
        self, mock_selenium_driver, sample_image_metadata, temp_dir
    ):
        """Test extraction of image URLs from page."""
        mock_selenium_driver.execute_script.return_value = sample_image_metadata

        downloader = DrayDogDownloader(download_dir=str(temp_dir))
        downloader.driver = mock_selenium_driver

        with patch("modules.downloader.selenium_client.WebDriverWait"):
            image_data = downloader.extract_image_urls()

        assert len(image_data) == 2
        assert image_data[0]["url"] == sample_image_metadata[0]["url"]
        assert image_data[0]["filename"] == sample_image_metadata[0]["filename"]

    def test_extract_image_urls_no_driver(self, temp_dir):
        """Test image URL extraction without driver."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        # Should return empty list when driver is not initialized
        result = downloader.extract_image_urls()
        assert result == []

    @patch("modules.downloader.selenium_client.requests.get")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_image_success(
        self, mock_file, mock_requests, mock_requests_get, temp_dir
    ):
        """Test successful image download."""
        mock_requests.return_value = mock_requests_get

        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        image_info = {
            "url": "https://cdn.draydog.com/test/2025-01-15T10-30-00-in_gate.jpg",
            "filename": "2025-01-15T10-30-00-in_gate.jpg",
            "timestamp": "2025-01-15T10:30:00",
            "streamName": "in_gate",
        }

        with patch("os.path.exists", return_value=False), patch("os.makedirs"), patch(
            "os.path.getsize", return_value=1024
        ), patch.object(downloader, "_calculate_file_hash", return_value="fake_hash"):

            result = downloader.download_image(image_info)

        assert result is not None
        assert "2025-01-15T10-30-00-in_gate.jpg" in result
        mock_requests.assert_called_once()
        mock_file.assert_called()

    @patch("modules.downloader.selenium_client.requests.get")
    def test_download_image_already_exists(self, mock_requests, temp_dir):
        """Test download when image already exists."""
        from datetime import datetime
        
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        # Use current date for the test to match the logic in download_image
        today = datetime.now().strftime("%Y-%m-%d")
        date_dir = temp_dir / today
        date_dir.mkdir()
        image_path = date_dir / "existing_image.jpg"
        image_path.write_text("fake image data")

        image_info = {
            "url": "https://cdn.draydog.com/test/existing_image.jpg",
            "filename": "existing_image.jpg",
            "timestamp": "2025-01-15T10:30:00",
            "streamName": "in_gate",
        }

        result = downloader.download_image(image_info)

        assert result == str(image_path)
        mock_requests.assert_not_called()  # Should not download if file exists

    @patch("modules.downloader.selenium_client.requests.get")
    def test_download_image_network_error(self, mock_requests, temp_dir):
        """Test image download with network error."""
        import requests

        mock_requests.side_effect = requests.RequestException("Network error")

        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        image_info = {
            "url": "https://cdn.draydog.com/test/image.jpg",
            "filename": "image.jpg",
            "timestamp": "2025-01-15T10:30:00",
            "streamName": "in_gate",
        }

        result = downloader.download_image(image_info)

        assert result is None

    def test_calculate_file_hash(self, temp_dir):
        """Test file hash calculation."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        # Create a test file
        test_file = temp_dir / "test_file.txt"
        test_file.write_text("test content")

        file_hash = downloader._calculate_file_hash(str(test_file))

        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA-256 hash length

        # Test with same content should produce same hash
        hash2 = downloader._calculate_file_hash(str(test_file))
        assert file_hash == hash2

    def test_calculate_file_hash_nonexistent_file(self, temp_dir):
        """Test file hash calculation for non-existent file."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        file_hash = downloader._calculate_file_hash("nonexistent_file.txt")

        assert file_hash == ""

    def test_download_images_for_date_complete_workflow(self, temp_dir):
        """Test complete workflow for downloading images for a date."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        # Mock all the methods (no authentication needed)
        with patch.object(
            downloader, "navigate_to_camera_history", return_value=True
        ) as mock_navigate, patch.object(
            downloader,
            "extract_image_urls",
            return_value=[
                {"url": "https://test.com/img1.jpg", "filename": "img1.jpg"},
                {"url": "https://test.com/img2.jpg", "filename": "img2.jpg"},
            ],
        ), patch.object(
            downloader,
            "_retry_operation",
            side_effect=["/path/img1.jpg", "/path/img2.jpg"],
        ), patch.object(
            downloader, "cleanup"
        ):

            # No username/password parameters
            result = downloader.download_images_for_date("2025-01-15", "in_gate")
            
            # Assert within the context manager
            assert len(result) == 2
            assert "/path/img1.jpg" in result
            assert "/path/img2.jpg" in result
            mock_navigate.assert_called_once_with(
                "in_gate", "2025-01-15"
            )

    def test_download_images_for_date_navigation_failure(self, temp_dir):
        """Test workflow with navigation failure."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        with patch.object(
            downloader, "navigate_to_camera_history", return_value=False
        ), patch.object(downloader, "cleanup"):

            result = downloader.download_images_for_date("2025-01-15", "in_gate")

        assert result == []

    def test_download_images_for_date_range(self, temp_dir):
        """Test downloading images for a date range."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        with patch.object(
            downloader, "download_images_for_date", return_value=["/path/img.jpg"]
        ):
            result = downloader.download_images_for_date_range(
                "2025-01-15", "2025-01-17", "in_gate"
            )

        assert len(result) == 3  # 3 days
        assert "2025-01-15" in result
        assert "2025-01-16" in result
        assert "2025-01-17" in result
        for date_images in result.values():
            assert date_images == ["/path/img.jpg"]

    def test_cleanup_with_driver(self, mock_selenium_driver, temp_dir):
        """Test cleanup when driver exists."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))
        downloader.driver = mock_selenium_driver

        downloader.cleanup()

        mock_selenium_driver.quit.assert_called_once()
        assert downloader.driver is None

    def test_cleanup_without_driver(self, temp_dir):
        """Test cleanup when no driver exists."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        # Should not raise an exception
        downloader.cleanup()

    def test_cleanup_driver_error(self, temp_dir):
        """Test cleanup when driver quit raises exception."""
        mock_driver = Mock()
        mock_driver.quit.side_effect = Exception("Driver error")

        downloader = DrayDogDownloader(download_dir=str(temp_dir))
        downloader.driver = mock_driver

        # Should not raise an exception
        downloader.cleanup()
        assert downloader.driver is None

    def test_context_manager(self, temp_dir):
        """Test using DrayDogDownloader as context manager."""
        with patch.object(DrayDogDownloader, "cleanup") as mock_cleanup:
            with DrayDogDownloader(download_dir=str(temp_dir)) as downloader:
                assert isinstance(downloader, DrayDogDownloader)

            mock_cleanup.assert_called_once()


@pytest.mark.selenium
class TestDrayDogDownloaderIntegration:
    """Integration tests for DrayDogDownloader (require actual Selenium setup)."""

    def test_setup_driver_real(self, temp_dir):
        """Test setting up real Chrome WebDriver (if available)."""
        pytest.importorskip("selenium")

        try:
            downloader = DrayDogDownloader(download_dir=str(temp_dir), headless=True)
            driver = downloader._setup_driver()
            assert driver is not None
            driver.quit()
        except Exception as e:
            pytest.skip(f"Chrome WebDriver not available: {e}")

    def test_download_real_images_one_day(self):
        """Test downloading actual images from Dray Dog for one day."""
        pytest.importorskip("selenium")
        from datetime import datetime, timedelta
        
        # Use yesterday's date to ensure images are available
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Set up the download directory
        download_dir = "/Users/rayli/Documents/container-analytics/data/images"
        
        try:
            downloader = DrayDogDownloader(
                download_dir=download_dir,
                headless=True,  # Run in headless mode
                max_retries=3,
                retry_delay=2.0,
                timeout=60
            )
            
            print(f"\n=== Starting download for {yesterday} ===")
            
            # Download images for the in_gate camera
            result = downloader.download_images_for_date(
                date_str=yesterday,
                stream_name="in_gate"
            )
            
            print(f"Downloaded {len(result)} images")
            
            # Print downloaded files
            if result:
                print("\nDownloaded files:")
                for file_path in result[:5]:  # Show first 5 files
                    print(f"  - {file_path}")
                if len(result) > 5:
                    print(f"  ... and {len(result) - 5} more files")
            
            # Verify files exist
            assert len(result) > 0, "No images were downloaded"
            
            for file_path in result:
                assert Path(file_path).exists(), f"File does not exist: {file_path}"
                assert Path(file_path).stat().st_size > 0, f"File is empty: {file_path}"
            
            print(f"\n✅ Successfully downloaded {len(result)} images to {download_dir}")
            
            # Cleanup
            downloader.cleanup()
            
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            if 'downloader' in locals():
                downloader.cleanup()
            # If Chrome WebDriver is not available, skip the test
            if "Chrome" in str(e) or "chromedriver" in str(e):
                pytest.skip(f"Chrome WebDriver not available: {e}")
            else:
                raise


class TestDrayDogDownloaderScheduler:
    """Tests for scheduler functionality."""

    @patch("modules.downloader.scheduler.APScheduler")
    def test_scheduler_initialization(self, mock_scheduler):
        """Test scheduler initialization."""
        # This would test the scheduler.py module
        # For now, this is a placeholder since scheduler.py wasn't shown
        pass

    @patch("modules.downloader.scheduler.DrayDogDownloader")
    def test_scheduled_download_job(self, mock_downloader_class):
        """Test scheduled download job execution."""
        # This would test periodic download scheduling
        # Placeholder for scheduler tests
        pass


class TestDrayDogDownloaderErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_date_format(self, temp_dir):
        """Test handling of invalid date format."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        with patch.object(
            downloader, "navigate_to_camera_history", return_value=True
        ), patch.object(
            downloader, "extract_image_urls", return_value=[]
        ), patch.object(
            downloader, "cleanup"
        ):

            # Should handle invalid date gracefully
            result = downloader.download_images_for_date("invalid-date", "in_gate")

        assert result == []

    def test_empty_image_list(self, temp_dir):
        """Test handling when no images are found."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        with patch.object(
            downloader, "navigate_to_camera_history", return_value=True
        ), patch.object(
            downloader, "extract_image_urls", return_value=[]
        ), patch.object(
            downloader, "cleanup"
        ):

            result = downloader.download_images_for_date("2025-01-15", "in_gate")

        assert result == []

    def test_partial_download_failure(self, temp_dir):
        """Test handling when some downloads fail."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        def mock_retry_side_effect(operation, image_info):
            if "fail" in image_info["filename"]:
                raise Exception("Download failed")
            return f"/path/{image_info['filename']}"

        with patch.object(
            downloader, "navigate_to_camera_history", return_value=True
        ), patch.object(
            downloader,
            "extract_image_urls",
            return_value=[
                {"filename": "success.jpg", "url": "http://test.com/success.jpg"},
                {"filename": "fail.jpg", "url": "http://test.com/fail.jpg"},
                {"filename": "success2.jpg", "url": "http://test.com/success2.jpg"},
            ],
        ), patch.object(
            downloader, "_retry_operation", side_effect=mock_retry_side_effect
        ), patch.object(
            downloader, "cleanup"
        ):

            result = downloader.download_images_for_date("2025-01-15", "in_gate")

        # Should return only successful downloads
        assert len(result) == 2
        assert "/path/success.jpg" in result
        assert "/path/success2.jpg" in result
        assert "/path/fail.jpg" not in result


@pytest.mark.slow
class TestDrayDogDownloaderPerformance:
    """Performance tests for DrayDogDownloader."""

    def test_large_image_list_processing(self, temp_dir):
        """Test processing large number of images."""
        downloader = DrayDogDownloader(download_dir=str(temp_dir))

        # Create a large list of mock image metadata
        large_image_list = []
        for i in range(100):
            large_image_list.append(
                {
                    "url": f"https://test.com/image_{i}.jpg",
                    "filename": f"image_{i}.jpg",
                    "timestamp": f"2025-01-15T{i%24:02d}:30:00",
                    "streamName": "in_gate",
                }
            )

        with patch.object(
            downloader, "navigate_to_camera_history", return_value=True
        ), patch.object(
            downloader, "extract_image_urls", return_value=large_image_list
        ), patch.object(
            downloader, "_retry_operation", return_value="/path/img.jpg"
        ), patch.object(
            downloader, "cleanup"
        ):

            import time

            start_time = time.time()
            result = downloader.download_images_for_date("2025-01-15", "in_gate")
            end_time = time.time()

        assert len(result) == 100
        # Should complete in reasonable time (adjust threshold as needed)
        assert end_time - start_time < 10.0
