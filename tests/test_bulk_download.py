"""
Tests for the bulk download module.

Tests cover:
- BulkImageDownloader initialization and configuration
- Date range downloading functionality
- Database integration for metadata storage
- File organization and management
- Download reporting and statistics
- Error handling and recovery
- Progress tracking
"""

import pytest
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from dataclasses import asdict

from utils.bulk_download import BulkImageDownloader, DownloadStats
from modules.database.models import Image


class TestDownloadStats:
    """Test class for DownloadStats dataclass."""
    
    def test_download_stats_init(self):
        """Test DownloadStats initialization with defaults."""
        stats = DownloadStats()
        
        assert stats.total_requested == 0
        assert stats.successful_downloads == 0
        assert stats.failed_downloads == 0
        assert stats.already_existed == 0
        assert stats.total_file_size == 0
        assert stats.start_time is None
        assert stats.end_time is None
    
    def test_download_stats_duration_calculation(self):
        """Test duration calculation property."""
        stats = DownloadStats()
        
        # No times set - should return None
        assert stats.duration is None
        
        # Set start and end times
        start = datetime(2025, 9, 1, 10, 0, 0)
        end = datetime(2025, 9, 1, 10, 5, 30)
        stats.start_time = start
        stats.end_time = end
        
        expected_duration = timedelta(minutes=5, seconds=30)
        assert stats.duration == expected_duration
    
    def test_download_stats_success_rate(self):
        """Test success rate calculation."""
        stats = DownloadStats()
        
        # No requests - should return 0
        assert stats.success_rate == 0.0
        
        # Some successes
        stats.total_requested = 100
        stats.successful_downloads = 85
        assert stats.success_rate == 85.0
        
        # Perfect success
        stats.successful_downloads = 100
        assert stats.success_rate == 100.0


class TestBulkImageDownloader:
    """Test class for BulkImageDownloader functionality."""
    
    def test_init_default_params(self, temp_dir):
        """Test BulkImageDownloader initialization with default parameters."""
        download_dir = str(temp_dir / "bulk_downloads")
        
        with patch('utils.bulk_download.DrayDogDownloader') as mock_downloader:
            downloader = BulkImageDownloader(download_dir=download_dir)
            
            assert downloader.download_dir == Path(download_dir)
            assert downloader.headless == True
            assert downloader.max_retries == 3
            assert downloader.retry_delay == 1.0
            assert downloader.timeout == 30
            assert downloader.use_direct_download == True
            assert isinstance(downloader.stats, DownloadStats)
            
            # Check that download directory was created
            assert downloader.download_dir.exists()
            
            # Verify DrayDogDownloader was initialized correctly
            mock_downloader.assert_called_once_with(
                download_dir=download_dir,
                headless=True,
                max_retries=3,
                retry_delay=1.0,
                timeout=30
            )
    
    def test_init_custom_params(self, temp_dir):
        """Test BulkImageDownloader initialization with custom parameters."""
        download_dir = str(temp_dir / "custom_bulk")
        
        with patch('utils.bulk_download.DrayDogDownloader') as mock_downloader:
            downloader = BulkImageDownloader(
                download_dir=download_dir,
                headless=False,
                max_retries=5,
                retry_delay=0.5,
                timeout=60,
                use_direct_download=False
            )
            
            assert downloader.download_dir == Path(download_dir)
            assert downloader.headless == False
            assert downloader.max_retries == 5
            assert downloader.retry_delay == 0.5
            assert downloader.timeout == 60
            assert downloader.use_direct_download == False
            
            # Verify custom parameters passed to DrayDogDownloader
            mock_downloader.assert_called_once_with(
                download_dir=download_dir,
                headless=False,
                max_retries=5,
                retry_delay=0.5,
                timeout=60
            )
    
    def test_download_date_range_invalid_dates(self, temp_dir):
        """Test download_date_range with invalid date formats."""
        download_dir = str(temp_dir / "downloads")
        
        with patch('utils.bulk_download.DrayDogDownloader'):
            downloader = BulkImageDownloader(download_dir=download_dir)
            
            # Invalid date format
            with pytest.raises(ValueError, match="Dates must be in YYYY-MM-DD format"):
                downloader.download_date_range("2025/09/01", "2025-09-07", ["in_gate"])
            
            # Start date after end date
            with pytest.raises(ValueError, match="Start date must be before or equal to end date"):
                downloader.download_date_range("2025-09-07", "2025-09-01", ["in_gate"])
    
    @patch('utils.bulk_download.datetime')
    def test_download_date_range_success(self, mock_datetime, temp_dir):
        """Test successful date range download."""
        download_dir = str(temp_dir / "downloads")
        
        # Mock current time for stats
        mock_now = datetime(2025, 9, 12, 10, 0, 0)
        mock_datetime.utcnow.return_value = mock_now
        mock_datetime.strptime.side_effect = datetime.strptime
        
        with patch('utils.bulk_download.DrayDogDownloader') as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader_class.return_value = mock_downloader
            
            # Mock successful downloads
            mock_files = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
            mock_downloader.download_images_direct.return_value = mock_files
            
            with patch('os.path.exists', return_value=True), \
                 patch('os.path.getsize', return_value=1024):
                
                downloader = BulkImageDownloader(
                    download_dir=download_dir, 
                    use_direct_download=True
                )
                
                results = downloader.download_date_range(
                    start_date="2025-09-01",
                    end_date="2025-09-02",
                    streams=["in_gate"]
                )
                
                # Verify results structure
                assert "2025-09-01" in results
                assert "2025-09-02" in results
                assert "in_gate" in results["2025-09-01"]
                assert "in_gate" in results["2025-09-02"]
                assert results["2025-09-01"]["in_gate"] == mock_files
                assert results["2025-09-02"]["in_gate"] == mock_files
                
                # Verify stats were updated
                assert downloader.stats.successful_downloads == 4  # 2 dates × 1 stream × 2 files
                assert downloader.stats.total_file_size == 4 * 1024  # 4 files × 1024 bytes
                assert downloader.stats.start_time == mock_now
                assert downloader.stats.end_time == mock_now
    
    @patch('utils.bulk_download.datetime')
    def test_download_date_range_with_failures(self, mock_datetime, temp_dir):
        """Test date range download with some failures."""
        download_dir = str(temp_dir / "downloads")
        
        mock_now = datetime(2025, 9, 12, 10, 0, 0)
        mock_datetime.utcnow.return_value = mock_now
        mock_datetime.strptime.side_effect = datetime.strptime
        
        with patch('utils.bulk_download.DrayDogDownloader') as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader_class.return_value = mock_downloader
            
            # First call succeeds, second fails
            mock_downloader.download_images_direct.side_effect = [
                ["/path/to/image1.jpg"],
                Exception("Network error")
            ]
            
            with patch('os.path.exists', return_value=True), \
                 patch('os.path.getsize', return_value=1024):
                
                downloader = BulkImageDownloader(download_dir=download_dir)
                
                results = downloader.download_date_range(
                    start_date="2025-09-01",
                    end_date="2025-09-01",
                    streams=["in_gate", "out_gate"]
                )
                
                # Verify results - one success, one failure
                assert results["2025-09-01"]["in_gate"] == ["/path/to/image1.jpg"]
                assert results["2025-09-01"]["out_gate"] == []
                
                # Verify stats
                assert downloader.stats.successful_downloads == 1
                assert downloader.stats.failed_downloads == 1
    
    def test_save_to_database(self, temp_dir):
        """Test saving image metadata to database."""
        download_dir = str(temp_dir / "downloads")
        
        with patch('utils.bulk_download.DrayDogDownloader'), \
             patch('utils.bulk_download.queries') as mock_queries, \
             patch('utils.bulk_download.session_scope') as mock_session_scope:
            
            # Mock successful database inserts
            mock_queries.insert_image.side_effect = [1, 2, 3]
            mock_session_scope.return_value.__enter__.return_value = Mock()
            
            downloader = BulkImageDownloader(download_dir=download_dir)
            
            metadata_list = [
                {
                    'filepath': '/path/to/image1.jpg',
                    'camera_id': 'in_gate',
                    'timestamp': datetime(2025, 9, 1, 10, 0, 0),
                    'file_size': 1024
                },
                {
                    'filepath': '/path/to/image2.jpg',
                    'camera_id': 'out_gate',
                    'timestamp': datetime(2025, 9, 1, 10, 10, 0),
                    'file_size': 2048
                },
                {
                    'filepath': '/path/to/image3.jpg',
                    'camera_id': 'in_gate',
                    'file_size': 512  # Missing timestamp - should use default
                }
            ]
            
            saved_count = downloader.save_to_database(metadata_list)
            
            assert saved_count == 3
            assert mock_queries.insert_image.call_count == 3
    
    def test_save_to_database_with_failures(self, temp_dir):
        """Test database save with some failures."""
        download_dir = str(temp_dir / "downloads")
        
        with patch('utils.bulk_download.DrayDogDownloader'), \
             patch('utils.bulk_download.queries') as mock_queries, \
             patch('utils.bulk_download.session_scope') as mock_session_scope:
            
            # First insert succeeds, second fails, third succeeds
            mock_queries.insert_image.side_effect = [1, Exception("DB error"), 3]
            mock_session_scope.return_value.__enter__.return_value = Mock()
            
            downloader = BulkImageDownloader(download_dir=download_dir)
            
            metadata_list = [
                {'filepath': '/path/1.jpg', 'camera_id': 'in_gate', 'file_size': 1024},
                {'filepath': '/path/2.jpg', 'camera_id': 'out_gate', 'file_size': 2048},
                {'filepath': '/path/3.jpg', 'camera_id': 'in_gate', 'file_size': 512}
            ]
            
            saved_count = downloader.save_to_database(metadata_list)
            
            assert saved_count == 2  # Only 2 successful saves
    
    def test_organize_files_success(self, temp_dir):
        """Test successful file organization."""
        download_dir = str(temp_dir / "downloads")
        source_dir = temp_dir / "source"
        target_dir = temp_dir / "target"
        
        # Create test file structure
        source_dir.mkdir()
        test_file = source_dir / "20250901123000_in_gate.jpg"
        test_file.write_text("fake image")
        
        with patch('utils.bulk_download.DrayDogDownloader'):
            downloader = BulkImageDownloader(download_dir=download_dir)
            
            result = downloader.organize_files(str(source_dir), str(target_dir))
            
            assert result == True
            
            # Check that file was moved to correct structure
            expected_path = target_dir / "2025-09-01" / "in_gate" / "20250901123000_in_gate.jpg"
            assert expected_path.exists()
            assert not test_file.exists()  # Original should be moved
    
    def test_organize_files_source_not_exists(self, temp_dir):
        """Test file organization when source doesn't exist."""
        download_dir = str(temp_dir / "downloads")
        
        with patch('utils.bulk_download.DrayDogDownloader'):
            downloader = BulkImageDownloader(download_dir=download_dir)
            
            result = downloader.organize_files("nonexistent", str(temp_dir / "target"))
            
            assert result == False
    
    def test_generate_download_report(self, temp_dir):
        """Test download report generation."""
        download_dir = str(temp_dir / "downloads")
        
        with patch('utils.bulk_download.DrayDogDownloader'):
            downloader = BulkImageDownloader(
                download_dir=download_dir,
                headless=False,
                max_retries=5
            )
            
            # Set up some stats
            start_time = datetime(2025, 9, 12, 10, 0, 0)
            end_time = datetime(2025, 9, 12, 10, 5, 0)
            downloader.stats.start_time = start_time
            downloader.stats.end_time = end_time
            downloader.stats.successful_downloads = 85
            downloader.stats.failed_downloads = 15
            downloader.stats.total_file_size = 1024 * 1024 * 10  # 10 MB
            
            report = downloader.generate_download_report()
            
            # Verify report structure
            assert "summary" in report
            assert "timestamps" in report
            assert "configuration" in report
            
            # Verify summary data
            summary = report["summary"]
            assert summary["successful_downloads"] == 85
            assert summary["failed_downloads"] == 15
            assert summary["success_rate_percent"] == 0.0  # No total_requested set
            assert summary["total_file_size_mb"] == 10.0
            assert summary["duration_seconds"] == 300.0  # 5 minutes
            
            # Verify configuration
            config = report["configuration"]
            assert config["download_dir"] == download_dir
            assert config["headless"] == False
            assert config["max_retries"] == 5
    
    def test_save_report(self, temp_dir):
        """Test saving download report to file."""
        download_dir = temp_dir / "downloads"
        download_dir.mkdir()
        
        with patch('utils.bulk_download.DrayDogDownloader'):
            downloader = BulkImageDownloader(download_dir=str(download_dir))
            
            test_report = {
                "summary": {"successful_downloads": 100},
                "timestamps": {"start_time": "2025-09-12T10:00:00"},
                "configuration": {"download_dir": str(download_dir)}
            }
            
            # Test with default path
            with patch('utils.bulk_download.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime(2025, 9, 12, 15, 30, 45)
                report_path = downloader.save_report(test_report)
            
            expected_path = download_dir / "bulk_download_report_20250912_153045.json"
            assert report_path == str(expected_path)
            assert expected_path.exists()
            
            # Verify file contents
            with open(expected_path) as f:
                saved_report = json.load(f)
            assert saved_report["summary"]["successful_downloads"] == 100
    
    def test_save_report_custom_path(self, temp_dir):
        """Test saving report with custom path."""
        download_dir = str(temp_dir / "downloads")
        custom_path = temp_dir / "custom_report.json"
        
        with patch('utils.bulk_download.DrayDogDownloader'):
            downloader = BulkImageDownloader(download_dir=download_dir)
            
            test_report = {"summary": {"test": "data"}}
            
            report_path = downloader.save_report(test_report, str(custom_path))
            
            assert report_path == str(custom_path)
            assert custom_path.exists()
    
    def test_extract_file_info(self, temp_dir):
        """Test extracting date and stream info from file paths."""
        download_dir = str(temp_dir / "downloads")
        
        with patch('utils.bulk_download.DrayDogDownloader'):
            downloader = BulkImageDownloader(download_dir=download_dir)
            
            # Test filename with date and stream
            test_path = Path("20250901120000_in_gate.jpg")
            date_str, stream_name = downloader._extract_file_info(test_path)
            assert date_str == "2025-09-01"
            assert stream_name == "in_gate"
            
            # Test with out_gate
            test_path = Path("20250902150000_out_gate.jpg")
            date_str, stream_name = downloader._extract_file_info(test_path)
            assert date_str == "2025-09-02"
            assert stream_name == "out_gate"
            
            # Test path with directory structure
            test_path = Path("data/images/2025-09-03/in_gate/image.jpg")
            date_str, stream_name = downloader._extract_file_info(test_path)
            assert date_str == "2025-09-03"
            assert stream_name == "in_gate"
            
            # Test unrecognizable pattern
            test_path = Path("unknown_file.jpg")
            date_str, stream_name = downloader._extract_file_info(test_path)
            assert date_str is None
            assert stream_name is None
    
    def test_context_manager(self, temp_dir):
        """Test BulkImageDownloader as context manager."""
        download_dir = str(temp_dir / "downloads")
        
        with patch('utils.bulk_download.DrayDogDownloader') as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader_class.return_value = mock_downloader
            
            # Test context manager enter/exit
            with BulkImageDownloader(download_dir=download_dir) as downloader:
                assert downloader is not None
                assert isinstance(downloader, BulkImageDownloader)
            
            # Verify cleanup was called
            mock_downloader.cleanup.assert_called_once()
    
    def test_cleanup(self, temp_dir):
        """Test cleanup functionality."""
        download_dir = str(temp_dir / "downloads")
        
        with patch('utils.bulk_download.DrayDogDownloader') as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader_class.return_value = mock_downloader
            
            downloader = BulkImageDownloader(download_dir=download_dir)
            downloader.cleanup()
            
            mock_downloader.cleanup.assert_called_once()


@patch('utils.bulk_download.logger')
def test_main_function(mock_logger, temp_dir):
    """Test the main function."""
    with patch('utils.bulk_download.BulkImageDownloader') as mock_downloader_class:
        mock_downloader = Mock()
        mock_downloader_class.return_value.__enter__.return_value = mock_downloader
        
        # Mock download results
        mock_results = {
            "2025-09-01": {"in_gate": ["/file1.jpg"], "out_gate": ["/file2.jpg"]},
            "2025-09-02": {"in_gate": ["/file3.jpg"], "out_gate": ["/file4.jpg"]}
        }
        mock_downloader.download_date_range.return_value = mock_results
        
        # Mock report
        mock_report = {
            "summary": {
                "success_rate_percent": 95.5,
                "successful_downloads": 4,
                "total_file_size_mb": 12.5,
                "duration_seconds": 120.0
            }
        }
        mock_downloader.generate_download_report.return_value = mock_report
        mock_downloader.save_report.return_value = "/path/to/report.json"
        
        # Import and run main
        from utils.bulk_download import main
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            main()
        
        # Verify downloader was configured correctly
        mock_downloader_class.assert_called_once_with(
            download_dir="data/images",
            headless=True,
            use_direct_download=True
        )
        
        # Verify download was called with correct parameters
        mock_downloader.download_date_range.assert_called_once_with(
            start_date="2025-09-01",
            end_date="2025-09-07",
            streams=["in_gate", "out_gate"]
        )
        
        # Verify report generation
        mock_downloader.generate_download_report.assert_called_once()
        mock_downloader.save_report.assert_called_once()
        
        # Verify some print statements were made
        assert mock_print.call_count >= 1


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def sample_image_metadata():
    """Sample image metadata for testing."""
    return [
        {
            'filepath': '/data/images/2025-09-01/image1.jpg',
            'camera_id': 'in_gate',
            'timestamp': datetime(2025, 9, 1, 10, 0, 0),
            'file_size': 1024
        },
        {
            'filepath': '/data/images/2025-09-01/image2.jpg',
            'camera_id': 'out_gate',
            'timestamp': datetime(2025, 9, 1, 10, 10, 0),
            'file_size': 2048
        }
    ]


@pytest.fixture
def mock_download_results():
    """Mock download results for testing."""
    return {
        "2025-09-01": {
            "in_gate": ["/data/images/2025-09-01/in_gate/img1.jpg"],
            "out_gate": ["/data/images/2025-09-01/out_gate/img2.jpg"]
        },
        "2025-09-02": {
            "in_gate": ["/data/images/2025-09-02/in_gate/img3.jpg"],
            "out_gate": []  # Simulated failure
        }
    }