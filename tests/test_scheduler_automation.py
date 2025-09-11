"""
Tests for scheduler automation functionality including retry logic, health checks, and job configuration.
"""

import pytest
import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from modules.downloader.scheduler import (
    ImageDownloadScheduler,
    DownloadConfig,
    DownloadStats,
)


@pytest.fixture
def test_config():
    """Create a test configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = DownloadConfig(
            stream_names=["test_stream"],
            download_dir=temp_dir,
            download_interval_minutes=10,
            max_retries=3,
            retry_delay_seconds=1,
            exponential_backoff=True,
            backoff_multiplier=2.0,
            max_retry_delay_seconds=600,
            enable_health_check=True,
            headless=True,
        )
        yield config


@pytest.fixture
def mock_scheduler(test_config):
    """Create a mock scheduler instance."""
    with patch("modules.downloader.scheduler.session_scope"), patch(
        "modules.downloader.scheduler.get_image_stats"
    ), patch("modules.downloader.scheduler.logger"):

        scheduler = ImageDownloadScheduler(test_config)
        scheduler._shutdown_event = Mock()
        scheduler._shutdown_event.wait = Mock(return_value=False)
        yield scheduler


class TestDownloadConfig:
    """Test the enhanced DownloadConfig class."""

    def test_config_validation_valid(self, test_config):
        """Test that valid configuration passes validation."""
        test_config.validate()  # Should not raise

    def test_config_validation_invalid_interval(self, test_config):
        """Test that invalid download interval raises error."""
        test_config.download_interval_minutes = 3
        with pytest.raises(
            ValueError, match="Download interval must be at least 5 minutes"
        ):
            test_config.validate()

    def test_config_validation_invalid_retention(self, test_config):
        """Test that invalid retention days raises error."""
        test_config.retention_days = 0
        with pytest.raises(ValueError, match="Retention days must be at least 1"):
            test_config.validate()

    def test_config_validation_invalid_batch_size(self, test_config):
        """Test that invalid batch size raises error."""
        test_config.batch_size = 0
        with pytest.raises(ValueError, match="Batch size must be between 1 and 1000"):
            test_config.validate()

        test_config.batch_size = 1001
        with pytest.raises(ValueError, match="Batch size must be between 1 and 1000"):
            test_config.validate()

    def test_config_validation_invalid_retries(self, test_config):
        """Test that invalid max retries raises error."""
        test_config.max_retries = 0
        with pytest.raises(ValueError, match="Max retries must be at least 1"):
            test_config.validate()

    def test_config_validation_invalid_backoff(self, test_config):
        """Test that invalid backoff multiplier raises error."""
        test_config.backoff_multiplier = 1.0
        with pytest.raises(
            ValueError, match="Backoff multiplier must be greater than 1.0"
        ):
            test_config.validate()

    def test_legacy_compatibility(self):
        """Test backward compatibility with legacy field names."""
        config = DownloadConfig(interval_minutes=15, cleanup_days=14)

        # Should sync legacy fields to new fields
        assert config.download_interval_minutes == 15
        assert config.retention_days == 14


class TestRetryLogic:
    """Test retry logic with exponential backoff."""

    def test_retry_success_first_attempt(self, mock_scheduler):
        """Test that successful function executes without retry."""
        mock_func = Mock(return_value="success")

        result = mock_scheduler._download_with_retry(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_with("arg1", kwarg1="value1")

    def test_retry_success_after_failures(self, mock_scheduler):
        """Test retry with exponential backoff until success."""
        mock_func = Mock(
            side_effect=[Exception("Error 1"), Exception("Error 2"), "success"]
        )

        # Mock the wait method to avoid actual delays
        mock_scheduler._shutdown_event.wait = Mock(return_value=False)

        result = mock_scheduler._download_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3
        assert (
            mock_scheduler.stats.failed_downloads == 0
        )  # Should be 0 since it eventually succeeded

    def test_retry_all_attempts_fail(self, mock_scheduler):
        """Test that all retry attempts fail and exception is raised."""
        mock_func = Mock(side_effect=Exception("Persistent error"))

        # Mock the wait method to avoid actual delays
        mock_scheduler._shutdown_event.wait = Mock(return_value=False)

        with pytest.raises(Exception, match="Persistent error"):
            mock_scheduler._download_with_retry(mock_func)

        assert mock_func.call_count == 3  # max_retries
        assert mock_scheduler.stats.failed_downloads == 1

    def test_retry_exponential_backoff_calculation(self, mock_scheduler):
        """Test that exponential backoff calculates delays correctly."""
        mock_func = Mock(
            side_effect=[Exception("Error 1"), Exception("Error 2"), "success"]
        )

        # Track calls to wait method to verify backoff delays
        wait_calls = []

        def mock_wait(delay):
            wait_calls.append(delay)
            return False

        # Mock the shutdown event's wait method
        mock_scheduler._shutdown_event = Mock()
        mock_scheduler._shutdown_event.wait = Mock(side_effect=mock_wait)

        result = mock_scheduler._download_with_retry(mock_func)

        assert result == "success"
        assert len(wait_calls) == 2  # Two retries needed
        assert wait_calls[0] == 1  # Initial delay
        assert wait_calls[1] == 2  # Doubled delay (1 * 2.0)

    def test_retry_max_delay_cap(self, mock_scheduler):
        """Test that retry delay respects maximum cap."""
        mock_scheduler.config.max_retry_delay_seconds = 3
        mock_func = Mock(
            side_effect=[Exception("Error 1"), Exception("Error 2"), "success"]
        )

        wait_calls = []

        def mock_wait(delay):
            wait_calls.append(delay)
            return False

        mock_scheduler._shutdown_event.wait = Mock(side_effect=mock_wait)

        result = mock_scheduler._download_with_retry(mock_func)

        assert result == "success"
        assert all(delay <= 3 for delay in wait_calls)  # Should not exceed max

    def test_retry_shutdown_during_wait(self, mock_scheduler):
        """Test that retry stops when shutdown is requested."""
        mock_func = Mock(side_effect=Exception("Error"))

        # Simulate shutdown signal on first wait
        mock_scheduler._shutdown_event.wait = Mock(return_value=True)

        with pytest.raises(Exception, match="Error"):
            mock_scheduler._download_with_retry(mock_func)

        assert mock_func.call_count == 1  # Should stop after first attempt
        assert mock_scheduler.stats.failed_downloads == 1


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_healthy_status(self, mock_scheduler):
        """Test health check returns healthy status when all is well."""
        # Mock scheduler jobs
        mock_job = Mock()
        mock_job.id = "test_job"
        mock_job.name = "Test Job"
        mock_job.next_run_time = datetime.now()
        mock_job.pending = False

        mock_scheduler.scheduler = Mock()
        mock_scheduler.scheduler.get_jobs = Mock(return_value=[mock_job])
        mock_scheduler.is_running = True

        # Mock database check
        with patch("modules.downloader.scheduler.session_scope"):
            # Mock disk space check
            with patch.object(mock_scheduler, "_check_disk_space", return_value=50.0):
                health = mock_scheduler._health_check()

        assert health["status"] == "healthy"
        assert health["is_running"] is True
        assert len(health["jobs"]) == 1
        assert health["jobs"][0]["id"] == "test_job"
        assert health["disk_usage_percent"] == 50.0

    def test_health_check_degraded_status(self, mock_scheduler):
        """Test health check returns degraded status when database is down."""
        mock_scheduler.scheduler = Mock()
        mock_scheduler.scheduler.get_jobs = Mock(return_value=[])
        mock_scheduler.is_running = True

        # Mock database failure
        with patch(
            "modules.downloader.scheduler.session_scope",
            side_effect=Exception("DB Error"),
        ):
            with patch.object(mock_scheduler, "_check_disk_space", return_value=50.0):
                health = mock_scheduler._health_check()

        assert health["status"] == "degraded"
        assert health["database"] == "disconnected"

    def test_health_check_critical_disk_usage(self, mock_scheduler):
        """Test health check returns critical status for high disk usage."""
        mock_scheduler.scheduler = Mock()
        mock_scheduler.scheduler.get_jobs = Mock(return_value=[])
        mock_scheduler.is_running = True

        with patch("modules.downloader.scheduler.session_scope"):
            with patch.object(mock_scheduler, "_check_disk_space", return_value=95.0):
                with patch.object(mock_scheduler, "_send_alert") as mock_alert:
                    health = mock_scheduler._health_check()

        assert health["status"] == "critical"
        assert health["disk_usage_percent"] == 95.0
        mock_alert.assert_called_once_with("Disk usage critical: 95.0%")

    def test_health_check_warning_disk_usage(self, mock_scheduler):
        """Test health check returns warning status for high disk usage."""
        mock_scheduler.scheduler = Mock()
        mock_scheduler.scheduler.get_jobs = Mock(return_value=[])
        mock_scheduler.is_running = True

        with patch("modules.downloader.scheduler.session_scope"):
            with patch.object(mock_scheduler, "_check_disk_space", return_value=85.0):
                health = mock_scheduler._health_check()

        assert health["status"] == "warning"
        assert health["disk_usage_percent"] == 85.0

    def test_health_check_file_creation(self, mock_scheduler):
        """Test that health check creates .health file."""
        mock_scheduler.scheduler = Mock()
        mock_scheduler.scheduler.get_jobs = Mock(return_value=[])
        mock_scheduler.is_running = True

        with patch("modules.downloader.scheduler.session_scope"):
            with patch.object(mock_scheduler, "_check_disk_space", return_value=50.0):
                mock_scheduler._health_check()

        health_file = os.path.join(mock_scheduler.config.download_dir, ".health")
        assert os.path.exists(health_file)

        # Verify file content
        with open(health_file, "r") as f:
            health_data = json.load(f)

        assert "status" in health_data
        assert "timestamp" in health_data


class TestJobConfiguration:
    """Test that jobs are configured correctly."""

    def test_scheduler_job_configuration(self, mock_scheduler):
        """Test that jobs are configured with correct intervals."""
        # Mock APScheduler classes to avoid actually starting scheduler
        from apscheduler.schedulers.background import BackgroundScheduler

        with patch.object(BackgroundScheduler, "add_job") as mock_add_job:
            with patch.object(BackgroundScheduler, "add_listener") as mock_add_listener:
                with patch.object(BackgroundScheduler, "start"):
                    with patch.object(BackgroundScheduler, "get_jobs", return_value=[]):
                        with patch.object(mock_scheduler, "_print_schedule"):
                            mock_scheduler.start(blocking=False)

        # Verify add_job was called for each expected job
        add_job_calls = mock_add_job.call_args_list

        # Extract job IDs from calls
        job_ids = [call[1]["id"] for call in add_job_calls]

        assert "download_images" in job_ids
        assert "cleanup_images" in job_ids
        assert "health_check" in job_ids  # Should be included if enabled

    def test_download_job_interval(self, mock_scheduler):
        """Test that download job has correct interval."""
        from apscheduler.schedulers.background import BackgroundScheduler

        with patch.object(BackgroundScheduler, "add_job") as mock_add_job:
            with patch.object(BackgroundScheduler, "add_listener"):
                with patch.object(BackgroundScheduler, "start"):
                    with patch.object(BackgroundScheduler, "get_jobs", return_value=[]):
                        with patch.object(mock_scheduler, "_print_schedule"):
                            mock_scheduler.start(blocking=False)

        # Find the download job call
        add_job_calls = mock_add_job.call_args_list
        download_job_call = next(
            call for call in add_job_calls if call[1]["id"] == "download_images"
        )

        assert download_job_call[1]["minutes"] == 10  # download_interval_minutes
        assert download_job_call[1]["max_instances"] == 1
        assert download_job_call[1]["coalesce"] is True

    def test_cleanup_job_interval(self, mock_scheduler):
        """Test that cleanup job has correct interval."""
        from apscheduler.schedulers.background import BackgroundScheduler

        with patch.object(BackgroundScheduler, "add_job") as mock_add_job:
            with patch.object(BackgroundScheduler, "add_listener"):
                with patch.object(BackgroundScheduler, "start"):
                    with patch.object(BackgroundScheduler, "get_jobs", return_value=[]):
                        with patch.object(mock_scheduler, "_print_schedule"):
                            mock_scheduler.start(blocking=False)

        # Find the cleanup job call
        add_job_calls = mock_add_job.call_args_list
        cleanup_job_call = next(
            call for call in add_job_calls if call[1]["id"] == "cleanup_images"
        )

        assert cleanup_job_call[1]["hours"] == 24  # cleanup_interval_hours
        assert cleanup_job_call[1]["max_instances"] == 1
        assert cleanup_job_call[1]["coalesce"] is True

    def test_health_check_job_interval(self, mock_scheduler):
        """Test that health check job has correct interval."""
        from apscheduler.schedulers.background import BackgroundScheduler

        mock_scheduler.config.enable_health_check = True

        with patch.object(BackgroundScheduler, "add_job") as mock_add_job:
            with patch.object(BackgroundScheduler, "add_listener"):
                with patch.object(BackgroundScheduler, "start"):
                    with patch.object(BackgroundScheduler, "get_jobs", return_value=[]):
                        with patch.object(mock_scheduler, "_print_schedule"):
                            mock_scheduler.start(blocking=False)

        # Find the health check job call
        add_job_calls = mock_add_job.call_args_list
        health_job_call = next(
            call for call in add_job_calls if call[1]["id"] == "health_check"
        )

        assert health_job_call[1]["minutes"] == 5  # Health check every 5 minutes
        assert health_job_call[1]["max_instances"] == 1
        assert health_job_call[1]["coalesce"] is True

    def test_job_configuration_without_health_check(self, mock_scheduler):
        """Test that health check job is not added when disabled."""
        from apscheduler.schedulers.background import BackgroundScheduler

        mock_scheduler.config.enable_health_check = False

        with patch.object(BackgroundScheduler, "add_job") as mock_add_job:
            with patch.object(BackgroundScheduler, "add_listener"):
                with patch.object(BackgroundScheduler, "start"):
                    with patch.object(BackgroundScheduler, "get_jobs", return_value=[]):
                        with patch.object(mock_scheduler, "_print_schedule"):
                            mock_scheduler.start(blocking=False)

        # Verify health check job was not added
        add_job_calls = mock_add_job.call_args_list
        job_ids = [call[1]["id"] for call in add_job_calls]

        assert "health_check" not in job_ids

    def test_event_listener_configuration(self, mock_scheduler):
        """Test that event listener is configured correctly."""
        from apscheduler.schedulers.background import BackgroundScheduler

        with patch.object(BackgroundScheduler, "add_job"):
            with patch.object(BackgroundScheduler, "add_listener") as mock_add_listener:
                with patch.object(BackgroundScheduler, "start"):
                    with patch.object(BackgroundScheduler, "get_jobs", return_value=[]):
                        with patch.object(mock_scheduler, "_print_schedule"):
                            mock_scheduler.start(blocking=False)

        # Verify listener was added with correct events
        mock_add_listener.assert_called_once()
        call_args = mock_add_listener.call_args

        # Should listen for executed, error, and missed events
        assert call_args[0][0] == mock_scheduler._job_listener


class TestDiskSpaceCheck:
    """Test disk space checking functionality."""

    def test_check_disk_space(self, mock_scheduler):
        """Test disk space calculation."""
        # Mock shutil.disk_usage
        mock_usage = Mock()
        mock_usage.total = 1000
        mock_usage.used = 200

        with patch("shutil.disk_usage", return_value=mock_usage):
            usage_percent = mock_scheduler._check_disk_space()

        assert usage_percent == 20.0  # 200/1000 * 100


class TestAlertSystem:
    """Test alert notification system."""

    def test_send_alert_with_email(self, mock_scheduler):
        """Test alert sending when email is configured."""
        mock_scheduler.config.alert_email = "test@example.com"

        with patch("modules.downloader.scheduler.logger") as mock_logger:
            mock_scheduler._send_alert("Test alert message")

        mock_logger.warning.assert_called_once_with("ALERT: Test alert message")

    def test_send_alert_without_email(self, mock_scheduler):
        """Test alert sending when no email is configured."""
        mock_scheduler.config.alert_email = None

        with patch("modules.downloader.scheduler.logger") as mock_logger:
            mock_scheduler._send_alert("Test alert message")

        mock_logger.warning.assert_called_once_with(
            "ALERT (no email configured): Test alert message"
        )


class TestJobRetryWrapper:
    """Test the download job retry wrapper."""

    def test_download_job_with_retry_success(self, mock_scheduler):
        """Test that retry wrapper calls download job successfully."""
        with patch.object(mock_scheduler, "_download_with_retry") as mock_retry:
            with patch.object(mock_scheduler, "_download_images_job") as mock_job:
                mock_scheduler._download_images_job_with_retry()

        mock_retry.assert_called_once_with(mock_job)

    def test_download_job_with_retry_failure(self, mock_scheduler):
        """Test that retry wrapper handles failures correctly."""
        with patch.object(
            mock_scheduler,
            "_download_with_retry",
            side_effect=Exception("Download failed"),
        ):
            with patch.object(mock_scheduler, "_send_alert") as mock_alert:
                mock_scheduler._download_images_job_with_retry()

        mock_alert.assert_called_once_with("Download job failed: Download failed")
