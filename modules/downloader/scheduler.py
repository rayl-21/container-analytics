"""
APScheduler-based automatic image downloader for Dray Dog terminal cameras.

This module provides scheduled downloading capabilities using APScheduler,
running downloads every 10 minutes to match the Dray Dog capture interval.
"""

import os
import sys
import signal
import atexit
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import json
import threading

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from loguru import logger

from .selenium_client import DrayDogDownloader


@dataclass
class DownloadConfig:
    """Configuration for scheduled downloads."""

    stream_names: List[str] = None
    download_dir: str = "data/images"
    interval_minutes: int = 10
    headless: bool = True
    max_retries: int = 3
    enable_cleanup: bool = True
    cleanup_days: int = 7

    def __post_init__(self):
        if self.stream_names is None:
            self.stream_names = ["in_gate"]


@dataclass
class DownloadStats:
    """Statistics tracking for downloads."""

    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_images: int = 0
    last_run_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_error: Optional[str] = None


class ImageDownloadScheduler:
    """
    Automated scheduler for downloading camera images from Dray Dog.

    This class manages:
    - Scheduled downloads every 10 minutes
    - Multiple camera streams
    - Download statistics and logging
    - Automatic cleanup of old images
    - Graceful shutdown handling
    """

    def __init__(self, config: DownloadConfig):
        """
        Initialize the download scheduler.

        Args:
            config: Configuration object with download settings
        """
        self.config = config
        self.scheduler = None
        self.stats = DownloadStats()
        self.is_running = False
        self._shutdown_event = threading.Event()

        # Setup logging
        self._setup_logging()

        # Load existing stats if available
        self._load_stats()

        # Setup graceful shutdown
        self._setup_signal_handlers()

        logger.info("ImageDownloadScheduler initialized")

    def _setup_logging(self):
        """Configure logging for the scheduler."""
        log_dir = os.path.join(self.config.download_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "scheduler.log")

        # Add file logging with rotation
        logger.add(
            log_file,
            rotation="10 MB",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def _load_stats(self):
        """Load existing statistics from file."""
        stats_file = os.path.join(self.config.download_dir, "scheduler_stats.json")

        try:
            if os.path.exists(stats_file):
                with open(stats_file, "r") as f:
                    stats_data = json.load(f)

                # Convert datetime strings back to datetime objects
                if stats_data.get("last_run_time"):
                    stats_data["last_run_time"] = datetime.fromisoformat(
                        stats_data["last_run_time"]
                    )
                if stats_data.get("last_success_time"):
                    stats_data["last_success_time"] = datetime.fromisoformat(
                        stats_data["last_success_time"]
                    )

                self.stats = DownloadStats(**stats_data)
                logger.info("Loaded existing scheduler statistics")

        except Exception as e:
            logger.warning(f"Failed to load scheduler stats: {e}")

    def _save_stats(self):
        """Save current statistics to file."""
        stats_file = os.path.join(self.config.download_dir, "scheduler_stats.json")

        try:
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)

            # Convert datetime objects to strings for JSON serialization
            stats_dict = asdict(self.stats)
            if stats_dict["last_run_time"]:
                stats_dict["last_run_time"] = self.stats.last_run_time.isoformat()
            if stats_dict["last_success_time"]:
                stats_dict["last_success_time"] = (
                    self.stats.last_success_time.isoformat()
                )

            with open(stats_file, "w") as f:
                json.dump(stats_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save scheduler stats: {e}")

    def _download_images_job(self):
        """
        Job function that performs the actual image downloads.
        This is called by the scheduler at regular intervals.
        """
        job_start_time = datetime.now()
        self.stats.total_runs += 1
        self.stats.last_run_time = job_start_time

        logger.info(f"Starting scheduled download job (run #{self.stats.total_runs})")

        total_downloaded = 0
        errors = []

        try:
            # Download images for each configured stream
            for stream_name in self.config.stream_names:
                try:
                    logger.info(f"Downloading images for stream: {stream_name}")

                    with DrayDogDownloader(
                        download_dir=self.config.download_dir,
                        headless=self.config.headless,
                        max_retries=self.config.max_retries,
                    ) as downloader:

                        # Download images for current date (no authentication needed)
                        downloaded_files = downloader.download_images_for_date(
                            stream_name=stream_name
                        )

                        downloaded_count = len(downloaded_files)
                        total_downloaded += downloaded_count

                        logger.info(
                            f"Downloaded {downloaded_count} images for stream {stream_name}"
                        )

                except Exception as e:
                    error_msg = (
                        f"Failed to download images for stream {stream_name}: {str(e)}"
                    )
                    logger.error(error_msg)
                    errors.append(error_msg)

            # Update statistics
            if errors:
                self.stats.failed_runs += 1
                self.stats.last_error = "; ".join(errors[-3:])  # Keep last 3 errors
            else:
                self.stats.successful_runs += 1
                self.stats.last_success_time = job_start_time
                self.stats.last_error = None

            self.stats.total_images += total_downloaded

            # Log summary
            job_duration = (datetime.now() - job_start_time).total_seconds()
            logger.info(
                f"Download job completed in {job_duration:.1f}s: {total_downloaded} images downloaded"
            )

            if errors:
                logger.warning(f"Job completed with {len(errors)} errors")

            # Perform cleanup if enabled
            if self.config.enable_cleanup:
                self._cleanup_old_images()

        except Exception as e:
            self.stats.failed_runs += 1
            self.stats.last_error = str(e)
            logger.error(f"Download job failed: {e}")

        finally:
            self._save_stats()

    def _cleanup_old_images(self):
        """Remove images older than configured retention period."""
        if self.config.cleanup_days <= 0:
            return

        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.cleanup_days)
            cleaned_files = 0

            logger.info(
                f"Starting cleanup of images older than {self.config.cleanup_days} days"
            )

            # Walk through download directory
            for root, dirs, files in os.walk(self.config.download_dir):
                for file in files:
                    if file.endswith((".jpg", ".jpeg", ".png", ".json")):
                        file_path = os.path.join(root, file)

                        try:
                            # Check file modification time
                            file_mtime = datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            )

                            if file_mtime < cutoff_date:
                                os.remove(file_path)
                                cleaned_files += 1
                                logger.debug(f"Cleaned up old file: {file}")

                        except Exception as e:
                            logger.warning(f"Failed to clean up file {file}: {e}")

            if cleaned_files > 0:
                logger.info(f"Cleanup completed: {cleaned_files} old files removed")

        except Exception as e:
            logger.error(f"Cleanup process failed: {e}")

    def _job_listener(self, event):
        """Event listener for scheduler job events."""
        if event.exception:
            logger.error(f"Job {event.job_id} crashed: {event.exception}")
        else:
            logger.debug(f"Job {event.job_id} executed successfully")

    def start(self, blocking: bool = True):
        """
        Start the scheduled downloads.

        Args:
            blocking: If True, run scheduler in blocking mode (main thread)
                     If False, run in background thread
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        try:
            # Choose scheduler type based on blocking preference
            if blocking:
                self.scheduler = BlockingScheduler()
            else:
                self.scheduler = BackgroundScheduler()

            # Add job event listener
            self.scheduler.add_listener(
                self._job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
            )

            # Schedule the download job with interval trigger
            self.scheduler.add_job(
                func=self._download_images_job,
                trigger=IntervalTrigger(minutes=self.config.interval_minutes),
                id="download_images",
                name="Download Dray Dog Images",
                max_instances=1,  # Prevent overlapping runs
                coalesce=True,  # Combine missed executions
                misfire_grace_time=300,  # 5 minute grace period for missed jobs
            )

            # Add daily cleanup job at 3 AM
            if self.config.enable_cleanup:
                self.scheduler.add_job(
                    func=self._cleanup_old_images,
                    trigger=CronTrigger(hour=3, minute=0),
                    id="cleanup_images",
                    name="Cleanup Old Images",
                    max_instances=1,
                )

            # Start scheduler
            self.scheduler.start()
            self.is_running = True

            logger.info(
                f"Scheduler started - downloading every {self.config.interval_minutes} minutes"
            )
            logger.info(f"Configured streams: {', '.join(self.config.stream_names)}")

            # Run initial download job if requested
            logger.info("Running initial download job...")
            self._download_images_job()

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.cleanup()
            raise

    def stop(self):
        """Stop the scheduler gracefully."""
        if not self.is_running:
            return

        logger.info("Stopping scheduler...")

        if self.scheduler:
            try:
                self.scheduler.shutdown(wait=True)
                logger.info("Scheduler stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")

        self.is_running = False
        self._shutdown_event.set()

    def get_stats(self) -> Dict:
        """
        Get current download statistics.

        Returns:
            Dictionary containing download statistics
        """
        stats_dict = asdict(self.stats)

        # Add calculated fields
        success_rate = (
            (self.stats.successful_runs / self.stats.total_runs * 100)
            if self.stats.total_runs > 0
            else 0
        )
        stats_dict["success_rate"] = f"{success_rate:.1f}%"

        avg_images = (
            self.stats.total_images / self.stats.successful_runs
            if self.stats.successful_runs > 0
            else 0
        )
        stats_dict["average_images_per_run"] = f"{avg_images:.1f}"

        return stats_dict

    def print_stats(self):
        """Print current statistics to console."""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print("DOWNLOAD SCHEDULER STATISTICS")
        print("=" * 50)
        print(f"Total Runs:           {stats['total_runs']}")
        print(f"Successful Runs:      {stats['successful_runs']}")
        print(f"Failed Runs:          {stats['failed_runs']}")
        print(f"Success Rate:         {stats['success_rate']}")
        print(f"Total Images:         {stats['total_images']}")
        print(f"Avg Images/Run:       {stats['average_images_per_run']}")

        if stats["last_run_time"]:
            print(f"Last Run:             {stats['last_run_time']}")
        if stats["last_success_time"]:
            print(f"Last Success:         {stats['last_success_time']}")
        if stats["last_error"]:
            print(f"Last Error:           {stats['last_error']}")

        print("=" * 50)

    def cleanup(self):
        """Clean up resources."""
        if self.is_running:
            self.stop()

        # Save final stats
        self._save_stats()

        logger.info("Scheduler cleanup completed")


# CLI interface and main execution
def main():
    """Main function for running the scheduler from command line."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Automated Dray Dog image download scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration File Format (JSON):
{
  "stream_names": ["in_gate", "out_gate"],
  "download_dir": "data/images",
  "interval_minutes": 10,
  "headless": true,
  "max_retries": 3,
  "enable_cleanup": true,
  "cleanup_days": 7
}

Examples:
  python scheduler.py --config config.json
  python scheduler.py --streams in_gate out_gate
  python scheduler.py --config config.json --stats
        """,
    )

    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument(
        "--streams", nargs="+", default=["in_gate"], help="Camera stream names"
    )
    parser.add_argument("--output", default="data/images", help="Output directory")
    parser.add_argument(
        "--interval", type=int, default=10, help="Download interval in minutes"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run browser in headless mode"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable automatic cleanup of old images",
    )
    parser.add_argument(
        "--cleanup-days", type=int, default=7, help="Days to keep images before cleanup"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum retry attempts"
    )
    parser.add_argument("--stats", action="store_true", help="Show statistics and exit")
    parser.add_argument(
        "--daemon", action="store_true", help="Run as background daemon"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        try:
            with open(args.config, "r") as f:
                config_dict = json.load(f)
            config = DownloadConfig(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            return 1
    else:
        # Create config from command line arguments (no authentication needed)
        config = DownloadConfig(
            stream_names=args.streams,
            download_dir=args.output,
            interval_minutes=args.interval,
            headless=args.headless,
            max_retries=args.max_retries,
            enable_cleanup=not args.no_cleanup,
            cleanup_days=args.cleanup_days,
        )

    # Initialize scheduler
    scheduler = ImageDownloadScheduler(config)

    # Show stats and exit if requested
    if args.stats:
        scheduler.print_stats()
        return 0

    try:
        # Start scheduler
        logger.info("Starting Dray Dog image download scheduler...")
        logger.info(f"Configuration: {config}")

        # Run in appropriate mode
        blocking_mode = not args.daemon
        scheduler.start(blocking=blocking_mode)

        if args.daemon:
            # Keep daemon running until shutdown signal
            logger.info("Running as background daemon. Send SIGTERM or SIGINT to stop.")
            scheduler._shutdown_event.wait()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        return 1
    finally:
        scheduler.cleanup()

    logger.info("Scheduler shutdown complete")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
