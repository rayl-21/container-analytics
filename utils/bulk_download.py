"""
Bulk image downloader for Dray Dog terminal cameras.

This module provides functionality to download images from multiple dates
and organize them efficiently. It extends the existing selenium_client.py
functionality for batch operations.
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

# Import existing downloader
from ..modules.downloader.selenium_client import DrayDogDownloader
from ..modules.database import queries
from ..modules.database.models import session_scope


@dataclass
class DownloadStats:
    """Statistics for a bulk download operation."""
    total_requested: int = 0
    successful_downloads: int = 0
    failed_downloads: int = 0
    already_existed: int = 0
    total_file_size: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate download duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requested == 0:
            return 0.0
        return (self.successful_downloads / self.total_requested) * 100


class BulkImageDownloader:
    """
    Bulk image downloader for Dray Dog terminal cameras.
    
    This class handles:
    - Downloading images across multiple date ranges
    - Organizing files by date and stream
    - Saving metadata to database
    - Progress tracking and error handling
    - Generating comprehensive download reports
    """
    
    def __init__(
        self,
        download_dir: str = "data/images",
        headless: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
        use_direct_download: bool = True
    ):
        """
        Initialize the bulk downloader.
        
        Args:
            download_dir: Base directory for downloaded images
            headless: Whether to run browser in headless mode
            max_retries: Maximum retry attempts per image
            retry_delay: Base delay between retries
            timeout: Timeout for web operations
            use_direct_download: Use direct URL construction vs Selenium
        """
        self.download_dir = Path(download_dir)
        self.headless = headless
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.use_direct_download = use_direct_download
        
        # Ensure base download directory exists
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize downloader instance
        self.downloader = DrayDogDownloader(
            download_dir=str(self.download_dir),
            headless=headless,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout
        )
        
        # Statistics tracking
        self.stats = DownloadStats()
        
        logger.info(f"BulkImageDownloader initialized with download_dir: {self.download_dir}")
    
    def download_date_range(
        self,
        start_date: str,
        end_date: str,
        streams: List[str],
        max_images_per_date: Optional[int] = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Download images for a range of dates across multiple streams.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (inclusive)
            streams: List of stream names (e.g., ['in_gate', 'out_gate'])
            max_images_per_date: Optional limit on images per date per stream
            
        Returns:
            Dictionary mapping dates -> streams -> list of downloaded file paths
        """
        logger.info(f"Starting bulk download: {start_date} to {end_date} for streams: {streams}")
        
        # Initialize stats
        self.stats = DownloadStats()
        self.stats.start_time = datetime.utcnow()
        
        # Parse date range
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            raise ValueError(f"Dates must be in YYYY-MM-DD format: {e}")
        
        if start_dt > end_dt:
            raise ValueError("Start date must be before or equal to end date")
        
        results = {}
        current_dt = start_dt
        
        # Download for each date
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y-%m-%d")
            logger.info(f"Processing date: {date_str}")
            
            results[date_str] = {}
            
            # Download for each stream
            for stream in streams:
                logger.info(f"Downloading {stream} stream for {date_str}")
                
                try:
                    if self.use_direct_download:
                        downloaded_files = self.downloader.download_images_direct(
                            date_str=date_str,
                            stream_name=stream,
                            max_images=max_images_per_date,
                            use_actual_timestamps=True
                        )
                    else:
                        downloaded_files = self.downloader.download_images_for_date(
                            date_str=date_str,
                            stream_name=stream
                        )
                    
                    results[date_str][stream] = downloaded_files
                    
                    # Update statistics
                    self.stats.successful_downloads += len(downloaded_files)
                    
                    # Calculate file sizes
                    for filepath in downloaded_files:
                        if os.path.exists(filepath):
                            self.stats.total_file_size += os.path.getsize(filepath)
                    
                    logger.info(f"Downloaded {len(downloaded_files)} images for {stream} on {date_str}")
                    
                except Exception as e:
                    logger.error(f"Failed to download {stream} for {date_str}: {e}")
                    results[date_str][stream] = []
                    self.stats.failed_downloads += 1
            
            # Move to next day
            current_dt += timedelta(days=1)
        
        # Finalize stats
        self.stats.end_time = datetime.utcnow()
        
        # Organize files if downloads were successful
        if self.stats.successful_downloads > 0:
            self._organize_downloaded_files(results)
        
        logger.info(f"Bulk download completed. Success rate: {self.stats.success_rate:.1f}%")
        return results
    
    def save_to_database(self, image_metadata: List[Dict]) -> int:
        """
        Save image metadata to database using existing models.
        
        Args:
            image_metadata: List of metadata dictionaries with keys:
                - filepath: Path to image file
                - camera_id: Camera/stream identifier  
                - timestamp: When image was captured
                - file_size: File size in bytes
                
        Returns:
            Number of records successfully saved
        """
        saved_count = 0
        
        try:
            with session_scope() as session:
                for metadata in image_metadata:
                    try:
                        # Use existing insert_image function
                        image_id = queries.insert_image(
                            filepath=metadata['filepath'],
                            camera_id=metadata['camera_id'],
                            timestamp=metadata.get('timestamp'),
                            file_size=metadata.get('file_size')
                        )
                        
                        if image_id:
                            saved_count += 1
                            logger.debug(f"Saved image {image_id}: {metadata['filepath']}")
                    
                    except Exception as e:
                        logger.error(f"Failed to save image metadata: {e}")
                        continue
                
                logger.info(f"Successfully saved {saved_count}/{len(image_metadata)} image records")
                
        except Exception as e:
            logger.error(f"Database save operation failed: {e}")
        
        return saved_count
    
    def organize_files(self, source_path: str, target_path: str) -> bool:
        """
        Organize files from source to target directory structure.
        
        Creates organized structure: target_path/YYYY-MM-DD/stream_name/
        
        Args:
            source_path: Source directory containing downloaded files
            target_path: Target directory for organized structure
            
        Returns:
            True if organization successful, False otherwise
        """
        try:
            source_dir = Path(source_path)
            target_dir = Path(target_path)
            
            if not source_dir.exists():
                logger.error(f"Source directory does not exist: {source_path}")
                return False
            
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            files_moved = 0
            
            # Walk through source directory
            for file_path in source_dir.rglob("*.jpg"):
                if file_path.is_file():
                    # Extract date and stream from filename or path
                    date_str, stream_name = self._extract_file_info(file_path)
                    
                    if date_str and stream_name:
                        # Create target directory structure
                        target_subdir = target_dir / date_str / stream_name
                        target_subdir.mkdir(parents=True, exist_ok=True)
                        
                        # Move file
                        target_file = target_subdir / file_path.name
                        if not target_file.exists():
                            shutil.move(str(file_path), str(target_file))
                            files_moved += 1
                            logger.debug(f"Moved {file_path.name} to {target_subdir}")
                        else:
                            logger.debug(f"File already exists at target: {target_file}")
            
            logger.info(f"Organized {files_moved} files from {source_path} to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"File organization failed: {e}")
            return False
    
    def generate_download_report(self) -> Dict:
        """
        Generate a comprehensive download report.
        
        Returns:
            Dictionary containing download statistics and summary
        """
        report = {
            "summary": {
                "total_requested": self.stats.total_requested,
                "successful_downloads": self.stats.successful_downloads,
                "failed_downloads": self.stats.failed_downloads,
                "already_existed": self.stats.already_existed,
                "success_rate_percent": self.stats.success_rate,
                "total_file_size_mb": self.stats.total_file_size / (1024 * 1024),
                "duration_seconds": self.stats.duration.total_seconds() if self.stats.duration else 0,
                "download_speed_mb_per_sec": (
                    (self.stats.total_file_size / (1024 * 1024)) / self.stats.duration.total_seconds()
                    if self.stats.duration and self.stats.duration.total_seconds() > 0
                    else 0
                )
            },
            "timestamps": {
                "start_time": self.stats.start_time.isoformat() if self.stats.start_time else None,
                "end_time": self.stats.end_time.isoformat() if self.stats.end_time else None
            },
            "configuration": {
                "download_dir": str(self.download_dir),
                "headless": self.headless,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "timeout": self.timeout,
                "use_direct_download": self.use_direct_download
            }
        }
        
        return report
    
    def save_report(self, report: Dict, report_path: Optional[str] = None) -> str:
        """
        Save download report to JSON file.
        
        Args:
            report: Report dictionary from generate_download_report()
            report_path: Optional path for report file
            
        Returns:
            Path to saved report file
        """
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.download_dir / f"bulk_download_report_{timestamp}.json"
        else:
            report_path = Path(report_path)
        
        # Ensure parent directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Download report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise
    
    def _organize_downloaded_files(self, results: Dict[str, Dict[str, List[str]]]) -> None:
        """
        Organize downloaded files into proper directory structure.
        
        Args:
            results: Download results dictionary
        """
        try:
            for date_str, streams in results.items():
                for stream_name, file_paths in streams.items():
                    if file_paths:
                        # Ensure proper directory structure
                        target_dir = self.download_dir / date_str / stream_name
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Move files if they're not already in the right place
                        for file_path in file_paths:
                            source_file = Path(file_path)
                            if source_file.exists():
                                target_file = target_dir / source_file.name
                                
                                # Only move if not already in correct location
                                if source_file.parent != target_dir:
                                    if not target_file.exists():
                                        shutil.move(str(source_file), str(target_file))
                                        logger.debug(f"Moved {source_file.name} to {target_dir}")
        
        except Exception as e:
            logger.warning(f"File organization warning: {e}")
    
    def _extract_file_info(self, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract date and stream information from file path or filename.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (date_str, stream_name) or (None, None) if extraction fails
        """
        try:
            # Try to extract from filename
            filename = file_path.stem
            
            # Look for date pattern in filename (YYYYMMDD format)
            import re
            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
            if date_match:
                year, month, day = date_match.groups()
                date_str = f"{year}-{month}-{day}"
            else:
                # Try to extract from parent directory
                parts = file_path.parts
                for part in parts:
                    if re.match(r'\d{4}-\d{2}-\d{2}', part):
                        date_str = part
                        break
                else:
                    date_str = None
            
            # Extract stream name
            stream_name = None
            if 'in_gate' in filename:
                stream_name = 'in_gate'
            elif 'out_gate' in filename:
                stream_name = 'out_gate'
            else:
                # Check parent directory
                for part in file_path.parts:
                    if part in ['in_gate', 'out_gate']:
                        stream_name = part
                        break
            
            return date_str, stream_name
            
        except Exception as e:
            logger.debug(f"Failed to extract file info from {file_path}: {e}")
            return None, None
    
    def cleanup(self):
        """Clean up resources."""
        if self.downloader:
            self.downloader.cleanup()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


def main():
    """
    Main function for running bulk download as a script.
    Downloads images from 2025-09-01 to 2025-09-07 for both streams.
    """
    # Default configuration
    start_date = "2025-09-01"
    end_date = "2025-09-07"
    streams = ["in_gate", "out_gate"]
    
    logger.info(f"Starting bulk download: {start_date} to {end_date}")
    
    try:
        with BulkImageDownloader(
            download_dir="data/images",
            headless=True,
            use_direct_download=True
        ) as downloader:
            
            # Download images
            results = downloader.download_date_range(
                start_date=start_date,
                end_date=end_date,
                streams=streams
            )
            
            # Generate and save report
            report = downloader.generate_download_report()
            report_path = downloader.save_report(report)
            
            # Print summary
            print(f"\n=== Bulk Download Complete ===")
            print(f"Date Range: {start_date} to {end_date}")
            print(f"Streams: {', '.join(streams)}")
            print(f"Success Rate: {report['summary']['success_rate_percent']:.1f}%")
            print(f"Total Files: {report['summary']['successful_downloads']}")
            print(f"Total Size: {report['summary']['total_file_size_mb']:.2f} MB")
            print(f"Duration: {report['summary']['duration_seconds']:.1f} seconds")
            print(f"Report saved to: {report_path}")
            
    except Exception as e:
        logger.error(f"Bulk download failed: {e}")
        raise


if __name__ == "__main__":
    main()