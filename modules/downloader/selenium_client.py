"""
Selenium-based image downloader for Dray Dog terminal cameras.

This module provides automated downloading of camera images from the Dray Dog
camera history page using Selenium WebDriver.
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.chrome import ChromeDriverManager
from loguru import logger


class DrayDogDownloader:
    """
    Automated image downloader for Dray Dog terminal camera system.

    This class handles:
    - Direct navigation to camera history pages (no login required)
    - Image URL extraction
    - Downloading images with metadata
    - Retry logic with exponential backoff
    """

    def __init__(
        self,
        download_dir: str = "data/images",
        headless: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
    ):
        """
        Initialize the Dray Dog downloader.

        Args:
            download_dir: Directory to save downloaded images
            headless: Whether to run browser in headless mode
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (seconds)
            timeout: Default timeout for web elements (seconds)
        """
        self.download_dir = download_dir
        self.headless = headless
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        # Ensure download directory exists
        os.makedirs(self.download_dir, exist_ok=True)

        # Initialize driver as None
        self.driver: Optional[webdriver.Chrome] = None

        logger.info(f"DrayDogDownloader initialized with download_dir: {download_dir}")

    def _setup_driver(self) -> webdriver.Chrome:
        """
        Set up Chrome WebDriver with optimal settings.

        Returns:
            Configured Chrome WebDriver instance
        """
        try:
            options = Options()

            if self.headless:
                options.add_argument("--headless")

            # Stealth options to avoid detection
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-extensions")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            # Disable images for faster loading (we'll download them separately)
            prefs = {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_settings.popups": 0,
            }
            options.add_experimental_option("prefs", prefs)

            # Use webdriver-manager to handle driver installation
            driver = webdriver.Chrome(options=options)

            # Remove automation indicators
            driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            logger.info("Chrome WebDriver initialized successfully")
            return driver

        except Exception as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {e}")
            raise

    def _retry_operation(self, operation, *args, **kwargs):
        """
        Execute operation with exponential backoff retry logic.

        Args:
            operation: Function to execute
            *args, **kwargs: Arguments for the operation

        Returns:
            Result of the operation

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

        logger.error(f"Operation failed after {self.max_retries} attempts")
        raise last_exception

    def navigate_to_camera_history(
        self, stream_name: str = "in_gate", date_str: Optional[str] = None
    ) -> bool:
        """
        Navigate directly to the camera history page for a specific stream.
        No authentication required - direct public access.

        Args:
            stream_name: Camera stream name (e.g., 'in_gate', 'out_gate')
            date_str: Specific date in YYYY-MM-DD format, None for today

        Returns:
            True if navigation successful, False otherwise
        """
        try:
            if not self.driver:
                self.driver = self._setup_driver()

            # Construct URL - directly access without login
            base_url = f"https://app.draydog.com/terminal_cameras/apm?streamName={stream_name}#camera-history"

            logger.info(f"Navigating to camera history for stream: {stream_name}")
            self.driver.get(base_url)

            # Wait for page to load
            wait = WebDriverWait(self.driver, self.timeout)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "el-image")))

            # If specific date requested, navigate to it
            if date_str:
                self._navigate_to_date(date_str)

            logger.info("Successfully navigated to camera history page")
            return True

        except Exception as e:
            logger.error(f"Failed to navigate to camera history: {e}")
            return False

    def _navigate_to_date(self, date_str: str) -> bool:
        """
        Navigate to a specific date in the camera history.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            True if date navigation successful, False otherwise
        """
        try:
            # Click on date picker
            date_picker = self.driver.find_element(By.CLASS_NAME, "el-date-editor")
            date_picker.click()

            # Clear existing date and enter new date
            date_input = self.driver.find_element(
                By.CSS_SELECTOR, ".el-date-editor input"
            )
            date_input.clear()
            date_input.send_keys(date_str)

            # Press Enter or click outside to apply date change
            date_input.send_keys("\n")
            time.sleep(2)  # Wait for page to reload with new date

            logger.info(f"Navigated to date: {date_str}")
            return True

        except Exception as e:
            logger.warning(f"Failed to navigate to specific date {date_str}: {e}")
            return False

    def extract_image_urls(self) -> List[Dict[str, str]]:
        """
        Extract full-size image URLs and metadata from the current camera history page.
        Filters out thumbnail images and only returns full-resolution images.

        Returns:
            List of dictionaries containing image URL and metadata
        """
        try:
            if not self.driver:
                raise RuntimeError("Driver not initialized")

            # Wait for images to load
            wait = WebDriverWait(self.driver, self.timeout)
            wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".el-image__inner"))
            )

            # Extract image data using JavaScript
            image_data = self.driver.execute_script(
                """
                const images = [];
                const imageElements = document.querySelectorAll('.el-image__inner');
                
                imageElements.forEach((img, index) => {
                    if (img.src && img.src.includes('cdn.draydog.com')) {
                        // Skip thumbnail images - only get full-size images
                        if (img.src.includes('-thumbnail') || img.src.includes('_thumb')) {
                            return;
                        }
                        
                        const urlParts = img.src.split('/');
                        const filename = urlParts[urlParts.length - 1];
                        
                        // Also construct the full-size URL by removing any thumbnail suffix
                        let fullSizeUrl = img.src;
                        if (img.src.includes('-thumbnail')) {
                            fullSizeUrl = img.src.replace('-thumbnail', '');
                        }
                        
                        const timestampPart = filename.split('-')[0];
                        
                        // Try to extract additional metadata from DOM
                        const parentElement = img.closest('.el-image');
                        let captureTime = null;
                        
                        // Look for timestamp information in nearby elements
                        const timeElement = parentElement ? 
                            parentElement.querySelector('[data-timestamp], .timestamp, .time') : null;
                        if (timeElement) {
                            captureTime = timeElement.textContent || timeElement.getAttribute('data-timestamp');
                        }
                        
                        images.push({
                            url: fullSizeUrl,
                            filename: filename.replace('-thumbnail', '').replace('_thumb', ''),
                            timestamp: timestampPart,
                            captureTime: captureTime,
                            index: index,
                            streamName: new URLSearchParams(window.location.search).get('streamName') || 'unknown'
                        });
                    }
                });
                
                // If no full-size images found, try to get them from thumbnail URLs
                if (images.length === 0) {
                    imageElements.forEach((img, index) => {
                        if (img.src && img.src.includes('cdn.draydog.com')) {
                            // Convert thumbnail URL to full-size URL
                            let fullSizeUrl = img.src.replace('-thumbnail', '').replace('_thumb', '');
                            const urlParts = fullSizeUrl.split('/');
                            const filename = urlParts[urlParts.length - 1];
                            const timestampPart = filename.split('-')[0];
                            
                            images.push({
                                url: fullSizeUrl,
                                filename: filename,
                                timestamp: timestampPart,
                                captureTime: null,
                                index: index,
                                streamName: new URLSearchParams(window.location.search).get('streamName') || 'unknown'
                            });
                        }
                    });
                }
                
                return images;
            """
            )

            logger.info(f"Extracted {len(image_data)} full-size image URLs")
            return image_data

        except Exception as e:
            logger.error(f"Failed to extract image URLs: {e}")
            return []

    def download_image(self, image_info: Dict[str, str]) -> Optional[str]:
        """
        Download a single image with metadata.

        Args:
            image_info: Dictionary containing image URL and metadata

        Returns:
            Path to downloaded image file, or None if download failed
        """
        try:
            url = image_info["url"]
            filename = image_info["filename"]

            # Create subdirectory based on date
            date_part = (
                filename.split("T")[0]
                if "T" in filename
                else datetime.now().strftime("%Y-%m-%d")
            )
            date_dir = os.path.join(self.download_dir, date_part)
            os.makedirs(date_dir, exist_ok=True)

            # Full path for image
            image_path = os.path.join(date_dir, filename)

            # Skip if file already exists
            if os.path.exists(image_path):
                logger.debug(f"Image already exists: {filename}")
                return image_path

            # Download image
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Save image
            with open(image_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Save metadata
            metadata_path = os.path.join(date_dir, f"{filename}.metadata.json")
            metadata = {
                "filename": filename,
                "url": url,
                "download_timestamp": datetime.now().isoformat(),
                "capture_timestamp": image_info.get("timestamp"),
                "capture_time": image_info.get("captureTime"),
                "stream_name": image_info.get("streamName"),
                "file_size": os.path.getsize(image_path),
                "file_hash": self._calculate_file_hash(image_path),
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Successfully downloaded: {filename}")
            return image_path

        except requests.RequestException as e:
            logger.error(
                f"Network error downloading {image_info.get('filename', 'unknown')}: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to download {image_info.get('filename', 'unknown')}: {e}"
            )
            return None

    def _calculate_file_hash(self, filepath: str) -> str:
        """
        Calculate SHA-256 hash of a file.

        Args:
            filepath: Path to the file

        Returns:
            SHA-256 hash string
        """
        hash_sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""

    def download_images_for_date(
        self, date_str: Optional[str] = None, stream_name: str = "in_gate"
    ) -> List[str]:
        """
        Complete workflow to download all images for a specific date.
        No authentication required - direct public access.

        Args:
            date_str: Date in YYYY-MM-DD format, None for today
            stream_name: Camera stream name

        Returns:
            List of paths to downloaded images
        """
        downloaded_files = []

        try:
            # Navigate directly to camera history (no authentication)
            if not self.navigate_to_camera_history(stream_name, date_str):
                raise RuntimeError("Failed to navigate to camera history")

            # Extract image URLs
            image_data = self.extract_image_urls()
            if not image_data:
                logger.warning("No images found on current page")
                return downloaded_files

            # Download each image with retry logic
            for img_info in image_data:
                try:
                    filepath = self._retry_operation(self.download_image, img_info)
                    if filepath:
                        downloaded_files.append(filepath)
                except Exception as e:
                    logger.error(f"Failed to download image after retries: {e}")
                    continue

            logger.info(
                f"Download completed. {len(downloaded_files)} images downloaded successfully."
            )

        except Exception as e:
            logger.error(f"Download workflow failed: {e}")

        finally:
            self.cleanup()

        return downloaded_files

    def get_available_timestamps(self, date_str: str = "2025-09-06", stream_name: str = "in_gate") -> List[str]:
        """
        Fetch actual available timestamps from Dray Dog website for a specific date.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            stream_name: Camera stream name (default: in_gate)
            
        Returns:
            List of available timestamp strings (e.g., ["2025-09-06T23:07:34", ...])
        """
        timestamps = []
        
        try:
            # Setup driver if not already initialized
            if not self.driver:
                self.driver = self._setup_driver()
            
            # Navigate to the camera history page
            url = f"https://app.draydog.com/terminal_cameras/apm?streamName={stream_name}#camera-history"
            logger.info(f"Navigating to {url}")
            self.driver.get(url)
            
            # Wait for page to load
            time.sleep(5)
            
            # Try to select the correct date using the date picker
            try:
                # Parse the target date
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
                # Format like "Sat, Sep 6" - handle both single and double digit days
                day = target_date.day
                date_text = target_date.strftime(f"%a, %b {day}")  # Format like "Sat, Sep 6"
                
                # Try to find and click the date button
                date_button_script = f"""
                    const buttons = document.querySelectorAll('button');
                    for (const button of buttons) {{
                        if (button.textContent && button.textContent.includes('{date_text}')) {{
                            button.click();
                            return true;
                        }}
                    }}
                    return false;
                """
                
                clicked = self.driver.execute_script(date_button_script)
                if clicked:
                    logger.info(f"Clicked date button for {date_text}")
                    time.sleep(3)  # Wait for images to load
                else:
                    logger.warning(f"Could not find date button for {date_text}")
            except Exception as e:
                logger.warning(f"Could not select date: {e}")
            
            # Extract timestamps from thumbnail images
            script = f"""
                const images = document.querySelectorAll('img.el-image__inner');
                const timestamps = [];
                images.forEach(img => {{
                    if (img.src && img.src.includes('{date_str}')) {{
                        const match = img.src.match(/{date_str}T(\\d{{2}}:\\d{{2}}:\\d{{2}})/);
                        if (match) {{
                            timestamps.push('{date_str}T' + match[1]);
                        }}
                    }}
                }});
                return timestamps;
            """
            
            found_timestamps = self.driver.execute_script(script)
            
            if found_timestamps:
                timestamps = found_timestamps
                logger.info(f"Found {len(timestamps)} timestamps for {date_str}")
            else:
                logger.warning(f"No timestamps found for {date_str}, using default intervals")
                # Fallback to approximate timestamps
                timestamps = self._generate_fallback_timestamps(date_str)
            
        except Exception as e:
            logger.error(f"Error fetching timestamps: {e}")
            # Use fallback timestamps
            timestamps = self._generate_fallback_timestamps(date_str)
        
        return timestamps
    
    def _generate_fallback_timestamps(self, date_str: str) -> List[str]:
        """
        Generate fallback timestamps at approximate intervals.
        
        Based on observed patterns, images are roughly every 10 minutes
        but with varying seconds (not at exact :00).
        
        Args:
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            List of fallback timestamp strings
        """
        timestamps = []
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Common second offsets observed in real data
        second_offsets = [7, 17, 27, 39, 49, 57, 9, 19, 28, 39]
        
        for hour in range(24):
            for minute in range(0, 60, 10):
                # Use rotating second offsets to approximate real patterns
                seconds = second_offsets[(hour * 6 + minute // 10) % len(second_offsets)]
                timestamp = target_date.replace(hour=hour, minute=minute, second=seconds)
                timestamps.append(timestamp.strftime("%Y-%m-%dT%H:%M:%S"))
        
        return timestamps

    def download_images_direct(
        self, 
        date_str: str = "2025-09-06",
        stream_name: str = "in_gate",
        max_images: Optional[int] = None,
        interval_minutes: int = 10,
        use_actual_timestamps: bool = True
    ) -> List[str]:
        """
        Download images directly using URL construction without Selenium.
        
        This method constructs image URLs based on the known Dray Dog CDN pattern
        and downloads them directly without needing browser automation.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            stream_name: Camera stream name (default: in_gate)  
            max_images: Maximum number of images to download (None for all)
            interval_minutes: Interval between images in minutes (default: 10, ignored if use_actual_timestamps=True)
            use_actual_timestamps: If True, fetch actual timestamps from website first
        
        Returns:
            List of downloaded file paths
        """
        downloaded_files = []
        
        try:
            # Create date-specific directory
            date_dir = os.path.join(self.download_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)
            
            # Get timestamps to download
            if use_actual_timestamps:
                logger.info("Fetching actual timestamps from Dray Dog...")
                timestamps = self.get_available_timestamps(date_str, stream_name)
            else:
                # Use the old method with exact intervals
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
                timestamps = []
                current_time = target_date.replace(hour=0, minute=0, second=0)
                end_time = target_date.replace(hour=23, minute=59, second=59)
                
                while current_time <= end_time:
                    timestamps.append(current_time.strftime("%Y-%m-%dT%H:%M:%S"))
                    current_time += timedelta(minutes=interval_minutes)
            
            # Limit timestamps if max_images is set
            if max_images:
                timestamps = timestamps[:max_images]
            
            logger.info(f"Downloading {len(timestamps)} images...")
            
            for idx, timestamp_str in enumerate(timestamps):
                # Parse timestamp to get date and hour folders
                dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                date_folder = dt.strftime("%Y-%m-%d")
                hour_folder = str(dt.hour)
                
                # Construct the image URL
                # Pattern: https://cdn.draydog.com/apm/YYYY-MM-DD/H/YYYY-MM-DDTHH:MM:SS-stream_name.jpeg
                full_url = f"https://cdn.draydog.com/apm/{date_folder}/{hour_folder}/{timestamp_str}-{stream_name}.jpeg"
                thumbnail_url = f"https://cdn.draydog.com/apm/{date_folder}/{hour_folder}/{timestamp_str}-{stream_name}-thumbnail.jpeg"
                
                # Create filename for local storage
                filename = f"{timestamp_str.replace(':', '').replace('-', '')}_{stream_name}.jpg"
                filepath = os.path.join(date_dir, filename)
                
                # Skip if already downloaded
                if os.path.exists(filepath):
                    logger.debug(f"Already exists: {filename}")
                    downloaded_files.append(filepath)
                    continue
                
                # Try to download the image
                success = False
                
                # Try full-size image first
                try:
                    logger.info(f"Downloading [{idx+1}/{len(timestamps)}]: {full_url}")
                    response = requests.get(full_url, timeout=30, headers={
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                    })
                    
                    if response.status_code == 200:
                        # Check if we got actual image data
                        content_type = response.headers.get('content-type', '')
                        if 'image' in content_type or len(response.content) > 1000:
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            success = True
                            logger.info(f"✓ Downloaded full image: {filename}")
                    
                except Exception as e:
                    logger.debug(f"Full image failed: {e}")
                
                # Fall back to thumbnail if full image failed
                if not success:
                    try:
                        logger.info(f"Trying thumbnail: {thumbnail_url}")
                        response = requests.get(thumbnail_url, timeout=30, headers={
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                        })
                        
                        if response.status_code == 200:
                            content_type = response.headers.get('content-type', '')
                            if 'image' in content_type or len(response.content) > 1000:
                                with open(filepath, 'wb') as f:
                                    f.write(response.content)
                                success = True
                                logger.info(f"✓ Downloaded thumbnail: {filename}")
                        
                    except Exception as e:
                        logger.debug(f"Thumbnail also failed: {e}")
                
                if success:
                    downloaded_files.append(filepath)
                else:
                    logger.warning(f"✗ Failed to download image for {timestamp_str}")
            
            # Save metadata
            metadata_path = os.path.join(date_dir, "download_metadata.json")
            metadata = {
                "date": date_str,
                "stream": stream_name,
                "total_images": len(downloaded_files),
                "download_timestamp": datetime.now().isoformat(),
                "files": downloaded_files,
                "method": "direct_url_construction"
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Downloaded {len(downloaded_files)} images to {date_dir}")
            
        except Exception as e:
            logger.error(f"Direct download failed: {e}")
        
        return downloaded_files

    def download_images_for_date_range(
        self, start_date: str, end_date: str, stream_name: str = "in_gate"
    ) -> Dict[str, List[str]]:
        """
        Download images for a range of dates.
        No authentication required - direct public access.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            stream_name: Camera stream name

        Returns:
            Dictionary mapping dates to lists of downloaded file paths
        """
        results = {}

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            current_dt = start_dt
            while current_dt <= end_dt:
                date_str = current_dt.strftime("%Y-%m-%d")
                logger.info(f"Downloading images for date: {date_str}")

                downloaded_files = self.download_images_for_date(date_str, stream_name)
                results[date_str] = downloaded_files

                # Move to next day
                current_dt += timedelta(days=1)

                # Add delay between date downloads to be respectful
                time.sleep(5)

        except Exception as e:
            logger.error(f"Date range download failed: {e}")

        return results

    def cleanup(self):
        """Clean up resources by closing the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("WebDriver closed successfully")
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {e}")
            finally:
                self.driver = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Example usage and CLI interface
if __name__ == "__main__":
    import argparse
    import sys
    from datetime import datetime

    def main():
        parser = argparse.ArgumentParser(
            description="Download Dray Dog camera images",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python selenium_client.py --username myuser --password mypass
  python selenium_client.py --username myuser --password mypass --date 2025-09-07
  python selenium_client.py --username myuser --password mypass --stream out_gate
  python selenium_client.py --username myuser --password mypass --date-range 2025-09-01 2025-09-07
            """,
        )

        parser.add_argument("--username", required=True, help="Dray Dog username")
        parser.add_argument("--password", required=True, help="Dray Dog password")
        parser.add_argument("--date", help="Date in YYYY-MM-DD format (default: today)")
        parser.add_argument(
            "--date-range",
            nargs=2,
            metavar=("START", "END"),
            help="Date range in YYYY-MM-DD format",
        )
        parser.add_argument(
            "--stream", default="in_gate", help="Stream name (default: in_gate)"
        )
        parser.add_argument(
            "--output",
            default="data/images",
            help="Output directory (default: data/images)",
        )
        parser.add_argument(
            "--headless", action="store_true", help="Run browser in headless mode"
        )
        parser.add_argument(
            "--max-retries",
            type=int,
            default=3,
            help="Maximum retry attempts (default: 3)",
        )

        args = parser.parse_args()

        # Setup logging
        logger.add(sys.stderr, level="INFO")

        # Create downloader
        with DrayDogDownloader(
            download_dir=args.output,
            headless=args.headless,
            max_retries=args.max_retries,
        ) as downloader:

            if args.date_range:
                # Download date range
                results = downloader.download_images_for_date_range(
                    args.date_range[0],
                    args.date_range[1],
                    args.stream,
                )

                total_files = sum(len(files) for files in results.values())
                print(f"\nDownload completed!")
                print(f"Total images downloaded: {total_files}")
                for date, files in results.items():
                    print(f"  {date}: {len(files)} images")

            else:
                # Download single date
                date_str = args.date or datetime.now().strftime("%Y-%m-%d")
                downloaded_files = downloader.download_images_for_date(
                    date_str, args.stream
                )

                print(f"\nDownload completed!")
                print(f"Date: {date_str}")
                print(f"Stream: {args.stream}")
                print(f"Images downloaded: {len(downloaded_files)}")
                print(f"Output directory: {args.output}")

    if __name__ == "__main__":
        main()
