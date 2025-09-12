"""
Selenium-based image downloader for Dray Dog terminal cameras.

This module provides automated downloading of camera images from the Dray Dog
camera history page using Selenium WebDriver.
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)
from webdriver_manager.chrome import ChromeDriverManager
from loguru import logger

# Database imports
from ..database import queries
from ..database.models import session_scope


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

            # Essential Chrome options for Docker container
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-web-security")
            options.add_argument("--disable-features=VizDisplayCompositor")
            
            # Critical for Docker: disable single process mode (it causes crashes)
            # options.add_argument("--single-process")  # REMOVED - causes crashes
            
            # Additional stability options for containerized environments
            options.add_argument("--disable-setuid-sandbox")
            options.add_argument("--disable-dev-tools")
            options.add_argument("--no-zygote")
            options.add_argument("--use-gl=swiftshader")
            options.add_argument("--disable-software-rasterizer")
            
            # Set window size for consistent rendering
            options.add_argument("--window-size=1920,1080")
            
            # Disable automation detection
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            
            # Performance optimizations
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-logging")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-default-apps")
            options.add_argument("--disable-background-timer-throttling")
            options.add_argument("--disable-backgrounding-occluded-windows")
            options.add_argument("--disable-breakpad")
            options.add_argument("--disable-component-extensions-with-background-pages")
            options.add_argument("--disable-features=TranslateUI")
            options.add_argument("--disable-ipc-flooding-protection")
            options.add_argument("--disable-renderer-backgrounding")
            options.add_argument("--force-color-profile=srgb")
            options.add_argument("--metrics-recording-only")
            options.add_argument("--mute-audio")
            
            # User agent to avoid detection
            options.add_argument(
                "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            # Memory and resource management
            options.add_argument("--memory-pressure-off")
            options.add_argument("--max_old_space_size=4096")
            
            # Disable images for faster loading (we'll download them separately)
            prefs = {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_settings.popups": 0,
                "download.default_directory": self.download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": False
            }
            options.add_experimental_option("prefs", prefs)
            
            # Check if running in Docker container
            import os
            is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER", False)
            
            if is_docker:
                logger.info("Running in Docker container, using system Chrome")
                # Use system Chrome/Chromium in Docker
                chrome_paths = [
                    "/usr/bin/google-chrome-stable",
                    "/usr/bin/google-chrome",
                    "/usr/bin/chromium-browser",
                    "/usr/bin/chromium"
                ]
                
                chrome_binary = None
                for path in chrome_paths:
                    if os.path.exists(path):
                        chrome_binary = path
                        logger.info(f"Found Chrome at: {path}")
                        break
                
                if chrome_binary:
                    options.binary_location = chrome_binary
                
                # Try to find chromedriver
                chromedriver_paths = [
                    "/usr/local/bin/chromedriver",
                    "/usr/bin/chromedriver",
                    "/usr/lib/chromium/chromedriver",
                ]
                chromedriver_path = None
                for path in chromedriver_paths:
                    if os.path.exists(path):
                        chromedriver_path = path
                        logger.info(f"Found ChromeDriver at: {path}")
                        break
                
                if chromedriver_path:
                    from selenium.webdriver.chrome.service import Service
                    service = Service(chromedriver_path)
                    driver = webdriver.Chrome(service=service, options=options)
                else:
                    # Fallback to webdriver without explicit path
                    logger.warning("ChromeDriver not found, trying default")
                    driver = webdriver.Chrome(options=options)
            else:
                # Use webdriver-manager for non-Docker environments
                logger.info("Running in non-Docker environment, using webdriver-manager")
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)

            # Set additional capabilities
            driver.implicitly_wait(10)
            driver.set_page_load_timeout(60)
            
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
            stream_name: Camera stream name (e.g., 'in_gate')
            date_str: Specific date in YYYY-MM-DD format, None for today

        Returns:
            True if navigation successful, False otherwise
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if not self.driver:
                    self.driver = self._setup_driver()

                # Construct URL - directly access without login
                base_url = f"https://app.draydog.com/terminal_cameras/apm?streamName={stream_name}#camera-history"

                logger.info(f"Navigating to camera history for stream: {stream_name} (attempt {attempt + 1}/{max_attempts})")
                
                # Set page load timeout
                self.driver.set_page_load_timeout(60)
                
                # Navigate to URL
                self.driver.get(base_url)
                
                # Add explicit wait for page to fully load
                time.sleep(3)
                
                # Wait for page to load with multiple fallback selectors
                wait = WebDriverWait(self.driver, self.timeout)
                
                # Try multiple possible selectors for the camera history page
                selectors_to_try = [
                    (By.CLASS_NAME, "el-image"),
                    (By.CSS_SELECTOR, "img[src*='cdn.draydog.com']"),
                    (By.CSS_SELECTOR, "[class*='camera']"),
                    (By.CSS_SELECTOR, "[class*='history']"),
                    (By.TAG_NAME, "img"),
                ]
                
                element_found = False
                for selector_type, selector_value in selectors_to_try:
                    try:
                        wait.until(EC.presence_of_element_located((selector_type, selector_value)))
                        element_found = True
                        logger.info(f"Found element with selector: {selector_type}={selector_value}")
                        break
                    except TimeoutException:
                        continue
                
                if not element_found:
                    # Try to check if page loaded at all
                    page_source = self.driver.page_source
                    if "camera" in page_source.lower() or "draydog" in page_source.lower():
                        logger.warning("Page loaded but expected elements not found, proceeding anyway")
                        element_found = True
                    else:
                        raise TimeoutException("Page did not load expected content")

                # If specific date requested, navigate to it
                if date_str:
                    self._navigate_to_date(date_str)

                logger.info("Successfully navigated to camera history page")
                return True

            except TimeoutException as e:
                logger.warning(f"Timeout navigating to camera history (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    logger.info("Retrying navigation...")
                    time.sleep(2)
                    continue
                else:
                    logger.error("All navigation attempts failed")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to navigate to camera history: {e}")
                if attempt < max_attempts - 1:
                    logger.info("Retrying navigation...")
                    time.sleep(2)
                    continue
                else:
                    return False
        
        return False

    def _navigate_to_date(self, date_str: str) -> bool:
        """
        Navigate to a specific date in the camera history.
        
        The Dray Dog UI uses a client-side date picker that:
        1. Doesn't change the URL when selecting a date
        2. Loads images dynamically after date selection
        3. May show thumbnails from nearby dates at the top

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            True if date navigation successful, False otherwise
        """
        try:
            wait = WebDriverWait(self.driver, self.timeout)
            
            # Find and click the date picker (it's a combobox element)
            logger.info(f"Opening date picker to navigate to {date_str}")
            date_picker = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".el-date-editor"))
            )
            date_picker.click()
            
            # Wait for calendar to open
            time.sleep(1)
            
            # Find the input field within the date picker
            date_input = self.driver.find_element(
                By.CSS_SELECTOR, ".el-date-editor input"
            )
            
            # Clear the input field completely using keyboard shortcuts
            date_input.click()
            # Use Command+A on Mac, Ctrl+A on other platforms
            if sys.platform == "darwin":
                date_input.send_keys(Keys.COMMAND + "a")
            else:
                date_input.send_keys(Keys.CONTROL + "a")
            date_input.send_keys(Keys.DELETE)
            
            # Enter the new date
            date_input.send_keys(date_str)
            date_input.send_keys(Keys.ENTER)
            
            logger.info(f"Date entered, waiting for images to load for {date_str}")
            
            # Wait for the page to update with new images
            # The page loads images dynamically, so we need to wait
            time.sleep(3)
            
            # Verify that images from the selected date are loaded
            # Note: The page may show thumbnails from other dates, so we check for majority
            verification_attempts = 3
            for attempt in range(verification_attempts):
                # Get all image URLs
                images = self.driver.find_elements(By.CSS_SELECTOR, "img[src*='cdn.draydog.com']")
                
                if images:
                    # Count how many images are from the target date
                    target_date_count = 0
                    total_checked = min(20, len(images))  # Check first 20 images
                    
                    for img in images[:total_checked]:
                        src = img.get_attribute("src")
                        if src and date_str in src:
                            target_date_count += 1
                    
                    # If at least 30% of checked images are from target date, consider it successful
                    if target_date_count >= total_checked * 0.3:
                        logger.info(f"Successfully navigated to {date_str} ({target_date_count}/{total_checked} images verified)")
                        return True
                    
                    logger.debug(f"Attempt {attempt + 1}: Found {target_date_count}/{total_checked} images from {date_str}")
                
                if attempt < verification_attempts - 1:
                    # Wait a bit more for images to load
                    time.sleep(2)
            
            # If we couldn't verify images, check if the date picker at least shows the correct date
            current_value = date_input.get_attribute("value")
            if date_str in current_value:
                logger.warning(f"Date picker shows {current_value} but couldn't verify images are from {date_str}")
                # Give it one more chance to load
                time.sleep(2)
                return True
            
            logger.error(f"Failed to navigate to {date_str}")
            return False

        except Exception as e:
            logger.error(f"Error navigating to date {date_str}: {e}")
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

            # Wait for page content with multiple fallback options
            wait = WebDriverWait(self.driver, self.timeout)
            
            # Try different selectors for image elements
            image_selectors = [
                ".el-image__inner",
                "img[src*='cdn.draydog.com']",
                "img[src*='draydog']",
                ".el-image img",
                "img"
            ]
            
            images_found = False
            for selector in image_selectors:
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    images_found = True
                    logger.info(f"Found images with selector: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if not images_found:
                logger.warning("No image elements found on page")
                # Try to extract from page source as fallback
                page_source = self.driver.page_source
                import re
                pattern = r'https?://[^"\s]*cdn\.draydog\.com[^"\s]*\.(?:jpg|jpeg|png)'
                urls = re.findall(pattern, page_source)
                if urls:
                    logger.info(f"Found {len(urls)} image URLs in page source")
                    image_data = []
                    for url in urls:
                        if '-thumbnail' not in url and '_thumb' not in url:
                            filename = url.split('/')[-1]
                            image_data.append({
                                'url': url,
                                'filename': filename,
                                'timestamp': filename.split('-')[0] if '-' in filename else 'unknown',
                                'captureTime': None,
                                'index': len(image_data),
                                'streamName': 'unknown'
                            })
                    return image_data
                return []

            # Extract image data using JavaScript with improved error handling
            image_data = self.driver.execute_script("""
                try {
                    const images = [];
                    
                    // Try multiple ways to find images
                    let imageElements = document.querySelectorAll('.el-image__inner');
                    if (imageElements.length === 0) {
                        imageElements = document.querySelectorAll('img[src*="cdn.draydog.com"]');
                    }
                    if (imageElements.length === 0) {
                        imageElements = document.querySelectorAll('img[src*="draydog"]');
                    }
                    if (imageElements.length === 0) {
                        imageElements = document.querySelectorAll('img');
                    }
                    
                    console.log('Found ' + imageElements.length + ' image elements');
                    
                    imageElements.forEach((img, index) => {
                        if (img.src && (img.src.includes('cdn.draydog.com') || img.src.includes('draydog'))) {
                            // Skip thumbnail images - only get full-size images
                            let fullSizeUrl = img.src;
                            let isThumbnail = false;
                            
                            if (img.src.includes('-thumbnail') || img.src.includes('_thumb')) {
                                isThumbnail = true;
                                // Convert thumbnail to full-size URL
                                fullSizeUrl = img.src.replace('-thumbnail', '').replace('_thumb', '');
                            }
                            
                            const urlParts = fullSizeUrl.split('/');
                            const filename = urlParts[urlParts.length - 1];
                            
                            // Clean up the filename
                            const cleanFilename = filename.replace('-thumbnail', '').replace('_thumb', '');
                            
                            const timestampPart = cleanFilename.split('-')[0] || 'unknown';
                            
                            // Try to get stream name from URL or page
                            let streamName = 'unknown';
                            try {
                                const urlParams = new URLSearchParams(window.location.search);
                                streamName = urlParams.get('streamName') || 'unknown';
                            } catch(e) {
                                console.log('Could not extract stream name');
                            }
                            
                            images.push({
                                url: fullSizeUrl,
                                filename: cleanFilename,
                                timestamp: timestampPart,
                                captureTime: null,
                                index: index,
                                streamName: streamName,
                                wasThumbnail: isThumbnail
                            });
                        }
                    });
                    
                    // Deduplicate by URL
                    const uniqueImages = [];
                    const seenUrls = new Set();
                    for (const img of images) {
                        if (!seenUrls.has(img.url)) {
                            seenUrls.add(img.url);
                            uniqueImages.push(img);
                        }
                    }
                    
                    console.log('Extracted ' + uniqueImages.length + ' unique images');
                    return uniqueImages;
                } catch(error) {
                    console.error('Error extracting images:', error);
                    return [];
                }
            """)

            if not image_data:
                logger.warning("JavaScript extraction returned no images")
                return []

            logger.info(f"Extracted {len(image_data)} full-size image URLs")
            return image_data

        except Exception as e:
            logger.error(f"Failed to extract image URLs: {e}")
            # Try one more fallback - extract from page source
            try:
                page_source = self.driver.page_source
                import re
                pattern = r'https?://[^"\s]*cdn\.draydog\.com[^"\s]*\.(?:jpg|jpeg|png)'
                urls = re.findall(pattern, page_source)
                if urls:
                    logger.info(f"Fallback: Found {len(urls)} image URLs in page source")
                    image_data = []
                    for url in urls:
                        if '-thumbnail' not in url and '_thumb' not in url:
                            filename = url.split('/')[-1]
                            image_data.append({
                                'url': url,
                                'filename': filename,
                                'timestamp': filename.split('-')[0] if '-' in filename else 'unknown',
                                'captureTime': None,
                                'index': len(image_data),
                                'streamName': 'unknown'
                            })
                    return image_data
            except:
                pass
            
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

            # Extract date from URL or filename to ensure proper folder structure
            # URL pattern: https://cdn.draydog.com/apm/2025-09-01/23/2025-09-01T23:00:00-in_gate.jpeg
            import re
            date_match = re.search(r'/(\d{4}-\d{2}-\d{2})/', url)
            if date_match:
                date_part = date_match.group(1)
            elif "T" in filename:
                date_part = filename.split("T")[0]
            else:
                # Try to extract from the timestamp in filename
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
                if date_match:
                    date_part = date_match.group(1)
                else:
                    date_part = datetime.now().strftime("%Y-%m-%d")
            
            # Create subdirectory based on date
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

            # Save to database
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)

                # Extract timestamp from filename or URL
                timestamp = self._extract_timestamp_from_url(url)

                # Save to database
                try:
                    image_id = queries.insert_image(
                        filepath=image_path,
                        camera_id=image_info.get("streamName", "unknown"),
                        timestamp=timestamp,
                        file_size=file_size,
                    )
                    logger.info(f"Saved to database with ID: {image_id}")
                except Exception as e:
                    logger.error(f"Failed to save to database: {e}")
                    # Don't fail the download if DB insert fails

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

    def _extract_timestamp_from_url(self, url: str) -> datetime:
        """
        Extract timestamp from DrayDog URL format.

        DrayDog URLs follow the pattern:
        https://cdn.draydog.com/apm/[date]/[hour]/[timestamp]-[stream].jpeg

        Args:
            url: The URL to extract timestamp from

        Returns:
            datetime: Parsed timestamp, or current time if parsing fails
        """
        import re

        # URL format: https://cdn.draydog.com/apm/[date]/[hour]/[timestamp]-[stream].jpeg
        # Example: https://cdn.draydog.com/apm/2025-01-15/10/2025-01-15T10:30:00-in_gate.jpeg
        pattern = (
            r"/(\d{4}-\d{2}-\d{2})/(\d{1,2})/(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})-"
        )
        match = re.search(pattern, url)

        if match:
            timestamp_str = match.group(3)
            try:
                return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                logger.warning(f"Failed to parse timestamp from URL: {url}")

        # Also try to extract from filename if available
        filename_pattern = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
        filename_match = re.search(filename_pattern, url)
        if filename_match:
            try:
                return datetime.strptime(filename_match.group(1), "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                pass

        logger.warning(
            f"Could not extract timestamp from URL: {url}, using current time"
        )
        return datetime.utcnow()

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

    def get_available_timestamps(
        self, date_str: str = "2025-09-06", stream_name: str = "in_gate"
    ) -> List[str]:
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
                date_text = target_date.strftime(
                    f"%a, %b {day}"
                )  # Format like "Sat, Sep 6"

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
                logger.warning(
                    f"No timestamps found for {date_str}, using default intervals"
                )
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
                seconds = second_offsets[
                    (hour * 6 + minute // 10) % len(second_offsets)
                ]
                timestamp = target_date.replace(
                    hour=hour, minute=minute, second=seconds
                )
                timestamps.append(timestamp.strftime("%Y-%m-%dT%H:%M:%S"))

        return timestamps

    def download_images_direct(
        self,
        date_str: str = "2025-09-06",
        stream_name: str = "in_gate",
        max_images: Optional[int] = None,
        interval_minutes: int = 10,
        use_actual_timestamps: bool = False,
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
        downloaded_file_info = []  # For database batch insertion

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
                    # Still add to database info for existing files in case they're not in DB
                    parsed_timestamp = datetime.strptime(
                        timestamp_str, "%Y-%m-%dT%H:%M:%S"
                    )
                    downloaded_file_info.append(
                        {
                            "filepath": filepath,
                            "camera_id": stream_name,
                            "timestamp": parsed_timestamp,
                            "file_size": os.path.getsize(filepath),
                        }
                    )
                    continue

                # Try to download the image
                success = False

                # Try full-size image first
                try:
                    logger.info(f"Downloading [{idx+1}/{len(timestamps)}]: {full_url}")
                    response = requests.get(
                        full_url,
                        timeout=30,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                        },
                    )

                    if response.status_code == 200:
                        # Check if we got actual image data
                        content_type = response.headers.get("content-type", "")
                        if "image" in content_type or len(response.content) > 1000:
                            with open(filepath, "wb") as f:
                                f.write(response.content)
                            success = True
                            logger.info(f"✓ Downloaded full image: {filename}")

                except Exception as e:
                    logger.debug(f"Full image failed: {e}")

                # Fall back to thumbnail if full image failed
                if not success:
                    try:
                        logger.info(f"Trying thumbnail: {thumbnail_url}")
                        response = requests.get(
                            thumbnail_url,
                            timeout=30,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                            },
                        )

                        if response.status_code == 200:
                            content_type = response.headers.get("content-type", "")
                            if "image" in content_type or len(response.content) > 1000:
                                with open(filepath, "wb") as f:
                                    f.write(response.content)
                                success = True
                                logger.info(f"✓ Downloaded thumbnail: {filename}")

                    except Exception as e:
                        logger.debug(f"Thumbnail also failed: {e}")

                if success:
                    downloaded_files.append(filepath)
                    # Collect info for database insertion
                    parsed_timestamp = datetime.strptime(
                        timestamp_str, "%Y-%m-%dT%H:%M:%S"
                    )
                    downloaded_file_info.append(
                        {
                            "filepath": filepath,
                            "camera_id": stream_name,
                            "timestamp": parsed_timestamp,
                            "file_size": os.path.getsize(filepath),
                        }
                    )
                else:
                    logger.warning(f"✗ Failed to download image for {timestamp_str}")

            # Batch insert to database
            if downloaded_file_info:
                try:
                    with session_scope() as session:
                        for file_info in downloaded_file_info:
                            image_id = queries.insert_image(**file_info)
                            logger.debug(f"Inserted image {image_id}")
                    logger.info(
                        f"Batch inserted {len(downloaded_file_info)} images to database"
                    )
                except Exception as e:
                    logger.error(f"Batch database insert failed: {e}")
                    # Don't fail the download if DB insert fails

            # Save metadata
            metadata_path = os.path.join(date_dir, "download_metadata.json")
            metadata = {
                "date": date_str,
                "stream": stream_name,
                "total_images": len(downloaded_files),
                "download_timestamp": datetime.now().isoformat(),
                "files": downloaded_files,
                "method": "direct_url_construction",
            }

            with open(metadata_path, "w") as f:
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
        
        This method now uses direct URL construction for better reliability
        and proper folder organization.

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
            
            logger.info(f"Downloading images from {start_date} to {end_date} for stream: {stream_name}")
            
            # Process each date in the range
            current_dt = start_dt
            while current_dt <= end_dt:
                date_str = current_dt.strftime("%Y-%m-%d")
                logger.info(f"Processing date: {date_str}")
                
                # Use direct download method which creates proper folder structure
                downloaded_files = self.download_images_direct(
                    date_str=date_str,
                    stream_name=stream_name,
                    use_actual_timestamps=False  # Use predictable 10-minute intervals
                )
                
                results[date_str] = downloaded_files
                logger.info(f"Downloaded {len(downloaded_files)} images for {date_str}")
                
                # Move to next day
                current_dt += timedelta(days=1)
                
                # Small delay between dates to avoid overwhelming the server
                if current_dt <= end_dt:
                    time.sleep(1)

            logger.info(f"Date range download completed. Processed {len(results)} dates.")

        except Exception as e:
            logger.error(f"Date range download failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup will be called by context manager or explicitly
            pass

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
            description="Download Dray Dog camera images (no authentication required)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python selenium_client.py
  python selenium_client.py --date 2025-09-07
  python selenium_client.py --stream in_gate
  python selenium_client.py --date-range 2025-09-01 2025-09-07
            """,
        )

        # Make username and password optional since Dray Dog doesn't require authentication
        parser.add_argument("--username", required=False, help="Dray Dog username (not required)")
        parser.add_argument("--password", required=False, help="Dray Dog password (not required)")
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
