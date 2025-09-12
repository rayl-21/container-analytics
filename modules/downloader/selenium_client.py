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
from selenium.webdriver.chrome.service import Service
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
    Simplified Dray Dog image downloader using Selenium for timestamp discovery
    and direct HTTP requests for downloading.
    """

    def __init__(
        self,
        download_dir: str = "data/images",
        headless: bool = True,
        max_retries: int = 3,
        retry_delay: int = 2,
        timeout: int = 30,
    ):
        """
        Initialize the Dray Dog downloader.

        Args:
            download_dir: Directory to save downloaded images
            headless: Run browser in headless mode
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            timeout: Timeout for web driver operations in seconds
        """
        self.download_dir = download_dir
        self.headless = headless
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
        # Driver will be initialized when needed
        self.driver = None

    def _setup_driver(self) -> webdriver.Chrome:
        """
        Set up Chrome WebDriver with appropriate options.
        
        Returns:
            Configured Chrome WebDriver instance
        """
        logger.info("Setting up Chrome WebDriver...")
        
        options = webdriver.ChromeOptions()
        
        # Basic options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--window-size=1920,1080")
        
        if self.headless:
            options.add_argument("--headless=new")
        
        # Enable images - we need them to load to get URLs
        prefs = {
            "download.default_directory": os.path.abspath(self.download_dir),
        }
        options.add_experimental_option("prefs", prefs)
        
        # Suppress logs
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        options.add_argument("--log-level=3")
        
        # Initialize the driver
        service = Service(log_path=os.devnull)
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(60)
        
        logger.info("Chrome WebDriver setup complete")
        return driver

    def navigate_to_date(self, date_str: str, stream_name: str = "in_gate") -> bool:
        """
        Navigate to a specific date using the date picker.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            stream_name: Camera stream name
            
        Returns:
            True if navigation successful, False otherwise
        """
        try:
            if not self.driver:
                self.driver = self._setup_driver()
            
            # Get current URL to check if we need initial navigation
            current_url = self.driver.current_url if self.driver.current_url != "data:" else None
            
            # Navigate to the camera page if not already there
            if not current_url or "terminal_cameras" not in current_url:
                url = f"https://app.draydog.com/terminal_cameras/apm?streamName={stream_name}"
                logger.info(f"Initial navigation to {url}")
                self.driver.get(url)
                time.sleep(3)  # Wait for initial page load
            
            # Wait for page elements to be present
            wait = WebDriverWait(self.driver, 30)
            
            # Find and click the date picker input
            logger.info(f"Looking for date picker to navigate to {date_str}")
            
            # Based on user feedback, the date picker has class "el-input__inner" and placeholder "Pick a day"
            date_picker_selectors = [
                "input.el-input__inner[placeholder='Pick a day']",  # Most specific
                "input[placeholder='Pick a day']",
                "input.el-input__inner",
                "input[role='combobox']",
                "input[aria-haspopup='dialog']"
            ]
            
            date_picker = None
            for selector in date_picker_selectors:
                try:
                    date_picker = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if date_picker:
                        logger.info(f"Found date picker with selector: {selector}")
                        break
                except NoSuchElementException:
                    continue
            
            if not date_picker:
                logger.error("Could not find date picker element")
                return False
            
            # Click on the date picker to open calendar
            logger.info("Clicking on date picker to open calendar...")
            date_picker.click()
            time.sleep(1)  # Wait for calendar to appear
            
            # Parse the target date
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            target_year = dt.year
            target_month = dt.month
            target_day = dt.day
            
            logger.info(f"Looking for date: {target_year}-{target_month:02d}-{target_day:02d}")
            
            # Navigate to correct month/year if needed
            # First, check current month/year in calendar header
            try:
                # Look for calendar header with format "2025 September"
                calendar_header = self.driver.find_element(By.CSS_SELECTOR, ".el-date-picker__header, [class*='calendar-header'], [class*='picker-header']")
                header_text = calendar_header.text if calendar_header else ""
                
                # If not found, try looking for any element containing year and month
                if not header_text:
                    possible_headers = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{target_year}')]")
                    for header in possible_headers:
                        if str(target_year) in header.text and any(month in header.text for month in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]):
                            header_text = header.text
                            break
                
                logger.info(f"Calendar header shows: {header_text}")
                
                # Navigate to correct month if needed
                current_year = None
                current_month_name = None
                
                # Parse header (format like "2025 September")
                import re
                match = re.search(r'(\d{4})\s+(\w+)', header_text)
                if match:
                    current_year = int(match.group(1))
                    current_month_name = match.group(2)
                    
                    # Convert month name to number
                    month_names = ["January", "February", "March", "April", "May", "June", 
                                  "July", "August", "September", "October", "November", "December"]
                    if current_month_name in month_names:
                        current_month = month_names.index(current_month_name) + 1
                        
                        # Navigate to target month if different
                        while current_year != target_year or current_month != target_month:
                            if current_year > target_year or (current_year == target_year and current_month > target_month):
                                # Go to previous month
                                prev_button = self.driver.find_element(By.CSS_SELECTOR, "[class*='arrow-left'], [class*='prev'], button[aria-label*='Previous']")
                                prev_button.click()
                            else:
                                # Go to next month
                                next_button = self.driver.find_element(By.CSS_SELECTOR, "[class*='arrow-right'], [class*='next'], button[aria-label*='Next']")
                                next_button.click()
                            
                            time.sleep(0.5)
                            
                            # Update current month/year
                            calendar_header = self.driver.find_element(By.CSS_SELECTOR, ".el-date-picker__header, [class*='calendar-header'], [class*='picker-header']")
                            header_text = calendar_header.text if calendar_header else ""
                            match = re.search(r'(\d{4})\s+(\w+)', header_text)
                            if match:
                                current_year = int(match.group(1))
                                current_month_name = match.group(2)
                                if current_month_name in month_names:
                                    current_month = month_names.index(current_month_name) + 1
                            
                            logger.debug(f"Navigated to: {current_year}-{current_month:02d}")
                
            except Exception as e:
                logger.warning(f"Could not navigate calendar months: {e}")
            
            # Now click on the specific day
            try:
                # Look for the day number in the calendar
                # Try multiple strategies to find the day
                day_clicked = False
                
                # Strategy 1: Look for td or div containing exact day number
                day_selectors = [
                    f"td:not(.prev-month):not(.next-month) span:contains('{target_day}')",  # For tables
                    f"td[class*='available'] span:contains('{target_day}')",
                    f"div[class*='calendar-day']:contains('{target_day}')",
                    f"button:contains('{target_day}')",
                    f"span:contains('{target_day}')"
                ]
                
                # Look for the specific day in the calendar
                # The calendar shows days as numbers in a table
                # We need to find the right cell and click it
                
                # Look for the correct day in the current month
                # IMPORTANT: Calendar may show days from prev/next month, we need the one for current month
                
                # First, get all day cells to understand the calendar structure
                all_day_cells = self.driver.find_elements(By.XPATH, "//table[contains(@class, 'el-date-table')]//td")
                logger.debug(f"Found {len(all_day_cells)} total calendar cells")
                
                # Look for the target day that's in the current month (not prev/next month)
                elements = self.driver.find_elements(By.XPATH, 
                    f"//table[contains(@class, 'el-date-table')]//td[not(contains(@class, 'prev-month')) and not(contains(@class, 'next-month'))]//span[text()='{target_day}']")
                
                if not elements:
                    # Try finding available cells with the day number
                    elements = self.driver.find_elements(By.XPATH, 
                        f"//td[contains(@class, 'available') and not(contains(@class, 'prev-month')) and not(contains(@class, 'next-month'))]//span[text()='{target_day}']")
                
                if not elements:
                    # More specific: look for current month cells
                    elements = self.driver.find_elements(By.XPATH, 
                        f"//td[contains(@class, 'current')]//span[text()='{target_day}']")
                
                logger.info(f"Found {len(elements)} potential day {target_day} elements in current month")
                
                # If we have multiple matches, prefer the one that's marked as "current" or "available"
                best_element = None
                for elem in elements:
                    try:
                        if elem.is_displayed() and elem.is_enabled():
                            parent_td = elem.find_element(By.XPATH, "./ancestor::td")
                            if parent_td:
                                td_class = parent_td.get_attribute("class") or ""
                                
                                # Skip if it's from previous/next month
                                if "prev-month" in td_class or "next-month" in td_class:
                                    logger.debug(f"Skipping day {target_day} from prev/next month")
                                    continue
                                
                                # Skip if disabled
                                if "disabled" in td_class:
                                    logger.debug(f"Skipping disabled day {target_day}")
                                    continue
                                
                                # Prefer "current" (today) or "available" cells
                                if "current" in td_class or "today" in td_class:
                                    logger.info(f"Found current/today cell for day {target_day}")
                                    best_element = elem
                                    break
                                elif "available" in td_class:
                                    logger.info(f"Found available cell for day {target_day}")
                                    best_element = elem
                                    # Continue looking in case there's a "current" one
                                elif not best_element:
                                    # Use this as fallback if no better option
                                    best_element = elem
                                    
                    except Exception as e:
                        logger.debug(f"Error checking element: {e}")
                        continue
                
                if best_element:
                    try:
                        parent_td = best_element.find_element(By.XPATH, "./ancestor::td")
                        td_class = parent_td.get_attribute("class") if parent_td else ""
                        logger.info(f"Clicking on day {target_day} (td class: {td_class})")
                        best_element.click()
                        day_clicked = True
                    except Exception as e:
                        logger.error(f"Error clicking day {target_day}: {e}")
                
                if not day_clicked:
                    # Fallback: Look for any clickable element with the day number
                    all_elements = self.driver.find_elements(By.XPATH, f"//*[text()='{target_day}']")
                    for elem in all_elements:
                        try:
                            parent = elem.find_element(By.XPATH, "..")
                            parent_class = parent.get_attribute("class") or ""
                            elem_class = elem.get_attribute("class") or ""
                            
                            # Check if this looks like a calendar day
                            if any(cal_word in parent_class + elem_class for cal_word in ["date", "day", "cell", "calendar"]):
                                if "disabled" not in parent_class and "prev" not in parent_class and "next" not in parent_class:
                                    logger.info(f"Clicking on day {target_day} (fallback)")
                                    elem.click()
                                    day_clicked = True
                                    break
                        except:
                            continue
                
                if not day_clicked:
                    logger.error(f"Could not find clickable day {target_day} in calendar")
                    return False
                    
            except Exception as e:
                logger.error(f"Error clicking on day: {e}")
                return False
            
            # Wait for calendar to close and page to update with new images
            time.sleep(3)  # Give time for History section to update
            
            # Simplified verification - just check that we have images in the History section
            # We trust that after selecting the date, the History section shows the correct date
            try:
                logger.info(f"Verifying History section has loaded after selecting {date_str}...")
                
                # Scroll down to ensure History section is visible
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                time.sleep(1)
                
                # Look for images in the History section
                history_images = []
                
                # Find elements that might indicate the History section
                try:
                    # Look for History text/heading
                    history_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'History') or contains(text(), 'history')]")
                    if history_elements:
                        # Find images that come after the History element
                        for hist_elem in history_elements:
                            # Get all images after this element
                            following_images = self.driver.find_elements(By.XPATH, 
                                f"./following::img[contains(@src, 'cdn.draydog.com')]")
                            history_images.extend(following_images)
                            if following_images:
                                break
                except:
                    pass
                
                # If no History heading found, use all images except the first few (which are likely In Gate section)
                if not history_images:
                    all_images = self.driver.find_elements(By.CSS_SELECTOR, "img[src*='cdn.draydog.com']")
                    # Skip first few images (likely from In Gate section) and check the rest
                    if len(all_images) > 5:
                        history_images = all_images[5:]  # Skip first 5 images
                    elif len(all_images) > 2:
                        history_images = all_images[2:]  # Skip first 2 if fewer images
                
                if history_images:
                    # Just verify we have images - trust they are for the selected date
                    logger.info(f"✓ Successfully navigated to {date_str} - Found {len(history_images)} images in History section")
                    # Log first image URL for reference
                    if history_images[0]:
                        first_img_src = history_images[0].get_attribute("src")
                        if first_img_src:
                            logger.debug(f"First History image: {first_img_src[:100]}...")
                    return True
                else:
                    logger.error(f"No images found in History section after navigating to {date_str}")
                    return False
                        
            except Exception as e:
                logger.error(f"Error verifying History section: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error navigating to date: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_timestamps_for_date(
        self, date_str: str, stream_name: str = "in_gate"
    ) -> List[str]:
        """
        Get actual timestamps from Dray Dog for a specific date.
        Trusts that History section shows images for the selected date.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            stream_name: Camera stream name
            
        Returns:
            List of timestamp strings in format YYYY-MM-DDTHH:MM:SS
        """
        timestamps = []
        
        try:
            # Navigate to the specific date using date picker
            if not self.navigate_to_date(date_str, stream_name):
                logger.error(f"Failed to navigate to {date_str}")
                return []
            
            # Wait a bit for all images to load
            time.sleep(3)
            
            # Now extract actual image URLs from the History section
            # IMPORTANT: Skip In Gate section (live images) and focus on History section
            
            # Scroll to make sure History section is loaded
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            
            # Find all image elements
            all_images = self.driver.find_elements(By.CSS_SELECTOR, "img[src*='cdn.draydog.com']")
            
            logger.info(f"Found {len(all_images)} total image elements on page")
            
            # Try to identify History section images
            history_images = []
            
            # Strategy 1: Look for images after History heading
            try:
                history_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'History') or contains(text(), 'history')]")
                if history_elements:
                    # Use JavaScript to find images that come after the History element
                    for hist_elem in history_elements:
                        following_images = self.driver.execute_script("""
                            var histElem = arguments[0];
                            var images = [];
                            var allImgs = document.querySelectorAll('img[src*="cdn.draydog.com"]');
                            var histPosition = histElem.getBoundingClientRect().top;
                            for (var i = 0; i < allImgs.length; i++) {
                                if (allImgs[i].getBoundingClientRect().top > histPosition) {
                                    images.push(allImgs[i]);
                                }
                            }
                            return images;
                        """, hist_elem)
                        if following_images:
                            history_images = following_images
                            logger.info(f"Found {len(history_images)} images in History section")
                            break
            except Exception as e:
                logger.debug(f"Could not find History section by heading: {e}")
            
            # Strategy 2: If no History heading found, skip first few images (In Gate section)
            if not history_images and len(all_images) > 5:
                # Typically In Gate shows 1-4 images, so skip first 5 to be safe
                history_images = all_images[5:]
                logger.info(f"Using images after position 5 as History section ({len(history_images)} images)")
            elif not history_images:
                # Fallback: use all images except the very first ones
                history_images = all_images[2:] if len(all_images) > 2 else all_images
                logger.info(f"Using all images except first 2 as History section ({len(history_images)} images)")
            
            # Extract timestamps from History section images
            # We trust these are for the selected date
            import re
            for img in history_images:
                try:
                    img_src = img.get_attribute("src")
                    
                    if img_src and "cdn.draydog.com" in img_src:
                        # Log first few URLs for debugging
                        if len(timestamps) < 3:
                            logger.debug(f"History image URL: {img_src}")
                        
                        # Extract timestamp from URL
                        # Pattern: YYYY-MM-DDTHH:MM:SS
                        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', img_src)
                        if match:
                            timestamp = match.group(1)
                            # Add all timestamps from History section (trust they're for selected date)
                            if timestamp not in timestamps:
                                timestamps.append(timestamp)
                                logger.debug(f"Found timestamp: {timestamp}")
                
                except Exception as e:
                    logger.debug(f"Error processing image: {e}")
                    continue
            
            if timestamps:
                logger.info(f"Found {len(timestamps)} unique timestamps for {date_str} from History section")
            else:
                logger.warning(f"No timestamps found in History section for {date_str}")
                    
        except Exception as e:
            logger.error(f"Error getting timestamps: {e}")
            import traceback
            traceback.print_exc()
        
        return sorted(list(set(timestamps)))


    def get_full_size_url(self, thumbnail_url: str) -> str:
        """
        Convert thumbnail URL to full-size image URL.
        
        Args:
            thumbnail_url: Thumbnail image URL
            
        Returns:
            Full-size image URL
        """
        # Remove -thumbnail suffix if present
        full_url = thumbnail_url.replace("-thumbnail.jpeg", ".jpeg")
        full_url = full_url.replace("-thumbnail.jpg", ".jpg")
        return full_url

    def download_image(self, url: str, filepath: str) -> bool:
        """
        Download a single image from URL.
        
        Args:
            url: Image URL
            filepath: Local file path to save image
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            response = requests.get(
                url,
                timeout=30,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
            
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "image" in content_type or len(response.content) > 1000:
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    return True
            return False
            
        except Exception as e:
            logger.debug(f"Download failed for {url}: {e}")
            return False

    def download_images_for_date(
        self,
        date_str: str,
        stream_name: str = "in_gate",
        max_images: Optional[int] = None
    ) -> List[str]:
        """
        Download images for a specific date.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            stream_name: Camera stream name
            max_images: Maximum number of images to download
            
        Returns:
            List of downloaded file paths
        """
        downloaded_files = []
        
        # Create date-specific directory
        date_dir = os.path.join(self.download_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        # Get actual timestamps from website
        logger.info(f"Getting timestamps for {date_str}...")
        timestamps = self.get_timestamps_for_date(date_str, stream_name)
        
        if not timestamps:
            logger.warning(f"No timestamps found for {date_str}")
            return []
        
        if max_images:
            timestamps = timestamps[:max_images]
        
        logger.info(f"Downloading {len(timestamps)} images for {date_str}")
        
        for idx, timestamp in enumerate(timestamps, 1):
            # Parse timestamp
            dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
            date_folder = dt.strftime("%Y-%m-%d")
            hour_folder = str(dt.hour)
            
            # Construct full-size URL (prioritize full-size over thumbnail)
            full_url = f"https://cdn.draydog.com/apm/{date_folder}/{hour_folder}/{timestamp}-{stream_name}.jpeg"
            thumbnail_url = f"https://cdn.draydog.com/apm/{date_folder}/{hour_folder}/{timestamp}-{stream_name}-thumbnail.jpeg"
            
            # Try full-size first, then thumbnail as fallback
            urls = [full_url, thumbnail_url]
            
            # Create local filename
            filename = f"{timestamp.replace(':', '').replace('-', '')}_{stream_name}.jpg"
            filepath = os.path.join(date_dir, filename)
            
            # Skip if already exists
            if os.path.exists(filepath):
                logger.debug(f"Already exists: {filename}")
                downloaded_files.append(filepath)
                continue
            
            # Try downloading from each URL
            success = False
            for url_idx, url in enumerate(urls):
                if self.download_image(url, filepath):
                    url_type = "full-size" if url_idx == 0 else "thumbnail"
                    logger.info(f"✓ Downloaded [{idx}/{len(timestamps)}] ({url_type}): {filename}")
                    downloaded_files.append(filepath)
                    success = True
                    break
            
            if not success:
                logger.warning(f"✗ Failed [{idx}/{len(timestamps)}]: {timestamp}")
        
        # Save metadata
        if downloaded_files:
            metadata_path = os.path.join(date_dir, "download_metadata.json")
            metadata = {
                "date": date_str,
                "stream": stream_name,
                "total_images": len(downloaded_files),
                "download_timestamp": datetime.now().isoformat(),
                "files": [os.path.basename(f) for f in downloaded_files]
            }
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Downloaded {len(downloaded_files)} images to {date_dir}")
        return downloaded_files

    def download_date_range(
        self,
        start_date: str,
        end_date: str,
        stream_name: str = "in_gate"
    ) -> Dict[str, List[str]]:
        """
        Download images for a date range.
        
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
            
            logger.info(f"Downloading images from {start_date} to {end_date}")
            
            # Initialize driver once for the entire date range
            if not self.driver:
                self.driver = self._setup_driver()
            
            current_dt = start_dt
            while current_dt <= end_dt:
                date_str = current_dt.strftime("%Y-%m-%d")
                logger.info(f"\n--- Processing {date_str} ---")
                
                # Use the existing download method which will now use date picker navigation
                downloaded = self.download_images_for_date(date_str, stream_name)
                results[date_str] = downloaded
                
                current_dt += timedelta(days=1)
                
                # Small delay between dates to avoid rate limiting
                if current_dt <= end_dt:
                    time.sleep(2)
            
            # Summary
            total_images = sum(len(files) for files in results.values())
            logger.info(f"\n=== Download Complete ===")
            logger.info(f"Processed {len(results)} dates")
            logger.info(f"Total images downloaded: {total_images}")
            
        except Exception as e:
            logger.error(f"Date range download failed: {e}")
            import traceback
            traceback.print_exc()
        
        return results

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
                logger.info("WebDriver closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Example usage and CLI interface
def main():
    """
    Main function to test the Dray Dog downloader.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Download images from Dray Dog cameras")
    parser.add_argument(
        "--start-date",
        default="2025-09-01",
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date",
        default="2025-09-07",
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--stream",
        default="in_gate",
        help="Camera stream name (default: in_gate)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode"
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run browser with GUI"
    )
    
    args = parser.parse_args()
    
    # Create downloader and download images
    with DrayDogDownloader(headless=args.headless) as downloader:
        results = downloader.download_date_range(
            start_date=args.start_date,
            end_date=args.end_date,
            stream_name=args.stream
        )
        
        # Print summary
        print("\n=== Download Summary ===")
        for date, files in results.items():
            print(f"{date}: {len(files)} images")
        
        total = sum(len(files) for files in results.values())
        print(f"\nTotal: {total} images downloaded")


if __name__ == "__main__":
    main()
