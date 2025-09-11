"""
Image Downloader Module for Container Analytics

This module handles downloading camera images from Dray Dog terminal cameras.
It includes Selenium-based web scraping capabilities and scheduled downloading.
"""

from .selenium_client import DrayDogDownloader
from .scheduler import ImageDownloadScheduler

__all__ = ["DrayDogDownloader", "ImageDownloadScheduler"]

__version__ = "1.0.0"
