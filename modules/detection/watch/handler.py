"""
File System Event Handler for YOLO Watch Mode

This module provides the ImageFileHandler class that monitors file system events
for new image files and adds them to the processing queue.

Features:
- Configurable file extensions support
- Automatic queue management
- File write completion detection
- Comprehensive logging
"""

import logging
import time
from pathlib import Path
from typing import Set, Optional
from watchdog.events import FileSystemEventHandler

# Import processing queue
from ..processing import ImageProcessingQueue

logger = logging.getLogger(__name__)


class ImageFileHandler(FileSystemEventHandler):
    """
    File system event handler for new images.

    Monitors file system events and processes new image files by adding them
    to the processing queue. Supports configurable file extensions and includes
    logic to ensure files are fully written before processing.
    """

    # Default supported image file extensions
    DEFAULT_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp', '.gif'}

    def __init__(
        self,
        processing_queue: ImageProcessingQueue,
        supported_extensions: Optional[Set[str]] = None,
        file_write_delay: float = 0.5
    ):
        """
        Initialize the ImageFileHandler.

        Args:
            processing_queue: Queue for managing image processing
            supported_extensions: Set of supported file extensions (case-insensitive)
            file_write_delay: Seconds to wait after file creation to ensure complete write
        """
        super().__init__()
        self.processing_queue = processing_queue
        self.supported_extensions = supported_extensions or self.DEFAULT_EXTENSIONS
        self.file_write_delay = file_write_delay

        # Normalize extensions to lowercase for case-insensitive comparison
        self.supported_extensions = {ext.lower() for ext in self.supported_extensions}

        logger.info(f"Image file handler initialized with extensions: {self.supported_extensions}")
        logger.info(f"File write delay set to {self.file_write_delay}s")

    def is_supported_image(self, file_path: Path) -> bool:
        """
        Check if the file is a supported image type.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file has a supported extension, False otherwise
        """
        return file_path.suffix.lower() in self.supported_extensions

    def on_created(self, event):
        """
        Handle file creation events.

        Args:
            event: File system event containing information about the created file
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if it's a supported image file
        if self.is_supported_image(file_path):
            logger.info(f"New image detected: {file_path.name}")

            # Wait to ensure file is fully written
            if self.file_write_delay > 0:
                time.sleep(self.file_write_delay)

            # Verify file still exists and is readable
            if not file_path.exists():
                logger.warning(f"File {file_path.name} no longer exists after write delay")
                return

            try:
                # Check if file is readable
                with open(file_path, 'rb') as f:
                    f.read(1)  # Try to read first byte

                # Add to processing queue
                if self.processing_queue.add_image(file_path):
                    logger.info(f"Queued {file_path.name} for processing")
                else:
                    logger.debug(f"Image {file_path.name} already processed or queue full")

            except (IOError, OSError) as e:
                logger.error(f"Cannot read file {file_path.name}: {e}")
        else:
            logger.debug(f"Ignoring non-image file: {file_path.name}")

    def on_moved(self, event):
        """
        Handle file move events (treat as creation).

        Args:
            event: File system event containing information about the moved file
        """
        if not event.is_directory:
            # Treat moved files as newly created files
            event.src_path = event.dest_path
            self.on_created(event)

    def add_supported_extension(self, extension: str):
        """
        Add a new supported file extension.

        Args:
            extension: File extension to add (with or without leading dot)
        """
        if not extension.startswith('.'):
            extension = '.' + extension

        extension = extension.lower()
        self.supported_extensions.add(extension)
        logger.info(f"Added supported extension: {extension}")

    def remove_supported_extension(self, extension: str):
        """
        Remove a supported file extension.

        Args:
            extension: File extension to remove (with or without leading dot)
        """
        if not extension.startswith('.'):
            extension = '.' + extension

        extension = extension.lower()
        if extension in self.supported_extensions:
            self.supported_extensions.remove(extension)
            logger.info(f"Removed supported extension: {extension}")
        else:
            logger.warning(f"Extension {extension} was not in supported list")

    def get_supported_extensions(self) -> Set[str]:
        """
        Get the current set of supported file extensions.

        Returns:
            Set of supported file extensions
        """
        return self.supported_extensions.copy()

    def update_file_write_delay(self, delay: float):
        """
        Update the file write delay.

        Args:
            delay: New delay in seconds
        """
        self.file_write_delay = max(0.0, delay)
        logger.info(f"Updated file write delay to {self.file_write_delay}s")