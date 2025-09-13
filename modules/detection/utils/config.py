"""
Configuration and settings for YOLO detection.

This module contains configuration classes and constants used throughout
the YOLO detection system, including thresholds, class mappings, and
model parameters.
"""

from typing import Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Define class mappings for container-relevant objects
CONTAINER_CLASSES = {
    2: 'car',           # Cars/vehicles
    3: 'motorcycle',    # Motorcycles
    5: 'bus',          # Buses
    7: 'truck',        # Trucks (most relevant for containers)
    8: 'boat',         # Boats (for port context)
}

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "model_path": "yolov12x.pt",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.7,
    "device": None,  # Auto-detect
    "verbose": True
}

# Default paths
DEFAULT_PATHS = {
    "models_dir": "data/models",
    "images_dir": "data/images",
    "cache_dir": "data/cache"
}

# Performance settings
PERFORMANCE_CONFIG = {
    "default_batch_size": 8,
    "max_queue_size": 100,
    "max_workers": 2,
    "cache_enabled": True,
    "max_cache_size": 1000
}


class DetectionConfig:
    """
    Configuration class for YOLO detection parameters.

    This class manages all configuration settings for the YOLO detector,
    including model parameters, thresholds, and paths.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_CONFIG["model_path"],
        confidence_threshold: float = DEFAULT_MODEL_CONFIG["confidence_threshold"],
        iou_threshold: float = DEFAULT_MODEL_CONFIG["iou_threshold"],
        device: Optional[str] = DEFAULT_MODEL_CONFIG["device"],
        verbose: bool = DEFAULT_MODEL_CONFIG["verbose"],
        models_dir: str = DEFAULT_PATHS["models_dir"],
        batch_size: int = PERFORMANCE_CONFIG["default_batch_size"],
        enable_caching: bool = PERFORMANCE_CONFIG["cache_enabled"]
    ):
        """
        Initialize detection configuration.

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            verbose: Whether to display verbose output
            models_dir: Directory containing model files
            batch_size: Default batch size for processing
            enable_caching: Whether to enable result caching
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.verbose = verbose
        self.models_dir = models_dir
        self.batch_size = batch_size
        self.enable_caching = enable_caching

        # Validate and normalize model path
        self._normalize_model_path()

    def _normalize_model_path(self):
        """Normalize the model path to use the models directory."""
        model_path_obj = Path(self.model_path)

        # If it's just a filename (no directory), prepend models directory
        if not model_path_obj.parent.name or model_path_obj.parent == Path('.'):
            self.model_path = str(Path(self.models_dir) / model_path_obj.name)
            # Create the directory if it doesn't exist
            Path(self.models_dir).mkdir(parents=True, exist_ok=True)

    def update_thresholds(
        self,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> None:
        """
        Update detection thresholds.

        Args:
            confidence_threshold: New confidence threshold
            iou_threshold: New IoU threshold
        """
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("Confidence threshold must be between 0.0 and 1.0")
            self.confidence_threshold = confidence_threshold
            logger.info(f"Updated confidence threshold to {confidence_threshold}")

        if iou_threshold is not None:
            if not 0.0 <= iou_threshold <= 1.0:
                raise ValueError("IoU threshold must be between 0.0 and 1.0")
            self.iou_threshold = iou_threshold
            logger.info(f"Updated IoU threshold to {iou_threshold}")

    def update_device(self, device: str) -> None:
        """
        Update the inference device.

        Args:
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        valid_devices = ['cpu', 'cuda', 'auto', None]
        if device not in valid_devices and not device.startswith('cuda:'):
            raise ValueError(f"Invalid device: {device}. Must be one of {valid_devices} or 'cuda:N'")

        self.device = device
        logger.info(f"Updated device to {device}")

    def get_model_config(self) -> Dict:
        """
        Get model configuration as dictionary.

        Returns:
            Dictionary containing model configuration
        """
        return {
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "verbose": self.verbose
        }

    def get_performance_config(self) -> Dict:
        """
        Get performance configuration as dictionary.

        Returns:
            Dictionary containing performance configuration
        """
        return {
            "batch_size": self.batch_size,
            "enable_caching": self.enable_caching,
            "models_dir": self.models_dir
        }

    def validate_config(self) -> bool:
        """
        Validate the current configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate thresholds
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"Invalid confidence threshold: {self.confidence_threshold}")

        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"Invalid IoU threshold: {self.iou_threshold}")

        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch size: {self.batch_size}")

        # Validate model path exists or can be created
        model_path = Path(self.model_path)
        if not model_path.exists() and not model_path.parent.exists():
            try:
                model_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create model directory: {e}")

        logger.info("Configuration validation passed")
        return True

    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "verbose": self.verbose,
            "models_dir": self.models_dir,
            "batch_size": self.batch_size,
            "enable_caching": self.enable_caching
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'DetectionConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            DetectionConfig instance
        """
        return cls(**config_dict)

    def copy(self) -> 'DetectionConfig':
        """
        Create a copy of the configuration.

        Returns:
            New DetectionConfig instance with same parameters
        """
        return DetectionConfig.from_dict(self.to_dict())

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"DetectionConfig(model={Path(self.model_path).name}, conf={self.confidence_threshold}, iou={self.iou_threshold}, device={self.device})"

    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return f"DetectionConfig({self.to_dict()})"


def get_default_config() -> DetectionConfig:
    """
    Get default detection configuration.

    Returns:
        DetectionConfig with default settings
    """
    return DetectionConfig()


def create_config_for_environment(environment: str = "development") -> DetectionConfig:
    """
    Create configuration optimized for specific environment.

    Args:
        environment: Environment type ('development', 'production', 'testing')

    Returns:
        DetectionConfig optimized for the environment
    """
    if environment == "development":
        return DetectionConfig(
            confidence_threshold=0.3,  # Lower threshold for development
            batch_size=4,              # Smaller batches for development
            verbose=True,
            enable_caching=True
        )
    elif environment == "production":
        return DetectionConfig(
            confidence_threshold=0.5,  # Standard threshold for production
            batch_size=8,              # Larger batches for efficiency
            verbose=False,             # Less verbose for production
            enable_caching=True
        )
    elif environment == "testing":
        return DetectionConfig(
            confidence_threshold=0.1,  # Very low threshold for testing
            batch_size=2,              # Small batches for testing
            verbose=True,
            enable_caching=False       # Disable caching for testing
        )
    else:
        raise ValueError(f"Unknown environment: {environment}")


def validate_container_classes(classes: Dict[int, str]) -> bool:
    """
    Validate container class mapping.

    Args:
        classes: Dictionary mapping class IDs to class names

    Returns:
        True if valid

    Raises:
        ValueError: If class mapping is invalid
    """
    if not isinstance(classes, dict):
        raise ValueError("Container classes must be a dictionary")

    for class_id, class_name in classes.items():
        if not isinstance(class_id, int):
            raise ValueError(f"Class ID must be integer, got {type(class_id)}")

        if not isinstance(class_name, str):
            raise ValueError(f"Class name must be string, got {type(class_name)}")

        if class_id < 0:
            raise ValueError(f"Class ID must be non-negative, got {class_id}")

    return True