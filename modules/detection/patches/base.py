"""
Base classes and registry for model patches.

This module provides the foundation for implementing model patches in a modular
and extensible way. It includes an abstract base class for patches and a
registry pattern for managing different patches.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Any

logger = logging.getLogger(__name__)


class ModelPatch(ABC):
    """
    Abstract base class for model patches.

    A model patch is responsible for modifying or extending model behavior
    to fix compatibility issues or add functionality.
    """

    def __init__(self, name: str, version: Optional[str] = None):
        """
        Initialize the model patch.

        Args:
            name: Descriptive name of the patch
            version: Optional version identifier for the patch
        """
        self.name = name
        self.version = version
        self._applied = False

    @abstractmethod
    def apply(self) -> bool:
        """
        Apply the patch to the target model/library.

        Returns:
            True if patch was successfully applied, False otherwise
        """
        pass

    def is_applied(self) -> bool:
        """Check if the patch has been applied."""
        return self._applied

    def _mark_applied(self) -> None:
        """Mark the patch as applied (internal use)."""
        self._applied = True

    def __str__(self) -> str:
        """String representation of the patch."""
        version_str = f" v{self.version}" if self.version else ""
        return f"{self.name}{version_str}"


class PatchRegistry:
    """
    Registry for managing model patches.

    This class provides a centralized way to register, retrieve, and apply
    model patches. It ensures patches are only applied once and provides
    logging for patch operations.
    """

    def __init__(self):
        """Initialize the patch registry."""
        self._patches: Dict[str, Type[ModelPatch]] = {}
        self._applied_patches: Dict[str, ModelPatch] = {}

    def register(self, name: str, patch_class: Type[ModelPatch]) -> None:
        """
        Register a patch class with the registry.

        Args:
            name: Unique identifier for the patch
            patch_class: The patch class to register
        """
        if name in self._patches:
            logger.warning(f"Patch '{name}' is already registered. Overwriting...")

        self._patches[name] = patch_class
        logger.debug(f"Registered patch: {name}")

    def get_patch(self, name: str, **kwargs) -> Optional[ModelPatch]:
        """
        Get a patch instance by name.

        Args:
            name: Name of the patch to retrieve
            **kwargs: Arguments to pass to the patch constructor

        Returns:
            Patch instance if found, None otherwise
        """
        if name not in self._patches:
            logger.error(f"Patch '{name}' not found in registry")
            return None

        try:
            patch_class = self._patches[name]
            return patch_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create patch '{name}': {e}")
            return None

    def apply_patch(self, name: str, **kwargs) -> bool:
        """
        Apply a patch by name.

        Args:
            name: Name of the patch to apply
            **kwargs: Arguments to pass to the patch constructor

        Returns:
            True if patch was successfully applied, False otherwise
        """
        # Check if patch is already applied
        if name in self._applied_patches:
            logger.info(f"Patch '{name}' already applied")
            return True

        # Get and apply the patch
        patch = self.get_patch(name, **kwargs)
        if patch is None:
            return False

        try:
            if patch.apply():
                self._applied_patches[name] = patch
                logger.info(f"Successfully applied patch: {patch}")
                return True
            else:
                logger.warning(f"Patch '{name}' application returned False")
                return False
        except Exception as e:
            logger.error(f"Failed to apply patch '{name}': {e}")
            return False

    def is_patch_applied(self, name: str) -> bool:
        """
        Check if a patch has been applied.

        Args:
            name: Name of the patch to check

        Returns:
            True if patch is applied, False otherwise
        """
        return name in self._applied_patches

    def list_patches(self) -> Dict[str, str]:
        """
        List all registered patches.

        Returns:
            Dictionary mapping patch names to their class names
        """
        return {name: cls.__name__ for name, cls in self._patches.items()}

    def list_applied_patches(self) -> Dict[str, str]:
        """
        List all applied patches.

        Returns:
            Dictionary mapping patch names to their string representations
        """
        return {name: str(patch) for name, patch in self._applied_patches.items()}


# Global patch registry instance
patch_registry = PatchRegistry()