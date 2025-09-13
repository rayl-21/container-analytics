"""
Model patches for YOLOv12 and other models.

This package provides a modular system for applying patches to fix
compatibility issues and add functionality to various model implementations.

The patch system is designed to be:
- Modular: Each patch is self-contained
- Extensible: New patches can be easily added
- Safe: Patches are applied only once and handle errors gracefully
- Traceable: Full logging of patch operations

Usage:
    # Simple API - apply YOLOv12 patch
    from modules.detection.patches import apply_yolov12_patch
    apply_yolov12_patch()

    # Advanced API - use registry directly
    from modules.detection.patches import patch_registry
    patch_registry.apply_patch('yolov12_aattn')

    # Check if patches are applied
    from modules.detection.patches import is_yolov12_patched
    if is_yolov12_patched():
        print("YOLOv12 patches are active")
"""

import logging
from typing import Dict, List, Optional

from .base import ModelPatch, PatchRegistry, patch_registry

logger = logging.getLogger(__name__)

# Lazy imports to avoid heavy dependencies on module import
def _get_yolov12_classes():
    """Lazy import of YOLOv12 classes to avoid torch dependency on import."""
    try:
        from .yolov12 import YOLOv12Patch, YOLOv12PerformancePatch, AAttnV12
        return YOLOv12Patch, YOLOv12PerformancePatch, AAttnV12
    except ImportError as e:
        logger.warning(f"Failed to import YOLOv12 patches: {e}")
        return None, None, None

# Register patches with lazy loading
def _register_patches():
    """Register all available patches with the registry."""
    YOLOv12Patch, YOLOv12PerformancePatch, _ = _get_yolov12_classes()

    if YOLOv12Patch:
        patch_registry.register('yolov12_aattn', YOLOv12Patch)
        logger.debug("Registered YOLOv12 AAttn patch")

    if YOLOv12PerformancePatch:
        patch_registry.register('yolov12_performance', YOLOv12PerformancePatch)
        logger.debug("Registered YOLOv12 performance patch")

# Register patches on module import
_register_patches()

# Export main classes and registry
__all__ = [
    'ModelPatch',
    'PatchRegistry',
    'patch_registry',
    'apply_yolov12_patch',
    'apply_all_yolov12_patches',
    'is_yolov12_patched',
    'is_patch_applied',
    'list_patches',
    'list_applied_patches',
]

# Lazy export of YOLOv12 classes
def __getattr__(name):
    """Lazy loading of YOLOv12 classes to avoid import errors."""
    if name in ['YOLOv12Patch', 'YOLOv12PerformancePatch', 'AAttnV12']:
        YOLOv12Patch, YOLOv12PerformancePatch, AAttnV12 = _get_yolov12_classes()
        if name == 'YOLOv12Patch':
            return YOLOv12Patch
        elif name == 'YOLOv12PerformancePatch':
            return YOLOv12PerformancePatch
        elif name == 'AAttnV12':
            return AAttnV12

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def apply_yolov12_patch() -> bool:
    """
    Apply the core YOLOv12 AAttn compatibility patch.

    This is the main patch needed for YOLOv12 models to work properly
    with the ultralytics library. It fixes the AAttn attribute error.

    Returns:
        True if patch was successfully applied, False otherwise
    """
    return patch_registry.apply_patch('yolov12_aattn')


def apply_all_yolov12_patches() -> Dict[str, bool]:
    """
    Apply all available YOLOv12 patches.

    This applies both the core AAttn patch and any performance patches.

    Returns:
        Dictionary mapping patch names to their application status
    """
    results = {}
    yolov12_patches = ['yolov12_aattn', 'yolov12_performance']

    for patch_name in yolov12_patches:
        try:
            results[patch_name] = patch_registry.apply_patch(patch_name)
        except Exception as e:
            logger.error(f"Failed to apply patch '{patch_name}': {e}")
            results[patch_name] = False

    return results


def is_yolov12_patched() -> bool:
    """
    Check if the core YOLOv12 patch has been applied.

    Returns:
        True if the YOLOv12 AAttn patch is applied, False otherwise
    """
    return patch_registry.is_patch_applied('yolov12_aattn')


def is_patch_applied(patch_name: str) -> bool:
    """
    Check if a specific patch has been applied.

    Args:
        patch_name: Name of the patch to check

    Returns:
        True if patch is applied, False otherwise
    """
    return patch_registry.is_patch_applied(patch_name)


def list_patches() -> Dict[str, str]:
    """
    List all available patches.

    Returns:
        Dictionary mapping patch names to their class names
    """
    return patch_registry.list_patches()


def list_applied_patches() -> Dict[str, str]:
    """
    List all currently applied patches.

    Returns:
        Dictionary mapping patch names to their descriptions
    """
    return patch_registry.list_applied_patches()


def get_patch_status() -> Dict[str, Dict[str, any]]:
    """
    Get comprehensive status of all patches.

    Returns:
        Dictionary with patch information including availability and application status
    """
    available = list_patches()
    applied = list_applied_patches()

    status = {}
    for name, class_name in available.items():
        status[name] = {
            'class_name': class_name,
            'applied': name in applied,
            'description': applied.get(name, 'Not applied')
        }

    return status


# Auto-apply critical patches on import (optional behavior)
def _auto_apply_critical_patches() -> None:
    """
    Automatically apply critical patches on module import.

    This is disabled by default but can be enabled by setting the
    AUTO_APPLY_PATCHES environment variable.
    """
    import os

    if os.getenv('AUTO_APPLY_PATCHES', '').lower() in ('true', '1', 'yes'):
        logger.info("Auto-applying critical patches...")
        if apply_yolov12_patch():
            logger.info("Successfully auto-applied YOLOv12 patch")
        else:
            logger.warning("Failed to auto-apply YOLOv12 patch")


# Initialize logging for this module
logger.debug(f"Initialized patches module with {len(list_patches())} available patches")

# Optionally auto-apply patches
_auto_apply_critical_patches()