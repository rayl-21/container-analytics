# YOLO Model Patches

This package provides a modular system for applying patches to fix compatibility issues and add functionality to YOLO models, particularly YOLOv12.

## Overview

The patch system is designed to be:
- **Modular**: Each patch is self-contained and can be applied independently
- **Extensible**: New patches can be easily added for different models
- **Safe**: Patches are applied only once and handle errors gracefully
- **Traceable**: Full logging of patch operations

## Architecture

### Core Components

1. **`base.py`** - Foundation classes
   - `ModelPatch`: Abstract base class for all patches
   - `PatchRegistry`: Registry pattern for managing patches
   - `patch_registry`: Global registry instance

2. **`yolov12.py`** - YOLOv12-specific patches
   - `AAttnV12`: Fixed Area Attention module for YOLOv12
   - `YOLOv12Patch`: Main patch for AAttn compatibility
   - `YOLOv12PerformancePatch`: Future performance optimizations

3. **`__init__.py`** - Public API and exports
   - Simple functions for applying patches
   - Lazy loading to avoid heavy dependencies
   - Registry management functions

## Usage

### Quick Start

```python
# Apply the main YOLOv12 compatibility patch
from modules.detection.patches import apply_yolov12_patch
apply_yolov12_patch()
```

### Advanced Usage

```python
# Use the registry directly
from modules.detection.patches import patch_registry

# List available patches
patches = patch_registry.list_patches()
print(f"Available patches: {list(patches.keys())}")

# Apply a specific patch
success = patch_registry.apply_patch('yolov12_aattn')

# Check if patch is applied
is_applied = patch_registry.is_patch_applied('yolov12_aattn')
```

### Creating New Patches

```python
from modules.detection.patches.base import ModelPatch, patch_registry

class MyCustomPatch(ModelPatch):
    def __init__(self):
        super().__init__("My Custom Patch", "1.0")

    def apply(self) -> bool:
        try:
            # Your patch logic here
            self._mark_applied()
            return True
        except Exception as e:
            logger.error(f"Patch failed: {e}")
            return False

# Register the patch
patch_registry.register('my_custom_patch', MyCustomPatch)
```

## YOLOv12 Compatibility Issue

The main patch (`YOLOv12Patch`) fixes a critical compatibility issue where YOLOv12 models fail with:
```
AttributeError: 'AAttn' object has no attribute 'qkv'
```

This occurs because YOLOv12 uses separate `qk` and `v` convolutions instead of the combined `qkv` approach used in other YOLO versions. Our `AAttnV12` class provides the correct implementation.

## Dependencies

- **Core**: No additional dependencies beyond Python standard library
- **YOLOv12 patches**: Requires `torch`, `ultralytics`
- **Flash Attention**: Optional `flash_attn` for performance (auto-detected)

## Files

```
patches/
├── __init__.py          # Public API and lazy loading
├── base.py              # Abstract base classes and registry
├── yolov12.py          # YOLOv12-specific patches
└── README.md           # This documentation
```

## Migration from Legacy Code

If you were previously using the inline `patch_yolov12_aattn()` function:

**Before:**
```python
# Old approach
patch_yolov12_aattn()
```

**After:**
```python
# New approach
from modules.detection.patches import apply_yolov12_patch
apply_yolov12_patch()
```

The functionality is identical, but the new approach provides better modularity and extensibility.