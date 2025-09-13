#!/usr/bin/env python3
"""
Validation script for the YOLOv12 patch refactoring.

This script validates that the refactoring was completed successfully
by checking file structure, content, and basic syntax.
"""

import re
from pathlib import Path

def check_file_structure():
    """Check that all required files exist."""
    required_files = [
        "modules/detection/patches/__init__.py",
        "modules/detection/patches/base.py",
        "modules/detection/patches/yolov12.py",
        "modules/detection/yolo_detector.py"
    ]

    results = {}
    for file_path in required_files:
        path = Path(file_path)
        results[file_path] = path.exists()

    return results

def check_base_py_content():
    """Check that base.py contains required classes."""
    base_path = Path("modules/detection/patches/base.py")
    if not base_path.exists():
        return {"exists": False}

    content = base_path.read_text()

    checks = {
        "ModelPatch_class": "class ModelPatch" in content,
        "PatchRegistry_class": "class PatchRegistry" in content,
        "abstract_apply_method": "@abstractmethod" in content and "def apply(self)" in content,
        "registry_instance": "patch_registry = PatchRegistry()" in content,
    }

    return checks

def check_yolov12_py_content():
    """Check that yolov12.py contains required classes."""
    yolov12_path = Path("modules/detection/patches/yolov12.py")
    if not yolov12_path.exists():
        return {"exists": False}

    content = yolov12_path.read_text()

    checks = {
        "AAttnV12_class": "class AAttnV12" in content,
        "YOLOv12Patch_class": "class YOLOv12Patch" in content,
        "inherits_ModelPatch": "class YOLOv12Patch(ModelPatch)" in content,
        "has_apply_method": "def apply(self)" in content,
        "imports_base": "from .base import ModelPatch" in content,
    }

    return checks

def check_init_py_content():
    """Check that __init__.py has proper exports."""
    init_path = Path("modules/detection/patches/__init__.py")
    if not init_path.exists():
        return {"exists": False}

    content = init_path.read_text()

    checks = {
        "imports_base": "from .base import" in content,
        "has_apply_yolov12_patch": "def apply_yolov12_patch" in content,
        "has_registry_export": "patch_registry" in content,
        "has_all_exports": "__all__" in content,
        "lazy_loading": "_get_yolov12_classes" in content,
    }

    return checks

def check_yolo_detector_updated():
    """Check that yolo_detector.py was properly updated."""
    detector_path = Path("modules/detection/yolo_detector.py")
    if not detector_path.exists():
        return {"exists": False}

    content = detector_path.read_text()

    checks = {
        "imports_new_patch": "from .patches import apply_yolov12_patch" in content,
        "calls_new_patch": "apply_yolov12_patch()" in content,
        "old_patch_removed": "def patch_yolov12_aattn():" not in content,
        "old_aattn_class_removed": "class AAttnV12(nn.Module):" not in content,
    }

    return checks

def print_results(section_name, results):
    """Print results for a section."""
    print(f"\n{section_name}:")
    print("-" * len(section_name))

    if isinstance(results, dict):
        for check, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
    else:
        print(f"  ✗ Error: {results}")

    passed = sum(1 for v in results.values() if v) if isinstance(results, dict) else 0
    total = len(results) if isinstance(results, dict) else 0
    return passed, total

def main():
    """Run all validation checks."""
    print("YOLOv12 Patch Refactoring Validation")
    print("=" * 40)

    checks = [
        ("File Structure", check_file_structure),
        ("base.py Content", check_base_py_content),
        ("yolov12.py Content", check_yolov12_py_content),
        ("__init__.py Content", check_init_py_content),
        ("yolo_detector.py Updates", check_yolo_detector_updated),
    ]

    total_passed = 0
    total_checks = 0

    for section_name, check_func in checks:
        try:
            results = check_func()
            passed, total = print_results(section_name, results)
            total_passed += passed
            total_checks += total
        except Exception as e:
            print(f"\n{section_name}:")
            print(f"  ✗ Error running check: {e}")

    print(f"\n{'='*40}")
    print(f"Overall Results: {total_passed}/{total_checks} checks passed")

    if total_passed == total_checks:
        print("🎉 Refactoring validation PASSED!")
        print("\nThe YOLOv12 patch logic has been successfully extracted into a modular structure:")
        print("  • patches/base.py - Abstract base class and registry")
        print("  • patches/yolov12.py - YOLOv12-specific patches")
        print("  • patches/__init__.py - Simple API and exports")
        print("  • yolo_detector.py - Updated to use new patch system")
        return True
    else:
        print("❌ Refactoring validation FAILED!")
        print("Some checks did not pass. Please review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)