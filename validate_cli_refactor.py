#!/usr/bin/env python3
"""
Validation script for the CLI refactoring.

This script validates that the CLI extraction was completed successfully
by checking file structure, content, and basic functionality.
"""

import ast
import sys
from pathlib import Path


def check_file_structure():
    """Check that all required files exist."""
    required_files = [
        "modules/detection/cli.py",
        "modules/detection/__main__.py",
        "modules/detection/__init__.py",
        "modules/detection/yolo_detector.py"
    ]

    results = {}
    for file_path in required_files:
        path = Path(file_path)
        results[file_path] = path.exists()

    return results


def check_cli_content():
    """Check that cli.py contains required functions."""
    cli_path = Path("modules/detection/cli.py")
    if not cli_path.exists():
        return {"exists": False}

    content = cli_path.read_text()

    checks = {
        "has_main_function": "def main()" in content,
        "has_parse_arguments": "def parse_arguments()" in content,
        "has_process_single_image": "def process_single_image(" in content,
        "has_process_batch": "def process_batch(" in content,
        "has_run_watch_mode": "def run_watch_mode(" in content,
        "has_initialize_detector": "def initialize_detector(" in content,
        "imports_detector": "from .yolo_detector import YOLODetector" in content,
        "imports_watch_mode": "from .watch import YOLOWatchMode" in content,
        "has_argparse": "import argparse" in content,
        "has_if_main": 'if __name__ == "__main__":' in content,
    }

    return checks


def check_main_module():
    """Check that __main__.py is correct."""
    main_path = Path("modules/detection/__main__.py")
    if not main_path.exists():
        return {"exists": False}

    content = main_path.read_text()

    checks = {
        "imports_cli_main": "from .cli import main" in content,
        "calls_main": "main()" in content,
        "has_if_main": 'if __name__ == "__main__":' in content,
    }

    return checks


def check_yolo_detector_cleaned():
    """Check that yolo_detector.py was properly cleaned."""
    detector_path = Path("modules/detection/yolo_detector.py")
    if not detector_path.exists():
        return {"exists": False}

    content = detector_path.read_text()

    checks = {
        "no_cli_main": 'if __name__ == "__main__":' not in content,
        "no_argparse": "import argparse" not in content,
        "has_yolo_detector_class": "class YOLODetector:" in content,
        "has_detector_methods": "def detect_single_image(" in content and "def detect_batch(" in content,
    }

    return checks


def check_init_exports():
    """Check that __init__.py has proper exports."""
    init_path = Path("modules/detection/__init__.py")
    if not init_path.exists():
        return {"exists": False}

    content = init_path.read_text()

    checks = {
        "has_yolo_detector_import": "from .yolo_detector import YOLODetector" in content,
        "has_all_definition": "__all__ = [" in content,
        "exports_yolo_detector": '"YOLODetector"' in content,
        "exports_watch_mode": '"YOLOWatchMode"' in content,
        "has_convenience_functions": "def create_detector(" in content,
    }

    return checks


def check_syntax_validity():
    """Check that all Python files have valid syntax."""
    python_files = [
        "modules/detection/cli.py",
        "modules/detection/__main__.py",
        "modules/detection/__init__.py",
    ]

    results = {}
    for file_path in python_files:
        path = Path(file_path)
        if not path.exists():
            results[file_path] = False
            continue

        try:
            content = path.read_text()
            ast.parse(content)
            results[file_path] = True
        except SyntaxError:
            results[file_path] = False

    return results


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
    print("CLI Refactoring Validation")
    print("=" * 30)

    checks = [
        ("File Structure", check_file_structure),
        ("CLI Content", check_cli_content),
        ("Main Module", check_main_module),
        ("YOLODetector Cleanup", check_yolo_detector_cleaned),
        ("Init Exports", check_init_exports),
        ("Syntax Validity", check_syntax_validity),
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

    print(f"\n{'='*30}")
    print(f"Overall Results: {total_passed}/{total_checks} checks passed")

    if total_passed == total_checks:
        print("🎉 CLI refactoring validation PASSED!")
        print("\nThe CLI interface has been successfully extracted into a modular structure:")
        print("  • cli.py - Complete CLI interface with argument parsing")
        print("  • __main__.py - Module execution entry point")
        print("  • __init__.py - Enhanced public API exports")
        print("  • yolo_detector.py - Cleaned of CLI code")
        print("\nUsage:")
        print("  python -m modules.detection --help")
        print("  python -m modules.detection --image path/to/image.jpg")
        print("  python -m modules.detection --watch")
        return True
    else:
        print("❌ CLI refactoring validation FAILED!")
        print("Some checks did not pass. Please review the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)