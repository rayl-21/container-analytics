"""
Module execution entry point for the detection module.

This allows the detection module to be run directly as:
    python -m modules.detection [args...]

Which will invoke the CLI interface.
"""

from .cli import main

if __name__ == "__main__":
    main()