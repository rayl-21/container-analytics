#!/usr/bin/env python3
"""
CLI Interface for YOLO Container Detector

This module provides a command-line interface for the YOLO container detection
system, supporting single image detection, batch processing, and watch mode
for continuous processing.

Features:
- Single image detection with optional annotation output
- Batch processing of image directories
- Watch mode for continuous processing of new images
- Configurable detection parameters (confidence, device, etc.)
- Performance statistics and reporting

Usage:
    # Single image detection
    python -m modules.detection.cli --image path/to/image.jpg --output results/

    # Batch processing
    python -m modules.detection.cli --batch path/to/images/ --output results/

    # Watch mode for continuous processing
    python -m modules.detection.cli --watch

    # Custom model and parameters
    python -m modules.detection.cli --image test.jpg --model yolov12x.pt --conf 0.7
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Import the detector and watch mode from the refactored modules
from .yolo_detector import YOLODetector
from .watch import YOLOWatchMode


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the YOLO detector CLI.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="YOLO Container Detector - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image photo.jpg --output results/
  %(prog)s --batch images/ --conf 0.7
  %(prog)s --watch --model yolov12x.pt
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to single image file for detection"
    )
    input_group.add_argument(
        "--batch",
        type=str,
        help="Path to directory containing images for batch processing"
    )
    input_group.add_argument(
        "--watch",
        action="store_true",
        help="Enable watch mode for continuous detection of new images"
    )

    # Model and detection parameters
    parser.add_argument(
        "--model",
        type=str,
        default="yolov12x.pt",
        help="Path to YOLO model weights (default: yolov12x.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for Non-Maximum Suppression (default: 0.7)"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device for inference (cpu/cuda, auto-detect if not specified)"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for annotated images (optional)"
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Save detection results to database (default for watch mode)"
    )

    # Batch processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for batch processing (default: 8)"
    )

    # Watch mode options
    parser.add_argument(
        "--watch-dir",
        type=str,
        default="data/images",
        help="Directory to watch for new images (default: data/images)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of worker threads for watch mode (default: 2)"
    )
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="Process existing images when starting watch mode"
    )

    # Display options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display detailed performance statistics"
    )

    return parser.parse_args()


def process_single_image(detector: YOLODetector, args: argparse.Namespace) -> None:
    """
    Process a single image file.

    Args:
        detector: Initialized YOLO detector
        args: Command line arguments
    """
    print(f"Processing single image: {args.image}")

    # Check if image file exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # Run detection
    try:
        result = detector.detect_single_image(
            args.image,
            return_annotated=bool(args.output)
        )

        # Display results
        num_detections = len(result['detections'])
        processing_time = result['metadata']['processing_time']

        print(f"Detections found: {num_detections}")
        print(f"Processing time: {processing_time:.3f} seconds")

        if args.verbose and num_detections > 0:
            print("\nDetection details:")
            for i, (class_id, confidence) in enumerate(
                zip(result['detections'].class_id, result['detections'].confidence)
            ):
                class_name = detector.CONTAINER_CLASSES.get(int(class_id), 'unknown')
                print(f"  {i+1}. {class_name}: {confidence:.3f}")

        # Save annotated image if output directory specified
        if args.output and "annotated_image" in result:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"annotated_{image_path.name}"
            result["annotated_image"].save(output_path)
            print(f"Saved annotated image: {output_path}")

        # Save to database if requested
        if args.save_db:
            image_id = detector.save_detection_to_database(
                args.image,
                result['detections'],
                processing_time
            )
            if image_id:
                print(f"Saved results to database (Image ID: {image_id})")

    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)


def process_batch(detector: YOLODetector, args: argparse.Namespace) -> None:
    """
    Process a batch of images from a directory.

    Args:
        detector: Initialized YOLO detector
        args: Command line arguments
    """
    print(f"Processing image batch from: {args.batch}")

    # Check if directory exists
    image_dir = Path(args.batch)
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"Error: Directory not found or not a directory: {args.batch}")
        sys.exit(1)

    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))

    if not image_paths:
        print(f"No image files found in {image_dir}")
        print(f"Supported extensions: {', '.join(image_extensions)}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images to process")

    # Process batch
    try:
        results = detector.detect_batch(
            image_paths,
            batch_size=args.batch_size,
            return_annotated=bool(args.output)
        )

        # Display summary
        total_detections = sum(len(result['detections']) for result in results)
        print(f"\nBatch processing complete:")
        print(f"  Images processed: {len(results)}")
        print(f"  Total detections: {total_detections}")

        # Show performance statistics
        if args.stats:
            perf_stats = detector.get_performance_stats()
            print(f"\nPerformance Statistics:")
            print(f"  Mean processing time: {perf_stats['mean_time']:.3f}s")
            print(f"  Mean FPS: {perf_stats['fps_mean']:.1f}")
            print(f"  Min/Max time: {perf_stats['min_time']:.3f}s / {perf_stats['max_time']:.3f}s")

        # Save annotated images if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            saved_count = 0
            for result in results:
                if "annotated_image" in result:
                    image_name = Path(result["metadata"]["image_path"]).name
                    output_path = output_dir / f"annotated_{image_name}"
                    result["annotated_image"].save(output_path)
                    saved_count += 1

            print(f"Saved {saved_count} annotated images to {output_dir}")

        # Save to database if requested
        if args.save_db:
            print("Saving results to database...")
            saved_count = 0
            for result in results:
                image_id = detector.save_detection_to_database(
                    result["metadata"]["image_path"],
                    result['detections'],
                    result["metadata"]["processing_time"]
                )
                if image_id:
                    saved_count += 1

            print(f"Saved {saved_count} image results to database")

    except Exception as e:
        print(f"Error during batch processing: {e}")
        sys.exit(1)


def run_watch_mode(detector: YOLODetector, args: argparse.Namespace) -> None:
    """
    Run watch mode for continuous image processing.

    Args:
        detector: Initialized YOLO detector
        args: Command line arguments
    """
    watch_dir = Path(args.watch_dir)

    # Create watch directory if it doesn't exist
    if not watch_dir.exists():
        watch_dir.mkdir(parents=True)
        print(f"Created watch directory: {watch_dir}")

    print(f"Starting watch mode on directory: {watch_dir}")
    print(f"Workers: {args.max_workers}, Batch size: {args.batch_size}")
    print("Press Ctrl+C to stop...")

    # Initialize watch mode
    watch_mode = YOLOWatchMode(
        detector=detector,
        watch_directory=watch_dir,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        process_existing=args.process_existing
    )

    try:
        with watch_mode.running_context(process_existing=args.process_existing):
            # Print periodic statistics
            while True:
                try:
                    time.sleep(10)  # Update every 10 seconds
                    stats = watch_mode.get_stats()
                    monitor = stats['monitor']
                    workers = stats['workers']
                    queue = stats['queue']

                    if args.verbose or args.stats:
                        print(f"\n--- Watch Mode Stats ---")
                        print(f"Running: {monitor['is_running']}")
                        print(f"Runtime: {monitor['runtime_seconds']:.1f}s")
                        print(f"Images Processed: {workers['total_images_processed']}")
                        print(f"Images Failed: {workers['total_images_failed']}")
                        print(f"Total Detections: {workers['total_detections']}")
                        print(f"Queue Size: {queue['current_size']}")
                        print(f"Success Rate: {workers['overall_success_rate']:.1f}%")
                        print(f"Active Workers: {workers['active_workers']}/{workers['total_workers']}")
                        print("Press Ctrl+C to stop...")
                    else:
                        # Show minimal status
                        print(f"Processed: {workers['total_images_processed']}, "
                              f"Detections: {workers['total_detections']}, "
                              f"Queue: {queue['current_size']}")

                except KeyboardInterrupt:
                    break

    except KeyboardInterrupt:
        print("\nShutting down watch mode...")

    # Show final statistics
    final_stats = watch_mode.get_stats()
    workers = final_stats['workers']
    print(f"\nFinal Statistics:")
    print(f"  Images processed: {workers['total_images_processed']}")
    print(f"  Images failed: {workers['total_images_failed']}")
    print(f"  Total detections: {workers['total_detections']}")
    print(f"  Success rate: {workers['overall_success_rate']:.1f}%")


def initialize_detector(args: argparse.Namespace) -> YOLODetector:
    """
    Initialize the YOLO detector with the specified arguments.

    Args:
        args: Command line arguments

    Returns:
        YOLODetector: Initialized detector instance
    """
    if args.verbose:
        print(f"Initializing YOLO detector...")
        print(f"  Model: {args.model}")
        print(f"  Confidence threshold: {args.conf}")
        print(f"  IoU threshold: {args.iou}")
        print(f"  Device: {args.device or 'auto-detect'}")

    try:
        detector = YOLODetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            verbose=args.verbose
        )

        if args.verbose:
            print(f"  Initialized on device: {detector.device}")

        return detector

    except Exception as e:
        print(f"Error initializing detector: {e}")
        sys.exit(1)


def main() -> None:
    """
    Main CLI entry point.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Initialize detector
    detector = initialize_detector(args)

    # Route to appropriate processing function
    try:
        if args.image:
            process_single_image(detector, args)
        elif args.batch:
            process_batch(detector, args)
        elif args.watch:
            run_watch_mode(detector, args)
        else:
            print("Error: No operation specified. Use --help for usage information.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()