#!/usr/bin/env python3
"""
Run YOLO detection on images from specified date range.
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set testing environment to avoid authentication
os.environ['APP_ENV'] = 'testing'

from modules.detection.yolo_detector import YOLODetector
from modules.database.models import session_scope, Image, Detection
from modules.database.queries import DatabaseQueries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize detector with YOLOv12 model
    detector = YOLODetector(
        model_path="data/models/yolov12x.pt",
        confidence_threshold=0.1,
        iou_threshold=0.85,
        verbose=True
    )

    # Class ID to name mapping from YOLODetector
    CLASS_NAMES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        8: 'boat',
    }

    # Find all images in date range
    image_dir = Path("data/images")
    date_dirs = [
        "2025-09-01", "2025-09-02", "2025-09-03",
        "2025-09-04", "2025-09-05", "2025-09-06", "2025-09-07"
    ]

    all_images = []
    for date_dir in date_dirs:
        date_path = image_dir / date_dir
        if date_path.exists():
            images = list(date_path.glob("*.jpg")) + list(date_path.glob("*.jpeg"))
            all_images.extend([str(img) for img in images])
            logger.info(f"Found {len(images)} images in {date_dir}")

    logger.info(f"Total images to process: {len(all_images)}")

    if not all_images:
        logger.error("No images found to process")
        return

    # Process images in batches
    batch_size = 32
    total_detections = 0
    failed_images = []

    with session_scope() as session:
        db_queries = DatabaseQueries()

        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_images) + batch_size - 1)//batch_size}")

            try:
                # Run detection on batch
                results = detector.detect_batch(batch)

                # Process each result
                for idx, result in enumerate(results):
                    image_path = batch[idx]

                    # Extract detections from result dictionary
                    if isinstance(result, dict) and 'detections' in result:
                        detections = result['detections']
                        # detections is a supervision.Detections object
                        if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                            # Log detection details
                            detection_count = len(detections.xyxy)
                            total_detections += detection_count
                            logger.info(f"Found {detection_count} detections in {Path(image_path).name}")

                            # Save to database
                            try:
                                # Get or create image record
                                image = session.query(Image).filter_by(image_path=str(image_path)).first()
                                if not image:
                                    # Create image record if it doesn't exist
                                    image = Image(
                                        image_path=str(image_path),
                                        camera_id='in_gate',
                                        timestamp=datetime.now(),
                                        processed=True
                                    )
                                    session.add(image)
                                    session.flush()

                                # Save each detection
                                for i in range(len(detections.xyxy)):
                                    x1, y1, x2, y2 = detections.xyxy[i]
                                    conf = detections.confidence[i]
                                    class_id = detections.class_id[i]

                                    # Get object type name from class ID
                                    object_type = CLASS_NAMES.get(int(class_id), f'class_{class_id}')

                                    det_record = Detection(
                                        image_id=image.id,
                                        class_id=int(class_id),
                                        object_type=object_type,  # Add object_type
                                        confidence=float(conf),
                                        x1=float(x1),
                                        y1=float(y1),
                                        x2=float(x2),
                                        y2=float(y2),
                                        timestamp=datetime.now()
                                    )
                                    session.add(det_record)

                                session.commit()
                                logger.info(f"Saved {detection_count} detections for {Path(image_path).name}")

                            except Exception as e:
                                logger.error(f"Error saving detections: {e}")
                                session.rollback()
                                failed_images.append(image_path)
                        else:
                            logger.debug(f"No detections found in {Path(image_path).name}")
                    else:
                        logger.debug(f"No detections in result for {Path(image_path).name}")

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                failed_images.extend(batch)

        # Commit all changes
        session.commit()

    # Print summary
    logger.info("=" * 60)
    logger.info("DETECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images processed: {len(all_images)}")
    logger.info(f"Total detections saved: {total_detections}")
    logger.info(f"Failed images: {len(failed_images)}")

    if failed_images:
        logger.warning("Failed to process the following images:")
        for img in failed_images[:10]:  # Show first 10 failures
            logger.warning(f"  - {img}")
        if len(failed_images) > 10:
            logger.warning(f"  ... and {len(failed_images) - 10} more")

    # Get performance statistics
    try:
        stats = detector.get_performance_stats()
        if 'avg_detection_time' in stats:
            logger.info(f"Average detection time: {stats['avg_detection_time']:.3f} seconds")
        if 'total_detection_time' in stats:
            logger.info(f"Total detection time: {stats['total_detection_time']:.2f} seconds")
        if 'total_images_processed' in stats:
            logger.info(f"Images processed: {stats['total_images_processed']}")
    except Exception as e:
        logger.warning(f"Could not retrieve performance statistics: {e}")

if __name__ == "__main__":
    main()