#!/usr/bin/env python3
"""
Quick test to verify detection saving works.
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

    # Find just a few images to test
    image_dir = Path("data/images")
    test_images = []

    # Get first 5 images from 2025-09-02 (which has more detections)
    date_path = image_dir / "2025-09-02"
    if date_path.exists():
        images = list(date_path.glob("*.jpg"))[:5]
        test_images.extend([str(img) for img in images])
        logger.info(f"Found {len(images)} test images from 2025-09-02")

    if not test_images:
        logger.error("No test images found")
        return

    logger.info(f"Testing with {len(test_images)} images")

    # Process images
    with session_scope() as session:
        for image_path in test_images:
            logger.info(f"Processing {Path(image_path).name}")

            try:
                # Run detection on single image
                results = detector.detect_batch([image_path])

                if results and len(results) > 0:
                    result = results[0]

                    # Extract detections from result dictionary
                    if isinstance(result, dict) and 'detections' in result:
                        detections = result['detections']

                        if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                            detection_count = len(detections.xyxy)
                            logger.info(f"Found {detection_count} detections")

                            # Get or create image record
                            image = session.query(Image).filter_by(image_path=str(image_path)).first()
                            if not image:
                                image = Image(
                                    image_path=str(image_path),
                                    camera_id='in_gate',
                                    timestamp=datetime.now(),
                                    processed=True
                                )
                                session.add(image)
                                session.flush()
                                logger.info(f"Created new Image record with id={image.id}")
                            else:
                                logger.info(f"Found existing Image record with id={image.id}")

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
                                logger.info(f"  Detection {i+1}: {object_type} (class_id={class_id}, conf={conf:.2f})")

                            session.commit()
                            logger.info(f"✓ Successfully saved {detection_count} detections")
                        else:
                            logger.info("No detections found in image")
                    else:
                        logger.warning("Unexpected result format")

            except Exception as e:
                logger.error(f"Error processing {Path(image_path).name}: {e}")
                session.rollback()

    # Check database
    with session_scope() as session:
        total_images = session.query(Image).count()
        total_detections = session.query(Detection).count()
        logger.info("=" * 60)
        logger.info("DATABASE CHECK")
        logger.info("=" * 60)
        logger.info(f"Total Images in DB: {total_images}")
        logger.info(f"Total Detections in DB: {total_detections}")

        # Show recent detections
        recent_detections = session.query(Detection).order_by(Detection.timestamp.desc()).limit(5).all()
        if recent_detections:
            logger.info("\nRecent detections:")
            for det in recent_detections:
                logger.info(f"  - Detection {det.id}: {det.object_type or 'unknown'} (class_id={det.class_id}, conf={det.confidence:.2f})")

if __name__ == "__main__":
    main()