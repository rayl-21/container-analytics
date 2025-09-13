"""
Image annotation utilities for YOLO detection results.

This module provides functionality to annotate images with detection results,
including bounding boxes, labels, and confidence scores.
"""

import numpy as np
import cv2
from PIL import Image
import supervision as sv
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ImageAnnotator:
    """
    Handles image annotation for detection results.

    This class provides methods to annotate images with bounding boxes,
    labels, and other visual indicators for detected objects.
    """

    def __init__(self, container_classes: Dict[int, str]):
        """
        Initialize the image annotator.

        Args:
            container_classes: Dictionary mapping class IDs to class names
        """
        self.container_classes = container_classes
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def annotate_image(self, image: np.ndarray, detections: sv.Detections) -> Image.Image:
        """
        Annotate image with detection bounding boxes and labels.

        Args:
            image: Input image as numpy array
            detections: Detection results from YOLO

        Returns:
            PIL Image with annotations
        """
        try:
            # Generate labels with class names and confidence scores
            labels = [
                f"{self.container_classes.get(class_id, 'unknown')} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]

            # Apply bounding box annotations
            annotated_image = self.box_annotator.annotate(image.copy(), detections)

            # Apply label annotations
            annotated_image = self.label_annotator.annotate(annotated_image, detections, labels)

            # Convert from BGR to RGB for PIL
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            return Image.fromarray(annotated_image_rgb)

        except Exception as e:
            logger.error(f"Error annotating image: {e}")
            # Return original image as PIL if annotation fails
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)

    def annotate_with_custom_style(
        self,
        image: np.ndarray,
        detections: sv.Detections,
        box_color: tuple = (0, 255, 0),
        text_color: tuple = (255, 255, 255),
        thickness: int = 2
    ) -> Image.Image:
        """
        Annotate image with custom styling options.

        Args:
            image: Input image as numpy array
            detections: Detection results from YOLO
            box_color: RGB color for bounding boxes
            text_color: RGB color for text labels
            thickness: Line thickness for bounding boxes

        Returns:
            PIL Image with custom styled annotations
        """
        try:
            # Create custom annotators with specified colors
            box_annotator = sv.BoxAnnotator(
                color=sv.Color.from_rgb_tuple(box_color),
                thickness=thickness
            )
            label_annotator = sv.LabelAnnotator(
                color=sv.Color.from_rgb_tuple(text_color)
            )

            # Generate labels
            labels = [
                f"{self.container_classes.get(class_id, 'unknown')} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]

            # Apply annotations
            annotated_image = box_annotator.annotate(image.copy(), detections)
            annotated_image = label_annotator.annotate(annotated_image, detections, labels)

            # Convert to PIL Image
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(annotated_image_rgb)

        except Exception as e:
            logger.error(f"Error annotating image with custom style: {e}")
            # Fallback to default annotation
            return self.annotate_image(image, detections)

    def create_detection_summary_image(
        self,
        image: np.ndarray,
        detections: sv.Detections,
        metadata: Dict
    ) -> Image.Image:
        """
        Create an annotated image with detection summary information.

        Args:
            image: Input image as numpy array
            detections: Detection results from YOLO
            metadata: Detection metadata including processing time

        Returns:
            PIL Image with annotations and summary information
        """
        try:
            # Start with basic annotation
            annotated_image = self.annotate_image(image, detections)
            annotated_array = np.array(annotated_image)

            # Add summary text at the top
            summary_text = [
                f"Detections: {len(detections)}",
                f"Processing time: {metadata.get('processing_time', 0):.3f}s",
                f"Confidence threshold: {metadata.get('model_confidence', 'N/A')}"
            ]

            # Draw background rectangle for text
            cv2.rectangle(annotated_array, (10, 10), (400, 80), (0, 0, 0), -1)
            cv2.rectangle(annotated_array, (10, 10), (400, 80), (255, 255, 255), 2)

            # Add text lines
            for i, text in enumerate(summary_text):
                y_pos = 30 + i * 20
                cv2.putText(
                    annotated_array, text, (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                )

            return Image.fromarray(annotated_array)

        except Exception as e:
            logger.error(f"Error creating detection summary image: {e}")
            # Fallback to basic annotation
            return self.annotate_image(image, detections)

    def get_detection_overlay_data(self, detections: sv.Detections) -> List[Dict]:
        """
        Get structured overlay data for frontend visualization.

        Args:
            detections: Detection results from YOLO

        Returns:
            List of dictionaries containing overlay information
        """
        overlay_data = []

        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            confidence = detections.confidence[i]
            class_id = detections.class_id[i]

            overlay_info = {
                'id': i,
                'class_name': self.container_classes.get(int(class_id), 'unknown'),
                'class_id': int(class_id),
                'confidence': float(confidence),
                'bbox': {
                    'x': float(bbox[0]),
                    'y': float(bbox[1]),
                    'width': float(bbox[2] - bbox[0]),
                    'height': float(bbox[3] - bbox[1])
                },
                'center': {
                    'x': float((bbox[0] + bbox[2]) / 2),
                    'y': float((bbox[1] + bbox[3]) / 2)
                },
                'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            }

            overlay_data.append(overlay_info)

        return overlay_data