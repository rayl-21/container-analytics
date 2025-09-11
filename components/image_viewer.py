"""
Image display component with YOLO detection annotations.

This module provides an interactive image viewer that displays
container images with bounding boxes, labels, and detection metadata.
"""

import streamlit as st
import cv2
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional, Tuple, Any
import json
from datetime import datetime
import os


class ImageViewer:
    """Interactive image viewer with YOLO detection annotations."""
    
    def __init__(self):
        """Initialize the ImageViewer."""
        self.colors = {
            'container': (46, 134, 171),      # Blue
            'truck': (242, 66, 54),           # Red  
            'person': (246, 174, 45),         # Yellow
            'vehicle': (47, 155, 105),        # Green
            'other': (128, 128, 128)          # Gray
        }
        
    def display_image_with_detections(
        self,
        image_path: str,
        detections: List[Dict],
        title: str = "Image with Detections",
        show_confidence: bool = True,
        show_labels: bool = True,
        min_confidence: float = 0.5,
        zoom_enabled: bool = True
    ) -> None:
        """
        Display image with YOLO detection bounding boxes.
        
        Args:
            image_path: Path to the image file
            detections: List of detection dictionaries
            title: Title for the image display
            show_confidence: Whether to show confidence scores
            show_labels: Whether to show class labels
            min_confidence: Minimum confidence threshold for display
            zoom_enabled: Enable zoom functionality
        """
        if not os.path.exists(image_path):
            st.error(f"Image not found: {image_path}")
            return
            
        # Load and process image
        image = self._load_image(image_path)
        annotated_image = self._annotate_image(
            image, detections, show_confidence, show_labels, min_confidence
        )
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(title)
            
            if zoom_enabled:
                # Use st.image with use_column_width for responsive display
                st.image(
                    annotated_image,
                    caption=f"Detections: {len([d for d in detections if d.get('confidence', 0) >= min_confidence])}",
                    use_column_width=True
                )
            else:
                st.image(annotated_image, caption=title)
        
        with col2:
            self._display_detection_metadata(detections, min_confidence)
    
    def display_image_grid(
        self,
        images_data: List[Dict],
        columns: int = 3,
        show_metadata: bool = True
    ) -> None:
        """
        Display multiple images in a grid layout.
        
        Args:
            images_data: List of dicts with 'path', 'detections', 'timestamp'
            columns: Number of columns in the grid
            show_metadata: Whether to show detection counts
        """
        if not images_data:
            st.warning("No images to display")
            return
        
        # Create grid layout
        for i in range(0, len(images_data), columns):
            cols = st.columns(columns)
            
            for j, col in enumerate(cols):
                if i + j < len(images_data):
                    img_data = images_data[i + j]
                    
                    with col:
                        if os.path.exists(img_data['path']):
                            image = self._load_image(img_data['path'])
                            annotated = self._annotate_image(
                                image, 
                                img_data.get('detections', []),
                                show_confidence=False,
                                show_labels=False
                            )
                            
                            st.image(annotated, use_column_width=True)
                            
                            if show_metadata:
                                detection_count = len(img_data.get('detections', []))
                                timestamp = img_data.get('timestamp', 'Unknown')
                                st.caption(f"🕒 {timestamp}")
                                st.caption(f"📦 {detection_count} detections")
    
    def create_image_carousel(
        self,
        images_data: List[Dict],
        key: str = "image_carousel"
    ) -> Optional[Dict]:
        """
        Create an interactive image carousel.
        
        Args:
            images_data: List of image data dictionaries
            key: Unique key for the carousel
            
        Returns:
            Selected image data or None
        """
        if not images_data:
            st.warning("No images available")
            return None
        
        # Image selection slider
        image_index = st.select_slider(
            "Select Image",
            options=range(len(images_data)),
            format_func=lambda x: f"Image {x + 1} ({images_data[x].get('timestamp', 'Unknown')})",
            key=f"{key}_slider"
        )
        
        selected_image = images_data[image_index]
        
        # Display selected image with controls
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous", key=f"{key}_prev", disabled=image_index == 0):
                st.rerun()
        
        with col2:
            st.write(f"**Image {image_index + 1} of {len(images_data)}**")
        
        with col3:
            if st.button("➡️ Next", key=f"{key}_next", disabled=image_index == len(images_data) - 1):
                st.rerun()
        
        # Display the selected image
        self.display_image_with_detections(
            selected_image['path'],
            selected_image.get('detections', []),
            title=f"Image {image_index + 1}",
            zoom_enabled=True
        )
        
        return selected_image
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _annotate_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True,
        show_labels: bool = True,
        min_confidence: float = 0.5
    ) -> np.ndarray:
        """
        Annotate image with detection bounding boxes and labels.
        
        Args:
            image: Input image as numpy array
            detections: List of detection dictionaries
            show_confidence: Whether to show confidence scores
            show_labels: Whether to show class labels
            min_confidence: Minimum confidence threshold
            
        Returns:
            Annotated image as numpy array
        """
        annotated = image.copy()
        height, width = image.shape[:2]
        
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(annotated)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            # Try to load a font, fallback to default if not available
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for detection in detections:
            confidence = detection.get('confidence', 0.0)
            if confidence < min_confidence:
                continue
            
            # Extract bounding box coordinates
            bbox = detection.get('bbox', {})
            if isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
            else:
                x1 = bbox.get('x1', bbox.get('xmin', 0))
                y1 = bbox.get('y1', bbox.get('ymin', 0))
                x2 = bbox.get('x2', bbox.get('xmax', width))
                y2 = bbox.get('y2', bbox.get('ymax', height))
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            
            # Get class and color
            class_name = detection.get('class', detection.get('label', 'unknown'))
            color = self.colors.get(class_name.lower(), self.colors['other'])
            
            # Draw bounding box
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=2
            )
            
            # Prepare label text
            label_parts = []
            if show_labels:
                label_parts.append(class_name.capitalize())
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            label_text = " | ".join(label_parts)
            
            if label_text:
                # Calculate text size and position
                bbox_text = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # Position label above the bounding box
                label_x = x1
                label_y = max(0, y1 - text_height - 5)
                
                # Draw background rectangle for text
                draw.rectangle(
                    [(label_x, label_y), (label_x + text_width, label_y + text_height)],
                    fill=color,
                    outline=color
                )
                
                # Draw text
                draw.text(
                    (label_x, label_y),
                    label_text,
                    fill=(255, 255, 255),
                    font=font
                )
            
            # Add container number if available
            container_number = detection.get('container_number')
            if container_number and show_labels:
                container_text = f"ID: {container_number}"
                bbox_container = draw.textbbox((0, 0), container_text, font=font)
                container_width = bbox_container[2] - bbox_container[0]
                container_height = bbox_container[3] - bbox_container[1]
                
                # Position below the main label
                container_x = x1
                container_y = y1 + 5
                
                draw.rectangle(
                    [(container_x, container_y), (container_x + container_width, container_y + container_height)],
                    fill=(0, 0, 0),
                    outline=(0, 0, 0)
                )
                
                draw.text(
                    (container_x, container_y),
                    container_text,
                    fill=(255, 255, 255),
                    font=font
                )
        
        return np.array(pil_image)
    
    def _display_detection_metadata(
        self,
        detections: List[Dict],
        min_confidence: float = 0.5
    ) -> None:
        """Display detection metadata in sidebar."""
        st.subheader("Detection Details")
        
        # Filter detections by confidence
        filtered_detections = [
            d for d in detections if d.get('confidence', 0) >= min_confidence
        ]
        
        if not filtered_detections:
            st.info("No detections above confidence threshold")
            return
        
        # Summary statistics
        st.metric("Total Detections", len(filtered_detections))
        
        # Class distribution
        class_counts = {}
        for detection in filtered_detections:
            class_name = detection.get('class', 'unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            st.subheader("Object Types")
            for class_name, count in sorted(class_counts.items()):
                st.write(f"**{class_name.capitalize()}**: {count}")
        
        # Confidence statistics
        confidences = [d.get('confidence', 0) for d in filtered_detections]
        if confidences:
            st.subheader("Confidence Stats")
            st.write(f"**Average**: {np.mean(confidences):.3f}")
            st.write(f"**Max**: {np.max(confidences):.3f}")
            st.write(f"**Min**: {np.min(confidences):.3f}")
        
        # Detailed detection list
        with st.expander("Detailed Detections"):
            for i, detection in enumerate(filtered_detections):
                st.write(f"**Detection {i + 1}**")
                st.json(detection)
    
    def create_detection_filter_controls(
        self,
        detections: List[Dict],
        key_prefix: str = "filter"
    ) -> Dict[str, Any]:
        """
        Create filter controls for detections.
        
        Args:
            detections: List of detection dictionaries
            key_prefix: Prefix for widget keys
            
        Returns:
            Dictionary with filter settings
        """
        st.subheader("Detection Filters")
        
        # Confidence threshold
        confidences = [d.get('confidence', 0) for d in detections if 'confidence' in d]
        if confidences:
            min_conf = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"{key_prefix}_confidence"
            )
        else:
            min_conf = 0.5
        
        # Class filter
        all_classes = list(set(d.get('class', 'unknown') for d in detections))
        selected_classes = st.multiselect(
            "Object Types",
            options=all_classes,
            default=all_classes,
            key=f"{key_prefix}_classes"
        )
        
        # Display options
        show_confidence = st.checkbox(
            "Show Confidence Scores",
            value=True,
            key=f"{key_prefix}_show_conf"
        )
        
        show_labels = st.checkbox(
            "Show Labels",
            value=True,
            key=f"{key_prefix}_show_labels"
        )
        
        return {
            'min_confidence': min_conf,
            'selected_classes': selected_classes,
            'show_confidence': show_confidence,
            'show_labels': show_labels
        }
    
    def export_annotated_image(
        self,
        image_path: str,
        detections: List[Dict],
        output_path: str,
        **kwargs
    ) -> bool:
        """
        Export annotated image to file.
        
        Args:
            image_path: Source image path
            detections: Detection data
            output_path: Output file path
            **kwargs: Additional annotation options
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image = self._load_image(image_path)
            annotated = self._annotate_image(image, detections, **kwargs)
            
            # Convert RGB to BGR for OpenCV
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(output_path, annotated_bgr)
            return True
            
        except Exception as e:
            st.error(f"Failed to export image: {str(e)}")
            return False


# Utility functions
def load_detection_data(file_path: str) -> List[Dict]:
    """
    Load detection data from JSON file.
    
    Args:
        file_path: Path to JSON file with detection data
        
    Returns:
        List of detection dictionaries
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load detection data: {str(e)}")
        return []


def save_detection_data(detections: List[Dict], file_path: str) -> bool:
    """
    Save detection data to JSON file.
    
    Args:
        detections: List of detection dictionaries
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(detections, f, indent=2, default=str)
        return True
    except Exception as e:
        st.error(f"Failed to save detection data: {str(e)}")
        return False


def create_detection_summary(detections: List[Dict]) -> Dict[str, Any]:
    """
    Create summary statistics from detections.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary with summary statistics
    """
    if not detections:
        return {}
    
    # Count by class
    class_counts = {}
    confidences = []
    
    for detection in detections:
        class_name = detection.get('class', 'unknown')
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if 'confidence' in detection:
            confidences.append(detection['confidence'])
    
    summary = {
        'total_detections': len(detections),
        'class_distribution': class_counts,
        'unique_classes': len(class_counts)
    }
    
    if confidences:
        summary['confidence_stats'] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
    
    return summary