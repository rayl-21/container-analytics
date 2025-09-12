"""
Container Analytics - Live Camera Feed

This page shows the latest camera images with real-time detections,
confidence scores, and container tracking information.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from modules.database import queries
from modules.database.models import session_scope, Container, Detection, Image as DBImage
from modules.detection.yolo_detector import YOLODetector
from modules.downloader.selenium_client import DrayDogDownloader
import os
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

# Configure page
st.set_page_config(
    page_title="Live Feed - Container Analytics",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for live feed styling
st.markdown("""
<style>
.live-header {
    font-size: 2.2rem;
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 1rem;
}

.detection-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    margin: 0.5rem 0;
}

.detection-card.warning {
    border-left-color: #ffc107;
}

.detection-card.error {
    border-left-color: #dc3545;
}

.confidence-bar {
    background-color: #e9ecef;
    border-radius: 10px;
    overflow: hidden;
    height: 20px;
    margin: 0.25rem 0;
}

.confidence-fill {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.8rem;
    font-weight: bold;
}

.live-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    background-color: #28a745;
    border-radius: 50%;
    animation: blink 2s infinite;
    margin-right: 0.5rem;
}

@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0.3; }
}

.camera-feed {
    border: 2px solid #dee2e6;
    border-radius: 0.5rem;
    padding: 1rem;
    background-color: #fff;
}

.tracking-info {
    background-color: #e3f2fd;
    padding: 0.75rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


"""
Container Analytics - Simplified Live Feed

This page shows a clean 7-day image gallery (2025-09-01 to 2025-09-07)
with optional detection overlays and truck count badges.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from modules.database import queries
from modules.database.models import session_scope, Container, Detection, Image as DBImage
from sqlalchemy import and_, desc

# Configure page
st.set_page_config(
    page_title="Live Feed - Container Analytics",
    page_icon="🖼️",
    layout="wide"
)

# Simplified CSS for clean design
st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
}

.image-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 16px;
    background-color: #fafafa;
}

.metric-badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: bold;
    color: white;
}

.truck-badge {
    background-color: #28a745;
}

.truck-badge.zero {
    background-color: #6c757d;
}

.date-header {
    background-color: #f8f9fa;
    padding: 12px;
    border-radius: 6px;
    border-left: 4px solid #007bff;
    margin: 16px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)


# This function has been removed and replaced with load_7day_gallery()
pass


def load_actual_camera_image(camera_id: str = "CAM-01", detection_data: Optional[Dict] = None, images_list: Optional[List[Dict]] = None) -> Optional[str]:
    """Load actual camera image from fetched images or database."""
    # First try to use fetched images if available
    if images_list:
        # Map camera IDs to stream names
        stream_map = {
            "CAM-01": "in_gate",
            "CAM-02": "out_gate",
            "CAM-03": "in_gate",
            "CAM-04": "out_gate"
        }
        
        target_stream = stream_map.get(camera_id, "in_gate")
        
        # Find most recent image for the target stream
        for img in images_list:
            if img['stream'] == target_stream and os.path.exists(img['path']):
                return img['path']
    
    # Fall back to database
    recent_images = queries.get_recent_images(limit=1, camera_id=camera_id.lower().replace('cam-', '').replace('-', '_') + '_gate')
    
    if not recent_images:
        # Fall back to any recent image
        all_recent = queries.get_recent_images(limit=10)
        if all_recent:
            # Filter by camera type if possible
            gate_images = [img for img in all_recent if 'gate' in img['filepath'].lower()]
            if gate_images:
                return gate_images[0]['filepath']
            return all_recent[0]['filepath']
        return None
    
    return recent_images[0]['filepath']

def load_7day_gallery() -> List[Dict[str, Any]]:
    """
    Load images from the 7-day period (2025-09-01 to 2025-09-07) for gallery display.
    
    Returns:
        List of image data dictionaries with detections
    """
    from datetime import date
    
    # Define the 7-day date range
    start_date = datetime(2025, 9, 1)
    end_date = datetime(2025, 9, 7, 23, 59, 59)
    
    gallery_images = []
    
    try:
        with session_scope() as session:
            # Query images in date range, sorted by timestamp (newest first)
            images = session.query(DBImage).filter(
                and_(
                    DBImage.timestamp >= start_date,
                    DBImage.timestamp <= end_date
                )
            ).order_by(desc(DBImage.timestamp)).all()
            
            for img in images:
                # Get detections for this image
                detections = session.query(Detection).filter(
                    Detection.image_id == img.id
                ).all()
                
                # Format detections for display
                detection_data = []
                for det in detections:
                    det_dict = {
                        'id': det.id,
                        'object_type': det.object_type,
                        'confidence': det.confidence,
                        'bbox': det.bbox,
                        'tracking_id': det.tracking_id,
                        'container_number': getattr(det, 'container_number', None)
                    }
                    detection_data.append(det_dict)
                
                # Create image data dictionary
                img_data = {
                    'id': img.id,
                    'filepath': img.filepath,
                    'camera_id': img.camera_id,
                    'timestamp': img.timestamp,
                    'processed': img.processed,
                    'file_size': img.file_size,
                    'detections': detection_data
                }
                
                gallery_images.append(img_data)
    
    except Exception as e:
        st.error(f"Database error loading gallery: {str(e)}")
        return []
    
    return gallery_images


def display_image_card(img_data: Dict[str, Any], show_detections: bool = True):
    """
    Display a single image card with truck count badge and optional detections.
    
    Args:
        img_data: Image data dictionary
        show_detections: Whether to show detection overlays
    """
    if not os.path.exists(img_data['filepath']):
        st.warning(f"Image not found: {os.path.basename(img_data['filepath'])}")
        return
    
    try:
        # Load image
        img = Image.open(img_data['filepath'])
        
        # Calculate truck/container count
        detections = img_data.get('detections', [])
        truck_count = len([
            d for d in detections 
            if d.get('object_type', '').lower() in ['truck', 'container', 'vehicle']
        ])
        
        # Apply detection overlays if requested
        if show_detections and detections:
            img = add_detection_overlays(img, detections)
        
        # Display image with responsive sizing
        st.image(img, use_column_width=True)
        
        # Image metadata and truck count badge
        timestamp_str = img_data['timestamp'].strftime('%H:%M:%S')
        camera_name = img_data.get('camera_id', 'Unknown').upper()
        
        # Create badge-style display for truck count
        badge_color = "#28a745" if truck_count > 0 else "#6c757d"
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
            <div style="font-size: 0.8rem; color: #666;">
                🕒 {timestamp_str}<br>
                📹 {camera_name}
            </div>
            <div style="
                background-color: {badge_color}; 
                color: white; 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 0.7rem; 
                font-weight: bold;
            ">
                🚛 {truck_count}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Optional: Show detection details in expander
        if detections and st.button(f"Details", key=f"details_{img_data['id']}", use_container_width=True):
            with st.expander("Detection Details", expanded=True):
                for i, detection in enumerate(detections):
                    confidence = detection.get('confidence', 0)
                    obj_type = detection.get('object_type', 'Unknown')
                    st.write(f"**{i+1}.** {obj_type.title()} ({confidence:.1%})")
    
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")


def add_detection_overlays(img: Image.Image, detections: List[Dict]) -> Image.Image:
    """
    Add detection bounding boxes and labels to image.
    
    Args:
        img: PIL Image object
        detections: List of detection dictionaries
        
    Returns:
        Image with overlays applied
    """
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to load a font, fallback to default
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Color mapping for different object types
    colors = {
        'truck': '#FF6B6B',      # Red
        'container': '#4ECDC4',   # Teal
        'vehicle': '#45B7D1',     # Blue
        'person': '#FFA726',      # Orange
        'default': '#95E1D3'      # Light green
    }
    
    for detection in detections:
        try:
            # Extract bounding box
            bbox = detection.get('bbox', {})
            if isinstance(bbox, dict):
                x1 = bbox.get('x1', bbox.get('xmin', 0))
                y1 = bbox.get('y1', bbox.get('ymin', 0))  
                x2 = bbox.get('x2', bbox.get('xmax', img.width))
                y2 = bbox.get('y2', bbox.get('ymax', img.height))
            else:
                continue  # Skip invalid bbox format
            
            # Ensure coordinates are valid
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img.width, int(x2)), min(img.height, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes
            
            # Get object type and color
            obj_type = detection.get('object_type', 'unknown').lower()
            color = colors.get(obj_type, colors['default'])
            
            # Draw bounding box
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            
            # Prepare label
            confidence = detection.get('confidence', 0)
            label = f"{obj_type.title()} {confidence:.1%}"
            
            # Draw label background
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            label_y = max(0, y1 - text_height - 2)
            draw.rectangle(
                [(x1, label_y), (x1 + text_width, label_y + text_height)],
                fill=color
            )
            
            # Draw label text
            draw.text((x1, label_y), label, fill='white', font=font)
            
        except Exception as e:
            continue  # Skip problematic detections
    
    return img


# This function has been removed and replaced with display_image_card()
pass


# This function has been removed - detection info now shown in image cards
pass


# This function has been removed - camera status now shown in simplified sidebar
pass


# This function has been removed for simplified UI
pass


def main():
    """Simplified Live Feed dashboard with 7-day image gallery."""
    
    # Header
    st.markdown("# 🖼️ Live Feed")
    st.markdown("*Simple 7-day image gallery with detection overlays*")
    
    # Sidebar - Clean and minimal
    with st.sidebar:
        st.title("Feed Controls")
        
        # Pull New Data toggle
        st.subheader("📡 Data Sync")
        pull_new_data = st.toggle("Pull New Data", value=False, help="Enable to refresh data from cameras")
        
        # Status indicators
        st.subheader("📊 Status")
        
        # Mock status for now - in real implementation this would check actual system status
        if pull_new_data:
            status = "🟢 Active"
            last_update = datetime.now().strftime("%H:%M:%S")
        else:
            status = "🟡 Paused"
            last_update = "Manual mode"
        
        st.write(f"**Status:** {status}")
        st.write(f"**Last Update:** {last_update}")
        
        # Detection overlay toggle
        st.subheader("🎯 Display Options")
        show_detections = st.toggle("Show Detection Overlays", value=True)
        
        # Manual refresh
        if st.button("🔄 Refresh Gallery", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Navigation
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("app.py")
    
    # Load 7-day image data (2025-09-01 to 2025-09-07)
    with st.spinner("Loading image gallery..."):
        try:
            gallery_data = load_7day_gallery()
        except Exception as e:
            st.error(f"Failed to load gallery data: {str(e)}")
            gallery_data = []
    
    if not gallery_data:
        st.info("📷 No images available for the selected date range (2025-09-01 to 2025-09-07)")
        st.markdown("""
        The image gallery will show camera data when available. 
        
        **Tips:**
        - Enable "Pull New Data" to sync with cameras
        - Check that the camera service is running
        - Verify images exist in the data/images directory
        """)
        return
    
    # Display total image count and date range
    st.markdown(f"**Showing {len(gallery_data)} images** from September 1-7, 2025")
    
    # Image gallery grid - 3 columns for optimal viewing
    st.subheader("📸 Image Gallery")
    
    cols_per_row = 3
    
    # Group images by date for better organization
    grouped_images = {}
    for img_data in gallery_data:
        img_date = img_data['timestamp'].strftime('%Y-%m-%d')
        if img_date not in grouped_images:
            grouped_images[img_date] = []
        grouped_images[img_date].append(img_data)
    
    # Display images grouped by date
    for date_str in sorted(grouped_images.keys(), reverse=True):  # Most recent first
        date_images = grouped_images[date_str]
        
        # Date header
        st.markdown(f"### 📅 {date_str} ({len(date_images)} images)")
        
        # Create image grid for this date
        for i in range(0, len(date_images), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(date_images):
                    img_data = date_images[i + j]
                    
                    with col:
                        display_image_card(img_data, show_detections)
        
        st.divider()
    
    # Footer with summary stats
    total_detections = sum(len(img_data.get('detections', [])) for img_data in gallery_data)
    total_trucks = sum(
        len([d for d in img_data.get('detections', []) if d.get('object_type', '').lower() in ['truck', 'container']])
        for img_data in gallery_data
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", len(gallery_data))
    
    with col2:
        st.metric("Total Detections", total_detections)
    
    with col3:
        st.metric("Truck/Container Count", total_trucks)
    
    with col4:
        date_range = "Sep 1-7, 2025"
        st.metric("Date Range", date_range)


if __name__ == "__main__":
    main()