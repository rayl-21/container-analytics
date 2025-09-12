"""
Container Analytics - Simplified Live Feed

This page shows a clean 7-day image gallery (2025-09-01 to 2025-09-07)
with optional detection overlays and truck count badges.
"""

import streamlit as st

# Configure page - MUST be the first Streamlit command
st.set_page_config(
    page_title="Live Feed - Container Analytics",
    page_icon="🖼️",
    layout="wide"
)

import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from modules.database import queries
from modules.database.models import session_scope, Container, Detection, Image as DBImage
from sqlalchemy import and_, desc

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
            "CAM-02": "in_gate",
            "CAM-03": "in_gate",
            "CAM-04": "in_gate"
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

def load_7day_gallery(selected_date: Optional[date] = None) -> List[Dict[str, Any]]:
    """
    Load images for a specific date or date range for gallery display.
    
    Args:
        selected_date: Specific date to load images for. If None, loads all 7 days.
    
    Returns:
        List of image data dictionaries with detections
    """
    from datetime import date, timedelta
    
    # Define the fixed date range (2025-09-01 to 2025-09-07)
    if selected_date:
        # Load images for specific date
        start_date = datetime.combine(selected_date, datetime.min.time())
        end_date = datetime.combine(selected_date, datetime.max.time())
    else:
        # Load all 7 days
        start_date = datetime(2025, 9, 1, 0, 0, 0)
        end_date = datetime(2025, 9, 7, 23, 59, 59)
    
    gallery_images = []
    
    try:
        with session_scope() as session:
            # Query images in date range, sorted by timestamp
            images = session.query(DBImage).filter(
                and_(
                    DBImage.timestamp >= start_date,
                    DBImage.timestamp <= end_date
                )
            ).order_by(DBImage.timestamp).all()
            
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
                # Use filepath or image_path, whichever is available
                image_filepath = img.filepath or img.image_path
                img_data = {
                    'id': img.id,
                    'filepath': image_filepath,
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
    """Live Feed dashboard with date-based pagination for Sept 1-7, 2025."""
    
    # Header
    st.markdown("# 🖼️ Live Feed")
    st.markdown("*Image gallery with detection overlays - September 1-7, 2025*")
    
    # Check if navigation button was clicked
    if 'selected_nav_date' in st.session_state:
        nav_date = st.session_state.selected_nav_date
        del st.session_state.selected_nav_date  # Clear after using
    else:
        nav_date = None
    
    # Date selector for pagination
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Date selector - limited to Sept 1-7, 2025
        available_dates = [date(2025, 9, d) for d in range(1, 8)]
        date_options = {d: d.strftime('%A, %B %d, %Y') for d in available_dates}
        
        # If navigation date was selected, use it, otherwise default to Sept 1
        if nav_date and nav_date in available_dates:
            default_index = available_dates.index(nav_date)
        else:
            default_index = 0
        
        selected_date = st.selectbox(
            "📅 Select Date",
            options=available_dates,
            format_func=lambda x: date_options[x],
            index=default_index,
            key="date_selector"
        )
    
    with col2:
        # Quick navigation buttons
        st.markdown("##### Quick Navigation")
        button_cols = st.columns(7)
        for i, (d, btn_col) in enumerate(zip(available_dates, button_cols)):
            with btn_col:
                if st.button(f"{d.day}", use_container_width=True, 
                           type="primary" if d == selected_date else "secondary",
                           key=f"nav_btn_{d.day}"):
                    st.session_state.selected_nav_date = d
                    st.rerun()
    
    with col3:
        # Show all toggle
        show_all = st.toggle("Show All Days", value=False, 
                           help="Display all images from Sept 1-7 (1000+ images)")
    
    # Sidebar - Simplified with only essential controls
    with st.sidebar:
        # Data Sync toggle at top
        pull_new_data = st.toggle("📡 Pull New Data", value=False, help="Enable to refresh data from cameras")
        
        # Status display
        st.subheader("📊 Status")
        status = "🟡 Paused"
        last_update = "Manual mode"
        st.write(f"Status: {status}")
        st.write(f"Last Update: {last_update}")
        
        st.divider()
        
        # Display Options
        st.subheader("🎯 Display Options")
        show_detections = st.toggle("Show Detection Overlays", value=True)
        
        # Images per row
        images_per_row = st.slider(
            "Images per row",
            min_value=2,
            max_value=5,
            value=3
        )
        
        st.divider()
        
        # Refresh button
        if st.button("🔄 Refresh Gallery", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Statistics section
        st.subheader("📈 Statistics")
    
    # Load image data based on selection
    with st.spinner("Loading images..."):
        try:
            if show_all:
                gallery_data = load_7day_gallery(None)  # Load all 7 days
                date_display = "September 1-7, 2025"
            else:
                gallery_data = load_7day_gallery(selected_date)  # Load specific date
                date_display = selected_date.strftime('%B %d, %Y')
        except Exception as e:
            st.error(f"Failed to load gallery data: {str(e)}")
            gallery_data = []
    
    # Display image count and date
    if gallery_data:
        st.markdown(f"### 📸 {date_display}")
        
        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", len(gallery_data))
        
        with col2:
            total_detections = sum(len(img.get('detections', [])) for img in gallery_data)
            st.metric("Total Detections", total_detections)
        
        with col3:
            # Count containers/trucks
            total_containers = sum(
                len([d for d in img.get('detections', []) 
                     if d.get('object_type', '').lower() in ['truck', 'container']])
                for img in gallery_data
            )
            st.metric("Containers/Trucks", total_containers)
        
        with col4:
            # Images per hour
            if gallery_data:
                hours = len(set(img['timestamp'].hour for img in gallery_data))
                st.metric("Hours Covered", hours)
        
        st.divider()
        
        # Image gallery
        st.subheader("Image Gallery")
        
        # Progress bar for loading many images
        if len(gallery_data) > 50:
            progress_bar = st.progress(0)
            progress_text = st.empty()
        
        # Display images in grid
        total_images = len(gallery_data)
        
        for i in range(0, total_images, images_per_row):
            cols = st.columns(images_per_row)
            
            for j, col in enumerate(cols):
                idx = i + j
                if idx < total_images:
                    img_data = gallery_data[idx]
                    
                    with col:
                        display_image_card(img_data, show_detections)
                    
                    # Update progress if showing many images
                    if len(gallery_data) > 50:
                        progress = (idx + 1) / total_images
                        progress_bar.progress(progress)
                        progress_text.text(f"Loading: {idx + 1}/{total_images} images")
        
        # Clear progress indicators
        if len(gallery_data) > 50:
            progress_bar.empty()
            progress_text.empty()
        
        # Pagination controls at bottom
        if not show_all:
            st.divider()
            
            nav_cols = st.columns([1, 3, 1])
            
            with nav_cols[0]:
                # Previous day button
                current_idx = available_dates.index(selected_date)
                if current_idx > 0:
                    if st.button("⬅️ Previous Day", use_container_width=True):
                        st.session_state.selected_nav_date = available_dates[current_idx - 1]
                        st.rerun()
            
            with nav_cols[1]:
                st.markdown(f"<center>Day {current_idx + 1} of 7</center>", unsafe_allow_html=True)
            
            with nav_cols[2]:
                # Next day button
                if current_idx < len(available_dates) - 1:
                    if st.button("Next Day ➡️", use_container_width=True):
                        st.session_state.selected_nav_date = available_dates[current_idx + 1]
                        st.rerun()
    
    else:
        st.info(f"📷 No images available for {date_display}")
        st.markdown("""
        The image gallery will show camera data when available. 
        
        **Tips:**
        - Enable "Pull New Data" to sync with cameras
        - Check that the camera service is running
        - Verify images exist in the database for this date range
        - Images should be in data/images/2025-09-01 through 2025-09-07
        """)
    
    # Update sidebar statistics
    if gallery_data:
        with st.sidebar:
            st.metric("Current View", f"{len(gallery_data)} images")
            if not show_all:
                st.caption(f"Date: {selected_date.strftime('%b %d, %Y')}")


if __name__ == "__main__":
    main()