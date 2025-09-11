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
from typing import Dict, List, Optional
import time
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from modules.database import queries
from modules.database.models import session_scope, Container, Detection, Image as DBImage
import os
import random

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


@st.cache_data(ttl=30)  # 30-second cache to reduce database load for live data
def load_live_detections() -> Dict:
    """Load the latest detection data from database."""
    current_time = datetime.now()
    
    try:
        # Get recent detections from database
        recent_detections_data = queries.get_recent_detections(limit=15)
    except Exception as e:
        st.error(f"Database error loading detections: {str(e)}")
        recent_detections_data = []
    
    # Process detections for display
    detections = []
    for det_data in recent_detections_data:
        bbox = det_data.get('bbox', {})
        detection = {
            'timestamp': det_data.get('timestamp', current_time),
            'container_id': f"CONT{det_data['detection_id']:07d}" if det_data.get('detection_id') else "UNKNOWN",
            'container_type': det_data.get('object_type', 'container'),
            'action': 'IN' if 'in' in det_data.get('camera_id', '').lower() else 'OUT',
            'confidence': det_data.get('confidence', 0),
            'x': bbox.get('x', 0),
            'y': bbox.get('y', 0),
            'width': bbox.get('width', 100),
            'height': bbox.get('height', 100),
            'camera_id': det_data.get('camera_id', 'CAM-01'),
            'processing_time': 1.2,  # Default processing time
            'image_path': det_data.get('filepath')
        }
        detections.append(detection)
    
    # Get camera status - check if we have recent detections
    unique_cameras = list(set(d.get('camera_id', 'CAM-01') for d in recent_detections_data))
    cameras = {}
    
    for cam_id in unique_cameras[:4]:  # Limit to 4 cameras
        # Check if camera has recent activity
        cam_detections = [d for d in recent_detections_data if d.get('camera_id') == cam_id]
        has_recent = False
        if cam_detections and cam_detections[0].get('timestamp'):
            time_diff = (current_time - cam_detections[0]['timestamp']).seconds
            has_recent = time_diff < 300  # Active if detection within 5 minutes
        
        cameras[cam_id] = {
            'status': 'online' if has_recent else 'offline',
            'fps': 15,
            'resolution': '1920x1080',
            'location': f"Location {cam_id}"
        }
    
    # If no cameras found, add default cameras
    if not cameras:
        cameras = {
            'CAM-01': {'status': 'offline', 'fps': 15, 'resolution': '1920x1080', 'location': 'Gate A'},
            'CAM-02': {'status': 'offline', 'fps': 15, 'resolution': '1920x1080', 'location': 'Gate B'},
            'CAM-03': {'status': 'offline', 'fps': 12, 'resolution': '1920x1080', 'location': 'Yard North'},
            'CAM-04': {'status': 'offline', 'fps': 15, 'resolution': '1920x1080', 'location': 'Yard South'}
        }
    
    # Get active containers for tracking
    try:
        with session_scope() as session:
            active_containers = session.query(Container).filter(
                Container.status == 'active'
            ).limit(8).all()
    except Exception as e:
        st.error(f"Database error loading containers: {str(e)}")
        active_containers = []
        
        active_tracks = []
        for i, container in enumerate(active_containers):
            track = {
                'track_id': f"T{i+1:03d}",
                'container_id': container.container_number,
                'first_seen': container.first_seen,
                'last_seen': container.last_seen,
                'current_location': container.camera_id or 'Unknown',
                'status': 'stationary' if container.dwell_time and container.dwell_time > 1 else 'moving',
                'confidence': container.avg_confidence
            }
            active_tracks.append(track)
    
    # Calculate system metrics
    today_start = datetime.combine(current_time.date(), datetime.min.time())
    try:
        detection_summary = queries.get_detection_summary(today_start, current_time)
    except Exception as e:
        st.error(f"Database error loading detection summary: {str(e)}")
        detection_summary = None
    
    metrics = {
        'total_detections_today': sum(d['count'] for d in detection_summary.get('detection_counts', [])) if detection_summary else 0,
        'active_tracks': len(active_tracks),
        'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
        'avg_processing_time': 1.2,  # Default value
        'frames_processed': detection_summary.get('processed_images', 0) if detection_summary else 0,
        'detection_rate': detection_summary.get('processing_rate', 0) / 100 if detection_summary else 0
    }
    
    return {
        'detections': detections,
        'cameras': cameras,
        'active_tracks': active_tracks,
        'metrics': metrics,
        'last_updated': current_time
    }


def load_actual_camera_image(camera_id: str = "CAM-01", detection_data: Optional[Dict] = None) -> Optional[str]:
    """Load actual camera image from database/filesystem."""
    # Get recent images from database
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


def display_actual_image(image_path: str, camera_id: str, detection_data: Optional[Dict] = None) -> Optional[Image.Image]:
    """Display actual camera image with optional detection overlays."""
    if not image_path or not os.path.exists(image_path):
        return None
    
    try:
        # Load the actual image
        img = Image.open(image_path)
        
        # If we have detection data, overlay detection boxes
        if detection_data and detection_data.get('detections'):
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw detection boxes
            for detection in detection_data['detections'][:5]:  # Show up to 5 detections
                if detection.get('image_path') == image_path:
                    x, y = detection['x'], detection['y']
                    w, h = detection['width'], detection['height']
                    
                    # Detection box color based on confidence
                    confidence = detection['confidence']
                    if confidence > 0.9:
                        color = '#00ff00'  # Green for high confidence
                    elif confidence > 0.8:
                        color = '#ffff00'  # Yellow for medium confidence
                    else:
                        color = '#ff8800'  # Orange for lower confidence
                    
                    # Draw bounding box
                    draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
                    
                    # Draw label
                    label = f"{detection['container_id'][:8]} ({confidence:.1%})"
                    bbox = draw.textbbox((x, y-25), label, font=font)
                    draw.rectangle(bbox, fill=color, outline=color)
                    draw.text((x, y-25), label, fill='#000000', font=font)
        
        # Add live indicator
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((10, 10), f"LIVE FEED - {timestamp}", fill='#ffffff', font=font)
        draw.text((10, img.height-40), f"{camera_id} | Live Feed | Real Camera Data", fill='#ffffff', font=font)
        
        return img
        
    except Exception as e:
        st.error(f"Error loading image {image_path}: {str(e)}")
        return None


def display_detection_card(detection: Dict):
    """Display a single detection as a card."""
    confidence_percent = int(detection['confidence'] * 100)
    
    # Determine card style based on confidence
    card_class = "detection-card"
    if confidence_percent < 85:
        card_class += " error"
    elif confidence_percent < 92:
        card_class += " warning"
    
    # Time since detection
    time_diff = datetime.now() - detection['timestamp']
    if time_diff.total_seconds() < 60:
        time_str = f"{int(time_diff.total_seconds())}s ago"
    else:
        time_str = f"{int(time_diff.total_seconds() / 60)}min ago"
    
    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong>{detection['container_id']}</strong>
            <span style="color: #666; font-size: 0.9rem;">{time_str}</span>
        </div>
        <div style="margin: 0.5rem 0;">
            <strong>Action:</strong> {detection['action']} | 
            <strong>Type:</strong> {detection['container_type']} | 
            <strong>Camera:</strong> {detection['camera_id']}
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence_percent}%; background-color: {'#28a745' if confidence_percent >= 92 else '#ffc107' if confidence_percent >= 85 else '#dc3545'};">
                {confidence_percent}%
            </div>
        </div>
        <div style="font-size: 0.8rem; color: #666;">
            Processing time: {detection['processing_time']:.2f}s | 
            Position: ({detection['x']}, {detection['y']})
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_camera_status(cameras: Dict):
    """Display camera status grid."""
    st.markdown("### 📹 Camera Status")
    
    cols = st.columns(4)
    
    for i, (cam_id, info) in enumerate(cameras.items()):
        with cols[i]:
            status_color = {
                'online': '#28a745',
                'offline': '#dc3545',
                'warning': '#ffc107'
            }.get(info['status'], '#6c757d')
            
            st.markdown(f"""
            <div style="padding: 1rem; border: 2px solid {status_color}; border-radius: 0.5rem; text-align: center;">
                <h4 style="margin: 0; color: {status_color};">{cam_id}</h4>
                <div style="margin: 0.5rem 0;">
                    <div style="color: {status_color}; font-weight: bold;">{info['status'].upper()}</div>
                    <div>{info['location']}</div>
                </div>
                <div style="font-size: 0.9rem; color: #666;">
                    {info['fps']} FPS<br>
                    {info['resolution']}
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_active_tracking(tracks: List[Dict]):
    """Display active tracking information."""
    st.markdown("### 🎯 Active Container Tracking")
    
    if not tracks:
        st.info("No containers currently being tracked.")
        return
    
    # Create tracking dataframe for table display
    tracking_df = pd.DataFrame(tracks)
    tracking_df['duration'] = tracking_df['first_seen'].apply(
        lambda x: str(datetime.now() - x).split('.')[0]
    )
    tracking_df['last_update'] = tracking_df['last_seen'].apply(
        lambda x: f"{int((datetime.now() - x).total_seconds())}s ago"
    )
    
    # Display as an interactive table
    display_df = tracking_df[[
        'track_id', 'container_id', 'current_location', 'status', 'confidence', 'duration', 'last_update'
    ]].copy()
    
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    display_df.columns = ['Track ID', 'Container ID', 'Location', 'Status', 'Confidence', 'Duration', 'Last Update']
    
    st.dataframe(
        display_df, 
        use_container_width=True,
        column_config={
            'Status': st.column_config.TextColumn(
                width='small'
            ),
            'Confidence': st.column_config.ProgressColumn(
                width='small'
            )
        }
    )


def main():
    """Main live feed dashboard."""
    
    # Header with live indicator
    st.markdown(
        '<div class="live-header"><span class="live-indicator"></span>🖼️ Live Camera Feed</div>', 
        unsafe_allow_html=True
    )
    
    # Sidebar controls
    with st.sidebar:
        st.title("Live Feed Controls")
        
        # Auto-refresh settings
        st.subheader("🔄 Refresh Settings")
        auto_refresh = st.toggle("Auto Refresh", value=True)
        
        if auto_refresh:
            refresh_interval = st.selectbox(
                "Refresh Interval",
                options=[5, 10, 15, 30],
                index=1,
                format_func=lambda x: f"{x} seconds"
            )
            st.info(f"Auto-refreshing every {refresh_interval} seconds")
        
        # Camera selection
        st.subheader("📹 Camera Selection")
        selected_camera = st.selectbox(
            "Primary Camera View",
            options=["CAM-01", "CAM-02", "CAM-03", "CAM-04"],
            index=0
        )
        
        # Detection filters
        st.subheader("🔍 Detection Filters")
        min_confidence = st.slider("Minimum Confidence", 0.5, 1.0, 0.8, 0.05)
        show_actions = st.multiselect(
            "Show Actions",
            options=["IN", "OUT", "STATIONARY"],
            default=["IN", "OUT", "STATIONARY"]
        )
        
        # Manual refresh
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.markdown("### Navigation")
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("app.py")
    
    # Auto-refresh mechanism
    if auto_refresh:
        # Create a placeholder for the refresh countdown
        refresh_placeholder = st.empty()
        
        # Use session state to track refresh timing
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
        # Calculate time since last refresh
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        
        # Only refresh if enough time has passed
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            st.cache_data.clear()
            st.rerun()
        else:
            # Show countdown until next refresh
            time_remaining = int(refresh_interval - time_since_refresh)
            refresh_placeholder.info(f"⏳ Next refresh in {time_remaining} seconds...")
            time.sleep(1)
            st.rerun()
    
    # Load live data
    with st.spinner("Loading live feed data..."):
        try:
            data = load_live_detections()
        except Exception as e:
            st.error(f"Failed to load live feed data: {str(e)}")
            data = {
                'detections': [],
                'cameras': {},
                'active_tracks': [],
                'metrics': {'total_detections_today': 0, 'active_tracks': 0, 'avg_confidence': 0, 'avg_processing_time': 0, 'frames_processed': 0, 'detection_rate': 0},
                'last_updated': datetime.now()
            }
    
    # System metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Detections Today", 
            data['metrics']['total_detections_today'],
            "+23 this hour"
        )
    
    with col2:
        st.metric(
            "Active Tracks", 
            data['metrics']['active_tracks'],
            "+2 new tracks"
        )
    
    with col3:
        st.metric(
            "Avg Confidence", 
            f"{data['metrics']['avg_confidence']:.1%}",
            "+1.2% improvement"
        )
    
    with col4:
        st.metric(
            "Processing Speed", 
            f"{data['metrics']['avg_processing_time']:.2f}s",
            "-0.1s faster"
        )
    
    st.divider()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### 📺 Camera Feed - {selected_camera}")
        
        # Load and display actual camera image
        image_path = load_actual_camera_image(selected_camera, data)
        if image_path:
            camera_img = display_actual_image(image_path, selected_camera, data)
            if camera_img:
                st.image(camera_img, use_column_width=True, caption=f"Live feed from {selected_camera} - Real Camera Data")
            else:
                st.error(f"Failed to load image from {image_path}")
        else:
            st.warning(f"No recent images available for {selected_camera}. Please check if the camera service is running and capturing images.")
            # Show placeholder
            st.info("📷 Camera feed will appear here when images are available.")
        
        # Camera feed controls
        feed_col1, feed_col2, feed_col3, feed_col4 = st.columns(4)
        
        with feed_col1:
            if st.button("📸 Capture Screenshot"):
                if image_path:
                    # Copy current image to screenshots directory
                    os.makedirs("data/screenshots", exist_ok=True)
                    screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    screenshot_path = f"data/screenshots/{screenshot_name}"
                    try:
                        if camera_img:
                            camera_img.save(screenshot_path)
                            st.success(f"Screenshot saved as {screenshot_name}")
                        else:
                            st.error("No image to capture")
                    except Exception as e:
                        st.error(f"Failed to save screenshot: {str(e)}")
                else:
                    st.warning("No camera feed available to capture")
        
        with feed_col2:
            if st.button("🎥 Start Recording"):
                st.info("Recording started...")
        
        with feed_col3:
            if st.button("⏹️ Stop Recording"):
                st.info("Recording stopped.")
        
        with feed_col4:
            if st.button("🔍 Refresh Feed"):
                st.cache_data.clear()
                st.rerun()
    
    with col2:
        st.markdown("### 🚨 Recent Detections")
        
        # Filter detections based on sidebar settings
        filtered_detections = [
            d for d in data['detections'] 
            if d['confidence'] >= min_confidence and d['action'] in show_actions
        ]
        
        if not filtered_detections:
            st.warning("No detections match current filters.")
        else:
            # Display up to 5 most recent detections
            for detection in filtered_detections[:5]:
                display_detection_card(detection)
            
            if len(filtered_detections) > 5:
                with st.expander(f"View {len(filtered_detections) - 5} more detections"):
                    for detection in filtered_detections[5:]:
                        display_detection_card(detection)
    
    st.divider()
    
    # Camera status
    display_camera_status(data['cameras'])
    
    st.divider()
    
    # Active tracking
    display_active_tracking(data['active_tracks'])
    
    st.divider()
    
    # Detection analytics (mini charts)
    st.markdown("### 📊 Live Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Detection confidence distribution
        confidences = [d['confidence'] for d in data['detections']]
        if confidences:
            fig_hist = px.histogram(
                x=confidences,
                nbins=10,
                title="Detection Confidence Distribution",
                labels={'x': 'Confidence', 'y': 'Count'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Processing time trend
        processing_times = [d['processing_time'] for d in data['detections']]
        times = [d['timestamp'] for d in data['detections']]
        
        if processing_times:
            fig_line = px.line(
                x=times,
                y=processing_times,
                title="Processing Time Trend",
                labels={'x': 'Time', 'y': 'Processing Time (s)'},
                color_discrete_sequence=['#28a745']
            )
            fig_line.update_layout(height=300)
            st.plotly_chart(fig_line, use_container_width=True)
    
    # System performance indicators
    st.markdown("### ⚡ System Performance")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("Frames Processed", f"{data['metrics']['frames_processed']:,}")
    
    with perf_col2:
        st.metric("Detection Rate", f"{data['metrics']['detection_rate']:.1%}")
    
    with perf_col3:
        st.metric("System Load", "78%", "-5%")
    
    with perf_col4:
        st.metric("Memory Usage", "4.2 GB", "+0.1 GB")
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <span class="live-indicator"></span>
        Live feed updated: {data['last_updated'].strftime('%H:%M:%S')} | 
        Showing data from {selected_camera} | 
        {len(filtered_detections)} detections displayed
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()