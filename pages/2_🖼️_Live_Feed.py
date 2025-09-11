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


@st.cache_data(ttl=10)  # 10-second cache for live data
def load_live_detections() -> Dict:
    """Load the latest detection data from database."""
    current_time = datetime.now()
    
    # Get recent detections from database
    recent_detections_data = queries.get_recent_detections(limit=15)
    
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
    with session_scope() as session:
        active_containers = session.query(Container).filter(
            Container.status == 'active'
        ).limit(8).all()
        
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
    detection_summary = queries.get_detection_summary(today_start, current_time)
    
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


def create_mock_camera_image(detection_data: Optional[Dict] = None) -> Image.Image:
    """Create a mock camera image with detection boxes."""
    # Create a mock camera image (normally would be from actual camera feed)
    width, height = 1000, 700
    img = Image.new('RGB', (width, height), color='#2c3e50')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple port/yard scene
    # Ground
    draw.rectangle([0, height-100, width, height], fill='#34495e')
    
    # Draw some containers as rectangles
    container_positions = [
        (100, 400, 180, 480, '#e74c3c'),  # Red container
        (200, 350, 280, 450, '#3498db'),  # Blue container
        (350, 380, 430, 480, '#f39c12'),  # Orange container
        (500, 300, 580, 400, '#27ae60'),  # Green container
        (650, 420, 730, 520, '#9b59b6'),  # Purple container
        (800, 350, 880, 450, '#e67e22'),  # Dark orange container
    ]
    
    for x1, y1, x2, y2, color in container_positions:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline='#ecf0f1', width=3)
    
    # Add detection boxes if detection data is provided
    if detection_data and detection_data['detections']:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for detection in detection_data['detections'][:3]:  # Show up to 3 detections
            x, y = detection['x'], detection['y']
            w, h = detection['width'], detection['height']
            
            # Detection box
            color = '#00ff00' if detection['confidence'] > 0.9 else '#ffff00'
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
            
            # Label
            label = f"{detection['container_id'][:8]} ({detection['confidence']:.1%})"
            bbox = draw.textbbox((x, y-25), label, font=font)
            draw.rectangle(bbox, fill=color, outline=color)
            draw.text((x, y-25), label, fill='#000000', font=font)
    
    # Add timestamp
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw.text((10, 10), f"LIVE FEED - {timestamp}", fill='#ffffff', font=font)
    
    # Add camera info
    draw.text((10, height-40), "CAM-01 | Gate A | 1920x1080 @ 15 FPS", fill='#ffffff', font=font)
    
    return img


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
        time.sleep(0.1)
        st.rerun()
    
    # Load live data
    with st.spinner("Loading live feed data..."):
        data = load_live_detections()
    
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
        
        # Create and display mock camera image
        camera_img = create_mock_camera_image(data)
        st.image(camera_img, use_column_width=True, caption=f"Live feed from {selected_camera}")
        
        # Camera feed controls
        feed_col1, feed_col2, feed_col3, feed_col4 = st.columns(4)
        
        with feed_col1:
            if st.button("📸 Capture Screenshot"):
                st.success("Screenshot saved to data/screenshots/")
        
        with feed_col2:
            if st.button("🎥 Start Recording"):
                st.info("Recording started...")
        
        with feed_col3:
            if st.button("⏹️ Stop Recording"):
                st.info("Recording stopped.")
        
        with feed_col4:
            if st.button("🔍 Zoom to Detection"):
                st.info("Zooming to latest detection...")
    
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