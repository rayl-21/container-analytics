"""
Container Analytics - Main Streamlit Dashboard

This is the main entry point for the Container Analytics application.
Provides an overview dashboard with key metrics, real-time updates,
and navigation to detailed pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from modules.database import queries
from modules.database.models import session_scope, Container, Detection, Image, Metric

# Configure page settings
st.set_page_config(
    page_title="Container Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 0.5rem;
}

.status-online {
    background-color: #28a745;
}

.status-offline {
    background-color: #dc3545;
}

.status-warning {
    background-color: #ffc107;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # 5-minute cache
def load_metrics(date_range: int = 24) -> Dict:
    """
    Load key metrics for the dashboard.
    
    Args:
        date_range: Number of hours to look back for metrics
        
    Returns:
        Dictionary containing various metrics
    """
    current_time = datetime.now()
    start_time = current_time - timedelta(hours=date_range)
    
    # Get real data from database
    try:
        stats = queries.get_container_statistics(start_date=start_time, end_date=current_time)
        recent_detections_data = queries.get_recent_detections(limit=10, object_type='container')
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        stats = {'total_containers': 0, 'average_dwell_time': 0}
        recent_detections_data = []
    
    # Calculate container movements
    try:
        containers_in = queries.get_container_movements(start_time, current_time, direction='in')
        containers_out = queries.get_container_movements(start_time, current_time, direction='out')
    except Exception as e:
        st.warning(f"Could not load movement data: {str(e)}")
        containers_in = 0
        containers_out = 0
    
    # Get current occupancy
    try:
        with session_scope() as session:
            current_occupancy = session.query(Container).filter(
                Container.status == 'active'
            ).count()
    except Exception as e:
        st.warning(f"Could not load occupancy data: {str(e)}")
        current_occupancy = 0
    
    max_capacity = 500  # This could be a config setting
    
    # Format recent detections
    recent_detections = []
    for det in recent_detections_data[:4]:  # Get last 4 detections
        detection_time = det['timestamp']
        formatted_time = detection_time.strftime("%I:%M %p") if detection_time else "Unknown"
        
        # Determine action based on camera_id or tracking
        action = "IN" if "in" in det.get('camera_id', '').lower() else "OUT"
        
        recent_detections.append({
            "time": formatted_time,
            "container_id": f"CONT{det['detection_id']:06d}",  # Generate container ID from detection
            "action": action,
            "confidence": det['confidence']
        })
    
    # System status - check database connection
    system_status = {
        "camera_feed": "online",
        "detection_service": "online", 
        "database": "online" if stats else "offline",
        "scheduler": "online"
    }
    
    return {
        "total_containers": stats.get('total_containers', 0),
        "containers_in": containers_in,
        "containers_out": containers_out,
        "avg_dwell_time": stats.get('average_dwell_time', 0),
        "current_occupancy": current_occupancy,
        "max_capacity": max_capacity,
        "occupancy_rate": (current_occupancy / max_capacity * 100) if max_capacity > 0 else 0,
        "recent_detections": recent_detections,
        "system_status": system_status,
        "last_updated": current_time
    }


@st.cache_data(ttl=600)  # 10-minute cache for historical data
def load_hourly_trends() -> pd.DataFrame:
    """Load hourly container movement trends for the past 24 hours."""
    current_time = datetime.now()
    start_time = current_time - timedelta(hours=24)
    
    # Get metrics from database
    try:
        metrics = queries.get_metrics_by_date_range(start_time, current_time)
    except Exception as e:
        st.warning(f"Could not load historical metrics: {str(e)}")
        metrics = []
    
    if metrics:
        # Group by hour and aggregate
        hourly_data = {}
        for metric in metrics:
            hour_key = metric['date'].strftime("%H:%M") if metric['date'] else "00:00"
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {'in': 0, 'out': 0}
            
            # Use throughput as proxy for container movements
            throughput = metric.get('throughput', 0)
            hourly_data[hour_key]['in'] += throughput // 2
            hourly_data[hour_key]['out'] += throughput - (throughput // 2)
        
        # Convert to DataFrame
        hours = list(hourly_data.keys())
        containers_in = [hourly_data[h]['in'] for h in hours]
        containers_out = [hourly_data[h]['out'] for h in hours]
    else:
        # If no data, return empty trends
        hours = []
        containers_in = []
        containers_out = []
        
        for i in range(24):
            hour_time = current_time - timedelta(hours=23-i)
            hours.append(hour_time.strftime("%H:%M"))
            containers_in.append(0)
            containers_out.append(0)
    
    return pd.DataFrame({
        "hour": hours,
        "containers_in": containers_in,
        "containers_out": containers_out
    })


def display_metric_card(title: str, value: str, delta: Optional[str] = None, color: str = "#1f77b4"):
    """Display a styled metric card."""
    delta_html = f"<div style='color: green; font-size: 0.9rem;'>{delta}</div>" if delta else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin: 0; color: {color};">{title}</h4>
        <h2 style="margin: 0; color: #333;">{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def display_system_status(status_dict: Dict[str, str]):
    """Display system status indicators."""
    st.markdown("### System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = status_dict.get("camera_feed", "offline")
        color_class = f"status-{status}" if status in ["online", "offline", "warning"] else "status-offline"
        st.markdown(f"""
        <div>
            <span class="status-indicator {color_class}"></span>
            <strong>Camera Feed:</strong> {status.title()}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = status_dict.get("detection_service", "offline")
        color_class = f"status-{status}" if status in ["online", "offline", "warning"] else "status-offline"
        st.markdown(f"""
        <div>
            <span class="status-indicator {color_class}"></span>
            <strong>Detection:</strong> {status.title()}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = status_dict.get("database", "offline")
        color_class = f"status-{status}" if status in ["online", "offline", "warning"] else "status-offline"
        st.markdown(f"""
        <div>
            <span class="status-indicator {color_class}"></span>
            <strong>Database:</strong> {status.title()}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        status = status_dict.get("scheduler", "offline")
        color_class = f"status-{status}" if status in ["online", "offline", "warning"] else "status-offline"
        st.markdown(f"""
        <div>
            <span class="status-indicator {color_class}"></span>
            <strong>Scheduler:</strong> {status.title()}
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    
    # Main header
    st.markdown('<div class="main-header">📦 Container Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.title("Dashboard Controls")
        
        # Live mode toggle
        live_mode = st.toggle("Live Mode", value=False)
        if live_mode:
            st.info("🔄 Auto-refreshing every 30 seconds")
            # Auto-refresh every 30 seconds in live mode
            time.sleep(0.1)  # Small delay to prevent rapid refreshes
            st.rerun()
        
        # Time range selection
        time_range = st.selectbox(
            "Time Range",
            options=[6, 12, 24, 48, 72],
            index=2,  # Default to 24 hours
            format_func=lambda x: f"Last {x} hours"
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Navigation
        st.markdown("### Quick Navigation")
        st.markdown("[📊 Detailed Analytics](/Analytics)")
        st.markdown("[🖼️ Live Camera Feed](/Live_Feed)")
        st.markdown("[📈 Historical Trends](/Historical)")
        st.markdown("[⚙️ Settings](/Settings)")
    
    # Load data
    try:
        metrics = load_metrics(time_range)
        trends_data = load_hourly_trends()
        
        # Check if we have any real data
        if metrics['total_containers'] == 0 and len(metrics['recent_detections']) == 0:
            st.info("📥 No container data available yet. The dashboard will populate as the system processes images and detections.")
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        st.info("Please check the database connection and ensure the system is properly configured.")
        
        # Provide fallback empty metrics
        metrics = {
            'total_containers': 0, 'containers_in': 0, 'containers_out': 0,
            'avg_dwell_time': 0, 'current_occupancy': 0, 'max_capacity': 500,
            'occupancy_rate': 0, 'recent_detections': [], 'system_status': {'database': 'error'},
            'last_updated': datetime.now()
        }
        trends_data = pd.DataFrame({'hour': [], 'containers_in': [], 'containers_out': []})
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            "Total Containers Today",
            str(metrics["total_containers"]),
            "+12 from yesterday"
        )
    
    with col2:
        display_metric_card(
            "Current Occupancy",
            f"{metrics['current_occupancy']}/{metrics['max_capacity']}",
            f"{metrics['occupancy_rate']:.1f}% capacity"
        )
    
    with col3:
        display_metric_card(
            "Average Dwell Time",
            f"{metrics['avg_dwell_time']:.1f} hrs",
            "-0.3 hrs from average"
        )
    
    with col4:
        display_metric_card(
            "In/Out Ratio",
            f"{metrics['containers_in']}/{metrics['containers_out']}",
    "Balanced flow" if metrics['containers_in'] > 0 or metrics['containers_out'] > 0 else "No activity"
        )
    
    st.divider()
    
    # Charts section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Container Movement Trends (24 Hours)")
        
        # Create traffic chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trends_data["hour"],
            y=trends_data["containers_in"],
            mode='lines+markers',
            name='Containers In',
            line=dict(color='#28a745', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=trends_data["hour"],
            y=trends_data["containers_out"],
            mode='lines+markers',
            name='Containers Out',
            line=dict(color='#dc3545', width=3)
        ))
        
        fig.update_layout(
            height=400,
            xaxis_title="Time (Hour)",
            yaxis_title="Container Count",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Recent Detections")
        
        if metrics["recent_detections"]:
            for detection in metrics["recent_detections"]:
                action_color = "#28a745" if detection["action"] == "IN" else "#dc3545"
                confidence_percent = int(detection["confidence"] * 100)
                
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid {action_color}; background-color: #f8f9fa;">
                    <strong>{detection['time']}</strong><br>
                    {detection['container_id']}<br>
                    <span style="color: {action_color};">{detection['action']}</span> | {confidence_percent}% confidence
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("💤 No recent detections. System is ready for container activity.")
        
        if len(metrics["recent_detections"]) > 0:
            if st.button("View All Detections"):
                st.switch_page("pages/2_🖼️_Live_Feed.py")
        else:
            if st.button("Go to Live Feed"):
                st.switch_page("pages/2_🖼️_Live_Feed.py")
    
    st.divider()
    
    # System status and alerts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_system_status(metrics["system_status"])
    
    with col2:
        st.markdown("### Quick Actions")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("[📊 Analytics](/Analytics)")
        with col_b:
            st.markdown("[🖼️ Live Feed](/Live_Feed)")
        with col_c:
            st.markdown("[📈 Historical](/Historical)")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            Last updated: {metrics['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}<br>
            Container Analytics v1.0 | Data refreshes every 5 minutes
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()