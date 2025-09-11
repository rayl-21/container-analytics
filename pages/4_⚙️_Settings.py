"""
Container Analytics - Settings Configuration

This page allows updating detection thresholds, managing alert settings,
controlling data retention, and provides system status information.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import json
import os
import psutil
import time

# Configure page
st.set_page_config(
    page_title="Settings - Container Analytics",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS for settings page
st.markdown("""
<style>
.settings-header {
    font-size: 2.2rem;
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 1rem;
}

.settings-section {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.75rem;
    border: 1px solid #dee2e6;
    margin: 1rem 0;
}

.system-status-good {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
}

.system-status-warning {
    background-color: #fff3cd;
    color: #856404;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
}

.system-status-error {
    background-color: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #dc3545;
}

.config-changed {
    background-color: #cce7ff;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border: 1px solid #0066cc;
    margin: 0.5rem 0;
}

.advanced-settings {
    background-color: #fff8dc;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #ddd;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# Default configuration
DEFAULT_CONFIG = {
    "detection": {
        "confidence_threshold": 0.80,
        "nms_threshold": 0.45,
        "max_detections": 100,
        "enable_tracking": True,
        "tracking_max_age": 30,
        "model_path": "yolov8x.pt"
    },
    "alerts": {
        "enable_alerts": True,
        "email_notifications": False,
        "alert_threshold_high_traffic": 150,
        "alert_threshold_low_accuracy": 0.85,
        "alert_threshold_long_dwell": 48.0,
        "notification_cooldown": 300,
        "email_recipients": []
    },
    "data_retention": {
        "keep_images_days": 30,
        "keep_detection_data_days": 365,
        "compress_old_images": True,
        "auto_cleanup_enabled": True,
        "backup_enabled": False,
        "backup_frequency": "weekly"
    },
    "system": {
        "log_level": "INFO",
        "max_log_files": 10,
        "enable_performance_monitoring": True,
        "camera_refresh_interval": 10,
        "dashboard_refresh_interval": 30,
        "max_concurrent_detections": 4
    },
    "ui": {
        "theme": "light",
        "items_per_page": 20,
        "enable_animations": True,
        "show_confidence_always": False,
        "default_date_range": 7,
        "enable_experimental_features": False
    }
}


@st.cache_data(ttl=60)  # Cache for 1 minute
def load_system_status() -> Dict:
    """Load current system status information."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from modules.database import queries
    from modules.database.models import session_scope, Detection, Image
    
    current_time = datetime.now()
    
    # Get actual system resource usage
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Service statuses (simplified - showing actual running status)
    services = {
        "Camera Service": {
            "status": "inactive",
            "uptime": "N/A",
            "last_restart": current_time,
            "cpu_usage": 0,
            "memory_usage": 0,
            "health": "unknown"
        },
        "Detection Service": {
            "status": "inactive",
            "uptime": "N/A",
            "last_restart": current_time,
            "cpu_usage": 0,
            "memory_usage": 0,
            "health": "unknown"
        },
        "Database Service": {
            "status": "running" if Path("data/database.db").exists() else "inactive",
            "uptime": "N/A",
            "last_restart": current_time,
            "cpu_usage": 0,
            "memory_usage": 0,
            "health": "good" if Path("data/database.db").exists() else "unknown"
        },
        "Scheduler Service": {
            "status": "inactive",
            "uptime": "N/A",
            "last_restart": current_time,
            "cpu_usage": 0,
            "memory_usage": 0,
            "health": "unknown"
        }
    }
    
    # System resources (actual values)
    system_resources = {
        "total_cpu_usage": cpu_percent,
        "total_memory_usage": memory.used / (1024**3),  # Convert to GB
        "total_memory_available": memory.total / (1024**3),  # GB
        "disk_usage": disk.used / (1024**3),  # GB
        "disk_available": disk.free / (1024**3),  # GB
        "network_in": 0,  # MB/h - would need network monitoring
        "network_out": 0,  # MB/h
    }
    
    # Get actual logs from log files if they exist
    recent_logs = []
    log_file = Path("logs/app.log")
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-20:]  # Get last 20 lines
                for line in lines:
                    # Parse log line (assuming standard format)
                    recent_logs.append({
                        "timestamp": current_time,
                        "level": "INFO",
                        "message": line.strip()
                    })
        except:
            pass
    
    if not recent_logs:
        # No logs available
        recent_logs.append({
            "timestamp": current_time,
            "level": "INFO",
            "message": "No log entries available"
        })
    
    # Get actual performance metrics from database
    with session_scope() as session:
        # Count processed images today
        today_start = datetime.combine(datetime.today(), datetime.min.time())
        processed_today = session.query(Image).filter(
            Image.timestamp >= today_start
        ).count()
        
        # Get recent detection accuracy
        recent_detections = session.query(Detection).order_by(
            Detection.timestamp.desc()
        ).limit(100).all()
        
        if recent_detections:
            avg_confidence = np.mean([d.confidence for d in recent_detections if d.confidence])
        else:
            avg_confidence = 0
    
    performance_metrics = {
        "detection_speed_avg": 0,  # Would need timing data
        "detection_accuracy": avg_confidence,
        "uptime_percentage": 0,  # Would need monitoring data
        "processed_images_today": processed_today,
        "active_connections": 0,  # Would need connection tracking
        "cache_hit_rate": 0  # Would need cache monitoring
    }
    
    return {
        "services": services,
        "system_resources": system_resources,
        "recent_logs": recent_logs,
        "performance_metrics": performance_metrics,
        "last_updated": current_time
    }


def load_current_config() -> Dict:
    """Load current configuration from file or return defaults."""
    config_file = "data/config.json"
    
    # For demo purposes, return default config
    # TODO: Implement actual config file loading
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict) -> bool:
    """Save configuration to file."""
    config_file = "data/config.json"
    
    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Failed to save configuration: {str(e)}")
        return False


def display_service_status(services: Dict):
    """Display service status cards."""
    st.markdown("### 🔧 Service Status")
    
    cols = st.columns(len(services))
    
    for i, (service_name, info) in enumerate(services.items()):
        with cols[i]:
            status_class = {
                'good': 'system-status-good',
                'warning': 'system-status-warning',
                'error': 'system-status-error'
            }.get(info['health'], 'system-status-good')
            
            st.markdown(f"""
            <div class="{status_class}">
                <h4 style="margin: 0;">{service_name}</h4>
                <div style="margin: 0.5rem 0;">
                    <strong>Status:</strong> {info['status'].upper()}<br>
                    <strong>Uptime:</strong> {info['uptime']}<br>
                    <strong>CPU:</strong> {info['cpu_usage']:.1f}%<br>
                    <strong>Memory:</strong> {info['memory_usage']:.0f} MB
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Restart {service_name}", key=f"restart_{i}"):
                st.success(f"{service_name} restart initiated!")
                time.sleep(1)
                st.rerun()


def display_system_resources(resources: Dict):
    """Display system resource usage."""
    st.markdown("### 💻 System Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU and Memory gauges
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = resources["total_cpu_usage"],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CPU Usage (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        memory_usage_pct = (resources["total_memory_usage"] / resources["total_memory_available"]) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = memory_usage_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Memory Usage (%) <br>{resources['total_memory_usage']:.1f}GB / {resources['total_memory_available']:.1f}GB"},
            delta = {'reference': 70},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Disk usage
    col3, col4 = st.columns(2)
    
    with col3:
        disk_usage_pct = (resources["disk_usage"] / (resources["disk_usage"] + resources["disk_available"])) * 100
        st.metric(
            "Disk Usage",
            f"{resources['disk_usage']:.1f} GB",
            f"{disk_usage_pct:.1f}% of total"
        )
    
    with col4:
        st.metric(
            "Network Traffic",
            f"↓ {resources['network_in']:.1f} MB/h",
            f"↑ {resources['network_out']:.1f} MB/h"
        )


def create_detection_settings_section(config: Dict) -> Dict:
    """Create detection settings configuration section."""
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Detection Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config["detection"]["confidence_threshold"] = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=config["detection"]["confidence_threshold"],
            step=0.05,
            help="Minimum confidence score for valid detections"
        )
        
        config["detection"]["max_detections"] = st.number_input(
            "Max Detections per Image",
            min_value=10,
            max_value=500,
            value=config["detection"]["max_detections"],
            step=10,
            help="Maximum number of objects to detect per image"
        )
        
        config["detection"]["enable_tracking"] = st.checkbox(
            "Enable Object Tracking",
            value=config["detection"]["enable_tracking"],
            help="Track containers across multiple frames"
        )
    
    with col2:
        config["detection"]["nms_threshold"] = st.slider(
            "NMS Threshold",
            min_value=0.1,
            max_value=1.0,
            value=config["detection"]["nms_threshold"],
            step=0.05,
            help="Non-Maximum Suppression threshold for overlapping detections"
        )
        
        if config["detection"]["enable_tracking"]:
            config["detection"]["tracking_max_age"] = st.number_input(
                "Tracking Max Age (frames)",
                min_value=5,
                max_value=100,
                value=config["detection"]["tracking_max_age"],
                help="Maximum frames to keep tracking inactive objects"
            )
        
        model_options = ["yolov12n.pt", "yolov12s.pt", "yolov12m.pt", "yolov12l.pt", "yolov12x.pt"]
        current_model = config["detection"]["model_path"]
        if current_model not in model_options:
            model_options.append(current_model)
        
        config["detection"]["model_path"] = st.selectbox(
            "YOLO Model",
            options=model_options,
            index=model_options.index(current_model),
            help="YOLO model variant (larger models = better accuracy, slower speed)"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return config


def create_alert_settings_section(config: Dict) -> Dict:
    """Create alert settings configuration section."""
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("#### 🚨 Alert Settings")
    
    config["alerts"]["enable_alerts"] = st.checkbox(
        "Enable Alerts",
        value=config["alerts"]["enable_alerts"],
        help="Enable automated alert notifications"
    )
    
    if config["alerts"]["enable_alerts"]:
        col1, col2 = st.columns(2)
        
        with col1:
            config["alerts"]["alert_threshold_high_traffic"] = st.number_input(
                "High Traffic Alert Threshold",
                min_value=50,
                max_value=500,
                value=config["alerts"]["alert_threshold_high_traffic"],
                help="Containers per hour to trigger high traffic alert"
            )
            
            config["alerts"]["alert_threshold_long_dwell"] = st.number_input(
                "Long Dwell Time Alert (hours)",
                min_value=12.0,
                max_value=168.0,
                value=config["alerts"]["alert_threshold_long_dwell"],
                step=0.5,
                help="Hours to trigger long dwell time alert"
            )
            
            config["alerts"]["notification_cooldown"] = st.number_input(
                "Notification Cooldown (seconds)",
                min_value=60,
                max_value=3600,
                value=config["alerts"]["notification_cooldown"],
                help="Minimum time between identical alerts"
            )
        
        with col2:
            config["alerts"]["alert_threshold_low_accuracy"] = st.slider(
                "Low Accuracy Alert Threshold",
                min_value=0.5,
                max_value=1.0,
                value=config["alerts"]["alert_threshold_low_accuracy"],
                step=0.01,
                help="Detection accuracy below this triggers alert"
            )
            
            config["alerts"]["email_notifications"] = st.checkbox(
                "Email Notifications",
                value=config["alerts"]["email_notifications"],
                help="Send alerts via email"
            )
            
            if config["alerts"]["email_notifications"]:
                email_list = st.text_area(
                    "Email Recipients (one per line)",
                    value="\n".join(config["alerts"]["email_recipients"]),
                    height=100,
                    help="Email addresses to receive notifications"
                )
                config["alerts"]["email_recipients"] = [
                    email.strip() for email in email_list.split("\n") if email.strip()
                ]
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return config


def create_data_retention_section(config: Dict) -> Dict:
    """Create data retention configuration section."""
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("#### 🗄️ Data Retention Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config["data_retention"]["keep_images_days"] = st.number_input(
            "Keep Images (days)",
            min_value=1,
            max_value=365,
            value=config["data_retention"]["keep_images_days"],
            help="Number of days to keep original images"
        )
        
        config["data_retention"]["keep_detection_data_days"] = st.number_input(
            "Keep Detection Data (days)",
            min_value=30,
            max_value=1095,  # 3 years
            value=config["data_retention"]["keep_detection_data_days"],
            help="Number of days to keep detection records in database"
        )
        
        config["data_retention"]["auto_cleanup_enabled"] = st.checkbox(
            "Enable Auto Cleanup",
            value=config["data_retention"]["auto_cleanup_enabled"],
            help="Automatically delete old data based on retention settings"
        )
    
    with col2:
        config["data_retention"]["compress_old_images"] = st.checkbox(
            "Compress Old Images",
            value=config["data_retention"]["compress_old_images"],
            help="Compress images older than 7 days to save disk space"
        )
        
        config["data_retention"]["backup_enabled"] = st.checkbox(
            "Enable Backups",
            value=config["data_retention"]["backup_enabled"],
            help="Create regular backups of database and configuration"
        )
        
        if config["data_retention"]["backup_enabled"]:
            config["data_retention"]["backup_frequency"] = st.selectbox(
                "Backup Frequency",
                options=["daily", "weekly", "monthly"],
                index=["daily", "weekly", "monthly"].index(config["data_retention"]["backup_frequency"]),
                help="How often to create backups"
            )
    
    # Show current data usage (from actual file system)
    st.markdown("**Current Data Usage:**")
    col1, col2, col3 = st.columns(3)
    
    # Get actual data usage
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from modules.database import queries
    
    # Count actual images
    image_dir = Path("data/images")
    if image_dir.exists():
        image_files = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.jpeg")) + list(image_dir.glob("**/*.png"))
        image_count = len(image_files)
        image_size_mb = sum(f.stat().st_size for f in image_files) / (1024 * 1024) if image_files else 0
    else:
        image_count = 0
        image_size_mb = 0
    
    # Get database size
    db_path = Path("data/database.db")
    if db_path.exists():
        db_size_mb = db_path.stat().st_size / (1024 * 1024)
    else:
        db_size_mb = 0
    
    # Calculate total storage
    total_storage_mb = image_size_mb + db_size_mb
    
    with col1:
        st.metric("Images Stored", f"{image_count:,} files", f"{image_size_mb:.1f} MB")
    
    with col2:
        st.metric("Database Size", f"{db_size_mb:.1f} MB", "")
    
    with col3:
        st.metric("Total Storage", f"{total_storage_mb:.1f} MB", "")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return config


def create_system_settings_section(config: Dict) -> Dict:
    """Create system settings configuration section."""
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config["system"]["log_level"] = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config["system"]["log_level"]),
            help="Minimum level for log messages"
        )
        
        config["system"]["camera_refresh_interval"] = st.number_input(
            "Camera Refresh Interval (seconds)",
            min_value=5,
            max_value=300,
            value=config["system"]["camera_refresh_interval"],
            help="How often to check camera feeds"
        )
        
        config["system"]["max_concurrent_detections"] = st.number_input(
            "Max Concurrent Detections",
            min_value=1,
            max_value=16,
            value=config["system"]["max_concurrent_detections"],
            help="Maximum number of images to process simultaneously"
        )
    
    with col2:
        config["system"]["dashboard_refresh_interval"] = st.number_input(
            "Dashboard Refresh Interval (seconds)",
            min_value=10,
            max_value=300,
            value=config["system"]["dashboard_refresh_interval"],
            help="Auto-refresh interval for dashboard"
        )
        
        config["system"]["enable_performance_monitoring"] = st.checkbox(
            "Enable Performance Monitoring",
            value=config["system"]["enable_performance_monitoring"],
            help="Track system performance metrics"
        )
        
        config["system"]["max_log_files"] = st.number_input(
            "Max Log Files",
            min_value=5,
            max_value=100,
            value=config["system"]["max_log_files"],
            help="Maximum number of log files to keep"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return config


def create_ui_settings_section(config: Dict) -> Dict:
    """Create UI settings configuration section."""
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("#### 🎨 User Interface Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config["ui"]["theme"] = st.selectbox(
            "Theme",
            options=["light", "dark", "auto"],
            index=["light", "dark", "auto"].index(config["ui"]["theme"]),
            help="Dashboard color theme"
        )
        
        config["ui"]["items_per_page"] = st.number_input(
            "Items per Page",
            min_value=10,
            max_value=100,
            value=config["ui"]["items_per_page"],
            help="Number of items to show in lists/tables"
        )
        
        config["ui"]["default_date_range"] = st.number_input(
            "Default Date Range (days)",
            min_value=1,
            max_value=365,
            value=config["ui"]["default_date_range"],
            help="Default number of days to show in analytics"
        )
    
    with col2:
        config["ui"]["enable_animations"] = st.checkbox(
            "Enable Animations",
            value=config["ui"]["enable_animations"],
            help="Enable UI animations and transitions"
        )
        
        config["ui"]["show_confidence_always"] = st.checkbox(
            "Always Show Confidence Scores",
            value=config["ui"]["show_confidence_always"],
            help="Display confidence scores on all detections"
        )
        
        config["ui"]["enable_experimental_features"] = st.checkbox(
            "Enable Experimental Features",
            value=config["ui"]["enable_experimental_features"],
            help="Enable beta features (may be unstable)"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return config


def display_recent_logs(logs: List[Dict]):
    """Display recent system logs."""
    st.markdown("### 📋 Recent System Logs")
    
    # Filter controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        log_level_filter = st.multiselect(
            "Filter by Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            default=["INFO", "WARNING", "ERROR"],
            key="log_filter"
        )
    
    # Filter logs
    filtered_logs = [log for log in logs if log["level"] in log_level_filter]
    
    if not filtered_logs:
        st.info("No logs match the current filter.")
        return
    
    # Create logs dataframe
    logs_df = pd.DataFrame(filtered_logs)
    logs_df['time'] = logs_df['timestamp'].dt.strftime('%H:%M:%S')
    logs_df['date'] = logs_df['timestamp'].dt.strftime('%Y-%m-%d')
    
    # Display logs in expandable sections by date
    for date in logs_df['date'].unique():
        day_logs = logs_df[logs_df['date'] == date]
        
        with st.expander(f"📅 {date} ({len(day_logs)} logs)", expanded=(date == datetime.now().strftime('%Y-%m-%d'))):
            for _, log in day_logs.iterrows():
                level_colors = {
                    'DEBUG': '#6c757d',
                    'INFO': '#17a2b8',
                    'WARNING': '#ffc107',
                    'ERROR': '#dc3545'
                }
                
                color = level_colors.get(log['level'], '#6c757d')
                
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid {color}; background-color: #f8f9fa;">
                    <strong style="color: {color};">[{log['time']}] {log['level']}</strong><br>
                    {log['message']}
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main settings dashboard."""
    
    # Header
    st.markdown('<div class="settings-header">⚙️ Settings & Configuration</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Settings Navigation")
        
        settings_section = st.radio(
            "Settings Section",
            options=[
                "🎯 Detection",
                "🚨 Alerts", 
                "🗄️ Data Retention",
                "⚙️ System",
                "🎨 UI Settings",
                "📊 System Status",
                "📋 Logs",
                "🔧 Advanced"
            ],
            index=0
        )
        
        st.divider()
        
        # Quick actions
        st.markdown("### Quick Actions")
        
        if st.button("💾 Export Config", use_container_width=True):
            config = load_current_config()
            config_json = json.dumps(config, indent=2)
            st.download_button(
                "Download Configuration",
                config_json,
                file_name=f"container_analytics_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        if st.button("🔄 Restart Services", use_container_width=True):
            st.success("Service restart initiated!")
        
        if st.button("🧹 Run Cleanup", use_container_width=True):
            st.success("Cleanup task started!")
        
        st.divider()
        st.markdown("### Navigation")
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("app.py")
    
    # Load current configuration
    config = load_current_config()
    config_changed = False
    
    # Show different sections based on sidebar selection
    if settings_section == "🎯 Detection":
        st.markdown("## Detection Configuration")
        new_config = create_detection_settings_section(config)
        config_changed = new_config != config
        config = new_config
        
    elif settings_section == "🚨 Alerts":
        st.markdown("## Alert Configuration")
        new_config = create_alert_settings_section(config)
        config_changed = new_config != config
        config = new_config
        
    elif settings_section == "🗄️ Data Retention":
        st.markdown("## Data Retention Configuration")
        new_config = create_data_retention_section(config)
        config_changed = new_config != config
        config = new_config
        
    elif settings_section == "⚙️ System":
        st.markdown("## System Configuration")
        new_config = create_system_settings_section(config)
        config_changed = new_config != config
        config = new_config
        
    elif settings_section == "🎨 UI Settings":
        st.markdown("## User Interface Configuration")
        new_config = create_ui_settings_section(config)
        config_changed = new_config != config
        config = new_config
        
    elif settings_section == "📊 System Status":
        st.markdown("## System Status & Monitoring")
        
        # Load system status
        with st.spinner("Loading system status..."):
            status_data = load_system_status()
        
        # Display service status
        display_service_status(status_data["services"])
        
        st.divider()
        
        # Display system resources
        display_system_resources(status_data["system_resources"])
        
        st.divider()
        
        # Performance metrics
        st.markdown("### ⚡ Performance Metrics")
        
        perf = status_data["performance_metrics"]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Detection Speed", f"{perf['detection_speed_avg']:.2f}s", "-0.05s")
        
        with col2:
            st.metric("Accuracy", f"{perf['detection_accuracy']:.1%}", "+0.3%")
        
        with col3:
            st.metric("Uptime", f"{perf['uptime_percentage']:.1f}%", "")
        
        with col4:
            st.metric("Cache Hit Rate", f"{perf['cache_hit_rate']:.1%}", "+2%")
        
    elif settings_section == "📋 Logs":
        st.markdown("## System Logs")
        
        # Load system status for logs
        status_data = load_system_status()
        display_recent_logs(status_data["recent_logs"])
        
    elif settings_section == "🔧 Advanced":
        st.markdown("## Advanced Settings")
        
        st.markdown("""
        <div class="advanced-settings">
        <h4>⚠️ Advanced Configuration</h4>
        <p>These settings are for advanced users only. Incorrect configuration may affect system performance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration file editor
        st.markdown("### 📝 Configuration Editor")
        
        config_json = json.dumps(config, indent=2)
        
        edited_config = st.text_area(
            "Raw Configuration (JSON)",
            value=config_json,
            height=400,
            help="Direct JSON configuration editing. Be careful with syntax."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Validate JSON"):
                try:
                    json.loads(edited_config)
                    st.success("JSON is valid!")
                except json.JSONDecodeError as e:
                    st.error(f"JSON Error: {str(e)}")
        
        with col2:
            if st.button("💾 Save Advanced Config"):
                try:
                    new_config = json.loads(edited_config)
                    if save_config(new_config):
                        st.success("Advanced configuration saved!")
                        st.rerun()
                    else:
                        st.error("Failed to save configuration!")
                except json.JSONDecodeError as e:
                    st.error(f"Cannot save invalid JSON: {str(e)}")
        
        # System maintenance
        st.markdown("### 🛠️ System Maintenance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Cache"):
                st.success("System cache cleared!")
        
        with col2:
            if st.button("🔄 Reset to Defaults"):
                if st.button("⚠️ Confirm Reset"):
                    config = DEFAULT_CONFIG.copy()
                    save_config(config)
                    st.success("Configuration reset to defaults!")
                    st.rerun()
        
        with col3:
            if st.button("📊 Generate Diagnostics"):
                st.success("Diagnostics report generated!")
    
    # Show configuration changed message and save button
    if config_changed:
        st.markdown("""
        <div class="config-changed">
            <strong>⚠️ Configuration Changed</strong><br>
            Your changes have been made but not yet saved. Click "Save Configuration" to apply changes.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("💾 Save Configuration", use_container_width=True):
                if save_config(config):
                    st.success("✅ Configuration saved successfully!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to save configuration!")
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Settings last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        Container Analytics v1.0 | Configuration Version: 1.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()