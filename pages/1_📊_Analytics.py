"""
Container Analytics - Detailed Analytics Dashboard

This page provides detailed analytics charts, KPI breakdowns,
date range selection, and data export functionality.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import numpy as np
from typing import Dict, List, Tuple
import io
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from modules.database import queries
from modules.database.models import session_scope, Container, Detection, Image, Metric

# Configure page
st.set_page_config(
    page_title="Analytics - Container Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.analytics-header {
    font-size: 2.2rem;
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 1rem;
}

.kpi-container {
    background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.export-section {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_analytics_data(start_date: date, end_date: date) -> Dict:
    """
    Load comprehensive analytics data for the specified date range.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        Dictionary containing analytics data
    """
    # Convert dates to datetime for database queries
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Get metrics from database
    metrics = queries.get_metrics_by_date_range(start_datetime, end_datetime)
    container_stats = queries.get_container_statistics(start_datetime, end_datetime)
    detection_summary = queries.get_detection_summary(start_datetime, end_datetime)
    
    # Process daily data
    daily_data = []
    if metrics:
        # Group metrics by date
        metrics_by_date = {}
        for metric in metrics:
            date_key = metric['date'].date() if metric['date'] else start_date
            if date_key not in metrics_by_date:
                metrics_by_date[date_key] = []
            metrics_by_date[date_key].append(metric)
        
        # Aggregate by day
        for d in pd.date_range(start_date, end_date):
            day_metrics = metrics_by_date.get(d.date(), [])
            if day_metrics:
                daily_data.append({
                    'date': d.date(),
                    'containers_in': sum(m.get('throughput', 0) // 2 for m in day_metrics),
                    'containers_out': sum(m.get('throughput', 0) - m.get('throughput', 0) // 2 for m in day_metrics),
                    'avg_dwell_time': np.mean([m.get('avg_dwell_time', 0) for m in day_metrics if m.get('avg_dwell_time')]) or 0,
                    'peak_occupancy': max((m.get('container_count', 0) for m in day_metrics), default=0),
                    'detection_accuracy': np.mean([m.get('avg_confidence', 0) for m in day_metrics if m.get('avg_confidence')]) or 0.95,
                    'processing_time': 1.5
                })
            else:
                daily_data.append({
                    'date': d.date(),
                    'containers_in': 0,
                    'containers_out': 0,
                    'avg_dwell_time': 0,
                    'peak_occupancy': 0,
                    'detection_accuracy': 0,
                    'processing_time': 0
                })
    else:
        # No metrics data - create empty dataset
        for d in pd.date_range(start_date, end_date):
            daily_data.append({
                'date': d.date(),
                'containers_in': 0,
                'containers_out': 0,
                'avg_dwell_time': 0,
                'peak_occupancy': 0,
                'detection_accuracy': 0,
                'processing_time': 0
            })
    
    df = pd.DataFrame(daily_data)
    
    # Get hourly data for last 3 days
    three_days_ago = datetime.now() - timedelta(days=3)
    hourly_metrics = queries.get_metrics_by_date_range(three_days_ago, datetime.now())
    
    hourly_data = []
    if hourly_metrics:
        for metric in hourly_metrics:
            if metric['date'] and metric['hour'] is not None:
                hour_time = datetime.combine(metric['date'], datetime.min.time()) + timedelta(hours=metric['hour'])
                hourly_data.append({
                    'datetime': hour_time,
                    'hour': metric['hour'],
                    'containers_in': metric.get('throughput', 0) // 2,
                    'containers_out': metric.get('throughput', 0) - metric.get('throughput', 0) // 2,
                    'occupancy': metric.get('container_count', 0)
                })
    
    if not hourly_data:
        for i in range(72):
            hour_time = datetime.now() - timedelta(hours=71-i)
            hourly_data.append({
                'datetime': hour_time,
                'hour': hour_time.hour,
                'containers_in': 0,
                'containers_out': 0,
                'occupancy': 0
            })
    
    hourly_df = pd.DataFrame(hourly_data)
    
    # Container type breakdown - based on detection types
    container_types = {}
    if detection_summary and 'detection_counts' in detection_summary:
        for det in detection_summary['detection_counts']:
            container_types[det['object_type']] = det['count']
    
    if not container_types:
        container_types = {'No data': 1}
    
    # Get dwell time data
    dwell_time_data = queries.get_dwell_time_data(start_datetime, end_datetime)
    if dwell_time_data:
        dwell_times = [d['dwell_time'] for d in dwell_time_data if d['dwell_time'] is not None]
    else:
        dwell_times = []
    
    return {
        'daily_df': df,
        'hourly_df': hourly_df,
        'container_types': container_types,
        'dwell_times': dwell_times,
        'total_containers': int(df['containers_in'].sum() + df['containers_out'].sum()),
        'avg_dwell_time': float(df['avg_dwell_time'].mean()),
        'avg_occupancy': float(df['peak_occupancy'].mean()),
        'avg_accuracy': float(df['detection_accuracy'].mean()),
        'date_range': f"{start_date} to {end_date}"
    }


def create_traffic_flow_chart(df: pd.DataFrame) -> go.Figure:
    """Create a traffic flow chart showing containers in/out over time."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Container Movement', 'Net Flow'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )
    
    # Daily movement
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['containers_in'],
            mode='lines+markers',
            name='Containers In',
            line=dict(color='#28a745', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['containers_out'],
            mode='lines+markers',
            name='Containers Out',
            line=dict(color='#dc3545', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Net flow
    net_flow = df['containers_in'] - df['containers_out']
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=net_flow,
            name='Net Flow',
            marker_color=['#28a745' if x >= 0 else '#dc3545' for x in net_flow]
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Container Count", row=1, col=1)
    fig.update_yaxes(title_text="Net Flow", row=2, col=1)
    
    return fig


def create_dwell_time_analysis(dwell_times: np.ndarray) -> go.Figure:
    """Create dwell time analysis charts."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Dwell Time Distribution', 'Cumulative Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=dwell_times,
            nbinsx=20,
            name='Frequency',
            marker_color='#1f77b4',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Cumulative distribution
    sorted_times = np.sort(dwell_times)
    cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times) * 100
    
    fig.add_trace(
        go.Scatter(
            x=sorted_times,
            y=cumulative,
            mode='lines',
            name='Cumulative %',
            line=dict(color='#ff7f0e', width=3)
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Dwell Time (hours)", row=1, col=1)
    fig.update_xaxes(title_text="Dwell Time (hours)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative %", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def create_occupancy_heatmap(hourly_df: pd.DataFrame) -> go.Figure:
    """Create hourly occupancy heatmap."""
    # Prepare data for heatmap
    hourly_df['date'] = hourly_df['datetime'].dt.date
    pivot_data = hourly_df.pivot_table(
        values='occupancy', 
        index='hour', 
        columns='date', 
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[str(d) for d in pivot_data.columns],
        y=[f"{h:02d}:00" for h in pivot_data.index],
        colorscale='RdYlBu_r',
        colorbar=dict(title="Occupancy")
    ))
    
    fig.update_layout(
        title="Occupancy Patterns by Hour",
        xaxis_title="Date",
        yaxis_title="Hour of Day",
        height=500
    )
    
    return fig


def export_data_section(data: Dict):
    """Create data export section."""
    st.markdown('<div class="export-section">', unsafe_allow_html=True)
    st.markdown("#### 📤 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Daily Summary (CSV)", use_container_width=True):
            csv = data['daily_df'].to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"container_analytics_daily_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
    
    with col2:
        if st.button("📊 Export Hourly Data (CSV)", use_container_width=True):
            csv = data['hourly_df'].to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"container_analytics_hourly_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
    
    with col3:
        if st.button("📈 Export Summary Report (JSON)", use_container_width=True):
            summary = {
                'date_range': data['date_range'],
                'total_containers': data['total_containers'],
                'avg_dwell_time': data['avg_dwell_time'],
                'avg_occupancy': data['avg_occupancy'],
                'avg_accuracy': data['avg_accuracy'],
                'container_types': data['container_types']
            }
            
            import json
            json_str = json.dumps(summary, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"container_analytics_summary_{datetime.now().strftime('%Y%m%d')}.json",
                mime='application/json'
            )
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main analytics dashboard."""
    
    # Header
    st.markdown('<div class="analytics-header">📊 Detailed Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.title("Analytics Controls")
        
        # Date range selection
        st.subheader("📅 Date Range")
        
        # Quick presets
        preset = st.selectbox(
            "Quick Select",
            ["Custom", "Last 7 days", "Last 30 days", "Last 90 days", "This month", "Last month"]
        )
        
        if preset == "Last 7 days":
            end_date = date.today()
            start_date = end_date - timedelta(days=7)
        elif preset == "Last 30 days":
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
        elif preset == "Last 90 days":
            end_date = date.today()
            start_date = end_date - timedelta(days=90)
        elif preset == "This month":
            end_date = date.today()
            start_date = end_date.replace(day=1)
        elif preset == "Last month":
            end_date = date.today().replace(day=1) - timedelta(days=1)
            start_date = end_date.replace(day=1)
        else:  # Custom
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", date.today() - timedelta(days=30))
            with col2:
                end_date = st.date_input("End Date", date.today())
        
        # Analytics options
        st.subheader("🔧 Analytics Options")
        show_trends = st.checkbox("Show trend lines", True)
        show_averages = st.checkbox("Show moving averages", True)
        include_weekends = st.checkbox("Include weekends", True)
        
        # Refresh data
        if st.button("🔄 Refresh Analytics"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.markdown("### Navigation")
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("app.py")
    
    # Validate date range
    if start_date > end_date:
        st.error("Start date must be before end date!")
        return
    
    if (end_date - start_date).days > 365:
        st.warning("Date range too large. Limiting to 365 days for performance.")
        start_date = end_date - timedelta(days=365)
    
    # Load data
    with st.spinner("Loading analytics data..."):
        data = load_analytics_data(start_date, end_date)
    
    # Key metrics summary
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-container">
            <h4>Total Containers</h4>
            <h2 style="color: #1f77b4;">{data['total_containers']:,}</h2>
            <small>During selected period</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-container">
            <h4>Avg Dwell Time</h4>
            <h2 style="color: #28a745;">{data['avg_dwell_time']:.1f} hrs</h2>
            <small>Per container</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-container">
            <h4>Avg Occupancy</h4>
            <h2 style="color: #fd7e14;">{data['avg_occupancy']:.0f}</h2>
            <small>Containers in terminal</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-container">
            <h4>Detection Accuracy</h4>
            <h2 style="color: #20c997;">{data['avg_accuracy']:.1%}</h2>
            <small>Computer vision</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="kpi-container">
            <h4>Analysis Period</h4>
            <h2 style="color: #6f42c1;">{(end_date - start_date).days}</h2>
            <small>Days analyzed</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Main charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🚢 Container Traffic Flow")
        traffic_chart = create_traffic_flow_chart(data['daily_df'])
        st.plotly_chart(traffic_chart, use_container_width=True)
    
    with col2:
        st.markdown("### 📦 Container Types")
        fig_pie = px.pie(
            values=list(data['container_types'].values()),
            names=list(data['container_types'].keys()),
            title="Distribution by Type"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Dwell time analysis
    st.markdown("### ⏰ Dwell Time Analysis")
    dwell_chart = create_dwell_time_analysis(data['dwell_times'])
    st.plotly_chart(dwell_chart, use_container_width=True)
    
    # Dwell time statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Median Dwell Time", f"{np.median(data['dwell_times']):.1f} hrs")
    with col2:
        st.metric("90th Percentile", f"{np.percentile(data['dwell_times'], 90):.1f} hrs")
    with col3:
        st.metric("Max Dwell Time", f"{np.max(data['dwell_times']):.1f} hrs")
    with col4:
        st.metric("Std Deviation", f"{np.std(data['dwell_times']):.1f} hrs")
    
    # Occupancy heatmap
    st.markdown("### 🔥 Occupancy Heatmap (Last 3 Days)")
    occupancy_chart = create_occupancy_heatmap(data['hourly_df'])
    st.plotly_chart(occupancy_chart, use_container_width=True)
    
    st.divider()
    
    # Export section
    export_data_section(data)
    
    # Additional insights
    st.markdown("### 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Peak Traffic Day**: {data['daily_df'].loc[data['daily_df']['containers_in'].idxmax(), 'date'].strftime('%Y-%m-%d')} 
        ({data['daily_df']['containers_in'].max()} containers in)
        """)
        
        st.success(f"""
        **Best Performance Day**: {data['daily_df'].loc[data['daily_df']['detection_accuracy'].idxmax(), 'date'].strftime('%Y-%m-%d')} 
        ({data['daily_df']['detection_accuracy'].max():.1%} accuracy)
        """)
    
    with col2:
        avg_daily_in = data['daily_df']['containers_in'].mean()
        avg_daily_out = data['daily_df']['containers_out'].mean()
        
        if avg_daily_in > avg_daily_out * 1.1:
            st.warning(f"**Trend Alert**: Inbound traffic ({avg_daily_in:.1f}) exceeds outbound ({avg_daily_out:.1f}) by >10%")
        elif avg_daily_out > avg_daily_in * 1.1:
            st.warning(f"**Trend Alert**: Outbound traffic ({avg_daily_out:.1f}) exceeds inbound ({avg_daily_in:.1f}) by >10%")
        else:
            st.success("**Traffic Balance**: Inbound and outbound traffic are well balanced")
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Analytics generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        Data range: {data['date_range']} | Total data points: {len(data['daily_df'])} days
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()