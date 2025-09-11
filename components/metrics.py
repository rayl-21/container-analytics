"""
KPI cards and metrics display components for Container Analytics dashboard.

This module provides components for displaying key performance indicators
with trend indicators, color coding, and real-time updates.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import math


class MetricsCard:
    """Component for displaying KPI metrics in card format."""
    
    def __init__(self):
        """Initialize the MetricsCard component."""
        self.trend_colors = {
            'up': '#2F9B69',      # Green
            'down': '#F24236',    # Red
            'neutral': '#808080'   # Gray
        }
        
        self.status_colors = {
            'good': '#2F9B69',     # Green
            'warning': '#F6AE2D',  # Yellow
            'critical': '#F24236', # Red
            'info': '#2E86AB'      # Blue
        }
    
    def display_metric_card(
        self,
        title: str,
        value: Union[int, float, str],
        previous_value: Optional[Union[int, float]] = None,
        unit: str = "",
        format_func: Optional[callable] = None,
        delta_label: str = "vs previous",
        status: str = "info",
        description: Optional[str] = None,
        target_value: Optional[Union[int, float]] = None,
        show_progress: bool = False
    ) -> None:
        """
        Display a single metric card with trend indicator.
        
        Args:
            title: Metric title
            value: Current metric value
            previous_value: Previous value for trend calculation
            unit: Unit of measurement
            format_func: Function to format the value
            delta_label: Label for the trend indicator
            status: Status color ('good', 'warning', 'critical', 'info')
            description: Optional description text
            target_value: Target value for progress indicator
            show_progress: Whether to show progress bar
        """
        # Format the main value
        if format_func:
            formatted_value = format_func(value)
        elif isinstance(value, (int, float)):
            if abs(value) >= 1000000:
                formatted_value = f"{value/1000000:.1f}M"
            elif abs(value) >= 1000:
                formatted_value = f"{value/1000:.1f}K"
            else:
                formatted_value = f"{value:,.1f}" if isinstance(value, float) else f"{value:,}"
        else:
            formatted_value = str(value)
        
        # Calculate trend if previous value is provided
        trend_direction = None
        trend_percentage = None
        if previous_value is not None and isinstance(value, (int, float)) and isinstance(previous_value, (int, float)):
            if previous_value != 0:
                trend_percentage = ((value - previous_value) / previous_value) * 100
                trend_direction = 'up' if trend_percentage > 0 else 'down' if trend_percentage < 0 else 'neutral'
        
        # Get status color
        card_color = self.status_colors.get(status, self.status_colors['info'])
        
        # Create the card using columns for layout
        with st.container():
            # Add custom CSS for the card
            st.markdown(f"""
                <div style="
                    background-color: white;
                    padding: 1rem;
                    border-radius: 10px;
                    border-left: 4px solid {card_color};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 1rem;
                ">
                    <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">
                        {title}
                    </div>
                    <div style="font-size: 2rem; font-weight: bold; color: #333; margin-bottom: 0.5rem;">
                        {formatted_value}{' ' + unit if unit else ''}
                    </div>
            """, unsafe_allow_html=True)
            
            # Add trend indicator if available
            if trend_direction is not None and trend_percentage is not None:
                trend_color = self.trend_colors[trend_direction]
                trend_arrow = "↗️" if trend_direction == 'up' else "↘️" if trend_direction == 'down' else "➡️"
                
                st.markdown(f"""
                    <div style="color: {trend_color}; font-size: 0.9rem; margin-bottom: 0.5rem;">
                        {trend_arrow} {abs(trend_percentage):.1f}% {delta_label}
                    </div>
                """, unsafe_allow_html=True)
            
            # Add progress bar if target is provided
            if show_progress and target_value is not None and isinstance(value, (int, float)):
                progress = min(value / target_value, 1.0) if target_value > 0 else 0
                progress_color = card_color
                
                st.markdown(f"""
                    <div style="margin-bottom: 0.5rem;">
                        <div style="background-color: #f0f0f0; border-radius: 10px; height: 8px;">
                            <div style="
                                background-color: {progress_color}; 
                                width: {progress * 100}%; 
                                height: 8px; 
                                border-radius: 10px;
                            "></div>
                        </div>
                        <div style="font-size: 0.8rem; color: #666; margin-top: 0.25rem;">
                            Target: {target_value:,}{' ' + unit if unit else ''}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Add description if provided
            if description:
                st.markdown(f"""
                    <div style="color: #666; font-size: 0.8rem;">
                        {description}
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def display_metrics_grid(
        self,
        metrics_data: List[Dict[str, Any]],
        columns: int = 3
    ) -> None:
        """
        Display multiple metrics in a grid layout.
        
        Args:
            metrics_data: List of metric dictionaries
            columns: Number of columns in the grid
        """
        if not metrics_data:
            st.warning("No metrics data provided")
            return
        
        # Create grid layout
        for i in range(0, len(metrics_data), columns):
            cols = st.columns(columns)
            
            for j, col in enumerate(cols):
                if i + j < len(metrics_data):
                    metric = metrics_data[i + j]
                    
                    with col:
                        self.display_metric_card(**metric)
    
    def display_metric_with_sparkline(
        self,
        title: str,
        value: Union[int, float],
        historical_data: List[Union[int, float]],
        unit: str = "",
        height: int = 60,
        status: str = "info"
    ) -> None:
        """
        Display metric card with embedded sparkline chart.
        
        Args:
            title: Metric title
            value: Current value
            historical_data: Historical values for sparkline
            unit: Unit of measurement
            height: Height of sparkline
            status: Status color
        """
        card_color = self.status_colors.get(status, self.status_colors['info'])
        
        # Create sparkline chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=historical_data,
            mode='lines',
            line=dict(color=card_color, width=2),
            showlegend=False,
            hovertemplate='Value: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            hovermode='x'
        )
        
        # Display in columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.display_metric_card(
                title=title,
                value=value,
                unit=unit,
                status=status
            )
        
        with col2:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    def create_comparison_metrics(
        self,
        current_data: Dict[str, Union[int, float]],
        previous_data: Dict[str, Union[int, float]],
        period_label: str = "Previous Period",
        columns: int = 2
    ) -> None:
        """
        Create comparison metrics between two time periods.
        
        Args:
            current_data: Current period metrics
            previous_data: Previous period metrics
            period_label: Label for comparison period
            columns: Number of columns for display
        """
        st.subheader(f"Comparison vs {period_label}")
        
        comparison_metrics = []
        
        for key in current_data.keys():
            if key in previous_data:
                current_val = current_data[key]
                previous_val = previous_data[key]
                
                # Calculate change
                if isinstance(current_val, (int, float)) and isinstance(previous_val, (int, float)):
                    if previous_val != 0:
                        change_pct = ((current_val - previous_val) / previous_val) * 100
                        status = 'good' if change_pct > 0 else 'critical' if change_pct < -10 else 'warning'
                    else:
                        change_pct = 0
                        status = 'info'
                    
                    comparison_metrics.append({
                        'title': key.replace('_', ' ').title(),
                        'value': current_val,
                        'previous_value': previous_val,
                        'status': status,
                        'delta_label': f"vs {period_label.lower()}"
                    })
        
        self.display_metrics_grid(comparison_metrics, columns=columns)
    
    def display_gauge_metric(
        self,
        title: str,
        value: float,
        min_value: float = 0,
        max_value: float = 100,
        target_value: Optional[float] = None,
        unit: str = "",
        color_ranges: Optional[List[Dict]] = None
    ) -> None:
        """
        Display a gauge-style metric.
        
        Args:
            title: Metric title
            value: Current value
            min_value: Minimum scale value
            max_value: Maximum scale value
            target_value: Optional target line
            unit: Unit of measurement
            color_ranges: List of color range definitions
        """
        if color_ranges is None:
            color_ranges = [
                {'range': [min_value, max_value * 0.6], 'color': '#F24236'},  # Red
                {'range': [max_value * 0.6, max_value * 0.8], 'color': '#F6AE2D'},  # Yellow
                {'range': [max_value * 0.8, max_value], 'color': '#2F9B69'}  # Green
            ]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 16}},
            number={'suffix': f" {unit}" if unit else ""},
            gauge={
                'axis': {'range': [min_value, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [cr['range'][0], cr['range'][1]], 'color': cr['color']}
                    for cr in color_ranges
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': target_value or max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    def create_status_indicator(
        self,
        title: str,
        status: str,
        message: str = "",
        show_timestamp: bool = True
    ) -> None:
        """
        Create a simple status indicator.
        
        Args:
            title: Status title
            status: Status level ('good', 'warning', 'critical', 'info')
            message: Optional status message
            show_timestamp: Whether to show current timestamp
        """
        status_icons = {
            'good': '✅',
            'warning': '⚠️',
            'critical': '❌',
            'info': 'ℹ️'
        }
        
        icon = status_icons.get(status, 'ℹ️')
        color = self.status_colors.get(status, self.status_colors['info'])
        
        timestamp_str = ""
        if show_timestamp:
            timestamp_str = f" - {datetime.now().strftime('%H:%M:%S')}"
        
        st.markdown(f"""
            <div style="
                background-color: white;
                padding: 0.75rem;
                border-radius: 8px;
                border-left: 4px solid {color};
                margin-bottom: 0.5rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                <div style="font-weight: bold; color: #333;">
                    {icon} {title}{timestamp_str}
                </div>
                {f'<div style="color: #666; font-size: 0.9rem; margin-top: 0.25rem;">{message}</div>' if message else ''}
            </div>
        """, unsafe_allow_html=True)


def format_number(value: Union[int, float], precision: int = 1) -> str:
    """Format number with appropriate units (K, M, B)."""
    if abs(value) >= 1e9:
        return f"{value/1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}" if isinstance(value, float) else str(value)


def format_duration(seconds: Union[int, float]) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def format_percentage(value: Union[int, float], precision: int = 1) -> str:
    """Format value as percentage."""
    return f"{value:.{precision}f}%"


def calculate_trend(
    current_value: Union[int, float],
    previous_value: Union[int, float]
) -> Dict[str, Any]:
    """
    Calculate trend information between two values.
    
    Args:
        current_value: Current value
        previous_value: Previous value
        
    Returns:
        Dictionary with trend information
    """
    if previous_value == 0:
        return {
            'direction': 'neutral',
            'percentage': 0,
            'absolute_change': current_value
        }
    
    absolute_change = current_value - previous_value
    percentage_change = (absolute_change / previous_value) * 100
    
    if percentage_change > 0:
        direction = 'up'
    elif percentage_change < 0:
        direction = 'down'
    else:
        direction = 'neutral'
    
    return {
        'direction': direction,
        'percentage': abs(percentage_change),
        'absolute_change': absolute_change
    }


def create_metric_summary(
    metrics_data: Dict[str, Union[int, float]],
    title: str = "Metrics Summary"
) -> None:
    """
    Create a summary table of metrics.
    
    Args:
        metrics_data: Dictionary of metric names and values
        title: Summary title
    """
    st.subheader(title)
    
    # Convert to DataFrame for display
    df = pd.DataFrame(list(metrics_data.items()), columns=['Metric', 'Value'])
    df['Metric'] = df['Metric'].str.replace('_', ' ').str.title()
    
    # Format values
    df['Formatted Value'] = df['Value'].apply(lambda x: format_number(x) if isinstance(x, (int, float)) else str(x))
    
    # Display as table
    st.dataframe(
        df[['Metric', 'Formatted Value']],
        use_container_width=True,
        hide_index=True
    )


def create_alert_banner(
    message: str,
    alert_type: str = "info",
    dismissible: bool = True
) -> None:
    """
    Create an alert banner at the top of the page.
    
    Args:
        message: Alert message
        alert_type: Type of alert ('info', 'warning', 'error', 'success')
        dismissible: Whether the alert can be dismissed
    """
    alert_styles = {
        'info': {'color': '#2E86AB', 'bg': '#E3F2FD', 'icon': 'ℹ️'},
        'warning': {'color': '#F6AE2D', 'bg': '#FFF3E0', 'icon': '⚠️'},
        'error': {'color': '#F24236', 'bg': '#FFEBEE', 'icon': '❌'},
        'success': {'color': '#2F9B69', 'bg': '#E8F5E8', 'icon': '✅'}
    }
    
    style = alert_styles.get(alert_type, alert_styles['info'])
    
    dismiss_button = ""
    if dismissible:
        dismiss_button = '''
            <button onclick="this.parentElement.style.display='none'" 
                    style="float: right; background: none; border: none; font-size: 1.2rem; cursor: pointer;">
                ✖️
            </button>
        '''
    
    st.markdown(f"""
        <div style="
            background-color: {style['bg']};
            border: 1px solid {style['color']};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            color: {style['color']};
        ">
            {dismiss_button}
            <strong>{style['icon']} {message}</strong>
        </div>
    """, unsafe_allow_html=True)