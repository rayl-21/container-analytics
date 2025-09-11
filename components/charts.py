"""
Plotly chart generators for Container Analytics dashboard.

This module provides functions to create various types of charts
for visualizing container flow data, throughput metrics, and analytics.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import streamlit as st


def create_time_series_chart(
    data: pd.DataFrame,
    x_col: str = 'timestamp',
    y_col: str = 'container_count',
    title: str = 'Container Flow Over Time',
    height: int = 400,
    show_trend: bool = True
) -> go.Figure:
    """
    Create a time series chart for container flow data.
    
    Args:
        data: DataFrame with time series data
        x_col: Column name for x-axis (timestamp)
        y_col: Column name for y-axis (values)
        title: Chart title
        height: Chart height in pixels
        show_trend: Whether to show trend line
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Main time series line
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines+markers',
        name='Container Count',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4, color='#2E86AB'),
        hovertemplate='<b>%{x}</b><br>Containers: %{y}<extra></extra>'
    ))
    
    # Add trend line if requested
    if show_trend and len(data) > 2:
        z = np.polyfit(data.index, data[y_col], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=p(data.index),
            mode='lines',
            name='Trend',
            line=dict(color='#F24236', width=2, dash='dash'),
            hovertemplate='Trend: %{y:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font_size=16),
        xaxis_title='Time',
        yaxis_title='Container Count',
        height=height,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def create_hourly_throughput_chart(
    data: pd.DataFrame,
    title: str = 'Hourly Container Throughput',
    height: int = 400,
    show_average: bool = True
) -> go.Figure:
    """
    Create a bar chart showing hourly container throughput.
    
    Args:
        data: DataFrame with hourly data (columns: 'hour', 'throughput')
        title: Chart title
        height: Chart height in pixels
        show_average: Whether to show average line
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Color bars based on throughput levels
    colors = []
    max_throughput = data['throughput'].max()
    for value in data['throughput']:
        if value >= max_throughput * 0.8:
            colors.append('#F24236')  # High - Red
        elif value >= max_throughput * 0.6:
            colors.append('#F6AE2D')  # Medium - Yellow
        else:
            colors.append('#2E86AB')  # Low - Blue
    
    # Bar chart
    fig.add_trace(go.Bar(
        x=data['hour'],
        y=data['throughput'],
        name='Throughput',
        marker_color=colors,
        hovertemplate='Hour: %{x}:00<br>Throughput: %{y} containers<extra></extra>'
    ))
    
    # Add average line if requested
    if show_average and len(data) > 0:
        avg_throughput = data['throughput'].mean()
        fig.add_hline(
            y=avg_throughput,
            line_dash="dash",
            line_color="#F24236",
            annotation_text=f"Average: {avg_throughput:.1f}",
            annotation_position="top right"
        )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font_size=16),
        xaxis_title='Hour of Day',
        yaxis_title='Container Count',
        height=height,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    
    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def create_peak_hour_heatmap(
    data: pd.DataFrame,
    title: str = 'Peak Hours Analysis',
    height: int = 400
) -> go.Figure:
    """
    Create a heatmap showing peak hours across days of the week.
    
    Args:
        data: DataFrame with columns ['day_of_week', 'hour', 'activity_level']
        title: Chart title
        height: Chart height in pixels
        
    Returns:
        Plotly figure object
    """
    # Pivot data for heatmap
    pivot_data = data.pivot(index='day_of_week', columns='hour', values='activity_level')
    
    # Ensure we have all days and hours
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))
    
    pivot_data = pivot_data.reindex(index=days, columns=hours, fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=hours,
        y=days,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Activity: %{z}<extra></extra>',
        colorbar=dict(title="Activity Level")
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font_size=16),
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=height,
        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_container_type_pie_chart(
    data: pd.DataFrame,
    title: str = 'Container Type Distribution',
    height: int = 400
) -> go.Figure:
    """
    Create a pie chart showing container type distribution.
    
    Args:
        data: DataFrame with columns ['container_type', 'count']
        title: Chart title
        height: Chart height in pixels
        
    Returns:
        Plotly figure object
    """
    # Define colors for different container types
    color_map = {
        '20ft': '#2E86AB',
        '40ft': '#A23B72', 
        '45ft': '#F18F01',
        'Other': '#C73E1D'
    }
    
    colors = [color_map.get(ct, '#808080') for ct in data['container_type']]
    
    fig = go.Figure(data=[go.Pie(
        labels=data['container_type'],
        values=data['count'],
        hole=0.3,  # Create donut chart
        marker_colors=colors,
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    # Add total count in center
    total_count = data['count'].sum()
    fig.add_annotation(
        text=f"Total<br>{total_count:,}",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font_size=16),
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_dwell_time_chart(
    data: pd.DataFrame,
    chart_type: str = 'line',
    title: str = 'Container Dwell Time Trends',
    height: int = 400
) -> go.Figure:
    """
    Create a chart showing container dwell time trends.
    
    Args:
        data: DataFrame with dwell time data
        chart_type: 'line' for line chart, 'box' for box plot, 'histogram' for distribution
        title: Chart title
        height: Chart height in pixels
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if chart_type == 'line':
        # Time series of average dwell time
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['avg_dwell_time'],
            mode='lines+markers',
            name='Average Dwell Time',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6),
            hovertemplate='Date: %{x}<br>Avg Dwell Time: %{y:.1f} hours<extra></extra>'
        ))
        
        # Add target line if exists
        if 'target_dwell_time' in data.columns:
            fig.add_hline(
                y=data['target_dwell_time'].iloc[0],
                line_dash="dash",
                line_color="#F24236",
                annotation_text="Target",
                annotation_position="bottom right"
            )
        
        fig.update_xaxis(title_text='Date')
        fig.update_yaxis(title_text='Dwell Time (hours)')
        
    elif chart_type == 'box':
        # Box plot by container type or time period
        if 'container_type' in data.columns:
            fig = px.box(
                data, 
                x='container_type', 
                y='dwell_time',
                title=title,
                color='container_type',
                color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            )
        else:
            fig.add_trace(go.Box(
                y=data['dwell_time'],
                name='Dwell Time',
                marker_color='#2E86AB',
                hovertemplate='Dwell Time: %{y:.1f} hours<extra></extra>'
            ))
        
        fig.update_yaxis(title_text='Dwell Time (hours)')
        
    elif chart_type == 'histogram':
        # Distribution histogram
        fig.add_trace(go.Histogram(
            x=data['dwell_time'],
            nbinsx=30,
            name='Distribution',
            marker_color='#2E86AB',
            opacity=0.7,
            hovertemplate='Dwell Time: %{x:.1f} hours<br>Count: %{y}<extra></extra>'
        ))
        
        # Add mean line
        mean_dwell = data['dwell_time'].mean()
        fig.add_vline(
            x=mean_dwell,
            line_dash="dash",
            line_color="#F24236",
            annotation_text=f"Mean: {mean_dwell:.1f}h",
            annotation_position="top"
        )
        
        fig.update_xaxis(title_text='Dwell Time (hours)')
        fig.update_yaxis(title_text='Count')
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font_size=16),
        height=height,
        showlegend=True if chart_type == 'box' and 'container_type' in data.columns else False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified' if chart_type == 'line' else 'closest'
    )
    
    if chart_type in ['line', 'box']:
        fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def create_multi_metric_chart(
    data: pd.DataFrame,
    metrics: List[str],
    title: str = 'Multiple Metrics Comparison',
    height: int = 500
) -> go.Figure:
    """
    Create a multi-axis chart for comparing different metrics.
    
    Args:
        data: DataFrame with time series data
        metrics: List of column names to plot
        title: Chart title
        height: Chart height in pixels
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    colors = ['#2E86AB', '#F24236', '#F6AE2D', '#2F9B69']
    
    for i, metric in enumerate(metrics):
        if metric in data.columns:
            use_secondary = i >= len(metrics) // 2
            
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'] if 'timestamp' in data.columns else data.index,
                    y=data[metric],
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                ),
                secondary_y=use_secondary,
            )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font_size=16),
        height=height,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


# Utility function to format data for charts
def prepare_chart_data(
    raw_data: Any,
    chart_type: str,
    **kwargs
) -> pd.DataFrame:
    """
    Prepare and format data for different chart types.
    
    Args:
        raw_data: Raw data from database or API
        chart_type: Type of chart to prepare data for
        **kwargs: Additional parameters
        
    Returns:
        Formatted DataFrame ready for charting
    """
    if isinstance(raw_data, dict):
        data = pd.DataFrame(raw_data)
    elif isinstance(raw_data, list):
        data = pd.DataFrame(raw_data)
    else:
        data = raw_data.copy()
    
    # Chart-specific data preparation
    if chart_type == 'time_series':
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp')
    
    elif chart_type == 'hourly_throughput':
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data = data.groupby('hour').agg({'container_count': 'sum'}).reset_index()
            data.rename(columns={'container_count': 'throughput'}, inplace=True)
    
    elif chart_type == 'peak_hours':
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['day_of_week'] = data['timestamp'].dt.day_name()
            data['hour'] = data['timestamp'].dt.hour
            data = data.groupby(['day_of_week', 'hour']).size().reset_index(name='activity_level')
    
    return data