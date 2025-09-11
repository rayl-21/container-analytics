"""
Container Analytics - Historical Trends

This page shows long-term trends and patterns, comparison tools,
statistical analysis, and supports data export.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional
import calendar
from scipy import stats
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from modules.database import queries
from modules.database.models import session_scope, Container, Detection, Image, Metric

# Configure page
st.set_page_config(
    page_title="Historical Trends - Container Analytics",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.historical-header {
    font-size: 2.2rem;
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 1rem;
}

.trend-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 0.5rem 0;
    text-align: center;
}

.comparison-section {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.75rem;
    border: 1px solid #dee2e6;
    margin: 1rem 0;
}

.statistical-insight {
    background-color: #e8f4f8;
    padding: 1rem;
    border-left: 4px solid #17a2b8;
    border-radius: 0.25rem;
    margin: 0.5rem 0;
}

.forecast-alert {
    background-color: #fff3cd;
    color: #856404;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #ffeaa7;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # 1-hour cache
def load_historical_data(months_back: int = 6) -> pd.DataFrame:
    """
    Load historical data for analysis.
    
    Args:
        months_back: Number of months of historical data to load
        
    Returns:
        Dictionary containing historical datasets
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=months_back * 30)
    
    # Convert to datetime for database queries
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Get metrics from database
    metrics = queries.get_metrics_by_date_range(start_datetime, end_datetime)
    
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
        
        # Create daily aggregates
        current_date = start_date
        while current_date <= end_date:
            day_metrics = metrics_by_date.get(current_date, [])
            
            if day_metrics:
                containers_in = sum(m.get('throughput', 0) // 2 for m in day_metrics)
                containers_out = sum(m.get('throughput', 0) - m.get('throughput', 0) // 2 for m in day_metrics)
                avg_dwell = np.mean([m.get('avg_dwell_time', 0) for m in day_metrics if m.get('avg_dwell_time')]) or 0
                peak_occupancy = max((m.get('container_count', 0) for m in day_metrics), default=0)
                detection_accuracy = np.mean([m.get('avg_confidence', 0) for m in day_metrics if m.get('avg_confidence')]) or 0.95
            else:
                containers_in = 0
                containers_out = 0
                avg_dwell = 0
                peak_occupancy = 0
                detection_accuracy = 0
            
            daily_data.append({
                'date': current_date,
                'containers_in': containers_in,
                'containers_out': containers_out,
                'net_flow': containers_in - containers_out,
                'total_volume': containers_in + containers_out,
                'avg_dwell_time': avg_dwell,
                'peak_occupancy': peak_occupancy,
                'detection_accuracy': detection_accuracy,
                'day_of_week': current_date.weekday(),
                'month': current_date.month,
                'week_of_year': current_date.isocalendar()[1],
                'is_weekend': current_date.weekday() >= 5
            })
            
            current_date += timedelta(days=1)
    else:
        # No data - create empty entries
        current_date = start_date
        while current_date <= end_date:
            daily_data.append({
                'date': current_date,
                'containers_in': 0,
                'containers_out': 0,
                'net_flow': 0,
                'total_volume': 0,
                'avg_dwell_time': 0,
                'peak_occupancy': 0,
                'detection_accuracy': 0,
                'day_of_week': current_date.weekday(),
                'month': current_date.month,
                'week_of_year': current_date.isocalendar()[1],
                'is_weekend': current_date.weekday() >= 5
            })
            current_date += timedelta(days=1)
    
    daily_df = pd.DataFrame(daily_data)
    # Convert date column to datetime
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Monthly aggregations
    monthly_df = daily_df.groupby(daily_df['date'].dt.to_period('M')).agg({
        'containers_in': 'sum',
        'containers_out': 'sum',
        'total_volume': 'sum',
        'avg_dwell_time': 'mean',
        'peak_occupancy': 'max',
        'detection_accuracy': 'mean'
    }).reset_index()
    monthly_df['date'] = monthly_df['date'].dt.to_timestamp()
    
    # Weekly patterns
    weekly_df = daily_df.groupby('day_of_week').agg({
        'containers_in': 'mean',
        'containers_out': 'mean',
        'total_volume': 'mean',
        'avg_dwell_time': 'mean'
    }).reset_index()
    
    # Hourly patterns from recent data
    recent_hourly_metrics = queries.get_metrics_by_date_range(
        datetime.now() - timedelta(days=30),
        datetime.now()
    )
    
    hourly_data = []
    if recent_hourly_metrics:
        hourly_groups = {}
        for metric in recent_hourly_metrics:
            hour = metric.get('hour', 0)
            if hour not in hourly_groups:
                hourly_groups[hour] = []
            hourly_groups[hour].append(metric)
        
        for hour in range(24):
            hour_metrics = hourly_groups.get(hour, [])
            if hour_metrics:
                avg_in = np.mean([m.get('throughput', 0) // 2 for m in hour_metrics])
                avg_out = np.mean([m.get('throughput', 0) - m.get('throughput', 0) // 2 for m in hour_metrics])
                avg_occ = np.mean([m.get('container_count', 0) for m in hour_metrics])
            else:
                avg_in = 0
                avg_out = 0
                avg_occ = 0
            
            hourly_data.append({
                'hour': hour,
                'avg_containers_in': avg_in,
                'avg_containers_out': avg_out,
                'avg_occupancy': avg_occ
            })
    else:
        for hour in range(24):
            hourly_data.append({
                'hour': hour,
                'avg_containers_in': 0,
                'avg_containers_out': 0,
                'avg_occupancy': 0
            })
    
    hourly_df = pd.DataFrame(hourly_data)
    
    # YoY comparison (simplified for now)
    yoy_data = {
        'current_year': daily_df[daily_df['date'].dt.year == end_date.year]['containers_in'].sum() if not daily_df.empty else 0,
        'previous_year': 0,  # Would need previous year data from database
        'yoy_growth': 0
    }
    
    return {
        'daily_df': daily_df,
        'monthly_df': monthly_df,
        'weekly_df': weekly_df,
        'hourly_df': hourly_df,
        'yoy_data': yoy_data,
        'total_days': len(daily_df),
        'date_range': f"{start_date} to {end_date}"
    }


def perform_statistical_analysis(df: pd.DataFrame) -> Dict:
    """Perform statistical analysis on historical data."""
    
    # Basic statistics
    stats_dict = {
        'total_volume': {
            'mean': df['total_volume'].mean(),
            'median': df['total_volume'].median(),
            'std': df['total_volume'].std(),
            'min': df['total_volume'].min(),
            'max': df['total_volume'].max(),
            'trend_slope': None,
            'trend_r2': None
        }
    }
    
    # Linear regression for trend analysis
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['total_volume'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    stats_dict['total_volume']['trend_slope'] = model.coef_[0]
    stats_dict['total_volume']['trend_r2'] = model.score(X, y)
    
    # Correlation analysis
    correlations = df[['containers_in', 'containers_out', 'avg_dwell_time', 'peak_occupancy']].corr()
    
    # Seasonality analysis
    df_copy = df.copy()
    df_copy['day_of_year'] = df_copy['date'].dt.dayofyear
    seasonal_corr = df_copy[['day_of_year', 'total_volume']].corr().iloc[0, 1]
    
    # Weekly patterns
    weekday_avg = df.groupby('day_of_week')['total_volume'].mean()
    weekend_vs_weekday = {
        'weekday_avg': df[df['is_weekend'] == False]['total_volume'].mean(),
        'weekend_avg': df[df['is_weekend'] == True]['total_volume'].mean()
    }
    weekend_vs_weekday['difference_pct'] = (
        (weekend_vs_weekday['weekend_avg'] - weekend_vs_weekday['weekday_avg']) / 
        weekend_vs_weekday['weekday_avg'] * 100
    )
    
    return {
        'basic_stats': stats_dict,
        'correlations': correlations,
        'seasonal_correlation': seasonal_corr,
        'weekday_patterns': weekday_avg,
        'weekend_vs_weekday': weekend_vs_weekday
    }


def create_trend_analysis_chart(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Create a comprehensive trend analysis chart."""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{title} - Time Series', f'{title} - Moving Average & Trend'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )
    
    # Raw time series
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df[metric],
            mode='lines+markers',
            name='Daily Values',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Moving averages
    df_copy = df.copy()
    df_copy['ma_7'] = df_copy[metric].rolling(window=7).mean()
    df_copy['ma_30'] = df_copy[metric].rolling(window=30).mean()
    
    fig.add_trace(
        go.Scatter(
            x=df_copy['date'],
            y=df_copy['ma_7'],
            mode='lines',
            name='7-day MA',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_copy['date'],
            y=df_copy['ma_30'],
            mode='lines',
            name='30-day MA',
            line=dict(color='#2ca02c', width=3)
        ),
        row=2, col=1
    )
    
    # Linear trend
    X = np.arange(len(df)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, df[metric])
    trend_line = model.predict(X)
    
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=trend_line,
            mode='lines',
            name='Linear Trend',
            line=dict(color='#d62728', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text=title, row=1, col=1)
    fig.update_yaxes(title_text=title, row=2, col=1)
    
    return fig


def create_seasonal_analysis(df: pd.DataFrame) -> go.Figure:
    """Create seasonal analysis charts."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Monthly Averages', 
            'Day of Week Patterns',
            'Hourly Patterns', 
            'Weekend vs Weekday'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )
    
    # Monthly averages
    monthly_avg = df.groupby('month')['total_volume'].mean()
    month_names = [calendar.month_abbr[i] for i in monthly_avg.index]
    
    fig.add_trace(
        go.Bar(x=month_names, y=monthly_avg.values, name='Monthly Avg', showlegend=False),
        row=1, col=1
    )
    
    # Day of week patterns
    dow_avg = df.groupby('day_of_week')['total_volume'].mean()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig.add_trace(
        go.Bar(x=dow_names, y=dow_avg.values, name='Day of Week', showlegend=False),
        row=1, col=2
    )
    
    # Hourly patterns (get from data if available)
    hours = list(range(24))
    # Get hourly data from the historical data function
    recent_hourly_metrics = queries.get_metrics_by_date_range(
        datetime.now() - timedelta(days=7),
        datetime.now()
    )
    
    hourly_pattern = []
    if recent_hourly_metrics:
        hourly_groups = {}
        for metric in recent_hourly_metrics:
            hour = metric.get('hour', 0)
            if hour not in hourly_groups:
                hourly_groups[hour] = []
            hourly_groups[hour].append(metric)
        
        for hour in hours:
            hour_metrics = hourly_groups.get(hour, [])
            if hour_metrics:
                avg_volume = np.mean([m.get('throughput', 0) for m in hour_metrics])
            else:
                avg_volume = 0
            hourly_pattern.append(avg_volume)
    else:
        # If no data, use zeros
        hourly_pattern = [0] * 24
    
    fig.add_trace(
        go.Scatter(
            x=hours, 
            y=hourly_pattern, 
            mode='lines+markers', 
            name='Hourly Pattern',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Weekend vs Weekday
    weekend_comparison = df.groupby('is_weekend')['total_volume'].mean()
    
    fig.add_trace(
        go.Bar(
            x=['Weekday', 'Weekend'], 
            y=weekend_comparison.values,
            name='Weekend vs Weekday',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=False)
    
    return fig


def create_comparison_interface(df: pd.DataFrame):
    """Create comparison interface for different time periods."""
    
    st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Period Comparison Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Period 1**")
        period1_start = st.date_input(
            "Start Date", 
            value=df['date'].min().date(),
            key="p1_start"
        )
        period1_end = st.date_input(
            "End Date", 
            value=df['date'].min().date() + timedelta(days=30),
            key="p1_end"
        )
    
    with col2:
        st.markdown("**Period 2**")
        period2_start = st.date_input(
            "Start Date", 
            value=df['date'].max().date() - timedelta(days=30),
            key="p2_start"
        )
        period2_end = st.date_input(
            "End Date", 
            value=df['date'].max().date(),
            key="p2_end"
        )
    
    if st.button("📊 Compare Periods"):
        # Filter data for each period
        period1_data = df[
            (df['date'].dt.date >= period1_start) & 
            (df['date'].dt.date <= period1_end)
        ]
        period2_data = df[
            (df['date'].dt.date >= period2_start) & 
            (df['date'].dt.date <= period2_end)
        ]
        
        if len(period1_data) == 0 or len(period2_data) == 0:
            st.error("No data found for one or both periods.")
            return
        
        # Calculate comparison metrics
        metrics = ['total_volume', 'containers_in', 'containers_out', 'avg_dwell_time']
        
        comparison_data = []
        for metric in metrics:
            p1_avg = period1_data[metric].mean()
            p2_avg = period2_data[metric].mean()
            change = ((p2_avg - p1_avg) / p1_avg) * 100 if p1_avg != 0 else 0
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Period 1': f"{p1_avg:.1f}",
                'Period 2': f"{p2_avg:.1f}",
                'Change (%)': f"{change:+.1f}%",
                'Change_numeric': change
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.markdown("##### Comparison Results")
        
        # Color-code the changes
        def color_change(val):
            if isinstance(val, str) and '%' in val:
                num = float(val.replace('%', '').replace('+', ''))
                if num > 5:
                    return 'background-color: #d4edda; color: #155724;'
                elif num < -5:
                    return 'background-color: #f8d7da; color: #721c24;'
            return ''
        
        styled_df = comparison_df[['Metric', 'Period 1', 'Period 2', 'Change (%)']].copy()
        st.dataframe(
            styled_df.style.applymap(color_change, subset=['Change (%)']),
            use_container_width=True
        )
        
        # Visualization of comparison
        fig_comp = go.Figure()
        
        for i, metric in enumerate(metrics):
            p1_avg = period1_data[metric].mean()
            p2_avg = period2_data[metric].mean()
            
            fig_comp.add_trace(go.Bar(
                name=f'Period 1 ({period1_start} to {period1_end})',
                x=[metric],
                y=[p1_avg],
                showlegend=(i == 0)
            ))
            
            fig_comp.add_trace(go.Bar(
                name=f'Period 2 ({period2_start} to {period2_end})',
                x=[metric],
                y=[p2_avg],
                showlegend=(i == 0)
            ))
        
        fig_comp.update_layout(
            title="Period Comparison",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main historical trends dashboard."""
    
    # Header
    st.markdown('<div class="historical-header">📈 Historical Trends & Analysis</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.title("Historical Analysis")
        
        # Time range selection
        st.subheader("📅 Analysis Period")
        months_back = st.selectbox(
            "Historical Period",
            options=[3, 6, 12, 24, 36],
            index=2,
            format_func=lambda x: f"Last {x} months"
        )
        
        # Analysis options
        st.subheader("📊 Analysis Options")
        show_trends = st.checkbox("Show trend lines", True)
        show_seasonality = st.checkbox("Seasonal analysis", True)
        show_forecasts = st.checkbox("Predictive forecasts", False)
        include_statistics = st.checkbox("Statistical insights", True)
        
        # Metrics selection
        st.subheader("📈 Metrics to Analyze")
        selected_metrics = st.multiselect(
            "Select Metrics",
            options=['total_volume', 'containers_in', 'containers_out', 'avg_dwell_time', 'peak_occupancy'],
            default=['total_volume', 'containers_in', 'containers_out'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Refresh data
        if st.button("🔄 Refresh Analysis"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.markdown("### Navigation")
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("app.py")
    
    # Load historical data
    with st.spinner("Loading historical data..."):
        data = load_historical_data(months_back)
        stats_analysis = perform_statistical_analysis(data['daily_df'])
    
    # Key historical insights
    st.markdown("### 📊 Historical Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_containers = data['daily_df']['total_volume'].sum()
    avg_daily = data['daily_df']['total_volume'].mean()
    trend_slope = stats_analysis['basic_stats']['total_volume']['trend_slope']
    peak_day = data['daily_df'].loc[data['daily_df']['total_volume'].idxmax()]
    
    with col1:
        st.markdown(f"""
        <div class="trend-card">
            <h3>{total_containers:,}</h3>
            <p>Total Containers</p>
            <small>Over {data['total_days']} days</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="trend-card">
            <h3>{avg_daily:.1f}</h3>
            <p>Daily Average</p>
            <small>Containers per day</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        trend_direction = "📈" if trend_slope > 0 else "📉" if trend_slope < 0 else "➡️"
        st.markdown(f"""
        <div class="trend-card">
            <h3>{trend_direction} {abs(trend_slope):.2f}</h3>
            <p>Daily Trend</p>
            <small>Containers/day change</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="trend-card">
            <h3>{peak_day['total_volume']:.0f}</h3>
            <p>Peak Day</p>
            <small>{peak_day['date'].strftime('%Y-%m-%d')}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Main trend analysis
    if selected_metrics:
        for metric in selected_metrics:
            if metric in data['daily_df'].columns:
                st.markdown(f"### 📈 {metric.replace('_', ' ').title()} Trend Analysis")
                
                trend_chart = create_trend_analysis_chart(
                    data['daily_df'], 
                    metric, 
                    metric.replace('_', ' ').title()
                )
                st.plotly_chart(trend_chart, use_container_width=True)
                
                # Statistical insights for this metric
                if include_statistics and metric in ['total_volume']:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="statistical-insight">
                            <strong>Mean:</strong> {stats_analysis['basic_stats'][metric]['mean']:.1f}<br>
                            <strong>Median:</strong> {stats_analysis['basic_stats'][metric]['median']:.1f}<br>
                            <strong>Std Dev:</strong> {stats_analysis['basic_stats'][metric]['std']:.1f}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="statistical-insight">
                            <strong>Min:</strong> {stats_analysis['basic_stats'][metric]['min']:.1f}<br>
                            <strong>Max:</strong> {stats_analysis['basic_stats'][metric]['max']:.1f}<br>
                            <strong>Range:</strong> {stats_analysis['basic_stats'][metric]['max'] - stats_analysis['basic_stats'][metric]['min']:.1f}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        r2_score = stats_analysis['basic_stats'][metric]['trend_r2']
                        trend_strength = "Strong" if r2_score > 0.7 else "Moderate" if r2_score > 0.3 else "Weak"
                        st.markdown(f"""
                        <div class="statistical-insight">
                            <strong>Trend R²:</strong> {r2_score:.3f}<br>
                            <strong>Trend Strength:</strong> {trend_strength}<br>
                            <strong>Slope:</strong> {stats_analysis['basic_stats'][metric]['trend_slope']:.3f}
                        </div>
                        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Seasonal analysis
    if show_seasonality:
        st.markdown("### 🌞 Seasonal & Pattern Analysis")
        seasonal_chart = create_seasonal_analysis(data['daily_df'])
        st.plotly_chart(seasonal_chart, use_container_width=True)
        
        # Weekend vs Weekday insights
        weekend_stats = stats_analysis['weekend_vs_weekday']
        if weekend_stats['difference_pct'] > 10:
            insight_type = "higher"
            insight_color = "#28a745"
        elif weekend_stats['difference_pct'] < -10:
            insight_type = "lower"
            insight_color = "#dc3545"
        else:
            insight_type = "similar"
            insight_color = "#17a2b8"
        
        st.markdown(f"""
        <div class="statistical-insight" style="border-left-color: {insight_color};">
            <strong>Weekend Pattern:</strong> Weekend activity is {abs(weekend_stats['difference_pct']):.1f}% {insight_type} 
            than weekday activity. Weekend average: {weekend_stats['weekend_avg']:.1f} containers/day, 
            Weekday average: {weekend_stats['weekday_avg']:.1f} containers/day.
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Period comparison
    st.markdown("### ⚖️ Period Comparison")
    create_comparison_interface(data['daily_df'])
    
    st.divider()
    
    # Forecasting (if enabled)
    if show_forecasts:
        st.markdown("### 🔮 Predictive Forecasting")
        
        st.markdown("""
        <div class="forecast-alert">
            <strong>⚠️ Forecast Disclaimer:</strong> These are statistical projections based on historical trends. 
            Actual results may vary due to external factors not captured in historical data.
        </div>
        """, unsafe_allow_html=True)
        
        # Simple linear forecast for next 30 days
        X = np.arange(len(data['daily_df'])).reshape(-1, 1)
        y = data['daily_df']['total_volume'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast future values
        future_days = 30
        future_X = np.arange(len(data['daily_df']), len(data['daily_df']) + future_days).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Create forecast dates
        last_date = data['daily_df']['date'].max()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
        
        # Plot historical + forecast
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=data['daily_df']['date'],
            y=data['daily_df']['total_volume'],
            mode='lines',
            name='Historical Data',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecast
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Add confidence bands (simple approach)
        std_error = np.std(y - model.predict(X))
        upper_bound = forecast + 1.96 * std_error
        lower_bound = forecast - 1.96 * std_error
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='95% Confidence Interval',
            showlegend=False
        ))
        
        fig_forecast.update_layout(
            title=f"30-Day Volume Forecast (R² = {model.score(X, y):.3f})",
            xaxis_title="Date",
            yaxis_title="Container Volume",
            height=500
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast summary
        forecast_avg = np.mean(forecast)
        historical_avg = data['daily_df']['total_volume'].tail(30).mean()
        forecast_change = ((forecast_avg - historical_avg) / historical_avg) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Forecast Daily Avg", f"{forecast_avg:.1f}", f"{forecast_change:+.1f}%")
        
        with col2:
            st.metric("Forecast Total (30 days)", f"{np.sum(forecast):,.0f}")
        
        with col3:
            confidence_level = "High" if model.score(X, y) > 0.8 else "Medium" if model.score(X, y) > 0.5 else "Low"
            st.metric("Forecast Confidence", confidence_level, f"R² = {model.score(X, y):.3f}")
    
    # Data export section
    st.divider()
    st.markdown("### 📤 Export Historical Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_daily = data['daily_df'].to_csv(index=False)
        st.download_button(
            "📄 Download Daily Data (CSV)",
            csv_daily,
            file_name=f"historical_daily_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        csv_monthly = data['monthly_df'].to_csv(index=False)
        st.download_button(
            "📊 Download Monthly Summary (CSV)",
            csv_monthly,
            file_name=f"historical_monthly_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Create analysis summary
        summary_report = {
            'analysis_date': datetime.now().isoformat(),
            'data_period': data['date_range'],
            'total_days_analyzed': data['total_days'],
            'key_statistics': stats_analysis['basic_stats'],
            'seasonal_correlation': stats_analysis['seasonal_correlation'],
            'weekend_vs_weekday': stats_analysis['weekend_vs_weekday']
        }
        
        import json
        json_summary = json.dumps(summary_report, indent=2, default=str)
        st.download_button(
            "📈 Download Analysis Report (JSON)",
            json_summary,
            file_name=f"historical_analysis_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Historical analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        Analysis period: {data['date_range']} | Data points: {data['total_days']} days | 
        Metrics analyzed: {len(selected_metrics)}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()