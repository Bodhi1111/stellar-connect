#!/usr/bin/env python3
"""
Josh Vaughan Sales Console Dashboard - Updated to match existing sales data schema
Bloomberg-style sales analytics using real McAdams transcript data
"""

import streamlit as st
import pandas as pd
import sys
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import tempfile

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.stellar_crew import (
    run_crew,
    create_general_query_tasks,
    create_structured_record_tasks,
    create_email_recap_tasks
)
from src.ingestion import process_new_file

# Page Configuration
st.set_page_config(
    page_title="Josh Vaughan Sales Console",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bloomberg-style dark theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f2937 0%, #374151 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: #1f2937;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3b82f6;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-active { color: #10b981; font-weight: bold; }
    .status-pending { color: #f59e0b; font-weight: bold; }
    .status-closed-won { color: #3b82f6; font-weight: bold; }
    .status-closed-lost { color: #ef4444; font-weight: bold; }
    .live-indicator {
        animation: pulse 2s infinite;
        color: #10b981;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Load Combined Sales Data
@st.cache_data
def load_combined_sales_data():
    """Load both existing sales data and transcript data"""

    # Load existing sales data
    try:
        existing_df = pd.read_csv("existing_sales_data.csv")
        existing_df['source'] = 'existing'
        print(f"Loaded {len(existing_df)} existing sales records")
    except Exception as e:
        st.warning(f"Could not load existing sales data: {e}")
        existing_df = pd.DataFrame()

    # Load transcript sales data
    try:
        transcript_df = pd.read_csv("transcript_sales_data.csv")
        transcript_df['source'] = 'transcript'
        print(f"Loaded {len(transcript_df)} transcript sales records")
    except Exception as e:
        st.warning(f"Could not load transcript sales data: {e}")
        transcript_df = pd.DataFrame()

    # Combine datasets
    if not existing_df.empty and not transcript_df.empty:
        combined_df = pd.concat([existing_df, transcript_df], ignore_index=True)
    elif not existing_df.empty:
        combined_df = existing_df
    elif not transcript_df.empty:
        combined_df = transcript_df
    else:
        # Fallback data
        combined_df = pd.DataFrame([{
            'deal_id': 1,
            'Date': datetime.now().strftime('%Y/%m/%d'),
            'Lead ': 'Loading...',
            'Stage': 'Follow up',
            'Demo duration': 0,
            'Objection': '',
            'Reason': '',
            'Payment': 0,
            'Deposit': 0,
            'Notes': 'Loading sales data...',
            'source': 'fallback'
        }])

    # Standardize date format
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')

    # Extract estate values from notes for calculations
    combined_df['estate_value'] = combined_df.apply(extract_estate_value, axis=1)

    return combined_df

def extract_estate_value(row):
    """Extract estate value from notes or use payment as proxy"""
    import re

    if pd.notna(row['Notes']):
        # Look for dollar amounts in notes
        value_match = re.search(r'\$([0-9,]+(?:\.[0-9]+)?)', str(row['Notes']))
        if value_match:
            try:
                value_str = value_match.group(1).replace(',', '')
                value = float(value_str)
                if value > 10000:  # Likely an estate value
                    return int(value)
            except:
                pass

    # Use payment as a proxy (multiply by typical percentage)
    if pd.notna(row['Payment']) and row['Payment'] > 0:
        # Assuming 2% fee rate, reverse calculate estate value
        return int(row['Payment'] * 50)

    # Default fallback
    return 1000000

# Dashboard Header
st.markdown(f"""
<div class="main-header">
    <h1>📊 Josh Vaughan Sales Console</h1>
    <p>Estate Planning Sales Analytics & Performance Dashboard</p>
    <div style="display: flex; align-items: center; gap: 10px;">
        <span class="live-indicator">●</span>
        <span>LIVE</span>
        <span style="margin-left: 20px;">Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load sales data
df = load_combined_sales_data()

# Sidebar - Control Panel
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")

    # Real-time metrics refresh
    if st.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # Quick Actions
    st.markdown("### ⚡ Quick Actions")

    # File Upload
    st.markdown("#### 📁 Upload Transcript")
    uploaded_file = st.file_uploader(
        "Choose a transcript file",
        type=['txt', 'pdf', 'docx'],
        key="sidebar_upload"
    )

    if uploaded_file and st.button("🚀 Process", key="process_sidebar"):
        with st.spinner("Processing transcript..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                result = process_new_file(temp_file_path)
                os.unlink(temp_file_path)

                if result:
                    st.success(f"✅ {uploaded_file.name} processed successfully!")
                else:
                    st.error("❌ Processing failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown("---")

    # Filters
    st.markdown("### 🔍 Filters")

    # Data source filter
    sources = ['All'] + list(df['source'].unique())
    source_filter = st.selectbox("Data Source", sources, index=0)

    if source_filter != 'All':
        df = df[df['source'] == source_filter]

    # Fixed 4 stage options in logical order
    stage_options = ["Closed Won", "Follow up", "No Show", "Closed Lost"]
    stage_filter = st.multiselect(
        "Stage",
        options=stage_options,
        default=stage_options
    )

    value_range = st.slider(
        "Estate Value Range ($)",
        min_value=int(df['estate_value'].min()),
        max_value=int(df['estate_value'].max()),
        value=(int(df['estate_value'].min()), int(df['estate_value'].max())),
        step=100000,
        format="$%d"
    )

# Apply filters
filtered_df = df[
    (df['Stage'].isin(stage_filter)) &
    (df['estate_value'] >= value_range[0]) &
    (df['estate_value'] <= value_range[1])
]

# Main Dashboard Content
col1, col2, col3, col4 = st.columns(4)

# Key Metrics
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(filtered_df)}</div>
        <div class="metric-label">Total Clients</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_revenue = filtered_df['Payment'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${total_revenue:,.0f}</div>
        <div class="metric-label">Total Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    closed_won = len(filtered_df[filtered_df['Stage'] == 'Closed Won'])
    total_deals = len(filtered_df[filtered_df['Stage'].isin(['Closed Won', 'Closed Lost'])])
    close_rate = (closed_won / total_deals * 100) if total_deals > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{close_rate:.1f}%</div>
        <div class="metric-label">Close Rate</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_deal_size = filtered_df[filtered_df['Payment'] > 0]['Payment'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${avg_deal_size:.0f}</div>
        <div class="metric-label">Avg Deal Size</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Charts Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Revenue by Stage")
    stage_revenue = filtered_df.groupby('Stage')['Payment'].sum().sort_values(ascending=False)

    # Stage color mapping
    stage_colors = {
        "Closed Won": "#10b981",
        "Follow up": "#3b82f6",
        "Closed Lost": "#ef4444",
        "No Show": "#a16207"
    }

    fig_bar = px.bar(
        x=stage_revenue.index,
        y=stage_revenue.values,
        title="Revenue Distribution by Stage",
        color=stage_revenue.index,
        color_discrete_map=stage_colors
    )
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#374151',
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("### 🎯 Stage Distribution")
    stage_counts = filtered_df['Stage'].value_counts()

    # Use same stage colors for pie chart
    fig_pie = px.pie(
        values=stage_counts.values,
        names=stage_counts.index,
        title="Client Distribution by Stage",
        color=stage_counts.index,
        color_discrete_map=stage_colors
    )
    fig_pie.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#374151'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Performance Timeline
st.markdown("### 📊 Josh Vaughan Performance Timeline")
if len(filtered_df) > 0:
    # Create monthly performance
    filtered_df['month'] = filtered_df['Date'].dt.to_period('M').astype(str)
    monthly_performance = filtered_df.groupby('month').agg({
        'Payment': 'sum',
        'deal_id': 'count'
    }).rename(columns={'deal_id': 'deals'})

    if len(monthly_performance) > 0:
        fig_timeline = px.bar(
            x=monthly_performance.index,
            y=monthly_performance['Payment'],
            title="Monthly Revenue Performance",
            labels={'x': 'Month', 'y': 'Revenue ($)'},
            color=monthly_performance['deals'],
            color_continuous_scale='Blues'
        )
        fig_timeline.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#374151'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

# Win/Loss Analysis
col1, col2, col3 = st.columns(3)
with col1:
    closed_won_count = len(filtered_df[filtered_df['Stage'] == 'Closed Won'])
    st.metric("Closed Won", closed_won_count)
with col2:
    follow_up_count = len(filtered_df[filtered_df['Stage'] == 'Follow up'])
    st.metric("Follow Up", follow_up_count)
with col3:
    closed_lost_count = len(filtered_df[filtered_df['Stage'] == 'Closed Lost'])
    st.metric("Closed Lost", closed_lost_count)

st.markdown("---")

# Sales Records Table
st.markdown("### 📋 Sales Records")

# Search functionality
search_term = st.text_input("🔍 Search clients...", placeholder="Enter client name or notes")
if search_term:
    filtered_df = filtered_df[
        filtered_df['Lead '].str.contains(search_term, case=False, na=False) |
        filtered_df['Notes'].str.contains(search_term, case=False, na=False)
    ]

# Display the dataframe
display_df = filtered_df.copy()
display_df['Payment'] = display_df['Payment'].apply(lambda x: f"${x:,.0f}" if x > 0 else "$0")
display_df['Date'] = display_df['Date'].dt.strftime('%Y/%m/%d')

st.dataframe(
    display_df[['Date', 'Lead ', 'Stage', 'Demo duration', 'Objection', 'Payment', 'Notes', 'source']],
    use_container_width=True,
    height=400,
    column_config={
        "Date": st.column_config.TextColumn("Date", width="small"),
        "Lead ": st.column_config.TextColumn("Client Name", width="medium"),
        "Stage": st.column_config.TextColumn("Stage", width="small"),
        "Demo duration": st.column_config.NumberColumn("Duration (min)", width="small"),
        "Objection": st.column_config.TextColumn("Objection", width="medium"),
        "Payment": st.column_config.TextColumn("Revenue", width="small"),
        "Notes": st.column_config.TextColumn("Notes", width="large"),
        "source": st.column_config.TextColumn("Source", width="small")
    }
)

# AI Agent Actions Section
st.markdown("---")
st.markdown("### 🤖 AI Agent Actions")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📊 Generate Structured Record")
    client_name_record = st.selectbox(
        "Select Client for Record Generation:",
        options=[""] + filtered_df['Lead '].tolist(),
        key="record_client"
    )
    if st.button("🤖 Generate Record", type="primary") and client_name_record:
        with st.spinner(f"Generating structured record for {client_name_record}..."):
            try:
                tasks = create_structured_record_tasks(client_name_record)
                result = run_crew(tasks)
                st.json(json.loads(result))
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    st.markdown("#### ✉️ Draft Email Recap")
    client_name_email = st.selectbox(
        "Select Client for Email:",
        options=[""] + filtered_df['Lead '].tolist(),
        key="email_client"
    )
    if st.button("✉️ Draft Email", type="primary") and client_name_email:
        with st.spinner(f"Drafting email recap for {client_name_email}..."):
            try:
                tasks = create_email_recap_tasks(client_name_email)
                result = run_crew(tasks)
                st.text_area("Email Draft:", value=result, height=200)
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col3:
    st.markdown("#### 🔍 General Query")
    query = st.text_area("Ask about your sales data:", placeholder="What patterns do you see in successful deals?")
    if st.button("🔍 Ask AI", type="primary") and query:
        with st.spinner("Analyzing your data..."):
            try:
                tasks = create_general_query_tasks(query)
                result = run_crew(tasks)
                st.markdown(result)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>Josh Vaughan Sales Console | McAdams Estate Planning | AI-Powered Analytics</p>
    <p>📊 {len(filtered_df)} records displayed | Last update: {datetime.now().strftime("%H:%M:%S")}</p>
</div>
""", unsafe_allow_html=True)