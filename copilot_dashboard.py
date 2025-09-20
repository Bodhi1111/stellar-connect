#!/usr/bin/env python3
"""
Stellar Connect Sales Copilot Dashboard
Layout: Chat Interface → Sales Spreadsheet → Analytics
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
    page_title="Stellar Connect Sales Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for copilot interface
st.markdown("""
<style>
    .copilot-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
    }
    .chat-container {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        min-height: 200px;
    }
    .sales-spreadsheet {
        background: white;
        border: 2px solid #3b82f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .analytics-section {
        background: #f9fafb;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3b82f6;
    }
    .live-indicator {
        animation: pulse 2s infinite;
        color: #10b981;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
    }

    /* Stage Color Coding */
    .stage-closed-won {
        background-color: #10b981 !important;
        color: white !important;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .stage-follow-up {
        background-color: #3b82f6 !important;
        color: white !important;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .stage-closed-lost {
        background-color: #ef4444 !important;
        color: white !important;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .stage-no-show {
        background-color: #a16207 !important;
        color: white !important;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Stage Color Mapping
def get_stage_color_mapping():
    """Return color mapping for stages"""
    return {
        "Closed Won": {"color": "#10b981", "class": "stage-closed-won"},
        "Follow up": {"color": "#3b82f6", "class": "stage-follow-up"},
        "Closed Lost": {"color": "#ef4444", "class": "stage-closed-lost"},
        "No Show": {"color": "#a16207", "class": "stage-no-show"}
    }

def format_stage_with_color(stage):
    """Format stage name with color badge"""
    color_mapping = get_stage_color_mapping()
    if stage in color_mapping:
        color = color_mapping[stage]["color"]
        return f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.8em;">{stage}</span>'
    return stage

# Load Combined Sales Data
@st.cache_data
def load_all_sales_data():
    """Load all sales data sources"""

    data_sources = []
    debug_info = []

    # Try to load existing sales data from multiple locations
    existing_files = [
        "existing_sales_data.csv",
        "/Users/joshuavaughan/Downloads/SALES REPORT DASHBOARD 259a8c5acfd980bca654e78f043e87fc.csv"
    ]

    existing_df = pd.DataFrame()
    for file_path in existing_files:
        try:
            existing_df = pd.read_csv(file_path)
            existing_df['source'] = 'existing'
            debug_info.append(f"✅ Loaded {len(existing_df)} existing records from {file_path}")
            break
        except Exception as e:
            debug_info.append(f"❌ Failed to load {file_path}: {str(e)}")

    if not existing_df.empty:
        data_sources.append(existing_df)

    # Try to load granular AI data
    try:
        granular_df = pd.read_csv("granular_sales_data.csv")
        granular_df['source'] = 'AI_extracted'
        data_sources.append(granular_df)
        debug_info.append(f"✅ Loaded {len(granular_df)} AI-extracted records")
    except Exception as e:
        debug_info.append(f"❌ No AI-extracted data: {str(e)}")

    # Try to load transcript data
    try:
        transcript_df = pd.read_csv("transcript_sales_data.csv")
        transcript_df['source'] = 'transcript'
        data_sources.append(transcript_df)
        debug_info.append(f"✅ Loaded {len(transcript_df)} transcript records")
    except Exception as e:
        debug_info.append(f"❌ No transcript data: {str(e)}")

    # Combine all data sources
    if data_sources:
        combined_df = pd.concat(data_sources, ignore_index=True, sort=False)
        debug_info.append(f"📊 Combined total: {len(combined_df)} records")
    else:
        # Create fallback data
        combined_df = pd.DataFrame([
            {
                'deal_id': 1,
                'Date': '2025/09/19',
                'Lead ': 'Sample Client',
                'Stage': 'Follow up',
                'Demo duration': 60,
                'Objection': 'Spouse',
                'Reason': 'Needs to discuss with spouse',
                'Payment': 0,
                'Deposit': 0,
                'Notes': 'Sample data - please load your sales data',
                'source': 'fallback'
            }
        ])
        debug_info.append("⚠️ Using fallback sample data")

    # Standardize date format
    if 'Date' in combined_df.columns:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')

    # Store debug info in session state for display
    st.session_state.data_debug = debug_info

    return combined_df

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_query' not in st.session_state:
    st.session_state.processing_query = False

# Load data
df = load_all_sales_data()

# Show data loading status in sidebar
if hasattr(st.session_state, 'data_debug'):
    with st.sidebar:
        st.markdown("### 📊 Data Status")
        for info in st.session_state.data_debug:
            st.markdown(f"- {info}")
        if st.button("🔄 Force Refresh Data"):
            st.cache_data.clear()
            st.rerun()

# 1. STELLAR CONNECT SALES COPILOT HEADER
st.markdown(f"""
<div class="copilot-header">
    <h1>🤖 Stellar Connect Sales Copilot</h1>
    <p>Local Agentic RAG Assistant | AI-Powered Sales Intelligence</p>
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 10px;">
        <span class="live-indicator">●</span>
        <span>COPILOT ACTIVE</span>
        <span style="margin-left: 20px;">Last Updated: {datetime.now().strftime("%H:%M:%S")}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. CHAT INTERFACE
st.markdown("### 💬 Ask Your Sales Copilot")

# Chat input
user_query = st.text_input(
    "Ask about your clients, sales data, or estate planning insights:",
    placeholder="What patterns do you see in successful deals? Who are my highest value prospects?",
    key="copilot_query"
)

# Process query
if user_query and not st.session_state.processing_query:
    st.session_state.processing_query = True

    with st.spinner("🤖 Copilot analyzing your data..."):
        try:
            # Add user message to chat
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            # Get AI response
            tasks = create_general_query_tasks(user_query)
            ai_response = run_crew(tasks)

            # Add AI response to chat
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"I encountered an error: {str(e)}. Please try rephrasing your question.",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

    st.session_state.processing_query = False
    st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.markdown("#### 💬 Conversation History")
    for i, message in enumerate(reversed(st.session_state.chat_history[-6:])):  # Show last 6 messages
        if message["role"] == "user":
            st.markdown(f"**You ({message['timestamp']}):** {message['content']}")
        else:
            st.markdown(f"**🤖 Copilot ({message['timestamp']}):** {message['content']}")

        if i < len(st.session_state.chat_history[-6:]) - 1:
            st.markdown("---")

# Clear chat button
if st.button("🗑️ Clear Chat History", key="clear_chat"):
    st.session_state.chat_history = []
    st.rerun()

# 3. OPERATIONAL SALES SPREADSHEET
st.markdown("## 📊 Your Daily Sales Dashboard")

# Quick stats row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_clients = len(df)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_clients}</div>
        <div style="color: #6b7280; font-size: 0.875rem;">Total Clients</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_revenue = df['Payment'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${total_revenue:,.0f}</div>
        <div style="color: #6b7280; font-size: 0.875rem;">Total Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    active_deals = len(df[df['Stage'] == 'Follow up'])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{active_deals}</div>
        <div style="color: #6b7280; font-size: 0.875rem;">Active Deals</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    closed_won = len(df[df['Stage'] == 'Closed Won'])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{closed_won}</div>
        <div style="color: #6b7280; font-size: 0.875rem;">Closed Won</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    avg_deal = df[df['Payment'] > 0]['Payment'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${avg_deal:.0f}</div>
        <div style="color: #6b7280; font-size: 0.875rem;">Avg Deal Size</div>
    </div>
    """, unsafe_allow_html=True)

# Filters for the spreadsheet
col1, col2, col3 = st.columns(3)

with col1:
    # Fixed 4 stage options in logical order
    stage_options = ["Closed Won", "Follow up", "No Show", "Closed Lost"]

    # Initialize session state for stage filter if not exists
    if "stage_filter_state" not in st.session_state:
        st.session_state.stage_filter_state = stage_options

    stage_filter = st.multiselect(
        "Filter by Stage:",
        options=stage_options,
        default=st.session_state.stage_filter_state,
        key="main_stage_filter"
    )

    # Update session state when filter changes
    if stage_filter != st.session_state.stage_filter_state:
        st.session_state.stage_filter_state = stage_filter

with col2:
    # Date range filter
    date_filter = st.selectbox(
        "Time Period:",
        ["All Time", "Last 30 Days", "Last 90 Days", "This Year"],
        key="date_filter"
    )

with col3:
    # Search
    search_term = st.text_input("🔍 Search clients:", placeholder="Client name, location, notes...")

# Apply filters
# If no stages selected, show all stages
if not stage_filter:
    stage_filter = stage_options

filtered_df = df[df['Stage'].isin(stage_filter)].copy()

if date_filter != "All Time":
    today = datetime.now()
    if date_filter == "Last 30 Days":
        cutoff = today - timedelta(days=30)
    elif date_filter == "Last 90 Days":
        cutoff = today - timedelta(days=90)
    else:  # This Year
        cutoff = datetime(today.year, 1, 1)

    filtered_df = filtered_df[filtered_df['Date'] >= cutoff]

if search_term:
    filtered_df = filtered_df[
        filtered_df['Lead '].str.contains(search_term, case=False, na=False) |
        filtered_df.get('Notes', pd.Series(dtype='object')).str.contains(search_term, case=False, na=False) |
        filtered_df.get('client_location', pd.Series(dtype='object')).str.contains(search_term, case=False, na=False)
    ]

# Core operational columns for daily use
base_columns = ['Date', 'Lead ', 'Stage', 'Demo duration', 'Objection', 'Reason', 'Payment', 'Deposit']
granular_columns = [
    'client_location', 'marital_status', 'client_age', 'spouse_age',
    'num_beneficiaries', 'estate_value', 'real_estate_count',
    'investment_assets', 'current_estate_docs', 'risk_factors'
]
social_email_columns = [
    'email_sent', 'linkedin_connected', 'follow_up_scheduled',
    'email_recap_sent', 'social_media_engagement'
]
notes_column = ['Notes']

# Build operational columns list
operational_columns = base_columns.copy()

# Add granular columns if they exist
for col in granular_columns:
    if col in filtered_df.columns:
        operational_columns.append(col)

# Add social/email columns if they exist
for col in social_email_columns:
    if col in filtered_df.columns:
        operational_columns.append(col)

# Add notes at the end
if 'Notes' in filtered_df.columns:
    operational_columns.extend(notes_column)

# Filter to only include columns that actually exist in the dataframe
available_columns = [col for col in operational_columns if col in filtered_df.columns]

# Format display data
display_df = filtered_df[available_columns].copy()

# Handle missing/null values - replace various null representations
display_df = display_df.replace(['None', 'none', 'null', 'NULL', 'nan', 'NaN'], "")
display_df = display_df.fillna("")

# Clean up text columns to remove "None" strings
for col in display_df.columns:
    if display_df[col].dtype == 'object':  # Text columns
        display_df[col] = display_df[col].astype(str).replace('None', '').replace('nan', '')

# Format specific columns - handle date formatting safely
try:
    display_df['Date'] = pd.to_datetime(display_df['Date'], errors='coerce').dt.strftime('%Y/%m/%d')
except Exception as e:
    # Fallback: leave dates as-is if formatting fails
    print(f"Date formatting warning: {e}")

# Fix Payment column formatting - handle both string and numeric values
def format_payment(x):
    if pd.isna(x) or x == "" or str(x).lower() in ['none', 'nan', 'null']:
        return "$0"
    try:
        # Convert to string first, then clean
        x_str = str(x)
        # Remove any existing $ and , symbols
        x_clean = x_str.replace('$', '').replace(',', '').strip()
        if x_clean == '' or x_clean.lower() in ['nan', 'none', 'null']:
            return "$0"
        x_float = float(x_clean)
        return f"${x_float:,.0f}" if x_float > 0 else "$0"
    except (ValueError, TypeError):
        return "$0"

# Apply payment formatting safely
if 'Payment' in display_df.columns:
    display_df['Payment'] = display_df['Payment'].apply(format_payment)

# Format Deposit column the same way
if 'Deposit' in display_df.columns:
    display_df['Deposit'] = display_df['Deposit'].apply(format_payment)

# Add stage visual indicators
def format_stage_visual(stage):
    if pd.isna(stage) or stage == "":
        return "⚪ Unknown"

    stage_icons = {
        "Closed Won": "🟢",
        "Follow up": "🔵",
        "Closed Lost": "🔴",
        "No Show": "🟤"
    }
    icon = stage_icons.get(str(stage), "⚪")
    return f"{icon} {stage}"

# Apply stage formatting - handle null values
if 'Stage' in display_df.columns:
    display_df['Stage'] = display_df['Stage'].apply(format_stage_visual)

# Main sales spreadsheet
st.markdown("#### 📋 Sales Records (Editable)")
st.dataframe(
    display_df,
    use_container_width=True,
    height=500,
    column_config={
        "Date": st.column_config.DateColumn("Date", width="small"),
        "Lead ": st.column_config.TextColumn("Client Name", width="medium"),
        "Stage": st.column_config.TextColumn("Stage", width="small"),
        "Demo duration": st.column_config.NumberColumn("Duration", width="small"),
        "Objection": st.column_config.TextColumn("Objection", width="medium"),
        "Reason": st.column_config.TextColumn("Reason", width="medium"),
        "Payment": st.column_config.TextColumn("Revenue", width="small"),
        "Deposit": st.column_config.TextColumn("Deposit", width="small"),
        "client_location": st.column_config.TextColumn("Location", width="small"),
        "marital_status": st.column_config.TextColumn("Marital", width="small"),
        "client_age": st.column_config.NumberColumn("Age", width="small"),
        "spouse_age": st.column_config.NumberColumn("Spouse Age", width="small"),
        "num_beneficiaries": st.column_config.NumberColumn("Beneficiaries", width="small"),
        "estate_value": st.column_config.NumberColumn("Estate Value", width="medium", format="$%.0f"),
        "real_estate_count": st.column_config.NumberColumn("Properties", width="small"),
        "investment_assets": st.column_config.NumberColumn("Investments", width="medium", format="$%.0f"),
        "current_estate_docs": st.column_config.TextColumn("Existing Docs", width="medium"),
        "risk_factors": st.column_config.TextColumn("Risk Factors", width="medium"),
        "email_sent": st.column_config.CheckboxColumn("Email Sent", width="small"),
        "linkedin_connected": st.column_config.CheckboxColumn("LinkedIn", width="small"),
        "follow_up_scheduled": st.column_config.CheckboxColumn("Follow-up", width="small"),
        "email_recap_sent": st.column_config.CheckboxColumn("Recap Sent", width="small"),
        "social_media_engagement": st.column_config.TextColumn("Social Media", width="small"),
        "Notes": st.column_config.TextColumn("Notes", width="large")
    }
)

# Quick actions for the spreadsheet
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📥 Add New Client", type="secondary"):
        st.info("New client form would open here")

with col2:
    if st.button("📤 Export to CSV", type="secondary"):
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("🔄 Refresh Data", type="secondary"):
        st.cache_data.clear()
        st.rerun()

with col4:
    if st.button("📧 Email Reports", type="secondary"):
        st.info("Email integration would trigger here")

# 4. ANALYTICS SECTION (Below the operational spreadsheet)
st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
st.markdown("## 📈 Sales Analytics & Insights")

# Charts row
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 Revenue by Stage")
    stage_revenue = filtered_df.groupby('Stage')['Payment'].sum()
    if len(stage_revenue) > 0:
        # Get stage colors
        color_mapping = get_stage_color_mapping()
        stage_colors = [color_mapping.get(stage, {"color": "#6b7280"})["color"] for stage in stage_revenue.index]

        fig_revenue = px.bar(
            x=stage_revenue.index,
            y=stage_revenue.values,
            title="Revenue Distribution by Stage",
            color=stage_revenue.index,
            color_discrete_map={stage: color_mapping.get(stage, {"color": "#6b7280"})["color"] for stage in stage_revenue.index}
        )
        fig_revenue.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_revenue, use_container_width=True)

with col2:
    st.markdown("#### 🎯 Pipeline Distribution")
    stage_counts = filtered_df['Stage'].value_counts()
    if len(stage_counts) > 0:
        # Get stage colors for pie chart
        color_mapping = get_stage_color_mapping()

        fig_pipeline = px.pie(
            values=stage_counts.values,
            names=stage_counts.index,
            title="Client Distribution by Stage",
            color=stage_counts.index,
            color_discrete_map={stage: color_mapping.get(stage, {"color": "#6b7280"})["color"] for stage in stage_counts.index}
        )
        fig_pipeline.update_layout(height=400)
        st.plotly_chart(fig_pipeline, use_container_width=True)

# Granular analytics (if available)
if 'client_location' in filtered_df.columns:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🗺️ Geographic Distribution")
        location_counts = filtered_df['client_location'].value_counts().head(10)
        if len(location_counts) > 0:
            fig_geo = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title="Top 10 Client Locations"
            )
            fig_geo.update_layout(height=400)
            st.plotly_chart(fig_geo, use_container_width=True)

    with col2:
        if 'client_age' in filtered_df.columns:
            st.markdown("#### 👥 Age Demographics")
            numeric_ages = pd.to_numeric(filtered_df['client_age'], errors='coerce').dropna()
            if len(numeric_ages) > 0:
                fig_age = px.histogram(
                    numeric_ages,
                    nbins=8,
                    title="Client Age Distribution"
                )
                fig_age.update_layout(height=400)
                st.plotly_chart(fig_age, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer with status
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>🤖 Stellar Connect Sales Copilot | AI-Enhanced Sales Intelligence</p>
    <p>📊 {len(filtered_df)} records displayed | 🤖 Copilot active | Last update: {datetime.now().strftime("%H:%M:%S")}</p>
</div>
""", unsafe_allow_html=True)