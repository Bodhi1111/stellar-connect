#!/usr/bin/env python3
"""
Enhanced Granular Sales Dashboard - With AI-Extracted Client Details
Bloomberg-style analytics with detailed client demographics and estate information
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
    page_title="Josh Vaughan Granular Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .granular-section {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
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
</style>
""", unsafe_allow_html=True)

# Load Enhanced Sales Data
@st.cache_data
def load_granular_sales_data():
    """Load granular sales data with detailed client information"""

    try:
        # Try to load granular data first
        granular_df = pd.read_csv("granular_sales_data.csv")
        granular_df['source'] = 'granular_ai'
        print(f"Loaded {len(granular_df)} granular AI-extracted records")
    except Exception as e:
        st.info("Granular AI extraction in progress. Showing basic data for now.")
        granular_df = pd.DataFrame()

    # Load existing sales data
    try:
        existing_df = pd.read_csv("/Users/joshuavaughan/Downloads/SALES REPORT DASHBOARD 259a8c5acfd980bca654e78f043e87fc.csv")
        existing_df['source'] = 'existing'

        # Add missing granular fields to existing data
        granular_fields = [
            'client_location', 'marital_status', 'client_age', 'spouse_age', 'num_beneficiaries',
            'beneficiary_details', 'estate_value', 'real_estate_count', 'real_estate_details',
            'investment_assets', 'business_interests', 'current_estate_docs', 'primary_concerns',
            'recommended_services', 'risk_factors', 'follow_up_needed'
        ]

        for field in granular_fields:
            if field not in existing_df.columns:
                existing_df[field] = extract_from_notes(existing_df, field) if field in ['client_location', 'marital_status', 'client_age', 'estate_value'] else ''

        print(f"Loaded {len(existing_df)} existing sales records")
    except Exception as e:
        st.warning(f"Could not load existing sales data: {e}")
        existing_df = pd.DataFrame()

    # Load basic transcript data as fallback
    try:
        transcript_df = pd.read_csv("transcript_sales_data.csv")
        transcript_df['source'] = 'transcript'
        # Add empty granular fields
        for field in granular_fields:
            if field not in transcript_df.columns:
                transcript_df[field] = ''
        print(f"Loaded {len(transcript_df)} transcript records")
    except Exception as e:
        transcript_df = pd.DataFrame()

    # Combine all data sources
    all_dfs = [df for df in [granular_df, existing_df, transcript_df] if not df.empty]

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    else:
        # Fallback data
        combined_df = pd.DataFrame([{
            'deal_id': 1,
            'Date': datetime.now().strftime('%Y/%m/%d'),
            'Lead ': 'Loading...',
            'Stage': 'Follow up',
            'Payment': 0,
            'source': 'fallback'
        }])

    # Standardize date format
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')

    return combined_df

def extract_from_notes(df, field):
    """Extract granular fields from existing notes using regex"""
    if 'Notes' not in df.columns:
        return ''

    def extract_field(notes_text, field_name):
        if pd.isna(notes_text):
            return ''

        text = str(notes_text)

        if field_name == 'client_location':
            # Look for state names
            states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
            for state in states:
                if state in text:
                    return state
            return ''

        elif field_name == 'marital_status':
            if 'Married' in text:
                return 'Married'
            elif 'Single' in text:
                return 'Single'
            elif 'Divorced' in text:
                return 'Divorced'
            elif 'Widowed' in text:
                return 'Widowed'
            return ''

        elif field_name == 'client_age':
            # Look for age patterns like "66 yo" or "69 yo"
            import re
            age_match = re.search(r'(\d{2})\s*yo', text)
            if age_match:
                return int(age_match.group(1))
            return None

        elif field_name == 'estate_value':
            # Look for estate value patterns
            import re
            value_match = re.search(r'Estate ~?\$([0-9,]+(?:\.[0-9]+)?)([kKmM]?)', text)
            if value_match:
                value = float(value_match.group(1).replace(',', ''))
                unit = value_match.group(2).lower()
                if unit in ['k']:
                    value *= 1000
                elif unit in ['m']:
                    value *= 1000000
                return int(value)
            return None

        return ''

    return df['Notes'].apply(lambda x: extract_field(x, field))

# Dashboard Header
st.markdown(f"""
<div class="main-header">
    <h1>📊 Josh Vaughan Granular Sales Analytics</h1>
    <p>AI-Enhanced Estate Planning Client Intelligence Dashboard</p>
    <div style="display: flex; align-items: center; gap: 10px;">
        <span class="live-indicator">●</span>
        <span>LIVE AI EXTRACTION</span>
        <span style="margin-left: 20px;">Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load sales data
df = load_granular_sales_data()

# Sidebar - Control Panel
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")

    # Real-time metrics refresh
    if st.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # Granular Analytics Toggle
    st.markdown("### 🔬 Granular Analytics")
    show_granular = st.checkbox("Show Detailed Client Fields", value=True)
    show_demographics = st.checkbox("Show Demographics Breakdown", value=True)
    show_location_analysis = st.checkbox("Show Geographic Analysis", value=True)

    st.markdown("---")

    # Data source filter
    st.markdown("### 📊 Data Sources")
    sources = ['All'] + list(df['source'].unique())
    source_filter = st.selectbox("Data Source", sources, index=0)

    if source_filter != 'All':
        df = df[df['source'] == source_filter]

    # Advanced Filters
    st.markdown("### 🔍 Advanced Filters")

    # Location filter
    if 'client_location' in df.columns:
        locations = ['All'] + [loc for loc in df['client_location'].dropna().unique() if loc]
        location_filter = st.selectbox("Client Location", locations)
        if location_filter != 'All':
            df = df[df['client_location'] == location_filter]

    # Marital status filter
    if 'marital_status' in df.columns:
        marital_statuses = ['All'] + [status for status in df['marital_status'].dropna().unique() if status]
        marital_filter = st.selectbox("Marital Status", marital_statuses)
        if marital_filter != 'All':
            df = df[df['marital_status'] == marital_filter]

    # Age range filter
    if 'client_age' in df.columns:
        # Convert client_age to numeric, handling mixed types
        df['client_age'] = pd.to_numeric(df['client_age'], errors='coerce')
        ages = df['client_age'].dropna()
        if len(ages) > 0:
            min_age, max_age = int(ages.min()), int(ages.max())
            age_range = st.slider("Client Age Range", min_age, max_age, (min_age, max_age))
            df = df[(df['client_age'].isna()) | ((df['client_age'] >= age_range[0]) & (df['client_age'] <= age_range[1]))]

    # Stage filter - Fixed 4 stage options in logical order
    stage_options = ["Closed Won", "Follow up", "No Show", "Closed Lost"]
    stage_filter = st.multiselect(
        "Stage",
        options=stage_options,
        default=stage_options
    )
    df = df[df['Stage'].isin(stage_filter)]

# Apply final filtering
filtered_df = df.copy()

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
    # Convert to numeric and calculate average age
    if 'client_age' in filtered_df.columns:
        numeric_ages = pd.to_numeric(filtered_df['client_age'], errors='coerce').dropna()
        avg_age = numeric_ages.mean() if len(numeric_ages) > 0 else 0
    else:
        avg_age = 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{avg_age:.0f}</div>
        <div class="metric-label">Avg Client Age</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    closed_won = len(filtered_df[filtered_df['Stage'] == 'Closed Won'])
    total_deals = len(filtered_df[filtered_df['Stage'].isin(['Closed Won', 'Closed Lost'])])
    close_rate = (closed_won / total_deals * 100) if total_deals > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{close_rate:.1f}%</div>
        <div class="metric-label">Close Rate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Granular Analytics Section
if show_granular and 'client_location' in filtered_df.columns:
    st.markdown("### 🔬 Granular Client Analytics")

    col1, col2, col3 = st.columns(3)

    # Demographics breakdown
    if show_demographics:
        with col1:
            st.markdown("#### 👥 Demographics")

            # Marital status breakdown
            if 'marital_status' in filtered_df.columns:
                marital_counts = filtered_df['marital_status'].value_counts()
                if len(marital_counts) > 0:
                    fig_marital = px.pie(
                        values=marital_counts.values,
                        names=marital_counts.index,
                        title="Marital Status Distribution"
                    )
                    st.plotly_chart(fig_marital, use_container_width=True)

            # Age distribution
            if 'client_age' in filtered_df.columns:
                numeric_ages = pd.to_numeric(filtered_df['client_age'], errors='coerce').dropna()
                if len(numeric_ages) > 0:
                    fig_age = px.histogram(
                        numeric_ages,
                        nbins=10,
                        title="Age Distribution",
                        color_discrete_sequence=['#3b82f6']
                    )
                    st.plotly_chart(fig_age, use_container_width=True)

    # Geographic analysis
    if show_location_analysis:
        with col2:
            st.markdown("#### 🗺️ Geographic Analysis")

            if 'client_location' in filtered_df.columns:
                location_counts = filtered_df['client_location'].value_counts()
                if len(location_counts) > 0:
                    fig_location = px.bar(
                        x=location_counts.values,
                        y=location_counts.index,
                        orientation='h',
                        title="Clients by Location",
                        color=location_counts.values,
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_location, use_container_width=True)

    # Estate value analysis
    with col3:
        st.markdown("#### 💰 Estate Value Analysis")

        if 'estate_value' in filtered_df.columns:
            numeric_estates = pd.to_numeric(filtered_df['estate_value'], errors='coerce').dropna()
            if len(numeric_estates) > 0:
                fig_estate = px.box(
                    y=numeric_estates,
                    title="Estate Value Distribution"
                )
                st.plotly_chart(fig_estate, use_container_width=True)

# Enhanced Client Records Table
st.markdown("---")
st.markdown("### 📋 Enhanced Client Records")

# Search functionality
search_term = st.text_input("🔍 Search clients...", placeholder="Enter client name, location, or notes")
if search_term:
    filtered_df = filtered_df[
        filtered_df['Lead '].str.contains(search_term, case=False, na=False) |
        filtered_df.get('client_location', pd.Series(dtype='object')).str.contains(search_term, case=False, na=False) |
        filtered_df.get('Notes', pd.Series(dtype='object')).str.contains(search_term, case=False, na=False)
    ]

# Select columns to display
base_columns = ['Date', 'Lead ', 'Stage', 'Payment', 'source']
granular_columns = []

if 'client_location' in filtered_df.columns:
    granular_columns.extend(['client_location', 'marital_status', 'client_age'])

if show_granular:
    display_columns = base_columns + granular_columns + ['Notes']
else:
    display_columns = base_columns + ['Notes']

# Format the dataframe for display
display_df = filtered_df[display_columns].copy()
display_df['Payment'] = display_df['Payment'].apply(lambda x: f"${x:,.0f}" if x > 0 else "$0")
display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y/%m/%d')

st.dataframe(
    display_df,
    use_container_width=True,
    height=500,
    column_config={
        "Date": st.column_config.TextColumn("Date", width="small"),
        "Lead ": st.column_config.TextColumn("Client Name", width="medium"),
        "Stage": st.column_config.TextColumn("Stage", width="small"),
        "Payment": st.column_config.TextColumn("Revenue", width="small"),
        "client_location": st.column_config.TextColumn("Location", width="medium"),
        "marital_status": st.column_config.TextColumn("Marital Status", width="small"),
        "client_age": st.column_config.NumberColumn("Age", width="small"),
        "Notes": st.column_config.TextColumn("Notes", width="large"),
        "source": st.column_config.TextColumn("Source", width="small")
    }
)

# Granular Details Expander
if show_granular and 'client_location' in filtered_df.columns:
    with st.expander("🔬 View Detailed Client Information"):
        selected_client = st.selectbox(
            "Select client for detailed view:",
            options=filtered_df['Lead '].tolist()
        )

        if selected_client:
            client_data = filtered_df[filtered_df['Lead '] == selected_client].iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Demographics:**")
                st.write(f"Location: {client_data.get('client_location', 'N/A')}")
                st.write(f"Marital Status: {client_data.get('marital_status', 'N/A')}")
                st.write(f"Age: {client_data.get('client_age', 'N/A')}")
                st.write(f"Spouse Age: {client_data.get('spouse_age', 'N/A')}")
                st.write(f"Beneficiaries: {client_data.get('num_beneficiaries', 'N/A')}")

            with col2:
                st.markdown("**Estate Information:**")
                st.write(f"Estate Value: ${client_data.get('estate_value', 0):,.0f}")
                st.write(f"Real Estate: {client_data.get('real_estate_details', 'N/A')}")
                st.write(f"Current Documents: {client_data.get('current_estate_docs', 'N/A')}")
                st.write(f"Primary Concerns: {client_data.get('primary_concerns', 'N/A')}")
                st.write(f"Follow-up Needed: {client_data.get('follow_up_needed', 'N/A')}")

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
    query = st.text_area("Ask about your granular client data:", placeholder="What patterns do you see in client ages vs estate values?")
    if st.button("🔍 Ask AI", type="primary") and query:
        with st.spinner("Analyzing your granular data..."):
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
    <p>Josh Vaughan Granular Sales Analytics | AI-Enhanced Client Intelligence</p>
    <p>📊 {len(filtered_df)} records displayed | 🤖 AI extraction active | Last update: {datetime.now().strftime("%H:%M:%S")}</p>
</div>
""", unsafe_allow_html=True)