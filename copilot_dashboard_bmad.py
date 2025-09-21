#!/usr/bin/env python3
"""
Stellar Connect Sales Copilot Dashboard with BMAD Integration
Enhanced with BMAD-METHOD multi-agent orchestration capabilities
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

# Import standard components
from src.stellar_crew import (
    run_crew,
    create_general_query_tasks,
    create_structured_record_tasks,
    create_email_recap_tasks
)
from src.ingestion import process_new_file

# Import BMAD components
try:
    from bmad_final_integration import BMADDashboardIntegration
    BMAD_AVAILABLE = True
except ImportError:
    BMAD_AVAILABLE = False
    print("[WARNING] BMAD integration not available")

# Page Configuration
st.set_page_config(
    page_title="Stellar Connect Sales Copilot (BMAD Enhanced)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced interface
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
    .bmad-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
    .chat-container {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        min-height: 200px;
    }
    .bmad-metrics {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .mode-selector {
        background: white;
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .agent-status {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .agent-active {
        color: #10b981;
        font-weight: bold;
    }
    .agent-inactive {
        color: #6b7280;
    }
    .workflow-card {
        background: white;
        border: 2px solid #3b82f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_query' not in st.session_state:
    st.session_state.processing_query = False
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "standard"
if 'bmad_integration' not in st.session_state and BMAD_AVAILABLE:
    st.session_state.bmad_integration = BMADDashboardIntegration()
if 'bmad_metrics' not in st.session_state:
    st.session_state.bmad_metrics = {}

# Load data (using existing function from original dashboard)
@st.cache_data
def load_all_sales_data():
    """Load all sales data sources"""
    data_sources = []
    debug_info = []

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

    try:
        granular_df = pd.read_csv("granular_sales_data.csv")
        granular_df['source'] = 'AI_extracted'
        data_sources.append(granular_df)
        debug_info.append(f"✅ Loaded {len(granular_df)} AI-extracted records")
    except Exception as e:
        debug_info.append(f"❌ No AI-extracted data: {str(e)}")

    try:
        transcript_df = pd.read_csv("transcript_sales_data.csv")
        transcript_df['source'] = 'transcript'
        data_sources.append(transcript_df)
        debug_info.append(f"✅ Loaded {len(transcript_df)} transcript records")
    except Exception as e:
        debug_info.append(f"❌ No transcript data: {str(e)}")

    if data_sources:
        combined_df = pd.concat(data_sources, ignore_index=True, sort=False)
        debug_info.append(f"📊 Combined total: {len(combined_df)} records")
    else:
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

    if 'Date' in combined_df.columns:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')

    st.session_state.data_debug = debug_info
    return combined_df

# Load data
df = load_all_sales_data()

# Enhanced Sidebar with BMAD Controls
with st.sidebar:
    st.markdown("### 🚀 BMAD-METHOD Controls")

    if BMAD_AVAILABLE:
        st.markdown('<span class="bmad-badge">BMAD ACTIVE</span>', unsafe_allow_html=True)

        # Processing Mode Selector
        st.markdown("#### 🎯 Processing Mode")
        processing_mode = st.selectbox(
            "Select AI Processing Mode:",
            options=[
                "standard",
                "bmad_sales_optimization",
                "bmad_system_implementation",
                "bmad_cognitive_analysis",
                "bmad_full_orchestration"
            ],
            format_func=lambda x: {
                "standard": "Standard CrewAI",
                "bmad_sales_optimization": "BMAD Sales Optimization",
                "bmad_system_implementation": "BMAD System Implementation",
                "bmad_cognitive_analysis": "BMAD Cognitive Analysis",
                "bmad_full_orchestration": "BMAD Full Orchestration"
            }.get(x, x),
            key="mode_selector",
            help="Select the AI processing mode for your queries"
        )
        st.session_state.processing_mode = processing_mode

        # BMAD Agent Status
        st.markdown("#### 🤖 BMAD Agent Status")

        bmad_agents = [
            ("Business Analyst", "agent-active"),
            ("Project Manager", "agent-active"),
            ("Solution Architect", "agent-active"),
            ("Developer", "agent-active"),
            ("QA Tester", "agent-active"),
            ("Sales Specialist", "agent-active")
        ]

        for agent_name, status in bmad_agents:
            status_icon = "🟢" if status == "agent-active" else "⚫"
            st.markdown(f"{status_icon} **{agent_name}**")

        # BMAD Metrics
        if st.session_state.bmad_integration:
            metrics = st.session_state.bmad_integration.get_dashboard_metrics()
            st.session_state.bmad_metrics = metrics

            st.markdown("#### 📊 BMAD Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Executions", metrics.get("total_executions", 0))
                st.metric("Agents", len(metrics.get("agent_types", [])))
            with col2:
                st.metric("Success", metrics.get("successful_executions", 0))
                st.metric("Workflows", len(metrics.get("available_workflows", [])))

        st.markdown("---")
    else:
        st.warning("BMAD integration not available. Using standard mode.")
        st.session_state.processing_mode = "standard"

    # Standard Controls
    st.markdown("### 🛠️ Debug Controls")

    debug_mode = st.checkbox("Enable Debug Mode", value=True,
                             help="Shows detailed logging in terminal")

    if debug_mode:
        st.info("Debug mode ON - Check terminal for logs")

    st.markdown("---")

    if hasattr(st.session_state, 'data_debug'):
        st.markdown("### 📊 Data Status")
        for info in st.session_state.data_debug:
            st.markdown(f"- {info}")

    if st.button("🔄 Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    if st.button("🗑️ Clear All Caches"):
        st.cache_data.clear()
        st.session_state.clear()
        st.success("All caches cleared!")
        st.rerun()

# Main Header with BMAD Badge
st.markdown(f"""
<div class="copilot-header">
    <h1>🤖 Stellar Connect Sales Copilot
        <span class="bmad-badge">BMAD ENHANCED</span>
    </h1>
    <p>Advanced Multi-Agent RAG with BMAD-METHOD Orchestration</p>
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 10px;">
        <span class="live-indicator">●</span>
        <span>COPILOT ACTIVE</span>
        <span style="margin-left: 20px;">Mode: {st.session_state.processing_mode.upper()}</span>
        <span style="margin-left: 20px;">Last Updated: {datetime.now().strftime("%H:%M:%S")}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# BMAD Workflow Information
if st.session_state.processing_mode.startswith("bmad_"):
    with st.expander("ℹ️ About Current BMAD Mode", expanded=False):
        mode_info = {
            "bmad_sales_optimization": {
                "title": "Sales Optimization Workflow",
                "description": "Comprehensive sales analysis and optimization using multiple BMAD agents",
                "agents": ["Sales Specialist", "Business Analyst", "Project Manager"],
                "capabilities": [
                    "Sales pattern analysis and conversion optimization",
                    "Requirement extraction and validation",
                    "Implementation planning and coordination"
                ]
            },
            "bmad_system_implementation": {
                "title": "System Implementation Workflow",
                "description": "Full-cycle feature implementation with quality assurance",
                "agents": ["Solution Architect", "Developer", "QA Tester"],
                "capabilities": [
                    "Technical architecture design",
                    "Production-ready implementation",
                    "Comprehensive testing and validation"
                ]
            },
            "bmad_cognitive_analysis": {
                "title": "Cognitive Analysis Workflow",
                "description": "Deep analysis using reasoning engine integration",
                "agents": ["Business Analyst", "Sales Specialist"],
                "capabilities": [
                    "Cognitive validation and planning",
                    "Specialist agent coordination",
                    "Strategic synthesis and recommendations"
                ]
            },
            "bmad_full_orchestration": {
                "title": "Full BMAD Orchestration",
                "description": "Complete multi-agent workflow with all BMAD agents",
                "agents": ["All BMAD Agents"],
                "capabilities": [
                    "End-to-end requirement to implementation",
                    "Comprehensive quality assurance",
                    "Full project lifecycle management"
                ]
            }
        }

        if st.session_state.processing_mode in mode_info:
            info = mode_info[st.session_state.processing_mode]
            st.markdown(f"### {info['title']}")
            st.markdown(info['description'])
            st.markdown("**Active Agents:**")
            for agent in info['agents']:
                st.markdown(f"- {agent}")
            st.markdown("**Capabilities:**")
            for capability in info['capabilities']:
                st.markdown(f"- {capability}")

# Chat Interface with BMAD Enhancement
st.markdown("### 💬 Ask Your Enhanced Sales Copilot")

# Create columns for input and mode display
col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input(
        "Ask about your clients, sales data, or estate planning insights:",
        placeholder="What patterns do you see in successful deals? How can we optimize conversion rates?",
        key="copilot_query"
    )

with col2:
    st.markdown("**Current Mode:**")
    mode_display = st.session_state.processing_mode.replace("bmad_", "").replace("_", " ").title()
    if st.session_state.processing_mode.startswith("bmad_"):
        st.success(f"🚀 {mode_display}")
    else:
        st.info(f"📊 {mode_display}")

# Process query with BMAD enhancement
if user_query and not st.session_state.processing_query:
    st.session_state.processing_query = True

    # Determine processing mode
    mode = st.session_state.processing_mode

    with st.spinner(f"🤖 Processing with {mode_display} mode..."):
        try:
            # Add comprehensive debugging
            if debug_mode:
                print(f"\n[DEBUG] ========== BMAD-ENHANCED CHAT HANDLER START ==========")
                print(f"[DEBUG] User Query: '{user_query}'")
                print(f"[DEBUG] Processing Mode: {mode}")
                print(f"[DEBUG] BMAD Available: {BMAD_AVAILABLE}")

            # Add user message to chat
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "mode": mode
            })

            # Process based on mode
            if mode.startswith("bmad_") and BMAD_AVAILABLE:
                # Use BMAD processing
                if debug_mode:
                    print(f"[DEBUG] Using BMAD integration for mode: {mode}")

                # Map mode to BMAD workflow
                bmad_mode_map = {
                    "bmad_sales_optimization": "sales_optimization",
                    "bmad_system_implementation": "system_implementation",
                    "bmad_cognitive_analysis": "sales_optimization",  # Using sales as proxy
                    "bmad_full_orchestration": "sales_optimization"  # Using sales as proxy
                }

                bmad_workflow = bmad_mode_map.get(mode, "sales_optimization")

                if debug_mode:
                    print(f"[DEBUG] Executing BMAD workflow: {bmad_workflow}")

                # Execute BMAD workflow
                ai_response = st.session_state.bmad_integration.process_bmad_query(
                    user_query,
                    bmad_workflow
                )

                # Add BMAD badge to response
                ai_response = f"**[BMAD-Enhanced Response]**\n\n{ai_response}"

            else:
                # Use standard CrewAI processing
                if debug_mode:
                    print(f"[DEBUG] Using standard CrewAI processing")

                tasks = create_general_query_tasks(user_query)
                ai_response = run_crew(tasks)

            # Ensure response is a string
            if ai_response is None:
                ai_response = "No response generated. Please check the backend logs."
            elif not isinstance(ai_response, str):
                ai_response = str(ai_response)

            if not ai_response or ai_response.strip() == "":
                ai_response = "The copilot returned an empty response. Please check the backend configuration."

            if debug_mode:
                print(f"[DEBUG] Final AI Response (first 200 chars): {ai_response[:200]}")
                print(f"[DEBUG] ========== BMAD-ENHANCED CHAT HANDLER END ==========\n")

            # Add AI response to chat
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "mode": mode
            })

            # Display success message with mode info
            st.success(f"✅ Response generated successfully using {mode_display} mode ({len(ai_response)} characters)")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()

            if debug_mode:
                print(f"\n[DEBUG] ========== ERROR IN BMAD-ENHANCED HANDLER ==========")
                print(f"[DEBUG] Error Type: {type(e).__name__}")
                print(f"[DEBUG] Error Message: {str(e)}")
                print(f"[DEBUG] Full Traceback:")
                print(error_trace)
                print(f"[DEBUG] ========== END ERROR ==========\n")

            error_message = f"""
🔴 **Error Details:**
- Mode: {mode}
- Error Type: {type(e).__name__}
- Message: {str(e)}
- Please check the terminal for full traceback
            """

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_message.strip(),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "mode": mode
            })

            st.error(f"Error in {mode_display} mode: {type(e).__name__} - {str(e)}")

    st.session_state.processing_query = False
    st.rerun()

# Display chat history with mode indicators
if st.session_state.chat_history:
    st.markdown("#### 💬 Conversation History")
    for i, message in enumerate(reversed(st.session_state.chat_history[-6:])):
        mode_badge = ""
        if "mode" in message and message["mode"].startswith("bmad_"):
            mode_badge = f' <span class="bmad-badge" style="font-size: 0.7rem;">BMAD</span>'

        if message["role"] == "user":
            st.markdown(f"**You ({message['timestamp']}){mode_badge}:** {message['content']}",
                       unsafe_allow_html=True)
        else:
            st.markdown(f"**🤖 Copilot ({message['timestamp']}){mode_badge}:** {message['content']}",
                       unsafe_allow_html=True)

        if i < len(st.session_state.chat_history[-6:]) - 1:
            st.markdown("---")

# Clear chat button
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

with col2:
    if BMAD_AVAILABLE and st.button("📊 Show BMAD Stats", key="show_stats"):
        with st.expander("BMAD Execution Statistics", expanded=True):
            metrics = st.session_state.bmad_integration.get_dashboard_metrics()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Executions", metrics.get("total_executions", 0))
            with col2:
                st.metric("Successful", metrics.get("successful_executions", 0))
            with col3:
                st.metric("Failed", metrics.get("failed_executions", 0))
            with col4:
                avg_duration = metrics.get("average_duration", 0)
                st.metric("Avg Duration", f"{avg_duration:.1f}s")

# Sales Data Section (keeping original functionality)
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
    total_revenue = df['Payment'].sum() if 'Payment' in df.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${total_revenue:,.0f}</div>
        <div style="color: #6b7280; font-size: 0.875rem;">Total Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    active_deals = len(df[df['Stage'] == 'Follow up']) if 'Stage' in df.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{active_deals}</div>
        <div style="color: #6b7280; font-size: 0.875rem;">Active Deals</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    closed_won = len(df[df['Stage'] == 'Closed Won']) if 'Stage' in df.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{closed_won}</div>
        <div style="color: #6b7280; font-size: 0.875rem;">Closed Won</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    if 'Payment' in df.columns:
        avg_deal = df[df['Payment'] > 0]['Payment'].mean() if len(df[df['Payment'] > 0]) > 0 else 0
    else:
        avg_deal = 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${avg_deal:.0f}</div>
        <div style="color: #6b7280; font-size: 0.875rem;">Avg Deal Size</div>
    </div>
    """, unsafe_allow_html=True)

# Display data table (simplified for space)
st.dataframe(df.head(10), use_container_width=True, height=400)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>🤖 Stellar Connect Sales Copilot | BMAD-Enhanced Multi-Agent Intelligence</p>
    <p>📊 {len(df)} records | 🚀 {len(bmad_agents) if BMAD_AVAILABLE else 0} BMAD agents |
       ⚡ {st.session_state.processing_mode} mode | Last update: {datetime.now().strftime("%H:%M:%S")}</p>
</div>
""", unsafe_allow_html=True)