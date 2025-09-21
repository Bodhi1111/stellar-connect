#!/usr/bin/env python3
"""
Fast Stellar Connect Sales Copilot Dashboard
Optimized for quick responses with timeout controls
"""

import streamlit as st
import pandas as pd
import sys
import os
import json
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import concurrent.futures
from threading import Thread
import time

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import components
from src.stellar_crew import (
    run_crew,
    create_general_query_tasks,
)
from src.agent_tools import vector_tool

# Import BMAD with timeout
try:
    from bmad_final_integration import BMADDashboardIntegration
    BMAD_AVAILABLE = True
except ImportError:
    BMAD_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="Fast Stellar Connect Sales Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .fast-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
    .processing-status {
        background: #f0f9ff;
        border: 2px solid #0ea5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .timeout-warning {
        background: #fef3c7;
        border: 2px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_query' not in st.session_state:
    st.session_state.processing_query = False
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "fast_vector"
if 'bmad_integration' not in st.session_state and BMAD_AVAILABLE:
    st.session_state.bmad_integration = BMADDashboardIntegration()

# Fast processing functions
def fast_vector_search(query: str) -> str:
    """Quick vector search without full CrewAI pipeline"""
    try:
        print(f"[FAST] Vector search for: {query}")
        result = vector_tool._run(query)
        return f"**Quick Analysis:**\n\n{result}"
    except Exception as e:
        return f"Quick search error: {str(e)}"

def run_with_timeout(func, args, timeout=30):
    """Run function with timeout"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return "⏱️ **Timeout:** Query took longer than expected. Try a simpler question or use Fast mode."

def fast_crew_execution(query: str) -> str:
    """Fast CrewAI execution with timeout"""
    try:
        print(f"[FAST] Creating tasks for: {query}")
        tasks = create_general_query_tasks(query)
        print(f"[FAST] Running crew with 30-second timeout...")

        result = run_with_timeout(run_crew, [tasks], timeout=30)
        return str(result)
    except Exception as e:
        return f"Fast execution error: {str(e)}"

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚡ Fast Processing Controls")

    # Mode selector
    processing_mode = st.selectbox(
        "Processing Mode:",
        options=[
            "fast_vector",
            "fast_crew",
            "bmad_quick",
            "standard"
        ],
        format_func=lambda x: {
            "fast_vector": "⚡ Fast Vector (2-3s)",
            "fast_crew": "🚀 Fast CrewAI (30s timeout)",
            "bmad_quick": "🤖 BMAD Quick (45s timeout)",
            "standard": "📊 Standard (no timeout)"
        }.get(x, x),
        help="Select processing speed vs. depth trade-off"
    )
    st.session_state.processing_mode = processing_mode

    # Timeout settings
    if processing_mode in ["fast_crew", "bmad_quick"]:
        timeout_seconds = st.slider(
            "Timeout (seconds):",
            min_value=15,
            max_value=120,
            value=30 if processing_mode == "fast_crew" else 45,
            step=15
        )
    else:
        timeout_seconds = None

    # Status
    st.markdown("#### 🔧 System Status")
    st.success("⚡ Fast Mode Active")
    if BMAD_AVAILABLE:
        st.info("🤖 BMAD Available")

    # Quick stats
    st.markdown("#### 📊 Quick Stats")
    st.metric("Chat Messages", len(st.session_state.chat_history))
    if st.session_state.chat_history:
        last_mode = st.session_state.chat_history[-1].get('mode', 'unknown')
        st.metric("Last Mode", last_mode)

# Header
st.markdown(f"""
<div class="copilot-header">
    <h1>⚡ Fast Stellar Connect Sales Copilot
        <span class="fast-badge">SPEED OPTIMIZED</span>
    </h1>
    <p>Quick Responses | Optimized Performance | Timeout Protection</p>
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 10px;">
        <span style="color: #10b981;">●</span>
        <span>FAST MODE: {processing_mode.upper()}</span>
        <span style="margin-left: 20px;">Updated: {datetime.now().strftime("%H:%M:%S")}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Mode explanation
mode_explanations = {
    "fast_vector": {
        "title": "⚡ Fast Vector Search",
        "description": "Direct vector database search for immediate results",
        "time": "2-3 seconds",
        "best_for": "Quick pattern lookups, similar cases, document search"
    },
    "fast_crew": {
        "title": "🚀 Fast CrewAI",
        "description": "CrewAI pipeline with 30-second timeout",
        "time": "15-30 seconds",
        "best_for": "Structured analysis, comprehensive answers"
    },
    "bmad_quick": {
        "title": "🤖 BMAD Quick",
        "description": "BMAD agents with timeout protection",
        "time": "30-45 seconds",
        "best_for": "Business analysis, sales optimization"
    },
    "standard": {
        "title": "📊 Standard Mode",
        "description": "Full processing without timeouts",
        "time": "1-5 minutes",
        "best_for": "Complex analysis, detailed reports"
    }
}

if processing_mode in mode_explanations:
    exp = mode_explanations[processing_mode]
    st.info(f"**{exp['title']}** | ⏱️ {exp['time']} | 🎯 Best for: {exp['best_for']}")

# Chat Interface
st.markdown("### 💬 Ask Your Fast Sales Copilot")

col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input(
        "Quick question:",
        placeholder="What patterns do successful deals show? Who are top clients?",
        key="fast_query"
    )

with col2:
    st.markdown("**Response Time:**")
    if processing_mode == "fast_vector":
        st.success("⚡ 2-3 sec")
    elif processing_mode == "fast_crew":
        st.warning("🚀 15-30 sec")
    elif processing_mode == "bmad_quick":
        st.warning("🤖 30-45 sec")
    else:
        st.error("📊 1-5 min")

# Process query
if user_query and not st.session_state.processing_query:
    st.session_state.processing_query = True

    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "mode": processing_mode
    })

    # Processing status
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    start_time = time.time()

    try:
        if processing_mode == "fast_vector":
            # Fast vector search
            status_placeholder.markdown('<div class="processing-status">⚡ Fast vector search...</div>', unsafe_allow_html=True)
            progress_bar.progress(50)

            ai_response = fast_vector_search(user_query)
            progress_bar.progress(100)

        elif processing_mode == "fast_crew":
            # Fast CrewAI with timeout
            status_placeholder.markdown('<div class="processing-status">🚀 Fast CrewAI processing...</div>', unsafe_allow_html=True)
            progress_bar.progress(25)

            ai_response = fast_crew_execution(user_query)
            progress_bar.progress(100)

        elif processing_mode == "bmad_quick" and BMAD_AVAILABLE:
            # BMAD with timeout
            status_placeholder.markdown('<div class="processing-status">🤖 BMAD quick processing...</div>', unsafe_allow_html=True)
            progress_bar.progress(25)

            def bmad_with_timeout():
                return st.session_state.bmad_integration.process_bmad_query(user_query, "sales_optimization")

            ai_response = run_with_timeout(bmad_with_timeout, [], timeout=timeout_seconds or 45)
            progress_bar.progress(100)

        else:
            # Standard mode
            status_placeholder.markdown('<div class="processing-status">📊 Standard processing...</div>', unsafe_allow_html=True)
            progress_bar.progress(25)

            tasks = create_general_query_tasks(user_query)
            progress_bar.progress(50)
            ai_response = run_crew(tasks)
            progress_bar.progress(100)

        # Calculate response time
        response_time = time.time() - start_time

        # Ensure response is valid
        if ai_response is None or (isinstance(ai_response, str) and not ai_response.strip()):
            ai_response = "No response generated. Please try a different question or mode."
        elif not isinstance(ai_response, str):
            ai_response = str(ai_response)

        # Add timing info
        ai_response = f"{ai_response}\n\n---\n⏱️ **Response time:** {response_time:.1f} seconds | **Mode:** {processing_mode}"

        # Add to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "mode": processing_mode,
            "response_time": response_time
        })

        # Clear status
        status_placeholder.empty()
        progress_bar.empty()

        # Show success
        if response_time < 5:
            st.success(f"⚡ Fast response in {response_time:.1f}s!")
        elif response_time < 30:
            st.info(f"🚀 Good response in {response_time:.1f}s")
        else:
            st.warning(f"📊 Response took {response_time:.1f}s")

    except Exception as e:
        status_placeholder.empty()
        progress_bar.empty()

        error_message = f"❌ **Error in {processing_mode} mode:** {str(e)}\n\n💡 **Try:** Switching to Fast Vector mode for quicker results."

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_message,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "mode": processing_mode
        })

        st.error(f"Error: {str(e)}")

    st.session_state.processing_query = False
    st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.markdown("#### 💬 Quick Chat History")

    for i, message in enumerate(reversed(st.session_state.chat_history[-4:])):  # Show last 4
        mode_badge = ""
        time_badge = ""

        if "mode" in message:
            if message["mode"] == "fast_vector":
                mode_badge = '<span style="background: #10b981; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem;">⚡ FAST</span>'
            elif message["mode"] == "fast_crew":
                mode_badge = '<span style="background: #3b82f6; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem;">🚀 CREW</span>'
            elif message["mode"] == "bmad_quick":
                mode_badge = '<span style="background: #8b5cf6; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem;">🤖 BMAD</span>'

        if "response_time" in message:
            rt = message["response_time"]
            if rt < 5:
                time_badge = f'<span style="background: #10b981; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem;">{rt:.1f}s</span>'
            elif rt < 30:
                time_badge = f'<span style="background: #f59e0b; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem;">{rt:.1f}s</span>'
            else:
                time_badge = f'<span style="background: #ef4444; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem;">{rt:.1f}s</span>'

        if message["role"] == "user":
            st.markdown(f"**You ({message['timestamp']})** {mode_badge}: {message['content']}", unsafe_allow_html=True)
        else:
            st.markdown(f"**🤖 Copilot ({message['timestamp']})** {mode_badge} {time_badge}: {message['content']}", unsafe_allow_html=True)

        if i < len(st.session_state.chat_history[-4:]) - 1:
            st.markdown("---")

# Quick actions
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

with col2:
    if st.button("⚡ Test Fast Mode"):
        if not st.session_state.processing_query:
            test_result = fast_vector_search("What are successful deal patterns?")
            st.info(f"Fast test completed: {test_result[:100]}...")

with col3:
    if st.button("📊 Performance Stats"):
        if st.session_state.chat_history:
            with st.expander("Performance Statistics", expanded=True):
                response_times = [m.get("response_time", 0) for m in st.session_state.chat_history if "response_time" in m]
                if response_times:
                    avg_time = sum(response_times) / len(response_times)
                    fast_responses = len([t for t in response_times if t < 5])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Response", f"{avg_time:.1f}s")
                    with col2:
                        st.metric("Fast Responses", f"{fast_responses}/{len(response_times)}")
                    with col3:
                        st.metric("Success Rate", f"{len(response_times)}/{len(st.session_state.chat_history)//2}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>⚡ Fast Stellar Connect Sales Copilot | Speed-Optimized for Quick Insights</p>
    <p>Current Mode: {processing_mode} | Chat Messages: {len(st.session_state.chat_history)} |
       Last Updated: {datetime.now().strftime("%H:%M:%S")}</p>
</div>
""", unsafe_allow_html=True)