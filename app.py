# app.py
import streamlit as st
import sys
import os
import json
import tempfile
from datetime import datetime

# Add the src directory to the Python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.stellar_crew import (
    run_crew,
    create_general_query_tasks,
    create_structured_record_tasks,
    create_email_recap_tasks
)
from src.ingestion import process_new_file

st.set_page_config(page_title="Stellar Connect Copilot", page_icon="⭐", layout="wide")

st.title("⭐ Stellar Connect Sales Copilot")
st.subheader("Your Local Agentic RAG Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 👋 Upload a transcript in the sidebar to get started, or use the chat for questions and automated tasks."}]

# --- Sidebar for File Upload ---
st.sidebar.header("📁 Upload Transcript")
st.sidebar.info("Upload a sales call transcript for immediate processing and analysis.")

uploaded_file = st.sidebar.file_uploader(
    "Choose a transcript file",
    type=['txt', 'pdf', 'docx'],
    help="Supported formats: TXT, PDF, DOCX"
)

if uploaded_file is not None:
    if st.sidebar.button("🚀 Process Transcript"):
        with st.spinner("Processing transcript... This may take 1-3 minutes."):
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                # Process the file using the ingestion pipeline
                result = process_new_file(temp_file_path)

                # Clean up temporary file
                os.unlink(temp_file_path)

                # Create success message
                success_message = f"""**✅ Transcript Processed Successfully!**

**File:** {uploaded_file.name}
**Size:** {uploaded_file.size} bytes
**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The transcript has been:
- ✅ Parsed and chunked semantically
- ✅ Added to vector store for semantic search
- ✅ Added to knowledge graph for relationship mapping
- ✅ Available for agent analysis and queries

You can now ask questions about this transcript in the chat below!"""

                st.session_state.messages.append({"role": "assistant", "content": success_message})
                st.rerun()

            except Exception as e:
                st.error(f"Error processing transcript: {str(e)}")

st.sidebar.markdown("---")

# --- Sidebar for Specific Tasks ---
st.sidebar.header("🤖 Automated CoPilot Tasks")
st.sidebar.info("These tasks utilize specialized agents for data extraction and content generation.")
client_name_input = st.sidebar.text_input("Enter Client Name for Tasks:")

# Helper function to execute sidebar tasks
def execute_task(task_creator, task_name, output_format="markdown"):
    if client_name_input:
        # Display spinner in the main area while working
        with st.spinner(f"Executing {task_name} for {client_name_input}... This may take 1-3 minutes."):
            try:
                tasks = task_creator(client_name_input)
                result = run_crew(tasks)

                # Format the output for display
                if output_format == "json":
                    # Pretty print JSON
                    formatted_result = f"**{task_name} for {client_name_input}:**\n\n```json\n{json.dumps(json.loads(result), indent=2)}\n```"
                else:
                    formatted_result = f"**{task_name} for {client_name_input}:**\n\n{result}"

                # Add the result to the chat history
                st.session_state.messages.append({"role": "assistant", "content": formatted_result})
                # Rerun the app to display the new message immediately
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.sidebar.warning("Please enter a client name.")

if st.sidebar.button("Generate Structured Sales Record (JSON)"):
    execute_task(create_structured_record_tasks, "Structured Record (JSON)", output_format="json")

if st.sidebar.button("Draft Email Recap"):
    execute_task(create_email_recap_tasks, "Draft Email Recap")

# --- Main Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input (General Q&A)
if prompt := st.chat_input("Ask a question about your sales calls..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.info("Thinking... (Agents are working. This may take 1-3 minutes.)")
        try:
            # Create the tasks for the general query
            tasks = create_general_query_tasks(prompt)
            result = run_crew(tasks)
            message_placeholder.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
        except Exception as e:
            st.error(f"An error occurred: {e}")
            message_placeholder.error("An error occurred while processing the request.")