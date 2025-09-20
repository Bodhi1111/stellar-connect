# app.py
import streamlit as st
import sys
import os
import json

# Add the src directory to the Python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.stellar_crew import (
    run_crew,
    create_general_query_tasks,
    create_structured_record_tasks,
    create_email_recap_tasks
)

st.set_page_config(page_title="Stellar Connect Copilot", page_icon="⭐", layout="wide")

st.title("⭐ Stellar Connect Sales Copilot")
st.subheader("Your Local Agentic RAG Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Use the chat for questions or the sidebar for automated tasks."}]

# --- Sidebar for Specific Tasks ---
st.sidebar.header("Automated CoPilot Tasks")
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