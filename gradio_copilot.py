#!/usr/bin/env python3
"""
Stellar Connect Sales Copilot - Gradio RAG Chat Interface
Local Agentic RAG Assistant for Josh Vaughan's Estate Planning Sales
"""

import gradio as gr
import sys
import os
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Use lazy loading for stellar_crew imports
def lazy_import_stellar_crew():
    """Lazy load stellar_crew modules when needed"""
    try:
        from src.stellar_crew import (
            run_crew,
            create_general_query_tasks,
            create_structured_record_tasks,
            create_email_recap_tasks
        )
        return run_crew, create_general_query_tasks, create_structured_record_tasks, create_email_recap_tasks
    except Exception as e:
        print(f"❌ Error loading stellar_crew: {str(e)}")
        raise e

def respond_to_query(message, history):
    """Process user query with CrewAI and return response"""
    try:
        print(f"\n🤖 Processing query: {message}")

        # Lazy load stellar_crew modules
        run_crew, create_general_query_tasks, create_structured_record_tasks, create_email_recap_tasks = lazy_import_stellar_crew()

        # Create tasks for general query
        tasks = create_general_query_tasks(message)

        # Run the crew
        result = run_crew(tasks)

        # Convert result to string if it's not already
        response = str(result) if result else "I couldn't process that query. Please try rephrasing."

        print(f"✅ Response generated: {response[:100]}...")
        return response

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(f"❌ {error_msg}")
        return f"I encountered an error: {str(e)}. Please try again."

def generate_client_record(client_name):
    """Generate structured record for a client"""
    try:
        print(f"\n📊 Generating record for: {client_name}")

        # Lazy load stellar_crew modules
        run_crew, create_general_query_tasks, create_structured_record_tasks, create_email_recap_tasks = lazy_import_stellar_crew()

        tasks = create_structured_record_tasks(client_name)
        result = run_crew(tasks)

        # Try to parse as JSON for better formatting
        try:
            json_result = json.loads(result)
            formatted_result = json.dumps(json_result, indent=2)
        except:
            formatted_result = str(result)

        print(f"✅ Record generated for {client_name}")
        return formatted_result

    except Exception as e:
        error_msg = f"Error generating record: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

def draft_email_recap(client_name):
    """Draft email recap for a client"""
    try:
        print(f"\n✉️ Drafting email for: {client_name}")

        # Lazy load stellar_crew modules
        run_crew, create_general_query_tasks, create_structured_record_tasks, create_email_recap_tasks = lazy_import_stellar_crew()

        tasks = create_email_recap_tasks(client_name)
        result = run_crew(tasks)

        print(f"✅ Email drafted for {client_name}")
        return str(result)

    except Exception as e:
        error_msg = f"Error drafting email: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# Custom CSS for professional styling
css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.chat-message {
    padding: 10px;
    margin: 5px;
    border-radius: 10px;
}
.header-text {
    text-align: center;
    color: #2c3e50;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.tab-nav {
    background: linear-gradient(90deg, #1f2937 0%, #374151 100%);
}
"""

# Create the Gradio interface
with gr.Blocks(css=css, title="Stellar Connect Sales Copilot") as demo:

    # Header
    gr.Markdown("""
    <div class="header-text">
        <h1>🌟 Stellar Connect Sales Copilot</h1>
        <h3>AI-Powered Estate Planning Sales Assistant for Josh Vaughan</h3>
        <p>Ask questions about your clients, sales data, and get insights from your estate planning meetings</p>
    </div>
    """)

    with gr.Tabs():

        # Main Chat Interface
        with gr.TabItem("💬 Sales Copilot Chat"):
            gr.Markdown("### Ask questions about your sales data, client insights, or estate planning trends")

            chatbot = gr.Chatbot(
                height=500,
                placeholder="Ask me anything about your sales data, client meetings, or estate planning insights...",
                show_label=False,
                type="messages"
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="What patterns do you see in successful deals?",
                    label="Your Question",
                    scale=4
                )
                submit_btn = gr.Button("Ask Copilot", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat", scale=1)

            # Example questions
            gr.Markdown("### 💡 Example Questions:")
            example_questions = gr.Examples(
                examples=[
                    ["What patterns do you see in successful deals?"],
                    ["Which clients have the highest estate values?"],
                    ["What are the common objections from clients?"],
                    ["Show me clients who need follow-up"],
                    ["What's the average deal size for closed won deals?"],
                    ["Which geographic areas have the most clients?"],
                    ["What are the main concerns clients have about estate planning?"]
                ],
                inputs=msg
            )

        # Client Record Generation
        with gr.TabItem("📊 Generate Client Record"):
            gr.Markdown("### Generate structured sales record for a specific client")

            with gr.Row():
                client_name_input = gr.Textbox(
                    label="Client Name",
                    placeholder="Enter client name (e.g., John Murphy, THOMAS EDWARDS)",
                    scale=3
                )
                generate_btn = gr.Button("Generate Record", variant="primary", scale=1)

            record_output = gr.Textbox(
                label="Generated Client Record (JSON)",
                lines=15,
                max_lines=20,
                show_copy_button=True
            )

        # Email Recap Generation
        with gr.TabItem("✉️ Draft Email Recap"):
            gr.Markdown("### Generate professional email recap for client meetings")

            with gr.Row():
                email_client_input = gr.Textbox(
                    label="Client Name",
                    placeholder="Enter client name for email recap",
                    scale=3
                )
                email_btn = gr.Button("Draft Email", variant="primary", scale=1)

            email_output = gr.Textbox(
                label="Email Draft",
                lines=12,
                max_lines=15,
                show_copy_button=True
            )

    # Event handlers
    def submit_message(message, history):
        if not message.strip():
            return history, ""

        # Add user message to history using OpenAI format
        history.append({"role": "user", "content": message})
        return history, ""

    def get_response(history):
        if not history or (history[-1]["role"] == "assistant"):
            return history

        user_message = history[-1]["content"]

        # Get AI response
        ai_response = respond_to_query(user_message, history[:-1])

        # Add AI response to history
        history.append({"role": "assistant", "content": ai_response})
        return history

    def clear_chat():
        return []

    # Wire up the chat interface
    submit_btn.click(
        submit_message,
        [msg, chatbot],
        [chatbot, msg],
        queue=False
    ).then(
        get_response,
        chatbot,
        chatbot
    )

    msg.submit(
        submit_message,
        [msg, chatbot],
        [chatbot, msg],
        queue=False
    ).then(
        get_response,
        chatbot,
        chatbot
    )

    clear_btn.click(clear_chat, outputs=chatbot, queue=False)

    # Wire up other functions
    generate_btn.click(
        generate_client_record,
        inputs=client_name_input,
        outputs=record_output
    )

    email_btn.click(
        draft_email_recap,
        inputs=email_client_input,
        outputs=email_output
    )

    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #6b7280; font-size: 0.9em;">
        <p>🌟 Stellar Connect Sales Copilot | McAdams Estate Planning | Powered by Local AI</p>
        <p>Built with CrewAI, LlamaIndex, and Gradio | Running on Mistral-7B</p>
    </div>
    """)

if __name__ == "__main__":
    print("🚀 Starting Stellar Connect Sales Copilot...")
    print("🤖 AI Models: Mistral-7B (Generation) + Nomic-Embed (Embeddings)")
    print("📊 Data Sources: Estate Planning Meeting Transcripts + Sales Records")
    print("🔗 Access: http://localhost:7860")

    try:
        print("🔄 Launching Gradio interface...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            debug=True
        )
        print("✅ Gradio interface launched successfully")
    except Exception as e:
        print(f"❌ Error launching Gradio: {str(e)}")
        import traceback
        traceback.print_exc()