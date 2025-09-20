#!/usr/bin/env python3
"""
Simple Gradio Test - To isolate the issue
"""

import gradio as gr
import sys
import os
import json
from datetime import datetime

def respond_to_query(message, history):
    """Simple response function"""
    return f"You said: {message}"

# Custom CSS for professional styling
css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=css, title="Test Copilot") as demo:
    gr.Markdown("# Test Sales Copilot")

    chatbot = gr.Chatbot(
        height=500,
        placeholder="Test chat...",
        show_label=False,
        type="messages"
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Your message",
            label="Your Question",
            scale=4
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

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

if __name__ == "__main__":
    print("🚀 Starting Test Copilot...")

    try:
        print("🔄 Launching Gradio interface...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            show_error=True,
            debug=True
        )
        print("✅ Gradio interface launched successfully")
    except Exception as e:
        print(f"❌ Error launching Gradio: {str(e)}")
        import traceback
        traceback.print_exc()