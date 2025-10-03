import gradio as gr
import os
from typing import List, Tuple, Generator

# Lazy-load API clients
_anthropic_client = None
_openai_client = None

def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

DEFAULT_QUESTION = """A mid-sized country faces a resurgence of a novel respiratory virus. Vaccination rates have plateaued between 45-65%, ICU capacity varies between 75-95% across regions, and economic recovery remains fragile. Recent epidemiological studies suggest transmission rates may be 30-60% higher than initial models predicted. The government is considering reintroducing strict lockdowns for 4-8 weeks to suppress transmission before winter. Should it do so? Justify your position with evidence-backed analysis of epidemiological risk, economic stability, civil liberties, and public trust. Provide specific citations for key empirical claims and measurable predictions your approach would generate."""

def stream_anthropic(messages: List[dict]) -> Generator[str, None, str]:
    """Stream from Anthropic Claude API"""
    client = get_anthropic_client()
    full_response = ""
    
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            full_response += text
            yield full_response
    
    return full_response

def stream_openai(messages: List[dict]) -> Generator[str, None, str]:
    """Stream from OpenAI GPT API"""
    client = get_openai_client()
    full_response = ""
    
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            yield full_response
    
    return full_response

def critique_and_review(
    user_prompt: str,
    primary_model: str,
    critique_model: str,
    history: List[Tuple[str, str]]
) -> Generator[Tuple[List[Tuple[str, str]], str], None, None]:
    """
    Execute single round of Critique-and-Review with streaming
    Yields: (updated_history, display_output)
    """
    
    # Build conversation history
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": user_prompt})
    
    # Step 1: Primary Response
    output = "### 🤖 Primary Response\n**Model:** " + primary_model + "\n\n"
    yield history, output
    
    primary_response = ""
    if primary_model == "Claude (Anthropic)":
        for chunk in stream_anthropic(messages):
            primary_response = chunk
            yield history, output + chunk
    else:
        for chunk in stream_openai(messages):
            primary_response = chunk
            yield history, output + chunk
    
    output += primary_response + "\n\n---\n\n"
    
    # Step 2: Critique
    output += "### 🔍 Critique\n**Model:** " + critique_model + "\n\n"
    yield history, output
    
    critique_context = messages + [{"role": "assistant", "content": primary_response}]
    
    critique_response = ""
    if critique_model == "Claude (Anthropic)":
        for chunk in stream_anthropic(critique_context):
            critique_response = chunk
            yield history, output + chunk
    else:
        for chunk in stream_openai(critique_context):
            critique_response = chunk
            yield history, output + chunk
    
    output += critique_response + "\n\n---\n\n"
    
    # Step 3: Revised Response
    output += "### ✨ Revised Response\n**Model:** " + primary_model + "\n\n"
    yield history, output
    
    review_context = critique_context + [{"role": "user", "content": critique_response}]
    
    final_response = ""
    if primary_model == "Claude (Anthropic)":
        for chunk in stream_anthropic(review_context):
            final_response = chunk
            yield history, output + chunk
    else:
        for chunk in stream_openai(review_context):
            final_response = chunk
            yield history, output + chunk
    
    # Update history with final response
    updated_history = history + [(user_prompt, final_response)]
    
    yield updated_history, output

def reset_conversation():
    """Reset conversation state"""
    return [], ""

# Create Gradio Interface
with gr.Blocks(title="Discursus: Critique-and-Review") as demo:
    gr.Markdown("""
    # Discursus: Critique-and-Review System
    
    This system runs a single round of deliberative critique between two LLMs:
    1. **Primary LLM** generates initial response (streaming)
    2. **Critique LLM** reviews and critiques the response (streaming)
    3. **Primary LLM** revises its position based on critique (streaming)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            primary_model = gr.Dropdown(
                choices=["Claude (Anthropic)", "GPT-4 (OpenAI)"],
                value="Claude (Anthropic)",
                label="Primary Model"
            )
            critique_model = gr.Dropdown(
                choices=["GPT-4 (OpenAI)", "Claude (Anthropic)"],
                value="GPT-4 (OpenAI)",
                label="Critique Model"
            )
        
        with gr.Column(scale=3):
            user_input = gr.Textbox(
                lines=8,
                value=DEFAULT_QUESTION,
                label="Your Question",
                placeholder="Enter your question here..."
            )
    
    with gr.Row():
        submit_btn = gr.Button("Run Critique-and-Review", variant="primary")
        reset_btn = gr.Button("Reset Conversation")
    
    output_display = gr.Markdown(label="Deliberation Output")
    
    # Hidden state for conversation history
    conversation_state = gr.State([])
    
    submit_btn.click(
        fn=critique_and_review,
        inputs=[user_input, primary_model, critique_model, conversation_state],
        outputs=[conversation_state, output_display]
    )
    
    reset_btn.click(
        fn=reset_conversation,
        outputs=[conversation_state, output_display]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False
    )