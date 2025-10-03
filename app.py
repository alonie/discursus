import gradio as gr
import anthropic
import openai
import os
from typing import List, Tuple

# Initialize API clients
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_QUESTION = """A mid-sized country faces a resurgence of a novel respiratory virus. Vaccination rates have plateaued between 45-65%, ICU capacity varies between 75-95% across regions, and economic recovery remains fragile. Recent epidemiological studies suggest transmission rates may be 30-60% higher than initial models predicted. The government is considering reintroducing strict lockdowns for 4-8 weeks to suppress transmission before winter. Should it do so? Justify your position with evidence-backed analysis of epidemiological risk, economic stability, civil liberties, and public trust. Provide specific citations for key empirical claims and measurable predictions your approach would generate."""

def call_anthropic(messages: List[dict]) -> str:
    """Call Anthropic Claude API"""
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=messages
    )
    return response.content[0].text

def call_openai(messages: List[dict]) -> str:
    """Call OpenAI GPT API"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000
    )
    return response.choices[0].message.content

def critique_and_review(
    user_prompt: str,
    primary_model: str,
    critique_model: str,
    history: List[Tuple[str, str]]
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Execute single round of Critique-and-Review
    Returns: (updated_history, process_log)
    """
    
    process_log = "### Discursus Process Log\n\n"
    
    # Build conversation history
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add current user prompt
    messages.append({"role": "user", "content": user_prompt})
    
    # Step 1: Primary LLM Response
    process_log += "**Step 1: Primary Response**\n"
    process_log += f"Model: {primary_model}\n\n"
    
    if primary_model == "Claude (Anthropic)":
        primary_response = call_anthropic(messages)
    else:
        primary_response = call_openai(messages)
    
    process_log += f"Response: {primary_response[:200]}...\n\n"
    
    # Step 2: Critique
    process_log += "**Step 2: Critique**\n"
    process_log += f"Model: {critique_model}\n\n"
    
    critique_context = messages + [{"role": "assistant", "content": primary_response}]
    
    if critique_model == "Claude (Anthropic)":
        critique_response = call_anthropic(critique_context)
    else:
        critique_response = call_openai(critique_context)
    
    process_log += f"Critique: {critique_response[:200]}...\n\n"
    
    # Step 3: Review
    process_log += "**Step 3: Revised Response**\n"
    process_log += f"Model: {primary_model}\n\n"
    
    review_context = critique_context + [{"role": "user", "content": critique_response}]
    
    if primary_model == "Claude (Anthropic)":
        final_response = call_anthropic(review_context)
    else:
        final_response = call_openai(review_context)
    
    # Update history with final response
    updated_history = history + [(user_prompt, final_response)]
    
    # Create display output
    full_output = f"""### Primary Response
{primary_response}

---

### Critique
{critique_response}

---

### Revised Response
{final_response}"""
    
    return updated_history, full_output, process_log

def reset_conversation():
    """Reset conversation state"""
    return [], "", ""

# Create Gradio Interface
with gr.Blocks(title="Discursus: Critique-and-Review") as demo:
    gr.Markdown("""
    # Discursus: Critique-and-Review System
    
    This system runs a single round of deliberative critique between two LLMs:
    1. **Primary LLM** generates initial response
    2. **Critique LLM** reviews and critiques the response
    3. **Primary LLM** revises its position based on critique
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
    process_log = gr.Markdown(label="Process Log", visible=False)
    
    # Hidden state for conversation history
    conversation_state = gr.State([])
    
    submit_btn.click(
        fn=critique_and_review,
        inputs=[user_input, primary_model, critique_model, conversation_state],
        outputs=[conversation_state, output_display, process_log]
    )
    
    reset_btn.click(
        fn=reset_conversation,
        outputs=[conversation_state, output_display, process_log]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False
    )