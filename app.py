import gradio as gr
import os
from typing import List, Tuple, Generator

# Lazy-load API clients
_anthropic_client = None
_openai_client = None
_gemini_client = None

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

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _gemini_client = genai
    return _gemini_client

DEFAULT_QUESTION = """A mid-sized country faces a resurgence of a novel respiratory virus. Vaccination rates have plateaued between 45-65%, ICU capacity varies between 75-95% across regions, and economic recovery remains fragile. Recent epidemiological studies suggest transmission rates may be 30-60% higher than initial models predicted. The government is considering reintroducing strict lockdowns for 4-8 weeks to suppress transmission before winter. Should it do so? Justify your position with evidence-backed analysis of epidemiological risk, economic stability, civil liberties, and public trust. Provide specific citations for key empirical claims and measurable predictions your approach would generate."""

MODEL_MAP = {
    "Claude 4.5 Sonnet": ("anthropic", "claude-sonnet-4-20250514"),
    "GPT-5": ("openai", "gpt-5"),
    "GPT-5 Mini": ("openai", "gpt-5-mini"),
    "Gemini 2.5 Flash": ("gemini", "gemini-2.5-flash"),
    "Gemini 2.5 Pro": ("gemini", "gemini-2.5-pro")
}

def stream_model(messages: List[dict], model_name: str) -> Generator[str, None, str]:
    """Stream from any model with batched updates for smoother display"""
    provider, model_id = MODEL_MAP[model_name]
    full_response = ""
    batch = ""
    batch_size = 5  # Accumulate tokens before yielding
    
    if provider == "anthropic":
        client = get_anthropic_client()
        with client.messages.stream(
            model=model_id,
            max_tokens=2000,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                batch += text
                if len(batch) >= batch_size:
                    full_response += batch
                    batch = ""
                    yield full_response
            # Yield remaining batch
            if batch:
                full_response += batch
                yield full_response
    
    elif provider == "openai":
        client = get_openai_client()
        try:
            # Try streaming first
            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_completion_tokens=2000,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    batch += chunk.choices[0].delta.content
                    if len(batch) >= batch_size:
                        full_response += batch
                        batch = ""
                        yield full_response
            # Yield remaining batch
            if batch:
                full_response += batch
                yield full_response
        except Exception as e:
            # Fallback to non-streaming if organization not verified
            if "must be verified to stream" in str(e):
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_completion_tokens=2000,
                    stream=False
                )
                full_response = response.choices[0].message.content
                yield full_response
            else:
                raise
    
    elif provider == "gemini":
        genai = get_gemini_client()
        
        # Proper Gemini safety settings format
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        model = genai.GenerativeModel(
            model_id,
            safety_settings=safety_settings
        )
        
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})
        
        response = model.generate_content(
            gemini_messages,
            stream=True
        )
        
        for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                batch += chunk.text
                if len(batch) >= batch_size:
                    full_response += batch
                    batch = ""
                    yield full_response
        # Yield remaining batch
        if batch:
            full_response += batch
            yield full_response
    
    return full_response

def critique_and_review(
    user_prompt: str,
    primary_model: str,
    critique_model: str,
    history: List[Tuple[str, str]]
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """
    Execute single round of Critique-and-Review with streaming
    Yields: (updated_history, primary_output, critique_output)
    """
    
    # Build conversation history
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": user_prompt})
    
    # Step 1: Primary Response
    primary_output = f"### 🤖 Initial Response\n**Model:** {primary_model}\n\n"
    critique_output = ""
    yield history, primary_output, critique_output
    
    primary_response = ""
    for chunk in stream_model(messages, primary_model):
        primary_response = chunk
        yield history, primary_output + chunk, critique_output
    
    # Step 2: Critique
    critique_output = f"### 🔍 Critique\n**Model:** {critique_model}\n\n"
    yield history, primary_output + primary_response, critique_output
    
    critique_context = messages + [{"role": "assistant", "content": primary_response}]
    
    critique_response = ""
    for chunk in stream_model(critique_context, critique_model):
        critique_response = chunk
        yield history, primary_output + primary_response, critique_output + chunk
    
    # Validate critique response
    if not critique_response or not critique_response.strip():
        critique_response = "No critique provided - response was blocked or empty."
        yield history, primary_output + primary_response, critique_output + critique_response
    
    # Step 3: Revised Response
    primary_output += primary_response + "\n\n---\n\n### ✨ Revised Response\n**Model:** " + primary_model + "\n\n"
    yield history, primary_output, critique_output + critique_response
    
    review_context = critique_context + [{"role": "user", "content": critique_response}]
    
    final_response = ""
    for chunk in stream_model(review_context, primary_model):
        final_response = chunk
        yield history, primary_output + chunk, critique_output + critique_response
    
    # Update history with final response
    updated_history = history + [(user_prompt, final_response)]
    
    yield updated_history, primary_output + final_response, critique_output + critique_response

def reset_conversation():
    """Reset conversation state"""
    return [], "", ""

# Create Gradio Interface
with gr.Blocks(title="Discursus: Critique-and-Review", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 Discursus: Critique-and-Review System
    
    Multi-model deliberation system: Primary model generates response → Critique model reviews → Primary model revises
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            primary_model = gr.Dropdown(
                choices=list(MODEL_MAP.keys()),
                value="Claude 4.5 Sonnet",
                label="🤖 Primary Model"
            )
            critique_model = gr.Dropdown(
                choices=list(MODEL_MAP.keys()),
                value="Gemini 2.5 Pro",
                label="🔍 Critique Model"
            )
    
    user_input = gr.Textbox(
        lines=6,
        value=DEFAULT_QUESTION,
        label="Your Question",
        placeholder="Enter your question here..."
    )
    
    with gr.Row():
        submit_btn = gr.Button("▶️ Run Critique-and-Review", variant="primary", size="lg")
        reset_btn = gr.Button("🔄 Reset", size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            primary_output = gr.Markdown(label="Primary Model", container=True)
        
        with gr.Column(scale=1):
            critique_output = gr.Markdown(label="Critique Model", container=True)
    
    # Hidden state for conversation history
    conversation_state = gr.State([])
    
    submit_btn.click(
        fn=critique_and_review,
        inputs=[user_input, primary_model, critique_model, conversation_state],
        outputs=[conversation_state, primary_output, critique_output]
    )
    
    reset_btn.click(
        fn=reset_conversation,
        outputs=[conversation_state, primary_output, critique_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False
    )