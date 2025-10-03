import gradio as gr
import os
import html as _html
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
    # Sanitize incoming messages: drop any with empty/whitespace-only content and ensure keys exist
    sanitized = []
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if content and isinstance(content, str) and content.strip():
            role = m.get("role", "user")
            sanitized.append({"role": role, "content": content})
    # If no valid messages remain, return empty stream
    if not sanitized:
        return ""

    messages = sanitized

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
                    yield batch  # Yield the batch, not full_response
                    batch = ""
            # Yield remaining batch
            if batch:
                full_response += batch
                yield batch  # Yield the batch, not full_response
    
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
                        yield batch  # Yield the batch, not full_response
                        batch = ""
            # Yield remaining batch
            if batch:
                full_response += batch
                yield batch  # Yield the batch, not full_response
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
        
        # Ensure conversation ends with user message for Gemini
        if gemini_messages and gemini_messages[-1]["role"] != "user":
            # If last message is from model, we need to restructure or handle differently
            # For now, we'll add a simple continuation prompt
            gemini_messages.append({"role": "user", "parts": ["Please continue or elaborate."]})
        
        response = model.generate_content(
            gemini_messages,
            stream=True
        )
        
        for chunk in response:
            # Prefer quick accessor but guard against the ValueError when no Part is present.
            text = None
            try:
                if hasattr(chunk, "text"):
                    t = chunk.text
                    if isinstance(t, str) and t.strip():
                        text = t
            except Exception:
                text = None

            # Fallback: try to collect text from candidates -> parts if quick accessor failed
            if not text:
                try:
                    candidates = getattr(chunk, "candidates", None)
                    if candidates:
                        parts = []
                        for cand in candidates:
                            cand_text = getattr(cand, "text", None)
                            if isinstance(cand_text, str) and cand_text.strip():
                                parts.append(cand_text)
                            cand_parts = getattr(cand, "parts", None)
                            if cand_parts:
                                for p in cand_parts:
                                    p_text = getattr(p, "text", None) or getattr(p, "content", None)
                                    if isinstance(p_text, str) and p_text.strip():
                                        parts.append(p_text)
                        if parts:
                            text = "".join(parts)
                except Exception:
                    text = None

            if not text:
                # nothing useful in this chunk
                continue

            batch += text
            if len(batch) >= batch_size:
                full_response += batch
                yield batch  # Yield the batch, not full_response
                batch = ""
        # Yield remaining batch
        if batch:
            full_response += batch
            yield batch  # Yield the batch, not full_response
    
    return full_response

def format_html_for_output(text: str, elem_id: str, height: int = 360) -> str:
    """Simple scrollable container that auto-scrolls using CSS and a scroll anchor."""
    safe = _html.escape(text or "")
    return (
        f'<div style="overflow-y:auto;height:{height}px;padding:8px;border:1px solid rgba(0,0,0,0.08);'
        f'background:var(--background-secondary, #fff);color:var(--text-primary,#000);'
        f'scroll-behavior:smooth;">'
        f'<pre style="white-space:pre-wrap;margin:0">{safe}</pre>'
        f'<div style="height:1px;overflow-anchor:auto;" id="{elem_id}-anchor"></div>'
        f'</div>'
        f'<script>document.getElementById("{elem_id}-anchor")?.scrollIntoView();</script>'
    )

def format_text_for_output(text: str) -> str:
    """Simple text formatting for Gradio textbox output"""
    return text or ""

def critique_and_review(
    user_prompt: str,
    primary_model: str,
    critique_model: str,
    history: List[Tuple[str, str]]
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """
    Execute single round of Critique-and-Review with streaming
    Yields: (updated_history, primary_output_text, critique_output_text)
    """
    
    # Build conversation history (skip empty messages)
    messages = []
    for user_msg, assistant_msg in history:
        if user_msg and str(user_msg).strip():
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg and str(assistant_msg).strip():
            messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": user_prompt})
    
    # Step 1: Primary Response
    primary_output = f"### 🤖 Initial Response\n**Model:** {primary_model}\n\n"
    critique_output = ""
    # initial empty UI with wrappers
    yield history, primary_output, critique_output
    
    primary_response = ""
    for chunk in stream_model(messages, primary_model):
        # accumulate full stream so final stored response is complete
        primary_response += chunk
        yield history, primary_output + primary_response, critique_output
    
    # Step 2: Critique
    critique_output = f"### 🔍 Critique\n**Model:** {critique_model}\n\n"
    yield history, primary_output + primary_response, critique_output
    
    critique_context = messages + [{"role": "assistant", "content": primary_response}]
    
    critique_response = ""
    for chunk in stream_model(critique_context, critique_model):
        critique_response += chunk
        yield history, primary_output + primary_response, critique_output + critique_response
    
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
        final_response += chunk
        yield history, primary_output + final_response, critique_output + critique_response
    
    # Update history with final response
    updated_history = history + [(user_prompt, final_response)]
    
    yield updated_history, primary_output + final_response, critique_output + critique_response

def single_reply(
    user_prompt: str,
    primary_model: str,
    history: List[Tuple[str, str]]
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """
    Produce a single-model reply (no critique) and keep full conversation context.
    Yields: (updated_history, primary_output_text, critique_output_text)
    """
    # Build conversation history (skip empty messages)
    messages = []
    for user_msg, assistant_msg in history:
        if user_msg and str(user_msg).strip():
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg and str(assistant_msg).strip():
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_prompt})
    
    primary_output = f"### 🤖 Single Reply\n**Model:** {primary_model}\n\n"
    critique_output = ""  # no critique for this path
    yield history, primary_output, critique_output
    
    primary_response = ""
    for chunk in stream_model(messages, primary_model):
        primary_response += chunk
        yield history, primary_output + primary_response, critique_output
    
    updated_history = history + [(user_prompt, primary_response)]
    yield updated_history, primary_output + primary_response, critique_output

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
    
    # Show the initial (seed) question at the top and keep it visible
    initial_question_html = _html.escape(DEFAULT_QUESTION)
    initial_question_display = gr.HTML(
        f'<div style="padding:10px;border-radius:6px;border:1px solid rgba(0,0,0,0.06);'
        f'background:var(--background-secondary,#f8f9fa);margin-bottom:10px;"><strong>Initial Question:</strong>'
        f'<pre style="white-space:pre-wrap;margin:6px 0 0 0;font-size:14px;">{initial_question_html}</pre></div>',
        label="Current Question",
        container=True
    )
    
    # Use Textbox with fixed height for auto-scrolling outputs
    with gr.Row():
        with gr.Column(scale=1):
            primary_output = gr.Textbox(
                value="",
                label="Primary Model",
                lines=25,
                max_lines=25,  # Fixed height - prevents growing
                autoscroll=True,
                container=True,
                interactive=False,
                show_copy_button=True  # Useful for long responses
            )
        
        with gr.Column(scale=1):
            critique_output = gr.Textbox(
                value="",
                label="Critique Model", 
                lines=25,
                max_lines=25,  # Fixed height - prevents growing
                autoscroll=True,
                container=True,
                interactive=False,
                show_copy_button=True  # Useful for long responses
            )
    
    # Hidden state for conversation history
    conversation_state = gr.State([])
    
    # User controls below the outputs
    user_input = gr.Textbox(
        lines=4,
        value="",
        label="Follow-up Question or New Topic",
        placeholder="Enter a follow-up question, ask for clarification, or introduce a new topic..."
    )
    
    # Single set of buttons that work for both initial and follow-up
    with gr.Row():
        cr_btn = gr.Button("▶️ Critique-and-Review", variant="primary", size="lg")
        single_btn = gr.Button("💬 Single Reply", size="lg")
        reset_btn = gr.Button("🔄 Reset", size="lg")
    
    # Helper functions
    def handle_cr_click(user_input_val, primary_model_val, critique_model_val, history):
        # If user input is empty, use the default question
        prompt = user_input_val.strip() if user_input_val.strip() else DEFAULT_QUESTION
        # Return the generator directly for streaming
        for result in critique_and_review(prompt, primary_model_val, critique_model_val, history):
            yield result
    
    def handle_single_click(user_input_val, primary_model_val, history):
        # If user input is empty, use the default question
        prompt = user_input_val.strip() if user_input_val.strip() else DEFAULT_QUESTION
        # Return the generator directly for streaming
        for result in single_reply(prompt, primary_model_val, history):
            yield result
    
    def handle_reset():
        return [], "", "", ""
    
    # Event handlers
    cr_btn.click(
        fn=handle_cr_click,
        inputs=[user_input, primary_model, critique_model, conversation_state],
        outputs=[conversation_state, primary_output, critique_output]
    )
    
    single_btn.click(
        fn=handle_single_click,
        inputs=[user_input, primary_model, conversation_state],
        outputs=[conversation_state, primary_output, critique_output]
    )
    
    reset_btn.click(
        fn=handle_reset,
        outputs=[conversation_state, primary_output, critique_output, user_input]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False
    )
