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

SUGGESTED_QUESTION = """A mid-sized country faces a resurgence of a novel respiratory virus. Vaccination rates have plateaued between 45-65%, ICU capacity varies between 75-95% across regions, and economic recovery remains fragile. Recent epidemiological studies suggest transmission rates may be 30-60% higher than initial models predicted. The government is considering reintroducing strict lockdowns for 4-8 weeks to suppress transmission before winter. Should it do so? Justify your position with evidence-backed analysis of epidemiological risk, economic stability, civil liberties, and public trust. Provide specific citations for key empirical claims and measurable predictions your approach would generate."""

MODEL_MAP = {
    "Claude 4.5 Sonnet": ("anthropic", "claude-sonnet-4-20250514"),
    "GPT-4o": ("openai", "gpt-4o"),
    "GPT-4o Mini": ("openai", "gpt-4o-mini"),
    "Gemini 2.5 Flash": ("gemini", "gemini-2.5-flash"),
    "Gemini 2.5 Pro": ("gemini", "gemini-2.5-pro")
}

def process_uploaded_files(files) -> str:
    """Process uploaded text files and return their content as a formatted string"""
    if not files:
        return ""
    
    file_contents = []
    
    for file in files:
        try:
            # Read file content
            with open(file.name, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get just the filename without the full path
            filename = os.path.basename(file.name)
            
            # Format the content with clear delimiters
            file_contents.append(f"=== FILE: {filename} ===\n{content}\n=== END FILE ===\n")
            
        except UnicodeDecodeError:
            # Handle non-text files gracefully
            filename = os.path.basename(file.name)
            file_contents.append(f"=== FILE: {filename} ===\n[Error: File appears to be binary or uses unsupported encoding]\n=== END FILE ===\n")
        except Exception as e:
            filename = os.path.basename(file.name) if hasattr(file, 'name') else "unknown"
            file_contents.append(f"=== FILE: {filename} ===\n[Error reading file: {str(e)}]\n=== END FILE ===\n")
    
    return "\n".join(file_contents)

def build_prompt_with_context(user_prompt: str, uploaded_files, system_context: str = "") -> str:
    """Combine user prompt with file context and any system context"""
    parts = []
    
    # Add system context if provided
    if system_context and system_context.strip():
        parts.append(f"CONTEXT:\n{system_context}\n")
    
    # Add uploaded file contents
    file_context = process_uploaded_files(uploaded_files)
    if file_context:
        parts.append(f"UPLOADED FILES FOR REFERENCE:\n{file_context}")
    
    # Add the user's actual question/prompt
    parts.append(f"USER QUESTION:\n{user_prompt}")
    
    return "\n".join(parts)

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

def critique_and_review(
    user_question: str,
    primary_model: str,
    critique_model: str,
    history: List[Tuple[str, str]],
    uploaded_files=None
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """
    Execute Critique-and-Review with streaming
    Yields: (updated_history, primary_output_text, critique_output_text)
    """
    
    # Build the complete prompt with file context
    complete_prompt = build_prompt_with_context(user_question, uploaded_files)
    
    # Build conversation history (skip empty messages)
    messages = []
    for user_msg, assistant_msg in history:
        if user_msg and str(user_msg).strip():
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg and str(assistant_msg).strip():
            messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": complete_prompt})
    
    # Determine header based on conversation state
    response_type = "Initial Response" if not history else "Follow-up Response"
    
    # Step 1: Primary Response
    primary_output = f"### 🤖 {response_type}\n**Model:** {primary_model}\n\n"
    critique_output = ""
    yield history, primary_output, critique_output
    
    primary_response = ""
    for chunk in stream_model(messages, primary_model):
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
    updated_history = history + [(user_question, final_response)]
    
    yield updated_history, primary_output + final_response, critique_output + critique_response

def single_reply(
    user_question: str,
    primary_model: str,
    history: List[Tuple[str, str]],
    uploaded_files=None
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """
    Produce a single-model reply (no critique) and keep full conversation context.
    Yields: (updated_history, primary_output_text, critique_output_text)
    """
    # Build the complete prompt with file context
    complete_prompt = build_prompt_with_context(user_question, uploaded_files)
    
    # Build conversation history (skip empty messages)
    messages = []
    for user_msg, assistant_msg in history:
        if user_msg and str(user_msg).strip():
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg and str(assistant_msg).strip():
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": complete_prompt})
    
    # Determine header based on conversation state
    response_type = "Single Reply" if not history else "Follow-up Single Reply"
    
    primary_output = f"### 🤖 {response_type}\n**Model:** {primary_model}\n\n"
    critique_output = ""  # no critique for this path
    yield history, primary_output, critique_output
    
    primary_response = ""
    for chunk in stream_model(messages, primary_model):
        primary_response += chunk
        yield history, primary_output + primary_response, critique_output
    
    # Update history with final response
    updated_history = history + [(user_question, primary_response)]
    yield updated_history, primary_output + primary_response, critique_output

# Create Gradio Interface
with gr.Blocks(title="Discursus: Critique-and-Review", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 Discursus: Critique-and-Review System
    
    Multi-model deliberation system: Primary model generates response → Critique model reviews → Primary model revises
    
    **File Upload**: You can upload text files (.txt, .md, .py, etc.) to provide additional context for your questions.
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
    
    # Initial question input - now editable by user
    initial_question_input = gr.Textbox(
        lines=8,
        value=SUGGESTED_QUESTION,
        label="Initial Question",
        placeholder="Enter your initial question or modify the suggested one...",
        container=True
    )
    
    # File upload for initial question context
    initial_file_upload = gr.File(
        label="📎 Upload Context Files for Initial Question (Optional)",
        file_count="multiple",
        file_types=["text", ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv"],
        height=100
    )
    
    # Buttons for initial question
    with gr.Row():
        initial_cr_btn = gr.Button("🚀 Start with Critique-and-Review", variant="primary", size="lg")
        initial_single_btn = gr.Button("🚀 Start with Single Reply", size="lg")
    
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
    
    # Follow-up controls below the outputs
    gr.Markdown("### Follow-up Questions")
    with gr.Row():
        with gr.Column(scale=2):
            follow_up_input = gr.Textbox(
                lines=4,
                value="",
                label="Follow-up Question or New Topic",
                placeholder="Enter a follow-up question, ask for clarification, or introduce a new topic..."
            )
        with gr.Column(scale=1):
            follow_up_file_upload = gr.File(
                label="📎 Upload Additional Context Files (Optional)",
                file_count="multiple",
                file_types=["text", ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv"],
                height=120
            )
    
    # Follow-up buttons
    with gr.Row():
        follow_up_cr_btn = gr.Button("▶️ Follow-up Critique-and-Review", variant="primary", size="lg")
        follow_up_single_btn = gr.Button("💬 Follow-up Single Reply", size="lg")
        reset_btn = gr.Button("🔄 Reset", size="lg")
    
    # Helper functions for initial question
    def handle_initial_cr_click(initial_question, primary_model_val, critique_model_val, history, uploaded_files):
        prompt = initial_question.strip() if initial_question.strip() else SUGGESTED_QUESTION
        for result in critique_and_review(prompt, primary_model_val, critique_model_val, history, uploaded_files):
            yield result
    
    def handle_initial_single_click(initial_question, primary_model_val, history, uploaded_files):
        prompt = initial_question.strip() if initial_question.strip() else SUGGESTED_QUESTION
        for result in single_reply(prompt, primary_model_val, history, uploaded_files):
            yield result
    
    # Helper functions for follow-up questions
    def handle_follow_up_cr_click(follow_up_question, primary_model_val, critique_model_val, history, uploaded_files):
        if not follow_up_question.strip():
            return  # Don't process empty follow-up questions
        for result in follow_up_critique_and_review(follow_up_question, primary_model_val, critique_model_val, history, uploaded_files):
            yield result
    
    def handle_follow_up_single_click(follow_up_question, primary_model_val, history, uploaded_files):
        if not follow_up_question.strip():
            return  # Don't process empty follow-up questions
        for result in follow_up_single_reply(follow_up_question, primary_model_val, history, uploaded_files):
            yield result
    
    def handle_reset():
        return [], "", "", "", SUGGESTED_QUESTION, None, None  # Reset everything including initial question and file uploads
    
    # Event handlers for initial question
    initial_cr_btn.click(
        fn=handle_initial_cr_click,
        inputs=[initial_question_input, primary_model, critique_model, conversation_state, initial_file_upload],
        outputs=[conversation_state, primary_output, critique_output]
    )
    
    initial_single_btn.click(
        fn=handle_initial_single_click,
        inputs=[initial_question_input, primary_model, conversation_state, initial_file_upload],
        outputs=[conversation_state, primary_output, critique_output]
    )
    
    # Event handlers for follow-up questions
    follow_up_cr_btn.click(
        fn=handle_follow_up_cr_click,
        inputs=[follow_up_input, primary_model, critique_model, conversation_state, follow_up_file_upload],
        outputs=[conversation_state, primary_output, critique_output]
    )
    
    follow_up_single_btn.click(
        fn=handle_follow_up_single_click,
        inputs=[follow_up_input, primary_model, conversation_state, follow_up_file_upload],
        outputs=[conversation_state, primary_output, critique_output]
    )
    
    reset_btn.click(
        fn=handle_reset,
        outputs=[conversation_state, primary_output, critique_output, follow_up_input, initial_question_input, initial_file_upload, follow_up_file_upload]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False
    )
