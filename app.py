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
            with open(file.name, 'r', encoding='utf-8') as f:
                content = f.read()
            filename = os.path.basename(file.name)
            file_contents.append(f"=== FILE: {filename} ===\n{content}\n=== END FILE ===\n")
        except Exception as e:
            filename = os.path.basename(file.name) if hasattr(file, 'name') else "unknown"
            file_contents.append(f"=== FILE: {filename} ===\n[Error reading file: {str(e)}]\n=== END FILE ===\n")
    
    return "\n".join(file_contents)

def build_prompt_with_context(user_prompt: str, uploaded_files) -> str:
    """Combine user prompt with file context"""
    file_context = process_uploaded_files(uploaded_files)
    if file_context:
        return f"UPLOADED FILES FOR REFERENCE:\n{file_context}\nUSER QUESTION:\n{user_prompt}"
    return user_prompt

def stream_model(messages: List[dict], model_name: str) -> Generator[str, None, str]:
    """Stream from any model, handling different providers."""
    provider, model_id = MODEL_MAP[model_name]
    
    if not messages:
        return

    def _stream_logic(streamer):
        batch = ""
        batch_size = 5
        for chunk_text in streamer:
            if chunk_text:
                batch += chunk_text
                if len(batch) >= batch_size:
                    yield batch
                    batch = ""
        if batch:
            yield batch

    if provider == "anthropic":
        client = get_anthropic_client()
        with client.messages.stream(model=model_id, max_tokens=2000, messages=messages) as stream:
            yield from _stream_logic(stream.text_stream)
    
    elif provider == "openai":
        client = get_openai_client()
        stream = client.chat.completions.create(model=model_id, messages=messages, stream=True)
        def openai_streamer():
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        yield from _stream_logic(openai_streamer())

    elif provider == "gemini":
        genai = get_gemini_client()
        model = genai.GenerativeModel(model_id)
        gemini_messages = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in messages]
        if gemini_messages and gemini_messages[-1]["role"] != "user":
            gemini_messages.append({"role": "user", "parts": ["Please continue or elaborate."]})
        
        response = model.generate_content(gemini_messages, stream=True)
        def gemini_streamer():
            for chunk in response:
                try:
                    if chunk.text:
                        yield chunk.text
                except Exception:
                    continue
        yield from _stream_logic(gemini_streamer())

def critique_and_review(user_question: str, primary_model: str, critique_model: str, history: List[Tuple[str, str]], uploaded_files) -> Generator[Tuple[List[Tuple[str, str]], str, str, gr.update, gr.update], None, None]:
    """Execute a full Critique-and-Review cycle."""
    complete_prompt = build_prompt_with_context(user_question, uploaded_files)
    
    messages = []
    for user_msg, assistant_msg in history:
        if user_msg and str(user_msg).strip():
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg and str(assistant_msg).strip():
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": complete_prompt})

    response_type = "Initial Response" if not history else "Follow-up Response"
    primary_output = f"### 🤖 {response_type}\n**Model:** {primary_model}\n\n"
    critique_output = ""
    yield history, primary_output, critique_output, gr.update(), gr.update()

    primary_response = ""
    for chunk in stream_model(messages, primary_model):
        primary_response += chunk
        yield history, primary_output + primary_response, critique_output, gr.update(), gr.update()

    critique_output = f"### 🔍 Critique\n**Model:** {critique_model}\n\n"
    yield history, primary_output + primary_response, critique_output, gr.update(), gr.update()
    
    critique_context = messages + [{"role": "assistant", "content": primary_response}]
    critique_response = ""
    for chunk in stream_model(critique_context, critique_model):
        critique_response += chunk
        yield history, primary_output + primary_response, critique_output + critique_response, gr.update(), gr.update()

    primary_output += primary_response + "\n\n---\n\n> ### ✨ Revised Response\n> **Model:** " + primary_model + "\n\n"
    yield history, primary_output, critique_output + critique_response, gr.update(), gr.update()
    
    review_context = critique_context + [{"role": "user", "content": critique_response}]
    final_response = ""
    for chunk in stream_model(review_context, primary_model):
        final_response += chunk
        indented_chunk = "> " + chunk.replace("\n", "\n> ")
        yield history, primary_output + indented_chunk, critique_output + critique_response, gr.update(), gr.update()

    updated_history = history + [(user_question, final_response)]
    
    yield updated_history, primary_output, critique_output + critique_response, gr.update(value="", placeholder="Enter a follow-up question..."), gr.update(value=None)

def single_reply(user_question: str, primary_model: str, history: List[Tuple[str, str]], uploaded_files) -> Generator[Tuple[List[Tuple[str, str]], str, str, gr.update, gr.update], None, None]:
    """Produce a single-model reply."""
    complete_prompt = build_prompt_with_context(user_question, uploaded_files)
    
    messages = []
    for user_msg, assistant_msg in history:
        if user_msg and str(user_msg).strip():
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg and str(assistant_msg).strip():
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": complete_prompt})

    response_type = "Single Reply" if not history else "Follow-up Single Reply"
    primary_output = f"### 🤖 {response_type}\n**Model:** {primary_model}\n\n"
    critique_output = ""
    yield history, primary_output, critique_output, gr.update(), gr.update()

    primary_response = ""
    for chunk in stream_model(messages, primary_model):
        primary_response += chunk
        yield history, primary_output + primary_response, critique_output, gr.update(), gr.update()

    updated_history = history + [(user_question, primary_response)]
    
    yield updated_history, primary_output + primary_response, critique_output, gr.update(value="", placeholder="Enter a follow-up question..."), gr.update(value=None)

with gr.Blocks(title="Discursus: Critique-and-Review", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 Discursus: Critique-and-Review System")
    
    with gr.Row():
        primary_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Claude 4.5 Sonnet", label="🤖 Primary Model", scale=1)
        critique_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Gemini 2.5 Pro", label="🔍 Critique Model", scale=1)

    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(lines=4, value=SUGGESTED_QUESTION, label="Question", placeholder="Enter your question or modify the suggested one...")
        with gr.Column(scale=1):
            with gr.Row():
                cr_btn = gr.Button("🚀 C&R", variant="primary")
                single_btn = gr.Button("💬 Single",)
                reset_btn = gr.Button("🔄 Reset")
                upload_btn = gr.UploadButton("📎", file_count="multiple", file_types=["text", ".md", ".py", ".csv", ".json"])
            file_status = gr.Textbox(label="Uploaded Files", interactive=False, placeholder="No files uploaded")

    with gr.Row():
        primary_output = gr.Textbox(label="Primary Model", lines=30, max_lines=30, autoscroll=True, interactive=False, show_copy_button=True)
        critique_output = gr.Textbox(label="Critique Model", lines=30, max_lines=30, autoscroll=True, interactive=False, show_copy_button=True)
    
    conversation_state = gr.State([])
    file_state = gr.State([])

    def handle_cr_click(user_question, p_model, c_model, history, files):
        if not user_question.strip(): 
            yield history, "", "", gr.update(), gr.update()
            return
        for result in critique_and_review(user_question, p_model, c_model, history, files):
            yield result

    def handle_single_click(user_question, p_model, history, files):
        if not user_question.strip(): 
            yield history, "", "", gr.update(), gr.update()
            return
        for result in single_reply(user_question, p_model, history, files):
            yield result

    def handle_reset():
        return [], "", "", SUGGESTED_QUESTION, None, "No files uploaded"

    def update_file_status(files):
        if not files:
            return "No files uploaded"
        return "\n".join([os.path.basename(f.name) for f in files])

    upload_btn.upload(update_file_status, upload_btn, file_status)
    
    cr_btn.click(handle_cr_click, [user_input, primary_model, critique_model, conversation_state, upload_btn], [conversation_state, primary_output, critique_output, user_input, file_state])
    single_btn.click(handle_single_click, [user_input, primary_model, conversation_state, upload_btn], [conversation_state, primary_output, critique_output, user_input, file_state])
    reset_btn.click(handle_reset, outputs=[conversation_state, primary_output, critique_output, user_input, file_state, file_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False)
