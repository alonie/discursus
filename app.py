import gradio as gr
import os
import html as _html
from typing import List, Tuple, Generator
import re

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

def build_messages_with_context(user_prompt: str, history: List[Tuple[str, str]], uploaded_files) -> List[dict]:
    """Combine user prompt with history and file context into a message list."""
    messages = []
    file_context = process_uploaded_files(uploaded_files)
    if file_context:
        messages.append({"role": "user", "content": f"Please use the following files as context for the entire conversation:\n{file_context}\n(This is a system-level instruction and should not be repeated in your response.)"})
        messages.append({"role": "assistant", "content": "Understood. I will use the provided files as context."})

    for user_msg, assistant_msg_html in history:
        if user_msg and str(user_msg).strip():
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg_html and str(assistant_msg_html).strip():
            # Clean the HTML formatting before sending to the model
            content_part = re.search(r"<div class='response-content'.*?>(.*)</div>", assistant_msg_html, re.DOTALL)
            if content_part:
                cleaned_msg = content_part.group(1).replace('<br>', '\n').strip()
                messages.append({"role": "assistant", "content": cleaned_msg})

    messages.append({"role": "user", "content": user_prompt})
    return messages

def generate_styled_block(content: str, purpose: str, model_name: str = "") -> str:
    """Generates a styled HTML block for a message."""
    escaped_content = _html.escape(content).replace('\n', '<br>')

    if purpose == "User":
        return f"<div style='padding: 10px; border-radius: 10px; background-color: #e1f5fe; margin: 10px 40px 10px 5px; text-align: left; border: 1px solid #b3e5fc;'><b>You:</b><br>{escaped_content}</div>"

    style_map = {
        "Response": "background-color: #f9f9f9; border: 1px solid #ddd;",
        "Critique": "background-color: #fffbe6; border: 1px solid #fff1b8;",
        "Revision": "background-color: #e6ffed; border: 1px solid #b3ffc6;"
    }
    purpose_map = {
        "Response": "Primary Response",
        "Critique": "Critique",
        "Revision": "Revised Response"
    }
    
    block_style = style_map.get(purpose, style_map["Response"])
    block_purpose = purpose_map.get(purpose, purpose)
    
    return f"""
    <div style='{block_style} padding: 15px; border-radius: 8px; margin: 10px 5px 10px 40px;'>
        <div style='font-size: 1.1em; font-weight: bold; color: #333; margin-bottom: 8px;'>
            {block_purpose} ({model_name})
        </div>
        <div class='response-content' style='color: #222; line-height: 1.6;'>
            {escaped_content}
        </div>
    </div>
    """

def history_to_html(history: List[Tuple[str, str]]) -> str:
    """Converts conversation history to a single HTML string."""
    html_string = ""
    for user_msg, assistant_msg in history:
        if user_msg:
            html_string += generate_styled_block(user_msg, "User")
        if assistant_msg:
            html_string += assistant_msg
    return html_string

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

def chat_turn(user_question: str, history: List[Tuple[str, str]], primary_model: str, uploaded_files) -> Generator:
    """A single turn in the primary chat conversation."""
    history.append((user_question, None))
    yield history_to_html(history), gr.update(value="")

    messages = build_messages_with_context(user_question, history[:-1], uploaded_files)
    
    response_stream = stream_model(messages, primary_model)
    
    full_response = ""
    for chunk in response_stream:
        full_response += chunk
        history[-1] = (user_question, generate_styled_block(full_response, "Response", primary_model))
        yield history_to_html(history), gr.update(value="")

def critique_and_review_workflow(history: List[Tuple[str, str]], primary_model: str, critique_model: str, uploaded_files) -> Generator:
    """The full C&R workflow, displaying all output in the main display."""
    if not history:
        history.append((None, generate_styled_block("Cannot perform critique on an empty conversation.", "Critique", "System")))
        yield history_to_html(history)
        return

    # 1. Generate Critique
    critique_prompt = "Please provide a concise, constructive critique of the assistant's reasoning, accuracy, and helpfulness throughout the preceding conversation. Identify any potential biases, logical fallacies, or missed opportunities for a more comprehensive response."
    critique_messages = build_messages_with_context(critique_prompt, history, uploaded_files)
    
    history.append((None, generate_styled_block("...", "Critique", critique_model)))
    yield history_to_html(history)

    critique_response = ""
    for chunk in stream_model(critique_messages, critique_model):
        critique_response += chunk
        history[-1] = (None, generate_styled_block(critique_response, "Critique", critique_model))
        yield history_to_html(history)

    # 2. Generate Revised Response
    review_prompt = f"Based on the entire conversation history and the following critique, please provide a revised, improved version of your last response. Synthesize the critique into your reasoning and address any shortcomings identified.\n\n--- CRITIQUE ---\n{critique_response}\n--- END CRITIQUE ---"
    review_messages = build_messages_with_context(review_prompt, history, uploaded_files)

    history.append((None, generate_styled_block("...", "Revision", primary_model)))
    yield history_to_html(history)

    revised_response = ""
    for chunk in stream_model(review_messages, primary_model):
        revised_response += chunk
        history[-1] = (None, generate_styled_block(revised_response, "Revision", primary_model))
        yield history_to_html(history)


with gr.Blocks(title="Discursus", theme=gr.themes.Default(), css="#conversation-container { height: 600px; overflow-y: auto; }") as demo:
    gr.Markdown("# Discursus: A System for Critical LLM Discourse")
    
    with gr.Row():
        primary_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Claude 4.5 Sonnet", label="Primary Model")
        critique_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Gemini 2.5 Pro", label="Critique Model")

    with gr.Column(elem_id="conversation-container"):
        conversation_display = gr.Markdown()
    
    with gr.Row():
        with gr.Column(scale=10):
            user_input = gr.Textbox(show_label=False, placeholder="Enter your message or use the suggested question...", lines=3, value=SUGGESTED_QUESTION)
        with gr.Column(scale=1, min_width=80):
            send_btn = gr.Button("Send", variant="primary")

    with gr.Row():
        critique_btn = gr.Button("🔍 Initiate Critique & Review")
        upload_btn = gr.UploadButton("📎 Upload Files", file_count="multiple", file_types=["text", ".md", ".py", ".csv", ".json"])
        reset_btn = gr.Button("🔄 New Conversation")

    history_state = gr.State([])
    file_state = gr.State([])

    def handle_send(user_question, history, p_model, files):
        if not user_question.strip():
            return history_to_html(history), gr.update(value="")
        for html_output, user_input_update in chat_turn(user_question, history, p_model, files):
            yield html_output, user_input_update

    def handle_critique(history, p_model, c_model, files):
        for html_output in critique_and_review_workflow(history, p_model, c_model, files):
            yield html_output

    def handle_upload(files):
        file_names = [os.path.basename(f.name) for f in files]
        upload_status = f"📎 Uploaded: {', '.join(file_names)}"
        return files, gr.update(value=upload_status)

    autoscroll_js = """
    () => {
        const container = document.querySelector('#conversation-container');
        if (container) {
            const observer = new MutationObserver(() => {
                container.scrollTop = container.scrollHeight;
            });
            observer.observe(container, { childList: true, subtree: true });
        }
    }
    """

    send_btn.click(
        handle_send,
        [user_input, history_state, primary_model, file_state],
        [conversation_display, user_input]
    )

    user_input.submit(
        handle_send,
        [user_input, history_state, primary_model, file_state],
        [conversation_display, user_input]
    )
    
    critique_btn.click(
        handle_critique,
        [history_state, primary_model, critique_model, file_state],
        [conversation_display]
    )

    upload_btn.upload(
        handle_upload,
        [upload_btn],
        [file_state, user_input]
    )

    reset_btn.click(
        lambda: ([], [], "", gr.update(placeholder="Enter your message or use the suggested question...", value="")),
        [],
        [history_state, file_state, conversation_display, user_input]
    )

    demo.load(None, None, None, js=autoscroll_js)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False)
