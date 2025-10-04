import gradio as gr
import os
from typing import List, Tuple, Generator
import re
import tiktoken

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

# --- Token Counting Utility ---
def count_tokens(history: List[dict]) -> int:
    """Counts the total tokens in the conversation history."""
    # Using cl100k_base encoder as a general-purpose choice
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return 0 # In case of issue loading encoding
    
    total_tokens = 0
    for message in history:
        if message.get("content"):
            total_tokens += len(encoding.encode(message["content"]))
    return total_tokens

SUGGESTED_QUESTION = """A mid-sized country faces a resurgence of a novel respiratory virus. Vaccination rates have plateaued between 45-65%, ICU capacity varies between 75-95% across regions, and economic recovery remains fragile. Recent epidemiological studies suggest transmission rates may be 30-60% higher than initial models predicted. The government is considering reintroducing strict lockdowns for 4-8 weeks to suppress transmission before winter. Should it do so? Justify your position with evidence-backed analysis of epidemiological risk, economic stability, civil liberties, and public trust. Provide specific citations for key empirical claims and measurable predictions your approach would generate."""

MODEL_MAP = {
    "Claude 4.5 Sonnet": ("anthropic", "claude-sonnet-4-5"),
    "Claude 4 Sonnet": ("anthropic", "claude-4-sonnet-20250115"),
    "Claude 3.5 Sonnet": ("anthropic", "claude-3-5-sonnet-20240620"),
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

def format_bot_message(content: str, purpose: str, model_name: str) -> str:
    """Formats a bot message with a large, bold Markdown header."""
    purpose_map = {
        "Response": "Primary Response",
        "Critique": "Critique",
        "Revision": "Revised Response"
    }
    # Use Markdown for a bold, larger header and a horizontal rule for separation.
    header = f"<br>### **{purpose_map.get(purpose, purpose)}** ({model_name})\n---\n"
    return header + content

def build_messages_with_context(user_prompt: str, history: List[dict], uploaded_files) -> List[dict]:
    """Combine user prompt with history and file context into a message list."""
    messages = []
    file_context = process_uploaded_files(uploaded_files)
    if file_context:
        messages.append({"role": "user", "content": f"Please use the following files as context for the entire conversation:\n{file_context}\n(This is a system-level instruction and should not be repeated in your response.)"})
        messages.append({"role": "assistant", "content": "Understood. I will use the provided files as context."})

    # Reconstruct the history to ensure only 'role' and 'content' are present.
    # This strips any extra keys Gradio might add (like 'metadata').
    for msg in history:
        # Filter out dummy user messages used for UI justification.
        if msg.get("content") not in ["Critique Request", "Review Request"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    messages.append({"role": "user", "content": user_prompt})
    return messages

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

def chat_turn(user_question: str, history: List[dict], primary_model: str, uploaded_files) -> Generator:
    """A single turn in the primary chat conversation."""
    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": ""})
    yield history, gr.update(value=""), f"Token Count: {count_tokens(history)}"

    messages = build_messages_with_context(user_question, history[:-2], uploaded_files)
    
    response_stream = stream_model(messages, primary_model)
    
    full_response = ""
    for chunk in response_stream:
        full_response += chunk
        history[-1]["content"] = format_bot_message(full_response, "Response", primary_model)
        yield history, gr.update(value=""), f"Token Count: {count_tokens(history)}"

def critique_and_review_workflow(history: List[dict], primary_model: str, critique_model: str, uploaded_files, critique_prompt: str, review_prompt_template: str) -> Generator:
    """The full C&R workflow, displaying all output in the main display."""
    if not history:
        history.append({"role": "assistant", "content": format_bot_message("Cannot perform critique on an empty conversation.", "Critique", "System")})
        yield history, f"Token Count: {count_tokens(history)}"
        return

    # 1. Generate Critique
    critique_messages = build_messages_with_context(critique_prompt, history, uploaded_files)
    
    # Add a dummy user message for display purposes, then the bot response
    history.append({"role": "user", "content": "Critique Request"})
    history.append({"role": "assistant", "content": format_bot_message("...", "Critique", critique_model)})
    yield history, f"Token Count: {count_tokens(history)}"

    critique_response = ""
    for chunk in stream_model(critique_messages, critique_model):
        critique_response += chunk
        history[-1]["content"] = format_bot_message(critique_response, "Critique", critique_model)
        yield history, f"Token Count: {count_tokens(history)}"

    # 2. Generate Revised Response
    review_prompt = f"{review_prompt_template}\n\n--- CRITIQUE ---\n{critique_response}\n--- END CRITIQUE ---"
    review_messages = build_messages_with_context(review_prompt, history, uploaded_files)

    # Add another dummy user message for the revision
    history.append({"role": "user", "content": "Review Request"})
    history.append({"role": "assistant", "content": format_bot_message("...", "Revision", primary_model)})
    yield history, f"Token Count: {count_tokens(history)}"

    revised_response = ""
    for chunk in stream_model(review_messages, primary_model):
        revised_response += chunk
        history[-1]["content"] = format_bot_message(revised_response, "Revision", primary_model)
        yield history, f"Token Count: {count_tokens(history)}"


with gr.Blocks(
    title="Discursus", 
    theme=gr.themes.Default(),
    css="#header-row {align-items: center;}"
) as demo:
    with gr.Row(elem_id="header-row"):
        with gr.Column(scale=1, min_width=100):
            gr.Image("logo.png", height=80, interactive=False, container=False)
        with gr.Column(scale=8):
            gr.Markdown("# Discursus: A System for Critical LLM Discourse")

    with gr.Row():
        primary_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Claude 4.5 Sonnet", label="Primary Model")
        critique_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Gemini 2.5 Pro", label="Critique Model")
        with gr.Column(min_width=200):
            summarize_btn = gr.Button("Summarise Context")
            token_count_display = gr.Textbox(label="Token Count", value="Token Count: 0", interactive=False)

    chatbot = gr.Chatbot(label="Conversation", height=600, type="messages")

    with gr.Row():
        with gr.Column(scale=10):
            user_input = gr.Textbox(show_label=False, placeholder="Enter your message or use the suggested question...", lines=3, value=SUGGESTED_QUESTION)
        with gr.Column(scale=1, min_width=80):
            send_btn = gr.Button("Send", variant="primary")
        with gr.Column(scale=1, min_width=120):
             upload_btn = gr.UploadButton("📎 Upload Files", file_count="multiple", file_types=["text", ".md", ".py", ".csv", ".json"])


    gr.Markdown("---")
    gr.Markdown("### Critique & Review Workflow")
    
    with gr.Row():
        critique_btn = gr.Button("Critique & Review", variant="secondary")

    with gr.Row():
        critique_prompt_textbox = gr.Textbox(
            label="Critique Prompt",
            lines=5,
            value=(
                "Please provide a concise, constructive critique of the assistant's reasoning, accuracy, and helpfulness throughout the preceding conversation. "
                "Identify any potential biases, logical fallacies, or missed opportunities for a more comprehensive response. "
                "Be extremely critical of citations and confirm or refute each specific citation, as existing or hallucinated, and then as relevant or not. "
                "Provide a clear assessment of each and every citation and mark each with a Status (green existing/red hallucinated) and a Relevance (red 'X', or an orange, or yellow or green). "
                "On this basis, and the overall assessment of the response, provide an overall critique rating out of 10 for the response."
            )
        )
        review_prompt_textbox = gr.Textbox(
            label="Review Prompt Template",
            lines=5,
            value="Based on the entire conversation history and the following critique, please provide a revised, improved version of your last response. " \
            "Synthesize the critique into your reasoning and address any shortcomings identified."
        )

    reset_btn = gr.Button("🔄 New Conversation")


    file_state = gr.State([])

    def handle_send(user_question, history, p_model, files):
        if not user_question.strip():
            return history, gr.update(value=""), f"Token Count: {count_tokens(history)}"
        yield from chat_turn(user_question, history, p_model, files)

    def handle_critique(history, p_model, c_model, files, critique_prompt, review_template):
        yield from critique_and_review_workflow(history, p_model, c_model, files, critique_prompt, review_template)

    def handle_upload(files):
        file_names = [os.path.basename(f.name) for f in files]
        upload_status = f"📎 Uploaded: {', '.join(file_names)}"
        return files, gr.update(value=upload_status)

    def handle_summarize(history: List[dict]) -> Generator:
        """Summarizes the conversation history to reduce token count."""
        if len(history) < 5: # Don't summarize very short conversations
            yield history, f"Token Count: {count_tokens(history)}"
            return

        # Keep the first user message and the last 2 turns (4 messages)
        first_user_message = history[0]
        last_turns = history[-4:]
        to_summarize = history[1:-4]

        # Create the prompt for summarization
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in to_summarize])
        summary_prompt = f"Please provide a concise summary of the following conversation:\n\n{conversation_text}"
        
        # Use a fast model for summarization
        summary_messages = [{"role": "user", "content": summary_prompt}]
        
        summary_response = ""
        # Use a fast model like GPT-4o Mini or Gemini Flash for summarization
        summarization_model = "GPT-4o Mini" 
        yield history, "Summarizing..." # Update UI to show activity
        
        for chunk in stream_model(summary_messages, summarization_model):
            summary_response += chunk
        
        summary_message = {
            "role": "assistant", 
            "content": f"<br>### **Summary of Previous Conversation**\n---\n{summary_response}"
        }

        # Reconstruct the history
        new_history = [first_user_message, summary_message] + last_turns
        
        yield new_history, f"Token Count: {count_tokens(new_history)}"


    send_btn.click(
        handle_send,
        [user_input, chatbot, primary_model, file_state],
        [chatbot, user_input, token_count_display]
    )

    user_input.submit(
        handle_send,
        [user_input, chatbot, primary_model, file_state],
        [chatbot, user_input, token_count_display]
    )

    critique_btn.click(
        handle_critique,
        [chatbot, primary_model, critique_model, file_state, critique_prompt_textbox, review_prompt_textbox],
        [chatbot, token_count_display]
    )

    upload_btn.upload(
        handle_upload,
        [upload_btn],
        [file_state, user_input]
    )

    summarize_btn.click(
        handle_summarize,
        [chatbot],
        [chatbot, token_count_display]
    )

    reset_btn.click(
        lambda: ([], [], gr.update(placeholder="Enter your message or use the suggested question...", value=""), "Token Count: 0"),
        [],
        [chatbot, file_state, user_input, token_count_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False)
