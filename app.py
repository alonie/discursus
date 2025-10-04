import gradio as gr
import os
from typing import List, Tuple, Generator
import re
import tiktoken
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# Lazy-load API clients
_openrouter_client = None
_openai_client = None
_anthropic_client = None
_gemini_client = None

def get_openrouter_client():
    """Initializes the OpenAI client to connect to OpenRouter."""
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    return _openrouter_client

def get_openai_client():
    """Initializes the native OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

def get_anthropic_client():
    """Initializes the native Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client

def get_gemini_client():
    """Initializes the native Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _gemini_client = genai
    return _gemini_client

# --- Token & Cost Utilities ---
def get_encoder():
    """Gets the tiktoken encoder."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def count_tokens(text: str, encoder) -> int:
    """Counts tokens in a single string."""
    if not encoder or not text:
        return 0
    return len(encoder.encode(text))

def calculate_cost_and_tokens(history: List[dict], model_map: dict):
    """Calculates the total cost and token count for the conversation history."""
    total_tokens = 0
    total_cost = 0.0
    encoder = get_encoder()
    if not encoder:
        return 0, 0.0

    # Create a reverse map from model ID to display name
    id_to_name_map = {v['id']: k for k, v in model_map.items()}

    # Find the last explicit model used for user inputs
    last_model_name = "GPT-5" # Default fallback
    for message in reversed(history):
        if message["role"] == "assistant":
            content = message.get("content", "")
            match = re.search(r'\((.*?)\)', content)
            if match:
                last_model_name = match.group(1)
                break
    
    for message in history:
        content = message.get("content", "")
        if not content:
            continue
            
        num_tokens = count_tokens(content, encoder)
        total_tokens += num_tokens
        
        model_name = last_model_name # Assume user input is for the last used model
        cost_key = "input_cost_pm"

        if message["role"] == "assistant":
            cost_key = "output_cost_pm"
            # Try to find the model name in the assistant's message
            match = re.search(r'\((.*?)\)', content)
            if match:
                model_name = match.group(1)

        model_info = model_map.get(model_name)
        if model_info:
            cost_per_million = model_info.get(cost_key, 0)
            total_cost += (num_tokens / 1_000_000) * cost_per_million

    return total_tokens, total_cost


SUGGESTED_QUESTION = """A mid-sized country faces a resurgence of a novel respiratory virus. Vaccination rates have plateaued between 45-65%, ICU capacity varies between 75-95% across regions, and economic recovery remains fragile. Recent epidemiological studies suggest transmission rates may be 30-60% higher than initial models predicted. The government is considering reintroducing strict lockdowns for 4-8 weeks to suppress transmission before winter. Should it do so? Justify your position with evidence-backed analysis of epidemiological risk, economic stability, civil liberties, and public trust. Provide specific citations for key empirical claims and measurable predictions your approach would generate."""

MODEL_MAP = {
    # Hypothetical
    "GPT-5":           {"id": "openai/gpt-5",           "provider": "openai", "native_id": "gpt-5", "input_cost_pm": 10.0, "output_cost_pm": 30.0},
    "GPT-5 Mini":      {"id": "openai/gpt-5-mini",      "provider": "openai", "native_id": "gpt-5-mini", "input_cost_pm": 0.5,  "output_cost_pm": 1.5},
    # Real
    "Claude 4.5 Sonnet": {"id": "anthropic/claude-4.5-sonnet", "provider": "anthropic", "native_id": "claude-4.5-sonnet", "input_cost_pm": 5.0, "output_cost_pm": 25.0},
    "Claude 4 Sonnet":   {"id": "anthropic/claude-4-sonnet",   "provider": "anthropic", "native_id": "claude-4-sonnet", "input_cost_pm": 4.0, "output_cost_pm": 20.0},
    "Claude 3.5 Sonnet": {"id": "anthropic/claude-3.5-sonnet", "provider": "anthropic", "native_id": "claude-3.5-sonnet-20240620", "input_cost_pm": 3.0,  "output_cost_pm": 15.0},
    "GPT-4o":          {"id": "openai/gpt-4o",          "provider": "openai", "native_id": "gpt-4o", "input_cost_pm": 5.0,  "output_cost_pm": 15.0},
    "GPT-4o Mini":     {"id": "openai/gpt-4o-mini",     "provider": "openai", "native_id": "gpt-4o-mini", "input_cost_pm": 0.15, "output_cost_pm": 0.6},
    "Gemini 2.5 Pro":  {"id": "google/gemini-2.5-pro",  "provider": "google", "native_id": "gemini-2.5-pro", "input_cost_pm": 5.0, "output_cost_pm": 15.0},
    "Gemini 2.5 Flash":{"id": "google/gemini-2.5-flash","provider": "google", "native_id": "gemini-2.5-flash", "input_cost_pm": 0.35, "output_cost_pm": 0.7},
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

def stream_model(messages: List[dict], model_name: str, use_openrouter: bool) -> Generator[str, None, str]:
    """Stream from any model via OpenRouter or native APIs."""
    if not messages:
        return

    model_info = MODEL_MAP[model_name]

    if use_openrouter:
        client = get_openrouter_client()
        stream = client.chat.completions.create(
            model=model_info["id"], 
            messages=messages, 
            stream=True, 
            max_tokens=8192
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        provider = model_info["provider"]
        native_id = model_info["native_id"]

        if provider == "openai":
            client = get_openai_client()
            stream = client.chat.completions.create(model=native_id, messages=messages, stream=True, max_tokens=8192)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        elif provider == "anthropic":
            client = get_anthropic_client()
            # Anthropic requires system prompt to be handled differently
            system_prompt = ""
            if messages and messages[0]['role'] == 'system':
                system_prompt = messages[0]['content']
                messages = messages[1:]

            with client.messages.stream(model=native_id, system=system_prompt, messages=messages, max_tokens=8192) as stream:
                for text in stream.text_stream:
                    yield text

        elif provider == "google":
            client = get_gemini_client()
            model = client.GenerativeModel(native_id)
            # Gemini has specific content validation rules
            cleaned_messages = []
            for msg in messages:
                # Ensure user/model roles alternate correctly
                if cleaned_messages and cleaned_messages[-1]['role'] == msg['role']:
                    cleaned_messages[-1]['content'] += f"\n\n{msg['content']}"
                else:
                    cleaned_messages.append(msg)
            
            response = model.generate_content(cleaned_messages, stream=True)
            for chunk in response:
                yield chunk.text


def chat_turn(user_question: str, history: List[dict], primary_model: str, uploaded_files, use_openrouter: bool) -> Generator:
    """A single turn in the primary chat conversation."""
    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": format_bot_message("...", "Response", primary_model)})
    
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, gr.update(value=""), f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}"

    # Provide immediate feedback for non-streaming models
    model_info = MODEL_MAP[primary_model]
    if not model_info.get("supports_streaming", True):
        wait_message = f"Generating response with {primary_model} (non-streaming). This may take a moment..."
        history[-1]["content"] = format_bot_message(wait_message, "Response", primary_model)
        yield history, gr.update(value=""), f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}"

    messages = build_messages_with_context(user_question, history[:-2], uploaded_files)
    
    response_stream = stream_model(messages, primary_model, use_openrouter)
    
    full_response = ""
    for chunk in response_stream:
        full_response += chunk
        history[-1]["content"] = format_bot_message(full_response, "Response", primary_model)
        tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
        yield history, gr.update(value=""), f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}"

def handle_critique(history: List[dict], critique_model: str, uploaded_files, critique_prompt: str, use_openrouter: bool) -> Generator:
    """Generates a critique of the conversation."""
    if not history:
        history.append({"role": "assistant", "content": format_bot_message("Cannot perform critique on an empty conversation.", "Critique", "System")})
        tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
        yield history, f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}", ""
        return

    critique_messages = build_messages_with_context(critique_prompt, history, uploaded_files)
    
    history.append({"role": "user", "content": "Critique Request"})
    history.append({"role": "assistant", "content": format_bot_message("...", "Critique", critique_model)})
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}", ""

    # Provide immediate feedback for non-streaming models
    model_info = MODEL_MAP[critique_model]
    if not model_info.get("supports_streaming", True):
        wait_message = f"Generating critique with {critique_model} (non-streaming). This may take a moment..."
        history[-1]["content"] = format_bot_message(wait_message, "Critique", critique_model)
        yield history, f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}", ""

    critique_response = ""
    for chunk in stream_model(critique_messages, critique_model, use_openrouter):
        critique_response += chunk
        history[-1]["content"] = format_bot_message(critique_response, "Critique", critique_model)
        tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
        yield history, f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}", critique_response

def handle_review(history: List[dict], primary_model: str, uploaded_files, review_prompt_template: str, last_critique: str, use_openrouter: bool) -> Generator:
    """Generates a revised response based on the last critique."""
    if not last_critique:
        history.append({"role": "assistant", "content": format_bot_message("A critique must be generated before a revision can be made.", "Revision", "System")})
        tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
        yield history, f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}"
        return

    review_prompt = f"{review_prompt_template}\n\n--- CRITIQUE ---\n{last_critique}\n--- END CRITIQUE ---"
    review_messages = build_messages_with_context(review_prompt, history, uploaded_files)

    history.append({"role": "user", "content": "Review Request"})
    history.append({"role": "assistant", "content": format_bot_message("...", "Revision", primary_model)})
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}"

    # Provide immediate feedback for non-streaming models
    model_info = MODEL_MAP[primary_model]
    if not model_info.get("supports_streaming", True):
        wait_message = f"Generating revision with {primary_model} (non-streaming). This may take a moment..."
        history[-1]["content"] = format_bot_message(wait_message, "Revision", primary_model)
        yield history, f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}"

    revised_response = ""
    for chunk in stream_model(review_messages, primary_model, use_openrouter):
        revised_response += chunk
        history[-1]["content"] = format_bot_message(revised_response, "Revision", primary_model)
        tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
        yield history, f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}"


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
        primary_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="GPT-5", label="Primary Model", scale=3)
        critique_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Claude 4.5 Sonnet", label="Critique Model", scale=3)
        api_provider_switch = gr.Checkbox(label="Use OpenRouter", value=True, scale=1)
        cost_display = gr.Textbox(label="Est. Cost", value="Est. Cost: $0.0000", interactive=False, scale=1)
        token_count_display = gr.Textbox(label="Context Size", value="Context Size: 0", interactive=False, scale=1)
        summarize_btn = gr.Button("Summarise Context", scale=1)

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
        with gr.Column():
            critique_btn = gr.Button("Generate Critique", variant="secondary")
            critique_prompt_textbox = gr.Textbox(
                label="Critique Prompt",
                lines=5,
                value=(
                    "Please provide a concise, constructive critique of the assistant's reasoning, accuracy, and helpfulness throughout the preceding conversation. "
                    "Identify any potential biases, logical fallacies, or missed opportunities for a more comprehensive response. "
                    "Be extremely critical of citations and confirm or refute each specific citation, as existing or hallucinated, and then as relevant or not. "
                    "Provide a clear assessment of each and every citation and mark each with a Status icon (green existing/red hallucinated) + explanation "
                    "and a Relevance icon (red 'X', or an orange, or yellow or green) + explanation. "
                    "On this basis, and the overall assessment of the response, provide an overall critique rating out of 10 for the response."
                )
            )
        with gr.Column():
            review_btn = gr.Button("Generate Revision", variant="secondary")
            review_prompt_textbox = gr.Textbox(
                label="Review Prompt Template",
                lines=5,
                value="Based on the entire conversation history and the following critique, please provide a revised, improved version of your last response. " \
                "Synthesize the critique into your reasoning and address any shortcomings identified."
            )

    reset_btn = gr.Button("🔄 New Conversation")


    file_state = gr.State([])
    last_critique_state = gr.State("")

    def handle_send(user_question, history, p_model, files, use_openrouter):
        if not user_question.strip():
            tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
            return history, gr.update(value=""), f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}"
        yield from chat_turn(user_question, history, p_model, files, use_openrouter)

    def handle_upload(files):
        """Handles the file upload event and updates the UI."""
        file_names = [os.path.basename(f.name) for f in files]
        upload_status = f"📎 Uploaded: {', '.join(file_names)}. You can now ask questions about them."
        return files, upload_status

    def handle_summarize(history: List[dict]) -> Generator:
        """Summarizes the conversation history to reduce token count."""
        if len(history) < 5: # Don't summarize very short conversations
            yield history, f"Context Size: {count_tokens(history)}"
            return

        # Keep the first user message and the last 2 turns (4 messages)
        first_user_message = history[0]
        last_turns = history[-4:]
        to_summarize = history[1:-4]

        # Create the prompt for summarization
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in to_summarize])
        summary_prompt = f"Please provide a concise summary of the following conversation:\n\n{conversation_text}"
        
        summary_messages = [{"role": "user", "content": summary_prompt}]
        
        summary_response = ""
        # Use a fast model like GPT-4o Mini for summarization
        summarization_model = "GPT-4o Mini" 
        yield history, "Summarizing...", "Calculating..." # Update UI to show activity
        
        # Summarization will use OpenRouter by default for simplicity and cost-effectiveness
        for chunk in stream_model(summary_messages, summarization_model, use_openrouter=True):
            summary_response += chunk
        
        summary_message = {
            "role": "assistant", 
            "content": f"<br>### **Summary of Previous Conversation**\n---\n{summary_response}"
        }

        # Reconstruct the history
        new_history = [first_user_message, summary_message] + last_turns
        
        tokens, cost = calculate_cost_and_tokens(new_history, MODEL_MAP)
        yield new_history, f"Context Size: {tokens}", f"Est. Cost: ${cost:.4f}"

    send_btn.click(
        handle_send,
        [user_input, chatbot, primary_model, file_state, api_provider_switch],
        [chatbot, user_input, token_count_display, cost_display]
    )

    user_input.submit(
        handle_send,
        [user_input, chatbot, primary_model, file_state, api_provider_switch],
        [chatbot, user_input, token_count_display, cost_display]
    )

    critique_btn.click(
        handle_critique,
        [chatbot, critique_model, file_state, critique_prompt_textbox, api_provider_switch],
        [chatbot, token_count_display, cost_display, last_critique_state]
    )

    review_btn.click(
        handle_review,
        [chatbot, primary_model, file_state, review_prompt_textbox, last_critique_state, api_provider_switch],
        [chatbot, token_count_display, cost_display]
    )

    upload_btn.upload(
        handle_upload,
        [upload_btn],
        [file_state, user_input]
    )

    summarize_btn.click(
        handle_summarize,
        [chatbot],
        [chatbot, token_count_display, cost_display]
    )

    reset_btn.click(
        lambda: ([], [], gr.update(placeholder="Enter your message or use the suggested question...", value=""), "Context Size: 0", "Est. Cost: $0.0000"),
        [],
        [chatbot, file_state, user_input, token_count_display, cost_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False)
