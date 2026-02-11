import gradio as gr
import os
from typing import List, Tuple, Generator
import re
import tiktoken
from dotenv import load_dotenv
import json
import tempfile
import time
from pypdf import PdfReader
from mongodb_persistence import get_mongodb_persistence

load_dotenv() # Load variables from .env file

CONTENT_DIR = os.path.join(os.path.dirname(__file__), "content")
MODEL_MAP_PATH = os.path.join(CONTENT_DIR, "model_map.json")
PROMPTS_PATH = os.path.join(CONTENT_DIR, "prompts.json")

def _load_json_file(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: Missing JSON config at {path}. Using default.")
        return default
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {path}: {e}. Using default.")
        return default

PROMPTS = _load_json_file(PROMPTS_PATH, {})
if not PROMPTS:
    print(f"WARNING: Missing prompts at {PROMPTS_PATH}. Using defaults.")

SEND_SYSTEM_PROMPT = PROMPTS.get("send_system_prompt", "")
POLYCOMB_PROMPT = PROMPTS.get("polycomb_prompt", "")
DEMO_SCENARIOS = PROMPTS.get("demo_scenarios", {})
DEMO_CRITIQUE_PROMPT = PROMPTS.get("demo_critique_prompt", "")
DEMO_SYNTHESIS_PROMPT = PROMPTS.get("demo_synthesis_prompt", "")
SHORT_CRITIQUE_PROMPT = PROMPTS.get(
    "short_critique_prompt",
    "Please provide a concise, constructive critique focusing on: factual accuracy, logical reasoning, evidence quality, and practical feasibility...",
)
SHORT_REVIEW_PROMPT = PROMPTS.get(
    "short_review_prompt",
    "Based on the critique, provide a revised response that corrects any errors and addresses the identified concerns...",
)

def _badge_html(label: str, value: str) -> str:
    """Render the full HTML used by the token/cost badges so handlers can return it intact."""
    return (
        "<div style='text-align:center;'>"
        f"<div style='font-size:11px;color:#666;margin-bottom:4px;'>{label}</div>"
        f"<div class='badge'>{value}</div>"
        "</div>"
    )

def _status_html(use_openrouter: bool) -> str:
    """Render a compact status badge for API + MongoDB."""
    mongo_status = "OK"
    try:
        get_mongodb_persistence()
    except Exception:
        mongo_status = "ERR"

    if use_openrouter:
        api_status = "OpenRouter" if os.getenv("OPENROUTER_API_KEY") else "OpenRouter (missing key)"
    else:
        available = []
        if os.getenv("OPENAI_API_KEY"):
            available.append("OpenAI")
        if os.getenv("ANTHROPIC_API_KEY"):
            available.append("Anthropic")
        if os.getenv("GEMINI_API_KEY"):
            available.append("Gemini")
        api_status = "Native: " + ("+".join(available) if available else "missing keys")

    return (
        "<div style='text-align:center;'>"
        "<div style='font-size:11px;color:#666;margin-bottom:4px;'>Status</div>"
        f"<div class='badge'>API: {api_status} | Mongo: {mongo_status}</div>"
        "</div>"
    )

# --- MongoDB Persistence Functions ---
# These functions maintain the same interface as the file-based ones
# but use MongoDB for storage

def set_autosave_flag(value: bool):
    """Set autosave flag in MongoDB."""
    if not os.getenv("MONGODB_URI"):
        return
    try:
        db = get_mongodb_persistence()
        db.set_autosave_flag(value)
    except Exception as e:
        print(f"Error setting autosave flag: {e}")

def read_autosave_flag() -> bool:
    """Read autosave flag from MongoDB."""
    if not os.getenv("MONGODB_URI"):
        return False
    try:
        db = get_mongodb_persistence()
        return db._get_autosave_flag()
    except Exception as e:
        print(f"Error reading autosave flag: {e}")
        return False

def save_conversation(history: List[dict]):
    """Save conversation history to MongoDB."""
    if not os.getenv("MONGODB_URI"):
        return
    try:
        db = get_mongodb_persistence()
        db.save_conversation(history)
    except Exception as e:
        print(f"Error saving conversation: {e}")

def load_conversation() -> List[dict]:
    """Load conversation history from MongoDB."""
    if not os.getenv("MONGODB_URI"):
        return []
    try:
        db = get_mongodb_persistence()
        return db.load_conversation()
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return []

# --- Session Management Functions ---

def list_sessions() -> List[str]:
    """List all saved session names from MongoDB."""
    if not os.getenv("MONGODB_URI"):
        return []
    try:
        db = get_mongodb_persistence()
        return db.list_sessions()
    except Exception as e:
        print(f"Error listing sessions: {e}")
        return []

def save_session(name: str, history: List[dict]):
    """Save a named session to MongoDB."""
    if not os.getenv("MONGODB_URI"):
        return
    try:
        db = get_mongodb_persistence()
        db.save_session(name, history)
    except Exception as e:
        print(f"Error saving session: {e}")

def load_session(name: str) -> List[dict]:
    """Load a named session from MongoDB."""
    if not os.getenv("MONGODB_URI"):
        return []
    try:
        db = get_mongodb_persistence()
        return db.load_session(name)
    except Exception as e:
        print(f"Error loading session: {e}")
        return []

def delete_session(name: str):
    """Delete a named session from MongoDB."""
    if not os.getenv("MONGODB_URI"):
        return
    try:
        db = get_mongodb_persistence()
        db.delete_session(name)
    except Exception as e:
        print(f"Error deleting session: {e}")

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


def load_test_cases(filepath: str) -> dict:
    """Loads test cases from the new JSON format into a dictionary mapping titles to prompts."""
    cases = {"Custom Question": ""} # Default option
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # The new structure is a list under the "test_cases" key
        for case in data.get("test_cases", []):
            domain = case.get("domain")
            complexity = case.get("complexity")
            title = case.get("title")
            prompt = case.get("prompt")
            
            if all([domain, complexity, title, prompt]):
                # Format a more descriptive title for the dropdown
                display_title = f"[{domain.replace('_', ' ').title()}/{complexity.title()}] {title}"
                cases[display_title] = prompt

    except (FileNotFoundError, json.JSONDecodeError):
        # If file is missing or corrupt, add a fallback question
        cases["Default Scenario"] = "A mid-sized country faces a resurgence of a novel respiratory virus. Vaccination rates have plateaued between 45-65%, ICU capacity varies between 75-95% across regions, and economic recovery remains fragile. Recent epidemiological studies suggest transmission rates may be 30-60% higher than initial models predicted. The government is considering reintroducing strict lockdowns for 4-8 weeks to suppress transmission before winter. Should it do so? Justify your position with evidence-backed analysis of epidemiological risk, economic stability, civil liberties, and public trust. Provide specific citations for key empirical claims and measurable predictions your approach would generate."
    return cases

# Load test cases at startup, allowing dataset overrides via env var.
DEFAULT_TEST_CASES_PATH = "content/4domains_3complexity_16testcases_4Oct25.json"
TEST_CASES_PATH = os.getenv("TEST_CASES_PATH", DEFAULT_TEST_CASES_PATH)
TEST_CASES = load_test_cases(TEST_CASES_PATH)

DEFAULT_BENCHMARK_CASES_PATH = "content/test_data_humanities_reasoning_mmlu_25.json"
AUTO_HLE_BENCHMARK_PATH = "content/test_data_hle_100.json"
BENCHMARK_CASES_PATH = os.getenv(
    "BENCHMARK_CASES_PATH",
    AUTO_HLE_BENCHMARK_PATH if os.path.exists(AUTO_HLE_BENCHMARK_PATH) else DEFAULT_BENCHMARK_CASES_PATH,
)

def load_benchmark_cases(filepath: str) -> dict:
    """Loads benchmark cases keyed by dropdown title."""
    cases = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for case in data.get("test_cases", []):
            title = case.get("title")
            domain = case.get("domain")
            complexity = case.get("complexity")
            prompt = case.get("prompt")
            if all([title, domain, complexity, prompt]):
                display_title = f"[{domain.replace('_', ' ').title()}/{complexity.title()}] {title}"
                cases[display_title] = case
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return cases

BENCHMARK_CASES = load_benchmark_cases(BENCHMARK_CASES_PATH)

def _scan_benchmark_files(content_dir: str) -> dict:
    """
    Discover benchmark JSON files in content/ that include a `test_cases` list.
    Returns: {display_label: file_path}
    """
    options = {}
    try:
        for filename in sorted(os.listdir(content_dir)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(content_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cases = data.get("test_cases", [])
                if not isinstance(cases, list) or not cases:
                    continue
                benchmark_name = data.get("benchmark", "Benchmark")
                notes = data.get("notes", "")
                short_notes = (notes[:70] + "…") if len(notes) > 70 else notes
                label = f"{filename} | {benchmark_name} | n={len(cases)}"
                if short_notes:
                    label += f" | {short_notes}"
                options[label] = path
            except Exception:
                continue
    except Exception:
        return {}
    return options

BENCHMARK_FILE_OPTIONS = _scan_benchmark_files(CONTENT_DIR)

# Add demo scenarios to showcase ensemble capabilities - put them first for visibility
demo_cases = {}
for label, key in [
    ("🎯 DEMO: Historical Counterfactual", "Historical Counterfactual"),
    ("🎯 DEMO: Scientific Controversy", "Scientific Controversy"),
    ("🎯 DEMO: Urban Development", "Urban Development Dilemma"),
]:
    prompt = DEMO_SCENARIOS.get(key)
    if prompt:
        demo_cases[label] = prompt

# Create new ordered dict with demos first
ordered_cases = {}
ordered_cases.update(demo_cases)
ordered_cases.update(TEST_CASES)
TEST_CASES = ordered_cases

print(f"DEBUG: Total test cases loaded: {len(TEST_CASES)}")
print(f"DEBUG: Test cases file: {TEST_CASES_PATH}")
print(f"DEBUG: Benchmark cases loaded: {len(BENCHMARK_CASES)} from {BENCHMARK_CASES_PATH}")
print(f"DEBUG: Demo scenarios: {[k for k in TEST_CASES.keys() if '🎯' in k]}")
print(f"DEBUG: First 5 keys: {list(TEST_CASES.keys())[:5]}")


DEFAULT_MODEL_MAP = {
    # Hypothetical
    "GPT-5":           {"id": "openai/gpt-5",           "provider": "openai", "native_id": "gpt-5", "input_cost_pm": 10.0, "output_cost_pm": 30.0},
    "GPT-5.2":         {"id": "openai/gpt-5.2",         "provider": "openai", "native_id": "gpt-5.2", "input_cost_pm": 1.75, "output_cost_pm": 14.0},
    "GPT-5.2 Pro":     {"id": "openai/gpt-5.2-pro",     "provider": "openai", "native_id": "gpt-5.2-pro", "input_cost_pm": 1.75, "output_cost_pm": 14.0},
    "GPT-5 Mini":      {"id": "openai/gpt-5-mini",      "provider": "openai", "native_id": "gpt-5-mini", "input_cost_pm": 0.5,  "output_cost_pm": 1.5},
    # Real
    "Claude 4.5 Sonnet": {"id": "anthropic/claude-4.5-sonnet", "provider": "anthropic", "native_id": "claude-4.5-sonnet", "input_cost_pm": 5.0, "output_cost_pm": 25.0},
    "Claude 4 Sonnet":   {"id": "anthropic/claude-4-sonnet",   "provider": "anthropic", "native_id": "claude-4-sonnet", "input_cost_pm": 4.0, "output_cost_pm": 20.0},
    "Claude 3.5 Sonnet": {"id": "anthropic/claude-3.5-sonnet", "provider": "anthropic", "native_id": "claude-3.5-sonnet-20240620", "input_cost_pm": 3.0,  "output_cost_pm": 15.0},
    "GPT-4o":          {"id": "openai/gpt-4o",          "provider": "openai", "native_id": "gpt-4o", "input_cost_pm": 5.0,  "output_cost_pm": 15.0},
    "GPT-4o Mini":     {"id": "openai/gpt-4o-mini",     "provider": "openai", "native_id": "gpt-4o-mini", "input_cost_pm": 0.15, "output_cost_pm": 0.6},
    "Gemini 2.5 Pro":  {"id": "google/gemini-2.5-pro",  "provider": "google", "native_id": "gemini-2.5-pro", "input_cost_pm": 5.0, "output_cost_pm": 15.0},
    "Gemini 2.5 Flash":{"id": "google/gemini-2.5-flash","provider": "google", "native_id": "gemini-2.5-flash", "input_cost_pm": 0.35, "output_cost_pm": 0.7},
    "Gemini 3 Flash Preview":{"id": "google/gemini-3-flash-preview", "provider": "google", "native_id": "gemini-3-flash-preview", "input_cost_pm": 0.5, "output_cost_pm": 3.0},
}

MODEL_MAP = _load_json_file(MODEL_MAP_PATH, DEFAULT_MODEL_MAP)

def _read_pdf_text(path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            parts.append(page_text)
    return "\n".join(parts)

def process_uploaded_files(files) -> str:
    """Process uploaded files (text + PDFs) and return their content as a formatted string"""
    if not files:
        return ""
    
    file_contents = []
    
    for file in files:
        try:
            filename = os.path.basename(file.name)
            _, ext = os.path.splitext(filename.lower())
            if ext == ".pdf":
                content = _read_pdf_text(file.name)
            else:
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
            file_contents.append(f"=== FILE: {filename} ===\n{content}\n=== END FILE ===\n")
        except Exception as e:
            filename = os.path.basename(file.name) if hasattr(file, 'name') else "unknown"
            file_contents.append(f"=== FILE: {filename} ===\n[Error reading file: {str(e)}]\n=== END FILE ===\n")
    
    return "\n".join(file_contents)

def build_files_preview(files, max_total_chars: int = 2000, per_file_chars: int = 600) -> str:
    """Build a compact preview for uploaded files."""
    if not files:
        return "No files uploaded."

    previews = []
    total = 0
    for file in files:
        try:
            filename = os.path.basename(file.name)
            _, ext = os.path.splitext(filename.lower())
            if ext == ".pdf":
                content = _read_pdf_text(file.name)
            else:
                with open(file.name, "r", encoding="utf-8") as f:
                    content = f.read()
        except Exception as e:
            previews.append(f"**{filename}**\n\nError reading file: {e}\n")
            continue

        snippet = (content or "").strip().replace("\r\n", "\n")
        if len(snippet) > per_file_chars:
            snippet = snippet[:per_file_chars].rstrip() + "…"
        if not snippet:
            snippet = "[No extractable text]"

        block = f"**{filename}**\n\n{snippet}\n"
        if total + len(block) > max_total_chars:
            previews.append("**[Preview truncated]**")
            break
        previews.append(block)
        total += len(block)

    return "\n\n---\n\n".join(previews)

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
    if SEND_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SEND_SYSTEM_PROMPT})
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

    try:
        if use_openrouter:
            if not os.getenv("OPENROUTER_API_KEY"):
                yield "Missing OPENROUTER_API_KEY. Add it to `.env` or disable OpenRouter."
                return
            try:
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
                return
            except Exception as e:
                # If OpenRouter rejects the model, fall back to native provider
                msg = str(e)
                if "not a valid model" not in msg and "invalid model" not in msg:
                    raise
                provider = model_info["provider"]
                native_id = model_info["native_id"]
                if provider == "openai":
                    if not os.getenv("OPENAI_API_KEY"):
                        yield "Missing OPENAI_API_KEY. Add it to `.env` or disable OpenRouter."
                        return
                    client = get_openai_client()
                    stream = client.chat.completions.create(model=native_id, messages=messages, stream=True, max_tokens=8192)
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
                if provider == "anthropic":
                    if not os.getenv("ANTHROPIC_API_KEY"):
                        yield "Missing ANTHROPIC_API_KEY. Add it to `.env` or disable OpenRouter."
                        return
                    client = get_anthropic_client()
                    system_prompt = ""
                    if messages and messages[0]['role'] == 'system':
                        system_prompt = messages[0]['content']
                        messages = messages[1:]
                    with client.messages.stream(model=native_id, system=system_prompt, messages=messages, max_tokens=8192) as stream:
                        for text in stream.text_stream:
                            yield text
                    return
                if provider == "google":
                    if not os.getenv("GEMINI_API_KEY"):
                        yield "Missing GEMINI_API_KEY. Add it to `.env` or disable OpenRouter."
                        return
                    client = get_gemini_client()
                    model = client.GenerativeModel(native_id)
                    cleaned_messages = []
                    for msg in messages:
                        if cleaned_messages and cleaned_messages[-1]['role'] == msg['role']:
                            cleaned_messages[-1]['content'] += f"\n\n{msg['content']}"
                        else:
                            cleaned_messages.append(msg)
                    response = model.generate_content(cleaned_messages, stream=True)
                    for chunk in response:
                        yield chunk.text
                    return
                raise
        else:
            provider = model_info["provider"]
            native_id = model_info["native_id"]

            if provider == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    yield "Missing OPENAI_API_KEY. Add it to `.env` or enable OpenRouter."
                    return
                client = get_openai_client()
                stream = client.chat.completions.create(model=native_id, messages=messages, stream=True, max_tokens=8192)
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
            elif provider == "anthropic":
                if not os.getenv("ANTHROPIC_API_KEY"):
                    yield "Missing ANTHROPIC_API_KEY. Add it to `.env` or enable OpenRouter."
                    return
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
                if not os.getenv("GEMINI_API_KEY"):
                    yield "Missing GEMINI_API_KEY. Add it to `.env` or enable OpenRouter."
                    return
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
    except Exception as e:
        yield f"Request failed: {type(e).__name__}: {e}"


def chat_turn(user_question: str, history: List[dict], primary_model: str, uploaded_files, use_openrouter: bool) -> Generator:
    """A single turn in the primary chat conversation."""
    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": format_bot_message("...", "Response", primary_model)})
    
    # Calculate initial tokens for UI display
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, gr.update(value="", placeholder=""), _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")

    # Provide immediate feedback for non-streaming models
    model_info = MODEL_MAP[primary_model]
    if not model_info.get("supports_streaming", True):
        wait_message = f"Generating response with {primary_model} (non-streaming). This may take a moment..."
        history[-1]["content"] = format_bot_message(wait_message, "Response", primary_model)
        yield history, gr.update(value="", placeholder=""), _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")

    messages = build_messages_with_context(user_question, history[:-2], uploaded_files)
    
    response_stream = stream_model(messages, primary_model, use_openrouter)
    
    full_response = ""
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    
    for chunk in response_stream:
        full_response += chunk
        history[-1]["content"] = format_bot_message(full_response, "Response", primary_model)
        
        # No database saves during streaming - only UI updates
        yield history, gr.update(value="", placeholder=""), _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")
    
    # Save to database ONLY at the end and calculate final costs
    save_conversation(history)
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, gr.update(value="", placeholder=""), _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")

def handle_critique(history: List[dict], critique_model: str, uploaded_files, critique_prompt: str, use_openrouter: bool) -> Generator:
    """Generates a critique of the conversation."""
    if not history:
        history.append({"role": "assistant", "content": format_bot_message("Cannot perform critique on an empty conversation.", "Critique", "System")})
        save_conversation(history)
        tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
        yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}"), ""
        return

    # Use enhanced demo prompt if user hasn't customized it (still shows short version)
    actual_prompt = DEMO_CRITIQUE_PROMPT if critique_prompt == SHORT_CRITIQUE_PROMPT else critique_prompt
    critique_messages = build_messages_with_context(actual_prompt, history, uploaded_files)
    
    history.append({"role": "user", "content": "Critique Request"})
    history.append({"role": "assistant", "content": format_bot_message("...", "Critique", critique_model)})
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}"), ""

    # streaming loop with NO database saves during streaming
    critique_response = ""
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    
    for chunk in stream_model(critique_messages, critique_model, use_openrouter):
        critique_response += chunk
        history[-1]["content"] = format_bot_message(critique_response, "Critique", critique_model)
        
        # No database saves during streaming - only UI updates
        yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}"), critique_response
    
    # Save to database ONLY at the end and calculate final costs
    save_conversation(history)
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}"), critique_response

def handle_review(history: List[dict], primary_model: str, uploaded_files, review_prompt_template: str, last_critique: str, use_openrouter: bool) -> Generator:
    """Generates a revised response based on the last critique."""
    # If no explicit last_critique provided, try to extract the most recent critique from history.
    if not last_critique:
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Look for the Critique header rendered by format_bot_message or plain text "Critique"
                if "Critique" in content:
                    # Remove the header block if present (handles HTML and plain markdown forms)
                    cleaned = re.sub(r'(?s)<br>### \*\*Critique\*\*.*?---\n', '', content)
                    cleaned = re.sub(r'(?s)### \*\*Critique\*\*.*?---\n', '', cleaned)
                    cleaned = re.sub(r'(?s)Critique[:\n-]+', '', cleaned)
                    last_critique = cleaned.strip()
                    break

    if not last_critique:
        history.append({"role": "assistant", "content": format_bot_message("A critique must be generated before a revision can be made.", "Revision", "System")})
        save_conversation(history)
        tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
        yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")
        return

    # Use enhanced demo prompt if user hasn't customized it (still shows short version)
    actual_template = DEMO_SYNTHESIS_PROMPT if review_prompt_template == SHORT_REVIEW_PROMPT else review_prompt_template
    review_prompt = f"{actual_template}\n\n--- CRITIQUE ---\n{last_critique}\n--- END CRITIQUE ---"
    review_messages = build_messages_with_context(review_prompt, history, uploaded_files)

    history.append({"role": "user", "content": "Review Request"})
    history.append({"role": "assistant", "content": format_bot_message("...", "Revision", primary_model)})
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")

    revised_response = ""
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    
    for chunk in stream_model(review_messages, primary_model, use_openrouter):
        revised_response += chunk
        history[-1]["content"] = format_bot_message(revised_response, "Revision", primary_model)
        
        # No database saves during streaming - only UI updates
        yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")
    
    # Save to database ONLY at the end and calculate final costs
    save_conversation(history)
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")


BENCHMARK_INITIAL_TEMPLATE = (
    "You are answering a benchmark multiple-choice question.\n"
    "Think carefully, show concise reasoning, and end with exactly one line in this format:\n"
    "Final Answer: <LETTER>\n\n"
    "Question:\n{question}"
)

BENCHMARK_CRITIQUE_TEMPLATE = (
    "You are an adversarial critic. Evaluate the initial answer for factual errors, reasoning flaws, "
    "option elimination mistakes, and overconfidence.\n"
    "Recommend whether the answer should change. End with:\n"
    "Recommended Answer: <LETTER>\n\n"
    "Question:\n{question}\n\n"
    "Initial Answer:\n{initial_answer}"
)

BENCHMARK_REVIEW_TEMPLATE = (
    "Revise the answer using the critique. Keep reasoning concise and resolve errors explicitly.\n"
    "End with exactly one line:\n"
    "Final Answer: <LETTER>\n\n"
    "Question:\n{question}\n\n"
    "Initial Answer:\n{initial_answer}\n\n"
    "Critique:\n{critique}"
)

def _safe_format(template: str, **kwargs) -> str:
    """Best-effort format that tolerates user-edited templates."""
    if not template:
        return ""
    try:
        return template.format(**kwargs)
    except Exception:
        formatted = template
        for k, v in kwargs.items():
            formatted = formatted.replace("{" + k + "}", str(v))
        return formatted

def _extract_answer_letter(text: str) -> str:
    """Extract A-J answer letter from model output."""
    if not text:
        return ""
    patterns = [
        r"(?im)final\s+answer\s*[:\-]\s*\(?([A-J])\)?",
        r"(?im)recommended\s+answer\s*[:\-]\s*\(?([A-J])\)?",
        r"(?im)the\s+answer\s+is\s*\(?([A-J])\)?",
        r"\(([A-J])\)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)

    # Last fallback: final standalone A-J token
    matches = re.findall(r"\b([A-J])\b", text)
    return matches[-1] if matches else ""

def _extract_answer_text(text: str) -> str:
    """Extract free-text answer from common completion formats."""
    if not text:
        return ""
    patterns = [
        r"(?im)^\s*final\s+answer\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*recommended\s+answer\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*answer\s*[:\-]\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def _normalize_answer_text(text: str) -> str:
    s = (text or "").strip().lower()
    # Normalize punctuation and spacing for exact-match style benchmarks.
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`]", "", s)
    s = s.strip(" .,:;!?")
    return s

def _answers_equivalent(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a = a.strip()
    b = b.strip()
    if len(a) == 1 and len(b) == 1 and a.upper() in "ABCDEFGHIJ" and b.upper() in "ABCDEFGHIJ":
        return a.upper() == b.upper()
    na = _normalize_answer_text(a)
    nb = _normalize_answer_text(b)
    if na == nb:
        return True
    # Secondary leniency: remove all whitespace.
    return re.sub(r"\s+", "", na) == re.sub(r"\s+", "", nb)

def _key_from_metadata(metadata: dict) -> Tuple[str, str]:
    """Return (key_value, key_mode) where key_mode in {letter,text,none}."""
    key_letter = (metadata.get("correct_answer") or "").strip()
    if key_letter:
        return key_letter, "letter"
    key_text = (metadata.get("expected_answer_text") or "").strip()
    if key_text:
        return key_text, "text"
    return "", "none"

def _score_response_for_case(response_text: str, metadata: dict) -> Tuple[str, str, bool, str]:
    """
    Return (predicted, score_label, is_correct, key_display).
    Uses metadata scoring mode when available, and falls back appropriately.
    """
    metadata = metadata or {}
    key_value, key_mode = _key_from_metadata(metadata)
    scoring_mode = (metadata.get("scoring") or "").strip().lower()

    if key_mode == "none":
        # Still try to extract something so extraction-rate can be measured.
        pred_letter = _extract_answer_letter(response_text)
        pred_text = _extract_answer_text(response_text)
        predicted = pred_letter or pred_text
        return predicted, "Unscored (no answer key)", False, "N/A"

    if key_mode == "letter":
        predicted = _extract_answer_letter(response_text)
        if not predicted:
            return "", f"Incorrect (no extractable answer; expected {key_value})", False, key_value
        is_correct = _answers_equivalent(predicted, key_value)
        return predicted, ("Correct" if is_correct else f"Incorrect (expected {key_value})"), is_correct, key_value

    # key_mode == text
    # Prefer explicit answer-text extraction; for MCQ-like outputs, fall back to letter only if key is letter-like.
    predicted_text = _extract_answer_text(response_text)
    if not predicted_text:
        predicted_text = response_text.strip()
    if not predicted_text:
        return "", "Incorrect (no extractable answer)", False, key_value
    # For HLE exactMatch-like tasks, use normalized exact text comparison.
    is_correct = _answers_equivalent(predicted_text, key_value)
    if scoring_mode not in {"exact_match_text", "manual_or_custom", "exactmatch"}:
        # Unknown mode but text key exists; still evaluate via normalized exact match.
        is_correct = _answers_equivalent(predicted_text, key_value)
    return predicted_text, ("Correct" if is_correct else "Incorrect"), is_correct, key_value

def _short_cell(text: str, limit: int = 48) -> str:
    s = str(text or "")
    return s if len(s) <= limit else (s[: limit - 1] + "…")

def _score_mcq_response(response_text: str, correct_answer: str) -> Tuple[str, str, bool]:
    """Return (predicted, score_label, is_correct)."""
    predicted = _extract_answer_letter(response_text)
    if not correct_answer:
        return predicted, "No answer key available", False
    if not predicted:
        return "", f"Incorrect (no extractable answer; expected {correct_answer})", False
    is_correct = predicted == correct_answer
    return predicted, ("Correct" if is_correct else f"Incorrect (expected {correct_answer})"), is_correct

def _extract_change_decision(text: str) -> str:
    """Extract explicit KEEP/CHANGE decisions when present."""
    if not text:
        return ""
    m = re.search(r"(?im)change\s+decision\s*[:\-]\s*(KEEP|CHANGE)\b", text)
    return m.group(1).upper() if m else ""

def _has_evidence_signal(text: str) -> bool:
    """Heuristic check for non-trivial justification language."""
    if not text:
        return False
    return bool(re.search(r"(?i)\b(because|since|therefore|however|evidence|incorrect|error|flaw|misread|eliminate)\b", text))

def _apply_revision_guardrail(
    initial_pred: str,
    revised_pred: str,
    critique_text: str,
    review_text: str,
    mode: str = "conservative",
) -> Tuple[str, str]:
    """
    Decide whether to accept revised answer.
    Default is conservative: keep initial unless critique/review provide explicit support for change.
    """
    mode = (mode or "conservative").strip().lower()

    if not initial_pred and revised_pred:
        return revised_pred, "accept_revised_no_initial"
    if not revised_pred:
        return initial_pred, "keep_initial_no_revised"
    if revised_pred == initial_pred:
        return initial_pred, "no_change"

    if mode == "off":
        return revised_pred, "guardrail_off_accept_revised"

    critique_rec = _extract_answer_text(critique_text)
    if len(critique_rec) == 1 and critique_rec.upper() in "ABCDEFGHIJ":
        critique_rec = critique_rec.upper()
    review_decision = _extract_change_decision(review_text)
    critique_flags_initial_wrong = bool(
        re.search(r"(?is)\binitial\b.{0,80}\b(wrong|incorrect|error|flaw|misread)\b", critique_text)
        or re.search(r"(?is)\b(wrong|incorrect|error|flaw|misread)\b.{0,80}\binitial\b", critique_text)
    )
    evidence_present = _has_evidence_signal(critique_text) and _has_evidence_signal(review_text)

    support_for_revised = (_answers_equivalent(critique_rec, revised_pred) and critique_rec != "") or critique_flags_initial_wrong

    if mode == "balanced":
        # Balanced mode: allow updates with critique support and any evidence signal.
        should_change = support_for_revised and (_has_evidence_signal(critique_text) or _has_evidence_signal(review_text))
    else:
        # Conservative mode: require stronger support.
        should_change = support_for_revised and evidence_present

    if review_decision == "KEEP":
        should_change = False
    elif review_decision == "CHANGE" and support_for_revised:
        should_change = True

    if should_change:
        return revised_pred, f"accept_revised_supported_{mode}"
    return initial_pred, "guardrail_keep_initial"

def _default_benchmark_title() -> str:
    return next(iter(BENCHMARK_CASES.keys()), "")

def _benchmark_question_text(case_title: str) -> str:
    case = BENCHMARK_CASES.get(case_title) or {}
    return case.get("prompt", "")

def _default_benchmark_file_label() -> str:
    for label, path in BENCHMARK_FILE_OPTIONS.items():
        if path == BENCHMARK_CASES_PATH:
            return label
    return next(iter(BENCHMARK_FILE_OPTIONS.keys()), "No benchmark files found")

def _cost_delta(model_name: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    info = MODEL_MAP.get(model_name, {})
    input_cost_pm = info.get("input_cost_pm", 0.0)
    output_cost_pm = info.get("output_cost_pm", 0.0)
    return (input_tokens / 1_000_000) * input_cost_pm + (output_tokens / 1_000_000) * output_cost_pm

def _live_cost_text(total_input_tokens: int, total_output_tokens: int, total_cost: float) -> str:
    return (
        f"Input tokens: {total_input_tokens} | "
        f"Output tokens: {total_output_tokens} | "
        f"Estimated cost: ${total_cost:.6f}"
    )

def _log_benchmark(message: str):
    """Timestamped terminal logging for benchmark runs."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[BENCHMARK {ts}] {message}", flush=True)

def run_benchmark_case(
    case_title: str,
    initial_model: str,
    critique_model: str,
    review_model: str,
    initial_template: str,
    critique_template: str,
    review_template: str,
    guardrail_mode: str,
    use_openrouter: bool,
) -> Generator:
    """Run sequential initial -> critique -> review for one benchmark case."""
    _log_benchmark(f"Single-case run started | case='{case_title}' | guardrail='{guardrail_mode}'")
    encoder = get_encoder()
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    def cost_snapshot() -> str:
        return _live_cost_text(total_input_tokens, total_output_tokens, total_cost)

    case = BENCHMARK_CASES.get(case_title)
    if not case:
        msg = "Benchmark case not found."
        yield msg, "", "N/A", "", "", "N/A", "No run performed.", "Idle.", msg, cost_snapshot()
        return

    question = case.get("prompt", "")
    metadata = case.get("metadata", {})

    initial_prompt = _safe_format(initial_template, question=question)
    critique_prompt = ""
    review_prompt = ""

    initial_input_tokens = count_tokens(initial_prompt, encoder)
    total_input_tokens += initial_input_tokens
    total_cost += _cost_delta(initial_model, input_tokens=initial_input_tokens)
    yield question, "", "Running...", "", "", "Pending", "", "Starting initial answer stage...", "Initial answer stage started.", cost_snapshot()
    initial_response = ""
    for chunk in stream_model([{"role": "user", "content": initial_prompt}], initial_model, use_openrouter):
        initial_response += chunk
        chunk_tokens = count_tokens(chunk, encoder)
        total_output_tokens += chunk_tokens
        total_cost += _cost_delta(initial_model, output_tokens=chunk_tokens)
        yield question, initial_response, "Running...", "", "", "Pending", "", "Initial answer stage...", "Initial answer streaming...", cost_snapshot()

    initial_pred, initial_score, initial_correct, key_display = _score_response_for_case(initial_response, metadata)
    initial_score_label = f"{initial_score} | Predicted: {initial_pred or 'N/A'} | Key: {key_display}"
    yield question, initial_response, initial_score_label, "", "", "Pending", "", "Starting critique stage...", "Initial answer completed. Starting critique.", cost_snapshot()

    critique_prompt = _safe_format(
        critique_template,
        question=question,
        initial_answer=initial_response,
    )
    critique_input_tokens = count_tokens(critique_prompt, encoder)
    total_input_tokens += critique_input_tokens
    total_cost += _cost_delta(critique_model, input_tokens=critique_input_tokens)
    critique_response = ""
    for chunk in stream_model([{"role": "user", "content": critique_prompt}], critique_model, use_openrouter):
        critique_response += chunk
        chunk_tokens = count_tokens(chunk, encoder)
        total_output_tokens += chunk_tokens
        total_cost += _cost_delta(critique_model, output_tokens=chunk_tokens)
        yield question, initial_response, initial_score_label, critique_response, "", "Pending", "", "Critique stage...", "Critique streaming...", cost_snapshot()

    yield question, initial_response, initial_score_label, critique_response, "", "Running...", "", "Starting review stage...", "Critique completed. Starting review.", cost_snapshot()
    review_prompt = _safe_format(
        review_template,
        question=question,
        initial_answer=initial_response,
        critique=critique_response,
    )
    review_input_tokens = count_tokens(review_prompt, encoder)
    total_input_tokens += review_input_tokens
    total_cost += _cost_delta(review_model, input_tokens=review_input_tokens)
    revised_response = ""
    for chunk in stream_model([{"role": "user", "content": review_prompt}], review_model, use_openrouter):
        revised_response += chunk
        chunk_tokens = count_tokens(chunk, encoder)
        total_output_tokens += chunk_tokens
        total_cost += _cost_delta(review_model, output_tokens=chunk_tokens)
        yield question, initial_response, initial_score_label, critique_response, revised_response, "Running...", "", "Review stage...", "Review streaming...", cost_snapshot()

    revised_pred, revised_score, revised_correct, key_display = _score_response_for_case(revised_response, metadata)
    revised_score_label = f"{revised_score} | Predicted: {revised_pred or 'N/A'} | Key: {key_display}"

    final_pred, guardrail_reason = _apply_revision_guardrail(
        initial_pred,
        revised_pred,
        critique_response,
        revised_response,
        mode=guardrail_mode,
    )
    key_value, key_mode = _key_from_metadata(metadata)
    final_correct = _answers_equivalent(final_pred, key_value) if key_mode != "none" else False
    final_score = "Correct" if final_correct else ("Incorrect" if key_mode != "none" else "Unscored (no answer key)")
    final_score_label = (
        f"{final_score} | Final (guardrailed): {final_pred or 'N/A'} | "
        f"Raw revised: {revised_pred or 'N/A'} | Key: {key_display} | Guardrail: {guardrail_reason}"
    )
    if key_mode == "none":
        delta = "unscored"
    else:
        delta = "improved" if (not initial_correct and final_correct) else "no_change" if initial_correct == final_correct else "regressed"
    summary = (
        f"Case: {case_title}\n"
        f"Initial model: {initial_model} -> {initial_score_label}\n"
        f"Critique model: {critique_model}\n"
        f"Review model: {review_model} -> Raw revised: {revised_score_label}\n"
        f"Guardrail mode: {guardrail_mode}\n"
        f"Final (guardrailed): {final_score_label}\n"
        f"Ensemble delta: {delta}"
    )
    _log_benchmark(
        f"Single-case run completed | case='{case_title}' | initial={initial_pred or 'N/A'} | "
        f"revised={revised_pred or 'N/A'} | final={final_pred or 'N/A'} | guardrail_reason={guardrail_reason}"
    )
    yield question, initial_response, initial_score_label, critique_response, revised_response, final_score_label, summary, "Completed.", "Completed.", cost_snapshot()

def run_benchmark_batch(
    initial_model: str,
    critique_model: str,
    review_model: str,
    initial_template: str,
    critique_template: str,
    review_template: str,
    guardrail_mode: str,
    max_cases: int,
    use_openrouter: bool,
) -> Generator:
    """Run sequential ensemble across multiple benchmark cases and report aggregate metrics."""
    _log_benchmark(
        f"Batch run started | guardrail='{guardrail_mode}' | max_cases={max_cases} | "
        f"models=({initial_model}, {critique_model}, {review_model})"
    )
    encoder = get_encoder()
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    def cost_snapshot() -> str:
        return _live_cost_text(total_input_tokens, total_output_tokens, total_cost)

    try:
        all_titles = list(BENCHMARK_CASES.keys())
        if not all_titles:
            msg = "No benchmark cases loaded."
            yield msg, "No run performed.", "", msg, cost_snapshot()
            return

        selected_titles = all_titles[: max(1, int(max_cases))]
        total = len(selected_titles)
        results = []
        _log_benchmark(f"Batch selected {total} case(s)")

        yield f"Running batch benchmark on {total} case(s)...", "Starting batch...", "", f"Starting batch run with {total} case(s).", cost_snapshot()

        for idx, case_title in enumerate(selected_titles, start=1):
            case_started = time.time()
            _log_benchmark(f"[{idx}/{total}] Case start | {case_title}")
            try:
                case = BENCHMARK_CASES.get(case_title, {})
                question = case.get("prompt", "")
                case_metadata = case.get("metadata", {}) or {}

                initial_prompt = _safe_format(initial_template, question=question)
                stage_input_tokens = count_tokens(initial_prompt, encoder)
                total_input_tokens += stage_input_tokens
                total_cost += _cost_delta(initial_model, input_tokens=stage_input_tokens)
                _log_benchmark(f"[{idx}/{total}] Initial stage start | {case_title}")
                initial_response = ""
                for chunk in stream_model([{"role": "user", "content": initial_prompt}], initial_model, use_openrouter):
                    initial_response += chunk
                    chunk_tokens = count_tokens(chunk, encoder)
                    total_output_tokens += chunk_tokens
                    total_cost += _cost_delta(initial_model, output_tokens=chunk_tokens)
                _log_benchmark(f"[{idx}/{total}] Initial stage end | {case_title} | chars={len(initial_response)}")
                initial_pred, _, initial_ok, key_display = _score_response_for_case(initial_response, case_metadata)

                critique_prompt = _safe_format(
                    critique_template,
                    question=question,
                    initial_answer=initial_response,
                )
                stage_input_tokens = count_tokens(critique_prompt, encoder)
                total_input_tokens += stage_input_tokens
                total_cost += _cost_delta(critique_model, input_tokens=stage_input_tokens)
                _log_benchmark(f"[{idx}/{total}] Critique stage start | {case_title}")
                critique_response = ""
                for chunk in stream_model([{"role": "user", "content": critique_prompt}], critique_model, use_openrouter):
                    critique_response += chunk
                    chunk_tokens = count_tokens(chunk, encoder)
                    total_output_tokens += chunk_tokens
                    total_cost += _cost_delta(critique_model, output_tokens=chunk_tokens)
                _log_benchmark(f"[{idx}/{total}] Critique stage end | {case_title} | chars={len(critique_response)}")

                review_prompt = _safe_format(
                    review_template,
                    question=question,
                    initial_answer=initial_response,
                    critique=critique_response,
                )
                stage_input_tokens = count_tokens(review_prompt, encoder)
                total_input_tokens += stage_input_tokens
                total_cost += _cost_delta(review_model, input_tokens=stage_input_tokens)
                _log_benchmark(f"[{idx}/{total}] Review stage start | {case_title}")
                revised_response = ""
                for chunk in stream_model([{"role": "user", "content": review_prompt}], review_model, use_openrouter):
                    revised_response += chunk
                    chunk_tokens = count_tokens(chunk, encoder)
                    total_output_tokens += chunk_tokens
                    total_cost += _cost_delta(review_model, output_tokens=chunk_tokens)
                _log_benchmark(f"[{idx}/{total}] Review stage end | {case_title} | chars={len(revised_response)}")

                revised_pred, _, revised_ok, key_display = _score_response_for_case(revised_response, case_metadata)
                final_pred, guardrail_reason = _apply_revision_guardrail(
                    initial_pred,
                    revised_pred,
                    critique_response,
                    revised_response,
                    mode=guardrail_mode,
                )
                key_value, key_mode = _key_from_metadata(case_metadata)
                final_ok = _answers_equivalent(final_pred, key_value) if key_mode != "none" else False
            except Exception as case_err:
                _log_benchmark(f"[{idx}/{total}] Case error | {case_title} | {type(case_err).__name__}: {case_err}")
                results.append({
                    "case": case_title,
                    "domain": "unknown",
                    "key": "N/A",
                    "initial_pred": "ERR",
                    "initial_correct": False,
                    "revised_pred": "ERR",
                    "revised_correct": False,
                    "final_pred": "ERR",
                    "final_correct": False,
                    "guardrail": f"error:{type(case_err).__name__}",
                    "delta": "error",
                })
                yield (
                    f"Running batch benchmark on {total} case(s)...",
                    f"Progress: {idx}/{total} (latest: {case_title})",
                    "",
                    f"Error on {case_title}: {type(case_err).__name__}",
                    cost_snapshot(),
                )
                continue

            if key_mode == "none":
                delta = "unscored"
            else:
                if not initial_ok and final_ok:
                    delta = "improved"
                elif initial_ok and not final_ok:
                    delta = "regressed"
                else:
                    delta = "no_change"

            results.append({
                "case": case_title,
                "domain": case.get("domain", "unknown"),
                "key": key_display or "N/A",
                "initial_pred": initial_pred or "N/A",
                "initial_correct": initial_ok,
                "revised_pred": revised_pred or "N/A",
                "revised_correct": revised_ok,
                "final_pred": final_pred or "N/A",
                "final_correct": final_ok,
                "scoreable": key_mode != "none",
                "guardrail": guardrail_reason,
                "delta": delta,
            })
            _log_benchmark(
                f"[{idx}/{total}] Case done | {case_title} | initial={initial_pred or 'N/A'} "
                f"revised={revised_pred or 'N/A'} final={final_pred or 'N/A'} "
                f"delta={delta} elapsed={time.time()-case_started:.1f}s"
            )

            yield (
                f"Running batch benchmark on {total} case(s)...",
                f"Progress: {idx}/{total} (latest: {case_title})",
                "",
                f"Processed {idx}/{total}: {case_title}",
                cost_snapshot(),
            )

        scoreable_n = sum(1 for r in results if r.get("scoreable"))
        unscoreable_n = len(results) - scoreable_n
        initial_hits = sum(1 for r in results if r.get("scoreable") and r["initial_correct"])
        final_hits = sum(1 for r in results if r.get("scoreable") and r["final_correct"])
        initial_extract = sum(1 for r in results if (r.get("initial_pred") or "") not in {"", "N/A", "ERR"})
        revised_extract = sum(1 for r in results if (r.get("revised_pred") or "") not in {"", "N/A", "ERR"})
        improved = sum(1 for r in results if r["delta"] == "improved")
        regressed = sum(1 for r in results if r["delta"] == "regressed")
        unchanged = sum(1 for r in results if r["delta"] == "no_change")
        unscored_delta = sum(1 for r in results if r["delta"] == "unscored")

        by_domain = {}
        for r in results:
            d = r["domain"]
            by_domain.setdefault(d, {"n": 0, "scoreable": 0, "initial": 0, "revised": 0})
            by_domain[d]["n"] += 1
            by_domain[d]["scoreable"] += int(r.get("scoreable", False))
            by_domain[d]["initial"] += int(r["initial_correct"] and r.get("scoreable", False))
            by_domain[d]["revised"] += int(r["final_correct"] and r.get("scoreable", False))

        summary = (
            f"Cases: {total}\n"
            f"Guardrail mode: {guardrail_mode}\n"
            f"Scoreable cases: {scoreable_n}/{total} ({(100*scoreable_n/total):.1f}%)\n"
            f"Unscored cases (no key): {unscoreable_n}\n"
            f"Initial answer extraction: {initial_extract}/{total} ({(100*initial_extract/total):.1f}%)\n"
            f"Revised answer extraction: {revised_extract}/{total} ({(100*revised_extract/total):.1f}%)\n"
            f"Initial accuracy on scored subset: {initial_hits}/{scoreable_n if scoreable_n else 1} ({(100*initial_hits/(scoreable_n if scoreable_n else 1)):.1f}%)\n"
            f"Final guardrailed accuracy on scored subset: {final_hits}/{scoreable_n if scoreable_n else 1} ({(100*final_hits/(scoreable_n if scoreable_n else 1)):.1f}%)\n"
            f"Delta: improved={improved}, regressed={regressed}, no_change={unchanged}, unscored={unscored_delta}\n\n"
            "By domain:\n" +
            "\n".join(
                f"- {d}: scored {v['scoreable']}/{v['n']}, "
                f"initial {v['initial']}/{v['scoreable'] if v['scoreable'] else 1} "
                f"({(100*v['initial']/(v['scoreable'] if v['scoreable'] else 1)):.1f}%), "
                f"revised {v['revised']}/{v['scoreable'] if v['scoreable'] else 1} "
                f"({(100*v['revised']/(v['scoreable'] if v['scoreable'] else 1)):.1f}%)"
                for d, v in sorted(by_domain.items())
            )
        )

        lines = [
            "| Case | Domain | Key | Initial | Revised (Raw) | Final (Guardrailed) | Delta | Guardrail |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| {_short_cell(r['case'])} | {_short_cell(r['domain'])} | {_short_cell(r['key'])} | "
                f"{_short_cell(r['initial_pred'])} ({'✓' if r['initial_correct'] else '✗'}) | "
                f"{_short_cell(r['revised_pred'])} ({'✓' if r['revised_correct'] else '✗'}) | "
                f"{_short_cell(r['final_pred'])} ({'✓' if r['final_correct'] else '✗'}) | {r['delta']} | {_short_cell(r['guardrail'])} |"
            )
        detail_md = "\n".join(lines)

        yield "Batch run completed.", summary, detail_md, "Batch completed.", cost_snapshot()
        _log_benchmark(
            f"Batch run completed | cases={total} | initial_hits={initial_hits} | final_hits={final_hits} | "
            f"cost=${total_cost:.6f}"
        )
    except Exception as e:
        _log_benchmark(f"Batch run failed | {type(e).__name__}: {e}")
        yield f"Batch run failed: {type(e).__name__}", str(e), "", f"Batch run failed: {type(e).__name__}", cost_snapshot()


with gr.Blocks(
    title="Discursus",
    theme=gr.themes.Default(),
    css="""
    /* compact header */
    #header-row { display:flex; align-items:center; gap:12px; padding:6px 10px; }
    #header-row .gr-row { margin:0; }
    /* logo sizing now controlled by component height */
    #header-row img { width:auto; object-fit:contain; }

    /* compact numeric badges (replace textboxes visually) */
    .badge {
        display:inline-block;
        min-width:80px;
        padding:6px 10px;
        border-radius:14px;
        background: #ffffff;
        border: 1px solid #e6e6e6;
        box-shadow: none;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace;
        font-size:13px;
        color:#111;
        text-align:center;
    }

    /* subtle label styling placed above badges (uses native gradio labels if present) */
    #token-box .gr-label, #cost-box .gr-label, #status-box .gr-label {
        font-size:11px;
        margin-bottom:4px;
        color:#666;
    }

    /* ensure small footprint */
    #token-box, #cost-box, #status-box { padding:0; margin:0 6px; }
    """ 
) as demo:
    
    # MAIN LAYOUT: Side-by-side conversation and controls
    with gr.Row() as home_page_row:
        # LEFT COLUMN: Conversation (main focus) - now gets more space
        with gr.Column(scale=8, min_width=500):
            chatbot = gr.Chatbot(label="Conversation", height=700, type="messages")
        
        # RIGHT COLUMN: Header + All controls 
        with gr.Column(scale=3, min_width=350):
            # Header moved to right column: logo + token + cost (badges)
            with gr.Row(elem_id="header-row"):
                with gr.Column(scale=0, min_width=48):
                    logo_url = "https://github.com/alonie/discursus/raw/main/logo.png"
                    logo_image = gr.Image(logo_url, height=144, interactive=False, container=False)
                with gr.Column(scale=1, min_width=80, elem_id="token-box"):
                    token_count_display = gr.HTML(
                        "<div style='text-align:center;'>"
                        "<div style='font-size:11px;color:#666;margin-bottom:4px;'>Tokens</div>"
                        "<div class='badge' id='token-badge'>0</div>"
                        "</div>",
                        elem_id="token-box"
                    )
                with gr.Column(scale=1, min_width=90, elem_id="cost-box"):
                    cost_display = gr.HTML(
                        "<div style='text-align:center;'>"
                        "<div style='font-size:11px;color:#666;margin-bottom:4px;'>Cost</div>"
                        "<div class='badge' id='cost-badge'>$0.00</div>"
                        "</div>",
                        elem_id="cost-box"
                    )
                with gr.Column(scale=2, min_width=140, elem_id="status-box"):
                    status_display = gr.HTML(
                        _status_html(use_openrouter=True),
                        elem_id="status-box"
                    )
            
            gr.Markdown("---")  # Separator between header and controls
            # Example Questions
            example_questions_dd = gr.Dropdown(
                choices=list(TEST_CASES.keys()), 
                label="📚 Example Questions (🎯 = Demo Scenarios)",
                value=list(TEST_CASES.keys())[0]
            )
            open_benchmark_btn = gr.Button("🧪 Open Benchmarking Portal", variant="secondary")
            
            # Input Section
            gr.Markdown("### Send Message")
            question_size_slider = gr.Slider(
                minimum=1,
                maximum=12,
                value=7,
                step=1,
                label="Question Box Height (lines)",
            )
            user_input = gr.Textbox(
                label="Your Question",
                value="",
                placeholder=POLYCOMB_PROMPT or "Enter your question...",
                lines=7,
                max_lines=7,
                elem_id="question-box"
            )
            
            with gr.Row():
                send_model_dropdown = gr.Dropdown(
                    choices=list(MODEL_MAP.keys()), 
                    value="Gemini 3 Flash", 
                    label="Model",
                    scale=2
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            upload_btn = gr.UploadButton(
                "📎 Upload Files", 
                file_count="multiple", 
                file_types=["text", ".md", ".py", ".csv", ".json", ".pdf"],
                size="sm"
            )
            clear_files_btn = gr.Button("Clear Files", variant="secondary", size="sm")
            uploaded_files_dd = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                label="Uploaded Files",
                interactive=False,
            )
            with gr.Accordion("File Preview", open=False):
                file_preview_md = gr.Markdown("No files uploaded.")
            
            gr.Markdown("---")
            
            # Critique Section
            gr.Markdown("### Critique")
            with gr.Row():
                critique_model_dropdown = gr.Dropdown(
                    choices=list(MODEL_MAP.keys()), 
                    value="Claude 4.5 Sonnet", 
                    label="Model",
                    scale=2
                )
                critique_btn = gr.Button("Critique", variant="secondary", scale=1)
            
            critique_prompt_textbox = gr.Textbox(
                label="Critique Prompt",
                lines=3,
                value=SHORT_CRITIQUE_PROMPT,
                placeholder="Customize critique instructions..."
            )
            gr.Markdown("*Default uses detailed critique prompt. Edit to customize.*", elem_classes=["text-xs", "text-gray-500"])
            
            gr.Markdown("---")
            
            # Review Section  
            gr.Markdown("### Revise")
            with gr.Row():
                review_model_dropdown = gr.Dropdown(
                    choices=list(MODEL_MAP.keys()), 
                    value="GPT-5 Mini", 
                    label="Model",
                    scale=2
                )
                review_btn = gr.Button("Revise", variant="secondary", scale=1)
            
            review_prompt_textbox = gr.Textbox(
                label="Review Prompt",
                lines=3,
                value=SHORT_REVIEW_PROMPT,
                placeholder="Customize revision instructions..."
            )
            gr.Markdown("*Default uses detailed review prompt. Edit to customize.*", elem_classes=["text-xs", "text-gray-500"])
            
            gr.Markdown("---")
            
            # Quick Actions
            with gr.Row():
                reset_btn = gr.Button("🔄 New", variant="stop", scale=1)
                view_context_btn = gr.Button("📄 Context", scale=1)
                summarize_btn = gr.Button("📝 Summarize", scale=1)
            
            # Settings Accordion (collapsed but always accessible)
            with gr.Accordion("Settings", open=False):
                # Global Model Settings (for sync)
                primary_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="GPT-5 Mini", label="Default Primary Model")
                critique_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Claude 4.5 Sonnet", label="Default Critique Model")
                api_provider_switch = gr.Checkbox(label="Use OpenRouter", value=True)
                
                # Session Management
                gr.Markdown("**Sessions**")
                _session_choices = list_sessions()
                session_dropdown = gr.Dropdown(
                    choices=_session_choices,
                    label="Saved Sessions",
                    value=_session_choices[0] if _session_choices else None
                )
                session_name_input = gr.Textbox(label="Session Name", placeholder="Enter name to save...")
                
                with gr.Row():
                    save_session_btn = gr.Button("Save", size="sm", scale=1)
                    load_session_btn = gr.Button("Load", size="sm", scale=1)
                    delete_session_btn = gr.Button("Delete", size="sm", scale=1)
                    new_session_btn = gr.Button("New", size="sm", scale=1)
                
                autosave_checkbox = gr.Checkbox(label="Auto-save to selected session", value=read_autosave_flag())

    # Context Viewer (hidden by default, overlays when opened)
    with gr.Column(visible=False) as context_viewer_col:
        gr.Markdown("### Conversation Context")
        context_display = gr.Markdown()
        with gr.Row():
            download_file_btn = gr.File(label="Download Conversation", interactive=False)
            close_context_btn = gr.Button("Close Context Viewer")

    # Benchmarking Portal (hidden by default; page-like view)
    with gr.Column(visible=False) as benchmarking_portal_col:
        gr.Markdown("## Benchmarking Portal")
        gr.Markdown(
            "Sequential Epistemic Adversarial Ensemble: Initial answer -> adversarial critique -> revised answer. "
            "Scores are computed objectively from benchmark answer keys."
        )
        gr.Markdown("### Live Monitor")
        benchmark_live_log_md = gr.Markdown("Idle.")
        benchmark_live_cost_tb = gr.Textbox(
            label="Live Cost Estimate",
            value=_live_cost_text(0, 0, 0.0),
            interactive=False,
        )
        with gr.Row():
            back_home_btn = gr.Button("⬅ Back to Home", variant="secondary")
            run_benchmark_btn = gr.Button("Run Sequential Ensemble", variant="primary")
            run_batch_btn = gr.Button("Run Batch Benchmark", variant="primary")

        with gr.Row():
            benchmark_file_dd = gr.Dropdown(
                choices=list(BENCHMARK_FILE_OPTIONS.keys()) or ["No benchmark files found"],
                value=_default_benchmark_file_label(),
                label="Benchmark Dataset File",
            )
            refresh_benchmark_files_btn = gr.Button("Refresh Files", variant="secondary")

        benchmark_choices = list(BENCHMARK_CASES.keys()) or ["No benchmark cases available"]
        benchmark_case_dd = gr.Dropdown(
            choices=benchmark_choices,
            value=benchmark_choices[0],
            label="Benchmark Question",
        )
        benchmark_question_box = gr.Textbox(
            label="Question",
            lines=10,
            max_lines=20,
            interactive=False,
            value=_benchmark_question_text(benchmark_choices[0]) if BENCHMARK_CASES else "No benchmark cases loaded.",
        )

        with gr.Row():
            benchmark_initial_model_dd = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="GPT-5 Mini", label="Initial Model")
            benchmark_critique_model_dd = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Claude 4.5 Sonnet", label="Critique Model")
            benchmark_review_model_dd = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="GPT-5 Mini", label="Review Model")
            benchmark_guardrail_mode_dd = gr.Dropdown(
                choices=["conservative", "balanced", "off"],
                value="conservative",
                label="Guardrail Mode",
            )

        with gr.Accordion("Stage Prompt Templates (editable)", open=False):
            benchmark_initial_template_tb = gr.Textbox(
                label="Initial Prompt Template (use {question})",
                lines=6,
                value=BENCHMARK_INITIAL_TEMPLATE,
            )
            benchmark_critique_template_tb = gr.Textbox(
                label="Critique Prompt Template (use {question}, {initial_answer})",
                lines=6,
                value=BENCHMARK_CRITIQUE_TEMPLATE,
            )
            benchmark_review_template_tb = gr.Textbox(
                label="Review Prompt Template (use {question}, {initial_answer}, {critique})",
                lines=6,
                value=BENCHMARK_REVIEW_TEMPLATE,
            )
            batch_max_cases_slider = gr.Slider(
                minimum=1,
                maximum=max(1, len(BENCHMARK_CASES)),
                value=max(1, min(16, len(BENCHMARK_CASES))),
                step=1,
                label="Batch Max Cases",
            )

        benchmark_status_md = gr.Markdown("Idle.")
        with gr.Accordion("Single-Case Results (Initial Stage Score -> Run Summary)", open=False):
            with gr.Row():
                benchmark_initial_score_tb = gr.Textbox(label="Initial Stage Score", interactive=False)
                benchmark_revised_score_tb = gr.Textbox(label="Revised Stage Score", interactive=False)

            benchmark_initial_output_tb = gr.Textbox(label="Initial Answer Output", lines=10, max_lines=20)
            benchmark_critique_output_tb = gr.Textbox(label="Critique Output", lines=10, max_lines=20)
            benchmark_revised_output_tb = gr.Textbox(label="Revised Answer Output", lines=10, max_lines=20)
            benchmark_summary_tb = gr.Textbox(label="Run Summary", lines=4, max_lines=12)
        gr.Markdown("### Batch Results")
        benchmark_batch_status_md = gr.Markdown("Idle.")
        benchmark_batch_summary_tb = gr.Textbox(label="Batch Summary", lines=8, max_lines=20)
        benchmark_batch_table_md = gr.Markdown("")

    file_state = gr.State([])
    last_critique_state = gr.State("")

    def update_input_from_example(example_title):
        """Updates the user input textbox when an example is selected."""
        return gr.update(value=TEST_CASES.get(example_title, ""))

    def open_benchmarking_portal():
        default_title = _default_benchmark_title()
        question = _benchmark_question_text(default_title)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=default_title if default_title else "No benchmark cases available"),
            gr.update(value=question if question else "No benchmark cases loaded."),
        )

    def close_benchmarking_portal():
        return gr.update(visible=True), gr.update(visible=False)

    def on_benchmark_case_change(case_title: str):
        return gr.update(value=_benchmark_question_text(case_title))

    def refresh_benchmark_files():
        global BENCHMARK_FILE_OPTIONS
        BENCHMARK_FILE_OPTIONS = _scan_benchmark_files(CONTENT_DIR)
        labels = list(BENCHMARK_FILE_OPTIONS.keys()) or ["No benchmark files found"]
        preferred = _default_benchmark_file_label()
        if preferred not in labels:
            preferred = labels[0]
        return gr.update(choices=labels, value=preferred)

    def on_benchmark_file_change(file_label: str):
        global BENCHMARK_CASES, BENCHMARK_CASES_PATH
        path = BENCHMARK_FILE_OPTIONS.get(file_label)
        if not path:
            return (
                gr.update(choices=["No benchmark cases available"], value="No benchmark cases available"),
                gr.update(value="No benchmark cases loaded."),
                gr.update(maximum=1, value=1),
                "Benchmark file not found.",
            )

        BENCHMARK_CASES_PATH = path
        BENCHMARK_CASES = load_benchmark_cases(path)
        case_choices = list(BENCHMARK_CASES.keys()) or ["No benchmark cases available"]
        first_case = case_choices[0]
        question = _benchmark_question_text(first_case) if BENCHMARK_CASES else "No benchmark cases loaded."
        max_cases = max(1, len(BENCHMARK_CASES))
        slider_value = max_cases
        status = f"Loaded benchmark file: {os.path.basename(path)} ({len(BENCHMARK_CASES)} case(s))"
        return (
            gr.update(choices=case_choices, value=first_case),
            gr.update(value=question),
            gr.update(maximum=max_cases, value=slider_value),
            status,
        )

    def start_batch_run_ui():
        return "Queued batch run...", "Preparing run...", "", "Queued batch run...", _live_cost_text(0, 0, 0.0)

    # Sync model selections between advanced and per-action dropdowns
    def sync_primary_to_send(model):
        return gr.update(value=model)
    
    def sync_send_to_primary(model):
        return gr.update(value=model)
    
    def sync_critique_to_critique(model):
        return gr.update(value=model)
    
    def sync_critique_action_to_critique(model):
        return gr.update(value=model)

    def update_question_box_lines(lines):
        lines = int(lines)
        return gr.update(lines=lines, max_lines=lines)

    example_questions_dd.change(
        fn=update_input_from_example,
        inputs=[example_questions_dd],
        outputs=[user_input]
    )
    benchmark_case_dd.change(
        fn=on_benchmark_case_change,
        inputs=[benchmark_case_dd],
        outputs=[benchmark_question_box],
    )
    benchmark_file_dd.change(
        fn=on_benchmark_file_change,
        inputs=[benchmark_file_dd],
        outputs=[benchmark_case_dd, benchmark_question_box, batch_max_cases_slider, benchmark_live_log_md],
    )
    refresh_benchmark_files_btn.click(
        fn=refresh_benchmark_files,
        inputs=[],
        outputs=[benchmark_file_dd],
    )
    open_benchmark_btn.click(
        fn=open_benchmarking_portal,
        inputs=[],
        outputs=[home_page_row, benchmarking_portal_col, benchmark_case_dd, benchmark_question_box],
    )
    back_home_btn.click(
        fn=close_benchmarking_portal,
        inputs=[],
        outputs=[home_page_row, benchmarking_portal_col],
    )
    run_benchmark_btn.click(
        fn=run_benchmark_case,
        inputs=[
            benchmark_case_dd,
            benchmark_initial_model_dd,
            benchmark_critique_model_dd,
            benchmark_review_model_dd,
            benchmark_initial_template_tb,
            benchmark_critique_template_tb,
            benchmark_review_template_tb,
            benchmark_guardrail_mode_dd,
            api_provider_switch,
        ],
        outputs=[
            benchmark_question_box,
            benchmark_initial_output_tb,
            benchmark_initial_score_tb,
            benchmark_critique_output_tb,
            benchmark_revised_output_tb,
            benchmark_revised_score_tb,
            benchmark_summary_tb,
            benchmark_status_md,
            benchmark_live_log_md,
            benchmark_live_cost_tb,
        ],
    )
    run_batch_btn.click(
        fn=start_batch_run_ui,
        inputs=[],
        outputs=[
            benchmark_batch_status_md,
            benchmark_batch_summary_tb,
            benchmark_batch_table_md,
            benchmark_live_log_md,
            benchmark_live_cost_tb,
        ],
        queue=False,
    ).then(
        fn=run_benchmark_batch,
        inputs=[
            benchmark_initial_model_dd,
            benchmark_critique_model_dd,
            benchmark_review_model_dd,
            benchmark_initial_template_tb,
            benchmark_critique_template_tb,
            benchmark_review_template_tb,
            benchmark_guardrail_mode_dd,
            batch_max_cases_slider,
            api_provider_switch,
        ],
        outputs=[
            benchmark_batch_status_md,
            benchmark_batch_summary_tb,
            benchmark_batch_table_md,
            benchmark_live_log_md,
            benchmark_live_cost_tb,
        ],
    )

    # Sync model selections
    primary_model.change(sync_primary_to_send, [primary_model], [send_model_dropdown])
    send_model_dropdown.change(sync_send_to_primary, [send_model_dropdown], [primary_model])
    critique_model.change(sync_critique_to_critique, [critique_model], [critique_model_dropdown])
    critique_model_dropdown.change(sync_critique_action_to_critique, [critique_model_dropdown], [critique_model])
    api_provider_switch.change(lambda v: _status_html(v), [api_provider_switch], [status_display])
    question_size_slider.input(
        fn=None,
        inputs=[question_size_slider],
        outputs=[],
        js="""
        (lines) => {
            const el = document.querySelector("#question-box textarea");
            if (el) {
                const rows = Math.max(1, Math.round(lines));
                el.rows = rows;
                el.style.height = "auto";
            }
        }
        """,
    )
    question_size_slider.change(update_question_box_lines, [question_size_slider], [user_input])
    def handle_view_context(history: List[dict]):
        """Formats the conversation history for viewing and downloading."""
        if not history:
            return "The conversation is empty.", None, gr.update(visible=True)

        # Create a simple text representation for the file
        file_content = []
        for msg in history:
            role = msg.get("role", "unknown").title()
            content = msg.get("content", "")
            # Basic cleaning of Markdown for the text file
            content = re.sub(r'<br>### \*\*(.*?)\*\* \(.*?\)\n---\n', r'### \1 ###\n', content)
            file_content.append(f"--- {role} ---\n{content}\n")
        
        full_log = "\n".join(file_content)

        # Create a temporary file for downloading
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt", encoding='utf-8') as f:
            f.write(full_log)
            temp_filepath = f.name

        # Create a more readable Markdown version for display
        markdown_display = full_log.replace("\n", "<br>")

        return markdown_display, temp_filepath, gr.update(visible=True)

    def handle_send(user_question, history, p_model, files, use_openrouter):
        question = user_question.strip()
        if not question:
            question = (POLYCOMB_PROMPT or "").strip()
        if not question:
            tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
            return history, gr.update(value="", placeholder=""), _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")
        yield from chat_turn(question, history, p_model, files, use_openrouter)

    def handle_upload(files):
        """Handles the file upload event and updates the UI."""
        file_names = [os.path.basename(f.name) for f in files]
        # File metadata could be stored in MongoDB if needed
        default_upload_prompt = "Analyse the uploaded documents and summarise"
        preview = build_files_preview(files)
        return files, default_upload_prompt, gr.update(choices=file_names, value=file_names), preview

    def clear_files_handler():
        return [], gr.update(choices=[], value=[]), "No files uploaded."

    def handle_summarize(history: List[dict]) -> Generator:
        """Summarizes the conversation history to reduce token count."""
        if len(history) < 5: # Don't summarize very short conversations
            tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
            yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")
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
        
        # persist summarized history
        save_conversation(new_history)

        tokens, cost = calculate_cost_and_tokens(new_history, MODEL_MAP)
        yield new_history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")

    # --- Session handlers wired to the UI ---
    def save_session_handler(name: str, history: List[dict]):
        # allow automatic name if empty
        if not name:
            name = time.strftime("session-%A-%d-%B-%Y_%H-%M-%S", time.localtime())
        save_session(name, history)
        sessions = list_sessions()
        return gr.update(choices=sessions, value=name), gr.update(value=name)

    def load_session_handler(name: str):
        hist = load_session(name)
        # Remember this as the last session so autosave can target it
        try:
            if name:
                db = get_mongodb_persistence()
                db._set_last_session(name)
        except Exception:
            pass
        tokens, cost = calculate_cost_and_tokens(hist, MODEL_MAP)
        return hist, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}"), gr.update(value="", placeholder="")

    def delete_session_handler(name: str):
        delete_session(name)
        sessions = list_sessions()
        new_value = sessions[0] if sessions else ""
        return gr.update(choices=sessions, value=new_value), gr.update(value="")

    def new_session_handler():
        # clear session & prefill send box with default prompt
        return [], gr.update(value=""), gr.update(value=""), "0", "$0.0000", gr.update(choices=[], value=[]), "No files uploaded."

    save_session_btn.click(save_session_handler, [session_name_input, chatbot], [session_dropdown, session_name_input])
    load_session_btn.click(load_session_handler, [session_dropdown], [chatbot, token_count_display, cost_display, user_input])
    delete_session_btn.click(delete_session_handler, [session_dropdown], [session_dropdown, session_name_input])
    new_session_btn.click(new_session_handler, [], [chatbot, session_name_input, user_input, token_count_display, cost_display, uploaded_files_dd, file_preview_md])

    send_btn.click(
        handle_send,
        [user_input, chatbot, send_model_dropdown, file_state, api_provider_switch],
        [chatbot, user_input, token_count_display, cost_display]
    )

    user_input.submit(
        handle_send,
        [user_input, chatbot, send_model_dropdown, file_state, api_provider_switch],
        [chatbot, user_input, token_count_display, cost_display]
    )

    critique_btn.click(
        handle_critique,
        [chatbot, critique_model_dropdown, file_state, critique_prompt_textbox, api_provider_switch],
        [chatbot, token_count_display, cost_display, last_critique_state]
    )

    review_btn.click(
        handle_review,
        [chatbot, review_model_dropdown, file_state, review_prompt_textbox, last_critique_state, api_provider_switch],
        [chatbot, token_count_display, cost_display]
    )

    upload_btn.upload(
        handle_upload,
        [upload_btn],
        [file_state, user_input, uploaded_files_dd, file_preview_md]
    )
    clear_files_btn.click(
        clear_files_handler,
        [],
        [file_state, uploaded_files_dd, file_preview_md]
    )

    summarize_btn.click(
        handle_summarize,
        [chatbot],
        [chatbot, token_count_display, cost_display]
    )

    view_context_btn.click(
        handle_view_context,
        [chatbot],
        [context_display, download_file_btn, context_viewer_col]
    )

    close_context_btn.click(
        lambda: gr.update(visible=False),
        [],
        [context_viewer_col]
    )

    def on_load():
        """Initialize UI with empty conversation and available sessions."""
        sessions = list_sessions()
        history = []
        token_display = _badge_html("Token Count", "0")
        cost_display = _badge_html("Estimated Cost", "$0.0000")
        
        # Set the session dropdown to the last used session
        try:
            db = get_mongodb_persistence()
            last_session = db._get_last_session()
            selected_session = last_session if last_session in sessions else (sessions[0] if sessions else None)
        except:
            selected_session = sessions[0] if sessions else None
        
        return history, token_display, cost_display, gr.update(choices=sessions, value=selected_session), gr.update(value=""), read_autosave_flag()

    # ensure persisted conversation restores on page reload
    demo.load(on_load, [], [chatbot, token_count_display, cost_display, session_dropdown, session_name_input, autosave_checkbox])

    def reset_app():
        try:
            # Clear current conversation in MongoDB
            db = get_mongodb_persistence()
            db.save_conversation([])  # Save empty conversation
        except Exception:
            pass
        # Clear persisted data and prefill send box with default prompt
        return [], [], gr.update(value="", placeholder=POLYCOMB_PROMPT or "Enter your question..."), "0", "$0.0000", gr.update(choices=[], value=[]), "No files uploaded."

    reset_btn.click(
        reset_app,
        [],
        [chatbot, file_state, user_input, token_count_display, cost_display, uploaded_files_dd, file_preview_md]
    )

if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=7860):
        """Find a free port starting from the given port."""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return start_port  # fallback
    
    # Use environment PORT or find a free port starting from 7860
    preferred_port = int(os.getenv("PORT", 7860))
    if os.getenv("PORT"):
        # If PORT is explicitly set, use it (might fail if occupied)
        port = preferred_port
    else:
        # Auto-find free port starting from 7860
        port = find_free_port(preferred_port)
        if port != preferred_port:
            print(f"ℹ️  Port {preferred_port} is busy, using port {port} instead")
    
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
