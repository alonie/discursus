import gradio as gr
import os
from typing import List, Tuple, Generator
import re
import tiktoken
from dotenv import load_dotenv
import json
import tempfile
import time
from mongodb_persistence import get_mongodb_persistence

load_dotenv() # Load variables from .env file

def _badge_html(label: str, value: str) -> str:
    """Render the full HTML used by the token/cost badges so handlers can return it intact."""
    return (
        "<div style='text-align:center;'>"
        f"<div style='font-size:11px;color:#666;margin-bottom:4px;'>{label}</div>"
        f"<div class='badge'>{value}</div>"
        "</div>"
    )

# --- MongoDB Persistence Functions ---
# These functions maintain the same interface as the file-based ones
# but use MongoDB for storage

def set_autosave_flag(value: bool):
    """Set autosave flag in MongoDB."""
    try:
        db = get_mongodb_persistence()
        db.set_autosave_flag(value)
    except Exception as e:
        print(f"Error setting autosave flag: {e}")

def read_autosave_flag() -> bool:
    """Read autosave flag from MongoDB."""
    try:
        db = get_mongodb_persistence()
        return db._get_autosave_flag()
    except Exception as e:
        print(f"Error reading autosave flag: {e}")
        return False

def save_conversation(history: List[dict]):
    """Save conversation history to MongoDB."""
    try:
        db = get_mongodb_persistence()
        db.save_conversation(history)
    except Exception as e:
        print(f"Error saving conversation: {e}")

def load_conversation() -> List[dict]:
    """Load conversation history from MongoDB."""
    try:
        db = get_mongodb_persistence()
        return db.load_conversation()
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return []

# --- Session Management Functions ---

def list_sessions() -> List[str]:
    """List all saved session names from MongoDB."""
    try:
        db = get_mongodb_persistence()
        return db.list_sessions()
    except Exception as e:
        print(f"Error listing sessions: {e}")
        return []

def save_session(name: str, history: List[dict]):
    """Save a named session to MongoDB."""
    try:
        db = get_mongodb_persistence()
        db.save_session(name, history)
    except Exception as e:
        print(f"Error saving session: {e}")

def load_session(name: str) -> List[dict]:
    """Load a named session from MongoDB."""
    try:
        db = get_mongodb_persistence()
        return db.load_session(name)
    except Exception as e:
        print(f"Error loading session: {e}")
        return []

def delete_session(name: str):
    """Delete a named session from MongoDB."""
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

# Demo scenarios designed to showcase ensemble LLM capabilities  
DEMO_SCENARIOS = {
    "Historical Counterfactual": """It's 1947. You're advising the newly formed UN on the Palestine partition plan. You have access to: British Mandate reports, population data, religious significance maps, economic surveys, and testimonies from all parties. Knowing what we know about nation-building, conflict resolution, and regional stability - but constrained by 1947 knowledge and possibilities - what alternative approach might have prevented decades of conflict? Consider: demographic realities, religious significance, economic interdependence, and governance models available in 1947.""",
    
    "Scientific Controversy": """Recent studies on intermittent fasting show contradictory results. Study A (n=50,000, 5-year follow-up) shows 23% reduction in cardiovascular events and improved insulin sensitivity. Study B (n=30,000, 2-year follow-up) shows 15% increased eating disorder rates and social dysfunction. Study C (n=15,000, metabolic ward) questions long-term metabolic benefits. Meta-analysis D identifies publication bias favoring positive results. Should major health authorities recommend intermittent fasting for general population health? Consider: study quality, population differences, implementation challenges, and alternative approaches.""",
    
    "Urban Development Dilemma": """A mid-sized city faces a critical decision: approve a $2.8B urban redevelopment project that would create 18,000 jobs and modernize aging infrastructure, but requires demolishing the Riverside Historic District - home to 3,200 low-income residents, 40+ small businesses, and buildings dating to the 1880s. Environmental groups cite groundwater contamination risks. Economists argue the project is essential for regional competitiveness. The housing authority reports a 15-year wait list for affordable units. You have 8 months to decide before federal funding expires. What should the city do?"""
}

# Load test cases at startup from the new file
TEST_CASES = load_test_cases("content/4domains_3complexity_16testcases_4Oct25.json")

# Add demo scenarios to showcase ensemble capabilities - put them first for visibility
demo_cases = {
    "🎯 DEMO: Historical Counterfactual": DEMO_SCENARIOS["Historical Counterfactual"],
    "🎯 DEMO: Scientific Controversy": DEMO_SCENARIOS["Scientific Controversy"], 
    "🎯 DEMO: Urban Development": DEMO_SCENARIOS["Urban Development Dilemma"]
}

# Create new ordered dict with demos first
ordered_cases = {}
ordered_cases.update(demo_cases)
ordered_cases.update(TEST_CASES)
TEST_CASES = ordered_cases

print(f"DEBUG: Total test cases loaded: {len(TEST_CASES)}")
print(f"DEBUG: Demo scenarios: {[k for k in TEST_CASES.keys() if '🎯' in k]}")
print(f"DEBUG: First 5 keys: {list(TEST_CASES.keys())[:5]}")


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
    
    # Calculate initial tokens for UI display
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, gr.update(value=""), _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")

    # Provide immediate feedback for non-streaming models
    model_info = MODEL_MAP[primary_model]
    if not model_info.get("supports_streaming", True):
        wait_message = f"Generating response with {primary_model} (non-streaming). This may take a moment..."
        history[-1]["content"] = format_bot_message(wait_message, "Response", primary_model)
        yield history, gr.update(value=""), _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")

    messages = build_messages_with_context(user_question, history[:-2], uploaded_files)
    
    response_stream = stream_model(messages, primary_model, use_openrouter)
    
    full_response = ""
    tokens, cost = 0, 0.0  # Initialize to avoid recalculating every chunk
    
    for chunk in response_stream:
        full_response += chunk
        history[-1]["content"] = format_bot_message(full_response, "Response", primary_model)
        
        # No database saves during streaming - only UI updates
        yield history, gr.update(value=""), _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")
    
    # Save to database ONLY at the end and calculate final costs
    save_conversation(history)
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, gr.update(value=""), _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")

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
    tokens, cost = 0, 0.0  # Initialize to avoid recalculating every chunk
    
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
    tokens, cost = 0, 0.0  # Initialize to avoid recalculating every chunk
    
    for chunk in stream_model(review_messages, primary_model, use_openrouter):
        revised_response += chunk
        history[-1]["content"] = format_bot_message(revised_response, "Revision", primary_model)
        
        # No database saves during streaming - only UI updates
        yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")
    
    # Save to database ONLY at the end and calculate final costs
    save_conversation(history)
    tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
    yield history, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")


# Human-readable default prompt (edit as you prefer) - Using demo scenario for impact
POLYCOMB_PROMPT = """A mid-sized city faces a critical decision: approve a $2.8B urban redevelopment project that would create 18,000 jobs and modernize aging infrastructure, but requires demolishing the Riverside Historic District - home to 3,200 low-income residents, 40+ small businesses, and buildings dating to the 1880s. Environmental groups cite groundwater contamination risks. Economists argue the project is essential for regional competitiveness. The housing authority reports a 15-year wait list for affordable units. You have 8 months to decide before federal funding expires. What should the city do?"""

# Demo scenarios designed to showcase ensemble LLM capabilities  
DEMO_SCENARIOS = {
    "Historical Counterfactual": """It's 1947. You're advising the newly formed UN on the Palestine partition plan. You have access to: British Mandate reports, population data, religious significance maps, economic surveys, and testimonies from all parties. Knowing what we know about nation-building, conflict resolution, and regional stability - but constrained by 1947 knowledge and possibilities - what alternative approach might have prevented decades of conflict? Consider: demographic realities, religious significance, economic interdependence, and governance models available in 1947.""",
    
    "Scientific Controversy": """Recent studies on intermittent fasting show contradictory results. Study A (n=50,000, 5-year follow-up) shows 23% reduction in cardiovascular events and improved insulin sensitivity. Study B (n=30,000, 2-year follow-up) shows 15% increased eating disorder rates and social dysfunction. Study C (n=15,000, metabolic ward) questions long-term metabolic benefits. Meta-analysis D identifies publication bias favoring positive results. Should major health authorities recommend intermittent fasting for general population health? Consider: study quality, population differences, implementation challenges, and alternative approaches.""",
    
    "Urban Development Dilemma": """A mid-sized city faces a critical decision: approve a $2.8B urban redevelopment project that would create 18,000 jobs and modernize aging infrastructure, but requires demolishing the Riverside Historic District - home to 3,200 low-income residents, 40+ small businesses, and buildings dating to the 1880s. Environmental groups cite groundwater contamination risks. Economists argue the project is essential for regional competitiveness. The housing authority reports a 15-year wait list for affordable units. You have 8 months to decide before federal funding expires. What should the city do?"""
}

# Enhanced deliberative reasoning prompts for sophisticated ensemble LLM capabilities
DEMO_CRITIQUE_PROMPT = """You are an expert adversarial reasoner tasked with rigorously examining the preceding response. Your goal is not just evaluation, but intellectual transformation through deep scrutiny.

**PART I: SURFACE-LEVEL ANALYSIS**
Rate each dimension (1-10) with specific justification:
• **Factual Accuracy**: Are claims supported by evidence? Are facts, statistics, dates, and causal relationships correct? Are there demonstrable errors or misrepresentations?
• **Logical Coherence**: Internal consistency, valid inferences, premise-conclusion alignment
• **Evidence Quality**: Information reliability, source credibility, data interpretation, acknowledgment of limitations and conflicting evidence
• **Stakeholder Mapping**: Completeness of affected parties, power dynamics, voice representation
• **Implementation Realism**: Resource constraints, timeline feasibility, political/social barriers
• **Unintended Consequences**: Second/third-order effects, feedback loops, systemic interactions

**PART II: DEEP COGNITIVE EXAMINATION**
• **Cognitive Biases**: What systematic thinking errors are present? (confirmation bias, availability heuristic, anchoring, etc.)
• **Epistemic Blind Spots**: What crucial unknowns are being treated as knowns? Where is false certainty masquerading as analysis?
• **Frame Examination**: What assumptions about the problem definition itself may be flawed? Are we solving the right problem?
• **Alternative Paradigms**: What fundamentally different ways of understanding this situation are being ignored?

**PART III: ADVERSARIAL CHALLENGE**
• **Steel-manning Opposition**: What is the strongest possible case against this recommendation?
• **Assumption Inversion**: What if the core assumptions were reversed - how would that change everything?
• **Temporal Perspectives**: How might this look catastrophically wrong in 5/20/50 years?
• **Stakeholder Role-Play**: Argue from the perspective of the three most disadvantaged groups

**PART IV: META-REASONING**
• **Process Critique**: What flaws exist in HOW this reasoning was conducted, not just the conclusions?
• **Information Architecture**: What critical information is missing that would fundamentally alter the analysis?
• **Uncertainty Quantification**: Where should we be much more uncertain than the response suggests?

Conclude with: (1) The single most dangerous flaw in this reasoning, (2) The most important question that remains unasked, (3) A completely different framing that might yield superior insights."""

DEMO_SYNTHESIS_PROMPT = """You are a master synthesizer tasked with transcending the original analysis through deep integration of critique insights. This is not mere revision—it's intellectual evolution.

**SYNTHESIS PRINCIPLES:**
• **Integration over Addition**: Don't just append critique points; weave them into a fundamentally transformed understanding
• **Tension Resolution**: Where the critique reveals legitimate competing values/perspectives, develop sophisticated approaches that honor multiple valid concerns
• **Emergent Insights**: Look for new understanding that emerges from the interaction between original reasoning and critique
• **Uncertainty Embrace**: Be comfortable with irreducible uncertainties; make them productive rather than paralyzing

**REQUIRED SYNTHESIS ELEMENTS:**

1. **Factual Correction & Verification**: Address any factual errors, misrepresentations, or unsupported claims identified in the critique. Provide corrected information with appropriate caveats.

2. **Reframed Problem Statement**: How does the critique suggest we should understand the core challenge differently?

3. **Enhanced Evidence Architecture**: Build a more robust foundation by incorporating critique insights, addressing information gaps, and acknowledging conflicting evidence

4. **Multi-Stakeholder Harmony**: Develop approaches that address the critique's stakeholder concerns without abandoning feasibility

5. **Adaptive Implementation Strategy**: Create implementation approaches that anticipate and adapt to the consequences identified in the critique

6. **Epistemic Humility Framework**: Explicitly acknowledge key uncertainties, areas of ignorance, and build learning/adjustment mechanisms into recommendations

7. **Alternative Scenario Planning**: Address the critique's concerns about different future contexts or assumption failures

**OUTPUT STRUCTURE:**
- **Executive Synthesis**: 2-3 sentences capturing how your thinking has fundamentally evolved
- **Factual Clarifications**: Any corrections to errors or misrepresentations in the original response
- **Transformed Analysis**: Your new understanding of the situation incorporating critique insights
- **Refined Recommendations**: Specific actions that address both original goals and critique concerns
- **Uncertainty Map**: Key unknowns and how you'll gather information/adapt as you learn
- **Success/Failure Indicators**: How you'll know if your approach is working or needs revision

Remember: The goal is not to defend the original analysis or simply fix its flaws, but to achieve a higher level of understanding that incorporates the wisdom from both perspectives. Accuracy is paramount - admit when you don't know something rather than making unsupported claims."""

# Shortened versions for UI display
SHORT_CRITIQUE_PROMPT = "Please provide a concise, constructive critique focusing on: factual accuracy, logical reasoning, evidence quality, and practical feasibility..."
SHORT_REVIEW_PROMPT = "Based on the critique, provide a revised response that corrects any errors and addresses the identified concerns..."

# Demo scenarios designed to showcase ensemble LLM capabilities
DEMO_SCENARIOS = {
    "Historical Counterfactual": """It's 1947. You're advising the newly formed UN on the Palestine partition plan. You have access to: British Mandate reports, population data, religious significance maps, economic surveys, and testimonies from all parties. Knowing what we know about nation-building, conflict resolution, and regional stability - but constrained by 1947 knowledge and possibilities - what alternative approach might have prevented decades of conflict? Consider: demographic realities, religious significance, economic interdependence, and governance models available in 1947.""",
    
    "Scientific Controversy": """Recent studies on intermittent fasting show contradictory results. Study A (n=50,000, 5-year follow-up) shows 23% reduction in cardiovascular events and improved insulin sensitivity. Study B (n=30,000, 2-year follow-up) shows 15% increased eating disorder rates and social dysfunction. Study C (n=15,000, metabolic ward) questions long-term metabolic benefits. Meta-analysis D identifies publication bias favoring positive results. Should major health authorities recommend intermittent fasting for general population health? Consider: study quality, population differences, implementation challenges, and alternative approaches.""",
    
    "Urban Development Dilemma": """A mid-sized city faces a critical decision: approve a $2.8B urban redevelopment project that would create 18,000 jobs and modernize aging infrastructure, but requires demolishing the Riverside Historic District - home to 3,200 low-income residents, 40+ small businesses, and buildings dating to the 1880s. Environmental groups cite groundwater contamination risks. Economists argue the project is essential for regional competitiveness. The housing authority reports a 15-year wait list for affordable units. You have 8 months to decide before federal funding expires. What should the city do?"""
}

with gr.Blocks(
    title="Discursus",
    theme=gr.themes.Default(),
    css="""
    /* compact header */
    #header-row { display:flex; align-items:center; gap:12px; padding:6px 10px; }
    #header-row .gr-row { margin:0; }
    /* make logo compact */
    #header-row img { height:48px; width:auto; object-fit:contain; }

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
    #token-box .gr-label, #cost-box .gr-label {
        font-size:11px;
        margin-bottom:4px;
        color:#666;
    }

    /* ensure small footprint */
    #token-box, #cost-box { padding:0; margin:0 6px; }
    """ 
) as demo:
    
    # MAIN LAYOUT: Side-by-side conversation and controls
    with gr.Row():
        # LEFT COLUMN: Conversation (main focus) - now gets more space
        with gr.Column(scale=8, min_width=500):
            chatbot = gr.Chatbot(label="Conversation", height=700, type="messages")
        
        # RIGHT COLUMN: Header + All controls 
        with gr.Column(scale=3, min_width=350):
            # Header moved to right column: logo + token + cost (badges)
            with gr.Row(elem_id="header-row"):
                with gr.Column(scale=0, min_width=48):
                    logo_url = "https://github.com/alonie/discursus/raw/main/logo.png"
                    gr.Image(logo_url, height=48, interactive=False, container=False)
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
            
            gr.Markdown("---")  # Separator between header and controls
            # Example Questions
            example_questions_dd = gr.Dropdown(
                choices=list(TEST_CASES.keys()), 
                label="📚 Example Questions (🎯 = Demo Scenarios)",
                value=list(TEST_CASES.keys())[0]
            )
            
            # Input Section
            gr.Markdown("### Send Message")
            user_input = gr.Textbox(
                label="Your Question", 
                value=POLYCOMB_PROMPT, 
                placeholder="Enter your question...", 
                lines=3
            )
            
            with gr.Row():
                send_model_dropdown = gr.Dropdown(
                    choices=list(MODEL_MAP.keys()), 
                    value="Gemini 2.5 Flash", 
                    label="Model",
                    scale=2
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            upload_btn = gr.UploadButton(
                "📎 Upload Files", 
                file_count="multiple", 
                file_types=["text", ".md", ".py", ".csv", ".json"],
                size="sm"
            )
            
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
                    value="Gemini 2.5 Flash", 
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
                primary_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Gemini 2.5 Flash", label="Default Primary Model")
                critique_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value="Claude 4.5 Sonnet", label="Default Critique Model")
                api_provider_switch = gr.Checkbox(label="Use OpenRouter", value=True)
                
                # Session Management
                gr.Markdown("**Sessions**")
                session_dropdown = gr.Dropdown(choices=list_sessions(), label="Saved Sessions", value=list_sessions()[0] if list_sessions() else "")
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

    file_state = gr.State([])
    last_critique_state = gr.State("")

    def update_input_from_example(example_title):
        """Updates the user input textbox when an example is selected."""
        return gr.update(value=TEST_CASES.get(example_title, ""))

    # Sync model selections between advanced and per-action dropdowns
    def sync_primary_to_send(model):
        return gr.update(value=model)
    
    def sync_send_to_primary(model):
        return gr.update(value=model)
    
    def sync_critique_to_critique(model):
        return gr.update(value=model)
    
    def sync_critique_action_to_critique(model):
        return gr.update(value=model)

    example_questions_dd.change(
        fn=update_input_from_example,
        inputs=[example_questions_dd],
        outputs=[user_input]
    )

    # Sync model selections
    primary_model.change(sync_primary_to_send, [primary_model], [send_model_dropdown])
    send_model_dropdown.change(sync_send_to_primary, [send_model_dropdown], [primary_model])
    critique_model.change(sync_critique_to_critique, [critique_model], [critique_model_dropdown])
    critique_model_dropdown.change(sync_critique_action_to_critique, [critique_model_dropdown], [critique_model])

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
        if not user_question.strip():
            tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
            return history, gr.update(value=""), _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")
        yield from chat_turn(user_question, history, p_model, files, use_openrouter)

    def handle_upload(files):
        """Handles the file upload event and updates the UI."""
        file_names = [os.path.basename(f.name) for f in files]
        upload_status = f"📎 Uploaded: {', '.join(file_names)}. You can now ask questions about them."
        # File metadata could be stored in MongoDB if needed
        return files, upload_status

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
        return hist, _badge_html("Token Count", str(tokens)), _badge_html("Estimated Cost", f"${cost:.4f}")

    def delete_session_handler(name: str):
        delete_session(name)
        sessions = list_sessions()
        new_value = sessions[0] if sessions else ""
        return gr.update(choices=sessions, value=new_value), gr.update(value="")

    def new_session_handler():
        # clear session & prefill send box with default prompt
        return [], gr.update(value=""), gr.update(value=POLYCOMB_PROMPT), "0", "$0.0000"

    save_session_btn.click(save_session_handler, [session_name_input, chatbot], [session_dropdown, session_name_input])
    load_session_btn.click(load_session_handler, [session_dropdown], [chatbot, token_count_display, cost_display])
    delete_session_btn.click(delete_session_handler, [session_dropdown], [session_dropdown, session_name_input])
    new_session_btn.click(new_session_handler, [], [chatbot, session_name_input, user_input, token_count_display, cost_display])

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
        [file_state, user_input]
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

    reset_btn.click(
        lambda: ([], [], gr.update(placeholder="Enter your message or use the suggested question...", value=""), "0", "$0.0000"),
        [],
        [chatbot, file_state, user_input, token_count_display, cost_display]
    )

    def on_load():
        """Initialize UI with persisted conversation and available sessions."""
        sessions = list_sessions()
        
        # Load the current conversation from MongoDB (like the original file-based version)
        history = load_conversation()
        
        # Calculate costs for the loaded conversation
        if history:
            tokens, cost = calculate_cost_and_tokens(history, MODEL_MAP)
            token_display = _badge_html("Token Count", str(tokens))
            cost_display = _badge_html("Estimated Cost", f"${cost:.4f}")
        else:
            token_display = _badge_html("Token Count", "0")
            cost_display = _badge_html("Estimated Cost", "$0.0000")
        
        # Set the session dropdown to the last used session
        try:
            db = get_mongodb_persistence()
            last_session = db._get_last_session()
            selected_session = last_session if last_session in sessions else (sessions[0] if sessions else "")
        except:
            selected_session = sessions[0] if sessions else ""
        
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
        return [], [], gr.update(placeholder="Enter your message or use the suggested question...", value=POLYCOMB_PROMPT), "0", "$0.0000"

    reset_btn.click(
        reset_app,
        [],
        [chatbot, file_state, user_input, token_count_display, cost_display]
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
