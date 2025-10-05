# Discursus

**An advanced chatbot interface for critical analysis and iterative refinement.**

Discursus is a Gradio-based web application designed for structured, in-depth conversations with a variety of large language models (LLMs). It goes beyond a simple chat interface by incorporating a unique **Critique & Review** workflow, allowing users to critically evaluate a model's response and prompt it to generate a revised, improved version.

![Discursus Logo](Discursus_logo_med_23Sep25.png)

---

## Key Features

*   **Multi-Model Interaction**: Seamlessly switch between top-tier models from OpenAI, Anthropic, and Google, or access a wider variety through a single OpenRouter integration.
*   **Critique-Revision Loop**: Use a dedicated "Critique Model" to analyze a response for accuracy, bias, and logical fallacies. Then, use that critique to prompt the "Primary Model" for a revised and improved answer.
*   **Context-Aware Conversations**: Upload text-based files (`.txt`, `.md`, `.py`, etc.) to provide rich, persistent context for your entire conversation.
*   **Real-time Cost & Token Tracking**: Monitor the token count and estimated API costs of your conversation as it grows, helping you manage usage effectively.
*   **Full Session Management**: Save, load, name, and delete entire conversation sessions. An autosave feature ensures your work is never lost.
*   **Conversation Summarization**: Condense long conversations with a single click to reduce token usage while preserving the essential context.
*   **Pre-built Scenarios**: Kickstart your analysis with a library of complex, domain-specific test cases.

## Getting Started

### Prerequisites

*   Python 3.11+
*   An environment file for your API keys.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/discursus.git
    cd discursus
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a file named `.env` in the root of the project and add your API keys. You only need to provide keys for the services you intend to use.

    ```env
    # .env

    # For OpenRouter.ai access (recommended)
    OPENROUTER_API_KEY="sk-or-..."

    # For direct API access (optional)
    OPENAI_API_KEY="sk-..."
    ANTHROPIC_API_KEY="sk-ant-..."
    GEMINI_API_KEY="..."
    ```

### Running the Application

Launch the Gradio app with the following command:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:7860` (or another port if 7860 is in use).

## How to Use

The Discursus interface is organized for a structured analytical workflow.

1.  **Select Models**: In the "Advanced" section, choose your `Primary Model` for generating responses and a `Critique Model` for analysis.
2.  **Choose API Provider**: Use the `Use OpenRouter` checkbox to toggle between OpenRouter and direct native APIs.
3.  **Start a Conversation**: Type a message in the main input box or select a pre-built scenario from the `Select an Example Question` dropdown.
4.  **Upload Context**: Use the `Upload Files` button to add documents that the model should reference.
5.  **Generate a Critique**: After receiving a response, click `Generate Critique`. The critique model will analyze the conversation and provide feedback.
6.  **Generate a Revision**: Click `Generate Revision`. The primary model will use the critique to create an improved version of its last response.
7.  **Manage Sessions**: Use the controls under "Saved Sessions" to save your current chat, load a previous one, or start fresh.

## File Structure

```
.
├── app.py                    # Main Gradio application logic
├── requirements.txt          # Python dependencies
├── .env                      # API keys and environment variables (user-created)
├── data/                     # Default directory for saved sessions and logs
│   ├── sessions/
│   └── conversation.json
├── content/                  # Location for test cases and other static content
└── README.md                 # This file
```

## Configuration

### Adding Models

The models available in the dropdowns are defined in the `MODEL_MAP` dictionary in [`app.py`](app.py). You can add new models or edit existing ones by following the established format:

````python
// filepath: app.py
// ...existing code...
MODEL_MAP = {
    "GPT-4o": {
        "id": "openai/gpt-4o",
        "provider": "openai",
        "native_id": "gpt-4o",
        "input_cost_pm": 5.0,
        "output_cost_pm": 15.0
    },
    // Add your custom model here
}
// ...existing code...
````