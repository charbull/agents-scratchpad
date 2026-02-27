# Research Agent

An autonomous research agent built with the Google Agent Development Kit (ADK).
Inpsired by https://www.anthropic.com/engineering/multi-agent-research-system

## Setup

1.  **Install dependencies:**
    ```bash
    uv sync
    ```

2.  **Configure API Key:**
    Create a `.env` file in the root directory and add your Gemini API key:
    ```
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

## Running the Agent

### Command Line Interface (CLI)

You can run the agent directly from your terminal:

```bash
uv run python research_agent/agent.py "Your research query here"
```

If you don't provide a query, it will prompt you for one.

### Web UI

You can also interact with the agent through a local web interface:

```bash
uv run adk web .
```

Then open your browser to [http://127.0.0.1:8000](http://127.0.0.1:8000).
