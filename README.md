# Text Insights

A Streamlit app that analyzes text with classic NLP and machine learning, then lets you chat with an LLM (Claude or any OpenRouter model) about the same text.

## Features

**Analyze page**
- Paste text, or upload a `.txt`, `.md`, `.csv`, `.pdf` or `.docx` file (up to 50 MB)
- Stats: word and sentence counts, Flesch reading ease
- Sentiment (VADER): an overall score, a per-sentence chart, and the text highlighted green, gray or red by sentiment
- Keywords (TF-IDF) and word frequencies, as sorted bar charts
- Extractive summary (TextRank), topics (LDA) and sentence clusters (K-Means)
- Download the results as a Markdown report

**Ask AI page**
- Chat about the loaded text. Replies stream in as they're written.
- Pick the provider and model in the sidebar:
  - **Claude:** uses your Anthropic API key. The model list loads from your account.
  - **OpenRouter:** uses your OpenRouter key and can reach hundreds of models. The model list is public, so it loads even before you enter a key.
  - You can also type any model ID by hand.
- One-click prompts: summarize, tone and sentiment, named entities, critique the ML analysis, questions the text answers
- The model sees both the document and the ML results, so it can explain them or point out where they're wrong.

## Run

**Windows:** `run.bat`
**Linux / macOS:** `chmod +x run.sh && ./run.sh`

The script creates a `.venv`, installs the requirements, and opens http://localhost:8501.

Manual setup:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## API keys

Enter a key in the sidebar. It stays only in your browser session.

To prefill keys, choose one of:
- Set the `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY` environment variable.
- Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in. That file is git-ignored.

Get keys at [console.anthropic.com](https://console.anthropic.com) and [openrouter.ai/keys](https://openrouter.ai/keys).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the NLP logic with plain pytest and the UI with Streamlit's headless `AppTest`. LLM calls are mocked, so the tests need no key and cost nothing. GitHub Actions runs them on Ubuntu and Windows.

## Project layout

| Path | What it is |
|------|------------|
| `app.py` | Entry point: page navigation and the sidebar's LLM settings |
| `app_pages/analyze.py` | Input and the NLP/ML results |
| `app_pages/assistant.py` | LLM chat about the text |
| `nlp_utils.py` | NLP/ML functions (no Streamlit imports) |
| `llm.py` | Claude (Anthropic SDK) and OpenRouter (OpenAI-compatible SDK) streaming |
| `.streamlit/config.toml` | Light/dark theme and upload limit |
