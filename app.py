"""Text Insights: NLP + ML analysis with an LLM assistant.

Run with:  streamlit run app.py
"""

import os

import streamlit as st

import llm

st.set_page_config(page_title="Text insights", page_icon=":material/text_snippet:", layout="wide")

page = st.navigation([
    st.Page("app_pages/analyze.py", title="Analyze", icon=":material/analytics:", default=True),
    st.Page("app_pages/assistant.py", title="Ask AI", icon=":material/smart_toy:"),
])


def _configured_key(provider: str) -> str:
    """API key from .streamlit/secrets.toml or an environment variable, if set."""
    name = llm.ENV_KEYS[provider]
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except FileNotFoundError:  # no secrets.toml
        pass
    return os.environ.get(name, "")


@st.cache_data(ttl="1h", show_spinner=False)
def _models(provider: str, api_key: str) -> list[str]:
    return llm.list_models(provider, api_key)


with st.sidebar:
    st.subheader("AI assistant", icon=":material/key:")
    provider = st.segmented_control("Provider", llm.PROVIDERS, default=llm.CLAUDE, required=True,
                                    key="llm_provider")
    configured = _configured_key(provider)
    api_key = st.text_input(
        "API key", type="password", key=f"api_key_{provider}", value=configured, persist_state="session",
        placeholder="sk-ant-..." if provider == llm.CLAUDE else "sk-or-...",
        help=f"Kept only in this browser session. You can also set {llm.ENV_KEYS[provider]} "
             "as an environment variable or in .streamlit/secrets.toml.",
    )

    models = llm.DEFAULT_MODELS[provider]
    if api_key or provider == llm.OPENROUTER:  # OpenRouter's list is public
        try:
            models = _models(provider, api_key) or models
        except llm.LLMError as exc:
            st.caption(f":orange[:material/warning:] Couldn't load models: {exc}")
    default_model = llm.DEFAULT_MODELS[provider][0]
    st.selectbox(
        "Model", models, key=f"model_{provider}", persist_state="session",
        index=models.index(default_model) if default_model in models else 0,
        accept_new_options=True, help="Pick from the list or type any model ID.",
    )
    st.caption(
        "Get a key at [console.anthropic.com](https://console.anthropic.com)"
        if provider == llm.CLAUDE else
        "Get a key at [openrouter.ai/keys](https://openrouter.ai/keys)"
    )

page.run()
