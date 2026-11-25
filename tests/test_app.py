"""UI tests with Streamlit's headless AppTest. LLM calls are mocked, so no key or network is needed."""

from pathlib import Path
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

import llm

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Never call real providers: model listing returns a fixed list."""
    monkeypatch.setattr(llm, "list_models", lambda provider, key: llm.DEFAULT_MODELS[provider])
    for name in llm.ENV_KEYS.values():
        monkeypatch.delenv(name, raising=False)


def analyzed_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="pasted").input(
        "I love this wonderful app. It is fast. The installer was terrible. Overall it is great."
    ).run()
    at.button(key="analyze").click().run()
    assert not at.exception, at.exception
    return at


def test_analyze_page_shows_results():
    at = analyzed_app()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Sentences"] == "4"
    assert metrics["Sentiment"] == "Positive"
    assert len(at.tabs) == 5


def test_short_text_warns():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="pasted").input("Too short").run()
    at.button(key="analyze").click().run()
    assert at.warning and "more text" in at.warning[0].value


def test_assistant_needs_text_then_key():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.switch_page("app_pages/assistant.py").run()
    assert "analyze a text first" in at.info[0].value

    at = analyzed_app()
    at.switch_page("app_pages/assistant.py").run()
    assert not at.exception, at.exception
    assert any("API key" in i.value for i in at.info)
    assert at.chat_input[0].proto.disabled


def test_assistant_streams_reply_with_document_context():
    at = analyzed_app()
    at.text_input(key="api_key_Claude").input("sk-ant-test").run()
    at.switch_page("app_pages/assistant.py").run()

    calls = []

    def fake_stream(provider, api_key, model, system, messages):
        calls.append((provider, api_key, model, system, list(messages)))
        yield "The text "
        yield "is mostly positive."

    with mock.patch.object(llm, "stream_reply", fake_stream):
        at.chat_input[0].set_value("What is the tone?").run()

    assert not at.exception, at.exception
    provider, key, model, system, messages = calls[0]
    assert (provider, key, model) == ("Claude", "sk-ant-test", "claude-opus-5-5")
    assert "installer was terrible" in system and "<ml_analysis>" in system
    assert messages == [{"role": "user", "content": "What is the tone?"}]
    assert at.session_state.chat["messages"][-1]["content"] == "The text is mostly positive."


def test_assistant_shows_provider_error():
    at = analyzed_app()
    at.text_input(key="api_key_Claude").input("bad-key").run()
    at.switch_page("app_pages/assistant.py").run()

    def failing(*args):
        raise llm.LLMError("The API key was rejected.")
        yield  # pragma: no cover

    with mock.patch.object(llm, "stream_reply", failing):
        at.chat_input[0].set_value("Hello?").run()
    assert "rejected" in at.error[0].value
    assert at.session_state.chat["messages"] == []
