"""LLM providers: Claude (Anthropic SDK) and OpenRouter (OpenAI-compatible API).

No Streamlit imports here, so the module can be tested on its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import anthropic
import openai
from anthropic.types import MessageParam, TextBlockParam
from anthropic.types.beta import BetaMessageParam
from openai.types.chat import ChatCompletionMessageParam

CLAUDE = "Claude"
OPENROUTER = "OpenRouter"
PROVIDERS = [CLAUDE, OPENROUTER]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Shown before an API key is entered (and if listing models fails).
DEFAULT_MODELS = {
    CLAUDE: ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2"],
    OPENROUTER: ["anthropic/claude-opus-5.5", "openai/gpt-5", "google/gemini-2.5-pro",
                 "meta-llama/llama-4-maverick"],
}

# Claude models that support server-side refusal fallbacks
# (a declined request is retried on a suitable fallback model in the same call).
_FALLBACK_MODELS = {
    "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
}

ENV_KEYS = {CLAUDE: "ANTHROPIC_API_KEY", OPENROUTER: "OPENROUTER_API_KEY"}

MAX_TOKENS = 16000


class LLMError(Exception):
    """A provider error turned into a message that can be shown to the user."""


def list_models(provider: str, api_key: str) -> list[str]:
    """Model IDs available to this key, newest/most relevant first."""
    try:
        if provider == CLAUDE:
            client = anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=20)
            return [m.id for m in client.models.list()]
        # OpenRouter's model list is public, so it works before a key is entered.
        client = openai.OpenAI(api_key=api_key or "public", base_url=OPENROUTER_BASE_URL,
                               max_retries=1, timeout=20)
        return sorted(m.id for m in client.models.list())
    except (anthropic.AnthropicError, openai.OpenAIError) as exc:
        raise _friendly(exc) from exc


def build_system_prompt(text: str, analysis: str) -> str:
    return (
        "You are a careful text analyst inside an NLP app. The user has loaded the document below, "
        "and the app has already run classical NLP/ML analysis on it (VADER sentiment, TF-IDF keywords, "
        "TextRank summary, LDA topics, K-Means clusters).\n\n"
        "Answer questions about the document. Ground every claim in the document text and quote short "
        "passages when useful. If the document doesn't contain the answer, say so. When asked about the "
        "ML results, interpret them and point out where they are misleading (for example, lexicon-based "
        "sentiment missing sarcasm). Use Markdown; keep answers concise unless asked for detail.\n\n"
        f"<ml_analysis>\n{analysis}\n</ml_analysis>\n\n<document>\n{text}\n</document>"
    )


def stream_reply(
    provider: str, api_key: str, model: str, system: str, messages: list[dict[str, str]]
) -> Iterator[str]:
    """Yield the assistant's reply as text chunks (for st.write_stream)."""
    try:
        if provider == CLAUDE:
            yield from _stream_claude(api_key, model, system, messages)
        else:
            yield from _stream_openrouter(api_key, model, system, messages)
    except (anthropic.AnthropicError, openai.OpenAIError) as exc:
        raise _friendly(exc) from exc


def _stream_claude(api_key: str, model: str, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
    client = anthropic.Anthropic(api_key=api_key)
    # The document lives in the system prompt; caching it makes follow-up questions cheaper and faster.
    system_block: TextBlockParam = {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}

    if model in _FALLBACK_MODELS:
        stream_ctx = client.beta.messages.stream(
            model=model, max_tokens=MAX_TOKENS,
            system=[system_block],
            messages=cast(list[BetaMessageParam], messages),
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
    else:
        stream_ctx = client.messages.stream(
            model=model, max_tokens=MAX_TOKENS, system=[system_block],
            messages=cast(list[MessageParam], messages),
        )

    with stream_ctx as stream:
        yield from stream.text_stream
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        yield "\n\n*The model declined to answer this request.*"
    elif final.stop_reason == "max_tokens":
        yield "\n\n*Reply cut off at the length limit. Ask me to continue.*"


def _stream_openrouter(api_key: str, model: str, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
    client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    chat: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system}, *cast(list[ChatCompletionMessageParam], messages)
    ]
    stream = client.chat.completions.create(model=model, max_tokens=MAX_TOKENS, messages=chat, stream=True)
    finish = None
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.delta and choice.delta.content:
            yield choice.delta.content
        finish = choice.finish_reason or finish
    if finish == "length":
        yield "\n\n*Reply cut off at the length limit. Ask me to continue.*"


def _friendly(exc: Exception) -> LLMError:
    """Map SDK exceptions (most specific first) to short user-facing messages."""
    if isinstance(exc, (anthropic.AuthenticationError, openai.AuthenticationError)):
        return LLMError("The API key was rejected. Check that it's correct and active.")
    if isinstance(exc, (anthropic.PermissionDeniedError, openai.PermissionDeniedError)):
        return LLMError("This key doesn't have access to that model.")
    if isinstance(exc, (anthropic.NotFoundError, openai.NotFoundError)):
        return LLMError("Model not found. Pick another model ID.")
    if isinstance(exc, (anthropic.RateLimitError, openai.RateLimitError)):
        return LLMError("Rate limited or out of credits. Wait a moment and retry.")
    if isinstance(exc, (anthropic.BadRequestError, openai.BadRequestError)):
        return LLMError(f"The provider rejected the request: {getattr(exc, 'message', exc)}")
    if isinstance(exc, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return LLMError("Couldn't reach the provider. Check your internet connection or proxy.")
    if isinstance(exc, (anthropic.APIStatusError, openai.APIStatusError)):
        return LLMError(f"Provider error ({exc.status_code}). Try again shortly.")
    return LLMError(str(exc))
