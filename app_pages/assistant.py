import streamlit as st

import llm
import nlp_utils as nlp

SUGGESTIONS = {
    ":blue[:material/summarize:] Summarize": "Summarize this text in 3–5 bullet points.",
    ":green[:material/mood:] Tone and sentiment": (
        "Describe the tone and sentiment of this text. Does it agree with the VADER result? "
        "Point out anything a lexicon-based model would miss."
    ),
    ":violet[:material/person_search:] Named entities": (
        "List the people, organizations, places, products and dates mentioned, grouped by type."
    ),
    ":orange[:material/fact_check:] Critique the ML analysis": (
        "Review the ML analysis results (keywords, topics, summary). Which are useful, which are noise, "
        "and what do they miss compared with your own reading?"
    ),
    ":gray[:material/quiz:] Questions it answers": "Suggest 5 questions this text answers, each with a short answer.",
}
LARGE_DOC_CHARS = 400_000  # ~100k tokens: fine for Claude, too big for some OpenRouter models

st.session_state.setdefault("doc", None)
st.session_state.setdefault("chat", {"doc_id": None, "messages": []})
st.session_state.setdefault("pending_prompt", None)


@st.cache_data(show_spinner=False, max_entries=20)
def analysis_digest(text: str) -> str:
    return nlp.analysis_digest(nlp.analyze_all(text))


def use_suggestion() -> None:
    """Turn a clicked chip into a one-shot prompt, then clear the chip."""
    picked = st.session_state.suggestion
    if picked:
        st.session_state.pending_prompt = SUGGESTIONS[picked]
    st.session_state.suggestion = None

st.title("Ask AI", icon=":material/smart_toy:")

doc = st.session_state.doc
if doc is None:
    st.info("Load and analyze a text first, then come back to chat about it.",
            icon=":material/info:", title="No text yet")
    if st.button("Go to Analyze", icon=":material/arrow_back:"):
        st.switch_page("app_pages/analyze.py")
    st.stop()

provider = st.session_state.llm_provider
api_key = st.session_state.get(f"api_key_{provider}", "")
model = st.session_state.get(f"model_{provider}") or llm.DEFAULT_MODELS[provider][0]

chat = st.session_state.chat
if chat["doc_id"] != doc["id"]:  # new document → fresh conversation
    chat.update(doc_id=doc["id"], messages=[])

with st.container(horizontal=True, vertical_alignment="center"):
    st.badge(doc["source"], icon=":material/description:", color="gray")
    st.badge(f"{provider} · {model}", icon=":material/memory:", color="blue")
    st.caption(f"{len(doc['text']):,} characters")
    st.space("stretch")
    if st.button("New chat", icon=":material/refresh:", type="tertiary", disabled=not chat["messages"]):
        chat["messages"] = []
        st.rerun()

if len(doc["text"]) > LARGE_DOC_CHARS:
    st.warning("This document is very long and may exceed the context window of smaller models. "
               "If you get an error, pick a model with a larger context.", icon=":material/warning:")

if not api_key:
    st.info(f"Enter your {provider} API key in the sidebar to start chatting.", icon=":material/key:")

for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything about the text", key="chat_prompt", submit_mode="disable",
                       disabled=not api_key)

if st.session_state.pending_prompt:
    prompt = prompt or st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if not chat["messages"] and api_key and not prompt:
    st.pills("Try asking", list(SUGGESTIONS), label_visibility="collapsed", key="suggestion",
             on_change=use_suggestion)

if prompt:
    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system = llm.build_system_prompt(doc["text"], analysis_digest(doc["text"]))
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(llm.stream_reply(provider, api_key, model, system, chat["messages"]))
        except llm.LLMError as exc:
            chat["messages"].pop()  # let the user retry the same question
            st.error(str(exc), icon=":material/error:")
            st.stop()
    chat["messages"].append({"role": "assistant", "content": reply if isinstance(reply, str) else str(reply)})
