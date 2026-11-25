import hashlib
import re

import altair as alt
import streamlit as st

import nlp_utils as nlp

SAMPLE_TEXT = (
    "Streamlit makes it easy to turn Python scripts into shareable web apps. "
    "Data scientists love how quickly they can build dashboards without writing any front-end code. "
    "However, some users find that complex layouts are harder to achieve. "
    "Machine learning models can be wrapped in a few lines and exposed to non-technical colleagues. "
    "The community is very active and new components are released every week. "
    "Performance can suffer with very large datasets, which is frustrating. "
    "Caching helps a lot, and the documentation explains it clearly. "
    "Overall, it is a fantastic tool for prototyping and internal tools."
)
SENTIMENT_COLORS = alt.Scale(domain=["Positive", "Neutral", "Negative"],
                             range=["#22C55E", "#A1A1AA", "#EF4444"])

st.session_state.setdefault("doc", None)  # {"text", "source", "id"} once analyzed


@st.cache_data(show_spinner=False, max_entries=20)
def read_file(name: str, data: bytes) -> str:
    return nlp.read_uploaded_file(name, data)


@st.cache_data(show_spinner=False, max_entries=20)
def run_analysis(text: str, n_summary: int, n_keywords: int, n_topics: int, n_clusters: int) -> dict:
    return nlp.analyze_all(text, n_summary, n_keywords, n_topics, n_clusters)


def md_escape(text: str) -> str:
    """Escape Markdown so arbitrary document text renders literally."""
    return re.sub(r"([\\`*_{}\[\]<>()#+\-.!|$~:])", r"\\\1", text)


def load_sample() -> None:
    st.session_state.input_mode = "Paste text"
    st.session_state.pasted = SAMPLE_TEXT


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
st.title("Text insights", icon=":material/text_snippet:")
st.caption("Paste text or upload a file, explore it with NLP and machine learning, then ask an AI about it.")

with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        mode = st.segmented_control("Input", ["Paste text", "Upload file"], default="Paste text",
                                    required=True, key="input_mode", persist_state="session")
        st.button("Load sample", icon=":material/auto_awesome:", on_click=load_sample, type="tertiary")

    text, source = "", ""
    if mode == "Paste text":
        text = st.text_area("Your text", key="pasted", height=200, persist_state="session",
                            placeholder="Paste or type some English text here…",
                            label_visibility="collapsed")
        source = "Pasted text"
    else:
        uploaded = st.file_uploader("Choose a file", type=["txt", "md", "csv", "pdf", "docx"])
        if uploaded is not None:
            try:
                text = read_file(uploaded.name, uploaded.getvalue())
                source = uploaded.name
                if text.strip():
                    st.caption(f":material/check_circle: Loaded {uploaded.name}: {len(text):,} characters")
                else:
                    st.warning("No text found in this file. Scanned PDFs need OCR first.",
                               icon=":material/warning:")
            except Exception as exc:  # noqa: BLE001 - show any parsing error to the user
                st.error(f"Couldn't read the file: {exc}", icon=":material/error:")

    with st.container(horizontal=True, vertical_alignment="center"):
        analyze_clicked = st.button("Analyze", type="primary", icon=":material/search:", key="analyze")
        with st.popover("Options", icon=":material/tune:"):
            n_summary = st.slider("Summary sentences", 1, 10, 3, key="n_summary", persist_state="session")
            n_keywords = st.slider("Keywords", 5, 30, 15, key="n_keywords", persist_state="session")
            n_topics = st.slider("Topics (LDA)", 1, 8, 3, key="n_topics", persist_state="session")
            n_clusters = st.slider("Sentence clusters (K-Means)", 1, 8, 3, key="n_clusters",
                                   persist_state="session")

if analyze_clicked:
    if len(text.strip()) < 20:
        st.warning("Please provide a bit more text (at least a sentence or two).", icon=":material/warning:")
    else:
        st.session_state.doc = {"text": text, "source": source,
                                "id": hashlib.sha1(text.encode()).hexdigest()[:12]}

doc = st.session_state.doc
if doc is None:
    st.stop()

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
with st.spinner("Analyzing…"):
    r = run_analysis(doc["text"], n_summary, n_keywords, n_topics, n_clusters)

if not r["sentences"]:
    st.warning("Couldn't find any sentences to analyze in this text.", icon=":material/warning:")
    st.stop()

with st.container(horizontal=True, vertical_alignment="center"):
    st.subheader("Results", icon=":material/insights:")
    st.badge(doc["source"], icon=":material/description:", color="gray")
    st.space("stretch")
    st.download_button("Report", nlp.build_report(r, doc["source"]), "text-insights-report.md",
                       mime="text/markdown", icon=":material/download:")
    if st.button("Ask AI about this text", icon=":material/smart_toy:", type="primary"):
        st.switch_page("app_pages/assistant.py")

s = r["sentiment"]
with st.container(horizontal=True):
    for label, value in r["stats"].items():
        st.metric(label, f"{value:,}" if isinstance(value, int) else value, border=True, height="stretch")
    st.metric("Sentiment", s["label"], f"{s['compound']:+.2f}", border=True, height="stretch",
              delta_color="normal" if s["compound"] else "off", help="VADER compound score, from -1 to +1")
st.caption("Reading ease (Flesch): 60+ is plain English, below 30 is very hard to read.")

t_sum, t_sent, t_kw, t_topics, t_clu = st.tabs([
    ":material/summarize: Summary", ":material/mood: Sentiment", ":material/key: Keywords",
    ":material/category: Topics", ":material/hub: Clusters",
])

with t_sum:
    st.caption("Extractive summary: the most central sentences, ranked by TextRank (TF-IDF + PageRank).")
    st.markdown("\n".join(f"- {md_escape(x)}" for x in r["summary"]))

with t_sent:
    df = r["sent_sentiment"].copy()
    df["label"] = df["compound"].map(lambda c: "Positive" if c >= 0.05 else "Negative" if c <= -0.05
                                     else "Neutral")
    with st.container(horizontal=True):
        st.metric("Positive", f"{s['pos']:.0%}", border=True)
        st.metric("Neutral", f"{s['neu']:.0%}", border=True)
        st.metric("Negative", f"{s['neg']:.0%}", border=True)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("#:O", title="Sentence"),
        y=alt.Y("compound:Q", title="Sentiment", scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("label:N", scale=SENTIMENT_COLORS, legend=None),
        tooltip=["#", "sentence", alt.Tooltip("compound:Q", format="+.2f")],
    )
    st.altair_chart(chart, alt="Sentiment score of each sentence, from -1 (negative) to +1 (positive)")

    st.caption("Sentences highlighted by sentiment")
    color = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
    with st.container(border=True, height=300):
        st.markdown(" ".join(f":{color[row.label]}-background[{md_escape(row.sentence)}]"
                             for row in df.itertuples()))

with t_kw:
    left, right = st.columns(2)
    with left:
        st.caption("TF-IDF keywords: distinctive words and phrases")
        st.altair_chart(
            alt.Chart(r["keywords"]).mark_bar().encode(
                x=alt.X("score:Q", title="TF-IDF score"), y=alt.Y("keyword:N", sort="-x", title=None),
                tooltip=["keyword", "score"]),
            alt="Top keywords ranked by TF-IDF score")
    with right:
        st.caption("Most frequent words (stop words removed)")
        st.altair_chart(
            alt.Chart(r["freq"]).mark_bar().encode(
                x=alt.X("count:Q", title="Count"), y=alt.Y("word:N", sort="-x", title=None),
                tooltip=["word", "count"]),
            alt="Most frequent words by count")

with t_topics:
    st.caption("Latent Dirichlet Allocation: groups of words that tend to appear together.")
    st.dataframe(r["topics"], hide_index=True, alt="Topics found by LDA with their top words")

with t_clu:
    st.caption("K-Means on TF-IDF vectors: sentences grouped by similar wording.")
    for cid, group in r["clusters"].groupby("cluster"):
        with st.expander(f"Cluster {cid}: {len(group)} sentence(s)", expanded=True,
                         icon=":material/workspaces:"):
            st.markdown("\n".join(f"- {md_escape(x)}" for x in group["sentence"]))
