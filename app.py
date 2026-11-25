"""Text Insights — a small Streamlit NLP/ML app.

Run with:  streamlit run app.py
"""

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

st.set_page_config(page_title="Text Insights", page_icon="📝", layout="wide")
st.title("📝 Text Insights")
st.caption("Paste text or upload a file, then explore it with NLP and machine learning.")

# --------------------------------------------------------------------------- #
# Sidebar settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    n_summary = st.slider("Summary sentences", 1, 10, 3)
    n_keywords = st.slider("Keywords", 5, 30, 15)
    n_topics = st.slider("Topics (LDA)", 1, 8, 3)
    n_clusters = st.slider("Sentence clusters (K-Means)", 1, 8, 3)
    st.divider()
    st.caption("Models: VADER sentiment · TF-IDF · TextRank · LDA · K-Means")

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
tab_paste, tab_upload = st.tabs(["✍️ Paste text", "📂 Upload file"])

with tab_paste:
    if st.button("Load sample text"):
        st.session_state["pasted"] = SAMPLE_TEXT
    pasted = st.text_area("Your text", key="pasted", height=220,
                          placeholder="Paste or type some English text here…")

with tab_upload:
    uploaded = st.file_uploader("Choose a file", type=["txt", "md", "csv", "pdf", "docx"])
    file_text = ""
    if uploaded is not None:
        try:
            file_text = nlp.read_uploaded_file(uploaded.name, uploaded.getvalue())
            st.success(f"Loaded **{uploaded.name}** ({len(file_text):,} characters)")
            with st.expander("Preview"):
                st.text(file_text[:3000] + ("…" if len(file_text) > 3000 else ""))
        except Exception as exc:  # noqa: BLE001 - show any parsing error to the user
            st.error(f"Could not read file: {exc}")

source = st.radio("Analyze", ["Pasted text", "Uploaded file"], horizontal=True,
                  index=1 if uploaded is not None and not pasted else 0)
text = (pasted if source == "Pasted text" else file_text) or ""

if st.button("🔍 Analyze", type="primary"):
    st.session_state["analyzed"] = True
if not st.session_state.get("analyzed"):
    st.stop()

if len(text.strip()) < 20:
    st.warning("Please provide a bit more text (at least a sentence or two).")
    st.stop()


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def analyze(text: str, n_summary: int, n_keywords: int, n_topics: int, n_clusters: int):
    sentences = nlp.split_sentences(text)
    return {
        "sentences": sentences,
        "stats": nlp.text_stats(text),
        "sentiment": nlp.sentiment(text),
        "sent_sentiment": nlp.sentence_sentiment(sentences),
        "keywords": nlp.tfidf_keywords(sentences, n_keywords),
        "freq": nlp.word_frequencies(text),
        "summary": nlp.extractive_summary(sentences, n_summary),
        "topics": nlp.topic_model(sentences, n_topics),
        "clusters": nlp.cluster_sentences(sentences, n_clusters),
    }


with st.spinner("Analyzing…"):
    r = analyze(text, n_summary, n_keywords, n_topics, n_clusters)

# Overview metrics
cols = st.columns(len(r["stats"]) + 1)
for col, (k, v) in zip(cols, r["stats"].items()):
    col.metric(k, f"{v:,}" if isinstance(v, int) else v)
s = r["sentiment"]
emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}[s["label"]]
cols[-1].metric("Sentiment", f"{emoji} {s['label']}", f"{s['compound']:+.2f}")

t_sum, t_sent, t_kw, t_topics, t_clu = st.tabs(
    ["Summary", "Sentiment", "Keywords", "Topics", "Clusters"]
)

with t_sum:
    st.subheader("Extractive summary (TextRank)")
    for sentence in r["summary"]:
        st.markdown(f"- {sentence}")

with t_sent:
    st.subheader("Overall")
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", f"{s['pos']:.0%}")
    c2.metric("Neutral", f"{s['neu']:.0%}")
    c3.metric("Negative", f"{s['neg']:.0%}")
    st.subheader("Sentiment across sentences")
    df = r["sent_sentiment"]
    st.line_chart(df.set_index("#")["compound"])
    st.dataframe(df, hide_index=True, width="stretch")

with t_kw:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("TF-IDF keywords")
        st.bar_chart(r["keywords"].set_index("keyword")["score"], horizontal=True)
    with c2:
        st.subheader("Most frequent words")
        st.bar_chart(r["freq"].set_index("word")["count"], horizontal=True)

with t_topics:
    st.subheader("Latent Dirichlet Allocation")
    st.dataframe(r["topics"], hide_index=True, width="stretch")

with t_clu:
    st.subheader("K-Means sentence clusters")
    for cid, group in r["clusters"].groupby("cluster"):
        with st.expander(f"Cluster {cid} — {len(group)} sentence(s)", expanded=True):
            for sentence in group["sentence"]:
                st.markdown(f"- {sentence}")
