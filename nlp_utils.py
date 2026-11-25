"""NLP and machine-learning helpers used by the Streamlit app.

Everything here is pure Python (scikit-learn, numpy, pandas, vaderSentiment,
pypdf, python-docx), so it installs and runs the same on Windows and Linux.
"""

from __future__ import annotations

import io
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")

_sentiment = SentimentIntensityAnalyzer()


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def read_uploaded_file(name: str, data: bytes) -> str:
    """Extract plain text from an uploaded .txt/.md/.csv/.pdf/.docx file."""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(data))
        text_cols = df.select_dtypes(include="object").columns
        return "\n".join(df[text_cols].astype(str).agg(" ".join, axis=1))

    # txt, md and anything else: try common encodings
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Basic processing
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation and on line breaks (headings, bullets, rows)."""
    text = re.sub(r"[ \t]*\r?\n[ \t]*(?=[a-z])", " ", text)  # re-join hard-wrapped lines
    sentences = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[#>*\-+]+|\d+[.)])\s*", "", line)  # strip md/list markers
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(line))
    return [s for s in sentences if len(_WORD.findall(s)) >= 2]


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def text_stats(text: str) -> dict:
    words = tokenize(text)
    sentences = split_sentences(text)
    n_words = len(words)
    n_sent = max(len(sentences), 1)
    syllables = sum(_count_syllables(w) for w in words)
    # Flesch Reading Ease (higher = easier to read)
    flesch = (
        206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / max(n_words, 1))
        if n_words
        else 0.0
    )
    return {
        "Characters": len(text),
        "Words": n_words,
        "Unique words": len(set(words)),
        "Sentences": len(sentences),
        "Avg words / sentence": round(n_words / n_sent, 1),
        "Reading ease (Flesch)": round(flesch, 1),
    }


def _count_syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups) - (1 if word.endswith("e") and len(groups) > 1 else 0)
    return max(count, 1)


def word_frequencies(text: str, top_n: int = 20) -> pd.DataFrame:
    words = [w for w in tokenize(text) if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    counts = Counter(words).most_common(top_n)
    return pd.DataFrame(counts, columns=["word", "count"])


# --------------------------------------------------------------------------- #
# Machine learning
# --------------------------------------------------------------------------- #
def sentiment(text: str) -> dict:
    """Overall sentiment (VADER) for the whole text."""
    scores = _sentiment.polarity_scores(text)
    c = scores["compound"]
    label = "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral"
    return {"label": label, **scores}


def sentence_sentiment(sentences: list[str]) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(sentences, 1):
        c = _sentiment.polarity_scores(s)["compound"]
        rows.append({"#": i, "sentence": s, "compound": c})
    return pd.DataFrame(rows)


def tfidf_keywords(sentences: list[str], top_n: int = 15) -> pd.DataFrame:
    """Keywords/keyphrases ranked by TF-IDF, treating each sentence as a document."""
    if not sentences:
        return pd.DataFrame(columns=["keyword", "score"])
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    order = scores.argsort()[::-1][:top_n]
    return pd.DataFrame({"keyword": terms[order], "score": scores[order].round(3)})


def extractive_summary(sentences: list[str], n: int = 3) -> list[str]:
    """TextRank-style summary: PageRank over a TF-IDF sentence-similarity graph."""
    if len(sentences) <= n:
        return sentences
    X = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)

    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    M = sim / row_sums

    d, k = 0.85, len(sentences)
    rank = np.ones(k) / k
    for _ in range(100):
        new = (1 - d) / k + d * M.T @ rank
        if np.abs(new - rank).sum() < 1e-6:
            break
        rank = new

    top = sorted(rank.argsort()[::-1][:n])  # keep original order
    return [sentences[i] for i in top]


def topic_model(sentences: list[str], n_topics: int = 3, n_words: int = 6) -> pd.DataFrame:
    """Latent Dirichlet Allocation topics over sentences."""
    vec = CountVectorizer(stop_words="english", min_df=1)
    X = vec.fit_transform(sentences)
    n_topics = max(1, min(n_topics, X.shape[0], X.shape[1]))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    terms = vec.get_feature_names_out()
    rows = []
    for i, comp in enumerate(lda.components_, 1):
        top = comp.argsort()[::-1][:n_words]
        rows.append({"topic": f"Topic {i}", "top words": ", ".join(terms[top])})
    return pd.DataFrame(rows)


def cluster_sentences(sentences: list[str], n_clusters: int = 3) -> pd.DataFrame:
    """Group similar sentences with K-Means on TF-IDF vectors."""
    X = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    n_clusters = max(1, min(n_clusters, len(sentences)))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return (
        pd.DataFrame({"cluster": labels + 1, "sentence": sentences})
        .sort_values("cluster", kind="stable")
        .reset_index(drop=True)
    )
