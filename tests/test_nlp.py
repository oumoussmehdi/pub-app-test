import nlp_utils as nlp

TEXT = (
    "I love this wonderful app. It is fast and the docs are great. "
    "The installer was terrible and the errors were confusing. "
    "Overall the team did a fantastic job."
)


def test_split_sentences_handles_markdown_and_wrapped_lines():
    text = "# Title here\n- First bullet item\nThis line is hard\nwrapped across lines.\n"
    assert nlp.split_sentences(text) == [
        "Title here", "First bullet item", "This line is hard wrapped across lines."
    ]


def test_analyze_all_returns_every_section():
    r = nlp.analyze_all(TEXT, n_summary=2, n_keywords=5, n_topics=2, n_clusters=2)
    assert r["stats"]["Sentences"] == 4
    assert r["sentiment"]["label"] == "Positive"
    assert len(r["summary"]) == 2
    assert len(r["keywords"]) == 5
    assert set(r["clusters"]["cluster"]) <= {1, 2}
    assert len(r["sent_sentiment"]) == 4


def test_negative_sentence_scores_negative():
    df = nlp.sentence_sentiment(nlp.split_sentences(TEXT))
    assert df.loc[2, "compound"] < 0


def test_report_and_digest_mention_results():
    r = nlp.analyze_all(TEXT)
    assert "## Sentiment" in nlp.build_report(r, "sample.txt")
    assert "Overall sentiment (VADER): Positive" in nlp.analysis_digest(r)


def test_read_uploaded_text_with_legacy_encoding():
    assert nlp.read_uploaded_file("a.txt", "café".encode("cp1252")) == "café"
