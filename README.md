# Text Insights

A small Streamlit app that takes text (pasted or loaded from a file) and runs NLP and machine-learning analysis on it.

## Features

- **Input**: paste text, or upload `.txt`, `.md`, `.csv`, `.pdf` or `.docx`
- **Stats**: word/sentence counts and Flesch reading ease
- **Sentiment**: overall and per-sentence (VADER)
- **Keywords**: TF-IDF keyphrases and word frequencies
- **Summary**: extractive TextRank (TF-IDF + PageRank)
- **Topics**: Latent Dirichlet Allocation
- **Clusters**: K-Means grouping of similar sentences

Every dependency is pure Python or ships prebuilt wheels, so the app runs the same on Windows and Linux. It doesn't download any models at runtime.

## Run

**Windows**
```bat
run.bat
```

**Linux / macOS**
```bash
chmod +x run.sh
./run.sh
```

Each script creates a `.venv` on first run, installs the requirements, and opens the app at http://localhost:8501.

**Manual setup**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.9+. Text analysis is tuned for English.
