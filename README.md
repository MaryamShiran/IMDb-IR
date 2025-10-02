# IMDb-IR: Information Retrieval System for IMDb

<img src="./IMDB_Logo.jpeg" alt="IMDb Logo" width="100%" height="auto" />

An end-to-end educational IR system built for the Modern Information Retrieval course at Sharif University of Technology (Instructor: Dr. Mahdieh Soleymani Baghshah). The project covers crawling IMDb, preprocessing, indexing, multiple retrieval models, ranking, spell correction, snippets, evaluation, link analysis (HITS), word embeddings, classification, clustering, and a Streamlit UI.

## Key Features
- Crawler for IMDb movie data with schema validation
- Text preprocessing with custom stopwords
- Indexing (document, tiered, document-length, metadata)
- Retrieval models: Vector Space Model (tf-idf variants), Okapi BM25, Unigram Language Model with smoothing
- Spell correction using shingles + Jaccard with TF-aware re-ranking
- Snippet generation with query-term highlighting and window merging
- Evaluation metrics: Precision, Recall, F1, MAP, NDCG, MRR
- Link Analysis (HITS) for actors/movies graphs
- Word Embeddings (FastText) utilities
- Sentiment classification of reviews (Naive Bayes, SVM, KNN, MLP)
- Clustering and dimensionality reduction (PCA, t-SNE) with visualizations
- Streamlit UI for interactive search and exploration

## Project Structure
```
codes/
  Logic/
    core/
      utility/         # preprocess, evaluation, scorer, snippet, spell_correction, stopwords
      indexer/         # inverted indexes, tiered index, metadata, doc lengths, LSH
      link_analysis/   # HITS analyzer and graph utilities
      classification/  # classifiers and data loader
      clustering/      # clustering utilities, metrics, DR, examples
      word_embedding/  # FastText data loader and model wrapper
      rag/             # RAG notebook (mask)
      recommender_system/  # recommender notebook
      search.py        # search engine orchestration
    README.md          # Logic module overview
    tests/             # test for crawler (schema/size checks)
  UI/
    main.py            # Streamlit app entrypoint
    TopRecent.py       # UI helper
    README.md          # UI usage
    requirements.txt   # UI dependencies
  documentation/       # Sphinx docs sources
  README.md            # You are here
```

## Getting Started

### Prerequisites
- Python 3.9+ recommended
- A virtual environment (e.g., `venv`)

### Setup
```bash
cd codes
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# UI dependencies
pip install -r UI/requirements.txt

# Install additional libraries as you implement optional modules (classification, embedding, etc.).
```

## Data
- Raw crawled IMDb JSON (Phase 1) is available here: https://drive.google.com/file/d/1Lq2zVJlN_B4kUAu4VafQ4jXMIQiAR9vI/view?usp=sharing
- For near-duplicate detection (LSH), integrate `Logic/core/indexer/LSHFakeData.json` into your main dataset during LSH only, then remove fake entries afterwards.

## Running the UI
```bash
cd codes/UI
streamlit run main.py
```
Then open `http://localhost:8501/` in your browser.

## Core Workflows (Logic)
- Crawler: see `Logic/core/utility/crawler.py`; validate with `Logic/tests/test_crawler.py`.
- Preprocess: `Logic/core/utility/preprocess.py` (uses `Logic/core/utility/stopwords.txt`).
- Indexing: `Logic/core/indexer/index.py` (+ `document_lengths_index.py`, `metadata_index.py`, `tiered_index.py`).
- Search: `Logic/core/search.py` orchestrates retrieval over fields (stars, genres, summaries) with safe/tiered ranking. Also supports Unigram LM via `Scorer`.
- Scorer: `Logic/core/utility/scorer.py` implements VSM, BM25, and Unigram LM scoring.
- Spell Correction: `Logic/core/utility/spell_correction.py` (shingles + Jaccard + TF normalization).
- Snippets: `Logic/core/utility/snippet.py` creates merged windows and highlights query terms with `***term***`.
- Evaluation: `Logic/core/utility/evaluation.py` provides standard IR metrics.
- Link Analysis (HITS): `Logic/core/link_analysis/analyzer.py` with `graph.py`.
- Word Embedding: `Logic/core/word_embedding` (FastText utils).
- Classification: `Logic/core/classification` (Naive Bayes, SVM, KNN, MLP) on IMDb reviews.
- Clustering: `Logic/core/clustering` with DR and plots.

## Configuration Notes
- Index paths: `Logic/core/search.py` currently sets `path` to a local Windows path. Update this to your local indexes directory before running search, e.g. set an absolute path on your machine.
- Stopwords: modify `Logic/core/utility/stopwords.txt` as needed.

## Testing
Run the crawler schema/size check:
```bash
python Logic/tests/test_crawler.py
```
Adjust `json_file_path` inside the test to point to your crawled JSON.

## Documentation
Sphinx docs are under `documentation/`. To build locally:
```bash
cd documentation
make html  # or `make.bat html` on Windows
```
Open `_build/html/index.html` in your browser.

## Contributing
Issues and PRs are welcome for improvements and fixes. For course participants, follow your assignment instructions and keep your fork private if required.

## Acknowledgements
Developed as part of the Modern Information Retrieval course at Sharif University of Technology. IMDb is a trademark of IMDb.com, Inc.; data used for educational purposes.
