# Mini Search Engine:NLP based document retrieval system
<table border="0" cellspacing="0" cellpadding="15" align="center">
  <tr>
    <td align="center" valign="top" width="16%">
      <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /><br /><br />
      <b>PYTHON</b><br />
      <sub>Vectorisation</sub>
    </td>
    <td align="center" valign="top" width="16%">
      <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" /><br /><br />
      <b>FLASK</b><br />
      <sub>Interface Web</sub>
    </td>
    <td align="center" valign="top" width="16%">
      <img src="https://img.shields.io/badge/NLTK-2C8EBB?style=for-the-badge&logo=python&logoColor=white" alt="NLTK" /><br /><br />
      <b>NLTK</b><br />
      <sub>Nettoyage (NLP)</sub>
    </td>
    <td align="center" valign="top" width="16%">
      <img src="https://img.shields.io/badge/Gensim-B32134?style=for-the-badge&logo=databricks&logoColor=white" alt="Gensim" /><br /><br />
      <b>GENSIM</b><br />
      <sub>Modèle Word2Vec</sub>
    </td>
    <td align="center" valign="top" width="16%">
      <img src="https://img.shields.io/badge/BM25-FFB000?style=for-the-badge&logo=apache&logoColor=white" alt="BM25" /><br /><br />
      <b>RANK_BM25</b><br />
      <sub>Score Lexical</sub>
    </td>
    <td align="center" valign="top" width="16%">
      <img src="https://img.shields.io/badge/PDFPlumber-47A141?style=for-the-badge&logo=read-the-docs&logoColor=white" alt="PDFPlumber" /><br /><br />
      <b>PDFPLUMBER</b><br />
      <sub>Extraction Texte</sub>
    </td>
  </tr>
</table>

This repository contains the source code for a bilingual (French/English) document indexing and retrieval system applied to a corpus of 249 academic computer science papers. The project provides a critical evaluation comparing a traditional probabilistic lexical approach (BM25) against a semantic vector space approach (Word2Vec + Cosine Similarity).

For a complete theoretical, mathematical, and technical analysis, please refer to the [Project Report](report.pdf).

## Project Architecture

The software pipeline is structured as follows:

### Phase 1: Ingestion and Data Preparation
* data.py: Automated paper collection via the arXiv API and integration of university course materials.
* extract.py: Page by page text extraction using pdfplumber, optimized to preserve reading order in multicolumn layouts.
* clean.py: Text noise removal, bilingual stopword filtering (English and French), and stemming via SnowballStemmer.
* representation.py: Local training of the Word2Vec model and Mean Pooling aggregation to generate dense 100 dimensional vectors.

### Phase 2: Search Engine and Evaluation
* metadata.py: Document card extraction and structuring (titles, authors, file paths) to decouple the mathematical data from the user display.
* search_engine.py: Core query execution running both the BM25 probabilistic algorithm and a nearest neighbor search using cosine similarity.
* app.py: Dynamic web user interface built with the Flask framework.
* evaluate.py: Quantitative evaluation script measuring Precision, Recall, and F1-Score across validation test queries.

## Results and Critical Analysis

The empirical evaluation of the system highlighted the following global performance metrics:

* BM25 (Lexical Approach): F1-Score of 0.2738 (Precision: 0.1750, Recall: 0.6875)
* Word2Vec (Semantic Approach): F1-Score of 0.0937 (Precision: 0.0750, Recall: 0.1250)

### Data Engineering Diagnoses
1. Small Corpus Bottleneck: With only 249 documents, the local co-occurrence density of technical terms is too low for a locally trained neural network to build a perfectly separated vector topology.
2. Forced Top-K Trap: The cosine similarity engine is hardcoded to extract exactly the 5 closest vectors. When a query is highly specific and absent from the dataset, the system still pulls the least distant vectors, drowning precision in mathematical false positives.

## Future Perspectives

* Hybrid Search: Implementing a combined scoring function in production: Score_Final = alpha * Score_BM25 + (1 - alpha) * Score_W2V.
* Pre-trained Models: Replacing the local Word2Vec instance with global embeddings (FastText or GloVe) pre-trained on massive scientific datasets.

## Installation and Setup

1. Dependencies: Install required libraries (Flask, pdfplumber, nltk, gensim, rank_bm25).
2. Indexing: Run Phase 1 scripts to build the dense matrices and lexical index.
3. Server: Run app.py to start the local web interface on port 5000.
