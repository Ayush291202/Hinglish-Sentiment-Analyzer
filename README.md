# 🇮🇳 Hinglish Sentiment Analyzer

A sentiment analysis system for **code-mixed Hindi-English (Hinglish)** text using traditional NLP techniques — custom Word2Vec embeddings and machine learning classifiers.

## Problem Statement

Given a Hinglish text input, predict its sentiment polarity: **Positive**, **Negative**, or **Neutral**.

```
Input:  "Yeh product bilkul bakwaas hai, waste of money"
Output: NEGATIVE (confidence: 91%)
```

## Pipeline

```
Raw Text → Preprocessing → Word2Vec Embeddings → Avg-Word2Vec → Classifier → Sentiment
```

1. **Preprocessing** — lowercasing, URL/mention removal, spelling normalization, tokenization
2. **Word2Vec** — skip-gram embeddings (dim=150) trained from scratch on 21K+ Hinglish sentences
3. **Avg-Word2Vec** — sentence vector = mean of all token embeddings
4. **Classifier** — XGBoost (best) and SVM compared; XGBoost selected with **0.64 macro F1**

## Dataset

~21,369 labeled Hinglish sentences merged from:
- SemEval-2020 Task 9 (train/val/test) — ~20K tweets
- Additional Hinglish sentiment corpus — ~2.7K sentences

| Class    | Count |
|----------|-------|
| Negative | 5,976 |
| Neutral  | 8,210 |
| Positive | 7,183 |

## Project Structure

```
hinglish-sentiment-analyzer/
├── data/
│   └── merged.csv              # Combined dataset
├── models/
│   ├── word2vec.model          # Trained Word2Vec
│   └── classifier.pkl          # Trained XGBoost
├── src/
│   ├── merge_data.py           # Dataset merging & deduplication
│   ├── preprocess.py           # Text cleaning & normalization
│   ├── train_word2vec.py       # Word2Vec training
│   ├── train_classifier.py     # SVM & XGBoost training + evaluation
│   └── app.py                  # Streamlit web app
├── requirements.txt
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain from scratch
cd src
python merge_data.py
python train_word2vec.py
python train_classifier.py

# Launch the web app
cd src
streamlit run app.py
```

## Results

| Model   | Macro F1 | Accuracy |
|---------|----------|----------|
| SVM     | 0.604    | 60.0%    |
| XGBoost | **0.640**| **64.0%**|

### Per-Class Performance (XGBoost)

| Class    | Precision | Recall | F1   |
|----------|-----------|--------|------|
| Negative | 0.67      | 0.62   | 0.64 |
| Neutral  | 0.58      | 0.60   | 0.59 |
| Positive | 0.68      | 0.69   | 0.69 |

## Tech Stack

- **Language:** Python 3
- **Embeddings:** gensim (Word2Vec)
- **Classifiers:** scikit-learn (SVM), XGBoost
- **Web App:** Streamlit
- **Evaluation:** sklearn.metrics (F1, confusion matrix)

## Key Challenges Addressed

- **Spelling normalization** — mapped common variants (nhi/nai/nahin → nahi)
- **Code-mixing** — custom embeddings trained on actual Hinglish, not monolingual corpora
- **No pretrained models** — Word2Vec trained from scratch on collected corpus

## Author

Ayush Kundu
