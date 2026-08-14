# Hinglish Sentiment Analyzer
An **interview-ready NLP pipeline** for sentiment analysis on **code-mixed Hindi-English (Hinglish)** text using rigorous data validation, two-phase TF-IDF feature extraction, and Logistic Regression classification.

## Problem Statement
Given a Hinglish text input, predict its sentiment polarity: **Positive**, **Negative**, or **Neutral**.
```
Input:  "Yeh product bilkul bakwaas hai, waste of money"
Output: NEGATIVE (confidence: 89%)

Input:  "Bhai kal ka match toh ekdum mast tha!"
Output: POSITIVE (confidence: 85%)
```

## Pipeline Architecture
```
Raw Text → Overlap Audit → Preprocessing → Phase 1 TF-IDF → Model Selection (Validation)
         ↓
    Remove Overlaps (Train/Val only; Test protected)
                                  ↓
                        Phase 2 TF-IDF → Final Training (Train+Val) → Sentiment Output
                                           ↓
                                    Test Evaluation (Once, Held-Out)
```

**Key Stages:**
1. **Data Validation** — Multi-level overlap detection (raw + normalized text); 1,577 overlapping samples removed
2. **Preprocessing** — Lowercasing, URL/mention removal, repeated character normalization, punctuation removal, stopword filtering with negation preservation, spelling normalization
3. **Phase 1 TF-IDF** — Fit only on training data (13,017 samples) for unbiased model selection
4. **Model Selection** — Compare 4 candidates on validation set; select Logistic Regression (C=10)
5. **Phase 2 TF-IDF** — Refit on train+validation combined (16,017 samples) to maximize feature quality
6. **Final Training** — Retrain selected model on Phase 2 features
7. **Test Evaluation** — Evaluate once on held-out test set (3,000 samples)

## Dataset
**SemEval-2020 Task 9 Hinglish Sentiment Dataset** — cleaned and validated:

| Split | Original | After Cleaning | Description |
|-------|----------|----------------|-------------|
| Train | 14,594 | 13,017 | Removed 1,577 overlapping samples |
| Validation | 3,000 | 3,000 | Unchanged |
| Test | 3,000 | 3,000 | Protected; never modified |
| **Total** | **20,594** | **19,017** | Rigorous data validation applied |

**Label Distribution (Final):**
| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Negative | 4,215 (35.0%) | 900 (30.0%) | - | 5,115 |
| Neutral | 3,824 (31.8%) | 1,100 (36.7%) | - | 4,924 |
| Positive | 4,000 (33.2%) | 1,000 (33.3%) | - | 5,000 |

## Project Structure
```
hinglish-sentiment-analyzer/
├── notebooks/
│   └── Hinglish_Sentiment_Analyzer.ipynb    # Complete pipeline (Google Colab)
├── data/
│   ├── FinalTrainingOnly.tsv                # Training set
│   ├── ValidationOnly.tsv                   # Validation set
│   ├── FinalTest.tsv                        # Test set
│   └── Ty.txt                               # Test labels
├── models/
│   └── final_pipeline.pkl                   # Serialized artifacts
├── reports/
│   ├── Hinglish_Report_v2.pdf              # Complete technical report
│   └── confusion_matrix.pdf                 # Test set confusion matrix
└── README.md
```

## How to Run

### **Quick Start (Google Colab)**
1. Open `Hinglish_Sentiment_Analyzer.ipynb` in Google Colab
2. Upload 5 data files: `FinalTrainingOnly.tsv`, `ValidationOnly.tsv`, `FinalTest.tsv`, `Ty.txt`
3. Run all cells sequentially
4. Model trains and saves to `/models/final_pipeline.pkl`

### **Local Setup**
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run notebook
jupyter notebook Hinglish_Sentiment_Analyzer.ipynb
```

### **Inference (Using Saved Model)**
```python
import pickle
import re

# Load model
with open('models/final_pipeline.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
tfidf = artifacts['tfidf']
config = artifacts['preprocessing_config']
labels = artifacts['label_map']

# Preprocess text
def preprocess(text, config):
    text = text.lower().strip()
    text = re.sub(r'http\S+|@\w+|#', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [config['normalization'].get(t, t) for t in tokens]
    stopwords = config['stopwords']
    negation = config['negation_words']
    tokens = [t for t in tokens if t not in stopwords or t in negation]
    return ' '.join([t for t in tokens if len(t) > 1])

# Predict
text = "Yeh product bilkul bakwaas hai!"
processed = preprocess(text, config)
X = tfidf.transform([processed])
label_idx = model.predict(X)[0]
proba = model.predict_proba(X)[0]
confidence = proba[label_idx] * 100

print(f"Sentiment: {labels[label_idx].upper()} ({confidence:.1f}%)")
```

## Results

### **Overall Performance (Test Set)**
| Metric | Value |
|--------|-------|
| **Macro F1-Score** | **0.6257** |
| **Accuracy** | 62.27% |
| Test Samples | 3,000 (held-out) |

### **Model Selection (Validation)**
| Model | Validation F1 | Selected? |
|-------|---------------|-----------|
| LR (C=0.1) | 0.6144 | ✗ |
| LR (C=1.0) | 0.6657 | ✗ |
| **LR (C=10)** | **0.6717** | **✓** |
| SVM (C=1.0, calibrated) | 0.6649 | ✗ |

### **Per-Class Performance (Test Set)**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.6486 | 0.6644 | 0.6564 | 900 |
| Neutral | 0.5465 | 0.5236 | 0.5348 | 1,100 |
| Positive | 0.6777 | 0.6940 | 0.6858 | 1,000 |
| **Weighted Avg** | **0.6209** | **0.6227** | **0.6216** | 3,000 |

## Key Technical Contributions

### **1. Rigorous Data Validation**
- Multi-level overlap auditing: raw-text and normalized-text levels
- Detected 1,451 unique overlapping samples; removed 1,577 training observations (10.8% of data)
- Diagnostic reporting of test-set overlaps (92 normalized overlaps with train, 38 with val) without modification
- Complete data leakage prevention while maintaining honest metrics

### **2. Two-Phase TF-IDF Architecture**
- **Phase 1:** Fit TF-IDF on training data only (13,017 samples) for unbiased model selection
- **Phase 2:** Refit TF-IDF on train+validation combined (16,017 samples) for maximum feature quality
- Prevents information leakage while optimizing final model performance

### **3. Validation-Based Model Selection**
- Four classifiers compared on validation set only (never test)
- Best model frozen before test evaluation
- Single test evaluation prevents overfitting

### **4. Reproducible Preprocessing**
- Exact configuration saved: stopwords, negation words, normalization rules
- SentimentPredictor class ensures identical preprocessing for inference
- Complete artifact serialization for production use

## Preprocessing Configuration
```python
PREPROCESSING_CONFIG = {
    'version': '1.0',
    'normalization': {'nhi': 'nahi', 'h': 'hai', 'kia': 'kya'},
    'stopwords': {'ka', 'ki', 'ke', 'ko', 'se', 'me', 'the', 'a', 'an', 'is'},
    'negation_words': {'nahi', 'no', 'not', 'never'}
}
```

## Tech Stack
- **Language:** Python 3.8+
- **Data Processing:** Pandas, NumPy
- **Feature Extraction:** scikit-learn (TfidfVectorizer)
- **Classification:** scikit-learn (LogisticRegression, LinearSVC)
- **Evaluation:** sklearn.metrics (F1, accuracy, confusion matrix)
- **Visualization:** Matplotlib, Seaborn
- **Development:** Google Colab (Jupyter Notebook)
- **Serialization:** Python pickle

## Limitations & Future Work

### **Current Limitations**
- TF-IDF lacks semantic information (no word relationships)
- Sarcasm detection is poor (surface-level features only)
- Out-of-vocabulary words are ignored
- Neutral class inherently ambiguous
- Basic negation preservation (scope not modeled)
- Romanized-only (Devanagari script not supported)
- No domain adaptation

### **Future Improvements**
- Subword embeddings (FastText) for spelling variants and OOV words
- Sequence models (BiLSTM, XLM-RoBERTa) for word order and dependencies
- Sarcasm-specific features and datasets
- Explicit negation tokens for scope modeling
- Class rebalancing (SMOTE) for minority Neutral class
- Devanagari script support
- Domain-specific fine-tuning

## How to Cite
If you use this pipeline, please cite:
```
Kundu, A. (2026). Hinglish Sentiment Analyzer: An Interview-Ready NLP Pipeline for Code-Mixed Hindi-English Text.
```

## Author
**Ayush Kundu**  
Indian Institute of Technology Kanpur  
Email: ayushkundu25@iitk.ac.in  
GitHub: [Ayush291202/Hinglish-Sentiment-Analyzer](https://github.com/Ayush291202/Hinglish-Sentiment-Analyzer)

---

**Last Updated:** August 14, 2026  
**Status:** Interview-Ready | Data Leakage-Free | Reproducible
