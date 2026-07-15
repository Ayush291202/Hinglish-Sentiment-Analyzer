"""Train Word2Vec embeddings on the Hinglish corpus."""

import pandas as pd
from gensim.models import Word2Vec
from preprocess import preprocess
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "merged.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "word2vec.model")

VECTOR_SIZE = 150
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 30
SG = 1  # skip-gram


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} sentences")

    # Tokenize all sentences (keep stopwords for W2V context)
    sentences = [preprocess(str(text), remove_stops=False) for text in df["text"]]
    sentences = [s for s in sentences if len(s) > 0]
    print(f"Tokenized {len(sentences)} non-empty sentences")

    vocab_tokens = set()
    for s in sentences:
        vocab_tokens.update(s)
    print(f"Unique tokens: {len(vocab_tokens)}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=4,
        epochs=EPOCHS,
    )

    model.save(MODEL_PATH)
    print(f"\nWord2Vec model saved to {MODEL_PATH}")
    print(f"Vocabulary size: {len(model.wv)}")

    # Sanity checks
    test_words = ["acha", "bakwas", "mast", "bekar", "shandar", "ghatiya"]
    for w in test_words:
        if w in model.wv:
            similar = model.wv.most_similar(w, topn=5)
            print(f"\n'{w}' → {[s[0] for s in similar]}")
        else:
            print(f"\n'{w}' not in vocabulary")


if __name__ == "__main__":
    main()
