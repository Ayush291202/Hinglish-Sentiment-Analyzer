"""Merge all uploaded datasets into one unified CSV."""

import pandas as pd
import os

UPLOAD = os.path.join(os.path.dirname(__file__), "..", "raw_data")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "merged.csv")

LABEL_MAP_TEXT = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def load_semeval_tsv(path):
    """Load SemEval-style TSV: id\ttext\tlabel"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                rows.append({"id": parts[0], "text": parts[1], "label": int(parts[2])})
            elif len(parts) == 2:
                rows.append({"id": parts[0], "text": parts[1], "label": None})
    return pd.DataFrame(rows)


def load_test_with_labels():
    """Load FinalTest + Ty.txt to get labeled test data."""
    test = load_semeval_tsv(os.path.join(UPLOAD, "FinalTest.tsv"))
    labels = pd.read_csv(os.path.join(UPLOAD, "Ty.txt"))
    labels.columns = [c.strip() for c in labels.columns]
    labels["id"] = labels["Uid"].astype(str)
    labels["label"] = labels["Sentiment"].str.strip().str.lower().map(LABEL_MAP_TEXT)
    test["id"] = test["id"].astype(str)
    merged = test.merge(labels[["id", "label"]], on="id", suffixes=("_drop", ""))
    merged = merged[["text", "label"]].dropna(subset=["label"])
    merged["label"] = merged["label"].astype(int)
    return merged


def load_output_csv():
    """Load output.csv with columns: _, text, sentiment, label"""
    df = pd.read_csv(os.path.join(UPLOAD, "output.csv"))
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].str.strip()
    return df


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames = []

    # SemEval train
    train = load_semeval_tsv(os.path.join(UPLOAD, "FinalTrainingOnly.tsv"))
    train = train[["text", "label"]].dropna(subset=["label"])
    train["label"] = train["label"].astype(int)
    print(f"Train: {len(train)}")
    frames.append(train)

    # SemEval validation
    val = load_semeval_tsv(os.path.join(UPLOAD, "ValidationOnly.tsv"))
    val = val[["text", "label"]].dropna(subset=["label"])
    val["label"] = val["label"].astype(int)
    print(f"Validation: {len(val)}")
    frames.append(val)

    # Test + labels
    test = load_test_with_labels()
    print(f"Test (labeled): {len(test)}")
    frames.append(test)

    # output.csv
    out = load_output_csv()
    print(f"output.csv: {len(out)}")
    frames.append(out)

    # Merge and deduplicate
    combined = pd.concat(frames, ignore_index=True)
    combined["text_clean"] = combined["text"].str.strip().str.lower()
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text_clean"]).drop(columns=["text_clean"])
    print(f"\nTotal before dedup: {before}")
    print(f"Total after dedup:  {len(combined)}")
    print(f"\nLabel distribution:")
    for lbl, name in LABEL_NAMES.items():
        count = (combined["label"] == lbl).sum()
        print(f"  {name}: {count}")

    combined.to_csv(OUT, index=False)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
