"""Train SVM and XGBoost classifiers on Avg-Word2Vec sentence vectors."""

import numpy as np
import pandas as pd
import pickle
import os
from gensim.models import Word2Vec
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from preprocess import preprocess

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "merged.csv")
W2V_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "word2vec.model")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def sentence_vector(tokens, w2v_model, dim):
    """Compute average Word2Vec vector for a token list."""
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Load data and model
    df = pd.read_csv(DATA_PATH)
    w2v = Word2Vec.load(W2V_PATH)
    dim = w2v.wv.vector_size
    print(f"Data: {len(df)} rows, W2V dim: {dim}")

    # Build feature matrix
    tokens_list = [preprocess(str(t)) for t in df["text"]]
    X = np.array([sentence_vector(tok, w2v, dim) for tok in tokens_list])
    y = df["label"].values

    # Stratified train/test split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # --- SVM ---
    print("\n" + "=" * 50)
    print("Training SVM (linear)...")
    svm_base = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    svm = CalibratedClassifierCV(svm_base, cv=3)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=list(LABEL_NAMES.values())))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))

    # --- XGBoost ---
    print("\n" + "=" * 50)
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42,
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=list(LABEL_NAMES.values())))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))

    # Compare on test set
    from sklearn.metrics import f1_score
    svm_f1 = f1_score(y_test, y_pred_svm, average="macro")
    xgb_f1 = f1_score(y_test, y_pred_xgb, average="macro")
    print(f"\nSVM     Test F1 (macro): {svm_f1:.4f}")
    print(f"XGBoost Test F1 (macro): {xgb_f1:.4f}")

    # Save best model
    best_name = "svm" if svm_f1 >= xgb_f1 else "xgboost"
    best_model = svm if best_name == "svm" else xgb
    print(f"\nBest model: {best_name}")

    # Retrain best on full data
    best_model.fit(X, y)
    model_path = os.path.join(MODEL_DIR, "classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "name": best_name}, f)
    print(f"Saved to {model_path}")


if __name__ == "__main__":
    main()
