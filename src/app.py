"""Streamlit app for Hinglish Sentiment Analysis."""

import streamlit as st
import numpy as np
import pickle
import os
from gensim.models import Word2Vec
from preprocess import preprocess

BASE = os.path.dirname(__file__)
W2V_PATH = os.path.join(BASE, "..", "models", "word2vec.model")
CLF_PATH = os.path.join(BASE, "..", "models", "classifier.pkl")

LABELS = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}

EXAMPLES = [
    "Yaar yeh movie bahut boring thi!",
    "Bhai kal ka match toh ekdum mast tha!",
    "Theek hi tha, kuch khaas nahi",
    "Yeh product bilkul bakwaas hai, waste of money",
    "Bohot accha experience tha, definitely recommend karunga",
    "Mujhe koi fark nahi padta",
]


@st.cache_resource
def load_models():
    w2v = Word2Vec.load(W2V_PATH)
    with open(CLF_PATH, "rb") as f:
        clf_data = pickle.load(f)
    return w2v, clf_data["model"], clf_data["name"]


def predict(text, w2v, clf):
    tokens = preprocess(text)
    dim = w2v.wv.vector_size
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vecs:
        return 1, [0.0, 1.0, 0.0]  # default neutral
    vec = np.mean(vecs, axis=0).reshape(1, -1)
    label = clf.predict(vec)[0]
    proba = clf.predict_proba(vec)[0]
    return label, proba


def main():
    st.set_page_config(page_title="Hinglish Sentiment Analyzer", page_icon="🇮🇳", layout="centered")
    st.title("🇮🇳 Hinglish Sentiment Analyzer")
    st.caption("Analyze sentiment of code-mixed Hindi-English text")

    w2v, clf, clf_name = load_models()
    st.sidebar.markdown(f"**Model:** {clf_name.upper()}")
    st.sidebar.markdown(f"**Embedding dim:** {w2v.wv.vector_size}")
    st.sidebar.markdown(f"**Vocabulary:** {len(w2v.wv):,} words")

    text = st.text_area("Enter Hinglish text:", height=100, placeholder="Type or paste Hinglish text here...")

    st.markdown("**Try an example:**")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex[:30] + "...", key=f"ex_{i}"):
            text = ex

    if text and text.strip():
        label, proba = predict(text, w2v, clf)
        confidence = proba[label] * 100

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Sentiment", f"{COLORS[label]} {LABELS[label]}")
        col2.metric("Confidence", f"{confidence:.1f}%")

        st.markdown("**Probability Distribution:**")
        for lbl_id in [2, 1, 0]:
            st.progress(proba[lbl_id], text=f"{LABELS[lbl_id]}: {proba[lbl_id]*100:.1f}%")

        with st.expander("Preprocessing details"):
            tokens = preprocess(text)
            st.code(f"Tokens: {tokens}")
            oov = [t for t in tokens if t not in w2v.wv]
            st.write(f"In vocabulary: {len(tokens) - len(oov)}/{len(tokens)}")
            if oov:
                st.write(f"OOV words: {oov}")


if __name__ == "__main__":
    main()
