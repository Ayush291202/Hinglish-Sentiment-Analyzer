"""Hinglish text preprocessing: cleaning, normalization, tokenization."""

import re

# Spelling normalization map for common Hinglish variants
NORMALIZE_MAP = {
    "nhi": "nahi", "nahin": "nahi", "nai": "nahi", "ni": "nahi",
    "hai": "hai", "h": "hai", "he": "hai",
    "thi": "tha", "tha": "tha",
    "kya": "kya", "kia": "kya", "kiya": "kya",
    "bahut": "bahut", "bohot": "bahut", "boht": "bahut", "bhot": "bahut",
    "acha": "acha", "accha": "acha", "achha": "acha", "achchha": "acha",
    "mast": "mast", "mazaa": "mast",
    "bakwas": "bakwas", "bakwaas": "bakwas", "bakwass": "bakwas",
    "sahi": "sahi", "shi": "sahi",
    "bhai": "bhai", "bro": "bhai",
    "yaar": "yaar", "yar": "yaar",
    "theek": "theek", "thik": "theek", "theik": "theek",
    "haan": "haan", "han": "haan", "haa": "haan",
    "arre": "arre", "are": "arre", "arey": "arre",
    "kuch": "kuch", "kuchh": "kuch", "kuj": "kuch",
    "bilkul": "bilkul", "bilkl": "bilkul",
    "pasand": "pasand", "psand": "pasand",
    "zyada": "zyada", "jyada": "zyada", "jada": "zyada",
    "bekar": "bekar", "bekaar": "bekar",
    "ghatiya": "ghatiya", "ghatia": "ghatiya",
    "wahiyat": "wahiyat", "wahiyaat": "wahiyat",
    "shandar": "shandar", "shandaar": "shandar",
    "lajawab": "lajawab", "lajawaab": "lajawab",
    "kamaal": "kamaal", "kamal": "kamaal",
    "paisa": "paisa", "paise": "paisa", "pese": "paisa",
    "acchaa": "acha", "sach": "sach", "such": "sach",
}

# Hinglish stopwords (low-signal words)
STOPWORDS = {
    "ka", "ki", "ke", "ko", "se", "me", "men", "par", "pe",
    "ne", "jo", "ye", "wo", "is", "us", "ek", "aur", "ya",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "of", "in", "to", "for", "with", "on", "at", "by", "it",
    "and", "or", "but", "if", "so", "as", "that", "this",
}


def clean_text(text: str) -> str:
    """Clean raw Hinglish text."""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)  # mentions
    text = re.sub(r"#(\w+)", r"\1", text)  # hashtags → word
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation
    text = re.sub(r"\d+", "", text)  # numbers
    text = re.sub(r"\s+", " ", text).strip()  # whitespace
    return text


def normalize_tokens(tokens: list) -> list:
    """Apply spelling normalization."""
    return [NORMALIZE_MAP.get(t, t) for t in tokens]


def remove_stopwords(tokens: list) -> list:
    """Remove low-signal stopwords."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def preprocess(text: str, remove_stops: bool = True) -> list:
    """Full preprocessing pipeline: clean → tokenize → normalize → stopwords."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    tokens = normalize_tokens(tokens)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return tokens


def preprocess_to_string(text: str) -> str:
    """Return preprocessed text as a single string."""
    return " ".join(preprocess(text))
