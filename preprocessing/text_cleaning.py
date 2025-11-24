import re
import unicodedata
import nltk
from rapidfuzz import process, fuzz
import json
from pathlib import Path
import pandas as pd
from collections import Counter
import os

# Ensure NLTK tokenizer is available
nltk.download("punkt", quiet=True)

# -------------------------------
# Load Ayurveda canonical terms
# -------------------------------
with open("resources/ayurveda_terms.json", "r", encoding="utf-8") as f:
    AYUR_TERMS = json.load(f)

CANONICAL_NAMES = list(AYUR_TERMS.keys())

def load_sanskrit_vocab():
    csv_path = Path("sanskrit_terms_collection/final_language_dataset.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        sanskrit = set(df[df['label'] == 1]['ASCII'].str.lower())
        print(f"✓ Loaded {len(sanskrit)} Sanskrit terms for entity extraction")
        return sanskrit
    return set()

SANSKRIT_VOCAB = load_sanskrit_vocab()

# -------------------------------
# Detect repeated lines (headers/footers)
# -------------------------------
def detect_repeated_lines(all_pages: list[list[str]], min_repeat=0.3):
    """Detect repeated lines (likely headers/footers)."""
    line_counts = Counter()
    total_pages = len(all_pages)
    for page in all_pages:
        unique_lines = set(l.strip().lower() for l in page if l.strip())
        for line in unique_lines:
            line_counts[line] += 1
    return {
        line for line, count in line_counts.items()
        if count / total_pages >= min_repeat
    }

# -------------------------------
# Devanagari & Gibberish Removal
# -------------------------------
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]+")

def remove_devanagari_and_gibberish(lines: list[str]) -> list[str]:
    """Remove true Devanagari script and pseudo-Devanagari gibberish."""
    cleaned = []
    for line in lines:
        # Strip real Devanagari characters
        line = DEVANAGARI_PATTERN.sub(" ", line)

        # If still noisy gibberish, drop it
        symbol_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,;:'\"!?()-]", line)) / max(1, len(line))
        word_ratio = len(re.findall(r"[a-zA-Z]", line)) / max(1, len(line))

        if symbol_ratio > 0.25 or word_ratio < 0.4:
            continue  # skip gibberish
        cleaned.append(line)
    return cleaned


def remove_non_english_garbage(lines: list[str]) -> list[str]:
    """Remove lines that are not proper English (likely OCR of Devanagari/footnotes)."""
    cleaned = []
    for line in lines:
        # Remove if mostly symbols
        symbol_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,;:'\"!?()-]", line)) / max(1, len(line))
        alpha_ratio = len(re.findall(r"[a-zA-Z]", line)) / max(1, len(line))

        # Drop if more than 30% symbols OR less than 40% alphabetic
        if symbol_ratio > 0.3 or alpha_ratio < 0.4:
            continue

        cleaned.append(line)
    return cleaned



def remove_headers_footers(lines: list[str], repeated_lines=set()) -> list[str]:
    """Remove headers/footers but keep Ayurveda terms if present."""
    cleaned = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if l.lower() in repeated_lines:
            if any(term.lower() in l.lower() for term in CANONICAL_NAMES):
                cleaned.append(line)
            continue
        cleaned.append(line)
    return cleaned

# -------------------------------
# OCR Noise Filtering
# -------------------------------
def is_noise(line: str) -> bool:
    """Detect if a line is mostly OCR garbage but avoid killing valid English."""
    words = re.findall(r"[a-zA-Z]+", line)
    if len(words) >= 3:
        return False  # clearly English

    clean_chars = re.sub(r"[^a-zA-Z0-9]", "", line)
    if not clean_chars:
        return True

    ratio = len(clean_chars) / max(len(line), 1)
    return ratio < 0.25  # less aggressive than before
# -------------------------------
# Spelling & Semantic Normalization
# -------------------------------
def normalize_spelling(word: str, raw_text: str = None) -> str:
    """Correct spelling using Ayurveda canonical list and fuzzy matching."""
    word_clean = re.sub(r"[^a-zA-Z]", "", word)
    if not word_clean:
        return word

    # Exact or variant match
    for canonical, variants in AYUR_TERMS.items():
        if word_clean in variants or word_clean.lower() == canonical.lower():
            return canonical

    # Fuzzy match with safeguard
    match = process.extractOne(word_clean, CANONICAL_NAMES, scorer=fuzz.ratio)
    if match and match[1] >= 90:
        candidate = match[0]
        if raw_text and candidate.lower() not in raw_text.lower():
            return word  # avoid phantom terms
        return candidate

    return word

def semantic_normalization(text: str, raw_text: str = None) -> str:
    """Apply spelling normalization to each word in text."""
    words = text.split()
    return " ".join([normalize_spelling(w, raw_text) for w in words])

# -------------------------------
# Paragraph & Sentence Utilities
# -------------------------------
def to_paragraphs(sentences, n=3):
    """Group every n sentences into a paragraph."""
    return "\n\n".join(
        " ".join(sentences[i:i+n]) for i in range(0, len(sentences), n)
    )

# -------------------------------
# Main Text Cleaning
# -------------------------------
def clean_text(text: str, raw_text: str = None) -> str:
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Fix ligatures
    ligature_map = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
        "ﬅ": "ft", "ﬆ": "st"
    }
    for k, v in ligature_map.items():
        text = text.replace(k, v)

    # Remove intrusive special symbols inside words
    text = re.sub(r"(?<=\w)[\^<>\*~¬¦§¶]+(?=\w)", "", text)

    # Fix spaced letters ("s u s h r u t a" → "sushruta")
    text = re.sub(r"(?:\b\w\s){2,}", lambda m: m.group(0).replace(" ", ""), text)

    # Remove hyphenations across lines
    text = re.sub(r"-\s*\n\s*", "", text)

    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Sentence tokenization
    sentences = nltk.sent_tokenize(text)

    # Semantic normalization
    sentences = [semantic_normalization(s, raw_text) for s in sentences]

    # Remove duplicate consecutive words
    sentences = [re.sub(r"\b(\w+)(\s+\1){1,}\b", r"\1", s, flags=re.IGNORECASE) for s in sentences]

    # Capitalize sentences
    sentences = [s[0].upper() + s[1:] if len(s) > 1 else s for s in sentences]

    return to_paragraphs(sentences, n=3) if sentences else text

# -------------------------------
# Extract Ayurveda entities
# -------------------------------
def extract_ayurveda_entities(text: str) -> list:
    entities_found = set()
    text_lower = text.lower()
    for canonical in CANONICAL_NAMES:
        if canonical.lower() in text_lower:
            entities_found.add(canonical)
    return list(entities_found)

# -------------------------------
# Wrapper with Debug Support
# -------------------------------
def postprocess_text(text: str, repeated_lines=set(), raw_text: str = None, page_num=None, pdf_name=None) -> str:
    if not text.strip():
        return ""

    # Split into lines
    lines = text.split("\n")

    # Step 1: Remove headers/footers
    cleaned_lines = remove_headers_footers(lines, repeated_lines)

    # Step 2: Remove OCR garbage
    cleaned_lines = remove_devanagari_and_gibberish(cleaned_lines)
    cleaned_lines = remove_non_english_garbage(cleaned_lines)
    cleaned_lines = [l for l in cleaned_lines if not is_noise(l)]

    # Step 3: Join back
    text_cleaned = " ".join(cleaned_lines)

    # Step 4: Apply cleaning
    cleaned_text = clean_text(text_cleaned, raw_text=raw_text)


    return cleaned_text
