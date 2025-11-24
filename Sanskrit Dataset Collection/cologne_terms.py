import os
import re
import csv
from indic_transliteration.sanscript import transliterate, SLP1, IAST

INPUT_FOLDER = r"G:\Ayurproject\Cologne_dict"
OUTPUT_CSV = r"G:\Ayurproject\sanskrit_terms_collection\sanskrit_words_iast_ascii_cologne.csv"

# Vowel presence ensures Sanskrit-like word
VOWEL_PATTERN = re.compile(r"[aAiIuUfFxXeEoO]")

# Illegal characters (non-Sanskrit diacritics)
ILLEGAL_CHARS = set("ḻṟ")

def normalize_word(word: str) -> str:
    """Clean Sanskrit tokens in SLP1 before transliteration"""
    word = word.strip()

    # remove hyphens/dashes
    word = word.replace("-", "").replace("—", "")

    # remove dictionary artifacts
    word = re.sub(r"srs+", "", word)

    # keep only letters
    word = re.sub(r"[^a-zA-Z]", "", word)

    # reject short junk
    if len(word) < 3:
        return ""

    # reject if no vowel
    if not VOWEL_PATTERN.search(word):
        return ""

    return word

def iast_to_ascii(word: str) -> str:
    """Convert IAST → phonetic ASCII (better than raw strip)."""
    replacements = {
        "ā": "aa", "ī": "ii", "ū": "uu",
        "ṛ": "ri", "ṝ": "rri", "ḷ": "li", "ḹ": "lli",
        "ṅ": "n", "ñ": "n", "ṭ": "t", "ḍ": "d", "ṇ": "n",
        "ś": "sh", "ṣ": "sh",
        "ṃ": "m", "ḥ": "h",
    }
    for src, tgt in replacements.items():
        word = word.replace(src, tgt)
    return word

def extract_sanskrit_from_line(line: str):
    """Extract Sanskrit terms from <s>...</s> and <k1>...</k1>"""
    results = []

    for tag in ["s", "k1"]:
        matches = re.findall(rf"<{tag}>(.*?)</{tag}>", line)
        for m in matches:
            term = normalize_word(m)
            if term:
                try:
                    # Convert SLP1 → IAST
                    iast_term = transliterate(term, SLP1, IAST)

                    # reject words with illegal chars
                    if any(ch in iast_term for ch in ILLEGAL_CHARS):
                        continue

                    # Convert IAST → ASCII fallback
                    ascii_term = iast_to_ascii(iast_term)

                    results.append((iast_term, ascii_term))

                except Exception:
                    continue

    return results

def main():
    all_words = set()

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(INPUT_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    sanskrit_terms = extract_sanskrit_from_line(line)
                    all_words.update(sanskrit_terms)

    # Write to CSV with 2 columns: IAST + ASCII
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["iast", "ascii"])  # header
        for iast, ascii_ in sorted(all_words):
            writer.writerow([iast, ascii_])

    print(f" Extracted {len(all_words)} Sanskrit words (IAST + ASCII) → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
