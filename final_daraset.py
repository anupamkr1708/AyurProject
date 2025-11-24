import pandas as pd

# ---------------- Paths ----------------
ENGLISH_TXT = r"G:\Ayurproject\words_alpha.txt"
SANSKRIT_CSV = r"G:\Ayurproject\sanskrit_terms_collection\master_sanskrit_ascii_final.csv"
OUTPUT_CSV = r"G:\Ayurproject\sanskrit_terms_collection\final_language_dataset.csv"

# ---------------- Load English words ----------------
def load_english_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        words = [line.strip().lower().replace(" ", "_") for line in f if line.strip()]
    df = pd.DataFrame(words, columns=["ASCII"])
    df["label"] = 0  # 0 for English
    return df

# ---------------- Main Merge ----------------
def main():
    # Load English
    english_df = load_english_words(ENGLISH_TXT)
    print(f" Loaded {len(english_df)} English words")

    # Load Sanskrit dataset
    sanskrit_df = pd.read_csv(SANSKRIT_CSV, dtype=str)
    print(f" Loaded {len(sanskrit_df)} Sanskrit words")

    # Merge datasets
    combined = pd.concat([sanskrit_df, english_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ASCII"]).reset_index(drop=True)

    # Shuffle dataset (optional for training)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save final dataset
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f" Final dataset ready with {len(combined)} words → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
