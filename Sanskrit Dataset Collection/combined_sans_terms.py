import pandas as pd

# ---------------- CSV Files ----------------
WHO_CSV = r"G:\Ayurproject\sanskrit_terms_collection\sanskrit_terms_iast_ascii_who.csv"        # column: ASCII
COLOGNE_CSV = r"G:\Ayurproject\sanskrit_terms_collection\sanskrit_words_iast_ascii_cologne.csv"   # column: ascii
KENDRA_CSV = r"G:\Ayurproject\sanskrit_terms_collection\ayurveda_terms_from_ak.csv"        # column: sanskrit_ayurvedic_terms
KRISHNA_CSV = r"G:\Ayurproject\sanskrit_terms_collection\sanskrit_glossary.csv"            # column: sanskrit_terms

OUTPUT_CSV = r"G:\Ayurproject\sanskrit_terms_collection\master_sanskrit_ascii_final.csv"

# ---------------- Function to load ASCII column ----------------
def load_ascii_column(file, column_name):
    df = pd.read_csv(file, dtype=str).fillna("")
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in {file}")
    # Normalize: replace spaces or hyphens with underscores, lowercase
    df["ASCII"] = df[column_name].str.strip().str.replace(r"\s+|-", "_", regex=True).str.lower()
    return df[["ASCII"]]

# ---------------- Main Merge ----------------
def main():
    dfs = []

    # Load ASCII terms from each CSV
    dfs.append(load_ascii_column(WHO_CSV, "ASCII"))
    dfs.append(load_ascii_column(COLOGNE_CSV, "ascii"))
    dfs.append(load_ascii_column(KENDRA_CSV, "sanskrit_ayurvedic_terms"))
    dfs.append(load_ascii_column(KRISHNA_CSV, "Sanskrit_Term"))

    # Concatenate all and drop duplicates
    master = pd.concat(dfs, ignore_index=True)
    master = master.drop_duplicates(subset=["ASCII"]).reset_index(drop=True)

    # Remove empty strings
    master = master[master["ASCII"] != ""]

    # Sort alphabetically
    master = master.sort_values(by="ASCII").reset_index(drop=True)

    # Add label column
    master["label"] = 1  # 1 for Sanskrit

    # Save final CSV
    master.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f" Merged {len(master)} unique ASCII Sanskrit terms with label → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
