import fitz
import csv
import re

INPUT_PDF = r"G:\Ayurproject\Whosanskritterms.pdf"
OUTPUT_CSV = r"sanskrit_terms_iast_ascii.csv"

# Custom IAST → ASCII replacements
REPLACEMENTS = {
    "ā": "aa", "ī": "ii", "ū": "uu",
    "ṛ": "ri", "ṝ": "rri", "ḷ": "li", "ḹ": "lli",
    "ṅ": "n", "ñ": "n", "ṭ": "t", "ḍ": "d", "ṇ": "n",
    "ś": "sh", "ṣ": "sh",
    "ṃ": "m", "ḥ": "h",
}

def to_ascii(iast: str) -> str:
    """Convert IAST diacritics → ASCII-friendly form."""
    s = iast
    for k, v in REPLACEMENTS.items():
        s = s.replace(k, v)
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()

def normalize_iast(term: str) -> str:
    """Clean up Sanskrit IAST terms (remove hyphens, normalize underscores)."""
    s = term.strip()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def extract_terms_from_pages(pdf_path, start_page, end_page):
    """Extract only IAST + ASCII terms from WHO tables."""
    results = []
    with fitz.open(pdf_path) as doc:
        for pno in range(start_page - 1, end_page):  # 1-based to 0-based
            page = doc[pno]
            tables = page.find_tables()
            if not tables:
                continue

            for table in tables:
                for row in table.extract():  # each row is a list of cells
                    if len(row) < 1:
                        continue
                    iast_term = row[0]

                    if not iast_term or "Sanskrit" in iast_term:
                        continue  # skip headers / empty

                    norm = normalize_iast(iast_term)
                    ascii_ = to_ascii(norm)

                    results.append((norm, ascii_))

    return results

def main():
    terms = extract_terms_from_pages(INPUT_PDF, 499, 606)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["IAST_normalized", "ASCII"])
        for row in terms:
            writer.writerow(row)

    print(f" Extracted {len(terms)} Sanskrit IAST terms (499–606) → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
