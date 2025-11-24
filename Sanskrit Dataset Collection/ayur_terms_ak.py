import requests
from bs4 import BeautifulSoup
import re
import csv

GLOSSARY_URL = "https://www.ayurvedakendra.in/discover-ayurveda/ayurvedic-glossary/"
OUTPUT_CSV = r"G:\Ayurproject\sanskrit_terms_collection\ayurveda_terms_from_ak.csv"

def normalize_term(term: str) -> str:
    """Normalize term by replacing spaces/hyphens with underscores and lowering case."""
    term = term.strip()
    term = term.replace("-", "_").replace(" ", "_")
    term = re.sub(r"_+", "_", term)  # collapse multiple underscores
    return term.lower()

def scrape_ayurveda_glossary():
    resp = requests.get(GLOSSARY_URL)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    terms = set()  # use set to avoid duplicates

    for p in soup.find_all("p"):
        text = p.get_text().strip()
        if not text:
            continue

        # Match "Term - Description"
        m = re.match(r"^([A-Za-z \-()]+?)\s*[-–]\s*(.*)$", text)
        if m:
            term = m.group(1).strip()

            # Clean main term
            term_clean = re.sub(r"\(.*?\)", "", term).strip()
            terms.add(normalize_term(term_clean))

            # Add variant inside parentheses (if present)
            variant_match = re.search(r"\((.*?)\)", term)
            if variant_match:
                variant = variant_match.group(1).strip()
                terms.add(normalize_term(variant))

    return sorted(terms)

def main():
    terms = scrape_ayurveda_glossary()
    print(f" Extracted {len(terms)} unique normalized terms.")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sanskrit_ayurvedic_terms"])
        for term in terms:
            writer.writerow([term])

    print(f" Saved → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
