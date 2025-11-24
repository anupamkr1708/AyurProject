import requests
from bs4 import BeautifulSoup
import csv

# Base URL
BASE_URL = "https://www.swami-krishnananda.org/glossary/"

# Sections from the site
sections = [
    "glossary_a.html", "glossary_bc.html", "glossary_degh.html",
    "glossary_ijkl.html", "glossary_mn.html", "glossary_opr.html",
    "glossary_s.html", "glossary_tuvy.html"
]

# Function to clean and normalize terms
def normalize_term(term: str) -> str:
    term = term.strip()
    # Remove trailing colon
    if term.endswith(":"):
        term = term[:-1]
    # Normalize hyphens -> underscores
    term = term.replace("-", "_")
    return term

# Collect all terms
all_terms = []

for section in sections:
    url = BASE_URL + section
    print(f"Scraping: {url}")
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Failed to fetch {url}")
        continue
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Find all <strong> tags inside <p>
    strong_tags = soup.find_all("strong")
    for tag in strong_tags:
        term = tag.get_text(strip=True)
        term = normalize_term(term)
        if term:  # Ensure not empty
            all_terms.append(term)

# Save to CSV
with open("sanskrit_glossary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Sanskrit_Term"])
    for term in sorted(set(all_terms)):
        writer.writerow([term])

print(f" Saved {len(set(all_terms))} unique Sanskrit terms to sanskrit_glossary.csv")
