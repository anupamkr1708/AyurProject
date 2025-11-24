import os
from utils.fileutils import get_all_pdfs, save_jsonl
from utils.pdfutils import is_scanned_pdf
from doc_loaders.textloader import load_text_pdf
from doc_loaders.scanned_loader import load_scanned_pdf
from preprocessing.text_cleaning import (
    postprocess_text,
    extract_ayurveda_entities,
    detect_repeated_lines
)

INPUT_DIR = r"G:\Ayurproject\datadir"
OUTPUT_FILE = "output/ayurveda_docs_final.jsonl"

def process_all_pdfs():
    all_pdfs = get_all_pdfs(INPUT_DIR)
    print(f"Found {len(all_pdfs)} PDFs.\n")

    docs = []

    for pdf_path in all_pdfs:
        print(f"Processing: {pdf_path}")

        # Load raw pages
        if is_scanned_pdf(pdf_path):
            raw_docs = load_scanned_pdf(pdf_path)
        else:
            raw_docs = load_text_pdf(pdf_path)

        # Split pages into lines to detect repeated headers/footers
        all_lines_per_page = [d["content"].split("\n") for d in raw_docs]
        repeated_lines = detect_repeated_lines(all_lines_per_page)

        # Process each page
        for d in raw_docs:
            metadata = d.get("metadata", {})
            page_num = metadata.get("page", "unknown")
            raw_text = d.get("content", "")

            # ✅ Pass raw_text, repeated_lines, page_num, pdf_name
            cleaned_text = postprocess_text(
                raw_text,
                repeated_lines=repeated_lines,
                raw_text=raw_text,
                page_num=page_num,
                pdf_name=pdf_path
            )

            if not cleaned_text.strip():
                print(f"⚠️ Empty after cleaning on page {page_num}, using raw text.\n")
                cleaned_text = raw_text

            # Extract Ayurveda entities
            entities = extract_ayurveda_entities(cleaned_text)

            # Append to final docs
            docs.append({
                "content": cleaned_text,
                "metadata": {
                    "source": pdf_path,
                    "page": page_num,
                    "doc_name": metadata.get("doc_name", ""),
                    "entities": entities
                }
            })

    print(f"\nTotal cleaned paragraphs extracted: {len(docs)}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    save_jsonl(docs, OUTPUT_FILE)
    print(f"Saved structured dataset → {OUTPUT_FILE}")


if __name__ == "__main__":
    process_all_pdfs()
