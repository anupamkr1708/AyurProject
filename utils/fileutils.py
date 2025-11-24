import glob
import os

def get_all_pdfs(directory: str):
    """Recursively collect all PDFs in a directory."""
    return glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)

def save_jsonl(docs, output_file: str):
    """Save documents to JSONL file for RAG ingestion."""
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
