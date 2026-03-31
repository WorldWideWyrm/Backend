import fitz
import json
from pathlib import Path

pdf_path = Path("pdfs/Player's Handbook [2024].pdf")
doc = fitz.open(pdf_path)

all_pages = []

for i, page in enumerate(doc):
    text = page.get_text("text")
    all_pages.append({
        "page_number": i + 1,
        "text": text
    })

out_path = Path("pdfs/raw_pages.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    json.dump(all_pages, f, ensure_ascii=False, indent=2)