import fitz
import json
from pathlib import Path
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\n\d+\n", "\n", text)      # fjern sidetal
    text = re.sub(r"[ \t]+", " ", text)        # fix spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # for mange linjeskift
    return text.strip()

def pdfToText():
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


    cleaned_pages = []

    for page in all_pages:
        cleaned_pages.append({
            "page_number": page["page_number"],
            "text": clean_text(page["text"])
        })


    out_path = Path("pdfs/cleaned_pages.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_pages, f, ensure_ascii=False, indent=2)

class Spells:
    def __init__(self, chunk_id, title, level, school, classes, casting_time, range, components, duration, text, chapter, page, upcasting=0, ritual=0, concentration=0):
        self.chunk_id = chunk_id
        self.title = title
        self.level = level
        self.school = school
        self.classes = classes
        self.casting_time = casting_time
        self.range = range
        self.components = components
        self.duration = duration
        self.text = text
        self.chapter = chapter
        self.page = page

class Feats:
    pass

class Glossary:
    pass

class Rules:
    pass
