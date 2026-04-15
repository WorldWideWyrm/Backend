import json
import re
from pathlib import Path

INPUT_PATH = Path("pdfs/cleaned_pages.json")
OUTPUT_PATH = Path("pdfs/rules_chunks.json")

APPENDIX_START_PAGE = 359

NON_RULE_HEADINGS = {
    "OBJECT ARMOR CLASS",
    "OBJECT HIT POINTS",
    "CARRYING CAPACITY",
    "DAMAGE TYPES",
    "WATER NEEDS PER DAY",
    "FOOD NEEDS PER DAY",
    "INFLUENCE CHECKS",
    "SEARCH",
    "AREAS OF KNOWLEDGE",
}

SKIP_HEADINGS = {
    "GLOSSARY CONVENTIONS",
    "RULES DEFINITIONS",
}


def load_pages(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Filen blev ikke fundet: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON-filen skal indeholde en liste")

    return data


def is_heading(line: str) -> bool:
    line = line.strip()

    if not line:
        return False

    if line in {"~", "-", "•"}:
        return False

    if not re.search(r"[A-Za-z]", line):
        return False

    cleaned = re.sub(r"[\[\]\(\)\-\"'.,:;!/0-9]", "", line)
    cleaned = " ".join(cleaned.split())

    if not cleaned:
        return False

    letters_only = re.sub(r"[^A-Za-z ]", "", cleaned).strip()
    if not letters_only:
        return False

    return letters_only == letters_only.upper()


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def finalize_chunks(chunks: list[dict]) -> list[dict]:
    for chunk in chunks:
        title = chunk["metadata"]["title"]
        body = chunk["text"]
        chunk["text"] = f"{title}\n\n{body}"
    return chunks


def chunk_rules_from_pages(pages: list[dict]) -> list[dict]:
    pages = [p for p in pages if p.get("page_number", 0) >= APPENDIX_START_PAGE]

    chunks = []
    current = None
    seen_rules_definitions = False

    for page in pages:
        page_number = page["page_number"]
        raw_text = page["text"]

        lines = [normalize_line(line) for line in raw_text.split("\n")]
        lines = [line for line in lines if line]

        for line in lines:
            if not seen_rules_definitions:
                if line == "RULES DEFINITIONS":
                    seen_rules_definitions = True
                continue

            if line.upper().startswith("INDEX"):
                if current is not None:
                    current["text"] = current["text"].strip()
                    chunks.append(current)
                return finalize_chunks(chunks)

            if line in SKIP_HEADINGS:
                continue

            if is_heading(line) and line not in NON_RULE_HEADINGS:
                if current is not None:
                    current["text"] = current["text"].strip()
                    chunks.append(current)

                current = {
                    "id": f"rule_{len(chunks):04d}",
                    "text": "",
                    "metadata": {
                        "type": "rule",
                        "title": line,
                        "start_page": page_number,
                        "end_page": page_number,
                        "source": "PHB 2024 Rule Glossary"
                    }
                }
            else:
                if current is not None:
                    if current["text"]:
                        current["text"] += " " + line
                    else:
                        current["text"] = line
                    current["metadata"]["end_page"] = page_number

    if current is not None:
        current["text"] = current["text"].strip()
        chunks.append(current)

    return finalize_chunks(chunks)


def save_chunks(path: Path, chunks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main() -> None:
    pages = load_pages(INPUT_PATH)
    chunks = chunk_rules_from_pages(pages)
    save_chunks(OUTPUT_PATH, chunks)

    print(f"Antal rule chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5], 1):
        print(f"\n--- CHUNK {i} ---")
        print(chunk["metadata"]["title"])
        print(f"Sider: {chunk['metadata']['start_page']}–{chunk['metadata']['end_page']}")
        print(chunk["text"][:500])


if __name__ == "__main__":
    main()