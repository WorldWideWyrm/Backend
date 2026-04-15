import json
from pathlib import Path


INPUT_PATH = Path("pdfs/cleaned_pages.json")
OUTPUT_PATH = Path("pdfs/rule_glossary.txt")

APPENDIX_START_PAGE = 360
START_IDX = APPENDIX_START_PAGE - 1


def load_pages(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON-file needs to contain a list of pages.")

    for i, page in enumerate(data):
        if not isinstance(page, dict):
            raise ValueError(f"Page {i} is not a dictionary/object.")
        if "text" not in page:
            raise ValueError(f"Page {i} is missing: 'text'")

    return data


def find_real_start_index(pages: list, fallback_idx: int) -> int:
    search_start = max(0, fallback_idx - 3)
    search_end = min(len(pages), fallback_idx + 5)

    for i in range(search_start, search_end):
        text = pages[i]["text"].lower()
        if "rule glossary" in text or "appendix c" in text:
            return i

    return fallback_idx


def find_end_index(pages: list, start_idx: int) -> int | None:
    for i in range(start_idx + 1, len(pages)):
        text = pages[i]["text"].lower()

        if "index" in text:
            return i

    return None


def extract_appendix_c(pages: list) -> tuple[list, int, int | None]:
    if START_IDX >= len(pages):
        raise IndexError(
            f"START_IDX={START_IDX} is outside is the list. "
            f"Pages in JSON: {len(pages)}"
        )

    real_start_idx = find_real_start_index(pages, START_IDX)
    end_idx = find_end_index(pages, real_start_idx)

    appendix_pages = (
        pages[real_start_idx:end_idx]
        if end_idx is not None
        else pages[real_start_idx:]
    )

    return appendix_pages, real_start_idx, end_idx


def combine_text(pages: list) -> str:
    return "\n\n".join(page["text"] for page in pages)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def print_debug_preview(pages: list, start_idx: int, end_idx: int | None, text: str) -> None:
    print(f"Expected startindex: {START_IDX} (page {APPENDIX_START_PAGE})")
    print(f"actual startindex: {start_idx}")
    print(f"Endindex: {end_idx}")
    print(f"Number of pages in Appendix C: {len(pages)}")

    print("\n--- FIRST PAGE IN APPENDIX C ---\n")
    print(pages[0]["text"][:1500] if pages else "No pages found.")

    print("\n--- START OF RULE GLOSSARY ---\n")
    print(text[:2000])


def main() -> None:
    pages = load_pages(INPUT_PATH)
    appendix_pages, start_idx, end_idx = extract_appendix_c(pages)

    if not appendix_pages:
        print("No pages found for Appendix C / Rule Glossary.")
        return

    glossary_text = combine_text(appendix_pages)
    save_text(OUTPUT_PATH, glossary_text)
    print_debug_preview(appendix_pages, start_idx, end_idx, glossary_text)

    print(f"\nRule Glossary saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()