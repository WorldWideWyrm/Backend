import json
from pathlib import Path


INPUT_PATH = Path("pdfs/cleaned_pages.json")
OUTPUT_PATH = Path("pdfs/rule_glossary.txt")

# Hvis Appendix C starter på bogens side 359,
# og pages-listen er 0-indekseret:
APPENDIX_START_PAGE = 360
START_IDX = APPENDIX_START_PAGE - 1


def load_pages(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Filen blev ikke fundet: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON-filen forventes at indeholde en liste af sider.")

    for i, page in enumerate(data):
        if not isinstance(page, dict):
            raise ValueError(f"Side {i} er ikke et dictionary/object.")
        if "text" not in page:
            raise ValueError(f"Side {i} mangler nøgle: 'text'")

    return data


def find_real_start_index(pages: list, fallback_idx: int) -> int:
    """
    Leder efter den rigtige start omkring den forventede side.
    Finder helst en side med 'rule glossary' eller 'appendix c'.
    Hvis intet findes, bruges fallback_idx.
    """
    search_start = max(0, fallback_idx - 3)
    search_end = min(len(pages), fallback_idx + 5)

    for i in range(search_start, search_end):
        text = pages[i]["text"].lower()
        if "rule glossary" in text or "appendix c" in text:
            return i

    return fallback_idx


def find_end_index(pages: list, start_idx: int) -> int | None:
    """
    Finder første side efter start, hvor Index begynder.
    Returnerer None hvis Index ikke findes.
    """
    for i in range(start_idx + 1, len(pages)):
        text = pages[i]["text"].lower()

        # Du kan udvide med flere stopord hvis nødvendigt
        if "index" in text:
            return i

    return None


def extract_appendix_c(pages: list) -> tuple[list, int, int | None]:
    if START_IDX >= len(pages):
        raise IndexError(
            f"START_IDX={START_IDX} er uden for listen. "
            f"Antal sider i JSON: {len(pages)}"
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
    print(f"Forventet startindex: {START_IDX} (bogside {APPENDIX_START_PAGE})")
    print(f"Faktisk startindex: {start_idx}")
    print(f"Slutindex: {end_idx}")
    print(f"Antal sider i Appendix C: {len(pages)}")

    print("\n--- FØRSTE SIDE I APPENDIX C ---\n")
    print(pages[0]["text"][:1500] if pages else "Ingen sider fundet.")

    print("\n--- START OF RULE GLOSSARY ---\n")
    print(text[:2000])


def main() -> None:
    pages = load_pages(INPUT_PATH)
    appendix_pages, start_idx, end_idx = extract_appendix_c(pages)

    if not appendix_pages:
        print("Ingen sider fundet til Appendix C / Rule Glossary.")
        return

    glossary_text = combine_text(appendix_pages)
    save_text(OUTPUT_PATH, glossary_text)
    print_debug_preview(appendix_pages, start_idx, end_idx, glossary_text)

    print(f"\nRule Glossary gemt i: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()