import json
from pathlib import Path


def is_spell_start(lines, i):
    return (
        i + 2 < len(lines)
        and "Casting Time" in lines[i + 2]
        and not lines[i].lower().startswith("chapter")
        and not lines[i].lower().startswith("appendix")
    )


def main():
    input_path = Path("pdfs/cleaned_pages.json")
    output_path = Path("pdfs/spell_chunks.json")

    with input_path.open("r", encoding="utf-8") as f:
        pages = json.load(f)

    all_spells = []
    current_spell = None

    for page in pages:
        lines = [line.strip() for line in page["text"].split("\n") if line.strip()]

        i = 0
        while i < len(lines):
            if is_spell_start(lines, i):
                if current_spell is not None:
                    all_spells.append(current_spell)

                current_spell = {
                    "chunk_id": f"spell_{len(all_spells) + 1}",
                    "start_page": page["page_number"],
                    "text": ""
                }

            if current_spell is not None:
                current_spell["text"] += lines[i] + "\n"

            i += 1

    if current_spell is not None:
        all_spells.append(current_spell)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_spells, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_spells)} spells in {output_path}")

    for spell in all_spells[:3]:
        print("\n----- SPELL -----")
        print(f"Start page: {spell['start_page']}")
        print(spell["text"][:1200])


if __name__ == "__main__":
    main() 