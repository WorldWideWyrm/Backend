from base_guideline import ChunkingStrategy

class SpellChunkingStrategy(ChunkingStrategy):
    def __init__(self):
        self.SPELL_SECTION_START = 239
        self.SPELL_SECTION_END = 343

    def is_spell_start(self, lines, i):
        return (
            i + 3 < len(lines)
            and ("Level" in lines[i + 1] or "Cantrip" in lines[i + 1])
            and ("Casting Time" in lines[i + 2] or "Casting Time" in lines[i + 3])
            and not lines[i].lower().startswith("chapter")
            and not lines[i].lower().startswith("appendix")
        )

    def chunking(self, pages: list[dict])-> list[dict]:
        all_spells = []
        current_spell = None

        for page in pages:
            lines = [line.strip() for line in page["text"].split("\n") if line.strip()]

            i = 0
            while i < len(lines) and page["page_number"] >= self.SPELL_SECTION_START and page["page_number"] <= self.SPELL_SECTION_END:
                
                if self.is_spell_start(lines, i):
                    if current_spell is not None:
                        all_spells.append(current_spell)

                    current_spell = {
                        "chunk_id": f"spell_{len(all_spells) + 1}",
                        "type": "Spell",
                        "title": lines[i],
                        "start_page": page["page_number"],
                        "text": ""
                    }

                if current_spell is not None:
                    current_spell["text"] += lines[i] + "\n"

                i += 1

        if current_spell is not None:
            all_spells.append(current_spell)
        return all_spells


    def normalize(self, chunks: list[dict])-> list[dict]:
        normalized = []

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id") or chunk.get("chunk_id") or f"spell_{i:04d}"
            text = chunk.get("text", "").strip()

            if not text:
                continue

            if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                metadata = dict(chunk["metadata"])
            else:
                metadata = {
                    "title": chunk.get("title", "Unknown Spell"),
                    "start_page": chunk.get("start_page"),
                    "end_page": chunk.get("end_page", chunk.get("start_page")),
                    "source": "PHB 2024 Spells",
                }

            metadata["type"] = "spell"

            normalized.append({
                "id": chunk_id,
                "text": text,
                "metadata": self.clean_metadata(metadata),
            })

        return normalized