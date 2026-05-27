from base_guideline import ChunkingStrategy
import re

class RuleChunkingStrategy(ChunkingStrategy):
    def __init__(self):
        self.APPENDIX_START_PAGE = 360
        self.SPELL_START_PAGE = 235
        self.SPELL_END_PAGE = 238
        self.NON_RULE_HEADINGS = {
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
        self.SKIP_HEADINGS = {
            "GLOSSARY CONVENTIONS",
            "RULES DEFINITIONS",
            "OBJECT ARMOR CLASS"
        }
        self.SPELL_RULE_HEADINGS = {
            "CASTING TIME",
            "RANGE",
            "COMPONENTS",
            "DURATION",
            "CONCENTRATION",
            "RITUAL",
            "VERBAL COMPONENT",
            "SOMATIC COMPONENT",
            "MATERIAL COMPONENT",
            "LONGER CASTING TIMES",
        }
    def chunking(self, pages: list[dict])-> list[dict]:
        spell_rule_chunks = self.chunk_spell_rules_from_pages(pages)
        glossary_chunks = self.chunk_rules_from_pages(pages)
        return spell_rule_chunks + glossary_chunks

    def normalize(self, chunks: list[dict])-> list[dict]:
        normalized = []

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id") or f"rule_{i:04d}"
            text = chunk.get("text", "").strip()
            metadata = chunk.get("metadata", {})

            if not text:
                continue

            if not isinstance(metadata, dict):
                metadata = {}

            metadata.setdefault("type", "rule")
            metadata.setdefault("source", "PHB 2024 Rules")

            normalized.append({
                "id": chunk_id,
                "text": text,
                "metadata": self.clean_metadata(metadata),
            })

        return normalized

    def normalize_line(self, line: str) -> str:
        return re.sub(r"\s+", " ", line).strip()

    def chunk_rules_from_pages(self, pages: list[dict]) -> list[dict]:
        pages = [p for p in pages if p.get("page_number", 0) >= self.APPENDIX_START_PAGE]

        chunks = []
        current = None
        seen_rules_definitions = False

        for page in pages:
            page_number = page["page_number"]
            raw_text = page["text"]

            lines = [self.normalize_line(line) for line in raw_text.split("\n")]
            lines = [line for line in lines if line]

            for line in lines:
                upper_line = line.upper()

                if not seen_rules_definitions:
                    if upper_line == "RULES DEFINITIONS":
                        seen_rules_definitions = True
                    continue

                if upper_line.startswith("INDEX"):
                    if current is not None:
                        current["text"] = current["text"].strip()
                        chunks.append(current)
                    return self.finalize_chunks(chunks)

                if upper_line in self.SKIP_HEADINGS:
                    continue

                if self.is_heading(line):
                    if current is not None:
                        current["text"] = current["text"].strip()
                        chunks.append(current)

                    current = {
                        "id": f"rule_{len(chunks):04d}",
                        "text": "",
                        "metadata": {
                            "type": "rule",
                            "title": upper_line,
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

        return self.finalize_chunks(chunks)
    
    def is_heading(self, line: str) -> bool:
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

    def finalize_chunks(self, chunks: list[dict]) -> list[dict]:
        for chunk in chunks:
            title = chunk["metadata"]["title"]
            body = chunk["text"]
            chunk_type = chunk["metadata"].get("type", "rule")

            if chunk_type == "spell_rule":
                chunk["text"] = f"Spellcasting Rule (Chapter 7): {title}\n\n{body}"
            else:
                chunk["text"] = f"{title}\n\n{body}"

        return chunks

    def chunk_spell_rules_from_pages(self, pages: list[dict]) -> list[dict]:
        pages = [
            p for p in pages
            if self.SPELL_START_PAGE <= p.get("page_number", 0) <= self.SPELL_END_PAGE
        ]

        chunks = []
        current = None

        for page in pages:
            page_number = page["page_number"]
            raw_text = page["text"]

            lines = [self.normalize_line(line) for line in raw_text.split("\n")]
            lines = [line for line in lines if line]

            for line in lines:
                upper_line = line.upper()

                if upper_line in self.SPELL_RULE_HEADINGS:
                    if current is not None:
                        current["text"] = current["text"].strip()
                        chunks.append(current)

                    current = {
                        "id": f"spell_rule_{len(chunks):04d}",
                        "text": "",
                        "metadata": {
                            "type": "spell_rule",
                            "title": upper_line,
                            "start_page": page_number,
                            "end_page": page_number,
                            "source": "PHB 2024 Chapter 7 Spell Rules",
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

        return self.finalize_chunks(chunks)