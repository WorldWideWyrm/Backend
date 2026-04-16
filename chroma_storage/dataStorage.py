import json
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

INPUT_RULES = Path("pdfs/rules_chunks.json")
INPUT_SPELLS = Path("pdfs/spell_chunks.json")

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "handbook"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of chunks")

    return data


def clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def normalize_rule_chunks(chunks: list[dict]) -> list[dict]:
    normalized = []

    for chunk in chunks:
        chunk_id = chunk.get("id")
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})

        if not chunk_id or not text:
            continue

        if not isinstance(metadata, dict):
            metadata = {}

        metadata["type"] = "rule"

        normalized.append({
            "id": chunk_id,
            "text": text,
            "metadata": clean_metadata(metadata),
        })

    return normalized


def normalize_spell_chunks(chunks: list[dict]) -> list[dict]:
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
            "metadata": clean_metadata(metadata),
        })

    return normalized


def main() -> None:
    rule_chunks = normalize_rule_chunks(load_json(INPUT_RULES))
    spell_chunks = normalize_spell_chunks(load_json(INPUT_SPELLS))

    all_chunks = rule_chunks + spell_chunks

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        print("No existing collection found")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "PHB 2024 handbook: rules + spells"},
    )

    ids = [chunk["id"] for chunk in all_chunks]
    documents = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Indsat {len(rule_chunks)} rules")
    print(f"Indsat {len(spell_chunks)} spells")
    print(f"Indsat {len(all_chunks)} chunks i collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()