import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

INPUT_PATH = Path("pdfs/spell_chunks.json")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "spells"


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Filen blev ikke fundet: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON-filen skal indeholde en liste")

    return data


def clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def main() -> None:
    chunks = load_chunks(INPUT_PATH)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        print("No existing collection found")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "PHB 2024 spells"},
    )

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"spell_{i:04d}")
        text = chunk.get("text", "").strip()

        if not text:
            continue

        metadata = {
            "type": chunk.get("type", "Spell"),
            "title": chunk.get("title", "Unknown Spell"),
            "start_page": chunk.get("start_page"),
            "source": "PHB 2024 Spells"
        }

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(clean_metadata(metadata))

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Indsat {len(ids)} spells i Chroma ✨")


if __name__ == "__main__":
    main()