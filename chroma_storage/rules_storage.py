import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

INPUT_PATH = Path("pdfs/rules_chunks.json")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rules"


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

    # Opret Chroma client (gemt på disk)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Embedding model
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Slet gammel collection (så du undgår duplicates)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Slettede eksisterende collection: {COLLECTION_NAME}")
    except Exception:
        print("Ingen eksisterende collection fundet")

    # Opret ny collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "PHB 2024 rules glossary"},
    )

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["id"])
        documents.append(chunk["text"])
        metadatas.append(clean_metadata(chunk["metadata"]))

    # Batch insert (bedre performance)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Indsat {len(ids)} rules i Chroma 🎲")


if __name__ == "__main__":
    main()