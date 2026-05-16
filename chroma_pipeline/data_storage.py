from pathlib import Path
from base_guideline import ChunkingStrategy
from rules_chunking import RuleChunkingStrategy
from spells_chunking import SpellChunkingStrategy
import json

current = Path(__file__).resolve()

# walk upward through parent folders
for parent in current.parents:
    storage_path = parent / "Storage"

    if storage_path.exists() and storage_path.is_dir():
        STORAGE_DIR = storage_path
        break
else:
    raise FileNotFoundError("Could not find storage folder")

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import os

INPUT_PATH = Path("pdfs/cleaned_pages.json")

CHROMA_PATH = os.path.join(storage_path, "chroma_db") 
COLLECTION_NAME = "handbook"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def run_strategy(strategy: ChunkingStrategy, pages: list[dict]) -> list[dict]:
    chunks = strategy.chunking(pages)
    chunks = strategy.normalize(chunks)
    return chunks

def load_pages() -> list:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON-file needs to contain a list of pages.")

    for i, page in enumerate(data):
        if not isinstance(page, dict):
            raise ValueError(f"Page {i} is not a dictionary/object.")
        if "text" not in page:
            raise ValueError(f"Page {i} is missing: 'text'")

    return data

def main() -> None:
    pages: list[dict] = load_pages()

    strategies: list[ChunkingStrategy] = [
        RuleChunkingStrategy(),
        SpellChunkingStrategy(),
    ]
     
    all_chunks: list[dict] = []

    for strategy in strategies:
        all_chunks.extend(run_strategy(strategy, pages))

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
        metadata={"description": "PHB 2024 handbook: rules + spell rules + spells"},
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


if __name__ == "__main__":
    main()