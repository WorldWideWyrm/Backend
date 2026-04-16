import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "handbook"


def main() -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    query = input("What would you like to know? ").strip()
    if not query:
        print("No query.")
        return

    result = collection.query(
        query_texts=[query],
        n_results=5
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0] if "distances" in result else []

    if not documents:
        print("No results found.")
        return

    for i, doc in enumerate(documents, 1):
        meta = metadatas[i - 1] if i - 1 < len(metadatas) else {}
        distance = distances[i - 1] if i - 1 < len(distances) else None

        print(f"\n--- RESULTS {i} ---")
        print(f"Titel: {meta.get('title', 'Unknown')}")
        print(f"Pages: {meta.get('start_page', '?')}–{meta.get('end_page', '?')}")
        if distance is not None:
            print(f"Distance: {distance}")
        print()
        print(doc)
        print()


if __name__ == "__main__":
    main()