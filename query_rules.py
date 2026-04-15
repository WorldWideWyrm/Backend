import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rules"


def main() -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    query = input("Skriv et spørgsmål: ").strip()
    if not query:
        print("Ingen query skrevet.")
        return

    result = collection.query(
        query_texts=[query],
        n_results=5
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0] if "distances" in result else []

    if not documents:
        print("Ingen resultater fundet.")
        return

    for i, doc in enumerate(documents, 1):
        meta = metadatas[i - 1] if i - 1 < len(metadatas) else {}
        distance = distances[i - 1] if i - 1 < len(distances) else None

        print(f"\n--- RESULTAT {i} ---")
        print(f"Titel: {meta.get('title', 'Ukendt')}")
        print(f"Sider: {meta.get('start_page', '?')}–{meta.get('end_page', '?')}")
        if distance is not None:
            print(f"Distance: {distance}")
        print()
        print(doc)
        print()


if __name__ == "__main__":
    main()