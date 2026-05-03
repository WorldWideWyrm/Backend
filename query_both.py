from sentence_transformers import SentenceTransformer
import chromadb

RULES_DB_PATH = "chroma_db"
RULES_COLLECTION = "handbook"

MEMORY_DB_PATH = "./chroma_db2"
MEMORY_COLLECTION = "dnd_memory"
MEMORY_CONTEXT_RADIUS = 10

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_collection(path, name):
    client = chromadb.PersistentClient(path=path)
    return client.get_collection(name=name)


def embed(text):
    return embedding_model.encode(text).tolist()


def query_collection(collection, query, n_results=5):
    query_embedding = embed(query)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )


def query_rules(collection, query):
    result = query_collection(
        collection,
        query
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    print("\n==============================")
    print("RULES / HANDBOOK RESULTS")
    print("==============================")

    if not documents:
        print("No rule results found.")
        return

    for i, doc in enumerate(documents, 1):
        meta = metadatas[i - 1] if i - 1 < len(metadatas) else {}
        distance = distances[i - 1] if i - 1 < len(distances) else None

        print(f"\n--- Rule Result {i} ---")
        print(f"Title: {meta.get('title', 'Unknown')}")
        print(f"Pages: {meta.get('start_page', '?')}–{meta.get('end_page', '?')}")

        if distance is not None:
            print(f"Distance: {distance}")

        print()
        print(doc)


def query_memory(collection, query):
    result = query_collection(
            collection,
            query
        )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    print("\n==============================")
    print("DND MEMORY / CAMPAIGN RESULTS")
    print("==============================")

    if not documents:
        print("No campaign memory results found.")
        return

    already_printed = set()

    for i, doc in enumerate(documents, 1):
        meta = metadatas[i - 1] if i - 1 < len(metadatas) else {}
        distance = distances[i - 1] if i - 1 < len(distances) else None

        session = meta.get("session")
        chunk_index = meta.get("chunk_index")

        print(f"\n--- Memory Match {i} ---")
        print(f"Session: {session}")
        print(f"Chunk: {chunk_index}")

        if distance is not None:
            print(f"Distance: {distance}")

        if session is None or chunk_index is None:
            print()
            print(doc)
            continue

        context_start = max(1, chunk_index - MEMORY_CONTEXT_RADIUS)
        context_end = chunk_index + MEMORY_CONTEXT_RADIUS

        where_filter = {
            "$and": [
                {"session": {"$eq": session}},
                {"chunk_index": {"$gte": context_start}},
                {"chunk_index": {"$lte": context_end}},
            ]
        }

        context = collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        context_items = []

        for context_doc, context_meta in zip(
            context.get("documents", []),
            context.get("metadatas", []),
        ):
            key = (
                context_meta.get("session"),
                context_meta.get("chunk_index"),
            )

            if key in already_printed:
                continue

            already_printed.add(key)

            context_items.append({
                "document": context_doc,
                "metadata": context_meta,
            })

        context_items.sort(
            key=lambda item: item["metadata"].get("chunk_index", 0)
        )

        print()
        print(
            f"Context from session {session}, "
            f"chunks {context_start}–{context_end}:"
        )
        print()

        for item in context_items:
            context_meta = item["metadata"]
            print(f"[Chunk {context_meta.get('chunk_index')}]")
            print(item["document"])
            print()


def main():

    rules_collection = get_collection(
        RULES_DB_PATH,
        RULES_COLLECTION
    )

    memory_collection = get_collection(
        MEMORY_DB_PATH,
        MEMORY_COLLECTION
    )

    query = input("What would you like to know? ").strip()

    if not query:
        print("No query.")
        return

    query_rules(rules_collection, query)
    query_memory(memory_collection, query)


if __name__ == "__main__":
    main()