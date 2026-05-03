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
    result = query_collection(collection, query)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    output = []
    output.append("\n==============================")
    output.append("RULES / HANDBOOK RESULTS")
    output.append("==============================")

    if not documents:
        output.append("No rule results found.")
        return "\n".join(output)

    for i, doc in enumerate(documents, 1):
        meta = metadatas[i - 1] if i - 1 < len(metadatas) else {}
        distance = distances[i - 1] if i - 1 < len(distances) else None

        output.append(f"\n--- Rule Result {i} ---")
        output.append(f"Title: {meta.get('title', 'Unknown')}")
        output.append(
            f"Pages: {meta.get('start_page', '?')}–{meta.get('end_page', '?')}"
        )

        if distance is not None:
            output.append(f"Distance: {distance}")

        output.append("")
        output.append(doc)

    return "\n".join(output)


def query_memory(collection, query):
    result = query_collection(collection, query)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    output = []
    output.append("\n==============================")
    output.append("DND MEMORY / CAMPAIGN RESULTS")
    output.append("==============================")

    if not documents:
        output.append("No campaign memory results found.")
        return "\n".join(output)

    already_printed = set()

    for i, doc in enumerate(documents, 1):
        meta = metadatas[i - 1] if i - 1 < len(metadatas) else {}
        distance = distances[i - 1] if i - 1 < len(distances) else None

        session = meta.get("session")
        chunk_index = meta.get("chunk_index")

        output.append(f"\n--- Memory Match {i} ---")
        output.append(f"Session: {session}")
        output.append(f"Chunk: {chunk_index}")

        if distance is not None:
            output.append(f"Distance: {distance}")

        if session is None or chunk_index is None:
            output.append("")
            output.append(doc)
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

        output.append("")
        output.append(
            f"Context from session {session}, "
            f"chunks {context_start}–{context_end}:"
        )
        output.append("")

        for item in context_items:
            context_meta = item["metadata"]
            output.append(f"[Chunk {context_meta.get('chunk_index')}]")
            output.append(item["document"])
            output.append("")

    return "\n".join(output)


def main(query=None):

    rules_collection = get_collection(
        RULES_DB_PATH,
        RULES_COLLECTION
    )

    memory_collection = get_collection(
        MEMORY_DB_PATH,
        MEMORY_COLLECTION
    )
    if query == None: 
        query = input("What would you like to know? ").strip()

    if not query:
        print("No query.")
        return

    main_output = []
    main_output.append(query_rules(rules_collection, query))
    main_output.append(query_memory(memory_collection, query))
    return "\n".join(main_output)



if __name__ == "__main__":
    print(main())