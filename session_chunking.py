import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="dnd_memory")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_session_text(text, chunk_size=120, overlap=20):
    """
    Split a Whisper transcript into smaller overlapping chunks.

    Args:
        text (str): Full transcript text
        chunk_size (int): Number of words in each chunk
        overlap (int): Number of overlapping words between chunks

    Returns:
        list[str]: List of text chunks
    """

    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    test_text = """
    The party entered the old ruins. The wizard found a red gem near the altar.
    The rogue checked for traps. Later the group fought two goblins outside
    the temple and questioned a guard about the missing caravan.
    """
    test_text2 = """
    The party entered the old ruins again. The wizard placed the red gem on the altar.
    A Dragons got summoned and the party was wiped out.
    """

    array = [test_text, test_text2]

    for z, text in enumerate(array, start=1):


        chunks = chunk_session_text(text, chunk_size=20, overlap=5)
        print(f"\n--- Session {z} ---")
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, start=1):
            collection.add(
                documents=[chunk], 
                embeddings=[embedding_model.encode(chunk).tolist()],
                metadatas=[{
                    "session": z,
                    "chunk_index": i,
                    "raw": chunk
                }],
                ids=[f"s{z}_c{i}"]
            )
            print(f"\n--- Chunk {i} ---")
            print(chunk)