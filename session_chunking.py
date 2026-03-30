import chromadb
from sentence_transformers import SentenceTransformer
import os

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

def get_files():
    parent_dir = os.path.dirname(os.getcwd())

    target_path = os.path.join(parent_dir, "front_end", "previous_sessions")

    file_texts = []
    
    for filename in os.listdir(target_path):
        file_path = os.path.join(target_path, filename)
        
        if os.path.isfile(file_path):  # skip directories
            with open(file_path, "r", encoding="utf-8") as f:
                file_texts.append(f.read())
    return file_texts

if __name__ == "__main__":
    


    file_texts = get_files()

    print(file_texts)

    for z, text in enumerate(file_texts, start=1):


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