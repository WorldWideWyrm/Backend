from transformers import GPT2TokenizerFast
import Stemmer
import os
import math
import torch
import torch.nn as nn
import chromadb
from chromadb.utils import embedding_functions

d_size=50257
d_model = 512
n=6
h=8
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "handbook"




client = chromadb.PersistentClient(path=CHROMA_PATH)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

stemmer = Stemmer.Stemmer("english")
tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4", local_files_only=True)

text = input("User: ")

result = collection.query(
        query_texts=[text],
        n_results=5
    )
stemmed = " ".join(stemmer.stemWords(text.split()))

documents = result.get("documents", [[]])[0]
metadatas = result.get("metadatas", [[]])[0]
distances = result.get("distances", [[]])[0] if "distances" in result else []

if not documents:
    print("No results found.")

for i, doc in enumerate(documents, 1):
    meta = metadatas[i - 1] if i - 1 < len(metadatas) else {}
    distance = distances[i - 1] if i - 1 < len(distances) else None

    text = text + f"--- RESULTS {i} ---" + f"Titel: {meta.get('title', 'Unknown')}" +f"Pages: {meta.get('start_page', '?')}–{meta.get('end_page', '?')}"
    if distance is not None:
        text = text + f"Distance: {distance}"
    text = text + doc



tokens = tokenizer.encode("User:" + stemmed )


model = load_model()

tokens = token_vectors(model, tokens)

print(tokens.shape)

tokens, next_start_pos = positional_encoding(tokens, 0, d_model)

input_matrice = input_encoding(model, tokens, d_model, n, h)


output_ids = tokenizer.encode("Model: ")
output_tokens = token_vectors(model, output_ids)



for i in range(500):
    output_tokens = token_vectors(model, output_ids)

    t = output_decifiring(output_tokens, model, input_matrice, d_model, n, h, next_start_pos)

    output_ids.append(t)

    if t == tokenizer.eos_token_id:
        break

text = tokenizer.decode(output_ids)
print(text)