from transformers import GPT2TokenizerFast
import Stemmer
import os
import math
import torch
import torch.nn as nn
import chromadb
from chromadb.utils import embedding_functions
import sys
sys.path.append("Backend")
import torch_module

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

vocab_size = len(tokenizer)

model =  torch_module.MyTransformer(vocab_size)

checkpoint = torch.load("first_run_model.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model.load_model(checkpoint["model"])

tokens = model.token_vectors(model, tokens)

print(tokens.shape)

tokens, next_start_pos = model.positional_encoding(tokens, 0, d_model)

input_matrice = model.input_encoding(model, tokens)


output_ids = tokenizer.encode("Model: ")



for i in range(500):
    output_tokens = model.token_vectors( output_ids)

    t = model.output_decifiring(output_tokens, input_matrice, next_start_pos)
    t = torch.argmax(model.softmax(t)[-1]).item()
    output_ids.append(t)

    if t == tokenizer.eos_token_id:
        break

text = tokenizer.decode(output_ids)
print(text)