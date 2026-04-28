from transformers import GPT2TokenizerFast
import os
import torch
import torch.nn as nn
import math 
import chromadb
from chromadb.utils import embedding_functions
import Stemmer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "handbook"

class MyTransformer(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.d_size=vocab_size
        self.d_model = 512
        self.n=6
        self.h=8
        self.load_model(os.path.join(os.getcwd(), "torch_model.pkl"))


    def softmax(self, x, dim=-1):
        return torch.softmax(x, dim=dim)
    
    def load_model(self, path):
        
        if isinstance(path, str):
            model_data = torch.load(path)
        else:
            model_data = path 

        # Embedding
        self.dictionary_vectors = nn.Parameter(model_data['dictonary_vectors'])

        # Multihead attention (encoder input)
        self.multihead_input = nn.ParameterList([
            nn.Parameter(model_data['multihead_matrices_input'][i])
            for i in range(self.n)
        ])

        self.W_O_input = nn.ParameterList([
            nn.Parameter(model_data['W_O_input'][i])
            for i in range(self.n)
        ])

        # Masked attention (decoder)
        self.multihead_masked = nn.ParameterList([
            nn.Parameter(model_data['multihead_matrices_masked'][i])
            for i in range(self.n)
        ])

        self.W_O_masked = nn.ParameterList([
            nn.Parameter(model_data['W_O_masked'][i])
            for i in range(self.n)
        ])

        # Cross attention
        self.multihead_output = nn.ParameterList([
            nn.Parameter(model_data['multihead_matrices_output'][i])
            for i in range(self.n)
        ])

        self.W_O_output = nn.ParameterList([
            nn.Parameter(model_data['W_O_output'][i])
            for i in range(self.n)
        ])

        # Feedforward (input)
        self.ff_input = nn.ModuleList([
            nn.ParameterDict({
                "W1": nn.Parameter(layer["W1"]),
                "W2": nn.Parameter(layer["W2"])
            })
            for layer in model_data['feedforward_matrices_input']
        ])

        # Feedforward (output)
        self.ff_output = nn.ModuleList([
            nn.ParameterDict({
                "W1": nn.Parameter(layer["W1"]),
                "W2": nn.Parameter(layer["W2"])
            })
            for layer in model_data['feedforward_matrices_output']
        ])

        # Final projection
        self.last_linear = nn.Parameter(model_data["last_linear_matrices"])

    def token_vectors(self, tokens):
        return self.dictionary_vectors[tokens]

    def PE(self, pos, i, d_model):
        if i % 2 == 0:
            return math.sin(pos / (10000 ** (i / d_model)))
        else:
            return math.cos(pos / (10000 ** ((i - 1) / d_model)))

    def return_model(self):
        model_data = {
            'dictonary_vectors': self.dictionary_vectors,
            'multihead_matrices_input': self.multihead_input,
            'multihead_matrices_output': self.multihead_output,
            'multihead_matrices_masked': self.multihead_masked,
            'feedforward_matrices_input': self.ff_input,
            'feedforward_matrices_output': self.ff_output,
            'W_O_input': self.W_O_input,
            'W_O_output': self.W_O_output,
            'W_O_masked': self.W_O_masked,
            "last_linear_matrices": self.last_linear
        }
        return model_data


    def positional_encoding(self, tokens, start_pos):
        seq_len = tokens.shape[0]
        pe = torch.zeros((seq_len, self.d_model), dtype=torch.float32)

        for pos in range(seq_len):
            actual_pos = start_pos + pos
            for i in range(self.d_model):
                pe[pos, i] = self.PE(actual_pos, i, self.d_model)

        return tokens + pe, seq_len + 1

    def input_encoding(self, tokens):
        tokens_vektors = tokens
        for i in range(self.n):
            temp_matrice = self.multiheaded_attention( self.multihead_input[i],tokens_vektors, tokens_vektors)
            temp_matrice = torch.matmul(temp_matrice, self.W_O_input[i])
            tokens_vektors = self.add_norm(tokens_vektors, temp_matrice)
            temp_matrice = self.feedfarward(self.ff_input[i],tokens_vektors)
            tokens_vektors = self.add_norm(tokens_vektors, temp_matrice)
        return tokens_vektors

    def add_norm(self, tokens_vektors, temp_matric,  eps=1e-6):
        # Residual connection
        added = tokens_vektors + temp_matric

        # Layer normalization
        mean = torch.mean(added, dim=-1, keepdims=True)
        var = torch.mean((added - mean) ** 2, dim=-1, keepdims=True)

        return (added - mean) / torch.sqrt(var + eps)



    def multiheaded_attention(self, attention_model, q_input, kv_input, masked=False):
        d_k = self.d_model // self.h

        squeeze_batch = False
        if q_input.dim() == 2:
            q_input = q_input.unsqueeze(0)
            kv_input = kv_input.unsqueeze(0)
            squeeze_batch = True

        Wq, Wk, Wv = torch.chunk(attention_model, 3, dim=1)

        q = torch.matmul(q_input, Wq)
        k = torch.matmul(kv_input, Wk)
        v = torch.matmul(kv_input, Wv)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]

        q = q.view(batch_size, q_len, self.h, d_k).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.h, d_k).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.h, d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if masked:
            mask = torch.triu(
                torch.ones(q_len, kv_len, device=scores.device),
                diagonal=1
            ).bool()
            scores = scores.masked_fill(mask, -1e9)

        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.d_model
        )

        if squeeze_batch:
            out = out.squeeze(0)

        return out

    def feedfarward(self, feedfarward_model, tokens_vektors):
        W1 = feedfarward_model["W1"]
        W2 = feedfarward_model["W2"]

        # First linear
        hidden = torch.matmul(tokens_vektors, W1)

        hidden = torch.relu(hidden)

        # Second linear
        output = torch.matmul(hidden, W2)

        return output

    def output_decifiring(self, output_tokens, input_matrice,  next_start_pos=0):

        tokens, _ = self.positional_encoding( output_tokens, next_start_pos)

        for i in range(self.n):
            temp_matrice = self.multiheaded_attention( self.multihead_masked[i],tokens, tokens, masked=True)
            temp_matrice = torch.matmul(temp_matrice, self.W_O_masked[i])
            tokens = self.add_norm(tokens, temp_matrice)
            temp_matrice = self.multiheaded_attention( self.multihead_output[i],tokens, input_matrice)
            temp_matrice = torch.matmul(temp_matrice,self.W_O_output[i])
            tokens = self.add_norm(tokens, temp_matrice)
            temp_matrice = self.feedfarward(self.ff_output[i],tokens)
            tokens = self.add_norm(tokens, temp_matrice)
        logits = torch.matmul(tokens, self.last_linear)
        #probs = self.softmax(logits)
        #next_token = torch.argmax(probs[-1]).item()
        return logits
    
    def forward(self, input_ids, decoder_input_ids, target_ids=None):
        # --- Encoder ---
        enc = self.token_vectors(input_ids)
        enc, _ = self.positional_encoding(enc, 0)
        enc = self.input_encoding(enc)

        # --- Decoder ---
        dec = self.token_vectors(decoder_input_ids)
        logits = self.output_decifiring(dec, enc)

        if target_ids is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            return logits, loss

        return logits
        
    def getSentence(inputs):

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

        text = inputs

        result = collection.query(
                query_texts=[text],
                n_results=5
            )
        stemmed = " ".join(stemmer.stemWords(text.split()))

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0] if "distances" in result else []

        if not documents:
            return "No results found."

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

        checkpoint = torch.load(
            "first_run_model.pt",
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=False
        )

        model.load_model(checkpoint["model"])

        tokens = model.token_vectors(tokens)

        # print(tokens.shape)

        tokens, next_start_pos = model.positional_encoding(tokens, 0)

        input_matrice = model.input_encoding(tokens)


        output_ids = tokenizer.encode("Model: ")



        for i in range(500):
            output_tokens = model.token_vectors( output_ids)

            t = model.output_decifiring(output_tokens, input_matrice, next_start_pos)
            print(t)
            t = torch.argmax(model.softmax(t, dim=-1)[-1]).item()
            output_ids.append(t)

            if t == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(output_ids)
        return text