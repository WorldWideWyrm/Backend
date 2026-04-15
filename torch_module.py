from transformers import GPT2TokenizerFast
import os
import torch
import torch.nn as nn

class MyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_size=50257
        self.d_model = 512
        self.n=6
        self.h=8


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
            nn.ModuleDict({
                "W1": nn.Parameter(layer["W1"]),
                "W2": nn.Parameter(layer["W2"])
            })
            for layer in model_data['feedforward_matrices_input']
        ])

        # Feedforward (output)
        self.ff_output = nn.ModuleList([
            nn.ModuleDict({
                "W1": nn.Parameter(layer["W1"]),
                "W2": nn.Parameter(layer["W2"])
            })
            for layer in model_data['feedforward_matrices_output']
        ])

        # Final projection
        self.last_linear = nn.Parameter(model_data["last_linear_matrices"])

    def token_vectors(self, tokens):
        return self.dictionary_vectors[tokens]

    def PE(pos, i, d_model):
        if i % 2 == 0:
            return torch.sin(pos / (10000 ** (i / d_model)))
        else:
            return torch.cos(pos / (10000 ** ((i - 1) / d_model)))


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
        mean = torch.mean(added, axis=1, keepdims=True)
        var = torch.mean((added - mean) ** 2, axis=1, keepdims=True)

        return (added - mean) / torch.sqrt(var + eps)



    def multiheaded_attention(self, attention_model, q_input, kv_input, masked=False):
        d_k = self.d_model // self.h

        # separate projections
        Wq, Wk, Wv = torch.chunk(attention_model, 3, axis=1)

        q = torch.matmul(q_input, Wq)
        k = torch.matmul(kv_input, Wk)
        v = torch.matmul(kv_input, Wv)

        def split_heads(x):
            return x.reshape(x.shape[0], self.h, d_k)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        outputs = []

        for i in range(self.h):
            qi = q[:, i, :]
            ki = k[:, i, :]
            vi = v[:, i, :]

            scores = torch.matmul(qi, ki.T) / torch.sqrt(d_k)

            if masked:
                mask = torch.triu(torch.ones_like(scores),diagonal=1) * -1e9
                scores = scores + mask

            weights = self.softmax(scores)
            outputs.append(torch.matmul(weights, vi))

        return torch.cat(outputs, dim=1)

    def feedfarward(self, feedfarward_model, tokens_vektors):
        W1 = feedfarward_model["W1"]
        W2 = feedfarward_model["W2"]

        # First linear
        hidden = torch.matmul(tokens_vektors, W1)

        hidden = torch.relu(hidden)

        # Second linear
        output = torch.matmul(hidden, W2)

        return output

    def output_decifiring(self, output_tokens, input_matrice, d_model, next_start_pos):

        tokens, _ = self.positional_encoding( output_tokens, next_start_pos, d_model)

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
        logits = self.decode(dec, enc)

        if target_ids is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            return logits, loss

        return logits
        
