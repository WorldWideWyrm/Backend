from transformers import GPT2TokenizerFast
import Stemmer
import os
import math
import torch
import torch.nn as nn

d_size=50257
d_model = 512
n=6
h=8



def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)
def load_model():
    model_data = torch.load(os.path.join("first_run_model.pt"))
    return model_data

def token_vectors(model, tokens):
    return model['dictonary_vectors'][tokens]

def PE(pos, i, d_model):
    if i % 2 == 0:
        return math.sin(pos / (10000 ** (i / d_model)))
    else:
        return math.cos(pos / (10000 ** ((i - 1) / d_model)))


def positional_encoding(tokens, start_pos, d_model):
    seq_len = tokens.shape[0]
    pe = torch.zeros((seq_len, d_model), dtype=torch.float32)

    for pos in range(seq_len):
        actual_pos = start_pos + pos
        for i in range(d_model):
            pe[pos, i] = PE(actual_pos, i, d_model)

    return tokens + pe, seq_len + 1

def input_encoding(model, tokens, d_model, n, h):
    tokens_vektors = tokens
    for i in range(n):
        temp_matrice = multiheaded_attention(model['multihead_matrices_input'][i],tokens_vektors, tokens_vektors, d_model, h)
        temp_matrice = torch.matmul(temp_matrice, model['W_O_input'][i])
        tokens_vektors = add_norm(tokens_vektors, temp_matrice)
        temp_matrice = feedfarward(model['feedforward_matrices_input'][i],tokens_vektors)
        tokens_vektors = add_norm(tokens_vektors, temp_matrice)
    return tokens_vektors

def add_norm(tokens_vektors, temp_matric,  eps=1e-6):
    # Residual connection
    added = tokens_vektors + temp_matric

    # Layer normalization
    mean = torch.mean(added, axis=1, keepdims=True)
    var = torch.mean((added - mean) ** 2, axis=1, keepdims=True)

    return (added - mean) / torch.sqrt(var + eps)



def multiheaded_attention(attention_model, q_input, kv_input, d_model, h, masked=False):
    d_k = d_model // h

    # separate projections
    Wq, Wk, Wv = torch.chunk(attention_model, 3, axis=1)

    q = torch.matmul(q_input, Wq)
    k = torch.matmul(kv_input, Wk)
    v = torch.matmul(kv_input, Wv)

    def split_heads(x):
        return x.reshape(x.shape[0], h, d_k)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    outputs = []

    for i in range(h):
        qi = q[:, i, :]
        ki = k[:, i, :]
        vi = v[:, i, :]

        scores = torch.matmul(qi, ki.T) / math.sqrt(d_k)

        if masked:
            mask = torch.triu(torch.ones_like(scores),diagonal=1) * -1e9
            scores = scores + mask

        weights = softmax(scores)
        outputs.append(torch.matmul(weights, vi))

    return torch.cat(outputs, dim=1)

def feedfarward(feedfarward_model, tokens_vektors):
    W1 = feedfarward_model["W1"]
    W2 = feedfarward_model["W2"]

    # First linear
    hidden = torch.matmul(tokens_vektors, W1)

    hidden = torch.relu(hidden)

    # Second linear
    output = torch.matmul(hidden, W2)

    return output

def output_decifiring(output_tokens, model, input_matrice, d_model, n, h, next_start_pos):

    tokens, _ = positional_encoding(output_tokens, next_start_pos, d_model)

    for i in range(n):
        temp_matrice = multiheaded_attention(model['multihead_matrices_masked'][i],tokens, tokens, d_model, h, True)
        temp_matrice = torch.matmul(temp_matrice, model['W_O_masked'][i])
        tokens = add_norm(tokens, temp_matrice)
        temp_matrice = multiheaded_attention(model['multihead_matrices_output'][i],tokens, input_matrice, d_model, h)
        temp_matrice = torch.matmul(temp_matrice, model['W_O_output'][i])
        tokens = add_norm(tokens, temp_matrice)
        temp_matrice = feedfarward(model['feedforward_matrices_output'][i],tokens)
        tokens = add_norm(tokens, temp_matrice)
    logits = torch.matmul(tokens, model["last_linear_matrices"])
    probs = softmax(logits)
    next_token = torch.argmax(probs[-1]).item()
    return next_token

    

stemmer = Stemmer.Stemmer("english")
tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4", local_files_only=True)

text = input("User: ")

stemmed = " ".join(stemmer.stemWords(text.split()))

tokens = tokenizer.encode("User:" + stemmed )

print("stemmed:", stemmed)
print("tokens:", tokens)


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