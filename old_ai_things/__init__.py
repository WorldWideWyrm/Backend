from transformers import GPT2TokenizerFast
import Stemmer
import numpy as np
import os
import pickle
import math
import torch

d_size=50257
d_model = 512
n=6
h=8



def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=axis, keepdims=True)

def load_model():
    with open(os.path.join(os.getcwd(), "model.pkl"), 'rb') as file:
        model_data = pickle.load(file)
    return model_data

def token_vectors(model, tokens):
    return np.array([model['dictonary_vectors'][t] for t in tokens])

def PE(pos, i, d_model):
    if i % 2 == 0:
        return math.sin(pos / (10000 ** (i / d_model)))
    else:
        return math.cos(pos / (10000 ** ((i - 1) / d_model)))


def positional_encoding(tokens, start_pos, d_model):
    seq_len = tokens.shape[0]
    pe = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        actual_pos = start_pos + pos
        for i in range(d_model):
            pe[pos, i] = PE(actual_pos, i, d_model)

    return tokens + pe, seq_len + 1

def input_encoding(model, tokens, d_model, n, h):
    tokens_vektors = tokens
    for i in range(n):
        temp_matrice = multiheaded_attention(model['multihead_matrices_input'][i],tokens_vektors, tokens_vektors, d_model, h)
        temp_matrice = temp_matrice @ model['W_O_input'][i]
        tokens_vektors = add_norm(tokens_vektors, temp_matrice)
        temp_matrice = feedfarward(model['feedforward_matrices_input'][i],tokens_vektors)
        tokens_vektors = add_norm(tokens_vektors, temp_matrice)
    return tokens_vektors

def add_norm(tokens_vektors, temp_matric,  eps=1e-6):
    # Residual connection
    added = tokens_vektors + temp_matric

    # Layer normalization
    mean = np.mean(added, axis=1, keepdims=True)
    var = np.mean((added - mean) ** 2, axis=1, keepdims=True)

    return (added - mean) / np.sqrt(var + eps)



def multiheaded_attention(attention_model, q_input, kv_input, d_model, h, masked=False):
    d_k = d_model // h

    # separate projections
    Wq, Wk, Wv = np.split(attention_model, 3, axis=1)

    q = q_input @ Wq
    k = kv_input @ Wk
    v = kv_input @ Wv

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

        scores = (qi @ ki.T) / math.sqrt(d_k)

        if masked:
            mask = np.triu(np.ones(scores.shape), k=1) * -1e9
            scores += mask

        weights = softmax(scores)
        outputs.append(weights @ vi)

    return np.concatenate(outputs, axis=1)

def feedfarward(feedfarward_model, tokens_vektors):
    W1 = feedfarward_model["W1"]
    W2 = feedfarward_model["W2"]

    # First linear
    hidden = tokens_vektors @ W1

    # Activation (ReLU)
    hidden = np.maximum(0, hidden)

    # Second linear
    output = hidden @ W2

    return output

def output_decifiring(output_tokens, model, input_matrice, d_model, n, h, next_start_pos):

    tokens, _ = positional_encoding(output_tokens, next_start_pos, d_model)

    for i in range(n):
        temp_matrice = multiheaded_attention(model['multihead_matrices_masked'][i],tokens, tokens, d_model, h, True)
        temp_matrice = temp_matrice @ model['W_O_masked'][i]
        tokens = add_norm(tokens, temp_matrice)
        temp_matrice = multiheaded_attention(model['multihead_matrices_output'][i],tokens, input_matrice, d_model, h)
        temp_matrice = temp_matrice @ model['W_O_output'][i]
        tokens = add_norm(tokens, temp_matrice)
        temp_matrice = feedfarward(model['feedforward_matrices_output'][i],tokens)
        tokens = add_norm(tokens, temp_matrice)
    logits = tokens @ model["last_linear_matrices"]
    probs = softmax(logits)
    next_token = np.argmax(probs[-1])  
    return next_token

    

stemmer = Stemmer.Stemmer("english")
tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4", local_files_only=True)

text = input("input: ")

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