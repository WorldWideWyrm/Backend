from transformers import GPT2TokenizerFast
import Stemmer
import numpy as np
import os
import pickle
import math

d_model = 512
n=6
h=8



def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)

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
    for i in len(n):
        temp_matrice = multiheaded_attention(model['multihead_matrices_input'][n],tokens_vektors, d_model, h)

def multiheaded_attention(attention_model, tokens, d_model, h):
    wqkv = attention_model * tokens
    qkv = []
    for i in len(h):
        q = wqkv.
        k = wqkv.
        v = wqkv.
        qkv.append((softmax((q*k.trans)/math.sqrt(d_model/h)))*v)

def feedfarward(feedfarward_model, tokens_vektors, temp_matrice):
    

stemmer = Stemmer.Stemmer("english")
tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4", local_files_only=True)

text = input("input: ")

stemmed = " ".join(stemmer.stemWords(text.split()))

tokens = tokenizer.encode("User:" + stemmed + " Model:")

print("stemmed:", stemmed)
print("tokens:", tokens)


model = load_model()

tokens = token_vectors(model, tokens)

print(tokens.shape)

tokens, next_start_pos = positional_encoding(tokens, 0, d_model)




