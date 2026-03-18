import random
from transformers import GPT2TokenizerFast
import math 
import Stemmer
import numpy as np
import os
import pickle



feedforward_matrices = []
d_model = 512
d_size = 50257
fan_in = d_model
fan_out = d_model
qkv = 3*8*64
n = 6
limit = math.sqrt(6 / (d_model *2))
d_ff = 2048

def matricies_init_ ():
    dictonary_vectors = np.array([[ random.uniform(-limit, limit) for _ in range(d_model)] for _ in range(d_size)])
    multihead_matrices_input = np.array([[[ random.uniform(- limit, limit) for _ in range(qkv)] for _ in range(d_model)] for _ in range(n)])
    W_O_input = np.random.uniform(-limit, limit, (n, d_model, d_model))
    multihead_matrices_masked = np.array([[[ random.uniform(- limit, limit) for _ in range(qkv)] for _ in range(d_model)] for _ in range(n)])
    W_O_masked = np.random.uniform(-limit, limit, (n, d_model, d_model))
    multihead_matrices_output = np.array([[[ random.uniform(- limit, limit) for _ in range(qkv)] for _ in range(d_model)] for _ in range(n)])
    W_O_output = np.random.uniform(-limit, limit, (n, d_model, d_model))
    feedforward_matrices_input = [{"W1": np.array([[random.uniform(-limit, limit) for _ in range(d_ff)]for _ in range(d_model)]),
                             "W2": np.array([[random.uniform(-limit, limit) for _ in range(d_model)]for _ in range(d_ff)])}for _ in range(n)]
    feedforward_matrices_output = [{"W1": np.array([[random.uniform(-limit, limit) for _ in range(d_ff)]for _ in range(d_model)]),
                             "W2": np.array([[random.uniform(-limit, limit) for _ in range(d_model)]for _ in range(d_ff)])}for _ in range(n)]
    last_linear_matrices = np.random.uniform(-limit, limit, ( d_model, d_size))
    return dictonary_vectors, multihead_matrices_input, multihead_matrices_output, multihead_matrices_masked, feedforward_matrices_input, feedforward_matrices_output, W_O_input, W_O_output, W_O_masked

def save(dictonary_vectors, multihead_matrices_input, multihead_matrices_output, multihead_matrices_masked, feedforward_matrices_input, feedforward_matrices_output, W_O_input, W_O_output, W_O_masked, filepath):
        model_data = {
            'dictonary_vectors': dictonary_vectors,
            'multihead_matrices_input': multihead_matrices_input,
            'multihead_matrices_output': multihead_matrices_output,
            'multihead_matrices_masked': multihead_matrices_masked,
            'feedforward_matrices_input': feedforward_matrices_input,
            'feedforward_matrices_output': feedforward_matrices_output,
            'W_O_input': W_O_input,
            'W_O_output': W_O_output,
            'W_O_masked': W_O_masked
        }
        with open(filepath, 'wb') as file:
            pickle.dump(model_data, file)

dictonary_vectors, multihead_matrices_input, multihead_matrices_output, multihead_matrices_masked, feedforward_matrices_input, feedforward_matrices_output, W_O_input, W_O_output, W_O_masked = matricies_init_ ()


save(dictonary_vectors, multihead_matrices_input, multihead_matrices_output, multihead_matrices_masked, feedforward_matrices_input, feedforward_matrices_output, W_O_input, W_O_output, W_O_masked, os.path.join(os.getcwd(), "model.pkl"))
