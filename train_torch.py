from transformers import GPT2TokenizerFast
import math 
import os
import torch



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
    dictonary_vectors = torch.nn.Parameter(torch.rand(d_size, d_model) * 2 * limit - limit)
    multihead_matrices_input = torch.nn.Parameter(torch.rand(n, d_model, qkv) * 2 * limit - limit)
    W_O_input = torch.nn.Parameter(torch.rand(n, d_model, d_model) * 2 * limit - limit)
    multihead_matrices_masked = torch.nn.Parameter(torch.rand(n, d_model, qkv) * 2 * limit - limit)
    W_O_masked = torch.nn.Parameter(torch.rand(n, d_model, d_model) * 2 * limit - limit)
    multihead_matrices_output = torch.nn.Parameter(torch.rand(n, d_model, qkv) * 2 * limit - limit)
    W_O_output = torch.nn.Parameter(torch.rand(n, d_model, d_model) * 2 * limit - limit)
    feedforward_matrices_input = [{
        "W1": torch.nn.Parameter(torch.rand(d_model, d_ff) * 2 * limit - limit),
        "W2": torch.nn.Parameter(torch.rand(d_ff, d_model) * 2 * limit - limit)
        }for _ in range(n)]
    feedforward_matrices_output = [{
        "W1": torch.nn.Parameter(torch.rand(d_model, d_ff) * 2 * limit - limit),
        "W2": torch.nn.Parameter(torch.rand(d_ff, d_model) * 2 * limit - limit)
        }for _ in range(n)]
    last_linear_matrices = torch.nn.Parameter(torch.rand(d_model, d_size) * 2 * limit - limit)
    return dictonary_vectors, multihead_matrices_input, multihead_matrices_output, multihead_matrices_masked, feedforward_matrices_input, feedforward_matrices_output, W_O_input, W_O_output, W_O_masked, last_linear_matrices

def save(dictonary_vectors, multihead_matrices_input, multihead_matrices_output, multihead_matrices_masked, feedforward_matrices_input, feedforward_matrices_output, W_O_input, W_O_output, W_O_masked, last_linear_matrices,  filepath):
        model_data = {
            'dictonary_vectors': dictonary_vectors,
            'multihead_matrices_input': multihead_matrices_input,
            'multihead_matrices_output': multihead_matrices_output,
            'multihead_matrices_masked': multihead_matrices_masked,
            'feedforward_matrices_input': feedforward_matrices_input,
            'feedforward_matrices_output': feedforward_matrices_output,
            'W_O_input': W_O_input,
            'W_O_output': W_O_output,
            'W_O_masked': W_O_masked,
            "last_linear_matrices": last_linear_matrices
        }
        torch.save(model_data, filepath)

dictonary_vectors, multihead_matrices_input, multihead_matrices_output, multihead_matrices_masked, feedforward_matrices_input, feedforward_matrices_output, W_O_input, W_O_output, W_O_masked, last_linear_matrices  = matricies_init_ ()


save(dictonary_vectors, multihead_matrices_input, multihead_matrices_output, multihead_matrices_masked, feedforward_matrices_input, feedforward_matrices_output, W_O_input, W_O_output, W_O_masked, last_linear_matrices , os.path.join(os.getcwd(), "torch_model.pkl"))
