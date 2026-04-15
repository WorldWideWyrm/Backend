from transformers import GPT2TokenizerFast
from transformers import GPT2TokenizerFast
import Stemmer
import sys
sys.path.append("Backend")
import torch_module
import math 
import os
import torch
import time
import random



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
        return 

def log(message, path="train_log.txt"):
    with open(path, "a") as f:
        f.write(message + "\n")

def train(model, input_words, decoder_inputs, targets, save_path, device, batch_size=8, lr=3e-4):
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    
    start_step = 0
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_model(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        print(f"Resumed from step {start_step}")

    # ---- Timing ----
    START_TIME = time.time()
    MAX_TIME = 11.5 * 60 * 60
    last_save = time.time()

    num_samples = len(input_words)
    indices = list(range(num_samples))

    step = start_step

    while True:
        random.shuffle(indices)

        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]

            # ---- Build batch ----
            batch_input = torch.stack([input_words[j] for j in batch_idx]).to(device)
            batch_decoder = torch.stack([decoder_inputs[j] for j in batch_idx]).to(device)
            batch_targets = torch.stack([targets[j] for j in batch_idx]).to(device)

            optimizer.zero_grad()

            logits, loss = model(batch_input, batch_decoder, batch_targets)

            loss.backward()
            optimizer.step()

            step += 1

            # ---- Logging ----
            if step % 50 == 0:
                log(f"Step {step} | Loss: {loss.item():.4f}")

            # ---- Save every 5 minutes ----
            if time.time() - last_save > 300:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step
                }, save_path)

                log(f"Checkpoint saved at step {step}")
                last_save = time.time()

            # ---- Stop before cutoff ----
            if time.time() - START_TIME > MAX_TIME:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step
                }, save_path.replace(".pt", "_final.pt"))

                log("Final checkpoint saved. Exiting safely.")
                return


dictonary_vectors, multihead_matrices_input, multihead_matrices_output, multihead_matrices_masked, feedforward_matrices_input, feedforward_matrices_output, W_O_input, W_O_output, W_O_masked, last_linear_matrices  = matricies_init_ ()


save(dictonary_vectors, multihead_matrices_input, multihead_matrices_output, multihead_matrices_masked, feedforward_matrices_input, feedforward_matrices_output, W_O_input, W_O_output, W_O_masked, last_linear_matrices , os.path.join(os.getcwd(), "torch_model.pkl"))


ai_model = torch_module.MyTransformer()


      