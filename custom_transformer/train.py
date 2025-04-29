"""
Patrick Marshall
CSC 5611 Deep Learning
01 May 2025

train.py
This file contains the training loop for the Transformer model. It initializes the model,
loads the dataset, and trains the model using a simple next-token prediction task. The training
loop includes forward and backward passes, loss calculation, and parameter updates.
"""

import torch
import random
import matplotlib.pyplot as plt
from transformer import Transformer
from layers import Input, MSELoss, DEVICE

# --- Load and Prepare Dataset ---
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Simple character-level tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return ''.join([itos[i] for i in indices])

# Encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Training hyperparameters
MAX_SEQ_LEN = 64
D_MODEL = 128
NUM_HEADS = 8
D_FF = 256
BATCH_SIZE = 1
EPOCHS = 1000
LR = 0.001

# Initialize model
transformer = Transformer(vocab_size, D_MODEL, NUM_HEADS, D_FF, MAX_SEQ_LEN)

# For tracking loss
losses = []

# --- Training Loop ---
for epoch in range(1, EPOCHS + 1):
    # Pick random start index
    idx = random.randint(0, len(data) - MAX_SEQ_LEN - 1)
    src_seq = data[idx:idx + MAX_SEQ_LEN]
    tgt_seq = data[idx + 1:idx + MAX_SEQ_LEN + 1]

    # One-hot encode source
    src_input = torch.nn.functional.one_hot(src_seq, num_classes=vocab_size).float().unsqueeze(0).to(DEVICE)

    # Forward pass
    logits = transformer.forward(src_input)

    # Predict next token from final time step
    preds = logits[:, -1, :]
    target = torch.nn.functional.one_hot(tgt_seq[-1].unsqueeze(0), num_classes=vocab_size).float().to(DEVICE)

    # Setup for autograd
    pred_layer = Input(preds.shape)
    pred_layer.set(preds)
    tgt_layer = Input(target.shape)
    tgt_layer.set(target)

    loss_layer = MSELoss(pred_layer, tgt_layer)
    loss_layer.forward()
    loss = loss_layer.output

    # Track loss
    losses.append(loss.item())

    # Zero gradients
    for param in transformer.parameters():
        param.clear_grad()

    # Backward pass
    loss_layer.grad = torch.ones_like(loss)
    loss_layer.backward()

    # Gradient step
    for param in transformer.parameters():
        param.step(LR)

    if epoch % 10 == 0:
        pred_idx = preds.argmax(dim=-1).item()
        tgt_idx = target.argmax(dim=-1).item()
        pred_char = decode([pred_idx])
        tgt_char = decode([tgt_idx])
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | Predicted: '{pred_char}' | Target: '{tgt_char}'")

# --- After Training: Plot Loss ---
plt.figure(figsize=(10,6))
plt.plot(range(1, EPOCHS + 1), losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Training Loss over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss.png")
print("\nSaved loss plot to 'loss.png'.")
