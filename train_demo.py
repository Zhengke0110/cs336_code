import json
import torch
import torch.nn as nn
import numpy as np
import os
from transformer.Modules import (
    TransformerLM,
    Adamw,
    CrossEntropyLoss,
    get_batch,
    GradientClip,
)
from BPETokenizer import Tokenizer


def load_tokenizer(vocab_path, merges_path):
    print(f"Loading vocab from {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_str = json.load(f)

    # Convert keys to int and values to bytes
    vocab = {int(k): v.encode("utf-8") for k, v in vocab_str.items()}

    print(f"Loading merges from {merges_path}")
    merges = []
    if os.path.exists(merges_path):
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if not line:
                    continue
                # Split at the last space to separate the two tokens
                head, sep, tail = line.rpartition(" ")
                if sep:
                    t1 = head
                    t2 = tail
                    merges.append((t1.encode("utf-8"), t2.encode("utf-8")))
    else:
        print("Warning: merges.txt not found.")

    return vocab, merges


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    vocab_path = "models/tinystories_vocab/vocab.json"
    merges_path = "models/tinystories_vocab/merges.txt"

    vocab, merges = load_tokenizer(vocab_path, merges_path)
    tokenizer = Tokenizer(vocab, merges)

    data_path = "data/TinyStories-valid.txt"
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Use a small subset for testing
    text = text[:50000]
    print(f"Encoding data (subset length {len(text)})...")
    data_ids = tokenizer.encode(text)
    data_ids = np.array(data_ids, dtype=np.int64)  # Use int64 for torch compatibility

    print(f"Total tokens: {len(data_ids)}")

    # Model Config
    # Ensure vocab_size covers all tokens
    max_id = max(vocab.keys()) if vocab else 0
    vocab_size = max(max_id + 1, 1000)  # Ensure at least some size

    print(f"Vocab size: {vocab_size}")

    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 4
    d_ff = 256
    rope_theta = 10000.0

    print("Initializing model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        weights=None,  # Train from scratch
        device=device,
    ).to(device)

    optimizer = Adamw(model.parameters(), lr=1e-3)

    batch_size = 8
    num_steps = 20

    model.train()
    print("Starting training loop...")
    for step in range(num_steps):
        inputs, targets = get_batch(data_ids, batch_size, context_length, device=device)

        logits = model(inputs)

        B, T, V = logits.shape
        logits_flat = logits.view(B * T, V)
        targets_flat = targets.view(B * T)

        loss_module = CrossEntropyLoss(logits_flat, targets_flat)
        loss = loss_module.forward()

        # Normalize loss
        loss = loss / (B * T)

        optimizer.zero_grad()
        loss.backward()

        clip = GradientClip(model.parameters(), max_l2_norm=1.0)
        clip()

        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print("Training finished successfully.")

    # Save model
    out_path = "model_test.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    train()
