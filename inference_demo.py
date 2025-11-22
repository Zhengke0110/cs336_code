import json
import torch
import os
import numpy as np
from transformer.Modules import TransformerLM, decode
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


def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    vocab_path = "models/tinystories_vocab/vocab.json"
    merges_path = "models/tinystories_vocab/merges.txt"
    model_path = "model_test.pt"

    if not os.path.exists(model_path):
        print(
            f"Error: Model file {model_path} not found. Please run train_demo.py first."
        )
        return

    vocab, merges = load_tokenizer(vocab_path, merges_path)
    tokenizer = Tokenizer(vocab, merges)

    # Model Config (Must match training config)
    max_id = max(vocab.keys()) if vocab else 0
    vocab_size = max(max_id + 1, 1000)

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
        weights=None,
        device=device,
    ).to(device)

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Prompt
    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt}")

    input_ids = tokenizer.encode(prompt)

    print("Generating...")
    # Generate
    output_ids = decode(
        model=model,
        input_tokens=input_ids,
        max_tokens_to_generate=50,
        len_context=context_length,
        temperature=0.8,
        top_p=0.9,
    )

    generated_text = tokenizer.decode(output_ids)
    print(f"\nGenerated Text:\n{generated_text}")


if __name__ == "__main__":
    generate()
