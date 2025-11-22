import json
import torch
import torch.nn as nn
import numpy as np
import os
import time
from tqdm import tqdm
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

    vocab = {int(k): v.encode("utf-8") for k, v in vocab_str.items()}

    print(f"Loading merges from {merges_path}")
    merges = []
    if os.path.exists(merges_path):
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if not line:
                    continue
                head, sep, tail = line.rpartition(" ")
                if sep:
                    merges.append((head.encode("utf-8"), tail.encode("utf-8")))
    return vocab, merges


def get_data(tokenizer, data_path, cache_path):
    """
    加载数据。如果存在缓存的 .npy 文件则直接加载，否则读取文本、分词并保存缓存。
    """
    if os.path.exists(cache_path):
        print(f"Loading tokenized data from cache: {cache_path}")
        return np.load(cache_path, mmap_mode="r")  # 使用 mmap 节省内存

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Reading text from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Tokenizing text (length: {len(text)})... This may take a while.")
    data_ids = tokenizer.encode(text)

    print(f"Saving tokenized data to {cache_path}...")
    data_ids_np = np.array(
        data_ids, dtype=np.uint16
    )  # 使用 uint16 节省空间 (如果 vocab_size < 65535)
    np.save(cache_path, data_ids_np)

    return data_ids_np


def train():
    # --- 配置 ---
    # 路径
    data_path = "data/TinyStories-train.txt"
    cache_path = "data/TinyStories-train.npy"
    vocab_dir = "models/tinystories_vocab_full"
    vocab_path = os.path.join(vocab_dir, "vocab.json")
    merges_path = os.path.join(vocab_dir, "merges.txt")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 模型参数 (针对更强的电脑)
    context_length = 512  # 上下文长度
    d_model = 512  # 嵌入维度
    num_layers = 8  # 层数
    num_heads = 8  # 注意力头数
    d_ff = 2048  # 前馈网络维度
    rope_theta = 10000.0

    # 训练参数
    batch_size = 32  # 根据显存调整
    lr = 3e-4
    num_steps = 10000  # 训练步数
    save_interval = 1000  # 保存间隔
    log_interval = 100  # 日志间隔

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # --- 准备 ---
    # 1. 加载分词器
    if not os.path.exists(vocab_path):
        print(
            f"Error: Vocab not found at {vocab_path}. Please run run_bpe_full.py first."
        )
        return
    vocab, merges = load_tokenizer(vocab_path, merges_path)
    tokenizer = Tokenizer(vocab, merges)

    vocab_size = max(max(vocab.keys()) + 1, 1000)
    print(f"Vocab size: {vocab_size}")

    # 2. 加载数据
    data = get_data(tokenizer, data_path, cache_path)
    print(f"Total tokens: {len(data)}")

    # 3. 初始化模型
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

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = Adamw(model.parameters(), lr=lr, weight_decay=0.1)

    # --- 训练循环 ---
    model.train()
    print("Starting training loop...")

    pbar = tqdm(range(1, num_steps + 1), desc="Training")
    start_time = time.time()

    for step in pbar:
        inputs, targets = get_batch(data, batch_size, context_length, device=device)

        logits = model(inputs)

        B, T, V = logits.shape
        logits_flat = logits.view(B * T, V)
        targets_flat = targets.view(B * T)

        loss = CrossEntropyLoss(logits_flat, targets_flat).forward()
        loss = loss / (B * T)  # Normalize

        optimizer.zero_grad()
        loss.backward()

        GradientClip(model.parameters(), max_l2_norm=1.0)()
        optimizer.step()

        # 日志
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (batch_size * context_length * log_interval) / elapsed
            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Tokens/s": f"{tokens_per_sec:.2f}"}
            )
            start_time = time.time()

        # 保存检查点
        if step % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
            tqdm.write(f"Saving checkpoint to {ckpt_path}...")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                ckpt_path,
            )

    print("Training finished.")
    final_path = "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    train()
