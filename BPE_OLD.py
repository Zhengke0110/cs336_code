import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Dict, Set
import collections
import os
import re
import json
from tqdm import tqdm


def bytes_to_unicode():
    """把256个字节映射到Unicode字符上，确保每个字节都有对应的可打印字符"""
    # 选出本身就可以打印的字节（188个）
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    # 复制一份数组
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:  # 如果这个字节不可被打印
            bs.append(b)
            cs.append(2**8 + n)  # 映射到256+n的位置的Unicode字符
            n += 1
    cs = [chr(n) for n in cs]  # 将Unicode码位转位实际字符
    return dict(zip(bs, cs))  # 返回字节-字符的映射表


def get_stats(token_sequences: List[List[str]]) -> collections.Counter:
    """计算所有相邻符号对的频率"""
    pairs = collections.Counter()  # 需要加括号来实例化

    for token in token_sequences:
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pairs[pair] += 1
    return pairs


def merge(
    sequences: List[List[str]],
    pair: Tuple[str, str],
    new_token: str,
) -> List[List[str]]:
    """将所有序列中的指定token对合并为新token"""
    result = []
    (t1, t2) = pair
    for seq in sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            # 如果当前位置和下一位置匹配pair，则合并
            if i < len(seq) - 1 and seq[i] == t1 and seq[i + 1] == t2:
                new_seq.append(new_token)
                i += 2  # 跳过已合并的两个token
            else:
                new_seq.append(seq[i])
                i += 1
        result.append(new_seq)
    return result


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    """训练BPE tokenizer，返回词汇表和合并规则"""
    # 建立字节<->Unicode字符的映射
    byte_to_unicode = bytes_to_unicode()
    unicode_to_bytes = {v: bytes([k]) for k, v in byte_to_unicode.items()}

    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must > 0")

    # 初始化词汇表：256个基础字节
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id: int = 256
    existing_bytes: Set[bytes] = set(vocab.values())

    # 添加特殊token到词汇表
    for token in special_tokens:
        if len(vocab) >= vocab_size:
            break
        token_bytes = token.encode("utf-8")
        if token_bytes not in existing_bytes:
            vocab[next_id] = token_bytes
            existing_bytes.add(token_bytes)
            next_id += 1

    # 读取训练文本
    print(f"📖 读取训练文件: {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    print(f"✓ 文件大小: {len(text):,} 字符")

    # 按单词分割文本（保留空格）
    print("🔍 分词处理中...")
    words: List[str] = re.findall(r"\s*\S+", text)
    print(f"✓ 共找到 {len(words):,} 个词")

    # 将每个单词转换为Unicode字符序列
    print("🔄 转换为字节序列...")
    sequences: List[List[str]] = []
    for word in tqdm(words, desc="处理单词", unit="词", ncols=80):
        word_bytes: bytes = word.encode("utf-8")
        if not word_bytes:
            continue
        sequences.append([byte_to_unicode[b] for b in word_bytes])

    merges: List[Tuple[bytes, bytes]] = []

    # 计算需要的merge次数
    num_merges = vocab_size - len(vocab)
    print(f"\n🚀 开始BPE训练")
    print(f"   初始词汇表: {len(vocab)}")
    print(f"   目标词汇表: {vocab_size}")
    print(f"   需要合并: {num_merges} 次\n")

    # 迭代合并最频繁的token对
    pbar = tqdm(total=num_merges, desc="训练进度", unit="merge", ncols=100)

    while len(vocab) < vocab_size:
        if not sequences:
            break

        # 统计所有相邻token对的频率
        pair_counts = get_stats(sequences)
        if not pair_counts:
            break

        # 找出频率最高的token对
        best_pair: Tuple[str, str] = max(pair_counts, key=lambda x: pair_counts[x])
        freq = pair_counts[best_pair]

        # 创建新token（拼接两个Unicode字符）
        new_token: str = best_pair[0] + best_pair[1]

        # 转换为字节表示
        b1 = unicode_to_bytes[best_pair[0]]
        b2 = unicode_to_bytes[best_pair[1]]
        new_bytes: bytes = b1 + b2

        # 更新映射和词汇表
        unicode_to_bytes[new_token] = new_bytes
        vocab[next_id] = new_bytes
        merges.append((b1, b2))

        # 在所有序列中应用这次合并
        sequences = merge(sequences, best_pair, new_token)

        # 更新进度条，显示当前合并的token信息
        try:
            token_display = new_bytes.decode("utf-8", errors="replace")[:20]
        except:
            token_display = str(new_bytes)[:20]
        pbar.set_postfix({"token": token_display, "freq": f"{freq:,}"})
        pbar.update(1)

        next_id += 1

    pbar.close()

    # 保存词汇表到JSON
    print("\n💾 保存词汇表到 vocab.json...")
    with open("vocab.json", "w", encoding="utf-8") as f:
        vocab_dict = {
            token_id: token_bytes.decode("utf-8", errors="replace")
            for token_id, token_bytes in vocab.items()
        }
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)

    # 保存合并规则到文本文件
    print("💾 保存合并规则到 merges.txt...")
    with open("merges.txt", "w", encoding="utf-8") as f:
        for b1, b2 in merges:
            s1 = b1.decode("utf-8", errors="replace")
            s2 = b2.decode("utf-8", errors="replace")
            f.write(f"{s1} {s2}\n")

    print(f"\n✨ 训练完成!")
    print(f"   最终词汇表大小: {len(vocab)}")
    print(f"   合并操作次数: {len(merges)}")

    return vocab, merges


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe("./data/TinyStories-valid.txt", 20000, special_tokens)

    print(vocab[:10])
