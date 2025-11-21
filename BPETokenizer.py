import regex
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Set, Tuple

import torch

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges, special_tokens=None):
        # 初始化词汇表映射：ID -> Token (bytes)
        self.id_to_token = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # 构建合并规则优先级映射：Pair -> Priority (index)
        # 索引越小，优先级越高
        self.merges_priority_map = {pair: i for i, pair in enumerate(self.merges)}

        # 初始化反向词汇表映射：Token (bytes) -> ID
        self.token_to_id = {v: k for k, v in vocab.items()}

    def _merge_tokens(self, token_bytes: bytes) -> List[bytes]:
        # 将输入的字节序列转换为初始的 token 列表（每个字节一个 token）
        tokens = [bytes([b]) for b in token_bytes]

        while len(tokens) > 1:
            # 找出当前 token 列表中所有可能的相邻对
            pairs = set()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges_priority_map:
                    pairs.add(pair)

            if not pairs:
                break

            # 找到优先级最高的合并对（在 merges 中索引最小的）
            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])

            new_tokens = []
            i = 0

            # 执行合并操作
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        # 处理特殊 token，确保长 token 优先匹配
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_token_pattern = "|".join(map(regex.escape, sorted_special_tokens))

        # 使用特殊 token 分割文本
        if self.special_tokens:
            text_chunks = regex.split(f"({special_token_pattern})", text)
        else:
            text_chunks = [text]

        final_ids = []
        for chunk in text_chunks:
            if not chunk:
                continue
            # 如果是特殊 token，直接查找 ID
            if chunk in self.special_tokens:
                final_ids.append(self.token_to_id[chunk.encode("utf-8")])
            else:
                # 对普通文本使用 GPT-2 的正则模式进行预分词
                for text_token in regex.findall(PAT, chunk):
                    if not text_token:
                        continue
                    # 对每个预分词后的片段应用 BPE 合并
                    bpe_tokens = self._merge_tokens(text_token.encode("utf-8"))

                    # 将合并后的 token 转换为 ID
                    for bpe_token in bpe_tokens:
                        final_ids.append(self.token_to_id[bpe_token])
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text=text)

    def decode(self, ids):
        # 将 ID 列表转换回字节序列
        all_bytes = b"".join(self.id_to_token[id] for id in ids)
        # 将字节序列解码为 UTF-8 字符串，忽略错误
        return all_bytes.decode("utf-8", errors="replace")
