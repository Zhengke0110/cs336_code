import os
import heapq
import regex
import time
import random
import multiprocessing
import json

from functools import partial
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict, Any, Union

import mmap
import re
from collections import defaultdict


# GPT-2预分词模式
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def load_and_sample_data(
    file_path: str, sample_size: int = 22000, special_token: str = "<|endoftext|>"
) -> str:

    try:
        # 以读写模式打开文件，忽略编码错误
        with open(file_path, "r+", encoding="utf-8", errors="ignore") as file:
            # 使用 mmap 进行内存映射，提高读取效率
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped_file:
                documents = []
                start_idx = 0
                special_token_bytes = special_token.encode("utf-8")
                # 注意：这里应该使用字节长度，因为 mmap 操作的是字节
                token_byte_len = len(special_token_bytes)

                while start_idx < len(mapped_file):
                    # 查找下一个分隔符的位置
                    end_idx = mapped_file.find(special_token_bytes, start_idx)

                    if end_idx == -1:
                        # 处理最后一个文档
                        content = (
                            mapped_file[start_idx:]
                            .decode("utf-8", errors="replace")
                            .strip()
                        )
                        if content:
                            documents.append(content)
                        break

                    # 解码当前文档内容
                    content = (
                        mapped_file[start_idx:end_idx]
                        .decode("utf-8", errors="replace")
                        .strip()
                    )
                    if content:
                        documents.append(content)

                    # 更新起始位置
                    start_idx = end_idx + token_byte_len

                # 如果文档数量超过采样大小，则进行随机采样
                if len(documents) > sample_size:
                    documents = random.sample(documents, sample_size)

                return special_token.join(documents)

    except Exception as e:
        raise IOError(f"load datasets error: {e}")


def bytes_to_unicode() -> Dict[int, str]:
    # 构建字节到 Unicode 字符的映射，用于处理所有可能的字节值。
    # 初始包含可打印字符的字节值
    printable_bytes = (
        list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    )

    byte_values = printable_bytes[:]
    char_codepoints = printable_bytes[:]
    offset = 0

    # 处理剩余的不可打印字节，映射到 256 之后的 Unicode 码点
    for byte_val in range(256):
        if byte_val not in printable_bytes:
            byte_values.append(byte_val)
            char_codepoints.append(256 + offset)
            offset += 1

    return {b: chr(c) for b, c in zip(byte_values, char_codepoints)}


def pre_tokenize_document(
    doc: str, bytes_to_unicode_map: Dict[int, str]
) -> List[List[str]]:
    tokens = regex.findall(GPT2_SPLIT_PATTERN, doc, flags=regex.UNICODE)
    sequences = []
    for token in tokens:
        token_unicode = "".join(bytes_to_unicode_map[b] for b in token.encode("utf-8"))
        sequences.append(list(token_unicode))

    return sequences


# 全局变量用于多进程
global_worker_byte_map = None


def init_worker(byte_map: Dict[int, str]):
    global global_worker_byte_map
    global_worker_byte_map = byte_map


def pre_tokenize_worker(doc: str) -> List[List[str]]:
    return pre_tokenize_document(doc, global_worker_byte_map)


def parallel_pre_tokenize(
    documents: List[str], num_processes: int, bytes_to_unicode_map: Dict[int, str]
) -> list[list[str]]:
    if num_processes <= 1:
        all_sequences = []
        for doc in documents:
            sequences = pre_tokenize_document(doc, bytes_to_unicode_map)
            all_sequences.extend(sequences)
        return all_sequences

    with multiprocessing.Pool(
        num_processes, initializer=init_worker, initargs=(bytes_to_unicode_map,)
    ) as pool:

        results = list(
            tqdm(
                pool.imap(pre_tokenize_worker, documents, chunksize=50),
                total=len(documents),
                desc="预分词",
                mininterval=1,
            )
        )
    return [seq for doc_sequences in results for seq in doc_sequences]


class BPEIndex:
    def __init__(self, token_sequences: List[List[str]]):
        self.token_sequences = token_sequences
        self.pair_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.pair_locations: DefaultDict[Tuple[str, str], List[Tuple[int, int]]] = (
            defaultdict(list)
        )

        self.max_heap = []  # Max heap

        for seq_id, seq in enumerate(
            tqdm(self.token_sequences, desc="Building BPE Index", mininterval=1)
        ):
            for token_idx in range(len(seq) - 1):
                pair = (seq[token_idx], seq[token_idx + 1])
                self.pair_counts[pair] += 1
                self.pair_locations[pair].append((seq_id, token_idx))

        for pair, count in tqdm(
            self.pair_counts.items(), desc="Building Heap", mininterval=1
        ):
            if count > 1:
                entry = [-count, pair]
                heapq.heappush(self.max_heap, entry)

    def _update_pair_count(self, token_pair: Tuple[str, str], count_delta: int):
        if count_delta == 0:
            return
        if token_pair not in self.pair_counts:
            self.pair_counts[token_pair] = 0
        new_count = self.pair_counts[token_pair] + count_delta
        self.pair_counts[token_pair] = new_count

        if new_count < 0:
            new_count = 0
            self.pair_counts[token_pair] = 0

        # 懒更新：直接推入新计数，旧的无效条目会在 get_most_frequent 中被过滤
        if new_count > 1:
            entry = [-new_count, token_pair]
            heapq.heappush(self.max_heap, entry)

    def _add_position(self, token_pair: Tuple[str, str], seq_id: int, token_idx: int):
        self.pair_locations[token_pair].append((seq_id, token_idx))

    def get_most_frequent(self) -> Tuple[str, str]:
        while self.max_heap:
            neg_count, pair = self.max_heap[0]
            current_count = self.pair_counts.get(pair, 0)

            # 检查堆顶元素是否有效（计数是否匹配）
            if -neg_count == current_count:
                if current_count > 1:
                    return pair
                else:
                    # 计数 <= 1，不再需要合并
                    heapq.heappop(self.max_heap)
            else:
                # 过期条目（计数已改变），丢弃
                heapq.heappop(self.max_heap)
        return None

    def merge(self, target_pair: Tuple[str, str], new_token: str) -> int:
        if (
            target_pair not in self.pair_locations
            or not self.pair_locations[target_pair]
        ):
            return 0
        indices_by_seq_id = defaultdict(list)
        for seq_id, token_idx in self.pair_locations[target_pair]:
            indices_by_seq_id[seq_id].append(token_idx)

        merge_count = 0
        for seq_id, token_indices in indices_by_seq_id.items():
            seq = self.token_sequences[seq_id]

            token_indices.sort(reverse=True)  # 倒序
            last_merged_idx = -2

            for token_idx in token_indices:
                # 检查索引是否越界（因为序列变短了）
                if token_idx >= len(seq) - 1 or token_idx == last_merged_idx - 1:
                    continue
                if (
                    seq[token_idx] != target_pair[0]
                    or seq[token_idx + 1] != target_pair[1]
                ):
                    continue
                seq[token_idx] = new_token
                del seq[token_idx + 1]
                merge_count += 1
                last_merged_idx = token_idx

                if token_idx > 0:
                    left_pair = (seq[token_idx - 1], target_pair[0])
                    self._update_pair_count(left_pair, -1)
                    new_left_pair = (seq[token_idx - 1], new_token)
                    self._update_pair_count(new_left_pair, 1)
                    self._add_position(new_left_pair, seq_id, token_idx - 1)

                if token_idx < len(seq) - 1:
                    right_pair = (target_pair[1], seq[token_idx + 1])
                    self._update_pair_count(right_pair, -1)

                    new_right_pair = (new_token, seq[token_idx + 1])
                    self._update_pair_count(new_right_pair, 1)
                    self._add_position(new_right_pair, seq_id, token_idx)

        if target_pair in self.pair_counts:
            del self.pair_counts[target_pair]
        if target_pair in self.pair_locations:
            del self.pair_locations[target_pair]

        return merge_count


def train_bpe(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: List[str] = ["<|endoftext|>"],
    num_processes: int = 8,
    sample_size: int = 22000,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    base_vocab_size = 256 + len(special_tokens)
    if vocab_size < base_vocab_size:
        raise ValueError(f"vocab_size must > {base_vocab_size}")

    # 1.字节 -> Unicode 映射
    bytes_to_unicode_map = bytes_to_unicode()

    unicode_to_bytes_map = {v: bytes([k]) for k, v in bytes_to_unicode_map.items()}

    # 2.初始化词汇表
    vocab = {i: bytes([i]) for i in range(256)}

    next_token_id = 256
    existing_bytes = set(vocab.values())

    # 3.添加特殊token
    for special_token in special_tokens:
        special_tokens_bytes = special_token.encode("utf-8")
        if special_tokens_bytes not in existing_bytes and len(vocab) < vocab_size:
            vocab[next_token_id] = special_tokens_bytes
            existing_bytes.add(special_tokens_bytes)
            next_token_id += 1

    # 4.加载采样数据
    text = load_and_sample_data(
        file_path=input_path, sample_size=sample_size, special_token=special_tokens[0]
    )
    # 5.分割文档
    escaped_tokens = [re.escape(special_token) for special_token in special_tokens]
    split_pattern = "|".join(escaped_tokens)
    documents = [part for part in re.split(split_pattern, text) if part]

    # 6.并行预分词
    sequences = parallel_pre_tokenize(documents, num_processes, bytes_to_unicode_map)

    # 7.构建初始化索引
    bpe_index = BPEIndex(sequences)

    merges = []
    vocab_progress = len(vocab)

    total_merges = vocab_size - vocab_progress

    progress_bar = tqdm(
        total=total_merges, desc="Training BPE", unit="merge", mininterval=0.5
    )

    while vocab_progress < vocab_size:
        best_pair = bpe_index.get_most_frequent()
        if best_pair is None:
            break
        new_token_str = best_pair[0] + best_pair[1]

        # 确保我们能找到对应的字节表示
        # 检查合并对是否在映射中（理论上一定在）
        if (
            best_pair[0] not in unicode_to_bytes_map
            or best_pair[1] not in unicode_to_bytes_map
        ):
            continue

        p1_bytes = unicode_to_bytes_map[best_pair[0]]
        p2_bytes = unicode_to_bytes_map[best_pair[1]]
        new_token_bytes = p1_bytes + p2_bytes

        merge_count = bpe_index.merge(best_pair, new_token_str)

        if merge_count == 0:
            continue

        if new_token_bytes not in existing_bytes:
            vocab[next_token_id] = new_token_bytes
            existing_bytes.add(new_token_bytes)
            merges.append((p1_bytes, p2_bytes))
            next_token_id += 1
            vocab_progress += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"vocab_size": vocab_progress})

        # 更新映射，以便后续合并可以使用
        unicode_to_bytes_map[new_token_str] = new_token_bytes

    progress_bar.close()
    return vocab, merges
