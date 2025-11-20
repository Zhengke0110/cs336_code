import unittest
import os
import tempfile
from BPE import (
    load_and_sample_data,
    bytes_to_unicode,
    pre_tokenize_document,
    parallel_pre_tokenize,
    BPEIndex,
    train_bpe
)

class TestTrainBPE(unittest.TestCase):
    def test_train_bpe_simple(self):
        """测试 train_bpe 在简单数据集上的表现"""
        # 创建一个临时文件，包含重复模式 "ababab"
        # 期望 BPE 能够学习到 "ab" 这个 token
        content = "ababab ababab"
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        try:
            # 基础词表大小是 256 + 1 (special_token) = 257
            # 我们设置 vocab_size = 260，意味着要学习 3 个新 token
            vocab_size = 260
            vocab, merges = train_bpe(
                input_path=tmp_path,
                vocab_size=vocab_size,
                special_tokens=["<|endoftext|>"],
                num_processes=1, # 测试时用单进程
                sample_size=100
            )
            
            # 验证词表大小
            # 注意：实际学习到的 token 数量可能少于 vocab_size，如果数据中没有更多可合并的 pair
            # 但在这个例子中，"ab" 肯定会被合并
            self.assertTrue(len(vocab) > 257)
            
            # 验证是否学习到了 "ab" (对应的字节序列)
            # 'a' -> b'a', 'b' -> b'b'
            # 'ab' -> b'ab'
            target_token = b'ab'
            self.assertIn(target_token, vocab.values())
            
            # 验证 merges 列表不为空
            self.assertTrue(len(merges) > 0)
            
            # 检查 merges 中是否包含 (b'a', b'b')
            self.assertIn((b'a', b'b'), merges)
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

class TestBPEIndex(unittest.TestCase):
    def test_bpe_index_basic(self):
        """测试 BPEIndex 的基本功能：计数、获取最高频、合并"""
        # 构造简单的序列数据
        # "hug", "pug", "pun", "bun"
        sequences = [
            ['h', 'u', 'g'],
            ['p', 'u', 'g'],
            ['p', 'u', 'n'],
            ['b', 'u', 'n']
        ]
        index = BPEIndex(sequences)
        
        # 检查初始计数
        # ('u', 'g'): 2 (hug, pug)
        # ('u', 'n'): 2 (pun, bun)
        # ('p', 'u'): 2 (pug, pun)
        self.assertEqual(index.pair_counts[('u', 'g')], 2)
        self.assertEqual(index.pair_counts[('u', 'n')], 2)
        self.assertEqual(index.pair_counts[('p', 'u')], 2)
        
        # 获取最高频
        top_pair = index.get_most_frequent()
        self.assertIn(top_pair, [('u', 'g'), ('u', 'n'), ('p', 'u')])
        
        # 测试合并 ('u', 'g') -> 'ug'
        # 假设我们要合并 ('u', 'g') -> 'ug'
        # 注意：如果 top_pair 不是 ('u', 'g')，这里强制测试 ('u', 'g') 的合并逻辑也是可以的
        # 但为了严谨，我们先测试 merge 指定 pair
        merge_count = index.merge(('u', 'g'), 'ug')
        self.assertEqual(merge_count, 2)
        
        # 检查序列更新
        # sequences[0] 应该是 ['h', 'ug']
        self.assertEqual(index.token_sequences[0], ['h', 'ug'])
        # sequences[1] 应该是 ['p', 'ug']
        self.assertEqual(index.token_sequences[1], ['p', 'ug'])
        
        # 检查计数更新
        # ('u', 'g') 应该没了或者计数为0
        self.assertTrue(('u', 'g') not in index.pair_counts or index.pair_counts[('u', 'g')] == 0)
        
        # 新的 pair ('h', 'ug') 应该出现 1 次
        self.assertEqual(index.pair_counts[('h', 'ug')], 1)
        # 新的 pair ('p', 'ug') 应该出现 1 次
        self.assertEqual(index.pair_counts[('p', 'ug')], 1)

    def test_bpe_index_complex_merge(self):
        """测试连续合并的情况 aa -> A"""
        # a a a a -> A A
        sequences = [['a', 'a', 'a', 'a']]
        index = BPEIndex(sequences)
        
        # ('a', 'a') 出现了 3 次: (0,1), (1,2), (2,3)
        self.assertEqual(index.pair_counts[('a', 'a')], 3)
        
        # 合并 ('a', 'a') -> 'A'
        merge_count = index.merge(('a', 'a'), 'A')
        
        # 预期结果: ['A', 'A']，合并次数 2
        # 解释：
        # 1. 处理 (2,3): a a a A (last_merged_pos=2)
        # 2. 处理 (1,2): a a A -> 此时 seq[1]='a', seq[2]='A' != 'a' -> 跳过
        # 3. 处理 (0,1): A A (last_merged_pos=0)
        self.assertEqual(merge_count, 2)
        self.assertEqual(index.token_sequences[0], ['A', 'A'])

    def test_overlapping_merge_odd(self):
        """测试奇数个字符的重叠合并: aaa -> aA (因为是倒序合并)"""
        sequences = [['a', 'a', 'a']]
        index = BPEIndex(sequences)
        index.merge(('a', 'a'), 'A')
        self.assertEqual(index.token_sequences[0], ['a', 'A'])

    def test_merge_updates_counts(self):
        """测试合并后，相邻 pair 的计数更新是否正确"""
        # x a b y
        # 合并 a b -> AB
        # 应该减少 (x, a), (b, y) 的计数
        # 应该增加 (x, AB), (AB, y) 的计数
        sequences = [['x', 'a', 'b', 'y']]
        index = BPEIndex(sequences)
        
        # 初始计数
        self.assertEqual(index.pair_counts[('x', 'a')], 1)
        self.assertEqual(index.pair_counts[('a', 'b')], 1)
        self.assertEqual(index.pair_counts[('b', 'y')], 1)
        
        index.merge(('a', 'b'), 'AB')
        
        # 检查旧的 pair 计数减少
        self.assertEqual(index.pair_counts[('x', 'a')], 0)
        self.assertEqual(index.pair_counts[('b', 'y')], 0)
        # ('a', 'b') 本身会被删除或置0
        self.assertTrue(('a', 'b') not in index.pair_counts or index.pair_counts[('a', 'b')] == 0)
        
        # 检查新的 pair 计数增加
        self.assertEqual(index.pair_counts[('x', 'AB')], 1)
        self.assertEqual(index.pair_counts[('AB', 'y')], 1)

    def test_heap_correctness(self):
        """测试堆能否正确返回最高频 pair"""
        # a b: 3次
        # c d: 2次
        # e f: 1次
        sequences = [
            ['a', 'b'], ['a', 'b'], ['a', 'b'],
            ['c', 'd'], ['c', 'd'],
            ['e', 'f']
        ]
        index = BPEIndex(sequences)
        
        # 第一个应该是 ('a', 'b')
        self.assertEqual(index.get_most_frequent(), ('a', 'b'))
        
        # 模拟合并 ('a', 'b') -> 'AB'
        index.merge(('a', 'b'), 'AB')
        
        # 现在最高频应该是 ('c', 'd')
        self.assertEqual(index.get_most_frequent(), ('c', 'd'))

    def test_multiple_occurrences_in_one_sequence(self):
        """测试同一个序列中多次出现且不重叠的情况"""
        # a b x a b
        sequences = [['a', 'b', 'x', 'a', 'b']]
        index = BPEIndex(sequences)
        
        self.assertEqual(index.pair_counts[('a', 'b')], 2)
        
        index.merge(('a', 'b'), 'AB')
        
        self.assertEqual(index.token_sequences[0], ['AB', 'x', 'AB'])
        self.assertEqual(index.pair_counts[('AB', 'x')], 1)
        self.assertEqual(index.pair_counts[('x', 'AB')], 1)

class TestBPE(unittest.TestCase):
    def setUp(self):
        self.byte_encoder = bytes_to_unicode()

    def test_bytes_to_unicode(self):
        """测试字节到Unicode字符的映射"""
        mapping = self.byte_encoder
        self.assertIsInstance(mapping, dict)
        self.assertEqual(len(mapping), 256)
        # 验证所有字节0-255都在映射中
        for i in range(256):
            self.assertIn(i, mapping)
        # 验证值都是唯一的
        self.assertEqual(len(set(mapping.values())), 256)

    def test_pre_tokenize_document(self):
        """测试单个文档的预分词"""
        doc = "Hello world!"
        # GPT2_SPLIT_PATTERN 会把 "Hello", " world", "!" 分开 (取决于具体正则)
        sequences = pre_tokenize_document(doc, self.byte_encoder)
        
        # 检查返回类型 List[List[str]]
        self.assertIsInstance(sequences, list)
        for seq in sequences:
            self.assertIsInstance(seq, list)
            for char in seq:
                self.assertIsInstance(char, str)
        
        # 简单验证内容不为空
        self.assertTrue(len(sequences) > 0)
        
        # 验证基本的分割逻辑 (Hello 和 world 应该是分开的)
        # 注意：具体的分割取决于正则，这里只验证结构有效性

    def test_parallel_pre_tokenize(self):
        """测试并行预分词"""
        documents = ["Hello world", "Testing BPE", "Python is great"]
        
        # 测试单进程模式 (num_processes=1)
        results_seq = parallel_pre_tokenize(documents, num_processes=1, bytes_to_unicode_map=self.byte_encoder)
        
        # 验证结果是一个列表
        self.assertIsInstance(results_seq, list)
        
        # 测试多进程模式 (num_processes=2)
        # 注意：在某些环境（如macOS）多进程可能需要特殊处理，但在unittest中通常可以
        results_parallel = parallel_pre_tokenize(documents, num_processes=2, bytes_to_unicode_map=self.byte_encoder)
        
        # 结果应该一致
        self.assertEqual(len(results_seq), len(results_parallel))
        # 深度比较内容
        self.assertEqual(results_seq, results_parallel)

    def test_load_and_sample_data(self):
        """测试数据加载和采样"""
        special_token = "<|endoftext|>"
        # 创建包含3个文档的内容
        content = f"Doc1{special_token}Doc2{special_token}Doc3{special_token}"
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        try:
            # 测试加载所有 (sample_size > 实际数量)
            loaded = load_and_sample_data(tmp_path, sample_size=10, special_token=special_token)
            self.assertIn("Doc1", loaded)
            self.assertIn("Doc2", loaded)
            self.assertIn("Doc3", loaded)
            # 检查分隔符数量
            self.assertEqual(loaded.count(special_token), 2) # 3个文档中间有2个分隔符
            
            # 测试采样 (sample_size < 实际数量)
            loaded_sample = load_and_sample_data(tmp_path, sample_size=1, special_token=special_token)
            # 结果应该只包含1个文档，没有分隔符（因为只有一个文档）
            self.assertFalse(special_token in loaded_sample)
            self.assertTrue(len(loaded_sample) > 0)
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
