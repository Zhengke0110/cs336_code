import os
import time
import random
from BPE import train_bpe, save_vocab


def test_real_dataset():
    # 设置随机种子以确保可复现性
    random.seed(42)
    
    # 数据集路径
    data_path = "./data/TinyStories-valid.txt"

    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print(f"Starting BPE training on {data_path}...")

    # 设置参数
    vocab_size = 1000  # 与 test.py 保持一致
    special_tokens = ["<|endoftext|>", "<pad>", "<unk>"]

    start_time = time.time()

    try:
        vocab, merges = train_bpe(
            input_path=data_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            num_processes=4,  # 根据你的机器核心数调整
            sample_size=2200,  # 与 test.py 保持一致
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nTraining completed in {duration:.2f} seconds.")
        print(f"Final vocab size: {len(vocab)}")
        print(f"Number of merges: {len(merges)}")

        save_dir = "./models/tinystories_vocab"
        print(f"Saving vocab to {save_dir}...")
        save_vocab(vocab, merges, save_dir)

        # 打印一些合并规则示例
        print("\nTop 10 merges:")
        for i, (p1, p2) in enumerate(merges[:10]):
            try:
                s1 = p1.decode("utf-8", errors="replace")
                s2 = p2.decode("utf-8", errors="replace")
                merged = (p1 + p2).decode("utf-8", errors="replace")
                print(f"{i+1}. '{s1}' + '{s2}' -> '{merged}'")
            except:
                print(f"{i+1}. {p1} + {p2} -> (bytes)")

        # 打印词表最后几个词
        print("\nLast 10 tokens in vocab:")
        sorted_vocab = sorted(vocab.items())
        for idx, token_bytes in sorted_vocab[-10:]:
            try:
                token_str = token_bytes.decode("utf-8", errors="replace")
                print(f"ID {idx}: '{token_str}'")
            except:
                print(f"ID {idx}: {token_bytes}")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_real_dataset()
