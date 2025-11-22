import os
import time
import random
from BPE import train_bpe, save_vocab

def train_bpe_full():
    # 设置随机种子
    random.seed(42)
    
    # 数据集路径 (请确保文件在您的强力电脑上存在)
    data_path = "data/TinyStories-train.txt"
    
    # 输出目录
    save_dir = "models/tinystories_vocab_full"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please ensure you have downloaded TinyStories-train.txt to the data folder.")
        return

    print(f"Starting BPE training on {data_path}...")
    print(f"Output directory: {save_dir}")

    # 针对大数据集的配置
    vocab_size = 10000  # 增加词表大小
    special_tokens = ["<|endoftext|>", "<pad>", "<unk>"]
    
    # 增加采样行数以获得更准确的统计信息
    # 如果内存足够，可以适当增加 sample_size
    sample_size = 100000 

    start_time = time.time()

    try:
        vocab, merges = train_bpe(
            input_path=data_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            num_processes=8,  # 强力电脑通常核心更多，可以设为 8 或更高
            sample_size=sample_size,
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nTraining completed in {duration:.2f} seconds.")
        print(f"Final vocab size: {len(vocab)}")
        print(f"Number of merges: {len(merges)}")

        print(f"Saving vocab to {save_dir}...")
        save_vocab(vocab, merges, save_dir)
        print("Done!")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_bpe_full()
