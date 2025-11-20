from typing import Iterable
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import random


class Linear(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(input_dim, output_dim) / np.sqrt(input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class Cruncher(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Linear(dim, dim) for i in range(num_layers)])
        self.final = Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        x = x.squeeze(-1)
        return x


def get_num_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def get_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")


def custom_model():
    D = 64
    num_layers = 2
    print(f"模型配置: 维度={D}, 层数={num_layers}")

    model = Cruncher(dim=D, num_layers=num_layers)

    # 参数检查
    param_sizes = [(name, param.numel()) for name, param in model.state_dict().items()]
    expected_sizes = [
        ("layers.0.weight", D * D),
        ("layers.1.weight", D * D),
        ("final.weight", D),
    ]

    for (name, size), (_, exp_size) in zip(param_sizes, expected_sizes):
        status = "✓" if size == exp_size else "✗"
        print(f"{status} {name}: {size}")

    num_parameters = get_num_parameters(model)
    expected_params = (D * D) + (D * D) + D
    print(
        f"总参数: {num_parameters:,} {'✓' if num_parameters == expected_params else '✗'}"
    )

    # 前向传播测试
    device = get_device()
    model = model.to(device)
    B = 8
    x = torch.randn(B, D, device=device)
    y = model(x)
    print(
        f"输入{tuple(x.size())} -> 输出{tuple(y.size())} {'✓' if y.size() == torch.Size([B]) else '✗'}"
    )


def get_batch(
    data: np.array, batch_size: int, sequence_length: int, device: str
) -> torch.Tensor:
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))
    assert start_indices.size() == torch.Size([batch_size])

    # 先构建numpy数组，然后一次性转换为tensor（避免性能警告）
    batch_data = np.array(
        [data[start : start + sequence_length] for start in start_indices]
    )
    x = torch.from_numpy(batch_data)
    assert x.size() == torch.Size([batch_size, sequence_length])

    if torch.cuda.is_available():
        x = x.pin_memory()

    x = x.to(device, non_blocking=True)
    return x


def data_loading():
    orig_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    orig_data.tofile("data.npy")

    data = np.memmap("data.npy", dtype=np.int32)
    print(f"数据验证: {'✓' if np.array_equal(data, orig_data) else '✗'}")

    B, L = 2, 4
    x = get_batch(data, batch_size=B, sequence_length=L, device=get_device())
    print(
        f"批次形状: {tuple(x.size())} {'✓' if x.size() == torch.Size([B, L]) else '✗'}"
    )

    print(x)


class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                grad = p.grad.data

                g2 = state.get("g2", torch.zeros_like(grad))

                g2 += torch.square(grad)
                state["g2"] = g2

                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)


def optimizer():
    B = 2
    D = 4
    num_layers = 2
    model = Cruncher(dim=D, num_layers=num_layers).to(get_device())

    optimizer = AdaGrad(model.parameters(), lr=0.01)

    x = torch.randn(B, D, device=get_device())
    y = torch.tensor([4.0, 5.0], device=get_device())
    pred_y = model(x)

    loss = F.mse_loss(input=pred_y, target=y)
    print(f"损失: {loss.item():.4f}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(SGD, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                grad = p.grad.data
                p.data -= lr * grad


def train(
    name: str,
    get_batch,
    D: int,
    num_layers: int,
    B: int,
    num_train_steps: int,
    lr: float,
):
    model = Cruncher(dim=D, num_layers=0).to(get_device())
    optimizer = SGD(model.parameters(), lr=0.01)

    print(f"训练: {name}, lr={lr}, steps={num_train_steps}")
    for t in range(num_train_steps):
        # Get data
        x, y = get_batch(B=B)

        # Forward (compute loss)
        pred_y = model(x)
        loss = F.mse_loss(pred_y, y)

        # Backward (compute gradients)
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if t == 0 or (t + 1) % 5 == 0 or t == num_train_steps - 1:
            print(f"  步骤 {t+1}/{num_train_steps}: 损失 = {loss.item():.4f}")


def train_loop():
    D = 16
    true_w = torch.arange(D, dtype=torch.float32, device=get_device())

    def get_batch(B: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(B, D).to(get_device())
        true_y = x @ true_w
        return (x, true_y)

    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.01)
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.1)


def checkpointing():
    model = Cruncher(dim=64, num_layers=3).to(get_device())
    optimizer = AdaGrad(model.parameters(), lr=0.01)

    print(f"模型参数量: {get_num_parameters(model):,}")
    print("\n模型结构:")
    for name, param in model.state_dict().items():
        print(f"  {name}: {tuple(param.size())}")

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    checkpoint_path = "model_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"\n已保存检查点: {checkpoint_path}")

    loaded_checkpoint = torch.load(checkpoint_path)
    print(f"已加载检查点 ✓")
    print(f"\n检查点内容:")
    print(f"  主键: {list(loaded_checkpoint.keys())}")
    print(f"  模型参数: {len(loaded_checkpoint['model'])} 个")
    print(f"  优化器状态: {list(loaded_checkpoint['optimizer'].keys())}")

    print(f"\n验证加载的参数:")
    for name in list(loaded_checkpoint["model"].keys())[:3]:
        print(f"  {name}: {tuple(loaded_checkpoint['model'][name].size())}")


if __name__ == "__main__":
    # custom_model()
    # data_loading()
    # optimizer()
    # train_loop()
    checkpointing()
