import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.w = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = 2 / (self.in_features + self.out_features) ** 0.5
        nn.init.trunc_normal_(self.w, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.embedding_matrix = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        std = 1 / sqrt(embedding_dim)
        nn.init.trunc_normal_(self.embedding_matrix, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)

        variance = x.pow(2).mean(-1, keepdim=True)

        # x = x * torch.rsqrt(variance + self.eps)
        x *= torch.rsqrt(variance + self.eps)
        # x = x * self.weight
        x *= self.weight

        return x.to(input_dtype)


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) 激活函数层。
    参考文献: "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)

    公式: SwiGLU(x) = (xW_1 * Swish(xW_3))W_2
    其中 Swish(x) = x * Sigmoid(x) (在 PyTorch 中通常使用 F.silu)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # W1: 门控机制的一部分，将输入维度 d_model 映射到隐藏层维度 d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        # W2: 输出投影，将隐藏层维度 d_ff 映射回输入维度 d_model
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        # W3: 门控机制的另一部分，同样将输入维度 d_model 映射到 d_ff
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. self.w1(x): 线性变换
        # 2. self.w3(x): 另一路线性变换
        # 3. F.silu(...): 对 w1 的输出应用 SiLU (Swish) 激活函数
        # 4. * : 逐元素相乘 (Gating)
        # 5. self.w2(...): 最后通过 W2 投影回原维度
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) 旋转位置编码。
    参考文献: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (https://arxiv.org/abs/2104.09864)

    RoPE 通过将 token 的 query 和 key 向量在复数平面上旋转一个角度来实现位置编码。
    旋转角度由 token 的位置和维度索引决定。
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 计算频率: theta^(-2i/d)
        # i 取值范围 [0, 2, ..., d_k-2]
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        # 生成位置索引 [0, 1, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device)
        # 计算外积得到每个位置在每个频率上的角度: m * theta_i
        sinusoids = torch.outer(positions, freqs)

        # 预计算 cos 和 sin 值并缓存
        # 形状: (max_seq_len, d_k/2)
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False)
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        应用 RoPE 到输入张量 x。

        Args:
            x: 输入张量，形状通常为 (batch_size, seq_len, num_heads, head_dim) 或 (batch_size, seq_len, head_dim)
               注意：这里的实现假设 x 的最后一个维度是 head_dim (即 d_k)
            token_positions: 每个 token 的位置索引，形状为 (batch_size, seq_len)

        Returns:
            应用位置编码后的张量，形状与 x 相同。
        """
        # 获取对应位置的 cos 和 sin 值 cos/sin shape: (batch_size, seq_len, d_k/2)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # 调整 cos 和 sin 的形状以支持广播
        if x.ndim > 3:
            # 计算需要插入的维度数量 x: (B, S, ..., D) -> ndim, cos: (B, S, D/2) -> 3 dims,需要插入 ndim - 3 个维度
            view_shape = list(cos.shape)
            for _ in range(x.ndim - 3):
                view_shape.insert(2, 1)
            cos = cos.view(view_shape)
            sin = sin.view(view_shape)

        # 将输入 x 分为偶数索引和奇数索引部分
        x_even = x[..., 0::2]  # x_even: [x_0, x_2, ...]
        x_odd = x[..., 1::2]  # x_odd:  [x_1, x_3, ...]

        # 执行旋转操作
        # [x_even, x_odd] * [cos, -sin; sin, cos]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # 重新组合偶数和奇数部分
        # stack 后 shape: (..., d_k/2, 2) -> flatten -> (..., d_k)
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)

        return out
