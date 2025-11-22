from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt, cos, pi
from collections.abc import Callable, Iterable

import numpy as np


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = 2 / (self.in_features + self.out_features) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
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
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)

        variance = x.pow(2).mean(-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.weight

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
        device: torch.device = None,
        dtype: torch.dtype = None,
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
        device: torch.device = None,
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


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


class CrossEntropyLoss:
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets
        self.vocab_size = inputs.shape[1]
        self.batch_size = inputs.shape[0]

    def forward(self):
        y_pred = softmax(self.inputs, dim=1)

        p = y_pred[range(self.batch_size), self.targets]
        return -torch.sum(torch.log(p))


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)


class CausalMulHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: nn.Module = None):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.rope = rope

        max_seq_len = 2048
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )

        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q: torch.Tensor = self.wq(x)
        k: torch.Tensor = self.wk(x)
        v: torch.Tensor = self.wv(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        if self.rope is not None:
            positions = (
                torch.arange(seq_len, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, seq_len)
            )
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        mask = self.causal_mask[:seq_len, :seq_len]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = softmax(attn_scores, dim=-1)
        attn_output = (
            torch.matmul(attn_weights, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        output = self.wo(attn_output)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        w_o: torch.Tensor,
        w_ln1: torch.Tensor,
        w_ln2: torch.Tensor,
        w_ffn_w1: torch.Tensor,
        w_ffn_w2: torch.Tensor,
        w_ffn_w3: torch.Tensor,
        device: torch.device,
    ):
        super(TransformerBlock, self).__init__()
        self.w_q, self.w_k, self.w_v, self.w_o = w_q, w_k, w_v, w_o
        self.w_ln1, self.w_ln2 = w_ln1, w_ln2
        self.w_ffn_w1, self.w_ffn_w2, self.w_ffn_w3 = w_ffn_w1, w_ffn_w2, w_ffn_w3
        self.device = device

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.rms_norm1 = RMSNorm(d_model=d_model, eps=1e-5, device=device)
        self.rms_norm1.load_state_dict({"weight": w_ln1})
        self.rms_norm2 = RMSNorm(d_model=d_model, eps=1e-5, device=device)
        self.rms_norm2.load_state_dict({"weight": w_ln2})

        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.swiglu.load_state_dict(
            {
                "w1.weight": w_ffn_w1,
                "w2.weight": w_ffn_w2,
                "w3.weight": w_ffn_w3,
            }
        )
        self.causal_multi_head_attn = CausalMulHeadAttention(
            d_model=d_model,
            num_heads=n_heads,
            rope=RoPE(
                theta=theta,
                d_k=d_model // n_heads,
                max_seq_len=max_seq_len,
                device=device,
            ),
        )
        self.causal_multi_head_attn.load_state_dict(
            {
                "wq.weight": w_q,
                "wk.weight": w_k,
                "wv.weight": w_v,
                "wo.weight": w_o,
            },
            strict=False,
        )

    def forward(self, x: torch.Tensor):
        h = x + self.causal_multi_head_attn(self.rms_norm1(x))

        # x = x + FFN(Norm2(x))
        out = h + self.swiglu(self.rms_norm2(h))
        return out


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, torch.Tensor],
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights

        # Infer device from weights
        device = list(weights.values())[0].device if weights else None

        # 1. 初始化 Embedding 并加载权重
        self.embedding = Embedding(self.vocab_size, self.d_model)
        self.embedding.load_state_dict(
            {"embedding_matrix": self.weights["token_embedings.weight"]}
        )

        # 2. 初始化 Transformer 层列表
        self.layers = nn.ModuleList()
        for layer in range(self.num_layers):
            w_q = self.weights[f"layers.{layer}.w_q"]
            w_k = self.weights[f"layers.{layer}.w_k"]
            w_v = self.weights[f"layers.{layer}.w_v"]
            w_o = self.weights[f"layers.{layer}.w_o"]
            w_ln1 = self.weights[f"layers.{layer}.w_ln1"]
            w_ln2 = self.weights[f"layers.{layer}.w_ln2"]
            w_ffn_w1 = self.weights[f"layers.{layer}.w_ffn_w1"]
            w_ffn_w2 = self.weights[f"layers.{layer}.w_ffn_w2"]
            w_ffn_w3 = self.weights[f"layers.{layer}.w_ffn_w3"]

            block = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.num_heads,
                d_ff=self.d_ff,
                max_seq_len=self.context_length,
                theta=self.rope_theta,
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                w_o=w_o,
                w_ln1=w_ln1,
                w_ln2=w_ln2,
                w_ffn_w1=w_ffn_w1,
                w_ffn_w2=w_ffn_w2,
                w_ffn_w3=w_ffn_w3,
                device=device,
            )
            self.layers.append(block)

        # 3. 初始化 Final Norm
        self.final_norm = RMSNorm(self.d_model, eps=1e-5)
        self.final_norm.load_state_dict({"weight": self.weights["final_norm.weight"]})

        # 4. 初始化 Head (Linear)
        self.head = Linear(self.d_model, self.vocab_size)
        self.head.weight.data = self.weights["head.weight"]

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(in_indices)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.head(x)

        return logits


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate:{lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                step = state.get("step", 0)
                grad = p.grad.data
                p.data -= lr / sqrt(step + 1) * grad

                state["step"] = step + 1

        return loss


class Adamw(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params=params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m: torch.Tensor = state["m"]
                v: torch.Tensor = state["v"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Ensure types for operations
                beta1 = float(beta1)
                beta2 = float(beta2)
                lr = float(group["lr"])
                weight_decay = float(group["weight_decay"])
                eps = float(group["eps"])

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr / bias_correction1

                denom = (v / bias_correction2).sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)

                p.data.add_(p.data, alpha=-weight_decay * lr)
        return loss


class CosineSchedule:
    def __init__(self, max_lr: float, min_lr: float, warmup: int, cosine_cycle: int):
        self.max_lr, self.min_lr = max_lr, min_lr
        self.warmup, self.cosine_cycle = warmup, cosine_cycle

    def __call__(self, current: int):
        if current < self.warmup:
            return self.max_lr * current / self.warmup
        elif current > self.cosine_cycle:
            return self.min_lr
        else:
            progress = (current - self.warmup) / (self.cosine_cycle - self.warmup)
            cosine_decay = (1 + cos(pi * progress)) / 2
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay


class GradientClip:
    def __init__(self, parameters, max_l2_norm, eps: float = 1e-6):
        self.parameters = parameters
        self.max_l2_norm = max_l2_norm
        self.eps = eps

    def __call__(self):
        valid_gradients = [p.grad for p in self.parameters if p.grad is not None]
        flattened_grads = torch.cat([grad.flatten() for grad in valid_gradients])
        total_norm = torch.norm(flattened_grads, 2)

        if total_norm > self.max_l2_norm:
            clip_coeff = self.max_l2_norm / (total_norm + self.eps)
            for grad in valid_gradients:
                grad.detach().mul_(clip_coeff)


def get_batch(
    x: np.ndarray, batch_size: int, len_context: int, device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(x) > len_context + 1

    max_start = len(x) - len_context - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    inputs = np.stack([x[i : i + len_context] for i in starts])
    targets = np.stack([x[i + 1 : i + len_context + 1] for i in starts])
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
