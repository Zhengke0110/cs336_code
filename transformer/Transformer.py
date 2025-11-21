import torch
from torch import nn
from Modules import Linear, Embedding, TransformerBlock, RMSNorm


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
