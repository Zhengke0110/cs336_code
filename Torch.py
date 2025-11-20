from jaxtyping import Float
import torch
from einops import rearrange, einsum, reduce

x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
y: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)

z = x @ y.transpose(-2, -1)
print(z.shape)

z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
print(z.shape)

z = einsum(x, y, "... seq1 hidden, ... seq2 hidden->... seq1 seq2")
print(z.shape)

print("=" * 60)
y = x.mean(dim=-1)
print(y)

y = reduce(x, "... hidden -> ...", "sum")
print(y)
print("=" * 60)


x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)
print(x.shape)
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)

print(x.shape)
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
print(x.shape)

x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
print(x.shape)
