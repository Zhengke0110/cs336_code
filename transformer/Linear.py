import torch
from torch import nn

input_dim = 16384
hiddem_dim = 32
w = nn.Parameter(torch.randn(input_dim, hiddem_dim))
x = nn.Parameter(torch.randn(input_dim))

out = x @ w
print(out)

import numpy as np

w = nn.Parameter(torch.randn(input_dim, hiddem_dim) / np.sqrt(input_dim))

out = x @ w
print(out)
