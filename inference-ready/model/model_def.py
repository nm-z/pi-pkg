import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, residual_init: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(1) * float(residual_init))
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + self.alpha * x

class CustomResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, dropout, residual_init: float = 1.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout, residual_init=residual_init) for _ in range(num_blocks)
        ])
        self.output = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)