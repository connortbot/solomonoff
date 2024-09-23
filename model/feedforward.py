"""
Filename: feedforward.py

Description: Implementation for FFN sublayer

Notes:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers.transformer_args import TransformerArgs

class FeedForward(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()

        self.llm_type = args.llm_type
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [L, D] --> [L, hidden_D)
        x1, x2 = F.silu(self.gate_proj(x)), self.up_proj(x)
        # [L, hidden_D] --> [L, D]
        return self.down_proj(x1 * x2)