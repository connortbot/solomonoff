"""
Filename: norm.py

Description: Implementation for Transformers normalization sublayer

Notes:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# move up one directory level
sys.path.append(str(Path(__file__).parent.parent))


from helpers.transformer_args import TransformerArgs

class RMSNorm(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.llm_type = args.llm_type
        self.hidden_size, self.eps = args.hidden_size, args.rms_norm_eps

        self.weight = nn.Parameter(torch.ones(self.hidden_size))
    
    def forward(self, x) -> torch.Tensor:
        """
        RMSNorm

        Args:
            x: Tensor of shape [L, D]

        Returns:
            Tensor of shape [L, D] after RMS normalization
        """
        # [L, D] --> [L, D]
        x = self._norm(x.float()).type_as(x)
        
        # [D] * [L, D] --> [L, D]
        return self.weight * x

    def _norm(self, x):
        # compute mean of squared values on last dimension D
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)