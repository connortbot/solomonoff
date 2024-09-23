"""
Filename: transformer_block.py

Description: TransformerBlock holding Norm->Attn->Norm->FFN

Notes:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import SelfAttention
from model.norm import RMSNorm
from model.feedforward import FeedForward

import sys
from pathlib import Path

# move up one directory level
sys.path.append(str(Path(__file__).parent.parent))


from helpers.transformer_args import TransformerArgs

class TransformerBlock(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.args = args

        self.self_attn = SelfAttention(args)
        self.mlp = FeedForward(args)
        self.input_layernorm = RMSNorm(args)
        self.post_attention_layernorm = RMSNorm(args)

    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), start_index)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x