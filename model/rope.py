"""
Filename: rope.py

Description: Implements rotary embeddings RoPE

Notes:
Needs double checking:
f_complex = self.freqs_complex[start_index : start_index + L].view(L, 1) is probably wrong.
I did a unit test and I don't think the dimensions are right because it tries reshaping from [2,2] to [2,1] for example

"""

import torch
import torch.nn as nn

import sys
from pathlib import Path

# move up one directory level
sys.path.append(str(Path(__file__).parent.parent))

from helpers.transformer_args import TransformerArgs

class RoPE(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()

        self.max_seq_len, self.inv_theta = args.max_seq_len, 1.0 / args.rope_theta
        self.hidden_size = args.hidden_size // args.num_attention_heads
        self.rope_hidden_size = self.hidden_size
        if args.rope_partial_factor is not None:
            self.rope_hidden_size = int(args.rope_partial_factor * self.hidden_size)
        assert self.rope_hidden_size % 2 == 0, f"Hidden Size ({self.rope_hidden_size}) must be divisible by 2"

        # complex freqs of shape: [max_seq_len, rope_hidden_size//2]
        freqs_complex: torch.Tensor = self._precompute_inv_freq()
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)
    
    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        """
        Applies Rotary Position Embeddings to the input tensor (similarly to other sublayers, without batching).

        Args:
            x: Tensor of shape [L, H, D]
            start_index: Starting position index for applying RoPE

        Returns:
            Tensor of shape [L, H, D] with RoPE applied
        """
        assert x.ndim == 3 and x.shape[-1] == self.hidden_size, f"Expected input shape [L, H, {self.hidden_size}], got {x.shape}"
        L, _ , _ = x.shape
        assert start_index + L <= self.max_seq_len, "Sequence length with start_index exceeds max_seq_len"
        if self.hidden_size == self.rope_hidden_size:
            return self._forward(x, start_index)
        else:
            # Apply RoPE to the first rope_hidden_size dimensions, leave the rest untouched
            x_rope, x_pass = x[..., :self.rope_hidden_size].contiguous(), x[..., self.rope_hidden_size:]
            x_rope = self._forward(x_rope, start_index)
            return torch.cat([x_rope, x_pass], dim=-1)
        
    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        assert x.ndim == 3 and x.shape[-1] == self.hidden_size
        L, _, _ = x.shape
        assert start_index + L <= self.max_seq_len
        if self.dim == self.rope_hidden_size:
            return self._forward(x, start_index)
        x, x_pass = x[..., : self.rope_hidden_size].contiguous(), x[..., self.rope_hidden_size :]
        x = self._forward(x, start_index)
        return torch.cat([x, x_pass], dim=-1)
    
    def _forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        """
        Internal method to apply RoPE to the input tensor.

        Args:
            x: Tensor of shape [L, H, rope_hidden_size]
            start_index: Starting position index for applying RoPE

        Returns:
            Tensor of shape [L, H, rope_hidden_size] with RoPE applied
        """
        L, H, rope_hidden_size = x.shape
        # Reshape to [L, H, 2, rope_hidden_size//2]
        x = x.reshape(L, H, 2, -1).transpose(-1, -2).contiguous().float()
        # Convert to complex numbers: [L, H, rope_hidden_size//2]
        x_complex = torch.view_as_complex(x)
        
        # Select frequencies for current positions: [L, rope_hidden_size//2]
        #f_complex = self.freqs_complex[start_index : start_index + L].view(L, 1)

        # this passed the first unit test but unsure if correct
        f_complex = self.freqs_complex[start_index : start_index + L].view(L, 1, -1)


        # Apply rotations: [L, 1] * [L, H, rope_hidden_size//2] -> [L, H, rope_hidden_size//2]
        x_rotated = f_complex * x_complex
        # Convert back to real numbers: [L, H, rope_hidden_size//2, 2]
        x_rotated = torch.view_as_real(x_rotated).transpose(-1, -2)
        # Reshape back to [L, H, rope_hidden_size]
        return x_rotated.reshape(L, H, rope_hidden_size).type_as(x)
    
    def _precompute_inv_freq(self) -> torch.Tensor:
        """
        Precomputes the inverse frequencies used for RoPE.

        Returns:
            Tensor of shape [max_seq_len, rope_hidden_size//2] containing complex freqs
        """
        dtype = torch.float32
        # theta_i = 10000^(-2(i-1)/rope_dim), i = [1, 2, ..., rope_dim/2]
        i = torch.arange(0, self.rope_hidden_size, 2, dtype=dtype)
        inv_freq = self.inv_theta ** (i / self.rope_hidden_size)  # shape: [rope_dim/2]
        t = torch.arange(0, self.max_seq_len, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # shape: [max_seq_len, dim/2]
        # --> exp(j * freqs) = cos(freqs) + j * sin(freqs), complex of shape: [max_seq_len, dim/2]
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex