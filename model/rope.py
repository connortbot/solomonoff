"""
Filename: rope.py

Description: Implements rotary embeddings RoPE

Notes:

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
            self.rope_dim = int(args.rope_partial_factor * self.dim)
        assert self.rope_dim % 2 == 0, f"Dim ({self.rope_dim}) must be divisible by 2"

        # complex freqs of shape: [max_seq_len, rope_dim//2]
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
        assert x.ndim == 3 and x.shape[-1] == self.dim, f"Expected input shape [L, H, {self.dim}], got {x.shape}"
        L, H, _ = x.shape
        assert start_index + L <= self.max_seq_len, "Sequence length with start_index exceeds max_seq_len"
        if self.dim == self.rope_dim:
            return self._forward(x, start_index)
        else:
            # Apply RoPE to the first rope_dim dimensions, leave the rest untouched
            x_rope, x_pass = x[..., :self.rope_dim].contiguous(), x[..., self.rope_dim:]
            x_rope = self._forward(x_rope, start_index)
            return torch.cat([x_rope, x_pass], dim=-1)
    
    def _forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        """
        Internal method to apply RoPE to the input tensor.

        Args:
            x: Tensor of shape [L, H, rope_dim]
            start_index: Starting position index for applying RoPE

        Returns:
            Tensor of shape [L, H, rope_dim] with RoPE applied
        """
        L, H, rope_dim = x.shape
        # Reshape to [L, H, 2, rope_dim//2]
        x = x.reshape(L, H, 2, -1).transpose(-1, -2).contiguous().float()
        # Convert to complex numbers: [L, H, rope_dim//2]
        x_complex = torch.view_as_complex(x)
        # Select frequencies for current positions: [L, rope_dim//2]
        f_complex = self.freqs_complex[start_index : start_index + L].view(L, 1)
        # Apply rotations: [L, 1] * [L, H, rope_dim//2] -> [L, H, rope_dim//2]
        x_rotated = f_complex * x_complex
        # Convert back to real numbers: [L, H, rope_dim//2, 2]
        x_rotated = torch.view_as_real(x_rotated).transpose(-1, -2)
        # Reshape back to [L, H, rope_dim]
        return x_rotated.reshape(L, H, rope_dim).type_as(x)
    
    def _precompute_inv_freq(self) -> torch.Tensor:
        """
        Precomputes the inverse frequencies used for RoPE.

        Returns:
            Tensor of shape [max_seq_len, rope_dim//2] containing complex frequencies
        """

        dtype = torch.float32
        # Indices for frequency calculation: [0, 2, 4, ..., rope_dim-2]
        i = torch.arange(0, self.rope_dim, 2, dtype=dtype)
        # Inverse frequencies: [rope_dim//2]
        inv_freq = self.inv_theta ** (i / self.rope_dim)
        # Positions: [max_seq_len]
        t = torch.arange(0, self.max_seq_len, dtype=dtype)
        # Outer product: [max_seq_len, rope_dim//2]
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Complex exponentials: [max_seq_len, rope_dim//2]
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex


if __name__ == "__main__":
    pass