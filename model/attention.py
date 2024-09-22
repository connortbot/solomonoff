"""
Filename: attention.py

Description: Implements attention sublayer (Multi-Head Attention) and helpers such as GQA scaled dot product attentino.

Notes:

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE

import sys
from pathlib import Path

# move up one directory level
sys.path.append(str(Path(__file__).parent.parent))


from helpers.transformer_args import TransformerArgs

def scaled_dot_product_attention_GQA(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs):
    """
    Simplified scaled dot product attention with support for Grouped Query Attention (GQA).

    Args:
        query: Tensor of shape [n_heads, L, E]
        key: Tensor of shape [n_kv_heads, S, E]
        value: Tensor of shape [n_kv_heads, S, Ev]
        kwargs: Optional arguments like attn_mask, dropout_p, is_causal, scale

    Returns:
        Tensor of shape [n_heads, L, Ev]
    """

    n_heads, L, E = query.shape
    n_kv_heads, S, _ = key.shape
    assert n_heads % n_kv_heads == 0 and L <= S

    attn_mask = kwargs.get("attn_mask", None)
    dropout_p = kwargs.get("dropout_p", 0.0)
    is_causal = kwargs.get("is_causal", False)
    scale = kwargs.get("scale", None)

    # If asked, create causal attention mask
    if is_causal and attn_mask is None:
        attn_mask = torch.tril(torch.ones(L, S, dtype=torch.bool, device=query.device)) # triangular mask

    if n_heads == n_kv_heads:
        # Standard multi-head attention
        output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale
        )
        return output  # Shape: [n_heads, L, Ev]
    else:
        # Grouped Query Attention
        n_heads_per_kv = n_heads // n_kv_heads
        outputs = []
        for i in range(n_kv_heads):
            q_start = i * n_heads_per_kv
            q_end = (i + 1) * n_heads_per_kv
            q_group = query[q_start:q_end, :, :]  # Shape: [n_heads_per_kv, L, E]
            k = key[i:i+1, :, :]  # Shape: [1, S, E]
            v = value[i:i+1, :, :]  # Shape: [1, S, Ev]

            if attn_mask is not None:
                if attn_mask.dim() == 3 and attn_mask.shape[0] == n_heads:
                    # attn_mask is of shape [n_heads, L, S], extract relevant portion
                    mask = attn_mask[q_start:q_end, :, :]
                elif attn_mask.dim() == 2:
                    # attn_mask is of shape [L, S], use as is
                    mask = attn_mask
                else:
                    # Invalid shape
                    raise ValueError("Invalid attention mask shape")
            else:
                mask = None

            # Expand key and value to match the query group's size
            k_expanded = k.expand(n_heads_per_kv, -1, -1)
            v_expanded = v.expand(n_heads_per_kv, -1, -1)

            # Compute attention for this group
            output = F.scaled_dot_product_attention(
                q_group, k_expanded, v_expanded,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=scale
            )
            outputs.append(output)

        # Concatenate outputs from all groups
        output = torch.cat(outputs, dim=0)  # Shape: [n_heads, L, Ev]
        return output


class SelfAttention(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.args = args

        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        self.d_head = args.hidden_size // args.num_attention_heads

        bias = False # for llama

        self.q_proj = nn.Linear(args.hidden_size, self.num_attention_heads * self.d_head, bias=bias)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.d_head, bias=bias)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.d_head, bias=bias)

        self.o_proj = nn.Linear(self.num_attention_heads * self.d_head, args.hidden_size, bias=bias)

        self.kv_cache = KVCache(
            max_seq_len=args.max_seq_len,
            n_kv_heads=self.num_key_value_heads,
            d_head=self.d_head,
        )
        self.rope = RoPE(args)
    
    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        seq_len, _ = x.shape

        if start_index == 0:
            self.kv_cache.reset()
        
        # [L, D] --> [L, D]
        q: torch.Tensor = self.q_proj(x)
        # [L, D] --> [L, D_kv], D_kv may smaller than D
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        # [L, D] --> [L, n_heads, d_head], or [L, H, D]
        q = q.view(seq_len, self.num_attention_heads, self.d_head)
        k = k.view(seq_len, self.num_key_value_heads, self.d_head)
        v = v.view(seq_len, self.num_key_value_heads, self.d_head)

        # apply rotary position embedding
        # [L, H, D] -> [L, H, D]
        q = self.rope(q, start_index)
        k = self.rope(k, start_index)

        # write new key and new value into the kv cache
        self.kv_cache(k, v)
        # read out all cached key and value --> [L_kv, num_key_value_heads, d_head]
        k, v = self.kv_cache()

        # [L, num_attention_heads, d_head] --> [num_attention_heads, L, d_head]
        q = q.permute(1, 0, 2).contiguous()
        # [L_kv, num_key_value_heads, d_head] --> [num_key_value_heads, L_kv, d_head]
        k = k.permute(1, 0, 2).contiguous()
        v = v.permute(1, 0, 2).contiguous()

        # --> [num_attention_heads, L, d_head], if query seq length == 1,
        # set is_causal to False to avoid attention mask construction to save computation
        output = scaled_dot_product_attention_GQA(q, k, v, is_causal=q.shape[1] > 1)
        # [num_attention_heads, L, d_head] --> [L, num_attention_heads, d_head] --> [L, D]
        output = output.permute(1, 0, 2).reshape(seq_len, -1)

        # [L, D] --> [L, D]
        return self.o_proj(output)

# based of feixyz10s implementation but without batching
class KVCache(nn.Module):
    def __init__(self, max_seq_len: int, n_kv_heads: int, d_head: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head

        # Initialize caches with shape [max_seq_len, n_kv_heads, d_head]
        kv_cache_shape = (self.max_seq_len, self.n_kv_heads, self.d_head)
        self.register_buffer("k_cache", torch.zeros(kv_cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(kv_cache_shape), persistent=False)

        self.seq_len = 0  # Sequence length that has been cached

    def reset(self):
        self.seq_len = 0

    def forward(self, k: torch.Tensor = None, v: torch.Tensor = None):
        if k is not None and v is not None:
            self._write(k, v)
            return
        assert k is None and v is None
        assert self.seq_len > 0, "Nothing has been cached"
        return self._read()

    def _write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        assert k.ndim == 3 and k.shape == v.shape, "k and v must be 3D tensors of the same shape"
        assert k.shape[1] == self.n_kv_heads and k.shape[2] == self.d_head, \
            f"Expected k/v shape [L, {self.n_kv_heads}, {self.d_head}], got {k.shape}"
        assert self.seq_len + k.shape[0] <= self.max_seq_len, "Exceeds maximum sequence length"

        # Write the new k and v to the caches
        self.k_cache[self.seq_len : self.seq_len + k.shape[0]] = k
        self.v_cache[self.seq_len : self.seq_len + k.shape[0]] = v
        self.seq_len += k.shape[0]
    
    def _read(self):
        # Return the cached keys and values up to the current sequence length
        k = self.k_cache[: self.seq_len]
        v = self.v_cache[: self.seq_len]
        return k, v

if __name__ == "__main__":
    # scaled_dot_product_attention_GQA Test/Usage
    n_heads = 8
    n_kv_heads = 2
    L = 10  # Length of the query sequence
    S = 15  # Length of the key/value sequence
    E = 64  # Embedding dimension
    Ev = 64

    # Random tensors
    query = torch.randn(n_heads, L, E)
    key = torch.randn(n_kv_heads, S, E)
    value = torch.randn(n_kv_heads, S, Ev)

    # Compute attention
    output = scaled_dot_product_attention_GQA(query, key, value)
    print(output.shape)  # Should be [n_heads, L, Ev]

    # KVCache Test/Usage
    max_seq_len = 100
    n_kv_heads = 8
    d_head = 64
    kv_cache = KVCache(max_seq_len, n_kv_heads, d_head)

    # Suppose you have new keys and values to cache
    L = 10  # Length of the new sequence chunk
    k_new = torch.randn(L, n_kv_heads, d_head)
    v_new = torch.randn(L, n_kv_heads, d_head)

    # Write to the cache
    kv_cache(k_new, v_new)

    # Read from the cache
    cached_k, cached_v = kv_cache()

    # Check the shapes
    print(cached_k.shape)  # Should be [L, n_kv_heads, d_head]
    print(cached_v.shape)  # Should be [L, n_kv_heads, d_head]