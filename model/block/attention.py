"""
Filename: attention.py

Description: Implements attention sublayer (Multi-Head Attention) and helpers such as GQA scaled dot product attentino.

Notes:

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MultiHeadAttention(nn.Module):
    def __init__(self):
        pass

if __name__ == "__main__":
    pass