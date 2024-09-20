"""
Filename: causal.py
Description: Implements broad class for CausalLM inference

Notes:

"""

import torch.nn as nn
from collections import OrderedDict

class TransformerArgs():
    def __init__(
        self,
        llm_type = "llama",

        hidden_size: int = -1,

        num_attention_heads: int = -1,
        num_key_value_heads: int = -1
    ):
        self.llm_type = llm_type
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

class CausalLM(nn.Module):
    def __init__(self):
        # Set args like number of attn heads, vocab size, etc.
        

        self.model = nn.ModuleDict(
            OrderedDict({})
        )

if __name__ == "__main__":
    model_args = TransformerArgs(llm_type="llama")