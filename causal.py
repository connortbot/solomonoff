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
        llm_type = "llama"
    ):
        self.llm_type = llm_type

class CausalLM(nn.Module):
    def __init__(self):
        # Set args like number of attn heads, vocab size, etc.
        

        self.model = nn.ModuleDict(
            OrderedDict({})
        )

if __name__ == "__main__":
    model_args = TransformerArgs(llm_type="llama")