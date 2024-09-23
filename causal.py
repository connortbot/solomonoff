"""
Filename: causal.py
Description: Implements broad class for CausalLM class

Notes:

"""

import torch
import torch.nn as nn
from collections import OrderedDict

from helpers.transformer_args import TransformerArgs, ARGS_MAP
from model.transformer_block import TransformerBlock
from model.norm import RMSNorm

from helpers.loader import load_model

from pathlib import Path

class CausalLM(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        self.hidden_size = args.hidden_size

        self.num_hidden_layers = args.num_hidden_layers
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = (
            args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads
        )
        self.d_head = self.hidden_size // self.num_attention_heads
        self.intermediate_size = args.intermediate_size
        self.rms_norm_eps = args.rms_norm_eps
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        

        self.model = nn.ModuleDict(
            OrderedDict(
                {
                    "embed_tokens": nn.Embedding(self.vocab_size, self.hidden_size), # Embeddings layer
                    "layers": nn.ModuleList(
                        [TransformerBlock(args) for _ in range(self.num_hidden_layers)]
                    ),
                    "norm": RMSNorm(args)
                }
            )
        )
        # llama has no lm_head bias
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.start_index = 0
    
    def reset(self):
        self.start_index = 0
        return self

    def set_start_index(self, start_index: int):
        self.start_index = start_index
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.ndim == 1 # no batching, shape [L]
        L = tokens.shape

        h = self.model["embed_tokens"](tokens)  # [L] --> [L, D]
        for layer in self.model["layers"]:
            h = layer(h, self.start_index)  # [L, D] --> [L, D]
        h = self.model["norm"](h)  # [L, D] --> [L, D]

        logits = self.lm_head(h)  # [L, D] --> [L, V]

        self.start_index += L
        return logits
    
    @staticmethod
    def from_pretrained(
        model_path: str,
        model_args: TransformerArgs,
        strict=True,
    ) -> "CausalLM":
        state_dict: OrderedDict = load_model(model_path)
        
        # For Llama, convert function
        state_dict_new = OrderedDict()
        for k, v in state_dict.items():
            if "rotary_emb" in k:
                continue
            state_dict_new[k] = v

        model = CausalLM(model_args)
        model.load_state_dict(state_dict_new, strict=strict)
        return model

if __name__ == "__main__":
    model_args = ARGS_MAP["TinyLlama-1.1B-Chat-v1.0"]
    model = CausalLM(model_args)
    with open("./files/tinyllama-causal-structure.txt", "w") as f:
        for k, v in model.state_dict().items():
            f.write(f"{k}: {[*v.shape]}\n")