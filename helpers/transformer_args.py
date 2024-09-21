"""
Filename: transformer_args.py
Description: Struct for arguments for transformer architecture

Notes:

"""

class TransformerArgs():
    def __init__(
        self,
        llm_type = "llama",

        hidden_size: int = -1,
        max_seq_len: int = 2048, # tinyllamas max length
        num_attention_heads: int = -1,
        num_key_value_heads: int = -1
    ):
        self.llm_type = llm_type
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads