"""
Filename: transformer_args.py
Description: Struct for arguments for transformer architecture

Notes:

"""

class TransformerArgs():
    def __init__(
        self,
        llm_type = "llama",
        vocab_size: int = -1,

        hidden_size: int = -1,
        num_hidden_layers: int = -1,
        max_batch_size: int = 1,
        max_seq_len: int = 2048, # tinyllamas max length
        
        # Attention
        num_attention_heads: int = -1,
        num_key_value_heads: int = -1,

        # RoPe
        rope_theta: float = 10000.0,
        rope_partial_factor = None,

        # Norm
        rms_norm_eps: float = 1e-5,

        # Feed Forward
        intermediate_size: int = -1
    ):
        self.llm_type = llm_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.rope_partial_factor = rope_partial_factor
        self.rms_norm_eps = rms_norm_eps
        self.intermediate_size = intermediate_size


ARGS_MAP = {
    "TinyLlama-1.1B-Chat-v1.0": TransformerArgs(
        llm_type="llama",
        vocab_size=32000,
        hidden_size=2048,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=4,
        intermediate_size=5632,
        max_batch_size=1,
        max_seq_len=2048,
    )
}