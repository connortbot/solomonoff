"""
=================================================
Solomonoff
Author: Connor Loi, Pranav Bedi, Jonathan Wang
Version: v0.0.0
=================================================
"""

"""
Filename: pipeline.py

Description: User-facing script. Pipeline takes in an input and calls external classes for inference to generate output.

Notes:

"""

"""
?????????????????? TO DO LIST ????????????????????

1. Tokenization [FINISHED]
- decided against external file, will make helpers inside of Pipeline
2. Positional Encodings/Rotary Embeddings [FINISHED]
- only RoPE done, not flat embeddings
3. Attention Mechanism (Scaled Dot-Product Attention) -> (Multi-Head Attention) [FINISHED]
- should double check
4. Layer Normalization [FINISHED]
5. Feed Forward Network [FINISHED]
6. Transformer Block Class (combines all elements of transformer block) [FINISHED]
7. Positional Masking (?)
- fairly sure this is done in Attention already
8. CausalLM [FINISHED]
- Includes output layer
- and embedding layer
11. Greedy Decoding [FINISHED]
12. Token Decoding
13. Pipeline

??????????????????????????????????????????????????
"""

import torch
from transformers import AutoTokenizer

class Pipeline():
    def __init__(
        self,
        model_dir
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )

    def _tokenize_encode(self, prompt):
        token_ids_original = self.tokenizer.encode(prompt)
        # should trim to max prompt length here
        token_ids_original = torch.tensor(token_ids_original, dtype=torch.int64).to("cpu")
        return token_ids_original


if __name__ == "__main__":

    # Model IDs and their paths
    models = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "files/TinyLlama-1.1B-Chat-v1.0/model.safetensors"
    }

    pipeline = Pipeline(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    print(pipeline._tokenize_encode("Hello, my name is Connor!"))