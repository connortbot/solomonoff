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

import os
import platform

import torch
from transformers import AutoTokenizer

from causal import CausalLM
from model.samplers import SamplerBase

from helpers.transformer_args import ARGS_MAP

class Pipeline():
    def __init__(
        self,
        model: CausalLM,
        tokenizer: AutoTokenizer,
        model_name: str = None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
    
    # Tokenizer
    def _tokenize_encode(self, prompt):
        token_ids_original = self.tokenizer.encode(prompt)
        if len(token_ids_original) > 512: # max_prompt_length
            token_ids_original = token_ids_original[-512 :]
        token_ids_original = torch.tensor(token_ids_original, dtype=torch.int64).to("cpu")
        return token_ids_original
    
    @torch.inference_mode()
    def _generate(
        self,
        prompt: str,
        history = None, # array of tuples (out, response?)
        # config = None, (config for users, paramsl like tmep, etc.)
        device: torch.device = None,
    ):
        prompt = self._preprocess(prompt, history)

        input_token_ids = self._tokenize_encode(prompt)

        def construct_output(output_token_ids): # takes list of ints
            out_all = self.tokenizer.decode(
                [*input_token_ids, *output_token_ids], skip_special_tokens=True
            )
            out = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)
            return out_all, out

        self.model = self.model.to(device).eval().reset()
        sampler = SamplerBase() # if config, then pass in temp and top_k
        for out in generate_yield(
            self.model,
            input_token_ids,
            sampler,
            100, # max_length of response
            eos_token_id=self.tokenizer.eos_token_id,
        ):
            yield construct_output(out)
    
    def _preprocess(self, prompt: str, history=None, system_prompt=None) -> str:
        user_prompt = prompt.strip()
        sys_prompt = "You are a chatbot who can help answer questions."
        if system_prompt is not None:
            sys_prompt = system_prompt.strip()
        if history is None or len(history) == 0:
            return f"<|system|>\n{sys_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
        prompt = f"<|system|>\n{sys_prompt}</s>\n"
        for _, (u, r) in enumerate(history):
            prompt += f"<|user|>\n{u}</s>\n<|assistant|>\n{r}</s>\n"
        prompt += f"<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
        return prompt



@torch.inference_mode()
def generate_yield(
    model: CausalLM,
    prompt: torch.Tensor,  # [L]
    sampler: SamplerBase,
    max_length: int,
    eos_token_id: int
):
    assert prompt.ndim <= 1
    L = prompt.shape[0] if prompt.ndim >= 1 else 1
    outputs = []

    # Create logits
    model = model.reset()
    logits = model(prompt)

    next_token = sampler.sample_index_from_logits(logits[:, -1])
    outputs.append(next_token.item())
    total_seq_len = L + len(outputs)
    for next_tokens in autoregressive_decode_yield(
        model, next_token, max_length - total_seq_len, sampler, eos_token_id
    ):
        yield [*outputs, *[x.item() for x in next_tokens]]

@torch.inference_mode()
def autoregressive_decode_yield(
    model: CausalLM,
    curr_token: torch.Tensor,  # [1]
    num_new_tokens: int,
    sampler: SamplerBase,
    eos_token_id: int = None,
    return_probs: bool = False,
):

    def get_return(ts, ps):
        if return_probs:
            return (
                torch.cat(ts, dim=-1),
                torch.cat(ps, dim=-2),
            )
        else:
            return torch.cat(ts, dim=-1)

    new_tokens, new_probs = [], []
    for _ in range(num_new_tokens):
        logits = model(curr_token)
        if return_probs:
            new_token, new_prob = sampler.sample_from_logits(logits[:, -1])
            new_tokens.append(new_token)
            new_probs.append(new_prob)
        else:
            new_token = sampler.sample_index_from_logits(logits[:, -1])
            # new_tokens.append(new_token)
            new_tokens.append(new_token.unsqueeze(0)) 
        if eos_token_id is not None and new_token.item() == eos_token_id:
            break
        curr_token = new_token
        yield get_return(new_tokens, new_probs)
    yield get_return(new_tokens, new_probs)

def get_clear_command():
    os_name = platform.system()
    clear_command = "cls" if os_name == "Windows" else "clear"
    return clear_command

if __name__ == "__main__":

    # Model IDs and their paths
    models = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "files/TinyLlama-1.1B-Chat-v1.0/model.safetensors"
    }
    model_args = ARGS_MAP["TinyLlama-1.1B-Chat-v1.0"]

    # Create Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True
    )
    if model_args.vocab_size != len(tokenizer):
        print(
            f"WARNING: {'TinyLlama-1.1B-Chat-v1.0'}: model_args.vocab_size ({model_args.vocab_size}) != len(tokenizer) "
            f"({len(tokenizer)})"
        )

    model = CausalLM.from_pretrained(
        "files/TinyLlama-1.1B-Chat-v1.0/model.safetensors",
        model_args,
        strict=True,
    )
    model.eval()
    pipeline = Pipeline(
        model=model,
        tokenizer=tokenizer,
        model_name="TinyLlama-1.1B-Chat-v1.0", # not used
    )

    prompts = [
        "I don't know why, I'm struggling to maintain focus while studying. Any suggestions?"
    ]
    prompt = prompts[0]

    history = []
    for out, response in pipeline._generate(
            prompt, history=history, device="cpu"
        ):
            os.system(get_clear_command())
            print(out, flush=True)