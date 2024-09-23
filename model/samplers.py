"""
Filename: samplers.py

Description: ???

Notes:

"""

import torch
import torch.nn.functional as F

class SamplerBase:
    def __init__(self, temperature: float = 1.0, top_k: int = None):
        self.top_k = top_k
        self.temperature = max(temperature, 1e-4)

    @torch.inference_mode()
    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / self.temperature
        if self.top_k is not None:
            k = min(self.top_k, logits.shape[-1])
            v_topk, _ = torch.topk(logits, k=k, dim=-1)
            thresh = v_topk[..., -1].unsqueeze(-1)
            logits = torch.where(logits < thresh, -float("inf"), logits)
        probs = F.softmax(logits, dim=-1)
        return probs

    @torch.inference_mode()
    def sample_from_logits(self, logits: torch.Tensor, keepdim=False) -> torch.Tensor:
        probs = self.logits_to_probs(logits)
        idx = self.sample_index(probs, keepdim=keepdim)
        return idx, probs

    @torch.inference_mode()
    def sample_index_from_logits(self, logits: torch.Tensor, keepdim=False) -> torch.Tensor:
        p = self.logits_to_probs(logits)
        return self.sample_index(p, keepdim=keepdim) # does not call greedy decode, should it?

    @torch.inference_mode()
    def sample_index(self, p: torch.Tensor, keepdim=False) -> torch.Tensor:
        q = torch.empty_like(p).exponential_(1)
        return torch.argmax(p / q, dim=-1, keepdim=keepdim)

    @torch.inference_mode()
    def greedy_decode(self, logits: torch.Tensor, keepdim=False) -> torch.Tensor:
        # Greedy decoding simply selects the index with the highest probability
        idx = torch.argmax(logits, dim=-1, keepdim=keepdim)
        return idx

if __name__ == "__main__":
    # Set a random seed for reproducibility
    torch.manual_seed(16)

    # Create a sample logits tensor with random values
    x = torch.randn(1, 2, 10)  # Shape (batch_size=1, sequence_length=2, vocab_size=10)
    probs = F.softmax(x, dim=-1)

    # Create an instance of the SamplerBase class
    sampler = SamplerBase(temperature=1.0, top_k=3)

    # Use greedy decoding on the logits tensor
    greedy_indices = sampler.greedy_decode(probs)

    # Print the results
    print("Logits Tensor:")
    print(probs)
    print("\nGreedy Decoding Indices:")
    print(greedy_indices)
