"""
Filename: test_sampler_base.py

Description: Tests for the SamplerBase implementation.

"""

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from model.samplers import SamplerBase  # Assuming the class is defined in sampler_base.py

class TestSamplerBase(unittest.TestCase):
    
    def setUp(self):
        # Setting up a default SamplerBase instance
        self.sampler = SamplerBase(temperature=1.0, top_k=3)

    def test_logits_to_probs_shape(self):
        # Test 1: Ensure the output shape of logits_to_probs matches input shape
        logits = torch.randn(1, 2, 10)  # Shape [batch_size, seq_len, vocab_size]
        probs = self.sampler.logits_to_probs(logits)
        
        # Assert that the output has the same shape as the logits input
        self.assertEqual(logits.shape, probs.shape)
    
    def test_logits_to_probs_sum_to_one(self):
        # Test 2: Ensure that probabilities sum to 1 across the last dimension
        logits = torch.randn(1, 2, 10)
        probs = self.sampler.logits_to_probs(logits)
        
        # Assert that the sum of probabilities is 1 along the last dimension
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5))
    
    def test_sample_from_logits_shape(self):
        # Test 3: Ensure the output of sample_from_logits has the correct shape
        logits = torch.randn(1, 2, 10)
        idx, probs = self.sampler.sample_from_logits(logits)
        
        # Assert that idx has the correct shape
        self.assertEqual(idx.shape, (1, 2))
        # Assert that probs has the correct shape matching logits
        self.assertEqual(probs.shape, logits.shape)

    def test_greedy_decode_output(self):
        # Test 4: Ensure greedy_decode correctly returns the highest probability indices
        logits = torch.tensor([[[0.1, 0.2, 0.7], [0.3, 0.6, 0.1]]])  # Shape [1, 2, 3]
        greedy_indices = self.sampler.greedy_decode(logits)
        
        # Check that the indices correspond to the maximum logits
        self.assertTrue(torch.equal(greedy_indices, torch.tensor([[2, 1]])))

    def test_sample_index_deterministic(self):
        # Test 5: Sample index method should produce consistent results given identical input
        probs = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])  # Shape [1, 2, 2]
        idx1 = self.sampler.sample_index(probs)
        idx2 = self.sampler.sample_index(probs)
        
        # Assert the sampled indices are the same (with fixed random seed)
        self.assertTrue(torch.equal(idx1, idx2))

    def test_top_k_truncation(self):
        # Test 6: Ensure logits are correctly truncated to top_k elements
        logits = torch.tensor([[[0.1, 0.2, 0.7, 0.5], [0.3, 0.6, 0.1, 0.9]]])  # Shape [1, 2, 4]
        sampler = SamplerBase(temperature=1.0, top_k=2)
        probs = sampler.logits_to_probs(logits)
        
        # Check if the elements outside the top_k are set to very low probabilities
        self.assertEqual((probs < 1e-5).sum().item(), 4)  # 4 elements outside top_k
    
    def test_temperature_effect(self):
        # Test 7: Ensure that lowering temperature makes the distribution more peaked
        logits = torch.tensor([[[1.0, 2.0, 3.0]]])
        sampler_high_temp = SamplerBase(temperature=10.0)
        sampler_low_temp = SamplerBase(temperature=0.1)
        
        probs_high_temp = sampler_high_temp.logits_to_probs(logits)
        probs_low_temp = sampler_low_temp.logits_to_probs(logits)
        
        # Assert that low temperature peaks the distribution more strongly
        self.assertTrue(probs_low_temp.max() > probs_high_temp.max())
    

if __name__ == "__main__":
    unittest.main()
