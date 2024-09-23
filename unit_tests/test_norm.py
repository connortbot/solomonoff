"""
Filename: test_norm.py

Description: Tests for RMSNorm implementation

Notes:

"""

import unittest

import torch

import sys
from pathlib import Path

# move up one directory level
sys.path.append(str(Path(__file__).parent.parent))

from model.norm import RMSNorm
from helpers.transformer_args import TransformerArgs

class TestRMSNorm(unittest.TestCase):
    
    def setUp(self):
        # Setup default args for RMSNorm
        self.args = TransformerArgs(llm_type="default", hidden_size=64, rms_norm_eps=1e-6)
        self.rmsnorm = RMSNorm(self.args)
    
    def test_output_shape(self):
        # Test 1: Check if the output shape matches the input shape
        L = 10  # Sequence length
        D = 64  # Hidden size (same as self.args.hidden_size)
        
        x = torch.randn(L, D)  # create random input tensor of shape [L, D]
        output = self.rmsnorm(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_forward_rmsnorm(self):
        # Test 2: normalizes the input correctly
        L = 10
        D = 64
        x = torch.randn(L, D)  # Create random input tensor
        
        output = self.rmsnorm(x)
        
        # Compute RMS for the output
        rms_output = torch.sqrt(torch.mean(output**2, dim=-1))
        
        self.assertTrue(torch.allclose(rms_output, torch.ones_like(rms_output), atol=1e-5))
    
    def test_weight_scaling(self):
        # Test 3: if the learnable weight scales the output properly
        L = 10
        D = 64
        x = torch.randn(L, D)
        
        # Set the learnable weight to a fixed value
        self.rmsnorm.weight.data = torch.full((D,), 2.0)
        
        output = self.rmsnorm(x)
        
        # Apply RMSNorm manually and scale by the weight
        normalized_x = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.args.rms_norm_eps)
        expected_output = normalized_x * 2.0  # Weight is set to 2.0
        
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-5))

    def test_eps_handling(self):
        # Test 4: Check that eps is correctly used for numerical stability
        L = 10
        D = 64
        x = torch.zeros(L, D)  # input with all zeros
        
        output = self.rmsnorm(x)
        
        # Since the input is zero, the output should also be zero (given that rms_norm_eps avoids division by zero)
        self.assertTrue(torch.allclose(output, torch.zeros_like(output), atol=1e-5))
    
    def test_custom_epsilon(self):
        # Test 5: Check custom epsilon handling by setting a large value of epsilon
        custom_args = TransformerArgs(llm_type="default", hidden_size=64, rms_norm_eps=1.0)
        custom_rmsnorm = RMSNorm(custom_args)
        
        L = 10
        D = 64
        x = torch.randn(L, D)
        
        output = custom_rmsnorm(x)
        
        # Compute the manual RMS normalization with the custom epsilon
        normalized_x = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1.0)  # eps = 1.0
        
        self.assertTrue(torch.allclose(output, custom_rmsnorm.weight * normalized_x, atol=1e-5))

if __name__ == "__main__":
    unittest.main()