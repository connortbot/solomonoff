"""
Filename: rope.py

Description: Tests for RoPE implementation

Notes:

"""

import unittest
import math
import torch

import sys
from pathlib import Path

# move up one directory level
sys.path.append(str(Path(__file__).parent.parent))

from model.rope import RoPE
from helpers.transformer_args import TransformerArgs

class TestRoPE(unittest.TestCase):
    
    def setUp(self):
        # Setting up default arguments for RoPE (these values can be tuned based on your specific setup)
        args = TransformerArgs(
            max_seq_len=128,
            rope_theta=10000,
            hidden_size=64,
            num_attention_heads=8,
            rope_partial_factor=0.5  # Only apply RoPE to half of the hidden size
        )
        self.rope = RoPE(args)
    
    def test_shape_unchanged(self):
        # Test 1: Ensure that input and output have the same shape
        L = 10  # Sequence length
        H = 8   # Number of attention heads
        D = 64  # Hidden size
        start_index = 0

        x = torch.randn(L, H, D // H)  # Create random input tensor of shape [L, H, D]
        x_rope = self.rope(x, start_index)
        
        # Assert that the output has the same shape as the input
        self.assertEqual(x.shape, x_rope.shape)
    
    def test_partial_rope_application(self):
        # Test 2: Ensure RoPE is applied to the first 'rope_hidden_size' dimensions
        L = 10
        H = 8
        D = 64
        start_index = 0

        x = torch.randn(L, H, D // H)  # Create random input tensor of shape [L, H, D]
        x_rope = self.rope(x, start_index)
        
        rope_hidden_size = self.rope.rope_hidden_size
        
        # Assert that the first rope_hidden_size values have changed (RoPE applied)
        self.assertFalse(torch.allclose(x[..., :rope_hidden_size], x_rope[..., :rope_hidden_size], atol=1e-5))
        # Assert that the remaining dimensions haven't changed (RoPE not applied)
        self.assertTrue(torch.allclose(x[..., rope_hidden_size:], x_rope[..., rope_hidden_size:], atol=1e-5))
    
    def test_freqs_precomputed(self):
        # Test 3: Ensure that frequency precomputation is done correctly
        freqs_complex = self.rope.freqs_complex
        max_seq_len = self.rope.max_seq_len
        rope_hidden_size = self.rope.rope_hidden_size
        
        # Assert the shape of the precomputed frequencies is [max_seq_len, rope_hidden_size//2]
        self.assertEqual(freqs_complex.shape, (max_seq_len, rope_hidden_size // 2))
    
    def test_with_varied_start_index(self):
        # Test 4: Test RoPE with different start indices
        L = 10
        H = 8
        D = 64
        start_index = 5

        x = torch.randn(L, H, D // H)  # Create random input tensor of shape [L, H, D]
        x_rope_start_5 = self.rope(x, start_index)

        # Check with a different start_index
        start_index_2 = 2
        x_rope_start_2 = self.rope(x, start_index_2)
        
        # Output should differ with different start indices
        self.assertFalse(torch.allclose(x_rope_start_5, x_rope_start_2, atol=1e-5))
    
    def test_invalid_sequence_length(self):
        # Test 5: Ensure error is raised when sequence length exceeds max_seq_len
        L = 130  # Sequence length larger than max_seq_len (which is 128)
        H = 8
        D = 64
        start_index = 0
        
        x = torch.randn(L, H, D // H)
        
        with self.assertRaises(AssertionError):
            self.rope(x, start_index)

if __name__ == "__main__":
    unittest.main()