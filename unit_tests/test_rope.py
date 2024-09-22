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
        # Common setup for all tests
        self.args = TransformerArgs(
            max_seq_len=10, rope_theta=10000.0, hidden_size=4, num_attention_heads=1, rope_partial_factor=None
        )
        self.rope = RoPE(self.args)

    def test_rope_identity_at_position_zero(self):
        """
        Test that RoPE does not alter the input at position 0 (identity transformation).
        """
        # Create input tensor with position 0
        # Shape: [L, H, D] = [1, 1, 4]
        x = torch.tensor([
            [[1.0, 0.0, 1.0, 0.0]],  # Position 0
        ])  # Shape [1, 1, 4]

        # Apply RoPE with start_index=0
        x_rope = self.rope(x, start_index=0)

        # Assert that output equals input
        self.assertTrue(torch.allclose(x, x_rope, atol=1e-6),
                        "RoPE should not alter the input at position 0")
    
    def test_rope_norm_preservation(self):
        """
        Test that RoPE preserves the norm of the input embeddings.
        """
        # Create a random input tensor
        # Shape: [L, H, D] = [5, 2, 8]
        torch.manual_seed(42)  # For reproducibility
        x = torch.randn(5, 2, 4)  # [L, H, D]

        # Compute norms before applying RoPE
        norm_before = torch.norm(x, dim=-1)

        # Apply RoPE
        x_rope = self.rope(x, start_index=0)

        # Compute norms after applying RoPE
        norm_after = torch.norm(x_rope, dim=-1)

        # Assert that norms are approximately equal
        self.assertTrue(torch.allclose(norm_before, norm_after, atol=1e-6),
                        "RoPE should preserve the norm of the input embeddings")
    
    def test_rope_correct_rotation(self):
        """
        Test that RoPE correctly applies rotation to the input embeddings.
        Specifically, verify that position 0 remains unchanged and position 1 is rotated as expected.
        """
        # Create input tensor with two positions
        # Shape: [L, H, D] = [2, 1, 4]
        x = torch.tensor([
            [[1.0, 0.0, 0.0, 1.0]],  # Position 0
            [[1.0, 0.0, 0.0, 1.0]],  # Position 1
        ])  # Shape [2, 1, 4]

        # Apply RoPE
        x_rope = self.rope(x, start_index=0)

        # Position 0 should remain unchanged
        self.assertTrue(torch.allclose(x_rope[0], x[0], atol=1e-6),
                        "RoPE should not alter the input at position 0")

        # Manually compute expected rotation for position 1
        # Retrieve the complex frequencies for position 1
        freqs_complex = self.rope.freqs_complex[1]  # Shape: [rope_hidden_size//2]

        # Reshape input at position 1 to complex numbers
        # Original shape: [H, D] = [1, 4]
        # Reshaped to [H, 2, D//2] = [1, 2, 2]
        x_pos1 = x[1].reshape(1, 2, 2).transpose(-1, -2).contiguous()  # [H, 2, 2]
        x_complex = torch.view_as_complex(x_pos1)  # [H, 2]

        # Apply rotation: multiply by the corresponding frequency
        # freqs_complex shape: [rope_hidden_size//2] = [2]
        # x_complex shape: [H, rope_hidden_size//2] = [1, 2]
        # To multiply element-wise, reshape freqs_complex to [1, 2]
        f_complex = freqs_complex.view(1, -1)  # [1, 2]
        x_rotated_complex = x_complex * f_complex  # [1, 2]

        # Convert back to real numbers
        x_rotated = torch.view_as_real(x_rotated_complex).transpose(-1, -2).reshape(4)  # [D]

        # Assert that the rotated output matches the expected rotation
        self.assertTrue(torch.allclose(x_rope[1], x_rotated, atol=1e-6),
                        "RoPE did not correctly rotate the input at position 1")

    def test_rope_partial_application(self):
        """
        Test that RoPE is correctly applied only to a subset of the hidden dimensions when rope_partial_factor is set.
        """
        # Define a TransformerArgs with rope_partial_factor
        partial_args = TransformerArgs(rope_partial_factor=0.5)
        rope_partial = RoPE(partial_args)

        # Create input tensor
        # Shape: [L, H, D] = [2, 1, 4]
        x = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0]],  # Position 0
            [[5.0, 6.0, 7.0, 8.0]],  # Position 1
        ])  # Shape [2, 1, 4]

        # Apply RoPE
        x_rope = rope_partial(x, start_index=0)

        # The first half of the hidden dimensions should be rotated
        # The second half should remain unchanged
        # hidden_size=4, rope_hidden_size=2

        # Manually compute rotation for the first two dimensions
        freqs_complex = rope_partial.freqs_complex[start_index : start_index + 2]  # [2, 1]

        # Position 0 rotation (should be identity)
        x_pos0 = x[0, 0, :2].reshape(1, 1) + 0j  # [1, 1]
        x_rotated_pos0 = freqs_complex[0].unsqueeze(0) * x_pos0  # [1,1] * [1,1] -> [1,1]
        x_rotated_pos0 = torch.view_as_real(x_rotated_pos0).reshape(2)

        # Position 1 rotation
        x_pos1 = x[1, 0, :2].reshape(1, 1) + 0j  # [1,1]
        x_rotated_pos1 = freqs_complex[1].unsqueeze(0) * x_pos1  # [1,1] * [1,1] -> [1,1]
        x_rotated_pos1 = torch.view_as_real(x_rotated_pos1).reshape(2)

        # Expected output
        expected = torch.stack([
            torch.cat([x_rotated_pos0, x[0, 0, 2:]]),
            torch.cat([x_rotated_pos1, x[1, 0, 2:]]),
        ])

        # Assert that the rotated parts match and the untouched parts are equal
        self.assertTrue(torch.allclose(x_rope, expected, atol=1e-6),
                        "RoPE did not correctly apply partial rotation")

if __name__ == "__main__":
    unittest.main()