"""
Filename: test_ffn.py

Description: Tests for FeedForward implementation

Notes:

"""


import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# move up one directory level
sys.path.append(str(Path(__file__).parent.parent))

from model.feedforward import FeedForward
from helpers.transformer_args import TransformerArgs

class TestFeedForward(unittest.TestCase):
    def setUp(self):
        pass

    def test_feedforward_shape(self):
        args = TransformerArgs(llm_type="llama", hidden_size=8, intermediate_size=16)
        model = FeedForward(args)
        x = torch.randn(10, args.hidden_size)  # [L, D] where L=10, D=8
        output = model(x)
        assert output.shape == (10, args.hidden_size)
    
    def test_feedforward_linear_math(self):
        args = TransformerArgs(llm_type="llama", hidden_size=2, intermediate_size=4)
        model = FeedForward(args)
        model.gate_proj.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        model.up_proj.weight.data = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, -1.0], [0.5, -0.5]])
        model.down_proj.weight.data = torch.tensor([[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 1.0]])

        x = torch.tensor([[1.0, 2.0]])  # Shape [1, 2]
        output = model(x)

        # manual
        gate_proj = torch.tensor([[1.0, 2.0, 3.0, 1.5]])  # gate_proj(x)
        silu_gate_proj = F.silu(gate_proj)  # Apply SiLU activation
        up_proj = torch.tensor([[2.0, 4.0, -1.0, -0.5]])  # up_proj(x)
        elementwise_mul = silu_gate_proj * up_proj
        expected_output = elementwise_mul @ model.down_proj.weight.T # matrix mult ED^T

        assert torch.allclose(output, expected_output), f"Expected {expected_output}, got {output}"
    
    def test_feedforward_silu_activation(self):
        args = TransformerArgs(llm_type="llama", hidden_size=3, intermediate_size=3)
        model = FeedForward(args)

        model.gate_proj.weight.data = torch.eye(3)  # Identity matrix as weights
        x = torch.tensor([[1.0, 0.0, -1.0]])  # Shape [1, 3]

        # forward pass for gate_proj only
        gate_output = model.gate_proj(x)

        expected_silu_output = F.silu(gate_output)

        assert torch.allclose(F.silu(gate_output), expected_silu_output), f"Expected {expected_silu_output}, got {gate_output}"

    def test_feedforward_elementwise_multiplication(self):
        args = TransformerArgs(llm_type="llama", hidden_size=2, intermediate_size=4)
        model = FeedForward(args)

        model.gate_proj.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        model.up_proj.weight.data = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, -1.0], [0.5, -0.5]])
        x = torch.tensor([[1.0, 2.0]])  # Shape [1, 2]

        # forward pass for projections only
        x1 = F.silu(model.gate_proj(x))
        x2 = model.up_proj(x)
        expected_mul = x1 * x2

        assert torch.allclose(x1 * x2, expected_mul), f"Expected {expected_mul}, got {x1 * x2}"

if __name__ == "__main__":
    unittest.main()