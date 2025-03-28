"""Sparsemax activation function.

PyTorch implementation of Sparsemax function from:
"From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
by André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
"""

import torch
import torch.nn as nn


class SparsemaxFunction(torch.autograd.Function):
    """Sparsemax autograd function implementing forward and backward passes."""

    @staticmethod
    def forward(ctx, input, dim=-1):
        """Forward pass for Sparsemax.

        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Input tensor of any shape
            dim (int): Dimension along which to apply sparsemax

        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        ctx.dim = dim
        input_dim = input.dim()
        dim = dim if dim >= 0 else input_dim + dim

        # Move target dimension to last position
        input_trans = input.transpose(dim, -1)
        input_flat = input_trans.contiguous().view(-1, input_trans.size(-1))

        # Translate for numerical stability
        input_flat = input_flat - input_flat.max(dim=1, keepdim=True)[0]

        # Sort inputs in descending order
        sorted_input, _ = torch.sort(input_flat, dim=1, descending=True)

        # Compute cumulative sums and thresholds
        cumsum = sorted_input.cumsum(dim=1)
        range_tensor = torch.arange(
            1, sorted_input.size(1) + 1, device=input.device, dtype=input.dtype
        ).unsqueeze(0)
        bound = sorted_input - (cumsum - 1) / range_tensor

        # Find partition threshold
        k = (bound > 0).sum(dim=1, keepdim=True)
        k = torch.clamp(k, min=1)  # Ensure k >= 1

        # Compute tau and output
        tau = (cumsum.gather(1, k - 1) - 1) / k.type(input.dtype)
        output_flat = torch.clamp(input_flat - tau, min=0)

        # Reshape back to original dimensions
        output = output_flat.view(input_trans.size()).transpose(dim, -1).contiguous()

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for Sparsemax.

        Args:
            ctx: Context object with saved tensors
            grad_output (torch.Tensor): Gradient from subsequent layers

        Returns:
            torch.Tensor: Gradient with respect to input tensor
        """
        (output,) = ctx.saved_tensors
        dim = ctx.dim

        # Compute gradient using support elements
        nonzeros = (output > 0).float()
        sum_grad = (grad_output * nonzeros).sum(dim=dim, keepdim=True)
        count = nonzeros.sum(dim=dim, keepdim=True).clamp(min=1)
        grad_input = nonzeros * (grad_output - sum_grad / count)

        return grad_input, None


class Sparsemax(nn.Module):
    """Sparsemax activation module handling tensors of arbitrary dimensionality."""

    def __init__(self, dim=-1):
        """Initialize Sparsemax activation.

        Args:
            dim (int, optional): Dimension along which to apply sparsemax. Default: -1
        """
        super().__init__()
        self.dim = dim

    def forward(self, input):
        """Forward pass through Sparsemax activation.

        Args:
            input (torch.Tensor): Input tensor of any shape

        Returns:
            torch.Tensor: Activated output with same shape as input
        """
        return SparsemaxFunction.apply(input, self.dim)
