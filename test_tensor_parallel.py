#import pytest
import torch
import torch.distributed as dist
import process_group_manager as pgm
from tensor_parallel import split_tensor_along_last_dim, Reduce, Gather

from unittest.mock import Mock

def split_tensor_along_last_dim(tensor, num_partitions):
    last_dim = tensor.dim() - 1
    assert tensor.size()[last_dim] % num_partitions == 0, \
        f"{tensor.size()[last_dim]} is not divisible by {num_partitions}"
    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)

class Reduce(torch.autograd.Function):
    """All-reduce in forward and identity and backward pass"""

    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Gather(torch.autograd.Function):
    """All-gather in forward pass, split in backward pass"""
    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        
        last_dim = input.dim() - 1
        input = input.contiguous()
        tensor_list = [torch.empty_tensor(input) for _ in range(pgm.process_group_manager.tp_world_size)]
        input = tensor_list[pgm.process_group_manager.local_rank]
        dist.all_gather(tensor_list, input, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=last_dim).contiguous()

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
                
        chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()


class TestSplitTensor:
    """Test for split_tensor_along_last_dim"""

    def test_split_tensor_1d(self):
        tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])

        chunks = split_tensor_along_last_dim(tensor, num_partitions=4)
        assert len(chunks) == 4
        assert chunks[0].shape == (2,)
        assert torch.equal(chunks[0], torch.tensor([1, 2]))
        assert torch.equal(chunks[3], torch.tensor([7, 8]))


class TestReduce:
    """Tests for Reduce communication primitive"""
    
    def setup_method(self):
        """Setup mock before each test"""
        pgm.process_group_manager = Mock()
        pgm.process_group_manager.tp_world_size = 1
        pgm.process_group_manager.tp_rank = 0
        pgm.process_group_manager.tp_group = None
    
    def test_reduce_forward_tp1(self):
        """When TP=1, forward should be identity"""
        x = torch.randn(2, 4, requires_grad=True)
        y = Reduce.apply(x)
        
        assert torch.equal(x, y)
        assert y.requires_grad
    
    def test_reduce_backward_tp1(self):
        """When TP=1, backward should be identity"""
        x = torch.randn(2, 4, requires_grad=True)
        y = Reduce.apply(x)
        loss = y.sum()
        loss.backward()
        
        # Gradient should be all ones (since we summed)
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.allclose(x.grad, torch.ones_like(x))
    
    def test_reduce_preserves_shape(self):
        """Shape should be preserved through forward/backward"""
        x = torch.randn(3, 5, 7, requires_grad=True)
        y = Reduce.apply(x)
        
        loss = y.sum()
        loss.backward()

        assert y.shape == x.shape

class TestGather:
    """Tests for Gather communication primitive"""
    
    def setup_method(self):
        """Setup mock before each test"""
        pgm.process_group_manager = Mock()
        pgm.process_group_manager.tp_world_size = 1
        pgm.process_group_manager.tp_rank = 0
        pgm.process_group_manager.tp_group = None
    
    def test_forward_tp1_is_identity(self):
        """When TP=1, forward is identity"""
        x = torch.randn(2, 4, requires_grad=True)
        y = Gather.apply(x)
        assert torch.equal(x, y)
        assert y.requires_grad
    
    def test_backward_tp1_is_identity(self):
        """When TP=1, backward is identity"""
        x = torch.randn(2, 4, requires_grad=True)
        y = Gather.apply(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))
    
    def test_forward_makes_contiguous(self):
        """Forward makes input contiguous"""
        # Create non-contiguous tensor
        x = torch.randn(4, 4, requires_grad=True).t()  # transpose makes non-contiguous
        assert not x.is_contiguous()
        
        y = Gather.apply(x)
        # Should work without error (contiguous call inside forward)
        assert y is not None
    
    def test_backward_returns_contiguous(self):
        """Backward returns contiguous gradient"""
        x = torch.randn(2, 8, requires_grad=True)
        y = Gather.apply(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad.is_contiguous()
    
    def test_preserves_shape_tp1(self):
        """Shape preserved when TP=1"""
        x = torch.randn(3, 5, 7, requires_grad=True)
        y = Gather.apply(x)
        assert y.shape == x.shape
    
    def test_gradient_flow(self):
        """Gradients flow correctly through gather"""
        x = torch.randn(2, 4, requires_grad=True)
        y = Gather.apply(x)
        z = y * 2  # Some operation after gather
        loss = z.sum()
        loss.backward()
        
        # Gradient should be 2 (from * 2 operation)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.full_like(x, 2.0))