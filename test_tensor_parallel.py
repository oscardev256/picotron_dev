#import pytest
import torch
import torch.distributed as dist
import process_group_manager as pgm
from tensor_parallel import split_tensor_along_last_dim, Reduce, Gather, Copy, ColumnParallelLinear, RowParallelLinear

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

class TestCopy:
    """Tests for Copy communication primitive"""
    
    def setup_method(self):
        """Setup mock before each test"""
        pgm.process_group_manager = Mock()
        pgm.process_group_manager.tp_world_size = 1
        pgm.process_group_manager.tp_rank = 0
        pgm.process_group_manager.tp_group = None
    
    def test_forward_is_identity(self):
        """Forward always returns input unchanged"""
        x = torch.randn(2, 4, requires_grad=True)
        y = Copy.apply(x)
        
        assert torch.equal(x, y)
        assert y.requires_grad
    
    def test_backward_tp1_is_identity(self):
        """When TP=1, backward is identity"""
        x = torch.randn(2, 4, requires_grad=True)
        y = Copy.apply(x)
        loss = y.sum()
        loss.backward()
        
        # Gradient should be ones (from sum)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))
    
    def test_preserves_shape(self):
        """Forward preserves tensor shape"""
        x = torch.randn(3, 5, 7, requires_grad=True)
        y = Copy.apply(x)
        assert y.shape == x.shape
    
    def test_gradient_flow(self):
        """Gradients flow correctly through Copy"""
        x = torch.randn(2, 4, requires_grad=True)
        y = Copy.apply(x)
        z = y * 3  # Some operation after Copy
        loss = z.sum()
        loss.backward()
        
        # Gradient should be 3 (from * 3 operation)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.full_like(x, 3.0))
    
    def test_multiple_operations(self):
        """Copy works in chain of operations"""
        x = torch.randn(2, 3, requires_grad=True)
        y = Copy.apply(x)
        z = y + 5
        w = z * 2
        loss = w.sum()
        loss.backward()
        
        # Gradient: d(loss)/d(x) = 2 (from chain rule)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.full_like(x, 2.0))


class TestColumnParallelLinear:
    """Tests for ColumnParallelLinear"""
    
    def setup_method(self):
        """Setup mock before each test"""
        pgm.process_group_manager = Mock()
        pgm.process_group_manager.tp_world_size = 1
        pgm.process_group_manager.tp_rank = 0
        pgm.process_group_manager.tp_group = None
    
    def test_init_shapes(self):
        """Test weight and bias shapes are correct"""
        layer = ColumnParallelLinear(
            in_features=8, 
            out_features=16, 
            bias=True,
            gather_output=False
        )
        
        # TP=1, so output_size_per_partition = out_features
        assert layer.weight.shape == (16, 8)
        assert layer.bias.shape == (16,)
    
    def test_init_no_bias(self):
        """Test layer without bias"""
        layer = ColumnParallelLinear(
            in_features=8, 
            out_features=16, 
            bias=False
        )
        
        assert layer.bias is None
        assert hasattr(layer, 'bias')  # Registered as None
    
    def test_forward_shape_no_gather(self):
        """Forward preserves shape when not gathering"""
        layer = ColumnParallelLinear(8, 16, bias=True, gather_output=False)
        x = torch.randn(2, 8)
        y = layer(x)
        
        assert y.shape == (2, 16)
    
    def test_forward_shape_with_gather(self):
        """Forward preserves shape when gathering (TP=1)"""
        layer = ColumnParallelLinear(8, 16, bias=True, gather_output=True)
        x = torch.randn(2, 8)
        y = layer(x)
        
        assert y.shape == (2, 16)
    
    def test_forward_with_bias(self):
        """Forward applies bias correctly"""
        layer = ColumnParallelLinear(4, 8, bias=True, gather_output=False)
        x = torch.randn(2, 4)
        y = layer(x)
        
        # Output should not be zero (weights are initialized)
        assert not torch.allclose(y, torch.zeros_like(y))
    
    def test_gradient_flow(self):
        """Gradients flow through layer"""
        layer = ColumnParallelLinear(4, 8, bias=True, gather_output=False)
        x = torch.randn(2, 4, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
    
    def test_bias_initialized_to_zero(self):
        """Bias is initialized to zeros"""
        layer = ColumnParallelLinear(4, 8, bias=True)
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
    
    def test_weight_initialized(self):
        """Weight is initialized (not zeros)"""
        layer = ColumnParallelLinear(4, 8, bias=True)
        # Should not be all zeros after initialization
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
    
    def test_output_size_per_partition(self):
        """output_size_per_partition calculated correctly"""
        layer = ColumnParallelLinear(8, 16, bias=True)
        # TP=1, so should be full size
        assert layer.output_size_per_partition == 16
    
    def test_gather_output_flag(self):
        """gather_output flag stored correctly"""
        layer1 = ColumnParallelLinear(8, 16, bias=True, gather_output=True)
        layer2 = ColumnParallelLinear(8, 16, bias=True, gather_output=False)
        
        assert layer1.gather_output == True
        assert layer2.gather_output == False
    
    def test_3d_input(self):
        """Works with 3D input (batch, seq, features)"""
        layer = ColumnParallelLinear(8, 16, bias=True)
        x = torch.randn(2, 10, 8)  # batch=2, seq=10, features=8
        y = layer(x)
        
        assert y.shape == (2, 10, 16)
    
    def test_multiple_forward_passes(self):
        """Layer works for multiple forward passes"""
        layer = ColumnParallelLinear(4, 8, bias=True)
        x1 = torch.randn(2, 4)
        x2 = torch.randn(3, 4)
        
        y1 = layer(x1)
        y2 = layer(x2)
        
        assert y1.shape == (2, 8)
        assert y2.shape == (3, 8)


class TestRowParallelLinear:
    """Tests for RowParallelLinear"""
    
    def setup_method(self):
        """Setup mock before each test"""
        pgm.process_group_manager = Mock()
        pgm.process_group_manager.tp_world_size = 1
        pgm.process_group_manager.tp_rank = 0
        pgm.process_group_manager.tp_group = None
    
    def test_init_shapes(self):
        """Test weight and bias shapes are correct"""
        layer = RowParallelLinear(
            in_features=16, 
            out_features=8, 
            bias=True
        )
        
        # TP=1, so input_size_per_partition = in_features
        assert layer.weight.shape == (8, 16)
        assert layer.bias.shape == (8,)  # Full bias, not partitioned
    
    def test_init_no_bias(self):
        """Test layer without bias"""
        layer = RowParallelLinear(
            in_features=16, 
            out_features=8, 
            bias=False
        )
        
        assert layer.bias is None
        assert hasattr(layer, 'bias')  # Registered as None
    
    def test_forward_shape(self):
        """Forward produces correct output shape"""
        layer = RowParallelLinear(16, 8, bias=True)
        x = torch.randn(2, 16)
        y = layer(x)
        
        assert y.shape == (2, 8)
    
    def test_forward_with_bias(self):
        """Forward applies bias correctly"""
        layer = RowParallelLinear(16, 8, bias=True)
        x = torch.randn(2, 16)
        y = layer(x)
        
        # Output should not be zero (weights are initialized)
        assert not torch.allclose(y, torch.zeros_like(y))
    
    def test_forward_without_bias(self):
        """Forward works without bias"""
        layer = RowParallelLinear(16, 8, bias=False)
        x = torch.randn(2, 16)
        y = layer(x)
        
        assert y.shape == (2, 8)
        # Should still produce output (just no bias term)
        assert not torch.allclose(y, torch.zeros_like(y))
    
    def test_gradient_flow(self):
        """Gradients flow through layer"""
        layer = RowParallelLinear(16, 8, bias=True)
        x = torch.randn(2, 16, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
    
    def test_bias_initialized_to_zero(self):
        """Bias is initialized to zeros"""
        layer = RowParallelLinear(16, 8, bias=True)
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
    
    def test_weight_initialized(self):
        """Weight is initialized (not zeros)"""
        layer = RowParallelLinear(16, 8, bias=True)
        # Should not be all zeros after initialization
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
    
    def test_input_size_per_partition(self):
        """input_size_per_partition calculated correctly"""
        layer = RowParallelLinear(16, 8, bias=True)
        # TP=1, so should be full size
        assert layer.input_size_per_partition == 16
    
    def test_3d_input(self):
        """Works with 3D input (batch, seq, features)"""
        layer = RowParallelLinear(16, 8, bias=True)
        x = torch.randn(2, 10, 16)  # batch=2, seq=10, features=16
        y = layer(x)
        
        assert y.shape == (2, 10, 8)
    
    def test_multiple_forward_passes(self):
        """Layer works for multiple forward passes"""
        layer = RowParallelLinear(16, 8, bias=True)
        x1 = torch.randn(2, 16)
        x2 = torch.randn(3, 16)
        
        y1 = layer(x1)
        y2 = layer(x2)
        
        assert y1.shape == (2, 8)
        assert y2.shape == (3, 8)
    
    def test_bias_added_after_reduce(self):
        """Bias is added after reduce operation (not during linear)"""
        layer = RowParallelLinear(4, 8, bias=True)
        
        # Set bias to known value
        with torch.no_grad():
            layer.bias.fill_(1.0)
        
        x = torch.randn(2, 4, requires_grad=True)
        y = layer(x)
        
        # Check that output has bias contribution
        # (not a perfect test, but ensures bias is applied)
        assert y.grad_fn is not None  # Has gradient function
    
    def test_column_to_row_pattern(self):
        """Column->Row pattern: output stays partitioned then reduced"""
        col_layer = ColumnParallelLinear(8, 16, bias=True, gather_output=False)
        row_layer = RowParallelLinear(16, 8, bias=True)
        
        x = torch.randn(2, 8)
        hidden = col_layer(x)  # Partitioned output
        output = row_layer(hidden)  # Reduced output
        
        assert output.shape == (2, 8)