import torch
import torch.distributed as dist
import process_group_manager as pgm
import torch.nn as nn

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

class Copy(torch.autograd.Function):
    """Identity in forward pass, all-reduce in backward"""
    @staticmethod
    def forward(ctx, input):
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return grad_output
    
class ColumnParallelLinear(nn.Modules):
    def __init__(self, in_features: int, out_features: int, bias: bool, gather_output: bool = False):

        super(ColumnParallelLinear, self).init()
        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank
        self.apply_gather = gather_output

        self.in_features = in_features
        self.out_features = out_features
        assert out_features % self.tp_world_size == 0
        self.output_size_per_partitions = out_features // self.tp_world_size

    def reset_parameters(self):
        pass

    def forward(self, input):
        pass

