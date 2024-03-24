# `.\lucidrains\mirasol-pytorch\mirasol_pytorch\distributed.py`

```py
# 导入必要的库
from functools import cache

import torch
from torch.autograd import Function
import torch.distributed as distributed

from einops import rearrange

# 辅助函数

# 使用缓存装饰器缓存结果，判断当前是否处于分布式环境
@cache
def get_is_distributed():
    return distributed.is_initialized() and distributed.get_world_size() > 1

# 在指定维度上对张量进行填充，使其达到指定长度
def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

# 分布式辅助函数

# 在所有进程中收集具有可变维度的张量，根据给定的维度和大小
def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, distributed.get_rank(), distributed.get_world_size()

    if not exists(sizes):
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
        distributed.all_gather(sizes, size)
        sizes = torch.stack(sizes)

    max_size = sizes.amax().item()
    padded_t = pad_dim_to(t, max_size, dim = dim)

    gathered_tensors = [torch.empty(padded_t.shape, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    distributed.all_gather(gathered_tensors, padded_t)

    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes

# 自定义的 Function 类，用于实现 all_gather 操作
class AllGather(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        assert get_is_distributed()
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, distributed.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

# 将自定义的 Function 应用到 all_gather 函数上
all_gather = AllGather.apply
```