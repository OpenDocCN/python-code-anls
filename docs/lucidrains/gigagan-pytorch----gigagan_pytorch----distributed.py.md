# `.\lucidrains\gigagan-pytorch\gigagan_pytorch\distributed.py`

```
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 torch.autograd 模块中导入 Function 类
from torch.autograd import Function
# 导入 torch 分布式模块
import torch.distributed as dist
# 从 einops 库中导入 rearrange 函数

from einops import rearrange

# helpers

# 判断变量是否存在的辅助函数
def exists(val):
    return val is not None

# 在指定维度上对张量进行填充的辅助函数
def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

# distributed helpers

# 在所有进程中收集具有可变维度的张量的辅助函数
def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, world_size = t.device, dist.get_world_size()

    if not exists(sizes):
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
        dist.all_gather(sizes, size)
        sizes = torch.stack(sizes)

    max_size = sizes.amax().item()
    padded_t = pad_dim_to(t, max_size, dim = dim)

    gathered_tensors = [torch.empty(padded_t.shape, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, padded_t)

    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes

# 自定义 Function 类 AllGather
class AllGather(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        is_dist = dist.is_initialized() and dist.get_world_size() > 1
        ctx.is_dist = is_dist

        if not is_dist:
            return x, None

        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        if not ctx.is_dist:
            return grads, None, None

        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

# 将 AllGather 类应用为函数
all_gather = AllGather.apply
```