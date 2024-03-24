# `.\lucidrains\st-moe-pytorch\st_moe_pytorch\distributed.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 torch.autograd 模块中导入 Function 类
from torch.autograd import Function

# 从 torch.distributed 模块中导入 dist 对象
import torch.distributed as dist

# 从 einops 库中导入 rearrange, pack, unpack 函数
from einops import rearrange, pack, unpack

# 定义函数，判断变量是否存在
def exists(val):
    return val is not None

# 定义函数，返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义函数，判断两个数是否整除
def divisible_by(num, den):
    return (num % den) == 0

# 定义函数，将张量在指定维度上进行填充
def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length)

# 定义函数，对所有进程进行相同维度的全局收集
def all_gather_same_dim(t):
    t = t.contiguous()
    world_size = dist.get_world_size()
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors

# 定义函数，收集指定维度的大小信息
def gather_sizes(t, *, dim):
    size = torch.tensor(t.shape[dim], device = t.device, dtype = torch.long)
    sizes = all_gather_same_dim(size)
    return torch.stack(sizes)

# 定义函数，判断张量是否只有一个值
def has_only_one_value(t):
    return (t == t[0]).all()

# 定义函数，对所有进程进行变量维度的全局收集
def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    if not exists(sizes):
        sizes = gather_sizes(t, dim = dim)

    if has_only_one_value(sizes):
        gathered_tensors = all_gather_same_dim(t)
        gathered_tensors = torch.cat(gathered_tensors, dim = dim)
        return gathered_tensors, sizes

    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = dim)
    gathered_tensors = all_gather_same_dim(padded_t)

    gathered_tensors = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensors = gathered_tensors.index_select(dim, indices)

    return gathered_tensors, sizes

# 定义 AllGatherFunction 类，继承自 Function 类
class AllGatherFunction(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

# 定义 AllGather 类，继承自 nn.Module 类
class AllGather(nn.Module):
    def __init__(self, *, dim = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x, sizes = None):
        return AllGatherFunction.apply(x, self.dim, sizes)

# 定义函数，根据进程排名拆分张量
def split_by_rank(x):
    rank = dist.get_rank()
    out = x[rank]

    if isinstance(x, tuple):
        sizes = tuple(map(lambda t: t.shape[0], x))
    else:
        sizes = (x.shape[1],) * x.shape[0]

    sizes = torch.tensor(sizes, device = out.device, dtype = torch.long)
    return out, sizes
```