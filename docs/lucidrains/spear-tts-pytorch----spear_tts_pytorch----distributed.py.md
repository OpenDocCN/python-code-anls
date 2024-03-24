# `.\lucidrains\spear-tts-pytorch\spear_tts_pytorch\distributed.py`

```
# 导入 torch 库
import torch
# 从 torch.autograd 模块中导入 Function 类
from torch.autograd import Function
# 导入 torch.distributed 模块
import torch.distributed as distributed
# 从 einops 库中导入 rearrange 函数

from einops import rearrange

# distributed helpers

# 定义一个函数用于在所有进程中收集具有可变维度的张量
def all_gather_variable_dim(t, dim = 0, sizes = None):
    # 获取当前设备、进程的排名和总进程数
    device, rank, world_size = t.device, distributed.get_rank(), distributed.get_world_size()

    # 如果 sizes 不存在
    if not exists(sizes):
        # 创建一个张量表示 t 在指定维度上的大小
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        # 创建一个列表，用于存储各个进程的大小信息
        sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
        # 在所有进程中收集各个进程的大小信息
        distributed.all_gather(sizes, size)
        # 将收集到的大小信息堆叠成一个张量
        sizes = torch.stack(sizes)

    # 获取所有进程中最大的大小
    max_size = sizes.amax().item()
    # 将 t 在指定维度上填充到最大大小
    padded_t = pad_dim_to(t, max_size, dim = dim)

    # 创建一个列表，用于存储各个进程收集到的张量
    gathered_tensors = [torch.empty(padded_t.shape, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    # 在所有进程中收集填充后的张量
    distributed.all_gather(gathered_tensors, padded_t)

    # 将所有进程收集到的张量在指定维度上拼接
    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    # 创建一个序列张量
    seq = torch.arange(max_size, device = device)

    # 创建一个掩码，用于选择有效的数据
    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    # 根据掩码选择有效的数据
    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes

# 定义一个继承自 Function 的类 AllGather
class AllGather(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        # 检查是否处于分布式环境中且进程数大于 1
        is_dist = distributed.is_initialized() and distributed.get_world_size() > 1
        ctx.is_dist = is_dist

        # 如果不处于分布式环境中，直接返回输入张量和空值
        if not is_dist:
            return x, None

        # 在所有进程中收集具有可变维度的张量
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        # 如果不处于分布式环境中，直接返回梯度和空值
        if not ctx.is_dist:
            return grads, None, None

        # 获取各个进程的大小信息和当前进程的排名
        batch_sizes, rank = ctx.batch_sizes, distributed.get_rank()
        # 根据各个进程的大小信息拆分梯度
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

# 将 AllGather 类应用为一个函数
all_gather = AllGather.apply
```