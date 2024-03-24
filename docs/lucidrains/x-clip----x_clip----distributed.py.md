# `.\lucidrains\x-clip\x_clip\distributed.py`

```
# 导入 torch 库
import torch
# 从 torch.autograd 模块中导入 Function 类
from torch.autograd import Function
# 导入 torch.distributed 模块
import torch.distributed as distributed
# 从 einops 库中导入 rearrange 函数

from einops import rearrange

# 定义函数 pad_dim_to，用于在指定维度上对张量进行填充
def pad_dim_to(t, length, dim = 0):
    # 计算需要填充的长度
    pad_length = length - t.shape[dim]
    # 计算需要填充的维度对数
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    # 对张量进行填充操作
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

# distributed helpers

# 定义函数 all_gather_variable_dim，用于在分布式环境下收集不同维度的张量
def all_gather_variable_dim(t, dim = 0, sizes = None):
    # 获取当前设备、进程排名和世界大小
    device, rank, world_size = t.device, distributed.get_rank(), distributed.get_world_size()

    # 如果未提供 sizes 参数，则进行计算
    if not exists(sizes):
        # 创建包含当前维度大小的张量
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        # 创建用于存储各进程维度大小的列表
        sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
        # 使用 all_gather 函数收集各进程的维度大小
        distributed.all_gather(sizes, size)
        # 将结果堆叠成张量
        sizes = torch.stack(sizes)

    # 获取最大维度大小
    max_size = sizes.amax().item()
    # 对输入张量进行填充操作
    padded_t = pad_dim_to(t, max_size, dim = dim)

    # 创建用于存储收集到的张量的列表
    gathered_tensors = [torch.empty(padded_t.shape, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    # 使用 all_gather 函数收集张量
    distributed.all_gather(gathered_tensors, padded_t)

    # 将收集到的张量拼接在一起
    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    # 创建序列张量
    seq = torch.arange(max_size, device = device)

    # 创建掩码，用于选择有效数据
    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    # 根据掩码选择有效数据
    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes

# 定义 AllGather 类，继承自 Function 类
class AllGather(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        # 断言分布式环境已初始化且世界大小大于 1
        assert distributed.is_initialized() and distributed.get_world_size() > 1
        # 调用 all_gather_variable_dim 函数进行数据收集
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        # 获取批次大小和当前进程排名
        batch_sizes, rank = ctx.batch_sizes, distributed.get_rank()
        # 根据批次大小拆分梯度
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

# 将 AllGather 类应用为函数
all_gather = AllGather.apply
```