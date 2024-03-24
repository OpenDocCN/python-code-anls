# `.\lucidrains\musiclm-pytorch\musiclm_pytorch\distributed.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.autograd 模块中导入 Function 类
from torch.autograd import Function
# 从 torch.distributed 模块中导入 dist 模块
import torch.distributed as dist
# 从 einops 库中导入 rearrange 函数

from einops import rearrange

# 分布式辅助函数

# 定义一个函数，用于在所有进程中收集具有相同维度的张量
def all_gather_same_dim(t):
    # 获取世界大小
    world_size = dist.get_world_size()
    # 创建一个空列表，用于存储收集到的张量
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    # 在所有进程中收集张量
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors

# 定义一个函数，用于在所有进程中收集具有可变维度的张量
def all_gather_variable_dim(t, dim = 0, sizes = None):
    # 获取设备、进程编号和世界大小
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    # 如果 sizes 不存在
    if not exists(sizes):
        # 创建一个张量，表示张量在指定维度上的大小
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        # 在所有进程中收集大小信息
        sizes = all_gather_same_dim(size)
        sizes = torch.stack(sizes)

    # 如果所有进程收集到的大小信息都相同
    if torch.unique(sizes).numel() == 1:
        # 在所有进程中收集张量
        gathered_tensors = all_gather_same_dim(t)
        return torch.cat(gathered_tensors, dim = dim), sizes

    # 获取最大的大小
    max_size = sizes.amax().item()

    # 将张量在指定维度上填充到最大大小
    padded_t = pad_dim_to(t, max_size, dim = dim)
    # 在所有进程中收集填充后的张量
    gathered_tensors = all_gather_same_dim(padded_t)

    # 拼接所有进程中收集到的张量
    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    # 创建一个序列
    seq = torch.arange(max_size, device = device)

    # 创建一个掩码，用于选择有效的数据
    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    # 根据掩码选择有效的数据
    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes

# 定义一个自定义函数类 AllGatherFunction
class AllGatherFunction(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes, all_reduce_grads):
        # 调用 all_gather_variable_dim 函数
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.dim = dim
        ctx.all_reduce_grads = all_reduce_grads
        ctx.batch_sizes = batch_sizes.tolist()
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        # 获取批次大小和进程编号
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        # 如果需要对梯度进行全局归约
        if ctx.all_reduce_grads:
            dist.all_reduce(grads)

        # 根据批次大小拆分梯度
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None, None

# 定义一个类 AllGather，继承自 nn.Module
class AllGather(nn.Module):
    def __init__(
        self,
        dim,
        *,
        all_reduce_grads = False
    ):
        super().__init__()
        self.dim = dim
        self.all_reduce_grads = all_reduce_grads
        # 判断是否处于分布式环境中
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    def forward(
        self,
        x,
        sizes = None
    ):
        # 如果不处于分布式环境中，直接返回输入张量
        if not self.is_distributed:
            return x, None

        # 调用 AllGatherFunction 类的 apply 方法
        return AllGatherFunction.apply(x, self.dim, sizes, self.all_reduce_grads)
```