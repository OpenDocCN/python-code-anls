# `.\cogvideo-finetune\sat\sgm\modules\cp_enc_dec.py`

```py
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 导入 PyTorch 分布式计算库
import torch.distributed
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 从上级模块导入实用函数
from ..util import (
    get_context_parallel_group,  # 获取并行计算组
    get_context_parallel_rank,   # 获取当前并行计算的排名
    get_context_parallel_world_size,  # 获取并行计算的世界大小
)

# 设置使用的计算模式为 CP (Context Parallel)
_USE_CP = True

# 将输入转换为元组，确保长度为指定值
def cast_tuple(t, length=1):
    # 如果 t 已经是元组，直接返回；否则返回重复的元组
    return t if isinstance(t, tuple) else ((t,) * length)

# 检查 num 是否可以被 den 整除
def divisible_by(num, den):
    # 返回 num 除以 den 的余数是否为 0
    return (num % den) == 0

# 检查 n 是否为奇数
def is_odd(n):
    # 返回 n 除以 2 的结果是否为偶数的相反值
    return not divisible_by(n, 2)

# 检查值 v 是否存在（不为 None）
def exists(v):
    # 返回 v 是否不为 None
    return v is not None

# 将输入 t 转换为成对的元组
def pair(t):
    # 如果 t 是元组，直接返回；否则返回重复的元组
    return t if isinstance(t, tuple) else (t, t)

# 获取时间步嵌入
def get_timestep_embedding(timesteps, embedding_dim):
    """
    该实现与 Denoising Diffusion Probabilistic Models 中的实现相匹配：
    来自 Fairseq。
    构建正弦嵌入。
    与 tensor2tensor 中的实现相匹配，但与“Attention Is All You Need”第 3.5 节的描述略有不同。
    """
    # 确保 timesteps 是一维数组
    assert len(timesteps.shape) == 1

    # 计算一半维度
    half_dim = embedding_dim // 2
    # 计算正弦嵌入的缩放因子
    emb = math.log(10000) / (half_dim - 1)
    # 生成正弦嵌入
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将嵌入移动到与 timesteps 相同的设备
    emb = emb.to(device=timesteps.device)
    # 根据时间步生成嵌入
    emb = timesteps.float()[:, None] * emb[None, :]
    # 拼接正弦和余弦嵌入
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度为奇数，则填充一个零
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回嵌入结果
    return emb

# 应用非线性函数（swish）
def nonlinearity(x):
    # swish 激活函数
    return x * torch.sigmoid(x)

# 创建 LeakyReLU 激活函数
def leaky_relu(p=0.1):
    # 返回带有给定负斜率的 LeakyReLU 实例
    return nn.LeakyReLU(p)

# 在并行计算中拆分输入
def _split(input_, dim):
    # 获取并行计算的世界大小
    cp_world_size = get_context_parallel_world_size()

    # 如果并行计算的世界大小为 1，直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取当前并行计算的排名
    cp_rank = get_context_parallel_rank()

    # print('in _split, cp_rank:', cp_rank, 'input_size:', input_.shape)

    # 获取输入的第一帧，并保持连续性
    inpu_first_frame_ = input_.transpose(0, dim)[:1].transpose(0, dim).contiguous()
    # 更新输入，去掉第一帧
    input_ = input_.transpose(0, dim)[1:].transpose(0, dim).contiguous()
    # 计算每个并行计算的维度大小
    dim_size = input_.size()[dim] // cp_world_size

    # 按指定维度拆分输入
    input_list = torch.split(input_, dim_size, dim=dim)
    # 获取当前排名对应的输出
    output = input_list[cp_rank]

    # 如果当前排名为 0，拼接第一帧
    if cp_rank == 0:
        output = torch.cat([inpu_first_frame_, output], dim=dim)
    # 确保输出是连续的
    output = output.contiguous()

    # print('out _split, cp_rank:', cp_rank, 'output_size:', output.shape)

    # 返回拆分后的输出
    return output

# 在并行计算中收集输入
def _gather(input_, dim):
    # 获取并行计算的世界大小
    cp_world_size = get_context_parallel_world_size()

    # 如果并行计算的世界大小为 1，直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取并行计算组
    group = get_context_parallel_group()
    # 获取当前并行计算的排名
    cp_rank = get_context_parallel_rank()

    # print('in _gather, cp_rank:', cp_rank, 'input_size:', input_.shape)

    # 获取输入的第一帧，并保持连续性
    input_first_frame_ = input_.transpose(0, dim)[:1].transpose(0, dim).contiguous()
    # 如果当前排名为 0，更新输入，去掉第一帧
    if cp_rank == 0:
        input_ = input_.transpose(0, dim)[1:].transpose(0, dim).contiguous()

    # 创建一个包含空张量的列表，用于收集输入
    tensor_list = [torch.empty_like(torch.cat([input_first_frame_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]

    # 如果当前排名为 0，拼接第一帧到输入中
    if cp_rank == 0:
        input_ = torch.cat([input_first_frame_, input_], dim=dim)
    # 将输入张量存入指定的 tensor_list 中，索引由 cp_rank 确定
    tensor_list[cp_rank] = input_
    # 从所有进程中收集输入张量，并将结果存入 tensor_list 中，使用指定的分组
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # 将 tensor_list 中的所有张量在指定维度上连接成一个新的张量，并确保内存是连续的
    output = torch.cat(tensor_list, dim=dim).contiguous()

    # 调试输出当前进程的 cp_rank 和输出张量的尺寸（此行已被注释）
    # print('out _gather, cp_rank:', cp_rank, 'output_size:', output.shape)

    # 返回连接后的输出张量
    return output
# 定义函数 _conv_split，接收输入张量、维度和卷积核大小
def _conv_split(input_, dim, kernel_size):
    # 获取当前并行上下文的进程数量
    cp_world_size = get_context_parallel_world_size()

    # 如果并行上下文进程数为 1，则直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取当前进程在并行上下文中的排名
    cp_rank = get_context_parallel_rank()

    # 计算每个进程处理的维度大小
    dim_size = (input_.size()[dim] - kernel_size) // cp_world_size

    # 如果当前进程是 0 号进程，处理输入的前一部分
    if cp_rank == 0:
        output = input_.transpose(dim, 0)[: dim_size + kernel_size].transpose(dim, 0)
    else:
        # 其他进程处理输入的对应部分
        output = input_.transpose(dim, 0)[cp_rank * dim_size + 1 : (cp_rank + 1) * dim_size + kernel_size].transpose(
            dim, 0
        )
    # 确保输出张量在内存中的连续性
    output = output.contiguous()

    # 返回处理后的输出
    return output


# 定义函数 _conv_gather，接收输入张量、维度和卷积核大小
def _conv_gather(input_, dim, kernel_size):
    # 获取当前并行上下文的进程数量
    cp_world_size = get_context_parallel_world_size()

    # 如果并行上下文进程数为 1，则直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取当前的并行组
    group = get_context_parallel_group()
    # 获取当前进程在并行上下文中的排名
    cp_rank = get_context_parallel_rank()

    # 处理输入的第一部分卷积核
    input_first_kernel_ = input_.transpose(0, dim)[:kernel_size].transpose(0, dim).contiguous()
    # 如果当前进程是 0 号进程，处理剩余的输入
    if cp_rank == 0:
        input_ = input_.transpose(0, dim)[kernel_size:].transpose(0, dim).contiguous()
    else:
        # 其他进程处理相应的输入部分
        input_ = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim).contiguous()

    # 创建张量列表以存储各进程的输出
    tensor_list = [torch.empty_like(torch.cat([input_first_kernel_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]
    # 如果当前进程是 0 号进程，合并输入
    if cp_rank == 0:
        input_ = torch.cat([input_first_kernel_, input_], dim=dim)

    # 将当前进程的输入保存到张量列表中
    tensor_list[cp_rank] = input_
    # 收集所有进程的输入到张量列表
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # 合并张量列表中的所有输出，确保输出在内存中连续
    output = torch.cat(tensor_list, dim=dim).contiguous()

    # 返回处理后的输出
    return output
```