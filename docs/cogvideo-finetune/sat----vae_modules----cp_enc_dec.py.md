# `.\cogvideo-finetune\sat\vae_modules\cp_enc_dec.py`

```py
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 导入分布式训练模块
import torch.distributed
# 导入神经网络模块
import torch.nn as nn
# 导入神经网络功能模块
import torch.nn.functional as F
# 导入 NumPy 库
import numpy as np

# 从 beartype 导入装饰器和类型
from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List
# 从 einops 导入重排功能
from einops import rearrange

# 从自定义模块中导入上下文相关函数
from sgm.util import (
    get_context_parallel_group,  # 获取并行组
    get_context_parallel_rank,   # 获取当前并行任务的排名
    get_context_parallel_world_size,  # 获取并行世界的大小
    get_context_parallel_group_rank,   # 获取当前组的排名
)

# 尝试导入 SafeConv3d，如果失败则注释掉
# try:
from vae_modules.utils import SafeConv3d as Conv3d  # 从 utils 中导入安全的 3D 卷积
# except:
#     # 如果 SafeConv3d 不可用，则降级为普通的 Conv3d
#     from torch.nn import Conv3d  # 从 PyTorch 导入标准的 3D 卷积


def cast_tuple(t, length=1):
    # 如果 t 不是元组，则返回由 t 组成的元组，长度为 length
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    # 检查 num 是否能被 den 整除
    return (num % den) == 0


def is_odd(n):
    # 检查 n 是否为奇数
    return not divisible_by(n, 2)


def exists(v):
    # 检查 v 是否存在（不为 None）
    return v is not None


def pair(t):
    # 如果 t 不是元组，则返回一个由 t 组成的元组
    return t if isinstance(t, tuple) else (t, t)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    该函数实现与去噪扩散概率模型中的相匹配：
    来自 Fairseq。
    构建正弦嵌入。
    该实现与 tensor2tensor 中的相匹配，但与“Attention Is All You Need”第 3.5 节中的描述略有不同。
    """
    # 确保 timesteps 是一维的
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2  # 计算嵌入维度的一半
    emb = math.log(10000) / (half_dim - 1)  # 计算基础值
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)  # 计算正弦和余弦嵌入的值
    emb = emb.to(device=timesteps.device)  # 将嵌入移动到与 timesteps 相同的设备
    emb = timesteps.float()[:, None] * emb[None, :]  # 根据时间步调整嵌入
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # 合并正弦和余弦嵌入
    if embedding_dim % 2 == 1:  # 如果嵌入维度为奇数，则进行零填充
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb  # 返回最终的嵌入


def nonlinearity(x):
    # 实现 swish 非线性激活函数
    return x * torch.sigmoid(x)


def leaky_relu(p=0.1):
    # 返回具有指定负斜率的 Leaky ReLU 激活函数
    return nn.LeakyReLU(p)


def _split(input_, dim):
    cp_world_size = get_context_parallel_world_size()  # 获取并行世界的大小

    if cp_world_size == 1:  # 如果只有一个并行任务，直接返回输入
        return input_

    cp_rank = get_context_parallel_rank()  # 获取当前并行任务的排名

    # print('in _split, cp_rank:', cp_rank, 'input_size:', input_.shape)

    # 获取输入的第一帧并进行转换
    inpu_first_frame_ = input_.transpose(0, dim)[:1].transpose(0, dim).contiguous()
    input_ = input_.transpose(0, dim)[1:].transpose(0, dim).contiguous()  # 处理输入，去除第一帧
    dim_size = input_.size()[dim] // cp_world_size  # 计算分割后的维度大小

    input_list = torch.split(input_, dim_size, dim=dim)  # 根据维度大小进行分割
    output = input_list[cp_rank]  # 获取当前任务对应的输出

    if cp_rank == 0:  # 如果是第一个任务，将第一帧和输出合并
        output = torch.cat([inpu_first_frame_, output], dim=dim)
    output = output.contiguous()  # 确保输出是连续的内存块

    # print('out _split, cp_rank:', cp_rank, 'output_size:', output.shape)

    return output  # 返回最终的输出


def _gather(input_, dim):
    cp_world_size = get_context_parallel_world_size()  # 获取并行世界的大小

    # 如果只有一个并行任务，直接返回输入
    if cp_world_size == 1:
        return input_

    group = get_context_parallel_group()  # 获取并行组
    cp_rank = get_context_parallel_rank()  # 获取当前并行任务的排名

    # print('in _gather, cp_rank:', cp_rank, 'input_size:', input_.shape)

    # 获取输入的第一帧并进行转换
    input_first_frame_ = input_.transpose(0, dim)[:1].transpose(0, dim).contiguous()
    # 如果当前进程的排名为 0
        if cp_rank == 0:
            # 对输入张量进行转置，取出第一个维度之后的所有数据，再转置回原形状
            input_ = input_.transpose(0, dim)[1:].transpose(0, dim).contiguous()
    
        # 创建一个张量列表，包含一个空张量和 cp_world_size - 1 个与 input_ 相同形状的空张量
        tensor_list = [torch.empty_like(torch.cat([input_first_frame_, input_], dim=dim))] + [
            torch.empty_like(input_) for _ in range(cp_world_size - 1)
        ]
    
        # 如果当前进程的排名为 0
        if cp_rank == 0:
            # 将第一个帧的输入与当前输入张量在指定维度上拼接
            input_ = torch.cat([input_first_frame_, input_], dim=dim)
    
        # 将当前进程的输入张量赋值给张量列表中的相应位置
        tensor_list[cp_rank] = input_
        # 在所有进程之间收集输入张量，存入张量列表
        torch.distributed.all_gather(tensor_list, input_, group=group)
    
        # 在指定维度上拼接所有收集到的张量，并返回连续的内存块
        output = torch.cat(tensor_list, dim=dim).contiguous()
    
        # 可能用于调试输出当前进程的排名和输出张量的形状
        # print('out _gather, cp_rank:', cp_rank, 'output_size:', output.shape)
    
        # 返回最终的输出张量
        return output
# 定义一个用于分割输入的函数，参数包括输入张量、维度和卷积核大小
def _conv_split(input_, dim, kernel_size):
    # 获取并行环境中的总进程数量
    cp_world_size = get_context_parallel_world_size()

    # 如果进程数量为1，则直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取当前进程的排名
    cp_rank = get_context_parallel_rank()

    # 计算在指定维度上，每个进程处理的元素数量
    dim_size = (input_.size()[dim] - kernel_size) // cp_world_size

    # 对于第一个进程，计算输出张量
    if cp_rank == 0:
        output = input_.transpose(dim, 0)[: dim_size + kernel_size].transpose(dim, 0)
    else:
        # 对于其他进程，计算输出张量
        output = input_.transpose(dim, 0)[
            cp_rank * dim_size + kernel_size : (cp_rank + 1) * dim_size + kernel_size
        ].transpose(dim, 0)
    # 确保输出张量在内存中是连续的
    output = output.contiguous()

    # 返回输出张量
    return output


# 定义一个用于聚合输入的函数，参数包括输入张量、维度和卷积核大小
def _conv_gather(input_, dim, kernel_size):
    # 获取并行环境中的总进程数量
    cp_world_size = get_context_parallel_world_size()

    # 如果进程数量为1，则直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取当前进程的组和排名
    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()

    # 获取输入张量的首个卷积核并确保是连续的
    input_first_kernel_ = input_.transpose(0, dim)[:kernel_size].transpose(0, dim).contiguous()
    # 对于第一个进程，更新输入张量
    if cp_rank == 0:
        input_ = input_.transpose(0, dim)[kernel_size:].transpose(0, dim).contiguous()
    else:
        # 对于其他进程，处理输入张量
        input_ = input_.transpose(0, dim)[max(kernel_size - 1, 0) :].transpose(0, dim).contiguous()

    # 创建一个张量列表，用于存储各进程的输入
    tensor_list = [torch.empty_like(torch.cat([input_first_kernel_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]
    # 对于第一个进程，合并张量
    if cp_rank == 0:
        input_ = torch.cat([input_first_kernel_, input_], dim=dim)

    # 将当前进程的输入放入列表
    tensor_list[cp_rank] = input_
    # 收集所有进程的输入
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # 合并张量列表并确保是连续的
    output = torch.cat(tensor_list, dim=dim).contiguous()

    # 返回输出张量
    return output


# 定义一个用于从前一个进程传递输入的函数，参数包括输入张量、维度和卷积核大小
def _pass_from_previous_rank(input_, dim, kernel_size):
    # 如果卷积核大小为1，则直接返回输入
    if kernel_size == 1:
        return input_

    # 获取当前进程的组、排名和总进程数量
    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    cp_group_rank = get_context_parallel_group_rank()
    cp_world_size = get_context_parallel_world_size()

    # 获取全局进程排名和数量
    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()

    # 将输入张量在指定维度上转置
    input_ = input_.transpose(0, dim)

    # 确定发送和接收的进程排名
    send_rank = global_rank + 1
    recv_rank = global_rank - 1
    # 如果发送排名能被总进程数量整除，则调整发送排名
    if send_rank % cp_world_size == 0:
        send_rank -= cp_world_size
    # 检查接收进程的排名，决定是否需要增加排名值以避免冲突
        if recv_rank % cp_world_size == cp_world_size - 1:
            recv_rank += cp_world_size
    
        # 检查当前进程排名，是否小于世界大小减一
        if cp_rank < cp_world_size - 1:
            # 发送输入的最后一部分到指定发送进程
            req_send = torch.distributed.isend(input_[-kernel_size + 1 :].contiguous(), send_rank, group=group)
        # 检查当前进程排名，是否大于零
        if cp_rank > 0:
            # 创建与输入相同形状的空缓冲区以接收数据
            recv_buffer = torch.empty_like(input_[-kernel_size + 1 :]).contiguous()
            # 异步接收数据到接收缓冲区
            req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)
    
        # 检查当前进程是否为主进程
        if cp_rank == 0:
            # 将输入的第一部分重复并与原始输入连接
            input_ = torch.cat([input_[:1]] * (kernel_size - 1) + [input_], dim=0)
        else:
            # 等待接收请求完成
            req_recv.wait()
            # 将接收到的缓冲区与原始输入连接
            input_ = torch.cat([recv_buffer, input_], dim=0)
    
        # 转置输入张量的维度以适应后续操作
        input_ = input_.transpose(0, dim).contiguous()
    
        # 打印当前进程排名及输入大小（调试用）
        # print('out _pass_from_previous_rank, cp_rank:', cp_rank, 'input_size:', input_.shape)
    
        # 返回最终处理后的输入张量
        return input_
# 定义一个私有函数，用于从之前的并行排名获取数据，处理卷积运算
def _fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding=None):
    # 如果卷积核大小为1，则直接返回输入数据
    if kernel_size == 1:
        return input_

    # 获取当前并行上下文的组
    group = get_context_parallel_group()
    # 获取当前的并行排名
    cp_rank = get_context_parallel_rank()
    # 获取当前组的排名
    cp_group_rank = get_context_parallel_group_rank()
    # 获取当前并行组的世界大小
    cp_world_size = get_context_parallel_world_size()

    # print('in _pass_from_previous_rank, cp_rank:', cp_rank, 'input_size:', input_.shape)

    # 获取全局排名和全局世界大小
    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()

    # 在指定维度上转置输入数据
    input_ = input_.transpose(0, dim)

    # 从上一个排名传递数据
    send_rank = global_rank + 1  # 发送给下一个排名
    recv_rank = global_rank - 1  # 接收来自上一个排名的数据
    # 如果发送排名超出范围，则调整为循环发送
    if send_rank % cp_world_size == 0:
        send_rank -= cp_world_size
    # 如果接收排名是最后一个，则调整为循环接收
    if recv_rank % cp_world_size == cp_world_size - 1:
        recv_rank += cp_world_size

    # 创建接收缓冲区，用于存储接收到的数据
    recv_buffer = torch.empty_like(input_[-kernel_size + 1 :]).contiguous()
    # 如果当前排名小于最后一个，则发送数据
    if cp_rank < cp_world_size - 1:
        req_send = torch.distributed.isend(input_[-kernel_size + 1 :].contiguous(), send_rank, group=group)
    # 如果当前排名大于0，则接收数据
    if cp_rank > 0:
        req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)

    # 如果当前排名是0，则处理输入数据
    if cp_rank == 0:
        # 如果有缓存填充，则将其与输入数据拼接
        if cache_padding is not None:
            input_ = torch.cat([cache_padding.transpose(0, dim).to(input_.device), input_], dim=0)
        # 否则，重复输入数据的第一项以填充
        else:
            input_ = torch.cat([input_[:1]] * (kernel_size - 1) + [input_], dim=0)
    else:
        # 等待接收请求完成，然后拼接接收到的数据
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)

    # 再次转置输入数据并确保其内存连续
    input_ = input_.transpose(0, dim).contiguous()
    # 返回处理后的输入数据
    return input_


# 定义一个私有函数，用于从之前的排名丢弃数据，处理卷积运算
def _drop_from_previous_rank(input_, dim, kernel_size):
    # 转置输入数据，然后丢弃前 kernel_size - 1 个元素
    input_ = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim)
    # 返回处理后的输入数据
    return input_


# 定义一个类，继承自 torch.autograd.Function，表示卷积散射操作
class _ConvolutionScatterToContextParallelRegion(torch.autograd.Function):
    @staticmethod
    # 前向传播函数
    def forward(ctx, input_, dim, kernel_size):
        # 保存维度和卷积核大小到上下文
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        # 调用分卷积函数进行处理
        return _conv_split(input_, dim, kernel_size)

    @staticmethod
    # 反向传播函数
    def backward(ctx, grad_output):
        # 调用收集卷积函数进行处理，并返回结果
        return _conv_gather(grad_output, ctx.dim, ctx.kernel_size), None, None


# 定义一个类，继承自 torch.autograd.Function，表示卷积收集操作
class _ConvolutionGatherFromContextParallelRegion(torch.autograd.Function):
    @staticmethod
    # 前向传播函数
    def forward(ctx, input_, dim, kernel_size):
        # 保存维度和卷积核大小到上下文
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        # 调用收集卷积函数进行处理
        return _conv_gather(input_, dim, kernel_size)

    @staticmethod
    # 反向传播函数
    def backward(ctx, grad_output):
        # 调用分卷积函数进行处理，并返回结果
        return _conv_split(grad_output, ctx.dim, ctx.kernel_size), None, None
# 定义一个用于前一秩的卷积操作的自定义 PyTorch 函数
class _ConvolutionPassFromPreviousRank(torch.autograd.Function):
    # 定义前向传播的静态方法
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        # 将维度和卷积核大小存储在上下文中以供后向传播使用
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        # 调用外部函数执行前向卷积操作并返回结果
        return _pass_from_previous_rank(input_, dim, kernel_size)

    # 定义后向传播的静态方法
    @staticmethod
    def backward(ctx, grad_output):
        # 调用外部函数处理梯度并返回，None 表示不返回额外的梯度
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size), None, None


# 定义一个用于前一秩的假 CP 卷积操作的自定义 PyTorch 函数
class _FakeCPConvolutionPassFromPreviousRank(torch.autograd.Function):
    # 定义前向传播的静态方法
    @staticmethod
    def forward(ctx, input_, dim, kernel_size, cache_padding):
        # 将维度、卷积核大小和缓存填充存储在上下文中
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        # 调用外部函数执行假 CP 卷积操作并返回结果
        return _fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding)

    # 定义后向传播的静态方法
    @staticmethod
    def backward(ctx, grad_output):
        # 调用外部函数处理梯度并返回，None 表示不返回额外的梯度
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size), None, None, None


# 定义一个函数用于将输入数据分散到上下文并进行并行区域处理
def conv_scatter_to_context_parallel_region(input_, dim, kernel_size):
    # 调用自定义函数并返回结果
    return _ConvolutionScatterToContextParallelRegion.apply(input_, dim, kernel_size)


# 定义一个函数用于从上下文并行区域汇聚输入数据
def conv_gather_from_context_parallel_region(input_, dim, kernel_size):
    # 调用自定义函数并返回结果
    return _ConvolutionGatherFromContextParallelRegion.apply(input_, dim, kernel_size)


# 定义一个函数用于从最后一秩进行卷积传递
def conv_pass_from_last_rank(input_, dim, kernel_size):
    # 调用自定义函数并返回结果
    return _ConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size)


# 定义一个函数用于进行假 CP 从前一秩的传递
def fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding):
    # 调用自定义函数并返回结果
    return _FakeCPConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size, cache_padding)


# 定义一个 3D 因果卷积的上下文并行模块
class ContextParallelCausalConv3d(nn.Module):
    # 初始化模块，设置输入输出通道、卷积核大小、步幅等参数
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], stride=1, **kwargs):
        # 调用父类初始化方法
        super().__init__()
        # 将卷积核大小转换为元组，确保有三个维度
        kernel_size = cast_tuple(kernel_size, 3)

        # 分别获取时间、高度和宽度的卷积核大小
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # 确保高度和宽度的卷积核大小为奇数，以便中心对齐
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        # 计算填充大小，以保持卷积输出的维度
        time_pad = time_kernel_size - 1
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        # 存储填充和卷积核的相关参数
        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_kernel_size = time_kernel_size
        self.temporal_dim = 2

        # 设置步幅和扩张参数为三维的相同值
        stride = (stride, stride, stride)
        dilation = (1, 1, 1)
        # 初始化 3D 卷积层
        self.conv = Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        # 初始化缓存填充为 None
        self.cache_padding = None
    # 定义前向传播函数，接受输入数据和清除缓存的标志
        def forward(self, input_, clear_cache=True):
            # 如果输入的形状第三维为1，处理图像数据
            #     # 对第一帧进行填充
            #     input_parallel = torch.cat([input_] * self.time_kernel_size, dim=2)
            # else:
            #     # 从最后一维进行卷积处理
            #     input_parallel = conv_pass_from_last_rank(input_, self.temporal_dim, self.time_kernel_size)
    
            # 设置2D填充的大小
            # padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            # 对输入进行填充，填充值为0
            # input_parallel = F.pad(input_parallel, padding_2d, mode = 'constant', value = 0)
    
            # 对填充后的输入进行卷积操作
            # output_parallel = self.conv(input_parallel)
            # 赋值输出
            # output = output_parallel
            # 返回输出结果
            # return output
    
            # 从上一个层的输出中获取并处理输入，添加时间维度和缓存填充
            input_parallel = fake_cp_pass_from_previous_rank(
                input_, self.temporal_dim, self.time_kernel_size, self.cache_padding
            )
    
            # 删除旧的缓存填充
            del self.cache_padding
            # 将缓存填充设置为None
            self.cache_padding = None
            # 如果不清除缓存
            if not clear_cache:
                # 获取并行计算的当前排名和总大小
                cp_rank, cp_world_size = get_context_parallel_rank(), get_context_parallel_world_size()
                # 获取全局排名
                global_rank = torch.distributed.get_rank()
                # 如果并行计算的大小为1
                if cp_world_size == 1:
                    # 保存最后一帧的缓存填充
                    self.cache_padding = (
                        input_parallel[:, :, -self.time_kernel_size + 1 :].contiguous().detach().clone().cpu()
                    )
                else:
                    # 如果是最后一个并行计算的排名
                    if cp_rank == cp_world_size - 1:
                        # 发送最后一帧数据到下一个全局排名
                        torch.distributed.isend(
                            input_parallel[:, :, -self.time_kernel_size + 1 :].contiguous(),
                            global_rank + 1 - cp_world_size,
                            group=get_context_parallel_group(),
                        )
                    # 如果是第一个并行计算的排名
                    if cp_rank == 0:
                        # 创建接收缓存并接收数据
                        recv_buffer = torch.empty_like(input_parallel[:, :, -self.time_kernel_size + 1 :]).contiguous()
                        torch.distributed.recv(
                            recv_buffer, global_rank - 1 + cp_world_size, group=get_context_parallel_group()
                        )
                        # 保存接收的数据作为缓存填充
                        self.cache_padding = recv_buffer.contiguous().detach().clone().cpu()
    
            # 设置2D填充的大小
            padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            # 对输入进行填充，填充值为0
            input_parallel = F.pad(input_parallel, padding_2d, mode="constant", value=0)
    
            # 对填充后的输入进行卷积操作
            output_parallel = self.conv(input_parallel)
            # 赋值输出
            output = output_parallel
            # 返回输出结果
            return output
# 定义一个名为 ContextParallelGroupNorm 的类，继承自 torch.nn.GroupNorm
class ContextParallelGroupNorm(torch.nn.GroupNorm):
    # 定义前向传播方法
    def forward(self, input_):
        # 检查输入的第三个维度大小是否大于1，用于决定是否进行上下文并行处理
        gather_flag = input_.shape[2] > 1
        # 如果需要进行上下文并行处理
        if gather_flag:
            # 从上下文并行区域聚合输入数据，维度为2，卷积核大小为1
            input_ = conv_gather_from_context_parallel_region(input_, dim=2, kernel_size=1)
        # 调用父类的前向传播方法，处理输入数据
        output = super().forward(input_)
        # 如果需要进行上下文并行处理
        if gather_flag:
            # 从上下文并行区域散播输出数据，维度为2，卷积核大小为1
            output = conv_scatter_to_context_parallel_region(output, dim=2, kernel_size=1)
        # 返回处理后的输出数据
        return output


# 定义一个函数 Normalize，接受输入通道数和其他参数
def Normalize(in_channels, gather=False, **kwargs):  # 适用于3D和2D情况
    # 如果需要聚合
    if gather:
        # 返回一个上下文并行的 GroupNorm 实例
        return ContextParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        # 返回一个普通的 GroupNorm 实例
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# 定义一个名为 SpatialNorm3D 的类，继承自 nn.Module
class SpatialNorm3D(nn.Module):
    # 定义初始化方法，接受多个参数
    def __init__(
        self,
        f_channels,
        zq_channels,
        freeze_norm_layer=False,
        add_conv=False,
        pad_mode="constant",
        gather=False,
        **norm_layer_params,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 如果需要聚合
        if gather:
            # 初始化上下文并行的归一化层
            self.norm_layer = ContextParallelGroupNorm(num_channels=f_channels, **norm_layer_params)
        else:
            # 初始化普通的归一化层
            self.norm_layer = torch.nn.GroupNorm(num_channels=f_channels, **norm_layer_params)
        # self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)  # 注释掉的代码
        # 如果需要冻结归一化层的参数
        if freeze_norm_layer:
            # 遍历归一化层的参数，将其 requires_grad 属性设置为 False
            for p in self.norm_layer.parameters:
                p.requires_grad = False

        # 保存是否添加卷积层的标志
        self.add_conv = add_conv
        # 如果需要添加卷积层
        if add_conv:
            # 初始化上下文并行的因果卷积层，输入和输出通道均为 zq_channels，卷积核大小为3
            self.conv = ContextParallelCausalConv3d(
                chan_in=zq_channels,
                chan_out=zq_channels,
                kernel_size=3,
            )

        # 初始化上下文并行的因果卷积层，输入通道为 zq_channels，输出通道为 f_channels，卷积核大小为1
        self.conv_y = ContextParallelCausalConv3d(
            chan_in=zq_channels,
            chan_out=f_channels,
            kernel_size=1,
        )
        # 初始化另一个上下文并行的因果卷积层，参数同上
        self.conv_b = ContextParallelCausalConv3d(
            chan_in=zq_channels,
            chan_out=f_channels,
            kernel_size=1,
        )
    # 定义前向传播方法，接受输入张量 f、zq 和一个可选参数
        def forward(self, f, zq, clear_fake_cp_cache=True):
            # 检查 f 的第三维度是否大于 1 且为奇数
            if f.shape[2] > 1 and f.shape[2] % 2 == 1:
                # 将 f 分为第一帧和其余帧
                f_first, f_rest = f[:, :, :1], f[:, :, 1:]
                # 获取第一帧和其余帧的大小
                f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
                # 将 zq 分为第一帧和其余帧
                zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]
                # 使用最近邻插值调整 zq_first 的大小
                zq_first = torch.nn.functional.interpolate(zq_first, size=f_first_size, mode="nearest")
                # 使用最近邻插值调整 zq_rest 的大小
                zq_rest = torch.nn.functional.interpolate(zq_rest, size=f_rest_size, mode="nearest")
                # 在第三维度上连接调整后的 zq_first 和 zq_rest
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                # 对 zq 进行最近邻插值调整，匹配 f 的大小
                zq = torch.nn.functional.interpolate(zq, size=f.shape[-3:], mode="nearest")
    
            # 如果需要，使用卷积层处理 zq
            if self.add_conv:
                zq = self.conv(zq, clear_cache=clear_fake_cp_cache)
    
            # 对输入 f 进行归一化处理
            norm_f = self.norm_layer(f)
            # norm_f = conv_scatter_to_context_parallel_region(norm_f, dim=2, kernel_size=1)
    
            # 计算新的特征 f，结合 norm_f 和 zq 的卷积输出
            new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
            # 返回新的特征 f
            return new_f
# 定义一个用于3D标准化的函数
def Normalize3D(
    # 输入通道数
    in_channels,
    # 量化通道
    zq_ch,
    # 是否添加卷积层
    add_conv,
    # 是否进行聚合操作
    gather=False,
):
    # 返回经过3D空间标准化的结果
    return SpatialNorm3D(
        # 输入通道数
        in_channels,
        # 量化通道
        zq_ch,
        # 聚合参数
        gather=gather,
        # 不冻结标准化层
        freeze_norm_layer=False,
        # 添加卷积参数
        add_conv=add_conv,
        # 组数设置
        num_groups=32,
        # 防止除零的微小值
        eps=1e-6,
        # 启用仿射变换
        affine=True,
    )


# 定义一个3D上采样的神经网络模块
class Upsample3D(nn.Module):
    # 初始化方法
    def __init__(
        # 输入通道数
        self,
        in_channels,
        # 是否添加卷积层
        with_conv,
        # 是否压缩时间维度
        compress_time=False,
    ):
        # 调用父类构造方法
        super().__init__()
        # 设置卷积参数
        self.with_conv = with_conv
        # 如果需要卷积层
        if self.with_conv:
            # 创建一个卷积层
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 设置时间压缩参数
        self.compress_time = compress_time

    # 前向传播方法
    def forward(self, x):
        # 如果压缩时间且时间维度大于1
        if self.compress_time and x.shape[2] > 1:
            # 如果时间维度是奇数
            if x.shape[2] % 2 == 1:
                # 分离第一帧
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]

                # 对第一帧进行插值上采样
                x_first = torch.nn.functional.interpolate(x_first, scale_factor=2.0, mode="nearest")
                # 对剩余帧进行插值上采样
                x_rest = torch.nn.functional.interpolate(x_rest, scale_factor=2.0, mode="nearest")
                # 将第一帧和剩余帧合并
                x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2)
            else:
                # 对所有帧进行插值上采样
                x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

        else:
            # 仅进行2D插值上采样
            t = x.shape[2]
            # 调整维度以便插值处理
            x = rearrange(x, "b c t h w -> (b t) c h w")
            # 进行插值上采样
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            # 调整维度回到原形状
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        # 如果需要卷积层
        if self.with_conv:
            t = x.shape[2]
            # 调整维度以便卷积处理
            x = rearrange(x, "b c t h w -> (b t) c h w")
            # 进行卷积操作
            x = self.conv(x)
            # 调整维度回到原形状
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        # 返回处理后的张量
        return x


# 定义一个3D下采样的神经网络模块
class DownSample3D(nn.Module):
    # 初始化方法
    def __init__(self, in_channels, with_conv, compress_time=False, out_channels=None):
        # 调用父类构造方法
        super().__init__()
        # 设置卷积参数
        self.with_conv = with_conv
        # 如果未指定输出通道数，使用输入通道数
        if out_channels is None:
            out_channels = in_channels
        # 如果需要卷积层
        if self.with_conv:
            # 因为 PyTorch 的卷积不支持不对称填充，手动处理填充
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        # 设置时间压缩参数
        self.compress_time = compress_time
    # 定义前向传播函数，输入为 x
        def forward(self, x):
            # 如果启用了压缩且输入的时间维度大于 1
            if self.compress_time and x.shape[2] > 1:
                # 获取输入的高和宽
                h, w = x.shape[-2:]
                # 重新排列张量的维度
                x = rearrange(x, "b c t h w -> (b h w) c t")
    
                # 如果最后一个维度的大小为奇数
                if x.shape[-1] % 2 == 1:
                    # 分离第一帧和其余帧
                    x_first, x_rest = x[..., 0], x[..., 1:]
    
                    # 如果其余帧的时间维度大于 0，进行平均池化
                    if x_rest.shape[-1] > 0:
                        x_rest = torch.nn.functional.avg_pool1d(x_rest, kernel_size=2, stride=2)
                    # 将第一帧与池化后的其余帧连接
                    x = torch.cat([x_first[..., None], x_rest], dim=-1)
                    # 重新排列张量的维度
                    x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)
                else:
                    # 对输入进行平均池化
                    x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
                    # 重新排列张量的维度
                    x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)
    
            # 如果启用了卷积操作
            if self.with_conv:
                # 定义填充的大小
                pad = (0, 1, 0, 1)
                # 对输入进行填充
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                # 获取时间维度的大小
                t = x.shape[2]
                # 重新排列张量的维度
                x = rearrange(x, "b c t h w -> (b t) c h w")
                # 进行卷积操作
                x = self.conv(x)
                # 重新排列张量的维度
                x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            else:
                # 获取时间维度的大小
                t = x.shape[2]
                # 重新排列张量的维度
                x = rearrange(x, "b c t h w -> (b t) c h w")
                # 对输入进行平均池化
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
                # 重新排列张量的维度
                x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            # 返回处理后的张量
            return x
# 定义一个三维卷积的上下文并行残差块，继承自 nn.Module
class ContextParallelResnetBlock3D(nn.Module):
    # 初始化函数，设置各项参数
    def __init__(
        self,
        *,
        in_channels,  # 输入通道数
        out_channels=None,  # 输出通道数，可选
        conv_shortcut=False,  # 是否使用卷积捷径
        dropout,  # dropout 概率
        temb_channels=512,  # 时间嵌入通道数
        zq_ch=None,  # 可选的 zq 通道数
        add_conv=False,  # 是否添加卷积
        gather_norm=False,  # 是否使用聚合归一化
        normalization=Normalize,  # 归一化方法
    ):
        super().__init__()  # 调用父类的初始化方法
        self.in_channels = in_channels  # 设置输入通道数
        out_channels = in_channels if out_channels is None else out_channels  # 设置输出通道数，若未指定则与输入通道数相同
        self.out_channels = out_channels  # 保存输出通道数
        self.use_conv_shortcut = conv_shortcut  # 保存是否使用卷积捷径的标志

        # 初始化归一化层，输入通道数和其他参数
        self.norm1 = normalization(
            in_channels,
            zq_ch=zq_ch,
            add_conv=add_conv,
            gather=gather_norm,
        )

        # 创建上下文并行因果卷积层，输入和输出通道数以及卷积核大小
        self.conv1 = ContextParallelCausalConv3d(
            chan_in=in_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        # 如果时间嵌入通道数大于0，则创建线性投影层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化第二个归一化层，输出通道数和其他参数
        self.norm2 = normalization(
            out_channels,
            zq_ch=zq_ch,
            add_conv=add_conv,
            gather=gather_norm,
        )
        # 创建 dropout 层
        self.dropout = torch.nn.Dropout(dropout)
        # 创建第二个上下文并行因果卷积层，输入和输出通道数
        self.conv2 = ContextParallelCausalConv3d(
            chan_in=out_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        # 如果输入通道数与输出通道数不相等
        if self.in_channels != self.out_channels:
            # 如果使用卷积捷径，创建卷积捷径层
            if self.use_conv_shortcut:
                self.conv_shortcut = ContextParallelCausalConv3d(
                    chan_in=in_channels,
                    chan_out=out_channels,
                    kernel_size=3,
                )
            # 否则创建 1x1 卷积捷径层
            else:
                self.nin_shortcut = Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
    # 定义前向传播函数，接收输入数据x、时间嵌入temb、可选参数zq和是否清除虚假缓存的标志
    def forward(self, x, temb, zq=None, clear_fake_cp_cache=True):
        # 初始化隐藏状态h为输入x
        h = x

        # 判断self.norm1是否为GroupNorm类型（此行被注释）
        # if isinstance(self.norm1, torch.nn.GroupNorm):
        #     # 在并行区域中从上下文聚合输入h
        #     h = conv_gather_from_context_parallel_region(h, dim=2, kernel_size=1)
        # 如果zq不为None，则使用zq和清除缓存标志调用规范化层norm1
        if zq is not None:
            h = self.norm1(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)
        else:
            # 否则仅使用规范化层norm1
            h = self.norm1(h)
        # 判断self.norm1是否为GroupNorm类型（此行被注释）
        # if isinstance(self.norm1, torch.nn.GroupNorm):
        #     # 在并行区域中将输入h散射到上下文
        #     h = conv_scatter_to_context_parallel_region(h, dim=2, kernel_size=1)

        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过卷积层conv1处理h，并清除虚假缓存
        h = self.conv1(h, clear_cache=clear_fake_cp_cache)

        # 如果temb不为None，将其嵌入到h中
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        # 判断self.norm2是否为GroupNorm类型（此行被注释）
        # if isinstance(self.norm2, torch.nn.GroupNorm):
        #     # 在并行区域中从上下文聚合输入h
        #     h = conv_gather_from_context_parallel_region(h, dim=2, kernel_size=1)
        # 如果zq不为None，则使用zq和清除缓存标志调用规范化层norm2
        if zq is not None:
            h = self.norm2(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)
        else:
            # 否则仅使用规范化层norm2
            h = self.norm2(h)
        # 判断self.norm2是否为GroupNorm类型（此行被注释）
        # if isinstance(self.norm2, torch.nn.GroupNorm):
        #     # 在并行区域中将输入h散射到上下文
        #     h = conv_scatter_to_context_parallel_region(h, dim=2, kernel_size=1)

        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过dropout层进行正则化
        h = self.dropout(h)
        # 通过卷积层conv2处理h，并清除虚假缓存
        h = self.conv2(h, clear_cache=clear_fake_cp_cache)

        # 如果输入通道数与输出通道数不同
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷方式
            if self.use_conv_shortcut:
                # 通过卷积快捷方式处理输入x，并清除虚假缓存
                x = self.conv_shortcut(x, clear_cache=clear_fake_cp_cache)
            else:
                # 否则通过nin快捷方式处理输入x
                x = self.nin_shortcut(x)

        # 返回x和h的相加结果
        return x + h
# 定义一个名为 ContextParallelEncoder3D 的类，继承自 nn.Module
class ContextParallelEncoder3D(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(
        # 使用关键字参数定义初始化需要的参数
        self,
        *,
        # 输入通道数
        ch,
        # 输出通道数
        out_ch,
        # 通道倍增的元组，控制不同层的通道数
        ch_mult=(1, 2, 4, 8),
        # 残差块的数量
        num_res_blocks,
        # 注意力分辨率
        attn_resolutions,
        # dropout 的比例，默认为 0
        dropout=0.0,
        # 是否使用卷积进行上采样，默认为 True
        resamp_with_conv=True,
        # 输入数据的通道数
        in_channels,
        # 输入数据的分辨率
        resolution,
        # 潜在空间的通道数
        z_channels,
        # 是否使用双重潜在空间，默认为 True
        double_z=True,
        # 填充模式，默认为 "first"
        pad_mode="first",
        # 时间压缩次数，默认为 4
        temporal_compress_times=4,
        # 是否收集归一化，默认为 False
        gather_norm=False,
        # 其余不需要的关键字参数
        **ignore_kwargs,
    ):
        # 调用父类构造函数
        super().__init__()
        # 设置当前类的通道数
        self.ch = ch
        # 初始化时间嵌入通道数
        self.temb_ch = 0
        # 计算分辨率数量
        self.num_resolutions = len(ch_mult)
        # 记录残差块数量
        self.num_res_blocks = num_res_blocks
        # 设置分辨率
        self.resolution = resolution
        # 设置输入通道数
        self.in_channels = in_channels

        # 计算 temporal_compress_times 的以 2 为底的对数值
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        # 初始化输入卷积层，使用 3x3 卷积
        self.conv_in = ContextParallelCausalConv3d(
            chan_in=in_channels,  # 输入通道数
            chan_out=self.ch,     # 输出通道数
            kernel_size=3,        # 卷积核大小
        )

        # 当前分辨率
        curr_res = resolution
        # 输入通道数的倍数，包含 1 作为初始值
        in_ch_mult = (1,) + tuple(ch_mult)
        # 创建一个模块列表，用于存储每个分辨率的网络层
        self.down = nn.ModuleList()
        # 遍历每个分辨率
        for i_level in range(self.num_resolutions):
            # 创建模块列表用于存储块和注意力层
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # 当前块的输入通道数
            block_in = ch * in_ch_mult[i_level]
            # 当前块的输出通道数
            block_out = ch * ch_mult[i_level]
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks):
                # 添加一个残差块到块列表
                block.append(
                    ContextParallelResnetBlock3D(
                        in_channels=block_in,   # 输入通道数
                        out_channels=block_out,  # 输出通道数
                        dropout=dropout,         # dropout 参数
                        temb_channels=self.temb_ch,  # 时间嵌入通道数
                        gather_norm=gather_norm,      # 归一化设置
                    )
                )
                # 更新输入通道数为输出通道数
                block_in = block_out
            # 创建一个新的模块，用于下采样
            down = nn.Module()
            down.block = block  # 将块赋值给下采样模块
            down.attn = attn    # 将注意力层赋值给下采样模块
            # 如果不是最后一个分辨率
            if i_level != self.num_resolutions - 1:
                # 如果当前层小于时间压缩层
                if i_level < self.temporal_compress_level:
                    # 使用卷积下采样
                    down.downsample = DownSample3D(block_in, resamp_with_conv, compress_time=True)
                else:
                    # 不使用卷积下采样
                    down.downsample = DownSample3D(block_in, resamp_with_conv, compress_time=False)
                # 更新当前分辨率为一半
                curr_res = curr_res // 2
            # 将下采样模块添加到下采样列表中
            self.down.append(down)

        # middle
        # 创建中间模块
        self.mid = nn.Module()
        # 添加第一个中间残差块
        self.mid.block_1 = ContextParallelResnetBlock3D(
            in_channels=block_in,    # 输入通道数
            out_channels=block_in,    # 输出通道数
            temb_channels=self.temb_ch,  # 时间嵌入通道数
            dropout=dropout,         # dropout 参数
            gather_norm=gather_norm,  # 归一化设置
        )

        # 添加第二个中间残差块
        self.mid.block_2 = ContextParallelResnetBlock3D(
            in_channels=block_in,    # 输入通道数
            out_channels=block_in,    # 输出通道数
            temb_channels=self.temb_ch,  # 时间嵌入通道数
            dropout=dropout,         # dropout 参数
            gather_norm=gather_norm,  # 归一化设置
        )

        # end
        # 初始化输出归一化层
        self.norm_out = Normalize(block_in, gather=gather_norm)

        # 初始化输出卷积层，使用 3x3 卷积
        self.conv_out = ContextParallelCausalConv3d(
            chan_in=block_in,                       # 输入通道数
            chan_out=2 * z_channels if double_z else z_channels,  # 输出通道数，根据条件决定
            kernel_size=3,                          # 卷积核大小
        )
    # 定义前向传播方法，接收输入 x 和其他可选参数
    def forward(self, x, **kwargs):
        # 初始化时间步嵌入为 None
        temb = None
    
        # 进行下采样操作
        h = self.conv_in(x)  # 输入通过初始卷积层处理
        for i_level in range(self.num_resolutions):  # 遍历每个分辨率级别
            for i_block in range(self.num_res_blocks):  # 遍历每个残差块
                h = self.down[i_level].block[i_block](h, temb)  # 通过当前块处理 h
                if len(self.down[i_level].attn) > 0:  # 如果有注意力机制
                    h = self.down[i_level].attn[i_block](h)  # 通过注意力机制处理 h
            if i_level != self.num_resolutions - 1:  # 如果不是最后一个分辨率级别
                h = self.down[i_level].downsample(h)  # 对 h 进行下采样
    
        # 经过中间处理
        h = self.mid.block_1(h, temb)  # 通过中间块 1 处理 h
        h = self.mid.block_2(h, temb)  # 通过中间块 2 处理 h
    
        # 最终处理
        # h = conv_gather_from_context_parallel_region(h, dim=2, kernel_size=1)  # 选择性操作，未启用
        h = self.norm_out(h)  # 对 h 进行归一化处理
        # h = conv_scatter_to_context_parallel_region(h, dim=2, kernel_size=1)  # 选择性操作，未启用
    
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.conv_out(h)  # 通过输出卷积层处理 h
    
        return h  # 返回处理后的结果
# 定义一个名为 ContextParallelDecoder3D 的类，继承自 nn.Module
class ContextParallelDecoder3D(nn.Module):
    # 初始化方法，接受多种参数用于配置
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍增因子
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力机制的分辨率
        dropout=0.0,  # dropout 概率
        resamp_with_conv=True,  # 是否使用卷积进行上采样
        in_channels,  # 输入的通道数
        resolution,  # 输入分辨率
        z_channels,  # z 的通道数
        give_pre_end=False,  # 是否给出前置结束输出
        zq_ch=None,  # 可选的 zq 通道数
        add_conv=False,  # 是否添加卷积
        pad_mode="first",  # 填充模式
        temporal_compress_times=4,  # 时间压缩次数
        gather_norm=False,  # 是否聚集归一化
        **ignorekwargs,  # 其他忽略的关键字参数
    ):
        # 省略具体的初始化实现
        pass

    # 前向传播方法，定义网络的前向计算过程
    def forward(self, z, clear_fake_cp_cache=True, **kwargs):
        # 保存输入 z 的形状，用于后续处理
        self.last_z_shape = z.shape

        # 时间步嵌入初始化为 None
        temb = None

        # 获取 z 的时间维度大小
        t = z.shape[2]
        # 将 z 赋值给 zq，准备后续处理
        zq = z
        # 使用输入 z 进行初步卷积处理，生成特征图 h
        h = self.conv_in(z, clear_cache=clear_fake_cp_cache)

        # 中间层处理
        h = self.mid.block_1(h, temb, zq, clear_fake_cp_cache=clear_fake_cp_cache)  # 通过第一个中间块处理 h
        h = self.mid.block_2(h, temb, zq, clear_fake_cp_cache=clear_fake_cp_cache)  # 通过第二个中间块处理 h

        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):  # 从高分辨率到低分辨率反向遍历
            for i_block in range(self.num_res_blocks + 1):  # 遍历每个残差块
                h = self.up[i_level].block[i_block](h, temb, zq, clear_fake_cp_cache=clear_fake_cp_cache)  # 通过上采样块处理 h
                if len(self.up[i_level].attn) > 0:  # 如果当前级别有注意力机制
                    h = self.up[i_level].attn[i_block](h, zq)  # 通过注意力机制处理 h
            if i_level != 0:  # 如果不是最后一层
                h = self.up[i_level].upsample(h)  # 对 h 进行上采样

        # 结束层处理
        if self.give_pre_end:  # 如果需要前置结束输出
            return h  # 返回当前的特征图 h

        # 归一化输出
        h = self.norm_out(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)  # 通过归一化层处理 h
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.conv_out(h, clear_cache=clear_fake_cp_cache)  # 通过输出卷积层处理 h

        return h  # 返回最终的特征图 h

    # 获取最后一层的卷积权重
    def get_last_layer(self):
        return self.conv_out.conv.weight  # 返回最后输出卷积层的权重
```