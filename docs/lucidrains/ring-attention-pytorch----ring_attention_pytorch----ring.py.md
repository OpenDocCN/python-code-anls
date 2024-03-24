# `.\lucidrains\ring-attention-pytorch\ring_attention_pytorch\ring.py`

```py
# 导入必要的模块
from typing import Optional
from functools import lru_cache, partial, wraps
from collections import namedtuple

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import Function

import torch.distributed as dist

# 辅助函数

# 检查变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 将输入转换为元组，如果不是元组则重复多次
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# 缓存装饰器，用于缓存函数的结果
cache = partial(lru_cache, maxsize = None)

# 分布式全局变量

# 获取当前进程的排名
@cache()
def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

# 获取世界中进程的数量
@cache()
def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

# 判断是否处于分布式环境
@cache()
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

# 环函数

# 左循环索引
def circular_index_left(pos, ring_size, num = 1):
    return ((pos - num) + ring_size) % ring_size

# 右循环索引
def circular_index_right(pos, ring_size, num = 1):
    return (pos + num) % ring_size

# 分布式环

# 左循环排名
def circular_rank_left(rank = None, ring_size = None, num = 1):
    rank = default(rank, get_rank())
    ring_size = default(ring_size, get_world_size())
    ring_set_num = rank // ring_size
    offset = ring_set_num * ring_size
    return circular_index_left(rank, ring_size, num) + offset

# 右循环排名
def circular_rank_right(rank = None, ring_size = None, num = 1):
    rank = default(rank, get_rank())
    ring_size = default(ring_size, get_world_size())
    ring_set_num = rank // ring_size
    offset = ring_set_num * ring_size
    return circular_index_right(rank, ring_size, num) + offset

# 单次环传递

# 发送和接收数据
def send_and_receive_(x, receive_buffer, send_to_rank, receive_from_rank):
    send_op = dist.P2POp(dist.isend, x, send_to_rank)
    recv_op = dist.P2POp(dist.irecv, receive_buffer, receive_from_rank)

    reqs = dist.batch_isend_irecv([send_op, recv_op])

    for req in reqs:
        req.wait()

    dist.barrier()

# 环传递
def ring_pass(
    num_ring_passes: int,
    x: Tensor,
    receive_buffer: Optional[Tensor] = None,
    ring_size: Optional[int] = None
):
    ring_size = default(ring_size, get_world_size())
    x = x.contiguous()

    if not exists(receive_buffer):
        receive_buffer = torch.zeros_like(x)
    else:
        receive_buffer = receive_buffer.contiguous()

    send_and_receive_(x, receive_buffer, circular_rank_right(ring_size = ring_size), circular_rank_left(ring_size = ring_size))
    return receive_buffer, x

# 一次环传递
one_ring_pass = partial(ring_pass, 1)

# 迭代器，用于所有张量的所有环传递

# 环信息命名元组
RingInfo = namedtuple('RingInfo', ['ring_rank', 'iter_info'])

# 空环传递
def null_ring_pass(*tensors, max_iters = None, receive_buffers = None, ring_size = None):
    yield RingInfo(0, (True, True)), (tensors, receive_buffers)

# 所有环传递
def all_ring_pass(*tensors, max_iters = None, receive_buffers = None, ring_size = None):
    ring_size = default(ring_size, get_world_size())
    max_iters = default(max_iters, ring_size)

    receive_buffers = cast_tuple(receive_buffers, len(tensors))

    # 确保迭代次数在1和世界大小之间
    total_iters = max(1, min(ring_size, max_iters))

    curr_ring_pos = get_rank()

    for ind in range(total_iters):
        is_first = ind == 0
        is_last = ind == (total_iters - 1)

        yield RingInfo(curr_ring_pos, (is_first,  is_last)), (tensors, receive_buffers)

        curr_ring_pos = circular_index_left(curr_ring_pos, ring_size)

        if is_last:
            continue

        new_tensors = []
        new_receive_buffers = []

        for tensor, receive_buffer in zip(tensors, receive_buffers):
            if exists(tensor):
                new_tensor, new_receive_buffer = one_ring_pass(tensor, receive_buffer, ring_size)
            else:
                new_tensor, new_receive_buffer = None, None

            new_tensors.append(new_tensor)
            new_receive_buffers.append(new_receive_buffer)

        tensors = new_tensors
        receive_buffers = new_receive_buffers
```