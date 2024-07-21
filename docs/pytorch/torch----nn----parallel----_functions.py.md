# `.\pytorch\torch\nn\parallel\_functions.py`

```
import warnings  # 导入警告模块
from typing import List, Optional  # 导入类型提示模块

import torch  # 导入PyTorch库
from torch._utils import _get_device_index  # 导入PyTorch内部工具函数
from torch.autograd import Function  # 导入PyTorch自动求导函数

from . import comm  # 导入当前包中的comm模块


class Broadcast(Function):
    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        assert all(
            i.device.type != "cpu" for i in inputs
        ), "Broadcast function not implemented for CPU tensors"  # 断言所有输入张量不是CPU张量
        target_gpus = [_get_device_index(x, True) for x in target_gpus]  # 获取目标GPU索引列表
        ctx.target_gpus = target_gpus  # 将目标GPU索引列表保存到上下文中
        if len(inputs) == 0:
            return tuple()
        ctx.num_inputs = len(inputs)  # 保存输入张量的数量到上下文中
        ctx.input_device = inputs[0].get_device()  # 获取第一个输入张量的设备索引
        outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)  # 调用comm模块的广播函数
        non_differentiables = []
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                for output in outputs:
                    non_differentiables.append(output[idx])  # 将不需要梯度的输出添加到非可微列表中
        ctx.mark_non_differentiable(*non_differentiables)  # 标记非可微张量
        return tuple([t for tensors in outputs for t in tensors])  # 返回输出张量的元组

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + ReduceAddCoalesced.apply(
            ctx.input_device, ctx.num_inputs, *grad_outputs
        )  # 调用ReduceAddCoalesced的反向传播


class ReduceAddCoalesced(Function):
    @staticmethod
    def forward(ctx, destination, num_inputs, *grads):
        ctx.target_gpus = [
            grads[i].get_device() for i in range(0, len(grads), num_inputs)
        ]  # 获取每个输入梯度张量的设备索引列表

        grads_ = [grads[i : i + num_inputs] for i in range(0, len(grads), num_inputs)]  # 分组输入梯度张量
        return comm.reduce_add_coalesced(grads_, destination)  # 调用comm模块的reduce_add_coalesced函数进行梯度累加

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (
            None,
            None,
        ) + Broadcast.apply(ctx.target_gpus, *grad_outputs)  # 调用Broadcast的反向传播


class Gather(Function):
    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        assert all(
            i.device.type != "cpu" for i in inputs
        ), "Gather function not implemented for CPU tensors"  # 断言所有输入张量不是CPU张量
        if target_device == "cpu":
            ctx.target_device = "cpu"  # 如果目标设备是CPU，则设置目标设备为CPU
        else:
            target_device = _get_device_index(target_device, True)  # 否则获取目标设备索引
            ctx.target_device = target_device  # 设置目标设备索引到上下文中
        ctx.dim = dim  # 保存维度信息到上下文中
        ctx.input_gpus = tuple(i.get_device() for i in inputs)  # 获取所有输入张量的设备索引组成元组
        if all(t.dim() == 0 for t in inputs) and dim == 0:
            inputs = tuple(t.view(1) for t in inputs)  # 如果所有输入张量是标量且维度为0，则展开为向量
            warnings.warn(
                "Was asked to gather along dimension 0, but all "
                "input tensors were scalars; will instead unsqueeze "
                "and return a vector."
            )
            ctx.unsqueezed_scalar = True  # 标记为展开的标量
        else:
            ctx.unsqueezed_scalar = False  # 标记为非展开的标量
        ctx.input_sizes = tuple(i.size(ctx.dim) for i in inputs)  # 获取所有输入张量在指定维度上的尺寸组成元组
        return comm.gather(inputs, ctx.dim, ctx.target_device)  # 调用comm模块的gather函数进行聚集操作
    # 定义一个反向传播函数，接受上下文和梯度输出作为参数
    def backward(ctx, grad_output):
        # 使用自定义的Scatter类的apply方法，将梯度grad_output分散到多个GPU上
        scattered_grads = Scatter.apply(
            ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output
        )
        # 如果ctx.unsqueezed_scalar为True，表示标量被unsqueeze过，需要将分散的梯度重新组织为元组
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        # 返回一个元组，前两个元素为None，后面跟着分散的梯度
        return (None, None) + scattered_grads
class Scatter(Function):
    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        # 将目标 GPU 列表中的每个设备索引获取为整数
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        # 设置上下文对象的维度和输入设备索引
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None
        if torch.cuda.is_available() and ctx.input_device == -1:
            # 在后台流中执行 CPU 到 GPU 的拷贝
            streams = [
                _get_stream(torch.device("cuda", device)) for device in target_gpus
            ]
        # 使用 comm.scatter 函数将输入张量分散到多个设备
        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)
        # 如果有后台流，则同步拷贝流
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.cuda.device(target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        # 返回分散后的输出张量列表
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        # 在反向传播时，返回空值表示无需梯度传播到输入
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)


# 用于拷贝的后台流
_streams: Optional[List[Optional[torch.Stream]]] = None


def _get_stream(device: torch.device):
    """获取用于 CPU 和目标设备之间拷贝的后台流."""
    global _streams
    if device.type == "cpu":
        return None
    device_mod = getattr(torch, device.type, None)
    if device_mod is None:
        return None
    if _streams is None:
        _streams = [None] * device_mod.device_count()
    if _streams[device.index] is None:
        _streams[device.index] = device_mod.Stream(device.index)
    return _streams[device.index]
```