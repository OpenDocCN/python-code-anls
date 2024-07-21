# `.\pytorch\torch\distributed\pipelining\_utils.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入日志模块
import logging
# 导入 dataclass 模块，用于定义数据类
from dataclasses import dataclass
# 导入类型提示模块
from typing import List, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 torch 模块中导入 fx 子模块
from torch import fx

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


def flatten_args_detach(args):
    """
    Flatten the args into a list form and detach the tensors from computational graph.
    将参数 args 扁平化为列表形式，并从计算图中分离张量。
    """
    # 存储分离后的扁平化参数
    flat_detached_args = []

    def extract_tensor_args(a):
        nonlocal flat_detached_args
        # 如果参数 a 是 torch.Tensor 类型，则分离它并保留梯度信息
        if isinstance(a, torch.Tensor):
            val = a.detach().requires_grad_(a.requires_grad)
            flat_detached_args.append(val)
            return val
        else:
            flat_detached_args.append(a)
            return a

    # 使用 fx.node.map_aggregate 函数将 extract_tensor_args 应用于 args 中的每个元素
    new_args = fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return new_args, flat_detached_args


def flatten_args(args):
    """
    Flatten the args into a list form.
    将参数 args 扁平化为列表形式。
    """
    # 存储扁平化后的参数
    flat_args = []

    def extract_tensor_args(a):
        nonlocal flat_args
        # 将参数 a 添加到 flat_args 中
        flat_args.append(a)
        return a

    # 使用 fx.node.map_aggregate 函数将 extract_tensor_args 应用于 args 中的每个元素
    fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return flat_args


class PipeliningShapeError(RuntimeError):
    """Shape mismatch between configured and runtime values."""
    # 自定义异常类，用于捕获配置和运行时数值之间的形状不匹配错误


def validate_tensor_metadata(desc, expected, given):
    # 验证张量元数据，包括形状、数据类型和步幅
    if not expected.shape == given.shape:
        raise PipeliningShapeError(
            f"{desc} has a shape mismatch: expected {expected.shape} actual {given.shape}"
        )
    if not expected.dtype == given.dtype:
        raise PipeliningShapeError(
            f"{desc} has a dtype mismatch: expected {expected.dtype} actual {given.dtype}"
        )
    if not expected.stride() == given.stride():
        raise PipeliningShapeError(
            f"{desc} has a stride mismatch: expected {expected.stride()} actual {given.stride()}"
        )


def validate_tensors_metadata(
    desc,
    expected_tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    actual_tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
):
    # 验证张量元数据列表，确保数量和每个张量的元数据匹配
    if len(expected_tensors) != len(actual_tensors):
        raise PipeliningShapeError(
            f"{desc}: Number of values ({len(actual_tensors)}) does not match expected number ({len(expected_tensors)})"
        )
    for i in range(len(expected_tensors)):
        # 针对每个张量进行 validate_tensor_metadata 的验证
        validate_tensor_metadata(
            f"{desc}: value {i}", expected_tensors[i], actual_tensors[i]
        )


@dataclass
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    用于捕获管道（`Pipe` 对象）的信息。
    """

    # 管道的计算图
    graph: fx.Graph
    # 管道的阶段数量
    num_stages: int
    # 是否包含损失函数和反向传播
    has_loss_and_backward: bool
```