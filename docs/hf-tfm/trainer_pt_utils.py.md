# `.\trainer_pt_utils.py`

```
# coding=utf-8
# 版权 2020-present HuggingFace Inc. 团队
#
# 根据 Apache 许可证 2.0 版本许可
# 除非符合许可证，否则不得使用此文件
# 您可以在以下地址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何明示或暗示的担保或条件
# 有关特定语言的权限，请参阅许可证
"""
用于 Trainer 类的 Torch 实用程序。
"""

import copy
import datetime
import io
import json
import math
import os
import sys
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import StreamHandler
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler

from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .tokenization_utils_base import BatchEncoding
from .utils import is_sagemaker_mp_enabled, is_torch_xla_available, is_training_run_on_sagemaker, logging

# 如果在 SageMaker 上运行训练，将日志处理器添加到标准输出流
if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))

# 如果 Torch XLA 可用，则导入相关模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

# 用于抑制 PyTorch 版本 1.4.2-1.7.0 发出的不希望的警告
try:
    from torch.optim.lr_scheduler import SAVE_STATE_WARNING
except ImportError:
    SAVE_STATE_WARNING = ""

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


def get_dataloader_sampler(dataloader):
    if hasattr(dataloader, "batch_sampler") and dataloader.batch_sampler is not None:
        return get_dataloader_sampler(dataloader.batch_sampler)
    elif hasattr(dataloader, "sampler"):
        return dataloader.sampler


def atleast_1d(tensor_or_array: Union[torch.Tensor, np.ndarray]):
    # 至少将输入的 Tensor 或数组转换为至少一维的形式
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """连接 `tensor1` 和 `tensor2` 在第一轴上，如果需要在第二轴上进行填充。"""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # 计算新的形状
    # 计算新的张量形状，其中行数为两个输入张量行数之和，列数为两个输入张量列数的最大值，其它维度保持不变
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]
    
    # 创建一个用指定值填充的新张量，形状为 new_shape，填充值为 padding_index
    result = tensor1.new_full(new_shape, padding_index)
    
    # 将 tensor1 的内容复制到结果张量的前 tensor1.shape[0] 行、tensor1.shape[1] 列的区域
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    
    # 将 tensor2 的内容复制到结果张量的从 tensor1.shape[0] 行开始、tensor2.shape[1] 列的区域
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    
    # 返回填充好的结果张量
    return result
def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    # Ensure array1 and array2 are at least 1-dimensional
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    # Check if array1 is 1-dimensional or if both arrays have the same second dimension
    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        # Concatenate array1 and array2 along the first axis
        return np.concatenate((array1, array2), axis=0)

    # Determine the new shape for the result tensor
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Create a result tensor filled with the padding_index value
    result = np.full_like(array1, padding_index, shape=new_shape)
    # Copy array1 into the appropriate position of the result tensor
    result[: array1.shape[0], : array1.shape[1]] = array1
    # Copy array2 into the appropriate position of the result tensor
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    # Ensure tensors and new_tensors have the same type
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    
    # Handle case where tensors and new_tensors are lists or tuples
    if isinstance(tensors, (list, tuple)):
        # Recursively concatenate each pair of tensors/new_tensors elements
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    # Handle case where tensors is a torch.Tensor
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    # Handle case where tensors is a Mapping (like dict)
    elif isinstance(tensors, Mapping):
        # Recursively concatenate each pair of tensors/new_tensors items
        return type(tensors)({k: nested_concat(t, new_tensors[k], padding_index=padding_index) for k, t in tensors.items()})
    # Handle case where tensors is a numpy.ndarray
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        # Raise an error for unsupported types
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def find_batch_size(tensors):
    """
    Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
    """
    # Handle case where tensors is a list or tuple
    if isinstance(tensors, (list, tuple)):
        # Recursively search for the first dimension in each element
        for t in tensors:
            result = find_batch_size(t)
            if result is not None:
                return result
    # Handle case where tensors is a Mapping (like dict)
    elif isinstance(tensors, Mapping):
        # Recursively search for the first dimension in each value
        for key, value in tensors.items():
            result = find_batch_size(value)
            if result is not None:
                return result
    # Handle case where tensors is a torch.Tensor
    elif isinstance(tensors, torch.Tensor):
        # Return the size of the first dimension of the tensor, or None if the tensor is empty
        return tensors.shape[0] if len(tensors.shape) >= 1 else None
    # Handle case where tensors is a numpy.ndarray
    elif isinstance(tensors, np.ndarray):
        # Return the size of the first dimension of the array, or None if the array is empty
        return tensors.shape[0] if len(tensors.shape) >= 1 else None


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    # Handle case where tensors is a list or tuple
    if isinstance(tensors, (list, tuple)):
        # Recursively convert each element to numpy
        return type(tensors)(nested_numpify(t) for t in tensors)
    # Handle case where tensors is a Mapping (like dict)
    if isinstance(tensors, Mapping):
        # Recursively convert each value to numpy
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    # Convert tensor to numpy array assuming it's on CPU
    t = tensors.cpu()
    # 如果张量 `t` 的数据类型是 torch.bfloat16
    if t.dtype == torch.bfloat16:
        # 截至 NumPy 1.21.4 版本，NumPy 不支持 bfloat16 数据类型（参见链接：
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ）。
        # 在 NumPy 添加对 bfloat16 的支持之前，我们需要将数据类型转换为 float32。
        t = t.to(torch.float32)
    # 将张量 `t` 转换为 NumPy 数组并返回
    return t.numpy()
# 分离 `tensors`，即使它是张量的嵌套列表/元组/字典
def nested_detach(tensors):
    if isinstance(tensors, (list, tuple)):  # 如果是列表或元组
        return type(tensors)(nested_detach(t) for t in tensors)  # 递归地对每个元素进行分离操作
    elif isinstance(tensors, Mapping):  # 如果是字典类型
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})  # 递归地对每个值进行分离操作
    return tensors.detach()  # 对单个张量进行分离操作


# 对 `tensors` 进行 XLA 网格归约操作，使用给定的 `name`
def nested_xla_mesh_reduce(tensors, name):
    if is_torch_xla_available():  # 检查是否安装了 Torch XLA
        import torch_xla.core.xla_model as xm

        if isinstance(tensors, (list, tuple)):  # 如果是列表或元组
            return type(tensors)(nested_xla_mesh_reduce(t, f"{name}_{i}") for i, t in enumerate(tensors))
        if isinstance(tensors, Mapping):  # 如果是字典类型
            return type(tensors)(
                {k: nested_xla_mesh_reduce(t, f"{name}_{i}") for i, (k, t) in enumerate(tensors.items())}
            )

        tensors = atleast_1d(tensors)  # 将张量至少视为1维张量
        return xm.mesh_reduce(name, tensors, torch.cat)  # 使用 XLA 进行网格归约操作
    else:
        raise ImportError("Torch xla must be installed to use `nested_xla_mesh_reduce`")  # 抛出导入错误


# 分布式环境下对 `tensor` 进行拼接操作，支持截断操作
def distributed_concat(tensor: Any, num_total_examples: Optional[int] = None) -> Any:
    try:
        if isinstance(tensor, (tuple, list)):  # 如果是元组或列表
            return type(tensor)(distributed_concat(t, num_total_examples) for t in tensor)  # 递归地对每个元素进行拼接操作
        if isinstance(tensor, Mapping):  # 如果是字典类型
            return type(tensor)({k: distributed_concat(t, num_total_examples) for k, t in tensor.items()})  # 递归地对每个值进行拼接操作

        tensor = atleast_1d(tensor).contiguous()  # 将张量至少视为1维张量，并确保其连续性
        output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]  # 创建每个进程的输出张量副本
        dist.all_gather(output_tensors, tensor)  # 在所有进程间进行全局收集操作
        concat = torch.cat(output_tensors, dim=0)  # 沿指定维度拼接张量

        # 截断由 SequentialDistributedSampler 添加的虚拟元素
        if num_total_examples is not None:
            concat = concat[:num_total_examples]

        return concat  # 返回拼接后的结果张量
    except AssertionError:
        raise AssertionError("Not currently using distributed training")  # 抛出断言错误


# 在分布式环境下广播标量的张量化值
def distributed_broadcast_scalars(
    scalars: List[Union[int, float]],
    num_total_examples: Optional[int] = None,
    device: Optional[torch.device] = torch.device("cuda"),
) -> torch.Tensor:
    try:
        tensorized_scalar = torch.tensor(scalars).to(device)  # 将标量列表转换为张量，并移到指定设备
        output_tensors = [tensorized_scalar.clone() for _ in range(dist.get_world_size())]  # 创建每个进程的输出张量副本
        dist.all_gather(output_tensors, tensorized_scalar)  # 在所有进程间进行全局收集操作
        concat = torch.cat(output_tensors, dim=0)  # 沿指定维度拼接张量

        # 截断由 SequentialDistributedSampler 添加的虚拟元素
        if num_total_examples is not None:
            concat = concat[:num_total_examples]

        return concat  # 返回拼接后的结果张量
    except AssertionError:
        raise AssertionError("Not currently using distributed training")  # 抛出断言错误


# 重新发布未捕获的 PyTorch 警告
def reissue_pt_warnings(caught_warnings):
    if len(caught_warnings) > 1:  # 如果捕获到的警告数量大于1
        for w in caught_warnings:  # 对每个警告进行迭代
            if w.category != UserWarning or w.message != SAVE_STATE_WARNING:  # 如果不是用户警告或不是 SAVE_STATE_WARNING
                warnings.warn(w.message, w.category)  # 发出警告信息
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    分布式训练中的装饰器，使所有进程等待每个本地主节点执行某些操作。

    Args:
        local_rank (`int`): 本地进程的排名。
    """
    # 如果本地排名不是-1或0，则进行同步
    if local_rank not in [-1, 0]:
        dist.barrier()
    yield
    # 如果本地排名是0，则进行同步
    if local_rank == 0:
        dist.barrier()


class DistributedSamplerWithLoop(DistributedSampler):
    """
    类似于`torch.utils.data.distributed.DistributedSampler`，但在洗牌样本末尾循环以使每个进程具有批次大小的整数倍样本。

    Args:
        dataset (`torch.utils.data.Dataset`):
            用于采样的数据集。
        batch_size (`int`):
            此采样器使用的批次大小。
        kwargs (`Dict[str, Any]`, *可选*):
            传递给`DistributedSampler`的所有其他关键字参数。
    """

    def __init__(self, dataset, batch_size, **kwargs):
        super().__init__(dataset, **kwargs)
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        # 如果索引数量是批次大小的整数倍，则余数为0；否则余数为批次大小减去索引数量模批次大小的余数
        remainder = 0 if len(indices) % self.batch_size == 0 else self.batch_size - len(indices) % self.batch_size
        # DistributedSampler已经从开头添加了样本，使样本数是世界大小的整数倍，因此我们跳过这些样本。
        start_remainder = 1 if self.rank < len(self.dataset) % self.num_replicas else 0
        indices += indices[start_remainder : start_remainder + remainder]
        return iter(indices)


class SequentialDistributedSampler(Sampler):
    """
    顺序子采样器，按顺序子采样索引，使最终在收集所有结果时更容易。

    即使我们只在评估和预测中使用此采样器（没有训练），这意味着模型参数不必同步（即使前向传递次数不同，也不会挂起同步），
    我们仍然向采样器添加额外的样本，使其可以被`gather`或`reduce`，以便在循环结束时轻松处理。
    """
    # 初始化函数，用于设置分布式采样器的参数和状态
    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=None):
        # 发出警告信息，提示 SequentialDistributedSampler 将在 Transformers v5 版本中移除
        warnings.warn(
            "SequentialDistributedSampler is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        
        # 如果未指定 num_replicas，则检查分布式环境是否可用并获取全局大小
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        
        # 如果未指定 rank，则检查分布式环境是否可用并获取当前进程的排名
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        # 将输入的 dataset 存储在实例变量中
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        
        # 计算数据集的样本总数
        num_samples = len(self.dataset)
        
        # 如果指定了 batch_size，则将 num_samples 调整为 batch_size 的整数倍
        if batch_size is not None:
            self.num_samples = int(math.ceil(num_samples / (batch_size * num_replicas))) * batch_size
        else:
            self.num_samples = int(math.ceil(num_samples / num_replicas))
        
        # 计算总体的样本大小，考虑了 replica 数量
        self.total_size = self.num_samples * self.num_replicas
        self.batch_size = batch_size

    # 迭代器方法，返回当前进程应处理的数据样本索引
    def __iter__(self):
        # 创建初始索引列表，长度与数据集相同
        indices = list(range(len(self.dataset)))

        # 添加额外的样本以确保能够均匀分割
        indices += indices[: (self.total_size - len(indices))]
        # 断言索引列表长度与 total_size 是否匹配
        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # 子采样，根据当前进程的排名和 num_samples 获取对应的样本索引
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        # 断言子采样后的索引列表长度与 num_samples 是否匹配
        assert (
            len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        # 返回迭代器，包含当前进程应处理的数据样本索引
        return iter(indices)

    # 返回每个进程应处理的数据样本数
    def __len__(self):
        return self.num_samples
# 根据给定的数据集和批量大小，返回适合在TPU环境下使用的数据采样器
def get_tpu_sampler(dataset: torch.utils.data.Dataset, batch_size: int):
    # 如果只有一个进程或者没有TPU环境，返回一个随机采样器
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    # 使用分布式采样器，设置副本数为TPU进程数，当前进程的排名为序号
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


# 创建与给定数组结构相同的嵌套结构，但是第一维的大小固定为num_samples
def nested_new_like(arrays, num_samples, padding_index=-100):
    if isinstance(arrays, (list, tuple)):
        return type(arrays)(nested_new_like(x, num_samples) for x in arrays)
    # 使用padding_index填充数组的元素，使其形状变为(num_samples, arrays.shape[1], ...)
    return np.full_like(arrays, padding_index, shape=(num_samples, *arrays.shape[1:]))


# 将给定数组扩展到新的序列长度，使用padding_index进行填充
def expand_like(arrays, new_seq_length, padding_index=-100):
    # 创建与arrays相同结构的结果数组，第二维度扩展到new_seq_length，使用padding_index填充
    result = np.full_like(arrays, padding_index, shape=(arrays.shape[0], new_seq_length) + arrays.shape[2:])
    # 将原数组的数据复制到结果数组中对应位置
    result[:, : arrays.shape[1]] = arrays
    return result


# 对嵌套的张量列表/元组/字典进行截断处理，使其最大长度为limit
def nested_truncate(tensors, limit):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_truncate(t, limit) for k, t in tensors.items()})
    # 对于张量，直接截取到指定的limit长度
    return tensors[:limit]


class DistributedTensorGatherer:
    """
    一个负责在CPU上按块正确聚合张量（或嵌套的张量列表/元组/字典）的类。

    如果我们的数据集有16个样本，每个进程批量大小为2，有3个进程，并且我们在每一步都聚合并传输到CPU，
    我们的采样器将生成以下索引：

        `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]`

    为了使每个进程获得相同的数据集长度的多个数目，然后进程0、1和2将负责为以下样本做出预测：

        - P0: `[0, 1, 2, 3, 4, 5]`
        - P1: `[6, 7, 8, 9, 10, 11]`
        - P2: `[12, 13, 14, 15, 0, 1]`

    每个进程的第一个批次将是

        - P0: `[0, 1]`
        - P1: `[6, 7]`
        - P2: `[12, 13]`

    因此，如果我们在第一个批次结束时聚合，我们将获得一个对应于以下索引的张量（或嵌套的张量列表/元组）：

        `[0, 1, 6, 7, 12, 13]`

    如果我们直接连接我们的结果而不采取任何预防措施，用户最终会在预测循环结束时按以下顺序获得索引的预测结果：

        `[0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1]`

    由于某种原因，这种情况并不会符合他们的期望。这个类就是为了解决这个问题。
    """
    pass
    Args:
        world_size (`int`):
            分布式训练中使用的进程数。
        num_samples (`int`):
            数据集中的样本数。
        make_multiple_of (`int`, *可选*):
            如果传入，表示每个进程处理的数据集大小应该是此参数的倍数（通过增加样本数来实现）。
        padding_index (`int`, *可选*, 默认为 -100):
            如果数组的序列长度不相同时使用的填充索引。

    """
    初始化方法，初始化分布式张量收集器对象。
    """
    def __init__(self, world_size, num_samples, make_multiple_of=None, padding_index=-100):
        # 发出警告，提醒该类在 Transformers 的 v5 版本中将被移除
        warnings.warn(
            "DistributedTensorGatherer is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 设置对象属性：分布式训练使用的进程数
        self.world_size = world_size
        # 设置对象属性：数据集中的样本数
        self.num_samples = num_samples
        # 计算处理后的总样本数，考虑是否需要使总样本数为 make_multiple_of 的倍数
        total_size = world_size if make_multiple_of is None else world_size * make_multiple_of
        # 计算并设置对象属性：处理后的总样本数，确保是 make_multiple_of 的倍数
        self.total_samples = int(np.ceil(num_samples / total_size)) * total_size
        # 计算并设置对象属性：每个进程需要处理的样本数
        self.process_length = self.total_samples // world_size
        # 初始化对象私有属性
        self._storage = None
        self._offsets = None
        # 设置对象属性：填充索引，用于处理长度不同的数组时的填充
        self.padding_index = padding_index

    """
    添加数组到内部存储，如果是第一次添加数组，则初始化存储到完整大小，以便在开始时发生内存溢出。
    """
    def add_arrays(self, arrays):
        if arrays is None:
            return
        # 如果内部存储为空，则根据总样本数和填充索引初始化存储空间
        if self._storage is None:
            self._storage = nested_new_like(arrays, self.total_samples, padding_index=self.padding_index)
            # 根据每个进程需要处理的样本数初始化偏移量列表
            self._offsets = list(range(0, self.total_samples, self.process_length))

        # 将数组添加到内部存储，并返回添加的片段长度和更新后的存储空间
        slice_len, self._storage = self._nested_set_tensors(self._storage, arrays)
        
        # 更新每个进程的偏移量，以便下一次添加数据时从正确的位置开始
        for i in range(self.world_size):
            self._offsets[i] += slice_len
    # 递归设置张量数据到存储中
    def _nested_set_tensors(self, storage, arrays):
        # 如果数组是列表或元组
        if isinstance(arrays, (list, tuple)):
            # 递归调用_nested_set_tensors来处理每个元素，返回结果的第一个元素和类型一致的数组
            result = [self._nested_set_tensors(x, y) for x, y in zip(storage, arrays)]
            # 返回结果的第一个元素的第一个元素，以及类型与输入数组相同的生成器表达式生成的数组
            return result[0][0], type(arrays)(r[1] for r in result)
        
        # 断言：传入的数组的第一个维度应为self.world_size的整数倍
        assert (
            arrays.shape[0] % self.world_size == 0
        ), f"Arrays passed should all have a first dimension multiple of {self.world_size}, found {arrays.shape[0]}."
        
        # 计算每个分片的长度
        slice_len = arrays.shape[0] // self.world_size
        
        # 遍历每个进程
        for i in range(self.world_size):
            # 如果数组的维度为1
            if len(arrays.shape) == 1:
                # 将arrays的第i个分片复制到storage的相应位置
                storage[self._offsets[i] : self._offsets[i] + slice_len] = arrays[i * slice_len : (i + 1) * slice_len]
            else:
                # 如果storage的维度大于1且小于arrays的第二维度，则动态扩展storage
                if len(storage.shape) > 1 and storage.shape[1] < arrays.shape[1]:
                    storage = expand_like(storage, arrays.shape[1], padding_index=self.padding_index)
                # 将arrays的第i个分片复制到storage的相应位置，并限定复制的列数为arrays的第二维度
                storage[self._offsets[i] : self._offsets[i] + slice_len, : arrays.shape[1]] = arrays[
                    i * slice_len : (i + 1) * slice_len
                ]
        
        # 返回每个分片的长度和更新后的storage
        return slice_len, storage

    # 完成最终处理，返回正确收集的数组，并截断到样本数量（因为采样器添加了一些额外样本以保证每个进程得到相同长度的数据集）
    def finalize(self):
        """
        Return the properly gathered arrays and truncate to the number of samples (since the sampler added some extras
        to get each process a dataset of the same length).
        """
        # 如果_storage为None，则返回空
        if self._storage is None:
            return
        # 如果第一个偏移值不等于process_length，则记录警告
        if self._offsets[0] != self.process_length:
            logger.warning("Not all data has been set. Are you sure you passed all values?")
        # 调用nested_truncate函数对_storage进行截断，返回截断后的样本数量
        return nested_truncate(self._storage, self.num_samples)
@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False):
        # Extract logits from model output, handling both dict and list inputs
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        
        # If shift_labels is True, remove the last token from logits and labels
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        # Calculate negative log probabilities for each label using log_softmax
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        
        # Ensure labels have an additional dimension if necessary
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        # Create a mask for padding positions
        padding_mask = labels.eq(self.ignore_index)
        
        # Clamp labels to be non-negative (in case ignore_index is -100)
        labels = torch.clamp(labels, min=0)
        
        # Compute negative log likelihood loss for each element
        nll_loss = log_probs.gather(dim=-1, index=labels)
        
        # Calculate smoothed loss by summing log_probs over the last dimension
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        # Zero out losses corresponding to padding positions
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Calculate the number of active elements (non-padded)
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()

        # Compute mean nll_loss over all active elements
        nll_loss = nll_loss.sum() / num_active_elements
        
        # Compute mean smoothed_loss over all active elements and label dimensions
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        
        # Combine nll_loss and smoothed_loss with label smoothing factor epsilon
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=None, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # Generate random permutation of indices using torch.randperm
    indices = torch.randperm(len(lengths), generator=generator)
    
    # Calculate mega-batch size
    megabatch_size = mega_batch_mult * batch_size
    # 将索引列表按照 megabatch_size 分割成多个子列表，每个子列表转换为列表形式
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    # 对每个 megabatch 根据其包含的元素在 lengths 中的长度进行降序排序
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]

    # 获取每个 megabatch 中最长元素的长度，构成一个列表
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    # 找出最长元素长度列表中的最大值的索引
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # 将最长元素置换到第一个 megabatch 的第一个位置
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    # 展开多个 megabatch，以形成一个扁平的索引列表
    return [i for megabatch in megabatches for i in megabatch]
# 定义一个继承自Sampler的自定义采样器，用于按照数据集特征长度分组采样数据索引，并保留一定的随机性。
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        # 如果未提供 dataset 和 lengths 中的任何一个，则抛出错误
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        # 如果未提供 lengths，则尝试从 dataset 推断长度
        if lengths is None:
            # 如果数据集的第一个元素不是字典或BatchEncoding对象，或者 model_input_name 不在第一个元素的键中，则抛出错误
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            # 推断数据集每个元素的长度，并存储在 lengths 中
            lengths = [len(feature[model_input_name]) for feature in dataset]
        # 如果 lengths 是 torch.Tensor 类型，则转换为 List[int] 类型
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        # 存储采样器的长度信息
        self.lengths = lengths
        self.generator = generator

    # 返回采样器的长度，即数据集的样本数
    def __len__(self):
        return len(self.lengths)

    # 返回一个迭代器，生成按照长度分组并加入一定随机性的数据索引
    def __iter__(self):
        # 调用函数获取按长度分组的数据索引
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        return iter(indices)


# 定义一个继承自DistributedSampler的分布式采样器，用于按照数据集特征长度分组采样数据索引，并保留一定的随机性。
class DistributedLengthGroupedSampler(DistributedSampler):
    r"""
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """

    # 从PyTorch的DistributedSampler复制并调整的构造函数。
    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        # generator 参数用于控制随机数生成器
        generator=None,
    ):
        # 调用父类的构造函数，初始化分布式采样器的基本参数
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, seed=seed)
        # 如果未提供 lengths，则尝试从 dataset 推断长度
        if lengths is None:
            # 如果数据集的第一个元素不是字典或BatchEncoding对象，或者 model_input_name 不在第一个元素的键中，则抛出错误
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            # 推断数据集每个元素的长度，并存储在 lengths 中
            lengths = [len(feature[model_input_name]) for feature in dataset]
        # 如果 lengths 是 torch.Tensor 类型，则转换为 List[int] 类型
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, DistributedLengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        # 存储采样器的长度信息
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator
        ):
            # 检查 dataset 和 lengths 是否至少提供了一个，否则抛出 ValueError 异常
            raise ValueError("One of dataset and lengths must be provided.")
        if num_replicas is None:
            # 如果未指定 num_replicas，则根据是否可用判断是否需要分布式支持
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            # 获取分布式环境下的进程数作为 num_replicas
            num_replicas = dist.get_world_size()
        if rank is None:
            # 如果未指定 rank，则根据是否可用判断是否需要分布式支持
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            # 获取当前进程在分布式环境中的 rank
            rank = dist.get_rank()

        # 设置批量大小、进程数、当前进程的 rank、当前 epoch 数及是否丢弃最后一批数据的标志
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if lengths is None:
            # 如果未提供 lengths，则尝试自动推断
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                # 检查 dataset 的第一个元素是否为字典或 BatchEncoding，并确保包含指定的 model_input_name
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            # 推断每个样本的长度
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            # 如果 lengths 是 torch.Tensor，则警告使用 DistributedLengthGroupedSampler 会很慢，将 lengths 转换为 List[int]
            logger.info(
                "If lengths is a torch.Tensor, DistributedLengthGroupedSampler will be slow. Converting lengths to"
                " List[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths

        # 如果设置丢弃最后一批数据，并且数据集长度不能被进程数整除，则需要计算实际的样本数以确保每个进程获得相等数量的数据
        if self.drop_last and len(self.lengths) % self.num_replicas != 0:
            # 计算每个进程应分配的样本数，向上取整以确保每个进程分配相同数量的数据
            self.num_samples = math.ceil((len(self.lengths) - self.num_replicas) / self.num_replicas)
        else:
            # 否则，每个进程平均分配数据，向上取整
            self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)
        # 总共的样本数为每个进程的样本数乘以进程数
        self.total_size = self.num_samples * self.num_replicas
        # 设置随机种子
        self.seed = seed
    def __iter__(self) -> Iterator:
        # 根据当前的 epoch 和 seed 决定性地对索引进行洗牌
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # 使用自定义的生成器 g，根据长度信息和批次大小生成分组索引
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=g)

        if not self.drop_last:
            # 如果不丢弃最后一部分数据，添加额外的样本使得索引列表能够均匀分割
            indices += indices[: (self.total_size - len(indices))]
        else:
            # 如果丢弃最后一部分数据，截取索引列表使其长度符合要求
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # 对索引进行子采样，根据排名、总大小和副本数目筛选索引
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # 返回迭代器对象，用于遍历子采样后的索引
        return iter(indices)
class ShardSampler(Sampler):
    """
    Sampler that shards batches between several processes. Dispatches indices batch by batch: on 2 processes with batch
    size 4, the first two batches are `[0, 1, 2, 3, 4, 5, 6, 7]` and `[8, 9, 10, 11, 12, 13, 14, 15]`, which shard into
    `[0, 1, 2, 3]` and `[8, 9, 10, 11]` for GPU-0 and `[4, 5, 6, 7]` and `[12, 13, 14, 15]` for GPU-1.

    The sampler thus yields `[0, 1, 2, 3, 8, 9, 10, 11]` on GPU-0 and `[4, 5, 6, 7, 12, 13, 14, 15]` on GPU-1.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
    ):
        # Initialize the sampler with dataset, batch size, drop_last flag,
        # number of processes, and process index.
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index

        # Calculate the total batch size across all processes.
        self.total_batch_size = total_batch_size = batch_size * num_processes

        # Determine the total number of samples considering drop_last option.
        num_batches = len(dataset) // total_batch_size if drop_last else math.ceil(len(dataset) / total_batch_size)
        self.total_num_samples = num_batches * total_batch_size

    def __iter__(self):
        # Generate initial indices from the dataset.
        indices = list(range(len(self.dataset)))

        # Add extra samples to make the number of samples evenly divisible by total_num_samples.
        while len(indices) < self.total_num_samples:
            indices += indices[: (self.total_num_samples - len(indices))]

        # Generate indices for the current process based on batch size and process index.
        result = []
        for batch_start in range(self.batch_size * self.process_index, self.total_num_samples, self.total_batch_size):
            result += indices[batch_start : batch_start + self.batch_size]

        return iter(result)

    def __len__(self):
        # Each shard only sees a fraction of total_num_samples.
        return self.total_num_samples // self.num_processes


class IterableDatasetShard(IterableDataset):
    """
    Wraps a PyTorch `IterableDataset` to generate samples for one of the processes only. Instances of this class will
    always yield a number of samples that is a round multiple of the actual batch size (which is `batch_size x
    num_processes`). Depending on the value of the `drop_last` attribute, it will either stop the iteration at the
    first batch that would be too small or loop with indices from the beginning.

    On two processes with an iterable dataset yielding of `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` with a batch size of
    2:

    - the shard on process 0 will yield `[0, 1, 4, 5, 8, 9]` so will see batches `[0, 1]`, `[4, 5]`, `[8, 9]`
    - the shard on process 1 will yield `[2, 3, 6, 7, 10, 11]` so will see batches `[2, 3]`, `[6, 7]`, `[10, 11]`
    """
    # 如果 IterableDataset 实现了一些需要在所有进程上以相同方式应用的随机化（例如，洗牌），则应该在 dataset 的 generator 属性中使用 torch.Generator 生成你的随机数，并调用此对象的 set_epoch 方法。它将在开始迭代之前，将该 generator 的种子设置为 seed + epoch 的值在所有进程上。
    # 或者，你也可以在你的可迭代数据集中实现 set_epoch() 方法来处理这个问题。

    # 初始化方法，初始化 IterableDatasetShard 实例
    def __init__(
        self,
        dataset: IterableDataset,  # 要分割为多个分片的批次采样器
        batch_size: int = 1,  # 每个分片的批次大小，默认为1
        drop_last: bool = False,  # 是否舍弃最后不完整的批次，或者使用从开头取样的样本来补全最后的批次
        num_processes: int = 1,  # 并行运行的进程数，默认为1
        process_index: int = 0,  # 当前进程的索引，默认为0
        seed: int = 0,  # 在 set_epoch 方法中用于随机数生成的随机种子，默认为0
    ):
        self.dataset = dataset  # 分片的数据集
        self.batch_size = batch_size  # 批次大小
        self.drop_last = drop_last  # 是否舍弃最后不完整的批次
        self.num_processes = num_processes  # 并行进程数
        self.process_index = process_index  # 当前进程的索引
        self.seed = seed  # 随机种子
        self.epoch = 0  # 初始化时的迭代周期为0
        self.num_examples = 0  # 样本数量为0

    # 设置迭代周期的方法
    def set_epoch(self, epoch): 
        self.epoch = epoch  # 将当前迭代周期设置为传入的值
        if hasattr(self.dataset, "set_epoch"):  # 如果数据集有 set_epoch 方法
            self.dataset.set_epoch(epoch)  # 调用数据集的 set_epoch 方法，将数据集中的迭代周期设置为传入的值
    # 定义迭代器的方法，用于迭代数据集中的元素
    def __iter__(self):
        # 初始化示例数为零
        self.num_examples = 0
        # 如果数据集没有 set_epoch 方法，但有 generator 属性且其类型是 torch.Generator
        if (
            not hasattr(self.dataset, "set_epoch")
            and hasattr(self.dataset, "generator")
            and isinstance(self.dataset.generator, torch.Generator)
        ):
            # 设置数据集的随机种子为当前 epoch 加上初始种子值
            self.dataset.generator.manual_seed(self.seed + self.epoch)
        
        # 计算真实的批量大小，考虑到进程数量
        real_batch_size = self.batch_size * self.num_processes
        # 计算当前进程的数据片段范围
        process_slice = range(self.process_index * self.batch_size, (self.process_index + 1) * self.batch_size)

        # 初始化第一个批次为 None
        first_batch = None
        # 初始化当前批次列表
        current_batch = []
        
        # 遍历数据集中的每一个元素
        for element in self.dataset:
            # 增加示例数计数
            self.num_examples += 1
            # 将元素添加到当前批次中
            current_batch.append(element)
            
            # 当当前批次达到真实批量大小时，开始生成批次中的元素
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                # 如果第一个批次还未设置，则复制当前批次为第一个批次
                if first_batch is None:
                    first_batch = current_batch.copy()
                # 重置当前批次列表
                current_batch = []
        
        # 如果 drop_last 为 False，并且当前批次列表中还有剩余元素
        if not self.drop_last and len(current_batch) > 0:
            # 如果第一个批次还未设置，则复制当前批次为第一个批次
            if first_batch is None:
                first_batch = current_batch.copy()
            # 将当前批次补齐至真实批量大小
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            # 生成批次中的元素
            for i in process_slice:
                yield current_batch[i]

    # 定义返回数据集长度的方法
    def __len__(self):
        # 如果 drop_last 为 True，计算不包括不完整批次的数据集长度
        if self.drop_last:
            return (len(self.dataset) // (self.batch_size * self.num_processes)) * self.batch_size
        else:
            # 否则，计算包括不完整批次的数据集长度
            return math.ceil(len(self.dataset) / (self.batch_size * self.num_processes)) * self.batch_size
# 获取当前学习率的辅助方法
def _get_learning_rate(self):
    if self.is_deepspeed_enabled:
        # 如果使用了 DeepSpeed，并且启用了 fp16 和动态损失缩放，优化器/调度器在前几十步可能不会运行，
        # 因此在这个热身阶段调用 `get_last_lr` 可能会失败，因此需要进行以下处理：
        try:
            last_lr = self.lr_scheduler.get_last_lr()[0]
        except AssertionError as e:
            if "need to call step" in str(e):
                logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                last_lr = 0
            else:
                raise
    else:
        # 如果没有使用 DeepSpeed，根据不同的调度器类型获取最后的学习率
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            last_lr = self.optimizer.param_groups[0]["lr"]
        else:
            last_lr = self.lr_scheduler.get_last_lr()[0]
        # 如果最后的学习率是一个 Tensor，则将其转换为 Python 数字
        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()
    return last_lr


# 将秒数转换为 hh:mm:ss.msec 格式，毫秒部分保留两位小数
def _secs2timedelta(secs):
    msec = int(abs(secs - int(secs)) * 100)
    return f"{datetime.timedelta(seconds=int(secs))}.{msec:02d}"


# 将 Trainer 返回的指标格式化为人类可读的格式
def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Args:
        metrics (`Dict[str, float]`):
            训练/评估/预测返回的指标

    Returns:
        metrics (`Dict[str, float]`): 格式化后的指标
    """

    metrics_copy = metrics.copy()
    for k, v in metrics_copy.items():
        if "_mem_" in k:
            # 如果指标名称中包含 `_mem_`，将其转换为以 MB 为单位的字符串表示
            metrics_copy[k] = f"{ v >> 20 }MB"
        elif "_runtime" in k:
            # 如果指标名称中包含 `_runtime`，将其转换为 hh:mm:ss.msec 格式
            metrics_copy[k] = _secs2timedelta(v)
        elif k == "total_flos":
            # 如果是 `total_flos` 指标，将其转换为以 GF 为单位的字符串表示
            metrics_copy[k] = f"{ int(v) >> 30 }GF"
        elif isinstance(metrics_copy[k], float):
            # 对于其他浮点数指标，保留四位小数
            metrics_copy[k] = round(v, 4)

    return metrics_copy


# 以特定格式记录指标
def log_metrics(self, split, metrics):
    """
    Args:
        split (`str`):
            模式/分割名称：`train`, `eval`, `test` 中的一个
        metrics (`Dict[str, float]`):
            训练/评估/预测返回的指标字典

    Notes on memory reports:

    要获取内存使用报告，需要安装 `psutil`。您可以使用 `pip install psutil` 进行安装。

    当运行此方法时，您将看到一个报告，其中包含：

    ```
    init_mem_cpu_alloc_delta   =     1301MB
    init_mem_cpu_peaked_delta  =      154MB
    init_mem_gpu_alloc_delta   =      230MB
    init_mem_gpu_peaked_delta  =        0MB
    train_mem_cpu_alloc_delta  =     1345MB
    train_mem_cpu_peaked_delta =        0MB
    ```
    """
    # 训练过程中 GPU 内存分配的变化，增加了 693MB
    train_mem_gpu_alloc_delta  =      693MB
    # 训练过程中 GPU 内存达到的峰值，增加了 7MB
    train_mem_gpu_peaked_delta =        7MB
    """
    Print formatted metrics for a given split.

    This function prints out metrics in a formatted way for a specified split (e.g., 'train', 'validation').
    It calculates the maximum width required for keys and values to ensure aligned printing. Metrics are sorted
    by keys before printing.

    Args:
    - split (str): The name of the split (e.g., 'train', 'validation') for which metrics are being printed.
    - metrics (dict): A dictionary containing metrics to be printed.

    Notes:
    - This function is intended to be called to print metrics during training or evaluation.
    - Metrics are formatted to ensure proper alignment based on the widest key and value in the provided dictionary.
    """
    if not self.is_world_process_zero():
        # If the current process is not the primary process, do not print metrics
        return

    # Print section header indicating the type of metrics being printed
    print(f"***** {split} metrics *****")

    # Format metrics for printing
    metrics_formatted = self.metrics_format(metrics)

    # Determine the maximum width needed for keys and values for proper alignment
    k_width = max(len(str(x)) for x in metrics_formatted.keys())
    v_width = max(len(str(x)) for x in metrics_formatted.values())

    # Iterate through sorted metrics and print each key-value pair
    for key in sorted(metrics_formatted.keys()):
        # Print each metric key-value pair with proper alignment
        print(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")
#`
def save_metrics(self, split, metrics, combined=True):
    """
    Save metrics into a json file for that split, e.g. `train_results.json`.

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (`str`):
            Mode/split name: one of `train`, `eval`, `test`, `all`
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predict
        combined (`bool`, *optional*, defaults to `True`):
            Creates combined metrics by updating `all_results.json` with metrics of this call

    To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
    unformatted numbers are saved in the current method.

    """
    # 检查是否为分布式环境中的主进程
    if not self.is_world_process_zero():
        return

    # 构建输出文件路径，命名为 split_results.json
    path = os.path.join(self.args.output_dir, f"{split}_results.json")
    # 打开文件并写入 metrics 数据，格式化 JSON，缩进为 4，按键排序
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)

    # 如果需要合并结果
    if combined:
        # 构建合并结果文件路径，命名为 all_results.json
        path = os.path.join(self.args.output_dir, "all_results.json")
        # 如果合并结果文件已存在，读取其内容
        if os.path.exists(path):
            with open(path, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        # 更新合并结果
        all_metrics.update(metrics)
        # 写入更新后的合并结果到文件
        with open(path, "w") as f:
            json.dump(all_metrics, f, indent=4, sort_keys=True)


def save_state(self):
    """
    Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model

    Under distributed environment this is done only for a process with rank 0.
    """
    # 检查是否为分布式环境中的主进程
    if not self.is_world_process_zero():
        return

    # 构建状态保存文件路径，命名为 trainer_state.json
    path = os.path.join(self.args.output_dir, "trainer_state.json")
    # 调用模型状态对象保存方法，将状态保存到文件
    self.state.save_to_json(path)


def get_model_param_count(model, trainable_only=False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    # 如果启用了 DeepSpeed Zero-3，定义 numel 函数以支持 DeepSpeed 的参数数量计算
    if is_deepspeed_zero3_enabled():

        def numel(p):
            return p.ds_numel if hasattr(p, "ds_numel") else p.numel()

    else:
        # 否则，使用标准参数数量计算方法
        def numel(p):
            return p.numel()

    # 返回模型参数数量之和，依据是否只计算可训练参数
    return sum(numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    # 遍历模型的子模块，递归获取参数名，排除禁止层类型的参数
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # 添加模型中定义的参数（使用 nn.Parameter），因为这些参数不在任何子模块中
    result += list(model._parameters.keys())
    return result


def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    获取给定模块中特定类名的类对象。
    
    参数：
    module: 给定的模块对象
    name: 要查找的类名
    
    返回值：
    如果找到与给定名称匹配的类对象，则返回该类对象；如果找不到，返回 None。
    """
    modules_children = list(module.children())  # 获取模块的所有子模块列表
    if module.__class__.__name__ == name:  # 如果当前模块的类名与目标名称相同
        return module.__class__  # 返回当前模块的类对象
    elif len(modules_children) == 0:  # 如果当前模块没有子模块
        return  # 返回空，表示未找到目标类
    else:
        for child_module in modules_children:  # 遍历所有子模块
            module_class = get_module_class_from_name(child_module, name)  # 递归调用获取目标类对象
            if module_class is not None:  # 如果找到了目标类对象
                return module_class  # 返回目标类对象
# 如果当前进程为主进程，则执行删除指定目录下的文件
def remove_dummy_checkpoint(is_main_process, output_dir, filenames):
    if is_main_process:
        for filename in filenames:
            file = os.path.join(output_dir, filename)
            # 如果文件存在，则删除该文件
            if os.path.isfile(file):
                os.remove(file)

# 检查是否启用了SageMaker的模型并行功能
if is_sagemaker_mp_enabled():
    # 导入SageMaker模型并行的Torch扩展
    import smdistributed.modelparallel.torch as smp

    # 使用SMP的step装饰器定义前向传播和反向传播步骤
    @smp.step()
    def smp_forward_backward(model, inputs, gradient_accumulation_steps=1):
        # 执行模型前向传播
        outputs = model(**inputs)
        # 提取损失值，如果输出是字典则取loss键，否则取第一个元素
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 根据梯度累积步数计算平均损失
        loss /= gradient_accumulation_steps
        # 执行模型反向传播
        model.backward(loss)
        return loss

    # 使用SMP的step装饰器定义仅前向传播步骤
    @smp.step()
    def smp_forward_only(model, inputs):
        return model(**inputs)

    # SMP下的数据收集函数，递归地收集嵌套的列表、元组或字典中的张量
    def smp_gather(tensor):
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(smp_gather(t) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: smp_gather(v) for k, v in tensor.items()})
        # 如果不是张量类型则抛出类型错误
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )
        # 使用SMP的allgather函数在DP_GROUP中收集所有张量
        all_tensors = smp.allgather(tensor, smp.CommGroup.DP_GROUP)
        # 将每个张量至少转换为1维，并将它们连接起来
        all_tensors = [atleast_1d(t) for t in all_tensors]
        return torch.cat([t.cpu() for t in all_tensors], dim=0)

    # SMP下的嵌套张量连接函数，递归地连接嵌套的列表、元组或字典中的张量
    def smp_nested_concat(tensor):
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(smp_nested_concat(t) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: smp_nested_concat(v) for k, v in tensor.items()})
        # 如果不是StepOutput类型，则进行连接并将结果从计算图中分离并移到CPU上
        # 注：这里由于StepOutput与smp.step同名，Python可能会混淆
        return tensor.concat().detach().cpu()

# 加速器配置类，用于自定义加速器相关参数
@dataclass
class AcceleratorConfig:
    """
    A subset of arguments relating to the underlying [`accelerate.Accelerator`]
    implementation utilized in the `Trainer` that can be customized.
    Mostly relating to data.
    """
    # Parameters:
    # split_batches (`bool`, *optional*, defaults to `False`):
    #     Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
    #     `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
    #     round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
    #     in your script multiplied by the number of processes.
    # dispatch_batches (`bool`, *optional*):
    #     If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
    #     and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
    #     underlying dataset is an `IterableDataset`, `False` otherwise.
    # even_batches (`bool`, *optional*, defaults to `True`):
    #     If set to `True`, in cases where the total batch size across all processes does not exactly divide the
    #     dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
    #     all workers.
    # use_seedable_sampler (`bool`, *optional*, defaults to `True`):
    #     Whether or not use a fully seedable random sampler ([`accelerate.data_loader.SeedableRandomSampler`]). Ensures
    #     training results are fully reproducable using a different sampling technique. While seed-to-seed results
    #     may differ, on average the differences are neglible when using multiple different seeds to compare. Should
    #     also be ran with [`~utils.set_seed`] for the best results.
    even_batches: bool = field(
        default=True,
        metadata={
            "help": "If set to `True`, in cases where the total batch size across all processes does not exactly divide the"
            " dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among"
            " all workers."
        },
    )
    # 是否使用偶数批次
    # 如果设置为 `True`，在所有进程的总批次大小无法精确地整除数据集的情况下，
    # 将在数据集的开头复制样本，以便批次可以在所有工作器之间均匀分配

    use_seedable_sampler: bool = field(
        default=True,
        metadata={
            "help": "Whether or not use a fully seedable random sampler ([`accelerate.data_loader.SeedableRandomSampler`])."
            "Ensures training results are fully reproducable using a different sampling technique. "
            "While seed-to-seed results may differ, on average the differences are neglible when using"
            "multiple different seeds to compare. Should also be ran with [`~utils.set_seed`] for the best results."
        },
    )
    # 是否使用可种子化的随机采样器
    # 如果设置为 `True`，将使用可完全种子化的随机采样器（[`accelerate.data_loader.SeedableRandomSampler`]）。
    # 确保使用不同的采样技术可以完全复制训练结果。
    # 尽管种子之间的结果可能会有所不同，但在使用多个不同种子进行比较时，平均差异微乎其微。
    # 应与 [`~utils.set_seed`] 一起使用以获得最佳结果。

    @classmethod
    def from_json_file(cls, json_file):
        # 检查文件是否存在，选择合适的打开方式
        open_file = io.open if os.path.exists(json_file) else open
        with open_file(json_file, "r", encoding="utf-8") as f:
            # 加载 JSON 文件内容为字典
            config_dict = json.load(f)
        # 检查字典中是否有未知键，并加载合理的默认值
        extra_keys = sorted(key for key in config_dict.keys() if key not in cls.__dataclass_fields__.keys())
        if len(extra_keys) > 0:
            # 如果配置文件中存在未知键，抛出 ValueError 异常
            raise ValueError(
                f"The config file at {json_file} had unknown keys ({extra_keys}), please try upgrading your `transformers`"
                " version or fix (and potentially remove these keys) from your config file."
            )
        # 使用加载的配置字典创建当前类的实例并返回
        return cls(**config_dict)

    # 将对象转换为字典的方法
    def to_dict(self):
        # 使用深度复制来创建当前对象的属性字典并返回
        return copy.deepcopy(self.__dict__)
# 创建一个自定义的优化器类 LayerWiseDummyOptimizer，继承自 torch.optim.Optimizer
class LayerWiseDummyOptimizer(torch.optim.Optimizer):
    """
    对于像 GaLoRE 优化器这样的分层优化器，优化步骤已经通过后梯度钩子完成。
    因此，关键在于创建一个虚拟的优化器，在训练过程中返回空操作。

    初始想法来自 LLaMA-Factory 中的 @hiyouga：
    https://github.com/hiyouga/LLaMA-Factory/commit/8664262cde3919e10eaecbd66e8c5d356856362e#diff-ebe08ab14496dfb9e06075f0fdd36799ef6d1535cc4dd4715b74c4e3e06fe3ba
    """

    # 初始化函数，接受 optimizer_dict 和任意的 args 和 kwargs
    def __init__(self, optimizer_dict=None, *args, **kwargs):
        # 创建一个虚拟张量
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        # 调用父类的初始化函数，传入一个包含虚拟张量的列表和学习率 lr 的字典
        super().__init__([dummy_tensor], {"lr": 1e-03})

    # 定义了 zero_grad 方法，设置为无操作
    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    # 定义了 step 方法，设置为无操作，并返回空值
    def step(self, closure=None) -> Optional[float]:
        pass


# 创建一个自定义的调度器类 LayerWiseDummyScheduler，继承自 LRScheduler
class LayerWiseDummyScheduler(LRScheduler):
    """
    对于像 GaLoRE 优化器这样的分层优化器，优化和调度步骤已经通过后梯度钩子完成。
    因此，关键在于创建一个虚拟的调度器，在训练过程中返回空操作。
    """

    # 初始化函数，接受任意的 args 和 kwargs
    def __init__(self, *args, **kwargs):
        # 创建一个 LayerWiseDummyOptimizer 的实例作为优化器
        optimizer = LayerWiseDummyOptimizer()
        last_epoch = -1
        verbose = False
        # 调用父类 LRScheduler 的初始化函数，传入虚拟优化器、上一个 epoch、是否详细输出
        super().__init__(optimizer, last_epoch, verbose)

    # 定义了 get_lr 方法，返回当前优化器各参数组的学习率列表
    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    # 定义了 _get_closed_form_lr 方法，返回基础学习率的列表
    def _get_closed_form_lr(self):
        return self.base_lrs
```