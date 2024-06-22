# `.\transformers\trainer_pt_utils.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""
# 导入模块
import datetime
import json
import math
import os
import sys
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from logging import StreamHandler
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler

# 导入深度学习框架相关模块
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .tokenization_utils_base import BatchEncoding
from .utils import is_sagemaker_mp_enabled, is_torch_tpu_available, is_training_run_on_sagemaker, logging

# 如果在 SageMaker 上运行训练，则将日志输出到标准输出
if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))

# 如果 Torch TPU 可用，则导入相关模块
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

# 用于抑制 PyTorch 版本 1.4.2-1.7.0 发出的不希望的警告
try:
    from torch.optim.lr_scheduler import SAVE_STATE_WARNING
except ImportError:
    SAVE_STATE_WARNING = ""

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 获取 DataLoader 的 Sampler
def get_dataloader_sampler(dataloader):
    if hasattr(dataloader, "batch_sampler") and dataloader.batch_sampler is not None:
        return get_dataloader_sampler(dataloader.batch_sampler)
    elif hasattr(dataloader, "sampler"):
        return dataloader.sampler

# 将输入张量或数组至少转换为 1 维
def atleast_1d(tensor_or_array: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array

# 在第一轴上连接 tensor1 和 tensor2，如果需要则在第二轴上应用填充
def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # 计算新形状
    # 计算新张量的形状，行数为两个张量行数之和，列数为两个张量列数的最大值，其它维度与第一个张量相同
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # 填充结果张量
    result = tensor1.new_full(new_shape, padding_index)  # 创建一个新的张量，使用指定形状和填充值
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1  # 将第一个张量的数据复制到结果张量的对应位置
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2  # 将第二个张量的数据复制到结果张量的对应位置
    return result  # 返回填充后的结果张量
# 将两个数组在第一个轴上连接起来，如果需要的话在第二个轴上进行填充
def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    # 将array1至少转换为1维数组
    array1 = atleast_1d(array1)
    # 将array2至少转换为1维数组

    array2 = atleast_1d(array2)

    # 如果array1是1维数组或者array1和array2的第二维大小相同，则直接连接两个数组并返回
    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # 计算新的形状
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # 填充结果张量
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    # 检查`tensors`和`new_tensors`是否具有相同的类型
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    # 如果`tensors`是列表或元组
    if isinstance(tensors, (list, tuple)):
        # 递归调用nested_concat函数，对每个对应的元素进行连接和填充
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    # 如果`tensors`是torch张量
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    # 如果`tensors`是映射
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {k: nested_concat(t, new_tensors[k], padding_index=padding_index) for k, t in tensors.items()}
        )
    # 如果`tensors`是numpy数组
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def find_batch_size(tensors):
    """
    Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
    """
    # 如果`tensors`是列表或元组
    if isinstance(tensors, (list, tuple)):
        # 遍历每个元素，查找第一个维度的大��
        for t in tensors:
            result = find_batch_size(t)
            if result is not None:
                return result
    # 如果`tensors`是映射
    elif isinstance(tensors, Mapping):
        # 遍历每个键值对，查找第一个维度的大小
        for key, value in tensors.items():
            result = find_batch_size(value)
            if result is not None:
                return result
    # 如果`tensors`是torch张量
    elif isinstance(tensors, torch.Tensor):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None
    # 如果`tensors`是numpy数组
    elif isinstance(tensors, np.ndarray):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    # 如果`tensors`是列表或元组
    if isinstance(tensors, (list, tuple)):
        # 递归调用nested_numpify函数，对每个元素进行numpify
        return type(tensors)(nested_numpify(t) for t in tensors)
    # 如果`tensors`是映射
    if isinstance(tensors, Mapping):
        # 递归调用nested_numpify函数，对每个值进行numpify
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    # 将张量移动到CPU上
    t = tensors.cpu()
    # 如果张量的数据类型是 torch.bfloat16
    if t.dtype == torch.bfloat16:
        # 在 Numpy 1.21.4 版本中，NumPy 不支持 bfloat16 数据类型（参见链接）
        # 在 NumPy 添加对 bfloat16 的支持之前，我们必须将数据类型转换为 float32
        t = t.to(torch.float32)
    # 返回张量的 NumPy 数组表示
    return t.numpy()
# 从嵌套的张量中分离出张量（即使它是张量的嵌套列表/元组/字典）
def nested_detach(tensors):
    if isinstance(tensors, (list, tuple)):
        # 递归地对列表/元组中的每个张量进行分离操作
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        # 递归地对字典中的每个张量进行分离操作
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    # 对单个张量进行分离操作
    return tensors.detach()


def nested_xla_mesh_reduce(tensors, name):
    # 如果 Torch XLA 可用
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm

        if isinstance(tensors, (list, tuple)):
            # 递归地对列表/元组中的每个张量进行 XLA 网格归约
            return type(tensors)(nested_xla_mesh_reduce(t, f"{name}_{i}") for i, t in enumerate(tensors))
        if isinstance(tensors, Mapping):
            # 递归地对字典中的每个张量进行 XLA 网格归约
            return type(tensors)(
                {k: nested_xla_mesh_reduce(t, f"{name}_{i}") for i, (k, t) in enumerate(tensors.items())}
            )

        # 将张量转换为至少是一维的张量，并执行网格归约
        tensors = atleast_1d(tensors)
        return xm.mesh_reduce(name, tensors, torch.cat)
    else:
        # 如果 Torch XLA 不可用，则抛出 ImportError
        raise ImportError("Torch xla must be installed to use `nested_xla_mesh_reduce`")


def distributed_concat(tensor: Any, num_total_examples: Optional[int] = None) -> Any:
    try:
        if isinstance(tensor, (tuple, list)):
            # 递归地对元组/列表中的每个张量进行分布式连接
            return type(tensor)(distributed_concat(t, num_total_examples) for t in tensor)
        if isinstance(tensor, Mapping):
            # 递归地对字典中的每个张量进行分布式连接
            return type(tensor)({k: distributed_concat(t, num_total_examples) for k, t in tensor.items()})
        # 将张量转换为至少是一维的连续张量
        tensor = atleast_1d(tensor).contiguous()
        output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
        # 在所有进程之间进行全局收集
        dist.all_gather(output_tensors, tensor)
        # 在指定维度上连接张量
        concat = torch.cat(output_tensors, dim=0)

        # 截断由 SequentialDistributedSampler 添加的虚拟元素
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        # 如果不是在使用分布式训练，则抛出 AssertionError
        raise AssertionError("Not currently using distributed training")


def distributed_broadcast_scalars(
    scalars: List[Union[int, float]],
    num_total_examples: Optional[int] = None,
    device: Optional[torch.device] = torch.device("cuda"),
) -> torch.Tensor:
    try:
        # 将标量列表转换为张量并放置到指定设备上
        tensorized_scalar = torch.tensor(scalars).to(device)
        output_tensors = [tensorized_scalar.clone() for _ in range(dist.get_world_size())]
        # 在所有进程之间进行全局广播
        dist.all_gather(output_tensors, tensorized_scalar)
        # 在指定维度上连接张量
        concat = torch.cat(output_tensors, dim=0)

        # 截断由 SequentialDistributedSampler 添加的虚拟元素
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        # 如果不是在使用分布式训练，则抛出 AssertionError
        raise AssertionError("Not currently using distributed training")


def reissue_pt_warnings(caught_warnings):
    # 重新发出不是 SAVE_STATE_WARNING 的警告
    if len(caught_warnings) > 1:
        for w in caught_warnings:
            if w.category != UserWarning or w.message != SAVE_STATE_WARNING:
                warnings.warn(w.message, w.category)
```  
@contextmanager
# 创建一个上下文管理器，用于在分布式训练中使所有进程等待每个本地主节点执行某项操作
def torch_distributed_zero_first(local_rank: int):
    if local_rank not in [-1, 0]:
        # 如果本地进程不是主节点，执行分布式同步，确保所有进程都等待
        dist.barrier()
    yield
    # 如果本地进程是主节点，执行分布式同步，确保所有进程都等待
    if local_rank == 0:
        dist.barrier()


class DistributedSamplerWithLoop(DistributedSampler):
    """
    Like a torch.utils.data.distributed.DistributedSampler` but loops at the end back to the beginning of the shuffled
    samples to make each process have a round multiple of batch_size samples.

    Args:
        dataset (`torch.utils.data.Dataset`):
            Dataset used for sampling.
        batch_size (`int`):
            The batch size used with this sampler
        kwargs (`Dict[str, Any]`, *optional*):
            All other keyword arguments passed to `DistributedSampler`.
    """

    def __init__(self, dataset, batch_size, **kwargs):
        super().__init__(dataset, **kwargs)
        self.batch_size = batch_size

    def __iter__(self):
        # 获取样本的索引列表
        indices = list(super().__iter__())
        # 计算补足的样本数量，以使样本总数成为 batch_size 的整数倍
        remainder = 0 if len(indices) % self.batch_size == 0 else self.batch_size - len(indices) % self.batch_size
        # DistributedSampler 已经添加了从开头开始的样本，使样本数量成为 world size 的整数倍，因此我们跳过这些样本
        start_remainder = 1 if self.rank < len(self.dataset) % self.num_replicas else 0
        # 添加额外的样本以使样本数量成为 batch_size 的整数倍
        indices += indices[start_remainder : start_remainder + remainder]
        return iter(indices)


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """
    # 初始化函数，用于创建SequentialDistributedSampler对象
    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=None):
        # 发出警告，提示SequentialDistributedSampler将在Transformers的v5版本中被移除
        warnings.warn(
            "SequentialDistributedSampler is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 如果未指定副本数，则根据环境判断并获取副本数
        if num_replicas is None:
            # 若分布式包不可用，则抛出运行时错误
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        # 如果未指定rank，则根据环境判断并获取rank
        if rank is None:
            # 若分布式包不可用，则抛出运行时错误
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        # 设置数据集
        self.dataset = dataset
        # 设置副本数
        self.num_replicas = num_replicas
        # 设置rank
        self.rank = rank
        # 获取数据集长度
        num_samples = len(self.dataset)
        # 如果指定了批量大小，则将num_samples增加到批量大小的倍数
        if batch_size is not None:
            self.num_samples = int(math.ceil(num_samples / (batch_size * num_replicas))) * batch_size
        else:
            self.num_samples = int(math.ceil(num_samples / num_replicas))
        # 计算总样本数
        self.total_size = self.num_samples * self.num_replicas
        # 设置批量大小
        self.batch_size = batch_size

    # 迭代函数，返回一个迭代器
    def __iter__(self):
        # 获取数据集的索引列表
        indices = list(range(len(self.dataset)))

        # 添加额外的样本以使其能够被整除
        indices += indices[: (self.total_size - len(indices))]
        # 断言索引列表的长度与总样本数相等
        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # 对索引进行子采样
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        # 断言子采样后索引列表的长度与样本数相等
        assert (
            len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        # 返回索引列表的迭代器
        return iter(indices)

    # 返回样本数
    def __len__(self):
        return self.num_samples
# 根据数据集和批量大小创建一个用于分布式训练的采样器
def get_tpu_sampler(dataset: torch.utils.data.Dataset, batch_size: int):
    # 如果只有一个处理单元，返回一个随机采样器
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    # 否则返回一个分布式采样器，设置副本数量和当前处理单元的等级
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


# 创建一个具有与给定数组相同嵌套结构的数组，但第一维总是 num_samples
def nested_new_like(arrays, num_samples, padding_index=-100):
    """Create the same nested structure as `arrays` with a first dimension always at `num_samples`."""
    # 如果数组是列表或元组，递归创建相同嵌套结构的数组
    if isinstance(arrays, (list, tuple)):
        return type(arrays)(nested_new_like(x, num_samples) for x in arrays)
    # 返回一个形状与给定数组相同，填充值为 padding_index 的数组
    return np.full_like(arrays, padding_index, shape=(num_samples, *arrays.shape[1:]))


# 将给定数组扩展到新的序列长度，使用 padding_index 进行填充
def expand_like(arrays, new_seq_length, padding_index=-100):
    """Expand the `arrays` so that the second dimension grows to `new_seq_length`. Uses `padding_index` for padding."""
    # 创建一个形状为 (arrays.shape[0], new_seq_length) 的填充数组
    result = np.full_like(arrays, padding_index, shape=(arrays.shape[0], new_seq_length) + arrays.shape[2:])
    # 将原始数组的内容复制到新数组的前 arrays.shape[1] 列
    result[:, : arrays.shape[1]] = arrays
    # 返回结果数组
    return result


# 截断 tensors 到指定 limit（即使它是张量的嵌套列表/元组/字典）
def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple/dict of tensors)."
    # 如果 tensors 是列表或元组，递归截断每个元素
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    # 如果 tensors 是字典，递归截断每个值
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_truncate(t, limit) for k, t in tensors.items()})
    # 截断张量到指定 limit
    return tensors[:limit]


# 一个负责在 CPU 上按块正确聚集张量（或张量的嵌套列表/元组）的类
class DistributedTensorGatherer:
    """
    A class responsible for properly gathering tensors (or nested list/tuple of tensors) on the CPU by chunks.

    If our dataset has 16 samples with a batch size of 2 on 3 processes and we gather then transfer on CPU at every
    step, our sampler will generate the following indices:

        `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]`

    to get something of size a multiple of 3 (so that each process gets the same dataset length). Then process 0, 1 and
    2 will be responsible of making predictions for the following samples:

        - P0: `[0, 1, 2, 3, 4, 5]`
        - P1: `[6, 7, 8, 9, 10, 11]`
        - P2: `[12, 13, 14, 15, 0, 1]`

    The first batch treated on each process will be

        - P0: `[0, 1]`
        - P1: `[6, 7]`
        - P2: `[12, 13]`

    So if we gather at the end of the first batch, we will get a tensor (nested list/tuple of tensor) corresponding to
    the following indices:

        `[0, 1, 6, 7, 12, 13]`

    If we directly concatenate our results without taking any precautions, the user will then get the predictions for
    the indices in this order at the end of the prediction loop:

        `[0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1]`

    For some reason, that's not going to roll their boat. This class is there to solve that problem.
"""
    Args:
        world_size (`int`):
            分布式训练中使用的进程数。
        num_samples (`int`):
            数据集中的样本数。
        make_multiple_of (`int`, *optional*):
            如果传递了此参数，类会假设传递给每个进程的数据集是该参数的倍数（通过添加样本）。
        padding_index (`int`, *optional*, 默认为 -100):
            如果数组的所有序列长度不相等，则使用的填充索引。

    def __init__(self, world_size, num_samples, make_multiple_of=None, padding_index=-100):
        警告用户该类已弃用，将在 Transformers 的 v5 版本中移除。
        warnings.warn(
            "DistributedTensorGatherer is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        初始化类的实例。
        self.world_size = world_size
        设定类实例中的 world_size 属性为参数传入的 world_size。
        self.num_samples = num_samples
        设定类实例中的 num_samples 属性为参数传入的 num_samples。
        total_size = world_size if make_multiple_of is None else world_size * make_multiple_of
        计算 total_size，如果 make_multiple_of 参数为 None，则设定为 world_size，否则设定为 world_size 乘以 make_multiple_of。
        self.total_samples = int(np.ceil(num_samples / total_size)) * total_size
        计算 total_samples，即数据集的样本数，保证是 total_size 的倍数。
        self.process_length = self.total_samples // world_size
        计算 process_length，每个进程处理的样本数。
        self._storage = None
        初始化存储数组的属性。
        self._offsets = None
        初始化偏移量数组的属性。
        self.padding_index = padding_index
        设定类实例中的 padding_index 属性为参数传入的 padding_index。

    def add_arrays(self, arrays):
        """
        将 `arrays` 添加到内部存储中，将在首次传递数组时初始化存储到完整大小，以便如果发生内存溢出，它会在开始时发生。
        """
        如果 arrays 为 None，则直接返回。
        if arrays is None:
            return
        如果存储数组为空：
            初始化存储数组为与 arrays 结构相同的新数组，长度为 total_samples，使用 padding_index 进行填充。
            初始化偏移量数组为从 0 到 total_samples，步长为 process_length 的列表。
        if self._storage is None:
            self._storage = nested_new_like(arrays, self.total_samples, padding_index=self.padding_index)
            self._offsets = list(range(0, self.total_samples, self.process_length))

        获取切片长度，并更新存储数组。
        slice_len, self._storage = self._nested_set_tensors(self._storage, arrays)
        遍历每个进程：
            更新偏移量数组中的偏移量。
            self._offsets[i] += slice_len
    # 递归函数，用于将存储和数组进行嵌套设置
    def _nested_set_tensors(self, storage, arrays):
        # 如果数组是列表或元组
        if isinstance(arrays, (list, tuple)):
            # 递归调用_nested_set_tensors函数，对storage和arrays中的每个元素进行处理
            result = [self._nested_set_tensors(x, y) for x, y in zip(storage, arrays)]
            # 返回处理后的结果
            return result[0][0], type(arrays)(r[1] for r in result)
        # 断言，确保数组的第一个维度是self.world_size的倍数
        assert (
            arrays.shape[0] % self.world_size == 0
        ), f"Arrays passed should all have a first dimension multiple of {self.world_size}, found {arrays.shape[0]}."

        # 计算每个切片的长度
        slice_len = arrays.shape[0] // self.world_size
        # 遍历每个进程
        for i in range(self.world_size):
            # 如果数组的维度为1
            if len(arrays.shape) == 1:
                # 将数组的切片赋值给storage的对应位置
                storage[self._offsets[i] : self._offsets[i] + slice_len] = arrays[i * slice_len : (i + 1) * slice_len]
            else:
                # 如果storage的维度大于1且小于arrays的维度，根据arrays的维度扩展storage
                if len(storage.shape) > 1 and storage.shape[1] < arrays.shape[1]:
                    storage = expand_like(storage, arrays.shape[1], padding_index=self.padding_index)
                # 将数组的切片赋值给storage的对应位置
                storage[self._offsets[i] : self._offsets[i] + slice_len, : arrays.shape[1]] = arrays[
                    i * slice_len : (i + 1) * slice_len
                ]
        # 返回切片长度和处理后的storage
        return slice_len, storage

    # 完成数据收集
    def finalize(self):
        """
        Return the properly gathered arrays and truncate to the number of samples (since the sampler added some extras
        to get each process a dataset of the same length).
        """
        # 如果_storage为None，则返回空
        if self._storage is None:
            return
        # 如果第一个偏移量不等于process_length，则发出警告
        if self._offsets[0] != self.process_length:
            logger.warning("Not all data has been set. Are you sure you passed all values?")
        # 返回经过截断后的_storage和num_samples
        return nested_truncate(self._storage, self.num_samples)
from dataclasses import dataclass
import torch
import torch.nn as nn

# 创建一个数据类，用于添加标签平滑到 Transformers 模型的预计算输出上
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

    # 定义类的调用方法，对模型输出进行标签平滑处理
    def __call__(self, model_output, labels, shift_labels=False):
        # 如果模型输出是字典，则获取 logits；否则获取第一个元素作为 logits
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        # 如果需要移动标签，则对 logits 和 labels 进行切片处理
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        # 计算 log softmax
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        # 如果标签的维度比 log_probs 少一维，则在最后增加一个维度
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        # 创建 padding_mask，用于忽略特定索引处的标签
        padding_mask = labels.eq(self.ignore_index)
        # 替换标签中小于 0 的值为 0
        labels = torch.clamp(labels, min=0)
        # 使用 gather 获取负对数似然损失
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # 对 log_probs 求和，得到平滑损失
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        # 将 padding_mask 应用于损失
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # 计算有效元素的数量
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        # 计算负对数似然损失的均值
        nll_loss = nll_loss.sum() / num_active_elements
        # 计算平滑损失的均值
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        # 返回加权的损失值
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


# 返回一个索引列表，以便每个 `batch_size` 连续索引片段对应于相似长度的元素
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
    # 默认 mega_batch_mult 为 50 或者使得有 4 个超级批次的数量中的最小值
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # 针对小数据集的情况
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # 使用 torch 生成随机排列的索引
    indices = torch.randperm(len(lengths), generator=generator)
    # 计算超级批次的大小
    megabatch_size = mega_batch_mult * batch_size
    # 将数据索引按照指定大小划分成多个大批次，每个大批次是一个列表，包含多个索引
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    # 对每个大批次中的索引列表按照其对应序列长度排序，长度越长越靠前
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]

    # 以下代码段用于确保将最长的序列放在第一个位置
    # 由于每个大批次都是按照序列长度降序排列的，因此每个大批次中的第一个元素即为最长序列
    # 找出所有大批次中最长序列的长度，并找到其中最大值的索引
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # 将最长序列所在的大批次的第一个元素与第一个大批次的第一个元素互换位置，确保最长序列在第一个大批次中
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    # 将所有大批次中的索引列表展开为一个列表，并返回
    return [i for megabatch in megabatches for i in megabatch]
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
        # 检查输入参数，确保 dataset 和 lengths 至少提供一个
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        # 如果 lengths 未提供，则尝试根据 dataset 推断长度
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            # 检查 dataset 是否是字典或者 BatchEncoding 类型，并且是否包含指定的 model_input_name
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            # 计算每个样本的长度
            lengths = [len(feature[model_input_name]) for feature in dataset]
        # 如果 lengths 是 torch.Tensor 类型，则转换为 List[int]，因为 torch.Tensor 在处理上会较慢
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        # 获取按长度分组的样本索引
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        return iter(indices)


class DistributedLengthGroupedSampler(DistributedSampler):
    r"""
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """

    # Copied and adapted from PyTorch DistributedSampler.
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
        # 如果 dataset 和 lengths 都没有提供，则抛出 ValueError 异常
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        # 如果 num_replicas 没有提供，则根据是否支持分布式计算来确定
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            # 获取分布式环境下的进程数量
            num_replicas = dist.get_world_size()
        # 如果 rank 没有提供，则根据是否支持分布式计算来确定
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            # 获取当前进程在分布式环境下的排名
            rank = dist.get_rank()

        # 设置批量大小、进程数量、当前进程的排名、当前 epoch 数和是否丢弃最后一批数据的标志
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # 如果没有提供 lengths，则尝试自动推断
        if lengths is None:
            # 如果 model_input_name 未提供，则默认使用 "input_ids"
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            # 检查 dataset 中是否有 model_input_name 对应的特征
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            # 获取每个样本的长度
            lengths = [len(feature[model_input_name]) for feature in dataset]
        # 如果 lengths 是 torch.Tensor，则转换为 List[int] 类型
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, DistributedLengthGroupedSampler will be slow. Converting lengths to"
                " List[int]..."
            )
            lengths = lengths.tolist()

        # 存储 lengths
        self.lengths = lengths

        # 如果设置了丢弃最后一批数据，并且数据集长度不能被进程数量整除，则计算实际样本数，确保每个进程分配到的样本数相近
        if self.drop_last and len(self.lengths) % self.num_replicas != 0:
            # 分割到最近的可被整除的长度，确保使用该 Sampler 时每个进程接收到相同数量的数据
            self.num_samples = math.ceil((len(self.lengths) - self.num_replicas) / self.num_replicas)
        else:
            # 计算每个进程需要处理的样本数
            self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)
        # 计算总样本数
        self.total_size = self.num_samples * self.num_replicas
        # 设置随机种子
        self.seed = seed
    # 返回一个迭代器对象，用于按批次生成数据索引
    def __iter__(self) -> Iterator:
        # 根据 epoch 和 seed 生成确定性随机数生成器
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # 使用指定的长度列表、批大小和生成器确定性地洗牌索引
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=g)

        # 如果不舍弃最后一个不足一批的数据
        if not self.drop_last:
            # 添加额外的样本以使其能够被批次大小整除
            indices += indices[: (self.total_size - len(indices))]
        else:
            # 移除末尾的数据以使其能够被批次大小整除
            indices = indices[: self.total_size]
        # 断言索引列表的长度与总大小相等
        assert len(indices) == self.total_size

        # 对索引进行子采样，根据总大小、当前进程排名和复制数
        indices = indices[self.rank : self.total_size : self.num_replicas]
        # 断言子采样后索引列表的长度与样本数相等
        assert len(indices) == self.num_samples

        # 返回索引列表的迭代器
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
        # 初始化ShardSampler类的实例
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index

        # 计算总的批次大小
        self.total_batch_size = total_batch_size = batch_size * num_processes

        # 计算总的样本数
        num_batches = len(dataset) // total_batch_size if drop_last else math.ceil(len(dataset) / total_batch_size)
        self.total_num_samples = num_batches * total_batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # 添加额外的样本使其能够被整除。在极端情况下，我们有一个很小的数据集，需要多次执行。
        while len(indices) < self.total_num_samples:
            indices += indices[: (self.total_num_samples - len(indices)]

        result = []
        for batch_start in range(self.batch_size * self.process_index, self.total_num_samples, self.total_batch_size):
            result += indices[batch_start : batch_start + self.batch_size]

        return iter(result)

    def __len__(self):
        # 每个分片只看到总样本数的一部分
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
    <Tip warning={true}>

        If your IterableDataset implements some randomization that needs to be applied the same way on all processes
        (for instance, a shuffling), you should use a `torch.Generator` in a `generator` attribute of the `dataset` to
        generate your random numbers and call the [`~trainer_pt_utils.IterableDatasetShard.set_epoch`] method of this
        object. It will set the seed of this `generator` to `seed + epoch` on all processes before starting the
        iteration. Alternatively, you can also implement a `set_epoch()` method in your iterable dataset to deal with
        this.

    </Tip>

    Args:
        dataset (`torch.utils.data.IterableDataset`):
            The batch sampler to split in several shards. # 输入参数dataset: 可迭代数据集，用于分割成多个分片。
        batch_size (`int`, *optional*, defaults to 1):
            The size of the batches per shard. # 输入参数batch_size: 每个分片的批次大小，默认为1。
        drop_last (`bool`, *optional*, defaults to `False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning. # 输入参数drop_last: 是否丢弃最后不完整的批次，或者使用从头开始的样本来完成最后的批次，默认为False。
        num_processes (`int`, *optional*, defaults to 1):
            The number of processes running concurrently. # 输入参数num_processes: 并行运行的进程数量，默认为1。
        process_index (`int`, *optional*, defaults to 0):
            The index of the current process. # 输入参数process_index: 当前进程的索引，默认为0。
        seed (`int`, *optional*, defaults to 0):
            A random seed that will be used for the random number generation in
            [`~trainer_pt_utils.IterableDatasetShard.set_epoch`]. # 输入参数seed: 用于随机数生成的随机种子，在[`~trainer_pt_utils.IterableDatasetShard.set_epoch`]中使用。
    """

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        seed: int = 0,
    ):
        self.dataset = dataset  # 初始化数据集
        self.batch_size = batch_size  # 初始化批次大小
        self.drop_last = drop_last  # 初始化是否丢弃最后不完整的批次
        self.num_processes = num_processes  # 初始化并行进程数量
        self.process_index = process_index  # 初始化当前进程索引
        self.seed = seed  # 初始化随机种子
        self.epoch = 0  # 初始化时期为0
        self.num_examples = 0  # 初始化样本数量为0

    def set_epoch(self, epoch):
        self.epoch = epoch  # 设置时期为给定时期
        if hasattr(self.dataset, "set_epoch"):  # 如果数据集有set_epoch方法
            self.dataset.set_epoch(epoch)  # 调用数据集的set_epoch方法设置时期
    # 定义迭代器方法，用于在数据集上进行迭代
    def __iter__(self):
        # 初始化样本计数器
        self.num_examples = 0
        # 如果数据集没有 "set_epoch" 属性，
        # 且具有 "generator" 属性且该属性是 torch.Generator 实例
        if (
            not hasattr(self.dataset, "set_epoch")
            and hasattr(self.dataset, "generator")
            and isinstance(self.dataset.generator, torch.Generator)
        ):
            # 根据当前 epoch 和种子设置生成器的种子
            self.dataset.generator.manual_seed(self.seed + self.epoch)
        # 计算每个进程的真实批量大小
        real_batch_size = self.batch_size * self.num_processes
        # 计算当前进程的数据切片范围
        process_slice = range(self.process_index * self.batch_size, (self.process_index + 1) * self.batch_size)

        # 初始化第一个批次
        first_batch = None
        # 初始化当前批次列表
        current_batch = []
        # 对数据集进行迭代
        for element in self.dataset:
            # 增加样本计数器
            self.num_examples += 1
            # 将元素添加到当前批次列表中
            current_batch.append(element)
            # 等待形成完整批次后才产生元素
            if len(current_batch) == real_batch_size:
                # 生成当前批次中每个进程所需的元素
                for i in process_slice:
                    yield current_batch[i]
                # 如果第一个批次为空，记录当前批次为第一个批次
                if first_batch is None:
                    first_batch = current_batch.copy()
                # 重置当前批次列表
                current_batch = []

        # 如果 drop_last 为 True，则迭代结束
        # 否则，用开始的元素填充最后一个批次
        if not self.drop_last and len(current_batch) > 0:
            # 如果第一个批次为空，记录当前批次为第一个批次
            if first_batch is None:
                first_batch = current_batch.copy()
            # 将第一个批次的元素添加到当前批次，直到当前批次达到真实批量大小
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            # 生成最后一个批次中每个进程所需的元素
            for i in process_slice:
                yield current_batch[i]

    # 定义长度方法，返回数据集的长度
    def __len__(self):
        # 如果 drop_last 为 True，则舍弃不完整的最后一个批次
        if self.drop_last:
            # 计算舍弃不完整批次后的数据集长度
            return (len(self.dataset) // (self.batch_size * self.num_processes)) * self.batch_size
        # 否则，向上取整以确保包含所有样本
        else:
            return math.ceil(len(self.dataset) / (self.batch_size * self.num_processes)) * self.batch_size
    # 获取学习率的方法
    def _get_learning_rate(self):
        # 如果启用了 DeepSpeed，则可能在前几十个步骤内，由于损失尺度过大，优化器/调度器步骤可能不会运行，
        # 因此在这段时间内，如果在热身阶段调用 `get_last_lr`，会失败，因此需要处理这种情况：
        if self.is_deepspeed_enabled:
            try:
                # 获取最后一个学习率值
                last_lr = self.lr_scheduler.get_last_lr()[0]
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning("尝试在调度器/优化器开始步进之前获取lr值，返回lr=0")
                    last_lr = 0
                else:
                    raise
        else:
            # 如果 lr_scheduler 是 ReduceLROnPlateau 类型，则获取第一个参数组的学习率
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                last_lr = self.optimizer.param_groups[0]["lr"]
            else:
                # 否则，获取最后一个学习率值
                last_lr = self.lr_scheduler.get_last_lr()[0]
            # 如果学习率是张量类型，则转换为标量值
            if torch.is_tensor(last_lr):
                last_lr = last_lr.item()
        # 返回最后一个学习率值
        return last_lr


    # 将秒转换为 hh:mm:ss.msec 格式，毫秒保留两位小数
    def _secs2timedelta(secs):
        msec = int(abs(secs - int(secs)) * 100)
        return f"{datetime.timedelta(seconds=int(secs))}.{msec:02d}"


    # 将 Trainer 指标值重新格式化为易读格式
    def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        重新格式化 Trainer 指标值为人类可读格式

        Args:
            metrics (`Dict[str, float]`):
                训练/评估/预测返回的指标

        Returns:
            metrics (`Dict[str, float]`): 重新格式化后的指标
        """

        # 拷贝一份指标字典
        metrics_copy = metrics.copy()
        # 遍历指标字典
        for k, v in metrics_copy.items():
            # 如果指标键包含 "_mem_"，则将值转换为 MB
            if "_mem_" in k:
                metrics_copy[k] = f"{ v >> 20 }MB"
            # 如果指标键包含 "_runtime"，则将值转换为 hh:mm:ss.msec 格式
            elif "_runtime" in k:
                metrics_copy[k] = _secs2timedelta(v)
            # 如果指标键是 "total_flos"，则将值转换为 GF
            elif k == "total_flos":
                metrics_copy[k] = f"{ int(v) >> 30 }GF"
            # 如果指标值是浮点数类型，则将其四舍五入保留四位小数
            elif isinstance(metrics_copy[k], float):
                metrics_copy[k] = round(v, 4)

        return metrics_copy


    # 以特殊格式记录指标
    def log_metrics(self, split, metrics):
        """
        以特殊格式记录指标

        在分布式环境下，仅对进程等级为 0 的进程执行此操作。

        Args:
            split (`str`):
                模式/分割名称：其中之一为 `train`、`eval`、`test`
            metrics (`Dict[str, float]`):
                train/evaluate/predict 返回的指标字典

        内存报告说明：

        要获取内存使用情况报告，您需要安装 `psutil`。您可以使用 `pip install psutil` 进行安装。

        现在当运行此方法时，您将看到一个报告，其中包括：

        ```py
        init_mem_cpu_alloc_delta   =     1301MB
        init_mem_cpu_peaked_delta  =      154MB
        init_mem_gpu_alloc_delta   =      230MB
        init_mem_gpu_peaked_delta  =        0MB
        train_mem_cpu_alloc_delta  =     1345MB
        train_mem_cpu_peaked_delta =        0MB
        ```
        """
    train_mem_gpu_alloc_delta  =      693MB
    # 训练阶段 GPU 内存分配增量为 693MB
    train_mem_gpu_peaked_delta =        7MB
    # 训练阶段 GPU 内存峰值增量为 7MB

```py  
    **Understanding the reports:**
    # 理解报告内容：
    
    - the first segment, e.g., `train__`, tells you which stage the metrics are for. Reports starting with `init_`
        will be added to the first stage that gets run. So that if only evaluation is run, the memory usage for the
        `__init__` will be reported along with the `eval_` metrics.
    # 第一个部分，例如 `train__`，告诉您指标所属的阶段。以 `init_` 开头的报告将添加到第一个运行的阶段中。因此，如果只运行评估，那么 `__init__` 的内存使用情况将与 `eval_` 指标一起报告。
    
    - the third segment, is either `cpu` or `gpu`, tells you whether it's the general RAM or the gpu0 memory
        metric.
    # 第三个部分是 `cpu` 或 `gpu`，告诉您它是一般 RAM 还是 gpu0 内存指标。
    
    - `*_alloc_delta` - is the difference in the used/allocated memory counter between the end and the start of the
        stage - it can be negative if a function released more memory than it allocated.
    # `*_alloc_delta` - 是阶段结束和开始时已使用/已分配内存计数器之间的差异 - 如果函数释放的内存多于分配的内存，则可能为负。
    
    - `*_peaked_delta` - is any extra memory that was consumed and then freed - relative to the current allocated
        memory counter - it is never negative. When you look at the metrics of any stage you add up `alloc_delta` +
        `peaked_delta` and you know how much memory was needed to complete that stage.
    # `*_peaked_delta` - 是任何额外消耗然后释放的内存 - 相对于当前已分配的内存计数器 - 它永远不会为负。当您查看任何阶段的指标时，您将 `alloc_delta` + `peaked_delta` 相加，就知道完成该阶段需要多少内存。
    
    The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the
    main process does the bulk of work, but it could be not quite so if model parallel is used and then other GPUs may
    use a different amount of gpu memory. This is also not the same under DataParallel where gpu0 may require much more
    memory than the rest since it stores the gradient and optimizer states for all participating GPUS. Perhaps in the
    future these reports will evolve to measure those too.
    # 报告仅针对排名为 0 和 gpu 0 的进程进行。通常这已经足够，��为主进程执行大部分工作，但如果使用模型并行，则其他 GPU 可能使用不同数量的 GPU 内存。在 DataParallel 下也不同，因为 gpu0 可能需要比其余 GPU 更多的内存，因为它存储了所有参与 GPU 的梯度和优化器状态。也许在未来，这些报告将发展到测量这些内容。
    
    The CPU RAM metric measures RSS (Resident Set Size) includes both the memory which is unique to the process and the
    memory shared with other processes. It is important to note that it does not include swapped out memory, so the
    reports could be imprecise.
    # CPU RAM 指标测量 RSS（Resident Set Size），包括进程独有的内存和与其他进程共享的内存。重要的是要注意，它不包括交换出的内存，因此报告可能不够精确。
    
    The CPU peak memory is measured using a sampling thread. Due to python's GIL it may miss some of the peak memory if
    that thread didn't get a chance to run when the highest memory was used. Therefore this report can be less than
    reality. Using `tracemalloc` would have reported the exact peak memory, but it doesn't report memory allocations
    outside of python. So if some C++ CUDA extension allocated its own memory it won't be reported. And therefore it
    was dropped in favor of the memory sampling approach, which reads the current process memory usage.
    # CPU 峰值内存使用采用采样线程进行测量。由于 Python 的 GIL，如果该线程在使用最高内存时没有运行的机会，可能会错过一些峰值内存。因此，此报告可能低于实际情况。使用 `tracemalloc` 将报告确切的峰值内存，但它不会报告 Python 之外的内存分配。因此，如果某个 C++ CUDA 扩展分配了自己的内存，它将不会被报告。因此，它被放弃，采用内存采样方法，读取当前进程的内存使用情况。
    
    The GPU allocated and peak memory reporting is done with `torch.cuda.memory_allocated()` and
    `torch.cuda.max_memory_allocated()`. This metric reports only "deltas" for pytorch-specific allocations, as
    `torch.cuda` memory management system doesn't track any memory allocated outside of pytorch. For example, the very
    first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory.
    # GPU 分配和峰值内存报告使用 `torch.cuda.memory_allocated()` 和 `torch.cuda.max_memory_allocated()` 进行。此指标仅报告 pytorch 特定分配的“增量”，因为 `torch.cuda` 内存管理系统不跟踪 pytorch 之外分配的任何内存。例如，第一个 cuda 调用通常会加载 CUDA 内核，可能需要从 0.5 到 2GB 的 GPU 内存。
    
    Note that this tracker doesn't account for memory allocations outside of [`Trainer`]'s `__init__`, `train`,
    # 请注意，此跟踪器不考虑 [`Trainer`] 的 `__init__`、`train` 之外的内存分配。

```  
    """
    在 `evaluate` 和 `predict` 调用期间记录内存使用情况。
    
    因为 `evaluation` 调用可能发生在 `train` 过程中，我们不能处理嵌套调用，因为 `torch.cuda.max_memory_allocated` 是一个单一计数器，所以如果它被嵌套的评估调用重置，`train` 的跟踪器将报告不正确的信息。如果这个 [pytorch issue](https://github.com/pytorch/pytorch/issues/16266) 得到解决，将有可能将这个类改为可重入的。在此之前，我们只会跟踪 `train`、`evaluate` 和 `predict` 方法的外层。这意味着如果在 `train` 过程中调用 `eval`，将计入后者的内存使用量及前者的内存使用量。
    
    这也意味着如果其他工具在 [`Trainer`] 调用期间调用 `torch.cuda.reset_peak_memory_stats`，GPU 峰值内存统计数据可能无效。并且 [`Trainer`] 将干扰任何依赖于自行调用 `torch.cuda.reset_peak_memory_stats` 的此类工具的正常行为。
    
    为了获得最佳性能，您可能希望考虑在生产运行中关闭内存分析。
    """
    if not self.is_world_process_zero():
        return
    
    # 打印分割线和指标名称及值
    print(f"***** {split} metrics *****")
    # 格式化指标
    metrics_formatted = self.metrics_format(metrics)
    # 计算键和值的最大宽度
    k_width = max(len(str(x)) for x in metrics_formatted.keys())
    v_width = max(len(str(x)) for x in metrics_formatted.values())
    # 遍历并打印格式化后的指标
    for key in sorted(metrics_formatted.keys()):
        print(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")
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
    # 检查当前进程是否是全局进程中的第一个进程
    if not self.is_world_process_zero():
        return

    # 拼接保存路径
    path = os.path.join(self.args.output_dir, f"{split}_results.json")
    # 打开文件，将指标以JSON格式写入文件
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)

    # 如果设置了 combined 为 True，则更新合并的指标文件
    if combined:
        path = os.path.join(self.args.output_dir, "all_results.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        all_metrics.update(metrics)
        with open(path, "w") as f:
            json.dump(all_metrics, f, indent=4, sort_keys=True)


def save_state(self):
    """
    Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model

    Under distributed environment this is done only for a process with rank 0.
    """
    # 检查当前进程是否是全局进程中的第一个进程
    if not self.is_world_process_zero():
        return

    # 拼接保存路径，保存 Trainer 状态信息
    path = os.path.join(self.args.output_dir, "trainer_state.json")
    self.state.save_to_json(path)


def get_model_param_count(model, trainable_only=False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    # 如果启用了 DeepSpeed，定义一个函数来获取参数数量
    if is_deepspeed_zero3_enabled():

        def numel(p):
            return p.ds_numel if hasattr(p, "ds_numel") else p.numel()

    else:

        def numel(p):
            return p.numel()

    # 返回模型的参数总数
    return sum(numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    # 遍历模型的所有子模块
    for name, child in model.named_children():
        # 递归调用以获取每个子模块的参数名
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # 将模型特定参数（使用 nn.Parameter 定义）添加到结果中，因为它们不属于任何子模块
    result += list(model._parameters.keys())
    return result


def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.

    """
```py  
    """
    # 将模块的子模块列表存储在modules_children中
    modules_children = list(module.children())
    # 如果模块的类名与给定的名称相同，则返回模块的类
    if module.__class__.__name__ == name:
        return module.__class__
    # 如果模块没有子模块，则返回空
    elif len(modules_children) == 0:
        return
    # 否则，遍历子模块并递归调用该函数
    else:
        for child_module in modules_children:
            # 从子模块中获取指定名称的模块类
            module_class = get_module_class_from_name(child_module, name)
            # 如果找到了模块类，则返回该模块类
            if module_class is not None:
                return module_class
# 如果是主进程，则删除输出目录下的指定文件
def remove_dummy_checkpoint(is_main_process, output_dir, filenames):
    # 判断是否为主进程
    if is_main_process:
        # 遍历文件名列表
        for filename in filenames:
            # 构建文件路径
            file = os.path.join(output_dir, filename)
            # 如果文件存在则删除
            if os.path.isfile(file):
                os.remove(file)

# 如果 SageMaker 的模型并行功能被启用，则导入相关库
if is_sagemaker_mp_enabled():
    # 导入 torch 的模型并行库
    import smdistributed.modelparallel.torch as smp

    # 定义模型并行环境下的前向传播和反向传播函数装饰器
    @smp.step()
    def smp_forward_backward(model, inputs, gradient_accumulation_steps=1):
        # 执行模型的前向传播
        outputs = model(**inputs)
        # 获取损失值
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 损失值除以梯度累积步数
        loss /= gradient_accumulation_steps
        # 执行模型的反向传播
        model.backward(loss)
        # 返回损失值
        return loss

    # 定义模型并行环境下的仅前向传播函数装饰器
    @smp.step()
    def smp_forward_only(model, inputs):
        # 执行模型的前向传播
        return model(**inputs)

    # 定义模型并行环境下的全局数据收集函数
    def smp_gather(tensor):
        # 如果 tensor 是列表或元组类型，则递归地对其中的每个元素进行数据收集
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(smp_gather(t) for t in tensor)
        # 如果 tensor 是字典类型，则递归地对其中的每个值进行数据收集
        elif isinstance(tensor, dict):
            return type(tensor)({k: smp_gather(v) for k, v in tensor.items()})
        # 如果 tensor 不是 torch.Tensor 类型，则抛出类型错误异常
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )
        # 使用模型并行库进行全局数据收集
        all_tensors = smp.allgather(tensor, smp.CommGroup.DP_GROUP)
        # 将收集到的数据转移到 CPU 上，并连接成一个张量
        all_tensors = [atleast_1d(t) for t in all_tensors]
        return torch.cat([t.cpu() for t in all_tensors], dim=0)

    # 定义模型并行环境下的嵌套张量连接函数
    def smp_nested_concat(tensor):
        # 如果 tensor 是列表或元组类型，则递归地对其中的每个元素进行连接操作
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(smp_nested_concat(t) for t in tensor)
        # 如果 tensor 是字典类型，则递归地对其中的每个值进行连接操作
        elif isinstance(tensor, dict):
            return type(tensor)({k: smp_nested_concat(v) for k, v in tensor.items()})
        # 对于其他情况（通常是 StepOutput），直接执行连接操作并将结果转移到 CPU 上
        return tensor.concat().detach().cpu()
```