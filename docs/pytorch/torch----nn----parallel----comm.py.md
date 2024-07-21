# `.\pytorch\torch\nn\parallel\comm.py`

```
# mypy: allow-untyped-defs
# 导入警告模块，用于处理警告信息
import warnings
# 导入类型提示模块 List
from typing import List

# 导入 PyTorch 模块
import torch
# 从 torch._utils 中导入多个函数
from torch._utils import (
    _flatten_dense_tensors,
    _get_device_index,
    _handle_complex,
    _reorder_tensors_as,
    _take_tensors,
    _unflatten_dense_tensors,
)
# 导入 torch.cuda 中的 nccl 模块
from torch.cuda import nccl


# 定义函数 broadcast，用于将张量广播到指定的 GPU 设备
def broadcast(tensor, devices=None, *, out=None):
    r"""Broadcasts a tensor to specified GPU devices.

    Args:
        tensor (Tensor): tensor to broadcast. Can be on CPU or GPU.
        devices (Iterable[torch.device, str or int], optional): an iterable of
          GPU devices, among which to broadcast.
        out (Sequence[Tensor], optional, keyword-only): the GPU tensors to
          store output results.

    .. note::
        Exactly one of :attr:`devices` and :attr:`out` must be specified.

    Returns:
        - If :attr:`devices` is specified,
            a tuple containing copies of :attr:`tensor`, placed on
            :attr:`devices`.
        - If :attr:`out` is specified,
            a tuple containing :attr:`out` tensors, each containing a copy of
            :attr:`tensor`.
    """
    # 处理复数张量
    tensor = _handle_complex(tensor)
    # 检查 devices 和 out 是否只有一个被指定
    if not ((devices is None) ^ (out is None)):
        raise RuntimeError(
            f"Exactly one of 'devices' and 'out' must be specified, but got devices={devices} and out={out}"
        )
    # 如果指定了 devices，则将 tensor 广播到指定的设备上
    if devices is not None:
        devices = [_get_device_index(d) for d in devices]
        return torch._C._broadcast(tensor, devices)
    else:
        # 如果指定了 out，则将 tensor 广播到 out 指定的张量上
        return torch._C._broadcast_out(tensor, out)


# 定义函数 broadcast_coalesced，用于将一系列张量广播到指定的 GPU 设备
def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """Broadcast a sequence of tensors to the specified GPUs.

    Small tensors are first coalesced into a buffer to reduce the number of synchronizations.

    Args:
        tensors (sequence): tensors to broadcast. Must be on the same device,
          either CPU or GPU.
        devices (Iterable[torch.device, str or int]): an iterable of GPU
          devices, among which to broadcast.
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple containing copies of :attr:`tensor`, placed on :attr:`devices`.
    """
    # 获取设备索引列表
    devices = [_get_device_index(d) for d in devices]
    # 处理复数张量
    tensors = [_handle_complex(t) for t in tensors]
    # 调用 torch._C._broadcast_coalesced 函数进行张量的合并广播
    return torch._C._broadcast_coalesced(tensors, devices, buffer_size)


# 定义函数 reduce_add，用于对多个 GPU 上的张量进行求和
def reduce_add(inputs, destination=None):
    """Sum tensors from multiple GPUs.

    All inputs should have matching shapes, dtype, and layout. The output tensor
    will be of the same shape, dtype, and layout.

    Args:
        inputs (Iterable[Tensor]): an iterable of tensors to add.
        destination (int, optional): a device on which the output will be
            placed (default: current device).

    Returns:
        A tensor containing an elementwise sum of all inputs, placed on the
        :attr:`destination` device.
    """
    # 获取目标设备的索引
    destination = _get_device_index(destination, optional=True)
    # 获取输入张量的尺寸
    input_size = inputs[0].size()
    root_index = None  # 初始化一个变量，用于记录已经在正确设备上的输入张量的索引

    # 遍历输入张量列表，检查每个张量是否在 GPU 上，同时查找目标设备的索引
    for i, inp in enumerate(inputs):
        assert inp.device.type != "cpu", "reduce_add expects all inputs to be on GPUs"
        if inp.get_device() == destination:
            root_index = i  # 找到目标设备的张量索引
        if inp.size() != input_size:
            got = "x".join(str(x) for x in inp.size())
            expected = "x".join(str(x) for x in input_size)
            raise ValueError(
                f"input {i} has invalid size: got {got}, but expected {expected}"
            )

    # 如果未找到目标设备的张量，则抛出运行时错误
    if root_index is None:
        raise RuntimeError(
            "reduce_add expects destination to be on the same GPU with one of the tensors"
        )

    # 如果只有一个输入张量，则直接返回该张量
    if len(inputs) == 1:
        return inputs[0]

    # 如果使用了 NCCL 库，则进行张量的 reduce 操作
    if nccl.is_available(inputs):
        result = torch.empty_like(inputs[root_index])
        nccl.reduce(inputs, output=result, root=root_index)
    else:
        # 否则，根据指定的目标设备创建新的结果张量
        destination_device = torch.device(inputs[root_index].device.type, destination)
        nonroot = [t for i, t in enumerate(inputs) if i != root_index]
        # 不使用 clone 方法创建新张量
        result = inputs[root_index] + nonroot[0].to(
            device=destination_device, non_blocking=True
        )
        for other in nonroot[1:]:
            result.add_(other.to(device=destination_device, non_blocking=True))

    return result
def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """Sum tensors from multiple GPUs.

    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Args:
        inputs (Iterable[Iterable[Tensor]]): iterable of iterables that
            contain tensors from a single device.
        destination (int, optional): a device on which the output will be
            placed (default: current device).
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple of tensors containing an elementwise sum of each group of
        inputs, placed on the ``destination`` device.
    """
    # TODO: When `len(inputs) == 1` and all inputs are on `destination`, just
    #       return `inputs`.

    # 初始化一个列表，每个子列表代表一个 GPU 上的输入张量集合
    dense_tensors: List[List] = [[] for _ in inputs]  # shape (num_gpus, num_tensors)
    
    # 初始化输出结果列表
    output = []
    
    # 初始化参考顺序列表
    ref_order = []
    
    # 首先处理稀疏张量，因为它们可能在不同 GPU 上有不同的大小
    for tensor_at_gpus in zip(*inputs):
        if all(t.is_sparse for t in tensor_at_gpus):
            # 如果所有的张量都是稀疏的，使用 reduce_add 函数对它们进行求和（结果也是稀疏张量）
            result = reduce_add(tensor_at_gpus, destination)
            output.append(result)
            ref_order.append(tensor_at_gpus[0])
        else:
            # 否则，将稠密张量添加到对应 GPU 的列表中
            for coll, t in zip(dense_tensors, tensor_at_gpus):
                coll.append(t.to_dense() if t.is_sparse else t)
            ref_order.append(dense_tensors[0][-1])
    
    # 准备用于处理稠密张量的迭代器列表
    itrs = [_take_tensors(tensors, buffer_size) for tensors in dense_tensors]
    
    # 现在处理稠密张量，它们具有一致的大小
    for chunks in zip(*itrs):
        # 将每个 GPU 的分块稠密张量展平
        flat_tensors = [
            _flatten_dense_tensors(chunk) for chunk in chunks
        ]  # (num_gpus,)
        
        # 对展平后的张量进行求和
        flat_result = reduce_add(flat_tensors, destination)
        
        # 将求和后的张量重新展平为原始形状
        for t in _unflatten_dense_tensors(flat_result, chunks[0]):
            # 这些未展平的张量不共享存储空间，并且我们也不暴露基础的展平张量，因此给它们不同的版本计数器。
            # 参见注释 [ Version Counter in comm.*_coalesced ]
            output.append(t.data)
    
    # 返回结果，按照参考顺序重新排列张量
    return tuple(_reorder_tensors_as(output, ref_order))


def scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None, *, out=None):
    """Scatters tensor across multiple GPUs.
    Args:
        tensor (Tensor): 需要分散的张量，可以是在 CPU 或 GPU 上。
        devices (Iterable[torch.device, str or int], optional): GPU 设备的可迭代对象，用于分散张量。
        chunk_sizes (Iterable[int], optional): 每个设备上放置的数据块大小。应与 :attr:`devices` 的长度相匹配，并且总和应等于 ``tensor.size(dim)``。如果未指定，则将 :attr:`tensor` 均匀分成块。
        dim (int, optional): 在哪个维度上进行分块。默认为 ``0``。
        streams (Iterable[torch.cuda.Stream], optional): GPU 流的可迭代对象，用于执行分散操作。如果未指定，则使用默认流。
        out (Sequence[Tensor], optional, keyword-only): 存储输出结果的 GPU 张量。这些张量的大小必须与 :attr:`tensor` 的大小相匹配，除了 :attr:`dim` 外，总大小必须等于 ``tensor.size(dim)``。

    .. note::
        必须确保 :attr:`devices` 和 :attr:`out` 中至少有一个被指定。当指定了 :attr:`out` 时，不能指定 :attr:`chunk_sizes`，而会从 :attr:`out` 的大小推断出来。

    Returns:
        - 如果指定了 :attr:`devices`，
            返回一个元组，包含分散到 :attr:`devices` 上的 :attr:`tensor` 的各个块。
        - 如果指定了 :attr:`out`，
            返回一个元组，包含 :attr:`out` 张量，每个张量包含 :attr:`tensor` 的一个块。
def gather(tensors, dim=0, destination=None, *, out=None):
    r"""Gathers tensors from multiple GPU devices.

    Args:
        tensors (Iterable[Tensor]): an iterable of tensors to gather.
          Tensor sizes in all dimensions other than :attr:`dim` have to match.
        dim (int, optional): a dimension along which the tensors will be
          concatenated. Default: ``0``.
        destination (torch.device, str, or int, optional): the output device.
          Can be CPU or CUDA. Default: the current CUDA device.
        out (Tensor, optional, keyword-only): the tensor to store gather result.
          Its sizes must match those of :attr:`tensors`, except for :attr:`dim`,
          where the size must equal ``sum(tensor.size(dim) for tensor in tensors)``.
          Can be on CPU or CUDA.

    .. note::
        :attr:`destination` must not be specified when :attr:`out` is specified.

    Returns:
        - If :attr:`destination` is specified,
            a tensor located on :attr:`destination` device, that is a result of
            concatenating :attr:`tensors` along :attr:`dim`.
        - If :attr:`out` is specified,
            the :attr:`out` tensor, now containing results of concatenating
            :attr:`tensors` along :attr:`dim`.
    """
    # 对输入的每个张量进行处理，确保处理后的张量是标量
    tensors = [_handle_complex(t) for t in tensors]
    
    # 如果没有提供输出张量，根据目的地设备获取索引，并调用底层的 torch._C._gather 函数
    if out is None:
        # 如果目的地是 -1，发出警告，建议使用字符串 "cpu" 代替 -1
        if destination == -1:
            warnings.warn(
                "Using -1 to represent CPU tensor is deprecated. Please use a "
                'device object or string instead, e.g., "cpu".',
                FutureWarning,
                stacklevel=2,
            )
        # 获取目标设备的索引，允许使用 CPU，如果未指定则使用当前 CUDA 设备
        destination = _get_device_index(destination, allow_cpu=True, optional=True)
        # 调用底层的 torch._C._gather 函数，返回结果张量
        return torch._C._gather(tensors, dim, destination)
    else:
        # 如果同时指定了目的地和输出张量，则抛出运行时错误
        if destination is not None:
            raise RuntimeError(
                f"'destination' must not be specified when 'out' is specified, but got destination={destination}"
            )
        # 调用底层的 torch._C._gather_out 函数，使用指定的输出张量，返回结果张量
        return torch._C._gather_out(tensors, out, dim)
```