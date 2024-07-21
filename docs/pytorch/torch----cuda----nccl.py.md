# `.\pytorch\torch\cuda\nccl.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import collections
import warnings
from typing import Optional, Sequence, Union

# 导入 PyTorch 的 CUDA 模块
import torch.cuda

# 定义可以导出的符号列表
__all__ = ["all_reduce", "reduce", "broadcast", "all_gather", "reduce_scatter"]

# 定义操作类型为求和
SUM = 0  # ncclRedOp_t

# 检查是否支持 NCCL 操作
def is_available(tensors):
    if not hasattr(torch._C, "_nccl_all_reduce"):
        # 如果没有 NCCL 支持，发出警告并返回 False
        warnings.warn("PyTorch is not compiled with NCCL support")
        return False

    devices = set()
    for tensor in tensors:
        # 检查是否为稀疏张量，如果是则返回 False
        if tensor.is_sparse:
            return False
        # 检查是否为连续张量，如果不是则返回 False
        if not tensor.is_contiguous():
            return False
        # 检查是否在 CUDA 上，如果不是则返回 False
        if not tensor.is_cuda:
            return False
        # 获取张量所在的设备 ID，检查是否重复，如果有重复则返回 False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    return True


# 获取 NCCL 的版本信息
def version():
    """
    Returns the version of the NCCL.

    This function returns a tuple containing the major, minor, and patch version numbers of the NCCL.
    The suffix is also included in the tuple if a version suffix exists.
    Returns:
        tuple: The version information of the NCCL.
    """
    ver = torch._C._nccl_version()
    major = ver >> 32
    minor = (ver >> 16) & 65535
    patch = ver & 65535
    suffix = torch._C._nccl_version_suffix().decode("utf-8")
    if suffix == "":
        return (major, minor, patch)
    else:
        return (major, minor, patch, suffix)


# 获取 NCCL 的唯一 ID
def unique_id():
    return torch._C._nccl_unique_id()


# 使用 NCCL 初始化当前进程的排名信息
def init_rank(num_ranks, uid, rank):
    return torch._C._nccl_init_rank(num_ranks, uid, rank)


# 检查输入参数的类型，确保是张量的序列或单个张量
def _check_sequence_type(inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> None:
    if not isinstance(inputs, collections.abc.Container) or isinstance(
        inputs, torch.Tensor
    ):
        raise TypeError("Inputs should be a collection of tensors")


# 执行所有张量的全局归约操作
def all_reduce(inputs, outputs=None, op=SUM, streams=None, comms=None):
    _check_sequence_type(inputs)
    if outputs is None:
        outputs = inputs
    _check_sequence_type(outputs)
    # 调用 PyTorch 的 NCCL 接口进行全局归约
    torch._C._nccl_all_reduce(inputs, outputs, op, streams, comms)


# 执行张量的全局归约操作
def reduce(
    inputs: Sequence[torch.Tensor],
    output: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    root: int = 0,
    op: int = SUM,
    streams: Optional[Sequence[torch.cuda.Stream]] = None,
    comms=None,
    *,
    outputs: Optional[Sequence[torch.Tensor]] = None,
) -> None:
    _check_sequence_type(inputs)
    _output: torch.Tensor
    # 如果 `outputs` 参数不为空
    if outputs is not None:
        # 如果 `output` 参数也不为空，抛出数值错误
        if output is not None:
            raise ValueError(
                "'output' and 'outputs' can not be both specified. 'outputs' is deprecated in "
                "favor of 'output', taking in a single output tensor. The signature of reduce is: "
                "reduce(inputs, output=None, root=0, op=SUM, streams=None, comms=None)."
            )
        else:
            # 如果 `output` 参数为空，发出警告，说明 `outputs` 参数已经被废弃
            warnings.warn(
                "`nccl.reduce` with an output tensor list is deprecated. "
                "Please specify a single output tensor with argument 'output' instead instead.",
                FutureWarning,
                stacklevel=2,
            )
            # 取出指定索引 `root` 处的输出张量
            _output = outputs[root]
    # 如果 `outputs` 参数为空
    elif not isinstance(output, torch.Tensor) and isinstance(
        output, collections.abc.Sequence
    ):
        # 用户使用了旧的 API，用列表形式的输出张量作为位置参数调用
        warnings.warn(
            "nccl.reduce with an output tensor list is deprecated. "
            "Please specify a single output tensor.",
            FutureWarning,
            stacklevel=2,
        )
        # 取出指定索引 `root` 处的输出张量
        _output = output[root]
    else:
        # 如果以上条件都不满足，根据情况选择 `inputs` 或者 `output` 作为 `_output`
        _output = inputs[root] if output is None else output
    # 调用底层的 NCCL 函数进行张量归约操作
    torch._C._nccl_reduce(inputs, _output, root, op, streams, comms)
# 广播操作，将输入的张量广播给所有参与者
def broadcast(
    inputs: Sequence[torch.Tensor], root: int = 0, streams=None, comms=None
) -> None:
    # 检查输入张量列表的类型是否正确
    _check_sequence_type(inputs)
    # 使用 NCCL 库执行张量的广播操作
    torch._C._nccl_broadcast(inputs, root, streams, comms)


# 全部收集操作，将各参与者的输入张量收集到指定的输出张量列表中
def all_gather(
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    streams=None,
    comms=None,
) -> None:
    # 检查输入和输出张量列表的类型是否正确
    _check_sequence_type(inputs)
    _check_sequence_type(outputs)
    # 使用 NCCL 库执行全部收集操作
    torch._C._nccl_all_gather(inputs, outputs, streams, comms)


# 归约分散操作，根据指定的操作将输入张量分散到各参与者，并将结果写入输出张量列表
def reduce_scatter(
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    op: int = SUM,
    streams=None,
    comms=None,
) -> None:
    # 检查输入和输出张量列表的类型是否正确
    _check_sequence_type(inputs)
    _check_sequence_type(outputs)
    # 使用 NCCL 库执行归约分散操作，op 参数指定了归约的操作类型（例如求和）
    torch._C._nccl_reduce_scatter(inputs, outputs, op, streams, comms)
```