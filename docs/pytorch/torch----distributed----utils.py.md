# `.\pytorch\torch\distributed\utils.py`

```
# 设置类型检查，允许未类型化的函数定义
# 导入必要的模块和类
import dataclasses  # 数据类的支持库
import traceback  # 追踪异常的支持库
from typing import (
    Any,  # 通用类型
    Callable,  # 可调用对象
    Container,  # 容器类型
    Dict,  # 字典类型
    List,  # 列表类型
    Optional,  # 可选类型
    OrderedDict,  # 有序字典类型
    overload,  # 函数重载支持
    Tuple,  # 元组类型
    TypeVar,  # 泛型变量
)

# 导入 PyTorch 相关模块
import torch  # PyTorch 核心库
import torch.distributed as dist  # 分布式支持
from torch import nn  # 神经网络模块
from torch.nn.parallel._functions import _get_stream  # 并行计算中的流获取函数
from torch.nn.parallel.scatter_gather import _is_namedtuple  # 并行计算中的命名元组检查函数
from torch.nn.utils.rnn import PackedSequence  # 压缩序列的支持

# 定义 __all__ 以避免 IDE 警告
__all__ = []  # type: ignore[var-annotated]


def _pack_kwargs(*args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
    """
    将参数列表拆分为独立的关键字列表和值列表（unpack_kwargs 则相反）。

    参考链接: https://github.com/facebookresearch/fairscale/blob/eeb6684/fairscale/internal/containers.py#L70
    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}

    Returns:
        Tuple[Tuple[Any, ...], Tuple[str, ...]]: 第一个元组元素同时给出了位置参数和关键字参数的值列表，
        其中位置参数在前，关键字参数在后，并且关键字参数的顺序与关键字列表一致。
        第二个元组元素给出了关键字参数的键列表，长度最多不超过第一个元组元素的长度。
    """
    kwarg_keys: List[str] = []  # 初始化空的关键字列表
    flat_args: List[Any] = list(args)  # 将位置参数转换为列表形式

    # 遍历关键字参数，将键和值分别添加到对应的列表中
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)

    return tuple(flat_args), tuple(kwarg_keys)  # 返回打包后的位置参数和关键字参数的元组


def _cast_forward_inputs(
    dtype: Optional[torch.dtype],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    将 ``args`` 和 ``kwargs`` 中的浮点数张量转换为 ``dtype`` 指定的类型。

    这会保留张量上已有的 ``requires_grad`` 标志。
    """
    if dtype is None:
        return args, kwargs

    def cast_fn(x: torch.Tensor) -> torch.Tensor:
        # 如果张量不是浮点数类型或已经是指定的 dtype，则直接返回
        if not torch.is_floating_point(x) or x.dtype == dtype:
            return x
        return x.to(dtype)  # 否则将张量转换为指定的 dtype

    return (_apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs))


def _unpack_kwargs(
    flat_args: Tuple[Any, ...], kwarg_keys: Tuple[str, ...]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    参见 _pack_kwargs。
    """
    assert len(kwarg_keys) <= len(
        flat_args
    ), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"

    if len(kwarg_keys) == 0:
        return flat_args, {}  # 如果关键字列表为空，则直接返回位置参数和空的关键字参数字典

    args = flat_args[: -len(kwarg_keys)]  # 获取位置参数
    kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :]))  # 构建关键字参数字典

    return args, kwargs  # 返回解包后的位置参数和关键字参数字典


S = TypeVar("S", dict, list, tuple)  # 定义泛型变量 S，可以是 dict、list 或 tuple 类型
T = TypeVar("T", torch.Tensor, PackedSequence)  # 定义泛型变量 T，可以是 torch.Tensor 或 PackedSequence 类型


@overload
def _recursive_to(
    inputs: S, target_device: torch.device, use_side_stream_for_tensor_copies: bool
) -> List[S]:
    ...


@overload
def _recursive_to(
    inputs: T, target_device: torch.device, use_side_stream_for_tensor_copies: bool
) -> List[T]:
    ...
    # 定义函数参数：
    # - T: 输入的张量或数据
    # - target_device: 目标设备，指定张量将被复制到的设备
    # - use_side_stream_for_tensor_copies: 布尔值，指示是否使用侧通道进行张量复制操作
# 定义一个函数签名，接受任意数量的参数并返回单个类型参数的元组
) -> Tuple[T]:
    ...


# 递归地将输入对象移动到指定的目标设备
def _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies):
    r"""Recursively moves input to the target_device."""

    # 定义一个递归函数，将输入对象移动到目标设备
    def to_map(obj):
        if isinstance(obj, (torch.Tensor, PackedSequence)):
            # 获取对象的当前设备
            device = obj.data.device if isinstance(obj, PackedSequence) else obj.device
            # 如果对象已在目标设备上，则直接返回
            if device == target_device:
                return (obj,)
            # 如果不允许使用辅助流进行张量拷贝，直接将对象拷贝到目标设备
            if not use_side_stream_for_tensor_copies:
                return (obj.to(target_device),)
            else:
                # 如果设备类型为 CPU 或者对应的设备模块不存在，直接拷贝到目标设备
                device_mod = getattr(torch, device.type, None)
                if device.type == "cpu" or device_mod is None:
                    return (obj.to(target_device),)
                # 在后台流中执行 CPU -> 目标设备的拷贝
                stream = _get_stream(target_device)
                with device_mod.stream(stream):
                    output = obj.to(target_device)
                # 同步拷贝流
                with device_mod.device(target_device.index):
                    current_stream = device_mod.current_stream()
                    # 将当前流与拷贝流同步
                    current_stream.wait_stream(stream)
                    # 确保在主流上的工作完成前不要重用张量内存
                    if isinstance(obj, PackedSequence):
                        output.data.record_stream(current_stream)  # type: ignore[arg-type]
                    else:
                        assert isinstance(output, torch.Tensor)
                        output.record_stream(current_stream)  # type: ignore[arg-type]
                return (output,)
        if _is_namedtuple(obj):
            # 如果是命名元组，则递归处理每个成员
            return [type(obj)(*args) for args in zip(*map(to_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            # 如果是元组且不为空，则递归处理每个元素
            return list(zip(*map(to_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            # 如果是列表且不为空，则递归处理每个元素
            return [list(i) for i in zip(*map(to_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            # 如果是字典且不为空，则递归处理每个键值对
            return [type(obj)(i) for i in zip(*map(to_map, obj.items()))]
        # 其他情况直接返回对象本身
        return [obj]

    # 避免循环引用
    try:
        res = to_map(inputs)
    finally:
        to_map = None  # type: ignore[assignment]
    return res


# 在反向上下文中打印错误消息而不是使用 assert 来打印错误消息的替代方法
def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error message ``s`` since otherwise, it is swallowed."""
    if not cond:
        print(s)
        traceback.print_stack()
        if raise_assertion_error:
            raise AssertionError(s)


# 为给定大小为 ``size`` 的 ``tensor`` 分配存储空间
def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.
    """
    # 使用 `torch.no_grad()` 上下文管理器，禁用梯度计算以优化内存和速度
    with torch.no_grad():
        # 检查当前环境是否处于 TorchDynamo 编译状态
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            # 检查张量的存储空间是否已经分配
            already_allocated = tensor._typed_storage()._size() == size.numel()
            # 如果存储空间还未分配
            if not already_allocated:
                # 获取当前张量的存储空间大小
                tensor_storage_size = tensor._typed_storage()._size()
                # 断言当前张量的存储空间应该为 0，如果不是则抛出异常
                _p_assert(
                    tensor_storage_size == 0,
                    "Tensor storage should have been resized to be 0 but got PLACEHOLDEr",
                )
                # 调整张量的存储空间大小为指定的元素数目
                tensor._typed_storage()._resize_(size.numel())
def _free_storage(tensor: torch.Tensor):
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    # 使用 torch.no_grad() 上下文管理器，确保在计算图中不记录梯度信息
    with torch.no_grad():
        # 检查当前是否处于 torchdynamo 编译环境下，若不是则执行释放操作
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            # 判断张量的存储是否已经被释放
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                # 断言张量的存储偏移为0，以及在不是唯一占用存储时释放存储是不安全的
                _p_assert(
                    tensor.storage_offset() == 0,
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}",
                )
                # 调整张量的存储大小为0，实现释放存储操作
                tensor._typed_storage()._resize_(0)


Q = TypeVar("Q")
R = TypeVar("R", dict, list, tuple, set, OrderedDict, PackedSequence, Any)


@overload
def _apply_to_tensors(fn: Callable[[torch.Tensor], Q], container: torch.Tensor) -> Q:
    ...


@overload
def _apply_to_tensors(fn: Callable[[torch.Tensor], Any], container: R) -> R:
    ...


def _apply_to_tensors(fn, container):
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(x):
        # 若 x 是 torch.Tensor 类型，则应用给定函数 fn 到该张量上
        if isinstance(x, torch.Tensor):
            return fn(x)
        # 若 x 是具有 __dataclass_fields__ 属性的数据类对象，则对其字段递归应用函数
        elif hasattr(x, "__dataclass_fields__"):
            dc = dataclasses.replace(x)
            for f in dataclasses.fields(dc):
                name = f.name
                setattr(dc, name, apply(getattr(dc, name)))
            return dc
        # 若 x 是 OrderedDict 类型，则递归应用到其每个键值对上
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        # 若 x 是 PackedSequence 类型，则递归应用到其数据上
        elif isinstance(x, PackedSequence):
            apply(x.data)
            return x
        # 若 x 是 dict 类型，则递归应用到其每个键值对的值上
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        # 若 x 是命名元组类型，则递归应用到其每个元素上，并返回相同类型的新命名元组
        elif _is_namedtuple(x):
            res = (apply(el) for el in x)
            return type(x)(*res)
        # 若 x 是 list、tuple 或 set 类型，则递归应用到其中每个元素上
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            # 若 x 不是上述类型之一，则直接返回 x
            return x

    return apply(container)


def _to_kwargs(
    inputs: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    target_device: torch.device,
    use_side_stream_for_tensor_copies: bool,
) -> Tuple[Tuple[Any, ...], Tuple[Dict[str, Any], ...]]:
    # 将输入和关键字参数递归移动到目标设备上，用于张量复制时使用
    moved_inputs = (
        _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies)
        if inputs
        else []
    )
    moved_kwargs = (
        _recursive_to(kwargs, target_device, use_side_stream_for_tensor_copies)
        if kwargs
        else []
    )
    # 如果移动后的输入数量少于关键字参数数量，则用空元组补齐
    if len(moved_inputs) < len(moved_kwargs):
        moved_inputs.extend([() for _ in range(len(moved_kwargs) - len(inputs))])
    # 如果移动的关键字参数数量少于移动的输入参数数量，则进行补充
    elif len(moved_kwargs) < len(moved_inputs):
        # 使用空字典补充移动的关键字参数，直到其数量与移动的输入参数数量相等
        moved_kwargs.extend([{} for _ in range(len(moved_inputs) - len(moved_kwargs))])
    # 返回移动后的输入参数和关键字参数，转换为元组形式
    return tuple(moved_inputs), tuple(moved_kwargs)
def _verify_param_shape_across_processes(
    process_group: dist.ProcessGroup,
    tensors: List[torch.Tensor],
    logger: Optional["dist.Logger"] = None,
):
    """
    Verify that the shapes of parameters across processes are consistent.

    Args:
        process_group (dist.ProcessGroup): Distributed process group.
        tensors (List[torch.Tensor]): List of tensors representing parameters.
        logger (Optional["dist.Logger"], optional): Optional logger for logging messages.

    Returns:
        bool: True if parameters have consistent shapes across processes, False otherwise.
    """
    return dist._verify_params_across_processes(process_group, tensors, logger)


def _sync_module_states(
    module: nn.Module,
    process_group: dist.ProcessGroup,
    broadcast_bucket_size: int,
    src: int,
    params_and_buffers_to_ignore: Container[str],
    broadcast_buffers: bool = True,
) -> None:
    """
    Sync ``module``'s parameters and buffers state across all processes.

    Args:
        module (nn.Module): Module whose parameters and buffers are to be synchronized.
        process_group (dist.ProcessGroup): Distributed process group.
        broadcast_bucket_size (int): Size of buckets used for broadcasting tensors.
        src (int): Source rank from which to broadcast.
        params_and_buffers_to_ignore (Container[str]): Set of parameter and buffer names to ignore during synchronization.
        broadcast_buffers (bool, optional): Flag indicating whether to broadcast buffers as well (default is True).

    Returns:
        None
    """
    module_states: List[torch.Tensor] = []
    for name, param in module.named_parameters():
        if name not in params_and_buffers_to_ignore:
            module_states.append(param.detach())

    if broadcast_buffers:
        for name, buffer in module.named_buffers():
            if name not in params_and_buffers_to_ignore:
                module_states.append(buffer.detach())

    _sync_params_and_buffers(process_group, module_states, broadcast_bucket_size, src)


def _sync_params_and_buffers(
    process_group: dist.ProcessGroup,
    module_states: List[torch.Tensor],
    broadcast_bucket_size: int,
    src: int,
) -> None:
    """
    Synchronize ``module_states`` across all processes by broadcasting them from rank 0.

    Args:
        process_group (dist.ProcessGroup): Distributed process group.
        module_states (List[torch.Tensor]): List of tensors representing module states to synchronize.
        broadcast_bucket_size (int): Size of buckets used for broadcasting tensors.
        src (int): Source rank from which to broadcast.

    Returns:
        None
    """
    if len(module_states) > 0:
        dist._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, src
        )


def _replace_by_prefix(
    state_dict: Dict[str, Any],
    old_prefix: str,
    new_prefix: str,
) -> None:
    """
    Replace all keys in ``state_dict`` that match ``old_prefix`` with ``new_prefix`` in-place.

    Args:
        state_dict (Dict[str, Any]): State dictionary to modify.
        old_prefix (str): Prefix to search for in keys.
        new_prefix (str): Prefix to replace with in keys.

    Returns:
        None
    """
    if old_prefix == new_prefix:
        raise ValueError("old_prefix and new_prefix must be distinct")
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]


def _data_ptr_allocated(tensor: torch.Tensor) -> bool:
    """
    Check if the data pointer of the tensor's storage is allocated.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        bool: True if the data pointer is greater than 0, False otherwise.
    """
    return tensor.untyped_storage().data_ptr() > 0
```