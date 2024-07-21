# `.\pytorch\torch\distributed\_shard\sharded_tensor\__init__.py`

```
# mypy: allow-untyped-defs
# 导入 functools 库，用于支持高阶函数
import functools
# 导入 List 和 TYPE_CHECKING 类型，用于类型提示
from typing import List, TYPE_CHECKING

# 导入 torch 库
import torch
# 从 torch.distributed._shard.op_registry_utils 模块导入 _decorator_func 函数
from torch.distributed._shard.op_registry_utils import _decorator_func

# 从 .api 模块中导入以下符号：
from .api import (
    _CUSTOM_SHARDED_OPS,
    _SHARDED_OPS,
    Shard,
    ShardedTensor,
    ShardedTensorBase,
    ShardedTensorMetadata,
    TensorProperties,
)
# 从 .metadata 模块导入 ShardMetadata 符号，忽略 F401 警告
from .metadata import ShardMetadata  # noqa: F401

# 如果 TYPE_CHECKING 为 True，导入 torch.distributed._shard.sharding_spec 中的 ShardingSpec 类
if TYPE_CHECKING:
    from torch.distributed._shard.sharding_spec import ShardingSpec
# 否则，将 ShardingSpec 定义为字符串 "ShardingSpec"
else:
    ShardingSpec = "ShardingSpec"

# 定义 empty 函数，返回一个未初始化数据填充的 ShardedTensor 对象
def empty(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    """
    Returns a :class:`ShardedTensor` filled with uninitialized data.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.contiguous_format``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    # 调用 ShardedTensor 构造函数，返回根据参数初始化的 ShardedTensor 对象
    return ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


# 定义 ones 函数，返回一个根据参数初始化为 1 的 ShardedTensor 对象
def ones(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    init_rrefs=False,


    # 初始化参数，控制是否初始化远程引用，默认为 False
def zeros(
    # 定义一个函数 `zeros`，用于创建一个填充了标量值 0 的 ShardedTensor 对象
    sharding_spec: ShardingSpec,
    # 参数 `sharding_spec` 指定如何对张量进行分片的规范
    *size,
    # 参数 `size` 是一个可变长度的整数序列，定义输出张量的形状
    dtype=None,
    # 关键字参数 `dtype`，指定返回张量的数据类型，如果为 `None`，则使用全局默认类型
    layout=torch.strided,
    # 关键字参数 `layout`，指定返回张量的布局，默认为 torch.strided
    requires_grad=False,
    # 关键字参数 `requires_grad`，指定是否需要记录张量的操作用于自动求导，默认为 False
    pin_memory=False,
    # 关键字参数 `pin_memory`，如果设置为 True，则张量将被分配在固定内存中，仅适用于 CPU 张量，默认为 False
    memory_format=torch.contiguous_format,
    # 关键字参数 `memory_format`，指定张量的存储格式，默认为 torch.contiguous_format
    process_group=None,
    # 关键字参数 `process_group`，指定要操作的进程组，默认为 None，将使用默认的进程组
    init_rrefs=False,
    # 关键字参数 `init_rrefs`，指定是否初始化指向远程分片的 RRef 对象，默认为 False
) -> ShardedTensor:
    # 函数返回一个 ShardedTensor 对象，填充了标量值 0
    """
    Returns a :class:`ShardedTensor` filled with the scalar value 0.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.
    """
    return full(
        sharding_spec,
        # 调用 full 函数，传递 `sharding_spec` 参数
        size,
        # 传递 `size` 参数，定义输出张量的形状
        fill_value=0,
        # 使用标量值 0 来填充张量
        dtype=dtype,
        # 传递 `dtype` 参数，指定数据类型
        layout=layout,
        # 传递 `layout` 参数，指定张量布局
        requires_grad=requires_grad,
        # 传递 `requires_grad` 参数，指定是否记录张量操作用于自动求导
        pin_memory=pin_memory,
        # 传递 `pin_memory` 参数，指定是否分配固定内存
        memory_format=memory_format,
        # 传递 `memory_format` 参数，指定张量的存储格式
        process_group=process_group,
        # 传递 `process_group` 参数，指定操作的进程组
        init_rrefs=init_rrefs,
        # 传递 `init_rrefs` 参数，指定是否初始化远程分片的 RRef 对象
    )
    # 使用指定的参数创建一个新的分片张量对象
    return full(
        sharding_spec,          # 分片规格，指定张量如何分布在多个设备上
        size,                   # 张量的总大小，即各个维度的大小乘积
        fill_value=0,           # 填充值，用于初始化张量的所有元素
        dtype=dtype,            # 数据类型，张量中元素的数据类型
        layout=layout,          # 张量的布局（如何存储在内存中）
        requires_grad=requires_grad,  # 是否记录张量上的操作以支持自动求导
        pin_memory=pin_memory,  # 是否将张量分配在锁定内存中（仅适用于CPU张量）
        memory_format=memory_format,  # 内存格式，用于张量存储优化
        process_group=process_group,  # 进程组，用于分布式处理的指定处理组
        init_rrefs=init_rrefs,  # 是否初始化指向远程分片的远程引用对象
    )
# 创建一个填充了指定值的分片张量对象。张量的数据类型将根据填充值自动推断，如果指定了dtype，则会覆盖从填充值推断出的类型。需要在所有 ranks 中以 SPMD 方式调用。
def full(
    sharding_spec: ShardingSpec,
    size,
    fill_value,
    *,
    dtype=None,  # 可选参数：返回张量的期望数据类型，默认为全局默认值
    layout=torch.strided,  # 可选参数：返回张量的期望布局，默认为 torch.strided
    requires_grad=False,  # 可选参数：是否记录返回张量的自动求导操作，默认为 False
    pin_memory=False,  # 可选参数：如果设置为 True，则返回的张量将分配在固定内存中，仅适用于 CPU 张量，默认为 False
    memory_format=torch.contiguous_format,  # 可选参数：返回张量的内存格式，默认为 torch.contiguous_format
    process_group=None,  # 可选参数：要处理的进程组，默认使用默认进程组
    init_rrefs=False,  # 可选参数：是否初始化指向远程分片的 torch.distributed.rpc.RRef 对象，默认为 False
) -> ShardedTensor:
    """
    Creates a :class:`ShardedTensor` filled with fill_value. The tensor's dtype
        is inferred from fill_value. If dtype is specified, it will override the
        inferred type from fill_value. Needs to be called on all ranks in an SPMD fashion.
    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...):  a list, tuple, or `torch.Size` of integers defining the shape of the
            output tensor.
        fill_value (Scalar) - the value to fill the output tensor with.
    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.
    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    # 使用给定的参数创建一个 ShardedTensor 对象
    sharded_tensor = ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )
    # 使用常数 fill_value 来初始化 sharded_tensor，忽略类型检查
    torch.nn.init.constant_(sharded_tensor, fill_value)  # type: ignore[arg-type]
    # 返回创建的 ShardedTensor 对象
    return sharded_tensor


def rand(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    """
    Creates a :class:`ShardedTensor` filled with random numbers from a uniform distribution
        on the interval :math:`[0, 1)`. The shape of the tensor is defined by the
        variable argument `size`. Needs to be called on all ranks in an SPMD fashion.
    """
    """
    Creates a sharded tensor with specified sharding and properties, and initializes it uniformly.
    
    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...):  a list, tuple, or `torch.Size` of integers defining the shape of the
            output tensor.
    
    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.
    
    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    sharded_tensor = ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )
    torch.nn.init.uniform_(sharded_tensor, 0, 1)  # Initialize the sharded tensor uniformly
    return sharded_tensor
def randn(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    """
    Creates a :class:`ShardedTensor` filled with random numbers from a uniform distribution
        with mean `0` and variance `1` (also called standard normal distribution). The shape
        of the tensor is defined by the variable argument `size`. Needs to be called on all ranks
        in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...):  a list, tuple, or `torch.Size` of integers defining the shape of the
            output tensor.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    # 创建一个使用给定参数初始化的ShardedTensor对象
    sharded_tensor = ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )
    # 在ShardedTensor对象上应用正态分布的随机初始化
    torch.nn.init.normal_(sharded_tensor, 0, 1)  # type: ignore[arg-type]
    # 返回初始化后的ShardedTensor对象
    return sharded_tensor


def init_from_local_shards(
    local_shards: List[Shard], *global_size, process_group=None, init_rrefs=False
) -> ShardedTensor:
    """
    Creates an :class:`ShardedTensor` from local shards and the global metadata.
    Needs to be called on all ranks in an SPMD fashion.

    Args:
        local_shards (List[:class `torch.distributed._shard.sharded_tensor.Shard`]): A list
            of shards that represent the local shards on this rank.
        global_size (int...):  a list, tuple, or `torch.Size` of integers defining the
            shape of the overall sharded tensor.


    """
    # 创建一个ShardedTensor对象，使用本地分片和全局元数据初始化
    return ShardedTensor(
        local_shards,
        *global_size,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )
    Keyword args:
        process_group (ProcessGroup, optional): 要操作的进程组。如果为 None，
            将使用默认进程组。
        init_rrefs (bool, optional): 是否初始化指向远程分片的
            :class:`torch.distributed.rpc.RRef`。如果设为 ``True``，
            需要初始化 RPC 框架。
            默认值为 ``False``。

    Returns:
        A :class:`ShardedTensor` object handle on this rank
        返回一个当前进程的 :class:`ShardedTensor` 对象句柄

    Examples:
        Suppose we want construct a sharded tensor on two ranks, global size = (10, 5),
        each shard have a (5, 5) local tensor, we can do it like below:

        on rank 0:
        >>> # xdoctest: +SKIP("not distributed")
        >>> local_shard_metadata = ShardMetadata(
        >>>     shard_offsets=[0, 0],
        >>>     shard_lengths=[5, 5],
        >>>     placement="rank:0/cuda:0"
        >>> )
        >>> local_shards = [Shard(torch.randn(5, 5), local_shard_metadata)]
        >>> sharded_tensor = init_from_local_shards(local_shards, [10, 5])

        on rank 1:
        >>> # xdoctest: +SKIP("not distributed")
        >>> local_shard_metadata = ShardMetadata(
        >>>     shard_offsets=[5, 0],
        >>>     shard_lengths=[5, 5],
        >>>     placement="rank:1/cuda:1"
        >>> )
        >>> local_shards = [Shard(torch.randn(5, 5), local_shard_metadata)]
        >>> sharded_tensor = init_from_local_shards(local_shards, [10, 5])
    """
    使用本地分片初始化一个 :class:`ShardedTensor` 对象，
    在指定的进程组和初始化远程引用（如果指定）的情况下。
    返回初始化后的 :class:`ShardedTensor` 对象。
    return ShardedTensor._init_from_local_shards(
        local_shards, *global_size, process_group=process_group, init_rrefs=init_rrefs
    )
# 定义一个函数，用于将 ShardedTensor 添加到模块的 state_dict 中，需要使用 torch.nn.Module._register_state_dict_hook 方法注册到模块中
def state_dict_hook(module, destination, prefix, local_metadata):
    # 遍历模块的所有子模块
    for submodule_name, submodule in module.named_modules():
        # 遍历子模块的所有属性
        for attr_name, attr in submodule.__dict__.items():
            # 如果属性是 ShardedTensor 类型，则将其添加到 state_dict 中
            if isinstance(attr, ShardedTensor):
                mod_prefix = prefix + submodule_name
                key = mod_prefix + ("." if mod_prefix else "") + attr_name
                destination[key] = attr


# 定义一个函数，用于在加载 state_dict 之前添加 ShardedTensor 到模块中
def pre_load_state_dict_hook(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # 遍历模块的所有子模块
    for submodule_name, submodule in module.named_modules():
        # 遍历子模块的所有属性名
        for attr_name in submodule.__dict__.keys():
            mod_prefix = prefix + submodule_name
            key = mod_prefix + ("." if mod_prefix else "") + attr_name
            # 如果 key 在 state_dict 中存在且对应的值是 ShardedTensor 类型，则将其设置为子模块的属性
            if key in state_dict:
                if isinstance(state_dict[key], ShardedTensor):
                    setattr(submodule, attr_name, state_dict[key])


# 提供一种方式让用户编写自定义的分片操作符
def custom_sharded_op_impl(func):
    """
    Provides a way for users to write their own custom sharded operator. This
    can be used to override existing ShardedTensor operators or write a new
    one not supported by ShardedTensor. If the operator in question is covered
    by ``__torch_function__`` dispatch and has a ShardedTensor as any of its
    parameters, the function provided will be invoked for that operator.

    Example::
        >>> # xdoctest: +SKIP
        >>> @custom_sharded_op_impl(torch.nn.functional.linear)
        >>> def my_custom_sharded_linear(types, args, kwargs, process_group):
        >>>     ...
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> input = torch.rand(10, 32)
        >>> weight = sharded_tensor.rand(32, 16)
        >>> bias = torch.rand(16)
        >>> # This will call 'my_custom_sharded_linear'
        >>> torch.nn.functional.linear(input, weight, bias)

    The types, args and kwargs parameters are the same parameters that are
    passed to ``__torch_function__`` dispatch API
    (https://pytorch.org/docs/stable/notes/extending.html#extending-torch).
    There is an additional ``process_group`` parameter which is the
    process_group used for the ShardedTensor and can be used by
    implementations for communications within a sharded implementation.

    Args:
        func(Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.nn.functional.linear)
    """
    return functools.partial(_decorator_func, op=func, op_table=_CUSTOM_SHARDED_OPS)


# 注册默认的分片操作符
def _sharded_op_impl(func):
    """
    Decorator to register a default sharded op.
    """
    return functools.partial(_decorator_func, op=func, op_table=_SHARDED_OPS)


# 导入所有内置的分片操作符
# 导入所有来自 _ops 模块的符号，以便在当前模块中使用它们
from ._ops import *  # noqa: F403
```