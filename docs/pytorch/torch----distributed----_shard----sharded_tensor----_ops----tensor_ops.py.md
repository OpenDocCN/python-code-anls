# `.\pytorch\torch\distributed\_shard\sharded_tensor\_ops\tensor_ops.py`

```
# mypy: allow-untyped-defs
# 引入必要的库和模块
import copy

import torch
from torch.distributed._shard.common_op_utils import _register_default_op
from torch.distributed._shard.sharded_tensor import (
    _sharded_op_impl,
    Shard,
    ShardedTensor,
)

from ._common import _register_sharded_op_on_local_shards


# 定义函数 _register_default_op，注册默认操作函数到指定的属性访问器
_register_default_op(torch.Tensor.shape.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.dtype.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.layout.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.size, _sharded_op_impl)
_register_default_op(torch.Tensor.dim, _sharded_op_impl)
_register_default_op(torch.Tensor.ndim.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.is_contiguous, _sharded_op_impl)
_register_default_op(torch.Tensor.contiguous, _sharded_op_impl)
_register_default_op(torch.Tensor.is_floating_point, _sharded_op_impl)

# 注册默认操作函数到 __reduce_ex__ 方法，用于序列化和反序列化
_register_default_op(torch.Tensor.__reduce_ex__, _sharded_op_impl)

# 注册默认操作函数到 requires_grad 属性的访问器函数
_register_default_op(torch.Tensor.requires_grad.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
# 注册默认操作函数到 grad 属性的访问器函数
# TODO: 使用包含所有本地梯度的 ShardedTensor 设置 grad
_register_default_op(torch.Tensor.grad.__get__, _sharded_op_impl)  # type: ignore[union-attr]
# 注册默认操作函数到 grad_fn 属性的访问器函数
_register_default_op(torch.Tensor.grad_fn.__get__, _sharded_op_impl)  # type: ignore[union-attr]
# 注册默认操作函数到 is_leaf 属性的访问器函数
_register_default_op(torch.Tensor.is_leaf.__get__, _sharded_op_impl)  # type: ignore[attr-defined]


# device 属性的处理函数，根据本地张量的当前设备返回设备属性
# 对于 ShardedTensor，选择本地分片的第一个张量的设备作为代表
@_sharded_op_impl(torch.Tensor.device.__get__)
def tensor_device(types, args=(), kwargs=None, pg=None):
    self_st = args[0]
    # 验证类型，确保输入是 ShardedTensor 类型
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")
    dev: torch.device
    # 如果存在本地分片，则返回第一个本地分片的设备
    if self_st._local_shards:
        dev = self_st._local_shards[0].tensor.device
    # 如果存在进程组并且使用 gloo 后端，则返回 CPU 设备
    elif pg and pg._get_backend_name() == "gloo":
        dev = torch.device("cpu")
    # 否则返回当前 CUDA 设备
    else:
        dev = torch.device(torch.cuda.current_device())
    return dev


# is_meta 属性的处理函数，返回本地张量是否为元张量
@_sharded_op_impl(torch.Tensor.is_meta.__get__)  # type: ignore[attr-defined]
def st_is_meta(types, args=(), kwargs=None, pg=None):
    return args[0].local_tensor().is_meta


# sharded_type_as_check 函数，用于 sharded_type_as 操作的额外检查
# 输入必须是 Tensor 或 ShardedTensor
def sharded_type_as_check(*args, **kwargs):
    """
    Perform extra checks for the sharded_type_as op such as the input needs to
    be either a Tensor or ShardedTensor.

    Args: same as ``torch.Tensor.type_as``.

    Return: None
    """
    if len(args) < 2:
        raise ValueError("Needs to give a tensor to cast type as!")
    # 检查 args[1] 是否不是 torch.Tensor 类型，并且不是 ShardedTensor 类型
    if not isinstance(args[1], torch.Tensor) and not isinstance(args[1], ShardedTensor):
        # 如果不是期望的类型，抛出 ValueError 异常，提示需要传入 Tensor 或 ShardedTensor 类型
        raise ValueError("Needs to give a Tensor or ShardedTensor to cast type as!")
# 定义函数，用于检查两个输入张量的数据类型是否相同
def same_dtype(*args, **kwargs):
    return args[0].dtype == args[1].dtype


# 处理 torch.Tensor.type_as 操作的 __torch_function__ 分发
def sharded_type_as(args, kwargs, pg):
    # 获取第一个参数作为 ShardedTensor
    st = args[0]
    # 获取第二个参数作为张量
    tensor = args[1]
    # 如果第二个参数是 ShardedTensor，则获取其本地张量
    if isinstance(tensor, ShardedTensor):
        tensor = tensor.local_tensor()
    # 初始化一个空列表，用于存储新的本地分片
    new_local_shards = []
    # 遍历当前 ShardedTensor 的本地分片
    for shard in st.local_shards():
        # 将每个分片中的张量类型转换为与给定张量相同类型，并保留其元数据，构建新的分片列表
        new_local_shards.append(Shard(shard.tensor.type_as(tensor), shard.metadata))
    # 深度复制当前 ShardedTensor 的元数据
    st_meta = copy.deepcopy(st._metadata)
    # 设置新的张量属性的数据类型为给定张量的数据类型
    st_meta.tensor_properties.dtype = tensor.dtype
    # 返回新的本地分片列表和更新后的元数据
    return new_local_shards, st_meta


# 注册 sharded_type_as 函数到本地分片的操作上
_register_sharded_op_on_local_shards(
    torch.Tensor.type_as,
    early_stop_func=same_dtype,
    extra_check=sharded_type_as_check,
    customized_func=sharded_type_as,
)


# 处理 ShardedTensor 的深度复制操作
def sharded_deepcopy(args, kwargs, pg):
    # 直接实现深度复制的魔法方法，而不使用默认的 tensor.__deepcopy__ 和实现 clone() 方法
    # 因为默认的 tensor deepcopy 会复制所有属性，但 ShardedTensor 中的 process_group 无法深复制
    self_st = args[0]
    # 深度复制当前 ShardedTensor 的本地分片列表
    new_local_shards = copy.deepcopy(self_st.local_shards())
    # 深度复制当前 ShardedTensor 的元数据
    new_metadata = copy.deepcopy(self_st.metadata())
    # 返回新的本地分片列表和元数据
    return new_local_shards, new_metadata


# 注册 sharded_deepcopy 函数到本地分片的操作上
_register_sharded_op_on_local_shards(
    torch.Tensor.__deepcopy__,
    customized_func=sharded_deepcopy,
)


# 处理 ShardedTensor 的原地复制操作
@_sharded_op_impl(torch.Tensor.copy_)
def sharded_inplace_copy(types, args, kwargs, pg):
    # 原地操作无需重新封装
    # 如果 kwargs 为 None，则设为一个空字典
    kwargs = {} if kwargs is None else kwargs
    # 获取第一个参数作为 self_st
    self_st = args[0]
    # 获取第二个参数作为 new_st
    new_st = args[1]
    # 获取非阻塞标志位
    nonblocking = kwargs.get("non_blocking", False)
    # 遍历当前 ShardedTensor 和新 ShardedTensor 的本地分片
    for local_shard, new_shard in zip(self_st.local_shards(), new_st.local_shards()):
        # 如果本地分片的元数据与新分片的元数据不相同，则抛出运行时错误
        if local_shard.metadata != new_shard.metadata:
            raise RuntimeError(
                "inplace copy can only happen between two ShardedTensor with same metadata!"
            )
    # 遍历当前 ShardedTensor 和新 ShardedTensor 的本地分片
    for local_shard, new_shard in zip(self_st.local_shards(), new_st.local_shards()):
        # 在本地分片上执行原地复制操作，传入非阻塞标志位
        local_shard.tensor.copy_(new_shard.tensor, nonblocking)

    # 返回原始 ShardedTensor
    return self_st


# 处理 ShardedTensor 的克隆操作
def sharded_clone(args, kwargs, pg):
    # 获取第一个参数作为 self_st
    self_st = args[0]
    # 获取内存格式选项
    desire_memory_format = kwargs.get("memory_format", None)
    # 如果指定了内存格式且不是 torch.preserve_format，则抛出运行时错误
    if desire_memory_format and desire_memory_format != torch.preserve_format:
        raise RuntimeError("Only support torch.preserve_format for ShardedTensor!")
    # 通过列表推导式创建克隆后的本地分片对象列表
    cloned_local_shards = [
        # 对每个本地分片对象进行克隆操作，指定内存格式为 desire_memory_format
        Shard(
            local_shard.tensor.clone(memory_format=desire_memory_format),
            # 复制本地分片对象的元数据
            metadata=copy.deepcopy(local_shard.metadata),
        )
        # 遍历 self_st 对象的本地分片列表
        for local_shard in self_st.local_shards()
    ]
    # 复制 self_st 对象的元数据
    new_metadata = copy.deepcopy(self_st.metadata())
    # 返回克隆后的本地分片列表和复制后的元数据
    return cloned_local_shards, new_metadata
# 在本地分片上注册分片操作
_register_sharded_op_on_local_shards(
    torch.Tensor.clone,  # 注册 torch.Tensor.clone 函数
    customized_func=sharded_clone,  # 使用 sharded_clone 作为自定义函数
)


def sharded_detach(args, kwargs, pg):
    self_st = args[0]  # 获取第一个参数作为 self_st
    # 将每个本地分片的 tensor 分离，并复制其 metadata，组成新的 Shard 对象列表
    detached_local_shards = [
        Shard(
            local_shard.tensor.detach(),
            metadata=copy.deepcopy(local_shard.metadata),
        )
        for local_shard in self_st.local_shards()
    ]
    new_metadata = copy.deepcopy(self_st.metadata())  # 深拷贝 self_st 的 metadata
    new_metadata.tensor_properties.requires_grad = False  # 设置新 metadata 的 requires_grad 为 False
    return detached_local_shards, new_metadata  # 返回分离后的本地分片列表和新的 metadata


# 在本地分片上注册分片操作
_register_sharded_op_on_local_shards(
    torch.Tensor.detach,  # 注册 torch.Tensor.detach 函数
    customized_func=sharded_detach,  # 使用 sharded_detach 作为自定义函数
)


@_sharded_op_impl(torch.Tensor.requires_grad_)
def tensor_requires_grad_set(types, args=(), kwargs=None, pg=None):
    self_st = args[0]  # 获取第一个参数作为 self_st
    # 验证类型，确保 self_st 是 ShardedTensor 类型
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")

    if kwargs is None:
        kwargs = {}

    requires_grad = args[1] if len(args) > 1 else kwargs.get("requires_grad", True)  # 获取 requires_grad 参数
    if requires_grad == self_st.requires_grad:
        return self_st

    # 针对每个本地分片的 tensor 设置 requires_grad 属性
    for local_shard in self_st.local_shards():
        local_shard.tensor.requires_grad_(requires_grad)

    # 使用 torch._C.DisableTorchFunctionSubclass() 禁用 Torch 函数子类化
    with torch._C.DisableTorchFunctionSubclass():
        self_st.requires_grad_(requires_grad)  # 设置 self_st 的 requires_grad 属性

    # 更新 self_st 的 metadata 中的 requires_grad 属性
    self_st._metadata.tensor_properties.requires_grad = requires_grad
    return self_st  # 返回更新后的 self_st
```