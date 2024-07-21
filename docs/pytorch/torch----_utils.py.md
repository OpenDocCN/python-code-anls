# `.\pytorch\torch\_utils.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import copyreg
import functools
import logging
import sys
import threading
import traceback
import warnings
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Generic, List, Optional
from typing_extensions import ParamSpec

import torch


def _type(self, dtype=None, non_blocking=False, **kwargs):
    """Returns the type if `dtype` is not provided, else casts this object to
    the specified type.

    If this is already of the correct type, no copy is performed and the
    original object is returned.

    Args:
        dtype (type or string): The desired type
        non_blocking (bool): If ``True``, and the source is in pinned memory
            and destination is on the GPU or vice versa, the copy is performed
            asynchronously with respect to the host. Otherwise, the argument
            has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument. The ``async`` arg is deprecated.
    """
    # 根据参数 `non_blocking` 和 `kwargs` 获取异步标志
    non_blocking = _get_async_or_non_blocking("type", non_blocking, kwargs)
    # 如果 `dtype` 为 None，则返回对象的类型表示
    if dtype is None:
        return self.__module__ + "." + self.__class__.__name__

    # 如果 `dtype` 是字符串，则将其导入为类型
    if isinstance(dtype, str):
        dtype = _import_dotted_name(dtype)
    # 如果对象已经是指定的类型，则直接返回
    if dtype == type(self):
        return self
    # 如果对象是稀疏张量且目标类型不是稀疏张量，则抛出异常
    if self.is_sparse:
        if not dtype.is_sparse:
            raise RuntimeError("Cannot cast sparse tensor to dense tensor")
        # 创建新的稀疏张量对象
        new_module_name = dtype.__module__.replace(".sparse", "")
        new_values_type_name = new_module_name + "." + dtype.__name__
        new_values = torch.Tensor._values(self).type(new_values_type_name, non_blocking)
        new_indices_type_name = new_module_name + ".LongTensor"
        new_indices = torch.Tensor._indices(self).type(
            new_indices_type_name, non_blocking
        )
        return dtype(new_indices, new_values, self.size())
    # 如果对象是稠密张量且目标类型是稀疏张量，则抛出异常
    if dtype.is_sparse:
        raise RuntimeError("Cannot cast dense tensor to sparse tensor")
    # 否则，创建目标类型的对象，并拷贝数据
    return dtype(self.size()).copy_(self, non_blocking)


def _to(self, device, non_blocking=False):
    """Returns a copy of this object in device memory.

    If this object is already on the correct device, then no copy is performed
    and the original object is returned.

    Args:
        device (int): The destination device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
    """
    # 如果对象已经在目标设备上，则直接返回
    if self.device == device:
        return self

    # 获取目标设备的模块
    device_module = getattr(torch, device.type, None)
    # 如果设备模块不存在，则抛出异常
    assert (
        device_module is not None
    ), f"{device.type.upper()} device module is not loaded"
    # 使用指定的设备上下文执行以下代码块
    with device_module.device(device):
        # 如果当前张量是稀疏的，并且设备模块具有稀疏张量的相关属性
        if self.is_sparse and hasattr(device_module, "sparse"):
            # 获取当前张量对应设备上的稀疏类型
            new_type = getattr(device_module.sparse, self.__class__.__name__)
            # 获取张量的索引，并将其移动到指定设备上，非阻塞操作
            indices = getattr(torch.Tensor._indices(self), device.type)(
                device, non_blocking
            )
            # 获取张量的值，并将其移动到指定设备上，非阻塞操作
            values = getattr(torch.Tensor._values(self), device.type)(
                device, non_blocking
            )
            # 返回一个新的稀疏张量，包括移动后的索引、值和原张量的大小
            return new_type(indices, values, self.size())
        else:
            # 断言当前张量不是稀疏张量，否则抛出异常
            assert (
                not self.is_sparse
            ), f"sparse storage is not supported for {device.type.upper()} tensors"
            # 创建一个未命名类型的存储，使用指定的设备
            untyped_storage = torch.UntypedStorage(self.size(), device=device)
            # 将当前张量的数据复制到未命名类型的存储中，非阻塞操作
            untyped_storage.copy_(self, non_blocking)
            # 返回新创建的未命名类型的存储
            return untyped_storage
# 根据函数名和关键字参数 kwargs 返回非阻塞标志。
def _get_async_or_non_blocking(function_name, non_blocking, kwargs):
    # 如果没有传入关键字参数，直接返回默认的非阻塞标志
    if not kwargs:
        return non_blocking
    # 如果关键字参数数量不是1或者不包含 'async'，则抛出类型错误异常
    if len(kwargs) != 1 or "async" not in kwargs:
        message = "{}() got an unexpected keyword argument '{}'"
        argument = list(kwargs.keys()).pop()
        raise TypeError(message.format(function_name, argument))
    # 发出警告，提示 'async' 已弃用，请使用 'non_blocking'
    warnings.warn("'async' is deprecated; use 'non_blocking'")
    # 返回 'async' 关键字参数的值
    return kwargs["async"]


# 创建线程本地状态对象
_thread_local_state = threading.local()


# 返回重建函数中的映射位置。
def _get_restore_location(device):
    # 获取线程本地状态中的 map_location 属性
    map_location = getattr(_thread_local_state, "map_location", None)
    # 如果 map_location 为 None，则返回传入的 device 参数
    if map_location is None:
        return device
    else:
        # 如果 map_location 是字典，则返回对应 device 的值，否则直接返回 map_location
        if isinstance(map_location, dict):
            return map_location.get(device, device)
        # 如果 map_location 是字符串或者 torch.device 类型，则直接返回 map_location
        elif isinstance(map_location, (str, torch.device)):
            return map_location
        # 如果 map_location 是可调用对象，则抛出运行时错误
        else:
            assert callable(map_location)
            raise RuntimeError(
                "Callable map_location not supported with _rebuild_wrapper_subclass "
                "or _rebuild_device_tensor_from_numpy"
            )


# Note [Don't serialize hooks]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 自时间久远以来，我们已经序列化了与变量关联的反向钩子。
# 这种方式部分工作正常--Python 可以pickle全局函数（但不能pickle闭包！）--但也存在问题。
#
#   - 它很脆弱。如果将一个反向钩子序列化到保存的模型中，然后重命名与钩子关联的函数，
#     现在你的保存模型就损坏了，无法再加载它。
#
#   - 它实际上并没有被使用。标准建议是序列化模型的 *state_dict*，而不是模型本身
#     （因为这对影响模型序列化的代码变更更加稳定），而状态字典仅保存了 "数据"，因此
#     剥离了反向钩子。在某些情况下，钩子对模型的良好功能是至关重要的（例如，DDP），
#     但是 DDP 已经管理了重新添加钩子！
#
#   - 我们在许多情况下并没有序列化它们。在 #10220 之前，我们在 ForkingPickler 中
#     放弃了反向钩子。我们“修复”了这个问题，以便与其他序列化站点方便地进行匹配，但
#     实际上不序列化反向钩子并不是 bug 的根本原因。
#
# 有了这些情况的考虑，我们决定一个更好的策略是根本不序列化钩子。
#
# 鉴于这是一个破坏向后兼容性的更改，我们应该在我们先前序列化了一个钩子但现在不再
# 这样做时发出警告。这将通过添加一个特殊的
# 当 _torch_serialize_ignore 属性被设置时，将不会发出此警告。如果一个 hook
# 拥有 _torch_serialize_ignore 属性，当我们尝试序列化一个附有该 hook 的 Tensor 时，将不会发出警告。
#
# 另外，当跳过 _backward_hooks 时，如果传递 None，将会违反 #12219，因此我们必须传递一个空的 OrderedDict()。

# TODO: 一旦我们决定中断序列化 FC，`storage` 就不再需要是 TypedStorage。
def _rebuild_tensor(storage, storage_offset, size, stride):
    # 首先构建一个具有正确 dtype/device 的 Tensor
    t = torch.empty((0,), dtype=storage.dtype, device=storage._untyped_storage.device)
    return t.set_(storage._untyped_storage, storage_offset, size, stride)


def get_tensor_metadata(tensor):
    # Tensor 的序列化元数据。
    # 目前，这只返回一个 dict[string, bool]，指定是否设置了 `conj` 或 `neg` 位。
    assert isinstance(tensor, torch.Tensor)
    return torch._C._get_tensor_metadata(tensor)  # type: ignore[attr-defined]


def set_tensor_metadata(tensor, metadata):
    # 参见上面的 `get_tensor_metadata`
    assert isinstance(metadata, dict)
    assert isinstance(tensor, torch.Tensor)
    torch._C._set_tensor_metadata(tensor, metadata)  # type: ignore[attr-defined]


def _rebuild_tensor_v2(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    metadata=None,
):
    tensor = _rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    if metadata:
        set_tensor_metadata(tensor, metadata)

    # 注意：此行仅用于向后兼容性；通常期望 backward_hooks 是一个空的 OrderedDict。参见 Note [Don't serialize hooks]
    tensor._backward_hooks = backward_hooks
    return tensor


def _rebuild_tensor_v3(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    dtype,
    metadata=None,
):
    t = torch.empty(
        (0,),
        dtype=dtype,
        device=storage._untyped_storage.device,
        requires_grad=requires_grad,
    )
    t.set_(storage._untyped_storage, storage_offset, size, stride)
    if metadata:
        set_tensor_metadata(t, metadata)
    t._backward_hooks = backward_hooks
    return t


_sparse_tensors_to_validate: List["torch.Tensor"] = []


# 在 serialization.py 的 _legacy_load() 中，我们在反序列化稀疏张量之后解压存储器。
# 这些存储器包含用于验证稀疏张量的必要数据：索引和值。
# 这就是为什么稀疏张量首先在没有任何验证的情况下进行反序列化，然后在 _legacy_load() 返回之前调用该函数的原因，
# 以便所有稀疏张量可以一次性进行验证。
#
# _load() 在 serialization.py 中必须遵循相同的过程，因为由于 Pickler 的语义，我们必须使用相同的（非验证的）函数来
# 加载数据。
# 验证加载的稀疏张量，不考虑调用者。
def _validate_loaded_sparse_tensors():
    try:
        # 遍历需要验证的稀疏张量列表
        for t in _sparse_tensors_to_validate:
            # 如果张量的布局是 COO 格式
            if t.layout is torch.sparse_coo:
                # 调用 PyTorch 内部方法验证 COO 格式的稀疏张量参数
                torch._validate_sparse_coo_tensor_args(
                    t._indices(), t._values(), t.size(), t.is_coalesced()
                )
            # 如果布局在 CSR、CSC、BSR 或 BSC 中
            elif t.layout in {
                torch.sparse_csr,
                torch.sparse_csc,
                torch.sparse_bsr,
                torch.sparse_bsc,
            }:
                # TODO: 当前验证涉及到一个昂贵的 CPU 遍历操作，可能包括设备传输。
                # 如果布局是 CSR 或 BSR
                if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    # 提取压缩索引和普通索引
                    compressed_indices, plain_indices = (
                        t.crow_indices(),
                        t.col_indices(),
                    )
                else:
                    # 提取压缩索引和普通索引
                    compressed_indices, plain_indices = (
                        t.ccol_indices(),
                        t.row_indices(),
                    )
                # 调用 PyTorch 内部方法验证压缩格式的稀疏张量参数
                torch._validate_sparse_compressed_tensor_args(
                    compressed_indices, plain_indices, t.values(), t.size(), t.layout
                )
            else:
                # 如果遇到未实现的布局类型，抛出未实现错误
                raise NotImplementedError(
                    f"_validate_loaded_sparse_tensors for layout `{t.layout}`"
                )

    finally:
        # 清空需要验证的稀疏张量列表
        _sparse_tensors_to_validate.clear()


def _rebuild_sparse_tensor(layout, data):
    """
    从稀疏存储表示重建稀疏张量。

    Args:
        layout (str): 张量的稀疏存储布局。
        data (tuple): 张量的稀疏存储表示。
    """
    # 如果布局是 COO 格式
    if layout == torch.sparse_coo:
        # 根据数据的长度确定参数格式
        if len(data) == 3:
            indices, values, size = data
            is_coalesced = None
        else:
            indices, values, size, is_coalesced = data
        # 创建 COO 格式的稀疏张量
        result = torch.sparse_coo_tensor(
            indices, values, size, check_invariants=False, is_coalesced=is_coalesced
        )
        # 将重建的稀疏张量添加到验证列表中
        _sparse_tensors_to_validate.append(result)
        return result

    # 如果布局在 CSR、CSC、BSR 或 BSC 中
    elif layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        # 提取压缩索引、普通索引、值和大小
        compressed_indices, plain_indices, values, size = data
        # 创建压缩格式的稀疏张量
        result = torch.sparse_compressed_tensor(
            compressed_indices,
            plain_indices,
            values,
            size,
            layout=layout,
            check_invariants=False,
        )
        # 将重建的稀疏张量添加到验证列表中
        _sparse_tensors_to_validate.append(result)
        return result

    # 如果遇到未实现的布局类型，抛出未实现错误
    raise NotImplementedError(f"rebuilding sparse tensor for layout {layout}")


def _rebuild_nested_tensor(buffer, sizes, strides, storage_offsets):
    # 调用 PyTorch 内部方法，从缓冲区重建嵌套张量视图
    return torch._nested_view_from_buffer(buffer, sizes, strides, storage_offsets)


def _rebuild_device_tensor_from_numpy(data, dtype, device, requires_grad):
    # 获取数据恢复的设备位置
    device = _get_restore_location(device)
    # 从给定的 NumPy 数组创建一个 Torch 张量，并指定数据类型和设备位置
    tensor = torch.from_numpy(data).to(dtype=dtype, device=device)
    # 设置张量是否需要梯度计算
    tensor.requires_grad = requires_grad
    # 返回创建的张量对象
    return tensor
# 用于向后兼容，不建议使用，仅用于加载使用旧版本PyTorch序列化的张量
_rebuild_xla_tensor = _rebuild_device_tensor_from_numpy

# 重建没有存储的元数据张量
def _rebuild_meta_tensor_no_storage(dtype, size, stride, requires_grad):
    return torch.empty_strided(
        size, stride, dtype=dtype, device="meta", requires_grad=requires_grad
    )

# 重建包装器子类
def _rebuild_wrapper_subclass(
    cls,
    dtype,
    size,
    stride,
    storage_offset,
    layout,
    device,
    requires_grad,
):
    # 获取恢复位置的设备
    device = _get_restore_location(device)
    return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
        cls,
        size,
        strides=stride,
        dtype=dtype,
        storage_offset=storage_offset,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )

# TODO: 一旦决定打破序列化FC，`storage`不再需要是一个TypedStorage
# 重建量化张量
def _rebuild_qtensor(
    storage,
    storage_offset,
    size,
    stride,
    quantizer_params,
    requires_grad,
    backward_hooks,
):
    # 获取量化方案
    qscheme = quantizer_params[0]
    if qscheme == torch.per_tensor_affine:
        # 对于每个张量仿射量化
        _, scale, zero_point = quantizer_params
        tensor = torch._empty_affine_quantized(
            size,
            scale=scale,
            zero_point=zero_point,
            dtype=storage.dtype,
            device=storage.device,
        )
    elif qscheme in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
        # 对于每个通道仿射量化
        _, scales, zero_points, axis = quantizer_params
        if type(scales) is list and type(zero_points) is list:
            if qscheme == torch.per_channel_affine:
                scales = torch.tensor(scales, dtype=torch.double, device=storage.device)
                zero_points = torch.tensor(
                    zero_points, dtype=torch.long, device=storage.device
                )
            else:
                scales = torch.tensor(scales, dtype=torch.float, device=storage.device)
                zero_points = torch.tensor(
                    zero_points, dtype=torch.float, device=storage.device
                )
        tensor = torch._empty_per_channel_affine_quantized(
            size,
            scales=scales,
            zero_points=zero_points,
            axis=axis,
            dtype=storage.dtype,
            device=storage.device,
        )
    else:
        # 抛出运行时错误，无法反序列化给定的量化张量方案
        raise RuntimeError(f"Can't deserialize quantized tensor with qscheme {qscheme}")
    # 设置张量的数据，偏移量，大小和步长
    tensor.set_(storage, storage_offset, size, stride)
    # 设置张量是否需要梯度
    tensor.requires_grad = requires_grad
    # 注意：这一行仅用于向后兼容；一般期望是backward_hooks是一个空的OrderedDict。参见注释[不序列化hooks]
    tensor._backward_hooks = backward_hooks
    return tensor

# 重建参数
def _rebuild_parameter(data, requires_grad, backward_hooks):
    param = torch.nn.Parameter(data, requires_grad)
    # 注意：这一行仅用于向后兼容；
    # 将参数对象的反向传播钩子设置为传入的 backward_hooks
    # 一般期望 backward_hooks 是一个空的 OrderedDict。参见 Note [不要序列化钩子]
    param._backward_hooks = backward_hooks

    # 返回设置完反向传播钩子的参数对象
    return param
# 重新构建带有状态的参数对象
def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
    # 使用给定的数据和requires_grad创建一个torch.nn.Parameter对象
    param = torch.nn.Parameter(data, requires_grad)
    # 注意：此行代码仅用于向后兼容性；通常期望backward_hooks是一个空的OrderedDict。
    # 参见备注 [Don't serialize hooks]
    param._backward_hooks = backward_hooks

    # 恢复Parameter对象的状态，类似于Python属性
    param = _set_obj_state(param, state)
    return param


def _get_obj_state(obj):
    # 获取Python子类的状态
    # 这部分代码大致模仿了对象类上的函数，但由于Tensor不继承它，无法直接调用该函数
    # 参考：https://github.com/python/cpython/blob/c83919bd635f4433f1c6ae8504996a9fe3c215e5/Objects/typeobject.c#L4891
    # 注意：从Python 3.11开始，__getstate__总是被定义的，因此else分支将永远不会执行。
    getstate_fn = getattr(obj, "__getstate__", None)
    if getstate_fn:
        state = getstate_fn()
    else:
        # 如果没有定义__getstate__方法，则尝试获取类的_slots_属性名列表，并将其保存为状态
        slots_to_save = copyreg._slotnames(obj.__class__)  # type: ignore[attr-defined]
        if slots_to_save:
            state = (
                obj.__dict__,
                {
                    name: getattr(obj, name)
                    for name in slots_to_save
                    if hasattr(obj, name)
                },
            )
        else:
            state = obj.__dict__

    return state


def _set_obj_state(obj, state):
    # 根据提供的状态(state)设置对象(obj)的属性
    if isinstance(state, tuple):
        if not len(state) == 2:
            raise RuntimeError(f"Invalid serialized state: {state}")
        dict_state = state[0]
        slots_state = state[1]
    else:
        dict_state = state
        slots_state = None

    # 从Python 3.11开始，__dict__属性是延迟创建的，并且在不需要时序列化为None。
    if dict_state:
        # 设置对象的__dict__属性中的键值对
        for k, v in dict_state.items():
            setattr(obj, k, v)

    if slots_state:
        # 设置对象的_slots_属性中的键值对
        for k, v in slots_state.items():
            setattr(obj, k, v)
    return obj


def _import_dotted_name(name):
    # 根据点分隔的名称(name)，导入并返回相应的对象
    components = name.split(".")
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj


def _flatten_dense_tensors(tensors):
    """将稠密张量展平为连续的1D缓冲区。假设张量都是相同的稠密类型。

    由于输入是稠密的，结果张量将是一个连接的1D缓冲区。对该缓冲区的逐元素操作等同于
    对各个张量进行操作。

    Args:
        tensors (Iterable[Tensor]): 要展平的稠密张量。

    Returns:
        包含输入张量的连续1D缓冲区。
    """
    return torch._C._nn.flatten_dense_tensors(tensors)


def _flatten_sparse_tensors(tensors):
    """将稀疏张量展平为两个连续的1D缓冲区，一个是索引，一个是值。假设张量都是相同的稀疏类型。
    ```
    Args:
        tensors (Iterable[Tensor]): sparse tensors to flatten.

    Returns:
        A tuple of two contiguous 1D buffers, one containing input tensors'
        indices and the other containing the values.
    """
    # 将输入稀疏张量的索引展平为一个连续的1维缓冲区
    flat_indices = torch._C._nn.flatten_dense_tensors(
        [torch.Tensor._indices(t) for t in tensors]
    )
    # 将输入稀疏张量的值展平为一个连续的1维缓冲区
    flat_values = torch._C._nn.flatten_dense_tensors(
        [torch.Tensor._values(t) for t in tensors]
    )
    # 返回展平后的索引和值组成的元组
    return flat_indices, flat_values
def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    # 使用 Torch C++ 后端函数 unflatten_dense_tensors，根据给定的 flat 和 tensors 解析成未压缩的稠密张量
    return torch._C._nn.unflatten_dense_tensors(flat, tensors)


def _unflatten_sparse_tensors(flat, tensors):
    """View flat buffer (containing indices and values) using the sizes of
    tensors. Assume that tensors are of same sparse type, and that flat is given
    by _flatten_sparse_tensors.

    Args:
        flat (tuple(Tensor, Tensor)): flattened indices and values of sparse
          tensors to unflatten.
        tensors (Iterable[Tensor]): sparse tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened sparse tensors with sizes same as tensors and values from
        flat.
    """
    flat_indices, flat_values = flat
    # 使用 Torch C++ 后端函数 unflatten_dense_tensors，根据给定的 flat_indices 和 tensors 的索引生成器，解析成未压缩的稀疏张量的索引
    indices = torch._C._nn.unflatten_dense_tensors(
        flat_indices, [torch.Tensor._indices(t) for t in tensors]
    )
    # 使用 Torch C++ 后端函数 unflatten_dense_tensors，根据给定的 flat_values 和 tensors 的值生成器，解析成未压缩的稀疏张量的值
    values = torch._C._nn.unflatten_dense_tensors(
        flat_values, [torch.Tensor._values(t) for t in tensors]
    )
    outputs = []
    # 根据输入的 tensors、indices 和 values 创建新的稀疏张量
    for t, i, v in zip(tensors, indices, values):
        outputs.append(t.new(i, v, t.size()))
    return tuple(outputs)


def _reorder_tensors_as(tensors, ordered_tensors):
    """Assume that tensors are of same order as ordered_tensors within their
    types, e.g., from _take_tensors. Reorder them to be of same order as
    ordered_tensors.

    Args:
        tensors (Iterable[Tensor]): tensors to be reordered. They should be of
          the same order as ordered_tensors within their own types.
        ordered_tensors (Iterable[Tensor]): tensors whose order will be the
          reference.

    Returns:
        Ordered tuple of tensors with contents from tensors and order of
        ordered_tensors.
    """
    # 使用 defaultdict 创建类型字典，将输入的 tensors 按类型分组
    type_dict = defaultdict(list)
    for tensor in tensors:
        type_dict[tensor.type()].append(tensor)
    # 创建类型字典的副本，其中每个值是相应类型 tensors 的迭代器
    type_dict_ = {t: iter(coll) for t, coll in type_dict.items()}
    # 根据 ordered_tensors 的顺序重排输入 tensors，并返回为元组
    return tuple(next(type_dict_[tensor.type()]) for tensor in ordered_tensors)


def _take_tensors(tensors, size_limit):
    """Group tensors into chunks. This generator yields a chunk at each time,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Yields:
        Blocks of tensors of same type and within size_limit. The yielded
        tensors are only ordered as the original sequence within its types.
    """
    # 创建默认字典 buf_dict，用于按类型分组并记录每个分组的总大小
    buf_dict: DefaultDict[str, List] = defaultdict(lambda: [[], 0])
    # 遍历给定的张量列表
    for tensor in tensors:
        # 获取张量的类型
        t = tensor.type()
        # 检查张量是否稀疏
        if tensor.is_sparse:
            # 如果张量是稀疏的，获取稀疏张量的索引和数值
            indices = torch.Tensor._indices(tensor)
            values = torch.Tensor._values(tensor)
            # 计算稀疏张量占用的空间大小
            size = (
                indices.numel() * indices.element_size()
                + values.numel() * values.element_size()
            )
        else:
            # 如果张量不是稀疏的，计算张量占用的空间大小
            size = tensor.numel() * tensor.element_size()
        
        # 查找当前张量类型在缓冲区字典中的条目
        buf_and_size = buf_dict[t]
        
        # 如果当前缓冲区的大小加上新张量的大小超过了限制，并且当前缓冲区不为空
        if buf_and_size[1] + size > size_limit and buf_and_size[1] > 0:
            # 生成当前缓冲区
            yield buf_and_size[0]
            # 重置当前类型的缓冲区为新的空列表和大小为0
            buf_and_size = buf_dict[t] = [[], 0]
        
        # 将当前张量添加到缓冲区中
        buf_and_size[0].append(tensor)
        # 更新当前缓冲区的大小
        buf_and_size[1] += size
    
    # 遍历缓冲区字典中的每一个缓冲区
    for buf, _ in buf_dict.values():
        # 如果缓冲区不为空，则生成该缓冲区
        if len(buf) > 0:
            yield buf
# annotation decorator to get annotations in a way that is compatible
# with both Python 2 and 3
def annotate(ret, **kwargs):
    # 返回一个装饰器函数，用于设置函数的注解
    def dec(fun):
        # 设置函数的注解，包括参数和返回值的注解
        fun.__annotations__ = dict(kwargs)
        fun.__annotations__["return"] = ret
        return fun

    return dec


def render_call(fn, args, kwargs):
    # 解析函数名称为字符串表示形式
    str_fn = torch.overrides.resolve_name(fn)
    if str_fn is None:
        str_fn = str(fn)

    # 初始化参数字符串列表
    str_args: List[str] = []
    # 设置打印选项，控制输出格式
    with torch._tensor_str.printoptions(threshold=0, edgeitems=0):
        # 将参数及其值转换为字符串表示形式并加入列表
        str_args.extend(repr(a) for a in args)
        str_args.extend(f"{k}={repr(v)}" for k, v in kwargs.items())
        # 格式化函数调用字符串
        r = f"{str_fn}({', '.join(str_args)})"
    return r


# NOTE [ Python Traceback Reference Cycle Problem ]
#
# When using sys.exc_info(), it is important to **not** store the exc_info[2],
# which is the traceback, because otherwise you will run into the traceback
# reference cycle problem, i.e., the traceback holding reference to the frame,
# and the frame (which holds reference to all the object in its temporary scope)
# holding reference the traceback.


class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""

    def __repr__(self):
        return self


class ExceptionWrapper:
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        # 初始化异常类型和消息，避免存储 traceback，详见上述注意事项
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        # 格式化异常信息，包括异常类型和消息
        msg = f"Caught {self.exc_type.__name__} {self.where}.\nOriginal {self.exc_msg}"
        # 处理特定异常类型 KeyError，避免其输出不可读性
        if self.exc_type == KeyError:
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            # 某些异常类型具有非字符串的第一个参数，但显式具有消息字段
            raise self.exc_type(message=msg)
        try:
            # 尝试创建包含格式化消息的新异常对象
            exception = self.exc_type(msg)
        except TypeError:
            # 如果异常类型需要多个参数，则不尝试实例化，直接引发 RuntimeError
            raise RuntimeError(msg) from None
        raise exception


def _get_available_device_type():
    # 检查是否支持 CUDA，返回相应设备类型字符串
    if torch.cuda.is_available():
        return "cuda"
    # 检查是否存在 torch 模块中的属性 "xpu" 并且 torch.xpu 可用，类型标注忽略其属性定义检查
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        # 如果满足条件，返回字符串 "xpu"
        return "xpu"
    
    # 检查是否存在 torch 模块中的属性 "mtia" 并且 torch.mtia 可用
    if hasattr(torch, "mtia") and torch.mtia.is_available():
        # 如果满足条件，返回字符串 "mtia"
        return "mtia"
    
    # 获取私有使用1的后端名称
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    
    # 根据获取的后端名称，获取 torch 模块中对应的模块对象
    custom_device_mod = getattr(torch, custom_backend_name, None)
    
    # 检查获取的模块对象是否存在并且可用
    if custom_device_mod and custom_device_mod.is_available():
        # 如果满足条件，返回获取的后端名称
        return custom_backend_name
    
    # 如果以上条件均不满足，可以在此处添加更多可用设备类型
    
    # 如果没有匹配的可用设备类型，返回 None
    return None
# 根据给定的获取成员函数 `_get_member`，获取当前可用设备类型
def _get_device_attr(get_member):
    device_type = _get_available_device_type()  # 获取当前可用设备类型
    # 如果设备类型存在且是 CUDA，则调用 get_member 函数获取 torch.cuda 的成员
    if device_type and device_type.lower() == "cuda":
        return get_member(torch.cuda)
    # 如果设备类型存在且是 XPU，则调用 get_member 函数获取 torch.xpu 的成员
    if device_type and device_type.lower() == "xpu":
        return get_member(torch.xpu)  # type: ignore[attr-defined]
    # 如果设备类型存在且是 MTIA，则调用 get_member 函数获取 torch.mtia 的成员
    if device_type and device_type.lower() == "mtia":
        return get_member(torch.mtia)
    # 如果设备类型是 torch._C._get_privateuse1_backend_name() 返回其对应成员
    if device_type == torch._C._get_privateuse1_backend_name():
        return get_member(getattr(torch, device_type))
    # 如果没有匹配的设备类型，则返回 None
    # 添加更多可用设备类型在这里
    return None


# 获取当前设备的索引
def _get_current_device_index():
    # 调用 _get_device_attr 函数，获取当前设备的索引
    return _get_device_attr(lambda m: m.current_device())


# 获取所有设备的索引列表
def _get_all_device_indices():
    # 调用 _get_device_attr 函数，获取所有设备的索引列表
    return _get_device_attr(lambda m: list(range(m.device_count())))


# 获取指定设备 IDs 的所有设备属性
def _get_devices_properties(device_ids):
    # 使用列表推导式，对每个设备 ID 调用 _get_device_attr 函数获取设备属性
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]


# 获取当前 CUDA 设备的索引
def get_current_device_index() -> int:
    r"""检查是否有可用的 CUDA 设备，并返回当前默认 CUDA 设备的索引。
    如果没有可用的 CUDA 设备，则返回 -1。
    参数: ``None``
    """
    if torch.cuda.device_count() > 0:
        return torch.cuda.current_device()
    return -1


# 获取设备的索引，可以是 torch.device 对象、整数或 None
def _get_device_index(
    device: Any,
    optional: bool = False,
    allow_cpu: bool = False,
) -> int:
    r"""从 :attr:`device` 中获取设备索引，可以是 torch.device 对象、Python 整数或 ``None``。

    如果 :attr:`device` 是 torch.device 对象，返回其设备索引。如果 :attr:`optional` 是 ``True``，
    对于没有指定索引的设备，即 ``torch.device('xxx')``，将返回该类型的当前默认设备索引。
    如果 :attr:`allow_cpu` 是 ``True``，则接受 CPU 设备，并在这种情况下返回 ``-1``。

    如果 :attr:`device` 是 Python 整数，直接返回该整数。

    如果 :attr:`device` 是 ``None``，如果 :attr:`optional` 是 ``True``，
    则返回当前支持运行平台的默认设备，例如支持 CUDA 运行时的情况下将返回当前默认的 CUDA 设备索引。
    """
    if isinstance(device, str):
        device = torch.device(device)
    device_idx: Optional[int] = None  # 设备索引默认为 None
    if isinstance(device, torch.device):
        # 如果不允许 CPU 设备且当前设备是 CPU，则抛出 ValueError 异常
        if not allow_cpu and device.type == "cpu":
            raise ValueError(f"Expected a non cpu device, but got: {device}")
        # 如果是 CPU 设备，则设备索引为 -1；否则设备索引为设备对象的索引值
        device_idx = -1 if device.type == "cpu" else device.index
    if isinstance(device, int):
        device_idx = device  # 如果 device 是整数，则直接使用该整数作为设备索引
    # 如果未指定设备索引
    if device_idx is None:
        # 如果可选，根据当前是否在脚本模式下选择相应的 API 获取当前设备索引
        if optional:
            # 在 JIT 编译中，不支持 `lambda` 函数，因此无法使用 _get_current_device_index() 函数。
            # 可以使用 get_current_device_index() 作为 JIT 的替代 API 来获取当前设备索引。
            # 使用 torch.jit.is_scripting() 检查当前模式，并调用适当的 API。
            if torch.jit.is_scripting():
                device_idx = get_current_device_index()
            else:
                device_idx = _get_current_device_index()
        else:
            # 如果不可选，抛出 ValueError 异常，要求提供带有指定索引的 torch.device 或整数
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got:{device}"
            )
    # 返回设备索引
    return device_idx
# 返回一个实数视图的张量，如果是复数数据类型，则返回原始张量
# 需要检查是否为未初始化的参数，否则对 LazyModule 检查 is_complex 会导致错误
def _handle_complex(tensor):
    return (
        torch.view_as_real(tensor)  # 如果张量不是未初始化参数且是复数，则返回实数视图
        if not isinstance(tensor, torch.nn.UninitializedParameter)
        and tensor.is_complex()
        else tensor  # 否则返回原始张量
    )


# 返回指定数据类型的元素大小，单位为字节
def _element_size(dtype):
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"expected torch.dtype, but got {type(dtype)}")

    if dtype.is_complex:
        return torch.finfo(dtype).bits >> 2  # 复数类型的元素大小
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits >> 3  # 浮点数类型的元素大小
    elif dtype == torch.bool:
        return 1  # 布尔类型的元素大小为1字节
    else:
        return torch.iinfo(dtype).bits >> 3  # 整数类型的元素大小


# 用于描述类属性的辅助类
class _ClassPropertyDescriptor:
    def __init__(self, fget, fset=None):
        self.fget = fget

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()


# 将函数转换为类属性
def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


# 返回当前是否处于编译/跟踪状态，例如使用 torch.compile() 或 torch.export()
def is_compiling() -> bool:
    """
    Indicates whether we are tracing/compiling with torch.compile() or torch.export().

    TODO(khabinov): we should deprecate this function and use torch.compiler.is_compiling().
    """
    return torch.compiler.is_compiling()


# 由于在 C++ 中条件化某个 Python 子类更加麻烦，因此此处的代码在 Python 中实现
def _functionalize_sync(t):
    # This code lives in python instead of C++ since conditioning on a certain python subclass
    # is much more of a pain in C++.
    from torch._subclasses.functional_tensor import FunctionalTensor
    # 检查对象 t 是否为 FunctionalTensor 类型
    if isinstance(t, FunctionalTensor):
        # 如果在同步期间激活了 FunctionalTensorMode，我们不希望它拦截任何被调用的操作
        # 当我们同步内部张量时。
        
        # 为什么？
        # (1) 如果图中存在输入变化，则在 AOTAutograd 调用 _sync() 时，这些变化将被重新应用。
        # (2) _sync() 导致我们从更新的基础生成更新后的张量，
        #     这会调度一系列视图操作。
        # (3) 这些视图操作的输入是我们的内部 FunctionalTensorWrapper
        #     (因为同步是从 C++ 中调用的)，而不是 python 中的 FunctionalTensor。
        # (4) 如果激活了 python 中的 FunctionalTensorMode，它在拦截视图操作时会发出警告，
        #     因为它会看到一个作为输入的是 C++ 的 FunctionalTensorWrapper
        #     (也就是普通的 torch.Tensor)，而不是 python 的 `FunctionalTensor`。

        # 临时取消功能模式，以确保不会在同步期间触发功能化模式的拦截
        maybe_functional_mode = torch._C._unset_dispatch_mode(
            torch._C._TorchDispatchModeKey.FUNCTIONAL
        )
        try:
            # 调用 Torch 的功能化同步方法，对 t.elem 进行功能化同步
            torch._functionalize_sync(t.elem)  # type: ignore[attr-defined]
        finally:
            # 如果之前有临时取消的功能模式，将其重新设置回去
            if maybe_functional_mode is not None:
                torch._C._set_dispatch_mode(maybe_functional_mode)
    else:
        # 如果 t 不是 FunctionalTensor 类型，直接对 t 进行功能化同步
        torch._functionalize_sync(t)  # type: ignore[attr-defined]
# 使用 functools 模块的 lru_cache 装饰器，缓存最近两次调用结果，以优化性能
@functools.lru_cache(2)
# 根据设备类型获取对应的 Torch 模块
def _get_device_module(device_type: str):
    # 尝试从 torch 模块中获取设备类型对应的模块
    device_module = getattr(torch, device_type, None)
    # 如果未找到对应的模块，则抛出运行时异常
    if device_module is None:
        raise RuntimeError(
            f"Device '{device_type}' does not have a corresponding module registered as 'torch.{device_type}'."
        )
    # 返回获取到的设备模块
    return device_module


# 创建并返回一个虚拟的类型，用于表示未实现的类
def _dummy_type(name: str) -> type:
    # 返回一个错误函数，根据 is_init 参数决定返回不同的错误信息
    def get_err_fn(is_init: bool):
        def err_fn(obj, *args, **kwargs):
            # 根据 is_init 决定使用对象的类名或类别名作为错误信息的一部分
            if is_init:
                class_name = obj.__class__.__name__
            else:
                class_name = obj.__name__
            # 抛出运行时异常，指示试图实例化虚拟基类
            raise RuntimeError(f"Tried to instantiate dummy base class {class_name}")

        return err_fn

    # 创建并返回一个新类型，继承自 object 类，拥有 __init__ 和 __new__ 方法
    return type(
        name, (object,), {"__init__": get_err_fn(True), "__new__": get_err_fn(False)}
    )


# _LazySeedTracker 类，用于跟踪种子设置的顺序
class _LazySeedTracker:
    # 构造函数，初始化实例变量
    def __init__(self):
        self.manual_seed_all_cb = None  # 记录 manual_seed_all 的回调函数及其回溯信息
        self.manual_seed_cb = None      # 记录 manual_seed 的回调函数及其回溯信息
        self.call_order = []            # 用于存储调用的顺序列表

    # 设置 manual_seed_all 回调函数及其回溯信息，并更新调用顺序
    def queue_seed_all(self, cb, traceback):
        self.manual_seed_all_cb = (cb, traceback)
        self.call_order = [self.manual_seed_cb, self.manual_seed_all_cb]

    # 设置 manual_seed 回调函数及其回溯信息，并更新调用顺序
    def queue_seed(self, cb, traceback):
        self.manual_seed_cb = (cb, traceback)
        self.call_order = [self.manual_seed_all_cb, self.manual_seed_cb]

    # 返回当前存储的调用顺序列表
    def get_calls(self) -> List:
        return self.call_order


# 创建一个日志记录器对象，用于记录日志消息
logger = logging.getLogger(__name__)

# 创建 ParamSpec 类的实例 P
P = ParamSpec("P")


# CallbackRegistry 类，用于管理回调函数的注册和触发
class CallbackRegistry(Generic[P]):
    # 构造函数，初始化实例变量
    def __init__(self, name: str):
        self.name = name                          # 回调注册表的名称
        self.callback_list: List[Callable[P, None]] = []  # 存储回调函数的列表

    # 向回调函数列表中添加新的回调函数
    def add_callback(self, cb: Callable[P, None]) -> None:
        self.callback_list.append(cb)

    # 触发所有注册的回调函数，传递指定的参数和关键字参数
    def fire_callbacks(self, *args: P.args, **kwargs: P.kwargs) -> None:
        for cb in self.callback_list:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                # 记录异常信息到日志中
                logger.exception(
                    "Exception in callback for %s registered with gpu trace", self.name
                )


# IMPORT_MAPPING 和 NAME_MAPPING 是从 https://github.com/python/cpython/blob/main/Lib/_compat_pickle.py
# 适配而来，用于在 weights_only Unpickler 中使用。

# IMPORT_MAPPING 字典，用于将某些模块名映射到其兼容的 Python 3 名称
IMPORT_MAPPING = {
    "__builtin__": "builtins",
    "copy_reg": "copyreg",
    "Queue": "queue",
    "repr": "reprlib",
    "_abcoll": "collections.abc",
    # 非相互映射的条目
    "UserDict": "collections",
    "UserList": "collections",
    "UserString": "collections",
    "whichdb": "dbm",
    "StringIO": "io",
    "cStringIO": "io",
}
# 定义一个名称映射字典，用于在导入模块时进行名称转换
NAME_MAPPING = {
    # 将 "__builtin__.xrange" 映射到 "builtins.range"
    ("__builtin__", "xrange"): ("builtins", "range"),
    # 将 "__builtin__.reduce" 映射到 "functools.reduce"
    ("__builtin__", "reduce"): ("functools", "reduce"),
    # 将 "__builtin__.intern" 映射到 "sys.intern"
    ("__builtin__", "intern"): ("sys", "intern"),
    # 将 "__builtin__.unichr" 映射到 "builtins.chr"
    ("__builtin__", "unichr"): ("builtins", "chr"),
    # 将 "__builtin__.unicode" 映射到 "builtins.str"
    ("__builtin__", "unicode"): ("builtins", "str"),
    # 将 "__builtin__.long" 映射到 "builtins.int"
    ("__builtin__", "long"): ("builtins", "int"),
    # 将 "itertools.izip" 映射到 "builtins.zip"
    ("itertools", "izip"): ("builtins", "zip"),
    # 将 "itertools.imap" 映射到 "builtins.map"
    ("itertools", "imap"): ("builtins", "map"),
    # 将 "itertools.ifilter" 映射到 "builtins.filter"
    ("itertools", "ifilter"): ("builtins", "filter"),
    # 将 "itertools.ifilterfalse" 映射到 "itertools.filterfalse"
    ("itertools", "ifilterfalse"): ("itertools", "filterfalse"),
    # 将 "itertools.izip_longest" 映射到 "itertools.zip_longest"
    ("itertools", "izip_longest"): ("itertools", "zip_longest"),
    # 将 "UserDict.IterableUserDict" 映射到 "collections.UserDict"
    ("UserDict", "IterableUserDict"): ("collections", "UserDict"),
    # 将 "UserList.UserList" 映射到 "collections.UserList"
    ("UserList", "UserList"): ("collections", "UserList"),
    # 将 "UserString.UserString" 映射到 "collections.UserString"
    ("UserString", "UserString"): ("collections", "UserString"),
    # 将 "__builtin__.basestring" 映射到 "builtins.str"
    ("__builtin__", "basestring"): ("builtins", "str"),
    # 将 "exceptions.StandardError" 映射到 "builtins.Exception"
    ("exceptions", "StandardError"): ("builtins", "Exception"),
    # 将 "UserDict.UserDict" 映射到 "collections.UserDict"
    ("UserDict", "UserDict"): ("collections", "UserDict"),
}
```