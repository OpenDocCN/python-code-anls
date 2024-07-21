# `.\pytorch\torch\distributed\_tensor\__init__.py`

```py
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 引入必要的类型注解
from typing import Optional, Sequence

# 引入torch及其分布式张量操作模块
import torch
import torch.distributed._tensor.ops
import torch.distributed._tensor.random as random

# 引入计算本地形状的工具函数
from torch.distributed._tensor._utils import compute_local_shape

# 引入分布式张量相关的公共API
from torch.distributed._tensor.api import distribute_module, distribute_tensor, DTensor

# 引入工具函数，用于规范化到torch的大小
from torch.distributed._tensor.ops.utils import normalize_to_torch_size

# 引入分布式张量的放置类型
from torch.distributed._tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

# 引入设备网格相关的模块及其初始化函数
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh, init_device_mesh

# 引入优化器相关的foreach支持函数
from torch.optim.optimizer import (
    _foreach_supported_types as _optim_foreach_supported_types,
)

# 引入工具函数，用于foreach支持
from torch.utils._foreach_utils import (
    _foreach_supported_types as _util_foreach_supported_types,
)

# 定义dtensor包中所有的公共API
__all__ = [
    "DTensor",
    "DeviceMesh",
    "distribute_tensor",
    "distribute_module",
    "init_device_mesh,",
    "Shard",
    "Replicate",
    "Partial",
]

# 将DTensor添加到优化器foreach支持的类型列表中，以及工具函数foreach支持的类型列表中
if DTensor not in _optim_foreach_supported_types:
    _optim_foreach_supported_types.append(DTensor)

if DTensor not in _util_foreach_supported_types:
    _util_foreach_supported_types.append(DTensor)

# 定义私有函数_dtensor_init_helper，用于初始化分布式张量
def _dtensor_init_helper(
    init_op,
    size: torch.Size,
    device_mesh=None,
    placements=None,
    **kwargs,
) -> DTensor:
    from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta

    # 如果device_mesh为None，则使用当前mesh资源中的设备网格
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    kwargs["device"] = device_mesh.device_type

    # 如果placements未指定，默认使用Replicate来放置张量
    placements = placements or tuple(Replicate() for _ in range(device_mesh.ndim))

    # 检查device_mesh的维度是否与placements的长度匹配
    assert device_mesh.ndim == len(
        placements
    ), "mesh dimension does not match the length of placements"

    # 断言layout为torch.strided，不支持其他layout
    assert kwargs["layout"] == torch.strided, "layout value not supported!"
    
    # 使用torch._prims_common.make_contiguous_strides_for获取torch的连续步幅
    torch_stride = torch._prims_common.make_contiguous_strides_for(size)

    # 计算本地张量的形状
    local_shape = compute_local_shape(size, device_mesh, placements)

    # 根据init_op进行本地张量的初始化
    if init_op == torch.full:
        fill_value = kwargs.pop("fill_value", 0)
        local_tensor = init_op(local_shape, fill_value, **kwargs)
    # 如果初始化操作是 torch.rand 或者 torch.randn
    elif init_op == torch.rand or init_op == torch.randn:
        # 除了 `shape` 外，这个张量元数据没有被使用
        dtype = kwargs.get("dtype", torch.get_default_dtype())

        # 使用给定的 size、空的 torch_stride 和 dtype 创建张量元数据
        tensor_meta = TensorMeta(size, (0,), dtype)
        
        # 创建 DTensorSpec 对象，指定设备网格和放置方案，并传入张量元数据
        spec = DTensorSpec(device_mesh, placements, tensor_meta=tensor_meta)

        # 如果设备网格支持随机数生成，并且 RNG 跟踪器尚未初始化
        if random.is_rng_supported_mesh(device_mesh) and not random._rng_tracker:
            # 初始化一个基于偏移的 RNG 跟踪器
            random._rng_tracker = random.OffsetBasedRNGTracker()

        # 断言 RNG 跟踪器不为 None
        assert random._rng_tracker is not None
        
        # 在 RNG 跟踪器的分布区域内执行初始化操作，得到本地张量
        with random._rng_tracker._distribute_region(spec):
            local_tensor = init_op(local_shape, **kwargs)
    else:
        # 使用给定的 local_shape 和 kwargs 执行初始化操作，得到本地张量
        local_tensor = init_op(local_shape, **kwargs)

    # 创建 DTensorSpec 对象，指定设备网格、放置方案和张量元数据（包括 size、torch_stride 和本地张量的 dtype）
    spec = DTensorSpec(
        device_mesh,
        tuple(placements),
        tensor_meta=TensorMeta(
            size,
            torch_stride,
            local_tensor.dtype,
        ),
    )

    # 返回一个 DTensor 对象，包括本地张量、规格信息和是否需要梯度的设置
    return DTensor(
        local_tensor,
        spec,
        requires_grad=kwargs["requires_grad"],
    )
# 返回一个由标量值1填充的`DTensor`对象，其形状由变量参数`size`定义
def ones(
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 1, with the shape defined
    by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    # 标准化参数 size 到 Torch 的尺寸表示
    torch_size = normalize_to_torch_size(size)

    # 调用 _dtensor_init_helper 函数，返回初始化后的 DTensor 对象
    return _dtensor_init_helper(
        torch.ones,  # 使用 torch.ones 函数创建填充值为1的张量
        torch_size,  # 传入标准化后的尺寸参数
        dtype=dtype,  # 可选参数：数据类型
        layout=layout,  # 可选参数：张量布局
        requires_grad=requires_grad,  # 可选参数：是否需要梯度信息
        device_mesh=device_mesh,  # 可选参数：设备网格信息
        placements=placements,  # 可选参数：放置策略信息
    )


# 返回一个由未初始化数据填充的`DTensor`对象，其形状由变量参数`size`定义
def empty(
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with uninitialized data. The shape of the :class:`DTensor`
    is defined by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: empty(1,2,3..) or empty([1,2,3..]) or empty((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).\
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    """
    # 标准化参数 size 到 Torch 的尺寸表示
    torch_size = normalize_to_torch_size(size)

    # 调用 _dtensor_init_helper 函数，返回未初始化的 DTensor 对象
    return _dtensor_init_helper(
        torch.empty,  # 使用 torch.empty 函数创建未初始化的张量
        torch_size,   # 传入标准化后的尺寸参数
        dtype=dtype,  # 可选参数：数据类型
        layout=layout,  # 可选参数：张量布局
        requires_grad=requires_grad,  # 可选参数：是否需要梯度信息
        device_mesh=device_mesh,  # 可选参数：设备网格信息
        placements=placements,  # 可选参数：放置策略信息
    )
    # 返回一个由 `DTensor` 对象组成的列表，每个对象对应一个处理单元的结果
    torch_size = normalize_to_torch_size(size)
    # 将输入的 size 规范化为 PyTorch 的 size 表示形式
    
    # 调用 _dtensor_init_helper 函数来初始化 DTensor 对象
    return _dtensor_init_helper(
        torch.empty,  # 使用 torch.empty 函数创建一个空的张量
        torch_size,   # 使用规范化后的 size 参数作为张量的形状
        dtype=dtype,  # 指定张量的数据类型
        layout=layout,  # 指定张量的布局方式
        requires_grad=requires_grad,  # 指定是否需要计算梯度
        device_mesh=device_mesh,  # 指定张量所在的设备网格
        placements=placements,  # 指定张量在多个处理单元上的分布
    )
# 返回一个以指定值填充的 `DTensor` 对象
def full(
    size,
    fill_value,
    *,
    dtype: Optional[torch.dtype] = None,  # 可选参数：返回 `DTensor` 的数据类型，默认为全局设定值
    layout: torch.layout = torch.strided,  # 可选参数：返回 `DTensor` 的布局，默认为 strided
    requires_grad: bool = False,  # 可选参数：是否记录返回的 `DTensor` 的操作以支持自动微分，默认为 False
    device_mesh: Optional[DeviceMesh] = None,  # 可选参数：设备网格对象，包含排名的网格信息
    placements: Optional[Sequence[Placement]] = None,  # 可选参数：一系列放置选项，如 Shard 或 Replicate
) -> DTensor:  # 返回类型声明为 `DTensor`

    """
    Returns a :class:`DTensor` filled with ``fill_value``. The scalar value type should match
        ``device_mesh.device_type``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))
        fill_value(Scalar): the value to fill the output tensor with.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """

    torch_size = normalize_to_torch_size(size)  # 调用函数将输入的 size 规范化为 PyTorch 的 size

    return _dtensor_init_helper(
        torch.full,  # 调用 PyTorch 的 full 函数来创建填充的 `DTensor`
        torch_size,  # 传递规范化后的 size 给 full 函数
        fill_value=fill_value,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


# 返回一个以均匀分布的随机数填充的 `DTensor` 对象
def rand(
    *size,  # 可变长度的位置参数，定义了输出 `DTensor` 的形状
    requires_grad: bool = False,  # 可选参数：是否记录返回的 `DTensor` 的操作以支持自动微分，默认为 False
    dtype: Optional[torch.dtype] = None,  # 可选参数：返回 `DTensor` 的数据类型，默认为全局设定值
    layout: torch.layout = torch.strided,  # 可选参数：返回 `DTensor` 的布局，默认为 strided
    device_mesh: Optional[DeviceMesh] = None,  # 可选参数：设备网格对象，包含排名的网格信息
    placements: Optional[Sequence[Placement]] = None,  # 可选参数：一系列放置选项，如 Shard 或 Replicate
) -> DTensor:  # 返回类型声明为 `DTensor`

    """
    Returns a :class:`DTensor` filled with random numbers from a uniform distribution
        on the interval ``[0, 1)``. The shape of the tensor is defined by the variable
        argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    Keyword args:
        dtype (:class:`torch.dtype`, optional): 返回的 :class:`DTensor` 的数据类型，如果为 ``None``，则使用全局默认值（参见 :func:`torch.set_default_dtype`）。
        layout (:class:`torch.layout`, optional): 返回的 DTensor 的布局类型。默认为 ``torch.strided``。
        requires_grad (bool, optional): 如果希望在返回的 :class:`DTensor` 上记录 autograd 操作，则设为 ``True``。默认为 ``False``。
        device_mesh: :class:`DeviceMesh` 类型，包含各个 rank 的网格信息。
        placements: 一系列 :class:`Placement` 类型对象，如 ``Shard``、``Replicate``。

    Returns:
        每个 rank 上的 :class:`DTensor` 对象
# 返回一个填充有从标准正态分布中随机采样得到的数值的 DTensor 对象
def randn(
    *size,
    requires_grad: bool = False,  # 是否需要计算梯度，默认为 False
    dtype: Optional[torch.dtype] = None,  # 返回的 DTensor 的数据类型，默认为全局默认类型
    layout: torch.layout = torch.strided,  # 返回的 DTensor 的布局方式，默认为 torch.strided
    device_mesh: Optional[DeviceMesh] = None,  # 包含分布在不同计算节点上的网格信息
    placements: Optional[Sequence[Placement]] = None,  # 分布在不同计算节点上的放置方式
) -> DTensor:
    """
    返回一个由标准正态分布中的随机数填充的 DTensor 对象。张量的形状由变长参数 size 定义。

    Args:
        size (int...): 定义输出 DTensor 的形状的整数序列。可以是可变数量的参数或类似列表或元组的集合。
            例如：ones(1,2,3..) 或 ones([1,2,3..]) 或 ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): 返回的 DTensor 的期望数据类型。默认情况下使用全局默认类型。
        layout (:class:`torch.layout`, optional): 返回的 DTensor 的期望布局方式。默认为 torch.strided。
        requires_grad (bool, optional): 是否应记录对返回的 DTensor 的操作以进行自动求导。默认为 False。
        device_mesh: :class:`DeviceMesh` 类型，包含各个计算节点的网格信息。
        placements: :class:`Placement` 类型的序列，描述在不同计算节点上的放置方式，如 "Shard"、"Replicate"

    Returns:
        A :class:`DTensor` object on each rank  # 返回每个计算节点上的一个 DTensor 对象
    """
    # 将输入的 size 规范化为 torch_size
    torch_size = normalize_to_torch_size(size)

    # 调用辅助函数 _dtensor_init_helper，返回初始化后的 DTensor 对象
    return _dtensor_init_helper(
        torch.randn,  # 使用 torch.randn 函数初始化
        torch_size,  # 使用规范化后的 size
        dtype=dtype,  # 指定数据类型
        layout=layout,  # 指定布局方式
        requires_grad=requires_grad,  # 指定是否需要计算梯度
        device_mesh=device_mesh,  # 指定计算节点的网格信息
        placements=placements,  # 指定节点的放置方式
    )


# 返回一个填充有标量值 0 的 DTensor 对象
def zeros(
    *size,
    requires_grad: bool = False,  # 是否需要计算梯度，默认为 False
    dtype: Optional[torch.dtype] = None,  # 返回的 DTensor 的数据类型，默认为全局默认类型
    layout: torch.layout = torch.strided,  # 返回的 DTensor 的布局方式，默认为 torch.strided
    device_mesh: Optional[DeviceMesh] = None,  # 包含分布在不同计算节点上的网格信息
    placements: Optional[Sequence[Placement]] = None,  # 分布在不同计算节点上的放置方式
) -> DTensor:
    """
    返回一个由标量值 0 填充的 DTensor 对象。

    Args:
        size (int...): 定义输出 DTensor 的形状的整数序列。可以是可变数量的参数或类似列表或元组的集合。
            例如：zeros(1,2,3..) 或 zeros([1,2,3..]) 或 zeros((1,2,3..))

    Keyword args:
        requires_grad (bool, optional): 是否应记录对返回的 DTensor 的操作以进行自动求导。默认为 False。
        dtype (:class:`torch.dtype`, optional): 返回的 DTensor 的期望数据类型。默认情况下使用全局默认类型。
        layout (:class:`torch.layout`, optional): 返回的 DTensor 的期望布局方式。默认为 torch.strided。
        device_mesh: :class:`DeviceMesh` 类型，包含各个计算节点的网格信息。
        placements: :class:`Placement` 类型的序列，描述在不同计算节点上的放置方式，如 "Shard"、"Replicate"

    Returns:
        A :class:`DTensor` object on each rank  # 返回每个计算节点上的一个 DTensor 对象
    """
    Returns:
        返回一个在每个rank上的DTensor对象
    """
    # 将输入的size参数规范化为torch的size表示形式
    torch_size = normalize_to_torch_size(size)

    # 调用_dtensor_init_helper函数来创建一个DTensor对象并返回
    return _dtensor_init_helper(
        torch.zeros,            # 使用torch.zeros函数来初始化DTensor对象
        torch_size,             # 使用规范化后的torch_size作为初始化函数的参数
        dtype=dtype,            # 指定数据类型
        layout=layout,          # 指定布局
        requires_grad=requires_grad,    # 指定是否需要梯度
        device_mesh=device_mesh,        # 指定设备网格
        placements=placements,          # 指定放置策略
    )
```