# `.\pytorch\torch\distributed\_tensor\experimental\local_map.py`

```py
# 设置类型提示允许未标注的定义
# 版权声明: Meta Platforms, Inc. 及其关联公司
from typing import Callable, Optional, Sequence, Tuple, Union  # 导入需要的类型提示模块

import torch  # 导入 PyTorch 库
from torch.distributed._functional_collectives import AsyncCollectiveTensor  # 导入异步集合张量模块
from torch.distributed._tensor import DeviceMesh, DTensor  # 导入设备网格和分布式张量模块
from torch.distributed._tensor.placement_types import Placement  # 导入放置类型模块

try:
    from torch.utils import _cxx_pytree as pytree  # 尝试导入 C++ 实现的 pytree 模块
except ImportError:
    from torch.utils import _pytree as pytree  # 如果导入失败，导入 Python 实现的 pytree 模块，并忽略重新定义的警告

PlacementType = Optional[Sequence[Placement]]  # 定义可选的放置类型序列
InputPlacements = Optional[Tuple[PlacementType, ...]]  # 定义可选的输入放置类型元组
OutputPlacements = Union[PlacementType, Tuple[PlacementType, ...]]  # 定义输出放置类型的联合类型

def local_map(
    func: Callable,  # 函数参数，接受可调用对象
    out_placements: OutputPlacements,  # 输出放置类型参数，可以是单个或多个放置类型
    in_placements: Optional[InputPlacements] = None,  # 可选的输入放置类型参数，默认为 None
    device_mesh: Optional[DeviceMesh] = None,  # 可选的设备网格参数，默认为 None
    *,
    redistribute_inputs: bool = False,  # 关键字参数，指示是否重新分配输入，默认为 False
):
    """
    ``local_map`` 是一个实验性的 API，允许用户在 :class:`DTensors` 上应用在 :class:`~torch.Tensors` 上编写的函数。
    """
    Args:
        func (Callable): 用于在每个本地分片上应用的函数，其参数是 :class:`DTensor`。
        out_placements (Union[`PlacementType`, Tuple[`PlacementType`, ...]]):
            :class:`DTensor`在`func`的展平输出中的期望放置方式。如果展平的`output`是单个值，
            则`out_placements`应为`PlacementType`类型。如果展平的`output`有多个值，则`out_placements`
            应是一个与展平的`output`长度相同的`PlacementType`元组，与其一一对应。
            对于 :class:`Tensor` 输出，我们使用 `PlacementType` 作为其放置方式（一个 `Tuple[Placement]` 值）。
            对于非 :class:`Tensor` 输出，`PlacementType` 应为 `None`。
            需要注意的是，即使没有传入 :class:`DTensor` 参数，结果函数也应忽略期望的放置方式，
            因为此时应用并非针对 :class:`DTensors`。
        in_placements (Tuple[`PlacementType`, ...], optional):
            `func`的展平输入中 :class:`DTensor` 的必需放置方式。如果指定了 `in_placements`，
            `local_map` 将检查每个 :class:`DTensor` 参数的放置方式是否与所需放置方式相同。
            如果放置方式不同且 `redistribute_inputs` 为 `False`，将引发异常。
            如果 `redistribute_inputs` 为 `True`，将首先将参数重新分片到所需的分片放置方式，
            然后将其本地张量传递给 `func`。唯一的例外情况是当所需放置方式不为 `None` 且参数为 :class:`torch.Tensor` 时，
            放置方式检查将被跳过，并直接将参数传递给 `func`。
            如果 `in_placements` 为 `None`，将不执行放置方式检查。默认值：`None`。
        device_mesh (:class:`DeviceMesh`, optional):
            所有 :class:`DTensor` 放置的设备网格。如果未指定，则将从输入 :class:`DTensor` 推断。
            `local_map` 要求每个 :class:`DTensor` 放置在同一个设备网格上。默认值：`None`。
        redistribute_inputs (bool, optional):
            布尔值，指示是否在输入 :class:`DTensor` 的放置方式与所需输入放置方式不同时重新分片。
            如果此值为 `False` 并且某些 :class:`DTensor` 输入具有不同的放置方式，将引发异常。默认值：`False`。
    # 返回一个 `Callable`，该函数将 `func` 应用于输入的每个本地分片的 :class:`DTensor`，
    # 并返回一个由 `func` 的返回值构造的 :class:`DTensor`。
    
    # 如果输入的 :class:`DTensor` 不在同一设备网格上，或者它们位于与传入的 `device_mesh`
    # 参数不同的设备网格上，则会引发 AssertionError。
    
    # 如果任何非 :class:`DTensor` 的输出，我们要求其在 `out_placements` 中对应的输出位置是 `None`。
    # 如果不是这种情况，将会引发 AssertionError。
    
    # 如果 `redistribute_inputs=False` 但是输入的 :class:`DTensor` 需要根据 `in_placements`
    # 进行重新分布，则会引发 ValueError。
    
    # 示例：
    # >>> # xdoctest: +SKIP("distributed")
    # >>> def mm_allreduce_forward(device_mesh, W, X):
    # >>>     partial_sum_tensor = torch.mm(W, X)
    # >>>     reduced_tensor = funcol.all_reduce(partial_sum_tensor, "sum", device_mesh)
    # >>>     return reduced_tensor
    # >>>
    # >>> W = torch.randn(12, 8, requires_grad=False)
    # >>> X = torch.randn(8, 16, requires_grad=False)
    # >>> Y = torch.mm(W, X)
    # >>> row_wise = [Shard(0)]  # 在 1 维网格上进行行切片放置
    # >>> col_wise = [Shard(1)]  # 在 1 维网格上进行列切片放置
    # >>>
    # >>> # local_mm_allreduce_forward 是使用 DTensor/Tensor 转换包装的函数
    # >>> local_mm_allreduce_forward = local_map(
    # >>>     mm_allreduce_forward,
    # >>>     out_placements=[Replicate()],
    # >>>     in_placements=[col_wise, row_wise],
    # >>>     device_mesh=device_mesh,
    # >>> )
    # >>>
    # >>> W_dt = distribute_tensor(W, device_mesh, (col_wise))  # 对 W 张量进行列切片分布
    # >>> X_dt = distribute_tensor(X, device_mesh, (row_wise))  # 对 X 张量进行行切片分布
    # >>> Y_dt = local_mm_allreduce_forward(device_mesh, W_dt, X_dt)  # 将 local_mm_allreduce_forward 应用于 DTensors
    
    # 注意：此 API 目前处于实验阶段，可能会发生变化。
```