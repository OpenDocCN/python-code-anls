# `.\pytorch\torch\distributed\_tensor\_redistribute.py`

```py
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入必要的库和模块
from functools import lru_cache
from typing import cast, Dict, List, NamedTuple, Tuple

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed._tensor.api as dtensor
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
    TensorMeta,
)

# 定义命名元组 _TransformInfo，表示转换信息
class _TransformInfo(NamedTuple):
    mesh_dim: int
    src_dst_placements: Tuple[Placement, Placement]
    # 逻辑形状在此网格维度上的描述
    logical_shape: List[int]

# 定义函数 _replicate_then_shard，用于重新排序 _TransformInfo 列表
def _replicate_then_shard(val: _TransformInfo) -> int:
    """
    This is a helper function to allow reordering _TransformInfo list. The high level
    idea is that we want to reorder the sharding redistributions so that the DTensor
    redistribution is consistent with its full tensor. This is built on top of two simple
    assumptions:
    1. Replication happens from inner to outer dimension. i.e. Shard -> Replicate
    2. Sharding happens from outer to inner dimension, i.e. Replicate -> Shard

    So we always put the replication first and put sharding later.
    """
    mesh_dim = val.mesh_dim
    src, dst = val.src_dst_placements
    if (dst.is_replicate() or dst.is_partial()) and src.is_shard():
        return -mesh_dim
    elif (src.is_replicate() or src.is_partial()) and dst.is_shard():
        return mesh_dim
    else:
        return 0

# 使用 lru_cache 装饰器定义函数 _gen_transform_infos，生成从源位置到目标位置的转换信息列表
@lru_cache(maxsize=None)
def _gen_transform_infos(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
) -> List[_TransformInfo]:
    """
    Generate the transform infos from the source placements to the target placements.

    To transform from source to target placement it might have multiple steps, i.e. it
    might decompose Si -> Sj into Si -> R -> Sj.
    This would detects if there're mis-aligned shardings between src/dst placements.
    i.e. (Shard(0), Shard(0)) -> (Replicate(), Shard(0)), in this case Shard(0) -> Shard(0)
    for mesh dimension 1 actually needs reshard, because in the first case it's a sub-sharding
    of an already tensor dimension 0, and in the second case, it's the first sharding on tensor
    dimension 0.
    """
    # 初始化源维度计数器和目标维度计数器
    src_dim_counts: Dict[int, int] = {}
    dst_dim_counts: Dict[int, int] = {}
    # 初始化转换信息列表
    transform_infos: List[_TransformInfo] = []

    # 获取源和目标的位置规格
    src_placements = src_spec.placements
    dst_placements = dst_spec.placements
    # 获取设备网格对象
    device_mesh = src_spec.device_mesh
    # 获取当前设备的坐标
    my_coordinate = device_mesh.get_coordinate()
    # 断言当前设备坐标不为空
    assert my_coordinate is not None

    # 逻辑形状记录网格维度上的逻辑张量形状
    # 这对确保不均匀的分片得到正确的输出形状很有用
    initial_logical_shape = list(src_spec.shape)
    mesh_dims_to_logical_shape = [initial_logical_shape]
    # 记录网格维度数
    mesh_ndim = len(src_placements)
    # 遍历枚举 src_placements 和 dst_placements 中的元素对
    for i, (src, dst) in enumerate(zip(src_placements, dst_placements)):
        # 检测是否存在错位的分片，并构建逻辑形状
        current_logical_shape = mesh_dims_to_logical_shape[i]
        
        # 如果 src 是 Shard 类型
        if isinstance(src, Shard):
            # 增加 src.dim 维度的分片计数
            src_dim_counts[src.dim] = src_dim_counts.get(src.dim, 0) + 1
            
            # 如果当前索引 i 小于 mesh_ndim - 1
            if i < mesh_ndim - 1:
                # 计算并保存此分片的逻辑形状
                mesh_dim_size = device_mesh.size(mesh_dim=i)
                local_shard_size, _ = src._local_shard_size_on_dim(
                    current_logical_shape[src.dim],
                    mesh_dim_size,
                    my_coordinate[i],
                )
                new_logical_shape = list(current_logical_shape)
                new_logical_shape[src.dim] = local_shard_size
                mesh_dims_to_logical_shape.append(new_logical_shape)
        else:
            # 否则直接添加当前的逻辑形状到 mesh_dims_to_logical_shape
            mesh_dims_to_logical_shape.append(current_logical_shape)
        
        # 如果 dst 是 Shard 类型
        if isinstance(dst, Shard):
            # 增加 dst.dim 维度的分片计数
            dst_dim_counts[dst.dim] = dst_dim_counts.get(dst.dim, 0) + 1
        
        # 如果同时 src 和 dst 都是 Shard 类型，且满足条件 mesh_ndim > 1 或者分片维度计数不同
        if (
            isinstance(src, Shard)
            and isinstance(dst, Shard)
            and (mesh_ndim > 1 or src_dim_counts[src.dim] != dst_dim_counts[dst.dim])
        ):
            # 在 mesh_dim=i 的情况下，将 Shard(i) -> Shard(j) 分解为 Shard(i) -> Replicate() -> Shard(j)
            transform_infos.append(
                _TransformInfo(
                    mesh_dim=i,
                    src_dst_placements=(src, Replicate()),
                    logical_shape=mesh_dims_to_logical_shape[i],
                )
            )
            transform_infos.append(
                _TransformInfo(
                    mesh_dim=i,
                    src_dst_placements=(Replicate(), dst),
                    logical_shape=mesh_dims_to_logical_shape[i],
                )
            )
        else:
            # 否则，将当前的 src 和 dst 对及其逻辑形状添加到 transform_infos 中
            transform_infos.append(
                _TransformInfo(
                    mesh_dim=i,
                    src_dst_placements=(src, dst),
                    logical_shape=mesh_dims_to_logical_shape[i],
                )
            )

    # 按照 _replicate_then_shard 函数定义的规则对 transform_infos 排序
    transform_infos.sort(key=_replicate_then_shard)
    # 返回排序后的 transform_infos 列表
    return transform_infos
def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = False,
    is_backward: bool = False,
) -> torch.Tensor:
    """
    This function redistributes a local tensor from the current DTensorSpec to
    the target DTensorSpec, potentially involving collective operations to 
    adjust the tensor's placement on the device mesh.
    """

    if current_spec.mesh != target_spec.mesh:
        # 如果当前和目标的设备网格不同，抛出未实现错误
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    new_local_tensor = None
    device_mesh = current_spec.mesh

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # 如果当前进程不在网格内，则直接返回原始的本地张量
        # 这种情况通常是本地张量为空的情况
        return local_tensor

    # 生成从当前规格到目标规格的转换信息
    transform_infos = _gen_transform_infos(current_spec, target_spec)

    # 确保新的本地张量不为空，否则抛出分发失败的断言错误
    assert new_local_tensor is not None, "redistribute failed!"

    # 如果不是异步操作，并且新的本地张量是异步集体张量，则等待其完成
    if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
        new_local_tensor = new_local_tensor.wait()

    return new_local_tensor


class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: "dtensor.DTensor",
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        async_op: bool = False,
    ):
        current_spec = input._spec
        ctx.current_spec = current_spec
        ctx.async_op = async_op

        if current_spec.placements != placements:
            # 如果当前规格的放置不等于目标放置，创建一个新的目标规格
            target_spec = DTensorSpec(
                device_mesh, placements, tensor_meta=input._spec.tensor_meta
            )

            local_tensor = input._local_tensor
            # 调用分发本地张量函数，以获取重新分布的输出
            output = redistribute_local_tensor(
                local_tensor, current_spec, target_spec, async_op=async_op
            )
        else:
            # 如果当前规格的放置等于目标放置，直接使用原始的本地张量
            output = input._local_tensor
            target_spec = current_spec

        # 返回一个带有目标规格的新的 DTensor 对象
        return dtensor.DTensor(
            output,
            target_spec,
            requires_grad=input.requires_grad,
        )
    # 定义 backward 方法，用于计算反向传播梯度
    def backward(ctx, grad_output: "dtensor.DTensor"):  # type: ignore[override]
        # 获取当前计算图节点的规格信息
        previous_spec = ctx.current_spec
        # 获取传入梯度张量的规格信息
        current_spec = grad_output._spec
        # 获取异步操作标志
        async_op = ctx.async_op

        # 获取传入梯度张量的本地数据
        local_tensor = grad_output._local_tensor
        # 对本地数据进行重分布操作，以匹配前一节点的规格
        output = redistribute_local_tensor(
            local_tensor,
            current_spec,
            previous_spec,
            async_op=async_op,
            is_backward=True,
        )

        # 将目标位置规格进行标准化处理，如果是部分规格则复制以替代
        normalized_placements: List[Placement] = []
        for previous_placement in previous_spec.placements:
            if previous_placement.is_partial():
                # 如果前一节点的位置是部分的，则选择复制以替代部分规格
                normalized_placements.append(Replicate())
            else:
                normalized_placements.append(previous_placement)

        # 根据前一节点的设备网格、标准化后的位置规格，以及梯度张量的元信息，构建新的规格对象
        spec = DTensorSpec(
            previous_spec.device_mesh,
            tuple(normalized_placements),
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=grad_output.dtype,
            ),
        )

        # 基于输出数据、新的规格和梯度张量的梯度要求，创建新的分布式张量对象
        output_dtensor = dtensor.DTensor(
            output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

        # 返回计算得到的结果，包括输出张量和其它空占位符
        return (
            output_dtensor,
            None,
            None,
            None,
        )
```