# `.\pytorch\torch\distributed\_tensor\ops\embedding_ops.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

# 导入必要的模块和类
from dataclasses import dataclass, field
from typing import cast, List, Optional

# 导入 PyTorch 库
import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor._op_schema import OpSchema, OpStrategy, StrategyType
from torch.distributed._tensor.ops.utils import (
    expand_to_full_mesh_op_strategy,
    register_op_strategy,
)
from torch.distributed._tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh

# 使用 PyTorch 的原生操作库
aten = torch.ops.aten

# 定义数据类 MaskBuffer，用于管理掩码数据
@dataclass
class MaskBuffer:
    data: Optional[torch.Tensor] = None

    # 将掩码材料化，即将掩码数据存储在实例中
    def materialize_mask(self, mask):
        if self.data is not None:
            raise RuntimeError("MaskBuffer has already been materialized")
        self.data = mask

    # 释放掩码，即清除存储的掩码数据
    def release_mask(self):
        # TODO: evaluate if we need to release the mask buffer or the buffer
        # can just have the same lifetime as the Partial placement
        if self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")
        self.data = None

    # 应用掩码到给定的张量上
    def apply_mask(self, tensor):
        if self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")

        # NOTE: _MaskPartial is being used by the embedding op and the gather op.
        # For gather, the mask has the same dimension as the output tensor, whereas
        # the output of the embedding op has an additional dimension compare to the input,
        # hence the output masking logic below having two different cases.
        # 根据张量的维度和掩码的维度进行适当的掩码操作
        if tensor.ndim == self.data.ndim:
            tensor[self.data] = 0.0
        else:
            tensor[self.data, :] = 0.0

# 定义冻结的数据类 _MaskPartial，继承自 Partial
@dataclass(frozen=True)
class _MaskPartial(Partial):
    """
    A partial mask placement devised for rowwise sharded embedding op, where we need
    to mask and adjust the indices to the local embedding shard, embedding masking
    is a special type of the Partial placement

    NOTE: the lifecycle of this MaskPartial placement follows the corresponding DTensor
    lifecycle, i.e. the indices_mask would only be alive during the lifetime of the DTensor.
    """

    logical_dim_size: int = -1
    mask_buffer: MaskBuffer = field(default_factory=MaskBuffer)

    # 分区值的方法，用于在设备网格上划分张量的值
    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # 覆盖父类逻辑以执行嵌入的部分遮罩
        num_chunks = mesh.size(mesh_dim)
        # 获取本地分片大小和嵌入维度上的本地偏移量
        local_shard_size, local_offset_on_dim = Shard._local_shard_size_on_dim(
            self.logical_dim_size,
            num_chunks,
            mesh.get_local_rank(mesh_dim),
            return_offset=True,
        )
        # 构建输入遮罩并保存到当前部分放置的位置
        # 这样嵌入操作的输出可以重复使用相同的部分放置保存的遮罩来执行遮罩和减少操作
        mask = (tensor < local_offset_on_dim) | (
            tensor >= local_offset_on_dim + local_shard_size
        )
        # 对输入张量应用遮罩
        masked_tensor = tensor.clone() - local_offset_on_dim
        masked_tensor[mask] = 0
        # 实现遮罩缓冲区，用于后续减少操作使用
        self.mask_buffer.materialize_mask(mask)
        return masked_tensor

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # 判断遮罩缓冲区数据是否已存在
        assert self.mask_buffer.data is not None

        # 将遮罩应用到待减少的张量上
        self.mask_buffer.apply_mask(tensor)

        # 清除遮罩缓冲区
        self.mask_buffer.release_mask()

        # 执行求和减少操作
        return funcol.all_reduce(
            tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim)
        )

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # 判断遮罩缓冲区数据是否已存在
        assert self.mask_buffer.data is not None

        # 将遮罩应用到待减少的张量上
        self.mask_buffer.apply_mask(tensor)

        # 清除遮罩缓冲区
        self.mask_buffer.release_mask()

        # 调用shard_spec的reduce_shard_tensor方法
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MaskPartial):
            return False

        # 如果任一数据不为None，说明当前MaskPartial放置仍在使用中，不应用于缓存命中
        if self.mask_buffer.data is not None or other.mask_buffer.data is not None:
            return False

        return (
            self.reduce_op == other.reduce_op
            and self.logical_dim_size == other.logical_dim_size
        )

    def __hash__(self) -> int:
        return 1 + hash(
            (self.logical_dim_size, id(self.mask_buffer.data), self.reduce_op)
        )
    # 返回一个机器可读的字符串表示，描述了MaskPartial的布局
    def __repr__(self) -> str:
        """
        machine readable representation of the MaskPartial placement
        """
        return f"_MaskPartial(logical_dim_size={self.logical_dim_size})"

    # 返回一个人类可读的字符串表示，描述了MaskPartial的布局
    def __str__(self) -> str:
        """
        human readable representation of the MaskPartial placement
        """
        return "MaskP"
@register_op_strategy(aten.embedding.default)
def embedding_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    """
    # 获取权重和索引的策略对象
    weight_strategy = cast(OpStrategy, op_schema.args_schema[0])
    indices_strategy = cast(OpStrategy, op_schema.args_schema[1])

    # 获取权重和索引的形状信息
    weight_shape = weight_strategy.shape
    indices_shape = indices_strategy.shape
    output_emd_dim = len(indices_shape)

    # 单个网格维度策略列表
    single_mesh_dim_strategies = []

    # 放置列表存储[输出, 权重, 输入索引]的放置方式
    # 首先，我们对所有输入和输出都使用复制策略
    all_replicate: List[Placement] = [Replicate()] * 3
    single_mesh_dim_strategies.append(all_replicate)

    # 列分片，输出在最后一个维度上分片，权重在第一个维度上分片，输入索引复制
    colwise_sharding = [Shard(output_emd_dim), Shard(1), Replicate()]
    single_mesh_dim_strategies.append(colwise_sharding)

    # 行分片，输出为嵌入部分，权重在第0个维度上分片，输入索引接受嵌入部分
    embedding_partial_placement = _MaskPartial(logical_dim_size=weight_shape[0])
    rowwise_sharding = [
        embedding_partial_placement,
        Shard(0),
        embedding_partial_placement,
    ]
    single_mesh_dim_strategies.append(rowwise_sharding)

    # 批次维度分片，权重复制，输入可以在任何维度上分片，输出跟随输入
    for input_dim in range(len(indices_shape)):
        batch_sharding = [Shard(input_dim), Replicate(), Shard(input_dim)]
        single_mesh_dim_strategies.append(batch_sharding)

    # 扩展到完整网格的操作策略
    return expand_to_full_mesh_op_strategy(mesh, op_schema, single_mesh_dim_strategies)


@register_op_strategy(aten.embedding_dense_backward.default)
def embedding_dense_backward_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    """
    # 获取梯度输出和索引的策略对象
    grad_out_strategy = cast(OpStrategy, op_schema.args_schema[0])
    indices_strategy = cast(OpStrategy, op_schema.args_schema[1])

    # 获取梯度输出和索引的形状信息
    grad_out_shape = grad_out_strategy.shape
    indices_shape = indices_strategy.shape
    grad_out_ndim = len(grad_out_shape)

    # 单个网格维度策略列表
    single_mesh_dim_strategies = []

    # 放置列表存储[输出梯度, 权重梯度, 输入索引]的放置方式
    # 首先，我们对所有输入和输出都使用复制策略
    all_replicate: List[Placement] = [Replicate()] * 3
    single_mesh_dim_strategies.append(all_replicate)

    # 列分片反向传播，梯度输出在最后一个维度上分片，输入复制，
    # 权重梯度在列分片
    colwise_sharding = [Shard(1), Shard(grad_out_ndim - 1), Replicate()]
    single_mesh_dim_strategies.append(colwise_sharding)
    # 批处理维度分片，权重复制，grad_out/input 具有相同的分片方式
    # 可以在任意维度上进行分片，权重梯度部分化
    for input_dim in range(len(indices_shape)):
        # 创建批处理维度的分片策略列表，其中第一维和第二维使用 Partial 分片，第三维使用 Shard 分片
        batch_sharding = [Partial(), Shard(input_dim), Shard(input_dim)]
        single_mesh_dim_strategies.append(batch_sharding)

    # grad_out 使用部分化分片，input 使用复制分片，权重梯度保持部分化
    partial_sharding = [Partial(), Partial(), Replicate()]
    single_mesh_dim_strategies.append(partial_sharding)

    # 调用函数将分片策略扩展到完整的网格操作策略
    return expand_to_full_mesh_op_strategy(mesh, op_schema, single_mesh_dim_strategies)
```