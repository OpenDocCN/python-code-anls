# `.\pytorch\torch\distributed\_tensor\ops\utils.py`

```py
`
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# 引入必要的模块和库
import functools
import itertools
import operator
from typing import cast, Iterable, List, Sequence, Tuple, Union

# 引入PyTorch相关模块
import torch
from torch.distributed._tensor._collective_utils import redistribute_cost
from torch.distributed._tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    RuntimeSchemaInfo,
)
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
)

# 注册函数，用于注册分片传播规则
# 参数op可以是单个操作或操作列表，schema_info用于指定运行时模式信息
# 返回一个装饰器函数wrapper
def register_prop_rule(op, schema_info=None):
    # wrapper函数，用于实际装饰操作实现函数impl
    # overloads根据op类型确定操作列表
    def wrapper(impl):
        overloads = op if isinstance(op, list) else [op]
        # 遍历每个操作，并将传播规则注册到DTensor的操作调度器中
        for overload in overloads:
            DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(
                overload, impl, schema_info
            )
        return impl

    return wrapper


# 注册函数，用于注册操作策略
# 参数op可以是单个操作或操作列表，schema_info用于指定运行时模式信息
# 返回一个装饰器函数wrapper
def register_op_strategy(op, schema_info=None):
    def wrapper(impl):
        # 根据op类型确定操作列表
        if isinstance(op, list):
            overloads = op
        else:
            overloads = [op]

        # 遍历每个操作，并根据特定参数设置运行时模式信息
        for overload in overloads:
            curr_schema_info = None
            if schema_info is None:
                specialized_args = [
                    a.name
                    for a in overload._schema.arguments
                    if a.name in arg_names_that_require_specializing_cache_strategy
                ]
                if any(specialized_args):
                    curr_schema_info = RuntimeSchemaInfo(
                        static_kwargkey=specialized_args
                    )
            else:
                curr_schema_info = schema_info
            # 将操作策略注册到DTensor的操作调度器中
            DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
                overload, impl, curr_schema_info
            )
        return impl

    return wrapper


# 函数定义，用于将输入转换为列表形式
# 参数x可以是对象或对象列表
# 返回一个包含对象的列表
def as_list(
    x: Union[List[object], object]
    # 参数类型未定义的注释
def is_tensor_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is shardable according to the spec."""
    # 初始化每个维度的分片数量为1
    shards_map = [1] * len(shape)
    # 遍历规格中的每个放置位置
    for i, placement in enumerate(spec.placements):
        # 如果是分片位置，则更新对应维度的分片数量
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            shards_map[shard_dim] *= spec.mesh.size(i)

    # 检查每个维度的大小是否足够大以容纳其分片数量
    for i, dim_size in enumerate(shape):
        if shards_map[i] > 1 and dim_size < shards_map[i]:
            return False

    return True


def is_tensor_evenly_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is evenly shardable according to the spec."""
    # 初始化每个维度的分片数量为1
    shards_map = [1] * len(shape)
    # 遍历规格中的每个放置位置
    for i, placement in enumerate(spec.placements):
        # 如果是分片位置，则更新对应维度的分片数量
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            shards_map[shard_dim] *= spec.mesh.size(i)

    # 检查每个维度是否可以均匀分片
    for i, dim_size in enumerate(shape):
        if shards_map[i] > 1 and (dim_size % shards_map[i] != 0):
            return False

    return True


def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool:
    """Return True if tensor dim is sharded."""
    # 对于规格中的每个位置，检查是否是给定维度的碎片
    return any(p.is_shard(dim) for p in spec.placements)
def is_tensor_partial(spec: DTensorSpec) -> bool:
    """Return True if tensor is partial on the mesh."""
    # 检查是否存在任何一个放置策略标记为部分数据
    return any(p.is_partial() for p in spec.placements)


def infer_broadcast_dims_map(
    common_shape: torch.Size, input_shape: torch.Size
) -> List[int]:
    # 推断广播维度映射，将通用形状的维度映射到输入形状的维度
    # 这与广播语义一致
    common_ndim = len(common_shape)
    input_ndim = len(input_shape)
    broadcast_dims_map = [-1] * common_ndim
    for idx in range(-1, -1 - input_ndim, -1):
        if input_shape[idx] == common_shape[idx]:
            broadcast_dims_map[common_ndim + idx] = input_ndim + idx
    return broadcast_dims_map


def map_placements_after_broadcast(
    placements: Tuple[Placement, ...],
    shape: torch.Size,
    broadcast_dims_map: List[int],
) -> Tuple[Placement, ...]:
    """Map each placement based on the output shape after broadcast."""
    new_placements: List[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, Partial)):
            new_placements.append(placement)
        else:
            assert isinstance(placement, Shard)
            shard_dim = normalize_dim(placement.dim, len(shape))
            new_shard_dim = broadcast_dims_map[shard_dim]
            if new_shard_dim != -1:
                # 存在从通用形状分片维度到输入形状分片维度的映射，
                # 使用该映射替代
                new_placements.append(Shard(new_shard_dim))
            else:
                # 不存在通用形状分片维度与输入形状分片维度的映射，
                # 这表示在该维度上隐式广播发生，
                # 因此将其标记为复制，隐式广播将自动广播到分片形状
                new_placements.append(Replicate())

    return tuple(new_placements)


def generate_redistribute_costs(
    src_strategy: OpStrategy, dst_spec: DTensorSpec
) -> List[float]:
    redistribute_costs: List[float] = []
    for strat in src_strategy.strategies:
        redistribute_costs.append(redistribute_cost(strat.output_spec, dst_spec))

    return redistribute_costs


def expand_to_full_mesh_op_strategy(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_mesh_dim_strategies: List[List[Placement]],
    *,
    input_index: int = 1,
    inplace_op: bool = False,
) -> OpStrategy:
    # 将单一网格维度策略扩展到完整网格维度策略
    all_mesh_dim_strategies = [single_mesh_dim_strategies] * mesh.ndim

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    # 遍历每个策略组合中的策略组
    for strategy_comb in strategy_combs:
        # 初始化一个空列表，用于存储生成的 DTensorSpec 对象
        spec_list = []
        # 对于每个策略组合中的规格，使用 zip 将其解压并逐个处理
        for specs in zip(*strategy_comb):
            # 根据 mesh 和 specs 创建一个 DTensorSpec 对象，并添加到 spec_list 中
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        # 从输入索引开始获取输入规格
        input_specs = spec_list[input_index:]
        # 获取操作模式的参数策略
        input_args_strategy = op_schema.args_strategy
        # 断言输入规格的长度与参数策略的长度相等
        assert len(input_specs) == len(input_args_strategy)
        # 获取自身规格的输出规格
        self_spec = input_args_strategy[0].strategies[0].output_spec
        # 如果是原地操作并且自身规格的位置与第一个输入规格的位置不匹配，则跳过当前循环
        if inplace_op and self_spec.placements != input_specs[0].placements:
            continue

        # 检查所有输入是否可分片
        inputs_shardable = all(
            is_tensor_shardable(inp.shape, s)
            for inp, s in zip(input_args_strategy, input_specs)
        )

        # 当所有输入都可分片时，生成重新分布成本列表
        if inputs_shardable:
            redistribute_cost = [
                # 为每个输入参数策略和输入规格生成重新分布成本
                generate_redistribute_costs(input_strategy, input_spec)
                for input_strategy, input_spec in zip(input_args_strategy, input_specs)
            ]
            # 创建一个 PlacementStrategy 对象，并添加到 all_strategies 列表中
            strategy = PlacementStrategy(
                output_specs=tuple(spec_list[:input_index])
                if input_index > 1
                else spec_list[0],
                input_specs=input_specs,
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strategy)

    # 返回 OpStrategy 对象，其中包含所有生成的策略
    return OpStrategy(all_strategies)
```