# `.\pytorch\torch\distributed\_tensor\ops\tensor_ops.py`

```py
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 引入所需的类型声明
from typing import cast, List, Optional, Sequence, Tuple

# 引入 PyTorch 库
import torch

# 从 torch.distributed._tensor._op_schema 模块导入相关符号
from torch.distributed._tensor._op_schema import (
    _is_inplace_op,
    OpSchema,
    OpStrategy,
    OutputSharding,
    PlacementStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)

# 从 torch.distributed._tensor.ops.common_rules 模块导入 pointwise_rule 符号
from torch.distributed._tensor.ops.common_rules import pointwise_rule

# 从 torch.distributed._tensor.ops.embedding_ops 模块导入 _MaskPartial 符号
from torch.distributed._tensor.ops.embedding_ops import _MaskPartial

# 从 torch.distributed._tensor.ops.utils 模块导入多个函数和类
from torch.distributed._tensor.ops.utils import (
    expand_to_full_mesh_op_strategy,
    is_tensor_dim_sharded,
    is_tensor_evenly_shardable,
    is_tensor_partial,
    normalize_dim,
    register_op_strategy,
    register_prop_rule,
)

# 从 torch.distributed._tensor.placement_types 模块导入多个类
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
)

# 从 torch.distributed.device_mesh 模块导入 DeviceMesh 类
from torch.distributed.device_mesh import DeviceMesh

# 使用 torch.ops.aten 别名定义 aten
aten = torch.ops.aten


def default_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # 默认策略：默认情况下只传播第一个输入的策略
    select_strategy = op_schema.args_schema[0]
    assert isinstance(select_strategy, OpStrategy)
    default_strategy = []
    for strategy in select_strategy.strategies:
        # 为了确保参数和输出的张量元数据是独立的，即使是默认策略也创建新的 DTensorSpecs
        default_strategy.append(
            PlacementStrategy(
                output_specs=DTensorSpec(
                    mesh=strategy.output_spec.mesh,
                    placements=strategy.output_spec.placements,
                )
            )
        )
    return OpStrategy(default_strategy)


# 注册默认策略到指定的操作列表
register_op_strategy(
    [
        aten.clone.default,
        aten.contiguous.default,
        aten.copy_.default,
        aten.detach.default,
        aten.fill_.Scalar,
        aten.zero_.default,
    ]
)(default_strategy)

# 注册特定操作的默认策略，使用 RuntimeSchemaInfo 来指定静态关键字参数
register_op_strategy(
    aten._to_copy.default, schema_info=RuntimeSchemaInfo(static_kwargkey=["dtype"])
)(default_strategy)


@register_op_strategy(
    [
        aten.equal.default,
        aten.is_same_size.default,
    ]
)
def equal_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # equal_strategy 处理比较两个张量的操作，确保两个操作数具有相同的分片布局，
    # 我们选择跟随具有较大分片数的参数，is_same_size 也保留在此处以完整性考虑，因为它们在理论上共享相同的策略。
    self_strategy, other_strategy = op_schema.args_schema
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(other_strategy, OpStrategy)

    # 选择较大分片数的参数策略作为主策略
    select_strategy = (
        self_strategy
        if self_strategy.max_num_shards() >= other_strategy.max_num_shards()
        else other_strategy
    )
    equal_strategy = OpStrategy([])
    # 遍历选择策略对象中的每一个策略
    for arg_strategy in select_strategy.strategies:
        # 获取当前策略的输出规范
        arg_spec = arg_strategy.output_spec
        # 检查输出规范是否包含部分张量
        if is_tensor_partial(arg_spec):
            # 如果输出规范包含部分张量，将其重新分片以复制
            # 否则，本地分片张量的比较将无效
            output_spec = DTensorSpec(
                mesh=arg_spec.mesh,  # 使用相同的网格
                placements=tuple(
                    Replicate() if isinstance(p, Partial) else p
                    for p in arg_spec.placements  # 遍历每个放置方式
                ),
            )
            # 将复制后的输出规范添加到相等策略对象中的策略列表
            equal_strategy.strategies.append(
                PlacementStrategy(output_specs=output_spec)
            )
        else:
            # 如果输出规范没有部分张量，直接将当前输出规范添加到相等策略对象中的策略列表
            equal_strategy.strategies.append(PlacementStrategy(arg_spec))
    # 返回填充后的相等策略对象
    return equal_strategy
@register_op_strategy(
    [
        aten.empty_like.default,  # 注册空张量生成策略，默认参数
        aten.ones_like.default,   # 注册全1张量生成策略，默认参数
        aten.rand_like.default,   # 注册随机张量生成策略，默认参数
        aten.randn_like.default,  # 注册标准正态随机张量生成策略，默认参数
        aten.zeros_like.default,  # 注册全0张量生成策略，默认参数
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),  # 设置运行时模式信息，1表示一个参数，包含dtype
)
@register_op_strategy(
    [aten.full_like.default],  # 注册全填充张量生成策略，默认参数
    schema_info=RuntimeSchemaInfo(2, ["dtype"]),  # 设置运行时模式信息，2表示两个参数，包含dtype
)
@register_op_strategy(
    [
        aten.randint_like.default,        # 注册整数随机张量生成策略，默认参数
        aten.randint_like.low_dtype,     # 注册低dtype整数随机张量生成策略，默认参数
        aten.randint_like.low_dtype_out,  # 注册低dtype整数随机张量生成策略输出策略，默认参数
    ],
    schema_info=RuntimeSchemaInfo(3, ["dtype"]),  # 设置运行时模式信息，3表示三个参数，包含dtype
)
def create_like_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # create_like_strategy处理创建与输入形状相同但内容特定的张量操作，
    # 可以传播分片，但必须确保从部分到复制的转换。
    select_strategy = op_schema.args_schema[0]
    create_like_strategy = OpStrategy([])  # 初始化操作策略对象

    assert isinstance(select_strategy, OpStrategy)  # 确保选择策略是OpStrategy类型
    for arg_strategy in select_strategy.strategies:
        arg_spec = arg_strategy.output_spec  # 获取参数策略的输出规格
        if is_tensor_partial(arg_spec):  # 如果参数规格是部分张量
            # 如果参数规格包含部分张量，则接受输入规格中的部分，
            # 但对应的网格维度输出为复制
            output_spec = DTensorSpec(
                mesh=arg_spec.mesh,
                placements=tuple(
                    Replicate() if isinstance(p, Partial) else p
                    for p in arg_spec.placements
                ),
            )
            create_like_strategy.strategies.append(
                PlacementStrategy(output_specs=output_spec, input_specs=(arg_spec,))
            )
        else:
            create_like_strategy.strategies.append(PlacementStrategy(arg_spec))  # 否则直接添加参数策略

    return create_like_strategy  # 返回生成的操作策略对象


@register_op_strategy(
    [
        aten.new_empty.default,          # 注册新空张量生成策略，默认参数
        aten.new_full.default,           # 注册新全填充张量生成策略，默认参数
        aten.new_ones.default,           # 注册新全1张量生成策略，默认参数
        aten.new_zeros.default,          # 注册新全0张量生成策略，默认参数
        aten.new_empty_strided.default,  # 注册新空步长张量生成策略，默认参数
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),  # 设置运行时模式信息，1表示一个参数，包含dtype
)
def new_factory_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # 目前有两种策略：
    # 1. 让输出为复制
    # 2. 如果输入和输出具有相同的形状，则让输出跟随输入
    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)  # 确保输入策略是OpStrategy类型
    input_shape = input_strategy.shape  # 获取输入形状
    output_shape = op_schema.args_schema[1]  # 获取输出形状
    assert isinstance(output_shape, list)

    new_factory_strategy = OpStrategy([])  # 初始化操作策略对象
    # 遍历输入策略中的各个策略对象
    for arg_strategy in input_strategy.strategies:
        # 获取当前策略对象的输出规格
        input_spec = arg_strategy.output_spec
        # 创建一个指定副本的规格对象，用于副本策略
        replica_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
        # 将新的副本策略添加到新工厂策略中
        new_factory_strategy.strategies.append(
            PlacementStrategy(
                output_specs=replica_spec,
                input_specs=(input_spec,),
                redistribute_cost=[[0.0] * mesh.ndim],
            )
        )

        # 检查输入形状和输出形状是否相同，并且输入规格是否为分片的
        if tuple(input_shape) == tuple(output_shape) and input_spec.is_sharded():
            # 对于 new_empty_strided，默认操作，只有当形状可以均匀分片时才支持非副本分片
            if (
                op_schema.op == aten.new_empty_strided.default
                and not is_tensor_evenly_shardable(input_shape, input_spec)
            ):
                # 如果不支持非均匀分片，则跳过当前策略
                continue

            # 将新的非副本策略添加到新工厂策略中
            new_factory_strategy.strategies.append(
                PlacementStrategy(
                    output_specs=input_spec,
                    input_specs=(input_spec,),
                    # 鼓励新张量的放置与输入相同
                    redistribute_cost=[[-0.1] * mesh.ndim],
                )
            )

    # 返回更新后的新工厂策略对象
    return new_factory_strategy
@register_op_strategy(aten.bucketize.Tensor)
def gen_bucketize_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """Just propagate input sharding, but expect replicated for boundaries input."""
    # 获取输入参数的策略信息
    input_strategy = op_schema.args_schema[0]
    # 创建空的操作策略对象
    bucketize_strategy = OpStrategy([])
    # 断言输入策略为 OpStrategy 类型
    assert isinstance(input_strategy, OpStrategy)
    # 遍历输入策略中的每一个参数策略
    for arg_strategy in input_strategy.strategies:
        # 根据参数策略的输出规格创建 DTensorSpec 对象
        arg_spec = DTensorSpec(mesh, arg_strategy.output_spec.placements)
        # 创建一个包含复制放置的 DTensorSpec 对象
        replica_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
        # 将包含输入和复制规格的放置策略添加到 bucketize_strategy 中
        bucketize_strategy.strategies.append(
            PlacementStrategy(
                output_specs=arg_spec, input_specs=(arg_spec, replica_spec)
            )
        )

    return bucketize_strategy


@register_op_strategy(aten.slice.Tensor, schema_info=RuntimeSchemaInfo(1))
def gen_slice_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """Forward all shardings except the slice dimension."""
    # 默认参数
    defaults = (None, 0, None, None, 1)
    # 解包操作模式的参数
    input_strategy, dim, start, end, step = (
        op_schema.args_schema + defaults[len(op_schema.args_schema) :]
    )
    # 断言输入策略为 OpStrategy 类型
    assert isinstance(input_strategy, OpStrategy)
    # 获取输入策略的形状和维度数
    input_shape = input_strategy.shape
    input_ndim = input_strategy.ndim
    # 断言维度为整数类型
    assert isinstance(dim, int)
    # 如果起始索引为 None，则设为 0
    if start is None:
        start = 0
    # 如果结束索引为 None 或超出维度范围，则设为维度的最大值
    if end is None or end > input_shape[dim]:
        end = input_shape[dim]
    # 断言起始、结束和步长为整数类型
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(step, int)

    # 规范化切片参数
    slice_dim = normalize_dim(dim, input_ndim)
    start = normalize_dim(start, input_shape[dim])
    end = normalize_dim(end, input_shape[dim])

    # 检查是否是冗余切片
    redundant_slice = start == 0 and end == input_shape[dim] and step == 1

    # 创建空的操作策略对象
    slice_strategy = OpStrategy([])

    # 遍历输入策略中的每一个参数策略
    for arg_strategy in input_strategy.strategies:
        # 获取参数策略的输出规格
        arg_spec = arg_strategy.output_spec
        # 如果参数策略在切片维度上未分片或存在冗余切片
        if not is_tensor_dim_sharded(arg_spec, dim=slice_dim) or redundant_slice:
            # 创建一个包含输出规格放置策略的 DTensorSpec 对象
            out_spec = DTensorSpec(mesh, arg_spec.placements)
            # 将该放置策略添加到 slice_strategy 中
            slice_strategy.strategies.append(PlacementStrategy(output_specs=out_spec))
    
    # 如果 slice_strategy 中没有策略
    if not slice_strategy.strategies:
        # 将输入策略中切片维度上的所有规格取消分片，并作为操作策略返回
        for arg_strategy in input_strategy.strategies:
            arg_spec = arg_strategy.output_spec
            unshard_spec = DTensorSpec(
                mesh, unshard_tensor_dim(arg_spec.placements, dim=slice_dim)
            )
            slice_strategy.strategies.append(
                PlacementStrategy(output_specs=unshard_spec)
            )
    
    return slice_strategy


def unshard_tensor_dim(
    placements: Sequence[Placement], dim: int
) -> Tuple[Placement, ...]:
    """Disallow the given tensor dimension to be sharded."""
    # 取消给定张量维度的分片
    # 返回一个元组，对给定的列表 placements 中的每个元素进行条件判断和处理
    return tuple(
        # 如果当前元素 p 不是 Shard 类型或者其维度不等于 dim，则保持不变
        p if (not isinstance(p, Shard) or p.dim != dim) else Replicate()
        # 遍历 placements 列表中的每个元素 p
        for p in placements
    )
# 强制指定张量维度进行复制。
def replicate_tensor_dim(
    placements: Sequence[Placement], dim: int
) -> Tuple[Placement, ...]:
    """Force the given tensor dimension to be replicated."""
    # 遍历给定的位置信息序列，根据条件选择复制或保持原样的位置对象。
    return tuple(
        Replicate() if p.is_partial() or isinstance(p, Shard) and p.dim == dim else p
        for p in placements
    )


# 注册函数策略，用于生成切片分散操作的策略类型。
@register_op_strategy(aten.slice_scatter.default, schema_info=RuntimeSchemaInfo(2))
def gen_slice_scatter_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # 1. 输入和源张量的维度数必须匹配。
    # 2. 输入和源张量在非切片维度上的元素数必须匹配。
    # 3. 源张量在切片维度上的元素数必须匹配切片大小。
    # 根据上述条件：
    # - 我们建议源张量遵循输入张量的分片，除了在分散维度上，我们目前最好的选择是作为备用进行复制。
    #   TODO: 理想情况下，我们希望确保输出在保持输入分片后重新分片。

    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    input_ndim = input_strategy.ndim
    slice_dim = (
        cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else 0
    )
    slice_dim = normalize_dim(slice_dim, input_ndim)

    slice_scatter_strategy = OpStrategy([])
    # 默认情况下，对于输入和源张量都遵循输入策略
    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if not (
            is_tensor_dim_sharded(arg_spec, dim=slice_dim)
            or is_tensor_partial(arg_spec)
        ):
            # 仅在分散切片维度不分片或部分分片时添加策略
            slice_scatter_strategy.strategies.append(
                PlacementStrategy(output_specs=arg_spec)
            )

    if not slice_scatter_strategy.strategies:
        # 如果所有策略都被过滤掉，在输入策略的分片切片维度上复制所有规范，并将其用作操作策略
        for arg_strategy in input_strategy.strategies:
            arg_spec = arg_strategy.output_spec
            replicate_spec = DTensorSpec(
                mesh, replicate_tensor_dim(arg_spec.placements, dim=slice_dim)
            )
            slice_scatter_strategy.strategies.append(
                PlacementStrategy(output_specs=replicate_spec)
            )
    return slice_scatter_strategy


# 注册函数策略，用于本地标量密集操作的策略类型。
@register_op_strategy(aten._local_scalar_dense.default)
def replica_only_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """Only allow replication on the input/output."""
    # 创建一个复制所有维度的规范
    replicate_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
    return OpStrategy([PlacementStrategy(replicate_spec)])


# 注册函数策略，用于scatter操作的策略类型。
@register_op_strategy(
    [aten.scatter_.value, aten.scatter.value, aten.scatter_.src, aten.scatter.src],
    # 创建一个 RuntimeSchemaInfo 对象，传入参数 1 初始化
    schema_info=RuntimeSchemaInfo(1),
)
def scatter_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # Cast the first argument strategy to OpStrategy
    input_strategy = cast(OpStrategy, op_schema.args_schema[0])
    # Initialize an empty list to store strategies for single mesh dimension
    single_mesh_dim_strategies = []

    # placement list stores placements of [output, input, index, src]
    # first we always have replicate all for inputs and output
    # Check if the length of args_strategy is less than 3
    if len(op_schema.args_strategy) < 3:
        # Define all_replicate as a list of three Replicate placements
        all_replicate: List[Placement] = [Replicate()] * 3
    else:
        # Define all_replicate as a list of four Replicate placements
        all_replicate = [Replicate()] * 4
    # Append all_replicate to single_mesh_dim_strategies
    single_mesh_dim_strategies.append(all_replicate)

    # TODO: see if we can support input sharding pattern
    # Check if the operation is in-place
    inplace_op = _is_inplace_op(op_schema.op)

    # Expand the single mesh dimension strategies to a full mesh operation strategy
    op_strategy = expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, inplace_op=inplace_op
    )
    # Return the computed operation strategy
    return op_strategy


@register_op_strategy(aten.gather.default)
def gather_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # Cast the first argument strategy to OpStrategy
    input_strategy = cast(OpStrategy, op_schema.args_schema[0])
    # Cast the second argument schema to integer for dimension
    dim = cast(int, op_schema.args_schema[1])
    # Cast the third argument strategy to OpStrategy for index
    index_strategy = cast(OpStrategy, op_schema.args_schema[2])

    # Retrieve the shape of the input and index strategies
    input_shape = input_strategy.shape
    index_shape = index_strategy.shape

    # Initialize an empty list to store strategies for single mesh dimension
    single_mesh_dim_strategies = []

    # placement list stores placements of [output, input, index]
    # first we always have replicate all for inputs and output
    # Define all_replicate as a list of three Replicate placements
    all_replicate: List[Placement] = [Replicate()] * 3
    # Append all_replicate to single_mesh_dim_strategies
    single_mesh_dim_strategies.append(all_replicate)

    # input sharding, input sharded, index accepts mask partial, output follows index
    # this only works when the input is sharded on the gather dimension, and
    # index has size 1 on the gather dimension
    if index_shape[dim] == 1:
        # Create index_partial_placement using _MaskPartial with input_shape[dim]
        index_partial_placement = _MaskPartial(logical_dim_size=input_shape[dim])
        # Define input_sharding as a list of placements
        input_sharding = [
            index_partial_placement,
            Shard(dim),
            index_partial_placement,
        ]
        # Append input_sharding to single_mesh_dim_strategies
        single_mesh_dim_strategies.append(input_sharding)

    # index sharding, input replicated, index sharded, output follows index
    # this only works when the sharding dimension is the gather dimension
    # Define index_sharding as a list of placements
    index_sharding = [Shard(dim), Replicate(), Shard(dim)]
    # Append index_sharding to single_mesh_dim_strategies
    single_mesh_dim_strategies.append(index_sharding)

    # Expand the single mesh dimension strategies to a full mesh operation strategy
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )


def _derive_follow_placements_from_tuple_strategy(
    tuple_strategy: TupleStrategy,
) -> Sequence[Placement]:
    """
    derive the placements to follow from the tuple strategy, mainly used by
    aten.stack, aten.cat, where each operand have the same shape, and correspondingly
    expecting the same sharding
    """
    
    def merge_placement(
        cur_placement: Placement, new_placement: Placement

        cur_placement: Placement, new_placement: Placement
    ):
        # Placeholder function for merging placements
        pass
    # Placeholder function for deriving placements from tuple strategy
    pass
    ) -> Placement:
        # 如果当前的放置与新的放置相同，则直接返回当前放置
        if cur_placement == new_placement:
            return cur_placement

        # 如果当前放置是部分放置
        if cur_placement.is_partial():
            # 如果新放置是分片放置，则跟随新的放置
            if new_placement.is_shard():
                return new_placement
            # 如果新放置也是部分放置，但是类型不同，无法合并，需要全部复制
            elif new_placement.is_partial():
                return Replicate()
            # 其他情况下，跟随当前部分放置
            else:
                return cur_placement
        # 如果当前放置是分片放置
        elif cur_placement.is_shard():
            # 如果新放置也是分片放置，但是维度不同，需要全部复制
            if new_placement.is_shard():
                return Replicate()
            # 对于部分放置或者全部复制，跟随当前分片放置
            else:
                return cur_placement
        # 如果当前放置是全部复制
        else:
            # 直接跟随新的放置
            return new_placement

    # 初始化跟随放置列表为 None
    follow_placements: Optional[List[Placement]] = None
    # 遍历所有子策略
    for arg_strategy in tuple_strategy.childs:
        assert isinstance(arg_strategy, OpStrategy)
        # 遍历每个策略的放置策略
        for placement_strategy in arg_strategy.strategies:
            # 获取每个策略的输出放置列表
            arg_placements = placement_strategy.output_spec.placements
            # 如果跟随放置列表为空，则直接复制当前策略的放置列表
            if follow_placements is None:
                follow_placements = list(arg_placements)
                continue
            # 获取跟随放置列表的维度
            mesh_ndim = len(follow_placements)
            # 断言跟随放置列表不为空
            assert follow_placements is not None
            # 遍历每个维度，将当前策略的放置与跟随放置列表中的放置进行合并
            for mesh_idx in range(mesh_ndim):
                follow_placements[mesh_idx] = merge_placement(
                    follow_placements[mesh_idx], arg_placements[mesh_idx]
                )
    # 最终确保跟随放置列表不为空，否则抛出异常
    assert follow_placements is not None, "follow placements should not be None!"
    # 返回最终的跟随放置列表
    return follow_placements
def normalize_shard_for_stack(
    placements: Sequence[Placement], insert_dim: int = 0
) -> Sequence[Placement]:
    # stack op would "insert" new dim, so all sharded dim >= the inserted dim need to
    # be normalized with the new Shard placement
    # 创建一个空列表，用于存放经过规范化后的 Placement 对象
    normalized_placements: List[Placement] = []
    # 遍历传入的 placements 列表
    for placement in placements:
        # 检查当前 placement 是否为 Shard 类型，并且其维度大于等于 insert_dim
        if isinstance(placement, Shard) and placement.dim >= insert_dim:
            # 如果是，则创建一个新的 Shard 对象，维度加一，并添加到 normalized_placements 中
            normalized_placements.append(Shard(placement.dim + 1))
        else:
            # 否则直接将当前 placement 添加到 normalized_placements 中
            normalized_placements.append(placement)
    # 返回经过规范化后的 placements 列表
    return normalized_placements


@register_op_strategy(aten.stack.default, RuntimeSchemaInfo(1, needs_pytree=True))
def stack_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # 从操作模式架构中获取参数架构
    args_schema = op_schema.args_schema
    # 从参数架构中获取输入元组策略
    input_tuple_strategy = args_schema[0]
    assert isinstance(input_tuple_strategy, TupleStrategy), f"{input_tuple_strategy}"
    # 获取第一个输入策略
    first_input_strategy = input_tuple_strategy.childs[0]
    assert isinstance(first_input_strategy, OpStrategy), f"{first_input_strategy}"
    # 获取共同的输入维度
    common_input_ndim = first_input_strategy.ndim
    # 获取操作的维度参数，如果不存在则默认为0
    dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    # 将维度参数规范化，确保在共同输入维度范围内
    dim = normalize_dim(dim, common_input_ndim)

    # 从输入元组策略中推导 follow placements
    follow_placements = _derive_follow_placements_from_tuple_strategy(
        input_tuple_strategy
    )

    # 根据规范化后的维度规范化 follow placements
    follow_placements = normalize_shard_for_stack(follow_placements, dim)

    # 创建基于 follow placements 的操作策略对象
    op_strategy = OpStrategy([])

    # 创建输入规格列表，每个元素都是一个 DTensorSpec 对象，使用 follow placements
    input_specs = tuple(
        DTensorSpec(mesh, tuple(follow_placements))
        for _ in range(len(input_tuple_strategy.childs))
    )

    # 将输出规格和输入规格添加到操作策略中
    op_strategy.strategies.append(
        PlacementStrategy(
            output_specs=DTensorSpec(mesh, tuple(follow_placements)),
            input_specs=input_specs,
        )
    )
    # 返回操作策略对象
    return op_strategy


@register_op_strategy(aten.cat.default, RuntimeSchemaInfo(1, needs_pytree=True))
def cat_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # 从操作模式架构中获取参数架构
    args_schema = op_schema.args_schema
    # 从参数架构中获取输入元组策略
    input_tuple_strategy = args_schema[0]
    assert isinstance(input_tuple_strategy, TupleStrategy), f"{input_tuple_strategy}"
    # 获取第一个输入策略
    first_input_strategy = input_tuple_strategy.childs[0]
    assert isinstance(first_input_strategy, OpStrategy), f"{first_input_strategy}"
    # 获取共同的输入维度
    common_input_ndim = first_input_strategy.ndim
    # 获取操作的维度参数，如果不存在则默认为0
    dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    # 将维度参数规范化，确保在共同输入维度范围内
    dim = normalize_dim(dim, common_input_ndim)

    # 从输入元组策略中推导 follow placements
    follow_placements = _derive_follow_placements_from_tuple_strategy(
        input_tuple_strategy
    )

    # 在 cat 操作中，如果 cat 维度已经被分片，需要解除其分片
    follow_placements = unshard_tensor_dim(follow_placements, dim)

    # 创建基于 follow placements 的操作策略对象
    op_strategy = OpStrategy([])
    # 生成一个元组，包含多个 DTensorSpec 对象，每个对象使用给定的 mesh 和 follow_placements 参数
    input_specs = tuple(
        DTensorSpec(mesh, tuple(follow_placements))
        for _ in range(len(input_tuple_strategy.childs))
    )
    # 创建一个新的 PlacementStrategy 对象，并将其 output_specs 设置为一个 DTensorSpec 对象
    # 使用给定的 mesh 和 follow_placements 参数
    # 将 input_specs 设置为刚刚生成的 input_specs 元组
    op_strategy.strategies.append(
        PlacementStrategy(
            output_specs=DTensorSpec(mesh, tuple(follow_placements)),
            input_specs=input_specs,
        )
    )
    # 返回已经配置好的 op_strategy 对象
    return op_strategy
@register_prop_rule(aten.index_select.default, schema_info=RuntimeSchemaInfo(1))
def prop_index_select(op_schema: OpSchema) -> OutputSharding:
    # 从操作模式中解析出张量值、维度和索引规范
    values_spec, dim, indices_spec = op_schema.args_schema

    # 确保值规范是张量规范
    assert isinstance(values_spec, DTensorSpec)
    # 确保维度是整数
    assert isinstance(dim, int)
    # 确保索引规范是张量规范
    assert isinstance(indices_spec, DTensorSpec)

    # 创建一个包含所有维度的索引规范列表，其他维度除了当前维度 dim 外为 None
    all_indices_spec: List[Optional[DTensorSpec]] = [
        indices_spec if dim == i else None for i in range(values_spec.ndim)
    ]

    # 对 prop_index 函数进行调用，使用新的参数模式进行处理
    result = prop_index(
        OpSchema(
            op=op_schema.op,
            args_schema=(values_spec, all_indices_spec),
            kwargs_schema=op_schema.kwargs_schema,
        )
    )

    # 如果结果需要重新分布模式
    if result.redistribute_schema:
        schema_suggestion = result.redistribute_schema
        # 更新结果的重新分布模式，根据当前维度调整参数模式
        result.redistribute_schema = OpSchema(
            op=op_schema.op,
            args_schema=(
                schema_suggestion.args_schema[0],
                dim,
                schema_suggestion.args_schema[1][dim],
            ),
            kwargs_schema=op_schema.kwargs_schema,
        )

    # 返回处理后的结果
    return result
    # 如果条件不成立，触发断言错误，确保 indices_out.redistribute_schema 不为空
    else:
        assert indices_out.redistribute_schema is not None
        # 将 indices_out.redistribute_schema 赋值给 valid_indices_suggestion
        valid_indices_suggestion = indices_out.redistribute_schema
        # 遍历 valid_indices_suggestion.args_spec 中的每个元素，将其与 valid_indices_spec 对应位置的值组成 multi_indices_spec 字典
        for i, v in enumerate(valid_indices_suggestion.args_spec):
            multi_indices_spec[valid_indices_spec[i][0]] = v
        # 重新调用 pointwise_rule 函数获取 indices_spec，用于计算理想的 values_spec
        indices_output_spec = pointwise_rule(valid_indices_suggestion).output_spec
        # 确保 indices_output_spec 是 DTensorSpec 类型
        assert isinstance(indices_output_spec, DTensorSpec)
        # 将 indices_output_spec 赋值给 indices_spec
        indices_spec = indices_output_spec

    # 创建 lookup_dims 集合，包含 valid_indices_spec 中每个元组的第一个元素
    lookup_dims = {v[0] for v in valid_indices_spec}

    # 计算 need_reshard_on_values 元组，判断是否需要在 values_spec.placements 上重新分片
    need_reshard_on_values = tuple(
        (isinstance(vp, Shard) and (vp.dim in lookup_dims or isinstance(ip, Shard)))
        for vp, ip in zip(values_spec.placements, indices_spec.placements)
    )

    # 如果不需要在 indices 和 values 上重新分片，则直接返回原始的 values_spec.placements
    if not need_reshard_on_indices and not any(need_reshard_on_values):
        value_placements = values_spec.placements

        # 检查 valid_indices_spec 中的所有维度是否连续
        all_dims_consecutive = all(
            b[0] - a[0] == 1
            for b, a in zip(valid_indices_spec[1:], valid_indices_spec[:-1])
        )
        # 如果所有维度都连续，将插入维度设置为第一个索引的维度
        if all_dims_consecutive:
            insert_dim: int = valid_indices_spec[0][0]
        else:
            # 否则，在第一个维度上进行插入
            insert_dim = 0

        # 定义 place 函数，根据 vp 和 ip 的类型返回相应的 Shard 或者保持不变
        def place(vp: Placement, ip: Placement) -> Placement:
            if isinstance(vp, Shard):
                return Shard(
                    vp.dim
                    if vp.dim < insert_dim
                    else vp.dim + indices_spec.ndim - sum(1 if vp.dim > v[0] else 0 for v in valid_indices_spec)
                )
            if isinstance(ip, Shard):
                return Shard(ip.dim + insert_dim)
            # Partial or Replicated
            return vp

        # 应用 place 函数得到更新后的 value_placements 元组
        value_placements = tuple(
            place(vp, ip)
            for vp, ip in zip(values_spec.placements, indices_spec.placements)
        )

        # 创建 OutputSharding 对象作为结果，使用更新后的 value_placements 和原始的 values_spec.mesh
        result = OutputSharding(
            output_spec=DTensorSpec(
                mesh=values_spec.mesh,
                placements=value_placements,
            )
        )
        return result
    else:
        result = OutputSharding(  # 创建一个 OutputSharding 对象并赋值给 result 变量
            output_spec=None,  # 设置 OutputSharding 对象的 output_spec 为 None
            redistribute_schema=OpSchema(  # 设置 OutputSharding 对象的 redistribute_schema 属性为 OpSchema 对象
                op=op_schema.op,  # 从 op_schema 对象中获取 op 属性，并赋给 OpSchema 对象的 op 属性
                args_schema=(  # 设置 OpSchema 对象的 args_schema 属性为一个元组
                    DTensorSpec(  # 创建一个 DTensorSpec 对象
                        mesh=values_spec.mesh,  # 将 values_spec 对象的 mesh 属性赋给 DTensorSpec 对象的 mesh 属性
                        placements=tuple(  # 将以下内容作为元组赋给 DTensorSpec 对象的 placements 属性
                            [
                                Replicate() if need_reshard_on_values[i] else v  # 根据条件创建 Replicate() 对象或直接使用 values_spec.placements 中的值，并放入列表中
                                for i, v in enumerate(values_spec.placements)  # 遍历 values_spec.placements 中的每个元素及其索引
                            ]
                        ),
                        tensor_meta=values_spec.tensor_meta,  # 将 values_spec 对象的 tensor_meta 属性赋给 DTensorSpec 对象的 tensor_meta 属性
                    ),
                    multi_indices_spec,  # 将 multi_indices_spec 对象直接放入 args_schema 元组中
                ),
                kwargs_schema=op_schema.kwargs_schema,  # 将 op_schema 对象的 kwargs_schema 属性赋给 OpSchema 对象的 kwargs_schema 属性
            ),
        )
        return result  # 返回 result 变量，即创建的 OutputSharding 对象
@register_prop_rule(
    [
        aten.split.Tensor,  # 注册操作规则，支持对Tensor对象的split操作
        aten.split_with_sizes.default,  # 默认的split_with_sizes函数
        aten.split_with_sizes_copy.default,  # 默认的split_with_sizes_copy函数
    ],
    schema_info=RuntimeSchemaInfo(1),  # 设置操作规则的运行时模式信息
)
def split_rule(op_schema: OpSchema) -> OutputSharding:
    output_spec_list: List[DTensorSpec] = []  # 初始化输出规格列表，用于存储分割后的Tensor规格信息
    input_spec = cast(DTensorSpec, op_schema.args_schema[0])  # 获取输入Tensor的规格信息
    ndim = input_spec.ndim  # 获取输入Tensor的维度数
    split_size_or_sections = op_schema.args_schema[1]  # 获取分割尺寸或分割段数
    dim = cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else 0  # 获取分割的维度，默认为0
    dim = normalize_dim(dim, ndim)  # 根据输入的维度信息和Tensor的实际维度数，标准化分割维度

    # TODO: tensor to split cannot have Partial
    # in its placements for now. Will need to
    # support in future.
    if input_spec.sums:  # 如果输入的Tensor规格中有sums属性
        raise NotImplementedError(
            f"splitting distributed tensor with "
            f"Partial placement is not implemented!\n"
            f"DTensorSpec={input_spec}"
        )

    # TODO: just like slice op, split replicates before
    # splitting on a sharded dimension
    need_reshard = False  # 初始化标志，表示是否需要重新分片
    if is_tensor_dim_sharded(input_spec, dim=dim):  # 如果输入的Tensor在指定维度上是分片的
        need_reshard = True  # 设置需要重新分片的标志
        # 更新输入规格，移除在指定维度上的分片信息
        input_spec = DTensorSpec(
            mesh=input_spec.mesh,
            placements=unshard_tensor_dim(input_spec.placements, dim=dim),
            tensor_meta=input_spec.tensor_meta,
        )

    if need_reshard:  # 如果需要重新分片
        return OutputSharding(
            None,
            redistribute_schema=OpSchema(
                op=op_schema.op,
                args_schema=(input_spec,) + op_schema.args_schema[1:],  # 更新操作模式的参数列表
                kwargs_schema=op_schema.kwargs_schema,
            ),
        )

    def size_split(N, i):
        # Last chunk will be smaller if the tensor size N
        # along the given dimension dim is not divisible by i.
        assert i > 0  # 确保分割段数大于0
        # 根据给定维度dim上的Tensor尺寸N和分割段数i，计算每个分割块的尺寸列表
        return [i] * (N // i) + ([N % i] if N % i != 0 else [])

    # 根据分割尺寸或分割段数的类型，生成输出尺寸列表
    output_size_list = (
        size_split(input_spec.shape[dim], split_size_or_sections)
        if isinstance(split_size_or_sections, int)
        else split_size_or_sections
    )
    # 根据输出尺寸列表，生成对应数量的Tensor规格信息并存储到输出规格列表中
    output_spec_list = [
        DTensorSpec(
            mesh=input_spec.mesh,
            placements=input_spec.placements,
        )
        for _ in range(len(output_size_list))
    ]
    return OutputSharding(output_spec_list)  # 返回输出规格列表作为输出的分割结果
```