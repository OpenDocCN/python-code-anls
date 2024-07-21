# `.\pytorch\torch\distributed\_tensor\ops\math_ops.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入数学模块
import math
# 导入用于定义数据类的模块
from dataclasses import dataclass
# 导入枚举类型的模块
from enum import Enum
# 导入类型提示相关的模块
from typing import cast, List, Optional, Sequence, Tuple, Union

# 导入PyTorch库
import torch
# 导入PyTorch分布式张量操作相关的模块
from torch.distributed._tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    RuntimeSchemaInfo,
    TupleStrategy,
)
# 导入PyTorch分布式张量操作工具函数
from torch.distributed._tensor.ops.utils import (
    as_list,
    expand_to_full_mesh_op_strategy,
    generate_redistribute_costs,
    is_tensor_evenly_shardable,
    normalize_dim,
    normalize_dims,
    normalize_to_torch_size,
    register_op_strategy,
)
# 导入PyTorch分布式张量的位置类型相关模块
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
)
# 导入PyTorch设备网格模块
from torch.distributed.device_mesh import DeviceMesh

# 使用torch.ops.aten作为aten的别名
aten = torch.ops.aten

# 定义枚举类型Reduction，包含了几种归约操作类型
class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2

# 定义数据类NormReduction，用于描述归一化操作类型
@dataclass(frozen=True)
class NormReduction:
    norm_type: Union[int, float, str]

# 定义归约操作类型ReductionOpType，可以是NormReduction对象或字符串
ReductionOpType = Union[NormReduction, str]

# 定义数据类_NormPartial，继承自Partial类，用于部分向量范数的计算
@dataclass(frozen=True)
class _NormPartial(Partial):
    """
    This placement is used for partial vector norm.
    
    For p-norms (where p not inf or -inf), the p-norm over n elements computes
        (sum_i x_i^p)^(1/p)
    where the sum is from i=1 to n. The reduction op is the p-norm itself.
    For example, consider 2 ranks, a (4,) tensor sharded on dim-0, and 2-norm:
        Rank 0: [t1, t2] | Rank 1: [t3, t4]
    After computing 2-norm per gradient (partial placement):
        Rank 0: [sqrt(t1^2 + t2^2)] | Rank 1: [sqrt(t3^2 + t4^2)]
    Converting from partial to replicate wants to ultimately get:
        Rank 0/1: [sqrt(t1^2 + t2^2 + t3^2 + t4^2)]
    This can be achieved by computing 2-norm on each rank's result. This holds
    similarly for inf and -inf norm. For 0-norm, the reduction op is sum.
    """

    norm_type: Union[int, float, str] = 2

    def __post_init__(self):
        """Set the appropriate reduce op based on the norm type."""
        # 使用`object.__setattr__`绕过数据类冻结检查
        if self.norm_type in (float("inf"), "inf"):
            object.__setattr__(self, "reduce_op", "max")
        elif self.norm_type in (float("-inf"), "-inf"):
            object.__setattr__(self, "reduce_op", "min")
        elif isinstance(self.norm_type, (int, float)):
            object.__setattr__(self, "reduce_op", "sum")
        else:
            raise NotImplementedError(f"Unsupported norm type: {self.norm_type}")

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    def __eq__(self, other: object) -> bool:
        # 检查是否是相同类型的 _NormPartial 对象，并比较 norm_type 属性是否相同
        if not isinstance(other, _NormPartial):
            return False
        return self.norm_type == other.norm_type

    def __hash__(self) -> int:
        # 返回基于 norm_type 属性计算的哈希值加一
        return 1 + hash(self.norm_type)
# 推断减少维度的方式，并返回一个整数列表或空值
def _infer_reduction_dims(dims_arg: object, ndim: int) -> Optional[List[int]]:
    if dims_arg is None:
        return None
    # 将dims_arg转换为整数列表
    dims = cast(List[int], as_list(dims_arg))
    # 标准化维度列表，确保维度在合法范围内
    dims = cast(List[int], normalize_dims(dims, ndim))
    # 定义特殊的空维度列表
    empty_dims = [[0], [-1], []]
    # 如果输入维度为0且dims_arg在空维度列表中，则返回空值
    if ndim == 0 and dims_arg in empty_dims:
        return None
    # 否则返回推断的维度列表
    return dims


def _infer_reduce_dims_map(
    reduction_dims: List[int], input_ndim: int, keep_dim=False
) -> List[int]:
    reduction_dims_map = []
    new_dim_count = 0
    # 遍历输入的维度数量
    for input_dim in range(input_ndim):
        # 如果输入维度在减少维度列表中且不保留维度
        if input_dim in reduction_dims and not keep_dim:
            # 将其映射为-1
            reduction_dims_map.append(-1)
        else:
            # 否则按顺序映射为新维度
            reduction_dims_map.append(new_dim_count)
            new_dim_count += 1

    return reduction_dims_map


def _replicate_dims_start_at(
    placements: Sequence[Placement], start_dim: int = 0
) -> Tuple[Placement, ...]:
    new_placements: List[Placement] = []
    # 遍历给定的放置顺序
    for p in placements:
        # 如果是部分放置或者是Shard类型且维度大于等于指定的起始维度
        if p.is_partial() or (isinstance(p, Shard) and p.dim >= start_dim):
            # 替换为Replicate类型的放置
            new_placements.append(Replicate())
        else:
            # 保持原来的放置类型不变
            new_placements.append(p)
    return tuple(new_placements)


# 返回与给定的放置顺序相匹配的新放置顺序，但跳过指定的维度
def _skip_dim(
    placements: Tuple[Placement, ...], skipped_dim: int
) -> Tuple[Placement, ...]:
    new_placements: List[Placement] = []
    # 遍历给定的放置顺序
    for p in placements:
        # 如果是Shard类型且维度大于等于被跳过的维度
        if isinstance(p, Shard) and p.dim >= skipped_dim:
            # 将维度减一后保持为Shard类型的放置
            new_placements.append(Shard(p.dim - 1))
        else:
            # 保持原来的放置类型不变
            new_placements.append(p)
    return tuple(new_placements)


def replicate_reduction_dims(
    placements: Tuple[Placement, ...], reduction_dims: List[int]
) -> Tuple[Placement, ...]:
    # 如果不是线性减少，则复制减少的维度
    new_placements: List[Placement] = []

    for p in placements:
        # 如果是部分放置或者是Shard类型且维度在减少维度列表中
        if p.is_partial() or isinstance(p, Shard) and p.dim in reduction_dims:
            # 替换为Replicate类型的放置
            new_placements.append(Replicate())
        else:
            # 保持原来的放置类型不变
            new_placements.append(p)

    return tuple(new_placements)


def map_placements_after_reduction(
    placements: Tuple[Placement, ...],
    reduction_dims: List[int],
    reduction_dims_map: List[int],
    reduction_op: ReductionOpType,
) -> Tuple[Placement, ...]:
    """
    根据减少操作后的输出形状，映射每个放置位置。
    """
    new_placements: List[Placement] = []
   `
    for placement in placements:
        # 遍历输入的placements列表
        if isinstance(placement, (Replicate, Partial)):
            # 如果当前placement是Replicate或Partial类型，则直接添加到new_placements列表中
            new_placements.append(placement)
        else:
            # 如果当前placement是Shard类型
            assert isinstance(placement, Shard)
            # 断言当前placement确实是Shard类型
            shard_dim = placement.dim
            # 获取当前Shard对象的维度信息
            new_shard_dim = reduction_dims_map[shard_dim]
            # 查找reduction_dims_map字典中与当前shard_dim对应的新维度信息
            if new_shard_dim == -1 or shard_dim in reduction_dims:
                # 如果新维度信息为-1（表示折叠）或者shard_dim在reduction_dims中
                # （即对于keepdims=True的情况），生成一个部分对象并添加到new_placements列表中
                new_placements.append(get_placement_from_reduction_op(reduction_op))
            else:
                # 否则，创建一个新的Shard对象，使用新的shard维度信息，并添加到new_placements列表中
                new_placements.append(Shard(new_shard_dim))
    # 返回新的placements列表作为元组
    return tuple(new_placements)
# 根据给定的规约操作类型获取对应的放置策略
def get_placement_from_reduction_op(reduction_op: ReductionOpType) -> Placement:
    # 如果规约操作是 NormReduction 类型，则返回对应的 _NormPartial 对象
    if isinstance(reduction_op, NormReduction):
        return _NormPartial(norm_type=reduction_op.norm_type)
    # 否则返回一个 Partial 对象，使用给定的 reduction_op
    return Partial(reduction_op)


def common_reduction_strategy(
    mesh: DeviceMesh,
    input_strategy: OpStrategy,
    reduce_dims: List[int],
    keep_dim: bool = False,
    reduction_linear: bool = True,
    reduction_op: ReductionOpType = "sum",
) -> OpStrategy:
    """
    reduction_linear means that the reduction `f` follows this rule:
        f([f(a), f(b)]) = f([a, b])

    reduction linear should be super set of linearity.
    """
    # 默认情况下遵循规约输入策略
    reduction_strategy = OpStrategy([])

    # 遍历输入策略中的各个子策略
    for strtg in input_strategy.strategies:
        if not reduction_linear:
            # 如果不是线性规约，清除规约维度上的挂起的求和和分片
            # 使用 replicate_reduction_dims 函数处理 strtg.output_spec.placements
            input_placements = replicate_reduction_dims(
                strtg.output_spec.placements, reduce_dims
            )
        else:
            # 如果是线性规约，直接使用 strtg.output_spec.placements
            input_placements = strtg.output_spec.placements

        # 创建 DTensorSpec 对象来描述输入规约操作的规约维度和放置策略
        input_spec = DTensorSpec(
            mesh=mesh,
            placements=input_placements,
            tensor_meta=strtg.output_spec.tensor_meta,
        )

        # 推断规约维度映射
        reduce_dims_map = _infer_reduce_dims_map(reduce_dims, input_spec.ndim, keep_dim)

        # 映射规约后的放置策略
        out_placements = map_placements_after_reduction(
            input_spec.placements, reduce_dims, reduce_dims_map, reduction_op
        )

        # 生成重新分配成本的列表
        redistribute_cost = [generate_redistribute_costs(input_strategy, input_spec)]

        # 将生成的放置策略添加到 reduction_strategy 中
        reduction_strategy.strategies.append(
            PlacementStrategy(
                output_specs=DTensorSpec(
                    mesh=mesh,
                    placements=out_placements,
                ),
                input_specs=(input_spec,),
                redistribute_cost=redistribute_cost,
            )
        )

    return reduction_strategy


# 线性规约操作的映射关系字典
LINEAR_REDUCTION_OP_MAP = {
    aten.all.default: "sum",
    aten.all.dim: "sum",
    aten.sum.default: "sum",
    aten.sum.dim_IntList: "sum",
    aten.prod.default: "product",
    aten.prod.dim_int: "product",
    aten.prod.int_out: "product",
    aten.mean.default: "avg",
    aten.mean.dim: "avg",
    aten.mean.out: "avg",
    aten.max.default: "max",
    aten.max.dim: "max",
    aten.max.out: "max",
    aten.min.default: "min",
    aten.min.dim: "min",
    aten.min.out: "min",
}

# 注册线性规约策略的函数装饰器
@register_op_strategy(
    list(LINEAR_REDUCTION_OP_MAP.keys()), schema_info=RuntimeSchemaInfo(1)
)
def linear_reduction_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 获取操作模式的参数模式
    args_schema = op_schema.args_schema
    # 第一个参数应为 OpStrategy 类型
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    dims = None
    # 如果参数模式大于 1，推断规约维度
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_strategy.ndim)
    # 如果未提供dims参数，则使用输入策略的维度列表作为减少维度的依据
    reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims
    
    # 检查操作模式参数列表是否大于2，并且第三个参数值为真，则保持维度
    keep_dim = len(op_schema.args_schema) > 2 and bool(op_schema.args_schema[2])
    
    # 根据操作模式确定线性减少操作的具体方法
    reduction_op = LINEAR_REDUCTION_OP_MAP[op_schema.op]
    
    # 调用通用的减少策略函数，传入相关参数进行减少维度操作
    return common_reduction_strategy(
        mesh,
        input_strategy,
        reduce_dims,
        keep_dim=keep_dim,
        reduction_linear=True,
        reduction_op=reduction_op,
    )
# 注册操作的策略函数，针对变量的纠正操作和纠正输出操作
@register_op_strategy(
    [aten.var.correction, aten.var.correction_out],
    schema_info=RuntimeSchemaInfo(1, ["keepdim"]),
)
def var_reduction_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 获取操作模式的参数模式
    args_schema = op_schema.args_schema
    # 获取输入策略，确保其为 OpStrategy 类型
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    dims = None
    # 如果参数模式长度大于 1，推断减少维度的维度
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_strategy.ndim)

    # 如果未指定减少的维度，则默认为输入策略的所有维度
    reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims

    # 获取 keepdim 参数，如果未指定则为 False
    keep_dim = cast(bool, op_schema.kwargs_schema.get("keepdim", False))

    # 返回通用的减少维度策略
    return common_reduction_strategy(
        mesh, input_strategy, reduce_dims, keep_dim=keep_dim, reduction_linear=False
    )


# 注册操作的策略函数，针对向量范数的线性代数操作
@register_op_strategy(
    [aten.linalg_vector_norm.default], schema_info=RuntimeSchemaInfo(1)
)
def vector_norm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 获取操作模式的参数模式
    args_schema = op_schema.args_schema
    # 获取输入策略，确保其为 OpStrategy 类型
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    # 获取范数类型参数，如果未指定则为 2
    norm_type = args_schema[1] if len(args_schema) > 1 else 2
    assert isinstance(norm_type, (int, float, str)), f"{norm_type}"
    # 获取维度参数，如果未指定则为 None
    dim = args_schema[2] if len(args_schema) > 2 else None
    # 获取 keepdim 参数，如果未指定则为 False
    keepdim = args_schema[3] if len(args_schema) > 3 else False
    # 推断减少的维度，如果未指定则为输入策略的所有维度
    dims = _infer_reduction_dims(dim, input_strategy.ndim)
    reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims
    # 返回通用的减少维度策略，用于范数计算
    return common_reduction_strategy(
        mesh,
        input_strategy,
        reduce_dims,
        keep_dim=cast(bool, keepdim),
        reduction_linear=True,
        reduction_op=NormReduction(norm_type),
    )


# 注册操作的策略函数，针对标量的批处理范数计算
@register_op_strategy(
    [aten._foreach_norm.Scalar], schema_info=RuntimeSchemaInfo(1, needs_pytree=True)
)
def foreach_norm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> TupleStrategy:
    # 获取操作模式的参数模式
    args_schema = op_schema.args_schema
    # 获取输入元组策略，确保其为 TupleStrategy 类型
    input_tuple_strategy = args_schema[0]
    assert isinstance(input_tuple_strategy, TupleStrategy)
    # 获取范数类型参数
    norm_type = args_schema[1]
    assert isinstance(norm_type, (int, float, str)), f"{norm_type}"
    # 初始化输出元组策略的子策略列表
    output_tuple_strategy_childs: List[OpStrategy] = []
    # 遍历输入元组的子策略
    for op_strategy in input_tuple_strategy.childs:
        assert isinstance(op_strategy, OpStrategy), f"{op_strategy}"
        # 对每个子策略进行减少维度的通用策略计算
        reduce_dims = list(range(op_strategy.ndim))
        output_strategy = common_reduction_strategy(
            mesh,
            op_strategy,
            reduce_dims,
            reduction_linear=True,
            reduction_op=NormReduction(norm_type),
        )
        output_tuple_strategy_childs.append(output_strategy)
    # 返回包含输出子策略的元组策略
    return TupleStrategy(output_tuple_strategy_childs)


# 注册操作的策略函数，针对奇异值分解操作
@register_op_strategy([aten._linalg_svd.default], schema_info=RuntimeSchemaInfo(1))
def linalg_svd_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 由于没有简单的方式来计算分片的 SVD，始终使用复制的方式
    args_schema = op_schema.args_schema
    # 获取输入策略
    input_strategy = args_schema[0]
    # 断言输入的input_strategy是OpStrategy的实例，如果不是，则抛出带有input_strategy字符串表示的异常信息
    assert isinstance(input_strategy, OpStrategy), f"{input_strategy}"
    
    # 初始化一个空列表output_strategies，用于存放输出的PlacementStrategy对象
    output_strategies: List[PlacementStrategy] = []
    
    # 遍历输入策略input_strategy中的所有策略
    for placement_strategy in input_strategy.strategies:
        
        # 创建一个包含mesh.ndim个Replicate对象的元组replicate_placements
        replicate_placements = tuple(Replicate() for _ in range(mesh.ndim))
        
        # 创建一个新的DTensorSpec对象replicate_spec，设置其参数：
        # - mesh为给定的mesh对象
        # - placements为前面创建的replicate_placements元组
        # - tensor_meta来自placement_strategy的output_spec中的tensor_meta
        replicate_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_placements,
            tensor_meta=placement_strategy.output_spec.tensor_meta,
        )
        
        # 生成一个关于重分布成本的列表redistribute_cost，调用generate_redistribute_costs函数
        redistribute_cost = [
            generate_redistribute_costs(input_strategy, replicate_spec)
        ]
        
        # 创建一个新的PlacementStrategy对象replicate_strategy，设置其参数：
        # - output_specs为replicate_spec对象
        # - input_specs为包含replicate_spec对象的元组
        # - redistribute_cost为前面生成的redistribute_cost列表
        replicate_strategy = PlacementStrategy(
            output_specs=replicate_spec,
            input_specs=(replicate_spec,),
            redistribute_cost=redistribute_cost,
        )
        
        # 将新创建的replicate_strategy对象添加到output_strategies列表中
        output_strategies.append(replicate_strategy)
    
    # 返回一个新的OpStrategy对象，其包含前面生成的output_strategies列表
    return OpStrategy(output_strategies)
# 注册操作策略函数，用于处理 _log_softmax.default 和 _softmax.default 操作
# schema_info 指定了运行时架构信息为 1
@register_op_strategy(
    [aten._log_softmax.default, aten._softmax.default], schema_info=RuntimeSchemaInfo(1)
)
def softmax_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 从操作模式中获取输入策略、softmax 维度和一个占位符
    input_strategy, softmax_dim, _ = op_schema.args_schema
    input_strategy = cast(OpStrategy, input_strategy)  # 强制类型转换为 OpStrategy 类型
    softmax_dim = cast(int, softmax_dim)  # 强制类型转换为整数
    softmax_dim = normalize_dim(softmax_dim, input_strategy.ndim)  # 规范化 softmax 维度

    # 创建一个空的输出策略对象
    output_strategy = OpStrategy([])
    # 遍历输入策略中的每一个策略
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        redistribute_costs = []  # 初始化重分配成本列表
        input_src_spec = input_placement_strategy.output_spec  # 获取输入策略的输出规格

        # 确保输入在 softmax 维度上被复制
        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_reduction_dims(
                input_src_spec.placements, [softmax_dim]
            ),  # 在输入源规格的放置方案上复制 softmax 维度
            tensor_meta=input_src_spec.tensor_meta,  # 使用相同的张量元数据
        )
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_target_spec)
        )  # 生成重分配成本并添加到列表中
        output_target_spec = input_target_spec  # 输出目标规格与输入目标规格相同
        output_strategy.strategies.append(
            PlacementStrategy(
                output_specs=output_target_spec,  # 输出规格为目标规格
                input_specs=[input_target_spec],  # 输入规格为输入目标规格
                redistribute_cost=redistribute_costs,  # 重分配成本为之前生成的成本列表
            )
        )

    return output_strategy  # 返回输出策略对象


# 注册操作策略函数，用于处理 _log_softmax_backward_data.default 和 _softmax_backward_data.default 操作
# schema_info 指定了运行时架构信息为 2
@register_op_strategy(
    [
        aten._log_softmax_backward_data.default,
        aten._softmax_backward_data.default,
    ],
    schema_info=RuntimeSchemaInfo(2),
)
def softmax_backward_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 从操作模式中获取梯度输出策略、输出策略、softmax 维度和一个占位符
    grad_out_strategy, out_strategy, softmax_dim, _ = op_schema.args_schema
    grad_out_strategy = cast(OpStrategy, grad_out_strategy)  # 强制类型转换为 OpStrategy 类型
    out_strategy = cast(OpStrategy, out_strategy)  # 强制类型转换为 OpStrategy 类型
    softmax_dim = cast(int, softmax_dim)  # 强制类型转换为整数
    softmax_dim = normalize_dim(softmax_dim, grad_out_strategy.ndim)  # 规范化 softmax 维度

    # 创建一个空的梯度输入策略对象
    grad_in_strategy = OpStrategy([])
    # 遍历梯度输出策略和输出策略中的每一对策略
    for grad_out_placement_strat, out_placement_strat in zip(
        grad_out_strategy.strategies, out_strategy.strategies
    ):
        # 根据梯度输出或者输出的分片数量较多的那个来确定源规格
        grad_out_src_spec = grad_out_placement_strat.output_spec
        out_src_spec = out_placement_strat.output_spec
        src_spec = (
            grad_out_src_spec
            if grad_out_src_spec.num_shards >= out_src_spec.num_shards
            else out_src_spec
        )

        # 确保输入沿着 softmax 维度进行复制
        tgt_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_reduction_dims(src_spec.placements, [softmax_dim]),
        )
        # 生成重新分配成本，用于梯度输出策略和输出策略
        redist_grad_out_cost = generate_redistribute_costs(grad_out_strategy, tgt_spec)
        redist_out_cost = generate_redistribute_costs(out_strategy, tgt_spec)
        # 将生成的策略添加到梯度输入策略中
        grad_in_strategy.strategies.append(
            PlacementStrategy(
                output_specs=tgt_spec,
                redistribute_cost=[redist_grad_out_cost, redist_out_cost],
            )
        )

    # 返回梯度输入策略
    return grad_in_strategy
@register_op_strategy(
    [aten.nll_loss_forward.default, aten.nll_loss2d_forward.default],
    schema_info=RuntimeSchemaInfo(3),
)
def nll_loss_forward_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 确保操作模式的参数数量为5
    assert len(op_schema.args_schema) == 5
    (
        input_strategy,
        target_strategy,
        weight_strategy,
        reduction,
        _,  # 使用下划线占位未使用的变量
    ) = op_schema.args_schema
    input_strategy = cast(OpStrategy, input_strategy)  # 强制类型转换为 OpStrategy
    target_strategy = cast(OpStrategy, target_strategy)  # 强制类型转换为 OpStrategy
    reduction = cast(int, reduction)  # 强制类型转换为整数类型

    input_shape = input_strategy.shape  # 获取输入数据的形状
    channel_dim = 1 if len(input_shape) >= 2 else 0  # 确定通道维度位置

    output_strategy = OpStrategy([])  # 创建一个空的操作策略对象作为输出
    return output_strategy


@register_op_strategy(
    [aten.nll_loss_backward.default, aten.nll_loss2d_backward.default],
    schema_info=RuntimeSchemaInfo(4),
)
def nll_loss_backward_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 确保操作模式的参数数量为7
    assert len(op_schema.args_schema) == 7
    (
        grad_out_strategy,
        input_strategy,
        target_strategy,
        weight_strategy,
        reduction,
        _,  # 使用下划线占位未使用的变量
        total_weight_strategy,
    ) = op_schema.args_schema
    grad_out_strategy = cast(OpStrategy, grad_out_strategy)  # 强制类型转换为 OpStrategy
    input_strategy = cast(OpStrategy, input_strategy)  # 强制类型转换为 OpStrategy
    target_strategy = cast(OpStrategy, target_strategy)  # 强制类型转换为 OpStrategy
    reduction = cast(int, reduction)  # 强制类型转换为整数类型
    total_weight_strategy = cast(OpStrategy, total_weight_strategy)  # 强制类型转换为 OpStrategy

    input_shape = input_strategy.shape  # 获取输入数据的形状
    channel_dim = 1 if len(input_shape) >= 2 else 0  # 确定通道维度位置

    grad_in_strategy = OpStrategy([])  # 创建一个空的操作策略对象作为梯度输入
    return grad_in_strategy


@register_op_strategy(
    [aten.native_layer_norm.default],
    schema_info=RuntimeSchemaInfo(1),
)
def layer_norm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # args must be: input, normalized_shape, weight, bias, eps
    # for None weight and bias, their corresponding objects will
    # be None as well. layer_norm_strategy returns one OpStrategy
    # for the triple return values (out, mean, rstd).
    # 确保操作模式的参数数量为5
    assert len(op_schema.args_schema) == 5
    (
        input_strategy,
        normalized_shape,
        weight_strategy,
        bias_strategy,
        _,  # 使用下划线占位未使用的变量
    ) = op_schema.args_schema

    # 当前的 layer norm 实现要求所有输入的 DTensor 分片必须以 OpStrategy 形式存在
    assert isinstance(input_strategy, OpStrategy)  # 断言输入策略是 OpStrategy 类型
    assert isinstance(normalized_shape, (int, Sequence, torch.Size))  # 断言标准化形状是 int、Sequence 或 torch.Size 类型
    normalized_size = normalize_to_torch_size(normalized_shape)  # 标准化为 torch.Size 格式

    input_ndim = input_strategy.ndim  # 获取输入数据的维度
    axis = input_ndim - len(normalized_size)  # 计算轴的位置

    # 使用 OpStrategy 因为输出 (out, mean, rstd) 应该具有相同的位置分布
    output_strategy = OpStrategy([])  # 创建一个空的操作策略对象作为输出
    # 遍历输入策略中的每个策略及其索引
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        # 初始化操作的目标规格列表和重新分发成本列表
        op_args_target_specs = []
        redistribute_costs = []

        # 获取当前输入策略的输出规格
        input_src_spec = input_placement_strategy.output_spec

        # 对于输入张量，在必要时在内部维度上进行复制
        # TODO: 一旦我们找出如何分解层规范，就可以避免强制重新分发
        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_target_spec)
        )

        if weight_strategy is not None:
            assert isinstance(weight_strategy, OpStrategy)
            weight_src_spec = weight_strategy.strategies[idx].output_spec

            # 对于权重张量，在必要时在所有维度上进行复制
            # TODO: 一旦我们找出如何分解层规范，就可以避免强制重新分发
            weight_target_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            op_args_target_specs.append(weight_target_spec)
            redistribute_costs.append(
                generate_redistribute_costs(weight_strategy, weight_target_spec)
            )

        if bias_strategy is not None:
            assert isinstance(bias_strategy, OpStrategy)
            bias_src_spec = bias_strategy.strategies[idx].output_spec

            # 对于偏置张量，在必要时在所有维度上进行复制
            # TODO: 一旦我们找出如何分解层规范，就可以避免强制重新分发
            bias_target_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(bias_src_spec.placements),
                tensor_meta=bias_src_spec.tensor_meta,
            )
            op_args_target_specs.append(bias_target_spec)
            redistribute_costs.append(
                generate_redistribute_costs(bias_strategy, bias_target_spec)
            )

        # 输出规格与输入规格相同
        output_target_spec = input_target_spec

        # 将输出规格添加到输出策略的策略列表中
        output_strategy.strategies.append(
            PlacementStrategy(
                output_specs=output_target_spec,
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    # 返回输出策略
    return output_strategy
# 注册操作策略函数，用于处理 layer_norm_backward 操作
@register_op_strategy(
    [aten.native_layer_norm_backward.default],
    schema_info=RuntimeSchemaInfo(2),
)
def layer_norm_bwd_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 确保操作模式的参数个数为 8
    assert len(op_schema.args_schema) == 8
    (
        grad_out_strategy,
        input_strategy,
        normalized_shape,
        mean_strategy,
        rstd_strategy,
        weight_strategy,
        bias_strategy,
        output_mask,
    ) = op_schema.args_schema

    # 确保参数的类型正确
    assert isinstance(grad_out_strategy, OpStrategy)
    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(mean_strategy, OpStrategy)
    assert isinstance(rstd_strategy, OpStrategy)

    # 确保 normalized_shape 的类型是 int、Sequence 或 torch.Size
    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    # 将 normalized_shape 规范化为 torch.Size 对象
    normalized_size = normalize_to_torch_size(normalized_shape)
    # 获取 input_strategy 的维度
    input_ndim = input_strategy.ndim
    # 计算 axis，即 normalized_size 所对应的轴数
    axis = input_ndim - len(normalized_size)
    # 获取外部维度列表
    outer_dims = list(range(axis))

    # 确保 output_mask 是一个包含 3 个元素的列表
    assert isinstance(output_mask, List) and len(output_mask) == 3

    # 创建并返回空的 OpStrategy 对象，表示没有任何具体的策略输出
    out_tuple_strategy = OpStrategy([])
    return out_tuple_strategy


# 注册操作策略函数，用于处理 topk 操作
@register_op_strategy(
    [aten.topk.default],
    schema_info=RuntimeSchemaInfo(2),
)
def topk_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # 获取输入参数的策略对象和 k 的值
    input_strategy = cast(OpStrategy, op_schema.args_schema[0])
    k = cast(int, op_schema.args_schema[1])
    # 获取输入参数的形状信息
    input_shape = input_strategy.shape
    # 获取 topk_dim 的值，如果不存在则默认为 -1
    topk_dim = (
        cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else -1
    )
    # 规范化 topk_dim，确保其在输入策略的有效维度范围内
    topk_dim = normalize_dim(topk_dim, input_strategy.ndim)

    # 单个网格维度策略列表
    single_mesh_dim_strategies = []

    # 对于两个输出 (values, indices) 和一个输入，使用 Replicate 策略
    all_replicate: List[Placement] = [Replicate()] * 3
    single_mesh_dim_strategies.append(all_replicate)

    # 对除 topk_dim 外的每个维度应用 Shard 策略
    for dim in range(input_strategy.ndim):
        if dim != topk_dim:
            dim_shardings: List[Placement] = [Shard(dim)] * 3
            single_mesh_dim_strategies.append(dim_shardings)

    # TODO: 针对分片维度的 topk 需要复杂的减少操作，稍后处理

    # 扩展到完整网格操作策略并返回
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=2
    )
```