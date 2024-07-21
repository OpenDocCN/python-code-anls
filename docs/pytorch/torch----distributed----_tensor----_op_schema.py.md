# `.\pytorch\torch\distributed\_tensor\_op_schema.py`

```py
# mypy: allow-untyped-defs
# 引入需要的模块和类型定义
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# 引入 PyTorch 相关模块
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh

# 尝试引入 C++ 扩展模块，如果失败则引入 Python 扩展模块
try:
    from torch.utils._cxx_pytree import tree_leaves, tree_map_only, TreeSpec
except ImportError:
    from torch.utils._pytree import (  # type: ignore[no-redef, assignment]
        tree_leaves,
        tree_map_only,
        TreeSpec,
    )

# Common type aliases
# 定义常见的类型别名
ArgsType = Tuple[object, ...]
KwargsType = Dict[str, object]
# ATen op schemas could have Tensor, Tuple[Tensor] and List[Tensor], so output type sould
# be the same set of possibilities.
# ATen 操作的 schema 可能包含 Tensor、Tuple[Tensor] 和 List[Tensor]，因此输出类型应该是相同的可能性集合。
OutputSpecType = Optional[Union[DTensorSpec, Sequence[Optional[DTensorSpec]]]]


def _rebuild_tensor_from_dtensor_meta(arg) -> object:
    """
    This is used to propagate tensor metadata, must be under fake mode
    """
    # 用于传播张量元数据，必须处于伪模式下
    assert arg.tensor_meta is not None, "DTensorSpec does not contain tensor_meta."
    return torch.empty_strided(
        arg.tensor_meta.shape,
        arg.tensor_meta.stride,
        dtype=arg.tensor_meta.dtype,
    )


def _is_inplace_op(op: OpOverload):
    # 简单分析函数 schema 来确定是否为原地操作的变种
    # 这可能并不完全正确，但目前已足够
    return op._schema.name[-1] == "_"


def _is_out_variant_op(op: OpOverload):
    # 简单分析函数 schema 来确定是否为输出变种操作
    # 这可能并不完全正确，但目前已足够
    return "out" in op._schema.overload_name


def _pretty_print_spec(spec: object) -> str:
    if spec is None:
        return "None"
    elif isinstance(spec, DTensorSpec):
        return "".join([str(p) for p in spec.placements])
    elif isinstance(spec, Sequence):
        return "(" + ", ".join([_pretty_print_spec(s) for s in spec]) + ")"
    else:
        raise RuntimeError(f"Unknown spec type to print: spec={spec}")


@dataclass
class PlacementStrategy:
    """
    A placement strategy describes acceptable sharding placements of the output
    and the tensor arguments of an operation.

    note: when the op return value is a single DTensor object, output_specs is
    DTensorSpec; when the return value is a tuple of Optional[DTensor],
    output_specs is a tuple of Optional[DTensorSpec].
    """

    # 放置策略描述了操作的输出和张量参数的可接受分片位置
    # 当操作返回单个 DTensor 对象时，output_specs 是 DTensorSpec；
    # 当返回值是 Optional[DTensor] 的元组时，output_specs 是 Optional[DTensorSpec] 的元组。
    output_specs: Union[DTensorSpec, Tuple[Optional[DTensorSpec], ...]]
    input_specs: Optional[Sequence[DTensorSpec]] = None

    # redistribute costs for this op placement strategy
    # we need a nested list to record the cost for each
    # operand of this operator, and for each operand of
    # this operator it might have multiple placement strategies
    # 重新分配此操作放置策略的成本
    # 我们需要一个嵌套列表来记录每个操作数的成本，
    # 对于该操作的每个操作数，可能有多个放置策略
    redistribute_cost: Optional[List[List[float]]] = None

    @cached_property
    # 返回该策略的输出规范（DTensorSpec）。要求输出规范必须是一个单独的 DTensorSpec 对象。
    # 如果 output_specs 是一个元组，则抛出异常。
    def output_spec(self) -> DTensorSpec:
        """
        This function requires that the strategy have exactly one DTensorSpec as the
        output spec. If the output_specs is a tuple, we throw an exception.
        """
        if isinstance(self.output_specs, DTensorSpec):
            # 如果 output_specs 是单个的 DTensorSpec 对象，则直接返回它
            return self.output_specs
        else:
            # 如果 output_specs 不是单个对象，则抛出 ValueError 异常
            raise ValueError(
                f"function output_spec expects a single DTensorSpec but got: {self.output_specs}"
            )

    # 返回指定索引处的输入规范（DTensorSpec）。
    # 如果 input_specs 为 None，则会引发 AssertionError。
    # 如果索引超出 input_specs 的长度范围，则会引发 AssertionError。
    def input_spec(self, index: int = 0) -> DTensorSpec:
        assert self.input_specs is not None, "input_specs of PlacementStrategy is None!"
        assert len(self.input_specs) > index, (
            f"Invalid index {index} for input_specs of length "
            f"{len(self.input_specs)}: {self.input_specs}"
        )
        # 返回指定索引处的 DTensorSpec 对象
        return self.input_specs[index]

    # 返回策略对象的字符串表示形式。
    # 如果 input_specs 不为 None，则将其格式化为字符串，否则为空字符串。
    # 将 output_specs 格式化为字符串。
    def __str__(self) -> str:
        if self.input_specs is not None:
            input_specs_str = f"{_pretty_print_spec(self.input_specs)} -> "
        else:
            input_specs_str = ""
        # 格式化输出规范为字符串
        output_spec_str = _pretty_print_spec(self.output_specs)
        # 返回格式化后的字符串表示
        return f"{input_specs_str}{output_spec_str}"
class StrategyType:
    """
    Base class type for op strategy, We have two StrategyType:
        OpStrategy and TupleStrategy
    """
    # 策略类型的基类，定义了两种具体的策略类型：OpStrategy 和 TupleStrategy
    pass


class OpStrategy(StrategyType):
    """
    OpStrategy that consists of a list of placement strategies associated with the op
    """

    def __init__(self, strategies: List[PlacementStrategy]) -> None:
        super().__init__()
        self.strategies: List[PlacementStrategy] = strategies
        # 初始化 OpStrategy 实例，传入一个放置策略的列表

    def __str__(self) -> str:
        strategy_list_str = ", ".join([str(strategy) for strategy in self.strategies])
        mesh_shape = self.mesh_shape
        return f"[{strategy_list_str}] @ mesh: {mesh_shape}"
        # 返回 OpStrategy 实例的字符串表示形式，包括放置策略列表和网格形状信息

    def max_num_shards(self) -> int:
        """
        Returns the max number of shards across all placement strategies
        """
        return max(strategy.output_spec.num_shards for strategy in self.strategies)
        # 返回所有放置策略中的最大分片数

    @property
    def mesh_shape(self):
        output_spec = self.strategies[0].output_specs
        if isinstance(output_spec, DTensorSpec):
            return output_spec.mesh.shape
        else:
            assert isinstance(
                output_spec, tuple
            ), "found no DTensorSpec in the OpStrategy!"
            assert output_spec[0] is not None
            return output_spec[0].mesh.shape
        # 获取 OpStrategy 实例的网格形状属性，处理不同输出规范的情况

    @property
    def ndim(self):
        return self.strategies[0].output_spec.ndim
        # 返回 OpStrategy 实例中第一个放置策略的维度数属性

    @property
    def shape(self):
        return self.strategies[0].output_spec.shape
        # 返回 OpStrategy 实例中第一个放置策略的形状属性


class TupleStrategy(StrategyType):
    """
    TupleStrategy represents the output strategy of this op is a tuple
    of strategy, i.e. If the output of this op is a tuple of tensors or list of tensors
    with possibly different placement strategies, we should return a TupleStrategy that
    contains a tuple of OpStrategy, where each child represents the sharding strategy
    of "each element" of the tuple/list of tensors the op returns.

    NOTE: if the output of the op is a List[Tensor] and they share the same placement
    strategy, then we should return a single OpStrategy instead of a TupleStrategy
    """

    def __init__(self, childs: Sequence[StrategyType]) -> None:
        super().__init__()
        self.childs: Sequence[StrategyType] = childs
        # 初始化 TupleStrategy 实例，传入一个子策略序列

    def __str__(self) -> str:
        child_strategies_str = ", ".join(
            [f"{str(strat)}" for idx, strat in enumerate(self.childs)]
        )
        return f"TupleStrategy({child_strategies_str})"
        # 返回 TupleStrategy 实例的字符串表示形式，包括所有子策略的信息


@dataclass
class RuntimeSchemaInfo:
    """
    RuntimeSchemaInfo stores the operator schema related information for runtime (eager)
    execution. This is mainly used for two ways: 1. to generate hash for args to determine
    whether to re-run sharding prop or not 2. to determine if we need pytree
    """

    # This static_argnum records static arg "starting index" for ops that have non-tensor
    # args/kwargs which would affect sharding propagation results. All args starting from
    # this index would be hashed to our sharding cache.
    # 运行时模式信息存储了运行时（即时执行）操作符的模式相关信息。主要用于两种方式：
    # 1. 生成参数哈希以确定是否重新运行分片属性或不运行
    # 2. 确定是否需要 pytree 的使用
    # 设置静态参数编号，仅有少数操作需要此信息，例如 view、transpose、var.dim 等。
    static_argnum: int = 100
    
    # 记录静态关键字参数的名称列表，这些参数可能影响分片属性。
    static_kwargkey: Optional[List[str]] = None
    
    # 每个操作可以决定是否在操作期间使用 pytree 的 flatten/unflatten 功能。
    # 在即时执行期间，默认情况下我们不需要执行 flatten/unflatten 操作，
    # 只有在操作指示需要时才会执行，这样可以加速即时执行的性能。
    needs_pytree: bool = False
@dataclass
class OpSchema:
    """
    OpSchema is a data class that describes an operator input schemas, it
    includes DTensor DTensorSpecs and non-tensor args/kwargs (positional order
    preserved). It is mainly used by the dispatching logic below to run things like
    sharding propagation.

    NOTE: this should be used as a read only data class
    TODO: make this a frozen dataclass

    Args:
        op: the operator overload we are intercepting
        args_schema: contains args except that the DTensor args have been replaced
            with its DTensorSpec
        kwargs_schema: contains kwargs except that the DTensor kwargs have been replaced
            with its DTensorSpec
    """

    op: OpOverload
    args_schema: ArgsType
    kwargs_schema: KwargsType

    schema_info: Optional[RuntimeSchemaInfo] = None

    @property
    def args_spec(self) -> Tuple[DTensorSpec, ...]:
        """
        args_spec: Tuple[DTensorSpec, ...]: contains a clean list of args spec list
            with NO non-DTensor positional arguments (i.e. int/float/tuple, etc)
            mainly used by sharding propagation to propagate the output spec
        """
        # Extracts leaves from args_schema if schema_info requires pytree, otherwise uses args_schema directly
        args = (
            tree_leaves(self.args_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else self.args_schema
        )
        # Returns a tuple of items from args that are instances of DTensorSpec
        return tuple(item for item in args if isinstance(item, DTensorSpec))

    @property
    def args_strategy(self) -> Tuple[OpStrategy, ...]:
        # Filters out non-relevant values from args schema to get a clean OpStrategy list
        # Separates from args_spec for better type annotation clarity
        # TODO: Consider merging this with args_spec
        args = (
            tree_leaves(self.args_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else self.args_schema
        )
        # Returns a tuple of items from args that are instances of OpStrategy
        return tuple(item for item in args if isinstance(item, OpStrategy))

    def __repr__(self) -> str:
        # Generates a string representation of OpSchema object for debugging and logging
        args_schema = ", ".join([str(arg_schema) for arg_schema in self.args_schema])
        return (
            f"OpSchema(op={self.op},"
            f" args_schema=({args_schema}),"
            f" kwargs_schema={self.kwargs_schema})"
        )
    # 返回对象的字符串表示形式，用于调试和打印
    def __str__(self) -> str:
        # 初始化参数列表
        args_sharding: List[str] = []
        # 网格形状初始化为None
        mesh_shape = None
        # 遍历参数模式列表
        for arg in self.args_schema:
            # 如果参数是DTensorSpec类型
            if isinstance(arg, DTensorSpec):
                # 将参数转换为字符串并添加到参数分片列表中
                args_sharding.append(str(arg))
                # 更新网格形状为当前参数的网格形状
                mesh_shape = arg.mesh.shape
            # 如果参数是OpStrategy类型
            elif isinstance(arg, OpStrategy):
                # 断言该策略的长度为1
                assert len(arg.strategies) == 1
                # 将策略的输出规范转换为字符串并添加到参数分片列表中
                args_sharding.append(_pretty_print_spec(arg.strategies[0].output_specs))
                # 更新网格形状为策略的网格形状
                mesh_shape = arg.mesh_shape
            # 如果参数是TupleStrategy类型
            elif isinstance(arg, TupleStrategy):
                # 获取第一个子策略
                first_op_strtgy = arg.childs[0]
                # 断言第一个子策略是OpStrategy类型
                assert isinstance(first_op_strtgy, OpStrategy)
                # 更新网格形状为第一个子策略的网格形状
                mesh_shape = first_op_strtgy.mesh_shape
                # 将TupleStrategy转换为字符串并添加到参数分片列表中
                args_sharding.append(str(arg))
            else:
                # 其他类型的参数直接转换为字符串并添加到参数分片列表中
                args_sharding.append(str(arg))
        # 返回格式化后的字符串表示，包括操作名称、参数分片列表和网格形状
        return f"Op(op={self.op}, args_sharding={', '.join(args_sharding)} @ mesh: {mesh_shape})"

    # 初始化后处理方法，设置是否包含符号整数的标志
    def __post_init__(self) -> None:
        # 初始化是否包含符号整数的标志为False
        has_symints = False
        # 遍历参数模式列表
        for a in self.args_schema:
            # 如果参数是DTensorSpec类型且具有张量元信息
            if isinstance(a, DTensorSpec) and a.tensor_meta is not None:
                # 如果张量形状中包含任何符号整数，则设置标志为True并退出循环
                if any(isinstance(s, torch.SymInt) for s in a.tensor_meta.shape):
                    has_symints = True
                    break
        # 将结果保存到实例变量中
        self.has_symints = has_symints

    # 判断指定索引的参数是否为张量或类张量列表
    def arg_type_tensor_or_tensor_list_like(self, arg_idx: int) -> bool:
        # 获取指定索引的参数
        arg = self.args_schema[arg_idx]
        # 判断参数是否为DTensorSpec类型
        is_tensor = isinstance(arg, DTensorSpec)
        # 如果是张量类型，则返回True
        if is_tensor:
            return True
        # 如果参数不是列表类型，则返回False
        if not isinstance(arg, list):
            return False
        # 如果参数是列表类型，检查是否所有元素都是DTensorSpec类型或为None
        return all(isinstance(e, DTensorSpec) or e is None for e in arg)

    # 判断返回类型是否为元组形式的张量类对象
    def return_type_tuple_tensor_like(self) -> bool:
        # 获取操作的返回类型列表
        return_types = self.op._schema.returns
        # 返回类型列表长度大于1且第一个元素是torch.TensorType类型的张量
        return len(return_types) > 1 and isinstance(
            return_types[0].type, torch.TensorType
        )

    # 判断返回类型是否为张量类对象
    def return_type_tensor(self) -> bool:
        # 获取操作的返回类型列表
        return_types = self.op._schema.returns
        # 返回第一个返回类型是否为torch.TensorType类型的张量
        return isinstance(return_types[0].type, torch.TensorType)
    def __hash__(self) -> int:
        # 只对需要进行哈希的参数和关键字参数进行哈希计算
        # 如果没有 schema_info，使用 args_schema 的长度作为静态参数数量，静态关键字参数键设为 None
        if not self.schema_info:
            static_argnum = len(self.args_schema)
            static_kwargkey = None
        else:
            static_argnum = self.schema_info.static_argnum
            static_kwargkey = self.schema_info.static_kwargkey

        # 根据条件选择需要进行哈希计算的参数
        args_to_hash = tuple(
            tuple(e) if isinstance(e, list) else e
            for i, e in enumerate(self.args_schema)
            if self.arg_type_tensor_or_tensor_list_like(i) or i >= static_argnum
        )
        
        # 如果存在静态关键字参数键，则计算其哈希值
        if static_kwargkey is not None:
            kwargs_to_hash = tuple(
                self.kwargs_schema.get(k, None) for k in static_kwargkey
            )
            return hash((self.op, args_to_hash, kwargs_to_hash))
        else:
            return hash((self.op, args_to_hash))

    def __eq__(self, other: object) -> bool:
        # 快速返回检查
        # 如果 other 不是 OpSchema 类的实例，则直接返回 False
        if not isinstance(other, OpSchema):
            return False

        # 检查操作符是否相同
        if self.op != other.op:
            return False

        # 检查参数列表长度是否相同
        if len(self.args_schema) != len(other.args_schema):
            return False

        # 逐个比较每个元素，如果有不同则提前返回 False
        # 如果没有 schema_info，使用 args_schema 的长度作为静态参数数量，静态关键字参数键设为 None
        if not self.schema_info:
            static_argnum = len(self.args_schema)
            static_kwargkey = None
        else:
            static_argnum = self.schema_info.static_argnum
            static_kwargkey = self.schema_info.static_kwargkey

        for i, (self_arg, other_arg) in enumerate(
            zip(self.args_schema, other.args_schema)
        ):
            if isinstance(self_arg, DTensorSpec) and self_arg != other_arg:
                return False
            elif i >= static_argnum and self_arg != other_arg:
                return False

        # 当存在静态关键字参数键时，检查关键字参数的相等性
        if static_kwargkey:
            for key in static_kwargkey:
                if self.kwargs_schema.get(key, None) != other.kwargs_schema.get(
                    key, None
                ):
                    return False

        return True

    def gen_fake_args(self) -> ArgsType:
        """
        gen_fake_args: 为操作符生成虚拟参数，主要用于分片传播规则，为操作符生成虚拟参数，
            以便运行本地张量操作符并获取输出规格。
        """
        return tree_map_only(
            DTensorSpec, _rebuild_tensor_from_dtensor_meta, self.args_schema
        )

    def gen_fake_kwargs(self) -> KwargsType:
        """
        gen_fake_kwargs: 为操作符生成虚拟关键字参数，主要用于分片传播规则，为操作符生成虚拟关键字参数，
            以便运行本地张量操作符并获取输出规格。
        """
        return tree_map_only(
            DTensorSpec, _rebuild_tensor_from_dtensor_meta, self.kwargs_schema
        )
    # 在给定的操作模式模式建议中，重新包装模式建议以适应特定需求
    def _inplace_rewrap_schema_suggestion(self, origin_schema: "OpSchema") -> None:
        # 复制当前操作模式建议的参数规范
        suggestion_args_spec = self.args_spec
        # 初始化一个空列表来存储新的参数模式
        new_arg_schema: List[object] = []
        # 索引，用于追踪参数规范列表的位置
        idx_of_args_spec = 0
        # 检查原始模式信息是否存在并且需要 PyTree 结构
        if (
            origin_schema.schema_info is not None
            and origin_schema.schema_info.needs_pytree
        ):
            # 如果需要 PyTree 结构，则展平原始模式的参数列表
            args_schema: Sequence[Any] = tree_leaves(origin_schema.args_schema)
        else:
            # 否则直接使用原始的参数模式
            args_schema = origin_schema.args_schema
        # 遍历原始参数模式
        for arg in args_schema:
            # 如果参数是 DTensorSpec 类型，则使用建议的参数规范替换
            if isinstance(arg, DTensorSpec):
                new_arg_schema.append(suggestion_args_spec[idx_of_args_spec])
                idx_of_args_spec += 1
            else:
                # 否则保持参数不变
                new_arg_schema.append(arg)
        # 将更新后的参数模式转换为元组并更新到当前对象的参数模式中
        self.args_schema = tuple(new_arg_schema)
        # 复制原始模式的关键字参数模式到当前对象的关键字参数模式中
        self.kwargs_schema = origin_schema.kwargs_schema
@dataclass
class OutputSharding:
    """
    OutputSharding 是一个数据类，用于分片传播规则，成功传播后可以设置 output_spec，
    如果失败，output_spec 将变为 None，分片传播规则可以提供输入的建议列表以进行重新分片。

    注意：由分片传播生成的 schema_suggestion 应与操作符 OpSchema 完全相同，除了 DTensor 和 DTensorSpecs。
    """

    output_spec: OutputSpecType  # 输出规范的类型
    redistribute_schema: Optional[OpSchema] = None  # 需要重新分配的模式，可选
    needs_redistribute: bool = False  # 是否需要重新分片的标志


@dataclass
class OpInfo:
    """
    所有运行时操作执行信息都打包在这里
    """

    mesh: DeviceMesh  # 设备网格信息
    schema: OpSchema  # 操作模式
    flat_args_schema: List[object]  # 扁平化参数模式列表
    local_args: Sequence[object]  # 本地参数序列
    local_kwargs: Dict[str, object]  # 本地关键字参数字典
    args_tree_spec: Optional[TreeSpec] = None  # 参数树规范，可选

    output_sharding: Optional[OutputSharding] = None  # 输出分片信息，可选
```