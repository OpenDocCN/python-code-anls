# `.\pytorch\torch\distributed\_tensor\_sharding_prop.py`

```py
# mypy: allow-untyped-defs
from functools import lru_cache  # 导入 functools 模块中的 lru_cache 装饰器，用于缓存函数调用结果
from itertools import chain  # 导入 itertools 模块中的 chain 函数，用于将多个可迭代对象连接在一起
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, Union  # 导入多个类型提示，用于静态类型检查

import torch  # 导入 torch 库，主要用于科学计算和机器学习
from torch._ops import OpOverload  # 从 torch._ops 中导入 OpOverload 类
from torch._subclasses import FakeTensorMode  # 从 torch._subclasses 中导入 FakeTensorMode 类
from torch.distributed._tensor._op_schema import (  # 导入分布式张量操作相关的模块和类
    OpInfo,
    OpSchema,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    PlacementStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed._tensor._utils import (  # 导入分布式张量操作相关的实用工具函数
    compute_local_shape,
    compute_local_stride,
    try_find_mesh_from_args,
)
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta  # 导入张量分布类型相关的类
from torch.distributed.device_mesh import DeviceMesh  # 导入设备网格相关的类


aten = torch.ops.aten  # 设置 aten 作为 torch.ops.aten 的别名，用于调用底层的 ATen 操作


def _length(obj) -> int:
    # 返回对象的长度，如果对象为空则返回 0，如果对象不是序列则返回 1，否则返回对象的长度
    if obj is None:
        return 0
    if not isinstance(obj, Sequence):
        return 1
    return len(obj)


class ShardingPropagator:
    def __init__(self) -> None:
        # 初始化 ShardingPropagator 类的实例
        self.op_to_rules: Dict[OpOverload, Callable[[OpSchema], OutputSharding]] = {}
        # 存储操作重载到其对应的规则函数的映射
        self.op_strategy_funcs: Dict[
            OpOverload,
            Callable[[DeviceMesh, OpSchema], StrategyType],
        ] = {}
        # 存储操作重载到其对应的策略函数的映射
        self.op_to_schema_info: Dict[OpOverload, RuntimeSchemaInfo] = {}
        # 存储操作重载到其对应的运行时模式信息的映射
        self.propagate_op_sharding = lru_cache(None)(self.propagate_op_sharding_non_cached)  # type: ignore[method-assign]
        # 使用 lru_cache 装饰器对 propagate_op_sharding_non_cached 方法进行缓存，用于操作分片传播
        self.op_to_shape_and_stride_idx: Dict[
            OpOverload, Union[int, Tuple[int, int]]
        ] = {
            # 存储操作重载到其对应的需要修改的形状（和步长）参数的索引
            # 新工厂操作
            aten.new_empty.default: 1,
            aten.new_full.default: 1,
            aten.new_ones.default: 1,
            aten.new_zeros.default: 1,
            aten.new_empty_strided.default: (1, 2),
            # 视图操作
            aten.expand.default: 1,
            aten.reshape.default: 1,
            aten.view.default: 1,
            aten._unsafe_view.default: 1,
        }

    def register_sharding_prop_rule(
        self,
        op_overload: OpOverload,
        rule_func: Callable[[OpSchema], OutputSharding],
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ):
        """
        注册操作的分片传播规则。
        """
        self.op_to_rules[op_overload] = rule_func
        # 将操作重载映射到其对应的分片传播规则函数
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info
        # 如果提供了模式信息，则将操作重载映射到其对应的运行时模式信息

    def register_op_strategy(
        self,
        op_overload: OpOverload,
        strategy_func: Callable[[DeviceMesh, OpSchema], StrategyType],
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ):
        """
        注册操作的策略函数。
        """
        self.op_strategy_funcs[op_overload] = strategy_func
        # 将操作重载映射到其对应的策略函数
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info
        # 如果提供了模式信息，则将操作重载映射到其对应的运行时模式信息
    ):
        """
        Register a sharding strategy generator for an operator.
        """
        # 将操作符的分片策略生成函数注册到字典中
        self.op_strategy_funcs[op_overload] = strategy_func
        # 如果有模式信息，将操作符与模式信息关联起来
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    @lru_cache  # noqa: B019
    def _propagate_tensor_meta(
        self, op_schema: OpSchema
    ) -> Union[None, TensorMeta, Sequence[Optional[TensorMeta]]]:
        """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas
        """
        # 如果操作符是等号，默认不做虚拟传播
        if op_schema.op == aten.equal.default:
            return None

        # 注意: 我们必须在假张量模式下调用跟踪，以避免实现内存
        with FakeTensorMode():
            # 生成假参数和假关键字参数
            fake_args = op_schema.gen_fake_args()
            fake_kwargs = op_schema.gen_fake_kwargs()
            # 在假张量模式下执行操作
            fake_out = op_schema.op(*fake_args, **fake_kwargs)

        # 如果返回的是 torch.Tensor 类型的对象
        if isinstance(fake_out, torch.Tensor):
            return TensorMeta(
                shape=fake_out.shape, stride=fake_out.stride(), dtype=fake_out.dtype
            )

        # 如果返回的是 tuple 或者 list 类型的对象
        elif isinstance(fake_out, (tuple, list)):
            tensor_meta_list: List[Optional[TensorMeta]] = []
            for fake_out_item in fake_out:
                if isinstance(fake_out_item, torch.Tensor):
                    tensor_meta_list.append(
                        TensorMeta(
                            shape=fake_out_item.shape,
                            stride=fake_out_item.stride(),
                            dtype=fake_out_item.dtype,
                        )
                    )
                else:
                    tensor_meta_list.append(None)
            return (
                tuple(tensor_meta_list)
                if isinstance(fake_out, tuple)
                else tensor_meta_list
            )
        else:
            # 如果返回的既不是 tensor 也不是 tensor 的 tuple/list，返回 None
            return None

    def _wrap_output_spec_tensor_meta(
        self,
        op: OpOverload,
        output_specs: OutputSpecType,
        output_tensor_meta: Union[None, TensorMeta, Sequence[Optional[TensorMeta]]],
    ) -> None:
        """
        Wrap the output_specs with the tensor metadata from the output.
        """

        # Check if output_specs is an instance of DTensorSpec
        if isinstance(output_specs, DTensorSpec):
            # Ensure output_tensor_meta is an instance of TensorMeta
            if not isinstance(output_tensor_meta, TensorMeta):
                # Raise ValueError if output_tensor_meta is not a TensorMeta or a tuple/list
                if not isinstance(output_tensor_meta, (tuple, list)):
                    raise ValueError(
                        "ShardingPropagator error: output does not have an associated TensorMeta"
                    )
                raise ValueError(
                    f"For the op {op.name()}, `output_specs` has 1 output which does not equal the "
                    f"number of op outputs: {len(output_tensor_meta)}."
                )
            # Assign tensor metadata from output_tensor_meta to output_specs
            output_specs.tensor_meta = output_tensor_meta
        # Check if output_specs is a tuple or list
        elif isinstance(output_specs, (tuple, list)):
            # Ensure output_tensor_meta is also a tuple or list of the same length as output_specs
            if not isinstance(output_tensor_meta, (tuple, list)) or len(
                output_specs
            ) != len(output_tensor_meta):
                raise ValueError(
                    f"For the op {op.name()}, `output_specs` has {len(output_specs)} outputs which does not equal the "
                    f"number of op outputs {_length(output_tensor_meta)}."
                )
            # Iterate over each element in output_specs
            for i, spec in enumerate(output_specs):
                # Check if each spec is an instance of DTensorSpec
                if isinstance(spec, DTensorSpec):
                    # Retrieve corresponding output_tensor_meta for the current index
                    output_tensor_meta_i = output_tensor_meta[i]
                    # Ensure output_tensor_meta_i is an instance of TensorMeta
                    if not isinstance(output_tensor_meta_i, TensorMeta):
                        raise ValueError(
                            f"ShardingPropagator error: output {i} does not have an associated TensorMeta"
                        )
                    # Assign tensor metadata from output_tensor_meta_i to spec
                    spec.tensor_meta = output_tensor_meta_i

    def propagate(self, op_info: OpInfo) -> None:
        # We cannot use an lru cache if we know that inputs will have dynamic shapes,
        # because SymInts are not hashable.
        # This is generally ok because this only happens during tracing in torch.compile,
        # and tracing does not need to be as fast as eagermode DTensor usages.
        
        # Check if op_info.schema has symints (symbolic integers)
        if op_info.schema.has_symints:
            # Compute output sharding without using LRU cache
            output_sharding = self.propagate_op_sharding_non_cached(op_info.schema)
        else:
            # Compute output sharding using LRU cache
            output_sharding = self.propagate_op_sharding(op_info.schema)
        # Assign computed output_sharding to op_info.output_sharding
        op_info.output_sharding = output_sharding
    # 选择操作策略的方法，返回一个适合的放置策略对象
    def _select_strategy(self, strategy: OpStrategy) -> PlacementStrategy:
        # 如果策略列表中只有一个策略，直接返回这个策略对象作为结果
        if len(strategy.strategies) == 1:
            # 一种快速路径，只有一个可能的策略
            return strategy.strategies[0]

        # 存储各个策略的重分配成本
        strategy_costs: List[float] = []
        for strtg in strategy.strategies:
            # 确保每个策略的重分配成本不为 None
            assert (
                strtg.redistribute_cost is not None
            ), "must set redistribute cost each strategy!"
            # 计算并累加每个策略的重分配成本
            redistribute_cost = sum(chain.from_iterable(strtg.redistribute_cost))
            strategy_costs.append(redistribute_cost)

        # 在急切执行模式下，选择重分配成本最小的策略
        return strategy.strategies[strategy_costs.index(min(strategy_costs))]

    # 调整形状和步幅参数的方法，更新操作模式的架构
    def _adjust_shape_and_stride_args(
        self,
        out_tensor_meta: TensorMeta,
        schema: OpSchema,
        spec: DTensorSpec,
        mesh: DeviceMesh,
    ) -> OpSchema:
        # 获取操作符对应的形状和步幅索引
        shape_stride_idx = self.op_to_shape_and_stride_idx[schema.op]
        if isinstance(shape_stride_idx, tuple):
            shape_idx, stride_idx = shape_stride_idx
        else:
            shape_idx = shape_stride_idx
            stride_idx = None

        # 复制预期的输入模式列表
        expected_input_schema = list(schema.args_schema)
        # 调整形状以匹配 _local_tensor 的形状
        # DTensor 输入参数的索引为 0，该形状是通过计算推断得到的
        expected_input_schema[shape_idx] = compute_local_shape(
            out_tensor_meta.shape, mesh, spec.placements
        )

        # 调整步幅参数以适应 aten.new_empty_strided.default
        if stride_idx:
            expected_input_schema[stride_idx] = compute_local_stride(
                out_tensor_meta.stride, mesh, spec.placements
            )

        # 返回更新后的操作模式对象
        return OpSchema(schema.op, tuple(expected_input_schema), schema.kwargs_schema)
```