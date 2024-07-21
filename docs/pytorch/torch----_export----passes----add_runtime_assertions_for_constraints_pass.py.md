# `.\pytorch\torch\_export\passes\add_runtime_assertions_for_constraints_pass.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import math  # 导入数学库
import operator  # 导入操作符模块
import traceback  # 导入异常跟踪模块
from functools import partial  # 导入偏函数模块
from typing import Callable, Dict, List, NamedTuple, Set  # 导入类型提示相关模块

import sympy  # 导入符号计算库

import torch  # 导入PyTorch库
import torch.fx  # 导入PyTorch的特定函数库
from torch.utils._sympy.value_ranges import ValueRanges  # 导入值范围模块
from torch.utils._sympy.numbers import int_oo  # 导入无穷大整数模块
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols  # 导入符号形状实验模块
from torch.fx.passes.infra.pass_base import PassBase, PassResult  # 导入基础通行证和通行结果模块

__all__ = ["InputDim"]  # 模块公开的名称列表


class InputDim(NamedTuple):
    input_name: str  # 输入名称字段
    dim: int  # 维度字段


def _convert_to_int(val):
    # 将简单的符号整数转换为具体整数
    if val in (sympy.oo, int_oo):
        return math.inf  # 如果是正无穷，返回Python中的正无穷
    if val in (-sympy.oo, -int_oo):
        return -math.inf  # 如果是负无穷，返回Python中的负无穷
    if isinstance(val, sympy.Integer):
        return int(val)  # 如果是符号整数，返回其整数值
    raise RuntimeError(
        "Export constraints cannot be non-integer expressions"
    )  # 抛出运行时异常，表明导出约束不能是非整数表达式


def _convert_range_to_int(range: ValueRanges):
    assert isinstance(range, ValueRanges)  # 断言值范围类型是ValueRanges
    min_val = _convert_to_int(range.lower)  # 将范围的下限转换为整数
    max_val = _convert_to_int(range.upper)  # 将范围的上限转换为整数
    return min_val, max_val  # 返回整数化的范围


class _AddRuntimeAssertionsForInlineConstraintsPass(PassBase):
    def __init__(
        self,
        range_constraints: Dict[sympy.Symbol, ValueRanges],  # 构造函数，接受符号到值范围字典作为参数
    ):
        super().__init__()  # 调用基类的构造函数
        self.range_constraints: Dict[sympy.Symbol, ValueRanges] = range_constraints  # 初始化约束范围字典
        self._asserts_generated_unbacked_symbols: Set[sympy.Symbol] = set()  # 初始化未支持符号的断言生成集合
        self.counter = 0  # 计数器初始化为0

    def _assert_range_constraint(self, node, lower, upper, assert_msg):
        last_node = node  # 记录最后一个节点
        if lower > -math.inf:
            # 如果下限大于负无穷，插入异步断言检查下限
            last_node = self._insert_assert_async(last_node, operator.ge, node, lower, assert_msg)

        if upper < math.inf:
            # 如果上限小于正无穷，插入异步断言检查上限
            last_node = self._insert_assert_async(last_node, operator.le, node, upper, assert_msg)

    def _insert_assert_async(self, last_node, op, lower, upper, assert_msg):
        """
        Inserts assert_async call_function nodes in the graph. This function is
        called **during** the interpreter-based pass.
        """
        self.counter += 1  # 计数器加一
        graph = last_node.graph  # 获取图形对象
        with graph.inserting_after(last_node):
            cmp = graph.call_function(op, (lower, upper), {})  # 调用函数比较下限和上限
        with graph.inserting_after(cmp):
            cmp_tensor = graph.call_function(torch.ops.aten.scalar_tensor.default, (cmp,), {})  # 将比较结果转换为张量
        with graph.inserting_after(cmp_tensor):
            assert_async = graph.call_function(
                torch.ops.aten._assert_async.msg,  # 调用异步断言函数
                (cmp_tensor, assert_msg),  # 传递比较张量和断言消息
                {},  # 空的关键字参数
            )
        return assert_async  # 返回异步断言节点

def _get_existing_inline_assertions(
    graph_module: torch.fx.GraphModule,
    range_constraints: Dict[sympy.Symbol, ValueRanges],
) -> Dict[sympy.Symbol, ValueRanges]:
    existing_inline_assertions: Dict[sympy.Symbol, ValueRanges] = {}  # 初始化内联断言字典
    for module in graph_module.modules():
        # 遍历图模块中的每一个模块
        if not isinstance(module, torch.fx.GraphModule):
            continue
        # 如果模块不是 torch.fx.GraphModule 类型，则继续下一个模块的处理

        # 查找所有现有的内联断言。它们通常如下所示：
        # %_local_scalar_dense = call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%arg1_1,), kwargs = {})
        # %ge = call_function[target=operator.ge](args = (%_local_scalar_dense, 0), kwargs = {})
        # %_assert_scalar = call_function[target=torch.ops.aten._assert_scalar.default](args = (%scalar_tensor, "..."), kwargs = {})
        for node in module.graph.nodes:
            # 遍历每一个节点
            if node.target != torch.ops.aten._assert_scalar.default:
                continue
            # 如果节点的目标不是 torch.ops.aten._assert_scalar.default，则继续下一个节点的处理

            compare_arg = node.args[0]
            # 获取比较参数

            if not (
                isinstance(compare_arg, torch.fx.Node) and
                compare_arg.op == "call_function" and
                compare_arg.target in (operator.le, operator.ge) and
                len(compare_arg.args) == 2
            ):
                continue
            # 如果比较参数不满足条件，则继续下一个节点的处理

            compare_op = compare_arg.target
            lhs, rhs = compare_arg.args
            # 获取比较操作符、左右操作数

            def maybe_get_symint(x):
                # 辅助函数，用于获取符号整数表达式
                if (
                    isinstance(x, torch.fx.Node) and
                    "val" in x.meta and
                    isinstance(x.meta["val"], torch.SymInt)
                ):
                    return x.meta["val"].node.expr
                return x

            lhs = maybe_get_symint(lhs)
            rhs = maybe_get_symint(rhs)
            # 可能获取左右操作数的符号整数表达式

            if compare_op == operator.ge:
                lhs, rhs = rhs, lhs
            # 如果比较操作是大于等于，则交换左右操作数

            if isinstance(lhs, sympy.Symbol) and isinstance(rhs, int):
                symint = lhs
                scalar = rhs
            elif isinstance(rhs, sympy.Symbol) and isinstance(lhs, int):
                symint = rhs
                scalar = lhs
            else:
                continue
            # 确定符号整数和标量值的关系，如果不满足条件，则继续下一个节点的处理

            if symint not in range_constraints:
                raise RuntimeError(f"Unable to find symint {symint} in {range_constraints}")
            # 如果符号整数不在范围约束中，则抛出异常

            previous_range = existing_inline_assertions.get(symint, ValueRanges(-math.inf, math.inf))
            # 获取先前的内联断言范围，默认为无限范围

            if symint is lhs:
                bounds = ValueRanges(-math.inf, scalar)
            else:
                bounds = ValueRanges(scalar, math.inf)
            # 根据符号整数和标量值确定边界范围

            existing_inline_assertions[symint] = previous_range & bounds
            # 更新内联断言范围

    return existing_inline_assertions
    # 返回更新后的内联断言字典
```