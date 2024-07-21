# `.\pytorch\torch\fx\passes\runtime_assert.py`

```
# mypy: allow-untyped-defs
# 引入日志模块
import logging
# 引入运算符模块
import operator
# 引入类型注解相关模块
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

# 在类型检查时导入 sympy 和 ShapeEnv，因为导入 sympy 较慢
if TYPE_CHECKING:
    # 从 torch.fx.experimental.symbolic_shapes 导入 ShapeEnv 类型
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
else:
    # 在非类型检查时，将 ShapeEnv 定义为任意类型
    ShapeEnv = Any

# 引入 torch 库
import torch
# 引入 torch.utils._pytree 模块
import torch.utils._pytree as pytree
# 从 torch 模块中导入 fx 子模块
from torch import fx
# 从 torch.fx._compatibility 模块中导入 compatibility 函数
from torch.fx._compatibility import compatibility
# 从 torch.fx._utils 模块中导入 lazy_format_graph_code 函数
from torch.fx._utils import lazy_format_graph_code
# 从 torch.fx.experimental.sym_node 模块中导入 SymNode 类
from torch.fx.experimental.sym_node import SymNode
# 从 torch.fx.graph_module 模块中导入 GraphModule 类
from torch.fx.graph_module import GraphModule

# 设置日志记录器
log = logging.getLogger(__name__)
# 设置用于记录图形代码的日志记录器
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code")

def _get_example_value(node: fx.Node) -> Optional[str]:
    """
    获取节点的示例值键名，因为 dynamo 使用 "example_value"
    而非严格导出使用 "val"。
    """
    if "example_value" in node.meta:
        return node.meta["example_value"]
    elif "val" in node.meta:
        return node.meta["val"]
    else:
        return None

@compatibility(is_backward_compatible=True)
def insert_deferred_runtime_asserts(
    gm: GraphModule,
    shape_env: ShapeEnv,
    name: str,
    export: bool = False,
) -> None:
    """
    在追踪过程中，可能会发现某些数据相关的值有运行时断言；
    例如，torch.empty(x.item()) 会导致运行时断言 x.item() >= 0。
    这些断言可能在假张量传播期间不可预测地发生，
    因此我们不能方便地在 FX 图中插入它们。
    相反，我们将它们积累在 ShapeEnv 中，在此过程中将它们作为正确的测试插入图中。
    """

    # 用于存储已经具有符号约束范围的节点集合，以 (node_name, min_val, max_val) 形式存储
    nodes_that_already_have_sym_constraint_range = set()

    # 仅用节点名称进行哈希，因为尺寸不考虑最小/最大值
    nodes_that_already_have_sym_constraint_size = set()
    # TODO: 当前仅适用于顶级节点，此外我们应该考虑使用它以避免创建重复的 assert_async 节点
    for node in gm.graph.nodes:
        # 检查节点是否为调用函数且目标为 torch.ops.aten.sym_constrain_range.default
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.sym_constrain_range.default
        ):
            assert len(node.args) == 1
            # 将节点名称及其最小/最大值加入已具有符号约束范围的节点集合
            nodes_that_already_have_sym_constraint_range.add(
                (node.args[0], node.kwargs.get("min"), node.kwargs.get("max"))
            )
        # 检查节点是否为调用函数且目标为 torch.ops.aten.sym_constrain_range_for_size.default
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.sym_constrain_range_for_size.default
        ):
            assert len(node.args) == 1
            # 将节点名称加入已具有符号约束尺寸的节点集合
            nodes_that_already_have_sym_constraint_size.add(node.args[0])

    # 在本地导入 sympy
    import sympy

    # 从 torch.fx.experimental.symbolic_shapes 模块中导入一系列函数和类
    from torch.fx.experimental.symbolic_shapes import (
        CallMethodKey,
        cast_symbool_to_symint_guardless,
        ConvertIntKey,
        DivideByKey,
        free_symbols,
        InnerTensorKey,
    )
    # 从 torch.utils._sympy.interp 模块中导入 sympy_interp 函数
    from torch.utils._sympy.interp import sympy_interp
    # 从 torch.utils._sympy.numbers 导入 int_oo
    from torch.utils._sympy.numbers import int_oo
    # 从 torch.utils._sympy.reference 导入 PythonReferenceAnalysis
    from torch.utils._sympy.reference import PythonReferenceAnalysis

    # TODO: 在发出运行时断言之前请求简化
    # 将 shape_env 中延迟的运行时断言复制到 ras_by_symbol 字典中
    ras_by_symbol = shape_env.deferred_runtime_asserts.copy()
    # 获取图形对象 gm 的引用
    graph = gm.graph

    # 如果 ras_by_symbol 的值中没有任何运行时断言，则直接返回
    if not any(ras for ras in ras_by_symbol.values()):
        return

    # 记录调试日志，输出格式化的图形代码片段，带有名称和彩色标记
    graph_code_log.debug(
        "%s",
        lazy_format_graph_code(
            f"pre insert_deferred_runtime_asserts {name}", gm, colored=True
        ),
    )

    # 删除未关联的运行时断言的重复项
    # 我们可以更好地处理，一些保护可能是多余的，
    # 例如 Eq(s0, 4) & Eq(2*s0, 8)
    # 但目前不清楚如何处理所有这些情况。
    # TODO(pianpwk): 更好的方法
    new_ras = []
    ras_exprs: Set[sympy.Expr] = set()
    for ras in ras_by_symbol.pop(None, []):  # type: ignore[call-overload]
        if ras.expr not in ras_exprs:
            new_ras.append(ras)
            ras_exprs.add(ras.expr)
    ras_by_symbol[None] = new_ras  # type: ignore[index]

    # 我们将要修改字典
    symbol_to_proxy: Dict[sympy.Symbol, fx.Proxy] = {}
    # 创建一个占位符集合
    placeholders = set()
    # 最后一个占位符节点的初始化
    last_placeholder = None
    # 遍历图中的节点
    for node in graph.nodes:
        # 如果节点的操作类型不是 "placeholder"，则跳出循环
        if node.op != "placeholder":
            break
        last_placeholder = node
        # 将节点添加到占位符集合中
        placeholders.add(node)
    # 如果没有找到任何占位符，将最后一个占位符设为图中的第一个节点
    if last_placeholder is None:
        last_placeholder = next(iter(graph.nodes))

    # 确定我们需要重新实例化的符号。虽然这不是严格必需的，
    # 但有助于减少图中的变动
    needed_symbols: Set[sympy.Symbol] = set()
    for ras in ras_by_symbol.values():
        for ra in ras:
            # 更新所需符号集合，包括在 ra.expr 中自由的符号
            needed_symbols.update(free_symbols(ra.expr))

    # 记录调试日志，输出所需的符号集合
    log.debug("needed_symbols = %s", needed_symbols)
    def add_runtime_asserts(ras):
        # 遍历传入的运行时断言列表
        for ra in ras:
            # 记录调试信息，插入运行时断言表达式
            log.debug("inserting runtime assert %s", ra.expr)
            # 获取表达式中的所有自由符号
            fvs = free_symbols(ra.expr)
            # 找出在symbol_to_proxy字典中不存在的符号集合
            missing = fvs - symbol_to_proxy.keys()
            if missing:
                # 从缺失的符号中选择字典序最小的一个
                i1 = min(missing, key=str)
                # 将当前的运行时断言添加到ras_by_symbol字典中，以符号为键
                ras_by_symbol.setdefault(i1, []).append(ra)
            else:
                # 将sympy表达式转换为一系列FX节点
                res = sympy_interp(
                    PythonReferenceAnalysis, symbol_to_proxy, ra.expr
                ).node
                # 调用graph对象的call_function方法，执行特定的运行时断言操作
                graph.call_function(
                    torch.ops.aten._assert_scalar.default,
                    # TODO: 在这里使用ra.msg，但目前它还没有什么用处
                    (
                        res,
                        f"Runtime assertion failed for expression {ra.expr} on node '{res}'",
                    ),
                )

    inserted_sym_nodes = 0  # 用于插入未关联的运行时断言
    nodes = list(graph.nodes)
```