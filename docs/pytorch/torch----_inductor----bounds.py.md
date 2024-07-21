# `.\pytorch\torch\_inductor\bounds.py`

```
# 设置允许未类型化的函数定义，用于静态类型检查工具mypy
# 导入日志模块
import logging
# 导入运算符模块
import operator
# 导入functools模块中的partial函数
from functools import partial
# 导入类型提示模块中的Any和Callable类型
from typing import Any, Callable, Dict

# 导入sympy库中的Expr表达式类
from sympy import Expr

# 导入PyTorch库
import torch
# 导入torch.utils._sympy.value_ranges模块中的bound_sympy、ValueRangeAnalysis和ValueRanges类
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeAnalysis, ValueRanges
# 导入本地模块.ir中的InterpreterShim、LoopBody和LoopBodyBlock类
from .ir import InterpreterShim, LoopBody, LoopBodyBlock
# 导入本地模块.utils中的cache_on_self和dominated_nodes函数
from .utils import cache_on_self, dominated_nodes
# 导入本地模块.virtualized中的V类
from .virtualized import V

# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)


class BoundVars:
    """
    对LoopBody的fx图执行值范围分析，通过调用BoundVars.run()方法
    它公开了`bounds`变量中节点的范围

    注意：该分析的当前限制是仅在每个循环基础上工作。
    我们应该能够在整个图之间传播这些范围。这可能有利于有界变量由内核返回并输入到另一个内核的情况。
    """

    def __init__(self, loop_body: LoopBody) -> None:
        # 定义一个函数，返回给定表达式的上界
        def upper_bound(v):
            return bound_sympy(v).upper if isinstance(v, Expr) else v

        # 设置循环体对象
        self.loop_body = loop_body
        # 创建替换值字典，基于循环体中变量的范围
        self.replacement_vals = {
            k: ValueRanges[Expr](0, upper_bound(v) - 1)
            for k, v in loop_body.var_ranges.items()
        }
        # 避免计算这些值，悲观地假设它们是无界的
        self.unbounded_vars = dominated_nodes(
            node
            for node in self.loop_body.get_nodes()
            if node.target in ["load", "reduction", operator.getitem]
            or "masked_subblock" in node.target
        )
        # 要访问此变量，请调用`get_bounds()`方法
        self._bounds: Dict[torch.fx.Node, ValueRanges[Expr]] = {}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"loop_body={self.loop_body},\n "
            f"replacement_vals={self.replacement_vals}, \n"
            f"unbounded_vars={self.unbounded_vars}, \n"
            f"_bounds={self._bounds})"
        )

    @cache_on_self
    def get_bounds(self) -> Dict[torch.fx.Node, ValueRanges[Expr]]:
        # 交换子模块以获取操作处理器
        submodules = self.swap_submodules(self.loop_body.submodules)

        # 使用未绑定的变量初始化环境
        for node in self.unbounded_vars:
            # 我们需要评估masked_subblock来递归，还需要设置间接值
            if not isinstance(node.target, str) or (
                "masked_subblock" not in node.target
                and "set_indirect" not in node.target
            ):
                self._bounds[node] = ValueRanges[Expr].unknown()

        # 使用ValueRangeAnalysis设置操作处理器
        with V.set_ops_handler(ValueRangeAnalysis()):
            # 创建解释器对象
            interpreter = InterpreterShim(self.loop_body.root_block.graph, submodules)
            log.debug("get_bounds:\n%s", self.loop_body.root_block.graph)
            # 运行解释器以计算边界
            interpreter.run(V.get_ops_handler(), initial_env=self._bounds)
        # 返回计算后的边界字典
        return self._bounds

    def swap_submodules(
        self, submodules: Dict[str, Callable[..., Any]]
    ) -> Dict[str, Callable[..., ValueRanges[Expr]]]:
        result: Dict[str, Callable[..., ValueRanges[Expr]]] = {}
        # 初始化一个空的结果字典，用于存放函数名到函数对象的映射关系
        for key in submodules.keys():
            # 遍历子模块字典的键
            if key == "get_index":
                # 如果键为"get_index"
                # 将"get_index"键映射到当前对象的get_index方法
                result[key] = self.get_index
            elif "masked_subblock" in key:
                # 如果键包含"masked_subblock"
                # 获取对应的子模块对象
                subblock = self.loop_body.subblocks[key]
                # 定义一个函数make_fn，返回一个lambda函数，该lambda函数引用了当前循环中的subblock对象
                def make_fn(subblock):
                    return lambda mask, value: self.masked_subblock(
                        subblock, self._bounds, mask, value, result
                    )
                # 将键映射到make_fn返回的lambda函数
                result[key] = make_fn(subblock)
            elif "set_indirect" in key:
                # 如果键包含"set_indirect"
                # 解析出变量索引
                idx = int(key[len("set_indirect"):])
                # 获取间接变量对象
                var = self.loop_body.indirect_vars[idx]
                # 创建一个部分应用的函数indirect，该函数调用self.set_indirect方法
                indirect = partial(self.set_indirect, var)
                # 将键映射到indirect函数
                result[key] = indirect
            else:
                # 否则，键中必须包含"scan"
                assert "scan" in key
                # 直接将键映射到submodules中对应的值
                result[key] = submodules[key]

        # 返回最终构建好的result字典
        return result

    def masked_subblock(
        self,
        subblock: LoopBodyBlock,
        env: Dict[torch.fx.Node, ValueRanges[Expr]],
        mask: Any,
        value: Any,
        submodules: Dict[str, Callable[..., Any]],
    ) -> ValueRanges[Expr]:
        # 使用InterpreterShim对subblock进行解释
        interp = InterpreterShim(subblock.graph, submodules)
        # 运行InterpreterShim的run方法，使用给定的env作为初始环境
        interp.run(V.get_ops_handler(), initial_env=env)
        # 获取输出节点列表，通常只会有一个节点目标为"output"
        output = [node for node in subblock.graph.nodes if node.target == "output"]
        assert len(output) == 1
        # 返回输出节点的环境值
        return interp.env[output[0]]

    def set_indirect(self, old: Expr, new: ValueRanges[Expr]) -> ValueRanges[Expr]:
        # 断言new是ValueRanges类型的实例
        assert isinstance(new, ValueRanges)
        # 将replacement_vals字典中old对应的值更新为new
        self.replacement_vals[old] = new
        # 返回更新后的new值
        return new

    def get_index(self, name: Expr) -> ValueRanges[Expr]:
        # 获取名为name的索引表达式
        expr = self.loop_body.indexing_exprs[name]
        # 尝试从replacement_vals中获取表达式的替代值
        bound = self.replacement_vals.get(expr)
        if bound is None:
            # 如果替代值为None，则调用bound_sympy函数计算表达式的边界值
            bound = bound_sympy(expr, self.replacement_vals)
        # 将name和bound值更新到replacement_vals字典中
        self.replacement_vals[name] = bound
        # 返回计算得到的bound值
        return bound
```