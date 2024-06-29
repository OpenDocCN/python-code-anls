# `D:\src\scipysrc\pandas\pandas\core\computation\engines.py`

```
"""
Engine classes for :func:`~pandas.eval`
"""

# 引入未来的注解特性，允许在类型检查时引用类本身
from __future__ import annotations

# 引入类型检查相关的模块
import abc
from typing import TYPE_CHECKING

# 导入可能引发 NumExprClobberingError 的错误类
from pandas.errors import NumExprClobberingError

# 导入需要的计算相关模块和函数
from pandas.core.computation.align import (
    align_terms,
    reconstruct_object,
)
from pandas.core.computation.ops import (
    MATHOPS,
    REDUCTIONS,
)

# 导入输出格式相关模块
from pandas.io.formats import printing

# 如果是类型检查模式，导入 Expr 类
if TYPE_CHECKING:
    from pandas.core.computation.expr import Expr

# 定义不可变的内建函数集合，包括数学操作和归约操作
_ne_builtins = frozenset(MATHOPS + REDUCTIONS)


def _check_ne_builtin_clash(expr: Expr) -> None:
    """
    Attempt to prevent foot-shooting in a helpful way.

    Parameters
    ----------
    expr : Expr
        Terms can contain
    """
    # 获取表达式中的名称集合
    names = expr.names
    # 计算内建函数和表达式名称的交集
    overlap = names & _ne_builtins

    # 如果有重叠的部分，抛出 NumExprClobberingError 错误
    if overlap:
        s = ", ".join([repr(x) for x in overlap])
        raise NumExprClobberingError(
            f'Variables in expression "{expr}" overlap with builtins: ({s})'
        )


class AbstractEngine(metaclass=abc.ABCMeta):
    """Object serving as a base class for all engines."""

    # 默认没有负数和分数
    has_neg_frac = False

    def __init__(self, expr) -> None:
        # 初始化引擎对象，设置表达式、对齐轴、结果类型和结果名称
        self.expr = expr
        self.aligned_axes = None
        self.result_type = None
        self.result_name = None

    def convert(self) -> str:
        """
        Convert an expression for evaluation.

        Defaults to return the expression as a string.
        """
        # 转换表达式以便评估，使用 pprint_thing 方法进行格式化输出
        return printing.pprint_thing(self.expr)

    def evaluate(self) -> object:
        """
        Run the engine on the expression.

        This method performs alignment which is necessary no matter what engine
        is being used, thus its implementation is in the base class.

        Returns
        -------
        object
            The result of the passed expression.
        """
        # 如果未对齐，则进行对齐操作
        if not self._is_aligned:
            self.result_type, self.aligned_axes, self.result_name = align_terms(
                self.expr.terms
            )

        # 执行具体的评估操作
        res = self._evaluate()
        # 重建对象，确保返回的对象符合预期的结果类型和名称
        return reconstruct_object(
            self.result_type,
            res,
            self.aligned_axes,
            self.expr.terms.return_type,
            self.result_name,
        )

    @property
    def _is_aligned(self) -> bool:
        # 返回对齐状态，判断对齐轴和结果类型是否已设置
        return self.aligned_axes is not None and self.result_type is not None

    @abc.abstractmethod
    def _evaluate(self):
        """
        Return an evaluated expression.

        Parameters
        ----------
        env : Scope
            The local and global environment in which to evaluate an
            expression.

        Notes
        -----
        Must be implemented by subclasses.
        """


class NumExprEngine(AbstractEngine):
    """NumExpr engine class"""

    # NumExpr 引擎类，具有负数和分数的处理能力
    has_neg_frac = True
    def _evaluate(self):
        import numexpr as ne  # 导入numexpr库，用于高效计算数值表达式

        # 将表达式转换为有效的numexpr表达式
        s = self.convert()

        env = self.expr.env  # 获取表达式所在的环境对象
        scope = env.full_scope  # 获取环境对象的完整作用域
        _check_ne_builtin_clash(self.expr)  # 检查表达式是否与numexpr内置函数冲突
        # 使用numexpr库对表达式求值，传入本地作用域字典作为参数
        return ne.evaluate(s, local_dict=scope)
class PythonEngine(AbstractEngine):
    """
    Evaluate an expression in Python space.

    Mostly for testing purposes.
    """

    # 定义一个 PythonEngine 类，继承自 AbstractEngine 抽象类，
    # 用于在 Python 空间中评估表达式，主要用于测试目的。
    
    has_neg_frac = False
    # 类属性 has_neg_frac，用于标识是否存在负数和分数。

    def evaluate(self):
        # 定义 evaluate 方法，用于评估表达式，实际执行 self.expr() 方法。
        return self.expr()

    def _evaluate(self) -> None:
        # 定义 _evaluate 方法，返回类型为 None。
        pass
        # 该方法暂未实现具体功能。

ENGINES: dict[str, type[AbstractEngine]] = {
    "numexpr": NumExprEngine,
    # ENGINES 字典的键 "numexpr"，对应值为 NumExprEngine 类型。
    
    "python": PythonEngine,
    # ENGINES 字典的键 "python"，对应值为 PythonEngine 类型。
}
# ENGINES 字典，包含不同引擎类型，用于选择合适的引擎进行表达式求值。
```