# `.\pytorch\torch\fx\experimental\sym_node.py`

```py
"""
This file does three things:
- Contains the definition of SymNode
- Installs all the magic methods into SymBool, SymFloat, SymFloat at import time
- Does not depend on sympy at import time

As this file is imported from within torch/__init__.py we do not want it to depend on SymPy
to avoid having to load SymPy at import time, as doing so is *very* slow.
"""

import builtins               # 内建函数和异常
import itertools              # 生成迭代器的函数
import logging                # 日志记录模块
import math                   # 数学函数
import operator               # 操作符函数
import sys                    # 系统相关的参数和函数
from functools import lru_cache, update_wrapper  # 缓存函数和更新包装器
from typing import Optional, Type, TYPE_CHECKING, Union  # 类型提示

import torch                 # 引入 PyTorch 库

# NB: The sym_* functions are used via getattr() and must be imported here.
from torch import (           # 从 torch 模块中导入以下符号，但不直接使用
    sym_float,
    sym_ite,
    sym_max,
    sym_min,
    sym_not,
    SymBool,
    SymFloat,
    SymInt,
)

from torch.fx.experimental._sym_dispatch_mode import (  # 导入符号分发模式相关函数
    handle_sym_dispatch,
    sym_function_mode,
)

if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv  # 类型检查时导入符号形状环境

log = logging.getLogger(__name__)    # 获取当前模块的日志记录器
sym_node_log = torch._logging.getArtifactLogger(__name__, "sym_node")  # 获取符号节点相关的日志记录器


__all__ = ["SymNode", "method_to_operator", "magic_methods"]   # 在使用 `from module import *` 时导入的公共名称列表


SymTypes = (SymInt, SymFloat, SymBool)   # 定义 SymInt、SymFloat 和 SymBool 的元组


def _to_symtype(t):
    if t is bool:
        return SymBool    # 将 Python 中的 bool 转换为 SymBool
    if t is int:
        return SymInt     # 将 Python 中的 int 转换为 SymInt
    if t is float:
        return SymFloat   # 将 Python 中的 float 转换为 SymFloat
    return t


# TODO: An incomplete list
# 1. Set variables to be equal when we do equality
# 2. Specialize on 0/1 when we do subtraction
class SymNode:
    """
    This is a type erased SymInt/SymFloat which we use to do actual operations.
    End users don't touch this.  Magic methods are NOT defined on this object.
    """

    def __init__(
        self,
        expr,
        shape_env,
        pytype,
        hint: Optional[Union[int, float, bool]],
        constant=None,
        fx_node=None,
        ):
        self._expr = expr
        self.shape_env = shape_env
        self.pytype = pytype
        # What's the difference between hint and constant?
        #
        # - A constant is known to be invariant across invocations of the model;
        #   it will always be this value.  We only really know this when we
        #   encounter an honest-to-goodness literal (when wrapping it into
        #   a SymNode, we set constant.)  Most of the time, constant is None
        #
        # - A hint is a *particular* value from the particular run we are
        #   tracing, but it may vary the next time around.  It's useful to
        #   keep this around, as if we need a concrete value from a SymNode,
        #   we will return the hint and guard on the expression that produced
        #   it giving the same hint next time around.  The hint is not
        #   guaranteed to be set either: if you have an unbacked SymNode,
        #   there won't be any hint; it was the result of some tensor-dependent
        #   computation, but we don't know what it actually is because we
        #   haven't actually run the tensor computation.
        #
        # If _hint is None, we will query maybe_evaluate_static(compute_hint=True)
        # in hopes that we've learned enough about the unbacked symints to
        # discharge the hint; otherwise, you're likely to just error out.
        #
        # (A previous version of this system had some optimizations to only
        # recompute when it was possible we had learned enough about the
        # unbacked symint that a hint was now possible, but as we added more
        # potential refinements to unbacked symints this got harder to keep
        # in sync, so we've deleted it for now.)
        if hint is not None:
            assert type(hint) is pytype or type(hint) is _to_symtype(pytype), (
                "Cannot create SymNode of type "
                f"{pytype} with incompatible hint of type {type(hint)}"
            )
        self._hint = hint
        self.constant: Optional[Union[int, float, bool]] = constant

        # Record the FX node of the current node if we are doing translation
        # validation. They will be used for building the input assertions for
        # the translation validation problem.
        self.fx_node = (
            fx_node if self.shape_env._translation_validation_enabled else None
        )
    # 返回对象的提示（hint），如果未设置则调用 _update_hint 方法进行更新
    def hint(self):
        if self._hint is None:
            self._update_hint()
        return self._hint

    # 检查对象是否有提示（hint），如果未设置则调用 _update_hint 方法进行更新
    def has_hint(self):
        if self._hint is None:
            self._update_hint()
        return self._hint is not None

    # 要求对象的提示（hint），如果未设置则调用 _update_hint 方法进行更新
    # 如果提示（hint）仍未设置，则根据条件返回默认值或调用 size_hint 方法获取大小提示
    def require_hint(self, fallback=None):
        if self._hint is None:
            self._update_hint()
        if self._hint is None:
            if fallback is not None:
                return fallback
            # NB: we expect this to raise
            return self.shape_env.size_hint(self.expr)
        return self._hint

    # 尝试将对象表达式转换为整数，如果不可转换则返回 None
    def maybe_as_int(self):
        if self.expr.is_number:
            return int(self.expr)
        else:
            return None

    # NB: 这里进行转换，但不确定是否合适
    # 尝试将对象表达式转换为浮点数，如果不可转换则返回 None
    def maybe_as_float(self):
        import sympy

        if isinstance(self.expr, sympy.Float):
            return float(self.expr)
        else:
            return None

    # 尝试将对象表达式转换为布尔值，如果不可转换则返回 None
    def maybe_as_bool(self):
        import sympy

        if self.expr is sympy.true:
            return True
        elif self.expr is sympy.false:
            return False
        else:
            return None

    # 检查对象的 Python 类型是否为整数
    def is_int(self):
        return self.pytype is int

    # 检查对象的 Python 类型是否为浮点数
    def is_float(self):
        return self.pytype is float

    # 检查对象的 Python 类型是否为布尔值
    def is_bool(self):
        return self.pytype is bool

    # 检查对象的提示（hint）是否为 SymInt 类型，并且提示（hint）的节点是否为嵌套整数
    def is_nested_int(self):
        # Unbacked SymInts cannot be nested int today
        return (
            self._hint is not None
            and isinstance(self._hint, SymInt)
            and self._hint.node.is_nested_int()
        )

    # 包装给定的整数为 SymNode 对象，要求 num 必须为整数类型
    def wrap_int(self, num):
        assert type(num) is int
        import sympy

        return SymNode(
            sympy.Integer(num), self.shape_env, int, num, constant=num, fx_node=num
        )

    # 包装给定的浮点数为 SymNode 对象，要求 num 必须为浮点数类型
    def wrap_float(self, num):
        assert type(num) is float
        import sympy

        return SymNode(
            sympy.Float(num), self.shape_env, float, num, constant=num, fx_node=num
        )

    # 包装给定的布尔值为 SymNode 对象，要求 num 必须为布尔值类型
    def wrap_bool(self, num):
        assert type(num) is bool
        import sympy

        return SymNode(
            sympy.true if num else sympy.false,
            self.shape_env,
            bool,
            num,
            constant=num,
            fx_node=num,
        )

    # 克隆当前对象，返回自身
    def clone(self):
        return self

    # 返回对象表达式的字符串表示
    def str(self):
        return f"{self.expr}"

    # 返回对象表达式的字符串表示，与 str 方法功能相同
    def __str__(self):
        return self.str()

    # 返回对象表达式的字符串表示，与 str 方法功能相同
    def __repr__(self):
        return self.str()

    # 这些方法调用元编程方法，手动编写以获取良好的堆栈跟踪
    # 返回对象的绝对值 SymNode 对象
    def abs(self) -> "SymNode":
        return self._abs()  # type: ignore[attr-defined]

    # 返回对象的正数 SymNode 对象
    def pos(self) -> "SymNode":
        return self._pos()  # type: ignore[attr-defined]

    # 返回对象的四舍五入后的 SymNode 对象，如果指定 ndigits 则四舍五入到指定小数位数
    def round(self, ndigits=None) -> "SymNode":
        return self._round(ndigits)  # type: ignore[attr-defined]

    # 返回对象的截断整数 SymNode 对象
    def trunc(self) -> "SymNode":
        return self._trunc()  # type: ignore[attr-defined]
    def add(self, other) -> "SymNode":
        return self._add(other)  # 调用内部方法 _add，执行加法操作，返回结果作为 SymNode 对象

    def sub(self, other) -> "SymNode":
        return self._sub(other)  # 调用内部方法 _sub，执行减法操作，返回结果作为 SymNode 对象

    def mul(self, other) -> "SymNode":
        return self._mul(other)  # 调用内部方法 _mul，执行乘法操作，返回结果作为 SymNode 对象

    def mod(self, other) -> "SymNode":
        return self._mod(other)  # 调用内部方法 _mod，执行取模操作，返回结果作为 SymNode 对象

    def float_pow(self, other) -> "SymNode":
        return self._float_pow(other)  # 调用内部方法 _float_pow，执行浮点数指数运算，返回结果作为 SymNode 对象

    def pow_by_natural(self, other) -> "SymNode":
        return self._pow_by_natural(other)  # 调用内部方法 _pow_by_natural，执行自然数幂运算，返回结果作为 SymNode 对象

    def and_(self, other) -> "SymNode":
        return self._and_(other)  # 调用内部方法 _and_，执行按位与操作，返回结果作为 SymNode 对象

    def or_(self, other) -> "SymNode":
        return self._or_(other)  # 调用内部方法 _or_，执行按位或操作，返回结果作为 SymNode 对象

    def float_truediv(self, other) -> "SymNode":
        return self._float_truediv(other)  # 调用内部方法 _float_truediv，执行浮点数真除操作，返回结果作为 SymNode 对象

    def int_truediv(self, other) -> "SymNode":
        return self._int_truediv(other)  # 调用内部方法 _int_truediv，执行整数真除操作，返回结果作为 SymNode 对象

    def int_floordiv(self, other) -> "SymNode":
        return self._int_floordiv(other)  # 调用内部方法 _int_floordiv，执行整数地板除操作，返回结果作为 SymNode 对象

    def lshift(self, other) -> "SymNode":
        return self._lshift(other)  # 调用内部方法 _lshift，执行左移操作，返回结果作为 SymNode 对象

    def rshift(self, other) -> "SymNode":
        return self._rshift(other)  # 调用内部方法 _rshift，执行右移操作，返回结果作为 SymNode 对象

    def sym_not(self) -> "SymNode":  # noqa: F811
        return self._sym_not()  # 调用内部方法 _sym_not，执行逻辑非操作，返回结果作为 SymNode 对象

    def eq(self, other) -> "SymNode":
        return self._eq(other)  # 调用内部方法 _eq，执行等于比较操作，返回结果作为 SymNode 对象

    def ne(self, other) -> "SymNode":
        return self._ne(other)  # 调用内部方法 _ne，执行不等于比较操作，返回结果作为 SymNode 对象

    def gt(self, other) -> "SymNode":
        return self._gt(other)  # 调用内部方法 _gt，执行大于比较操作，返回结果作为 SymNode 对象

    def lt(self, other) -> "SymNode":
        return self._lt(other)  # 调用内部方法 _lt，执行小于比较操作，返回结果作为 SymNode 对象

    def le(self, other) -> "SymNode":
        return self._le(other)  # 调用内部方法 _le，执行小于等于比较操作，返回结果作为 SymNode 对象

    def ge(self, other) -> "SymNode":
        return self._ge(other)  # 调用内部方法 _ge，执行大于等于比较操作，返回结果作为 SymNode 对象

    def floor(self) -> "SymNode":
        return self._floor()  # 调用内部方法 _floor，执行取下界操作，返回结果作为 SymNode 对象

    def is_integer(self) -> "SymNode":
        return self._is_integer()  # 调用内部方法 _is_integer，判断是否为整数，返回结果作为 SymNode 对象

    def sym_float(self) -> "SymNode":  # noqa: F811
        return self._sym_float()  # 调用内部方法 _sym_float，执行转换为浮点数操作，返回结果作为 SymNode 对象

    def sym_int(self) -> "SymNode":
        return self._sym_int()  # 调用内部方法 _sym_int，执行转换为整数操作，返回结果作为 SymNode 对象

    def ceil(self) -> "SymNode":
        return self._ceil()  # 调用内部方法 _ceil，执行取上界操作，返回结果作为 SymNode 对象

    def neg(self) -> "SymNode":
        return self._neg()  # 调用内部方法 _neg，执行取负操作，返回结果作为 SymNode 对象

    def sym_min(self, other) -> "SymNode":  # noqa: F811
        return self._sym_min(other)  # 调用内部方法 _sym_min，执行取最小值操作，返回结果作为 SymNode 对象

    def sym_max(self, other) -> "SymNode":  # noqa: F811
        return self._sym_max(other)  # 调用内部方法 _sym_max，执行取最大值操作，返回结果作为 SymNode 对象
    # 返回一个符号节点，表示条件表达式的三元运算符
    def sym_ite(self, then_val, else_val) -> "SymNode":
        return self._sym_ite(then_val, else_val)  # type: ignore[attr-defined]

    # 返回一个符号节点，表示检查数组是否是连续存储的
    def is_contiguous(self, sizes, strides) -> "SymNode":
        return self._is_contiguous(sizes, strides)  # type: ignore[attr-defined]

    # 返回一个符号节点，表示检查二维数组是否是按通道最后一的顺序连续存储的
    def is_channels_last_contiguous_2d(self, sizes, strides) -> "SymNode":
        return self._is_channels_last_contiguous_2d(sizes, strides)  # type: ignore[attr-defined]

    # 返回一个符号节点，表示检查三维数组是否是按通道最后一的顺序连续存储的
    def is_channels_last_contiguous_3d(self, sizes, strides) -> "SymNode":
        return self._is_channels_last_contiguous_3d(sizes, strides)  # type: ignore[attr-defined]

    # 返回一个符号节点，表示检查二维数组的步幅是否按通道最后一的顺序存储
    def is_channels_last_strides_2d(self, sizes, strides) -> "SymNode":
        return self._is_channels_last_strides_2d(sizes, strides)  # type: ignore[attr-defined]

    # 返回一个符号节点，表示检查三维数组的步幅是否按通道最后一的顺序存储
    def is_channels_last_strides_3d(self, sizes, strides) -> "SymNode":
        return self._is_channels_last_strides_3d(sizes, strides)  # type: ignore[attr-defined]

    # 返回一个符号节点，表示检查数组是否是非重叠且密集的指示器
    def is_non_overlapping_and_dense_indicator(self, sizes, strides) -> "SymNode":
        return self._is_non_overlapping_and_dense_indicator(sizes, strides)  # type: ignore[attr-defined]

    # 返回一个符号节点，表示执行逻辑或运算
    # 用于使 C++ 代码兼容
    def sym_or(self, other):
        return self.or_(other)

    # 返回一个符号节点，表示执行逻辑与运算
    def sym_and(self, other):
        return self.and_(other)

    # 返回浮点数除法的符号节点
    # 用于使 C++ 代码兼容，因为没有 int_truediv 函数
    def truediv(self, other):
        return self.float_truediv(other)

    # 返回整数除法的符号节点
    def floordiv(self, other) -> "SymNode":
        return self.int_floordiv(other)

    # 返回浮点数乘方的符号节点
    # 用于使 C++ 代码兼容，因为没有 integer pow 函数
    def pow(self, other):
        return self.float_pow(other)

    # 返回一个符号节点，表示检查数组是否是非重叠且密集的
    def is_non_overlapping_and_dense(self, sizes, strides):
        return self.is_non_overlapping_and_dense_indicator(sizes, strides).eq(to_node(self, 1))  # type: ignore[attr-defined]

    # 返回一个整数的符号节点
    # 当需要手动触发 guard 时使用
    def int_(self):
        return self.guard_int("", 0)  # NB: uses Python backtrace

    # 返回一个整数 guard，用于诊断为什么需要 guard
    def guard_int(self, file, line):
        # TODO: 使用 file/line 提供有用的诊断信息，说明为什么需要 guard
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
        try:
            return int(r)
        except Exception:
            log.warning("Failed to convert to int: %s", r)
            raise

    # 返回一个浮点数 guard，用于诊断为什么需要 guard
    def guard_float(self, file, line):
        # TODO: 使用 file/line 提供有用的诊断信息，说明为什么需要 guard
        r = self.shape_env.evaluate_expr(
            self.expr, self.hint, fx_node=self.fx_node, expect_rational=False
        )
        try:
            return float(r)
        except Exception:
            log.warning("Failed to convert to float: %s", r)
            raise
    # 使用给定的文件名和行号，在形状环境中评估表达式，返回布尔结果
    def guard_bool(self, file, line):
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
        try:
            # 尝试将评估结果转换为布尔值并返回
            return bool(r)
        except Exception:
            # 如果转换失败，记录警告信息并重新抛出异常
            log.warning("Failed to convert to bool: %s", r)
            raise

    # 使用给定的文件名和行号，检查是否可以生成守护条件
    def expect_true(self, file, line):
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        if (
            self.has_hint()
            and not free_unbacked_symbols(self.expr)
            and not self.shape_env.prefer_deferred_runtime_asserts_over_guards
        ):
            # 如果符号表达式有提示，并且没有自由未支持的符号，并且不优先使用延迟运行时断言，则可以生成守护条件
            return self.guard_bool(file, line)
        # 否则，生成延迟运行时断言，并使用给定的文件名和行号
        return self.shape_env.defer_runtime_assert(
            self.expr, f"{file}:{line}", fx_node=self.fx_node
        )

    # 使用给定的文件名和行号，检查是否可以生成守护条件，并返回结果
    def expect_size(self, file, line):
        from torch.fx.experimental.symbolic_shapes import _advise_is_size

        b = self.ge(self.wrap_int(0))
        # 生成延迟运行时断言，并使用给定的文件名和行号
        r = b.expect_true(file, line)
        # 如果结果为真且没有提示信息，推荐对大小进行进一步的编译时范围精化
        if r and not self.has_hint():
            _advise_is_size(SymInt(self))
        return r

    # 使用给定的文件名和行号，在形状环境中评估表达式，返回布尔结果，支持大小无关
    def guard_size_oblivious(self, file, line):
        """
        Like guard_bool, but if we encounter unbacked symbols, if those symbols
        are size-like, we will treat them as >= 2 for the purposes of the analysis.

        This CHANGES the runtime semantics, but all size-oblivious sites have been
        audited to ensure that the runtime semantics don't change in a material way.
        Acceptable runtime semantic changes are, e.g., squeeze() no longer dropping
        an unbacked one size, or a tensor reporting as non-contiguous even if it's
        contiguous if it would have been reported contiguous due to being empty.
        """
        # 使用给定的文件名和行号，在形状环境中评估表达式，返回布尔结果，支持大小无关
        r = self.shape_env.evaluate_expr(
            self.expr, self.hint, fx_node=self.fx_node, size_oblivious=True
        )
        try:
            # 尝试将评估结果转换为布尔值并返回
            return bool(r)
        except Exception:
            # 如果转换失败，记录警告信息并重新抛出异常
            log.warning("Failed to convert to bool: %s", r)
            raise

    # 返回空字符串和零，调用guard_bool方法
    def bool_(self):
        return self.guard_bool("", 0)

    # 返回True
    def is_symbolic(self):
        return True

    # 返回None
    def nested_int(self):
        return None

    # 返回False
    def is_constant(self):
        return False
# 定义一个字典，将方法名映射到对应的操作函数或方法
METHOD_TO_OPERATOR = {
    "pos": operator.pos,                    # 将正号运算符映射到operator模块中的pos函数
    "abs": operator.abs,                    # 将绝对值运算符映射到operator模块中的abs函数
    "add": operator.add,                    # 将加法运算符映射到operator模块中的add函数
    "and": operator.and_,                   # 将按位与运算符映射到operator模块中的and_函数
    "ceil": math.ceil,                      # 将向上取整函数映射到math模块中的ceil函数
    "eq": operator.eq,                      # 将相等运算符映射到operator模块中的eq函数
    "floor": math.floor,                    # 将向下取整函数映射到math模块中的floor函数
    "trunc": math.trunc,                    # 将截断函数映射到math模块中的trunc函数
    "int_floordiv": operator.floordiv,      # 将整数除法运算符映射到operator模块中的floordiv函数
    "ge": operator.ge,                      # 将大于等于运算符映射到operator模块中的ge函数
    "gt": operator.gt,                      # 将大于运算符映射到operator模块中的gt函数
    "is_integer": lambda x: x.is_integer(), # 将判断是否为整数的方法映射到lambda表达式
    "le": operator.le,                      # 将小于等于运算符映射到operator模块中的le函数
    "lshift": operator.lshift,              # 将左移位运算符映射到operator模块中的lshift函数
    "lt": operator.lt,                      # 将小于运算符映射到operator模块中的lt函数
    "mod": operator.mod,                    # 将取模运算符映射到operator模块中的mod函数
    "mul": operator.mul,                    # 将乘法运算符映射到operator模块中的mul函数
    "ne": operator.ne,                      # 将不等于运算符映射到operator模块中的ne函数
    "neg": operator.neg,                    # 将负号运算符映射到operator模块中的neg函数
    "or": operator.or_,                     # 将按位或运算符映射到operator模块中的or_函数
    "float_pow": operator.pow,              # 将浮点数幂运算符映射到operator模块中的pow函数
    "pow_by_natural": operator.pow,         # 将自然数幂运算符映射到operator模块中的pow函数
    "round": builtins.round,                # 将四舍五入函数映射到builtins模块中的round函数
    "rshift": operator.rshift,              # 将右移位运算符映射到operator模块中的rshift函数
    "sub": operator.sub,                    # 将减法运算符映射到operator模块中的sub函数
    "sym_float": sym_float,                 # 将自定义的sym_float函数映射到sym_float变量
    "sym_ite": sym_ite,                     # 将自定义的sym_ite函数映射到sym_ite变量
    "sym_max": sym_max,                     # 将自定义的sym_max函数映射到sym_max变量
    "sym_min": sym_min,                     # 将自定义的sym_min函数映射到sym_min变量
    "sym_not": sym_not,                     # 将自定义的sym_not函数映射到sym_not变量
    "float_truediv": operator.truediv,      # 将浮点数除法运算符映射到operator模块中的truediv函数
    "int_truediv": operator.truediv,        # 将整数除法运算符映射到operator模块中的truediv函数
}

# 包含一元魔术方法名的集合
unary_magic_methods = {
    "abs", "sym_float", "sym_int", "ceil", "floor", "neg", "sym_not", "pos", "trunc"
}

# 定义一个函数，用于返回按指定名称调用的符号节点的方法
def _get_sym_node_fn(name):
    def fn(self):
        return getattr(self, f"_sym_{name}")()
    return fn

# 数学运算名称的元组
math_op_names = (
    "sqrt", "cos", "cosh", "sin", "sinh", "tan", "tanh", "asin", "acos", "atan"
)

# 对于每个数学操作名称，动态定义相关的符号方法和操作映射
for name in math_op_names:
    sym_name = f"sym_{name}"
    priv_sym_name = f"_{sym_name}"
    setattr(SymNode, sym_name, _get_sym_node_fn(name))         # 在SymNode类上动态设置符号方法
    METHOD_TO_OPERATOR[sym_name] = getattr(torch, priv_sym_name)  # 将torch模块中的函数映射到操作字典中
    unary_magic_methods.add(sym_name)                          # 将符号方法名添加到一元魔术方法集合中
    __all__.append(sym_name)                                   # 将符号方法名添加到__all__列表中

# 包含非魔术一元方法名的集合
unary_nonmagic_methods = {
    "is_integer",
}

# 包含所有一元方法名的集合
unary_methods = unary_magic_methods | unary_nonmagic_methods

# 大多数方法仅在SymInt和SymFloat上注册
# 一些方法仅在SymBool上注册
only_bool_magic_methods = {"and", "or", "sym_not", "sym_ite"}

# 将SymBool隐式转换为SymInt的方法集合
bool_becomes_int_magic_methods = {"add", "sub", "mul"}

# 除了在SymInt和SymFloat上之外，还在SymBool上注册的方法集合
also_bool_magic_methods = {"eq"}

# 包含所有布尔运算方法名的集合
bool_magic_methods = only_bool_magic_methods | also_bool_magic_methods

# 仅用于浮点数的方法集合
only_float_magic_methods = {"is_integer", "round", "sym_int"}

# 方法名末尾带有下划线的运算符魔术方法集合
magic_methods_on_operator_with_trailing_underscore = {"and", "or"}

# 总是返回浮点数的方法集合
always_float_magic_methods = {"int_truediv", "float_truediv", "sym_float", "float_pow"}

# 总是返回整数的方法集合
always_int_magic_methods = {"ceil", "floor", "trunc", "pow_by_natural"}

# 总是返回布尔值的方法集合
always_bool_magic_methods = {
    "eq", "ne", "gt", "lt", "le", "ge", "and", "or", "sym_not",
    "is_non_overlapping_and_dense", "is_integer"
}

# 具有`__foo__`和`__rfoo__`的方法集合
    # 从 torch.utils._sympy.functions 模块中导入 FloatTrueDiv 函数
    from torch.utils._sympy.functions import FloatTrueDiv
    
    # 调用 FloatTrueDiv 函数，执行浮点数的真除法操作，并返回结果
    return FloatTrueDiv(a, b)
# 定义一个函数，实现符号表达式中的整数真除法
def _sympy_int_truediv(a, b):
    # 从torch.utils._sympy.functions模块导入IntTrueDiv函数并调用
    from torch.utils._sympy.functions import IntTrueDiv
    return IntTrueDiv(a, b)

# 定义一个函数，实现符号表达式中的整数向下取整除法
def _sympy_floordiv(a, b):
    # 从torch.utils._sympy.functions模块导入FloorDiv函数并调用
    from torch.utils._sympy.functions import FloorDiv
    return FloorDiv(a, b)

# 定义一个函数，实现符号表达式中的取模运算，选择性使用不同的取模函数
def _sympy_mod(a, b):
    # 从torch.utils._sympy.functions模块导入Mod和PythonMod函数，并根据条件选择使用其中之一
    from torch.utils._sympy.functions import Mod, PythonMod
    if a.is_nonnegative and b.is_nonnegative:
        return Mod(a, b)
    else:
        return PythonMod(a, b)

# 定义一个函数，实现符号表达式中的自然数幂运算
def _sympy_pow_by_natural(a, b):
    # 从torch.utils._sympy.functions模块导入PowByNatural函数并调用
    from torch.utils._sympy.functions import PowByNatural
    return PowByNatural(a, b)

# 定义一个函数，实现符号表达式中的浮点数幂运算
def _sympy_float_pow(a, b):
    # 从torch.utils._sympy.functions模块导入FloatPow函数并调用
    from torch.utils._sympy.functions import FloatPow
    return FloatPow(a, b)

# 定义一个函数，实现符号表达式中的逻辑与运算
def _sympy_and(a, b):
    # 导入sympy模块并调用其中的And函数
    import sympy
    return sympy.And(a, b)

# 定义一个函数，实现符号表达式中的逻辑或运算
def _sympy_or(a, b):
    # 导入sympy模块并调用其中的Or函数
    import sympy
    return sympy.Or(a, b)

# 定义一个函数，实现符号表达式中的左移运算
def _sympy_lshift(a, b):
    # 从torch.utils._sympy.functions模块导入LShift函数并调用
    from torch.utils._sympy.functions import LShift
    return LShift(a, b)

# 定义一个函数，实现符号表达式中的右移运算
def _sympy_rshift(a, b):
    # 从torch.utils._sympy.functions模块导入RShift函数并调用
    from torch.utils._sympy.functions import RShift
    return RShift(a, b)

# 定义一个字典，将操作符映射到相应的函数
reflectable_magic_methods = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "mod": _sympy_mod,
    "pow_by_natural": _sympy_pow_by_natural,
    "float_pow": _sympy_float_pow,
    "and": _sympy_and,
    "or": _sympy_or,
    "float_truediv": _sympy_float_truediv,  # 注意：此处未定义_sympy_float_truediv函数
    "int_truediv": _sympy_int_truediv,
    "int_floordiv": _sympy_floordiv,
    "lshift": _sympy_lshift,
    "rshift": _sympy_rshift,
}

# 定义一个辅助函数，用于处理符号表达式的地板和天花板操作
def _floor_ceil_helper(a, fn):
    import sympy

    # 如果a是乘法表达式并且包含两个参数，且第一个参数是浮点数且第二个参数是整数，则返回整数倍
    if isinstance(a, sympy.Mul):
        aa = a.args
        if len(aa) == 2 and isinstance(aa[0], sympy.Float) and aa[1].is_integer:
            coef = sympy.Integer(aa[0])
            if aa[0] == coef:  # 结构相等性测试
                return coef * aa[1]
    # 如果a是浮点数并且与其整数等同或者a是整数，则返回a的整数形式
    if (
        isinstance(a, sympy.Float)
        and a == sympy.Integer(a)
        or isinstance(a, sympy.Integer)
    ):
        return sympy.Integer(a)
    # 否则应用传入的函数fn到a上
    return fn(a)

# 定义一个函数，实现符号表达式中的向下取整操作
def _sympy_floor(a):
    # 从torch.utils._sympy.functions模块导入FloorToInt函数并调用
    from torch.utils._sympy.functions import FloorToInt
    return FloorToInt(a)

# 注意：此函数实现了Python中的截断语义，返回一个整数。不要用于表示torch.trunc（它是浮点到浮点的操作）
def _sympy_trunc(a):
    # 从torch.utils._sympy.functions模块导入TruncToInt函数并调用
    from torch.utils._sympy.functions import TruncToInt
    return TruncToInt(a)

# 定义一个函数，实现符号表达式中的向上取整操作
def _sympy_ceil(a):
    # 从torch.utils._sympy.functions模块导入CeilToInt函数并调用
    from torch.utils._sympy.functions import CeilToInt
    return CeilToInt(a)

# 定义一个函数，实现符号表达式中的等于比较操作
def _sympy_eq(a, b):
    # 导入sympy模块并调用其中的Eq函数
    import sympy
    return sympy.Eq(a, b)

# 定义一个函数，实现符号表达式中的不等于比较操作
def _sympy_ne(a, b):
    # 导入sympy模块并调用其中的Ne函数
    import sympy
    return sympy.Ne(a, b)

# 定义一个函数，实现符号表达式中的大于比较操作
def _sympy_gt(a, b):
    # 导入sympy模块并调用其中的Gt函数
    import sympy
    return sympy.Gt(a, b)

# 定义一个函数，实现符号表达式中的小于比较操作
def _sympy_lt(a, b):
    # 导入sympy模块并调用其中的Lt函数
    import sympy
    return sympy.Lt(a, b)

# 定义一个函数，实现符号表达式中的小于等于比较操作
def _sympy_le(a, b):
    # 导入sympy模块并调用其中的Le函数
    import sympy
    return sympy.Le(a, b)

# 定义一个函数，实现符号表达式中的大于等于比较操作
def _sympy_ge(a, b):
    # 导入sympy模块并调用其中的Ge函数
    import sympy
    return sympy.Ge(a, b)

# 定义一个函数，实现符号表达式中的最小值操作
def _sympy_min(a, b):
    # 导入sympy模块并调用其中的Min函数
    import sympy
    return sympy.Min(a, b)

# 定义一个函数，实现符号表达式中的最大值操作
def _sympy_max(a, b):
    # 导入sympy模块并调用其中的Max函数
    import sympy
    return sympy.Max(a, b)

# 定义一个函数，实现符号表达式中的条件表达式操作
def _sympy_ite(a, t, f):
    # 导入sympy模块并调用其中的Ite函数
    import sympy
    # 返回一个 sympy.Piecewise 对象，根据条件选择返回 t 或者 f
    return sympy.Piecewise((t, a), (f, True))
current_module = sys.modules[__name__]
# 获取当前模块对象，用于动态设置属性

def _get_sym_math_fn(name):
    # 定义一个函数，返回一个闭包函数，用于执行特定数学操作
    def fn(a):
        import torch.utils._sympy.functions
        # 动态导入 torch.utils._sympy.functions 模块中的函数
        return getattr(torch.utils._sympy.functions, f"OpaqueUnaryFn_{name}")(a)
        # 调用指定名称的函数，并传入参数 a

    return fn
    # 返回闭包函数 fn

for name in math_op_names:
    # 遍历 math_op_names 中的每个名称
    priv_sympy_name = f"_sympy_{name}"
    # 根据每个名称生成私有符号名称
    fn = _get_sym_math_fn(name)
    # 调用 _get_sym_math_fn 函数获取特定数学操作的闭包函数
    fn.__qualname__ = fn.__name__ = priv_sympy_name
    # 设置闭包函数的名称属性为生成的私有符号名称
    setattr(current_module, priv_sympy_name, fn)
    # 将闭包函数作为属性设置到当前模块对象中

del fn, name, priv_sympy_name  # type: ignore[possibly-undefined]
# 删除变量 fn、name 和 priv_sympy_name，用于清理作用域

def _sympy_abs(a):
    import sympy
    # 导入 sympy 模块

    return sympy.Abs(a)
    # 返回参数 a 的绝对值

def _sympy_round(number, ndigits=None):
    from torch.utils._sympy.functions import RoundDecimal, RoundToInt
    # 导入 RoundDecimal 和 RoundToInt 函数

    if ndigits is None:
        return RoundToInt(number)
        # 如果 ndigits 为 None，则调用 RoundToInt 函数取整数部分
    else:
        return RoundDecimal(number, ndigits)
        # 否则调用 RoundDecimal 函数进行指定位数的四舍五入

def _sympy_sym_float(a):
    from torch.utils._sympy.functions import ToFloat
    # 导入 ToFloat 函数

    # NB: 不能使用 a * 1.0，因为 0 * 1.0 是 0，会错误地报告为整数
    return ToFloat(a)
    # 返回将参数 a 转换为浮点数的结果

def _sympy_is_integer(a):
    import sympy
    from torch.utils._sympy.functions import ToFloat
    # 导入 sympy 和 ToFloat 函数

    return sympy.Eq(ToFloat(sympy.floor(a)), a)
    # 返回判断参数 a 是否为整数的结果

magic_methods = {
    **reflectable_magic_methods,
    "sym_not": operator.invert,
    "pos": operator.pos,
    "eq": _sympy_eq,
    "ne": _sympy_ne,
    "gt": _sympy_gt,
    "lt": _sympy_lt,
    "le": _sympy_le,
    "ge": _sympy_ge,
    "floor": _sympy_floor,
    "trunc": _sympy_trunc,
    "sym_float": _sympy_sym_float,
    "ceil": _sympy_ceil,
    "neg": operator.neg,
    "sym_min": _sympy_min,
    "sym_max": _sympy_max,
    "sym_ite": _sympy_ite,
    "abs": _sympy_abs,
    "round": _sympy_round,
    "is_integer": _sympy_is_integer,
}
# 定义包含各种数学操作的字典 magic_methods

for name in math_op_names:
    # 遍历 math_op_names 中的每个名称
    sym_name = f"sym_{name}"
    # 根据每个名称生成带有前缀 "sym_" 的名称
    magic_methods[sym_name] = getattr(current_module, f"_sympy_{name}")
    # 将当前模块中对应的私有函数添加到 magic_methods 字典中

del name, sym_name, math_op_names, current_module  # type: ignore[possibly-undefined]
# 删除变量 name、sym_name、math_op_names 和 current_module，用于清理作用域

def sympy_is_contiguous(sizes, strides):
    dim = len(sizes)
    # 获取尺寸大小的维度

    return sympy_is_contiguous_generic(sizes, strides, list(range(dim - 1, -1, -1)))
    # 调用通用版本的连续性检查函数，传入尺寸、步长和反向的维度顺序列表

def sympy_is_contiguous_generic(sizes, strides, dim_order):
    import sympy
    # 导入 sympy 模块

    dim = len(sizes)
    # 获取尺寸大小的维度

    if len(dim_order) != dim:
        return sympy.false
        # 如果维度顺序列表长度与尺寸维度不符，则返回 false

    is_contiguous = sympy.true
    # 初始化 is_contiguous 为 true
    z = sympy.Integer(1)
    # 初始化 z 为整数 1

    # 若步长合理（或维度大小为 1），则是连续的
    for d in dim_order:
        is_contiguous &= sympy.Eq(sizes[d], sympy.Integer(1)) | sympy.Eq(strides[d], z)
        z *= sizes[d]
    # 或者若任何尺寸为零，则是连续的
    for d in range(dim):
        is_contiguous |= sympy.Eq(sizes[d], sympy.Integer(0))

    return is_contiguous
    # 返回连续性检查的结果

# NB: C++ 中有一个 TODO，允许省略批次维度。如果发生这种情况，您需要重新设计此部分
# 笔记：C++ 中有一个待办事项，允许省略批次维度。如果这种情况发生，需要重新设计此部分

def sympy_is_channels_last_contiguous_2d(sizes, strides):
    return sympy_is_contiguous_generic(sizes, strides, [1, 3, 2, 0])
    # 检查 2D 通道是否以最后的方式连续

def sympy_is_channels_last_contiguous_3d(sizes, strides):
    return sympy_is_contiguous_generic(sizes, strides, [1, 4, 3, 2, 0])
    # 检查 3D 通道是否以最后的方式连续
# 定义一个函数，用于检查是否使用了通道末尾存储顺序的泛化步长
def sympy_is_channels_last_strides_generic(sizes, strides, dim_order):
    # 导入 sympy 库
    import sympy

    # 获取维度数量
    dim = len(sizes)

    # 如果维度数量与维度顺序列表长度不一致，则返回 False
    if dim != len(dim_order):
        return sympy.false

    # 初始化 m 为整数 0
    m = sympy.Integer(0)
    # 初始化 r 为逻辑真值
    r = sympy.true

    # 对于通道 C 维度的特殊情况，默认使用 NCHW 排列
    r &= sympy.Ne(strides[1], 0)

    # 遍历维度顺序列表
    for d in dim_order:
        # 检查当前维度大小不为 0 且步长大于等于 m
        r &= sympy.Ne(sizes[d], 0) & (strides[d] >= m)

        # 对于维度为 0 的情况，如果 m 不等于 strides[1]，则返回 False
        if d == 0:
            r &= sympy.Ne(m, strides[1])

        # 以下内容用于：
        # 1. 区分 N1H1 的存储顺序；
        # 2. 处理 1C1W 的置换情况，不应将其识别为通道末尾存储顺序
        # 这些情况是隐式内存格式与步长的缺陷所在。
        # 例如，对于尺寸为 1 的维度，可能存在相同步长的两种不同情况。
        m = strides[d] * sympy.Max(sizes[d], 1)

    # 返回计算结果 r
    return r


# 定义一个函数，用于检查是否使用了通道末尾存储顺序的二维步长
def sympy_is_channels_last_strides_2d(sizes, strides):
    # 调用泛化函数检查通道末尾存储顺序
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 3, 2, 0])


# 定义一个函数，用于检查是否使用了通道末尾存储顺序的三维步长
def sympy_is_channels_last_strides_3d(sizes, strides):
    # 调用泛化函数检查通道末尾存储顺序
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 4, 3, 2, 0])


# 定义一个函数，用于返回非重叠且密集指示器的 sympy 实现
def _sympy_is_non_overlapping_and_dense_indicator(sizes, strides):
    # 导入 torch.utils._sympy.functions 模块中的 IsNonOverlappingAndDenseIndicator 函数
    from torch.utils._sympy.functions import IsNonOverlappingAndDenseIndicator

    # 调用 IsNonOverlappingAndDenseIndicator 函数
    return IsNonOverlappingAndDenseIndicator(*sizes, *strides)


# 定义一个字典，包含各种尺寸和步长方法的名称及其对应的函数
sizes_strides_methods = {
    # TODO: 这些也可以使用指示器完成，也许这样推理更好
    "is_contiguous": sympy_is_contiguous,
    "is_channels_last_contiguous_2d": sympy_is_channels_last_contiguous_2d,
    "is_channels_last_contiguous_3d": sympy_is_channels_last_contiguous_3d,
    "is_channels_last_strides_2d": sympy_is_channels_last_strides_2d,
    "is_channels_last_strides_3d": sympy_is_channels_last_strides_3d,
    "is_non_overlapping_and_dense_indicator": _sympy_is_non_overlapping_and_dense_indicator,
}

# 定义一个字典，包含基于提示进行替代实现的方法名称及其对应的函数
alternate_impl_if_hinted_methods = {
    "sym_min": builtins.min,
    "sym_max": builtins.max,
}


# 定义一个方法，用于将数值转换为节点表示
def to_node(self, num):
    # 如果 num 是 SymTypes 类型的实例，则返回其节点表示
    if isinstance(num, SymTypes):
        return num.node
    # 如果 num 是布尔类型，则包装成相应的布尔节点
    elif type(num) is bool:
        return self.wrap_bool(num)
    # 如果 num 是整数类型，则包装成相应的整数节点
    elif type(num) is int:
        return self.wrap_int(num)
    # 如果 num 是浮点数类型，则包装成相应的浮点数节点
    elif type(num) is float:
        return self.wrap_float(num)
    else:
        # 否则返回 NotImplemented，以便 Python 尝试其他魔术方法
        return NotImplemented


# 定义一个方法，用于包装节点表示
def wrap_node(x):
    # TODO: 让 C++ 也能充分利用这一点
    # 如果 x 是 SymNode 类型的实例且其常量部分不为空，则返回其常量部分
    if isinstance(x, SymNode) and x.constant is not None:
        return x.constant
    # 检查对象 x 是否为整数类型
    if x.is_int():
        # 如果是整数类型，返回一个 SymInt 类型的对象，该对象使用 x 初始化
        return SymInt(x)
    # 如果 x 不是整数类型，则检查是否为浮点数类型
    elif x.is_float():
        # 如果是浮点数类型，返回一个 SymFloat 类型的对象，该对象使用 x 初始化
        return SymFloat(x)
    # 如果 x 既不是整数也不是浮点数类型，则检查是否为布尔类型
    elif x.is_bool():
        # 如果是布尔类型，返回一个 SymBool 类型的对象，该对象使用 x 初始化
        return SymBool(x)
    else:
        # 如果 x 类型未被识别，引发断言错误，提供错误信息指明未识别的返回类型
        raise AssertionError(f"unrecognized return type {x}")
def method_to_operator(method):
    return METHOD_TO_OPERATOR[method]

# 根据给定的方法名 `method`，返回其对应的操作符或函数


def _make_node_magic(method, func):
    func = lru_cache(256)(func)

# 使用 `lru_cache(256)` 装饰函数 `func`，使其缓存最近调用的 256 个结果


    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f"{method}_"
    else:
        method_attr = method

# 根据 `method` 是否在 `magic_methods_on_operator_with_trailing_underscore` 中，决定生成的方法属性名 `method_attr`


    def unary_magic_impl(self):
        from torch.fx.experimental.symbolic_shapes import safe_expand

        op = method_to_operator(method)

# 定义一个名为 `unary_magic_impl` 的内部函数，获取方法对应的操作符或函数 `op`


        if sym_function_mode():
            return to_node(self, handle_sym_dispatch(op, (wrap_node(self),), {}))

# 如果处于符号函数模式，则调用 `handle_sym_dispatch` 处理符号分派，并返回处理结果的节点表示


        expr = self.expr
        if method == "floor" or method == "ceiling":
            expr = self.shape_env._simplify_floor_div(expr)

# 如果方法是 "floor" 或 "ceiling"，则使用 `self.shape_env._simplify_floor_div` 简化表达式 `expr`


        try:
            out = func(expr)
        except Exception:
            log.warning("failed to eval %s(%s)", method, expr)
            raise

# 尝试使用 `func` 对表达式 `expr` 进行评估，捕获可能的异常并记录警告信息


        sym_node_log.debug("%s %s -> %s", func, expr, out)

# 记录调试信息，显示函数 `func`、表达式 `expr` 和输出 `out`


        out_hint = None
        if self.hint is not None:
            out_hint = op(self.hint)
        out = safe_expand(out)

# 如果存在提示信息 `self.hint`，则使用 `op` 对其进行操作；然后使用 `safe_expand` 扩展输出 `out`


        pytype: Type
        if method in always_int_magic_methods:
            pytype = int
        elif method in always_bool_magic_methods:
            pytype = bool
        elif method in always_float_magic_methods:
            pytype = float
        else:
            pytype = self.pytype

# 根据 `method` 是否在特定的方法集合中，确定 `pytype` 的类型；否则使用 `self.pytype`


        fx_node, _ = self.shape_env._create_fx_call_function(op, (self.fx_node,))
        return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)

# 使用 `self.shape_env._create_fx_call_function` 创建功能调用函数节点 `fx_node`，并返回一个新的 `SymNode` 对象


    if method in unary_methods:
        setattr(SymNode, f"_{method_attr}", unary_magic_impl)

# 如果 `method` 在一元方法集合 `unary_methods` 中，则将 `unary_magic_impl` 设置为 `SymNode` 类的方法 `_method_attr`
    # 如果方法选择为 "sym_ite"，则定义一个名为 sym_ite_impl 的内部函数
    def sym_ite_impl(pred_node, then_node, else_node):
        # 导入必要的模块
        from torch.fx.experimental.symbolic_shapes import safe_expand

        # 确定输出提示信息为 then_node.hint 或 else_node.hint 中的一个
        out_hint = then_node.hint if pred_node.hint else else_node.hint
        
        # 如果处于符号函数模式，则调用符号分发处理函数，并返回结果节点
        if sym_function_mode():
            return to_node(
                pred_node,
                handle_sym_dispatch(
                    sym_ite,
                    (
                        wrap_node(pred_node),
                        wrap_node(then_node),
                        wrap_node(else_node),
                    ),
                    {},
                ),
            )

        # 尝试执行 func 函数，处理 pred_node.expr, then_node.expr, else_node.expr 的表达式
        try:
            out = func(pred_node.expr, then_node.expr, else_node.expr)
        # 捕获异常，记录警告信息，并重新抛出异常
        except Exception:
            log.warning(
                "failed to eval %s(%s, %s, %s)",
                method,
                pred_node.expr,
                then_node.expr,
                else_node.expr,
            )
            raise

        # 对输出结果进行安全扩展
        out = safe_expand(out)
        
        # 创建用于 FX 调用的函数节点 fx_node，并返回符号节点 SymNode
        fx_node, _ = pred_node.shape_env._create_fx_call_function(
            sym_ite, (pred_node.fx_node, then_node.fx_node, else_node.fx_node)
        )
        return SymNode(
            out, pred_node.shape_env, then_node.pytype, out_hint, fx_node=fx_node
        )

    # 将 sym_ite_impl 函数作为 SymNode 类的特定方法属性设置
    setattr(SymNode, f"_{method_attr}", sym_ite_impl)
    elif method == "round":
        # 如果方法是 "round"，定义一个 round_impl 方法来处理圆整操作
        def round_impl(self, ndigits=None):
            # 导入需要的模块和函数
            from torch.fx.experimental.symbolic_shapes import safe_expand
            
            # 获取内置函数 round
            op = builtins.round
            # 如果处于符号函数模式，则调用符号分派处理
            if sym_function_mode():
                return to_node(
                    self, handle_sym_dispatch(op, (wrap_node(self), ndigits), {})
                )

            # 获取表达式
            expr = self.expr
            try:
                # 尝试执行函数操作
                out = func(expr, ndigits)
            except Exception:
                # 如果执行失败，记录警告信息并抛出异常
                log.warning("failed to eval %s(%s, ndigits=%s)", method, expr, ndigits)
                raise

            # 对输出进行安全扩展
            out = safe_expand(out)

            # 根据 ndigits 是否为 None，确定 pytype 类型
            if ndigits is None:
                pytype = int
            else:
                pytype = self.pytype

            # 如果存在提示信息，通过 op 函数获取提示信息
            out_hint = None
            if self.hint is not None:
                out_hint = op(self.hint, ndigits)

            # 内部使用 None 作为标记来表示不是 FX 图上的节点。但同时，无法将纯 None 包装成 FX 节点。因此，这里通过
            # 构建 args 的方式传递 ndigits=None，以避免 FX 图的问题。
            # TODO: 如果 FX 使用不同的标记，可以考虑移除下面的 args 构建逻辑。
            # ezyang(May 2024): LOL
            args = [self.fx_node]
            if ndigits is not None:
                args.append(ndigits)
            # 创建 FX 调用函数，并获取 FX 节点
            fx_node, _ = self.shape_env._create_fx_call_function(op, tuple(args))
            # 返回符号节点对象
            return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)

        # 将 round_impl 方法设置为 SymNode 类的 "_round" 属性
        setattr(SymNode, f"_{method_attr}", round_impl)
    else:
        # 如果方法不是 "round"，则将 binary_magic_impl 方法设置为 SymNode 类的对应属性
        setattr(SymNode, f"_{method_attr}", binary_magic_impl)
def _make_node_sizes_strides(method, func):
    # NB: don't LRU cache, lots of arguments
    # 定义一个内部函数 sizes_strides_impl，用于处理 sizes 和 strides 的方法
    def sizes_strides_impl(self, sizes, strides):
        # 根据方法名动态获取模块中的函数对象
        op = getattr(sys.modules[__name__], method)
        # 如果处于符号函数模式，进行符号调度处理并返回节点
        if sym_function_mode():
            return to_node(
                self,
                handle_sym_dispatch(
                    op,
                    ([wrap_node(s) for s in sizes], [wrap_node(s) for s in strides]),
                    {},
                ),
            )
        # 提取 sizes 和 strides 中的表达式列表
        size_exprs = [s.expr for s in sizes]
        stride_exprs = [s.expr for s in strides]
        try:
            # 调用给定的 func 处理 size_exprs 和 stride_exprs
            out = func(size_exprs, stride_exprs)
        except Exception:
            # 记录警告信息并抛出异常
            log.warning("failed to eval %s(%s, %s)", method, size_exprs, stride_exprs)
            raise
        # 如果方法名以 '_indicator' 结尾，类型为 int；否则为 bool
        pytype: Type
        if method.endswith("_indicator"):
            pytype = int
        else:
            pytype = bool
        # 返回 SymNode 对象，包括处理结果 out、形状环境 self.shape_env、Python 类型 pytype 和输出提示 out_hint
        return SymNode(out, self.shape_env, pytype, out_hint)

    # 将 sizes_strides_impl 函数作为 SymNode 类的方法绑定到 _{method} 属性上
    setattr(SymNode, f"_{method}", sizes_strides_impl)

    # 如果模块中不存在指定的方法名，则将 sizes_strides_user 函数绑定到模块中的 method 属性上
    # TODO: 这是一个潜在的热路径，但在理想状态下，对此进行的保护将在更高级别上解决，因此您不会在此代码中花费时间
    def sizes_strides_user(sizes, strides):
        import sympy

        from torch.fx.experimental.symbolic_shapes import (
            eval_is_non_overlapping_and_dense,
        )

        for a in itertools.chain(sizes, strides):
            # 如果 a 是 SymInt 类型，则调用 a.node 的 method 方法
            if isinstance(a, SymInt):
                return wrap_node(
                    getattr(a.node, method)(
                        [to_node(a.node, b) for b in sizes],
                        [to_node(a.node, b) for b in strides],
                    )
                )
        # 如果方法名为 'is_non_overlapping_and_dense_indicator'，则调用 eval_is_non_overlapping_and_dense 函数
        if method == "is_non_overlapping_and_dense_indicator":
            return eval_is_non_overlapping_and_dense(sizes, strides)
        else:
            # 否则，调用 func 函数，将 sizes 和 strides 转换为 sympy 符号并返回布尔值
            # TODO: 这是一个糟糕的实现
            return bool(
                func(
                    [sympy.sympify(a) for a in sizes],
                    [sympy.sympify(a) for a in strides],
                )
            )

    # 如果模块中不存在指定的方法名，将 sizes_strides_user 函数绑定到模块中的 method 属性上
    if not hasattr(sys.modules[__name__], method):
        setattr(sys.modules[__name__], method, sizes_strides_user)


# 对 magic_methods 中的每个方法调用 _make_node_magic 方法
for method, func in magic_methods.items():
    _make_node_magic(method, func)

# 对 sizes_strides_methods 中的每个方法调用 _make_node_sizes_strides 方法
for method, func in sizes_strides_methods.items():
    _make_node_sizes_strides(method, func)
    # 如果方法在需要在操作符上添加下划线的魔术方法列表中
    # 则使用带有“sym_”前缀的属性名
    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f"sym_{method}"
    else:
        # 否则直接使用方法名
        method_attr = method

    # 定义一个函数用于获取常量值
    # 参数 x 可以是 SymInt、int、SymFloat、float、SymBool 或 bool 类型
    def get_constant(x: Union[SymInt, int, SymFloat, float, SymBool, bool]):
        if isinstance(x, (int, float, bool)):
            return x
        if isinstance(x, SymBool):
            # 如果是 SymBool 类型，则调用其节点的 guard_bool 方法获取常量值
            return x.node.guard_bool("", 0)
        # 如果参数类型不符合预期，则抛出断言错误
        raise AssertionError("expect to be called with constant SymBools")

    # 定义一个函数用于判断是否为常量
    def is_constant(x):
        if isinstance(x, (int, float, bool)):
            return True
        if isinstance(x, (SymInt, SymFloat, SymBool)):
            # 如果是 SymInt、SymFloat 或 SymBool 类型，则调用其节点的 is_constant 方法判断是否为常量
            return x.node.is_constant()
        return False

    # 二元操作符的提升规则说明
    # 注意：我们保留 Python 的语义
    #   - 如果两个参数类型相同，则不做任何处理
    #   - 如果一个参数是 float，则将另一个参数提升为 float
    #       - 注意：这也适用于 floordiv，即使输出是整数（仍然是 float）
    #   - pow 是一种特殊情况
    #       - 如果两个参数都是整数，则触发指数 >= 0 的保护
    #           - 如果指数是非负数，则输出是整数
    #           - 否则输出是 float
    #   - 否则将另一个参数提升为 float
    #       - 注意：处理复数的情况几乎不可能正确地处理，如果基数为负数且指数是整数 float，则语义将分歧，并且总是返回复数。
    #         哼哼，假装这个问题不存在
    #   - 相等性是痛苦的：Python 进行了一些花哨的事情，它从浮点数中提取尾数，然后与整数比较。
    #     这意味着它能够判断出 9007199254740993 != 9007199254740992。（而不是如果左侧提升为 float，则它将截断到右侧然后随后相等）。
    #     我们将通过特殊的混合类型相等操作来精确模拟这一点。不幸的是，我们需要为所有比较操作实现这一点（也许我只会实现比较）
    #   - sym_ite 啥啥真的不应该允许混合，但无论如何

    # 如果方法在布尔型变成整数的魔术方法列表中
    if method in bool_becomes_int_magic_methods:

        # 定义一个函数用于提升类型
        # 实现 True + True = 2，在 Python 中有效但在 sympy 中不适用
        def promote(x):
            if isinstance(x, SymBool):
                return SymInt(x.node.wrap_int(int(x)))
            return x

    else:

        # 否则简单地返回参数本身
        def promote(x):
            return x
    # 定义一个实例方法 `promote2`，接受另一个对象 `other` 作为参数
    def promote2(self, other):
        # TODO: 从这个列表中移除 eq 和其他关系。
        # CPython 对这些关系有复杂的实现，以获取尽可能多的精度，而不仅仅是升级到 float64 然后祈祷，
        # 因此我们也需要特别处理它们。
        # 另外，注意 int_truediv 不会经过这个路径：两个参数都是 "int"，因此没有任何升级。
        
        # 如果方法不在以下列表中，则直接返回 self 和 other
        if method not in [
            "add",
            "sub",
            "mul",
            "mod",
            "float_pow",
            "float_truediv",
            "int_floordiv",
            "sym_min",
            "sym_max",
            # TODO: 移除以下内容
            "eq",
            "ne",
            "gt",
            "lt",
            "le",
            "ge",
        ]:
            return self, other
        
        # 检查 self 和 other 是否为 float 或 torch.SymFloat 类型
        f_self = isinstance(self, (float, torch.SymFloat))
        f_other = isinstance(other, (float, torch.SymFloat))
        
        # 如果 self 或 other 至少有一个不是 float 类型，则将其转换为 torch.SymFloat 类型
        if f_self or f_other:
            if not f_self:
                self = torch.sym_float(self)
            if not f_other:
                other = torch.sym_float(other)
        
        # 返回处理后的 self 和 other
        return self, other

    # 实现一元操作的方法 `unary_magic_impl`
    def unary_magic_impl(self):
        # 提升 self 的类型（根据上下文推测具体实现）
        self = promote(self)
        
        # 如果 self 是常量，则通过调用操作符来获取常量值
        if is_constant(self):
            return (method_to_operator(method))(get_constant(self))
        
        # 否则，调用相关节点的方法来执行操作，并将结果封装为节点
        return wrap_node(getattr(self.node, method_attr)())

    # 实现二元操作的方法 `binary_magic_impl`
    def binary_magic_impl(self, other):
        # 如果 other 不是 int、float、bool、SymInt、SymFloat 或 SymBool 类型，则返回 NotImplemented
        if not isinstance(other, (int, float, bool, SymInt, SymFloat, SymBool)):
            return NotImplemented
        
        # 记录日志
        sym_node_log.debug("MAGIC %s %s %s", method, self, other)
        
        # 提升 self 和 other 的类型
        self = promote(self)
        other = promote(other)
        self, other = promote2(self, other)
        
        # 如果 self 是常量，则通过调用操作符将常量和 other 进行操作
        if is_constant(self):
            return (method_to_operator(method))(get_constant(self), other)
        
        # 如果 other 是常量，则将其提取为常量值
        if is_constant(other):
            other = get_constant(other)
        
        # 将 self 转换为节点，再与 other 执行操作
        other_node = to_node(self.node, other)
        
        # 如果转换失败，则返回 NotImplemented
        if other_node is NotImplemented:
            return NotImplemented
        
        # 调用节点的方法执行操作，并将结果封装为节点
        ret = wrap_node(getattr(self.node, method_attr)(other_node))
        
        # 如果结果是常量，则返回其常量值，否则返回结果节点
        return get_constant(ret) if is_constant(ret) else ret
    def rbinary_magic_impl(self, other):
        # 检查参数 `other` 是否为 int、float、bool 或符号类型的实例
        if not isinstance(other, (int, float, bool, SymInt, SymFloat, SymBool)):
            return NotImplemented
        # 将自身和参数都提升为相应类型
        self = promote(self)
        other = promote(other)
        self, other = promote2(self, other)
        # 如果 self 是常量，则直接调用相应方法并返回结果
        if is_constant(self):
            return (method_to_operator(method))(get_constant(self), other)
        # 如果 other 是常量，则将其提取为普通值
        if is_constant(other):
            other = get_constant(other)
        # 将 other 转换为节点，并尝试调用方法，包装结果为节点
        other_node = to_node(self.node, other)
        # 如果操作未实现，则返回 NotImplemented
        if other_node is NotImplemented:
            return NotImplemented
        # 包装节点，并返回常量值或节点对象
        ret = wrap_node(getattr(other_node, method_attr)(self.node))
        return get_constant(ret) if is_constant(ret) else ret

    # 如果方法在一元魔术方法列表中
    if method in unary_magic_methods:
        # 将 unary_magic_impl 方法设置为 user_type 的魔术方法
        setattr(user_type, f"__{method}__", unary_magic_impl)
    # 如果方法在非魔术一元方法列表中
    elif method in unary_nonmagic_methods:
        # 获取原始方法并更新为使用 unary_magic_impl 方法
        orig = getattr(user_type, method)
        setattr(user_type, method, update_wrapper(unary_magic_impl, orig))
    # 如果方法是 "sym_ite"
    elif method == "sym_ite":

        def sym_ite_magic_impl(pred, then_val, else_val):
            # 获取预测、then 值、else 值的节点
            pred_node = pred.node
            then_node = to_node(pred_node, then_val)
            else_node = to_node(pred_node, else_val)
            # 如果 then_node 或 else_node 无法转换为节点，则返回 NotImplemented
            if then_node is NotImplemented or else_node is NotImplemented:
                return NotImplemented
            # 确保 then_node 和 else_node 是符号节点，且类型相同
            assert (
                isinstance(then_node, SymNode)
                and isinstance(else_node, SymNode)
                and then_node.pytype == else_node.pytype
            )
            # 包装节点，并返回常量值或节点对象
            ret = wrap_node(getattr(pred.node, method_attr)(then_node, else_node))
            return get_constant(ret) if ret.node.is_constant() else ret

        # 将 sym_ite_magic_impl 方法设置为 user_type 的魔术方法
        setattr(user_type, f"__{method}__", sym_ite_magic_impl)
    # 如果方法是 "round"
    elif method == "round":

        def round_magic_impl(self, ndigits=None):
            # 如果 self 是常量，则调用内置 round 函数并返回结果
            if is_constant(self):
                return builtins.round(get_constant(self), ndigits)
            # 否则，调用节点的 round 方法，并包装结果为节点
            return wrap_node(getattr(self.node, method)(ndigits))

        # 将 round_magic_impl 方法设置为 user_type 的魔术方法
        setattr(user_type, f"__{method}__", round_magic_impl)
    else:
        # 将 binary_magic_impl 方法设置为 user_type 的魔术方法
        setattr(user_type, f"__{method}__", binary_magic_impl)
        # 如果方法在可反射魔术方法列表中，设置反向操作方法
        if method in reflectable_magic_methods:
            setattr(user_type, f"__r{method}__", rbinary_magic_impl)
# 遍历 magic_methods 字典，其中 key 为方法名，value 为对应的函数
for method, func in magic_methods.items():  # type: ignore[assignment]
    # 如果方法名在 only_bool_magic_methods 中，则创建 SymBool 类型的用户自定义魔术方法
    if method in only_bool_magic_methods:
        _make_user_magic(method, SymBool)
        # 继续下一次循环
        continue
    # 如果方法名在 only_float_magic_methods 中，则创建 SymFloat 类型的用户自定义魔术方法
    if method in only_float_magic_methods:
        _make_user_magic(method, SymFloat)
        # 继续下一次循环
        continue
    # 如果方法名在 also_bool_magic_methods 或 bool_becomes_int_magic_methods 中，则创建 SymBool 类型的用户自定义魔术方法
    if method in also_bool_magic_methods or method in bool_becomes_int_magic_methods:
        _make_user_magic(method, SymBool)
    # 创建 SymInt 类型的用户自定义魔术方法
    _make_user_magic(method, SymInt)
    # 创建 SymFloat 类型的用户自定义魔术方法

# 删除 method 变量
del method
# 删除 func 变量
del func
```