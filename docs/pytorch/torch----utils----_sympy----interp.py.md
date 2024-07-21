# `.\pytorch\torch\utils\_sympy\interp.py`

```
"""
这是一个简单的Sympy表达式解释器，根据torch._inductor.virtualized的调用约定分派给不同的类处理。
为了直接性，解释器直接使用处理程序，而不是查询TLS（线程本地存储）。
它不使用完整处理程序上的大多数方法，仅使用与Sympy表达式对应的方法。
要查看完整处理程序的示例，请参阅torch.utils._sympy.value_ranges.ValueRangeAnalysis。
"""

import functools  # 导入functools模块，用于函数装饰器
import logging    # 导入logging模块，用于日志记录
from typing import Any, Dict, Union  # 导入类型提示相关模块

import sympy     # 导入sympy库，用于符号计算
from sympy.logic.boolalg import Boolean as SympyBoolean, BooleanAtom  # 导入Sympy中的布尔代数相关类

import torch     # 导入torch库，用于深度学习框架
from .functions import (  # 从当前目录下的functions模块中导入多个函数
    CeilToInt,
    CleanDiv,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    FloorToInt,
    Identity,
    IntTrueDiv,
    IsNonOverlappingAndDenseIndicator,
    Mod,
    ModularIndexing,
    PowByNatural,
    PythonMod,
    RoundDecimal,
    RoundToInt,
    ToFloat,
    TruncToFloat,
    TruncToInt,
    Where,
)

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@functools.lru_cache(None)
def handlers():
    """
    返回处理程序的缓存版本，用于处理Sympy表达式。
    """
    # TODO add CeilDiv (it doesn't appear in the index_expr)
    # 添加CeilDiv（它在index_expr中没有出现）

    # TODO default to some decompositions if the interpreter doesn't have them
    # like decomposing ModularIndexing or implementing Le(a,b) as Ge(b, a)
    # 如果解释器没有它们，则默认为一些分解，比如分解ModularIndexing或将Le(a,b)实现为Ge(b, a)
    # 定义一个字典 HANDLERS，用于将 sympy 中的数学函数映射到相应的字符串处理函数
    HANDLERS = {
        sympy.Or: "or_",  # 将 sympy.Or 映射到字符串 "or_"
        sympy.And: "and_",  # 将 sympy.And 映射到字符串 "and_"
        sympy.Eq: "eq",  # 将 sympy.Eq 映射到字符串 "eq"
        sympy.Ne: "ne",  # 将 sympy.Ne 映射到字符串 "ne"
        sympy.Lt: "lt",  # 将 sympy.Lt 映射到字符串 "lt"
        sympy.Gt: "gt",  # 将 sympy.Gt 映射到字符串 "gt"
        sympy.Le: "le",  # 将 sympy.Le 映射到字符串 "le"
        sympy.Ge: "ge",  # 将 sympy.Ge 映射到字符串 "ge"
        sympy.Not: "not_",  # 将 sympy.Not 映射到字符串 "not_"
        IntTrueDiv: "int_truediv",  # 将 IntTrueDiv 映射到字符串 "int_truediv"
        FloatTrueDiv: "truediv",  # 将 FloatTrueDiv 映射到字符串 "truediv"
        FloorDiv: "floordiv",  # 将 FloorDiv 映射到字符串 "floordiv"
        CleanDiv: "floordiv",  # 将 CleanDiv 映射到字符串 "floordiv"，注释中有待解决的问题
        TruncToFloat: "trunc",  # 将 TruncToFloat 映射到字符串 "trunc"
        Where: "where",  # 将 Where 映射到字符串 "where"
        sympy.Add: "add",  # 将 sympy.Add 映射到字符串 "add"
        sympy.Mul: "mul",  # 将 sympy.Mul 映射到字符串 "mul"
        FloatPow: "pow",  # 将 FloatPow 映射到字符串 "pow"
        PowByNatural: "pow_by_natural",  # 将 PowByNatural 映射到字符串 "pow_by_natural"
        # 对 sympy.Pow 的特别处理注释，因为 sympy 会将 x * x 简化为 Pow(x, 2)，但我们只希望处理整数
        sympy.Pow: "pow_by_natural",  # 将 sympy.Pow 映射到字符串 "pow_by_natural"
        Mod: "mod",  # 将 Mod 映射到字符串 "mod"
        PythonMod: "mod",  # 将 PythonMod 映射到字符串 "mod"，注释中有待解决的问题
        # 对 sympy.Mod 的处理注释，表示需要进一步澄清语义
        sympy.Mod: "mod",  # 将 sympy.Mod 映射到字符串 "mod"
        sympy.Abs: "abs",  # 将 sympy.Abs 映射到字符串 "abs"
        sympy.log: "log",  # 将 sympy.log 映射到字符串 "log"
        sympy.exp: "exp",  # 将 sympy.exp 映射到字符串 "exp"
        sympy.Min: "minimum",  # 将 sympy.Min 映射到字符串 "minimum"
        sympy.Max: "maximum",  # 将 sympy.Max 映射到字符串 "maximum"
        ModularIndexing: "modular_indexing",  # 将 ModularIndexing 映射到字符串 "modular_indexing"
        sympy.functions.elementary.piecewise.ExprCondPair: "expr_cond_pair",  # 将 sympy.functions.elementary.piecewise.ExprCondPair 映射到字符串 "expr_cond_pair"
        sympy.Piecewise: "piecewise",  # 将 sympy.Piecewise 映射到字符串 "piecewise"
        Identity: "identity",  # 将 Identity 映射到字符串 "identity"
        IsNonOverlappingAndDenseIndicator: "is_non_overlapping_and_dense_indicator",  # 将 IsNonOverlappingAndDenseIndicator 映射到字符串 "is_non_overlapping_and_dense_indicator"
        RoundDecimal: "round_decimal",  # 将 RoundDecimal 映射到字符串 "round_decimal"
    }
    
    # 对一组数学函数名进行循环，将每个函数名映射到相应的字符串处理函数名
    for name in ["cos", "sin", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan"]:
        HANDLERS[getattr(sympy, name)] = name
    
    # 返回最终的 HANDLERS 字典，其中包含了所有的映射关系
    return HANDLERS
# 定义了一个集合，包含了可以进行关联操作的字符串名称
ASSOCIATIVE_OPS = {"minimum", "maximum", "mul", "add", "and_", "or_"}

# 使用 sympy 库进行表达式解析和转换的函数
def sympy_interp(
    analysis,
    env: Dict[sympy.Symbol, Any],
    expr: Union[sympy.Expr, SympyBoolean],
    *,
    index_dtype=torch.int64,
):
    # 处理基本情况，根据表达式类型确定对应的 Torch 数据类型
    dtype = None
    if isinstance(expr, BooleanAtom):
        dtype = torch.bool
    elif isinstance(expr, sympy.Integer):
        dtype = torch.int64
    elif isinstance(expr, sympy.Number):
        dtype = torch.double

    if dtype is not None:
        # 如果表达式是常量，调用分析对象的 constant 方法进行处理
        return analysis.constant(expr, dtype)
    elif isinstance(expr, sympy.Symbol):
        # 如果表达式是符号，返回在环境字典中该符号对应的值
        return env[expr]

    # 处理特殊情况
    if isinstance(expr, sympy.Pow) and isinstance(
        expr.args[1], sympy.core.numbers.Half
    ):
        # 如果是幂操作且指数为 0.5，调用 sqrt 方法处理
        return analysis.sqrt(sympy_interp(analysis, env, expr.args[0]))
    if isinstance(expr, ToFloat):
        # 如果是 ToFloat 类型，调用 to_dtype 方法转换为指定的 Torch 浮点数类型
        return analysis.to_dtype(
            sympy_interp(analysis, env, expr.args[0]), torch.float64
        )

    # 递归情况，处理表达式的参数列表
    args = [sympy_interp(analysis, env, arg) for arg in expr.args]  # type: ignore[arg-type]

    # 这些处理器特殊在于它们接受额外的 dtype 参数指定转换后的类型，
    # 当从 Sympy 转换时需要适当设置这些参数，当你发现可以缩小索引范围时，
    # 可以保守地选择 int64，稍后再缩小这些参数，但如果已知 32 位索引是 OK 的，
    # 可以直接使用 index_dtype=torch.int32 进行 sympy 转换
    INDEX_DTYPE_HANDLERS = {
        TruncToInt: "trunc_to_int",
        sympy.floor: "floor_to_int",
        sympy.ceiling: "ceil_to_int",
        FloorToInt: "floor_to_int",
        CeilToInt: "ceil_to_int",
        RoundToInt: "round_to_int",
    }
    if (handler_name := INDEX_DTYPE_HANDLERS.get(expr.func)) is not None:
        # 如果表达式的函数在索引处理器字典中，调用对应的处理方法进行处理
        return getattr(analysis, handler_name)(*args, index_dtype)

    if hasattr(expr.func, "_torch_handler_name"):
        handler_name = expr.func._torch_handler_name
    else:
        handler_name = handlers()[expr.func]
    handler = getattr(analysis, handler_name)
    try:
        if handler_name in ASSOCIATIVE_OPS:
            # 如果处理器名称在关联操作集合中，执行关联操作
            assert len(args) > 1
            acc = handler(args[0], args[1])
            for i in range(2, len(args)):
                acc = handler(acc, args[i])
            log.debug("%s(%s) -> %s", handler_name, args, acc)
            return acc
        else:
            # 否则执行普通的处理器操作
            r = handler(*args)
            log.debug("%s(%s) -> %s", handler_name, args, r)
            return r
    except Exception:
        # 捕获并记录执行过程中的异常
        log.warning("failed while executing %s(%s)", handler_name, args)
        raise
```