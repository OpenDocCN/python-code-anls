# `D:\src\scipysrc\pandas\pandas\core\computation\expressions.py`

```
"""
Expressions
-----------

Offer fast expression evaluation through numexpr

"""

# 引入将来的注释语法支持
from __future__ import annotations

# 引入操作符模块
import operator

# 引入类型检查相关模块
from typing import TYPE_CHECKING

# 引发警告的模块
import warnings

# 引入 NumPy 库
import numpy as np

# 引入 Pandas 配置管理模块
from pandas._config import get_option

# 引入异常处理相关模块
from pandas.util._exceptions import find_stack_level

# 引入 Pandas 核心运算模块
from pandas.core import roperator

# 引入 Pandas 计算检查模块
from pandas.core.computation.check import NUMEXPR_INSTALLED

# 如果 Numexpr 安装成功，则引入 numexpr 库
if NUMEXPR_INSTALLED:
    import numexpr as ne

# 如果正在进行类型检查，则引入 FuncType
if TYPE_CHECKING:
    from pandas._typing import FuncType

# 测试模式标志，默认为 None
_TEST_MODE: bool | None = None

# 测试结果列表，默认为空
_TEST_RESULT: list[bool] = []

# 是否使用 Numexpr，默认根据 Numexpr 是否安装决定
USE_NUMEXPR = NUMEXPR_INSTALLED

# 用于评估表达式的函数类型，初始为 None
_evaluate: FuncType | None = None

# 用于条件运算的函数类型，初始为 None
_where: FuncType | None = None

# 允许传递给 numexpr 的数据类型集合
_ALLOWED_DTYPES = {
    "evaluate": {"int64", "int32", "float64", "float32", "bool"},
    "where": {"int64", "float64", "bool"},
}

# 使用 numexpr 的最小元素数量阈值
_MIN_ELEMENTS = 1_000_000


def set_use_numexpr(v: bool = True) -> None:
    # 设置是否使用 numexpr
    global USE_NUMEXPR
    if NUMEXPR_INSTALLED:
        USE_NUMEXPR = v

    # 根据 USE_NUMEXPR 的值选择适当的评估和条件运算函数
    global _evaluate, _where
    _evaluate = _evaluate_numexpr if USE_NUMEXPR else _evaluate_standard
    _where = _where_numexpr if USE_NUMEXPR else _where_standard


def set_numexpr_threads(n=None) -> None:
    # 如果使用 numexpr，则设置线程数为 n
    # 否则重置线程设置
    if NUMEXPR_INSTALLED and USE_NUMEXPR:
        if n is None:
            n = ne.detect_number_of_cores()
        ne.set_num_threads(n)


def _evaluate_standard(op, op_str, a, b):
    """
    标准的表达式评估函数。
    """
    if _TEST_MODE:
        _store_test_result(False)
    return op(a, b)


def _can_use_numexpr(op, op_str, a, b, dtype_check) -> bool:
    """
    判断是否应该使用 numexpr 进行计算。
    返回一个布尔值。
    """
    if op_str is not None:
        # 检查是否超过最小元素数量阈值
        if a.size > _MIN_ELEMENTS:
            # 检查数据类型兼容性
            dtypes: set[str] = set()
            for o in [a, b]:
                if hasattr(o, "dtype"):
                    dtypes |= {o.dtype.name}

            # 检查数据类型是否在允许的集合中
            if not len(dtypes) or _ALLOWED_DTYPES[dtype_check] >= dtypes:
                return True

    return False


def _evaluate_numexpr(op, op_str, a, b):
    # 使用 numexpr 进行表达式评估的函数，尚未实现内容
    result = None
    # 如果可以使用 numexpr 库来评估操作(op)，
    # 则检查是否为反向调用的操作方法
    if _can_use_numexpr(op, op_str, a, b, "evaluate"):
        is_reversed = op.__name__.strip("_").startswith("r")
        # 如果是反向调用，则交换操作数a和b的值
        if is_reversed:
            # 我们最初是由反向操作方法调用的
            a, b = b, a

        a_value = a
        b_value = b

        try:
            # 使用 numexpr 库来计算表达式"a_value op_str b_value"
            result = ne.evaluate(
                f"a_value {op_str} b_value",
                local_dict={"a_value": a_value, "b_value": b_value},
                casting="safe",
            )
        except TypeError:
            # 处理 numexpr 报出的 TypeError 异常，
            # 例如对整数数组 ** 整数数组的情况
            # (https://github.com/pydata/numexpr/issues/379)
            pass
        except NotImplementedError:
            # 如果 numexpr 抛出 NotImplementedError 异常，
            # 并且不能通过 bool 类型的算术回退处理，则重新抛出异常
            if _bool_arith_fallback(op_str, a, b):
                pass
            else:
                raise

        if is_reversed:
            # 恢复操作数a和b的原始顺序，用于回退操作
            a, b = b, a

    # 如果处于测试模式(_TEST_MODE)，记录测试结果是否为非空
    if _TEST_MODE:
        _store_test_result(result is not None)

    # 如果结果为None，则使用标准的求值函数来计算结果
    if result is None:
        result = _evaluate_standard(op, op_str, a, b)

    # 返回计算得到的结果
    return result
# 定义操作符到字符串表示的映射字典
_op_str_mapping = {
    operator.add: "+",          # 加法操作符
    roperator.radd: "+",        # 反向加法操作符
    operator.mul: "*",          # 乘法操作符
    roperator.rmul: "*",        # 反向乘法操作符
    operator.sub: "-",          # 减法操作符
    roperator.rsub: "-",        # 反向减法操作符
    operator.truediv: "/",      # 真除法操作符
    roperator.rtruediv: "/",    # 反向真除法操作符
    # floordiv 在 numexpr 2.x 中不受支持
    operator.floordiv: None,    # 地板除法操作符（在 numexpr 中不支持）
    roperator.rfloordiv: None,  # 反向地板除法操作符（在 numexpr 中不支持）
    # 为了向后兼容性，我们需要负数求模运算的 Python 语义
    # 参见 https://github.com/pydata/numexpr/issues/365
    # 因此暂时保持未加速的状态 GH#36552
    operator.mod: None,         # 模运算操作符（在 numexpr 中不支持）
    roperator.rmod: None,       # 反向模运算操作符（在 numexpr 中不支持）
    operator.pow: "**",         # 幂运算操作符
    roperator.rpow: "**",       # 反向幂运算操作符
    operator.eq: "==",          # 等于比较操作符
    operator.ne: "!=",          # 不等于比较操作符
    operator.le: "<=",          # 小于等于比较操作符
    operator.lt: "<",           # 小于比较操作符
    operator.ge: ">=",          # 大于等于比较操作符
    operator.gt: ">",           # 大于比较操作符
    operator.and_: "&",         # 位与操作符
    roperator.rand_: "&",       # 反向位与操作符
    operator.or_: "|",          # 位或操作符
    roperator.ror_: "|",        # 反向位或操作符
    operator.xor: "^",          # 位异或操作符
    roperator.rxor: "^",        # 反向位异或操作符
    divmod: None,               # divmod 函数（在 numexpr 中不支持）
    roperator.rdivmod: None,    # 反向 divmod 函数（在 numexpr 中不支持）
}


def _where_standard(cond, a, b):
    # 调用者需负责在必要时提取 ndarray
    return np.where(cond, a, b)


def _where_numexpr(cond, a, b):
    # 调用者需负责在必要时提取 ndarray
    result = None

    if _can_use_numexpr(None, "where", a, b, "where"):
        result = ne.evaluate(
            "where(cond_value, a_value, b_value)",
            local_dict={"cond_value": cond, "a_value": a, "b_value": b},
            casting="safe",
        )

    if result is None:
        result = _where_standard(cond, a, b)

    return result


# 启用 numexpr
set_use_numexpr(get_option("compute.use_numexpr"))


def _has_bool_dtype(x):
    try:
        return x.dtype == bool
    except AttributeError:
        return isinstance(x, (bool, np.bool_))


# 不支持的布尔运算操作映射
_BOOL_OP_UNSUPPORTED = {"+": "|", "*": "&", "-": "^"}


def _bool_arith_fallback(op_str, a, b) -> bool:
    """
    检查在 numexpr 不支持的布尔运算操作时是否应回退到 Python 的 `_evaluate_standard` 函数。
    """
    if _has_bool_dtype(a) and _has_bool_dtype(b):
        if op_str in _BOOL_OP_UNSUPPORTED:
            warnings.warn(
                f"evaluating in Python space because the {op_str!r} "
                "operator is not supported by numexpr for the bool dtype, "
                f"use {_BOOL_OP_UNSUPPORTED[op_str]!r} instead.",
                stacklevel=find_stack_level(),
            )
            return True
    return False


def evaluate(op, a, b, use_numexpr: bool = True):
    """
    评估并返回对 a 和 b 进行的操作 op 的表达式结果。

    参数
    ----------
    op : 实际操作数
    a : 左操作数
    b : 右操作数
    use_numexpr : bool, 默认为 True
        是否尝试使用 numexpr。
    """
    op_str = _op_str_mapping[op]
    if op_str is not None:
        if use_numexpr:
            # 错误："None" 不可调用
            return _evaluate(op, op_str, a, b)  # type: ignore[misc]
    # 调用名为 _evaluate_standard 的函数，并返回其计算结果
    return _evaluate_standard(op, op_str, a, b)
def where(cond, a, b, use_numexpr: bool = True):
    """
    Evaluate the where condition cond on a and b.

    Parameters
    ----------
    cond : np.ndarray[bool]
        Boolean array representing the condition to be evaluated.
    a : return if cond is True
        Value returned when the condition is True.
    b : return if cond is False
        Value returned when the condition is False.
    use_numexpr : bool, default True
        Whether to try to use numexpr for evaluation.
    """
    assert _where is not None
    # Use numexpr or standard where function based on the flag
    return _where(cond, a, b) if use_numexpr else _where_standard(cond, a, b)


def set_test_mode(v: bool = True) -> None:
    """
    Keeps track of whether numexpr was used.

    Stores an additional ``True`` for every successful use of evaluate with
    numexpr since the last ``get_test_result``.

    Parameters
    ----------
    v : bool, default True
        Boolean flag indicating whether to set test mode.
    """
    global _TEST_MODE, _TEST_RESULT
    # Set test mode to the provided value
    _TEST_MODE = v
    # Reset the test result list
    _TEST_RESULT = []


def _store_test_result(used_numexpr: bool) -> None:
    """
    Store test result based on whether numexpr was used.

    Parameters
    ----------
    used_numexpr : bool
        Boolean indicating whether numexpr was used in the test.
    """
    if used_numexpr:
        # Append True to the test result list if numexpr was used
        _TEST_RESULT.append(used_numexpr)


def get_test_result() -> list[bool]:
    """
    Get test result and reset test_results.

    Returns
    -------
    list[bool]
        List of boolean values indicating whether numexpr was used in tests
        since the last call to this function.
    """
    global _TEST_RESULT
    # Capture the current test result list
    res = _TEST_RESULT
    # Reset the test result list
    _TEST_RESULT = []
    return res
```