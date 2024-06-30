# `D:\src\scipysrc\sympy\sympy\assumptions\handlers\common.py`

```
"""
This module defines base class for handlers and some core handlers:
``Q.commutative`` and ``Q.is_true``.
"""

from sympy.assumptions import Q, ask, AppliedPredicate  # 导入符号计算模块的假设相关内容
from sympy.core import Basic, Symbol  # 导入符号计算模块的核心基础类和符号类
from sympy.core.logic import _fuzzy_group  # 导入符号计算模块的逻辑操作相关内容
from sympy.core.numbers import NaN, Number  # 导入符号计算模块的NaN和数字类
from sympy.logic.boolalg import (And, BooleanTrue, BooleanFalse, conjuncts,  # 导入符号计算模块的布尔代数相关内容
    Equivalent, Implies, Not, Or)
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入符号计算模块的异常处理相关内容

from ..predicates.common import CommutativePredicate, IsTruePredicate  # 导入自定义模块中的可交换性和真值断言谓词

class AskHandler:
    """Base class that all Ask Handlers must inherit."""
    def __new__(cls, *args, **kwargs):
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. The AskHandler class should
            be replaced with the multipledispatch handler of Predicate
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
        )
        return super().__new__(cls, *args, **kwargs)  # 调用父类的构造方法

class CommonHandler(AskHandler):
    # Deprecated
    """Defines some useful methods common to most Handlers. """

    @staticmethod
    def AlwaysTrue(expr, assumptions):
        return True  # 返回常量True，表示总是成立

    @staticmethod
    def AlwaysFalse(expr, assumptions):
        return False  # 返回常量False，表示总是不成立

    @staticmethod
    def AlwaysNone(expr, assumptions):
        return None  # 返回None，表示结果未定

    NaN = AlwaysFalse  # NaN属性指向AlwaysFalse方法，表示NaN值总是不成立

# CommutativePredicate

@CommutativePredicate.register(Symbol)
def _(expr, assumptions):
    """Objects are expected to be commutative unless otherwise stated"""
    assumps = conjuncts(assumptions)  # 将假设条件转换为子句列表
    if expr.is_commutative is not None:
        return expr.is_commutative and not ~Q.commutative(expr) in assumps  # 检查表达式是否声明为可交换，并检查相应的假设
    if Q.commutative(expr) in assumps:
        return True  # 如果假设表明表达式是可交换的，则返回True
    elif ~Q.commutative(expr) in assumps:
        return False  # 如果假设表明表达式不可交换，则返回False
    return True  # 默认情况下假设对象为可交换

@CommutativePredicate.register(Basic)
def _(expr, assumptions):
    for arg in expr.args:
        if not ask(Q.commutative(arg), assumptions):
            return False  # 对表达式中的每个参数检查其是否可交换，如果有一个不可交换，则返回False
    return True  # 如果所有参数都可交换，则返回True

@CommutativePredicate.register(Number)
def _(expr, assumptions):
    return True  # 数字类型对象默认为可交换

@CommutativePredicate.register(NaN)
def _(expr, assumptions):
    return True  # NaN类型对象默认为可交换


# IsTruePredicate

@IsTruePredicate.register(bool)
def _(expr, assumptions):
    return expr  # 直接返回布尔值本身

@IsTruePredicate.register(BooleanTrue)
def _(expr, assumptions):
    return True  # 返回True，表示为真

@IsTruePredicate.register(BooleanFalse)
def _(expr, assumptions):
    return False  # 返回False，表示为假

@IsTruePredicate.register(AppliedPredicate)
def _(expr, assumptions):
    return ask(expr, assumptions)  # 对应用谓词进行询问，返回其真假情况

@IsTruePredicate.register(Not)
def _(expr, assumptions):
    arg = expr.args[0]
    if arg.is_Symbol:
        # symbol used as abstract boolean object
        return None  # 如果参数是符号，则返回None，表示结果未定
    value = ask(arg, assumptions=assumptions)
    if value in (True, False):
        return not value  # 如果参数是布尔表达式，则返回其取反结果
    else:
        return None  # 如果无法确定参数的真假，则返回None

@IsTruePredicate.register(Or)
def _(expr, assumptions):
    result = False  # 初始化结果为False
    # 对于表达式中的每个参数进行遍历
    for arg in expr.args:
        # 调用 ask 函数询问参数 arg，并传入假设列表 assumptions
        p = ask(arg, assumptions=assumptions)
        # 如果 ask 返回 True，则直接返回 True
        if p is True:
            return True
        # 如果 ask 返回 None，则将 result 设为 None
        if p is None:
            result = None
    # 循环结束后，返回最终的 result 值（可能为 None 或者初始值）
    return result
# 注册一个针对 And 表达式的 IsTruePredicate 的函数
@IsTruePredicate.register(And)
def _(expr, assumptions):
    # 初始化结果为 True
    result = True
    # 遍历 And 表达式的每个参数
    for arg in expr.args:
        # 调用 ask 函数询问每个参数的真值，基于给定的假设
        p = ask(arg, assumptions=assumptions)
        # 如果询问结果为 False，直接返回 False
        if p is False:
            return False
        # 如果询问结果为 None，则结果更新为 None
        if p is None:
            result = None
    # 返回最终结果，可能为 True 或者 None
    return result

# 注册一个针对 Implies 表达式的 IsTruePredicate 的函数
@IsTruePredicate.register(Implies)
def _(expr, assumptions):
    # 将 Implies 表达式分解为 p 和 q
    p, q = expr.args
    # 调用 ask 函数询问 ~p | q 的真值，基于给定的假设
    return ask(~p | q, assumptions=assumptions)

# 注册一个针对 Equivalent 表达式的 IsTruePredicate 的函数
@IsTruePredicate.register(Equivalent)
def _(expr, assumptions):
    # 将 Equivalent 表达式分解为 p 和 q
    p, q = expr.args
    # 询问 p 的真值，基于给定的假设
    pt = ask(p, assumptions=assumptions)
    # 如果 p 的真值为 None，则结果返回 None
    if pt is None:
        return None
    # 询问 q 的真值，基于给定的假设
    qt = ask(q, assumptions=assumptions)
    # 如果 q 的真值为 None，则结果返回 None
    if qt is None:
        return None
    # 返回 p 和 q 的真值比较结果
    return pt == qt


#### Helper methods

# 测试闭合组成员资格是否符合当前操作的方法
def test_closed_group(expr, assumptions, key):
    """
    Test for membership in a group with respect
    to the current operation.
    """
    # 使用 _fuzzy_group 函数对表达式的参数进行成员资格测试，
    # 使用快速退出选项进行处理
    return _fuzzy_group(
        (ask(key(a), assumptions) for a in expr.args), quick_exit=True)
```