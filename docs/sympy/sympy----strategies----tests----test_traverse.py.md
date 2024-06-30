# `D:\src\scipysrc\sympy\sympy\strategies\tests\test_traverse.py`

```
from sympy.strategies.traverse import (
    top_down, bottom_up, sall, top_down_once, bottom_up_once, basic_fns)
# 从 sympy.strategies.traverse 导入所需的函数和策略

from sympy.strategies.rl import rebuild
# 导入 sympy.strategies.rl 模块中的 rebuild 函数

from sympy.strategies.util import expr_fns
# 导入 sympy.strategies.util 模块中的 expr_fns 函数

from sympy.core.add import Add
# 从 sympy.core.add 导入 Add 类

from sympy.core.basic import Basic
# 从 sympy.core.basic 导入 Basic 类

from sympy.core.numbers import Integer
# 从 sympy.core.numbers 导入 Integer 类

from sympy.core.singleton import S
# 从 sympy.core.singleton 导入 S 单例

from sympy.core.symbol import Str, Symbol
# 从 sympy.core.symbol 导入 Str 和 Symbol 类

from sympy.abc import x, y, z
# 导入 sympy.abc 中的 x, y, z 符号变量


def zero_symbols(expression):
    # 如果表达式是 Symbol 类型，则返回 S.Zero，否则返回原始表达式
    return S.Zero if isinstance(expression, Symbol) else expression


def test_sall():
    # 使用 sall 函数对 zero_symbols 函数进行一级遍历
    zero_onelevel = sall(zero_symbols)

    # 断言对 Basic(x, y, Basic(x, z)) 应用 zero_onelevel 后的结果
    assert zero_onelevel(Basic(x, y, Basic(x, z))) == \
        Basic(S(0), S(0), Basic(x, z))


def test_bottom_up():
    # 测试 bottom_up 函数的全局遍历特性
    _test_global_traversal(bottom_up)
    # 测试 bottom_up 函数在非基本对象上的停止特性


def test_top_down():
    # 测试 top_down 函数的全局遍历特性
    _test_global_traversal(top_down)
    # 测试 top_down 函数在非基本对象上的停止特性


def _test_global_traversal(trav):
    # 使用给定的遍历函数 trav 对 zero_symbols 函数进行全局遍历
    zero_all_symbols = trav(zero_symbols)

    # 断言对 Basic(x, y, Basic(x, z)) 应用 zero_all_symbols 后的结果
    assert zero_all_symbols(Basic(x, y, Basic(x, z))) == \
        Basic(S(0), S(0), Basic(S(0), S(0)))


def _test_stop_on_non_basics(trav):
    # 定义一个函数 add_one_if_can，尝试对表达式添加 1，如果类型错误则返回原始表达式
    def add_one_if_can(expr):
        try:
            return expr + 1
        except TypeError:
            return expr

    # 创建测试用例 expr 和期望结果 expected
    expr = Basic(S(1), Str('a'), Basic(S(2), Str('b')))
    expected = Basic(S(2), Str('a'), Basic(S(3), Str('b')))
    # 使用给定的遍历函数 trav 对 add_one_if_can 函数进行应用
    rl = trav(add_one_if_can)

    # 断言对 expr 应用 rl 后的结果与期望结果 expected 相同
    assert rl(expr) == expected


class Basic2(Basic):
    pass


def rl(x):
    # 如果 x 的参数存在且第一个参数不是 Integer 类型，则返回 Basic2 类型的新实例，否则返回原始 x
    if x.args and not isinstance(x.args[0], Integer):
        return Basic2(*x.args)
    return x


def test_top_down_once():
    # 使用 top_down_once 函数对 rl 函数进行应用
    top_rl = top_down_once(rl)

    # 断言对 Basic(S(1.0), S(2.0), Basic(S(3), S(4))) 应用 top_rl 后的结果
    assert top_rl(Basic(S(1.0), S(2.0), Basic(S(3), S(4)))) == \
        Basic2(S(1.0), S(2.0), Basic(S(3), S(4)))


def test_bottom_up_once():
    # 使用 bottom_up_once 函数对 rl 函数进行应用
    bottom_rl = bottom_up_once(rl)

    # 断言对 Basic(S(1), S(2), Basic(S(3.0), S(4.0))) 应用 bottom_rl 后的结果
    assert bottom_rl(Basic(S(1), S(2), Basic(S(3.0), S(4.0)))) == \
        Basic(S(1), S(2), Basic2(S(3.0), S(4.0)))


def test_expr_fns():
    # 创建表达式 expr = x + y**3
    expr = x + y**3
    # 使用 bottom_up 函数应用 lambda 函数，将表达式中的每个元素加 1，并使用 expr_fns 函数
    e = bottom_up(lambda v: v + 1, expr_fns)(expr)
    # 使用 bottom_up 函数应用 lambda 函数，将 Basic.__new__(Add, v, S(1)) 应用于表达式，并使用 basic_fns 函数
    b = bottom_up(lambda v: Basic.__new__(Add, v, S(1)), basic_fns)(expr)

    # 断言 rebuild 函数对 b 的结果与 e 相同
    assert rebuild(b) == e
```