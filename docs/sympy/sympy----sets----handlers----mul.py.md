# `D:\src\scipysrc\sympy\sympy\sets\handlers\mul.py`

```
from sympy.core import Basic, Expr  # 导入基础符号和表达式类
from sympy.core.numbers import oo  # 导入无穷大符号
from sympy.core.symbol import symbols  # 导入符号定义函数
from sympy.multipledispatch import Dispatcher  # 导入多重分发函数
from sympy.sets.setexpr import set_mul  # 导入集合乘法函数
from sympy.sets.sets import Interval, Set  # 导入区间和集合类


_x, _y = symbols("x y")  # 定义符号变量 _x 和 _y

_set_mul = Dispatcher('_set_mul')  # 创建名为 _set_mul 的多重分发对象
_set_div = Dispatcher('_set_div')  # 创建名为 _set_div 的多重分发对象

# 注册多重分发函数，实现不同参数类型的乘法操作
@_set_mul.register(Basic, Basic)
def _(x, y):
    return None  # 如果参数类型为 Basic, Basic，则返回 None

@_set_mul.register(Set, Set)
def _(x, y):
    return None  # 如果参数类型为 Set, Set，则返回 None

@_set_mul.register(Expr, Expr)
def _(x, y):
    return x*y  # 如果参数类型为 Expr, Expr，则返回它们的乘积

@_set_mul.register(Interval, Interval)
def _(x, y):
    """
    区间算术乘法
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    # TODO: some intervals containing 0 and oo will fail as 0*oo returns nan.
    # 计算四种组合情况下的乘积和开闭状态
    comvals = (
        (x.start * y.start, bool(x.left_open or y.left_open)),
        (x.start * y.end, bool(x.left_open or y.right_open)),
        (x.end * y.start, bool(x.right_open or y.left_open)),
        (x.end * y.end, bool(x.right_open or y.right_open)),
    )
    # TODO: handle symbolic intervals
    # 取得最小值和最大值的乘积以及对应的开闭状态
    minval, minopen = min(comvals)
    maxval, maxopen = max(comvals)
    return Interval(
        minval,
        maxval,
        minopen,
        maxopen
    )

# 注册多重分发函数，实现不同参数类型的除法操作
@_set_div.register(Basic, Basic)
def _(x, y):
    return None  # 如果参数类型为 Basic, Basic，则返回 None

@_set_div.register(Expr, Expr)
def _(x, y):
    return x/y  # 如果参数类型为 Expr, Expr，则返回它们的商

@_set_div.register(Set, Set)
def _(x, y):
    return None  # 如果参数类型为 Set, Set，则返回 None

@_set_div.register(Interval, Interval)
def _(x, y):
    """
    区间算术除法
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    # 如果乘积的区间含有负数，则返回 (-oo, oo)
    if (y.start*y.end).is_negative:
        return Interval(-oo, oo)
    # 计算区间 y 的倒数
    if y.start == 0:
        s2 = oo
    else:
        s2 = 1/y.start
    if y.end == 0:
        s1 = -oo
    else:
        s1 = 1/y.end
    # 返回 x 和 y 区间的乘积
    return set_mul(x, Interval(s1, s2, y.right_open, y.left_open))
```