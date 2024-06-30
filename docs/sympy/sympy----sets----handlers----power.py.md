# `D:\src\scipysrc\sympy\sympy\sets\handlers\power.py`

```
from sympy.core import Basic, Expr  # 导入基本的符号和表达式类
from sympy.core.function import Lambda  # 导入Lambda函数类
from sympy.core.numbers import oo, Infinity, NegativeInfinity, Zero, Integer  # 导入无穷大、正无穷大、负无穷大、零、整数类
from sympy.core.singleton import S  # 导入单例类S
from sympy.core.symbol import symbols  # 导入符号类
from sympy.functions.elementary.miscellaneous import (Max, Min)  # 导入最大值和最小值函数
from sympy.sets.fancysets import ImageSet  # 导入映射集类
from sympy.sets.setexpr import set_div  # 导入集合表达式类中的set_div函数
from sympy.sets.sets import Set, Interval, FiniteSet, Union  # 导入集合、区间、有限集、并集类
from sympy.multipledispatch import Dispatcher  # 导入多分派调度器类


_x, _y = symbols("x y")  # 定义符号_x和_y为符号类的实例，分别表示"x"和"y"


_set_pow = Dispatcher('_set_pow')  # 创建名为_set_pow的多分派调度器对象，用于处理指数运算


@_set_pow.register(Basic, Basic)
def _(x, y):
    return None  # 如果参数都是Basic类型，则返回None


@_set_pow.register(Set, Set)
def _(x, y):
    return ImageSet(Lambda((_x, _y), (_x ** _y)), x, y)
    # 如果参数都是Set类型，则返回一个映射集，其元素由Lambda函数生成，表示集合x中元素的y次幂


@_set_pow.register(Expr, Expr)
def _(x, y):
    return x**y  # 如果参数都是Expr类型，则返回x的y次幂的表达式


@_set_pow.register(Interval, Zero)
def _(x, z):
    return FiniteSet(S.One)
    # 如果第一个参数是Interval类型，第二个参数是Zero类型，则返回包含S.One的有限集


@_set_pow.register(Interval, Integer)
def _(x, exponent):
    """
    Powers in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    s1 = x.start**exponent  # 计算区间起始点的指数幂
    s2 = x.end**exponent    # 计算区间结束点的指数幂
    # 根据指数的正负确定区间端点开闭性
    if ((s2 > s1) if exponent > 0 else (x.end > -x.start)) == True:
        left_open = x.left_open
        right_open = x.right_open
        sleft = s2
    else:
        left_open = x.right_open
        right_open = x.left_open
        sleft = s1

    if x.start.is_positive:
        return Interval(
            Min(s1, s2),
            Max(s1, s2), left_open, right_open)
    elif x.end.is_negative:
        return Interval(
            Min(s1, s2),
            Max(s1, s2), left_open, right_open)

    # 处理区间起始点为负、结束点为正的情况
    if exponent.is_odd:
        if exponent.is_negative:
            if x.start.is_zero:
                return Interval(s2, oo, x.right_open)
            if x.end.is_zero:
                return Interval(-oo, s1, True, x.left_open)
            return Union(Interval(-oo, s1, True, x.left_open), Interval(s2, oo, x.right_open))
        else:
            return Interval(s1, s2, x.left_open, x.right_open)
    elif exponent.is_even:
        if exponent.is_negative:
            if x.start.is_zero:
                return Interval(s2, oo, x.right_open)
            if x.end.is_zero:
                return Interval(s1, oo, x.left_open)
            return Interval(0, oo)
        else:
            return Interval(S.Zero, sleft, S.Zero not in x, left_open)


@_set_pow.register(Interval, Infinity)
def _(b, e):
    # 处理区间与无穷大指数的情况
    if b.start.is_nonnegative:
        if b.end < 1:
            return FiniteSet(S.Zero)
        if b.start > 1:
            return FiniteSet(S.Infinity)
        return Interval(0, oo)
    elif b.end.is_negative:
        if b.start > -1:
            return FiniteSet(S.Zero)
        if b.end < -1:
            return FiniteSet(-oo, oo)
        return Interval(-oo, oo)
    # 如果以上条件都不满足，则执行以下代码块
    else:
        # 如果起始值大于 -1
        if b.start > -1:
            # 如果终止值小于 1
            if b.end < 1:
                # 返回包含唯一元素 S.Zero 的有限集
                return FiniteSet(S.Zero)
            # 返回区间 [0, 正无穷)
            return Interval(0, oo)
        # 返回区间 (-无穷, +无穷)
        return Interval(-oo, oo)
# 注册函数 `_set_pow` 的特定实现，处理参数为 `Interval` 类型和 `NegativeInfinity` 类型的情况
@_set_pow.register(Interval, NegativeInfinity)
# 当函数参数为 Interval 类型和 NegativeInfinity 类型时，执行以下逻辑
def _(b, e):
    # 返回将 S.One 除以 b 所得的结果，并传递正无穷大作为指数
    return _set_pow(set_div(S.One, b), oo)
```