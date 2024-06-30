# `D:\src\scipysrc\sympy\sympy\codegen\approximations.py`

```
import math
from sympy.sets.sets import Interval  # 导入Interval类，用于表示数学区间
from sympy.calculus.singularities import is_increasing, is_decreasing  # 导入判断单调性的函数
from sympy.codegen.rewriting import Optimization  # 导入优化重写相关的类
from sympy.core.function import UndefinedFunction  # 导入UndefinedFunction类，用于表示未定义的数学函数

"""
This module collects classes useful for approimate rewriting of expressions.
This can be beneficial when generating numeric code for which performance is
of greater importance than precision (e.g. for preconditioners used in iterative
methods).
"""

class SumApprox(Optimization):
    """
    Approximates sum by neglecting small terms.

    Explanation
    ===========

    If terms are expressions which can be determined to be monotonic, then
    bounds for those expressions are added.

    Parameters
    ==========

    bounds : dict
        Mapping expressions to length 2 tuple of bounds (low, high).
    reltol : number
        Threshold for when to ignore a term. Taken relative to the largest
        lower bound among bounds.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.abc import x, y, z
    >>> from sympy.codegen.rewriting import optimize
    >>> from sympy.codegen.approximations import SumApprox
    >>> bounds = {x: (-1, 1), y: (1000, 2000), z: (-10, 3)}
    >>> sum_approx3 = SumApprox(bounds, reltol=1e-3)
    >>> sum_approx2 = SumApprox(bounds, reltol=1e-2)
    >>> sum_approx1 = SumApprox(bounds, reltol=1e-1)
    >>> expr = 3*(x + y + exp(z))
    >>> optimize(expr, [sum_approx3])
    3*(x + y + exp(z))
    >>> optimize(expr, [sum_approx2])
    3*y + 3*exp(z)
    >>> optimize(expr, [sum_approx1])
    3*y

    """

    def __init__(self, bounds, reltol, **kwargs):
        super().__init__(**kwargs)  # 调用父类的构造函数
        self.bounds = bounds  # 设置表达式到边界元组的映射
        self.reltol = reltol  # 设置忽略项的相对容差阈值

    def __call__(self, expr):
        return expr.factor().replace(self.query, lambda arg: self.value(arg))

    def query(self, expr):
        return expr.is_Add  # 查询表达式是否是加法运算的形式
    # 定义一个方法 `value`，接收一个参数 `add`
    def value(self, add):
        # 遍历加法表达式中的每个项
        for term in add.args:
            # 如果当前项是数字、已在边界中，或者自由符号数不为1，则跳过
            if term.is_number or term in self.bounds or len(term.free_symbols) != 1:
                continue
            # 获取当前项的唯一自由符号
            fs, = term.free_symbols
            # 如果该自由符号不在已知边界内，则跳过
            if fs not in self.bounds:
                continue
            # 根据已知的边界创建一个区间对象
            intrvl = Interval(*self.bounds[fs])
            # 如果当前项在指定区间内是增函数
            if is_increasing(term, intrvl, fs):
                # 更新当前项的边界为区间的左右端点值
                self.bounds[term] = (
                    term.subs({fs: self.bounds[fs][0]}),
                    term.subs({fs: self.bounds[fs][1]})
                )
            # 如果当前项在指定区间内是减函数
            elif is_decreasing(term, intrvl, fs):
                # 更新当前项的边界为区间的右左端点值（反向更新）
                self.bounds[term] = (
                    term.subs({fs: self.bounds[fs][1]}),
                    term.subs({fs: self.bounds[fs][0]})
                )
            else:
                # 如果无法确定当前项的增减性质，则返回原始加法表达式
                return add

        # 如果所有项都是数字或已在边界中
        if all(term.is_number or term in self.bounds for term in add.args):
            # 对于每个项，确定其边界（如果是数字则边界为自身）
            bounds = [(term, term) if term.is_number else self.bounds[term] for term in add.args]
            # 初始化最大绝对保证值
            largest_abs_guarantee = 0
            # 遍历所有项的边界
            for lo, hi in bounds:
                # 如果边界包含0，则跳过
                if lo <= 0 <= hi:
                    continue
                # 更新最大绝对保证值为当前项边界中的最小绝对值
                largest_abs_guarantee = max(largest_abs_guarantee,
                                            min(abs(lo), abs(hi)))
            # 初始化一个新的项列表
            new_terms = []
            # 遍历每个项及其边界
            for term, (lo, hi) in zip(add.args, bounds):
                # 如果当前项的边界绝对值大于等于最大绝对保证值乘以相对误差容忍度
                if max(abs(lo), abs(hi)) >= largest_abs_guarantee * self.reltol:
                    # 将当前项添加到新的项列表中
                    new_terms.append(term)
            # 返回一个新的加法表达式，仅包含新的符合条件的项
            return add.func(*new_terms)
        else:
            # 如果不是所有项都是数字或已在边界中，则返回原始加法表达式
            return add
    """ Approximates functions by expanding them as a series.

    Parameters
    ==========

    bounds : dict
        Mapping expressions to length 2 tuple of bounds (low, high).
    reltol : number
        Threshold for when to ignore a term. Taken relative to the largest
        lower bound among bounds.
    max_order : int
        Largest order to include in series expansion
    n_point_checks : int (even)
        The validity of an expansion (with respect to reltol) is checked at
        discrete points (linearly spaced over the bounds of the variable). The
        number of points used in this numerical check is given by this number.

    Examples
    ========

    >>> from sympy import sin, pi
    >>> from sympy.abc import x, y
    >>> from sympy.codegen.rewriting import optimize
    >>> from sympy.codegen.approximations import SeriesApprox
    >>> bounds = {x: (-.1, .1), y: (pi-1, pi+1)}
    >>> series_approx2 = SeriesApprox(bounds, reltol=1e-2)
    >>> series_approx3 = SeriesApprox(bounds, reltol=1e-3)
    >>> series_approx8 = SeriesApprox(bounds, reltol=1e-8)
    >>> expr = sin(x)*sin(y)
    >>> optimize(expr, [series_approx2])
    x*(-y + (y - pi)**3/6 + pi)
    >>> optimize(expr, [series_approx3])
    (-x**3/6 + x)*sin(y)
    >>> optimize(expr, [series_approx8])
    sin(x)*sin(y)

    """
    def __init__(self, bounds, reltol, max_order=4, n_point_checks=4, **kwargs):
        # 调用父类构造函数进行初始化
        super().__init__(**kwargs)
        # 设置边界
        self.bounds = bounds
        # 设置相对误差阈值
        self.reltol = reltol
        # 设置最大展开阶数
        self.max_order = max_order
        # 如果点检查数为奇数，则抛出异常
        if n_point_checks % 2 == 1:
            raise ValueError("Checking the solution at expansion point is not helpful")
        # 设置点检查数
        self.n_point_checks = n_point_checks
        # 设置精度，基于相对误差的对数值
        self._prec = math.ceil(-math.log10(self.reltol))

    def __call__(self, expr):
        # 对表达式进行因式分解，并替换查询结果
        return expr.factor().replace(self.query, lambda arg: self.value(arg))

    def query(self, expr):
        # 查询函数，判断是否为单参数函数且不是未定义函数
        return (expr.is_Function and not isinstance(expr, UndefinedFunction)
                and len(expr.args) == 1)
    # 定义一个方法 `value`，用于处理表达式 `fexpr` 的计算
    def value(self, fexpr):
        # 获取表达式中的自由符号
        free_symbols = fexpr.free_symbols
        # 如果自由符号数量不等于1，则直接返回原表达式
        if len(free_symbols) != 1:
            return fexpr
        # 解构获取唯一的自由符号
        symb, = free_symbols
        # 如果该符号不在预设的边界中，则返回原表达式
        if symb not in self.bounds:
            return fexpr
        # 获取符号对应的上下界
        lo, hi = self.bounds[symb]
        # 计算符号的初始值
        x0 = (lo + hi) / 2
        cheapest = None
        # 从最高阶次到最低阶次遍历
        for n in range(self.max_order + 1, 0, -1):
            # 对表达式关于符号的 n 阶级数展开，并去除高阶无关项
            fseri = fexpr.series(symb, x0=x0, n=n).removeO()
            n_ok = True
            # 对若干个点进行检查，以验证级数展开的准确性
            for idx in range(self.n_point_checks):
                x = lo + idx * (hi - lo) / (self.n_point_checks - 1)
                # 计算级数展开在当前点 x 处的值
                val = fseri.xreplace({symb: x})
                # 计算原始表达式在当前点 x 处的值
                ref = fexpr.xreplace({symb: x})
                # 检查相对误差是否超出允许的相对误差限制
                if abs((1 - val/ref).evalf(self._prec)) > self.reltol:
                    n_ok = False
                    break

            # 如果通过所有点的检查，则更新最优的级数展开结果
            if n_ok:
                cheapest = fseri
            else:
                break

        # 如果没有找到满足条件的级数展开，则返回原表达式；否则返回最优的级数展开结果
        if cheapest is None:
            return fexpr
        else:
            return cheapest
```