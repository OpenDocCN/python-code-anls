# `D:\src\scipysrc\sympy\sympy\functions\special\bsplines.py`

```
from sympy.core import S, sympify  # 导入 SymPy 核心功能中的 S 和 sympify 函数
from sympy.core.symbol import (Dummy, symbols)  # 导入 SymPy 核心符号功能中的 Dummy 和 symbols
from sympy.functions import Piecewise, piecewise_fold  # 导入 SymPy 函数模块中的 Piecewise 和 piecewise_fold 函数
from sympy.logic.boolalg import And  # 导入 SymPy 逻辑布尔代数中的 And 类
from sympy.sets.sets import Interval  # 导入 SymPy 集合模块中的 Interval 类

from functools import lru_cache  # 导入 functools 模块中的 lru_cache 装饰器


def _ivl(cond, x):
    """return the interval corresponding to the condition

    Conditions in spline's Piecewise give the range over
    which an expression is valid like (lo <= x) & (x <= hi).
    This function returns (lo, hi).
    """
    if isinstance(cond, And) and len(cond.args) == 2:
        a, b = cond.args
        if a.lts == x:
            a, b = b, a
        return a.lts, b.gts
    raise TypeError('unexpected cond type: %s' % cond)


def _add_splines(c, b1, d, b2, x):
    """Construct c*b1 + d*b2."""

    if S.Zero in (b1, c):
        rv = piecewise_fold(d * b2)
    elif S.Zero in (b2, d):
        rv = piecewise_fold(c * b1)
    else:
        new_args = []
        # Just combining the Piecewise without any fancy optimization
        p1 = piecewise_fold(c * b1)
        p2 = piecewise_fold(d * b2)

        # Search all Piecewise arguments except (0, True)
        p2args = list(p2.args[:-1])

        # This merging algorithm assumes the conditions in
        # p1 and p2 are sorted
        for arg in p1.args[:-1]:
            expr = arg.expr
            cond = arg.cond

            lower = _ivl(cond, x)[0]

            # Check p2 for matching conditions that can be merged
            for i, arg2 in enumerate(p2args):
                expr2 = arg2.expr
                cond2 = arg2.cond

                lower_2, upper_2 = _ivl(cond2, x)
                if cond2 == cond:
                    # Conditions match, join expressions
                    expr += expr2
                    # Remove matching element
                    del p2args[i]
                    # No need to check the rest
                    break
                elif lower_2 < lower and upper_2 <= lower:
                    # Check if arg2 condition smaller than arg1,
                    # add to new_args by itself (no match expected
                    # in p1)
                    new_args.append(arg2)
                    del p2args[i]
                    break

            # Checked all, add expr and cond
            new_args.append((expr, cond))

        # Add remaining items from p2args
        new_args.extend(p2args)

        # Add final (0, True)
        new_args.append((0, True))

        rv = Piecewise(*new_args, evaluate=False)

    return rv.expand()


@lru_cache(maxsize=128)
def bspline_basis(d, knots, n, x):
    """
    The $n$-th B-spline at $x$ of degree $d$ with knots.

    Explanation
    ===========

    B-Splines are piecewise polynomials of degree $d$. They are defined on a
    set of knots, which is a sequence of integers or floats.

    Examples
    ========
    
    """
    # 这个函数计算给定次数、结点和参数的 B-样条基函数
    # 在给定参数 x 处的值。这些基函数是定义在结点序列上的分段多项式。
    pass
    # 确保变量 x 没有任何假设条件，以防止条件语句的评估
    xvar = x
    # 创建一个没有任何假设条件的虚拟变量 x
    x = Dummy()
    
    # 将 knots 中的每个元素转换为符号表示
    knots = tuple(sympify(k) for k in knots)
    # 将 d 转换为整数类型
    d = int(d)
    # 将 n 转换为整数类型
    n = int(n)
    # 计算 knots 的长度
    n_knots = len(knots)
    # 计算区间的数量，即 knots 的长度减一
    n_intervals = n_knots - 1
    
    # 如果 n + d + 1 大于区间的数量，引发值错误异常
    if n + d + 1 > n_intervals:
        raise ValueError("n + d + 1 must not exceed len(knots) - 1")
    
    # 如果 d 等于 0，则生成一个 Piecewise 对象
    if d == 0:
        # 根据给定的区间定义 Piecewise 对象，包括一个为 True 的默认情况
        result = Piecewise(
            (S.One, Interval(knots[n], knots[n + 1]).contains(x)),
            (0, True)
        )
    # 如果 d 大于 0，则生成复杂的 B-spline
    elif d > 0:
        # 计算分母值
        denom = knots[n + d + 1] - knots[n + 1]
        # 如果分母不为零，则计算 B-spline 的一部分
        if denom != S.Zero:
            B = (knots[n + d + 1] - x) / denom
            b2 = bspline_basis(d - 1, knots, n + 1, x)
        else:
            b2 = B = S.Zero
    
        # 计算另一个分母值
        denom = knots[n + d] - knots[n]
        # 如果分母不为零，则计算 B-spline 的另一部分
        if denom != S.Zero:
            A = (x - knots[n]) / denom
            b1 = bspline_basis(d - 1, knots, n, x)
        else:
            b1 = A = S.Zero
    
        # 调用内部函数 _add_splines 来合并计算得到的结果
        result = _add_splines(A, b1, B, b2, x)
    else:
        # 如果 d 小于 0，则引发值错误异常
        raise ValueError("degree must be non-negative: %r" % n)
    
    # 返回最终计算结果，使用原始的用户给定的 x 变量
    return result.xreplace({x: xvar})
# 导入 sympy 中的求解器和矩阵模块
from sympy.solvers.solveset import linsolve
from sympy.matrices.dense import Matrix

# 输入参数的规范化处理
d = sympify(d)
# 如果 d 不是正整数，则引发值错误异常
if not (d.is_Integer and d.is_positive):
    raise ValueError("Spline degree must be a positive integer, not %s." % d)
    # 检查输入的 X 和 Y 是否长度相同，如果不同则抛出值错误异常
    if len(X) != len(Y):
        raise ValueError("Number of X and Y coordinates must be the same.")
    
    # 检查输入的 X 是否至少有 d + 1 个控制点，如果少于则抛出值错误异常
    if len(X) < d + 1:
        raise ValueError("Degree must be less than the number of control points.")
    
    # 检查输入的 X 是否严格递增，如果不是则抛出值错误异常
    if not all(a < b for a, b in zip(X, X[1:])):
        raise ValueError("The x-coordinates must be strictly increasing.")
    
    # 将输入的 X 转换为符号表达式
    X = [sympify(i) for i in X]

    # 计算内部结点的值
    if d.is_odd:
        j = (d + 1) // 2
        interior_knots = X[j:-j]
    else:
        j = d // 2
        interior_knots = [
            (a + b)/2 for a, b in zip(X[j : -j - 1], X[j + 1 : -j])
        ]

    # 构造完整的结点向量
    knots = [X[0]] * (d + 1) + list(interior_knots) + [X[-1]] * (d + 1)

    # 计算 B-样条基函数
    basis = bspline_basis_set(d, knots, x)

    # 构造系数矩阵 A
    A = [[b.subs(x, v) for b in basis] for v in X]

    # 解线性方程组，求出系数向量
    coeff = linsolve((Matrix(A), Matrix(Y)), symbols("c0:{}".format(len(X)), cls=Dummy))
    coeff = list(coeff)[0]

    # 提取所有基函数的区间端点
    intervals = {c for b in basis for (e, c) in b.args if c != True}

    # 对区间进行排序，按照 _ivl 函数返回值排序
    intervals = sorted(intervals, key=lambda c: _ivl(c, x))

    # 为每个基函数构造字典，将系数与基函数关联起来
    basis_dicts = [{c: e for (e, c) in b.args} for b in basis]

    # 构造样条函数
    spline = []
    for i in intervals:
        # 计算每个区间的样条函数片段
        piece = sum(
            [c * d.get(i, S.Zero) for (c, d) in zip(coeff, basis_dicts)], S.Zero
        )
        spline.append((piece, i))

    # 返回构造的分段函数
    return Piecewise(*spline)
```