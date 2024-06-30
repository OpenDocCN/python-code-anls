# `D:\src\scipysrc\sympy\sympy\calculus\euler.py`

```
"""
This module implements a method to find
Euler-Lagrange Equations for given Lagrangian.
"""
# 从 itertools 模块导入 combinations_with_replacement 函数
from itertools import combinations_with_replacement
# 从 sympy.core.function 中导入 Derivative, Function, diff 函数
from sympy.core.function import (Derivative, Function, diff)
# 从 sympy.core.relational 中导入 Eq 类
from sympy.core.relational import Eq
# 从 sympy.core.singleton 中导入 S 对象
from sympy.core.singleton import S
# 从 sympy.core.symbol 中导入 Symbol 类
from sympy.core.symbol import Symbol
# 从 sympy.core.sympify 中导入 sympify 函数
from sympy.core.sympify import sympify
# 从 sympy.utilities.iterables 中导入 iterable 函数
from sympy.utilities.iterables import iterable


def euler_equations(L, funcs=(), vars=()):
    r"""
    Find the Euler-Lagrange equations [1]_ for a given Lagrangian.

    Parameters
    ==========

    L : Expr
        The Lagrangian that should be a function of the functions listed
        in the second argument and their derivatives.

        For example, in the case of two functions $f(x,y)$, $g(x,y)$ and
        two independent variables $x$, $y$ the Lagrangian has the form:

            .. math:: L\left(f(x,y),g(x,y),\frac{\partial f(x,y)}{\partial x},
                      \frac{\partial f(x,y)}{\partial y},
                      \frac{\partial g(x,y)}{\partial x},
                      \frac{\partial g(x,y)}{\partial y},x,y\right)

        In many cases it is not necessary to provide anything, except the
        Lagrangian, it will be auto-detected (and an error raised if this
        cannot be done).

    funcs : Function or an iterable of Functions
        The functions that the Lagrangian depends on. The Euler equations
        are differential equations for each of these functions.

    vars : Symbol or an iterable of Symbols
        The Symbols that are the independent variables of the functions.

    Returns
    =======

    eqns : list of Eq
        The list of differential equations, one for each function.

    Examples
    ========

    >>> from sympy import euler_equations, Symbol, Function
    >>> x = Function('x')
    >>> t = Symbol('t')
    >>> L = (x(t).diff(t))**2/2 - x(t)**2/2
    >>> euler_equations(L, x(t), t)
    [Eq(-x(t) - Derivative(x(t), (t, 2)), 0)]
    >>> u = Function('u')
    >>> x = Symbol('x')
    >>> L = (u(t, x).diff(t))**2/2 - (u(t, x).diff(x))**2/2
    >>> euler_equations(L, u(t, x), [t, x])
    [Eq(-Derivative(u(t, x), (t, 2)) + Derivative(u(t, x), (x, 2)), 0)]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation

    """

    # 将 funcs 转换为元组形式，如果 funcs 不可迭代则转换为单元素元组
    funcs = tuple(funcs) if iterable(funcs) else (funcs,)

    # 如果 funcs 为空，则自动检测 Lagrangian 中的函数
    if not funcs:
        funcs = tuple(L.atoms(Function))
    else:
        # 检查 funcs 中的每个元素是否为 Function 类型
        for f in funcs:
            if not isinstance(f, Function):
                raise TypeError('Function expected, got: %s' % f)

    # 将 vars 转换为元组形式，如果 vars 不可迭代则转换为单元素元组
    vars = tuple(vars) if iterable(vars) else (vars,)

    # 如果 vars 为空，则使用 funcs 的第一个函数的参数作为 vars
    if not vars:
        vars = funcs[0].args
    else:
        # 将 vars 中的每个元素转换为 Symbol 类型
        vars = tuple(sympify(var) for var in vars)

    # 检查 vars 中的每个元素是否为 Symbol 类型
    if not all(isinstance(v, Symbol) for v in vars):
        raise TypeError('Variables are not symbols, got %s' % vars)

    # 检查 funcs 中的每个函数是否与 vars 匹配其参数
    for f in funcs:
        if not vars == f.args:
            raise ValueError("Variables %s do not match args: %s" % (vars, f))
    # 计算函数列表中特定表达式的导数的最大阶数
    order = max([len(d.variables) for d in L.atoms(Derivative)
                        if d.expr in funcs] + [0])

    # 初始化空的方程列表
    eqns = []
    # 遍历每个函数
    for f in funcs:
        # 计算函数 f 对于拉格朗日量 L 的导数
        eq = diff(L, f)
        # 对每个阶数从 1 到 order 进行遍历
        for i in range(1, order + 1):
            # 对变量列表 vars 进行组合，包括重复组合
            for p in combinations_with_replacement(vars, i):
                # 计算 f 的多重偏导数，并将其添加到方程中
                eq = eq + S.NegativeOne**i * diff(L, diff(f, *p), *p)
        # 创建新的方程 Eq(eq, 0)，表示当前等式为零
        new_eq = Eq(eq, 0)
        # 如果新创建的对象是一个方程 Eq 类型的实例，则将其添加到方程列表中
        if isinstance(new_eq, Eq):
            eqns.append(new_eq)

    # 返回构建的方程列表
    return eqns
```