# `D:\src\scipysrc\sympy\sympy\calculus\finite_diff.py`

```
"""
Finite difference weights
=========================

This module implements an algorithm for efficient generation of finite
difference weights for ordinary differentials of functions for
derivatives from 0 (interpolation) up to arbitrary order.

The core algorithm is provided in the finite difference weight generating
function (``finite_diff_weights``), and two convenience functions are provided
for:

- estimating a derivative (or interpolate) directly from a series of points
    is also provided (``apply_finite_diff``).
- differentiating by using finite difference approximations
    (``differentiate_finite``).

"""

from sympy.core.function import Derivative  # 导入 Derivative 函数，用于表示导数
from sympy.core.singleton import S  # 导入 S，表示符号 1
from sympy.core.function import Subs  # 导入 Subs 函数，用于符号代换
from sympy.core.traversal import preorder_traversal  # 导入 preorder_traversal 函数，用于树的前序遍历
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入 sympy_deprecation_warning，用于显示 SymPy 废弃警告
from sympy.utilities.iterables import iterable  # 导入 iterable 函数，用于检查是否为可迭代对象


def finite_diff_weights(order, x_list, x0=S.One):
    """
    Calculates the finite difference weights for an arbitrarily spaced
    one-dimensional grid (``x_list``) for derivatives at ``x0`` of order
    0, 1, ..., up to ``order`` using a recursive formula. Order of accuracy
    is at least ``len(x_list) - order``, if ``x_list`` is defined correctly.

    Parameters
    ==========

    order: int
        Up to what derivative order weights should be calculated.
        0 corresponds to interpolation.
    x_list: sequence
        Sequence of (unique) values for the independent variable.
        It is useful (but not necessary) to order ``x_list`` from
        nearest to furthest from ``x0``; see examples below.
    x0: Number or Symbol
        Root or value of the independent variable for which the finite
        difference weights should be generated. Default is ``S.One``.

    Returns
    =======

    list
        A list of sublists, each corresponding to coefficients for
        increasing derivative order, and each containing lists of
        coefficients for increasing subsets of x_list.

    Examples
    ========

    >>> from sympy import finite_diff_weights, S
    >>> res = finite_diff_weights(1, [-S(1)/2, S(1)/2, S(3)/2, S(5)/2], 0)
    >>> res
    [[[1, 0, 0, 0],
      [1/2, 1/2, 0, 0],
      [3/8, 3/4, -1/8, 0],
      [5/16, 15/16, -5/16, 1/16]],
     [[0, 0, 0, 0],
      [-1, 1, 0, 0],
      [-1, 1, 0, 0],
      [-23/24, 7/8, 1/8, -1/24]]]
    >>> res[0][-1]  # FD weights for 0th derivative, using full x_list
    [5/16, 15/16, -5/16, 1/16]
    >>> res[1][-1]  # FD weights for 1st derivative
    [-23/24, 7/8, 1/8, -1/24]
    >>> res[1][-2]  # FD weights for 1st derivative, using x_list[:-1]
    [-1, 1, 0, 0]
    >>> res[1][-1][0]  # FD weight for 1st deriv. for x_list[0]
    -23/24
    >>> res[1][-1][1]  # FD weight for 1st deriv. for x_list[1], etc.
    7/8

    Each sublist contains the most accurate formula at the end.
    Note, that in the above example ``res[1][1]`` is the same as ``res[1][2]``.

    """
    Since res[1][2] has an order of accuracy of
    ``len(x_list[:3]) - order = 3 - 1 = 2``, the same is true for ``res[1][1]``!

    >>> res = finite_diff_weights(1, [S(0), S(1), -S(1), S(2), -S(2)], 0)[1]
    >>> res
    [[0, 0, 0, 0, 0],  # Initialization of a matrix for storing finite difference weights
     [-1, 1, 0, 0, 0],  # Classic forward step approximation weights
     [0, 1/2, -1/2, 0, 0],  # Classic centered approximation weights
     [-1/2, 1, -1/3, -1/6, 0],  # Higher order approximation weights
     [0, 2/3, -2/3, -1/12, 1/12]]  # Higher order approximation weights

    >>> res[0]  # no approximation possible, using x_list[0] only
    [0, 0, 0, 0, 0]  # Zero weights indicating no approximation

    >>> res[1]  # classic forward step approximation
    [-1, 1, 0, 0, 0]  # Finite difference weights for classic forward step approximation

    >>> res[2]  # classic centered approximation
    [0, 1/2, -1/2, 0, 0]  # Finite difference weights for classic centered approximation

    >>> res[3:]  # higher order approximations
    [[-1/2, 1, -1/3, -1/6, 0], [0, 2/3, -2/3, -1/12, 1/12]]  # Finite difference weights for higher order approximations

    Let us compare this to a differently defined ``x_list``. Pay attention to
    ``foo[i][k]`` corresponding to the gridpoint defined by ``x_list[k]``.

    >>> foo = finite_diff_weights(1, [-S(2), -S(1), S(0), S(1), S(2)], 0)[1]
    >>> foo
    [[0, 0, 0, 0, 0],  # Initialization of a matrix for storing finite difference weights
     [-1, 1, 0, 0, 0],  # Classic forward step approximation weights
     [1/2, -2, 3/2, 0, 0],  # Classic double backward step approximation weights
     [1/6, -1, 1/2, 1/3, 0],  # Finite difference weights for higher order approximations
     [1/12, -2/3, 0, 2/3, -1/12]]  # Finite difference weights for higher order approximations

    >>> foo[1]  # not the same and of lower accuracy as res[1]!
    [-1, 1, 0, 0, 0]  # Comparison comment

    >>> foo[2]  # classic double backward step approximation
    [1/2, -2, 3/2, 0, 0]  # Finite difference weights for classic double backward step approximation

    >>> foo[4]  # the same as res[4]
    [1/12, -2/3, 0, 2/3, -1/12]  # Matching finite difference weights with res[4]

    Note that, unless you plan on using approximations based on subsets of
    ``x_list``, the order of gridpoints does not matter.

    The capability to generate weights at arbitrary points can be
    used e.g. to minimize Runge's phenomenon by using Chebyshev nodes:

    >>> from sympy import cos, symbols, pi, simplify
    >>> N, (h, x) = 4, symbols('h x')
    >>> x_list = [x+h*cos(i*pi/(N)) for i in range(N,-1,-1)] # chebyshev nodes
    >>> print(x_list)
    [-h + x, -sqrt(2)*h/2 + x, x, sqrt(2)*h/2 + x, h + x]  # List of Chebyshev nodes

    >>> mycoeffs = finite_diff_weights(1, x_list, 0)[1][4]
    >>> [simplify(c) for c in  mycoeffs] #doctest: +NORMALIZE_WHITESPACE
    [(h**3/2 + h**2*x - 3*h*x**2 - 4*x**3)/h**4,
    (-sqrt(2)*h**3 - 4*h**2*x + 3*sqrt(2)*h*x**2 + 8*x**3)/h**4,
    (6*h**2*x - 8*x**3)/h**4,
    (sqrt(2)*h**3 - 4*h**2*x - 3*sqrt(2)*h*x**2 + 8*x**3)/h**4,
    (-h**3/2 + h**2*x + 3*h*x**2 - 4*x**3)/h**4]  # Simplified coefficients for finite difference weights

    Notes
    =====

    If weights for a finite difference approximation of 3rd order
    derivative is wanted, weights for 0th, 1st and 2nd order are
    calculated "for free", so are formulae using subsets of ``x_list``.
    This is something one can take advantage of to save computational cost.
    Be aware that one should define ``x_list`` from nearest to furthest from
    ``x0``. If not, subsets of ``x_list`` will yield poorer approximations,
    which might not grant an order of accuracy of ``len(x_list) - order``.

    See also
    ========

    sympy.calculus.finite_diff.apply_finite_diff

    References
    ==========
    """
    # 根据文献 [1] 中描述的方法生成有限差分公式。
    order = S(order)  # 将 order 转换为符号表达式 S(order)
    if not order.is_number:  # 检查 order 是否为数值，如果不是则引发错误
        raise ValueError("Cannot handle symbolic order.")
    if order < 0:  # 检查 order 是否小于零，如果是则引发错误
        raise ValueError("Negative derivative order illegal.")
    if int(order) != order:  # 检查 order 是否为整数，如果不是则引发错误
        raise ValueError("Non-integer order illegal")
    M = order  # 将 order 赋值给 M
    N = len(x_list) - 1  # 计算 x_list 的长度减一，赋值给 N
    delta = [[[0 for nu in range(N+1)] for n in range(N+1)] for
             m in range(M+1)]  # 创建一个三维列表 delta，用于存储有限差分公式的系数
    delta[0][0][0] = S.One  # 设置 delta[0][0][0] 为符号表达式 S.One
    c1 = S.One  # 初始化 c1 为符号表达式 S.One
    for n in range(1, N+1):  # 遍历 n 从 1 到 N
        c2 = S.One  # 初始化 c2 为符号表达式 S.One
        for nu in range(n):  # 遍历 nu 从 0 到 n-1
            c3 = x_list[n] - x_list[nu]  # 计算差值 c3
            c2 = c2 * c3  # 更新 c2 的值
            if n <= M:  # 如果 n 小于等于 M
                delta[n][n-1][nu] = 0  # 将 delta[n][n-1][nu] 设为 0
            for m in range(min(n, M)+1):  # 遍历 m 从 0 到 min(n, M)
                delta[m][n][nu] = (x_list[n]-x0)*delta[m][n-1][nu] -\
                    m*delta[m-1][n-1][nu]  # 计算 delta[m][n][nu] 的值
                delta[m][n][nu] /= c3  # 对 delta[m][n][nu] 进行除法运算
        for m in range(min(n, M)+1):  # 遍历 m 从 0 到 min(n, M)
            delta[m][n][n] = c1/c2*(m*delta[m-1][n-1][n-1] -
                                    (x_list[n-1]-x0)*delta[m][n-1][n-1])  # 计算 delta[m][n][n] 的值
        c1 = c2  # 更新 c1 的值
    return delta  # 返回生成的有限差分公式的系数
    """
# 返回一个有限差分公式的函数近似值，计算函数在离散值上的加权和
def _as_finite_diff(derivative, points=1, x0=None, wrt=None):
    """
    Returns an approximation of a derivative of a function in
    the form of a finite difference formula. The expression is a
    weighted sum of the function at a number of discrete values of
    (one of) the independent variable(s).

    Parameters
    ==========

    derivative: sympy expression
        The expression representing the derivative of the function.
    points: int, optional
        Number of discrete points to use for finite differencing.
    x0: Number or Symbol, optional
        Value(s) of the independent variable(s) where the derivative
        is evaluated. If None, uses a default value.
    wrt: Symbol, optional
        Symbol representing the independent variable with respect to
        which the derivative is taken.

    Returns
    =======

    sympy expression
        A weighted sum expression representing the finite difference
        approximation of the derivative.

    Notes
    =====

    This function constructs a finite difference formula to approximate
    the derivative of a function based on discrete evaluations at several
    points. The accuracy of the approximation depends on the number of
    points used and the behavior of the function in the vicinity of x0.

    """
    """
    Check the type of the derivative object and compute finite differences if needed.

    Parameters
    ==========

    derivative: a Derivative instance
        Represents the symbolic expression of a derivative.

    points: sequence or coefficient, optional
        If a sequence: discrete values (length >= order+1) of the
        independent variable used for generating the finite
        difference weights.
        If it is a coefficient, it will be used as the step-size
        for generating an equidistant sequence of length order+1
        centered around `x0`. default: 1 (step-size 1)

    x0: number or Symbol, optional
        The value of the independent variable (`wrt`) at which the
        derivative is to be approximated. Default: same as `wrt`.

    wrt: Symbol, optional
        "With respect to" the variable for which the (partial)
        derivative is to be approximated. If not provided, it
        is required that the Derivative is ordinary. Default: `None`.

    Returns
    =======

    The computed derivative expression, possibly using finite differences.

    Examples
    ========

    See examples in the docstring for usage scenarios.

    See also
    ========

    sympy.calculus.finite_diff.apply_finite_diff
    sympy.calculus.finite_diff.finite_diff_weights
    """
    # Check if the derivative is a Derivative instance
    if derivative.is_Derivative:
        pass
    # If derivative is an Atom (like a symbol or number), return it directly
    elif derivative.is_Atom:
        return derivative
    else:
        # Recursively compute finite differences for each argument in derivative.args
        return derivative.fromiter(
            [_as_finite_diff(ar, points, x0, wrt) for ar
             in derivative.args], **derivative.assumptions0)

    # If wrt is not specified, compute finite differences for each variable in derivative
    if wrt is None:
        old = None
        for v in derivative.variables:
            # Skip consecutive same variables
            if old is v:
                continue
            derivative = _as_finite_diff(derivative, points, x0, v)
            old = v
        return derivative

    # Count the order of the derivative with respect to wrt variable
    order = derivative.variables.count(wrt)
    # 如果初始点 x0 为 None，则将其设为 wrt
    if x0 is None:
        x0 = wrt

    # 如果 points 不可迭代
    if not iterable(points):
        # 如果 points 是函数，并且其中包含 wrt
        if getattr(points, 'is_Function', False) and wrt in points.args:
            # 将 points 中的 wrt 替换为 x0
            points = points.subs(wrt, x0)
        
        # points 现在仅是步长，让我们将其转换为围绕 x0 的等距序列
        if order % 2 == 0:
            # 偶数阶 => 奇数个点，包括网格点
            points = [x0 + points*i for i
                      in range(-order//2, order//2 + 1)]
        else:
            # 奇数阶 => 偶数个点，位于网格点之间的中点
            points = [x0 + points*S(i)/2 for i
                      in range(-order, order + 1, 2)]
    
    # others 列表初始化为 [wrt, 0]
    others = [wrt, 0]
    # 遍历导数变量的集合
    for v in set(derivative.variables):
        # 如果 v 等于 wrt，则跳过
        if v == wrt:
            continue
        # 将 v 添加到 others 中，并统计导数变量中 v 的出现次数
        others += [v, derivative.variables.count(v)]
    
    # 如果 points 的长度小于 order+1，则抛出数值错误
    if len(points) < order+1:
        raise ValueError("Too few points for order %d" % order)
    
    # 应用有限差分，返回结果
    return apply_finite_diff(order, points, [
        # 对 points 中的每个点 x，计算导数表达式在 x 处的导数
        Derivative(derivative.expr.subs({wrt: x}), *others) for
        x in points], x0)
def differentiate_finite(expr, *symbols,
                         points=1, x0=None, wrt=None, evaluate=False):
    r""" Differentiate expr and replace Derivatives with finite differences.

    Parameters
    ==========

    expr : expression
    \*symbols : differentiate with respect to symbols
    points: sequence, coefficient or undefined function, optional
        see ``Derivative.as_finite_difference``
    x0: number or Symbol, optional
        see ``Derivative.as_finite_difference``
    wrt: Symbol, optional
        see ``Derivative.as_finite_difference``

    Examples
    ========

    >>> from sympy import sin, Function, differentiate_finite
    >>> from sympy.abc import x, y, h
    >>> f, g = Function('f'), Function('g')
    >>> differentiate_finite(f(x)*g(x), x, points=[x-h, x+h])
    -f(-h + x)*g(-h + x)/(2*h) + f(h + x)*g(h + x)/(2*h)

    ``differentiate_finite`` works on any expression, including the expressions
    with embedded derivatives:

    >>> differentiate_finite(f(x) + sin(x), x, 2)
    -2*f(x) + f(x - 1) + f(x + 1) - 2*sin(x) + sin(x - 1) + sin(x + 1)
    >>> differentiate_finite(f(x, y), x, y)
    f(x - 1/2, y - 1/2) - f(x - 1/2, y + 1/2) - f(x + 1/2, y - 1/2) + f(x + 1/2, y + 1/2)
    >>> differentiate_finite(f(x)*g(x).diff(x), x)
    (-g(x) + g(x + 1))*f(x + 1/2) - (g(x) - g(x - 1))*f(x - 1/2)

    To make finite difference with non-constant discretization step use
    undefined functions:

    >>> dx = Function('dx')
    >>> differentiate_finite(f(x)*g(x).diff(x), points=dx(x))
    -(-g(x - dx(x)/2 - dx(x - dx(x)/2)/2)/dx(x - dx(x)/2) +
    g(x - dx(x)/2 + dx(x - dx(x)/2)/2)/dx(x - dx(x)/2))*f(x - dx(x)/2)/dx(x) +
    (-g(x + dx(x)/2 - dx(x + dx(x)/2)/2)/dx(x + dx(x)/2) +
    g(x + dx(x)/2 + dx(x + dx(x)/2)/2)/dx(x + dx(x)/2))*f(x + dx(x)/2)/dx(x)

    """
    # 检查表达式中是否包含 Derivative 对象，如果有则强制不评估
    if any(term.is_Derivative for term in list(preorder_traversal(expr))):
        evaluate = False

    # 对表达式进行符号 symbols 的求导，使用 evaluate 标志控制是否评估
    Dexpr = expr.diff(*symbols, evaluate=evaluate)
    
    # 如果 evaluate 为 True，发出 sympy 的弃用警告，并用有限差分替换 Derivative 对象
    if evaluate:
        sympy_deprecation_warning("""
        The evaluate flag to differentiate_finite() is deprecated.

        evaluate=True expands the intermediate derivatives before computing
        differences, but this usually not what you want, as it does not
        satisfy the product rule.
        """,
            deprecated_since_version="1.5",
            active_deprecations_target="deprecated-differentiate_finite-evaluate",
        )
        return Dexpr.replace(
            # 替换 Derivative 对象为有限差分
            lambda arg: arg.is_Derivative,
            lambda arg: arg.as_finite_difference(points=points, x0=x0, wrt=wrt))
    else:
        # 否则直接将求导后的表达式使用有限差分进行替换
        DFexpr = Dexpr.as_finite_difference(points=points, x0=x0, wrt=wrt)
        return DFexpr.replace(
            # 替换 Subs 对象为其表达式的有限差分
            lambda arg: isinstance(arg, Subs),
            lambda arg: arg.expr.as_finite_difference(
                    points=points, x0=arg.point[0], wrt=arg.variables[0]))
```