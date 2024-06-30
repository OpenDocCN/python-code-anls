# `D:\src\scipysrc\sympy\sympy\series\acceleration.py`

```
"""
Convergence acceleration / extrapolation methods for series and
sequences.

References:
Carl M. Bender & Steven A. Orszag, "Advanced Mathematical Methods for
Scientists and Engineers: Asymptotic Methods and Perturbation Theory",
Springer 1999. (Shanks transformation: pp. 368-375, Richardson
extrapolation: pp. 375-377.)
"""

# 导入 sympy 库中所需的模块和函数
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial


def richardson(A, k, n, N):
    """
    Calculate an approximation for lim k->oo A(k) using Richardson
    extrapolation with the terms A(n), A(n+1), ..., A(n+N+1).
    Choosing N ~= 2*n often gives good results.

    Examples
    ========

    A simple example is to calculate exp(1) using the limit definition.
    This limit converges slowly; n = 100 only produces two accurate
    digits:

        >>> from sympy.abc import n
        >>> e = (1 + 1/n)**n
        >>> print(round(e.subs(n, 100).evalf(), 10))
        2.7048138294

    Richardson extrapolation with 11 appropriately chosen terms gives
    a value that is accurate to the indicated precision:

        >>> from sympy import E
        >>> from sympy.series.acceleration import richardson
        >>> print(round(richardson(e, n, 10, 20).evalf(), 10))
        2.7182818285
        >>> print(round(E.evalf(), 10))
        2.7182818285

    Another useful application is to speed up convergence of series.
    Computing 100 terms of the zeta(2) series 1/k**2 yields only
    two accurate digits:

        >>> from sympy.abc import k, n
        >>> from sympy import Sum
        >>> A = Sum(k**-2, (k, 1, n))
        >>> print(round(A.subs(n, 100).evalf(), 10))
        1.6349839002

    Richardson extrapolation performs much better:

        >>> from sympy import pi
        >>> print(round(richardson(A, n, 10, 20).evalf(), 10))
        1.6449340668
        >>> print(round(((pi**2)/6).evalf(), 10))     # Exact value
        1.6449340668

    """
    # 初始化 Richardson 变换的总和
    s = S.Zero
    # 循环计算 Richardson 变换的每一项
    for j in range(0, N + 1):
        s += (A.subs(k, Integer(n + j)).doit() * (n + j)**N *
              S.NegativeOne**(j + N) / (factorial(j) * factorial(N - j)))
    return s


def shanks(A, k, n, m=1):
    """
    Calculate an approximation for lim k->oo A(k) using the n-term Shanks
    transformation S(A)(n). With m > 1, calculate the m-fold recursive
    Shanks transformation S(S(...S(A)...))(n).

    The Shanks transformation is useful for summing Taylor series that
    converge slowly near a pole or singularity, e.g. for log(2):

        >>> from sympy.abc import k, n
        >>> from sympy import Sum, Integer
        >>> from sympy.series.acceleration import shanks
        >>> A = Sum(Integer(-1)**(k+1) / k, (k, 1, n))
        >>> print(round(A.subs(n, 100).doit().evalf(), 10))
        0.6881721793
        >>> print(round(shanks(A, n, 25).evalf(), 10))
        0.6931396564
        >>> print(round(shanks(A, n, 25, 5).evalf(), 10))
        0.6931471806

    """
    # 在这里实现 Shanks 变换的递归计算
    pass
    The correct value is 0.6931471805599453094172321215.
    """
    # 创建一个列表，其中包含符号表达式 A.subs(k, Integer(j)).doit() 的计算结果，从 j = 0 到 j = n + m + 1
    table = [A.subs(k, Integer(j)).doit() for j in range(n + m + 2)]
    # 复制 table 列表到 table2
    table2 = table[:]

    # 外层循环从 i = 1 到 i = m
    for i in range(1, m + 1):
        # 内层循环从 j = i 到 j = n + m
        for j in range(i, n + m + 1):
            # 从 table 中取出三个相邻元素
            x, y, z = table[j - 1], table[j], table[j + 1]
            # 计算新的 table2[j] 值并存储
            table2[j] = (z*x - y**2) / (z + x - 2*y)
        # 将更新后的 table2 复制回 table
        table = table2[:]
    
    # 返回 table 中索引为 n 的值
    return table[n]
```