# `D:\src\scipysrc\sympy\sympy\physics\wigner.py`

```
# -*- coding: utf-8 -*-
r"""
Wigner, Clebsch-Gordan, Racah, and Gaunt coefficients

Collection of functions for calculating Wigner 3j, 6j, 9j,
Clebsch-Gordan, Racah as well as Gaunt coefficients exactly, all
evaluating to a rational number times the square root of a rational
number [Rasch03]_.

Please see the description of the individual functions for further
details and examples.

References
==========

.. [Regge58] 'Symmetry Properties of Clebsch-Gordan Coefficients',
  T. Regge, Nuovo Cimento, Volume 10, pp. 544 (1958)
.. [Regge59] 'Symmetry Properties of Racah Coefficients',
  T. Regge, Nuovo Cimento, Volume 11, pp. 116 (1959)
.. [Edmonds74] A. R. Edmonds. Angular momentum in quantum mechanics.
  Investigations in physics, 4.; Investigations in physics, no. 4.
  Princeton, N.J., Princeton University Press, 1957.
.. [Rasch03] J. Rasch and A. C. H. Yu, 'Efficient Storage Scheme for
  Pre-calculated Wigner 3j, 6j and Gaunt Coefficients', SIAM
  J. Sci. Comput. Volume 25, Issue 4, pp. 1416-1428 (2003)
.. [Liberatodebrito82] 'FORTRAN program for the integral of three
  spherical harmonics', A. Liberato de Brito,
  Comput. Phys. Commun., Volume 25, pp. 81-85 (1982)
.. [Homeier96] 'Some Properties of the Coupling Coefficients of Real
  Spherical Harmonics and Their Relation to Gaunt Coefficients',
  H. H. H. Homeier and E. O. Steinborn J. Mol. Struct., Volume 368,
  pp. 31-37 (1996)

Credits and Copyright
=====================

This code was taken from Sage with the permission of all authors:

https://groups.google.com/forum/#!topic/sage-devel/M4NZdu-7O38

Authors
=======

- Jens Rasch (2009-03-24): initial version for Sage

- Jens Rasch (2009-05-31): updated to sage-4.0

- Oscar Gerardo Lazo Arjona (2017-06-18): added Wigner D matrices

- Phil Adam LeMaitre (2022-09-19): added real Gaunt coefficient

Copyright (C) 2008 Jens Rasch <jyr2000@gmail.com>

"""
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.numbers import int_valued
from sympy.core.function import Function
from sympy.core.numbers import (Float, I, Integer, pi, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.utilities.misc import as_int

# This list of precomputed factorials is needed to massively
# accelerate future calculations of the various coefficients
_Factlist = [1]


def _calc_factlist(nn):
    r"""
    Function calculates a list of precomputed factorials in order to
    """
    massively accelerate future calculations of the various
    coefficients.


    # 大幅加速未来各种系数的计算。



    Parameters
    ==========

    nn : integer
        Highest factorial to be computed.


    # 参数
    # ====
    # nn : 整数
    #     要计算的最高阶乘数。



    Returns
    =======

    list of integers :
        The list of precomputed factorials.


    # 返回
    # =======
    # 整数列表 :
    #     预先计算好的阶乘列表。



    Examples
    ========

    Calculate list of factorials::


    # 示例
    # ======
    # 计算阶乘列表::



        sage: from sage.functions.wigner import _calc_factlist
        sage: _calc_factlist(10)
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    """


        # 使用 Sage 中的 _calc_factlist 函数计算阶乘列表
        sage: from sage.functions.wigner import _calc_factlist
        sage: _calc_factlist(10)
        # 返回预期的阶乘列表结果
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    """



    if nn >= len(_Factlist):


    # 如果要计算的最高阶乘数大于等于已经计算的阶乘列表长度



        for ii in range(len(_Factlist), int(nn + 1)):
            _Factlist.append(_Factlist[ii - 1] * ii)


        # 遍历从已计算列表长度到 nn+1 的范围
        for ii in range(len(_Factlist), int(nn + 1)):
            # 计算并添加新的阶乘值到 _Factlist 中
            _Factlist.append(_Factlist[ii - 1] * ii)



    return _Factlist[:int(nn) + 1]


    # 返回计算好的阶乘列表，截取到指定长度 nn+1
    return _Factlist[:int(nn) + 1]
# 返回 Python 整数，除非 value 是半整数（此时返回浮点数）
def _int_or_halfint(value):
    if isinstance(value, int):
        return value  # 如果 value 是整数，直接返回
    elif type(value) is float:
        if value.is_integer():
            return int(value)  # 如果 value 是整数浮点数，返回整数部分
        if (2*value).is_integer():
            return value  # 如果 value 是半整数浮点数，返回浮点数本身
    elif isinstance(value, Rational):
        if value.q == 2:
            return value.p/value.q  # 如果 value 是有理数且分母为 2，返回其浮点数值
        elif value.q == 1:
            return value.p  # 如果 value 是有理数且分母为 1，返回其整数值
    elif isinstance(value, Float):
        return _int_or_halfint(float(value))  # 如果 value 是浮点数对象，转换成浮点数后递归调用本函数
    raise ValueError("expecting integer or half-integer, got %s" % value)  # 若 value 类型不符合预期，抛出异常

# 计算 Wigner 3j 符号 `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)` 的值
def wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3):
    r"""
    计算 Wigner 3j 符号 `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`。

    Parameters
    ==========

    j_1, j_2, j_3, m_1, m_2, m_3 :
        整数或半整数。

    Returns
    =======

    有理数乘以有理数的平方根。

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_3j
    >>> wigner_3j(2, 6, 4, 0, 0, 0)
    sqrt(715)/143
    >>> wigner_3j(2, 6, 4, 0, 0, 1)
    0

    参数不是整数或半整数值会引发错误::

        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer
        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer or half integer

    Notes
    =====

    Wigner 3j 符号遵循以下对称性规则：

    - 在列的任何排列下不变（除了当 `J:=j_1+j_2+j_3` 时会有符号变化）。

    - 在空间反演下不变。

    - 关于 [Regge58]_ 的工作具有对称性。

    - 不符合三角关系的 `j_1`, `j_2`, `j_3` 为零。

    - `m_1 + m_2 + m_3 \neq 0` 为零。

    - 违反以下条件之一为零：
         `m_1  \in \{-|j_1|, \ldots, |j_1|\}`,
         `m_2  \in \{-|j_2|, \ldots, |j_2|\}`,
         `m_3  \in \{-|j_3|, \ldots, |j_3|\}`

    Algorithm
    =========

    此函数使用 [Edmonds74]_ 的算法精确计算 3j 符号的值。注意公式包含
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """
    # 将输入的参数 j_1, j_2, j_3, m_1, m_2, m_3 转换为整数或半整数类型
    j_1, j_2, j_3, m_1, m_2, m_3 = map(_int_or_halfint,
                                       [j_1, j_2, j_3, m_1, m_2, m_3])

    # 如果 m_1 + m_2 + m_3 不等于 0，则返回零
    if m_1 + m_2 + m_3 != 0:
        return S.Zero
    # 计算 a1，并检查是否小于 0，如果是则返回零
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return S.Zero
    # 计算 a2，并检查是否小于 0，如果是则返回零
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return S.Zero
    # 计算 a3，并检查是否小于 0，如果是则返回零
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return S.Zero
    # 检查是否有任何 m_i 的绝对值大于对应的 j_i，如果是则返回零
    if (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3):
        return S.Zero
    # 检查 j_i - m_i 是否为整数值，如果不是则返回零
    if not (int_valued(j_1 - m_1) and \
            int_valued(j_2 - m_2) and \
            int_valued(j_3 - m_3)):
        return S.Zero

    # 计算 maxfact，是 j_1, j_2, j_3, m_1, m_2, m_3 加上 1 后的最大值
    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2),
                  j_3 + abs(m_3))
    # 调用 _calc_factlist 函数计算阶乘列表
    _calc_factlist(int(maxfact))

    # 计算 argsqrt，用于后续计算平方根
    argsqrt = Integer(_Factlist[int(j_1 + j_2 - j_3)] *
                     _Factlist[int(j_1 - j_2 + j_3)] *
                     _Factlist[int(-j_1 + j_2 + j_3)] *
                     _Factlist[int(j_1 - m_1)] *
                     _Factlist[int(j_1 + m_1)] *
                     _Factlist[int(j_2 - m_2)] *
                     _Factlist[int(j_2 + m_2)] *
                     _Factlist[int(j_3 - m_3)] *
                     _Factlist[int(j_3 + m_3)]) / \
        _Factlist[int(j_1 + j_2 + j_3 + 1)]

    # 计算平方根
    ressqrt = sqrt(argsqrt)
    # 如果平方根是复数或无穷大，取其实部
    if ressqrt.is_complex or ressqrt.is_infinite:
        ressqrt = ressqrt.as_real_imag()[0]

    # 计算 imin 和 imax 的值
    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0
    # 计算 sumres 的值，这是一个累加求和的过程
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * \
            _Factlist[int(ii + j_3 - j_1 - m_2)] * \
            _Factlist[int(j_2 + m_2 - ii)] * \
            _Factlist[int(j_1 - ii - m_1)] * \
            _Factlist[int(ii + j_3 - j_2 + m_1)] * \
            _Factlist[int(j_1 + j_2 - j_3 - ii)]
        sumres = sumres + Integer((-1) ** ii) / den

    # 计算 prefid 的值
    prefid = Integer((-1) ** int(j_1 - j_2 - m_3))
    # 计算结果 res
    res = ressqrt * sumres * prefid
    return res
# 计算 Clebsch-Gordan 系数 `\left\langle j_1 m_1 \; j_2 m_2 | j_3 m_3 \right\rangle`
# 参考文献为 [Edmonds74]_
def clebsch_gordan(j_1, j_2, j_3, m_1, m_2, m_3):
    # 计算 Clebsch-Gordan 系数的表达式
    res = (-1) ** sympify(j_1 - j_2 + m_3) * sqrt(2 * j_3 + 1) * \
        wigner_3j(j_1, j_2, j_3, m_1, m_2, -m_3)
    # 返回计算结果
    return res


# 计算 Racah 符号的 3 个角动量的 Delta 系数
# 同时检查差值是否为整数
def _big_delta_coeff(aa, bb, cc, prec=None):
    # 如果角动量不满足三角形关系，抛出 ValueError
    if not int_valued(aa + bb - cc):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if not int_valued(aa + cc - bb):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if not int_valued(bb + cc - aa):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    # 如果三个角动量之和中有小于零的情况，返回 0
    if (aa + bb - cc) < 0:
        return S.Zero
    if (aa + cc - bb) < 0:
        return S.Zero
    if (bb + cc - aa) < 0:
        return S.Zero
    
    # 计算最大因子，并调用 _calc_factlist 函数
    maxfact = max(aa + bb - cc, aa + cc - bb, bb + cc - aa, aa + bb + cc + 1)
    _calc_factlist(maxfact)
    # 计算阶乘并求解平方根的参数 argsqrt
    argsqrt = Integer(_Factlist[int(aa + bb - cc)] *
                     _Factlist[int(aa + cc - bb)] *
                     _Factlist[int(bb + cc - aa)]) / \
        Integer(_Factlist[int(aa + bb + cc + 1)])

    # 计算 argsqrt 的平方根
    ressqrt = sqrt(argsqrt)

    # 如果指定了精度 prec，则对结果进行数值计算并返回实部
    if prec:
        ressqrt = ressqrt.evalf(prec).as_real_imag()[0]

    # 返回计算结果的平方根或数值计算结果的实部
    return ressqrt
def racah(aa, bb, cc, dd, ee, ff, prec=None):
    r"""
    Calculate the Racah symbol `W(a,b,c,d;e,f)`.

    Parameters
    ==========

    a, ..., f :
        Integer or half integer.
    prec :
        Precision, default: ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import racah
    >>> racah(3,3,3,3,3,3)
    -1/14

    Notes
    =====

    The Racah symbol is related to the Wigner 6j symbol:

    .. math::

       \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
       =(-1)^{j_1+j_2+j_4+j_5} W(j_1,j_2,j_5,j_4,j_3,j_6)

    Please see the 6j symbol for its much richer symmetries and for
    additional properties.

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 6j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """
    # 计算 Racah 符号的前导系数，基于给定的参数和精度
    prefac = _big_delta_coeff(aa, bb, ee, prec) * \
        _big_delta_coeff(cc, dd, ee, prec) * \
        _big_delta_coeff(aa, cc, ff, prec) * \
        _big_delta_coeff(bb, dd, ff, prec)
    
    # 若前导系数为零，直接返回零
    if prefac == 0:
        return S.Zero
    
    # 确定求和的下界和上界
    imin = max(aa + bb + ee, cc + dd + ee, aa + cc + ff, bb + dd + ff)
    imax = min(aa + bb + cc + dd, aa + dd + ee + ff, bb + cc + ee + ff)

    # 计算需要预先计算的阶乘表
    maxfact = max(imax + 1, aa + bb + cc + dd, aa + dd + ee + ff,
                 bb + cc + ee + ff)
    _calc_factlist(maxfact)

    # 初始化求和结果
    sumres = 0
    # 对求和范围内的每一个 kk 进行循环求和
    for kk in range(int(imin), int(imax) + 1):
        # 计算求和项的分母
        den = _Factlist[int(kk - aa - bb - ee)] * \
            _Factlist[int(kk - cc - dd - ee)] * \
            _Factlist[int(kk - aa - cc - ff)] * \
            _Factlist[int(kk - bb - dd - ff)] * \
            _Factlist[int(aa + bb + cc + dd - kk)] * \
            _Factlist[int(aa + dd + ee + ff - kk)] * \
            _Factlist[int(bb + cc + ee + ff - kk)]
        
        # 计算并累加求和项
        sumres = sumres + Integer((-1) ** kk * _Factlist[kk + 1]) / den

    # 计算最终的 Racah 符号值
    res = prefac * sumres * (-1) ** int(aa + bb + cc + dd)
    return res


def wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6, prec=None):
    r"""
    Calculate the Wigner 6j symbol `\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)`.

    Parameters
    ==========

    j_1, ..., j_6 :
        Integer or half integer.
    prec :
        Precision, default: ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_6j
    >>> wigner_6j(3,3,3,3,3,3)
    ```
    # 计算 Wigner 6j 符号的值
    >>> wigner_6j(5,5,5,5,5,5)
    # 返回计算结果的分数形式
    1/52

    # 当传递的参数不是整数或半整数值，或者不满足三角关系时，会引发错误
    # 示例中展示了两个错误情况：

    # 抛出 ValueError，要求 j 值必须是整数或半整数，并且满足三角关系
    sage: wigner_6j(2.5,2.5,2.5,2.5,2.5,2.5)
    Traceback (most recent call last):
    ...
    ValueError: j values must be integer or half integer and fulfill the triangle relation
    # 抛出 ValueError，要求 j 值必须是整数或半整数，并且满足三角关系
    sage: wigner_6j(0.5,0.5,1.1,0.5,0.5,1.1)
    Traceback (most recent call last):
    ...
    ValueError: j values must be integer or half integer and fulfill the triangle relation

    # Wigner 6j 符号与 Racah 符号相关，但具有更多的对称性，如下详细说明
    #
    # Wigner 6j 符号服从以下对称性规则：
    #
    # - 在任意列的排列变换下保持不变
    #
    # - 在任意两列中上下参数交换后保持不变
    #
    # - 还有其他 6 种对称性 [Regge59]_ 共 144 种对称性
    #
    # - 只有当任意三个 `j` 的值满足三角关系时才非零

    # 算法
    # =========
    #
    # 此函数使用 [Edmonds74]_ 的算法精确计算 6j 符号的值。注意，公式中涉及
    # 大数阶乘的交替求和，因此不适合有限精度算术，仅适用于计算机代数系统 [Rasch03]_。

    """
    # 计算 Wigner 6j 符号的值，根据其定义使用 Racah 符号计算
    res = (-1) ** int(j_1 + j_2 + j_4 + j_5) * \
        racah(j_1, j_2, j_5, j_4, j_3, j_6, prec)
    # 返回计算结果
    return res
def wigner_9j(j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9, prec=None):
    r"""
    Calculate the Wigner 9j symbol
    `\operatorname{Wigner9j}(j_1,j_2,j_3,j_4,j_5,j_6,j_7,j_8,j_9)`.

    Parameters
    ==========

    j_1, ..., j_9 :
        Integer or half integer.
    prec : precision, default
        ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_9j
    >>> wigner_9j(1,1,1, 1,1,1, 1,1,0, prec=64)
    0.05555555555555555555555555555555555555555555555555555555555555555

    >>> wigner_9j(1/2,1/2,0, 1/2,3/2,1, 0,1,1, prec=64)
    0.1666666666666666666666666666666666666666666666666666666666666667

    It is an error to have arguments that are not integer or half
    integer values or do not fulfill the triangle relation::

        sage: wigner_9j(0.5,0.5,0.5, 0.5,0.5,0.5, 0.5,0.5,0.5,prec=64)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation
        sage: wigner_9j(1,1,1, 0.5,1,1.5, 0.5,1,2.5,prec=64)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.
    """
    # Determine the maximum value for the loop range
    imax = int(min(j_1 + j_9, j_2 + j_6, j_4 + j_8) * 2)
    # Ensure imin is even to match the loop iteration requirement
    imin = imax % 2
    # Initialize the sum variable
    sumres = 0
    # Iterate through even values starting from imin to imax
    for kk in range(imin, int(imax) + 1, 2):
        # Compute the contribution to the sum using Racah functions
        sumres = sumres + (kk + 1) * \
            racah(j_1, j_2, j_9, j_6, j_3, kk / 2, prec) * \
            racah(j_4, j_6, j_8, j_2, j_5, kk / 2, prec) * \
            racah(j_1, j_4, j_9, j_8, j_7, kk / 2, prec)
    # Return the accumulated sum as the result of the Wigner 9j symbol calculation
    return sumres


def gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    r"""
    Calculate the Gaunt coefficient.

    Explanation
    ===========

    The Gaunt coefficient is defined as the integral over three
    spherical harmonics:

    .. math::

        \begin{aligned}
        \operatorname{Gaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\int Y_{l_1,m_1}(\Omega)
         Y_{l_2,m_2}(\Omega) Y_{l_3,m_3}(\Omega) \,d\Omega \\
        &=\sqrt{\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\pi}}
         \operatorname{Wigner3j}(l_1,l_2,l_3,0,0,0)
         \operatorname{Wigner3j}(l_1,l_2,l_3,m_1,m_2,m_3)
        \end{aligned}

    Parameters
    ==========

    l_1, l_2, l_3, m_1, m_2, m_3 :
        Integer.
    prec - precision, default: ``None``.
        Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    The Gaunt coefficient as a floating-point number.

    """
    # 导入必要的函数和模块
    from sympy.physics.wigner import gaunt
    
    # 定义计算 Gaunt 系数的函数，输入参数为三个角动量量子数和三个角动量投影量子数
    def gaunt(l_1, l_2, l_3, m_1, m_2, m_3):
        # 将输入的角动量量子数和角动量投影量子数转换为整数
        l_1, l_2, l_3, m_1, m_2, m_3 = [
            as_int(i) for i in (l_1, l_2, l_3, m_1, m_2, m_3)]
    
        # 检查是否满足三角不等式的条件，如果不满足返回零
        if l_1 + l_2 - l_3 < 0:
            return S.Zero
        if l_1 - l_2 + l_3 < 0:
            return S.Zero
        if -l_1 + l_2 + l_3 < 0:
            return S.Zero
        # 检查角动量投影量子数之和是否为零，如果不是返回零
        if (m_1 + m_2 + m_3) != 0:
            return S.Zero
        # 检查是否满足角动量投影量子数的限制条件，如果不满足返回零
        if (abs(m_1) > l_1) or (abs(m_2) > l_2) or (abs(m_3) > l_3):
            return S.Zero
        # 计算三个角动量量子数之和的一半，并检查是否为偶数，如果不是返回零
        bigL, remL = divmod(l_1 + l_2 + l_3, 2)
        if remL % 2:
            return S.Zero
    
        # 计算循环求和的下界和上界
        imin = max(-l_3 + l_1 + m_2, -l_3 + l_2 - m_1, 0)
        imax = min(l_2 + m_2, l_1 - m_1, l_1 + l_2 - l_3)
    
        # 计算阶乘的列表，参数为三个角动量量子数之和和上界加一的较大者
        _calc_factlist(max(l_1 + l_2 + l_3 + 1, imax + 1))
    
    
    这段代码是用于计算Gaunt系数的函数，它用于量子力学中描述角动量的耦合关系。
    # 计算并返回三个整数和平方根的乘积
    ressqrt = sqrt((2 * l_1 + 1) * (2 * l_2 + 1) * (2 * l_3 + 1) * \
        _Factlist[l_1 - m_1] * _Factlist[l_1 + m_1] * _Factlist[l_2 - m_2] * \
        _Factlist[l_2 + m_2] * _Factlist[l_3 - m_3] * _Factlist[l_3 + m_3] / \
        (4*pi))
    
    # 计算比例因子 prefac，分子为大 L 和三个 l 的阶乘乘积，分母为相关阶乘的乘积
    prefac = Integer(_Factlist[bigL] * _Factlist[l_2 - l_1 + l_3] *
                     _Factlist[l_1 - l_2 + l_3] * _Factlist[l_1 + l_2 - l_3])/ \
        _Factlist[2 * bigL + 1]/ \
        (_Factlist[bigL - l_1] *
         _Factlist[bigL - l_2] * _Factlist[bigL - l_3])
    
    # 初始化 sumres 为零，计算 sumres 的值
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        # 计算 sumres 中的分母 den，使用相关的阶乘值
        den = _Factlist[ii] * _Factlist[ii + l_3 - l_1 - m_2] * \
            _Factlist[l_2 + m_2 - ii] * _Factlist[l_1 - ii - m_1] * \
            _Factlist[ii + l_3 - l_2 + m_1] * _Factlist[l_1 + l_2 - l_3 - ii]
        # 计算 sumres 的累加值，根据 (-1) 的幂次决定加减
        sumres = sumres + Integer((-1) ** ii) / den
    
    # 计算结果 res，包括 ressqrt、prefac、sumres 和额外的 (-1) 的幂次
    res = ressqrt * prefac * sumres * Integer((-1) ** (bigL + l_3 + m_1 - m_2))
    
    # 如果指定了精度 prec，则将 res 转换为数值并返回，否则直接返回 res
    if prec is not None:
        res = res.n(prec)
    return res
def real_gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    r"""
    Calculate the real Gaunt coefficient.

    Explanation
    ===========

    The real Gaunt coefficient is defined as the integral over three
    real spherical harmonics:

    .. math::
        \begin{aligned}
        \operatorname{RealGaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\int Z^{m_1}_{l_1}(\Omega)
         Z^{m_2}_{l_2}(\Omega) Z^{m_3}_{l_3}(\Omega) \,d\Omega \\
        \end{aligned}

    Alternatively, it can be defined in terms of the standard Gaunt
    coefficient by relating the real spherical harmonics to the standard
    spherical harmonics via a unitary transformation `U`, i.e.
    `Z^{m}_{l}(\Omega)=\sum_{m'}U^{m}_{m'}Y^{m'}_{l}(\Omega)` [Homeier96]_.
    The real Gaunt coefficient is then defined as

    .. math::
        \begin{aligned}
        \operatorname{RealGaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\int Z^{m_1}_{l_1}(\Omega)
         Z^{m_2}_{l_2}(\Omega) Z^{m_3}_{l_3}(\Omega) \,d\Omega \\
        &=\sum_{m'_1 m'_2 m'_3} U^{m_1}_{m'_1}U^{m_2}_{m'_2}U^{m_3}_{m'_3}
         \operatorname{Gaunt}(l_1,l_2,l_3,m'_1,m'_2,m'_3)
        \end{aligned}

    The unitary matrix `U` has components

    .. math::
        \begin{aligned}
        U^m_{m'} = \delta_{|m||m'|}*(\delta_{m'0}\delta_{m0} + \frac{1}{\sqrt{2}}\big[\Theta(m)
        \big(\delta_{m'm}+(-1)^{m'}\delta_{m'-m}\big)+i\Theta(-m)\big((-1)^{-m}
        \delta_{m'-m}-\delta_{m'm}*(-1)^{m'-m}\big)\big])
        \end{aligned}

    where `\delta_{ij}` is the Kronecker delta symbol and `\Theta` is a step
    function defined as

    .. math::
        \begin{aligned}
        \Theta(x) = \begin{cases} 1 \,\text{for}\, x > 0 \\ 0 \,\text{for}\, x \leq 0 \end{cases}
        \end{aligned}

    Parameters
    ==========

    l_1, l_2, l_3, m_1, m_2, m_3 :
        Integers representing the angular momentum quantum numbers.

    prec : Optional[float]
        Precision parameter, if provided, can improve calculation speed.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy.physics.wigner import real_gaunt
    >>> real_gaunt(2,2,4,-1,-1,0)
    -2/(7*sqrt(pi))
    >>> real_gaunt(10,10,20,-9,-9,0).n(64)
    -0.00002480019791932209313156167176797577821140084216297395518482071448

    It is an error to use non-integer values for `l` and `m`::
        real_gaunt(2.8,0.5,1.3,0,0,0)
        Traceback (most recent call last):
        ...
        ValueError: l values must be integer
        real_gaunt(2,2,4,0.7,1,-3.4)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer

    Notes
    =====

    The real Gaunt coefficient inherits from the standard Gaunt coefficient,
    the invariance under any permutation of the pairs `(l_i, m_i)` and the
    requirement that the sum of the `l_i` be even to yield a non-zero value.
    It also obeys the following symmetry rules:

    """
    # 将输入的整数参数转换为整数类型，确保参数是整数
    l_1, l_2, l_3, m_1, m_2, m_3 = [
        as_int(i) for i in (l_1, l_2, l_3, m_1, m_2, m_3)]
    
    # 快速检查退出条件
    # 如果 m_1, m_2, m_3 中负数的个数为奇数，返回零
    if sum(1 for i in (m_1, m_2, m_3) if i < 0) % 2:
        return S.Zero
    
    # 如果 l_1 + l_2 + l_3 是奇数，返回零
    if (l_1 + l_2 + l_3) % 2:
        return S.Zero
    
    # 计算 l_max = l_2 + l_3
    lmax = l_2 + l_3
    
    # 计算 l_min = max(|l_2 - l_3|, min(|m_2 + m_3|, |m_2 - m_3|))
    lmin = max(abs(l_2 - l_3), min(abs(m_2 + m_3), abs(m_2 - m_3)))
    
    # 如果 l_min + l_max 是奇数，则将 l_min 增加 1
    if (lmin + lmax) % 2:
        lmin += 1
    
    # 如果 l_min 不在范围内 [l_max, l_min - 2, ..., l_min] 中，则返回零
    if lmin not in range(lmax, lmin - 2, -2):
        return S.Zero
    
    # 定义 Kronecker δ 函数
    kron_del = lambda i, j: 1 if i == j else 0
    
    # 定义 (-1)**e 的函数，用来避免在 e < 0 时产生浮点数，保证返回 +/-1
    s = lambda e: -1 if e % 2 else 1
    
    # 定义 A 函数
    A = lambda a, b: (-kron_del(a, b)*s(a-b) + kron_del(a, -b)*s(b)) if b < 0 else 0
    
    # 定义 B 函数
    B = lambda a, b: (kron_del(a, b) + kron_del(a, -b)*s(a)) if b > 0 else 0
    
    # 定义 C 函数
    C = lambda a, b: kron_del(abs(a), abs(b))*(kron_del(a, 0)*kron_del(b, 0) +
                                          (B(a, b) + I*A(a, b))/sqrt(2))
    
    # 初始化 ugnt 变量
    ugnt = 0
    
    # 嵌套循环计算 Gaunt 系数的值
    for i in range(-l_1, l_1+1):
        U1 = C(i, m_1)
        for j in range(-l_2, l_2+1):
            U2 = C(j, m_2)
            U3 = C(-i-j, m_3)
            # 计算并累加实部乘积与 Gaunt 系数的乘积
            ugnt = ugnt + re(U1*U2*U3)*gaunt(l_1, l_2, l_3, i, j, -i-j)
    
    # 如果指定了精度 prec，则将结果 ugnt 转换为数值精确到 prec 位小数
    if prec is not None:
        ugnt = ugnt.n(prec)
    
    # 返回计算得到的 Gaunt 系数值
    return ugnt
class Wigner3j(Function):
    # 定义 Wigner3j 类，继承自 Function 类

    def doit(self, **hints):
        # 如果所有参数都是数值类型
        if all(obj.is_number for obj in self.args):
            # 调用 wigner_3j 函数计算 Wigner 3j 符号
            return wigner_3j(*self.args)
        else:
            # 如果参数不全为数值，则返回自身
            return self

def dot_rot_grad_Ynm(j, p, l, m, theta, phi):
    r"""
    Returns dot product of rotational gradients of spherical harmonics.

    Explanation
    ===========

    This function returns the right hand side of the following expression:

    .. math ::
        \vec{R}Y{_j^{p}} \cdot \vec{R}Y{_l^{m}} = (-1)^{m+p}
        \sum\limits_{k=|l-j|}^{l+j}Y{_k^{m+p}}  * \alpha_{l,m,j,p,k} *
        \frac{1}{2} (k^2-j^2-l^2+k-j-l)


    Arguments
    =========

    j, p, l, m .... indices in spherical harmonics (expressions or integers)
    theta, phi .... angle arguments in spherical harmonics

    Example
    =======

    >>> from sympy import symbols
    >>> from sympy.physics.wigner import dot_rot_grad_Ynm
    >>> theta, phi = symbols("theta phi")
    >>> dot_rot_grad_Ynm(3, 2, 2, 0, theta, phi).doit()
    3*sqrt(55)*Ynm(5, 2, theta, phi)/(11*sqrt(pi))

    """
    # 将输入的 j, p, l, m, theta, phi 转换为 SymPy 的符号对象
    j = sympify(j)
    p = sympify(p)
    l = sympify(l)
    m = sympify(m)
    theta = sympify(theta)
    phi = sympify(phi)
    k = Dummy("k")  # 创建虚拟变量 k

    def alpha(l,m,j,p,k):
        # 定义 alpha 函数，计算旋转梯度球谐函数的系数 alpha
        return sqrt((2*l+1)*(2*j+1)*(2*k+1)/(4*pi)) * \
                Wigner3j(j, l, k, S.Zero, S.Zero, S.Zero) * \
                Wigner3j(j, l, k, p, m, -m-p)

    # 返回右手边的表达式，即旋转梯度球谐函数的点积
    return (S.NegativeOne)**(m+p) * Sum(Ynm(k, m+p, theta, phi) * alpha(l,m,j,p,k) / 2 \
        *(k**2-j**2-l**2+k-j-l), (k, abs(l-j), l+j))


def wigner_d_small(J, beta):
    """Return the small Wigner d matrix for angular momentum J.

    Explanation
    ===========

    J : An integer, half-integer, or SymPy symbol for the total angular
        momentum of the angular momentum space being rotated.
    beta : A real number representing the Euler angle of rotation about
        the so-called line of nodes. See [Edmonds74]_.

    Returns
    =======

    A matrix representing the corresponding Euler angle rotation( in the basis
    of eigenvectors of `J_z`).

    .. math ::
        \\mathcal{d}_{\\beta} = \\exp\\big( \\frac{i\\beta}{\\hbar} J_y\\big)

    The components are calculated using the general form [Edmonds74]_,
    equation 4.1.15.

    Examples
    ========

    >>> from sympy import Integer, symbols, pi, pprint
    >>> from sympy.physics.wigner import wigner_d_small
    >>> half = 1/Integer(2)
    >>> beta = symbols("beta", real=True)
    >>> pprint(wigner_d_small(half, beta), use_unicode=True)
    ⎡   ⎛β⎞      ⎛β⎞⎤
    ⎢cos⎜─⎟   sin⎜─⎟⎥
    ⎢   ⎝2⎠      ⎝2⎠⎥
    ⎢               ⎥
    ⎢    ⎛β⎞     ⎛β⎞⎥
    ⎢-sin⎜─⎟  cos⎜─⎟⎥
    ⎣    ⎝2⎠     ⎝2⎠⎦

    >>> pprint(wigner_d_small(2*half, beta), use_unicode=True)
    ⎡        2⎛β⎞              ⎛β⎞    ⎛β⎞           2⎛β⎞     ⎤
    ⎢     cos ⎜─⎟        √2⋅sin⎜─⎟⋅cos⎜─⎟        sin ⎜─⎟     ⎥
    ⎢         ⎝2⎠              ⎝2⎠    ⎝2⎞            ⎝2⎠     ⎥
    ⎢                                                        ⎥
    """
        M = [J-i for i in range(2*J+1)]
        d = zeros(2*J+1)
        for i, Mi in enumerate(M):
            for j, Mj in enumerate(M):
    
                # 计算 sigma 的最大和最小值
                sigmamax = min([J-Mi, J-Mj])
                sigmamin = max([0, -Mi-Mj])
    
                # 计算 d[i, j] 的值
                dij = sqrt(factorial(J+Mi)*factorial(J-Mi) /
                           factorial(J+Mj)/factorial(J-Mj))
                
                # 计算 d[i, j] 中的每一项
                terms = [(-1)**(J-Mi-s) *
                         binomial(J+Mj, J-Mi-s) *
                         binomial(J-Mj, s) *
                         cos(beta/2)**(2*s+Mi+Mj) *
                         sin(beta/2)**(2*J-2*s-Mj-Mi)
                         for s in range(sigmamin, sigmamax+1)]
    
                # 将所有项加和得到 d[i, j]
                d[i, j] = dij * Add(*terms)
    
        return ImmutableMatrix(d)
    """
# 定义函数 wigner_d，返回角动量 J 对应的 Wigner D 矩阵

def wigner_d(J, alpha, beta, gamma):
    # 导入 wigner_d_small 函数，用于计算小 Wigner D 矩阵
    d = wigner_d_small(J, beta)
    
    # 构造角动量空间中 J_z 的所有可能取值
    M = [J-i for i in range(2*J+1)]
    
    # 计算 Wigner D 矩阵的每个元素，使用复数指数函数 exp 表示旋转
    D = [[exp(I*Mi*alpha)*d[i, j]*exp(I*Mj*gamma)
          for j, Mj in enumerate(M)] for i, Mi in enumerate(M)]
    
    # 将结果构造成不可变的矩阵并返回
    return ImmutableMatrix(D)
```