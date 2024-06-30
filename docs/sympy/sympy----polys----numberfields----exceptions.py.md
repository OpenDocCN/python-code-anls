# `D:\src\scipysrc\sympy\sympy\polys\numberfields\exceptions.py`

```
# 特殊的异常类，用于处理数域相关的异常情况

class ClosureFailure(Exception):
    r"""
    信号表明尝试在某个特定的 :py:class:`Module` 中表示 :py:class:`ModuleElement` 时失败了。

    Examples
    ========

    >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.numberfields.modules import PowerBasis, to_col
    >>> T = Poly(cyclotomic_poly(5))
    >>> A = PowerBasis(T)
    >>> B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))

    因为我们处于一个旋转多项式域中，幂基础 ``A`` 是一个整数基础，而子模块 ``B`` 只是理想 $(2)$。因此 ``B`` 能够表示一个在幂基础上所有系数都是偶数的元素：

    >>> a1 = A(to_col([2, 4, 6, 8]))
    >>> print(B.represent(a1))
    DomainMatrix([[1], [2], [3], [4]], (4, 1), ZZ)

    但是 ``B`` 无法表示一个具有奇数系数的元素：

    >>> a2 = A(to_col([1, 2, 2, 2]))
    >>> B.represent(a2)
    Traceback (most recent call last):
    ...
    ClosureFailure: Element in QQ-span but not ZZ-span of this basis.

    """

class StructureError(Exception):
    r"""
    表示期望某个代数结构具有某种属性或某种类型，但实际情况并非如此。
    """
    pass


class MissingUnityError(StructureError):
    r"""结构应包含一个单位元素，但未包含。"""
    pass


__all__ = [
    'ClosureFailure', 'StructureError', 'MissingUnityError',
]
```