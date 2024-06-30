# `D:\src\scipysrc\sympy\sympy\matrices\normalforms.py`

```
'''Functions returning normal forms of matrices'''

from sympy.polys.domains.integerring import ZZ  # 导入整数环 ZZ
from sympy.polys.polytools import Poly  # 导入多项式工具 Poly
from sympy.polys.matrices import DomainMatrix  # 导入域矩阵
from sympy.polys.matrices.normalforms import (
        smith_normal_form as _snf,  # 导入 Smith 正则形
        invariant_factors as _invf,  # 导入不变因子
        hermite_normal_form as _hnf,  # 导入 Hermite 正则形
    )


def _to_domain(m, domain=None):
    """Convert Matrix to DomainMatrix"""
    # XXX: deprecated support for RawMatrix:
    ring = getattr(m, "ring", None)  # 获取矩阵 m 的环属性
    m = m.applyfunc(lambda e: e.as_expr() if isinstance(e, Poly) else e)  # 将矩阵元素转换为表达式

    dM = DomainMatrix.from_Matrix(m)  # 从普通矩阵创建域矩阵

    domain = domain or ring
    if domain is not None:
        dM = dM.convert_to(domain)  # 将域矩阵转换为指定域 domain
    return dM  # 返回域矩阵


def smith_normal_form(m, domain=None):
    '''
    Return the Smith Normal Form of a matrix `m` over the ring `domain`.
    This will only work if the ring is a principal ideal domain.

    Examples
    ========

    >>> from sympy import Matrix, ZZ
    >>> from sympy.matrices.normalforms import smith_normal_form
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> print(smith_normal_form(m, domain=ZZ))
    Matrix([[1, 0, 0], [0, 10, 0], [0, 0, -30]])

    '''
    dM = _to_domain(m, domain)  # 将矩阵 m 转换为域矩阵
    return _snf(dM).to_Matrix()  # 返回域矩阵的 Smith 正则形转换为普通矩阵


def invariant_factors(m, domain=None):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    .. [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf

    '''
    dM = _to_domain(m, domain)  # 将矩阵 m 转换为域矩阵
    factors = _invf(dM)  # 计算域矩阵的不变因子
    factors = tuple(dM.domain.to_sympy(f) for f in factors)  # 将不变因子转换为 SymPy 符号
    # XXX: deprecated.
    if hasattr(m, "ring"):
        if m.ring.is_PolynomialRing:
            K = m.ring
            to_poly = lambda f: Poly(f, K.symbols, domain=K.domain)
            factors = tuple(to_poly(f) for f in factors)  # 如果矩阵 m 是多项式环，则将因子转换为多项式
    return factors  # 返回不变因子的元组


def hermite_normal_form(A, *, D=None, check_rank=False):
    r"""
    Compute the Hermite Normal Form of a Matrix *A* of integers.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.matrices.normalforms import hermite_normal_form
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> print(hermite_normal_form(m))
    Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]])

    Parameters
    ==========

    A : $m \times n$ ``Matrix`` of integers.

    D : int, optional
        Let $W$ be the HNF of *A*. If known in advance, a positive integer *D*
        being any multiple of $\det(W)$ may be provided. In this case, if *A*
        also has rank $m$, then we may use an alternative algorithm that works
        mod *D* in order to prevent coefficient explosion.


    '''
    # 参数 check_rank：布尔型，可选参数，默认为 False
    # 基本假设是，如果传入了 *D* 的值，则已经认为 *A* 的秩为 m，因此不会浪费时间再次检查。
    # 如果希望进行检查（并且在检查失败时使用普通的非模 *D* 算法），则将 *check_rank* 设置为 True。

    返回值
    ======

    ``Matrix``
        矩阵 *A* 的 Hermite 标准形（HNF）。

    异常
    ======

    DMDomainError
        如果矩阵的域不是 :ref:`ZZ`。

    DMShapeError
        如果使用模 *D* 算法但矩阵的行数大于列数。

    参考文献
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       （参见算法 2.4.5 和 2.4.8。）
    """
    # 接受 Python 的 int、SymPy 的 Integer 和 ZZ 本身的任何一种类型：
    if D is not None and not ZZ.of_type(D):
        D = ZZ(int(D))
    # 调用内部函数 _hnf，返回其结果并转换为 Matrix 类型
    return _hnf(A._rep, D=D, check_rank=check_rank).to_Matrix()
```