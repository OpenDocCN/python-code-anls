# `D:\src\scipysrc\sympy\sympy\matrices\expressions\fourier.py`

```
from sympy.core.sympify import _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt

class DFT(MatrixExpr):
    r"""
    Returns a discrete Fourier transform matrix. The matrix is scaled
    with :math:`\frac{1}{\sqrt{n}}` so that it is unitary.

    Parameters
    ==========

    n : integer or Symbol
        Size of the transform.

    Examples
    ========

    >>> from sympy.abc import n
    >>> from sympy.matrices.expressions.fourier import DFT
    >>> DFT(3)
    DFT(3)
    >>> DFT(3).as_explicit()
    Matrix([
    [sqrt(3)/3,                sqrt(3)/3,                sqrt(3)/3],
    [sqrt(3)/3, sqrt(3)*exp(-2*I*pi/3)/3,  sqrt(3)*exp(2*I*pi/3)/3],
    [sqrt(3)/3,  sqrt(3)*exp(2*I*pi/3)/3, sqrt(3)*exp(-2*I*pi/3)/3]])
    >>> DFT(n).shape
    (n, n)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/DFT_matrix

    """

    def __new__(cls, n):
        n = _sympify(n)  # 将输入参数 n 转换为符号表达式
        cls._check_dim(n)  # 调用类方法 _check_dim 检查 n 的维度

        obj = super().__new__(cls, n)
        return obj

    n = property(lambda self: self.args[0])  # type: ignore
    shape = property(lambda self: (self.n, self.n))  # type: ignore

    def _entry(self, i, j, **kwargs):
        w = exp(-2*S.Pi*I/self.n)  # 计算 DFT 矩阵的单个元素
        return w**(i*j) / sqrt(self.n)

    def _eval_inverse(self):
        return IDFT(self.n)  # 返回 DFT 矩阵的逆矩阵


class IDFT(DFT):
    r"""
    Returns an inverse discrete Fourier transform matrix. The matrix is scaled
    with :math:`\frac{1}{\sqrt{n}}` so that it is unitary.

    Parameters
    ==========

    n : integer or Symbol
        Size of the transform

    Examples
    ========

    >>> from sympy.matrices.expressions.fourier import DFT, IDFT
    >>> IDFT(3)
    IDFT(3)
    >>> IDFT(4)*DFT(4)
    I

    See Also
    ========

    DFT

    """
    def _entry(self, i, j, **kwargs):
        w = exp(-2*S.Pi*I/self.n)  # 计算 IDFT 矩阵的单个元素
        return w**(-i*j) / sqrt(self.n)

    def _eval_inverse(self):
        return DFT(self.n)  # 返回 IDFT 矩阵的逆矩阵，即 DFT 矩阵
```