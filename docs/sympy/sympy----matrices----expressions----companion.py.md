# `D:\src\scipysrc\sympy\sympy\matrices\expressions\companion.py`

```
# 导入必要的符号计算库中的类和函数
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.polys.polytools import Poly

# 导入矩阵表达式的基类
from .matexpr import MatrixExpr

# CompanionMatrix 类，表示一个多项式的符号伴随矩阵
class CompanionMatrix(MatrixExpr):
    """A symbolic companion matrix of a polynomial.

    Examples
    ========

    >>> from sympy import Poly, Symbol, symbols
    >>> from sympy.matrices.expressions import CompanionMatrix
    >>> x = Symbol('x')
    >>> c0, c1, c2, c3, c4 = symbols('c0:5')
    >>> p = Poly(c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + x**5, x)
    >>> CompanionMatrix(p)
    CompanionMatrix(Poly(x**5 + c4*x**4 + c3*x**3 + c2*x**2 + c1*x + c0,
    x, domain='ZZ[c0,c1,c2,c3,c4]'))
    """

    # __new__ 方法用于创建 CompanionMatrix 的新实例
    def __new__(cls, poly):
        # 将输入的 poly 转化为符号表达式
        poly = _sympify(poly)
        # 检查 poly 是否为多项式类型
        if not isinstance(poly, Poly):
            raise ValueError("{} must be a Poly instance.".format(poly))
        # 检查 poly 是否为首一多项式
        if not poly.is_monic:
            raise ValueError("{} must be a monic polynomial.".format(poly))
        # 检查 poly 是否为一元多项式
        if not poly.is_univariate:
            raise ValueError(
                "{} must be a univariate polynomial.".format(poly))
        # 检查 poly 的次数是否至少为1
        if not poly.degree() >= 1:
            raise ValueError(
                "{} must have degree not less than 1.".format(poly))

        # 调用父类的 __new__ 方法创建实例
        return super().__new__(cls, poly)

    # shape 属性用于获取 CompanionMatrix 的形状信息
    @property
    def shape(self):
        # 获取多项式对象
        poly = self.args[0]
        # 矩阵的大小等于多项式的次数
        size = poly.degree()
        return size, size

    # _entry 方法用于计算 CompanionMatrix 的每个元素值
    def _entry(self, i, j):
        # 如果 j 是最后一列，则返回多项式的负常数系数
        if j == self.cols - 1:
            return -self.args[0].all_coeffs()[-1 - i]
        # 如果 i 在 j 的下一行，则返回 1
        elif i == j + 1:
            return S.One
        # 其他情况返回 0
        return S.Zero

    # as_explicit 方法将 CompanionMatrix 转化为其明显形式的矩阵
    def as_explicit(self):
        # 导入不可变密集矩阵类
        from sympy.matrices.immutable import ImmutableDenseMatrix
        # 调用 companion 方法生成明显形式的 CompanionMatrix
        return ImmutableDenseMatrix.companion(self.args[0])
```