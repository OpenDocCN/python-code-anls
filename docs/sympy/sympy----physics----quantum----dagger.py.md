# `D:\src\scipysrc\sympy\sympy\physics\quantum\dagger.py`

```
"""Hermitian conjugation."""

# 导入所需的类和函数
from sympy.core import Expr, Mul, sympify
from sympy.functions.elementary.complexes import adjoint

# 将 'Dagger' 加入到 __all__ 列表中，表明它是模块的公开接口
__all__ = [
    'Dagger'
]

# 定义类 Dagger，继承自 adjoint 类
class Dagger(adjoint):
    """General Hermitian conjugate operation.

    Explanation
    ===========

    Take the Hermetian conjugate of an argument [1]_. For matrices this
    operation is equivalent to transpose and complex conjugate [2]_.

    Parameters
    ==========

    arg : Expr
        The SymPy expression that we want to take the dagger of.
    evaluate : bool
        Whether the resulting expression should be directly evaluated.

    Examples
    ========

    Daggering various quantum objects:

        >>> from sympy.physics.quantum.dagger import Dagger
        >>> from sympy.physics.quantum.state import Ket, Bra
        >>> from sympy.physics.quantum.operator import Operator
        >>> Dagger(Ket('psi'))
        <psi|
        >>> Dagger(Bra('phi'))
        |phi>
        >>> Dagger(Operator('A'))
        Dagger(A)

    Inner and outer products::

        >>> from sympy.physics.quantum import InnerProduct, OuterProduct
        >>> Dagger(InnerProduct(Bra('a'), Ket('b')))
        <b|a>
        >>> Dagger(OuterProduct(Ket('a'), Bra('b')))
        |b><a|

    Powers, sums and products::

        >>> A = Operator('A')
        >>> B = Operator('B')
        >>> Dagger(A*B)
        Dagger(B)*Dagger(A)
        >>> Dagger(A+B)
        Dagger(A) + Dagger(B)
        >>> Dagger(A**2)
        Dagger(A)**2

    Dagger also seamlessly handles complex numbers and matrices::

        >>> from sympy import Matrix, I
        >>> m = Matrix([[1,I],[2,I]])
        >>> m
        Matrix([
        [1, I],
        [2, I]])
        >>> Dagger(m)
        Matrix([
        [ 1,  2],
        [-I, -I]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermitian_adjoint
    .. [2] https://en.wikipedia.org/wiki/Hermitian_transpose
    """

    # 构造函数，创建新的 Dagger 对象
    def __new__(cls, arg, evaluate=True):
        # 如果参数 arg 有 adjoint 属性且 evaluate 为真，则调用其 adjoint 方法
        if hasattr(arg, 'adjoint') and evaluate:
            return arg.adjoint()
        # 如果参数 arg 同时具有 conjugate 和 transpose 属性且 evaluate 为真，则先进行共轭再转置
        elif hasattr(arg, 'conjugate') and hasattr(arg, 'transpose') and evaluate:
            return arg.conjugate().transpose()
        # 否则，调用父类 Expr 的构造函数来创建一个新的表达式对象
        return Expr.__new__(cls, sympify(arg))

    # 定义乘法运算，处理与 IdentityOperator 的特殊情况
    def __mul__(self, other):
        from sympy.physics.quantum import IdentityOperator
        # 如果 other 是 IdentityOperator，则返回 self
        if isinstance(other, IdentityOperator):
            return self
        # 否则返回 self 与 other 的乘积
        return Mul(self, other)

# 将 adjoint 类的一些属性重新定义为适合 Dagger 的值
adjoint.__name__ = "Dagger"
adjoint._sympyrepr = lambda a, b: "Dagger(%s)" % b._print(a.args[0])
```