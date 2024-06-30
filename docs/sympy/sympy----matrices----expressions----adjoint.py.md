# `D:\src\scipysrc\sympy\sympy\matrices\expressions\adjoint.py`

```
from sympy.core import Basic
from sympy.functions import adjoint, conjugate
from sympy.matrices.expressions.matexpr import MatrixExpr

# 定义一个继承自 MatrixExpr 的类 Adjoint，表示一个矩阵表达式的共轭转置
class Adjoint(MatrixExpr):
    """
    The Hermitian adjoint of a matrix expression.

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the adjoint, use the ``adjoint()``
    function.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Adjoint, adjoint
    >>> A = MatrixSymbol('A', 3, 5)
    >>> B = MatrixSymbol('B', 5, 3)
    >>> Adjoint(A*B)
    Adjoint(A*B)
    >>> adjoint(A*B)
    Adjoint(B)*Adjoint(A)
    >>> adjoint(A*B) == Adjoint(A*B)
    False
    >>> adjoint(A*B) == Adjoint(A*B).doit()
    True
    """
    
    is_Adjoint = True  # 表示这是一个 Adjoint 类型的对象的标志

    # 执行对该对象的实际计算
    def doit(self, **hints):
        arg = self.arg
        if hints.get('deep', True) and isinstance(arg, Basic):
            return adjoint(arg.doit(**hints))  # 如果指定了深度计算，则递归调用参数的 doit 方法
        else:
            return adjoint(self.arg)  # 否则直接计算参数的共轭转置

    @property
    def arg(self):
        return self.args[0]  # 返回对象的第一个参数作为实际计算的对象

    @property
    def shape(self):
        return self.arg.shape[::-1]  # 返回参数对象的转置形状作为共轭转置后的形状

    # 计算共轭转置后的矩阵元素
    def _entry(self, i, j, **kwargs):
        return conjugate(self.arg._entry(j, i, **kwargs))

    # 返回对象自身的参数作为共轭转置的结果
    def _eval_adjoint(self):
        return self.arg

    # 返回对象参数的共轭转置结果
    def _eval_transpose(self):
        return self.arg.conjugate()

    # 返回对象参数的转置结果
    def _eval_conjugate(self):
        return self.arg.transpose()

    # 返回对象参数的迹的共轭结果
    def _eval_trace(self):
        from sympy.matrices.expressions.trace import Trace
        return conjugate(Trace(self.arg))
```