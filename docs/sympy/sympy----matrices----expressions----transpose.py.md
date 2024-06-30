# `D:\src\scipysrc\sympy\sympy\matrices\expressions\transpose.py`

```
from sympy.core.basic import Basic  # 导入 SymPy 核心基础类 Basic
from sympy.matrices.expressions.matexpr import MatrixExpr  # 导入 SymPy 矩阵表达式类 MatrixExpr


class Transpose(MatrixExpr):
    """
    The transpose of a matrix expression.

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the transpose, use the ``transpose()``
    function, or the ``.T`` attribute of matrices.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Transpose, transpose
    >>> A = MatrixSymbol('A', 3, 5)
    >>> B = MatrixSymbol('B', 5, 3)
    >>> Transpose(A)
    A.T
    >>> A.T == transpose(A) == Transpose(A)
    True
    >>> Transpose(A*B)
    (A*B).T
    >>> transpose(A*B)
    B.T*A.T

    """
    is_Transpose = True  # 设置 is_Transpose 属性为 True，表示这是一个转置操作对象

    def doit(self, **hints):
        arg = self.arg  # 获取转置对象的参数
        if hints.get('deep', True) and isinstance(arg, Basic):
            arg = arg.doit(**hints)  # 如果 hints 中指定了深度计算并且参数是 Basic 类型，则进行深度计算
        _eval_transpose = getattr(arg, '_eval_transpose', None)
        if _eval_transpose is not None:
            result = _eval_transpose()
            return result if result is not None else Transpose(arg)  # 尝试调用参数的 _eval_transpose 方法进行转置
        else:
            return Transpose(arg)  # 如果无法直接转置，则返回 Transpose 对象

    @property
    def arg(self):
        return self.args[0]  # 返回转置对象的第一个参数

    @property
    def shape(self):
        return self.arg.shape[::-1]  # 返回转置对象参数的形状的逆序

    def _entry(self, i, j, expand=False, **kwargs):
        return self.arg._entry(j, i, expand=expand, **kwargs)  # 返回转置后矩阵中 (i, j) 元素对应的元素

    def _eval_adjoint(self):
        return self.arg.conjugate()  # 返回转置对象参数的共轭转置

    def _eval_conjugate(self):
        return self.arg.adjoint()  # 返回转置对象参数的转置共轭

    def _eval_transpose(self):
        return self.arg  # 返回转置对象参数本身，表示转置操作

    def _eval_trace(self):
        from .trace import Trace
        return Trace(self.arg)  # 返回转置对象参数的迹

    def _eval_determinant(self):
        from sympy.matrices.expressions.determinant import det
        return det(self.arg)  # 返回转置对象参数的行列式值

    def _eval_derivative(self, x):
        # x is a scalar:
        return self.arg._eval_derivative(x)  # 对参数进行导数求解

    def _eval_derivative_matrix_lines(self, x):
        lines = self.args[0]._eval_derivative_matrix_lines(x)
        return [i.transpose() for i in lines]  # 对矩阵的行向量求导数后，返回其转置行向量


def transpose(expr):
    """Matrix transpose"""
    return Transpose(expr).doit(deep=False)  # 返回表达式的转置


from sympy.assumptions.ask import ask, Q  # 导入 SymPy 的假设模块中的 ask 和 Q 函数
from sympy.assumptions.refine import handlers_dict  # 导入 SymPy 的假设模块中的 handlers_dict


def refine_Transpose(expr, assumptions):
    """
    >>> from sympy import MatrixSymbol, Q, assuming, refine
    >>> X = MatrixSymbol('X', 2, 2)
    >>> X.T
    X.T
    >>> with assuming(Q.symmetric(X)):
    ...     print(refine(X.T))
    X
    """
    if ask(Q.symmetric(expr), assumptions):
        return expr.arg  # 如果表达式被假设为对称的，则返回其参数

    return expr  # 否则返回原始表达式

handlers_dict['Transpose'] = refine_Transpose  # 将 refine_Transpose 函数注册到 handlers_dict 中处理转置操作
```