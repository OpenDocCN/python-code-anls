# `D:\src\scipysrc\sympy\sympy\matrices\expressions\inverse.py`

```
from sympy.core.sympify import _sympify  # 导入_sympify函数，用于将输入转换为符号表达式
from sympy.core import S, Basic  # 导入S和Basic类

from sympy.matrices.exceptions import NonSquareMatrixError  # 导入NonSquareMatrixError异常类
from sympy.matrices.expressions.matpow import MatPow  # 导入MatPow类


class Inverse(MatPow):
    """
    The multiplicative inverse of a matrix expression

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the inverse, use the ``.inverse()``
    method of matrices.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Inverse
    >>> A = MatrixSymbol('A', 3, 3)
    >>> B = MatrixSymbol('B', 3, 3)
    >>> Inverse(A)
    A**(-1)
    >>> A.inverse() == Inverse(A)
    True
    >>> (A*B).inverse()
    B**(-1)*A**(-1)
    >>> Inverse(A*B)
    (A*B)**(-1)

    """
    is_Inverse = True  # 类属性，表示这是一个Inverse对象
    exp = S.NegativeOne  # 类属性，指定默认的指数为-1

    def __new__(cls, mat, exp=S.NegativeOne):
        # 构造函数，创建Inverse对象
        # exp参数用于保持与inverse.func(*inverse.args) == inverse的一致性
        mat = _sympify(mat)  # 将mat符号化处理
        exp = _sympify(exp)  # 将exp符号化处理
        if not mat.is_Matrix:  # 检查mat是否为矩阵
            raise TypeError("mat should be a matrix")
        if mat.is_square is False:  # 检查mat是否为方阵
            raise NonSquareMatrixError("Inverse of non-square matrix %s" % mat)
        return Basic.__new__(cls, mat, exp)  # 调用基类Basic的构造函数创建对象

    @property
    def arg(self):
        return self.args[0]  # 返回Inverse对象的第一个参数

    @property
    def shape(self):
        return self.arg.shape  # 返回Inverse对象参数的形状信息

    def _eval_inverse(self):
        return self.arg  # 返回Inverse对象参数的逆矩阵

    def _eval_transpose(self):
        return Inverse(self.arg.transpose())  # 返回Inverse对象参数的转置矩阵的逆

    def _eval_adjoint(self):
        return Inverse(self.arg.adjoint())  # 返回Inverse对象参数的伴随矩阵的逆

    def _eval_conjugate(self):
        return Inverse(self.arg.conjugate())  # 返回Inverse对象参数的共轭矩阵的逆

    def _eval_determinant(self):
        from sympy.matrices.expressions.determinant import det
        return 1/det(self.arg)  # 返回Inverse对象参数的行列式的倒数作为结果

    def doit(self, **hints):
        if 'inv_expand' in hints and hints['inv_expand'] == False:
            return self  # 如果hints中inv_expand为False，直接返回自身

        arg = self.arg  # 获取Inverse对象的参数
        if hints.get('deep', True):  # 如果hints中的deep为True，进行深度处理
            arg = arg.doit(**hints)  # 对参数执行深度处理

        return arg.inverse()  # 返回参数的逆矩阵

    def _eval_derivative_matrix_lines(self, x):
        arg = self.args[0]  # 获取Inverse对象的参数
        lines = arg._eval_derivative_matrix_lines(x)  # 调用参数的求导矩阵行方法得到行列表
        for line in lines:
            line.first_pointer *= -self.T  # 调整行列表中每一行的第一个指针
            line.second_pointer *= self  # 调整行列表中每一行的第二个指针
        return lines  # 返回调整后的行列表


from sympy.assumptions.ask import ask, Q  # 导入ask函数和Q类
from sympy.assumptions.refine import handlers_dict  # 导入handlers_dict字典


def refine_Inverse(expr, assumptions):
    """
    >>> from sympy import MatrixSymbol, Q, assuming, refine
    >>> X = MatrixSymbol('X', 2, 2)
    >>> X.I
    X**(-1)
    >>> with assuming(Q.orthogonal(X)):
    ...     print(refine(X.I))
    X.T
    """
    if ask(Q.orthogonal(expr), assumptions):  # 如果expr符合正交条件
        return expr.arg.T  # 返回expr参数的转置矩阵
    elif ask(Q.unitary(expr), assumptions):  # 如果expr符合酉条件
        return expr.arg.conjugate()  # 返回expr参数的共轭矩阵
    elif ask(Q.singular(expr), assumptions):  # 如果expr符合奇异条件
        raise ValueError("Inverse of singular matrix %s" % expr.arg)  # 抛出奇异矩阵的异常

    return expr  # 返回未变化的expr参数

handlers_dict['Inverse'] = refine_Inverse  # 将refine_Inverse函数添加到handlers_dict字典中作为Inverse的处理函数
```