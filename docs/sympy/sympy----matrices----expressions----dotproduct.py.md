# `D:\src\scipysrc\sympy\sympy\matrices\expressions\dotproduct.py`

```
from sympy.core import Basic, Expr  # 导入基本的符号运算和表达式类
from sympy.core.sympify import _sympify  # 导入符号化函数
from sympy.matrices.expressions.transpose import transpose  # 导入转置函数

class DotProduct(Expr):
    """
    Dot product of vector matrices

    The input should be two 1 x n or n x 1 matrices. The output represents the
    scalar dotproduct.

    This is similar to using MatrixElement and MatMul, except DotProduct does
    not require that one vector to be a row vector and the other vector to be
    a column vector.

    >>> from sympy import MatrixSymbol, DotProduct
    >>> A = MatrixSymbol('A', 1, 3)
    >>> B = MatrixSymbol('B', 1, 3)
    >>> DotProduct(A, B)
    DotProduct(A, B)
    >>> DotProduct(A, B).doit()
    A[0, 0]*B[0, 0] + A[0, 1]*B[0, 1] + A[0, 2]*B[0, 2]
    """

    def __new__(cls, arg1, arg2):
        arg1, arg2 = _sympify((arg1, arg2))  # 将输入参数符号化

        if not arg1.is_Matrix:
            raise TypeError("Argument 1 of DotProduct is not a matrix")  # 抛出异常，如果参数 1 不是矩阵
        if not arg2.is_Matrix:
            raise TypeError("Argument 2 of DotProduct is not a matrix")  # 抛出异常，如果参数 2 不是矩阵
        if not (1 in arg1.shape):
            raise TypeError("Argument 1 of DotProduct is not a vector")  # 抛出异常，如果参数 1 不是向量
        if not (1 in arg2.shape):
            raise TypeError("Argument 2 of DotProduct is not a vector")  # 抛出异常，如果参数 2 不是向量

        if set(arg1.shape) != set(arg2.shape):
            raise TypeError("DotProduct arguments are not the same length")  # 抛出异常，如果参数长度不一致

        return Basic.__new__(cls, arg1, arg2)

    def doit(self, expand=False, **hints):
        if self.args[0].shape == self.args[1].shape:
            if self.args[0].shape[0] == 1:
                mul = self.args[0]*transpose(self.args[1])  # 如果参数是行向量，则计算其转置与列向量的乘积
            else:
                mul = transpose(self.args[0])*self.args[1]  # 如果参数是列向量，则计算其转置与行向量的乘积
        else:
            if self.args[0].shape[0] == 1:
                mul = self.args[0]*self.args[1]  # 如果参数是行向量，则直接计算其与列向量的乘积
            else:
                mul = transpose(self.args[0])*transpose(self.args[1])  # 如果参数是列向量，则计算其转置与转置的乘积

        return mul[0]  # 返回乘积的第一个元素，即标量结果
```