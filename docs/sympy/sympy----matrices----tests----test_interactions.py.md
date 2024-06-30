# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_interactions.py`

```
"""
We have a few different kind of Matrices
Matrix, ImmutableMatrix, MatrixExpr

Here we test the extent to which they cooperate
"""

# 导入必要的符号和矩阵类
from sympy.core.symbol import symbols
from sympy.matrices import (Matrix, MatrixSymbol, eye, Identity,
        ImmutableMatrix)
from sympy.matrices.expressions import MatrixExpr, MatAdd
from sympy.matrices.matrixbase import classof
from sympy.testing.pytest import raises

# 创建一个矩阵符号
SM = MatrixSymbol('X', 3, 3)
SV = MatrixSymbol('v', 3, 1)
# 创建普通矩阵和不可变矩阵
MM = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
IM = ImmutableMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 创建单位矩阵和不可变单位矩阵
meye = eye(3)
imeye = ImmutableMatrix(eye(3))
ideye = Identity(3)
a, b, c = symbols('a,b,c')


def test_IM_MM():
    # 测试不可变矩阵和普通矩阵的加法，结果应为不可变矩阵
    assert isinstance(MM + IM, ImmutableMatrix)
    assert isinstance(IM + MM, ImmutableMatrix)
    # 测试不可变矩阵和普通矩阵的数乘，结果应为不可变矩阵
    assert isinstance(2*IM + MM, ImmutableMatrix)
    # 测试普通矩阵与不可变矩阵的相等性
    assert MM.equals(IM)


def test_ME_MM():
    # 测试单位矩阵与普通矩阵相加，结果应为矩阵表达式
    assert isinstance(Identity(3) + MM, MatrixExpr)
    # 测试矩阵符号与普通矩阵相加，结果应为矩阵加法表达式
    assert isinstance(SM + MM, MatAdd)
    assert isinstance(MM + SM, MatAdd)
    # 测试矩阵表达式的元素访问
    assert (Identity(3) + MM)[1, 1] == 6


def test_equality():
    # 测试不同类型矩阵的相等性
    a, b, c = Identity(3), eye(3), ImmutableMatrix(eye(3))
    for x in [a, b, c]:
        for y in [a, b, c]:
            assert x.equals(y)


def test_matrix_symbol_MM():
    # 测试矩阵符号与普通矩阵相加的元素访问
    X = MatrixSymbol('X', 3, 3)
    Y = eye(3) + X
    assert Y[1, 1] == 1 + X[1, 1]


def test_matrix_symbol_vector_matrix_multiplication():
    # 测试矩阵乘法
    A = MM * SV
    B = IM * SV
    assert A == B
    C = (SV.T * MM.T).T
    assert B == C
    D = (SV.T * IM.T).T
    assert C == D


def test_indexing_interactions():
    # 测试矩阵索引操作
    assert (a * IM)[1, 1] == 5*a
    assert (SM + IM)[1, 1] == SM[1, 1] + IM[1, 1]
    assert (SM * IM)[1, 1] == SM[1, 0]*IM[0, 1] + SM[1, 1]*IM[1, 1] + \
        SM[1, 2]*IM[2, 1]


def test_classof():
    # 测试矩阵类型判断函数
    A = Matrix(3, 3, range(9))
    B = ImmutableMatrix(3, 3, range(9))
    C = MatrixSymbol('C', 3, 3)
    assert classof(A, A) == Matrix
    assert classof(B, B) == ImmutableMatrix
    assert classof(A, B) == ImmutableMatrix
    assert classof(B, A) == ImmutableMatrix
    raises(TypeError, lambda: classof(A, C))
```