# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_inverse.py`

```
from sympy.core import symbols, S   # 导入符号和S对象
from sympy.matrices.expressions import MatrixSymbol, Inverse, MatPow, ZeroMatrix, OneMatrix  # 导入矩阵相关的表达式和操作
from sympy.matrices.exceptions import NonInvertibleMatrixError, NonSquareMatrixError  # 导入矩阵异常类
from sympy.matrices import eye, Identity  # 导入单位矩阵相关函数
from sympy.testing.pytest import raises  # 导入用于测试的raises函数
from sympy.assumptions.ask import Q  # 导入符号逻辑的问询模块
from sympy.assumptions.refine import refine  # 导入符号逻辑的精化模块

n, m, l = symbols('n m l', integer=True)  # 声明符号变量 n, m, l，均为整数类型
A = MatrixSymbol('A', n, m)  # 声明矩阵符号 A，大小为 n x m
B = MatrixSymbol('B', m, l)  # 声明矩阵符号 B，大小为 m x l
C = MatrixSymbol('C', n, n)  # 声明矩阵符号 C，大小为 n x n
D = MatrixSymbol('D', n, n)  # 声明矩阵符号 D，大小为 n x n
E = MatrixSymbol('E', m, n)  # 声明矩阵符号 E，大小为 m x n

def test_inverse():
    assert Inverse(C).args == (C, S.NegativeOne)  # 断言逆矩阵 C 的参数是 (C, -1)
    assert Inverse(C).shape == (n, n)  # 断言逆矩阵 C 的形状为 (n, n)
    assert Inverse(A*E).shape == (n, n)  # 断言逆矩阵 A*E 的形状为 (n, n)
    assert Inverse(E*A).shape == (m, m)  # 断言逆矩阵 E*A 的形状为 (m, m)
    assert Inverse(C).inverse() == C  # 断言逆矩阵的逆仍为自身
    assert Inverse(Inverse(C)).doit() == C  # 断言两次逆操作等同于原矩阵
    assert isinstance(Inverse(Inverse(C)), Inverse)  # 断言连续两次逆操作返回的类型是逆矩阵类型

    assert Inverse(*Inverse(E*A).args) == Inverse(E*A)  # 断言通过参数传递的方式获得逆矩阵相等

    assert C.inverse().inverse() == C  # 断言逆矩阵的逆逆矩阵等于原矩阵

    assert C.inverse()*C == Identity(C.rows)  # 断言逆矩阵乘以原矩阵等于单位矩阵

    assert Identity(n).inverse() == Identity(n)  # 断言单位矩阵的逆矩阵是自身
    assert (3*Identity(n)).inverse() == Identity(n)/3  # 断言数乘单位矩阵的逆矩阵

    # 简化乘法，如果可能的话（即子矩阵是方阵）
    assert (C*D).inverse() == D.I*C.I
    # 即使不可能时仍然有效
    assert isinstance((A*E).inverse(), Inverse)
    assert Inverse(C*D).doit(inv_expand=False) == Inverse(C*D)

    assert Inverse(eye(3)).doit() == eye(3)
    assert Inverse(eye(3)).doit(deep=False) == eye(3)

    assert OneMatrix(1, 1).I == Identity(1)
    assert isinstance(OneMatrix(n, n).I, Inverse)

def test_inverse_non_invertible():
    raises(NonInvertibleMatrixError, lambda: ZeroMatrix(n, n).I)  # 测试零矩阵的逆不存在
    raises(NonInvertibleMatrixError, lambda: OneMatrix(2, 2).I)  # 测试非方阵单位矩阵的逆不存在

def test_refine():
    assert refine(C.I, Q.orthogonal(C)) == C.T  # 断言逆矩阵 C 的精化结果与 C 转置相等


def test_inverse_matpow_canonicalization():
    A = MatrixSymbol('A', 3, 3)
    assert Inverse(MatPow(A, 3)).doit() == MatPow(Inverse(A), 3).doit()

def test_nonsquare_error():
    A = MatrixSymbol('A', 3, 4)
    raises(NonSquareMatrixError, lambda: Inverse(A))  # 测试非方阵的逆操作引发异常

def test_adjoint_trnaspose_conjugate():
    A = MatrixSymbol('A', n, n)
    assert A.transpose().inverse() == A.inverse().transpose()  # 断言矩阵转置的逆矩阵等于逆矩阵的转置
    assert A.conjugate().inverse() == A.inverse().conjugate()  # 断言矩阵共轭的逆矩阵等于逆矩阵的共轭
    assert A.adjoint().inverse() == A.inverse().adjoint()  # 断言矩阵伴随的逆矩阵等于逆矩阵的伴随
```