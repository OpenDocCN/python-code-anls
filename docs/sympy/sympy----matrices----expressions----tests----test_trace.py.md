# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_trace.py`

```
from sympy.core import Lambda, S, symbols  # 导入所需的符号和Lambda函数
from sympy.concrete import Sum  # 导入求和相关的模块
from sympy.functions import adjoint, conjugate, transpose  # 导入共轭、转置等函数
from sympy.matrices import (
    eye, Matrix, ShapeError, ImmutableMatrix  # 导入单位矩阵、普通矩阵、形状错误异常类和不可变矩阵
)
from sympy.matrices.expressions import (  # 导入矩阵表达式相关的类和函数
    Adjoint, Identity, FunctionMatrix, MatrixExpr, MatrixSymbol, Trace,  # Adjoint、单位矩阵、函数矩阵、矩阵表达式、矩阵符号、迹
    ZeroMatrix, trace, MatPow, MatAdd, MatMul  # 零矩阵、迹函数、矩阵幂、矩阵加法、矩阵乘法
)
from sympy.matrices.expressions.special import OneMatrix  # 导入特殊的全1矩阵
from sympy.testing.pytest import raises  # 导入异常测试函数
from sympy.abc import i  # 导入符号i


n = symbols('n', integer=True)  # 定义整数符号n
A = MatrixSymbol('A', n, n)  # 定义一个符号矩阵A，大小为n x n
B = MatrixSymbol('B', n, n)  # 定义一个符号矩阵B，大小为n x n
C = MatrixSymbol('C', 3, 4)  # 定义一个符号矩阵C，大小为3 x 4


def test_Trace():
    # 测试Trace函数的基本属性
    assert isinstance(Trace(A), Trace)
    assert not isinstance(Trace(A), MatrixExpr)
    raises(ShapeError, lambda: Trace(C))  # 检查对于形状错误是否引发异常
    assert trace(eye(3)) == 3  # 检查单位矩阵的迹是否为3
    assert trace(Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])) == 15  # 检查指定矩阵的迹是否为15

    # 检查Trace的共轭、转置、伴随是否与对应操作后的迹相等
    assert adjoint(Trace(A)) == trace(Adjoint(A))
    assert conjugate(Trace(A)) == trace(Adjoint(A))
    assert transpose(Trace(A)) == Trace(A)

    _ = A / Trace(A)  # 确保可以进行这样的除法操作

    # 一些简单的化简
    assert trace(Identity(5)) == 5  # 检查5阶单位矩阵的迹是否为5
    assert trace(ZeroMatrix(5, 5)) == 0  # 检查5阶零矩阵的迹是否为0
    assert trace(OneMatrix(1, 1)) == 1  # 检查1阶全1矩阵的迹是否为1
    assert trace(OneMatrix(2, 2)) == 2  # 检查2阶全1矩阵的迹是否为2
    assert trace(OneMatrix(n, n)) == n  # 检查n阶全1矩阵的迹是否为n
    assert trace(2*A*B) == 2*Trace(A*B)  # 检查2倍矩阵A*B的迹是否等于2乘以A*B的迹
    assert trace(A.T) == trace(A)  # 检查矩阵A的转置的迹是否等于矩阵A的迹

    i, j = symbols('i j')  # 定义符号i和j
    F = FunctionMatrix(3, 3, Lambda((i, j), i + j))  # 创建一个3x3的函数矩阵F
    assert trace(F) == (0 + 0) + (1 + 1) + (2 + 2)  # 检查函数矩阵F的迹是否为3

    raises(TypeError, lambda: Trace(S.One))  # 检查对于类型错误是否引发异常

    assert Trace(A).arg is A  # 检查Trace函数的参数是否为矩阵A

    assert str(trace(A)) == str(Trace(A).doit())  # 检查trace(A)的字符串表示是否与doit()后的Trace(A)字符串表示相等

    assert Trace(A).is_commutative is True  # 检查Trace(A)是否是可交换的

def test_Trace_A_plus_B():
    assert trace(A + B) == Trace(A) + Trace(B)  # 检查矩阵A+B的迹是否等于矩阵A的迹加上矩阵B的迹
    assert Trace(A + B).arg == MatAdd(A, B)  # 检查Trace(A+B)的参数是否为A+B
    assert Trace(A + B).doit() == Trace(A) + Trace(B)  # 检查doit()后Trace(A+B)的结果是否等于Trace(A) + Trace(B)

def test_Trace_MatAdd_doit():
    # 检查MatAdd对象的迹计算是否正确
    X = ImmutableMatrix([[1, 2, 3]]*3)
    Y = MatrixSymbol('Y', 3, 3)
    q = MatAdd(X, 2*X, Y, -3*Y)
    assert Trace(q).arg == q  # 检查Trace(q)的参数是否为q
    assert Trace(q).doit() == 18 - 2*Trace(Y)  # 检查doit()后Trace(q)的结果是否等于18 - 2*Trace(Y)

def test_Trace_MatPow_doit():
    X = Matrix([[1, 2], [3, 4]])  # 创建一个2x2的矩阵X
    assert Trace(X).doit() == 5  # 检查矩阵X的迹是否等于5
    q = MatPow(X, 2)  # 创建矩阵幂对象q
    assert Trace(q).arg == q  # 检查Trace(q)的参数是否为q
    assert Trace(q).doit() == 29  # 检查doit()后Trace(q)的结果是否等于29

def test_Trace_MutableMatrix_plus():
    # 检查可变矩阵对象的加法操作
    X = Matrix([[1, 2], [3, 4]])  # 创建一个2x2的矩阵X
    assert Trace(X) + Trace(X) == 2*Trace(X)  # 检查两个Trace(X)的加法结果是否等于2乘以Trace(X)

def test_Trace_doit_deep_False():
    X = Matrix([[1, 2], [3, 4]])  # 创建一个2x2的矩阵X
    q = MatPow(X, 2)  # 创建矩阵幂对象q
    assert Trace(q).doit(deep=False).arg == q  # 检查深度为False时doit()后Trace(q)的参数是否为q
    q = MatAdd(X, 2*X)  # 创建矩阵加法对象q
    assert Trace(q).doit(deep=False).arg == q  # 检查深度为False时doit()后Trace(q)的参数是否为q
    q = MatMul(X, 2*X)  # 创建矩阵乘法对象q
    assert Trace(q).doit(deep=False).arg == q  # 检查深度为False时doit()后Trace(q)的参数是否为q

def test_trace_constant_factor():
    # 检查常数因子的迹运算
    assert trace(2*A) == 2*Trace(A)  # 检查2倍矩阵A的迹是否等于2乘以Trace(A)
    X = ImmutableMatrix([[1, 2], [3, 4]])  # 创建一个不
# 定义一个名为 test_trace_normalize 的测试函数，用于验证跟踪操作的规范化行为
def test_trace_normalize():
    # 断言对于矩阵 B*A 的迹不等于 A*B 的迹
    assert Trace(B*A) != Trace(A*B)
    # 断言规范化了的 B*A 的迹等于 A*B 的迹
    assert Trace(B*A)._normalize() == Trace(A*B)
    # 断言规范化了的 B*A 转置的迹等于 A*B 转置的迹
    assert Trace(B*A.T)._normalize() == Trace(A*B.T)


# 定义一个名为 test_trace_as_explicit 的测试函数，用于验证将迹表示为显式形式的行为
def test_trace_as_explicit():
    # 断言尝试将矩阵 A 的迹作为显式形式会引发 ValueError 异常
    raises(ValueError, lambda: Trace(A).as_explicit())

    # 定义一个名为 X 的 3x3 矩阵符号
    X = MatrixSymbol("X", 3, 3)
    # 断言矩阵 X 的迹作为显式形式等于 X[0, 0] + X[1, 1] + X[2, 2]
    assert Trace(X).as_explicit() == X[0, 0] + X[1, 1] + X[2, 2]
    # 断言单位矩阵 eye(3) 的迹作为显式形式等于 3
    assert Trace(eye(3)).as_explicit() == 3
```