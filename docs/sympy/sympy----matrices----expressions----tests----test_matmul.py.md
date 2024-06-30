# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_matmul.py`

```
from sympy.core import I, symbols, Basic, Mul, S
from sympy.core.mul import mul
from sympy.functions import adjoint, transpose
from sympy.matrices.exceptions import ShapeError
from sympy.matrices import (Identity, Inverse, Matrix, MatrixSymbol, ZeroMatrix,
        eye, ImmutableMatrix)
from sympy.matrices.expressions import Adjoint, Transpose, det, MatPow
from sympy.matrices.expressions.special import GenericIdentity
from sympy.matrices.expressions.matmul import (factor_in_front, remove_ids,
        MatMul, combine_powers, any_zeros, unpack, only_squares)
from sympy.strategies import null_safe
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.symbol import Symbol

# 定义整数符号变量
n, m, l, k = symbols('n m l k', integer=True)
# 定义通用符号变量
x = symbols('x')
# 定义矩阵符号变量
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)
D = MatrixSymbol('D', n, n)
E = MatrixSymbol('E', m, n)

# 定义测试函数，用于测试矩阵表达式的运算和等价性

def test_evaluate():
    # 测试矩阵乘法的求值
    assert MatMul(C, C, evaluate=True) == MatMul(C, C).doit()

def test_adjoint():
    # 测试共轭转置操作
    assert adjoint(A*B) == Adjoint(B)*Adjoint(A)
    assert adjoint(2*A*B) == 2*Adjoint(B)*Adjoint(A)
    assert adjoint(2*I*C) == -2*I*Adjoint(C)

    # 测试普通矩阵的共轭转置
    M = Matrix(2, 2, [1, 2 + I, 3, 4])
    MA = Matrix(2, 2, [1, 3, 2 - I, 4])
    assert adjoint(M) == MA
    assert adjoint(2*M) == 2*MA
    assert adjoint(MatMul(2, M)) == MatMul(2, MA).doit()


def test_transpose():
    # 测试转置操作
    assert transpose(A*B) == Transpose(B)*Transpose(A)
    assert transpose(2*A*B) == 2*Transpose(B)*Transpose(A)
    assert transpose(2*I*C) == 2*I*Transpose(C)

    # 测试普通矩阵的转置
    M = Matrix(2, 2, [1, 2 + I, 3, 4])
    MT = Matrix(2, 2, [1, 3, 2 + I, 4])
    assert transpose(M) == MT
    assert transpose(2*M) == 2*MT
    assert transpose(x*M) == x*MT
    assert transpose(MatMul(2, M)) == MatMul(2, MT).doit()


def test_factor_in_front():
    # 测试将数值因子移到前面的操作
    assert factor_in_front(MatMul(A, 2, B, evaluate=False)) ==\
                           MatMul(2, A, B, evaluate=False)


def test_remove_ids():
    # 测试移除单位矩阵的操作
    assert remove_ids(MatMul(A, Identity(m), B, evaluate=False)) == \
                      MatMul(A, B, evaluate=False)
    assert null_safe(remove_ids)(MatMul(Identity(n), evaluate=False)) == \
                                 MatMul(Identity(n), evaluate=False)


def test_combine_powers():
    # 测试组合幂运算的操作
    assert combine_powers(MatMul(D, Inverse(D), D, evaluate=False)) == \
                 MatMul(Identity(n), D, evaluate=False)
    assert combine_powers(MatMul(B.T, Inverse(E*A), E, A, B, evaluate=False)) == \
        MatMul(B.T, Identity(m), B, evaluate=False)
    assert combine_powers(MatMul(A, E, Inverse(A*E), D, evaluate=False)) == \
        MatMul(Identity(n), D, evaluate=False)


def test_any_zeros():
    # 测试检查是否存在零矩阵的操作
    assert any_zeros(MatMul(A, ZeroMatrix(m, k), evaluate=False)) == \
                     ZeroMatrix(n, k)


def test_unpack():
    # 测试解包操作
    assert unpack(MatMul(A, evaluate=False)) == A
    x = MatMul(A, B)
    assert unpack(x) == x


def test_only_squares():
    # 断言只包含平方矩阵 C 的函数调用，期望返回包含 C 的列表
    assert only_squares(C) == [C]
    # 断言同时包含矩阵 C 和 D 的函数调用，期望返回包含 C 和 D 的列表
    assert only_squares(C, D) == [C, D]
    # 断言同时包含矩阵 C、A、A 的转置和矩阵 D 的函数调用，
    # 期望返回包含 C、A 乘以 A 的转置后的结果和 D 的列表
    assert only_squares(C, A, A.T, D) == [C, A*A.T, D]
def test_determinant():
    # 检查行列式乘以标量的性质
    assert det(2*C) == 2**n*det(C)
    # 检查行列式乘积乘以标量的性质
    assert det(2*C*D) == 2**n*det(C)*det(D)
    # 检查行列式乘以常数、转置和逆转置的混合项的性质
    assert det(3*C*A*A.T*D) == 3**n*det(C)*det(A*A.T)*det(D)


def test_doit():
    # 检查MatMul对象的参数是否正确设置
    assert MatMul(C, 2, D).args == (C, 2, D)
    # 检查MatMul对象执行后参数的正确性
    assert MatMul(C, 2, D).doit().args == (2, C, D)
    # 检查MatMul对象包含转置的参数是否正确设置
    assert MatMul(C, Transpose(D*C)).args == (C, Transpose(D*C))
    # 检查MatMul对象深度执行转置操作后参数的正确性
    assert MatMul(C, Transpose(D*C)).doit(deep=True).args == (C, C.T, D.T)


def test_doit_drills_down():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[2, 3], [4, 5]])
    # 检查MatMul对象深度执行操作后是否与手动计算结果相同
    assert MatMul(X, MatPow(Y, 2)).doit() == X*Y**2
    # 检查MatMul对象执行后包含转置的参数的正确性
    assert MatMul(C, Transpose(D*C)).doit().args == (C, C.T, D.T)


def test_doit_deep_false_still_canonical():
    # 检查MatMul对象执行深度为False时是否仍保持规范形式
    assert (MatMul(C, Transpose(D*C), 2).doit(deep=False).args ==
            (2, C, Transpose(D*C)))


def test_matmul_scalar_Matrix_doit():
    # Issue 9053
    X = Matrix([[1, 2], [3, 4]])
    # 检查MatMul对象执行后是否正确处理标量与矩阵的乘法
    assert MatMul(2, X).doit() == 2*X


def test_matmul_sympify():
    # 检查MatMul对象的第一个参数是否为Basic类型
    assert isinstance(MatMul(eye(1), eye(1)).args[0], Basic)


def test_collapse_MatrixBase():
    A = Matrix([[1, 1], [1, 1]])
    B = Matrix([[1, 2], [3, 4]])
    # 检查MatMul对象执行后是否正确折叠结果矩阵
    assert MatMul(A, B).doit() == ImmutableMatrix([[4, 6], [4, 6]])


def test_refine():
    # 检查MatMul对象执行精化操作后是否得到正确的结果
    assert refine(C*C.T*D, Q.orthogonal(C)).doit() == D

    kC = k*C
    # 检查MatMul对象执行精化操作后包含常数的情况
    assert refine(kC*C.T, Q.orthogonal(C)).doit() == k*Identity(n)
    assert refine(kC* kC.T, Q.orthogonal(C)).doit() == (k**2)*Identity(n)


def test_matmul_no_matrices():
    # 检查MatMul对象是否正确处理没有矩阵的情况
    assert MatMul(1) == 1
    assert MatMul(n, m) == n*m
    assert not isinstance(MatMul(n, m), MatMul)


def test_matmul_args_cnc():
    # 检查MatMul对象的args_cnc方法对于不同的参数组合是否返回正确的结果
    assert MatMul(n, A, A.T).args_cnc() == [[n], [A, A.T]]
    assert MatMul(A, A.T).args_cnc() == [[], [A, A.T]]


@XFAIL
def test_matmul_args_cnc_symbols():
    # Not currently supported
    a, b = symbols('a b', commutative=False)
    # 检查MatMul对象对于非交换符号的支持情况（目前不支持）
    assert MatMul(n, a, b, A, A.T).args_cnc() == [[n], [a, b, A, A.T]]
    assert MatMul(n, a, A, b, A.T).args_cnc() == [[n], [a, A, b, A.T]]


def test_issue_12950():
    M = Matrix([[Symbol("x")]]) * MatrixSymbol("A", 1, 1)
    # 检查符号矩阵的构建与展开是否正确
    assert MatrixSymbol("A", 1, 1).as_explicit()[0]*Symbol('x') == M.as_explicit()[0]


def test_construction_with_Mul():
    # 检查MatMul对象与Mul对象的等价性
    assert Mul(C, D) == MatMul(C, D)
    assert Mul(D, C) == MatMul(D, C)


def test_construction_with_mul():
    # 检查matmul函数与MatMul对象的等价性
    assert mul(C, D) == MatMul(C, D)
    assert mul(D, C) == MatMul(D, C)
    assert mul(C, D) != MatMul(D, C)


def test_generic_identity():
    # 检查MatMul对象的identity属性是否与GenericIdentity对象相等
    assert MatMul.identity == GenericIdentity()
    assert MatMul.identity != S.One


def test_issue_23519():
    N = Symbol("N", integer=True)
    M1 = MatrixSymbol("M1", N, N)
    M2 = MatrixSymbol("M2", N, N)
    I = Identity(N)
    z = (M2 + 2 * (M2 + I) * M1 + I)
    # 检查表达式中的系数是否正确
    assert z.coeff(M1) == 2*I + 2*M2


def test_shape_error():
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 3, 3)
    # 检查MatMul对象对于形状错误是否引发异常
    raises(ShapeError, lambda: MatMul(A, B))
```