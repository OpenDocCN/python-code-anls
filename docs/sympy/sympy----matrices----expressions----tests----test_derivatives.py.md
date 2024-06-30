# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_derivatives.py`

```
"""
Some examples have been taken from:

http://www.math.uwaterloo.ca/~hwolkowi//matrixcookbook.pdf
"""
# 从Sympy库中导入各种函数和类
from sympy import KroneckerProduct
from sympy.combinatorics import Permutation
from sympy.concrete.summations import Sum
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import (HadamardPower, HadamardProduct, hadamard_product)
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import OneMatrix
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import (Identity, ZeroMatrix)
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.matrices.expressions import hadamard_power
from sympy.tensor.array.expressions.array_expressions import ArrayAdd, ArrayTensorProduct, PermuteDims

# 定义符号变量
i, j, k = symbols("i j k")
m, n = symbols("m n")

# 定义矩阵符号
X = MatrixSymbol("X", k, k)
x = MatrixSymbol("x", k, 1)
y = MatrixSymbol("y", k, 1)

A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)
c = MatrixSymbol("c", k, 1)
d = MatrixSymbol("d", k, 1)

# 定义Kronecker Delta函数
KDelta = lambda i, j: KroneckerDelta(i, j, (0, k-1))

# 定义函数_check_derivative_with_explicit_matrix，用于检查具有显式矩阵的导数
def _check_derivative_with_explicit_matrix(expr, x, diffexpr, dim=2):
    # TODO: this is commented because it slows down the tests.
    # 此函数当前被注释掉，因为它会减慢测试速度
    return

# 定义测试函数test_matrix_derivative_by_scalar，用于测试矩阵关于标量的导数
def test_matrix_derivative_by_scalar():
    assert A.diff(i) == ZeroMatrix(k, k)
    assert (A*(X + B)*c).diff(i) == ZeroMatrix(k, 1)
    assert x.diff(i) == ZeroMatrix(k, 1)
    assert (x.T*y).diff(i) == ZeroMatrix(1, 1)
    assert (x*x.T).diff(i) == ZeroMatrix(k, k)
    assert (x + y).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_power(x, 2).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_power(x, i).diff(i).dummy_eq(
        HadamardProduct(x.applyfunc(log), HadamardPower(x, i)))
    assert hadamard_product(x, y).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_product(i*OneMatrix(k, 1), x, y).diff(i) == hadamard_product(x, y)
    assert (i*x).diff(i) == x
    assert (sin(i)*A*B*x).diff(i) == cos(i)*A*B*x
    # 断言：对应矩阵 x 应用 sin 函数后再进行 i 次微分得到零矩阵
    assert x.applyfunc(sin).diff(i) == ZeroMatrix(k, 1)
    
    # 断言：对矩阵 i^2*X 求迹后再进行 i 次微分得到的结果是 2*i*Trace(X)
    assert Trace(i**2*X).diff(i) == 2*i*Trace(X)
    
    # 定义符号变量 mu
    mu = symbols("mu")
    
    # 定义表达式：2*mu*x
    expr = (2*mu*x)
    
    # 断言：对表达式 expr 关于变量 x 的偏导数应该等于 2*mu*单位矩阵 k
    assert expr.diff(x) == 2*mu*Identity(k)
def test_one_matrix():
    # Assert that the derivative of the transpose of x times a matrix of ones with shape (k, 1)
    # with respect to x is a matrix of ones with shape (k, 1).
    assert MatMul(x.T, OneMatrix(k, 1)).diff(x) == OneMatrix(k, 1)


def test_matrix_derivative_non_matrix_result():
    # This is a 4-dimensional array:
    I = Identity(k)
    # Permute dimensions of the tensor product of identity matrices I and I
    AdA = PermuteDims(ArrayTensorProduct(I, I), Permutation(3)(1, 2))
    assert A.diff(A) == AdA
    # Differentiate transpose of A with respect to A
    assert A.T.diff(A) == PermuteDims(ArrayTensorProduct(I, I), Permutation(3)(1, 2, 3))
    # Differentiate 2*A with respect to A
    assert (2*A).diff(A) == PermuteDims(ArrayTensorProduct(2*I, I), Permutation(3)(1, 2))
    # Differentiate sum of A and A with respect to A
    assert MatAdd(A, A).diff(A) == ArrayAdd(AdA, AdA)
    # Differentiate sum of A and B with respect to A
    assert (A + B).diff(A) == AdA


def test_matrix_derivative_trivial_cases():
    # Cookbook example 33:
    # TODO: find a way to represent a four-dimensional zero-array:
    assert X.diff(A) == ArrayDerivative(X, A)


def test_matrix_derivative_with_inverse():
    # Cookbook example 61:
    expr = a.T*Inverse(X)*b
    assert expr.diff(X) == -Inverse(X).T*a*b.T*Inverse(X).T

    # Cookbook example 62:
    expr = Determinant(Inverse(X))
    # Not implemented yet:
    # assert expr.diff(X) == -Determinant(X.inv())*(X.inv()).T

    # Cookbook example 63:
    expr = Trace(A*Inverse(X)*B)
    assert expr.diff(X) == -(X**(-1)*B*A*X**(-1)).T

    # Cookbook example 64:
    expr = Trace(Inverse(X + A))
    assert expr.diff(X) == -(Inverse(X + A)).T**2


def test_matrix_derivative_vectors_and_scalars():
    # Assert identity matrix of size k when differentiating x with respect to x
    assert x.diff(x) == Identity(k)
    # Assert Kronecker delta when differentiating x[i, 0] with respect to x[m, 0]
    assert x[i, 0].diff(x[m, 0]).doit() == KDelta(m, i)

    # Differentiate transpose of x with respect to x
    assert x.T.diff(x) == Identity(k)

    # Cookbook example 69:
    expr = x.T*a
    assert expr.diff(x) == a
    assert expr[0, 0].diff(x[m, 0]).doit() == a[m, 0]
    expr = a.T*x
    assert expr.diff(x) == a

    # Cookbook example 70:
    expr = a.T*X*b
    assert expr.diff(X) == a*b.T

    # Cookbook example 71:
    expr = a.T*X.T*b
    assert expr.diff(X) == b*a.T

    # Cookbook example 72:
    expr = a.T*X*a
    assert expr.diff(X) == a*a.T
    expr = a.T*X.T*a
    assert expr.diff(X) == a*a.T

    # Cookbook example 77:
    expr = b.T*X.T*X*c
    assert expr.diff(X) == X*b*c.T + X*c*b.T

    # Cookbook example 78:
    expr = (B*x + b).T*C*(D*x + d)
    assert expr.diff(x) == B.T*C*(D*x + d) + D.T*C.T*(B*x + b)

    # Cookbook example 81:
    expr = x.T*B*x
    assert expr.diff(x) == B*x + B.T*x

    # Cookbook example 82:
    expr = b.T*X.T*D*X*c
    assert expr.diff(X) == D.T*X*b*c.T + D*X*c*b.T

    # Cookbook example 83:
    expr = (X*b + c).T*D*(X*b + c)
    assert expr.diff(X) == D*(X*b + c)*b.T + D.T*(X*b + c)*b.T
    assert str(expr[0, 0].diff(X[m, n]).doit()) == \
        'b[n, 0]*Sum((c[_i_1, 0] + Sum(X[_i_1, _i_3]*b[_i_3, 0], (_i_3, 0, k - 1)))*D[_i_1, m], (_i_1, 0, k - 1)) + Sum((c[_i_2, 0] + Sum(X[_i_2, _i_4]*b[_i_4, 0], (_i_4, 0, k - 1)))*D[m, _i_2]*b[n, 0], (_i_2, 0, k - 1))'

    # See https://github.com/sympy/sympy/issues/16504#issuecomment-1018339957
    expr = x*x.T*x
    I = Identity(k)
    assert expr.diff(x) == KroneckerProduct(I, x.T*x) + 2*x*x.T


def test_matrix_derivatives_of_traces():
    pass  # This function is not implemented yet
    expr = Trace(A)*A
    # 定义一个表达式，Trace(A) 乘以 A

    I = Identity(k)
    # 创建一个 k 阶单位矩阵 I

    assert expr.diff(A) == ArrayAdd(ArrayTensorProduct(I, A), PermuteDims(ArrayTensorProduct(Trace(A)*I, I), Permutation(3)(1, 2)))
    # 断言：expr 对 A 的偏导数应等于特定的张量运算结果

    assert expr[i, j].diff(A[m, n]).doit() == (
        KDelta(i, m)*KDelta(j, n)*Trace(A) +
        KDelta(m, n)*A[i, j]
    )
    # 断言：expr 的索引 i, j 对 A[m, n] 的偏导数应等于特定的表达式

    ## First order:

    # Cookbook example 99:
    expr = Trace(X)
    # 定义一个表达式，对矩阵 X 求迹

    assert expr.diff(X) == Identity(k)
    # 断言：expr 对 X 的偏导数应为 k 阶单位矩阵 I

    assert expr.rewrite(Sum).diff(X[m, n]).doit() == KDelta(m, n)
    # 断言：将迹表达式改写成求和形式后，对 X[m, n] 的偏导数应为克罗内克 δ 符号

    # Cookbook example 100:
    expr = Trace(X*A)
    # 定义一个表达式，对矩阵 X*A 求迹

    assert expr.diff(X) == A.T
    # 断言：expr 对 X 的偏导数应为 A 的转置

    assert expr.rewrite(Sum).diff(X[m, n]).doit() == A[n, m]
    # 断言：将迹表达式改写成求和形式后，对 X[m, n] 的偏导数应为 A[n, m]

    # Cookbook example 101:
    expr = Trace(A*X*B)
    # 定义一个表达式，对矩阵 A*X*B 求迹

    assert expr.diff(X) == A.T*B.T
    # 断言：expr 对 X 的偏导数应为 A 的转置乘以 B 的转置

    assert expr.rewrite(Sum).diff(X[m, n]).doit().dummy_eq((A.T*B.T)[m, n])
    # 断言：将迹表达式改写成求和形式后，对 X[m, n] 的偏导数应与 (A.T*B.T)[m, n] 相等

    # Cookbook example 102:
    expr = Trace(A*X.T*B)
    # 定义一个表达式，对矩阵 A*X.T*B 求迹

    assert expr.diff(X) == B*A
    # 断言：expr 对 X 的偏导数应为 B 乘以 A

    # Cookbook example 103:
    expr = Trace(X.T*A)
    # 定义一个表达式，对矩阵 X.T*A 求迹

    assert expr.diff(X) == A
    # 断言：expr 对 X 的偏导数应为 A

    # Cookbook example 104:
    expr = Trace(A*X.T)
    # 定义一个表达式，对矩阵 A*X.T 求迹

    assert expr.diff(X) == A
    # 断言：expr 对 X 的偏导数应为 A

    # Cookbook example 105:
    # TODO: TensorProduct is not supported
    #expr = Trace(TensorProduct(A, X))
    #assert expr.diff(X) == Trace(A)*Identity(k)

    ## Second order:

    # Cookbook example 106:
    expr = Trace(X**2)
    # 定义一个表达式，对矩阵 X^2 求迹

    assert expr.diff(X) == 2*X.T
    # 断言：expr 对 X 的偏导数应为 2*X 的转置

    # Cookbook example 107:
    expr = Trace(X**2*B)
    # 定义一个表达式，对矩阵 X^2*B 求迹

    assert expr.diff(X) == (X*B + B*X).T
    # 断言：expr 对 X 的偏导数应为 (X*B + B*X) 的转置

    expr = Trace(MatMul(X, X, B))
    # 定义一个表达式，对矩阵 X*X*B 求迹（MatMul 表示矩阵乘法）

    assert expr.diff(X) == (X*B + B*X).T
    # 断言：expr 对 X 的偏导数应为 (X*B + B*X) 的转置

    # Cookbook example 108:
    expr = Trace(X.T*B*X)
    # 定义一个表达式，对矩阵 X.T*B*X 求迹

    assert expr.diff(X) == B*X + B.T*X
    # 断言：expr 对 X 的偏导数应为 B*X + B 的转置乘以 X

    # Cookbook example 109:
    expr = Trace(B*X*X.T)
    # 定义一个表达式，对矩阵 B*X*X.T 求迹

    assert expr.diff(X) == B*X + B.T*X
    # 断言：expr 对 X 的偏导数应为 B*X + B 的转置乘以 X

    # Cookbook example 110:
    expr = Trace(X*X.T*B)
    # 定义一个表达式，对矩阵 X*X.T*B 求迹

    assert expr.diff(X) == B*X + B.T*X
    # 断言：expr 对 X 的偏导数应为 B*X + B 的转置乘以 X

    # Cookbook example 111:
    expr = Trace(X*B*X.T)
    # 定义一个表达式，对矩阵 X*B*X.T 求迹

    assert expr.diff(X) == X*B.T + X*B
    # 断言：expr 对 X 的偏导数应为 X 乘以 B 的转置再加上 X 乘以 B

    # Cookbook example 112:
    expr = Trace(B*X.T*X)
    # 定义一个表达式，对矩阵 B*X.T*X 求迹

    assert expr.diff(X) == X*B.T + X*B
    # 断言：expr 对 X 的偏导数应为 X 乘以 B 的转置再加上 X 乘以 B

    # Cookbook example 113:
    expr = Trace(X.T*X*B)
    # 定义一个表达式，对矩阵 X.T*X*B 求迹

    assert expr.diff(X) == X*B.T + X*B
    # 断言：expr 对 X 的偏导数应为 X 乘以 B 的转置再加上 X 乘以 B

    # Cookbook example 114:
    expr = Trace(A*X*B*X)
    # 定义一个表达式，对矩阵 A*X*B*X 求迹

    assert expr.diff(X) == A.T*X.T*B.T + B.T*X.T*A.T
    # 断言：expr 对 X 的偏导数应为 A 的转置乘以 X 的转置乘以 B 的转置再加上 B 的转置乘以 X 的转置乘以 A 的转置

    # Cookbook example 115:
    expr = Trace(X.T*X)
    # 定义一个表达式，对矩阵 X.T*X 求迹

    assert expr.diff(X) == 2*X
    # 断言：expr 对 X 的偏导数应为 2*X

    expr = Trace(X*X.T)
    # 定义一个表达式，对矩阵 X*X.T 求迹

    assert expr.diff(X) == 2*X
    # 断言：expr 对 X 的偏导数应为 2*X

    # Cookbook example 116:
    expr = Trace(B.T*X.T*C*X*B)
    # 定义一个表达式，对矩阵 B.T*X.T*C*X*B 求迹

    assert expr.diff(X) == C.T*X*B*B.T + C*X*B*B.T
    # 断言：expr 对 X 的偏导数应为 C 的转置乘以 X 乘以 B 乘以 B 的转置再加上 C 乘以 X 乘以 B
    #assert expr.diff(X) == k*(X**(k-1)).T

    # Cookbook example 122:
    # 计算迹函数 Trace(A*X**k) 对变量 X 的导数
    expr = Trace(A*X**k)
    #assert expr.diff(X) == # 需要指定索引

    # Cookbook example 123:
    # 计算迹函数 Trace(B.T*X.T*C*X*X.T*C*X*B) 对变量 X 的导数，并进行断言验证
    expr = Trace(B.T*X.T*C*X*X.T*C*X*B)
    assert expr.diff(X) == C*X*X.T*C*X*B*B.T + C.T*X*B*B.T*X.T*C.T*X + C*X*B*B.T*X.T*C*X + C.T*X*X.T*C.T*X*B*B.T

    # Other

    # Cookbook example 124:
    # 计算迹函数 Trace(A*X**(-1)*B) 对变量 X 的导数，并进行断言验证
    expr = Trace(A*X**(-1)*B)
    assert expr.diff(X) == -Inverse(X).T*A.T*B.T*Inverse(X).T

    # Cookbook example 125:
    # 计算迹函数 Trace(Inverse(X.T*C*X)*A) 对变量 X 的导数，并进行断言验证
    # 警告：如果 B 和 C 是对称的，则结果等价
    assert expr.diff(X) == - X.inv().T*A.T*X.inv()*C.inv().T*X.inv().T - X.inv().T*A*X.inv()*C.inv()*X.inv().T

    # Cookbook example 126:
    # 计算迹函数 Trace((X.T*C*X).inv()*(X.T*B*X)) 对变量 X 的导数，并进行断言验证
    assert expr.diff(X) == -2*C*X*(X.T*C*X).inv()*X.T*B*X*(X.T*C*X).inv() + 2*B*X*(X.T*C*X).inv()

    # Cookbook example 127:
    # 计算迹函数 Trace((A + X.T*C*X).inv()*(X.T*B*X)) 对变量 X 的导数，并进行断言验证
    # 警告：如果 B 和 C 是对称的，则结果等价
    assert expr.diff(X) == B*X*Inverse(A + X.T*C*X) - C*X*Inverse(A + X.T*C*X)*X.T*B*X*Inverse(A + X.T*C*X) - C.T*X*Inverse(A.T + (C*X).T*X)*X.T*B.T*X*Inverse(A.T + (C*X).T*X) + B.T*X*Inverse(A.T + (C*X).T*X)
def test_derivatives_of_complicated_matrix_expr():
    expr = a.T*(A*X*(X.T*B + X*A) + B.T*X.T*(a*b.T*(X*D*X.T + X*(X.T*B + A*X)*D*B - X.T*C.T*A)*B + B*(X*D.T + B*A*X*A.T - 3*X*D))*B + 42*X*B*X.T*A.T*(X + X.T))*b
    result = (B*(B*A*X*A.T - 3*X*D + X*D.T) + a*b.T*(X*(A*X + X.T*B)*D*B + X*D*X.T - X.T*C.T*A)*B)*B*b*a.T*B.T + B**2*b*a.T*B.T*X.T*a*b.T*X*D + 42*A*X*B.T*X.T*a*b.T + B*D*B**3*b*a.T*B.T*X.T*a*b.T*X + B*b*a.T*A*X + a*b.T*(42*X + 42*X.T)*A*X*B.T + b*a.T*X*B*a*b.T*B.T**2*X*D.T + b*a.T*X*B*a*b.T*B.T**3*D.T*(B.T*X + X.T*A.T) + 42*b*a.T*X*B*X.T*A.T + A.T*(42*X + 42*X.T)*b*a.T*X*B + A.T*B.T**2*X*B*a*b.T*B.T*A + A.T*a*b.T*(A.T*X.T + B.T*X) + A.T*X.T*b*a.T*X*B*a*b.T*B.T**3*D.T + B.T*X*B*a*b.T*B.T*D - 3*B.T*X*B*a*b.T*B.T*D.T - C.T*A*B**2*b*a.T*B.T*X.T*a*b.T + X.T*A.T*a*b.T*A.T
    assert expr.diff(X) == result


def test_mixed_deriv_mixed_expressions():
    expr = 3*Trace(A)
    assert expr.diff(A) == 3*Identity(k)

    expr = k
    deriv = expr.diff(A)
    assert isinstance(deriv, ZeroMatrix)
    assert deriv == ZeroMatrix(k, k)

    expr = Trace(A)**2
    assert expr.diff(A) == (2*Trace(A))*Identity(k)

    expr = Trace(A)*A
    I = Identity(k)
    assert expr.diff(A) == ArrayAdd(ArrayTensorProduct(I, A), PermuteDims(ArrayTensorProduct(Trace(A)*I, I), Permutation(3)(1, 2)))

    expr = Trace(Trace(A)*A)
    assert expr.diff(A) == (2*Trace(A))*Identity(k)

    expr = Trace(Trace(Trace(A)*A)*A)
    assert expr.diff(A) == (3*Trace(A)**2)*Identity(k)


def test_derivatives_matrix_norms():
    expr = x.T*y
    assert expr.diff(x) == y
    assert expr[0, 0].diff(x[m, 0]).doit() == y[m, 0]

    expr = (x.T*y)**S.Half
    assert expr.diff(x) == y/(2*sqrt(x.T*y))

    expr = (x.T*x)**S.Half
    assert expr.diff(x) == x*(x.T*x)**Rational(-1, 2)

    expr = (c.T*a*x.T*b)**S.Half
    assert expr.diff(x) == b*a.T*c/sqrt(c.T*a*x.T*b)/2

    expr = (c.T*a*x.T*b)**Rational(1, 3)
    assert expr.diff(x) == b*a.T*c*(c.T*a*x.T*b)**Rational(-2, 3)/3

    expr = (a.T*X*b)**S.Half
    assert expr.diff(X) == a/(2*sqrt(a.T*X*b))*b.T

    expr = d.T*x*(a.T*X*b)**S.Half*y.T*c
    assert expr.diff(X) == a/(2*sqrt(a.T*X*b))*x.T*d*y.T*c*b.T


def test_derivatives_elementwise_applyfunc():
    expr = x.applyfunc(tan)
    assert expr.diff(x).dummy_eq(
        DiagMatrix(x.applyfunc(lambda x: tan(x)**2 + 1)))
    assert expr[i, 0].diff(x[m, 0]).doit() == (tan(x[i, 0])**2 + 1)*KDelta(i, m)
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    expr = (i**2*x).applyfunc(sin)
    assert expr.diff(i).dummy_eq(
        HadamardProduct((2*i)*x, (i**2*x).applyfunc(cos)))
    assert expr[i, 0].diff(i).doit() == 2*i*x[i, 0]*cos(i**2*x[i, 0])
    _check_derivative_with_explicit_matrix(expr, i, expr.diff(i))

    expr = (log(i)*A*B).applyfunc(sin)
    assert expr.diff(i).dummy_eq(
        HadamardProduct(A*B/i, (log(i)*A*B).applyfunc(cos)))
    _check_derivative_with_explicit_matrix(expr, i, expr.diff(i))

    expr = A*x.applyfunc(exp)
    # TODO: restore this result (currently returning the transpose):
    #  assert expr.diff(x).dummy_eq(DiagMatrix(x.applyfunc(exp))*A.T)
    # 调用函数检查带有显式矩阵的导数表达式
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    # 构造表达式，包括转置运算、矩阵乘法和元素级函数
    expr = x.T*A*x + k*y.applyfunc(sin).T*x
    # 断言表达式关于 x 的导数是否符合预期
    assert expr.diff(x).dummy_eq(A.T*x + A*x + k*y.applyfunc(sin))
    # 再次调用函数检查带有显式矩阵的导数表达式
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    # 构造表达式，包括转置运算和元素级函数
    expr = x.applyfunc(sin).T*y
    # TODO: restore (currently returning the transpose):
    #  assert expr.diff(x).dummy_eq(DiagMatrix(x.applyfunc(cos))*y)
    # 调用函数检查带有显式矩阵的导数表达式
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    # 构造表达式，包括矩阵乘法和元素级函数
    expr = (a.T * X * b).applyfunc(sin)
    # 断言表达式关于矩阵 X 的导数是否符合预期
    assert expr.diff(X).dummy_eq(a*(a.T*X*b).applyfunc(cos)*b.T)
    # 再次调用函数检查带有显式矩阵的导数表达式
    _check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    # 构造表达式，包括元素级函数和矩阵乘法
    expr = a.T * X.applyfunc(sin) * b
    # 断言表达式关于矩阵 X 的导数是否符合预期
    assert expr.diff(X).dummy_eq(
        DiagMatrix(a)*X.applyfunc(cos)*DiagMatrix(b))
    # 再次调用函数检查带有显式矩阵的导数表达式
    _check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    # 构造表达式，包括矩阵乘法、元素级函数和向量转置
    expr = a.T * (A*X*B).applyfunc(sin) * b
    # 断言表达式关于矩阵 X 的导数是否符合预期
    assert expr.diff(X).dummy_eq(
        A.T*DiagMatrix(a)*(A*X*B).applyfunc(cos)*DiagMatrix(b)*B.T)
    # TODO: not implemented
    #assert expr.diff(X) == ...
    #_check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    # 构造表达式，包括矩阵乘法、元素级函数和向量转置
    expr = a.T * (A*X*b).applyfunc(sin) * b.T
    # 断言表达式关于矩阵 X 的导数是否符合预期
    # TODO: not implemented
    #assert expr.diff(X) == ...
    #_check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    # 构造表达式，包括矩阵乘法、元素级函数和哈达玛积
    expr = a.T*A*X.applyfunc(sin)*B*b
    # 断言表达式关于矩阵 X 的导数是否符合预期
    assert expr.diff(X).dummy_eq(
        HadamardProduct(A.T * a * b.T * B.T, X.applyfunc(cos)))

    # 构造表达式，包括矩阵乘法、元素级函数和对数函数
    expr = a.T * (A*X.applyfunc(sin)*B).applyfunc(log) * b
    # TODO: wrong
    # assert expr.diff(X) == A.T*DiagMatrix(a)*(A*X.applyfunc(sin)*B).applyfunc(Lambda(k, 1/k))*DiagMatrix(b)*B.T

    # 构造表达式，包括元素级函数和对数函数
    expr = a.T * (X.applyfunc(sin)).applyfunc(log) * b
    # TODO: wrong
    # assert expr.diff(X) == DiagMatrix(a)*X.applyfunc(sin).applyfunc(Lambda(k, 1/k))*DiagMatrix(b)
def test_derivatives_of_hadamard_expressions():

    # Hadamard Product

    # 计算 Hadamard 乘积表达式
    expr = hadamard_product(a, x, b)
    # 断言对 x 的导数应该等于对角矩阵，其对角线元素是 b 与 a 的对应元素相乘
    assert expr.diff(x) == DiagMatrix(hadamard_product(b, a))

    # 计算复合的 Hadamard 乘积表达式
    expr = a.T*hadamard_product(A, X, B)*b
    # 断言对 X 的导数应该等于 Hadamard 乘积 (a * b^T) 和 (A, B) 的 Hadamard 乘积
    assert expr.diff(X) == HadamardProduct(a*b.T, A, B)

    # Hadamard Power

    # 计算 Hadamard 幂次表达式
    expr = hadamard_power(x, 2)
    # 断言对 x 的导数应该等于 2 * 对角矩阵 x
    assert expr.diff(x).doit() == 2*DiagMatrix(x)

    # 计算 Hadamard 幂次表达式
    expr = hadamard_power(x.T, 2)
    # 断言对 x 的导数应该等于 2 * 对角矩阵 x
    assert expr.diff(x).doit() == 2*DiagMatrix(x)

    # 计算 Hadamard 幂次表达式
    expr = hadamard_power(x, S.Half)
    # 断言对 x 的导数应该等于 S.Half * 对角矩阵 (x 的负半幂次)
    assert expr.diff(x) == S.Half*DiagMatrix(hadamard_power(x, Rational(-1, 2)))

    # 计算复合的 Hadamard 幂次表达式
    expr = hadamard_power(a.T*X*b, 2)
    # 断言对 X 的导数应该等于 2 * (a * a^T * X * b * b^T)
    assert expr.diff(X) == 2*a*a.T*X*b*b.T

    # 计算复合的 Hadamard 幂次表达式
    expr = hadamard_power(a.T*X*b, S.Half)
    # 断言对 X 的导数应该等于 a / (2 * sqrt(a^T * X * b)) * b^T
    assert expr.diff(X) == a/(2*sqrt(a.T*X*b))*b.T
```