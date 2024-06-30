# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_indexing.py`

```
# 导入 Sympy 库中的具体模块和函数
from sympy.concrete.summations import Sum
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import eye
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.hadamard import HadamardPower
from sympy.matrices.expressions.matexpr import (MatrixSymbol,
    MatrixExpr, MatrixElement)
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.special import (ZeroMatrix, Identity,
    OneMatrix)
from sympy.matrices.expressions.trace import Trace, trace
from sympy.matrices.immutable import ImmutableMatrix
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
from sympy.testing.pytest import XFAIL, raises

# 定义整数类型的符号变量
k, l, m, n = symbols('k l m n', integer=True)
i, j = symbols('i j', integer=True)

# 定义符号矩阵变量
W = MatrixSymbol('W', k, l)
X = MatrixSymbol('X', l, m)
Y = MatrixSymbol('Y', l, m)
Z = MatrixSymbol('Z', m, n)

X1 = MatrixSymbol('X1', m, m)
X2 = MatrixSymbol('X2', m, m)
X3 = MatrixSymbol('X3', m, m)
X4 = MatrixSymbol('X4', m, m)

A = MatrixSymbol('A', 2, 2)
B = MatrixSymbol('B', 2, 2)
x = MatrixSymbol('x', 1, 2)
y = MatrixSymbol('x', 2, 1)

# 定义符号索引测试函数
def test_symbolic_indexing():
    # 获取矩阵 X 中位置 (1, 2) 的元素
    x12 = X[1, 2]
    # 断言确保结果字符串包含 '1'、'2'、以及矩阵 X 的名称
    assert all(s in str(x12) for s in ['1', '2', X.name])
    # 我们不关心确切的形式，但确保所有这些特征都存在

# 定义符号加法测试函数
def test_add_index():
    # 断言矩阵 X 和 Y 相加后的元素 (i, j) 等于 X[i, j] + Y[i, j]
    assert (X + Y)[i, j] == X[i, j] + Y[i, j]

# 定义符号乘法测试函数
def test_mul_index():
    # 断言矩阵 A 与列向量 y 的乘积的元素 (0, 0)
    assert (A*y)[0, 0] == A[0, 0]*y[0, 0] + A[0, 1]*y[1, 0]
    # 将 A 和 B 转换为可变矩阵后比较它们的乘积
    assert (A*B).as_mutable() == (A.as_mutable() * B.as_mutable())
    # 定义矩阵 X 和 Y，计算它们乘积的元素 (4, 2)
    X = MatrixSymbol('X', n, m)
    Y = MatrixSymbol('Y', m, k)
    result = (X*Y)[4,2]
    # 断言结果的第一个参数是预期的和
    expected = Sum(X[4, i]*Y[i, 2], (i, 0, m - 1))
    assert result.args[0].dummy_eq(expected.args[0], i)
    # 断言结果的其他参数与预期的其他参数相等
    assert result.args[1][1:] == expected.args[1][1:]

# 定义符号幂运算测试函数
def test_pow_index():
    # 定义矩阵 A 的平方 Q，并断言其元素 (0, 0)
    Q = MatPow(A, 2)
    assert Q[0, 0] == A[0, 0]**2 + A[0, 1]*A[1, 0]
    n = symbols("n")
    # 定义 A 的 n 次幂 Q2，断言其元素 (0, 0)
    Q2 = A**n
    assert Q2[0, 0] == 2*(
            -sqrt((A[0, 0] + A[1, 1])**2 - 4*A[0, 0]*A[1, 1] +
            4*A[0, 1]*A[1, 0])/2 + A[0, 0]/2 + A[1, 1]/2
        )**n * \
        A[0, 1]*A[1, 0]/(
            (sqrt(A[0, 0]**2 - 2*A[0, 0]*A[1, 1] + 4*A[0, 1]*A[1, 0] +
                  A[1, 1]**2) + A[0, 0] - A[1, 1])*
            sqrt(A[0, 0]**2 - 2*A[0, 0]*A[1, 1] + 4*A[0, 1]*A[1, 0] + A[1, 1]**2)
        ) - 2*(
            sqrt((A[0, 0] + A[1, 1])**2 - 4*A[0, 0]*A[1, 1] +
            4*A[0, 1]*A[1, 0])/2 + A[0, 0]/2 + A[1, 1]/2
        )**n * A[0, 1]*A[1, 0]/(
            (-sqrt(A[0, 0]**2 - 2*A[0, 0]*A[1, 1] + 4*A[0, 1]*A[1, 0] +
            A[1, 1]**2) + A[0, 0] - A[1, 1])*
            sqrt(A[0, 0]**2 - 2*A[0, 0]*A[1, 1] + 4*A[0, 1]*A[1, 0] + A[1, 1]**2)
        )

# 定义符号转置测试函数
def test_transpose_index():
    # 断言矩阵 X 的转置的元素 (i, j) 等于 X 的元素 (j, i)
    assert X.T[i, j] == X[j, i]

# 定义符号单位矩阵测试函数
def test_Identity_index():
    # 定义大小为 3 的单位矩阵 I
    I = Identity(3)
    # 断言第一行对角线上的元素都应该等于1
    assert I[0, 0] == I[1, 1] == I[2, 2] == 1
    # 断言第二行的特定元素应该为0，表明特定位置的对称性
    assert I[1, 0] == I[0, 1] == I[2, 1] == 0
    # 断言索引 i 处的第一列元素具有预期的 delta 范围
    assert I[i, 0].delta_range == (0, 2)
    # 使用 lambda 函数检查访问超出矩阵界限的索引是否会引发 IndexError 异常
    raises(IndexError, lambda: I[3, 3])
def test_block_index():
    # 创建一个 3x3 的单位矩阵
    I = Identity(3)
    # 创建一个 3x3 的零矩阵
    Z = ZeroMatrix(3, 3)
    # 创建一个块矩阵，包含两个单位矩阵
    B = BlockMatrix([[I, I], [I, I]])
    # 创建一个 3x3 的不可变矩阵
    e3 = ImmutableMatrix(eye(3))
    # 创建一个块矩阵，包含两个 3x3 的不可变矩阵
    BB = BlockMatrix([[e3, e3], [e3, e3]])
    # 断言块矩阵 B 的特定元素为 1
    assert B[0, 0] == B[3, 0] == B[0, 3] == B[3, 3] == 1
    # 断言块矩阵 B 的特定元素为 0
    assert B[4, 3] == B[5, 1] == 0

    # 重新定义 BB，确保其与 B 相等
    BB = BlockMatrix([[e3, e3], [e3, e3]])
    # 断言 B 和 BB 的显式表示相等
    assert B.as_explicit() == BB.as_explicit()

    # 创建一个块对角矩阵 BI
    BI = BlockMatrix([[I, Z], [Z, I]])
    # 断言 BI 的显式表示等于一个 6x6 的单位矩阵
    assert BI.as_explicit().equals(eye(6))


def test_block_index_symbolic():
    # 注意：这些矩阵可能是零大小的，索引可能是负数，导致所有简单化的假设都无效
    A1 = MatrixSymbol('A1', n, k)
    A2 = MatrixSymbol('A2', n, l)
    A3 = MatrixSymbol('A3', m, k)
    A4 = MatrixSymbol('A4', m, l)
    A = BlockMatrix([[A1, A2], [A3, A4]])
    # 断言 A 的特定索引处是 MatrixElement(A, i, j)，而不是简单的 A1[0, 0] 形式
    assert A[0, 0] == MatrixElement(A, 0, 0)
    assert A[n - 1, k - 1] == A1[n - 1, k - 1]
    assert A[n, k] == A4[0, 0]
    assert A[n + m - 1, 0] == MatrixElement(A, n + m - 1, 0)
    assert A[0, k + l - 1] == MatrixElement(A, 0, k + l - 1)
    assert A[n + m - 1, k + l - 1] == MatrixElement(A, n + m - 1, k + l - 1)
    assert A[i, j] == MatrixElement(A, i, j)
    assert A[n + i, k + j] == MatrixElement(A, n + i, k + j)
    assert A[n - i - 1, k - j - 1] == MatrixElement(A, n - i - 1, k - j - 1)


def test_block_index_symbolic_nonzero():
    # 所有来自 test_block_index_symbolic() 的无效简化，在所有矩阵都有非零大小且所有索引为非负时变为有效
    k, l, m, n = symbols('k l m n', integer=True, positive=True)
    i, j = symbols('i j', integer=True, nonnegative=True)
    A1 = MatrixSymbol('A1', n, k)
    A2 = MatrixSymbol('A2', n, l)
    A3 = MatrixSymbol('A3', m, k)
    A4 = MatrixSymbol('A4', m, l)
    A = BlockMatrix([[A1, A2], [A3, A4]])
    assert A[0, 0] == A1[0, 0]
    assert A[n + m - 1, 0] == A3[m - 1, 0]
    assert A[0, k + l - 1] == A2[0, l - 1]
    assert A[n + m - 1, k + l - 1] == A4[m - 1, l - 1]
    assert A[i, j] == MatrixElement(A, i, j)
    assert A[n + i, k + j] == A4[i, j]
    assert A[n - i - 1, k - j - 1] == A1[n - i - 1, k - j - 1]
    assert A[2 * n, 2 * k] == A4[n, k]


def test_block_index_large():
    n, m, k = symbols('n m k', integer=True, positive=True)
    i = symbols('i', integer=True, nonnegative=True)
    A1 = MatrixSymbol('A1', n, n)
    A2 = MatrixSymbol('A2', n, m)
    A3 = MatrixSymbol('A3', n, k)
    A4 = MatrixSymbol('A4', m, n)
    A5 = MatrixSymbol('A5', m, m)
    A6 = MatrixSymbol('A6', m, k)
    A7 = MatrixSymbol('A7', k, n)
    A8 = MatrixSymbol('A8', k, m)
    A9 = MatrixSymbol('A9', k, k)
    A = BlockMatrix([[A1, A2, A3], [A4, A5, A6], [A7, A8, A9]])
    # 断言 A 的特定索引处是 MatrixElement(A, n + i, n + i)
    assert A[n + i, n + i] == MatrixElement(A, n + i, n + i)
    # 创建一个符号矩阵 A1，其维度为 n x 1
    A1 = MatrixSymbol('A1', n, 1)
    # 创建另一个符号矩阵 A2，其维度为 m x 1
    A2 = MatrixSymbol('A2', m, 1)
    # 创建一个分块矩阵 A，由 A1 和 A2 组成，分别位于两行
    A = BlockMatrix([[A1], [A2]])
    # 断言：A 中索引为 (2 * n, 0) 的元素应该等于 A2 中索引为 (n, 0) 的元素
    assert A[2 * n, 0] == A2[n, 0]
def test_slicing():
    A.as_explicit()[0, :]  # does not raise an error
    # 使用 as_explicit 方法将矩阵 A 转换为显式表达式，然后取第一行的所有列，这里不会引发错误


def test_errors():
    raises(IndexError, lambda: Identity(2)[1, 2, 3, 4, 5])
    # 断言在使用 Identity(2) 这个矩阵时，使用了超出索引范围的索引 [1, 2, 3, 4, 5]，会抛出 IndexError 异常
    raises(IndexError, lambda: Identity(2)[[1, 2, 3, 4, 5]])
    # 断言在使用 Identity(2) 这个矩阵时，使用了超出索引范围的索引 [[1, 2, 3, 4, 5]]，会抛出 IndexError 异常


def test_matrix_expression_to_indices():
    i, j = symbols("i, j")
    i1, i2, i3 = symbols("i_1:4")

    def replace_dummies(expr):
        repl = {i: Symbol(i.name) for i in expr.atoms(Dummy)}
        return expr.xreplace(repl)

    expr = W*X*Z
    assert replace_dummies(expr._entry(i, j)) == \
        Sum(W[i, i1]*X[i1, i2]*Z[i2, j], (i1, 0, l-1), (i2, 0, m-1))
    # 断言使用 replace_dummies 函数替换表达式中的虚拟变量后，表达式中的项等于对应的求和表达式
    assert MatrixExpr.from_index_summation(expr._entry(i, j)) == expr
    # 断言使用 from_index_summation 函数从表达式的索引求和形式重建出原始表达式，结果应该与原始表达式相同

    expr = Z.T*X.T*W.T
    assert replace_dummies(expr._entry(i, j)) == \
        Sum(W[j, i2]*X[i2, i1]*Z[i1, i], (i1, 0, m-1), (i2, 0, l-1))
    # 断言使用 replace_dummies 函数替换表达式中的虚拟变量后，表达式中的项等于对应的求和表达式
    assert MatrixExpr.from_index_summation(expr._entry(i, j), i) == expr
    # 断言使用 from_index_summation 函数从表达式的索引求和形式重建出原始表达式，结果应该与原始表达式相同

    expr = W*X*Z + W*Y*Z
    assert replace_dummies(expr._entry(i, j)) == \
        Sum(W[i, i1]*X[i1, i2]*Z[i2, j], (i1, 0, l-1), (i2, 0, m-1)) +\
        Sum(W[i, i1]*Y[i1, i2]*Z[i2, j], (i1, 0, l-1), (i2, 0, m-1))
    # 断言使用 replace_dummies 函数替换表达式中的虚拟变量后，表达式中的项等于对应的求和表达式
    assert MatrixExpr.from_index_summation(expr._entry(i, j)) == expr
    # 断言使用 from_index_summation 函数从表达式的索引求和形式重建出原始表达式，结果应该与原始表达式相同

    expr = 2*W*X*Z + 3*W*Y*Z
    assert replace_dummies(expr._entry(i, j)) == \
        2*Sum(W[i, i1]*X[i1, i2]*Z[i2, j], (i1, 0, l-1), (i2, 0, m-1)) +\
        3*Sum(W[i, i1]*Y[i1, i2]*Z[i2, j], (i1, 0, l-1), (i2, 0, m-1))
    # 断言使用 replace_dummies 函数替换表达式中的虚拟变量后，表达式中的项等于对应的求和表达式
    assert MatrixExpr.from_index_summation(expr._entry(i, j)) == expr
    # 断言使用 from_index_summation 函数从表达式的索引求和形式重建出原始表达式，结果应该与原始表达式相同

    expr = W*(X + Y)*Z
    assert replace_dummies(expr._entry(i, j)) == \
            Sum(W[i, i1]*(X[i1, i2] + Y[i1, i2])*Z[i2, j], (i1, 0, l-1), (i2, 0, m-1))
    # 断言使用 replace_dummies 函数替换表达式中的虚拟变量后，表达式中的项等于对应的求和表达式
    assert MatrixExpr.from_index_summation(expr._entry(i, j)) == expr
    # 断言使用 from_index_summation 函数从表达式的索引求和形式重建出原始表达式，结果应该与原始表达式相同

    expr = A*B**2*A
    #assert replace_dummies(expr._entry(i, j)) == \
    #        Sum(A[i, i1]*B[i1, i2]*B[i2, i3]*A[i3, j], (i1, 0, 1), (i2, 0, 1), (i3, 0, 1))
    # 注释部分：替换虚拟变量后，表达式中的项应该等于对应的求和表达式

    expr = (X1*X2 + X2*X1)*X3
    assert replace_dummies(expr._entry(i, j)) == \
           Sum((Sum(X1[i, i2] * X2[i2, i1], (i2, 0, m - 1)) + Sum(X1[i3, i1] * X2[i, i3], (i3, 0, m - 1))) * X3[
               i1, j], (i1, 0, m - 1))
    # 断言使用 replace_dummies 函数替换表达式中的虚拟变量后，表达式中的项等于对应的求和表达式


def test_matrix_expression_from_index_summation():
    from sympy.abc import a,b,c,d
    A = MatrixSymbol("A", k, k)
    B = MatrixSymbol("B", k, k)
    C = MatrixSymbol("C", k, k)
    w1 = MatrixSymbol("w1", k, 1)

    i0, i1, i2, i3, i4 = symbols("i0:5", cls=Dummy)

    expr = Sum(W[a,b]*X[b,c]*Z[c,d], (b, 0, l-1), (c, 0, m-1))
    assert MatrixExpr.from_index_summation(expr, a) == W*X*Z
    # 断言从索引求和形式的表达式 expr 中恢复出原始表达式，结果应该等于 W*X*Z

    expr = Sum(W.T[b,a]*X[b,c]*Z[c,d], (b, 0, l-1), (c, 0, m-1))
    assert MatrixExpr.from_index_summation(expr, a) == W*X*Z
    # 断言从索引求和形式的表达式 expr 中恢复出原始表达式，结果应该等于 W*X*Z

    expr = Sum(A[b, a]*B[b, c]*C[c, d], (b, 0, k-1), (c, 0, k-1))
    assert MatrixSymbol.from_index_summation(expr, a) == A.T*B*C
    # 断言从索引求和形式的表达式 expr 中恢复出原始矩阵符号，结果应该等于 A.T*B*C

    expr = Sum(A[b, a]*B[c, b]*C[c, d], (b, 0, k-1), (c, 0, k-1))
    assert MatrixSymbol.from_index_summation(expr, a) == A.T*B.T*C
    # 断言从索引求和形式的表达
    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(C[c, d]*A[b, a]*B[c, b], (b, 0, k-1), (c, 0, k-1))
    # 使用 `MatrixSymbol.from_index_summation` 函数将 `expr` 转换为矩阵符号表示，并断言其结果等于 `A.T*B.T*C`
    assert MatrixSymbol.from_index_summation(expr, a) == A.T*B.T*C

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(A[a, b] + B[a, b], (a, 0, k-1), (b, 0, k-1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `OneMatrix(1, k)*A*OneMatrix(k, 1) + OneMatrix(1, k)*B*OneMatrix(k, 1)`
    assert MatrixExpr.from_index_summation(expr, a) == OneMatrix(1, k)*A*OneMatrix(k, 1) + OneMatrix(1, k)*B*OneMatrix(k, 1)

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(A[a, b]**2, (a, 0, k - 1), (b, 0, k - 1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `Trace(A * A.T)`
    assert MatrixExpr.from_index_summation(expr, a) == Trace(A * A.T)

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(A[a, b]**3, (a, 0, k - 1), (b, 0, k - 1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `Trace(HadamardPower(A.T, 2) * A)`
    assert MatrixExpr.from_index_summation(expr, a) == Trace(HadamardPower(A.T, 2) * A)

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum((A[a, b] + B[a, b])*C[b, c], (b, 0, k-1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `(A+B)*C`
    assert MatrixExpr.from_index_summation(expr, a) == (A+B)*C

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum((A[a, b] + B[b, a])*C[b, c], (b, 0, k-1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `(A+B.T)*C`
    assert MatrixExpr.from_index_summation(expr, a) == (A+B.T)*C

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(A[a, b]*A[b, c]*A[c, d], (b, 0, k-1), (c, 0, k-1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `A**3`
    assert MatrixExpr.from_index_summation(expr, a) == A**3

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(A[a, b]*A[b, c]*B[c, d], (b, 0, k-1), (c, 0, k-1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `A**2*B`
    assert MatrixExpr.from_index_summation(expr, a) == A**2*B

    # 解析矩阵的迹：

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(A[a, a], (a, 0, k-1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `trace(A)`
    assert MatrixExpr.from_index_summation(expr, None) == trace(A)

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(A[a, a]*B[b, c]*C[c, d], (a, 0, k-1), (c, 0, k-1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `trace(A)*B*C`
    assert MatrixExpr.from_index_summation(expr, b) == trace(A)*B*C

    # 检查错误的求和范围（应该引发异常）：

    # Case 1: 求和范围中 b 的范围应为 0 到 l-1，但是现在是 0 到 m
    expr = Sum(W[a,b]*X[b,c]*Z[c,d], (b, 0, l-1), (c, 0, m))
    raises(ValueError, lambda: MatrixExpr.from_index_summation(expr, a))

    # Case 2: 求和范围中 c 的范围应为 0 到 m-1，但是现在是 1 到 m-1
    expr = Sum(W[a,b]*X[b,c]*Z[c,d], (b, 0, l-1), (c, 1, m-1))
    raises(ValueError, lambda: MatrixExpr.from_index_summation(expr, a))

    # 解析嵌套求和：

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(A[a, b]*Sum(B[b, c]*C[c, d], (c, 0, k-1)), (b, 0, k-1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `A*B*C`
    assert MatrixExpr.from_index_summation(expr, a) == A*B*C

    # 测试 Kronecker δ：

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(A[a, b]*KroneckerDelta(b, c)*B[c, d], (b, 0, k-1), (c, 0, k-1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `A*B`
    assert MatrixExpr.from_index_summation(expr, a) == A*B

    # 定义表达式 `expr`，这是一个带有索引求和的矩阵表达式
    expr = Sum(KroneckerDelta(i1, m)*KroneckerDelta(i2, n)*A[i, i1]*A[j, i2], (i1, 0, k-1), (i2, 0, k-1))
    # 使用 `MatrixExpr.from_index_summation` 函数将 `expr` 转换为矩阵表达式，并断言其结果等于 `ArrayTensorProduct(A.T, A)`
    assert MatrixExpr.from_index_summation(expr, m) == ArrayTensorProduct(A.T, A)

    # 测试带编号的索
```