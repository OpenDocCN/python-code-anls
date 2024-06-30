# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_hadamard.py`

```
# 导入所需的类和函数
from sympy.matrices.dense import Matrix, eye
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import Identity, OneMatrix, ZeroMatrix
from sympy.core import symbols
from sympy.testing.pytest import raises, warns_deprecated_sympy

# 导入矩阵符号相关模块
from sympy.matrices import MatrixSymbol
from sympy.matrices.expressions import (HadamardProduct, hadamard_product, HadamardPower, hadamard_power)

# 定义符号变量
n, m, k = symbols('n,m,k')

# 定义矩阵符号
Z = MatrixSymbol('Z', n, n)
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', n, m)
C = MatrixSymbol('C', m, k)

# 测试函数：测试Hadamard积的基本用法和边界情况
def test_HadamardProduct():
    # 断言：Hadamard积的形状与第一个矩阵A相同
    assert HadamardProduct(A, B, A).shape == A.shape

    # 断言：应抛出TypeError异常，因为第二个参数不是矩阵
    raises(TypeError, lambda: HadamardProduct(A, n))
    # 断言：应抛出TypeError异常，因为第二个参数不是矩阵
    raises(TypeError, lambda: HadamardProduct(A, 1))

    # 断言：验证Hadamard积的特定元素计算是否正确
    assert HadamardProduct(A, 2*B, -A)[1, 1] == \
            -2 * A[1, 1] * B[1, 1] * A[1, 1]

    # 计算混合Hadamard积的形状
    mix = HadamardProduct(Z*A, B)*C
    assert mix.shape == (n, k)

    # 断言：验证Hadamard积的转置结果是否符合预期
    assert set(HadamardProduct(A, B, A).T.args) == {A.T, A.T, B.T}

# 测试函数：验证Hadamard积不满足交换律
def test_HadamardProduct_isnt_commutative():
    assert HadamardProduct(A, B) != HadamardProduct(B, A)

# 测试函数：测试混合索引计算结果
def test_mixed_indexing():
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    Z = MatrixSymbol('Z', 2, 2)

    # 断言：验证混合索引计算的结果是否正确
    assert (X*HadamardProduct(Y, Z))[0, 0] == \
            X[0, 0]*Y[0, 0]*Z[0, 0] + X[0, 1]*Y[1, 0]*Z[1, 0]

# 测试函数：测试Hadamard积的规范化处理
def test_canonicalize():
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    with warns_deprecated_sympy():
        expr = HadamardProduct(X, check=False)
    assert isinstance(expr, HadamardProduct)
    expr2 = expr.doit() # unpack is called
    assert isinstance(expr2, MatrixSymbol)
    Z = ZeroMatrix(2, 2)
    U = OneMatrix(2, 2)
    assert HadamardProduct(Z, X).doit() == Z
    assert HadamardProduct(U, X, X, U).doit() == HadamardPower(X, 2)
    assert HadamardProduct(X, U, Y).doit() == HadamardProduct(X, Y)
    assert HadamardProduct(X, Z, U, Y).doit() == Z

# 测试函数：测试Hadamard积函数的基本用法和边界情况
def test_hadamard():
    m, n, p = symbols('m, n, p', integer=True)
    A = MatrixSymbol('A', m, n)
    B = MatrixSymbol('B', m, n)
    X = MatrixSymbol('X', m, m)
    I = Identity(m)

    # 断言：应抛出TypeError异常，因为没有提供任何参数
    raises(TypeError, lambda: hadamard_product())
    # 断言：验证单个矩阵的Hadamard积结果是否正确
    assert hadamard_product(A) == A
    # 断言：验证Hadamard积的返回类型是否正确
    assert isinstance(hadamard_product(A, B), HadamardProduct)
    # 断言：验证未展开Hadamard积的结果是否正确
    assert hadamard_product(A, B).doit() == hadamard_product(A, B)
    # 断言：验证特定情况下Hadamard积的结果是否正确
    assert hadamard_product(X, I) == HadamardProduct(I, X)
    assert isinstance(hadamard_product(X, I), HadamardProduct)

    # 创建一个列向量a
    a = MatrixSymbol("a", k, 1)
    # 创建表达式expr
    expr = MatAdd(ZeroMatrix(k, 1), OneMatrix(k, 1))
    # 断言：验证Hadamard积和MatAdd的结果是否正确
    expr = HadamardProduct(expr, a)
    assert expr.doit() == a

    # 断言：应抛出ValueError异常，因为没有提供任何参数
    raises(ValueError, lambda: HadamardProduct())

# 测试函数：测试使用显式矩阵的Hadamard积计算结果
def test_hadamard_product_with_explicit_mat():
    A = MatrixSymbol("A", 3, 3).as_explicit()
    B = MatrixSymbol("B", 3, 3).as_explicit()
    X = MatrixSymbol("X", 3, 3)
    expr = hadamard_product(A, B)
    ret = Matrix([i*j for i, j in zip(A, B)]).reshape(3, 3)
    assert expr == ret
    # 调用 hadamard_product 函数计算表达式，其中 A, X, B 是函数的参数
    expr = hadamard_product(A, X, B)
    # 使用 assert 断言验证表达式的结果是否等于 HadamardProduct(ret, X)
    assert expr == HadamardProduct(ret, X)
    
    # 调用 hadamard_product 函数计算表达式，其中参数为单位矩阵 eye(3) 和 A
    expr = hadamard_product(eye(3), A)
    # 使用 assert 断言验证表达式的结果是否等于指定的矩阵
    assert expr == Matrix([[A[0, 0], 0, 0], [0, A[1, 1], 0], [0, 0, A[2, 2]]])
    
    # 调用 hadamard_product 函数计算表达式，其中参数为两个单位矩阵 eye(3)
    expr = hadamard_product(eye(3), eye(3))
    # 使用 assert 断言验证表达式的结果是否等于单位矩阵 eye(3)
    assert expr == eye(3)
# 定义一个测试函数，用于测试 Hadamard 幂运算的各种情况
def test_hadamard_power():
    # 定义符号变量 m, n, p，均为整数类型
    m, n, p = symbols('m, n, p', integer=True)
    # 创建一个矩阵符号 A，大小为 m x n

    A = MatrixSymbol('A', m, n)

    # 断言：对于幂为1的情况，hadamard_power 返回 A 本身
    assert hadamard_power(A, 1) == A
    # 断言：对于幂为2的情况，hadamard_power 返回 HadamardPower 类型的对象
    assert isinstance(hadamard_power(A, 2), HadamardPower)
    # 断言：幂为 n 时，A 的转置的幂等于 A 转置后的幂
    assert hadamard_power(A, n).T == hadamard_power(A.T, n)
    # 断言：返回的 HadamardPower 对象中，第一个元素应为 A[0, 0] 的 n 次幂
    assert hadamard_power(A, n)[0, 0] == A[0, 0]**n
    # 断言：当参数为非矩阵符号时，应抛出 ValueError 异常
    raises(ValueError, lambda: hadamard_power(A, A))


# 定义另一个测试函数，测试 Hadamard 幂运算的显示转换
def test_hadamard_power_explicit():
    # 创建两个 2x2 大小的矩阵符号 A 和 B
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 2, 2)
    # 创建符号变量 a 和 b
    a, b = symbols('a b')

    # 断言：HadamardPower(a, b) 返回 a 的 b 次幂
    assert HadamardPower(a, b) == a**b

    # 断言：HadamardPower(a, B) 的显示转换应与预期的矩阵相匹配
    assert HadamardPower(a, B).as_explicit() == \
        Matrix([
            [a**B[0, 0], a**B[0, 1]],
            [a**B[1, 0], a**B[1, 1]]])

    # 断言：HadamardPower(A, b) 的显示转换应与预期的矩阵相匹配
    assert HadamardPower(A, b).as_explicit() == \
        Matrix([
            [A[0, 0]**b, A[0, 1]**b],
            [A[1, 0]**b, A[1, 1]**b]])

    # 断言：HadamardPower(A, B) 的显示转换应与预期的矩阵相匹配
    assert HadamardPower(A, B).as_explicit() == \
        Matrix([
            [A[0, 0]**B[0, 0], A[0, 1]**B[0, 1]],
            [A[1, 0]**B[1, 0], A[1, 1]**B[1, 1]]])


# 定义一个测试函数，测试形状错误的情况
def test_shape_error():
    # 创建两个不匹配大小的矩阵符号 A 和 B
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 3, 3)
    # 断言：HadamardProduct(A, B) 应抛出 ShapeError 异常
    raises(ShapeError, lambda: HadamardProduct(A, B))
    # 断言：HadamardPower(A, B) 应抛出 ShapeError 异常
    raises(ShapeError, lambda: HadamardPower(A, B))
    
    # 重新定义 A 为另一组不匹配大小的矩阵符号
    A = MatrixSymbol('A', 3, 2)
    # 断言：HadamardProduct(A, B) 应抛出 ShapeError 异常
    raises(ShapeError, lambda: HadamardProduct(A, B))
    # 断言：HadamardPower(A, B) 应抛出 ShapeError 异常
    raises(ShapeError, lambda: HadamardPower(A, B))
```