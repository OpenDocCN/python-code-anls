# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_special.py`

```
# 导入特定的符号和函数类别，从 sympy.core.add 模块中导入 Add 类
from sympy.core.add import Add
# 从 sympy.core.expr 模块中导入 unchanged 函数
from sympy.core.expr import unchanged
# 从 sympy.core.mul 模块中导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core.symbol 模块中导入 symbols 函数
from sympy.core.symbol import symbols
# 从 sympy.core.relational 模块中导入 Eq 类
from sympy.core.relational import Eq
# 从 sympy.concrete.summations 模块中导入 Sum 类
from sympy.concrete.summations import Sum
# 从 sympy.functions.elementary.complexes 模块中导入 im 和 re 函数
from sympy.functions.elementary.complexes import im, re
# 从 sympy.functions.elementary.piecewise 模块中导入 Piecewise 类
from sympy.functions.elementary.piecewise import Piecewise
# 从 sympy.matrices.immutable 模块中导入 ImmutableDenseMatrix 类
from sympy.matrices.immutable import ImmutableDenseMatrix
# 从 sympy.matrices.expressions.matexpr 模块中导入 MatrixSymbol 类
from sympy.matrices.expressions.matexpr import MatrixSymbol
# 从 sympy.matrices.expressions.matadd 模块中导入 MatAdd 类
from sympy.matrices.expressions.matadd import MatAdd
# 从 sympy.matrices.expressions.special 模块中导入 ZeroMatrix, GenericZeroMatrix, Identity,
# GenericIdentity, OneMatrix 类
from sympy.matrices.expressions.special import (
    ZeroMatrix, GenericZeroMatrix, Identity, GenericIdentity, OneMatrix)
# 从 sympy.matrices.expressions.matmul 模块中导入 MatMul 类
from sympy.matrices.expressions.matmul import MatMul
# 从 sympy.testing.pytest 模块中导入 raises 函数
from sympy.testing.pytest import raises


# 定义测试函数，用于测试 ZeroMatrix 的创建
def test_zero_matrix_creation():
    # 断言 ZeroMatrix(2, 2) 的行为不变
    assert unchanged(ZeroMatrix, 2, 2)
    # 断言 ZeroMatrix(0, 0) 的行为不变
    assert unchanged(ZeroMatrix, 0, 0)
    # 测试 ZeroMatrix(-1, 2) 是否引发 ValueError 异常
    raises(ValueError, lambda: ZeroMatrix(-1, 2))
    # 测试 ZeroMatrix(2.0, 2) 是否引发 ValueError 异常
    raises(ValueError, lambda: ZeroMatrix(2.0, 2))
    # 测试 ZeroMatrix(2j, 2) 是否引发 ValueError 异常
    raises(ValueError, lambda: ZeroMatrix(2j, 2))
    # 测试 ZeroMatrix(2, -1) 是否引发 ValueError 异常
    raises(ValueError, lambda: ZeroMatrix(2, -1))
    # 测试 ZeroMatrix(2, 2.0) 是否引发 ValueError 异常
    raises(ValueError, lambda: ZeroMatrix(2, 2.0))
    # 测试 ZeroMatrix(2, 2j) 是否引发 ValueError 异常
    raises(ValueError, lambda: ZeroMatrix(2, 2j))

    # 定义符号 n
    n = symbols('n')
    # 断言 ZeroMatrix(n, n) 的行为不变
    assert unchanged(ZeroMatrix, n, n)
    # 定义非整数符号 n
    n = symbols('n', integer=False)
    # 测试 ZeroMatrix(n, n) 是否引发 ValueError 异常
    raises(ValueError, lambda: ZeroMatrix(n, n))
    # 定义负数符号 n
    n = symbols('n', negative=True)
    # 测试 ZeroMatrix(n, n) 是否引发 ValueError 异常
    raises(ValueError, lambda: ZeroMatrix(n, n))


# 定义测试函数，用于测试 GenericZeroMatrix 的行为
def test_generic_zero_matrix():
    # 创建 GenericZeroMatrix 对象 z
    z = GenericZeroMatrix()
    # 定义整数符号 n
    n = symbols('n', integer=True)
    # 创建 n x n 的 MatrixSymbol 对象 A
    A = MatrixSymbol("A", n, n)

    # 断言 z 等于自身
    assert z == z
    # 断言 z 不等于 A
    assert z != A
    # 断言 A 不等于 z

    # 断言 z 是 ZeroMatrix 类型
    assert z.is_ZeroMatrix

    # 测试 z.shape 是否引发 TypeError 异常
    raises(TypeError, lambda: z.shape)
    # 测试 z.rows 是否引发 TypeError 异常
    raises(TypeError, lambda: z.rows)
    # 测试 z.cols 是否引发 TypeError 异常

    # 断言 MatAdd() 等于 z
    assert MatAdd() == z
    # 断言 MatAdd(z, A) 等于 MatAdd(A)
    assert MatAdd(z, A) == MatAdd(A)
    # 确保 z 是可散列的
    hash(z)


# 定义测试函数，用于测试 Identity 的创建
def test_identity_matrix_creation():
    # 断言 Identity(2) 的行为符合预期
    assert Identity(2)
    # 断言 Identity(0) 的行为符合预期
    assert Identity(0)
    # 测试 Identity(-1) 是否引发 ValueError 异常
    raises(ValueError, lambda: Identity(-1))
    # 测试 Identity(2.0) 是否引发 ValueError 异常
    raises(ValueError, lambda: Identity(2.0))
    # 测试 Identity(2j) 是否引发 ValueError 异常
    raises(ValueError, lambda: Identity(2j))

    # 定义符号 n
    n = symbols('n')
    # 断言 Identity(n) 的行为符合预期
    assert Identity(n)
    # 定义非整数符号 n
    n = symbols('n', integer=False)
    # 测试 Identity(n) 是否引发 ValueError 异常
    raises(ValueError, lambda: Identity(n))
    # 定义负数符号 n
    n = symbols('n', negative=True)
    # 测试 Identity(n) 是否引发 ValueError 异常
    raises(ValueError, lambda: Identity(n))


# 定义测试函数，用于测试 GenericIdentity 的行为
def test_generic_identity():
    # 创建 GenericIdentity 对象 I
    I = GenericIdentity()
    # 定义整数符号 n
    n = symbols('n', integer=True)
    # 创建 n x n 的 MatrixSymbol 对象 A
    A = MatrixSymbol("A", n, n)

    # 断言 I 等于自身
    assert I == I
    # 断言 I 不等于 A
    assert I != A
    # 断言 A 不等于 I

    # 断言 I 是 Identity 类型
    assert I.is_Identity
    # 断言 I 的逆矩阵为自身
    assert I**-1 == I

    # 测试 I.shape 是否引发 TypeError 异常
    raises(TypeError, lambda: I.shape)
    # 测试 I.rows 是否引发 TypeError 异常
    raises(TypeError, lambda: I.rows)
    # 测试 I.cols 是否引发 TypeError 异常

    # 断言 MatMul() 等于 I
    assert MatMul() == I
    # 断言 MatMul(I, A) 等于 MatMul(A)
    assert MatMul(I, A) == MatMul(A)
    # 确保 I 是可散列的
    hash(I)


# 定义测试函数，用于测试 OneMatrix 的创建
def test_one_matrix_creation():
    # 断言 OneMatrix(2, 2) 的行为符合预期
    assert OneMatrix(2, 2)
    # 断言 OneMatrix(0, 0) 的行为符合预期
    assert OneMatrix(0, 0)
    # 断言 Eq(OneMatrix(1, 1), Identity(1)) 的行为符合预期
    assert Eq(OneMatrix(1, 1), Identity(1))
    # 测试 OneMatrix(-1, 2) 是否引发 ValueError 异常
    raises(ValueError, lambda: OneMatrix(-1, 2))
    # 测试 OneMatrix(2.0, 2) 是否引发 ValueError 异常
    raises(ValueError, lambda: OneMatrix(2.0, 2))
    # 测试 OneMatrix(2j, 2) 是否引发 ValueError 异常
    raises(ValueError, lambda: OneMatrix(2j, 2))


这段代码包含了多个函数和类的定义，主要用于测试不同
    # 调用 raises 函数，验证 OneMatrix 对象在特定参数下是否会引发 ValueError 异常
    raises(ValueError, lambda: OneMatrix(2, -1))
    
    # 调用 raises 函数，验证 OneMatrix 对象在特定参数下是否会引发 ValueError 异常
    raises(ValueError, lambda: OneMatrix(2, 2.0))
    
    # 调用 raises 函数，验证 OneMatrix 对象在特定参数下是否会引发 ValueError 异常
    raises(ValueError, lambda: OneMatrix(2, 2j))
    
    # 创建符号 n，表示一个未指定类型的符号变量
    n = symbols('n')
    
    # 断言 OneMatrix 对象能够正确创建，使用 n 作为行和列的参数
    assert OneMatrix(n, n)
    
    # 创建符号 n，表示一个非整数的符号变量
    n = symbols('n', integer=False)
    
    # 调用 raises 函数，验证 OneMatrix 对象在特定参数下是否会引发 ValueError 异常
    raises(ValueError, lambda: OneMatrix(n, n))
    
    # 创建符号 n，表示一个负数的符号变量
    n = symbols('n', negative=True)
    
    # 调用 raises 函数，验证 OneMatrix 对象在特定参数下是否会引发 ValueError 异常
    raises(ValueError, lambda: OneMatrix(n, n))
# 定义一个名为 test_ZeroMatrix 的测试函数
def test_ZeroMatrix():
    # 声明两个符号变量 n 和 m，表示矩阵的行数和列数
    n, m = symbols('n m', integer=True)
    # 创建一个 n 行 m 列的符号矩阵 A
    A = MatrixSymbol('A', n, m)
    # 创建一个零矩阵 Z，其大小为 n 行 m 列
    Z = ZeroMatrix(n, m)

    # 断言：A 加上 Z 等于 A 自身
    assert A + Z == A
    # 断言：A 乘以 Z 的转置等于一个 n 阶零矩阵
    assert A*Z.T == ZeroMatrix(n, n)
    # 断言：Z 乘以 A 的转置等于一个 n 阶零矩阵
    assert Z*A.T == ZeroMatrix(n, n)
    # 断言：A 减去自身等于一个与 A 相同大小的零矩阵
    assert A - A == ZeroMatrix(*A.shape)

    # 断言：Z 本身被视为真值（非零）
    assert Z

    # 断言：Z 的转置是一个 m 行 n 列的零矩阵
    assert Z.transpose() == ZeroMatrix(m, n)
    # 断言：Z 的共轭等于它自身
    assert Z.conjugate() == Z
    # 断言：Z 的伴随等于一个 m 行 n 列的零矩阵
    assert Z.adjoint() == ZeroMatrix(m, n)
    # 断言：Z 的实部等于它自身
    assert re(Z) == Z
    # 断言：Z 的虚部等于一个 n 行 m 列的零矩阵
    assert im(Z) == Z

    # 断言：n 阶零矩阵的零次幂等于 n 阶单位矩阵
    assert ZeroMatrix(n, n)**0 == Identity(n)
    # 断言：一个 3 阶 3 列的零矩阵转换为显式的零矩阵对象
    assert ZeroMatrix(3, 3).as_explicit() == ImmutableDenseMatrix.zeros(3, 3)


# 定义一个名为 test_ZeroMatrix_doit 的测试函数
def test_ZeroMatrix_doit():
    # 声明一个符号变量 n，表示矩阵的行数
    n = symbols('n', integer=True)
    # 创建一个零矩阵，其行数为 2*n，列数为 n
    Znn = ZeroMatrix(Add(n, n, evaluate=False), n)
    # 断言：Znn 的行数为 Add(n, n) 对象
    assert isinstance(Znn.rows, Add)
    # 断言：Znn 的 doit() 方法返回一个行数为 2*n，列数为 n 的零矩阵
    assert Znn.doit() == ZeroMatrix(2*n, n)
    # 断言：Znn 的 doit() 方法返回的零矩阵的行数为 Mul(n, 2) 对象
    assert isinstance(Znn.doit().rows, Mul)


# 定义一个名为 test_OneMatrix 的测试函数
def test_OneMatrix():
    # 声明两个符号变量 n 和 m，表示矩阵的行数和列数
    n, m = symbols('n m', integer=True)
    # 创建一个 n 行 m 列的符号矩阵 A
    A = MatrixSymbol('A', n, m)
    # 创建一个单位矩阵 U，其大小为 n 行 m 列
    U = OneMatrix(n, m)

    # 断言：U 的形状为 (n, m)
    assert U.shape == (n, m)
    # 断言：A 加上 U 是一个 Add 对象
    assert isinstance(A + U, Add)
    # 断言：U 的转置是一个 m 行 n 列的单位矩阵
    assert U.transpose() == OneMatrix(m, n)
    # 断言：U 的共轭等于它自身
    assert U.conjugate() == U
    # 断言：U 的伴随是一个 m 行 n 列的单位矩阵
    assert U.adjoint() == OneMatrix(m, n)
    # 断言：U 的实部等于它自身
    assert re(U) == U
    # 断言：U 的虚部是一个 n 行 m 列的零矩阵
    assert im(U) == ZeroMatrix(n, m)

    # 断言：n 阶单位矩阵的零次幂等于 n 阶单位矩阵
    assert OneMatrix(n, n) ** 0 == Identity(n)

    # 断言：一个 n 阶单位矩阵的元素 (1, 2) 等于 1
    U = OneMatrix(n, n)
    assert U[1, 2] == 1

    # 断言：一个 2 行 3 列的单位矩阵转换为显式的全一矩阵对象
    U = OneMatrix(2, 3)
    assert U.as_explicit() == ImmutableDenseMatrix.ones(2, 3)


# 定义一个名为 test_OneMatrix_doit 的测试函数
def test_OneMatrix_doit():
    # 声明一个符号变量 n，表示矩阵的行数
    n = symbols('n', integer=True)
    # 创建一个单位矩阵，其行数为 2*n，列数为 n
    Unn = OneMatrix(Add(n, n, evaluate=False), n)
    # 断言：Unn 的行数为 Add(n, n) 对象
    assert isinstance(Unn.rows, Add)
    # 断言：Unn 的 doit() 方法返回一个行数为 2*n，列数为 n 的单位矩阵
    assert Unn.doit() == OneMatrix(2 * n, n)
    # 断言：Unn 的 doit() 方法返回的单位矩阵的行数为 Mul(n, 2) 对象
    assert isinstance(Unn.doit().rows, Mul)


# 定义一个名为 test_OneMatrix_mul 的测试函数
def test_OneMatrix_mul():
    # 声明三个符号变量 n、m 和 k，表示矩阵的行数、列数和宽度
    n, m, k = symbols('n m k', integer=True)
    # 创建一个 n 行 1 列的符号矩阵 w
    w = MatrixSymbol('w', n, 1)
    # 断言：一个 n 行 m 列的全一矩阵乘以一个 m 行 k 列的全一矩阵等于一个 n 行 k 列的全一矩阵乘以 m
    assert OneMatrix(n, m) * OneMatrix(m, k) == OneMatrix(n, k) * m
    # 断言：w 乘以一个 1 行 1 列的全一矩阵等于 w 自身
    assert w * OneMatrix(1, 1) == w
    # 断言：一个 1 行 1 列的全一矩阵乘以 w 的转置等于 w 的转置
    assert OneMatrix(1, 1) * w.T == w.T


# 定义一个名为 test_Identity 的测试函数
def test_Identity():
    # 声明两个符号变量 n 和 m，表示矩阵的行数和列数
    n, m = symbols('n m', integer=True)
    # 创建一个 n 行 m 列的符号矩阵 A
    A = MatrixSymbol('A', n, m)
    # 声明两个符号变量 i 和 j
    i, j = symbols('i j')

    # 创建一个 n 阶单位矩阵 In 和一个 m 阶单位矩阵 Im
    In = Identity(n)
    Im = Identity(m)

    # 断言：A 乘以 Im 等于 A
    assert A*Im == A
    # 断言：In
    # 导入符号运算模块，声明符号变量 n 为整数类型
    n = symbols('n', integer=True)
    
    # 创建一个表示矩阵的标识符 Inn，它是由一个加法表达式组成，表达式中是两个 n 的加法，但不进行求值
    Inn = Identity(Add(n, n, evaluate=False))
    
    # 断言 Inn 的行数是一个加法表达式，验证 Inn 对象的类型
    assert isinstance(Inn.rows, Add)
    
    # 对 Inn 进行求值操作，期望结果是一个表示 2n 的标识符对象
    assert Inn.doit() == Identity(2*n)
    
    # 断言 Inn 执行 doit 后的结果，其行数是一个乘法表达式，验证对象类型
    assert isinstance(Inn.doit().rows, Mul)
```