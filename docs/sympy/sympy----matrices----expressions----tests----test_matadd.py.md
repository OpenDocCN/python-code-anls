# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_matadd.py`

```
# 导入 SymPy 库中的矩阵表达式和异常处理相关模块
from sympy.matrices.expressions import MatrixSymbol, MatAdd, MatPow, MatMul
from sympy.matrices.expressions.special import GenericZeroMatrix, ZeroMatrix
from sympy.matrices.exceptions import ShapeError
from sympy.matrices import eye, ImmutableMatrix
from sympy.core import Add, Basic, S
from sympy.core.add import add
from sympy.testing.pytest import XFAIL, raises

# 创建两个2x2的矩阵符号 X 和 Y
X = MatrixSymbol('X', 2, 2)
Y = MatrixSymbol('Y', 2, 2)

# 定义测试函数，用于测试矩阵表达式的评估
def test_evaluate():
    # 断言 MatAdd 对象与 add 函数的结果相等，并且进行评估
    assert MatAdd(X, X, evaluate=True) == add(X, X, evaluate=True) == MatAdd(X, X).doit()

# 定义测试函数，用于测试矩阵表达式的排序键
def test_sort_key():
    # 断言 MatAdd 对象和 add 函数排序后的参数相同，并进行评估
    assert MatAdd(Y, X).doit().args == add(Y, X).doit().args == (X, Y)

# 定义测试函数，用于测试矩阵表达式的 sympify 方法
def test_matadd_sympify():
    # 断言 MatAdd 对象中的第一个参数是 Basic 类型的对象
    assert isinstance(MatAdd(eye(1), eye(1)).args[0], Basic)
    # 断言 add 函数中的第一个参数是 Basic 类型的对象
    assert isinstance(add(eye(1), eye(1)).args[0], Basic)

# 定义测试函数，用于测试矩阵相加的结果
def test_matadd_of_matrices():
    # 断言对于给定的矩阵表达式，MatAdd 对象的评估结果为不可变矩阵的相加结果
    assert MatAdd(eye(2), 4*eye(2), eye(2)).doit() == ImmutableMatrix(6*eye(2))
    # 断言对于给定的矩阵表达式，add 函数的评估结果为不可变矩阵的相加结果
    assert add(eye(2), 4*eye(2), eye(2)).doit() == ImmutableMatrix(6*eye(2))

# 定义测试函数，用于测试矩阵表达式的 doit 方法对参数的处理
def test_doit_args():
    A = ImmutableMatrix([[1, 2], [3, 4]])
    B = ImmutableMatrix([[2, 3], [4, 5]])
    # 断言对于给定的 MatAdd 对象，其 doit 方法的评估结果等于 A + B**2
    assert MatAdd(A, MatPow(B, 2)).doit() == A + B**2
    # 断言对于给定的 MatAdd 对象，其 doit 方法的评估结果等于 A + A*B
    assert MatAdd(A, MatMul(A, B)).doit() == A + A*B
    # 断言对于给定的 MatAdd 对象，其 doit 方法的评估结果等于 3*A + A*B + B
    assert (MatAdd(A, X, MatMul(A, B), Y, MatAdd(2*A, B)).doit() ==
            add(A, X, MatMul(A, B), Y, add(2*A, B)).doit() ==
            MatAdd(3*A + A*B + B, X, Y))

# 定义测试函数，用于测试 MatAdd 对象的 identity 属性
def test_generic_identity():
    # 断言 MatAdd 对象的 identity 属性等于 GenericZeroMatrix
    assert MatAdd.identity == GenericZeroMatrix()
    # 断言 MatAdd 对象的 identity 属性不等于 S.Zero
    assert MatAdd.identity != S.Zero

# 定义测试函数，用于测试 ZeroMatrix 的相加
def test_zero_matrix_add():
    # 断言两个 ZeroMatrix 相加的结果等于一个同样大小的 ZeroMatrix
    assert Add(ZeroMatrix(2, 2), ZeroMatrix(2, 2)) == ZeroMatrix(2, 2)

# 定义测试函数，用于测试带有标量的矩阵相加的异常处理
@XFAIL
def test_matrix_Add_with_scalar():
    # 断言对于标量和 ZeroMatrix 相加时会引发 TypeError 异常
    raises(TypeError, lambda: Add(0, ZeroMatrix(2, 2)))

# 定义测试函数，用于测试矩阵表达式的形状异常处理
def test_shape_error():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 3, 3)
    # 断言对于不兼容的矩阵形状 MatAdd(A, B) 会引发 ShapeError 异常
    raises(ShapeError, lambda: MatAdd(A, B))

    A = MatrixSymbol('A', 3, 2)
    # 断言对于不兼容的矩阵形状 MatAdd(A, B) 会引发 ShapeError 异常
    raises(ShapeError, lambda: MatAdd(A, B))
```