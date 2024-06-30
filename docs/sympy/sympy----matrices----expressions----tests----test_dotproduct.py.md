# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_dotproduct.py`

```
# 从 sympy 库中导入所需模块和类
from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.matrices import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.dotproduct import DotProduct
from sympy.testing.pytest import raises

# 创建 Matrix 对象 A，形状为 3x1，数据为 [1, 2, 3]
A = Matrix(3, 1, [1, 2, 3])
# 创建 Matrix 对象 B，形状为 3x1，数据为 [1, 3, 5]
B = Matrix(3, 1, [1, 3, 5])
# 创建 Matrix 对象 C，形状为 4x1，数据为 [1, 2, 4, 5]
C = Matrix(4, 1, [1, 2, 4, 5])
# 创建 Matrix 对象 D，形状为 2x2，数据为 [1, 2, 3, 4]
D = Matrix(2, 2, [1, 2, 3, 4])

# 定义测试函数 test_dotproduct，用于测试 DotProduct 类的功能
def test_docproduct():
    # 断言：计算 A 和 B 的点积，应该得到结果 22
    assert DotProduct(A, B).doit() == 22
    # 断言：计算 A 的转置和 B 的点积，应该得到结果 22
    assert DotProduct(A.T, B).doit() == 22
    # 断言：计算 A 和 B 的转置的点积，应该得到结果 22
    assert DotProduct(A, B.T).doit() == 22
    # 断言：计算 A 和 B 的转置的点积，应该得到结果 22
    assert DotProduct(A.T, B.T).doit() == 22

    # 引发异常：传入非法参数类型进行点积计算
    raises(TypeError, lambda: DotProduct(1, A))
    raises(TypeError, lambda: DotProduct(A, 1))
    raises(TypeError, lambda: DotProduct(A, D))
    raises(TypeError, lambda: DotProduct(D, A))

    # 引发异常：传入形状不匹配的矩阵进行点积计算
    raises(TypeError, lambda: DotProduct(B, C).doit())

# 定义测试函数 test_dotproduct_symbolic，用于测试符号表达式下的 DotProduct 类功能
def test_dotproduct_symbolic():
    # 创建符号矩阵符号 A 和 B，形状为 3x1
    A = MatrixSymbol('A', 3, 1)
    B = MatrixSymbol('B', 3, 1)

    # 创建 DotProduct 对象 dot，表示矩阵 A 和 B 的点积
    dot = DotProduct(A, B)
    # 断言：dot 对象是一个标量
    assert dot.is_scalar == True
    # 断言：对 dot 进行未改变的乘法运算
    assert unchanged(Mul, 2, dot)
    # 断言：修复强制评估的算术表达式，用于包含矩阵表达式的算术
    assert dot * A == (A[0, 0]*B[0, 0] + A[1, 0]*B[1, 0] + A[2, 0]*B[2, 0])*A
```