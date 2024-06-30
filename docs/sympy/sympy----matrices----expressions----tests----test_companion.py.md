# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_companion.py`

```
# 导入必要的符号计算模块和函数
from sympy.core.expr import unchanged
from sympy.core.symbol import Symbol, symbols
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.matrices.expressions.companion import CompanionMatrix
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises

# 定义测试函数，测试 CompanionMatrix 类的创建条件
def test_creation():
    # 定义符号变量 x 和 y
    x = Symbol('x')
    y = Symbol('y')
    # 测试 CompanionMatrix 构造函数对不同输入的异常抛出
    raises(ValueError, lambda: CompanionMatrix(1))
    raises(ValueError, lambda: CompanionMatrix(Poly([1], x)))
    raises(ValueError, lambda: CompanionMatrix(Poly([2, 1], x)))
    raises(ValueError, lambda: CompanionMatrix(Poly(x*y, [x, y])))
    # 测试 CompanionMatrix 构造函数对于给定多项式的正确性
    assert unchanged(CompanionMatrix, Poly([1, 2, 3], x))

# 定义测试函数，测试 CompanionMatrix 对象的形状
def test_shape():
    # 定义符号变量和常数符号
    c0, c1, c2 = symbols('c0:3')
    x = Symbol('x')
    # 断言不同多项式生成的 CompanionMatrix 对象的形状
    assert CompanionMatrix(Poly([1, c0], x)).shape == (1, 1)
    assert CompanionMatrix(Poly([1, c1, c0], x)).shape == (2, 2)
    assert CompanionMatrix(Poly([1, c2, c1, c0], x)).shape == (3, 3)

# 定义测试函数，测试 CompanionMatrix 对象的元素值
def test_entry():
    # 定义符号变量和常数符号
    c0, c1, c2 = symbols('c0:3')
    x = Symbol('x')
    # 创建 CompanionMatrix 对象 A
    A = CompanionMatrix(Poly([1, c2, c1, c0], x))
    # 断言 CompanionMatrix 对象 A 的特定元素值
    assert A[0, 0] == 0
    assert A[1, 0] == 1
    assert A[1, 1] == 0
    assert A[2, 1] == 1
    assert A[0, 2] == -c0
    assert A[1, 2] == -c1
    assert A[2, 2] == -c2

# 定义测试函数，测试 CompanionMatrix 对象的显式表示
def test_as_explicit():
    # 定义符号变量和常数符号
    c0, c1, c2 = symbols('c0:3')
    x = Symbol('x')
    # 断言 CompanionMatrix 对象的 as_explicit 方法的输出
    assert CompanionMatrix(Poly([1, c0], x)).as_explicit() == \
        ImmutableDenseMatrix([-c0])
    assert CompanionMatrix(Poly([1, c1, c0], x)).as_explicit() == \
        ImmutableDenseMatrix([[0, -c0], [1, -c1]])
    assert CompanionMatrix(Poly([1, c2, c1, c0], x)).as_explicit() == \
        ImmutableDenseMatrix([[0, 0, -c0], [1, 0, -c1], [0, 1, -c2]])
```