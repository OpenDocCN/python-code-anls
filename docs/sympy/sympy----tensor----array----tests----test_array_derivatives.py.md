# `D:\src\scipysrc\sympy\sympy\tensor\array\tests\test_array_derivatives.py`

```
# 从 sympy 库中导入符号、矩阵和张量相关的类和函数
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.array.ndim_array import NDimArray
from sympy.matrices.matrixbase import MatrixBase
from sympy.tensor.array.array_derivatives import ArrayDerivative

# 创建符号变量 x, y, z, t
x, y, z, t = symbols("x y z t")

# 创建一个二阶矩阵 m，其中元素为符号变量 x, y, z, t
m = Matrix([[x, y], [z, t]])

# 创建两个矩阵符号 M 和 N，分别表示 3x2 和 4x3 的矩阵

M = MatrixSymbol("M", 3, 2)
N = MatrixSymbol("N", 4, 3)


def test_array_derivative_construction():
    # 创建一个数组导数对象 d，作用在 m 上，但不进行求值
    d = ArrayDerivative(x, m, evaluate=False)
    # 断言导数对象的形状为 (2, 2)
    assert d.shape == (2, 2)
    # 对导数对象进行求值，返回一个矩阵基类的实例
    expr = d.doit()
    # 断言求值后的结果是 MatrixBase 类的实例，并且形状为 (2, 2)
    assert isinstance(expr, MatrixBase)
    assert expr.shape == (2, 2)

    # 创建一个数组导数对象 d，作用在 m 上，但不进行求值
    d = ArrayDerivative(m, m, evaluate=False)
    # 断言导数对象的形状为 (2, 2, 2, 2)
    assert d.shape == (2, 2, 2, 2)
    # 对导数对象进行求值，返回一个 NDimArray 类的实例
    expr = d.doit()
    # 断言求值后的结果是 NDimArray 类的实例，并且形状为 (2, 2, 2, 2)
    assert isinstance(expr, NDimArray)
    assert expr.shape == (2, 2, 2, 2)

    # 创建一个数组导数对象 d，作用在 m 上，但不进行求值
    d = ArrayDerivative(m, x, evaluate=False)
    # 断言导数对象的形状为 (2, 2)
    assert d.shape == (2, 2)
    # 对导数对象进行求值，返回一个矩阵基类的实例
    expr = d.doit()
    # 断言求值后的结果是 MatrixBase 类的实例，并且形状为 (2, 2)
    assert isinstance(expr, MatrixBase)
    assert expr.shape == (2, 2)

    # 创建一个数组导数对象 d，作用在 M 和 N 上，但不进行求值
    d = ArrayDerivative(M, N, evaluate=False)
    # 断言导数对象的形状为 (4, 3, 3, 2)
    assert d.shape == (4, 3, 3, 2)
    # 对导数对象进行求值，返回一个数组导数的实例
    expr = d.doit()
    # 断言求值后的结果是 ArrayDerivative 类的实例，并且形状为 (4, 3, 3, 2)
    assert isinstance(expr, ArrayDerivative)
    assert expr.shape == (4, 3, 3, 2)

    # 创建一个数组导数对象 d，作用在 M 和 (N, 2) 上，但不进行求值
    d = ArrayDerivative(M, (N, 2), evaluate=False)
    # 断言导数对象的形状为 (4, 3, 4, 3, 3, 2)
    assert d.shape == (4, 3, 4, 3, 3, 2)
    # 对导数对象进行求值，返回一个数组导数的实例
    expr = d.doit()
    # 断言求值后的结果是 ArrayDerivative 类的实例，并且形状为 (4, 3, 4, 3, 3, 2)
    assert isinstance(expr, ArrayDerivative)
    assert expr.shape == (4, 3, 4, 3, 3, 2)

    # 创建一个数组导数对象 d，作用在 M.as_explicit() 和 (N.as_explicit(), 2) 上，但不进行求值
    d = ArrayDerivative(M.as_explicit(), (N.as_explicit(), 2), evaluate=False)
    # 对求值后的导数对象进行断言，形状应为 (4, 3, 4, 3, 3, 2)
    assert d.doit().shape == (4, 3, 4, 3, 3, 2)
    expr = d.doit()
    # 断言求值后的结果是 NDimArray 类的实例，并且形状为 (4, 3, 4, 3, 3, 2)
    assert isinstance(expr, NDimArray)
    assert expr.shape == (4, 3, 4, 3, 3, 2)
```