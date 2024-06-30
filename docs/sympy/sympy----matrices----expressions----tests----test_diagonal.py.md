# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_diagonal.py`

```
from sympy.matrices.expressions import MatrixSymbol  # 导入 MatrixSymbol 类
from sympy.matrices.expressions.diagonal import DiagonalMatrix, DiagonalOf, DiagMatrix, diagonalize_vector  # 导入对角矩阵相关的类和函数
from sympy.assumptions.ask import (Q, ask)  # 导入符号逻辑相关的类和函数
from sympy.core.symbol import Symbol  # 导入符号变量类 Symbol
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入 KroneckerDelta 函数
from sympy.matrices.dense import Matrix  # 导入稠密矩阵类
from sympy.matrices.expressions.matmul import MatMul  # 导入矩阵乘法类
from sympy.matrices.expressions.special import Identity  # 导入单位矩阵类
from sympy.testing.pytest import raises  # 导入测试工具 raises

# 定义符号变量 n 和 m
n = Symbol('n')
m = Symbol('m')

# 测试 DiagonalMatrix 类的功能
def test_DiagonalMatrix():
    # 创建一个 n 行 m 列的 MatrixSymbol 对象 x
    x = MatrixSymbol('x', n, m)
    # 创建 DiagonalMatrix 对象 D，其对角线为 x
    D = DiagonalMatrix(x)
    assert D.diagonal_length is None  # 断言 D 的对角线长度为 None
    assert D.shape == (n, m)  # 断言 D 的形状为 (n, m)

    # 创建一个 n 阶方阵的 MatrixSymbol 对象 x
    x = MatrixSymbol('x', n, n)
    # 创建 DiagonalMatrix 对象 D，其对角线为 x
    D = DiagonalMatrix(x)
    assert D.diagonal_length == n  # 断言 D 的对角线长度为 n
    assert D.shape == (n, n)  # 断言 D 的形状为 (n, n)
    assert D[1, 2] == 0  # 断言 D 的非对角线元素 D[1, 2] 为 0
    assert D[1, 1] == x[1, 1]  # 断言 D 的对角线元素 D[1, 1] 等于 x[1, 1]

    # 创建符号变量 i 和 j
    i = Symbol('i')
    j = Symbol('j')
    # 创建一个 3x3 的 MatrixSymbol 对象 x
    x = MatrixSymbol('x', 3, 3)
    # 获取 DiagonalMatrix 对象 D 的 ij 元素
    ij = DiagonalMatrix(x)[i, j]
    assert ij != 0  # 断言 ij 元素不为 0
    assert ij.subs({i:0, j:0}) == x[0, 0]  # 断言 ij 在 i=0, j=0 时的取值为 x[0, 0]
    assert ij.subs({i:0, j:1}) == 0  # 断言 ij 在 i=0, j=1 时的取值为 0
    assert ij.subs({i:1, j:1}) == x[1, 1]  # 断言 ij 在 i=1, j=1 时的取值为 x[1, 1]
    assert ask(Q.diagonal(D))  # 断言 D 是对角矩阵

    # 创建一个 n 行 3 列的 MatrixSymbol 对象 x
    x = MatrixSymbol('x', n, 3)
    # 创建 DiagonalMatrix 对象 D，其对角线为 x
    D = DiagonalMatrix(x)
    assert D.diagonal_length == 3  # 断言 D 的对角线长度为 3
    assert D.shape == (n, 3)  # 断言 D 的形状为 (n, 3)
    assert D[2, m] == KroneckerDelta(2, m)*x[2, m]  # 断言 D 的元素 D[2, m] 等于 KroneckerDelta(2, m)*x[2, m]
    assert D[3, m] == 0  # 断言 D 的元素 D[3, m] 等于 0
    raises(IndexError, lambda: D[m, 3])  # 断言访问 D[m, 3] 会抛出 IndexError 异常

    # 创建一个 3 行 n 列的 MatrixSymbol 对象 x
    x = MatrixSymbol('x', 3, n)
    # 创建 DiagonalMatrix 对象 D，其对角线为 x
    D = DiagonalMatrix(x)
    assert D.diagonal_length == 3  # 断言 D 的对角线长度为 3
    assert D.shape == (3, n)  # 断言 D 的形状为 (3, n)
    assert D[m, 2] == KroneckerDelta(m, 2)*x[m, 2]  # 断言 D 的元素 D[m, 2] 等于 KroneckerDelta(m, 2)*x[m, 2]
    assert D[m, 3] == 0  # 断言 D 的元素 D[m, 3] 等于 0
    raises(IndexError, lambda: D[3, m])  # 断言访问 D[3, m] 会抛出 IndexError 异常

    # 创建一个 n 行 m 列的 MatrixSymbol 对象 x
    x = MatrixSymbol('x', n, m)
    # 创建 DiagonalMatrix 对象 D，其对角线为 x
    D = DiagonalMatrix(x)
    assert D.diagonal_length is None  # 断言 D 的对角线长度为 None
    assert D.shape == (n, m)  # 断言 D 的形状为 (n, m)
    assert D[m, 4] != 0  # 断言 D 的元素 D[m, 4] 不等于 0

    # 创建一个 3 行 4 列的 MatrixSymbol 对象 x
    x = MatrixSymbol('x', 3, 4)
    assert [DiagonalMatrix(x)[i] for i in range(12)] == [
        x[0, 0], 0, 0, 0, 0, x[1, 1], 0, 0, 0, 0, x[2, 2], 0]  # 断言 DiagonalMatrix(x) 的前 12 个元素的列表

    # 断言两个对角矩阵相乘的形状结果
    assert (
        DiagonalMatrix(MatrixSymbol('x', 3, 4))*
        DiagonalMatrix(MatrixSymbol('x', 4, 2))).shape == (3, 2)


# 测试 DiagonalOf 类的功能
def test_DiagonalOf():
    # 创建一个 n 阶方阵的 MatrixSymbol 对象 x
    x = MatrixSymbol('x', n, n)
    # 创建 DiagonalOf 对象 d，其对角线为 x
    d = DiagonalOf(x)
    assert d.shape == (n, 1)  # 断言 d 的形状为 (n, 1)
    assert d.diagonal_length == n  # 断言 d 的对角线长度为 n
    assert d[2, 0] == d[2] == x[2, 2]  # 断言 d 的元素 d[2, 0] 等于 x[2, 2]

    # 创建一个 n 行 m 列的 MatrixSymbol 对象 x
    x = MatrixSymbol('x', n, m)
    # 创建 DiagonalOf 对象 d，其对角线为 x
    d = DiagonalOf(x)
    assert d.shape == (None, 1)  # 断言 d 的形状为 (None, 1)
    assert d.diagonal_length is None  # 断言 d 的对角线长度为 None
    assert d[2, 0] == d[2] == x[2, 2]  # 断言 d 的元素 d[2, 0] 等于 x[2, 2]

    # 创建一个 4 行 3 列的 MatrixSymbol 对象 x，并创建 DiagonalOf 对象 d
    d = DiagonalOf(MatrixSymbol('x', 4, 3))
    assert d.shape == (3, 1)  # 断言 d 的形状为 (3, 1)

    # 创建一个 n 行 3 列的 MatrixSymbol 对象 x，并创建 DiagonalOf 对象 d
    d = DiagonalOf(MatrixSymbol('x', n, 3
    # 对向量进行对角化，返回一个 MatrixSymbol 对象
    d = diagonalize_vector(a)
    # 断言 d 是 MatrixSymbol 类的实例
    assert isinstance(d, MatrixSymbol)
    # 断言对角化后的向量 d 和原始向量 a 相等
    assert a == d
    # 断言对单位矩阵进行对角化的结果等于单位矩阵本身
    assert diagonalize_vector(Identity(3)) == Identity(3)
    # 断言对单位矩阵进行 DiagMatrix 处理后执行 doit() 结果等于单位矩阵本身
    assert DiagMatrix(Identity(3)).doit() == Identity(3)
    # 断言 DiagMatrix(Identity(3)) 是 DiagMatrix 类的实例
    assert isinstance(DiagMatrix(Identity(3)), DiagMatrix)

    # 对角矩阵与其转置相等的断言
    assert DiagMatrix(x).T == DiagMatrix(x)
    # 断言对向量 x 进行对角化的结果与 DiagMatrix(x) 相等
    assert diagonalize_vector(x.T) == DiagMatrix(x)

    # 创建 DiagMatrix 对象 dx，并进行以下断言
    dx = DiagMatrix(x)
    assert dx[0, 0] == x[0, 0]
    assert dx[1, 1] == x[1, 0]
    assert dx[0, 1] == 0
    # 使用 KroneckerDelta 函数断言 dx[0, m] 的值
    assert dx[0, m] == x[0, 0]*KroneckerDelta(0, m)

    # 创建 MatrixSymbol 对象 z，并创建对应的 DiagMatrix 对象 dz
    z = MatrixSymbol('z', 1, n)
    dz = DiagMatrix(z)
    # 断言 dz[0, 0] 等于 z[0, 0]
    assert dz[0, 0] == z[0, 0]
    # 断言 dz[1, 1] 等于 z[0, 1]
    assert dz[1, 1] == z[0, 1]
    # 断言 dz[0, 1] 等于 0
    assert dz[0, 1] == 0
    # 使用 KroneckerDelta 函数断言 dz[0, m] 的值
    assert dz[0, m] == z[0, m]*KroneckerDelta(0, m)

    # 创建 MatrixSymbol 对象 v，并创建对应的 DiagMatrix 对象 dv
    v = MatrixSymbol('v', 3, 1)
    dv = DiagMatrix(v)
    # 断言 dv 的 as_explicit() 方法返回的矩阵等于指定的 Matrix 对象
    assert dv.as_explicit() == Matrix([
        [v[0, 0], 0, 0],
        [0, v[1, 0], 0],
        [0, 0, v[2, 0]],
    ])

    # 创建 MatrixSymbol 对象 v，并创建对应的 DiagMatrix 对象 dv
    v = MatrixSymbol('v', 1, 3)
    dv = DiagMatrix(v)
    # 断言 dv 的 as_explicit() 方法返回的矩阵等于指定的 Matrix 对象
    assert dv.as_explicit() == Matrix([
        [v[0, 0], 0, 0],
        [0, v[0, 1], 0],
        [0, 0, v[0, 2]],
    ])

    # 创建 DiagMatrix 对象 dv，并进行以下断言
    dv = DiagMatrix(3*v)
    # 断言 dv 的参数是 (3*v,)
    assert dv.args == (3*v,)
    # 断言 dv.doit() 的结果等于 3*DiagMatrix(v)
    assert dv.doit() == 3*DiagMatrix(v)
    # 断言 dv.doit() 是 MatMul 类的实例
    assert isinstance(dv.doit(), MatMul)

    # 创建 MatrixSymbol 对象 a，并转换为其具体矩阵表示
    a = MatrixSymbol("a", 3, 1).as_explicit()
    # 创建 DiagMatrix 对象 expr，并进行以下断言
    expr = DiagMatrix(a)
    # 创建期望的结果矩阵
    result = Matrix([
        [a[0, 0], 0, 0],
        [0, a[1, 0], 0],
        [0, 0, a[2, 0]],
    ])
    # 断言 expr.doit() 的结果等于 result
    assert expr.doit() == result
    # 创建 DiagMatrix 对象 expr，并进行以下断言
    expr = DiagMatrix(a.T)
    # 断言 expr.doit() 的结果等于 result
    assert expr.doit() == result
```