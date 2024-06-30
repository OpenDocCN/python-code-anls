# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_subspaces.py`

```
from sympy.matrices import Matrix  # 导入 SymPy 中的矩阵模块 Matrix
from sympy.core.numbers import Rational  # 导入 SymPy 中的有理数模块 Rational
from sympy.core.symbol import symbols  # 导入 SymPy 中的符号模块 symbols
from sympy.solvers import solve  # 导入 SymPy 中的求解器模块 solve


def test_columnspace_one():
    m = Matrix([[ 1,  2,  0,  2,  5],  # 定义一个 4x5 的矩阵 m
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    basis = m.columnspace()  # 计算矩阵 m 的列空间基础
    assert basis[0] == Matrix([1, -2, 0, 3])  # 断言列空间的第一个基向量
    assert basis[1] == Matrix([2, -5, -3, 6])  # 断言列空间的第二个基向量
    assert basis[2] == Matrix([2, -1, 4, -7])  # 断言列空间的第三个基向量

    assert len(basis) == 3  # 断言列空间的基向量个数为 3
    assert Matrix.hstack(m, *basis).columnspace() == basis  # 断言由 m 和其基向量组成的新矩阵的列空间与原始列空间相同


def test_rowspace():
    m = Matrix([[ 1,  2,  0,  2,  5],  # 定义一个 4x5 的矩阵 m
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    basis = m.rowspace()  # 计算矩阵 m 的行空间基础
    assert basis[0] == Matrix([[1, 2, 0, 2, 5]])  # 断言行空间的第一个基向量
    assert basis[1] == Matrix([[0, -1, 1, 3, 2]])  # 断言行空间的第二个基向量
    assert basis[2] == Matrix([[0, 0, 0, 5, 5]])  # 断言行空间的第三个基向量

    assert len(basis) == 3  # 断言行空间的基向量个数为 3


def test_nullspace_one():
    m = Matrix([[ 1,  2,  0,  2,  5],  # 定义一个 4x5 的矩阵 m
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    basis = m.nullspace()  # 计算矩阵 m 的零空间基础
    assert basis[0] == Matrix([-2, 1, 1, 0, 0])  # 断言零空间的第一个基向量
    assert basis[1] == Matrix([-1, -1, 0, -1, 1])  # 断言零空间的第二个基向量
    # 确保零空间的第一个基向量和第二个基向量都是真正的零空间向量
    assert all(e.is_zero for e in m*basis[0])
    assert all(e.is_zero for e in m*basis[1])


def test_nullspace_second():
    # 首先测试行简化阶梯形式
    R = Rational  # 定义有理数的简写

    M = Matrix([[5, 7, 2,  1],  # 定义一个 2x4 的矩阵 M
                [1, 6, 2, -1]])
    out, tmp = M.rref()  # 计算矩阵 M 的行简化阶梯形式
    assert out == Matrix([[1, 0, -R(2)/23, R(13)/23],  # 断言行简化阶梯形式的正确性
                          [0, 1,  R(8)/23, R(-6)/23]])

    M = Matrix([[-5, -1,  4, -3, -1],  # 定义一个 5x5 的矩阵 M
                [ 1, -1, -1,  1,  0],
                [-1,  0,  0,  0,  0],
                [ 4,  1, -4,  3,  1],
                [-2,  0,  2, -2, -1]])
    assert M*M.nullspace()[0] == Matrix(5, 1, [0]*5)  # 断言 M 乘以其零空间的第一个基向量结果全为零

    M = Matrix([[ 1,  3, 0,  2,  6, 3, 1],  # 定义一个 4x7 的矩阵 M
                [-2, -6, 0, -2, -8, 3, 1],
                [ 3,  9, 0,  0,  6, 6, 2],
                [-1, -3, 0,  1,  0, 9, 3]])
    out, tmp = M.rref()  # 计算矩阵 M 的行简化阶梯形式
    assert out == Matrix([[1, 3, 0, 0, 2, 0, 0],  # 断言行简化阶梯形式的正确性
                          [0, 0, 0, 1, 2, 0, 0],
                          [0, 0, 0, 0, 0, 1, R(1)/3],
                          [0, 0, 0, 0, 0, 0, 0]])

    # 现在检查向量
    basis = M.nullspace()  # 计算矩阵 M 的零空间基础
    assert basis[0] == Matrix([-3, 1, 0, 0, 0, 0, 0])  # 断言零空间的第一个基向量
    assert basis[1] == Matrix([0, 0, 1, 0, 0, 0, 0])  # 断言零空间的第二个基向量
    assert basis[2] == Matrix([-2, 0, 0, -2, 1, 0, 0])  # 断言零空间的第三个基向量
    assert basis[3] == Matrix([0, 0, 0, 0, 0, R(-1)/3, 1])  # 断言零空间的第四个基向量

    # 问题 4797; 只需检查当行数大于列数时是否可以执行
    M = Matrix([[1, 2], [2, 4], [3, 6]])  # 定义一个 3x2 的矩阵 M
    assert M.nullspace()


def test_columnspace_second():
    M = Matrix([[ 1,  2,  0,  2,  5],  # 定义一个 4x5 的矩阵 M
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    # 现在检查向量
    basis = M.columnspace()  # 计算矩阵 M 的列空间基础
    # 断言：验证基向量的值是否符合预期
    assert basis[0] == Matrix([1, -2, 0, 3])
    assert basis[1] == Matrix([2, -5, -3, 6])
    assert basis[2] == Matrix([2, -1, 4, -7])

    # 检查列空间的定义是否成立
    a, b, c, d, e = symbols('a b c d e')
    X = Matrix([a, b, c, d, e])
    for i in range(len(basis)):
        # 构造线性方程组 M*X = basis[i]，并断言其有解
        eq = M*X - basis[i]
        assert len(solve(eq, X)) != 0

    # 检查秩-零空间定理是否成立
    assert M.rank() == len(basis)
    assert len(M.nullspace()) + len(M.columnspace()) == M.cols
```