# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_eigen.py`

```
# 导入必要的 Sympy 模块和函数，以下为具体导入的模块和函数
from sympy.core.evalf import N
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices import eye, Matrix
from sympy.core.singleton import S
from sympy.testing.pytest import raises, XFAIL
from sympy.matrices.exceptions import NonSquareMatrixError, MatrixError
from sympy.matrices.expressions.fourier import DFT
from sympy.simplify.simplify import simplify
from sympy.matrices.immutable import ImmutableMatrix
from sympy.testing.pytest import slow
from sympy.testing.matrices import allclose


# 定义一个测试函数，用于测试矩阵的特征值和特征向量计算
def test_eigen():
    # 定义有理数 R
    R = Rational
    # 创建一个 3x3 的单位矩阵 M
    M = Matrix.eye(3)
    # 断言 M 的非重复特征值为 {1: 3}
    assert M.eigenvals(multiple=False) == {S.One: 3}
    # 断言 M 的所有特征值（可能重复）为 [1, 1, 1]
    assert M.eigenvals(multiple=True) == [1, 1, 1]

    # 断言 M 的特征值及其代数重数和特征向量
    assert M.eigenvects() == (
        [(1, 3, [Matrix([1, 0, 0]),
                 Matrix([0, 1, 0]),
                 Matrix([0, 0, 1])])])

    # 断言 M 的左特征向量及其特征值
    assert M.left_eigenvects() == (
        [(1, 3, [Matrix([[1, 0, 0]]),
                 Matrix([[0, 1, 0]]),
                 Matrix([[0, 0, 1]])])])

    # 创建一个自定义的 3x3 矩阵 M
    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])

    # 断言 M 的特征值
    assert M.eigenvals() == {2*S.One: 1, -S.One: 1, S.Zero: 1}

    # 断言 M 的特征值及其代数重数和特征向量
    assert M.eigenvects() == (
        [
            (-1, 1, [Matrix([-1, 1, 0])]),
            ( 0, 1, [Matrix([0, -1, 1])]),
            ( 2, 1, [Matrix([R(2, 3), R(1, 3), 1])])
        ])

    # 断言 M 的左特征向量及其特征值
    assert M.left_eigenvects() == (
        [
            (-1, 1, [Matrix([[-2, 1, 1]])]),
            (0, 1, [Matrix([[-1, -1, 1]])]),
            (2, 1, [Matrix([[1, 1, 1]])])
        ])

    # 创建一个包含符号变量的 2x2 矩阵 M
    a = Symbol('a')
    M = Matrix([[a, 0],
                [0, 1]])

    # 断言 M 的特征值
    assert M.eigenvals() == {a: 1, S.One: 1}

    # 创建一个 2x2 矩阵 M
    M = Matrix([[1, -1],
                [1,  3]])

    # 断言 M 的特征向量
    assert M.eigenvects() == ([(2, 2, [Matrix(2, 1, [-1, 1])])])
    # 断言 M 的左特征向量
    assert M.left_eigenvects() == ([(2, 2, [Matrix([[1, 1]])])])

    # 创建一个 3x3 矩阵 M
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 定义几个有理数
    a = R(15, 2)
    b = 3*33**R(1, 2)
    c = R(13, 2)
    d = (R(33, 8) + 3*b/8)
    e = (R(33, 8) - 3*b/8)

    # 定义一个数值转换函数 NS
    def NS(e, n):
        return str(N(e, n))

    # 定义预期的特征值和特征向量结果列表 r
    r = [
        (a - b/2, 1, [Matrix([(12 + 24/(c - b/2))/((c - b/2)*e) + 3/(c - b/2),
                              (6 + 12/(c - b/2))/e, 1])]),
        (      0, 1, [Matrix([1, -2, 1])]),
        (a + b/2, 1, [Matrix([(12 + 24/(c + b/2))/((c + b/2)*d) + 3/(c + b/2),
                              (6 + 12/(c + b/2))/d, 1])]),
    ]
    # 转换 r 中的数值为字符串，保留两位小数，形成 r1
    r1 = [(NS(r[i][0], 2), NS(r[i][1], 2),
           [NS(j, 2) for j in r[i][2][0]]) for i in range(len(r))]
    # 计算 M 的特征值和特征向量 r，并转换为字符串，保留两位小数，形成 r2
    r = M.eigenvects()
    r2 = [(NS(r[i][0], 2), NS(r[i][1], 2),
           [NS(j, 2) for j in r[i][2][0]]) for i in range(len(r))]
    # 断言 r1 和 r2 排序后相等
    assert sorted(r1) == sorted(r2)

    # 定义一个实数符号 eps
    eps = Symbol('eps', real=True)

    # 创建一个 2x2 矩阵 M，其元素包含复数和绝对值函数
    M = Matrix([[abs(eps), I*eps    ],
                [-I*eps,   abs(eps) ]])
    # 对称矩阵 M 的特征向量验证
    assert M.eigenvects() == (
        [
            # 特征值 0 的特征向量和代数重数 1，向量为 Matrix([[-I*eps/abs(eps)], [1]])
            ( 0, 1, [Matrix([[-I*eps/abs(eps)], [1]])]),
            # 特征值 2*abs(eps) 的特征向量和代数重数 1，向量为 Matrix([[I*eps/abs(eps)], [1]])
            ( 2*abs(eps), 1, [ Matrix([[I*eps/abs(eps)], [1]]) ] ),
        ])

    # 对称矩阵 M 的左特征向量验证
    assert M.left_eigenvects() == (
        [
            # 特征值 0 的左特征向量和代数重数 1，向量为 Matrix([[I*eps/Abs(eps), 1]])
            (0, 1, [Matrix([[I*eps/Abs(eps), 1]])]),
            # 特征值 2*Abs(eps) 的左特征向量和代数重数 1，向量为 Matrix([[-I*eps/Abs(eps), 1]])
            (2*Abs(eps), 1, [Matrix([[-I*eps/Abs(eps), 1]])])
        ])

    # 创建一个新的 3x3 矩阵 M
    M = Matrix(3, 3, [1, 2, 0, 0, 3, 0, 2, -4, 2])

    # 使用 simplify=False 计算 M 的特征向量
    M._eigenvects = M.eigenvects(simplify=False)
    # 确保 M 的第一个特征向量中的所有元素的虚部大于 1
    assert max(i.q for i in M._eigenvects[0][2][0]) > 1

    # 使用 simplify=True 计算 M 的特征向量
    M._eigenvects = M.eigenvects(simplify=True)
    # 确保 M 的第一个特征向量中的所有元素的虚部等于 1
    assert max(i.q for i in M._eigenvects[0][2][0]) == 1

    # 创建一个新的 2x2 矩阵 M
    M = Matrix([[Rational(1, 4), 1], [1, 1]])

    # 验证 M 的特征向量计算
    assert M.eigenvects() == [
        # 特征值为 Rational(5, 8) - sqrt(73)/8 时，特征向量为 Matrix([[-sqrt(73)/8 - Rational(3, 8)], [1]])
        (Rational(5, 8) - sqrt(73)/8, 1, [Matrix([[-sqrt(73)/8 - Rational(3, 8)], [1]])]),
        # 特征值为 Rational(5, 8) + sqrt(73)/8 时，特征向量为 Matrix([[Rational(-3, 8) + sqrt(73)/8], [1]])
        (Rational(5, 8) + sqrt(73)/8, 1, [Matrix([[Rational(-3, 8) + sqrt(73)/8], [1]])])
    ]

    # 针对 issue 10719 的测试
    assert Matrix([]).eigenvals() == {}  # 空矩阵的特征值为空字典
    assert Matrix([]).eigenvals(multiple=True) == []  # 空矩阵的特征值（多个）为空列表
    assert Matrix([]).eigenvects() == []  # 空矩阵的特征向量为空列表

    # 针对 issue 15119 的测试：测试非方阵的特征值计算是否引发 NonSquareMatrixError 异常
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 2], [0, 4], [0, 0]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 0], [3, 4], [5, 6]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 2, 3], [0, 5, 6]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 0, 0], [4, 5, 0]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 2, 3], [0, 5, 6]]).eigenvals(
               error_when_incomplete = False))
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 0, 0], [4, 5, 0]]).eigenvals(
               error_when_incomplete = False))

    # 创建一个新的 2x2 矩阵 m
    m = Matrix([[1, 2], [3, 4]])

    # 验证简化和非简化方式计算特征值的返回类型
    assert isinstance(m.eigenvals(simplify=True, multiple=False), dict)
    assert isinstance(m.eigenvals(simplify=True, multiple=True), list)
    assert isinstance(m.eigenvals(simplify=lambda x: x, multiple=False), dict)
    assert isinstance(m.eigenvals(simplify=lambda x: x, multiple=True), list)
def test_float_eigenvals():
    # 创建一个 3x3 的矩阵对象，其中包含浮点数
    m = Matrix([[1, .6, .6], [.6, .9, .9], [.9, .6, .6]])
    # 定义一个精确的有理数列表
    evals = [
        Rational(5, 4) - sqrt(385)/20,
        sqrt(385)/20 + Rational(5, 4),
        S.Zero]

    # 计算矩阵的有理数特征值，允许多个特征值返回
    n_evals = m.eigenvals(rational=True, multiple=True)
    # 对特征值列表进行排序
    n_evals = sorted(n_evals)
    # 对精确特征值列表中每个特征值进行数值求解
    s_evals = [x.evalf() for x in evals]
    # 对数值特征值列表进行排序
    s_evals = sorted(s_evals)

    # 断言计算特征值与预期特征值之间的误差小于 10^-9
    for x, y in zip(n_evals, s_evals):
        assert abs(x-y) < 10**-9


@XFAIL
def test_eigen_vects():
    # 创建一个 2x2 的矩阵对象，包含复数单位
    m = Matrix(2, 2, [1, 0, 0, I])
    # 使用 lambda 函数断言 m 不可对角化
    raises(NotImplementedError, lambda: m.is_diagonalizable(True))
    # 断言 m 不可对角化
    assert not m.is_diagonalizable(True)
    # 使用 lambda 函数断言 m 报错并抛出 MatrixError
    raises(MatrixError, lambda: m.diagonalize(True))
    # 尝试对 m 进行对角化操作，获取对角化矩阵 P 和对角线矩阵 D
    (P, D) = m.diagonalize(True)


def test_issue_8240():
    # 创建符号变量 x 和 y
    x, y = symbols('x y')
    # 定义一个较大的方阵阶数
    n = 200

    # 创建对角线上有符号变量的方阵 M
    diagonal_variables = [Symbol('x%s' % i) for i in range(n)]
    M = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        M[i][i] = diagonal_variables[i]
    M = Matrix(M)

    # 计算 M 的特征值，并断言特征值数量等于方阵的阶数
    eigenvals = M.eigenvals()
    assert len(eigenvals) == n
    # 对每个对角线元素，断言其特征值为 1
    for i in range(n):
        assert eigenvals[diagonal_variables[i]] == 1

    # 计算 M 的多重特征值，并断言特征值集合与对角线变量集合相同
    eigenvals = M.eigenvals(multiple=True)
    assert set(eigenvals) == set(diagonal_variables)

    # 对具有多重性质的特征值进行测试
    M = Matrix([[x, 0, 0], [1, y, 0], [2, 3, x]])
    eigenvals = M.eigenvals()
    assert eigenvals == {x: 2, y: 1}

    # 计算 M 的多重特征值，并断言特征值数量为 3，并检查每个特征值的个数
    eigenvals = M.eigenvals(multiple=True)
    assert len(eigenvals) == 3
    assert eigenvals.count(x) == 2
    assert eigenvals.count(y) == 1


def test_eigenvals():
    # 创建一个 3x3 的矩阵对象
    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])
    # 断言计算 M 的特征值
    assert M.eigenvals() == {2*S.One: 1, -S.One: 1, S.Zero: 1}

    m = Matrix([
        [3,  0,  0, 0, -3],
        [0, -3, -3, 0,  3],
        [0,  3,  0, 3,  0],
        [0,  0,  3, 0,  3],
        [3,  0,  0, 3,  0]])

    # 使用 dry-run 测试，因为在 CRootOf 中出现的任意符号可能不唯一
    assert m.eigenvals()


def test_eigenvects():
    # 创建一个 3x3 的矩阵对象
    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])
    # 计算 M 的特征向量
    vecs = M.eigenvects()
    # 对每个特征值、特征向量列表进行断言
    for val, mult, vec_list in vecs:
        assert len(vec_list) == 1
        assert M*vec_list[0] == val*vec_list[0]


def test_left_eigenvects():
    # 创建一个 3x3 的矩阵对象
    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])
    # 计算 M 的左特征向量
    vecs = M.left_eigenvects()
    # 对每个特征值、特征向量列表进行断言
    for val, mult, vec_list in vecs:
        assert len(vec_list) == 1
        assert vec_list[0]*M == val*vec_list[0]


@slow
def test_bidiagonalize():
    # 创建一个 3x3 的单位矩阵对象 M
    M = Matrix([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    # 断言对 M 的双对角化操作返回原始矩阵 M
    assert M.bidiagonalize() == M
    # 断言对 M 的非上三角双对角化操作返回原始矩阵 M
    assert M.bidiagonalize(upper=False) == M
    # 重复对 M 的默认双对角化操作，断言返回原始矩阵 M
    assert M.bidiagonalize() == M
    # 断言对 M 的双对角分解操作返回三个相同的矩阵 M
    assert M.bidiagonal_decomposition() == (M, M, M)
    # 断言对 M 的非上三角双对角分解操作返回三个相同的矩阵 M
    assert M.bidiagonal_decomposition(upper=False) == (M, M, M)
    # 再次断言对 M 的默认双对角化操作，返回原始矩阵 M
    assert M.bidiagonalize() == M

    import random
    # 真实测试用例
    # 对于两次循环进行测试
    for real_test in range(2):
        # 初始化测试值列表为空
        test_values = []
        # 定义行数和列数为2
        row = 2
        col = 2
        # 循环生成行数乘以列数个随机整数，范围在-1000000000到1000000000之间
        for _ in range(row * col):
            value = random.randint(-1000000000, 1000000000)
            # 将生成的随机整数添加到测试值列表中
            test_values = test_values + [value]
        # 创建可变矩阵 M，使用给定的行数、列数和测试值列表
        M = Matrix(row, col, test_values)
        # 创建不可变矩阵 N，从可变矩阵 M 中构造
        N = ImmutableMatrix(M)

        # 对不可变矩阵 N 进行双对角分解，得到三个分解矩阵 N1, N2, N3
        N1, N2, N3 = N.bidiagonal_decomposition()
        # 对可变矩阵 M 进行双对角分解，得到三个分解矩阵 M1, M2, M3
        M1, M2, M3 = M.bidiagonal_decomposition()
        # 将可变矩阵 M 转换为其双对角化形式 M0
        M0 = M.bidiagonalize()
        # 将不可变矩阵 N 转换为其双对角化形式 N0
        N0 = N.bidiagonalize()

        # 计算不可变矩阵 N 的三个分解矩阵的乘积 N1 * N2 * N3 得到 N4
        N4 = N1 * N2 * N3
        # 计算可变矩阵 M 的三个分解矩阵的乘积 M1 * M2 * M3 得到 M4
        M4 = M1 * M2 * M3

        # 简化不可变矩阵 N 的第二个分解矩阵 N2
        N2.simplify()
        # 简化不可变矩阵 N 的乘积矩阵 N4
        N4.simplify()
        # 简化不可变矩阵 N 的双对角化形式 N0
        N0.simplify()

        # 简化可变矩阵 M 的双对角化形式 M0
        M0.simplify()
        # 简化可变矩阵 M 的第二个分解矩阵 M2
        M2.simplify()
        # 简化可变矩阵 M 的乘积矩阵 M4
        M4.simplify()

        # 在不使用上三角化的情况下重新计算可变矩阵 M 的双对角化形式 LM0
        LM0 = M.bidiagonalize(upper=False)
        # 在不使用上三角化的情况下重新计算可变矩阵 M 的三个分解矩阵 LM1, LM2, LM3
        LM1, LM2, LM3 = M.bidiagonal_decomposition(upper=False)
        # 在不使用上三角化的情况下重新计算不可变矩阵 N 的双对角化形式 LN0
        LN0 = N.bidiagonalize(upper=False)
        # 在不使用上三角化的情况下重新计算不可变矩阵 N 的三个分解矩阵 LN1, LN2, LN3
        LN1, LN2, LN3 = N.bidiagonal_decomposition(upper=False)

        # 计算不可变矩阵 N 的乘积矩阵 LN1 * LN2 * LN3 得到 LN4
        LN4 = LN1 * LN2 * LN3
        # 计算可变矩阵 M 的乘积矩阵 LM1 * LM2 * LM3 得到 LM4
        LM4 = LM1 * LM2 * LM3

        # 简化不可变矩阵 N 的第二个分解矩阵 LN2
        LN2.simplify()
        # 简化不可变矩阵 N 的乘积矩阵 LN4
        LN4.simplify()
        # 简化不可变矩阵 N 的双对角化形式 LN0
        LN0.simplify()

        # 简化可变矩阵 M 的双对角化形式 LM0
        LM0.simplify()
        # 简化可变矩阵 M 的第二个分解矩阵 LM2
        LM2.simplify()
        # 简化可变矩阵 M 的乘积矩阵 LM4
        LM4.simplify()

        # 断言：可变矩阵 M 应当等于其乘积矩阵 M4
        assert M == M4
        # 断言：可变矩阵 M 的第二个分解矩阵应当等于其双对角化形式 M0
        assert M2 == M0
        # 断言：不可变矩阵 N 应当等于其乘积矩阵 N4
        assert N == N4
        # 断言：不可变矩阵 N 的第二个分解矩阵应当等于其双对角化形式 N0
        assert N2 == N0
        # 断言：可变矩阵 M 应当等于其在不使用上三角化的情况下重新计算的乘积矩阵 LM4
        assert M == LM4
        # 断言：可变矩阵 M 的第二个分解矩阵应当等于其在不使用上三角化的情况下重新计算的双对角化形式 LM0
        assert LM2 == LM0
        # 断言：不可变矩阵 N 应当等于其在不使用上三角化的情况下重新计算的乘积矩阵 LN4
        assert N == LN4
        # 断言：不可变矩阵 N 的第二个分解矩阵应当等于其在不使用上三角化的情况下重新计算的双对角化形式 LN0

    # 复杂测试
    # 进行两次复杂测试
    for complex_test in range(2):
        # 初始化测试值列表
        test_values = []
        size = 2
        # 根据指定的大小生成随机复数
        for _ in range(size * size):
            real = random.randint(-1000000000, 1000000000)
            comp = random.randint(-1000000000, 1000000000)
            value = real + comp * I
            # 将生成的复数添加到测试值列表中
            test_values = test_values + [value]
        # 创建一个 size x size 的可变矩阵 M，并使用测试值初始化
        M = Matrix(size, size, test_values)
        # 使用 M 创建一个不可变矩阵 N
        N = ImmutableMatrix(M)

        # 下面是不同矩阵操作的注释

        # 对不可变矩阵 N 进行双对角分解，得到 N 的分解结果
        # N1, N2, N3 分别是双对角分解的三个矩阵
        N1, N2, N3 = N.bidiagonal_decomposition()
        # 对可变矩阵 M 进行双对角分解，得到 M 的分解结果
        M1, M2, M3 = M.bidiagonal_decomposition()
        # 对可变矩阵 M 进行双对角化，得到 M 的双对角化形式
        M0 = M.bidiagonalize()
        # 对不可变矩阵 N 进行双对角化，得到 N 的双对角化形式
        N0 = N.bidiagonalize()

        # 计算不可变矩阵 N 的分解结果的乘积 N1 * N2 * N3
        N4 = N1 * N2 * N3
        # 计算可变矩阵 M 的分解结果的乘积 M1 * M2 * M3
        M4 = M1 * M2 * M3

        # 对不可变矩阵 N 的第二个分解结果 N2 进行简化操作
        N2.simplify()
        # 对不可变矩阵 N 的乘积 N4 进行简化操作
        N4.simplify()
        # 对不可变矩阵 N 的双对角化形式 N0 进行简化操作
        N0.simplify()

        # 对可变矩阵 M 的双对角化形式 M0 进行简化操作
        M0.simplify()
        # 对可变矩阵 M 的第二个分解结果 M2 进行简化操作
        M2.simplify()
        # 对可变矩阵 M 的乘积 M4 进行简化操作
        M4.simplify()

        # 对可变矩阵 M 的下三角双对角化形式进行简化操作
        LM0 = M.bidiagonalize(upper=False)
        # 对可变矩阵 M 的下三角双对角分解结果进行简化操作
        LM1, LM2, LM3 = M.bidiagonal_decomposition(upper=False)
        # 对不可变矩阵 N 的下三角双对角化形式进行简化操作
        LN0 = N.bidiagonalize(upper=False)
        # 对不可变矩阵 N 的下三角双对角分解结果进行简化操作
        LN1, LN2, LN3 = N.bidiagonal_decomposition(upper=False)

        # 计算不可变矩阵 N 下三角双对角分解结果的乘积 LN1 * LN2 * LN3
        LN4 = LN1 * LN2 * LN3
        # 计算可变矩阵 M 下三角双对角分解结果的乘积 LM1 * LM2 * LM3
        LM4 = LM1 * LM2 * LM3

        # 对不可变矩阵 N 的下三角双对角分解结果 LN2 进行简化操作
        LN2.simplify()
        # 对不可变矩阵 N 的下三角双对角化形式 LN4 进行简化操作
        LN4.simplify()
        # 对不可变矩阵 N 的下三角双对角化形式 LN0 进行简化操作
        LN0.simplify()

        # 对可变矩阵 M 的下三角双对角化形式 LM0 进行简化操作
        LM0.simplify()
        # 对可变矩阵 M 的下三角双对角分解结果 LM2 进行简化操作
        LM2.simplify()
        # 对可变矩阵 M 的下三角双对角分解结果的乘积 LM4 进行简化操作
        LM4.simplify()

        # 断言以下相等性
        assert M == M4
        assert M2 == M0
        assert N == N4
        assert N2 == N0
        assert M == LM4
        assert LM2 == LM0
        assert N == LN4
        assert LN2 == LN0

    # 创建一个 18x8 的矩阵 M，其中元素为从 1 到 144 的连续整数
    M = Matrix(18, 8, range(1, 145))
    # 将矩阵 M 中的元素转换为浮点数类型
    M = M.applyfunc(lambda i: Float(i))

    # 断言以下相等性
    assert M.bidiagonal_decomposition()[1] == M.bidiagonalize()
    assert M.bidiagonal_decomposition(upper=False)[1] == M.bidiagonalize(upper=False)

    # 对矩阵 M 进行双对角分解，并将结果分别赋值给 a, b, c
    a, b, c = M.bidiagonal_decomposition()
    # 计算 a * b * c - M 的差异
    diff = a * b * c - M
    # 断言差异的最大绝对值小于 10^-12
    assert abs(max(diff)) < 10**-12
def test_diagonalize():
    # 创建一个 2x2 的矩阵 m，元素为 [0, -1, 1, 0]
    m = Matrix(2, 2, [0, -1, 1, 0])
    # 断言调用 m.diagonalize(reals_only=True) 会引发 MatrixError 异常
    raises(MatrixError, lambda: m.diagonalize(reals_only=True))
    # 对矩阵 m 进行对角化，并返回对角化后的矩阵 P 和 D
    P, D = m.diagonalize()
    # 断言 D 是否为对角矩阵
    assert D.is_diagonal()
    # 断言 D 的内容是否与指定的复数矩阵相等
    assert D == Matrix([
                 [-I, 0],
                 [ 0, I]])

    # 确保如果输入为浮点数，则输出也应为浮点数
    m = Matrix(2, 2, [0, .5, .5, 0])
    P, D = m.diagonalize()
    # 断言 D 中的所有元素是否均为 Float 类型
    assert all(isinstance(e, Float) for e in D.values())
    # 断言 P 中的所有元素是否均为 Float 类型
    assert all(isinstance(e, Float) for e in P.values())

    # 调用 m.diagonalize(reals_only=True)，返回的第二个对角矩阵为 D2
    _, D2 = m.diagonalize(reals_only=True)
    # 断言 D 和 D2 是否相等
    assert D == D2

    # 创建一个 4x4 的矩阵 m
    m = Matrix(
        [[0, 1, 0, 0], [1, 0, 0, 0.002], [0.002, 0, 0, 1], [0, 0, 1, 0]])
    # 对矩阵 m 进行对角化，并返回对角化后的矩阵 P 和 D
    P, D = m.diagonalize()
    # 断言 P*D 是否接近于 m*P
    assert allclose(P*D, m*P)


def test_is_diagonalizable():
    # 定义符号变量 a, b, c
    a, b, c = symbols('a b c')
    # 创建一个 2x2 的矩阵 m，元素为 [a, c, c, b]
    m = Matrix(2, 2, [a, c, c, b])
    # 断言矩阵 m 是否为对称矩阵
    assert m.is_symmetric()
    # 断言矩阵 m 是否可对角化
    assert m.is_diagonalizable()
    # 断言一个非对角矩阵是否不可对角化
    assert not Matrix(2, 2, [1, 1, 0, 1]).is_diagonalizable()

    # 创建一个 2x2 的矩阵 m，元素为 [0, -1, 1, 0]
    m = Matrix(2, 2, [0, -1, 1, 0])
    # 断言矩阵 m 是否可对角化
    assert m.is_diagonalizable()
    # 断言在仅考虑实数情况下，矩阵 m 是否不可对角化
    assert not m.is_diagonalizable(reals_only=True)


def test_jordan_form():
    # 创建一个 3x2 的矩阵 m
    m = Matrix(3, 2, [-3, 1, -3, 20, 3, 10])
    # 断言调用 m.jordan_form() 会引发 NonSquareMatrixError 异常
    raises(NonSquareMatrixError, lambda: m.jordan_form())

    # 对两个特定的 4x4 矩阵进行 Jordan 形式计算
    m = Matrix(4, 4, [2, 1, 0, 0,
                    0, 2, 1, 0,
                    0, 0, 2, 0,
                    0, 0, 0, 2
    ])
    P, J = m.jordan_form()
    # 断言矩阵 m 是否等于其 Jordan 形式 J
    assert m == J

    m = Matrix(4, 4, [2, 1, 0, 0,
                    0, 2, 0, 0,
                    0, 0, 2, 1,
                    0, 0, 0, 2
    ])
    P, J = m.jordan_form()
    # 断言矩阵 m 是否等于其 Jordan 形式 J
    assert m == J

    A = Matrix([[ 2,  4,  1,  0],
                [-4,  2,  0,  1],
                [ 0,  0,  2,  4],
                [ 0,  0, -4,  2]])
    P, J = A.jordan_form()
    # 断言简化后的 P*J*P.inv() 是否等于原始矩阵 A
    assert simplify(P*J*P.inv()) == A

    # 断言对于 1x1 的单位矩阵，其 Jordan 形式应为其自身
    assert Matrix(1, 1, [1]).jordan_form() == (Matrix([1]), Matrix([1]))
    # 断言对于 1x1 的单位矩阵，如果不计算变换矩阵，则 Jordan 形式应为其自身
    assert Matrix(1, 1, [1]).jordan_form(calc_transform=False) == Matrix([1])

    # 如果输入的矩阵具有 CRootOf 形式的特征值，则断言调用 m.jordan_form() 会引发 MatrixError 异常
    m = Matrix([[3, 0, 0, 0, -3], [0, -3, -3, 0, 3], [0, 3, 0, 3, 0], [0, 0, 3, 0, 3], [3, 0, 0, 3, 0]])
    raises(MatrixError, lambda: m.jordan_form())

    # 确保如果输入包含浮点数，则输出也应包含浮点数
    m = Matrix([
        [                0.6875, 0.125 + 0.1875*sqrt(3)],
        [0.125 + 0.1875*sqrt(3),                 0.3125]])
    P, J = m.jordan_form()
    # 断言 P 和 J 中的所有元素是否为 Float 类型或等于 0
    assert all(isinstance(x, Float) or x == 0 for x in P)
    assert all(isinstance(x, Float) or x == 0 for x in J)


def test_singular_values():
    x = Symbol('x', real=True)

    A = Matrix([[0, 1*I], [2, 0]])
    # 断言矩阵 A 的奇异值是否按降序排列
    assert A.singular_values() == [2, 1]

    A = eye(3)
    # 将矩阵 A 的 (1, 1) 元素设置为变量 x 的值
    A[1, 1] = x
    # 将矩阵 A 的 (2, 2) 元素设置为常数 5
    A[2, 2] = 5
    # 计算矩阵 A 的奇异值，并将结果存储在 vals 变量中
    vals = A.singular_values()
    # 由于 Abs(x) 的值无法排序，因此测试其集合是否相等
    assert set(vals) == {5, 1, Abs(x)}

    # 创建一个包含三角函数的矩阵 A
    A = Matrix([[sin(x), cos(x)], [-cos(x), sin(x)]])
    # 对矩阵 A 的每个奇异值进行三角化简，并存储在 vals 变量中
    vals = [sv.trigsimp() for sv in A.singular_values()]
    # 断言矩阵 A 的奇异值经过三角化简后是否全部为 1
    assert vals == [S.One, S.One]

    # 创建一个新的矩阵 A，包含四行两列的整数
    A = Matrix([
        [2, 4],
        [1, 3],
        [0, 0],
        [0, 0]
        ])
    # 断言矩阵 A 的奇异值是否为特定的数学表达式的列表
    assert A.singular_values() == \
        [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221))]
    # 断言矩阵 A 的转置的奇异值是否为特定的数学表达式的列表，其中有两个零元素
    assert A.T.singular_values() == \
        [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221)), 0, 0]
def test___eq__():
    # 断言：Matrix 对象与空字典比较应返回 False
    assert (Matrix(
        [[0, 1, 1],
        [1, 0, 0],
        [1, 1, 1]]) == {}) is False


def test_definite():
    # Gilbert Strang 的《线性代数导论》中的例子
    # 正定矩阵
    m = Matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    m = Matrix([[5, 4], [4, 5]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # 半正定矩阵
    m = Matrix([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    m = Matrix([[1, 2], [2, 4]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # Mathematica 文档中的例子
    # 非厄米正定矩阵
    m = Matrix([[2, 3], [4, 8]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # 厄米矩阵
    m = Matrix([[1, 2*I], [-I, 4]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # 符号矩阵示例
    a = Symbol('a', positive=True)
    b = Symbol('b', negative=True)
    m = Matrix([[a, 0, 0], [0, a, 0], [0, 0, a]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    m = Matrix([[b, 0, 0], [0, b, 0], [0, 0, b]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == False
    assert m.is_negative_definite == True
    assert m.is_negative_semidefinite == True
    assert m.is_indefinite == False

    m = Matrix([[a, 0], [0, b]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == False
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == True
    # 创建一个4x4的矩阵，其中每个元素是浮点数
    m = Matrix([
        [0.0228202735623867, 0.00518748979085398,
         -0.0743036351048907, -0.00709135324903921],
        [0.00518748979085398, 0.0349045359786350,
         0.0830317991056637, 0.00233147902806909],
        [-0.0743036351048907, 0.0830317991056637,
         1.15859676366277, 0.340359081555988],
        [-0.00709135324903921, 0.00233147902806909,
         0.340359081555988, 0.928147644848199]
    ])
    # 断言矩阵 m 是正定的
    assert m.is_positive_definite == True
    # 断言矩阵 m 是半正定的
    assert m.is_positive_semidefinite == True
    # 断言矩阵 m 不是非定的
    assert m.is_indefinite == False

    # 对问题 19547 进行测试：https://github.com/sympy/sympy/issues/19547
    # 创建一个3x3的矩阵，其中元素全为整数
    m = Matrix([
        [0, 0, 0],
        [0, 1, 2],
        [0, 2, 1]
    ])
    # 断言矩阵 m 不是正定的
    assert not m.is_positive_definite
    # 断言矩阵 m 不是半正定的
    assert not m.is_positive_semidefinite
# 定义一个测试函数，用于验证 _is_positive_semidefinite_cholesky 函数的行为
def test_positive_semidefinite_cholesky():
    # 导入 _is_positive_semidefinite_cholesky 函数
    from sympy.matrices.eigen import _is_positive_semidefinite_cholesky

    # 创建一个 3x3 的零矩阵 m
    m = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # 断言矩阵 m 是否为正半定的，应为 True
    assert _is_positive_semidefinite_cholesky(m) == True

    # 重新赋值矩阵 m，包含复数项
    m = Matrix([[0, 0, 0], [0, 5, -10*I], [0, 10*I, 5]])
    # 断言矩阵 m 是否为正半定的，应为 False
    assert _is_positive_semidefinite_cholesky(m) == False

    # 重新赋值矩阵 m，包含不符合条件的对角元素
    m = Matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    # 断言矩阵 m 是否为正半定的，应为 False
    assert _is_positive_semidefinite_cholesky(m) == False

    # 重新赋值矩阵 m，符合正半定条件
    m = Matrix([[0, 1], [1, 0]])
    # 断言矩阵 m 是否为正半定的，应为 False
    assert _is_positive_semidefinite_cholesky(m) == False

    # 使用链接中的示例矩阵 m
    m = Matrix([[4, -2, -6], [-2, 10, 9], [-6, 9, 14]])
    # 断言矩阵 m 是否为正半定的，应为 True
    assert _is_positive_semidefinite_cholesky(m) == True

    # 使用链接中的示例矩阵 m
    m = Matrix([[9, -3, 3], [-3, 2, 1], [3, 1, 6]])
    # 断言矩阵 m 是否为正半定的，应为 True
    assert _is_positive_semidefinite_cholesky(m) == True

    # 使用链接中的示例矩阵 m
    m = Matrix([[4, -2, 2], [-2, 1, -1], [2, -1, 5]])
    # 断言矩阵 m 是否为正半定的，应为 True
    assert _is_positive_semidefinite_cholesky(m) == True

    # 使用链接中的示例矩阵 m
    m = Matrix([[1, 2, -1], [2, 5, 1], [-1, 1, 9]])
    # 断言矩阵 m 是否为正半定的，应为 False
    assert _is_positive_semidefinite_cholesky(m) == False


# 定义一个测试函数，用于验证 Matrix 类的 eigenvects 方法在处理特定情况下的行为
def test_issue_20582():
    # 创建一个具有特定内容的 5x5 矩阵 A
    A = Matrix([
        [5, -5, -3, 2, -7],
        [-2, -5, 0, 2, 1],
        [-2, -7, -5, -2, -6],
        [7, 10, 3, 9, -2],
        [4, -10, 3, -8, -4]
    ])
    # 断言 A 的特征向量是否存在
    # XXX 使用干扰测试，因为出现在 CRootOf 中的任意符号可能不唯一。
    assert A.eigenvects()


# 定义一个测试函数，用于验证 Symbol 类的行为
def test_issue_19210():
    # 创建一个符号变量 t
    t = Symbol('t')
    # 创建一个特定的 4x4 矩阵 H
    H = Matrix([[3, 0, 0, 0], [0, 1 , 2, 0], [0, 2, 2, 0], [0, 0, 0, 4]])
    # 计算 (-I * H * t) 的 Jordan 标准形
    A = (-I * H * t).jordan_form()
    # 断言 A 是否等于给定的 Jordan 标准形元组
    assert A == (Matrix([
                    [0, 1,                  0,                0],
                    [0, 0, -4/(-1 + sqrt(17)), 4/(1 + sqrt(17))],
                    [0, 0,                  1,                1],
                    [1, 0,                  0,                0]]), Matrix([
                    [-4*I*t,      0,                         0,                         0],
                    [     0, -3*I*t,                         0,                         0],
                    [     0,      0, t*(-3*I/2 + sqrt(17)*I/2),                         0],
                    [     0,      0,                         0, t*(-sqrt(17)*I/2 - 3*I/2)]]))


# 定义一个测试函数，用于验证 DFT 类的行为
def test_issue_20275():
    # XXX 使用复杂扩展，因为复数指数在 polys.domains 中无法识别
    # 创建一个 3x3 离散傅立叶变换的显式表示 A
    A = DFT(3).as_explicit().expand(complex=True)
    # 计算 A 的特征向量
    eigenvects = A.eigenvects()
    # 断言第一个特征向量是否符合预期
    assert eigenvects[0] == (
        -1, 1,
        [Matrix([[1 - sqrt(3)], [1], [1]])]
    )
    # 断言第二个特征向量是否符合预期
    assert eigenvects[1] == (
        1, 1,
        [Matrix([[1 + sqrt(3)], [1], [1]])]
    )
    # 断言第三个特征向量是否符合预期
    assert eigenvects[2] == (
        -I, 1,
        [Matrix([[0], [-1], [1]])]
    )

    # 创建一个 4x4 离散傅立叶变换的显式表示 A
    A = DFT(4).as_explicit().expand(complex=True)
    # 计算 A 的特征向量
    eigenvects = A.eigenvects()
    # 断言第一个特征向量是否符合预期
    assert eigenvects[0] == (
        -1, 1,
        [Matrix([[-1], [1], [1], [1]])]
    )
    # 断言第二个特征向量是否符合预期
    assert eigenvects[1] == (
        1, 2,
        [Matrix([[1], [0], [1], [0]]), Matrix([[2], [1], [0], [1]])]
    )
    # 断言第三个特征向量是否与预期相符
    assert eigenvects[2] == (
        -I, 1,
        [Matrix([[0], [-1], [0], [1]])]
    )

    # XXX 我们跳过对部分特征向量的测试，因为它们在表达式树变化时非常复杂且易碎
    # 创建一个大小为 5 的离散傅立叶变换对象，转换为显式表示，并展开为复数形式
    A = DFT(5).as_explicit().expand(complex=True)
    # 计算变换矩阵 A 的特征向量
    eigenvects = A.eigenvects()
    # 断言第一个特征向量是否与预期相符
    assert eigenvects[0] == (
        -1, 1,
        [Matrix([[1 - sqrt(5)], [1], [1], [1], [1]])]
    )
    # 断言第二个特征向量是否与预期相符
    assert eigenvects[1] == (
        1, 2,
        [Matrix([[S(1)/2 + sqrt(5)/2], [0], [1], [1], [0]]),
         Matrix([[S(1)/2 + sqrt(5)/2], [1], [0], [0], [1]])]
    )
# 定义一个测试函数，用于检查问题编号为 20752 的情况
def test_issue_20752():
    # 创建一个符号变量 b，确保其非零
    b = symbols('b', nonzero=True)
    # 创建一个 3x3 的矩阵 m，其中除了对角线上的元素 b 外，其余都是 0
    m = Matrix([[0, 0, 0], [0, b, 0], [0, 0, b]])
    # 断言矩阵 m 的正半定性为 None
    assert m.is_positive_semidefinite is None


# 定义一个测试函数，用于检查问题编号为 25282 的情况
def test_issue_25282():
    # 初始化两个列表 dd 和 sd，它们均包含 11 个 0 和 1 个 1
    dd = sd = [0] * 11 + [1]
    # 初始化列表 ds，包含特定的数字序列
    ds = [2, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    # 创建列表 ss，复制 ds 的内容并将第 8 个元素替换为 2
    ss = ds.copy()
    ss[8] = 2

    # 定义一个函数 rotate，用于将列表 x 的元素按索引 i 进行循环移动
    def rotate(x, i):
        return x[i:] + x[:i]

    # 初始化一个空列表 mat
    mat = []
    # 使用循环生成 12 个旋转后的 ss 和 sd 的组合，添加到 mat 中
    for i in range(12):
        mat.append(rotate(ss, i) + rotate(sd, i))
    # 使用循环生成 12 个旋转后的 ds 和 dd 的组合，添加到 mat 中
    for i in range(12):
        mat.append(rotate(ds, i) + rotate(dd, i))

    # 断言矩阵 mat 的特征值的和为 24
    assert sum(Matrix(mat).eigenvals().values()) == 24
```