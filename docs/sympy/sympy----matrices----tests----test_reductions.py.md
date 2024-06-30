# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_reductions.py`

```
# 导入所需的符号、矩阵和测试相关模块和函数
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.testing.pytest import raises
from sympy.matrices import Matrix, zeros, eye
from sympy.core.symbol import Symbol
from sympy.core.numbers import Rational
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.simplify import simplify
from sympy.abc import x

# Matrix tests
# 定义行操作测试函数
def test_row_op():
    # 创建一个3x3的单位矩阵
    e = eye(3)

    # 检查各种无效操作是否引发 ValueError 异常
    raises(ValueError, lambda: e.elementary_row_op("abc"))
    raises(ValueError, lambda: e.elementary_row_op())
    raises(ValueError, lambda: e.elementary_row_op('n->kn', row=5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->kn', row=-5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=1, row2=5))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=5, row2=1))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=-5, row2=1))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=1, row2=-5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=1, row2=5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=5, row2=1, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=-5, row2=1, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=1, row2=-5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=1, row2=1, k=5))

    # 测试不同设置参数的方法，验证行操作的正确性
    assert e.elementary_row_op("n->kn", 0, 5) == Matrix([[5, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->kn", 1, 5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->kn", row=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->kn", row1=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_row_op("n<->m", 0, 1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_row_op("n<->m", row1=0, row2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_row_op("n<->m", row=0, row2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->n+km", 0, 5, 1) == Matrix([[1, 5, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->n+km", row=0, k=5, row2=1) == Matrix([[1, 5, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->n+km", row1=0, k=5, row2=1) == Matrix([[1, 5, 0], [0, 1, 0], [0, 0, 1]])

    # 确保矩阵的大小未发生变化
    a = Matrix(2, 3, [0]*6)
    assert a.elementary_row_op("n->kn", 1, 5) == Matrix(2, 3, [0]*6)
    assert a.elementary_row_op("n<->m", 0, 1) == Matrix(2, 3, [0]*6)
    assert a.elementary_row_op("n->n+km", 0, 5, 1) == Matrix(2, 3, [0]*6)


# 定义列操作测试函数
def test_col_op():
    # 创建一个3x3的单位矩阵
    e = eye(3)

    # 检查各种无效操作是否引发 ValueError 异常
    raises(ValueError, lambda: e.elementary_col_op("abc"))
    raises(ValueError, lambda: e.elementary_col_op())
    raises(ValueError, lambda: e.elementary_col_op('n->kn', col=5, k=5))
    # 检查对 elementary_col_op 方法的异常输入是否会引发 ValueError 异常
    raises(ValueError, lambda: e.elementary_col_op('n->kn', col=-5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=1, col2=5))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=5, col2=1))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=-5, col2=1))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=1, col2=-5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=5, col2=1, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=-5, col2=1, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=-5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=1, k=5))

    # 测试 elementary_col_op 方法不同参数设置的结果是否符合预期
    assert e.elementary_col_op("n->kn", 0, 5) == Matrix([[5, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->kn", 1, 5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->kn", col=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->kn", col1=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op("n<->m", 0, 1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op("n<->m", col1=0, col2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op("n<->m", col=0, col2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->n+km", 0, 5, 1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->n+km", col=0, k=5, col2=1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->n+km", col1=0, k=5, col2=1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])

    # 确保矩阵在应用 elementary_col_op 方法后没有改变尺寸
    a = Matrix(2, 3, [0]*6)
    assert a.elementary_col_op("n->kn", 1, 5) == Matrix(2, 3, [0]*6)
    assert a.elementary_col_op("n<->m", 0, 1) == Matrix(2, 3, [0]*6)
    assert a.elementary_col_op("n->n+km", 0, 5, 1) == Matrix(2, 3, [0]*6)
def test_is_echelon():
    # 创建一个大小为 3x3 的零矩阵
    zro = zeros(3)
    # 创建一个大小为 3x3 的单位矩阵
    ident = eye(3)

    # 断言零矩阵是梯形矩阵
    assert zro.is_echelon
    # 断言单位矩阵是梯形矩阵
    assert ident.is_echelon

    # 创建一个空的 0x0 矩阵
    a = Matrix(0, 0, [])
    # 断言空矩阵是梯形矩阵
    assert a.is_echelon

    # 创建一个大小为 2x3 的矩阵，包含数据 [3, 2, 1, 0, 0, 6]
    a = Matrix(2, 3, [3, 2, 1, 0, 0, 6])
    # 断言该矩阵是梯形矩阵
    assert a.is_echelon

    # 创建一个大小为 2x3 的矩阵，包含数据 [0, 0, 6, 3, 2, 1]
    a = Matrix(2, 3, [0, 0, 6, 3, 2, 1])
    # 断言该矩阵不是梯形矩阵
    assert not a.is_echelon

    # 创建一个包含符号 'x' 的大小为 3x1 的矩阵
    x = Symbol('x')
    a = Matrix(3, 1, [x, 0, 0])
    # 断言该矩阵是梯形矩阵
    assert a.is_echelon

    # 创建一个包含符号 'x' 的大小为 3x1 的矩阵
    a = Matrix(3, 1, [x, x, 0])
    # 断言该矩阵不是梯形矩阵
    assert not a.is_echelon

    # 创建一个大小为 3x3 的矩阵，包含数据 [0, 0, 0, 1, 2, 3, 0, 0, 0]
    a = Matrix(3, 3, [0, 0, 0, 1, 2, 3, 0, 0, 0])
    # 断言该矩阵不是梯形矩阵
    assert not a.is_echelon


def test_echelon_form():
    # 梯形形式不唯一，但结果必须与原矩阵行等价，并且处于梯形形式中。

    # 创建一个大小为 3x3 的零矩阵
    a = zeros(3)
    # 创建一个大小为 3x3 的单位矩阵
    e = eye(3)

    # 断言零矩阵的梯形形式等于其自身
    assert a.echelon_form() == a
    # 断言单位矩阵的梯形形式等于其自身
    assert e.echelon_form() == e

    # 创建一个空的 0x0 矩阵
    a = Matrix(0, 0, [])
    # 断言空矩阵的梯形形式等于其自身
    assert a.echelon_form() == a

    # 创建一个大小为 1x1 的矩阵，包含数据 [5]
    a = Matrix(1, 1, [5])
    # 断言该矩阵的梯形形式等于其自身
    assert a.echelon_form() == a

    # 现在开始真正的测试

    # 定义函数，验证零空间中的行向量和零向量
    def verify_row_null_space(mat, rows, nulls):
        for v in nulls:
            # 断言零空间中每个向量与矩阵乘积结果都是零向量
            assert all(t.is_zero for t in a_echelon*v)
        for v in rows:
            if not all(t.is_zero for t in v):
                # 断言非零行向量与其与梯形矩阵转置乘积结果不全是零向量
                assert not all(t.is_zero for t in a_echelon*v.transpose())

    # 创建一个大小为 3x3 的矩阵，包含数据 [1, 2, 3, 4, 5, 6, 7, 8, 9]
    a = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 零空间中的零向量
    nulls = [Matrix([
                     [ 1],
                     [-2],
                     [ 1]])]
    # 矩阵的行向量
    rows = [a[i, :] for i in range(a.rows)]
    # 获取矩阵的梯形形式
    a_echelon = a.echelon_form()
    # 断言矩阵的梯形形式是梯形矩阵
    assert a_echelon.is_echelon
    # 验证零空间中的行向量和零向量
    verify_row_null_space(a, rows, nulls)


    # 创建一个大小为 3x3 的矩阵，包含数据 [1, 2, 3, 4, 5, 6, 7, 8, 8]
    a = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 8])
    # 零空间中的零向量
    nulls = []
    # 矩阵的行向量
    rows = [a[i, :] for i in range(a.rows)]
    # 获取矩阵的梯形形式
    a_echelon = a.echelon_form()
    # 断言矩阵的梯形形式是梯形矩阵
    assert a_echelon.is_echelon
    # 验证零空间中的行向量和零向量
    verify_row_null_space(a, rows, nulls)

    # 创建一个大小为 3x3 的矩阵，包含数据 [2, 1, 3, 0, 0, 0, 2, 1, 3]
    a = Matrix(3, 3, [2, 1, 3, 0, 0, 0, 2, 1, 3])
    # 零空间中的零向量
    nulls = [Matrix([
             [Rational(-1, 2)],
             [   1],
             [   0]]),
             Matrix([
             [Rational(-3, 2)],
             [   0],
             [   1]])]
    # 矩阵的行向量
    rows = [a[i, :] for i in range(a.rows)]
    # 获取矩阵的梯形形式
    a_echelon = a.echelon_form()
    # 断言矩阵的梯形形式是梯形矩阵
    assert a_echelon.is_echelon
    # 验证零空间中的行向量和零向量
    verify_row_null_space(a, rows, nulls)

    # 创建一个大小为 3x3 的矩阵，包含数据 [2, 1, 3, 0, 0, 0, 1, 1, 3]
    a = Matrix(3, 3, [2, 1, 3, 0, 0, 0, 1, 1, 3])
    # 零空间中的零向量
    nulls = [Matrix([
             [   0],
             [  -3],
             [   1]])]
    # 矩阵的行向量
    rows = [a[i, :] for i in range(a.rows)]
    # 获取矩阵的梯形形式
    # 创建一个包含单个 Matrix 对象的列表，Matrix 对象包含一个特定的 3x1 的矩阵
    nulls = [Matrix([
             [-1],
             [1],
             [0]])]
    
    # 使用列表推导式创建一个列表，其中包含矩阵 a 每一行的引用
    rows = [a[i, :] for i in range(a.rows)]
    
    # 对矩阵 a 进行行梯形化，并将结果赋给 a_echelon 变量
    a_echelon = a.echelon_form()
    
    # 断言 a_echelon 是行梯形矩阵（即具有梯形矩阵形式）
    assert a_echelon.is_echelon
    
    # 调用 verify_row_null_space 函数，验证矩阵 a 的行空间和零空间
    verify_row_null_space(a, rows, nulls)
# 定义名为 test_rref 的测试函数，用于测试矩阵的行简化阶梯形态（Reduced Row Echelon Form，RREF）
def test_rref():
    # 创建一个空的 0x0 矩阵 e
    e = Matrix(0, 0, [])
    # 断言 e 的行简化阶梯形态等于自身，不包含主元信息
    assert e.rref(pivots=False) == e

    # 创建一个 1x1 的单位矩阵 e 和一个 1x1 的矩阵 a，它包含元素 5
    e = Matrix(1, 1, [1])
    a = Matrix(1, 1, [5])
    # 断言 e 和 a 的行简化阶梯形态均等于自身，不包含主元信息
    assert e.rref(pivots=False) == a.rref(pivots=False) == e

    # 创建一个 3x1 的矩阵 a，包含元素 [1, 2, 3]
    a = Matrix(3, 1, [1, 2, 3])
    # 断言 a 的行简化阶梯形态，不包含主元信息，应该等于 3x1 矩阵 [[1], [0], [0]]
    assert a.rref(pivots=False) == Matrix([[1], [0], [0]])

    # 创建一个 1x3 的矩阵 a，包含元素 [1, 2, 3]
    a = Matrix(1, 3, [1, 2, 3])
    # 断言 a 的行简化阶梯形态，不包含主元信息，应该等于 1x3 矩阵 [[1, 2, 3]]
    assert a.rref(pivots=False) == Matrix([[1, 2, 3]])

    # 创建一个 3x3 的矩阵 a，包含元素 [1, 2, 3, 4, 5, 6, 7, 8, 9]
    a = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 断言 a 的行简化阶梯形态，不包含主元信息，应该等于 3x3 矩阵
    # [[1, 0, -1],
    #  [0, 1,  2],
    #  [0, 0,  0]]
    assert a.rref(pivots=False) == Matrix([
                                     [1, 0, -1],
                                     [0, 1,  2],
                                     [0, 0,  0]])

    # 创建一个 3x3 的矩阵 a，包含元素 [1, 2, 3, 1, 2, 3, 1, 2, 3]
    a = Matrix(3, 3, [1, 2, 3, 1, 2, 3, 1, 2, 3])
    # 创建三个预期的 3x3 矩阵 b、c、d，分别包含特定的行简化阶梯形态
    b = Matrix(3, 3, [1, 2, 3, 0, 0, 0, 0, 0, 0])
    c = Matrix(3, 3, [0, 0, 0, 1, 2, 3, 0, 0, 0])
    d = Matrix(3, 3, [0, 0, 0, 0, 0, 0, 1, 2, 3])
    # 断言 a、b、c、d 的行简化阶梯形态，不包含主元信息，均等于矩阵 b
    assert a.rref(pivots=False) == \
            b.rref(pivots=False) == \
            c.rref(pivots=False) == \
            d.rref(pivots=False) == b

    # 创建一个 3x3 的单位矩阵 e 和一个 3x3 的零矩阵 z
    e = eye(3)
    z = zeros(3)
    # 断言 e 和 z 的行简化阶梯形态，不包含主元信息，分别等于自身
    assert e.rref(pivots=False) == e
    assert z.rref(pivots=False) == z

    # 创建一个 4x7 的矩阵 a，包含特定的数值
    a = Matrix([
            [ 0, 0,  1,  2,  2, -5,  3],
            [-1, 5,  2,  2,  1, -7,  5],
            [ 0, 0, -2, -3, -3,  8, -5],
            [-1, 5,  0, -1, -2,  1,  0]])
    # 调用矩阵 a 的行简化阶梯形态方法，同时获取主元偏移量信息
    mat, pivot_offsets = a.rref()
    # 断言 mat 的结果应与预期的 4x7 矩阵相等
    assert mat == Matrix([
                     [1, -5, 0, 0, 1,  1, -1],
                     [0,  0, 1, 0, 0, -1,  1],
                     [0,  0, 0, 1, 1, -2,  1],
                     [0,  0, 0, 0, 0,  0,  0]])
    # 断言 pivot_offsets 的结果应为元组 (0, 2, 3)
    assert pivot_offsets == (0, 2, 3)

    # 创建一个 4x4 的有理数矩阵 a，包含特定的有理数元素
    a = Matrix([[Rational(1, 19),  Rational(1, 5),    2,    3],
                        [   4,    5,    6,    7],
                        [   8,    9,   10,   11],
                        [  12,   13,   14,   15]])
    # 断言 a 的行简化阶梯形态，不包含主元信息，应与预期的 4x4 矩阵相等
    assert a.rref(pivots=False) == Matrix([
                                         [1, 0, 0, Rational(-76, 157)],
                                         [0, 1, 0,  Rational(-5, 157)],
                                         [0, 0, 1, Rational(238, 157)],
                                         [0, 0, 0,       0]])

    # 创建一个 2x3 的符号矩阵 a，包含符号 x 和数值 1、sqrt(x)
    x = Symbol('x')
    a = Matrix(2, 3, [x, 1, 1, sqrt(x), x, 1])
    # 使用 zip 函数遍历 a 的行简化阶梯形态，与预期的列表进行逐一断言
    for i, j in zip(a.rref(pivots=False),
            [1, 0, sqrt(x)*(-x + 1)/(-x**Rational(5, 2) + x),
                0, 1, 1/(sqrt(x) + x + 1)]):
        # 断言简化后的结果与预期值 j 经过简化后是否为零
        assert simplify(i - j).is_zero


# 定义名为 test_rref_rhs 的测试函数，用于测试矩阵的右侧向量进行行简化阶梯形态的操作
def test_rref_rhs():
    # 定义符号 a、b、c、d
    a, b, c, d = symbols('a b c d')
    # 创建一个 4x2 矩阵 A 和一个 4x1 矩阵 B，用于测试右侧向量的简化操作
    A = Matrix([[0, 0], [0, 0], [1, 2], [3, 4]])
    B = Matrix([a, b, c, d])
    # 断言
    # 应用行变换 'n->kn' 到矩阵 C 的第 5 行，乘数 k=2
    F = C.elementary_row_op('n->kn', row=5, k=2)
    # 断言确认矩阵 D 的第 5 行是否变为 [[0, 2, 1, 3]]
    assert(D[5, :] == Matrix([[0, 2, 1, 3]]))
    # 断言确认矩阵 E 的第 5 行是否变为 [[0, 3, 0, 14]]
    assert(E[5, :] == Matrix([[0, 3, 0, 14]]))
    # 断言确认矩阵 F 的第 5 行是否变为 [[16, 30, 0, 12]]
    assert(F[5, :] == Matrix([[16, 30, 0, 12]]))
    # 测试行/列索引超出范围的情况
    raises(ValueError, lambda: C.elementary_row_op('n<->m', row1=2, row2=6))
    raises(ValueError, lambda: C.elementary_row_op('n->kn', row=7, k=2))
    raises(ValueError, lambda: C.elementary_row_op('n->n+km', row1=-1, row2=5, k=2))
# 定义一个函数用于测试矩阵的秩
def test_rank():
    # 创建一个 2x2 的矩阵对象 M，包含符号变量 x
    m = Matrix([[1, 2], [x, 1 - 1/x]])
    # 断言矩阵 M 的秩为 2
    assert m.rank() == 2
    # 创建一个 3x3 的矩阵对象 N，填充为 1 到 9 的整数序列
    n = Matrix(3, 3, range(1, 10))
    # 断言矩阵 N 的秩为 2
    assert n.rank() == 2
    # 创建一个 3x1 的零矩阵对象 P
    p = zeros(3)
    # 断言矩阵 P 的秩为 0
    assert p.rank() == 0

# 定义一个函数用于测试解决 GitHub 上的问题 #11434
def test_issue_11434():
    # 定义一组符号变量
    ax, ay, bx, by, cx, cy, dx, dy, ex, ey, t0, t1 = \
        symbols('a_x a_y b_x b_y c_x c_y d_x d_y e_x e_y t_0 t_1')
    # 创建一个 5x5 的符号矩阵 M
    M = Matrix([[ax, ay, ax*t0, ay*t0, 0],
                [bx, by, bx*t0, by*t0, 0],
                [cx, cy, cx*t0, cy*t0, 1],
                [dx, dy, dx*t0, dy*t0, 1],
                [ex, ey, 2*ex*t1 - ex*t0, 2*ey*t1 - ey*t0, 0]])
    # 断言矩阵 M 的秩为 4
    assert M.rank() == 4

# 定义一个函数用于测试解决 Stack Overflow 上的回归问题
def test_rank_regression_from_so():
    # 定义符号变量 nu 和 lamb
    nu, lamb = symbols('nu, lambda')
    # 创建一个 4x4 的符号矩阵 A
    A = Matrix([[-3*nu,         1,                  0,  0],
                [ 3*nu, -2*nu - 1,                  2,  0],
                [    0,      2*nu, (-1*nu) - lamb - 2,  3],
                [    0,         0,          nu + lamb, -3]])
    # 预期的行阶梯形式化简矩阵
    expected_reduced = Matrix([[1, 0, 0, 1/(nu**2*(-lamb - nu))],
                               [0, 1, 0,    3/(nu*(-lamb - nu))],
                               [0, 0, 1,         3/(-lamb - nu)],
                               [0, 0, 0,                      0]])
    # 预期的主元位置
    expected_pivots = (0, 1, 2)

    # 对矩阵 A 进行行阶梯形式化简
    reduced, pivots = A.rref()

    # 断言化简后的矩阵与预期结果相差的简化结果为零矩阵
    assert simplify(expected_reduced - reduced) == zeros(*A.shape)
    # 断言主元位置与预期的相符
    assert pivots == expected_pivots

# 定义一个函数用于测试解决 GitHub 上的问题 #15872
def test_issue_15872():
    # 创建一个 4x4 的矩阵 A
    A = Matrix([[1, 1, 1, 0], [-2, -1, 0, -1], [0, 0, -1, -1], [0, 0, 2, 1]])
    # 创建一个 B 矩阵，其元素为 A 减去单位矩阵乘以标量 I
    B = A - Matrix.eye(4) * I
    # 断言矩阵 B 的秩为 3
    assert B.rank() == 3
    # 断言矩阵 B 的平方的秩为 2
    assert (B**2).rank() == 2
    # 断言矩阵 B 的立方的秩为 2
    assert (B**3).rank() == 2
```