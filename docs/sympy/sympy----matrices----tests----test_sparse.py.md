# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_sparse.py`

```
from sympy.core.numbers import (Float, I, Rational)  # 导入 Float, I, Rational 类
from sympy.core.singleton import S  # 导入 S 单例
from sympy.core.symbol import (Symbol, symbols)  # 导入 Symbol, symbols 符号类
from sympy.functions.elementary.complexes import Abs  # 导入 Abs 函数
from sympy.polys.polytools import PurePoly  # 导入 PurePoly 多项式工具类
from sympy.matrices import \
    Matrix, MutableSparseMatrix, ImmutableSparseMatrix, SparseMatrix, eye, \
    ones, zeros, ShapeError, NonSquareMatrixError  # 导入矩阵相关类和异常
from sympy.testing.pytest import raises  # 导入 raises 函数用于测试

# 定义测试函数，用于测试稀疏矩阵的创建
def test_sparse_creation():
    a = SparseMatrix(2, 2, {(0, 0): [[1, 2], [3, 4]]})  # 创建一个稀疏矩阵 a
    assert a == SparseMatrix([[1, 2], [3, 4]])  # 断言 a 等于指定的稀疏矩阵
    a = SparseMatrix(2, 2, {(0, 0): [[1, 2]]})  # 创建另一个稀疏矩阵 a
    assert a == SparseMatrix([[1, 2], [0, 0]])  # 断言 a 等于指定的稀疏矩阵
    a = SparseMatrix(2, 2, {(0, 0): [1, 2]})  # 创建第三个稀疏矩阵 a
    assert a == SparseMatrix([[1, 0], [2, 0]])  # 断言 a 等于指定的稀疏矩阵

# 定义测试函数，用于测试稀疏矩阵的各种操作
def test_sparse_matrix():
    def sparse_eye(n):
        return SparseMatrix.eye(n)  # 返回大小为 n 的单位稀疏矩阵

    def sparse_zeros(n):
        return SparseMatrix.zeros(n)  # 返回大小为 n 的全零稀疏矩阵

    # 测试创建参数异常的情况
    raises(TypeError, lambda: SparseMatrix(1, 2))

    a = SparseMatrix((
        (1, 0),
        (0, 1)
    ))  # 创建一个稀疏矩阵 a
    assert SparseMatrix(a) == a  # 断言创建的稀疏矩阵与 a 相等

    from sympy.matrices import MutableDenseMatrix
    a = MutableSparseMatrix([])  # 创建一个可变稀疏矩阵 a
    b = MutableDenseMatrix([1, 2])  # 创建一个可变稠密矩阵 b
    assert a.row_join(b) == b  # 断言行连接操作的结果与 b 相等
    assert a.col_join(b) == b  # 断言列连接操作的结果与 b 相等
    assert type(a.row_join(b)) == type(a)  # 断言行连接操作的结果类型与 a 相同
    assert type(a.col_join(b)) == type(a)  # 断言列连接操作的结果类型与 a 相同

    # 确保对 0 x n 矩阵进行正确的堆叠操作
    sparse_matrices = [SparseMatrix.zeros(0, n) for n in range(4)]
    assert SparseMatrix.hstack(*sparse_matrices) == Matrix(0, 6, [])  # 断言水平堆叠结果正确
    sparse_matrices = [SparseMatrix.zeros(n, 0) for n in range(4)]
    assert SparseMatrix.vstack(*sparse_matrices) == Matrix(6, 0, [])  # 断言垂直堆叠结果正确

    # 测试元素赋值操作
    a = SparseMatrix((
        (1, 0),
        (0, 1)
    ))  # 创建一个稀疏矩阵 a

    a[3] = 4  # 设置索引为 3 的元素为 4
    assert a[1, 1] == 4  # 断言索引 (1, 1) 的值为 4
    a[3] = 1  # 再次设置索引为 3 的元素为 1

    a[0, 0] = 2  # 设置索引为 (0, 0) 的元素为 2
    assert a == SparseMatrix((
        (2, 0),
        (0, 1)
    ))  # 断言稀疏矩阵 a 的内容与指定的稀疏矩阵相等
    a[1, 0] = 5  # 设置索引为 (1, 0) 的元素为 5
    assert a == SparseMatrix((
        (2, 0),
        (5, 1)
    ))  # 断言稀疏矩阵 a 的内容与指定的稀疏矩阵相等
    a[1, 1] = 0  # 设置索引为 (1, 1) 的元素为 0
    assert a == SparseMatrix((
        (2, 0),
        (5, 0)
    ))  # 断言稀疏矩阵 a 的内容与指定的稀疏矩阵相等
    assert a.todok() == {(0, 0): 2, (1, 0): 5}  # 断言将稀疏矩阵 a 转换为 dok 格式的结果符合预期

    # 测试矩阵乘法
    a = SparseMatrix((
        (1, 2),
        (3, 1),
        (0, 6),
    ))  # 创建稀疏矩阵 a

    b = SparseMatrix((
        (1, 2),
        (3, 0),
    ))  # 创建稀疏矩阵 b

    c = a*b  # 计算稀疏矩阵 a 和 b 的乘积
    assert c[0, 0] == 7  # 断言乘积结果的特定元素值正确
    assert c[0, 1] == 2  # 断言乘积结果的特定元素值正确
    assert c[1, 0] == 6  # 断言乘积结果的特定元素值正确
    assert c[1, 1] == 6  # 断言乘积结果的特定元素值正确
    assert c[2, 0] == 18  # 断言乘积结果的特定元素值正确
    assert c[2, 1] == 0  # 断言乘积结果的特定元素值正确

    try:
        eval('c = a @ b')
    except SyntaxError:
        pass
    else:
        assert c[0, 0] == 7  # 断言乘积结果的特定元素值正确
        assert c[0, 1] == 2  # 断言乘积结果的特定元素值正确
        assert c[1, 0] == 6  # 断言乘积结果的特定元素值正确
        assert c[1, 1] == 6  # 断言乘积结果的特定元素值正确
        assert c[2, 0] == 18  # 断言乘积结果的特定元素值正确
        assert c[2, 1] == 0  # 断言乘积结果的特定元素值正确

    x = Symbol("x")  # 创建符号变量 x

    c = b * Symbol("x")  # 稀疏矩阵 b 与符号变量 x 的乘积
    assert isinstance(c, SparseMatrix)  # 断言结果是稀疏矩阵类型
    assert c[0, 0] == x  # 断言乘积结果的特定元素值正确
    assert c[0, 1] == 2*x
    # 断言检查矩阵 c 中第 (1, 1) 元素是否为 0
    assert c[1, 1] == 0

    # 测试矩阵乘法的功能
    A = SparseMatrix([[2, 3], [4, 5]])
    assert (A**5)[:] == [6140, 8097, 10796, 14237]
    A = SparseMatrix([[2, 1, 3], [4, 2, 4], [6, 12, 1]])
    assert (A**3)[:] == [290, 262, 251, 448, 440, 368, 702, 954, 433]

    # 测试稀疏矩阵的创建与基本操作
    x = Symbol("x")
    a = SparseMatrix([[x, 0], [0, 0]])
    m = a
    assert m.cols == m.rows
    assert m.cols == 2
    assert m[:] == [x, 0, 0, 0]
    b = SparseMatrix(2, 2, [x, 0, 0, 0])
    m = b
    assert m.cols == m.rows
    assert m.cols == 2
    assert m[:] == [x, 0, 0, 0]

    # 断言矩阵 a 和 b 是否相等
    assert a == b

    # 测试稀疏单位矩阵的创建与删除行列操作
    S = sparse_eye(3)
    S.row_del(1)
    assert S == SparseMatrix([
                             [1, 0, 0],
                             [0, 0, 1]])

    S = sparse_eye(3)
    S.col_del(1)
    assert S == SparseMatrix([
                             [1, 0],
                             [0, 0],
                             [0, 1]])

    # 测试稀疏单位矩阵的元素修改和行列交换操作
    S = SparseMatrix.eye(3)
    S[2, 1] = 2
    S.col_swap(1, 0)
    assert S == SparseMatrix([
        [0, 1, 0],
        [1, 0, 0],
        [2, 0, 1]])
    S.row_swap(0, 1)
    assert S == SparseMatrix([
        [1, 0, 0],
        [0, 1, 0],
        [2, 0, 1]])

    # 测试稀疏矩阵的复制和删除行列操作
    a = SparseMatrix(1, 2, [1, 2])
    b = a.copy()
    c = a.copy()
    assert a[0] == 1
    a.row_del(0)
    assert a == SparseMatrix(0, 2, [])
    b.col_del(1)
    assert b == SparseMatrix(1, 1, [1])

    # 测试稀疏矩阵与普通矩阵之间的转换和比较
    assert SparseMatrix([[1, 2, 3], [1, 2], [1]]) == Matrix([
        [1, 2, 3],
        [1, 2, 0],
        [1, 0, 0]])
    assert SparseMatrix(4, 4, {(1, 1): sparse_eye(2)}) == Matrix([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]])
    raises(ValueError, lambda: SparseMatrix(1, 1, {(1, 1): 1}))
    assert SparseMatrix(1, 2, [1, 2]).tolist() == [[1, 2]]
    assert SparseMatrix(2, 2, [1, [2, 3]]).tolist() == [[1, 0], [2, 3]]
    raises(ValueError, lambda: SparseMatrix(2, 2, [1]))
    raises(ValueError, lambda: SparseMatrix(1, 1, [[1, 2]]))

    # 测试稀疏矩阵的属性和自动调整尺寸
    assert SparseMatrix([.1]).has(Float)
    assert SparseMatrix(None, {(0, 1): 0}).shape == (0, 0)
    assert SparseMatrix(None, {(0, 1): 1}).shape == (1, 2)
    assert SparseMatrix(None, None, {(0, 1): 1}).shape == (1, 2)
    raises(ValueError, lambda: SparseMatrix(None, 1, [[1, 2]]))
    raises(ValueError, lambda: SparseMatrix(1, None, [[1, 2]]))
    raises(ValueError, lambda: SparseMatrix(3, 3, {(0, 0): ones(2), (1, 1): 2}))

    # 测试稀疏矩阵的行列式计算功能
    x, y = Symbol('x'), Symbol('y')

    assert SparseMatrix(1, 1, [0]).det() == 0
    assert SparseMatrix([[1]]).det() == 1
    assert SparseMatrix(((-3, 2), (8, -5))).det() == -1
    assert SparseMatrix(((x, 1), (y, 2*y))).det() == 2*x*y - y
    assert SparseMatrix(( (1, 1, 1),
                          (1, 2, 3),
                          (1, 3, 6) )).det() == 1
    assert SparseMatrix(( ( 3, -2,  0, 5),
                          (-2,  1, -2, 2),
                          ( 0, -2,  5, 0),
                          ( 5,  0,  3, 4) )).det() == -289
    # 断言：对于给定的稀疏矩阵，计算其行列式，判断是否等于0
    assert SparseMatrix(( ( 1,  2,  3,  4),
                          ( 5,  6,  7,  8),
                          ( 9, 10, 11, 12),
                          (13, 14, 15, 16) )).det() == 0

    # 断言：对于给定的稀疏矩阵，计算其行列式，判断是否等于275
    assert SparseMatrix(( (3, 2, 0, 0, 0),
                          (0, 3, 2, 0, 0),
                          (0, 0, 3, 2, 0),
                          (0, 0, 0, 3, 2),
                          (2, 0, 0, 0, 3) )).det() == 275

    # 断言：对于给定的稀疏矩阵，计算其行列式，判断是否等于-55
    assert SparseMatrix(( (1, 0,  1,  2, 12),
                          (2, 0,  1,  1,  4),
                          (2, 1,  1, -1,  3),
                          (3, 2, -1,  1,  8),
                          (1, 1,  1,  0,  6) )).det() == -55

    # 断言：对于给定的稀疏矩阵，计算其行列式，判断是否等于11664
    assert SparseMatrix(( (-5,  2,  3,  4,  5),
                          ( 1, -4,  3,  4,  5),
                          ( 1,  2, -3,  4,  5),
                          ( 1,  2,  3, -2,  5),
                          ( 1,  2,  3,  4, -1) )).det() == 11664

    # 断言：对于给定的稀疏矩阵，计算其行列式，判断是否等于60
    assert SparseMatrix(( ( 3,  0,  0, 0),
                          (-2,  1,  0, 0),
                          ( 0, -2,  5, 0),
                          ( 5,  0,  3, 4) )).det() == 60

    # 断言：对于给定的稀疏矩阵，计算其行列式，判断是否等于0
    assert SparseMatrix(( ( 1,  0,  0,  0),
                          ( 5,  0,  0,  0),
                          ( 9, 10, 11, 0),
                          (13, 14, 15, 16) )).det() == 0

    # 断言：对于给定的稀疏矩阵，计算其行列式，判断是否等于243
    assert SparseMatrix(( (3, 2, 0, 0, 0),
                          (0, 3, 2, 0, 0),
                          (0, 0, 3, 2, 0),
                          (0, 0, 0, 3, 2),
                          (0, 0, 0, 0, 3) )).det() == 243

    # 断言：对于给定的稀疏矩阵，计算其行列式，判断是否等于123
    assert SparseMatrix(( ( 2,  7, -1, 3, 2),
                          ( 0,  0,  1, 0, 1),
                          (-2,  0,  7, 0, 2),
                          (-3, -2,  4, 5, 3),
                          ( 1,  0,  0, 0, 1) )).det() == 123

    # test_slicing 测试切片功能

    # 创建一个4x4的单位稀疏矩阵，并断言其前3x3切片与3x3单位稀疏矩阵相等
    m0 = sparse_eye(4)
    assert m0[:3, :3] == sparse_eye(3)

    # 创建一个4x4的零稀疏矩阵，并断言其2x2切片与2x2零稀疏矩阵相等
    assert m0[2:4, 0:2] == sparse_zeros(2)

    # 创建一个3x3的稀疏矩阵，元素为i+j，并断言其第一行与一个3x3稀疏矩阵相等
    m1 = SparseMatrix(3, 3, lambda i, j: i + j)
    assert m1[0, :] == SparseMatrix(1, 3, (0, 1, 2))

    # 创建一个3x3的稀疏矩阵，元素为i+j，并断言其第2到3行的第1列与一个2x1稀疏矩阵相等
    assert m1[1:3, 1] == SparseMatrix(2, 1, (2, 3))

    # 创建一个4x4的稀疏矩阵，并断言其每一列的最后一个元素与一个4x1稀疏矩阵相等
    m2 = SparseMatrix(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    assert m2[:, -1] == SparseMatrix(4, 1, [3, 7, 11, 15])

    # 创建一个4x4的稀疏矩阵，并断言其倒数第2行到最后一行与一个2x4的稀疏矩阵相等
    assert m2[-2:, :] == SparseMatrix([[8, 9, 10, 11], [12, 13, 14, 15]])

    # 断言：对于给定的稀疏矩阵，获取其中第1行第1列的元素，应与一个普通矩阵相等
    assert SparseMatrix([[1, 2], [3, 4]])[[1], [1]] == Matrix([[4]])

    # test_submatrix_assignment 测试子矩阵赋值功能

    # 创建一个4x4的零稀疏矩阵，并将其第2到3行、第2到3列赋值为2x2的单位稀疏矩阵，断言两者相等
    m = sparse_zeros(4)
    m[2:4, 2:4] = sparse_eye(2)
    assert m == SparseMatrix([(0, 0, 0, 0),
                              (0, 0, 0, 0),
                              (0, 0, 1, 0),
                              (0, 0, 0, 1)])

    # 断言：将一个4x4的零稀疏矩阵的第1到2行、第1到2列赋值为2x2的单位稀疏矩阵，断言两者相等
    assert len(m.todok()) == 2

    # 创建一个4x4的零稀疏矩阵，并将其第1到2行、第1到2列赋值为2x2的单位稀疏矩阵，断言两者相等
    m[:2, :2] = sparse_eye(2)
    assert m == sparse_eye(4)

    # 创建一个4x4的单位稀疏矩阵，并将其每一列的第1个元素赋值为4x1的稀疏矩阵，断言两者相等
    m[:, 0] = SparseMatrix(4, 1, (1, 2, 3,
    # 初始化一个稀疏矩阵 m，赋值为一个二维元组列表 ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16))
    m[:, :] = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16))
    # 使用断言验证 m 是否等于预期的稀疏矩阵 SparseMatrix((( 1,  2,  3,  4), ( 5,  6,  7,  8), ( 9, 10, 11, 12), (13, 14, 15, 16)))
    assert m == SparseMatrix((( 1,  2,  3,  4),
                              ( 5,  6,  7,  8),
                              ( 9, 10, 11, 12),
                              (13, 14, 15, 16)))
    # 修改稀疏矩阵 m 的前两行第一列，将其值设为 [0, 0]
    m[:2, 0] = [0, 0]
    # 使用断言验证 m 是否等于修改后的稀疏矩阵 SparseMatrix((( 0,  2,  3,  4), ( 0,  6,  7,  8), ( 9, 10, 11, 12), (13, 14, 15, 16)))
    assert m == SparseMatrix((( 0,  2,  3,  4),
                              ( 0,  6,  7,  8),
                              ( 9, 10, 11, 12),
                              (13, 14, 15, 16)))

    # test_reshape
    # 创建一个单位稀疏矩阵 m0，大小为 3x3
    m0 = sparse_eye(3)
    # 使用断言验证 m0 重塑为 1x9 的稀疏矩阵后是否与预期的稀疏矩阵 SparseMatrix(1, 9, (1, 0, 0, 0, 1, 0, 0, 0, 1)) 相等
    assert m0.reshape(1, 9) == SparseMatrix(1, 9, (1, 0, 0, 0, 1, 0, 0, 0, 1))
    # 创建一个稀疏矩阵 m1，大小为 3x4，元素为函数 i+j
    m1 = SparseMatrix(3, 4, lambda i, j: i + j)
    # 使用断言验证 m1 重塑为 4x3 的稀疏矩阵后是否与预期的稀疏矩阵 SparseMatrix([(0, 1, 2), (3, 1, 2), (3, 4, 2), (3, 4, 5)]) 相等
    assert m1.reshape(4, 3) == \
        SparseMatrix([(0, 1, 2), (3, 1, 2), (3, 4, 2), (3, 4, 5)])
    # 使用断言验证 m1 重塑为 2x6 的稀疏矩阵后是否与预期的稀疏矩阵 SparseMatrix([(0, 1, 2, 3, 1, 2), (3, 4, 2, 3, 4, 5)]) 相等
    assert m1.reshape(2, 6) == \
        SparseMatrix([(0, 1, 2, 3, 1, 2), (3, 4, 2, 3, 4, 5)])

    # test_applyfunc
    # 创建一个单位稀疏矩阵 m0，大小为 3x3
    m0 = sparse_eye(3)
    # 使用 applyfunc 函数应用 lambda 函数 2*x 到 m0，使用断言验证结果是否与 sparse_eye(3)*2 相等
    assert m0.applyfunc(lambda x: 2*x) == sparse_eye(3)*2
    # 使用 applyfunc 函数应用 lambda 函数 0 到 m0，使用断言验证结果是否与 sparse_zeros(3) 相等
    assert m0.applyfunc(lambda x: 0 ) == sparse_zeros(3)

    # test__eval_Abs
    # 使用断言验证绝对值函数 abs 应用到 SparseMatrix(((x, 1), (y, 2*y))) 后是否等于预期的稀疏矩阵 SparseMatrix(((Abs(x), 1), (Abs(y), 2*Abs(y))))
    assert abs(SparseMatrix(((x, 1), (y, 2*y)))) == SparseMatrix(((Abs(x), 1), (Abs(y), 2*Abs(y))))

    # test_LUdecomp
    # 创建一个测试稀疏矩阵 testmat
    testmat = SparseMatrix([[ 0, 2, 5, 3],
                            [ 3, 3, 7, 4],
                            [ 8, 4, 0, 2],
                            [-2, 6, 3, 4]])
    # 对 testmat 进行 LU 分解，返回下三角矩阵 L，上三角矩阵 U，行置换 p
    L, U, p = testmat.LUdecomposition()
    # 使用断言验证 L 是否为下三角矩阵
    assert L.is_lower
    # 使用断言验证 U 是否为上三角矩阵
    assert U.is_upper
    # 使用断言验证 L*U 经过行置换 p（向后）后减去 testmat 是否等于稀疏矩阵 sparse_zeros(4)
    assert (L*U).permute_rows(p, 'backward') - testmat == sparse_zeros(4)

    # 创建另一个测试稀疏矩阵 testmat
    testmat = SparseMatrix([[ 6, -2, 7, 4],
                            [ 0,  3, 6, 7],
                            [ 1, -2, 7, 4],
                            [-9,  2, 6, 3]])
    # 对 testmat 进行 LU 分解，返回下三角矩阵 L，上三角矩阵 U，行置换 p
    L, U, p = testmat.LUdecomposition()
    # 使用断言验证 L 是否为下三角矩阵
    assert L.is_lower
    # 使用断言验证 U 是否为上三角矩阵
    assert U.is_upper
    # 使用断言验证 L*U 经过行置换 p（向后）后减去 testmat 是否等于稀疏矩阵 sparse_zeros(4)
    assert (L*U).permute_rows(p, 'backward') - testmat == sparse_zeros(4)

    # 创建符号变量 x, y, z
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    # 创建一个一般矩阵 M，元素为 ((1, x, 1), (2, y, 0), (y, 0, z))
    M = Matrix(((1, x, 1), (2, y, 0), (y, 0, z)))
    # 对 M 进行 LU 分解，返回下三角矩阵 L，上三角矩阵 U，行置换 p
    L, U, p = M.LUdecomposition()
    # 使用断言验证 L 是否为下三角矩阵
    assert L.is_lower
    # 使用断言验证 U 是否为上三角矩阵
    assert U.is_upper
    # 使用断言验证 L*U 经过行置换 p（向后）后减去 M 是否等于稀疏矩阵 sparse_zeros(3)
    assert (L*U).permute_rows(p, 'backward') - M == sparse_zeros(3)

    # test_LUsolve
    # 创建一个测试稀疏矩阵 A
    A = SparseMatrix([[2, 3, 5],
    # 创建一个稀疏矩阵 A，包含整数值
    A = SparseMatrix([[2, 3, 5],
                      [3, 6, 2],
                      [5, 2, 6]])
    # 计算 A 的逆矩阵，然后转换为稀疏矩阵 Ainv
    Ainv = SparseMatrix(Matrix(A).inv())
    # 断言 A 乘以 Ainv 等于稀疏单位矩阵
    assert A*Ainv == sparse_eye(3)
    # 断言 A 的逆矩阵（使用 CH 分解方法）等于 Ainv
    assert A.inv(method="CH") == Ainv
    # 断言 A 的逆矩阵（使用 LDL 分解方法）等于 Ainv
    assert A.inv(method="LDL") == Ainv

    # test_cross
    # 创建两个行向量 v1 和 v2
    v1 = Matrix(1, 3, [1, 2, 3])
    v2 = Matrix(1, 3, [3, 4, 5])
    # 断言 v1 叉乘 v2 的结果
    assert v1.cross(v2) == Matrix(1, 3, [-2, 4, -2])
    # 断言 v1 的二范数的平方
    assert v1.norm(2)**2 == 14

    # conjugate
    # 创建一个稀疏矩阵 a，包含复数值
    a = SparseMatrix(((1, 2 + I), (3, 4)))
    # 断言 a 的共轭转置矩阵 C
    assert a.C == SparseMatrix([
        [1, 2 - I],
        [3,     4]
    ])

    # mul
    # 断言 a 乘以一个 2x2 的单位矩阵结果等于 a
    assert a*Matrix(2, 2, [1, 0, 0, 1]) == a
    # 断言 a 加上一个 2x2 矩阵的结果
    assert a + Matrix(2, 2, [1, 1, 1, 1]) == SparseMatrix([
        [2, 3 + I],
        [4,     5]
    ])

    # col join
    # 断言 a 和一个 2x2 稀疏单位矩阵的列连接结果
    assert a.col_join(sparse_eye(2)) == SparseMatrix([
        [1, 2 + I],
        [3,     4],
        [1,     0],
        [0,     1]
    ])

    # row insert
    # 断言 a 插入一个 2x2 稀疏单位矩阵的行后的结果
    assert a.row_insert(2, sparse_eye(2)) == SparseMatrix([
        [1, 2 + I],
        [3,     4],
        [1,     0],
        [0,     1]
    ])

    # col insert
    # 断言 a 插入一个 2x1 的稀疏零列后的结果
    assert a.col_insert(2, SparseMatrix.zeros(2, 1)) == SparseMatrix([
        [1, 2 + I, 0],
        [3,     4, 0],
    ])

    # symmetric
    # 断言 a 不是对称矩阵（未简化）
    assert not a.is_symmetric(simplify=False)

    # col op
    # 创建一个 3x3 的单位矩阵 M，乘以 2
    M = SparseMatrix.eye(3)*2
    # 对 M 的第一列进行行操作
    M[1, 0] = -1
    M.col_op(1, lambda v, i: v + 2*M[i, 0])
    # 断言行操作后的结果
    assert M == SparseMatrix([
        [ 2, 4, 0],
        [-1, 0, 0],
        [ 0, 0, 2]
    ])

    # fill
    # 创建一个 3x3 的单位矩阵 M，填充为 2
    M = SparseMatrix.eye(3)
    M.fill(2)
    # 断言填充后的结果
    assert M == SparseMatrix([
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ])

    # test_cofactor
    # 断言 3x3 稀疏单位矩阵的余子式矩阵
    assert sparse_eye(3) == sparse_eye(3).cofactor_matrix()
    # 创建一个 3x3 的测试矩阵，计算其余子式矩阵
    test = SparseMatrix([[1, 3, 2], [2, 6, 3], [2, 3, 6]])
    # 断言测试矩阵的余子式矩阵
    assert test.cofactor_matrix() == \
        SparseMatrix([[27, -6, -6], [-12, 2, 3], [-3, 1, 0]])
    test = SparseMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 断言测试矩阵的余子式矩阵
    assert test.cofactor_matrix() == \
        SparseMatrix([[-3, 6, -3], [6, -12, 6], [-3, 6, -3]])

    # test_jacobian
    # 创建一个 1x2 的稀疏矩阵 L，包含符号 x 和 y 的表达式
    x = Symbol('x')
    y = Symbol('y')
    L = SparseMatrix(1, 2, [x**2*y, 2*y**2 + x*y])
    syms = [x, y]
    # 断言 L 对于符号 x 和 y 的雅可比矩阵
    assert L.jacobian(syms) == Matrix([[2*x*y, x**2], [y, 4*y + x]])

    L = SparseMatrix(1, 2, [x, x**2*y**3])
    # 断言 L 对于符号 x 和 y 的雅可比矩阵
    assert L.jacobian(syms) == SparseMatrix([[1, 0], [2*x*y**3, x**2*3*y**2]])

    # test_QR
    # 创建一个 2x2 的矩阵 A
    A = Matrix([[1, 2], [2, 3]])
    # 进行 QR 分解
    Q, S = A.QRdecomposition()
    R = Rational
    # 断言 Q 矩阵的结果
    assert Q == Matrix([
        [  5**R(-1, 2),  (R(2)/5)*(R(1)/5)**R(-1, 2)],
        [2*5**R(-1, 2), (-R(1)/5)*(R(1)/5)**R(-1, 2)]])
    # 断言 S 矩阵的结果
    assert S == Matrix([
        [5**R(1, 2),     8*5**R(-1, 2)],
        [         0, (R(1)/5)**R(1, 2)]])
    # 断言 Q*S 等于矩阵 A
    assert Q*S == A
    # 断言 Q 的转置乘以 Q 等于稀疏单位矩阵
    assert Q.T * Q == sparse_eye(2)

    R = Rational
    # test nullspace
    # 创建一个 2x4 的稀疏矩阵 M
    M = SparseMatrix([[5, 7, 2, 1],
               [1, 6, 2, -1]])
    # 计算 M 的简化行阶梯形式和一些临时结果
    out, tmp = M.rref()
    # 断言简化行阶梯形式的结果
    assert out == Matrix([[1, 0, -R(2)/23, R(13)/23],
                          [0, 1,  R(8)/23, R(-6)/23]])
    # 创建稀疏矩阵 M，使用 SparseMatrix 类，传入一个二维数组作为初始化数据
    M = SparseMatrix([[ 1,  3, 0,  2,  6, 3, 1],
                      [-2, -6, 0, -2, -8, 3, 1],
                      [ 3,  9, 0,  0,  6, 6, 2],
                      [-1, -3, 0,  1,  0, 9, 3]])

    # 对稀疏矩阵 M 进行行简化阶梯形操作，返回简化阶梯形矩阵 out 和临时对象 tmp
    out, tmp = M.rref()

    # 使用断言验证 out 是否等于指定的矩阵
    assert out == Matrix([[1, 3, 0, 0, 2, 0, 0],
                          [0, 0, 0, 1, 2, 0, 0],
                          [0, 0, 0, 0, 0, 1, R(1)/3],
                          [0, 0, 0, 0, 0, 0, 0]])

    # 获取 M 的零空间的基础向量，存储在 basis 中
    basis = M.nullspace()

    # 使用断言验证 basis 中的各个向量是否等于指定的矩阵
    assert basis[0] == Matrix([-3, 1, 0, 0, 0, 0, 0])
    assert basis[1] == Matrix([0, 0, 1, 0, 0, 0, 0])
    assert basis[2] == Matrix([-2, 0, 0, -2, 1, 0, 0])
    assert basis[3] == Matrix([0, 0, 0, 0, 0, R(-1)/3, 1])

    # 创建符号变量 x 和 y
    x = Symbol('x')
    y = Symbol('y')

    # 创建一个大小为 3 的单位稀疏矩阵 sparse_eye3
    sparse_eye3 = sparse_eye(3)

    # 使用断言验证 sparse_eye3 对象的特征多项式是否等于预期的 PurePoly 对象
    assert sparse_eye3.charpoly(x) == PurePoly((x - 1)**3)
    assert sparse_eye3.charpoly(y) == PurePoly((y - 1)**3)

    # 创建一个特定的矩阵 M
    M = Matrix([( 0, 1, -1),
                ( 1, 1,  0),
                (-1, 0,  1)])

    # 计算 M 的特征值，并存储在 vals 中
    vals = M.eigenvals()

    # 使用断言验证 vals 的键按升序排列是否等于预期的列表
    assert sorted(vals.keys()) == [-1, 1, 2]

    # 定义 Rational 类的别名 R
    R = Rational

    # 创建一个单位矩阵 M
    M = Matrix([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

    # 使用断言验证 M 的特征向量是否等于预期的列表形式
    assert M.eigenvects() == [(1, 3, [
        Matrix([1, 0, 0]),
        Matrix([0, 1, 0]),
        Matrix([0, 0, 1])])]

    # 创建一个特定的矩阵 M
    M = Matrix([[5, 0, 2],
                [3, 2, 0],
                [0, 0, 1]])

    # 使用断言验证 M 的特征向量是否等于预期的列表形式
    assert M.eigenvects() == [(1, 1, [Matrix([R(-1)/2, R(3)/2, 1])]),
                              (2, 1, [Matrix([0, 1, 0])]),
                              (5, 1, [Matrix([1, 1, 0])])]

    # 使用断言验证 SparseMatrix.zeros 方法生成的稀疏矩阵的非零元素数是否为预期值
    assert M.zeros(3, 5) == SparseMatrix(3, 5, {})

    # 创建一个特定的稀疏矩阵 A
    A =  SparseMatrix(10, 10, {(0, 0): 18, (0, 9): 12, (1, 4): 18, (2, 7): 16, (3, 9): 12, (4, 2): 19, (5, 7): 16, (6, 2): 12, (9, 7): 18})

    # 使用断言验证 A 的行列表形式是否等于预期的列表
    assert A.row_list() == [(0, 0, 18), (0, 9, 12), (1, 4, 18), (2, 7, 16), (3, 9, 12), (4, 2, 19), (5, 7, 16), (6, 2, 12), (9, 7, 18)]

    # 使用断言验证 A 的列列表形式是否等于预期的列表
    assert A.col_list() == [(0, 0, 18), (4, 2, 19), (6, 2, 12), (1, 4, 18), (2, 7, 16), (5, 7, 16), (9, 7, 18), (0, 9, 12), (3, 9, 12)]

    # 使用断言验证 SparseMatrix.eye(2) 方法生成的单位稀疏矩阵的非零元素数是否为预期值
    assert SparseMatrix.eye(2).nnz() == 2
# 测试标量乘法函数 `scalar_multiply()` 是否按预期工作
def test_scalar_multiply():
    assert SparseMatrix([[1, 2]]).scalar_multiply(3) == SparseMatrix([[3, 6]])

# 测试转置函数 `transpose()` 是否按预期工作
def test_transpose():
    assert SparseMatrix(((1, 2), (3, 4))).transpose() == \
        SparseMatrix(((1, 3), (2, 4)))

# 测试迹函数 `trace()` 是否按预期工作
def test_trace():
    assert SparseMatrix(((1, 2), (3, 4))).trace() == 5
    assert SparseMatrix(((0, 0), (0, 4))).trace() == 4

# 测试行列表函数 `row_list()` 是否按预期工作
def test_CL_RL():
    assert SparseMatrix(((1, 2), (3, 4))).row_list() == \
        [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4)]
    assert SparseMatrix(((1, 2), (3, 4))).col_list() == \
        [(0, 0, 1), (1, 0, 3), (0, 1, 2), (1, 1, 4)]

# 测试矩阵加法函数 `__add__()` 是否按预期工作
def test_add():
    assert SparseMatrix(((1, 0), (0, 1))) + SparseMatrix(((0, 1), (1, 0))) == \
        SparseMatrix(((1, 1), (1, 1)))
    a = SparseMatrix(100, 100, lambda i, j: int(j != 0 and i % j == 0))
    b = SparseMatrix(100, 100, lambda i, j: int(i != 0 and j % i == 0))
    assert (len(a.todok()) + len(b.todok()) - len((a + b).todok()) > 0)

# 测试错误处理情况
def test_errors():
    raises(ValueError, lambda: SparseMatrix(1.4, 2, lambda i, j: 0))
    raises(TypeError, lambda: SparseMatrix([1, 2, 3], [1, 2]))
    raises(ValueError, lambda: SparseMatrix([[1, 2], [3, 4]])[(1, 2, 3)])
    raises(IndexError, lambda: SparseMatrix([[1, 2], [3, 4]])[5])
    raises(ValueError, lambda: SparseMatrix([[1, 2], [3, 4]])[1, 2, 3])
    raises(TypeError,
        lambda: SparseMatrix([[1, 2], [3, 4]]).copyin_list([0, 1], set()))
    raises(
        IndexError, lambda: SparseMatrix([[1, 2], [3, 4]])[1, 2])
    raises(TypeError, lambda: SparseMatrix([1, 2, 3]).cross(1))
    raises(IndexError, lambda: SparseMatrix(1, 2, [1, 2])[3])
    raises(ShapeError,
        lambda: SparseMatrix(1, 2, [1, 2]) + SparseMatrix(2, 1, [2, 1]))

# 测试长度函数 `__len__()` 是否按预期工作
def test_len():
    assert not SparseMatrix()
    assert SparseMatrix() == SparseMatrix([])
    assert SparseMatrix() == SparseMatrix([[]])

# 测试稀疏矩阵的 `eye()` 和 `zeros()` 函数是否按预期工作
def test_sparse_zeros_sparse_eye():
    assert SparseMatrix.eye(3) == eye(3, cls=SparseMatrix)
    assert len(SparseMatrix.eye(3).todok()) == 3
    assert SparseMatrix.zeros(3) == zeros(3, cls=SparseMatrix)
    assert len(SparseMatrix.zeros(3).todok()) == 0

# 测试复制函数 `__setitem__()` 是否按预期工作
def test_copyin():
    s = SparseMatrix(3, 3, {})
    s[1, 0] = 1
    assert s[:, 0] == SparseMatrix(Matrix([0, 1, 0]))
    assert s[3] == 1
    assert s[3: 4] == [1]
    s[1, 1] = 42
    assert s[1, 1] == 42
    assert s[1, 1:] == SparseMatrix([[42, 0]])
    s[1, 1:] = Matrix([[5, 6]])
    assert s[1, :] == SparseMatrix([[1, 5, 6]])
    s[1, 1:] = [[42, 43]]
    assert s[1, :] == SparseMatrix([[1, 42, 43]])
    s[0, 0] = 17
    assert s[:, :1] == SparseMatrix([17, 1, 0])
    s[0, 0] = [1, 1, 1]
    assert s[:, 0] == SparseMatrix([1, 1, 1])
    s[0, 0] = Matrix([1, 1, 1])
    assert s[:, 0] == SparseMatrix([1, 1, 1])
    s[0, 0] = SparseMatrix([1, 1, 1])
    assert s[:, 0] == SparseMatrix([1, 1, 1])

# 测试稀疏矩阵解函数 `solve()` 是否按预期工作
def test_sparse_solve():
    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    # 断言：验证矩阵 A 的 Cholesky 分解结果是否与给定的矩阵相等
    assert A.cholesky() == Matrix([
        [ 5, 0, 0],
        [ 3, 3, 0],
        [-1, 1, 3]])
    
    # 断言：验证矩阵 A 的 Cholesky 分解后，再与其转置相乘的结果是否与给定的矩阵相等
    assert A.cholesky() * A.cholesky().T == Matrix([
        [25, 15, -5],
        [15, 18, 0],
        [-5, 0, 11]])
    
    # 创建稀疏矩阵 A，并进行 LDL 分解
    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L, D = A.LDLdecomposition()
    
    # 断言：验证 LDL 分解后的 L 矩阵是否满足给定的条件
    assert 15*L == Matrix([
        [15, 0, 0],
        [ 9, 15, 0],
        [-3, 5, 15]])
    
    # 断言：验证 LDL 分解后的 D 对角阵是否满足给定的条件
    assert D == Matrix([
        [25, 0, 0],
        [ 0, 9, 0],
        [ 0, 0, 9]])
    
    # 断言：验证 LDL 分解后重新组合得到的矩阵是否与原始矩阵 A 相等
    assert L * D * L.T == A
    
    # 创建稀疏矩阵 A，并验证其逆矩阵乘以自身是否等于单位矩阵
    A = SparseMatrix(((3, 0, 2), (0, 0, 1), (1, 2, 0)))
    assert A.inv() * A == SparseMatrix(eye(3))
    
    # 创建稀疏矩阵 A 和预期结果 ans，并分别验证使用 'CH' 和 'LDL' 方法计算的逆矩阵是否等于预期结果
    A = SparseMatrix([
        [ 2, -1, 0],
        [-1, 2, -1],
        [ 0, 0, 2]])
    ans = SparseMatrix([
        [Rational(2, 3), Rational(1, 3), Rational(1, 6)],
        [Rational(1, 3), Rational(2, 3), Rational(1, 3)],
        [             0,              0,        S.Half]])
    assert A.inv(method='CH') == ans
    assert A.inv(method='LDL') == ans
    
    # 断言：验证矩阵 A 与其逆矩阵相乘是否等于单位矩阵
    assert A * ans == SparseMatrix(eye(3))
    
    # 解线性方程 A*s = A[:, 0]，使用 'LDL' 方法求解并验证结果
    s = A.solve(A[:, 0], 'LDL')
    assert A*s == A[:, 0]
    
    # 解线性方程 A*s = A[:, 0]，使用 'CH' 方法求解并验证结果
    s = A.solve(A[:, 0], 'CH')
    assert A*s == A[:, 0]
    
    # 将矩阵 A 与其自身列连接，并使用 'CH' 方法求最小二乘解，验证结果
    A = A.col_join(A)
    s = A.solve_least_squares(A[:, 0], 'CH')
    assert A*s == A[:, 0]
    
    # 将矩阵 A 与其自身列连接，并使用 'LDL' 方法求最小二乘解，验证结果
    s = A.solve_least_squares(A[:, 0], 'LDL')
    assert A*s == A[:, 0]
def test_lower_triangular_solve():
    # 测试非方阵输入时是否会引发异常
    raises(NonSquareMatrixError, lambda:
        SparseMatrix([[1, 2]]).lower_triangular_solve(Matrix([[1, 2]])))
    # 测试形状不匹配的情况是否会引发异常
    raises(ShapeError, lambda:
        SparseMatrix([[1, 2], [0, 4]]).lower_triangular_solve(Matrix([1])))
    # 测试矩阵维度不匹配的情况是否会引发异常
    raises(ValueError, lambda:
        SparseMatrix([[1, 2], [3, 4]]).lower_triangular_solve(Matrix([[1, 2], [3, 4]])))

    # 定义符号变量
    a, b, c, d = symbols('a:d')
    u, v, w, x = symbols('u:x')

    # 创建稀疏矩阵 A、可变稀疏矩阵 B、不可变稀疏矩阵 C
    A = SparseMatrix([[a, 0], [c, d]])
    B = MutableSparseMatrix([[u, v], [w, x]])
    C = ImmutableSparseMatrix([[u, v], [w, x]])

    # 预期的解
    sol = Matrix([[u/a, v/a], [(w - c*u/a)/d, (x - c*v/a)/d]])
    # 断言稀疏矩阵 A 对 B 和 C 的下三角解是否符合预期解
    assert A.lower_triangular_solve(B) == sol
    assert A.lower_triangular_solve(C) == sol


def test_upper_triangular_solve():
    # 测试非方阵输入时是否会引发异常
    raises(NonSquareMatrixError, lambda:
        SparseMatrix([[1, 2]]).upper_triangular_solve(Matrix([[1, 2]])))
    # 测试形状不匹配的情况是否会引发异常
    raises(ShapeError, lambda:
        SparseMatrix([[1, 2], [0, 4]]).upper_triangular_solve(Matrix([1])))
    # 测试类型错误的情况是否会引发异常
    raises(TypeError, lambda:
        SparseMatrix([[1, 2], [3, 4]]).upper_triangular_solve(Matrix([[1, 2], [3, 4]])))

    # 定义符号变量
    a, b, c, d = symbols('a:d')
    u, v, w, x = symbols('u:x')

    # 创建稀疏矩阵 A、可变稀疏矩阵 B、不可变稀疏矩阵 C
    A = SparseMatrix([[a, b], [0, d]])
    B = MutableSparseMatrix([[u, v], [w, x]])
    C = ImmutableSparseMatrix([[u, v], [w, x]])

    # 预期的解
    sol = Matrix([[(u - b*w/d)/a, (v - b*x/d)/a], [w/d, x/d]])
    # 断言稀疏矩阵 A 对 B 和 C 的上三角解是否符合预期解
    assert A.upper_triangular_solve(B) == sol
    assert A.upper_triangular_solve(C) == sol


def test_diagonal_solve():
    # 定义符号变量
    a, d = symbols('a d')
    u, v, w, x = symbols('u:x')

    # 创建对角稀疏矩阵 A、可变稀疏矩阵 B、不可变稀疏矩阵 C
    A = SparseMatrix([[a, 0], [0, d]])
    B = MutableSparseMatrix([[u, v], [w, x]])
    C = ImmutableSparseMatrix([[u, v], [w, x]])

    # 预期的解
    sol = Matrix([[u/a, v/a], [w/d, x/d]])
    # 断言对角稀疏矩阵 A 对 B 和 C 的对角线解是否符合预期解
    assert A.diagonal_solve(B) == sol
    assert A.diagonal_solve(C) == sol


def test_hermitian():
    # 定义符号变量
    x = Symbol('x')
    # 创建复数域的稀疏矩阵 a
    a = SparseMatrix([[0, I], [-I, 0]])
    # 断言矩阵 a 是否是厄米矩阵
    assert a.is_hermitian
    # 修改矩阵 a 的元素
    a = SparseMatrix([[1, I], [-I, 1]])
    # 断言修改后的矩阵 a 是否是厄米矩阵
    assert a.is_hermitian
    # 修改矩阵 a 的元素
    a[0, 0] = 2*I
    # 断言修改后的矩阵 a 是否不是厄米矩阵
    assert a.is_hermitian is False
    # 修改矩阵 a 的元素
    a[0, 0] = x
    # 断言修改后的矩阵 a 是否无法确定是否是厄米矩阵
    assert a.is_hermitian is None
    # 修改矩阵 a 的元素
    a[0, 1] = a[1, 0]*I
    # 断言修改后的矩阵 a 是否不是厄米矩阵
    assert a.is_hermitian is False
```