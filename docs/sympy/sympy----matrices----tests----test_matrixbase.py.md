# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_matrixbase.py`

```
# 导入并发执行库中的 futures 模块，用于支持并行任务执行
import concurrent.futures
# 导入随机数生成模块
import random
# 导入支持哈希操作的抽象基类
from collections.abc import Hashable

# 导入 sympy 符号计算库中的多个模块和类
from sympy import (
    Abs, Add, Array, DeferredVector, E, Expr, FiniteSet, Float, Function,
    GramSchmidt, I, ImmutableDenseMatrix, ImmutableMatrix,
    ImmutableSparseMatrix, Integer, KroneckerDelta, MatPow, Matrix,
    MatrixSymbol, Max, Min, MutableDenseMatrix, MutableSparseMatrix, Poly, Pow,
    PurePoly, Q, Quaternion, Rational, RootOf, S, SparseMatrix, Symbol, Tuple,
    Wild, banded, casoratian, cos, diag, diff, exp, expand, eye, hessian,
    integrate, log, matrix_multiply_elementwise, nan, ones, oo, pi, randMatrix,
    rot_axis1, rot_axis2, rot_axis3, rot_ccw_axis1, rot_ccw_axis2,
    rot_ccw_axis3, signsimp, simplify, sin, sqrt, sstr, symbols, sympify, tan,
    trigsimp, wronskian, zeros)
# 导入 sympy 符号计算库中的符号变量
from sympy.abc import a, b, c, d, t, x, y, z
# 导入 sympy 符号计算库中的数值类型和未定义类型
from sympy.core.kind import NumberKind, UndefinedKind
# 导入 sympy 矩阵计算库中的函数和异常处理类
from sympy.matrices.determinant import _find_reasonable_pivot_naive
from sympy.matrices.exceptions import (
    MatrixError, NonSquareMatrixError, ShapeError)
# 导入 sympy 矩阵计算库中的矩阵类型
from sympy.matrices.kind import MatrixKind
# 导入 sympy 矩阵计算库中的实用工具函数
from sympy.matrices.utilities import _dotprodsimp_state, _simplify, dotprodsimp
# 导入 sympy 张量计算库中的数组导数计算类
from sympy.tensor.array.array_derivatives import ArrayDerivative
# 导入 sympy 测试框架中的测试辅助函数和装饰器
from sympy.testing.pytest import (
    ignore_warnings, raises, skip, skip_under_pyodide, slow,
    warns_deprecated_sympy)
# 导入 sympy 工具函数中的捕获函数
from sympy.utilities.iterables import capture, iterable
# 导入 importlib.metadata 中的版本查询函数
from importlib.metadata import version

# 定义所有的矩阵类
all_classes = (Matrix, SparseMatrix, ImmutableMatrix, ImmutableSparseMatrix)
# 定义可变矩阵类
mutable_classes = (Matrix, SparseMatrix)
# 定义不可变矩阵类
immutable_classes = (ImmutableMatrix, ImmutableSparseMatrix)


# 定义测试函数 test__MinimalMatrix，用于测试最小矩阵的创建和操作
def test__MinimalMatrix():
    # 创建一个 2x3 的矩阵 x，包含指定的数据
    x = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    # 断言矩阵 x 的行数为 2
    assert x.rows == 2
    # 断言矩阵 x 的列数为 3
    assert x.cols == 3
    # 断言矩阵 x 在索引位置 2 处的值为 3
    assert x[2] == 3
    # 断言矩阵 x 在索引位置 (1, 1) 处的值为 5
    assert x[1, 1] == 5
    # 断言将矩阵 x 转换为列表后的结果与预期一致
    assert list(x) == [1, 2, 3, 4, 5, 6]
    # 断言取矩阵 x 第 1 行的所有元素组成的列表与预期一致
    assert list(x[1, :]) == [4, 5, 6]
    # 断言取矩阵 x 第 1 列的所有元素组成的列表与预期一致
    assert list(x[:, 1]) == [2, 5]
    # 断言取矩阵 x 的所有元素组成的列表与预期一致
    assert list(x[:, :]) == list(x)
    # 断言矩阵 x 与其自身比较相等
    assert x[:, :] == x
    # 断言复制矩阵 x 后的结果与 x 相等
    assert Matrix(x) == x
    # 断言使用不同的方式创建矩阵并与 x 比较相等
    assert Matrix([[1, 2, 3], [4, 5, 6]]) == x
    assert Matrix(([1, 2, 3], [4, 5, 6])) == x
    assert Matrix([(1, 2, 3), (4, 5, 6)]) == x
    assert Matrix(((1, 2, 3), (4, 5, 6))) == x
    # 断言使用不同数据创建的矩阵与 x 不相等
    assert not (Matrix([[1, 2], [3, 4], [5, 6]]) == x)


# 定义测试函数 test_kind，用于测试矩阵的类型判断
def test_kind():
    # 断言普通矩阵类型为数值类型
    assert Matrix([[1, 2], [3, 4]]).kind == MatrixKind(NumberKind)
    # 断言零矩阵类型为数值类型
    assert Matrix([[0, 0], [0, 0]]).kind == MatrixKind(NumberKind)
    # 断言空矩阵类型为数值类型
    assert Matrix(0, 0, []).kind == MatrixKind(NumberKind)
    # 断言包含符号变量的矩阵类型为数值类型
    assert Matrix([[x]]).kind == MatrixKind(NumberKind)
    # 断言包含混合类型的矩阵类型为未定义类型
    assert Matrix([[1, Matrix([[1]])]]).kind == MatrixKind(UndefinedKind)
    # 断言稀疏矩阵类型为数值类型
    assert SparseMatrix([[1]]).kind == MatrixKind(NumberKind)
    # 断言包含混合类型的稀疏矩阵类型为未定义类型
    assert SparseMatrix([[1, Matrix([[1]])]]).kind == MatrixKind(UndefinedKind)


# 定义测试函数 test_todok，用于测试创建不同类型矩阵的可变性和不可变性
def test_todok():
    # 定义符号变量 a, b, c, d
    a, b, c, d = symbols('a:d')
    # 创建可变密集矩阵 m1
    m1 = MutableDenseMatrix([[a, b], [c, d]])
    # 创建不可变密集矩阵 m2
    m2 = ImmutableDenseMatrix([[a, b], [c, d]])
    # 创建可变稀疏矩阵 m3
    m3 = MutableSparseMatrix([[a, b], [c, d]])
    # 创建不可变稀疏矩阵 m4
    m4 = ImmutableSparseMatrix([[a, b], [c, d]])
    # 断言语句：验证多个稀疏矩阵对象的稀疏表示是否相等
    assert m1.todok() == m2.todok() == m3.todok() == m4.todok() == \
        {(0, 0): a, (0, 1): b, (1, 0): c, (1, 1): d}
def test_tolist():
    # 创建一个包含四个子列表的二维列表
    lst = [[S.One, S.Half, x*y, S.Zero], [x, y, z, x**2], [y, -S.One, z*x, 3]]
    # 将二维列表展开成一个包含所有元素的一维列表
    flat_lst = [S.One, S.Half, x*y, S.Zero, x, y, z, x**2, y, -S.One, z*x, 3]
    # 使用一维列表创建一个 3x4 的矩阵对象
    m = Matrix(3, 4, flat_lst)
    # 断言矩阵对象的 tolist 方法返回的结果与 lst 相同
    assert m.tolist() == lst


def test_todod():
    # 创建一个包含特定元素的 3x2 矩阵对象
    m = Matrix([[S.One, 0], [0, S.Half], [x, 0]])
    # 创建一个预期的字典对象
    dict = {0: {0: S.One}, 1: {1: S.Half}, 2: {0: x}}
    # 断言矩阵对象的 todod 方法返回的结果与 dict 相同
    assert m.todod() == dict


def test_row_col_del():
    # 创建一个 3x3 的不可变矩阵对象
    e = ImmutableMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 对行的删除操作，验证是否会抛出 IndexError 异常
    raises(IndexError, lambda: e.row_del(5))
    raises(IndexError, lambda: e.row_del(-5))
    # 对列的删除操作，验证是否会抛出 IndexError 异常
    raises(IndexError, lambda: e.col_del(5))
    raises(IndexError, lambda: e.col_del(-5))

    # 验证行删除方法的正确性，并断言结果与预期的矩阵对象相同
    assert e.row_del(2) == e.row_del(-1) == Matrix([[1, 2, 3], [4, 5, 6]])
    # 验证列删除方法的正确性，并断言结果与预期的矩阵对象相同
    assert e.col_del(2) == e.col_del(-1) == Matrix([[1, 2], [4, 5], [7, 8]])

    # 再次验证行删除方法的正确性，并断言结果与预期的矩阵对象相同
    assert e.row_del(1) == e.row_del(-2) == Matrix([[1, 2, 3], [7, 8, 9]])
    # 再次验证列删除方法的正确性，并断言结果与预期的矩阵对象相同
    assert e.col_del(1) == e.col_del(-2) == Matrix([[1, 3], [4, 6], [7, 9]])


def test_get_diag_blocks1():
    # 创建三个不同的 2x2 矩阵对象
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    # 验证各矩阵对象的 get_diag_blocks 方法返回的结果与预期相同
    assert a.get_diag_blocks() == [a]
    assert b.get_diag_blocks() == [b]
    assert c.get_diag_blocks() == [c]


def test_get_diag_blocks2():
    # 创建三个不同的 2x2 和 3x3 的矩阵对象，并使用 diag 函数创建对角块矩阵
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    A, B, C, D = diag(a, b, b), diag(a, b, c), diag(a, c, b), diag(c, c, b)
    # 将对角块矩阵对象转换为 Matrix 类型的矩阵对象
    A = Matrix(A.rows, A.cols, A)
    B = Matrix(B.rows, B.cols, B)
    C = Matrix(C.rows, C.cols, C)
    D = Matrix(D.rows, D.cols, D)

    # 验证各矩阵对象的 get_diag_blocks 方法返回的结果与预期相同
    assert A.get_diag_blocks() == [a, b, b]
    assert B.get_diag_blocks() == [a, b, c]
    assert C.get_diag_blocks() == [a, c, b]
    assert D.get_diag_blocks() == [c, c, b]


def test_row_col():
    # 创建一个 3x3 的矩阵对象
    m = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 验证矩阵对象的行提取方法的正确性，并断言结果与预期的矩阵对象相同
    assert m.row(0) == Matrix(1, 3, [1, 2, 3])
    # 验证矩阵对象的列提取方法的正确性，并断言结果与预期的矩阵对象相同
    assert m.col(0) == Matrix(3, 1, [1, 4, 7])


def test_row_join():
    # 验证单位矩阵与一个行向量的行连接操作的正确性，并断言结果与预期的矩阵对象相同
    assert eye(3).row_join(Matrix([7, 7, 7])) == \
           Matrix([[1, 0, 0, 7],
                   [0, 1, 0, 7],
                   [0, 0, 1, 7]])


def test_col_join():
    # 验证单位矩阵与一个列向量的列连接操作的正确性，并断言结果与预期的矩阵对象相同
    assert eye(3).col_join(Matrix([[7, 7, 7]])) == \
           Matrix([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [7, 7, 7]])


def test_row_insert():
    # 创建一个包含 [4, 4, 4] 的行向量对象
    r4 = Matrix([[4, 4, 4]])
    # 对于给定的范围，验证单位矩阵在指定位置插入行向量后的结果与预期相同
    for i in range(-4, 5):
        l = [1, 0, 0]
        l.insert(i, 4)
        assert eye(3).row_insert(i, r4).col(0).flat() == l


def test_col_insert():
    # 创建一个包含 [4, 4, 4] 的列向量对象
    c4 = Matrix([4, 4, 4])
    # 对于给定的范围，验证全零矩阵在指定位置插入列向量后的结果与预期相同
    for i in range(-4, 5):
        l = [0, 0, 0]
        l.insert(i, 4)
        assert zeros(3).col_insert(i, c4).row(0).flat() == l
    # issue 13643
    # 断言：验证 eye(6).col_insert(3, Matrix([[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])) 的结果是否等于以下矩阵
    assert eye(6).col_insert(3, Matrix([[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])) == \
           Matrix([[1, 0, 0, 2, 2, 0, 0, 0],
                   [0, 1, 0, 2, 2, 0, 0, 0],
                   [0, 0, 1, 2, 2, 0, 0, 0],
                   [0, 0, 0, 2, 2, 1, 0, 0],
                   [0, 0, 0, 2, 2, 0, 1, 0],
                   [0, 0, 0, 2, 2, 0, 0, 1]])
def test_extract():
    # 创建一个 4x3 的矩阵 m，使用 lambda 函数填充矩阵元素
    m = Matrix(4, 3, lambda i, j: i*3 + j)
    # 测试提取指定行列的子矩阵，验证结果是否符合预期
    assert m.extract([0, 1, 3], [0, 1]) == Matrix(3, 2, [0, 1, 3, 4, 9, 10])
    assert m.extract([0, 3], [0, 0, 2]) == Matrix(2, 3, [0, 0, 2, 9, 9, 11])
    # 测试提取整个矩阵的情况，期望返回原始矩阵 m
    assert m.extract(range(4), range(3)) == m
    # 测试超出索引范围的情况是否会引发 IndexError 异常
    raises(IndexError, lambda: m.extract([4], [0]))
    raises(IndexError, lambda: m.extract([0], [3]))


def test_hstack():
    # 创建两个矩阵 m 和 m2，分别为 4x3 和 3x4 的矩阵
    m = Matrix(4, 3, lambda i, j: i*3 + j)
    m2 = Matrix(3, 4, lambda i, j: i*3 + j)
    # 测试水平堆叠函数 hstack
    assert m == m.hstack(m)
    # 测试多个矩阵水平堆叠，验证结果是否符合预期
    assert m.hstack(m, m, m) == Matrix.hstack(m, m, m) == Matrix([
                [0,  1,  2, 0,  1,  2, 0,  1,  2],
                [3,  4,  5, 3,  4,  5, 3,  4,  5],
                [6,  7,  8, 6,  7,  8, 6,  7,  8],
                [9, 10, 11, 9, 10, 11, 9, 10, 11]])
    # 测试形状不匹配时是否会引发 ShapeError 异常
    raises(ShapeError, lambda: m.hstack(m, m2))
    # 测试空矩阵的水平堆叠是否返回空矩阵
    assert Matrix.hstack() == Matrix()

    # 测试回归问题 #12938
    # 创建多个空矩阵 M1 到 M4，然后进行水平堆叠
    M1 = Matrix.zeros(0, 0)
    M2 = Matrix.zeros(0, 1)
    M3 = Matrix.zeros(0, 2)
    M4 = Matrix.zeros(0, 3)
    m = Matrix.hstack(M1, M2, M3, M4)
    # 验证水平堆叠后的矩阵 m 的行数和列数是否符合预期
    assert m.rows == 0 and m.cols == 6


def test_vstack():
    # 创建两个矩阵 m 和 m2，分别为 4x3 和 3x4 的矩阵
    m = Matrix(4, 3, lambda i, j: i*3 + j)
    m2 = Matrix(3, 4, lambda i, j: i*3 + j)
    # 测试垂直堆叠函数 vstack
    assert m == m.vstack(m)
    # 测试多个矩阵垂直堆叠，验证结果是否符合预期
    assert m.vstack(m, m, m) == Matrix.vstack(m, m, m) == Matrix([
                                [0,  1,  2],
                                [3,  4,  5],
                                [6,  7,  8],
                                [9, 10, 11],
                                [0,  1,  2],
                                [3,  4,  5],
                                [6,  7,  8],
                                [9, 10, 11],
                                [0,  1,  2],
                                [3,  4,  5],
                                [6,  7,  8],
                                [9, 10, 11]])
    # 测试形状不匹配时是否会引发 ShapeError 异常
    raises(ShapeError, lambda: m.vstack(m, m2))
    # 测试空矩阵的垂直堆叠是否返回空矩阵
    assert Matrix.vstack() == Matrix()


def test_has():
    # 创建一个符号矩阵 A
    A = Matrix(((x, y), (2, 3)))
    # 测试矩阵是否包含特定符号 x、z、Symbol
    assert A.has(x)
    assert not A.has(z)
    assert A.has(Symbol)

    # 创建另一个符号矩阵 A
    A = Matrix(((2, y), (2, 3)))
    # 测试矩阵是否包含特定符号 x
    assert not A.has(x)


def test_is_anti_symmetric():
    x = symbols('x')
    # 测试一个非对称矩阵的反对称性，预期结果为 False
    assert Matrix(2, 1, [1, 2]).is_anti_symmetric() is False
    # 创建一个具有反对称性质的矩阵 m，验证结果是否符合预期
    m = Matrix(3, 3, [0, x**2 + 2*x + 1, y, -(x + 1)**2, 0, x*y, -y, -x*y, 0])
    assert m.is_anti_symmetric() is True
    assert m.is_anti_symmetric(simplify=False) is None
    assert m.is_anti_symmetric(simplify=lambda x: x) is None

    # 对矩阵 m 进行展开，验证简化后的反对称性判定
    m = Matrix(3, 3, [x.expand() for x in m])
    assert m.is_anti_symmetric(simplify=False) is True
    m = Matrix(3, 3, [x.expand() for x in [S.One] + list(m)[1:]])
    assert m.is_anti_symmetric() is False


def test_is_hermitian():
    a = Matrix([[1, I], [-I, 1]])
    # 测试 Hermite 矩阵的判定，预期结果为 True
    assert a.is_hermitian
    a = Matrix([[2*I, I], [-I, 1]])
    # 测试非 Hermite 矩阵的判定，预期结果为 False
    assert a.is_hermitian is False
    a = Matrix([[x, I], [-I, 1]])
    # 测试无法确定 Hermite 矩阵的判定，预期结果为 None
    assert a.is_hermitian is None
    a = Matrix([[x, 1], [-I, 1]])
    # 测试非方阵的矩阵是否被正确处理，预期结果为 False
    assert a.is_hermitian is False


def test_is_symbolic():
    # 测试符号矩阵的判定
    # 创建一个二维矩阵a，其中元素为变量x，所有元素相同
    a = Matrix([[x, x], [x, x]])
    # 断言矩阵a是否完全由符号表示的元素组成，即是否所有元素都是符号表达式
    assert a.is_symbolic() is True
    
    # 创建一个二维矩阵a，其中元素为整数，长度不同的子列表
    a = Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
    # 断言矩阵a是否完全由符号表示的元素组成，预期结果为False
    assert a.is_symbolic() is False
    
    # 创建一个二维矩阵a，其中包含整数和变量x，长度不同的子列表
    a = Matrix([[1, 2, 3, 4], [5, 6, x, 8]])
    # 断言矩阵a是否完全由符号表示的元素组成，即是否所有元素都是符号表达式
    assert a.is_symbolic() is True
    
    # 创建一个二维矩阵a，其中包含整数和变量x，长度为3的子列表
    a = Matrix([[1, x, 3]])
    # 断言矩阵a是否完全由符号表示的元素组成，即是否所有元素都是符号表达式
    assert a.is_symbolic() is True
    
    # 创建一个二维矩阵a，其中只包含整数，长度为3的子列表
    a = Matrix([[1, 2, 3]])
    # 断言矩阵a是否完全由符号表示的元素组成，预期结果为False
    assert a.is_symbolic() is False
    
    # 创建一个二维矩阵a，其中包含整数和变量x，长度为1的子列表
    a = Matrix([[1], [x], [3]])
    # 断言矩阵a是否完全由符号表示的元素组成，即是否所有元素都是符号表达式
    assert a.is_symbolic() is True
    
    # 创建一个二维矩阵a，其中只包含整数，长度为1的子列表
    a = Matrix([[1], [2], [3]])
    # 断言矩阵a是否完全由符号表示的元素组成，预期结果为False
    assert a.is_symbolic() is False
# 定义测试函数，用于检查矩阵类中的is_square属性
def test_is_square():
    # 创建两个不同形状的矩阵实例
    m = Matrix([[1], [1]])
    m2 = Matrix([[2, 2], [2, 2]])
    # 断言第一个矩阵不是方阵
    assert not m.is_square
    # 断言第二个矩阵是方阵
    assert m2.is_square


# 定义测试函数，用于检查矩阵类中的is_symmetric方法
def test_is_symmetric():
    # 创建两个2x2矩阵实例
    m = Matrix(2, 2, [0, 1, 1, 0])
    # 断言第一个矩阵是对称矩阵
    assert m.is_symmetric()
    # 创建另一个2x2矩阵实例
    m = Matrix(2, 2, [0, 1, 0, 1])
    # 断言第二个矩阵不是对称矩阵
    assert not m.is_symmetric()


# 定义测试函数，用于检查矩阵类中的上/下Hessenberg性质
def test_is_hessenberg():
    # 创建一个3x3矩阵实例
    A = Matrix([[3, 4, 1], [2, 4, 5], [0, 1, 2]])
    # 断言矩阵A是上Hessenberg矩阵
    assert A.is_upper_hessenberg
    # 创建另一个3x3矩阵实例
    A = Matrix(3, 3, [3, 2, 0, 4, 4, 1, 1, 5, 2])
    # 断言矩阵A是下Hessenberg矩阵
    assert A.is_lower_hessenberg
    # 创建另一个3x3矩阵实例
    A = Matrix(3, 3, [3, 2, -1, 4, 4, 1, 1, 5, 2])
    # 断言矩阵A不是下Hessenberg矩阵
    assert A.is_lower_hessenberg is False
    # 断言矩阵A不是上Hessenberg矩阵
    assert A.is_upper_hessenberg is False

    # 创建另一个3x3矩阵实例
    A = Matrix([[3, 4, 1], [2, 4, 5], [3, 1, 2]])
    # 断言矩阵A不是上Hessenberg矩阵
    assert not A.is_upper_hessenberg


# 定义测试函数，用于检查矩阵类中的values方法
def test_values():
    # 创建一个2x2矩阵实例，其中元素为0到3
    assert set(Matrix(2, 2, [0, 1, 2, 3]).values()) == {1, 2, 3}
    x = Symbol('x', real=True)
    # 创建一个2x2矩阵实例，其中元素包含符号x和1
    assert set(Matrix(2, 2, [x, 0, 0, 1]).values()) == {x, 1}


# 定义测试函数，用于检查矩阵类中的conjugate和transpose属性
def test_conjugate():
    # 创建一个2x3矩阵实例M
    M = Matrix([[0, I, 5],
                [1, 2, 0]])

    # 断言M的转置矩阵与指定的矩阵相等
    assert M.T == Matrix([[0, 1],
                          [I, 2],
                          [5, 0]])

    # 断言M的共轭矩阵与指定的矩阵相等
    assert M.C == Matrix([[0, -I, 5],
                          [1,  2, 0]])
    # 断言M的共轭矩阵与M的conjugate方法生成的矩阵相等
    assert M.C == M.conjugate()

    # 断言M的厄米矩阵与M的转置共轭矩阵相等
    assert M.H == M.T.C
    # 断言M的厄米矩阵与指定的矩阵相等
    assert M.H == Matrix([[ 0, 1],
                          [-I, 2],
                          [ 5, 0]])


# 定义测试函数，用于检查矩阵类中的doit方法
def test_doit():
    # 创建一个包含未评估加法的矩阵实例a
    a = Matrix([[Add(x, x, evaluate=False)]])
    # 断言a的第一个元素不等于2*x
    assert a[0] != 2*x
    # 断言a经过doit方法后的结果与指定的矩阵相等
    assert a.doit() == Matrix([[2*x]])


# 定义测试函数，用于检查矩阵类中的evalf方法
def test_evalf():
    # 创建一个2x1矩阵实例a，其中元素为sqrt(5)和6
    a = Matrix(2, 1, [sqrt(5), 6])
    # 断言a的evalf方法后每个元素与其原始元素的evalf方法结果相等
    assert all(a.evalf()[i] == a[i].evalf() for i in range(2))
    # 断言a的evalf方法(精度为2)后每个元素与其原始元素的evalf方法(精度为2)结果相等
    assert all(a.evalf(2)[i] == a[i].evalf(2) for i in range(2))
    # 断言a的n方法(精度为2)后每个元素与其原始元素的n方法(精度为2)结果相等
    assert all(a.n(2)[i] == a[i].n(2) for i in range(2))


# 定义测试函数，用于检查矩阵类中的replace方法
def test_replace():
    # 创建函数符号F和G
    F, G = symbols('F, G', cls=Function)
    # 创建一个2x2矩阵实例K，其中元素根据函数G的参数i+j生成
    K = Matrix(2, 2, lambda i, j: G(i+j))
    # 创建一个2x2矩阵实例M，其中元素根据函数F的参数i+j生成
    M = Matrix(2, 2, lambda i, j: F(i+j))
    # 使用replace方法将矩阵M中的函数符号F替换为G，得到新矩阵N
    N = M.replace(F, G)
    # 断言N与预期的矩阵K相等
    assert N == K


# 定义测试函数，用于检查矩阵类中的replace方法和返回的映射
def test_replace_map():
    # 创建函数符号F和G
    F, G = symbols('F, G', cls=Function)
    # 创建一个2x2矩阵实例M，其中元素根据函数F的参数i+j生成
    M = Matrix(2, 2, lambda i, j: F(i+j))
    # 使用replace方法将矩阵M中的函数符号F替换为G，并返回新矩阵N及其生成的映射d
    N, d = M.replace(F, G, True)
    # 断言N与预期的矩阵，即每个元素根据函数G的参数i+j生成的矩阵相等
    assert N == Matrix(2, 2, lambda i, j: G(i+j))
    # 断言映射d与预期的字典相等，即映射了每个F(i)到对应的G(i)
    assert d == {F(0): G(0), F(1): G(1
    # 断言检查矩阵 A 在不同旋转情况下的结果是否相等
    assert A.rot90() == A.rot90(-7) == A.rot90(-3) == Matrix(((3, 1), (4, 2)))
# 定义一个测试函数，用于测试矩阵对象的替换操作
def test_subs():
    # 使用 subs 方法替换矩阵中的符号 x 为 5，验证结果是否符合预期
    assert Matrix([[1, x], [x, 4]]).subs(x, 5) == Matrix([[1, 5], [5, 4]])
    
    # 使用 subs 方法通过列表形式替换多个符号，验证结果是否符合预期
    assert Matrix([[x, 2], [x + y, 4]]).subs([[x, -1], [y, -2]]) == \
           Matrix([[-1, 2], [-3, 4]])
    
    # 使用 subs 方法通过元组形式替换多个符号，验证结果是否符合预期
    assert Matrix([[x, 2], [x + y, 4]]).subs([(x, -1), (y, -2)]) == \
           Matrix([[-1, 2], [-3, 4]])
    
    # 使用 subs 方法通过字典形式替换多个符号，验证结果是否符合预期
    assert Matrix([[x, 2], [x + y, 4]]).subs({x: -1, y: -2}) == \
           Matrix([[-1, 2], [-3, 4]])
    
    # 使用 subs 方法替换矩阵中的符号 x 和 y，同时替换，验证结果是否符合预期
    assert Matrix([[x*y]]).subs({x: y - 1, y: x - 1}, simultaneous=True) == \
           Matrix([[(x - 1)*(y - 1)]])


# 定义一个测试函数，用于测试矩阵对象的置换操作
def test_permute():
    # 创建一个 3x4 的矩阵对象 a
    a = Matrix(3, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # 测试对行进行置换操作，预期会触发 IndexError 异常
    raises(IndexError, lambda: a.permute([[0, 5]]))
    
    # 测试传递非法参数 Symbol('x') 给 permute 方法，预期会触发 ValueError 异常
    raises(ValueError, lambda: a.permute(Symbol('x')))
    
    # 测试对行进行置换操作，验证结果是否符合预期
    b = a.permute_rows([[0, 2], [0, 1]])
    assert a.permute([[0, 2], [0, 1]]) == b == Matrix([
                                            [5,  6,  7,  8],
                                            [9, 10, 11, 12],
                                            [1,  2,  3,  4]])

    # 测试对列进行置换操作，验证结果是否符合预期
    b = a.permute_cols([[0, 2], [0, 1]])
    assert a.permute([[0, 2], [0, 1]], orientation='cols') == b == \
           Matrix([
               [ 2,  3, 1,  4],
               [ 6,  7, 5,  8],
               [10, 11, 9, 12]])

    # 测试对列进行置换操作，指定反向置换，验证结果是否符合预期
    b = a.permute_cols([[0, 2], [0, 1]], direction='backward')
    assert a.permute([[0, 2], [0, 1]], orientation='cols', direction='backward') == b == \
           Matrix([
               [ 3, 1,  2,  4],
               [ 7, 5,  6,  8],
               [11, 9, 10, 12]])

    # 测试对向量进行置换操作，验证结果是否符合预期
    assert a.permute([1, 2, 0, 3]) == Matrix([
                                            [5,  6,  7,  8],
                                            [9, 10, 11, 12],
                                            [1,  2,  3,  4]])

    # 测试对向量使用 Permutation 对象进行置换操作，验证结果是否符合预期
    from sympy.combinatorics import Permutation
    assert a.permute(Permutation([1, 2, 0, 3])) == Matrix([
                                            [5,  6,  7,  8],
                                            [9, 10, 11, 12],
                                            [1,  2,  3,  4]])


# 定义一个测试函数，用于测试矩阵对象的上三角矩阵生成操作
def test_upper_triangular():
    # 创建一个 4x4 的全 1 矩阵 A
    A = Matrix([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ])

    # 对 A 进行上三角矩阵生成，k=2，验证结果是否符合预期
    R = A.upper_triangular(2)
    assert R == Matrix([
                        [0, 0, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]
                    ])

    # 对 A 进行上三角矩阵生成，k=-2，验证结果是否符合预期
    R = A.upper_triangular(-2)
    assert R == Matrix([
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 1, 1, 1]
                    ])

    # 对 A 进行默认上三角矩阵生成操作，验证是否正常运行
    R = A.upper_triangular()
    # 使用断言来验证矩阵 R 是否等于给定的 4x4 矩阵
    assert R == Matrix([
                        [1, 1, 1, 1],
                        [0, 1, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]
                    ])
# 定义测试函数，用于测试矩阵的下三角性质
def test_lower_triangular():
    # 创建一个4x4的矩阵A
    A = Matrix([
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]
                    ])

    # 测试默认情况下的下三角矩阵
    L = A.lower_triangular()
    assert L == Matrix([
                        [1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1]])

    # 测试从第2个对角线开始的下三角矩阵
    L = A.lower_triangular(2)
    assert L == Matrix([
                        [1, 1, 1, 0],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]
                    ])

    # 测试从第-2个对角线开始的下三角矩阵
    L = A.lower_triangular(-2)
    assert L == Matrix([
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 1, 0, 0]
                    ])


# 定义测试函数，用于测试矩阵的加法
def test_add():
    # 创建矩阵m
    m = Matrix([[1, 2, 3], [x, y, x], [2*y, -50, z*x]])
    # 测试矩阵的加法操作
    assert m + m == Matrix([[2, 4, 6], [2*x, 2*y, 2*x], [4*y, -100, 2*z*x]])
    # 创建矩阵n，测试与m的形状不匹配时引发ShapeError异常
    n = Matrix(1, 2, [1, 2])
    raises(ShapeError, lambda: m + n)


# 定义测试函数，用于测试矩阵的矩阵乘法
def test_matmul():
    # 创建矩阵a
    a = Matrix([[1, 2], [3, 4]])

    # 测试特殊情况：使用@运算符对非矩阵对象进行操作，应返回NotImplemented
    assert a.__matmul__(2) == NotImplemented

    assert a.__rmatmul__(2) == NotImplemented

    # 检查@运算符在Python 3.5+中的行为，分别测试2@a和a@2的情况
    # 预期抛出SyntaxError或TypeError异常，这取决于返回的是NotImplemented
    try:
        eval('2 @ a')
    except SyntaxError:
        pass
    except TypeError:
        pass

    try:
        eval('a @ 2')
    except SyntaxError:
        pass
    except TypeError:
        pass


# 定义测试函数，用于测试非矩阵乘法的情况
def test_non_matmul():
    """
    测试明确指定为非矩阵时，乘法应退化为标量乘法。
    """
    # 定义一个类foo，模拟非矩阵对象
    class foo(Expr):
        is_Matrix=False
        is_MatrixLike=False
        shape = (1, 1)

    A = Matrix([[1, 2], [3, 4]])
    b = foo()
    # 测试矩阵与非矩阵对象之间的乘法操作
    assert b*A == Matrix([[b, 2*b], [3*b, 4*b]])
    assert A*b == Matrix([[b, 2*b], [3*b, 4*b]])


# 定义测试函数，用于测试矩阵的负号操作
def test_neg():
    # 创建矩阵n
    n = Matrix(1, 2, [1, 2])
    # 测试矩阵的负号操作
    assert -n == Matrix(1, 2, [-1, -2])


# 定义测试函数，用于测试矩阵的减法
def test_sub():
    # 创建矩阵n
    n = Matrix(1, 2, [1, 2])
    # 测试矩阵的减法操作
    assert n - n == Matrix(1, 2, [0, 0])


# 定义测试函数，用于测试矩阵的除法
def test_div():
    # 创建矩阵n
    n = Matrix(1, 2, [1, 2])
    # 测试矩阵的除法操作
    assert n/2 == Matrix(1, 2, [S.Half, S(2)/2])


# 定义测试函数，用于测试单位矩阵的生成
def test_eye():
    # 测试生成2x2单位矩阵的不同方式
    assert list(Matrix.eye(2, 2)) == [1, 0, 0, 1]
    assert list(Matrix.eye(2)) == [1, 0, 0, 1]
    # 检查生成的类型是否为Matrix
    assert type(Matrix.eye(2)) == Matrix
    assert type(Matrix.eye(2, cls=Matrix)) == Matrix


# 定义测试函数，用于测试全1矩阵的生成
def test_ones():
    # 测试生成2x2全1矩阵的不同方式
    assert list(Matrix.ones(2, 2)) == [1, 1, 1, 1]
    assert list(Matrix.ones(2)) == [1, 1, 1, 1]
    assert Matrix.ones(2, 3) == Matrix([[1, 1, 1], [1, 1, 1]])
    # 检查生成的类型是否为Matrix
    assert type(Matrix.ones(2)) == Matrix
    assert type(Matrix.ones(2, cls=Matrix)) == Matrix


# 定义测试函数，用于测试全0矩阵的生成
def test_zeros():
    # 测试生成2x2全0矩阵的不同方式
    assert list(Matrix.zeros(2, 2)) == [0, 0, 0, 0]
    assert list(Matrix.zeros(2)) == [0, 0, 0, 0]
    # 断言：创建一个 2x3 的零矩阵，并验证其是否等于指定的矩阵
    assert Matrix.zeros(2, 3) == Matrix([[0, 0, 0], [0, 0, 0]])
    
    # 断言：创建一个 2x2 的零矩阵，并验证其类型是否为 Matrix 类型
    assert type(Matrix.zeros(2)) == Matrix
    
    # 断言：创建一个 2x2 的零矩阵，并验证其类型是否为 Matrix 类型（使用 Matrix 类来创建）
    assert type(Matrix.zeros(2, cls=Matrix)) == Matrix
def test_diag_make():
    # 将 Matrix 类的 diag 方法赋值给 diag 变量
    diag = Matrix.diag
    # 创建矩阵 a，b，c 作为测试数据
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z]])
    # 断言调用 diag 方法，使用 a、b、b 作为参数，比较返回的矩阵是否与指定的矩阵相同
    assert diag(a, b, b) == Matrix([
        [1, 2, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0],
        [0, 0, 3, x, 0, 0],
        [0, 0, y, 3, 0, 0],
        [0, 0, 0, 0, 3, x],
        [0, 0, 0, 0, y, 3],
    ])
    # 断言调用 diag 方法，使用 a、b、c 作为参数，比较返回的矩阵是否与指定的矩阵相同
    assert diag(a, b, c) == Matrix([
        [1, 2, 0, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0, 0],
        [0, 0, 3, x, 0, 0, 0],
        [0, 0, y, 3, 0, 0, 0],
        [0, 0, 0, 0, 3, x, 3],
        [0, 0, 0, 0, y, 3, z],
        [0, 0, 0, 0, x, y, z],
    ])
    # 断言调用 diag 方法，使用 a、c、b 作为参数，比较返回的矩阵是否与指定的矩阵相同
    assert diag(a, c, b) == Matrix([
        [1, 2, 0, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0, 0],
        [0, 0, 3, x, 3, 0, 0],
        [0, 0, y, 3, z, 0, 0],
        [0, 0, x, y, z, 0, 0],
        [0, 0, 0, 0, 0, 3, x],
        [0, 0, 0, 0, 0, y, 3],
    ])
    # 重新赋值矩阵 a、b、c 作为测试数据
    a = Matrix([x, y, z])
    b = Matrix([[1, 2], [3, 4]])
    c = Matrix([[5, 6]])
    # 断言调用 diag 方法，使用 a、7、b、c 作为参数，比较返回的矩阵是否与指定的矩阵相同
    assert diag(a, 7, b, c) == Matrix([
        [x, 0, 0, 0, 0, 0],
        [y, 0, 0, 0, 0, 0],
        [z, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 3, 4, 0, 0],
        [0, 0, 0, 0, 5, 6]])
    # 断言使用 lambda 表达式调用 diag 方法，使用 a、7、b、c、rows=5 作为参数，检查是否引发 ValueError 异常
    raises(ValueError, lambda: diag(a, 7, b, c, rows=5))
    # 断言调用 diag 方法，使用参数 1，比较返回的矩阵是否为指定的矩阵
    assert diag(1) == Matrix([[1]])
    # 断言调用 diag 方法，使用参数 1、rows=2，比较返回的矩阵是否为指定的矩阵
    assert diag(1, rows=2) == Matrix([[1, 0], [0, 0]])
    # 断言调用 diag 方法，使用参数 1、cols=2，比较返回的矩阵是否为指定的矩阵
    assert diag(1, cols=2) == Matrix([[1, 0], [0, 0]])
    # 断言调用 diag 方法，使用参数 1、rows=3、cols=2，比较返回的矩阵是否为指定的矩阵
    assert diag(1, rows=3, cols=2) == Matrix([[1, 0], [0, 0], [0, 0]])
    # 断言调用 diag 方法，使用参数 *[2, 3]，比较返回的矩阵是否为指定的矩阵
    assert diag(*[2, 3]) == Matrix([
        [2, 0],
        [0, 3]])
    # 断言调用 diag 方法，使用参数 Matrix([2, 3])，比较返回的矩阵是否为指定的矩阵
    assert diag(Matrix([2, 3])) == Matrix([
        [2],
        [3]])
    # 断言调用 diag 方法，使用参数 [1, [2, 3], 4]、unpack=False，比较返回的矩阵是否为指定的矩阵
    assert diag([1, [2, 3], 4], unpack=False) == \
            diag([[1], [2, 3], [4]], unpack=False) == Matrix([
        [1, 0],
        [2, 3],
        [4, 0]])
    # 断言调用 diag 方法，使用参数 1，检查返回值的类型是否为 Matrix 类型
    assert type(diag(1)) == Matrix
    # 断言调用 diag 方法，使用参数 1、cls=Matrix，检查返回值的类型是否为 Matrix 类型
    assert type(diag(1, cls=Matrix)) == Matrix
    # 断言调用 Matrix 类的 diag 静态方法，使用参数 [1, 2, 3]，比较返回的矩阵是否与 Matrix.diag(1, 2, 3) 相同
    assert Matrix.diag([1, 2, 3]) == Matrix.diag(1, 2, 3)
    # 断言调用 Matrix 类的 diag 静态方法，使用参数 [1, 2, 3]、unpack=False，检查返回的矩阵的形状是否为 (3, 1)
    assert Matrix.diag([1, 2, 3], unpack=False).shape == (3, 1)
    # 断言调用 Matrix 类的 diag 静态方法，使用参数 [[1, 2, 3]]，检查返回的矩阵的形状是否为 (3, 1)
    assert Matrix.diag([[1, 2, 3]]).shape == (3, 1)
    # 断言调用 Matrix 类的 diag 静态方法，使用参数 [[1, 2, 3]]、unpack=False，检查返回的矩阵的形状是否为 (1, 3)
    assert Matrix.diag([[1, 2, 3]], unpack=False).shape == (1, 3)
    # 断言调用 Matrix 类的 diag 静态方法，使用参数 [[[1, 2, 3]]]，检查返回的矩阵的形状是否为 (1, 3)
    assert Matrix.diag([[[1, 2, 3]]]).shape == (1, 3)
    # 断言调用 Matrix 类的 diag 静态方法，使用参数 ones(0, 2)、1、2，比较返回的矩阵是否与指定的矩阵相同
    assert Matrix.diag(ones(0, 2), 1, 2) == Matrix([
        [0, 0, 1, 0],
        [0, 0,
    # 检查是否会引发 ValueError 异常，因为指定的对角线超出矩阵的范围
    raises(ValueError, lambda: m.diagonal(3))
    # 检查是否会引发 ValueError 异常，因为指定的对角线超出矩阵的范围（负数）
    raises(ValueError, lambda: m.diagonal(-3))
    # 检查是否会引发 ValueError 异常，因为指定的对角线参数不是整数
    raises(ValueError, lambda: m.diagonal(pi))
    # 创建一个 2x3 的全一矩阵 M
    M = ones(2, 3)
    # 断言条件：将 M 的带状矩阵转换为字典，并与原始矩阵 M 进行比较
    assert banded({i: list(M.diagonal(i))
        for i in range(1-M.rows, M.cols)}) == M
# 定义测试函数 test_jordan_block，用于测试 Matrix 类的 jordan_block 方法
def test_jordan_block():
    # 断言不同参数形式下 jordan_block 方法返回相同的 Jordan 块矩阵
    assert Matrix.jordan_block(3, 2) == Matrix.jordan_block(3, eigenvalue=2) \
            == Matrix.jordan_block(size=3, eigenvalue=2) \
            == Matrix.jordan_block(3, 2, band='upper') \
            == Matrix.jordan_block(
                size=3, eigenval=2, eigenvalue=2) \
            == Matrix([
                [2, 1, 0],
                [0, 2, 1],
                [0, 0, 2]])

    # 断言使用 'lower' 带宽参数时返回正确的下三角 Jordan 块矩阵
    assert Matrix.jordan_block(3, 2, band='lower') == Matrix([
                    [2, 0, 0],
                    [1, 2, 0],
                    [0, 1, 2]])

    # 断言在没有给定特征值时引发 ValueError 异常
    raises(ValueError, lambda: Matrix.jordan_block(2))
    # 断言当 size 参数不为整数时引发 ValueError 异常
    raises(ValueError, lambda: Matrix.jordan_block(3.5, 2))
    # 断言没有指定 size 参数时引发 ValueError 异常
    raises(ValueError, lambda: Matrix.jordan_block(eigenvalue=2))
    # 断言给定矛盾的特征值参数时引发 ValueError 异常
    raises(ValueError,
    lambda: Matrix.jordan_block(
        eigenvalue=2, eigenval=4))

    # 断言使用别名关键字 size 与 eigenval 时返回相同的结果
    assert Matrix.jordan_block(size=3, eigenvalue=2) == \
        Matrix.jordan_block(size=3, eigenval=2)


# 定义测试函数 test_orthogonalize，用于测试 Matrix 类的 orthogonalize 方法
def test_orthogonalize():
    # 创建矩阵 m
    m = Matrix([[1, 2], [3, 4]])
    # 断言 orthogonalize 方法能够正确正交化给定向量
    assert m.orthogonalize(Matrix([[2], [1]])) == [Matrix([[2], [1]])]
    # 断言使用 normalize 参数正交化后的向量长度为 1
    assert m.orthogonalize(Matrix([[2], [1]]), normalize=True) == \
        [Matrix([[2*sqrt(5)/5], [sqrt(5)/5]])]
    # 断言正交化多个向量后返回正确的结果列表
    assert m.orthogonalize(Matrix([[1], [2]]), Matrix([[-1], [4]])) == \
        [Matrix([[1], [2]]), Matrix([[Rational(-12, 5)], [Rational(6, 5)]])]
    # 断言正交化零向量后返回空列表
    assert m.orthogonalize(Matrix([[0], [0]]), Matrix([[-1], [4]])) == \
        [Matrix([[-1], [4]])]
    # 断言正交化多个零向量后返回空列表
    assert m.orthogonalize(Matrix([[0], [0]])) == []

    # 创建矩阵 n 和向量列表 vecs
    n = Matrix([[9, 1, 9], [3, 6, 10], [8, 5, 2]])
    vecs = [Matrix([[-5], [1]]), Matrix([[-5], [2]]), Matrix([[-5], [-2]])]
    # 断言正交化多个向量后返回正确的结果列表
    assert n.orthogonalize(*vecs) == \
        [Matrix([[-5], [1]]), Matrix([[Rational(5, 26)], [Rational(25, 26)]])]

    # 创建零向量列表 vecs
    vecs = [Matrix([0, 0, 0]), Matrix([1, 2, 3]), Matrix([1, 4, 5])]
    # 断言正交化零向量列表时引发 ValueError 异常
    raises(ValueError, lambda: Matrix.orthogonalize(*vecs, rankcheck=True))

    # 创建不满秩向量列表 vecs
    vecs = [Matrix([1, 2, 3]), Matrix([4, 5, 6]), Matrix([7, 8, 9])]
    # 断言正交化不满秩向量列表时引发 ValueError 异常
    raises(ValueError, lambda: Matrix.orthogonalize(*vecs, rankcheck=True))


# 定义测试函数 test_wilkinson，用于测试 Matrix 类的 wilkinson 方法
def test_wilkinson():

    # 测试特征值为 1 时的 wilkinson 方法返回结果
    wminus, wplus = Matrix.wilkinson(1)
    assert wminus == Matrix([
                                [-1, 1, 0],
                                [1, 0, 1],
                                [0, 1, 1]])
    assert wplus == Matrix([
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1]])

    # 测试特征值为 3 时的 wilkinson 方法返回结果
    wminus, wplus = Matrix.wilkinson(3)
    assert wminus == Matrix([
                                [-3,  1,  0, 0, 0, 0, 0],
                                [1, -2,  1, 0, 0, 0, 0],
                                [0,  1, -1, 1, 0, 0, 0],
                                [0,  0,  1, 0, 1, 0, 0],
                                [0,  0,  0, 1, 1, 1, 0],
                                [0,  0,  0, 0, 1, 2, 1],
                                [0,  0,  0, 0, 0, 1, 3]])
    # 断言语句，用于确保 wplus 矩阵与给定的矩阵相同
    assert wplus == Matrix([
                            [3, 1, 0, 0, 0, 0, 0],
                            [1, 2, 1, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 1, 2, 1],
                            [0, 0, 0, 0, 0, 1, 3]])
def test_limit():
    # 定义符号变量 x 和 y
    x, y = symbols('x y')
    # 创建一个 2x1 的矩阵 m，包含表达式 1/x 和 y
    m = Matrix(2, 1, [1/x, y])
    # 断言当 x 趋近于 5 时，矩阵 m 的极限为 2x1 矩阵 [1/5, y]
    assert m.limit(x, 5) == Matrix(2, 1, [Rational(1, 5), y])
    # 创建一个 3x3 的矩阵 A，包含表达式 (1, 4, sin(x)/x), (y, 2, 4), (10, 5, x**2 + 1)
    A = Matrix(((1, 4, sin(x)/x), (y, 2, 4), (10, 5, x**2 + 1)))
    # 断言当 x 趋近于 0 时，矩阵 A 的极限为 3x3 矩阵 ((1, 4, 1), (y, 2, 4), (10, 5, 1))
    assert A.limit(x, 0) == Matrix(((1, 4, 1), (y, 2, 4), (10, 5, 1)))


def test_issue_13774():
    # 创建一个 3x3 的矩阵 M
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 创建一个长度为 3 的向量 v
    v = [1, 1, 1]
    # 断言使用矩阵 M 左乘向量 v 会抛出 TypeError 异常
    raises(TypeError, lambda: M*v)
    # 断言使用向量 v 右乘矩阵 M 会抛出 TypeError 异常
    raises(TypeError, lambda: v*M)


def test_companion():
    # 创建符号变量 x 和 y
    x = Symbol('x')
    y = Symbol('y')
    # 断言 Matrix.companion(1) 会抛出 ValueError 异常
    raises(ValueError, lambda: Matrix.companion(1))
    # 断言 Matrix.companion(Poly([1], x)) 会抛出 ValueError 异常
    raises(ValueError, lambda: Matrix.companion(Poly([1], x)))
    # 断言 Matrix.companion(Poly([2, 1], x)) 会抛出 ValueError 异常
    raises(ValueError, lambda: Matrix.companion(Poly([2, 1], x)))
    # 断言 Matrix.companion(Poly(x*y, [x, y])) 会抛出 ValueError 异常
    raises(ValueError, lambda: Matrix.companion(Poly(x*y, [x, y])))

    # 定义符号变量 c0, c1, c2
    c0, c1, c2 = symbols('c0:3')
    # 断言 Matrix.companion(Poly([1, c0], x)) 的结果是一个包含 [-c0] 的矩阵
    assert Matrix.companion(Poly([1, c0], x)) == Matrix([-c0])
    # 断言 Matrix.companion(Poly([1, c1, c0], x)) 的结果是一个 2x2 的矩阵 [[0, -c0], [1, -c1]]
    assert Matrix.companion(Poly([1, c1, c0], x)) == \
        Matrix([[0, -c0], [1, -c1]])
    # 断言 Matrix.companion(Poly([1, c2, c1, c0], x)) 的结果是一个 3x3 的矩阵 [[0, 0, -c0], [1, 0, -c1], [0, 1, -c2]]
    assert Matrix.companion(Poly([1, c2, c1, c0], x)) == \
        Matrix([[0, 0, -c0], [1, 0, -c1], [0, 1, -c2]])


def test_issue_10589():
    # 定义符号变量 x, y, z
    x, y, z = symbols("x, y z")
    # 创建一个 3x1 的矩阵 M1，包含向量 [x, y, z]
    M1 = Matrix([x, y, z])
    # 使用字典替换向量中的符号变量 x, y, z，得到矩阵 [[1], [2], [3]]
    M1 = M1.subs(zip([x, y, z], [1, 2, 3]))
    # 断言 M1 等于 3x1 的矩阵 [[1], [2], [3]]
    assert M1 == Matrix([[1], [2], [3]])

    # 创建一个 3x5 的矩阵 M2，所有元素为符号变量 x
    M2 = Matrix([[x, x, x, x, x], [x, x, x, x, x], [x, x, x, x, x]])
    # 使用字典替换所有符号变量 x，得到矩阵 [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    M2 = M2.subs(zip([x], [1]))
    # 断言 M2 等于 3x5 的矩阵 [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] 
    assert M2 == Matrix([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])


def test_rmul_pr19860():
    # 定义一个继承自 ImmutableDenseMatrix 的类 Foo
    class Foo(ImmutableDenseMatrix):
        _op_priority = MutableDenseMatrix._op_priority + 0.01

    # 创建一个 2x2 的矩阵 a
    a = Matrix(2, 2, [1, 2, 3, 4])
    # 创建一个 2x2 的 Foo 类型的矩阵 b
    b = Foo(2, 2, [1, 2, 3, 4])

    # 对 a 和 b 进行乘法运算，赋值给 c
    c = a*b

    # 断言 c 是 Foo 类型的实例
    assert isinstance(c, Foo)
    # 断言 c 等于矩阵 [[7, 10], [15, 22]]
    assert c == Matrix([[7, 10], [15, 22]])


def test_issue_18956():
    # 创建一个 Array 类型的矩阵 A
    A = Array([[1, 2], [3, 4]])
    # 创建一个 Matrix 类型的矩阵 B
    B = Matrix([[1,2],[3,4]])
    # 断言 Matrix 类型的矩阵 B 和 Array 类型的矩阵 A 相加会抛出 TypeError 异常
    raises(TypeError, lambda: B + A)
    # 断言 Array 类型的矩阵 A 和 Matrix 类型的矩阵 B 相加会抛出 TypeError 异常
    raises(TypeError, lambda: A + B)


def test__eq__():
    # 定义一个自定义类 My
    class My(object):
        # 定义 __iter__ 方法，返回迭代器
        def __iter__(self):
            yield 1
            yield 2
            return
        # 定义 __getitem__ 方法，返回索引值
        def __getitem__(self, i):
            return list(self)[i]

    # 创建一个 2x1 的矩阵 a，包含向量 [1, 2]
    a = Matrix(2, 1, [1, 2])
    # 断言 a 不等于 My 类的实例
    assert a != My()

    # 定义一个继承自 My 的子类 My_sympy
    class My_sympy(My):
        # 定义 _sympy_ 方法，返回 Matrix 对象
        def _sympy_(self):
            return Matrix(self)

    # 断
    # 遍历 Matrix 和 ImmutableMatrix 两个类
    for cls in Matrix, ImmutableMatrix:
        # 使用指定类的静态方法创建一个3x2的零矩阵
        m = cls.zeros(3, 2)
        # 使用 warns_deprecated_sympy 上下文管理器捕获过时警告
        with warns_deprecated_sympy():
            # 获取矩阵 m 的底层数据结构 _mat
            mat = m._mat
        # 断言 _mat 应该等于矩阵 m 的扁平化表示
        assert mat == m.flat()

    # 遍历 SparseMatrix 和 ImmutableSparseMatrix 两个类
    for cls in SparseMatrix, ImmutableSparseMatrix:
        # 使用指定类的静态方法创建一个3x2的零稀疏矩阵
        m = cls.zeros(3, 2)
        # 使用 warns_deprecated_sympy 上下文管理器捕获过时警告
        with warns_deprecated_sympy():
            # 获取稀疏矩阵 m 的底层数据结构 _smat
            smat = m._smat
        # 断言 _smat 应该等于稀疏矩阵 m 的以dok形式表示的转置
        assert smat == m.todok()
def test_division():
    # 创建一个矩阵对象 v，包含两个变量 x 和 y
    v = Matrix(1, 2, [x, y])
    # 断言 v 除以标量 z 的结果等于矩阵对象，其元素为 x/z 和 y/z
    assert v/z == Matrix(1, 2, [x/z, y/z])


def test_sum():
    # 创建一个矩阵对象 m，包含一个二维列表
    m = Matrix([[1, 2, 3], [x, y, x], [2*y, -50, z*x]])
    # 断言矩阵 m 与自身的加法结果等于特定的矩阵对象
    assert m + m == Matrix([[2, 4, 6], [2*x, 2*y, 2*x], [4*y, -100, 2*z*x]])
    # 创建一个矩阵对象 n，包含一个一维列表
    n = Matrix(1, 2, [1, 2])
    # 断言对 m 加 n 抛出 ShapeError 异常
    raises(ShapeError, lambda: m + n)


def test_abs():
    # 创建一个矩阵对象 m，包含一个二维列表
    m = Matrix([[1, -2], [x, y]])
    # 断言矩阵 m 的绝对值等于特定的矩阵对象
    assert abs(m) == Matrix([[1, 2], [Abs(x), Abs(y)]])
    # 重新赋值矩阵 m，包含一个一维列表
    m = Matrix(1, 2, [-3, x])
    # 创建一个矩阵对象 n，包含一个一维列表
    n = Matrix(1, 2, [3, Abs(x)])
    # 断言矩阵 m 的绝对值等于矩阵 n
    assert abs(m) == n


def test_addition():
    # 创建一个矩阵对象 a，包含一个二维元组
    a = Matrix((
        (1, 2),
        (3, 1),
    ))

    # 创建一个矩阵对象 b，包含一个二维元组
    b = Matrix((
        (1, 2),
        (3, 0),
    ))

    # 断言矩阵 a 与矩阵 b 的加法结果等于调用 a.add(b) 的结果，都等于特定的矩阵对象
    assert a + b == a.add(b) == Matrix([[2, 4], [6, 1]])


def test_fancy_index_matrix():
    # 遍历 Matrix 和 SparseMatrix 两种类型
    for M in (Matrix, SparseMatrix):
        # 创建一个 3x3 的矩阵对象 a，其元素为 0 到 8
        a = M(3, 3, range(9))
        # 断言矩阵 a 等于自身的所有元素
        assert a == a[:, :]
        # 断言矩阵 a 的第 1 行等于特定的矩阵对象
        assert a[1, :] == Matrix(1, 3, [3, 4, 5])
        # 断言矩阵 a 的第 1 列等于特定的矩阵对象
        assert a[:, 1] == Matrix([1, 4, 7])
        # 断言矩阵 a 的指定行索引等于特定的矩阵对象
        assert a[[0, 1], :] == Matrix([[0, 1, 2], [3, 4, 5]])
        # 断言矩阵 a 的指定行和列索引等于特定的矩阵对象
        assert a[[0, 1], 2] == a[[0, 1], [2]]
        # 断言矩阵 a 的指定行和列索引等于特定的矩阵对象
        assert a[2, [0, 1]] == a[[2], [0, 1]]
        # 断言矩阵 a 的指定列索引等于特定的矩阵对象
        assert a[:, [0, 1]] == Matrix([[0, 1], [3, 4], [6, 7]])
        # 断言矩阵 a 的指定元素等于特定值
        assert a[0, 0] == 0
        # 断言矩阵 a 的切片等于特定的矩阵对象
        assert a[0:2, :] == Matrix([[0, 1, 2], [3, 4, 5]])
        # 断言矩阵 a 的切片等于特定的矩阵对象
        assert a[:, 0:2] == Matrix([[0, 1], [3, 4], [6, 7]])
        # 断言矩阵 a 的步长切片等于特定的矩阵对象
        assert a[::2, 1] == a[[0, 2], 1]
        # 断言矩阵 a 的步长切片等于特定的矩阵对象
        assert a[1, ::2] == a[1, [0, 2]]
        # 重新赋值矩阵 a，包含一个 3x3 的矩阵对象，元素为 0 到 8
        a = M(3, 3, range(9))
        # 断言矩阵 a 的指定行索引等于特定的矩阵对象
        assert a[[0, 2, 1, 2, 1], :] == Matrix([
            [0, 1, 2],
            [6, 7, 8],
            [3, 4, 5],
            [6, 7, 8],
            [3, 4, 5]])
        # 断言矩阵 a 的指定列索引等于特定的矩阵对象
        assert a[:, [0,2,1,2,1]] == Matrix([
            [0, 2, 1, 2, 1],
            [3, 5, 4, 5, 4],
            [6, 8, 7, 8, 7]])

    # 创建一个 SparseMatrix 类型的零矩阵对象 a，行数为 3
    a = SparseMatrix.zeros(3)
    # 设置矩阵 a 的指定元素值
    a[1, 2] = 2
    a[0, 1] = 3
    a[2, 0] = 4
    # 断言从矩阵 a 中提取指定行和列索引的子矩阵等于特定的矩阵对象
    assert a.extract([1, 1], [2]) == Matrix([
    [2],
    [2]])
    # 断言从矩阵 a 中提取指定行和列索引的子矩阵等于特定的矩阵对象
    assert a.extract([1, 0], [2, 2, 2]) == Matrix([
    [2, 2, 2],
    [0, 0, 0]])
    # 断言从矩阵 a 中提取指定行和列索引的子矩阵等于特定的矩阵对象
    assert a.extract([1, 0, 1, 2], [2, 0, 1, 0]) == Matrix([
        [2, 0, 0, 0],
        [0, 0, 3, 0],
        [2, 0, 0, 0],
        [0, 4, 0, 4]])


def test_multiplication():
    # 创建一个矩阵对象 a，包含一个二维元组
    a = Matrix((
        (1, 2),
        (3, 1),
        (0, 6),
    ))

    # 创建一个矩阵对象 b，包含一个二维元组
    b = Matrix((
        (1, 2),
        (3, 0),
    ))

    # 断言 b 乘以 a 抛出 ShapeError 异常
    raises(ShapeError,
    # 断言：验证矩阵 c 中特定位置的值与预期值 x 相等
    assert c[0, 0] == x
    # 断言：验证矩阵 c 中特定位置的值为 2*x
    assert c[0, 1] == 2*x
    # 断言：验证矩阵 c 中特定位置的值为 3*x
    assert c[1, 0] == 3*x
    # 断言：验证矩阵 c 中特定位置的值为 0
    assert c[1, 1] == 0

    # 创建新的矩阵 c2，其每个元素为原矩阵 b 中对应元素乘以 x
    c2 = x * b
    # 断言：验证矩阵 c 和 c2 在每个位置上的元素值相等
    assert c == c2

    # 将矩阵 c 中的每个元素乘以 5
    c = 5 * b
    # 断言：验证结果矩阵 c 是 Matrix 类型的实例
    assert isinstance(c, Matrix)
    # 断言：验证矩阵 c 中特定位置的值为 5
    assert c[0, 0] == 5
    # 断言：验证矩阵 c 中特定位置的值为 2*5
    assert c[0, 1] == 2*5
    # 断言：验证矩阵 c 中特定位置的值为 3*5
    assert c[1, 0] == 3*5
    # 断言：验证矩阵 c 中特定位置的值为 0
    assert c[1, 1] == 0

    # 创建一个无穷大元素构成的矩阵 M
    M = Matrix([[oo, 0], [0, oo]])
    # 断言：验证 M 的平方等于 M 自身
    assert M ** 2 == M

    # 创建一个包含无穷大和零元素的矩阵 M
    M = Matrix([[oo, oo], [0, 0]])
    # 断言：验证 M 的平方等于一个包含 NaN (Not a Number) 元素的矩阵
    assert M ** 2 == Matrix([[nan, nan], [nan, nan]])

    # 创建一个列向量 A，其元素为 1
    A = Matrix(ones(3, 1))
    # 创建标量 _h，其值为 -1/2
    _h = -Rational(1, 2)
    # 创建列向量 B，其元素均为 _h
    B = Matrix([_h, _h, _h])
    # 断言：验证 A 和 B 的逐元素乘积等于特定的矩阵
    assert A.multiply_elementwise(B) == Matrix([
        [_h],
        [_h],
        [_h]])
# 定义一个测试函数 test_power
def test_power():
    # 测试非方阵的幂次运算是否引发 NonSquareMatrixError 异常
    raises(NonSquareMatrixError, lambda: Matrix((1, 2))**2)

    # 创建一个 2x2 的矩阵 A
    A = Matrix([[2, 3], [4, 5]])
    # 断言 A 的 5 次幂结果是否等于给定的矩阵
    assert A**5 == Matrix([[6140, 8097], [10796, 14237]])

    # 重新定义 A 为一个 3x3 的矩阵
    A = Matrix([[2, 1, 3], [4, 2, 4], [6, 12, 1]])
    # 断言 A 的 3 次幂结果是否等于给定的矩阵
    assert A**3 == Matrix([[290, 262, 251], [448, 440, 368], [702, 954, 433]])

    # 断言 A 的 0 次幂结果是否等于单位矩阵
    assert A**0 == eye(3)

    # 断言 A 的 1 次幂结果是否等于 A 自身
    assert A**1 == A

    # 断言一个单元素矩阵的 100 次幂结果是否等于该元素的 100 次方
    assert (Matrix([[2]]) ** 100)[0, 0] == 2**100

    # 断言一个 2x2 矩阵的整数幂次运算结果是否正确
    assert Matrix([[1, 2], [3, 4]])**Integer(2) == Matrix([[7, 10], [15, 22]])

    # 重新定义 A 为一个 2x2 的矩阵
    A = Matrix([[1,2],[4,5]])
    # 使用两种不同的方法计算 A 的 20 次幂，断言结果是否相等
    assert A.pow(20, method='cayley') == A.pow(20, method='multiply')

    # 断言 A 的整数幂次运算结果是否正确
    assert A**Integer(2) == Matrix([[9, 12], [24, 33]])

    # 断言单位矩阵的任意正整数次幂结果是否等于单位矩阵本身
    assert eye(2)**10000000 == eye(2)

    # 重新定义 A 为一个 2x2 的矩阵
    A = Matrix([[33, 24], [48, 57]])
    # 断言 A 的 S.Half 次幂的所有元素是否与指定的列表相等
    assert (A**S.Half)[:] == [5, 2, 4, 7]

    # 重新定义 A 为一个 2x2 的矩阵
    A = Matrix([[0, 4], [-1, 5]])
    # 断言 A 的 S.Half 次幂再平方是否等于 A 本身
    assert (A**S.Half)**2 == A

    # 断言一个特定的 2x2 矩阵的 S.Half 次幂结果是否等于给定的矩阵
    assert Matrix([[1, 0], [1, 1]])**S.Half == Matrix([[1, 0], [S.Half, 1]])

    # 断言一个特定的 2x2 矩阵的 0.5 次幂结果是否等于给定的矩阵
    assert Matrix([[1, 0], [1, 1]])**0.5 == Matrix([[1, 0], [0.5, 1]])

    # 导入符号变量 n，断言一个特定的 2x2 矩阵的 n 次幂结果是否等于给定的矩阵
    from sympy.abc import n
    assert Matrix([[1, a], [0, 1]])**n == Matrix([[1, a*n], [0, 1]])

    # 断言一个特定的 2x2 矩阵的 n 次幂结果是否符合预期的数学表达式
    assert Matrix([[b, a], [0, b]])**n == Matrix([[b**n, a*b**(n-1)*n], [0, b**n]])

    # 断言一个特定的 3x3 矩阵的各元素幂次运算结果是否符合预期
    assert Matrix([
        [a**n, a**(n - 1)*n, (a**n*n**2 - a**n*n)/(2*a**2)],
        [   0,         a**n,                  a**(n - 1)*n],
        [   0,            0,                          a**n]]) 

    # 断言一个特定的 3x3 矩阵的各元素幂次运算结果是否符合预期
    assert Matrix([[a, 1, 0], [0, a, 0], [0, 0, b]])**n == Matrix([
        [a**n, a**(n-1)*n, 0],
        [0, a**n, 0],
        [0, 0, b**n]])

    # 重新定义 A 为一个 2x2 的矩阵
    A = Matrix([[1, 0], [1, 7]])
    # 断言 A 的 _matrix_pow_by_jordan_blocks 方法计算结果是否等于 _eval_pow_by_recursion 方法计算结果
    assert A._matrix_pow_by_jordan_blocks(S(3)) == A._eval_pow_by_recursion(3)

    # 重新定义 A 为一个单元素的矩阵
    A = Matrix([[2]])
    # 断言 A 的 10 次幂结果是否等于给定的矩阵，并且与使用不同方法计算的结果也相等
    assert A**10 == Matrix([[2**10]]) == A._matrix_pow_by_jordan_blocks(S(10)) == \
        A._eval_pow_by_recursion(10)

    # 测试无法进行乔丹块分解的矩阵，预期引发 MatrixError 异常
    m = Matrix([[3, 0, 0, 0, -3], [0, -3, -3, 0, 3], [0, 3, 0, 3, 0], [0, 0, 3, 0, 3], [3, 0, 0, 3, 0]])
    raises(MatrixError, lambda: m._matrix_pow_by_jordan_blocks(S(10)))

    # 测试一个特定的矩阵，预期引发 MatrixError 异常
    raises(MatrixError, lambda: Matrix([[1, 1], [3, 3]])._matrix_pow_by_jordan_blocks(S(-10)))

    # 重新定义 A 为一个特定的 3x3 矩阵
    A = Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # Nilpotent jordan block size 3
    # 断言 A 的 10.0 次幂结果是否等于预期的矩阵
    assert A**10.0 == Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    # 测试不支持非整数幂次的情况，预期引发 ValueError 异常
    raises(ValueError, lambda: A**2.1)
    raises(ValueError, lambda: A**Rational(3, 2))

    # 重新定义 A 为一个特定的 2x2 矩阵
    A = Matrix([[8, 1], [3, 2]])
    # 断言 A 的 10.0 次幂结果是否等于预期的矩阵
    assert A**10.0 == Matrix([[1760744107, 272388050], [817164150, 126415807]])

    # 重新定义 A 为一个特定的 3x3 矩阵
    A = Matrix([[0, 0, 1], [0, 0, 1], [0, 0, 1]])  # Nilpotent jordan block size 1
    # 断言 A 的 10.0 次
    # 断言：验证 A 的 n 次幂是否等于指定的矩阵
    assert A**n == Matrix([
        [KroneckerDelta(0, n), KroneckerDelta(1, n), -KroneckerDelta(0, n) - KroneckerDelta(1, n) + 1],
        [                   0, KroneckerDelta(0, n),                         1 - KroneckerDelta(0, n)],
        [                   0,                    0,                                                1]])

    # 断言：验证 A 的 n+2 次幂是否等于特定的矩阵
    assert A**(n + 2) == Matrix([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    # 引发异常：验证 A 的有理次幂是否引发 ValueError 异常
    raises(ValueError, lambda: A**Rational(3, 2))

    # 初始化矩阵 A
    A = Matrix([[0, 0, 1], [3, 0, 1], [4, 3, 1]])

    # 断言：验证 A 的 5.0 次幂是否等于指定的矩阵
    assert A**5.0 == Matrix([[168,  72,  89], [291, 144, 161], [572, 267, 329]])

    # 断言：验证 A 的 5.0 次幂是否等于 A 的整数次幂
    assert A**5.0 == A**5

    # 重新定义矩阵 A
    A = Matrix([[0, 1, 0],[-1, 0, 0],[0, 0, 0]])

    # 定义符号 n
    n = Symbol("n")

    # 计算 A 的 n 次幂
    An = A**n

    # 断言：验证用 n 替换后的 An 是否等于 A 的平方
    assert An.subs(n, 2).doit() == A**2

    # 引发异常：验证 An 在 n 为负数时是否引发 ValueError 异常
    raises(ValueError, lambda: An.subs(n, -2).doit())

    # 断言：验证 An 乘以自身是否等于 A 的 2n 次幂
    assert An * An == A**(2*n)

    # concretizing behavior for non-integer and complex powers

    # 重新定义矩阵 A
    A = Matrix([[0,0,0],[0,0,0],[0,0,0]])

    # 定义符号 n，限制 n 为正整数
    n = Symbol('n', integer=True, positive=True)

    # 断言：验证 A 的 n 次幂是否等于 A 自身
    assert A**n == A

    # 定义符号 n，限制 n 为非负整数
    n = Symbol('n', integer=True, nonnegative=True)

    # 断言：验证 A 的 n 次幂是否为对角矩阵，对角线元素为 0**n
    assert A**n == diag(0**n, 0**n, 0**n)

    # 断言：验证用 n 替换后的 A**n 是否为单位矩阵
    assert (A**n).subs(n, 0) == eye(3)

    # 断言：验证用 n 替换后的 A**n 是否为零矩阵
    assert (A**n).subs(n, 1) == zeros(3)

    # 重新定义矩阵 A
    A = Matrix([[2,0,0],[0,2,0],[0,0,2]])

    # 断言：验证 A 的 2.1 次幂是否为对角矩阵，对角线元素为 2**2.1
    assert A**2.1 == diag(2**2.1, 2**2.1, 2**2.1)

    # 断言：验证 A 的虚数次幂是否为对角矩阵，对角线元素为 2**I
    assert A**I == diag(2**I, 2**I, 2**I)

    # 重新定义矩阵 A
    A = Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 1]])

    # 引发异常：验证 A 的 2.1 次幂是否引发 ValueError 异常
    raises(ValueError, lambda: A**2.1)

    # 引发异常：验证 A 的虚数次幂是否引发 ValueError 异常
    raises(ValueError, lambda: A**I)

    # 重新定义矩阵 A
    A = Matrix([[S.Half, S.Half], [S.Half, S.Half]])

    # 断言：验证 A 的 S.Half 次幂是否等于 A 自身
    assert A**S.Half == A

    # 重新定义矩阵 A
    A = Matrix([[1, 1],[3, 3]])

    # 断言：验证 A 的 S.Half 次幂是否等于指定的矩阵
    assert A**S.Half == Matrix([[S.Half, S.Half], [3*S.Half, 3*S.Half]])
def test_issue_17247_expression_blowup_1():
    # 创建一个 2x2 的矩阵 M，其中元素是关于变量 x 的表达式
    M = Matrix([[1+x, 1-x], [1-x, 1+x]])
    # 使用 dotprodsimp 上下文环境，简化矩阵 M 的指数函数，并展开结果
    with dotprodsimp(True):
        # 断言展开后的矩阵是否等于给定的 2x2 矩阵
        assert M.exp().expand() == Matrix([
            [ (exp(2*x) + exp(2))/2, (-exp(2*x) + exp(2))/2],
            [(-exp(2*x) + exp(2))/2,  (exp(2*x) + exp(2))/2]])

def test_issue_17247_expression_blowup_2():
    # 创建一个 2x2 的矩阵 M，其中元素是关于变量 x 的表达式
    M = Matrix([[1+x, 1-x], [1-x, 1+x]])
    # 使用 dotprodsimp 上下文环境
    with dotprodsimp(True):
        # 计算矩阵 M 的 Jordan 形式，并分别赋值给 P 和 J
        P, J = M.jordan_form ()
        # 断言矩阵乘法 P*J*P.inv() 是否成立
        assert P*J*P.inv()

def test_issue_17247_expression_blowup_3():
    # 创建一个 2x2 的矩阵 M，其中元素是关于变量 x 的表达式
    M = Matrix([[1+x, 1-x], [1-x, 1+x]])
    # 使用 dotprodsimp 上下文环境
    with dotprodsimp(True):
        # 断言矩阵 M 的 100 次方是否等于给定的 2x2 矩阵
        assert M**100 == Matrix([
            [633825300114114700748351602688*x**100 + 633825300114114700748351602688, 633825300114114700748351602688 - 633825300114114700748351602688*x**100],
            [633825300114114700748351602688 - 633825300114114700748351602688*x**100, 633825300114114700748351602688*x**100 + 633825300114114700748351602688]])

def test_issue_17247_expression_blowup_4():
    # 由于当前主干版本上这个矩阵计算耗时极长，使用缩写版本进行测试，这里保留用于未来优化的测试
    # M = Matrix(S('''[
    #     [             -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,      1/4 - 5*I/16,      65/128 + 87*I/64,         -9/32 - I/16,      183/256 - 97*I/128,       3/64 + 13*I/64,         -23/32 - 59*I/256,      15/128 - 3*I/32,        19/256 + 551*I/1024],
    #     [-149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,  85/256 - 33*I/16,  805/128 + 2415*I/512, -219/128 + 115*I/256, 6301/4096 - 6609*I/1024,  119/128 + 143*I/128, -10879/2048 + 4343*I/4096,  129/256 - 549*I/512, 42533/16384 + 29103*I/8192],
    #     [          1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,         1/4 - 5*I/16,        65/128 + 87*I/64,         -9/32 - I/16,        183/256 - 97*I/128,       3/64 + 13*I/64,          -23/32 - 59*I/256],
    #     [   -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,     85/256 - 33*I/16,    805/128 + 2415*I/512, -219/128 + 115*I/256,   6301/4096 - 6609*I/1024,  119/128 + 143*I/128,  -10879/2048 + 4343*I/4096],
    #     [            1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,            1/4 + I/2,        -129/64 - 9*I/64,         1/4 - 5*I/16,          65/128 + 87*I/64,         -9/32 - I/16,         183/256 - 97*I/128],
    #     [         21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,     125/64 + 87*I/64,   -2063/256 + 541*I/128,     85/256 - 33*I/16,      805/128 + 2415*I/512, -219/128 + 115*I/256,    6301/4096 - 6609*I/1024],
    # '''))
    pass  # 该矩阵测试用例暂时被禁用，因为其计算时间过长
# 创建一个包含复数的矩阵，每个元素都是复数值
M = Matrix([
    [               -2,         17/4 - 13*I/2,             1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,                 -3/4,         45/32 - 37*I/16,            1/4 + I/2,          -129/64 - 9*I/64,         1/4 - 5*I/16,           65/128 + 87*I/64],
    [     1/4 + 13*I/4,    -825/64 - 147*I/32,          21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64,    -149/64 + 49*I/32,   -177/128 - 1369*I/128,     125/64 + 87*I/64,     -2063/256 + 541*I/128,     85/256 - 33*I/16,       805/128 + 2415*I/512],
    [             -4*I,            27/2 + 6*I,                -2,         17/4 - 13*I/2,             1 + I,         -19/4 + 5*I/4,              1/2 - I,           9/4 + 55*I/16,                 -3/4,           45/32 - 37*I/16,            1/4 + I/2,           -129/64 - 9*I/64],
    [      1/4 + 5*I/2,       -23/8 - 57*I/16,      1/4 + 13*I/4,    -825/64 - 147*I/32,          21/8 + I,    -537/64 + 143*I/16,       -5/8 - 39*I/16,     2473/256 + 137*I/64,    -149/64 + 49*I/32,     -177/128 - 1369*I/128,     125/64 + 87*I/64,      -2063/256 + 541*I/128],
    [               -4,               9 - 5*I,              -4*I,            27/2 + 6*I,                -2,         17/4 - 13*I/2,                1 + I,           -19/4 + 5*I/4,              1/2 - I,             9/4 + 55*I/16,                 -3/4,            45/32 - 37*I/16],
    [             -2*I,        119/8 + 29*I/4,       1/4 + 5*I/2,       -23/8 - 57*I/16,      1/4 + 13*I/4,    -825/64 - 147*I/32,             21/8 + I,      -537/64 + 143*I/16,       -5/8 - 39*I/16,       2473/256 + 137*I/64,    -149/64 + 49*I/32,      -177/128 - 1369*I/128]
])
# 断言矩阵 M 的 10 次方等于给定的矩阵
assert M**10 == Matrix([
# 计算列表中的复数值，每个值是一个复数常数乘以一个复数
[
    # 第一个复数常数乘以一个复数
    7*(-221393644768594642173548179825793834595 - 1861633166167425978847110897013541127952*I)/9671406556917033397649408,
    # 第二个复数常数乘以一个复数
    15*(31670992489131684885307005100073928751695 + 10329090958303458811115024718207404523808*I)/77371252455336267181195264,
    # 第三个复数常数乘以一个复数
    7*(-3710978679372178839237291049477017392703 + 1377706064483132637295566581525806894169*I)/19342813113834066795298816,
    # 第四个复数常数乘以一个复数
    (9727707023582419994616144751727760051598 - 59261571067013123836477348473611225724433*I)/9671406556917033397649408,
    # 第五个复数常数乘以一个复数
    (31896723509506857062605551443641668183707 + 54643444538699269118869436271152084599580*I)/38685626227668133590597632,
    # 第六个复数常数乘以一个复数
    (-2024044860947539028275487595741003997397402 + 130959428791783397562960461903698670485863*I)/309485009821345068724781056,
    # 第七个复数常数乘以一个复数
    3*(26190251453797590396533756519358368860907 - 27221191754180839338002754608545400941638*I)/77371252455336267181195264,
    # 第八个复数常数乘以一个复数
    (1154643595139959842768960128434994698330461 + 3385496216250226964322872072260446072295634*I)/618970019642690137449562112,
    # 第九个复数常数乘以一个复数
    3*(-31849347263064464698310044805285774295286 - 11877437776464148281991240541742691164309*I)/77371252455336267181195264,
    # 第十个复数常数乘以一个复数
    (4661330392283532534549306589669150228040221 - 4171259766019818631067810706563064103956871*I)/1237940039285380274899124224,
    # 第十一个复数常数乘以一个复数
    (9598353794289061833850770474812760144506 + 358027153990999990968244906482319780943983*I)/309485009821345068724781056,
    # 第十二个复数常数乘以一个复数
    (-9755135335127734571547571921702373498554177 - 4837981372692695195747379349593041939686540*I)/2475880078570760549798248448
]
# 复杂的数学表达式，包含多个复数的列表
[
    # 第一个复数，实部 (-379516731607474268954110071392894274962069 - 422272153179747548473724096872271700878296*I) / 77371252455336267181195264，虚部 (41324748029613152354787280677832014263339501 - 12715121258662668420833935373453570749288074*I) / 1237940039285380274899124224
    (-379516731607474268954110071392894274962069 - 422272153179747548473724096872271700878296*I) / 77371252455336267181195264,

    # 第二个复数，实部 (41324748029613152354787280677832014263339501 - 12715121258662668420833935373453570749288074*I) / 1237940039285380274899124224，虚部 (-339216903907423793947110742819264306542397 + 494174755147303922029979279454787373566517*I) / 77371252455336267181195264
    (41324748029613152354787280677832014263339501 - 12715121258662668420833935373453570749288074*I) / 1237940039285380274899124224,

    # 第三个复数，实部 (-339216903907423793947110742819264306542397 + 494174755147303922029979279454787373566517*I) / 77371252455336267181195264，虚部 (-18121350839962855576667529908850640619878381 - 37413012454129786092962531597292531089199003*I) / 1237940039285380274899124224
    (-339216903907423793947110742819264306542397 + 494174755147303922029979279454787373566517*I) / 77371252455336267181195264,

    # 第四个复数，实部 (-18121350839962855576667529908850640619878381 - 37413012454129786092962531597292531089199003*I) / 1237940039285380274899124224，虚部 (2489661087330511608618880408199633556675926 + 1137821536550153872137379935240732287260863*I) / 309485009821345068724781056
    (-18121350839962855576667529908850640619878381 - 37413012454129786092962531597292531089199003*I) / 1237940039285380274899124224,

    # 第五个复数，实部 (2489661087330511608618880408199633556675926 + 1137821536550153872137379935240732287260863*I) / 309485009821345068724781056，虚部 (-136644109701594123227587016790354220062972119 + 110130123468183660555391413889600443583585272*I) / 4951760157141521099596496896
    (2489661087330511608618880408199633556675926 + 1137821536550153872137379935240732287260863*I) / 309485009821345068724781056,

    # 第六个复数，实部 (-136644109701594123227587016790354220062972119 + 110130123468183660555391413889600443583585272*I) / 4951760157141521099596496896，虚部 (1488043981274920070468141664150073426459593 - 9691968079933445130866371609614474474327650*I) / 1237940039285380274899124224
    (-136644109701594123227587016790354220062972119 + 110130123468183660555391413889600443583585272*I) / 4951760157141521099596496896,

    # 第七个复数，实部 (1488043981274920070468141664150073426459593 - 9691968079933445130866371609614474474327650*I) / 1237940039285380274899124224，虚部 27 * (4636797403026872518131756991410164760195942 + 3369103221138229204457272860484005850416533*I) / 4951760157141521099596496896
    (1488043981274920070468141664150073426459593 - 9691968079933445130866371609614474474327650*I) / 1237940039285380274899124224,

    # 第八个复数，实部 27 * (4636797403026872518131756991410164760195942 + 3369103221138229204457272860484005850416533*I) / 4951760157141521099596496896，虚部 (-8534279107365915284081669381642269800472363 + 2241118846262661434336333368511372725482742*I) / 1237940039285380274899124224
    27 * (4636797403026872518131756991410164760195942 + 3369103221138229204457272860484005850416533*I) / 4951760157141521099596496896,

    # 第九个复数，实部 (-8534279107365915284081669381642269800472363 + 2241118846262661434336333368511372725482742*I) / 1237940039285380274899124224，虚部 (60923350128174260992536531692058086830950875 - 263673488093551053385865699805250505661590126*I) / 9903520314283042199192993792
    (-8534279107365915284081669381642269800472363 + 2241118846262661434336333368511372725482742*I) / 1237940039285380274899124224,

    # 第十个复数，实部 (60923350128174260992536531692058086830950875 - 263673488093551053385865699805250505661590126*I) / 9903520314283042199192993792，虚部 (18520943561240714459282253753348921824172569 + 24846649186468656345966986622110971925703604*I) / 4951760157141521099596496896
    (60923350128174260992536531692058086830950875 - 263673488093551053385865699805250505661590126*I) / 9903520314283042199192993792,

    # 第十一个复数，实部 (18520943561240714459282253753348921824172569 + 24846649186468656345966986622110971925703604*I) / 4951760157141521099596496896，虚部 (-232781130692604829085973604213529649638644431 + 35981505277760667933017117949103953338570617*I) / 9903520314283042199192993792
    (18520943561240714459282253753348921824172569 + 24846649186468656345966986622110971925703604*I) / 4951760157141521099596496896,

    # 第十二个复数，实部 (-232781130692604829085973604213529649638644431 + 35981505277760667933017117949103953338570617*I) / 9903520314283042199192993792
    (-232781130692604829085973604213529649638644431 + 35981505277760667933017117949103953338570617*I) / 9903520314283042199192993792
]
# 定义一个包含复数的列表，每个复数由实部和虚部组成
[
    # 第一个复数
    (
        8742968295129404279528270438201520488950 + 3061473358639249112126847237482570858327j
    ) / 4835703278458516698824704,
    # 第二个复数
    (
        -245657313712011778432792959787098074935273 + 253113767861878869678042729088355086740856j
    ) / 38685626227668133590597632,
    # 第三个复数
    (
        1947031161734702327107371192008011621193 - 19462330079296259148177542369999791122762j
    ) / 9671406556917033397649408,
    # 第四个复数
    (
        552856485625209001527688949522750288619217 + 392928441196156725372494335248099016686580j
    ) / 77371252455336267181195264,
    # 第五个复数
    (
        -44542866621905323121630214897126343414629 + 3265340021421335059323962377647649632959j
    ) / 19342813113834066795298816,
    # 第六个复数
    (
        136272594005759723105646069956434264218730 - 330975364731707309489523680957584684763587j
    ) / 38685626227668133590597632,
    # 第七个复数
    (
        27392593965554149283318732469825168894401 + 75157071243800133880129376047131061115278j
    ) / 38685626227668133590597632,
    # 第八个复数
    7 * (
        -357821652913266734749960136017214096276154 - 45509144466378076475315751988405961498243j
    ) / 309485009821345068724781056,
    # 第九个复数
    (
        104485001373574280824835174390219397141149 - 99041000529599568255829489765415726168162j
    ) / 77371252455336267181195264,
    # 第十个复数
    (
        1198066993119982409323525798509037696321291 + 4249784165667887866939369628840569844519936j
    ) / 618970019642690137449562112,
    # 第十一个复数
    (
        -114985392587849953209115599084503853611014 - 52510376847189529234864487459476242883449j
    ) / 77371252455336267181195264,
    # 第十二个复数
    (
        6094620517051332877965959223269600650951573 - 4683469779240530439185019982269137976201163j
    ) / 1237940039285380274899124224,
]
# 定义一个包含复数的列表，每个元素都是一个复数对象
[ (611292255597977285752123848828590587708323 - 216821743518546668382662964473055912169502*I)/77371252455336267181195264,  
  (-1144023204575811464652692396337616594307487 + 12295317806312398617498029126807758490062855*I)/309485009821345068724781056, 
  (-374093027769390002505693378578475235158281 - 573533923565898290299607461660384634333639*I)/77371252455336267181195264,   
  (47405570632186659000138546955372796986832987 - 2837476058950808941605000274055970055096534*I)/1237940039285380274899124224,   
  (-571573207393621076306216726219753090535121 + 533381457185823100878764749236639320783831*I)/77371252455336267181195264,     
  (-7096548151856165056213543560958582513797519 - 24035731898756040059329175131592138642195366*I)/618970019642690137449562112,  
  (2396762128833271142000266170154694033849225 + 1448501087375679588770230529017516492953051*I)/309485009821345068724781056, 
  (-150609293845161968447166237242456473262037053 + 92581148080922977153207018003184520294188436*I)/4951760157141521099596496896,  
  5*(270278244730804315149356082977618054486347 - 1997830155222496880429743815321662710091562*I)/1237940039285380274899124224,   
  (62978424789588828258068912690172109324360330 + 44803641177219298311493356929537007630129097*I)/2475880078570760549798248448, 
  19*(-451431106327656743945775812536216598712236 + 114924966793632084379437683991151177407937*I)/1237940039285380274899124224,   
  (63417747628891221594106738815256002143915995 - 261508229397507037136324178612212080871150958*I)/9903520314283042199192993792]
# 创建一个包含复数的列表，每个复数都是形如 (实部 + 虚部 * I) 的形式
[ 
    # 第一个复数 (-2144231934021288786200752920446633703357 + 2305614436009705803670842248131563850246*I)/1208925819614629174706176
    (-2144231934021288786200752920446633703357 + 2305614436009705803670842248131563850246*I) / 1208925819614629174706176, 
    
    # 第二个复数 (-90720949337459896266067589013987007078153 - 221951119475096403601562347412753844534569*I)/19342813113834066795298816
    (-90720949337459896266067589013987007078153 - 221951119475096403601562347412753844534569*I) / 19342813113834066795298816, 
    
    # 第三个复数 (11590973613116630788176337262688659880376 + 6514520676308992726483494976339330626159*I)/4835703278458516698824704
    (11590973613116630788176337262688659880376 + 6514520676308992726483494976339330626159*I) / 4835703278458516698824704, 
    
    # 第四个复数 3*(-131776217149000326618649542018343107657237 + 79095042939612668486212006406818285287004*I)/38685626227668133590597632
    3 * (-131776217149000326618649542018343107657237 + 79095042939612668486212006406818285287004*I) / 38685626227668133590597632, 
    
    # 第五个复数 (10100577916793945997239221374025741184951 - 28631383488085522003281589065994018550748*I)/9671406556917033397649408
    (10100577916793945997239221374025741184951 - 28631383488085522003281589065994018550748*I) / 9671406556917033397649408, 
    
    # 第六个复数 67*(10090295594251078955008130473573667572549 + 10449901522697161049513326446427839676762*I)/77371252455336267181195264
    67 * (10090295594251078955008130473573667572549 + 10449901522697161049513326446427839676762*I) / 77371252455336267181195264, 
    
    # 第七个复数 (-54270981296988368730689531355811033930513 - 3413683117592637309471893510944045467443*I)/19342813113834066795298816
    (-54270981296988368730689531355811033930513 - 3413683117592637309471893510944045467443*I) / 19342813113834066795298816, 
    
    # 第八个复数 (440372322928679910536575560069973699181278 - 736603803202303189048085196176918214409081*I)/77371252455336267181195264
    (440372322928679910536575560069973699181278 - 736603803202303189048085196176918214409081*I) / 77371252455336267181195264, 
    
    # 第九个复数 (33220374714789391132887731139763250155295 + 92055083048787219934030779066298919603554*I)/38685626227668133590597632
    (33220374714789391132887731139763250155295 + 92055083048787219934030779066298919603554*I) / 38685626227668133590597632, 
    
    # 第十个复数 5*(-594638554579967244348856981610805281527116 - 82309245323128933521987392165716076704057*I)/309485009821345068724781056
    5 * (-594638554579967244348856981610805281527116 - 82309245323128933521987392165716076704057*I) / 309485009821345068724781056, 
    
    # 第十一个复数 (128056368815300084550013708313312073721955 - 114619107488668120303579745393765245911404*I)/77371252455336267181195264
    (128056368815300084550013708313312073721955 - 114619107488668120303579745393765245911404*I) / 77371252455336267181195264, 
    
    # 第十二个复数 21*(59839959255173222962789517794121843393573 + 241507883613676387255359616163487405826334*I)/618970019642690137449562112
    21 * (59839959255173222962789517794121843393573 + 241507883613676387255359616163487405826334*I) / 618970019642690137449562112
]
# 创建一个包含复数的列表，每个复数由实部和虚部组成
[
    # 第一个复数 (-13454485022325376674626653802541391955147 + 184471402121905621396582628515905949793486*I) / 19342813113834066795298816
    (-13454485022325376674626653802541391955147 + 184471402121905621396582628515905949793486*I) / 19342813113834066795298816,
    # 第二个复数 (-6158730123400322562149780662133074862437105 - 3416173052604643794120262081623703514107476*I) / 154742504910672534362390528
    (-6158730123400322562149780662133074862437105 - 3416173052604643794120262081623703514107476*I) / 154742504910672534362390528,
    # 第三个复数 (770558003844914708453618983120686116100419 - 127758381209767638635199674005029818518766*I) / 77371252455336267181195264
    (770558003844914708453618983120686116100419 - 127758381209767638635199674005029818518766*I) / 77371252455336267181195264,
    # 第四个复数 (-4693005771813492267479835161596671660631703 + 12703585094750991389845384539501921531449948*I) / 309485009821345068724781056
    (-4693005771813492267479835161596671660631703 + 12703585094750991389845384539501921531449948*I) / 309485009821345068724781056,
    # 第五个复数 (-295028157441149027913545676461260860036601 - 841544569970643160358138082317324743450770*I) / 77371252455336267181195264
    (-295028157441149027913545676461260860036601 - 841544569970643160358138082317324743450770*I) / 77371252455336267181195264,
    # 第六个复数 (56716442796929448856312202561538574275502893 + 7216818824772560379753073185990186711454778*I) / 1237940039285380274899124224
    (56716442796929448856312202561538574275502893 + 7216818824772560379753073185990186711454778*I) / 1237940039285380274899124224,
    # 第七个复数 15 * (-87061038932753366532685677510172566368387 + 61306141156647596310941396434445461895538*I) / 154742504910672534362390528
    15 * (-87061038932753366532685677510172566368387 + 61306141156647596310941396434445461895538*I) / 154742504910672534362390528,
    # 第八个复数 (-3455315109680781412178133042301025723909347 - 24969329563196972466388460746447646686670670*I) / 618970019642690137449562112
    (-3455315109680781412178133042301025723909347 - 24969329563196972466388460746447646686670670*I) / 618970019642690137449562112,
    # 第九个复数 (2453418854160886481106557323699250865361849 + 1497886802326243014471854112161398141242514*I) / 309485009821345068724781056
    (2453418854160886481106557323699250865361849 + 1497886802326243014471854112161398141242514*I) / 309485009821345068724781056,
    # 第十个复数 (-151343224544252091980004429001205664193082173 + 90471883264187337053549090899816228846836628*I) / 4951760157141521099596496896
    (-151343224544252091980004429001205664193082173 + 90471883264187337053549090899816228846836628*I) / 4951760157141521099596496896,
    # 第十一个复数 (1652018205533026103358164026239417416432989 - 9959733619236515024261775397109724431400162*I) / 1237940039285380274899124224
    (1652018205533026103358164026239417416432989 - 9959733619236515024261775397109724431400162*I) / 1237940039285380274899124224,
    # 第十二个复数 3 * (40676374242956907656984876692623172736522006 + 31023357083037817469535762230872667581366205*I) / 4951760157141521099596496896
    3 * (40676374242956907656984876692623172736522006 + 31023357083037817469535762230872667581366205*I) / 4951760157141521099596496896
]
# 复杂的数学表达式，可能是某种数值计算的结果或者表达式
[(-1226990509403328460274658603410696548387 - 4131739423109992672186585941938392788458*I)/1208925819614629174706176,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 (162392818524418973411975140074368079662703 + 23706194236915374831230612374344230400704*I)/9671406556917033397649408,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 (-3935678233089814180000602553655565621193 + 2283744757287145199688061892165659502483*I)/1208925819614629174706176,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 (-2400210250844254483454290806930306285131 - 315571356806370996069052930302295432758205*I)/19342813113834066795298816,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 (13365917938215281056563183751673390817910 + 15911483133819801118348625831132324863881*I)/4835703278458516698824704,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 3*(-215950551370668982657516660700301003897855 + 51684341999223632631602864028309400489378*I)/38685626227668133590597632,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 (20886089946811765149439844691320027184765 - 30806277083146786592790625980769214361844*I)/9671406556917033397649408,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 (562180634592713285745940856221105667874855 + 1031543963988260765153550559766662245114916*I)/77371252455336267181195264,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 (-65820625814810177122941758625652476012867 - 12429918324787060890804395323920477537595*I)/19342813113834066795298816,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 (319147848192012911298771180196635859221089 - 402403304933906769233365689834404519960394*I)/38685626227668133590597632,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 (23035615120921026080284733394359587955057 + 115351677687031786114651452775242461310624*I)/38685626227668133590597632,
 # 复杂的数学表达式，可能是某种数值计算的结果或者表达式
 (-3426830634881892756966440108592579264936130 - 1022954961164128745603407283836365128598559*I)/309485009821345068724781056]
# 定义一个包含复数的列表，每个元素都是一个复数
[
    # 第一个复数：实部和虚部的数学表达式
    (-192574788060137531023716449082856117537757 - 69222967328876859586831013062387845780692*I) / 19342813113834066795298816,
    # 第二个复数：实部和虚部的数学表达式
    (2736383768828013152914815341491629299773262 - 2773252698016291897599353862072533475408743*I) / 77371252455336267181195264,
    # 第三个复数：实部和虚部的数学表达式
    (-23280005281223837717773057436155921656805 + 214784953368021840006305033048142888879224*I) / 19342813113834066795298816,
    # 第四个复数：实部和虚部的数学表达式
    (-3035247484028969580570400133318947903462326 - 2195168903335435855621328554626336958674325*I) / 77371252455336267181195264,
    # 第五个复数：实部和虚部的数学表达式
    (984552428291526892214541708637840971548653 - 64006622534521425620714598573494988589378*I) / 77371252455336267181195264,
    # 第六个复数：实部和虚部的数学表达式
    (-3070650452470333005276715136041262898509903 + 7286424705750810474140953092161794621989080*I) / 154742504910672534362390528,
    # 第七个复数：实部和虚部的数学表达式
    (-147848877109756404594659513386972921139270 - 416306113044186424749331418059456047650861*I) / 38685626227668133590597632,
    # 第八个复数：实部和虚部的数学表达式
    (55272118474097814260289392337160619494260781 + 7494019668394781211907115583302403519488058*I) / 1237940039285380274899124224,
    # 第九个复数：实部和虚部的数学表达式
    (-581537886583682322424771088996959213068864 + 542191617758465339135308203815256798407429*I) / 77371252455336267181195264,
    # 第十个复数：实部和虚部的数学表达式
    (-6422548983676355789975736799494791970390991 - 23524183982209004826464749309156698827737702*I) / 618970019642690137449562112,
    # 第十一个复数：实部和虚部的数学表达式
    7 * (180747195387024536886923192475064903482083 + 84352527693562434817771649853047924991804*I) / 154742504910672534362390528,
    # 第十二个复数：实部和虚部的数学表达式
    (-135485179036717001055310712747643466592387031 + 102346575226653028836678855697782273460527608*I) / 4951760157141521099596496896
]
# 定义一个复数列表，包含多个复数元素
[
    # 第一个复数：实部 (3384238362616083147067025892852431152105)，虚部 (156724444932584900214919898954874618256)
    (3384238362616083147067025892852431152105 + 156724444932584900214919898954874618256*I) / 604462909807314587353088,
    # 第二个复数：实部 (-59558300950677430189587207338385764871866)，虚部 (114427143574375271097298201388331237478857)
    (-59558300950677430189587207338385764871866 + 114427143574375271097298201388331237478857*I) / 4835703278458516698824704,
    # 第三个复数：实部 (-1356835789870635633517710130971800616227)，虚部 (-7023484098542340388800213478357340875410)
    (-1356835789870635633517710130971800616227 - 7023484098542340388800213478357340875410*I) / 1208925819614629174706176,
    # 第四个复数：实部 (234884918567993750975181728413524549575881)，虚部 (79757294640629983786895695752733890213506)
    (234884918567993750975181728413524549575881 + 79757294640629983786895695752733890213506*I) / 9671406556917033397649408,
    # 第五个复数：实部 (-7632732774935120473359202657160313866419)，虚部 (2905452608512927560554702228553291839465)
    (-7632732774935120473359202657160313866419 + 2905452608512927560554702228553291839465*I) / 1208925819614629174706176,
    # 第六个复数：实部 (52291747908702842344842889809762246649489)，虚部 (-520996778817151392090736149644507525892649)
    (52291747908702842344842889809762246649489 - 520996778817151392090736149644507525892649*I) / 19342813113834066795298816,
    # 第七个复数：实部 (17472406829219127839967951180375981717322)，虚部 (23464704213841582137898905375041819568669)
    (17472406829219127839967951180375981717322 + 23464704213841582137898905375041819568669*I) / 4835703278458516698824704,
    # 第八个复数：实部 (-911026971811893092350229536132730760943307)，虚部 (150799318130900944080399439626714846752360)
    (-911026971811893092350229536132730760943307 + 150799318130900944080399439626714846752360*I) / 38685626227668133590597632,
    # 第九个复数：实部 (26234457233977042811089020440646443590687)，虚部 (-45650293039576452023692126463683727692890)
    (26234457233977042811089020440646443590687 - 45650293039576452023692126463683727692890*I) / 9671406556917033397649408,
    # 第十个复数：实部 3*(288348388717468992528382586652654351121357)，虚部 (-91583492367747094223295011999405657956347)
    3*(288348388717468992528382586652654351121357 + 454526517721403048270274049572136109264668*I) / 77371252455336267181195264,
    # 第十一个复数：实部 (-91583492367747094223295011999405657956347)，虚部 (-12704691128268298435362255538069612411331)
    (-91583492367747094223295011999405657956347 - 12704691128268298435362255538069612411331*I) / 19342813113834066795298816,
    # 第十二个复数：实部 (411208730251327843849027957710164064354221)，虚部 (-569898526380691606955496789378230959965898)
    (411208730251327843849027957710164064354221 - 569898526380691606955496789378230959965898*I) / 38685626227668133590597632
]
# 定义一个复数列表，包含多个复数对象
[
    # 第一个复数对象，实部和虚部分别由长整型计算得出
    (27127513117071487872628354831658811211795 - 37765296987901990355760582016892124833857*I) / 4835703278458516698824704,
    # 第二个复数对象，实部和虚部分别由长整型和整型计算得出
    (1741779916057680444272938534338833170625435 + 3083041729779495966997526404685535449810378*I) / 77371252455336267181195264,
    # 第三个复数对象，实部由长整型和整型计算得出，虚部由长整型和整型计算得出
    3 * (-60642236251815783728374561836962709533401 - 24630301165439580049891518846174101510744*I) / 19342813113834066795298816,
    # 第四个复数对象，实部由长整型和整型计算得出，虚部由长整型和整型计算得出
    3 * (445885207364591681637745678755008757483408 - 350948497734812895032502179455610024541643*I) / 38685626227668133590597632,
    # 第五个复数对象，实部由长整型和整型计算得出，虚部由长整型和整型计算得出
    (-47373295621391195484367368282471381775684 + 219122969294089357477027867028071400054973*I) / 19342813113834066795298816,
    # 第六个复数对象，实部由长整型和整型计算得出，虚部由长整型和整型计算得出
    (-2801565819673198722993348253876353741520438 - 2250142129822658548391697042460298703335701*I) / 77371252455336267181195264,
    # 第七个复数对象，实部由长整型和整型计算得出，虚部由长整型和整型计算得出
    (801448252275607253266997552356128790317119 - 50890367688077858227059515894356594900558*I) / 77371252455336267181195264,
    # 第八个复数对象，实部由长整型和整型计算得出，虚部由长整型和整型计算得出
    (-5082187758525931944557763799137987573501207 + 11610432359082071866576699236013484487676124*I) / 309485009821345068724781056,
    # 第九个复数对象，实部由长整型和整型计算得出，虚部由长整型和整型计算得出
    (-328925127096560623794883760398247685166830 - 643447969697471610060622160899409680422019*I) / 77371252455336267181195264,
    # 第十个复数对象，实部由整型计算得出，虚部由整型计算得出
    15 * (2954944669454003684028194956846659916299765 + 33434406416888505837444969347824812608566*I) / 1237940039285380274899124224,
    # 第十一个复数对象，实部由长整型和整型计算得出，虚部由长整型和整型计算得出
    (-415749104352001509942256567958449835766827 + 479330966144175743357171151440020955412219*I) / 77371252455336267181195264,
    # 第十二个复数对象，实部由长整型和整型计算得出，虚部由长整型和整型计算得出
    3 * (-4639987285852134369449873547637372282914255 - 11994411888966030153196659207284951579243273*I) / 1237940039285380274899124224
]
# 创建一个包含复数的列表，每个复数由实部和虚部组成
[
    # 第一个复数，实部为 -478846096206269117345024348666145495601，虚部为 1249092488629201351470551186322814883283
    (-478846096206269117345024348666145495601 + 1249092488629201351470551186322814883283*I) / 302231454903657293676544,
    # 第二个复数，实部为 -17749319421930878799354766626365926894989，虚部为 -18264580106418628161818752318217357231971
    (-17749319421930878799354766626365926894989 - 18264580106418628161818752318217357231971*I) / 1208925819614629174706176,
    # 第三个复数，实部为 2801110795431528876849623279389579072819，虚部为 363258850073786330770713557775566973248
    (2801110795431528876849623279389579072819 + 363258850073786330770713557775566973248*I) / 604462909807314587353088,
    # 第四个复数，实部为 -59053496693129013745775512127095650616252，虚部为 78143588734197260279248498898321500167517
    (-59053496693129013745775512127095650616252 + 78143588734197260279248498898321500167517*I) / 4835703278458516698824704,
    # 第五个复数，实部为 -283186724922498212468162690097101115349，虚部为 -6443437753863179883794497936345437398276
    (-283186724922498212468162690097101115349 - 6443437753863179883794497936345437398276*I) / 1208925819614629174706176,
    # 第六个复数，实部为 188799118826748909206887165661384998787543，虚部为 84274736720556630026311383931055307398820
    (188799118826748909206887165661384998787543 + 84274736720556630026311383931055307398820*I) / 9671406556917033397649408,
    # 第七个复数，实部为 -5482217151670072904078758141270295025989，虚部为 1818284338672191024475557065444481298568
    (-5482217151670072904078758141270295025989 + 1818284338672191024475557065444481298568*I) / 1208925819614629174706176,
    # 第八个复数，实部为 56564463395350195513805521309731217952281，虚部为 -360208541416798112109946262159695452898431
    (56564463395350195513805521309731217952281 - 360208541416798112109946262159695452898431*I) / 19342813113834066795298816,
    # 第九个复数，实部为 11 * (1259539805728870739006416869463689438068 + 1409136581547898074455004171305324917387*I)
    11 * (1259539805728870739006416869463689438068 + 1409136581547898074455004171305324917387*I) / 4835703278458516698824704,
    # 第十个复数，实部为 5 * (-123701190701414554945251071190688818343325 + 30997157322590424677294553832111902279712*I)
    5 * (-123701190701414554945251071190688818343325 + 30997157322590424677294553832111902279712*I) / 38685626227668133590597632,
    # 第十一个复数，实部为 16130917381301373033736295883982414239781，虚部为 -32752041297570919727145380131926943374516
    (16130917381301373033736295883982414239781 - 32752041297570919727145380131926943374516*I) / 9671406556917033397649408,
    # 第十二个复数，实部为 650301385108223834347093740500375498354925，虚部为 899526407681131828596801223402866051809258
    (650301385108223834347093740500375498354925 + 899526407681131828596801223402866051809258*I) / 77371252455336267181195264
]
# 定义一个测试函数，用于检查 issue 17247 中的表达式膨胀问题
def test_issue_17247_expression_blowup_5():
    # 创建一个 6x6 的复数矩阵 M，每个元素都是 1 + (-1)**(i+j)*I 的形式
    M = Matrix(6, 6, lambda i, j: 1 + (-1)**(i+j)*I)
    
    # 使用 dotprodsimp(True) 上下文
    with dotprodsimp(True):
        # 断言矩阵 M 的特征多项式为 x^6 + (-6 - 6*I)*x^5 + 36*I*x^4，域为 'EX'
        assert M.charpoly('x') == PurePoly(x**6 + (-6 - 6*I)*x**5 + 36*I*x**4, x, domain='EX')


# 定义另一个测试函数，检查 issue 17247 中的表达式膨胀问题
def test_issue_17247_expression_blowup_6():
    # 创建一个 8x8 的矩阵 M，每个元素为 x+i，其中 i 的取值从 0 到 63
    M = Matrix(8, 8, [x+i for i in range(64)])
    
    # 使用 dotprodsimp(True) 上下文
    with dotprodsimp(True):
        # 断言矩阵 M 的 Bareiss 行列式为 0
        assert M.det('bareiss') == 0


# 定义另一个测试函数，检查 issue 17247 中的表达式膨胀问题
def test_issue_17247_expression_blowup_7():
    # 创建一个 6x6 的复数矩阵 M，每个元素都是 1 + (-1)**(i+j)*I 的形式
    M = Matrix(6, 6, lambda i, j: 1 + (-1)**(i+j)*I)
    
    # 使用 dotprodsimp(True) 上下文
    with dotprodsimp(True):
        # 断言矩阵 M 的 Berkowitz 行列式为 0
        assert M.det('berkowitz') == 0


# 定义另一个测试函数，检查 issue 17247 中的表达式膨胀问题
def test_issue_17247_expression_blowup_8():
    # 创建一个 8x8 的矩阵 M，矩阵元素按列主序填充，从 0 到 63
    M = Matrix(8, 8, [x+i for i in range(64)])
    
    # 启用 dotprodsimp 来简化线性代数中的点积操作
    with dotprodsimp(True):
        # 断言：使用 LU 分解计算矩阵 M 的行列式，预期结果为 0
        assert M.det('lu') == 0
def test_issue_17247_expression_blowup_9():
    # 创建一个 8x8 的矩阵 M，其元素为 x+i，其中 i 在范围 (0, 63) 内循环
    M = Matrix(8, 8, [x+i for i in range (64)])
    # 使用 dotprodsimp(True) 上下文，简化乘积项
    with dotprodsimp(True):
        # 断言 M 的行简化阶梯形式等于给定的矩阵
        assert M.rref() == (Matrix([
            [1, 0, -1, -2, -3, -4, -5, -6],
            [0, 1,  2,  3,  4,  5,  6,  7],
            [0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0]]), (0, 1))


def test_issue_17247_expression_blowup_10():
    # 创建一个 6x6 的矩阵 M，其元素为 lambda 函数 1 + (-1)**(i+j)*I
    M = Matrix(6, 6, lambda i, j: 1 + (-1)**(i+j)*I)
    # 使用 dotprodsimp(True) 上下文，简化乘积项
    with dotprodsimp(True):
        # 断言 M 的 (0, 0) 余子式等于 0
        assert M.cofactor(0, 0) == 0


def test_issue_17247_expression_blowup_11():
    # 创建一个 6x6 的矩阵 M，其元素为 lambda 函数 1 + (-1)**(i+j)*I
    M = Matrix(6, 6, lambda i, j: 1 + (-1)**(i+j)*I)
    # 使用 dotprodsimp(True) 上下文，简化乘积项
    with dotprodsimp(True):
        # 断言 M 的余子式矩阵等于 6x6 的零矩阵
        assert M.cofactor_matrix() == Matrix(6, 6, [0]*36)


def test_issue_17247_expression_blowup_12():
    # 创建一个 6x6 的矩阵 M，其元素为 lambda 函数 1 + (-1)**(i+j)*I
    M = Matrix(6, 6, lambda i, j: 1 + (-1)**(i+j)*I)
    # 使用 dotprodsimp(True) 上下文，简化乘积项
    with dotprodsimp(True):
        # 断言 M 的特征值及其代数重数等于给定的字典
        assert M.eigenvals() == {6: 1, 6*I: 1, 0: 4}


def test_issue_17247_expression_blowup_13():
    # 创建一个 4x4 的矩阵 M，其元素为具体值或表达式
    M = Matrix([
        [    0, 1 - x, x + 1, 1 - x],
        [1 - x, x + 1,     0, x + 1],
        [    0, 1 - x, x + 1, 1 - x],
        [    0,     0,     1 - x, 0]])
    # 计算 M 的特征向量
    ev = M.eigenvects()
    # 断言特定特征向量的返回值与预期相等
    assert ev[0] == (0, 2, [Matrix([0, -1, 0, 1])])
    assert ev[1][0] == x - sqrt(2)*(x - 1) + 1
    assert ev[1][1] == 1
    # 断言特定特征向量的数值化表达式与预期相等
    assert ev[1][2][0].expand(deep=False, numer=True) == Matrix([
        [(-x + sqrt(2)*(x - 1) - 1)/(x - 1)],
        [-4*x/(x**2 - 2*x + 1) + (x + 1)*(x - sqrt(2)*(x - 1) + 1)/(x**2 - 2*x + 1)],
        [(-x + sqrt(2)*(x - 1) - 1)/(x - 1)],
        [1]
    ])
    assert ev[2][0] == x + sqrt(2)*(x - 1) + 1
    assert ev[2][1] == 1
    # 断言特定特征向量的数值化表达式与预期相等
    assert ev[2][2][0].expand(deep=False, numer=True) == Matrix([
        [(-x - sqrt(2)*(x - 1) - 1)/(x - 1)],
        [-4*x/(x**2 - 2*x + 1) + (x + 1)*(x + sqrt(2)*(x - 1) + 1)/(x**2 - 2*x + 1)],
        [(-x - sqrt(2)*(x - 1) - 1)/(x - 1)],
        [1]
    ])


def test_issue_17247_expression_blowup_14():
    # 创建一个 8x8 的矩阵 M，其元素为特定列表的循环重复
    M = Matrix(8, 8, ([1+x, 1-x]*4 + [1-x, 1+x]*4)*4)
    # 使用 dotprodsimp(True) 上下文，简化乘积项
    with dotprodsimp(True):
        # 断言 M 的梯形形式等于给定的矩阵
        assert M.echelon_form() == Matrix([
            [x + 1, 1 - x, x + 1, 1 - x, x + 1, 1 - x, x + 1, 1 - x],
            [    0,   4*x,     0,   4*x,     0,   4*x,     0,   4*x],
            [    0,     0,     0,     0,     0,     0,     0,     0],
            [    0,     0,     0,     0,     0,     0,     0,     0],
            [    0,     0,     0,     0,     0,     0,     0,     0],
            [    0,     0,     0,     0,     0,     0,     0,     0],
            [    0,     0,     0,     0,     0,     0,     0,     0],
            [    0,     0,     0,     0,     0,     0,     0,     0]])


def test_issue_17247_expression_blowup_15():
    # 创建一个 8x8 的矩阵 M，其元素为特定列表的循环重复
    M = Matrix(8, 8, ([1+x, 1-x]*4 + [1-x, 1+x]*4)*4)
    # 使用 dotprodsimp(True) 上下文管理器来简化点积
    with dotprodsimp(True):
        # 断言 M 的行空间等于给定的两个矩阵
        assert M.rowspace() == [
            Matrix([[x + 1, 1 - x, x + 1, 1 - x, x + 1, 1 - x, x + 1, 1 - x]]), 
            Matrix([[0, 4*x, 0, 4*x, 0, 4*x, 0, 4*x]])
        ]
def test_issue_17247_expression_blowup_16():
    # 创建一个 8x8 的矩阵 M，使用表达式 [1+x, 1-x]*4 + [1-x, 1+x]*4 的重复结构
    M = Matrix(8, 8, ([1+x, 1-x]*4 + [1-x, 1+x]*4)*4)
    # 使用 dotprodsimp(True) 上下文环境
    with dotprodsimp(True):
        # 断言 M 的列空间与指定的矩阵列表相等
        assert M.columnspace() == [Matrix([[x + 1],[1 - x],[x + 1],[1 - x],[x + 1],[1 - x],[x + 1],[1 - x]]), Matrix([[1 - x],[x + 1],[1 - x],[x + 1],[1 - x],[x + 1],[1 - x],[x + 1]])]


def test_issue_17247_expression_blowup_17():
    # 创建一个 8x8 的矩阵 M，元素为 x+i 的列表表达式
    M = Matrix(8, 8, [x+i for i in range (64)])
    # 使用 dotprodsimp(True) 上下文环境
    with dotprodsimp(True):
        # 断言 M 的零空间与指定的矩阵列表相等
        assert M.nullspace() == [
            Matrix([[1],[-2],[1],[0],[0],[0],[0],[0]]),
            Matrix([[2],[-3],[0],[1],[0],[0],[0],[0]]),
            Matrix([[3],[-4],[0],[0],[1],[0],[0],[0]]),
            Matrix([[4],[-5],[0],[0],[0],[1],[0],[0]]),
            Matrix([[5],[-6],[0],[0],[0],[0],[1],[0]]),
            Matrix([[6],[-7],[0],[0],[0],[0],[0],[1]])]


def test_issue_17247_expression_blowup_18():
    # 创建一个 6x6 的矩阵 M，使用表达式 [1+x, 1-x]*3 + [1-x, 1+x]*3 的重复结构
    M = Matrix(6, 6, ([1+x, 1-x]*3 + [1-x, 1+x]*3)*3)
    # 使用 dotprodsimp(True) 上下文环境
    with dotprodsimp(True):
        # 断言 M 不是幂零矩阵
        assert not M.is_nilpotent()


def test_issue_17247_expression_blowup_19():
    # 创建一个 4x4 的复数矩阵 M，元素为给定的复数表达式
    M = Matrix(S('''[
        [             -3/4,                     0,         1/4 + I/2,                     0],
        [                0, -177/128 - 1369*I/128,                 0, -2063/256 + 541*I/128],
        [          1/2 - I,                     0,                 0,                     0],
        [                0,                     0,                 0, -177/128 - 1369*I/128]]'''))
    # 使用 dotprodsimp(True) 上下文环境
    with dotprodsimp(True):
        # 断言 M 不可对角化
        assert not M.is_diagonalizable()


def test_issue_17247_expression_blowup_20():
    # 创建一个 4x4 的矩阵 M，元素为给定的数学表达式
    M = Matrix([
    [x + 1,  1 - x,      0,      0],
    [1 - x,  x + 1,      0,  x + 1],
    [    0,  1 - x,  x + 1,      0],
    [    0,      0,      0,  x + 1]])
    # 使用 dotprodsimp(True) 上下文环境
    with dotprodsimp(True):
        # 断言 M 的对角化结果与指定的矩阵对角化结果相等
        assert M.diagonalize() == (Matrix([
            [1,  1, 0, (x + 1)/(x - 1)],
            [1, -1, 0,               0],
            [1,  1, 1,               0],
            [0,  0, 0,               1]]),
            Matrix([
            [2,   0,     0,     0],
            [0, 2*x,     0,     0],
            [0,   0, x + 1,     0],
            [0,   0,     0, x + 1]]))


def test_issue_17247_expression_blowup_21():
    # 创建一个 4x4 的复数矩阵 M，元素为给定的复数表达式
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    # 使用 dotprodsimp(True) 上下文环境
    with dotprodsimp(True):
        # 此处没有 assert 语句，因为未指定断言内容
    # 使用 dotprodsimp(True) 上下文管理器，可能是某种数学运算的设置或上下文
    with dotprodsimp(True):
        # 断言矩阵 M 的逆矩阵存在，并使用高斯消元法求解
        assert M.inv(method='GE') == Matrix(S('''[
            # 第一行矩阵元素
            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],
            # 第二行矩阵元素
            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],
            # 第三行矩阵元素
            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],
            # 第四行矩阵元素
            [0, 0, 0, -11328/952745 + 87616*I/952745]]'''))
# 定义一个测试函数，用于测试问题编号为 17247 的表达式扩展是否会导致异常
def test_issue_17247_expression_blowup_22():
    # 创建一个有理数矩阵 M，包含复数
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    # 使用 dotprodsimp 上下文确保表达式被简化
    with dotprodsimp(True):
        # 断言 M 的逆矩阵使用 LU 分解方法与给定的精确矩阵相等
        assert M.inv(method='LU') == Matrix(S('''[
            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],
            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],
            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],
            [0, 0, 0, -11328/952745 + 87616*I/952745]]'''))


# 定义另一个测试函数，用于测试问题编号为 17247 的表达式扩展是否会导致异常
def test_issue_17247_expression_blowup_23():
    # 创建一个有理数矩阵 M，包含复数
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    # 使用 dotprodsimp 上下文确保表达式被简化
    with dotprodsimp(True):
        # 断言 M 的逆矩阵使用 ADJ 方法并展开后与给定的精确矩阵相等
        assert M.inv(method='ADJ').expand() == Matrix(S('''[
            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],
            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],
            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],
            [0, 0, 0, -11328/952745 + 87616*I/952745]]'''))


# 定义另一个测试函数，用于测试问题编号为 17247 的表达式扩展是否会导致异常
def test_issue_17247_expression_blowup_24():
    # 创建一个稀疏矩阵 M，包含复数
    M = SparseMatrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    # 使用 dotprodsimp(True) 来设置 dot product 简化标志
    with dotprodsimp(True):
        # 断言矩阵 M 的逆矩阵，使用 CH 方法计算，并与给定的矩阵进行比较
        assert M.inv(method='CH') == Matrix(S('''[
            # 第一行矩阵元素
            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],
            # 第二行矩阵元素
            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],
            # 第三行矩阵元素
            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],
            # 第四行矩阵元素
            [0, 0, 0, -11328/952745 + 87616*I/952745]]'''))
# 定义测试函数，用于验证问题 17247 中表达式扩展的情况
def test_issue_17247_expression_blowup_25():
    # 创建稀疏矩阵 M，从 SymPy 字符串转换为矩阵对象
    M = SparseMatrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    # 开启 dotprodsimp（点积简化）模式
    with dotprodsimp(True):
        # 断言 M 的逆矩阵等于给定的 SymPy 矩阵
        assert M.inv(method='LDL') == Matrix(S('''[
            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],
            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],
            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],
            [0, 0, 0, -11328/952745 + 87616*I/952745]]'''))


# 定义另一个测试函数，继续验证问题 17247 中的表达式扩展
def test_issue_17247_expression_blowup_26():
    # 创建矩阵 M，从 SymPy 字符串转换为矩阵对象
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,      1/4 - 5*I/16,      65/128 + 87*I/64,         -9/32 - I/16,      183/256 - 97*I/128],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,  85/256 - 33*I/16,  805/128 + 2415*I/512, -219/128 + 115*I/256, 6301/4096 - 6609*I/1024],
        [          1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,         1/4 - 5*I/16,        65/128 + 87*I/64],
        [   -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,     85/256 - 33*I/16,    805/128 + 2415*I/512],
        [            1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,            1/4 + I/2,        -129/64 - 9*I/64],
        [         21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,     125/64 + 87*I/64,   -2063/256 + 541*I/128],
        [               -2,         17/4 - 13*I/2,             1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,                 -3/4,         45/32 - 37*I/16],
        [     1/4 + 13*I/4,    -825/64 - 147*I/32,          21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64,    -149/64 + 49*I/32,   -177/128 - 1369*I/128]]'''))
    # 开启 dotprodsimp（点积简化）模式
    with dotprodsimp(True):
        # 断言 M 的秩等于 4
        assert M.rank() == 4


# 定义另一个测试函数，继续验证问题 17247 中的表达式扩展
def test_issue_17247_expression_blowup_27():
    # 创建矩阵 M，该矩阵直接以列表形式提供
    M = Matrix([
        [    0, 1 - x, x + 1, 1 - x],
        [1 - x, x + 1,     0, x + 1],
        [    0, 1 - x, x + 1, 1 - x],
        [    0,     0,     1 - x, 0]])
    # 使用 dotprodsimp 函数来简化矩阵运算表达式
    with dotprodsimp(True):
        # 计算矩阵 M 的乔丹标准形 P 和相伴矩阵 J
        P, J = M.jordan_form()
        # 断言 P 扩展后的结果等于给定的矩阵表达式
        assert P.expand() == Matrix(S('''[
            [    0,  4*x/(x**2 - 2*x + 1), -(-17*x**4 + 12*sqrt(2)*x**4 - 4*sqrt(2)*x**3 + 6*x**3 - 6*x - 4*sqrt(2)*x + 12*sqrt(2) + 17)/(-7*x**4 + 5*sqrt(2)*x**4 - 6*sqrt(2)*x**3 + 8*x**3 - 2*x**2 + 8*x + 6*sqrt(2)*x - 5*sqrt(2) - 7), -(12*sqrt(2)*x**4 + 17*x**4 - 6*x**3 - 4*sqrt(2)*x**3 - 4*sqrt(2)*x + 6*x - 17 + 12*sqrt(2))/(7*x**4 + 5*sqrt(2)*x**4 - 6*sqrt(2)*x**3 - 8*x**3 + 2*x**2 - 8*x + 6*sqrt(2)*x - 5*sqrt(2) + 7)],
            [x - 1, x/(x - 1) + 1/(x - 1),                       (-7*x**3 + 5*sqrt(2)*x**3 - x**2 + sqrt(2)*x**2 - sqrt(2)*x - x - 5*sqrt(2) - 7)/(-3*x**3 + 2*sqrt(2)*x**3 - 2*sqrt(2)*x**2 + 3*x**2 + 2*sqrt(2)*x + 3*x - 3 - 2*sqrt(2)),                       (7*x**3 + 5*sqrt(2)*x**3 + x**2 + sqrt(2)*x**2 - sqrt(2)*x + x - 5*sqrt(2) + 7)/(2*sqrt(2)*x**3 + 3*x**3 - 3*x**2 - 2*sqrt(2)*x**2 - 3*x + 2*sqrt(2)*x - 2*sqrt(2) + 3)],
            [    0,                     1,                                                                                            -(-3*x**2 + 2*sqrt(2)*x**2 + 2*x - 3 - 2*sqrt(2))/(-x**2 + sqrt(2)*x**2 - 2*sqrt(2)*x + 1 + sqrt(2)),                                                                                            -(2*sqrt(2)*x**2 + 3*x**2 - 2*x - 2*sqrt(2) + 3)/(x**2 + sqrt(2)*x**2 - 2*sqrt(2)*x - 1 + sqrt(2))],
            [1 - x,                     0,                                                                                                                                                                                               1,                                                                                                                                                                                             1]]''')).expand()
        # 断言 J 等于给定的乔丹标准形矩阵表达式
        assert J == Matrix(S('''[
            [0, 1,                       0,                       0],
            [0, 0,                       0,                       0],
            [0, 0, x - sqrt(2)*(x - 1) + 1,                       0],
            [0, 0,                       0, x + sqrt(2)*(x - 1) + 1]]'''))
# 定义一个测试函数，用于检测问题 17247 中表达式膨胀的情况
def test_issue_17247_expression_blowup_28():
    # 创建一个复数矩阵 M，矩阵元素使用符号表达式定义
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))

# 定义一个测试函数，用于检测问题 16823 的情况
def test_issue_16823():
    # 这部分仍然需要修复，如果不使用 dotprodsimp 的话。
    # 导入符号计算模块 S 和 Matrix 类
    M = Matrix(S('''[
        [1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I,-9/32-1/16*I,183/256-97/128*I,3/64+13/64*I,-23/32-59/256*I,15/128-3/32*I,19/256+551/1024*I],
        [21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I,-219/128+115/256*I,6301/4096-6609/1024*I,119/128+143/128*I,-10879/2048+4343/4096*I,129/256-549/512*I,42533/16384+29103/8192*I],
        [-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I,-9/32-1/16*I,183/256-97/128*I,3/64+13/64*I,-23/32-59/256*I],
        [1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I,-219/128+115/256*I,6301/4096-6609/1024*I,119/128+143/128*I,-10879/2048+4343/4096*I],
        [-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I,-9/32-1/16*I,183/256-97/128*I],
        [1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I,-219/128+115/256*I,6301/4096-6609/1024*I],
        [-4,9-5*I,-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I],
        [-2*I,119/8+29/4*I,1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I],
        [0,-6,-4,9-5*I,-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I],
        [1,-9/4+3*I,-2*I,119/8+29/4*I,1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I],
        [0,-4*I,0,-6,-4,9-5*I,-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I],
        [0,1/4+1/2*I,1,-9/4+3*I,-2*I,119/8+29/4*I,1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I]]'''))
    
    # 打开 dotprodsimp 符号化简功能上下文
    with dotprodsimp(True):
        # 断言矩阵 M 的秩等于 8
        assert M.rank() == 8
def test_creation():
    # 测试Matrix类的各种创建和操作方法

    # 测试Matrix初始化时的异常处理：应当引发ValueError
    raises(ValueError, lambda: Matrix(5, 5, range(20)))
    # 测试Matrix初始化时的异常处理：应当引发ValueError
    raises(ValueError, lambda: Matrix(5, -1, []))
    # 测试Matrix索引操作时的异常处理：应当引发IndexError
    raises(IndexError, lambda: Matrix((1, 2))[2])
    # 使用with语句检查Matrix索引操作时的异常处理：应当引发IndexError
    with raises(IndexError):
        Matrix((1, 2))[3] = 5

    # 断言空Matrix对象相等的情况：Matrix() == Matrix([]) == Matrix([[]]) == Matrix(0, 0, [])
    assert Matrix() == Matrix([]) == Matrix([[]]) == Matrix(0, 0, [])

    # 使用过时警告检查Matrix允许创建的数据结构
    with warns_deprecated_sympy():
        assert Matrix([[[1], (2,)]]).tolist() == [[[1], (2,)]]
    with warns_deprecated_sympy():
        assert Matrix([[[1], (2,)]]).T.tolist() == [[[1]], [(2,)]]

    # 使用过时警告检查Matrix中空集的赋值
    M = Matrix([[0]])
    with warns_deprecated_sympy():
        M[0, 0] = S.EmptySet

    # 测试Matrix对象的基本操作
    a = Matrix([[x, 0], [0, 0]])
    m = a
    assert m.cols == m.rows  # 断言Matrix对象的列数与行数相等
    assert m.cols == 2  # 断言Matrix对象的列数为2
    assert m[:] == [x, 0, 0, 0]  # 断言Matrix对象的扁平化列表等于[x, 0, 0, 0]

    # 测试Matrix对象的创建和操作
    b = Matrix(2, 2, [x, 0, 0, 0])
    m = b
    assert m.cols == m.rows  # 断言Matrix对象的列数与行数相等
    assert m.cols == 2  # 断言Matrix对象的列数为2
    assert m[:] == [x, 0, 0, 0]  # 断言Matrix对象的扁平化列表等于[x, 0, 0, 0]

    # 断言两个Matrix对象相等
    assert a == b

    # 断言Matrix对象的复制操作
    assert Matrix(b) == b

    # 测试Matrix对象的连接操作
    c23 = Matrix(2, 3, range(1, 7))
    c13 = Matrix(1, 3, range(7, 10))
    c = Matrix([c23, c13])
    assert c.cols == 3  # 断言Matrix对象的列数为3
    assert c.rows == 3  # 断言Matrix对象的行数为3
    assert c[:] == [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 断言Matrix对象的扁平化列表等于[1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 断言单位矩阵对象与Matrix对象相等
    assert Matrix(eye(2)) == eye(2)
    # 断言ImmutableMatrix对象与ImmutableMatrix对象相等
    assert ImmutableMatrix(ImmutableMatrix(eye(2))) == ImmutableMatrix(eye(2))
    # 断言ImmutableMatrix对象与Matrix对象相等
    assert ImmutableMatrix(c) == c.as_immutable()
    # 断言Matrix对象与ImmutableMatrix对象相等
    assert Matrix(ImmutableMatrix(c)) == ImmutableMatrix(c).as_mutable()

    # 断言Matrix对象与其复制对象不相等
    assert c is not Matrix(c)

    # 测试Matrix对象从数据列表创建并进行比较
    dat = [[ones(3,2), ones(3,3)*2], [ones(2,3)*3, ones(2,2)*4]]
    M = Matrix(dat)
    assert M == Matrix([
        [1, 1, 2, 2, 2],
        [1, 1, 2, 2, 2],
        [1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4],
        [3, 3, 3, 4, 4]])
    assert M.tolist() != dat  # 断言Matrix对象转换为列表后不等于原始数据列表
    # 使用evaluate=False时保持矩阵块形式
    assert Matrix(dat, evaluate=False).tolist() == dat
    # 创建一个符号矩阵变量A，维度为2x2
    A = MatrixSymbol("A", 2, 2)
    
    # 定义一个包含两个元素的列表dat，其中第一个元素是一个2x2的全1矩阵，第二个元素是符号矩阵A
    dat = [ones(2), A]
    
    # 断言语句，验证将dat转换为矩阵后是否与指定的矩阵相同
    assert Matrix(dat) == Matrix([
        [      1,       1],
        [      1,       1],
        [A[0, 0], A[0, 1]],
        [A[1, 0], A[1, 1]]
    ])
    
    # 使用warns_deprecated_sympy()上下文管理器捕获Sympy中的过时警告
    with warns_deprecated_sympy():
        # 断言语句，验证将dat转换为矩阵但不进行评估时是否与指定的列表相同
        assert Matrix(dat, evaluate=False).tolist() == [[i] for i in dat]
    
    # 0维容差测试
    # 断言语句，验证包含2x2全1矩阵和一个空矩阵的列表转换为矩阵后是否与只包含2x2全1矩阵的矩阵相同
    assert Matrix([ones(2), ones(0)]) == Matrix([ones(2)])
    
    # 使用lambda函数捕获异常，验证包含2x2全1矩阵和一个非法形状的空矩阵的列表是否会引发值错误异常
    raises(ValueError, lambda: Matrix([ones(2), ones(0, 3)]))
    raises(ValueError, lambda: Matrix([ones(2), ones(3, 0)]))
    
    # Matrix和可迭代对象的混合使用
    # 创建一个2x2的矩阵M
    M = Matrix([[1, 2], [3, 4]])
    
    # 创建一个包含Matrix对象和元组(5, 6)的列表，将它们转换为矩阵后验证是否与指定的矩阵相同
    M2 = Matrix([M, (5, 6)])
    assert M2 == Matrix([[1, 2], [3, 4], [5, 6]])
# 定义一个测试函数，用于测试 Matrix 类的 irregular 方法
def test_irregular_block():
    # 断言调用 Matrix 类的 irregular 方法，检查其返回结果是否符合预期
    assert Matrix.irregular(3, ones(2,1), ones(3,3)*2, ones(2,2)*3,
        ones(1,1)*4, ones(2,2)*5, ones(1,2)*6, ones(1,2)*7) == Matrix([
        [1, 2, 2, 2, 3, 3],
        [1, 2, 2, 2, 3, 3],
        [4, 2, 2, 2, 5, 5],
        [6, 6, 7, 7, 5, 5]])

# 定义一个测试函数，用于测试 Matrix 类的 slicing 功能
def test_slicing():
    # 创建一个单位矩阵 m0
    m0 = eye(4)
    # 断言对 m0 进行切片操作是否等于一个 3x3 的单位矩阵
    assert m0[:3, :3] == eye(3)
    # 断言对 m0 进行切片操作是否等于一个 2x2 的零矩阵
    assert m0[2:4, 0:2] == zeros(2)

    # 创建一个 3x3 的矩阵 m1，元素由 lambda 函数 i + j 生成
    m1 = Matrix(3, 3, lambda i, j: i + j)
    # 断言对 m1 的行切片操作是否等于一个 1x3 的矩阵，元素为 (0, 1, 2)
    assert m1[0, :] == Matrix(1, 3, (0, 1, 2))
    # 断言对 m1 的列切片操作是否等于一个 2x1 的矩阵，元素为 (2, 3)
    assert m1[1:3, 1] == Matrix(2, 1, (2, 3))

    # 创建一个 4x4 的矩阵 m2
    m2 = Matrix([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    # 断言对 m2 的列切片操作是否等于一个 4x1 的矩阵，元素为 [3, 7, 11, 15]
    assert m2[:, -1] == Matrix(4, 1, [3, 7, 11, 15])
    # 断言对 m2 的行切片操作是否等于一个 2x4 的矩阵
    assert m2[-2:, :] == Matrix([[8, 9, 10, 11], [12, 13, 14, 15]])

# 定义一个测试函数，用于测试 Matrix 类的子矩阵赋值功能
def test_submatrix_assignment():
    # 创建一个 4x4 的零矩阵 m
    m = zeros(4)
    # 将 m 的右下角 2x2 子矩阵赋值为单位矩阵
    m[2:4, 2:4] = eye(2)
    # 断言 m 是否等于一个特定的 4x4 矩阵
    assert m == Matrix(((0, 0, 0, 0),
                        (0, 0, 0, 0),
                        (0, 0, 1, 0),
                        (0, 0, 0, 1)))
    # 将 m 的左上角 2x2 子矩阵赋值为单位矩阵
    m[:2, :2] = eye(2)
    # 断言 m 是否等于一个 4x4 的单位矩阵
    assert m == eye(4)
    # 将 m 的第一列赋值为一个 4x1 的矩阵
    m[:, 0] = Matrix(4, 1, (1, 2, 3, 4))
    # 断言 m 是否等于一个特定的 4x4 矩阵
    assert m == Matrix(((1, 0, 0, 0),
                        (2, 1, 0, 0),
                        (3, 0, 1, 0),
                        (4, 0, 0, 1)))
    # 将 m 所有元素赋值为零矩阵
    m[:, :] = zeros(4)
    # 断言 m 是否等于一个全零的 4x4 矩阵
    assert m == zeros(4)
    # 将 m 的所有元素赋值为一个特定的列表
    m[:, :] = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)]
    # 断言 m 是否等于一个特定的 4x4 矩阵
    assert m == Matrix(((1, 2, 3, 4),
                        (5, 6, 7, 8),
                        (9, 10, 11, 12),
                        (13, 14, 15, 16)))
    # 将 m 的前两行第一列赋值为 [0, 0]
    m[:2, 0] = [0, 0]
    # 断言 m 是否等于一个特定的 4x4 矩阵
    assert m == Matrix(((0, 2, 3, 4),
                        (0, 6, 7, 8),
                        (9, 10, 11, 12),
                        (13, 14, 15, 16)))

# 定义一个测试函数，用于测试 Matrix 类的 reshape 方法
def test_reshape():
    # 创建一个 3x3 的单位矩阵 m0
    m0 = eye(3)
    # 断言对 m0 进行 reshape 操作是否得到特定的 1x9 矩阵
    assert m0.reshape(1, 9) == Matrix(1, 9, (1, 0, 0, 0, 1, 0, 0, 0, 1))
    
    # 创建一个 3x4 的矩阵 m1，元素由 lambda 函数 i + j 生成
    m1 = Matrix(3, 4, lambda i, j: i + j)
    # 断言对 m1 进行 reshape 操作是否得到特定的 4x3 矩阵
    assert m1.reshape(4, 3) == Matrix(((0, 1, 2), (3, 1, 2), (3, 4, 2), (3, 4, 5)))
    # 断言对 m1 进行 reshape 操作是否得到特定的 2x6 矩阵
    assert m1.reshape(2, 6) == Matrix(((0, 1, 2, 3, 1, 2), (3, 4, 2, 3, 4, 5)))

# 定义一个测试函数，用于测试 Matrix 类的 applyfunc 方法
def test_applyfunc():
    # 创建一个 3x3 的单位矩阵 m0
    m0 = eye(3)
    # 断言对 m0 应用 lambda 函数 2*x 是否得到特定的 3x3 矩阵
    assert m0.applyfunc(lambda x: 2*x) == eye(3)*2
    # 断言对 m0 应用 lambda 函数 0 是否得到全零的 3x3 矩阵
    assert m0.applyfunc(lambda x: 0) == zeros(3)

# 定义一个测试函数，用于测试 Matrix 类的 expand 方法
def test_expand():
    # 创建一个 2x2 的矩阵 m0，元素是表达式
    m0 = Matrix([[x*(x + y), 2], [((x + y)*y)*x, x*(y + x*(x + y))]])
    # 调用 expand() 方法，生成一个新的矩阵 m1
    m1 = m0.expand()
    # 断言 m1 是否等于一个特定的 2x2
    # 对矩阵 m0 执行筛选操作，选择所有元素 x 和 y 均为负数的部分
    m1 = m0.refine(Q.negative(x) & Q.negative(y))
    # 断言：验证 m1 是否等于给定的矩阵
    assert m1 == Matrix([[x**2, -x], [-x*y**2, -x**2*y]])
# 定义一个测试函数，用于测试随机生成的矩阵和矩阵的逆操作
def test_random():
    # 生成一个 3x3 的随机矩阵 M
    M = randMatrix(3, 3)
    # 用种子值为 3 生成一个 3x3 的随机矩阵 M
    M = randMatrix(3, 3, seed=3)
    # 断言前两个生成的矩阵相等
    assert M == randMatrix(3, 3, seed=3)

    # 生成一个 3x4 的数值范围在 0 到 150 之间的随机矩阵 M
    M = randMatrix(3, 4, 0, 150)
    # 使用种子值为 4 生成一个对称的 3x3 随机矩阵 M
    M = randMatrix(3, seed=4, symmetric=True)
    # 断言生成的对称矩阵 M 符合预期
    assert M == randMatrix(3, seed=4, symmetric=True)

    # 复制矩阵 M 到 S
    S = M.copy()
    # 对 S 进行简化（假设该方法存在）
    S.simplify()
    # 断言简化后的 S 与 M 相等，即简化操作没有改变矩阵元素
    assert S == M  # doesn't fail when elements are Numbers, not int

    # 使用种子值为 4 创建一个随机数生成器 rng
    rng = random.Random(4)
    # 断言生成的对称矩阵 M 符合预期，使用指定的随机数生成器 rng
    assert M == randMatrix(3, symmetric=True, prng=rng)

    # 确保生成的对称矩阵 M 是对称的
    for size in (10, 11):  # 测试奇数和偶数大小的情况
        for percent in (100, 70, 30):
            # 生成一个大小为 size 的对称矩阵 M，指定非零元素的百分比 percent，使用随机数生成器 rng
            M = randMatrix(size, symmetric=True, percent=percent, prng=rng)
            # 断言 M 等于其转置矩阵 M.T，即 M 是对称的
            assert M == M.T

    # 生成一个大小为 10x10，非零元素占比为 70% 的随机矩阵 M
    M = randMatrix(10, min=1, percent=70)
    zero_count = 0
    # 计算 M 中零元素的数量
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == 0:
                zero_count += 1
    # 断言零元素的数量为 30
    assert zero_count == 30


# 定义一个测试函数，用于测试矩阵的逆操作
def test_inverse():
    # 创建一个单位矩阵 A，大小为 4x4
    A = eye(4)
    # 断言 A 的逆矩阵等于单位矩阵
    assert A.inv() == eye(4)
    # 使用 LU 方法计算 A 的逆矩阵，断言结果等于单位矩阵
    assert A.inv(method="LU") == eye(4)
    # 使用 ADJ 方法计算 A 的逆矩阵，断言结果等于单位矩阵
    assert A.inv(method="ADJ") == eye(4)
    # 使用 CH 方法计算 A 的逆矩阵，断言结果等于单位矩阵
    assert A.inv(method="CH") == eye(4)
    # 使用 LDL 方法计算 A 的逆矩阵，断言结果等于单位矩阵
    assert A.inv(method="LDL") == eye(4)
    # 使用 QR 方法计算 A 的逆矩阵，断言结果等于单位矩阵
    assert A.inv(method="QR") == eye(4)

    # 创建一个 3x3 的矩阵 A
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    # 计算 A 的逆矩阵 Ainv
    Ainv = A.inv()
    # 断言 A 乘以其逆矩阵 Ainv 等于单位矩阵
    assert A * Ainv == eye(3)
    # 使用 LU 方法计算 A 的逆矩阵，断言结果等于预期的 Ainv
    assert A.inv(method="LU") == Ainv
    # 使用 ADJ 方法计算 A 的逆矩阵，断言结果等于预期的 Ainv
    assert A.inv(method="ADJ") == Ainv
    # 使用 CH 方法计算 A 的逆矩阵，断言结果等于预期的 Ainv
    assert A.inv(method="CH") == Ainv
    # 使用 LDL 方法计算 A 的逆矩阵，断言结果等于预期的 Ainv
    assert A.inv(method="LDL") == Ainv
    # 使用 QR 方法计算 A 的逆矩阵，断言结果等于预期的 Ainv
    assert A.inv(method="QR") == Ainv
    # 创建一个名为 AA 的矩阵对象，包含 25 行 25 列的整数矩阵
    AA = Matrix([[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0]])
    # 断言：使用 "BLOCK" 方法计算 AA 的逆矩阵乘以 AA 应得到单位矩阵
    assert AA.inv(method="BLOCK") * AA == eye(AA.shape[0])
    
    # 测试不可变性是否会成为问题
    # 设置 cls 变量为 ImmutableMatrix 类
    cls = ImmutableMatrix
    # 创建一个 3x3 的 ImmutableMatrix 对象 m
    m = cls([[48, 49, 31],
# 定义一个测试函数 test_jacobian_hessian，用于测试雅可比矩阵和海森矩阵的计算
def test_jacobian_hessian():
    # 创建一个 1x2 的矩阵 L，包含表达式 x**2*y 和 2*y**2 + x*y
    L = Matrix(1, 2, [x**2*y, 2*y**2 + x*y])
    syms = [x, y]
    # 断言 L 对于变量 syms 的雅可比矩阵应为 Matrix([[2*x*y, x**2], [y, 4*y + x]])
    assert L.jacobian(syms) == Matrix([[2*x*y, x**2], [y, 4*y + x]])

    # 更新 L 为包含表达式 x 和 x**2*y**3 的 1x2 矩阵
    L = Matrix(1, 2, [x, x**2*y**3])
    # 断言 L 对于变量 syms 的雅可比矩阵应为 Matrix([[1, 0], [2*x*y**3, x**2*3*y**2]])
    assert L.jacobian(syms) == Matrix([[1, 0], [2*x*y**3, x**2*3*y**2]])

    # 定义一个函数 f = x**2*y
    f = x**2*y
    syms = [x, y]
    # 断言 f 对于变量 syms 的海森矩阵应为 Matrix([[2*y, 2*x], [2*x, 0]])
    assert hessian(f, syms) == Matrix([[2*y, 2*x], [2*x, 0]])

    # 更新 f 为 x**2*y**3
    f = x**2*y**3
    # 断言 f 对于变量 syms 的海森矩阵应为 Matrix([[2*y**3, 6*x*y**2], [6*x*y**2, 6*x**2*y]])
    assert hessian(f, syms) == Matrix([[2*y**3, 6*x*y**2], [6*x*y**2, 6*x**2*y]])

    # 定义 f = z + x*y**2 和 g = x**2 + 2*y**3
    f = z + x*y**2
    g = x**2 + 2*y**3
    # 定义预期结果矩阵 ans
    ans = Matrix([[0,   2*y],
                  [2*y, 2*x]])
    # 断言 f 对于 Matrix([x, y]) 的海森矩阵应为 ans
    assert ans == hessian(f, Matrix([x, y]))
    # 断言 f 对于 Matrix([x, y]).T 的海森矩阵应为 ans
    assert ans == hessian(f, Matrix([x, y]).T)
    # 断言 f 对于 (y, x) 的海森矩阵，带有 g 作为附加参数，应为以下 Matrix
    assert hessian(f, (y, x), [g]) == Matrix([
        [     0, 6*y**2, 2*x],
        [6*y**2,    2*x, 2*y],
        [   2*x,    2*y,   0]])

# 定义一个测试函数 test_wronskian，用于测试 Wronskian 行列式的计算
def test_wronskian():
    # 断言 Wronskian 行列式计算对于函数列表 [cos(x), sin(x)] 关于 x 的结果应为 cos(x)**2 + sin(x)**2
    assert wronskian([cos(x), sin(x)], x) == cos(x)**2 + sin(x)**2
    # 断言 Wronskian 行列式计算对于函数列表 [exp(x), exp(2*x)] 关于 x 的结果应为 exp(3*x)
    assert wronskian([exp(x), exp(2*x)], x) == exp(3*x)
    # 断言 Wronskian 行列式计算对于函数列表 [exp(x), x] 关于 x 的结果应为 exp(x) - x*exp(x)
    assert wronskian([exp(x), x], x) == exp(x) - x*exp(x)
    # 断言 Wronskian 行列式计算对于函数列表 [1, x, x**2] 关于 x 的结果应为 2
    assert wronskian([1, x, x**2], x) == 2
    # 定义预期的长表达式 w1
    w1 = -6*exp(x)*sin(x)*x + 6*cos(x)*exp(x)*x**2 - 6*exp(x)*cos(x)*x - \
        exp(x)*cos(x)*x**3 + exp(x)*sin(x)*x**3
    # 断言 Wronskian 行列式计算对于函数列表 [exp(x), cos(x), x**3] 关于 x 的结果应为 w1（展开后）
    assert wronskian([exp(x), cos(x), x**3], x).expand() == w1
    # 断言 Wronskian 行列式计算对于函数列表 [exp(x), cos(x), x**3] 关于 x 使用 'berkowitz' 方法的结果应为 w1（展开后）
    assert wronskian([exp(x), cos(x), x**3], x, method='berkowitz').expand() \
        == w1
    # 定义预期的长表达式 w2
    w2 = -x**3*cos(x)**2 - x**3*sin(x)**2 - 6*x*cos(x)**2 - 6*x*sin(x)**2
    # 断言 Wronskian 行列式计算对于函数列表 [sin(x), cos(x), x**3] 关于 x 的结果应为 w2（展开后）
    assert wronskian([sin(x), cos(x), x**3], x).expand() == w2
    # 断言 Wronskian 行列式计算对于函数列表 [sin(x), cos(x), x**3] 关于 x 使用 'berkowitz' 方法的结果应为 w2（展开后）
    assert wronskian([sin(x), cos(x), x**3], x, method='berkowitz').expand() \
        == w2
    # 断言空函数列表的 Wronskian 行列式计算结果为 1
    assert wronskian([], x) == 1

# 定义一个测试函数 test_xreplace，用于测试矩阵的元素替换
def test_xreplace():
    # 断言 Matrix([[1, x], [x, 4]]）对 x 进行替换为 5 后的结果应为 Matrix([[1, 5], [5, 4]])
    assert Matrix([[1, x], [x, 4]]).xreplace({x: 5}) == \
        Matrix([[1, 5], [5, 4]])
    # 断言 Matrix([[x, 2], [x + y, 4]]）对 x 替换为 -1，y 替换为 -2 后的结果应为 Matrix([[-1, 2], [-3, 4]])
    assert Matrix([[x, 2], [x + y, 4]]).xreplace({x: -1, y: -2}) == \
        Matrix([[-1, 2], [-3, 4]])
    # 对所有的矩阵类进行遍历，断言单位矩阵的元素替换为 2 后结果为正确的单位矩阵
    for cls in all_classes:
        assert Matrix([[2, 0], [0, 2]]) == cls.eye(2).xreplace({1: 2})

# 定义一个测试函数 test_simplify，用于测试矩阵的简化操作
def test_simplify():
    # 定义符号 n 和函数 f
    n = Symbol('n')
    f = Function('f')

    # 创建一个 2x2 的矩阵 M
    M = Matrix([[            1/x + 1/y,                 (x + x*y) / x  ],
                [ (f(x) + y*f(x))/f(x), 2 * (1/n - cos(n * pi)/n) / pi ]])
    # 简化矩阵 M 自身
    M.simplify()
    # 断言简化后的矩阵 M 应为指定的简化结果
    assert M == Matrix([[ (x + y)/(x
    # 使用 SymPy 中的 M 对象，调用 simplify 方法，传入 ratio=oo 参数进行化简，并断言其结果等于一个包含化简后等式的矩阵
    assert M.simplify(ratio=oo) == Matrix([[eq.simplify(ratio=oo)]])

    # 使用 SymPy 中的 simplify 函数，对包含 sin(x)**2 + cos(x)**2 的不可变矩阵进行化简，并断言其结果等于一个包含数字 1 的不可变矩阵
    assert simplify(ImmutableMatrix([[sin(x)**2 + cos(x)**2]])) == \
                    ImmutableMatrix([[1]])

    # 链接到 GitHub 上的一个问题页面，关于 SymPy 中的一个问题，创建一个 2x2 的整数矩阵 m
    m = Matrix([[30, 2], [3, 4]])
    # 断言 m 的迹的倒数经过化简后等于分数 Rational(1, 34)
    assert (1/(m.trace())).simplify() == Rational(1, 34)
# 定义一个测试函数，用于测试矩阵类的转置操作
def test_transpose():
    # 创建一个两行十列的矩阵对象 M
    M = Matrix([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
    # 断言 M 的转置应与给定的二维列表形式相等
    assert M.T == Matrix( [ [1, 1],
                            [2, 2],
                            [3, 3],
                            [4, 4],
                            [5, 5],
                            [6, 6],
                            [7, 7],
                            [8, 8],
                            [9, 9],
                            [0, 0] ])
    # 断言 M 的转置的转置应当等于原矩阵 M
    assert M.T.T == M
    # 断言 M 的转置应等同于调用矩阵的 transpose 方法得到的结果
    assert M.T == M.transpose()


# 定义一个测试函数，用于测试共轭转置操作
def test_conj_dirac():
    # 断言创建一个 3x3 的单位矩阵并尝试获取其属性 'D' 会引发 AttributeError
    raises(AttributeError, lambda: eye(3).D)

    # 创建一个复数元素构成的 4x4 矩阵 M
    M = Matrix([[1, I, I, I],
                [0, 1, I, I],
                [0, 0, 1, I],
                [0, 0, 0, 1]])

    # 断言对矩阵 M 执行 Dirac 共轭操作后得到的结果
    assert M.D == Matrix([[ 1,  0,  0,  0],
                          [-I,  1,  0,  0],
                          [-I, -I, -1,  0],
                          [-I, -I,  I, -1]])


# 定义一个测试函数，用于测试矩阵的迹运算
def test_trace():
    # 创建一个 3x3 的矩阵 M
    M = Matrix([[1, 0, 0],
                [0, 5, 0],
                [0, 0, 8]])
    # 断言矩阵 M 的迹应等于 14
    assert M.trace() == 14


# 定义一个测试函数，用于测试矩阵的形状属性
def test_shape():
    # 创建一个形状为 (1, 2) 的行向量 m
    m = Matrix(1, 2, [0, 0])
    # 断言 m 的形状应为 (1, 2)
    assert m.shape == (1, 2)
    
    # 创建一个 2x3 的矩阵 M，其中元素由符号 x 和 y 组成
    M = Matrix([[x, 0, 0],
                [0, y, 0]])
    # 断言矩阵 M 的形状应为 (2, 3)
    assert M.shape == (2, 3)


# 定义一个测试函数，用于测试矩阵的行操作和列操作
def test_col_row_op():
    # 创建一个 2x3 的矩阵 M，其中元素由符号 x 和 y 组成
    M = Matrix([[x, 0, 0],
                [0, y, 0]])
    
    # 对 M 执行行操作：第 1 行的每个元素加上其列索引加 1
    M.row_op(1, lambda r, j: r + j + 1)
    # 断言 M 应等于经过操作后的矩阵
    assert M == Matrix([[x,     0, 0],
                        [1, y + 2, 3]])

    # 对 M 执行列操作：第 0 列的每个元素加上 y 的指数幂
    M.col_op(0, lambda c, j: c + y**j)
    # 断言 M 应等于经过操作后的矩阵
    assert M == Matrix([[x + 1,     0, 0],
                        [1 + y, y + 2, 3]])

    # 对矩阵 M 执行一系列行和列的操作，验证操作后原矩阵未被修改
    assert M.row(0) == Matrix([[x + 1, 0, 0]])
    r1 = M.row(0)
    r1[0] = 42
    assert M[0, 0] == x + 1
    r1 = M[0, :-1]  # also testing negative slice
    r1[0] = 42
    assert M[0, 0] == x + 1
    c1 = M.col(0)
    assert c1 == Matrix([x + 1, 1 + y])
    c1[0] = 0
    assert M[0, 0] == x + 1
    c1 = M[:, 0]
    c1[0] = 42
    assert M[0, 0] == x + 1


# 定义一个测试函数，用于测试矩阵的行乘法操作
def test_row_mult():
    # 创建一个 2x3 的矩阵 M
    M = Matrix([[1,2,3],
               [4,5,6]])
    # 对 M 执行行乘法操作：第 1 行的每个元素乘以 3
    M.row_mult(1,3)
    # 断言 M 中特定位置的值应符合预期
    assert M[1,0] == 12
    assert M[0,0] == 1
    assert M[1,2] == 18


# 定义一个测试函数，用于测试矩阵的行加法操作
def test_row_add():
    # 创建一个 3x3 的矩阵 M
    M = Matrix([[1,2,3],
               [4,5,6],
               [1,1,1]])
    # 对 M 执行行加法操作：第 2 行每个元素加上第 0 行对应元素乘以 5
    M.row_add(2,0,5)
    # 断言 M 中特定位置的值应符合预期
    assert M[0,0] == 6
    assert M[1,0] == 4
    assert M[0,2] == 8


# 定义一个测试函数，用于测试可变矩阵类的行操作
def test_zip_row_op():
    # 遍历所有可变矩阵类，因为不可变矩阵不支持行操作
    for cls in mutable_classes: # XXX: immutable matrices don't support row ops
        # 创建一个 3x3 的单位矩阵 M
        M = cls.eye(3)
        # 对 M 执行 ZIP 行操作：第 1 行每个元素加上第 0 行对应元素的两倍
        M.zip_row_op(1, 0, lambda v, u: v + 2*u)
        # 断言 M 应等于经过操作后的矩阵
        assert M == cls([[1, 0, 0],
                         [2, 1, 0],
                         [0, 0, 1]])

        # 创建一个所有元素乘以 2 的 3x3 单位矩阵 M，并修改其中的元素
        M = cls.eye(3)*2
        M[0, 1] = -1
        # 对 M 执行 ZIP 行操作：第 1 行每个元素加上第 0 行对应元素的两倍
        M.zip_row_op(1, 0, lambda v, u: v + 2*u); M
        # 断言 M 应等于经过操作后的矩阵
        assert M == cls([[2, -1, 0],
                         [4,  0, 0],
                         [0,  0, 2]])


# 定义一个测试函数，用于测试问题 3950
def test_issue_3950():
    # 创建一个 1x3 的行向量 m 和两个相同元素
    # 断言：检查变量 m 是否等于变量 a，如果不相等则抛出 AssertionError
    assert m == a
    
    # 断言：检查变量 m 是否不等于变量 b，如果相等则抛出 AssertionError
    assert m != b
def test_issue_3981():
    # 定义一个返回索引值为1的类
    class Index1:
        def __index__(self):
            return 1

    # 定义一个返回索引值为2的类
    class Index2:
        def __index__(self):
            return 2

    # 创建 Index1 和 Index2 的实例
    index1 = Index1()
    index2 = Index2()

    # 创建一个包含 [1, 2, 3] 的 Matrix 对象
    m = Matrix([1, 2, 3])

    # 断言 m[index2] 应该等于 3
    assert m[index2] == 3

    # 设置 m[index2] 的值为 5
    m[index2] = 5
    # 断言 m[2] 应该等于 5
    assert m[2] == 5

    # 创建一个包含 [[1, 2, 3], [4, 5, 6]] 的 Matrix 对象
    m = Matrix([[1, 2, 3], [4, 5, 6]])

    # 断言 m[index1, index2] 应该等于 6
    assert m[index1, index2] == 6
    # 断言 m[1, index2] 应该等于 6
    assert m[1, index2] == 6
    # 断言 m[index1, 2] 应该等于 6
    assert m[index1, 2] == 6

    # 设置 m[index1, index2] 的值为 4
    m[index1, index2] = 4
    # 断言 m[1, 2] 应该等于 4
    assert m[1, 2] == 4

    # 设置 m[1, index2] 的值为 6
    m[1, index2] = 6
    # 断言 m[1, 2] 应该等于 6
    assert m[1, 2] == 6

    # 设置 m[index1, 2] 的值为 8
    m[index1, 2] = 8
    # 断言 m[1, 2] 应该等于 8
    assert m[1, 2] == 8


def test_is_upper():
    # 创建一个包含 [[1, 2, 3]] 的 Matrix 对象
    a = Matrix([[1, 2, 3]])
    # 断言 a.is_upper 应为 True
    assert a.is_upper is True

    # 创建一个包含 [[1], [2], [3]] 的 Matrix 对象
    a = Matrix([[1], [2], [3]])
    # 断言 a.is_upper 应为 False
    assert a.is_upper is False

    # 创建一个 4x2 的零矩阵
    a = zeros(4, 2)
    # 断言 a.is_upper 应为 True
    assert a.is_upper is True


def test_is_lower():
    # 创建一个包含 [[1, 2, 3]] 的 Matrix 对象
    a = Matrix([[1, 2, 3]])
    # 断言 a.is_lower 应为 False
    assert a.is_lower is False

    # 创建一个包含 [[1], [2], [3]] 的 Matrix 对象
    a = Matrix([[1], [2], [3]])
    # 断言 a.is_lower 应为 True
    assert a.is_lower is True


def test_is_nilpotent():
    # 创建一个 4x4 的 Matrix 对象，包含指定数据
    a = Matrix(4, 4, [0, 2, 1, 6, 0, 0, 1, 2, 0, 0, 0, 3, 0, 0, 0, 0])
    # 断言 a.is_nilpotent() 应为 True
    assert a.is_nilpotent()

    # 创建一个包含 [[1, 0], [0, 1]] 的 Matrix 对象
    a = Matrix([[1, 0], [0, 1]])
    # 断言 a.is_nilpotent() 应为 False
    assert not a.is_nilpotent()

    # 创建一个空的 Matrix 对象
    a = Matrix([])
    # 断言 a.is_nilpotent() 应为 True
    assert a.is_nilpotent()


def test_zeros_ones_fill():
    n, m = 3, 5

    # 创建一个 3x5 的零矩阵
    a = zeros(n, m)
    # 将矩阵中所有元素填充为 5
    a.fill(5)

    # 创建一个全为 5 的 3x5 矩阵
    b = 5 * ones(n, m)

    # 断言 a 和 b 相等
    assert a == b
    # 断言 a 和 b 的行数都为 3
    assert a.rows == b.rows == 3
    # 断言 a 和 b 的列数都为 5
    assert a.cols == b.cols == 5
    # 断言 a 和 b 的形状都为 (3, 5)
    assert a.shape == b.shape == (3, 5)
    # 断言 zeros(2) 等于 zeros(2, 2)
    assert zeros(2) == zeros(2, 2)
    # 断言 ones(2) 等于 ones(2, 2)
    assert ones(2) == ones(2, 2)
    # 断言 zeros(2, 3) 等于 Matrix(2, 3, [0]*6)
    assert zeros(2, 3) == Matrix(2, 3, [0]*6)
    # 断言 ones(2, 3) 等于 Matrix(2, 3, [1]*6)
    assert ones(2, 3) == Matrix(2, 3, [1]*6)

    # 将 a 中所有元素填充为 0
    a.fill(0)
    # 断言 a 等于一个 3x5 的零矩阵
    assert a == zeros(n, m)


def test_empty_zeros():
    # 创建一个空的 Matrix 对象
    a = zeros(0)
    # 断言 a 等于一个空的 Matrix 对象
    assert a == Matrix()

    # 创建一个 0x2 的零矩阵
    a = zeros(0, 2)
    # 断言 a 的行数为 0
    assert a.rows == 0
    # 断言 a 的列数为 2
    assert a.cols == 2

    # 创建一个 2x0 的零矩阵
    a = zeros(2, 0)
    # 断言 a 的行数为 2
    assert a.rows == 2
    # 断言 a 的列数为 0
    assert a.cols == 0


def test_issue_3749():
    # 创建一个包含表达式的 Matrix 对象
    a = Matrix([[x**2, x*y], [x*sin(y), x*cos(y)]])
    # 断言 a 对 x 的求导结果应为 Matrix([[2*x, y], [sin(y), cos(y)]])

    assert a.diff(x) == Matrix([[2*x, y], [sin(y), cos(y)]])

    # 断言 Matrix 对象在 x 趋向无穷大时的极限应为 Matrix([[oo, -oo, oo], [oo, 0, oo]])
    assert Matrix([
        [x, -x, x**2],
        [exp(x), 1/x - exp(-x), x + 1/x]]).limit(x, oo) == \
        Matrix([[oo, -oo, oo], [oo, 0, oo]])

    # 断言 Matrix 对象在 x 趋向 0 时的极限应为 Matrix([[1, 0, 1], [oo, 0, sin(1)]])
    assert Matrix([
        [(exp(x) - 1)/x, 2*x + y*x, x**x ],
        [1/x, abs(x), abs(sin(x + 1))]]).limit(x, 0) == \
        Matrix([[1, 0, 1], [oo, 0, sin(1)]])

    # 断言 a 对 x 的积分结果应为 Matrix([[Rational(1, 3)*x**3, y*x**2/2], [x**2*sin(y)/2, x**2*cos(y)/2]])
    assert a.integrate(x) == Matrix([
        [Rational(1, 3)*x**3, y*x**2/2],
        [x**2*sin(y)/2, x
    # 创建一个包含三个元素的向量 X，分别是 rho*cos(phi)，rho*sin(phi)，rho**2
    X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
    # 创建一个包含两个元素的向量 Y，分别是 rho 和 phi
    Y = Matrix([rho, phi])
    # 创建一个雅可比矩阵 J，其元素根据 rho 和 phi 计算得出
    J = Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi), rho*cos(phi)],
        [2*rho, 0],
    ])
    # 断言 X 对 Y 的雅可比矩阵与预期的矩阵 J 相等
    assert X.jacobian(Y) == J
def test_issue_4564():
    X = Matrix([exp(x + y + z), exp(x + y + z), exp(x + y + z)])
    Y = Matrix([x, y, z])
    for i in range(1, 3):  # 循环 i 从 1 到 2
        for j in range(1, 3):  # 循环 j 从 1 到 2
            X_slice = X[:i, :]  # 从 X 中取前 i 行，所有列的切片
            Y_slice = Y[:j, :]  # 从 Y 中取前 j 行，所有列的切片
            J = X_slice.jacobian(Y_slice)  # 计算 X_slice 对 Y_slice 的雅可比矩阵
            assert J.rows == i  # 断言雅可比矩阵的行数为 i
            assert J.cols == j  # 断言雅可比矩阵的列数为 j
            for k in range(j):  # 循环 k 从 0 到 j-1
                assert J[:, k] == X_slice  # 断言雅可比矩阵的第 k 列等于 X_slice


def test_nonvectorJacobian():
    X = Matrix([[exp(x + y + z), exp(x + y + z)],
                [exp(x + y + z), exp(x + y + z)]])  # 创建一个 2x2 的矩阵 X
    raises(TypeError, lambda: X.jacobian(Matrix([x, y, z])))  # 断言对于非向量输入会抛出 TypeError 异常
    X = X[0, :]  # 取 X 的第一行作为新的 X
    Y = Matrix([[x, y], [x, z]])  # 创建一个 2x2 的矩阵 Y
    raises(TypeError, lambda: X.jacobian(Y))  # 断言对于非向量输入会抛出 TypeError 异常
    raises(TypeError, lambda: X.jacobian(Matrix([ [x, y], [x, z] ])))  # 断言对于非向量输入会抛出 TypeError 异常


def test_vec():
    m = Matrix([[1, 3], [2, 4]])  # 创建一个 2x2 的矩阵 m
    m_vec = m.vec()  # 将矩阵 m 转换成列向量 m_vec
    assert m_vec.cols == 1  # 断言 m_vec 的列数为 1
    for i in range(4):  # 循环 i 从 0 到 3
        assert m_vec[i] == i + 1  # 断言 m_vec 的第 i 个元素等于 i + 1


def test_vech():
    m = Matrix([[1, 2], [2, 3]])  # 创建一个 2x2 的矩阵 m
    m_vech = m.vech()  # 计算 m 的上三角部分并展开成列向量 m_vech
    assert m_vech.cols == 1  # 断言 m_vech 的列数为 1
    for i in range(3):  # 循环 i 从 0 到 2
        assert m_vech[i] == i + 1  # 断言 m_vech 的第 i 个元素等于 i + 1
    m_vech = m.vech(diagonal=False)  # 计算 m 的非对角线部分并展开成列向量 m_vech
    assert m_vech[0] == 2  # 断言 m_vech 的第一个元素为 2

    m = Matrix([[1, x*(x + y)], [y*x + x**2, 1]])  # 创建一个 2x2 的矩阵 m
    m_vech = m.vech(diagonal=False)  # 计算 m 的非对角线部分并展开成列向量 m_vech
    assert m_vech[0] == y*x + x**2  # 断言 m_vech 的第一个元素符合预期

    m = Matrix([[1, x*(x + y)], [y*x, 1]])  # 创建一个 2x2 的矩阵 m
    m_vech = m.vech(diagonal=False, check_symmetry=False)  # 计算 m 的非对角线部分并展开成列向量 m_vech，不检查对称性
    assert m_vech[0] == y*x  # 断言 m_vech 的第一个元素符合预期

    raises(ShapeError, lambda: Matrix([[1, 3]]).vech())  # 断言非方阵调用 vech 方法会抛出 ShapeError 异常
    raises(ValueError, lambda: Matrix([[1, 3], [2, 4]]).vech())  # 断言非方阵调用 vech 方法会抛出 ValueError 异常
    raises(ShapeError, lambda: Matrix([[1, 3]]).vech())  # 断言非方阵调用 vech 方法会抛出 ShapeError 异常
    raises(ValueError, lambda: Matrix([[1, 3], [2, 4]]).vech())  # 断言非方阵调用 vech 方法会抛出 ValueError 异常


def test_diag():
    # mostly tested in testcommonmatrix.py
    assert diag([1, 2, 3]) == Matrix([1, 2, 3])  # 断言创建对角矩阵的正确性
    m = [1, 2, [3]]
    raises(ValueError, lambda: diag(m))  # 断言创建对角矩阵时，传入非法参数会抛出 ValueError 异常
    assert diag(m, strict=False) == Matrix([1, 2, 3])  # 断言创建对角矩阵的正确性


def test_inv_block():
    a = Matrix([[1, 2], [2, 3]])  # 创建一个 2x2 的矩阵 a
    b = Matrix([[3, x], [y, 3]])  # 创建一个 2x2 的矩阵 b
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])  # 创建一个 3x3 的矩阵 c
    A = diag(a, b, b)  # 创建对角矩阵 A，由 a, b, b 组成
    assert A.inv(try_block_diag=True) == diag(a.inv(), b.inv(), b.inv())  # 断言 A 的逆矩阵是否正确计算
    A = diag(a, b, c)  # 创建对角矩阵 A，由 a, b, c 组成
    assert A.inv(try_block_diag=True) == diag(a.inv(), b.inv(), c.inv())  # 断言 A 的逆矩阵是否正确计算
    A = diag(a, c, b)  # 创建对角矩阵 A，由 a, c, b 组成
    assert A.inv(try_block_diag=True) == diag(a.inv(), c.inv(), b.inv())  # 断言 A 的逆矩阵是否正确计算
    A = diag(a, a, b, a, c, a)  # 创建对角矩阵 A，由 a, a, b, a, c, a 组成
    assert A.inv(try_block_diag=True) == diag(
        a.inv(), a.inv(), b.inv(), a.inv(), c.inv(), a.inv())  # 断言 A 的逆矩阵是否正确计算
    assert A.inv(try_block_diag=True, method="ADJ") == diag(
        a.inv(method="ADJ"), a.inv(method="ADJ"), b.inv(method="ADJ"),
        a.inv(method="ADJ"), c.inv(method="ADJ"), a.inv(method="ADJ"))  # 断言 A 的逆矩阵是否正确计算（采用 ADJ 方法）


def test_creation_args():
    """
    Check that matrix dimensions can be specified using any reasonable type
    (see issue 4614).
    """
    raises(ValueError, lambda: zeros(3, -1))  # 断言传入非法维度参数会抛出 ValueError 异常
    raises(TypeError, lambda: zeros(1, 2, 3, 4))  # 断言传入多于两个参数会抛出 TypeError 异常
    assert zeros(int(3)) == zeros(3)  # 断言使用整
    # 断言调用 eye 函数生成的矩阵与直接使用整数参数调用的结果相同
    assert eye(int(3)) == eye(3)
    
    # 断言调用 eye 函数生成的矩阵与使用 Integer 类型整数参数调用的结果相同
    assert eye(Integer(3)) == eye(3)
    
    # 使用 lambda 函数和 raises 断言，验证传入浮点数参数调用 eye 函数会引发 ValueError 异常
    raises(ValueError, lambda: eye(3.))
    
    # 断言调用 ones 函数生成的矩阵与使用整数参数调用的结果相同
    assert ones(int(3), Integer(4)) == ones(3, 4)
    
    # 使用 lambda 函数和 raises 断言，验证传入单个整数参数调用 Matrix 构造函数会引发 TypeError 异常
    raises(TypeError, lambda: Matrix(5))
    
    # 使用 lambda 函数和 raises 断言，验证传入两个参数调用 Matrix 构造函数会引发 TypeError 异常
    raises(TypeError, lambda: Matrix(1, 2))
    
    # 使用 lambda 函数和 raises 断言，验证传入不合法的列表参数调用 Matrix 构造函数会引发 ValueError 异常
    raises(ValueError, lambda: Matrix([1, [2]]))
# 定义测试函数，用于检验对角矩阵和对称性的相关功能
def test_diagonal_symmetrical():
    # 创建一个 2x2 的矩阵 m，数据为 [0, 1, 1, 0]
    m = Matrix(2, 2, [0, 1, 1, 0])
    # 断言矩阵 m 不是对角矩阵
    assert not m.is_diagonal()
    # 断言矩阵 m 是对称的
    assert m.is_symmetric()
    # 断言矩阵 m 在不简化情况下仍然是对称的
    assert m.is_symmetric(simplify=False)

    # 创建一个 2x2 的对角矩阵 m，数据为 [1, 0, 0, 1]
    m = Matrix(2, 2, [1, 0, 0, 1])
    # 断言矩阵 m 是对角矩阵
    assert m.is_diagonal()

    # 创建一个对角矩阵 m，数据为 [1, 2, 3]
    m = diag(1, 2, 3)
    # 断言矩阵 m 是对角矩阵
    assert m.is_diagonal()
    # 断言矩阵 m 是对称的
    assert m.is_symmetric()

    # 创建一个 3x3 的矩阵 m，数据为 [1, 0, 0, 0, 2, 0, 0, 0, 3]
    m = Matrix(3, 3, [1, 0, 0, 0, 2, 0, 0, 0, 3])
    # 断言矩阵 m 等于对角矩阵 diag(1, 2, 3)
    assert m == diag(1, 2, 3)

    # 创建一个 2x3 的矩阵 m，数据为全零
    m = Matrix(2, 3, zeros(2, 3))
    # 断言矩阵 m 不是对称的
    assert not m.is_symmetric()
    # 断言矩阵 m 是对角矩阵
    assert m.is_diagonal()

    # 创建一个 3x2 的矩阵 m，数据为 ((5, 0), (0, 6), (0, 0))
    m = Matrix(((5, 0), (0, 6), (0, 0)))
    # 断言矩阵 m 是对角矩阵
    assert m.is_diagonal()

    # 创建一个 2x3 的矩阵 m，数据为 ((5, 0, 0), (0, 6, 0))
    m = Matrix(((5, 0, 0), (0, 6, 0)))
    # 断言矩阵 m 是对角矩阵
    assert m.is_diagonal()

    # 创建一个 3x3 的矩阵 m，数据包含符号变量 x 和 y
    m = Matrix(3, 3, [1, x**2 + 2*x + 1, y, (x + 1)**2, 2, 0, y, 0, 3])
    # 断言矩阵 m 是对称的
    assert m.is_symmetric()
    # 断言矩阵 m 在不简化情况下不是对称的
    assert not m.is_symmetric(simplify=False)
    # 断言矩阵 m 展开后在不简化情况下是对称的
    assert m.expand().is_symmetric(simplify=False)


# 定义测试函数，用于检验矩阵的对角化功能
def test_diagonalization():
    # 创建一个 2x2 的复数矩阵 m，数据包含复数
    m = Matrix([[1, 2+I], [2-I, 3]])
    # 断言矩阵 m 是可对角化的
    assert m.is_diagonalizable()

    # 创建一个 3x2 的矩阵 m，数据为 [-3, 1, -3, 20, 3, 10]
    m = Matrix(3, 2, [-3, 1, -3, 20, 3, 10])
    # 断言矩阵 m 不可对角化
    assert not m.is_diagonalizable()
    # 断言矩阵 m 不是对称的
    assert not m.is_symmetric()
    # 使用 lambda 表达式断言调用 diagonalize() 会引发 NonSquareMatrixError 异常
    raises(NonSquareMatrixError, lambda: m.diagonalize())

    # 对一个对角矩阵进行对角化测试
    m = diag(1, 2, 3)
    (P, D) = m.diagonalize()
    # 断言对角化后的 P 矩阵为单位矩阵
    assert P == eye(3)
    # 断言对角化后的 D 矩阵与原始对角矩阵 m 相等
    assert D == m

    # 创建一个 2x2 的矩阵 m，数据为 [0, 1, 1, 0]
    m = Matrix(2, 2, [0, 1, 1, 0])
    # 断言矩阵 m 是对称的
    assert m.is_symmetric()
    # 断言矩阵 m 是可对角化的
    assert m.is_diagonalizable()
    (P, D) = m.diagonalize()
    # 断言 P 的逆乘以 m 乘以 P 等于 D
    assert P.inv() * m * P == D

    # 创建一个 2x2 的矩阵 m，数据为 [1, 0, 0, 3]
    m = Matrix(2, 2, [1, 0, 0, 3])
    # 断言矩阵 m 是对称的
    assert m.is_symmetric()
    # 断言矩阵 m 是可对角化的
    assert m.is_diagonalizable()
    (P, D) = m.diagonalize()
    # 断言 P 的逆乘以 m 乘以 P 等于 D
    assert P.inv() * m * P == D
    # 断言 P 是单位矩阵
    assert P == eye(2)
    # 断言 D 等于矩阵 m
    assert D == m

    # 创建一个 2x2 的矩阵 m，数据为 [1, 1, 0, 0]
    m = Matrix(2, 2, [1, 1, 0, 0])
    # 断言矩阵 m 是可对角化的
    assert m.is_diagonalizable()
    (P, D) = m.diagonalize()
    # 断言 P 的逆乘以 m 乘以 P 等于 D
    assert P.inv() * m * P == D

    # 创建一个 3x3 的矩阵 m，数据为 [1, 2, 0, 0, 3, 0, 2, -4, 2]
    m = Matrix(3, 3, [1, 2, 0, 0, 3, 0, 2, -4, 2])
    # 断言矩阵 m 是可对角化的
    assert m.is_diagonalizable()
    (P, D) = m.diagonalize()
    # 断言 P 的逆乘以 m 乘以 P 等于 D
    assert P.inv() * m * P == D
    # 断言 P 的每个元素的分母为 1，即 P 是整数矩阵
    for i in P:
        assert i.as_numer_denom()[1] == 1

    # 创建一个 2x2 的矩阵 m，数据为 [1, 0, 0, 0]
    m = Matrix(2, 2, [1, 0, 0, 0])
    # 断言矩阵 m 是对角矩阵
    assert m.is_diagonal()
    # 断言矩阵 m 是可对角化的
    assert
    # 断言矩阵 a 不可对角化（即非对角化）
    assert a.is_diagonalizable() is False

    # 创建一个可变的二维密集矩阵 a，内容为 [[0, 1], [1, 0]]
    a = MutableDenseMatrix([[0, 1], [1, 0]])
    # 对矩阵 a 进行对角化操作
    a.diagonalize()
    # 修改矩阵 a 中索引为 (1, 0) 的元素为 0
    a[1, 0] = 0
    # 使用 lambda 函数结合 raises 函数检查执行对角化操作时是否抛出 MatrixError 异常
    raises(MatrixError, lambda: a.diagonalize())
# 定义测试函数 test_jordan_form
def test_jordan_form():

    # 创建一个非方阵 Matrix 对象 m，指定其尺寸和数据
    m = Matrix(3, 2, [-3, 1, -3, 20, 3, 10])
    # 断言调用 m.jordan_form() 会引发 NonSquareMatrixError 异常
    raises(NonSquareMatrixError, lambda: m.jordan_form())

    # diagonalizable
    # 创建一个方阵 Matrix 对象 m，指定其尺寸和数据
    m = Matrix(3, 3, [7, -12, 6, 10, -19, 10, 12, -24, 13])
    # 创建一个预期的 Jordan 形式 Matrix 对象 Jmust
    Jmust = Matrix(3, 3, [-1, 0, 0, 0, 1, 0, 0, 0, 1])
    # 调用 m.jordan_form() 方法，返回 P 和 J 两个 Matrix 对象
    P, J = m.jordan_form()
    # 断言 Jmust 和 J 相等
    assert Jmust == J
    # 断言 Jmust 和 m.diagonalize()[1] 相等
    assert Jmust == m.diagonalize()[1]

    # m = Matrix(3, 3, [0, 6, 3, 1, 3, 1, -2, 2, 1])
    # m.jordan_form()  # very long
    # m.jordan_form()  #

    # diagonalizable, complex only

    # Jordan cells
    # 创建一个方阵 Matrix 对象 m，指定其尺寸和数据
    m = Matrix(3, 3, [0, 1, 0, -4, 4, 0, -2, 1, 2])
    # 添加注释说明：块按照特征值的值排序，以便与 .diagonalize() 兼容
    # 创建一个预期的 Jordan 形式 Matrix 对象 Jmust
    Jmust = Matrix(3, 3, [2, 1, 0, 0, 2, 0, 0, 0, 2])
    # 调用 m.jordan_form() 方法，返回 P 和 J 两个 Matrix 对象
    P, J = m.jordan_form()
    # 断言 Jmust 和 J 相等
    assert Jmust == J

    # complexity: all of eigenvalues are equal
    # 创建一个方阵 Matrix 对象 m，指定其尺寸和数据
    m = Matrix(3, 3, [2, 6, -15, 1, 1, -5, 1, 2, -6])
    # 创建一个预期的 Jordan 形式 Matrix 对象 Jmust
    Jmust = Matrix(3, 3, [-1, 1, 0, 0, -1, 0, 0, 0, -1])
    # 调用 m.jordan_form() 方法，返回 P 和 J 两个 Matrix 对象
    P, J = m.jordan_form()
    # 断言 Jmust 和 J 相等
    assert Jmust == J

    # complexity: two of eigenvalues are zero
    # 创建一个方阵 Matrix 对象 m，指定其尺寸和数据
    m = Matrix(3, 3, [4, -5, 2, 5, -7, 3, 6, -9, 4])
    # 创建一个预期的 Jordan 形式 Matrix 对象 Jmust
    Jmust = Matrix(3, 3, [0, 1, 0, 0, 0, 0, 0, 0, 1])
    # 调用 m.jordan_form() 方法，返回 P 和 J 两个 Matrix 对象
    P, J = m.jordan_form()
    # 断言 Jmust 和 J 相等
    assert Jmust == J

    # 创建一个方阵 Matrix 对象 m，指定其尺寸和数据
    m = Matrix(4, 4, [6, 5, -2, -3, -3, -1, 3, 3, 2, 1, -2, -3, -1, 1, 5, 5])
    # 创建一个预期的 Jordan 形式 Matrix 对象 Jmust
    Jmust = Matrix(4, 4, [2, 1, 0, 0,
                          0, 2, 0, 0,
                          0, 0, 2, 1,
                          0, 0, 0, 2])
    # 调用 m.jordan_form() 方法，返回 P 和 J 两个 Matrix 对象
    P, J = m.jordan_form()
    # 断言 Jmust 和 J 相等
    assert Jmust == J

    # 创建一个方阵 Matrix 对象 m，指定其尺寸和数据
    m = Matrix(4, 4, [6, 2, -8, -6, -3, 2, 9, 6, 2, -2, -8, -6, -1, 0, 3, 4])
    # 创建一个预期的 Jordan 形式 Matrix 对象 Jmust
    Jmust = Matrix(4, 4, [-2, 0, 0, 0,
                         0, 2, 1, 0,
                         0, 0, 2, 0,
                         0, 0, 0, 2])
    # 调用 m.jordan_form() 方法，返回 P 和 J 两个 Matrix 对象
    P, J = m.jordan_form()
    # 断言 Jmust 和 J 相等
    assert Jmust == J

    # 创建一个方阵 Matrix 对象 m，指定其尺寸和数据
    m = Matrix(4, 4, [5, 4, 2, 1, 0, 1, -1, -1, -1, -1, 3, 0, 1, 1, -1, 2])
    # 断言 m 不可对角化
    assert not m.is_diagonalizable()
    # 创建一个预期的 Jordan 形式 Matrix 对象 Jmust
    Jmust = Matrix(4, 4, [1, 0, 0, 0,
                         0, 2, 0, 0,
                         0, 0, 4, 1,
                         0, 0, 0, 4])
    # 调用 m.jordan_form() 方法，返回 P 和 J 两个 Matrix 对象
    P, J = m.jordan_form()
    # 断言 Jmust 和 J 相等
    assert Jmust == J

    # checking for maximum precision to remain unchanged
    # 创建一个精度为 110 的 Float 对象组成的 Matrix 对象 m
    m = Matrix([[Float('1.0', precision=110), Float('2.0', precision=110)],
                [Float('3.14159265358979323846264338327', precision=110), Float('4.0', precision=110)]])
    # 调用 m.jordan_form() 方法，返回 P 和 J 两个 Matrix 对象
    P, J = m.jordan_form()
    # 遍历 J 中的每个元素
    for term in J.values():
        # 如果 term 是 Float 对象，则断言其精度为 110
        if isinstance(term, Float):
            assert term._prec == 110


# 定义测试函数 test_jordan_form_complex_issue_9274
def test_jordan_form_complex_issue_9274():
    # 创建一个复数 Matrix 对象 A，指定其尺寸和数据
    A = Matrix([[ 2,  4,  1,  0],
                [-4,  2,  0,  1],
                [ 0,  0,  2,  4],
                [ 0,  0, -4,  2]])
    # 定义复数 p 和 q
    p = 2 - 4*I;
    q = 2 + 4*I;
    # 创建一个预期的 Jordan 形式 Matrix 对象 Jmust1
    Jmust1 = Matrix([[p, 1, 0, 0],
                     [0, p, 0, 0],
                     [0, 0, q, 1],
                     [0, 0, 0, q]])
    # 创建一个4x4的矩阵 Jmust2，其中 q 和 p 是预定义的变量
    Jmust2 = Matrix([[q, 1, 0, 0],
                     [0, q, 0, 0],
                     [0, 0, p, 1],
                     [0, 0, 0, p]])
    
    # 计算矩阵 A 的乔尔当标准型，分别返回 P（相似变换矩阵）和 J（乔尔当形式矩阵）
    P, J = A.jordan_form()
    
    # 断言乔尔当形式矩阵 J 等于 Jmust1 或者 Jmust2
    assert J == Jmust1 or J == Jmust2
    
    # 断言 P*J*P.inv() 简化后等于矩阵 A 自身
    assert simplify(P*J*P.inv()) == A
def test_issue_10220():
    # 定义一个4x4的矩阵M，具有两个非正交的Jordan块，特征值为1
    M = Matrix([[1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1]])
    # 计算M的Jordan标准形P和J
    P, J = M.jordan_form()
    # 断言P与预期的矩阵相等
    assert P == Matrix([[0, 1, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])
    # 断言J与预期的矩阵相等
    assert J == Matrix([
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])


def test_jordan_form_issue_15858():
    # 定义一个4x4的矩阵A
    A = Matrix([
        [1, 1, 1, 0],
        [-2, -1, 0, -1],
        [0, 0, -1, -1],
        [0, 0, 2, 1]])
    # 计算矩阵A的Jordan标准形P和J
    (P, J) = A.jordan_form()
    # 断言P展开后与预期的矩阵相等
    assert P.expand() == Matrix([
        [    -I,          -I/2,      I,           I/2],
        [-1 + I,             0, -1 - I,             0],
        [     0, -S(1)/2 - I/2,      0, -S(1)/2 + I/2],
        [     0,             1,      0,             1]])
    # 断言J与预期的矩阵相等
    assert J == Matrix([
        [-I, 1, 0, 0],
        [0, -I, 0, 0],
        [0, 0, I, 1],
        [0, 0, 0, I]])


def test_Matrix_berkowitz_charpoly():
    # 定义符号变量
    UA, K_i, K_w = symbols('UA K_i K_w')

    # 定义一个2x2的矩阵A，包含符号变量
    A = Matrix([[-K_i - UA + K_i**2/(K_i + K_w),       K_i*K_w/(K_i + K_w)],
                [           K_i*K_w/(K_i + K_w), -K_w + K_w**2/(K_i + K_w)]])

    # 计算矩阵A的特征多项式charpoly
    charpoly = A.charpoly(x)

    # 断言charpoly与预期的多项式相等
    assert charpoly == \
        Poly(x**2 + (K_i*UA + K_w*UA + 2*K_i*K_w)/(K_i + K_w)*x +
        K_i*K_w*UA/(K_i + K_w), x, domain='ZZ(K_i,K_w,UA)')

    # 断言charpoly的类型为PurePoly
    assert type(charpoly) is PurePoly

    # 对于一个具体的2x2矩阵A
    A = Matrix([[1, 3], [2, 0]])
    # 断言A的特征多项式与预期的PurePoly相等
    assert A.charpoly() == A.charpoly(x) == PurePoly(x**2 - x - 6)

    # 对于一个包含符号变量的2x2矩阵A
    A = Matrix([[1, 2], [x, 0]])
    # 计算A的特征多项式p
    p = A.charpoly(x)
    # 断言p的生成元与x不相等
    assert p.gen != x
    # 断言将p的生成元替换为x后，结果与预期的多项式相等
    assert p.as_expr().subs(p.gen, x) == x**2 - 3*x


def test_exp_jordan_block():
    l = Symbol('lamda')

    # 创建一个大小为1x1的Jordan块m
    m = Matrix.jordan_block(1, l)
    # 断言m的指数矩阵表达式与预期的矩阵相等
    assert m._eval_matrix_exp_jblock() == Matrix([[exp(l)]])

    # 创建一个大小为3x3的Jordan块m
    m = Matrix.jordan_block(3, l)
    # 断言m的指数矩阵表达式与预期的矩阵相等
    assert m._eval_matrix_exp_jblock() == \
        Matrix([
            [exp(l), exp(l), exp(l)/2],
            [0, exp(l), exp(l)],
            [0, 0, exp(l)]])


def test_exp():
    # 定义一个2x2的矩阵m
    m = Matrix([[3, 4], [0, -2]])
    # 计算m的指数函数结果m_exp
    m_exp = Matrix([[exp(3), -4*exp(-2)/5 + 4*exp(3)/5], [0, exp(-2)]])
    # 断言m的指数函数与预期的矩阵相等
    assert m.exp() == m_exp
    # 断言将exp(m)与预期的矩阵相等
    assert exp(m) == m_exp

    # 对于一个单位矩阵m
    m = Matrix([[1, 0], [0, 1]])
    # 断言m的指数函数与预期的矩阵相等
    assert m.exp() == Matrix([[E, 0], [0, E]])
    # 断言将exp(m)与预期的矩阵相等
    assert exp(m) == Matrix([[E, 0], [0, E]])

    # 对于一个2x2的矩阵m
    m = Matrix([[1, -1], [1, 1]])
    # 断言m的指数函数与预期的矩阵相等
    assert m.exp() == Matrix([[E*cos(1), -E*sin(1)], [E*sin(1), E*cos(1)]])


def test_log():
    l = Symbol('lamda')

    # 创建一个大小为1x1的Jordan块m
    m = Matrix.jordan_block(1, l)
    # 断言m的对数矩阵表达式与预期的矩阵相等
    assert m._eval_matrix_log_jblock() == Matrix([[log(l)]])

    # 创建一个大小为4x4的Jordan块m
    m = Matrix.jordan_block(4, l)
    # 断言：调用对象 m 的 _eval_matrix_log_jblock 方法，期望其返回值等于以下 Matrix 对象
    assert m._eval_matrix_log_jblock() == \
        Matrix(
            [
                [log(l), 1/l, -1/(2*l**2), 1/(3*l**3)],  # 第一行：对数函数应用于 l 的各阶导数
                [0, log(l), 1/l, -1/(2*l**2)],          # 第二行：对数函数应用于 l 的各阶导数
                [0, 0, log(l), 1/l],                    # 第三行：对数函数应用于 l 的各阶导数
                [0, 0, 0, log(l)]                       # 第四行：对数函数应用于 l 的各阶导数
            ]
        )

    # 创建一个 3x3 的 Matrix 对象 m
    m = Matrix(
        [[0, 0, 1],
        [0, 0, 0],
        [-1, 0, 0]]
    )
    # 断言：调用 m 的 log 方法应该引发 MatrixError 异常
    raises(MatrixError, lambda: m.log())
# 测试 matrices._find_reasonable_pivot_naive() 函数，验证在存在符号表达式的候选主元时能够找到保证非零的主元。
# 关键字参数 simpfunc=None 表示在搜索过程中不执行任何简化操作。
def test_find_reasonable_pivot_naive_finds_guaranteed_nonzero1():
    x = Symbol('x')
    # 创建一个 3x1 的矩阵列向量，包含符号变量 x，三角函数的平方和，以及 S.Half 常量。
    column = Matrix(3, 1, [x, cos(x)**2 + sin(x)**2, S.Half])
    # 调用 _find_reasonable_pivot_naive() 函数，获取返回的多个变量赋值
    pivot_offset, pivot_val, pivot_assumed_nonzero, simplified =\
        _find_reasonable_pivot_naive(column)
    # 断言 pivot_val 的值应为 S.Half
    assert pivot_val == S.Half


# 测试 matrices._find_reasonable_pivot_naive() 函数，验证在存在符号表达式的候选主元时能够找到保证非零的主元。
# 关键字参数 simpfunc=_simplify 表示搜索过程中尝试简化候选主元。
def test_find_reasonable_pivot_naive_finds_guaranteed_nonzero2():
    x = Symbol('x')
    # 创建一个 3x1 的矩阵列向量，包含符号变量 x，三角函数的平方和加 x 的平方，以及三角函数的平方和。
    column = Matrix(3, 1,
                    [x,
                     cos(x)**2+sin(x)**2+x**2,
                     cos(x)**2+sin(x)**2])
    # 调用 _find_reasonable_pivot_naive() 函数，获取返回的多个变量赋值
    pivot_offset, pivot_val, pivot_assumed_nonzero, simplified =\
        _find_reasonable_pivot_naive(column, simpfunc=_simplify)
    # 断言 pivot_val 的值应为 1
    assert pivot_val == 1


# 测试 matrices._find_reasonable_pivot_naive() 函数，验证候选主元的简化过程，并正确报告它们的偏移量。
def test_find_reasonable_pivot_naive_simplifies():
    x = Symbol('x')
    # 创建一个 3x1 的矩阵列向量，包含符号变量 x，三角函数的平方和加 x，以及三角函数的平方和。
    column = Matrix(3, 1,
                    [x,
                     cos(x)**2+sin(x)**2+x,
                     cos(x)**2+sin(x)**2])
    # 调用 _find_reasonable_pivot_naive() 函数，获取返回的多个变量赋值
    pivot_offset, pivot_val, pivot_assumed_nonzero, simplified =\
        _find_reasonable_pivot_naive(column, simpfunc=_simplify)

    # 断言简化后的列表 simplified 应有两个元素
    assert len(simplified) == 2
    # 断言第一个简化后的元素的偏移量为 1，值为 1+x
    assert simplified[0][0] == 1
    assert simplified[0][1] == 1+x
    # 断言第二个简化后的元素的偏移量为 2，值为 1
    assert simplified[1][0] == 2
    assert simplified[1][1] == 1


# 测试各种错误情况是否能够正确抛出异常。
def test_errors():
    raises(ValueError, lambda: Matrix([[1, 2], [1]]))
    raises(IndexError, lambda: Matrix([[1, 2]])[1.2, 5])
    raises(IndexError, lambda: Matrix([[1, 2]])[1, 5.2])
    raises(ValueError, lambda: randMatrix(3, c=4, symmetric=True))
    raises(ValueError, lambda: Matrix([1, 2]).reshape(4, 6))
    raises(ShapeError,
           lambda: Matrix([[1, 2], [3, 4]]).copyin_matrix([1, 0], Matrix([1, 2])))
    raises(TypeError, lambda: Matrix([[1, 2], [3, 4]]).copyin_list([0,
           1], set()))
    raises(NonSquareMatrixError, lambda: Matrix([[1, 2, 3], [2, 3, 0]]).inv())
    raises(ShapeError,
           lambda: Matrix(1, 2, [1, 2]).row_join(Matrix([[1, 2], [3, 4]])))
    raises(ShapeError, lambda: Matrix([1, 2]).col_join(Matrix([[1, 2], [3, 4]])))
    raises(ShapeError, lambda: Matrix([1]).row_insert(1, Matrix([[1,
           2], [3, 4]])))
    raises(ShapeError, lambda: Matrix([1]).col_insert(1, Matrix([[1,
           2], [3, 4]])))
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).trace())
    raises(TypeError, lambda: Matrix([1]).applyfunc(1))
    `
    # 检查在行列式运算中引发 ValueError，验证位置 (4, 5) 超出矩阵边界
    raises(ValueError, lambda: Matrix([[1, 2], [3, 4]]).minor(4, 5))
    
    # 检查在子矩阵运算中引发 ValueError，验证位置 (4, 5) 超出矩阵边界
    raises(ValueError, lambda: Matrix([[1, 2], [3, 4]]).minor_submatrix(4, 5))
    
    # 检查在向量叉乘运算中引发 TypeError，验证非法操作参数为整数而非向量
    raises(TypeError, lambda: Matrix([1, 2, 3]).cross(1))
    
    # 检查在向量点乘运算中引发 TypeError，验证非法操作参数为整数而非向量
    raises(TypeError, lambda: Matrix([1, 2, 3]).dot(1))
    
    # 检查在矩阵点乘运算中引发 ShapeError，验证无法进行矩阵点乘因为矩阵形状不匹配
    raises(ShapeError, lambda: Matrix([1, 2, 3]).dot(Matrix([1, 2])))
    
    # 检查在矩阵点乘运算中引发 ShapeError，验证无法进行矩阵点乘因为向量为空
    raises(ShapeError, lambda: Matrix([1, 2]).dot([]))
    
    # 检查在矩阵点乘运算中引发 TypeError，验证无法进行矩阵点乘因为参数类型为字符串而非矩阵或向量
    raises(TypeError, lambda: Matrix([1, 2]).dot('a'))
    
    # 检查在矩阵点乘运算中引发 ShapeError，验证无法进行矩阵点乘因为向量维度不匹配
    raises(ShapeError, lambda: Matrix([1, 2]).dot([1, 2, 3]))
    
    # 检查在矩阵指数运算中引发 NonSquareMatrixError，验证矩阵不是方阵无法进行指数运算
    raises(NonSquareMatrixError, lambda: Matrix([1, 2, 3]).exp())
    
    # 检查在矩阵归一化运算中引发 ShapeError，验证无法归一化矩阵因为矩阵不是一维的
    raises(ShapeError, lambda: Matrix([[1, 2], [3, 4]]).normalized())
    
    # 检查在矩阵求逆运算中引发 ValueError，验证指定的求逆方法不合法
    raises(ValueError, lambda: Matrix([1, 2]).inv(method='not a method'))
    
    # 检查在矩阵高斯消元法求逆运算中引发 NonSquareMatrixError，验证矩阵不是方阵无法通过高斯消元法求逆
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).inverse_GE())
    
    # 检查在矩阵高斯消元法求逆运算中引发 ValueError，验证矩阵不可逆
    raises(ValueError, lambda: Matrix([[1, 2], [1, 2]]).inverse_GE())
    
    # 检查在矩阵伴随矩阵法求逆运算中引发 NonSquareMatrixError，验证矩阵不是方阵无法通过伴随矩阵法求逆
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).inverse_ADJ())
    
    # 检查在矩阵伴随矩阵法求逆运算中引发 ValueError，验证矩阵不可逆
    raises(ValueError, lambda: Matrix([[1, 2], [1, 2]]).inverse_ADJ())
    
    # 检查在矩阵 LU 分解法求逆运算中引发 NonSquareMatrixError，验证矩阵不是方阵无法通过 LU 分解法求逆
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).inverse_LU())
    
    # 检查在矩阵判断是否幂零运算中引发 NonSquareMatrixError，验证矩阵不是方阵无法判断是否幂零
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).is_nilpotent())
    
    # 检查在矩阵行列式计算中引发 NonSquareMatrixError，验证矩阵不是方阵无法计算行列式
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).det())
    
    # 检查在矩阵行列式计算中引发 ValueError，验证指定的行列式计算方法不合法
    raises(ValueError,
        lambda: Matrix([[1, 2], [3, 4]]).det(method='Not a real method'))
    
    # 检查在 Hessian 矩阵计算中引发 ValueError，验证参数传递错误
    raises(ValueError,
        lambda: hessian(Matrix([[1, 2], [3, 4]]), Matrix([[1, 2], [2, 1]])))
    
    # 检查在 Hessian 矩阵计算中引发 ValueError，验证参数传递错误
    raises(ValueError, lambda: hessian(Matrix([[1, 2], [3, 4]]), []))
    
    # 检查在 Hessian 矩阵计算中引发 ValueError，验证参数传递错误
    raises(ValueError, lambda: hessian(Symbol('x')**2, 'a'))
    
    # 检查在单位矩阵索引访问中引发 IndexError，验证索引超出边界
    raises(IndexError, lambda: eye(3)[5, 2])
    
    # 检查在单位矩阵索引访问中引发 IndexError，验证索引超出边界
    raises(IndexError, lambda: eye(3)[2, 5])
    
    # 创建矩阵 M
    M = Matrix(((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)))
    
    # 检查在矩阵行列式计算中引发 ValueError，验证指定的行列式计算方法不合法
    raises(ValueError, lambda: M.det('method=LU_decomposition()'))
    
    # 创建向量 V
    V = Matrix([[10, 10, 10]])
    
    # 创建矩阵 M
    M = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    
    # 检查在矩阵行插入操作中引发 ValueError，验证插入位置非整数
    raises(ValueError, lambda: M.row_insert(4.7, V))
    
    # 创建矩阵 M
    M = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    
    # 检查在矩阵列插入操作中引发 ValueError，验证插入位置非整数
    raises(ValueError, lambda: M.col_insert(-4.2, V))
def test_len():
    # 测试 Matrix 类的长度方法
    assert len(Matrix()) == 0  # 空 Matrix 对象长度为 0
    assert len(Matrix([[1, 2]])) == len(Matrix([[1], [2]])) == 2  # 二维数组的行数和列数都为 2
    assert len(Matrix(0, 2, lambda i, j: 0)) == \
        len(Matrix(2, 0, lambda i, j: 0)) == 0  # 给定维度为 0 的 Matrix 对象长度为 0
    assert len(Matrix([[0, 1, 2], [3, 4, 5]])) == 6  # 二维数组总共有 6 个元素
    assert Matrix([1]) == Matrix([[1]])  # 使用不同的方式创建 Matrix 对象，但相等
    assert not Matrix()  # 空 Matrix 对象在布尔上下文中为假
    assert Matrix() == Matrix([])  # 空 Matrix 对象和空列表相等


def test_integrate():
    A = Matrix(((1, 4, x), (y, 2, 4), (10, 5, x**2)))

    # 测试 Matrix 对象的积分方法对 x 的积分
    assert A.integrate(x) == \
        Matrix(((x, 4*x, x**2/2), (x*y, 2*x, 4*x), (10*x, 5*x, x**3/3)))

    # 测试 Matrix 对象的积分方法对 y 的积分
    assert A.integrate(y) == \
        Matrix(((y, 4*y, x*y), (y**2/2, 2*y, 4*y), (10*y, 5*y, y*x**2)))

    m = Matrix(2, 1, [x, y])
    assert m.integrate(x) == Matrix(2, 1, [x**2/2, y*x])


def test_diff():
    A = MutableDenseMatrix(((1, 4, x), (y, 2, 4), (10, 5, x**2 + 1)))

    # 测试 Matrix 对象的 diff 方法对 x 的偏导数
    assert isinstance(A.diff(x), type(A))
    assert A.diff(x) == MutableDenseMatrix(((0, 0, 1), (0, 0, 0), (0, 0, 2*x)))
    assert A.diff(y) == MutableDenseMatrix(((0, 0, 0), (1, 0, 0), (0, 0, 0)))

    # 使用全局函数 diff 测试对 x 和 y 的偏导数
    assert diff(A, x) == MutableDenseMatrix(((0, 0, 1), (0, 0, 0), (0, 0, 2*x)))
    assert diff(A, y) == MutableDenseMatrix(((0, 0, 0), (1, 0, 0), (0, 0, 0)))

    A_imm = A.as_immutable()

    # 测试不可变 Matrix 对象的 diff 方法对 x 和 y 的偏导数
    assert isinstance(A_imm.diff(x), type(A_imm))
    assert A_imm.diff(x) == ImmutableDenseMatrix(((0, 0, 1), (0, 0, 0), (0, 0, 2*x)))
    assert A_imm.diff(y) == ImmutableDenseMatrix(((0, 0, 0), (1, 0, 0), (0, 0, 0)))

    assert diff(A_imm, x) == ImmutableDenseMatrix(((0, 0, 1), (0, 0, 0), (0, 0, 2*x)))
    assert diff(A_imm, y) == ImmutableDenseMatrix(((0, 0, 0), (1, 0, 0), (0, 0, 0)))

    # 使用 evaluate=False 测试 diff 函数的效果
    assert A.diff(x, evaluate=False) == ArrayDerivative(A, x, evaluate=False)
    assert diff(A, x, evaluate=False) == ArrayDerivative(A, x, evaluate=False)


def test_diff_by_matrix():
    A = MutableDenseMatrix([[x, y], [z, t]])

    # 测试 Matrix 对象相对于自身的偏导数
    assert A.diff(A) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                               [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    assert diff(A, A) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                                [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    A_imm = A.as_immutable()

    # 测试不可变 Matrix 对象相对于自身的偏导数
    assert A_imm.diff(A_imm) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                                       [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    assert diff(A_imm, A_imm) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                                       [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    # 测试常数矩阵的偏导数
    assert A.diff(a) == MutableDenseMatrix([[0, 0], [0, 0]])

    B = ImmutableDenseMatrix([a, b])

    # 测试 Matrix 对象相对于另一个 Matrix 对象的偏导数
    assert A.diff(B) == Array.zeros(2, 1, 2, 2)
    assert A.diff(A) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                               [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    # 使用元组进行 diff 测试
    dB = B.diff([[a, b]])
    assert dB.shape == (2, 2, 1)
    assert dB == Array([[[1], [0]], [[0], [1]]])

    f = Function("f")
    fxyz = f(x, y, z)

    # 测试函数 f(x, y, z) 对 (x, y, z) 的偏导数
    assert fxyz.diff([[x, y, z]]) == Array([fxyz.diff(x), fxyz.diff(y), fxyz.diff(z)])
    # 断言：验证 fxyz 函数对 ([x, y, z], 2) 的二阶偏导数计算是否正确
    assert fxyz.diff(([x, y, z], 2)) == Array([
        [fxyz.diff(x, 2), fxyz.diff(x, y), fxyz.diff(x, z)],
        [fxyz.diff(x, y), fxyz.diff(y, 2), fxyz.diff(y, z)],
        [fxyz.diff(x, z), fxyz.diff(z, y), fxyz.diff(z, 2)],
    ])

    # 创建表达式 sin(x)*exp(y)
    expr = sin(x)*exp(y)
    # 断言：验证对表达式 expr 求以 [x, y] 为变量的偏导数是否正确
    assert expr.diff([[x, y]]) == Array([cos(x)*exp(y), sin(x)*exp(y)])
    # 断言：验证对表达式 expr 求 y 方向上的偏导数，使用 ((x, y),) 表示，是否正确
    assert expr.diff(y, ((x, y),)) == Array([cos(x)*exp(y), sin(x)*exp(y)])
    # 断言：验证对表达式 expr 求 x 方向上的偏导数，使用 ((x, y),) 表示，是否正确
    assert expr.diff(x, ((x, y),)) == Array([-sin(x)*exp(y), cos(x)*exp(y)])
    # 断言：验证对表达式 expr 求二阶混合偏导数，使用 ((y, x),) 表示，是否正确
    assert expr.diff(((y, x),), [[x, y]]) == Array([[cos(x)*exp(y), -sin(x)*exp(y)], [sin(x)*exp(y), cos(x)*exp(y)]])

    # 测试不同的表示方式：

    # 断言：验证 fxyz 对 x 求二阶偏导，再对 y 求一阶偏导，再对 x 求一阶偏导，是否等于 fxyz 对 ((x, y, z),) 求三阶偏导的第 [0, 1, 0] 项
    assert fxyz.diff(x).diff(y).diff(x) == fxyz.diff(((x, y, z),), 3)[0, 1, 0]
    # 断言：验证 fxyz 对 z 求二阶偏导，再对 y 求一阶偏导，再对 x 求一阶偏导，是否等于 fxyz 对 ((x, y, z),) 求三阶偏导的第 [2, 1, 0] 项
    assert fxyz.diff(z).diff(y).diff(x) == fxyz.diff(((x, y, z),), 3)[2, 1, 0]
    # 断言：验证 fxyz 对 [[x, y, z]] 求一阶偏导，再对 ((z, y, x),) 表示的顺序进行偏导，是否等于对应的偏导数矩阵
    assert fxyz.diff([[x, y, z]], ((z, y, x),)) == Array([[fxyz.diff(i).diff(j) for i in (x, y, z)] for j in (z, y, x)])

    # 断言：验证标量对矩阵求导后是否仍然是矩阵：
    # 对 x 求 x 方向上的导数，结果应为 ImmutableDenseMatrix 类型的矩阵
    res = x.diff(Matrix([[x, y]]))
    assert isinstance(res, ImmutableDenseMatrix)
    # 验证结果是否等于 Matrix([[1, 0]])
    assert res == Matrix([[1, 0]])
    # 对 x^3 求 x 方向上的导数，结果应为 ImmutableDenseMatrix 类型的矩阵
    res = (x**3).diff(Matrix([[x, y]]))
    assert isinstance(res, ImmutableDenseMatrix)
    # 验证结果是否等于 Matrix([[3*x**2, 0]])
    assert res == Matrix([[3*x**2, 0]])
def test_getattr():
    # 创建一个 3x3 的矩阵 A，其中包含数值和符号变量
    A = Matrix(((1, 4, x), (y, 2, 4), (10, 5, x**2 + 1)))
    # 测试访问不存在的属性，预期引发 AttributeError 异常
    raises(AttributeError, lambda: A.nonexistantattribute)
    # 使用 getattr() 动态调用 A 对象的 'diff' 方法，对变量 x 求导数
    assert getattr(A, 'diff')(x) == Matrix(((0, 0, 1), (0, 0, 0), (0, 0, 2*x)))


def test_hessenberg():
    # 创建一个 3x3 的矩阵 A
    A = Matrix([[3, 4, 1], [2, 4, 5], [0, 1, 2]])
    # 断言 A 是否为上 Hessenberg 矩阵
    assert A.is_upper_hessenberg
    # 对 A 进行转置操作
    A = A.T
    # 断言 A 是否为下 Hessenberg 矩阵
    assert A.is_lower_hessenberg
    # 修改 A 中特定位置的元素
    A[0, -1] = 1
    # 断言 A 是否仍为下 Hessenberg 矩阵（预期为假）
    assert A.is_lower_hessenberg is False

    # 创建一个不符合上 Hessenberg 矩阵条件的矩阵 A
    A = Matrix([[3, 4, 1], [2, 4, 5], [3, 1, 2]])
    # 断言 A 不是上 Hessenberg 矩阵
    assert not A.is_upper_hessenberg

    # 创建一个 5x2 的零矩阵 A
    A = zeros(5, 2)
    # 断言 A 是上 Hessenberg 矩阵
    assert A.is_upper_hessenberg


def test_cholesky():
    # 测试非方阵输入的 cholesky 分解，预期引发 NonSquareMatrixError 异常
    raises(NonSquareMatrixError, lambda: Matrix((1, 2)).cholesky())
    # 测试非正定矩阵的 cholesky 分解，预期引发 ValueError 异常
    raises(ValueError, lambda: Matrix(((1, 2), (3, 4))).cholesky())
    raises(ValueError, lambda: Matrix(((5 + I, 0), (0, 1))).cholesky())
    raises(ValueError, lambda: Matrix(((1, 5), (5, 1))).cholesky())
    raises(ValueError, lambda: Matrix(((1, 2), (3, 4))).cholesky(hermitian=False))
    # 断言对非正定矩阵进行 cholesky 分解的正确性
    assert Matrix(((5 + I, 0), (0, 1))).cholesky(hermitian=False) == Matrix([[sqrt(5 + I), 0], [0, 1]])

    # 创建一个符合条件的矩阵 A，进行 cholesky 分解
    A = Matrix(((1, 5), (5, 1)))
    L = A.cholesky(hermitian=False)
    # 断言 cholesky 分解结果的正确性
    assert L == Matrix([[1, 0], [5, 2*sqrt(6)*I]])
    assert L*L.T == A

    # 创建一个正定矩阵 A，进行 cholesky 分解
    A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L = A.cholesky()
    # 断言 cholesky 分解结果的正确性
    assert L * L.T == A
    assert L.is_lower
    assert L == Matrix([[5, 0, 0], [3, 3, 0], [-1, 1, 3]])

    # 创建一个符合条件的复数矩阵 A，进行 cholesky 分解
    A = Matrix(((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11)))
    assert A.cholesky().expand() == Matrix(((2, 0, 0), (I, 1, 0), (1 - I, 0, 3)))

    # 测试稀疏矩阵的 cholesky 分解，预期引发 NonSquareMatrixError 异常
    raises(NonSquareMatrixError, lambda: SparseMatrix((1, 2)).cholesky())
    raises(ValueError, lambda: SparseMatrix(((1, 2), (3, 4))).cholesky())
    raises(ValueError, lambda: SparseMatrix(((5 + I, 0), (0, 1))).cholesky())
    raises(ValueError, lambda: SparseMatrix(((1, 5), (5, 1))).cholesky())
    raises(ValueError, lambda: SparseMatrix(((1, 2), (3, 4))).cholesky(hermitian=False))
    assert SparseMatrix(((5 + I, 0), (0, 1))).cholesky(hermitian=False) == Matrix([[sqrt(5 + I), 0], [0, 1]])

    # 创建一个符合条件的稀疏矩阵 A，进行 cholesky 分解
    A = SparseMatrix(((1, 5), (5, 1)))
    L = A.cholesky(hermitian=False)
    assert L == Matrix([[1, 0], [5, 2*sqrt(6)*I]])
    assert L*L.T == A

    # 创建一个正定稀疏矩阵 A，进行 cholesky 分解
    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L = A.cholesky()
    assert L * L.T == A
    assert L.is_lower
    assert L == Matrix([[5, 0, 0], [3, 3, 0], [-1, 1, 3]])

    # 创建一个符合条件的复数稀疏矩阵 A，进行 cholesky 分解
    A = SparseMatrix(((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11)))
    assert A.cholesky() == Matrix(((2, 0, 0), (I, 1, 0), (1 - I, 0, 3)))


def test_matrix_norm():
    # 向量测试
    # 创建一个带有实数符号的符号变量 x
    x = Symbol('x', real=True)
    # 创建一个向量 v
    v = Matrix([cos(x), sin(x)])
    # 断言向量 v 的 L2 范数为 1
    assert trigsimp(v.norm(2)) == 1
    # 断言向量 v 的 L10 范数的计算结果
    assert v.norm(10) == Pow(cos(x)**10 + sin(x)**10, Rational(1, 10))

    # 行向量测试
    A = Matrix([[5, Rational(3, 2)]])
    # 断言行向量 A 的默认范数计算结果
    assert A.norm() == Pow(25 + Rational(9, 4), S.Half)
    # 断言行向量 A 的无穷范数计算结果
    assert A.norm(oo) == max(A)
    # 断言行向量 A 的负无穷范数计算结果
    assert A.norm(-oo) == min(A)

    # 矩阵测试
    # 下面的矩阵测试代码未完整给出，需要在后续添加
    # Intuitive test
    # 创建一个2x2的矩阵A，内容为[[1, 1], [1, 1]]
    A = Matrix([[1, 1], [1, 1]])
    # 断言矩阵A的2-范数为2
    assert A.norm(2) == 2
    # 断言矩阵A的-2-范数为0
    assert A.norm(-2) == 0
    # 断言矩阵A的Frobenius范数为2
    assert A.norm('frobenius') == 2
    # 断言单位矩阵eye(10)的2-范数和-2-范数均为1
    assert eye(10).norm(2) == eye(10).norm(-2) == 1
    # 断言矩阵A的oo-范数为2
    assert A.norm(oo) == 2

    # Test with Symbols and more complex entries
    # 创建一个包含符号和更复杂元素的2x2矩阵A
    A = Matrix([[3, y, y], [x, S.Half, -pi]])
    # 断言矩阵A的Frobenius范数等于给定的数学表达式
    assert (A.norm('fro')
           == sqrt(Rational(37, 4) + 2*abs(y)**2 + pi**2 + x**2))

    # Check non-square
    # 创建一个非方阵A
    A = Matrix([[1, 2, -3], [4, 5, Rational(13, 2)]])
    # 断言矩阵A的2-范数等于给定的数学表达式
    assert A.norm(2) == sqrt(Rational(389, 8) + sqrt(78665)/8)
    # 断言矩阵A的-2-范数为0
    assert A.norm(-2) is S.Zero
    # 断言矩阵A的Frobenius范数等于给定的数学表达式
    assert A.norm('frobenius') == sqrt(389)/2

    # Test properties of matrix norms
    # https://en.wikipedia.org/wiki/Matrix_norm#Definition
    # Two matrices
    # 创建四个矩阵A、B、C、D，每个矩阵包含不同的元素
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 5], [-2, 2]])
    C = Matrix([[0, -I], [I, 0]])
    D = Matrix([[1, 0], [0, -1]])
    # 将这些矩阵放入列表L中
    L = [A, B, C, D]
    # 创建一个实数符号alpha
    alpha = Symbol('alpha', real=True)

    for order in ['fro', 2, -2]:
        # Zero Check
        # 断言3x3零矩阵的给定范数为零
        assert zeros(3).norm(order) is S.Zero
        # Check Triangle Inequality for all Pairs of Matrices
        # 遍历矩阵列表L中的所有矩阵，检查三角不等式是否成立
        for X in L:
            for Y in L:
                dif = (X.norm(order) + Y.norm(order) -
                    (X + Y).norm(order))
                # 断言差值dif大于等于0
                assert (dif >= 0)
        # Scalar multiplication linearity
        # 遍历矩阵列表L中的所有矩阵，检查标量乘法线性性质
        for M in [A, B, C, D]:
            dif = simplify((alpha*M).norm(order) -
                    abs(alpha) * M.norm(order))
            # 断言差值dif为0
            assert dif == 0

    # Test Properties of Vector Norms
    # https://en.wikipedia.org/wiki/Vector_norm
    # Two column vectors
    # 创建五个列向量a、b、c、d、e，每个向量包含不同的元素
    a = Matrix([1, 1 - 1*I, -3])
    b = Matrix([S.Half, 1*I, 1])
    c = Matrix([-1, -1, -1])
    d = Matrix([3, 2, I])
    e = Matrix([Integer(1e2), Rational(1, 1e2), 1])
    # 将这些向量放入列表L中
    L = [a, b, c, d, e]
    # 创建一个实数符号alpha
    alpha = Symbol('alpha', real=True)

    for order in [1, 2, -1, -2, S.Infinity, S.NegativeInfinity, pi]:
        # Zero Check
        # 如果范数大于0，则断言零向量的给定范数为零
        if order > 0:
            assert Matrix([0, 0, 0]).norm(order) is S.Zero
        # Triangle inequality on all pairs
        # 如果范数大于等于1，遍历向量列表L中的所有向量，检查三角不等式是否成立
        if order >= 1:
            for X in L:
                for Y in L:
                    dif = (X.norm(order) + Y.norm(order) -
                        (X + Y).norm(order))
                    assert simplify(dif >= 0) is S.true
        # Linear to scalar multiplication
        # 如果范数是1、2、-1、-2、无穷大、负无穷大，则遍历向量列表L中的所有向量，检查标量乘法线性性质
        if order in [1, 2, -1, -2, S.Infinity, S.NegativeInfinity]:
            for X in L:
                dif = simplify((alpha*X).norm(order) -
                    (abs(alpha) * X.norm(order)))
                # 断言差值dif为0
                assert dif == 0

    # ord=1
    # 创建一个3x3的矩阵M
    M = Matrix(3, 3, [1, 3, 0, -2, -1, 0, 3, 9, 6])
    # 断言矩阵M的1-范数为给定的数学表达式结果13
    assert M.norm(1) == 13
# 测试条件数函数
def test_condition_number():
    # 定义实数符号变量 x
    x = Symbol('x', real=True)
    # 创建一个 3x3 的单位矩阵 A
    A = eye(3)
    # 修改 A 的第一行第一列为 10
    A[0, 0] = 10
    # 修改 A 的第三行第三列为 1/10
    A[2, 2] = Rational(1, 10)
    # 断言 A 的条件数为 100
    assert A.condition_number() == 100

    # 将 A 的第二行第二列设为变量 x
    A[1, 1] = x
    # 断言 A 的条件数为 Max(10, |x|) / Min(1/10, |x|)
    assert A.condition_number() == Max(10, Abs(x)) / Min(Rational(1, 10), Abs(x))

    # 创建一个旋转矩阵 M
    M = Matrix([[cos(x), sin(x)], [-sin(x), cos(x)]])
    # 计算旋转矩阵 M 的条件数
    Mc = M.condition_number()
    # 断言对于指定的一组值，浮点数 1.0 与 Mc 在数值上等价
    assert all(Float(1.).epsilon_eq(Mc.subs(x, val).evalf()) for val in
        [Rational(1, 5), S.Half, Rational(1, 10), pi/2, pi, pi*Rational(7, 4) ])

    # issue 10782
    # 断言空矩阵的条件数为 0
    assert Matrix([]).condition_number() == 0


# 测试矩阵相等性
def test_equality():
    # 创建矩阵 A 和 B
    A = Matrix(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    B = Matrix(((9, 8, 7), (6, 5, 4), (3, 2, 1)))
    # 断言 A 与 A 的切片相等
    assert A == A[:, :]
    # 断言 A 与 A 的切片不不相等
    assert not A != A[:, :]
    # 断言 A 与 B 不相等
    assert not A == B
    # 断言 A 与 B 不相等
    assert A != B
    # 断言 A 不等于整数 10
    assert A != 10
    # 断言 A 不等于整数 10
    assert not A == 10

    # SparseMatrix 可以与 Matrix 相等
    C = SparseMatrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    D = Matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    # 断言 C 与 D 相等
    assert C == D
    # 断言 C 与 D 不不相等
    assert not C != D


# 测试矩阵归一化
def test_normalized():
    # 断言矩阵 [3, 4] 归一化后的结果
    assert Matrix([3, 4]).normalized() == \
        Matrix([Rational(3, 5), Rational(4, 5)])

    # 零向量的特殊情况
    # 断言零向量归一化后结果仍为零向量
    assert Matrix([0, 0, 0]).normalized() == Matrix([0, 0, 0])

    # 机器精度误差截断的特殊情况
    m = Matrix([0,0,1.e-100])
    assert m.normalized(
    iszerofunc=lambda x: x.evalf(n=10, chop=True).is_zero
    ) == Matrix([0, 0, 0])


# 测试打印非零元素
def test_print_nonzero():
    # 捕获 eye(3) 打印非零元素的输出
    assert capture(lambda: eye(3).print_nonzero()) == \
        '[X  ]\n[ X ]\n[  X]\n'
    # 捕获带有自定义符号 '.' 的 eye(3) 打印非零元素的输出
    assert capture(lambda: eye(3).print_nonzero('.')) == \
        '[.  ]\n[ . ]\n[  .]\n'


# 测试单位矩阵和零矩阵
def test_zeros_eye():
    # 断言 Matrix 类的单位矩阵等于 eye(3)
    assert Matrix.eye(3) == eye(3)
    # 断言 Matrix 类的零矩阵等于 zeros(3)
    assert Matrix.zeros(3) == zeros(3)
    # 断言 ones(3, 4) 等于 Matrix(3, 4, [1]*12)
    assert ones(3, 4) == Matrix(3, 4, [1]*12)

    # 对所有类进行迭代，验证单位矩阵和零矩阵的特性
    i = Matrix([[1, 0], [0, 1]])
    z = Matrix([[0, 0], [0, 0]])
    for cls in all_classes:
        m = cls.eye(2)
        # 断言 m 与 i 相等
        assert i == m  # 但如果 m 是不可变的，则 m == i 将失败
        # 断言 m 与 eye(2, cls=cls) 相等
        assert i == eye(2, cls=cls)
        # 断言 m 的类型为 cls
        assert type(m) == cls
        m = cls.zeros(2)
        # 断言 m 与 z 相等
        assert z == m
        # 断言 m 与 zeros(2, cls=cls) 相等
        assert z == zeros(2, cls=cls)
        # 断言 m 的类型为 cls
        assert type(m) == cls


# 测试矩阵是否为零矩阵
def test_is_zero():
    # 断言空矩阵为零矩阵
    assert Matrix().is_zero_matrix
    # 断言全零矩阵为零矩阵
    assert Matrix([[0, 0], [0, 0]]).is_zero_matrix
    # 断言 3x4 的零矩阵为零矩阵
    assert zeros(3, 4).is_zero_matrix
    # 断言单位矩阵不是零矩阵
    assert not eye(3).is_zero_matrix
    # 断言含有符号变量 x 的矩阵的零矩阵属性为 None
    assert Matrix([[x, 0], [0, 0]]).is_zero_matrix == None
    # 断言稀疏矩阵的零矩阵属性为 None
    assert SparseMatrix([[x, 0], [0, 0]]).is_zero_matrix == None
    # 断言不可变矩阵的零矩阵属性为 None
    assert ImmutableMatrix([[x, 0], [0, 0]]).is_zero_matrix == None
    # 断言不可变稀疏矩阵的零矩阵属性为 None
    assert ImmutableSparseMatrix([[x, 0], [0, 0]]).is_zero_matrix == None
    # 断言包含符号变量 x 的矩阵不是零矩阵
    assert Matrix([[x, 1], [0, 0]]).is_zero_matrix == False
    # 创建非零的符号变量 a
    a = Symbol('a', nonzero=True)
    # 断言包含非零符号变量 a 的矩阵不是零矩阵
    assert Matrix([[a, 0], [0, 0]]).is_zero_matrix == False


# 测试旋转矩阵
def test_rotation_matrices():
    # 测试围绕轴旋转的旋转矩阵
    theta = pi/3
    r3_plus = rot_axis3(theta)
    r3_minus = rot_axis3(-theta)
    # 计算绕第一轴旋转 theta 角度后的旋转矩阵
    r1_plus = rot_axis1(theta)
    # 计算绕第一轴逆时针旋转 -theta 角度后的旋转矩阵
    r1_minus = rot_axis1(-theta)
    # 断言旋转矩阵 r3_minus*r3_plus*单位矩阵 等于 单位矩阵
    assert r3_minus*r3_plus*eye(3) == eye(3)
    # 断言旋转矩阵 r2_minus*r2_plus*单位矩阵 等于 单位矩阵
    assert r2_minus*r2_plus*eye(3) == eye(3)
    # 断言旋转矩阵 r1_minus*r1_plus*单位矩阵 等于 单位矩阵
    assert r1_minus*r1_plus*eye(3) == eye(3)

    # 检查旋转矩阵的迹的正确性
    assert r1_plus.trace() == 1 + 2*cos(theta)
    assert r2_plus.trace() == 1 + 2*cos(theta)
    assert r3_plus.trace() == 1 + 2*cos(theta)

    # 检查零角度旋转不改变任何东西
    assert rot_axis1(0) == eye(3)
    assert rot_axis2(0) == eye(3)
    assert rot_axis3(0) == eye(3)

    # 检查左手规则
    # 查看问题编号 #24529
    q1 = Quaternion.from_axis_angle([1, 0, 0], pi / 2)
    q2 = Quaternion.from_axis_angle([0, 1, 0], pi / 2)
    q3 = Quaternion.from_axis_angle([0, 0, 1], pi / 2)
    assert rot_axis1(- pi / 2) == q1.to_rotation_matrix()
    assert rot_axis2(- pi / 2) == q2.to_rotation_matrix()
    assert rot_axis3(- pi / 2) == q3.to_rotation_matrix()

    # 检查右手规则
    assert rot_ccw_axis1(+ pi / 2) == q1.to_rotation_matrix()
    assert rot_ccw_axis2(+ pi / 2) == q2.to_rotation_matrix()
    assert rot_ccw_axis3(+ pi / 2) == q3.to_rotation_matrix()
# 定义一个测试函数 test_DeferredVector，用于测试 DeferredVector 类的功能
def test_DeferredVector():
    # 断言获取 DeferredVector("vector") 的第四个元素的字符串表示应为 "vector[4]"
    assert str(DeferredVector("vector")[4]) == "vector[4]"
    # 断言 sympify(DeferredVector("d")) 的结果与 DeferredVector("d") 相等
    assert sympify(DeferredVector("d")) == DeferredVector("d")
    # 断言访问不存在的 DeferredVector("d") 的负索引会引发 IndexError 异常
    raises(IndexError, lambda: DeferredVector("d")[-1])
    # 断言获取 DeferredVector("d") 的字符串表示应为 "d"
    assert str(DeferredVector("d")) == "d"
    # 断言获取 DeferredVector("test") 的字符串表示应为 "DeferredVector('test')"
    assert repr(DeferredVector("test")) == "DeferredVector('test')"


# 定义一个测试函数 test_DeferredVector_not_iterable，测试 DeferredVector('X') 是否不可迭代
def test_DeferredVector_not_iterable():
    # 断言 DeferredVector('X') 不可迭代
    assert not iterable(DeferredVector('X'))


# 定义一个测试函数 test_DeferredVector_Matrix，测试将 DeferredVector("V") 作为 Matrix 的参数是否会引发 TypeError 异常
def test_DeferredVector_Matrix():
    # 断言将 DeferredVector("V") 作为 Matrix 的参数会引发 TypeError 异常
    raises(TypeError, lambda: Matrix(DeferredVector("V")))


# 定义一个测试函数 test_GramSchmidt，测试 GramSchmidt 函数的各种用例
def test_GramSchmidt():
    R = Rational
    # 创建两个 1x2 的矩阵 m1 和 m2
    m1 = Matrix(1, 2, [1, 2])
    m2 = Matrix(1, 2, [2, 3])
    # 断言对 m1 和 m2 应用 GramSchmidt 正交化后的结果
    assert GramSchmidt([m1, m2]) == \
        [Matrix(1, 2, [1, 2]), Matrix(1, 2, [R(2)/5, R(-1)/5])]
    # 断言对 m1.T 和 m2.T 应用 GramSchmidt 正交化后的结果
    assert GramSchmidt([m1.T, m2.T]) == \
        [Matrix(2, 1, [1, 2]), Matrix(2, 1, [R(2)/5, R(-1)/5])]
    # 断言对输入矩阵 [[3, 1], [2, 2]] 应用 GramSchmidt 正交化后的结果，使用简化模式
    assert GramSchmidt([Matrix([3, 1]), Matrix([2, 2])], True) == [
        Matrix([3*sqrt(10)/10, sqrt(10)/10]),
        Matrix([-sqrt(10)/10, 3*sqrt(10)/10])]
    # 断言对包含 Matrix([[1]]) 的有限集 L 应用 GramSchmidt 正交化后的结果
    L = FiniteSet(Matrix([1]))
    assert GramSchmidt(L) == [Matrix([[1]])]


# 定义一个测试函数 test_casoratian，测试 casoratian 函数的两种用例
def test_casoratian():
    # 断言对于给定的参数 [1, 2, 3, 4] 和 n=1，casoratian 返回值应为 0
    assert casoratian([1, 2, 3, 4], 1) == 0
    # 断言对于给定的参数 [1, 2, 3, 4]、n=1 和 zero=False，casoratian 返回值应为 0
    assert casoratian([1, 2, 3, 4], 1, zero=False) == 0


# 定义一个测试函数 test_zero_dimension_multiply，测试零维矩阵的乘法行为
def test_zero_dimension_multiply():
    # 断言 Matrix()*zeros(0, 3) 的形状应为 (0, 3)
    assert (Matrix()*zeros(0, 3)).shape == (0, 3)
    # 断言 zeros(3, 0)*zeros(0, 3) 的结果应为全零矩阵，形状为 (3, 3)
    assert zeros(3, 0)*zeros(0, 3) == zeros(3, 3)
    # 断言 zeros(0, 3)*zeros(3, 0) 的结果应为空 Matrix()
    assert zeros(0, 3)*zeros(3, 0) == Matrix()


# 定义一个测试函数 test_slice_issue_2884，测试 Matrix 切片的行为
def test_slice_issue_2884():
    # 创建一个 2x2 的矩阵 m，元素为 [0, 1, 2, 3]
    m = Matrix(2, 2, range(4))
    # 断言获取 m 的第一行应为 Matrix([[2, 3]])
    assert m[1, :] == Matrix([[2, 3]])
    # 断言获取 m 的倒数第一行应为 Matrix([[2, 3]])
    assert m[-1, :] == Matrix([[2, 3]])
    # 断言获取 m 的第一列应为 Matrix([[1, 3]]).T
    assert m[:, 1] == Matrix([[1, 3]]).T
    # 断言获取 m 的倒数第一列应为 Matrix([[1, 3]]).T
    assert m[:, -1] == Matrix([[1, 3]]).T
    # 断言访问超出索引范围的行会引发 IndexError 异常
    raises(IndexError, lambda: m[2, :])
    # 断言访问超出索引范围的行和列会引发 IndexError 异常
    raises(IndexError, lambda: m[2, 2])


# 定义一个测试函数 test_slice_issue_3401，测试针对零维矩阵的切片行为
def test_slice_issue_3401():
    # 断言对 zeros(0, 3) 的任何列的切片结果形状应为 (0, 1)
    assert zeros(0, 3)[:, -1].shape == (0, 1)
    # 断言对 zeros(3, 0) 的第一行切片结果应为 Matrix(1, 0, [])
    assert zeros(3, 0)[0, :] == Matrix(1, 0, [])


# 定义一个测试函数 test_copyin，测试矩阵赋值操作的行为
def test_copyin():
    # 创建一个 3x3 的零矩阵 s
    s = zeros(3, 3)
    # 将 s 的第四个元素赋值为 1
    s[3] = 1
    # 断言获取 s 的第一列应为 Matrix([0, 1, 0])
    assert s[:, 0] == Matrix([0, 1, 0])
    # 断言获取 s 的第四个元素应为 1
    assert s[3] == 1
    # 断言获取 s 的第四个元素到第五个元素（实际只有一个元素）应为 [1]
    assert s[3: 4] == [1]
    # 将 s 的第二行第二列的元素赋值为 42
    s[1, 1] = 42
    # 断言获取 s 的第二行第二列的元素应为 42
    assert s[1, 1] == 42
    # 断言获取 s 的第二行第二列到最后一个元素应为 Matrix([[42, 0]])
    assert s[1, 1:] == Matrix([[42, 0]])
    # 将 s 的第二行第二列到最后一个元素赋值为 Matrix([[5, 6]])
    s[1, 1:] = Matrix([[5, 6]])
    # 断言获取 s 的第二行所有元素应为 Matrix([[1, 5, 6]])
    assert s[1, :] == Matrix([[1, 5, 6]])
    # 将 s 的第二行第二列到最后一个元素赋值为 [[42, 43]]
    s[1, 1:] = [[42, 43
    # 断言矩阵 m 的行简化阶梯形式的第一个元素不等于单位矩阵，即矩阵 m 不可逆时也会返回
    assert m.rref()[0] != eye(3)
    # 断言使用符号简化后的矩阵 m 的行简化阶梯形式的第一个元素不等于单位矩阵，确保符号简化不影响结果
    assert m.rref(simplify=signsimp)[0] != eye(3)
    # 断言调用矩阵 m 的逆运算时会引发 ValueError 异常，使用伴随矩阵方法
    raises(ValueError, lambda: m.inv(method="ADJ"))
    # 断言调用矩阵 m 的逆运算时会引发 ValueError 异常，使用高斯消元法
    raises(ValueError, lambda: m.inv(method="GE"))
    # 断言调用矩阵 m 的逆运算时会引发 ValueError 异常，使用 LU 分解方法
    raises(ValueError, lambda: m.inv(method="LU"))
# 定义一个测试函数，用于检查问题 #3959
def test_issue_3959():
    # 创建符号变量 x 和 y
    x, y = symbols('x, y')
    # 创建表达式 e = x * y
    e = x * y
    # 使用替换操作检查表达式 e 中的 x 替换为 Matrix([3, 5, 3]) 的结果是否等于 Matrix([3, 5, 3])*y
    assert e.subs(x, Matrix([3, 5, 3])) == Matrix([3, 5, 3]) * y


# 定义一个测试函数，用于检查问题 #5964
def test_issue_5964():
    # 检查将 Matrix([[1, 2], [3, 4]]) 转换为字符串是否等于 'Matrix([[1, 2], [3, 4]])'
    assert str(Matrix([[1, 2], [3, 4]])) == 'Matrix([[1, 2], [3, 4]])'


# 定义一个测试函数，用于检查问题 #7604
def test_issue_7604():
    # 创建符号变量 x 和 y
    x, y = symbols("x y")
    # 检查将 Matrix([[x, 2*y], [y**2, x + 3]]) 转换为字符串是否等于 'Matrix([...])' 形式的字符串
    assert sstr(Matrix([[x, 2*y], [y**2, x + 3]])) == \
        'Matrix([\n[   x,   2*y],\n[y**2, x + 3]])'


# 定义一个测试函数，用于检查矩阵是否是单位矩阵
def test_is_Identity():
    # 检查 eye(3) 是否是单位矩阵
    assert eye(3).is_Identity
    # 检查 eye(3) 转为不可变形式后是否是单位矩阵
    assert eye(3).as_immutable().is_Identity
    # 检查 zeros(3) 是否是单位矩阵
    assert not zeros(3).is_Identity
    # 检查 ones(3) 是否是单位矩阵
    assert not ones(3).is_Identity
    # 检查 Matrix([[1, 0, 0]]) 是否是单位矩阵，预期结果应为 False
    assert not Matrix([[1, 0, 0]]).is_Identity
    # 检查 SparseMatrix(3,3, {(0,0):1, (1,1):1, (2,2):1}) 是否是单位矩阵
    assert SparseMatrix(3,3, {(0,0):1, (1,1):1, (2,2):1}).is_Identity
    # 检查 SparseMatrix(2,3, range(6)) 是否是单位矩阵，预期结果应为 False
    assert not SparseMatrix(2,3, range(6)).is_Identity
    # 检查 SparseMatrix(3,3, {(0,0):1, (1,1):1}) 是否是单位矩阵，预期结果应为 False
    assert not SparseMatrix(3,3, {(0,0):1, (1,1):1}).is_Identity
    # 检查 SparseMatrix(3,3, {(0,0):1, (1,1):1, (2,2):1, (0,1):2, (0,2):3}) 是否是单位矩阵，预期结果应为 False
    assert not SparseMatrix(3,3, {(0,0):1, (1,1):1, (2,2):1, (0,1):2, (0,2):3}).is_Identity


# 定义一个测试函数，用于检查矩阵点乘操作
def test_dot():
    # 检查 ones(1, 3) 和 ones(3, 1) 的点乘结果是否等于 3
    assert ones(1, 3).dot(ones(3, 1)) == 3
    # 检查 ones(1, 3) 和 [1, 1, 1] 的点乘结果是否等于 3
    assert ones(1, 3).dot([1, 1, 1]) == 3
    # 检查 Matrix([1, 2, 3]) 和 Matrix([1, 2, 3]) 的点乘结果是否等于 14
    assert Matrix([1, 2, 3]).dot(Matrix([1, 2, 3])) == 14
    # 检查 Matrix([1, 2, 3*I]) 和 Matrix([I, 2, 3*I]) 的点乘结果是否等于 -5 + I
    assert Matrix([1, 2, 3*I]).dot(Matrix([I, 2, 3*I])) == -5 + I
    # 检查 Matrix([1, 2, 3*I]) 和 Matrix([I, 2, 3*I]) 的点乘结果是否等于 -5 + I，不使用共轭
    assert Matrix([1, 2, 3*I]).dot(Matrix([I, 2, 3*I]), hermitian=False) == -5 + I
    # 检查 Matrix([1, 2, 3*I]) 和 Matrix([I, 2, 3*I]) 的点乘结果是否等于 13 + I，使用共轭
    assert Matrix([1, 2, 3*I]).dot(Matrix([I, 2, 3*I]), hermitian=True) == 13 + I
    # 检查 Matrix([1, 2, 3*I]) 和 Matrix([I, 2, 3*I]) 的点乘结果是否等于 13 - I，使用物理学约定的共轭
    assert Matrix([1, 2, 3*I]).dot(Matrix([I, 2, 3*I]), hermitian=True, conjugate_convention="physics") == 13 - I
    # 检查 Matrix([1, 2, 3*I]) 和 Matrix([4, 5*I, 6]) 的点乘结果是否等于 4 + 8*I，使用左边共轭
    assert Matrix([1, 2, 3*I]).dot(Matrix([4, 5*I, 6]), hermitian=True, conjugate_convention="left") == 4 - 8*I
    # 检查 Matrix([1, 2, 3*I]) 和 Matrix([4, 5*I, 6]) 的点乘结果是否等于 4 - 8*I，使用右边共轭
    assert Matrix([1, 2, 3*I]).dot(Matrix([4, 5*I, 6]), hermitian=True, conjugate_convention="right") == 4 + 8*I
    # 检查 Matrix([I, 2*I]) 和 Matrix([I, 2*I]) 的点乘结果是否等于 -5，使用左边共轭
    assert Matrix([I, 2*I]).dot(Matrix([I, 2*I]), hermitian=False, conjugate_convention="left") == -5
    # 检查 Matrix([I, 2*I]) 和 Matrix([I, 2*I]) 的点乘结果是否等于 5，使用右边共轭
    assert Matrix([I, 2*I]).dot(Matrix([I, 2*I]), conjugate_convention="left") == 5
    # 检查在设置未知的共轭约定时是否引发 ValueError
    raises(ValueError, lambda: Matrix([1, 2]).dot(Matrix([3, 4]), hermitian=True, conjugate_convention="test"))


# 定义一个测试函数，用于检查双线性形式的对偶
def test_dual():
    # 创建符号变量 B_x, B_y, B_z, E_x, E_y, E_z
    B_x, B_y, B_z, E_x, E_y, E_z = symbols('B_x B_y B_z E_x E_y E_z', real=True)
    # 创建矩阵 F
    F = Matrix((
        (   0,  E_x,  E_y,  E_z),
        (-E_x,    0,  B_z, -B_y),
        (-E_y, -B_z,    0,  B_x),
        (-E_z,  B_y, -B_x,    0)
    ))
    # 创建 F 的对偶矩阵 Fd
    Fd = Matrix((
    # 设置矩阵 m 的左上角元素为 1
    m[0, 0] = 1
    # 断言矩阵 m 不是反对称矩阵，如果是则抛出异常
    assert m.is_anti_symmetric() is False
def test_normalize_sort_diogonalization():
    A = Matrix(((1, 2), (2, 1)))
    # 对矩阵 A 进行对角化，返回 P 和 Q
    P, Q = A.diagonalize(normalize=True)
    # 断言 P*P.T 和 P.T*P 是单位矩阵
    assert P*P.T == P.T*P == eye(P.cols)
    # 对矩阵 A 进行对角化，并排序特征向量
    P, Q = A.diagonalize(normalize=True, sort=True)
    # 断言 P*P.T 和 P.T*P 是单位矩阵
    assert P*P.T == P.T*P == eye(P.cols)
    # 断言 P*Q*P.inv() 等于原始矩阵 A
    assert P*Q*P.inv() == A


def test_issue_5321():
    # 断言在创建矩阵时，如果提供无效参数，会引发 ValueError 异常
    raises(ValueError, lambda: Matrix([[1, 2, 3], Matrix(0, 1, [])]))


def test_issue_5320():
    # 断言 Matrix.hstack(eye(2), 2*eye(2)) 的结果与预期的矩阵相同
    assert Matrix.hstack(eye(2), 2*eye(2)) == Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2]
    ])
    # 断言 Matrix.vstack(eye(2), 2*eye(2)) 的结果与预期的矩阵相同
    assert Matrix.vstack(eye(2), 2*eye(2)) == Matrix([
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2]
    ])
    cls = SparseMatrix
    # 断言使用 SparseMatrix 进行横向堆叠的结果与预期的矩阵相同
    assert cls.hstack(cls(eye(2)), cls(2*eye(2))) == Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2]
    ])


def test_issue_11944():
    A = Matrix([[1]])
    AIm = sympify(A)
    # 断言在使用 Matrix.hstack(AIm, A) 时的结果与预期的矩阵相同
    assert Matrix.hstack(AIm, A) == Matrix([[1, 1]])
    # 断言在使用 Matrix.vstack(AIm, A) 时的结果与预期的矩阵相同
    assert Matrix.vstack(AIm, A) == Matrix([[1], [1]])


def test_cross():
    a = [1, 2, 3]
    b = [3, 4, 5]
    col = Matrix([-2, 4, -2])
    row = col.T

    def test(M, ans):
        # 断言计算的结果 M 等于预期的 ans
        assert ans == M
        # 断言 M 的类型是 cls
        assert type(M) == cls

    for cls in all_classes:
        A = cls(a)
        B = cls(b)
        # 断言向量 A 和 B 的叉乘结果与预期的列向量 col 相同
        test(A.cross(B), col)
        # 断言向量 A 和 B.T 的叉乘结果与预期的列向量 col 相同
        test(A.cross(B.T), col)
        # 断言向量 A.T 和 B.T 的叉乘结果与预期的行向量 row 相同
        test(A.T.cross(B.T), row)
        # 断言向量 A.T 和 B 的叉乘结果与预期的行向量 row 相同
        test(A.T.cross(B), row)
    # 断言尝试对不同形状的矩阵进行叉乘时会引发 ShapeError 异常
    raises(ShapeError, lambda:
        Matrix(1, 2, [1, 1]).cross(Matrix(1, 2, [1, 1])))


def test_hat_vee():
    v1 = Matrix([x, y, z])
    v2 = Matrix([a, b, c])
    # 断言 v1 的帽子操作后与 v1 和 v2 的叉乘结果相同
    assert v1.hat() * v2 == v1.cross(v2)
    # 断言 v1 的帽子操作后是反对称的
    assert v1.hat().is_anti_symmetric()
    # 断言 v1 的帽子操作的逆操作是 v1 本身
    assert v1.hat().vee() == v1


def test_hash():
    for cls in immutable_classes:
        s = {cls.eye(1), cls.eye(1)}
        # 断言不可变类型的单位矩阵在集合中只出现一次
        assert len(s) == 1 and s.pop() == cls.eye(1)
    # issue 3979
    for cls in mutable_classes:
        # 断言可变类型的单位矩阵不是可哈希的对象
        assert not isinstance(cls.eye(1), Hashable)


def test_adjoint():
    dat = [[0, I], [1, 0]]
    ans = Matrix([[0, 1], [-I, 0]])
    for cls in all_classes:
        # 断言对给定数据 dat 创建的矩阵，其共轭转置与预期的矩阵 ans 相同
        assert ans == cls(dat).adjoint()


def test_atoms():
    m = Matrix([[1, 2], [x, 1 - 1/x]])
    # 断言矩阵 m 中的原子元素集合与预期的集合相同
    assert m.atoms() == {S.One,S(2),S.NegativeOne, x}
    # 断言矩阵 m 中类型为 Symbol 的原子元素集合与预期的集合相同
    assert m.atoms(Symbol) == {x}


def test_pinv():
    # 可逆矩阵的广义逆矩阵是其逆矩阵
    A1 = Matrix([[a, b], [c, d]])
    assert simplify(A1.pinv(method="RD")) == simplify(A1.inv())

    # 测试各种矩阵的广义逆矩阵的四个性质
    As = [Matrix([[13, 104], [2212, 3], [-3, 5]]),
          Matrix([[1, 7, 9], [11, 17, 19]]),
          Matrix([a, b])]

    for A in As:
        A_pinv = A.pinv(method="RD")
        AAp = A * A_pinv
        ApA = A_pinv * A
        # 断言 AAp * A 等于 A
        assert simplify(AAp * A) == A
        # 断言 ApA * A_pinv 等于 A_pinv
        assert simplify(ApA * A_pinv) == A_pinv
        # 断言 AAp 的共轭转置等于 AAp
        assert AAp.H == AAp
        # 断言 ApA 的共轭转置等于 ApA
        assert ApA.H == ApA

    # XXX Pinv with diagonalization makes expression too complicated.
    # 对于每个矩阵 A 在列表 As 中进行以下操作
    for A in As:
        # 计算 A 的伪逆，并简化结果
        A_pinv = simplify(A.pinv(method="ED"))
        # 计算 A * A_pinv，并断言其简化后等于 A
        AAp = A * A_pinv
        assert simplify(AAp * A) == A
        # 计算 A_pinv * A，并断言其简化后等于 A_pinv
        ApA = A_pinv * A
        assert simplify(ApA * A_pinv) == A_pinv
        # 断言 AAp 的共轭转置等于自身
        assert AAp.H == AAp
        # 断言 ApA 的共轭转置等于自身
        assert ApA.H == ApA

    # XXX 使用对角化计算伪逆会生成一个过于复杂的表达式，难以简化
    # A1 = Matrix([[a, b], [c, d]])
    # 断言使用对角化计算的伪逆与直接求逆的结果在一个固定的随机点上数值上相等，以进行数值测试
    from sympy.core.numbers import comp
    # 计算 A1 的伪逆使用对角化方法
    q = A1.pinv(method="ED")
    # 计算 A1 的逆
    w = A1.inv()
    # 定义变量替换字典
    reps = {a: -73633, b: 11362, c: 55486, d: 62570}
    # 断言替换后的每对 q 和 w 的数值近似相等
    assert all(
        comp(i.n(), j.n())
        for i, j in zip(q.subs(reps), w.subs(reps))
        )
@slow
# 定义一个测试函数，用于测试在 A.H * A 的对角化失败时伪逆的四个属性。
def test_pinv_rank_deficient_when_diagonalization_fails():
    # 设置测试用例的矩阵 A
    As = [
        Matrix([
            [61, 89, 55, 20, 71, 0],
            [62, 96, 85, 85, 16, 0],
            [69, 56, 17,  4, 54, 0],
            [10, 54, 91, 41, 71, 0],
            [ 7, 30, 10, 48, 90, 0],
            [0, 0, 0, 0, 0, 0]])
    ]
    # 遍历每个矩阵 A 进行测试
    for A in As:
        # 计算 A 的伪逆，使用的方法为 "ED"
        A_pinv = A.pinv(method="ED")
        # 计算 A * A_pinv 和 A_pinv * A
        AAp = A * A_pinv
        ApA = A_pinv * A
        # 断言 AAp 的共轭转置等于自身
        assert AAp.H == AAp

        # 定义一个函数用于检查两个矩阵是否非常接近
        def allclose(M1, M2):
            # 找出 M1 中的所有 RootOf 符号，并生成它们的近似值字典
            rootofs = M1.atoms(RootOf)
            rootofs_approx = {r: r.evalf() for r in rootofs}
            # 替换 M1 中的 RootOf 符号为近似值后，计算与 M2 的差异，并进行 evalf 处理
            diff_approx = (M1 - M2).xreplace(rootofs_approx).evalf()
            # 检查差异的每个元素是否小于指定阈值，以判断两个矩阵是否近似相等
            return all(abs(e) < 1e-10 for e in diff_approx)

        # 断言 ApA 的共轭转置与自身非常接近
        assert allclose(ApA.H, ApA)


# 测试一个特定的问题，验证零矩阵与其自身相加的结果是否正确
def test_issue_7201():
    assert ones(0, 1) + ones(0, 1) == Matrix(0, 1, [])
    assert ones(1, 0) + ones(1, 0) == Matrix(1, 0, [])


# 测试不同类型的矩阵构造函数，验证它们是否能正确识别自由符号 x
def test_free_symbols():
    for M in ImmutableMatrix, ImmutableSparseMatrix, Matrix, SparseMatrix:
        assert M([[x], [0]]).free_symbols == {x}


# 测试从 ndarray 创建矩阵的功能，验证其能力和正确性
def test_from_ndarray():
    """See issue 7465."""
    try:
        from numpy import array
    except ImportError:
        skip('NumPy must be available to test creating matrices from ndarrays')

    # 使用 numpy 数组构造矩阵，并断言结果与预期一致
    assert Matrix(array([1, 2, 3])) == Matrix([1, 2, 3])
    assert Matrix(array([[1, 2, 3]])) == Matrix([[1, 2, 3]])
    assert Matrix(array([[1, 2, 3], [4, 5, 6]])) == Matrix([[1, 2, 3], [4, 5, 6]])
    assert Matrix(array([x, y, z])) == Matrix([x, y, z])
    # 断言对于尚未实现的多维数组，会抛出 NotImplementedError 异常
    raises(NotImplementedError,
           lambda: Matrix(array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])))
    # 验证对于不同形式的 numpy 数组，其转换成的矩阵是否正确
    assert Matrix([array([1, 2]), array([3, 4])]) == Matrix([[1, 2], [3, 4]])
    assert Matrix([array([1, 2]), [3, 4]]) == Matrix([[1, 2], [3, 4]])
    assert Matrix([array([]), array([])]) == Matrix([])


# 测试与 numpy 相关的功能，验证对 matrixified NumPy ndarray 和 matrix 的索引行为
def test_17522_numpy():
    from sympy.matrices.common import _matrixify
    try:
        from numpy import array, matrix
    except ImportError:
        skip('NumPy must be available to test indexing matrixified NumPy ndarrays and matrices')

    # 从 ndarray 和 matrix 创建矩阵，验证其索引行为是否符合预期
    m = _matrixify(array([[1, 2], [3, 4]]))
    assert m[3] == 4
    assert list(m) == [1, 2, 3, 4]

    with ignore_warnings(PendingDeprecationWarning):
        m = _matrixify(matrix([[1, 2], [3, 4]]))
    assert m[3] == 4
    assert list(m) == [1, 2, 3, 4]
def test_17522_mpmath():
    # 导入函数_matrixify用于将矩阵化，如果mpmath模块导入失败，则跳过测试
    from sympy.matrices.common import _matrixify
    try:
        # 尝试导入mpmath模块的matrix函数
        from mpmath import matrix
    except ImportError:
        # 如果导入失败，调用skip函数，输出跳过消息
        skip('mpmath must be available to test indexing matrixified mpmath matrices')

    # 调用_matrixify将给定矩阵转换为矩阵化表示，然后进行断言测试
    m = _matrixify(matrix([[1, 2], [3, 4]]))
    assert m[3] == 4.0
    assert list(m) == [1.0, 2.0, 3.0, 4.0]


def test_17522_scipy():
    # 导入函数_matrixify用于将矩阵化，如果scipy.sparse模块导入失败，则跳过测试
    from sympy.matrices.common import _matrixify
    try:
        # 尝试导入scipy.sparse模块的csr_matrix函数
        from scipy.sparse import csr_matrix
    except ImportError:
        # 如果导入失败，调用skip函数，输出跳过消息
        skip('SciPy must be available to test indexing matrixified SciPy sparse matrices')

    # 调用_matrixify将给定矩阵转换为矩阵化表示，然后进行断言测试
    m = _matrixify(csr_matrix([[1, 2], [3, 4]]))
    assert m[3] == 4
    assert list(m) == [1, 2, 3, 4]


def test_hermitian():
    # 创建一个复数矩阵a
    a = Matrix([[1, I], [-I, 1]])
    # 断言该矩阵是否为埃尔米特矩阵
    assert a.is_hermitian
    # 修改矩阵元素后，再次检查是否为埃尔米特矩阵，预期结果为False
    a[0, 0] = 2*I
    assert a.is_hermitian is False
    # 修改矩阵元素后，再次检查是否为埃尔米特矩阵，预期结果为None
    a[0, 0] = x
    assert a.is_hermitian is None
    # 修改矩阵元素后，再次检查是否为埃尔米特矩阵，预期结果为False
    a[0, 1] = a[1, 0]*I
    assert a.is_hermitian is False


def test_issue_9457_9467_9876():
    # 对于行删除操作(row_del(index))
    M = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    M.row_del(1)
    assert M == Matrix([[1, 2, 3], [3, 4, 5]])
    N = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    N.row_del(-2)
    assert N == Matrix([[1, 2, 3], [3, 4, 5]])
    O = Matrix([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
    O.row_del(-1)
    assert O == Matrix([[1, 2, 3], [5, 6, 7]])
    P = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    # 断言行删除索引超出边界时会抛出IndexError异常
    raises(IndexError, lambda: P.row_del(10))
    Q = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    # 断言行删除负索引超出边界时会抛出IndexError异常
    raises(IndexError, lambda: Q.row_del(-10))

    # 对于列删除操作(col_del(index))
    M = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    M.col_del(1)
    assert M == Matrix([[1, 3], [2, 4], [3, 5]])
    N = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    N.col_del(-2)
    assert N == Matrix([[1, 3], [2, 4], [3, 5]])
    P = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    # 断言列删除索引超出边界时会抛出IndexError异常
    raises(IndexError, lambda: P.col_del(10))
    Q = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    # 断言列删除负索引超出边界时会抛出IndexError异常
    raises(IndexError, lambda: Q.col_del(-10))


def test_issue_9422():
    # 定义符号变量x, y
    x, y = symbols('x y', commutative=False)
    # 定义符号变量a, b
    a, b = symbols('a b')
    # 创建单位矩阵M
    M = eye(2)
    # 创建一个自定义矩阵M1
    M1 = Matrix(2, 2, [x, y, y, z])
    # 进行矩阵乘法和断言测试，预期结果为不相等
    assert y*x*M != x*y*M
    # 进行矩阵乘法和断言测试，预期结果为相等
    assert b*a*M == a*b*M
    # 进行矩阵乘法和断言测试，预期结果为不相等
    assert x*M1 != M1*x
    # 进行矩阵乘法和断言测试，预期结果为相等
    assert a*M1 == M1*a
    # 进行矩阵乘法和断言测试，预期结果为相等
    assert y*x*M == Matrix([[y*x, 0], [0, y*x]])


def test_issue_10770():
    # 创建一个空矩阵M
    M = Matrix([])
    # 定义测试数据集
    a = ['col_insert', 'row_join'], Matrix([9, 6, 3])
    b = ['row_insert', 'col_join'], a[1].T
    c = ['row_insert', 'col_insert'], Matrix([[1, 2], [3, 4]])
    # 遍历数据集
    for ops, m in (a, b, c):
        for op in ops:
            # 获取矩阵M上的操作函数
            f = getattr(M, op)
            # 执行操作并进行断言测试，确保操作后的结果与预期相同且对象不同
            new = f(m) if 'join' in op else f(42, m)
            assert new == m and id(new) != id(m)


def test_issue_10658():
    # 创建一个矩阵A
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 断言提取部分行列后的矩阵是否符合预期
    assert A.extract([0, 1, 2], [True, True, False]) == \
        Matrix([[1, 2], [4, 5], [7, 8]])
    # 断言提取部分行列后的矩阵是否符合预期
    assert A.extract([0, 1, 2], [True, False, False]) == Matrix([[1], [4], [7]])
    # 断言：验证矩阵 A 对象的数据提取功能是否正确
    assert A.extract([True, False, False], [0, 1, 2]) == Matrix([[1, 2, 3]])
    # 断言：验证矩阵 A 对象从指定行和列提取数据的准确性
    assert A.extract([True, False, True], [0, 1, 2]) == \
        Matrix([[1, 2, 3], [7, 8, 9]])
    # 断言：验证矩阵 A 对象提取指定行的空矩阵的准确性
    assert A.extract([0, 1, 2], [False, False, False]) == Matrix(3, 0, [])
    # 断言：验证矩阵 A 对象提取指定列的空矩阵的准确性
    assert A.extract([False, False, False], [0, 1, 2]) == Matrix(0, 3, [])
    # 断言：验证矩阵 A 对象按指定条件提取数据的准确性
    assert A.extract([True, False, True], [False, True, False]) == \
        Matrix([[2], [8]])
def test_opportunistic_simplification():
    # this test relates to issue #10718, #9480, #11434

    # issue #9480
    # 创建一个包含符号表达式的 2x2 矩阵 m
    m = Matrix([[-5 + 5*sqrt(2), -5], [-5*sqrt(2)/2 + 5, -5*sqrt(2)/2]])
    # 断言矩阵的秩为 1
    assert m.rank() == 1

    # issue #10781
    # 创建一个包含复数的 2x2 矩阵 m
    m = Matrix([[3+3*sqrt(3)*I, -9],[4,-3+3*sqrt(3)*I]])
    # 断言简化后的矩阵的行阶梯形式减去给定矩阵的特定矩阵为零矩阵
    assert simplify(m.rref()[0] - Matrix([[1, -9/(3 + 3*sqrt(3)*I)], [0, 0]])) == zeros(2, 2)

    # issue #11434
    # 定义多个符号变量
    ax, ay, bx, by, cx, cy, dx, dy, ex, ey, t0, t1 = symbols('a_x a_y b_x b_y c_x c_y d_x d_y e_x e_y t_0 t_1')
    # 创建一个 5x5 的符号矩阵 m
    m = Matrix([[ax, ay, ax*t0, ay*t0, 0], [bx, by, bx*t0, by*t0, 0], [cx, cy, cx*t0, cy*t0, 1],
                [dx, dy, dx*t0, dy*t0, 1], [ex, ey, 2*ex*t1-ex*t0, 2*ey*t1-ey*t0, 0]])
    # 断言矩阵 m 的秩为 4
    assert m.rank() == 4


def test_partial_pivoting():
    # example from https://en.wikipedia.org/wiki/Pivot_element
    # partial pivoting with back substitution gives a perfect result
    # naive pivoting give an error ~1e-13, so anything better than
    # 1e-15 is good
    # 创建一个 2x3 的数值矩阵 mm
    mm = Matrix([[0.003, 59.14, 59.17], [5.291, -6.13, 46.78]])
    # 断言矩阵 mm 经过行阶梯形式化后与给定矩阵的差的范数小于 1e-15
    assert (mm.rref()[0] - Matrix([[1.0, 0, 10.0], [0, 1.0, 1.0]])).norm() < 1e-15

    # issue #11549
    # 创建两个数值矩阵 m_mixed 和 m_float
    m_mixed = Matrix([[6e-17, 1.0, 4], [-1.0, 0, 8], [0, 0, 1]])
    m_float = Matrix([[6e-17, 1.0, 4.], [-1.0, 0., 8.], [0., 0., 1.]])
    # 创建一个数值矩阵 m_inv
    m_inv = Matrix([[0, -1.0, 8.0], [1.0, 6.0e-17, -4.0], [0, 0, 1]])
    # 断言矩阵 m_mixed 和 m_float 的逆与预期的矩阵 m_inv 差的范数小于 1e-15
    assert (m_mixed.inv() - m_inv).norm() < 1e-15
    assert (m_float.inv() - m_inv).norm() < 1e-15


def test_iszero_substitution():
    """ When doing numerical computations, all elements that pass
    the iszerofunc test should be set to numerically zero if they
    aren't already. """

    # Matrix from issue #9060
    # 创建一个数值矩阵 m
    m = Matrix([[0.9, -0.1, -0.2, 0], [-0.8, 0.9, -0.4, 0], [-0.1, -0.8, 0.6, 0]])
    # 使用自定义的 iszerofunc 函数对矩阵 m 进行行阶梯形式化
    m_rref = m.rref(iszerofunc=lambda x: abs(x) < 6e-15)[0]
    # 预期的简化后的矩阵 m_correct
    m_correct = Matrix([[1.0, 0, -0.301369863013699, 0], [0, 1.0, -0.712328767123288, 0], [0, 0, 0, 0]])
    # 计算简化后的矩阵 m_rref 与预期矩阵 m_correct 的差
    m_diff = m_rref - m_correct
    # 断言差的范数小于 1e-15
    assert m_diff.norm() < 1e-15
    # 如果零替换没有进行，这个条目将是 -1.11022302462516e-16
    assert m_rref[2, 2] == 0


def test_issue_11238():
    from sympy.geometry.point import Point
    # 计算符号表达式 xx 和 yy
    xx = 8*tan(pi*Rational(13, 45))/(tan(pi*Rational(13, 45)) + sqrt(3))
    yy = (-8*sqrt(3)*tan(pi*Rational(13, 45))**2 + 24*tan(pi*Rational(13, 45)))/(-3 + tan(pi*Rational(13, 45))**2)
    # 创建三个点 p1, p2, p0
    p1 = Point(0, 0)
    p2 = Point(1, -sqrt(3))
    p0 = Point(xx, yy)
    # 创建矩阵 m1, m2, m3
    m1 = Matrix([p1 - simplify(p0), p2 - simplify(p0)])
    m2 = Matrix([p1 - p0, p2 - p0])
    m3 = Matrix([simplify(p1 - p0), simplify(p2 - p0)])

    # This system has expressions which are zero and
    # 定义一个 lambda 函数 Z，用于检查参数 x 的绝对值是否小于 1e-20
    Z = lambda x: abs(x.n()) < 1e-20
    # 使用 assert 断言，验证 m1 的简化秩是否等于 1，并且使用定义的 Z 函数作为零检查函数
    assert m1.rank(simplify=True, iszerofunc=Z) == 1
    # 使用 assert 断言，验证 m2 的简化秩是否等于 1，并且使用定义的 Z 函数作为零检查函数
    assert m2.rank(simplify=True, iszerofunc=Z) == 1
    # 使用 assert 断言，验证 m3 的简化秩是否等于 1，并且使用定义的 Z 函数作为零检查函数
    assert m3.rank(simplify=True, iszerofunc=Z) == 1
def test_as_real_imag():
    # 创建一个 2x2 的矩阵 m1，数据为 [1, 2, 3, 4]
    m1 = Matrix(2, 2, [1, 2, 3, 4])
    # 计算 m1 乘以虚数单位得到 m2
    m2 = m1 * S.ImaginaryUnit
    # 计算 m1 和 m2 的和得到 m3
    m3 = m1 + m2

    # 对所有的类进行迭代
    for kls in all_classes:
        # 将 m3 转换为实部和虚部
        a, b = kls(m3).as_real_imag()
        # 断言实部 a 应该与 m1 相同
        assert list(a) == list(m1)
        # 断言虚部 b 应该与 m1 相同
        assert list(b) == list(m1)


def test_deprecated():
    # 维护对已废弃函数的测试。需要捕获废弃警告。
    # 当废弃功能被移除时，对应的测试也应该被移除。

    # 创建一个 3x3 的矩阵 m
    m = Matrix(3, 3, [0, 1, 0, -4, 4, 0, -2, 1, 2])
    # 计算 m 的 Jordan 标准形和 Jordan 细胞
    P, Jcells = m.jordan_cells()
    # 断言第一个 Jordan 细胞 Jcells[1] 应该与给定的矩阵相同
    assert Jcells[1] == Matrix(1, 1, [2])
    # 断言第二个 Jordan 细胞 Jcells[0] 应该与给定的矩阵相同
    assert Jcells[0] == Matrix(2, 2, [2, 1, 0, 2])


def test_issue_14489():
    from sympy.core.mod import Mod
    # 创建一个列向量 A
    A = Matrix([-1, 1, 2])
    # 创建一个列向量 B
    B = Matrix([10, 20, -15])

    # 断言对 A 取模 3 的结果应该与给定的列向量相同
    assert Mod(A, 3) == Matrix([2, 1, 2])
    # 断言对 B 取模 4 的结果应该与给定的列向量相同
    assert Mod(B, 4) == Matrix([2, 0, 1])


def test_issue_14943():
    # 测试 __array__ 方法是否接受可选的 dtype 参数
    try:
        from numpy import array
    except ImportError:
        skip('NumPy must be available to test creating matrices from ndarrays')

    # 创建一个 2x2 的矩阵 M
    M = Matrix([[1,2], [3,4]])
    # 断言将 M 转换为 numpy 数组，并指定 dtype 为 float 后，其数据类型名称应为 'float64'
    assert array(M, dtype=float).dtype.name == 'float64'


def test_case_6913():
    # 创建一个矩阵符号 m，形状为 1x1
    m = MatrixSymbol('m', 1, 1)
    # 将符号 a 赋值为 m[0, 0] > 0 的布尔表达式
    a = Symbol("a")
    a = m[0, 0] > 0
    # 断言将布尔表达式 a 转换为字符串后应为 'm[0, 0] > 0'
    assert str(a) == 'm[0, 0] > 0'


def test_issue_11948():
    # 创建一个符号矩阵 A，形状为 3x3
    A = MatrixSymbol('A', 3, 3)
    # 创建一个通配符 a
    a = Wild('a')
    # 断言 A 匹配通配符 a 的结果应该是 {a: A}
    assert A.match(a) == {a: A}


def test_gramschmidt_conjugate_dot():
    # 创建两个向量，并组成列表 vecs
    vecs = [Matrix([1, I]), Matrix([1, -I])]
    # 断言对 vecs 中的向量进行正交化后的结果
    assert Matrix.orthogonalize(*vecs) == \
        [Matrix([[1], [I]]), Matrix([[1], [-I]])]

    # 更新 vecs
    vecs = [Matrix([1, I, 0]), Matrix([I, 0, -I])]
    # 断言对更新后的 vecs 中的向量进行正交化后的结果
    assert Matrix.orthogonalize(*vecs) == \
        [Matrix([[1], [I], [0]]), Matrix([[I/2], [S(1)/2], [-I]])]

    # 创建一个矩阵 mat
    mat = Matrix([[1, I], [1, -I]])
    # 计算 mat 的 QR 分解，分别得到 Q 和 R
    Q, R = mat.QRdecomposition()
    # 断言 Q 乘以其共轭转置应该等于单位矩阵
    assert Q * Q.H == Matrix.eye(2)


def test_issue_8207():
    # 创建一个矩阵 a，其元素为符号矩阵符号
    a = Matrix(MatrixSymbol('a', 3, 1))
    # 创建一个矩阵 b，其元素为符号矩阵符号
    b = Matrix(MatrixSymbol('b', 3, 1))
    # 计算 a 和 b 的点积得到 c
    c = a.dot(b)
    # 对 c 求关于 a[0, 0] 的导数得到 d
    d = diff(c, a[0, 0])
    # 对 d 求关于 a[0, 0] 的导数得到 e
    e = diff(d, a[0, 0])
    # 断言 d 应该与 b[0, 0] 相等
    assert d == b[0, 0]
    # 断言 e 应该等于 0
    assert e == 0


def test_func():
    from sympy.simplify.simplify import nthroot

    # 创建一个 2x2 的矩阵 A
    A = Matrix([[1, 2],[0, 3]])
    # 断言 A 应用 sin(x*t) 分析函数后的结果应该与给定的矩阵相同
    assert A.analytic_func(sin(x*t), x) == Matrix([[sin(t), sin(3*t) - sin(t)], [0, sin(3*t)]])

    # 创建一个 2x2 的矩阵 A
    A = Matrix([[2, 1],[1, 2]])
    # 断言将 pi * A / 6 应用 cos(x) 分析函数后的结果应该与给定的矩阵相同
    assert (pi * A / 6).analytic_func(cos(x), x) == Matrix([[sqrt(3)/4, -sqrt(3)/4], [-sqrt(3)/4, sqrt(3)/4]])

    # 断言应该引发 ValueError 异常，因为尝试对 5x5 的零矩阵应用 log(x) 分析函数
    raises(ValueError, lambda : zeros(5).analytic_func(log(x), x))
    # 断言应该引发 ValueError 异常，因为尝试对 A*x 应用 log(x) 分析函数
    raises(ValueError, lambda : (A*x).analytic_func(log(x), x))

    # 创建一个 4x4 的矩阵 A
    A = Matrix([[41, 12],[12, 34]])
    # 断言简化 A 应用 sqrt(x) 分析函数后的结果的平方应该等于 A 本身
    assert simplify(A.analytic_func(sqrt(x), x)**2) == A

    # 创建一个 3x3 的矩阵 A
    A = Matrix([[3, -12, 4], [-1, 0, -2], [-1, 5, -1]])
    # 断言简化 A 应用 nthroot(x, 3) 分析函数后的结果的立方应该等于 A 本身
    assert simplify(A.analytic_func(nthroot(x, 3), x
    # 创建一个4x4的矩阵A，包含特定的元素
    A = Matrix([[2, 0, 0, 0], [1, 2, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3]])
    # 对矩阵A调用analytic_func方法，传入exp(x)和x作为参数，断言结果等于A.exp()
    assert A.analytic_func(exp(x), x) == A.exp()
    
    # 创建另一个4x4的矩阵A，包含特定的元素
    A = Matrix([[0, 2, 1, 6], [0, 0, 1, 2], [0, 0, 0, 3], [0, 0, 0, 0]])
    # 对矩阵A调用analytic_func方法，传入exp(x*t)和x作为参数，断言结果等于expand(simplify((A*t).exp()))
    assert A.analytic_func(exp(x*t), x) == expand(simplify((A*t).exp()))
# 使用装饰器 @skip_under_pyodide 标记的测试函数，用于在 pyodide 环境下跳过不支持线程创建的情况
def test_issue_19809():

    # 定义内部函数 f，验证 _dotprodsimp_state 状态为 None，创建一个单元素矩阵 m，计算 m 乘以自身，然后返回 True
    def f():
        assert _dotprodsimp_state.state == None
        m = Matrix([[1]])
        m = m * m
        return True

    # 使用 dotprodsimp(True) 上下文管理器
    with dotprodsimp(True):
        # 使用 ThreadPoolExecutor 创建线程池 executor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交任务 f 到线程池 executor 中，并获取一个 Future 对象 future
            future = executor.submit(f)
            # 断言任务的执行结果为 True
            assert future.result()


# 测试函数，验证在指定区间内对矩阵 M 进行二重积分的结果是否符合预期
def test_issue_23276():
    M = Matrix([x, y])
    # 断言矩阵 M 在 (x, 0, 1), (y, 0, 1) 区间内的二重积分结果等于给定的矩阵
    assert integrate(M, (x, 0, 1), (y, 0, 1)) == Matrix([
        [S.Half],  # 预期的积分结果的第一行
        [S.Half]])  # 预期的积分结果的第二行
```