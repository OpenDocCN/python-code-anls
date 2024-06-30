# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_commonmatrix.py`

```
#
# 用于测试已弃用的矩阵类。新的测试代码不应添加在这里，而应添加到 test_matrixbase.py 中。
#
# 这整个测试模块和对应的 sympy/matrices/common.py 模块将在将来的版本中被移除。
#
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy

from sympy.assumptions import Q  # 导入 Q 假设系统
from sympy.core.expr import Expr  # 导入基本表达式类 Expr
from sympy.core.add import Add  # 导入加法类 Add
from sympy.core.function import Function  # 导入函数类 Function
from sympy.core.kind import NumberKind, UndefinedKind  # 导入数值和未定义类型类
from sympy.core.numbers import I, Integer, oo, pi, Rational  # 导入复数和常用数值类
from sympy.core.singleton import S  # 导入单例类 S
from sympy.core.symbol import Symbol, symbols  # 导入符号类 Symbol 和符号生成器 symbols
from sympy.functions.elementary.complexes import Abs  # 导入复数函数 Abs
from sympy.functions.elementary.exponential import exp  # 导入指数函数 exp
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数 sqrt
from sympy.functions.elementary.trigonometric import cos, sin  # 导入三角函数 cos 和 sin
from sympy.matrices.exceptions import ShapeError, NonSquareMatrixError  # 导入矩阵异常类
from sympy.matrices.kind import MatrixKind  # 导入矩阵类型类
from sympy.matrices.common import (  # 导入矩阵通用模块中的各个部分
    _MinimalMatrix, _CastableMatrix, MatrixShaping, MatrixProperties,
    MatrixOperations, MatrixArithmetic, MatrixSpecial)
from sympy.matrices.matrices import MatrixCalculus  # 导入矩阵计算类
from sympy.matrices import (  # 导入各种矩阵类和函数
    Matrix, diag, eye, matrix_multiply_elementwise, ones, zeros,
    SparseMatrix, banded, MutableDenseMatrix, MutableSparseMatrix,
    ImmutableDenseMatrix, ImmutableSparseMatrix)
from sympy.polys.polytools import Poly  # 导入多项式工具类 Poly
from sympy.utilities.iterables import flatten  # 导入展开函数 flatten
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray as Array  # 导入数组类 Array

from sympy.abc import x, y, z  # 导入符号变量 x, y, z


def test_matrix_deprecated_isinstance():
    # 测试例如 isinstance(M, MatrixCommon) 仍然在 M 是矩阵时返回 True 的功能，对每个已弃用的矩阵类进行测试。

    from sympy.matrices.common import (  # 导入矩阵通用模块中的各个部分，用于测试
        MatrixRequired, MatrixShaping, MatrixSpecial, MatrixProperties,
        MatrixOperations, MatrixArithmetic, MatrixCommon)
    from sympy.matrices.matrices import (  # 导入矩阵类，用于测试
        MatrixDeterminant, MatrixReductions, MatrixSubspaces, MatrixEigen,
        MatrixCalculus, MatrixDeprecated)
    from sympy import (  # 导入矩阵类，用于测试
        Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)

    all_mixins = (  # 定义包含所有混合类的元组
        MatrixRequired, MatrixShaping, MatrixSpecial, MatrixProperties,
        MatrixOperations, MatrixArithmetic, MatrixCommon,
        MatrixDeterminant, MatrixReductions, MatrixSubspaces, MatrixEigen,
        MatrixCalculus, MatrixDeprecated)
    all_matrices = (  # 定义包含所有矩阵类的元组
        Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)

    Ms = [M([[1, 2], [3, 4]]) for M in all_matrices]  # 创建每种矩阵类的示例列表 Ms
    t = ()  # 创建空元组 t
    # 遍历所有混合类 mixin 列表
    for mixin in all_mixins:
        # 遍历当前类列表 Ms
        for M in Ms:
            # 在警告已弃用的 sympy 上下文中，断言当前类 M 是 mixin 的一个实例
            with warns_deprecated_sympy():
                assert isinstance(M, mixin) is True
        # 在警告已弃用的 sympy 上下文中，断言对象 t 不是 mixin 的实例
        with warns_deprecated_sympy():
            assert isinstance(t, mixin) is False
# 测试用于弃用矩阵类的类。我们使用 warns_deprecated_sympy 函数来抑制弃用警告，因为子类化弃用的
# 类会引发警告。

with warns_deprecated_sympy():
    # 定义一个类 ShapingOnlyMatrix，继承自 _MinimalMatrix、_CastableMatrix 和 MatrixShaping
    class ShapingOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixShaping):
        pass


# 定义一个函数，返回一个 ShapingOnlyMatrix 对象，表示单位矩阵
def eye_Shaping(n):
    return ShapingOnlyMatrix(n, n, lambda i, j: int(i == j))


# 定义一个函数，返回一个 ShapingOnlyMatrix 对象，表示全零矩阵
def zeros_Shaping(n):
    return ShapingOnlyMatrix(n, n, lambda i, j: 0)


with warns_deprecated_sympy():
    # 定义一个类 PropertiesOnlyMatrix，继承自 _MinimalMatrix、_CastableMatrix 和 MatrixProperties
    class PropertiesOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixProperties):
        pass


# 定义一个函数，返回一个 PropertiesOnlyMatrix 对象，表示单位矩阵
def eye_Properties(n):
    return PropertiesOnlyMatrix(n, n, lambda i, j: int(i == j))


# 定义一个函数，返回一个 PropertiesOnlyMatrix 对象，表示全零矩阵
def zeros_Properties(n):
    return PropertiesOnlyMatrix(n, n, lambda i, j: 0)


with warns_deprecated_sympy():
    # 定义一个类 OperationsOnlyMatrix，继承自 _MinimalMatrix、_CastableMatrix 和 MatrixOperations
    class OperationsOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixOperations):
        pass


# 定义一个函数，返回一个 OperationsOnlyMatrix 对象，表示单位矩阵
def eye_Operations(n):
    return OperationsOnlyMatrix(n, n, lambda i, j: int(i == j))


# 定义一个函数，返回一个 OperationsOnlyMatrix 对象，表示全零矩阵
def zeros_Operations(n):
    return OperationsOnlyMatrix(n, n, lambda i, j: 0)


with warns_deprecated_sympy():
    # 定义一个类 ArithmeticOnlyMatrix，继承自 _MinimalMatrix、_CastableMatrix 和 MatrixArithmetic
    class ArithmeticOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixArithmetic):
        pass


# 定义一个函数，返回一个 ArithmeticOnlyMatrix 对象，表示单位矩阵
def eye_Arithmetic(n):
    return ArithmeticOnlyMatrix(n, n, lambda i, j: int(i == j))


# 定义一个函数，返回一个 ArithmeticOnlyMatrix 对象，表示全零矩阵
def zeros_Arithmetic(n):
    return ArithmeticOnlyMatrix(n, n, lambda i, j: 0)


with warns_deprecated_sympy():
    # 定义一个类 SpecialOnlyMatrix，继承自 _MinimalMatrix、_CastableMatrix 和 MatrixSpecial
    class SpecialOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixSpecial):
        pass


with warns_deprecated_sympy():
    # 定义一个类 CalculusOnlyMatrix，继承自 _MinimalMatrix、_CastableMatrix 和 MatrixCalculus
    class CalculusOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixCalculus):
        pass


# 测试 _MinimalMatrix 类的各种功能
def test__MinimalMatrix():
    # 创建一个 _MinimalMatrix 对象 x
    x = _MinimalMatrix(2, 3, [1, 2, 3, 4, 5, 6])
    # 断言对象的行数和列数
    assert x.rows == 2
    assert x.cols == 3
    # 断言对象的指定元素值
    assert x[2] == 3
    assert x[1, 1] == 5
    # 断言对象转换为列表的正确性
    assert list(x) == [1, 2, 3, 4, 5, 6]
    # 断言对象的切片操作
    assert list(x[1, :]) == [4, 5, 6]
    assert list(x[:, 1]) == [2, 5]
    assert list(x[:, :]) == list(x)
    # 断言对象的相等性和复制功能
    assert x[:, :] == x
    assert _MinimalMatrix(x) == x
    # 测试从不同格式创建对象
    assert _MinimalMatrix([[1, 2, 3], [4, 5, 6]]) == x
    assert _MinimalMatrix(([1, 2, 3], [4, 5, 6])) == x
    assert _MinimalMatrix([(1, 2, 3), (4, 5, 6)]) == x
    assert _MinimalMatrix(((1, 2, 3), (4, 5, 6))) == x
    assert not (_MinimalMatrix([[1, 2], [3, 4], [5, 6]]) == x)


# 测试矩阵类的 kind 属性
def test_kind():
    # 断言不同矩阵对象的 kind 属性
    assert Matrix([[1, 2], [3, 4]]).kind == MatrixKind(NumberKind)
    assert Matrix([[0, 0], [0, 0]]).kind == MatrixKind(NumberKind)
    assert Matrix(0, 0, []).kind == MatrixKind(NumberKind)
    assert Matrix([[x]]).kind == MatrixKind(NumberKind)
    assert Matrix([[1, Matrix([[1]])]]).kind == MatrixKind(UndefinedKind)
    assert SparseMatrix([[1]]).kind == MatrixKind(NumberKind)
    assert SparseMatrix([[1, Matrix([[1]])]]).kind == MatrixKind(UndefinedKind)


# 测试 ShapingOnlyMatrix 类的方法
def test_vec():
    # 创建一个 ShapingOnlyMatrix 对象 m
    m = ShapingOnlyMatrix(2, 2, [1, 3, 2, 4])
    # 调用对象的 vec 方法
    m_vec = m.vec()
    # 断言 vec 方法的返回值的列数和内容
    assert m_vec.cols == 1
    for i in range(4):
        assert m_vec[i] == i + 1


# 测试待完成的函数 todok
def test_todok():
    # 待完成，尚未提供实现
    pass
    # 定义符号变量 a, b, c, d
    a, b, c, d = symbols('a:d')
    
    # 创建可变的稠密矩阵 m1，包含符号变量 a, b, c, d
    m1 = MutableDenseMatrix([[a, b], [c, d]])
    
    # 创建不可变的稠密矩阵 m2，包含符号变量 a, b, c, d
    m2 = ImmutableDenseMatrix([[a, b], [c, d]])
    
    # 创建可变的稀疏矩阵 m3，包含符号变量 a, b, c, d
    m3 = MutableSparseMatrix([[a, b], [c, d]])
    
    # 创建不可变的稀疏矩阵 m4，包含符号变量 a, b, c, d
    m4 = ImmutableSparseMatrix([[a, b], [c, d]])
    
    # 断言：确保 m1, m2, m3, m4 都转换为字典表示（todok）后相等，字典内容为矩阵元素的坐标与值
    assert m1.todok() == m2.todok() == m3.todok() == m4.todok() == {(0, 0): a, (0, 1): b, (1, 0): c, (1, 1): d}
def test_tolist():
    # 定义一个包含子列表的列表
    lst = [[S.One, S.Half, x*y, S.Zero], [x, y, z, x**2], [y, -S.One, z*x, 3]]
    # 将子列表展开成一维列表
    flat_lst = [S.One, S.Half, x*y, S.Zero, x, y, z, x**2, y, -S.One, z*x, 3]
    # 使用 flat_lst 初始化 ShapingOnlyMatrix 对象 m
    m = ShapingOnlyMatrix(3, 4, flat_lst)
    # 断言 m 的 tolist 方法返回的结果与 lst 相等
    assert m.tolist() == lst

def test_todod():
    # 初始化 ShapingOnlyMatrix 对象 m
    m = ShapingOnlyMatrix(3, 2, [[S.One, 0], [0, S.Half], [x, 0]])
    # 构建预期的字典结构
    dict = {0: {0: S.One}, 1: {1: S.Half}, 2: {0: x}}
    # 断言 m 的 todod 方法返回的结果与 dict 相等
    assert m.todod() == dict

def test_row_col_del():
    # 初始化 ShapingOnlyMatrix 对象 e
    e = ShapingOnlyMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 测试行删除方法，预期抛出 IndexError
    raises(IndexError, lambda: e.row_del(5))
    raises(IndexError, lambda: e.row_del(-5))
    # 测试列删除方法，预期抛出 IndexError
    raises(IndexError, lambda: e.col_del(5))
    raises(IndexError, lambda: e.col_del(-5))

    # 断言行删除操作的结果与预期的 Matrix 对象相等
    assert e.row_del(2) == e.row_del(-1) == Matrix([[1, 2, 3], [4, 5, 6]])
    # 断言列删除操作的结果与预期的 Matrix 对象相等
    assert e.col_del(2) == e.col_del(-1) == Matrix([[1, 2], [4, 5], [7, 8]])

    # 再次测试行删除操作的结果与预期的 Matrix 对象相等
    assert e.row_del(1) == e.row_del(-2) == Matrix([[1, 2, 3], [7, 8, 9]])
    # 再次测试列删除操作的结果与预期的 Matrix 对象相等
    assert e.col_del(1) == e.col_del(-2) == Matrix([[1, 3], [4, 6], [7, 9]])

def test_get_diag_blocks1():
    # 初始化 Matrix 对象 a, b, c
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    # 断言 a 的 get_diag_blocks 方法返回的结果为包含 a 的列表
    assert a.get_diag_blocks() == [a]
    # 断言 b 的 get_diag_blocks 方法返回的结果为包含 b 的列表
    assert b.get_diag_blocks() == [b]
    # 断言 c 的 get_diag_blocks 方法返回的结果为包含 c 的列表
    assert c.get_diag_blocks() == [c]

def test_get_diag_blocks2():
    # 初始化 Matrix 对象 a, b, c
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    # 使用 diag 函数创建四个对角矩阵，并转换为 ShapingOnlyMatrix 对象 A, B, C, D
    A, B, C, D = diag(a, b, b), diag(a, b, c), diag(a, c, b), diag(c, c, b)
    A = ShapingOnlyMatrix(A.rows, A.cols, A)
    B = ShapingOnlyMatrix(B.rows, B.cols, B)
    C = ShapingOnlyMatrix(C.rows, C.cols, C)
    D = ShapingOnlyMatrix(D.rows, D.cols, D)

    # 断言 A 的 get_diag_blocks 方法返回的结果为包含 a, b, b 的列表
    assert A.get_diag_blocks() == [a, b, b]
    # 断言 B 的 get_diag_blocks 方法返回的结果为包含 a, b, c 的列表
    assert B.get_diag_blocks() == [a, b, c]
    # 断言 C 的 get_diag_blocks 方法返回的结果为包含 a, c, b 的列表
    assert C.get_diag_blocks() == [a, c, b]
    # 断言 D 的 get_diag_blocks 方法返回的结果为包含 c, c, b 的列表
    assert D.get_diag_blocks() == [c, c, b]

def test_shape():
    # 初始化 ShapingOnlyMatrix 对象 m
    m = ShapingOnlyMatrix(1, 2, [0, 0])
    # 断言 m 的 shape 属性为 (1, 2)
    assert m.shape == (1, 2)

def test_reshape():
    # 初始化 eye_Shaping(3) 对象 m0
    m0 = eye_Shaping(3)
    # 断言 m0 调用 reshape(1, 9) 方法返回的结果与预期的 Matrix 对象相等
    assert m0.reshape(1, 9) == Matrix(1, 9, (1, 0, 0, 0, 1, 0, 0, 0, 1))
    # 初始化 ShapingOnlyMatrix 对象 m1
    m1 = ShapingOnlyMatrix(3, 4, lambda i, j: i + j)
    # 断言 m1 调用 reshape(4, 3) 方法返回的结果与预期的 Matrix 对象相等
    assert m1.reshape(4, 3) == Matrix(((0, 1, 2), (3, 1, 2), (3, 4, 2), (3, 4, 5)))
    # 断言 m1 调用 reshape(2, 6) 方法返回的结果与预期的 Matrix 对象相等
    assert m1.reshape(2, 6) == Matrix(((0, 1, 2, 3, 1, 2), (3, 4, 2, 3, 4, 5)))

def test_row_col():
    # 初始化 ShapingOnlyMatrix 对象 m
    m = ShapingOnlyMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 断言 m 的 row(0) 方法返回的结果与预期的 Matrix 对象相等
    assert m.row(0) == Matrix(1, 3, [1, 2, 3])
    # 断言 m 的 col(0) 方法返回的结果与预期的 Matrix 对象相等
    assert m.col(0) == Matrix(3, 1, [1, 4, 7])

def test_row_join():
    # 断言调用 eye_Shaping(3) 的 row_join 方法后返回的结果与预期的 Matrix 对象相等
    assert eye_Shaping(3).row_join(Matrix([7, 7, 7])) == \
           Matrix([[1, 0, 0, 7],
                   [0, 1, 0, 7],
                   [0, 0, 1, 7]])

def test_col_join():
    # 断言调用 eye_Shaping(3) 的 col_join 方法后返回的结果与预期的 Matrix 对象相等
    assert eye_Shaping(3).col_join(Matrix([[7, 7, 7]])) == \
           Matrix([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [7, 7, 7]])

def test_row_insert():
    # 初始化 Matrix 对象 r4
    r4 = Matrix([[4, 4, 4]])
    # 对于 i 从 -4 到 4 的循环，执行以下操作：
    l = [1, 0, 0]  # 创建包含元素 [1, 0, 0] 的列表 l
    l.insert(i, 4)  # 在列表 l 的第 i 个位置插入数字 4
    # 使用 flatten 函数处理 eye_Shaping(3) 对象的结果，确保插入 r4 后的第 0 列（col(0)）展平后与列表 l 相等
    assert flatten(eye_Shaping(3).row_insert(i, r4).col(0).tolist()) == l
# 定义测试函数 `test_col_insert`
def test_col_insert():
    # 创建一个 4x1 的矩阵 c4 = [4, 4, 4]
    c4 = Matrix([4, 4, 4])
    # 对于范围从 -4 到 4 的每个整数 i
    for i in range(-4, 5):
        # 创建一个包含三个零的列表 l = [0, 0, 0]
        l = [0, 0, 0]
        # 在列表 l 的第 i 个位置插入数字 4
        l.insert(i, 4)
        # 断言将一个 3x3 的零矩阵调整为列插入矩阵后取第 0 行，并展平成列表后，其结果等于 l
        assert flatten(zeros_Shaping(3).col_insert(i, c4).row(0).tolist()) == l
    # 断言调用 issue 13643，验证将一个 6x6 的单位矩阵在第 3 列插入一个 6x2 的矩阵的结果
    assert eye_Shaping(6).col_insert(3, Matrix([[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])) == \
           Matrix([[1, 0, 0, 2, 2, 0, 0, 0],
                   [0, 1, 0, 2, 2, 0, 0, 0],
                   [0, 0, 1, 2, 2, 0, 0, 0],
                   [0, 0, 0, 2, 2, 1, 0, 0],
                   [0, 0, 0, 2, 2, 0, 1, 0],
                   [0, 0, 0, 2, 2, 0, 0, 1]])

# 定义测试函数 `test_extract`
def test_extract():
    # 创建一个 4x3 的矩阵 m，其元素为按照 lambda 函数生成的值
    m = ShapingOnlyMatrix(4, 3, lambda i, j: i*3 + j)
    # 断言从矩阵 m 中提取指定行和列的子矩阵
    assert m.extract([0, 1, 3], [0, 1]) == Matrix(3, 2, [0, 1, 3, 4, 9, 10])
    # 断言从矩阵 m 中提取指定行和列的子矩阵
    assert m.extract([0, 3], [0, 0, 2]) == Matrix(2, 3, [0, 0, 2, 9, 9, 11])
    # 断言从矩阵 m 中提取所有行和列的子矩阵等于矩阵 m 本身
    assert m.extract(range(4), range(3)) == m
    # 断言提取超出范围的行和列将引发 IndexError 异常
    raises(IndexError, lambda: m.extract([4], [0]))
    raises(IndexError, lambda: m.extract([0], [3]))

# 定义测试函数 `test_hstack`
def test_hstack():
    # 创建一个 4x3 的矩阵 m，其元素为按照 lambda 函数生成的值
    m = ShapingOnlyMatrix(4, 3, lambda i, j: i*3 + j)
    # 创建一个 3x4 的矩阵 m2，其元素为按照 lambda 函数生成的值
    m2 = ShapingOnlyMatrix(3, 4, lambda i, j: i*3 + j)
    # 断言矩阵 m 与其自身水平堆叠后的结果等于自身
    assert m == m.hstack(m)
    # 断言多个矩阵水平堆叠后的结果等于给定的预期矩阵
    assert m.hstack(m, m, m) == ShapingOnlyMatrix.hstack(m, m, m) == Matrix([
                [0,  1,  2, 0,  1,  2, 0,  1,  2],
                [3,  4,  5, 3,  4,  5, 3,  4,  5],
                [6,  7,  8, 6,  7,  8, 6,  7,  8],
                [9, 10, 11, 9, 10, 11, 9, 10, 11]])
    # 断言尝试堆叠不兼容形状的矩阵将引发 ShapeError 异常
    raises(ShapeError, lambda: m.hstack(m, m2))
    # 断言堆叠空矩阵的结果为一个空矩阵
    assert Matrix.hstack() == Matrix()

    # 测试回归 #12938
    M1 = Matrix.zeros(0, 0)
    M2 = Matrix.zeros(0, 1)
    M3 = Matrix.zeros(0, 2)
    M4 = Matrix.zeros(0, 3)
    # 断言将多个空矩阵水平堆叠的结果的行数为 0，列数为 6
    m = ShapingOnlyMatrix.hstack(M1, M2, M3, M4)
    assert m.rows == 0 and m.cols == 6

# 定义测试函数 `test_vstack`
def test_vstack():
    # 创建一个 4x3 的矩阵 m，其元素为按照 lambda 函数生成的值
    m = ShapingOnlyMatrix(4, 3, lambda i, j: i*3 + j)
    # 创建一个 3x4 的矩阵 m2，其元素为按照 lambda 函数生成的值
    m2 = ShapingOnlyMatrix(3, 4, lambda i, j: i*3 + j)
    # 断言矩阵 m 与其自身垂直堆叠后的结果等于自身
    assert m == m.vstack(m)
    # 断言多个矩阵垂直堆叠后的结果等于给定的预期矩阵
    assert m.vstack(m, m, m) == ShapingOnlyMatrix.vstack(m, m, m) == Matrix([
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
    # 断言尝试堆叠不兼容形状的矩阵将引发 ShapeError 异常
    raises(ShapeError, lambda: m.vstack(m, m2))
    # 断言堆叠空矩阵的结果为一个空矩阵
    assert Matrix.vstack() == Matrix()

# PropertiesOnlyMatrix 的测试
def test_atoms():
    # 创建一个 2x2 的矩阵 m，其元素包含有理数和符号 x
    m = PropertiesOnlyMatrix(2, 2, [1, 2, x, 1 - 1/x])
    # 断言矩阵 m 中的原子为 {1, 2, -1, x}
    assert m.atoms() == {S.One, S(2), S.NegativeOne, x}
    # 断言矩阵 m 中的符号为 {x}
    assert m.atoms(Symbol) == {x}

# 测试自由符号
def test_free_symbols():
    # 断言一个 2x1 的矩阵的自由符号为 {x}
    assert PropertiesOnlyMatrix([[x], [0]]).free_symbols == {x}

# 测试 has 方法
def test_has():
    # 创建一个 2x2 的矩阵 A
    A =
    # 断言 A 中是否包含符号 Symbol
    assert A.has(Symbol)
    
    # 创建一个仅包含属性的矩阵 A，其内容为 ((2, y), (2, 3))
    A = PropertiesOnlyMatrix(((2, y), (2, 3)))
    
    # 断言 A 中不包含变量 x
    assert not A.has(x)
# 测试是否反对称的函数
def test_is_anti_symmetric():
    # 定义符号变量 x
    x = symbols('x')
    # 检查给定矩阵是否不是反对称的
    assert PropertiesOnlyMatrix(2, 1, [1, 2]).is_anti_symmetric() is False
    # 创建一个包含符号变量的 3x3 矩阵 m
    m = PropertiesOnlyMatrix(3, 3, [0, x**2 + 2*x + 1, y, -(x + 1)**2, 0, x*y, -y, -x*y, 0])
    # 检查矩阵 m 是否是反对称的
    assert m.is_anti_symmetric() is True
    # 使用简化选项检查矩阵 m 是否不是反对称的
    assert m.is_anti_symmetric(simplify=False) is False
    # 使用自定义简化函数检查矩阵 m 是否不是反对称的
    assert m.is_anti_symmetric(simplify=lambda x: x) is False

    # 将矩阵 m 中的每个元素展开，并重新赋值给 m
    m = PropertiesOnlyMatrix(3, 3, [x.expand() for x in m])
    # 使用简化选项检查矩阵 m 是否是反对称的
    assert m.is_anti_symmetric(simplify=False) is True
    # 将矩阵 m 的第一个元素替换为 S.One，并检查其是否不是反对称的
    m = PropertiesOnlyMatrix(3, 3, [x.expand() for x in [S.One] + list(m)[1:]])
    assert m.is_anti_symmetric() is False


# 测试是否对角对称的函数
def test_diagonal_symmetrical():
    # 创建一个 2x2 的矩阵 m，判断其是否不是对角矩阵
    m = PropertiesOnlyMatrix(2, 2, [0, 1, 1, 0])
    assert not m.is_diagonal()
    # 检查矩阵 m 是否是对称的
    assert m.is_symmetric()
    # 使用不简化选项检查矩阵 m 是否是对称的
    assert m.is_symmetric(simplify=False)

    # 创建一个 2x2 的矩阵 m，判断其是否是对角矩阵
    m = PropertiesOnlyMatrix(2, 2, [1, 0, 0, 1])
    assert m.is_diagonal()

    # 创建一个对角元素为 [1, 2, 3] 的 3x3 矩阵 m，判断其是否是对角矩阵和对称的
    m = PropertiesOnlyMatrix(3, 3, diag(1, 2, 3))
    assert m.is_diagonal()
    assert m.is_symmetric()

    # 创建一个具有对角元素 [1, 2, 3] 的 3x3 矩阵 m，并将其与 diag(1, 2, 3) 进行比较
    m = PropertiesOnlyMatrix(3, 3, [1, 0, 0, 0, 2, 0, 0, 0, 3])
    assert m == diag(1, 2, 3)

    # 创建一个 2x3 的零矩阵 m，判断其是否不是对称的且是对角矩阵
    m = PropertiesOnlyMatrix(2, 3, zeros(2, 3))
    assert not m.is_symmetric()
    assert m.is_diagonal()

    # 创建一个对角元素为 [5, 6] 的 2x2 矩阵 m，判断其是否是对角矩阵
    m = PropertiesOnlyMatrix(((5, 0), (0, 6)))
    assert m.is_diagonal()

    # 创建一个对角元素为 [5, 6] 的 2x3 矩阵 m，判断其是否是对角矩阵
    m = PropertiesOnlyMatrix(((5, 0, 0), (0, 6, 0)))
    assert m.is_diagonal()

    # 创建一个 3x3 矩阵 m，其中元素包含符号变量 x 和 y
    m = Matrix(3, 3, [1, x**2 + 2*x + 1, y, (x + 1)**2, 2, 0, y, 0, 3])
    # 检查矩阵 m 是否是对称的
    assert m.is_symmetric()
    # 使用不简化选项检查矩阵 m 是否不是对称的
    assert not m.is_symmetric(simplify=False)
    # 对展开后的矩阵 m 使用不简化选项检查其是否不是对称的
    assert m.expand().is_symmetric(simplify=False)


# 测试是否厄米特的函数
def test_is_hermitian():
    # 创建一个复数矩阵 a，判断其是否是厄米特矩阵
    a = PropertiesOnlyMatrix([[1, I], [-I, 1]])
    assert a.is_hermitian
    # 创建一个复数矩阵 a，判断其是否不是厄米特矩阵
    a = PropertiesOnlyMatrix([[2*I, I], [-I, 1]])
    assert a.is_hermitian is False
    # 创建一个复数矩阵 a，判断其是否无法确定是否是厄米特矩阵
    a = PropertiesOnlyMatrix([[x, I], [-I, 1]])
    assert a.is_hermitian is None
    # 创建一个复数矩阵 a，判断其是否不是厄米特矩阵
    a = PropertiesOnlyMatrix([[x, 1], [-I, 1]])
    assert a.is_hermitian is False


# 测试是否单位矩阵的函数
def test_is_Identity():
    # 检查单位矩阵 eye_Properties(3) 是否是单位矩阵
    assert eye_Properties(3).is_Identity
    # 检查零矩阵是否不是单位矩阵
    assert not PropertiesOnlyMatrix(zeros(3)).is_Identity
    # 检查全1矩阵是否不是单位矩阵
    assert not PropertiesOnlyMatrix(ones(3)).is_Identity
    # issue 6242
    # 检查 [[1, 0, 0]] 是否不是单位矩阵
    assert not PropertiesOnlyMatrix([[1, 0, 0]]).is_Identity


# 测试矩阵是否包含符号变量的函数
def test_is_symbolic():
    # 创建一个矩阵 a，判断其是否包含符号变量
    a = PropertiesOnlyMatrix([[x, x], [x, x]])
    assert a.is_symbolic() is True
    # 创建一个矩阵 a，判断其是否不包含符号变量
    a = PropertiesOnlyMatrix([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert a.is_symbolic() is False
    # 创建一个矩阵 a，判断其是否包含符号变量
    a = PropertiesOnlyMatrix([[1, 2, 3, 4], [5, 6, x, 8]])
    assert a.is_symbolic() is True
    # 创建一个矩阵 a，判断其是否包含符号变量
    a = PropertiesOnlyMatrix([[1, x, 3]])
    assert a.is_symbolic() is True
    # 创建一个矩阵 a，判断其是否不包含符号变量
    a = PropertiesOnlyMatrix([[1, 2, 3]])
    assert a.is_symbolic() is False
    # 创建一个矩阵 a，判断其是否包含符号变量
    a = PropertiesOnlyMatrix([[1], [x], [3]])
    assert a.is_symbolic() is True
    # 创建一个矩阵 a，判断其是否不包含符号变量
    a = PropertiesOnlyMatrix([[1], [2], [3]])
    assert a.is_symbolic() is False


# 测试矩阵是否为上三角形式的函数
def test_is_upper():
    # 创建一个
    # 创建一个 PropertiesOnlyMatrix 对象，初始化为包含单个行向量 [[1, 2, 3]] 的矩阵
    a = PropertiesOnlyMatrix([[1, 2, 3]])
    # 断言该矩阵的 is_lower 属性为 False
    assert a.is_lower is False
    
    # 创建一个 PropertiesOnlyMatrix 对象，初始化为包含三个列向量 [[1], [2], [3]] 的矩阵
    a = PropertiesOnlyMatrix([[1], [2], [3]])
    # 断言该矩阵的 is_lower 属性为 True
    assert a.is_lower is True
# 测试是否为方阵的功能测试
def test_is_square():
    # 创建一个只具有属性的矩阵对象，并传入具有不同维度和元素的矩阵
    m = PropertiesOnlyMatrix([[1], [1]])
    m2 = PropertiesOnlyMatrix([[2, 2], [2, 2]])
    # 断言第一个矩阵不是方阵
    assert not m.is_square
    # 断言第二个矩阵是方阵
    assert m2.is_square


# 测试是否为对称矩阵的功能测试
def test_is_symmetric():
    # 创建一个具有特定元素的属性矩阵对象
    m = PropertiesOnlyMatrix(2, 2, [0, 1, 1, 0])
    # 断言矩阵是否对称
    assert m.is_symmetric()
    # 重新创建一个具有不同元素的属性矩阵对象
    m = PropertiesOnlyMatrix(2, 2, [0, 1, 0, 1])
    # 断言矩阵是否不对称
    assert not m.is_symmetric()


# 测试是否为 Hessenberg 矩阵的功能测试
def test_is_hessenberg():
    # 创建一个具有特定元素的属性矩阵对象 A
    A = PropertiesOnlyMatrix([[3, 4, 1], [2, 4, 5], [0, 1, 2]])
    # 断言矩阵 A 是否为上 Hessenberg 矩阵
    assert A.is_upper_hessenberg
    # 重新创建一个具有不同元素的属性矩阵对象 A
    A = PropertiesOnlyMatrix(3, 3, [3, 2, 0, 4, 4, 1, 1, 5, 2])
    # 断言矩阵 A 是否为下 Hessenberg 矩阵
    assert A.is_lower_hessenberg
    # 重新创建一个具有不同元素的属性矩阵对象 A
    A = PropertiesOnlyMatrix(3, 3, [3, 2, -1, 4, 4, 1, 1, 5, 2])
    # 断言矩阵 A 是否不是下 Hessenberg 矩阵
    assert A.is_lower_hessenberg is False
    # 断言矩阵 A 是否不是上 Hessenberg 矩阵
    assert A.is_upper_hessenberg is False

    # 重新创建一个具有不同元素的属性矩阵对象 A
    A = PropertiesOnlyMatrix([[3, 4, 1], [2, 4, 5], [3, 1, 2]])
    # 断言矩阵 A 是否不是上 Hessenberg 矩阵
    assert not A.is_upper_hessenberg


# 测试是否为零矩阵的功能测试
def test_is_zero():
    # 断言空矩阵是否为零矩阵
    assert PropertiesOnlyMatrix(0, 0, []).is_zero_matrix
    # 断言特定元素的属性矩阵是否为零矩阵
    assert PropertiesOnlyMatrix([[0, 0], [0, 0]]).is_zero_matrix
    # 断言特定维度的零矩阵是否为零矩阵
    assert PropertiesOnlyMatrix(zeros(3, 4)).is_zero_matrix
    # 断言特定单位矩阵是否不是零矩阵
    assert not PropertiesOnlyMatrix(eye(3)).is_zero_matrix
    # 断言包含符号的属性矩阵是否为未定义
    assert PropertiesOnlyMatrix([[x, 0], [0, 0]]).is_zero_matrix == None
    # 断言包含符号的属性矩阵是否为非零矩阵
    assert PropertiesOnlyMatrix([[x, 1], [0, 0]]).is_zero_matrix == False
    # 创建一个具有非零属性的符号
    a = Symbol('a', nonzero=True)
    # 断言包含非零符号的属性矩阵是否为非零矩阵
    assert PropertiesOnlyMatrix([[a, 0], [0, 0]]).is_zero_matrix == False


# 测试获取矩阵元素值的功能测试
def test_values():
    # 断言特定元素的属性矩阵对象的值集合是否为预期值
    assert set(PropertiesOnlyMatrix(2, 2, [0, 1, 2, 3]).values()) == {1, 2, 3}
    # 创建一个包含实数符号的属性矩阵对象
    x = Symbol('x', real=True)
    # 断言包含实数符号的属性矩阵对象的值集合是否为预期值
    assert set(PropertiesOnlyMatrix(2, 2, [x, 0, 0, 1]).values()) == {x, 1}


# OperationsOnlyMatrix 的 applyfunc 方法测试
def test_applyfunc():
    # 创建一个单位矩阵并应用指定函数
    m0 = OperationsOnlyMatrix(eye(3))
    assert m0.applyfunc(lambda x: 2*x) == eye(3)*2
    assert m0.applyfunc(lambda x: 0) == zeros(3)
    assert m0.applyfunc(lambda x: 1) == ones(3)


# OperationsOnlyMatrix 的 adjoint 方法测试
def test_adjoint():
    # 创建一个特定复数元素的矩阵和其伴随矩阵
    dat = [[0, I], [1, 0]]
    ans = OperationsOnlyMatrix([[0, 1], [-I, 0]])
    assert ans.adjoint() == Matrix(dat)


# OperationsOnlyMatrix 的 as_real_imag 方法测试
def test_as_real_imag():
    # 创建一个实数和复数元素混合的矩阵对象
    m1 = OperationsOnlyMatrix(2, 2, [1, 2, 3, 4])
    m3 = OperationsOnlyMatrix(2, 2, [1 + S.ImaginaryUnit, 2 + 2*S.ImaginaryUnit,
                                     3 + 3*S.ImaginaryUnit, 4 + 4*S.ImaginaryUnit])

    # 获取实部和虚部
    a, b = m3.as_real_imag()
    assert a == m1
    assert b == m1


# OperationsOnlyMatrix 的 conjugate 方法测试
def test_conjugate():
    # 创建一个具有复数元素的矩阵 M
    M = OperationsOnlyMatrix([[0, I, 5],
                              [1, 2, 0]])

    # 断言 M 的转置是否为预期结果
    assert M.T == Matrix([[0, 1],
                          [I, 2],
                          [5, 0]])

    # 断言 M 的共轭是否为预期结果
    assert M.C == Matrix([[0, -I, 5],
                          [1,  2, 0]])
    # 断言 M 的共轭是否等同于 M 的 conjugate 方法
    assert M.C == M.conjugate()

    # 断言 M 的共轭转置是否等同于 M 的 Hermitian 转置
    assert M.H == M.T.C
    assert M.H == Matrix([[ 0, 1],
                          [-I, 2],
                          [ 5, 0]])


# OperationsOnlyMatrix 的 doit 方法测试
def test_doit():
    # 创建一个包含未求值表达式的矩阵对象
    a = OperationsOnlyMatrix([[Add(x, x, evaluate=False)]])
    # 断言该表达式是否与预期结果不相等
    assert a[0] != 2*x
    # 断言应用 doit 方法后的矩阵是否与预期结果相等
    assert a.doit() == Matrix([[2*x]])


# OperationsOnlyMatrix 的 evalf 方法测试
def test_evalf():
    # 创建一个包含数学函数的矩阵对象
    a = OperationsOnlyMatrix(2, 1, [sqrt(5), 6])
    # 断言应用 evalf 方法后矩阵中每个元素的数值是否与预期相等
    assert all(a.evalf()[i] == a[i].evalf() for i in range(2))
    # 断言：验证所有元素的两位小数精度的数值等于元素自身的两位小数精度的数值
    assert all(a.evalf(2)[i] == a[i].evalf(2) for i in range(2))
    
    # 断言：验证所有元素的两位小数精度的数值等于元素自身调用两位小数精度方法后的数值
    assert all(a.n(2)[i] == a[i].n(2) for i in range(2))
def test_expand():
    # 创建一个 OperationsOnlyMatrix 对象，传入一个二维列表作为初始矩阵
    m0 = OperationsOnlyMatrix([[x*(x + y), 2], [((x + y)*y)*x, x*(y + x*(x + y))]])
    # 调用 expand() 方法，期望返回一个新的矩阵对象 m1
    m1 = m0.expand()
    # 使用 assert 断言 m1 应该等于给定的 Matrix 对象
    assert m1 == Matrix(
        [[x*y + x**2, 2], [x*y**2 + y*x**2, x*y + y*x**2 + x**3]])

    # 创建一个实数符号对象 a
    a = Symbol('a', real=True)

    # 使用 assert 断言对于一个包含 exp(I*a) 的 OperationsOnlyMatrix，调用 expand(complex=True) 后应该等于对应的 Matrix
    assert OperationsOnlyMatrix(1, 1, [exp(I*a)]).expand(complex=True) == \
           Matrix([cos(a) + I*sin(a)])


def test_refine():
    # 创建一个 OperationsOnlyMatrix 对象，传入一个二维列表作为初始矩阵
    m0 = OperationsOnlyMatrix([[Abs(x)**2, sqrt(x**2)],
                 [sqrt(x**2)*Abs(y)**2, sqrt(y**2)*Abs(x)**2]])
    # 调用 refine() 方法，对矩阵进行精化，期望返回一个新的矩阵对象 m1
    m1 = m0.refine(Q.real(x) & Q.real(y))
    # 使用 assert 断言 m1 应该等于给定的 Matrix 对象
    assert m1 == Matrix([[x**2, Abs(x)], [y**2*Abs(x), x**2*Abs(y)]])

    # 再次调用 refine() 方法，这次对不同的条件进行精化
    m1 = m0.refine(Q.positive(x) & Q.positive(y))
    # 使用 assert 断言 m1 应该等于给定的 Matrix 对象
    assert m1 == Matrix([[x**2, x], [x*y**2, x**2*y]])

    # 再次调用 refine() 方法，这次对不同的条件进行精化
    m1 = m0.refine(Q.negative(x) & Q.negative(y))
    # 使用 assert 断言 m1 应该等于给定的 Matrix 对象
    assert m1 == Matrix([[x**2, -x], [-x*y**2, -x**2*y]])


def test_replace():
    # 创建两个符号函数对象 F 和 G
    F, G = symbols('F, G', cls=Function)
    # 创建一个 OperationsOnlyMatrix 对象，传入一个 lambda 函数用于填充矩阵内容
    K = OperationsOnlyMatrix(2, 2, lambda i, j: G(i+j))
    # 创建一个 OperationsOnlyMatrix 对象，传入一个 lambda 函数用于填充矩阵内容
    M = OperationsOnlyMatrix(2, 2, lambda i, j: F(i+j))
    # 调用 replace() 方法，将矩阵 M 中的所有 F 替换为 G，期望返回一个新的矩阵对象 N
    N = M.replace(F, G)
    # 使用 assert 断言 N 应该等于给定的矩阵对象 K
    assert N == K


def test_replace_map():
    # 创建两个符号函数对象 F 和 G
    F, G = symbols('F, G', cls=Function)
    # 创建一个 OperationsOnlyMatrix 对象，传入一个 lambda 函数用于填充矩阵内容
    K = OperationsOnlyMatrix(2, 2, [(G(0), {F(0): G(0)}), (G(1), {F(1): G(1)}), (G(1), {F(1): G(1)}), (G(2), {F(2): G(2)})])
    # 创建一个 OperationsOnlyMatrix 对象，传入一个 lambda 函数用于填充矩阵内容
    M = OperationsOnlyMatrix(2, 2, lambda i, j: F(i+j))
    # 调用 replace() 方法，将矩阵 M 中的所有 F 替换为 G，同时指定 map=True，期望返回一个新的矩阵对象 N
    N = M.replace(F, G, True)
    # 使用 assert 断言 N 应该等于给定的矩阵对象 K
    assert N == K


def test_rot90():
    # 创建一个 2x2 的 Matrix 对象 A
    A = Matrix([[1, 2], [3, 4]])
    # 使用 assert 断言 A.rot90(0)、A.rot90(4) 等均应等于 A 本身
    assert A == A.rot90(0) == A.rot90(4)
    # 使用 assert 断言 A.rot90(2)、A.rot90(-2)、A.rot90(6) 等应该等于特定的 Matrix 对象
    assert A.rot90(2) == A.rot90(-2) == A.rot90(6) == Matrix(((4, 3), (2, 1)))
    # 使用 assert 断言 A.rot90(3)、A.rot90(-1)、A.rot90(7) 等应该等于特定的 Matrix 对象
    assert A.rot90(3) == A.rot90(-1) == A.rot90(7) == Matrix(((2, 4), (1, 3)))
    # 使用 assert 断言 A.rot90()、A.rot90(-7)、A.rot90(-3) 等应该等于特定的 Matrix 对象
    assert A.rot90() == A.rot90(-7) == A.rot90(-3) == Matrix(((3, 1), (4, 2)))


def test_simplify():
    # 创建一个符号变量 n 和函数 f
    n = Symbol('n')
    f = Function('f')

    # 创建一个 OperationsOnlyMatrix 对象，传入一个二维列表作为初始矩阵
    M = OperationsOnlyMatrix([[1/x + 1/y, (x + x*y) / x],
                [(f(x) + y*f(x))/f(x), 2 * (1/n - cos(n * pi)/n) / pi ]])
    # 使用 assert 断言 M.simplify() 应该等于给定的 Matrix 对象
    assert M.simplify() == Matrix([[ (x + y)/(x * y), 1 + y ],
                        [ 1 + y, 2*((1 - cos(pi*n))/(pi*n)) ]])
    # 创建一个等式对象 eq
    eq = (1 + x)**2
    # 创建一个 OperationsOnlyMatrix 对象，传入一个二维列表作为初始矩阵
    M = OperationsOnlyMatrix([[eq]])
    # 使用 assert 断言 M.simplify() 应该等于给定的 Matrix 对象
    assert M.simplify() == Matrix([[eq]])
    # 使用 assert 断言 M.simplify(ratio=oo) 应该等于给定的 Matrix 对象
    assert M.simplify(ratio=oo) == Matrix([[eq.simplify(ratio=oo)]])

    # 创建一个 2x2 的 Matrix 对象 m
    m = Matrix([[30, 2], [3, 4]])
    # 使用 assert 断言 (1/(m.trace())).simplify() 应该等于给定的 Rational 对象
    assert (1/(m.trace())).simplify() == Rational(1, 34)


def test_subs():
    # 使用 assert 断言 OperationsOnlyMatrix([[1, x], [x, 4]]).subs(x, 5) 应该等于给定的 Matrix 对象
    assert OperationsOnlyMatrix([[1, x], [x, 4]]).subs(x, 5) == Matrix([[1, 5], [5, 4]])
    # 使用 assert 断言 OperationsOnlyMatrix([[x, 2], [x + y, 4]]).subs([[x, -1], [y, -2]]) 应该等于给定的 Matrix 对象
    assert OperationsOnlyMatrix([[x, 2], [x + y, 4]]).subs([[x, -1], [y, -2]]) == \
           Matrix([[-1, 2], [-3, 4]])
    # 使用 assert 断言 OperationsOnlyMatrix([[x, 2], [x + y, 4]]).subs([(x, -1), (y, -2)]) 应该等于给定的 Matrix 对象
    assert OperationsOnlyMatrix([[x,
    # 断言语句：验证 OperationsOnlyMatrix([[x*y]]) 在进行变量替换后是否等于特定的 Matrix([[(x - 1)*(y - 1)]]).
    assert OperationsOnlyMatrix([[x*y]]).subs({x: y - 1, y: x - 1}, simultaneous=True) == \
           Matrix([[(x - 1)*(y - 1)]])
# 定义测试函数 `test_trace`，用于测试矩阵迹计算方法
def test_trace():
    # 创建一个 OperationsOnlyMatrix 实例，表示只支持特定操作的矩阵，包含指定的元素
    M = OperationsOnlyMatrix([[1, 0, 0],
                [0, 5, 0],
                [0, 0, 8]])
    # 断言矩阵 M 的迹（主对角线上元素之和）等于 14
    assert M.trace() == 14


# 定义测试函数 `test_xreplace`，测试矩阵元素替换方法
def test_xreplace():
    # 断言替换矩阵中的符号变量 x 为具体值 5 后，得到的矩阵与预期结果相等
    assert OperationsOnlyMatrix([[1, x], [x, 4]]).xreplace({x: 5}) == \
           Matrix([[1, 5], [5, 4]])
    # 断言替换矩阵中的符号变量 x 和 y 为具体值 -1 和 -2 后，得到的矩阵与预期结果相等
    assert OperationsOnlyMatrix([[x, 2], [x + y, 4]]).xreplace({x: -1, y: -2}) == \
           Matrix([[-1, 2], [-3, 4]])


# 定义测试函数 `test_permute`，测试矩阵排列方法
def test_permute():
    # 创建一个 OperationsOnlyMatrix 实例，包含特定维度和元素
    a = OperationsOnlyMatrix(3, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # 测试当指定的行置换包含无效索引时，是否引发 IndexError 异常
    raises(IndexError, lambda: a.permute([[0, 5]]))
    # 测试当指定的行置换参数不是列表时，是否引发 ValueError 异常
    raises(ValueError, lambda: a.permute(Symbol('x')))
    
    # 对矩阵进行行置换，然后断言结果与预期的矩阵 b 相等
    b = a.permute_rows([[0, 2], [0, 1]])
    assert a.permute([[0, 2], [0, 1]]) == b == Matrix([
                                            [5,  6,  7,  8],
                                            [9, 10, 11, 12],
                                            [1,  2,  3,  4]])

    # 对矩阵进行列置换，然后断言结果与预期的矩阵 b 相等
    b = a.permute_cols([[0, 2], [0, 1]])
    assert a.permute([[0, 2], [0, 1]], orientation='cols') == b ==\
                            Matrix([
                            [ 2,  3, 1,  4],
                            [ 6,  7, 5,  8],
                            [10, 11, 9, 12]])

    # 对矩阵进行列置换（反向），然后断言结果与预期的矩阵 b 相等
    b = a.permute_cols([[0, 2], [0, 1]], direction='backward')
    assert a.permute([[0, 2], [0, 1]], orientation='cols', direction='backward') == b ==\
                            Matrix([
                            [ 3, 1,  2,  4],
                            [ 7, 5,  6,  8],
                            [11, 9, 10, 12]])

    # 测试矩阵按给定顺序排列，然后断言结果与预期的矩阵相等
    assert a.permute([1, 2, 0, 3]) == Matrix([
                                            [5,  6,  7,  8],
                                            [9, 10, 11, 12],
                                            [1,  2,  3,  4]])

    # 使用 Permutation 对象进行矩阵排列，然后断言结果与预期的矩阵相等
    from sympy.combinatorics import Permutation
    assert a.permute(Permutation([1, 2, 0, 3])) == Matrix([
                                            [5,  6,  7,  8],
                                            [9, 10, 11, 12],
                                            [1,  2,  3,  4]])


# 定义测试函数 `test_upper_triangular`，测试上三角矩阵方法
def test_upper_triangular():
    # 创建一个 OperationsOnlyMatrix 实例，包含特定维度和元素
    A = OperationsOnlyMatrix([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ])

    # 测试获取 A 的上三角矩阵，k=2，然后断言结果与预期的矩阵 R 相等
    R = A.upper_triangular(2)
    assert R == OperationsOnlyMatrix([
                        [0, 0, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]
                    ])

    # 测试获取 A 的上三角矩阵，k=-2，然后断言结果与预期的矩阵 R 相等
    R = A.upper_triangular(-2)
    assert R == OperationsOnlyMatrix([
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 1, 1, 1]
                    ])

    # 测试获取 A 的上三角矩阵（默认 k=0），然后断言结果与预期的矩阵 R 相等
    R = A.upper_triangular()
    assert R == OperationsOnlyMatrix([
                        [1, 1, 1, 1],
                        [0, 1, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]
                    ])

# 定义测试函数 `test_lower_triangular`，测试下三角矩阵方法
    # 创建一个 OperationsOnlyMatrix 对象 A，该对象初始化为一个4x4的全1矩阵
    A = OperationsOnlyMatrix([
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]
                    ])

    # 调用 A 对象的 lower_triangular 方法，生成其下三角矩阵 L
    L = A.lower_triangular()
    # 使用断言检查生成的下三角矩阵 L 是否与期望的 ArithmeticOnlyMatrix 相等
    assert L == ArithmeticOnlyMatrix([
                        [1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1]])

    # 再次调用 A 对象的 lower_triangular 方法，传入参数 2，生成偏移为2的下三角矩阵 L
    L = A.lower_triangular(2)
    # 使用断言检查生成的下三角矩阵 L 是否与期望的 ArithmeticOnlyMatrix 相等
    assert L == ArithmeticOnlyMatrix([
                        [1, 1, 1, 0],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]
                    ])

    # 再次调用 A 对象的 lower_triangular 方法，传入参数 -2，生成偏移为-2的下三角矩阵 L
    L = A.lower_triangular(-2)
    # 使用断言检查生成的下三角矩阵 L 是否与期望的 ArithmeticOnlyMatrix 相等
    assert L == ArithmeticOnlyMatrix([
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 1, 0, 0]
                    ])
# ArithmeticOnlyMatrix tests

# 测试绝对值函数
def test_abs():
    # 创建一个包含整数和符号的算术矩阵对象
    m = ArithmeticOnlyMatrix([[1, -2], [x, y]])
    # 断言算术矩阵对象的绝对值等于给定的算术矩阵对象
    assert abs(m) == ArithmeticOnlyMatrix([[1, 2], [Abs(x), Abs(y)]])


# 测试矩阵加法
def test_add():
    # 创建两个不同的算术矩阵对象
    m = ArithmeticOnlyMatrix([[1, 2, 3], [x, y, x], [2*y, -50, z*x]])
    # 断言两个相同算术矩阵对象的加法结果与期望结果相等
    assert m + m == ArithmeticOnlyMatrix([[2, 4, 6], [2*x, 2*y, 2*x], [4*y, -100, 2*z*x]])
    # 创建一个与之前矩阵不匹配的算术矩阵对象，断言加法引发 ShapeError 错误
    n = ArithmeticOnlyMatrix(1, 2, [1, 2])
    raises(ShapeError, lambda: m + n)


# 测试矩阵乘法
def test_multiplication():
    # 创建两个不同的算术矩阵对象
    a = ArithmeticOnlyMatrix((
        (1, 2),
        (3, 1),
        (0, 6),
    ))

    b = ArithmeticOnlyMatrix((
        (1, 2),
        (3, 0),
    ))

    # 断言乘法引发 ShapeError 错误
    raises(ShapeError, lambda: b*a)
    # 断言乘法引发 TypeError 错误
    raises(TypeError, lambda: a*{})

    # 执行矩阵乘法，并检查结果
    c = a*b
    assert c[0, 0] == 7
    assert c[0, 1] == 2
    assert c[1, 0] == 6
    assert c[1, 1] == 6
    assert c[2, 0] == 18
    assert c[2, 1] == 0

    # 尝试使用 @ 运算符进行矩阵乘法（仅适用于 Python 3.5+）
    try:
        eval('c = a @ b')
    except SyntaxError:
        pass
    else:
        assert c[0, 0] == 7
        assert c[0, 1] == 2
        assert c[1, 0] == 6
        assert c[1, 1] == 6
        assert c[2, 0] == 18
        assert c[2, 1] == 0

    # 执行矩阵元素级乘法，并检查结果
    h = a.multiply_elementwise(c)
    assert h == matrix_multiply_elementwise(a, c)
    assert h[0, 0] == 7
    assert h[0, 1] == 4
    assert h[1, 0] == 18
    assert h[1, 1] == 6
    assert h[2, 0] == 0
    assert h[2, 1] == 0
    # 断言元素级乘法引发 ShapeError 错误
    raises(ShapeError, lambda: a.multiply_elementwise(b))

    # 测试矩阵与符号的乘法
    c = b * Symbol("x")
    assert isinstance(c, ArithmeticOnlyMatrix)
    assert c[0, 0] == x
    assert c[0, 1] == 2*x
    assert c[1, 0] == 3*x
    assert c[1, 1] == 0

    # 测试符号与矩阵的乘法，确保与之前结果相同
    c2 = x * b
    assert c == c2

    # 测试常数与矩阵的乘法
    c = 5 * b
    assert isinstance(c, ArithmeticOnlyMatrix)
    assert c[0, 0] == 5
    assert c[0, 1] == 2*5
    assert c[1, 0] == 3*5
    assert c[1, 1] == 0

    # 尝试使用 @ 运算符进行常数与矩阵的乘法（仅适用于 Python 3.5+）
    try:
        eval('c = 5 @ b')
    except SyntaxError:
        pass
    else:
        assert isinstance(c, ArithmeticOnlyMatrix)
        assert c[0, 0] == 5
        assert c[0, 1] == 2*5
        assert c[1, 0] == 3*5
        assert c[1, 1] == 0

    # 测试元素级乘法的特定案例
    # https://github.com/sympy/sympy/issues/22353
    A = Matrix(ones(3, 1))
    _h = -Rational(1, 2)
    B = Matrix([_h, _h, _h])
    assert A.multiply_elementwise(B) == Matrix([
        [_h],
        [_h],
        [_h]])


# 测试矩阵乘法运算符
def test_matmul():
    # 创建一个简单的 2x2 矩阵对象
    a = Matrix([[1, 2], [3, 4]])

    # 断言与整数的 @ 运算返回 NotImplemented
    assert a.__matmul__(2) == NotImplemented

    # 断言整数与矩阵的 @ 运算返回 NotImplemented
    assert a.__rmatmul__(2) == NotImplemented

    # 使用 eval 尝试执行 2 @ a 的运算，捕获 SyntaxError 和 TypeError 异常
    try:
        eval('2 @ a')
    except SyntaxError:
        pass
    except TypeError:  # NotImplemented 返回时引发 TypeError
        pass

    # 使用 eval 尝试执行 a @ 2 的运算，捕获 SyntaxError 和 TypeError 异常
    try:
        eval('a @ 2')
    except SyntaxError:
        pass
    except TypeError:  # NotImplemented 返回时引发 TypeError
        pass


# 测试非矩阵乘法情况
def test_non_matmul():
    """
    Test that if explicitly specified as non-matrix, mul reverts
    to scalar multiplication.
    """
    # 定义一个名为foo的类，它继承自Expr类（假设这是一个已经定义的类）
    class foo(Expr):
        # 类属性，表示该类实例不是矩阵
        is_Matrix = False
        # 类属性，表示该类实例不像矩阵
        is_MatrixLike = False
        # 类属性，表示该类实例的形状为(1, 1)
        shape = (1, 1)
    
    # 创建一个2x2的矩阵A，内容为[[1, 2], [3, 4]]
    A = Matrix([[1, 2], [3, 4]])
    
    # 创建一个foo类的实例b
    b = foo()
    
    # 断言，验证表达式b*A是否等于矩阵[[b, 2*b], [3*b, 4*b]]
    assert b * A == Matrix([[b, 2*b], [3*b, 4*b]])
    
    # 断言，验证表达式A*b是否等于矩阵[[b, 2*b], [3*b, 4*b]]
    assert A * b == Matrix([[b, 2*b], [3*b, 4*b]])
# 定义一个测试函数，用于测试矩阵乘方运算
def test_power():
    # 测试在非方阵上进行乘方操作是否引发 NonSquareMatrixError 异常
    raises(NonSquareMatrixError, lambda: Matrix((1, 2))**2)

    # 创建一个仅支持算术运算的矩阵 A
    A = ArithmeticOnlyMatrix([[2, 3], [4, 5]])
    # 断言 A 的 5 次方的所有元素是否等于给定的值
    assert (A**5)[:] == (6140, 8097, 10796, 14237)

    # 重新赋值 A 为另一个仅支持算术运算的矩阵
    A = ArithmeticOnlyMatrix([[2, 1, 3], [4, 2, 4], [6, 12, 1]])
    # 断言 A 的 3 次方的所有元素是否等于给定的值
    assert (A**3)[:] == (290, 262, 251, 448, 440, 368, 702, 954, 433)

    # 断言 A 的 0 次方是否等于单位矩阵
    assert A**0 == eye(3)

    # 断言 A 的 1 次方是否等于 A 本身
    assert A**1 == A

    # 断言一个单元素的矩阵 A 的 100 次方的第一个元素是否等于 2 的 100 次方
    assert (ArithmeticOnlyMatrix([[2]]) ** 100)[0, 0] == 2**100

    # 断言一个仅支持算术运算的矩阵的整数次方操作
    assert ArithmeticOnlyMatrix([[1, 2], [3, 4]])**Integer(2) == ArithmeticOnlyMatrix([[7, 10], [15, 22]])

    # 重新赋值 A 为一般矩阵
    A = Matrix([[1,2],[4,5]])
    # 断言 A 的 20 次方的计算方法为 Cayley-Hamilton 方法和普通乘法方法结果的一致性
    assert A.pow(20, method='cayley') == A.pow(20, method='multiply')

# 定义一个测试函数，用于测试矩阵取负操作
def test_neg():
    # 创建一个仅支持算术运算的矩阵 n
    n = ArithmeticOnlyMatrix(1, 2, [1, 2])
    # 断言矩阵取负后是否得到预期的结果
    assert -n == ArithmeticOnlyMatrix(1, 2, [-1, -2])

# 定义一个测试函数，用于测试矩阵减法操作
def test_sub():
    # 创建一个仅支持算术运算的矩阵 n
    n = ArithmeticOnlyMatrix(1, 2, [1, 2])
    # 断言两个相同矩阵相减是否得到预期的结果
    assert n - n == ArithmeticOnlyMatrix(1, 2, [0, 0])

# 定义一个测试函数，用于测试矩阵除法操作
def test_div():
    # 创建一个仅支持算术运算的矩阵 n
    n = ArithmeticOnlyMatrix(1, 2, [1, 2])
    # 断言矩阵除以数值后是否得到预期的结果
    assert n/2 == ArithmeticOnlyMatrix(1, 2, [S.Half, S(2)/2])

# SpecialOnlyMatrix 的测试
# 定义一个测试函数，用于测试单位矩阵的生成
def test_eye():
    # 断言生成单位矩阵的方法是否得到预期的结果
    assert list(SpecialOnlyMatrix.eye(2, 2)) == [1, 0, 0, 1]
    assert list(SpecialOnlyMatrix.eye(2)) == [1, 0, 0, 1]
    assert type(SpecialOnlyMatrix.eye(2)) == SpecialOnlyMatrix
    assert type(SpecialOnlyMatrix.eye(2, cls=Matrix)) == Matrix

# 定义一个测试函数，用于测试全一矩阵的生成
def test_ones():
    # 断言生成全一矩阵的方法是否得到预期的结果
    assert list(SpecialOnlyMatrix.ones(2, 2)) == [1, 1, 1, 1]
    assert list(SpecialOnlyMatrix.ones(2)) == [1, 1, 1, 1]
    assert SpecialOnlyMatrix.ones(2, 3) == Matrix([[1, 1, 1], [1, 1, 1]])
    assert type(SpecialOnlyMatrix.ones(2)) == SpecialOnlyMatrix
    assert type(SpecialOnlyMatrix.ones(2, cls=Matrix)) == Matrix

# 定义一个测试函数，用于测试全零矩阵的生成
def test_zeros():
    # 断言生成全零矩阵的方法是否得到预期的结果
    assert list(SpecialOnlyMatrix.zeros(2, 2)) == [0, 0, 0, 0]
    assert list(SpecialOnlyMatrix.zeros(2)) == [0, 0, 0, 0]
    assert SpecialOnlyMatrix.zeros(2, 3) == Matrix([[0, 0, 0], [0, 0, 0]])
    assert type(SpecialOnlyMatrix.zeros(2)) == SpecialOnlyMatrix
    assert type(SpecialOnlyMatrix.zeros(2, cls=Matrix)) == Matrix

# 定义一个测试函数，用于测试对角矩阵的生成
def test_diag_make():
    diag = SpecialOnlyMatrix.diag
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    # 断言对角矩阵的构造方法是否得到预期的结果
    assert diag(a, b, b) == Matrix([
        [1, 2, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0],
        [0, 0, 3, x, 0, 0],
        [0, 0, y, 3, 0, 0],
        [0, 0, 0, 0, 3, x],
        [0, 0, 0, 0, y, 3],
    ])
    assert diag(a, b, c) == Matrix([
        [1, 2, 0, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0, 0],
        [0, 0, 3, x, 0, 0, 0],
        [0, 0, y, 3, 0, 0, 0],
        [0, 0, 0, 0, 3, x, 3],
        [0, 0, 0, 0, y, 3, z],
        [0, 0, 0, 0, x, y, z],
    ])
    assert diag(a, c, b) == Matrix([
        [1, 2, 0, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0, 0],
        [0, 0, 3, x, 3, 0, 0],
        [0, 0, y, 3, z, 0, 0],
        [0, 0, x, y, z, 0, 0],
        [0, 0, 0, 0, 0, 3, x],
        [0, 0, 0, 0, 0, y, 3],
    ])
    a = Matrix([x, y, z])
    b = Matrix([[1, 2], [3, 4]])
    c = Matrix([[5, 6]])
    # 验证函数 diag 的基本功能，生成对角矩阵，每个块独立
    assert diag(a, 7, b, c) == Matrix([
        [x, 0, 0, 0, 0, 0],
        [y, 0, 0, 0, 0, 0],
        [z, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 3, 4, 0, 0],
        [0, 0, 0, 0, 5, 6]])
    
    # 验证在给定 rows=5 时函数 diag 是否会抛出 ValueError 异常
    raises(ValueError, lambda: diag(a, 7, b, c, rows=5))
    
    # 验证 diag(1) 生成的对角矩阵
    assert diag(1) == Matrix([[1]])
    
    # 验证 diag(1, rows=2) 生成的对角矩阵
    assert diag(1, rows=2) == Matrix([[1, 0], [0, 0]])
    
    # 验证 diag(1, cols=2) 生成的对角矩阵
    assert diag(1, cols=2) == Matrix([[1, 0], [0, 0]])
    
    # 验证 diag(1, rows=3, cols=2) 生成的对角矩阵
    assert diag(1, rows=3, cols=2) == Matrix([[1, 0], [0, 0], [0, 0]])
    
    # 验证 diag(*[2, 3]) 生成的对角矩阵
    assert diag(*[2, 3]) == Matrix([
        [2, 0],
        [0, 3]])
    
    # 验证 diag(Matrix([2, 3])) 生成的对角矩阵
    assert diag(Matrix([2, 3])) == Matrix([
        [2],
        [3]])
    
    # 验证 diag([1, [2, 3], 4], unpack=False) 生成的对角矩阵
    assert diag([1, [2, 3], 4], unpack=False) == \
            diag([[1], [2, 3], [4]], unpack=False) == Matrix([
        [1, 0],
        [2, 3],
        [4, 0]])
    
    # 验证 diag(1) 返回的对象类型是否为 SpecialOnlyMatrix
    assert type(diag(1)) == SpecialOnlyMatrix
    
    # 验证 diag(1, cls=Matrix) 返回的对象类型是否为 Matrix
    assert type(diag(1, cls=Matrix)) == Matrix
    
    # 验证 Matrix.diag([1, 2, 3]) 与 Matrix.diag(1, 2, 3) 返回结果是否相等
    assert Matrix.diag([1, 2, 3]) == Matrix.diag(1, 2, 3)
    
    # 验证 Matrix.diag([1, 2, 3], unpack=False) 返回的矩阵形状是否为 (3, 1)
    assert Matrix.diag([1, 2, 3], unpack=False).shape == (3, 1)
    
    # 验证 Matrix.diag([[1, 2, 3]]) 返回的矩阵形状是否为 (3, 1)
    assert Matrix.diag([[1, 2, 3]]).shape == (3, 1)
    
    # 验证 Matrix.diag([[1, 2, 3]], unpack=False) 返回的矩阵形状是否为 (1, 3)
    assert Matrix.diag([[1, 2, 3]], unpack=False).shape == (1, 3)
    
    # 验证 Matrix.diag([[[1, 2, 3]]]) 返回的矩阵形状是否为 (1, 3)
    assert Matrix.diag([[[1, 2, 3]]]).shape == (1, 3)
    
    # 验证 kerning 可以用于移动起始点
    assert Matrix.diag(ones(0, 2), 1, 2) == Matrix([
        [0, 0, 1, 0],
        [0, 0, 0, 2]])
    
    # 验证 kerning 可以用于移动起始点
    assert Matrix.diag(ones(2, 0), 1, 2) == Matrix([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 2]])
# 定义一个测试函数，用于测试 Matrix 类的 diagonal 方法
def test_diagonal():
    # 创建一个 3x3 的 Matrix 对象 m，其元素为 0 到 8
    m = Matrix(3, 3, range(9))
    # 获取 m 的主对角线元素，存入变量 d
    d = m.diagonal()
    # 断言 d 应与 m 的主对角线元素相等
    assert d == m.diagonal(0)
    # 断言将 d 转换为元组后应与 (0, 4, 8) 相等
    assert tuple(d) == (0, 4, 8)
    # 断言获取 m 的次对角线元素，转换为元组后应为 (1, 5)
    assert tuple(m.diagonal(1)) == (1, 5)
    # 断言获取 m 的次对角线元素，转换为元组后应为 (3, 7)
    assert tuple(m.diagonal(-1)) == (3, 7)
    # 断言获取 m 的第二条对角线元素，转换为元组后应为 (2,)
    assert tuple(m.diagonal(2)) == (2,)
    # 断言 d 的类型应与 m 相同
    assert type(m.diagonal()) == type(m)
    # 创建一个稀疏矩阵 SparseMatrix 对象 s，其中只有一个元素 (1, 1) 的值为 1
    s = SparseMatrix(3, 3, {(1, 1): 1})
    # 断言获取 s 的对角线元素后，其类型应与 s 相同
    assert type(s.diagonal()) == type(s)
    # 断言 m 和 s 的类型不相同
    assert type(m) != type(s)
    # 断言调用 diagonal 方法时传入超出范围的参数会引发 ValueError 异常
    raises(ValueError, lambda: m.diagonal(3))
    raises(ValueError, lambda: m.diagonal(-3))
    # 断言调用 diagonal 方法时传入非整数参数会引发 ValueError 异常
    raises(ValueError, lambda: m.diagonal(pi))
    # 创建一个 2x3 的全 1 矩阵 M
    M = ones(2, 3)
    # 断言将 M 的带状矩阵（从 M 的每个对角线取出元素组成的字典）与 M 相等
    assert banded({i: list(M.diagonal(i)) for i in range(1 - M.rows, M.cols)}) == M


# 定义一个测试函数，测试 SpecialOnlyMatrix 类的 jordan_block 方法
def test_jordan_block():
    # 断言调用 jordan_block 方法的多种方式得到的结果应相等
    assert SpecialOnlyMatrix.jordan_block(3, 2) == SpecialOnlyMatrix.jordan_block(3, eigenvalue=2) \
            == SpecialOnlyMatrix.jordan_block(size=3, eigenvalue=2) \
            == SpecialOnlyMatrix.jordan_block(3, 2, band='upper') \
            == SpecialOnlyMatrix.jordan_block(
                size=3, eigenval=2, eigenvalue=2) \
            == Matrix([
                [2, 1, 0],
                [0, 2, 1],
                [0, 0, 2]])

    # 断言以 lower 带参数调用 jordan_block 方法得到的结果
    assert SpecialOnlyMatrix.jordan_block(3, 2, band='lower') == Matrix([
                    [2, 0, 0],
                    [1, 2, 0],
                    [0, 1, 2]])
    # 断言调用 jordan_block 方法时缺少 eigenvalue 参数会引发 ValueError 异常
    raises(ValueError, lambda: SpecialOnlyMatrix.jordan_block(2))
    # 断言调用 jordan_block 方法时 size 参数为非整数会引发 ValueError 异常
    raises(ValueError, lambda: SpecialOnlyMatrix.jordan_block(3.5, 2))
    # 断言调用 jordan_block 方法时未指定 size 参数会引发 ValueError 异常
    raises(ValueError, lambda: SpecialOnlyMatrix.jordan_block(eigenvalue=2))
    # 断言调用 jordan_block 方法时传入不一致的 eigenvalue 参数会引发 ValueError 异常
    raises(ValueError,
    lambda: SpecialOnlyMatrix.jordan_block(
        eigenvalue=2, eigenval=4))

    # 使用别名关键字 alias 进行断言
    assert SpecialOnlyMatrix.jordan_block(size=3, eigenvalue=2) == \
        SpecialOnlyMatrix.jordan_block(size=3, eigenval=2)


# 定义一个测试函数，测试 Matrix 类的 orthogonalize 方法
def test_orthogonalize():
    # 创建一个 2x2 的矩阵 m
    m = Matrix([[1, 2], [3, 4]])
    # 断言调用 orthogonalize 方法后得到的结果与预期的列表相等
    assert m.orthogonalize(Matrix([[2], [1]])) == [Matrix([[2], [1]])]
    # 断言调用 orthogonalize 方法并设置 normalize=True 后得到的结果与预期的列表相等
    assert m.orthogonalize(Matrix([[2], [1]]), normalize=True) == \
        [Matrix([[2*sqrt(5)/5], [sqrt(5)/5]])]
    # 断言调用 orthogonalize 方法处理多个向量后得到的结果与预期的列表相等
    assert m.orthogonalize(Matrix([[1], [2]]), Matrix([[-1], [4]])) == \
        [Matrix([[1], [2]]), Matrix([[Rational(-12, 5)], [Rational(6, 5)]])]
    # 断言调用 orthogonalize 方法处理零向量后得到的结果与预期的列表相等
    assert m.orthogonalize(Matrix([[0], [0]]), Matrix([[-1], [4]])) == \
        [Matrix([[-1], [4]])]
    # 断言调用 orthogonalize 方法处理空列表后得到的结果为空列表
    assert m.orthogonalize(Matrix([[0], [0]])) == []

    # 创建一个 3x3 的矩阵 n
    n = Matrix([[9, 1, 9], [3, 6, 10], [8, 5, 2]])
    # 创建多个向量组成的列表 vecs
    vecs = [Matrix([[-5], [1]]), Matrix([[-5], [2]]), Matrix([[-5], [-2]])]
    # 断言调用 orthogonalize 方法处理 vecs 后得到的结果与预期的列表相等
    assert n.orthogonalize(*vecs) == \
        [Matrix([[-5], [1]]), Matrix([[Rational(5, 26)], [Rational(25, 26)]])]

    # 创建多个不满秩向量组成的列表 vecs
    vecs = [Matrix([0, 0, 0]), Matrix([1, 2, 3]), Matrix([1, 4, 5])]
    # 断言调用 orthogonalize 方法处理 vecs 时会引发 ValueError 异常
    raises(ValueError, lambda: Matrix.orthogonalize(*vecs, rankcheck=True))

    # 创建多个不满秩向量组成的列表 vecs
    vecs = [Matrix([1, 2, 3]), Matrix([4, 5, 6]), Matrix([7, 8, 9])]
    # 断言调用 orthogonalize 方法处理 vecs 时会引发 ValueError 异常
    raises(ValueError, lambda: Matrix.orthogonalize(*vecs, rankcheck=True))


# 定义一个空的测试函数，待完善
def test_wilkinson():
    pass
    # 调用 Matrix 类的静态方法 wilkinson，返回 wminus 和 wplus 两个矩阵
    wminus, wplus = Matrix.wilkinson(1)
    # 断言 wminus 等于指定的 3x3 矩阵
    assert wminus == Matrix([
                                [-1, 1, 0],
                                [1, 0, 1],
                                [0, 1, 1]])
    # 断言 wplus 等于指定的 3x3 矩阵
    assert wplus == Matrix([
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1]])

    # 调用 Matrix 类的静态方法 wilkinson，返回 wminus 和 wplus 两个矩阵
    wminus, wplus = Matrix.wilkinson(3)
    # 断言 wminus 等于指定的 7x7 矩阵
    assert wminus == Matrix([
                                [-3,  1,  0, 0, 0, 0, 0],
                                [1, -2,  1, 0, 0, 0, 0],
                                [0,  1, -1, 1, 0, 0, 0],
                                [0,  0,  1, 0, 1, 0, 0],
                                [0,  0,  0, 1, 1, 1, 0],
                                [0,  0,  0, 0, 1, 2, 1],
                                [0,  0,  0, 0, 0, 1, 3]])

    # 断言 wplus 等于指定的 7x7 矩阵
    assert wplus == Matrix([
                            [3, 1, 0, 0, 0, 0, 0],
                            [1, 2, 1, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 1, 2, 1],
                            [0, 0, 0, 0, 0, 1, 3]])
# 定义测试标记为XFAIL的函数test_diff，用于测试CalculusOnlyMatrix对象的求导功能
@XFAIL
def test_diff():
    # 定义符号变量x和y
    x, y = symbols('x y')
    # 创建一个CalculusOnlyMatrix对象m，包含两行一列的矩阵，元素为符号变量x和y
    m = CalculusOnlyMatrix(2, 1, [x, y])
    # 断言对m关于变量x的求导结果等于一个两行一列的矩阵，其中第一行为1，第二行为0
    # TODO: 目前因为``_MinimalMatrix``不能被sympified，该断言目前无法正常工作
    assert m.diff(x) == Matrix(2, 1, [1, 0])


# 定义测试函数test_integrate，用于测试CalculusOnlyMatrix对象的积分功能
def test_integrate():
    # 定义符号变量x和y
    x, y = symbols('x y')
    # 创建一个CalculusOnlyMatrix对象m，包含两行一列的矩阵，元素为符号变量x和y
    m = CalculusOnlyMatrix(2, 1, [x, y])
    # 断言对m关于变量x的积分结果等于一个两行一列的矩阵，其中第一行为x^2/2，第二行为y*x
    assert m.integrate(x) == Matrix(2, 1, [x**2/2, y*x])


# 定义测试函数test_jacobian2，用于测试CalculusOnlyMatrix对象的雅可比矩阵功能
def test_jacobian2():
    # 定义符号变量rho和phi
    rho, phi = symbols("rho,phi")
    # 创建一个CalculusOnlyMatrix对象X，包含三行一列的矩阵，元素为rho*cos(phi), rho*sin(phi), rho**2
    X = CalculusOnlyMatrix(3, 1, [rho*cos(phi), rho*sin(phi), rho**2])
    # 创建一个CalculusOnlyMatrix对象Y，包含两行一列的矩阵，元素为rho和phi
    Y = CalculusOnlyMatrix(2, 1, [rho, phi])
    # 创建一个2x2的Matrix对象J，表示X关于Y的雅可比矩阵
    J = Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)],
        [   2*rho,             0],
    ])
    # 断言X关于Y的雅可比矩阵等于J
    assert X.jacobian(Y) == J

    # 创建一个CalculusOnlyMatrix对象m，包含两行两列的矩阵，元素为1, 2, 3, 4
    m = CalculusOnlyMatrix(2, 2, [1, 2, 3, 4])
    # 创建一个CalculusOnlyMatrix对象m2，包含四行一列的矩阵，元素为1, 2, 3, 4
    m2 = CalculusOnlyMatrix(4, 1, [1, 2, 3, 4])
    # 断言m关于一个长度为2的Matrix对象的雅可比矩阵会引发TypeError异常
    raises(TypeError, lambda: m.jacobian(Matrix([1, 2])))
    # 断言m2关于m的雅可比矩阵会引发TypeError异常
    raises(TypeError, lambda: m2.jacobian(m))


# 定义测试函数test_limit，用于测试CalculusOnlyMatrix对象的极限功能
def test_limit():
    # 定义符号变量x和y
    x, y = symbols('x y')
    # 创建一个CalculusOnlyMatrix对象m，包含两行一列的矩阵，元素为1/x和y
    m = CalculusOnlyMatrix(2, 1, [1/x, y])
    # 断言m关于变量x在值为5时的极限结果等于一个两行一列的矩阵，其中第一行为1/5，第二行为y
    assert m.limit(x, 5) == Matrix(2, 1, [Rational(1, 5), y])


# 定义测试函数test_issue_13774，用于测试Matrix对象与列表之间乘法的异常处理功能
def test_issue_13774():
    # 创建一个3x3的Matrix对象M，元素为[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 创建一个长度为3的列表v，元素为[1, 1, 1]
    v = [1, 1, 1]
    # 断言Matrix对象M与列表v的乘法会引发TypeError异常
    raises(TypeError, lambda: M*v)
    # 断言列表v与Matrix对象M的乘法会引发TypeError异常
    raises(TypeError, lambda: v*M)


# 定义测试函数test_companion，用于测试Matrix对象的伴随矩阵生成功能
def test_companion():
    # 定义符号变量c0, c1, c2
    c0, c1, c2 = symbols('c0:3')
    # 断言使用Matrix类方法生成c0为参数的伴随矩阵等于一个包含一个元素-c0的Matrix对象
    assert Matrix.companion(Poly([1, c0], x)) == Matrix([-c0])
    # 断言使用Matrix类方法生成由c1, c0参数的伴随矩阵等于一个2x2的Matrix对象
    assert Matrix.companion(Poly([1, c1, c0], x)) == \
        Matrix([[0, -c0], [1, -c1]])
    # 断言使用Matrix类方法生成由c2, c1, c0参数的伴随矩阵等于一个3x3的Matrix对象
    assert Matrix.companion(Poly([1, c2, c1, c0], x)) == \
        Matrix([[0, 0, -c0], [1, 0, -c1], [0, 1, -c2]])


# 定义测试函数test_issue_10589，用于测试Matrix对象的元素替换功能
def test_issue_10589():
    # 定义符号变量x, y, z
    x, y, z = symbols("x, y z")
    # 创建一个包含x, y, z的列矩阵M1
    M1 = Matrix([x, y, z])
    # 将M1中的x, y, z分别替换为1, 2, 3
    M1 = M1.subs(zip([x, y, z], [1, 2, 3]))
    # 断言M1等于一个3x1的Matrix对象，元素为[[1], [2], [3]]
    assert M1 == Matrix([[1], [2], [3]])

    # 创建一个3x5的Matrix对象M2，元素为[[x, x, x, x, x], [x, x, x, x, x], [x, x, x, x, x]]
    M2 = Matrix([[x, x, x, x, x], [x, x, x, x, x], [x, x, x, x, x]])
    # 将M2中的x替换为1
    M2 = M2.subs(zip([x], [1]))
    # 断言M2等于一个3x5的Matrix对象，元素为[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    assert M2 == Matrix([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])


# 定义测试函数test_rmul_pr19860，用于测试Matrix对象的乘法重载功能
def test_rmul_pr19860():
    # 定义一个继承自ImmutableDenseMatrix的类Foo
    class Foo(ImmutableDenseMatrix):
        # 设置类属性_op_priority，比
    # 断言确保 a 不等于 My 类的实例
    assert a != My()

    # 定义一个名为 My_sympy 的类，继承自 My 类
    class My_sympy(My):
        # 定义 _sympy_ 方法，返回一个将 self 转换为 Matrix 对象的结果
        def _sympy_(self):
            return Matrix(self)

    # 断言确保 a 等于 My_sympy 类的实例
    assert a == My_sympy()
```