# `D:\src\scipysrc\sympy\sympy\polys\tests\test_polymatrix.py`

```
# 从 sympy.testing.pytest 导入 raises 函数，用于测试中预期的异常处理
from sympy.testing.pytest import raises

# 从 sympy.polys.polymatrix 模块导入 PolyMatrix 类
from sympy.polys.polymatrix import PolyMatrix
# 从 sympy.polys 模块导入 Poly 类
from sympy.polys import Poly

# 从 sympy.core.singleton 模块导入 S 对象，用于表示单例值
from sympy.core.singleton import S
# 从 sympy.matrices.dense 模块导入 Matrix 类
from sympy.matrices.dense import Matrix
# 从 sympy.polys.domains.integerring 模块导入 ZZ 对象，表示整数环
from sympy.polys.domains.integerring import ZZ
# 从 sympy.polys.domains.rationalfield 模块导入 QQ 对象，表示有理数域
from sympy.polys.domains.rationalfield import QQ

# 从 sympy.abc 模块导入 x, y 符号变量
from sympy.abc import x, y


def _test_polymatrix():
    # 创建 PolyMatrix 对象 pm1，包含多项式对象的矩阵
    pm1 = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(x**3, x), Poly(-1 + x, x)]])
    # 创建 PolyMatrix 对象 v1，使用整数环 'ZZ[x]' 初始化
    v1 = PolyMatrix([[1, 0], [-1, 0]], ring='ZZ[x]')
    # 创建 PolyMatrix 对象 m1，使用整数环 'ZZ[x]' 初始化
    m1 = PolyMatrix([[1, 0], [-1, 0]], ring='ZZ[x]')
    # 创建 PolyMatrix 对象 A，包含多项式对象的矩阵
    A = PolyMatrix([[Poly(x**2 + x, x), Poly(0, x)], \
                    [Poly(x**3 - x + 1, x), Poly(0, x)]])
    # 创建 PolyMatrix 对象 B，包含多项式对象的矩阵
    B = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(-x**2, x), Poly(x, x)]])
    
    # 断言 A 对象的环属性为 'ZZ[x]'
    assert A.ring == ZZ[x]
    # 断言 pm1*v1 的结果类型为 PolyMatrix 类型
    assert isinstance(pm1*v1, PolyMatrix)
    # 断言 pm1*v1 的结果与 A 相等
    assert pm1*v1 == A
    # 断言 pm1*m1 的结果与 A 相等
    assert pm1*m1 == A
    # 断言 v1*pm1 的结果与 B 相等
    assert v1*pm1 == B

    # 创建 PolyMatrix 对象 pm2，使用有理数域 'QQ[x]' 初始化
    pm2 = PolyMatrix([[Poly(x**2, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**2, x, domain='QQ'), \
                    Poly(x**3, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**3, x, domain='QQ')]])
    # 断言 pm2 的环属性为 'QQ[x]'
    assert pm2.ring == QQ[x]
    # 创建 PolyMatrix 对象 v2，使用整数环 'ZZ[x]' 初始化
    v2 = PolyMatrix([1, 0, 0, 0, 0, 0], ring='ZZ[x]')
    # 创建 PolyMatrix 对象 m2，使用整数环 'ZZ[x]' 初始化
    m2 = PolyMatrix([1, 0, 0, 0, 0, 0], ring='ZZ[x]')
    # 创建 PolyMatrix 对象 C，包含多项式对象的矩阵
    C = PolyMatrix([[Poly(x**2, x, domain='QQ')]])
    # 断言 pm2*v2 的结果与 C 相等
    assert pm2*v2 == C
    # 断言 pm2*m2 的结果与 C 相等
    assert pm2*m2 == C

    # 创建 PolyMatrix 对象 pm3，包含多项式对象的矩阵，使用整数环 'ZZ[x]' 初始化
    pm3 = PolyMatrix([[Poly(x**2, x), S.One]], ring='ZZ[x]')
    # 创建 PolyMatrix 对象 v3，使用 S.Half 对象乘以 pm3
    v3 = S.Half*pm3
    # 断言 v3 的结果与预期 PolyMatrix 对象相等
    assert v3 == PolyMatrix([[Poly(S.Half*x**2, x, domain='QQ'), S.Half]], ring='QQ[x]')
    # 断言 pm3 乘以 S.Half 结果与 v3 相等
    assert pm3*S.Half == v3
    # 断言 v3 的环属性为 'QQ[x]'
    assert v3.ring == QQ[x]

    # 创建 PolyMatrix 对象 pm4，包含多项式对象的矩阵，使用整数环 'ZZ' 初始化
    pm4 = PolyMatrix([[Poly(x**2, x, domain='ZZ'), Poly(-x**2, x, domain='ZZ')]])
    # 创建 PolyMatrix 对象 v4，使用整数环 'ZZ[x]' 初始化
    v4 = PolyMatrix([1, -1], ring='ZZ[x]')
    # 断言 pm4*v4 的结果与预期 PolyMatrix 对象相等
    assert pm4*v4 == PolyMatrix([[Poly(2*x**2, x, domain='ZZ')]])

    # 断言 PolyMatrix 对象 (ring=ZZ[x]) 的长度为 0
    assert len(PolyMatrix(ring=ZZ[x])) == 0
    # 断言 PolyMatrix 对象 [1, 0, 0, 1] 除以 -1 的结果与预期 PolyMatrix 对象相等
    assert PolyMatrix([1, 0, 0, 1], x)/(-1) == PolyMatrix([-1, 0, 0, -1], x)


def test_polymatrix_constructor():
    # 创建 PolyMatrix 对象 M1，包含多项式对象的矩阵，使用 'QQ[x,y]' 环初始化
    M1 = PolyMatrix([[x, y]], ring=QQ[x,y])
    # 断言 M1 的环属性为 'QQ[x,y]'
    assert M1.ring == QQ[x,y]
    # 断言 M1 的域属性为 QQ
    assert M1.domain == QQ
    # 断言 M1 的生成元组为 (x, y)
    assert M1.gens == (x, y)
    # 断言 M1 的形状为 (1, 2)
    assert M1.shape == (1, 2)
    # 断言 M1 的行数为 1
    assert M1.rows == 1
    # 断言 M1 的列数为 2
    assert M1.cols == 2
    # 断言 M1 的长度为 2
    assert len(M1) == 2
    # 断言 M1 转化为列表后的结果符合预期
    assert list(M1) == [Poly(x, (x, y), domain=QQ), Poly(y, (x, y), domain=QQ)]

    # 创建 PolyMatrix 对象 M2，包含多项式对象的矩阵，使用 'QQ[x][y]' 环初始化
    M2 = PolyMatrix([[x, y]], ring=QQ[x][y])
    # 断言 M2 的环属性为 'QQ[x][y]'
    assert M2.ring == QQ[x][y]
    # 断言 M2 的域属性为 QQ[x]
    assert M2.domain == QQ[x]
    # 断言 M2 的生成元组为 (y,)
    assert M2.gens == (y,)
    # 断言 M2 的形状为 (1, 2)
    assert M2.shape == (1, 2)
    # 断言 M2 的行数为 1
    assert M2.rows == 1
    # 断言 M2 的列数为 2
    assert M2.cols == 2
    # 断言 M2 的长度为 2
    assert len(M2) == 2
    # 断言 M2 转化为列表后的结果符合预期
    assert list(M2) == [Poly(x, (y,), domain=QQ[x]), Poly(y, (y,), domain=QQ[x])]

    # 断言 PolyMatrix 对象 [[x, y]] 使用 y 初始化的结果与 'ZZ.frac_field(x)[y]' 环初始化的 Poly
    # 断言：使用 PolyMatrix 类创建对象，并检查其是否等于指定的 PolyMatrix 对象
    assert PolyMatrix(1, 2, lambda i,j: [x,y][j]) == PolyMatrix([[x, y]], ring=QQ[x,y])
    
    # 断言：使用 PolyMatrix 类创建对象，并检查其形状是否为 (0, 2)
    assert PolyMatrix(0, 2, [], x, y).shape == (0, 2)
    
    # 断言：使用 PolyMatrix 类创建对象，并检查其形状是否为 (2, 0)
    assert PolyMatrix(2, 0, [], x, y).shape == (2, 0)
    
    # 断言：使用 PolyMatrix 类创建对象，并检查其形状是否为 (2, 0)
    assert PolyMatrix([[], []], x, y).shape == (2, 0)
    
    # 断言：使用 PolyMatrix 类创建对象，分别检查是否等于 PolyMatrix(0, 0, [], ring=QQ[x,y]) 和 PolyMatrix([], ring=QQ[x,y])
    assert PolyMatrix(ring=QQ[x,y]) == PolyMatrix(0, 0, [], ring=QQ[x,y]) == PolyMatrix([], ring=QQ[x,y])
    
    # 断言：使用 PolyMatrix 类创建对象时应引发 TypeError 异常，因为参数不足
    raises(TypeError, lambda: PolyMatrix())
    
    # 断言：使用 PolyMatrix 类创建对象时应引发 TypeError 异常，因为传递了一个参数而期望的是两个
    raises(TypeError, lambda: PolyMatrix(1))

    # 断言：使用 PolyMatrix 类创建对象，并检查其是否等于指定的 PolyMatrix 对象
    assert PolyMatrix([Poly(x), Poly(y)]) == PolyMatrix([[x], [y]], ring=ZZ[x,y])

    # XXX: 可能是 parallel_poly_from_expr 中的一个 bug（x 从 gens 和 domain 中丢失）：
    # 断言：使用 PolyMatrix 类创建对象，并检查其是否等于指定的 PolyMatrix 对象
    assert PolyMatrix([Poly(y, x), 1]) == PolyMatrix([[y], [1]], ring=QQ[y])
def test_polymatrix_eq():
    # 检查两个相同的 PolyMatrix 对象是否相等
    assert (PolyMatrix([x]) == PolyMatrix([x])) is True
    # 检查包含不同多项式的 PolyMatrix 对象是否不相等
    assert (PolyMatrix([y]) == PolyMatrix([x])) is False
    # 检查两个相同的 PolyMatrix 对象是否不等价
    assert (PolyMatrix([x]) != PolyMatrix([x])) is False
    # 检查包含不同多项式的 PolyMatrix 对象是否不等价
    assert (PolyMatrix([y]) != PolyMatrix([x])) is True

    # 检查 PolyMatrix 和 Matrix 对象之间的不等价关系
    assert PolyMatrix([[x, y]]) != PolyMatrix([x, y]) == PolyMatrix([[x], [y]])

    # 检查在不同环境下的 PolyMatrix 对象是否不等价
    assert PolyMatrix([x], ring=QQ[x]) != PolyMatrix([x], ring=ZZ[x])

    # 检查 PolyMatrix 和 Matrix 对象之间的不等价关系
    assert PolyMatrix([x]) != Matrix([x])
    # 将 PolyMatrix 转换为 Matrix 对象，检查它们是否相等
    assert PolyMatrix([x]).to_Matrix() == Matrix([x])

    # 检查在指定环境下的相同值的 PolyMatrix 对象是否相等
    assert PolyMatrix([1], x) == PolyMatrix([1], x)
    # 检查在不同变量环境下的相同值的 PolyMatrix 对象是否不等价
    assert PolyMatrix([1], x) != PolyMatrix([1], y)


def test_polymatrix_from_Matrix():
    # 将 Matrix 对象转换为 PolyMatrix 对象，并检查环境是否正确设置
    assert PolyMatrix.from_Matrix(Matrix([1, 2]), x) == PolyMatrix([1, 2], x, ring=QQ[x])
    # 将 Matrix 对象转换为 PolyMatrix 对象，并检查环境是否正确设置
    assert PolyMatrix.from_Matrix(Matrix([1]), ring=QQ[x]) == PolyMatrix([1], x)
    # 创建两个不同的 PolyMatrix 对象，并检查它们是否不等价
    pmx = PolyMatrix([1, 2], x)
    pmy = PolyMatrix([1, 2], y)
    assert pmx != pmy
    # 将 PolyMatrix 对象的变量环境设置为 y，并检查它与另一个 PolyMatrix 对象是否相等
    assert pmx.set_gens(y) == pmy


def test_polymatrix_repr():
    # 检查 PolyMatrix 对象的字符串表示是否正确
    assert repr(PolyMatrix([[1, 2]], x)) == 'PolyMatrix([[1, 2]], ring=QQ[x])'
    # 检查 PolyMatrix 对象的字符串表示是否正确
    assert repr(PolyMatrix(0, 2, [], x)) == 'PolyMatrix(0, 2, [], ring=QQ[x])'


def test_polymatrix_getitem():
    # 创建一个 PolyMatrix 对象 M
    M = PolyMatrix([[1, 2], [3, 4]], x)
    # 检查获取整个 PolyMatrix 对象是否正确
    assert M[:, :] == M
    # 检查获取 PolyMatrix 对象的部分行是否正确
    assert M[0, :] == PolyMatrix([[1, 2]], x)
    # 检查获取 PolyMatrix 对象的部分列是否正确
    assert M[:, 0] == PolyMatrix([1, 3], x)
    # 检查获取 PolyMatrix 对象的单个元素是否正确
    assert M[0, 0] == Poly(1, x, domain=QQ)
    # 检查获取 PolyMatrix 对象的整行是否正确
    assert M[0] == Poly(1, x, domain=QQ)
    # 检查获取 PolyMatrix 对象的前两行是否正确
    assert M[:2] == [Poly(1, x, domain=QQ), Poly(2, x, domain=QQ)]


def test_polymatrix_arithmetic():
    # 创建一个 PolyMatrix 对象 M
    M = PolyMatrix([[1, 2], [3, 4]], x)
    # 检查 PolyMatrix 对象的加法运算是否正确
    assert M + M == PolyMatrix([[2, 4], [6, 8]], x)
    # 检查 PolyMatrix 对象的减法运算是否正确
    assert M - M == PolyMatrix([[0, 0], [0, 0]], x)
    # 检查 PolyMatrix 对象的取负运算是否正确
    assert -M == PolyMatrix([[-1, -2], [-3, -4]], x)
    # 检查 PolyMatrix 对象与非 PolyMatrix 对象进行加法运算时是否引发 TypeError
    raises(TypeError, lambda: M + 1)
    raises(TypeError, lambda: M - 1)
    raises(TypeError, lambda: 1 + M)
    raises(TypeError, lambda: 1 - M)

    # 检查 PolyMatrix 对象的乘法运算是否正确
    assert M * M == PolyMatrix([[7, 10], [15, 22]], x)
    assert 2 * M == PolyMatrix([[2, 4], [6, 8]], x)
    assert M * 2 == PolyMatrix([[2, 4], [6, 8]], x)
    assert S(2) * M == PolyMatrix([[2, 4], [6, 8]], x)
    assert M * S(2) == PolyMatrix([[2, 4], [6, 8]], x)
    # 检查 PolyMatrix 对象与非 PolyMatrix 对象进行乘法运算时是否引发 TypeError
    raises(TypeError, lambda: [] * M)
    raises(TypeError, lambda: M * [])

    # 创建一个指定环境的 PolyMatrix 对象 M2
    M2 = PolyMatrix([[1, 2]], ring=ZZ[x])
    # 检查 PolyMatrix 对象与有理数进行乘法运算是否正确
    assert S.Half * M2 == PolyMatrix([[S.Half, 1]], ring=QQ[x])
    assert M2 * S.Half == PolyMatrix([[S.Half, 1]], ring=QQ[x])

    # 检查 PolyMatrix 对象的除法运算是否正确
    assert M / 2 == PolyMatrix([[S(1)/2, 1], [S(3)/2, 2]], x)
    assert M / Poly(2, x) == PolyMatrix([[S(1)/2, 1], [S(3)/2, 2]], x)
    # 检查 PolyMatrix 对象与非 PolyMatrix 对象进行除法运算时是否引发 TypeError
    raises(TypeError, lambda: M / [])


def test_polymatrix_manipulations():
    # 创建一个 PolyMatrix 对象 M1
    M1 = PolyMatrix([[1, 2], [3, 4]], x)
    # 检查 PolyMatrix 对象的转置操作是否正确
    assert M1.transpose() == PolyMatrix([[1, 3], [2, 4]], x)
    # 创建一个 PolyMatrix 对象 M2
    M2 = PolyMatrix([[5, 6], [7, 8]], x)
    # 检查 PolyMatrix 对象的行连接操作是否正确
    assert M1.row_join(M2) == PolyMatrix([[1, 2, 5, 6], [3, 4, 7, 8]], x)
    # 检查 PolyMatrix 对象的列连接操作是否正确
    assert M1.col_join(M2) == PolyMatrix([[1, 2], [3, 4], [5, 6], [7, 8]], x)
    # 检查 PolyMatrix 对象的元素级操作是否正确
    assert M1.applyfunc(lambda e: 2*e) == PolyMatrix([[2, 4], [6, 8]], x)


def test_polymatrix_ones_zeros():
    # 这个测试函数未提供完整的代码，因此不需要进一步注释
    pass
    # 断言：使用 PolyMatrix.zeros 方法创建一个 1x2 的多项式矩阵，其中元素为零，并与给定的多项式变量 x 相关联，结果应为 PolyMatrix([[0, 0]], x)
    assert PolyMatrix.zeros(1, 2, x) == PolyMatrix([[0, 0]], x)
    
    # 断言：使用 PolyMatrix.eye 方法创建一个 2x2 的单位多项式矩阵，其中对角线元素为 1，其余为 0，并与给定的多项式变量 x 相关联，结果应为 PolyMatrix([[1, 0], [0, 1]], x)
    assert PolyMatrix.eye(2, x) == PolyMatrix([[1, 0], [0, 1]], x)
# 定义一个测试函数，用于测试多项式矩阵的行简化阶梯形式（rref）方法
def test_polymatrix_rref():
    # 创建一个多项式矩阵 M，其元素为 [[1, 2], [3, 4]]，变量为 x
    M = PolyMatrix([[1, 2], [3, 4]], x)
    # 断言行简化阶梯形式（rref）方法应返回单位矩阵和主列索引元组 (0, 1)
    assert M.rref() == (PolyMatrix.eye(2, x), (0, 1))
    # 断言当输入不符合要求时，应引发 ValueError 异常，如在环 ZZ[x] 下使用 [1, 2] 创建矩阵
    raises(ValueError, lambda: PolyMatrix([1, 2], ring=ZZ[x]).rref())
    # 断言当输入不符合要求时，应引发 ValueError 异常，如在环 QQ[x] 下使用 [1, x] 创建矩阵
    raises(ValueError, lambda: PolyMatrix([1, x], ring=QQ[x]).rref())


# 定义一个测试函数，用于测试多项式矩阵的零空间（nullspace）方法
def test_polymatrix_nullspace():
    # 创建一个多项式矩阵 M，其元素为 [[1, 2], [3, 6]]，变量为 x
    M = PolyMatrix([[1, 2], [3, 6]], x)
    # 断言零空间（nullspace）方法应返回 [-2*x + 1] 的列表
    assert M.nullspace() == [PolyMatrix([-2, 1], x)]
    # 断言当输入不符合要求时，应引发 ValueError 异常，如在环 ZZ[x] 下使用 [1, 2] 创建矩阵
    raises(ValueError, lambda: PolyMatrix([1, 2], ring=ZZ[x]).nullspace())
    # 断言当输入不符合要求时，应引发 ValueError 异常，如在环 QQ[x] 下使用 [1, x] 创建矩阵
    raises(ValueError, lambda: PolyMatrix([1, x], ring=QQ[x]).nullspace())
    # 断言矩阵 M 的秩应为 1
    assert M.rank() == 1
```