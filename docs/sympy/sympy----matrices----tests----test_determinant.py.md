# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_determinant.py`

```
# 导入需要的库和模块
import random
import pytest
# 从 sympy 库中导入复数单位 I 和有理数 Rational
from sympy.core.numbers import I
from sympy.core.numbers import Rational
# 从 sympy 库中导入符号和符号集合
from sympy.core.symbol import (Symbol, symbols)
# 从 sympy 库中导入开平方函数 sqrt
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy 库中导入多项式工具 Poly
from sympy.polys.polytools import Poly
# 从 sympy 库中导入矩阵类 Matrix 和单位矩阵 eye、全一矩阵 ones
from sympy.matrices import Matrix, eye, ones
# 从 sympy 库中导入预定义的符号 x, y, z
from sympy.abc import x, y, z
# 从 sympy 库中导入测试工具 raises
from sympy.testing.pytest import raises
# 从 sympy 库中导入矩阵异常类 NonSquareMatrixError
from sympy.matrices.exceptions import NonSquareMatrixError
# 从 sympy 库中导入组合数学函数 factorial 和 subfactorial
from sympy.functions.combinatorial.factorials import factorial, subfactorial

# 使用 pytest 的 parametrize 装饰器，对 test_eval_determinant 函数进行多参数化测试
@pytest.mark.parametrize("method", [
    # 直接评估这些方法，因为它们不通过 M.det() 方法调用
    Matrix._eval_det_bareiss, Matrix._eval_det_berkowitz,
    Matrix._eval_det_bird, Matrix._eval_det_laplace, Matrix._eval_det_lu
])
# 对 test_eval_determinant 函数进行多参数化测试，参数包括 M 和 sol 的组合
@pytest.mark.parametrize("M, sol", [
    (Matrix(), 1),          # 空矩阵的行列式为 1
    (Matrix([[0]]), 0),     # 1x1 全零矩阵的行列式为 0
    (Matrix([[5]]), 5),     # 1x1 元素为 5 的矩阵的行列式为 5
])
def test_eval_determinant(method, M, sol):
    # 断言调用方法后的返回值与预期结果相等
    assert method(M) == sol

# 使用 pytest 的 parametrize 装饰器，对 test_determinant 函数进行多参数化测试
@pytest.mark.parametrize("method", [
    "domain-ge", "bareiss", "berkowitz", "bird", "laplace", "lu"])
# 对 test_determinant 函数进行多参数化测试，参数包括 M 和 sol 的组合
@pytest.mark.parametrize("M, sol", [
    # 不同大小的矩阵和预期的行列式结果
    (Matrix(( (-3,  2), ( 8, -5) )), -1),
    (Matrix(( (x,   1), (y, 2*y) )), 2*x*y - y),
    (Matrix(( (1, 1, 1), (1, 2, 3), (1, 3, 6) )), 1),
    (Matrix(( ( 3, -2,  0, 5), (-2,  1, -2, 2), ( 0, -2,  5, 0), ( 5,  0,  3, 4) )), -289),
    (Matrix(( ( 1,  2,  3,  4), ( 5,  6,  7,  8), ( 9, 10, 11, 12), (13, 14, 15, 16) )), 0),
    (Matrix(( (3, 2, 0, 0, 0), (0, 3, 2, 0, 0), (0, 0, 3, 2, 0), (0, 0, 0, 3, 2), (2, 0, 0, 0, 3) )), 275),
    (Matrix(( ( 3,  0,  0, 0), (-2,  1,  0, 0), ( 0, -2,  5, 0), ( 5,  0,  3, 4) )), 60),
    (Matrix(( ( 1,  0,  0,  0), ( 5,  0,  0,  0), ( 9, 10, 11, 0), (13, 14, 15, 16) )), 0),
    (Matrix(( (3, 2, 0, 0, 0), (0, 3, 2, 0, 0), (0, 0, 3, 2, 0), (0, 0, 0, 3, 2), (0, 0, 0, 0, 3) )), 243),
    (Matrix(( (1, 0,  1,  2, 12), (2, 0,  1,  1,  4), (2, 1,  1, -1,  3), (3, 2, -1,  1,  8), (1, 1,  1,  0,  6) )), -55),
    (Matrix(( (-5,  2,  3,  4,  5), ( 1, -4,  3,  4,  5), ( 1,  2, -3,  4,  5), ( 1,  2,  3, -2,  5), ( 1,  2,  3,  4, -1) )), 11664),
    (Matrix(( ( 2,  7, -1, 3, 2), ( 0,  0,  1, 0, 1), (-2,  0,  7, 0, 2), (-3, -2,  4, 5, 3), ( 1,  0,  0, 0, 1) )), 123),
    (Matrix(( (x, y, z), (1, 0, 0), (y, z, x) )), z**2 - x*y),
])
def test_determinant(method, M, sol):
    # 断言调用 M.det() 方法计算行列式后的返回值与预期结果相等
    assert M.det(method=method) == sol

# 定义测试函数 test_issue_13835
def test_issue_13835():
    # 使用 symbols 函数创建符号 a
    a = symbols('a')
    # 定义 lambda 函数 M，生成一个 n 阶方阵，元素为 i + a*j
    M = lambda n: Matrix([[i + a*j for i in range(n)] for j in range(n)])
    # 断言：调用 M 类创建一个对象，并计算其行列式是否为 0
    assert M(5).det() == 0
    # 断言：调用 M 类创建一个对象，并计算其行列式是否为 0
    assert M(6).det() == 0
    # 断言：调用 M 类创建一个对象，并计算其行列式是否为 0
    assert M(7).det() == 0
# 定义一个测试函数，用于测试特定问题（Issue 14517）
def test_issue_14517():
    # 创建一个4x4的复数矩阵M
    M = Matrix([
        [   0, 10*I,    10*I,       0],
        [10*I,    0,       0,    10*I],
        [10*I,    0, 5 + 2*I,    10*I],
        [   0, 10*I,    10*I, 5 + 2*I]])
    # 计算矩阵M的特征值
    ev = M.eigenvals()
    # 随机选择一个特征值进行测试，计算可能会有点慢
    test_ev = random.choice(list(ev.keys()))
    # 断言特征值test_ev是矩阵M减去test_ev乘以单位矩阵后的行列式为零
    assert (M - test_ev*eye(4)).det() == 0


# 使用pytest的参数化功能定义多组参数进行测试
@pytest.mark.parametrize("method", [
    "bareis", "det_lu", "det_LU", "Bareis", "BAREISS", "BERKOWITZ", "LU"])
@pytest.mark.parametrize("M, sol", [
    # 第一组参数：一个4x4矩阵和其行列式的确定值
    (Matrix(( ( 3, -2,  0, 5),
              (-2,  1, -2, 2),
              ( 0, -2,  5, 0),
              ( 5,  0,  3, 4) )), -289),
    # 第二组参数：一个5x5矩阵和其行列式的确定值
    (Matrix(( (-5,  2,  3,  4,  5),
              ( 1, -4,  3,  4,  5),
              ( 1,  2, -3,  4,  5),
              ( 1,  2,  3, -2,  5),
              ( 1,  2,  3,  4, -1) )), 11664),
])
# 定义测试函数，测试矩阵行列式计算的各种方法
def test_legacy_det(method, M, sol):
    # 最小支持legacy keys（方法名）在det()函数中的使用
    # 部分内容来自test_determinant()
    assert M.det(method=method) == sol


# 定义一个函数，返回一个n阶单位矩阵的对应Matrix对象
def eye_Determinant(n):
    return Matrix(n, n, lambda i, j: int(i == j))


# 定义一个函数，返回一个n阶全零矩阵的对应Matrix对象
def zeros_Determinant(n):
    return Matrix(n, n, lambda i, j: 0)


# 定义测试函数，测试Matrix对象的行列式计算
def test_det():
    # 创建一个2行3列的Matrix对象a
    a = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    # 断言调用a.det()会引发NonSquareMatrixError异常
    raises(NonSquareMatrixError, lambda: a.det())

    # 创建一个2阶全零矩阵z和一个2阶单位矩阵ey
    z = zeros_Determinant(2)
    ey = eye_Determinant(2)
    # 断言z的行列式为0，ey的行列式为1
    assert z.det() == 0
    assert ey.det() == 1

    # 创建一个Symbol对象x
    x = Symbol('x')
    # 创建不同尺寸和不同元素的Matrix对象a到h
    a = Matrix(0, 0, [])
    b = Matrix(1, 1, [5])
    c = Matrix(2, 2, [1, 2, 3, 4])
    d = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 8])
    e = Matrix(4, 4,
        [x, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 11, 12, 13, 14, 14])
    from sympy.abc import i, j, k, l, m, n
    f = Matrix(3, 3, [i, l, m, 0, j, n, 0, 0, k])
    g = Matrix(3, 3, [i, 0, 0, l, j, 0, m, n, k])
    h = Matrix(3, 3, [x**3, 0, 0, i, x**-1, 0, j, k, x**-2])
    # 'det'方法的关键字在4x4矩阵之前不会起作用，因此不需要在小矩阵上测试所有方法

    # 断言各个Matrix对象的行列式计算结果符合预期
    assert a.det() == 1
    assert b.det() == 5
    assert c.det() == -2
    assert d.det() == 3
    assert e.det() == 4*x - 24
    assert e.det(method="domain-ge") == 4*x - 24
    assert e.det(method='bareiss') == 4*x - 24
    assert e.det(method='berkowitz') == 4*x - 24
    assert f.det() == i*j*k
    assert g.det() == i*j*k
    assert h.det() == 1
    raises(ValueError, lambda: e.det(iszerofunc="test"))


# 定义测试函数，测试Matrix对象的永久(permanent)
def test_permanent():
    # 创建一个3x3的Matrix对象M
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 断言M的永久等于450
    assert M.per() == 450
    # 对于不同尺寸的单位矩阵，验证其永久计算结果正确
    for i in range(1, 12):
        assert ones(i, i).per() == ones(i, i).T.per() == factorial(i)
        assert (ones(i, i)-eye(i)).per() == (ones(i, i)-eye(i)).T.per() == subfactorial(i)

    # 创建一个Symbol对象序列a1到a5
    a1, a2, a3, a4, a5 = symbols('a_1 a_2 a_3 a_4 a_5')
    # 创建一个5行1列的Matrix对象M
    M = Matrix([a1, a2, a3, a4, a5])
    # 断言M的永久等于M的转置的永久，且等于a1 + a2 + a3 + a4 + a5
    assert M.per() == M.T.per() == a1 + a2 + a3 + a4 + a5


# 定义测试函数，测试Matrix对象的伴随矩阵(adjugate)
def test_adjugate():
    # 创建一个Symbol对象x
    x = Symbol('x')
    # 创建一个4x4的Matrix对象e
    e = Matrix(4, 4,
        [x, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 11, 12, 13, 14, 14])
    # 创建一个 4x4 的矩阵对象，包含特定的整数和变量表达式
    adj = Matrix([
        [   4,         -8,         4,         0],
        [  76, -14*x - 68,  14*x - 8, -4*x + 24],
        [-122, 17*x + 142, -21*x + 4,  8*x - 48],
        [  48,  -4*x - 72,       8*x, -4*x + 24]])
    
    # 使用默认方法计算矩阵 e 的伴随矩阵，并断言结果与预期的 adj 矩阵相等
    assert e.adjugate() == adj
    
    # 使用 'bareiss' 方法计算矩阵 e 的伴随矩阵，并断言结果与预期的 adj 矩阵相等
    assert e.adjugate(method='bareiss') == adj
    
    # 使用 'berkowitz' 方法计算矩阵 e 的伴随矩阵，并断言结果与预期的 adj 矩阵相等
    assert e.adjugate(method='berkowitz') == adj
    
    # 使用 'bird' 方法计算矩阵 e 的伴随矩阵，并断言结果与预期的 adj 矩阵相等
    assert e.adjugate(method='bird') == adj
    
    # 使用 'laplace' 方法计算矩阵 e 的伴随矩阵，并断言结果与预期的 adj 矩阵相等
    assert e.adjugate(method='laplace') == adj
    
    # 创建一个非方阵的矩阵对象 a，尝试计算其伴随矩阵，断言会引发 NonSquareMatrixError 异常
    a = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    raises(NonSquareMatrixError, lambda: a.adjugate())
# 定义一个测试函数 test_util
def test_util():
    # 将 Rational 类别名 R 赋值为当前作用域内的 Rational
    R = Rational

    # 创建一个行向量 v1，元素为 [1, 2, 3]
    v1 = Matrix(1, 3, [1, 2, 3])
    # 创建一个行向量 v2，元素为 [3, 4, 5]
    v2 = Matrix(1, 3, [3, 4, 5])
    # 断言 v1 的范数等于 sqrt(14)
    assert v1.norm() == sqrt(14)
    # 断言 v1 在 v2 上的投影向量为 [39/25, 52/25, 13/5]
    assert v1.project(v2) == Matrix(1, 3, [R(39)/25, R(52)/25, R(13)/5])
    # 断言创建一个 1x2 的零矩阵与给定的 [0, 0] 行向量相等
    assert Matrix.zeros(1, 2) == Matrix(1, 2, [0, 0])
    # 断言创建一个 1x2 的全1矩阵与给定的 [1, 1] 行向量相等
    assert ones(1, 2) == Matrix(1, 2, [1, 1])
    # 断言 v1 的复制与其自身相等
    assert v1.copy() == v1
    # 断言单位矩阵与其余因子矩阵相等
    assert eye(3) == eye(3).cofactor_matrix()
    
    # 创建一个测试矩阵 test
    test = Matrix([[1, 3, 2], [2, 6, 3], [2, 3, 6]])
    # 断言 test 的余子式矩阵与给定矩阵相等
    assert test.cofactor_matrix() == \
        Matrix([[27, -6, -6], [-12, 2, 3], [-3, 1, 0]])
    
    # 重新赋值 test 为另一个测试矩阵
    test = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 断言 test 的余子式矩阵与给定矩阵相等
    assert test.cofactor_matrix() == \
        Matrix([[-3, 6, -3], [6, -12, 6], [-3, 6, -3]])

# 定义另一个测试函数 test_cofactor_and_minors
def test_cofactor_and_minors():
    # 创建符号变量 x
    x = Symbol('x')
    # 创建一个 4x4 矩阵 e，包含给定的元素
    e = Matrix(4, 4, [x, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 11, 12, 13, 14, 14])

    # 创建一个 3x3 矩阵 m，从 e 中提取特定位置的子矩阵
    m = Matrix([
        [ x,  1,  3],
        [ 2,  9, 11],
        [12, 13, 14]])
    
    # 创建一个 3x3 矩阵 sub，从 e 中提取特定位置的子矩阵
    sub = Matrix([
            [x, 1,  2],
            [4, 5,  6],
            [2, 9, 10]])

    # 断言 e 中指定位置的子矩阵与给定矩阵 m 相等
    assert e.minor_submatrix(1, 2) == m
    # 断言 e 中指定位置的子矩阵与给定矩阵 sub 相等
    assert e.minor_submatrix(-1, -1) == sub
    # 断言 e 中指定位置的子矩阵的次元的值与给定的值相等
    assert e.minor(1, 2) == -17*x - 142
    # 断言 e 中指定位置的子矩阵的余子式的值与给定的值相等
    assert e.cofactor(1, 2) == 17*x + 142
    # 断言 e 的余子式矩阵与给定的矩阵 cm 相等
    assert e.cofactor_matrix() == cm
    # 断言 e 的余子式矩阵（使用不同的计算方法）与给定的矩阵 cm 相等
    assert e.cofactor_matrix(method="bareiss") == cm
    assert e.cofactor_matrix(method="berkowitz") == cm
    assert e.cofactor_matrix(method="bird") == cm
    assert e.cofactor_matrix(method="laplace") == cm

    # 检查是否引发指定异常
    raises(ValueError, lambda: e.cofactor(4, 5))
    raises(ValueError, lambda: e.minor(4, 5))
    raises(ValueError, lambda: e.minor_submatrix(4, 5))

    # 创建一个 2x3 矩阵 a，包含给定的元素
    a = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    # 断言 a 的指定位置的子矩阵与给定的矩阵相等
    assert a.minor_submatrix(0, 0) == Matrix([[5, 6]])

    # 检查是否引发指定异常
    raises(ValueError, lambda:
        Matrix(0, 0, []).minor_submatrix(0, 0))
    raises(NonSquareMatrixError, lambda: a.cofactor(0, 0))
    raises(NonSquareMatrixError, lambda: a.minor(0, 0))
    raises(NonSquareMatrixError, lambda: a.cofactor_matrix())

# 定义另一个测试函数 test_charpoly
def test_charpoly():
    # 创建符号变量 x, y, z, t
    x, y = Symbol('x'), Symbol('y')
    z, t = Symbol('z'), Symbol('t')

    # 从 sympy.abc 导入 a, b, c
    from sympy.abc import a,b,c

    # 创建一个 3x3 矩阵 m，包含给定的元素
    m = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    # 断言单位矩阵的特征多项式（行列式）与给定的多项式对象相等
    assert eye_Determinant(3).charpoly(x) == Poly((x - 1)**3, x)
    assert eye_Determinant(3).charpoly(y) == Poly((y - 1)**3, y)
    # 断言 m 的特征多项式与给定的多项式对象相等
    assert m.charpoly() == Poly(x**3 - 15*x**2 - 18*x, x)
    # 检查是否引发指定异常
    raises(NonSquareMatrixError, lambda: Matrix([[1], [2]]).charpoly())

    # 创建一个 4x4 零矩阵 n，包含给定的元素
    n = Matrix(4, 4, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 断言 n 的特征多项式与给定的多项式对象相等
    assert n.charpoly() == Poly(x**4, x)

    # 创建一个 4x4 矩阵 n，包含给定的元素
    n = Matrix(4, 4, [45, 0, 0, 0, 0, 23, 0, 0, 0, 0, 87, 0, 0, 0, 0, 12])
    # 断言 n 的特征多项式与给定的多项式对象相等
    assert n.charpoly() == Poly(x**4 - 167*x**3 + 8811*x**2 -
```