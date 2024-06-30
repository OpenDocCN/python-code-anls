# `D:\src\scipysrc\sympy\sympy\external\tests\test_numpy.py`

```
#`
# This testfile tests SymPy <-> NumPy compatibility
# 不要在这里测试任何 SymPy 功能，只测试与 NumPy 的纯粹交互。
# 总是为任何可以在纯 Python 中测试的内容编写常规的 SymPy 测试（不使用 numpy）。这里我们测试用户在使用 SymPy 和 NumPy 时可能需要的所有内容。

from sympy.external.importtools import version_tuple
from sympy.external import import_module

# 尝试导入 numpy 模块，如果成功，保存其导入的内容
numpy = import_module('numpy')
if numpy:
    array, matrix, ndarray = numpy.array, numpy.matrix, numpy.ndarray
else:
    # 如果 numpy 导入失败，标记测试为禁用
    disabled = True

from sympy.core.numbers import (Float, Integer, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import (Matrix, list2numpy, matrix2numpy, symarray)
from sympy.utilities.lambdify import lambdify
import sympy

import mpmath
from sympy.abc import x, y, z
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.exceptions import ignore_warnings
from sympy.testing.pytest import raises

# 首先，系统性地检查所有操作是否实现且不会抛出异常

def test_systematic_basic():
    def s(sympy_object, numpy_array):
        # 定义一个函数，测试 sympy 对象和 numpy 数组的各种运算
        _ = [sympy_object + numpy_array,
        numpy_array + sympy_object,
        sympy_object - numpy_array,
        numpy_array - sympy_object,
        sympy_object * numpy_array,
        numpy_array * sympy_object,
        sympy_object / numpy_array,
        numpy_array / sympy_object,
        sympy_object ** numpy_array,
        numpy_array ** sympy_object]
    x = Symbol("x")
    y = Symbol("y")
    # 定义一系列 sympy 对象
    sympy_objs = [
        Rational(2, 3),
        Float("1.3"),
        x,
        y,
        pow(x, y)*y,
        Integer(5),
        Float(5.5),
    ]
    # 定义一系列 numpy 对象
    numpy_objs = [
        array([1]),
        array([3, 8, -1]),
        array([x, x**2, Rational(5)]),
        array([x/y*sin(y), 5, Rational(5)]),
    ]
    # 遍历所有 sympy 对象和 numpy 对象，调用 s 函数进行测试
    for x in sympy_objs:
        for y in numpy_objs:
            s(x, y)

# 现在进行一些随机测试，测试特定问题，并检查运算结果是否正确

def test_basics():
    one = Rational(1)
    zero = Rational(0)
    # 测试基本数组相等性
    assert array(1) == array(one)
    assert array([one]) == array([one])
    assert array([x]) == array([x])
    assert array(x) == array(Symbol("x"))
    assert array(one + x) == array(1 + x)

    X = array([one, zero, zero])
    # 检查数组 X 是否与另一个数组相等
    assert (X == array([one, zero, zero])).all()
    assert (X == array([one, 0, 0])).all()

def test_arrays():
    one = Rational(1)
    zero = Rational(0)
    X = array([one, zero, zero])
    Y = one*X
    X = array([Symbol("a") + Rational(1, 2)])
    Y = X + X
    # 测试数组运算
    assert Y == array([1 + 2*Symbol("a")])
    Y = Y + 1
    assert Y == array([2 + 2*Symbol("a")])
    Y = X - X
    assert Y == array([0])

def test_conversion1():
    a = list2numpy([x**2, x])
    # 检查转换后的对象是否为 numpy 数组
    assert isinstance(a, ndarray)
    # 断言：验证数组 a 的第一个元素是否等于 x 的平方
    assert a[0] == x**2
    # 断言：验证数组 a 的第二个元素是否等于 x
    assert a[1] == x
    # 断言：验证数组 a 的长度是否为 2
    assert len(a) == 2
    # 注释：确认这是一个数组
def test_conversion2():
    # 计算列表元素乘以2，并转换为numpy数组
    a = 2*list2numpy([x**2, x])
    # 将列表元素乘以2，并转换为numpy数组
    b = list2numpy([2*x**2, 2*x])
    # 断言a与b的所有元素相等
    assert (a == b).all()

    # 创建分数类型的对象
    one = Rational(1)
    # 创建零的分数类型对象
    zero = Rational(0)
    # 创建包含分数类型对象的numpy数组
    X = list2numpy([one, zero, zero])
    # 用one乘以X的每个元素
    Y = one*X
    # 用符号'a'和1/2的分数类型对象创建numpy数组X
    X = list2numpy([Symbol("a") + Rational(1, 2)])
    # 将X加上自身
    Y = X + X
    # 断言Y等于数组[1 + 2*Symbol("a")]
    assert Y == array([1 + 2*Symbol("a")])
    # 将Y加上1
    Y = Y + 1
    # 断言Y等于数组[2 + 2*Symbol("a")]
    assert Y == array([2 + 2*Symbol("a")])
    # 将X减去自身
    Y = X - X
    # 断言Y等于数组[0]
    assert Y == array([0])


def test_list2numpy():
    # 断言将数组[x**2, x]转换为numpy数组的结果等于list2numpy([x**2, x])的结果
    assert (array([x**2, x]) == list2numpy([x**2, x])).all()


def test_Matrix1():
    # 创建一个包含符号x和x的平方的Matrix对象
    m = Matrix([[x, x**2], [5, 2/x]])
    # 断言将Matrix对象m中x替换为2的结果等于[[2, 4], [5, 1]]的numpy数组
    assert (array(m.subs(x, 2)) == array([[2, 4], [5, 1]])).all()
    # 创建一个包含sin(x)和x的平方的Matrix对象
    m = Matrix([[sin(x), x**2], [5, 2/x]])
    # 断言将Matrix对象m中x替换为2的结果等于[[sin(2), 4], [5, 1]]的numpy数组
    assert (array(m.subs(x, 2)) == array([[sin(2), 4], [5, 1]])).all()


def test_Matrix2():
    # 创建一个包含符号x和x的平方的Matrix对象
    m = Matrix([[x, x**2], [5, 2/x]])
    # 忽略挂起的过时警告，断言将Matrix对象m中x替换为2的结果等于[[2, 4], [5, 1]]的matrix对象
    with ignore_warnings(PendingDeprecationWarning):
        assert (matrix(m.subs(x, 2)) == matrix([[2, 4], [5, 1]])).all()
    # 创建一个包含sin(x)和x的平方的Matrix对象
    m = Matrix([[sin(x), x**2], [5, 2/x]])
    # 忽略挂起的过时警告，断言将Matrix对象m中x替换为2的结果等于[[sin(2), 4], [5, 1]]的matrix对象
    with ignore_warnings(PendingDeprecationWarning):
        assert (matrix(m.subs(x, 2)) == matrix([[sin(2), 4], [5, 1]])).all()


def test_Matrix3():
    # 创建一个numpy数组a
    a = array([[2, 4], [5, 1]])
    # 断言将numpy数组a转换为Matrix对象后等于Matrix([[2, 4], [5, 1]])
    assert Matrix(a) == Matrix([[2, 4], [5, 1]])
    # 断言将numpy数组a转换为Matrix对象后不等于Matrix([[2, 4], [5, 2]])
    assert Matrix(a) != Matrix([[2, 4], [5, 2]])
    # 创建一个numpy数组a
    a = array([[sin(2), 4], [5, 1]])
    # 断言将numpy数组a转换为Matrix对象后等于Matrix([[sin(2), 4], [5, 1]])
    assert Matrix(a) == Matrix([[sin(2), 4], [5, 1]])
    # 断言将numpy数组a转换为Matrix对象后不等于Matrix([[sin(0), 4], [5, 1]]])
    assert Matrix(a) != Matrix([[sin(0), 4], [5, 1]])


def test_Matrix4():
    # 忽略挂起的过时警告，创建一个matrix对象a
    with ignore_warnings(PendingDeprecationWarning):
        a = matrix([[2, 4], [5, 1]])
    # 断言将matrix对象a转换为Matrix对象后等于Matrix([[2, 4], [5, 1]])
    assert Matrix(a) == Matrix([[2, 4], [5, 1]])
    # 断言将matrix对象a转换为Matrix对象后不等于Matrix([[2, 4], [5, 2]])
    assert Matrix(a) != Matrix([[2, 4], [5, 2]])
    # 忽略挂起的过时警告，创建一个matrix对象a
    with ignore_warnings(PendingDeprecationWarning):
        a = matrix([[sin(2), 4], [5, 1]])
    # 断言将matrix对象a转换为Matrix对象后等于Matrix([[sin(2), 4], [5, 1]])
    assert Matrix(a) == Matrix([[sin(2), 4], [5, 1]])
    # 断言将matrix对象a转换为Matrix对象后不等于Matrix([[sin(0), 4], [5, 1]])
    assert Matrix(a) != Matrix([[sin(0), 4], [5, 1]])


def test_Matrix_sum():
    # 创建一个Matrix对象M
    M = Matrix([[1, 2, 3], [x, y, x], [2*y, -50, z*x]])
    # 忽略挂起的过时警告，创建一个matrix对象m
    with ignore_warnings(PendingDeprecationWarning):
        m = matrix([[2, 3, 4], [x, 5, 6], [x, y, z**2]])
    # 断言Matrix对象M与matrix对象m的和等于Matrix([[3, 5, 7], [2*x, y + 5, x + 6], [2*y + x, y - 50, z*x + z**2]])
    assert M + m == Matrix([[3, 5, 7], [2*x, y + 5, x + 6], [2*y + x, y - 50, z*x + z**2]])
    # 断言matrix对象m与Matrix对象M的和等于Matrix([[3, 5, 7], [2*x, y + 5, x + 6], [2*y + x, y - 50, z*x + z**2]])
    assert m + M == Matrix([[3, 5, 7], [2*x, y + 5, x + 6], [2*y + x, y - 50, z*x + z**2]])
    # 断言Matrix对象M与matrix对象m的和等于M.add(m)的结果
    assert M + m == M.add(m)


def test_Matrix_mul():
    # 创建一个Matrix对象M
    M = Matrix([[1, 2, 3], [x, y, x]])
    # 忽略挂起的过时警告，创建一个matrix对象m
    with ignore_warnings(PendingDeprecationWarning):
        m = matrix([[2, 4], [x, 6], [x, z**2]])
    # 断言Matrix对象M与matrix对象m的乘积等于Matrix([[2 + 5*x, 16 + 3*z**2], [2*x + x*y + x**2, 4*x + 6*y + x*z**2]])
    assert M*m == Matrix([
        [2 + 5*x, 16 + 3*z**2],
        [2*x + x*y + x**2, 4*x + 6*y + x*z**2],
    ])

    # 断言matrix对象m与Matrix对象M的乘积等于
    # 定义一个名为 matarray 的类，用于实现自定义的数组转换
    class matarray:
        # 定义 __array__ 方法，实现将该类实例转换为 ndarray 类型的数组
        def __array__(self, dtype=object, copy=None):
            # 如果 copy 参数为 False，则抛出类型错误，因为无法禁止复制
            if copy is not None and not copy:
                raise TypeError("Cannot implement copy=False when converting Matrix to ndarray")
            # 导入 numpy 的 array 函数，返回一个包含特定数据的二维数组
            from numpy import array
            return array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # 创建一个 matarray 的实例 matarr
    matarr = matarray()
    
    # 使用 assert 语句检查 Matrix(matarr) 是否等于预期的二维数组
    assert Matrix(matarr) == Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
def test_matrix2numpy():
    # 创建一个包含符号变量的矩阵，并将其转换为 NumPy 数组
    a = matrix2numpy(Matrix([[1, x**2], [3*sin(x), 0]]))
    # 断言 a 是 ndarray 类型
    assert isinstance(a, ndarray)
    # 断言 a 的形状为 (2, 2)
    assert a.shape == (2, 2)
    # 断言 a 的元素符合预期
    assert a[0, 0] == 1
    assert a[0, 1] == x**2
    assert a[1, 0] == 3*sin(x)
    assert a[1, 1] == 0


def test_matrix2numpy_conversion():
    # 创建一个符号矩阵 a 和其对应的 NumPy 数组 b
    a = Matrix([[1, 2, sin(x)], [x**2, x, Rational(1, 2)]])
    b = array([[1, 2, sin(x)], [x**2, x, Rational(1, 2)]])
    # 断言矩阵转换为 NumPy 数组后内容相等
    assert (matrix2numpy(a) == b).all()
    # 断言转换后的数组的数据类型为 'object'
    assert matrix2numpy(a).dtype == numpy.dtype('object')

    # 使用不同的数据类型进行转换，并断言其结果
    c = matrix2numpy(Matrix([[1, 2], [10, 20]]), dtype='int8')
    d = matrix2numpy(Matrix([[1, 2], [10, 20]]), dtype='float64')
    assert c.dtype == numpy.dtype('int8')
    assert d.dtype == numpy.dtype('float64')


def test_issue_3728():
    # 对 Rational 和 Float 数据类型的操作进行断言
    assert (Rational(1, 2)*array([2*x, 0]) == array([x, 0])).all()
    assert (Rational(1, 2) + array(
        [2*x, 0]) == array([2*x + Rational(1, 2), Rational(1, 2)])).all()
    assert (Float("0.5")*array([2*x, 0]) == array([Float("1.0")*x, 0])).all()
    assert (Float("0.5") + array(
        [2*x, 0]) == array([2*x + Float("0.5"), Float("0.5")])).all()


@conserve_mpmath_dps
def test_lambdify():
    # 设置 mpmath 精度，并测试 lambdify 函数的输出精度
    mpmath.mp.dps = 16
    sin02 = mpmath.mpf("0.198669330795061215459412627")
    f = lambdify(x, sin(x), "numpy")
    prec = 1e-15
    # 断言 lambdify 函数在给定输入下的精度
    assert -prec < f(0.2) - sin02 < prec

    # 根据 NumPy 的版本，测试 lambdify 是否抛出 TypeError 或 AttributeError
    if version_tuple(numpy.__version__) >= version_tuple('1.17'):
        with raises(TypeError):
            f(x)
    else:
        with raises(AttributeError):
            f(x)


def test_lambdify_matrix():
    # 使用 lambdify 将矩阵函数转换为 NumPy 数组表达式，并进行断言
    f = lambdify(x, Matrix([[x, 2*x], [1, 2]]), [{'ImmutableMatrix': numpy.array}, "numpy"])
    assert (f(1) == array([[1, 2], [1, 2]])).all()


def test_lambdify_matrix_multi_input():
    # 创建一个多输入符号矩阵，并使用 lambdify 进行转换和断言
    M = sympy.Matrix([[x**2, x*y, x*z],
                      [y*x, y**2, y*z],
                      [z*x, z*y, z**2]])
    f = lambdify((x, y, z), M, [{'ImmutableMatrix': numpy.array}, "numpy"])

    xh, yh, zh = 1.0, 2.0, 3.0
    expected = array([[xh**2, xh*yh, xh*zh],
                      [yh*xh, yh**2, yh*zh],
                      [zh*xh, zh*yh, zh**2]])
    actual = f(xh, yh, zh)
    # 断言转换后的 NumPy 数组与预期结果非常接近
    assert numpy.allclose(actual, expected)


def test_lambdify_matrix_vec_input():
    # 创建一个矩阵向量输入的符号矩阵，并使用 lambdify 进行转换和断言
    X = sympy.DeferredVector('X')
    M = Matrix([
        [X[0]**2, X[0]*X[1], X[0]*X[2]],
        [X[1]*X[0], X[1]**2, X[1]*X[2]],
        [X[2]*X[0], X[2]*X[1], X[2]**2]])
    f = lambdify(X, M, [{'ImmutableMatrix': numpy.array}, "numpy"])

    Xh = array([1.0, 2.0, 3.0])
    expected = array([[Xh[0]**2, Xh[0]*Xh[1], Xh[0]*Xh[2]],
                      [Xh[1]*Xh[0], Xh[1]**2, Xh[1]*Xh[2]],
                      [Xh[2]*Xh[0], Xh[2]*Xh[1], Xh[2]**2]])
    actual = f(Xh)
    # 断言转换后的 NumPy 数组与预期结果非常接近
    assert numpy.allclose(actual, expected)


def test_lambdify_transl():
    # 测试 lambdify 中的翻译字典是否正确映射了 NumPy 和 SymPy 的函数和方法
    from sympy.utilities.lambdify import NUMPY_TRANSLATIONS
    for sym, mat in NUMPY_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert mat in numpy.__dict__
# 测试创建包含 SymPy 符号的 numpy 数组的功能
def test_symarray():
    """Test creation of numpy arrays of SymPy symbols."""

    import numpy as np  # 导入 numpy 库，用于数值计算
    import numpy.testing as npt  # 导入 numpy.testing 库，用于测试 numpy 数组

    # 创建三个 SymPy 符号，并将其存储在 syms 变量中
    syms = symbols('_0,_1,_2')
    # 创建一个空字符串命名的 SymPy 数组 s1 和 s2，各包含三个符号
    s1 = symarray("", 3)
    s2 = symarray("", 3)
    # 断言 s1 和 s2 数组的内容相等，使用 numpy.testing 库中的 assert_array_equal 方法
    npt.assert_array_equal(s1, np.array(syms, dtype=object))
    # 断言 s1 和 s2 数组的第一个元素相等
    assert s1[0] == s2[0]

    # 使用 'a' 作为前缀创建命名为 a 的 SymPy 数组，各包含三个符号
    a = symarray('a', 3)
    b = symarray('b', 3)
    # 断言 a 和 b 数组的第一个元素不相等
    assert not(a[0] == b[0])

    # 创建三个带有 'a_' 前缀的 SymPy 符号，并将其存储在 asyms 变量中
    asyms = symbols('a_0,a_1,a_2')
    # 断言 a 数组与 asyms 数组相等，使用 numpy.testing 库中的 assert_array_equal 方法
    npt.assert_array_equal(a, np.array(asyms, dtype=object))

    # 多维数组检查
    # 创建一个名为 'a' 的 SymPy 二维数组，形状为 (2, 3)
    a2d = symarray('a', (2, 3))
    # 断言 a2d 数组的形状为 (2, 3)
    assert a2d.shape == (2, 3)
    # 创建两个带有 'a_' 前缀的 SymPy 符号，并将其分别存储在 a00 和 a12 变量中
    a00, a12 = symbols('a_0_0,a_1_2')
    # 断言 a2d 数组的特定元素值与对应的 SymPy 符号相等
    assert a2d[0, 0] == a00
    assert a2d[1, 2] == a12

    # 创建一个名为 'a' 的 SymPy 三维数组，形状为 (2, 3, 2)
    a3d = symarray('a', (2, 3, 2))
    # 断言 a3d 数组的形状为 (2, 3, 2)
    assert a3d.shape == (2, 3, 2)
    # 创建三个带有 'a_' 前缀的 SymPy 符号，并将其分别存储在 a000、a120 和 a121 变量中
    a000, a120, a121 = symbols('a_0_0_0,a_1_2_0,a_1_2_1')
    # 断言 a3d 数组的特定元素值与对应的 SymPy 符号相等
    assert a3d[0, 0, 0] == a000
    assert a3d[1, 2, 0] == a120
    assert a3d[1, 2, 1] == a121


# 测试向量化函数的功能
def test_vectorize():
    assert (numpy.vectorize(
        sin)([1, 2, 3]) == numpy.array([sin(1), sin(2), sin(3)])).all()
```