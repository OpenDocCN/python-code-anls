# `D:\src\scipysrc\sympy\sympy\printing\tests\test_numpy.py`

```
# 从 sympy.concrete.summations 模块导入 Sum 类
from sympy.concrete.summations import Sum
# 从 sympy.core.mod 模块导入 Mod 类
from sympy.core.mod import Mod
# 从 sympy.core.relational 模块导入 Equality 和 Unequality 类
from sympy.core.relational import (Equality, Unequality)
# 从 sympy.core.symbol 模块导入 Symbol 类
from sympy.core.symbol import Symbol
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.piecewise 模块导入 Piecewise 类
from sympy.functions.elementary.piecewise import Piecewise
# 从 sympy.functions.special.gamma_functions 模块导入 polygamma 函数
from sympy.functions.special.gamma_functions import polygamma
# 从 sympy.functions.special.error_functions 模块导入 Si 和 Ci 函数
from sympy.functions.special.error_functions import (Si, Ci)
# 从 sympy.matrices.expressions.blockmatrix 模块导入 BlockMatrix 类
from sympy.matrices.expressions.blockmatrix import BlockMatrix
# 从 sympy.matrices.expressions.matexpr 模块导入 MatrixSymbol 类
from sympy.matrices.expressions.matexpr import MatrixSymbol
# 从 sympy.matrices.expressions.special 模块导入 Identity 类
from sympy.matrices.expressions.special import Identity
# 从 sympy.utilities.lambdify 模块导入 lambdify 函数
from sympy.utilities.lambdify import lambdify
# 从 sympy 模块导入 symbols, Min, Max 函数
from sympy import symbols, Min, Max

# 从 sympy.abc 模块导入 x, i, j, a, b, c, d 符号变量
from sympy.abc import x, i, j, a, b, c, d
# 从 sympy.core 模块导入 Pow 类
from sympy.core import Pow
# 从 sympy.codegen.matrix_nodes 模块导入 MatrixSolve 类
from sympy.codegen.matrix_nodes import MatrixSolve
# 从 sympy.codegen.numpy_nodes 模块导入 logaddexp, logaddexp2 函数
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
# 从 sympy.codegen.cfunctions 模块导入 log1p, expm1, hypot, log10, exp2, log2, Sqrt 函数
from sympy.codegen.cfunctions import log1p, expm1, hypot, log10, exp2, log2, Sqrt
# 从 sympy.tensor.array 模块导入 Array 类
from sympy.tensor.array import Array
# 从 sympy.tensor.array.expressions.array_expressions 模块导入 ArrayTensorProduct, ArrayAdd,
# PermuteDims, ArrayDiagonal 类
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
    PermuteDims, ArrayDiagonal
# 从 sympy.printing.numpy 模块导入 NumPyPrinter, SciPyPrinter, _numpy_known_constants,
# _numpy_known_functions, _scipy_known_constants, _scipy_known_functions 函数
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter, _numpy_known_constants, \
    _numpy_known_functions, _scipy_known_constants, _scipy_known_functions
# 从 sympy.tensor.array.expressions.from_matrix_to_array 模块导入 convert_matrix_to_array 函数
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array

# 从 sympy.testing.pytest 模块导入 skip, raises 函数
from sympy.testing.pytest import skip, raises
# 从 sympy.external 模块导入 import_module 函数
from sympy.external import import_module

# 导入 numpy 模块并将其赋值给 np 变量
np = import_module('numpy')
# 导入 jax 模块并将其赋值给 jax 变量
jax = import_module('jax')

# 如果成功导入 numpy 模块，则获取其默认浮点数信息
if np:
    deafult_float_info = np.finfo(np.array([]).dtype)
    # 将浮点数的机器精度赋值给 NUMPY_DEFAULT_EPSILON 变量
    NUMPY_DEFAULT_EPSILON = deafult_float_info.eps

# 定义测试函数 test_numpy_piecewise_regression
def test_numpy_piecewise_regression():
    """
    NumPyPrinter 需要将 Piecewise() 的选择列表打印为列表，以避免与 numpy 1.8 兼容性问题。
    在 numpy 1.9+ 中不需要这样做。参见 gh-9747 和 gh-9749 获取详细信息。
    """
    # 创建 NumPyPrinter 对象并赋值给 printer 变量
    printer = NumPyPrinter()
    # 创建 Piecewise 对象 p
    p = Piecewise((1, x < 0), (0, True))
    # 断言打印 Piecewise 对象 p 的结果符合预期字符串
    assert printer.doprint(p) == \
        'numpy.select([numpy.less(x, 0),True], [1,0], default=numpy.nan)'
    # 断言 printer 的模块导入包含特定的 numpy 方法
    assert printer.module_imports == {'numpy': {'select', 'less', 'nan'}}

# 定义测试函数 test_numpy_logaddexp
def test_numpy_logaddexp():
    # 创建 logaddexp(a, b) 表达式并赋值给 lae 变量
    lae = logaddexp(a, b)
    # 断言 NumPyPrinter 打印 lae 的结果符合预期字符串
    assert NumPyPrinter().doprint(lae) == 'numpy.logaddexp(a, b)'
    # 创建 logaddexp2(a, b) 表达式并赋值给 lae2 变量
    lae2 = logaddexp2(a, b)
    # 断言 NumPyPrinter 打印 lae2 的结果符合预期字符串
    assert NumPyPrinter().doprint(lae2) == 'numpy.logaddexp2(a, b)'

# 定义测试函数 test_sum
def test_sum():
    # 如果未成功导入 numpy 模块，则跳过该测试
    if not np:
        skip("NumPy not installed")

    # 创建求和表达式 Sum(x ** i, (i, a, b)) 并赋值给 s 变量
    s = Sum(x ** i, (i, a, b))
    # 使用 lambdify 函数将 s 转换为 numpy 函数并赋值给 f 变量
    f = lambdify((a, b, x), s, 'numpy')

    # 定义 a_, b_ 的取值范围
    a_, b_ = 0, 10
    # 创建 numpy 数组 x_，其值在 -1 到 1 之间均匀分布
    x_ = np.linspace(-1, +1, 10)
    # 断言 numpy 的 allclose 方法判断 f(a_, b_, x_) 的计算结果与预期值的接近程度
    assert np.allclose(f(a_, b_, x_), sum(x_ ** i_ for i_ in range(a_, b_ + 1)))

    # 创建求和表达式 Sum(i * x, (i, a, b)) 并赋值给 s 变量
    s = Sum(i * x, (i, a, b))
    # 使用 lambdify 函数将 s 转换为 numpy 函数并赋值给 f 变量
    f = lambdify((a, b, x), s, 'numpy')

    # 定义 a_, b_ 的取值范围
    a_, b_ = 0, 10
    # 创建 numpy 数组 x_，其值在 -1 到 1 之间均匀分布
    x_ = np.linspace(-1, +1, 10)
    # 断言 numpy 的 allclose 方法判断 f(a_, b_, x_) 的计算结果与预期值的接近程度
    assert np.allclose(f(a_, b_, x_), sum(i_ * x_ for i_ in range(a_, b_ + 1)))

# 定义测试函数 test_multiple_sums
def test_multiple_sums():
    # 如果未成功导入 numpy 模块，则跳过该测试
    if not np:
        skip("NumPy not installed")

    # 创建多重求和表达式 Sum((x + j) * i, (i, a, b), (j, c, d)) 并赋
    # 定义常量 c_ 和 d_，分别赋值为 11 和 21
    c_, d_ = 11, 21
    # 使用 NumPy 的 linspace 函数生成一个包含 10 个元素的数组 x_
    x_ = np.linspace(-1, +1, 10)
    # 断言：验证函数 f(a_, b_, c_, d_, x_) 的计算结果是否与预期一致
    assert np.allclose(f(a_, b_, c_, d_, x_),
                       # 计算表达式的求和：(x_ + j_) * i_，其中 i_ 和 j_ 分别从 a_ 到 b_ 和 c_ 到 d_ 的范围内取值
                       sum((x_ + j_) * i_ for i_ in range(a_, b_ + 1) for j_ in range(c_, d_ + 1)))
def test_codegen_einsum():
    # 检查是否导入了 NumPy，如果没有则跳过测试
    if not np:
        skip("NumPy not installed")

    # 创建符号矩阵 M 和 N，各为 2x2 的矩阵
    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)

    # 将矩阵 M 和 N 的乘积转换为数组表达式
    cg = convert_matrix_to_array(M * N)
    # 使用 numpy 将表达式 cg 编译为可执行的函数 f
    f = lambdify((M, N), cg, 'numpy')

    # 创建测试用的 numpy 数组 ma 和 mb
    ma = np.array([[1, 2], [3, 4]])
    mb = np.array([[1,-2], [-1, 3]])

    # 断言 f(ma, mb) 的结果等于 numpy 中矩阵乘积的结果
    assert (f(ma, mb) == np.matmul(ma, mb)).all()


def test_codegen_extra():
    # 检查是否导入了 NumPy，如果没有则跳过测试
    if not np:
        skip("NumPy not installed")

    # 创建符号矩阵 M、N、P、Q，各为 2x2 的矩阵
    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)
    P = MatrixSymbol("P", 2, 2)
    Q = MatrixSymbol("Q", 2, 2)

    # 创建测试用的 numpy 数组 ma、mb、mc、md
    ma = np.array([[1, 2], [3, 4]])
    mb = np.array([[1,-2], [-1, 3]])
    mc = np.array([[2, 0], [1, 2]])
    md = np.array([[1,-1], [4, 7]])

    # 测试 ArrayTensorProduct 操作
    cg = ArrayTensorProduct(M, N)
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.einsum(ma, [0, 1], mb, [2, 3])).all()

    # 测试 ArrayAdd 操作
    cg = ArrayAdd(M, N)
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == ma+mb).all()

    cg = ArrayAdd(M, N, P)
    f = lambdify((M, N, P), cg, 'numpy')
    assert (f(ma, mb, mc) == ma+mb+mc).all()

    cg = ArrayAdd(M, N, P, Q)
    f = lambdify((M, N, P, Q), cg, 'numpy')
    assert (f(ma, mb, mc, md) == ma+mb+mc+md).all()

    # 测试 PermuteDims 操作
    cg = PermuteDims(M, [1, 0])
    f = lambdify((M,), cg, 'numpy')
    assert (f(ma) == ma.T).all()

    # 测试在 ArrayTensorProduct 结果上应用 PermuteDims 和 np.transpose 的组合操作
    cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.transpose(np.einsum(ma, [0, 1], mb, [2, 3]), (1, 2, 3, 0))).all()

    # 测试 ArrayDiagonal 操作
    cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.diagonal(np.einsum(ma, [0, 1], mb, [2, 3]), axis1=1, axis2=2)).all()


def test_relational():
    # 检查是否导入了 NumPy，如果没有则跳过测试
    if not np:
        skip("NumPy not installed")

    # 创建 Equality 和 Unequality 的测试条件
    e = Equality(x, 1)
    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [False, True, False])

    e = Unequality(x, 1)
    f = lambdify((x,), e)
    assert np.array_equal(f(x_), [True, False, True])

    # 创建不同的比较条件（小于、小于等于、大于、大于等于）的测试
    e = (x < 1)
    f = lambdify((x,), e)
    assert np.array_equal(f(x_), [True, False, False])

    e = (x <= 1)
    f = lambdify((x,), e)
    assert np.array_equal(f(x_), [True, True, False])

    e = (x > 1)
    f = lambdify((x,), e)
    assert np.array_equal(f(x_), [False, False, True])

    e = (x >= 1)
    f = lambdify((x,), e)
    assert np.array_equal(f(x_), [False, True, True])


def test_mod():
    # 检查是否导入了 NumPy，如果没有则跳过测试
    if not np:
        skip("NumPy not installed")

    # 创建 Mod 操作的测试条件
    e = Mod(a, b)
    f = lambdify((a, b), e)

    # 创建不同输入的测试数组 a_ 和 b_
    a_ = np.array([0, 1, 2, 3])
    b_ = 2
    assert np.array_equal(f(a_, b_), [0, 1, 0, 1])

    a_ = np.array([0, 1, 2, 3])
    b_ = np.array([2, 2, 2, 2])
    assert np.array_equal(f(a_, b_), [0, 1, 0, 1])

    a_ = np.array([2, 3, 4, 5])
    b_ = np.array([2, 3, 4, 5])
    assert np.array_equal(f(a_, b_), [0, 0, 0, 0])
    # 检查 NumPy 是否已经导入，如果没有导入则跳过执行后续代码
    if not np:
        skip('NumPy not installed')
    
    # 创建一个表示幂运算的表达式，指定不进行求值
    expr = Pow(2, -1, evaluate=False)
    
    # 使用 lambdify 函数将表达式转换为可调用的函数 f，使用 NumPy 进行处理
    f = lambdify([], expr, 'numpy')
    
    # 断言表达式 f() 的返回值为 0.5，用于确认表达式计算的正确性
    assert f() == 0.5
# 定义一个用于测试 expm1 函数的测试函数
def test_expm1():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")

    # 使用 lambdify 将 expm1 函数转换为可以在 NumPy 中使用的函数
    f = lambdify((a,), expm1(a), 'numpy')
    # 断言计算结果与预期值的差值在一定精度范围内
    assert abs(f(1e-10) - 1e-10 - 5e-21) <= 1e-10 * NUMPY_DEFAULT_EPSILON


# 定义一个用于测试 log1p 函数的测试函数
def test_log1p():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")

    # 使用 lambdify 将 log1p 函数转换为可以在 NumPy 中使用的函数
    f = lambdify((a,), log1p(a), 'numpy')
    # 断言计算结果与预期值的差值在一定精度范围内
    assert abs(f(1e-99) - 1e-99) <= 1e-99 * NUMPY_DEFAULT_EPSILON

# 定义一个用于测试 hypot 函数的测试函数
def test_hypot():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")
        
    # 断言计算结果与预期值的差值在一定精度范围内
    assert abs(lambdify((a, b), hypot(a, b), 'numpy')(3, 4) - 5) <= NUMPY_DEFAULT_EPSILON

# 定义一个用于测试 log10 函数的测试函数
def test_log10():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")
        
    # 断言计算结果与预期值的差值在一定精度范围内
    assert abs(lambdify((a,), log10(a), 'numpy')(100) - 2) <= NUMPY_DEFAULT_EPSILON


# 定义一个用于测试 exp2 函数的测试函数
def test_exp2():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")
        
    # 断言计算结果与预期值的差值在一定精度范围内
    assert abs(lambdify((a,), exp2(a), 'numpy')(5) - 32) <= NUMPY_DEFAULT_EPSILON


# 定义一个用于测试 log2 函数的测试函数
def test_log2():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")
        
    # 断言计算结果与预期值的差值在一定精度范围内
    assert abs(lambdify((a,), log2(a), 'numpy')(256) - 8) <= NUMPY_DEFAULT_EPSILON


# 定义一个用于测试 Sqrt 函数的测试函数
def test_Sqrt():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")
        
    # 断言计算结果与预期值的差值在一定精度范围内
    assert abs(lambdify((a,), Sqrt(a), 'numpy')(4) - 2) <= NUMPY_DEFAULT_EPSILON


# 定义一个用于测试 sqrt 函数的测试函数
def test_sqrt():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")
        
    # 断言计算结果与预期值的差值在一定精度范围内
    assert abs(lambdify((a,), sqrt(a), 'numpy')(4) - 2) <= NUMPY_DEFAULT_EPSILON


# 定义一个用于测试 matsolve 函数的测试函数
def test_matsolve():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")
        
    # 创建符号矩阵 M 和 x
    M = MatrixSymbol("M", 3, 3)
    x = MatrixSymbol("x", 3, 1)

    # 定义表达式和使用 MatrixSolve 的表达式
    expr = M**(-1) * x + x
    matsolve_expr = MatrixSolve(M, x) + x

    # 使用 lambdify 将表达式转换为可在 NumPy 中使用的函数
    f = lambdify((M, x), expr)
    f_matsolve = lambdify((M, x), matsolve_expr)

    # 创建一个测试用的 3x3 数组 m0
    m0 = np.array([[1, 2, 3], [3, 2, 5], [5, 6, 7]])
    # 断言矩阵 m0 的秩为 3
    assert np.linalg.matrix_rank(m0) == 3

    # 创建一个测试用的长度为 3 的数组 x0
    x0 = np.array([3, 4, 5])

    # 断言使用 MatrixSolve 和直接表达式计算的结果相等
    assert np.allclose(f_matsolve(m0, x0), f(m0, x0))


# 定义一个用于测试 BlockMatrix 的测试函数
def test_16857():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")
        
    # 创建四个 10x3 的 MatrixSymbol 对象，并组成 BlockMatrix A
    a_1 = MatrixSymbol('a_1', 10, 3)
    a_2 = MatrixSymbol('a_2', 10, 3)
    a_3 = MatrixSymbol('a_3', 10, 3)
    a_4 = MatrixSymbol('a_4', 10, 3)
    A = BlockMatrix([[a_1, a_2], [a_3, a_4]])
    
    # 断言 BlockMatrix A 的形状为 (20, 6)
    assert A.shape == (20, 6)

    # 创建一个 NumPyPrinter 对象，用于打印 BlockMatrix A 的 NumPy 表示
    printer = NumPyPrinter()
    # 断言使用 NumPyPrinter 打印 BlockMatrix A 的结果符合预期
    assert printer.doprint(A) == 'numpy.block([[a_1, a_2], [a_3, a_4]])'


# 定义一个用于测试 issue 17006 的测试函数
def test_issue_17006():
    # 如果没有安装 NumPy，则跳过测试并提示信息
    if not np:
        skip("NumPy not installed")
        
    # 创建一个 2x2 的 MatrixSymbol 对象 M
    M = MatrixSymbol("M", 2, 2)

    # 使用 lambdify 将 M + Identity(2) 转换为可以在 NumPy 中使用的函数
    f = lambdify(M, M + Identity(2))
    
    # 创建一个 2x2 的 NumPy 数组 ma 和预期的结果数组 mr
    ma = np.array([[1, 2], [3, 4]])
    mr = np.array([[2, 2], [3, 5]])

    # 断言使用 lambdify 计算的结果与预期的结果数组相等
    assert (f(ma) == mr).all()

    # 创建一个整数符号 n
    from sympy.core.symbol import symbols
    n = symbols('n', integer=True)
    # 创建一个 n x n 的 MatrixSymbol 对象 N
    N = MatrixSymbol("M", n, n)
    # 断言尝试使用 lambdify 处理 N + Identity(n) 会抛出 NotImplementedError
    raises(NotImplementedError, lambda: lambdify(N, N + Identity(n)))


# 定义一个用于测试 JAX 兼容性的测试函数
def test_jax_tuple_compatibility():
    # 如果没有安装 Jax，则跳过测试并提示信息
    if not jax:
        skip("Jax not installed")

    # 创建三个符号变量 x, y, z
    x, y, z = symbols('x y z')
    # 定义表达式 expr
    expr = Max(x, y, z) + Min(x, y, z)
    #
    # 断言：验证两个函数调用的返回值是否在数值上全部接近
    assert np.allclose(func(*input_tuple1), func(*input_array1))
    # 断言：验证两个函数调用的返回值是否在数值上全部接近
    assert np.allclose(func(*input_tuple2), func(*input_array2))
# 测试 NumPyPrinter 类的 doprint 方法，验证对给定数组的打印输出是否正确
def test_numpy_array():
    # 断言 NumPyPrinter 实例的 doprint 方法输出符合预期的字符串格式
    assert NumPyPrinter().doprint(Array(((1, 2), (3, 5)))) == 'numpy.array([[1, 2], [3, 5]])'
    assert NumPyPrinter().doprint(Array((1, 2))) == 'numpy.array((1, 2))'

# 测试 NumPyPrinter 类中已知函数和常量的映射是否正确
def test_numpy_known_funcs_consts():
    # 断言已知常量映射表中的值与预期的 NumPy 对应值相等
    assert _numpy_known_constants['NaN'] == 'numpy.nan'
    assert _numpy_known_constants['EulerGamma'] == 'numpy.euler_gamma'

    # 断言已知函数映射表中的值与预期的 NumPy 对应函数名相等
    assert _numpy_known_functions['acos'] == 'numpy.arccos'
    assert _numpy_known_functions['log'] == 'numpy.log'

# 测试 SciPyPrinter 类中已知函数和常量的映射是否正确
def test_scipy_known_funcs_consts():
    # 断言已知常量映射表中的值与预期的 SciPy 对应值相等
    assert _scipy_known_constants['GoldenRatio'] == 'scipy.constants.golden_ratio'
    assert _scipy_known_constants['Pi'] == 'scipy.constants.pi'

    # 断言已知函数映射表中的值与预期的 SciPy 对应函数名相等
    assert _scipy_known_functions['erf'] == 'scipy.special.erf'
    assert _scipy_known_functions['factorial'] == 'scipy.special.factorial'

# 测试 NumPyPrinter 类中打印方法的存在性
def test_numpy_print_methods():
    # 断言 NumPyPrinter 实例具有特定的打印方法
    prntr = NumPyPrinter()
    assert hasattr(prntr, '_print_acos')
    assert hasattr(prntr, '_print_log')

# 测试 SciPyPrinter 类中打印方法的存在性，以及对特定函数和常量的打印是否正确
def test_scipy_print_methods():
    # 断言 SciPyPrinter 实例具有特定的打印方法
    prntr = SciPyPrinter()
    assert hasattr(prntr, '_print_acos')
    assert hasattr(prntr, '_print_log')
    assert hasattr(prntr, '_print_erf')
    assert hasattr(prntr, '_print_factorial')
    assert hasattr(prntr, '_print_chebyshevt')

    # 使用符号变量进行打印测试，验证打印输出是否与预期相符
    k = Symbol('k', integer=True, nonnegative=True)
    x = Symbol('x', real=True)
    assert prntr.doprint(polygamma(k, x)) == "scipy.special.polygamma(k, x)"
    assert prntr.doprint(Si(x)) == "scipy.special.sici(x)[0]"
    assert prntr.doprint(Ci(x)) == "scipy.special.sici(x)[1]"
```