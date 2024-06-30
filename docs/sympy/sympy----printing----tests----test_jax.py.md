# `D:\src\scipysrc\sympy\sympy\printing\tests\test_jax.py`

```
from sympy.concrete.summations import Sum  # 导入求和表达式类
from sympy.core.mod import Mod  # 导入模运算类
from sympy.core.relational import (Equality, Unequality)  # 导入关系运算类（相等和不等式）
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数类
from sympy.matrices.expressions.blockmatrix import BlockMatrix  # 导入块矩阵类
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵符号类
from sympy.matrices.expressions.special import Identity  # 导入单位矩阵类
from sympy.utilities.lambdify import lambdify  # 导入符号表达式到数值函数的转换工具

from sympy.abc import x, i, j, a, b, c, d  # 导入符号变量
from sympy.core import Function, Pow, Symbol  # 导入核心函数、幂函数、符号类
from sympy.codegen.matrix_nodes import MatrixSolve  # 导入矩阵求解节点
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2  # 导入对数加指数和对数加指数2函数
from sympy.codegen.cfunctions import log1p, expm1, hypot, log10, exp2, log2, Sqrt  # 导入数学函数
from sympy.tensor.array import Array  # 导入数组类
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
    PermuteDims, ArrayDiagonal  # 导入数组表达式类（张量积、加法、维度置换、对角线数组）
from sympy.printing.numpy import JaxPrinter, _jax_known_constants, _jax_known_functions  # 导入用于打印成jax.numpy语法的打印机和相关常量、函数
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array  # 导入从矩阵到数组的转换函数

from sympy.testing.pytest import skip, raises  # 导入测试工具（跳过和异常）
from sympy.external import import_module  # 导入外部模块导入函数

# 与NumPy不同，JAX始终使用单精度而不会将操作数提升为双精度。
# 可以在调用`import jax`之前配置JAX的双精度，但这必须是显式配置的，
# 不是完全支持的。因此，这里的测试已经从test_numpy.py中修改，
# 仅在将lambdify函数的精度断言为单精度精度。
# 参考链接：https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

jax = import_module('jax')  # 导入JAX模块

if jax:
    deafult_float_info = jax.numpy.finfo(jax.numpy.array([]).dtype)  # 获取默认浮点数信息
    JAX_DEFAULT_EPSILON = deafult_float_info.eps  # 获取JAX默认的机器精度

def test_jax_piecewise_regression():
    """
    NumPyPrinter需要将Piecewise()的选择列表打印为列表，以避免与numpy 1.8的兼容性问题。
    在numpy 1.9+中，这不是必需的。详情请见gh-9747和gh-9749。
    """
    printer = JaxPrinter()  # 创建JaxPrinter对象
    p = Piecewise((1, x < 0), (0, True))  # 创建分段函数对象p
    assert printer.doprint(p) == \
        'jax.numpy.select([jax.numpy.less(x, 0),True], [1,0], default=jax.numpy.nan)'  # 断言打印结果符合预期格式
    assert printer.module_imports == {'jax.numpy': {'select', 'less', 'nan'}}  # 断言导入模块信息正确

def test_jax_logaddexp():
    lae = logaddexp(a, b)  # 计算对数加指数函数
    assert JaxPrinter().doprint(lae) == 'jax.numpy.logaddexp(a, b)'  # 断言打印结果正确
    lae2 = logaddexp2(a, b)  # 计算对数加指数2函数
    assert JaxPrinter().doprint(lae2) == 'jax.numpy.logaddexp2(a, b)'  # 断言打印结果正确

def test_jax_sum():
    if not jax:
        skip("JAX未安装，跳过测试")

    s = Sum(x ** i, (i, a, b))  # 创建求和表达式对象s
    f = lambdify((a, b, x), s, 'jax')  # 使用JAX转换成数值函数f

    a_, b_ = 0, 10  # 设置求和范围
    x_ = jax.numpy.linspace(-1, +1, 10)  # 创建等差数列作为x的取值
    assert jax.numpy.allclose(f(a_, b_, x_), sum(x_ ** i_ for i_ in range(a_, b_ + 1)))  # 断言数值函数结果与预期的求和结果相似

    s = Sum(i * x, (i, a, b))  # 创建求和表达式对象s
    f = lambdify((a, b, x), s, 'jax')  # 使用JAX转换成数值函数f

    a_, b_ = 0, 10  # 设置求和范围
    # 使用 JAX 库的函数 linspace 创建一个包含 10 个元素的数组，数组的元素从 -1 到 +1 等间距分布
    x_ = jax.numpy.linspace(-1, +1, 10)
    # 断言检查 f 函数对给定参数 a_, b_, x_ 的输出结果是否与 sum(i_ * x_ for i_ in range(a_, b_ + 1)) 相近
    assert jax.numpy.allclose(f(a_, b_, x_), sum(i_ * x_ for i_ in range(a_, b_ + 1)))
# 定义一个用于测试 JAX 多重求和的函数
def test_jax_multiple_sums():
    # 如果没有安装 JAX，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 定义求和表达式
    s = Sum((x + j) * i, (i, a, b), (j, c, d))
    # 将求和表达式转换为 JAX 可执行的函数
    f = lambdify((a, b, c, d, x), s, 'jax')

    # 定义测试参数
    a_, b_ = 0, 10
    c_, d_ = 11, 21
    x_ = jax.numpy.linspace(-1, +1, 10)
    # 执行测试，验证结果是否与预期的多重求和一致
    assert jax.numpy.allclose(f(a_, b_, c_, d_, x_),
                       sum((x_ + j_) * i_ for i_ in range(a_, b_ + 1) for j_ in range(c_, d_ + 1)))


# 定义一个用于测试 JAX 代码生成 einsum 的函数
def test_jax_codegen_einsum():
    # 如果没有安装 JAX，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 定义符号矩阵 M 和 N
    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)

    # 将矩阵乘法转换为数组表示
    cg = convert_matrix_to_array(M * N)
    # 将数组表示转换为 JAX 可执行的函数
    f = lambdify((M, N), cg, 'jax')

    # 定义测试用的矩阵参数
    ma = jax.numpy.array([[1, 2], [3, 4]])
    mb = jax.numpy.array([[1,-2], [-1, 3]])
    # 执行测试，验证结果是否与预期的矩阵乘法一致
    assert (f(ma, mb) == jax.numpy.matmul(ma, mb)).all()


# 定义一个用于测试 JAX 代码生成中其他操作的函数
def test_jax_codegen_extra():
    # 如果没有安装 JAX，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 定义符号矩阵 M、N、P、Q 和对应的数值矩阵
    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)
    P = MatrixSymbol("P", 2, 2)
    Q = MatrixSymbol("Q", 2, 2)
    ma = jax.numpy.array([[1, 2], [3, 4]])
    mb = jax.numpy.array([[1,-2], [-1, 3]])
    mc = jax.numpy.array([[2, 0], [1, 2]])
    md = jax.numpy.array([[1,-1], [4, 7]])

    # 测试 ArrayTensorProduct 操作
    cg = ArrayTensorProduct(M, N)
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.einsum(ma, [0, 1], mb, [2, 3])).all()

    # 测试 ArrayAdd 操作
    cg = ArrayAdd(M, N)
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == ma+mb).all()

    cg = ArrayAdd(M, N, P)
    f = lambdify((M, N, P), cg, 'jax')
    assert (f(ma, mb, mc) == ma+mb+mc).all()

    cg = ArrayAdd(M, N, P, Q)
    f = lambdify((M, N, P, Q), cg, 'jax')
    assert (f(ma, mb, mc, md) == ma+mb+mc+md).all()

    # 测试 PermuteDims 操作
    cg = PermuteDims(M, [1, 0])
    f = lambdify((M,), cg, 'jax')
    assert (f(ma) == ma.T).all()

    # 测试 ArrayTensorProduct 和 PermuteDims 结合操作
    cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.transpose(jax.numpy.einsum(ma, [0, 1], mb, [2, 3]), (1, 2, 3, 0))).all()

    # 测试 ArrayDiagonal 操作
    cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.diagonal(jax.numpy.einsum(ma, [0, 1], mb, [2, 3]), axis1=1, axis2=2)).all()


# 定义一个用于测试 JAX 关系操作的函数
def test_jax_relational():
    # 如果没有安装 JAX，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 测试相等关系
    e = Equality(x, 1)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [False, True, False])

    # 测试不相等关系
    e = Unequality(x, 1)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, False, True])

    # 测试小于关系
    e = (x < 1)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, False, False])

    # 测试小于等于关系
    e = (x <= 1)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, True, False])

    # 测试大于关系
    e = (x > 1)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [False, False, True])
    # 检查是否 x 大于等于 1 的布尔表达式
    e = (x >= 1)
    
    # 将布尔表达式转换为可以在 JAX 中运行的函数
    f = lambdify((x,), e, 'jax')
    
    # 创建包含 [0, 1, 2] 的 JAX 数组
    x_ = jax.numpy.array([0, 1, 2])
    
    # 断言检查 JAX 数组 f(x_) 是否与预期结果 [False, True, True] 相等
    assert jax.numpy.array_equal(f(x_), [False, True, True])
    
    # 多条件逻辑表达式：检查 x 是否大于等于 1 且小于 2 的布尔表达式
    e = (x >= 1) & (x < 2)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    
    # 断言检查 JAX 数组 f(x_) 是否与预期结果 [False, True, False] 相等
    assert jax.numpy.array_equal(f(x_), [False, True, False])
    
    # 多条件逻辑表达式：检查 x 是否大于等于 1 或小于 2 的布尔表达式
    e = (x >= 1) | (x < 2)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    
    # 断言检查 JAX 数组 f(x_) 是否与预期结果 [True, True, True] 相等
    assert jax.numpy.array_equal(f(x_), [True, True, True])
def test_jax_mod():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 创建模块化表达式 e = Mod(a, b)
    e = Mod(a, b)
    # 使用 JAX 将表达式 e 转换为可调用的函数 f
    f = lambdify((a, b), e, 'jax')

    # 定义测试用的输入数组 a_ 和标量 b_
    a_ = jax.numpy.array([0, 1, 2, 3])
    b_ = 2
    # 断言函数 f 在给定输入下的输出是否与预期相等
    assert jax.numpy.array_equal(f(a_, b_), [0, 1, 0, 1])

    # 重新定义测试用的输入数组 a_ 和数组 b_
    a_ = jax.numpy.array([0, 1, 2, 3])
    b_ = jax.numpy.array([2, 2, 2, 2])
    # 断言函数 f 在给定输入下的输出是否与预期相等
    assert jax.numpy.array_equal(f(a_, b_), [0, 1, 0, 1])

    # 重新定义测试用的输入数组 a_ 和数组 b_
    a_ = jax.numpy.array([2, 3, 4, 5])
    b_ = jax.numpy.array([2, 3, 4, 5])
    # 断言函数 f 在给定输入下的输出是否与预期相等
    assert jax.numpy.array_equal(f(a_, b_), [0, 0, 0, 0])


def test_jax_pow():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip('JAX not installed')

    # 创建指数表达式 expr = Pow(2, -1, evaluate=False)
    expr = Pow(2, -1, evaluate=False)
    # 使用 JAX 将表达式 expr 转换为可调用的函数 f
    f = lambdify([], expr, 'jax')
    # 断言函数 f 在没有输入的情况下的输出是否等于预期值 0.5
    assert f() == 0.5


def test_jax_expm1():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 使用 JAX 将 expm1(a) 转换为可调用的函数 f
    f = lambdify((a,), expm1(a), 'jax')
    # 断言函数 f 在给定输入下的输出是否在指定精度范围内与预期值相等
    assert abs(f(1e-10) - 1e-10 - 5e-21) <= 1e-10 * JAX_DEFAULT_EPSILON


def test_jax_log1p():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 使用 JAX 将 log1p(a) 转换为可调用的函数 f
    f = lambdify((a,), log1p(a), 'jax')
    # 断言函数 f 在给定输入下的输出是否在指定精度范围内与预期值相等
    assert abs(f(1e-99) - 1e-99) <= 1e-99 * JAX_DEFAULT_EPSILON


def test_jax_hypot():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")
    # 断言 hypot(a, b) 函数在给定输入下的输出是否在指定精度范围内与预期值相等
    assert abs(lambdify((a, b), hypot(a, b), 'jax')(3, 4) - 5) <= JAX_DEFAULT_EPSILON


def test_jax_log10():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 断言 log10(a) 函数在给定输入下的输出是否在指定精度范围内与预期值相等
    assert abs(lambdify((a,), log10(a), 'jax')(100) - 2) <= JAX_DEFAULT_EPSILON


def test_jax_exp2():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")
    # 断言 exp2(a) 函数在给定输入下的输出是否在指定精度范围内与预期值相等
    assert abs(lambdify((a,), exp2(a), 'jax')(5) - 32) <= JAX_DEFAULT_EPSILON


def test_jax_log2():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")
    # 断言 log2(a) 函数在给定输入下的输出是否在指定精度范围内与预期值相等
    assert abs(lambdify((a,), log2(a), 'jax')(256) - 8) <= JAX_DEFAULT_EPSILON


def test_jax_Sqrt():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")
    # 断言 Sqrt(a) 函数在给定输入下的输出是否在指定精度范围内与预期值相等
    assert abs(lambdify((a,), Sqrt(a), 'jax')(4) - 2) <= JAX_DEFAULT_EPSILON


def test_jax_sqrt():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")
    # 断言 sqrt(a) 函数在给定输入下的输出是否在指定精度范围内与预期值相等
    assert abs(lambdify((a,), sqrt(a), 'jax')(4) - 2) <= JAX_DEFAULT_EPSILON


def test_jax_matsolve():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 定义符号矩阵 M 和向量 x
    M = MatrixSymbol("M", 3, 3)
    x = MatrixSymbol("x", 3, 1)

    # 创建线性表达式 expr 和使用矩阵求解的表达式 matsolve_expr
    expr = M**(-1) * x + x
    matsolve_expr = MatrixSolve(M, x) + x

    # 使用 JAX 将表达式 expr 和 matsolve_expr 转换为可调用的函数 f 和 f_matsolve
    f = lambdify((M, x), expr, 'jax')
    f_matsolve = lambdify((M, x), matsolve_expr, 'jax')

    # 创建测试用的矩阵 m0 和向量 x0
    m0 = jax.numpy.array([[1, 2, 3], [3, 2, 5], [5, 6, 7]])
    # 断言矩阵 m0 的秩是否为 3
    assert jax.numpy.linalg.matrix_rank(m0) == 3

    x0 = jax.numpy.array([3, 4, 5])

    # 断言两种求解方法 f_matsolve 和 f 在给定输入下的输出是否非常接近
    assert jax.numpy.allclose(f_matsolve(m0, x0), f(m0, x0))


def test_16857():
    # 如果 JAX 没有安装，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 定义多个 MatrixSymbol 变量 a_1, a_2, a_3, a_4 和 BlockMatrix A
    a_1 = MatrixSymbol('a_1', 10, 3)
    a_2 = MatrixSymbol('a_2', 10, 3)
    a_3 = MatrixSymbol('a_3', 10, 3)
    a_4 = MatrixSymbol('a_4', 10, 3)
    A = BlockMatrix([[a_1, a_2], [a_3, a_4]])
    # 断言 BlockMatrix A 的形状是否为 (20, 6)
    assert A.shape == (20, 6)

    # 创建 JaxPrinter 对象 printer
    printer = JaxPrinter()
    # 定义一个使用 JAX 库进行符号计算的函数 f，它接受一个矩阵 M 并返回 M 加上一个 2x2 的单位矩阵的结果
    f = lambdify(M, M + Identity(2), 'jax')
    
    # 创建两个 2x2 的矩阵 ma 和 mr，它们分别是 JAX 数组
    ma = jax.numpy.array([[1, 2], [3, 4]])
    mr = jax.numpy.array([[2, 2], [3, 5]])
    
    # 使用断言验证 f 对于输入 ma 的结果是否等于预期的 mr，这里使用了 JAX 数组的 all 方法来进行全元素比较
    assert (f(ma) == mr).all()
    
    # 导入符号计算库 sympy 的符号类 symbols
    from sympy.core.symbol import symbols
    
    # 定义一个整数符号 n
    n = symbols('n', integer=True)
    
    # 创建一个符号矩阵 N，它的维度是 n x n
    N = MatrixSymbol("M", n, n)
    
    # 使用 lambda 表达式和 JAX 进行符号计算，尝试将 N 加上一个 n x n 的单位矩阵，期望抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: lambdify(N, N + Identity(n), 'jax'))
# 定义测试函数，用于测试 JaxPrinter 类的输出是否符合预期
def test_jax_array():
    # 断言 JaxPrinter 对象打印 Array(((1, 2), (3, 5))) 结果为 'jax.numpy.array([[1, 2], [3, 5]])'
    assert JaxPrinter().doprint(Array(((1, 2), (3, 5)))) == 'jax.numpy.array([[1, 2], [3, 5]])'
    # 断言 JaxPrinter 对象打印 Array((1, 2)) 结果为 'jax.numpy.array((1, 2))'
    assert JaxPrinter().doprint(Array((1, 2))) == 'jax.numpy.array((1, 2))'


# 定义测试函数，用于测试 _jax_known_constants 字典和 _jax_known_functions 字典中的预定义常量和函数
def test_jax_known_funcs_consts():
    # 断言 _jax_known_constants 字典中 'NaN' 对应的值为 'jax.numpy.nan'
    assert _jax_known_constants['NaN'] == 'jax.numpy.nan'
    # 断言 _jax_known_constants 字典中 'EulerGamma' 对应的值为 'jax.numpy.euler_gamma'
    assert _jax_known_constants['EulerGamma'] == 'jax.numpy.euler_gamma'

    # 断言 _jax_known_functions 字典中 'acos' 对应的值为 'jax.numpy.arccos'
    assert _jax_known_functions['acos'] == 'jax.numpy.arccos'
    # 断言 _jax_known_functions 字典中 'log' 对应的值为 'jax.numpy.log'
    assert _jax_known_functions['log'] == 'jax.numpy.log'


# 定义测试函数，用于测试 JaxPrinter 类的属性是否包含指定的打印方法
def test_jax_print_methods():
    # 创建 JaxPrinter 对象
    prntr = JaxPrinter()
    # 断言 prntr 对象包含 '_print_acos' 属性
    assert hasattr(prntr, '_print_acos')
    # 断言 prntr 对象包含 '_print_log' 属性
    assert hasattr(prntr, '_print_log')


# 定义测试函数，用于测试 JaxPrinter 类的 printmethod 属性
def test_jax_printmethod():
    # 创建 JaxPrinter 对象
    printer = JaxPrinter()
    # 断言 printer 对象包含 'printmethod' 属性
    assert hasattr(printer, 'printmethod')
    # 断言 printer 对象的 'printmethod' 属性值为 '_jaxcode'
    assert printer.printmethod == '_jaxcode'


# 定义测试函数，用于测试自定义函数 expm1 的打印输出
def test_jax_custom_print_method():
    # 定义 expm1 类，继承自 Function 类
    class expm1(Function):

        # 定义 _jaxcode 方法，根据 printer 对象打印函数表达式
        def _jaxcode(self, printer):
            x, = self.args
            function = f'expm1({printer._print(x)})'
            return printer._module_format(printer._module + '.' + function)

    # 创建 JaxPrinter 对象
    printer = JaxPrinter()
    # 断言 printer 对象打印 expm1(Symbol('x')) 结果为 'jax.numpy.expm1(x)'
    assert printer.doprint(expm1(Symbol('x'))) == 'jax.numpy.expm1(x)'
```