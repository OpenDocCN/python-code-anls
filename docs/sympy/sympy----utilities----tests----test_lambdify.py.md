# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_lambdify.py`

```
# 导入 itertools 模块中的 product 函数，用于生成迭代器，计算笛卡尔积
from itertools import product
# 导入 math 模块，提供标准数学函数
import math
# 导入 inspect 模块，提供检查函数和类的工具
import inspect

# 导入 mpmath 模块，提供高精度数学运算支持
import mpmath
# 导入 sympy.testing.pytest 模块中的 raises 和 warns_deprecated_sympy 函数，用于测试和警告相关功能
from sympy.testing.pytest import raises, warns_deprecated_sympy
# 导入 sympy.concrete.summations 中的 Sum 类，用于表示和的表达式
from sympy.concrete.summations import Sum
# 导入 sympy.core.function 模块中的 Function、Lambda 和 diff 函数，提供符号函数和导数计算
from sympy.core.function import (Function, Lambda, diff)
# 导入 sympy.core.numbers 模块中的数学常数和函数，如 E、Float、I、Rational 等
from sympy.core.numbers import (E, Float, I, Rational, all_close, oo, pi)
# 导入 sympy.core.relational 模块中的 Eq 类，用于表示符号相等关系
from sympy.core.relational import Eq
# 导入 sympy.core.singleton 模块中的 S 单例对象，用于表示符号表达式
from sympy.core.singleton import S
# 导入 sympy.core.symbol 模块中的 Dummy 和 symbols 函数，用于创建符号变量
from sympy.core.symbol import (Dummy, symbols)
# 导入 sympy.functions.combinatorial.factorials 模块中的阶乘函数，如 RisingFactorial 和 factorial
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
# 导入 sympy.functions.combinatorial.numbers 模块中的数学函数，如 bernoulli 和 harmonic
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
# 导入 sympy.functions.elementary.complexes 模块中的复数函数，如 Abs
from sympy.functions.elementary.complexes import Abs
# 导入 sympy.functions.elementary.exponential 模块中的指数函数，如 exp 和 log
from sympy.functions.elementary.exponential import exp, log
# 导入 sympy.functions.elementary.hyperbolic 模块中的双曲函数，如 acosh
from sympy.functions.elementary.hyperbolic import acosh
# 导入 sympy.functions.elementary.integers 模块中的整数函数，如 floor
from sympy.functions.elementary.integers import floor
# 导入 sympy.functions.elementary.miscellaneous 模块中的杂项函数，如 Max 和 Min
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
# 导入 sympy.functions.elementary.piecewise 模块中的分段函数，如 Piecewise
from sympy.functions.elementary.piecewise import Piecewise
# 导入 sympy.functions.elementary.trigonometric 模块中的三角函数，如 acos、cos 等
from sympy.functions.elementary.trigonometric import (acos, cos, cot, sin,
                                                      sinc, tan)
# 导入 sympy.functions.special.bessel 模块中的贝塞尔函数，如 besseli、besselj 等
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely, jn, yn)
# 导入 sympy.functions.special.beta_functions 模块中的贝塔函数，如 beta、betainc 等
from sympy.functions.special.beta_functions import (beta, betainc, betainc_regularized)
# 导入 sympy.functions.special.delta_functions 模块中的 delta 函数，如 Heaviside
from sympy.functions.special.delta_functions import (Heaviside)
# 导入 sympy.functions.special.error_functions 模块中的误差函数，如 Ei、erf 等
from sympy.functions.special.error_functions import (Ei, erf, erfc, fresnelc, fresnels, Si, Ci)
# 导入 sympy.functions.special.gamma_functions 模块中的 gamma 函数，如 digamma、gamma 等
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, polygamma)
# 导入 sympy.integrals.integrals 模块中的积分表达式，如 Integral
from sympy.integrals.integrals import Integral
# 导入 sympy.logic.boolalg 模块中的布尔代数函数，如 And、Not 等
from sympy.logic.boolalg import (And, false, ITE, Not, Or, true)
# 导入 sympy.matrices.expressions.dotproduct 模块中的点积表达式，如 DotProduct
from sympy.matrices.expressions.dotproduct import DotProduct
# 导入 sympy.simplify.cse_main 模块中的 CSE 函数，用于公共子表达式消除
from sympy.simplify.cse_main import cse
# 导入 sympy.tensor.array 模块中的张量数组相关函数，如 derive_by_array、Array
from sympy.tensor.array import derive_by_array, Array
# 导入 sympy.tensor.indexed 模块中的索引基类，如 IndexedBase
from sympy.tensor.indexed import IndexedBase
# 导入 sympy.utilities.lambdify 模块中的 lambdify 函数，用于将 SymPy 表达式转换为 Python 函数
from sympy.utilities.lambdify import lambdify
# 导入 sympy.utilities.iterables 模块中的 numbered_symbols 函数，用于生成编号的符号
from sympy.utilities.iterables import numbered_symbols
# 导入 sympy.vector 模块中的 CoordSys3D 类，用于处理三维笛卡尔坐标系
from sympy.vector import CoordSys3D
# 导入 sympy.core.expr 模块中的 UnevaluatedExpr 类，用于表示未评估的表达式
from sympy.core.expr import UnevaluatedExpr
# 导入 sympy.codegen.cfunctions 模块中的数学函数，如 expm1、log1p 等
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, log10, hypot
# 导入 sympy.codegen.numpy_nodes 模块中的数学函数，如 logaddexp、logaddexp2 等
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
# 导入 sympy.codegen.scipy_nodes 模块中的数学函数，如 cosm1、powm1 等
from sympy.codegen.scipy_nodes import cosm1, powm1
# 导入 sympy.functions.elementary.complexes 模块中的复数函数，如 re、im、arg 等
from sympy.functions.elementary.complexes import re, im, arg
# 导入 sympy.functions.special.polynomials 模块中的多项式函数，如 legendre、hermite 等
from sympy.functions.special.polynomials import \
    chebyshevt, chebyshevu, legendre, hermite, laguerre, gegenbauer, \
    assoc_legendre, assoc_laguerre, jacobi
# 导入 sympy.matrices 模块中的矩阵类，如 Matrix、MatrixSymbol、SparseMatrix
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix
# 导入 sympy.printing.lambdarepr 模块中的 LambdaPrinter 类，用于打印 Lambda 表达式
from sympy.printing.lambdarepr import LambdaPrinter
# 导入 sympy.printing.numpy 模块中的 NumPyPrinter 类，用于打印 NumPy 数组表达式
from sympy.printing.numpy import NumPyPrinter
# 导入 sympy.utilities.lambdify 模块中的 implemented_function 和 lambdastr 函数，用于函数实现和字符串表示
from sympy.utilities.lambdify import implemented_function, lambdastr
# 导入 sympy.testing.pytest 模块中的 skip 函数，用于跳过测试
from sympy.testing.pytest import skip
# 导入 sympy.utilities.decorator 模块中的 conserve_mpmath_dps 函数，用于保存 mpmath 的精度设置
from sympy.utilities.decorator import conserve
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
numexpr = import_module('numexpr')
tensorflow = import_module('tensorflow')
cupy = import_module('cupy')
jax = import_module('jax')
numba = import_module('numba')

if tensorflow:
    # Tensorflow存在时，设置环境变量以隐藏警告信息
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w, x, y, z = symbols('w,x,y,z')

#================== Test different arguments =======================

# 测试没有参数的情况
def test_no_args():
    f = lambdify([], 1)
    raises(TypeError, lambda: f(-1))
    assert f() == 1

# 测试单个参数的情况
def test_single_arg():
    f = lambdify(x, 2*x)
    assert f(1) == 2

# 测试列表参数的情况
def test_list_args():
    f = lambdify([x, y], x + y)
    assert f(1, 2) == 3

# 测试嵌套参数的情况
def test_nested_args():
    f1 = lambdify([[w, x]], [w, x])
    assert f1([91, 2]) == [91, 2]
    raises(TypeError, lambda: f1(1, 2))

    f2 = lambdify([(w, x), (y, z)], [w, x, y, z])
    assert f2((18, 12), (73, 4)) == [18, 12, 73, 4]
    raises(TypeError, lambda: f2(3, 4))

    f3 = lambdify([w, [[[x]], y], z], [w, x, y, z])
    assert f3(10, [[[52]], 31], 44) == [10, 52, 31, 44]

# 测试字符串参数的情况
def test_str_args():
    f = lambdify('x,y,z', 'z,y,x')
    assert f(3, 2, 1) == (1, 2, 3)
    assert f(1.0, 2.0, 3.0) == (3.0, 2.0, 1.0)
    # 确保传入正确数量的参数
    raises(TypeError, lambda: f(0))

# 测试自定义命名空间的情况（方式一）
def test_own_namespace_1():
    myfunc = lambda x: 1
    f = lambdify(x, sin(x), {"sin": myfunc})
    assert f(0.1) == 1
    assert f(100) == 1

# 测试自定义命名空间的情况（方式二）
def test_own_namespace_2():
    def myfunc(x):
        return 1
    f = lambdify(x, sin(x), {'sin': myfunc})
    assert f(0.1) == 1
    assert f(100) == 1

# 测试自定义模块的情况
def test_own_module():
    f = lambdify(x, sin(x), math)
    assert f(0) == 0.0

    p, q, r = symbols("p q r", real=True)
    ae = abs(exp(p+UnevaluatedExpr(q+r)))
    f = lambdify([p, q, r], [ae, ae], modules=math)
    results = f(1.0, 1e18, -1e18)
    refvals = [math.exp(1.0)]*2
    for res, ref in zip(results, refvals):
        assert abs((res-ref)/ref) < 1e-15

# 测试不良参数的情况
def test_bad_args():
    # 没有传入可变参数时
    raises(TypeError, lambda: lambdify(1))
    # 传入的是向量表达式时也应该报错
    raises(TypeError, lambda: lambdify([1, 2]))

# 测试非符号原子的情况
def test_atoms():
    # 非符号原子不应该从表达式命名空间中提取出来
    f = lambdify(x, pi + x, {"pi": 3.14})
    assert f(0) == 3.14
    f = lambdify(x, I + x, {"I": 1j})
    assert f(1) == 1 + 1j

#================== Test different modules =========================

# 用于测试 sympy lambda 的函数修饰器
@conserve_mpmath_dps
def test_sympy_lambda():
    mpmath.mp.dps = 50
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    f = lambdify(x, sin(x), "sympy")
    assert f(x) == sin(x)
    prec = 1e-15
    assert -prec < f(Rational(1, 5)).evalf() - Float(str(sin02)) < prec
    # arctan 是在 numpy 模块中，不应该在这里可用
    # 下面的 arctan 应该会导致 NameError，这个测试是什么意思？
    # 调用lambda函数，并断言其会引发NameError异常，lambda函数的定义为lambdify(x, arctan(x), "sympy")
    raises(NameError, lambda: lambdify(x, arctan(x), "sympy"))
# 使用装饰器@conserve_mpmath_dps保护测试函数，在测试完成后恢复mpmath的精度设置
@conserve_mpmath_dps
def test_math_lambda():
    # 设置mpmath的精度为50位小数
    mpmath.mp.dps = 50
    # 定义sin(0.2)的精确值
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    # 使用lambdify函数创建一个math模式下的函数f，该函数计算sin(x)
    f = lambdify(x, sin(x), "math")
    # 设定精度范围
    prec = 1e-15
    # 断言f(0.2)与sin(0.2)的差距在预期的精度范围内
    assert -prec < f(0.2) - sin02 < prec
    # 断言调用f(x)会引发TypeError异常，以确保f是一个Python math函数
    raises(TypeError, lambda: f(x))
           # 如果成功，说明它不是Python的math函数


# 使用装饰器@conserve_mpmath_dps保护测试函数，在测试完成后恢复mpmath的精度设置
@conserve_mpmath_dps
def test_mpmath_lambda():
    # 设置mpmath的精度为50位小数
    mpmath.mp.dps = 50
    # 定义sin(0.2)的精确值
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    # 使用lambdify函数创建一个mpmath模式下的函数f，该函数计算sin(x)
    f = lambdify(x, sin(x), "mpmath")
    # 设定精度范围
    prec = 1e-49  # mpmath的精度大约在50位小数左右
    # 断言f(mpmath.mpf("0.2"))与sin(0.2)的差距在预期的精度范围内
    assert -prec < f(mpmath.mpf("0.2")) - sin02 < prec
    # 断言调用f(x)会引发TypeError异常，以确保f是一个mpmath函数
    raises(TypeError, lambda: f(x))
           # 如果成功，说明它不是mpmath的函数

    # 计算复杂表达式ref2的值
    ref2 = (mpmath.mpf("1e-30")
            - mpmath.mpf("1e-45")/2
            + 5*mpmath.mpf("1e-60")/6
            - 3*mpmath.mpf("1e-75")/4
            + 33*mpmath.mpf("1e-90")/40
            )
    # 使用lambdify创建mpmath模式下的函数f2a、f2b和f2c，分别计算不同表达式的值
    f2a = lambdify((x, y), x**y - 1, "mpmath")
    f2b = lambdify((x, y), powm1(x, y), "mpmath")
    f2c = lambdify((x,), expm1(x*log1p(x)), "mpmath")
    # 计算各个表达式的结果
    ans2a = f2a(mpmath.mpf("1")+mpmath.mpf("1e-15"), mpmath.mpf("1e-15"))
    ans2b = f2b(mpmath.mpf("1")+mpmath.mpf("1e-15"), mpmath.mpf("1e-15"))
    ans2c = f2c(mpmath.mpf("1e-15"))
    # 断言计算结果与参考值ref2的差距在预期的精度范围内
    assert abs(ans2a - ref2) < 1e-51
    assert abs(ans2b - ref2) < 1e-67
    assert abs(ans2c - ref2) < 1e-80


# 使用装饰器@conserve_mpmath_dps保护测试函数，在测试完成后恢复mpmath的精度设置
@conserve_mpmath_dps
def test_number_precision():
    # 设置mpmath的精度为50位小数
    mpmath.mp.dps = 50
    # 定义sin(0.2)的精确值
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    # 使用lambdify函数创建一个mpmath模式下的函数f，该函数返回sin02的精确值
    f = lambdify(x, sin02, "mpmath")
    # 设定精度范围
    prec = 1e-49  # mpmath的精度大约在50位小数左右
    # 断言f(0)与sin(0.2)的差距在预期的精度范围内
    assert -prec < f(0) - sin02 < prec


# 使用装饰器@conserve_mpmath_dps保护测试函数，在测试完成后恢复mpmath的精度设置
@conserve_mpmath_dps
def test_mpmath_precision():
    # 设置mpmath的精度为100位小数
    mpmath.mp.dps = 100
    # 断言lambdify函数能够正确计算pi的值，并将结果转换为字符串进行比较
    assert str(lambdify((), pi.evalf(100), 'mpmath')()) == str(pi.evalf(100))

#================== Test Translations ==============================
# 我们只能检查所有翻译函数是否有效。必须手动检查它们是否完整。

# 检查math翻译表中的所有函数是否有效
def test_math_transl():
    from sympy.utilities.lambdify import MATH_TRANSLATIONS
    for sym, mat in MATH_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert mat in math.__dict__

# 检查mpmath翻译表中的所有函数是否有效
def test_mpmath_transl():
    from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
    for sym, mat in MPMATH_TRANSLATIONS.items():
        assert sym in sympy.__dict__ or sym == 'Matrix'
        assert mat in mpmath.__dict__

# 如果numpy未安装，跳过测试
def test_numpy_transl():
    if not numpy:
        skip("numpy not installed.")

    # 检查numpy翻译表中的所有函数是否有效
    from sympy.utilities.lambdify import NUMPY_TRANSLATIONS
    for sym, nump in NUMPY_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert nump in numpy.__dict__

# 如果scipy未安装，跳过测试
def test_scipy_transl():
    if not scipy:
        skip("scipy not installed.")

    # 检查scipy翻译表中的所有函数是否有效
    from sympy.utilities.lambdify import SCIPY_TRANSLATIONS
    # 对于每个 (sym, scip) 对，检查 sympy 模块中是否存在符号 sym
    # 然后检查 scipy 模块中是否存在符号 scip，或者在 scipy.special 模块中存在 scip
    for sym, scip in SCIPY_TRANSLATIONS.items():
        # 确保 sympy 模块中存在符号 sym
        assert sym in sympy.__dict__
        # 确保 scip 要么存在于 scipy 模块中，要么存在于 scipy.special 模块中
        assert scip in scipy.__dict__ or scip in scipy.special.__dict__
# 测试 numpy 的安装情况，如果未安装则跳过该测试
def test_numpy_translation_abs():
    if not numpy:
        skip("numpy not installed.")

    # 使用 lambdify 函数创建一个 numpy 环境下的 lambda 函数，计算绝对值
    f = lambdify(x, Abs(x), "numpy")
    # 断言绝对值函数的结果是否正确
    assert f(-1) == 1
    assert f(1) == 1


# 测试 numexpr 的安装情况，如果未安装则跳过该测试
def test_numexpr_printer():
    if not numexpr:
        skip("numexpr not installed.")

    # 导入 NumExprPrinter 类，用于测试 numexpr 表达式的正确打印和计算
    from sympy.printing.lambdarepr import NumExprPrinter

    # 黑名单中的函数不进行测试
    blacklist = ('where', 'complex', 'contains')
    arg_tuple = (x, y, z) # 一些函数可能需要多个参数
    # 遍历 numexpr 支持的函数列表
    for sym in NumExprPrinter._numexpr_functions.keys():
        if sym in blacklist:
            continue
        ssym = S(sym)
        if hasattr(ssym, '_nargs'):
            nargs = ssym._nargs[0]
        else:
            nargs = 1
        args = arg_tuple[:nargs]
        # 使用 lambdify 创建 numexpr 环境下的 lambda 函数，并进行断言
        f = lambdify(args, ssym(*args), modules='numexpr')
        assert f(*(1, )*nargs) is not None


# 测试 numexpr 和 numpy 的安装情况，如果任一未安装则跳过该测试
def test_issue_9334():
    if not numexpr:
        skip("numexpr not installed.")
    if not numpy:
        skip("numpy not installed.")
    expr = S('b*a - sqrt(a**2)')
    a, b = sorted(expr.free_symbols, key=lambda s: s.name)
    # 使用 lambdify 创建 numexpr 环境下的 lambda 函数，计算结果
    func_numexpr = lambdify((a,b), expr, modules=[numexpr], dummify=False)
    foo, bar = numpy.random.random((2, 4))
    func_numexpr(foo, bar)


# 测试 numexpr 的安装情况，如果未安装则跳过该测试
def test_issue_12984():
    if not numexpr:
        skip("numexpr not installed.")
    # 使用 lambdify 创建 numexpr 环境下的 lambda 函数，计算 Piecewise 表达式
    func_numexpr = lambdify((x,y,z), Piecewise((y, x >= 0), (z, x > -1)), numexpr)
    with ignore_warnings(RuntimeWarning):
        # 断言函数计算结果是否正确
        assert func_numexpr(1, 24, 42) == 24
        assert str(func_numexpr(-1, 24, 42)) == 'nan'


# 测试不使用任何模块的 lambdify 函数的行为
def test_empty_modules():
    x, y = symbols('x y')
    expr = -(x % y)

    # 创建不使用任何模块的 lambdify 函数，计算结果
    no_modules = lambdify([x, y], expr)
    empty_modules = lambdify([x, y], expr, modules=[])
    # 断言两种方式计算的结果是否一致
    assert no_modules(3, 7) == empty_modules(3, 7)
    assert no_modules(3, 7) == -3


# 测试指数函数的 lambdify 创建及计算
def test_exponentiation():
    # 使用 lambdify 创建指数函数的 lambda 函数
    f = lambdify(x, x**2)
    # 断言函数计算结果是否正确
    assert f(-1) == 1
    assert f(0) == 0
    assert f(1) == 1
    assert f(-2) == 4
    assert f(2) == 4
    assert f(2.5) == 6.25


# 测试平方根函数的 lambdify 创建及计算
def test_sqrt():
    # 使用 lambdify 创建平方根函数的 lambda 函数
    f = lambdify(x, sqrt(x))
    # 断言函数计算结果是否正确
    assert f(0) == 0.0
    assert f(1) == 1.0
    assert f(4) == 2.0
    assert abs(f(2) - 1.414) < 0.001
    assert f(6.25) == 2.5


# 测试三角函数的 lambdify 创建及计算
def test_trig():
    # 使用 lambdify 创建 math 模块下的 cos 和 sin 函数的 lambda 函数
    f = lambdify([x], [cos(x), sin(x)], 'math')
    d = f(pi)
    prec = 1e-11
    # 断言计算结果的精度是否满足要求
    assert -prec < d[0] + 1 < prec
    assert -prec < d[1] < prec
    d = f(3.14159)
    prec = 1e-5
    assert -prec < d[0] + 1 < prec
    assert -prec < d[1] < prec


# 测试积分函数的 lambdify 创建及计算
def test_integral():
    if numpy and not scipy:
        skip("scipy not installed.")
    # 创建指数函数的 Lambda 表达式
    f = Lambda(x, exp(-x**2))
    # 使用 lambdify 创建积分函数的 lambda 函数
    l = lambdify(y, Integral(f(x), (x, y, oo)))
    d = l(-oo)
    # 断言积分计算结果是否在指定范围内
    assert 1.77245385 < d < 1.772453851


# 测试双重积分的 lambdify 创建及计算
def test_double_integral():
    if numpy and not scipy:
        skip("scipy not installed.")
    # 创建双重积分的 Integral 对象
    i = Integral(1/(1 - x**2*y**2), (x, 0, 1), (y, 0, z))
    # 使用 lambdify 创建双重积分的 lambda 函数
    l = lambdify([z], i)
    d = l(1)
    # 断言语句，用于检查变量 d 的值是否在指定范围内
    assert 1.23370055 < d < 1.233700551
def test_spherical_bessel():
    # 检查是否安装了 numpy，但未安装 scipy，若未安装则跳过测试
    if numpy and not scipy:
        skip("scipy not installed.")
    # 随机选取测试点
    test_point = 4.2
    # 定义符号变量 x
    x = symbols("x")
    # 计算第二类球贝塞尔函数 J_2(x)
    jtest = jn(2, x)
    # 断言数值结果与符号表达式数值近似相等
    assert abs(lambdify(x,jtest)(test_point) -
            jtest.subs(x,test_point).evalf()) < 1e-8
    # 计算第二类修正球贝塞尔函数 Y_2(x)
    ytest = yn(2, x)
    # 断言数值结果与符号表达式数值近似相等
    assert abs(lambdify(x,ytest)(test_point) -
            ytest.subs(x,test_point).evalf()) < 1e-8


#================== Test vectors ===================================


def test_vector_simple():
    # 定义向量函数，交换输入的三个参数的顺序
    f = lambdify((x, y, z), (z, y, x))
    # 断言函数在输入 (3, 2, 1) 时返回 (1, 2, 3)
    assert f(3, 2, 1) == (1, 2, 3)
    # 断言函数在输入浮点数时正确返回
    assert f(1.0, 2.0, 3.0) == (3.0, 2.0, 1.0)
    # 确保函数要求正确的参数数量，检查是否会引发 TypeError
    raises(TypeError, lambda: f(0))


def test_vector_discontinuous():
    # 定义向量函数，处理不连续点
    f = lambdify(x, (-1/x, 1/x))
    # 确保在除以零时会引发 ZeroDivisionError
    raises(ZeroDivisionError, lambda: f(0))
    # 断言在特定输入下函数返回预期的值
    assert f(1) == (-1.0, 1.0)
    assert f(2) == (-0.5, 0.5)
    assert f(-2) == (0.5, -0.5)


def test_trig_symbolic():
    # 定义三角函数的符号化函数
    f = lambdify([x], [cos(x), sin(x)], 'math')
    # 计算结果并断言与预期的近似相等
    d = f(pi)
    assert abs(d[0] + 1) < 0.0001
    assert abs(d[1] - 0) < 0.0001


def test_trig_float():
    # 定义三角函数的浮点数函数
    f = lambdify([x], [cos(x), sin(x)])
    # 计算结果并断言与预期的近似相等
    d = f(3.14159)
    assert abs(d[0] + 1) < 0.0001
    assert abs(d[1] - 0) < 0.0001


def test_docs():
    # 测试文档中的示例
    f = lambdify(x, x**2)
    assert f(2) == 4
    f = lambdify([x, y, z], [z, y, x])
    assert f(1, 2, 3) == [3, 2, 1]
    f = lambdify(x, sqrt(x))
    assert f(4) == 2.0
    f = lambdify((x, y), sin(x*y)**2)
    assert f(0, 5) == 0


def test_math():
    # 测试数学模块的使用
    f = lambdify((x, y), sin(x), modules="math")
    assert f(0, 5) == 0


def test_sin():
    # 测试正弦函数的平方
    f = lambdify(x, sin(x)**2)
    assert isinstance(f(2), float)
    f = lambdify(x, sin(x)**2, modules="math")
    assert isinstance(f(2), float)


def test_matrix():
    # 测试矩阵运算
    A = Matrix([[x, x*y], [sin(z) + 4, x**z]])
    sol = Matrix([[1, 2], [sin(3) + 4, 1]])
    # 使用 sympy 模块进行符号化函数的定义
    f = lambdify((x, y, z), A, modules="sympy")
    assert f(1, 2, 3) == sol
    f = lambdify((x, y, z), (A, [A]), modules="sympy")
    assert f(1, 2, 3) == (sol, [sol])
    # 计算雅可比矩阵，并验证结果
    J = Matrix((x, x + y)).jacobian((x, y))
    v = Matrix((x, y))
    sol = Matrix([[1, 0], [1, 1]])
    assert lambdify(v, J, modules='sympy')(1, 2) == sol
    assert lambdify(v.T, J, modules='sympy')(1, 2) == sol


def test_numpy_matrix():
    # 检查是否安装了 numpy，若未安装则跳过测试
    if not numpy:
        skip("numpy not installed.")
    A = Matrix([[x, x*y], [sin(z) + 4, x**z]])
    sol_arr = numpy.array([[1, 2], [numpy.sin(3) + 4, 1]])
    # 使用 numpy 模块进行矩阵处理
    f = lambdify((x, y, z), A, ['numpy'])
    numpy.testing.assert_allclose(f(1, 2, 3), sol_arr)
    # 确保返回的类型为 numpy 数组
    assert isinstance(f(1, 2, 3), numpy.ndarray)

    # gh-15071
    # 定义一个自定义函数类 dot
    class dot(Function):
        pass
    # 创建 dot 类的实例，与矩阵进行乘法运算
    x_dot_mtx = dot(x, Matrix([[2], [1], [0]]))
    f_dot1 = lambdify(x, x_dot_mtx)
    # 创建一个形状为 (17, 3) 的全零 numpy 数组
    inp = numpy.zeros((17, 3))
    # 断言 dot 类的实例在输入数组 inp 时返回全零结果
    assert numpy.all(f_dot1(inp) == 0)

    # 定义严格关键字参数
    strict_kw = {"allow_unknown_functions": False, "inline": True, "fully_qualified_modules": False}
    # 创建一个 NumPyPrinter 对象 p2，配置用户函数为 'dot'，同时传入 strict_kw 参数的字典
    p2 = NumPyPrinter(dict(user_functions={'dot': 'dot'}, **strict_kw))
    
    # 使用 p2 作为打印器，将 x_dot_mtx 转换为一个 lambdify 函数 f_dot2
    f_dot2 = lambdify(x, x_dot_mtx, printer=p2)
    
    # 断言：确保 f_dot2 对于输入 inp 的结果全为零
    assert numpy.all(f_dot2(inp) == 0)

    # 创建一个 NumPyPrinter 对象 p3，仅使用 strict_kw 参数的字典
    p3 = NumPyPrinter(strict_kw)
    
    # 断言：预期在构造 lambda 函数时抛出异常，这里传入 x_dot_mtx 和输入 inp
    raises(Exception, lambda: lambdify(x, x_dot_mtx, printer=p3)(inp))
def test_numpy_transpose():
    # 检查是否安装了 numpy 库，若未安装则跳过测试
    if not numpy:
        skip("numpy not installed.")
    # 创建一个 2x2 的矩阵 A，其中元素是表达式 [1, x]
    A = Matrix([[1, x], [0, 1]])
    # 将矩阵 A 转置为数组表达式，使用 numpy 模块进行求值
    f = lambdify((x), A.T, modules="numpy")
    # 断言计算结果与预期数组相等
    numpy.testing.assert_array_equal(f(2), numpy.array([[1, 0], [2, 1]]))


def test_numpy_dotproduct():
    # 检查是否安装了 numpy 库，若未安装则跳过测试
    if not numpy:
        skip("numpy not installed")
    # 创建一个包含符号 x, y, z 的矩阵 A
    A = Matrix([x, y, z])
    # 创建函数 f1, f2, f3, f4，分别表示 A 与自身点积、A 与转置的点积等，使用 numpy 模块进行求值
    f1 = lambdify([x, y, z], DotProduct(A, A), modules='numpy')
    f2 = lambdify([x, y, z], DotProduct(A, A.T), modules='numpy')
    f3 = lambdify([x, y, z], DotProduct(A.T, A), modules='numpy')
    f4 = lambdify([x, y, z], DotProduct(A, A.T), modules='numpy')

    # 断言所有函数计算结果均等于预期的 numpy 数组 [14]
    assert f1(1, 2, 3) == \
           f2(1, 2, 3) == \
           f3(1, 2, 3) == \
           f4(1, 2, 3) == \
           numpy.array([14])


def test_numpy_inverse():
    # 检查是否安装了 numpy 库，若未安装则跳过测试
    if not numpy:
        skip("numpy not installed.")
    # 创建一个 2x2 的矩阵 A，其中元素是表达式 [[1, x], [0, 1]]
    A = Matrix([[1, x], [0, 1]])
    # 创建函数 f，表示 A 的逆矩阵，使用 numpy 模块进行求值
    f = lambdify((x), A**-1, modules="numpy")
    # 断言计算结果与预期数组相等
    numpy.testing.assert_array_equal(f(2), numpy.array([[1, -2], [0,  1]]))


def test_numpy_old_matrix():
    # 检查是否安装了 numpy 库，若未安装则跳过测试
    if not numpy:
        skip("numpy not installed.")
    # 创建一个 2x2 的矩阵 A，其中元素是包含符号 x, y, z 的复杂表达式
    A = Matrix([[x, x*y], [sin(z) + 4, x**z]])
    # 创建一个预期的 numpy 数组 sol_arr
    sol_arr = numpy.array([[1, 2], [numpy.sin(3) + 4, 1]])
    # 使用 lambdify 函数生成函数 f，将符号 x, y, z 替换为 numpy.matrix 类型，并使用 numpy 模块
    f = lambdify((x, y, z), A, [{'ImmutableDenseMatrix': numpy.matrix}, 'numpy'])
    # 忽略即将过时的警告，使用 numpy.testing.assert_allclose 断言计算结果与预期数组 sol_arr 很接近
    with ignore_warnings(PendingDeprecationWarning):
        numpy.testing.assert_allclose(f(1, 2, 3), sol_arr)
        # 断言函数 f 返回的结果类型是 numpy.matrix
        assert isinstance(f(1, 2, 3), numpy.matrix)


def test_scipy_sparse_matrix():
    # 检查是否安装了 scipy 库，若未安装则跳过测试
    if not scipy:
        skip("scipy not installed.")
    # 创建一个稀疏矩阵 A，其中元素是表达式 [[x, 0], [0, y]]
    A = SparseMatrix([[x, 0], [0, y]])
    # 使用 lambdify 函数生成函数 f，将符号 x, y 替换为 scipy 模块
    f = lambdify((x, y), A, modules="scipy")
    # 计算 f 的结果 B，并断言其类型为 scipy.sparse.coo_matrix
    B = f(1, 2)
    assert isinstance(B, scipy.sparse.coo_matrix)


def test_python_div_zero_issue_11306():
    # 检查是否安装了 numpy 库，若未安装则跳过测试
    if not numpy:
        skip("numpy not installed.")
    # 创建一个分段函数 p，根据不同条件返回不同表达式，其中包含符号 x, y
    p = Piecewise((1 / x, y < -1), (x, y < 1), (1 / x, True))
    # 使用 lambdify 函数生成函数 f，将符号 x, y 替换为 numpy 模块，并处理除以零的警告
    f = lambdify([x, y], p, modules='numpy')
    # 在忽略除以零警告的环境下，断言计算结果为预期值 0
    with numpy.errstate(divide='ignore'):
        assert float(f(numpy.array(0), numpy.array(0.5))) == 0
        assert float(f(numpy.array(0), numpy.array(1))) == float('inf')


def test_issue9474():
    # 定义一组可能的模块列表 mods
    mods = [None, 'math']
    # 若安装了 numpy，则将 'numpy' 加入 mods 列表中
    if numpy:
        mods.append('numpy')
    # 若安装了 mpmath，则将 'mpmath' 加入 mods 列表中
    if mpmath:
        mods.append('mpmath')
    # 遍历 mods 列表中的每个模块，生成函数 f，将符号 x 替换为 S.One/x，并使用指定的模块进行求值
    for mod in mods:
        f = lambdify(x, S.One/x, modules=mod)
        # 断言函数 f 计算结果等于预期值 0.5
        assert f(2) == 0.5
        # 创建函数 f，将符号 x 替换为 floor(S.One/x)，并使用指定的模块进行求值
        f = lambdify(x, floor(S.One/x), modules=mod)
        # 断言函数 f 计算结果等于预期值 0
        assert f(2) == 0

    # 对 mods 列表中的每个模块和绝对值函数进行组合
    for absfunc, modules in product([Abs, abs], mods):
        # 创建函数 f，将符号 x 替换为 absfunc(x)，并使用指定的模块进行求值
        f = lambdify(x, absfunc(x), modules=modules)
        # 断言函数 f 计算结果等于预期值 1 或 5
        assert f(-1) == 1
        assert f(1) == 1
        assert f(3+4j) == 5


def test_issue_9871():
    # 检查是否安装了 numexpr 库，若未安装则跳过测试
    if not numexpr:
        skip("numexpr not installed.")
    # 检查是否安装了 numpy 库，若未安装则跳过测试
    if not numpy:
        skip("numpy not installed.")

    # 创建表达式 r，表示 sqrt(x^2 + y^2)
    r = sqrt(x**2 + y**2)
    # 对 r 求关于 x 的偏导数，生成表达式 expr
    expr = diff(1/r, x)

    # 创建线性空间向量 xn 和 yn
    xn = yn = numpy.linspace(1, 10, 16)
    # 创建预期的 numpy 数组 fv_exact
    fv_exact = -numpy.sqrt(2.)**-3 * xn**-2

    # 使用 lambdify 函数生成函数 fv_numpy，将符号 x, y 替换为 numpy 模块，并计算结果
    fv_numpy = lamb
    # 使用 NumPy 测试工具检查 fv_numpy 和 fv_exact 是否在给定的相对容差范围内全部接近
    numpy.testing.assert_allclose(fv_numpy, fv_exact, rtol=1e-10)
    # 使用 NumPy 测试工具检查 fv_numexpr 和 fv_exact 是否在给定的相对容差范围内全部接近
    numpy.testing.assert_allclose(fv_numexpr, fv_exact, rtol=1e-10)
# 测试使用 numpy 的 piecewise 函数
def test_numpy_piecewise():
    # 检查 numpy 是否可用，如果不可用则跳过测试
    if not numpy:
        skip("numpy not installed.")
    
    # 定义 Piecewise 函数，根据条件返回不同的表达式
    pieces = Piecewise((x, x < 3), (x**2, x > 5), (0, True))
    
    # 将 Piecewise 函数转换为可调用的 lambda 函数 f
    f = lambdify(x, pieces, modules="numpy")
    
    # 断言调用 f 后返回的 numpy 数组与预期结果相等
    numpy.testing.assert_array_equal(f(numpy.arange(10)),
                                     numpy.array([0, 1, 2, 0, 0, 0, 36, 49, 64, 81]))
    
    # 如果所有条件都不满足，期望返回 NaN
    nodef_func = lambdify(x, Piecewise((x, x > 0), (-x, x < 0)))
    
    # 断言调用 nodef_func 后返回的 numpy 数组与预期结果相等
    numpy.testing.assert_array_equal(nodef_func(numpy.array([-1, 0, 1])),
                                     numpy.array([1, numpy.nan, 1]))


# 测试使用 numpy 的逻辑运算函数
def test_numpy_logical_ops():
    # 检查 numpy 是否可用，如果不可用则跳过测试
    if not numpy:
        skip("numpy not installed.")
    
    # 定义逻辑与函数 and_func，并将其转换为可调用的 lambda 函数
    and_func = lambdify((x, y), And(x, y), modules="numpy")
    and_func_3 = lambdify((x, y, z), And(x, y, z), modules="numpy")
    
    # 定义逻辑或函数 or_func，并将其转换为可调用的 lambda 函数
    or_func = lambdify((x, y), Or(x, y), modules="numpy")
    or_func_3 = lambdify((x, y, z), Or(x, y, z), modules="numpy")
    
    # 定义逻辑非函数 not_func，并将其转换为可调用的 lambda 函数
    not_func = lambdify((x), Not(x), modules="numpy")
    
    # 定义 numpy 数组 arr1, arr2, arr3
    arr1 = numpy.array([True, True])
    arr2 = numpy.array([False, True])
    arr3 = numpy.array([True, False])
    
    # 断言调用逻辑与函数后返回的 numpy 数组与预期结果相等
    numpy.testing.assert_array_equal(and_func(arr1, arr2), numpy.array([False, True]))
    numpy.testing.assert_array_equal(and_func_3(arr1, arr2, arr3), numpy.array([False, False]))
    
    # 断言调用逻辑或函数后返回的 numpy 数组与预期结果相等
    numpy.testing.assert_array_equal(or_func(arr1, arr2), numpy.array([True, True]))
    numpy.testing.assert_array_equal(or_func_3(arr1, arr2, arr3), numpy.array([True, True]))
    
    # 断言调用逻辑非函数后返回的 numpy 数组与预期结果相等
    numpy.testing.assert_array_equal(not_func(arr2), numpy.array([True, False]))


# 测试使用 numpy 的矩阵乘法
def test_numpy_matmul():
    # 检查 numpy 是否可用，如果不可用则跳过测试
    if not numpy:
        skip("numpy not installed.")
    
    # 定义 xmat 和 ymat 两个矩阵对象
    xmat = Matrix([[x, y], [z, 1+z]])
    ymat = Matrix([[x**2], [Abs(x)]])
    
    # 定义矩阵乘法函数 mat_func，并将其转换为可调用的 lambda 函数
    mat_func = lambdify((x, y, z), xmat*ymat, modules="numpy")
    
    # 断言调用 mat_func 后返回的 numpy 数组与预期结果相等
    numpy.testing.assert_array_equal(mat_func(0.5, 3, 4), numpy.array([[1.625], [3.5]]))
    numpy.testing.assert_array_equal(mat_func(-0.5, 3, 4), numpy.array([[1.375], [3.5]]))
    
    # 多个矩阵连续乘法的测试
    f = lambdify((x, y, z), xmat*xmat*xmat, modules="numpy")
    numpy.testing.assert_array_equal(f(0.5, 3, 4), numpy.array([[72.125, 119.25],
                                                                [159, 251]]))


# 测试使用 numpy 和 numexpr 的复杂数值表达式
def test_numpy_numexpr():
    # 检查 numpy 是否可用，如果不可用则跳过测试
    if not numpy:
        skip("numpy not installed.")
    
    # 检查 numexpr 是否可用，如果不可用则跳过测试
    if not numexpr:
        skip("numexpr not installed.")
    
    # 使用 numpy 生成随机数组 a, b, c
    a, b, c = numpy.random.randn(3, 128, 128)
    
    # 定义复杂数值表达式 expr
    expr = sin(x) + cos(y) + tan(z)**2 + Abs(z-y)*acos(sin(y*z)) + \
           Abs(y-z)*acosh(2+exp(y-x))- sqrt(x**2+I*y**2)
    
    # 将 expr 转换为可调用的 lambda 函数，使用 numpy 模块
    npfunc = lambdify((x, y, z), expr, modules='numpy')
    
    # 将 expr 转换为可调用的 lambda 函数，使用 numexpr 模块
    nefunc = lambdify((x, y, z), expr, modules='numexpr')
    
    # 断言调用 npfunc 和 nefunc 后返回的 numpy 数组近似相等
    assert numpy.allclose(npfunc(a, b, c), nefunc(a, b, c))


# 测试使用 numpy 和 numexpr 自定义函数
def test_numexpr_userfunctions():
    # 检查 numpy 是否可用，如果不可用则跳过测试
    if not numpy:
        skip("numpy not installed.")
    
    # 检查 numexpr 是否可用，如果不可用则跳过测试
    if not numexpr:
        skip("numexpr not installed.")
    # 从 numpy 模块中生成两个形状为 (10,) 的随机标准正态分布数组 a 和 b
    a, b = numpy.random.randn(2, 10)
    # 创建一个类 uf，继承自 Function 类，其中包含一个静态方法 eval，计算 y 的平方加 1
    uf = type('uf', (Function, ),
              {'eval' : classmethod(lambda x, y : y**2+1)})
    # 使用 lambdify 将表达式 1-uf(x) 转换为一个可计算函数 func，使用 numexpr 作为计算模块
    func = lambdify(x, 1-uf(x), modules='numexpr')
    # 断言 func(a) 中的所有元素近似等于 -(a**2)
    assert numpy.allclose(func(a), -(a**2))
    
    # 创建一个由 Function 类实现的函数 uf，其表达式为 2*x*y+1
    uf = implemented_function(Function('uf'), lambda x, y : 2*x*y+1)
    # 使用 lambdify 将表达式 uf(x, y) 转换为一个可计算函数 func，使用 numexpr 作为计算模块
    func = lambdify((x, y), uf(x, y), modules='numexpr')
    # 断言 func(a, b) 中的所有元素近似等于 2*a*b+1
    assert numpy.allclose(func(a, b), 2*a*b+1)
def test_tensorflow_basic_math():
    # 检查是否导入了 tensorflow 模块，若未导入则跳过测试
    if not tensorflow:
        skip("tensorflow not installed.")
    # 定义数学表达式，包括 sin(x) 和 Abs(1/(x+2)) 的最大值
    expr = Max(sin(x), Abs(1/(x+2)))
    # 使用 tensorflow 模块将表达式转换为可计算的函数
    func = lambdify(x, expr, modules="tensorflow")

    # 使用 TensorFlow 的会话（Session）执行以下代码块
    with tensorflow.compat.v1.Session() as s:
        # 创建一个常量张量 a，值为 0，数据类型为 tensorflow.float32
        a = tensorflow.constant(0, dtype=tensorflow.float32)
        # 断言计算函数 func 在会话 s 中对输入 a=0 的结果是否等于 0.5
        assert func(a).eval(session=s) == 0.5


def test_tensorflow_placeholders():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Max(sin(x), Abs(1/(x+2)))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        # 创建一个占位符（placeholder） a，数据类型为 tensorflow.float32
        a = tensorflow.compat.v1.placeholder(dtype=tensorflow.float32)
        # 断言计算函数 func 在会话 s 中通过 feed_dict 提供 a=0 的结果是否等于 0.5
        assert func(a).eval(session=s, feed_dict={a: 0}) == 0.5


def test_tensorflow_variables():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Max(sin(x), Abs(1/(x+2)))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        # 创建一个变量 a，初始值为 0，数据类型为 tensorflow.float32
        a = tensorflow.Variable(0, dtype=tensorflow.float32)
        # 初始化变量 a
        s.run(a.initializer)
        # 断言计算函数 func 在会话 s 中通过 feed_dict 提供 a=0 的结果是否等于 0.5
        assert func(a).eval(session=s, feed_dict={a: 0}) == 0.5


def test_tensorflow_logical_operations():
    if not tensorflow:
        skip("tensorflow not installed.")
    # 定义逻辑表达式，使用 Not、And 和 Or 进行逻辑运算
    expr = Not(And(Or(x, y), y))
    func = lambdify([x, y], expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        # 断言计算函数 func 在会话 s 中对输入 False, True 的结果是否等于 False
        assert func(False, True).eval(session=s) == False


def test_tensorflow_piecewise():
    if not tensorflow:
        skip("tensorflow not installed.")
    # 定义分段函数表达式，根据 x 的值返回不同的结果
    expr = Piecewise((0, Eq(x,0)), (-1, x < 0), (1, x > 0))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        # 断言计算函数 func 在会话 s 中对输入 -1, 0, 1 的结果分别是否等于 -1, 0, 1
        assert func(-1).eval(session=s) == -1
        assert func(0).eval(session=s) == 0
        assert func(1).eval(session=s) == 1


def test_tensorflow_multi_max():
    if not tensorflow:
        skip("tensorflow not installed.")
    # 定义多个数的最大值表达式
    expr = Max(x, -x, x**2)
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        # 断言计算函数 func 在会话 s 中对输入 -2 的结果是否等于 4
        assert func(-2).eval(session=s) == 4


def test_tensorflow_multi_min():
    if not tensorflow:
        skip("tensorflow not installed.")
    # 定义多个数的最小值表达式
    expr = Min(x, -x, x**2)
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        # 断言计算函数 func 在会话 s 中对输入 -2 的结果是否等于 -2
        assert func(-2).eval(session=s) == -2


def test_tensorflow_relational():
    if not tensorflow:
        skip("tensorflow not installed.")
    # 定义关系表达式，判断 x 是否大于等于 0
    expr = x >= 0
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        # 断言计算函数 func 在会话 s 中对输入 1 的结果是否等于 True
        assert func(1).eval(session=s) == True


def test_tensorflow_complexes():
    if not tensorflow:
        skip("tensorflow not installed")

    # 使用 tensorflow 模块处理复数函数
    func1 = lambdify(x, re(x), modules="tensorflow")
    func2 = lambdify(x, im(x), modules="tensorflow")
    func3 = lambdify(x, Abs(x), modules="tensorflow")
    func4 = lambdify(x, arg(x), modules="tensorflow")
    # 使用 TensorFlow 兼容性模块创建会话对象，并使用“s”作为别名
    with tensorflow.compat.v1.Session() as s:
        # 对于 TensorFlow 版本在 https://github.com/tensorflow/tensorflow/issues/30029 修复之前，
        # 使用 Python 数字类型可能无法正常工作
        # 创建一个复数常量张量
        a = tensorflow.constant(1+2j)
        
        # 使用会话对象“s”执行 func1 函数，并断言其结果为 1
        assert func1(a).eval(session=s) == 1
        # 使用会话对象“s”执行 func2 函数，并断言其结果为 2
        assert func2(a).eval(session=s) == 2

        # 使用会话对象“s”执行 func3 函数，并将结果存储在 tensorflow_result 中
        tensorflow_result = func3(a).eval(session=s)
        # 使用 sympy 计算绝对值表达式的数值结果
        sympy_result = Abs(1 + 2j).evalf()
        # 断言 TensorFlow 结果与 sympy 结果的差异小于 10**-6
        assert abs(tensorflow_result - sympy_result) < 10**-6

        # 使用会话对象“s”执行 func4 函数，并将结果存储在 tensorflow_result 中
        tensorflow_result = func4(a).eval(session=s)
        # 使用 sympy 计算幅角表达式的数值结果
        sympy_result = arg(1 + 2j).evalf()
        # 断言 TensorFlow 结果与 sympy 结果的差异小于 10**-6
        assert abs(tensorflow_result - sympy_result) < 10**-6
# 定义用于测试 TensorFlow 的函数参数的单元测试函数
def test_tensorflow_array_arg():
    # 测试是否安装了 TensorFlow，若未安装则跳过测试
    if not tensorflow:
        skip("tensorflow not installed.")

    # 使用 lambdify 函数创建一个 TensorFlow 可调用的函数 f，计算 x*x + y
    f = lambdify([[x, y]], x*x + y, 'tensorflow')

    # 使用 TensorFlow 的会话（Session）执行函数 f 的计算
    with tensorflow.compat.v1.Session() as s:
        # 对 f 进行函数调用，传入常量数组 [2.0, 1.0]
        fcall = f(tensorflow.constant([2.0, 1.0]))
        # 断言函数调用的结果是否等于 5.0
        assert fcall.eval(session=s) == 5.0


#================== Test symbolic ==================================


# 定义测试单个符号参数的 lambdify 函数
def test_sym_single_arg():
    # 使用 lambdify 创建一个函数 f，计算 x * y
    f = lambdify(x, x * y)
    # 断言 f 在给定 z 的情况下的值是否等于 z * y
    assert f(z) == z * y


# 定义测试列表符号参数的 lambdify 函数
def test_sym_list_args():
    # 使用 lambdify 创建一个函数 f，计算 x + y + z
    f = lambdify([x, y], x + y + z)
    # 断言 f 在给定参数 1 和 2 的情况下的值是否等于 3 + z
    assert f(1, 2) == 3 + z


# 定义测试符号积分的 lambdify 函数
def test_sym_integral():
    # 使用 Lambda 和 lambdify 创建一个函数 l，计算积分 exp(-x**2)
    f = Lambda(x, exp(-x**2))
    l = lambdify(x, Integral(f(x), (x, -oo, oo)), modules="sympy")
    # 断言函数 l 在给定参数 y 的情况下的值是否等于积分表达式 Integral(exp(-y**2), (y, -oo, oo))
    assert l(y) == Integral(exp(-y**2), (y, -oo, oo))
    # 断言 l(y) 的求值结果是否等于 sqrt(pi)
    assert l(y).doit() == sqrt(pi)


# 定义测试命名空间顺序的 lambdify 函数
def test_namespace_order():
    # 定义两个命名空间字典 n1 和 n2，分别包含函数 f 和 g
    n1 = {'f': lambda x: 'first f'}
    n2 = {'f': lambda x: 'second f',
          'g': lambda x: 'function g'}
    # 创建符号函数 f 和 g
    f = sympy.Function('f')
    g = sympy.Function('g')
    # 使用 lambdify 分别基于不同的命名空间创建函数 if1 和 if2
    if1 = lambdify(x, f(x), modules=(n1, "sympy"))
    # 断言 if1 在给定参数 1 的情况下的值是否等于 'first f'
    assert if1(1) == 'first f'
    if2 = lambdify(x, g(x), modules=(n2, "sympy"))
    # 断言 if1 在再次给定参数 1 的情况下的值仍然等于 'first f'
    assert if1(1) == 'first f'
    # 断言 if2 在给定参数 1 的情况下的值是否等于 'function g'
    assert if2(1) == 'function g'


# 定义测试 implemented_function 函数的 lambdify 函数
def test_imps():
    # 创建两个具有相同名称但不同实现的 implemented_function f 和 g
    f = implemented_function('f', lambda x: 2*x)
    g = implemented_function('f', lambda x: math.sqrt(x))
    # 使用 lambdify 分别基于 f 和 g 创建函数 l1 和 l2
    l1 = lambdify(x, f(x))
    l2 = lambdify(x, g(x))
    # 断言 f(x) 和 g(x) 的字符串表示是否相同
    assert str(f(x)) == str(g(x))
    # 断言 l1 在给定参数 3 的情况下的值是否等于 6
    assert l1(3) == 6
    # 断言 l2 在给定参数 3 的情况下的值是否等于 sqrt(3)
    assert l2(3) == math.sqrt(3)
    # 测试是否可以将 sympy.Function 作为输入传递
    func = sympy.Function('myfunc')
    # 断言 func 没有属性 '_imp_'
    assert not hasattr(func, '_imp_')
    my_f = implemented_function(func, lambda x: 2*x)
    # 断言 my_f 有属性 '_imp_'
    assert hasattr(my_f, '_imp_')
    # 断言对于具有相同名称但不同实现的 f 和 f2，使用 lambdify 会引发 ValueError
    f2 = implemented_function("f", lambda x: x + 101)
    raises(ValueError, lambda: lambdify(x, f(f2(x))))


# 定义测试 implemented_function 错误的 lambdify 函数
def test_imps_errors():
    # 测试 implemented_function 可以返回的错误类型，并且仍然能够形成表达式
    # 参见：https://github.com/sympy/sympy/issues/10810
    #
    # XXX: 此处删除了 AttributeError。这个测试是因为 issue 10810 添加的，
    # 但该问题涉及到 ValueError。在同一上下文中支持捕获 AttributeError 似乎不合理...
    # 使用 product 函数生成两个元组的笛卡尔积：(0, 0., 2, 2.0) 和 (TypeError, ValueError)
    for val, error_class in product((0, 0., 2, 2.0), (TypeError, ValueError)):

        # 定义一个内部函数 myfunc，接受参数 a
        def myfunc(a):
            # 如果 a 等于 0，则抛出 error_class 异常
            if a == 0:
                raise error_class
            # 如果 a 不等于 0，则返回 1
            return 1

        # 使用 implemented_function 函数创建一个名为 'f' 的函数，其实现为 myfunc
        f = implemented_function('f', myfunc)
        
        # 调用函数 f，并传入参数 val，得到表达式的值
        expr = f(val)
        
        # 使用断言验证 expr 等于 f(val)，即验证函数 f 的返回值与其调用结果相同
        assert expr == f(val)


这段代码演示了使用 Python 中的 `product` 函数生成多个参数组合，并在每个组合上定义一个函数并进行调用和断言验证。
def test_imps_wrong_args():
    # 测试传入错误参数的情况，期望触发 ValueError 异常，使用 lambda 匿名函数调用 implemented_function 函数并传入 sin 函数和一个简单的 lambda 函数
    raises(ValueError, lambda: implemented_function(sin, lambda x: x))


def test_lambdify_imps():
    # 测试 lambdify 函数与 implemented functions 的结合使用
    # 首先测试基本的 sympy lambdify
    f = sympy.cos
    assert lambdify(x, f(x))(0) == 1
    assert lambdify(x, 1 + f(x))(0) == 2
    assert lambdify((x, y), y + f(x))(0, 1) == 2
    # 创建一个自定义的 implemented function 并进行测试
    f = implemented_function("f", lambda x: x + 100)
    assert lambdify(x, f(x))(0) == 100
    assert lambdify(x, 1 + f(x))(0) == 101
    assert lambdify((x, y), y + f(x))(0, 1) == 101
    # lambdify 也能处理元组、列表、字典作为表达式
    lam = lambdify(x, (f(x), x))
    assert lam(3) == (103, 3)
    lam = lambdify(x, [f(x), x])
    assert lam(3) == [103, 3]
    lam = lambdify(x, [f(x), (f(x), x)])
    assert lam(3) == [103, (103, 3)]
    lam = lambdify(x, {f(x): x})
    assert lam(3) == {103: 3}
    lam = lambdify(x, {f(x): x})
    assert lam(3) == {103: 3}
    lam = lambdify(x, {x: f(x)})
    assert lam(3) == {3: 103}
    # 检查默认情况下 imp 的优先级高于其他命名空间
    d = {'f': lambda x: x + 99}
    lam = lambdify(x, f(x), d)
    assert lam(3) == 103
    # 除非传入了 use_imps=False 参数
    lam = lambdify(x, f(x), d, use_imps=False)
    assert lam(3) == 102


def test_dummification():
    t = symbols('t')
    F = Function('F')
    G = Function('G')
    # "\alpha" 不是有效的 Python 变量名
    # lambdify 应该为其替换一个虚拟变量，并返回，不应出现语法错误
    alpha = symbols(r'\alpha')
    some_expr = 2 * F(t)**2 / G(t)
    lam = lambdify((F(t), G(t)), some_expr)
    assert lam(3, 9) == 2
    lam = lambdify(sin(t), 2 * sin(t)**2)
    assert lam(F(t)) == 2 * F(t)**2
    # 测试 \alpha 是否被正确处理为虚拟变量
    lam = lambdify((alpha, t), 2*alpha + t)
    assert lam(2, 1) == 5
    raises(SyntaxError, lambda: lambdify(F(t) * G(t), F(t) * G(t) + 5))
    raises(SyntaxError, lambda: lambdify(2 * F(t), 2 * F(t) + 5))
    raises(SyntaxError, lambda: lambdify(2 * F(t), 4 * F(t) + 5))


def test_lambdify__arguments_with_invalid_python_identifiers():
    # 查看 sympy/sympy#26690
    N = CoordSys3D('N')
    xn, yn, zn = N.base_scalars()
    expr = xn + yn
    f = lambdify([xn, yn], expr)
    res = f(0.2, 0.3)
    ref = 0.2 + 0.3
    assert abs(res-ref) < 1e-15


def test_curly_matrix_symbol():
    # Issue #15009
    curlyv = sympy.MatrixSymbol("{v}", 2, 1)
    lam = lambdify(curlyv, curlyv)
    assert lam(1)==1
    lam = lambdify(curlyv, curlyv, dummify=True)
    assert lam(1)==1


def test_python_keywords():
    # 测试问题 #7452。自动的虚拟化应确保使用 Python 保留关键字作为符号名创建有效的 lambda 函数。这是一个额外的回归测试。
    python_if = symbols('if')
    expr = python_if / 2
    f = lambdify(python_if, expr)
    assert f(4.0) == 2.0


def test_lambdify_docstring():
    func = lambdify((w, x, y, z), w + x + y + z)
    # 将多行字符串赋给 ref 变量，表示由 lambdify 创建的函数的文档字符串的参考值
    ref = (
        "Created with lambdify. Signature:\n\n"
        "func(w, x, y, z)\n\n"
        "Expression:\n\n"
        "w + x + y + z"
    ).splitlines()
    
    # 使用断言检查 func 函数的文档字符串的前若干行是否与 ref 相符
    assert func.__doc__.splitlines()[:len(ref)] == ref
    
    # 使用 symbols 函数创建符号 'a1' 到 'a25' 的符号列表
    syms = symbols('a1:26')
    
    # 使用 lambdify 函数将符号列表 syms 和它们的总和 sum(syms) 转换为函数 func
    func = lambdify(syms, sum(syms))
    
    # 将多行字符串赋给 ref 变量，表示由 lambdify 创建的函数的文档字符串的参考值
    ref = (
        "Created with lambdify. Signature:\n\n"
        "func(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,\n"
        "        a16, a17, a18, a19, a20, a21, a22, a23, a24, a25)\n\n"
        "Expression:\n\n"
        "a1 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19 + a2 + a20 +..."
    ).splitlines()
    
    # 使用断言检查 func 函数的文档字符串的前若干行是否与 ref 相符
    assert func.__doc__.splitlines()[:len(ref)] == ref
#================== Test special printers ==========================

# 测试特殊的打印功能

def test_special_printers():
    # 从 sympy.printing.lambdarepr 导入 IntervalPrinter
    from sympy.printing.lambdarepr import IntervalPrinter

    # 定义 intervalrepr 函数，使用 IntervalPrinter 打印表达式 expr
    def intervalrepr(expr):
        return IntervalPrinter().doprint(expr)

    # 定义表达式 expr
    expr = sqrt(sqrt(2) + sqrt(3)) + S.Half

    # 使用 lambdify 创建函数 func0，使用 intervalrepr 作为打印器
    func0 = lambdify((), expr, modules="mpmath", printer=intervalrepr)
    # 使用 lambdify 创建函数 func1，使用 IntervalPrinter 作为打印器
    func1 = lambdify((), expr, modules="mpmath", printer=IntervalPrinter)
    # 使用 lambdify 创建函数 func2，使用 IntervalPrinter() 实例作为打印器
    func2 = lambdify((), expr, modules="mpmath", printer=IntervalPrinter())

    # 获取 mpmath.mpi 的类型
    mpi = type(mpmath.mpi(1, 2))

    # 断言 func0() 的返回值是 mpi 类型
    assert isinstance(func0(), mpi)
    # 断言 func1() 的返回值是 mpi 类型
    assert isinstance(func1(), mpi)
    # 断言 func2() 的返回值是 mpi 类型
    assert isinstance(func2(), mpi)

    # To check Is lambdify loggamma works for mpmath or not
    # 使用 lambdify 创建表达式 exp1, exp2, exp3，分别计算 loggamma(x) 的值
    exp1 = lambdify(x, loggamma(x), 'mpmath')(5)
    exp2 = lambdify(x, loggamma(x), 'mpmath')(1.8)
    exp3 = lambdify(x, loggamma(x), 'mpmath')(15)
    exp_ls = [exp1, exp2, exp3]

    # 计算 mpmath.loggamma 的结果 sol1, sol2, sol3
    sol1 = mpmath.loggamma(5)
    sol2 = mpmath.loggamma(1.8)
    sol3 = mpmath.loggamma(15)
    sol_ls = [sol1, sol2, sol3]

    # 断言 exp_ls 与 sol_ls 相等
    assert exp_ls == sol_ls


# 测试 true 和 false 函数
def test_true_false():
    # 确保精确比较，使用 is 操作符
    assert lambdify([], true)() is True
    assert lambdify([], false)() is False


# 测试 issue 2790
def test_issue_2790():
    # 断言 lambdify((x, (y, z)), x + y)(1, (2, 4)) 的返回值为 3
    assert lambdify((x, (y, z)), x + y)(1, (2, 4)) == 3
    # 断言 lambdify((x, (y, (w, z))), w + x + y + z)(1, (2, (3, 4))) 的返回值为 10
    assert lambdify((x, (y, (w, z))), w + x + y + z)(1, (2, (3, 4))) == 10
    # 断言 lambdify(x, x + 1, dummify=False)(1) 的返回值为 2
    assert lambdify(x, x + 1, dummify=False)(1) == 2


# 测试 issue 12092
def test_issue_12092():
    # 创建函数 f，使用 lambda x: x**2 作为实现函数
    f = implemented_function('f', lambda x: x**2)
    # 断言 f(f(2)).evalf() 的返回值为 Float(16)
    assert f(f(2)).evalf() == Float(16)


# 测试 issue 14911
def test_issue_14911():
    # 定义 Variable 类，继承自 sympy.Symbol
    class Variable(sympy.Symbol):
        # 定义 _sympystr 方法，使用 printer.doprint(self.name) 打印名称
        def _sympystr(self, printer):
            return printer.doprint(self.name)

        # 将 _lambdacode 和 _numpycode 设置为 _sympystr 方法
        _lambdacode = _sympystr
        _numpycode = _sympystr

    # 创建 Variable 类实例 x
    x = Variable('x')
    # 创建表达式 y = 2 * x
    y = 2 * x
    # 使用 LambdaPrinter 打印表达式 y
    code = LambdaPrinter().doprint(y)
    # 断言 code 去除空格后与 '2*x' 相等
    assert code.replace(' ', '') == '2*x'


# 测试 ITE 函数
def test_ITE():
    # 断言 lambdify((x, y, z), ITE(x, y, z))(True, 5, 3) 的返回值为 5
    assert lambdify((x, y, z), ITE(x, y, z))(True, 5, 3) == 5
    # 断言 lambdify((x, y, z), ITE(x, y, z))(False, 5, 3) 的返回值为 3
    assert lambdify((x, y, z), ITE(x, y, z))(False, 5, 3) == 3


# 测试 Min 和 Max 函数
def test_Min_Max():
    # 断言 lambdify((x, y, z), Min(x, y, z))(1, 2, 3) 的返回值为 1
    assert lambdify((x, y, z), Min(x, y, z))(1, 2, 3) == 1
    # 断言 lambdify((x, y, z), Max(x, y, z))(1, 2, 3) 的返回值为 3
    assert lambdify((x, y, z), Max(x, y, z))(1, 2, 3) == 3


# 测试 Indexed 函数
def test_Indexed():
    # Issue #10934
    # 如果 numpy 未安装，则跳过测试
    if not numpy:
        skip("numpy not installed")

    # 创建 IndexedBase 类实例 a
    a = IndexedBase('a')
    # 创建符号变量 i, j
    i, j = symbols('i j')
    # 创建 numpy 数组 b
    b = numpy.array([[1, 2], [3, 4]])
    # 断言 lambdify(a, Sum(a[x, y], (x, 0, 1), (y, 0, 1)))(b) 的返回值为 10
    assert lambdify(a, Sum(a[x, y], (x, 0, 1), (y, 0, 1)))(b) == 10


# 测试 issue 12173
def test_issue_12173():
    # 测试 issue 12173
    # 使用 lambdify((x, y), uppergamma(x, y),"mpmath")(1, 2) 计算表达式 expr1
    expr1 = lambdify((x, y), uppergamma(x, y),"mpmath")(1, 2)
    # 使用 lambdify((x, y), lowergamma(x, y),"mpmath")(1, 2) 计算表达式 expr2
    expr2 = lambdify((x, y), lowergamma(x, y),"mpmath")(1, 2)
    # 断言 expr1 等于 uppergamma(1, 2).evalf()
    assert expr1 == uppergamma(1, 2).evalf()
    # 断言 expr2 等于 lowergamma(1, 2).evalf()
    assert expr2 == lowergamma(1, 2).evalf()


# 测试 issue 13642
def test_issue_13642():
    # 如果 numpy 未安装，则跳过测试
    if not numpy:
        skip("numpy not installed")
    # 创建函数 f，使用 sinc(x) 作为表达式
    f = lambdify(x, sinc(x))
    # 断言 Abs(f(1) - sinc(1)).n() 小于 1e-15
    assert Abs(f(1) - sinc(1)).n() < 1e-15


# 测试 sinc 函数与 mpmath 的结合
def test_sinc_mpmath():
    # 使用 sinc(x) 与 "mpmath" 选项创建函数 f
    f = lambdify(x, sinc(x), "mpmath")
    #
    # 创建一个基于符号变量的匿名函数，接受一个参数 d1 并返回 d1 + 1 的结果
    f1 = lambdify(d1, d1 + 1, dummify=False)
    # 断言匿名函数 f1 在参数为 2 时返回值为 3
    assert f1(2) == 3
    
    # 创建一个基于符号变量的匿名函数，接受一个参数 d1 并返回 d1 + 1 的结果
    # 默认情况下，dummify 参数为 True，表示进行符号变量的虚拟化（Dummy）
    f1b = lambdify(d1, d1 + 1)
    # 断言匿名函数 f1b 在参数为 2 时返回值为 3
    assert f1b(2) == 3
    
    # 创建一个符号变量 'x'
    d2 = Dummy('x')
    # 创建一个基于符号变量的匿名函数，接受一个参数 d2 并返回 d2 + 1 的结果
    f2 = lambdify(d2, d2 + 1)
    # 断言匿名函数 f2 在参数为 2 时返回值为 3
    assert f2(2) == 3
    
    # 创建一个基于符号变量的匿名函数，接受一个列表 [[d2]] 作为参数，返回 d2 + 1 的结果
    f3 = lambdify([[d2]], d2 + 1)
    # 断言匿名函数 f3 在参数为 [2]（一个包含一个元素的列表）时返回值为 3
    assert f3([2]) == 3
def test_lambdify_mixed_symbol_dummy_args():
    # 创建一个虚拟的符号对象
    d = Dummy()
    # 使用虚拟符号对象创建一个 SymPy 符号
    dsym = symbols(str(d))
    # 创建一个 lambda 函数，接受两个参数 d 和 dsym，并返回它们的差值
    f = lambdify([d, dsym], d - dsym)
    # 断言 lambda 函数对于输入参数 (4, 1) 应返回 3
    assert f(4, 1) == 3


def test_numpy_array_arg():
    # 检查 numpy 是否安装，若未安装则跳过测试
    if not numpy:
        skip("numpy not installed")
    
    # 创建一个 numpy 数组作为 lambdify 的输入参数
    f = lambdify([[x, y]], x*x + y, 'numpy')
    # 断言 lambda 函数对于 numpy 数组 [2.0, 1.0] 应返回 5
    assert f(numpy.array([2.0, 1.0])) == 5


def test_scipy_fns():
    # 检查 scipy 是否安装，若未安装则跳过测试
    if not scipy:
        skip("scipy not installed")
    
    # 定义一组单参数的 SymPy 和 SciPy 函数
    single_arg_sympy_fns = [Ei, erf, erfc, factorial, gamma, loggamma, digamma, Si, Ci]
    single_arg_scipy_fns = [scipy.special.expi, scipy.special.erf, scipy.special.erfc,
        scipy.special.factorial, scipy.special.gamma, scipy.special.gammaln,
                            scipy.special.psi, scipy.special.sici, scipy.special.sici]
    
    # 设置随机种子
    numpy.random.seed(0)
    
    # 迭代处理 SymPy 和 SciPy 函数对
    for (sympy_fn, scipy_fn) in zip(single_arg_sympy_fns, single_arg_scipy_fns):
        # 创建 lambdify 函数，将 sympy_fn 应用于变量 x，使用 SciPy 的模块
        f = lambdify(x, sympy_fn(x), modules="scipy")
        
        # 进行 20 次迭代，生成随机复数
        for i in range(20):
            tv = numpy.random.uniform(-10, 10) + 1j*numpy.random.uniform(-5, 5)
            
            # 特殊处理：factorial(z) 在实部小于 0 时为 0，并且不支持复数
            if sympy_fn == factorial:
                tv = numpy.abs(tv)
            
            # 特殊处理：gammaln 仅支持实数参数，并且在负实轴有一个分支切割
            if sympy_fn == loggamma:
                tv = numpy.abs(tv)
            
            # 特殊处理：digamma 计算为 polygamma(0, z)，仅支持实数参数
            if sympy_fn == digamma:
                tv = numpy.real(tv)
            
            # 计算 SymPy 函数在 tv 处的结果
            sympy_result = sympy_fn(tv).evalf()
            # 计算 SciPy 函数在 tv 处的结果
            scipy_result = scipy_fn(tv)
            
            # 特殊处理：sici 返回一个包含 Si 和 Ci 的元组，需要解包
            if sympy_fn == Si:
                scipy_result = scipy_fn(tv)[0]
            if sympy_fn == Ci:
                scipy_result = scipy_fn(tv)[1]
            
            # 断言 lambdify 函数在 tv 处对 SymPy 和 SciPy 函数结果的误差小于指定精度
            assert abs(f(tv) - sympy_result) < 1e-13*(1 + abs(sympy_result))
            assert abs(f(tv) - scipy_result) < 1e-13*(1 + abs(sympy_result))
    
    # 定义一组双参数的 SymPy 和 SciPy 函数
    double_arg_sympy_fns = [RisingFactorial, besselj, bessely, besseli,
                            besselk, polygamma]
    double_arg_scipy_fns = [scipy.special.poch, scipy.special.jv,
                            scipy.special.yv, scipy.special.iv, scipy.special.kv, scipy.special.polygamma]
    # 对于每对 sympy_fn 和 scipy_fn 中的函数，创建一个使用 scipy 模块的 lambdify 函数
    f = lambdify((x, y), sympy_fn(x, y), modules="scipy")
    
    # 对于范围在 0 到 19 的每一个整数 i 进行循环迭代
    for i in range(20):
        # 生成一个位于 [-10, 10] 范围内的随机实数 tv1
        tv1 = numpy.random.uniform(-10, 10)
        
        # 生成一个复数，实部位于 [-10, 10]，虚部位于 [-5, 5] 的随机复数 tv2
        tv2 = numpy.random.uniform(-10, 10) + 1j*numpy.random.uniform(-5, 5)
        
        # 对于 RisingFactorial 和 polygamma 函数，SciPy 仅支持实数阶数的贝塞尔函数
        # 因此，如果 sympy_fn 是 RisingFactorial 或 polygamma，需要将 tv2 转换为其实部
        if sympy_fn in (RisingFactorial, polygamma):
            tv2 = numpy.real(tv2)
        
        # 如果 sympy_fn 是 polygamma 函数，确保 tv1 是一个非负整数
        if sympy_fn == polygamma:
            tv1 = abs(int(tv1))  # polygamma 函数的第一个参数必须是非负整数
        
        # 计算 sympy_fn(tv1, tv2) 的数值结果
        sympy_result = sympy_fn(tv1, tv2).evalf()
        
        # 断言计算得到的 f(tv1, tv2) 与 sympy_fn(tv1, tv2) 的差值小于给定的容差
        assert abs(f(tv1, tv2) - sympy_result) < 1e-13*(1 + abs(sympy_result))
        
        # 断言计算得到的 f(tv1, tv2) 与 scipy_fn(tv1, tv2) 的差值小于给定的容差
        assert abs(f(tv1, tv2) - scipy_fn(tv1, tv2)) < 1e-13*(1 + abs(sympy_result))
#python
# 测试 scipy.polys 模块的功能
def test_scipy_polys():
    # 如果没有安装 scipy，则跳过测试
    if not scipy:
        skip("scipy not installed")
    # 设置随机种子以便重现测试结果
    numpy.random.seed(0)

    # 定义符号变量
    params = symbols('n k a b')
    
    # 各多项式函数及其参数数量的列表
    polys = [
        (chebyshevt, 1),
        (chebyshevu, 1),
        (legendre, 1),
        (hermite, 1),
        (laguerre, 1),
        (gegenbauer, 2),
        (assoc_legendre, 2),
        (assoc_laguerre, 2),
        (jacobi, 3)
    ]

    # 错误消息模板，用于显示测试失败的详细信息
    msg = \
        "The random test of the function {func} with the arguments " \
        "{args} had failed because the SymPy result {sympy_result} " \
        "and SciPy result {scipy_result} had failed to converge " \
        "within the tolerance {tol} " \
        "(Actual absolute difference : {diff})"

    # 对每个多项式函数进行测试
    for sympy_fn, num_params in polys:
        # 准备参数列表
        args = params[:num_params] + (x,)
        # 将符号函数转换为可调用的数值函数
        f = lambdify(args, sympy_fn(*args))
        # 进行 10 次随机测试
        for _ in range(10):
            # 生成随机测试参数
            tn = numpy.random.randint(3, 10)
            tparams = tuple(numpy.random.uniform(0, 5, size=num_params-1))
            tv = numpy.random.uniform(-10, 10) + 1j*numpy.random.uniform(-5, 5)
            
            # 对于 Hermite 多项式，SciPy 只支持实数参数
            if sympy_fn == hermite:
                tv = numpy.real(tv)
            
            # 对于关联 Legendre 多项式，需要 x 在 (-1, 1) 范围内，并且参数为整数
            if sympy_fn == assoc_legendre:
                tv = numpy.random.uniform(-1, 1)
                tparams = tuple(numpy.random.randint(1, tn, size=1))

            # 组合参数
            vals = (tn,) + tparams + (tv,)
            
            # 计算 SciPy 和 SymPy 的结果
            scipy_result = f(*vals)
            sympy_result = sympy_fn(*vals).evalf()
            
            # 设置绝对容差值
            atol = 1e-9*(1 + abs(sympy_result))
            # 计算实际差值
            diff = abs(scipy_result - sympy_result)
            
            # 断言结果在容差范围内
            try:
                assert diff < atol
            except TypeError:
                # 若出现类型错误，则抛出详细错误消息
                raise AssertionError(
                    msg.format(
                        func=repr(sympy_fn),
                        args=repr(vals),
                        sympy_result=repr(sympy_result),
                        scipy_result=repr(scipy_result),
                        diff=diff,
                        tol=atol)
                    )


# 测试 lambdify 函数是否能正确生成函数并且 inspect.getsource 能正常工作
def test_lambdify_inspect():
    f = lambdify(x, x**2)
    # 断言 x**2 在 lambdify 生成的函数源代码中
    assert 'x**2' in inspect.getsource(f)


# 测试 lambdify 函数处理符号表达式转换成 Python 函数的正确性
def test_issue_14941():
    x, y = Dummy(), Dummy()

    # 测试字典类型的符号表达式转换
    f1 = lambdify([x, y], {x: 3, y: 3}, 'sympy')
    assert f1(2, 3) == {2: 3, 3: 3}

    # 测试元组类型的符号表达式转换
    f2 = lambdify([x, y], (y, x), 'sympy')
    assert f2(2, 3) == (3, 2)
    f2b = lambdify([], (1,))  # gh-23224
    assert f2b() == (1,)

    # 测试列表类型的符号表达式转换
    f3 = lambdify([x, y], [y, x], 'sympy')
    assert f3(2, 3) == [3, 2]


# 测试 lambdify 函数处理 Derivative 类型参数时的问题修复
def test_lambdify_Derivative_arg_issue_16468():
    f = Function('f')(x)
    fx = f.diff()
    assert lambdify((f, fx), f + fx)(10, 5) == 15
    assert eval(lambdastr((f, fx), f/fx))(10, 5) == 2
    # 使用 raises 函数测试 lambda 函数在 eval 中的错误处理能力
    raises(Exception, lambda:
        eval(lambdastr((f, fx), f/fx, dummify=False)))
    # 使用 assert 断言验证 lambda 函数的求值结果是否正确，使用 dummify=True 消除虚拟变量
    assert eval(lambdastr((f, fx), f/fx, dummify=True))(10, 5) == 2
    # 使用 assert 断言验证 lambda 函数在符号运算中的求值结果是否正确，使用 dummify=True 消除虚拟变量
    assert eval(lambdastr((fx, f), f/fx, dummify=True))(S(10), 5) == S.Half
    # 使用 lambdify 函数将符号表达式转化为可调用的 lambda 函数，并验证其在特定输入下的计算结果
    assert lambdify(fx, 1 + fx)(41) == 42
    # 使用 eval 函数验证 lambda 函数在符号运算中的求值结果是否正确，使用 dummify=True 消除虚拟变量
    assert eval(lambdastr(fx, 1 + fx, dummify=True))(41) == 42
# 定义测试函数，用于验证 lambdify 函数生成的 lambda 表达式的正确性
def test_imag_real():
    # 创建 lambda 函数 f_re，用于计算复数的实部
    f_re = lambdify([z], sympy.re(z))
    # 设定测试用例中的复数值
    val = 3+2j
    # 验证 lambda 函数计算的实部与预期值相等
    assert f_re(val) == val.real

    # 创建 lambda 函数 f_im，用于计算复数的虚部
    f_im = lambdify([z], sympy.im(z))  # see #15400
    # 验证 lambda 函数计算的虚部与预期值相等
    assert f_im(val) == val.imag


# 验证 MatrixSymbol 对象在求逆矩阵时的问题（Issue 15578）
def test_MatrixSymbol_issue_15578():
    # 若未安装 numpy，跳过测试
    if not numpy:
        skip("numpy not installed")
    # 定义一个 2x2 的矩阵符号 A
    A = MatrixSymbol('A', 2, 2)
    # 定义 A0 作为 numpy 数组，表示具体的矩阵值
    A0 = numpy.array([[1, 2], [3, 4]])
    # 创建 lambda 函数 f，用于计算矩阵 A 的逆矩阵
    f = lambdify(A, A**(-1))
    # 验证计算结果与预期的逆矩阵值相近
    assert numpy.allclose(f(A0), numpy.array([[-2., 1.], [1.5, -0.5]]))
    # 创建 lambda 函数 g，用于计算矩阵 A 的三次方
    g = lambdify(A, A**3)
    # 验证计算结果与预期的三次方矩阵值相近
    assert numpy.allclose(g(A0), numpy.array([[37, 54], [81, 118]]))


# 验证与 Issue 15654 相关的问题
def test_issue_15654():
    # 若未安装 scipy，跳过测试
    if not scipy:
        skip("scipy not installed")
    # 导入符号变量 n, l, r, Z
    from sympy.abc import n, l, r, Z
    # 设置符号变量的具体值
    nv, lv, rv, Zv = 1, 0, 3, 1
    # 使用 hydrogen.R_nl 函数计算符号变量对应的数值
    sympy_value = hydrogen.R_nl(nv, lv, rv, Zv).evalf()
    # 创建 lambda 函数 f，用于计算 hydrogen.R_nl 的数值结果
    f = lambdify((n, l, r, Z), hydrogen.R_nl(n, l, r, Z))
    # 使用 scipy 计算同样的数值结果
    scipy_value = f(nv, lv, rv, Zv)
    # 验证两种方法计算得到的结果在精度要求下相等
    assert abs(sympy_value - scipy_value) < 1e-15


# 验证与 Issue 15827 相关的问题
def test_issue_15827():
    # 若未安装 numpy，跳过测试
    if not numpy:
        skip("numpy not installed")
    # 定义矩阵符号 A, B, C, D，分别表示不同维度的矩阵
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 2, 3)
    C = MatrixSymbol("C", 3, 4)
    D = MatrixSymbol("D", 4, 5)
    # 定义符号变量 k
    k=symbols("k")
    # 创建 lambda 函数 f，用于计算 2*k*A
    f = lambdify(A, (2*k)*A)
    # 验证计算结果与预期的数值矩阵相等
    assert numpy.array_equal(f(numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])), \
    numpy.array([[2*k, 4*k, 6*k], [2*k, 4*k, 6*k], [2*k, 4*k, 6*k]], dtype=object))

    # 创建 lambda 函数 g，用于计算 (2+k)*A
    g = lambdify(A, (2+k)*A)
    # 验证计算结果与预期的数值矩阵相等
    assert numpy.array_equal(g(numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])), \
    numpy.array([[k + 2, 2*k + 4, 3*k + 6], [k + 2, 2*k + 4, 3*k + 6], \
    [k + 2, 2*k + 4, 3*k + 6]], dtype=object))

    # 创建 lambda 函数 h，用于计算 2*A
    h = lambdify(A, 2*A)
    # 验证计算结果与预期的数值矩阵相等
    assert numpy.array_equal(h(numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])), \
    numpy.array([[2, 4, 6], [2, 4, 6], [2, 4, 6]]))

    # 创建 lambda 函数 i，用于计算 2*B*C*D
    i = lambdify((B, C, D), 2*B*C*D)
    # 验证计算结果与预期的数值矩阵相等
    assert numpy.array_equal(i(numpy.array([[1, 2, 3], [1, 2, 3]]), numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]), \
    numpy.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])), numpy.array([[ 120, 240, 360, 480, 600], \
    [ 120, 240, 360, 480, 600]]))


# 验证与 Issue 16930 相关的问题
def test_issue_16930():
    # 若未安装 scipy，跳过测试
    if not scipy:
        skip("scipy not installed")

    # 定义符号变量 x
    x = symbols("x")
    # 创建 lambda 函数 f，用于计算 S.GoldenRatio * x**2
    f = lambda x:  S.GoldenRatio * x**2
    # 创建 lambdify 函数 f_，用于生成计算数值结果的 lambda 函数，使用 scipy 模块
    f_ = lambdify(x, f(x), modules='scipy')
    # 验证计算结果与黄金比例常数相等
    assert f_(1) == scipy.constants.golden_ratio


# 验证与 Issue 17898 相关的问题
def test_issue_17898():
    # 若未安装 scipy，跳过测试
    if not scipy:
        skip("scipy not installed")
    # 定义符号变量 x
    x = symbols("x")
    # 创建 lambdify 函数 f_，用于生成计算 sympy.LambertW(x,-1) 结果的 lambda 函数，使用 scipy 模块
    f_ = lambdify([x], sympy.LambertW(x,-1), modules='scipy')
    # 验证计算结果与 mpmath.lambertw(0.1, -1) 相等
    assert f_(0.1) == mpmath.lambertw(0.1, -1)


# 验证与 Issue 13167 和 21411 相关的问题
def test_issue_13167_21411():
    # 若未安装 numpy，跳过测试
    if not numpy:
        skip("numpy not installed")
    # 创建 lambda 函数 f1，用于计算 sympy.Heaviside(x) 的结果
    f1 = lambdify(x, sympy.Heaviside(x))
    # 创建 lambda 函数 f2，用于计算
    # 断言检查第一个返回值的绝对值是否小于 1e-15
    assert Abs(res2[0]).n() < 1e-15
    # 断言检查第二个返回值与 1 的差的绝对值是否小于 1e-15
    assert Abs(res2[1] - 1).n() < 1e-15
    # 断言检查第三个返回值与 1 的差的绝对值是否小于 1e-15
    assert Abs(res2[2] - 1).n() < 1e-15
def test_single_e():
    # 创建一个将变量 x 映射到常数 E 的函数
    f = lambdify(x, E)
    # 断言函数在参数 23 上的返回值等于 exp(1.0)
    assert f(23) == exp(1.0)

def test_issue_16536():
    if not scipy:
        # 如果 scipy 未安装，则跳过此测试
        skip("scipy not installed")

    # 创建符号变量 a
    a = symbols('a')
    # 计算 lowergamma(a, x) 的表达式
    f1 = lowergamma(a, x)
    # 使用 scipy 模块创建将 (a, x) 映射到 f1 的函数
    F = lambdify((a, x), f1, modules='scipy')
    # 断言 lowergamma(1, 3) 的值与 F(1, 3) 的值在误差范围内相等
    assert abs(lowergamma(1, 3) - F(1, 3)) <= 1e-10

    # 计算 uppergamma(a, x) 的表达式
    f2 = uppergamma(a, x)
    # 使用 scipy 模块创建将 (a, x) 映射到 f2 的函数
    F = lambdify((a, x), f2, modules='scipy')
    # 断言 uppergamma(1, 3) 的值与 F(1, 3) 的值在误差范围内相等
    assert abs(uppergamma(1, 3) - F(1, 3)) <= 1e-10


def test_issue_22726():
    if not numpy:
        # 如果 numpy 未安装，则跳过此测试
        skip("numpy not installed")

    # 创建符号变量 x1 和 x2
    x1, x2 = symbols('x1 x2')
    # 计算 Max(S.Zero, Min(x1, x2)) 的表达式
    f = Max(S.Zero, Min(x1, x2))
    # 对表达式 f 求偏导数，关于 (x1, x2)
    g = derive_by_array(f, (x1, x2))
    # 使用 numpy 模块创建将 (x1, x2) 映射到 g 的函数
    G = lambdify((x1, x2), g, modules='numpy')
    # 定义点的值字典
    point = {x1: 1, x2: 2}
    # 断言 g 在给定点的数值结果与 G(*point.values()) 在误差范围内相等
    assert (abs(g.subs(point) - G(*point.values())) <= 1e-10).all()


def test_issue_22739():
    if not numpy:
        # 如果 numpy 未安装，则跳过此测试
        skip("numpy not installed")

    # 创建符号变量 x1 和 x2
    x1, x2 = symbols('x1 x2')
    # 计算 Heaviside(Min(x1, x2)) 的表达式
    f = Heaviside(Min(x1, x2))
    # 使用 numpy 模块创建将 (x1, x2) 映射到 f 的函数
    F = lambdify((x1, x2), f, modules='numpy')
    # 定义点的值字典
    point = {x1: 1, x2: 2}
    # 断言 f 在给定点的数值结果与 F(*point.values()) 在误差范围内相等
    assert abs(f.subs(point) - F(*point.values())) <= 1e-10


def test_issue_22992():
    if not numpy:
        # 如果 numpy 未安装，则跳过此测试
        skip("numpy not installed")

    # 创建符号变量 a 和 t
    a, t = symbols('a t')
    # 计算表达式 a*(log(cot(t/2)) - cos(t))
    expr = a*(log(cot(t/2)) - cos(t))
    # 使用 numpy 模块创建将 (a, t) 映射到 expr 的函数
    F = lambdify([a, t], expr, 'numpy')

    # 定义点的值字典
    point = {a: 10, t: 2}

    # 断言 expr 在给定点的数值结果与 F(*point.values()) 在误差范围内相等
    assert abs(expr.subs(point) - F(*point.values())) <= 1e-10

    # 使用标准的数学库创建将 (a, t) 映射到 expr 的函数
    F = lambdify([a, t], expr)

    # 断言 expr 在给定点的数值结果与 F(*point.values()) 在误差范围内相等
    assert abs(expr.subs(point) - F(*point.values())) <= 1e-10


def test_issue_19764():
    if not numpy:
        # 如果 numpy 未安装，则跳过此测试
        skip("numpy not installed")

    # 创建 Array([x, x**2]) 的表达式
    expr = Array([x, x**2])
    # 使用 numpy 模块创建将 x 映射到 expr 的函数
    f = lambdify(x, expr, 'numpy')

    # 断言 f(1) 的返回类型为 numpy.ndarray
    assert f(1).__class__ == numpy.ndarray


def test_issue_20070():
    if not numba:
        # 如果 numba 未安装，则跳过此测试
        skip("numba not installed")

    # 创建将 x 映射到 sin(x) 的函数，使用 numpy 模块
    f = lambdify(x, sin(x), 'numpy')
    # 断言对 f 进行 numba 加速后，在参数 1 上的返回值等于 0.8414709848078965
    assert numba.jit(f, nopython=True)(1)==0.8414709848078965


def test_fresnel_integrals_scipy():
    if not scipy:
        # 如果 scipy 未安装，则跳过此测试
        skip("scipy not installed")

    # 计算 Fresnel 积分的两个函数
    f1 = fresnelc(x)
    f2 = fresnels(x)
    # 使用 scipy 模块创建将 x 映射到 f1 和 f2 的函数
    F1 = lambdify(x, f1, modules='scipy')
    F2 = lambdify(x, f2, modules='scipy')

    # 断言在参数 1.3 上计算的 Fresnel 积分值与 lambdify 函数的返回值在误差范围内相等
    assert abs(fresnelc(1.3) - F1(1.3)) <= 1e-10
    assert abs(fresnels(1.3) - F2(1.3)) <= 1e-10


def test_beta_scipy():
    if not scipy:
        # 如果 scipy 未安装，则跳过此测试
        skip("scipy not installed")

    # 计算 beta 函数的表达式
    f = beta(x, y)
    # 使用 scipy 模块创建将 (x, y) 映射到 f 的函数
    F = lambdify((x, y), f, modules='scipy')

    # 断言在参数 (1.3, 2.3) 上计算的 beta 函数值与 lambdify 函数的返回值在误差范围内相等
    assert abs(beta(1.3, 2.3) - F(1.3, 2.3)) <= 1e-10


def test_beta_math():
    # 计算 beta 函数的表达式
    f = beta(x, y)
    # 使用 math 模块创建将 (x, y) 映射到 f 的函数
    F = lambdify((x, y), f, modules='math')

    # 断言在参数 (1.3, 2.3) 上计算的 beta 函数值与 lambdify 函数的返回值在误差范围内相等
    assert abs(beta(1.3, 2.3) - F(1.3, 2.3)) <= 1e-10


def test_betainc_scipy():
    if not scipy:
        # 如果 scipy 未安装，则跳过此测试
        skip("scipy not installed")

    # 计算 betainc 函数的表达式
    f = betainc(w, x, y, z)
    #
    # 检查是否导入了 numpy 模块，若没有则跳过测试
    if not numpy:
        skip("numpy not installed")

    # 定义要测试的数学函数列表
    funcs = [expm1, log1p, exp2, log2, log10, hypot, logaddexp, logaddexp2]
    for func in funcs:
        # 检查函数的参数个数，根据不同情况设置表达式和参数
        if 2 in func.nargs:
            # 对具有两个参数的函数进行测试
            expr = func(x, y)
            args = (x, y)
            num_args = (0.3, 0.4)
        elif 1 in func.nargs:
            # 对具有一个参数的函数进行测试
            expr = func(x)
            args = (x,)
            num_args = (0.3,)
        else:
            # 抛出未实现的错误，暂时不处理除一元和二元函数以外的函数
            raise NotImplementedError("Need to handle other than unary & binary functions in test")

        # 将数学表达式转换为可调用的函数
        f = lambdify(args, expr)
        # 计算函数在给定参数下的结果
        result = f(*num_args)
        # 计算数学表达式在给定参数下的参考值
        reference = expr.subs(dict(zip(args, num_args))).evalf()
        # 断言计算结果与参考值在数值上相近
        assert numpy.allclose(result, float(reference))

    # 对于特定函数 logaddexp2 的测试
    # 创建 logaddexp2 的 lambdify 函数
    lae2 = lambdify((x, y), logaddexp2(log2(x), log2(y)))
    # 断言计算值与预期值之间的差异小于指定的误差范围
    assert abs(2.0**lae2(1e-50, 2.5e-50) - 3.5e-50) < 1e-62  # from NumPy's docstring
# 测试使用 scipy 特殊数学函数
def test_scipy_special_math():
    # 如果没有安装 scipy，则跳过测试
    if not scipy:
        skip("scipy not installed")

    # 将 cosm1 函数转换为使用 scipy 的 lambda 函数
    cm1 = lambdify((x,), cosm1(x), modules='scipy')
    # 断言 cosm1(1e-20) + 5e-41 的绝对值小于 1e-200
    assert abs(cm1(1e-20) + 5e-41) < 1e-200

    # 检查 scipy 版本是否大于等于 1.10
    have_scipy_1_10plus = tuple(map(int, scipy.version.version.split('.')[:2])) >= (1, 10)

    # 如果 scipy 版本大于等于 1.10
    if have_scipy_1_10plus:
        # 将 powm1 函数转换为使用 scipy 的 lambda 函数
        cm2 = lambdify((x, y), powm1(x, y), modules='scipy')
        # 断言 powm1(1.2, 1e-9) 的结果与给定值的差的绝对值小于 1e-17
        assert abs(cm2(1.2, 1e-9) - 1.82321557e-10)  < 1e-17


# 测试 scipy 的 Bernoulli 函数
def test_scipy_bernoulli():
    # 如果没有安装 scipy，则跳过测试
    if not scipy:
        skip("scipy not installed")

    # 将 bernoulli 函数转换为使用 scipy 的 lambda 函数
    bern = lambdify((x,), bernoulli(x), modules='scipy')
    # 断言 bernoulli(1) 的结果等于 0.5
    assert bern(1) == 0.5


# 测试 scipy 的 Harmonic 函数
def test_scipy_harmonic():
    # 如果没有安装 scipy，则跳过测试
    if not scipy:
        skip("scipy not installed")

    # 将 harmonic 函数转换为使用 scipy 的 lambda 函数
    hn = lambdify((x,), harmonic(x), modules='scipy')
    # 断言 harmonic(2) 的结果等于 1.5
    assert hn(2) == 1.5
    # 将 harmonic 函数转换为使用 scipy 的 lambda 函数，带两个参数
    hnm = lambdify((x, y), harmonic(x, y), modules='scipy')
    # 断言 harmonic(2, 2) 的结果等于 1.25
    assert hnm(2, 2) == 1.25


# 测试使用 CuPy 的数组参数
def test_cupy_array_arg():
    # 如果没有安装 CuPy，则跳过测试
    if not cupy:
        skip("CuPy not installed")

    # 将 x*x + y 转换为使用 CuPy 的 lambda 函数
    f = lambdify([[x, y]], x*x + y, 'cupy')
    # 对 CuPy 数组 [2.0, 1.0] 运行函数 f
    result = f(cupy.array([2.0, 1.0]))
    # 断言结果等于 5
    assert result == 5
    # 断言结果类型包含字符串 "cupy"
    assert "cupy" in str(type(result))


# 测试使用 CuPy 的数组参数，但使用了 NumPy 函数
def test_cupy_array_arg_using_numpy():
    # 对于 CuPy，可以运行 NumPy 函数，但官方支持程度不明确
    if not cupy:
        skip("CuPy not installed")

    # 将 x*x + y 转换为使用 NumPy 的 lambda 函数
    f = lambdify([[x, y]], x*x + y, 'numpy')
    # 对 CuPy 数组 [2.0, 1.0] 运行函数 f
    result = f(cupy.array([2.0, 1.0]))
    # 断言结果等于 5
    assert result == 5
    # 断言结果类型包含字符串 "cupy"
    assert "cupy" in str(type(result))


# 测试使用 JAX 的数组参数
def test_jax_array_arg():
    # 如果没有安装 JAX，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 将 x*x + y 转换为使用 JAX 的 lambda 函数
    f = lambdify([[x, y]], x*x + y, 'jax')
    # 对 JAX 数组 [2.0, 1.0] 运行函数 f
    result = f(jax.numpy.array([2.0, 1.0]))
    # 断言结果等于 5
    assert result == 5
    # 断言结果类型包含字符串 "jax"
    assert "jax" in str(type(result))


# 测试使用 JAX 的数组参数，但使用了 NumPy 函数
def test_jax_array_arg_using_numpy():
    # 如果没有安装 JAX，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 将 x*x + y 转换为使用 NumPy 的 lambda 函数
    f = lambdify([[x, y]], x*x + y, 'numpy')
    # 对 JAX 数组 [2.0, 1.0] 运行函数 f
    result = f(jax.numpy.array([2.0, 1.0]))
    # 断言结果等于 5
    assert result == 5
    # 断言结果类型包含字符串 "jax"
    assert "jax" in str(type(result))


# 测试使用 JAX 的 DotProduct 函数
def test_jax_dotproduct():
    # 如果没有安装 JAX，则跳过测试
    if not jax:
        skip("JAX not installed")

    # 创建 Matrix([x, y, z]) 对象 A
    A = Matrix([x, y, z])
    # 将 DotProduct(A, A) 转换为使用 JAX 的 lambda 函数
    f1 = lambdify([x, y, z], DotProduct(A, A), modules='jax')
    # 将 DotProduct(A, A.T) 转换为使用 JAX 的 lambda 函数
    f2 = lambdify([x, y, z], DotProduct(A, A.T), modules='jax')
    # 将 DotProduct(A.T, A) 转换为使用 JAX 的 lambda 函数
    f3 = lambdify([x, y, z], DotProduct(A.T, A), modules='jax')
    # 将 DotProduct(A, A.T) 转换为使用 JAX 的 lambda 函数
    f4 = lambdify([x, y, z], DotProduct(A, A.T), modules='jax')

    # 断言四个函数的结果都等于 JAX 数组 [14]
    assert f1(1, 2, 3) == \
        f2(1, 2, 3) == \
        f3(1, 2, 3) == \
        f4(1, 2, 3) == \
        jax.numpy.array([14])


# 定义一个不进行公共子表达式消除（CSE）的函数
def test_lambdify_cse(exprs):
    return (), exprs
    def dummy_cse(exprs):
        from sympy.simplify.cse_main import cse
        return cse(exprs, symbols=numbered_symbols(cls=Dummy))

从`sympy.simplify.cse_main`模块导入`cse`函数，对给定的表达式列表进行公共子表达式消除（Common Subexpression Elimination, CSE），使用`Dummy`类生成的符号作为变量。


    def minmem(exprs):
        from sympy.simplify.cse_main import cse_release_variables, cse
        return cse(exprs, postprocess=cse_release_variables)

从`sympy.simplify.cse_main`模块导入`cse_release_variables`和`cse`函数。对给定的表达式列表进行公共子表达式消除（CSE），并应用`cse_release_variables`后处理函数来释放变量。


    class Case:
        def __init__(self, *, args, exprs, num_args, requires_numpy=False):
            self.args = args
            self.exprs = exprs
            self.num_args = num_args
            subs_dict = dict(zip(self.args, self.num_args))
            self.ref = [e.subs(subs_dict).evalf() for e in exprs]
            self.requires_numpy = requires_numpy

定义了一个`Case`类，用于表示测试用例。初始化方法接受参数`args`（参数列表）、`exprs`（表达式列表）、`num_args`（数值参数列表）、`requires_numpy`（是否需要NumPy）。通过数值参数替换参数字典`subs_dict`，计算参考值`ref`，并将其数值化。


        def lambdify(self, *, cse):
            return lambdify(self.args, self.exprs, cse=cse)

定义了`lambdify`方法，用于将表达式列表转换为lambda函数，支持公共子表达式消除（CSE）通过`cse`参数控制。


        def assertAllClose(self, result, *, abstol=1e-15, reltol=1e-15):
            if self.requires_numpy:
                assert all(numpy.allclose(result[i], numpy.asarray(r, dtype=float),
                                          rtol=reltol, atol=abstol)
                           for i, r in enumerate(self.ref))
                return

            for i, r in enumerate(self.ref):
                abs_err = abs(result[i] - r)
                if r == 0:
                    assert abs_err < abstol
                else:
                    assert abs_err/abs(r) < reltol

定义了`assertAllClose`方法，用于断言结果和参考值的接近程度。如果需要NumPy，则使用NumPy的`allclose`函数进行比较，否则计算绝对误差和相对误差来进行断言。


    cases = [
        Case(
            args=(x, y, z),
            exprs=[
             x + y + z,
             x + y - z,
             2*x + 2*y - z,
             (x+y)**2 + (y+z)**2,
            ],
            num_args=(2., 3., 4.)
        ),
        Case(
            args=(x, y, z),
            exprs=[
            x + sympy.Heaviside(x),
            y + sympy.Heaviside(x),
            z + sympy.Heaviside(x, 1),
            z/sympy.Heaviside(x, 1)
            ],
            num_args=(0., 3., 4.)
        ),
        Case(
            args=(x, y, z),
            exprs=[
            x + sinc(y),
            y + sinc(y),
            z - sinc(y)
            ],
            num_args=(0.1, 0.2, 0.3)
        ),
        Case(
            args=(x, y, z),
            exprs=[
                Matrix([[x, x*y], [sin(z) + 4, x**z]]),
                x*y+sin(z)-x**z,
                Matrix([x*x, sin(z), x**z])
            ],
            num_args=(1.,2.,3.),
            requires_numpy=True
        ),
        Case(
            args=(x, y),
            exprs=[(x + y - 1)**2, x, x + y,
            (x + y)/(2*x + 1) + (x + y - 1)**2, (2*x + 1)**(x + y)],
            num_args=(1,2)
        )
    ]

定义了一个包含多个测试用例的列表`cases`，每个测试用例是一个`Case`类的实例，包括参数`args`、表达式列表`exprs`、数值参数`num_args`和是否需要NumPy`requires_numpy`。


    for case in cases:
        if not numpy and case.requires_numpy:
            continue
        for _cse in [False, True, minmem, no_op_cse, dummy_cse]:
            f = case.lambdify(cse=_cse)
            result = f(*case.num_args)
            case.assertAllClose(result)

对于每个测试用例`case`，如果当前环境不支持NumPy且测试用例要求NumPy，则跳过该测试用例。对于每个公共子表达式消除策略`_cse`，通过`lambdify`方法生成lambda函数`f`，并计算结果`result`，最后使用`assertAllClose`方法断言结果与参考值的接近程度。
# 测试函数，用于检查 GitHub 上的问题编号 25288 的问题
def test_issue_25288():
    # 使用 numbered_symbols 函数创建符号列表，其中每个符号是 Dummy 类的实例
    syms = numbered_symbols(cls=Dummy)
    # 使用 lambdify 函数创建一个函数 ok，将表达式 x**2 和 sin(x**2) 编译成可评估的 lambda 函数，
    # 使用 cse 函数对表达式进行公共子表达式消除
    ok = lambdify(x, [x**2, sin(x**2)], cse=lambda e: cse(e, symbols=syms))(2)
    # 断言 ok 的值
    assert ok


# 测试函数，用于检查废弃的 set 用法
def test_deprecated_set():
    # 使用 warns_deprecated_sympy 上下文管理器捕获关于 SymPy 库的废弃警告
    with warns_deprecated_sympy():
        # 使用 lambdify 函数创建一个函数，接受参数为集合 {x, y}，返回 x + y 的 lambda 函数
        lambdify({x, y}, x + y)


# 测试函数，用于检查 GitHub 上的问题编号 13881 的问题
def test_issue_13881():
    # 如果 numpy 模块未安装，则跳过测试
    if not numpy:
        skip("numpy not installed.")

    # 创建一个 3x1 的 MatrixSymbol 对象 X
    X = MatrixSymbol('X', 3, 1)

    # 使用 lambdify 函数创建一个 numpy 下的函数 f，接受 X 作为参数，返回 X^T * X 的 lambda 函数
    f = lambdify(X, X.T*X, 'numpy')
    # 断言 f 函数对输入数组 [1, 2, 3] 的计算结果为 14
    assert f(numpy.array([1, 2, 3])) == 14
    # 断言 f 函数对输入数组 [3, 2, 1] 的计算结果为 14
    assert f(numpy.array([3, 2, 1])) == 14

    # 使用 lambdify 函数创建一个 numpy 下的函数 f，接受 X 作为参数，返回 X*X^T 的 lambda 函数
    f = lambdify(X, X*X.T, 'numpy')
    # 断言 f 函数对输入数组 [1, 2, 3] 的计算结果为 14
    assert f(numpy.array([1, 2, 3])) == 14
    # 断言 f 函数对输入数组 [3, 2, 1] 的计算结果为 14
    assert f(numpy.array([3, 2, 1])) == 14

    # 使用 lambdify 函数创建一个 numpy 下的函数 f，接受 X 作为参数，返回 (X*X^T)*X 的 lambda 函数
    f = lambdify(X, (X*X.T)*X, 'numpy')
    # 创建输入数组 arr1 和期望的输出数组 arr2
    arr1 = numpy.array([[1], [2], [3]])
    arr2 = numpy.array([[14],[28],[42]])
    # 使用 numpy.array_equal 函数断言 f 函数对输入 arr1 的计算结果与 arr2 相等
    assert numpy.array_equal(f(arr1), arr2)


# 测试类，用于检查 GitHub 上的问题编号 23536 的问题
class test_23536_lambdify_cse_dummy:

    # 定义函数 f 和 g，接受 y 作为参数
    f = Function('x')(y)
    g = Function('w')(y)
    # 创建复杂表达式 expr
    expr = z + (f**4 + g**5)*(f**3 + (g*f)**3)
    # 对表达式进行展开
    expr = expr.expand()
    # 使用 lambdify 函数创建一个函数 eval_expr，接受 ((f, g), z) 作为参数，返回 expr 的 lambda 函数
    eval_expr = lambdify(((f, g), z), expr, cse=True)
    # 调用 eval_expr 函数，传入参数 (1.0, 2.0) 和 3.0，确保不会引发 NameError
    ans = eval_expr((1.0, 2.0), 3.0)
    # 断言 ans 的值为 300.0
    assert ans == 300.0


# LambdifyDocstringTestCase 类，用于测试 lambdify 的文档字符串大小限制
class LambdifyDocstringTestCase:

    # 类属性初始化
    SIGNATURE = None
    EXPR = None
    SRC = None

    def __init__(self, docstring_limit, expected_redacted):
        # 初始化实例属性
        self.docstring_limit = docstring_limit
        self.expected_redacted = expected_redacted

    @property
    def expected_expr(self):
        # 根据 expected_redacted 属性返回表达式或长度超限提示信息
        expr_redacted_msg = "EXPRESSION REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        return self.EXPR if not self.expected_redacted else expr_redacted_msg

    @property
    def expected_src(self):
        # 根据 expected_redacted 属性返回源代码或长度超限提示信息
        src_redacted_msg = "SOURCE CODE REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        return self.SRC if not self.expected_redacted else src_redacted_msg

    @property
    def expected_docstring(self):
        # 返回期望的完整文档字符串
        expected_docstring = (
            f'Created with lambdify. Signature:\n\n'
            f'func({self.SIGNATURE})\n\n'
            f'Expression:\n\n'
            f'{self.expected_expr}\n\n'
            f'Source code:\n\n'
            f'{self.expected_src}\n\n'
            f'Imported modules:\n\n'
        )
        return expected_docstring

    def __len__(self):
        # 返回文档字符串的长度
        return len(self.expected_docstring)

    def __repr__(self):
        # 返回实例的表示形式字符串
        return (
            f'{self.__class__.__name__}('
            f'docstring_limit={self.docstring_limit}, '
            f'expected_redacted={self.expected_redacted})'
        )


# 测试函数，用于检查 lambdify 的文档字符串大小限制，测试简单符号
def test_lambdify_docstring_size_limit_simple_symbol():

    # 定义 SimpleSymbolTestCase 类，继承自 LambdifyDocstringTestCase
    class SimpleSymbolTestCase(LambdifyDocstringTestCase):
        SIGNATURE = 'x'
        EXPR = 'x'
        SRC = (
            'def _lambdifygenerated(x):\n'
            '    return x\n'
        )

    # 创建符号 x
    x = symbols('x')
    # 定义测试用例元组，包含了不同的 SimpleSymbolTestCase 实例，每个实例用于测试不同的参数组合
    test_cases = (
        SimpleSymbolTestCase(docstring_limit=None, expected_redacted=False),  # 无限制的文档字符串长度，期望不进行文档字符串的隐藏
        SimpleSymbolTestCase(docstring_limit=100, expected_redacted=False),   # 文档字符串长度限制为100，期望不进行文档字符串的隐藏
        SimpleSymbolTestCase(docstring_limit=1, expected_redacted=False),     # 文档字符串长度限制为1，期望不进行文档字符串的隐藏
        SimpleSymbolTestCase(docstring_limit=0, expected_redacted=True),      # 文档字符串长度限制为0，期望进行文档字符串的隐藏
        SimpleSymbolTestCase(docstring_limit=-1, expected_redacted=True),     # 文档字符串长度限制为-1，期望进行文档字符串的隐藏
    )
    # 遍历测试用例元组，逐个执行测试
    for test_case in test_cases:
        # 调用 lambdify 函数，生成一个基于 sympy 的符号函数表达式
        lambdified_expr = lambdify(
            [x],                                # 使用符号 x 作为函数的输入变量
            x,                                  # 函数的输出表达式即为输入变量 x 自身
            'sympy',                            # 使用 sympy 来进行符号计算
            docstring_limit=test_case.docstring_limit,  # 将测试用例中的文档字符串长度限制参数传递给 lambdify 函数
        )
        # 断言：生成的 lambdified_expr 的文档字符串是否与测试用例中的期望文档字符串隐藏状态相符
        assert lambdified_expr.__doc__ == test_case.expected_docstring
def test_assoc_legendre_numerical_evaluation():
    # 设置数值容差
    tol = 1e-10
    
    # 计算整数参数的勒让德关联函数的数值结果
    sympy_result_integer = assoc_legendre(1, 1/2, 0.1).evalf()
    # 计算复数参数的勒让德关联函数的数值结果
    sympy_result_complex = assoc_legendre(2, 1, 3).evalf()
    # 使用mpmath计算整数参数的勒让德关联函数的数值结果
    mpmath_result_integer = -0.474572528387641
    # 定义一个复数，使用虚数单位 I 表示
    mpmath_result_complex = -25.45584412271571*I
    
    # 使用 assert 断言检查 sympy_result_integer 和 mpmath_result_integer 是否在指定的容差 tol 范围内相近
    assert all_close(sympy_result_integer, mpmath_result_integer, tol)
    # 使用 assert 断言检查 sympy_result_complex 和 mpmath_result_complex 是否在指定的容差 tol 范围内相近
    assert all_close(sympy_result_complex, mpmath_result_complex, tol)
```