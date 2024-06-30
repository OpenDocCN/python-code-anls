# `D:\src\scipysrc\sympy\sympy\printing\tests\test_mathml.py`

```
# 导入从SymPy库中导入的累积界限函数AccumBounds
from sympy.calculus.accumulationbounds import AccumBounds
# 导入从SymPy库中导入的求和函数Sum
from sympy.concrete.summations import Sum
# 导入从SymPy库中导入的基本数学对象Basic
from sympy.core.basic import Basic
# 导入从SymPy库中导入的元组容器Tuple
from sympy.core.containers import Tuple
# 导入从SymPy库中导入的导数、lambda函数、求导函数diff、函数对象Function
from sympy.core.function import Derivative, Lambda, diff, Function
# 导入从SymPy库中导入的各种数值对象，包括无穷大、圆周率、自然常数e等
from sympy.core.numbers import (zoo, Float, Integer, I, oo, pi, E,
    Rational)
# 导入从SymPy库中导入的比较运算符Less than、Greater than、Not equal、Equal
from sympy.core.relational import Lt, Ge, Ne, Eq
# 导入从SymPy库中导入的单例对象S
from sympy.core.singleton import S
# 导入从SymPy库中导入的符号对象symbols、符号对象Symbol
from sympy.core.symbol import symbols, Symbol
# 导入从SymPy库中导入的符号化函数sympify
from sympy.core.sympify import sympify
# 导入从SymPy库中导入的组合数学函数，如双阶乘、二项式系数、阶乘等
from sympy.functions.combinatorial.factorials import (factorial2,
    binomial, factorial)
# 导入从SymPy库中导入的组合数学函数，如卢卡斯数、贝尔数、卡特兰数等
from sympy.functions.combinatorial.numbers import (lucas, bell,
    catalan, euler, tribonacci, fibonacci, bernoulli, primenu, primeomega,
    totient, reduced_totient)
# 导入从SymPy库中导入的复数函数，如实部、虚部、共轭复数、绝对值等
from sympy.functions.elementary.complexes import re, im, conjugate, Abs
# 导入从SymPy库中导入的指数函数、Lambert W函数、对数函数等
from sympy.functions.elementary.exponential import exp, LambertW, log
# 导入从SymPy库中导入的双曲函数、反双曲函数、反双曲余切函数等
from sympy.functions.elementary.hyperbolic import (tanh, acoth, atanh,
    coth, asinh, acsch, asech, acosh, csch, sinh, cosh, sech)
# 导入从SymPy库中导入的整数函数、取整函数
from sympy.functions.elementary.integers import ceiling, floor
# 导入从SymPy库中导入的最大值、最小值函数
from sympy.functions.elementary.miscellaneous import Max, Min
# 导入从SymPy库中导入的三角函数、反三角函数
from sympy.functions.elementary.trigonometric import (csc, sec, tan,
    atan, sin, asec, cot, cos, acot, acsc, asin, acos)
# 导入从SymPy库中导入的海维赛德阶跃函数
from sympy.functions.special.delta_functions import Heaviside
# 导入从SymPy库中导入的椭圆积分函数
from sympy.functions.special.elliptic_integrals import (elliptic_pi,
    elliptic_f, elliptic_k, elliptic_e)
# 导入从SymPy库中导入的弗雷内尔余弦积分函数、弗雷内尔正弦积分函数、指数积分函数等
from sympy.functions.special.error_functions import (fresnelc,
    fresnels, Ei, expint)
# 导入从SymPy库中导入的伽玛函数、上不完全伽玛函数、下不完全伽玛函数等
from sympy.functions.special.gamma_functions import (gamma, uppergamma,
    lowergamma)
# 导入从SymPy库中导入的马修函数、马修函数的导数等
from sympy.functions.special.mathieu_functions import (mathieusprime,
    mathieus, mathieucprime, mathieuc)
# 导入从SymPy库中导入的多项式函数，如雅可比多项式、切比雪夫多项式等
from sympy.functions.special.polynomials import (jacobi, chebyshevu,
    chebyshevt, hermite, assoc_legendre, gegenbauer, assoc_laguerre,
    legendre, laguerre)
# 导入从SymPy库中导入的奇异函数
from sympy.functions.special.singularity_functions import SingularityFunction
# 导入从SymPy库中导入的多重对数函数、斯蒂尔切斯常数函数、勒让函数等
from sympy.functions.special.zeta_functions import (polylog, stieltjes,
    lerchphi, dirichlet_eta, zeta)
# 导入从SymPy库中导入的积分函数Integral
from sympy.integrals.integrals import Integral
# 导入从SymPy库中导入的布尔代数运算函数
from sympy.logic.boolalg import (Xor, Or, false, true, And, Equivalent,
    Implies, Not)
# 导入从SymPy库中导入的矩阵对象Matrix
from sympy.matrices.dense import Matrix
# 导入从SymPy库中导入的行列式对象Determinant
from sympy.matrices.expressions.determinant import Determinant
# 导入从SymPy库中导入的矩阵符号对象MatrixSymbol
from sympy.matrices.expressions.matexpr import MatrixSymbol
# 导入从SymPy库中导入的量子力学相关对象，如复数空间、Fock空间、希尔伯特空间等
from sympy.physics.quantum import (ComplexSpace, FockSpace, hbar,
    HilbertSpace, Dagger)
# 导入从SymPy库中导入的MathML打印相关对象，如数学表示、数学内容、数学ML打印器等
from sympy.printing.mathml import (MathMLPresentationPrinter,
    MathMLPrinter, MathMLContentPrinter, mathml)
# 导入从SymPy库中导入的极限对象Limit
from sympy.series.limits import Limit
# 导入从SymPy库中导入的集合包含对象Contains
from sympy.sets.contains import Contains
# 导入从SymPy库中导入的集合对象，如区间、并集、对称差等
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Interval, Union, SymmetricDifference,
    Complement, FiniteSet, Intersection, ProductSet)
# 导入从SymPy库中导入的随机变量符号对象RandomSymbol
from sympy.stats.rv import RandomSymbol
# 导入从SymPy库中导入的索引基对象IndexedBase
from sympy.tensor.indexed import IndexedBase
# 导入从SymPy库中导入的向量分析相关对象，如散度、梯度、旋度等
from sympy.vector import (Divergence, CoordSys3D, Cross, Curl, Dot,
    Laplacian, Gradient)
from sympy.testing.pytest import raises
# 导入需要的测试工具，raises 用于测试异常情况

x, y, z, a, b, c, d, e, n = symbols('x:z a:e n')
# 定义符号变量 x, y, z, a, b, c, d, e, n

mp = MathMLContentPrinter()
# 创建 MathMLContentPrinter 的实例对象 mp

mpp = MathMLPresentationPrinter()
# 创建 MathMLPresentationPrinter 的实例对象 mpp


def test_mathml_printer():
    # 定义测试函数 test_mathml_printer
    m = MathMLPrinter()
    # 创建 MathMLPrinter 的实例对象 m
    assert m.doprint(1+x) == mp.doprint(1+x)
    # 断言 m.doprint(1+x) 的输出与 mp.doprint(1+x) 相等


def test_content_printmethod():
    # 定义测试函数 test_content_printmethod
    assert mp.doprint(1 + x) == '<apply><plus/><ci>x</ci><cn>1</cn></apply>'
    # 断言 mp.doprint(1 + x) 的输出符合预期的 MathML 格式


def test_content_mathml_core():
    # 定义测试函数 test_content_mathml_core
    mml_1 = mp._print(1 + x)
    # 使用 mp._print 方法生成 MathML 表示 mml_1
    assert mml_1.nodeName == 'apply'
    # 断言 mml_1 的节点名为 'apply'
    nodes = mml_1.childNodes
    # 获取 mml_1 的子节点列表
    assert len(nodes) == 3
    # 断言 mml_1 的子节点数量为 3
    assert nodes[0].nodeName == 'plus'
    # 断言 mml_1 的第一个子节点名为 'plus'
    assert nodes[0].hasChildNodes() is False
    # 断言 mml_1 的第一个子节点没有子节点
    assert nodes[0].nodeValue is None
    # 断言 mml_1 的第一个子节点的节点值为 None
    assert nodes[1].nodeName in ['cn', 'ci']
    # 断言 mml_1 的第二个子节点名为 'cn' 或 'ci'
    if nodes[1].nodeName == 'cn':
        assert nodes[1].childNodes[0].nodeValue == '1'
        assert nodes[2].childNodes[0].nodeValue == 'x'
    else:
        assert nodes[1].childNodes[0].nodeValue == 'x'
        assert nodes[2].childNodes[0].nodeValue == '1'

    mml_2 = mp._print(x**2)
    # 使用 mp._print 方法生成 MathML 表示 mml_2
    assert mml_2.nodeName == 'apply'
    # 断言 mml_2 的节点名为 'apply'
    nodes = mml_2.childNodes
    # 获取 mml_2 的子节点列表
    assert nodes[1].childNodes[0].nodeValue == 'x'
    # 断言 mml_2 的第二个子节点的节点值为 'x'
    assert nodes[2].childNodes[0].nodeValue == '2'
    # 断言 mml_2 的第三个子节点的节点值为 '2'

    mml_3 = mp._print(2*x)
    # 使用 mp._print 方法生成 MathML 表示 mml_3
    assert mml_3.nodeName == 'apply'
    # 断言 mml_3 的节点名为 'apply'
    nodes = mml_3.childNodes
    # 获取 mml_3 的子节点列表
    assert nodes[0].nodeName == 'times'
    # 断言 mml_3 的第一个子节点名为 'times'
    assert nodes[1].childNodes[0].nodeValue == '2'
    # 断言 mml_3 的第二个子节点的节点值为 '2'
    assert nodes[2].childNodes[0].nodeValue == 'x'
    # 断言 mml_3 的第三个子节点的节点值为 'x'

    mml = mp._print(Float(1.0, 2)*x)
    # 使用 mp._print 方法生成 MathML 表示 mml
    assert mml.nodeName == 'apply'
    # 断言 mml 的节点名为 'apply'
    nodes = mml.childNodes
    # 获取 mml 的子节点列表
    assert nodes[0].nodeName == 'times'
    # 断言 mml 的第一个子节点名为 'times'
    assert nodes[1].childNodes[0].nodeValue == '1.0'
    # 断言 mml 的第二个子节点的节点值为 '1.0'
    assert nodes[2].childNodes[0].nodeValue == 'x'
    # 断言 mml 的第三个子节点的节点值为 'x'


def test_content_mathml_functions():
    # 定义测试函数 test_content_mathml_functions
    mml_1 = mp._print(sin(x))
    # 使用 mp._print 方法生成 MathML 表示 mml_1
    assert mml_1.nodeName == 'apply'
    # 断言 mml_1 的节点名为 'apply'
    assert mml_1.childNodes[0].nodeName == 'sin'
    # 断言 mml_1 的第一个子节点名为 'sin'
    assert mml_1.childNodes[1].nodeName == 'ci'
    # 断言 mml_1 的第二个子节点名为 'ci'

    mml_2 = mp._print(diff(sin(x), x, evaluate=False))
    # 使用 mp._print 方法生成 MathML 表示 mml_2
    assert mml_2.nodeName == 'apply'
    # 断言 mml_2 的节点名为 'apply'
    assert mml_2.childNodes[0].nodeName == 'diff'
    # 断言 mml_2 的第一个子节点名为 'diff'
    assert mml_2.childNodes[1].nodeName == 'bvar'
    # 断言 mml_2 的第二个子节点名为 'bvar'
    assert mml_2.childNodes[1].childNodes[0].nodeName == 'ci'
    # 断言 mml_2 的 bvar 子节点的第一个子节点名为 'ci'

    mml_3 = mp._print(diff(cos(x*y), x, evaluate=False))
    # 使用 mp._print 方法生成 MathML 表示 mml_3
    assert mml_3.nodeName == 'apply'
    # 断言 mml_3 的节点名为 'apply'
    assert mml_3.childNodes[0].nodeName == 'partialdiff'
    # 断言 mml_3 的第一个子节点名为 'partialdiff'
    assert mml_3.childNodes[1].nodeName == 'bvar'
    # 断言 mml_3 的第二个子节点名为 'bvar'
    assert mml_3.childNodes[1].childNodes[0].nodeName == 'ci'
    # 断言 mml_3 的 bvar 子节点的第一个子节点名为 'ci'

    mml_4 = mp._print(Lambda((x, y), x * y))
    # 使用 mp._print 方法生成 MathML 表示 mml_4
    assert mml_4.nodeName == 'lambda'
    # 断言 mml_4 的节点名为 'lambda'
    assert mml_4.childNodes[0].nodeName == 'bvar'
    # 断言 mml_4 的第一个子节点名为 'bvar'
    assert mml_4.childNodes[0].childNodes[0].nodeName == 'ci'
    # 断言 mml_4 的 bvar 子节点的第一个子节点名为 'ci'
    assert mml_4.childNodes[1].nodeName == 'bvar'
    # 断言 mml_4 的第二个子节点名为 'bvar'
    assert mml_4.childNodes[1].childNodes[0].nodeName == 'ci'
    # 断言 mml_4 的 bvar 子节点的第二个子节点名为 'ci'
    assert mml_4.childNodes[2].nodeName == 'apply'
    # 断言 mml_4 的第三个子节点名为 'apply'


def test_content_mathml_limits():
    # 定义测试函数 test_content_mathml_limits
    # XXX No unevaluated limits
    # 没有未评估的限制条件
    lim_fun = sin(x)/x
    # 创建 sin(x)/x 的表达式
    # 使用 SymPy 中的 `_print` 方法将 lim_fun 关于 x 趋向于 0 的极限表达式转换为 MathML 格式
    mml_1 = mp._print(Limit(lim_fun, x, 0))
    # 断言确保 MathML 格式中的第一个子节点是 'limit'
    assert mml_1.childNodes[0].nodeName == 'limit'
    # 断言确保 MathML 格式中的第二个子节点是 'bvar'
    assert mml_1.childNodes[1].nodeName == 'bvar'
    # 断言确保 MathML 格式中的第三个子节点是 'lowlimit'
    assert mml_1.childNodes[2].nodeName == 'lowlimit'
    # 断言确保 MathML 格式中的第四个子节点与 lim_fun 的打印形式一致
    assert mml_1.childNodes[3].toxml() == mp._print(lim_fun).toxml()
def test_content_mathml_integrals():
    # 设置被积函数为变量 x
    integrand = x
    # 生成积分的 MathML 表示
    mml_1 = mp._print(Integral(integrand, (x, 0, 1)))
    # 断言节点的名称是否为 'int'
    assert mml_1.childNodes[0].nodeName == 'int'
    # 断言节点的名称是否为 'bvar'
    assert mml_1.childNodes[1].nodeName == 'bvar'
    # 断言节点的名称是否为 'lowlimit'
    assert mml_1.childNodes[2].nodeName == 'lowlimit'
    # 断言节点的名称是否为 'uplimit'
    assert mml_1.childNodes[3].nodeName == 'uplimit'
    # 断言积分表达式与被积函数的 MathML 表示是否相等
    assert mml_1.childNodes[4].toxml() == mp._print(integrand).toxml()


def test_content_mathml_matrices():
    # 创建矩阵 A
    A = Matrix([1, 2, 3])
    # 创建矩阵 B
    B = Matrix([[0, 5, 4], [2, 3, 1], [9, 7, 9]])
    # 生成矩阵 A 的 MathML 表示
    mll_1 = mp._print(A)
    # 断言第一个 matrixrow 节点的名称为 'matrixrow'
    assert mll_1.childNodes[0].nodeName == 'matrixrow'
    # 断言第一个 matrixrow 节点的第一个子节点名称为 'cn'，且其值为 '1'
    assert mll_1.childNodes[0].childNodes[0].nodeName == 'cn'
    assert mll_1.childNodes[0].childNodes[0].childNodes[0].nodeValue == '1'
    # 断言第二个 matrixrow 节点的名称为 'matrixrow'
    assert mll_1.childNodes[1].nodeName == 'matrixrow'
    # 断言第二个 matrixrow 节点的第一个子节点名称为 'cn'，且其值为 '2'
    assert mll_1.childNodes[1].childNodes[0].nodeName == 'cn'
    assert mll_1.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    # 断言第三个 matrixrow 节点的名称为 'matrixrow'
    assert mll_1.childNodes[2].nodeName == 'matrixrow'
    # 断言第三个 matrixrow 节点的第一个子节点名称为 'cn'，且其值为 '3'
    assert mll_1.childNodes[2].childNodes[0].nodeName == 'cn'
    assert mll_1.childNodes[2].childNodes[0].childNodes[0].nodeValue == '3'
    # 生成矩阵 B 的 MathML 表示
    mll_2 = mp._print(B)
    # 断言第一个 matrixrow 节点的名称为 'matrixrow'
    assert mll_2.childNodes[0].nodeName == 'matrixrow'
    # 断言第一个 matrixrow 节点的第一个子节点名称为 'cn'，且其值为 '0'
    assert mll_2.childNodes[0].childNodes[0].nodeName == 'cn'
    assert mll_2.childNodes[0].childNodes[0].childNodes[0].nodeValue == '0'
    # 断言第一个 matrixrow 节点的第二个子节点名称为 'cn'，且其值为 '5'
    assert mll_2.childNodes[0].childNodes[1].nodeName == 'cn'
    assert mll_2.childNodes[0].childNodes[1].childNodes[0].nodeValue == '5'
    # 断言第一个 matrixrow 节点的第三个子节点名称为 'cn'，且其值为 '4'
    assert mll_2.childNodes[0].childNodes[2].nodeName == 'cn'
    assert mll_2.childNodes[0].childNodes[2].childNodes[0].nodeValue == '4'
    # 断言第二个 matrixrow 节点的名称为 'matrixrow'
    assert mll_2.childNodes[1].nodeName == 'matrixrow'
    # 断言第二个 matrixrow 节点的第一个子节点名称为 'cn'，且其值为 '2'
    assert mll_2.childNodes[1].childNodes[0].nodeName == 'cn'
    assert mll_2.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    # 断言第二个 matrixrow 节点的第二个子节点名称为 'cn'，且其值为 '3'
    assert mll_2.childNodes[1].childNodes[1].nodeName == 'cn'
    assert mll_2.childNodes[1].childNodes[1].childNodes[0].nodeValue == '3'
    # 断言第二个 matrixrow 节点的第三个子节点名称为 'cn'，且其值为 '1'
    assert mll_2.childNodes[1].childNodes[2].nodeName == 'cn'
    assert mll_2.childNodes[1].childNodes[2].childNodes[0].nodeValue == '1'
    # 断言第三个 matrixrow 节点的名称为 'matrixrow'
    assert mll_2.childNodes[2].nodeName == 'matrixrow'
    # 断言第三个 matrixrow 节点的第一个子节点名称为 'cn'，且其值为 '9'
    assert mll_2.childNodes[2].childNodes[0].nodeName == 'cn'
    assert mll_2.childNodes[2].childNodes[0].childNodes[0].nodeValue == '9'
    # 断言第三个 matrixrow 节点的第二个子节点名称为 'cn'，且其值为 '7'
    assert mll_2.childNodes[2].childNodes[1].nodeName == 'cn'
    assert mll_2.childNodes[2].childNodes[1].childNodes[0].nodeValue == '7'
    # 断言第三个 matrixrow 节点的第三个子节点名称为 'cn'，且其值为 '9'
    assert mll_2.childNodes[2].childNodes[2].nodeName == 'cn'
    assert mll_2.childNodes[2].childNodes[2].childNodes[0].nodeValue == '9'


def test_content_mathml_sums():
    # 设置被求和函数为变量 x
    summand = x
    # 生成求和的 MathML 表示
    mml_1 = mp._print(Sum(summand, (x, 1, 10)))
    # 断言节点的名称是否为 'sum'
    assert mml_1.childNodes[0].nodeName == 'sum'
    # 断言节点的名称是否为 'bvar'
    assert mml_1.childNodes[1].nodeName == 'bvar'
    # 断言节点的名称是否为 'lowlimit'
    assert mml_1.childNodes[2].nodeName == 'lowlimit'
    # 断言节点的名称是否为 'uplimit'
    assert mml_1.childNodes[3].nodeName == 'uplimit'
    # 断言求和表达式与被求和函数的 MathML 表示是否相等
    assert mml_1.childNodes[4].toxml() == mp._print(summand).toxml()


def test_content_mathml_tuples():
    # 生成包含数字 2 的元组的 MathML 表示
    mml_1 = mp._print([2])
    # 断言 mml_1 的节点名称为 'list'
    assert mml_1.nodeName == 'list'
    # 断言 mml_1 的第一个子节点的节点名称为 'cn'
    assert mml_1.childNodes[0].nodeName == 'cn'
    # 断言 mml_1 的子节点数量为 1
    assert len(mml_1.childNodes) == 1
    
    # 使用 mp._print 方法将列表 [2, Integer(1)] 转换为 MathML 表示
    mml_2 = mp._print([2, Integer(1)])
    # 断言 mml_2 的节点名称为 'list'
    assert mml_2.nodeName == 'list'
    # 断言 mml_2 的第一个子节点的节点名称为 'cn'
    assert mml_2.childNodes[0].nodeName == 'cn'
    # 断言 mml_2 的第二个子节点的节点名称为 'cn'
    assert mml_2.childNodes[1].nodeName == 'cn'
    # 断言 mml_2 的子节点数量为 2
    assert len(mml_2.childNodes) == 2
# 测试生成 MathML 表示的表达式 x**5 - x**4 + x
def test_content_mathml_add():
    # 生成 MathML 表示
    mml = mp._print(x**5 - x**4 + x)
    # 断言节点名称为 'plus'
    assert mml.childNodes[0].nodeName == 'plus'
    # 断言子节点的子节点的名称为 'minus'
    assert mml.childNodes[1].childNodes[0].nodeName == 'minus'
    # 断言子节点的子节点的名称为 'apply'
    assert mml.childNodes[1].childNodes[1].nodeName == 'apply'


# 测试生成 MathML 表示的有理数 Rational(1, 1) 和 Rational(2, 5)
def test_content_mathml_Rational():
    # 生成 MathML 表示
    mml_1 = mp._print(Rational(1, 1))
    """should just return a number"""
    # 断言节点名称为 'cn'
    assert mml_1.nodeName == 'cn'

    # 生成 MathML 表示
    mml_2 = mp._print(Rational(2, 5))
    # 断言子节点的节点名称为 'divide'
    assert mml_2.childNodes[0].nodeName == 'divide'


# 测试生成 MathML 表示的常数 I, E, oo, pi 和一些特殊常数
def test_content_mathml_constants():
    # 生成 MathML 表示
    mml = mp._print(I)
    # 断言节点名称为 'imaginaryi'
    assert mml.nodeName == 'imaginaryi'

    # 生成 MathML 表示
    mml = mp._print(E)
    # 断言节点名称为 'exponentiale'
    assert mml.nodeName == 'exponentiale'

    # 生成 MathML 表示
    mml = mp._print(oo)
    # 断言节点名称为 'infinity'
    assert mml.nodeName == 'infinity'

    # 生成 MathML 表示
    mml = mp._print(pi)
    # 断言节点名称为 'pi'
    assert mml.nodeName == 'pi'

    # 断言特殊常数的 MathML 表示
    assert mathml(hbar) == '<hbar/>'
    assert mathml(S.TribonacciConstant) == '<tribonacciconstant/>'
    assert mathml(S.GoldenRatio) == '<cn>&#966;</cn>'
    mml = mathml(S.EulerGamma)
    assert mml == '<eulergamma/>'

    mml = mathml(S.EmptySet)
    assert mml == '<emptyset/>'

    mml = mathml(S.true)
    assert mml == '<true/>'

    mml = mathml(S.false)
    assert mml == '<false/>'

    mml = mathml(S.NaN)
    assert mml == '<notanumber/>'


# 测试生成 MathML 表示的三角函数和反三角函数
def test_content_mathml_trig():
    # 生成 MathML 表示
    mml = mp._print(sin(x))
    # 断言节点名称为 'sin'
    assert mml.childNodes[0].nodeName == 'sin'

    # 生成 MathML 表示
    mml = mp._print(cos(x))
    # 断言节点名称为 'cos'
    assert mml.childNodes[0].nodeName == 'cos'

    # 生成 MathML 表示
    mml = mp._print(tan(x))
    # 断言节点名称为 'tan'
    assert mml.childNodes[0].nodeName == 'tan'

    # 生成 MathML 表示
    mml = mp._print(cot(x))
    # 断言节点名称为 'cot'
    assert mml.childNodes[0].nodeName == 'cot'

    # 生成 MathML 表示
    mml = mp._print(csc(x))
    # 断言节点名称为 'csc'
    assert mml.childNodes[0].nodeName == 'csc'

    # 生成 MathML 表示
    mml = mp._print(sec(x))
    # 断言节点名称为 'sec'
    assert mml.childNodes[0].nodeName == 'sec'

    # 生成 MathML 表示
    mml = mp._print(asin(x))
    # 断言节点名称为 'arcsin'
    assert mml.childNodes[0].nodeName == 'arcsin'

    # 生成 MathML 表示
    mml = mp._print(acos(x))
    # 断言节点名称为 'arccos'
    assert mml.childNodes[0].nodeName == 'arccos'

    # 生成 MathML 表示
    mml = mp._print(atan(x))
    # 断言节点名称为 'arctan'
    assert mml.childNodes[0].nodeName == 'arctan'

    # 生成 MathML 表示
    mml = mp._print(acot(x))
    # 断言节点名称为 'arccot'
    assert mml.childNodes[0].nodeName == 'arccot'

    # 生成 MathML 表示
    mml = mp._print(acsc(x))
    # 断言节点名称为 'arccsc'
    assert mml.childNodes[0].nodeName == 'arccsc'

    # 生成 MathML 表示
    mml = mp._print(asec(x))
    # 断言节点名称为 'arcsec'
    assert mml.childNodes[0].nodeName == 'arcsec'

    # 生成 MathML 表示
    mml = mp._print(sinh(x))
    # 断言节点名称为 'sinh'
    assert mml.childNodes[0].nodeName == 'sinh'

    # 生成 MathML 表示
    mml = mp._print(cosh(x))
    # 断言节点名称为 'cosh'
    assert mml.childNodes[0].nodeName == 'cosh'

    # 生成 MathML 表示
    mml = mp._print(tanh(x))
    # 断言节点名称为 'tanh'
    assert mml.childNodes[0].nodeName == 'tanh'

    # 生成 MathML 表示
    mml = mp._print(coth(x))
    # 断言节点名称为 'coth'
    assert mml.childNodes[0].nodeName == 'coth'

    # 生成 MathML 表示
    mml = mp._print(csch(x))
    # 断言节点名称为 'csch'
    assert mml.childNodes[0].nodeName == 'csch'

    # 生成 MathML 表示
    mml = mp._print(sech(x))
    # 断言节点名称为 'sech'
    assert mml.childNodes[0].nodeName == 'sech'

    # 生成 MathML 表示
    mml = mp._print(asinh(x))
    # 断言节点名称为 'arcsinh'
    assert mml.childNodes[0].nodeName == 'arcsinh'

    # 生成 MathML 表示
    mml = mp._print(atanh(x))
    # 断言节点名称为 'arctanh'
    assert mml.childNodes[0].nodeName == 'arctanh'

    # 生成 MathML 表示
    mml = mp._print(acosh(x))
    # 断言节点名称为 'arccosh'
    assert mml.childNodes[0].nodeName == 'arccosh'

    # 生成 MathML 表示
    mml = mp._print(acoth(x))
    # 断言节点名称为 'arccoth'
    assert mml.childNodes[0].nodeName == 'arccoth'
    # 调用 mp 模块的 _print 方法，将 acsch(x) 的结果转换为 MathML 格式的字符串
    mml = mp._print(acsch(x))
    # 使用断言验证 MathML 字符串的第一个子节点的节点名称是否为 'arccsch'
    assert mml.childNodes[0].nodeName == 'arccsch'

    # 调用 mp 模块的 _print 方法，将 asech(x) 的结果转换为 MathML 格式的字符串
    mml = mp._print(asech(x))
    # 使用断言验证 MathML 字符串的第一个子节点的节点名称是否为 'arcsech'
    assert mml.childNodes[0].nodeName == 'arcsech'
# 定义测试函数，用于测试生成 MathML 表达式中的关系运算符
def test_content_mathml_relational():
    # 生成 x = 1 的 MathML 表示
    mml_1 = mp._print(Eq(x, 1))
    # 断言 MathML 根节点是 'apply'
    assert mml_1.nodeName == 'apply'
    # 断言第一个子节点是 'eq'
    assert mml_1.childNodes[0].nodeName == 'eq'
    # 断言第二个子节点是 'ci'，表示变量 x
    assert mml_1.childNodes[1].nodeName == 'ci'
    # 断言变量节点的值是 'x'
    assert mml_1.childNodes[1].childNodes[0].nodeValue == 'x'
    # 断言第三个子节点是 'cn'，表示常数 1
    assert mml_1.childNodes[2].nodeName == 'cn'
    # 断言常数节点的值是 '1'
    assert mml_1.childNodes[2].childNodes[0].nodeValue == '1'

    # 生成 1 ≠ x 的 MathML 表示
    mml_2 = mp._print(Ne(1, x))
    assert mml_2.nodeName == 'apply'
    assert mml_2.childNodes[0].nodeName == 'neq'
    assert mml_2.childNodes[1].nodeName == 'cn'
    assert mml_2.childNodes[1].childNodes[0].nodeValue == '1'
    assert mml_2.childNodes[2].nodeName == 'ci'
    assert mml_2.childNodes[2].childNodes[0].nodeValue == 'x'

    # 生成 1 ≥ x 的 MathML 表示
    mml_3 = mp._print(Ge(1, x))
    assert mml_3.nodeName == 'apply'
    assert mml_3.childNodes[0].nodeName == 'geq'
    assert mml_3.childNodes[1].nodeName == 'cn'
    assert mml_3.childNodes[1].childNodes[0].nodeValue == '1'
    assert mml_3.childNodes[2].nodeName == 'ci'
    assert mml_3.childNodes[2].childNodes[0].nodeValue == 'x'

    # 生成 1 < x 的 MathML 表示
    mml_4 = mp._print(Lt(1, x))
    assert mml_4.nodeName == 'apply'
    assert mml_4.childNodes[0].nodeName == 'lt'
    assert mml_4.childNodes[1].nodeName == 'cn'
    assert mml_4.childNodes[1].childNodes[0].nodeValue == '1'
    assert mml_4.childNodes[2].nodeName == 'ci'
    assert mml_4.childNodes[2].childNodes[0].nodeValue == 'x'


# 定义测试函数，用于测试生成 MathML 表达式中的符号
def test_content_symbol():
    # 生成单一变量 x 的 MathML 表示
    mml = mp._print(x)
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeValue == 'x'
    del mml  # 删除变量以释放内存

    # 生成复合符号 "x^2" 的 MathML 表示
    mml = mp._print(Symbol("x^2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    del mml  # 删除变量以释放内存

    # 生成复合符号 "x__2" 的 MathML 表示
    mml = mp._print(Symbol("x__2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    del mml  # 删除变量以释放内存

    # 生成复合符号 "x_2" 的 MathML 表示
    mml = mp._print(Symbol("x_2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msub'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    del mml  # 删除变量以释放内存

    # 生成复合符号 "x^3_2" 的 MathML 表示，不完整的测试
    mml = mp._print(Symbol("x^3_2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msubsup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    # 确认第一个子节点的第一个子节点的值是否为 'x'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    # 确认第一个子节点的第二个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    # 确认第一个子节点的第二个子节点的第一个子节点的值是否为 '2'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    # 确认第一个子节点的第三个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[2].nodeName == 'mml:mi'
    # 确认第一个子节点的第三个子节点的第一个子节点的值是否为 '3'
    assert mml.childNodes[0].childNodes[2].childNodes[0].nodeValue == '3'
    # 删除变量 mml
    del mml

    # 生成符号 "x__3_2" 的数学表达式
    mml = mp._print(Symbol("x__3_2"))
    # 确认 mml 的节点名称是否为 'ci'
    assert mml.nodeName == 'ci'
    # 确认 mml 的第一个子节点的节点名称是否为 'mml:msubsup'
    assert mml.childNodes[0].nodeName == 'mml:msubsup'
    # 确认 mml 的第一个子节点的第一个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    # 确认 mml 的第一个子节点的第一个子节点的第一个子节点的值是否为 'x'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    # 确认 mml 的第一个子节点的第二个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    # 确认 mml 的第一个子节点的第二个子节点的第一个子节点的值是否为 '2'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    # 确认 mml 的第一个子节点的第三个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[2].nodeName == 'mml:mi'
    # 确认 mml 的第一个子节点的第三个子节点的第一个子节点的值是否为 '3'
    assert mml.childNodes[0].childNodes[2].childNodes[0].nodeValue == '3'
    # 删除变量 mml
    del mml

    # 生成符号 "x_2_a" 的数学表达式
    mml = mp._print(Symbol("x_2_a"))
    # 确认 mml 的节点名称是否为 'ci'
    assert mml.nodeName == 'ci'
    # 确认 mml 的第一个子节点的节点名称是否为 'mml:msub'
    assert mml.childNodes[0].nodeName == 'mml:msub'
    # 确认 mml 的第一个子节点的第一个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    # 确认 mml 的第一个子节点的第一个子节点的第一个子节点的值是否为 'x'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    # 确认 mml 的第一个子节点的第二个子节点的节点名称是否为 'mml:mrow'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mrow'
    # 确认 mml 的第一个子节点的第二个子节点的第一个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeName == 'mml:mi'
    # 确认 mml 的第一个子节点的第二个子节点的第一个子节点的第一个子节点的值是否为 '2'
    assert mml.childNodes[0].childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    # 确认 mml 的第一个子节点的第二个子节点的第二个子节点的节点名称是否为 'mml:mo'
    assert mml.childNodes[0].childNodes[1].childNodes[1].nodeName == 'mml:mo'
    # 确认 mml 的第一个子节点的第二个子节点的第二个子节点的第一个子节点的值是否为 ' '
    assert mml.childNodes[0].childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    # 确认 mml 的第一个子节点的第二个子节点的第三个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[2].nodeName == 'mml:mi'
    # 确认 mml 的第一个子节点的第二个子节点的第三个子节点的第一个子节点的值是否为 'a'
    assert mml.childNodes[0].childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    # 删除变量 mml
    del mml

    # 生成符号 "x^2^a" 的数学表达式
    mml = mp._print(Symbol("x^2^a"))
    # 确认 mml 的节点名称是否为 'ci'
    assert mml.nodeName == 'ci'
    # 确认 mml 的第一个子节点的节点名称是否为 'mml:msup'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    # 确认 mml 的第一个子节点的第一个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    # 确认 mml 的第一个子节点的第一个子节点的第一个子节点的值是否为 'x'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    # 确认 mml 的第一个子节点的第二个子节点的节点名称是否为 'mml:mrow'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mrow'
    # 确认 mml 的第一个子节点的第二个子节点的第一个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeName == 'mml:mi'
    # 确认 mml 的第一个子节点的第二个子节点的第一个子节点的第一个子节点的值是否为 '2'
    assert mml.childNodes[0].childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    # 确认 mml 的第一个子节点的第二个子节点的第二个子节点的节点名称是否为 'mml:mo'
    assert mml.childNodes[0].childNodes[1].childNodes[1].nodeName == 'mml:mo'
    # 确认 mml 的第一个子节点的第二个子节点的第二个子节点的第一个子节点的值是否为 ' '
    assert mml.childNodes[0].childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    # 确认 mml 的第一个子节点的第二个子节点的第三个子节点的节点名称是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[2].nodeName == 'mml:mi'
    # 确认 mml 的第一个子节点的第二个子节点的第三个子节点的第一个子节点的值是否为 'a'
    assert mml.childNodes[0].childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    # 删除变量 mml
    del mml

    # 生成符号 "x__2__a" 的数学表达式
    mml = mp._print(Symbol("x__2__a"))
    # 确认 mml 的节点名称是否为 'ci'
    assert mml.nodeName == 'ci'
    # 确认 mml 的第一个子节点的节点名称是否为 'mml:msup'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    # 确认 mml 的第一个子节点的第一个子节点的节点名称是否
    # 断言检查第一个子节点的第一个子节点的第一个子节点的标签名是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeName == 'mml:mi'
    
    # 断言检查第一个子节点的第一个子节点的第一个子节点的节点值是否为 '2'
    assert mml.childNodes[0].childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    
    # 断言检查第一个子节点的第一个子节点的第二个子节点的标签名是否为 'mml:mo'
    assert mml.childNodes[0].childNodes[1].childNodes[1].nodeName == 'mml:mo'
    
    # 断言检查第一个子节点的第一个子节点的第二个子节点的节点值是否为一个空格字符 ' '
    assert mml.childNodes[0].childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    
    # 断言检查第一个子节点的第一个子节点的第三个子节点的标签名是否为 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[2].nodeName == 'mml:mi'
    
    # 断言检查第一个子节点的第一个子节点的第三个子节点的节点值是否为 'a'
    assert mml.childNodes[0].childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    
    # 删除变量 mml，释放资源
    del mml
# 定义一个测试函数，用于测试生成 MathML 中希腊字母的输出
def test_content_mathml_greek():
    # 生成符号 'alpha' 对应的 MathML 表示，并检查其节点名是否为 'ci'
    mml = mp._print(Symbol('alpha'))
    assert mml.nodeName == 'ci'
    # 检查节点的第一个子节点的值是否为希腊小写字母 'alpha'
    assert mml.childNodes[0].nodeValue == '\N{GREEK SMALL LETTER ALPHA}'

    # 依次测试每个希腊字母的 MathML 表示
    assert mp.doprint(Symbol('alpha')) == '<ci>&#945;</ci>'
    assert mp.doprint(Symbol('beta')) == '<ci>&#946;</ci>'
    assert mp.doprint(Symbol('gamma')) == '<ci>&#947;</ci>'
    assert mp.doprint(Symbol('delta')) == '<ci>&#948;</ci>'
    assert mp.doprint(Symbol('epsilon')) == '<ci>&#949;</ci>'
    assert mp.doprint(Symbol('zeta')) == '<ci>&#950;</ci>'
    assert mp.doprint(Symbol('eta')) == '<ci>&#951;</ci>'
    assert mp.doprint(Symbol('theta')) == '<ci>&#952;</ci>'
    assert mp.doprint(Symbol('iota')) == '<ci>&#953;</ci>'
    assert mp.doprint(Symbol('kappa')) == '<ci>&#954;</ci>'
    assert mp.doprint(Symbol('lambda')) == '<ci>&#955;</ci>'
    assert mp.doprint(Symbol('mu')) == '<ci>&#956;</ci>'
    assert mp.doprint(Symbol('nu')) == '<ci>&#957;</ci>'
    assert mp.doprint(Symbol('xi')) == '<ci>&#958;</ci>'
    assert mp.doprint(Symbol('omicron')) == '<ci>&#959;</ci>'
    assert mp.doprint(Symbol('pi')) == '<ci>&#960;</ci>'
    assert mp.doprint(Symbol('rho')) == '<ci>&#961;</ci>'
    assert mp.doprint(Symbol('varsigma')) == '<ci>&#962;</ci>'
    assert mp.doprint(Symbol('sigma')) == '<ci>&#963;</ci>'
    assert mp.doprint(Symbol('tau')) == '<ci>&#964;</ci>'
    assert mp.doprint(Symbol('upsilon')) == '<ci>&#965;</ci>'
    assert mp.doprint(Symbol('phi')) == '<ci>&#966;</ci>'
    assert mp.doprint(Symbol('chi')) == '<ci>&#967;</ci>'
    assert mp.doprint(Symbol('psi')) == '<ci>&#968;</ci>'
    assert mp.doprint(Symbol('omega')) == '<ci>&#969;</ci>'

    # 测试每个大写希腊字母的 MathML 表示
    assert mp.doprint(Symbol('Alpha')) == '<ci>&#913;</ci>'
    assert mp.doprint(Symbol('Beta')) == '<ci>&#914;</ci>'
    assert mp.doprint(Symbol('Gamma')) == '<ci>&#915;</ci>'
    assert mp.doprint(Symbol('Delta')) == '<ci>&#916;</ci>'
    assert mp.doprint(Symbol('Epsilon')) == '<ci>&#917;</ci>'
    assert mp.doprint(Symbol('Zeta')) == '<ci>&#918;</ci>'
    assert mp.doprint(Symbol('Eta')) == '<ci>&#919;</ci>'
    assert mp.doprint(Symbol('Theta')) == '<ci>&#920;</ci>'
    assert mp.doprint(Symbol('Iota')) == '<ci>&#921;</ci>'
    assert mp.doprint(Symbol('Kappa')) == '<ci>&#922;</ci>'
    assert mp.doprint(Symbol('Lambda')) == '<ci>&#923;</ci>'
    assert mp.doprint(Symbol('Mu')) == '<ci>&#924;</ci>'
    assert mp.doprint(Symbol('Nu')) == '<ci>&#925;</ci>'
    assert mp.doprint(Symbol('Xi')) == '<ci>&#926;</ci>'
    assert mp.doprint(Symbol('Omicron')) == '<ci>&#927;</ci>'
    assert mp.doprint(Symbol('Pi')) == '<ci>&#928;</ci>'
    assert mp.doprint(Symbol('Rho')) == '<ci>&#929;</ci>'
    assert mp.doprint(Symbol('Sigma')) == '<ci>&#931;</ci>'
    assert mp.doprint(Symbol('Tau')) == '<ci>&#932;</ci>'
    assert mp.doprint(Symbol('Upsilon')) == '<ci>&#933;</ci>'
    assert mp.doprint(Symbol('Phi')) == '<ci>&#934;</ci>'
    assert mp.doprint(Symbol('Chi')) == '<ci>&#935;</ci>'
    # 确保符号 'Psi' 被正确打印为 XML 格式的字符串 '<ci>&#936;</ci>'
    assert mp.doprint(Symbol('Psi')) == '<ci>&#936;</ci>'
    # 确保符号 'Omega' 被正确打印为 XML 格式的字符串 '<ci>&#937;</ci>'
    assert mp.doprint(Symbol('Omega')) == '<ci>&#937;</ci>'
# 定义测试函数，用于测试 MathMLContentPrinter 类的功能，检查生成的 MathML 内容是否符合预期
def test_content_mathml_order():
    # 定义一个数学表达式
    expr = x**3 + x**2*y + 3*x*y**3 + y**4

    # 创建 MathMLContentPrinter 对象，设置排序顺序为 'lex'，生成数学表达式的 MathML 字符串
    mp = MathMLContentPrinter({'order': 'lex'})
    mml = mp._print(expr)

    # 断言语句，检查生成的 MathML 结构是否符合预期
    assert mml.childNodes[1].childNodes[0].nodeName == 'power'
    assert mml.childNodes[1].childNodes[1].childNodes[0].data == 'x'
    assert mml.childNodes[1].childNodes[2].childNodes[0].data == '3'

    assert mml.childNodes[4].childNodes[0].nodeName == 'power'
    assert mml.childNodes[4].childNodes[1].childNodes[0].data == 'y'
    assert mml.childNodes[4].childNodes[2].childNodes[0].data == '4'

    # 创建 MathMLContentPrinter 对象，设置排序顺序为 'rev-lex'，生成数学表达式的 MathML 字符串
    mp = MathMLContentPrinter({'order': 'rev-lex'})
    mml = mp._print(expr)

    # 断言语句，检查生成的 MathML 结构是否符合预期
    assert mml.childNodes[1].childNodes[0].nodeName == 'power'
    assert mml.childNodes[1].childNodes[1].childNodes[0].data == 'y'
    assert mml.childNodes[1].childNodes[2].childNodes[0].data == '4'

    assert mml.childNodes[4].childNodes[0].nodeName == 'power'
    assert mml.childNodes[4].childNodes[1].childNodes[0].data == 'x'
    assert mml.childNodes[4].childNodes[2].childNodes[0].data == '3'


# 定义测试函数，用于测试 mathml 函数的设置参数
def test_content_settings():
    # 断言语句，检查调用 mathml 函数时使用不存在的方法参数时是否会引发 TypeError 异常
    raises(TypeError, lambda: mathml(x, method="garbage"))


# 定义测试函数，用于测试 mathml 函数对逻辑运算符的处理
def test_content_mathml_logic():
    # 断言语句，检查 mathml 函数生成 And 逻辑的 MathML 字符串是否符合预期
    assert mathml(And(x, y)) == '<apply><and/><ci>x</ci><ci>y</ci></apply>'
    # 断言语句，检查 mathml 函数生成 Or 逻辑的 MathML 字符串是否符合预期
    assert mathml(Or(x, y)) == '<apply><or/><ci>x</ci><ci>y</ci></apply>'
    # 断言语句，检查 mathml 函数生成 Xor 逻辑的 MathML 字符串是否符合预期
    assert mathml(Xor(x, y)) == '<apply><xor/><ci>x</ci><ci>y</ci></apply>'
    # 断言语句，检查 mathml 函数生成 Implies 逻辑的 MathML 字符串是否符合预期
    assert mathml(Implies(x, y)) == '<apply><implies/><ci>x</ci><ci>y</ci></apply>'
    # 断言语句，检查 mathml 函数生成 Not 逻辑的 MathML 字符串是否符合预期
    assert mathml(Not(x)) == '<apply><not/><ci>x</ci></apply>'


# 定义测试函数，用于测试 mathml 函数对有限集合的处理
def test_content_finite_sets():
    # 断言语句，检查 mathml 函数生成单元素有限集合的 MathML 字符串是否符合预期
    assert mathml(FiniteSet(a)) == '<set><ci>a</ci></set>'
    # 断言语句，检查 mathml 函数生成多元素有限集合的 MathML 字符串是否符合预期
    assert mathml(FiniteSet(a, b)) == '<set><ci>a</ci><ci>b</ci></set>'
    # 断言语句，检查 mathml 函数生成嵌套有限集合的 MathML 字符串是否符合预期
    assert mathml(FiniteSet(FiniteSet(a, b), c)) == \
        '<set><ci>c</ci><set><ci>a</ci><ci>b</ci></set></set>'

    # 创建多个有限集合对象
    A = FiniteSet(a)
    B = FiniteSet(b)
    C = FiniteSet(c)
    D = FiniteSet(d)

    # 创建多个集合运算对象，不进行求值
    U1 = Union(A, B, evaluate=False)
    U2 = Union(C, D, evaluate=False)
    I1 = Intersection(A, B, evaluate=False)
    I2 = Intersection(C, D, evaluate=False)
    C1 = Complement(A, B, evaluate=False)
    C2 = Complement(C, D, evaluate=False)
    # 注意：ProductSet 不支持 evaluate 关键字

    # 断言语句，检查 mathml 函数生成并集的 MathML 字符串是否符合预期
    assert mathml(U1) == \
        '<apply><union/><set><ci>a</ci></set><set><ci>b</ci></set></apply>'
    # 断言语句，检查 mathml 函数生成交集的 MathML 字符串是否符合预期
    assert mathml(I1) == \
        '<apply><intersect/><set><ci>a</ci></set><set><ci>b</ci></set>' \
        '</apply>'
    # 断言语句，检查 mathml 函数生成补集的 MathML 字符串是否符合预期
    assert mathml(C1) == \
        '<apply><setdiff/><set><ci>a</ci></set><set><ci>b</ci></set></apply>'
    # 断言语句，检查 mathml 函数生成笛卡尔积的 MathML 字符串是否符合预期
    assert mathml(P1) == \
        '<apply><cartesianproduct/><set><ci>a</ci></set><set><ci>b</ci>' \
        '</set></apply>'

    # 断言语句，检查 mathml 函数生成集合运算的 MathML 字符串是否符合预期
    assert mathml(Intersection(A, U2, evaluate=False)) == \
        '<apply><intersect/><set><ci>a</ci></set><apply><union/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    # 断言检查交集操作的数学表达式是否正确转换为 MathML 格式
    assert mathml(Intersection(U1, U2, evaluate=False)) == \
        '<apply><intersect/><apply><union/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'

    # 断言检查补集操作的数学表达式是否正确转换为 MathML 格式
    # XXX 这些示例中括号在 MathJax 中显示正确吗？
    assert mathml(Intersection(C1, C2, evaluate=False)) == \
        '<apply><intersect/><apply><setdiff/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><setdiff/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    
    # 断言检查交集操作的数学表达式是否正确转换为 MathML 格式
    assert mathml(Intersection(P1, P2, evaluate=False)) == \
        '<apply><intersect/><apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><cartesianproduct/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'

    # 断言检查并集操作的数学表达式是否正确转换为 MathML 格式
    assert mathml(Union(A, I2, evaluate=False)) == \
        '<apply><union/><set><ci>a</ci></set><apply><intersect/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    
    # 断言检查并集操作的数学表达式是否正确转换为 MathML 格式
    assert mathml(Union(I1, I2, evaluate=False)) == \
        '<apply><union/><apply><intersect/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><intersect/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    
    # 断言检查并集操作的数学表达式是否正确转换为 MathML 格式
    assert mathml(Union(C1, C2, evaluate=False)) == \
        '<apply><union/><apply><setdiff/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><setdiff/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    
    # 断言检查并集操作的数学表达式是否正确转换为 MathML 格式
    assert mathml(Union(P1, P2, evaluate=False)) == \
        '<apply><union/><apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><cartesianproduct/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'

    # 断言检查补集操作的数学表达式是否正确转换为 MathML 格式
    assert mathml(Complement(A, C2, evaluate=False)) == \
        '<apply><setdiff/><set><ci>a</ci></set><apply><setdiff/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    
    # 断言检查补集操作的数学表达式是否正确转换为 MathML 格式
    assert mathml(Complement(U1, U2, evaluate=False)) == \
        '<apply><setdiff/><apply><union/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    
    # 断言检查补集操作的数学表达式是否正确转换为 MathML 格式
    assert mathml(Complement(I1, I2, evaluate=False)) == \
        '<apply><setdiff/><apply><intersect/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><intersect/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    
    # 断言检查补集操作的数学表达式是否正确转换为 MathML 格式
    assert mathml(Complement(P1, P2, evaluate=False)) == \
        '<apply><setdiff/><apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><cartesianproduct/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    # 断言检查 ProductSet(A, P2) 的数学表达式是否等于给定的 MathML 字符串
    assert mathml(ProductSet(A, P2)) == \
        '<apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<apply><cartesianproduct/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    
    # 断言检查 ProductSet(U1, U2) 的数学表达式是否等于给定的 MathML 字符串
    assert mathml(ProductSet(U1, U2)) == \
        '<apply><cartesianproduct/><apply><union/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    
    # 断言检查 ProductSet(I1, I2) 的数学表达式是否等于给定的 MathML 字符串
    assert mathml(ProductSet(I1, I2)) == \
        '<apply><cartesianproduct/><apply><intersect/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><intersect/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    
    # 断言检查 ProductSet(C1, C2) 的数学表达式是否等于给定的 MathML 字符串
    assert mathml(ProductSet(C1, C2)) == \
        '<apply><cartesianproduct/><apply><setdiff/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><setdiff/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
# 定义一个测试函数，用于测试数学表达式的打印方法
def test_presentation_printmethod():
    # 断言1：测试 1 + x 的打印结果是否为指定的 MathML 字符串
    assert mpp.doprint(1 + x) == '<mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow>'
    # 断言2：测试 x^2 的打印结果是否为指定的 MathML 字符串
    assert mpp.doprint(x**2) == '<msup><mi>x</mi><mn>2</mn></msup>'
    # 断言3：测试 x^-1 的打印结果是否为指定的 MathML 字符串
    assert mpp.doprint(x**-1) == '<mfrac><mn>1</mn><mi>x</mi></mfrac>'
    # 断言4：测试 x^-2 的打印结果是否为指定的 MathML 字符串
    assert mpp.doprint(x**-2) == \
        '<mfrac><mn>1</mn><msup><mi>x</mi><mn>2</mn></msup></mfrac>'
    # 断言5：测试 2*x 的打印结果是否为指定的 MathML 字符串
    assert mpp.doprint(2*x) == \
        '<mrow><mn>2</mn><mo>&InvisibleTimes;</mo><mi>x</mi></mrow>'


# 定义一个测试函数，用于测试核心 MathML 打印方法
def test_presentation_mathml_core():
    # 调用 _print 方法打印 1 + x 的 MathML 表示
    mml_1 = mpp._print(1 + x)
    # 断言1：检查 MathML 节点的名称是否为 'mrow'
    assert mml_1.nodeName == 'mrow'
    # 获取节点的子节点列表
    nodes = mml_1.childNodes
    # 断言2：检查子节点的数量是否为 3
    assert len(nodes) == 3
    # 断言3：检查第一个子节点的名称是否为 'mi' 或 'mn'
    assert nodes[0].nodeName in ['mi', 'mn']
    # 断言4：检查第二个子节点的名称是否为 'mo'
    assert nodes[1].nodeName == 'mo'
    # 根据第一个子节点的名称进行条件判断
    if nodes[0].nodeName == 'mn':
        # 如果第一个子节点是 'mn'，则检查其内容是否为 '1'
        assert nodes[0].childNodes[0].nodeValue == '1'
        # 同时检查第三个子节点的内容是否为 'x'
        assert nodes[2].childNodes[0].nodeValue == 'x'
    else:
        # 如果第一个子节点是 'mi'，则检查其内容是否为 'x'
        assert nodes[0].childNodes[0].nodeValue == 'x'
        # 同时检查第三个子节点的内容是否为 '1'
        assert nodes[2].childNodes[0].nodeValue == '1'

    # 调用 _print 方法打印 x^2 的 MathML 表示
    mml_2 = mpp._print(x**2)
    # 断言5：检查 MathML 节点的名称是否为 'msup'
    assert mml_2.nodeName == 'msup'
    # 获取节点的子节点列表
    nodes = mml_2.childNodes
    # 断言6：检查基底节点的内容是否为 'x'
    assert nodes[0].childNodes[0].nodeValue == 'x'
    # 断言7：检查指数节点的内容是否为 '2'
    assert nodes[1].childNodes[0].nodeValue == '2'

    # 调用 _print 方法打印 2*x 的 MathML 表示
    mml_3 = mpp._print(2*x)
    # 断言8：检查 MathML 节点的名称是否为 'mrow'
    assert mml_3.nodeName == 'mrow'
    # 获取节点的子节点列表
    nodes = mml_3.childNodes
    # 断言9：检查第一个子节点的内容是否为 '2'
    assert nodes[0].childNodes[0].nodeValue == '2'
    # 断言10：检查第二个子节点的内容是否为 '&InvisibleTimes;'
    assert nodes[1].childNodes[0].nodeValue == '&InvisibleTimes;'
    # 断言11：检查第三个子节点的内容是否为 'x'
    assert nodes[2].childNodes[0].nodeValue == 'x'

    # 调用 _print 方法打印 Float(1.0, 2)*x 的 MathML 表示
    mml = mpp._print(Float(1.0, 2)*x)
    # 断言12：检查 MathML 节点的名称是否为 'mrow'
    assert mml.nodeName == 'mrow'
    # 获取节点的子节点列表
    nodes = mml.childNodes
    # 断言13：检查第一个子节点的内容是否为 '1.0'
    assert nodes[0].childNodes[0].nodeValue == '1.0'
    # 断言14：检查第二个子节点的内容是否为 '&InvisibleTimes;'
    assert nodes[1].childNodes[0].nodeValue == '&InvisibleTimes;'
    # 断言15：检查第三个子节点的内容是否为 'x'
    assert nodes[2].childNodes[0].nodeValue == 'x'


# 定义一个测试函数，用于测试数学函数的 MathML 表示
def test_presentation_mathml_functions():
    # 调用 _print 方法打印 sin(x) 的 MathML 表示
    mml_1 = mpp._print(sin(x))
    # 断言1：检查 MathML 第一个子节点的内容是否为 'sin'
    assert mml_1.childNodes[0].childNodes[0].nodeValue == 'sin'
    # 断言2：检查 MathML 第二个子节点的内容是否为 'x'
    assert mml_1.childNodes[1].childNodes[0].nodeValue == 'x'

    # 调用 _print 方法打印 diff(sin(x), x, evaluate=False) 的 MathML 表示
    mml_2 = mpp._print(diff(sin(x), x, evaluate=False))
    # 断言3：检查 MathML 节点的名称是否为 'mrow'
    assert mml_2.nodeName == 'mrow'
    # 断言4：检查第一个子节点的内容是否为 '&dd;'
    assert mml_2.childNodes[0].childNodes[0].childNodes[0].nodeValue == '&dd;'
    # 断言5：检查第二个子节点的第二个子节点的名称是否为 'mfenced'
    assert mml_2.childNodes[1].childNodes[1].nodeName == 'mfenced'
    # 断言6：检查第一个子节点的第二个子节点的内容是否为 '&dd;'
    assert mml_2.childNodes[0].childNodes[1].childNodes[0].childNodes[0].nodeValue == '&dd;'

    # 调用 _print 方法打印 diff(cos(x*y), x, evaluate=False) 的 MathML 表示
    mml_3 = mpp._print(diff(cos(x*y), x, evaluate=False))
    # 断言7：检查 MathML 第一个子节点的名称是否为 'mfrac'
    assert mml_3.childNodes[0].nodeName == 'mfrac'
    # 断言8：检查第一个子节点的第一个子节点的内容是否为 '&#x2202;'
    assert mml_3.childNodes[0].childNodes[0].childNodes[0].childNodes[0].nodeValue == '&#x2202;'
    # 断言9：检查第二个子节点的内容是否为 'cos'


# 定义一个测试函数，用于测试导数表达式的打印
def test_print_derivative():
    # 创建函数 f 和导数对象 d
    f = Function('f')
    d = Derivative(f(x, y, z), x, z, x, z, z, y)
    # 断言1：检查 Derivative 对象 d 的 MathML 表示是否为指定的字符串
    assert mathml(d) == \
        '<apply><partialdiff/><bvar><ci>y</ci><ci>z</ci><degree><cn>2</cn></degree><ci>x</ci><ci>z</ci><ci>x</ci></bvar><apply><f/><ci>x</ci><ci>y</ci><ci>z</ci></apply></apply>'
    # 断言检查数学表达式的 MathML 表示是否与指定的字符串相等
    assert mathml(d, printer='presentation') == \
        '<mrow><mfrac><mrow><msup><mo>&#x2202;</mo><mn>6</mn></msup></mrow><mrow><mo>&#x2202;</mo><mi>y</mi><msup><mo>&#x2202;</mo><mn>2</mn></msup><mi>z</mi><mo>&#x2202;</mo><mi>x</mi><mo>&#x2202;</mo><mi>z</mi><mo>&#x2202;</mo><mi>x</mi></mrow></mfrac><mrow><mi>f</mi><mfenced><mi>x</mi><mi>y</mi><mi>z</mi></mfenced></mrow></mrow>'
# 测试用例：测试生成 MathML 表示的极限函数
def test_presentation_mathml_limits():
    # 定义一个极限函数 sin(x)/x
    lim_fun = sin(x)/x
    # 使用打印器 mpp._print() 生成极限函数 lim_fun 在 MathML 中的表示
    mml_1 = mpp._print(Limit(lim_fun, x, 0))
    # 断言第一个子节点的节点名称是 'munder'
    assert mml_1.childNodes[0].nodeName == 'munder'
    # 断言第一个子节点的第一个子节点的第一个子节点的 nodeValue 是 'lim'
    assert mml_1.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'lim'
    # 断言第一个子节点的第一个子节点的第二个子节点的第一个子节点的 nodeValue 是 'x'
    assert mml_1.childNodes[0].childNodes[1].childNodes[0].childNodes[0].nodeValue == 'x'
    # 断言第一个子节点的第一个子节点的第二个子节点的第二个子节点的第一个子节点的 nodeValue 是 '→'
    assert mml_1.childNodes[0].childNodes[1].childNodes[1].childNodes[0].nodeValue == '&#x2192;'
    # 断言第一个子节点的第一个子节点的第二个子节点的第三个子节点的第一个子节点的 nodeValue 是 '0'
    assert mml_1.childNodes[0].childNodes[1].childNodes[2].childNodes[0].nodeValue == '0'


# 测试用例：测试生成 MathML 表示的积分函数
def test_presentation_mathml_integrals():
    # 断言积分表达式 Integral(x, (x, 0, 1)) 在 MathML 中的打印结果
    assert mpp.doprint(Integral(x, (x, 0, 1))) == \
        '<mrow><msubsup><mo>&#x222B;</mo><mn>0</mn><mn>1</mn></msubsup>'\
        '<mi>x</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    # 断言积分表达式 Integral(log(x), x) 在 MathML 中的打印结果
    assert mpp.doprint(Integral(log(x), x)) == \
        '<mrow><mo>&#x222B;</mo><mrow><mi>log</mi><mfenced><mi>x</mi>'\
        '</mfenced></mrow><mo>&dd;</mo><mi>x</mi></mrow>'
    # 断言积分表达式 Integral(x*y, x, y) 在 MathML 中的打印结果
    assert mpp.doprint(Integral(x*y, x, y)) == \
        '<mrow><mo>&#x222C;</mo><mrow><mi>x</mi><mo>&InvisibleTimes;</mo>'\
        '<mi>y</mi></mrow><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    # 定义符号 z, w
    z, w = symbols('z w')
    # 断言积分表达式 Integral(x*y*z, x, y, z) 在 MathML 中的打印结果
    assert mpp.doprint(Integral(x*y*z, x, y, z)) == \
        '<mrow><mo>&#x222D;</mo><mrow><mi>x</mi><mo>&InvisibleTimes;</mo>'\
        '<mi>y</mi><mo>&InvisibleTimes;</mo><mi>z</mi></mrow><mo>&dd;</mo>'\
        '<mi>z</mi><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    # 断言积分表达式 Integral(x*y*z*w, x, y, z, w) 在 MathML 中的打印结果
    assert mpp.doprint(Integral(x*y*z*w, x, y, z, w)) == \
        '<mrow><mo>&#x222B;</mo><mo>&#x222B;</mo><mo>&#x222B;</mo>'\
        '<mo>&#x222B;</mo><mrow><mi>w</mi><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi>'\
        '<mo>&InvisibleTimes;</mo><mi>z</mi></mrow><mo>&dd;</mo><mi>w</mi>'\
        '<mo>&dd;</mo><mi>z</mi><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    # 断言积分表达式 Integral(x, (x, 0)) 在 MathML 中的打印结果
    assert mpp.doprint(Integral(x, (x, 0))) == \
        '<mrow><msup><mo>&#x222B;</mo><mn>0</mn></msup><mi>x</mi><mo>&dd;</mo>'\
        '<mi>x</mi></mrow>'


# 测试用例：测试生成 MathML 表示的矩阵
def test_presentation_mathml_matrices():
    # 创建矩阵 A 和 B
    A = Matrix([1, 2, 3])
    B = Matrix([[0, 5, 4], [2, 3, 1], [9, 7, 9]])
    # 使用打印器 mpp._print() 生成矩阵 A 在 MathML 中的表示
    mll_1 = mpp._print(A)
    # 断言第一个子节点是 'mtable'
    assert mll_1.childNodes[0].nodeName == 'mtable'
    # 断言第一个子节点的第一个子节点是 'mtr'
    assert mll_1.childNodes[0].childNodes[0].nodeName == 'mtr'
    # 断言第一个子节点的子节点数量是 3
    assert len(mll_1.childNodes[0].childNodes) == 3
    # 断言第一个子节点的第一个子节点的第一个子节点是 'mtd'
    assert mll_1.childNodes[0].childNodes[0].childNodes[0].nodeName == 'mtd'
    # 断言第一个子节点的第一个子节点的子节点数量是 1
    assert len(mll_1.childNodes[0].childNodes[0].childNodes) == 1
    # 断言第一个子节点的第一个子节点的第一个子节点的第一个子节点的 nodeValue 是 '1'
    assert mll_1.childNodes[0].childNodes[0].childNodes[0].childNodes[0].childNodes[0].nodeValue == '1'
    # 断言确保 mll_1 结构中指定路径的节点值为 '2'
    assert mll_1.childNodes[0].childNodes[1].childNodes[0].childNodes[0].childNodes[0].nodeValue == '2'
    # 断言确保 mll_1 结构中另一路径的节点值为 '3'
    assert mll_1.childNodes[0].childNodes[2].childNodes[0].childNodes[0].childNodes[0].nodeValue == '3'
    # 使用 mpp 对象调用 _print 方法，并将结果赋给 mll_2
    mll_2 = mpp._print(B)
    # 断言确保 mll_2 结构的根节点名为 'mtable'
    assert mll_2.childNodes[0].nodeName == 'mtable'
    # 断言确保 mll_2 结构的第一个子节点名为 'mtr'
    assert mll_2.childNodes[0].childNodes[0].nodeName == 'mtr'
    # 断言确保 mll_2 结构的子节点数为 3
    assert len(mll_2.childNodes[0].childNodes) == 3
    # 断言确保 mll_2 结构的第一个 mtr 节点的第一个子节点名为 'mtd'
    assert mll_2.childNodes[0].childNodes[0].childNodes[0].nodeName == 'mtd'
    # 断言确保 mll_2 结构的第一个 mtr 节点的子节点数为 3
    assert len(mll_2.childNodes[0].childNodes[0].childNodes) == 3
    # 断言确保 mll_2 结构的第一个 mtr 节点的第一个 mtd 节点的第一个子节点的节点值为 '0'
    assert mll_2.childNodes[0].childNodes[0].childNodes[0].childNodes[0].childNodes[0].nodeValue == '0'
    # 断言确保 mll_2 结构的第一个 mtr 节点的第二个 mtd 节点的第一个子节点的节点值为 '5'
    assert mll_2.childNodes[0].childNodes[0].childNodes[1].childNodes[0].childNodes[0].nodeValue == '5'
    # 断言确保 mll_2 结构的第一个 mtr 节点的第三个 mtd 节点的第一个子节点的节点值为 '4'
    assert mll_2.childNodes[0].childNodes[0].childNodes[2].childNodes[0].childNodes[0].nodeValue == '4'
    # 以下一系列断言确保 mll_2 结构中其余节点的特定路径的节点值符合预期
    assert mll_2.childNodes[0].childNodes[1].childNodes[0].childNodes[0].childNodes[0].nodeValue == '2'
    assert mll_2.childNodes[0].childNodes[1].childNodes[1].childNodes[0].childNodes[0].nodeValue == '3'
    assert mll_2.childNodes[0].childNodes[1].childNodes[2].childNodes[0].childNodes[0].nodeValue == '1'
    assert mll_2.childNodes[0].childNodes[2].childNodes[0].childNodes[0].childNodes[0].nodeValue == '9'
    assert mll_2.childNodes[0].childNodes[2].childNodes[1].childNodes[0].childNodes[0].nodeValue == '7'
    assert mll_2.childNodes[0].childNodes[2].childNodes[2].childNodes[0].childNodes[0].nodeValue == '9'
# 测试生成数学表示的函数：求和表达式
def test_presentation_mathml_sums():
    # 定义求和的被加数
    summand = x
    # 生成求和表达式的 MathML 表示
    mml_1 = mpp._print(Sum(summand, (x, 1, 10)))
    # 断言 MathML 结构的第一个子节点是 munderover
    assert mml_1.childNodes[0].nodeName == 'munderover'
    # 断言 munderover 节点下有三个子节点
    assert len(mml_1.childNodes[0].childNodes) == 3
    # 断言第一个子节点的第一个子节点的值是 '&#x2211;'
    assert mml_1.childNodes[0].childNodes[0].childNodes[0].nodeValue == '&#x2211;'
    # 断言第二个子节点的子节点数为 3
    assert len(mml_1.childNodes[0].childNodes[1].childNodes) == 3
    # 断言第三个子节点的第一个子节点的值是 '10'
    assert mml_1.childNodes[0].childNodes[2].childNodes[0].nodeValue == '10'
    # 断言第二个子节点的第一个子节点的值是 'x'
    assert mml_1.childNodes[1].childNodes[0].nodeValue == 'x'


# 测试生成数学表示的函数：加法表达式
def test_presentation_mathml_add():
    # 生成加法表达式的 MathML 表示
    mml = mpp._print(x**5 - x**4 + x)
    # 断言 MathML 结构的子节点数为 5
    assert len(mml.childNodes) == 5
    # 断言第一个子节点的第一个子节点的第一个子节点的值是 'x'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    # 断言第一个子节点的第二个子节点的第一个子节点的值是 '5'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '5'
    # 断言第二个子节点的第一个子节点的值是 '-'
    assert mml.childNodes[1].childNodes[0].nodeValue == '-'
    # 断言第三个子节点的第一个子节点的第一个子节点的值是 'x'
    assert mml.childNodes[2].childNodes[0].childNodes[0].nodeValue == 'x'
    # 断言第三个子节点的第二个子节点的第一个子节点的值是 '4'
    assert mml.childNodes[2].childNodes[1].childNodes[0].nodeValue == '4'
    # 断言第四个子节点的第一个子节点的值是 '+'
    assert mml.childNodes[3].childNodes[0].nodeValue == '+'
    # 断言第五个子节点的第一个子节点的值是 'x'
    assert mml.childNodes[4].childNodes[0].nodeValue == 'x'


# 测试生成数学表示的函数：有理数表达式
def test_presentation_mathml_Rational():
    # 生成有理数 1 的 MathML 表示
    mml_1 = mpp._print(Rational(1, 1))
    # 断言节点名是 'mn'
    assert mml_1.nodeName == 'mn'

    # 生成有理数 2/5 的 MathML 表示
    mml_2 = mpp._print(Rational(2, 5))
    # 断言节点名是 'mfrac'
    assert mml_2.nodeName == 'mfrac'
    # 断言分子节点的第一个子节点的值是 '2'
    assert mml_2.childNodes[0].childNodes[0].nodeValue == '2'
    # 断言分母节点的第一个子节点的值是 '5'
    assert mml_2.childNodes[1].childNodes[0].nodeValue == '5'


# 测试生成数学表示的函数：常数表达式
def test_presentation_mathml_constants():
    # 生成虚数单位的 MathML 表示
    mml = mpp._print(I)
    # 断言第一个子节点的值是 '&ImaginaryI;'
    assert mml.childNodes[0].nodeValue == '&ImaginaryI;'

    # 生成自然对数底的 MathML 表示
    mml = mpp._print(E)
    # 断言第一个子节点的值是 '&ExponentialE;'
    assert mml.childNodes[0].nodeValue == '&ExponentialE;'

    # 生成无穷大的 MathML 表示
    mml = mpp._print(oo)
    # 断言第一个子节点的值是 '&#x221E;'
    assert mml.childNodes[0].nodeValue == '&#x221E;'

    # 生成圆周率的 MathML 表示
    mml = mpp._print(pi)
    # 断言第一个子节点的值是 '&pi;'
    assert mml.childNodes[0].nodeValue == '&pi;'

    # 断言特定数学常数的 MathML 表示
    assert mathml(hbar, printer='presentation') == '<mi>&#x210F;</mi>'
    assert mathml(S.TribonacciConstant, printer='presentation') == '<mi>TribonacciConstant</mi>'
    assert mathml(S.EulerGamma, printer='presentation') == '<mi>&#x3B3;</mi>'
    assert mathml(S.GoldenRatio, printer='presentation') == '<mi>&#x3A6;</mi>'

    # 生成无限大的 MathML 表示
    assert mathml(zoo, printer='presentation') == '<mover><mo>&#x221E;</mo><mo>~</mo></mover>'

    # 生成非数值的 MathML 表示
    assert mathml(S.NaN, printer='presentation') == '<mi>NaN</mi>'


# 测试生成数学表示的函数：三角函数表达式
def test_presentation_mathml_trig():
    # 生成正弦函数的 MathML 表示
    mml = mpp._print(sin(x))
    # 断言第一个子节点的第一个子节点的值是 'sin'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'sin'

    # 生成余弦函数的 MathML 表示
    mml = mpp._print(cos(x))
    # 断言第一个子节点的第一个子节点的值是 'cos'

    assert mml.childNodes[0].childNodes[0].nodeValue == 'cos'

    # 生成正切函数的 MathML 表示
    mml = mpp._print(tan(x))
    # 断言第一个子节点的第一个子节点的值是 'tan'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'tan'

    # 生成反正弦函数的 MathML 表示
    mml = mpp._print(asin(x))
    # 断言第一个子节点的第一个子节点的值是 'arcsin'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arcsin'

    # 生成反余弦函数的 MathML 表示
    mml = mpp._print(acos(x))
    # 断言第一个子节点的第一个子节点的值是 'arccos'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arccos'

    # 生成反正切函数的 MathML 表示
    mml = mpp._print(atan(x))
    # 断言第一个子节点的第一个子节点的值是 'arctan'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arctan'

    # 生成双曲正弦函数的 MathML 表示
    mml = mpp._print(sinh(x))
    # 断言，确保节点树的第一个子节点的第一个子节点的值等于 'sinh'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'sinh'
    
    # 使用数学表达式打印器将 x 的双曲余弦函数表示为 MathML，并存储在 mml 中
    mml = mpp._print(cosh(x))
    # 断言，确保节点树的第一个子节点的第一个子节点的值等于 'cosh'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'cosh'
    
    # 使用数学表达式打印器将 x 的双曲正切函数表示为 MathML，并存储在 mml 中
    mml = mpp._print(tanh(x))
    # 断言，确保节点树的第一个子节点的第一个子节点的值等于 'tanh'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'tanh'
    
    # 使用数学表达式打印器将 x 的反双曲正弦函数表示为 MathML，并存储在 mml 中
    mml = mpp._print(asinh(x))
    # 断言，确保节点树的第一个子节点的第一个子节点的值等于 'arcsinh'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arcsinh'
    
    # 使用数学表达式打印器将 x 的反双曲正切函数表示为 MathML，并存储在 mml 中
    mml = mpp._print(atanh(x))
    # 断言，确保节点树的第一个子节点的第一个子节点的值等于 'arctanh'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arctanh'
    
    # 使用数学表达式打印器将 x 的反双曲余弦函数表示为 MathML，并存储在 mml 中
    mml = mpp._print(acosh(x))
    # 断言，确保节点树的第一个子节点的第一个子节点的值等于 'arccosh'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arccosh'
# 定义一个测试函数，用于测试生成数学表达式的 MathML 表示
def test_presentation_mathml_relational():
    # 生成 x = 1 的 MathML 表示
    mml_1 = mpp._print(Eq(x, 1))
    # 断言 MathML 结点的子结点数量为 3
    assert len(mml_1.childNodes) == 3
    # 断言第一个子结点的节点名称为 'mi'，表示数学标识符（math identifier）
    assert mml_1.childNodes[0].nodeName == 'mi'
    # 断言第一个子结点的第一个子结点的节点值为 'x'
    assert mml_1.childNodes[0].childNodes[0].nodeValue == 'x'
    # 断言第二个子结点的节点名称为 'mo'，表示数学运算符（math operator）
    assert mml_1.childNodes[1].nodeName == 'mo'
    # 断言第二个子结点的第一个子结点的节点值为 '='
    assert mml_1.childNodes[1].childNodes[0].nodeValue == '='
    # 断言第三个子结点的节点名称为 'mn'，表示数学数字（math number）
    assert mml_1.childNodes[2].nodeName == 'mn'
    # 断言第三个子结点的第一个子结点的节点值为 '1'
    assert mml_1.childNodes[2].childNodes[0].nodeValue == '1'

    # 生成 1 ≠ x 的 MathML 表示
    mml_2 = mpp._print(Ne(1, x))
    assert len(mml_2.childNodes) == 3
    assert mml_2.childNodes[0].nodeName == 'mn'
    assert mml_2.childNodes[0].childNodes[0].nodeValue == '1'
    assert mml_2.childNodes[1].nodeName == 'mo'
    assert mml_2.childNodes[1].childNodes[0].nodeValue == '&#x2260;'  # 表示不等于符号
    assert mml_2.childNodes[2].nodeName == 'mi'
    assert mml_2.childNodes[2].childNodes[0].nodeValue == 'x'

    # 生成 1 ≥ x 的 MathML 表示
    mml_3 = mpp._print(Ge(1, x))
    assert len(mml_3.childNodes) == 3
    assert mml_3.childNodes[0].nodeName == 'mn'
    assert mml_3.childNodes[0].childNodes[0].nodeValue == '1'
    assert mml_3.childNodes[1].nodeName == 'mo'
    assert mml_3.childNodes[1].childNodes[0].nodeValue == '&#x2265;'  # 表示大于等于符号
    assert mml_3.childNodes[2].nodeName == 'mi'
    assert mml_3.childNodes[2].childNodes[0].nodeValue == 'x'

    # 生成 1 < x 的 MathML 表示
    mml_4 = mpp._print(Lt(1, x))
    assert len(mml_4.childNodes) == 3
    assert mml_4.childNodes[0].nodeName == 'mn'
    assert mml_4.childNodes[0].childNodes[0].nodeValue == '1'
    assert mml_4.childNodes[1].nodeName == 'mo'
    assert mml_4.childNodes[1].childNodes[0].nodeValue == '<'  # 表示小于符号
    assert mml_4.childNodes[2].nodeName == 'mi'
    assert mml_4.childNodes[2].childNodes[0].nodeValue == 'x'


# 定义测试函数，测试单个符号的 MathML 表示
def test_presentation_symbol():
    # 生成单个标识符 'x' 的 MathML 表示
    mml = mpp._print(x)
    assert mml.nodeName == 'mi'
    assert mml.childNodes[0].nodeValue == 'x'
    del mml

    # 生成符号 "x^2" 的 MathML 表示
    mml = mpp._print(Symbol("x^2"))
    assert mml.nodeName == 'msup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    # 生成符号 "x__2" 的 MathML 表示
    mml = mpp._print(Symbol("x__2"))
    assert mml.nodeName == 'msup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    # 生成符号 "x_2" 的 MathML 表示
    mml = mpp._print(Symbol("x_2"))
    assert mml.nodeName == 'msub'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    # 生成符号 "x^3_2" 的 MathML 表示
    mml = mpp._print(Symbol("x^3_2"))
    assert mml.nodeName == 'msubsup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    # 确保 mml 的第三个子节点是一个 'mi' 元素
    assert mml.childNodes[2].nodeName == 'mi'
    # 确保 mml 的第三个子节点的节点值是 '3'
    assert mml.childNodes[2].childNodes[0].nodeValue == '3'
    # 删除变量 mml，释放内存
    del mml

    # 使用 mpp._print 方法打印符号 "x__3_2" 的数学表达式
    mml = mpp._print(Symbol("x__3_2"))
    # 确保 mml 的节点名是 'msubsup'
    assert mml.nodeName == 'msubsup'
    # 确保 mml 的第一个子节点是 'mi' 元素，并且其节点值是 'x'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    # 确保 mml 的第二个子节点是 'mi' 元素，并且其节点值是 '2'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    # 确保 mml 的第三个子节点是 'mi' 元素，并且其节点值是 '3'
    assert mml.childNodes[2].nodeName == 'mi'
    assert mml.childNodes[2].childNodes[0].nodeValue == '3'
    # 删除变量 mml，释放内存
    del mml

    # 使用 mpp._print 方法打印符号 "x_2_a" 的数学表达式
    mml = mpp._print(Symbol("x_2_a"))
    # 确保 mml 的节点名是 'msub'
    assert mml.nodeName == 'msub'
    # 确保 mml 的第一个子节点是 'mi' 元素，并且其节点值是 'x'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    # 确保 mml 的第二个子节点是 'mrow' 元素
    assert mml.childNodes[1].nodeName == 'mrow'
    # 确保 mml 的第二个子节点的第一个子节点是 'mi' 元素，并且其节点值是 '2'
    assert mml.childNodes[1].childNodes[0].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    # 确保 mml 的第二个子节点的第二个子节点是 'mo' 元素，并且其节点值是 ' '
    assert mml.childNodes[1].childNodes[1].nodeName == 'mo'
    assert mml.childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    # 确保 mml 的第二个子节点的第三个子节点是 'mi' 元素，并且其节点值是 'a'
    assert mml.childNodes[1].childNodes[2].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    # 删除变量 mml，释放内存
    del mml

    # 使用 mpp._print 方法打印符号 "x^2^a" 的数学表达式
    mml = mpp._print(Symbol("x^2^a"))
    # 确保 mml 的节点名是 'msup'
    assert mml.nodeName == 'msup'
    # 确保 mml 的第一个子节点是 'mi' 元素，并且其节点值是 'x'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    # 确保 mml 的第二个子节点是 'mrow' 元素
    assert mml.childNodes[1].nodeName == 'mrow'
    # 确保 mml 的第二个子节点的第一个子节点是 'mi' 元素，并且其节点值是 '2'
    assert mml.childNodes[1].childNodes[0].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    # 确保 mml 的第二个子节点的第二个子节点是 'mo' 元素，并且其节点值是 ' '
    assert mml.childNodes[1].childNodes[1].nodeName == 'mo'
    assert mml.childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    # 确保 mml 的第二个子节点的第三个子节点是 'mi' 元素，并且其节点值是 'a'
    assert mml.childNodes[1].childNodes[2].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    # 删除变量 mml，释放内存
    del mml

    # 使用 mpp._print 方法打印符号 "x__2__a" 的数学表达式
    mml = mpp._print(Symbol("x__2__a"))
    # 确保 mml 的节点名是 'msup'
    assert mml.nodeName == 'msup'
    # 确保 mml 的第一个子节点是 'mi' 元素，并且其节点值是 'x'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    # 确保 mml 的第二个子节点是 'mrow' 元素
    assert mml.childNodes[1].nodeName == 'mrow'
    # 确保 mml 的第二个子节点的第一个子节点是 'mi' 元素，并且其节点值是 '2'
    assert mml.childNodes[1].childNodes[0].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    # 确保 mml 的第二个子节点的第二个子节点是 'mo' 元素，并且其节点值是 ' '
    assert mml.childNodes[1].childNodes[1].nodeName == 'mo'
    assert mml.childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    # 确保 mml 的第二个子节点的第三个子节点是 'mi' 元素，并且其节点值是 'a'
    assert mml.childNodes[1].childNodes[2].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    # 删除变量 mml，释放内存
    del mml
# 定义一个测试函数，用于测试数学表达式转换为 MathML 的功能
def test_presentation_mathml_greek():
    # 将符号 'alpha' 转换为 MathML 表示，并断言其节点名称为 'mi'
    mml = mpp._print(Symbol('alpha'))
    assert mml.nodeName == 'mi'
    # 断言 'alpha' 的 MathML 表示中第一个子节点的值为 Unicode 的小写希腊字母 'α'
    assert mml.childNodes[0].nodeValue == '\N{GREEK SMALL LETTER ALPHA}'

    # 依次测试各希腊字母的 MathML 表示是否符合预期
    assert mpp.doprint(Symbol('alpha')) == '<mi>&#945;</mi>'
    assert mpp.doprint(Symbol('beta')) == '<mi>&#946;</mi>'
    assert mpp.doprint(Symbol('gamma')) == '<mi>&#947;</mi>'
    assert mpp.doprint(Symbol('delta')) == '<mi>&#948;</mi>'
    assert mpp.doprint(Symbol('epsilon')) == '<mi>&#949;</mi>'
    assert mpp.doprint(Symbol('zeta')) == '<mi>&#950;</mi>'
    assert mpp.doprint(Symbol('eta')) == '<mi>&#951;</mi>'
    assert mpp.doprint(Symbol('theta')) == '<mi>&#952;</mi>'
    assert mpp.doprint(Symbol('iota')) == '<mi>&#953;</mi>'
    assert mpp.doprint(Symbol('kappa')) == '<mi>&#954;</mi>'
    assert mpp.doprint(Symbol('lambda')) == '<mi>&#955;</mi>'
    assert mpp.doprint(Symbol('mu')) == '<mi>&#956;</mi>'
    assert mpp.doprint(Symbol('nu')) == '<mi>&#957;</mi>'
    assert mpp.doprint(Symbol('xi')) == '<mi>&#958;</mi>'
    assert mpp.doprint(Symbol('omicron')) == '<mi>&#959;</mi>'
    assert mpp.doprint(Symbol('pi')) == '<mi>&#960;</mi>'
    assert mpp.doprint(Symbol('rho')) == '<mi>&#961;</mi>'
    assert mpp.doprint(Symbol('varsigma')) == '<mi>&#962;</mi>'
    assert mpp.doprint(Symbol('sigma')) == '<mi>&#963;</mi>'
    assert mpp.doprint(Symbol('tau')) == '<mi>&#964;</mi>'
    assert mpp.doprint(Symbol('upsilon')) == '<mi>&#965;</mi>'
    assert mpp.doprint(Symbol('phi')) == '<mi>&#966;</mi>'
    assert mpp.doprint(Symbol('chi')) == '<mi>&#967;</mi>'
    assert mpp.doprint(Symbol('psi')) == '<mi>&#968;</mi>'
    assert mpp.doprint(Symbol('omega')) == '<mi>&#969;</mi>'

    # 测试大写希腊字母的 MathML 表示是否符合预期
    assert mpp.doprint(Symbol('Alpha')) == '<mi>&#913;</mi>'
    assert mpp.doprint(Symbol('Beta')) == '<mi>&#914;</mi>'
    assert mpp.doprint(Symbol('Gamma')) == '<mi>&#915;</mi>'
    assert mpp.doprint(Symbol('Delta')) == '<mi>&#916;</mi>'
    assert mpp.doprint(Symbol('Epsilon')) == '<mi>&#917;</mi>'
    assert mpp.doprint(Symbol('Zeta')) == '<mi>&#918;</mi>'
    assert mpp.doprint(Symbol('Eta')) == '<mi>&#919;</mi>'
    assert mpp.doprint(Symbol('Theta')) == '<mi>&#920;</mi>'
    assert mpp.doprint(Symbol('Iota')) == '<mi>&#921;</mi>'
    assert mpp.doprint(Symbol('Kappa')) == '<mi>&#922;</mi>'
    assert mpp.doprint(Symbol('Lambda')) == '<mi>&#923;</mi>'
    assert mpp.doprint(Symbol('Mu')) == '<mi>&#924;</mi>'
    assert mpp.doprint(Symbol('Nu')) == '<mi>&#925;</mi>'
    assert mpp.doprint(Symbol('Xi')) == '<mi>&#926;</mi>'
    assert mpp.doprint(Symbol('Omicron')) == '<mi>&#927;</mi>'
    assert mpp.doprint(Symbol('Pi')) == '<mi>&#928;</mi>'
    assert mpp.doprint(Symbol('Rho')) == '<mi>&#929;</mi>'
    assert mpp.doprint(Symbol('Sigma')) == '<mi>&#931;</mi>'
    assert mpp.doprint(Symbol('Tau')) == '<mi>&#932;</mi>'
    assert mpp.doprint(Symbol('Upsilon')) == '<mi>&#933;</mi>'
    assert mpp.doprint(Symbol('Phi')) == '<mi>&#934;</mi>'
    # 使用 sympy.printing.pretty.pretty 将符号 'Chi' 转换为 MathML 表示，并断言结果是否为 '<mi>&#935;</mi>'
    assert mpp.doprint(Symbol('Chi')) == '<mi>&#935;</mi>'
    
    # 使用 sympy.printing.pretty.pretty 将符号 'Psi' 转换为 MathML 表示，并断言结果是否为 '<mi>&#936;</mi>'
    assert mpp.doprint(Symbol('Psi')) == '<mi>&#936;</mi>'
    
    # 使用 sympy.printing.pretty.pretty 将符号 'Omega' 转换为 MathML 表示，并断言结果是否为 '<mi>&#937;</mi>'
    assert mpp.doprint(Symbol('Omega')) == '<mi>&#937;</mi>'
# 定义测试函数，用于测试 MathMLPresentationPrinter 类中不同方法的输出结果
def test_presentation_mathml_order():
    # 定义数学表达式
    expr = x**3 + x**2*y + 3*x*y**3 + y**4

    # 创建 MathMLPresentationPrinter 对象，设置排序顺序为 'lex'（字典序）
    mp = MathMLPresentationPrinter({'order': 'lex'})
    # 打印数学表达式为 MathML 格式
    mml = mp._print(expr)
    # 断言：检查第一个子节点的名称是否为 'msup'
    assert mml.childNodes[0].nodeName == 'msup'
    # 断言：检查第一个子节点的第一个子节点的值是否为 'x'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    # 断言：检查第一个子节点的第二个子节点的值是否为 '3'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '3'

    # 断言：检查第七个子节点的名称是否为 'msup'
    assert mml.childNodes[6].nodeName == 'msup'
    # 断言：检查第七个子节点的第一个子节点的值是否为 'y'
    assert mml.childNodes[6].childNodes[0].childNodes[0].nodeValue == 'y'
    # 断言：检查第七个子节点的第二个子节点的值是否为 '4'
    assert mml.childNodes[6].childNodes[1].childNodes[0].nodeValue == '4'

    # 创建 MathMLPresentationPrinter 对象，设置排序顺序为 'rev-lex'（逆字典序）
    mp = MathMLPresentationPrinter({'order': 'rev-lex'})
    # 再次打印数学表达式为 MathML 格式
    mml = mp._print(expr)

    # 断言：检查第一个子节点的名称是否为 'msup'
    assert mml.childNodes[0].nodeName == 'msup'
    # 断言：检查第一个子节点的第一个子节点的值是否为 'y'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'y'
    # 断言：检查第一个子节点的第二个子节点的值是否为 '4'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '4'

    # 断言：检查第七个子节点的名称是否为 'msup'
    assert mml.childNodes[6].nodeName == 'msup'
    # 断言：检查第七个子节点的第一个子节点的值是否为 'x'
    assert mml.childNodes[6].childNodes[0].childNodes[0].nodeValue == 'x'
    # 断言：检查第七个子节点的第二个子节点的值是否为 '3'


# 定义测试函数，用于测试 MathMLPresentationPrinter 类中处理区间的方法
def test_print_intervals():
    # 定义符号变量 a
    a = Symbol('a', real=True)
    # 断言：检查打印区间 [0, a] 的 MathML 输出是否正确
    assert mpp.doprint(Interval(0, a)) == \
        '<mrow><mfenced close="]" open="["><mn>0</mn><mi>a</mi></mfenced></mrow>'
    # 断言：检查打印区间 (0, a] 的 MathML 输出是否正确
    assert mpp.doprint(Interval(0, a, False, False)) == \
        '<mrow><mfenced close="]" open="["><mn>0</mn><mi>a</mi></mfenced></mrow>'
    # 断言：检查打印区间 (0, a) 的 MathML 输出是否正确
    assert mpp.doprint(Interval(0, a, True, False)) == \
        '<mrow><mfenced close="]" open="("><mn>0</mn><mi>a</mi></mfenced></mrow>'
    # 断言：检查打印区间 [0, a) 的 MathML 输出是否正确
    assert mpp.doprint(Interval(0, a, False, True)) == \
        '<mrow><mfenced close=")" open="["><mn>0</mn><mi>a</mi></mfenced></mrow>'
    # 断言：检查打印区间 (0, a) 的 MathML 输出是否正确
    assert mpp.doprint(Interval(0, a, True, True)) == \
        '<mrow><mfenced close=")" open="("><mn>0</mn><mi>a</mi></mfenced></mrow>'


# 定义测试函数，用于测试 MathMLPresentationPrinter 类中处理元组的方法
def test_print_tuples():
    # 断言：检查打印元组 (0,) 的 MathML 输出是否正确
    assert mpp.doprint(Tuple(0,)) == \
        '<mrow><mfenced><mn>0</mn></mfenced></mrow>'
    # 断言：检查打印元组 (0, a) 的 MathML 输出是否正确
    assert mpp.doprint(Tuple(0, a)) == \
        '<mrow><mfenced><mn>0</mn><mi>a</mi></mfenced></mrow>'
    # 断言：检查打印元组 (0, a, a) 的 MathML 输出是否正确
    assert mpp.doprint(Tuple(0, a, a)) == \
        '<mrow><mfenced><mn>0</mn><mi>a</mi><mi>a</mi></mfenced></mrow>'
    # 断言：检查打印元组 (0, 1, 2, 3, 4) 的 MathML 输出是否正确
    assert mpp.doprint(Tuple(0, 1, 2, 3, 4)) == \
        '<mrow><mfenced><mn>0</mn><mn>1</mn><mn>2</mn><mn>3</mn><mn>4</mn></mfenced></mrow>'
    # 断言：检查打印元组 (0, 1, (2, 3, 4)) 的 MathML 输出是否正确
    assert mpp.doprint(Tuple(0, 1, Tuple(2, 3, 4))) == \
        '<mrow><mfenced><mn>0</mn><mn>1</mn><mrow><mfenced><mn>2</mn><mn>3'\
        '</mn><mn>4</mn></mfenced></mrow></mfenced></mrow>'


# 定义测试函数，用于测试 MathMLPresentationPrinter 类中处理实部和虚部的方法
def test_print_re_im():
    # 断言：检查打印实部 re(x) 的 MathML 输出是否正确
    assert mpp.doprint(re(x)) == \
        '<mrow><mi mathvariant="fraktur">R</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言：检查打印虚部 im(x) 的 MathML 输出是否正确
    assert mpp.doprint(im(x)) == \
        '<mrow><mi mathvariant="fraktur">I</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言：检查打印实部 re(x + 1) 的 MathML 输出是否正确
    assert mpp.doprint(re(x + 1)) == \
        '<mrow><mrow><mi mathvariant="fraktur">R</mi><mfenced><mi>x</mi>'\
        '</mfenced></mrow><mo>+</mo><mn>1</mn></mrow>'
    # 使用断言来验证调用 mpp.doprint(im(x + 1)) 后的输出是否等于指定的数学表达式字符串
    assert mpp.doprint(im(x + 1)) == \
        '<mrow><mi mathvariant="fraktur">I</mi><mfenced><mi>x</mi></mfenced></mrow>'
# 定义测试函数，用于测试打印绝对值表达式的 MathML 表示
def test_print_Abs():
    # 断言绝对值函数对单个变量 x 的输出
    assert mpp.doprint(Abs(x)) == \
        '<mrow><mfenced close="|" open="|"><mi>x</mi></mfenced></mrow>'
    # 断言绝对值函数对表达式 x + 1 的输出
    assert mpp.doprint(Abs(x + 1)) == \
        '<mrow><mfenced close="|" open="|"><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow></mfenced></mrow>'


# 定义测试函数，用于测试打印矩阵行列式的 MathML 表示
def test_print_Determinant():
    # 断言行列式函数对 2x2 矩阵 [[1, 2], [3, 4]] 的输出
    assert mpp.doprint(Determinant(Matrix([[1, 2], [3, 4]]))) == \
        '<mrow><mfenced close="|" open="|"><mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable></mfenced></mfenced></mrow>'


# 定义测试函数，用于测试设置输出格式时的异常情况
def test_presentation_settings():
    # 断言设置输出格式为 presentation 时使用无效方法会引发 TypeError
    raises(TypeError, lambda: mathml(x, printer='presentation',
                                     method="garbage"))


# 定义测试函数，用于测试打印不同数学域的 MathML 表示
def test_print_domains():
    from sympy.sets import Integers, Naturals, Naturals0, Reals, Complexes

    # 断言打印复数集合的 MathML 表示
    assert mpp.doprint(Complexes) == '<mi mathvariant="normal">&#x2102;</mi>'
    # 断言打印整数集合的 MathML 表示
    assert mpp.doprint(Integers) == '<mi mathvariant="normal">&#x2124;</mi>'
    # 断言打印自然数集合的 MathML 表示
    assert mpp.doprint(Naturals) == '<mi mathvariant="normal">&#x2115;</mi>'
    # 断言打印非负整数集合的 MathML 表示
    assert mpp.doprint(Naturals0) == \
        '<msub><mi mathvariant="normal">&#x2115;</mi><mn>0</mn></msub>'
    # 断言打印实数集合的 MathML 表示
    assert mpp.doprint(Reals) == '<mi mathvariant="normal">&#x211D;</mi>'


# 定义测试函数，用于测试打印带负号的表达式的 MathML 表示
def test_print_expression_with_minus():
    # 断言打印负号表达式 -x 的 MathML 表示
    assert mpp.doprint(-x) == '<mrow><mo>-</mo><mi>x</mi></mrow>'
    # 断言打印带有负号的分数表达式 -x/y 的 MathML 表示
    assert mpp.doprint(-x/y) == \
        '<mrow><mo>-</mo><mfrac><mi>x</mi><mi>y</mi></mfrac></mrow>'
    # 断言打印有理数 -1/2 的 MathML 表示
    assert mpp.doprint(-Rational(1, 2)) == \
        '<mrow><mo>-</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mrow>'


# 定义测试函数，用于测试打印自定义的关联运算符类的 MathML 表示
def test_print_AssocOp():
    from sympy.core.operations import AssocOp

    # 定义一个测试用的关联运算符类 TestAssocOp
    class TestAssocOp(AssocOp):
        identity = 0

    # 创建一个 TestAssocOp 实例并断言其 MathML 表示
    expr = TestAssocOp(1, 2)
    assert mpp.doprint(expr) == \
        '<mrow><mi>testassocop</mi><mn>1</mn><mn>2</mn></mrow>'


# 定义测试函数，用于测试打印基本类的 MathML 表示
def test_print_basic():
    expr = Basic(S(1), S(2))
    # 断言打印基本类的 MathML 表示
    assert mpp.doprint(expr) == \
        '<mrow><mi>basic</mi><mfenced><mn>1</mn><mn>2</mn></mfenced></mrow>'
    # 断言打印基本类的 MathML 表示（不同的打印器）
    assert mp.doprint(expr) == '<basic><cn>1</cn><cn>2</cn></basic>'


# 定义测试函数，用于测试打印矩阵表达式的 MathML 表示，并指定不同的矩阵分隔符
def test_mat_delim_print():
    expr = Matrix([[1, 2], [3, 4]])
    # 断言使用 '[' 作为矩阵分隔符时的 MathML 表示
    assert mathml(expr, printer='presentation', mat_delim='[') == \
        '<mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd><mtd>'\
        '<mn>2</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn>'\
        '</mtd></mtr></mtable></mfenced>'
    # 断言使用 '(' 作为矩阵分隔符时的 MathML 表示
    assert mathml(expr, printer='presentation', mat_delim='(') == \
        '<mfenced><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd>'\
        '</mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable></mfenced>'
    # 断言不使用分隔符时的 MathML 表示
    assert mathml(expr, printer='presentation', mat_delim='') == \
        '<mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr>'\
        '<mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable>'


# 定义测试函数，用于测试对数函数的 MathML 表示
def test_ln_notation_print():
    expr = log(x)
    # 检查使用 mathml 函数生成的数学表达式是否符合预期输出
    assert mathml(expr, printer='presentation') == \
        '<mrow><mi>log</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 检查关闭 ln 表示法后，使用 mathml 函数生成的数学表达式是否符合预期输出
    assert mathml(expr, printer='presentation', ln_notation=False) == \
        '<mrow><mi>log</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 检查开启 ln 表示法后，使用 mathml 函数生成的数学表达式是否符合预期输出
    assert mathml(expr, printer='presentation', ln_notation=True) == \
        '<mrow><mi>ln</mi><mfenced><mi>x</mi></mfenced></mrow>'
# 定义测试函数，用于测试数学表达式的多种打印格式
def test_mul_symbol_print():
    # 创建数学表达式 x * y
    expr = x * y
    # 断言使用 presentation 打印器生成的 MathML 结果与预期值相等
    assert mathml(expr, printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi></mrow>'
    # 断言使用 presentation 打印器，并禁用乘号的生成，生成的 MathML 结果与预期值相等
    assert mathml(expr, printer='presentation', mul_symbol=None) == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi></mrow>'
    # 断言使用 presentation 打印器，并指定使用点号作为乘号，生成的 MathML 结果与预期值相等
    assert mathml(expr, printer='presentation', mul_symbol='dot') == \
        '<mrow><mi>x</mi><mo>&#xB7;</mo><mi>y</mi></mrow>'
    # 断言使用 presentation 打印器，并指定使用小圆点作为乘号，生成的 MathML 结果与预期值相等
    assert mathml(expr, printer='presentation', mul_symbol='ldot') == \
        '<mrow><mi>x</mi><mo>&#x2024;</mo><mi>y</mi></mrow>'
    # 断言使用 presentation 打印器，并指定使用乘号符号作为乘号，生成的 MathML 结果与预期值相等
    assert mathml(expr, printer='presentation', mul_symbol='times') == \
        '<mrow><mi>x</mi><mo>&#xD7;</mo><mi>y</mi></mrow>'
    '<mrow><mfenced close="}" open="{"><mn>1</mn><mn>3</mn><mi>x</mi>'\
    '</mfenced><mo>&#x222A;</mo><mfenced close="}" open="{"><mn>2</mn>'\
    '<mn>4</mn><mi>y</mi></mfenced></mrow>'
    # 第一个断言语句，测试 Intersection 函数的输出是否符合预期
    assert prntr(Intersection(f1, f2, evaluate=False)) == \
    '<mrow><mfenced close="}" open="{"><mn>1</mn><mn>3</mn><mi>x</mi>'\
    '</mfenced><mo>&#x2229;</mo><mfenced close="}" open="{"><mn>2</mn>'\
    '<mn>4</mn><mi>y</mi></mfenced></mrow>'
    # 第二个断言语句，测试 Complement 函数的输出是否符合预期
    assert prntr(Complement(f1, f2, evaluate=False)) == \
    '<mrow><mfenced close="}" open="{"><mn>1</mn><mn>3</mn><mi>x</mi>'\
    '</mfenced><mo>&#x2216;</mo><mfenced close="}" open="{"><mn>2</mn>'\
    '<mn>4</mn><mi>y</mi></mfenced></mrow>'
    # 第三个断言语句，测试 SymmetricDifference 函数的输出是否符合预期
    assert prntr(SymmetricDifference(f1, f2, evaluate=False)) == \
    '<mrow><mfenced close="}" open="{"><mn>1</mn><mn>3</mn><mi>x</mi>'\
    '</mfenced><mo>&#x2206;</mo><mfenced close="}" open="{"><mn>2</mn>'\
    '<mn>4</mn><mi>y</mi></mfenced></mrow>'

    A = FiniteSet(a)
    C = FiniteSet(c)
    D = FiniteSet(d)

    U1 = Union(C, D, evaluate=False)
    I1 = Intersection(C, D, evaluate=False)
    C1 = Complement(C, D, evaluate=False)
    D1 = SymmetricDifference(C, D, evaluate=False)
    # XXX ProductSet does not support evaluate keyword
    P1 = ProductSet(C, D)

    # 第一个断言语句，测试 Union 函数的输出是否符合预期
    assert prntr(Union(A, I1, evaluate=False)) == \
        '<mrow><mfenced close="}" open="{"><mi>a</mi></mfenced>' \
        '<mo>&#x222A;</mo><mfenced><mrow><mfenced close="}" open="{">' \
        '<mi>c</mi></mfenced><mo>&#x2229;</mo><mfenced close="}" open="{">' \
        '<mi>d</mi></mfenced></mrow></mfenced></mrow>'
    # 第二个断言语句，测试 Intersection 函数的输出是否符合预期
    assert prntr(Intersection(A, C1, evaluate=False)) == \
        '<mrow><mfenced close="}" open="{"><mi>a</mi></mfenced>' \
        '<mo>&#x2229;</mo><mfenced><mrow><mfenced close="}" open="{">' \
        '<mi>c</mi></mfenced><mo>&#x2216;</mo><mfenced close="}" open="{">' \
        '<mi>d</mi></mfenced></mrow></mfenced></mrow>'
    # 第三个断言语句，测试 Complement 函数的输出是否符合预期
    assert prntr(Complement(A, D1, evaluate=False)) == \
        '<mrow><mfenced close="}" open="{"><mi>a</mi></mfenced>' \
        '<mo>&#x2216;</mo><mfenced><mrow><mfenced close="}" open="{">' \
        '<mi>c</mi></mfenced><mo>&#x2206;</mo><mfenced close="}" open="{">' \
        '<mi>d</mi></mfenced></mrow></mfenced></mrow>'
    # 第四个断言语句，测试 SymmetricDifference 函数的输出是否符合预期
    assert prntr(SymmetricDifference(A, P1, evaluate=False)) == \
        '<mrow><mfenced close="}" open="{"><mi>a</mi></mfenced>' \
        '<mo>&#x2206;</mo><mfenced><mrow><mfenced close="}" open="{">' \
        '<mi>c</mi></mfenced><mo>&#x00d7;</mo><mfenced close="}" open="{">' \
        '<mi>d</mi></mfenced></mrow></mfenced></mrow>'
    # 第五个断言语句，测试 ProductSet 函数的输出是否符合预期
    assert prntr(ProductSet(A, U1)) == \
        '<mrow><mfenced close="}" open="{"><mi>a</mi></mfenced>' \
        '<mo>&#x00d7;</mo><mfenced><mrow><mfenced close="}" open="{">' \
        '<mi>c</mi></mfenced><mo>&#x222A;</mo><mfenced close="}" open="{">' \
        '<mi>d</mi></mfenced></mrow></mfenced></mrow>'
# 定义测试函数 test_print_logic
def test_print_logic():
    # 断言逻辑运算 And(x, y) 的打印结果
    assert mpp.doprint(And(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x2227;</mo><mi>y</mi></mrow>'
    # 断言逻辑运算 Or(x, y) 的打印结果
    assert mpp.doprint(Or(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x2228;</mo><mi>y</mi></mrow>'
    # 断言逻辑运算 Xor(x, y) 的打印结果
    assert mpp.doprint(Xor(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x22BB;</mo><mi>y</mi></mrow>'
    # 断言逻辑运算 Implies(x, y) 的打印结果
    assert mpp.doprint(Implies(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x21D2;</mo><mi>y</mi></mrow>'
    # 断言逻辑运算 Equivalent(x, y) 的打印结果
    assert mpp.doprint(Equivalent(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x21D4;</mo><mi>y</mi></mrow>'

    # 断言复合逻辑表达式 And(Eq(x, y), x > 4) 的打印结果
    assert mpp.doprint(And(Eq(x, y), x > 4)) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mi>y</mi></mrow><mo>&#x2227;</mo>'\
        '<mrow><mi>x</mi><mo>></mo><mn>4</mn></mrow></mrow>'
    # 断言复合逻辑表达式 And(Eq(x, 3), y < 3, x > y + 1) 的打印结果
    assert mpp.doprint(And(Eq(x, 3), y < 3, x > y + 1)) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mn>3</mn></mrow><mo>&#x2227;</mo>'\
        '<mrow><mi>x</mi><mo>></mo><mrow><mi>y</mi><mo>+</mo><mn>1</mn></mrow>'\
        '</mrow><mo>&#x2227;</mo><mrow><mi>y</mi><mo><</mo><mn>3</mn></mrow></mrow>'
    # 断言复合逻辑表达式 Or(Eq(x, y), x > 4) 的打印结果
    assert mpp.doprint(Or(Eq(x, y), x > 4)) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mi>y</mi></mrow><mo>&#x2228;</mo>'\
        '<mrow><mi>x</mi><mo>></mo><mn>4</mn></mrow></mrow>'
    # 断言复合逻辑表达式 And(Eq(x, 3), Or(y < 3, x > y + 1)) 的打印结果
    assert mpp.doprint(And(Eq(x, 3), Or(y < 3, x > y + 1))) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mn>3</mn></mrow><mo>&#x2227;</mo>'\
        '<mfenced><mrow><mrow><mi>x</mi><mo>></mo><mrow><mi>y</mi><mo>+</mo>'\
        '<mn>1</mn></mrow></mrow><mo>&#x2228;</mo><mrow><mi>y</mi><mo><</mo>'\
        '<mn>3</mn></mrow></mrow></mfenced></mrow>'

    # 断言逻辑非运算 Not(x) 的打印结果
    assert mpp.doprint(Not(x)) == '<mrow><mo>&#xAC;</mo><mi>x</mi></mrow>'
    # 断言逻辑非运算 Not(And(x, y)) 的打印结果
    assert mpp.doprint(Not(And(x, y))) == \
        '<mrow><mo>&#xAC;</mo><mfenced><mrow><mi>x</mi><mo>&#x2227;</mo>'\
        '<mi>y</mi></mrow></mfenced></mrow>'


# 定义测试函数 test_root_notation_print
def test_root_notation_print():
    # 断言 x 的分数幂（根式表示）的 MathML 输出
    assert mathml(x**(S.One/3), printer='presentation') == \
        '<mroot><mi>x</mi><mn>3</mn></mroot>'
    # 断言 x 的分数幂（分数幂表示）的 MathML 输出
    assert mathml(x**(S.One/3), printer='presentation', root_notation=False) ==\
        '<msup><mi>x</mi><mfrac><mn>1</mn><mn>3</mn></mfrac></msup>'
    # 断言 x 的分数幂（根式表示）的 content MathML 输出
    assert mathml(x**(S.One/3), printer='content') == \
        '<apply><root/><degree><cn>3</cn></degree><ci>x</ci></apply>'
    # 断言 x 的分数幂（分数幂表示）的 content MathML 输出
    assert mathml(x**(S.One/3), printer='content', root_notation=False) == \
        '<apply><power/><ci>x</ci><apply><divide/><cn>1</cn><cn>3</cn></apply></apply>'
    # 断言 x 的负分数幂（根式表示）的 MathML 输出
    assert mathml(x**(Rational(-1, 3)), printer='presentation') == \
        '<mfrac><mn>1</mn><mroot><mi>x</mi><mn>3</mn></mroot></mfrac>'
    # 断言 x 的负分数幂（分数幂表示）的 MathML 输出
    assert mathml(x**(Rational(-1, 3)), printer='presentation', root_notation=False) \
        == '<mfrac><mn>1</mn><msup><mi>x</mi><mfrac><mn>1</mn><mn>3</mn></mfrac></msup></mfrac>'


# 定义测试函数 test_fold_frac_powers_print
def test_fold_frac_powers_print():
    # 创建表达式 x 的分数幂（幂表示）的 MathML 输出
    expr = x ** Rational(5, 2)
    assert mathml(expr, printer='presentation') == \
        '<msup><mi>x</mi><mfrac><mn>5</mn><mn>2</mn></mfrac></msup>'
    # 断言：使用 mathml 函数生成数学表达式的 MathML 字符串，并与预期的字符串进行比较
    assert mathml(expr, printer='presentation', fold_frac_powers=True) == \
        '<msup><mi>x</mi><mfrac bevelled="true"><mn>5</mn><mn>2</mn></mfrac></msup>'
    # 断言：使用 mathml 函数生成数学表达式的 MathML 字符串，并与预期的字符串进行比较（不折叠分式指数）
    assert mathml(expr, printer='presentation', fold_frac_powers=False) == \
        '<msup><mi>x</mi><mfrac><mn>5</mn><mn>2</mn></mfrac></msup>'
def test_fold_short_frac_print():
    # 创建有理数表达式 Rational(2, 5)
    expr = Rational(2, 5)
    # 使用 presentation 模式生成数学表达式的 MathML，并断言其结果
    assert mathml(expr, printer='presentation') == \
        '<mfrac><mn>2</mn><mn>5</mn></mfrac>'
    # 使用 fold_short_frac=True 参数生成简化的有理数表达式 MathML
    assert mathml(expr, printer='presentation', fold_short_frac=True) == \
        '<mfrac bevelled="true"><mn>2</mn><mn>5</mn></mfrac>'
    # 使用 fold_short_frac=False 参数生成未简化的有理数表达式 MathML
    assert mathml(expr, printer='presentation', fold_short_frac=False) == \
        '<mfrac><mn>2</mn><mn>5</mn></mfrac>'


def test_print_factorials():
    # 计算并打印阶乘 factorial(x) 的 MathML 表示
    assert mpp.doprint(factorial(x)) == '<mrow><mi>x</mi><mo>!</mo></mrow>'
    # 计算并打印阶乘 factorial(x + 1) 的 MathML 表示
    assert mpp.doprint(factorial(x + 1)) == \
        '<mrow><mfenced><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow></mfenced><mo>!</mo></mrow>'
    # 计算并打印双阶乘 factorial2(x) 的 MathML 表示
    assert mpp.doprint(factorial2(x)) == '<mrow><mi>x</mi><mo>!!</mo></mrow>'
    # 计算并打印双阶乘 factorial2(x + 1) 的 MathML 表示
    assert mpp.doprint(factorial2(x + 1)) == \
        '<mrow><mfenced><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow></mfenced><mo>!!</mo></mrow>'
    # 计算并打印二项式系数 binomial(x, y) 的 MathML 表示
    assert mpp.doprint(binomial(x, y)) == \
        '<mfenced><mfrac linethickness="0"><mi>x</mi><mi>y</mi></mfrac></mfenced>'
    # 计算并打印二项式系数 binomial(4, x + y) 的 MathML 表示
    assert mpp.doprint(binomial(4, x + y)) == \
        '<mfenced><mfrac linethickness="0"><mn>4</mn><mrow><mi>x</mi>'\
        '<mo>+</mo><mi>y</mi></mrow></mfrac></mfenced>'


def test_print_floor():
    # 计算并打印 floor(x) 的 MathML 表示
    expr = floor(x)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mfenced close="&#8971;" open="&#8970;"><mi>x</mi></mfenced></mrow>'


def test_print_ceiling():
    # 计算并打印 ceiling(x) 的 MathML 表示
    expr = ceiling(x)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mfenced close="&#8969;" open="&#8968;"><mi>x</mi></mfenced></mrow>'


def test_print_Lambda():
    # 创建 Lambda 表达式 Lambda(x, x+1) 并打印其 MathML 表示
    expr = Lambda(x, x+1)
    assert mathml(expr, printer='presentation') == \
        '<mfenced><mrow><mi>x</mi><mo>&#x21A6;</mo><mrow><mi>x</mi><mo>+</mo>'\
        '<mn>1</mn></mrow></mrow></mfenced>'
    # 创建 Lambda 表达式 Lambda((x, y), x + y) 并打印其 MathML 表示
    assert mathml(expr, printer='presentation') == \
        '<mfenced><mrow><mrow><mfenced><mi>x</mi><mi>y</mi></mfenced></mrow>'\
        '<mo>&#x21A6;</mo><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow></mrow></mfenced>'


def test_print_conjugate():
    # 计算并打印复数的共轭 conjugate(x) 的 MathML 表示
    assert mpp.doprint(conjugate(x)) == \
        '<menclose notation="top"><mi>x</mi></menclose>'
    # 计算并打印带复数的共轭 conjugate(x + 1) 的 MathML 表示
    assert mpp.doprint(conjugate(x + 1)) == \
        '<mrow><menclose notation="top"><mi>x</mi></menclose><mo>+</mo><mn>1</mn></mrow>'


def test_print_AccumBounds():
    # 创建并打印积累边界 AccumBounds(0, 1) 的 MathML 表示
    a = Symbol('a', real=True)
    assert mpp.doprint(AccumBounds(0, 1)) == '<mfenced close="&#10217;" open="&#10216;"><mn>0</mn><mn>1</mn></mfenced>'
    # 创建并打印积累边界 AccumBounds(0, a) 的 MathML 表示
    assert mpp.doprint(AccumBounds(0, a)) == '<mfenced close="&#10217;" open="&#10216;"><mn>0</mn><mi>a</mi></mfenced>'
    # 创建并打印积累边界 AccumBounds(a + 1, a + 2) 的 MathML 表示
    assert mpp.doprint(AccumBounds(a + 1, a + 2)) == '<mfenced close="&#10217;" open="&#10216;"><mrow><mi>a</mi><mo>+</mo><mn>1</mn></mrow><mrow><mi>a</mi><mo>+</mo><mn>2</mn></mrow></mfenced>'


def test_print_Float():
    # 计算并打印浮点数 Float(1e100) 的 MathML 表示
    assert mpp.doprint(Float(1e100)) == '<mrow><mn>1.0</mn><mo>&#xB7;</mo><msup><mn>10</mn><mn>100</mn></msup></mrow>'
    # 对象 mpp 执行 doprint 方法，将浮点数 1e-100 打印为 MathML 格式的字符串
    assert mpp.doprint(Float(1e-100)) == '<mrow><mn>1.0</mn><mo>&#xB7;</mo><msup><mn>10</mn><mn>-100</mn></msup></mrow>'
    # 对象 mpp 执行 doprint 方法，将浮点数 -1e100 打印为 MathML 格式的字符串
    assert mpp.doprint(Float(-1e100)) == '<mrow><mn>-1.0</mn><mo>&#xB7;</mo><msup><mn>10</mn><mn>100</mn></msup></mrow>'
    # 对象 mpp 执行 doprint 方法，将浮点数 1.0*oo（正无穷大）打印为 MathML 格式的字符串
    assert mpp.doprint(Float(1.0*oo)) == '<mi>&#x221E;</mi>'
    # 对象 mpp 执行 doprint 方法，将浮点数 -1.0*oo（负无穷大）打印为 MathML 格式的字符串
    assert mpp.doprint(Float(-1.0*oo)) == '<mrow><mo>-</mo><mi>&#x221E;</mi></mrow>'
# 测试打印不同函数的 MathML 表达式
def test_print_different_functions():
    # 断言 gamma(x) 的 MathML 表达式是否等于 '<mrow><mi>&#x393;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    assert mpp.doprint(gamma(x)) == '<mrow><mi>&#x393;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 lowergamma(x, y) 的 MathML 表达式是否等于 '<mrow><mi>&#x3B3;</mi><mfenced><mi>x</mi><mi>y</mi></mfenced></mrow>'
    assert mpp.doprint(lowergamma(x, y)) == '<mrow><mi>&#x3B3;</mi><mfenced><mi>x</mi><mi>y</mi></mfenced></mrow>'
    # 断言 uppergamma(x, y) 的 MathML 表达式是否等于 '<mrow><mi>&#x393;</mi><mfenced><mi>x</mi><mi>y</mi></mfenced></mrow>'
    assert mpp.doprint(uppergamma(x, y)) == '<mrow><mi>&#x393;</mi><mfenced><mi>x</mi><mi>y</mi></mfenced></mrow>'
    # 断言 zeta(x) 的 MathML 表达式是否等于 '<mrow><mi>&#x3B6;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    assert mpp.doprint(zeta(x)) == '<mrow><mi>&#x3B6;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 zeta(x, y) 的 MathML 表达式是否等于 '<mrow><mi>&#x3B6;</mi><mfenced><mi>x</mi><mi>y</mi></mfenced></mrow>'
    assert mpp.doprint(zeta(x, y)) == '<mrow><mi>&#x3B6;</mi><mfenced><mi>x</mi><mi>y</mi></mfenced></mrow>'
    # 断言 dirichlet_eta(x) 的 MathML 表达式是否等于 '<mrow><mi>&#x3B7;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    assert mpp.doprint(dirichlet_eta(x)) == '<mrow><mi>&#x3B7;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 elliptic_k(x) 的 MathML 表达式是否等于 '<mrow><mi>&#x39A;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    assert mpp.doprint(elliptic_k(x)) == '<mrow><mi>&#x39A;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 totient(x) 的 MathML 表达式是否等于 '<mrow><mi>&#x3D5;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    assert mpp.doprint(totient(x)) == '<mrow><mi>&#x3D5;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 reduced_totient(x) 的 MathML 表达式是否等于 '<mrow><mi>&#x3BB;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    assert mpp.doprint(reduced_totient(x)) == '<mrow><mi>&#x3BB;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 primenu(x) 的 MathML 表达式是否等于 '<mrow><mi>&#x3BD;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    assert mpp.doprint(primenu(x)) == '<mrow><mi>&#x3BD;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 primeomega(x) 的 MathML 表达式是否等于 '<mrow><mi>&#x3A9;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    assert mpp.doprint(primeomega(x)) == '<mrow><mi>&#x3A9;</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 fresnels(x) 的 MathML 表达式是否等于 '<mrow><mi>S</mi><mfenced><mi>x</mi></mfenced></mrow>'
    assert mpp.doprint(fresnels(x)) == '<mrow><mi>S</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 fresnelc(x) 的 MathML 表达式是否等于 '<mrow><mi>C</mi><mfenced><mi>x</mi></mfenced></mrow>'
    assert mpp.doprint(fresnelc(x)) == '<mrow><mi>C</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 Heaviside(x) 的 MathML 表达式是否等于 '<mrow><mi>&#x398;</mi><mfenced><mi>x</mi><mfrac><mn>1</mn><mn>2</mn></mfrac></mfenced></mrow>'
    assert mpp.doprint(Heaviside(x)) == '<mrow><mi>&#x398;</mi><mfenced><mi>x</mi><mfrac><mn>1</mn><mn>2</mn></mfrac></mfenced></mrow>'


# 测试打印内置数学函数的 MathML 表达式
def test_mathml_builtins():
    # 断言 None 的 MathML 表达式是否等于 '<mi>None</mi>'
    assert mpp.doprint(None) == '<mi>None</mi>'
    # 断言 True 的 MathML 表达式是否等于 '<mi>True</mi>'
    assert mpp.doprint(true) == '<mi>True</mi>'
    # 断言 False 的 MathML 表达式是否等于 '<mi>False</mi>'
    assert mpp.doprint(false) == '<mi>False</mi>'


# 测试打印 Range 对象的 MathML 表达式
def test_mathml_Range():
    # 断言 Range(1, 51) 的 MathML 表达式是否等于 '<mfenced close="}" open="{"><mn>1</mn><mn>2</mn><mi>&#8230;</mi><mn>50</mn></mfenced>'
    assert mpp.doprint(Range(1, 51)) == '<mfenced close="}" open="{"><mn>1</mn><mn>2</mn><mi>&#8230;</mi><mn>50</mn></mfenced>'
    # 断言 Range(1, 4) 的 MathML 表达式是否等于 '<mfenced close="}" open="{"><mn>1</mn><mn>2</mn><mn>3</mn></mfenced>'
    assert mpp.doprint(Range(1, 4)) == '<mfenced close="}" open="{"><mn>1</mn><mn>2</mn><mn>3</mn></mfenced>'
    # 断言 Range(0, 3, 1) 的 MathML 表达式是否等于 '<mfenced close="}" open="{"><mn>0</mn><mn>1</mn><mn>2</mn></mfenced>'
    assert mpp.doprint(Range(0, 3, 1)) == '<mfenced close="}" open="{"><mn>0</mn><mn>1</mn><mn>2</mn></mfenced>'
    # 断言 Range(0, 30, 1) 的 MathML 表达式是否等于 '<mfenced close="}" open="{"><mn>0</mn><mn>1</mn><mi>&#8230;</mi><mn>29</mn></mfenced>'
    assert mpp.doprint(Range(0, 30, 1)) == '<mfenced close="}" open="{"><mn>0</mn><mn>1</mn><mi>&#8230;</mi><mn>29</mn></mfenced>'
    # 断言 Range(30, 1, -1) 的 MathML 表达式是否等于 '<mfenced close="}" open="{"><mn>30</mn><mn>29</mn><mi>&#8230;</mi><mn>2</mn></mfenced>'
    assert mpp.doprint(Range(30, 1, -1)) == '<mfenced close="}" open="{"><mn>30</mn><mn>29</mn><mi>&#8230;</mi><mn>2</mn></mfenced>'
    # 断言 Range(0, oo, 2)
    # 断言：验证调用 mpp 对象的 doprint 方法对 exp(1) + exp(2) 的输出结果
    assert mpp.doprint(exp(1) + exp(2)) == \
        '<mrow><mi>&ExponentialE;</mi><mo>+</mo><msup><mi>&ExponentialE;</mi><mn>2</mn></msup></mrow>'
def test_print_MinMax():
    # 测试打印最小值函数 Min(x, y)
    assert mpp.doprint(Min(x, y)) == \
        '<mrow><mo>min</mo><mfenced><mi>x</mi><mi>y</mi></mfenced></mrow>'
    # 测试打印最小值函数 Min(x, 2, x**3)
    assert mpp.doprint(Min(x, 2, x**3)) == \
        '<mrow><mo>min</mo><mfenced><mn>2</mn><mi>x</mi><msup><mi>x</mi>'\
        '<mn>3</mn></msup></mfenced></mrow>'
    # 测试打印最大值函数 Max(x, y)
    assert mpp.doprint(Max(x, y)) == \
        '<mrow><mo>max</mo><mfenced><mi>x</mi><mi>y</mi></mfenced></mrow>'
    # 测试打印最大值函数 Max(x, 2, x**3)
    assert mpp.doprint(Max(x, 2, x**3)) == \
        '<mrow><mo>max</mo><mfenced><mn>2</mn><mi>x</mi><msup><mi>x</mi>'\
        '<mn>3</mn></msup></mfenced></mrow>'


def test_mathml_presentation_numbers():
    n = Symbol('n')
    # 测试数学表达式的 MathML 表示：Catalan 数
    assert mathml(catalan(n), printer='presentation') == \
        '<msub><mi>C</mi><mi>n</mi></msub>'
    # 测试数学表达式的 MathML 表示：Bernoulli 数
    assert mathml(bernoulli(n), printer='presentation') == \
        '<msub><mi>B</mi><mi>n</mi></msub>'
    # 测试数学表达式的 MathML 表示：Bell 数
    assert mathml(bell(n), printer='presentation') == \
        '<msub><mi>B</mi><mi>n</mi></msub>'
    # 测试数学表达式的 MathML 表示：Euler 数
    assert mathml(euler(n), printer='presentation') == \
        '<msub><mi>E</mi><mi>n</mi></msub>'
    # 测试数学表达式的 MathML 表示：Fibonacci 数
    assert mathml(fibonacci(n), printer='presentation') == \
        '<msub><mi>F</mi><mi>n</mi></msub>'
    # 测试数学表达式的 MathML 表示：Lucas 数
    assert mathml(lucas(n), printer='presentation') == \
        '<msub><mi>L</mi><mi>n</mi></msub>'
    # 测试数学表达式的 MathML 表示：Tribonacci 数
    assert mathml(tribonacci(n), printer='presentation') == \
        '<msub><mi>T</mi><mi>n</mi></msub>'
    # 测试数学表达式的 MathML 表示：带参数的 Bernoulli 数
    assert mathml(bernoulli(n, x), printer='presentation') == \
        '<mrow><msub><mi>B</mi><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'
    # 测试数学表达式的 MathML 表示：带参数的 Bell 数
    assert mathml(bell(n, x), printer='presentation') == \
        '<mrow><msub><mi>B</mi><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'
    # 测试数学表达式的 MathML 表示：带参数的 Euler 数
    assert mathml(euler(n, x), printer='presentation') == \
        '<mrow><msub><mi>E</mi><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'
    # 测试数学表达式的 MathML 表示：带参数的 Fibonacci 数
    assert mathml(fibonacci(n, x), printer='presentation') == \
        '<mrow><msub><mi>F</mi><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'
    # 测试数学表达式的 MathML 表示：带参数的 Tribonacci 数
    assert mathml(tribonacci(n, x), printer='presentation') == \
        '<mrow><msub><mi>T</mi><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'


def test_mathml_presentation_mathieu():
    # 测试数学表达式的 MathML 表示：Mathieu 函数 C(x, y, z)
    assert mathml(mathieuc(x, y, z), printer='presentation') == \
        '<mrow><mi>C</mi><mfenced><mi>x</mi><mi>y</mi><mi>z</mi></mfenced></mrow>'
    # 测试数学表达式的 MathML 表示：Mathieu 函数 S(x, y, z)
    assert mathml(mathieus(x, y, z), printer='presentation') == \
        '<mrow><mi>S</mi><mfenced><mi>x</mi><mi>y</mi><mi>z</mi></mfenced></mrow>'
    # 测试数学表达式的 MathML 表示：Mathieu 函数 C'(x, y, z)
    assert mathml(mathieucprime(x, y, z), printer='presentation') == \
        '<mrow><mi>C&#x2032;</mi><mfenced><mi>x</mi><mi>y</mi><mi>z</mi></mfenced></mrow>'
    # 测试数学表达式的 MathML 表示：Mathieu 函数 S'(x, y, z)
    assert mathml(mathieusprime(x, y, z), printer='presentation') == \
        '<mrow><mi>S&#x2032;</mi><mfenced><mi>x</mi><mi>y</mi><mi>z</mi></mfenced></mrow>'


def test_mathml_presentation_stieltjes():
    # 测试数学表达式的 MathML 表示：Stieltjes Gamma 函数
    assert mathml(stieltjes(n), printer='presentation') == \
         '<msub><mi>&#x03B3;</mi><mi>n</mi></msub>'
    # 使用 assert 断言语句，验证 mathml 函数生成的结果是否与指定的数学表达式字符串相等
    assert mathml(stieltjes(n, x), printer='presentation') == \
         '<mrow><msub><mi>&#x03B3;</mi><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'
def test_print_matrix_symbol():
    # 创建一个1x2的矩阵符号'A'
    A = MatrixSymbol('A', 1, 2)
    # 使用默认打印器mpp，期望输出XML标记'<mi>A</mi>'
    assert mpp.doprint(A) == '<mi>A</mi>'
    # 使用打印器mp，期望输出XML标记'<ci>A</ci>'
    assert mp.doprint(A) == '<ci>A</ci>'
    # 使用presentation打印器和粗体样式，期望输出XML标记'<mi mathvariant="bold">A</mi>'
    assert mathml(A, printer='presentation', mat_symbol_style="bold") == \
        '<mi mathvariant="bold">A</mi>'
    # 在content打印器中没有效果
    assert mathml(A, mat_symbol_style="bold") == '<ci>A</ci>'


def test_print_hadamard():
    # 导入HadamardProduct和Transpose类
    from sympy.matrices.expressions import HadamardProduct
    from sympy.matrices.expressions import Transpose

    # 创建2x2矩阵符号'X'和'Y'
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)

    # 验证Hadamard乘积的presentation XML输出
    assert mathml(HadamardProduct(X, Y*Y), printer="presentation") == \
        '<mrow>' \
        '<mi>X</mi>' \
        '<mo>&#x2218;</mo>' \
        '<msup><mi>Y</mi><mn>2</mn></msup>' \
        '</mrow>'

    assert mathml(HadamardProduct(X, Y)*Y, printer="presentation") == \
        '<mrow>' \
        '<mfenced>' \
        '<mrow><mi>X</mi><mo>&#x2218;</mo><mi>Y</mi></mrow>' \
        '</mfenced>' \
        '<mo>&InvisibleTimes;</mo><mi>Y</mi>' \
        '</mrow>'

    assert mathml(HadamardProduct(X, Y, Y), printer="presentation") == \
        '<mrow>' \
        '<mi>X</mi><mo>&#x2218;</mo>' \
        '<mi>Y</mi><mo>&#x2218;</mo>' \
        '<mi>Y</mi>' \
        '</mrow>'

    # 验证Hadamard乘积与转置的presentation XML输出
    assert mathml(
        Transpose(HadamardProduct(X, Y)), printer="presentation") == \
            '<msup>' \
            '<mfenced>' \
            '<mrow><mi>X</mi><mo>&#x2218;</mo><mi>Y</mi></mrow>' \
            '</mfenced>' \
            '<mo>T</mo>' \
            '</msup>'


def test_print_random_symbol():
    # 创建一个随机符号R
    R = RandomSymbol(Symbol('R'))
    # 使用mpp打印器，期望输出XML标记'<mi>R</mi>'
    assert mpp.doprint(R) == '<mi>R</mi>'
    # 使用mp打印器，期望输出XML标记'<ci>R</ci>'
    assert mp.doprint(R) == '<ci>R</ci>'


def test_print_IndexedBase():
    # 验证IndexedBase对象的presentation XML输出
    assert mathml(IndexedBase(a)[b], printer='presentation') == \
        '<msub><mi>a</mi><mi>b</mi></msub>'
    assert mathml(IndexedBase(a)[b, c, d], printer='presentation') == \
        '<msub><mi>a</mi><mfenced><mi>b</mi><mi>c</mi><mi>d</mi></mfenced></msub>'
    assert mathml(IndexedBase(a)[b]*IndexedBase(c)[d]*IndexedBase(e),
                  printer='presentation') == \
                  '<mrow><msub><mi>a</mi><mi>b</mi></msub><mo>&InvisibleTimes;'\
                  '</mo><msub><mi>c</mi><mi>d</mi></msub><mo>&InvisibleTimes;</mo><mi>e</mi></mrow>'


def test_print_Indexed():
    # 验证IndexedBase对象的presentation XML输出
    assert mathml(IndexedBase(a), printer='presentation') == '<mi>a</mi>'
    assert mathml(IndexedBase(a/b), printer='presentation') == \
        '<mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow>'
    assert mathml(IndexedBase((a, b)), printer='presentation') == \
        '<mrow><mfenced><mi>a</mi><mi>b</mi></mfenced></mrow>'


def test_print_MatrixElement():
    # 创建符号'i'和'j'
    i, j = symbols('i j')
    # 创建一个矩阵符号'A'，其形状依赖于'i'和'j'
    A = MatrixSymbol('A', i, j)
    # 验证矩阵元素的presentation XML输出
    assert mathml(A[0,0], printer='presentation') == \
        '<msub><mi>A</mi><mfenced close="" open=""><mn>0</mn><mn>0</mn></mfenced></msub>'
    # 确保生成的 MathML 与预期的 MathML 表达式相匹配
    assert mathml(A[i,j], printer='presentation') == \
        '<msub><mi>A</mi><mfenced close="" open=""><mi>i</mi><mi>j</mi></mfenced></msub>'
    
    # 确保生成的 MathML 与预期的 MathML 表达式相匹配
    assert mathml(A[i*j,0], printer='presentation') == \
        '<msub><mi>A</mi><mfenced close="" open=""><mrow><mi>i</mi><mo>&InvisibleTimes;</mo><mi>j</mi></mrow><mn>0</mn></mfenced></msub>'
# 定义测试函数 test_print_Vector，用于测试数学表达式转换为 MathML 的准确性
def test_print_Vector():
    # 创建一个三维坐标系对象 ACS，命名为 'A'
    ACS = CoordSys3D('A')
    # 断言：将向量 ACS.i 和表达式 ACS.j*ACS.x*3 + ACS.k 的叉乘结果转换为 presentation 风格的 MathML
    assert mathml(Cross(ACS.i, ACS.j*ACS.x*3 + ACS.k), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><mfenced><mrow>'\
        '<mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub>'\
        '<mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub>'\
        '</mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><msub><mover>'\
        '<mi mathvariant="bold">k</mi><mo>^</mo></mover><mi mathvariant="bold">'\
        'A</mi></msub></mrow></mfenced></mrow>'
    
    # 断言：将向量 ACS.i 和 ACS.j 的叉乘结果转换为 presentation 风格的 MathML
    assert mathml(Cross(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    
    # 断言：将表达式 x*Cross(ACS.i, ACS.j) 的结果转换为 presentation 风格的 MathML
    assert mathml(x*Cross(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    
    # 断言：将表达式 x*ACS.i 和 ACS.j 的叉乘结果转换为 presentation 风格的 MathML
    assert mathml(Cross(x*ACS.i, ACS.j), printer='presentation') == \
        '<mrow><mo>-</mo><mrow><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub>'\
        '<mo>&#xD7;</mo><mfenced><mrow><mfenced><mi>x</mi></mfenced>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">i</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'\
        '</mfenced></mrow></mrow>'
    
    # 断言：将表达式 3*ACS.x*ACS.j 的旋度结果转换为 presentation 风格的 MathML
    assert mathml(Curl(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mfenced><mrow><mfenced><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub>'\
        '<mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub>'\
        '</mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    
    # 断言：将表达式 3*x*ACS.x*ACS.j 的旋度结果转换为 presentation 风格的 MathML
    assert mathml(Curl(3*x*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mfenced><mrow><mfenced><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x'\
        '</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    # 使用 assert 语句检查 mathml 函数的输出是否符合预期结果
    assert mathml(x*Curl(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><mo>&#x2207;</mo>'\
        '<mo>&#xD7;</mo><mfenced><mrow><mfenced><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mfenced>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'\
        '</mfenced></mrow></mfenced></mrow>'
    # 使用 assert 语句检查 mathml 函数的输出是否符合预期结果
    assert mathml(Curl(3*x*ACS.x*ACS.j + ACS.i), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mfenced><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><mfenced><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x'\
        '</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    # 使用 assert 语句检查 mathml 函数的输出是否符合预期结果
    assert mathml(Divergence(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xB7;</mo><mfenced><mrow><mfenced><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x'\
        '</mi><mi mathvariant="bold">A</mi></msub></mrow></mfenced>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    # 使用 assert 语句检查 mathml 函数的输出是否符合预期结果
    assert mathml(x*Divergence(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><mo>&#x2207;</mo>'\
        '<mo>&#xB7;</mo><mfenced><mrow><mfenced><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mfenced>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'\
        '</mfenced></mrow></mfenced></mrow>'
    # 使用 assert 语句检查 mathml 函数的输出是否符合预期结果
    assert mathml(Divergence(3*x*ACS.x*ACS.j + ACS.i), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xB7;</mo><mfenced><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><mfenced><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub>'\
        '<mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub>'\
        '<mo>&InvisibleTimes;</mo><mi>x</mi></mrow></mfenced>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    assert mathml(Dot(ACS.i, ACS.j*ACS.x*3+ACS.k), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><mfenced><mrow>'\
        '<mfenced><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub>'\
        '<mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub>'\
        '</mrow></mfenced><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><msub><mover>'\
        '<mi mathvariant="bold">k</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    # 验证数学表达式 Dot(ACS.i, ACS.j*ACS.x*3+ACS.k) 的 MathML 表示是否与预期的字符串相等

    assert mathml(Dot(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    # 验证数学表达式 Dot(ACS.i, ACS.j) 的 MathML 表示是否与预期的字符串相等

    assert mathml(Dot(x*ACS.i, ACS.j), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><mfenced><mrow>'\
        '<mfenced><mi>x</mi></mfenced><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    # 验证数学表达式 x*Dot(ACS.i, ACS.j) 的 MathML 表示是否与预期的字符串相等

    assert mathml(x*Dot(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mfenced></mrow>'
    # 验证数学表达式 x*Dot(ACS.i, ACS.j) 的 MathML 表示是否与预期的字符串相等

    assert mathml(Gradient(ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    # 验证梯度 Gradient(ACS.x) 的 MathML 表示是否与预期的字符串相等

    assert mathml(Gradient(ACS.x + 3*ACS.y), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mfenced><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">y</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mrow></mfenced></mrow>'
    # 验证梯度 Gradient(ACS.x + 3*ACS.y) 的 MathML 表示是否与预期的字符串相等

    assert mathml(x*Gradient(ACS.x), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><mo>&#x2207;</mo>'\
        '<msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi>'\
        '</msub></mrow></mfenced></mrow>'
    # 验证数学表达式 x*Gradient(ACS.x) 的 MathML 表示是否与预期的字符串相等

    assert mathml(Gradient(x*ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mfenced><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow></mfenced></mrow>'
    # 验证梯度 Gradient(x*ACS.x) 的 MathML 表示是否与预期的字符串相等
    # 断言：验证 Cross(ACS.x, ACS.z) + Cross(ACS.z, ACS.x) 的数学表达式是否等于指定的 presentation 格式字符串
    assert mathml(Cross(ACS.x, ACS.z) + Cross(ACS.z, ACS.x), printer='presentation') == \
        '<mover><mi mathvariant="bold">0</mi><mo>^</mo></mover>'
    
    # 断言：验证 Cross(ACS.z, ACS.x) 的数学表达式是否等于指定的 presentation 格式字符串
    assert mathml(Cross(ACS.z, ACS.x), printer='presentation') == \
        '<mrow><mo>-</mo><mrow><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub>'\
        '<mi mathvariant="bold">z</mi><mi mathvariant="bold">A</mi></msub></mrow></mrow>'
    
    # 断言：验证 Laplacian(ACS.x) 的数学表达式是否等于指定的 presentation 格式字符串
    assert mathml(Laplacian(ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2206;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    
    # 断言：验证 Laplacian(ACS.x + 3*ACS.y) 的数学表达式是否等于指定的 presentation 格式字符串
    assert mathml(Laplacian(ACS.x + 3*ACS.y), printer='presentation') == \
        '<mrow><mo>&#x2206;</mo><mfenced><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">y</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mrow></mfenced></mrow>'
    
    # 断言：验证 x * Laplacian(ACS.x) 的数学表达式是否等于指定的 presentation 格式字符串
    assert mathml(x*Laplacian(ACS.x), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mfenced><mrow><mo>&#x2206;</mo>'\
        '<msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi>'\
        '</msub></mrow></mfenced></mrow>'
    
    # 断言：验证 Laplacian(x * ACS.x) 的数学表达式是否等于指定的 presentation 格式字符串
    assert mathml(Laplacian(x*ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2206;</mo><mfenced><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow></mfenced></mrow>'
# 测试 elliptic_f 函数的输出是否符合预期
def test_print_elliptic_f():
    # 断言 elliptic_f 函数输出的 MathML 是否与预期相符
    assert mathml(elliptic_f(x, y), printer='presentation') == \
        '<mrow><mi>&#x1d5a5;</mi><mfenced separators="|"><mi>x</mi><mi>y</mi></mfenced></mrow>'
    # 断言 elliptic_f 函数在传入 x/y 和 y 时的输出是否与预期相符
    assert mathml(elliptic_f(x/y, y), printer='presentation') == \
        '<mrow><mi>&#x1d5a5;</mi><mfenced separators="|"><mrow><mfrac><mi>x</mi><mi>y</mi></mfrac></mrow><mi>y</mi></mfenced></mrow>'

# 测试 elliptic_e 函数的输出是否符合预期
def test_print_elliptic_e():
    # 断言 elliptic_e 函数在传入 x 时的输出是否与预期相符
    assert mathml(elliptic_e(x), printer='presentation') == \
        '<mrow><mi>&#x1d5a4;</mi><mfenced separators="|"><mi>x</mi></mfenced></mrow>'
    # 断言 elliptic_e 函数在传入 x 和 y 时的输出是否与预期相符
    assert mathml(elliptic_e(x, y), printer='presentation') == \
        '<mrow><mi>&#x1d5a4;</mi><mfenced separators="|"><mi>x</mi><mi>y</mi></mfenced></mrow>'

# 测试 elliptic_pi 函数的输出是否符合预期
def test_print_elliptic_pi():
    # 断言 elliptic_pi 函数在传入 x 和 y 时的输出是否与预期相符
    assert mathml(elliptic_pi(x, y), printer='presentation') == \
        '<mrow><mi>&#x1d6f1;</mi><mfenced separators="|"><mi>x</mi><mi>y</mi></mfenced></mrow>'
    # 断言 elliptic_pi 函数在传入 x, y 和 z 时的输出是否与预期相符
    assert mathml(elliptic_pi(x, y, z), printer='presentation') == \
        '<mrow><mi>&#x1d6f1;</mi><mfenced separators=";|"><mi>x</mi><mi>y</mi><mi>z</mi></mfenced></mrow>'

# 测试 Ei 函数的输出是否符合预期
def test_print_Ei():
    # 断言 Ei 函数在传入 x 时的输出是否与预期相符
    assert mathml(Ei(x), printer='presentation') == \
        '<mrow><mi>Ei</mi><mfenced><mi>x</mi></mfenced></mrow>'
    # 断言 Ei 函数在传入 x**y 时的输出是否与预期相符
    assert mathml(Ei(x**y), printer='presentation') == \
        '<mrow><mi>Ei</mi><mfenced><msup><mi>x</mi><mi>y</mi></msup></mfenced></mrow>'

# 测试 expint 函数的输出是否符合预期
def test_print_expint():
    # 断言 expint 函数在传入 x 和 y 时的输出是否与预期相符
    assert mathml(expint(x, y), printer='presentation') == \
        '<mrow><msub><mo>E</mo><mi>x</mi></msub><mfenced><mi>y</mi></mfenced></mrow>'
    # 断言 expint 函数在传入 IndexedBase(x)[1] 和 IndexedBase(x)[2] 时的输出是否与预期相符
    assert mathml(expint(IndexedBase(x)[1], IndexedBase(x)[2]), printer='presentation') == \
        '<mrow><msub><mo>E</mo><msub><mi>x</mi><mn>1</mn></msub></msub><mfenced><msub><mi>x</mi><mn>2</mn></msub></mfenced></mrow>'

# 测试 jacobi 函数的输出是否符合预期
def test_print_jacobi():
    # 断言 jacobi 函数在传入 n, a, b 和 x 时的输出是否与预期相符
    assert mathml(jacobi(n, a, b, x), printer='presentation') == \
        '<mrow><msubsup><mo>P</mo><mi>n</mi><mfenced><mi>a</mi><mi>b</mi></mfenced></msubsup><mfenced><mi>x</mi></mfenced></mrow>'

# 测试 gegenbauer 函数的输出是否符合预期
def test_print_gegenbauer():
    # 断言 gegenbauer 函数在传入 n, a 和 x 时的输出是否与预期相符
    assert mathml(gegenbauer(n, a, x), printer='presentation') == \
        '<mrow><msubsup><mo>C</mo><mi>n</mi><mfenced><mi>a</mi></mfenced></msubsup><mfenced><mi>x</mi></mfenced></mrow>'

# 测试 chebyshevt 函数的输出是否符合预期
def test_print_chebyshevt():
    # 断言 chebyshevt 函数在传入 n 和 x 时的输出是否与预期相符
    assert mathml(chebyshevt(n, x), printer='presentation') == \
        '<mrow><msub><mo>T</mo><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'

# 测试 chebyshevu 函数的输出是否符合预期
def test_print_chebyshevu():
    # 断言 chebyshevu 函数在传入 n 和 x 时的输出是否与预期相符
    assert mathml(chebyshevu(n, x), printer='presentation') == \
        '<mrow><msub><mo>U</mo><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'

# 测试 legendre 函数的输出是否符合预期
def test_print_legendre():
    # 断言 legendre 函数在传入 n 和 x 时的输出是否与预期相符
    assert mathml(legendre(n, x), printer='presentation') == \
        '<mrow><msub><mo>P</mo><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'
    # 使用 assert 断言检查生成的 MathML 是否符合预期的格式
    assert mathml(assoc_legendre(n, a, x), printer='presentation') == \
        '<mrow><msubsup><mo>P</mo><mi>n</mi><mfenced><mi>a</mi></mfenced></msubsup><mfenced><mi>x</mi></mfenced></mrow>'
# 定义测试函数 test_print_laguerre，用于测试 laguerre 函数生成的数学表达式是否符合预期
def test_print_laguerre():
    # 断言生成的 MathML 表达式与预期的展示格式一致
    assert mathml(laguerre(n, x), printer='presentation') == \
        '<mrow><msub><mo>L</mo><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'

# 定义测试函数 test_print_assoc_laguerre，用于测试 assoc_laguerre 函数生成的数学表达式是否符合预期
def test_print_assoc_laguerre():
    # 断言生成的 MathML 表达式与预期的展示格式一致
    assert mathml(assoc_laguerre(n, a, x), printer='presentation') == \
        '<mrow><msubsup><mo>L</mo><mi>n</mi><mfenced><mi>a</mi></mfenced></msubsup><mfenced><mi>x</mi></mfenced></mrow>'

# 定义测试函数 test_print_hermite，用于测试 hermite 函数生成的数学表达式是否符合预期
def test_print_hermite():
    # 断言生成的 MathML 表达式与预期的展示格式一致
    assert mathml(hermite(n, x), printer='presentation') == \
        '<mrow><msub><mo>H</mo><mi>n</mi></msub><mfenced><mi>x</mi></mfenced></mrow>'

# 定义测试函数 test_mathml_SingularityFunction，测试 SingularityFunction 类生成的数学表达式是否符合预期
def test_mathml_SingularityFunction():
    # 断言生成的 MathML 表达式与预期的展示格式一致
    assert mathml(SingularityFunction(x, 4, 5), printer='presentation') == \
        '<msup><mfenced close="&#10217;" open="&#10216;"><mrow><mi>x</mi>' \
        '<mo>-</mo><mn>4</mn></mrow></mfenced><mn>5</mn></msup>'
    assert mathml(SingularityFunction(x, -3, 4), printer='presentation') == \
        '<msup><mfenced close="&#10217;" open="&#10216;"><mrow><mi>x</mi>' \
        '<mo>+</mo><mn>3</mn></mrow></mfenced><mn>4</mn></msup>'
    assert mathml(SingularityFunction(x, 0, 4), printer='presentation') == \
        '<msup><mfenced close="&#10217;" open="&#10216;"><mi>x</mi></mfenced>' \
        '<mn>4</mn></msup>'
    assert mathml(SingularityFunction(x, a, n), printer='presentation') == \
        '<msup><mfenced close="&#10217;" open="&#10216;"><mrow><mrow>' \
        '<mo>-</mo><mi>a</mi></mrow><mo>+</mo><mi>x</mi></mrow></mfenced>' \
        '<mi>n</mi></msup>'
    assert mathml(SingularityFunction(x, 4, -2), printer='presentation') == \
        '<msup><mfenced close="&#10217;" open="&#10216;"><mrow><mi>x</mi>' \
        '<mo>-</mo><mn>4</mn></mrow></mfenced><mn>-2</mn></msup>'
    assert mathml(SingularityFunction(x, 4, -1), printer='presentation') == \
        '<msup><mfenced close="&#10217;" open="&#10216;"><mrow><mi>x</mi>' \
        '<mo>-</mo><mn>4</mn></mrow></mfenced><mn>-1</mn></msup>'

# 定义测试函数 test_mathml_matrix_functions，测试矩阵函数的数学表达式是否符合预期
def test_mathml_matrix_functions():
    # 导入必要的矩阵操作类
    from sympy.matrices import Adjoint, Inverse, Transpose
    # 创建两个矩阵符号 X 和 Y
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    # 断言生成的 MathML 表达式与预期的展示格式一致
    assert mathml(Adjoint(X), printer='presentation') == \
        '<msup><mi>X</mi><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(X + Y), printer='presentation') == \
        '<msup><mfenced><mrow><mi>X</mi><mo>+</mo><mi>Y</mi></mrow></mfenced><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(X) + Adjoint(Y), printer='presentation') == \
        '<mrow><msup><mi>X</mi><mo>&#x2020;</mo></msup><mo>+</mo><msup>' \
        '<mi>Y</mi><mo>&#x2020;</mo></msup></mrow>'
    assert mathml(Adjoint(X*Y), printer='presentation') == \
        '<msup><mfenced><mrow><mi>X</mi><mo>&InvisibleTimes;</mo>' \
        '<mi>Y</mi></mrow></mfenced><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(Y)*Adjoint(X), printer='presentation') == \
        '<mrow><msup><mi>Y</mi><mo>&#x2020;</mo></msup><mo>&InvisibleTimes;' \
        '</mo><msup><mi>X</mi><mo>&#x2020;</mo></msup></mrow>'
    # 断言：验证 Adjoint(X**2) 的 MathML 表示是否与预期的字符串相等
    assert mathml(Adjoint(X**2), printer='presentation') == \
        '<msup><mfenced><msup><mi>X</mi><mn>2</mn></msup></mfenced><mo>&#x2020;</mo></msup>'
    
    # 断言：验证 Adjoint(X)**2 的 MathML 表示是否与预期的字符串相等
    assert mathml(Adjoint(X)**2, printer='presentation') == \
        '<msup><mfenced><msup><mi>X</mi><mo>&#x2020;</mo></msup></mfenced><mn>2</mn></msup>'
    
    # 断言：验证 Adjoint(Inverse(X)) 的 MathML 表示是否与预期的字符串相等
    assert mathml(Adjoint(Inverse(X)), printer='presentation') == \
        '<msup><mfenced><msup><mi>X</mi><mn>-1</mn></msup></mfenced><mo>&#x2020;</mo></msup>'
    
    # 断言：验证 Inverse(Adjoint(X)) 的 MathML 表示是否与预期的字符串相等
    assert mathml(Inverse(Adjoint(X)), printer='presentation') == \
        '<msup><mfenced><msup><mi>X</mi><mo>&#x2020;</mo></msup></mfenced><mn>-1</mn></msup>'
    
    # 断言：验证 Adjoint(Transpose(X)) 的 MathML 表示是否与预期的字符串相等
    assert mathml(Adjoint(Transpose(X)), printer='presentation') == \
        '<msup><mfenced><msup><mi>X</mi><mo>T</mo></msup></mfenced><mo>&#x2020;</mo></msup>'
    
    # 断言：验证 Transpose(Adjoint(X)) 的 MathML 表示是否与预期的字符串相等
    assert mathml(Transpose(Adjoint(X)), printer='presentation') ==  \
        '<msup><mfenced><msup><mi>X</mi><mo>&#x2020;</mo></msup></mfenced><mo>T</mo></msup>'
    
    # 断言：验证 Transpose(Adjoint(X) + Y) 的 MathML 表示是否与预期的字符串相等
    assert mathml(Transpose(Adjoint(X) + Y), printer='presentation') ==  \
        '<msup><mfenced><mrow><msup><mi>X</mi><mo>&#x2020;</mo></msup>' \
        '<mo>+</mo><mi>Y</mi></mrow></mfenced><mo>T</mo></msup>'
    
    # 断言：验证 Transpose(X) 的 MathML 表示是否与预期的字符串相等
    assert mathml(Transpose(X), printer='presentation') == \
        '<msup><mi>X</mi><mo>T</mo></msup>'
    
    # 断言：验证 Transpose(X + Y) 的 MathML 表示是否与预期的字符串相等
    assert mathml(Transpose(X + Y), printer='presentation') == \
        '<msup><mfenced><mrow><mi>X</mi><mo>+</mo><mi>Y</mi></mrow></mfenced><mo>T</mo></msup>'
# 定义测试函数，用于测试生成 MathML 的特殊矩阵表示
def test_mathml_special_matrices():
    # 从 sympy.matrices 模块导入 Identity、ZeroMatrix 和 OneMatrix 类
    from sympy.matrices import Identity, ZeroMatrix, OneMatrix
    # 断言生成 Identity(4) 的 Presentation MathML 字符串是否为 '<mi>&#x1D540;</mi>'
    assert mathml(Identity(4), printer='presentation') == '<mi>&#x1D540;</mi>'
    # 断言生成 ZeroMatrix(2, 2) 的 Presentation MathML 字符串是否为 '<mn>&#x1D7D8</mn>'
    assert mathml(ZeroMatrix(2, 2), printer='presentation') == '<mn>&#x1D7D8</mn>'
    # 断言生成 OneMatrix(2, 2) 的 Presentation MathML 字符串是否为 '<mn>&#x1D7D9</mn>'
    assert mathml(OneMatrix(2, 2), printer='presentation') == '<mn>&#x1D7D9</mn>'

# 定义测试函数，用于测试生成 Piecewise 函数的 MathML 表示
def test_mathml_piecewise():
    # 从 sympy.functions.elementary.piecewise 模块导入 Piecewise 类
    from sympy.functions.elementary.piecewise import Piecewise
    # 断言生成 Piecewise((x, x <= 1), (x**2, True)) 的 Content MathML 字符串
    assert mathml(Piecewise((x, x <= 1), (x**2, True))) == \
        '<piecewise><piece><ci>x</ci><apply><leq/><ci>x</ci><cn>1</cn></apply></piece><otherwise><apply><power/><ci>x</ci><cn>2</cn></apply></otherwise></piecewise>'
    # 使用 lambda 函数检验生成 Piecewise((x, x <= 1)) 时是否引发 ValueError
    raises(ValueError, lambda: mathml(Piecewise((x, x <= 1))))

# 定义测试函数，用于测试生成 Range 对象的 Presentation MathML 表示
def test_issue_17857():
    # 断言生成 Range(-oo, oo) 的 Presentation MathML 字符串
    assert mathml(Range(-oo, oo), printer='presentation') == \
        '<mfenced close="}" open="{"><mi>&#8230;</mi><mn>-1</mn><mn>0</mn><mn>1</mn><mi>&#8230;</mi></mfenced>'
    # 断言生成 Range(oo, -oo, -1) 的 Presentation MathML 字符串
    assert mathml(Range(oo, -oo, -1), printer='presentation') == \
        '<mfenced close="}" open="{"><mi>&#8230;</mi><mn>1</mn><mn>0</mn><mn>-1</mn><mi>&#8230;</mi></mfenced>'

# 定义测试函数，用于测试 sympify 和 mp.doprint 的浮点数往返
def test_float_roundtrip():
    # 使用 sympify 将浮点数转换为符号表达式 x
    x = sympify(0.8975979010256552)
    # 使用 mp.doprint 将 x 转换为字符串，并移除尾部的 '</cn>'
    y = float(mp.doprint(x).strip('</cn>'))
    # 断言 x 和 y 相等
    assert x == y
```