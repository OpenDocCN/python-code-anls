# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_wester.py`

```
""" Tests from Michael Wester's 1999 paper "Review of CAS mathematical
capabilities".

http://www.math.unm.edu/~wester/cas/book/Wester.pdf
See also http://math.unm.edu/~wester/cas_review.html for detailed output of
each tested system.
"""

# 导入 SymPy 库中各种模块和函数，用于数学计算和符号运算

from sympy.assumptions.ask import Q, ask  # 导入符号逻辑推理相关模块
from sympy.assumptions.refine import refine  # 导入符号表达式的简化函数
from sympy.concrete.products import product  # 导入符号乘积计算函数
from sympy.core import EulerGamma  # 导入欧拉常数
from sympy.core.evalf import N  # 导入数值计算函数
from sympy.core.function import (Derivative, Function, Lambda, Subs,  # 导入函数相关类和方法
    diff, expand, expand_func)
from sympy.core.mul import Mul  # 导入符号乘法类
from sympy.core.intfunc import igcd  # 导入整数最大公约数计算函数
from sympy.core.numbers import (AlgebraicNumber, E, I, Rational,  # 导入符号数值类和常数
    nan, oo, pi, zoo)
from sympy.core.relational import Eq, Lt  # 导入符号关系表达式类
from sympy.core.singleton import S  # 导入符号表达式的单例类
from sympy.core.symbol import Dummy, Symbol, symbols  # 导入符号变量相关类和方法
from sympy.functions.combinatorial.factorials import (rf, binomial,  # 导入组合数学函数
    factorial, factorial2)
from sympy.functions.combinatorial.numbers import (bernoulli, fibonacci,  # 导入组合数学函数
    totient, partition)
from sympy.functions.elementary.complexes import (conjugate, im, re,  # 导入复数函数
    sign)
from sympy.functions.elementary.exponential import LambertW, exp, log  # 导入指数函数
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh,  # 导入双曲函数
    tanh)
from sympy.functions.elementary.integers import ceiling, floor  # 导入整数取上下整函数
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt  # 导入杂项函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.elementary.trigonometric import (acos, acot, asin,  # 导入三角函数
    atan, cos, cot, csc, sec, sin, tan)
from sympy.functions.special.bessel import besselj  # 导入贝塞尔函数
from sympy.functions.special.delta_functions import DiracDelta  # 导入狄拉克函数
from sympy.functions.special.elliptic_integrals import (elliptic_e,  # 导入椭圆积分函数
    elliptic_f)
from sympy.functions.special.gamma_functions import gamma, polygamma  # 导入伽马函数及其相关函数
from sympy.functions.special.hyper import hyper  # 导入超几何函数
from sympy.functions.special.polynomials import (assoc_legendre, chebyshevt)  # 导入特殊多项式函数
from sympy.functions.special.zeta_functions import polylog  # 导入多重对数函数
from sympy.geometry.util import idiff  # 导入几何模块中的差分函数
from sympy.logic.boolalg import And  # 导入布尔逻辑运算中的与运算
from sympy.matrices.dense import hessian, wronskian  # 导入矩阵计算函数
from sympy.matrices.expressions.matmul import MatMul  # 导入矩阵乘法表达式类
from sympy.ntheory.continued_fraction import (  # 导入连分数计算函数
    continued_fraction_convergents as cf_c,
    continued_fraction_iterator as cf_i,
    continued_fraction_periodic as cf_p,
    continued_fraction_reduce as cf_r)
from sympy.ntheory.factor_ import factorint  # 导入因子分解函数
from sympy.ntheory.generate import primerange  # 导入素数生成函数
from sympy.polys.domains.integerring import ZZ  # 导入整数环
from sympy.polys.orthopolys import legendre_poly  # 导入勒让德多项式计算函数
from sympy.polys.partfrac import apart  # 导入部分分式展开函数
from sympy.polys.polytools import Poly, factor, gcd, resultant  # 导入多项式处理函数
from sympy.series.limits import limit  # 导入极限计算函数
from sympy.series.order import O  # 导入符号级数展开的高阶小量类
from sympy.series.residues import residue  # 导入残差计算函数
from sympy.series.series import series  # 导入级数展开函数
from sympy.sets.fancysets import ImageSet  # 导入映射集合类
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Union)  # 导入集合操作函数
from sympy.simplify.combsimp import combsimp  # 导入组合简化函数
# 从 sympy 库中导入各种简化函数和求解函数

from sympy.simplify.hyperexpand import hyperexpand  # 导入超几何展开函数
from sympy.simplify.powsimp import powdenest, powsimp  # 导入幂简化函数
from sympy.simplify.radsimp import radsimp  # 导入根式简化函数
from sympy.simplify.simplify import logcombine, simplify  # 导入对数合并和全面简化函数
from sympy.simplify.sqrtdenest import sqrtdenest  # 导入平方根简化函数
from sympy.simplify.trigsimp import trigsimp  # 导入三角函数简化函数
from sympy.solvers.solvers import solve  # 导入求解器函数

# 导入 mpmath 库，支持高精度数学计算
import mpmath

# 从 sympy 库中导入组合数学函数
from sympy.functions.combinatorial.numbers import stirling

# 从 sympy 库中导入特殊函数
from sympy.functions.special.delta_functions import Heaviside  # 导入海维赛德函数
from sympy.functions.special.error_functions import Ci, Si, erf  # 导入误差函数
from sympy.functions.special.zeta_functions import zeta  # 导入 Riemann zeta 函数

# 从 sympy.testing.pytest 模块导入测试装饰器和函数
from sympy.testing.pytest import (XFAIL, slow, SKIP, tooslow, raises)

# 从 sympy.utilities.iterables 模块导入 partitions 函数
from sympy.utilities.iterables import partitions

# 从 mpmath 库中导入 mpi 和 mpc 类
from mpmath import mpi, mpc

# 从 sympy.matrices 模块中导入矩阵相关的类和函数
from sympy.matrices import Matrix, GramSchmidt, eye

# 从 sympy.matrices.expressions.blockmatrix 模块中导入块矩阵相关的类和函数
from sympy.matrices.expressions.blockmatrix import BlockMatrix, block_collapse

# 从 sympy.matrices.expressions 模块中导入矩阵符号和零矩阵类
from sympy.matrices.expressions import MatrixSymbol, ZeroMatrix

# 从 sympy.physics.quantum 模块中导入量子力学相关的类和函数
from sympy.physics.quantum import Commutator

# 从 sympy.polys.rings 模块中导入多项式环类
from sympy.polys.rings import PolyRing

# 从 sympy.polys.fields 模块中导入分式域类
from sympy.polys.fields import FracField

# 从 sympy.polys.solvers 模块中导入线性方程组求解函数
from sympy.polys.solvers import solve_lin_sys

# 从 sympy.concrete 模块中导入求和类
from sympy.concrete import Sum

# 从 sympy.concrete.products 模块中导入积类
from sympy.concrete.products import Product

# 从 sympy.integrals 模块中导入积分函数
from sympy.integrals import integrate

# 从 sympy.integrals.transforms 模块中导入变换函数
from sympy.integrals.transforms import laplace_transform, \
    inverse_laplace_transform, LaplaceTransform, fourier_transform, \
    mellin_transform, laplace_correspondence, laplace_initial_conds

# 从 sympy.solvers.recurr 模块中导入递推关系求解函数
from sympy.solvers.recurr import rsolve

# 从 sympy.solvers.solveset 模块中导入方程解集求解函数
from sympy.solvers.solveset import solveset, solveset_real, linsolve

# 从 sympy.solvers.ode 模块中导入常微分方程求解函数
from sympy.solvers.ode import dsolve

# 从 sympy.core.relational 模块中导入等式类
from sympy.core.relational import Equality

# 导入 itertools 模块中的部分函数
from itertools import islice, takewhile

# 从 sympy.series.formal 模块中导入形式幂级数类
from sympy.series.formal import fps

# 从 sympy.series.fourier 模块中导入傅里叶级数类
from sympy.series.fourier import fourier_series

# 从 sympy.calculus.util 模块中导入最小值函数
from sympy.calculus.util import minimum

# 从 sympy 中导入 EmptySet 和 Rational 类
EmptySet = S.EmptySet
R = Rational

# 定义符号变量 x, y, z 和整数符号变量 i, j, k, l, m, n
x, y, z = symbols('x y z')
i, j, k, l, m, n = symbols('i j k l m n', integer=True)

# 定义函数符号 f 和 g
f = Function('f')
g = Function('g')

# A. Boolean Logic and Quantifier Elimination
#   Not implemented.

# B. Set Theory

# 定义测试函数 test_B1，验证并集操作
def test_B1():
    assert (FiniteSet(i, j, j, k, k, k) | FiniteSet(l, k, j) |
            FiniteSet(j, m, j)) == FiniteSet(i, j, k, l, m)

# 定义测试函数 test_B2，验证交集操作
def test_B2():
    assert (FiniteSet(i, j, j, k, k, k) & FiniteSet(l, k, j) &
            FiniteSet(j, m, j)) == Intersection({j, m}, {i, j, k}, {j, k, l})
    # 上面的输出看起来不像预期输出。
    # 应该有一种方法来重写交集，但我不明白为什么一个交集应该被评估为这样。

# 定义测试函数 test_B3，验证差集操作
def test_B3():
    assert (FiniteSet(i, j, k, l, m) - FiniteSet(j) ==
            FiniteSet(i, k, l, m))

# 定义测试函数 test_B4，验证笛卡尔积操作
def test_B4():
    assert (FiniteSet(*(FiniteSet(i, j)*FiniteSet(k, l))) ==
            FiniteSet((i, k), (i, l), (j, k), (j, l)))

# C. Numbers

# 定义测试函数 test_C1，验证阶乘函数
def test_C1():
    assert (factorial(50) ==
        30414093201713378043612608166064768844377641568960512000000000000)
def test_C2():
    assert (factorint(factorial(50)) == {2: 47, 3: 22, 5: 12, 7: 8,
        11: 4, 13: 3, 17: 2, 19: 2, 23: 2, 29: 1, 31: 1, 37: 1,
        41: 1, 43: 1, 47: 1})

# 测试阶乘50的质因数分解结果是否正确

def test_C3():
    assert (factorial2(10), factorial2(9)) == (3840, 945)

# 测试 SymPy 中阶乘2函数的返回值是否正确

# Base conversions; not really implemented by SymPy
# Whatever. Take credit!
def test_C4():
    assert 0xABC == 2748

# 测试十六进制数 0xABC 是否等于十进制数 2748

def test_C5():
    assert 123 == int('234', 7)

# 测试将字符串 '234' 视为七进制数转换为十进制是否等于123

def test_C6():
    assert int('677', 8) == int('1BF', 16) == 447

# 测试八进制数 '677' 和十六进制数 '1BF' 转换为十进制是否都等于447

def test_C7():
    assert log(32768, 8) == 5

# 测试以8为底，32768的对数是否等于5

def test_C8():
    # Modular multiplicative inverse. Would be nice if divmod could do this.
    assert ZZ.invert(5, 7) == 3
    assert ZZ.invert(5, 6) == 5

# 测试在模7和模6下的数5的乘法逆元是否分别为3和5

def test_C9():
    assert igcd(igcd(1776, 1554), 5698) == 74

# 测试三个数 1776、1554、5698 的最大公约数是否等于74

def test_C10():
    x = 0
    for n in range(2, 11):
        x += R(1, n)
    assert x == R(4861, 2520)

# 计算分数序列 1/2 + 1/3 + ... + 1/10 的和，检查是否等于 4861/2520

def test_C11():
    assert R(1, 7) == S('0.[142857]')

# 测试 1/7 是否等于循环小数 0.[142857]

def test_C12():
    assert R(7, 11) * R(22, 7) == 2

# 测试两个有理数的乘积是否等于2

def test_C13():
    test = R(10, 7) * (1 + R(29, 1000)) ** R(1, 3)
    good = 3 ** R(1, 3)
    assert test == good

# 测试一个复合表达式是否等于一个已知好的值

def test_C14():
    assert sqrtdenest(sqrt(2*sqrt(3) + 4)) == 1 + sqrt(3)

# 测试平方根的嵌套表达式是否等于一个已知的好的值

def test_C15():
    test = sqrtdenest(sqrt(14 + 3*sqrt(3 + 2*sqrt(5 - 12*sqrt(3 - 2*sqrt(2))))))
    good = sqrt(2) + 3
    assert test == good

# 测试一个复杂的平方根嵌套表达式是否等于一个已知的好的值

def test_C16():
    test = sqrtdenest(sqrt(10 + 2*sqrt(6) + 2*sqrt(10) + 2*sqrt(15)))
    good = sqrt(2) + sqrt(3) + sqrt(5)
    assert test == good

# 测试另一个复杂的平方根嵌套表达式是否等于一个已知的好的值

def test_C17():
    test = radsimp((sqrt(3) + sqrt(2)) / (sqrt(3) - sqrt(2)))
    good = 5 + 2*sqrt(6)
    assert test == good

# 测试一个有理化简表达式是否等于一个已知的好的值

def test_C18():
    assert simplify((sqrt(-2 + sqrt(-5)) * sqrt(-2 - sqrt(-5))).expand(complex=True)) == 3

# 测试一个复杂的复数表达式是否等于一个已知的好的值

@XFAIL
def test_C19():
    assert radsimp(simplify((90 + 34*sqrt(7)) ** R(1, 3))) == 3 + sqrt(7)

# 测试一个无法通过的有理化简表达式

def test_C20():
    inside = (135 + 78*sqrt(3))
    test = AlgebraicNumber((inside**R(2, 3) + 3) * sqrt(3) / inside**R(1, 3))
    assert simplify(test) == AlgebraicNumber(12)

# 测试一个代数数表达式是否等于一个已知的好的值

def test_C21():
    assert simplify(AlgebraicNumber((41 + 29*sqrt(2)) ** R(1, 5))) == \
        AlgebraicNumber(1 + sqrt(2))

# 测试一个代数数表达式是否等于一个已知的好的值

@XFAIL
def test_C22():
    test = simplify(((6 - 4*sqrt(2))*log(3 - 2*sqrt(2)) + (3 - 2*sqrt(2))*log(17
        - 12*sqrt(2)) + 32 - 24*sqrt(2)) / (48*sqrt(2) - 72))
    good = sqrt(2)/3 - log(sqrt(2) - 1)/3
    assert test == good

# 测试一个无法通过的复杂表达式

def test_C23():
    assert 2 * oo - 3 is oo

# 测试一个无穷大的算术运算

@XFAIL
def test_C24():
    raise NotImplementedError("2**aleph_null == aleph_1")

# D. Numerical Analysis

def test_D1():
    assert 0.0 / sqrt(2) == 0.0

# 测试浮点数运算

def test_D2():
    assert str(exp(-1000000).evalf()) == '3.29683147808856e-434295'

# 测试数值运算，检查对 e 的负百万次方的数值近似是否正确

def test_D3():
    assert exp(pi*sqrt(163)).evalf(50).num.ae(262537412640768744)

# 测试数值运算，检查对 e 的 pi*sqrt(163) 的数值近似是否正确

def test_D4():
    assert floor(R(-5, 3)) == -2
    assert ceiling(R(-5, 3)) == -1

# 测试对有理数进行向下取整和向上取整

@XFAIL
def test_D5():
    raise NotImplementedError("cubic_spline([1, 2, 4, 5], [1, 4, 2, 3], x)(3) == 27/8")

# 测试一个未实现的数值计算方法

@XFAIL
def test_D6():
    pass

# 结束
    # 抛出未实现错误，指示需要将该数学表达式转换为 FORTRAN 语言的实现
    raise NotImplementedError("translate sum(a[i]*x**i, (i,1,n)) to FORTRAN")
@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_D7():
    raise NotImplementedError("translate sum(a[i]*x**i, (i,1,n)) to C")


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_D8():
    # 一种方法是将求和表达式转换为字符串，
    # 然后用空字符串替换 '[' 和 ']'。
    # 例如，horner(S(str(_).replace('[','').replace(']','')))
    raise NotImplementedError("apply Horner's rule to sum(a[i]*x**i, (i,1,5))")


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_D9():
    raise NotImplementedError("translate D8 to FORTRAN")


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_D10():
    raise NotImplementedError("translate D8 to C")


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_D11():
    # 是否有办法使用 count_ops 函数？
    raise NotImplementedError("flops(sum(product(f[i][k], (i,1,k)), (k,1,n)))")


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_D12():
    assert (mpi(-4, 2) * x + mpi(1, 3)) ** 2 == mpi(-8, 16)*x**2 + mpi(-24, 12)*x + mpi(1, 9)


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_D13():
    raise NotImplementedError("discretize a PDE: diff(f(x,t),t) == diff(diff(f(x,t),x),x)")


# E. Statistics
#   See scipy; all of this is numerical.

# F. Combinatorial Theory.

def test_F1():
    # 断言计算 rf(x, 3) 的结果是否等于 x*(1 + x)*(2 + x)
    assert rf(x, 3) == x*(1 + x)*(2 + x)


def test_F2():
    # 断言计算 binomial(n, 3) 展开后是否等于 n*(n - 1)*(n - 2)/6
    assert expand_func(binomial(n, 3)) == n*(n - 1)*(n - 2)/6


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_F3():
    # 断言对 combsimp 进行简化后，是否等于 factorial(2*n)
    assert combsimp(2**n * factorial(n) * factorial2(2*n - 1)) == factorial(2*n)


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_F4():
    # 断言对 combsimp 进行简化后，是否等于 factorial(2*n)
    assert combsimp(2**n * factorial(n) * product(2*k - 1, (k, 1, n))) == factorial(2*n)


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_F5():
    # 断言 gamma(n + R(1, 2)) / sqrt(pi) / factorial(n) 是否等于 factorial(2*n)/2**(2*n)/factorial(n)**2
    assert gamma(n + R(1, 2)) / sqrt(pi) / factorial(n) == factorial(2*n)/2**(2*n)/factorial(n)**2


def test_F6():
    # 创建 partitions(4) 的副本列表，并与预期的结果 partDesired 进行比较
    partTest = [p.copy() for p in partitions(4)]
    partDesired = [{4: 1}, {1: 1, 3: 1}, {2: 2}, {1: 2, 2:1}, {1: 4}]
    assert partTest == partDesired


def test_F7():
    # 断言 partition(4) 的结果是否等于 5
    assert partition(4) == 5


def test_F8():
    # 断言 stirling(5, 2, signed=True) 的结果是否等于 -50
    assert stirling(5, 2, signed=True) == -50  # if signed, then kind=1


def test_F9():
    # 断言 totient(1776) 的结果是否等于 576
    assert totient(1776) == 576

# G. Number Theory

def test_G1():
    # 断言 primerange(999983, 1000004) 的结果列表是否等于 [999983, 1000003]
    assert list(primerange(999983, 1000004)) == [999983, 1000003]


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_G2():
    raise NotImplementedError("find the primitive root of 191 == 19")


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_G3():
    raise NotImplementedError("(a+b)**p mod p == a**p + b**p mod p; p prime")

# ... G14 Modular equations are not implemented.

def test_G15():
    # 断言有理数 sqrt(3) 的有理数逼近是否等于 R(26, 15)
    assert Rational(sqrt(3).evalf()).limit_denominator(15) == R(26, 15)
    assert list(takewhile(lambda x: x.q <= 15, cf_c(cf_i(sqrt(3)))))[-1] == \
        R(26, 15)


def test_G16():
    # 断言对 pi 的连分数表示的前 10 个元素是否等于 [3, 7, 15, 1, 292, 1, 1, 1, 2, 1]
    assert list(islice(cf_i(pi),10)) == [3, 7, 15, 1, 292, 1, 1, 1, 2, 1]


def test_G17():
    # 断言对于 cf_p(0, 1, 23) 的结果是否等于 [4, [1, 3, 1, 8]]
    assert cf_p(0, 1, 23) == [4, [1, 3, 1, 8]]


def test_G18():
    # 断言对于 cf_p(1, 2, 5) 的结果是否等于 [[1]]
    assert cf_p(1, 2, 5) == [[1]]
    # 断言对于 cf_r([[1]]) 展开后是否等于 S.Half + sqrt(5)/2
    assert cf_r([[1]]).expand() == S.Half + sqrt(5)/2


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_G19():
    s = symbols('s', integer=True, positive=True)
    it = cf_i((exp(1/s) - 1)/(exp(1/s) + 1))
    # 断言对于迭代器 it 的前 5 个元素是否等于 [0, 2*s, 6*s, 10*s, 14*s]
    assert list(islice(it, 5)) == [0, 2*s, 6*s, 10*s, 14*s]


def test_G20():
    s = symbols('s', integer=True, positive=True)
    # Wester erroneously has this as -s + sqrt(s**2 + 1)
    # 断言对于 cf_r([[2*s]]) 的结果是否等于 s + sqrt(s**2 + 1)
    assert cf_r([[2*s]]) == s + sqrt(s**2 + 1)


@XFAIL
# 标记测试函数为预期失败状态，即预期此测试会失败
def test_G20b():
    pass  # Placeholder for future test case
    # 定义符号变量 s，其类型为整数并且为正数
    s = symbols('s', integer=True, positive=True)
    # 使用自定义函数 cf_p 对 s 进行计算，并断言结果是否等于 [[2*s]]
    assert cf_p(s, 1, s**2 + 1) == [[2*s]]
# H. Algebra

# 测试 H1：验证简化后的表达式相等
def test_H1():
    assert simplify(2*2**n) == simplify(2**(n + 1))
    assert powdenest(2*2**n) == simplify(2**(n + 1))

# 测试 H2：验证指数幂化简
def test_H2():
    assert powsimp(4 * 2**n) == 2**(n + 2)

# 测试 H3：验证幂运算的结果
def test_H3():
    assert (-1)**(n*(n + 1)) == 1

# 测试 H4：验证因式分解
def test_H4():
    expr = factor(6*x - 10)
    assert type(expr) is Mul
    assert expr.args[0] == 2
    assert expr.args[1] == 3*x - 5

# 定义多项式 p1, p2 和 q
p1 = 64*x**34 - 21*x**47 - 126*x**8 - 46*x**5 - 16*x**60 - 81
p2 = 72*x**60 - 25*x**25 - 19*x**23 - 22*x**39 - 83*x**52 + 54*x**10 + 81
q = 34*x**19 - 25*x**16 + 70*x**7 + 20*x**3 - 91*x - 86

# 测试 H5：验证最大公因数
def test_H5():
    assert gcd(p1, p2, x) == 1

# 测试 H6：验证最大公因数
def test_H6():
    assert gcd(expand(p1 * q), expand(p2 * q)) == q

# 定义多变量多项式 p1 和 p2
p1 = 24*x*y**19*z**8 - 47*x**17*y**5*z**8 + 6*x**15*y**9*z**2 - 3*x**22 + 5
p2 = 34*x**5*y**8*z**13 + 20*x**7*y**7*z**7 + 12*x**9*y**16*z**4 + 80*y**14*z

# 测试 H7：验证多变量多项式的最大公因数
def test_H7():
    assert gcd(p1, p2, x, y, z) == 1

# 测试 H8：验证多变量多项式乘积的最大公因数
def test_H8():
    q = 11*x**12*y**7*z**13 - 23*x**2*y**8*z**10 + 47*x**17*y**5*z**8
    assert gcd(p1 * q, p2 * q, x, y, z) == q

# 测试 H9：验证多项式的最大公因数
def test_H9():
    x = Symbol('x', zero=False)
    p1 = 2*x**(n + 4) - x**(n + 2)
    p2 = 4*x**(n + 1) + 3*x**n
    assert gcd(p1, p2) == x**n

# 测试 H10：验证多项式的 resultants
def test_H10():
    p1 = 3*x**4 + 3*x**3 + x**2 - x - 2
    p2 = x**3 - 3*x**2 + x + 5
    assert resultant(p1, p2, x) == 0

# 测试 H11：验证多项式的 resultants
def test_H11():
    assert resultant(p1 * q, p2 * q, x) == 0

# 测试 H12：验证多项式的化简
def test_H12():
    num = x**2 - 4
    den = x**2 + 4*x + 4
    assert simplify(num/den) == (x - 2)/(x + 2)

# 测试 H13：验证多项式的化简（预期失败）
@XFAIL
def test_H13():
    assert simplify((exp(x) - 1) / (exp(x/2) + 1)) == exp(x/2) - 1

# 测试 H14：验证多项式的展开和求导
def test_H14():
    p = (x + 1) ** 20
    ep = expand(p)
    assert ep == (1 + 20*x + 190*x**2 + 1140*x**3 + 4845*x**4 + 15504*x**5
        + 38760*x**6 + 77520*x**7 + 125970*x**8 + 167960*x**9 + 184756*x**10
        + 167960*x**11 + 125970*x**12 + 77520*x**13 + 38760*x**14 + 15504*x**15
        + 4845*x**16 + 1140*x**17 + 190*x**18 + 20*x**19 + x**20)
    dep = diff(ep, x)
    assert dep == (20 + 380*x + 3420*x**2 + 19380*x**3 + 77520*x**4
        + 232560*x**5 + 542640*x**6 + 1007760*x**7 + 1511640*x**8 + 1847560*x**9
        + 1847560*x**10 + 1511640*x**11 + 1007760*x**12 + 542640*x**13
        + 232560*x**14 + 77520*x**15 + 19380*x**16 + 3420*x**17 + 380*x**18
        + 20*x**19)
    assert factor(dep) == 20*(1 + x)**19

# 测试 H15：验证多项式的化简
def test_H15():
    assert simplify(Mul(*[x - r for r in solveset(x**3 + x**2 - 7)])) == x**3 + x**2 - 7

# 测试 H16：验证多项式的因式分解
def test_H16():
    assert factor(x**100 - 1) == ((x - 1)*(x + 1)*(x**2 + 1)*(x**4 - x**3
        + x**2 - x + 1)*(x**4 + x**3 + x**2 + x + 1)*(x**8 - x**6 + x**4
        - x**2 + 1)*(x**20 - x**15 + x**10 - x**5 + 1)*(x**20 + x**15 + x**10
        + x**5 + 1)*(x**40 - x**30 + x**20 - x**10 + 1))

# 测试 H17：验证多项式的化简和因式分解
def test_H17():
    assert simplify(factor(expand(p1 * p2)) - p1*p2) == 0

# 测试 H18：预期失败的测试（空测试，没有实现）
    # 在复数有理数域上因式分解多项式 4*x**4 + 8*x**3 + 77*x**2 + 18*x + 153。
    test = factor(4*x**4 + 8*x**3 + 77*x**2 + 18*x + 153)
    # 期望的正确因式分解结果
    good = (2*x + 3*I)*(2*x - 3*I)*(x + 1 - 4*I)*(x + 1 + 4*I)
    # 断言语句，验证计算出的因式分解结果是否与期望的结果相等
    assert test == good
def test_H19():
    # 定义符号变量 a
    a = symbols('a')
    # 断言：通过多项式求逆操作得到结果 a + 1，其中多项式为 Poly(a**2 - 2) / Poly(a - 1)
    assert Poly(a - 1).invert(Poly(a**2 - 2)) == a + 1


@XFAIL
def test_H20():
    # 抛出 NotImplementedError 异常，提示信息为一个复杂的算术表达式
    raise NotImplementedError("let a**2==2; (x**3 + (a-2)*x**2 - "
        + "(2*a+3)*x - 3*a) / (x**2-2) = (x**2 - 2*x - 3) / (x-a)")


@XFAIL
def test_H21():
    # 抛出 NotImplementedError 异常，提示信息为一个代数表达式
    raise NotImplementedError("evaluate (b+c)**4 assuming b**3==2, c**2==3. \
                              Answer is 2*b + 8*c + 18*b**2 + 12*b*c + 9")


def test_H22():
    # 断言：将 x**4 - 3*x**2 + 1 在模数 5 下进行因式分解，得到结果 (x - 2)**2 * (x + 2)**2
    assert factor(x**4 - 3*x**2 + 1, modulus=5) == (x - 2)**2 * (x + 2)**2


def test_H23():
    # 定义多项式 f 和 g
    f = x**11 + x + 1
    g = (x**2 + x + 1) * (x**9 - x**8 + x**6 - x**5 + x**3 - x**2 + 1)
    # 断言：将多项式 f 在模数 65537 下进行因式分解，得到结果 g
    assert factor(f, modulus=65537) == g


def test_H24():
    # 定义黄金比例的代数数 phi
    phi = AlgebraicNumber(S.GoldenRatio.expand(func=True), alias='phi')
    # 断言：将 x**4 - 3*x**2 + 1 在扩展域 phi 下进行因式分解，得到结果 (x - phi)*(x + 1 - phi)*(x - 1 + phi)*(x + phi)
    assert factor(x**4 - 3*x**2 + 1, extension=phi) == \
        (x - phi)*(x + 1 - phi)*(x - 1 + phi)*(x + phi)


def test_H25():
    # 定义表达式 e
    e = (x - 2*y**2 + 3*z**3) ** 20
    # 断言：将表达式 e 展开并因式分解，得到原表达式 e
    assert factor(expand(e)) == e


def test_H26():
    # 定义表达式 g
    g = expand((sin(x) - 2*cos(y)**2 + 3*tan(z)**3)**20)
    # 断言：将表达式 g 在不展开的情况下进行因式分解，得到 (-sin(x) + 2*cos(y)**2 - 3*tan(z)**3)**20
    assert factor(g, expand=False) == (-sin(x) + 2*cos(y)**2 - 3*tan(z)**3)**20


def test_H27():
    # 定义多项式 f, g 和 h
    f = 24*x*y**19*z**8 - 47*x**17*y**5*z**8 + 6*x**15*y**9*z**2 - 3*x**22 + 5
    g = 34*x**5*y**8*z**13 + 20*x**7*y**7*z**7 + 12*x**9*y**16*z**4 + 80*y**14*z
    h = -2*z*y**7 \
        *(6*x**9*y**9*z**3 + 10*x**7*z**6 + 17*y*x**5*z**12 + 40*y**7) \
        *(3*x**22 + 47*x**17*y**5*z**8 - 6*x**15*y**9*z**2 - 24*x*y**19*z**8 - 5)
    # 断言：将表达式 f*g 展开并因式分解，得到结果 h
    assert factor(expand(f*g)) == h


@XFAIL
def test_H28():
    # 抛出 NotImplementedError 异常，提示信息为一个复杂的代数表达式
    raise NotImplementedError("expand ((1 - c**2)**5 * (1 - s**2)**5 * "
        + "(c**2 + s**2)**10) with c**2 + s**2 = 1. Answer is c**10*s**10.")


@XFAIL
def test_H29():
    # 断言：将多项式 4*x**2 - 21*x*y + 20*y**2 在模数 3 下进行因式分解，得到结果 (x + y)*(x - y)
    assert factor(4*x**2 - 21*x*y + 20*y**2, modulus=3) == (x + y)*(x - y)


def test_H30():
    # 断言：将多项式 x**3 + y**3 在扩展域 sqrt(-3) 下进行因式分解，得到结果 (x + y)*(x + y*(-R(1, 2) - sqrt(3)/2*I))*(x + y*(-R(1, 2) + sqrt(3)/2*I))
    test = factor(x**3 + y**3, extension=sqrt(-3))
    answer = (x + y)*(x + y*(-R(1, 2) - sqrt(3)/2*I))*(x + y*(-R(1, 2) + sqrt(3)/2*I))
    assert answer == test


def test_H31():
    # 定义表达式 f 和 g
    f = (x**2 + 2*x + 3)/(x**3 + 4*x**2 + 5*x + 2)
    g = 2 / (x + 1)**2 - 2 / (x + 1) + 3 / (x + 2)
    # 断言：将表达式 f 进行部分分式分解，得到结果 g
    assert apart(f) == g


@XFAIL
def test_H32():  # issue 6558
    # 抛出 NotImplementedError 异常，提示信息为一个复杂的代数表达式
    raise NotImplementedError("[A*B*C - (A*B*C)**(-1)]*A*C*B (product \
                              of a non-commuting product and its inverse)")


def test_H33():
    # 定义非交换符号 A, B, C
    A, B, C = symbols('A, B, C', commutative=False)
    # 断言：验证恒等式是否成立
    assert (Commutator(A, Commutator(B, C))
        + Commutator(B, Commutator(C, A))
        + Commutator(C, Commutator(A, B))).doit().expand() == 0


# I. Trigonometry

def test_I1():
    # 断言：验证正切函数在 pi * 7/10 处的值，结果为 -sqrt(1 + 2/sqrt(5))
    assert tan(pi*R(7, 10)) == -sqrt(1 + 2/sqrt(5))


@XFAIL
def test_I2():
    # 断言：验证一个三角函数的平方根，结果为 -cos(3)
    assert sqrt((1 + cos(6))/2) == -cos(3)


def test_I3():
    # 断言：验证三角函数表达式，结果为 (-1)**n - 1
    assert cos(n*pi) + sin((4*n - 1)*pi/2) == (-1)**n - 1


def test_I4():
    # 断言：验证三角函数表达式，结果为 (-1)**n - 1
    assert refine(cos(pi*cos(n*pi)) + sin(pi/2*cos(n*pi)), Q.integer(n)) == (-1)**n - 1


@XFAIL
def test_I5():
    # 此处未完整提供代码，不需要注释
    pass


这段代码是 Python 中的单元测试，针对数学表达式和函数进行了多个测试，并使用注释对每个测试的目的和预期结果进行了说明。
    # 断言语句，用于验证给定表达式的结果是否为零
    assert sin((n**5/5 + n**4/2 + n**3/3 - n/30) * pi) == 0
# 标记为待修复测试用例
@XFAIL
def test_I6():
    # 抛出未实现错误，假设条件为 -3*pi<x<-5*pi/2，要求 abs(cos(x)) == -cos(x)，abs(sin(x)) == -sin(x)
    raise NotImplementedError("assuming -3*pi<x<-5*pi/2, abs(cos(x)) == -cos(x), abs(sin(x)) == -sin(x)")

# 标记为待修复测试用例
@XFAIL
def test_I7():
    # 断言条件：cos(3*x)/cos(x) == cos(x)**2 - 3*sin(x)**2
    assert cos(3*x)/cos(x) == cos(x)**2 - 3*sin(x)**2

# 标记为待修复测试用例
@XFAIL
def test_I8():
    # 断言条件：cos(3*x)/cos(x) == 2*cos(2*x) - 1
    assert cos(3*x)/cos(x) == 2*cos(2*x) - 1

# 标记为待修复测试用例
@XFAIL
def test_I9():
    # 使用重写规则执行操作
    # 断言条件：cos(3*x)/cos(x) == cos(x)**2 - 3*sin(x)**2
    assert cos(3*x)/cos(x) == cos(x)**2 - 3*sin(x)**2

def test_I10():
    # 断言条件：trigsimp((tan(x)**2 + 1 - cos(x)**-2) / (sin(x)**2 + cos(x)**2 - 1)) 返回 NaN
    assert trigsimp((tan(x)**2 + 1 - cos(x)**-2) / (sin(x)**2 + cos(x)**2 - 1)) is nan

# 标记为跳过测试，原因是测试会卡住
@SKIP("hangs")
@XFAIL
def test_I11():
    # 断言条件：极限 limit((tan(x)**2 + 1 - cos(x)**-2) / (sin(x)**2 + cos(x)**2 - 1), x, 0) 不等于 0
    assert limit((tan(x)**2 + 1 - cos(x)**-2) / (sin(x)**2 + cos(x)**2 - 1), x, 0) != 0

# 标记为待修复测试用例
@XFAIL
def test_I12():
    # 这应该失败或者返回 NaN 或其他值
    # 计算 diff((tan(x)**2 + 1 - cos(x)**-2) / (sin(x)**2 + cos(x)**2 - 1), x) 的结果
    res = diff((tan(x)**2 + 1 - cos(x)**-2) / (sin(x)**2 + cos(x)**2 - 1), x)
    # 断言结果为 NaN，trigsimp(res) 也会返回 NaN
    assert res is nan # trigsimp(res) gives nan

# J. 特殊函数。

def test_J1():
    # 断言条件：bernoulli(16) 等于 -3617/510
    assert bernoulli(16) == R(-3617, 510)

def test_J2():
    # 断言条件：elliptic_e(x, y**2) 对 y 的偏导数等于 (elliptic_e(x, y**2) - elliptic_f(x, y**2))/y
    assert diff(elliptic_e(x, y**2), y) == (elliptic_e(x, y**2) - elliptic_f(x, y**2))/y

# 标记为待修复测试用例
@XFAIL
def test_J3():
    # 抛出未实现错误，Jacobi 椭圆函数：diff(dn(u,k), u) == -k**2*sn(u,k)*cn(u,k)
    raise NotImplementedError("Jacobi elliptic functions: diff(dn(u,k), u) == -k**2*sn(u,k)*cn(u,k)")

def test_J4():
    # 断言条件：gamma(-1/2) 等于 -2*sqrt(pi)
    assert gamma(R(-1, 2)) == -2*sqrt(pi)

def test_J5():
    # 断言条件：polygamma(0, 1/3) 等于 -log(3) - sqrt(3)*pi/6 - EulerGamma - log(sqrt(3))
    assert polygamma(0, R(1, 3)) == -log(3) - sqrt(3)*pi/6 - EulerGamma - log(sqrt(3))

def test_J6():
    # 断言条件：mpmath.besselj(2, 1 + 1j) 约等于 0.04157988694396212 + 0.24739764151330632j
    assert mpmath.besselj(2, 1 + 1j).ae(mpc('0.04157988694396212', '0.24739764151330632'))

def test_J7():
    # 断言条件：simplify(besselj(-5/2, pi/2)) 等于 12/(pi**2)
    assert simplify(besselj(R(-5,2), pi/2)) == 12/(pi**2)

def test_J8():
    # 定义 p 和 q
    p = besselj(R(3,2), z)
    q = (sin(z)/z - cos(z))/sqrt(pi*z/2)
    # 断言条件：expand_func(p) - q 简化后等于 0
    assert simplify(expand_func(p) - q) == 0

def test_J9():
    # 断言条件：besselj(0, z).diff(z) 等于 -besselj(1, z)
    assert besselj(0, z).diff(z) == - besselj(1, z)

def test_J10():
    # 定义 mu 和 nu
    mu, nu = symbols('mu, nu', integer=True)
    # 断言条件：assoc_legendre(nu, mu, 0) 等于 2**mu*sqrt(pi)/gamma((nu - mu)/2 + 1)/gamma((-nu - mu + 1)/2)
    assert assoc_legendre(nu, mu, 0) == 2**mu*sqrt(pi)/gamma((nu - mu)/2 + 1)/gamma((-nu - mu + 1)/2)

def test_J11():
    # 断言条件：simplify(assoc_legendre(3, 1, x)) 等于 -3/2*sqrt(1 - x**2)*(5*x**2 - 1)
    assert simplify(assoc_legendre(3, 1, x)) == simplify(-R(3, 2)*sqrt(1 - x**2)*(5*x**2 - 1))

# 标记为耗时测试
@slow
def test_J12():
    # 断言条件：simplify(chebyshevt(1008, x) - 2*x*chebyshevt(1007, x) + chebyshevt(1006, x)) 等于 0
    assert simplify(chebyshevt(1008, x) - 2*x*chebyshevt(1007, x) + chebyshevt(1006, x)) == 0

def test_J13():
    # 定义 a
    a = symbols('a', integer=True, negative=False)
    # 断言条件：chebyshevt(a, -1) 等于 (-1)**a
    assert chebyshevt(a, -1) == (-1)**a

def test_J14():
    # 定义 p
    p = hyper([S.Half, S.Half], [R(3, 2)], z**2)
    # 断言条件：hyperexpand(p) 等于 asin(z)/z
    assert hyperexpand(p) == asin(z)/z

# 标记为待修复测试用例
@XFAIL
def test_J15():
    # 抛出未实现错误，F((n+2)/2,-(n-2)/2,R(3,2),sin(z)**2) == sin(n*z)/(n*sin(z)*cos(z)); F(.) is hypergeometric function
    raise NotImplementedError("F((n+2)/2,-(n-2)/2,R(3,2),sin(z)**2) == sin(n*z)/(n*sin(z)*cos(z)); F(.) is hypergeometric function")

# 标记为待修复测试用例
@XFAIL
def test_J16():
    # 抛出未实现错误，diff(zeta(x), x) @ x=0 == -log(2*pi)/2
    raise NotImplementedError("diff(zeta(x), x) @ x=0 == -log(2*pi)/2")

def test_J17():
    # 断言条件：int
def test_K2():
    # 断言：验证复数表达式的绝对误差是否等于1
    assert abs(3 - sqrt(7) + I*sqrt(6*sqrt(7) - 15)) == 1


@XFAIL
def test_K3():
    # 定义符号变量a, b，并断言简化后的复数表达式的绝对值是否等于1除以a和(a + b*i)的模长
    a, b = symbols('a, b', real=True)
    assert simplify(abs(1/(a + I/a + I*b))) == 1/sqrt(a**2 + (I/a + b)**2)


def test_K4():
    # 断言对复数3 + 4*i的对数展开是否等于对数5加上i乘以arctan(4/3)
    assert log(3 + 4*I).expand(complex=True) == log(5) + I*atan(R(4, 3))


def test_K5():
    # 定义实数符号变量x, y，并断言对tan(x + i*y)展开后的结果是否等于复数表达式
    assert tan(x + I*y).expand(complex=True) == (sin(2*x)/(cos(2*x) +
        cosh(2*y)) + I*sinh(2*y)/(cos(2*x) + cosh(2*y)))


def test_K6():
    # 断言复数表达式是否成立：sqrt(x*y*|z|^2) / (sqrt(x)*|z|) 等于 sqrt(x*y) / sqrt(x)
    assert sqrt(x*y*abs(z)**2)/(sqrt(x)*abs(z)) == sqrt(x*y)/sqrt(x)
    # 断言复数表达式是否不等于 sqrt(y)
    assert sqrt(x*y*abs(z)**2)/(sqrt(x)*abs(z)) != sqrt(y)


def test_K7():
    # 定义实数符号变量y，并计算表达式 sqrt(x*y*|z|^2) / (sqrt(x)*|z|) 的简化形式
    y = symbols('y', real=True, negative=False)
    expr = sqrt(x*y*abs(z)**2)/(sqrt(x)*abs(z))
    sexpr = simplify(expr)
    # 断言简化后的表达式是否等于 sqrt(y)
    assert sexpr == sqrt(y)


def test_K8():
    # 定义复数符号变量z，并断言对复数表达式 sqrt(1/z) - 1/sqrt(z) 的简化结果不等于0（应通过）
    z = symbols('z', complex=True)
    assert simplify(sqrt(1/z) - 1/sqrt(z)) != 0  # Passes
    # 重新定义复数符号变量z，并断言对复数表达式 sqrt(1/z) - 1/sqrt(z) 的简化结果等于0（应失败）
    z = symbols('z', complex=True, negative=False)
    assert simplify(sqrt(1/z) - 1/sqrt(z)) == 0  # Fails


def test_K9():
    # 定义正数符号变量z，并断言对复数表达式 sqrt(1/z) - 1/sqrt(z) 的简化结果等于0
    z = symbols('z', positive=True)
    assert simplify(sqrt(1/z) - 1/sqrt(z)) == 0


def test_K10():
    # 定义负数符号变量z，并断言对复数表达式 sqrt(1/z) + 1/sqrt(z) 的简化结果等于0
    z = symbols('z', negative=True)
    assert simplify(sqrt(1/z) + 1/sqrt(z)) == 0


# L. Determining Zero Equivalence


def test_L1():
    # 断言是否满足 sqrt(997) - (997**3)**(1/6) 等于0
    assert sqrt(997) - (997**3)**R(1, 6) == 0


def test_L2():
    # 断言是否满足 sqrt(999983) - (999983**3)**(1/6) 等于0
    assert sqrt(999983) - (999983**3)**R(1, 6) == 0


def test_L3():
    # 断言是否满足 ((2**(1/3) + 4**(1/3))**3 - 6*(2**(1/3) + 4**(1/3)) - 6) 简化后等于0
    assert simplify((2**R(1, 3) + 4**R(1, 3))**3 - 6*(2**R(1, 3) + 4**R(1, 3)) - 6) == 0


def test_L4():
    # 断言是否满足 trigsimp(cos(x)**3 + cos(x)*sin(x)**2 - cos(x)) 等于0
    assert trigsimp(cos(x)**3 + cos(x)*sin(x)**2 - cos(x)) == 0


@XFAIL
def test_L5():
    # 断言是否满足 log(tan(π/2*x + π/4)) - asinh(tan(x)) 等于0（应失败）
    assert log(tan(R(1, 2)*x + pi/4)) - asinh(tan(x)) == 0


def test_L6():
    # 断言在x=0时，表达式(log(tan(x/2 + π/4)) - asinh(tan(x))).diff(x) 的结果等于0
    assert (log(tan(x/2 + pi/4)) - asinh(tan(x))).diff(x).subs({x: 0}) == 0


@XFAIL
def test_L7():
    # 断言是否满足简化后的表达式 log((2*sqrt(x) + 1)/(sqrt(4*x + 4*sqrt(x) + 1))) 等于0（应失败）
    assert simplify(log((2*sqrt(x) + 1)/(sqrt(4*x + 4*sqrt(x) + 1)))) == 0


@XFAIL
def test_L8():
    # 断言是否满足简化后的表达式 ((4*x + 4*sqrt(x) + 1)**(sqrt(x)/(2*sqrt(x) + 1))
    # *(2*sqrt(x) + 1)**(1/(2*sqrt(x) + 1)) - 2*sqrt(x) - 1) 等于0（应失败）
    assert simplify((4*x + 4*sqrt(x) + 1)**(sqrt(x)/(2*sqrt(x) + 1)) \
        *(2*sqrt(x) + 1)**(1/(2*sqrt(x) + 1)) - 2*sqrt(x) - 1) == 0


@XFAIL
def test_L9():
    # 定义复数符号变量z，并断言是否满足简化后的表达式 2**(1 - z)*gamma(z)*zeta(z)*cos(z*pi/2) - pi**2*zeta(1 - z) 等于0（应失败）
    z = symbols('z', complex=True)
    assert simplify(2**(1 - z)*gamma(z)*zeta(z)*cos(z*pi/2) - pi**2*zeta(1 - z)) == 0

# M. Equations


@XFAIL
def test_M1():
    # 断言是否满足等式 x = 2 的一半加上等式 1 = 1
    assert Equality(x, 2)/2 + Equality(1, 1) == Equality(x/2 + 1, 2)


def test_M2():
    # 断言解3*x**3 - 18*x**2 + 33*x - 19的所有根都是实数（不验证根的正确性）
    sol = solveset(3*x**3 - 18*x**2 + 33*x - 19, x)
    assert all(s.expand(complex=True).is_real for s in sol)


@XFAIL
def test_M5():
    # 断言解x**6 - 9*x**4 - 4*x**3 + 27*x**2 - 36*x - 23的所有根是否等于指定的有限集合（应失败）
    assert solveset(x**6 - 9*x**4 - 4*x**3 + 27*x**2 - 36*x - 23, x) == FiniteSet(2**(1/3) + sqrt(3), 2**(1/3) - sqrt(3), +sqrt(3) - 1/2**(2/3) + I*sqrt(3)/2**(2/3), +sqrt(3) -
    # 论文要求使用 exp 项，但也可能接受 sin 和 cos 项；
    # 如果结果被简化，exp 项会出现在所有情况下，除了
    # -sin(pi/14) - I*cos(pi/14) 和 -sin(pi/14) + I*cos(pi/14)
    # 如果应用变换 foo.rewrite(exp).expand()，这些将会被简化。
def test_M7():
    # TODO: Replace solve with solveset, as of now test fails for solveset
    # 断言求解方程 x**8 - 8*x**7 + 34*x**6 - 92*x**5 + 175*x**4 - 236*x**3 + 226*x**2 - 140*x + 46 的解集
    assert set(solve(x**8 - 8*x**7 + 34*x**6 - 92*x**5 + 175*x**4 - 236*x**3 +
        226*x**2 - 140*x + 46, x)) == {
        # 列举方程的根，使用集合包含复数解
        1 - sqrt(2)*I*sqrt(-sqrt(-3 + 4*sqrt(3)) + 3)/2,
        1 - sqrt(2)*sqrt(-3 + I*sqrt(3 + 4*sqrt(3)))/2,
        1 - sqrt(2)*I*sqrt(sqrt(-3 + 4*sqrt(3)) + 3)/2,
        1 - sqrt(2)*sqrt(-3 - I*sqrt(3 + 4*sqrt(3)))/2,
        1 + sqrt(2)*I*sqrt(sqrt(-3 + 4*sqrt(3)) + 3)/2,
        1 + sqrt(2)*sqrt(-3 - I*sqrt(3 + 4*sqrt(3)))/2,
        1 + sqrt(2)*sqrt(-3 + I*sqrt(3 + 4*sqrt(3)))/2,
        1 + sqrt(2)*I*sqrt(-sqrt(-3 + 4*sqrt(3)) + 3)/2,
    }


@XFAIL  # There are an infinite number of solutions.
def test_M8():
    x = Symbol('x')
    z = symbols('z', complex=True)
    # 断言求解方程 exp(2*x) + 2*exp(x) + 1 - z = 0 的实数解集
    assert solveset(exp(2*x) + 2*exp(x) + 1 - z, x, S.Reals) == \
        FiniteSet(log(1 + z - 2*sqrt(z))/2, log(1 + z + 2*sqrt(z))/2)
    # 该断言可以更好地简化（1/2 可以作为平方根的一部分提取到对数中，对数内的函数可以因数分解为平方）
    # 还应该有无限多个解。
    # x = {log(sqrt(z) - 1), log(sqrt(z) + 1) + i pi} [+ n 2 pi i, + n 2 pi i]
    # 其中 n 是任意整数。详见上述详细输出的 URL。


@XFAIL
def test_M9():
    # x = symbols('x')
    # 解集是 1/2*(1 +/- sqrt(9 + 8*I*pi*n))，其中 n 是整数
    raise NotImplementedError("solveset(exp(2-x**2)-exp(-x),x) has complex solutions.")


def test_M10():
    # TODO: Replace solve with solveset when it gives Lambert solution
    # 断言求解方程 exp(x) - x = 0 的解集
    assert solve(exp(x) - x, x) == [-LambertW(-1)]


@XFAIL
def test_M11():
    # 断言求解方程 x**x - x = 0 的解集
    assert solveset(x**x - x, x) == FiniteSet(-1, 1)


def test_M12():
    # TODO: x = [-1, 2*(+/-asinh(1)*I + n*pi}, 3*(pi/6 + n*pi/3)]
    # TODO: Replace solve with solveset, as of now test fails for solveset
    # 断言求解方程 (x + 1)*(sin(x)**2 + 1)**2*cos(3*x)**3 = 0 的解集
    assert solve((x + 1)*(sin(x)**2 + 1)**2*cos(3*x)**3, x) == [
        -1, pi/6, pi/2,
           - I*log(1 + sqrt(2)),      I*log(1 + sqrt(2)),
        pi - I*log(1 + sqrt(2)), pi + I*log(1 + sqrt(2)),
    ]


@XFAIL
def test_M13():
    n = Dummy('n')
    # 断言求解实数域中的方程 sin(x) - cos(x) = 0 的解集
    assert solveset_real(sin(x) - cos(x), x) == ImageSet(Lambda(n, n*pi - pi*R(7, 4)), S.Integers)


@XFAIL
def test_M14():
    n = Dummy('n')
    # 断言求解实数域中的方程 tan(x) - 1 = 0 的解集
    assert solveset_real(tan(x) - 1, x) == ImageSet(Lambda(n, n*pi + pi/4), S.Integers)


def test_M15():
    n = Dummy('n')
    got = solveset(sin(x) - S.Half)
    # 断言求解方程 sin(x) - 1/2 = 0 的解集，检查是否存在满足条件的解集
    assert any(got.dummy_eq(i) for i in (
        Union(ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi*R(5, 6)), S.Integers)),
        Union(ImageSet(Lambda(n, 2*n*pi + pi*R(5, 6)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers))))


@XFAIL
def test_M16():
    n = Dummy('n')
    # 断言求解方程 sin(x) - tan(x) = 0 的解集
    assert solveset(sin(x) - tan(x), x) == ImageSet(Lambda(n, n*pi), S.Integers)


@XFAIL
def test_M17():
    # 断言求解实数域中的方程 asin(x) - atan(x) = 0 的解集
    assert solveset_real(asin(x) - atan(x), x) == FiniteSet(0)
@XFAIL
# 标记为预期失败的测试函数，表示预期该测试无法通过
def test_M18():
    # 断言解 acos(x) - atan(x) 的实数解为 FiniteSet(sqrt((sqrt(5) - 1)/2))
    assert solveset_real(acos(x) - atan(x), x) == FiniteSet(sqrt((sqrt(5) - 1)/2))


def test_M19():
    # TODO: 现在的解法失败，需要用 solveset 替换 solve
    # 断言解 ((x - 2) / x**(1/3)) 的解为 [2]
    assert solve((x - 2)/x**R(1, 3), x) == [2]


def test_M20():
    # 断言解 sqrt(x**2 + 1) - x + 2 的解为空集
    assert solveset(sqrt(x**2 + 1) - x + 2, x) == EmptySet


def test_M21():
    # 断言解 x + sqrt(x) - 2 的解为 FiniteSet(1)
    assert solveset(x + sqrt(x) - 2) == FiniteSet(1)


def test_M22():
    # 断言解 2*sqrt(x) + 3*x**(1/4) - 2 的解为 FiniteSet(1/16)
    assert solveset(2*sqrt(x) + 3*x**R(1, 4) - 2) == FiniteSet(R(1, 16))


def test_M23():
    x = symbols('x', complex=True)
    # TODO: 现在的解法失败，需要用 solveset 替换 solve
    # 断言解 x - 1/sqrt(1 + x**2) 的复数解为 [-I*sqrt(1/2 + sqrt(5)/2), sqrt(-1/2 + sqrt(5)/2)]
    assert solve(x - 1/sqrt(1 + x**2)) == [
        -I*sqrt(S.Half + sqrt(5)/2), sqrt(Rational(-1, 2) + sqrt(5)/2)]


def test_M24():
    # TODO: 现在的解法失败，需要用 solveset 替换 solve
    # 求解 1 - binomial(m, 2)*2**k = 0 关于 k 的解，验证第一个解的展开是否等于预期的答案
    solution = solve(1 - binomial(m, 2)*2**k, k)
    answer = log(2/(m*(m - 1)), 2)
    assert solution[0].expand() == answer.expand()


def test_M25():
    a, b, c, d = symbols(':d', positive=True)
    x = symbols('x')
    # TODO: 现在的解法失败，需要用 solveset 替换 solve
    # 断言解 a*b**x - c*d**x 的解中第一个解的展开是否等于 log(c/a)/log(b/d) 的展开
    assert solve(a*b**x - c*d**x, x)[0].expand() == (log(c/a)/log(b/d)).expand()


def test_M26():
    # TODO: 现在的解法失败，需要用 solveset 替换 solve
    # 断言解 sqrt(log(x)) - log(sqrt(x)) 的解为 [1, exp(4)]
    assert solve(sqrt(log(x)) - log(sqrt(x))) == [1, exp(4)]


def test_M27():
    x = symbols('x', real=True)
    b = symbols('b', real=True)
    # TODO: 现在的解法失败，需要用 solveset 替换 solve，并且希望得到两个解 [+/- 当前答案]
    # 注意 wester.pdf 中的方程有拼写错误，实际上没有实数解
    assert solve(log(acos(asin(x**R(2, 3) - b)) - 1) + 2, x) == [(b + sin(cos(exp(-2) + 1)))**R(3, 2)]


@XFAIL
# 标记为预期失败的测试函数，表示预期该测试无法通过
def test_M28():
    # 断言解 5*x + exp((x - 5)/2) - 8*x**3 的实数解为 [-0.784966, -0.016291, 0.802557]
    assert solveset_real(5*x + exp((x - 5)/2) - 8*x**3, x, assume=Q.real(x)) == [-0.784966, -0.016291, 0.802557]


def test_M29():
    x = symbols('x')
    # 断言解 |x - 1| - 2 在实数域内的解为 FiniteSet(-1, 3)
    assert solveset(abs(x - 1) - 2, domain=S.Reals) == FiniteSet(-1, 3)


def test_M30():
    # TODO: 现在的解法失败，需要用 solveset 替换 solve，因为 solveset 不支持假设
    # 断言解 |2*x + 5| - |x - 2| 的实数解为 FiniteSet(-1, -7)
    assert solveset_real(abs(2*x + 5) - abs(x - 2), x) == FiniteSet(-1, -7)


def test_M31():
    # TODO: 现在的解法失败，需要用 solveset 替换 solve，因为 solveset 不支持假设
    # 断言解 1 - |x| - max(-x - 2, x - 2) 的实数解为 FiniteSet(-3/2, 3/2)
    assert solveset_real(1 - abs(x) - Max(-x - 2, x - 2), x) == FiniteSet(R(-3, 2), R(3, 2))


@XFAIL
# 标记为预期失败的测试函数，表示预期该测试无法通过
def test_M32():
    # TODO: 现在的解法失败，需要用 solveset 替换 solve，因为 solveset 不支持假设
    # 断言解 Max(2 - x**2, x) - Max(-x, x**3/9) 的实数解为 FiniteSet(-1, 3)
    assert solveset_real(Max(2 - x**2, x) - Max(-x, (x**3)/9), x) == FiniteSet(-1, 3)


@XFAIL
# 标记为预期失败的测试函数，表示预期该测试无法通过
def test_M33():
    # TODO: 现在的解法失败，需要用 solveset 替换 solve，因为 solveset 不支持假设
    pass
    # 断言语句，用于验证解集
    assert solveset_real(Max(2 - x**2, x) - x**3/9, x) == FiniteSet(-3, -1.554894, 3)
@XFAIL
# 标记该测试为预期失败的测试用例
def test_M34():
    # 声明一个复数符号变量 z
    z = symbols('z', complex=True)
    # 断言解集求解给定复杂方程，预期结果是一个有限集合，包含复数 2 + 3*I
    assert solveset((1 + I) * z + (2 - I) * conjugate(z) + 3*I, z) == FiniteSet(2 + 3*I)


def test_M35():
    # 声明两个实数符号变量 x 和 y
    x, y = symbols('x y', real=True)
    # 断言解线性方程组的解集，预期结果是一个有限集合，包含元组 (3, 2)
    assert linsolve((3*x - 2*y - I*y + 3*I).as_real_imag(), y, x) == FiniteSet((3, 2))


def test_M36():
    # TODO: Replace solve with solveset, as of now
    # solveset doesn't supports solving for function
    # 断言用 solveset 解非线性方程，预期结果是一个有限集合，包含 -2 和 1
    assert solveset(f(x)**2 + f(x) - 2, f(x)) == FiniteSet(-2, 1)


def test_M37():
    # 断言解线性方程组的解集，预期结果是一个有限集合，包含元组 (-z + 4, 2, z)
    assert linsolve([x + y + z - 6, 2*x + y + 2*z - 10, x + 3*y + z - 10 ], x, y, z) == \
        FiniteSet((-z + 4, 2, z))


def test_M38():
    # 声明三个符号变量 a, b, c
    a, b, c = symbols('a, b, c')
    # 声明一个分式域对象，指定环的域为整数
    domain = FracField([a, b, c], ZZ).to_domain()
    # 声明一个多项式环对象，其中包含 50 个生成器
    ring = PolyRing('k1:50', domain)
    # 将生成器分配给多个变量
    (k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
    k11, k12, k13, k14, k15, k16, k17, k18, k19, k20,
    k21, k22, k23, k24, k25, k26, k27, k28, k29, k30,
    k31, k32, k33, k34, k35, k36, k37, k38, k39, k40,
    k41, k42, k43, k44, k45, k46, k47, k48, k49) = ring.gens

    # 声明一个期望的解决方案，其中多项式环的生成器对应值为零
    solution = {
        k49: 0, k48: 0, k47: 0, k46: 0, k45: 0, k44: 0, k41: 0, k40: 0,
        k38: 0, k37: 0, k36: 0, k35: 0, k33: 0, k32: 0, k30: 0, k29: 0,
        k28: 0, k27: 0, k25: 0, k24: 0, k22: 0, k21: 0, k20: 0, k19: 0,
        k18: 0, k17: 0, k16: 0, k15: 0, k14: 0, k13: 0, k12: 0, k11: 0,
        k10: 0, k9:  0, k8:  0, k7:  0, k6:  0, k5:  0, k4:  0, k3:  0,
        k2:  0, k1:  0,
        k34: b/c*k42, k31: k39, k26: a/c*k42, k23: k39
    }
    # 断言解线性系统，预期结果等于声明的解决方案
    assert solve_lin_sys(system, ring) == solution


def test_M39():
    # 声明三个复数符号变量 x, y, z
    x, y, z = symbols('x y z', complex=True)
    # TODO: Replace solve with solveset, as of now
    # solveset doesn't supports non-linear multivariate
    # 断言解非线性多元方程组，预期结果是包含多个字典的列表，每个字典表示一个解
    assert solve([x**2*y + 3*y*z - 4, -3*x**2*z + 2*y**2 + 1, 2*y*z**2 - z**2 - 1 ]) ==\
            [{y: 1, z: 1, x: -1}, {y: 1, z: 1, x: 1},\
             {y: sqrt(2)*I, z: R(1,3) - sqrt(2)*I/3, x: -sqrt(-1 - sqrt(2)*I)},\
             {y: sqrt(2)*I, z: R(1,3) - sqrt(2)*I/3, x: sqrt(-1 - sqrt(2)*I)},\
             {y: -sqrt(2)*I, z: R(1,3) + sqrt(2)*I/3, x: -sqrt(-1 + sqrt(2)*I)},\
             {y: -sqrt(2)*I, z: R(1,3) + sqrt(2)*I/3, x: sqrt(-1 + sqrt(2)*I)}]

# N. Inequalities


def test_N1():
    # 断言给定不等式是否成立
    assert ask(E**pi > pi**E)


@XFAIL
def test_N2():
    # 声明一个实数符号变量 x
    x = symbols('x', real=True)
    # 断言给定不等式是否成立
    assert ask(x**4 - x + 1 > 0) is True
    # 断言给定不等式是否成立
    assert ask(x**4 - x + 1 > 1) is False


@XFAIL
def test_N3():
    # 声明一个实数符号变量 x
    x = symbols('x', real=True)
    # 断言给定不等式是否成立
    assert ask(And(Lt(-1, x), Lt(x, 1)), abs(x) < 1 )

@XFAIL
def test_N4():
    # 声明两个实数符号变量 x, y
    x, y = symbols('x y', real=True)
    # 断言给定不等式是否成立
    assert ask(2*x**2 > 2*y**2, (x > y) & (y > 0)) is True


@XFAIL
def test_N5():
    # 声明三个实数符号变量 x, y, k
    x, y, k = symbols('x y k', real=True)
    # 断言给定不等式是否成立
    assert ask(k*x**2 > k*y**2, (x > y) & (y > 0) & (k > 0)) is True


@slow
@XFAIL
def test_N6():
    # 声明四个实数符号变量 x, y, k, n
    x, y, k, n = symbols('x y k n', real=True)
    # 断言给定不等式是否成立
    assert ask(k*x**n > k*y**n, (x > y) & (y > 0) & (k > 0) & (n > 0)) is True


@XFAIL
def test_N7():
    # 空测试，没有代码
    # 创建符号变量 x 和 y，指定它们为实数
    x, y = symbols('x y', real=True)
    # 使用符号逻辑推理，断言 y > 0 在条件 (x > 1) & (y >= x - 1) 下为真
    assert ask(y > 0, (x > 1) & (y >= x - 1)) is True
@XFAIL
@slow
# 标记为预期失败的测试用例，而且执行速度较慢
def test_N8():
    # 定义符号变量 x, y, z，限定其为实数
    x, y, z = symbols('x y z', real=True)
    # 断言条件：询问 x=y, y=z 是否成立，并同时检查 x>=y, y>=z, z>=x 是否成立
    assert ask(Eq(x, y) & Eq(y, z),
               (x >= y) & (y >= z) & (z >= x))


def test_N9():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言条件：求解绝对值不等式 |x-1| > 2 在实数域上的解集
    assert solveset(abs(x - 1) > 2, domain=S.Reals) == Union(Interval(-oo, -1, False, True),
                                             Interval(3, oo, True))


def test_N10():
    # 定义符号变量 x
    x = Symbol('x')
    # 定义多项式 p
    p = (x - 1)*(x - 2)*(x - 3)*(x - 4)*(x - 5)
    # 断言条件：求解多项式 p 扩展后小于 0 在实数域上的解集
    assert solveset(expand(p) < 0, domain=S.Reals) == Union(Interval(-oo, 1, True, True),
                                            Interval(2, 3, True, True),
                                            Interval(4, 5, True, True))


def test_N11():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言条件：求解不等式 6/(x-3) <= 3 在实数域上的解集
    assert solveset(6/(x - 3) <= 3, domain=S.Reals) == Union(Interval(-oo, 3, True, True), Interval(5, oo))


def test_N12():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言条件：求解开平方不等式 sqrt(x) < 2 在实数域上的解集
    assert solveset(sqrt(x) < 2, domain=S.Reals) == Interval(0, 4, False, True)


def test_N13():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言条件：求解三角函数不等式 sin(x) < 2 在实数域上的解集
    assert solveset(sin(x) < 2, domain=S.Reals) == S.Reals


@XFAIL
def test_N14():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言条件：求解三角函数不等式 sin(x) < 1 在实数域上的解集
    # 注意：预期这个测试用例会失败，因为当前只支持单变量不等式
    assert solveset(sin(x) < 1, x, domain=S.Reals) == Union(Interval(-oo, pi/2, True, True),
                                         Interval(pi/2, oo, True, True))


def test_N15():
    # 定义符号变量 r, t
    r, t = symbols('r t')
    # 抛出 NotImplementedError: 只支持单变量不等式
    solveset(abs(2*r*(cos(t) - 1) + 1) <= 1, r, S.Reals)


def test_N16():
    # 定义符号变量 r, t
    r, t = symbols('r t')
    # 求解不等式 (r**2)*((cos(t) - 4)**2)*sin(t)**2 < 9 在实数域上的解集
    solveset((r**2)*((cos(t) - 4)**2)*sin(t)**2 < 9, r, S.Reals)


@XFAIL
def test_N17():
    # 目前只支持单变量不等式
    assert solveset((x + y > 0, x - y < 0), (x, y)) == (abs(x) < y)


def test_O1():
    # 定义复数矩阵 M
    M = Matrix((1 + I, -2, 3*I))
    # 断言条件：对矩阵 M 的共轭转置与平方根的展开结果进行比较
    assert sqrt(expand(M.dot(M.H))) == sqrt(15)


def test_O2():
    # 断言条件：对向量 (2, 2, -3) 和 (1, 3, 1) 进行叉乘
    assert Matrix((2, 2, -3)).cross(Matrix((1, 3, 1))) == Matrix([[11],
                                                                  [-5],
                                                                  [4]])

# 向量模块无法不考虑基础表示的情况下表示向量
@XFAIL
def test_O3():
    # 抛出 NotImplementedError: 向量模块无法不考虑基础表示的情况下表示向量
    raise NotImplementedError("""The vector module has no way of representing
        vectors symbolically (without respect to a basis)""")


def test_O4():
    # 导入 3D 坐标系和 Del 操作
    from sympy.vector import CoordSys3D, Del
    N = CoordSys3D("N")
    delop = Del()
    # 获取基向量和基标量
    i, j, k = N.base_vectors()
    x, y, z = N.base_scalars()
    # 定义向量场 F
    F = i*(x*y*z) + j*((x*y*z)**2) + k*((y**2)*(z**3))
    # 断言条件：对向量场 F 的旋度进行计算，并验证结果
    assert delop.cross(F).doit() == (-2*x**2*y**2*z + 2*y*z**3)*i + x*y*j + (2*x*y**2*z**2 - x*z)*k

@XFAIL
def test_O5():
    #assert grad|(f^g)-g|(grad^f)+f|(grad^g)  == 0
    # 断言条件：预期这个测试用例会失败
    pass
    # 抛出 NotImplementedError 异常，表示向量模块无法以符号方式（不考虑基底）表示向量
    raise NotImplementedError("""The vector module has no way of representing
        vectors symbolically (without respect to a basis)""")
# 定义测试函数 test_O10，用于测试 Gram-Schmidt 正交化过程
def test_O10():
    # 创建矩阵列表 L，包含三个矩阵
    L = [Matrix([2, 3, 5]), Matrix([3, 6, 2]), Matrix([8, 3, 6])]
    # 断言 GramSchmidt 函数对 L 的应用结果与期望结果相同
    assert GramSchmidt(L) == [Matrix([
                              [2],
                              [3],
                              [5]]),
                              Matrix([
                              [R(23, 19)],
                              [R(63, 19)],
                              [R(-47, 19)]]),
                              Matrix([
                              [R(1692, 353)],
                              [R(-1551, 706)],
                              [R(-423, 706)]])]


# 定义测试函数 test_P1，验证 Matrix 对象的对角线偏移功能
def test_P1():
    # 断言对 Matrix(3, 3, lambda i, j: j - i) 对象进行 -1 偏移后的结果符合期望
    assert Matrix(3, 3, lambda i, j: j - i).diagonal(-1) == Matrix(
        1, 2, [-1, -1])


# 定义测试函数 test_P2，验证 Matrix 对象的行列删除功能
def test_P2():
    # 创建矩阵 M
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 删除 M 的第 1 行和第 2 列
    M.row_del(1)
    M.col_del(2)
    # 断言 M 的结果与期望的矩阵相同
    assert M == Matrix([[1, 2],
                        [7, 8]])


# 定义测试函数 test_P3，验证 Matrix 对象的切片和 BlockMatrix 功能
def test_P3():
    # 创建矩阵 A
    A = Matrix([
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44]])

    # 对矩阵 A 进行不同的切片操作，构造 BlockMatrix，并断言引发 ValueError
    A11 = A[0:3, 1:4]
    A12 = A[(0, 1, 3), (2, 0, 3)]
    A21 = A
    A221 = -A[0:2, 2:4]
    A222 = -A[(3, 0), (2, 1)]
    A22 = BlockMatrix([[A221, A222]]).T
    rows = [[-A11, A12], [A21, A22]]
    raises(ValueError, lambda: BlockMatrix(rows))
    # 构造新的 Matrix 对象 B，断言其结果与期望的矩阵相同
    B = Matrix(rows)
    assert B == Matrix([
        [-12, -13, -14, 13, 11, 14],
        [-22, -23, -24, 23, 21, 24],
        [-32, -33, -34, 43, 41, 44],
        [11, 12, 13, 14, -13, -23],
        [21, 22, 23, 24, -14, -24],
        [31, 32, 33, 34, -43, -13],
        [41, 42, 43, 44, -42, -12]])


# 定义测试函数 test_P5，验证 Matrix 对象的模运算功能
def test_P5():
    # 创建矩阵 M
    M = Matrix([[7, 11],
                [3, 8]])
    # 断言 M 对 2 取模的结果与期望的矩阵相同
    assert  M % 2 == Matrix([[1, 1],
                             [1, 0]])


# 定义测试函数 test_P6，验证 Matrix 对象的微分功能
def test_P6():
    # 创建矩阵 M
    M = Matrix([[cos(x), sin(x)],
                [-sin(x), cos(x)]])
    # 断言 M 对 x 进行二阶微分的结果与期望的矩阵相同
    assert M.diff(x, 2) == Matrix([[-cos(x), -sin(x)],
                                   [sin(x), -cos(x)]])


# 定义测试函数 test_P7，验证 Matrix 对象的矩阵乘法和加法功能
def test_P7():
    # 创建矩阵 M
    M = Matrix([[x, y]])*(
        z*Matrix([[1, 3, 5],
                  [2, 4, 6]]) + Matrix([[7, -9, 11],
                                        [-8, 10, -12]]))
    # 断言 M 的结果与期望的矩阵相同
    assert M == Matrix([[x*(z + 7) + y*(2*z - 8), x*(3*z - 9) + y*(4*z + 10),
                         x*(5*z + 11) + y*(6*z - 12)]])


# 定义测试函数 test_P8，验证 Matrix 对象的无穷范数计算功能
def test_P8():
    # 创建矩阵 M
    M = Matrix([[1, -2*I],
                [-3*I, 4]])
    # 断言 M 的无穷范数与期望的结果相等
    assert M.norm(ord=S.Infinity) == 7


# 定义测试函数 test_P9，验证 Matrix 对象的范数计算和因式分解功能
def test_P9():
    # 创建符号变量 a, b, c，并构造矩阵 M
    a, b, c = symbols('a b c', nonzero=True)
    M = Matrix([[a/(b*c), 1/c, 1/b],
                [1/c, b/(a*c), 1/a],
                [1/b, 1/a, c/(a*b)]])
    # 断言 M 的 Frobenius 范数的因式分解结果与期望的表达式相同
    assert factor(M.norm('fro')) == (a**2 + b**2 + c**2)/(abs(a)*abs(b)*abs(c))


# 定义测试函数 test_P10，用于测试 Matrix 对象的共轭转置和 XFAIL 标记
@XFAIL
def test_P10():
    # 创建矩阵 M
    M = Matrix([[1, 2 + 3*I],
                [f(4 - 5*I), 6]])
    # 断言 M 的共轭转置结果与期望的矩阵相同
    # 由于 XFAIL 标记，此处会引发 NotImplementedError
    assert M.H == Matrix([[1, f(4 + 5*I)],
                          [2 + 3*I, 6]])


# 定义测试函数 test_P11，未完成
@XFAIL
def test_P11():
    # 待补充内容，未完成的测试函数
    pass
    # 抛出未实现错误，说明矩阵 Matrix([[x,y],[1,x*y]]) 的逆矩阵计算没有简化以提取公因子
    # 这行代码包含了一个断言，用来验证计算结果是否符合预期
    assert Matrix([[x, y],
                   [1, x*y]]).inv() == (1/(x**2 - 1))*Matrix([[x, -1],
                                                              [-1/y, x/y]])
def test_P11_workaround():
    # 使用反转 ADJ 方法替代 'GE' 方法，因为 'GE' 方法返回的逆矩阵形式已经改变。
    M = Matrix([[x, y], [1, x*y]]).inv('ADJ')
    # 计算矩阵 M 的元组形式的最大公约数
    c = gcd(tuple(M))
    # 断言检查矩阵乘法的结果是否等于预期的矩阵
    assert MatMul(c, M/c, evaluate=False) == MatMul(c, Matrix([
        [x*y, -y],
        [ -1,  x]]), evaluate=False)


def test_P12():
    # 创建矩阵符号 A11, A12, A22
    A11 = MatrixSymbol('A11', n, n)
    A12 = MatrixSymbol('A12', n, n)
    A22 = MatrixSymbol('A22', n, n)
    # 创建块矩阵 B
    B = BlockMatrix([[A11, A12],
                     [ZeroMatrix(n, n), A22]])
    # 断言检查块矩阵 B 的逆是否等于预期的块矩阵
    assert block_collapse(B.I) == BlockMatrix([[A11.I, (-1)*A11.I*A12*A22.I],
                                               [ZeroMatrix(n, n), A22.I]])


def test_P13():
    # 创建矩阵 M
    M = Matrix([[1,     x - 2,                         x - 3],
                [x - 1, x**2 - 3*x + 6,       x**2 - 3*x - 2],
                [x - 2, x**2 - 8,       2*(x**2) - 12*x + 14]])
    # 进行 LU 分解，获取下三角矩阵 L，上三角矩阵 U
    L, U, _ = M.LUdecomposition()
    # 断言检查化简后的 L 是否等于预期的下三角矩阵
    assert simplify(L) == Matrix([[1,     0,     0],
                                  [x - 1, 1,     0],
                                  [x - 2, x - 3, 1]])
    # 断言检查化简后的 U 是否等于预期的上三角矩阵
    assert simplify(U) == Matrix([[1, x - 2, x - 3],
                                  [0,     4, x - 5],
                                  [0,     0, x - 7]])


def test_P14():
    # 创建矩阵 M
    M = Matrix([[1, 2, 3, 1, 3],
                [3, 2, 1, 1, 7],
                [0, 2, 4, 1, 1],
                [1, 1, 1, 1, 4]])
    # 进行行简化阶梯形式化，获取简化阶梯形矩阵 R
    R, _ = M.rref()
    # 断言检查简化阶梯形矩阵 R 是否等于预期的矩阵
    assert R == Matrix([[1, 0, -1, 0,  2],
                        [0, 1,  2, 0, -1],
                        [0, 0,  0, 1,  3],
                        [0, 0,  0, 0,  0]])


def test_P15():
    # 创建矩阵 M
    M = Matrix([[-1, 3,  7, -5],
                [4, -2,  1,  3],
                [2,  4, 15, -7]])
    # 断言检查矩阵 M 的秩是否等于预期值 2
    assert M.rank() == 2


def test_P16():
    # 创建矩阵 M
    M = Matrix([[2*sqrt(2), 8],
                [6*sqrt(6), 24*sqrt(3)]])
    # 断言检查矩阵 M 的秩是否等于预期值 1
    assert M.rank() == 1


def test_P17():
    t = symbols('t', real=True)
    # 创建矩阵 M
    M=Matrix([
        [sin(2*t), cos(2*t)],
        [2*(1 - (cos(t)**2))*cos(t), (1 - 2*(sin(t)**2))*sin(t)]])
    # 断言检查矩阵 M 的秩是否等于预期值 1
    assert M.rank() == 1


def test_P18():
    # 创建矩阵 M
    M = Matrix([[1,  0, -2, 0],
                [-2, 1,  0, 3],
                [-1, 2, -6, 6]])
    # 断言检查矩阵 M 的零空间是否等于预期的向量列表
    assert M.nullspace() == [Matrix([[2],
                                     [4],
                                     [1],
                                     [0]]),
                             Matrix([[0],
                                     [-3],
                                     [0],
                                     [1]])]


def test_P19():
    w = symbols('w')
    # 创建矩阵 M
    M = Matrix([[1,    1,    1,    1],
                [w,    x,    y,    z],
                [w**2, x**2, y**2, z**2],
                [w**3, x**3, y**3, z**3]])
    # 断言语句，验证行列式 M 的值是否等于给定的复杂表达式
    assert M.det() == (w**3*x**2*y   - w**3*x**2*z - w**3*x*y**2 + w**3*x*z**2
                       + w**3*y**2*z - w**3*y*z**2 - w**2*x**3*y + w**2*x**3*z
                       + w**2*x*y**3 - w**2*x*z**3 - w**2*y**3*z + w**2*y*z**3
                       + w*x**3*y**2 - w*x**3*z**2 - w*x**2*y**3 + w*x**2*z**3
                       + w*y**3*z**2 - w*y**2*z**3 - x**3*y**2*z + x**3*y*z**2
                       + x**2*y**3*z - x**2*y*z**3 - x*y**3*z**2 + x*y**2*z**3
                       )
@XFAIL
def test_P20():
    # 标记该测试为预期失败（Expected Failure），抛出未实现的错误信息
    raise NotImplementedError("Matrix minimal polynomial not supported")


def test_P21():
    # 创建一个 3x3 的整数矩阵 M
    M = Matrix([[5, -3, -7],
                [-2, 1,  2],
                [2, -3, -4]])
    # 断言矩阵 M 的特征多项式表达式等于给定的多项式 x**3 - 2*x**2 - 5*x + 6
    assert M.charpoly(x).as_expr() == x**3 - 2*x**2 - 5*x + 6


def test_P22():
    # 定义维度变量 d 并创建一个对角矩阵 M
    d = 100
    M = (2 - x)*eye(d)
    # 断言矩阵 M 的特征值是一个字典，包含一个特征值 -x + 2，出现 d 次
    assert M.eigenvals() == {-x + 2: d}


def test_P23():
    # 创建一个 5x5 的整数矩阵 M
    M = Matrix([
        [2, 1, 0, 0, 0],
        [1, 2, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 2, 1],
        [0, 0, 0, 1, 2]])
    # 断言矩阵 M 的特征值是一个字典，包含特征值 1, 2, 3, sqrt(3) + 2, -sqrt(3) + 2 各出现一次
    assert M.eigenvals() == {
        S('1'): 1,
        S('2'): 1,
        S('3'): 1,
        S('sqrt(3) + 2'): 1,
        S('-sqrt(3) + 2'): 1}


def test_P24():
    # 创建一个 8x8 的整数矩阵 M
    M = Matrix([[611,  196, -192,  407,   -8,  -52,  -49,   29],
                [196,  899,  113, -192,  -71,  -43,   -8,  -44],
                [-192,  113,  899,  196,   61,   49,    8,   52],
                [ 407, -192,  196,  611,    8,   44,   59,  -23],
                [  -8,  -71,   61,    8,  411, -599,  208,  208],
                [ -52,  -43,   49,   44, -599,  411,  208,  208],
                [ -49,   -8,    8,   59,  208,  208,   99, -911],
                [  29,  -44,   52,  -23,  208,  208, -911,   99]])
    # 断言矩阵 M 的特征值是一个字典，包含多个特征值及其出现次数
    assert M.eigenvals() == {
        S('0'): 1,
        S('10*sqrt(10405)'): 1,
        S('100*sqrt(26) + 510'): 1,
        S('1000'): 2,
        S('-100*sqrt(26) + 510'): 1,
        S('-10*sqrt(10405)'): 1,
        S('1020'): 1}


def test_P25():
    # 创建一个 8x8 的实数矩阵 MF，将其数值化
    MF = N(Matrix([[ 611,  196, -192,  407,   -8,  -52,  -49,   29],
                   [ 196,  899,  113, -192,  -71,  -43,   -8,  -44],
                   [-192,  113,  899,  196,   61,   49,    8,   52],
                   [ 407, -192,  196,  611,    8,   44,   59,  -23],
                   [  -8,  -71,   61,    8,  411, -599,  208,  208],
                   [ -52,  -43,   49,   44, -599,  411,  208,  208],
                   [ -49,   -8,    8,   59,  208,  208,   99, -911],
                   [  29,  -44,   52,  -23,  208,  208, -911,   99]]))
    # 计算 MF 的特征值并排序，与预期的特征值列表进行逐一比较
    ev_1 = sorted(MF.eigenvals(multiple=True))
    ev_2 = sorted(
        [-1020.0490184299969, 0.0, 0.09804864072151699, 1000.0, 1000.0,
        1019.9019513592784, 1020.0, 1020.0490184299969])

    for x, y in zip(ev_1, ev_2):
        # 断言两个特征值列表中的每对元素在给定的误差范围内相等
        assert abs(x - y) < 1e-12


def test_P26():
    # 定义符号变量 a0, a1, a2, a3, a4，并创建一个 9x9 的符号矩阵 M
    a0, a1, a2, a3, a4 = symbols('a0 a1 a2 a3 a4')
    M = Matrix([[-a4, -a3, -a2, -a1, -a0,  0,  0,  0,  0],
                [  1,   0,   0,   0,   0,  0,  0,  0,  0],
                [  0,   1,   0,   0,   0,  0,  0,  0,  0],
                [  0,   0,   1,   0,   0,  0,  0,  0,  0],
                [  0,   0,   0,   1,   0,  0,  0,  0,  0],
                [  0,   0,   0,   0,   0, -1, -1,  0,  0],
                [  0,   0,   0,   0,   0,  1,  0,  0,  0],
                [  0,   0,   0,   0,   0,  0,  1, -1, -1],
                [  0,   0,   0,   0,   0,  0,  0,  1,  0]])
    # 断言：验证矩阵 M 的特征值，当遇到不完整时不报错，应该等于以下字典
    assert M.eigenvals(error_when_incomplete=False) == {
        S('-1/2 - sqrt(3)*I/2'): 2,  # 键为 '-1/2 - sqrt(3)*I/2'，值为 2
        S('-1/2 + sqrt(3)*I/2'): 2   # 键为 '-1/2 + sqrt(3)*I/2'，值为 2
    }
def test_P27():
    # 定义符号变量 a
    a = symbols('a')
    # 创建一个 5x5 的符号矩阵 M
    M = Matrix([[a,  0, 0, 0, 0],
                [0,  0, 0, 0, 1],
                [0,  0, a, 0, 0],
                [0,  0, 0, a, 0],
                [0, -2, 0, 0, 2]])

    # 使用 SymPy 提供的 eigenvects() 方法计算矩阵 M 的特征向量
    assert M.eigenvects() == [
        # 特征值 a 的特征向量
        (a, 3, [
            Matrix([1, 0, 0, 0, 0]),
            Matrix([0, 0, 1, 0, 0]),
            Matrix([0, 0, 0, 1, 0])
        ]),
        # 特征值 1 - I 的特征向量
        (1 - I, 1, [
            Matrix([0, (1 + I)/2, 0, 0, 1])
        ]),
        # 特征值 1 + I 的特征向量
        (1 + I, 1, [
            Matrix([0, (1 - I)/2, 0, 0, 1])
        ]),
    ]


@XFAIL
def test_P28():
    # 抛出未实现错误，通用特征向量不支持
    raise NotImplementedError("Generalized eigenvectors not supported \
https://github.com/sympy/sympy/issues/5293")


@XFAIL
def test_P29():
    # 抛出未实现错误，通用特征向量不支持
    raise NotImplementedError("Generalized eigenvectors not supported \
https://github.com/sympy/sympy/issues/5293")


def test_P30():
    # 创建一个 5x5 的整数矩阵 M
    M = Matrix([[1,  0,  0,  1, -1],
                [0,  1, -2,  3, -3],
                [0,  0, -1,  2, -2],
                [1, -1,  1,  0,  1],
                [1, -1,  1, -1,  2]])
    # 使用 jordan_form() 方法计算其约当形式
    _, J = M.jordan_form()
    # 断言计算出的约当形式 J 是否与预期相等
    assert J == Matrix([[-1, 0, 0, 0, 0],
                        [0,  1, 1, 0, 0],
                        [0,  0, 1, 0, 0],
                        [0,  0, 0, 1, 1],
                        [0,  0, 0, 0, 1]])


@XFAIL
def test_P31():
    # 抛出未实现错误，未实现 Smith 正常形式
    raise NotImplementedError("Smith normal form not implemented")


def test_P32():
    # 创建一个 2x2 的矩阵 M
    M = Matrix([[1, -2],
                [2, 1]])
    # 断言 M 的指数函数重写为余弦形式并简化后是否与预期相等
    assert exp(M).rewrite(cos).simplify() == Matrix([[E*cos(2), -E*sin(2)],
                                                     [E*sin(2),  E*cos(2)]])


def test_P33():
    # 定义符号变量 w 和 t
    w, t = symbols('w t')
    # 创建一个 4x4 的符号矩阵 M
    M = Matrix([[0,    1,      0,   0],
                [0,    0,      0, 2*w],
                [0,    0,      0,   1],
                [0, -2*w, 3*w**2,   0]])
    # 断言 M*t 的指数函数重写为余弦形式并展开后是否与预期相等
    assert exp(M*t).rewrite(cos).expand() == Matrix([
        [1, -3*t + 4*sin(t*w)/w,  6*t*w - 6*sin(t*w), -2*cos(t*w)/w + 2/w],
        [0,      4*cos(t*w) - 3, -6*w*cos(t*w) + 6*w,          2*sin(t*w)],
        [0,  2*cos(t*w)/w - 2/w,     -3*cos(t*w) + 4,          sin(t*w)/w],
        [0,         -2*sin(t*w),        3*w*sin(t*w),            cos(t*w)]])


@XFAIL
def test_P34():
    # 定义实数符号变量 a, b, c
    a, b, c = symbols('a b c', real=True)
    # 创建一个 6x6 的符号矩阵 M
    M = Matrix([[a, 1, 0, 0, 0, 0],
                [0, a, 0, 0, 0, 0],
                [0, 0, b, 0, 0, 0],
                [0, 0, 0, c, 1, 0],
                [0, 0, 0, 0, c, 1],
                [0, 0, 0, 0, 0, c]])
    # 抛出未实现错误，sin(M) 和 exp(M*I) 不支持
    # https://github.com/sympy/sympy/issues/6218
    assert sin(M) == Matrix([[sin(a), cos(a), 0, 0, 0, 0],
                             [0, sin(a), 0, 0, 0, 0],
                             [0, 0, sin(b), 0, 0, 0],
                             [0, 0, 0, sin(c), cos(c), -sin(c)/2],
                             [0, 0, 0, 0, sin(c), cos(c)],
                             [0, 0, 0, 0, 0, sin(c)]])


@XFAIL
def test_P35():
    # 空的测试函数，未实现
    pass
    # 创建一个3x3的矩阵M，元素为π/2乘以给定的矩阵
    M = pi/2 * Matrix([[2, 1, 1],
                       [2, 3, 2],
                       [1, 1, 2]])
    
    # 断言sin(M)应该等于3阶单位矩阵eye(3)，但是由于sin(M)操作不受支持，会引发异常
    # 相关问题可以查看GitHub上的讨论链接：https://github.com/sympy/sympy/issues/6218
    assert sin(M) == eye(3)
@XFAIL
def test_P36():
    # 创建一个2x2的矩阵对象M
    M = Matrix([[10, 7],
                [7, 17]])
    # 断言平方根运算后的结果与指定的2x2矩阵相等
    assert sqrt(M) == Matrix([[3, 1],
                              [1, 4]])


def test_P37():
    # 创建一个3x3的矩阵对象M
    M = Matrix([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])
    # 断言矩阵M的1/2次幂结果与指定的3x3矩阵相等
    assert M**S.Half == Matrix([[1, R(1, 2), 0],
                                [0, 1,       0],
                                [0, 0,       1]])


@XFAIL
def test_P38():
    # 创建一个3x3的矩阵对象M
    M=Matrix([[0, 1, 0],
              [0, 0, 0],
              [0, 0, 0]])

    with raises(AssertionError):
        # 尝试计算M的1/2次幂，期望引发值错误(ValueError)，因为M的行列式为0，不可逆
        M**S.Half
        # 如果未引发错误，则会触发此断言，标记测试为非预期失败状态
        assert None

@XFAIL
def test_P39():
    """
    M=Matrix([
        [1, 1],
        [2, 2],
        [3, 3]])
    M.SVD()
    """
    # 抛出未实现错误，表示奇异值分解尚未实现
    raise NotImplementedError("Singular value decomposition not implemented")


def test_P40():
    # 定义实数符号变量r和t
    r, t = symbols('r t', real=True)
    # 创建一个2x1的矩阵对象M
    M = Matrix([r*cos(t), r*sin(t)])
    # 断言M对r和t的雅可比矩阵与指定的2x2矩阵相等
    assert M.jacobian(Matrix([r, t])) == Matrix([[cos(t), -r*sin(t)],
                                                 [sin(t),  r*cos(t)]])


def test_P41():
    # 定义实数符号变量r和t
    r, t = symbols('r t', real=True)
    # 断言函数r**2*sin(t)对(r, t)的黑塞矩阵与指定的2x2矩阵相等
    assert hessian(r**2*sin(t),(r,t)) == Matrix([[  2*sin(t),   2*r*cos(t)],
                                                 [2*r*cos(t), -r**2*sin(t)]])


def test_P42():
    # 断言(cos(x), sin(x))的朗斯基行列式在x处化简后等于1
    assert wronskian([cos(x), sin(x)], x).simplify() == 1


def test_P43():
    # 定义私有函数__my_jacobian，计算矩阵M对变量Y的雅可比矩阵
    def __my_jacobian(M, Y):
        return Matrix([M.diff(v).T for v in Y]).T
    # 定义实数符号变量r和t
    r, t = symbols('r t', real=True)
    # 创建一个2x1的矩阵对象M
    M = Matrix([r*cos(t), r*sin(t)])
    # 断言私有函数计算的雅可比矩阵与指定的2x2矩阵相等
    assert __my_jacobian(M,[r,t]) == Matrix([[cos(t), -r*sin(t)],
                                             [sin(t),  r*cos(t)]])


def test_P44():
    # 定义私有函数__my_hessian，计算函数f对变量Y的黑塞矩阵
    def __my_hessian(f, Y):
        V = Matrix([diff(f, v) for v in Y])
        return  Matrix([V.T.diff(v) for v in Y])
    # 定义实数符号变量r和t
    r, t = symbols('r t', real=True)
    # 断言私有函数计算的黑塞矩阵与指定的2x2矩阵相等
    assert __my_hessian(r**2*sin(t), (r, t)) == Matrix([
                                            [  2*sin(t),   2*r*cos(t)],
                                            [2*r*cos(t), -r**2*sin(t)]])


def test_P45():
    # 定义私有函数__my_wronskian，计算向量Y的朗斯基行列式在变量v处的值
    def __my_wronskian(Y, v):
        M = Matrix([Matrix(Y).T.diff(x, n) for n in range(0, len(Y))])
        return  M.det()
    # 断言私有函数计算的朗斯基行列式在(cos(x), sin(x))处化简后等于1
    assert __my_wronskian([cos(x), sin(x)], x).simplify() == 1

# Q1-Q6  Tensor tests missing


@XFAIL
def test_R1():
    # 定义整数符号变量i, j, n，且n为正数
    i, j, n = symbols('i j n', integer=True, positive=True)
    # 创建一个n行1列的矩阵符号变量xn
    xn = MatrixSymbol('xn', n, 1)
    # 构建求和表达式Sm，但尚未计算
    Sm = Sum((xn[i, 0] - Sum(xn[j, 0], (j, 0, n - 1))/n)**2, (i, 0, n - 1))
    # 尝试计算Sm的值，但结果未知
    Sm.doit()
    # 抛出未实现错误，表示结果未知
    raise NotImplementedError('Unknown result')

@XFAIL
def test_R2():
    # 定义符号变量m和b
    m, b = symbols('m b')
    # 定义整数符号变量i和n，且n为正数
    i, n = symbols('i n', integer=True, positive=True)
    # 创建n行1列的矩阵符号变量xn和yn
    xn = MatrixSymbol('xn', n, 1)
    yn = MatrixSymbol('yn', n, 1)
    # 构建函数f的表达式，但未求导
    f = Sum((yn[i, 0] - m*xn[i, 0] - b)**2, (i, 0, n - 1))
    # 对函数f分别关于m和b求偏导数f1和f2
    f1 = diff(f, m)
    f2 = diff(f, b)
    # 调用 solveset 函数来求解方程组 (f1, f2)，其中 (m, b) 是变量，domain=S.Reals 指定定义域为实数集
    solveset((f1, f2), (m, b), domain=S.Reals)
@XFAIL
# 定义测试函数 test_R3，用于测试符号数学表达式
def test_R3():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 计算序列 sk 中的表达式
    sk = ((-1)**k) * (binomial(2*n, k))**2
    # 对求和表达式 Sm 进行求值
    Sm = Sum(sk, (k, 1, oo))
    # 对 Sm 求值得到的结果进行简化
    T = Sm.doit()
    # 对简化后的结果再次进行组合简化
    T2 = T.combsimp()
    # 断言 T2 的结果符合预期的数学表达式
    # 返回的表达式应为 (-1)**n*binomial(2*n, n)
    assert T2 == (-1)**n*binomial(2*n, n)


@XFAIL
# 定义测试函数 test_R4，用于测试 Macsyma 不定积分的情况
def test_R4():
    # 抛出未实现错误，表示不支持不定积分的测试
    raise NotImplementedError("Indefinite sum not supported")


@XFAIL
# 定义测试函数 test_R5，用于测试复杂的符号数学表达式
def test_R5():
    # 定义符号变量 a, b, c, n, k，均为正整数
    a, b, c, n, k = symbols('a b c n k', integer=True, positive=True)
    # 计算序列 sk 中的表达式
    sk = ((-1)**k)*(binomial(a + b, a + k)
                    *binomial(b + c, b + k)*binomial(c + a, c + k))
    # 对求和表达式 Sm 进行求值
    Sm = Sum(sk, (k, 1, oo))
    # 对 Sm 求值得到的结果进行断言
    # 结果应为阶乘表达式的比值
    assert Sm.doit() == factorial(a+b+c)/(factorial(a)*factorial(b)*factorial(c))


# 定义测试函数 test_R6，用于测试矩阵符号的求和表达式
def test_R6():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 定义矩阵符号 gn，具有 n+2 行 1 列
    gn = MatrixSymbol('gn', n + 2, 1)
    # 对求和表达式 Sm 进行求值
    Sm = Sum(gn[k, 0] - gn[k - 1, 0], (k, 1, n + 1))
    # 断言 Sm 求值后的结果符合预期
    assert Sm.doit() == -gn[0, 0] + gn[n + 1, 0]


# 定义测试函数 test_R7，用于测试立方和的求和表达式
def test_R7():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 对求和表达式 T 进行求值
    T = Sum(k**3,(k,1,n)).doit()
    # 断言 T 的因式分解结果符合预期
    assert T.factor() == n**2*(n + 1)**2/4


@XFAIL
# 定义测试函数 test_R8，用于测试带有二项式系数的求和表达式
def test_R8():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 对求和表达式 Sm 进行求值
    Sm = Sum(k**2*binomial(n, k), (k, 1, n))
    # 对 Sm 求值后的结果进行组合简化
    T = Sm.doit()
    # 断言 T 的组合简化结果符合预期
    assert T.combsimp() == n*(n + 1)*2**(n - 2)


# 定义测试函数 test_R9，用于测试二项式系数的求和表达式
def test_R9():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 对求和表达式 Sm 进行求值
    Sm = Sum(binomial(n, k - 1)/k, (k, 1, n + 1))
    # 断言 Sm 求值后的结果进行简化后符合预期
    assert Sm.doit().simplify() == (2**(n + 1) - 1)/(n + 1)


@XFAIL
# 定义测试函数 test_R10，用于测试两个二项式系数的乘积的求和表达式
def test_R10():
    # 定义符号变量 n, m, r, k，均为正整数
    n, m, r, k = symbols('n m r k', integer=True, positive=True)
    # 对求和表达式 Sm 进行求值
    Sm = Sum(binomial(n, k)*binomial(m, r - k), (k, 0, r))
    # 对 Sm 求值后的结果进行组合简化并重写为阶乘形式
    T = Sm.doit()
    T2 = T.combsimp().rewrite(factorial)
    # 断言 T2 符合预期的阶乘形式
    assert T2 == factorial(m + n)/(factorial(r)*factorial(m + n - r))
    # 断言 T2 与二项式重写后的结果相等
    assert T2 == binomial(m + n, r).rewrite(factorial)
    # 对 T2 进行二项式重写，断言结果等于二项式形式
    T3 = T2.rewrite(binomial)
    assert T3 == binomial(m + n, r)


@XFAIL
# 定义测试函数 test_R11，用于测试斐波那契数乘二项式系数的求和表达式
def test_R11():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 计算序列 sk 中的表达式
    sk = binomial(n, k)*fibonacci(k)
    # 对求和表达式 Sm 进行求值
    Sm = Sum(sk, (k, 0, n))
    # 对 Sm 求值后的结果进行断言
    # 斐波那契简化未实现
    # 断言：验证 T 等于 fibonacci(2*n)，用于检查条件是否成立
    assert T == fibonacci(2*n)
@XFAIL
def test_R12():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 构造斐波那契数列的平方和表达式
    Sm = Sum(fibonacci(k)**2, (k, 0, n))
    # 计算和式
    T = Sm.doit()
    # 断言计算结果是否等于斐波那契数列的乘积形式
    assert T == fibonacci(n)*fibonacci(n + 1)


@XFAIL
def test_R13():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 构造正弦函数的和式
    Sm = Sum(sin(k*x), (k, 1, n))
    # 不计算和式
    T = Sm.doit()  # Sum is not calculated
    # 断言简化后的结果是否等于给定表达式
    assert T.simplify() == cot(x/2)/2 - cos(x*(2*n + 1)/2)/(2*sin(x/2))


@XFAIL
def test_R14():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 构造正弦函数乘积的和式
    Sm = Sum(sin((2*k - 1)*x), (k, 1, n))
    # 不计算和式
    T = Sm.doit()  # Sum is not calculated
    # 断言简化后的结果是否等于给定表达式
    assert T.simplify() == sin(n*x)**2/sin(x)


@XFAIL
def test_R15():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 构造二项式系数的和式
    Sm = Sum(binomial(n - k, k), (k, 0, floor(n/2)))
    # 不计算和式
    T = Sm.doit()  # Sum is not calculated
    # 断言简化后的结果是否等于给定表达式
    assert T.simplify() == fibonacci(n + 1)


def test_R16():
    # 定义符号变量 k，为正整数
    k = symbols('k', integer=True, positive=True)
    # 构造级数求和
    Sm = Sum(1/k**2 + 1/k**3, (k, 1, oo))
    # 断言级数求和的结果是否等于黎曼 zeta 函数值加上 pi 的平方除以 6
    assert Sm.doit() == zeta(3) + pi**2/6


def test_R17():
    # 定义符号变量 k，为正整数
    k = symbols('k', integer=True, positive=True)
    # 计算级数求和并转换为浮点数，断言其与给定值的绝对误差小于 1e-15
    assert abs(float(Sum(1/k**2 + 1/k**3, (k, 1, oo)))
               - 2.8469909700078206) < 1e-15


def test_R18():
    # 定义符号变量 k，为正整数
    k = symbols('k', integer=True, positive=True)
    # 构造级数求和
    Sm = Sum(1/(2**k*k**2), (k, 1, oo))
    # 计算级数求和并简化结果
    T = Sm.doit()
    # 断言简化后的结果是否等于给定表达式
    assert T.simplify() == -log(2)**2/2 + pi**2/12


@slow
@XFAIL
def test_R19():
    # 定义符号变量 k，为正整数
    k = symbols('k', integer=True, positive=True)
    # 构造级数求和
    Sm = Sum(1/((3*k + 1)*(3*k + 2)*(3*k + 3)), (k, 0, oo))
    # 不计算和式
    T = Sm.doit()
    # 断言简化后的结果是否等于给定表达式
    assert T.simplify() == -log(3)/4 + sqrt(3)*pi/12


@XFAIL
def test_R20():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 构造二项式系数的无穷和式
    Sm = Sum(binomial(n, 4*k), (k, 0, oo))
    # 不计算和式
    T = Sm.doit()
    # 断言简化后的结果是否等于给定表达式
    assert T.simplify() == 2**(n/2)*cos(pi*n/4)/2 + 2**(n - 1)/2


@XFAIL
def test_R21():
    # 定义符号变量 k，为正整数
    k = symbols('k', integer=True, positive=True)
    # 构造级数求和
    Sm = Sum(1/(sqrt(k*(k + 1)) * (sqrt(k) + sqrt(k + 1))), (k, 1, oo))
    # 不计算和式
    T = Sm.doit()  # Sum not calculated
    # 断言简化后的结果是否等于给定表达式
    assert T.simplify() == 1


# test_R22 answer not available in Wester samples
# Sum(Sum(binomial(n, k)*binomial(n - k, n - 2*k)*x**n*y**(n - 2*k),
#                 (k, 0, floor(n/2))), (n, 0, oo)) with abs(x*y)<1?


@XFAIL
def test_R23():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 构造双重级数求和
    Sm = Sum(Sum((factorial(n)/(factorial(k)**2*factorial(n - 2*k)))*
                 (x/y)**k*(x*y)**(n - k), (n, 2*k, oo)), (k, 0, oo))
    # 不计算和式
    T = Sm.doit()  # Sum not calculated
    # 断言结果是否等于给定表达式
    assert T == -1/sqrt(x**2*y**2 - 4*x**2 - 2*x*y + 1)


def test_R24():
    # 定义符号变量 m 和 k，均为正整数
    m, k = symbols('m k', integer=True, positive=True)
    # 构造双重乘积求和
    Sm = Sum(Product(k/(2*k - 1), (k, 1, m)), (m, 2, oo))
    # 断言级数求和的结果是否等于 pi 的一半
    assert Sm.doit() == pi/2


def test_S1():
    # 定义符号变量 k，为正整数
    k = symbols('k', integer=True, positive=True)
    # 构造乘积求和
    Pr = Product(gamma(k/3), (k, 1, 8))
    # 计算乘积求和并简化结果
    assert Pr.doit().simplify() == 640*sqrt(3)*pi**3/6561


def test_S2():
    # 导入符号 n 和 k，并指定它们为整数和正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 断言：对于给定的 n，从 1 到 n 的连乘结果等于 n 的阶乘
    assert Product(k, (k, 1, n)).doit() == factorial(n)
def test_S3():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 断言：计算 x**k 的乘积，从 k=1 到 k=n，并简化结果，判断是否等于 x**(n*(n + 1)/2)
    assert Product(x**k, (k, 1, n)).doit().simplify() == x**(n*(n + 1)/2)


def test_S4():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 断言：计算 1 + 1/k 的乘积，从 k=1 到 k=n-1，并简化结果，判断是否等于 n
    assert Product(1 + 1/k, (k, 1, n - 1)).doit().simplify() == n


def test_S5():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 断言：计算 (2*k - 1)/(2*k) 的乘积，从 k=1 到 k=n，应用 gammasimp 函数简化结果，
    # 判断是否等于 gamma(n + S.Half)/(sqrt(pi)*gamma(n + 1))
    assert (Product((2*k - 1)/(2*k), (k, 1, n)).doit().gammasimp() ==
            gamma(n + S.Half)/(sqrt(pi)*gamma(n + 1)))


@XFAIL
def test_S6():
    # 定义符号变量 n 和 k，均为正整数
    n, k = symbols('n k', integer=True, positive=True)
    # 断言：计算 x**2 - 2*x*cos(k*pi/n) + 1 的乘积，从 k=1 到 k=n-1，并简化结果，
    # 判断是否等于 (x**(2*n) - 1)/(x**2 - 1)
    # 注意：Product 不会被评估
    assert (Product(x**2 - 2*x*cos(k*pi/n) + 1, (k, 1, n - 1)).doit().simplify()
            == (x**(2*n) - 1)/(x**2 - 1))


@XFAIL
def test_S7():
    # 定义符号变量 k，为正整数
    k = symbols('k', integer=True, positive=True)
    # 创建 Product 对象 Pr，计算 (k**3 - 1)/(k**3 + 1) 的乘积，从 k=2 到无穷大
    Pr = Product((k**3 - 1)/(k**3 + 1), (k, 2, oo))
    # 计算 Pr 的值，Product 不会被评估
    T = Pr.doit()
    # 断言：简化结果 T 是否等于 R(2, 3)
    assert T.simplify() == R(2, 3)


@XFAIL
def test_S8():
    # 定义符号变量 k，为正整数
    k = symbols('k', integer=True, positive=True)
    # 创建 Product 对象 Pr，计算 1 - 1/(2*k)**2 的乘积，从 k=1 到无穷大
    Pr = Product(1 - 1/(2*k)**2, (k, 1, oo))
    # 计算 Pr 的值
    T = Pr.doit()
    # 断言：简化结果 T 是否等于 2/pi
    # 注意：Product 不会被评估
    assert T.simplify() == 2/pi


@XFAIL
def test_S9():
    # 定义符号变量 k，为正整数
    k = symbols('k', integer=True, positive=True)
    # 创建 Product 对象 Pr，计算 (1 + (-1)**(k + 1)/(2*k - 1)) 的乘积，从 k=1 到无穷大
    Pr = Product(1 + (-1)**(k + 1)/(2*k - 1), (k, 1, oo))
    # 计算 Pr 的值
    T = Pr.doit()
    # 断言：简化结果 T 是否等于 sqrt(2)
    # Product 会得出 0
    # 参考链接：https://github.com/sympy/sympy/issues/7133
    assert T.simplify() == sqrt(2)


@XFAIL
def test_S10():
    # 定义符号变量 k，为正整数
    k = symbols('k', integer=True, positive=True)
    # 创建 Product 对象 Pr，计算 ((k*(k + 1) + 1 + I)/(k*(k + 1) + 1 - I)) 的乘积，从 k=0 到无穷大
    Pr = Product((k*(k + 1) + 1 + I)/(k*(k + 1) + 1 - I), (k, 0, oo))
    # 计算 Pr 的值
    T = Pr.doit()
    # 断言：简化结果 T 是否等于 -1
    # 注意：Product 不会被评估
    assert T.simplify() == -1
def test_T12():
    x, t = symbols('x t', real=True)
    # 定义符号变量 x 和 t，限定为实数
    # 对 limit() 函数进行断言，计算极限但不求值，返回包含 erf 的表达式
    assert limit(x * integrate(exp(-t**2), (t, 0, x))/(1 - exp(-x**2)),
                 x, 0) == 1


def test_T13():
    x = symbols('x', real=True)
    # 定义符号变量 x，限定为实数
    # 对 limit() 函数进行断言，计算左右极限并分别断言其值为 -1 和 1
    assert [limit(x/abs(x), x, 0, dir='-'),
            limit(x/abs(x), x, 0, dir='+')] == [-1, 1]


def test_T14():
    x = symbols('x', real=True)
    # 定义符号变量 x，限定为实数
    # 对 limit() 函数进行断言，计算 x 趋于 0 时 atan(-log(x)) 的极限，断言其值为 pi/2
    assert limit(atan(-log(x)), x, 0, dir='+') == pi/2


def test_U1():
    x = symbols('x', real=True)
    # 定义符号变量 x，限定为实数
    # 对 diff() 函数进行断言，计算绝对值函数 abs(x) 对 x 的导数，断言其值为符号函数 sign(x)
    assert diff(abs(x), x) == sign(x)


def test_U2():
    f = Lambda(x, Piecewise((-x, x < 0), (x, x >= 0)))
    # 定义 Lambda 函数 f(x)，根据 x 的正负返回 -x 或 x
    # 对 diff() 函数进行断言，计算 f(x) 对 x 的导数，断言其在 x < 0 和 x >= 0 时的不同取值
    assert diff(f(x), x) == Piecewise((-1, x < 0), (1, x >= 0))


def test_U3():
    f = Lambda(x, Piecewise((x**2 - 1, x == 1), (x**3, x != 1)))
    # 定义 Lambda 函数 f(x)，根据 x 的取值返回不同表达式
    # 定义 Lambda 函数 f1(x)，表示 f(x) 对 x 的导数
    # 对 f1(x) 进行断言，计算 x^3 对 x 的导数，断言其值为 3*x^2
    assert f1(x) == 3*x**2
    # 对 f1(1) 进行断言，计算 x = 1 时的导数值，断言其值为 3
    assert f1(1) == 3


@XFAIL
def test_U4():
    n = symbols('n', integer=True, positive=True)
    x = symbols('x', real=True)
    # 定义符号变量 n（整数且正数）、x（实数）
    # 对 x^n 对 x 进行 n 次求导
    d = diff(x**n, x, n)
    # 对导数应用 factorial 重写，断言结果等于阶乘 factorial(n)
    assert d.rewrite(factorial) == factorial(n)


def test_U5():
    # issue 6681
    t = symbols('t')
    # 定义符号变量 t
    # 定义 Derivative 对象 ans，表示 f(g(t)).diff(t, 2)
    ans = (
        Derivative(f(g(t)), g(t))*Derivative(g(t), (t, 2)) +
        Derivative(f(g(t)), (g(t), 2))*Derivative(g(t), t)**2)
    # 对 f(g(t)).diff(t, 2) 进行断言，其值等于 ans
    assert f(g(t)).diff(t, 2) == ans
    # 对 ans 进行 doit() 操作，断言结果等于 ans 自身
    assert ans.doit() == ans


def test_U6():
    h = Function('h')
    # 定义符号函数 h
    # 定义积分 T，表示对 f(y) 在 y = h(x) 到 y = g(x) 之间积分
    T = integrate(f(y), (y, h(x), g(x)))
    # 对 T 对 x 求导，断言结果等于给定表达式
    assert T.diff(x) == (
        f(g(x))*Derivative(g(x), x) - f(h(x))*Derivative(h(x), x))


@XFAIL
def test_U7():
    p, t = symbols('p t', real=True)
    # 定义符号变量 p、t，限定为实数
    # 对 f(p, t) 求偏导数，因表达式有多个变量，必须指定 differentiation 变量
    # 该测试预期引发 ValueError
    diff(f(p, t))


def test_U8():
    x, y = symbols('x y', real=True)
    eq = cos(x*y) + x
    # 定义符号变量 x、y，限定为实数
    # 对 idiff() 函数进行断言，计算 y - eq 对 y 在给定条件下的不显式导数
    # 使用 hack 来避免 SymPy 中隐式求导函数的限制
    assert idiff(y - eq, y, x) == (-y*sin(x*y) + 1)/(x*sin(x*y) + 1)


def test_U9():
    # Wester sample case for Maple:
    # O29 := diff(f(x, y), x) + diff(f(x, y), y);
    #                      /d         \   /d         \
    #                      |-- f(x, y)| + |-- f(x, y)|
    #                      \dx        /   \dy        /
    #
    # O30 := factor(subs(f(x, y) = g(x^2 + y^2), %));
    #                                2    2
    #                        2 D(g)(x  + y ) (x + y)
    x, y = symbols('x y', real=True)
    # 定义符号变量 x、y，限定为实数
    # 定义符号表达式 su，表示 f(x, y) 对 x 和 y 的偏导数之和
    su = diff(f(x, y), x) + diff(f(x, y), y)
    # 对 su 应用 f(x, y) 替换为 g(x^2 + y^2)，得到 s2
    s2 = su.subs(f(x, y), g(x**2 + y**2))
    # 对 s2 进行因式分解，得到 s3
    s3 = s2.doit().factor()
    # 执行 Subs 操作未完成，s3 = 2*(x + y)*Subs(Derivative(g(_xi_1), _xi_1), _xi_1, x**2 + y**2)
    # SymPy 中 Derivative(g(x*2 + y**2), x**2 + y**2) 无效，且可能保持不变。你只能对类似符号或函数的原子表达式进行偏导数计算。
    # 断言语句用于验证条件是否为真，否则会抛出 AssertionError 异常。
    # 确保 s3 等于表达式 (x + y) * Subs(Derivative(g(x), x), x, x**2 + y**2) * 2。
    # 这个表达式涉及到 SymPy 的符号运算，具体包括对 g(x) 的导数的替换，以及乘以 (x + y) 和 2。
    assert s3 == (x + y) * Subs(Derivative(g(x), x), x, x**2 + y**2) * 2
def test_U10():
    # 通过检查 issue 2519 进行测试
    assert residue((z**3 + 5)/((z**4 - 1)*(z + 1)), z, -1) == R(-9, 4)

@XFAIL
def test_U11():
    # 断言，但未实现，因为抛出了 NotImplementedError 异常
    raise NotImplementedError

@XFAIL
def test_U12():
    # Wester 样例：
    # (c41) /* d(3 x^5 dy /\ dz + 5 x y^2 dz /\ dx + 8 z dx /\ dy)
    #    => (15 x^4 + 10 x y + 8) dx /\ dy /\ dz */
    # factor(ext_diff(3*x^5 * dy ~ dz + 5*x*y^2 * dz ~ dx + 8*z * dx ~ dy));
    #                      4
    # (d41)              (10 x y + 15 x  + 8) dx dy dz
    raise NotImplementedError(
        "External diff of differential form not supported")

def test_U13():
    # 断言最小值
    assert minimum(x**4 - x + 1, x) == -3*2**R(1,3)/8 + 1

@XFAIL
def test_U14():
    #f = 1/(x**2 + y**2 + 1)
    # 断言，但未实现，因为抛出了 NotImplementedError 异常
    raise NotImplementedError("minimize(), maximize() not supported")

@XFAIL
def test_U15():
    # 断言，但未实现，因为抛出了 NotImplementedError 异常
    raise NotImplementedError("minimize() not supported and also solve does not support multivariate inequalities")

@XFAIL
def test_U16():
    # 断言，但未实现，因为抛出了 NotImplementedError 异常
    raise NotImplementedError("minimize() not supported in SymPy and also solve does not support multivariate inequalities")

@XFAIL
def test_U17():
    # 断言，但未实现，因为抛出了 NotImplementedError 异常
    raise NotImplementedError("Linear programming, symbolic simplex not supported in SymPy")

def test_V1():
    # 定义符号 x 为实数
    x = symbols('x', real=True)
    # 断言积分结果
    assert integrate(abs(x), x) == Piecewise((-x**2/2, x <= 0), (x**2/2, True))

def test_V2():
    # 断言积分结果
    assert integrate(Piecewise((-x, x < 0), (x, x >= 0)), x
        ) == Piecewise((-x**2/2, x < 0), (x**2/2, True))

def test_V3():
    # 断言积分的导数化简结果
    assert integrate(1/(x**3 + 2),x).diff().simplify() == 1/(x**3 + 2)

def test_V4():
    # 断言积分结果
    assert integrate(2**x/sqrt(1 + 4**x), x) == asinh(2**x)/log(2)

@XFAIL
def test_V5():
    # 返回的表达式
    # (-45*x**2 + 80*x - 41)/(5*sqrt(2*x - 1)*(4*x**2 - 4*x + 1))
    assert (integrate((3*x - 5)**2/(2*x - 1)**R(7, 2), x).simplify() ==
            (-41 + 80*x - 45*x**2)/(5*(2*x - 1)**R(5, 2)))

@XFAIL
def test_V6():
    # 返回的表达式
    # RootSum(40*_z**2 - 1, Lambda(_i, _i*log(-4*_i + exp(-m*x))))/m
    assert (integrate(1/(2*exp(m*x) - 5*exp(-m*x)), x) == sqrt(10)*(
            log(2*exp(m*x) - sqrt(10)) - log(2*exp(m*x) + sqrt(10)))/(20*m))

def test_V7():
    # 计算双曲正弦函数的四次方除以双曲余弦函数的平方的积分，并简化结果后进行断言
    r1 = integrate(sinh(x)**4/cosh(x)**2)
    assert r1.simplify() == x*R(-3, 2) + sinh(x)**3/(2*cosh(x)) + 3*tanh(x)/2

@XFAIL
def test_V8_V9():
    # Macsyma 测试案例：
    # (c27) /* This example involves several symbolic parameters
    #   => 1/sqrt(b^2 - a^2) log([sqrt(b^2 - a^2) tan(x/2) + a + b]/
    #                            [sqrt(b^2 - a^2) tan(x/2) - a - b])   (a^2 < b^2)
    #      [Gradshteyn and Ryzhik 2.553(3)] */
    # assume(b^2 > a^2)$
    # (c28) integrate(1/(a + b*cos(x)), x);
    # (c29) trigsimp(ratsimp(diff(%, x)));
    #                        1
    # (d29)             ------------
    #                  b cos(x) + a
    raise NotImplementedError(
        "Integrate with assumption not supported")

def test_V10():
    # 空函数，未包含测试内容
    pass
    # 断言：验证积分结果是否与给定表达式相等
    assert integrate(1/(3 + 3*cos(x) + 4*sin(x)), x) == log(4*tan(x/2) + 3)/4
def test_V11():
    # 调用 integrate 函数对表达式进行积分
    r1 = integrate(1/(4 + 3*cos(x) + 4*sin(x)), x)
    # 对积分结果进行因式分解
    r2 = factor(r1)
    # 断言对数组合化后的结果是否等于给定表达式的对数形式的三分之一次方
    assert (logcombine(r2, force=True) ==
            log(((tan(x/2) + 1)/(tan(x/2) + 7))**R(1, 3)))


def test_V12():
    # 调用 integrate 函数对表达式进行积分
    r1 = integrate(1/(5 + 3*cos(x) + 4*sin(x)), x)
    # 断言积分结果是否等于给定表达式的负倒切一半加二的形式
    assert r1 == -1/(tan(x/2) + 2)


@XFAIL
def test_V13():
    # 调用 integrate 函数对表达式进行积分
    r1 = integrate(1/(6 + 3*cos(x) + 4*sin(x)), x)
    # 断言经简化后的积分结果是否等于给定的复杂表达式
    # expression not simplified, returns: -sqrt(11)*I*log(tan(x/2) + 4/3
    #   - sqrt(11)*I/3)/11 + sqrt(11)*I*log(tan(x/2) + 4/3 + sqrt(11)*I/3)/11
    assert r1.simplify() == 2*sqrt(11)*atan(sqrt(11)*(3*tan(x/2) + 4)/11)/11


@slow
@XFAIL
def test_V14():
    # 调用 integrate 函数对表达式进行积分
    r1 = integrate(log(abs(x**2 - y**2)), x)
    # 断言对积分结果应用简化后是否等于指定的表达式
    # Piecewise result does not simplify to the desired result.
    assert (r1.simplify() == x*log(abs(x**2  - y**2))
                            + y*log(x + y) - y*log(x - y) - 2*x)


def test_V15():
    # 调用 integrate 函数对表达式进行积分
    r1 = integrate(x*acot(x/y), x)
    # 断言对积分结果应用 simplify 函数后是否等于指定表达式
    assert simplify(r1 - (x*y + (x**2 + y**2)*acot(x/y))/2) == 0


@XFAIL
def test_V16():
    # 断言调用 integrate 函数对给定表达式进行积分是否失败
    # Integral not calculated
    assert integrate(cos(5*x)*Ci(2*x), x) == Ci(2*x)*sin(5*x)/5 - (Si(3*x) + Si(7*x))/10

@XFAIL
def test_V17():
    # 调用 integrate 函数对表达式进行积分
    r1 = integrate((diff(f(x), x)*g(x)
                   - f(x)*diff(g(x), x))/(f(x)**2 - g(x)**2), x)
    # 断言对积分结果应用 simplify 函数后是否等于指定表达式
    # integral not calculated
    assert simplify(r1 - (f(x) - g(x))/(f(x) + g(x))/2) == 0


@XFAIL
def test_W1():
    # 断言调用 integrate 函数对给定表达式进行积分结果是否等于零
    # The function has a pole at y.
    # The integral has a Cauchy principal value of zero but SymPy returns -I*pi
    # https://github.com/sympy/sympy/issues/7159
    assert integrate(1/(x - y), (x, y - 1, y + 1)) == 0


@XFAIL
def test_W2():
    # 断言调用 integrate 函数对给定表达式进行积分结果是否是 zoo
    # The function has a pole at y.
    # The integral is divergent but SymPy returns -2
    # https://github.com/sympy/sympy/issues/7160
    # Test case in Macsyma:
    # (c6) errcatch(integrate(1/(x - a)^2, x, a - 1, a + 1));
    # Integral is divergent
    assert integrate(1/(x - y)**2, (x, y - 1, y + 1)) is zoo


@XFAIL
@slow
def test_W3():
    # 断言调用 integrate 函数对给定表达式进行积分结果是否等于指定的数学表达式
    # integral is not  calculated
    # https://github.com/sympy/sympy/issues/7161
    assert integrate(sqrt(x + 1/x - 2), (x, 0, 1)) == R(4, 3)


@XFAIL
@slow
def test_W4():
    # 断言调用 integrate 函数对给定表达式进行积分结果是否等于指定的数学表达式
    # integral is not  calculated
    assert integrate(sqrt(x + 1/x - 2), (x, 1, 2)) == -2*sqrt(2)/3 + R(4, 3)


@XFAIL
@slow
def test_W5():
    # 断言调用 integrate 函数对给定表达式进行积分结果是否等于指定的数学表达式
    # integral is not  calculated
    assert integrate(sqrt(x + 1/x - 2), (x, 0, 2)) == -2*sqrt(2)/3 + R(8, 3)


@XFAIL
@slow
def test_W6():
    # 断言调用 integrate 函数对给定表达式进行积分结果是否等于指定的数学表达式
    # integral is not  calculated
    assert integrate(sqrt(2 - 2*cos(2*x))/2, (x, pi*R(-3, 4), -pi/4)) == sqrt(2)


def test_W7():
    # 定义符号 a 为正数
    a = symbols('a', positive=True)
    # 调用 integrate 函数对表达式进行积分
    r1 = integrate(cos(x)/(x**2 + a**2), (x, -oo, oo))
    # 断言对积分结果应用 simplify 函数后是否等于指定的数学表达式
    assert r1.simplify() == pi*exp(-a)/a


@XFAIL
def test_W8():
    # 断言调用 integrate 函数对给定表达式进行积分时是否抛出 NotImplementedError
    # Test case in Mathematica:
    # In[19]:= Integrate[t^(a - 1)/(1 + t), {t, 0, Infinity},
    #                    Assumptions -> 0 < a < 1]
    # Out[19]= Pi Csc[a Pi]
    raise NotImplementedError(
        "Integrate with assumption 0 < a < 1 not supported")


@XFAIL
@slow
def test_W9():
    # integral is not  calculated
    pass
    # 计算具有在无穷处的残留的被积函数的值
    # 使用积分技术处理，参考文献 [Levinson and Redheffer, p. 234]
    # 定义第一个积分，对 x 的范围为负无穷到正无穷
    r1 = integrate(5*x**3/(1 + x + x**2 + x**3 + x**4), (x, -oo, oo))
    
    # 对第一个积分的结果进行实际积分计算
    r2 = r1.doit()
    
    # 断言：验证 r2 的值等于 -2π * (sqrt(-sqrt(5)/8 + 5/8) + sqrt(sqrt(5)/8 + 5/8))
    assert r2 == -2*pi*(sqrt(-sqrt(5)/8 + 5/8) + sqrt(sqrt(5)/8 + 5/8))
@XFAIL
# 定义一个测试函数，用于测试一些复杂积分的结果
def test_W10():
    # 计算积分：∫(x / (1 + x + x^2 + x^4)) dx，从 -∞ 到 ∞
    r1 = integrate(x/(1 + x + x**2 + x**4), (x, -oo, oo))
    # 对积分结果进行求值
    r2 = r1.doit()
    # 断言积分结果应该等于给定的数学表达式
    assert r2 == 2*pi*(sqrt(5)/4 + 5/4)*csc(pi*R(2, 5))/5


@XFAIL
# 定义一个测试函数，用于测试未计算的积分结果
def test_W11():
    # 断言某个积分结果等于给定的数学表达式
    assert (integrate(sqrt(1 - x**2)/(1 + x**2), (x, -1, 1)) ==
            pi*(-1 + sqrt(2)))


def test_W12():
    # 定义正数和实数符号
    p = symbols('p', positive=True)
    q = symbols('q', real=True)
    # 计算积分：∫(x * exp(-p * x^2 + 2 * q * x)) dx，从 -∞ 到 ∞
    r1 = integrate(x*exp(-p*x**2 + 2*q*x), (x, -oo, oo))
    # 简化积分结果并断言其等于给定的数学表达式
    assert r1.simplify() == sqrt(pi)*q*exp(q**2/p)/p**R(3, 2)


@XFAIL
# 定义一个测试函数，用于测试未计算的积分结果
def test_W13():
    # 计算积分：∫(1/log(x) + 1/(1 - x) - log(log(1/x))) dx，从 0 到 1
    r1 = integrate(1/log(x) + 1/(1 - x) - log(log(1/x)), (x, 0, 1))
    # 断言积分结果应该等于给定的数学常数 EulerGamma 的两倍
    assert r1 == 2*EulerGamma


def test_W14():
    # 断言积分结果等于给定的数学表达式
    assert integrate(sin(x)/x*exp(2*I*x), (x, -oo, oo)) == 0


@XFAIL
# 定义一个测试函数，用于测试未计算的积分结果
def test_W15():
    # 断言某个积分结果等于给定的数学表达式
    assert integrate(log(gamma(x))*cos(6*pi*x), (x, 0, 1)) == R(1, 12)


def test_W16():
    # 断言积分结果等于给定的数学表达式
    assert integrate((1 + x)**3*legendre_poly(1, x)*legendre_poly(2, x),
                     (x, -1, 1)) == R(36, 35)


def test_W17():
    # 定义正数符号
    a, b = symbols('a b', positive=True)
    # 计算积分：∫(exp(-a*x)*besselj(0, b*x)) dx，从 0 到 ∞
    assert integrate(exp(-a*x)*besselj(0, b*x),
                     (x, 0, oo)) == 1/(b*sqrt(a**2/b**2 + 1))


def test_W18():
    # 断言积分结果等于给定的数学表达式
    assert integrate((besselj(1, x)/x)**2, (x, 0, oo)) == 4/(3*pi)


@XFAIL
# 定义一个测试函数，用于测试未计算的积分结果
def test_W19():
    # 断言积分结果应该等于给定的数学表达式
    assert integrate(Ci(x)*besselj(0, 2*sqrt(7*x)), (x, 0, oo)) == (cos(7) - 1)/7


@XFAIL
# 定义一个测试函数，用于测试未计算的积分结果
def test_W20():
    # 断言某个积分结果等于给定的复杂数学表达式
    assert (integrate(x**2*polylog(3, 1/(x + 1)), (x, 0, 1)) ==
            -pi**2/36 - R(17, 108) + zeta(3)/4 +
            (-pi**2/2 - 4*log(2) + log(2)**2 + 35/3)*log(2)/9)


def test_W21():
    # 断言数值化后的积分结果接近于给定的数值
    assert abs(N(integrate(x**2*polylog(3, 1/(x + 1)), (x, 0, 1)))
               - 0.210882859565594) < 1e-15


def test_W22():
    # 定义实数符号和一个分段函数
    t, u = symbols('t u', real=True)
    s = Lambda(x, Piecewise((1, And(x >= 1, x <= 2)), (0, True)))
    # 计算积分：∫(s(t) * cos(t)) dt，从 0 到 u
    assert integrate(s(t)*cos(t), (t, 0, u)) == Piecewise(
        (0, u < 0),
        (-sin(Min(1, u)) + sin(Min(2, u)), True))


@slow
def test_W23():
    # 定义正数符号
    a, b = symbols('a b', positive=True)
    # 计算二重积分：∫∫(x / (x^2 + y^2)) dx dy，x 从 a 到 b，y 从 -∞ 到 ∞
    r1 = integrate(integrate(x/(x**2 + y**2), (x, a, b)), (y, -oo, oo))
    # 断言化简并整理后的积分结果应该等于给定的数学表达式
    assert r1.collect(pi).cancel() == -pi*a + pi*b


def test_W23b():
    # 定义正数符号
    a, b = symbols('a b', positive=True)
    # 计算二重积分：∫∫(x / (x^2 + y^2)) dy dx，y 从 -∞ 到 ∞，x 从 a 到 b
    r2 = integrate(integrate(x/(x**2 + y**2), (y, -oo, oo)), (x, a, b))
    # 断言整理后的积分结果应该等于给定的数学表达式
    assert r2.collect(pi) == pi*(-a + b)


@XFAIL
@tooslow
def test_W24():
    # 定义实数符号
    x, y = symbols('x y', real=True)
    # 计算二重积分：∫∫(sqrt(x^2 + y^2)) dx dy，x 从 0 到 1，y 从 0 到 1
    r1 = integrate(integrate(sqrt(x**2 + y**2), (x, 0, 1)), (y, 0, 1))
    # 断言语句，用于验证表达式是否为真
    assert (r1 - (sqrt(2) + asinh(1))/3).simplify() == 0
@XFAIL  # 标记此测试为预期失败，即测试预计会失败
@tooslow  # 标记此测试为运行速度较慢

def test_W25():  # 定义测试函数test_W25
    a, x, y = symbols('a x y', real=True)  # 声明符号变量a, x, y为实数
    i1 = integrate(  # 对sin(a)*sin(y)/sqrt(1 - sin(a)**2*sin(x)**2*sin(y)**2)在x从0到pi/2的区间进行积分，结果赋值给i1
        sin(a)*sin(y)/sqrt(1 - sin(a)**2*sin(x)**2*sin(y)**2),
        (x, 0, pi/2))
    i2 = integrate(i1, (y, 0, pi/2))  # 对i1在y从0到pi/2的区间进行积分，结果赋值给i2
    assert (i2 - pi*a/2).simplify() == 0  # 断言i2 - pi*a/2经简化后等于0


def test_W26():  # 定义测试函数test_W26
    x, y = symbols('x y', real=True)  # 声明符号变量x, y为实数
    assert integrate(integrate(abs(y - x**2), (y, 0, 2)),  # 对abs(y - x**2)在y从0到2的区间进行积分，结果再对x从-1到1进行积分，断言等于46/15
                     (x, -1, 1)) == R(46, 15)


def test_W27():  # 定义测试函数test_W27
    a, b, c = symbols('a b c')  # 声明符号变量a, b, c
    assert integrate(integrate(integrate(1, (z, 0, c*(1 - x/a - y/b))),  # 对1在z从0到c*(1 - x/a - y/b)的区间进行积分，结果再对y从0到b*(1 - x/a)进行积分，最后对x从0到a进行积分，断言等于a*b*c/6
                               (y, 0, b*(1 - x/a))),
                     (x, 0, a)) == a*b*c/6


def test_X1():  # 定义测试函数test_X1
    v, c = symbols('v c', real=True)  # 声明符号变量v, c为实数
    assert (series(1/sqrt(1 - (v/c)**2), v, x0=0, n=8) ==  # 计算在v=0附近的前8项级数展开，断言等于给定的级数表达式
            5*v**6/(16*c**6) + 3*v**4/(8*c**4) + v**2/(2*c**2) + 1 + O(v**8))


def test_X2():  # 定义测试函数test_X2
    v, c = symbols('v c', real=True)  # 声明符号变量v, c为实数
    s1 = series(1/sqrt(1 - (v/c)**2), v, x0=0, n=8)  # 计算在v=0附近的前8项级数展开，结果赋值给s1
    assert (1/s1**2).series(v, x0=0, n=8) == -v**2/c**2 + 1 + O(v**8)  # 对s1的倒数平方进行v=0附近的前8项级数展开，断言等于给定的级数表达式


def test_X3():  # 定义测试函数test_X3
    s1 = (sin(x).series()/cos(x).series()).series()  # 计算sin(x)和cos(x)的级数展开后的比值，并对结果进行级数展开
    s2 = tan(x).series()  # 计算tan(x)的级数展开
    assert s2 == x + x**3/3 + 2*x**5/15 + O(x**6)  # 断言tan(x)的级数展开结果符合给定表达式
    assert s1 == s2  # 断言s1等于s2


def test_X4():  # 定义测试函数test_X4
    s1 = log(sin(x)/x).series()  # 计算log(sin(x)/x)的级数展开
    assert s1 == -x**2/6 - x**4/180 + O(x**6)  # 断言s1等于给定的级数表达式
    assert log(series(sin(x)/x)).series() == s1  # 对sin(x)/x的级数展开后再计算其对数的级数展开，断言等于s1


@XFAIL  # 标记此测试为预期失败
def test_X5():  # 定义测试函数test_X5
    h = Function('h')  # 声明一个函数h
    a, b, c, d = symbols('a b c d', real=True)  # 声明符号变量a, b, c, d为实数
    series(diff(f(a*x), x) + g(b*x) + integrate(h(c*y), (y, 0, x)),  # 对f(a*x)关于x的导数加上g(b*x)再加上h(c*y)在y从0到x的积分，计算在x=d附近的前2项级数展开
           x, x0=d, n=2)
    # assert missing, until exception is removed


def test_X6():  # 定义测试函数test_X6
    a, b = symbols('a b', commutative=False, scalar=False)  # 声明符号变量a, b为非交换和非标量
    assert (series(exp((a + b)*x) - exp(a*x) * exp(b*x), x, x0=0, n=3) ==
              x**2*(-a*b/2 + b*a/2) + O(x**3))  # 断言级数展开结果符合给定的表达式


def test_X7():  # 定义测试函数test_X7
    assert (series(1/(x*(exp(x) - 1)), x, 0, 7) == x**(-2) - 1/(2*x) +  # 计算1/(x*(exp(x) - 1))在x=0附近的前7项级数展开，断言等于给定的级数表达式
            R(1, 12) - x**2/720 + x**4/30240 - x**6/1209600 + O(x**7))


def test_X8():  # 定义测试函数test_X8
    pass  # 空函数体，暂无具体测试内容
    # Puiseux series (terms with fractional degree):
    # => 1/sqrt(x - 3/2 pi) + (x - 3/2 pi)^(3/2) / 12 + O([x - 3/2 pi]^(7/2))

    # see issue 7167:
    # 定义符号变量 x，要求其为实数
    x = symbols('x', real=True)
    # 断言 Puiseux 级数展开的结果符合以下表达式
    assert (series(sqrt(sec(x)), x, x0=pi*3/2, n=4) ==
            # 第一项：1 / sqrt(x - pi*3/2)
            1/sqrt(x - pi*R(3, 2)) +
            # 第二项：(x - pi*3/2)^(3/2) / 12
            (x - pi*R(3, 2))**R(3, 2)/12 +
            # 第三项：O((x - pi*3/2)^(7/2))
            (x - pi*R(3, 2))**R(7, 2)/160 +
            O((x - pi*R(3, 2))**4, (x, pi*R(3, 2))))
def test_X9():
    # 断言：计算 x^x 的泰勒级数，展开到 x0=0，n=4
    assert (series(x**x, x, x0=0, n=4) == 1 + x*log(x) + x**2*log(x)**2/2 +
            x**3*log(x)**3/6 + O(x**4*log(x)**4))


def test_X10():
    z, w = symbols('z w')
    # 断言：计算 log(sinh(z)) + log(cosh(z + w)) 的泰勒级数，展开到 x0=0，n=2
    assert (series(log(sinh(z)) + log(cosh(z + w)), z, x0=0, n=2) ==
            log(cosh(w)) + log(z) + z*sinh(w)/cosh(w) + O(z**2))


def test_X11():
    z, w = symbols('z w')
    # 断言：计算 log(sinh(z) * cosh(z + w)) 的泰勒级数，展开到 x0=0，n=2
    assert (series(log(sinh(z) * cosh(z + w)), z, x0=0, n=2) ==
            log(cosh(w)) + log(z) + z*sinh(w)/cosh(w) + O(z**2))


@XFAIL
def test_X12():
    # 跳过测试：查看围绕 x = 1 的广义泰勒级数
    # 结果 => (x - 1)^a/e^b [1 - (a + 2 b) (x - 1) / 2 + O((x - 1)^2)]
    a, b, x = symbols('a b x', real=True)
    # series 返回 O(log(x-1)**2)
    # https://github.com/sympy/sympy/issues/7168
    assert (series(log(x)**a*exp(-b*x), x, x0=1, n=2) ==
            (x - 1)**a/exp(b)*(1 - (a + 2*b)*(x - 1)/2 + O((x - 1)**2)))


def test_X13():
    # 断言：计算 sqrt(2*x**2 + 1) 的泰勒级数，展开到 x0=oo，n=1
    assert series(sqrt(2*x**2 + 1), x, x0=oo, n=1) == sqrt(2)*x + O(1/x, (x, oo))


@XFAIL
def test_X14():
    # 断言：计算 Wallis' product 的泰勒级数，展开到 n=1
    # Wallis' product => 1/sqrt(pi n) + ...   [Knopp, p. 385]
    assert series(1/2**(2*n)*binomial(2*n, n),
                  n, x==oo, n=1) == 1/(sqrt(pi)*sqrt(n)) + O(1/x, (x, oo))


@SKIP("https://github.com/sympy/sympy/issues/7164")
def test_X15():
    # 断言：计算 Ei 的渐近展开，展开到 x0=oo，n=5
    x, t = symbols('x t', real=True)
    # 引发 RuntimeError: maximum recursion depth exceeded
    # https://github.com/sympy/sympy/issues/7164
    # 2019-02-17: 引发 PoleError: 不支持关于 [-oo] 的 Ei 的渐近展开。
    e1 = integrate(exp(-t)/t, (t, x, oo))
    assert (series(e1, x, x0=oo, n=5) ==
            6/x**4 + 2/x**3 - 1/x**2 + 1/x + O(x**(-5), (x, oo)))


def test_X16():
    # 断言：计算 cos(x + y) 的多变量泰勒级数，展开到 x0=0，n=4
    assert (series(cos(x + y), x + y, x0=0, n=4) == 1 - (x + y)**2/2 +
            O(x**4 + x**3*y + x**2*y**2 + x*y**3 + y**4, x, y))


@XFAIL
def test_X17():
    # 断言：计算 log(sin(x)/x) 的幂级数，计算通用公式
    # (c41) powerseries(log(sin(x)/x), x, 0);
    # /aquarius/data2/opt/local/macsyma_422/library1/trgred.so being loaded.
    #              inf
    #              ====     i1  2 i1          2 i1
    #              \        (- 1)   2      bern(2 i1) x
    # (d41)         >        ------------------------------
    #              /             2 i1 (2 i1)!
    #              ====
    #              i1 = 1
    # fps does not calculate
    assert fps(log(sin(x)/x)) == \
        Sum((-1)**k*2**(2*k - 1)*bernoulli(2*k)*x**(2*k)/(k*factorial(2*k)), (k, 1, oo))


@XFAIL
def test_X18():
    # 断言：计算 exp(-x)*sin(x) 的幂级数，计算通用公式
    # Maple FPS: FormalPowerSeries(exp(-x)*sin(x), x = 0);
    #                        infinity
    #                         -----    (1/2 k)                k
    #                          \      2        sin(3/4 k Pi) x
    #                           )     -------------------------
    assert fps(exp(-x)*sin(x)) == \
        Sum(2**(2*k - 1)*sin(3*pi*k/4)*x**k/factorial(k), (k, 0, oo))
    # 创建一个虚拟符号 k，用于表示一个未知的数学变量
    k = Dummy('k')
    # 断言，验证 SymPy 库中的 fps 函数对 exp(-x)*sin(x) 的结果
    assert fps(exp(-x)*sin(x)) == \
        # 计算无穷级数表达式，展示为 Sum 对象
        Sum(2**(S.Half*k)*sin(R(3, 4)*k*pi)*x**k/factorial(k), (k, 0, oo))
@XFAIL
def test_X19():
    """
    # (c45) /* Derive an explicit Taylor series solution of y as a function of
    # x from the following implicit relation:
    #    y = x - 1 + (x - 1)^2/2 + 2/3 (x - 1)^3 + (x - 1)^4 +
    #        17/10 (x - 1)^5 + ...
    #    */
    # x = sin(y) + cos(y);
    # Time= 0 msecs
    # (d45)                   x = sin(y) + cos(y)
    #
    # (c46) taylor_revert(%, y, 7);
    """
    raise NotImplementedError("Solve using series not supported. \
Inverse Taylor series expansion also not supported")


@XFAIL
def test_X20():
    """
    # Pade (rational function) approximation => (2 - x)/(2 + x)
    # > numapprox[pade](exp(-x), x = 0, [1, 1]);
    # bytes used=9019816, alloc=3669344, time=13.12
    #                                    1 - 1/2 x
    #                                    ---------
    #                                    1 + 1/2 x
    # mpmath support numeric Pade approximant but there is
    # no symbolic implementation in SymPy
    # https://en.wikipedia.org/wiki/Pad%C3%A9_approximant
    """
    raise NotImplementedError("Symbolic Pade approximant not supported")


def test_X21():
    """
    Test whether `fourier_series` of x periodical on the [-p, p] interval equals
    `- (2 p / pi) sum( (-1)^n / n sin(n pi x / p), n = 1..infinity )`.
    """
    p = symbols('p', positive=True)
    n = symbols('n', positive=True, integer=True)
    s = fourier_series(x, (x, -p, p))

    # All cosine coefficients are equal to 0
    assert s.an.formula == 0

    # Check for sine coefficients
    assert s.bn.formula.subs(s.bn.variables[0], 0) == 0
    assert s.bn.formula.subs(s.bn.variables[0], n) == \
        -2*p/pi * (-1)**n / n * sin(n*pi*x/p)


@XFAIL
def test_X22():
    """
    # (c52) /* => p / 2
    #    - (2 p / pi^2) sum( [1 - (-1)^n] cos(n pi x / p) / n^2,
    #                        n = 1..infinity ) */
    # fourier_series(abs(x), x, p);
    #                       p
    # (e52)                      a  = -
    #                       0      2
    #
    #                       %nn
    #                   (2 (- 1)    - 2) p
    # (e53)                a    = ------------------
    #                 %nn         2    2
    #                       %pi  %nn
    #
    # (e54)                     b    = 0
    #                      %nn
    #
    # Time= 5290 msecs
    #            inf           %nn            %pi %nn x
    #            ====       (2 (- 1)    - 2) cos(---------)
    #            \                    p
    #          p  >       -------------------------------
    #            /               2
    #            ====                %nn
    #            %nn = 1                     p
    # (d54)          ----------------------------------------- + -
    #                       2                 2
    #                    %pi
    """
    raise NotImplementedError("Fourier series not supported")


def test_Y1():
    """
    t = symbols('t', positive=True)
    w = symbols('w', real=True)
    s = symbols('s')
    F, _, _ = laplace_transform(cos((w - 1)*t), t, s)
    """
    # 断言语句，用于验证表达式 F == s / (s**2 + (w - 1)**2) 是否为真
    assert F == s / (s**2 + (w - 1)**2)
# 定义函数 test_Y2，用于测试特定的符号变换和逆拉普拉斯变换结果
def test_Y2():
    # 定义正数符号变量 t
    t = symbols('t', positive=True)
    # 定义实数符号变量 w
    w = symbols('w', real=True)
    # 定义通用符号变量 s
    s = symbols('s')
    # 计算逆拉普拉斯变换，得到函数 f
    f = inverse_laplace_transform(s/(s**2 + (w - 1)**2), s, t, simplify=True)
    # 断言结果 f 应为 cos(t*(w - 1))

# 定义函数 test_Y3，用于测试双曲正弦和双曲余弦的拉普拉斯变换结果
def test_Y3():
    # 定义正数符号变量 t
    t = symbols('t', positive=True)
    # 定义实数符号变量 w
    w = symbols('w', real=True)
    # 定义通用符号变量 s
    s = symbols('s')
    # 计算拉普拉斯变换，得到 F 和未使用的参数
    F, _, _ = laplace_transform(sinh(w*t)*cosh(w*t), t, s, simplify=True)
    # 断言结果 F 应为 w/(s**2 - 4*w**2)

# 定义函数 test_Y4，用于测试误差函数的拉普拉斯变换结果
def test_Y4():
    # 定义正数符号变量 t
    t = symbols('t', positive=True)
    # 定义通用符号变量 s
    s = symbols('s')
    # 计算拉普拉斯变换，得到 F 和未使用的参数
    F, _, _ = laplace_transform(erf(3/sqrt(t)), t, s, simplify=True)
    # 断言结果 F 应为 1/s - exp(-6*sqrt(s))/s

# 定义函数 test_Y5_Y6，解决二阶常微分方程和其拉普拉斯变换的问题
def test_Y5_Y6():
    # 定义实数符号变量 t
    t = symbols('t', real=True)
    # 定义通用符号变量 s
    s = symbols('s')
    # 定义函数 y 和 Y
    y = Function('y')
    Y = Function('Y')
    # 计算方程的拉普拉斯变换，得到 F
    F = laplace_correspondence(laplace_transform(diff(y(t), t, 2) + y(t)
                                - 4*(Heaviside(t - 1) - Heaviside(t - 2)),
                                t, s, noconds=True), {y: Y})
    # 计算差分方程 D，即拉普拉斯变换后的方程
    D = (
        -F + s**2*Y(s) - s*y(0) + Y(s) - Subs(Derivative(y(t), t), t, 0) -
        4*exp(-s)/s + 4*exp(-2*s)/s)
    # 断言结果 D 应为 0

    # 解 Y(s) 的表达式
    Yf = solve(F, Y(s))[0]
    # 应用初值条件，得到 Yf
    Yf = laplace_initial_conds(Yf, t, {y: [1, 0]})
    # 断言 Yf 的结果
    assert Yf == (s**2*exp(2*s) + 4*exp(s) - 4)*exp(-2*s)/(s*(s**2 + 1))

    # 反变换得到 y(t)，并整理结果
    yf = inverse_laplace_transform(Yf, s, t)
    yf = yf.collect(Heaviside(t-1)).collect(Heaviside(t-2))
    # 断言 yf 的结果
    assert yf == (
        (4 - 4*cos(t - 1))*Heaviside(t - 1) +
        (4*cos(t - 2) - 4)*Heaviside(t - 2) +
        cos(t)*Heaviside(t))

# 定义函数 test_Y7，测试无穷方波的拉普拉斯变换
@XFAIL
def test_Y7():
    # 定义正数符号变量 t
    t = symbols('t', positive=True)
    # 定义实数符号变量 a
    a = symbols('a', real=True)
    # 定义通用符号变量 s
    s = symbols('s')
    # 计算拉普拉斯变换，得到 F 和未使用的参数
    F, _, _ = laplace_transform(1 + 2*Sum((-1)**n*Heaviside(t - n*a),
                                  (n, 1, oo)), t, s)
    # 断言结果 F 应为 2*Sum((-1)**n*exp(-a*n*s)/s, (n, 1, oo)) + 1/s

# 定义函数 test_Y8，测试傅里叶变换的结果
@XFAIL
def test_Y8():
    # 断言傅里叶变换结果应为 DiracDelta(z)
    assert fourier_transform(1, x, z) == DiracDelta(z)

# 定义函数 test_Y9，留空，等待进一步编写
def test_Y9():
    pass
    # 断言语句：验证傅立叶变换的特定表达式是否与期望的值相等
    
    assert (fourier_transform(exp(-9*x**2), x, z) ==
            sqrt(pi)*exp(-pi**2*z**2/9)/3)
def test_Y10():
    # 断言：傅立叶变换后的表达式应当等于 (-8*pi**2*z**2 + 18)/(16*pi**4*z**4 + 72*pi**2*z**2 + 81)
    assert (fourier_transform(abs(x)*exp(-3*abs(x)), x, z).cancel() ==
            (-8*pi**2*z**2 + 18)/(16*pi**4*z**4 + 72*pi**2*z**2 + 81))


@SKIP("https://github.com/sympy/sympy/issues/7181")
@slow
def test_Y11():
    # 用于测试的符号变量：x, s
    x, s = symbols('x s')
    # 会引发 RuntimeError: maximum recursion depth exceeded
    # 详细信息见：https://github.com/sympy/sympy/issues/7181
    # 更新于 2019-02-17 引发 TypeError: cannot unpack non-iterable MellinTransform object
    # 执行梅林变换，计算变换后的结果 F，期望 F 等于 pi*cot(pi*s)
    F, _, _ =  mellin_transform(1/(1 - x), x, s)
    assert F == pi*cot(pi*s)


@XFAIL
def test_Y12():
    # 用于测试的符号变量：x, s
    x, s = symbols('x s')
    # 返回错误值 -2**(s - 4)*gamma(s/2 - 3)/gamma(-s/2 + 1)
    # 详细信息见：https://github.com/sympy/sympy/issues/7182
    # 执行梅林变换，计算变换后的结果 F，期望 F 等于 -2**(s - 4)*gamma(s/2)/gamma(-s/2 + 4)
    F, _, _ = mellin_transform(besselj(3, x)/x**3, x, s)
    assert F == -2**(s - 4)*gamma(s/2)/gamma(-s/2 + 4)


@XFAIL
def test_Y13():
    # 抛出未实现错误，不支持 z 变换
    raise NotImplementedError("z-transform not supported")


@XFAIL
def test_Y14():
    # 抛出未实现错误，不支持 z 变换
    raise NotImplementedError("z-transform not supported")


def test_Z1():
    # 定义函数符号变量：r
    r = Function('r')
    # 断言递推解析解是否等于预期的表达式 n**2 + n*(m - 2) + 1
    assert (rsolve(r(n + 2) - 2*r(n + 1) + r(n) - 2, r(n),
                   {r(0): 1, r(1): m}).simplify() == n**2 + n*(m - 2) + 1)


def test_Z2():
    # 定义函数符号变量：r
    r = Function('r')
    # 断言递推解析解是否等于预期的表达式 -2**n + 3**n
    assert (rsolve(r(n) - (5*r(n - 1) - 6*r(n - 2)), r(n), {r(0): 0, r(1): 1})
            == -2**n + 3**n)


def test_Z3():
    # 定义函数符号变量：r
    r = Function('r')
    # 预期的解析解表达式
    expected = ((S(1)/2 - sqrt(5)/2)**n*(S(1)/2 - sqrt(5)/10)
              + (S(1)/2 + sqrt(5)/2)**n*(sqrt(5)/10 + S(1)/2))
    # 求解递推关系 r(n) - (r(n - 1) + r(n - 2))，并断言解是否等于预期的表达式
    sol = rsolve(r(n) - (r(n - 1) + r(n - 2)), r(n), {r(1): 1, r(2): 2})
    assert sol == expected


@XFAIL
def test_Z4():
    # 定义函数符号变量：r, c
    r = Function('r')
    c = symbols('c')
    # 抛出 ValueError: Polynomial or rational function expected, got '(c**2 - c**n)/(c - c**n)
    s = rsolve(r(n) - ((1 + c - c**(n-1) - c**(n+1))/(1 - c**n)*r(n - 1)
                   - c*(1 - c**(n-2))/(1 - c**(n-1))*r(n - 2) + 1),
           r(n), {r(1): 1, r(2): (2 + 2*c + c**2)/(1 + c)})
    assert (s - (c*(n + 1)*(c*(n + 1) - 2*c - 2) +
             (n + 1)*c**2 + 2*c - n)/((c-1)**3*(c+1)) == 0)


@XFAIL
def test_Z5():
    # 抛出未实现错误，不支持 z 变换
    # Second order ODE with initial conditions---solve directly
    # transform: f(t) = sin(2 t)/8 - t cos(2 t)/4
    C1, C2 = symbols('C1 C2')
    # initial conditions not supported, this is a manual workaround
    raise NotImplementedError("z-transform not supported")
    # 定义一个微分方程对象，求解二阶导数为 f(x) 的微分方程
    eq = Derivative(f(x), x, 2) + 4*f(x) - sin(2*x)
    # 求解微分方程 eq，并得到其通解
    sol = dsolve(eq, f(x))
    # 将通解转换为一个 Lambda 函数，表示为 f0(x)
    f0 = Lambda(x, sol.rhs)
    # 断言验证初始条件下 f0(x) 的值符合特定的表达式
    assert f0(x) == C2*sin(2*x) + (C1 - x/4)*cos(2*x)
    # 计算 f0(x) 的导数，表示为 f1(x)
    f1 = Lambda(x, diff(f0(x), x))
    # 使用初始条件 f0(0) 和 f1(0) 解方程组，得到常数 C1 和 C2 的值
    const_dict = solve((f0(0), f1(0)))
    # 计算带有确定的常数值 C1 和 C2 的结果
    result = f0(x).subs(C1, const_dict[C1]).subs(C2, const_dict[C2])
    # 断言验证结果 result 符合预期的表达式
    assert result == -x*cos(2*x)/4 + sin(2*x)/8
    # 提示该方法能够得到正确结果，但是应该支持更简单的方式解决带初始条件的微分方程
    raise NotImplementedError('ODE solving with initial conditions \
# 使用装饰器 @XFAIL 标记该测试函数为预期失败的测试用例
@XFAIL
# 定义测试函数 test_Z6，用于测试处理二阶常微分方程的 Laplace 变换求解
def test_Z6():
    # 声明符号变量 t 为正数
    t = symbols('t', positive=True)
    # 声明符号变量 s
    s = symbols('s')
    # 定义常微分方程，包含二阶导数、常数和 sin 函数
    eq = Derivative(f(t), t, 2) + 4*f(t) - sin(2*t)
    # 对上述常微分方程进行 Laplace 变换，得到变换后的表达式 F
    F, _, _ = laplace_transform(eq, t, s)
    # 断言 Laplace 变换结果 F 符合预期的表达式
    # Laplace 变换中的 diff() 部分未计算，详情参见 GitHub 问题链接
    assert (F == s**2*LaplaceTransform(f(t), t, s) +
            4*LaplaceTransform(f(t), t, s) - 2/(s**2 + 4))
    # 后续的测试用例尚未实现
    # rest of test case not implemented
```