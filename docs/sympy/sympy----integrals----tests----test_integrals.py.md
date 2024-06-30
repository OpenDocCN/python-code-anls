# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_integrals.py`

```
# 导入数学库中的常用函数和符号
import math
# 导入 SymPy 中的求和相关模块
from sympy.concrete.summations import (Sum, summation)
# 导入 SymPy 中的加法相关模块
from sympy.core.add import Add
# 导入 SymPy 中的元组容器
from sympy.core.containers import Tuple
# 导入 SymPy 中的表达式基类
from sympy.core.expr import Expr
# 导入 SymPy 中的导数、函数、Lambda 函数、微分等
from sympy.core.function import (Derivative, Function, Lambda, diff)
# 导入 SymPy 中的欧拉常数
from sympy.core import EulerGamma
# 导入 SymPy 中的特殊数值常数（如自然常数 e、虚数单位 i、有理数、无穷大等）
from sympy.core.numbers import (E, I, Rational, nan, oo, pi, zoo, all_close)
# 导入 SymPy 中的关系运算（如等号、不等号）
from sympy.core.relational import (Eq, Ne)
# 导入 SymPy 中的单例对象
from sympy.core.singleton import S
# 导入 SymPy 中的符号（变量）及其创建方法
from sympy.core.symbol import (Symbol, symbols)
# 导入 SymPy 中的符号转换方法
from sympy.core.sympify import sympify
# 导入 SymPy 中的复数函数（如绝对值、实部、虚部等）
from sympy.functions.elementary.complexes import (Abs, im, polar_lift, re, sign)
# 导入 SymPy 中的指数函数（如 Lambert W 函数、指数函数、极坐标指数函数、对数函数）
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
# 导入 SymPy 中的双曲函数（如反双曲余弦、反双曲正弦等）
from sympy.functions.elementary.hyperbolic import (acosh, asinh, cosh, coth, csch, sinh, tanh, sech)
# 导入 SymPy 中的基本函数（如最大值、最小值、平方根等）
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
# 导入 SymPy 中的分段函数
from sympy.functions.elementary.piecewise import Piecewise
# 导入 SymPy 中的三角函数（如反余弦、反正弦等）
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan, sec)
# 导入 SymPy 中的特殊函数（如狄拉克 δ 函数、海布赛德 θ 函数等）
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
# 导入 SymPy 中的误差函数（如 Ci 函数、Ei 函数、Si 函数、误差函数等）
from sympy.functions.special.error_functions import (Ci, Ei, Si, erf, erfc, erfi, fresnelc, li)
# 导入 SymPy 中的 Gamma 函数及其相关函数
from sympy.functions.special.gamma_functions import (gamma, polygamma)
# 导入 SymPy 中的超函函数（如超函函数、梅耶格函数等）
from sympy.functions.special.hyper import (hyper, meijerg)
# 导入 SymPy 中的奇异函数
from sympy.functions.special.singularity_functions import SingularityFunction
# 导入 SymPy 中的勒让德函数
from sympy.functions.special.zeta_functions import lerchphi
# 导入 SymPy 中的积分函数
from sympy.integrals.integrals import integrate
# 导入 SymPy 中的布尔代数模块
from sympy.logic.boolalg import And
# 导入 SymPy 中的矩阵类
from sympy.matrices.dense import Matrix
# 导入 SymPy 中的多项式相关工具（如多项式、因式分解等）
from sympy.polys.polytools import (Poly, factor)
# 导入 SymPy 中的字符串表示相关模块
from sympy.printing.str import sstr
# 导入 SymPy 中的级数相关模块
from sympy.series.order import O
# 导入 SymPy 中的集合相关模块（如区间）
from sympy.sets.sets import Interval
# 导入 SymPy 中的简化函数（如 Gamma 函数的简化）
from sympy.simplify.gammasimp import gammasimp
# 导入 SymPy 中的简化函数（如表达式的简化）
from sympy.simplify.simplify import simplify
# 导入 SymPy 中的简化三角函数相关模块
from sympy.simplify.trigsimp import trigsimp
# 导入 SymPy 中的张量索引相关模块
from sympy.tensor.indexed import (Idx, IndexedBase)
# 导入 SymPy 中的表达式相关模块
from sympy.core.expr import unchanged
# 导入 SymPy 中的整数相关模块
from sympy.functions.elementary.integers import floor
# 导入 SymPy 中的积分相关模块
from sympy.integrals.integrals import Integral
# 导入 SymPy 中的非初等积分相关模块
from sympy.integrals.risch import NonElementaryIntegral
# 导入 SymPy 中的物理单位模块
from sympy.physics import units
# 导入 SymPy 中的测试相关模块
from sympy.testing.pytest import raises, slow, warns_deprecated_sympy, warns
# 导入 SymPy 中的异常模块
from sympy.utilities.exceptions import SymPyDeprecationWarning
# 导入 SymPy 中的随机数模块
from sympy.core.random import verify_numerically


# 创建符号变量
x, y, z, a, b, c, d, e, s, t, x_1, x_2 = symbols('x y z a b c d e s t x_1 x_2')
# 创建整数符号变量 n
n = Symbol('n', integer=True)
# 创建函数符号 f
f = Function('f')


# 定义一个函数 NS，用于将表达式 e 转换为数值字符串
def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)


# 定义一个测试函数 test_poly_deprecated，用于测试多项式的过时功能
def test_poly_deprecated():
    # 创建一个多项式对象 p
    p = Poly(2*x, x)
    # 断言 p 的积分结果与指定多项式的积分形式相等
    assert p.integrate(x) == Poly(x**2, x, domain='QQ')
    # 使用 warns 函数检测 SymPyDeprecationWarning 警告，验证是否正确设置了 stacklevel
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        integrate(p, x)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        Integral(p, (x,))


# 定义一个慢速测试函数 test_principal_value，用于测试主值积分
@slow
def test_principal_value():
    # 创建一个函数 g
    g = 1 / x
    # 断言 g 在无穷区间上的主值积分结果为 0
    assert Integral(g, (x, -oo, oo)).principal_value() == 0
    # 断言：计算 g 函数在无穷区间上的主值积分是否等于无穷乘以符号函数对 1/x 的值
    assert Integral(g, (y, -oo, oo)).principal_value() == oo * sign(1 / x)
    
    # 引发异常：计算 g 函数在仅有一个 x 变量的积分的主值时，预期会引发 ValueError
    raises(ValueError, lambda: Integral(g, (x)).principal_value())
    
    # 引发异常：计算 g 函数在没有指定积分变量的情况下的主值时，预期会引发 ValueError
    raises(ValueError, lambda: Integral(g).principal_value())

    # 计算 l 函数在无穷区间上的主值积分，检查其化简结果是否等于 -sqrt(3)*pi/3
    l = 1 / ((x ** 3) - 1)
    assert Integral(l, (x, -oo, oo)).principal_value().together() == -sqrt(3)*pi/3
    
    # 引发异常：计算 l 函数在指定积分区间 [-oo, 1] 上的主值时，预期会引发 ValueError
    raises(ValueError, lambda: Integral(l, (x, -oo, 1)).principal_value())

    # 计算 d 函数在无穷区间上的主值积分，检查其结果是否等于 0
    d = 1 / (x ** 2 - 1)
    assert Integral(d, (x, -oo, oo)).principal_value() == 0
    
    # 计算 d 函数在区间 [-2, 2] 上的主值积分，检查其结果是否等于 -log(3)
    assert Integral(d, (x, -2, 2)).principal_value() == -log(3)

    # 计算 v 函数在无穷区间上的主值积分，检查其结果是否等于 0
    v = x / (x ** 2 - 1)
    assert Integral(v, (x, -oo, oo)).principal_value() == 0
    
    # 计算 v 函数在区间 [-2, 2] 上的主值积分，检查其结果是否等于 0
    assert Integral(v, (x, -2, 2)).principal_value() == 0

    # 计算 s 函数在无穷区间上的主值积分，检查其结果是否为无穷
    s = x ** 2 / (x ** 2 - 1)
    assert Integral(s, (x, -oo, oo)).principal_value() is oo
    
    # 计算 s 函数在区间 [-2, 2] 上的主值积分，检查其结果是否等于 -log(3) + 4
    assert Integral(s, (x, -2, 2)).principal_value() == -log(3) + 4

    # 计算 f 函数在无穷区间上的主值积分，检查其结果是否等于 -pi / 2
    f = 1 / ((x ** 2 - 1) * (1 + x ** 2))
    assert Integral(f, (x, -oo, oo)).principal_value() == -pi / 2
    
    # 计算 f 函数在区间 [-2, 2] 上的主值积分，检查其结果是否等于 -atan(2) - log(3) / 2
    assert Integral(f, (x, -2, 2)).principal_value() == -atan(2) - log(3) / 2
def diff_test(i):
    """Return the set of symbols, s, which were used in testing that
    i.diff(s) agrees with i.doit().diff(s). If there is an error then
    the assertion will fail, causing the test to fail."""
    # 获取表达式 i 中的自由符号集合
    syms = i.free_symbols
    # 遍历每个自由符号 s
    for s in syms:
        # 断言 i 对 s 的导数减去 i.doit() 对 s 的导数等于零
        assert (i.diff(s).doit() - i.doit().diff(s)).expand() == 0
    # 返回自由符号集合
    return syms


def test_improper_integral():
    # 断言对于 log(x) 函数在区间 [0, 1] 上的积分应该等于 -1
    assert integrate(log(x), (x, 0, 1)) == -1
    # 断言对于 x^(-2) 函数在区间 [1, 无穷) 上的积分应该等于 1
    assert integrate(x**(-2), (x, 1, oo)) == 1
    # 断言对于 1/(1 + exp(x)) 函数在区间 [0, 无穷) 上的积分应该等于 log(2)
    assert integrate(1/(1 + exp(x)), (x, 0, oo)) == log(2)


def test_constructor():
    # 测试 Integral 构造函数，这与 Sum 的构造函数等价
    s1 = Integral(n, n)
    # 断言 s1 的限制是 (Tuple(n),)
    assert s1.limits == (Tuple(n),)
    s2 = Integral(n, (n,))
    assert s2.limits == (Tuple(n),)
    s3 = Integral(Sum(x, (x, 1, y)))
    assert s3.limits == (Tuple(y),)
    s4 = Integral(n, Tuple(n,))
    assert s4.limits == (Tuple(n),)

    s5 = Integral(n, (n, Interval(1, 2)))
    assert s5.limits == (Tuple(n, 1, 2),)

    # 测试带不等式的构造函数：
    s6 = Integral(n, n > 10)
    assert s6.limits == (Tuple(n, 10, oo),)
    s7 = Integral(n, (n > 2) & (n < 5))
    assert s7.limits == (Tuple(n, 2, 5),)


def test_basics():
    # 断言 Integral(0, x) 不等于 0
    assert Integral(0, x) != 0
    # 断言 Integral(x, (x, 1, 1)) 不等于 0
    assert Integral(x, (x, 1, 1)) != 0
    # 断言 Integral(oo, x) 不等于 oo
    assert Integral(oo, x) != oo
    # 断言 Integral(S.NaN, x) 等于 S.NaN
    assert Integral(S.NaN, x) is S.NaN

    # 断言对不同的积分进行导数操作结果正确
    assert diff(Integral(y, y), x) == 0
    assert diff(Integral(x, (x, 0, 1)), x) == 0
    assert diff(Integral(x, x), x) == x
    assert diff(Integral(t, (t, 0, x)), x) == x

    e = (t + 1)**2
    # 断言对积分进行导数操作后结果正确
    assert diff(integrate(e, (t, 0, x)), x) == ((1 + x)**2).expand()
    assert diff(integrate(e, (t, 0, x)), t) == 0
    assert diff(integrate(e, (t, 0, x)), a) == 0
    assert diff(integrate(e, t), a) == 0

    assert integrate(e, (t, a, x)).diff(x) == ((1 + x)**2).expand()
    assert Integral(e, (t, a, x)).diff(x).doit() == ((1 + x)**2)
    assert integrate(e, (t, x, a)).diff(x).doit() == (-(1 + x)**2).expand()

    assert integrate(t**2, (t, x, 2*x)).diff(x) == 7*x**2

    assert Integral(x, x).atoms() == {x}
    assert Integral(f(x), (x, 0, 1)).atoms() == {S.Zero, S.One, x}

    assert diff_test(Integral(x, (x, 3*y))) == {y}
    assert diff_test(Integral(x, (a, 3*y))) == {x, y}

    assert integrate(x, (x, oo, oo)) == 0  # issue 8171
    assert integrate(x, (x, -oo, -oo)) == 0

    assert integrate(y + x + exp(x), x) == x*y + x**2/2 + exp(x)

    assert Integral(x).is_commutative
    n = Symbol('n', commutative=False)
    assert Integral(n + x, x).is_commutative is False


def test_diff_wrt():
    class Test(Expr):
        _diff_wrt = True
        is_commutative = True

    t = Test()
    assert integrate(t + 1, t) == t**2/2 + t
    # 断言：计算积分 t + 1 在 t 从 0 到 1 的结果，预期结果为有理数 3/2
    assert integrate(t + 1, (t, 0, 1)) == Rational(3, 2)
    
    # 断言：对于积分表达式 x + 1，使用 x + 1 作为积分变量，预期抛出 ValueError 异常
    raises(ValueError, lambda: integrate(x + 1, x + 1))
    
    # 断言：对于积分表达式 x + 1，使用 x + 1 作为积分变量，并指定积分区间为 0 到 1，预期抛出 ValueError 异常
    raises(ValueError, lambda: integrate(x + 1, (x + 1, 0, 1)))
# 定义一个测试函数，用于测试多重积分中的差异
def test_basics_multiple():
    # 测试第一个积分表达式的符号差异，返回一个只包含 x 的集合
    assert diff_test(Integral(x, (x, 3*x, 5*y), (y, x, 2*x))) == {x}
    # 测试第二个积分表达式的符号差异，返回一个只包含 x 的集合
    assert diff_test(Integral(x, (x, 5*y), (y, x, 2*x))) == {x}
    # 测试第三个积分表达式的符号差异，返回一个包含 x 和 y 的集合
    assert diff_test(Integral(x, (x, 5*y), (y, y, 2*x))) == {x, y}
    # 测试第四个积分表达式的符号差异，返回一个包含 x 和 y 的集合
    assert diff_test(Integral(y, y, x)) == {x, y}
    # 测试第五个积分表达式的符号差异，返回一个包含 x 和 y 的集合
    assert diff_test(Integral(y*x, x, y)) == {x, y}
    # 测试第六个积分表达式的符号差异，返回一个只包含 x 的集合
    assert diff_test(Integral(x + y, y, (y, 1, x))) == {x}
    # 测试第七个积分表达式的符号差异，返回一个包含 x 和 y 的集合
    assert diff_test(Integral(x + y, (x, x, y), (y, y, x))) == {x, y}


# 定义一个测试函数，用于测试共轭转置操作
def test_conjugate_transpose():
    # 声明符号 A 和 B，设定它们为非交换的符号变量
    A, B = symbols("A B", commutative=False)

    # 声明一个复数域中的符号 x
    x = Symbol("x", complex=True)
    # 创建积分对象 p，积分变量为 x
    p = Integral(A*B, (x,))
    # 断言共轭转置的结果等于积分的结果进行共轭转置后的结果
    assert p.adjoint().doit() == p.doit().adjoint()
    # 断言共轭的结果等于积分的结果进行共轭后的结果
    assert p.conjugate().doit() == p.doit().conjugate()
    # 断言转置的结果等于积分的结果进行转置后的结果
    assert p.transpose().doit() == p.doit().transpose()

    # 声明一个实数域中的符号 x
    x = Symbol("x", real=True)
    # 创建积分对象 p，积分变量为 x
    p = Integral(A*B, (x,))
    # 断言共轭转置的结果等于积分的结果进行共轭转置后的结果
    assert p.adjoint().doit() == p.doit().adjoint()
    # 断言共轭的结果等于积分的结果进行共轭后的结果
    assert p.conjugate().doit() == p.doit().conjugate()
    # 断言转置的结果等于积分的结果进行转置后的结果
    assert p.transpose().doit() == p.doit().transpose()


# 定义一个测试函数，用于测试积分计算
def test_integration():
    # 断言对常数函数积分在 [0, x] 区间的结果为 0
    assert integrate(0, (t, 0, x)) == 0
    # 断言对常数函数积分在 [0, x] 区间的结果为 3*x
    assert integrate(3, (t, 0, x)) == 3*x
    # 断言对 t 的积分在 [0, x] 区间的结果为 x^2/2
    assert integrate(t, (t, 0, x)) == x**2/2
    # 断言对 3*t 的积分在 [0, x] 区间的结果为 3*x^2/2
    assert integrate(3*t, (t, 0, x)) == 3*x**2/2
    # 断言对 3*t^2 的积分在 [0, x] 区间的结果为 x^3
    assert integrate(3*t**2, (t, 0, x)) == x**3
    # 断言对 1/t 的积分在 [1, x] 区间的结果为 ln(x)
    assert integrate(1/t, (t, 1, x)) == log(x)
    # 断言对 -1/t^2 的积分在 [1, x] 区间的结果为 1/x - 1
    assert integrate(-1/t**2, (t, 1, x)) == 1/x - 1
    # 断言对 t^2 + 5*t - 8 的积分在 [0, x] 区间的结果为 x^3/3 + 5*x^2/2 - 8*x
    assert integrate(t**2 + 5*t - 8, (t, 0, x)) == x**3/3 + 5*x**2/2 - 8*x
    # 断言对 x^2 的积分结果为 x^3/3
    assert integrate(x**2, x) == x**3/3
    # 断言对 (3*t*x)^5 的积分结果为 (3*t)^5 * x^6 / 6
    assert integrate((3*t*x)**5, x) == (3*t)**5 * x**6 / 6

    # 声明符号变量 b 和 c
    b = Symbol("b")
    c = Symbol("c")
    # 断言对 a*t 的积分在 [0, x] 区间的结果为 a*x^2/2
    assert integrate(a*t, (t, 0, x)) == a*x**2/2
    # 断言对 a*t^4 的积分在 [0, x] 区间的结果为 a*x^5/5
    assert integrate(a*t**4, (t, 0, x)) == a*x**5/5
    # 断言对 a*t^2 + b*t + c 的积分在 [0, x] 区间的结果为 a*x^3/3 + b*x^2/2 + c*x
    assert integrate(a*t**2 + b*t + c, (t, 0, x)) == a*x**3/3 + b*x**2/2 + c*x


# 定义一个测试函数，用于测试多重积分
def test_multiple_integration():
    # 断言对 (x^2)*(y^2) 的二重积分在 x 属于 [0, 1]，y 属于 [-1, 2] 区间的结果为 1
    assert integrate((x**2)*(y**2), (x, 0, 1), (y, -1, 2)) == Rational(1)
    # 断言对 (y^2)*(x^2) 的二重积分在 x 和 y 上的结果为 (1/9)*(x^3)*(y^3)
    assert integrate((y**2)*(x**2), x, y) == Rational(1, 9)*(x**3)*(y**3)
    # 断言对 1/(x + 3)/(1 + x)^3 的积分结果
    assert integrate(1/(x + 3)/(1 + x)**3, x) == \
        log(3 + x)*Rational(-1, 8) + log(1 + x)*Rational(1, 8) + x/(4 + 8*x + 4*x**2)
    # 断言对 sin(x*y)*y 的二重积分在 x 属于 [0, 1]，y 属于 [0, 1] 区间的结果为 -sin(1) + 1


# 定义一个测试函数，用于测试 issue 3532 的积分
def test_issue_3532():
    # 断言对 exp(-x) 的积分在 [0, 无穷] 区间的结果为 1
    assert integrate(exp(-x), (x, 0,
    # 确定 qx 的表达式是否等于 x^2/2 + x^3*y/3 + x*y^3
    assert qx.as_expr() == x**2/2 + x**3*y/3 + x*y**3
    
    # 确定 qy 的表达式是否等于 x*y + x^2*y^2/2 + y^4/4
    assert qy.as_expr() == x*y + x**2*y**2/2 + y**4/4
# 定义测试函数 test_integrate_poly_definite
def test_integrate_poly_definite():
    # 创建多项式 p = x + x**2*y + y**3，其中 x 和 y 是符号变量
    p = Poly(x + x**2*y + y**3, x, y)

    # 在使用过程中，警告使用了不推荐的 SymPy 函数或功能
    with warns_deprecated_sympy():
        # 计算多项式 p 在 x 轴上从 0 到 1 的定积分 Qx
        Qx = Integral(p, (x, 0, 1))
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        # 使用积分函数 integrate 计算多项式 p 在 x 轴上从 0 到 1 的定积分 Qx
        Qx = integrate(p, (x, 0, 1))
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        # 使用积分函数 integrate 计算多项式 p 在 y 轴上从 0 到 pi 的定积分 Qy
        Qy = integrate(p, (y, 0, pi))

    # 断言 Qx 和 Qy 的类型是多项式 Poly
    assert isinstance(Qx, Poly) is True
    assert isinstance(Qy, Poly) is True

    # 断言 Qx 的生成器是 (y,)
    assert Qx.gens == (y,)
    # 断言 Qy 的生成器是 (x,)
    assert Qy.gens == (x,)

    # 断言 Qx 的表达式形式等于 S.Half + y/3 + y**3
    assert Qx.as_expr() == S.Half + y/3 + y**3
    # 断言 Qy 的表达式形式等于 pi**4/4 + pi*x + pi**2*x**2/2
    assert Qy.as_expr() == pi**4/4 + pi*x + pi**2*x**2/2


# 定义测试函数 test_integrate_omit_var
def test_integrate_omit_var():
    y = Symbol('y')

    # 断言对 x 的不定积分 integrate(x) 结果为 x**2/2
    assert integrate(x) == x**2/2

    # 断言对于整数 2 和表达式 x*y 调用 integrate 函数会引发 ValueError 异常
    raises(ValueError, lambda: integrate(2))
    raises(ValueError, lambda: integrate(x*y))


# 定义测试函数 test_integrate_poly_accurately
def test_integrate_poly_accurately():
    y = Symbol('y')

    # 断言对 x 的不定积分 integrate(x*sin(y), x) 结果为 x**2*sin(y)/2
    assert integrate(x*sin(y), x) == x**2*sin(y)/2

    # 断言对 x 的不定积分 integrate(x**1000*sin(y), x) 结果为 x**1001*sin(y)/1001
    # 这是一个对于性能的检查，因为这个函数在使用 risch_norman 算法时会非常耗费 CPU
    assert integrate(x**1000*sin(y), x) == x**1001*sin(y)/1001


# 定义测试函数 test_issue_3635
def test_issue_3635():
    y = Symbol('y')

    # 断言对 y 的不定积分 integrate(x**2, y) 结果为 x**2*y
    assert integrate(x**2, y) == x**2*y

    # 断言对 y 在区间 [-1, 1] 上的定积分 integrate(x**2, (y, -1, 1)) 结果为 2*x**2
    assert integrate(x**2, (y, -1, 1)) == 2*x**2

    # 在 SymPy 和 py.test 中正常工作，但在 `setup.py test` 中会出现问题


# 定义测试函数 test_integrate_linearterm_pow
def test_integrate_linearterm_pow():
    # 检查 integrate((a*x+b)^c, x) 的问题，参考 issue 3499
    y = Symbol('y', positive=True)

    # TODO: 在这里移除 conds='none'，让前提条件来处理
    # 断言对 x 的不定积分 integrate(x**y, x, conds='none') 结果为 x**(y + 1)/(y + 1)
    assert integrate(x**y, x, conds='none') == x**(y + 1)/(y + 1)

    # 断言对 x 的不定积分 integrate((exp(y)*x + 1/y)**(1 + sin(y)), x, conds='none')
    # 结果为 exp(-y)*(exp(y)*x + 1/y)**(2 + sin(y)) / (2 + sin(y))
    assert integrate((exp(y)*x + 1/y)**(1 + sin(y)), x, conds='none') == \
        exp(-y)*(exp(y)*x + 1/y)**(2 + sin(y)) / (2 + sin(y))


# 定义测试函数 test_issue_3618
def test_issue_3618():
    # 断言对 x 的不定积分 integrate(pi*sqrt(x), x) 结果为 2*pi*sqrt(x)**3/3
    assert integrate(pi*sqrt(x), x) == 2*pi*sqrt(x)**3/3

    # 断言对 x 的不定积分 integrate(pi*sqrt(x) + E*sqrt(x)**3, x) 结果为
    # 2*pi*sqrt(x)**3/3 + 2*E *sqrt(x)**5/5
    assert integrate(pi*sqrt(x) + E*sqrt(x)**3, x) == \
        2*pi*sqrt(x)**3/3 + 2*E *sqrt(x)**5/5


# 定义测试函数 test_issue_3623
def test_issue_3623():
    # 断言对 x 的不定积分 integrate(cos((n + 1)*x), x) 结果为 Piecewise(
    # (sin(x*(n + 1))/(n + 1), Ne(n + 1, 0)), (x, True))
    assert integrate(cos((n + 1)*x), x) == Piecewise(
        (sin(x*(n + 1))/(n + 1), Ne(n + 1, 0)), (x, True))

    # 断言对 x 的不定积分 integrate(cos((n - 1)*x), x) 结果为 Piecewise(
    # (sin(x*(n - 1))/(n - 1), Ne(n - 1, 0)), (x, True))
    assert integrate(cos((n - 1)*x), x) == Piecewise(
        (sin(x*(n - 1))/(n - 1), Ne(n - 1, 0)), (x, True))

    # 断言对 x 的不定积分 integrate(cos((n + 1)*x) + cos((n - 1)*x), x) 结果为
    # Piecewise((sin(x*(n - 1))/(n - 1), Ne(n - 1, 0)), (x, True)) +
    # Piecewise((sin(x*(n + 1))/(n + 1), Ne(n + 1, 0)), (x, True))
    assert integrate(cos((n + 1)*x) + cos((n - 1)*x), x) == \
        Piecewise((sin(x*(n - 1))/(n - 1), Ne(n - 1, 0)), (x, True)) + \
        Piecewise((sin(x*(n + 1))/(n + 1), Ne(n + 1, 0)), (x, True))


# 定义测试函数 test_issue_3664
def test_issue_3664():
    # 定义符号变量 n，限定为整数且非零
    n = Symbol('n', integer=True, nonzero=True)

    # 断言对 x 在区间 [-2, 0] 的定积分 integrate(-1./2 * x * sin(n * pi * x/2), [x, -2, 0]) 结果为
    # 2.0*cos(pi*n)/(pi*n)
    assert integrate(-1./2 * x * sin(n * pi * x/2), [x, -2, 0]) == \
        2.0*cos(pi*n)/(pi*n)

    # 断言对 x 在区间 [-2, 0] 的定积分 integrate(x * sin(n * pi * x/2) * Rational(-1, 2), [x, -2, 0]) 结果为
    # 2*cos(pi*n)/(pi*n)
    assert integrate(x * sin(n *
    # 使用 assert 语句来检查函数 expand_func(integrate(sin(x**2), x)) 的返回值是否等于
    # sqrt(2)*sqrt(pi)*fresnels(sqrt(2)*x/sqrt(pi))/2
    assert expand_func(integrate(sin(x**2), x)) == \
        sqrt(2)*sqrt(pi)*fresnels(sqrt(2)*x/sqrt(pi))/2
# 定义一个测试函数，用于测试单位集成
def test_integrate_units():
    # 设置单位 m 和 s
    m = units.m
    s = units.s
    # 断言积分结果为 12*m*s，积分区间为从 1*s 到 5*s
    assert integrate(x * m/s, (x, 1*s, 5*s)) == 12*m*s


# 定义一个测试函数，用于测试超越函数的积分
def test_transcendental_functions():
    # 断言 LambertW(2*x) 的积分结果
    assert integrate(LambertW(2*x), x) == \
        -x + x*LambertW(2*x) + x/LambertW(2*x)


# 定义一个测试函数，用于测试对数和多项对数函数的积分
def test_log_polylog():
    # 断言 log(1 - x)/x 的积分结果
    assert integrate(log(1 - x)/x, (x, 0, 1)) == -pi**2/6
    # 断言 log(x)*(1 - x)**(-1) 的积分结果
    assert integrate(log(x)*(1 - x)**(-1), (x, 0, 1)) == -pi**2/6


# 定义一个测试函数，用于测试 GitHub 上的问题 #3740
def test_issue_3740():
    # 定义函数 f = 4*log(x) - 2*log(x)**2
    f = 4*log(x) - 2*log(x)**2
    # 计算对 f 的积分，并对 x 求导，然后在 x = 42 处进行数值计算
    fid = diff(integrate(f, x), x)
    assert abs(f.subs(x, 42).evalf() - fid.subs(x, 42).evalf()) < 1e-10


# 定义一个测试函数，用于测试 GitHub 上的问题 #3788
def test_issue_3788():
    # 断言 1/(1 + x**2) 的积分结果
    assert integrate(1/(1 + x**2), x) == atan(x)


# 定义一个测试函数，用于测试 GitHub 上的问题 #3952
def test_issue_3952():
    # 定义函数 f = sin(x)
    f = sin(x)
    # 断言 sin(x) 的积分结果
    assert integrate(f, x) == -cos(x)
    # 断言对 2*x 求积分会引发 ValueError 异常
    raises(ValueError, lambda: integrate(f, 2*x))


# 定义一个测试函数，用于测试 GitHub 上的问题 #4516
def test_issue_4516():
    # 断言 2**x - 2*x 的积分结果
    assert integrate(2**x - 2*x, x) == 2**x/log(2) - x**2


# 定义一个测试函数，用于测试 GitHub 上的问题 #7450
def test_issue_7450():
    # 计算指数函数的积分结果
    ans = integrate(exp(-(1 + I)*x), (x, 0, oo))
    # 断言结果的实部和虚部满足条件
    assert re(ans) == S.Half and im(ans) == Rational(-1, 2)


# 定义一个测试函数，用于测试 GitHub 上的问题 #8623
def test_issue_8623():
    # 断言 (1 + cos(2*x)) / (3 - 2*cos(2*x)) 的积分结果
    assert integrate((1 + cos(2*x)) / (3 - 2*cos(2*x)), (x, 0, pi)) == -pi/2 + sqrt(5)*pi/2
    # 断言 (1 + cos(2*x)) / (3 - 2*cos(2*x)) 的无限积分结果
    assert integrate((1 + cos(2*x))/(3 - 2*cos(2*x))) == -x/2 + sqrt(5)*(atan(sqrt(5)*tan(x)) + \
        pi*floor((x - pi/2)/pi))/2


# 定义一个测试函数，用于测试 GitHub 上的问题 #9569
def test_issue_9569():
    # 断言 1 / (2 - cos(x)) 的积分结果
    assert integrate(1 / (2 - cos(x)), (x, 0, pi)) == pi/sqrt(3)
    # 断言 1 / (2 - cos(x)) 的无限积分结果
    assert integrate(1/(2 - cos(x))) == 2*sqrt(3)*(atan(sqrt(3)*tan(x/2)) + pi*floor((x/2 - pi/2)/pi))/3


# 定义一个测试函数，用于测试 GitHub 上的问题 #13733
def test_issue_13733():
    # 定义符号 s，并要求其为正数
    s = Symbol('s', positive=True)
    # 定义概率密度函数 pz
    pz = exp(-(z - y)**2/(2*s*s))/sqrt(2*pi*s*s)
    # 对 pz 进行积分，积分变量为 z，积分区间为从 x 到无穷大
    pzgx = integrate(pz, (z, x, oo))
    # 断言对 pzgx 在 x 从 0 到无穷大的积分结果
    assert integrate(pzgx, (x, 0, oo)) == sqrt(2)*s*exp(-y**2/(2*s**2))/(2*sqrt(pi)) + \
        y*erf(sqrt(2)*y/(2*s))/2 + y/2


# 定义一个测试函数，用于测试 GitHub 上的问题 #13749
def test_issue_13749():
    # 断言 1 / (2 + cos(x)) 的积分结果
    assert integrate(1 / (2 + cos(x)), (x, 0, pi)) == pi/sqrt(3)
    # 断言 1 / (2 + cos(x)) 的无限积分结果
    assert integrate(1/(2 + cos(x))) == 2*sqrt(3)*(atan(sqrt(3)*tan(x/2)/3) + pi*floor((x/2 - pi/2)/pi))/3


# 定义一个测试函数，用于测试 GitHub 上的问题 #18133
def test_issue_18133():
    # 断言 exp(x)/(1 + x)**2 的积分结果
    assert integrate(exp(x)/(1 + x)**2, x) == NonElementaryIntegral(exp(x)/(x + 1)**2, x)


# 定义一个测试函数，用于测试 GitHub 上的问题 #21741
def test_issue_21741():
    # 定义常数 a 和 b
    a = 4e6
    b = 2.5e-7
    # 定义 Piecewise 对象 r
    r = Piecewise((b*I*exp(-a*I*pi*t*y)*exp(-a*I*pi*x*z)/(pi*x), Ne(x, 0)),
                  (z*exp(-a*I*pi*t*y), True))
    # 定义函数 fun
    fun = E**((-2*I*pi*(z*x+t*y))/(500*10**(-9)))
    # 断言 fun 对 z 的积分结果满足 all_close 条件
    assert all_close(integrate(fun, z), r)


# 定义一个测试函数，用于测试矩阵积分
def test_matrices():
    # 定义矩阵 M，其元素为 (i + j + 1)*sin((i + j + 1)*x)
    M = Matrix(2, 2, lambda i, j: (i + j + 1)*sin((i + j + 1)*x))
    # 断言矩阵 M 对 x 的积分结果
    assert integrate(M, x) == Matrix([
        [-cos(x), -cos(2*x)],
        [-cos(2*x), -cos(3*x)],
    ])


# 定义一个测试函数，用于测试函数积分
def test_integrate_functions():
    # issue 4111
    # 断言对 f(x) 的积分结果
    assert integrate(f(x), x) == Integral(f(x), x)
    # 断言对 f(x) 的积分结果，积分区间为从 0 到 1
    assert integrate(f(x), (x, 0, 1)) == Integral(f(x), (x, 0, 1))
    # 断言 f(x)*diff(f(x), x) 的积分结果
    assert integrate(f(x)*diff(f(x), x), x) == f(x)**2
    # 使用断言来验证积分结果是否等于对应的导数的平方的积分
    assert integrate(Derivative(f(x), x)**2, x) == \
        Integral(Derivative(f(x), x)**2, x)
# 定义测试函数 test_transform，用于测试积分变换的功能
def test_transform():
    # 定义积分表达式 a = ∫(x^2 + 1) dx 在区间 [-1, 2] 上的积分
    a = Integral(x**2 + 1, (x, -1, 2))
    # 定义变量 fx = x
    fx = x
    # 定义变量 fy = 3*y + 1
    fy = 3*y + 1
    # 断言积分 a 的值与将 fx, fy 应用的转换后的积分值相等
    assert a.doit() == a.transform(fx, fy).doit()
    # 断言对积分应用两次变换 fx, fy 和 fy, fx 后得到原始积分 a
    assert a.transform(fx, fy).transform(fy, fx) == a
    # 修改变量 fx = 3*x + 1, fy = y
    fx = 3*x + 1
    fy = y
    # 断言对积分应用两次变换 fx, fy 和 fy, fx 后得到原始积分 a
    assert a.transform(fx, fy).transform(fy, fx) == a
    # 定义积分表达式 a = ∫(sin(1/x)) dx 在区间 [0, 1] 上的积分
    a = Integral(sin(1/x), (x, 0, 1))
    # 断言将变量 x 替换为 1/y 后的积分结果
    assert a.transform(x, 1/y) == Integral(sin(y)/y**2, (y, 1, oo))
    # 断言对积分应用两次变换 x, 1/y 和 y, 1/x 后得到原始积分 a
    assert a.transform(x, 1/y).transform(y, 1/x) == a
    # 定义积分表达式 a = ∫(exp(-x^2)) dx 在区间 [-∞, ∞] 上的积分
    a = Integral(exp(-x**2), (x, -oo, oo))
    # 断言将变量 x 替换为 2*y 后的积分结果
    assert a.transform(x, 2*y) == Integral(2*exp(-4*y**2), (y, -oo, oo))
    # 断言对积分应用两次变换 x 和 a*y 后的结果
    assert Integral(x, x).transform(x, a*y).doit() == \
        Integral(y*a**2, y).doit()
    # 定义符号 _3 = 3
    _3 = S(3)
    # 断言对积分应用变换 x -> 1/y 后的结果
    assert Integral(x, (x, 0, -_3)).transform(x, 1/y).doit() == \
        Integral(-1/x**3, (x, -oo, -1/_3)).doit()
    # 断言对积分应用变换 x -> 1/y 后的结果
    assert Integral(x, (x, 0, _3)).transform(x, 1/y) == \
        Integral(y**(-3), (y, 1/_3, oo))
    # issue 8400 的问题
    # 创建一个复杂的积分对象 i
    i = Integral(x + y, (x, 1, 2), (y, 1, 2))
    # 断言对 i 应用变换 x -> (x + 2*y, x) 后得到的结果
    assert i.transform(x, (x + 2*y, x)).doit() == \
        i.transform(x, (x + 2*z, x)).doit() == 3

    # 创建一个积分对象 i = ∫(x) dx 在区间 [a, b] 上
    i = Integral(x, (x, a, b))
    # 断言对 i 应用变换 x -> 2*s 后的结果
    assert i.transform(x, 2*s) == Integral(4*s, (s, a/2, b/2))
    # 断言当尝试应用无效的变换时会抛出 ValueError 异常
    raises(ValueError, lambda: i.transform(x, 1))
    raises(ValueError, lambda: i.transform(x, s*t))
    raises(ValueError, lambda: i.transform(x, -s))
    raises(ValueError, lambda: i.transform(x, (s, t)))
    raises(ValueError, lambda: i.transform(2*x, 2*s))

    # 创建一个积分对象 i = ∫(x^2) dx 在区间 [1, 2] 上
    i = Integral(x**2, (x, 1, 2))
    # 断言当尝试应用无效的变换时会抛出 ValueError 异常
    raises(ValueError, lambda: i.transform(x**2, s))

    # 创建一个积分对象 i = ∫(x) dx 在区间 [b, a] 上
    am = Symbol('a', negative=True)
    bp = Symbol('b', positive=True)
    i = Integral(x, (x, bp, am))
    # 对积分 i 应用变换 x -> 2*s 后的结果
    i.transform(x, 2*s)
    assert i.transform(x, 2*s) == Integral(-4*s, (s, am/2, bp/2))

    # 创建一个积分对象 i = ∫(x) dx 在区间 [a, oo] 上
    i = Integral(x, (x, a))
    # 对积分 i 应用变换 x -> 2*s 后的结果
    assert i.transform(x, 2*s) == Integral(4*s, (s, a/2))


# 定义测试函数 test_issue_4052，用于测试积分问题 4052 的处理
def test_issue_4052():
    # 定义表达式 f = 1/2 * asin(x) + x * sqrt(1 - x**2)/2
    f = S.Half*asin(x) + x*sqrt(1 - x**2)/2

    # 断言对 cos(asin(x)) 的积分结果
    assert integrate(cos(asin(x)), x) == f
    # 断言对 sin(acos(x)) 的积分结果
    assert integrate(sin(acos(x)), x) == f


# 定义带有 @slow 装饰器的测试函数 test_evalf_integrals，用于测试数值积分的精度
@slow
def test_evalf_integrals():
    # 断言数值积分 ∫(x) dx 在区间 [2, 5] 的数值结果
    assert NS(Integral(x, (x, 2, 5)), 15) == '10.5000000000000'
    # 创建高斯积分对象 gauss = ∫(exp(-x**2)) dx 在区间 [-∞, ∞]
    gauss = Integral(exp(-x**2), (x, -oo, oo))
    # 断言高斯积分的数值结果
    assert NS(gauss, 15) == '1.77245385090552'
    # 断言高斯积分的平方减去 π 加上 E 乘以一个极小分数的数值结果
    assert NS(gauss**2 - pi + E*Rational(
        1, 10**20), 15) in ('2.71828182845904e-20', '2.71828182845905e-20')
    # 创建复杂的积分对象 t = Symbol('t')
    # 定义复杂积分的表达式 f
    t = Symbol('t')
    a = 8*sqrt(3)/(1 + 3*t**2)
    b = 16*sqrt(2)*(3*t + 1)*sqrt(4*t**2 + t + 1)**3
    c = (3*t**2 + 1)*(11*t**2 + 2*t + 3)**2
    d = sqrt(2)*(249*t**2 + 54*t + 65)/(11*t**2 + 2*t + 3)**2
    f = a - b/c - d
    # 断言复杂积分的数值结果
    assert NS(Integral(f, (t, 0, 1)), 50) == \
        NS((3*sqrt(2) - 49*pi + 162*atan(sqrt(2)))/12, 50)
    # 断言特定积分 http://math
    # 确保 Abel 积分结果正确至小数点后15位
    assert NS(Integral(atan(sqrt(x**2 + 2))/(sqrt(x**2 + 2)*(x**2 + 1)), (x,
              0, 1)), 15) == NS(5*pi**2/96, 15)
    # 参考数学世界上的 Abel 积分定义

    # 确保 Vardis 积分结果正确至小数点后15位，修剪复数部分
    assert NS(Integral(x/((exp(pi*x) - exp(
        -pi*x))*(x**2 + 1)), (x, 0, oo)), 15) == NS('log(2)/2-1/4', 15)
    # 参考数学世界上的 Vardis 积分定义

    # 确保对于给定区间 [pi/4, pi/2]，log(sin(x)/cos(x)) 的积分结果正确至小数点后15位，修剪复数部分
    assert NS(Integral(log(log(sin(x)/cos(x))), (x, pi/4, pi/2)), 15, chop=True) == \
        NS('pi/4*log(4*pi**3/gamma(1/4)**4)', 15)
    # 针对积分端点可能导致积分精度问题（积分点的四舍五入误差 -> 复数对数）

    # 确保对于给定区间 [-pi, pi]，log(2*cos(x/2)) 的积分结果正确至小数点后17位，修剪复数部分
    assert NS(
        2 + Integral(log(2*cos(x/2)), (x, -pi, pi)), 17, chop=True) == NS(2, 17)
    # 确保至小数点后20位的积分结果正确
    assert NS(
        2 + Integral(log(2*cos(x/2)), (x, -pi, pi)), 20, chop=True) == NS(2, 20)
    # 确保至小数点后22位的积分结果正确
    assert NS(
        2 + Integral(log(2*cos(x/2)), (x, -pi, pi)), 22, chop=True) == NS(2, 22)
    # 需要处理积分中的零情况

    # 确保对于给定区间 [0, 1]，sqrt(1-x**2) 的积分结果与 pi 的差正确至小数点后15位，最大迭代次数为30，修剪复数部分
    assert NS(pi - 4*Integral(
        'sqrt(1-x**2)', (x, 0, 1)), 15, maxn=30, chop=True) in ('0.0', '0')
    # 需要处理振荡积分

    # 计算 sin(x)/x**2 在区间 [1, oo) 上的积分并确保结果在 0.49 和 0.51 之间
    a = Integral(sin(x)/x**2, (x, 1, oo)).evalf(maxn=15)
    assert 0.49 < a < 0.51
    # 确保振荡积分 sin(x)/x**2 的结果正确至小数点后15位
    assert NS(
        Integral(sin(x)/x**2, (x, 1, oo)), quad='osc') == '0.504067061906928'

    # 确保对于给定区间 (-oo, -1]，cos(pi*x + 1)/x 的积分结果正确至小数点后15位
    assert NS(Integral(
        cos(pi*x + 1)/x, (x, -oo, -1)), quad='osc') == '0.276374705640365'

    # 对于不定积分，应该不进行数值计算，保持形式
    assert NS(Integral(x, x)) == 'Integral(x, x)'
    assert NS(Integral(x, (x, y))) == 'Integral(x, (x, y))'
def test_evalf_issue_939():
    # https://github.com/sympy/sympy/issues/4038

    # 对于积分的输出形式，可能会因版本更新而产生步函数的差异，
    # 使得这个测试有点无用。目前将所有这些值用于测试，但未来可能需要重新考虑。
    assert NS(integrate(1/(x**5 + 1), x).subs(x, 4), chop=True) in \
        ['-0.000976138910649103', '0.965906660135753', '1.93278945918216']

    # 求解定积分并数值化，比较结果字符串
    assert NS(Integral(1/(x**5 + 1), (x, 2, 4))) == '0.0144361088886740'
    
    # 求解定积分并数值化，比较结果字符串，应用数值截断
    assert NS(integrate(1/(x**5 + 1), (x, 2, 4)), chop=True) == '0.0144361088886740'


def test_double_previously_failing_integrals():
    # Double integrals not implemented <- Sure it is!
    # 计算双重积分
    res = integrate(sqrt(x) + x*y, (x, 1, 2), (y, -1, 1))
    
    # 旧的数值测试，比较结果字符串
    assert NS(res, 15) == '2.43790283299492'
    
    # 符号计算测试，比较结果表达式
    assert res == Rational(-4, 3) + 8*sqrt(2)/3
    
    # 双重积分加零检测
    assert integrate(sin(x + x*y), (x, -1, 1), (y, -1, 1)) is S.Zero


def test_integrate_SingularityFunction():
    in_1 = SingularityFunction(x, a, 3) + SingularityFunction(x, 5, -1)
    out_1 = SingularityFunction(x, a, 4)/4 + SingularityFunction(x, 5, 0)
    # 求解积分，比较结果
    assert integrate(in_1, x) == out_1

    in_2 = 10*SingularityFunction(x, 4, 0) - 5*SingularityFunction(x, -6, -2)
    out_2 = 10*SingularityFunction(x, 4, 1) - 5*SingularityFunction(x, -6, -1)
    # 求解积分，比较结果
    assert integrate(in_2, x) == out_2

    in_3 = 2*x**2*y -10*SingularityFunction(x, -4, 7) - 2*SingularityFunction(y, 10, -2)
    out_3_1 = 2*x**3*y/3 - 2*x*SingularityFunction(y, 10, -2) - 5*SingularityFunction(x, -4, 8)/4
    out_3_2 = x**2*y**2 - 10*y*SingularityFunction(x, -4, 7) - 2*SingularityFunction(y, 10, -1)
    # 分别对 x 和 y 求解积分，比较结果
    assert integrate(in_3, x) == out_3_1
    assert integrate(in_3, y) == out_3_2

    # 检查积分是否未改变
    assert unchanged(Integral, in_3, (x,))
    assert Integral(in_3, x) == Integral(in_3, (x,))
    assert Integral(in_3, x).doit() == out_3_1

    in_4 = 10*SingularityFunction(x, -4, 7) - 2*SingularityFunction(x, 10, -2)
    out_4 = 5*SingularityFunction(x, -4, 8)/4 - 2*SingularityFunction(x, 10, -1)
    # 求解定积分，并比较结果
    assert integrate(in_4, (x, -oo, x)) == out_4

    # 求解积分，比较结果
    assert integrate(SingularityFunction(x, 5, -1), x) == SingularityFunction(x, 5, 0)
    assert integrate(SingularityFunction(x, 0, -1), (x, -oo, oo)) == 1
    assert integrate(5*SingularityFunction(x, 5, -1), (x, -oo, oo)) == 5
    assert integrate(SingularityFunction(x, 5, -1) * f(x), (x, -oo, oo)) == f(5)


def test_integrate_DiracDelta():
    # This is here to check that deltaintegrate is being called, but also
    # to test definite integrals. More tests are in test_deltafunctions.py
    # 求解带有 DiracDelta 的积分，比较结果
    assert integrate(DiracDelta(x) * f(x), (x, -oo, oo)) == f(0)
    assert integrate(DiracDelta(x)**2, (x, -oo, oo)) == DiracDelta(0)
    # issue 4522
    assert integrate(integrate((4 - 4*x + x*y - 4*y) * \
        DiracDelta(x)*DiracDelta(y - 1), (x, 0, 1)), (y, 0, 1)) == 0
    # 定义概率密度函数 p(x, y) = exp(-(x**2 + y**2)) / pi，描述二维高斯分布的概率密度
    p = exp(-(x**2 + y**2))/pi
    # 断言四个积分结果相等，验证关于 DiracDelta 的积分性质
    assert integrate(p*DiracDelta(x - 10*y), (x, -oo, oo), (y, -oo, oo)) == \
           integrate(p*DiracDelta(x - 10*y), (y, -oo, oo), (x, -oo, oo)) == \
           integrate(p*DiracDelta(10*x - y), (x, -oo, oo), (y, -oo, oo)) == \
           integrate(p*DiracDelta(10*x - y), (y, -oo, oo), (x, -oo, oo)) == \
           1/sqrt(101*pi)
# 定义一个测试函数，用于测试积分函数的返回值是否符合预期
def test_integrate_returns_piecewise():
    # 断言积分 x**y 关于 x 的结果是否为 Piecewise 对象
    assert integrate(x**y, x) == Piecewise(
        (x**(y + 1)/(y + 1), Ne(y, -1)), (log(x), True))
    # 断言积分 x**y 关于 y 的结果是否为 Piecewise 对象
    assert integrate(x**y, y) == Piecewise(
        (x**y/log(x), Ne(log(x), 0)), (y, True))
    # 断言积分 exp(n*x) 关于 x 的结果是否为 Piecewise 对象
    assert integrate(exp(n*x), x) == Piecewise(
        (exp(n*x)/n, Ne(n, 0)), (x, True))
    # 断言积分 x*exp(n*x) 关于 x 的结果是否为 Piecewise 对象
    assert integrate(x*exp(n*x), x) == Piecewise(
        ((n*x - 1)*exp(n*x)/n**2, Ne(n**2, 0)), (x**2/2, True))
    # 断言积分 x**(n*y) 关于 x 的结果是否为 Piecewise 对象
    assert integrate(x**(n*y), x) == Piecewise(
        (x**(n*y + 1)/(n*y + 1), Ne(n*y, -1)), (log(x), True))
    # 断言积分 x**(n*y) 关于 y 的结果是否为 Piecewise 对象
    assert integrate(x**(n*y), y) == Piecewise(
        (x**(n*y)/(n*log(x)), Ne(n*log(x), 0)), (y, True))
    # 断言积分 cos(n*x) 关于 x 的结果是否为 Piecewise 对象
    assert integrate(cos(n*x), x) == Piecewise(
        (sin(n*x)/n, Ne(n, 0)), (x, True))
    # 断言积分 cos(n*x)**2 关于 x 的结果是否为 Piecewise 对象
    assert integrate(cos(n*x)**2, x) == Piecewise(
        ((n*x/2 + sin(n*x)*cos(n*x)/2)/n, Ne(n, 0)), (x, True))
    # 断言积分 x*cos(n*x) 关于 x 的结果是否为 Piecewise 对象
    assert integrate(x*cos(n*x), x) == Piecewise(
        (x*sin(n*x)/n + cos(n*x)/n**2, Ne(n, 0)), (x**2/2, True))
    # 断言积分 sin(n*x) 关于 x 的结果是否为 Piecewise 对象
    assert integrate(sin(n*x), x) == Piecewise(
        (-cos(n*x)/n, Ne(n, 0)), (0, True))
    # 断言积分 sin(n*x)**2 关于 x 的结果是否为 Piecewise 对象
    assert integrate(sin(n*x)**2, x) == Piecewise(
        ((n*x/2 - sin(n*x)*cos(n*x)/2)/n, Ne(n, 0)), (0, True))
    # 断言积分 x*sin(n*x) 关于 x 的结果是否为 Piecewise 对象
    assert integrate(x*sin(n*x), x) == Piecewise(
        (-x*cos(n*x)/n + sin(n*x)/n**2, Ne(n, 0)), (0, True))
    # 断言积分 exp(x*y) 关于 x 在区间 [0, z] 的结果是否为 Piecewise 对象
    assert integrate(exp(x*y), (x, 0, z)) == Piecewise(
        (exp(y*z)/y - 1/y, (y > -oo) & (y < oo) & Ne(y, 0)), (z, True))
    # 断言积分 exp(t)*exp(-t*sqrt(x - y)) 关于 t 的结果是否为 Piecewise 对象
    assert integrate(exp(t)*exp(-t*sqrt(x - y)), t) == Piecewise(
        (-exp(t)/(sqrt(x - y)*exp(t*sqrt(x - y)) - exp(t*sqrt(x - y))),
        Ne(x, y + 1)), (t, True))


# 定义一个测试函数，用于测试积分函数在极值和最小值情况下的表现
def test_integrate_max_min():
    # 定义符号 x 为实数
    x = symbols('x', real=True)
    # 断言积分 Min(x, 2) 在区间 [0, 3] 的结果是否为 4
    assert integrate(Min(x, 2), (x, 0, 3)) == 4
    # 断言积分 Max(x**2, x**3) 在区间 [0, 2] 的结果是否为 49/12
    assert integrate(Max(x**2, x**3), (x, 0, 2)) == Rational(49, 12)
    # 断言积分 Min(exp(x), exp(-x))**2 关于 x 的结果是否为 Piecewise 对象
    assert integrate(Min(exp(x), exp(-x))**2, x) == Piecewise( \
        (exp(2*x)/2, x <= 0), (1 - exp(-2*x)/2, True))
    # 定义符号 c 为扩展实数
    c = symbols('c', extended_real=True)
    # 计算积分 Max(c, x)*exp(-x**2) 在整个实数域上的积分结果
    int1 = integrate(Max(c, x)*exp(-x**2), (x, -oo, oo))
    # 计算积分 c*exp(-x**2) 在区间 [-oo, c] 的积分结果
    int2 = integrate(c*exp(-x**2), (x, -oo, c))
    # 计算积分 x*exp(-x**2) 在区间 [c, oo] 的积分结果
    int3 = integrate(x*exp(-x**2), (x, c, oo))
    # 断言 int1 等于 int2 加 int3，且都等于 sqrt(pi)*c*erf(c)/2 + sqrt(pi)*c/2 + exp(-c**2)/2
    assert int1 == int2 + int3 == sqrt(pi)*c*erf(c)/2 + \
        sqrt(pi)*c/2 + exp(-c**2)/2


# 定义一个测试函数，用于测试绝对值函数在不同区间内的积分结果
def test_integrate_Abs_sign():
    # 断言积分 Abs(x) 在区间 [-2, 1] 的结果是否为 5/2
    assert integrate(Abs(x), (x, -2, 1)) == Rational(5, 2)
    # 断言积分 Abs(x) 在区间 [0, 1] 的结果是否为 1/2
    assert integrate(Abs(x), (x, 0, 1)) == S.Half
    # 断言积分 Abs(x + 1) 在区间 [0, 1] 的结果是否为 3/2
    assert integrate(Abs(x + 1), (x, 0, 1)) == Rational(3, 2)
    # 断言积分 Abs(x**2 - 1) 在区间 [-2, 2] 的结果是否为 4
    assert integrate(Abs(x**2 - 1), (x, -2, 2)) == 4
    # 断言积分 Abs(x**2 - 3*x) 在区间 [-15, 15] 的结果是否为 2259
    assert integrate(Abs(x**2 - 3*x), (x, -15, 15)) == 2259
    # 断言积分 sign(x) 在区间 [-1, 2] 的结果是否为 1
    assert integrate(sign(x), (x, -1, 2)) == 1
    # 断言积分 sign(x)*sin(x) 在区间 [-pi, pi] 的结果是否为 4
    assert integrate(sign(x)*sin(x), (x, -pi, pi)) == 4
    # 断言积分 sign(x - 2) * x**2 在区间 [0, 3] 的结果是否为 11/3
    assert integrate(sign(x - 2) * x**2, (x, 0, 3)) == Rational(11, 3)

    # 定义符号 t, s 为实数
    t, s = symbols('t s', real=True)
    # 断言积分 Abs(t) 关于 t 的结果是否为 Piecewise 对象
    assert integrate(Abs
    # 断言：计算绝对值函数 abs(t - s**2) 在区间 [0, 2] 上的积分是否等于指定的表达式
    assert (integrate(abs(t - s**2), (t, 0, 2)) ==
        2*s**2*Min(2, s**2) - 2*s**2 - Min(2, s**2)**2 + 2)
    
    # 断言：计算指数函数 exp(-Abs(t)) 关于变量 t 的积分是否等于 Piecewise 函数
    assert integrate(exp(-Abs(t)), t) == Piecewise(
        (exp(t), t <= 0), (2 - exp(-t), True))
    
    # 断言：计算符号函数 sign(2*t - 6) 关于变量 t 的积分是否等于 Piecewise 函数
    assert integrate(sign(2*t - 6), t) == Piecewise(
        (-t, t < 3), (t - 6, True))
    
    # 断言：计算函数 2*t*sign(t**2 - 1) 关于变量 t 的积分是否等于 Piecewise 函数
    assert integrate(2*t*sign(t**2 - 1), t) == Piecewise(
        (t**2, t < -1), (-t**2 + 2, t < 1), (t**2, True))
    
    # 断言：计算符号函数 sign(t) 在区间 (s + 1) 上的积分是否等于 Piecewise 函数
    assert integrate(sign(t), (t, s + 1)) == Piecewise(
        (s + 1, s + 1 > 0), (-s - 1, s + 1 < 0), (0, True))
def test_subs1():
    # 创建积分表达式 e = ∫exp(x - y) dx
    e = Integral(exp(x - y), x)
    # 断言替换 y 为 3 后的结果为 ∫exp(x - 3) dx
    assert e.subs(y, 3) == Integral(exp(x - 3), x)
    
    # 创建积分表达式 e = ∫exp(x - y) dx 从 0 到 1
    e = Integral(exp(x - y), (x, 0, 1))
    # 断言替换 y 为 3 后的结果为 ∫exp(x - 3) dx 从 0 到 1
    assert e.subs(y, 3) == Integral(exp(x - 3), (x, 0, 1))
    
    # 创建 Lambda 函数 f(x) = exp(-x^2)
    f = Lambda(x, exp(-x**2))
    # 创建卷积积分 conv = ∫f(x - y)*f(y) dy 从 -∞ 到 ∞
    conv = Integral(f(x - y)*f(y), (y, -oo, oo))
    # 断言替换 x 为 0 后的结果为 ∫exp(-2*y^2) dy 从 -∞ 到 ∞
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo))


def test_subs2():
    # 创建积分表达式 e = ∫exp(x - y) dx 从 x 到 t
    e = Integral(exp(x - y), x, t)
    # 断言替换 y 为 3 后的结果为 ∫exp(x - 3) dx 从 x 到 t
    assert e.subs(y, 3) == Integral(exp(x - 3), x, t)
    
    # 创建积分表达式 e = ∫exp(x - y) dx 从 0 到 1，从 0 到 1
    e = Integral(exp(x - y), (x, 0, 1), (t, 0, 1))
    # 断言替换 y 为 3 后的结果为 ∫exp(x - 3) dx 从 0 到 1，从 0 到 1
    assert e.subs(y, 3) == Integral(exp(x - 3), (x, 0, 1), (t, 0, 1))
    
    # 创建 Lambda 函数 f(x) = exp(-x^2)
    f = Lambda(x, exp(-x**2))
    # 创建卷积积分 conv = ∫f(x - y)*f(y) dy 从 -∞ 到 ∞，从 0 到 1
    conv = Integral(f(x - y)*f(y), (y, -oo, oo), (t, 0, 1))
    # 断言替换 x 为 0 后的结果为 ∫exp(-2*y^2) dy 从 -∞ 到 ∞，从 0 到 1
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo), (t, 0, 1))


def test_subs3():
    # 创建积分表达式 e = ∫exp(x - y) dx 从 0 到 y，从 y 到 1
    e = Integral(exp(x - y), (x, 0, y), (t, y, 1))
    # 断言替换 y 为 3 后的结果为 ∫exp(x - 3) dx 从 0 到 3，从 3 到 1
    assert e.subs(y, 3) == Integral(exp(x - 3), (x, 0, 3), (t, 3, 1))
    
    # 创建 Lambda 函数 f(x) = exp(-x^2)
    f = Lambda(x, exp(-x**2))
    # 创建卷积积分 conv = ∫f(x - y)*f(y) dy 从 -∞ 到 ∞，从 x 到 1
    conv = Integral(f(x - y)*f(y), (y, -oo, oo), (t, x, 1))
    # 断言替换 x 为 0 后的结果为 ∫exp(-2*y^2) dy 从 -∞ 到 ∞，从 0 到 1
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo), (t, 0, 1))


def test_subs4():
    # 创建积分表达式 e = ∫exp(x) dx 从 0 到 y，从 y 到 1
    e = Integral(exp(x), (x, 0, y), (t, y, 1))
    # 断言替换 y 为 3 后的结果为 ∫exp(x) dx 从 0 到 3，从 3 到 1
    assert e.subs(y, 3) == Integral(exp(x), (x, 0, 3), (t, 3, 1))
    
    # 创建 Lambda 函数 f(x) = exp(-x^2)
    f = Lambda(x, exp(-x**2))
    # 创建卷积积分 conv = ∫f(y)*f(y) dy 从 -∞ 到 ∞，从 x 到 1
    conv = Integral(f(y)*f(y), (y, -oo, oo), (t, x, 1))
    # 断言替换 x 为 0 后的结果为 ∫exp(-2*y^2) dy 从 -∞ 到 ∞，从 0 到 1
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo), (t, 0, 1))


def test_subs5():
    # 创建积分表达式 e = ∫exp(-x^2) dx 从 -∞ 到 ∞
    e = Integral(exp(-x**2), (x, -oo, oo))
    # 断言替换 x 为 5 后的结果与原积分表达式相同
    assert e.subs(x, 5) == e
    
    # 创建积分表达式 e = ∫exp(-x^2 + y) dx
    e = Integral(exp(-x**2 + y), x)
    # 断言替换 y 为 5 后的结果为 ∫exp(-x^2 + 5) dx
    assert e.subs(y, 5) == Integral(exp(-x**2 + 5), x)
    
    # 创建积分表达式 e = ∫exp(-x^2 + y) dx 从 x 到 x
    e = Integral(exp(-x**2 + y), (x, x))
    # 断言替换 x 为 5 后的结果为 ∫exp(y - x^2) dx 从 5 到 x
    assert e.subs(x, 5) == Integral(exp(y - x**2), (x, 5))
    # 断言替换 y 为 5 后的结果为 ∫exp(-x^2 + 5) dx
    assert e.subs(y, 5) == Integral(exp(-x**2 + 5), x)
    
    # 创建积分表达式 e = ∫exp(-x^2 + y) dx 从 y 到 x，从 -∞ 到 ∞
    e = Integral(exp(-x**2 + y), (y, -oo, oo), (x, -oo, oo))
    # 断言替换 x 为 5 后的结果与原积分表达式相同
    assert e.subs(x, 5) == e
    # 断言替换 y 为 5 后的结果与原积分表达式相同
    assert e.subs(y, 5) == e
    
    # 测试反导数的评估
    # 创建积分表达式 e = ∫exp(-x^2) dx 从 x 到 x
    e = Integral(exp(-x**2), (x, x))
    # 断言替换 x 为 5 后的结果为 ∫exp(-x^2) dx 从 5 到 x
    assert e.subs(x, 5) == Integral(exp(-x**2), (x, 5))
    
    # 创建积分表达式 e = ∫exp(x) dx
    e = Integral(exp(x), x)
    # 断言结果
    # 创建一个积分对象 e，积分的被积函数是 f(x) + f(x**2)，积分变量是 x，积分区间是从 1 到 y
    e = Integral(f(x) + f(x**2), (x, 1, y))
    
    # 断言：对积分对象 e 调用 expand() 方法后的结果应当等于将被积函数拆分成两个积分的和
    assert e.expand() == Integral(f(x), (x, 1, y)) + Integral(f(x**2), (x, 1, y))
    
    # 创建一个积分对象 e，积分的被积函数是 f(x) + f(x**2)，积分变量是 x，积分区间是从 1 到正无穷
    e = Integral(f(x) + f(x**2), (x, 1, oo))
    
    # 断言：对积分对象 e 调用 expand() 方法后的结果应当仍然等于原始的积分对象 e，因为在积分区间为正无穷时无法展开
    assert e.expand() == e
    
    # 断言：对积分对象 e 调用 expand(force=True) 方法后的结果应当等于将被积函数拆分成两个积分的和
    assert e.expand(force=True) == Integral(f(x), (x, 1, oo)) + \
           Integral(f(x**2), (x, 1, oo))
# 定义一个测试函数，用于集成测试变量和函数的正确性
def test_integration_variable():
    # 断言：对于指数为负数的变量 x，Integral 抛出 ValueError 异常
    raises(ValueError, lambda: Integral(exp(-x**2), 3))
    # 断言：对于变量 x 在 (-oo, oo) 范围内的 Integral，抛出 ValueError 异常
    raises(ValueError, lambda: Integral(exp(-x**2), (3, -oo, oo)))


# 定义一个测试函数，用于扩展 Integral 对象的展开操作的正确性
def test_expand_integral():
    # 断言：展开含有 cos(x**2)*(sin(x**2) + 1) 的 Integral 对象在 (x, 0, 1) 范围内的结果
    assert Integral(cos(x**2)*(sin(x**2) + 1), (x, 0, 1)).expand() == \
        Integral(cos(x**2)*sin(x**2), (x, 0, 1)) + \
        Integral(cos(x**2), (x, 0, 1))
    # 断言：展开含有 cos(x**2)*(sin(x**2) + 1) 的 Integral 对象关于 x 的结果
    assert Integral(cos(x**2)*(sin(x**2) + 1), x).expand() == \
        Integral(cos(x**2)*sin(x**2), x) + \
        Integral(cos(x**2), x)


# 定义一个测试函数，用于测试使用中点法求和的 Integral 对象的正确性
def test_as_sum_midpoint1():
    # 创建一个 Integral 对象 e，表示 sqrt(x**3 + 1) 在 (x, 2, 10) 范围内的积分
    e = Integral(sqrt(x**3 + 1), (x, 2, 10))
    # 断言：使用中点法将 Integral 对象 e 分成 1 个子区间的结果
    assert e.as_sum(1, method="midpoint") == 8*sqrt(217)
    # 断言：使用中点法将 Integral 对象 e 分成 2 个子区间的结果
    assert e.as_sum(2, method="midpoint") == 4*sqrt(65) + 12*sqrt(57)
    # 断言：使用中点法将 Integral 对象 e 分成 3 个子区间的结果
    assert e.as_sum(3, method="midpoint") == 8*sqrt(217)/3 + \
        8*sqrt(3081)/27 + 8*sqrt(52809)/27
    # 断言：使用中点法将 Integral 对象 e 分成 4 个子区间的结果
    assert e.as_sum(4, method="midpoint") == 2*sqrt(730) + \
        4*sqrt(7) + 4*sqrt(86) + 6*sqrt(14)
    # 断言：使用中点法对 Integral 对象 e 分成 4 个子区间后，其数值结果与积分值的差的绝对值小于 0.5
    assert abs(e.as_sum(4, method="midpoint").n() - e.n()) < 0.5

    # 创建一个 Integral 对象 e，表示 sqrt(x**3 + y**3) 在 (x, 2, 10) 和 (y, 0, 10) 范围内的积分
    e = Integral(sqrt(x**3 + y**3), (x, 2, 10), (y, 0, 10))
    # 断言：尝试使用中点法对 e 进行 4 个子区间的求和，预期抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: e.as_sum(4))


# 定义一个测试函数，用于测试使用中点法求和的 Integral 对象的正确性（另一种表达方式）
def test_as_sum_midpoint2():
    # 创建一个 Integral 对象 e，表示 (x + y)**2 在 (x, 0, 1) 范围内的积分
    e = Integral((x + y)**2, (x, 0, 1))
    # 定义一个正整数符号变量 n
    n = Symbol('n', positive=True, integer=True)
    # 断言：使用中点法将 Integral 对象 e 分成 1 个子区间的展开结果
    assert e.as_sum(1, method="midpoint").expand() == Rational(1, 4) + y + y**2
    # 断言：使用中点法将 Integral 对象 e 分成 2 个子区间的展开结果
    assert e.as_sum(2, method="midpoint").expand() == Rational(5, 16) + y + y**2
    # 断言：使用中点法将 Integral 对象 e 分成 3 个子区间的展开结果
    assert e.as_sum(3, method="midpoint").expand() == Rational(35, 108) + y + y**2
    # 断言：使用中点法将 Integral 对象 e 分成 4 个子区间的展开结果
    assert e.as_sum(4, method="midpoint").expand() == Rational(21, 64) + y + y**2
    # 断言：使用中点法将 Integral 对象 e 分成 n 个子区间的展开结果
    assert e.as_sum(n, method="midpoint").expand() == \
        y**2 + y + Rational(1, 3) - 1/(12*n**2)


# 定义一个测试函数，用于测试使用左端点法求和的 Integral 对象的正确性
def test_as_sum_left():
    # 创建一个 Integral 对象 e，表示 (x + y)**2 在 (x, 0, 1) 范围内的积分
    e = Integral((x + y)**2, (x, 0, 1))
    # 断言：使用左端点法将 Integral 对象 e 分成 1 个子区间的展开结果
    assert e.as_sum(1, method="left").expand() == y**2
    # 断言：使用左端点法将 Integral 对象 e 分成 2 个子区间的展开结果
    assert e.as_sum(2, method="left").expand() == Rational(1, 8) + y/2 + y**2
    # 断言：使用左端点法将 Integral 对象 e 分成 3 个子区间的展开结果
    assert e.as_sum(3, method="left").expand() == Rational(5, 27) + y*Rational(2, 3) + y**2
    # 断言：使用左端点法将 Integral 对象 e 分成 4 个子区间的展开结果
    assert e.as_sum(4, method="left").expand() == Rational(7, 32) + y*Rational(3, 4) + y**2
    # 断言：使用左端点法将 Integral 对象 e 分成 n 个子区间的展开结果
    assert e.as_sum(n, method="left").expand() == \
        y**2 + y + Rational(1, 3) - y/n - 1/(2*n) + 1/(6*n**2)
    # 断言：使用左端点法将 Integral 对象 e 分成 10 个子区间，不进行求值，而是保留为 Sum 对象
    assert e.as_sum(10, method="left", evaluate=False).has(Sum)


# 定义一个测试函数，用于测试使用右端点法求和的 Integral 对象的正确性
def test_as_sum_right():
    # 创建一个 Integral 对象 e，表示 (x + y)**2 在 (x, 0, 1) 范围内的积分
    e = Integral((x + y)**2, (x, 0, 1))
    # 断言：使用右端点法将 Integral 对象 e 分成 1 个子区间的展开结果
    assert e.as_sum(1, method="right").expand() == 1 + 2*y + y**2
    # 断言：使用右端点法将 Integral 对象 e 分成 2 个子区间的展开结果
    assert e.as_sum(2, method="right").expand() == Rational(5, 8) + y*Rational(3, 2) + y**2
    # 断言：使用右端点法将 Integral 对象 e 分成 3 个子区间的展开
    # 断言，验证 e 的积分和为给定的表达式，使用梯形法计算
    assert e.as_sum(3, method="trapezoid").expand() == y**2 + y + Rational(19, 54)
    
    # 断言，验证 e 的积分和为给定的表达式，使用梯形法计算
    assert e.as_sum(4, method="trapezoid").expand() == y**2 + y + Rational(11, 32)
    
    # 断言，验证 e 的积分和为给定的表达式，使用梯形法计算
    assert e.as_sum(n, method="trapezoid").expand() == \
        y**2 + y + Rational(1, 3) + 1/(6*n**2)
    
    # 断言，验证在区间 [0, 1] 上的符号函数积分，使用梯形法计算，预期结果为 S.Half
    assert Integral(sign(x), (x, 0, 1)).as_sum(1, 'trapezoid') == S.Half
# 定义一个测试函数，用于测试 Integral 类的各种情况下的异常情况
def test_as_sum_raises():
    # 创建一个积分对象 e，表示 (x + y)**2 在 x 范围 [0, 1] 的积分
    e = Integral((x + y)**2, (x, 0, 1))
    # 检查 e.as_sum(-1) 是否会引发 ValueError 异常
    raises(ValueError, lambda: e.as_sum(-1))
    # 检查 e.as_sum(0) 是否会引发 ValueError 异常
    raises(ValueError, lambda: e.as_sum(0))
    # 检查 Integral(x).as_sum(3) 是否会引发 ValueError 异常
    raises(ValueError, lambda: Integral(x).as_sum(3))
    # 检查 e.as_sum(oo) 是否会引发 ValueError 异常
    raises(ValueError, lambda: e.as_sum(oo))
    # 检查 e.as_sum(3, method='xxxx2') 是否会引发 ValueError 异常
    raises(ValueError, lambda: e.as_sum(3, method='xxxx2'))


# 定义一个测试函数，测试嵌套积分的 doit 方法是否正确
def test_nested_doit():
    # 创建两个积分对象 e 和 f
    e = Integral(Integral(x, x), x)
    f = Integral(x, x, x)
    # 断言 e.doit() 的结果与 f.doit() 的结果相等
    assert e.doit() == f.doit()


# 定义一个测试函数，测试 Issue 4665 的情况
def test_issue_4665():
    # 创建两个积分对象 e 和 f
    e = Integral(x**2, (x, None, 1))
    f = Integral(x**2, (x, 1, None))
    # 断言 e.doit() 的结果是否等于有理数 1/3
    assert e.doit() == Rational(1, 3)
    # 断言 f.doit() 的结果是否等于有理数 -1/3
    assert f.doit() == Rational(-1, 3)
    # 测试积分对象中下限和上限的替换是否正确
    assert Integral(x*y, (x, None, y)).subs(y, t) == Integral(x*t, (x, None, t))
    assert Integral(x*y, (x, y, None)).subs(y, t) == Integral(x*t, (x, t, None))
    # 断言积分计算结果是否正确
    assert integrate(x**2, (x, None, 1)) == Rational(1, 3)
    assert integrate(x**2, (x, 1, None)) == Rational(-1, 3)
    assert integrate("x**2", ("x", "1", None)) == Rational(-1, 3)


# 定义一个测试函数，测试积分对象的重建是否正确
def test_integral_reconstruct():
    # 创建一个积分对象 e
    e = Integral(x**2, (x, -1, 1))
    # 断言 e 与由其自身的参数重建的积分对象相等
    assert e == Integral(*e.args)


# 定义一个测试函数，测试积分对象在不同情况下的 doit 方法是否正确
def test_doit_integrals():
    # 创建一个积分对象 e
    e = Integral(Integral(2*x), (x, 0, 1))
    # 断言 e.doit() 的结果是否等于有理数 1/3
    assert e.doit() == Rational(1, 3)
    # 断言使用 deep=False 参数时的计算结果是否等于有理数 1/3
    assert e.doit(deep=False) == Rational(1, 3)
    # 创建一个函数对象 f
    f = Function('f')
    # 测试不能执行积分的情况，期望结果为 0
    assert Integral(f(x), (x, 1, 1)).doit() == 0
    # 测试无法评估极限的情况，期望结果为 0
    assert Integral(0, (x, 1, Integral(f(x), x))).doit() == 0
    assert Integral(x, (a, 0)).doit() == 0
    # 创建一个限制元组 limits
    limits = ((a, 1, exp(x)), (x, 0))
    # 断言在给定的限制下积分计算结果是否等于有理数 1/4
    assert Integral(a, *limits).doit() == Rational(1, 4)
    # 断言在 limits 的反转顺序下积分计算结果是否等于 0
    assert Integral(a, *list(reversed(limits))).doit() == 0


# 定义一个测试函数，测试 Issue 4884 的情况
def test_issue_4884():
    # 断言对 sqrt(x)*(1 + x) 的积分计算结果是否符合 Piecewise 表达式
    assert integrate(sqrt(x)*(1 + x)) == \
        Piecewise(
            (2*sqrt(x)*(x + 1)**2/5 - 2*sqrt(x)*(x + 1)/15 - 4*sqrt(x)/15,
            Abs(x + 1) > 1),
            (2*I*sqrt(-x)*(x + 1)**2/5 - 2*I*sqrt(-x)*(x + 1)/15 -
            4*I*sqrt(-x)/15, True))
    # 断言对 x**x*(1 + log(x)) 的积分计算结果是否等于 x**x
    assert integrate(x**x*(1 + log(x))) == x**x


# 定义一个测试函数，测试 Issue 18153 的情况
def test_issue_18153():
    # 断言对 x**n*log(x) 的积分计算结果是否符合 Piecewise 表达式
    assert integrate(x**n*log(x),x) == \
    Piecewise(
        (n*x*x**n*log(x)/(n**2 + 2*n + 1) +
    x*x**n*log(x)/(n**2 + 2*n + 1) - x*x**n/(n**2 + 2*n + 1)
    , Ne(n, -1)), (log(x)**2/2, True)
    )


# 定义一个测试函数，测试积分对象的 is_number 属性
def test_is_number():
    from sympy.abc import x, y, z
    # 断言 Integral(x).is_number 的值为 False
    assert Integral(x).is_number is False
    # 断言 Integral(1, x).is_number 的值为 False
    assert Integral(1, x).is_number is False
    # 断言 Integral(1, (x, 1)).is_number 的值为 True
    assert Integral(1, (x, 1)).is_number is True
    # 断言 Integral(1, (x, 1, 2)).is_number 的值为 True
    assert Integral(1, (x, 1, 2)).is_number is True
    # 断言 Integral(1, (x, 1, y)).is_number 的值为 False
    assert Integral(1, (x, 1, y)).is_number is False
    # 断言 Integral(1, (x, y)).is_number 的值为 False
    assert Integral(1, (x, y)).is_number is False
    # 断言 Integral(x, y).is_number 的值为 False
    assert Integral(x, y).is_number is False
    # 断言 Integral(x, (y, 1, x)).is_number 的值为 False
    assert Integral(x, (y, 1, x)).is_number is False
    # 断言 Integral(x, (y, 1, 2)).is_number 的值为 False
    assert Integral(x, (y, 1, 2)).is_number is False
    # 断言 Integral(x, (x, 1, 2)).is_number 的值为 True
    assert Integral(x, (x, 1, 2)).is_number is True
    # `foo.is_number` 应始终等价于 `not foo.free_symbols`
    # 在每种情况下，都有伪自由符号
    i = Integral(x, (y, 1, 1))
    # 断言 i 的 is_number 属性为 False，并且调用 n() 方法返回 0
    assert i.is_number is False and i.n() == 0
    # 创建一个定积分对象 Integral，积分变量为 x，积分区间为 (y, z, z)
    i = Integral(x, (y, z, z))
    # 再次断言 i 的 is_number 属性为 False，并且调用 n() 方法返回 0
    assert i.is_number is False and i.n() == 0
    # 创建一个定积分对象 Integral，积分函数为常数 1，积分变量为 y，积分区间为 (z, z + 2)
    i = Integral(1, (y, z, z + 2))
    # 再次断言 i 的 is_number 属性为 False，并且调用 n() 方法返回 2.0
    assert i.is_number is False and i.n() == 2.0

    # 断言 Integral(x*y, (x, 1, 2), (y, 1, 3)) 的 is_number 属性为 True
    assert Integral(x*y, (x, 1, 2), (y, 1, 3)).is_number is True
    # 断言 Integral(x*y, (x, 1, 2), (y, 1, z)) 的 is_number 属性为 False
    assert Integral(x*y, (x, 1, 2), (y, 1, z)).is_number is False
    # 断言 Integral(x, (x, 1)) 的 is_number 属性为 True
    assert Integral(x, (x, 1)).is_number is True
    # 断言 Integral(x, (x, 1, Integral(y, (y, 1, 2)))) 的 is_number 属性为 True
    assert Integral(x, (x, 1, Integral(y, (y, 1, 2)))).is_number is True
    # 断言 Integral(Sum(z, (z, 1, 2)), (x, 1, 2)) 的 is_number 属性为 True
    assert Integral(Sum(z, (z, 1, 2)), (x, 1, 2)).is_number is True
    # 注释：如果被积函数实际上是一个未简化的零，is_number 可能会返回 False，但这是 is_number 的一般特性。
    # 断言 Integral(sin(x)**2 + cos(x)**2 - 1, x).is_number 的属性为 False
    assert Integral(sin(x)**2 + cos(x)**2 - 1, x).is_number is False
    # 断言 Integral(f(x), (x, 0, 1)) 的 is_number 属性为 True
    assert Integral(f(x), (x, 0, 1)).is_number is True
# 定义测试函数 test_free_symbols
def test_free_symbols():
    # 从 sympy.abc 模块导入符号 x, y, z
    from sympy.abc import x, y, z
    # 检查积分 Integral(0, x) 的自由符号集合是否为 {x}
    assert Integral(0, x).free_symbols == {x}
    # 检查积分 Integral(x) 的自由符号集合是否为 {x}
    assert Integral(x).free_symbols == {x}
    # 检查积分 Integral(x, (x, None, y)) 的自由符号集合是否为 {y}
    assert Integral(x, (x, None, y)).free_symbols == {y}
    # 检查积分 Integral(x, (x, y, None)) 的自由符号集合是否为 {y}
    assert Integral(x, (x, y, None)).free_symbols == {y}
    # 检查积分 Integral(x, (x, 1, y)) 的自由符号集合是否为 {y}
    assert Integral(x, (x, 1, y)).free_symbols == {y}
    # 检查积分 Integral(x, (x, y, 1)) 的自由符号集合是否为 {y}
    assert Integral(x, (x, y, 1)).free_symbols == {y}
    # 检查积分 Integral(x, (x, x, y)) 的自由符号集合是否为 {x, y}
    assert Integral(x, (x, x, y)).free_symbols == {x, y}
    # 检查积分 Integral(x, x, y) 的自由符号集合是否为 {x, y}
    assert Integral(x, x, y).free_symbols == {x, y}
    # 检查积分 Integral(x, (x, 1, 2)) 的自由符号集合是否为空集
    assert Integral(x, (x, 1, 2)).free_symbols == set()
    # 检查积分 Integral(x, (y, 1, 2)) 的自由符号集合是否为 {x}（伪自由符号情况）
    assert Integral(x, (y, 1, 2)).free_symbols == {x}
    # 检查积分 Integral(x, (y, z, z)) 的自由符号集合是否为 {x, z}
    assert Integral(x, (y, z, z)).free_symbols == {x, z}
    # 检查积分 Integral(x, (y, 1, 2), (y, None, None)) 的自由符号集合是否为 {x, y}
    assert Integral(x, (y, 1, 2), (y, None, None)).free_symbols == {x, y}
    # 检查积分 Integral(x, (y, 1, 2), (x, 1, y)) 的自由符号集合是否为 {y}
    assert Integral(x, (y, 1, 2), (x, 1, y)).free_symbols == {y}
    # 检查积分 Integral(2, (y, 1, 2), (y, 1, x), (x, 1, 2)) 的自由符号集合是否为空集
    assert Integral(2, (y, 1, 2), (y, 1, x), (x, 1, 2)).free_symbols == set()
    # 检查积分 Integral(2, (y, x, 2), (y, 1, x), (x, 1, 2)) 的自由符号集合是否为空集
    assert Integral(2, (y, x, 2), (y, 1, x), (x, 1, 2)).free_symbols == set()
    # 检查积分 Integral(2, (x, 1, 2), (y, x, 2), (y, 1, 2)) 的自由符号集合是否为 {x}
    assert Integral(2, (x, 1, 2), (y, x, 2), (y, 1, 2)).free_symbols == {x}
    # 检查积分 Integral(f(x), (f(x), 1, y)) 的自由符号集合是否为 {y}
    assert Integral(f(x), (f(x), 1, y)).free_symbols == {y}
    # 检查积分 Integral(f(x), (f(x), 1, x)) 的自由符号集合是否为 {x}
    assert Integral(f(x), (f(x), 1, x)).free_symbols == {x}


# 定义测试函数 test_is_zero
def test_is_zero():
    # 从 sympy.abc 模块导入符号 x, m
    from sympy.abc import x, m
    # 检查积分 Integral(0, (x, 1, x)) 是否为零
    assert Integral(0, (x, 1, x)).is_zero
    # 检查积分 Integral(1, (x, 1, 1)) 是否为零
    assert Integral(1, (x, 1, 1)).is_zero
    # 检查积分 Integral(1, (x, 1, 2), (y, 2)) 是否不为零
    assert Integral(1, (x, 1, 2), (y, 2)).is_zero is False
    # 检查积分 Integral(x, (m, 0)) 是否为零
    assert Integral(x, (m, 0)).is_zero
    # 检查积分 Integral(x + m, (m, 0)) 是否为未定
    assert Integral(x + m, (m, 0)).is_zero is None
    # 创建积分对象 i
    i = Integral(m, (m, 1, exp(x)), (x, 0))
    # 检查积分对象 i 是否为未定
    assert i.is_zero is None
    # 检查积分 Integral(m, (x, 0), (m, 1, exp(x))) 是否为零
    assert Integral(m, (x, 0), (m, 1, exp(x))).is_zero is True
    # 检查积分 Integral(x, (x, oo, oo)) 是否为零（问题 8171）
    assert Integral(x, (x, oo, oo)).is_zero
    # 检查积分 Integral(x, (x, -oo, -oo)) 是否为零
    assert Integral(x, (x, -oo, -oo)).is_zero
    # 检查积分 Integral(sin(x), (x, 0, 2*pi)) 是否为未定
    assert Integral(sin(x), (x, 0, 2*pi)).is_zero is None


# 定义测试函数 test_series
def test_series():
    # 从 sympy.abc 模块导入符号 x
    from sympy.abc import x
    # 创建积分对象 i
    i = Integral(cos(x), (x, x))
    # 获取积分对象 i 的级数展开结果 e
    e = i.lseries(x)
    # 检查积分对象 i 在 n=8 的级数展开结果是否移除了高阶无穷小项
    assert i.nseries(x, n=8).removeO() == Add(*[next(e) for j in range(4)])


# 定义测试函数 test_trig_nonelementary_integrals
def test_trig_nonelementary_integrals():
    # 导入符号 x
    x = Symbol('x')
    # 检查 integrate((1 + sin(x))/x, x) 的结果是否为 log(x) + Si(x)
    assert integrate((1 + sin(x))/x, x) == log(x) + Si(x)
    # 检查 integrate((cos(x) + 2)/x, x) 的结果是否包含 Ci 函数
    assert integrate((cos(x) + 2)/x, x).has(Ci)


# 定义测试函数 test_issue_4403
def test_issue_4403():
    # 定义符号 x, y, z，并设定 z 为正数
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z', positive=True)
    # 检查 integrate(sqrt(x**2 + z**2), x) 的结果是否为 z**2*asinh(x/z)/2 + x*sqrt(x**2 + z**2)/2
    assert integrate(sqrt(x**2 + z**2), x) == \
        z**2*
    # 断言语句，用于检查函数 integrate 的结果是否与预期相等
    assert integrate(sqrt(-x**2 - 4), x) == \
        # 计算函数 integrate 的预期结果，这里是负二倍反正切函数和 x 乘以根号下负四减 x 平方的一半
        -2*atan(x/sqrt(-4 - x**2)) + x*sqrt(-4 - x**2)/2
# 定义一个测试函数，用于测试数学符号积分的不同问题
def test_issue_4100():
    # 定义一个正数符号变量 R
    R = Symbol('R', positive=True)
    # 断言：计算半圆弧线段的定积分，验证其结果是否为 pi*R^2/4
    assert integrate(sqrt(R**2 - x**2), (x, 0, R)) == pi*R**2/4


# 定义一个测试函数，用于测试积分相关问题
def test_issue_5167():
    # 从 sympy.abc 模块导入符号变量 w, x, y, z
    from sympy.abc import w, x, y, z
    # 定义一个函数 f
    f = Function('f')
    # 断言：验证双重积分的表达式是否等同于使用两个变量的单重积分
    assert Integral(Integral(f(x), x), x) == Integral(f(x), x, x)
    # 断言：验证积分对象的参数是否符合预期 (f(x), (x,))
    assert Integral(f(x)).args == (f(x), Tuple(x))
    # 断言：验证双重积分对象的参数是否符合预期 (f(x), (x,), (x,))
    assert Integral(Integral(f(x))).args == (f(x), Tuple(x), Tuple(x))
    # 断言：验证双重积分对象的参数是否符合预期 (f(x), (x,), (y,))
    assert Integral(Integral(f(x)), y).args == (f(x), Tuple(x), Tuple(y))
    # 断言：验证双重积分对象的参数是否符合预期 (f(x), (z,), (y,))
    assert Integral(Integral(f(x), z), y).args == (f(x), Tuple(z), Tuple(y))
    # 断言：验证三重积分对象的参数是否符合预期 (f(x), (x,), (y,), (z,))
    assert Integral(Integral(Integral(f(x), x), y), z).args == \
        (f(x), Tuple(x), Tuple(y), Tuple(z))
    # 断言：验证积分函数的计算是否正确，是否等同于双重积分
    assert integrate(Integral(f(x), x), x) == Integral(f(x), x, x)
    # 断言：验证积分函数的计算是否正确，是否等同于 y*单重积分(f(x), x)
    assert integrate(Integral(f(x), y), x) == y*Integral(f(x), x)
    # 断言：验证积分函数的计算结果是否在给定的两个可能值中 (y*单重积分(f(x), x) 或者 y*单重积分(f(x), x))
    assert integrate(Integral(f(x), x), y) in [Integral(y*f(x), x), y*Integral(f(x), x)]
    # 断言：验证积分函数的计算是否正确，是否等同于 x^2
    assert integrate(Integral(2, x), x) == x**2
    # 断言：验证积分函数的计算是否正确，是否等同于 2*x*y
    assert integrate(Integral(2, x), y) == 2*x*y
    # 断言：不要重新排序给定的积分限制
    assert Integral(1, x, y).args != Integral(1, y, x).args
    # 断言：尽可能地进行积分
    assert Integral(f(x), y, x, y, x).doit() == y**2*Integral(f(x), x, x)/2
    # 断言：验证复杂积分的计算是否正确
    assert Integral(f(x), (x, 1, 2), (w, 1, x), (z, 1, y)).doit() == \
        y*(x - 1)*Integral(f(x), (x, 1, 2)) - (x - 1)*Integral(f(x), (x, 1, 2))


# 定义一个测试函数，用于测试指数函数积分的问题
def test_issue_4890():
    # 定义一个正数符号变量 z
    z = Symbol('z', positive=True)
    # 断言：验证指数函数积分的计算是否正确
    assert integrate(exp(-log(x)**2), x) == \
        sqrt(pi)*exp(Rational(1, 4))*erf(log(x) - S.Half)/2
    # 断言：验证指数函数积分的计算是否正确
    assert integrate(exp(log(x)**2), x) == \
        sqrt(pi)*exp(Rational(-1, 4))*erfi(log(x)+S.Half)/2
    # 断言：验证指数函数积分的计算是否正确
    assert integrate(exp(-z*log(x)**2), x) == \
        sqrt(pi)*exp(1/(4*z))*erf(sqrt(z)*log(x) - 1/(2*sqrt(z)))/(2*sqrt(z))


# 定义一个测试函数，用于测试被积函数中是否包含积分的问题
def test_issue_4551():
    # 断言：验证被积函数是否包含积分
    assert not integrate(1/(x*sqrt(1 - x**2)), x).has(Integral)


# 定义一个测试函数，用于测试符号参数的积分问题
def test_issue_4376():
    # 定义一个正整数符号变量 n
    n = Symbol('n', integer=True, positive=True)
    # 断言：验证积分简化后的结果是否为零
    assert simplify(integrate(n*(x**(1/n) - 1), (x, 0, S.Half)) -
                (n**2 - 2**(1/n)*n**2 - n*2**(1/n))/(2**(1 + 1/n) + n*2**(1 + 1/n))) == 0


# 定义一个测试函数，用于测试含有有理数幂次的积分问题
def test_issue_4517():
    # 断言：验证复杂函数的积分计算是否正确
    assert integrate((sqrt(x) - x**3)/x**Rational(1, 3), x) == \
        6*x**Rational(7, 6)/7 - 3*x**Rational(11, 3)/11


# 定义一个测试函数，用于测试三角函数积分的问题
def test_issue_4527():
    # 定义整数符号变量 k, m
    k, m = symbols('k m', integer=True)
    # 断言：验证三角函数积分的计算结果是否能简化为 Piecewise 形式
    assert integrate(sin(k*x)*sin(m*x), (x, 0, pi)).simplify() == \
        Piecewise((0, Eq(k, 0) | Eq(m, 0)),
                  (-pi/2, Eq(k, -m) | (Eq(k, 0) & Eq(m, 0))),
                  (pi/2, Eq(k, m) | (Eq(k, 0) & Eq(m, 0))),
                  (0, True))
    # 断言：应该进一步简化为 Piecewise 形式：
    # Piecewise(
    #    (0, Eq(k, 0) | Eq(m, 0)),
    #    (-pi/2, Eq(k, -m)),
    #    (pi/2, Eq(k, m)),
    #    (0, True))
    # 使用积分函数 `integrate` 计算给定表达式关于变量 x 的积分，与 Piecewise 对象进行比较
    assert integrate(sin(k*x)*sin(m*x), (x,)) == Piecewise(
        # 当 k 和 m 均为 0 时，积分结果为 0
        (0, And(Eq(k, 0), Eq(m, 0))),
        # 当 k = -m 时的积分表达式
        (-x*sin(m*x)**2/2 - x*cos(m*x)**2/2 + sin(m*x)*cos(m*x)/(2*m), Eq(k, -m)),
        # 当 k = m 时的积分表达式
        (x*sin(m*x)**2/2 + x*cos(m*x)**2/2 - sin(m*x)*cos(m*x)/(2*m), Eq(k, m)),
        # 对于其他情况的默认情况，使用 True 表示任意 k 和 m 的情况
        (m*sin(k*x)*cos(m*x)/(k**2 - m**2) - k*sin(m*x)*cos(k*x)/(k**2 - m**2), True))
# 定义用于测试问题 4199 的函数
def test_issue_4199():
    # 创建一个正数符号 'y'
    ypos = Symbol('y', positive=True)
    # 断言积分结果，对无穷区间 (x, -oo, oo) 的积分
    # TODO: Remove conds='none' below, let the assumption take care of it.
    assert integrate(exp(-I*2*pi*ypos*x)*x, (x, -oo, oo), conds='none') == \
        Integral(exp(-I*2*pi*ypos*x)*x, (x, -oo, oo))


# 定义用于测试问题 3940 的函数
def test_issue_3940():
    # 定义一些正数符号 'a', 'b', 'c', 'd'
    a, b, c, d = symbols('a:d', positive=True)
    # 断言积分结果，积分表达式包含复数单位 I
    assert integrate(exp(-x**2 + I*c*x), x) == \
        -sqrt(pi)*exp(-c**2/4)*erf(I*c/2 - x)/2
    # 断言积分结果，积分表达式包含复数单位 I 和 erf 函数
    assert integrate(exp(a*x**2 + b*x + c), x) == \
        sqrt(pi)*exp(c - b**2/(4*a))*erfi((2*a*x + b)/(2*sqrt(a)))

    # 导入 sympy 核心函数中的 expand_mul 函数，以及 sympy.abc 中的 k 符号
    from sympy.core.function import expand_mul
    from sympy.abc import k
    # 断言积分结果，对指数函数和正负无穷区间的积分
    assert expand_mul(integrate(exp(-x**2)*exp(I*k*x), (x, -oo, oo))) == \
        sqrt(pi)*exp(-k**2/4)
    # 定义符号 'a' 和 'd' 为正数
    a, d = symbols('a d', positive=True)
    # 断言积分结果，对指数函数和正负无穷区间的积分，应用 expand_mul 函数
    assert expand_mul(integrate(exp(-a*x**2 + 2*d*x), (x, -oo, oo))) == \
        sqrt(pi)*exp(d**2/a)/sqrt(a)


# 定义用于测试问题 5413 的函数
def test_issue_5413():
    # 注意这里的注释指出，这个测试不同于 ratint() 的测试，因为 integrate() 提取了系数。
    assert integrate(-a/(a**2 + x**2), x) == I*log(-I*a + x)/2 - I*log(I*a + x)/2


# 定义用于测试问题 4892a 的函数
def test_issue_4892a():
    # 定义符号 'A' 和 'z'
    A, z = symbols('A z')
    # 定义一个非零的符号 'c'
    c = Symbol('c', nonzero=True)
    # 定义表达式 P1 和 P2
    P1 = -A*exp(-z)
    P2 = -A/(c*t)*(sin(x)**2 + cos(y)**2)

    # 定义表达式 h1 和 h2
    h1 = -sin(x)**2 - cos(y)**2
    h2 = -sin(x)**2 + sin(y)**2 - 1

    # 这里注意到在 integrate 或 trigsimp 中仍然存在一些非确定性行为，
    # 允许以下其中一个结果
    assert integrate(c*(P2 - P1), t) in [
        c*(-A*(-h1)*log(c*t)/c + A*t*exp(-z)),
        c*(-A*(-h2)*log(c*t)/c + A*t*exp(-z)),
        c*( A* h1 *log(c*t)/c + A*t*exp(-z)),
        c*( A* h2 *log(c*t)/c + A*t*exp(-z)),
        (A*c*t - A*(-h1)*log(t)*exp(z))*exp(-z),
        (A*c*t - A*(-h2)*log(t)*exp(z))*exp(-z),
    ]


# 定义用于测试问题 4892b 的函数
def test_issue_4892b():
    # 与问题 4596 相关的问题使得难以测试实际结果。
    # 答案应该类似于 (-sin(y) + sqrt(-72 + 48*cos(y) - 8*cos(y)**2)/2)*log(x + sqrt(-72 +
    # 48*cos(y) - 8*cos(y)**2)/(2*(3 - cos(y)))) + (-sin(y) - sqrt(-72 +
    # 48*cos(y) - 8*cos(y)**2)/2)*log(x - sqrt(-72 + 48*cos(y) -
    # 8*cos(y)**2)/(2*(3 - cos(y)))) + x**2*sin(y)/2 + 2*x*cos(y)
    expr = (sin(y)*x**3 + 2*cos(y)*x**2 + 12)/(x**2 + 2)
    assert trigsimp(factor(integrate(expr, x).diff(x) - expr)) == 0


# 定义用于测试问题 5178 的函数
def test_issue_5178():
    # 断言三重积分结果
    assert integrate(sin(x)*f(y, z), (x, 0, pi), (y, 0, pi), (z, 0, pi)) == \
        2*Integral(f(y, z), (y, 0, pi), (z, 0, pi))


# 定义用于测试积分级数的函数
def test_integrate_series():
    # 计算正弦函数在 x=0 处的级数展开
    f = sin(x).series(x, 0, 10)
    # 定义期望的级数 g
    g = x**2/2 - x**4/24 + x**6/720 - x**8/40320 + x**10/3628800 + O(x**11)

    # 断言积分结果等于级数 g
    assert integrate(f, x) == g
    # 断言对积分结果求导后等于原函数 f
    assert diff(integrate(f, x), x) == f

    # 断言对 O(x**5) 的积分结果是 O(x**6)
    assert integrate(O(x**5), x) == O(x**6)


# 定义用于测试 atom_bug 的函数
def test_atom_bug():
    # 导入 heurisch 函数用于测试
    from sympy.integrals.heurisch import heurisch
    # 断言某个 meijerg 函数的 heurisch 积分为 None
    assert heurisch(meijerg([], [], [1], [], x), x) is None


# 定义用于测试 limit_bug 的函数
def test_limit_bug():
    # 定义一个非零符号 'z'
    z = Symbol('z', zero=False)
    # 断言：验证三重积分结果是否等于指定表达式
    assert integrate(sin(x*y*z), (x, 0, pi), (y, 0, pi)).together() == \
        (log(z) - Ci(pi**2*z) + EulerGamma + 2*log(pi))/z
def test_issue_4703():
    # 创建一个名为 g 的函数对象
    g = Function('g')
    # 断言对 exp(x) * g(x) 进行积分是否包含 Integral 符号
    assert integrate(exp(x)*g(x), x).has(Integral)


def test_issue_1888():
    # 创建一个名为 f 的函数对象
    f = Function('f')
    # 断言对 f(x).diff(x)**2 进行积分是否包含 Integral 符号
    assert integrate(f(x).diff(x)**2, x).has(Integral)

# The following tests work using meijerint.


def test_issue_3558():
    # 断言对 cos(x*y) 在区间 (x, -pi/2, pi/2), (y, 0, pi) 的二重积分结果
    assert integrate(cos(x*y), (x, -pi/2, pi/2), (y, 0, pi)) == 2*Si(pi**2/2)


def test_issue_4422():
    # 断言对 1/sqrt(16 + 4*x**2) 的积分结果
    assert integrate(1/sqrt(16 + 4*x**2), x) == asinh(x/2) / 2


def test_issue_4493():
    # 断言简化后对 x*sqrt(1 + 2*x) 的积分结果
    assert simplify(integrate(x*sqrt(1 + 2*x), x)) == \
        sqrt(2*x + 1)*(6*x**2 + x - 1)/15


def test_issue_4737():
    # 断言对 sin(x)/x 在 (-oo, oo) 区间的积分结果
    assert integrate(sin(x)/x, (x, -oo, oo)) == pi
    # 断言对 sin(x)/x 在 (0, oo) 区间的积分结果
    assert integrate(sin(x)/x, (x, 0, oo)) == pi/2
    # 断言对 sin(x)/x 的不定积分结果
    assert integrate(sin(x)/x, x) == Si(x)


def test_issue_4992():
    # 注：在 _check_antecedents 中，psi 变成 NaN。
    from sympy.core.function import expand_func
    # 创建一个名为 a 的正数符号对象
    a = Symbol('a', positive=True)
    # 断言简化后对 exp(-x)*log(x)*x**a 的积分结果
    assert simplify(expand_func(integrate(exp(-x)*log(x)*x**a, (x, 0, oo)))) == \
        (a*polygamma(0, a) + 1)*gamma(a)


def test_issue_4487():
    from sympy.functions.special.gamma_functions import lowergamma
    # 断言简化后对 exp(-x)*x**y 的积分结果
    assert simplify(integrate(exp(-x)*x**y, x)) == lowergamma(y + 1, x)


def test_issue_4215():
    x = Symbol("x")
    # 断言对 1/(x**2) 在区间 (x, -1, 1) 的积分结果为无穷大
    assert integrate(1/(x**2), (x, -1, 1)) is oo


def test_issue_4400():
    n = Symbol('n', integer=True, positive=True)
    # 断言对 (x**n)*log(x) 的积分结果
    assert integrate((x**n)*log(x), x) == \
        n*x*x**n*log(x)/(n**2 + 2*n + 1) + x*x**n*log(x)/(n**2 + 2*n + 1) - \
        x*x**n/(n**2 + 2*n + 1)


def test_issue_6253():
    # 注：此前会引发 NotImplementedError。
    # 注：在 _check_antecedents 中，psi 变成 NaN。
    # 断言使用 meijerg=True 对 (sqrt(1 - x) + sqrt(1 + x))**2/x 的积分结果
    assert integrate((sqrt(1 - x) + sqrt(1 + x))**2/x, x, meijerg=True) == \
        Integral((sqrt(-x + 1) + sqrt(x + 1))**2/x, x)


def test_issue_4153():
    # 断言对 1/(1 + x + y + z) 在区间 (x, 0, 1), (y, 0, 1), (z, 0, 1) 的积分结果
    assert integrate(1/(1 + x + y + z), (x, 0, 1), (y, 0, 1), (z, 0, 1)) in [
        -12*log(3) - 3*log(6)/2 + 3*log(8)/2 + 5*log(2) + 7*log(4),
        6*log(2) + 8*log(4) - 27*log(3)/2, 22*log(2) - 27*log(3)/2,
        -12*log(3) - 3*log(6)/2 + 47*log(2)/2]


def test_issue_4326():
    R, b, h = symbols('R b h')
    # 不关心能否进行积分，只需确保结果不包含 NaN。这是针对 _eval_interval 的测试。
    e = integrate(((h*(x - R + b))/b)*sqrt(R**2 - x**2), (x, R - b, R))
    assert not e.has(nan)
    # 确保它可以评估
    assert not e.has(Integral)


def test_powers():
    # 断言对 2**x + 3**x 的积分结果
    assert integrate(2**x + 3**x, x) == 2**x/log(2) + 3**x/log(3)


def test_manual_option():
    # 使用 manual=True 时会引发 ValueError
    raises(ValueError, lambda: integrate(1/x, x, manual=True, meijerg=True))
    # 一个 manual integration 无法处理的函数示例
    assert integrate(log(1+x)/x, (x, 0, 1), manual=True).has(Integral)


def test_meijerg_option():
    # 使用 meijerg=True 时会引发 ValueError
    raises(ValueError, lambda: integrate(1/x, x, meijerg=True, risch=True))
    # 一个 meijerg integration 无法处理的函数示例
    assert integrate(tan(x), x, meijerg=True) == Integral(tan(x), x)
# 定义测试函数 test_risch_option
def test_risch_option():
    # 只有在不定积分中才允许使用 risch=True
    raises(ValueError, lambda: integrate(1/log(x), (x, 0, oo), risch=True))
    # 断言在使用 risch=True 时积分 exp(-x**2) 得到 NonElementaryIntegral 对象
    assert integrate(exp(-x**2), x, risch=True) == NonElementaryIntegral(exp(-x**2), x)
    # 断言在使用 risch=True 时积分 log(1/x)*y 得到正确的表达式
    assert integrate(log(1/x)*y, x, y, risch=True) == y**2*(x*log(1/x)/2 + x/2)
    # 断言在使用 risch=True 时积分 erf(x) 得到 Integral(erf(x), x) 对象
    assert integrate(erf(x), x, risch=True) == Integral(erf(x), x)
    # TODO: 如何测试 risch=False?


# 标记为慢速测试
@slow
# 定义测试函数 test_heurisch_option
def test_heurisch_option():
    # 断言在使用 risch=True 和 heurisch=True 时会引发 ValueError
    raises(ValueError, lambda: integrate(1/x, x, risch=True, heurisch=True))
    # 断言 heurisch 能处理的积分 exp(x**2) 得到 sqrt(pi)*erfi(x)/2
    assert integrate(exp(x**2), x, heurisch=True) == sqrt(pi)*erfi(x)/2
    # 断言 heurisch 目前无法处理的积分 exp(x)/x 得到 Integral(exp(x)/x, x)
    assert integrate(exp(x)/x, x, heurisch=True) == Integral(exp(x)/x, x)
    # 断言 heurisch 目前无法处理的积分 log(x)*cos(log(x))/x**Rational(3, 4) 得到复杂表达式
    assert integrate(log(x)*cos(log(x))/x**Rational(3, 4), x, heurisch=False) == (
        -128*x**Rational(1, 4)*sin(log(x))/289 + 240*x**Rational(1, 4)*cos(log(x))/289 +
        (16*x**Rational(1, 4)*sin(log(x))/17 + 4*x**Rational(1, 4)*cos(log(x))/17)*log(x))


# 定义测试函数 test_issue_6828
def test_issue_6828():
    # 定义函数 f = 1/(1.08*x**2 - 4.3)
    f = 1/(1.08*x**2 - 4.3)
    # 计算 f 的积分并对其求导数，与原函数在数值上验证是否相等，精度为 1e-12
    g = integrate(f, x).diff(x)
    assert verify_numerically(f, g, tol=1e-12)


# 定义测试函数 test_issue_4803
def test_issue_4803():
    # 定义符号变量 x_max
    x_max = Symbol("x_max")
    # 断言积分 y/pi*exp(-(x_max - x)/cos(a)) 得到正确结果
    assert integrate(y/pi*exp(-(x_max - x)/cos(a)), x) == \
        y*exp((x - x_max)/cos(a))*cos(a)/pi


# 定义测试函数 test_issue_4234
def test_issue_4234():
    # 断言积分 1/sqrt(1 + tan(x)**2) 得到 tan(x)/sqrt(1 + tan(x)**2)
    assert integrate(1/sqrt(1 + tan(x)**2)) == tan(x)/sqrt(1 + tan(x)**2)


# 定义测试函数 test_issue_4492
def test_issue_4492():
    # 断言积分 x**2 * sqrt(5 - x**2) 简化后的因式分解形式
    assert simplify(integrate(x**2 * sqrt(5 - x**2), x)).factor(
        deep=True) == Piecewise(
        (I*(2*x**5 - 15*x**3 + 25*x - 25*sqrt(x**2 - 5)*acosh(sqrt(5)*x/5)) /
            (8*sqrt(x**2 - 5)), (x > sqrt(5)) | (x < -sqrt(5))),
        ((2*x**5 - 15*x**3 + 25*x - 25*sqrt(5 - x**2)*asin(sqrt(5)*x/5)) /
            (-8*sqrt(-x**2 + 5)), True))


# 定义测试函数 test_issue_2708
def test_issue_2708():
    # 定义函数 f = 1/(a + z + log(z))
    f = 1/(a + z + log(z))
    # 使用 NonElementaryIntegral 对象表示积分结果
    integral_f = NonElementaryIntegral(f, (z, 2, 3))
    # 断言积分结果与 NonElementaryIntegral 对象相等
    assert Integral(f, (z, 2, 3)).doit() == integral_f
    # 断言积分 f + exp(z) 在特定区间的结果
    assert integrate(f + exp(z), (z, 2, 3)) == integral_f - exp(2) + exp(3)
    # 断言积分 2*f + exp(z) 在特定区间的结果
    assert integrate(2*f + exp(z), (z, 2, 3)) == \
        2*integral_f - exp(2) + exp(3)
    # 断言积分 exp(1.2*n*s*z*(-t + z)/t) 的结果
    assert integrate(exp(1.2*n*s*z*(-t + z)/t), (z, 0, x)) == \
        NonElementaryIntegral(exp(-1.2*n*s*z)*exp(1.2*n*s*z**2/t),
                                  (z, 0, x))


# 定义测试函数 test_issue_2884
def test_issue_2884():
    # 定义复杂函数 f
    f = (4.000002016020*x + 4.000002016020*y + 4.000006024032)*exp(10.0*x)
    # 计算积分结果 e
    e = integrate(f, (x, 0.1, 0.2))
    # 断言积分结果的字符串表示与预期结果相等
    assert str(e) == '1.86831064982608*y + 2.16387491480008'


# 定义测试函数 test_issue_8368i
def test_issue_8368i():
    # 导入 arg 和 Abs 函数
    from sympy.functions.elementary.complexes import arg, Abs
    # 断言：计算指数衰减乘以双曲余弦的积分
    assert integrate(exp(-s*x)*cosh(x), (x, 0, oo)) == \
        Piecewise(
            (   # 条件1：当 s^2 < 1 时，返回 π 乘以 Piecewise 的值
                pi*Piecewise(
                    (   -s/(pi*(-s**2 + 1)),  # 当 |s^2| < 1 时的分段函数值
                        Abs(s**2) < 1),
                    (   1/(pi*s*(1 - 1/s**2)),  # 当 |s^(-2)| < 1 时的分段函数值
                        Abs(s**(-2)) < 1),
                    (   meijerg(
                            ((S.Half,), (0, 0)),
                            ((0, S.Half), (0,)),
                            polar_lift(s)**2),  # 对于其他情况，使用 meijerg 函数
                        True)
                ),
                s**2 > 1  # 条件2：当 s^2 > 1 时的返回值
            ),
            (
                Integral(exp(-s*x)*cosh(x), (x, 0, oo)),  # 否则返回积分表达式本身
                True  # 默认情况下返回 True
            ))
    
    # 断言：计算指数衰减乘以双曲正弦的积分
    assert integrate(exp(-s*x)*sinh(x), (x, 0, oo)) == \
        Piecewise(
            (   # 当条件满足时的分段函数值
                -1/(s + 1)/2 - 1/(-s + 1)/2,
                And(
                    Abs(s) > 1,  # 条件1：|s| > 1
                    Abs(arg(s)) < pi/2,  # 条件2：arg(s) 的绝对值小于 pi/2
                    Abs(arg(s)) <= pi/2  # 条件3：arg(s) 的绝对值小于等于 pi/2
                    )),
            (   Integral(exp(-s*x)*sinh(x), (x, 0, oo)),  # 否则返回积分表达式本身
                True  # 默认情况下返回 True
            ))
# 测试解决问题号 8901
def test_issue_8901():
    # 断言求解双曲正弦函数的积分，应等于双曲余弦函数
    assert integrate(sinh(1.0*x)) == 1.0*cosh(1.0*x)
    # 断言求解双曲正切函数的积分，应等于 x - log(tanh(x) + 1)
    assert integrate(tanh(1.0*x)) == 1.0*x - 1.0*log(tanh(1.0*x) + 1)
    # 断言求解双曲正切函数的积分，应等于 x - log(tanh(x) + 1)
    assert integrate(tanh(x)) == x - log(tanh(x) + 1)


# 标记为慢速测试
@slow
def test_issue_8945():
    # 断言对 sin(x)^3 / x 在区间 (0, 1) 上的积分，应等于 -Si(3)/4 + 3*Si(1)/4
    assert integrate(sin(x)**3/x, (x, 0, 1)) == -Si(3)/4 + 3*Si(1)/4
    # 断言对 sin(x)^3 / x 在区间 (0, 无穷) 上的积分，应等于 pi/4
    assert integrate(sin(x)**3/x, (x, 0, oo)) == pi/4
    # 断言对 cos(x)^2 / x^2 的积分，应等于 -Si(2*x) - cos(2*x)/(2*x) - 1/(2*x)
    assert integrate(cos(x)**2/x**2, x) == -Si(2*x) - cos(2*x)/(2*x) - 1/(2*x)


# 标记为慢速测试
@slow
def test_issue_7130():
    # 定义符号变量 i, L, a, b
    i, L, a, b = symbols('i L a b')
    # 计算积分被积函数 cos(pi*i*x/L)^2 / (a + b*x) 的指数重写为指数形式
    integrand = (cos(pi*i*x/L)**2 / (a + b*x)).rewrite(exp)
    # 断言积分结果中不包含符号 x
    assert x not in integrate(integrand, (x, 0, L)).free_symbols


# 测试解决问题号 10567
def test_issue_10567():
    # 定义符号变量 a, b, c, t
    a, b, c, t = symbols('a b c t')
    # 定义矩阵 vt
    vt = Matrix([a*t, b, c])
    # 断言对矩阵 vt 对 t 的积分结果等于对积分表达式 vt 的直接求值结果
    assert integrate(vt, t) == Integral(vt, t).doit()
    # 断言对矩阵 vt 对 t 的积分结果等于矩阵 [[a*t**2/2], [b*t], [c*t]]
    assert integrate(vt, t) == Matrix([[a*t**2/2], [b*t], [c*t]])


# 测试解决问题号 11742
def test_issue_11742():
    # 断言对方程 sqrt(-x**2 + 8*x + 48) 在区间 (4, 12) 上的积分，应等于 16*pi
    assert integrate(sqrt(-x**2 + 8*x + 48), (x, 4, 12)) == 16*pi


# 测试解决问题号 11856
def test_issue_11856():
    # 定义符号变量 t
    t = symbols('t')
    # 断言对 sinc(pi*t) 的积分，应等于 Si(pi*t)/pi
    assert integrate(sinc(pi*t), t) == Si(pi*t)/pi


# 标记为慢速测试
@slow
def test_issue_11876():
    # 断言对 sqrt(log(1/x)) 在区间 (0, 1) 上的积分，应等于 sqrt(pi)/2
    assert integrate(sqrt(log(1/x)), (x, 0, 1)) == sqrt(pi)/2


# 测试解决问题号 4950
def test_issue_4950():
    # 断言对表达式 (-60*exp(x) - 19.2*exp(4*x))*exp(4*x) 的积分，应等于 -2.4*exp(8*x) - 12.0*exp(5*x)
    assert integrate((-60*exp(x) - 19.2*exp(4*x))*exp(4*x), x) == -2.4*exp(8*x) - 12.0*exp(5*x)


# 测试解决问题号 4968
def test_issue_4968():
    # 断言对 sin(log(x**2)) 的积分，应等于 x*sin(log(x**2))/5 - 2*x*cos(log(x**2))/5
    assert integrate(sin(log(x**2))) == x*sin(log(x**2))/5 - 2*x*cos(log(x**2))/5


# 测试解决问题号 12645
def test_issue_12645():
    # 定义符号变量 x, y 为实数
    x, y = symbols('x y', real=True)
    # 断言对 sin(x^3 + y^2) 的二重积分
    assert (integrate(sin(x*x*x + y*y),
                      (x, -sqrt(pi - y*y), sqrt(pi - y*y)),
                      (y, -sqrt(pi), sqrt(pi)))
                == Integral(sin(x**3 + y**2),
                            (x, -sqrt(-y**2 + pi), sqrt(-y**2 + pi)),
                            (y, -sqrt(pi), sqrt(pi))))


# 测试解决问题号 12677
def test_issue_12677():
    # 断言对 sin(x) / (cos(x)^3) 在区间 (0, pi/6) 上的积分，应等于 1/6
    assert integrate(sin(x) / (cos(x)**3), (x, 0, pi/6)) == Rational(1, 6)


# 测试解决问题号 14078
def test_issue_14078():
    # 断言对 (cos(3*x)-cos(x))/x 在区间 (0, 无穷) 上的积分，应等于 -log(3)
    assert integrate((cos(3*x)-cos(x))/x, (x, 0, oo)) == -log(3)


# 测试解决问题号 14064
def test_issue_14064():
    # 断言对 1/cosh(x) 在区间 (0, 无穷) 上的积分，应等于 pi/2
    assert integrate(1/cosh(x), (x, 0, oo)) == pi/2


# 测试解决问题号 14027
def test_issue_14027():
    # 断言对 1/(1 + exp(x - S.Half)/(1 + exp(x))) 的积分
    assert integrate(1/(1 + exp(x - S.Half)/(1 + exp(x))), x) == \
        x - exp(S.Half)*log(exp(x) + exp(S.Half)/(1 + exp(S.Half)))/(exp(S.Half) + E)


# 测试解决问题号 8170
def test_issue_8170():
    # 断言对 tan(x) 在区间 (0, pi/2) 上的积分，应为正无穷
    assert integrate(tan(x), (x, 0, pi/2)) is S.Infinity


# 测试解决问题号 8440 和 14040
def test_issue_8440_14040():
    # 断言对 1/x 在区间 (-1, 1) 上的积分，应为未定义
    assert integrate(1/x, (x, -1, 1)) is S.NaN
    # 断言对 1/(x + 1) 在区间 (-2, 3) 上的积分，应为未定义
    assert integrate(1/(x + 1), (x, -2, 3)) is S.NaN


# 测试解决问题号 14096
def test_issue_14096():
    # 断言对 1/(x + y)^2 的积分，其中 x 在 (0, 1)，应等于 -1/(y + 1) + 1/y
    assert integrate
    # 断言语句，用于验证数值积分结果与预期值之间的误差是否小于 1e-6
    assert Abs(integrate(sqrt(1 - x**3), (x, 0, 1)).n() - 0.841309) < 1e-6
# 测试解决问题编号 14375
def test_issue_14375():
    # 断言积分表达式中是否包含 Ei
    assert integrate(exp(I*x)*log(x), x).has(Ei)


# 测试解决问题编号 14437
def test_issue_14437():
    # 创建带有三个参数的函数 f(x, y, z)
    f = Function('f')(x, y, z)
    # 断言多重积分的结果是否等于积分表达式本身
    assert integrate(f, (x, 0, 1), (y, 0, 2), (z, 0, 3)) == \
                Integral(f, (x, 0, 1), (y, 0, 2), (z, 0, 3))


# 测试解决问题编号 14470
def test_issue_14470():
    # 断言给定积分表达式的计算结果是否正确
    assert integrate(1/sqrt(exp(x) + 1), x) == log(sqrt(exp(x) + 1) - 1) - log(sqrt(exp(x) + 1) + 1)


# 测试解决问题编号 14877
def test_issue_14877():
    # 定义复杂的函数表达式 f(x)
    f = exp(1 - exp(x**2)*x + 2*x**2)*(2*x**3 + x)/(1 - exp(x**2)*x)**2
    # 断言对函数 f(x) 的积分结果是否正确
    assert integrate(f, x) == \
        -exp(2*x**2 - x*exp(x**2) + 1)/(x*exp(3*x**2) - exp(2*x**2))


# 测试解决问题编号 14782
def test_issue_14782():
    # 定义函数表达式 f(x)
    f = sqrt(-x**2 + 1)*(-x**2 + x)
    # 断言对函数 f(x) 在给定区间 [-1, 1] 的积分结果是否正确
    assert integrate(f, [x, -1, 1]) == - pi / 8


# 慢速测试解决问题编号 14782
@slow
def test_issue_14782_slow():
    # 定义函数表达式 f(x)
    f = sqrt(-x**2 + 1)*(-x**2 + x)
    # 断言对函数 f(x) 在给定区间 [0, 1] 的积分结果是否正确
    assert integrate(f, [x, 0, 1]) == S.One / 3 - pi / 16


# 测试解决问题编号 12081
def test_issue_12081():
    # 定义函数表达式 f(x)
    f = x**(Rational(-3, 2))*exp(-x)
    # 断言对函数 f(x) 在无穷区间 [0, oo] 的积分结果是否为无穷大
    assert integrate(f, [x, 0, oo]) is oo


# 测试解决问题编号 15285
def test_issue_15285():
    # 定义函数表达式 f(x)
    y = 1/x - 1
    f = 4*y*exp(-2*y)/x**2
    # 断言对函数 f(x) 在区间 [0, 1] 的积分结果是否等于 1
    assert integrate(f, [x, 0, 1]) == 1


# 测试解决问题编号 15432
def test_issue_15432():
    # 断言对函数表达式 x**n * exp(-x) * log(x) 的积分结果是否正确并进行 gamma 简化
    assert integrate(x**n * exp(-x) * log(x), (x, 0, oo)).gammasimp() == Piecewise(
        (gamma(n + 1)*polygamma(0, n) + gamma(n + 1)/n, re(n) + 1 > 0),
        (Integral(x**n*exp(-x)*log(x), (x, 0, oo)), True))


# 测试解决问题编号 15124
def test_issue_15124():
    # 定义符号 omega 和索引符号 m, p
    omega = IndexedBase('omega')
    m, p = symbols('m p', cls=Idx)
    # 断言对给定函数的复杂积分结果是否正确，条件设置为 'none'
    assert integrate(exp(x*I*(omega[m] + omega[p])), x, conds='none') == \
        -I*exp(I*x*omega[m])*exp(I*x*omega[p])/(omega[m] + omega[p])


# 测试解决问题编号 15218
def test_issue_15218():
    with warns_deprecated_sympy():
        # 断言对 deprecated 的情况进行警告处理
        Integral(Eq(x, y))
    with warns_deprecated_sympy():
        # 断言 deprecated 的积分结果是否正确
        assert Integral(Eq(x, y), x) == Eq(Integral(x, x), Integral(y, x))
    with warns_deprecated_sympy():
        # 断言 deprecated 的积分结果是否正确
        assert Integral(Eq(x, y), x).doit() == Eq(x**2/2, x*y)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        # 断言 deprecated 的积分结果是否正确
        assert Eq(x, y).integrate(x) == Eq(x**2/2, x*y)

    # 这些是明确的积分，不受 deprecated 影响
    assert integrate(Eq(x, y), (x, 0, 1)) == Eq(S.Half, y)
    assert Eq(x, y).integrate((x, 0, 1)) == Eq(S.Half, y)


# 测试解决问题编号 15292
def test_issue_15292():
    # 计算复杂函数的积分结果，并进行类型检查
    res = integrate(exp(-x**2*cos(2*t)) * cos(x**2*sin(2*t)), (x, 0, oo))
    assert isinstance(res, Piecewise)
    # 对积分结果进行 gamma 函数简化，并断言结果为零
    assert gammasimp((res - sqrt(pi)/2 * cos(t)).subs(t, pi/6)) == 0


# 测试解决问题编号 4514
def test_issue_4514():
    # 断言对给定函数的积分结果是否正确
    assert integrate(sin(2*x)/sin(x), x) == 2*sin(x)


# 测试解决问题编号 15457
def test_issue_15457():
    # 定义实数符号 x, a, b
    x, a, b = symbols('x a b', real=True)
    # 计算定积分和不定积分的结果
    definite = integrate(exp(Abs(x-2)), (x, a, b))
    indefinite = integrate(exp(Abs(x-2)), x)
    # 断言定积分在特定值下的结果是否正确
    assert definite.subs({a: 1, b: 3}) == -2 + 2*E
    # 断言不定积分在特定值下的差值是否正确
    assert indefinite.subs(x, 3) - indefinite.subs(x, 1) == -2 + 2*E
    # 确定积分的表达式在给定 a=-3 和 b=-1 的情况下是否正确
    assert definite.subs({a: -3, b: -1}) == -exp(3) + exp(5)
    
    # 计算不定积分在 x=-1 和 x=-3 处的值，并验证其差是否等于 -exp(3) + exp(5)
    assert indefinite.subs(x, -1) - indefinite.subs(x, -3) == -exp(3) + exp(5)
def test_issue_15431():
    # 断言积分计算结果与预期值相等
    assert integrate(x*exp(x)*log(x), x) == \
        (x*exp(x) - exp(x))*log(x) - exp(x) + Ei(x)


def test_issue_15640_log_substitutions():
    # 定义函数和对应的积分结果
    f = x/log(x)
    F = Ei(2*log(x))
    # 断言积分计算结果与预期值相等，以及对积分结果求导应与原函数相等
    assert integrate(f, x) == F and F.diff(x) == f
    f = x**3/log(x)**2
    F = -x**4/log(x) + 4*Ei(4*log(x))
    assert integrate(f, x) == F and F.diff(x) == f
    f = sqrt(log(x))/x**2
    F = -sqrt(pi)*erfc(sqrt(log(x)))/2 - sqrt(log(x))/x
    assert integrate(f, x) == F and F.diff(x) == f


def test_issue_15509():
    # 导入所需的模块和函数
    from sympy.vector import CoordSys3D
    N = CoordSys3D('N')
    x = N.x
    # 断言三维坐标系中的积分结果与预期值相等
    assert integrate(cos(a*x + b), (x, x_1, x_2), heurisch=True) == Piecewise(
        (-sin(a*x_1 + b)/a + sin(a*x_2 + b)/a, (a > -oo) & (a < oo) & Ne(a, 0)), \
            (-x_1*cos(b) + x_2*cos(b), True))


def test_issue_4311_fast():
    # 定义实数符号变量
    x = symbols('x', real=True)
    # 断言积分计算结果与预期值相等，使用分段函数表示
    assert integrate(x*abs(9-x**2), x) == Piecewise(
        (x**4/4 - 9*x**2/2, x <= -3),
        (-x**4/4 + 9*x**2/2 - Rational(81, 2), x <= 3),
        (x**4/4 - 9*x**2/2, True))


def test_integrate_with_complex_constants():
    # 定义复数常数符号变量
    K = Symbol('K', positive=True)
    x = Symbol('x', real=True)
    m = Symbol('m', real=True)
    t = Symbol('t', real=True)
    # 断言复杂常数情况下的积分计算结果与预期值相等
    assert integrate(exp(-I*K*x**2+m*x), x) == sqrt(pi)*exp(-I*m**2
                    /(4*K))*erfi((-2*I*K*x + m)/(2*sqrt(K)*sqrt(-I)))/(2*sqrt(K)*sqrt(-I))
    assert integrate(1/(1 + I*x**2), x) == (-I*(sqrt(-I)*log(x - I*sqrt(-I))/2
            - sqrt(-I)*log(x + I*sqrt(-I))/2))
    assert integrate(exp(-I*x**2), x) == sqrt(pi)*erf(sqrt(I)*x)/(2*sqrt(I))

    assert integrate((1/(exp(I*t)-2)), t) == -t/2 - I*log(exp(I*t) - 2)/2
    assert integrate((1/(exp(I*t)-2)), (t, 0, 2*pi)) == -pi


def test_issue_14241():
    # 定义符号变量和条件
    x = Symbol('x')
    n = Symbol('n', positive=True, integer=True)
    # 断言积分计算结果与预期值相等
    assert integrate(n * x ** (n - 1) / (x + 1), x) == \
           n**2*x**n*lerchphi(x*exp_polar(I*pi), 1, n)*gamma(n)/gamma(n + 1)


def test_issue_13112():
    # 断言积分计算结果与预期值相等
    assert integrate(sin(t)**2 / (5 - 4*cos(t)), [t, 0, 2*pi]) == pi / 4


def test_issue_14709b():
    # 定义正数符号变量
    h = Symbol('h', positive=True)
    # 计算积分并断言结果与预期值相等
    i = integrate(x*acos(1 - 2*x/h), (x, 0, h))
    assert i == 5*h**2*pi/16


def test_issue_8614():
    # 定义符号变量
    x = Symbol('x')
    t = Symbol('t')
    # 断言积分计算结果与预期值相等
    assert integrate(exp(t)/t, (t, -oo, x)) == Ei(x)
    assert integrate((exp(-x) - exp(-2*x))/x, (x, 0, oo)) == log(2)


@slow
def test_issue_15494():
    # 定义正数符号变量
    s = symbols('s', positive=True)

    # 定义被积函数和预期结果
    integrand = (exp(s/2) - 2*exp(1.6*s) + exp(s))*exp(s)
    solution = integrate(integrand, s)
    # 断言积分结果不为 NaN，并检查特定点处的数值精度
    assert solution != S.NaN
    assert abs(solution.subs(s, 1) - (-3.67440080236188)) <= 1e-8

    integrand = (exp(s/2) - 2*exp(S(8)/5*s) + exp(s))*exp(s)
    assert integrate(integrand, s) == -10*exp(13*s/5)/13 + 2*exp(3*s/2)/3 + exp(2*s)/2
def test_li_integral():
    # 定义符号变量 y
    y = Symbol('y')
    # 断言积分结果等于分段函数
    assert Integral(li(y*x**2), x).doit() == Piecewise((x*li(x**2*y) - \
        x*Ei(3*log(x**2*y)/2)/sqrt(x**2*y),
        Ne(y, 0)), (0, True))


def test_issue_17473():
    # 定义符号变量 x 和 n
    x = Symbol('x')
    n = Symbol('n')
    # 定义符号变量 h 为分数 1/2
    h = S.Half
    # 计算 ans 的表达式
    ans = x**(n + 1)*gamma(h + h/n)*hyper((h + h/n,),
        (3*h, 3*h + h/n), -x**(2*n)/4)/(2*n*gamma(3*h + h/n))
    # 计算 integrate(sin(x**n), x) 的结果
    got = integrate(sin(x**n), x)
    # 断言 got 等于 ans
    assert got == ans
    # 定义符号变量 _x，使得其不为零
    _x = Symbol('x', zero=False)
    reps = {x: _x}
    # 断言 integrate(sin(_x**n), _x) 等于 ans 经过替换 reps 后展开的结果
    assert integrate(sin(_x**n), _x) == ans.xreplace(reps).expand()


def test_issue_17671():
    # 断言积分结果
    assert integrate(log(log(x)) / x**2, [x, 1, oo]) == -EulerGamma
    assert integrate(log(log(x)) / x**3, [x, 1, oo]) == -log(2)/2 - EulerGamma/2
    assert integrate(log(log(x)) / x**10, [x, 1, oo]) == -log(9)/9 - EulerGamma/9


def test_issue_2975():
    # 定义符号变量 w 和 C
    w = Symbol('w')
    C = Symbol('C')
    y = Symbol('y')
    # 断言积分结果
    assert integrate(1/(y**2+C)**(S(3)/2), (y, -w/2, w/2)) == w/(C**(S(3)/2)*sqrt(1 + w**2/(4*C)))


def test_issue_7827():
    # 定义符号变量 x, n, M 和 N
    x, n, M = symbols('x n M')
    N = Symbol('N', integer=True)
    # 断言积分结果
    assert integrate(summation(x*n, (n, 1, N)), x) == x**2*(N**2/4 + N/4)
    assert integrate(summation(x*sin(n), (n,1,N)), x) == \
        Sum(x**2*sin(n)/2, (n, 1, N))
    assert integrate(summation(sin(n*x), (n,1,N)), x) == \
        Sum(Piecewise((-cos(n*x)/n, Ne(n, 0)), (0, True)), (n, 1, N))
    assert integrate(integrate(summation(sin(n*x), (n,1,N)), x), x) == \
        Piecewise((Sum(Piecewise((-sin(n*x)/n**2, Ne(n, 0)), (-x/n, True)),
        (n, 1, N)), (n > -oo) & (n < oo) & Ne(n, 0)), (0, True))
    assert integrate(Sum(x, (n, 1, M)), x) == M*x**2/2
    raises(ValueError, lambda: integrate(Sum(x, (x, y, n)), y))
    raises(ValueError, lambda: integrate(Sum(x, (x, 1, n)), n))
    raises(ValueError, lambda: integrate(Sum(x, (x, 1, y)), x))


def test_issue_4231():
    # 定义函数 f
    f = (1 + 2*x + sqrt(x + log(x))*(1 + 3*x) + x**2)/(x*(x + sqrt(x + log(x)))*sqrt(x + log(x)))
    # 断言积分结果
    assert integrate(f, x) == 2*sqrt(x + log(x)) + 2*log(x + sqrt(x + log(x)))


def test_issue_17841():
    # 计算 f 的导数
    f = diff(1/(x**2+x+I), x)
    # 断言积分结果
    assert integrate(f, x) == 1/(x**2 + x + I)


def test_issue_21034():
    # 定义符号变量 x，并限定其为实数且非零
    x = Symbol('x', real=True, nonzero=True)
    # 定义函数 f1 和 f2
    f1 = x*(-x**4/asin(5)**4 - x*sinh(x + log(asin(5))) + 5)
    f2 = (x + cosh(cos(4)))/(x*(x + 1/(12*x)))
    # 断言积分结果
    assert integrate(f1, x) == \
        -x**6/(6*asin(5)**4) - x**2*cosh(x + log(asin(5))) + 5*x**2/2 + 2*x*sinh(x + log(asin(5))) - 2*cosh(x + log(asin(5)))
    assert integrate(f2, x) == \
        log(x**2 + S(1)/12)/2 + 2*sqrt(3)*cosh(cos(4))*atan(2*sqrt(3)*x)


def test_issue_4187():
    # 断言积分结果
    assert integrate(log(x)*exp(-x), x) == Ei(-x) - exp(-x)*log(x)
    assert integrate(log(x)*exp(-x), (x, 0, oo)) == -EulerGamma


def test_issue_5547():
    # 定义符号变量 L, z, r0, R0
    L = Symbol('L')
    z = Symbol('z')
    r0 = Symbol('r0')
    R0 = Symbol('R0')
    # 断言：计算给定函数在指定区间内的定积分是否等于指定的值
    assert integrate(r0**2*cos(z)**2, (z, -L/2, L/2)) == -r0**2*(-L/4 -
                    sin(L/2)*cos(L/2)/2) + r0**2*(L/4 + sin(L/2)*cos(L/2)/2)
    
    # 断言：计算给定函数在指定区间内的定积分是否等于一个分段函数
    assert integrate(r0**2*cos(R0*z)**2, (z, -L/2, L/2)) == Piecewise(
        # 如果 R0 在负无穷到正无穷之间且不等于零，则计算如下表达式
        (-r0**2*(-L*R0/4 - sin(L*R0/2)*cos(L*R0/2)/2)/R0 +
         r0**2*(L*R0/4 + sin(L*R0/2)*cos(L*R0/2)/2)/R0, (R0 > -oo) & (R0 < oo) & Ne(R0, 0)),
        # 否则返回 L*r0**2
        (L*r0**2, True))
    
    # 计算频率变量 w
    w = 2*pi*z/L
    
    # 计算解析表达式 sol
    sol = sqrt(2)*sqrt(L)*r0**2*fresnelc(sqrt(2)*sqrt(L))*gamma(S.One/4)/(16*gamma(S(5)/4)) + L*r0**2/2
    
    # 断言：计算给定函数在指定区间内的定积分是否等于预先计算好的解析解 sol
    assert integrate(r0**2*cos(w*z)**2, (z, -L/2, L/2)) == sol
# 测试解决问题 #15810，验证数值积分的准确性
def test_issue_15810():
    assert integrate(1/(2**(2*x/3) + 1), (x, 0, oo)) == Rational(3, 2)


# 测试解决问题 #21024
def test_issue_21024():
    # 定义实数域中非零的符号变量 x
    x = Symbol('x', real=True, nonzero=True)
    # 定义函数 f
    f = log(x)*log(4*x) + log(3*x + exp(2))
    # 计算积分 F 并验证
    F = x*log(x)**2 + x*(1 - 2*log(2)) + (-2*x + 2*x*log(2))*log(x) + \
        (x + exp(2)/6)*log(3*x + exp(2)) + exp(2)*log(3*x + exp(2))/6
    assert F == integrate(f, x)

    # 更多类似的测试情况...
    
    # 确保每一个计算都得到预期的积分结果
    # 断言：计算给定积分并断言其结果等于 pi/12
    assert integrate(cos(3*theta)/(5-4*cos(theta)), (theta, 0, 2*pi)) == pi/12
    
    # 定义被积函数
    integrand = cos(2*theta)/(5 - 4*cos(theta))
    
    # 断言：计算给定积分并断言其结果等于 pi/6
    assert integrate(integrand, (theta, 0, 2*pi)) == pi/6
@slow
# 测试函数，用于验证 Issue 22033 的积分结果是否正确
def test_issue_22033_integral():
    # 断言积分结果是否等于 pi/32
    assert integrate((x**2 - Rational(1, 4))**2 * sqrt(1 - x**2), (x, -1, 1)) == pi/32


@slow
# 测试函数，用于验证 Issue 21671 的积分结果是否正确
def test_issue_21671():
    # 断言第一个积分结果是否等于 pi
    assert integrate(1, (z, x**2 + y**2, 2 - x**2 - y**2), (y, -sqrt(1 - x**2), sqrt(1 - x**2)), (x, -1, 1)) == pi
    # 断言第二个积分结果是否等于 pi
    assert integrate(-4*(1 - x**2)**(S(3)/2)/3 + 2*sqrt(1 - x**2)*(2 - 2*x**2), (x, -1, 1)) == pi


# 测试函数，用于验证 Issue 18527 的积分结果是否正确
def test_issue_18527():
    # 定义实数变量 xr
    xr = symbols('xr', real=True)
    # 定义表达式 expr
    expr = (cos(x)/(4 + (sin(x))**2))
    # 使用手动积分器求解 expr 的积分，然后替换 xr 并再次替换为 x，断言两者相等
    res_real = integrate(expr.subs(x, xr), xr, manual=True).subs(xr, x)
    assert integrate(expr, x, manual=True) == res_real == Integral(expr, x)


# 测试函数，用于验证 Issue 23718 的积分结果是否正确
def test_issue_23718():
    # 定义函数 f
    f = 1/(b*cos(x) + a*sin(x))
    # 定义 Fpos
    Fpos = (-log(-a/b + tan(x/2) - sqrt(a**2 + b**2)/b)/sqrt(a**2 + b**2)
            + log(-a/b + tan(x/2) + sqrt(a**2 + b**2)/b)/sqrt(a**2 + b**2))
    # 定义 Piecewise 函数 F，包含多个条件分支
    F = Piecewise(
        # XXX: 这里的 zoo 情况是 a=b=0，因此结果应该是 zoo 或者在原始被积函数在该情况下实际上是未定义的情况下可能不需要包含
        (zoo*(-log(tan(x/2) - 1) + log(tan(x/2) + 1)),  Eq(a, 0) & Eq(b, 0)),
        (log(tan(x/2))/a,                               Eq(b, 0)),
        (-I/(-I*b*sin(x) + b*cos(x)),                   Eq(a, -I*b)),
        (I/(I*b*sin(x) + b*cos(x)),                     Eq(a,  I*b)),
        (Fpos,                                          True),
    )
    # 断言函数 f 的积分结果是否等于 F
    assert integrate(f, x) == F

    # 定义正数变量 ap 和 bp
    ap, bp = symbols('a, b', positive=True)
    rep = {a: ap, b: bp}
    # 断言替换后的函数 f 的积分结果是否等于替换后的 Fpos 的结果
    assert integrate(f.subs(rep), x) == Fpos.subs(rep)


# 测试函数，用于验证 Issue 23566 的积分结果是否正确
def test_issue_23566():
    # 计算积分结果 i
    i = integrate(1/sqrt(x**2 - 1), (x, -2, -1))
    # 断言 i 是否等于 -log(2 - sqrt(3))
    assert i == -log(2 - sqrt(3))
    # 断言 i 的数值近似是否等于约 1.31695789692482
    assert math.isclose(i.n(), 1.31695789692482)


# 测试函数，用于验证 PR 23583 的积分结果是否正确
def test_pr_23583():
    # 断言积分结果是否正确，包含条件分支
    assert integrate(1/sqrt((x - I)**2 - 1)) == Piecewise((acosh(x - I), Abs((x - I)**2) > 1), (-I*asin(x - I), True))


# 测试函数，用于验证 Issue 7264 的积分结果是否正确
def test_issue_7264():
    # 断言积分结果是否正确
    assert integrate(exp(x)*sqrt(1 + exp(2*x))) == sqrt(exp(2*x) + 1)*exp(x)/2 + asinh(exp(x))/2


# 测试函数，用于验证 Issue 11254a 的积分结果是否正确
def test_issue_11254a():
    # 断言积分结果是否正确
    assert integrate(sech(x), (x, 0, 1)) == 2*atan(tanh(S.Half))


# 测试函数，用于验证 Issue 11254b 的积分结果是否正确
def test_issue_11254b():
    # 断言积分结果是否正确
    assert integrate(csch(x), x) == log(tanh(x/2))
    # 断言积分结果是否等于无穷大
    assert integrate(csch(x), (x, 0, 1)) == oo


# 测试函数，用于验证 Issue 11254d 的积分结果是否正确
def test_issue_11254d():
    # (sech(x)**2).rewrite(sinh)
    # 断言积分结果是否正确
    assert integrate(-1/sinh(x + I*pi/2, evaluate=False)**2, x) == -2/(exp(2*x) + 1)
    # 断言积分结果是否正确
    assert integrate(cosh(x)**(-2), x) == 2*tanh(x/2)/(tanh(x/2)**2 + 1)


# 测试函数，用于验证 Issue 22863 的积分结果是否正确
def test_issue_22863():
    # 计算积分结果 i
    i = integrate((3*x**3 - x**2 + 2*x - 4)/sqrt(x**2 - 3*x + 2), (x, 0, 1))
    # 断言 i 是否等于 -101*sqrt(2)/8 - 135*log(3 - 2*sqrt(2))/16
    assert i == -101*sqrt(2)/8 - 135*log(3 - 2*sqrt(2))/16
    # 断言 i 的数值近似是否等于约 -2.98126694400554
    assert math.isclose(i.n(), -2.98126694400554)


# 测试函数，用于验证 Issue 9723 的积分结果是否正确
def test_issue_9723():
    pass  # 未完成的测试，暂无代码
    # 断言，验证函数 integrate 的计算结果是否与预期的表达式相等
    assert integrate(sqrt(x + sqrt(x))) == \
        2*sqrt(sqrt(x) + x)*(sqrt(x)/12 + x/3 - S(1)/8) + log(2*sqrt(x) + 2*sqrt(sqrt(x) + x) + 1)/8
    
    # 断言，验证函数 integrate 的计算结果是否与预期的表达式相等
    assert integrate(sqrt(2*x+3+sqrt(4*x+5))**3) == \
        sqrt(2*x + sqrt(4*x + 5) + 3) * \
           (9*x/10 + 11*(4*x + 5)**(S(3)/2)/40 + sqrt(4*x + 5)/40 + (4*x + 5)**2/10 + S(11)/10)/2
def test_issue_23704():
    # 这是对异常不会在risch中引发的测试
    # 理想情况下，manualintegrate (manual=True) 能够计算这个表达式，
    # 但是对于这个例子，manualintegrate 很慢，所以我们这里不测试它。
    assert (integrate(log(x)/x**2/(c*x**2+b*x+a),x, risch=True)
        == NonElementaryIntegral(log(x)/(a*x**2 + b*x**3 + c*x**4), x))


def test_exp_substitution():
    # 测试指数替换的积分
    assert integrate(1/sqrt(1-exp(2*x))) == log(sqrt(1 - exp(2*x)) - 1)/2 - log(sqrt(1 - exp(2*x)) + 1)/2


def test_hyperbolic():
    # 测试双曲函数的积分
    assert integrate(coth(x)) == x - log(tanh(x) + 1) + log(tanh(x))
    assert integrate(sech(x)) == 2*atan(tanh(x/2))
    assert integrate(csch(x)) == log(tanh(x/2))


def test_nested_pow():
    # 测试嵌套幂次根的积分
    assert integrate(sqrt(x**2)) == x*sqrt(x**2)/2
    assert integrate(sqrt(x**(S(5)/3))) == 6*x*sqrt(x**(S(5)/3))/11
    assert integrate(1/sqrt(x**2)) == x*log(x)/sqrt(x**2)
    assert integrate(x*sqrt(x**(-4))) == x**2*sqrt(x**-4)*log(x)


def test_sqrt_quadratic():
    # 测试平方根型二次方程的积分
    assert integrate(1/sqrt(3*x**2+4*x+5)) == sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/3
    assert integrate(1/sqrt(-3*x**2+4*x+5)) == sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/3
    assert integrate(1/sqrt(3*x**2+4*x-5)) == sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/3
    assert integrate(1/sqrt(4*x**2-4*x+1)) == (x - S.Half)*log(x - S.Half)/(2*sqrt((x - S.Half)**2))
    assert integrate(1/sqrt(a+b*x+c*x**2), x) == \
        Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(c, 0) & Ne(a - b**2/(4*c), 0)),
                  ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), Ne(c, 0)),
                  (2*sqrt(a + b*x)/b, Ne(b, 0)), (x/sqrt(a), True))

    assert integrate((7*x+6)/sqrt(3*x**2+4*x+5)) == \
           7*sqrt(3*x**2 + 4*x + 5)/3 + 4*sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/9
    assert integrate((7*x+6)/sqrt(-3*x**2+4*x+5)) == \
           -7*sqrt(-3*x**2 + 4*x + 5)/3 + 32*sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/9
    assert integrate((7*x+6)/sqrt(3*x**2+4*x-5)) == \
           7*sqrt(3*x**2 + 4*x - 5)/3 + 4*sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/9
    assert integrate((d+e*x)/sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((-b*e/(2*c) + d) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   e*sqrt(a + b*x + c*x**2)/c, Ne(c, 0)),
                  ((2*d*sqrt(a + b*x) + 2*e*(-a*sqrt(a + b*x) + (a + b*x)**(S(3)/2)/3)/b)/b, Ne(b, 0)),
                  ((d*x + e*x**2/2)/sqrt(a), True))

    assert integrate((3*x**3-x**2+2*x-4)/sqrt(x**2-3*x+2)) == \
           sqrt(x**2 - 3*x + 2)*(x**2 + 13*x/4 + S(101)/8) + 135*log(2*x + 2*sqrt(x**2 - 3*x + 2) - 3)/16
    # 确定积分的结果等于右侧复杂表达式
    assert integrate(sqrt(53225*x**2 - 66732*x + 23013)) == \
           (x/2 - S(16683)/53225)*sqrt(53225*x**2 - 66732*x + 23013) + \
           111576969*sqrt(2129)*asinh(53225*x/10563 - S(11122)/3521)/1133160250
    
    # 对带有 a, b, c 参数的二次方程根号表达式进行积分
    assert integrate(sqrt(a + b*x + c*x**2), x) == \
        Piecewise(
            # 当 c 不等于 0 时的情况
            ((a/2 - b**2/(8*c)) *
             Piecewise(
                 # 如果 a - b**2/(4*c) 不等于 0，则使用 log 表达式
                 (log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                 # 否则使用 (b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2)
                 ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True))
             + (b/(4*c) + x/2)*sqrt(a + b*x + c*x**2), Ne(c, 0)),
            # 当 b 不等于 0 时的情况
            (2*(a + b*x)**(S(3)/2)/(3*b), Ne(b, 0)),
            # 默认情况下的情况，返回 sqrt(a)*x
            (sqrt(a)*x, True))
    
    # 对带有 x*sqrt(x**2+2*x+4) 的表达式进行积分
    assert integrate(x*sqrt(x**2 + 2*x + 4)) == \
        (x**2/3 + x/6 + S(5)/6)*sqrt(x**2 + 2*x + 4) - 3*asinh(sqrt(3)*(x + 1)/3)/2
def test_mul_pow_derivative():
    # 测试积分函数对 x * sec(x) * tan(x) 的计算结果是否正确
    assert integrate(x*sec(x)*tan(x)) == x*sec(x) - log(tan(x) + sec(x))
    # 测试积分函数对 x * sec(x)**2 的计算结果是否正确
    assert integrate(x*sec(x)**2, x) == x*tan(x) + log(cos(x))
    # 测试积分函数对 x**3 * Derivative(f(x), (x, 4)) 的计算结果是否正确
    assert integrate(x**3*Derivative(f(x), (x, 4))) == \
           x**3*Derivative(f(x), (x, 3)) - 3*x**2*Derivative(f(x), (x, 2)) + 6*x*Derivative(f(x), x) - 6*f(x)


def test_issue_20782():
    # 定义 Piecewise 函数 fun1 和 fun2
    fun1 = Piecewise((0, x < 0.0), (1, True))
    fun2 = -Piecewise((0, x < 1.0), (1, True))
    # 计算 fun1 和 fun2 的和
    fun_sum = fun1 + fun2
    L = (x, -float('Inf'), 1)

    # 测试积分函数对 Piecewise 函数的计算结果是否正确
    assert integrate(fun1, L) == 1
    assert integrate(fun2, L) == 0
    assert integrate(-fun1, L) == -1
    assert integrate(-fun2, L) == 0.
    assert integrate(fun_sum, L) == 1.
    assert integrate(-fun_sum, L) == -1.


def test_issue_20781():
    # 定义 Piecewise 函数 P 和 f
    P = lambda a: Piecewise((0, x < a), (1, x >= a))
    f = lambda a: P(int(a)) + P(float(a))
    L = (x, -float('Inf'), x)
    # 计算 f(1) 的积分
    f1 = integrate(f(1), L)
    assert f1 == 2*x - Min(1.0, x) - Min(x, Max(1.0, 1, evaluate=False))
    # 注释：XXX is_zero is True for S(0) and Float(0) and this is baked into
    # the code more deeply than the issue of Float(0) != S(0)
    assert integrate(f(0), (x, -float('Inf'), x)
        ) == 2*x - 2*Min(0, x)


@slow
def test_issue_19427():
    # <https://github.com/sympy/sympy/issues/19427>
    x = Symbol("x")

    # 测试对一些特定函数的积分结果是否正确
    assert integrate((x ** 4) * sqrt(1 - x ** 2), (x, -1, 1)) == pi / 16
    assert integrate((-2 * x ** 2) * sqrt(1 - x ** 2), (x, -1, 1)) == -pi / 4
    assert integrate((1) * sqrt(1 - x ** 2), (x, -1, 1)) == pi / 2

    # 测试对这些函数的和的积分结果是否正确
    assert integrate((x ** 4 - 2 * x ** 2 + 1) * sqrt(1 - x ** 2), (x, -1, 1)) == 5 * pi / 16


def test_issue_23942():
    # 定义两个积分对象 I1 和 I2
    I1 = Integral(1/sqrt(a*(1 + x)**3 + (1 + x)**2), (x, 0, z))
    assert I1.series(a, 1, n=1) == Integral(1/sqrt(x**3 + 4*x**2 + 5*x + 2), (x, 0, z)) + O(a - 1, (a, 1))
    I2 = Integral(1/sqrt(a*(4 - x)**4 + (5 + x)**2), (x, 0, z))
    assert I2.series(a, 2, n=1) == Integral(1/sqrt(2*x**4 - 32*x**3 + 193*x**2 - 502*x + 537), (x, 0, z)) + O(a - 2, (a, 2))


def test_issue_25886():
    # https://github.com/sympy/sympy/issues/25886
    f = (1-x)*exp(0.937098661j*x)
    F_exp = (1.0*(-1.0671234968289*I*y
             + 1.13875255748434
             + 1.0671234968289*I)*exp(0.937098661*I*y)
            - 1.13875255748434*exp(0.937098661*I))
    # 计算 f 关于 x 的积分结果
    F = integrate(f, (x, y, 1.0))
    assert F.is_same(F_exp, math.isclose)


def test_old_issues():
    # https://github.com/sympy/sympy/issues/5212
    I1 = integrate(cos(log(x**2))/x)
    assert I1 == sin(log(x**2))/2
    # https://github.com/sympy/sympy/issues/5462
    I2 = integrate(1/(x**2+y**2)**(Rational(3,2)),x)
    assert I2 == x/(y**3*sqrt(x**2/y**2 + 1))
    # https://github.com/sympy/sympy/issues/6278
    I3 = integrate(1/(cos(x)+2),(x,0,2*pi))
    assert I3 == 2*sqrt(3)*pi/3
```