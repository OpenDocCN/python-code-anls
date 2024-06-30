# `D:\src\scipysrc\sympy\sympy\series\tests\test_series.py`

```
# 导入 SymPy 的特定模块和函数
from sympy.core.evalf import N
from sympy.core.function import (Derivative, Function, PoleError, Subs)
from sympy.core.numbers import (E, Float, Rational, oo, pi, I)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral, integrate
from sympy.series.order import O
from sympy.series.series import series
from sympy.abc import x, y, n, k
from sympy.testing.pytest import raises
from sympy.core import EulerGamma


# 定义测试函数，用于验证 sin 函数的级数展开是否正确
def test_sin():
    # 使用 SymPy 中 sin 函数的级数展开，展开点为 x=0
    e1 = sin(x).series(x, 0)
    e2 = series(sin(x), x, 0)
    # 断言两个级数展开式相等
    assert e1 == e2


# 定义测试函数，用于验证 cos 函数的级数展开是否正确
def test_cos():
    # 使用 SymPy 中 cos 函数的级数展开，展开点为 x=0
    e1 = cos(x).series(x, 0)
    e2 = series(cos(x), x, 0)
    # 断言两个级数展开式相等
    assert e1 == e2


# 定义测试函数，用于验证 exp 函数的级数展开是否正确
def test_exp():
    # 使用 SymPy 中 exp 函数的级数展开，展开点为 x=0
    e1 = exp(x).series(x, 0)
    e2 = series(exp(x), x, 0)
    # 断言两个级数展开式相等
    assert e1 == e2


# 定义测试函数，用于验证 exp(cos(x)) 函数的级数展开是否正确
def test_exp2():
    # 使用 SymPy 中 exp(cos(x)) 函数的级数展开，展开点为 x=0
    e1 = exp(cos(x)).series(x, 0)
    e2 = series(exp(cos(x)), x, 0)
    # 断言两个级数展开式相等
    assert e1 == e2


# 定义测试函数，用于验证 SymPy 的 issue 5223 是否修复
def test_issue_5223():
    # 验证 1 在 x=0 处的级数展开结果为 1
    assert series(1, x) == 1
    # 验证 SymPy 中 S.Zero 在 x=0 处的极限级数展开结果为 0
    assert next(S.Zero.lseries(x)) == 0
    # 验证 cos(x) 在默认展开点 x=0 处的级数展开与以 x 为变量的级数展开结果相同
    assert cos(x).series() == cos(x).series(x)
    # 验证对 cos(x + y) 进行展开时会抛出 ValueError 异常
    raises(ValueError, lambda: cos(x + y).series())
    # 验证对 x 进行展开时会抛出 ValueError 异常
    raises(ValueError, lambda: x.series(dir=""))

    # 验证级数展开结果去除高阶无穷小后，cos(x+1) 在 x=1 处展开与 cos(x) 在 x=1 处展开结果相等
    assert (cos(x).series(x, 1) -
            cos(x + 1).series(x).subs(x, x - 1)).removeO() == 0

    # 验证对 cos(x) 在 x=1 处展开的前两项是否为 [cos(1), -((x-1)*sin(1))]
    e = cos(x).series(x, 1, n=None)
    assert [next(e) for i in range(2)] == [cos(1), -((x - 1)*sin(1))]

    # 验证对 cos(x) 在 x=1 处反向展开的前两项是否为 [cos(1), (1-x)*sin(1)]
    e = cos(x).series(x, 1, n=None, dir='-')
    assert [next(e) for i in range(2)] == [cos(1), (1 - x)*sin(1)]

    # 验证绝对值函数 abs(x) 在 x=1 处的级数展开结果为 x
    assert abs(x).series(x, 1, dir='+') == x

    # 验证 exp(x) 在 x=1 处反向展开并保留前三项的级数展开结果去除高阶无穷小后是否为 E - E*(-x+1) + E*(-x+1)**2/2
    assert exp(x).series(x, 1, dir='-', n=3).removeO() == \
        E - E*(-x + 1) + E*(-x + 1)**2/2

    # 创建 Derivative 对象 D
    D = Derivative
    # 验证二阶对 x 和一阶对 y 的导数在 x=0 处的级数展开结果是否为 12*x*y
    assert D(x**2 + x**3*y**2, x, 2, y, 1).series(x).doit() == 12*x*y

    # 验证 cos(x) 在 x=0 处的导数级数展开的第一项是否为 D(1, x)
    assert next(D(cos(x), x).lseries()) == D(1, x)

    # 验证 exp(x) 在 x=0 处展开并保留前三项的级数展开结果是否为 D(1, x) + D(x, x) + D(x**2/2, x) + D(x**3/6, x) + O(x**3)
    assert D(
        exp(x), x).series(n=3) == D(1, x) + D(x, x) + D(x**2/2, x) + D(x**3/6, x) + O(x**3)

    # 验证二重积分 Integral(x, (x, 1, 3), (y, 1, x)) 在 x=0 处的级数展开结果是否为 -4 + 4*x
    assert Integral(x, (x, 1, 3), (y, 1, x)).series(x) == -4 + 4*x

    # 验证带有 O(x**2) 项的 1 + x 在 x=0 处的级数展开中 O(x**2) 的阶数是否为 2
    assert (1 + x + O(x**2)).getn() == 2
    # 验证不带有 O(x**2) 项的 1 + x 的级数展开中 O(x**2) 的阶数是否为 None
    assert (1 + x).getn() is None

    # 验证对 (1/sin(x))**oo 进行展开时会抛出 PoleError 异常
    raises(PoleError, lambda: ((1/sin(x))**oo).series())

    # 创建符号 logx
    logx = Symbol('logx')
    # 验证 sin(x)**y 在 x=0 处的级数展开结果为 exp(y*logx) + O(x*exp(y*logx), x)
    assert ((sin(x))**y).nseries(x, n=1, logx=logx) == \
        exp(y*logx) + O(x*exp(y*logx), x)

    # 验证 abs(1/x) 在 x 无穷大处展开前五项的级数展开结果为 1/x - 1/(6*x**3) + O(x**(-5), (x, oo))
    assert sin(1/x).series(x, oo, n=5) == 1/x - 1/(6*x**3) + O(x**(-5), (x, oo))

    # 验证 abs(x) 在 x 无穷大处正向展开前五项的级数展开结果为 x
    assert abs(x).series(x, oo, n=5, dir='+') == x
    # 验证 abs(x) 在 x 无穷大处反向展开前五项的级数展开结果为 -x
    assert abs(x).series(x, -oo, n=5, dir='-') ==
    # 断言：对 exp(sqrt(p)**3*log(p)).series(n=3) 的结果进行验证，应该等于以下级数展开形式
    assert exp(sqrt(p)**3*log(p)).series(n=3) == \
        1 + p**S('3/2')*log(p) + O(p**3*log(p)**3)
    
    # 断言：对 exp(sin(x)*log(x)).series(n=2) 的结果进行验证，应该等于以下级数展开形式
    assert exp(sin(x)*log(x)).series(n=2) == 1 + x*log(x) + O(x**2*log(x)**2)
def test_issue_6350():
    # 计算积分表达式的级数展开，并断言其结果
    expr = integrate(exp(k*(y**3 - 3*y)), (y, 0, oo), conds='none')
    assert expr.series(k, 0, 3) == -(-1)**(S(2)/3)*sqrt(3)*gamma(S(1)/3)**2*gamma(S(2)/3)/(6*pi*k**(S(1)/3)) - \
        sqrt(3)*k*gamma(-S(2)/3)*gamma(-S(1)/3)/(6*pi) - \
        (-1)**(S(1)/3)*sqrt(3)*k**(S(1)/3)*gamma(-S(1)/3)*gamma(S(1)/3)*gamma(S(2)/3)/(6*pi) - \
        (-1)**(S(2)/3)*sqrt(3)*k**(S(5)/3)*gamma(S(1)/3)**2*gamma(S(2)/3)/(4*pi) - \
        (-1)**(S(1)/3)*sqrt(3)*k**(S(7)/3)*gamma(-S(1)/3)*gamma(S(1)/3)*gamma(S(2)/3)/(8*pi) + O(k**3)


def test_issue_11313():
    # 断言不同表达式的级数展开结果是否相等
    assert Integral(cos(x), x).series(x) == sin(x).series(x)
    assert Derivative(sin(x), x).series(x, n=3).doit() == cos(x).series(x, n=3)

    # 断言导数的主导项计算结果是否正确
    assert Derivative(x**3, x).as_leading_term(x) == 3*x**2
    assert Derivative(x**3, y).as_leading_term(x) == 0
    assert Derivative(sin(x), x).as_leading_term(x) == 1
    assert Derivative(cos(x), x).as_leading_term(x) == -x

    # 断言 Derivative(1, x) 的主导项的计算结果，应该等于 Derivative(1, x) 因为 Expr.series 方法当前不检测 x 是否是其 free_symbol 之一
    assert Derivative(1, x).as_leading_term(x) == Derivative(1, x)

    # 断言指数函数和对数函数的级数展开结果
    assert Derivative(exp(x), x).series(x).doit() == exp(x).series(x)
    assert 1 + Integral(exp(x), x).series(x) == exp(x).series(x)

    # 断言对数函数的级数展开结果
    assert Derivative(log(x), x).series(x).doit() == (1/x).series(x)
    assert Integral(log(x), x).series(x) == Integral(log(x), x).doit().series(x).removeO()


def test_series_of_Subs():
    from sympy.abc import z

    # 测试 Subs 对象的级数展开结果
    subs1 = Subs(sin(x), x, y)
    subs2 = Subs(sin(x) * cos(z), x, y)
    subs3 = Subs(sin(x * z), (x, z), (y, x))

    assert subs1.series(x) == subs1
    subs1_series = (Subs(x, x, y) + Subs(-x**3/6, x, y) +
        Subs(x**5/120, x, y) + O(y**6))
    assert subs1.series() == subs1_series
    assert subs1.series(y) == subs1_series
    assert subs1.series(z) == subs1
    assert subs2.series(z) == (Subs(z**4*sin(x)/24, x, y) +
        Subs(-z**2*sin(x)/2, x, y) + Subs(sin(x), x, y) + O(z**6))
    assert subs3.series(x).doit() == subs3.doit().series(x)
    assert subs3.series(z).doit() == sin(x*y)

    # 测试 Subs 对象中引发异常的情况
    raises(ValueError, lambda: Subs(x + 2*y, y, z).series())
    assert Subs(x + y, y, z).series(x).doit() == x + z


def test_issue_3978():
    f = Function('f')
    # 断言函数 f(x) 的级数展开结果
    assert f(x).series(x, 0, 3, dir='-') == \
            f(0) + x*Subs(Derivative(f(x), x), x, 0) + \
            x**2*Subs(Derivative(f(x), x, x), x, 0)/2 + O(x**3)
    assert f(x).series(x, 0, 3) == \
            f(0) + x*Subs(Derivative(f(x), x), x, 0) + \
            x**2*Subs(Derivative(f(x), x, x), x, 0)/2 + O(x**3)
    assert f(x**2).series(x, 0, 3) == \
            f(0) + x**2*Subs(Derivative(f(x), x), x, 0) + O(x**3)
    assert f(x**2+1).series(x, 0, 3) == \
            f(1) + x**2*Subs(Derivative(f(x), x), x, 1) + O(x**3)

    class TestF(Function):
        pass
    # 断言语句，用于验证函数 TestF(x) 的泰勒级数展开是否正确
    assert TestF(x).series(x, 0, 3) ==  TestF(0) + \
            x*Subs(Derivative(TestF(x), x), x, 0) + \
            x**2*Subs(Derivative(TestF(x), x, x), x, 0)/2 + O(x**3)
from sympy.series.acceleration import richardson, shanks  # 导入 richardson 和 shanks 函数
from sympy.concrete.summations import Sum  # 导入 Sum 类
from sympy.core.numbers import Integer  # 导入 Integer 类


def test_acceleration():
    e = (1 + 1/n)**n  # 定义 e 表达式
    assert round(richardson(e, n, 10, 20).evalf(), 10) == round(E.evalf(), 10)  # 断言 Richardson 加速函数的结果与 E 的数值近似相等

    A = Sum(Integer(-1)**(k + 1) / k, (k, 1, n))  # 定义一个求和表达式 A
    assert round(shanks(A, n, 25).evalf(), 4) == round(log(2).evalf(), 4)  # 断言 Shanks 加速函数的结果与 log(2) 的数值近似相等
    assert round(shanks(A, n, 25, 5).evalf(), 10) == round(log(2).evalf(), 10)  # 断言带额外参数的 Shanks 加速函数的结果与 log(2) 的数值近似相等


def test_issue_5852():
    assert series(1/cos(x/log(x)), x, 0) == 1 + x**2/(2*log(x)**2) + \
        5*x**4/(24*log(x)**4) + O(x**6)  # 断言级数展开结果


def test_issue_4583():
    assert cos(1 + x + x**2).series(x, 0, 5) == cos(1) - x*sin(1) + \
        x**2*(-sin(1) - cos(1)/2) + x**3*(-cos(1) + sin(1)/6) + \
        x**4*(-11*cos(1)/24 + sin(1)/2) + O(x**5)  # 断言级数展开结果


def test_issue_6318():
    eq = (1/x)**Rational(2, 3)  # 定义一个表达式 eq
    assert (eq + 1).as_leading_term(x) == eq  # 断言表达式的主导项


def test_x_is_base_detection():
    eq = (x**2)**Rational(2, 3)  # 定义一个表达式 eq
    assert eq.series() == x**Rational(4, 3)  # 断言级数展开结果


def test_issue_7203():
    assert series(cos(x), x, pi, 3) == \
        -1 + (x - pi)**2/2 + O((x - pi)**3, (x, pi))  # 断言级数展开结果


def test_exp_product_positive_factors():
    a, b = symbols('a, b', positive=True)  # 定义正数符号变量 a 和 b
    x = a * b  # 定义 x 表达式
    assert series(exp(x), x, n=8) == 1 + a*b + a**2*b**2/2 + \
        a**3*b**3/6 + a**4*b**4/24 + a**5*b**5/120 + a**6*b**6/720 + \
        a**7*b**7/5040 + O(a**8*b**8, a, b)  # 断言级数展开结果


def test_issue_8805():
    assert series(1, n=8) == 1  # 断言级数展开结果为 1


def test_issue_9549():
    y = (x**2 + x + 1) / (x**3 + x**2)  # 定义一个表达式 y
    assert series(y, x, oo) == x**(-5) - 1/x**4 + x**(-3) + 1/x + O(x**(-6), (x, oo))  # 断言级数展开结果


def test_issue_10761():
    assert series(1/(x**-2 + x**-3), x, 0) == x**3 - x**4 + x**5 + O(x**6)  # 断言级数展开结果


def test_issue_12578():
    y = (1 - 1/(x/2 - 1/(2*x))**4)**(S(1)/8)  # 定义一个表达式 y
    assert y.series(x, 0, n=17) == 1 - 2*x**4 - 8*x**6 - 34*x**8 - 152*x**10 - 714*x**12 - \
        3472*x**14 - 17318*x**16 + O(x**17)  # 断言级数展开结果


def test_issue_12791():
    beta = symbols('beta', positive=True)  # 定义正数符号变量 beta
    theta, varphi = symbols('theta varphi', real=True)  # 定义实数符号变量 theta 和 varphi

    expr = (-beta**2*varphi*sin(theta) + beta**2*cos(theta) + \
        beta*varphi*sin(theta) - beta*cos(theta) - beta + 1)/(beta*cos(theta) - 1)**2  # 定义一个复杂表达式 expr

    sol = (0.5/(0.5*cos(theta) - 1.0)**2 - 0.25*cos(theta)/(0.5*cos(theta) - 1.0)**2
        + (beta - 0.5)*(-0.25*varphi*sin(2*theta) - 1.5*cos(theta)
        + 0.25*cos(2*theta) + 1.25)/((0.5*cos(theta) - 1.0)**2*(0.5*cos(theta) - 1.0))
        + 0.25*varphi*sin(theta)/(0.5*cos(theta) - 1.0)**2
        + O((beta - S.Half)**2, (beta, S.Half)))  # 定义一个复杂表达式 sol

    assert expr.series(beta, 0.5, 2).trigsimp() == sol  # 断言级数展开结果


def test_issue_14384():
    x, a = symbols('x a')  # 定义符号变量 x 和 a
    assert series(x**a, x) == x**a  # 断言级数展开结果
    assert series(x**(-2*a), x) == x**(-2*a)  # 断言级数展开结果
    assert series(exp(a*log(x)), x) == exp(a*log(x))  # 断言级数展开结果
    raises(PoleError, lambda: series(x**I, x))  # 断言触发 PoleError 异常
    raises(PoleError, lambda: series(x**(I + 1), x))  # 断言触发 PoleError 异常
    raises(PoleError, lambda: series(exp(I*log(x)), x))  # 断言触发 PoleError 异常


def test_issue_14885():
    # 使用 SymPy 库中的 series 函数计算给定表达式的 Taylor 展开
    assert series(x**Rational(-3, 2)*exp(x), x, 0) == (
        x**Rational(-3, 2) + 1/sqrt(x) +  # 添加 x**(-3/2) 项
        sqrt(x)/2 +                       # 添加 sqrt(x)/2 项
        x**Rational(3, 2)/6 +             # 添加 x**(3/2)/6 项
        x**Rational(5, 2)/24 +            # 添加 x**(5/2)/24 项
        x**Rational(7, 2)/120 +           # 添加 x**(7/2)/120 项
        x**Rational(9, 2)/720 +           # 添加 x**(9/2)/720 项
        x**Rational(11, 2)/5040 +         # 添加 x**(11/2)/5040 项
        O(x**6)                           # 添加大O符号，表示高阶无穷小
    )
def test_issue_15539():
    # 斯特灵级数展开的测试，测试 x 趋向负无穷时的情况
    assert series(atan(x), x, -oo) == (-1/(5*x**5) + 1/(3*x**3) - 1/x - pi/2
        + O(x**(-6), (x, -oo)))
    # 斯特灵级数展开的测试，测试 x 趋向正无穷时的情况
    assert series(atan(x), x, oo) == (-1/(5*x**5) + 1/(3*x**3) - 1/x + pi/2
        + O(x**(-6), (x, oo)))


def test_issue_7259():
    # Lambert W 函数的级数展开测试
    assert series(LambertW(x), x) == x - x**2 + 3*x**3/2 - 8*x**4/3 + 125*x**5/24 + O(x**6)
    assert series(LambertW(x**2), x, n=8) == x**2 - x**4 + 3*x**6/2 + O(x**8)
    assert series(LambertW(sin(x)), x, n=4) == x - x**2 + 4*x**3/3 + O(x**4)

def test_issue_11884():
    # 余弦函数在 x = 1 处的一阶级数展开测试
    assert cos(x).series(x, 1, n=1) == cos(1) + O(x - 1, (x, 1))


def test_issue_18008():
    # 表达式 y 的 x 趋向正无穷时的级数展开测试
    y = x*(1 + x*(1 - x))/((1 + x*(1 - x)) - (1 - x)*(1 - x))
    assert y.series(x, oo, n=4) == -9/(32*x**3) - 3/(16*x**2) - 1/(8*x) + S(1)/4 + x/2 + \
        O(x**(-4), (x, oo))


def test_issue_18842():
    # 对数函数表达式在 x 趋向 0.491 时的级数展开测试
    f = log(x/(1 - x))
    assert f.series(x, 0.491, n=1).removeO().nsimplify() ==  \
        -S(180019443780011)/5000000000000000


def test_issue_19534():
    # 表达式 expr 在 dt 趋向 0 时的级数展开测试，保留 20 位小数
    dt = symbols('dt', real=True)
    assert N(expr.series(dt, 0, 8), 20) == (
            - Float('0.00092592592592592596126289', precision=70) * dt**7
            + Float('0.0027777777777777783174695', precision=70) * dt**6
            + Float('0.016666666666666656027029', precision=70) * dt**5
            + Float('0.083333333333333300951828', precision=70) * dt**4
            + Float('0.33333333333333337034077', precision=70) * dt**3
            + Float('1.0', precision=70) * dt**2
            + Float('1.0', precision=70) * dt
            + Float('1.0', precision=70)
        )


def test_issue_11407():
    # 平方根函数在 x 趋向 0 时的级数展开测试
    a, b, c, x = symbols('a b c x')
    assert series(sqrt(a + b + c*x), x, 0, 1) == sqrt(a + b) + O(x)
    assert series(sqrt(a + b + c + c*x), x, 0, 1) == sqrt(a + b + c) + O(x)


def test_issue_14037():
    # 正弦函数在 x 趋向 0 时的级数展开测试
    assert (sin(x**50)/x**51).series(x, n=0) == 1/x + O(1, x)


def test_issue_20551():
    # 指数函数除以 x 的级数展开的前三项测试
    expr = (exp(x)/x).series(x, n=None)
    terms = [ next(expr) for i in range(3) ]
    assert terms == [1/x, 1, x/2]


def test_issue_20697():
    # 多项式 Q 在 y 趋向 0 时的级数展开测试，化简
    p_0, p_1, p_2, p_3, b_0, b_1, b_2 = symbols('p_0 p_1 p_2 p_3 b_0 b_1 b_2')
    Q = (p_0 + (p_1 + (p_2 + p_3/y)/y)/y)/(1 + ((p_3/(b_0*y) + (b_0*p_2\
        - b_1*p_3)/b_0**2)/y + (b_0**2*p_1 - b_0*b_1*p_2 - p_3*(b_0*b_2\
        - b_1**2))/b_0**3)/y)
    assert Q.series(y, n=3).ratsimp() == b_2*y**2 + b_1*y + b_0 + O(y**3)


def test_issue_21245():
    # 表达式的级数展开测试，带有复杂分母的情况
    fi = (1 + sqrt(5))/2
    assert (1/(1 - x - x**2)).series(x, 1/fi, 1).factor() == \
        (-4812 - 2152*sqrt(5) + 1686*x + 754*sqrt(5)*x\
        + O((x - 2/(1 + sqrt(5)))**2, (x, 2/(1 + sqrt(5)))))/((1 + sqrt(5))\
        *(20 + 9*sqrt(5))**2*(x + sqrt(5)*x - 2))


def test_issue_21938():
    # 表达式在 x 趋向正无穷时的级数展开测试
    expr = sin(1/x + exp(-x)) - sin(1/x)
    assert expr.series(x, oo) == (1/(24*x**4) - 1/(2*x**2) + 1 + O(x**(-6), (x, oo)))*exp(-x)


def test_issue_23432():
    # 表达式在 x 趋向 0.5 时的级数展开测试，期望结果是一个加法表达式并且有 7 个参数
    expr = 1/sqrt(1 - x**2)
    result = expr.series(x, 0.5)
    assert result.is_Add and len(result.args) == 7


def test_issue_23727():
    pass  # 这个测试还没有完成
    # 调用 sympy 库中的 series 函数，生成以 x 为变量、以 sqrt(1 - x**2) 为中心展开的级数
    res = series(sqrt(1 - x**2), x, 0.1)
    # 使用断言（assertion）验证 res 对象的 is_Add 属性是否为 True
    assert res.is_Add == True
def test_issue_24266():
    #type1: exp(f(x))
    # 对于类型1，使用 exp(-I*pi*(2*x+1)) 对 x 展开到三阶
    assert (exp(-I*pi*(2*x+1))).series(x, 0, 3) == -1 + 2*I*pi*x + 2*pi**2*x**2 + O(x**3)
    # 对于类型1，使用 exp(-I*pi*(2*x+1))*gamma(1+x) 对 x 展开到三阶
    assert (exp(-I*pi*(2*x+1))*gamma(1+x)).series(x, 0, 3) == -1 + x*(EulerGamma + 2*I*pi) + \
        x**2*(-EulerGamma**2/2 + 23*pi**2/12 - 2*EulerGamma*I*pi) + O(x**3)

    #type2: c**f(x)
    # 对于类型2，使用 (2*I)**(-I*pi*(2*x+1)) 对 x 展开到二阶
    assert ((2*I)**(-I*pi*(2*x+1))).series(x, 0, 2) == exp(pi**2/2 - I*pi*log(2)) + \
          x*(pi**2*exp(pi**2/2 - I*pi*log(2)) - 2*I*pi*exp(pi**2/2 - I*pi*log(2))*log(2)) + O(x**2)
    # 对于类型2，使用 (2)**(-I*pi*(2*x+1)) 对 x 展开到二阶
    assert ((2)**(-I*pi*(2*x+1))).series(x, 0, 2) == exp(-I*pi*log(2)) - 2*I*pi*x*exp(-I*pi*log(2))*log(2) + O(x**2)

    #type3: f(y)**g(x)
    # 对于类型3，使用 (y)**(I*pi*(2*x+1)) 对 x 展开到二阶
    assert ((y)**(I*pi*(2*x+1))).series(x, 0, 2) == exp(I*pi*log(y)) + 2*I*pi*x*exp(I*pi*log(y))*log(y) + O(x**2)
    # 对于类型3，使用 (I*y)**(I*pi*(2*x+1)) 对 x 展开到二阶
    assert ((I*y)**(I*pi*(2*x+1))).series(x, 0, 2) == exp(I*pi*log(I*y)) + 2*I*pi*x*exp(I*pi*log(I*y))*log(I*y) + O(x**2)
```