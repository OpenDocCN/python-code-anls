# `D:\src\scipysrc\sympy\sympy\series\tests\test_formal.py`

```
# 导入需要的符号和函数模块
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, asech)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin)
from sympy.functions.special.bessel import airyai
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.series.formal import fps
from sympy.series.order import O
from sympy.series.formal import (rational_algorithm, FormalPowerSeries,
                                 FormalPowerSeriesProduct, FormalPowerSeriesCompose,
                                 FormalPowerSeriesInverse, simpleDE,
                                 rational_independent, exp_re, hyper_re)
from sympy.testing.pytest import raises, XFAIL, slow

# 定义符号变量
x, y, z = symbols('x y z')
n, m, k = symbols('n m k', integer=True)
f, r = Function('f'), Function('r')

# 定义测试函数 test_rational_algorithm
def test_rational_algorithm():
    # 测试用例1
    f = 1 / ((x - 1)**2 * (x - 2))
    assert rational_algorithm(f, x, k) == \
        (-2**(-k - 1) + 1 - (factorial(k + 1) / factorial(k)), 0, 0)

    # 测试用例2
    f = (1 + x + x**2 + x**3) / ((x - 1) * (x - 2))
    assert rational_algorithm(f, x, k) == \
        (-15*2**(-k - 1) + 4, x + 4, 0)

    # 测试用例3
    f = z / (y*m - m*x - y*x + x**2)
    assert rational_algorithm(f, x, k) == \
        (((-y**(-k - 1)*z) / (y - m)) + ((m**(-k - 1)*z) / (y - m)), 0, 0)

    # 测试用例4
    f = x / (1 - x - x**2)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        (((Rational(-1, 2) + sqrt(5)/2)**(-k - 1) *
         (-sqrt(5)/10 + S.Half)) +
         ((-sqrt(5)/2 - S.Half)**(-k - 1) *
         (sqrt(5)/10 + S.Half)), 0, 0)

    # 测试用例5
    f = 1 / (x**2 + 2*x + 2)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        ((I*(-1 + I)**(-k - 1)) / 2 - (I*(-1 - I)**(-k - 1)) / 2, 0, 0)

    # 测试用例6
    f = log(1 + x)
    assert rational_algorithm(f, x, k) == \
        (-(-1)**(-k) / k, 0, 1)

    # 测试用例7
    f = atan(x)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        (((I*I**(-k)) / 2 - (I*(-I)**(-k)) / 2) / k, 0, 1)

    # 测试用例8
    f = x*atan(x) - log(1 + x**2) / 2
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        (((I*I**(-k + 1)) / 2 - (I*(-I)**(-k + 1)) / 2) /
         (k*(k - 1)), 0, 2)

    # 测试用例9
    f = log((1 + x) / (1 - x)) / 2 - atan(x)
    assert rational_algorithm(f, x, k) is None
    # 断言：验证有理数算法的输出与期望值相等
    assert rational_algorithm(f, x, k, full=True) == \
        ((-(-1)**(-k) / 2 - (I*I**(-k)) / 2 + (I*(-I)**(-k)) / 2 +
          S.Half) / k, 0, 1)
    
    # 断言：验证对于 cos(x) 的有理数算法输出为 None
    assert rational_algorithm(cos(x), x, k) is None
def test_rational_independent():
    # 使用 rational_independent 函数进行独立性测试
    ri = rational_independent
    # 断言空列表的情况下，返回空列表
    assert ri([], x) == []
    # 断言包含 cos(x) 和 sin(x) 的列表，在变量 x 方面是独立的
    assert ri([cos(x), sin(x)], x) == [cos(x), sin(x)]
    # 断言包含 x**2, sin(x), x*sin(x), x**3 的列表，在变量 x 方面是独立的
    assert ri([x**2, sin(x), x*sin(x), x**3], x) == \
        [x**3 + x**2, x*sin(x) + sin(x)]
    # 断言包含 S.One, x*log(x), log(x), sin(x)/x, cos(x), sin(x), x 的列表，在变量 x 方面是独立的
    assert ri([S.One, x*log(x), log(x), sin(x)/x, cos(x), sin(x), x], x) == \
        [x + 1, x*log(x) + log(x), sin(x)/x + sin(x), cos(x)]


def test_simpleDE():
    # 测试 simpleDE 函数对各种简单微分方程的处理
    # 对于 exp(x) 的微分方程
    for DE in simpleDE(exp(x), x, f):
        assert DE == (-f(x) + Derivative(f(x), x), 1)
        break
    # 对于 sin(x) 的微分方程
    for DE in simpleDE(sin(x), x, f):
        assert DE == (f(x) + Derivative(f(x), x, x), 2)
        break
    # 对于 log(1 + x) 的微分方程
    for DE in simpleDE(log(1 + x), x, f):
        assert DE == ((x + 1)*Derivative(f(x), x, 2) + Derivative(f(x), x), 2)
        break
    # 对于 asin(x) 的微分方程
    for DE in simpleDE(asin(x), x, f):
        assert DE == (x*Derivative(f(x), x) + (x**2 - 1)*Derivative(f(x), x, x),
                      2)
        break
    # 对于 exp(x)*sin(x) 的微分方程
    for DE in simpleDE(exp(x)*sin(x), x, f):
        assert DE == (2*f(x) - 2*Derivative(f(x)) + Derivative(f(x), x, x), 2)
        break
    # 对于 ((1 + x)/(1 - x))**n 的微分方程
    for DE in simpleDE(((1 + x)/(1 - x))**n, x, f):
        assert DE == (2*n*f(x) + (x**2 - 1)*Derivative(f(x), x), 1)
        break
    # 对于 airyai(x) 的微分方程
    for DE in simpleDE(airyai(x), x, f):
        assert DE == (-x*f(x) + Derivative(f(x), x, x), 2)
        break


def test_exp_re():
    # 测试 exp_re 函数对不同微分方程形式的正则表达式替换
    d = -f(x) + Derivative(f(x), x)
    assert exp_re(d, r, k) == -r(k) + r(k + 1)

    d = f(x) + Derivative(f(x), x, x)
    assert exp_re(d, r, k) == r(k) + r(k + 2)

    d = f(x) + Derivative(f(x), x) + Derivative(f(x), x, x)
    assert exp_re(d, r, k) == r(k) + r(k + 1) + r(k + 2)

    d = Derivative(f(x), x) + Derivative(f(x), x, x)
    assert exp_re(d, r, k) == r(k) + r(k + 1)

    d = Derivative(f(x), x, 3) + Derivative(f(x), x, 4) + Derivative(f(x))
    assert exp_re(d, r, k) == r(k) + r(k + 2) + r(k + 3)


def test_hyper_re():
    # 测试 hyper_re 函数对不同超几何微分方程形式的处理
    d = f(x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == r(k) + (k+1)*(k+2)*r(k + 2)

    d = -x*f(x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == (k + 2)*(k + 3)*r(k + 3) - r(k)

    d = 2*f(x) - 2*Derivative(f(x), x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == \
        (-2*k - 2)*r(k + 1) + (k + 1)*(k + 2)*r(k + 2) + 2*r(k)

    d = 2*n*f(x) + (x**2 - 1)*Derivative(f(x), x)
    assert hyper_re(d, r, k) == \
        k*r(k) + 2*n*r(k + 1) + (-k - 2)*r(k + 2)

    d = (x**10 + 4)*Derivative(f(x), x) + x*(x**10 - 1)*Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == \
        (k*(k - 1) + k)*r(k) + (4*k - (k + 9)*(k + 10) + 40)*r(k + 10)

    d = ((x**2 - 1)*Derivative(f(x), x, 3) + 3*x*Derivative(f(x), x, x) +
         Derivative(f(x), x))
    assert hyper_re(d, r, k) == \
        ((k*(k - 2)*(k - 1) + 3*k*(k - 1) + k)*r(k) +
         (-k*(k + 1)*(k + 2))*r(k + 2))


def test_fps():
    # 测试 fps 函数对不同形式的形式幂级数的计算
    assert fps(1) == 1
    assert fps(2, x) == 2
    assert fps(2, x, dir='+') == 2
    assert fps(2, x, dir='-') == 2
    assert fps(1/x + 1/x**2) == 1/x + 1/x**2
    # 断言等式是否成立：fps(log(1 + x), hyper=False, rational=False) 等于 log(1 + x)
    assert fps(log(1 + x), hyper=False, rational=False) == log(1 + x)

    # 计算并返回 x^2 + x + 1 的形式幂级数对象 f
    f = fps(x**2 + x + 1)
    # 断言 f 是 FormalPowerSeries 类的实例
    assert isinstance(f, FormalPowerSeries)
    # 断言 f 的函数部分等于 x^2 + x + 1
    assert f.function == x**2 + x + 1
    # 断言 f 的零次项系数为 1
    assert f[0] == 1
    # 断言 f 的二次项系数为 x^2
    assert f[2] == x**2
    # 断言 f 截断到 x^4 项的结果为 x^2 + x + 1 + O(x^4)
    assert f.truncate(4) == x**2 + x + 1 + O(x**4)
    # 断言 f 的多项式形式为 x^2 + x + 1
    assert f.polynomial() == x**2 + x + 1

    # 计算并返回 log(1 + x) 的形式幂级数对象 f
    f = fps(log(1 + x))
    # 断言 f 是 FormalPowerSeries 类的实例
    assert isinstance(f, FormalPowerSeries)
    # 断言 f 的函数部分等于 log(1 + x)
    assert f.function == log(1 + x)
    # 断言 f 在变量 x 替换为 y 后依然等于原来的 f
    assert f.subs(x, y) == f
    # 断言 f 截取到前 5 项的结果为 [0, x, -x^2/2, x^3/3, -x^4/4]
    assert f[:5] == [0, x, -x**2/2, x**3/3, -x**4/4]
    # 断言 f 的主导项（Leading Term）在 x 处的值为 x
    assert f.as_leading_term(x) == x
    # 断言 f 的多项式形式展开到 6 项的结果为 x - x^2/2 + x^3/3 - x^4/4 + x^5/5
    assert f.polynomial(6) == x - x**2/2 + x**3/3 - x**4/4 + x**5/5

    # 获取 f 的 ak 属性中的变量 k
    k = f.ak.variables[0]
    # 断言 f 的无穷和（infinite sum）等于 Sum((-(-1)**(-k)*x**k)/k, (k, 1, oo))
    assert f.infinite == Sum((-(-1)**(-k)*x**k)/k, (k, 1, oo))

    # 将 f 截断到 n=None（即全部项）的形式幂级数 ft，并取前 5 项存入 s
    ft, s = f.truncate(n=None), f[:5]
    # 遍历 ft 中的项，并与 s 中相应的项进行断言比较
    for i, t in enumerate(ft):
        if i == 5:
            break
        assert s[i] == t

    # 计算 sin(x) 的形式幂级数对象 f
    f = sin(x).fps(x)
    # 断言 f 是 FormalPowerSeries 类的实例
    assert isinstance(f, FormalPowerSeries)
    # 断言 f 截断到默认项的结果为 x - x^3/6 + x^5/120 + O(x^6)
    assert f.truncate() == x - x**3/6 + x**5/120 + O(x**6)

    # 断言在 y*x 的情况下抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: fps(y*x))
    # 断言在 dir=0 的情况下抛出 ValueError 异常
    raises(ValueError, lambda: fps(x, dir=0))
# 定义一个测试函数，用来测试 fps 函数的各种情况
@slow
def test_fps__rational():
    # 断言：对于 1/x 的泰勒级数展开，应该得到 1/x
    assert fps(1/x) == (1/x)
    
    # 断言：对于 (x**2 + x + 1) / x**3 的泰勒级数展开，应该得到 (x**2 + x + 1) / x**3
    assert fps((x**2 + x + 1) / x**3, dir=-1) == (x**2 + x + 1) / x**3

    # 定义 f = 1 / ((x - 1)**2 * (x - 2))
    f = 1 / ((x - 1)**2 * (x - 2))
    # 断言：对 f 在 x 处的泰勒级数展开，并截取到指定阶数，应该得到指定的级数表达式
    assert fps(f, x).truncate() == \
        (Rational(-1, 2) - x*Rational(5, 4) - 17*x**2/8 - 49*x**3/16 - 129*x**4/32 -
         321*x**5/64 + O(x**6))

    # 定义 f = (1 + x + x**2 + x**3) / ((x - 1) * (x - 2))
    f = (1 + x + x**2 + x**3) / ((x - 1) * (x - 2))
    # 断言：对 f 在 x 处的泰勒级数展开，并截取到指定阶数，应该得到指定的级数表达式
    assert fps(f, x).truncate() == \
        (S.Half + x*Rational(5, 4) + 17*x**2/8 + 49*x**3/16 + 113*x**4/32 +
         241*x**5/64 + O(x**6))

    # 定义 f = x / (1 - x - x**2)
    f = x / (1 - x - x**2)
    # 断言：对 f 在 x 处的泰勒级数展开，展开到完全阶数，并截取到指定阶数，应该得到指定的级数表达式
    assert fps(f, x, full=True).truncate() == \
        x + x**2 + 2*x**3 + 3*x**4 + 5*x**5 + O(x**6)

    # 定义 f = 1 / (x**2 + 2*x + 2)
    f = 1 / (x**2 + 2*x + 2)
    # 断言：对 f 在 x 处的泰勒级数展开，展开到完全阶数，并截取到指定阶数，应该得到指定的级数表达式
    assert fps(f, x, full=True).truncate() == \
        S.Half - x/2 + x**2/4 - x**4/8 + x**5/8 + O(x**6)

    # 定义 f = log(1 + x)
    f = log(1 + x)
    # 断言：对 f 在 x 处的泰勒级数展开，并截取到指定阶数，应该得到指定的级数表达式
    assert fps(f, x).truncate() == \
        x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
    # 断言：对 f 在 x 处的泰勒级数展开，在两个方向上的展开结果应该相同
    assert fps(f, x, dir=1).truncate() == fps(f, x, dir=-1).truncate()
    # 断言：对 f 在 x = 2 处的泰勒级数展开，展开到指定阶数，应该得到指定的级数表达式
    assert fps(f, x, 2).truncate() == \
        (log(3) - Rational(2, 3) - (x - 2)**2/18 + (x - 2)**3/81 -
         (x - 2)**4/324 + (x - 2)**5/1215 + x/3 + O((x - 2)**6, (x, 2)))
    # 断言：对 f 在 x = 2 处的泰勒级数展开，在逆向方向上的展开结果应该相同
    assert fps(f, x, 2, dir=-1).truncate() == \
        (log(3) - Rational(2, 3) - (-x + 2)**2/18 - (-x + 2)**3/81 -
         (-x + 2)**4/324 - (-x + 2)**5/1215 + x/3 + O((x - 2)**6, (x, 2)))

    # 定义 f = atan(x)
    f = atan(x)
    # 断言：对 f 在 x 处的泰勒级数展开，展开到完全阶数，并截取到指定阶数，应该得到指定的级数表达式
    assert fps(f, x, full=True).truncate() == x - x**3/3 + x**5/5 + O(x**6)
    # 断言：对 f 在 x 处的泰勒级数展开，在两个方向上的展开结果应该相同
    assert fps(f, x, full=True, dir=1).truncate() == \
        fps(f, x, full=True, dir=-1).truncate()
    # 断言：对 f 在 x = 2 处的泰勒级数展开，展开到完全阶数，并截取到指定阶数，应该得到指定的级数表达式
    assert fps(f, x, 2, full=True).truncate() == \
        (atan(2) - Rational(2, 5) - 2*(x - 2)**2/25 + 11*(x - 2)**3/375 -
         6*(x - 2)**4/625 + 41*(x - 2)**5/15625 + x/5 + O((x - 2)**6, (x, 2)))
    # 断言：对 f 在 x = 2 处的泰勒级数展开，在逆向方向上的展开结果应该相同
    assert fps(f, x, 2, full=True, dir=-1).truncate() == \
        (atan(2) - Rational(2, 5) - 2*(-x + 2)**2/25 - 11*(-x + 2)**3/375 -
         6*(-x + 2)**4/625 - 41*(-x + 2)**5/15625 + x/5 + O((x - 2)**6, (x, 2)))

    # 定义 f = x*atan(x) - log(1 + x**2) / 2
    f = x*atan(x) - log(1 + x**2) / 2
    # 断言：对 f 在 x 处的泰勒级数展开，展开到完全阶数，并截取到指定阶数，应该得到指定的级数表达式
    assert fps(f, x, full=True).truncate() == x**2/2 - x**4/12 + O(x**6)

    # 定义 f = log((1 + x) / (1 - x)) / 2 - atan(x)
    f = log((1 + x) / (1 - x)) / 2 - atan(x)
    # 断言：对 f 在 x 处的泰勒级数展开，展开到完全阶数，并截取到指定阶数，应该得到指定的级数表达式，限定展开阶数为 10
    assert fps(f, x, full=True).truncate(n=10) == 2*x**3/3 + 2*x**7/7 + O(x**10)


# 定义一个测试函数，用来测试 fps 函数在另一组函数下的表现
@slow
def test_fps__hyper():
    # 定义 f = sin(x)
    f = sin(x)
    # 断言：对 f 在 x 处的泰勒级数展开，并截取到指定阶数，应
    # 计算并断言给定函数在指定点的泰勒级数展开结果是否符合预期
    
    # 定义函数 f(x) = x * atan(x) - log(1 + x**2) / 2
    f = x*atan(x) - log(1 + x**2) / 2
    # 断言：对 f(x) 进行泰勒级数展开，并截断到 x**5 项，与预期结果进行比较
    assert fps(f, x, rational=False).truncate() == x**2/2 - x**4/12 + O(x**6)
    
    # 定义函数 f(x) = log(1 + x)
    f = log(1 + x)
    # 断言：对 f(x) 进行泰勒级数展开，并截断到 x**5 项，与预期结果进行比较
    assert fps(f, x, rational=False).truncate() == \
        x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
    
    # 定义函数 f(x) = airyai(x**2)
    f = airyai(x**2)
    # 断言：对 f(x) 进行泰勒级数展开，并截断到 x**5 项，与预期结果进行比较
    assert fps(f, x).truncate() == \
        (3**Rational(5, 6)*gamma(Rational(1, 3))/(6*pi) -
         3**Rational(2, 3)*x**2/(3*gamma(Rational(1, 3))) + O(x**6))
    
    # 定义函数 f(x) = exp(x)*sin(x)
    f = exp(x)*sin(x)
    # 断言：对 f(x) 进行泰勒级数展开，并截断到 x**5 项，与预期结果进行比较
    assert fps(f, x).truncate() == x + x**2 + x**3/3 - x**5/30 + O(x**6)
    
    # 定义函数 f(x) = exp(x)*sin(x)/x
    f = exp(x)*sin(x)/x
    # 断言：对 f(x) 进行泰勒级数展开，并截断到 x**5 项，与预期结果进行比较
    assert fps(f, x).truncate() == 1 + x + x**2/3 - x**4/30 - x**5/90 + O(x**6)
    
    # 定义函数 f(x) = sin(x) * cos(x)
    f = sin(x) * cos(x)
    # 断言：对 f(x) 进行泰勒级数展开，并截断到 x**5 项，与预期结果进行比较
    assert fps(f, x).truncate() == x - 2*x**3/3 + 2*x**5/15 + O(x**6)
def test_fps_shift():
    # 定义第一个测试函数，计算 x**-5*sin(x) 的泰勒级数，并断言其截断结果与给定表达式相等
    f = x**-5*sin(x)
    assert fps(f, x).truncate() == \
        1/x**4 - 1/(6*x**2) + Rational(1, 120) - x**2/5040 + x**4/362880 + O(x**6)

    # 定义第二个测试函数，计算 x**2*atan(x) 的泰勒级数，使用非有理数形式，并断言其截断结果与给定表达式相等
    f = x**2*atan(x)
    assert fps(f, x, rational=False).truncate() == \
        x**3 - x**5/3 + O(x**6)

    # 定义第三个测试函数，计算 cos(sqrt(x))*x 的泰勒级数，并断言其截断结果与给定表达式相等
    f = cos(sqrt(x))*x
    assert fps(f, x).truncate() == \
        x - x**2/2 + x**3/24 - x**4/720 + x**5/40320 + O(x**6)

    # 定义第四个测试函数，计算 x**2*cos(sqrt(x)) 的泰勒级数，并断言其截断结果与给定表达式相等
    f = x**2*cos(sqrt(x))
    assert fps(f, x).truncate() == \
        x**2 - x**3/2 + x**4/24 - x**5/720 + O(x**6)


def test_fps__Add_expr():
    # 定义第五个测试函数，计算 x*atan(x) - log(1 + x**2) / 2 的泰勒级数，并断言其截断结果与给定表达式相等
    f = x*atan(x) - log(1 + x**2) / 2
    assert fps(f, x).truncate() == x**2/2 - x**4/12 + O(x**6)

    # 定义第六个测试函数，计算 sin(x) + cos(x) - exp(x) + log(1 + x) 的泰勒级数，并断言其截断结果与给定表达式相等
    f = sin(x) + cos(x) - exp(x) + log(1 + x)
    assert fps(f, x).truncate() == x - 3*x**2/2 - x**4/4 + x**5/5 + O(x**6)

    # 定义第七个测试函数，计算 1/x + sin(x) 的泰勒级数，并断言其截断结果与给定表达式相等
    f = 1/x + sin(x)
    assert fps(f, x).truncate() == 1/x + x - x**3/6 + x**5/120 + O(x**6)

    # 定义第八个测试函数，计算 sin(x) - cos(x) + 1/(x - 1) 的泰勒级数，并断言其截断结果与给定表达式相等
    f = sin(x) - cos(x) + 1/(x - 1)
    assert fps(f, x).truncate() == \
        -2 - x**2/2 - 7*x**3/6 - 25*x**4/24 - 119*x**5/120 + O(x**6)


def test_fps__asymptotic():
    # 定义第九个测试函数，计算 exp(x) 的泰勒级数，分别在无穷远和负无穷远处截断，并断言结果符合预期
    f = exp(x)
    assert fps(f, x, oo) == f
    assert fps(f, x, -oo).truncate() == O(1/x**6, (x, oo))

    # 定义第十个测试函数，计算 erf(x) 的泰勒级数，分别在无穷远和负无穷远处截断，并断言结果符合预期
    f = erf(x)
    assert fps(f, x, oo).truncate() == 1 + O(1/x**6, (x, oo))
    assert fps(f, x, -oo).truncate() == -1 + O(1/x**6, (x, oo))

    # 定义第十一个测试函数，计算 atan(x) 的泰勒级数，分别在无穷远和负无穷远处截断，并断言结果符合预期
    f = atan(x)
    assert fps(f, x, oo, full=True).truncate() == \
        -1/(5*x**5) + 1/(3*x**3) - 1/x + pi/2 + O(1/x**6, (x, oo))
    assert fps(f, x, -oo, full=True).truncate() == \
        -1/(5*x**5) + 1/(3*x**3) - 1/x - pi/2 + O(1/x**6, (x, oo))

    # 定义第十二个测试函数，计算 log(1 + x) 的泰勒级数，分别在无穷远和负无穷远处截断，并断言结果符合预期
    f = log(1 + x)
    assert fps(f, x, oo) != \
        (-1/(5*x**5) - 1/(4*x**4) + 1/(3*x**3) - 1/(2*x**2) + 1/x - log(1/x) +
         O(1/x**6, (x, oo)))
    assert fps(f, x, -oo) != \
        (-1/(5*x**5) - 1/(4*x**4) + 1/(3*x**3) - 1/(2*x**2) + 1/x + I*pi -
         log(-1/x) + O(1/x**6, (x, oo)))


def test_fps__fractional():
    # 定义第十三个测试函数，计算 sin(sqrt(x)) / x 的泰勒级数，并断言其截断结果与给定表达式相等
    f = sin(sqrt(x)) / x
    assert fps(f, x).truncate() == \
        (1/sqrt(x) - sqrt(x)/6 + x**Rational(3, 2)/120 -
         x**Rational(5, 2)/5040 + x**Rational(7, 2)/362880 -
         x**Rational(9, 2)/39916800 + x**Rational(11, 2)/6227020800 + O(x**6))

    # 定义第十四个测试函数，计算 sin(sqrt(x)) * x 的泰勒级数，并断言其截断结果与给定表达式相等
    f = sin(sqrt(x)) * x
    assert fps(f, x).truncate() == \
        (x**Rational(3, 2) - x**Rational(5, 2)/6 + x**Rational(7, 2)/120 -
         x**Rational(9, 2)/5040 + x**Rational(11, 2)/362880 + O(x**6))

    # 定义第十五个测试函数，计算 atan(sqrt(x)) / x**2 的泰勒级数，并断言其截断结果与给定表达式相等
    f = atan(sqrt(x)) / x**2
    assert fps(f, x).truncate() == \
        (x**Rational(-3, 2) - x**Rational(-1, 2)/3 + x**S.Half/5 -
         x**Rational(3, 2)/7 + x**Rational(5, 2)/9 - x**Rational(7, 2)/11 +
         x**Rational(9, 2)/13 - x**Rational(11, 2)/15 + O(x**6))

    # 定义第十六个测试函数，计算 exp(sqrt(x)) 的泰勒级数并展开，并断言其结果与给定表达式相等
    f = exp(sqrt(x))
    assert fps(f, x).truncate().expand() == \
        (1 + x/2 + x**2/24 + x**3/720 + x**4/40320 + x**5/3628800 + sqrt(x) +
         x**Rational(3, 2)/6 + x**Rational(5, 2)/120 + x**Rational(7, 2)/5040 +
         x**Rational(9, 2)/362880 +
    # 断言：验证 fps(f, x).truncate().expand() 的返回值是否等于以下表达式
    assert fps(f, x).truncate().expand() == \
        (x + x**2/2 + x**3/24 + x**4/720 + x**5/40320 + x**Rational(3, 2) +
         x**Rational(5, 2)/6 + x**Rational(7, 2)/120 + x**Rational(9, 2)/5040 +
         x**Rational(11, 2)/362880 + O(x**6))
# 定义测试函数 test_fps__logarithmic_singularity，用于测试对数奇点的生成级数
def test_fps__logarithmic_singularity():
    # 定义函数 f 为 log(1 + 1/x)
    f = log(1 + 1/x)
    # 断言生成级数 fps(f, x) 不等于 -log(x) + x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
    assert fps(f, x) != \
        -log(x) + x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
    # 断言使用非有理模式生成级数 fps(f, x, rational=False) 不等于 -log(x) + x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)


# 标记为 XFAIL 的测试函数 test_fps__logarithmic_singularity_fail，预期测试失败
@XFAIL
def test_fps__logarithmic_singularity_fail():
    # 定义函数 f 为 asech(x)，表示反双曲余弦，注释指出算法可能需要改进
    f = asech(x)  # Algorithms for computing limits probably needs improvements
    # 断言生成级数 fps(f, x) 等于 log(2) - log(x) - x**2/4 - 3*x**4/64 + O(x**6)
    assert fps(f, x) == log(2) - log(x) - x**2/4 - 3*x**4/64 + O(x**6)


# 定义测试函数 test_fps_symbolic，测试符号表达式的生成级数
def test_fps_symbolic():
    # 定义函数 f 为 x**n*sin(x**2)，断言其生成级数的截断版本符合预期
    f = x**n*sin(x**2)
    assert fps(f, x).truncate(8) == x**(n + 2) - x**(n + 6)/6 + O(x**(n + 8), x)

    # 定义函数 f 为 x**n*log(1 + x)，生成级数 fp
    f = x**n*log(1 + x)
    fp = fps(f, x)
    # 提取 fp 的 ak 变量
    k = fp.ak.variables[0]
    # 断言 fp 的无限部分符合预期的和式
    assert fp.infinite == \
        Sum((-(-1)**(-k)*x**(k + n))/k, (k, 1, oo))

    # 定义函数 f 为 (x - 2)**n*log(1 + x)，断言其生成级数的截断版本符合预期
    f = (x - 2)**n*log(1 + x)
    assert fps(f, x, 2).truncate() == \
        ((x - 2)**n*log(3) + (x - 2)**(n + 1)/3 - (x - 2)**(n + 2)/18 + (x - 2)**(n + 3)/81 -
         (x - 2)**(n + 4)/324 + (x - 2)**(n + 5)/1215 + O((x - 2)**(n + 6), (x, 2)))

    # 定义函数 f 为 x**(n - 2)*cos(x)，断言其生成级数的截断版本符合预期
    f = x**(n - 2)*cos(x)
    assert fps(f, x).truncate() == \
        (x**(n - 2) - x**n/2 + x**(n + 2)/24 + O(x**(n + 4), x))

    # 定义函数 f 为 x**(n - 2)*sin(x) + x**n*exp(x)，断言其生成级数的截断版本符合预期
    f = x**(n - 2)*sin(x) + x**n*exp(x)
    assert fps(f, x).truncate() == \
        (x**(n - 1) + x**(n + 1) + x**(n + 2)/2 + x**n +
         x**(n + 4)/24 + x**(n + 5)/60 + O(x**(n + 6), x))

    # 定义函数 f 为 x**n*atan(x)，断言其在 x 趋于无穷时的生成级数的截断版本符合预期
    f = x**(n/2)*cos(x)
    assert fps(f, x).truncate() == \
        x**(n/2) - x**(n/2 + 2)/2 + x**(n/2 + 4)/24 + O(x**(n/2 + 6), x)

    # 定义函数 f 为 x**(n + m)*sin(x)，断言其生成级数的截断版本符合预期
    f = x**(n + m)*sin(x)
    assert fps(f, x).truncate() == \
        x**(m + n + 1) - x**(m + n + 3)/6 + x**(m + n + 5)/120 + O(x**(m + n + 6), x)


# 定义测试函数 test_fps__slow，测试较慢的生成级数情况
def test_fps__slow():
    # 定义函数 f 为 x*exp(x)*sin(2*x)，断言其生成级数的截断版本符合预期
    f = x*exp(x)*sin(2*x)  # TODO: rsolve needs improvement
    assert fps(f, x).truncate() == 2*x**2 + 2*x**3 - x**4/3 - x**5 + O(x**6)


# 定义测试函数 test_fps__operations，测试生成级数的各种操作
def test_fps__operations():
    # 使用 fps 函数分别生成 sin(x) 和 cos(x) 的级数
    f1, f2 = fps(sin(x)), fps(cos(x))

    # 断言级数的和 f1 + f2 的函数部分和截断版本符合预期
    fsum = f1 + f2
    assert fsum.function == sin(x) + cos(x)
    assert fsum.truncate() == \
        1 + x - x**2/2 - x**3/6 + x**4/24 + x**5/120 + O(x**6)

    # 断言 f1 + 1 的函数部分和截断版本符合预期
    fsum = f1 + 1
    assert fsum.function == sin(x) + 1
    assert fsum.truncate() == 1 + x - x**3/6 + x**5/120 + O(x**6)

    # 断言 1 + f2 的函数部分和截断版本符合预期
    fsum = 1 + f2
    assert fsum.function == cos(x) + 1
    assert fsum.truncate() == 2 - x**2/2 + x**4/24 + O(x**6)

    # 断言 f1 + x 的结果等同于 Add(f1, x)
    assert (f1 + x) == Add(f1, x)

    # 断言负级数 -f2 的函数部分和截断版本符合预期
    assert -f2.truncate() == -1 + x**2/2 - x**4/24 + O(x**6)

    # 断言 f1 - f1 的结果是 S.Zero
    assert (f1 - f1) is S.Zero

    # 断言级数的差 f1 - f2 的函数部分和截断版本符合预期
    fsub = f1 - f2
    assert fsub.function == sin(x) - cos(x)
    assert fsub.truncate() == \
        -1 + x + x**2/2 - x**3/6 - x**4/24 + x**5/120 + O(x**6)

    # 断言 f1 - 1 的函数部分和截断版本符合预期
    fsub = f1 - 1
    assert fsub.function == sin(x) - 1
    assert fsub.truncate() == -1 + x - x**3/6 + x**5/120 + O(x**6)

    # 断言 1 - f2 的函数部分和截断版本符合预期
    fsub = 1 - f2
    assert fsub.function == -cos(x) + 1
    assert fsub.truncate() == x**2/2 - x**4/24 + O(x**6)

    # 断言在
    # 使用 raises 函数测试表达式 f1 + fps(exp(x), x0=1)，期望引发 ValueError 异常
    raises(ValueError, lambda: f1 + fps(exp(x), x0=1))

    # 将 f1 乘以常数 3，得到新的表达式 fm
    fm = f1 * 3

    # 断言 fm 的函数部分为 3*sin(x)
    assert fm.function == 3*sin(x)
    # 断言 fm 的截断表达式为 3*x - x**3/2 + x**5/40 + O(x**6)

    # 将 fm 重新赋值为 3*f2
    fm = 3 * f2

    # 断言 fm 的函数部分为 3*cos(x)
    assert fm.function == 3*cos(x)
    # 断言 fm 的截断表达式为 3 - 3*x**2/2 + x**4/8 + O(x**6)

    # 断言 f1 与 f2 的乘积等于 Mul(f1, f2)
    assert (f1 * f2) == Mul(f1, f2)
    # 断言 f1 与 x 的乘积等于 Mul(f1, x)

    # 计算 f1 的导数，并赋值给 fd
    fd = f1.diff()
    # 断言 fd 的函数部分为 cos(x)
    assert fd.function == cos(x)
    # 断言 fd 的截断表达式为 1 - x**2/2 + x**4/24 + O(x**6)

    # 计算 f2 的导数，并赋值给 fd
    fd = f2.diff()
    # 断言 fd 的函数部分为 -sin(x)
    assert fd.function == -sin(x)
    # 断言 fd 的截断表达式为 -x + x**3/6 - x**5/120 + O(x**6)

    # 连续对 f2 求两次导数，并赋值给 fd
    fd = f2.diff().diff()
    # 断言 fd 的函数部分为 -cos(x)
    assert fd.function == -cos(x)
    # 断言 fd 的截断表达式为 -1 + x**2/2 - x**4/24 + O(x**6)

    # 计算 exp(sqrt(x)) 的形式幂级数展开，并赋值给 f3
    f3 = fps(exp(sqrt(x)))
    # 计算 f3 的一阶导数，并赋值给 fd
    fd = f3.diff()
    # 断言 fd 的截断表达式展开结果
    assert fd.truncate().expand() == \
        (1/(2*sqrt(x)) + S.Half + x/12 + x**2/240 + x**3/10080 + x**4/725760 +
         x**5/79833600 + sqrt(x)/4 + x**Rational(3, 2)/48 + x**Rational(5, 2)/1440 +
         x**Rational(7, 2)/80640 + x**Rational(9, 2)/7257600 + x**Rational(11, 2)/958003200 +
         O(x**6))

    # 计算 f1 在区间 (0, 1) 上的定积分，并断言其结果
    assert f1.integrate((x, 0, 1)) == -cos(1) + 1
    # 使用 integrate 函数计算 f1 在区间 (0, 1) 上的定积分，并断言其结果
    assert integrate(f1, (x, 0, 1)) == -cos(1) + 1

    # 对 f1 进行不定积分，并赋值给 fi
    fi = integrate(f1, x)
    # 断言 fi 的函数部分为 -cos(x)
    assert fi.function == -cos(x)
    # 断言 fi 的截断表达式为 -1 + x**2/2 - x**4/24 + O(x**6)

    # 对 f2 进行不定积分，并赋值给 fi
    fi = f2.integrate(x)
    # 断言 fi 的函数部分为 sin(x)
    assert fi.function == sin(x)
    # 断言 fi 的截断表达式为 x - x**3/6 + x**5/120 + O(x**6)
# 定义测试函数 test_fps__product
def test_fps__product():
    # 创建三个形式幂级数对象 f1, f2, f3 分别对应 sin(x), exp(x), cos(x)
    f1, f2, f3 = fps(sin(x)), fps(exp(x)), fps(cos(x))

    # 测试 f1 乘积操作的异常情况：与 exp(x) 的乘积
    raises(ValueError, lambda: f1.product(exp(x), x))
    # 测试 f1 乘积操作的异常情况：与 exp(x) 反向形式幂级数的乘积
    raises(ValueError, lambda: f1.product(fps(exp(x), dir=-1), x, 4))
    # 测试 f1 乘积操作的异常情况：与 exp(x) 中心 x0=1 的形式幂级数的乘积
    raises(ValueError, lambda: f1.product(fps(exp(x), x0=1), x, 4))
    # 测试 f1 乘积操作的异常情况：与 exp(y) 的乘积
    raises(ValueError, lambda: f1.product(fps(exp(y)), x, 4))

    # 对 f1 和 f2 的乘积进行测试
    fprod = f1.product(f2, x)
    assert isinstance(fprod, FormalPowerSeriesProduct)  # 检查乘积结果的类型
    assert isinstance(fprod.ffps, FormalPowerSeries)    # 检查乘积中的第一个形式幂级数的类型
    assert isinstance(fprod.gfps, FormalPowerSeries)    # 检查乘积中的第二个形式幂级数的类型
    assert fprod.f == sin(x)                            # 检查乘积中的第一个函数成分
    assert fprod.g == exp(x)                            # 检查乘积中的第二个函数成分
    assert fprod.function == sin(x) * exp(x)            # 检查乘积函数
    assert fprod._eval_terms(4) == x + x**2 + x**3/3    # 检查乘积的前4项展开结果
    assert fprod.truncate(4) == x + x**2 + x**3/3 + O(x**4)  # 检查乘积的前4项截断结果

    # 测试未实现的方法调用
    raises(NotImplementedError, lambda: fprod._eval_term(5))
    raises(NotImplementedError, lambda: fprod.infinite)
    raises(NotImplementedError, lambda: fprod._eval_derivative(x))
    raises(NotImplementedError, lambda: fprod.integrate(x))

    # 对 f1 和 f3 的乘积进行测试
    assert f1.product(f3, x)._eval_terms(4) == x - 2*x**3/3    # 检查乘积的前4项展开结果
    assert f1.product(f3, x).truncate(4) == x - 2*x**3/3 + O(x**4)  # 检查乘积的前4项截断结果


# 定义测试函数 test_fps__compose
def test_fps__compose():
    # 创建三个形式幂级数对象 f1, f2, f3 分别对应 exp(x), sin(x), cos(x)
    f1, f2, f3 = fps(exp(x)), fps(sin(x)), fps(cos(x))

    # 测试 f1 复合操作的异常情况：与 sin(x) 的复合
    raises(ValueError, lambda: f1.compose(sin(x), x))
    # 测试 f1 复合操作的异常情况：与 sin(x) 反向形式幂级数的复合
    raises(ValueError, lambda: f1.compose(fps(sin(x), dir=-1), x, 4))
    # 测试 f1 复合操作的异常情况：与 sin(x) 中心 x0=1 的形式幂级数的复合
    raises(ValueError, lambda: f1.compose(fps(sin(x), x0=1), x, 4))
    # 测试 f1 复合操作的异常情况：与 sin(y) 的复合
    raises(ValueError, lambda: f1.compose(fps(sin(y)), x, 4))

    # 测试 f1 和 f3 的复合操作的异常情况
    raises(ValueError, lambda: f1.compose(f3, x))
    raises(ValueError, lambda: f2.compose(f3, x))

    # 对 f1 和 f2 的复合操作进行测试
    fcomp = f1.compose(f2, x)
    assert isinstance(fcomp, FormalPowerSeriesCompose)  # 检查复合结果的类型
    assert isinstance(fcomp.ffps, FormalPowerSeries)    # 检查复合中的第一个形式幂级数的类型
    assert isinstance(fcomp.gfps, FormalPowerSeries)    # 检查复合中的第二个形式幂级数的类型
    assert fcomp.f == exp(x)                            # 检查复合中的第一个函数成分
    assert fcomp.g == sin(x)                            # 检查复合中的第二个函数成分
    assert fcomp.function == exp(sin(x))                # 检查复合函数
    assert fcomp._eval_terms(6) == 1 + x + x**2/2 - x**4/8 - x**5/15  # 检查复合的前6项展开结果
    assert fcomp.truncate() == 1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)  # 检查复合的截断结果
    assert fcomp.truncate(5) == 1 + x + x**2/2 - x**4/8 + O(x**5)  # 检查复合的前5项截断结果

    # 测试未实现的方法调用
    raises(NotImplementedError, lambda: fcomp._eval_term(5))
    raises(NotImplementedError, lambda: fcomp.infinite)
    raises(NotImplementedError, lambda: fcomp._eval_derivative(x))
    raises(NotImplementedError, lambda: fcomp.integrate(x))

    # 对 f1 和 f2 的复合操作的截断结果进行测试
    assert f1.compose(f2, x).truncate(4) == 1 + x + x**2/2 + O(x**4)
    assert f1.compose(f2, x).truncate(8) == \
        1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + O(x**8)
    assert f1.compose(f2, x).truncate(6) == \
        1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)

    # 对 f2 和 f2 的复合操作的截断结果进行测试
    assert f2.compose(f2, x).truncate(4) == x - x**3/3 + O(x**4)
    assert f2.compose(f2, x).truncate(8) == x - x**3/3 + x**5/10 - 8*x**7/315 + O(x**8)
    assert f2.compose(f2, x).truncate(6) == x - x**3/3 + x**5/10 + O(x**6)


# 定义测试函数 test_fps__inverse
def test_fps__inverse():
    # 创建三个形式幂级数对象 f1, f2, f3 分别对应 sin(x), exp(x), cos(x)

    raises(ValueError, lambda: f1.inverse(x))  # 测试 f1 的逆操作的异常情况
    # 计算 f2 对 x 的逆序形式
    finv = f2.inverse(x)
    # 断言 finv 是 FormalPowerSeriesInverse 类型的实例
    assert isinstance(finv, FormalPowerSeriesInverse)
    # 断言 finv.ffps 是 FormalPowerSeries 类型的实例
    assert isinstance(finv.ffps, FormalPowerSeries)
    # 断言调用 lambda 函数会引发 ValueError 异常，因为 finv.gfps 不存在
    raises(ValueError, lambda: finv.gfps)

    # 断言 finv.f 等于 exp(x)
    assert finv.f == exp(x)
    # 断言 finv.function 等于 exp(-x)
    assert finv.function == exp(-x)
    # 断言 finv._eval_terms(5) 的结果符合给定的幂级数展开式
    assert finv._eval_terms(5) == 1 - x + x**2/2 - x**3/6 + x**4/24
    # 断言 finv.truncate() 的结果符合给定的幂级数截断形式
    assert finv.truncate() == 1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + O(x**6)
    # 断言 finv.truncate(5) 的结果符合给定的幂级数截断形式
    assert finv.truncate(5) == 1 - x + x**2/2 - x**3/6 + x**4/24 + O(x**5)

    # 断言调用 lambda 函数会引发 NotImplementedError 异常，因为 _eval_term 方法未实现
    raises(NotImplementedError, lambda: finv._eval_term(5))
    # 断言调用 lambda 函数会引发 ValueError 异常，因为 finv.g 方法不存在
    raises(ValueError, lambda: finv.g)
    # 断言调用 lambda 函数会引发 NotImplementedError 异常，因为 finv.infinite 方法未实现
    raises(NotImplementedError, lambda: finv.infinite)
    # 断言调用 lambda 函数会引发 NotImplementedError 异常，因为 finv._eval_derivative 方法未实现
    raises(NotImplementedError, lambda: finv._eval_derivative(x))
    # 断言调用 lambda 函数会引发 NotImplementedError 异常，因为 finv.integrate 方法未实现

    # 断言 f2 对 x 的逆序形式在截断到 8 阶时，符合给定的幂级数展开式
    assert f2.inverse(x).truncate(8) == \
        1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + x**6/720 - x**7/5040 + O(x**8)

    # 断言 f3 对 x 的逆序形式在默认截断时，符合给定的幂级数展开式
    assert f3.inverse(x).truncate() == 1 + x**2/2 + 5*x**4/24 + O(x**6)
    # 断言 f3 对 x 的逆序形式在截断到 8 阶时，符合给定的幂级数展开式
    assert f3.inverse(x).truncate(8) == 1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8)
```