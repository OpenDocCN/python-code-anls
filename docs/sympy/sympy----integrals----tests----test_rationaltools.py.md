# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_rationaltools.py`

```
# 导入需要的 SymPy 模块和函数

from sympy.core.numbers import (I, Rational)  # 导入虚数单位和有理数类
from sympy.core.singleton import S  # 导入 SymPy 的单例对象
from sympy.core.symbol import (Dummy, symbols)  # 导入虚拟符号和符号类
from sympy.functions.elementary.exponential import log  # 导入对数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import atan  # 导入反正切函数
from sympy.integrals.integrals import integrate  # 导入积分函数
from sympy.polys.polytools import Poly  # 导入多项式工具函数
from sympy.simplify.simplify import simplify  # 导入简化函数

from sympy.integrals.rationaltools import ratint, ratint_logpart, log_to_atan  # 导入有理函数积分相关函数

from sympy.abc import a, b, x, t  # 导入 SymPy 中常用的符号

half = S.Half  # 定义 SymPy 中的 1/2 作为 half

# 定义测试函数 test_ratint
def test_ratint():
    # 测试有理函数积分函数 ratint 的基本性质
    assert ratint(S.Zero, x) == 0
    assert ratint(S(7), x) == 7*x

    # 测试一次幂的积分
    assert ratint(x, x) == x**2/2
    assert ratint(2*x, x) == x**2
    assert ratint(-2*x, x) == -x**2

    # 测试多项式的积分
    assert ratint(8*x**7 + 2*x + 1, x) == x**8 + x**2 + x

    # 测试分式的积分
    f = S.One
    g = x + 1
    assert ratint(f / g, x) == log(x + 1)
    assert ratint((f, g), x) == log(x + 1)

    # 更复杂的分式积分测试
    f = x**3 - x
    g = x - 1
    assert ratint(f/g, x) == x**3/3 + x**2/2

    # 更复杂的分式积分测试
    f = x
    g = (x - a)*(x + a)
    assert ratint(f/g, x) == log(x**2 - a**2)/2

    # 测试含有复数解的分式积分
    f = S.One
    g = x**2 + 1
    assert ratint(f/g, x, real=None) == atan(x)
    assert ratint(f/g, x, real=True) == atan(x)
    assert ratint(f/g, x, real=False) == I*log(x + I)/2 - I*log(x - I)/2

    # 更复杂的分式积分测试
    f = S(36)
    g = x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2
    assert ratint(f/g, x) == \
        -4*log(x + 1) + 4*log(x - 2) + (12*x + 6)/(x**2 - 1)

    # 更复杂的分式积分测试
    f = x**4 - 3*x**2 + 6
    g = x**6 - 5*x**4 + 5*x**2 + 4
    assert ratint(f/g, x) == \
        atan(x) + atan(x**3) + atan(x/2 - Rational(3, 2)*x**3 + S.Half*x**5)

    # 更复杂的分式积分测试
    f = x**7 - 24*x**4 - 4*x**2 + 8*x - 8
    g = x**8 + 6*x**6 + 12*x**4 + 8*x**2
    assert ratint(f/g, x) == \
        (4 + 6*x + 8*x**2 + 3*x**3)/(4*x + 4*x**3 + x**5) + log(x)

    # 更复杂的分式积分测试
    assert ratint((x**3*f)/(x*g), x) == \
        -(12 - 16*x + 6*x**2 - 14*x**3)/(4 + 4*x**2 + x**4) - \
        5*sqrt(2)*atan(x*sqrt(2)/2) + S.Half*x**2 - 3*log(2 + x**2)

    # 更复杂的分式积分测试
    f = x**5 - x**4 + 4*x**3 + x**2 - x + 5
    g = x**4 - 2*x**3 + 5*x**2 - 4*x + 4
    assert ratint(f/g, x) == \
        x + S.Half*x**2 + S.Half*log(2 - x + x**2) + (9 - 4*x)/(7*x**2 - 7*x + 14) + \
        13*sqrt(7)*atan(Rational(-1, 7)*sqrt(7) + 2*x*sqrt(7)/7)/49

    # 更复杂的分式积分测试
    assert ratint(1/(x**2 + x + 1), x) == \
        2*sqrt(3)*atan(sqrt(3)/3 + 2*x*sqrt(3)/3)/3

    # 更复杂的分式积分测试
    assert ratint(1/(x**3 + 1), x) == \
        -log(1 - x + x**2)/6 + log(1 + x)/3 + sqrt(3)*atan(-sqrt(3)/3 + 2*x*sqrt(3)/3)/3

    # 更复杂的分式积分测试
    assert ratint(1/(x**2 + x + 1), x, real=False) == \
        -I*3**half*log(half + x - half*I*3**half)/3 + \
        I*3**half*log(half + x + half*I*3**half)/3

    # 更复杂的分式积分测试
    assert ratint(1/(x**3 + 1), x, real=False) == log(1 + x)/3 + \
        (Rational(-1, 6) + I*3**half/6)*log(-half + x + I*3**half/2) + \
        (Rational(-1, 6) - I*3**half/6)*log(-half + x - I*3**half/2)

    # issue 4991
    # 断言：使用 ratint 函数对表达式进行积分，验证结果是否等于指定的表达式
    assert ratint(1/(x*(a + b*x)**3), x) == \
        (3*a + 2*b*x)/(2*a**4 + 4*a**3*b*x + 2*a**2*b**2*x**2) + (
            log(x) - log(a/b + x))/a**3

    # 断言：使用 ratint 函数对表达式 x/(1 - x**2) 进行积分，验证结果是否等于 -log(x**2 - 1)/2
    assert ratint(x/(1 - x**2), x) == -log(x**2 - 1)/2

    # 断言：使用 ratint 函数对表达式 -x/(1 - x**2) 进行积分，验证结果是否等于 log(x**2 - 1)/2
    assert ratint(-x/(1 - x**2), x) == log(x**2 - 1)/2

    # 断言：使用 ratint 函数对表达式 (x/4 - 4/(1 - x)).diff(x) 进行积分，验证结果是否等于 x/4 + 4/(x - 1)
    assert ratint((x/4 - 4/(1 - x)).diff(x), x) == x/4 + 4/(x - 1)

    # 计算 atan(x) 并赋值给 ans
    ans = atan(x)
    # 断言：使用 ratint 函数对表达式 1/(x**2 + 1) 进行积分，验证结果是否等于 ans，其中 symbol=x
    assert ratint(1/(x**2 + 1), x, symbol=x) == ans
    # 断言：使用 ratint 函数对表达式 1/(x**2 + 1) 进行积分，验证结果是否等于 ans，其中 symbol='x'
    assert ratint(1/(x**2 + 1), x, symbol='x') == ans
    # 断言：使用 ratint 函数对表达式 1/(x**2 + 1) 进行积分，验证结果是否等于 ans，其中 symbol=a
    assert ratint(1/(x**2 + 1), x, symbol=a) == ans
    # 断言：使用 ratint 函数对表达式 1/(d**2 + 1) 进行积分，验证结果是否等于 atan(d)，d 是一个 Dummy 符号
    # 这个断言验证了即使 symbol 是一个 Dummy 符号，ratint 也能返回唯一的符号作为结果
    d = Dummy()
    assert ratint(1/(d**2 + 1), d, symbol=d) == atan(d)
def test_ratint_logpart():
    # 测试函数 `ratint_logpart` 的断言
    assert ratint_logpart(x, x**2 - 9, x, t) == [(Poly(x**2 - 9, x), Poly(-2*t + 1, t))]
    # 测试函数 `ratint_logpart` 的断言
    assert ratint_logpart(x**2, x**3 - 5, x, t) == [(Poly(x**3 - 5, x), Poly(-3*t + 1, t))]


def test_issue_5414():
    # 测试 `ratint` 函数对于 1/(x**2 + 16) 的结果是否等于 atan(x/4)/4
    assert ratint(1/(x**2 + 16), x) == atan(x/4)/4


def test_issue_5249():
    # 测试 `ratint` 函数对于 1/(x**2 + a**2) 的结果是否等于 (-I*log(-I*a + x)/2 + I*log(I*a + x)/2)/a
    assert ratint(1/(x**2 + a**2), x) == (-I*log(-I*a + x)/2 + I*log(I*a + x)/2)/a


def test_issue_5817():
    # 定义符号 a, b, c，并测试 `ratint` 函数对于 a/(b*c*x**2 + a**2 + b*a) 的简化结果
    a, b, c = symbols('a,b,c', positive=True)
    assert simplify(ratint(a/(b*c*x**2 + a**2 + b*a), x)) == sqrt(a)*atan(sqrt(b)*sqrt(c)*x/(sqrt(a)*sqrt(a + b)))/(sqrt(b)*sqrt(c)*sqrt(a + b))


def test_issue_5981():
    # 定义符号 u，并测试 `integrate` 函数对于 1/(u**2 + 1) 的结果是否等于 atan(u)
    u = symbols('u')
    assert integrate(1/(u**2 + 1)) == atan(u)


def test_issue_10488():
    # 定义符号 a, b, c, x，并测试 `integrate` 函数对于 x/(a*x+b) 的结果是否等于 x/a - b*log(a*x + b)/a**2
    a, b, c, x = symbols('a b c x', positive=True)
    assert integrate(x/(a*x+b), x) == x/a - b*log(a*x + b)/a**2


def test_issues_8246_12050_13501_14080():
    # 定义符号 a，并测试 `integrate` 函数对于 a/(x**2 + a**2) 的结果是否等于 atan(x/a)
    a = symbols('a', nonzero=True)
    assert integrate(a/(x**2 + a**2), x) == atan(x/a)
    # 测试 `integrate` 函数对于 1/(x**2 + a**2) 的结果是否等于 atan(x/a)/a
    assert integrate(1/(x**2 + a**2), x) == atan(x/a)/a
    # 测试 `integrate` 函数对于 1/(1 + a**2*x**2) 的结果是否等于 atan(a*x)/a
    assert integrate(1/(1 + a**2*x**2), x) == atan(a*x)/a


def test_issue_6308():
    # 定义符号 k, a0，并测试 `integrate` 函数对于 (x**2 + 1 - k**2)/(x**2 + 1 + a0**2) 的结果
    k, a0 = symbols('k a0', real=True)
    assert integrate((x**2 + 1 - k**2)/(x**2 + 1 + a0**2), x) == \
        x - (a0**2 + k**2)*atan(x/sqrt(a0**2 + 1))/sqrt(a0**2 + 1)


def test_issue_5907():
    # 定义符号 a，并测试 `integrate` 函数对于 1/(x**2 + a**2)**2 的结果
    a = symbols('a', nonzero=True)
    assert integrate(1/(x**2 + a**2)**2, x) == \
         x/(2*a**4 + 2*a**2*x**2) + atan(x/a)/(2*a**3)


def test_log_to_atan():
    # 定义多项式 f, g，并测试 `log_to_atan` 函数对它们的处理结果是否符合预期
    f, g = (Poly(x + S.Half, x, domain='QQ'), Poly(sqrt(3)/2, x, domain='EX'))
    fg_ans = 2*atan(2*sqrt(3)*x/3 + sqrt(3)/3)
    assert log_to_atan(f, g) == fg_ans
    assert log_to_atan(g, f) == -fg_ans


def test_issue_25896():
    # 测试 `ratint` 函数对于两个不同表达式的处理结果
    # 第一个断言测试特定表达式的结果是否为 log(x**3 + x**2 + x)
    e = (2*x + 1)/(x**2 + x + 1) + 1/x
    assert ratint(e, x) == log(x**3 + x**2 + x)
    # 第二个断言测试特定表达式的结果是否为具有更复杂表达式的集合
    assert ratint((4*x + 7)/(x**2 + 4*x + 6) + 2/x, x) == (
        2*log(x) + 2*log(x**2 + 4*x + 6) - sqrt(2)*atan(
        sqrt(2)*x/2 + sqrt(2))/2)
```