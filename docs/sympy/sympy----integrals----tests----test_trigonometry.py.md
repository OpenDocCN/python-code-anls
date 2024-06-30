# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_trigonometry.py`

```
# 导入必要的符号和函数模块
from sympy.core import Ne, Rational, Symbol
from sympy.functions import sin, cos, tan, csc, sec, cot, log, Piecewise
from sympy.integrals.trigonometry import trigintegrate

# 定义符号变量 x
x = Symbol('x')

# 定义测试函数 test_trigintegrate_odd，用于测试三角函数的积分
def test_trigintegrate_odd():
    # 断言：常数 1 的积分应为 x
    assert trigintegrate(Rational(1), x) == x
    # 断言：x 的积分应为 None
    assert trigintegrate(x, x) is None
    # 断言：x^2 的积分应为 None
    assert trigintegrate(x**2, x) is None

    # 断言：sin(x) 的积分应为 -cos(x)
    assert trigintegrate(sin(x), x) == -cos(x)
    # 断言：cos(x) 的积分应为 sin(x)
    assert trigintegrate(cos(x), x) == sin(x)

    # 断言：sin(3*x) 的积分应为 -cos(3*x)/3
    assert trigintegrate(sin(3*x), x) == -cos(3*x)/3
    # 断言：cos(3*x) 的积分应为 sin(3*x)/3
    assert trigintegrate(cos(3*x), x) == sin(3*x)/3

    # 定义符号变量 y
    y = Symbol('y')
    # 断言：sin(y*x) 的积分应为 Piecewise((-cos(y*x)/y, y ≠ 0), (0, True))
    assert trigintegrate(sin(y*x), x) == Piecewise((-cos(y*x)/y, Ne(y, 0)), (0, True))
    # 断言：cos(y*x) 的积分应为 Piecewise((sin(y*x)/y, y ≠ 0), (x, True))
    assert trigintegrate(cos(y*x), x) == Piecewise((sin(y*x)/y, Ne(y, 0)), (x, True))
    # 断言：sin(y*x)**2 的积分应为 Piecewise(((x*y/2 - sin(x*y)*cos(x*y)/2)/y, y ≠ 0), (0, True))
    assert trigintegrate(sin(y*x)**2, x) == Piecewise(((x*y/2 - sin(x*y)*cos(x*y)/2)/y, Ne(y, 0)), (0, True))
    # 断言：sin(y*x)*cos(y*x) 的积分应为 Piecewise((sin(x*y)**2/(2*y), y ≠ 0), (0, True))
    assert trigintegrate(sin(y*x)*cos(y*x), x) == Piecewise((sin(x*y)**2/(2*y), Ne(y, 0)), (0, True))
    # 断言：cos(y*x)**2 的积分应为 Piecewise(((x*y/2 + sin(x*y)*cos(x*y)/2)/y, y ≠ 0), (x, True))
    assert trigintegrate(cos(y*x)**2, x) == Piecewise(((x*y/2 + sin(x*y)*cos(x*y)/2)/y, Ne(y, 0)), (x, True))

    # 定义符号变量 y，并假设 y 是正数
    y = Symbol('y', positive=True)
    # TODO: remove conds='none' below. For this to work we would have to rule
    #       out (e.g. by trying solve) the condition y = 0, incompatible with
    #       y.is_positive being True.
    # 断言：sin(y*x) 的积分应为 -cos(y*x)/y
    assert trigintegrate(sin(y*x), x, conds='none') == -cos(y*x)/y
    # 断言：cos(y*x) 的积分应为 sin(y*x)/y
    assert trigintegrate(cos(y*x), x, conds='none') == sin(y*x)/y

    # 断言：sin(x)*cos(x) 的积分应为 sin(x)**2/2
    assert trigintegrate(sin(x)*cos(x), x) == sin(x)**2/2
    # 断言：sin(x)*cos(x)**2 的积分应为 -cos(x)**3/3
    assert trigintegrate(sin(x)*cos(x)**2, x) == -cos(x)**3/3
    # 断言：sin(x)**2*cos(x) 的积分应为 sin(x)**3/3
    assert trigintegrate(sin(x)**2*cos(x), x) == sin(x)**3/3

    # 断言：sin(x)**7 * cos(x) 的积分应为 sin(x)**8/8
    assert trigintegrate(sin(x)**7 * cos(x), x) == sin(x)**8/8
    # 断言：sin(x) * cos(x)**7 的积分应为 -cos(x)**8/8
    assert trigintegrate(sin(x) * cos(x)**7, x) == -cos(x)**8/8

    # 断言：sin(x)**7 * cos(x)**3 的积分应为 -sin(x)**10/10 + sin(x)**8/8
    assert trigintegrate(sin(x)**7 * cos(x)**3, x) == -sin(x)**10/10 + sin(x)**8/8
    # 断言：sin(x)**3 * cos(x)**7 的积分应为 cos(x)**10/10 - cos(x)**8/8
    assert trigintegrate(sin(x)**3 * cos(x)**7, x) == cos(x)**10/10 - cos(x)**8/8

    # 断言：sin(x)**-1*cos(x)**-1 的积分应为 -log(sin(x)**2 - 1)/2 + log(sin(x))
    assert trigintegrate(sin(x)**-1*cos(x)**-1, x) == -log(sin(x)**2 - 1)/2 + log(sin(x))


# 定义测试函数 test_trigintegrate_even，用于测试偶数幂次的三角函数的积分
def test_trigintegrate_even():
    # 断言：sin(x)**2 的积分应为 x/2 - cos(x)*sin(x)/2
    assert trigintegrate(sin(x)**2, x) == x/2 - cos(x)*sin(x)/2
    # 断言：cos(x)**2 的积分应为 x/2 + cos(x)*sin(x)/2
    assert trigintegrate(cos(x)**2, x) == x/2 + cos(x)*sin(x)/2

    # 断言：sin(3*x)**2 的积分应为 x/2 - cos(3*x)*sin(3*x)/6
    assert trigintegrate(sin(3*x)**2, x) == x/2 - cos(3*x)*sin(3*x)/6
    # 断言：cos(3*x)**2 的积分应为 x/2 + cos(3*x)*sin(3*x)/6
    assert trigintegrate(cos(3*x)**2, x) == x/2 + cos(3*x)*sin(3*x)/6
    # 断言：sin(x)**2 * cos(x)**2 的积分应为 x/8 - sin(2*x)*cos(2*x)/16
    assert trigintegrate(sin(x)**2 * cos(x)**2, x) == x/8 - sin(2*x)*cos(2*x)/16

    # 断言：sin(x)**4 * cos(x)**2 的积分应为 x/16 - sin(x)*cos(x)/16 - sin(x)**3*cos(x)/24 + sin(x)**5*cos(x)/6
    assert trigintegrate(sin(x)**4 * cos(x)**2, x) == x/16 - sin(x)*cos(x)/16 - sin(x)**3*cos(x)/24 + sin(x)**5*cos(x)/6

    # 断言：sin(x)**2 * cos(x)**4 的积分应为 x/16 + cos(x)*sin(x)/16 + cos(x)**3*sin(x)/24 - cos(x)**5*sin(x)/6
    assert trigintegrate(sin(x)**2 * cos(x)**4, x) == x/16 + cos(x)*sin(x)/16 + cos(x)**3*sin(x)/24 - cos(x)**5*sin(x)/6

    # 断言：sin(x)**(-4) 的积分应为 -2*cos(x)/(3*sin(x)) - cos(x)/(3*sin(x)**3)
    assert trigintegrate(sin(x)**(-4), x) == -2*cos(x)/(3*sin(x)) - cos(x)/(3*sin(x)**3)
    # 使用 assert 语句来检查 trigintegrate 函数对 cos(x)**(-6) 的积分是否正确
    assert trigintegrate(cos(x)**(-6), x) == sin(x)/(5*cos(x)**5) \
        + 4*sin(x)/(15*cos(x)**3) + 8*sin(x)/(15*cos(x))
# 定义一个测试函数，用于测试 trigintegrate 函数对混合三角函数积分的结果
def test_trigintegrate_mixed():
    # 断言：对 sin(x) * sec(x) 的积分结果应为 -log(cos(x))
    assert trigintegrate(sin(x)*sec(x), x) == -log(cos(x))
    # 断言：对 sin(x) * csc(x) 的积分结果应为 x
    assert trigintegrate(sin(x)*csc(x), x) == x
    # 断言：对 sin(x) * cot(x) 的积分结果应为 sin(x)
    assert trigintegrate(sin(x)*cot(x), x) == sin(x)

    # 断言：对 cos(x) * sec(x) 的积分结果应为 x
    assert trigintegrate(cos(x)*sec(x), x) == x
    # 断言：对 cos(x) * csc(x) 的积分结果应为 log(sin(x))
    assert trigintegrate(cos(x)*csc(x), x) == log(sin(x))
    # 断言：对 cos(x) * tan(x) 的积分结果应为 -cos(x)
    assert trigintegrate(cos(x)*tan(x), x) == -cos(x)
    # 断言：对 cos(x) * cot(x) 的积分结果应为 log((cos(x) - 1) / (cos(x) + 1)) / 2 + cos(x)
    assert trigintegrate(cos(x)*cot(x), x) == log(cos(x) - 1)/2 - log(cos(x) + 1)/2 + cos(x)
    # 断言：对 cot(x) * cos(x)^2 的积分结果应为 log(sin(x)) - sin(x)^2 / 2
    assert trigintegrate(cot(x)*cos(x)**2, x) == log(sin(x)) - sin(x)**2/2


# 定义另一个测试函数，用于测试 trigintegrate 函数对符号参数的情况
def test_trigintegrate_symbolic():
    # 定义一个符号变量 n，限定为整数
    n = Symbol('n', integer=True)
    # 断言：对 cos(x)^n 的积分结果应为 None
    assert trigintegrate(cos(x)**n, x) is None
    # 断言：对 sin(x)^n 的积分结果应为 None
    assert trigintegrate(sin(x)**n, x) is None
    # 断言：对 cot(x)^n 的积分结果应为 None
    assert trigintegrate(cot(x)**n, x) is None
```