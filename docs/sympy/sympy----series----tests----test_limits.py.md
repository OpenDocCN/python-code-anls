# `D:\src\scipysrc\sympy\sympy\series\tests\test_limits.py`

```
# 导入product函数，用于生成迭代器的笛卡尔积
from itertools import product

# 导入Sympy库中的Sum类，用于表示求和表达式
from sympy.concrete.summations import Sum
# 导入Sympy库中的Function和diff函数，用于函数操作和求导
from sympy.core.function import (Function, diff)
# 导入Sympy库中的常数EulerGamma和GoldenRatio
from sympy.core import EulerGamma, GoldenRatio
# 导入Sympy库中的Mod类，用于模运算
from sympy.core.mod import Mod
# 导入Sympy库中的常数E, I, Rational, oo, pi, zoo
from sympy.core.numbers import (E, I, Rational, oo, pi, zoo)
# 导入Sympy库中的S类，用于表示单例对象
from sympy.core.singleton import S
# 导入Sympy库中的Symbol和symbols类，用于符号的定义
from sympy.core.symbol import (Symbol, symbols)
# 导入Sympy库中的斐波那契数列和阶乘相关函数
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.combinatorial.factorials import (binomial, factorial, subfactorial)
# 导入Sympy库中的复数函数
from sympy.functions.elementary.complexes import (Abs, re, sign)
# 导入Sympy库中的指数函数和对数函数
from sympy.functions.elementary.exponential import (LambertW, exp, log)
# 导入Sympy库中的双曲函数
from sympy.functions.elementary.hyperbolic import (atanh, asinh, acosh, acoth, acsch, asech, tanh, sinh)
# 导入Sympy库中的整数函数
from sympy.functions.elementary.integers import (ceiling, floor, frac)
# 导入Sympy库中的立方根函数和实数根函数
from sympy.functions.elementary.miscellaneous import (cbrt, real_root, sqrt)
# 导入Sympy库中的分段函数
from sympy.functions.elementary.piecewise import Piecewise
# 导入Sympy库中的三角函数
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin,
                                                      atan, cos, cot, csc, sec, sin, tan)
# 导入Sympy库中的贝塞尔函数
from sympy.functions.special.bessel import (besseli, bessely, besselj, besselk)
# 导入Sympy库中的误差函数
from sympy.functions.special.error_functions import (Ei, erf, erfc, erfi, fresnelc, fresnels)
# 导入Sympy库中的Gamma函数
from sympy.functions.special.gamma_functions import (digamma, gamma, uppergamma)
# 导入Sympy库中的Meijer G 函数
from sympy.functions.special.hyper import meijerg
# 导入Sympy库中的积分相关类和函数
from sympy.integrals.integrals import (Integral, integrate)
# 导入Sympy库中的极限相关类和函数
from sympy.series.limits import (Limit, limit)
# 导入Sympy库中的简化函数
from sympy.simplify.simplify import (logcombine, simplify)
# 导入Sympy库中的超几何函数展开函数
from sympy.simplify.hyperexpand import hyperexpand

# 导入Sympy库中的积累边界计算类
from sympy.calculus.accumulationbounds import AccumBounds
# 导入Sympy库中的乘法类
from sympy.core.mul import Mul
# 导入Sympy库中的极限启发式方法
from sympy.series.limits import heuristics
# 导入Sympy库中的阶数类
from sympy.series.order import Order
# 导入Sympy库中的测试相关函数
from sympy.testing.pytest import XFAIL, raises

# 导入Sympy库中的椭圆积分函数
from sympy import elliptic_e, elliptic_k

# 导入Sympy库中的常用符号
from sympy.abc import x, y, z, k
# 定义一个整数符号n
n = Symbol('n', integer=True, positive=True)


# 定义一个测试函数test_basic1，用于测试极限计算
def test_basic1():
    # 断言语句：计算x趋向无穷大时的极限
    assert limit(x, x, oo) is oo
    # 断言语句：计算x趋向负无穷大时的极限
    assert limit(x, x, -oo) is -oo
    # 断言语句：计算-x趋向无穷大时的极限
    assert limit(-x, x, oo) is -oo
    # 断言语句：计算x^2趋向负无穷大时的极限
    assert limit(x**2, x, -oo) is oo
    # 断言语句：计算-x^2趋向无穷大时的极限
    assert limit(-x**2, x, oo) is -oo
    # 断言语句：计算x*log(x)在x趋向0时的极限，从正方向接近
    assert limit(x*log(x), x, 0, dir="+") == 0
    # 断言语句：计算1/x在x趋向无穷大时的极限
    assert limit(1/x, x, oo) == 0
    # 断言语句：计算exp(x)在x趋向无穷大时的极限
    assert limit(exp(x), x, oo) is oo
    # 断言语句：计算-exp(x)在x趋向无穷大时的极限
    assert limit(-exp(x), x, oo) is -oo
    # 断言语句：计算exp(x)/x在x趋向无穷大时的极限
    assert limit(exp(x)/x, x, oo) is oo
    # 断言语句：计算1/x - exp(-x)在x趋向无穷大时的极限
    assert limit(1/x - exp(-x), x, oo) == 0
    # 断言语句：计算x + 1/x在x趋向无穷大时的极限
    assert limit(x + 1/x, x, oo) is oo
    # 断言语句：计算x - x^2在x趋向无穷大时的极限
    assert limit(x - x**2, x, oo) is -oo
    # 断言语句：计算(1 + x)^(1 + sqrt(2))在x趋向0时的极限
    assert limit((1 + x)**(1 + sqrt(2)), x, 0) == 1
    # 断言语句：计算(1 + x)^oo在x趋向0时的极限
    assert limit((1 + x)**oo, x, 0) == Limit((x + 1)**oo, x, 0)
    # 断言语句：计算(1 + x)^oo在x趋向0时的极限，从负方向接近
    assert limit((1 + x)**oo, x, 0, dir='-') == Limit((x + 1)**oo, x, 0, dir='-')
    # 断言语句：计算(1 + x + y)^oo在x趋向0时的极限，从负方向接近
    assert limit((1 + x + y)**oo, x, 0, dir='-') == Limit((1 + x + y)**oo, x, 0, dir='-')
    # 断言语句：计算y/x/log(x)在x趋向0时的极限
    assert limit(y/x/log(x), x, 0) == -oo*sign(y)
    # 断言语句：计算cos(x + y)/x在x趋向0时的极限
    assert limit(cos(x + y)/x, x, 0) == sign(cos(y))*
    # 断言：当 x 趋向于无穷大时，Order(2)*x 的极限为 S.NaN
    assert limit(Order(2)*x, x, S.NaN) is S.NaN

    # 断言：当 x 从正方向趋近于 1 时，1/(x - 1) 的极限为正无穷大
    assert limit(1/(x - 1), x, 1, dir="+") is oo

    # 断言：当 x 从负方向趋近于 1 时，1/(x - 1) 的极限为负无穷大
    assert limit(1/(x - 1), x, 1, dir="-") is -oo

    # 断言：当 x 从正方向趋近于 5 时，1/(5 - x)**3 的极限为负无穷大
    assert limit(1/(5 - x)**3, x, 5, dir="+") is -oo

    # 断言：当 x 从负方向趋近于 5 时，1/(5 - x)**3 的极限为正无穷大
    assert limit(1/(5 - x)**3, x, 5, dir="-") is oo

    # 断言：当 x 从正方向趋近于 pi 时，1/sin(x) 的极限为负无穷大
    assert limit(1/sin(x), x, pi, dir="+") is -oo

    # 断言：当 x 从负方向趋近于 pi 时，1/sin(x) 的极限为正无穷大
    assert limit(1/sin(x), x, pi, dir="-") is oo

    # 断言：当 x 从正方向趋近于 pi/2 时，1/cos(x) 的极限为负无穷大
    assert limit(1/cos(x), x, pi/2, dir="+") is -oo

    # 断言：当 x 从负方向趋近于 pi/2 时，1/cos(x) 的极限为正无穷大
    assert limit(1/cos(x), x, pi/2, dir="-") is oo

    # 断言：当 x 从正方向趋近于 (2*pi)**Rational(1, 3) 时，1/tan(x**3) 的极限为正无穷大
    assert limit(1/tan(x**3), x, (2*pi)**Rational(1, 3), dir="+") is oo

    # 断言：当 x 从负方向趋近于 (2*pi)**Rational(1, 3) 时，1/tan(x**3) 的极限为负无穷大
    assert limit(1/tan(x**3), x, (2*pi)**Rational(1, 3), dir="-") is -oo

    # 断言：当 x 从正方向趋近于 pi*Rational(3, 2) 时，1/cot(x)**3 的极限为负无穷大
    assert limit(1/cot(x)**3, x, (pi*Rational(3, 2)), dir="+") is -oo

    # 断言：当 x 从负方向趋近于 pi*Rational(3, 2) 时，1/cot(x)**3 的极限为正无穷大
    assert limit(1/cot(x)**3, x, (pi*Rational(3, 2)), dir="-") is oo

    # 断言：当 x 趋向于无穷大时，tan(x) 的极限为 AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(tan(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)

    # 断言：当 x 趋向于无穷大时，cot(x) 的极限为 AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(cot(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)

    # 断言：当 x 趋向于无穷大时，sec(x) 的极限为 AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(sec(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)

    # 断言：当 x 趋向于无穷大时，csc(x) 的极限为 AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(csc(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)

    # 测试双向极限
    assert limit(sin(x)/x, x, 0, dir="+-") == 1
    assert limit(x**2, x, 0, dir="+-") == 0
    assert limit(1/x**2, x, 0, dir="+-") is oo

    # 测试失败的双向极限
    assert limit(1/x, x, 0, dir="+-") is zoo

    # 断言：当 x 趋向于 0 时，从正方向 dir='+' 计算，1 + 1/x 的极限为正无穷大
    assert limit(1 + 1/x, x, 0) is oo

    # 断言：当 x 趋向于 0 时，从负方向 dir='-' 计算，1 + 1/x 的极限为负无穷大
    assert limit(1 + 1/x, x, 0, dir='-') is -oo

    # 断言：当 x 趋向于 0 时，从负方向 dir='-' 计算，x**(-2) 的极限为正无穷大
    assert limit(x**(-2), x, 0, dir='-') is oo

    # 断言：当 x 趋向于 0 时，从负方向 dir='-' 计算，x**(-3) 的极限为负无穷大
    assert limit(x**(-3), x, 0, dir='-') is -oo

    # 断言：当 x 趋向于 0 时，从负方向 dir='-' 计算，1/sqrt(x) 的极限为 (-oo)*I
    assert limit(1/sqrt(x), x, 0, dir='-') == (-oo)*I

    # 断言：当 x 趋向于 0 时，从负方向 dir='-' 计算，x**2 的极限为 0
    assert limit(x**2, x, 0, dir='-') == 0

    # 断言：当 x 趋向于 0 时，从负方向 dir='-' 计算，sqrt(x) 的极限为 0
    assert limit(sqrt(x), x, 0, dir='-') == 0

    # 断言：当 x 趋向于 0 时，从负方向 dir='-' 计算，x**-pi 的极限为 -oo*(-1)**(1 - pi)
    assert limit(x**-pi, x, 0, dir='-') == -oo*(-1)**(1 - pi)

    # 断言：当 x 趋向于 0 时，(cos(x) + 1)**oo 的极限为 Limit((cos(x) + 1)**oo, x, 0)
    assert limit((1 + cos(x))**oo, x, 0) == Limit((cos(x) + 1)**oo, x, 0)

    # 测试拉请求 22491
    assert limit(1/asin(x), x, 0, dir = '+') == oo
    assert limit(1/asin(x), x, 0, dir = '-') == -oo
    assert limit(1/sinh(x), x, 0, dir = '+') == oo
    assert limit(1/sinh(x), x, 0, dir = '-') == -oo
    assert limit(log(1/x) + 1/sin(x), x, 0, dir = '+') == oo
    assert limit(log(1/x) + 1/x, x, 0, dir = '+') == oo
def test_basic2():
    # 断言极限计算结果为 1，其中 x^x 当 x 趋近于 0 时的正向极限
    assert limit(x**x, x, 0, dir="+") == 1
    # 断言极限计算结果为 1，其中 (exp(x) - 1)/x 当 x 趋近于 0 时的极限
    assert limit((exp(x) - 1)/x, x, 0) == 1
    # 断言极限计算结果为 1，其中 1 + 1/x 当 x 趋近于正无穷时的极限
    assert limit(1 + 1/x, x, oo) == 1
    # 断言极限计算结果为 -1，其中 -exp(1/x) 当 x 趋近于正无穷时的极限
    assert limit(-exp(1/x), x, oo) == -1
    # 断言极限计算结果为正无穷，其中 x + exp(-x) 当 x 趋近于正无穷时的极限
    assert limit(x + exp(-x), x, oo) is oo
    # 断言极限计算结果为正无穷，其中 x + exp(-x**2) 当 x 趋近于正无穷时的极限
    assert limit(x + exp(-x**2), x, oo) is oo
    # 断言极限计算结果为正无穷，其中 x + exp(-exp(x)) 当 x 趋近于正无穷时的极限
    assert limit(x + exp(-exp(x)), x, oo) is oo
    # 断言极限计算结果为 13，其中 13 + 1/x - exp(-x) 当 x 趋近于正无穷时的极限
    assert limit(13 + 1/x - exp(-x), x, oo) == 13


def test_basic3():
    # 断言极限计算结果为正无穷，其中 1/x 当 x 趋近于 0 时的正向极限
    assert limit(1/x, x, 0, dir="+") is oo
    # 断言极限计算结果为负无穷，其中 1/x 当 x 趋近于 0 时的负向极限
    assert limit(1/x, x, 0, dir="-") is -oo


def test_basic4():
    # 断言极限计算结果为 0，其中 2*x + y*x 当 x 趋近于 0 时的极限
    assert limit(2*x + y*x, x, 0) == 0
    # 断言极限计算结果为 2 + y，其中 2*x + y*x 当 x 趋近于 1 时的极限
    assert limit(2*x + y*x, x, 1) == 2 + y
    # 断言极限计算结果为 512 - y/8，其中 2*x**8 + y*x**(-3) 当 x 趋近于 -2 时的极限
    assert limit(2*x**8 + y*x**(-3), x, -2) == 512 - y/8
    # 断言极限计算结果为 0，其中 sqrt(x + 1) - sqrt(x) 当 x 趋近于正无穷时的极限
    assert limit(sqrt(x + 1) - sqrt(x), x, oo) == 0
    # 断言积分结果为 2*pi*sqrt(3)/9，其中 1/(x**3 + 1) 在区间 (0, 正无穷) 上的积分
    assert integrate(1/(x**3 + 1), (x, 0, oo)) == 2*pi*sqrt(3)/9


def test_log():
    # https://github.com/sympy/sympy/issues/21598
    a, b, c = symbols('a b c', positive=True)
    # A 的极限结果为 0，其中 log(a/b) - (log(a) - log(b)) 当 a 趋近于正无穷时的极限
    A = log(a/b) - (log(a) - log(b))
    assert A.limit(a, oo) == 0
    # A * c 的极限结果为 0，其中 (log(a/b) - (log(a) - log(b))) * c 当 a 趋近于正无穷时的极限
    assert (A * c).limit(a, oo) == 0

    tau, x = symbols('tau x', positive=True)
    # expr 的极限结果为 pi*tau**3/(tau**2 + 1)**2，当 x 趋近于正无穷时的极限
    expr = tau**2*((tau - 1)*(tau + 1)*log(x + 1)/(tau**2 + 1)**2 + 1/((tau**2\
            + 1)*(x + 1)) - (-2*tau*atan(x/tau) + (tau**2/2 - 1/2)*log(tau**2\
            + x**2))/(tau**2 + 1)**2)
    assert limit(expr, x, oo) == pi*tau**3/(tau**2 + 1)**2


def test_piecewise():
    # https://github.com/sympy/sympy/issues/18363
    # 断言分段函数极限结果为 1/12，其中 (real_root(x - 6, 3) + 2)/(x + 2) 当 x 趋近于 -2 时的极限
    assert limit((real_root(x - 6, 3) + 2)/(x + 2), x, -2, '+') == Rational(1, 12)


def test_piecewise2():
    func1 = 2*sqrt(x)*Piecewise(((4*x - 2)/Abs(sqrt(4 - 4*(2*x - 1)**2)), 4*x - 2\
            >= 0), ((2 - 4*x)/Abs(sqrt(4 - 4*(2*x - 1)**2)), True))
    func2 = Piecewise((x**2/2, x <= 0.5), (x/2 - 0.125, True))
    func3 = Piecewise(((x - 9) / 5, x < -1), ((x - 9) / 5, x > 4), (sqrt(Abs(x - 3)), True))
    # 断言分段函数极限结果分别为 1, 0, 2，其中 func1, func2, func3 当 x 趋近于 0, -1, -2 时的极限
    assert limit(func1, x, 0) == 1
    assert limit(func2, x, 0) == 0
    assert limit(func3, x, -1) == 2


def test_basic5():
    class my(Function):
        @classmethod
        def eval(cls, arg):
            if arg is S.Infinity:
                return S.NaN
    # 断言 my(x) 当 x 趋近于正无穷时的极限为 Limit(my(x), x, oo)
    assert limit(my(x), x, oo) == Limit(my(x), x, oo)


def test_issue_3885():
    # 断言 x*y + x*z 当 z 趋近于 2 时的极限结果为 x*y + 2*x
    assert limit(x*y + x*z, z, 2) == x*y + 2*x


def test_Limit():
    # 断言 Limit(sin(x)/x, x, 0) 不等于 1
    assert Limit(sin(x)/x, x, 0) != 1
    # 断言 Limit(sin(x)/x, x, 0).doit() 的计算结果为 1
    assert Limit(sin(x)/x, x, 0).doit() == 1
    # 断言 Limit(x, x, 0, dir='+-') 的参数为 (x, x, 0, Symbol('+-'))
    assert Limit(x, x, 0, dir='+-').args == (x, x, 0, Symbol('+-'))


def test_floor():
    # 断言 floor(x) 当 x 趋近于 -2 时的正向极限结果为 -2
    assert limit(floor(x), x, -2, "+") == -2
    # 断言 floor(x) 当 x 趋近于 -2 时的负向极
    # 断言：验证极限表达式是否等于有理数 3/2
    assert limit(x*floor(3/x)/2, x, 0, '+') == Rational(3, 2)
    
    # 断言：验证极限表达式在 x 趋向无穷时的值是否落在累积边界 [-1/2, 3/2] 内
    assert limit(floor(x + 1/2) - floor(x), x, oo) == AccumBounds(-S.Half, S(3)/2)
    
    # 测试问题编号 9158
    # 断言：验证 arctan(x) 的下取整在 x 趋向正无穷时的极限是否为 1
    assert limit(floor(atan(x)), x, oo) == 1
    
    # 断言：验证 arctan(x) 的下取整在 x 趋向负无穷时的极限是否为 -2
    assert limit(floor(atan(x)), x, -oo) == -2
    
    # 断言：验证 arctan(x) 的上取整在 x 趋向正无穷时的极限是否为 2
    assert limit(ceiling(atan(x)), x, oo) == 2
    
    # 断言：验证 arctan(x) 的上取整在 x 趋向负无穷时的极限是否为 -1
    assert limit(ceiling(atan(x)), x, -oo) == -1
def test_floor_requires_robust_assumptions():
    # 检查 sin(x) 的地板函数在 x 趋近 0 时的极限是否为 0
    assert limit(floor(sin(x)), x, 0, "+") == 0
    # 检查 sin(x) 的地板函数在 x 趋近 0 时的极限是否为 -1
    assert limit(floor(sin(x)), x, 0, "-") == -1
    # 检查 cos(x) 的地板函数在 x 趋近 0 时的极限是否为 0
    assert limit(floor(cos(x)), x, 0, "+") == 0
    # 检查 cos(x) 的地板函数在 x 趋近 0 时的极限是否为 0
    assert limit(floor(cos(x)), x, 0, "-") == 0
    # 检查 5 + sin(x) 的地板函数在 x 趋近 0 时的极限是否为 5
    assert limit(floor(5 + sin(x)), x, 0, "+") == 5
    # 检查 5 + sin(x) 的地板函数在 x 趋近 0 时的极限是否为 4
    assert limit(floor(5 + sin(x)), x, 0, "-") == 4
    # 检查 5 + cos(x) 的地板函数在 x 趋近 0 时的极限是否为 5
    assert limit(floor(5 + cos(x)), x, 0, "+") == 5
    # 检查 5 + cos(x) 的地板函数在 x 趋近 0 时的极限是否为 5
    assert limit(floor(5 + cos(x)), x, 0, "-") == 5


def test_ceiling():
    # 检查 x 的天花板函数在 x 趋近 -2 时的极限是否为 -1
    assert limit(ceiling(x), x, -2, "+") == -1
    # 检查 x 的天花板函数在 x 趋近 -2 时的极限是否为 -2
    assert limit(ceiling(x), x, -2, "-") == -2
    # 检查 x 的天花板函数在 x 趋近 -1 时的极限是否为 0
    assert limit(ceiling(x), x, -1, "+") == 0
    # 检查 x 的天花板函数在 x 趋近 -1 时的极限是否为 -1
    assert limit(ceiling(x), x, -1, "-") == -1
    # 检查 x 的天花板函数在 x 趋近 0 时的极限是否为 1
    assert limit(ceiling(x), x, 0, "+") == 1
    # 检查 x 的天花板函数在 x 趋近 0 时的极限是否为 0
    assert limit(ceiling(x), x, 0, "-") == 0
    # 检查 x 的天花板函数在 x 趋近 1 时的极限是否为 2
    assert limit(ceiling(x), x, 1, "+") == 2
    # 检查 x 的天花板函数在 x 趋近 1 时的极限是否为 1
    assert limit(ceiling(x), x, 1, "-") == 1
    # 检查 x 的天花板函数在 x 趋近 2 时的极限是否为 3
    assert limit(ceiling(x), x, 2, "+") == 3
    # 检查 x 的天花板函数在 x 趋近 2 时的极限是否为 2
    assert limit(ceiling(x), x, 2, "-") == 2
    # 检查 x 的天花板函数在 x 趋近 248 时的极限是否为 249
    assert limit(ceiling(x), x, 248, "+") == 249
    # 检查 x 的天花板函数在 x 趋近 248 时的极限是否为 248
    assert limit(ceiling(x), x, 248, "-") == 248

    # https://github.com/sympy/sympy/issues/14478
    # 检查 x*ceiling(3/x)/2 在 x 趋近 0 时的极限是否为 3/2
    assert limit(x*ceiling(3/x)/2, x, 0, '+') == Rational(3, 2)
    # 检查 ceiling(x + 1/2) - ceiling(x) 在 x 趋近 oo 时的极限是否为 (-1/2, 3/2)
    assert limit(ceiling(x + 1/2) - ceiling(x), x, oo) == AccumBounds(-S.Half, S(3)/2)


def test_ceiling_requires_robust_assumptions():
    # 检查 sin(x) 的天花板函数在 x 趋近 0 时的极限是否为 1
    assert limit(ceiling(sin(x)), x, 0, "+") == 1
    # 检查 sin(x) 的天花板函数在 x 趋近 0 时的极限是否为 0
    assert limit(ceiling(sin(x)), x, 0, "-") == 0
    # 检查 cos(x) 的天花板函数在 x 趋近 0 时的极限是否为 1
    assert limit(ceiling(cos(x)), x, 0, "+") == 1
    # 检查 cos(x) 的天花板函数在 x 趋近 0 时的极限是否为 1
    assert limit(ceiling(cos(x)), x, 0, "-") == 1
    # 检查 5 + sin(x) 的天花板函数在 x 趋近 0 时的极限是否为 6
    assert limit(ceiling(5 + sin(x)), x, 0, "+") == 6
    # 检查 5 + sin(x) 的天花板函数在 x 趋近 0 时的极限是否为 5
    assert limit(ceiling(5 + sin(x)), x, 0, "-") == 5
    # 检查 5 + cos(x) 的天花板函数在 x 趋近 0 时的极限是否为 6
    assert limit(ceiling(5 + cos(x)), x, 0, "+") == 6
    # 检查 5 + cos(x) 的天花板函数在 x 趋近 0 时的极限是否为 6
    assert limit(ceiling(5 + cos(x)), x, 0, "-") == 6


def test_frac():
    # 检查 frac(x) 在 x 趋近 oo 时的极限是否为 [0, 1]
    assert limit(frac(x), x, oo) == AccumBounds(0, 1)
    # 检查 frac(x)**(1/x) 在 x 趋近 oo 时的极限是否为 [0, 1]
    assert limit(frac(x)**(1/x), x, oo) == AccumBounds(0, 1)
    # 检查 frac(x)**(1/x) 在 x 趋近 -oo 时的极限是否为 [1, oo]
    assert limit(frac(x)**(1/x), x, -oo) == AccumBounds(1, oo)
    # 检查 frac(x)**x 在 x 趋近 oo 时的极限是否为 [0, oo]
    assert limit(frac(x)**x, x, oo) == AccumBounds(0, oo)  # wolfram gives (0, 1)
    # 检查 frac(sin(x)) 在 x 趋近 0 时的极限是否为 0
    assert limit(frac(sin(x)), x, 0, "+") == 0
    # 检查 frac(sin(x)) 在 x 趋近 0 时的极限是否为 1
    assert limit(frac(sin(x)), x, 0, "-") == 1
    # 检查 frac(cos(x)) 在 x 趋近 0 时的极限是否为 1
    assert limit(frac(cos(x
    # 断言：当 x 趋向于 0 时，sin(x) 的绝对值的极限应为 0
    assert limit(abs(sin(x)), x, 0) == 0
    
    # 断言：当 x 趋向于 0 时，cos(x) 的绝对值的极限应为 1
    assert limit(abs(cos(x)), x, 0) == 1
    
    # 断言：当 x 趋向于 0 时，sin(x + 1) 的绝对值的极限应为 sin(1)
    assert limit(abs(sin(x + 1)), x, 0) == sin(1)
    
    # https://github.com/sympy/sympy/issues/9449
    # 断言：当 x 趋向于 0 时，(Abs(x + y) - Abs(x - y))/(2*x) 的极限应为 sign(y)
    assert limit((Abs(x + y) - Abs(x - y))/(2*x), x, 0) == sign(y)
    
    # https://github.com/sympy/sympy/issues/12398
    # 断言：当 x 趋向于无穷大时，Abs(log(x)/x**3) 的极限应为 0
    assert limit(Abs(log(x)/x**3), x, oo) == 0
    
    # 断言：当 x 趋向于无穷大时，x*(Abs(log(x)/x**3)/Abs(log(x + 1)/(x + 1)**3) - 1) 的极限应为 3
    assert limit(x*(Abs(log(x)/x**3)/Abs(log(x + 1)/(x + 1)**3) - 1), x, oo) == 3
    
    # https://github.com/sympy/sympy/issues/18501
    # 断言：当 x 从右边趋向于 1 时，Abs(log(x - 1)**3 - 1) 的极限应为无穷大
    assert limit(Abs(log(x - 1)**3 - 1), x, 1, '+') == oo
    
    # https://github.com/sympy/sympy/issues/18997
    # 断言：当 x 趋向于 0 时，Abs(log(x)) 的极限应为无穷大
    assert limit(Abs(log(x)), x, 0) == oo
    
    # 断言：当 x 趋向于 0 时，Abs(log(Abs(x))) 的极限应为无穷大
    assert limit(Abs(log(Abs(x))), x, 0) == oo
    
    # https://github.com/sympy/sympy/issues/19026
    # 符号定义：声明 z 是一个正数符号
    z = Symbol('z', positive=True)
    # 断言：当 z 趋向于无穷大时，Abs(log(z) + 1)/log(z) 的极限应为 1
    assert limit(Abs(log(z) + 1)/log(z), z, oo) == 1
    
    # https://github.com/sympy/sympy/issues/20704
    # 断言：当 z 趋向于 0 时，z*(Abs(1/z + y) - Abs(y - 1/z))/2 的极限应为 0
    assert limit(z*(Abs(1/z + y) - Abs(y - 1/z))/2, z, 0) == 0
    
    # https://github.com/sympy/sympy/issues/21606
    # 断言：当 z 从左边趋向于 π 时，cos(z)/sign(z) 的极限应为 -1
    assert limit(cos(z)/sign(z), z, pi, '-') == -1
# 定义一个测试函数，用于验证数学表达式在极限情况下的计算结果是否正确
def test_heuristic():
    # 创建一个实数符号 x
    x = Symbol("x", real=True)
    # 验证在 x 接近 0 时，sin(1/x) + atan(x) 的极限计算是否为 AccumBounds(-1, 1)
    assert heuristics(sin(1/x) + atan(x), x, 0, '+') == AccumBounds(-1, 1)
    # 验证 log(2 + sqrt(atan(x))*sqrt(sin(1/x))) 在 x 接近 0 时的极限计算是否为 log(2)
    assert limit(log(2 + sqrt(atan(x))*sqrt(sin(1/x))), x, 0) == log(2)


# 定义一个测试函数，用于验证在特定条件下的极限计算是否正确
def test_issue_3871():
    # 创建一个正数符号 z
    z = Symbol("z", positive=True)
    # 定义一个表达式 f，用于验证在 x 接近 oo 时其极限是否为 0
    f = -1/z*exp(-z*x)
    assert limit(f, x, oo) == 0
    assert f.limit(x, oo) == 0


# 定义一个测试函数，用于验证指数函数的极限计算是否正确
def test_exponential():
    # 创建符号 n 和实数符号 x
    n = Symbol('n')
    x = Symbol('x', real=True)
    # 验证 (1 + x/n)**n 在 n 接近 oo 时的极限计算是否为 exp(x)
    assert limit((1 + x/n)**n, n, oo) == exp(x)
    # 验证 (1 + x/(2*n))**n 在 n 接近 oo 时的极限计算是否为 exp(x/2)
    assert limit((1 + x/(2*n))**n, n, oo) == exp(x/2)
    # 验证 (1 + x/(2*n + 1))**n 在 n 接近 oo 时的极限计算是否为 exp(x/2)
    assert limit((1 + x/(2*n + 1))**n, n, oo) == exp(x/2)
    # 验证 ((x - 1)/(x + 1))**x 在 x 接近 oo 时的极限计算是否为 exp(-2)
    assert limit(((x - 1)/(x + 1))**x, x, oo) == exp(-2)
    # 验证 1 + (1 + 1/x)**x 在 x 接近 oo 时的极限计算是否为 1 + S.Exp1
    assert limit(1 + (1 + 1/x)**x, x, oo) == 1 + S.Exp1
    # 验证 (2 + 6*x)**x / (6*x)**x 在 x 接近 oo 时的极限计算是否为 exp(1/3)
    assert limit((2 + 6*x)**x/(6*x)**x, x, oo) == exp(S('1/3'))


# 定义一个测试函数，用于验证指数函数的极限计算是否正确
def test_exponential2():
    # 创建符号 n
    n = Symbol('n')
    # 验证 (1 + x / (n + sin(n)))**n 在 n 接近 oo 时的极限计算是否为 exp(x)
    assert limit((1 + x/(n + sin(n)))**n, n, oo) == exp(x)


# 定义一个测试函数，用于验证积分和极限计算是否正确
def test_doit():
    # 创建一个积分对象 f
    f = Integral(2 * x, x)
    # 创建一个极限对象 l
    l = Limit(f, x, oo)
    # 验证极限对象调用 doit 方法后是否返回 oo
    assert l.doit() is oo


# 定义一个测试函数，用于验证级数和极限计算是否正确
def test_series_AccumBounds():
    # 验证 sin(k) - sin(k + 1) 在 k 接近 oo 时的极限计算是否为 AccumBounds(-2, 2)
    assert limit(sin(k) - sin(k + 1), k, oo) == AccumBounds(-2, 2)
    # 验证 cos(k) - cos(k + 1) + 1 在 k 接近 oo 时的极限计算是否为 AccumBounds(-1, 3)
    assert limit(cos(k) - cos(k + 1) + 1, k, oo) == AccumBounds(-1, 3)

    # 测试针对 issue #9934 的非精确边界情况
    assert limit(sin(k) - sin(k)*cos(k), k, oo) == AccumBounds(-2, 2)

    # 测试 issue #9934
    lo = (-3 + cos(1))/2
    hi = (1 + cos(1))/2
    t1 = Mul(AccumBounds(lo, hi), 1/(-1 + cos(1)), evaluate=False)
    assert limit(simplify(Sum(cos(n).rewrite(exp), (n, 0, k)).doit().rewrite(sin)), k, oo) == t1

    t2 = Mul(AccumBounds(-1 + sin(1)/2, sin(1)/2 + 1), 1/(1 - cos(1)))
    assert limit(simplify(Sum(sin(n).rewrite(exp), (n, 0, k)).doit().rewrite(sin)), k, oo) == t2

    # 验证 ((sin(x) + 1)/2)**x 在 x 接近 oo 时的极限计算是否为 AccumBounds(0, oo)，与 Wolfram Alpha 结果一致
    assert limit(((sin(x) + 1)/2)**x, x, oo) == AccumBounds(0, oo)

    # 验证 2**(-x)*(sin(x) + 1)**x 在 x 接近 oo 时的极限计算是否为 AccumBounds(0, oo)
    e = 2**(-x)*(sin(x) + 1)**x
    assert limit(e, x, oo) == AccumBounds(0, oo)


# 定义一个测试函数，用于验证贝塞尔函数在无穷远点附近的极限计算是否正确
def test_bessel_functions_at_infinity():
    # 验证贝塞尔函数 besselj(1, x) 在 x 接近 oo 时的极限计算是否为 0
    assert limit(besselj(1, x), x, oo) == 0
    assert limit(besselj(1, x), x, -oo) == 0
    assert limit(besselj(1, x), x, I*oo) == oo*I
    assert limit(besselj(1, x), x, -I*oo) == -oo*I
    # 验证贝塞尔函数 bessely(1, x) 在 x 接近 oo 时的极限计算是否为 0
    assert limit(bessely(1, x), x, oo) == 0
    assert limit(bessely(1, x), x, -oo) == 0
    assert limit(bessely(1, x), x, I*oo) == -oo
    assert limit(bessely(1, x), x, -I*oo) == -oo
    # 验证贝塞尔函数 besseli(1, x) 在 x 接近 oo 时的极限计算是否为 oo
    assert limit(besseli(1, x), x, oo) == oo
    assert limit(besseli(1, x), x, -oo) == -oo
    assert limit(besseli(1, x), x, I*oo) == 0
    assert limit(besseli(1, x), x, -I*oo) == 0
    # 验证贝塞尔函数 besselk(1, x) 在 x 接近 oo 时的极限计算是否为 0
    assert limit(besselk(1, x), x, oo) == 0
    assert limit(besselk(1, x), x, -oo) == -oo*I
    assert limit(besselk(1, x), x, I*oo) == 0
    assert limit(besselk(1, x), x, -I*oo) == 0

    # 验证 issue 14874
    assert limit(besselk(0, x), x, oo) == 0
    # 使用断言检查 l.doit(deep=False) 是否等于 l，确保 limit() 在包含的 Integral 上正常工作。
    assert l.doit(deep=False) == l
def test_issue_2929():
    # 断言极限计算：(x * exp(x)) / (exp(x) - 1) 在 x 趋向负无穷时的极限为 0
    assert limit((x * exp(x))/(exp(x) - 1), x, -oo) == 0


def test_issue_3792():
    # 断言极限计算：(1 - cos(x)) / x**2 在 x 趋向 1/2 时的极限为 4 - 4*cos(1/2)
    assert limit((1 - cos(x))/x**2, x, S.Half) == 4 - 4*cos(S.Half)
    # 断言极限计算：sin(sin(x + 1) + 1) 在 x 趋向 0 时的极限为 sin(1 + sin(1))
    assert limit(sin(sin(x + 1) + 1), x, 0) == sin(1 + sin(1))
    # 断言极限计算：abs(sin(x + 1) + 1) 在 x 趋向 0 时的极限为 1 + sin(1)
    assert limit(abs(sin(x + 1) + 1), x, 0) == 1 + sin(1)


def test_issue_4090():
    # 断言极限计算：1 / (x + 3) 在 x 趋向 2 时的极限为 1/5
    assert limit(1/(x + 3), x, 2) == Rational(1, 5)
    # 断言极限计算：1 / (x + pi) 在 x 趋向 2 时的极限为 1 / (2 + pi)
    assert limit(1/(x + pi), x, 2) == S.One/(2 + pi)
    # 断言极限计算：log(x) / (x**2 + 3) 在 x 趋向 2 时的极限为 log(2) / 7
    assert limit(log(x)/(x**2 + 3), x, 2) == log(2)/7
    # 断言极限计算：log(x) / (x**2 + pi) 在 x 趋向 2 时的极限为 log(2) / (4 + pi)
    assert limit(log(x)/(x**2 + pi), x, 2) == log(2)/(4 + pi)


def test_issue_4547():
    # 断言极限计算：cot(x) 在 x 趋向 0 (正方向) 时的极限为 正无穷
    assert limit(cot(x), x, 0, dir='+') is oo
    # 断言极限计算：cot(x) 在 x 趋向 pi/2 (正方向) 时的极限为 0
    assert limit(cot(x), x, pi/2, dir='+') == 0


def test_issue_5164():
    # 断言极限计算：x**0.5 在 x 趋向 正无穷 时的极限为 正无穷
    assert limit(x**0.5, x, oo) == oo
    # 断言极限计算：x**0.5 在 x 趋向 16 时的极限为 4
    assert limit(x**0.5, x, 16) == 4  # 这应该是一个浮点数吗？
    # 断言极限计算：x**0.5 在 x 趋向 0 时的极限为 0
    assert limit(x**0.5, x, 0) == 0
    # 断言极限计算：x**(-0.5) 在 x 趋向 正无穷 时的极限为 0
    assert limit(x**(-0.5), x, oo) == 0
    # 断言极限计算：x**(-0.5) 在 x 趋向 4 时的极限为 1/2
    assert limit(x**(-0.5), x, 4) == S.Half  # 这应该是一个浮点数吗？


def test_issue_5383():
    # 使用浮点数定义函数 func
    func = (1.0 * 1 + 1.0 * x)**(1.0 * 1 / x)
    # 断言极限计算：func 在 x 趋向 0 时的极限为 自然常数 e
    assert limit(func, x, 0) == E


def test_issue_14793():
    # 定义复杂表达式 expr
    expr = ((x + S(1)/2) * log(x) - x + log(2*pi)/2 - \
        log(factorial(x)) + S(1)/(12*x))*x**3
    # 断言极限计算：expr 在 x 趋向 正无穷 时的极限为 1/360
    assert limit(expr, x, oo) == S(1)/360


def test_issue_5183():
    # 断言极限计算：r*sin(1/r) 在 r 趋向 0 时的极限为 0
    r = Symbol('r', real=True)
    assert limit(r*sin(1/r), r, 0) == 0


def test_issue_5229():
    # 断言极限计算：((1 + y)**(1/y)) - Exp1 在 y 趋向 0 时的极限为 0
    assert limit((1 + y)**(1/y) - S.Exp1, y, 0) == 0


def test_issue_4546():
    # using list(...) so py.test can recalculate values
    # 使用 list(...) 以便 py.test 可以重新计算值
    tests = list(product([cot, tan],
                         [-pi/2, 0, pi/2, pi, pi*Rational(3, 2)],
                         ['-', '+']))
    results = (0, 0, -oo, oo, 0, 0, -oo, oo, 0, 0,
               oo, -oo, 0, 0, oo, -oo, 0, 0, oo, -oo)
    assert len(tests) == len(results)
    for i, (args, res) in enumerate(zip(tests, results)):
        y, s, e, d = args
        eq = y**(s*e)
        try:
            # 断言极限计算：eq 在 x 趋向 0 (方向为 d) 时的极限为 res
            assert limit(eq, x, 0, dir=d) == res
        except AssertionError:
            if 0:  # change to 1 if you want to see the failing tests
                print()
                print(i, res, eq, d, limit(eq, x, 0, dir=d))
            else:
                assert None
    # 使用 enumerate 遍历 tests 和 results 列表，同时获取索引 i 和元组 (args, res)
    for i, (args, res) in enumerate(zip(tests, results)):
        # 将 args 解包为 f, l, d
        f, l, d = args
        # 计算函数 f(x) 的极限 eq
        eq = f(x)
        # 尝试断言 limit(eq, x, l, dir=d) 的结果与 res 相等
        try:
            assert limit(eq, x, l, dir=d) == res
        # 如果断言失败，捕获 AssertionError 异常
        except AssertionError:
            # 如果条件为真（通常用于调试目的），则打印相关信息
            if 0:  # 改为 1 可查看失败的测试用例
                print()
                print(i, res, eq, l, d, limit(eq, x, l, dir=d))
            # 否则重新抛出 AssertionError
            else:
                assert None
def test_issue_3934():
    # 断言：计算极限 (1 + x**log(3))**(1/x)，当 x 趋向于 0 时结果为 1
    assert limit((1 + x**log(3))**(1/x), x, 0) == 1
    # 断言：计算极限 (5**(1/x) + 3**(1/x))**x，当 x 趋向于 0 时结果为 5
    assert limit((5**(1/x) + 3**(1/x))**x, x, 0) == 5


def test_calculate_series():
    # 注意：
    # calculate_series 方法正在被弃用，不再负责返回结果。
    # 现在使用简单的 leadterm 调用，而不是 calculate_series。
    
    # 断言：计算极限 x**Rational(77, 3)/(1 + x**Rational(77, 3))，当 x 趋向于无穷大时结果为 1
    assert limit(x**Rational(77, 3)/(1 + x**Rational(77, 3)), x, oo) == 1
    # 断言：计算极限 x**101.1/(1 + x**101.1)，当 x 趋向于无穷大时结果为 1
    assert limit(x**101.1/(1 + x**101.1), x, oo) == 1


def test_issue_5955():
    # 断言：计算极限 x**16/(1 + x**16)，当 x 趋向于无穷大时结果为 1
    assert limit((x**16)/(1 + x**16), x, oo) == 1
    # 断言：计算极限 x**100/(1 + x**100)，当 x 趋向于无穷大时结果为 1
    assert limit((x**100)/(1 + x**100), x, oo) == 1
    # 断言：计算极限 x**1885/(1 + x**1885)，当 x 趋向于无穷大时结果为 1
    assert limit((x**1885)/(1 + x**1885), x, oo) == 1
    # 断言：计算极限 x**1000/((x + 1)**1000 + exp(-x))，当 x 趋向于无穷大时结果为 1
    assert limit((x**1000/((x + 1)**1000 + exp(-x))), x, oo) == 1


def test_newissue():
    # 断言：计算极限 exp(1/sin(x))/exp(cot(x))，当 x 趋向于 0 时结果为 1
    assert limit(exp(1/sin(x))/exp(cot(x)), x, 0) == 1


def test_extended_real_line():
    # 断言：计算极限 x - oo，当 x 趋向于无穷大时结果为 Limit(x - oo, x, oo)
    assert limit(x - oo, x, oo) == Limit(x - oo, x, oo)
    # 断言：计算极限 1/(x + sin(x)) - oo，当 x 趋向于 0 时结果为 Limit(1/(x + sin(x)) - oo, x, 0)
    assert limit(1/(x + sin(x)) - oo, x, 0) == Limit(1/(x + sin(x)) - oo, x, 0)
    # 断言：计算极限 oo/x，当 x 趋向于无穷大时结果为 Limit(oo/x, x, oo)
    assert limit(oo/x, x, oo) == Limit(oo/x, x, oo)
    # 断言：计算极限 x - oo + 1/x，当 x 趋向于无穷大时结果为 Limit(x - oo + 1/x, x, oo)
    assert limit(x - oo + 1/x, x, oo) == Limit(x - oo + 1/x, x, oo)


@XFAIL
def test_order_oo():
    # 符号化 x 为正数
    x = Symbol('x', positive=True)
    # 断言：计算极限 oo/(x**2 - 4)，当 x 趋向于无穷大时结果为 oo
    assert limit(oo/(x**2 - 4), x, oo) is oo


def test_issue_5436():
    # 断言：调用 limit(exp(x*y), x, oo) 会抛出 NotImplementedError
    raises(NotImplementedError, lambda: limit(exp(x*y), x, oo))
    # 断言：调用 limit(exp(-x*y), x, oo) 会抛出 NotImplementedError
    raises(NotImplementedError, lambda: limit(exp(-x*y), x, oo))


def test_Limit_dir():
    # 断言：调用 Limit(x, x, 0, dir=0) 会抛出 TypeError
    raises(TypeError, lambda: Limit(x, x, 0, dir=0))
    # 断言：调用 Limit(x, x, 0, dir='0') 会抛出 ValueError
    raises(ValueError, lambda: Limit(x, x, 0, dir='0'))


def test_polynomial():
    # 断言：计算极限 (x + 1)**1000/((x + 1)**1000 + 1)，当 x 趋向于无穷大时结果为 1
    assert limit((x + 1)**1000/((x + 1)**1000 + 1), x, oo) == 1
    # 断言：计算极限 (x + 1)**1000/((x + 1)**1000 + 1)，当 x 趋向于负无穷大时结果为 1
    assert limit((x + 1)**1000/((x + 1)**1000 + 1), x, -oo) == 1


def test_rational():
    # 断言：计算极限 1/y - (1/(y + x) + x/(y + x)/y)/z，当 x 趋向于无穷大时结果为 (z - 1)/(y*z)
    assert limit(1/y - (1/(y + x) + x/(y + x)/y)/z, x, oo) == (z - 1)/(y*z)
    # 断言：计算极限 1/y - (1/(y + x) + x/(y + x)/y)/z，当 x 趋向于负无穷大时结果为 (z - 1)/(y*z)
    assert limit(1/y - (1/(y + x) + x/(y + x)/y)/z, x, -oo) == (z - 1)/(y*z)


def test_issue_5740():
    # 断言：计算极限 log(x)*z - log(2*x)*y，当 x 趋向于 0 时结果为 oo*sign(y - z)
    assert limit(log(x)*z - log(2*x)*y, x, 0) == oo*sign(y - z)


def test_issue_6366():
    # 符号化 n 为正整数
    n = Symbol('n', integer=True, positive=True)
    # 定义表达式 r
    r = (n + 1)*x**(n + 1)/(x**(n + 1) - 1) - x/(x - 1)
    # 断言：计算极限 r，当 x 趋向于 1 时化简结果为 n/2
    assert limit(r, x, 1).cancel() == n/2


def test_factorial():
    # 定义阶乘 f
    f = factorial(x)
    # 断言：计算极限 f，当 x 趋向于无穷大时结果为 oo
    assert limit(f, x, oo) is oo
    # 断言：计算极限 x/f，当 x 趋向于无穷大时结果为 0
    assert limit(x/f, x, oo) == 0
    # 断言：根据 Stirling's approximation 计算极限 f/(sqrt(2*pi*x)*(x/E)**x)，当 x 趋向于无穷大时结果为 1
    assert limit(f/(sqrt(2*pi*x)*(x/E)**x), x, oo) == 1
    # 断言：计算极限 f，当 x 趋向于负无穷大时结果为 gamma(-oo)
    assert limit(f, x, -oo
    # 计算表达式的值，表达式包含复杂的数学运算
    expr = ((2*n*(n - r + 1)/(n + r*(n - r + 1)))**c +
            (r - 1)*(n*(n - r + 2)/(n + r*(n - r + 1)))**c - n)/(n**c - n)
    # 将表达式中的符号 c 替换为 c + 1
    expr = expr.subs(c, c + 1)
    # 断言表达式在 n 趋于无穷时会抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: limit(expr, n, oo))
    # 断言当 c 被替换为 m 时，表达式在 n 趋于无穷时的极限值为 1
    assert limit(expr.subs(c, m), n, oo) == 1
    # 断言当 c 被替换为 p 时，表达式在 n 趋于无穷时的极限值经简化后等于 (2**(p + 1) + r - 1)/(r + 1)**(p + 1)
    assert limit(expr.subs(c, p), n, oo).simplify() == \
        (2**(p + 1) + r - 1)/(r + 1)**(p + 1)
# 定义一个测试函数，用于验证在符号计算中的极限表达式是否正确
def test_issue_7088():
    # 创建一个符号变量 'a'
    a = Symbol('a')
    # 断言：当 x 趋向无穷大时，sqrt(x/(x + a)) 的极限应为 1
    assert limit(sqrt(x/(x + a)), x, oo) == 1


# 定义一个测试函数，用于验证不同分支切割情况下的反三角函数的极限表达式是否正确
def test_branch_cuts():
    # 断言：当 x 趋向 0 时，asin(I*x + 2) 的极限应为 pi - asin(2)
    assert limit(asin(I*x + 2), x, 0) == pi - asin(2)
    # 断言：当 x 趋向 0 时，从负方向接近时，asin(I*x + 2) 的极限应为 asin(2)
    assert limit(asin(I*x + 2), x, 0, '-') == asin(2)
    # 更多类似的断言，验证不同情况下的极限表达式是否正确


# 定义一个测试函数，用于验证特定数学问题中的极限表达式是否正确
def test_issue_6364():
    # 创建一个符号变量 'a'
    a = Symbol('a')
    # 定义一个复杂的表达式 e
    e = z/(1 - sqrt(1 + z)*sin(a)**2 - sqrt(1 - z)*cos(a)**2)
    # 断言：当 z 趋向 0 时，e 的极限应为 1/(cos(a)**2 - S.Half)
    assert limit(e, z, 0) == 1/(cos(a)**2 - S.Half)


# 定义一个测试函数，用于验证特定数学问题中的极限表达式是否正确
def test_issue_6682():
    # 断言：当 x 趋向 0 时，exp(2*Ei(-x))/x**2 的极限应为 exp(2*EulerGamma)
    assert limit(exp(2*Ei(-x))/x**2, x, 0) == exp(2*EulerGamma)


# 定义一个测试函数，用于验证特定数学问题中的极限表达式是否正确
def test_issue_4099():
    # 创建一个符号变量 'a'
    a = Symbol('a')
    # 断言：当 x 趋向 0 时，a/x 的极限应为 oo*sign(a)
    assert limit(a/x, x, 0) == oo*sign(a)
    # 更多类似的断言，验证不同情况下的极限表达式是否正确


# 定义一个测试函数，用于验证特定数学问题中的极限表达式是否正确
def test_issue_4503():
    # 创建一个符号变量 'dx'
    dx = Symbol('dx')
    # 此处暂无需添加具体的测试代码，只是简单定义了一个符号变量
    # 断言：验证极限表达式是否正确
    assert limit((sqrt(1 + exp(x + dx)) - sqrt(1 + exp(x)))/dx, dx, 0) == \
        exp(x)/(2*sqrt(exp(x) + 1))
def test_issue_6052():
    # 定义一个 Meijer G 函数 G
    G = meijerg((), (), (1,), (0,), -x)
    # 对 G 进行超级展开
    g = hyperexpand(G)
    # 断言极限计算结果为 0
    assert limit(g, x, 0, '+-') == 0
    # 断言极限计算结果为 -oo
    assert limit(g, x, oo) == -oo


def test_issue_7224():
    # 定义一个表达式 expr
    expr = sqrt(x)*besseli(1,sqrt(8*x))
    # 断言极限计算结果为 2
    assert limit(x*diff(expr, x, x)/expr, x, 0) == 2
    # 断言极限计算结果为 2.0
    assert limit(x*diff(expr, x, x)/expr, x, 1).evalf() == 2.0


def test_issue_8208():
    # 断言极限计算结果为 0
    assert limit(n**(Rational(1, 1e9) - 1), n, oo) == 0


def test_issue_8229():
    # 断言极限计算结果为 0
    assert limit((x**Rational(1, 4) - 2)/(sqrt(x) - 4)**Rational(2, 3), x, 16) == 0


def test_issue_8433():
    d, t = symbols('d t', positive=True)
    # 断言极限计算结果为 -1
    assert limit(erf(1 - t/d), t, oo) == -1


def test_issue_8481():
    k = Symbol('k', integer=True, nonnegative=True)
    lamda = Symbol('lamda', positive=True)
    # 断言极限计算结果为 0
    assert limit(lamda**k * exp(-lamda) / factorial(k), k, oo) == 0


def test_issue_8462():
    # 断言极限计算结果为 oo
    assert limit(binomial(n, n/2), n, oo) == oo
    # 断言极限计算结果为 0
    assert limit(binomial(n, n/2) * 3 ** (-n), n, oo) == 0


def test_issue_8634():
    n = Symbol('n', integer=True, positive=True)
    x = Symbol('x')
    # 断言极限计算结果为 oo*sign((-1)**n)
    assert limit(x**n, x, -oo) == oo*sign((-1)**n)


def test_issue_8635_18176():
    x = Symbol('x', real=True)
    k = Symbol('k', positive=True)
    # 断言极限计算结果为 0
    assert limit(x**n - x**(n - 0), x, oo) == 0
    # 断言极限计算结果为 oo
    assert limit(x**n - x**(n - 5), x, oo) == oo
    # 断言极限计算结果为 oo
    assert limit(x**n - x**(n - 2.5), x, oo) == oo
    # 断言极限计算结果为 oo
    assert limit(x**n - x**(n - k - 1), x, oo) == oo
    x = Symbol('x', positive=True)
    # 断言极限计算结果为 oo
    assert limit(x**n - x**(n - 1), x, oo) == oo
    # 断言极限计算结果为 -oo
    assert limit(x**n - x**(n + 2), x, oo) == -oo


def test_issue_8730():
    # 断言极限计算结果为 oo
    assert limit(subfactorial(x), x, oo) is oo


def test_issue_9252():
    n = Symbol('n', integer=True)
    c = Symbol('c', positive=True)
    # 断言极限计算结果为 0
    assert limit((log(n))**(n/log(n)) / (1 + c)**n, n, oo) == 0
    # 期望抛出 NotImplementedError
    raises(NotImplementedError, lambda: limit((log(n))**(n/log(n)) / c**n, n, oo))


def test_issue_9558():
    # 断言极限计算结果为 0
    assert limit(sin(x)**15, x, 0, '-') == 0


def test_issue_10801():
    # 断言极限计算结果为 pi
    assert limit(16**k / (k * binomial(2*k, k)**2), k, oo) == pi


def test_issue_10976():
    s, x = symbols('s x', real=True)
    # 断言极限计算结果为 x
    assert limit(erf(s*x)/erf(s), s, 0) == x


def test_issue_9041():
    # 断言极限计算结果为 1
    assert limit(factorial(n) / ((n/exp(1))**n * sqrt(2*pi*n)), n, oo) == 1


def test_issue_9205():
    x, y, a = symbols('x, y, a')
    # 检查极限对象的自由符号集合为 {a}
    assert Limit(x, x, a).free_symbols == {a}
    # 检查极限对象的自由符号集合为 {a}
    assert Limit(x, x, a, '-').free_symbols == {a}
    # 检查极限对象的自由符号集合为 {a}
    assert Limit(x + y, x + y, a).free_symbols == {a}
    # 检查极限对象的自由符号集合为 {y, a}
    assert Limit(-x**2 + y, x**2, a).free_symbols == {y, a}


def test_issue_9471():
    # 断言极限计算结果为 1
    assert limit(((27**(log(n,3)))/n**3),n,oo) == 1
    # 断言极限计算结果为 27
    assert limit(((27**(log(n,3)+1))/n**3),n,oo) == 27


def test_issue_10382():
    # 断言极限计算结果为 GoldenRatio
    assert limit(fibonacci(n + 1)/fibonacci(n), n, oo) == GoldenRatio


def test_issue_11496():
    # 断言极限计算结果为 2
    assert limit(erfc(log(1/x)), x, oo) == 2


def test_issue_11879():
    # 断言极限化简结果为 n*x**(n-1)
    assert simplify(limit(((x+y)**n-x**n)/y, y, 0)) == n*x**(n-1)


def test_limit_with_Float():
    pass
    # 导入符号计算模块，并定义符号变量 k
    k = symbols("k")
    # 断言：当 k 趋向正无穷时，1.0 的 k 次方的极限为 1
    assert limit(1.0 ** k, k, oo) == 1
    # 断言：当 k 趋向正无穷时，0.3 乘以 1.0 的 k 次方的极限为 3/10，即 0.3
    assert limit(0.3*1.0**k, k, oo) == Rational(3, 10)
def test_issue_10610():
    # 断言极限计算结果是否等于有理数 1/3
    assert limit(3**x*3**(-x - 1)*(x + 1)**2/x**2, x, oo) == Rational(1, 3)


def test_issue_10868():
    # 断言正向趋近零时的极限
    assert limit(log(x) + asech(x), x, 0, '+') == log(2)
    # 断言负向趋近零时的极限
    assert limit(log(x) + asech(x), x, 0, '-') == log(2) + 2*I*pi
    # 断言在正负无穷方向的极限无效
    raises(ValueError, lambda: limit(log(x) + asech(x), x, 0, '+-'))
    # 断言正无穷方向的极限
    assert limit(log(x) + asech(x), x, oo) == oo
    # 断言正向趋近零时的极限
    assert limit(log(x) + acsch(x), x, 0, '+') == log(2)
    # 断言负向趋近零时的极限
    assert limit(log(x) + acsch(x), x, 0, '-') == -oo
    # 断言在正负无穷方向的极限无效
    raises(ValueError, lambda: limit(log(x) + acsch(x), x, 0, '+-'))
    # 断言正无穷方向的极限
    assert limit(log(x) + acsch(x), x, oo) == oo


def test_issue_6599():
    # 断言在正无穷方向的极限
    assert limit((n + cos(n))/n, n, oo) == 1


def test_issue_12555():
    # 断言在负无穷方向的极限
    assert limit((3**x + 2* x**10) / (x**10 + exp(x)), x, -oo) == 2
    # 断言在正无穷方向的极限
    assert limit((3**x + 2* x**10) / (x**10 + exp(x)), x, oo) is oo


def test_issue_12769():
    r, z, x = symbols('r z x', real=True)
    a, b, s0, K, F0, s, T = symbols('a b s0 K F0 s T', positive=True, real=True)
    # 定义复杂的符号表达式 fx
    fx = (F0**b*K**b*r*s0 - sqrt((F0**2*K**(2*b)*a**2*(b - 1) + \
        F0**(2*b)*K**2*a**2*(b - 1) + F0**(2*b)*K**(2*b)*s0**2*(b - 1)*(b**2 - 2*b + 1) - \
        2*F0**(2*b)*K**(b + 1)*a*r*s0*(b**2 - 2*b +  1) + \
        2*F0**(b + 1)*K**(2*b)*a*r*s0*(b**2 - 2*b + 1) - \
        2*F0**(b + 1)*K**(b + 1)*a**2*(b - 1))/((b - 1)*(b**2 - 2*b + 1))))*(b*r -  b - r + 1)
    # 断言在 K 趋近于 F0 时的极限因式化结果
    assert fx.subs(K, F0).factor(deep=True) == limit(fx, K, F0).factor(deep=True)


def test_issue_13332():
    # 断言在正无穷方向的极限
    assert limit(sqrt(30)*5**(-5*x - 1)*(46656*x)**x*(5*x + 2)**(5*x + 5*S.Half) *
                (6*x + 2)**(-6*x - 5*S.Half), x, oo) == Rational(25, 36)


def test_issue_12564():
    # 断言在负无穷方向的极限
    assert limit(x**2 + x*sin(x) + cos(x), x, -oo) is oo
    # 断言在正无穷方向的极限
    assert limit(x**2 + x*sin(x) + cos(x), x, oo) is oo
    # 断言在正无穷方向的极限
    assert limit(((x + cos(x))**2).expand(), x, oo) is oo
    # 断言在负无穷方向的极限
    assert limit(((x + sin(x))**2).expand(), x, -oo) is oo
    # 断言在负无穷方向的极限
    assert limit(((x + cos(x))**2).expand(), x, -oo) is oo
    # 断言在正无穷方向的极限
    assert limit(((x + sin(x))**2).expand(), x, -oo) is oo


def test_issue_14456():
    # 检查对于极限的未实现情况是否会引发 NotImplementedError
    raises(NotImplementedError, lambda: Limit(exp(x), x, zoo).doit())
    raises(NotImplementedError, lambda: Limit(x**2/(x+1), x, zoo).doit())


def test_issue_14411():
    # 断言在指定点的极限
    assert limit(3*sec(4*pi*x - x/3), x, 3*pi/(24*pi - 2)) is -oo


def test_issue_13382():
    # 断言在正无穷方向的极限
    assert limit(x*(((x + 1)**2 + 1)/(x**2 + 1) - 1), x, oo) == 2


def test_issue_13403():
    # 断言在正无穷方向的极限
    assert limit(x*(-1 + (x + log(x + 1) + 1)/(x + log(x))), x, oo) == 1


def test_issue_13416():
    # 断言在正无穷方向的极限
    assert limit((-x**3*log(x)**3 + (x - 1)*(x + 1)**2*log(x + 1)**3)/(x**2*log(x)**3), x, oo) == 1


def test_issue_13462():
    # 断言在正无穷方向的极限
    assert limit(n**2*(2*n*(-(1 - 1/(2*n))**x + 1) - x - (-x**2/4 + x/4)/n), n, oo) == x**3/24 - x**2/8 + x/12


def test_issue_13750():
    a = Symbol('a')
    # 断言在正无穷方向的极限
    assert limit(erf(a - x), x, oo) == -1
    # 断言在正无穷方向的极限
    assert limit(erf(sqrt(x) - x), x, oo) == -1


def test_issue_14276():
    # 断言极限返回的类型是否是 Limit 类型
    assert isinstance(limit(sin(x)**log(x), x, oo), Limit)
    assert isinstance(limit(sin(x)**cos(x), x, oo), Limit)
    # 断言确保 limit 函数返回的对象是 Limit 类型
    assert isinstance(limit(sin(log(cos(x))), x, oo), Limit)
    # 断言验证当 x 趋向无穷时，给定表达式的极限等于自然常数 e
    assert limit((1 + 1/(x**2 + cos(x)))**(x**2 + x), x, oo) == E
def test_issue_14514():
    # 断言测试极限函数的计算结果是否等于1
    assert limit((1/(log(x)**log(x)))**(1/x), x, oo) == 1


def test_issues_14525():
    # 断言测试多个表达式的极限是否符合累积边界的范围
    assert limit(sin(x)**2 - cos(x) + tan(x)*csc(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(sin(x)**2 - cos(x) + sin(x)*cot(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(cot(x) - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(cos(x) - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.One)
    assert limit(sin(x) - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.One)
    assert limit(cos(x)**2 - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.One)
    assert limit(tan(x)**2 + sin(x)**2 - cos(x), x, oo) == AccumBounds(-S.One, S.Infinity)


def test_issue_14574():
    # 断言测试特定函数在无穷大时的极限是否为0
    assert limit(sqrt(x)*cos(x - x**2) / (x + 1), x, oo) == 0


def test_issue_10102():
    # 断言测试弗雷涅尔积分函数在无穷大和负无穷大时的极限值
    assert limit(fresnels(x), x, oo) == S.Half
    assert limit(3 + fresnels(x), x, oo) == 3 + S.Half
    assert limit(5*fresnels(x), x, oo) == Rational(5, 2)
    assert limit(fresnelc(x), x, oo) == S.Half
    assert limit(fresnels(x), x, -oo) == Rational(-1, 2)
    assert limit(4*fresnelc(x), x, -oo) == -2


def test_issue_14377():
    # 断言测试某个表达式在计算其极限时是否引发NotImplementedError异常
    raises(NotImplementedError, lambda: limit(exp(I*x)*sin(pi*x), x, oo))


def test_issue_15146():
    # 断言测试复杂表达式在无穷大时的极限值是否为1/3
    e = (x/2) * (-2*x**3 - 2*(x**3 - 1) * x**2 * digamma(x**3 + 1) + \
        2*(x**3 - 1) * x**2 * digamma(x**3 + x + 1) + x + 3)
    assert limit(e, x, oo) == S(1)/3


def test_issue_15202():
    # 断言测试复杂表达式在无穷大时的极限值是否为exp(1)和10
    e = (2**x*(2 + 2**(-x)*(-2*2**x + x + 2))/(x + 1))**(x + 1)
    assert limit(e, x, oo) == exp(1)

    e = (log(x, 2)**7 + 10*x*factorial(x) + 5**x) / (factorial(x + 1) + 3*factorial(x) + 10**x)
    assert limit(e, x, oo) == 10


def test_issue_15282():
    # 断言测试复杂表达式在无穷大时的极限值是否为-2000
    assert limit((x**2000 - (x + 1)**2000) / x**1999, x, oo) == -2000


def test_issue_15984():
    # 断言测试特定表达式在接近0时的单侧极限是否为0
    assert limit((-x + log(exp(x) + 1))/x, x, oo, dir='-') == 0


def test_issue_13571():
    # 断言测试特定函数的极限是否为1
    assert limit(uppergamma(x, 1) / gamma(x), x, oo) == 1


def test_issue_13575():
    # 断言测试特定函数在给定点的极限是否等于acos(erfi(1))
    assert limit(acos(erfi(x)), x, 1) == acos(erfi(S.One))


def test_issue_17325():
    # 断言测试特定表达式在接近0时不同方向的极限是否正确计算
    assert Limit(sin(x)/x, x, 0, dir="+-").doit() == 1
    assert Limit(x**2, x, 0, dir="+-").doit() == 0
    assert Limit(1/x**2, x, 0, dir="+-").doit() is oo
    assert Limit(1/x, x, 0, dir="+-").doit() is zoo


def test_issue_10978():
    # 断言测试兰伯特W函数在接近0时的极限是否为0
    assert LambertW(x).limit(x, 0) == 0


def test_issue_14313_comment():
    # 断言测试取整函数在无穷大时的极限是否为无穷大
    assert limit(floor(n/2), n, oo) is oo


@XFAIL
def test_issue_15323():
    # 断言测试特定表达式在给定点的右侧导数是否为1
    d = ((1 - 1/x)**x).diff(x)
    assert limit(d, x, 1, dir='+') == 1


def test_issue_12571():
    # 断言测试特定表达式在接近1时的极限是否为1
    assert limit(-LambertW(-log(x))/log(x), x, 1) == 1


def test_issue_14590():
    # 断言测试特定表达式在无穷大时的极限是否为exp(1)
    assert limit((x**3*((x + 1)/x)**x)/((x + 1)*(x + 2)*(x + 3)), x, oo) == exp(1)


def test_issue_14393():
    # 断言测试特定表达式在给定点的极限是否正确计算
    a, b = symbols('a b')
    assert limit((x**b - y**b)/(x**a - y**a), x, y) == b*y**(-a + b)/a


def test_issue_14556():
    # 这个测试函数暂时没有给出具体的测试表达式或断言，注释留空
    pass
    # 断言：验证当 n 趋向无穷大时，表达式的极限是否等于 exp(-1)
    assert limit(factorial(n + 1)**(1/(n + 1)) - factorial(n)**(1/n), n, oo) == exp(-1)
def test_issue_14811():
    assert limit(((1 + ((S(2)/3)**(x + 1)))**(2**x))/(2**((S(4)/3)**(x - 1))), x, oo) == oo
    # 测试函数：计算表达式在 x 趋向无穷时的极限，期望结果为无穷


def test_issue_16222():
    assert limit(exp(x), x, 1000000000) == exp(1000000000)
    # 测试函数：计算指数函数在 x 趋向 1000000000 时的极限，期望结果为 exp(1000000000)


def test_issue_16714():
    assert limit(((x**(x + 1) + (x + 1)**x) / x**(x + 1))**x, x, oo) == exp(exp(1))
    # 测试函数：计算复合指数函数在 x 趋向无穷时的极限，期望结果为 exp(exp(1))


def test_issue_16722():
    z = symbols('z', positive=True)
    assert limit(binomial(n + z, n)*n**-z, n, oo) == 1/gamma(z + 1)
    z = symbols('z', positive=True, integer=True)
    assert limit(binomial(n + z, n)*n**-z, n, oo) == 1/gamma(z + 1)
    # 测试函数：计算二项式系数在 n 趋向无穷时的极限，期望结果为 1/gamma(z + 1)，其中 z 为正整数


def test_issue_17431():
    assert limit(((n + 1) + 1) / (((n + 1) + 2) * factorial(n + 1)) *
                 (n + 2) * factorial(n) / (n + 1), n, oo) == 0
    assert limit((n + 2)**2*factorial(n)/((n + 1)*(n + 3)*factorial(n + 1))
                 , n, oo) == 0
    assert limit((n + 1) * factorial(n) / (n * factorial(n + 1)), n, oo) == 0
    # 测试函数：计算数学表达式在 n 趋向无穷时的极限，期望结果为 0


def test_issue_17671():
    assert limit(Ei(-log(x)) - log(log(x))/x, x, 1) == EulerGamma
    # 测试函数：计算特殊函数在 x 趋向 1 时的极限，期望结果为 EulerGamma


def test_issue_17751():
    a, b, c, x = symbols('a b c x', positive=True)
    assert limit((a + 1)*x - sqrt((a + 1)**2*x**2 + b*x + c), x, oo) == -b/(2*a + 2)
    # 测试函数：计算复杂表达式在 x 趋向无穷时的极限，期望结果为 -b/(2*a + 2)


def test_issue_17792():
    assert limit(factorial(n)/sqrt(n)*(exp(1)/n)**n, n, oo) == sqrt(2)*sqrt(pi)
    # 测试函数：计算复合表达式在 n 趋向无穷时的极限，期望结果为 sqrt(2)*sqrt(pi)


def test_issue_18118():
    assert limit(sign(sin(x)), x, 0, "-") == -1
    assert limit(sign(sin(x)), x, 0, "+") == 1
    # 测试函数：计算带有符号极限的函数在 x 趋向 0 时的极限，分别为 -1 和 1


def test_issue_18306():
    assert limit(sin(sqrt(x))/sqrt(sin(x)), x, 0, '+') == 1
    # 测试函数：计算三角函数的复合表达式在 x 趋向 0 时的极限，期望结果为 1


def test_issue_18378():
    assert limit(log(exp(3*x) + x)/log(exp(x) + x**100), x, oo) == 3
    # 测试函数：计算对数表达式在 x 趋向无穷时的极限，期望结果为 3


def test_issue_18399():
    assert limit((1 - S(1)/2*x)**(3*x), x, oo) is zoo
    assert limit((-x)**x, x, oo) is zoo
    # 测试函数：计算在 x 趋向无穷时的极限，结果应为 zoo (无穷大)


def test_issue_18442():
    assert limit(tan(x)**(2**(sqrt(pi))), x, oo, dir='-') == Limit(tan(x)**(2**(sqrt(pi))), x, oo, dir='-')
    # 测试函数：计算正切函数的幂函数在 x 趋向无穷时的极限，期望结果为 Limit 对象


def test_issue_18452():
    assert limit(abs(log(x))**x, x, 0) == 1
    assert limit(abs(log(x))**x, x, 0, "-") == 1
    # 测试函数：计算对数函数的绝对值的幂函数在 x 趋向 0 时的极限，期望结果为 1


def test_issue_18473():
    assert limit(sin(x)**(1/x), x, oo) == Limit(sin(x)**(1/x), x, oo, dir='-')
    assert limit(cos(x)**(1/x), x, oo) == Limit(cos(x)**(1/x), x, oo, dir='-')
    assert limit(tan(x)**(1/x), x, oo) == Limit(tan(x)**(1/x), x, oo, dir='-')
    assert limit((cos(x) + 2)**(1/x), x, oo) == 1
    assert limit((sin(x) + 10)**(1/x), x, oo) == 1
    assert limit((cos(x) - 2)**(1/x), x, oo) == Limit((cos(x) - 2)**(1/x), x, oo, dir='-')
    assert limit((cos(x) + 1)**(1/x), x, oo) == AccumBounds(0, 1)
    assert limit((tan(x)**2)**(2/x) , x, oo) == AccumBounds(0, oo)
    assert limit((sin(x)**2)**(1/x), x, oo) == AccumBounds(0, 1)
    # 测试函数：计算三角函数的幂函数在 x 趋向无穷时的极限，以及在负无穷时的情况


def test_issue_18482():
    assert limit((2*exp(3*x)/(exp(2*x) + 1))**(1/x), x, oo) == exp(1)
    # 测试函数：计算指数函数的复合表达式在 x 趋向无穷时的极限，期望结果为 exp(1)


def test_issue_18508():
    pass
    # 待补充测试函数
    # 断言：计算函数 sin(x)/sqrt(1-cos(x)) 在 x 趋近 0 时的极限，预期结果为 sqrt(2)
    assert limit(sin(x)/sqrt(1-cos(x)), x, 0) == sqrt(2)
    # 断言：计算函数 sin(x)/sqrt(1-cos(x)) 在 x 从正方向趋近 0 时的极限，预期结果为 sqrt(2)
    assert limit(sin(x)/sqrt(1-cos(x)), x, 0, dir='+') == sqrt(2)
    # 断言：计算函数 sin(x)/sqrt(1-cos(x)) 在 x 从负方向趋近 0 时的极限，预期结果为 -sqrt(2)
    assert limit(sin(x)/sqrt(1-cos(x)), x, 0, dir='-') == -sqrt(2)
# 测试函数，验证问题 #18521 的限制条件
def test_issue_18521():
    # 断言抛出 NotImplementedError 异常，lambda 表达式计算 limit(exp((2 - n) * x), x, oo)
    raises(NotImplementedError, lambda: limit(exp((2 - n) * x), x, oo))


# 测试函数，验证问题 #18969 的限制条件
def test_issue_18969():
    # 声明符号变量 a, b，并设置为正数
    a, b = symbols('a b', positive=True)
    # 断言 LambertW(a) 的极限值为 LambertW(b)
    assert limit(LambertW(a), a, b) == LambertW(b)
    # 断言 exp(LambertW(a)) 的极限值为 exp(LambertW(b))
    assert limit(exp(LambertW(a)), a, b) == exp(LambertW(b))


# 测试函数，验证问题 #18992 的限制条件
def test_issue_18992():
    # 断言 limit(n/(factorial(n)**(1/n)), n, oo) 的极限值为 exp(1)
    assert limit(n/(factorial(n)**(1/n)), n, oo) == exp(1)


# 测试函数，验证问题 #19067 的限制条件
def test_issue_19067():
    # 声明符号变量 x
    x = Symbol('x')
    # 断言 limit(gamma(x)/(gamma(x - 1)*gamma(x + 2)), x, 0) 的极限值为 -1
    assert limit(gamma(x)/(gamma(x - 1)*gamma(x + 2)), x, 0) == -1


# 测试函数，验证问题 #19586 的限制条件
def test_issue_19586():
    # 断言 limit(x**(2**x*3**(-x)), x, oo) 的极限值为 1
    assert limit(x**(2**x*3**(-x)), x, oo) == 1


# 测试函数，验证问题 #13715 的限制条件
def test_issue_13715():
    # 声明符号变量 n 和 p，并设置 p 为零
    n = Symbol('n')
    p = Symbol('p', zero=True)
    # 断言 limit(n + p, n, 0) 的极限值为 0
    assert limit(n + p, n, 0) == 0


# 测试函数，验证问题 #15055 的限制条件
def test_issue_15055():
    # 断言 limit(n**3*((-n - 1)*sin(1/n) + (n + 2)*sin(1/(n + 1)))/(-n + 1), n, oo) 的极限值为 1
    assert limit(n**3*((-n - 1)*sin(1/n) + (n + 2)*sin(1/(n + 1)))/(-n + 1), n, oo) == 1


# 测试函数，验证问题 #16708 的限制条件
def test_issue_16708():
    # 声明符号变量 m, vi，并设置为正数；声明符号变量 B, ti, d
    m, vi = symbols('m vi', positive=True)
    B, ti, d = symbols('B ti d')
    # 断言 limit((B*ti*vi - sqrt(m)*sqrt(-2*B*d*vi + m*(vi)**2) + m*vi)/(B*vi), B, 0) 的极限值为 (d + ti*vi)/vi
    assert limit((B*ti*vi - sqrt(m)*sqrt(-2*B*d*vi + m*(vi)**2) + m*vi)/(B*vi), B, 0) == (d + ti*vi)/vi


# 测试函数，验证问题 #19154 的限制条件
def test_issue_19154():
    # 断言 limit(besseli(1, 3 *x)/(x *besseli(1, x)**3), x , oo) 的极限值为 2*sqrt(3)*pi/3
    assert limit(besseli(1, 3 *x)/(x *besseli(1, x)**3), x , oo) == 2*sqrt(3)*pi/3
    # 断言 limit(besseli(1, 3 *x)/(x *besseli(1, x)**3), x , -oo) 的极限值为 -2*sqrt(3)*pi/3
    assert limit(besseli(1, 3 *x)/(x *besseli(1, x)**3), x , -oo) == -2*sqrt(3)*pi/3


# 测试函数，验证问题 #19453 的限制条件
def test_issue_19453():
    # 声明符号变量 beta, h, m, w, g，并设置为正数
    beta = Symbol("beta", positive=True)
    h = Symbol("h", positive=True)
    m = Symbol("m", positive=True)
    w = Symbol("omega", positive=True)  # omega 符号表示 w
    g = Symbol("g", positive=True)

    # 计算数学常数 e 和相关的计算
    e = exp(1)
    q = 3*h**2*beta*g*e**(0.5*h*beta*w)
    p = m**2*w**2
    s = e**(h*beta*w) - 1
    Z = -q/(4*p*s) - q/(2*p*s**2) - q*(e**(h*beta*w) + 1)/(2*p*s**3)\
            + e**(0.5*h*beta*w)/s
    E = -diff(log(Z), beta)

    # 断言 limit(E - 0.5*h*w, beta, oo) 的极限值为 0
    assert limit(E - 0.5*h*w, beta, oo) == 0
    # 断言 limit(E.simplify() - 0.5*h*w, beta, oo) 的极限值为 0
    assert limit(E.simplify() - 0.5*h*w, beta, oo) == 0


# 测试函数，验证问题 #19739 的限制条件
def test_issue_19739():
    # 断言 limit((-S(1)/4)**x, x, oo) 的极限值为 0
    assert limit((-S(1)/4)**x, x, oo) == 0


# 测试函数，验证问题 #19766 的限制条件
def test_issue_19766():
    # 断言 limit(2**(-x)*sqrt(4**(x + 1) + 1), x, oo) 的极限值为 2
    assert limit(2**(-x)*sqrt(4**(x + 1) + 1), x, oo) == 2


# 测试函数，验证问题 #19770 的限制条件
def test_issue_19770():
    m = Symbol('m')
    # 对于非实数 m，断言 limit(cos(m*x)/x, x, oo) 的极限为非定值
    assert limit(cos(m*x)/x, x, oo) == Limit(cos(m*x)/x, x, oo, dir='-')
    m = Symbol('m', real=True)
    # 对于实数 m，断言 limit(cos(m*x)/x, x, oo) 的极限为 0
    assert limit(cos(m*x)/x, x, oo) == Limit(cos(m*x)/x, x, oo, dir='-')
    m = Symbol('m', nonzero=True)
    # 对于非零 m，断言 limit(cos(m*x), x, oo) 的极限为在区间 [-1, 1] 的累积边界
    assert limit(cos(m*x), x, oo) == AccumBounds(-1, 1)
    # 断言 limit(cos(m*x)/x, x, oo) 的极限值为 0
    assert limit(cos(m*x)/x, x, oo) == 0


# 测试函数，验证问题 #7535 的限制条件
def test_issue_7535():
    # 断言 limit(tan(x)/sin(tan(x)), x, pi/2) 的极限方向为正向
    assert limit(tan(x)/sin(tan(x)), x, pi/2) == Limit(tan(x)/sin(tan(x)), x, pi/2, dir='+')
    # 断言 limit(tan(x)/sin(tan(x)), x, pi/2, dir='-') 的极限方向为负向
    assert limit(tan(x)/sin(tan(x)), x, pi/2, dir='-') == Limit(tan(x)/sin(tan(x)), x, pi/2, dir='-')
    # 断言 limit(tan(x)/sin(tan(x
def test_issue_21031():
    # 断言测试极限表达式 ((1 + x)**(1/x) - (1 + 2*x)**(1/(2*x)))/asin(x) 在 x 趋向 0 时等于 E/2
    assert limit(((1 + x)**(1/x) - (1 + 2*x)**(1/(2*x)))/asin(x), x, 0) == E/2


def test_issue_21038():
    # 断言测试极限表达式 sin(pi*x)/(3*x - 12) 在 x 趋向 4 时等于 pi/3
    assert limit(sin(pi*x)/(3*x - 12), x, 4) == pi/3


def test_issue_20578():
    # 定义表达式 expr = abs(x) * sin(1/x)
    expr = abs(x) * sin(1/x)
    # 断言测试极限表达式 expr 在 x 趋向 0 时从正向接近等于 0
    assert limit(expr,x,0,'+') == 0
    # 断言测试极限表达式 expr 在 x 趋向 0 时从负向接近等于 0
    assert limit(expr,x,0,'-') == 0
    # 断言测试极限表达式 expr 在 x 趋向 0 时双向接近等于 0
    assert limit(expr,x,0,'+-') == 0


def test_issue_21227():
    # 定义函数 f = log(x)
    f = log(x)

    # 断言测试 f 的 n 次级数展开，关于 logx=y 的情况
    assert f.nseries(x, logx=y) == y
    # 断言测试 f 的 n 次级数展开，关于 logx=-x 的情况
    assert f.nseries(x, logx=-x) == -x

    # 修改 f 为 log(-log(x))
    f = log(-log(x))

    # 断言测试 f 的 n 次级数展开，关于 logx=y 的情况
    assert f.nseries(x, logx=y) == log(-y)
    # 断言测试 f 的 n 次级数展开，关于 logx=-x 的情况
    assert f.nseries(x, logx=-x) == log(x)

    # 修改 f 为 log(log(x))
    f = log(log(x))

    # 断言测试 f 的 n 次级数展开，关于 logx=y 的情况
    assert f.nseries(x, logx=y) == log(y)
    # 断言测试 f 的 n 次级数展开，关于 logx=-x 的情况
    assert f.nseries(x, logx=-x) == log(-x)
    # 断言测试 f 的 n 次级数展开，关于 logx=x 的情况
    assert f.nseries(x, logx=x) == log(x)

    # 修改 f 为 log(log(log(1/x)))
    f = log(log(log(1/x)))

    # 断言测试 f 的 n 次级数展开，关于 logx=y 的情况
    assert f.nseries(x, logx=y) == log(log(-y))
    # 断言测试 f 的 n 次级数展开，关于 logx=-y 的情况
    assert f.nseries(x, logx=-y) == log(log(y))
    # 断言测试 f 的 n 次级数展开，关于 logx=x 的情况
    assert f.nseries(x, logx=x) == log(log(-x))
    # 断言测试 f 的 n 次级数展开，关于 logx=-x 的情况
    assert f.nseries(x, logx=-x) == log(log(x))


def test_issue_21415():
    # 定义表达式 exp = (x-1)*cos(1/(x-1))
    exp = (x-1)*cos(1/(x-1))
    # 断言测试 exp 在 x 趋向 1 时的极限等于 0
    assert exp.limit(x,1) == 0
    # 断言测试对 exp 展开后在 x 趋向 1 时的极限等于 0
    assert exp.expand().limit(x,1) == 0


def test_issue_21530():
    # 断言测试极限表达式 sinh(n + 1)/sinh(n) 在 n 趋向 oo 时等于 E
    assert limit(sinh(n + 1)/sinh(n), n, oo) == E


def test_issue_21550():
    r = (sqrt(5) - 1)/2
    # 断言测试极限表达式 (x - r)/(x**2 + x - 1) 在 x 趋向 r 时等于 sqrt(5)/5
    assert limit((x - r)/(x**2 + x - 1), x, r) == sqrt(5)/5


def test_issue_21661():
    # 计算极限表达式 (x**(x + 1) * (log(x) + 1) + 1) / x 在 x 趋向 11 时的值
    out = limit((x**(x + 1) * (log(x) + 1) + 1) / x, x, 11)
    # 断言 out 的结果
    assert out == S(3138428376722)/11 + 285311670611*log(11)


def test_issue_21701():
    # 断言测试极限表达式 (besselj(z, x)/x**z).subs(z, 7) 在 x 趋向 0 时等于 1/645120
    assert limit((besselj(z, x)/x**z).subs(z, 7), x, 0) == S(1)/645120


def test_issue_21721():
    a = Symbol('a', real=True)
    # 计算积分表达式 I = integrate(1/(pi*(1 + (x - a)**2)), x)
    I = integrate(1/(pi*(1 + (x - a)**2)), x)
    # 断言测试 I 在 x 趋向 oo 时等于 1/2
    assert I.limit(x, oo) == S.Half


def test_issue_21756():
    # 定义表达式 term = (1 - exp(-2*I*pi*z))/(1 - exp(-2*I*pi*z/5))
    term = (1 - exp(-2*I*pi*z))/(1 - exp(-2*I*pi*z/5))
    # 断言测试 term 在 z 趋向 0 时的极限等于 5
    assert term.limit(z, 0) == 5
    # 断言测试 re(term) 在 z 趋向 0 时的实部极限等于 5
    assert re(term).limit(z, 0) == 5


def test_issue_21785():
    a = Symbol('a')
    # 断言测试 sqrt((-a**2 + x**2)/(1 - x**2)) 在 a 趋向 1 时从负向接近等于 I
    assert sqrt((-a**2 + x**2)/(1 - x**2)).limit(a, 1, '-') == I


def test_issue_22181():
    # 断言测试极限表达式 (-1)**x * 2**(-x) 在 x 趋向 oo 时等于 0
    assert limit((-1)**x * 2**(-x), x, oo) == 0


def test_issue_22220():
    # 定义表达式 e1 和 e2
    e1 = sqrt(30)*atan(sqrt(30)*tan(x/2)/6)/30
    e2 = sqrt(30)*I*(-log(sqrt(2)*tan(x/2) - 2*sqrt(15)*I/5) +
                     +log(sqrt(2)*tan(x/2) + 2*sqrt(15)*I/5))/60

    # 断言测试 e1 在 x 趋向 -pi 时的极限等于 -sqrt(30)*pi/60
    assert limit(e1, x, -pi) == -sqrt(30)*pi/60
    # 断言测试 e2 在 x 趋向 -pi 时的极限等于 -sqrt(30)*pi/30
    assert limit(e2, x, -pi) == -sqrt(30)*pi/30

    # 断言测试 e1 在 x 趋向 -pi 时的左侧极限等于 sqrt(30)*pi/
    # 断言：对于极限 lim_{n -> oo} ((n+1)**k / ((n+1)**(k+1) - (n)**(k+1)))，其结果应为 1/(k + 1)
    assert limit((n+1)**k / ((n+1)**(k+1) - (n)**(k+1)).expand(), n, oo) == 1/(k + 1)
    
    # 断言：对于极限 lim_{n -> oo} ((n+1)**k / (n*(-n**k + (n + 1)**k) + (n + 1)**k))，其结果应为 1/(k + 1)
    assert limit((n+1)**k / (n*(-n**k + (n + 1)**k) + (n + 1)**k), n, oo) == 1/(k + 1)
def test_sympyissue_22986():
    # Assert that the limit of acosh(1 + 1/x) * sqrt(x) as x approaches infinity is sqrt(2)
    assert limit(acosh(1 + 1/x)*sqrt(x), x, oo) == sqrt(2)


def test_issue_23231():
    # Define the function f(x) = (2**x - 2**(-x)) / (2**x + 2**(-x))
    f = (2**x - 2**(-x))/(2**x + 2**(-x))
    # Assert that the limit of f(x) as x approaches negative infinity is -1
    assert limit(f, x, -oo) == -1


def test_issue_23596():
    # Assert that the integral of ((1 + x) / x**2) * exp(-1/x) from 0 to infinity is infinity
    assert integrate(((1 + x)/x**2)*exp(-1/x), (x, 0, oo)) == oo


def test_issue_23752():
    # Define expressions expr1 and expr2 involving square roots with complex numbers
    expr1 = sqrt(-I*x**2 + x - 3)
    expr2 = sqrt(-I*x**2 + I*x - 3)
    # Assert the limits of expr1 and expr2 as x approaches 0 from both sides
    assert limit(expr1, x, 0, '+') == -sqrt(3)*I
    assert limit(expr1, x, 0, '-') == -sqrt(3)*I
    assert limit(expr2, x, 0, '+') == sqrt(3)*I
    assert limit(expr2, x, 0, '-') == -sqrt(3)*I


def test_issue_24276():
    # Define fx = log(tan(pi/2 * tanh(x))).diff(x)
    fx = log(tan(pi/2*tanh(x))).diff(x)
    # Assert various limits of fx after applying simplifications and rewrites
    assert fx.limit(x, oo) == 2
    assert fx.simplify().limit(x, oo) == 2
    assert fx.rewrite(sin).limit(x, oo) == 2
    assert fx.rewrite(sin).simplify().limit(x, oo) == 2


def test_issue_25230():
    # Define symbols with specific properties
    a = Symbol('a', real=True)
    b = Symbol('b', positive=True)
    c = Symbol('c', negative=True)
    n = Symbol('n', integer=True)
    # Raise NotImplementedError for limit(Mod(x, a), x, a)
    raises(NotImplementedError, lambda: limit(Mod(x, a), x, a))
    # Assert limits involving the Mod function with different signs and symbol properties
    assert limit(Mod(x, b), x, n*b, '+') == 0
    assert limit(Mod(x, b), x, n*b, '-') == b
    assert limit(Mod(x, c), x, n*c, '+') == c
    assert limit(Mod(x, c), x, n*c, '-') == 0


def test_issue_25582():
    # Assert limits involving inverse trigonometric and hyperbolic functions with exp(x) as x approaches infinity
    assert limit(asin(exp(x)), x, oo, '-') == -oo*I
    assert limit(acos(exp(x)), x, oo, '-') == oo*I
    assert limit(atan(exp(x)), x, oo, '-') == pi/2
    assert limit(acot(exp(x)), x, oo, '-') == 0
    assert limit(asec(exp(x)), x, oo, '-') == pi/2
    assert limit(acsc(exp(x)), x, oo, '-') == 0


def test_issue_25847():
    # Assert limits involving various trigonometric and hyperbolic functions with exp(1/x) as x approaches 0 from both sides
    # for atan, asin, acos, acot, asec, acsc, atanh, asinh, acosh
    assert limit(atan(sin(x)/x), x, 0, '+-') == pi/4
    assert limit(atan(exp(1/x)), x, 0, '+') == pi/2
    assert limit(atan(exp(1/x)), x, 0, '-') == 0

    assert limit(asin(sin(x)/x), x, 0, '+-') == pi/2
    assert limit(asin(exp(1/x)), x, 0, '+') == -oo*I
    assert limit(asin(exp(1/x)), x, 0, '-') == 0

    assert limit(acos(sin(x)/x), x, 0, '+-') == 0
    assert limit(acos(exp(1/x)), x, 0, '+') == oo*I
    assert limit(acos(exp(1/x)), x, 0, '-') == pi/2

    assert limit(acot(sin(x)/x), x, 0, '+-') == pi/4
    assert limit(acot(exp(1/x)), x, 0, '+') == 0
    assert limit(acot(exp(1/x)), x, 0, '-') == pi/2

    assert limit(asec(sin(x)/x), x, 0, '+-') == 0
    assert limit(asec(exp(1/x)), x, 0, '+') == pi/2
    assert limit(asec(exp(1/x)), x, 0, '-') == oo*I

    assert limit(acsc(sin(x)/x), x, 0, '+-') == pi/2
    assert limit(acsc(exp(1/x)), x, 0, '+') == 0
    assert limit(acsc(exp(1/x)), x, 0, '-') == -oo*I

    assert limit(atanh(sin(x)/x), x, 0, '+-') == oo
    assert limit(atanh(exp(1/x)), x, 0, '+') == -I*pi/2
    assert limit(atanh(exp(1/x)), x, 0, '-') == 0

    assert limit(asinh(sin(x)/x), x, 0, '+-') == log(1 + sqrt(2))
    assert limit(asinh(exp(1/x)), x, 0, '+') == oo
    assert limit(asinh(exp(1/x)), x, 0, '-') == 0

    assert limit(acosh(sin(x)/x), x, 0, '+-') == 0
    # 验证 x 趋近于 0 时 acosh(exp(1/x)) 的极限是否为正无穷
    assert limit(acosh(exp(1/x)), x, 0, '+') == oo
    # 验证 x 趋近于 0 时 acosh(exp(1/x)) 的极限是否为虚数轴上的π/2
    assert limit(acosh(exp(1/x)), x, 0, '-') == I*pi/2
    
    #acoth
    # 验证 x 趋近于 0 时 acoth(sin(x)/x) 的极限是否为正无穷或负无穷
    assert limit(acoth(sin(x)/x), x, 0, '+-') == oo
    # 验证 x 趋近于 0 时 acoth(exp(1/x)) 的极限是否为 0
    assert limit(acoth(exp(1/x)), x, 0, '+') == 0
    # 验证 x 趋近于 0 时 acoth(exp(1/x)) 的极限是否为虚数轴上的 -π/2
    assert limit(acoth(exp(1/x)), x, 0, '-') == -I*pi/2
    
    #asech
    # 验证 x 趋近于 0 时 asech(sin(x)/x) 的极限是否为 0
    assert limit(asech(sin(x)/x), x, 0, '+-') == 0
    # 验证 x 趋近于 0 时 asech(exp(1/x)) 的极限是否为虚数轴上的π/2
    assert limit(asech(exp(1/x)), x, 0, '+') == I*pi/2
    # 验证 x 趋近于 0 时 asech(exp(1/x)) 的极限是否为正无穷
    assert limit(asech(exp(1/x)), x, 0, '-') == oo
    
    #acsch
    # 验证 x 趋近于 0 时 acsch(sin(x)/x) 的极限是否为 log(1 + sqrt(2))
    assert limit(acsch(sin(x)/x), x, 0, '+-') == log(1 + sqrt(2))
    # 验证 x 趋近于 0 时 acsch(exp(1/x)) 的极限是否为 0
    assert limit(acsch(exp(1/x)), x, 0, '+') == 0
    # 验证 x 趋近于 0 时 acsch(exp(1/x)) 的极限是否为正无穷
    assert limit(acsch(exp(1/x)), x, 0, '-') == oo
# 测试问题编号 26040 的函数
def test_issue_26040():
    # 断言表达式的极限等于自然指数的底数 S.Exp1
    assert limit(besseli(0, x + 1)/besseli(0, x), x, oo) == S.Exp1


# 测试问题编号 26250 的函数
def test_issue_26250():
    # 计算椭圆函数 e(4*x/(x**2 + 2*x + 1))
    e = elliptic_e(4*x/(x**2 + 2*x + 1))
    # 计算椭圆函数 k(4*x/(x**2 + 2*x + 1))
    k = elliptic_k(4*x/(x**2 + 2*x + 1))
    # 计算 e1 的值
    e1 = ((1-3*x**2)*e**2/2 - (x**2-2*x+1)*e*k/2)
    # 计算 e2 的值
    e2 = pi**2*(x**8 - 2*x**7 - x**6 + 4*x**5 - x**4 - 2*x**3 + x**2)
    # 断言 e1/e2 的极限等于 -1/8，其中 x 趋近于 0
    assert limit(e1/e2, x, 0) == -S(1)/8
```