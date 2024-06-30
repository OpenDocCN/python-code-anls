# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_gamma_functions.py`

```
from sympy.core.function import expand_func, Subs  # 导入函数 expand_func 和 Subs
from sympy.core import EulerGamma  # 导入常数 EulerGamma
from sympy.core.numbers import (I, Rational, nan, oo, pi, zoo)  # 导入复数、有理数、特殊浮点数常量
from sympy.core.singleton import S  # 导入 SymPy 单例 S
from sympy.core.symbol import (Dummy, Symbol)  # 导入符号类 Dummy 和 Symbol
from sympy.functions.combinatorial.factorials import factorial  # 导入阶乘函数 factorial
from sympy.functions.combinatorial.numbers import harmonic  # 导入调和数函数 harmonic
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re)  # 导入复数函数
from sympy.functions.elementary.exponential import (exp, exp_polar, log)  # 导入指数和对数函数
from sympy.functions.elementary.hyperbolic import tanh  # 导入双曲函数 tanh
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import (cos, sin, atan)  # 导入三角函数
from sympy.functions.special.error_functions import (Ei, erf, erfc)  # 导入误差函数
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, lowergamma, multigamma, polygamma, trigamma, uppergamma)  # 导入 Gamma 函数及其相关函数
from sympy.functions.special.zeta_functions import zeta  # 导入 Riemann zeta 函数
from sympy.series.order import O  # 导入 O 大O符号

from sympy.core.expr import unchanged  # 导入 unchanged 函数
from sympy.core.function import ArgumentIndexError  # 导入异常类 ArgumentIndexError
from sympy.testing.pytest import raises  # 导入测试相关函数 raises
from sympy.core.random import (test_derivative_numerically as td,  # 导入数值微分测试函数
                                random_complex_number as randcplx,  # 导入生成随机复数函数
                                verify_numerically as tn)  # 导入数值验证函数

x = Symbol('x')  # 创建符号 x
y = Symbol('y')  # 创建符号 y
n = Symbol('n', integer=True)  # 创建整数符号 n
w = Symbol('w', real=True)  # 创建实数符号 w

def test_gamma():
    assert gamma(nan) is nan  # 检查 gamma 函数对 nan 的返回值
    assert gamma(oo) is oo  # 检查 gamma 函数对无穷大的返回值

    assert gamma(-100) is zoo  # 检查 gamma 函数对 -100 的返回值
    assert gamma(0) is zoo  # 检查 gamma 函数对 0 的返回值
    assert gamma(-100.0) is zoo  # 检查 gamma 函数对 -100.0 的返回值

    assert gamma(1) == 1  # 检查 gamma 函数对 1 的返回值
    assert gamma(2) == 1  # 检查 gamma 函数对 2 的返回值
    assert gamma(3) == 2  # 检查 gamma 函数对 3 的返回值

    assert gamma(102) == factorial(101)  # 检查 gamma 函数对 102 的返回值

    assert gamma(S.Half) == sqrt(pi)  # 检查 gamma 函数对 S.Half 的返回值

    assert gamma(Rational(3, 2)) == sqrt(pi)*S.Half  # 检查 gamma 函数对 3/2 的返回值
    assert gamma(Rational(5, 2)) == sqrt(pi)*Rational(3, 4)  # 检查 gamma 函数对 5/2 的返回值
    assert gamma(Rational(7, 2)) == sqrt(pi)*Rational(15, 8)  # 检查 gamma 函数对 7/2 的返回值

    assert gamma(Rational(-1, 2)) == -2*sqrt(pi)  # 检查 gamma 函数对 -1/2 的返回值
    assert gamma(Rational(-3, 2)) == sqrt(pi)*Rational(4, 3)  # 检查 gamma 函数对 -3/2 的返回值
    assert gamma(Rational(-5, 2)) == sqrt(pi)*Rational(-8, 15)  # 检查 gamma 函数对 -5/2 的返回值

    assert gamma(Rational(-15, 2)) == sqrt(pi)*Rational(256, 2027025)  # 检查 gamma 函数对 -15/2 的返回值

    assert gamma(Rational(
        -11, 8)).expand(func=True) == Rational(64, 33)*gamma(Rational(5, 8))  # 检查 gamma 函数对 -11/8 的展开值
    assert gamma(Rational(
        -10, 3)).expand(func=True) == Rational(81, 280)*gamma(Rational(2, 3))  # 检查 gamma 函数对 -10/3 的展开值
    assert gamma(Rational(
        14, 3)).expand(func=True) == Rational(880, 81)*gamma(Rational(2, 3))  # 检查 gamma 函数对 14/3 的展开值
    assert gamma(Rational(
        17, 7)).expand(func=True) == Rational(30, 49)*gamma(Rational(3, 7))  # 检查 gamma 函数对 17/7 的展开值
    assert gamma(Rational(
        19, 8)).expand(func=True) == Rational(33, 64)*gamma(Rational(3, 8))  # 检查 gamma 函数对 19/8 的展开值

    assert gamma(x).diff(x) == gamma(x)*polygamma(0, x)  # 检查 gamma 函数的导数

    assert gamma(x - 1).expand(func=True) == gamma(x)/(x - 1)  # 检查 gamma 函数在 x - 1 处的展开值
    assert gamma(x + 2).expand(func=True, mul=False) == x*(x + 1)*gamma(x)  # 检查 gamma 函数在 x + 2 处的展开值

    assert conjugate(gamma(x)) == gamma(conjugate(x))  # 检查 gamma 函数的共轭
    # 断言：验证 gamma 函数在给定表达式上的展开是否符合预期
    assert expand_func(gamma(x + Rational(3, 2))) == \
        (x + S.Half)*gamma(x + S.Half)

    # 断言：验证 gamma 函数在给定表达式上的展开是否符合预期
    assert expand_func(gamma(x - S.Half)) == \
        gamma(S.Half + x)/(x - S.Half)

    # 测试一个已知 bug：
    assert expand_func(gamma(x + Rational(3, 4))) == gamma(x + Rational(3, 4))

    # XXX: 对这些测试不确定。可以通过定义 exp_polar.is_integer 来修复，但不确定是否合理。
    # 断言：验证 gamma 函数在复数域上的性质
    assert gamma(3*exp_polar(I*pi)/4).is_nonnegative is False
    assert gamma(3*exp_polar(I*pi)/4).is_extended_nonpositive is True

    # 符号 y 被定义为非正整数
    y = Symbol('y', nonpositive=True, integer=True)
    # 断言：验证 gamma 函数在给定符号条件下是否为实数
    assert gamma(y).is_real == False
    # 符号 y 被重新定义为正数且非整数
    y = Symbol('y', positive=True, noninteger=True)
    # 断言：验证 gamma 函数在给定符号条件下是否为实数
    assert gamma(y).is_real == True

    # 断言：验证 gamma 函数在给定参数上不评估时是否为实数
    assert gamma(-1.0, evaluate=False).is_real == False
    assert gamma(0, evaluate=False).is_real == False
    assert gamma(-2, evaluate=False).is_real == False
# 定义测试函数，用于验证 gamma 函数的重写操作是否正确
def test_gamma_rewrite():
    assert gamma(n).rewrite(factorial) == factorial(n - 1)

# 定义测试函数，用于验证 gamma 函数的级数展开是否正确
def test_gamma_series():
    # 验证 gamma(x + 1) 的级数展开
    assert gamma(x + 1).series(x, 0, 3) == \
        1 - EulerGamma*x + x**2*(EulerGamma**2/2 + pi**2/12) + O(x**3)
    # 验证 gamma(x) 的级数展开
    assert gamma(x).series(x, -1, 3) == \
        -1/(x + 1) + EulerGamma - 1 + (x + 1)*(-1 - pi**2/12 - EulerGamma**2/2 + \
       EulerGamma) + (x + 1)**2*(-1 - pi**2/12 - EulerGamma**2/2 + EulerGamma**3/6 - \
       polygamma(2, 1)/6 + EulerGamma*pi**2/12 + EulerGamma) + O((x + 1)**3, (x, -1))

# 定义函数 tn_branch，用于测试给定函数在复平面上的特定分支处是否近似相等
def tn_branch(s, func):
    from sympy.core.random import uniform
    # 从均匀分布中取值 c，范围在 [1, 5] 之间
    c = uniform(1, 5)
    # 计算 func(s, c*exp_polar(I*pi)) 和 func(s, c*exp_polar(-I*pi)) 的差异
    expr = func(s, c*exp_polar(I*pi)) - func(s, c*exp_polar(-I*pi))
    eps = 1e-15
    # 计算 func(s + eps, -c + eps*I) 和 func(s + eps, -c - eps*I) 的差异
    expr2 = func(s + eps, -c + eps*I) - func(s + eps, -c - eps*I)
    # 返回是否差异小于给定的阈值 1e-10
    return abs(expr.n() - expr2.n()).n() < 1e-10

# 定义测试函数，用于验证 lowergamma 函数的多种性质
def test_lowergamma():
    from sympy.functions.special.error_functions import expint
    from sympy.functions.special.hyper import meijerg
    # 验证 lowergamma(x, 0) 是否为 0
    assert lowergamma(x, 0) == 0
    # 验证 lowergamma(x, y).diff(y) 的导数是否正确
    assert lowergamma(x, y).diff(y) == y**(x - 1)*exp(-y)
    # 验证 td(lowergamma(randcplx(), y), y) 是否成立
    assert td(lowergamma(randcplx(), y), y)
    # 验证 td(lowergamma(x, randcplx()), x) 是否成立
    assert td(lowergamma(x, randcplx()), x)
    # 验证 lowergamma(x, y).diff(x) 的导数是否正确
    assert lowergamma(x, y).diff(x) == \
        gamma(x)*digamma(x) - uppergamma(x, y)*log(y) \
        - meijerg([], [1, 1], [0, 0, x], [], y)

    # 验证特定情况下的 lowergamma 函数的结果是否正确
    assert lowergamma(S.Half, x) == sqrt(pi)*erf(sqrt(x))
    assert not lowergamma(S.Half - 3, x).has(lowergamma)
    assert not lowergamma(S.Half + 3, x).has(lowergamma)
    assert lowergamma(S.Half, x, evaluate=False).has(lowergamma)
    assert tn(lowergamma(S.Half + 3, x, evaluate=False),
              lowergamma(S.Half + 3, x), x)
    assert tn(lowergamma(S.Half - 3, x, evaluate=False),
              lowergamma(S.Half - 3, x), x)

    # 验证在不同复平面上的分支下，lowergamma 函数的行为是否正确
    assert tn_branch(-3, lowergamma)
    assert tn_branch(-4, lowergamma)
    assert tn_branch(Rational(1, 3), lowergamma)
    assert tn_branch(pi, lowergamma)
    assert lowergamma(3, exp_polar(4*pi*I)*x) == lowergamma(3, x)
    assert lowergamma(y, exp_polar(5*pi*I)*x) == \
        exp(4*I*pi*y)*lowergamma(y, x*exp_polar(pi*I))
    assert lowergamma(-2, exp_polar(5*pi*I)*x) == \
        lowergamma(-2, x*exp_polar(I*pi)) + 2*pi*I

    # 验证 lowergamma 函数的共轭是否正确
    assert conjugate(lowergamma(x, y)) == lowergamma(conjugate(x), conjugate(y))
    assert conjugate(lowergamma(x, 0)) == 0
    assert unchanged(conjugate, lowergamma(x, -oo))

    # 验证 lowergamma 函数在不同情况下是否是亚解析函数
    assert lowergamma(0, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1)/3, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1, x, evaluate=False)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(x, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x + 1, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1/x, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(0, x + 1)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1)/3, x + 1)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(1, x + 1, evaluate=False)._eval_is_meromorphic(x, 0) == True
    # 断言，验证 lowergamma(x, x + 1) 的亚黎函数在复平面中是否是亚黎子函数
    assert lowergamma(x, x + 1)._eval_is_meromorphic(x, 0) == True
    # 断言，验证 lowergamma(x + 1, x + 1) 的亚黎函数在复平面中是否是亚黎子函数
    assert lowergamma(x + 1, x + 1)._eval_is_meromorphic(x, 0) == True
    # 断言，验证 lowergamma(1/x, x + 1) 的亚黎函数在复平面中是否是亚黎子函数
    assert lowergamma(1/x, x + 1)._eval_is_meromorphic(x, 0) == False
    # 断言，验证 lowergamma(0, 1/x) 的亚黎函数在复平面中是否是亚黎子函数
    assert lowergamma(0, 1/x)._eval_is_meromorphic(x, 0) == False
    # 断言，验证 lowergamma(1/3, 1/x) 的亚黎函数在复平面中是否是亚黎子函数
    assert lowergamma(S(1)/3, 1/x)._eval_is_meromorphic(x, 0) == False
    # 断言，验证 lowergamma(1, 1/x, evaluate=False) 的亚黎函数在复平面中是否是亚黎子函数
    assert lowergamma(1, 1/x, evaluate=False)._eval_is_meromorphic(x, 0) == False
    # 断言，验证 lowergamma(x, 1/x) 的亚黎函数在复平面中是否是亚黎子函数
    assert lowergamma(x, 1/x)._eval_is_meromorphic(x, 0) == False
    # 断言，验证 lowergamma(x + 1, 1/x) 的亚黎函数在复平面中是否是亚黎子函数
    assert lowergamma(x + 1, 1/x)._eval_is_meromorphic(x, 0) == False
    # 断言，验证 lowergamma(1/x, 1/x) 的亚黎函数在复平面中是否是亚黎子函数
    assert lowergamma(1/x, 1/x)._eval_is_meromorphic(x, 0) == False
    
    # 断言，验证 lowergamma(x, 2) 的级数展开是否正确
    assert lowergamma(x, 2).series(x, oo, 3) == \
        2**x*(1 + 2/(x + 1))*exp(-2)/x + O(exp(x*log(2))/x**3, (x, oo))
    
    # 断言，验证 lowergamma(x, y) 通过 expint 重写是否正确
    assert lowergamma(x, y).rewrite(expint) == -y**x*expint(-x + 1, y) + gamma(x)
    # 定义整数符号 k
    k = Symbol('k', integer=True)
    # 断言，验证 lowergamma(k, y) 通过 expint 重写是否正确
    assert lowergamma(k, y).rewrite(expint) == -y**k*expint(-k + 1, y) + gamma(k)
    # 重新定义整数符号 k，要求 k 为非正整数
    k = Symbol('k', integer=True, positive=False)
    # 断言，验证 lowergamma(k, y) 在不可重写情况下是否等于自身
    assert lowergamma(k, y).rewrite(expint) == lowergamma(k, y)
    # 断言，验证 lowergamma(x, y) 通过 uppergamma 重写是否正确
    assert lowergamma(x, y).rewrite(uppergamma) == gamma(x) - uppergamma(x, y)
    
    # 断言，验证 lowergamma(70, 6) 的值是否正确
    assert lowergamma(70, 6) == factorial(69) - 69035724522603011058660187038367026272747334489677105069435923032634389419656200387949342530805432320 * exp(-6)
    # 断言，验证 lowergamma(S(77) / 2, 6) 的数值和其不进行求值的情况下是否在一定精度内相等
    assert (lowergamma(S(77) / 2, 6) - lowergamma(S(77) / 2, 6, evaluate=False)).evalf() < 1e-16
    # 断言，验证 lowergamma(-S(77) / 2, 6) 的数值和其不进行求值的情况下是否在一定精度内相等
    assert (lowergamma(-S(77) / 2, 6) - lowergamma(-S(77) / 2, 6, evaluate=False)).evalf() < 1e-16
def test_uppergamma():
    # 导入所需函数和模块
    from sympy.functions.special.error_functions import expint
    from sympy.functions.special.hyper import meijerg
    # 断言：计算特定参数下的 uppergamma 函数值是否等于预期值
    assert uppergamma(4, 0) == 6
    # 断言：计算 uppergamma 函数对第二个参数 y 的偏导数是否符合预期表达式
    assert uppergamma(x, y).diff(y) == -y**(x - 1)*exp(-y)
    # 断言：测试 uppergamma 函数对第一个参数和随机复数的应用
    assert td(uppergamma(randcplx(), y), y)
    # 断言：计算 uppergamma 函数对第一个参数 x 的偏导数是否符合预期表达式
    assert uppergamma(x, y).diff(x) == \
        uppergamma(x, y)*log(y) + meijerg([], [1, 1], [0, 0, x], [], y)
    # 断言：测试 uppergamma 函数对第一个参数和随机复数的应用
    assert td(uppergamma(x, randcplx()), x)

    # 创建一个正数符号变量 p
    p = Symbol('p', positive=True)
    # 断言：计算 uppergamma 函数在特定参数下的值是否等于特殊函数 Ei 的负数值
    assert uppergamma(0, p) == -Ei(-p)
    # 断言：计算 uppergamma 函数在特定参数下的值是否等于 gamma 函数的值
    assert uppergamma(p, 0) == gamma(p)
    # 断言：计算特定参数下的 uppergamma 函数值是否等于特定表达式
    assert uppergamma(S.Half, x) == sqrt(pi)*erfc(sqrt(x))
    # 断言：检查 uppergamma 函数是否不含有 uppergamma 表达式
    assert not uppergamma(S.Half - 3, x).has(uppergamma)
    assert not uppergamma(S.Half + 3, x).has(uppergamma)
    # 断言：检查 uppergamma 函数在 evaluate=False 时是否含有 uppergamma 表达式
    assert uppergamma(S.Half, x, evaluate=False).has(uppergamma)
    # 断言：检查 uppergamma 函数在 evaluate=False 时两个表达式是否相等
    assert tn(uppergamma(S.Half + 3, x, evaluate=False),
              uppergamma(S.Half + 3, x), x)
    assert tn(uppergamma(S.Half - 3, x, evaluate=False),
              uppergamma(S.Half - 3, x), x)

    # 断言：检查 uppergamma 函数在给定参数下是否不变
    assert unchanged(uppergamma, x, -oo)
    assert unchanged(uppergamma, x, 0)

    # 断言：检查 uppergamma 函数在给定参数下是否符合分支测试
    assert tn_branch(-3, uppergamma)
    assert tn_branch(-4, uppergamma)
    assert tn_branch(Rational(1, 3), uppergamma)
    assert tn_branch(pi, uppergamma)
    # 断言：计算 uppergamma 函数在给定复数参数下的值是否相等
    assert uppergamma(3, exp_polar(4*pi*I)*x) == uppergamma(3, x)
    # 断言：计算 uppergamma 函数在给定复数参数下的值是否符合特定表达式
    assert uppergamma(y, exp_polar(5*pi*I)*x) == \
        exp(4*I*pi*y)*uppergamma(y, x*exp_polar(pi*I)) + \
        gamma(y)*(1 - exp(4*pi*I*y))
    # 断言：计算 uppergamma 函数在给定复数参数下的值是否符合特定表达式
    assert uppergamma(-2, exp_polar(5*pi*I)*x) == \
        uppergamma(-2, x*exp_polar(I*pi)) - 2*pi*I

    # 断言：计算 uppergamma 函数在给定参数下的值是否等于 expint 函数的表达式
    assert uppergamma(-2, x) == expint(3, x)/x**2

    # 断言：检查 uppergamma 函数的共轭是否符合预期表达式
    assert conjugate(uppergamma(x, y)) == uppergamma(conjugate(x), conjugate(y))
    # 断言：检查 uppergamma 函数在特定参数下是否不变
    assert unchanged(conjugate, uppergamma(x, -oo))

    # 断言：计算 uppergamma 函数在给定参数下的重写表达式是否符合 expint 函数的表达式
    assert uppergamma(x, y).rewrite(expint) == y**x*expint(-x + 1, y)
    # 断言：计算 uppergamma 函数在给定参数下的重写表达式是否符合 lowergamma 函数的表达式
    assert uppergamma(x, y).rewrite(lowergamma) == gamma(x) - lowergamma(x, y)

    # 断言：计算 uppergamma 函数在给定参数下的值是否等于特定的数值
    assert uppergamma(70, 6) == 69035724522603011058660187038367026272747334489677105069435923032634389419656200387949342530805432320*exp(-6)
    # 断言：计算 uppergamma 函数在给定参数下的数值近似误差是否小于 1e-16
    assert (uppergamma(S(77) / 2, 6) - uppergamma(S(77) / 2, 6, evaluate=False)).evalf() < 1e-16
    assert (uppergamma(-S(77) / 2, 6) - uppergamma(-S(77) / 2, 6, evaluate=False)).evalf() < 1e-16


def test_polygamma():
    # 断言：检查 polygamma 函数在给定参数下的返回值是否为 nan
    assert polygamma(n, nan) is nan

    # 断言：检查 polygamma 函数在给定无穷大参数下的返回值
    assert polygamma(0, oo) is oo
    assert polygamma(0, -oo) is oo
    assert polygamma(0, I*oo) is oo
    assert polygamma(0, -I*oo) is oo
    assert polygamma(1, oo) == 0
    assert polygamma(5, oo) == 0

    # 断言：检查 polygamma 函数在给定参数下的返回值
    assert polygamma(0, -9) is zoo
    assert polygamma(0, -1) is zoo
    assert polygamma(Rational(3, 2), -1) is zoo

    assert polygamma(0, 0) is zoo

    assert polygamma(0, 1) == -EulerGamma
    assert polygamma(0, 7) == Rational(49, 20) - EulerGamma

    assert polygamma(1, 1) == pi**2/6
    assert polygamma(1, 2) == pi**2/6 - 1
    assert polygamma(1, 3) == pi**2/6 - Rational(5, 4)
    assert polygamma(3, 1) == pi**4 / 15
    # 断言：计算多阶波利γ函数的值并进行断言
    assert polygamma(3, 5) == 6*(Rational(-22369, 20736) + pi**4/90)
    # 断言：计算多阶波利γ函数的值并进行断言
    assert polygamma(5, 1) == 8 * pi**6 / 63

    # 断言：计算多阶波利γ函数的值并进行断言，其中阶数为1/2
    assert polygamma(1, S.Half) == pi**2 / 2
    # 断言：计算多阶波利γ函数的值并进行断言，其中阶数为1/2
    assert polygamma(2, S.Half) == -14*zeta(3)
    # 断言：计算多阶波利γ函数的值并进行断言，其中阶数为11和1/2
    assert polygamma(11, S.Half) == 176896*pi**12

    # 定义函数 t，计算多阶波利γ函数的值并进行比较
    def t(m, n):
        # 将 m/n 转换为 SymPy 的有理数对象
        x = S(m)/n
        # 计算波利γ函数的值
        r = polygamma(0, x)
        # 如果结果中包含波利γ函数，则返回 False
        if r.has(polygamma):
            return False
        # 返回计算结果与数值结果的绝对误差是否小于 1e-10
        return abs(polygamma(0, x.n()).n() - r.n()).n() < 1e-10
    # 断言：验证函数 t 的几个计算结果
    assert t(1, 2)
    assert t(3, 2)
    assert t(-1, 2)
    assert t(1, 4)
    assert t(-3, 4)
    assert t(1, 3)
    assert t(4, 3)
    assert t(3, 4)
    assert t(2, 3)
    assert t(123, 5)

    # 断言：利用重写规则将多阶波利γ函数转换为黎曼 zeta 函数
    assert polygamma(0, x).rewrite(zeta) == polygamma(0, x)
    assert polygamma(1, x).rewrite(zeta) == zeta(2, x)
    assert polygamma(2, x).rewrite(zeta) == -2*zeta(3, x)
    assert polygamma(I, 2).rewrite(zeta) == polygamma(I, 2)
    # 定义符号变量 n1 到 n5
    n1 = Symbol('n1')
    n2 = Symbol('n2', real=True)
    n3 = Symbol('n3', integer=True)
    n4 = Symbol('n4', positive=True)
    n5 = Symbol('n5', positive=True, integer=True)
    # 断言：利用重写规则将多阶波利γ函数转换为黎曼 zeta 函数
    assert polygamma(n1, x).rewrite(zeta) == polygamma(n1, x)
    assert polygamma(n2, x).rewrite(zeta) == polygamma(n2, x)
    assert polygamma(n3, x).rewrite(zeta) == polygamma(n3, x)
    assert polygamma(n4, x).rewrite(zeta) == polygamma(n4, x)
    assert polygamma(n5, x).rewrite(zeta) == (-1)**(n5 + 1) * factorial(n5) * zeta(n5 + 1, x)

    # 断言：验证多阶波利γ函数的导数
    assert polygamma(3, 7*x).diff(x) == 7*polygamma(4, 7*x)

    # 断言：利用重写规则将多阶波利γ函数转换为调和级数函数
    assert polygamma(0, x).rewrite(harmonic) == harmonic(x - 1) - EulerGamma
    assert polygamma(2, x).rewrite(harmonic) == 2*harmonic(x - 1, 3) - 2*zeta(3)
    # 定义整数符号变量 ni
    ni = Symbol("n", integer=True)
    # 断言：利用重写规则将多阶波利γ函数转换为调和级数函数
    assert polygamma(ni, x).rewrite(harmonic) == (-1)**(ni + 1)*(-harmonic(x - 1, ni + 1)
                                                                 + zeta(ni + 1))*factorial(ni)

    # 断言：验证多阶波利γ函数的周期性质
    k = Symbol('n', integer=True, nonnegative=True)
    assert polygamma(k, exp_polar(2*I*pi)*x) == polygamma(k, x)

    # 断言：验证多阶波利γ函数的周期性质（负整数阶）
    k = Symbol('n', integer=True)
    assert polygamma(k, exp_polar(2*I*pi)*x).args == (k, exp_polar(2*I*pi)*x)

    # 断言：验证多阶波利γ函数的特殊情况（阶数为 -1）
    assert polygamma(-1, x) == loggamma(x) - log(2*pi) / 2

    # 断言：验证多阶波利γ函数的特殊情况（阶数小于 -1）
    assert polygamma(-2, x).func is polygamma

    # 断言：验证多阶波利γ函数的特殊情况（参数为负数）
    assert polygamma(0, -x).expand(func=True) == polygamma(0, -x)

    # 断言：验证多阶波利γ函数的正负性质
    assert polygamma(2, 2.5).is_positive == False
    assert polygamma(2, -2.5).is_positive == False
    assert polygamma(3, 2.5).is_positive == True
    assert polygamma(3, -2.5).is_positive is True
    assert polygamma(-2, -2.5).is_positive is None
    assert polygamma(-3, -2.5).is_positive is None

    # 断言：验证多阶波利γ函数的正负性质
    assert polygamma(2, 2.5).is_negative == True
    assert polygamma(3, 2.5).is_negative == False
    assert polygamma(3, -2.5).is_negative == False
    assert polygamma(2, -2.5).is_negative is True
    # 断言polygamma(-2, -2.5)的符号为None
    assert polygamma(-2, -2.5).is_negative is None
    # 断言polygamma(-3, -2.5)的符号为None
    assert polygamma(-3, -2.5).is_negative is None

    # 断言polygamma(I, 2)的符号为None
    assert polygamma(I, 2).is_positive is None
    # 断言polygamma(I, 3)的符号为None
    assert polygamma(I, 3).is_negative is None

    # issue 17350
    # 断言I乘以polygamma(I, pi)的实部和虚部相同
    assert (I*polygamma(I, pi)).as_real_imag() == \
           (-im(polygamma(I, pi)), re(polygamma(I, pi)))
    # 断言tanh(polygamma(I, 1))重写为exp的形式
    assert (tanh(polygamma(I, 1))).rewrite(exp) == \
           (exp(polygamma(I, 1)) - exp(-polygamma(I, 1)))/(exp(polygamma(I, 1)) + exp(-polygamma(I, 1)))
    # 断言I除以polygamma(I, 4)重写为exp的形式
    assert (I / polygamma(I, 4)).rewrite(exp) == \
           I*exp(-I*atan(im(polygamma(I, 4))/re(polygamma(I, 4))))/Abs(polygamma(I, 4))

    # issue 12569
    # 断言im函数应用于polygamma(0, I)不变
    assert unchanged(im, polygamma(0, I))
    # 断言polygamma(Symbol('a', positive=True), Symbol('b', positive=True))为实数
    assert polygamma(Symbol('a', positive=True), Symbol('b', positive=True)).is_real is True
    # 断言polygamma(0, I)为实数
    assert polygamma(0, I).is_real is None

    # 断言polygamma(pi, 3)的数值求解结果为"0.1169314564"
    assert str(polygamma(pi, 3).evalf(n=10)) == "0.1169314564"
    # 断言polygamma(2.3, 1.0)的数值求解结果为"-3.003302909"
    assert str(polygamma(2.3, 1.0).evalf(n=10)) == "-3.003302909"
    # 断言polygamma(-1, 1)的数值求解结果为"-0.9189385332"，不为零
    assert str(polygamma(-1, 1).evalf(n=10)) == "-0.9189385332" # not zero
    # 断言polygamma(I, 1)的数值求解结果为"-3.109856569 + 1.89089016*I"
    assert str(polygamma(I, 1).evalf(n=10)) == "-3.109856569 + 1.89089016*I"
    # 断言polygamma(1, I)的数值求解结果为"-0.5369999034 - 0.7942335428*I"
    assert str(polygamma(1, I).evalf(n=10)) == "-0.5369999034 - 0.7942335428*I"
    # 断言polygamma(I, I)的数值求解结果为"6.332362889 + 45.92828268*I"
    assert str(polygamma(I, I).evalf(n=10)) == "6.332362889 + 45.92828268*I"
# 定义一个函数用于测试 polygamma 函数的展开
def test_polygamma_expand_func():
    # 断言 polygamma 函数在参数为 0 和 x 时展开的结果等于 polygamma(0, x)
    assert polygamma(0, x).expand(func=True) == polygamma(0, x)
    # 断言 polygamma 函数在参数为 0 和 2*x 时展开的结果等于以下表达式
    assert polygamma(0, 2*x).expand(func=True) == \
        polygamma(0, x)/2 + polygamma(0, S.Half + x)/2 + log(2)
    # 断言 polygamma 函数在参数为 1 和 2*x 时展开的结果等于以下表达式
    assert polygamma(1, 2*x).expand(func=True) == \
        polygamma(1, x)/4 + polygamma(1, S.Half + x)/4
    # 断言 polygamma 函数在参数为 2 和 x 时展开的结果等于 polygamma(2, x)
    assert polygamma(2, x).expand(func=True) == \
        polygamma(2, x)
    # 断言 polygamma 函数在参数为 0 和 -1 + x 时展开的结果等于以下表达式
    assert polygamma(0, -1 + x).expand(func=True) == \
        polygamma(0, x) - 1/(x - 1)
    # 断言 polygamma 函数在参数为 0 和 1 + x 时展开的结果等于以下表达式
    assert polygamma(0, 1 + x).expand(func=True) == \
        1/x + polygamma(0, x )
    # 断言 polygamma 函数在参数为 0 和 2 + x 时展开的结果等于以下表达式
    assert polygamma(0, 2 + x).expand(func=True) == \
        1/x + 1/(1 + x) + polygamma(0, x)
    # 断言 polygamma 函数在参数为 0 和 3 + x 时展开的结果等于以下表达式
    assert polygamma(0, 3 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x)
    # 断言 polygamma 函数在参数为 0 和 4 + x 时展开的结果等于以下表达式
    assert polygamma(0, 4 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x) + 1/(3 + x)
    # 断言 polygamma 函数在参数为 1 和 1 + x 时展开的结果等于以下表达式
    assert polygamma(1, 1 + x).expand(func=True) == \
        polygamma(1, x) - 1/x**2
    # 断言 polygamma 函数在参数为 1 和 2 + x 时展开的结果等于以下表达式，且不使用 multinomial 展开
    assert polygamma(1, 2 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2
    # 断言 polygamma 函数在参数为 1 和 3 + x 时展开的结果等于以下表达式，且不使用 multinomial 展开
    assert polygamma(1, 3 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - 1/(2 + x)**2
    # 断言 polygamma 函数在参数为 1 和 4 + x 时展开的结果等于以下表达式，且不使用 multinomial 展开
    assert polygamma(1, 4 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - \
        1/(2 + x)**2 - 1/(3 + x)**2
    # 断言 polygamma 函数在参数为 0 和 x + y 时展开的结果等于 polygamma(0, x + y)
    assert polygamma(0, x + y).expand(func=True) == \
        polygamma(0, x + y)
    # 断言 polygamma 函数在参数为 1 和 x + y 时展开的结果等于 polygamma(1, x + y)
    assert polygamma(1, x + y).expand(func=True) == \
        polygamma(1, x + y)
    # 断言 polygamma 函数在参数为 1 和 3 + 4*x + y 时展开的结果等于以下表达式，且不使用 multinomial 展开
    assert polygamma(1, 3 + 4*x + y).expand(func=True, multinomial=False) == \
        polygamma(1, y + 4*x) - 1/(y + 4*x)**2 - \
        1/(1 + y + 4*x)**2 - 1/(2 + y + 4*x)**2
    # 断言 polygamma 函数在参数为 3 和 3 + 4*x + y 时展开的结果等于以下表达式，且不使用 multinomial 展开
    assert polygamma(3, 3 + 4*x + y).expand(func=True, multinomial=False) == \
        polygamma(3, y + 4*x) - 6/(y + 4*x)**4 - \
        6/(1 + y + 4*x)**4 - 6/(2 + y + 4*x)**4
    # 断言 polygamma 函数在参数为 3 和 4*x + y + 1 时展开的结果等于以下表达式，且不使用 multinomial 展开
    assert polygamma(3, 4*x + y + 1).expand(func=True, multinomial=False) == \
        polygamma(3, y + 4*x) - 6/(y + 4*x)**4
    # 定义一个变量 e，用于保存 polygamma(3, 4*x + y + Rational(3, 2)) 的结果，并断言展开后等于 e 自身
    e = polygamma(3, 4*x + y + Rational(3, 2))
    assert e.expand(func=True) == e
    # 定义一个变量 e，用于保存 polygamma(3, x + y + Rational(3, 4)) 的结果，并断言展开后等于 e 自身，但不使用基本展开
    e = polygamma(3, x + y + Rational(3, 4))
    assert e.expand(func=True, basic=False) == e

    # 断言 polygamma 函数在参数为 -1 和 x 时不进行评估，展开的结果等于以下表达式
    assert polygamma(-1, x, evaluate=False).expand(func=True) == \
        loggamma(x) - log(pi)/2 - log(2)/2
    # 定义变量 p2，用于保存 polygamma(-2, x) 展开后的结果，并断言其类型为 Subs
    p2 = polygamma(-2, x).expand(func=True) + x**2/2 - x/2 + S(1)/12
    assert isinstance(p2, Subs)
    # 断言 p2 的点为 (-1,)
    assert p2.point == (-1,)


# 定义一个函数用于测试 digamma 函数
def test_digamma():
    # 断言 digamma 函数在参数为 nan 时的结果为 nan
    assert digamma(nan) == nan

    # 断言 digamma 函数在参数为 oo 时的结果为 oo
    assert digamma(oo) == oo
    # 断言 digamma 函数在参数为 -oo 时的结果为 oo
    assert digamma(-oo) == oo
    # 断言 digamma 函数在参数为 I*oo 时的结果为 oo
    assert digamma(I*oo) == oo
    # 断言 digamma 函数在参数为 -I*oo 时的结果为 oo
    assert digamma(-I*oo) == oo

    # 断言 digamma 函数在参数为 -9 时的结果为 zoo
    assert digamma(-9) == zoo

    # 断言 digamma 函数在参数为 -9 时的结果为 zoo
    assert digamma(-9) == zoo
    # 断言 digamma 函数在
    #`
    # 定义函数 t，接受两个参数 m 和 n
    def t(m, n):
        # 计算 S(m) 除以 n 的值
        x = S(m)/n
        # 计算 digamma 函数在 x 处的值
        r = digamma(x)
        # 检查 r 是否包含 digamma 函数，若包含则返回 False
        if r.has(digamma):
            return False
        # 返回 True，若 digamma(x.n()).n() 与 r.n() 的差小于 1e-10
        return abs(digamma(x.n()).n() - r.n()).n() < 1e-10

    # 测试函数 t，传入参数 1 和 2，断言其结果为 True
    assert t(1, 2)
    # 测试函数 t，传入参数 3 和 2，断言其结果为 True
    assert t(3, 2)
    # 测试函数 t，传入参数 -1 和 2，断言其结果为 True
    assert t(-1, 2)
    # 测试函数 t，传入参数 1 和 4，断言其结果为 True
    assert t(1, 4)
    # 测试函数 t，传入参数 -3 和 4，断言其结果为 True
    assert t(-3, 4)
    # 测试函数 t，传入参数 1 和 3，断言其结果为 True
    assert t(1, 3)
    # 测试函数 t，传入参数 4 和 3，断言其结果为 True
    assert t(4, 3)
    # 测试函数 t，传入参数 3 和 4，断言其结果为 True
    assert t(3, 4)
    # 测试函数 t，传入参数 2 和 3，断言其结果为 True
    assert t(2, 3)
    # 测试函数 t，传入参数 123 和 5，断言其结果为 True
    assert t(123, 5)

    # 验证 digamma(x) 可以重写为 zeta 函数的形式
    assert digamma(x).rewrite(zeta) == polygamma(0, x)

    # 验证 digamma(x) 可以重写为 harmonic 函数的形式
    assert digamma(x).rewrite(harmonic) == harmonic(x - 1) - EulerGamma

    # 检查 digamma 函数在复数单位 i 处是否是实数，结果为 None
    assert digamma(I).is_real is None

    # 检验 digamma(x, evaluate=False) 的一阶导数等于 polygamma(1, x)
    assert digamma(x,evaluate=False).fdiff() == polygamma(1, x)

    # 检查 digamma(x, evaluate=False) 是否是实数，结果为 None
    assert digamma(x,evaluate=False).is_real is None

    # 检查 digamma(x, evaluate=False) 是否是正数，结果为 None
    assert digamma(x,evaluate=False).is_positive is None

    # 检查 digamma(x, evaluate=False) 是否是负数，结果为 None
    assert digamma(x,evaluate=False).is_negative is None

    # 验证 digamma(x) 可以重写为 polygamma 函数的形式
    assert digamma(x,evaluate=False).rewrite(polygamma) == polygamma(0, x)
def test_digamma_expand_func():
    # 断言：digamma(x) 的展开结果等于 polygamma(0, x)
    assert digamma(x).expand(func=True) == polygamma(0, x)
    # 断言：digamma(2*x) 的展开结果
    assert digamma(2*x).expand(func=True) == \
        polygamma(0, x)/2 + polygamma(0, Rational(1, 2) + x)/2 + log(2)
    # 断言：digamma(-1 + x) 的展开结果
    assert digamma(-1 + x).expand(func=True) == \
        polygamma(0, x) - 1/(x - 1)
    # 断言：digamma(1 + x) 的展开结果
    assert digamma(1 + x).expand(func=True) == \
        1/x + polygamma(0, x)
    # 断言：digamma(2 + x) 的展开结果
    assert digamma(2 + x).expand(func=True) == \
        1/x + 1/(1 + x) + polygamma(0, x)
    # 断言：digamma(3 + x) 的展开结果
    assert digamma(3 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x)
    # 断言：digamma(4 + x) 的展开结果
    assert digamma(4 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x) + 1/(3 + x)
    # 断言：digamma(x + y) 的展开结果
    assert digamma(x + y).expand(func=True) == \
        polygamma(0, x + y)

def test_trigamma():
    # 断言：trigamma(nan) 的结果为 nan
    assert trigamma(nan) == nan

    # 断言：trigamma(oo) 的结果为 0
    assert trigamma(oo) == 0

    # 断言：trigamma(1) 的结果为 pi**2/6
    assert trigamma(1) == pi**2/6
    # 断言：trigamma(2) 的结果为 pi**2/6 - 1
    assert trigamma(2) == pi**2/6 - 1
    # 断言：trigamma(3) 的结果为 pi**2/6 - 5/4
    assert trigamma(3) == pi**2/6 - Rational(5, 4)

    # 断言：trigamma(x, evaluate=False) 重写为 zeta(2, x)
    assert trigamma(x, evaluate=False).rewrite(zeta) == zeta(2, x)
    # 断言：trigamma(x, evaluate=False) 重写为 harmonic(x).rewrite(polygamma).rewrite(harmonic)
    assert trigamma(x, evaluate=False).rewrite(harmonic) == \
        trigamma(x).rewrite(polygamma).rewrite(harmonic)

    # 断言：trigamma(x, evaluate=False) 的一阶导数为 polygamma(2, x)
    assert trigamma(x,evaluate=False).fdiff() == polygamma(2, x)

    # 断言：trigamma(x, evaluate=False) 的实部为 None
    assert trigamma(x,evaluate=False).is_real is None

    # 断言：trigamma(x, evaluate=False) 的正数性为 None
    assert trigamma(x,evaluate=False).is_positive is None

    # 断言：trigamma(x, evaluate=False) 的负数性为 None
    assert trigamma(x,evaluate=False).is_negative is None

    # 断言：trigamma(x, evaluate=False) 重写为 polygamma(1, x)
    assert trigamma(x,evaluate=False).rewrite(polygamma) == polygamma(1, x)

def test_trigamma_expand_func():
    # 断言：trigamma(2*x) 的展开结果
    assert trigamma(2*x).expand(func=True) == \
        polygamma(1, x)/4 + polygamma(1, Rational(1, 2) + x)/4
    # 断言：trigamma(1 + x) 的展开结果
    assert trigamma(1 + x).expand(func=True) == \
        polygamma(1, x) - 1/x**2
    # 断言：trigamma(2 + x) 的展开结果，不使用 multinomial 展开
    assert trigamma(2 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2
    # 断言：trigamma(3 + x) 的展开结果，不使用 multinomial 展开
    assert trigamma(3 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - 1/(2 + x)**2
    # 断言：trigamma(4 + x) 的展开结果，不使用 multinomial 展开
    assert trigamma(4 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - \
        1/(2 + x)**2 - 1/(3 + x)**2
    # 断言：trigamma(x + y) 的展开结果
    assert trigamma(x + y).expand(func=True) == \
        polygamma(1, x + y)
    # 断言：trigamma(3 + 4*x + y) 的展开结果，不使用 multinomial 展开
    assert trigamma(3 + 4*x + y).expand(func=True, multinomial=False) == \
        polygamma(1, y + 4*x) - 1/(y + 4*x)**2 - \
        1/(1 + y + 4*x)**2 - 1/(2 + y + 4*x)**2

def test_loggamma():
    # 断言：调用 loggamma(2, 3) 会引发 TypeError 异常
    raises(TypeError, lambda: loggamma(2, 3))
    # 断言：调用 loggamma(x).fdiff(2) 会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: loggamma(x).fdiff(2))

    # 断言：loggamma(-1) 的结果为 oo
    assert loggamma(-1) is oo
    # 断言：loggamma(-2) 的结果为 oo
    assert loggamma(-2) is oo
    # 断言：loggamma(0) 的结果为 oo
    assert loggamma(0) is oo
    # 断言：loggamma(1) 的结果为 0
    assert loggamma(1) == 0
    # 断言：loggamma(2) 的结果为 0
    assert loggamma(2) == 0
    # 断言：loggamma(3) 的结果为 log(2)
    assert loggamma(3) == log(2)
    # 断言：loggamma(4) 的结果为 log(6)
    assert loggamma(4) == log(6)

    # 定义整数 n 符号变量
    n = Symbol("n", integer=True, positive=True)
    # 断言：loggamma(n) 等于 log(gamma(n))
    assert loggamma(n) == log(gamma(n))
    # 断言：loggamma(-n) 的结果为 oo
    assert loggamma(-n) is oo
    # 断言：loggamma(n/2) 的结果
    assert loggamma(n/2) == log(2**(-n + 1)*sqrt(pi)*gamma(n)/gamma(n/2 + S.Half))

    # 断言：loggamma(oo) 的结果为 oo
    assert loggamma(oo) is oo
    # 断言：loggamma(-oo) 的结果为 zoo
    assert loggamma(-oo) is zoo
    # 断言：loggamma(I*oo) 的结果为 zoo
    assert loggamma(I*oo) is zoo
    # 断言loggamma(-I*oo)返回zoo
    assert loggamma(-I*oo) is zoo
    # 断言loggamma(zoo)返回zoo
    assert loggamma(zoo) is zoo
    # 断言loggamma(nan)返回nan
    assert loggamma(nan) is nan

    # 计算Rational(16, 3)的对数Gamma函数的自然对数
    L = loggamma(Rational(16, 3))
    # 计算期望值E，展开并计算-log(3) * 5 + loggamma(Rational(1, 3)) + log(4) + log(7) + log(10) + log(13)
    E = -5*log(3) + loggamma(Rational(1, 3)) + log(4) + log(7) + log(10) + log(13)
    # 断言展开函数后的L与期望值E相等
    assert expand_func(L).doit() == E
    # 断言L的数值近似等于E的数值近似
    assert L.n() == E.n()

    # 计算Rational(19, 4)的对数Gamma函数的自然对数
    L = loggamma(Rational(19, 4))
    # 计算期望值E，展开并计算-log(4) * 4 + loggamma(Rational(3, 4)) + log(3) + log(7) + log(11) + log(15)
    E = -4*log(4) + loggamma(Rational(3, 4)) + log(3) + log(7) + log(11) + log(15)
    # 断言展开函数后的L与期望值E相等
    assert expand_func(L).doit() == E
    # 断言L的数值近似等于E的数值近似
    assert L.n() == E.n()

    # 计算Rational(23, 7)的对数Gamma函数的自然对数
    L = loggamma(Rational(23, 7))
    # 计算期望值E，展开并计算-log(7) * 3 + log(2) + loggamma(Rational(2, 7)) + log(9) + log(16)
    E = -3*log(7) + log(2) + loggamma(Rational(2, 7)) + log(9) + log(16)
    # 断言展开函数后的L与期望值E相等
    assert expand_func(L).doit() == E
    # 断言L的数值近似等于E的数值近似
    assert L.n() == E.n()

    # 计算Rational(19, 4) - 7的对数Gamma函数的自然对数
    L = loggamma(Rational(19, 4) - 7)
    # 计算期望值E，展开并计算-log(9) - log(5) + loggamma(Rational(3, 4)) + 3*log(4) - 3*I*pi
    E = -log(9) - log(5) + loggamma(Rational(3, 4)) + 3*log(4) - 3*I*pi
    # 断言展开函数后的L与期望值E相等
    assert expand_func(L).doit() == E
    # 断言L的数值近似等于E的数值近似
    assert L.n() == E.n()

    # 计算Rational(23, 7) - 6的对数Gamma函数的自然对数
    L = loggamma(Rational(23, 7) - 6)
    # 计算期望值E，展开并计算-log(19) - log(12) - log(5) + loggamma(Rational(2, 7)) + 3*log(7) - 3*I*pi
    E = -log(19) - log(12) - log(5) + loggamma(Rational(2, 7)) + 3*log(7) - 3*I*pi
    # 断言展开函数后的L与期望值E相等
    assert expand_func(L).doit() == E
    # 断言L的数值近似等于E的数值近似
    assert L.n() == E.n()

    # 断言loggamma(x)对x的导数等于polygamma(0, x)
    assert loggamma(x).diff(x) == polygamma(0, x)
    # 计算1/(x + sin(x)) + cos(x)的loggamma函数的x的级数展开，展开到第4阶
    s1 = loggamma(1/(x + sin(x)) + cos(x)).nseries(x, n=4)
    # 期望的级数展开结果s2
    s2 = (-log(2*x) - 1)/(2*x) - log(x/pi)/2 + (4 - log(2*x))*x/24 + O(x**2) + \
        log(x)*x**2/2
    # 断言展开函数后的s1与期望值s2相等
    assert (s1 - s2).expand(force=True).removeO() == 0
    # 计算1/x的loggamma函数的x的级数展开
    s1 = loggamma(1/x).series(x)
    # 期望的级数展开结果s2
    s2 = (1/x - S.Half)*log(1/x) - 1/x + log(2*pi)/2 + \
        x/12 - x**3/360 + x**5/1260 + O(x**7)
    # 断言展开函数后的s1与期望值s2相等
    assert ((s1 - s2).expand(force=True)).removeO() == 0

    # 断言loggamma(x)按'intractable'方式重写等于log(gamma(x))
    assert loggamma(x).rewrite('intractable') == log(gamma(x))

    # 计算loggamma(x)的级数展开并化简
    s1 = loggamma(x).series(x).cancel()
    # 期望的级数展开结果，包含-log(x) - EulerGamma*x + pi**2*x**2/12 + x**3*polygamma(2, 1)/6 + \
    #     pi**4*x**4/360 + x**5*polygamma(4, 1)/120 + O(x**6)
    assert s1 == -log(x) - EulerGamma*x + pi**2*x**2/12 + x**3*polygamma(2, 1)/6 + \
        pi**4*x**4/360 + x**5*polygamma(4, 1)/120 + O(x**6)
    # 断言展开函数后的s1与按'intractable'方式重写的loggamma(x)的级数展开结果相等
    assert s1 == loggamma(x).rewrite('intractable').series(x).cancel()

    # 断言loggamma(x)的共轭等于loggamma(conjugate(x))
    assert conjugate(loggamma(x)) == loggamma(conjugate(x))
    # 断言loggamma(0)的共轭为oo
    assert conjugate(loggamma(0)) is oo
    # 断言loggamma(1)的共轭等于loggamma(conjugate(1))
    assert conjugate(loggamma(1)) == loggamma(conjugate(1))
    # 断言loggamma(-oo)的共轭等于conjugate(zoo)
    assert conjugate(loggamma(-oo)) == conjugate(zoo)

    # 断言loggamma(Symbol('v', positive=True))的实部为True
    assert loggamma(Symbol('v', positive=True)).is_real is True
    # 断言loggamma(Symbol('v', zero=True))的实部为False
    assert loggamma(Symbol('v', zero=True)).is_real is False
    # 断言loggamma(Symbol('v', negative=True))的实部为False
    assert loggamma(Symbol('v', negative=True)).is_real is False
    # 断言loggamma(Symbol('v', nonpositive=True))的实部为False
    assert loggamma(Symbol('v', nonpositive=True)).is_real is False
    # 断言loggamma(Symbol('v', nonnegative=True))的实部为None
    assert loggamma(Symbol('v', nonnegative=True)).is_real is None
    # 断言loggamma(Symbol('v', imaginary=True))的实部为None
    assert loggamma(Symbol('v', imaginary=True)).is_real is None
    # 断言loggamma(Symbol('
def test_multigamma():
    # 导入 Product 类
    from sympy.concrete.products import Product
    # 定义符号 p
    p = Symbol('p')
    # 创建虚拟变量 _k
    _k = Dummy('_k')

    # 验证多重 Gamma 函数的定义
    assert multigamma(x, p).dummy_eq(pi**(p*(p - 1)/4)*\
        Product(gamma(x + (1 - _k)/2), (_k, 1, p)))

    # 验证共轭多重 Gamma 函数的定义
    assert conjugate(multigamma(x, p)).dummy_eq(pi**((conjugate(p) - 1)*\
        conjugate(p)/4)*Product(gamma(conjugate(x) + (1-conjugate(_k))/2), (_k, 1, p)))
    
    # 验证共轭多重 Gamma 函数与单一 Gamma 函数的关系
    assert conjugate(multigamma(x, 1)) == gamma(conjugate(x))

    # 对于正的 p，验证共轭多重 Gamma 函数的定义
    p = Symbol('p', positive=True)
    assert conjugate(multigamma(x, p)).dummy_eq(pi**((p - 1)*p/4)*\
        Product(gamma(conjugate(x) + (1-conjugate(_k))/2), (_k, 1, p)))

    # 验证特殊情况下的多重 Gamma 函数的值
    assert multigamma(nan, 1) is nan
    assert multigamma(oo, 1).doit() is oo

    # 验证多重 Gamma 函数的特定输入值
    assert multigamma(1, 1) == 1
    assert multigamma(2, 1) == 1
    assert multigamma(3, 1) == 2

    # 验证多重 Gamma 函数的更多输入值
    assert multigamma(102, 1) == factorial(101)
    assert multigamma(S.Half, 1) == sqrt(pi)

    # 验证二阶多重 Gamma 函数的值
    assert multigamma(1, 2) == pi
    assert multigamma(2, 2) == pi/2
    # 检查 multigamma 函数在给定参数下返回的值是否为无穷大
    assert multigamma(1, 3) is zoo
    # 检查 multigamma 函数在给定参数下返回的值是否等于 pi^2/2
    assert multigamma(2, 3) == pi**2/2
    # 检查 multigamma 函数在给定参数下返回的值是否等于 3*pi^2/2
    assert multigamma(3, 3) == 3*pi**2/2
    
    # 检查 multigamma 函数关于第一个参数 x 的一阶导数是否等于 gamma(x) * polygamma(0, x)
    assert multigamma(x, 1).diff(x) == gamma(x)*polygamma(0, x)
    # 检查 multigamma 函数关于第一个参数 x 的二阶导数是否等于特定表达式
    assert multigamma(x, 2).diff(x) == sqrt(pi)*gamma(x)*gamma(x - S.Half)*polygamma(0, x) + \
                                        sqrt(pi)*gamma(x)*gamma(x - S.Half)*polygamma(0, x - S.Half)
    
    # 检查 multigamma 函数关于第一个参数 x - 1 的一阶展开是否等于 gamma(x) / (x - 1)
    assert multigamma(x - 1, 1).expand(func=True) == gamma(x)/(x - 1)
    # 检查 multigamma 函数关于第一个参数 x + 2 的一阶展开是否等于 x*(x + 1)*gamma(x)
    assert multigamma(x + 2, 1).expand(func=True, mul=False) == x*(x + 1)*gamma(x)
    # 检查 multigamma 函数关于第一个参数 x - 1 的二阶展开是否等于特定表达式
    assert multigamma(x - 1, 2).expand(func=True) == sqrt(pi)*gamma(x)*gamma(x + S.Half)/\
                                                      (x**3 - 3*x**2 + x*Rational(11, 4) - Rational(3, 4))
    # 检查 multigamma 函数关于第一个参数 x - 1 的三阶展开是否等于特定表达式
    assert multigamma(x - 1, 3).expand(func=True) == pi**Rational(3, 2)*gamma(x)**2*\
                                                      gamma(x + S.Half)/(x**5 - 6*x**4 + 55*x**3/4 - 15*x**2 + \
                                                      x*Rational(31, 4) - Rational(3, 2))
    
    # 检查 multigamma 函数在给定参数 n 下，是否通过重写为阶乘来得到 factorial(n - 1)
    assert multigamma(n, 1).rewrite(factorial) == factorial(n - 1)
    # 检查 multigamma 函数在给定参数 n 下，是否通过重写为阶乘来得到特定表达式
    assert multigamma(n, 2).rewrite(factorial) == sqrt(pi)*factorial(n - Rational(3, 2))*factorial(n - 1)
    # 检查 multigamma 函数在给定参数 n 下，是否通过重写为阶乘来得到特定表达式
    assert multigamma(n, 3).rewrite(factorial) == pi**Rational(3, 2)*factorial(n - 2)*\
                                                   factorial(n - Rational(3, 2))*factorial(n - 1)
    
    # 检查 multigamma 函数在给定参数 Rational(-1, 2) 下，是否返回一个非实数
    assert multigamma(Rational(-1, 2), 3, evaluate=False).is_real == False
    # 检查 multigamma 函数在给定参数 S.Half 下，是否返回一个非实数
    assert multigamma(S.Half, 3, evaluate=False).is_real == False
    # 检查 multigamma 函数在给定参数 0 下，是否返回一个非实数
    assert multigamma(0, 1, evaluate=False).is_real == False
    # 检查 multigamma 函数在给定参数 1 下，是否返回一个非实数
    assert multigamma(1, 3, evaluate=False).is_real == False
    # 检查 multigamma 函数在给定参数 -1.0 下，是否返回一个非实数
    assert multigamma(-1.0, 3, evaluate=False).is_real == False
    # 检查 multigamma 函数在给定参数 0.7 下，是否返回一个实数
    assert multigamma(0.7, 3, evaluate=False).is_real == True
    # 检查 multigamma 函数在给定参数 3 下，是否返回一个实数
    assert multigamma(3, 3, evaluate=False).is_real == True
# 定义测试函数 `test_gamma_as_leading_term()`
def test_gamma_as_leading_term():
    # 断言：对 gamma 函数应用 as_leading_term 方法，期望结果为 1/x
    assert gamma(x).as_leading_term(x) == 1/x
    # 断言：对 gamma 函数应用 as_leading_term 方法，期望结果为 1
    assert gamma(2 + x).as_leading_term(x) == S(1)
    # 断言：对 gamma 函数应用 as_leading_term 方法，期望结果为 1
    assert gamma(cos(x)).as_leading_term(x) == S(1)
    # 断言：对 gamma 函数应用 as_leading_term 方法，期望结果为 1/x
    assert gamma(sin(x)).as_leading_term(x) == 1/x
```