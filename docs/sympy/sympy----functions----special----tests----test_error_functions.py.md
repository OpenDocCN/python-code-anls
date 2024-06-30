# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_error_functions.py`

```
from sympy.core.function import (diff, expand, expand_func)  # 导入函数 diff, expand, expand_func
from sympy.core import EulerGamma  # 导入常数 EulerGamma
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi)  # 导入常数 E, Float, I, Rational, nan, oo, pi
from sympy.core.singleton import S  # 导入单例对象 S
from sympy.core.symbol import (Symbol, symbols, Dummy)  # 导入符号类 Symbol, symbols, Dummy
from sympy.functions.elementary.complexes import (conjugate, im, polar_lift, re)  # 导入复数相关函数 conjugate, im, polar_lift, re
from sympy.functions.elementary.exponential import (exp, exp_polar, log)  # 导入指数函数 exp, exp_polar, log
from sympy.functions.elementary.hyperbolic import (cosh, sinh)  # 导入双曲函数 cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数 sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)  # 导入三角函数 cos, sin, sinc
from sympy.functions.special.error_functions import (Chi, Ci, E1, Ei, Li, Shi, Si, erf, erf2, erf2inv, erfc, erfcinv, erfi, erfinv, expint, fresnelc, fresnels, li)  # 导入误差函数 erf 相关
from sympy.functions.special.gamma_functions import (gamma, uppergamma)  # 导入 gamma 函数和上不完全 gamma 函数
from sympy.functions.special.hyper import (hyper, meijerg)  # 导入超函 hyper 和 Meijer G 函数
from sympy.integrals.integrals import (Integral, integrate)  # 导入积分相关函数 Integral, integrate
from sympy.series.gruntz import gruntz  # 导入 Gruntz 序列
from sympy.series.limits import limit  # 导入极限计算函数 limit
from sympy.series.order import O  # 导入 O 记号
from sympy.core.expr import unchanged  # 导入未更改的表达式
from sympy.core.function import ArgumentIndexError  # 导入参数索引错误类
from sympy.functions.special.error_functions import _erfs, _eis  # 导入特殊误差函数 _erfs, _eis
from sympy.testing.pytest import raises  # 导入 pytest 的 raises 函数

x, y, z = symbols('x,y,z')  # 创建符号变量 x, y, z
w = Symbol("w", real=True)  # 创建实数符号变量 w
n = Symbol("n", integer=True)  # 创建整数符号变量 n
t = Dummy('t')  # 创建虚拟符号变量 t

def test_erf():
    assert erf(nan) is nan  # 检查 erf 函数在 nan 上的值是否为 nan

    assert erf(oo) == 1  # 检查 erf 函数在无穷大 oo 上的值是否为 1
    assert erf(-oo) == -1  # 检查 erf 函数在负无穷大 -oo 上的值是否为 -1

    assert erf(0) is S.Zero  # 检查 erf 函数在 0 上的值是否为 S.Zero

    assert erf(I*oo) == oo*I  # 检查 erf 函数在虚无穷大 I*oo 上的值是否为 oo*I
    assert erf(-I*oo) == -oo*I  # 检查 erf 函数在负虚无穷大 -I*oo 上的值是否为 -oo*I

    assert erf(-2) == -erf(2)  # 检查 erf 函数在 -2 上的值是否等于 -erf(2)
    assert erf(-x*y) == -erf(x*y)  # 检查 erf 函数在 -x*y 上的值是否等于 -erf(x*y)
    assert erf(-x - y) == -erf(x + y)  # 检查 erf 函数在 -x - y 上的值是否等于 -erf(x + y)

    assert erf(erfinv(x)) == x  # 检查 erf 函数在 erfinv(x) 上的反函数值是否等于 x
    assert erf(erfcinv(x)) == 1 - x  # 检查 erf 函数在 erfcinv(x) 上的反函数值是否等于 1 - x
    assert erf(erf2inv(0, x)) == x  # 检查 erf 函数在 erf2inv(0, x) 上的值是否等于 x
    assert erf(erf2inv(0, x, evaluate=False)) == x  # 检查 erf 函数在 erf2inv(0, x, evaluate=False) 上的值是否等于 x

    assert erf(erf2inv(0, erf(erfcinv(1 - erf(erfinv(x)))))) == x  # 检查嵌套调用情况下 erf 函数的计算是否正确

    assert erf(I).is_real is False  # 检查 erf(I) 的实部是否为 False
    assert erf(0, evaluate=False).is_real  # 检查 erf(0, evaluate=False) 的实部是否为真
    assert erf(0, evaluate=False).is_zero  # 检查 erf(0, evaluate=False) 是否为零

    assert conjugate(erf(z)) == erf(conjugate(z))  # 检查共轭 erf(z) 是否等于 erf 的共轭 z

    assert erf(x).as_leading_term(x) == 2*x/sqrt(pi)  # 检查 erf(x) 的主导项是否为 2*x/sqrt(pi)
    assert erf(x*y).as_leading_term(y) == 2*x*y/sqrt(pi)  # 检查 erf(x*y) 关于 y 的主导项是否为 2*x*y/sqrt(pi)
    assert (erf(x*y)/erf(y)).as_leading_term(y) == x  # 检查 erf(x*y)/erf(y) 关于 y 的主导项是否为 x
    assert erf(1/x).as_leading_term(x) == S.One  # 检查 erf(1/x) 关于 x 的主导项是否为 1

    assert erf(z).rewrite('uppergamma') == sqrt(z**2)*(1 - erfc(sqrt(z**2)))/z  # 使用 uppergamma 重写 erf(z)
    assert erf(z).rewrite('erfc') == S.One - erfc(z)  # 使用 erfc 重写 erf(z)
    assert erf(z).rewrite('erfi') == -I*erfi(I*z)  # 使用 erfi 重写 erf(z)
    assert erf(z).rewrite('fresnels') == (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))  # 使用 fresnels 重写 erf(z)
    assert erf(z).rewrite('fresnelc') == (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))  # 使用 fresnelc 重写 erf(z)
    assert erf(z).rewrite('hyper') == 2*z*hyper([S.Half], [3*S.Half], -z**2)/sqrt(pi)  # 使用 hyper 重写 erf(z)
    assert erf(z).rewrite('meijerg') == z*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)/sqrt(pi)  # 使用 meijerg 重写 erf(z)
    assert erf(z).rewrite('expint') == sqrt(z**2)/z - z*expint(S.Half, z**2)/sqrt(S.Pi)  # 使用 expint 重写 erf(z)
    # 断言：当 x 趋向无穷大时，计算表达式 exp(x)*exp(x**2)*(erf(x + 1/exp(x)) - erf(x)) 的极限应该等于 2/sqrt(pi)
    assert limit(exp(x)*exp(x**2)*(erf(x + 1/exp(x)) - erf(x)), x, oo) == \
        2/sqrt(pi)
    
    # 断言：当 z 趋向无穷大时，计算表达式 (1 - erf(z))*exp(z**2)*z 的极限应该等于 1/sqrt(pi)
    assert limit((1 - erf(z))*exp(z**2)*z, z, oo) == 1/sqrt(pi)
    
    # 断言：当 x 趋向无穷大时，计算表达式 (1 - erf(x))*exp(x**2)*sqrt(pi)*x 的极限应该等于 1
    assert limit((1 - erf(x))*exp(x**2)*sqrt(pi)*x, x, oo) == 1
    
    # 断言：当 x 趋向无穷大时，计算表达式 ((1 - erf(x))*exp(x**2)*sqrt(pi)*x - 1)*2*x**2 的极限应该等于 -1
    assert limit(((1 - erf(x))*exp(x**2)*sqrt(pi)*x - 1)*2*x**2, x, oo) == -1
    
    # 断言：当 x 趋向 0 时，计算 erf(x)/x 的极限应该等于 2/sqrt(pi)
    assert limit(erf(x)/x, x, 0) == 2/sqrt(pi)
    
    # 断言：计算 x**(-4) - sqrt(pi)*erf(x**2) / (2*x**6) 在 x 趋向 0 时的极限应该等于 1/3
    assert limit(x**(-4) - sqrt(pi)*erf(x**2) / (2*x**6), x, 0) == S(1)/3

    # 断言：计算 erf(x) 的实部和虚部的形式，返回元组
    assert erf(x).as_real_imag() == \
        (erf(re(x) - I*im(x))/2 + erf(re(x) + I*im(x))/2,
         -I*(-erf(re(x) - I*im(x)) + erf(re(x) + I*im(x)))/2)

    # 断言：以深度搜索方式计算 erf(x) 的实部和虚部的形式，返回元组
    assert erf(x).as_real_imag(deep=False) == \
        (erf(re(x) - I*im(x))/2 + erf(re(x) + I*im(x))/2,
         -I*(-erf(re(x) - I*im(x)) + erf(re(x) + I*im(x)))/2)

    # 断言：计算 erf(w) 的实部和虚部的形式，预期返回元组 (erf(w), 0)
    assert erf(w).as_real_imag() == (erf(w), 0)
    
    # 断言：以深度搜索方式计算 erf(w) 的实部和虚部的形式，预期返回元组 (erf(w), 0)
    assert erf(w).as_real_imag(deep=False) == (erf(w), 0)
    
    # 断言：对于复数 I，计算 erf(I) 的实部和虚部的形式，预期返回元组 (0, -I*erf(I))
    # 这是 issue 13575 的相关问题
    assert erf(I).as_real_imag() == (0, -I*erf(I))

    # 断言：测试 erf(x) 的二阶导数，预期引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: erf(x).fdiff(2))

    # 断言：验证 erf(x) 的反函数应该等于 erfinv
    assert erf(x).inverse() == erfinv
def test_erf_series():
    # 断言：计算误差函数 erf(x) 在 x=0 处的级数展开，验证其结果是否符合预期
    assert erf(x).series(x, 0, 7) == 2*x/sqrt(pi) - \
        2*x**3/3/sqrt(pi) + x**5/5/sqrt(pi) + O(x**7)

    # 断言：计算误差函数 erf(x) 在 x 为无穷大时的级数展开，验证其结果是否符合预期
    assert erf(x).series(x, oo) == \
        -exp(-x**2)*(3/(4*x**5) - 1/(2*x**3) + 1/x + O(x**(-6), (x, oo)))/sqrt(pi) + 1
    # 断言：计算误差函数 erf(x**2) 在 x 为无穷大时的级数展开，验证其结果是否符合预期，限定展开项数为8
    assert erf(x**2).series(x, oo, n=8) == \
        (-1/(2*x**6) + x**(-2) + O(x**(-8), (x, oo)))*exp(-x**4)/sqrt(pi)*-1 + 1
    # 断言：计算误差函数 erf(sqrt(x)) 在 x 为无穷大时的级数展开，验证其结果是否符合预期，限定展开项数为3
    assert erf(sqrt(x)).series(x, oo, n=3) == (sqrt(1/x) - (1/x)**(S(3)/2)/2\
        + 3*(1/x)**(S(5)/2)/4 + O(x**(-3), (x, oo)))*exp(-x)/sqrt(pi)*-1 + 1


def test_erf_evalf():
    # 断言：计算误差函数 erf(2.0) 的数值近似结果是否在指定的精度范围内
    assert abs( erf(Float(2.0)) - 0.995322265 ) < 1E-8 # XXX


def test__erfs():
    # 断言：计算 _erfs(z) 对 z 的导数是否符合预期结果
    assert _erfs(z).diff(z) == -2/sqrt(S.Pi) + 2*z*_erfs(z)

    # 断言：计算 _erfs(1/z) 在 z 为无穷大时的级数展开，验证其结果是否符合预期
    assert _erfs(1/z).series(z) == \
        z/sqrt(pi) - z**3/(2*sqrt(pi)) + 3*z**5/(4*sqrt(pi)) + O(z**6)

    # 断言：展开 erf(z) 在 'tractable' 变换下的导数，验证其结果是否等于原始 erf(z) 的导数
    assert expand(erf(z).rewrite('tractable').diff(z).rewrite('intractable')) \
        == erf(z).diff(z)
    # 断言：_erfs(z) 在 'intractable' 变换下的表达式是否符合预期
    assert _erfs(z).rewrite("intractable") == (-erf(z) + 1)*exp(z**2)
    # 断言：检查 _erfs(z) 的二阶导数是否会触发参数索引错误异常
    raises(ArgumentIndexError, lambda: _erfs(z).fdiff(2))


def test_erfc():
    # 断言：计算 erfc(nan) 是否返回 nan
    assert erfc(nan) is nan

    # 断言：计算 erfc(oo) 是否返回 0
    assert erfc(oo) is S.Zero
    # 断言：计算 erfc(-oo) 是否返回 2
    assert erfc(-oo) == 2

    # 断言：计算 erfc(0) 是否返回 1
    assert erfc(0) == 1

    # 断言：计算 erfc(I*oo) 是否返回 -oo*I
    assert erfc(I*oo) == -oo*I
    # 断言：计算 erfc(-I*oo) 是否返回 oo*I
    assert erfc(-I*oo) == oo*I

    # 断言：计算 erfc(-x) 是否等于 2 - erfc(x)
    assert erfc(-x) == S(2) - erfc(x)
    # 断言：计算 erfc(erfcinv(x)) 是否返回 x
    assert erfc(erfcinv(x)) == x

    # 断言：检查 erfc(I) 是否为非实数
    assert erfc(I).is_real is False
    # 断言：检查 erfc(0, evaluate=False) 是否为实数
    assert erfc(0, evaluate=False).is_real
    # 断言：检查 erfc(0, evaluate=False) 是否不为零
    assert erfc(0, evaluate=False).is_zero is False

    # 断言：计算 erfc(erfinv(x)) 是否返回 1 - x
    assert erfc(erfinv(x)) == 1 - x

    # 断言：检查 erfc(z) 的共轭是否等于 erfc(z) 的共轭
    assert conjugate(erfc(z)) == erfc(conjugate(z))

    # 断言：计算 erfc(x) 的主导项是否为 1
    assert erfc(x).as_leading_term(x) is S.One
    # 断言：计算 erfc(1/x) 的主导项是否为 0
    assert erfc(1/x).as_leading_term(x) == S.Zero

    # 断言：计算 erfc(z) 在 'erf' 变换下的重写结果是否等于 1 - erf(z)
    assert erfc(z).rewrite('erf') == 1 - erf(z)
    # 断言：计算 erfc(z) 在 'erfi' 变换下的重写结果是否等于 1 + I*erfi(I*z)
    assert erfc(z).rewrite('erfi') == 1 + I*erfi(I*z)
    # 断言：计算 erfc(z) 在 'fresnels' 变换下的重写结果是否符合预期
    assert erfc(z).rewrite('fresnels') == 1 - (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))
    # 断言：计算 erfc(z) 在 'fresnelc' 变换下的重写结果是否符合预期
    assert erfc(z).rewrite('fresnelc') == 1 - (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))
    # 断言：计算 erfc(z) 在 'hyper' 变换下的重写结果是否符合预期
    assert erfc(z).rewrite('hyper') == 1 - 2*z*hyper([S.Half], [3*S.Half], -z**2)/sqrt(pi)
    # 断言：计算 erfc(z) 在 'meijerg' 变换下的重写结果是否符合预期
    assert erfc(z).rewrite('meijerg') == 1 - z*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)/sqrt(pi)
    # 断言：计算 erfc(z) 在 'uppergamma' 变换下的重写结果是否符合预期
    assert erfc(z).rewrite('uppergamma') == 1 - sqrt(z**2)*(1 - erfc(sqrt(z**2)))/z
    # 断言：计算 erfc(z) 在 'expint' 变换下的重写结果是否符合预期
    assert erfc(z).rewrite('expint') == S.One - sqrt(z**2)/z + z*expint(S.Half, z**2)/sqrt(S.Pi)
    # 断言：计算 erfc(z) 在 'tractable' 变换下的重写结果是否符合预期
    assert erfc(z).rewrite('tractable') == _erfs(z)*exp(-z**2)
    # 断言：展开 erf(x) + erfc(x)，验证其结果是否为 1
    assert expand_func(erf(x) + erfc(x)) is S.One

    # 断言：计算 erfc(x) 的实部和虚部是否符合预期
    assert erfc(x).as_real_imag() == \
        (erfc(re(x) - I*im(x))/2
    # 使用 assert 断言来验证 erfc(x) 的逆函数是否等于 erfcinv
    assert erfc(x).inverse() == erfcinv
# 定义测试函数，用于验证 erfc 函数的级数展开和评估结果
def test_erfc_series():
    # 断言：erfc(x) 的级数展开在 x 接近 0 时的前7项系数与期望值相等
    assert erfc(x).series(x, 0, 7) == 1 - 2*x/sqrt(pi) + \
        2*x**3/3/sqrt(pi) - x**5/5/sqrt(pi) + O(x**7)

    # 断言：erfc(x) 的级数展开在 x 趋向无穷时的表达式与期望值相等
    assert erfc(x).series(x, oo) == \
            (3/(4*x**5) - 1/(2*x**3) + 1/x + O(x**(-6), (x, oo)))*exp(-x**2)/sqrt(pi)


# 定义测试函数，用于验证 erfc 函数的浮点数评估
def test_erfc_evalf():
    # 断言：erfc(2.0) 的绝对误差小于 1E-8
    assert abs( erfc(Float(2.0)) - 0.00467773 ) < 1E-8 # XXX


# 定义测试函数，用于验证 erfi 函数的各种边界条件和特殊情况
def test_erfi():
    # 断言：erfi(nan) 结果为 nan
    assert erfi(nan) is nan

    # 断言：erfi(oo) 结果为正无穷
    assert erfi(oo) is S.Infinity
    # 断言：erfi(-oo) 结果为负无穷
    assert erfi(-oo) is S.NegativeInfinity

    # 断言：erfi(0) 结果为零
    assert erfi(0) is S.Zero

    # 断言：erfi(I*oo) 结果为复数单位虚数单位
    assert erfi(I*oo) == I
    # 断言：erfi(-I*oo) 结果为复数单位虚数单位的负数
    assert erfi(-I*oo) == -I

    # 断言：erfi(-x) 等于 -erfi(x)
    assert erfi(-x) == -erfi(x)

    # 断言：erfi(I*erfinv(x)) 等于 I*x
    assert erfi(I*erfinv(x)) == I*x
    # 断言：erfi(I*erfcinv(x)) 等于 I*(1 - x)
    assert erfi(I*erfcinv(x)) == I*(1 - x)
    # 断言：erfi(I*erf2inv(0, x)) 等于 I*x
    assert erfi(I*erf2inv(0, x)) == I*x
    # 断言：erfi(I*erf2inv(0, x, evaluate=False)) 等于 I*x，用于覆盖 erfi 中的代码
    assert erfi(I*erf2inv(0, x, evaluate=False)) == I*x # To cover code in erfi

    # 断言：erfi(I) 的结果不是实数
    assert erfi(I).is_real is False
    # 断言：erfi(0, evaluate=False) 的结果是实数
    assert erfi(0, evaluate=False).is_real
    # 断言：erfi(0, evaluate=False) 的结果是零
    assert erfi(0, evaluate=False).is_zero

    # 断言：erfi(z) 的共轭等于 erfi(z) 的共轭
    assert conjugate(erfi(z)) == erfi(conjugate(z))

    # 断言：erfi(x).as_leading_term(x) 等于 2*x/sqrt(pi)
    assert erfi(x).as_leading_term(x) == 2*x/sqrt(pi)
    # 断言：erfi(x*y).as_leading_term(y) 等于 2*x*y/sqrt(pi)
    assert erfi(x*y).as_leading_term(y) == 2*x*y/sqrt(pi)
    # 断言：(erfi(x*y)/erfi(y)).as_leading_term(y) 等于 x
    assert (erfi(x*y)/erfi(y)).as_leading_term(y) == x
    # 断言：erfi(1/x).as_leading_term(x) 等于 erfi(1/x)
    assert erfi(1/x).as_leading_term(x) == erfi(1/x)

    # 断言：erfi(z).rewrite('erf') 等于 -I*erf(I*z)
    assert erfi(z).rewrite('erf') == -I*erf(I*z)
    # 断言：erfi(z).rewrite('erfc') 等于 I*erfc(I*z) - I
    assert erfi(z).rewrite('erfc') == I*erfc(I*z) - I
    # 断言：erfi(z).rewrite('fresnels') 等于 (1 - I)*(fresnelc(z*(1 + I)/sqrt(pi)) - I*fresnels(z*(1 + I)/sqrt(pi)))
    assert erfi(z).rewrite('fresnels') == (1 - I)*(fresnelc(z*(1 + I)/sqrt(pi)) -
        I*fresnels(z*(1 + I)/sqrt(pi)))
    # 断言：erfi(z).rewrite('fresnelc') 等于 (1 - I)*(fresnelc(z*(1 + I)/sqrt(pi)) - I*fresnels(z*(1 + I)/sqrt(pi)))
    assert erfi(z).rewrite('fresnelc') == (1 - I)*(fresnelc(z*(1 + I)/sqrt(pi)) -
        I*fresnels(z*(1 + I)/sqrt(pi)))
    # 断言：erfi(z).rewrite('hyper') 等于 2*z*hyper([S.Half], [3*S.Half], z**2)/sqrt(pi)
    assert erfi(z).rewrite('hyper') == 2*z*hyper([S.Half], [3*S.Half], z**2)/sqrt(pi)
    # 断言：erfi(z).rewrite('meijerg') 等于 z*meijerg([S.Half], [], [0], [Rational(-1, 2)], -z**2)/sqrt(pi)
    assert erfi(z).rewrite('meijerg') == z*meijerg([S.Half], [], [0], [Rational(-1, 2)], -z**2)/sqrt(pi)
    # 断言：erfi(z).rewrite('uppergamma') 等于 (sqrt(-z**2)/z*(uppergamma(S.Half, -z**2)/sqrt(S.Pi) - S.One))
    assert erfi(z).rewrite('uppergamma') == (sqrt(-z**2)/z*(uppergamma(S.Half,
        -z**2)/sqrt(S.Pi) - S.One))
    # 断言：erfi(z).rewrite('expint') 等于 sqrt(-z**2)/z - z*expint(S.Half, -z**2)/sqrt(S.Pi)
    assert erfi(z).rewrite('expint') == sqrt(-z**2)/z - z*expint(S.Half, -z**2)/sqrt(S.Pi)
    # 断言：erfi(z).rewrite('tractable') 等于 -I*(-_erfs(I*z)*exp(z**2) + 1)
    assert erfi(z).rewrite('tractable') == -I*(-_erfs(I*z)*exp(z**2) + 1)
    # 断言：expand_func(erfi(I*z)) 等于 I*erf(z)
    assert expand_func(erfi(I*z)) == I*erf(z)

    # 断言：erfi(x).as_real_imag() 的结果为两个部分，表示实部和虚部
    assert erfi(x).as_real_imag() == \
        (erfi(re(x) - I*im(x))/2 + erfi(re(x) + I*im(x))/2,
         -I*(-erfi(re(x) - I*im(x)) + erfi(re(x) + I*im(x)))/2)
    # 断言：erfi(x).as_real_imag(deep=False) 的结果为两个部分，表示实部和虚部，非深度展开
    assert erfi(x).as_real_imag(deep=False) == \
        (erfi(re(x) - I*im(x))/2 + erfi(re(x) + I*im(x))/2,
         -I*(-erfi(re(x) - I*im(x)) + erfi(re(x) + I
    # 断言，验证调用 erf2 函数时传入 nan 和 0 的返回值是否为 nan
    assert erf2(nan, 0) is nan

    # 断言，验证调用 erf2 函数时传入 -oo 和 y 的返回值是否等于 erf(y) + 1
    assert erf2(-oo,  y) ==  erf(y) + 1
    # 断言，验证调用 erf2 函数时传入 oo 和 y 的返回值是否等于 erf(y) - 1
    assert erf2( oo,  y) ==  erf(y) - 1
    # 断言，验证调用 erf2 函数时传入 x 和 oo 的返回值是否等于 1 - erf(x)
    assert erf2(  x, oo) ==  1 - erf(x)
    # 断言，验证调用 erf2 函数时传入 x 和 -oo 的返回值是否等于 -1 - erf(x)
    assert erf2(  x,-oo) == -1 - erf(x)
    # 断言，验证调用 erf2 函数时传入 x 和 erf2inv(x, y) 的返回值是否等于 y
    assert erf2(x, erf2inv(x, y)) == y

    # 断言，验证调用 erf2 函数时传入 -x 和 -y 的返回值是否等于 -erf2(x,y)
    assert erf2(-x, -y) == -erf2(x,y)
    # 断言，验证调用 erf2 函数时传入 -x 和 y 的返回值是否等于 erf(y) + erf(x)
    assert erf2(-x,  y) == erf(y) + erf(x)
    # 断言，验证调用 erf2 函数时传入 x 和 -y 的返回值是否等于 -erf(y) - erf(x)
    assert erf2( x, -y) == -erf(y) - erf(x)
    # 断言，验证调用 erf2 函数的 rewrite('fresnels') 方法后的返回值是否等于 erf(y).rewrite(fresnels) - erf(x).rewrite(fresnels)
    assert erf2(x, y).rewrite('fresnels') == erf(y).rewrite(fresnels) - erf(x).rewrite(fresnels)
    # 断言，验证调用 erf2 函数的 rewrite('fresnelc') 方法后的返回值是否等于 erf(y).rewrite(fresnelc) - erf(x).rewrite(fresnelc)
    assert erf2(x, y).rewrite('fresnelc') == erf(y).rewrite(fresnelc) - erf(x).rewrite(fresnelc)
    # 断言，验证调用 erf2 函数的 rewrite('hyper') 方法后的返回值是否等于 erf(y).rewrite(hyper) - erf(x).rewrite(hyper)
    assert erf2(x, y).rewrite('hyper') == erf(y).rewrite(hyper) - erf(x).rewrite(hyper)
    # 断言，验证调用 erf2 函数的 rewrite('meijerg') 方法后的返回值是否等于 erf(y).rewrite(meijerg) - erf(x).rewrite(meijerg)
    assert erf2(x, y).rewrite('meijerg') == erf(y).rewrite(meijerg) - erf(x).rewrite(meijerg)
    # 断言，验证调用 erf2 函数的 rewrite('uppergamma') 方法后的返回值是否等于 erf(y).rewrite(uppergamma) - erf(x).rewrite(uppergamma)
    assert erf2(x, y).rewrite('uppergamma') == erf(y).rewrite(uppergamma) - erf(x).rewrite(uppergamma)
    # 断言，验证调用 erf2 函数的 rewrite('expint') 方法后的返回值是否等于 erf(y).rewrite(expint) - erf(x).rewrite(expint)
    assert erf2(x, y).rewrite('expint') == erf(y).rewrite(expint) - erf(x).rewrite(expint)

    # 断言，验证调用 erf2 函数时传入复数 I 和 0 的返回值的 is_real 属性是否为 False
    assert erf2(I, 0).is_real is False
    # 断言，验证调用 erf2 函数时传入 0 和 0 的返回值的 is_real 属性是否为 True
    assert erf2(0, 0, evaluate=False).is_real
    # 断言，验证调用 erf2 函数时传入 0 和 0 的返回值的 is_zero 属性是否为 True
    assert erf2(0, 0, evaluate=False).is_zero
    # 断言，验证调用 erf2 函数时传入 x 和 x 的返回值的 is_zero 属性是否为 True
    assert erf2(x, x, evaluate=False).is_zero
    # 断言，验证调用 erf2 函数时传入 x 和 y 的返回值的 is_zero 属性是否为 None
    assert erf2(x, y).is_zero is None

    # 断言，验证调用 expand_func 函数时对 erf(x) + erf2(x, y) 展开后的结果是否等于 erf(y)
    assert expand_func(erf(x) + erf2(x, y)) == erf(y)

    # 断言，验证调用 conjugate 函数时对 erf2(x, y) 取共轭后的结果是否等于 erf2(conjugate(x), conjugate(y))
    assert conjugate(erf2(x, y)) == erf2(conjugate(x), conjugate(y))

    # 断言，验证调用 erf2 函数的 rewrite('erf') 方法后的返回值是否等于 erf(y) - erf(x)
    assert erf2(x, y).rewrite('erf')  == erf(y) - erf(x)
    # 断言，验证调用 erf2 函数的 rewrite('erfc') 方法后的返回值是否等于 erfc(x) - erfc(y)
    assert erf2(x, y).rewrite('erfc') == erfc(x) - erfc(y)
    # 断言，验证调用 erf2 函数的 rewrite('erfi') 方法后的返回值是否等于 I*(erfi(I*x) - erfi(I*y))
    assert erf2(x, y).rewrite('erfi') == I*(erfi(I*x) - erfi(I*y))

    # 断言，验证调用 erf2 函数时对 x 求偏导数后的结果是否等于 erf2(x, y) 的一阶偏导数
    assert erf2(x, y).diff(x) == erf2(x, y).fdiff(1)
    # 断言，验证调用 erf2 函数时对 y 求偏导数后的结果是否等于 erf2(x, y) 的二阶偏导数
    assert erf2(x, y).diff(y) == erf2(x, y).fdiff(2)
    # 断言，验证调用 erf2 函数时对 x 求偏导数后的结果是否等于 -2*exp(-x**2)/sqrt(pi)
    assert erf2(x, y).diff(x) == -2*exp(-x**2)/sqrt(pi)
    # 断言，验证调用 erf2 函数时对 y 求偏导数后的结果是否等于 2*exp(-y**2)/sqrt(pi)
    assert erf2(x, y).diff(y) == 2*exp(-y**2)/sqrt(pi)
    # 断言，验证调用 erf2 函数的 fdiff(3) 方法时是否引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: erf2(x, y).fdiff(3))

    # 断言，验证调用 erf2 函数时传入 x 和 y 是否满足 is_extended_real 属性为 None
    assert erf2(x, y).is_extended_real is None
    # 声明符号 xr 和 yr 为 extended_real=True，并断言，验证调用 erf2 函数时传入 xr 和 yr 是否满足 is_extended_real 属性为 True
    xr, yr = symbols('xr yr', extended_real=True)
    assert erf2(xr, yr).is_extended_real is True
# 定义测试函数 test_erfinv，用于测试 erfinv 函数的不同输入情况
def test_erfinv():
    # 断言当输入为 0 时，erfinv 返回 S.Zero
    assert erfinv(0) is S.Zero
    # 断言当输入为 1 时，erfinv 返回 S.Infinity
    assert erfinv(1) is S.Infinity
    # 断言当输入为 NaN 时，erfinv 返回 S.NaN
    assert erfinv(nan) is S.NaN
    # 断言当输入为 -1 时，erfinv 返回 S.NegativeInfinity
    assert erfinv(-1) is S.NegativeInfinity

    # 断言 erfinv(erf(w)) 应当返回 w
    assert erfinv(erf(w)) == w
    # 断言 erfinv(erf(-w)) 应当返回 -w
    assert erfinv(erf(-w)) == -w

    # 断言 erfinv(x).diff() 的导数应当等于 sqrt(pi)*exp(erfinv(x)**2)/2
    assert erfinv(x).diff() == sqrt(pi)*exp(erfinv(x)**2)/2
    # 使用 lambda 函数断言调用 erfinv(x).fdiff(2) 抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: erfinv(x).fdiff(2))

    # 断言 erfinv(z).rewrite('erfcinv') 应当等于 erfcinv(1-z)
    assert erfinv(z).rewrite('erfcinv') == erfcinv(1-z)
    # 断言 erfinv(z).inverse() 应当等于 erf
    assert erfinv(z).inverse() == erf


# 定义测试函数 test_erfinv_evalf，用于测试 erfinv 函数在浮点数上的近似值
def test_erfinv_evalf():
    # 断言计算结果 erfinv(Float(0.2)) 的绝对误差小于 1E-13
    assert abs( erfinv(Float(0.2)) - 0.179143454621292 ) < 1E-13


# 定义测试函数 test_erfcinv，用于测试 erfcinv 函数的不同输入情况
def test_erfcinv():
    # 断言当输入为 1 时，erfcinv 返回 S.Zero
    assert erfcinv(1) is S.Zero
    # 断言当输入为 0 时，erfcinv 返回 S.Infinity
    assert erfcinv(0) is S.Infinity
    # 断言当输入为 NaN 时，erfcinv 返回 S.NaN
    assert erfcinv(nan) is S.NaN

    # 断言 erfcinv(x).diff() 的导数应当等于 -sqrt(pi)*exp(erfcinv(x)**2)/2
    assert erfcinv(x).diff() == -sqrt(pi)*exp(erfcinv(x)**2)/2
    # 使用 lambda 函数断言调用 erfcinv(x).fdiff(2) 抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: erfcinv(x).fdiff(2))

    # 断言 erfcinv(z).rewrite('erfinv') 应当等于 erfinv(1-z)
    assert erfcinv(z).rewrite('erfinv') == erfinv(1-z)
    # 断言 erfcinv(z).inverse() 应当等于 erfc
    assert erfcinv(z).inverse() == erfc


# 定义测试函数 test_erf2inv，用于测试 erf2inv 函数的不同输入情况
def test_erf2inv():
    # 断言当输入为 (0, 0) 时，erf2inv 返回 S.Zero
    assert erf2inv(0, 0) is S.Zero
    # 断言当输入为 (0, 1) 时，erf2inv 返回 S.Infinity
    assert erf2inv(0, 1) is S.Infinity
    # 断言当输入为 (1, 0) 时，erf2inv 返回 S.One
    assert erf2inv(1, 0) is S.One
    # 断言当输入为 (0, y) 时，erf2inv 返回 erfinv(y)
    assert erf2inv(0, y) == erfinv(y)
    # 断言当输入为 (oo, y) 时，erf2inv 返回 erfcinv(-y)
    assert erf2inv(oo, y) == erfcinv(-y)
    # 断言当输入为 (x, 0) 时，erf2inv 返回 x
    assert erf2inv(x, 0) == x
    # 断言当输入为 (x, oo) 时，erf2inv 返回 erfinv(x)
    assert erf2inv(x, oo) == erfinv(x)
    # 断言当输入为 (nan, 0) 时，erf2inv 返回 nan
    assert erf2inv(nan, 0) is nan
    # 断言当输入为 (0, nan) 时，erf2inv 返回 nan
    assert erf2inv(0, nan) is nan

    # 断言 erf2inv(x, y) 对 x 的偏导数应当为 exp(-x**2 + erf2inv(x, y)**2)
    assert erf2inv(x, y).diff(x) == exp(-x**2 + erf2inv(x, y)**2)
    # 断言 erf2inv(x, y) 对 y 的偏导数应当为 sqrt(pi)*exp(erf2inv(x, y)**2)/2
    assert erf2inv(x, y).diff(y) == sqrt(pi)*exp(erf2inv(x, y)**2)/2
    # 使用 lambda 函数断言调用 erf2inv(x, y).fdiff(3) 抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: erf2inv(x, y).fdiff(3))


# NOTE we multiply by exp_polar(I*pi) and need this to be on the principal
# branch, hence take x in the lower half plane (d=0).

# 定义 mytn 函数，用于测试表达式 expr1, expr2, expr3 的数值是否相等
def mytn(expr1, expr2, expr3, x, d=0):
    # 导入相关模块和函数
    from sympy.core.random import verify_numerically, random_complex_number
    subs = {}
    # 遍历 expr1 的自由符号，如果不是 x，则使用随机复数替代
    for a in expr1.free_symbols:
        if a != x:
            subs[a] = random_complex_number()
    # 返回表达式 expr2 是否等于 expr3，并且验证数值上的相等性
    return expr2 == expr3 and verify_numerically(expr1.subs(subs),
                                               expr2.subs(subs), x, d=d)


# 定义 mytd 函数，用于测试表达式 expr1 对 x 的导数是否等于 expr2
def mytd(expr1, expr2, x):
    # 导入相关模块和函数
    from sympy.core.random import test_derivative_numerically, \
        random_complex_number
    subs = {}
    # 遍历 expr1 的自由符号，如果不是 x，则使用随机复数替代
    for a in expr1.free_symbols:
        if a != x:
            subs[a] = random_complex_number()
    # 返回 expr1 对 x 的导数是否等于 expr2，并且通过数值测试验证
    return expr1.diff(x) == expr2 and test_derivative_numerically(expr1.subs(subs), x)


# 定义 tn_branch 函数，用于测试函数 func 在特定点 s 的分支性质
def tn_branch(func, s=None):
    # 导入相关模块和函数
    from sympy.core.random import uniform

    def fn(x):
        if s is None:
            return func(x)
        return func(s, x)
    c = uniform(1, 5)
    # 计算 fn(c*exp_polar(I*pi)) 和 fn(c*exp_polar(-I*pi)) 的差值
    expr = fn(c*exp_polar(I*pi)) - fn(c*exp_polar(-I*pi))
    eps = 1e-15
    # 计算 fn(-c + eps*I) 和 fn(-c - eps*I) 的差值，并比较精度是否足够小
    return abs(expr.n() - fn(-c + eps*I).n() + fn(-c - eps*I).n()) < 1e-10


# 定义测试函数 test_ei，用
    # 断言：Ei(x)经由rewrite(expint)再rewrite(Ei)得到的结果应当等于Ei(x)本身
    assert Ei(x).rewrite(expint).rewrite(Ei) == Ei(x)
    
    # 断言：Ei(x * exp_polar(2*I*pi))应当等于Ei(x)加上2*I*pi
    assert Ei(x*exp_polar(2*I*pi)) == Ei(x) + 2*I*pi
    
    # 断言：Ei(x * exp_polar(-2*I*pi))应当等于Ei(x)减去2*I*pi
    assert Ei(x*exp_polar(-2*I*pi)) == Ei(x) - 2*I*pi

    # 断言：通过mytn函数验证 Ei(x)、Ei(x).rewrite(Shi)、Chi(x) + Shi(x)、x 四者的等价性
    assert mytn(Ei(x), Ei(x).rewrite(Shi), Chi(x) + Shi(x), x)
    
    # 断言：通过mytn函数验证 Ei(x*polar_lift(I))、Ei(x*polar_lift(I)).rewrite(Si)、Ci(x) + I*Si(x) + I*pi/2、x 四者的等价性
    assert mytn(Ei(x*polar_lift(I)), Ei(x*polar_lift(I)).rewrite(Si),
                Ci(x) + I*Si(x) + I*pi/2, x)

    # 断言：Ei(log(x))经由rewrite(li)得到的结果应当等于li(x)
    assert Ei(log(x)).rewrite(li) == li(x)
    
    # 断言：Ei(2*log(x))经由rewrite(li)得到的结果应当等于li(x**2)
    assert Ei(2*log(x)).rewrite(li) == li(x**2)

    # 断言：当x趋向无穷大时，gruntz(Ei(x+exp(-x))*exp(-x)*x, x, oo)应当等于1
    assert gruntz(Ei(x+exp(-x))*exp(-x)*x, x, oo) == 1

    # 断言：Ei(x)在x处的级数展开应当等于EulerGamma + log(x) + x + x**2/4 + x**3/18 + x**4/96 + x**5/600 + O(x**6)
    assert Ei(x).series(x) == EulerGamma + log(x) + x + x**2/4 + \
        x**3/18 + x**4/96 + x**5/600 + O(x**6)
    
    # 断言：Ei(x)在x=1处展开到三阶，应当等于Ei(1) + E*(x - 1) + O((x - 1)**3, (x, 1))
    assert Ei(x).series(x, 1, 3) == Ei(1) + E*(x - 1) + O((x - 1)**3, (x, 1))
    
    # 断言：Ei(x)在x趋向无穷大时的级数展开应当等于(120/x**5 + 24/x**4 + 6/x**3 + 2/x**2 + 1/x + 1 + O(x**(-6), (x, oo)))*exp(x)/x
    assert Ei(x).series(x, oo) == \
        (120/x**5 + 24/x**4 + 6/x**3 + 2/x**2 + 1/x + 1 + O(x**(-6), (x, oo)))*exp(x)/x

    # 断言：Ei(cos(2))的数值计算结果字符串表示应当等于 '-0.6760647401'
    assert str(Ei(cos(2)).evalf(n=10)) == '-0.6760647401'
    
    # 断言：调用Ei(x).fdiff(2)应当引发ArgumentIndexError异常
    raises(ArgumentIndexError, lambda: Ei(x).fdiff(2))
def test_expint():
    # 检查 expint 函数的数学等式是否成立
    assert mytn(expint(x, y), expint(x, y).rewrite(uppergamma),
                y**(x - 1)*uppergamma(1 - x, y), x)
    # 检查 expint 函数的数学等式是否成立
    assert mytd(
        expint(x, y), -y**(x - 1)*meijerg([], [1, 1], [0, 0, 1 - x], [], y), x)
    # 检查 expint 函数的数学等式是否成立
    assert mytd(expint(x, y), -expint(x - 1, y), y)
    # 检查 expint 函数的数学等式是否成立
    assert mytn(expint(1, x), expint(1, x).rewrite(Ei),
                -Ei(x*polar_lift(-1)) + I*pi, x)

    # 检查 expint 函数的数学等式是否成立
    assert expint(-4, x) == exp(-x)/x + 4*exp(-x)/x**2 + 12*exp(-x)/x**3 \
        + 24*exp(-x)/x**4 + 24*exp(-x)/x**5
    # 检查 expint 函数的数学等式是否成立
    assert expint(Rational(-3, 2), x) == \
        exp(-x)/x + 3*exp(-x)/(2*x**2) + 3*sqrt(pi)*erfc(sqrt(x))/(4*x**S('5/2'))

    # 检查 expint 函数的分支点
    assert tn_branch(expint, 1)
    # 检查 expint 函数的分支点
    assert tn_branch(expint, 2)
    # 检查 expint 函数的分支点
    assert tn_branch(expint, 3)
    # 检查 expint 函数的分支点
    assert tn_branch(expint, 1.7)
    # 检查 expint 函数的分支点
    assert tn_branch(expint, pi)

    # 检查 expint 函数在复数变换下的等式是否成立
    assert expint(y, x*exp_polar(2*I*pi)) == \
        x**(y - 1)*(exp(2*I*pi*y) - 1)*gamma(-y + 1) + expint(y, x)
    # 检查 expint 函数在复数变换下的等式是否成立
    assert expint(y, x*exp_polar(-2*I*pi)) == \
        x**(y - 1)*(exp(-2*I*pi*y) - 1)*gamma(-y + 1) + expint(y, x)
    # 检查 expint 函数在复数变换下的等式是否成立
    assert expint(2, x*exp_polar(2*I*pi)) == 2*I*pi*x + expint(2, x)
    # 检查 expint 函数在复数变换下的等式是否成立
    assert expint(2, x*exp_polar(-2*I*pi)) == -2*I*pi*x + expint(2, x)
    # 检查 expint 函数的数学等式是否成立
    assert expint(1, x).rewrite(Ei).rewrite(expint) == expint(1, x)
    # 检查 expint 函数的数学等式是否成立
    assert expint(x, y).rewrite(Ei) == expint(x, y)
    # 检查 expint 函数的数学等式是否成立
    assert expint(x, y).rewrite(Ci) == expint(x, y)

    # 检查 E1 函数的数学等式是否成立
    assert mytn(E1(x), E1(x).rewrite(Shi), Shi(x) - Chi(x), x)
    # 检查 E1 函数的数学等式是否成立
    assert mytn(E1(polar_lift(I)*x), E1(polar_lift(I)*x).rewrite(Si),
                -Ci(x) + I*Si(x) - I*pi/2, x)

    # 检查 expint 函数的数学等式是否成立
    assert mytn(expint(2, x), expint(2, x).rewrite(Ei).rewrite(expint),
                -x*E1(x) + exp(-x), x)
    # 检查 expint 函数的数学等式是否成立
    assert mytn(expint(3, x), expint(3, x).rewrite(Ei).rewrite(expint),
                x**2*E1(x)/2 + (1 - x)*exp(-x)/2, x)

    # 检查 expint 函数的级数展开是否成立
    assert expint(Rational(3, 2), z).nseries(z) == \
        2 + 2*z - z**2/3 + z**3/15 - z**4/84 + z**5/540 - \
        2*sqrt(pi)*sqrt(z) + O(z**6)

    # 检查 E1 函数的级数展开是否成立
    assert E1(z).series(z) == -EulerGamma - log(z) + z - \
        z**2/4 + z**3/18 - z**4/96 + z**5/600 + O(z**6)

    # 检查 expint 函数的级数展开是否成立
    assert expint(4, z).series(z) == Rational(1, 3) - z/2 + z**2/2 + \
        z**3*(log(z)/6 - Rational(11, 36) + EulerGamma/6 - I*pi/6) - z**4/24 + \
        z**5/240 + O(z**6)

    # 检查 expint 函数的级数展开是否成立
    assert expint(n, x).series(x, oo, n=3) == \
        (n*(n + 1)/x**2 - n/x + 1 + O(x**(-3), (x, oo)))*exp(-x)/x

    # 检查 expint 函数的级数展开是否成立
    assert expint(z, y).series(z, 0, 2) == exp(-y)/y - z*meijerg(((), (1, 1)),
                                  ((0, 0, 1), ()), y)/y + O(z**2)
    # 检查 expint 函数的异常情况处理是否成立
    raises(ArgumentIndexError, lambda: expint(x, y).fdiff(3))

    # 创建一个负数符号
    neg = Symbol('neg', negative=True)
    # 检查 Ei 函数在使用 Si 重写后的等式是否成立
    assert Ei(neg).rewrite(Si) == Shi(neg) + Chi(neg) - I*pi


def test__eis():
    # 检查 _eis 函数的导数是否正确
    assert _eis(z).diff(z) == -_eis(z) + 1/z

    # 检查 _eis 函数在级数展开时的等式是否成立
    assert _eis(1/z).series(z) == \
        z + z**2 + 2*z**3 + 6*z**4 + 24*z**5 + O(z**6)

    # 检查 Ei 函数在 'tractable' 重写模式下的等式是否成立
    assert Ei(z).rewrite('tractable') == exp(z)*_eis(z)
    # 检查 li 函数在 'tractable' 重写模式下的等式是否成立
    assert li(z).rewrite('tractable') == z*_eis(log(z))

    # 检查 _eis 函数在 'intractable' 重写模式下的等式是否成立
    assert _eis(z).rewrite('intractable') == exp(-z)*Ei(z)
    # 断言：使用符号代换 'tractable'，然后对 z 求导，并再次使用符号代换 'intractable'，最后展开表达式。
    assert expand(li(z).rewrite('tractable').diff(z).rewrite('intractable')) \
        == li(z).diff(z)

    # 断言：使用符号代换 'tractable'，然后对 z 求导，并再次使用符号代换 'intractable'，最后展开表达式。
    assert expand(Ei(z).rewrite('tractable').diff(z).rewrite('intractable')) \
        == Ei(z).diff(z)

    # 断言：对 _eis(z) 在 z 处展开到 n=3 的级数，期望结果是 EulerGamma + log(z) + z*(-log(z) - EulerGamma + 1) + z**2*(log(z)/2 - Rational(3, 4) + EulerGamma/2) + O(z**3*log(z))
    assert _eis(z).series(z, n=3) == EulerGamma + log(z) + z*(-log(z) - \
        EulerGamma + 1) + z**2*(log(z)/2 - Rational(3, 4) + EulerGamma/2)\
        + O(z**3*log(z))
    
    # 断言：尝试对 _eis(z) 进行二阶求导，期望引发 ArgumentIndexError 异常。
    raises(ArgumentIndexError, lambda: _eis(z).fdiff(2))
# 定义一个装饰器函数 `tn_arg`，接受一个函数 `func` 作为参数
def tn_arg(func):
    # 定义内部函数 `test`，接受三个参数 `arg`, `e1`, `e2`
    def test(arg, e1, e2):
        # 从 sympy.core.random 模块导入 uniform 函数
        from sympy.core.random import uniform
        # 在区间 [1, 5] 内生成一个随机数 v
        v = uniform(1, 5)
        # 计算 func(arg*x) 在 x=v 处的值，并转换为数值
        v1 = func(arg*x).subs(x, v).n()
        # 计算 func(e1*v + e2*1e-15) 在 x=v 处的值，并转换为数值
        v2 = func(e1*v + e2*1e-15).n()
        # 返回两个计算结果的绝对误差是否小于 1e-10
        return abs(v1 - v2).n() < 1e-10
    # 返回内部函数 `test` 的引用
    return test(exp_polar(I*pi/2), I, 1) and \
        test(exp_polar(-I*pi/2), -I, 1) and \
        test(exp_polar(I*pi), -1, I) and \
        test(exp_polar(-I*pi), -1, -I)

# 定义一个函数 `test_li`
def test_li():
    # 创建符号对象 z, zr, zp, zn
    z = Symbol("z")
    zr = Symbol("z", real=True)
    zp = Symbol("z", positive=True)
    zn = Symbol("z", negative=True)

    # 断言验证 li(0) 的返回值是 S.Zero
    assert li(0) is S.Zero
    # 断言验证 li(1) 的返回值是 -oo
    assert li(1) is -oo
    # 断言验证 li(oo) 的返回值是 oo
    assert li(oo) is oo

    # 断言验证 li(z) 返回的对象是否为 li 类型
    assert isinstance(li(z), li)
    # 断言验证 unchanged(li, -zp) 返回 True
    assert unchanged(li, -zp)
    # 断言验证 unchanged(li, zn) 返回 True
    assert unchanged(li, zn)

    # 断言验证 li(z) 对 z 的偏导数是否为 1/log(z)
    assert diff(li(z), z) == 1/log(z)

    # 断言验证 conjugate(li(z)) 等于 li(conjugate(z))
    assert conjugate(li(z)) == li(conjugate(z))
    # 断言验证 conjugate(li(-zr)) 等于 li(-zr)
    assert conjugate(li(-zr)) == li(-zr)
    # 断言验证 unchanged(conjugate, li(-zp)) 返回 True
    assert unchanged(conjugate, li(-zp))
    # 断言验证 unchanged(conjugate, li(zn)) 返回 True
    assert unchanged(conjugate, li(zn))

    # 断言验证 li(z) 用 Li 重写后的结果
    assert li(z).rewrite(Li) == Li(z) + li(2)
    # 断言验证 li(z) 用 Ei 重写后的结果
    assert li(z).rewrite(Ei) == Ei(log(z))
    # 断言验证 li(z) 用 uppergamma 重写后的结果
    assert li(z).rewrite(uppergamma) == (-log(1/log(z))/2 - log(-log(z)) +
                                         log(log(z))/2 - expint(1, -log(z)))
    # 断言验证 li(z) 用 Si 重写后的结果
    assert li(z).rewrite(Si) == (-log(I*log(z)) - log(1/log(z))/2 +
                                 log(log(z))/2 + Ci(I*log(z)) + Shi(log(z)))
    # 断言验证 li(z) 用 Ci 重写后的结果
    assert li(z).rewrite(Ci) == (-log(I*log(z)) - log(1/log(z))/2 +
                                 log(log(z))/2 + Ci(I*log(z)) + Shi(log(z)))
    # 断言验证 li(z) 用 Shi 重写后的结果
    assert li(z).rewrite(Shi) == (-log(1/log(z))/2 + log(log(z))/2 +
                                  Chi(log(z)) - Shi(log(z)))
    # 断言验证 li(z) 用 Chi 重写后的结果
    assert li(z).rewrite(Chi) == (-log(1/log(z))/2 + log(log(z))/2 +
                                  Chi(log(z)) - Shi(log(z)))
    # 断言验证 li(z) 用 hyper 重写后的结果
    assert li(z).rewrite(hyper) ==(log(z)*hyper((1, 1), (2, 2), log(z)) -
                                   log(1/log(z))/2 + log(log(z))/2 + EulerGamma)
    # 断言验证 li(z) 用 meijerg 重写后的结果
    assert li(z).rewrite(meijerg) == (-log(1/log(z))/2 - log(-log(z)) + log(log(z))/2 -
                                      meijerg(((), (1,)), ((0, 0), ()), -log(z)))

    # 断言验证 gruntz(1/li(z), z, oo) 的返回值是 S.Zero
    assert gruntz(1/li(z), z, oo) is S.Zero
    # 断言验证 li(z) 的级数展开结果
    assert li(z).series(z) == log(z)**5/600 + log(z)**4/96 + log(z)**3/18 + log(z)**2/4 + \
            log(z) + log(log(z)) + EulerGamma
    # 断言验证 lambda 函数抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: li(z).fdiff(2))


# 定义函数 `test_Li`
def test_Li():
    # 断言验证 Li(2) 的返回值是 S.Zero
    assert Li(2) is S.Zero
    # 断言验证 Li(oo) 的返回值是 oo
    assert Li(oo) is oo

    # 断言验证 Li(z) 返回的对象是否为 Li 类型
    assert isinstance(Li(z), Li)

    # 断言验证 Li(z) 对 z 的偏导数是否为 1/log(z)
    assert diff(Li(z), z) == 1/log(z)

    # 断言验证 gruntz(1/Li(z), z, oo) 的返回值是 S.Zero
    assert gruntz(1/Li(z), z, oo) is S.Zero
    # 断言验证 Li(z) 用 li 重写后的结果
    assert Li(z).rewrite(li) == li(z) - li(2)
    # 断言验证 Li(z) 的级数展开结果
    assert Li(z).series(z) == \
        log(z)**5/600 + log(z)**4/96 + log(z)**3/18 + log(z)**2/4 + log(z) + log(log(z)) - li(2) + EulerGamma
    # 断言验证 lambda 函数抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: Li(z).fdiff(2))


# 定义函数 `test_si`
def test_si():
    # 断言验证 Si(I*x) 等于 I*Shi(x)
    assert Si(I*x) == I*Shi(x)
    # 断言验证 Shi(I*x) 等于 I*Si(x)
    assert Shi(I*x) == I*Si(x)
    # 断言验证 Si(-I*x) 等于 -I*Shi(x)
    assert Si(-I*x) == -I*Shi(x)
    # 断言验证 Shi(-I*x) 等于 -I*Si(x)
    assert Shi(-I*x) == -I*Si(x)
    # 断言验证 Si(-x) 等于 -Si(x)
    assert Si(-x) == -Si(x)
    # 断言验证 Shi(-x) 等于 -Shi(x)
    assert Shi(-x) == -Shi(x)
    # 断言验证 Si(exp_polar(2*pi*I)*x) 等于 Si(x)
    assert Si(exp_polar(2*pi*I)*x) == Si(x)
    # 检查正弦积分的特定性质
    assert Si(exp_polar(-2*pi*I)*x) == Si(x)
    assert Shi(exp_polar(2*pi*I)*x) == Shi(x)
    assert Shi(exp_polar(-2*pi*I)*x) == Shi(x)

    # 检查正弦积分在无穷远点和负无穷远点的极限值
    assert Si(oo) == pi/2
    assert Si(-oo) == -pi/2
    assert Shi(oo) is oo
    assert Shi(-oo) is -oo

    # 检查自定义的特定导数
    assert mytd(Si(x), sin(x)/x, x)
    assert mytd(Shi(x), sinh(x)/x, x)

    # 检查正弦积分和双曲正弦积分的特定重写形式
    assert mytn(Si(x), Si(x).rewrite(Ei),
                -I*(-Ei(x*exp_polar(-I*pi/2))/2
               + Ei(x*exp_polar(I*pi/2))/2 - I*pi) + pi/2, x)
    assert mytn(Si(x), Si(x).rewrite(expint),
                -I*(-expint(1, x*exp_polar(-I*pi/2))/2 +
                    expint(1, x*exp_polar(I*pi/2))/2) + pi/2, x)
    assert mytn(Shi(x), Shi(x).rewrite(Ei),
                Ei(x)/2 - Ei(x*exp_polar(I*pi))/2 + I*pi/2, x)
    assert mytn(Shi(x), Shi(x).rewrite(expint),
                expint(1, x)/2 - expint(1, x*exp_polar(I*pi))/2 - I*pi/2, x)

    # 检查正弦积分和双曲正弦积分的参数
    assert tn_arg(Si)
    assert tn_arg(Shi)

    # 检查正弦积分的导数在给定表达式下的主导项
    assert Si(x)._eval_as_leading_term(x) == x
    assert Si(2*x)._eval_as_leading_term(x) == 2*x
    assert Si(sin(x))._eval_as_leading_term(x) == x
    assert Si(x + 1)._eval_as_leading_term(x) == Si(1)
    assert Si(1/x)._eval_as_leading_term(x, cdir=1) == \
        Si(1/x)._eval_as_leading_term(x, cdir=-1) == Si(1/x)

    # 检查正弦积分和双曲正弦积分的泰勒展开
    assert Si(x).nseries(x, n=8) == \
        x - x**3/18 + x**5/600 - x**7/35280 + O(x**8)
    assert Shi(x).nseries(x, n=8) == \
        x + x**3/18 + x**5/600 + x**7/35280 + O(x**8)
    assert Si(sin(x)).nseries(x, n=5) == x - 2*x**3/9 + O(x**5)
    assert Si(x).nseries(x, 1, n=3) == \
        Si(1) + (x - 1)*sin(1) + (x - 1)**2*(-sin(1)/2 + cos(1)/2) + O((x - 1)**3, (x, 1))

    # 检查正弦积分在无穷远点的级数展开
    assert Si(x).series(x, oo) == -sin(x)*(-6/x**4 + x**(-2) + O(x**(-6), (x, oo))) - \
        cos(x)*(24/x**5 - 2/x**3 + 1/x + O(x**(-6), (x, oo))) + pi/2

    # 使用虚数变量 t，检查正弦积分的另一种重写形式
    t = Symbol('t', Dummy=True)
    assert Si(x).rewrite(sinc).dummy_eq(Integral(sinc(t), (t, 0, x)))

    # 检查双曲正弦积分在无穷远点和负无穷远点的极限值
    assert limit(Shi(x), x, S.Infinity) == S.Infinity
    assert limit(Shi(x), x, S.NegativeInfinity) == S.NegativeInfinity
def test_ci():
    # 计算 exp_polar(I*pi)，得到复数的指数形式表示
    m1 = exp_polar(I*pi)
    # 计算 exp_polar(-I*pi)，得到复数的指数形式表示
    m1_ = exp_polar(-I*pi)
    # 计算 exp_polar(I*pi/2)，得到复数的指数形式表示
    pI = exp_polar(I*pi/2)
    # 计算 exp_polar(-I*pi/2)，得到复数的指数形式表示
    mI = exp_polar(-I*pi/2)

    # 断言 Ci(m1*x) 的值等于 Ci(x) + I*pi
    assert Ci(m1*x) == Ci(x) + I*pi
    # 断言 Ci(m1_*x) 的值等于 Ci(x) - I*pi
    assert Ci(m1_*x) == Ci(x) - I*pi
    # 断言 Ci(pI*x) 的值等于 Chi(x) + I*pi/2
    assert Ci(pI*x) == Chi(x) + I*pi/2
    # 断言 Ci(mI*x) 的值等于 Chi(x) - I*pi/2
    assert Ci(mI*x) == Chi(x) - I*pi/2
    # 断言 Chi(m1*x) 的值等于 Chi(x) + I*pi
    assert Chi(m1*x) == Chi(x) + I*pi
    # 断言 Chi(m1_*x) 的值等于 Chi(x) - I*pi
    assert Chi(m1_*x) == Chi(x) - I*pi
    # 断言 Chi(pI*x) 的值等于 Ci(x) + I*pi/2
    assert Chi(pI*x) == Ci(x) + I*pi/2
    # 断言 Chi(mI*x) 的值等于 Ci(x) - I*pi/2
    assert Chi(mI*x) == Ci(x) - I*pi/2
    # 断言 Ci(exp_polar(2*I*pi)*x) 的值等于 Ci(x) + 2*I*pi
    assert Ci(exp_polar(2*I*pi)*x) == Ci(x) + 2*I*pi
    # 断言 Chi(exp_polar(-2*I*pi)*x) 的值等于 Chi(x) - 2*I*pi
    assert Chi(exp_polar(-2*I*pi)*x) == Chi(x) - 2*I*pi
    # 断言 Chi(exp_polar(2*I*pi)*x) 的值等于 Chi(x) + 2*I*pi
    assert Chi(exp_polar(2*I*pi)*x) == Chi(x) + 2*I*pi
    # 断言 Ci(exp_polar(-2*I*pi)*x) 的值等于 Ci(x) - 2*I*pi
    assert Ci(exp_polar(-2*I*pi)*x) == Ci(x) - 2*I*pi

    # 断言 Ci(oo) 的值是 S.Zero
    assert Ci(oo) is S.Zero
    # 断言 Ci(-oo) 的值等于 I*pi
    assert Ci(-oo) == I*pi
    # 断言 Chi(oo) 的值是 oo
    assert Chi(oo) is oo
    # 断言 Chi(-oo) 的值是 oo
    assert Chi(-oo) is oo

    # 断言 mytd(Ci(x), cos(x)/x, x) 返回 True
    assert mytd(Ci(x), cos(x)/x, x)
    # 断言 mytd(Chi(x), cosh(x)/x, x) 返回 True
    assert mytd(Chi(x), cosh(x)/x, x)

    # 断言 mytn(Ci(x), Ci(x).rewrite(Ei), ...) 返回 True
    assert mytn(Ci(x), Ci(x).rewrite(Ei),
                Ei(x*exp_polar(-I*pi/2))/2 + Ei(x*exp_polar(I*pi/2))/2, x)
    # 断言 mytn(Chi(x), Chi(x).rewrite(Ei), ...) 返回 True
    assert mytn(Chi(x), Chi(x).rewrite(Ei),
                Ei(x)/2 + Ei(x*exp_polar(I*pi))/2 - I*pi/2, x)

    # 断言 tn_arg(Ci) 返回 True
    assert tn_arg(Ci)
    # 断言 tn_arg(Chi) 返回 True
    assert tn_arg(Chi)

    # 断言 Ci(x).nseries(x, n=4) 的级数展开结果
    assert Ci(x).nseries(x, n=4) == \
        EulerGamma + log(x) - x**2/4 + O(x**4)
    # 断言 Chi(x).nseries(x, n=4) 的级数展开结果
    assert Chi(x).nseries(x, n=4) == \
        EulerGamma + log(x) + x**2/4 + O(x**4)

    # 断言 Ci(x).series(x, oo) 的级数展开结果
    assert Ci(x).series(x, oo) == -cos(x)*(-6/x**4 + x**(-2) + O(x**(-6), (x, oo))) + \
        sin(x)*(24/x**5 - 2/x**3 + 1/x + O(x**(-6), (x, oo)))

    # 断言 Ci(x).series(x, -oo) 的级数展开结果
    assert Ci(x).series(x, -oo) == -cos(x)*(-6/x**4 + x**(-2) + O(x**(-6), (x, -oo))) + \
        sin(x)*(24/x**5 - 2/x**3 + 1/x + O(x**(-6), (x, -oo))) + I*pi

    # 断言 limit(log(x) - Ci(2*x), x, 0) 的极限结果
    assert limit(log(x) - Ci(2*x), x, 0) == -log(2) - EulerGamma
    # 断言 Ci(x).rewrite(uppergamma) 的重写结果
    assert Ci(x).rewrite(uppergamma) == -expint(1, x*exp_polar(-I*pi/2))/2 -\
                                        expint(1, x*exp_polar(I*pi/2))/2
    # 断言 Ci(x).rewrite(expint) 的重写结果
    assert Ci(x).rewrite(expint) == -expint(1, x*exp_polar(-I*pi/2))/2 -\
                                        expint(1, x*exp_polar(I*pi/2))/2
    # 断言 Ci(x).fdiff(2) 引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: Ci(x).fdiff(2))


def test_fresnel():
    # 断言 fresnels(0) 的值是 S.Zero
    assert fresnels(0) is S.Zero
    # 断言 fresnels(oo) 的值是 S.Half
    assert fresnels(oo) is S.Half
    # 断言 fresnels(-oo) 的值等于 Rational(-1, 2)
    assert fresnels(-oo) == Rational(-1, 2)
    # 断言 fresnels(I*oo) 的值等于 -I*S.Half
    assert fresnels(I*oo) == -I*S.Half

    # 断言 unchanged(fresnels, z) 返回 True
    assert unchanged(fresnels, z)
    # 断言 fresnels(-z) 的值等于 -fresnels(z)
    assert fresnels(-z) == -fresnels(z)
    # 断言 fresnels(I*z) 的值等于 -I*fresnels(z)
    assert fresnels(I*z) == -I*fresnels(z)
    # 断言 fresnels(-I*z) 的值等于 I*fresnels(z)
    assert fresnels(-I*z) == I*fresnels(z)

    # 断言 conjugate(fresnels(z)) 的值等于 fresnels(conjugate(z))
    assert conjugate(fresnels(z)) == fresnels(conjugate(z))

    # 断言 fresnels(z).diff(z) 的导数等于 sin(pi*z**2/2)
    assert fresnels(z).diff(z) == sin(pi*z**2/2)

    # 断言 fresnels(z).rewrite(erf) 的重写结果
    assert fresnels(z).rewrite(erf) == (S.One + I)/4 * (
        erf((S.One + I)/2*sqrt(pi)*z) - I*erf((S.One - I)/2*sqrt(pi)*z))

    # 断言 fresnels(z).rewrite(hyper) 的重写结果
    assert fresnels(z).rewrite(hyper) == \
        pi*z**3/6 * hyper([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)], -pi**2*z**4/16)

    # 断言 fresnels(z).series(z, n=15) 的级数展开
    # 断言：验证 fresnels(z) 的有限性
    assert fresnels(z).is_finite is None
    
    # 断言：验证 fresnels(z) 的实部和虚部分解
    assert fresnels(z).as_real_imag() == (
        fresnels(re(z) - I*im(z))/2 + fresnels(re(z) + I*im(z))/2,
        -I*(-fresnels(re(z) - I*im(z)) + fresnels(re(z) + I*im(z)))/2
    )
    
    # 断言：验证 fresnels(z) 的实部和虚部分解（浅层）
    assert fresnels(z).as_real_imag(deep=False) == (
        fresnels(re(z) - I*im(z))/2 + fresnels(re(z) + I*im(z))/2,
        -I*(-fresnels(re(z) - I*im(z)) + fresnels(re(z) + I*im(z)))/2
    )
    
    # 断言：验证 fresnels(w) 的实部和虚部分解
    assert fresnels(w).as_real_imag() == (fresnels(w), 0)
    
    # 断言：验证 fresnels(w) 的实部和虚部分解（深层）
    assert fresnels(w).as_real_imag(deep=True) == (fresnels(w), 0)
    
    # 断言：验证 fresnels(2 + 3*I) 的实部和虚部分解
    assert fresnels(2 + 3*I).as_real_imag() == (
        fresnels(2 + 3*I)/2 + fresnels(2 - 3*I)/2,
        -I*(fresnels(2 + 3*I) - fresnels(2 - 3*I))/2
    )
    
    # 断言：验证对 fresnels(z) 进行积分展开的结果
    assert expand_func(integrate(fresnels(z), z)) == \
        z*fresnels(z) + cos(pi*z**2/2)/pi
    
    # 断言：验证 fresnels(z) 重写为 meijerg 函数的形式
    assert fresnels(z).rewrite(meijerg) == sqrt(2)*pi*z**Rational(9, 4) * \
        meijerg(((), (1,)), ((Rational(3, 4),),
        (Rational(1, 4), 0)), -pi**2*z**4/16)/(2*(-z)**Rational(3, 4)*(z**2)**Rational(3, 4))
    
    # 断言：验证 fresnelc(0) 的值是否为 S.Zero
    assert fresnelc(0) is S.Zero
    
    # 断言：验证 fresnelc(oo) 的值是否为 S.Half
    assert fresnelc(oo) == S.Half
    
    # 断言：验证 fresnelc(-oo) 的值是否为 Rational(-1, 2)
    assert fresnelc(-oo) == Rational(-1, 2)
    
    # 断言：验证 fresnelc(I*oo) 的值是否为 I*S.Half
    assert fresnelc(I*oo) == I*S.Half
    
    # 断言：验证 unchanged 函数对 fresnelc(z) 的作用
    assert unchanged(fresnelc, z)
    
    # 断言：验证 fresnelc(-z) 和 -fresnelc(z) 的关系
    assert fresnelc(-z) == -fresnelc(z)
    
    # 断言：验证 fresnelc(I*z) 和 I*fresnelc(z) 的关系
    assert fresnelc(I*z) == I*fresnelc(z)
    
    # 断言：验证 fresnelc(-I*z) 和 -I*fresnelc(z) 的关系
    assert fresnelc(-I*z) == -I*fresnelc(z)
    
    # 断言：验证 fresnelc(z) 的共轭是否等于 fresnelc(z) 的共轭
    assert conjugate(fresnelc(z)) == fresnelc(conjugate(z))
    
    # 断言：验证对 fresnelc(z) 关于 z 的导数
    assert fresnelc(z).diff(z) == cos(pi*z**2/2)
    
    # 断言：验证 fresnelc(z) 重写为 erf 函数的形式
    assert fresnelc(z).rewrite(erf) == (S.One - I)/4 * (
        erf((S.One + I)/2*sqrt(pi)*z) + I*erf((S.One - I)/2*sqrt(pi)*z))
    
    # 断言：验证 fresnelc(z) 重写为 hyper 函数的形式
    assert fresnelc(z).rewrite(hyper) == \
        z * hyper([Rational(1, 4)], [S.Half, Rational(5, 4)], -pi**2*z**4/16)
    
    # 断言：验证 fresnelc(w) 是否是扩展实数
    assert fresnelc(w).is_extended_real is True
    
    # 断言：验证 fresnelc(z) 的实部和虚部分解
    assert fresnelc(z).as_real_imag() == \
        (fresnelc(re(z) - I*im(z))/2 + fresnelc(re(z) + I*im(z))/2,
         -I*(-fresnelc(re(z) - I*im(z)) + fresnelc(re(z) + I*im(z)))/2)
    
    # 断言：验证 fresnelc(z) 的实部和虚部分解（浅层）
    assert fresnelc(z).as_real_imag(deep=False) == \
        (fresnelc(re(z) - I*im(z))/2 + fresnelc(re(z) + I*im(z))/2,
         -I*(-fresnelc(re(z) - I*im(z)) + fresnelc(re(z) + I*im(z)))/2)
    
    # 断言：验证 fresnelc(2 + 3*I) 的实部和虚部分解
    assert fresnelc(2 + 3*I).as_real_imag() == (
        fresnelc(2 - 3*I)/2 + fresnelc(2 + 3*I)/2,
         -I*(fresnelc(2 + 3*I) - fresnelc(2 - 3*I))/2
    )
    
    # 断言：验证对 fresnelc(z) 进行积分展开的结果
    assert expand_func(integrate(fresnelc(z), z)) == \
        z*fresnelc(z) - sin(pi*z**2/2)/pi
    
    # 断言：验证 fresnelc(z) 重写为 meijerg 函数的形式
    assert fresnelc(z).rewrite(meijerg) == sqrt(2)*pi*z**Rational(3, 4) * \
        meijerg(((), (1,)), ((Rational(1, 4),),
        (Rational(3, 4), 0)), -pi**2*z**4/16)/(2*(-z)**Rational(1, 4)*(z**2)**Rational(1, 4))
    
    # 导入数值验证函数
    from sympy.core.random import verify_numerically
    
    # 验证 re(fresnels(z)) 是否数值上等于 fresnels(z) 的实部
    verify_numerically(re(fresnels(z)), fresnels(z).as_real_imag()[0], z)
    
    # 验证 im(fresnels(z)) 是否数值上等于 fresnels(z) 的虚部
    verify_numerically(im(fresnels(z)), fresnels(z).as_real_imag()[1], z)
    
    # 验证 fresnels(z) 是否数值上等于其 hyper 函数重写形式
    verify_numerically(fresnels(z), fresnels(z).rewrite(hyper), z)
    
    # 验证 fresnels(z) 是否数值上等于其 meijerg 函数重写形式
    verify_numerically(fresnels(z), fresnels(z).rewrite(meijerg), z)
    # 对于第一个函数调用，验证数值上的正确性：re(fresnelc(z)) 应该等于 fresnelc(z) 的实部，针对给定的 z 进行验证
    verify_numerically(re(fresnelc(z)), fresnelc(z).as_real_imag()[0], z)
    
    # 对于第二个函数调用，验证数值上的正确性：im(fresnelc(z)) 应该等于 fresnelc(z) 的虚部，针对给定的 z 进行验证
    verify_numerically(im(fresnelc(z)), fresnelc(z).as_real_imag()[1], z)
    
    # 对于第三个函数调用，验证数值上的正确性：fresnelc(z) 应该等于用超几何函数的形式重写后的 fresnelc(z)，针对给定的 z 进行验证
    verify_numerically(fresnelc(z), fresnelc(z).rewrite(hyper), z)
    
    # 对于第四个函数调用，验证数值上的正确性：fresnelc(z) 应该等于用梅杰尔函数的形式重写后的 fresnelc(z)，针对给定的 z 进行验证
    verify_numerically(fresnelc(z), fresnelc(z).rewrite(meijerg), z)

    # 确保 fresnels(z) 的二阶导数调用会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: fresnels(z).fdiff(2))
    
    # 确保 fresnelc(z) 的二阶导数调用会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: fresnelc(z).fdiff(2))

    # 断言：fresnels(x) 的贝塞尔级数展开中，指数为 -1 的项应该是零
    assert fresnels(x).taylor_term(-1, x) is S.Zero
    
    # 断言：fresnelc(x) 的贝塞尔级数展开中，指数为 -1 的项应该是零
    assert fresnelc(x).taylor_term(-1, x) is S.Zero
    
    # 断言：fresnelc(x) 的贝塞尔级数展开中，指数为 1 的项应该等于 -pi^2 * x^5 / 40
    assert fresnelc(x).taylor_term(1, x) == -pi**2*x**5/40
def test_fresnel_series():
    # 断言：调用 fresnelc(z) 的级数展开，应该等于给定的表达式
    assert fresnelc(z).series(z, n=15) == \
        z - pi**2*z**5/40 + pi**4*z**9/3456 - pi**6*z**13/599040 + O(z**15)

    # issues 6510, 10102
    # 计算 fresnels(z) 的级数展开，并与 fs 比较
    fs = (S.Half - sin(pi*z**2/2)/(pi**2*z**3)
        + (-1/(pi*z) + 3/(pi**3*z**5))*cos(pi*z**2/2))
    # 计算 fresnelc(z) 的级数展开，并与 fc 比较
    fc = (S.Half - cos(pi*z**2/2)/(pi**2*z**3)
        + (1/(pi*z) - 3/(pi**3*z**5))*sin(pi*z**2/2))
    assert fresnels(z).series(z, oo) == fs + O(z**(-6), (z, oo))
    assert fresnelc(z).series(z, oo) == fc + O(z**(-6), (z, oo))
    # 检查在负无穷处的级数展开是否为高阶无穷小
    assert (fresnels(z).series(z, -oo) + fs.subs(z, -z)).expand().is_Order
    assert (fresnelc(z).series(z, -oo) + fc.subs(z, -z)).expand().is_Order
    # 检查 1/z 处的级数展开与 fs, fc 在 z=1/z 处的差是否为高阶无穷小
    assert (fresnels(1/z).series(z) - fs.subs(z, 1/z)).expand().is_Order
    assert (fresnelc(1/z).series(z) - fc.subs(z, 1/z)).expand().is_Order
    # 检查 2*fresnels(3*z) 的级数展开是否与 2*fs 在 z=oo 处的级数展开的差为高阶无穷小
    assert ((2*fresnels(3*z)).series(z, oo) - 2*fs.subs(z, 3*z)).expand().is_Order
    # 检查 3*fresnelc(2*z) 的级数展开是否与 3*fc 在 z=oo 处的级数展开的差为高阶无穷小
    assert ((3*fresnelc(2*z)).series(z, oo) - 3*fc.subs(z, 2*z)).expand().is_Order


def test_integral_rewrites(): #issues 26134, 26144, 26306
    # 使用 rewrite 方法将 expint(n, x) 重写为积分形式，并与给定的积分表达式比较
    assert expint(n, x).rewrite(Integral).dummy_eq(Integral(t**-n * exp(-t*x), (t, 1, oo)))
    # 使用 rewrite 方法将 Si(x) 重写为积分形式，并与给定的积分表达式比较
    assert Si(x).rewrite(Integral).dummy_eq(Integral(sinc(t), (t, 0, x)))
    # 使用 rewrite 方法将 Ci(x) 重写为积分形式，并与给定的积分表达式比较
    assert Ci(x).rewrite(Integral).dummy_eq(log(x) - Integral((1 - cos(t))/t, (t, 0, x)) + EulerGamma)
    # 使用 rewrite 方法将 fresnels(x) 重写为积分形式，并与给定的积分表达式比较
    assert fresnels(x).rewrite(Integral).dummy_eq(Integral(sin(pi*t**2/2), (t, 0, x)))
    # 使用 rewrite 方法将 fresnelc(x) 重写为积分形式，并与给定的积分表达式比较
    assert fresnelc(x).rewrite(Integral).dummy_eq(Integral(cos(pi*t**2/2), (t, 0, x)))
    # 使用 rewrite 方法将 Ei(x) 重写为积分形式，并与给定的积分表达式比较
    assert Ei(x).rewrite(Integral).dummy_eq(Integral(exp(t)/t, (t, -oo, x)))
    # 检查 fresnels(x) 的导数是否与使用 rewrite 方法得到的积分形式的导数相等
    assert fresnels(x).diff(x) == fresnels(x).rewrite(Integral).diff(x)
    # 检查 fresnelc(x) 的导数是否与使用 rewrite 方法得到的积分形式的导数相等
    assert fresnelc(x).diff(x) == fresnelc(x).rewrite(Integral).diff(x)
```