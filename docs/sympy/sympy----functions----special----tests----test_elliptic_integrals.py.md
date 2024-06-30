# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_elliptic_integrals.py`

```
from sympy.core.numbers import (I, Rational, oo, pi, zoo)  # 导入 sympy 核心模块中的数学常数和对象
from sympy.core.singleton import S  # 导入 sympy 核心模块中的单例对象
from sympy.core.symbol import (Dummy, Symbol)  # 导入 sympy 核心模块中的符号相关对象
from sympy.functions.elementary.hyperbolic import atanh  # 导入双曲函数中的反双曲正切函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入一般函数中的平方根函数
from sympy.functions.elementary.trigonometric import (sin, tan)  # 导入三角函数中的正弦和正切函数
from sympy.functions.special.gamma_functions import gamma  # 导入 gamma 函数
from sympy.functions.special.hyper import (hyper, meijerg)  # 导入超函相关函数
from sympy.integrals.integrals import Integral  # 导入积分相关函数
from sympy.series.order import O  # 导入级数展开相关对象
from sympy.functions.special.elliptic_integrals import (elliptic_k as K,
    elliptic_f as F, elliptic_e as E, elliptic_pi as P)  # 导入椭圆积分相关函数，重命名为 K, F, E, P
from sympy.core.random import (test_derivative_numerically as td,
                                      random_complex_number as randcplx,
                                      verify_numerically as tn)  # 导入随机相关函数和数值验证函数
from sympy.abc import z, m, n  # 导入符号 z, m, n

i = Symbol('i', integer=True)  # 定义整数符号 i
j = Symbol('k', integer=True, positive=True)  # 定义正整数符号 j
t = Dummy('t')  # 定义虚拟符号 t

def test_K():  # 定义测试函数 test_K
    assert K(0) == pi/2  # 断言 K(0) 的值为 pi/2
    assert K(S.Half) == 8*pi**Rational(3, 2)/gamma(Rational(-1, 4))**2  # 断言 K(1/2) 的值为 8*pi**(3/2) / gamma(-1/4)**2
    assert K(1) is zoo  # 断言 K(1) 是无穷大
    assert K(-1) == gamma(Rational(1, 4))**2/(4*sqrt(2*pi))  # 断言 K(-1) 的值为 gamma(1/4)**2 / (4 * sqrt(2*pi))
    assert K(oo) == 0  # 断言 K(oo) 的值为 0
    assert K(-oo) == 0  # 断言 K(-oo) 的值为 0
    assert K(I*oo) == 0  # 断言 K(I*oo) 的值为 0
    assert K(-I*oo) == 0  # 断言 K(-I*oo) 的值为 0
    assert K(zoo) == 0  # 断言 K(zoo) 的值为 0

    assert K(z).diff(z) == (E(z) - (1 - z)*K(z))/(2*z*(1 - z))  # 断言 K(z) 对 z 的导数等于 (E(z) - (1 - z)*K(z)) / (2*z*(1 - z))
    assert td(K(z), z)  # 使用数值验证函数验证 K(z) 对 z 的导数

    zi = Symbol('z', real=False)  # 定义一个非实数符号 zi
    assert K(zi).conjugate() == K(zi.conjugate())  # 断言 K(zi) 的共轭等于 K(zi 的共轭)
    zr = Symbol('z', negative=True)  # 定义一个负数符号 zr
    assert K(zr).conjugate() == K(zr)  # 断言 K(zr) 的共轭等于 K(zr)

    assert K(z).rewrite(hyper) == \
        (pi/2)*hyper((S.Half, S.Half), (S.One,), z)  # 断言 K(z) 的超函展开形式
    assert tn(K(z), (pi/2)*hyper((S.Half, S.Half), (S.One,), z))  # 使用数值验证函数验证 K(z) 的超函展开形式
    assert K(z).rewrite(meijerg) == \
        meijerg(((S.Half, S.Half), []), ((S.Zero,), (S.Zero,)), -z)/2  # 断言 K(z) 的 Meijer G 函数展开形式
    assert tn(K(z), meijerg(((S.Half, S.Half), []), ((S.Zero,), (S.Zero,)), -z)/2)  # 使用数值验证函数验证 K(z) 的 Meijer G 函数展开形式

    assert K(z).series(z) == pi/2 + pi*z/8 + 9*pi*z**2/128 + \
        25*pi*z**3/512 + 1225*pi*z**4/32768 + 3969*pi*z**5/131072 + O(z**6)  # 断言 K(z) 的级数展开形式

    assert K(m).rewrite(Integral).dummy_eq(
        Integral(1/sqrt(1 - m*sin(t)**2), (t, 0, pi/2)))  # 断言 K(m) 的积分形式与指定积分相等

def test_F():  # 定义测试函数 test_F
    assert F(z, 0) == z  # 断言 F(z, 0) 的值为 z
    assert F(0, m) == 0  # 断言 F(0, m) 的值为 0
    assert F(pi*i/2, m) == i*K(m)  # 断言 F(pi*i/2, m) 的值为 i*K(m)
    assert F(z, oo) == 0  # 断言 F(z, oo) 的值为 0
    assert F(z, -oo) == 0  # 断言 F(z, -oo) 的值为 0

    assert F(-z, m) == -F(z, m)  # 断言 F(-z, m) 的值等于 -F(z, m)

    assert F(z, m).diff(z) == 1/sqrt(1 - m*sin(z)**2)  # 断言 F(z, m) 对 z 的导数
    assert F(z, m).diff(m) == E(z, m)/(2*m*(1 - m)) - F(z, m)/(2*m) - \
        sin(2*z)/(4*(1 - m)*sqrt(1 - m*sin(z)**2))  # 断言 F(z, m) 对 m 的导数
    r = randcplx()
    assert td(F(z, r), z)  # 使用数值验证函数验证 F(z, r) 对 z 的导数
    assert td(F(r, m), m)  # 使用数值验证函数验证 F(r, m) 对 m 的导数

    mi = Symbol('m', real=False)  # 定义一个非实数符号 mi
    assert F(z, mi).conjugate() == F(z.conjugate(), mi.conjugate())  # 断言 F(z, mi) 的共轭等于 F(z 的共轭, mi 的共轭)
    mr = Symbol('m', negative=True)  # 定义一个负数符号 mr
    assert F(z, mr).conjugate() == F(z.conjugate(), mr)  # 断言 F(z, mr) 的共轭等于 F(z 的共轭, mr)

    assert F(z, m).series(z) == \
        z + z**5*(3*m**2/40 - m/30) + m*z**3/6 + O(z**6)  # 断言 F(z, m) 的级数展开形式

    assert F(z, m).rewrite(Integral).dummy_eq(
        Integral(1/sqrt(1 - m*sin(t)**2), (t, 0, z)))  # 断言 F(z, m) 的积分形式与指定积分相等
    # 断言：E(i*pi/2, m) 应该等于 i*E(m)
    assert E(i*pi/2, m) == i*E(m)
    # 断言：E(z, oo) 应该是无穷大 zoo
    assert E(z, oo) is zoo
    # 断言：E(z, -oo) 应该是无穷大 zoo
    assert E(z, -oo) is zoo
    # 断言：E(0) 应该等于 π/2
    assert E(0) == pi/2
    # 断言：E(1) 应该等于 1
    assert E(1) == 1
    # 断言：E(oo) 应该等于 I*oo (虚数单位乘以无穷大)
    assert E(oo) == I*oo
    # 断言：E(-oo) 应该是无穷大 oo
    assert E(-oo) is oo
    # 断言：E(zoo) 应该是无穷大 zoo
    assert E(zoo) is zoo

    # 断言：E(-z, m) 应该等于 -E(z, m)
    assert E(-z, m) == -E(z, m)

    # 断言：E(z, m) 对 z 的偏导数应该等于 sqrt(1 - m*sin(z)**2)
    assert E(z, m).diff(z) == sqrt(1 - m*sin(z)**2)
    # 断言：E(z, m) 对 m 的偏导数应该等于 (E(z, m) - F(z, m))/(2*m)
    assert E(z, m).diff(m) == (E(z, m) - F(z, m))/(2*m)
    # 断言：E(z) 对 z 的偏导数应该等于 (E(z) - K(z))/(2*z)
    assert E(z).diff(z) == (E(z) - K(z))/(2*z)
    # 生成一个随机复数 r
    r = randcplx()
    # 断言：E(r, m) 对 m 的总导数
    assert td(E(r, m), m)
    # 断言：E(z, r) 对 z 的总导数
    assert td(E(z, r), z)
    # 断言：E(z) 对 z 的总导数
    assert td(E(z), z)

    # 创建一个非实数符号 mi
    mi = Symbol('m', real=False)
    # 断言：E(z, mi) 的共轭应该等于 E(z.conjugate(), mi.conjugate())
    assert E(z, mi).conjugate() == E(z.conjugate(), mi.conjugate())
    # 断言：E(mi) 的共轭应该等于 E(mi.conjugate())
    assert E(mi).conjugate() == E(mi.conjugate())
    # 创建一个负值的符号 mr
    mr = Symbol('m', negative=True)
    # 断言：E(z, mr) 的共轭应该等于 E(z.conjugate(), mr)
    assert E(z, mr).conjugate() == E(z.conjugate(), mr)
    # 断言：E(mr) 的共轭应该等于 E(mr)
    assert E(mr).conjugate() == E(mr)

    # 断言：E(z) 重写为超几何函数 hyper
    assert E(z).rewrite(hyper) == (pi/2)*hyper((Rational(-1, 2), S.Half), (S.One,), z)
    # 断言：E(z) 与 (pi/2)*hyper((Rational(-1, 2), S.Half), (S.One,), z) 相等
    assert tn(E(z), (pi/2)*hyper((Rational(-1, 2), S.Half), (S.One,), z))
    # 断言：E(z) 重写为 Meijer G 函数 meijerg
    assert E(z).rewrite(meijerg) == \
        -meijerg(((S.Half, Rational(3, 2)), []), ((S.Zero,), (S.Zero,)), -z)/4
    # 断言：E(z) 与 -meijerg(((S.Half, Rational(3, 2)), []), ((S.Zero,), (S.Zero,)), -z)/4 相等
    assert tn(E(z), -meijerg(((S.Half, Rational(3, 2)), []), ((S.Zero,), (S.Zero,)), -z)/4)

    # 断言：E(z, m) 关于 z 的级数展开
    assert E(z, m).series(z) == \
        z + z**5*(-m**2/40 + m/30) - m*z**3/6 + O(z**6)
    # 断言：E(z) 关于 z 的级数展开
    assert E(z).series(z) == pi/2 - pi*z/8 - 3*pi*z**2/128 - \
        5*pi*z**3/512 - 175*pi*z**4/32768 - 441*pi*z**5/131072 + O(z**6)
    # 断言：E(4*z/(z+1)) 关于 z 的级数展开
    assert E(4*z/(z+1)).series(z) == \
        pi/2 - pi*z/2 + pi*z**2/8 - 3*pi*z**3/8 - 15*pi*z**4/128 - 93*pi*z**5/128 + O(z**6)

    # 断言：E(z, m) 重写为积分形式后应该等价于积分表达式 Integral(sqrt(1 - m*sin(t)**2), (t, 0, z))
    assert E(z, m).rewrite(Integral).dummy_eq(
        Integral(sqrt(1 - m*sin(t)**2), (t, 0, z)))
    # 断言：E(m) 重写为积分形式后应该等价于积分表达式 Integral(sqrt(1 - m*sin(t)**2), (t, 0, pi/2))
    assert E(m).rewrite(Integral).dummy_eq(
        Integral(sqrt(1 - m*sin(t)**2), (t, 0, pi/2)))
def test_P():
    # 测试辅助函数 P 的各种情况

    # 测试 P 函数在特定参数下的返回值是否等于 F 函数在相同参数下的返回值
    assert P(0, z, m) == F(z, m)

    # 测试 P 函数在特定参数下的返回值是否满足复杂的数学表达式
    assert P(1, z, m) == F(z, m) + \
        (sqrt(1 - m*sin(z)**2)*tan(z) - E(z, m))/(1 - m)

    # 测试 P 函数在特定参数下是否满足与其自身乘以虚数 i 相等的关系
    assert P(n, i*pi/2, m) == i*P(n, m)

    # 测试 P 函数在特定参数下是否满足对 m 为 0 时的特殊情况
    assert P(n, z, 0) == atanh(sqrt(n - 1)*tan(z))/sqrt(n - 1)

    # 测试 P 函数在特定参数下是否满足复杂的数学表达式
    assert P(n, z, n) == F(z, n) - P(1, z, n) + tan(z)/sqrt(1 - n*sin(z)**2)

    # 测试 P 函数在特定参数下是否返回预期的数值
    assert P(oo, z, m) == 0
    assert P(-oo, z, m) == 0
    assert P(n, z, oo) == 0
    assert P(n, z, -oo) == 0

    # 测试 P 函数在特定参数下是否返回预期的数值
    assert P(0, m) == K(m)

    # 测试 P 函数在特定参数下是否返回预期的特殊值
    assert P(1, m) is zoo

    # 测试 P 函数在特定参数下是否返回预期的数值
    assert P(n, 0) == pi/(2*sqrt(1 - n))

    # 测试 P 函数在特定参数下是否返回预期的特殊值
    assert P(2, 1) is -oo
    assert P(-1, 1) is oo

    # 测试 P 函数在特定参数下是否返回预期的数值
    assert P(n, n) == E(n)/(1 - n)

    # 测试 P 函数的对称性质
    assert P(n, -z, m) == -P(n, z, m)

    # 测试 P 函数在复数和共轭的参数下的性质
    ni, mi = Symbol('n', real=False), Symbol('m', real=False)
    assert P(ni, z, mi).conjugate() == \
        P(ni.conjugate(), z.conjugate(), mi.conjugate())
    nr, mr = Symbol('n', negative=True), \
        Symbol('m', negative=True)
    assert P(nr, z, mr).conjugate() == P(nr, z.conjugate(), mr)
    assert P(n, m).conjugate() == P(n.conjugate(), m.conjugate())

    # 测试 P 函数对 n, z, m 的偏导数
    assert P(n, z, m).diff(n) == (E(z, m) + (m - n)*F(z, m)/n +
        (n**2 - m)*P(n, z, m)/n - n*sqrt(1 -
            m*sin(z)**2)*sin(2*z)/(2*(1 - n*sin(z)**2)))/(2*(m - n)*(n - 1))
    assert P(n, z, m).diff(z) == 1/(sqrt(1 - m*sin(z)**2)*(1 - n*sin(z)**2))
    assert P(n, z, m).diff(m) == (E(z, m)/(m - 1) + P(n, z, m) -
        m*sin(2*z)/(2*(m - 1)*sqrt(1 - m*sin(z)**2)))/(2*(n - m))

    # 测试 P 函数对 n, m 的偏导数
    assert P(n, m).diff(n) == (E(m) + (m - n)*K(m)/n +
        (n**2 - m)*P(n, m)/n)/(2*(m - n)*(n - 1))
    assert P(n, m).diff(m) == (E(m)/(m - 1) + P(n, m))/(2*(n - m))

    # 以下测试因以下问题而失败：
    # https://github.com/fredrik-johansson/mpmath/issues/571#issuecomment-777201962
    # https://github.com/sympy/sympy/issues/20933#issuecomment-777080385
    #
    # rx, ry = randcplx(), randcplx()
    # assert td(P(n, rx, ry), n)
    # assert td(P(rx, z, ry), z)
    # assert td(P(rx, ry, m), m)

    # 测试 P 函数在 z 展开的级数
    assert P(n, z, m).series(z) == z + z**3*(m/6 + n/3) + \
        z**5*(3*m**2/40 + m*n/10 - m/30 + n**2/5 - n/15) + O(z**6)

    # 测试 P 函数在 z 和 m 重写为积分形式的等价性
    assert P(n, z, m).rewrite(Integral).dummy_eq(
        Integral(1/((1 - n*sin(t)**2)*sqrt(1 - m*sin(t)**2)), (t, 0, z)))
    assert P(n, m).rewrite(Integral).dummy_eq(
        Integral(1/((1 - n*sin(t)**2)*sqrt(1 - m*sin(t)**2)), (t, 0, pi/2)))
```