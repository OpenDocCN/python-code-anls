# `D:\src\scipysrc\sympy\sympy\functions\combinatorial\tests\test_comb_numbers.py`

```
import string  # 导入字符串模块

from sympy.concrete.products import Product  # 导入 Product 符号计算模块
from sympy.concrete.summations import Sum  # 导入 Sum 符号计算模块
from sympy.core.function import (diff, expand_func)  # 导入函数求导和函数展开模块
from sympy.core import (EulerGamma, TribonacciConstant)  # 导入常数 EulerGamma 和 TribonacciConstant
from sympy.core.numbers import (Float, I, Rational, oo, pi)  # 导入数值常数模块
from sympy.core.singleton import S  # 导入 SymPy 单例模块
from sympy.core.symbol import (Dummy, Symbol, symbols)  # 导入符号和符号变量模块
from sympy.functions.combinatorial.numbers import carmichael  # 导入卡迈克尔函数模块
from sympy.functions.elementary.complexes import (im, re)  # 导入复数函数模块
from sympy.functions.elementary.integers import floor  # 导入向下取整函数模块
from sympy.polys.polytools import cancel  # 导入多项式工具模块
from sympy.series.limits import limit, Limit  # 导入极限计算模块
from sympy.series.order import O  # 导入阶数模块
from sympy.functions import (  # 导入各种特殊函数模块
    bernoulli, harmonic, bell, fibonacci, tribonacci, lucas, euler, catalan,
    genocchi, andre, partition, divisor_sigma, udivisor_sigma, legendre_symbol,
    jacobi_symbol, kronecker_symbol, mobius,
    primenu, primeomega, totient, reduced_totient, primepi,
    motzkin, binomial, gamma, sqrt, cbrt, hyper, log, digamma,
    trigamma, polygamma, factorial, sin, cos, cot, polylog, zeta, dirichlet_eta)
from sympy.functions.combinatorial.numbers import _nT  # 导入组合数论函数模块
from sympy.ntheory.factor_ import factorint  # 导入因数分解模块

from sympy.core.expr import unchanged  # 导入不变对象模块
from sympy.core.numbers import GoldenRatio, Integer  # 导入黄金比例和整数模块

from sympy.testing.pytest import raises, nocache_fail, warns_deprecated_sympy  # 导入测试模块
from sympy.abc import x  # 导入符号 x

# 定义测试卡迈克尔函数的函数
def test_carmichael():
    # 使用 warns_deprecated_sympy 上下文管理器检测是否有 Sympy 弃用警告
    with warns_deprecated_sympy():
        # 断言卡迈克尔数 2821 不是素数
        assert carmichael.is_prime(2821) == False

# 测试伯努利数的计算
def test_bernoulli():
    # 断言前几个伯努利数的值
    assert bernoulli(0) == 1
    assert bernoulli(1) == Rational(1, 2)
    assert bernoulli(2) == Rational(1, 6)
    assert bernoulli(3) == 0
    assert bernoulli(4) == Rational(-1, 30)
    assert bernoulli(5) == 0
    assert bernoulli(6) == Rational(1, 42)
    assert bernoulli(7) == 0
    assert bernoulli(8) == Rational(-1, 30)
    assert bernoulli(10) == Rational(5, 66)
    assert bernoulli(1000001) == 0

    # 断言伯努利多项式的计算
    assert bernoulli(0, x) == 1
    assert bernoulli(1, x) == x - S.Half
    assert bernoulli(2, x) == x**2 - x + Rational(1, 6)
    assert bernoulli(3, x) == x**3 - (3*x**2)/2 + x/2

    # 使用 mpmath 计算伯努利数的高精度值并断言
    b = bernoulli(1000)
    assert b.p % 10**10 == 7950421099
    assert b.q == 342999030

    # 使用 evaluate=False 计算大数伯努利数并评估其字符串表示
    b = bernoulli(10**6, evaluate=False).evalf()
    assert str(b) == '-2.23799235765713e+4767529'

    # 测试符号伯努利数的性质
    l = Symbol('l', integer=True)
    m = Symbol('m', integer=True, nonnegative=True)
    n = Symbol('n', integer=True, positive=True)
    assert isinstance(bernoulli(2 * l + 1), bernoulli)
    assert isinstance(bernoulli(2 * m + 1), bernoulli)
    assert bernoulli(2 * n + 1) == 0

    # 断言浮点数伯努利数的计算结果
    assert str(bernoulli(0.0, 2.3).evalf(n=10)) == '1.000000000'
    assert str(bernoulli(1.0).evalf(n=10)) == '0.5000000000'
    assert str(bernoulli(1.2).evalf(n=10)) == '0.4195995367'
    assert str(bernoulli(1.2, 0.8).evalf(n=10)) == '0.2144830348'
    # 断言：验证伯努利函数的计算结果是否与预期字符串相等
    assert str(bernoulli(1.2, -0.8).evalf(n=10)) == '-1.158865646 - 0.6745558744*I'
    
    # 断言：验证伯努利函数的计算结果是否与预期字符串相等
    assert str(bernoulli(3.0, 1j).evalf(n=10)) == '1.5 - 0.5*I'
    
    # 断言：验证伯努利函数的计算结果是否与预期字符串相等
    assert str(bernoulli(I).evalf(n=10)) == '0.9268485643 - 0.5821580598*I'
    
    # 断言：验证伯努利函数的计算结果是否与预期字符串相等
    assert str(bernoulli(I, I).evalf(n=10)) == '0.1267792071 + 0.01947413152*I'
    
    # 断言：验证伯努利函数在参数为变量 x 时的数值计算结果是否等于其在变量 x 上的伯努利函数
    assert bernoulli(x).evalf() == bernoulli(x)
# 测试 Bernoulli 数的重写方法
def test_bernoulli_rewrite():
    # 从 sympy 库中导入 Piecewise 类
    from sympy.functions.elementary.piecewise import Piecewise
    # 定义符号 n，为整数且非负
    n = Symbol('n', integer=True, nonnegative=True)

    # 断言语句：验证 Bernoulli 数在重写为 zeta 函数后的计算结果
    assert bernoulli(-1).rewrite(zeta) == pi**2/6
    # 断言语句：验证 Bernoulli 数在重写为 zeta 函数后的计算结果
    assert bernoulli(-2).rewrite(zeta) == 2*zeta(3)
    # 断言语句：验证 Bernoulli 数在重写为 zeta 函数后，不含 harmonic 函数
    assert not bernoulli(n, -3).rewrite(zeta).has(harmonic)
    # 断言语句：验证 Bernoulli 数在重写为 zeta 函数后的计算结果
    assert bernoulli(-4, x).rewrite(zeta) == 4*zeta(5, x)
    # 断言语句：验证 Bernoulli 数在重写为 zeta 函数后返回的对象类型为 Piecewise
    assert isinstance(bernoulli(n, x).rewrite(zeta), Piecewise)
    # 断言语句：验证 Bernoulli 数在重写为 zeta 函数后的计算结果
    assert bernoulli(n+1, x).rewrite(zeta) == -(n+1) * zeta(-n, x)


# 测试 Fibonacci 数列的计算
def test_fibonacci():
    # 断言语句：验证 Fibonacci 数列的前几项
    assert [fibonacci(n) for n in range(-3, 5)] == [2, -1, 1, 0, 1, 1, 2, 3]
    # 断言语句：验证 Fibonacci 数列第 100 项的计算结果
    assert fibonacci(100) == 354224848179261915075
    # 断言语句：验证 Lucas 数列的前几项
    assert [lucas(n) for n in range(-3, 5)] == [-4, 3, -1, 2, 1, 3, 4, 7]
    # 断言语句：验证 Lucas 数列第 100 项的计算结果
    assert lucas(100) == 792070839848372253127

    # 断言语句：验证 Fibonacci 数列的前几项关于变量 x 的计算结果
    assert fibonacci(1, x) == 1
    assert fibonacci(2, x) == x
    assert fibonacci(3, x) == x**2 + 1
    assert fibonacci(4, x) == x**3 + 2*x

    # 断言语句：测试问题 #8800
    n = Dummy('n')
    # 断言语句：验证 Fibonacci 数列在 n 趋向无穷大时的极限
    assert fibonacci(n).limit(n, S.Infinity) is S.Infinity
    # 断言语句：验证 Lucas 数列在 n 趋向无穷大时的极限
    assert lucas(n).limit(n, S.Infinity) is S.Infinity

    # 断言语句：验证 Fibonacci 数列重写为 sqrt 函数的结果
    assert fibonacci(n).rewrite(sqrt) == \
        2**(-n)*sqrt(5)*((1 + sqrt(5))**n - (-sqrt(5) + 1)**n) / 5
    # 断言语句：验证 Fibonacci 数列重写为 sqrt 函数后，代入 n=10 展开的结果等于 fibonacci(10)
    assert fibonacci(n).rewrite(sqrt).subs(n, 10).expand() == fibonacci(10)
    # 断言语句：验证 Fibonacci 数列重写为 GoldenRatio 函数后，代入 n=10 的数值计算结果等于 fibonacci(10)
    assert fibonacci(n).rewrite(GoldenRatio).subs(n,10).evalf() == \
        Float(fibonacci(10))
    # 断言语句：验证 Lucas 数列重写为 sqrt 函数的结果
    assert lucas(n).rewrite(sqrt) == \
        (fibonacci(n-1).rewrite(sqrt) + fibonacci(n+1).rewrite(sqrt)).simplify()
    # 断言语句：验证 Lucas 数列重写为 sqrt 函数后，代入 n=10 展开的结果等于 lucas(10)
    assert lucas(n).rewrite(sqrt).subs(n, 10).expand() == lucas(10)
    # 断言语句：验证在 n 为负数时调用 fibonacci(-3, x) 会引发 ValueError 异常
    raises(ValueError, lambda: fibonacci(-3, x))


# 测试 Tribonacci 数列的计算
def test_tribonacci():
    # 断言语句：验证 Tribonacci 数列的前几项
    assert [tribonacci(n) for n in range(8)] == [0, 1, 1, 2, 4, 7, 13, 24]
    # 断言语句：验证 Tribonacci 数列第 100 项的计算结果
    assert tribonacci(100) == 98079530178586034536500564

    # 断言语句：验证 Tribonacci 数列的前几项关于变量 x 的计算结果
    assert tribonacci(0, x) == 0
    assert tribonacci(1, x) == 1
    assert tribonacci(2, x) == x**2
    assert tribonacci(3, x) == x**4 + x
    assert tribonacci(4, x) == x**6 + 2*x**3 + 1
    assert tribonacci(5, x) == x**8 + 3*x**5 + 3*x**2

    n = Dummy('n')
    # 断言语句：验证 Tribonacci 数列在 n 趋向无穷大时的极限
    assert tribonacci(n).limit(n, S.Infinity) is S.Infinity

    # 定义常数 w、a、b、c
    w = (-1 + S.ImaginaryUnit * sqrt(3)) / 2
    a = (1 + cbrt(19 + 3*sqrt(33)) + cbrt(19 - 3*sqrt(33))) / 3
    b = (1 + w*cbrt(19 + 3*sqrt(33)) + w**2*cbrt(19 - 3*sqrt(33))) / 3
    c = (1 + w**2*cbrt(19 + 3*sqrt(33)) + w*cbrt(19 - 3*sqrt(33))) / 3
    # 断言语句：验证 Tribonacci 数列重写为 sqrt 函数的结果
    assert tribonacci(n).rewrite(sqrt) == \
      (a**(n + 1)/((a - b)*(a - c))
      + b**(n + 1)/((b - a)*(b - c))
      + c**(n + 1)/((c - a)*(c - b)))
    # 断言语句：验证 Tribonacci 数列重写为 sqrt 函数后，代入 n=4 简化的结果等于 tribonacci(4)
    assert tribonacci(n).rewrite(sqrt).subs(n, 4).simplify() == tribonacci(4)
    # 断言语句：验证 Tribonacci 数列重写为 GoldenRatio 函数后，代入 n=10 的数值计算结果等于 tribonacci(10)
    assert tribonacci(n).rewrite(GoldenRatio).subs(n,10).evalf() == \
        Float(tribonacci(10))
    # 断言语句：验证 Tribonacci 数列重写为 TribonacciConstant 常数的结果
    assert tribonacci(n).rewrite(TribonacciConstant) == floor(
            3*TribonacciConstant**n*(102*sqrt(33) + 586)**Rational(1, 3)/
            (-2*(102*sqrt(33) + 586)**Rational(1, 3) + 4 + (102*sqrt(33)
            + 586)**Rational(2, 3)) + S.Half)
    # 断言语句：验证在 n 为负数时调用 tribonacci(-1, x) 会引发 ValueError 异常
    raises(ValueError, lambda: tribonacci(-1, x))


@nocache_fail
def test_bell():
    # 断言：验证给定列表中前八个贝尔数的计算结果是否正确
    assert [bell(n) for n in range(8)] == [1, 1, 2, 5, 15, 52, 203, 877]

    # 断言：验证贝尔数 bell(0, x) 的计算结果是否为 1
    assert bell(0, x) == 1
    # 断言：验证贝尔数 bell(1, x) 的计算结果是否为 x
    assert bell(1, x) == x
    # 断言：验证贝尔数 bell(2, x) 的计算结果是否为 x^2 + x
    assert bell(2, x) == x**2 + x
    # 断言：验证贝尔数 bell(5, x) 的计算结果是否为 x^5 + 10*x^4 + 25*x^3 + 15*x^2 + x
    assert bell(5, x) == x**5 + 10*x**4 + 25*x**3 + 15*x**2 + x
    # 断言：验证贝尔数 bell(oo) 的计算结果是否为正无穷
    assert bell(oo) is S.Infinity
    # 断言：验证当输入无穷大时，贝尔数 bell(oo, x) 是否引发 ValueError 异常
    raises(ValueError, lambda: bell(oo, x))

    # 断言：验证当输入负数时，贝尔数 bell(-1) 是否引发 ValueError 异常
    raises(ValueError, lambda: bell(-1))
    # 断言：验证当输入 S.Half (分数 1/2) 时，贝尔数 bell(S.Half) 是否引发 ValueError 异常
    raises(ValueError, lambda: bell(S.Half))

    # 创建符号变量 x0 到 x5
    X = symbols('x:6')
    # 断言：验证带有给定符号变量的贝尔数 bell(6, 2, X[1:]) 的计算结果是否正确
    assert bell(6, 2, X[1:]) == 6*X[5]*X[1] + 15*X[4]*X[2] + 10*X[3]**2
    # 断言：验证带有给定符号变量的贝尔数 bell(6, 3, X[1:]) 的计算结果是否正确
    assert bell(6, 3, X[1:]) == 15*X[4]*X[1]**2 + 60*X[3]*X[2]*X[1] + 15*X[2]**3

    # 使用固定的 X 值重新进行验证
    X = (1, 10, 100, 1000, 10000)
    # 断言：验证带有给定常数序列 X 的贝尔数 bell(6, 2, X) 的计算结果是否正确
    assert bell(6, 2, X) == (6 + 15 + 10)*10000

    # 使用固定的 X 值重新进行验证
    X = (1, 2, 3, 3, 5)
    # 断言：验证带有给定常数序列 X 的贝尔数 bell(6, 2, X) 的计算结果是否正确
    assert bell(6, 2, X) == 6*5 + 15*3*2 + 10*3**2

    # 使用固定的 X 值重新进行验证
    X = (1, 2, 3, 5)
    # 断言：验证带有给定常数序列 X 的贝尔数 bell(6, 3, X) 的计算结果是否正确
    assert bell(6, 3, X) == 15*5 + 60*3*2 + 15*2**3

    # Dobinski's formula
    n = Symbol('n', integer=True, nonnegative=True)
    # 针对一些指定的整数进行验证，确保贝尔数的数值计算正确
    for i in [0, 2, 3, 7, 13, 42, 55]:
        # 断言：验证贝尔数 bell(i).evalf() 和 bell(n).rewrite(Sum).evalf(subs={n: i}) 的数值计算结果是否相等
        assert bell(i).evalf() == bell(n).rewrite(Sum).evalf(subs={n: i})

    m = Symbol("m")
    # 断言：验证贝尔数 bell(m).rewrite(Sum) 是否等于 bell(m)
    assert bell(m).rewrite(Sum) == bell(m)
    # 断言：验证贝尔数 bell(n, m).rewrite(Sum) 是否等于 bell(n, m)
    assert bell(n, m).rewrite(Sum) == bell(n, m)
    # issue 9184
    n = Dummy('n')
    # 断言：验证当 n 趋近无穷大时，贝尔数 bell(n).limit(n, S.Infinity) 的极限是否为正无穷
    assert bell(n).limit(n, S.Infinity) is S.Infinity
# 定义一个测试函数 test_harmonic，用于测试 harmonic 函数的各种输入情况
def test_harmonic():
    # 创建符号变量 n 和 m
    n = Symbol("n")
    m = Symbol("m")

    # 断言 harmonic(n, 0) 返回 n
    assert harmonic(n, 0) == n
    # 断言 harmonic(n).evalf() 等于 harmonic(n)
    assert harmonic(n).evalf() == harmonic(n)
    # 断言 harmonic(n, 1) 等于 harmonic(n)
    assert harmonic(n, 1) == harmonic(n)
    # 断言 harmonic(1, n) 等于 1
    assert harmonic(1, n) == 1

    # 下面是一系列断言语句，用于验证 harmonic 函数在不同参数下的计算结果
    assert harmonic(0, 1) == 0
    assert harmonic(1, 1) == 1
    assert harmonic(2, 1) == Rational(3, 2)
    assert harmonic(3, 1) == Rational(11, 6)
    assert harmonic(4, 1) == Rational(25, 12)
    assert harmonic(0, 2) == 0
    assert harmonic(1, 2) == 1
    assert harmonic(2, 2) == Rational(5, 4)
    assert harmonic(3, 2) == Rational(49, 36)
    assert harmonic(4, 2) == Rational(205, 144)
    assert harmonic(0, 3) == 0
    assert harmonic(1, 3) == 1
    assert harmonic(2, 3) == Rational(9, 8)
    assert harmonic(3, 3) == Rational(251, 216)
    assert harmonic(4, 3) == Rational(2035, 1728)

    # 断言 harmonic(oo, -1) 返回 S.NaN
    assert harmonic(oo, -1) is S.NaN
    # 断言 harmonic(oo, 0) 返回 oo (正无穷大)
    assert harmonic(oo, 0) is oo
    # 断言 harmonic(oo, S.Half) 返回 oo
    assert harmonic(oo, S.Half) is oo
    # 断言 harmonic(oo, 1) 返回 oo
    assert harmonic(oo, 1) is oo
    # 断言 harmonic(oo, 2) 等于 pi^2 / 6
    assert harmonic(oo, 2) == (pi**2)/6
    # 断言 harmonic(oo, 3) 等于 zeta(3) (黎曼 zeta 函数的特殊值)
    assert harmonic(oo, 3) == zeta(3)
    # 使用 Dummy 类创建一个符号变量 ip，满足 integer=True, positive=True
    ip = Dummy(integer=True, positive=True)
    # 如果 (1/ip <= 1) 为真，则执行以下断言
    if (1/ip <= 1) is True:
        # 断言 None，表明此处应删除 if-block 和下一行
        assert None, 'delete this if-block and the next line'
    # 使用 Dummy 类创建一个符号变量 ip，满足 even=True, positive=True
    ip = Dummy(even=True, positive=True)
    # 断言 harmonic(oo, 1/ip) 返回 oo
    assert harmonic(oo, 1/ip) is oo
    # 断言 harmonic(oo, 1 + ip) 等于 zeta(1 + ip)
    assert harmonic(oo, 1 + ip) is zeta(1 + ip)

    # 下面是一系列断言语句，用于验证 harmonic 函数在不同参数下的计算结果
    assert harmonic(0, m) == 0
    assert harmonic(-1, -1) == 0
    assert harmonic(-1, 0) == -1
    assert harmonic(-1, 1) is S.ComplexInfinity
    assert harmonic(-1, 2) is S.NaN
    assert harmonic(-3, -2) == -5
    assert harmonic(-3, -3) == 9
    # 计算调和序列 Heoo 对应的值
    Heoo = harmonic(ne + po/qo)
    # 计算调和序列 Aeoo 对应的值
    Aeoo = (-log(26) + 2*log(sin(pi*Rational(3, 13)))*cos(pi*Rational(54, 13)) + 2*log(sin(pi*Rational(4, 13)))*cos(pi*Rational(6, 13))
             + 2*log(sin(pi*Rational(6, 13)))*cos(pi*Rational(108, 13)) - 2*log(sin(pi*Rational(5, 13)))*cos(pi/13)
             - 2*log(sin(pi/13))*cos(pi*Rational(5, 13)) + pi*cot(pi*Rational(4, 13))/2
             - 2*log(sin(pi*Rational(2, 13)))*cos(pi*Rational(3, 13)) + Rational(11669332571, 3628714320))

    # 计算调和序列 Hoee 对应的值
    Hoee = harmonic(no + pe/qe)
    # 计算调和序列 Aoee 对应的值
    Aoee = (-log(10) + 2*(Rational(-1, 4) + sqrt(5)/4)*log(sqrt(-sqrt(5)/8 + Rational(5, 8)))
             + 2*(-sqrt(5)/4 - Rational(1, 4))*log(sqrt(sqrt(5)/8 + Rational(5, 8)))
             + pi*sqrt(2*sqrt(5)/5 + 1)/2 + Rational(779405, 277704))

    # 计算调和序列 Hoeo 对应的值
    Hoeo = harmonic(no + pe/qo)
    # 计算调和序列 Aoeo 对应的值
    Aoeo = (-log(26) + 2*log(sin(pi*Rational(3, 13)))*cos(pi*Rational(4, 13)) + 2*log(sin(pi*Rational(2, 13)))*cos(pi*Rational(32, 13))
             + 2*log(sin(pi*Rational(5, 13)))*cos(pi*Rational(80, 13)) - 2*log(sin(pi*Rational(6, 13)))*cos(pi*Rational(5, 13))
             - 2*log(sin(pi*Rational(4, 13)))*cos(pi/13) + pi*cot(pi*Rational(5, 13))/2
             - 2*log(sin(pi/13))*cos(pi*Rational(3, 13)) + Rational(53857323, 16331560))

    # 计算调和序列 Hooe 对应的值
    Hooe = harmonic(no + po/qe)
    # 计算调和序列 Aooe 对应的值
    Aooe = (-log(20) + 2*(Rational(1, 4) + sqrt(5)/4)*log(Rational(-1, 4) + sqrt(5)/4)
             + 2*(Rational(-1, 4) + sqrt(5)/4)*log(sqrt(-sqrt(5)/8 + Rational(5, 8)))
             + 2*(-sqrt(5)/4 - Rational(1, 4))*log(sqrt(sqrt(5)/8 + Rational(5, 8)))
             + 2*(-sqrt(5)/4 + Rational(1, 4))*log(Rational(1, 4) + sqrt(5)/4)
             + Rational(486853480, 186374097) + pi*sqrt(2*sqrt(5) + 5)/2)

    # 计算调和序列 Hooo 对应的值
    Hooo = harmonic(no + po/qo)
    # 计算调和序列 Aooo 对应的值
    Aooo = (-log(26) + 2*log(sin(pi*Rational(3, 13)))*cos(pi*Rational(54, 13)) + 2*log(sin(pi*Rational(4, 13)))*cos(pi*Rational(6, 13))
             + 2*log(sin(pi*Rational(6, 13)))*cos(pi*Rational(108, 13)) - 2*log(sin(pi*Rational(5, 13)))*cos(pi/13)
             - 2*log(sin(pi/13))*cos(pi*Rational(5, 13)) + pi*cot(pi*Rational(4, 13))/2
             - 2*log(sin(pi*Rational(2, 13)))*cos(3*pi/13) + Rational(383693479, 125128080))

    # 将所有调和序列和对应的结果存入列表 H 和 A 中
    H = [Heee, Heeo, Heoe, Heoo, Hoee, Hoeo, Hooe, Hooo]
    A = [Aeee, Aeeo, Aeoe, Aeoo, Aoee, Aoeo, Aooe, Aooo]
    # 遍历 H 和 A 的对应元素，并进行以下断言
    for h, a in zip(H, A):
        # 对 h 进行展开并求值
        e = expand_func(h).doit()
        # 断言取消 h/a 的结果为 1
        assert cancel(e/a) == 1
        # 断言 h 的数值与 a 的数值之差小于 1e-12
        assert abs(h.n() - a.n()) < 1e-12
# 定义测试函数 test_harmonic_evalf，用于测试 harmonic 函数的 evalf 方法返回的精确浮点数值是否符合预期
def test_harmonic_evalf():
    # 断言 harmonic(1.5).evalf(n=10) 返回的字符串形式等于 '1.280372306'
    assert str(harmonic(1.5).evalf(n=10)) == '1.280372306'
    # 断言 harmonic(1.5, 2).evalf(n=10) 返回的字符串形式等于 '1.154576311'，记录 issue 7443
    assert str(harmonic(1.5, 2).evalf(n=10)) == '1.154576311'  # issue 7443
    # 断言 harmonic(4.0, -3).evalf(n=10) 返回的字符串形式等于 '100.0000000'
    assert str(harmonic(4.0, -3).evalf(n=10)) == '100.0000000'
    # 断言 harmonic(7.0, 1.0).evalf(n=10) 返回的字符串形式等于 '2.592857143'
    assert str(harmonic(7.0, 1.0).evalf(n=10)) == '2.592857143'
    # 断言 harmonic(1, pi).evalf(n=10) 返回的字符串形式等于 '1.000000000'
    assert str(harmonic(1, pi).evalf(n=10)) == '1.000000000'
    # 断言 harmonic(2, pi).evalf(n=10) 返回的字符串形式等于 '1.113314732'
    assert str(harmonic(2, pi).evalf(n=10)) == '1.113314732'
    # 断言 harmonic(1000.0, pi).evalf(n=10) 返回的字符串形式等于 '1.176241563'
    assert str(harmonic(1000.0, pi).evalf(n=10)) == '1.176241563'
    # 断言 harmonic(I).evalf(n=10) 返回的字符串形式等于 '0.6718659855 + 1.076674047*I'
    assert str(harmonic(I).evalf(n=10)) == '0.6718659855 + 1.076674047*I'
    # 断言 harmonic(I, I).evalf(n=10) 返回的字符串形式等于 '-0.3970915266 + 1.9629689*I'
    assert str(harmonic(I, I).evalf(n=10)) == '-0.3970915266 + 1.9629689*I'

    # 断言 harmonic(-1.0, 1).evalf() 返回 S.NaN (Not a Number)
    assert harmonic(-1.0, 1).evalf() is S.NaN
    # 断言 harmonic(-2.0, 2.0).evalf() 返回 S.NaN (Not a Number)
    assert harmonic(-2.0, 2.0).evalf() is S.NaN


# 定义测试函数 test_harmonic_rewrite，用于测试 harmonic 函数的 rewrite 方法返回的各种重写形式是否符合预期
def test_harmonic_rewrite():
    from sympy.functions.elementary.piecewise import Piecewise
    # 定义符号 n 和 m
    n = Symbol("n")
    m = Symbol("m", integer=True, positive=True)
    # 定义正数符号 x1 和负数符号 x2
    x1 = Symbol("x1", positive=True)
    x2 = Symbol("x2", negative=True)

    # 断言 harmonic(n).rewrite(digamma) 等于 polygamma(0, n + 1) + EulerGamma
    assert harmonic(n).rewrite(digamma) == polygamma(0, n + 1) + EulerGamma
    # 断言 harmonic(n).rewrite(trigamma) 等于 polygamma(0, n + 1) + EulerGamma
    assert harmonic(n).rewrite(trigamma) == polygamma(0, n + 1) + EulerGamma
    # 断言 harmonic(n).rewrite(polygamma) 等于 polygamma(0, n + 1) + EulerGamma
    assert harmonic(n).rewrite(polygamma) == polygamma(0, n + 1) + EulerGamma

    # 断言 harmonic(n,3).rewrite(polygamma) 等于 polygamma(2, n + 1)/2 - polygamma(2, 1)/2
    assert harmonic(n,3).rewrite(polygamma) == polygamma(2, n + 1)/2 - polygamma(2, 1)/2
    # 断言 harmonic(n,m).rewrite(polygamma) 的类型是 Piecewise
    assert isinstance(harmonic(n,m).rewrite(polygamma), Piecewise)

    # 断言 expand_func(harmonic(n+4)) 等于 harmonic(n) + 1/(n + 4) + 1/(n + 3) + 1/(n + 2) + 1/(n + 1)
    assert expand_func(harmonic(n+4)) == harmonic(n) + 1/(n + 4) + 1/(n + 3) + 1/(n + 2) + 1/(n + 1)
    # 断言 expand_func(harmonic(n-4)) 等于 harmonic(n) - 1/(n - 1) - 1/(n - 2) - 1/(n - 3) - 1/n
    assert expand_func(harmonic(n-4)) == harmonic(n) - 1/(n - 1) - 1/(n - 2) - 1/(n - 3) - 1/n

    # 断言 harmonic(n, m).rewrite("tractable") 等于 harmonic(n, m).rewrite(polygamma)
    assert harmonic(n, m).rewrite("tractable") == harmonic(n, m).rewrite(polygamma)
    # 断言 harmonic(n, x1).rewrite("tractable") 等于 harmonic(n, x1)
    assert harmonic(n, x1).rewrite("tractable") == harmonic(n, x1)
    # 断言 harmonic(n, x1 + 1).rewrite("tractable") 等于 zeta(x1 + 1) - zeta(x1 + 1, n + 1)
    assert harmonic(n, x1 + 1).rewrite("tractable") == zeta(x1 + 1) - zeta(x1 + 1, n + 1)
    # 断言 harmonic(n, x2).rewrite("tractable") 等于 zeta(x2) - zeta(x2, n + 1)
    assert harmonic(n, x2).rewrite("tractable") == zeta(x2) - zeta(x2, n + 1)

    # 定义虚拟符号 _k
    _k = Dummy("k")
    # 断言 harmonic(n).rewrite(Sum).dummy_eq(Sum(1/_k, (_k, 1, n)))
    assert harmonic(n).rewrite(Sum).dummy_eq(Sum(1/_k, (_k, 1, n)))
    # 断言 harmonic(n, m).rewrite(Sum).dummy_eq(Sum(_k**(-m), (_k, 1, n)))
    assert harmonic(n, m).rewrite(Sum).dummy_eq(Sum(_k**(-m), (_k, 1, n)))


# 定义测试函数 test_harmonic_calculus，用于测试 harmonic 函数在微积分操作方面的结果是否符合预期
def test_harmonic_calculus():
    # 定义正数符号 y 和负数符号 z
    y = Symbol("y", positive=True)
    z = Symbol("z", negative=True)
    # 断言 harmonic(x, 1).limit(x, 0) 等于 0
    assert harmonic(x, 1).limit(x, 0) == 0
    # 断言 harmonic(x, y).limit(x, 0) 等于 0
    assert harmonic(x, y).limit(x, 0) == 0
    # 断言 harmonic(x, 1).series(x, y, 2) 等于 harmonic(y) + (x - y)*zeta(2, y + 1) + O((x - y)**2, (x, y))
    assert harmonic(x, 1).series(x, y, 2) == \
            harmonic(y) + (x - y)*zeta(2, y + 1) + O((x - y)**2, (x, y))
    # 断言 limit(harmonic(x, y), x, oo) 等于 harmonic(oo, y)
    assert limit(harmonic(x, y), x, oo) == harmonic(oo, y)
    # 断言 limit(harmonic(x, y + 1),
    # 断言，验证 euler 函数对于输入 8 的计算结果是否为 1385
    assert euler(8) == 1385
    
    # 断言，验证 euler 函数对于输入 20（evaluate=False）的计算结果不等于 370371188237525
    assert euler(20, evaluate=False) != 370371188237525
    
    # 创建一个整数符号变量 n
    n = Symbol('n', integer=True)
    
    # 断言，验证 euler 函数对于符号变量 n 的计算结果不等于 -1
    assert euler(n) != -1
    
    # 断言，验证 euler 函数对于符号变量 n，并将 n 替换为 2 的计算结果是否等于 -1
    assert euler(n).subs(n, 2) == -1
    
    # 断言，验证 euler 函数对于输入 -1 的计算结果是否为 π / 2
    assert euler(-1) == S.Pi / 2
    
    # 断言，验证 euler 函数对于输入 -1 和 1 的计算结果是否为 2 * log(2)
    assert euler(-1, 1) == 2*log(2)
    
    # 断言，验证 euler 函数对于输入 -2 的计算结果的数值近似是否等于 2 * Catalan 常数的数值近似
    assert euler(-2).evalf() == (2*S.Catalan).evalf()
    
    # 断言，验证 euler 函数对于输入 -3 的计算结果的数值近似是否等于 (π^3 / 16) 的数值近似
    assert euler(-3).evalf() == (S.Pi**3 / 16).evalf()
    
    # 断言，验证 euler 函数对于输入 2.3 的计算结果的数值近似是否等于 -1.052850274，保留 10 位小数
    assert str(euler(2.3).evalf(n=10)) == '-1.052850274'
    
    # 断言，验证 euler 函数对于输入 1.2 和 3.4 的计算结果的数值近似是否等于 3.575613489，保留 10 位小数
    assert str(euler(1.2, 3.4).evalf(n=10)) == '3.575613489'
    
    # 断言，验证 euler 函数对于复数单位 I 的计算结果的数值近似是否等于 1.248446443 - 0.7675445124*I，保留 10 位小数
    assert str(euler(I).evalf(n=10)) == '1.248446443 - 0.7675445124*I'
    
    # 断言，验证 euler 函数对于两个复数单位 I 和 I 的计算结果的数值近似是否等于 0.04812930469 + 0.01052411008*I，保留 10 位小数
    assert str(euler(I, I).evalf(n=10)) == '0.04812930469 + 0.01052411008*I'
    
    # 断言，验证 euler 函数对于输入 20 的计算结果的数值近似是否等于 370371188237525.0
    assert euler(20).evalf() == 370371188237525.0
    
    # 断言，验证 euler 函数对于输入 20（evaluate=False）的计算结果的数值近似是否等于 370371188237525.0
    assert euler(20, evaluate=False).evalf() == 370371188237525.0
    
    # 断言，验证 euler 函数对于符号变量 n 的重写为 Sum 符号是否与 euler(n) 相等
    assert euler(n).rewrite(Sum) == euler(n)
    
    # 将 n 重新定义为一个整数符号变量，且非负
    n = Symbol('n', integer=True, nonnegative=True)
    
    # 断言，验证 euler 函数对于 2*n + 1 的重写为 Sum 符号是否等于 0
    assert euler(2*n + 1).rewrite(Sum) == 0
    
    # 创建两个虚拟符号 _j 和 _k
    _j = Dummy('j')
    _k = Dummy('k')
    
    # 断言，验证 euler 函数对于 2*n 的重写为 Sum 符号是否与给定表达式相等，使用虚拟符号 _j 和 _k
    assert euler(2*n).rewrite(Sum).dummy_eq(
            I*Sum((-1)**_j*2**(-_k)*I**(-_k)*(-2*_j + _k)**(2*n + 1)*
                  binomial(_k, _j)/_k, (_j, 0, _k), (_k, 1, 2*n + 1)))
# 测试欧拉数函数对奇数符号正数的情况
def test_euler_odd():
    # 创建一个符号变量 n，设定其为奇数和正数
    n = Symbol('n', odd=True, positive=True)
    # 断言欧拉数函数对于给定的 n 应为 0
    assert euler(n) == 0
    # 更改 n 的属性，仅设定为奇数，断言欧拉数函数结果不为 0
    n = Symbol('n', odd=True)
    assert euler(n) != 0


# 测试欧拉多项式函数
def test_euler_polynomials():
    # 断言欧拉多项式函数 euler(0, x) 应为 1
    assert euler(0, x) == 1
    # 断言欧拉多项式函数 euler(1, x) 应为 x - S.Half
    assert euler(1, x) == x - S.Half
    # 断言欧拉多项式函数 euler(2, x) 应为 x**2 - x
    assert euler(2, x) == x**2 - x
    # 断言欧拉多项式函数 euler(3, x) 应为 x**3 - (3*x**2)/2 + Rational(1, 4)
    assert euler(3, x) == x**3 - (3*x**2)/2 + Rational(1, 4)
    # 创建一个符号变量 m
    m = Symbol('m')
    # 断言欧拉多项式函数 euler(m, x) 返回值类型为 euler 类
    assert isinstance(euler(m, x), euler)
    # 导入 Float 类，并从 Maple 中获取值赋给 A
    from sympy.core.numbers import Float
    A = Float('-0.46237208575048694923364757452876131e8')  # from Maple
    # 计算 euler(19, S.Pi) 的数值，并断言精度符合要求
    B = euler(19, S.Pi).evalf(32)
    assert abs((A - B)/A) < 1e-31


# 测试欧拉多项式函数重写
def test_euler_polynomial_rewrite():
    # 创建一个符号变量 m
    m = Symbol('m')
    # 将 euler(m, x) 重写为 Sum 类
    A = euler(m, x).rewrite('Sum');
    # 断言 euler(3, 5) 与重写结果相等
    assert A.subs({m:3, x:5}).doit() == euler(3, 5)


# 测试卡塔兰数函数
def test_catalan():
    # 创建整数符号变量 n
    n = Symbol('n', integer=True)
    # 创建正整数符号变量 m
    m = Symbol('m', integer=True, positive=True)
    # 创建非负整数符号变量 k
    k = Symbol('k', integer=True, nonnegative=True)
    # 创建非负符号变量 p
    p = Symbol('p', nonnegative=True)

    # 预先计算的卡塔兰数列表
    catalans = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786]
    # 遍历列表，断言卡塔兰数函数的结果与预期值相等
    for i, c in enumerate(catalans):
        assert catalan(i) == c
        assert catalan(n).rewrite(factorial).subs(n, i) == c
        assert catalan(n).rewrite(Product).subs(n, i).doit() == c

    # 断言 unchanged 函数对 catalan 函数的结果为真
    assert unchanged(catalan, x)
    # 断言 catalan(2*x) 重写为二项式系数的形式
    assert catalan(2*x).rewrite(binomial) == binomial(4*x, 2*x)/(2*x + 1)
    # 断言 catalan(S.Half) 重写为 gamma 函数的形式
    assert catalan(S.Half).rewrite(gamma) == 8/(3*pi)
    # 进一步重写和计算结果的断言
    assert catalan(S.Half).rewrite(factorial).rewrite(gamma) ==\
        8 / (3 * pi)
    assert catalan(3*x).rewrite(gamma) == 4**(
        3*x)*gamma(3*x + S.Half)/(sqrt(pi)*gamma(3*x + 2))
    assert catalan(x).rewrite(hyper) == hyper((-x + 1, -x), (2,), 1)

    # 断言 catalan(n) 重写为阶乘形式
    assert catalan(n).rewrite(factorial) == factorial(2*n) / (factorial(n + 1)
                                                              * factorial(n))
    # 断言 catalan(m) 和 catalan(n) 重写为 Product 类
    assert isinstance(catalan(n).rewrite(Product), catalan)
    assert isinstance(catalan(m).rewrite(Product), Product)

    # 断言 catalan(x) 对 x 的微分结果
    assert diff(catalan(x), x) == (polygamma(
        0, x + S.Half) - polygamma(0, x + 2) + log(4))*catalan(x)

    # 断言 catalan(x) 的数值计算等于其自身
    assert catalan(x).evalf() == catalan(x)
    # 对于 catalan(S.Half) 的数值结果进行断言
    c = catalan(S.Half).evalf()
    assert str(c) == '0.848826363156775'
    c = catalan(I).evalf(3)
    assert str((re(c), im(c))) == '(0.398, -0.0209)'

    # 对卡塔兰数函数的假设进行断言
    assert catalan(p).is_positive is True
    assert catalan(k).is_integer is True
    assert catalan(m+3).is_composite is True


# 测试格诺奇数函数
def test_genocchi():
    # 预先计算的格诺奇数列表
    genocchis = [0, -1, -1, 0, 1, 0, -3, 0, 17]
    # 遍历列表，断言格诺奇数函数的结果与预期值相等
    for n, g in enumerate(genocchis):
        assert genocchi(n) == g

    # 创建整数符号变量 m
    m = Symbol('m', integer=True)
    # 创建正整数符号变量 n
    n = Symbol('n', integer=True, positive=True)
    # 断言 unchanged 函数对 genocchi 函数的结果为真
    assert unchanged(genocchi, m)
    # 断言 genocchi(2*n + 1) 应为 0
    assert genocchi(2*n + 1) == 0
    # 计算 gn，并断言格诺奇数函数重写为伯努利数的形式
    gn = 2 * (1 - 2**n) * bernoulli(n)
    assert genocchi(n).rewrite(bernoulli).factor() == gn.factor()
    # 计算 gnx，并断言格诺奇数函数重写为伯努利数的形式
    gnx = 2 * (bernoulli(n, x) - 2**n * bernoulli(n, (x+1) / 2))
    assert genocchi(n, x).rewrite(bernoulli).factor() == gnx.factor()
    # 断言 genocchi(2 * n) 的奇偶性质
    assert genocchi(2 * n).is_odd
    assert genocchi(2 * n).is_even is False
    # 断言 genocchi(2 * n + 1) 是否为偶数
    assert genocchi(2 * n + 1).is_even
    # 断言 genocchi(n) 是否为整数
    assert genocchi(n).is_integer
    # 断言 genocchi(4 * n) 是否为正数
    assert genocchi(4 * n).is_positive
    # 这是唯一的两个质数 Genocchi 数
    assert genocchi(6, evaluate=False).is_prime == S(-3).is_prime
    # 断言 genocchi(8, evaluate=False) 是否为质数
    assert genocchi(8, evaluate=False).is_prime
    # 断言 genocchi(4 * n + 2) 是否为负数
    assert genocchi(4 * n + 2).is_negative
    # 断言 genocchi(4 * n + 1) 是否为非负数
    assert genocchi(4 * n + 1).is_negative is False
    # 断言 genocchi(4 * n - 2) 是否为负数
    assert genocchi(4 * n - 2).is_negative

    # 获取 genocchi(0, evaluate=False)
    g0 = genocchi(0, evaluate=False)
    # 断言 g0 是否为正数
    assert g0.is_positive is False
    # 断言 g0 是否为负数
    assert g0.is_negative is False
    # 断言 g0 是否为偶数
    assert g0.is_even is True
    # 断言 g0 是否为奇数
    assert g0.is_odd is False

    # 断言计算 genocchi(0, x) 的结果是否为 0
    assert genocchi(0, x) == 0
    # 断言计算 genocchi(1, x) 的结果是否为 -1
    assert genocchi(1, x) == -1
    # 断言计算 genocchi(2, x) 的结果是否为 1 - 2*x
    assert genocchi(2, x) == 1 - 2*x
    # 断言计算 genocchi(3, x) 的结果是否为 3*x - 3*x**2
    assert genocchi(3, x) == 3*x - 3*x**2
    # 断言计算 genocchi(4, x) 的结果是否为 -1 + 6*x**2 - 4*x**3
    assert genocchi(4, x) == -1 + 6*x**2 - 4*x**3
    # 创建一个新的符号变量 y
    y = Symbol("y")
    # 断言计算 genocchi(5, (x+y)**100) 的结果是否符合预期公式
    assert genocchi(5, (x+y)**100) == -5*(x+y)**400 + 10*(x+y)**300 - 5*(x+y)**100

    # 断言以 10 位精度计算 genocchi(5.0, 4.0) 的结果是否为 '-660.0000000'
    assert str(genocchi(5.0, 4.0).evalf(n=10)) == '-660.0000000'
    # 断言以 10 位精度计算 genocchi(Rational(5, 4)) 的结果是否为 '-1.104286457'
    assert str(genocchi(Rational(5, 4)).evalf(n=10)) == '-1.104286457'
    # 断言以 10 位精度计算 genocchi(-2) 的结果是否为 '3.606170709'
    assert str(genocchi(-2).evalf(n=10)) == '3.606170709'
    # 断言以 10 位精度计算 genocchi(1.3, 3.7) 的结果是否为 '-1.847375373'
    assert str(genocchi(1.3, 3.7).evalf(n=10)) == '-1.847375373'
    # 断言以 10 位精度计算 genocchi(I, 1.0) 的结果是否为 '-0.3161917278 - 1.45311955*I'
    assert str(genocchi(I, 1.0).evalf(n=10)) == '-0.3161917278 - 1.45311955*I'

    # 创建一个新的符号变量 n
    n = Symbol('n')
    # 断言将 genocchi(n, x) 重写为 dirichlet_eta 函数的结果是否符合预期
    assert genocchi(n, x).rewrite(dirichlet_eta) == -2*n * dirichlet_eta(1-n, x)
# 定义名为 test_andre 的测试函数
def test_andre():
    # 定义一个整数列表 nums
    nums = [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]
    # 使用 enumerate 遍历 nums 列表，n 是索引，a 是元素值
    for n, a in enumerate(nums):
        # 断言调用 andre(n) 的结果等于对应的 a
        assert andre(n) == a
    # 断言调用 andre(S.Infinity) 的结果等于 S.Infinity
    assert andre(S.Infinity) == S.Infinity
    # 断言调用 andre(-1) 的结果等于 -log(2)
    assert andre(-1) == -log(2)
    # 断言调用 andre(-2) 的结果等于 -2*S.Catalan
    assert andre(-2) == -2*S.Catalan
    # 断言调用 andre(-3) 的结果等于 3*zeta(3)/16
    assert andre(-3) == 3*zeta(3)/16
    # 断言调用 andre(-5) 的结果等于 -15*zeta(5)/256
    assert andre(-5) == -15*zeta(5)/256
    # 这里断言 andre(-2*n) 与 Dirichlet beta 函数相关，但 SymPy 不支持该函数（或一般的 L 函数）
    assert unchanged(andre, -4)

    # 创建整数符号 n
    n = Symbol('n', integer=True, nonnegative=True)
    # 断言调用 andre(n) 不变
    assert unchanged(andre, n)
    # 断言调用 andre(n) 的结果是整数
    assert andre(n).is_integer is True
    # 断言调用 andre(n) 的结果是正数
    assert andre(n).is_positive is True

    # 断言调用 andre(10, evaluate=False) 的字符串化结果等于 '50521.00000'
    assert str(andre(10, evaluate=False).evalf(n=10)) == '50521.00000'
    # 断言调用 andre(-1, evaluate=False) 的字符串化结果等于 '-0.6931471806'
    assert str(andre(-1, evaluate=False).evalf(n=10)) == '-0.6931471806'
    # 断言调用 andre(-2, evaluate=False) 的字符串化结果等于 '-1.831931188'
    assert str(andre(-2, evaluate=False).evalf(n=10)) == '-1.831931188'
    # 断言调用 andre(-4, evaluate=False) 的字符串化结果等于 '1.977889103'
    assert str(andre(-4, evaluate=False).evalf(n=10)) == '1.977889103'
    # 断言调用 andre(I, evaluate=False) 的字符串化结果等于 '2.378417833 + 0.6343322845*I'
    assert str(andre(I, evaluate=False).evalf(n=10)) == '2.378417833 + 0.6343322845*I'

    # 断言调用 andre(x) 的 polylog 重写结果
    assert andre(x).rewrite(polylog) == \
            (-I)**(x+1) * polylog(-x, I) + I**(x+1) * polylog(-x, -I)
    # 断言调用 andre(x) 的 zeta 重写结果
    assert andre(x).rewrite(zeta) == \
            2 * gamma(x+1) / (2*pi)**(x+1) * \
            (zeta(x+1, Rational(1,4)) - cos(pi*x) * zeta(x+1, Rational(3,4)))


# 带有 @nocache_fail 装饰器的测试函数 test_partition
@nocache_fail
def test_partition():
    # 定义整数分区数列表 partition_nums
    partition_nums = [1, 1, 2, 3, 5, 7, 11, 15, 22]
    # 使用 enumerate 遍历 partition_nums 列表，n 是索引，p 是元素值
    for n, p in enumerate(partition_nums):
        # 断言调用 partition(n) 的结果等于对应的 p
        assert partition(n) == p

    # 创建符号 x 和 y，m, n, p 分别为整数、负数、非负整数符号
    x = Symbol('x')
    y = Symbol('y', real=True)
    m = Symbol('m', integer=True)
    n = Symbol('n', integer=True, negative=True)
    p = Symbol('p', integer=True, nonnegative=True)
    # 断言调用 partition(m) 的结果是整数
    assert partition(m).is_integer
    # 断言调用 partition(m) 的结果不是负数
    assert not partition(m).is_negative
    # 断言调用 partition(m) 的结果是非负数
    assert partition(m).is_nonnegative
    # 断言调用 partition(n) 的结果是零
    assert partition(n).is_zero
    # 断言调用 partition(p) 的结果是正数
    assert partition(p).is_positive
    # 断言调用 partition(x) 的结果替换 x 为 7 等于 15
    assert partition(x).subs(x, 7) == 15
    # 断言调用 partition(y) 的结果替换 y 为 8 等于 22
    assert partition(y).subs(y, 8) == 22
    # 断言 partition(Rational(5, 4)) 抛出 TypeError 异常
    raises(TypeError, lambda: partition(Rational(5, 4)))


# 定义名为 test_divisor_sigma 的测试函数
def test_divisor_sigma():
    # 错误情况
    # 创建整数符号 m，调用 divisor_sigma(m) 抛出 TypeError 异常
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: divisor_sigma(m))
    # 调用 divisor_sigma(4.5) 抛出 TypeError 异常
    raises(TypeError, lambda: divisor_sigma(4.5))
    # 调用 divisor_sigma(1, m) 抛出 TypeError 异常
    raises(TypeError, lambda: divisor_sigma(1, m))
    # 调用 divisor_sigma(1, 4.5) 抛出 TypeError 异常
    raises(TypeError, lambda: divisor_sigma(1, 4.5))
    # 创建正数符号 m，调用 divisor_sigma(m) 抛出 ValueError 异常
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: divisor_sigma(m))
    # 调用 divisor_sigma(0) 抛出 ValueError 异常
    raises(ValueError, lambda: divisor_sigma(0))
    # 创建负数符号 m，调用 divisor_sigma(1, m) 抛出 ValueError 异常
    m = Symbol('m', negative=True)
    raises(ValueError, lambda: divisor_sigma(1, m))
    # 调用 divisor_sigma(1, -1) 抛出 ValueError 异常
    raises(ValueError, lambda: divisor_sigma(1, -1))

    # 特殊情况
    # 创建素数符号 p 和整数符号 k
    p = Symbol('p', prime=True)
    k = Symbol('k', integer=True)
    # 断言 divisor_sigma(p, 1) 的结果等于 p + 1
    assert divisor_sigma(p, 1) == p + 1
    # 断言 divisor_sigma(p, k) 的结果等于 p**k + 1
    assert divisor_sigma(p, k) == p**k + 1

    # 属性
    # 创建正整数符号 n
    n = Symbol('n', integer=True, positive=True)
    # 断言 divisor_sigma(n) 的结果是整数
    assert divisor_sigma(n).is_integer is True
    # 断言 divisor_sigma(n) 的结果是正数
    assert divisor_sigma(n).is_positive is True

    # 符号表示
    # 创建非零整数符号 k
    k = Symbol('k', integer=True, zero=False)
    # 断言 divisor_sigma(4, k) 的结果等于 2**(2*k) + 2**k + 1
    assert divisor_sigma(4, k) == 2**(2*k) + 2**k + 1
    # 验证 div_sigma 函数在特定输入和指数 k 下的计算结果是否正确
    assert divisor_sigma(6, k) == (2**k + 1) * (3**k + 1)
    
    # 测试对于整数 23450 的各种指数 k 下的 divisor_sigma 函数的计算结果是否正确
    assert divisor_sigma(23450) == 50592           # k 默认为 1
    assert divisor_sigma(23450, 0) == 24           # k 为 0，按照公式结果应为 2**0 + 1 * 3**0 + 1 = 2 * 3 = 6
    assert divisor_sigma(23450, 1) == 50592        # k 为 1，按照公式结果应为 (2**1 + 1) * (3**1 + 1) = 3 * 4 = 12
    assert divisor_sigma(23450, 2) == 730747500    # k 为 2，按照公式结果应为 (2**2 + 1) * (3**2 + 1) = 5 * 10 = 50
    assert divisor_sigma(23450, 3) == 14666785333344  # k 为 3，按照公式结果应为 (2**3 + 1) * (3**3 + 1) = 9 * 28 = 252
def test_udivisor_sigma():
    # 定义一个符号变量 m，要求其为非整数类型，期望抛出 TypeError 异常
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: udivisor_sigma(m))
    # 要求参数为浮点数，期望抛出 TypeError 异常
    raises(TypeError, lambda: udivisor_sigma(4.5))
    # 要求有且只有一个参数，期望抛出 TypeError 异常
    raises(TypeError, lambda: udivisor_sigma(1, m))
    # 第一个参数为整数，第二个参数为浮点数，期望抛出 TypeError 异常
    raises(TypeError, lambda: udivisor_sigma(1, 4.5))
    # 定义一个符号变量 m，要求其为非正数，期望抛出 ValueError 异常
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: udivisor_sigma(m))
    # 第一个参数为零，期望抛出 ValueError 异常
    raises(ValueError, lambda: udivisor_sigma(0))
    # 定义一个符号变量 m，要求其为负数，期望抛出 ValueError 异常
    m = Symbol('m', negative=True)
    raises(ValueError, lambda: udivisor_sigma(1, m))
    # 第一个参数为 1，第二个参数为 -1，期望抛出 ValueError 异常

    # 特殊情况
    p = Symbol('p', prime=True)
    k = Symbol('k', integer=True)
    # 断言特定情况下的 udivisor_sigma 函数返回值
    assert udivisor_sigma(p, 1) == p + 1
    assert udivisor_sigma(p, k) == p**k + 1

    # 属性检查
    n = Symbol('n', integer=True, positive=True)
    # 断言 udivisor_sigma(n) 返回值的属性 is_integer 是 True
    assert udivisor_sigma(n).is_integer is True
    # 断言 udivisor_sigma(n) 返回值的属性 is_positive 是 True

    # Integer
    # A034444 列表中每个元素对应的 udivisor_sigma(n, 0) 的返回值与该列表值相等
    A034444 = [1, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 4, 2, 2, 4, 2, 4,
               4, 4, 2, 4, 2, 4, 2, 4, 2, 8, 2, 2, 4, 4, 4, 4, 2, 4, 4, 4,
               2, 8, 2, 4, 4, 4, 2, 4, 2, 4, 4, 4, 2, 4, 4, 4, 4, 4, 2, 8]
    for n, val in enumerate(A034444, 1):
        assert udivisor_sigma(n, 0) == val
    # A034448 列表中每个元素对应的 udivisor_sigma(n, 1) 的返回值与该列表值相等
    A034448 = [1, 3, 4, 5, 6, 12, 8, 9, 10, 18, 12, 20, 14, 24, 24, 17, 18,
               30, 20, 30, 32, 36, 24, 36, 26, 42, 28, 40, 30, 72, 32, 33,
               48, 54, 48, 50, 38, 60, 56, 54, 42, 96, 44, 60, 60, 72, 48]
    for n, val in enumerate(A034448, 1):
        assert udivisor_sigma(n, 1) == val
    # A034676 列表中每个元素对应的 udivisor_sigma(n, 2) 的返回值与该列表值相等
    A034676 = [1, 5, 10, 17, 26, 50, 50, 65, 82, 130, 122, 170, 170, 250,
               260, 257, 290, 410, 362, 442, 500, 610, 530, 650, 626, 850,
               730, 850, 842, 1300, 962, 1025, 1220, 1450, 1300, 1394, 1370]
    for n, val in enumerate(A034676, 1):
        assert udivisor_sigma(n, 2) == val


def test_legendre_symbol():
    # error
    m = Symbol('m', integer=False)
    # 要求 m 为非整数，期望抛出 TypeError 异常
    raises(TypeError, lambda: legendre_symbol(m, 3))
    # 第一个参数为浮点数，期望抛出 TypeError 异常
    raises(TypeError, lambda: legendre_symbol(4.5, 3))
    # 第一个参数为整数，第二个参数为符号变量 m，期望抛出 TypeError 异常
    raises(TypeError, lambda: legendre_symbol(1, m))
    # 第一个参数为整数，第二个参数为浮点数，期望抛出 TypeError 异常
    raises(TypeError, lambda: legendre_symbol(1, 4.5))
    # 要求 m 为非素数，期望抛出 ValueError 异常
    m = Symbol('m', prime=False)
    raises(ValueError, lambda: legendre_symbol(1, m))
    # 第二个参数为非素数，期望抛出 ValueError 异常
    raises(ValueError, lambda: legendre_symbol(1, 6))
    # 要求 m 为非奇数，期望抛出 ValueError 异常
    m = Symbol('m', odd=False)
    raises(ValueError, lambda: legendre_symbol(1, m))
    # 第二个参数为非奇数，期望抛出 ValueError 异常
    raises(ValueError, lambda: legendre_symbol(1, 2))

    # 特殊情况
    p = Symbol('p', prime=True)
    k = Symbol('k', integer=True)
    # 断言特定情况下的 legendre_symbol 函数返回值
    assert legendre_symbol(p*k, p) == 0
    assert legendre_symbol(1, p) == 1

    # 属性检查
    n = Symbol('n')
    m = Symbol('m')
    # 断言 legendre_symbol(m, n) 返回值的属性 is_integer 是 True
    assert legendre_symbol(m, n).is_integer is True
    # 断言 legendre_symbol(m, n) 返回值的属性 is_prime 是 False

    # Integer
    # 断言一些具体的 legendre_symbol 函数调用的返回值
    assert legendre_symbol(5, 11) == 1
    assert legendre_symbol(25, 41) == 1
    assert legendre_symbol(67, 101) == -1
    assert legendre_symbol(0, 13) == 0
    assert legendre_symbol(9, 3) == 0


def test_jacobi_symbol():
    # error
    # 创建一个符号变量 'm'，指定其为非整数
    m = Symbol('m', integer=False)
    # 测试 jacobi_symbol 函数对于非整数参数 m 是否会引发 TypeError 异常
    raises(TypeError, lambda: jacobi_symbol(m, 3))
    # 测试 jacobi_symbol 函数对于浮点数参数是否会引发 TypeError 异常
    raises(TypeError, lambda: jacobi_symbol(4.5, 3))
    # 测试 jacobi_symbol 函数对于 m 参数为符号变量是否会引发 TypeError 异常
    raises(TypeError, lambda: jacobi_symbol(1, m))
    # 测试 jacobi_symbol 函数对于第二个参数为浮点数是否会引发 TypeError 异常
    raises(TypeError, lambda: jacobi_symbol(1, 4.5))
    # 创建一个符号变量 'm'，指定其为非正数
    m = Symbol('m', positive=False)
    # 测试 jacobi_symbol 函数对于第二个参数为非正数是否会引发 ValueError 异常
    raises(ValueError, lambda: jacobi_symbol(1, m))
    # 测试 jacobi_symbol 函数对于第二个参数为负数是否会引发 ValueError 异常
    raises(ValueError, lambda: jacobi_symbol(1, -6))
    # 创建一个符号变量 'm'，指定其为非奇数
    m = Symbol('m', odd=False)
    # 测试 jacobi_symbol 函数对于第二个参数为非奇数是否会引发 ValueError 异常
    raises(ValueError, lambda: jacobi_symbol(1, m))
    # 测试 jacobi_symbol 函数对于第二个参数为偶数是否会引发 ValueError 异常
    raises(ValueError, lambda: jacobi_symbol(1, 2))

    # 特殊情况的测试
    # 创建一个整数符号变量 'p'
    p = Symbol('p', integer=True)
    # 创建一个整数符号变量 'k'
    k = Symbol('k', integer=True)
    # 验证 jacobi_symbol 函数对于 p*k 和 p 的 Jacobi 符号是否为 0
    assert jacobi_symbol(p*k, p) == 0
    # 验证 jacobi_symbol 函数对于 1 和 p 的 Jacobi 符号是否为 1
    assert jacobi_symbol(1, p) == 1
    # 验证 jacobi_symbol 函数对于 1 和 1 的 Jacobi 符号是否为 1
    assert jacobi_symbol(1, 1) == 1
    # 验证 jacobi_symbol 函数对于 0 和 1 的 Jacobi 符号是否为 1
    assert jacobi_symbol(0, 1) == 1

    # 属性测试
    # 创建两个未指定类型的符号变量 'n' 和 'm'
    n = Symbol('n')
    m = Symbol('m')
    # 验证 jacobi_symbol 函数返回值的 is_integer 属性是否为 True
    assert jacobi_symbol(m, n).is_integer is True
    # 验证 jacobi_symbol 函数返回值的 is_prime 属性是否为 False
    assert jacobi_symbol(m, n).is_prime is False

    # 整数参数的测试
    assert jacobi_symbol(25, 41) == 1
    assert jacobi_symbol(-23, 83) == -1
    assert jacobi_symbol(3, 9) == 0
    assert jacobi_symbol(42, 97) == -1
    assert jacobi_symbol(3, 5) == -1
    assert jacobi_symbol(7, 9) == 1
    assert jacobi_symbol(0, 3) == 0
    assert jacobi_symbol(0, 1) == 1
    assert jacobi_symbol(2, 1) == 1
    assert jacobi_symbol(1, 3) == 1
def test_kronecker_symbol():
    # 测试 kronecker_symbol 函数的错误输入情况
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: kronecker_symbol(m, 3))
    raises(TypeError, lambda: kronecker_symbol(4.5, 3))
    raises(TypeError, lambda: kronecker_symbol(1, m))
    raises(TypeError, lambda: kronecker_symbol(1, 4.5))

    # 测试 kronecker_symbol 函数的特殊情况
    p = Symbol('p', integer=True)
    assert kronecker_symbol(1, p) == 1
    assert kronecker_symbol(1, 1) == 1
    assert kronecker_symbol(0, 1) == 1

    # 测试 kronecker_symbol 函数的属性
    n = Symbol('n')
    m = Symbol('m')
    assert kronecker_symbol(m, n).is_integer is True
    assert kronecker_symbol(m, n).is_prime is False

    # 整数测试
    for n in range(3, 10, 2):
        for a in range(-n, n):
            val = kronecker_symbol(a, n)
            assert val == jacobi_symbol(a, n)
            minus = kronecker_symbol(a, -n)
            if a < 0:
                assert -minus == val
            else:
                assert minus == val
            even = kronecker_symbol(a, 2 * n)
            if a % 2 == 0:
                assert even == 0
            elif a % 8 in [1, 7]:
                assert even == val
            else:
                assert -even == val
    assert kronecker_symbol(1, 0) == kronecker_symbol(-1, 0) == 1
    assert kronecker_symbol(0, 0) == 0


def test_mobius():
    # 测试 mobius 函数的错误输入情况
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: mobius(m))
    raises(TypeError, lambda: mobius(4.5))
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: mobius(m))
    raises(ValueError, lambda: mobius(-3))

    # 测试 mobius 函数的特殊情况
    p = Symbol('p', prime=True)
    assert mobius(p) == -1

    # 测试 mobius 函数的属性
    n = Symbol('n', integer=True, positive=True)
    assert mobius(n).is_integer is True
    assert mobius(n).is_prime is False

    # 符号表达式测试
    n = Symbol('n', integer=True, positive=True)
    k = Symbol('k', integer=True, positive=True)
    assert mobius(n**2) == 0
    assert mobius(4*n) == 0
    assert isinstance(mobius(n**k), mobius)
    assert mobius(n**(k+1)) == 0
    assert isinstance(mobius(3**k), mobius)
    assert mobius(3**(k+1)) == 0
    m = Symbol('m')
    assert isinstance(mobius(4*m), mobius)

    # 整数测试
    assert mobius(13*7) == 1
    assert mobius(1) == 1
    assert mobius(13*7*5) == -1
    assert mobius(13**2) == 0
    A008683 = [1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1, 0,
               -1, 0, 1, 1, -1, 0, 0, 1, 0, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1,
               1, 1, 0, -1, -1, -1, 0, 0, 1, -1, 0, 0, 0, 1, 0, -1, 0, 1, 0]
    for n, val in enumerate(A008683, 1):
        assert mobius(n) == val


def test_primenu():
    # 测试 primenu 函数的错误输入情况
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: primenu(m))
    raises(TypeError, lambda: primenu(4.5))
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: primenu(m))
    raises(ValueError, lambda: primenu(0))

    # 测试 primenu 函数的特殊情况
    p = Symbol('p', prime=True)
    assert primenu(p) == 1

    # 测试 primenu 函数的属性
    # 定义符号变量 n，要求其为正整数
    n = Symbol('n', integer=True, positive=True)
    # 断言 primenu(n) 返回的结果是整数
    assert primenu(n).is_integer is True
    # 断言 primenu(n) 返回的结果是非负数
    assert primenu(n).is_nonnegative is True

    # Integer
    # 断言 primenu(7*13) 的结果为 2
    assert primenu(7*13) == 2
    # 断言 primenu(2*17*19) 的结果为 3
    assert primenu(2*17*19) == 3
    # 断言 primenu(2**3 * 17 * 19**2) 的结果为 3
    assert primenu(2**3 * 17 * 19**2) == 3

    # A001221 序列，每个索引 n 对应的值为 val
    A001221 = [0, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2,
               1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 3, 1, 1, 2, 2, 2, 2]
    # 遍历 A001221 列表，索引从 1 开始，值为 val
    for n, val in enumerate(A001221, 1):
        # 断言 primenu(n) 的结果与 A001221 列表对应索引的值相等
        assert primenu(n) == val
def test_primeomega():
    # 对于非整数的符号m，应该引发TypeError异常
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: primeomega(m))
    # 对于浮点数输入，应该引发TypeError异常
    raises(TypeError, lambda: primeomega(4.5))
    # 对于非正整数的符号m，应该引发ValueError异常
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: primeomega(m))
    # 对于输入为0的情况，应该引发ValueError异常
    raises(ValueError, lambda: primeomega(0))

    # 特殊情况：如果p是质数符号，primeomega(p)应该等于1
    p = Symbol('p', prime=True)
    assert primeomega(p) == 1

    # 属性测试：对于整数且为正的符号n，primeomega(n)返回的结果应该是整数
    n = Symbol('n', integer=True, positive=True)
    assert primeomega(n).is_integer is True
    # 对于整数且为正的符号n，primeomega(n)返回的结果应该是非负的
    assert primeomega(n).is_nonnegative is True

    # 整数测试
    assert primeomega(7*13) == 2
    assert primeomega(2*17*19) == 3
    assert primeomega(2**3 * 17 * 19**2) == 6
    A001222 = [0, 1, 1, 2, 1, 2, 1, 3, 2, 2, 1, 3, 1, 2, 2, 4, 1, 3,
               1, 3, 2, 2, 1, 4, 2, 2, 3, 3, 1, 3, 1, 5, 2, 2, 2, 4]
    for n, val in enumerate(A001222, 1):
        assert primeomega(n) == val


def test_totient():
    # 对于非整数的符号m，应该引发TypeError异常
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: totient(m))
    # 对于浮点数输入，应该引发TypeError异常
    raises(TypeError, lambda: totient(4.5))
    # 对于非正整数的符号m，应该引发ValueError异常
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: totient(m))
    # 对于输入为0的情况，应该引发ValueError异常
    raises(ValueError, lambda: totient(0))

    # 特殊情况：如果p是质数符号，totient(p)应该等于p - 1
    p = Symbol('p', prime=True)
    assert totient(p) == p - 1

    # 属性测试：对于整数且为正的符号n，totient(n)返回的结果应该是整数
    n = Symbol('n', integer=True, positive=True)
    assert totient(n).is_integer is True
    # 对于整数且为正的符号n，totient(n)返回的结果应该是正的
    assert totient(n).is_positive is True

    # 整数测试
    assert totient(7*13) == totient(factorint(7*13)) == (7-1)*(13-1)
    assert totient(2*17*19) == totient(factorint(2*17*19)) == (17-1)*(19-1)
    assert totient(2**3 * 17 * 19**2) == totient({2: 3, 17: 1, 19: 2}) == 2**2 * (17-1) * 19*(19-1)
    A000010 = [1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16,
               6, 18, 8, 12, 10, 22, 8, 20, 12, 18, 12, 28, 8, 30, 16,
               20, 16, 24, 12, 36, 18, 24, 16, 40, 12, 42, 20, 24, 22]
    for n, val in enumerate(A000010, 1):
        assert totient(n) == val


def test_reduced_totient():
    # 对于非整数的符号m，应该引发TypeError异常
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: reduced_totient(m))
    # 对于浮点数输入，应该引发TypeError异常
    raises(TypeError, lambda: reduced_totient(4.5))
    # 对于非正整数的符号m，应该引发ValueError异常
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: reduced_totient(m))
    # 对于输入为0的情况，应该引发ValueError异常
    raises(ValueError, lambda: reduced_totient(0))

    # 特殊情况：如果p是质数符号，reduced_totient(p)应该等于p - 1
    p = Symbol('p', prime=True)
    assert reduced_totient(p) == p - 1

    # 属性测试：对于整数且为正的符号n，reduced_totient(n)返回的结果应该是整数
    n = Symbol('n', integer=True, positive=True)
    assert reduced_totient(n).is_integer is True
    # 对于整数且为正的符号n，reduced_totient(n)返回的结果应该是正的
    assert reduced_totient(n).is_positive is True

    # 整数测试
    assert reduced_totient(7*13) == reduced_totient(factorint(7*13)) == 12
    assert reduced_totient(2*17*19) == reduced_totient(factorint(2*17*19)) == 144
    assert reduced_totient(2**2 * 11) == reduced_totient({2: 2, 11: 1}) == 10
    assert reduced_totient(2**3 * 17 * 19**2) == reduced_totient({2: 3, 17: 1, 19: 2}) == 2736
    A002322 = [1, 1, 2, 2, 4, 2, 6, 2, 6, 4, 10, 2, 12, 6, 4, 4, 16, 6,
               18, 4, 6, 10, 22, 2, 20, 12, 18, 6, 28, 4, 30, 8, 10, 16,
               12, 6, 36, 18, 12, 4, 40, 6, 42, 10, 12, 22, 46, 4, 42]
    # 遍历 A002322 列表，enumerate 函数从1开始枚举每个元素及其索引
    for n, val in enumerate(A002322, 1):
        # 使用断言验证 reduced_totient 函数计算的结果与预期的值相等
        assert reduced_totient(n) == val
def test_primepi():
    # 检查 primepi 函数对非实数参数的处理，应该引发 TypeError
    z = Symbol('z', real=False)
    raises(TypeError, lambda: primepi(z))
    raises(TypeError, lambda: primepi(I))

    # 检查 primepi 函数的属性
    n = Symbol('n', integer=True, positive=True)
    assert primepi(n).is_integer is True
    assert primepi(n).is_nonnegative is True

    # 检查 primepi 函数对无穷大的处理
    assert primepi(oo) == oo
    assert primepi(-oo) == 0

    # 检查 primepi 函数对符号参数的处理
    x = Symbol('x')
    assert isinstance(primepi(x), primepi)

    # 检查 primepi 函数对整数参数的处理
    assert primepi(0) == 0
    A000720 = [0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8,
               8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11, 11, 11,
               12, 12, 12, 12, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15]
    for n, val in enumerate(A000720, 1):
        assert primepi(n) == primepi(n + 0.5) == val


def test__nT():
    # 检查 _nT 函数在多种参数组合下的返回结果是否符合预期
    assert [_nT(i, j) for i in range(5) for j in range(i + 2)] == [
        1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 2, 1, 1, 0]
    check = [_nT(10, i) for i in range(11)]
    assert check == [0, 1, 5, 8, 9, 7, 5, 3, 2, 1, 1]
    assert all(type(i) is int for i in check)
    assert _nT(10, 5) == 7
    assert _nT(100, 98) == 2
    assert _nT(100, 100) == 1
    assert _nT(10, 3) == 8


def test_nC_nP_nT():
    # 导入必要的函数和类
    from sympy.utilities.iterables import (
        multiset_permutations, multiset_combinations, multiset_partitions,
        partitions, subsets, permutations)
    from sympy.functions.combinatorial.numbers import (
        nP, nC, nT, stirling, _stirling1, _stirling2, _multiset_histogram, _AOP_product)

    from sympy.combinatorics.permutations import Permutation
    from sympy.core.random import choice

    c = string.ascii_lowercase
    # 针对随机生成的字符串 s 进行多次测试
    for i in range(100):
        s = ''.join(choice(c) for i in range(7))
        u = len(s) == len(set(s))
        try:
            tot = 0
            # 测试 nP 函数在不同参数下的结果是否正确
            for i in range(8):
                check = nP(s, i)
                tot += check
                assert len(list(multiset_permutations(s, i))) == check
                if u:
                    assert nP(len(s), i) == check
            assert nP(s) == tot
        except AssertionError:
            print(s, i, 'failed perm test')
            raise ValueError()

    # 再次针对随机生成的字符串 s 进行多次测试
    for i in range(100):
        s = ''.join(choice(c) for i in range(7))
        u = len(s) == len(set(s))
        try:
            tot = 0
            # 测试 nC 函数在不同参数下的结果是否正确
            for i in range(8):
                check = nC(s, i)
                tot += check
                assert len(list(multiset_combinations(s, i))) == check
                if u:
                    assert nC(len(s), i) == check
            assert nC(s) == tot
            if u:
                assert nC(len(s)) == tot
        except AssertionError:
            print(s, i, 'failed combo test')
            raise ValueError()
    # 对于每个 i 在 1 到 9 的范围内进行迭代
    for i in range(1, 10):
        # 初始化总和为 0
        tot = 0
        # 对于每个 j 在 1 到 i+1 的范围内进行迭代
        for j in range(1, i + 2):
            # 调用 nT 函数计算 Stirling 数 (i, j)
            check = nT(i, j)
            # 断言检查 Stirling 数结果是否为整数
            assert check.is_Integer
            # 将 Stirling 数结果加到总和中
            tot += check
            # 断言检查具有特定大小的分割在分区函数中出现的次数是否等于 Stirling 数 (i, j)
            assert sum(1 for p in partitions(i, j, size=True) if p[0] == j) == check
        # 断言检查 nT 函数计算 Stirling 数 (i) 是否等于当前总和
        assert nT(i) == tot

    # 对于每个 i 在 1 到 9 的范围内进行迭代
    for i in range(1, 10):
        # 初始化总和为 0
        tot = 0
        # 对于每个 j 在 1 到 i+2 的范围内进行迭代
        for j in range(1, i + 2):
            # 调用 nT 函数计算 Stirling 数 (range(i), j)
            check = nT(range(i), j)
            # 将 Stirling 数结果加到总和中
            tot += check
            # 断言检查多重集分区函数返回的列表长度是否等于 Stirling 数 (range(i), j)
            assert len(list(multiset_partitions(list(range(i)), j))) == check
        # 断言检查 nT 函数计算 Stirling 数 (range(i)) 是否等于当前总和
        assert nT(range(i)) == tot

    # 对于每个 i 在 0 到 99 的范围内进行迭代
    for i in range(100):
        # 生成一个长度为 7 的随机字符串
        s = ''.join(choice(c) for i in range(7))
        # 判断生成的字符串是否是唯一字符构成的
        u = len(s) == len(set(s))
        try:
            # 初始化总和为 0
            tot = 0
            # 对于每个 i 在 1 到 8 的范围内进行迭代
            for i in range(1, 8):
                # 调用 nT 函数计算 Stirling 数 (s, i)
                check = nT(s, i)
                # 将 Stirling 数结果加到总和中
                tot += check
                # 断言检查多重集分区函数返回的列表长度是否等于 Stirling 数 (s, i)
                assert len(list(multiset_partitions(s, i))) == check
                # 如果 s 是唯一字符构成的字符串，断言检查 nT 函数计算 Stirling 数 (range(len(s)), i) 是否等于当前总和
                if u:
                    assert nT(range(len(s)), i) == check
            # 如果 s 是唯一字符构成的字符串，断言检查 nT 函数计算 Stirling 数 (range(len(s))) 是否等于当前总和
            if u:
                assert nT(range(len(s))) == tot
            # 断言检查 nT 函数计算 Stirling 数 (s) 是否等于当前总和
            assert nT(s) == tot
        except AssertionError:
            # 如果有断言失败，则打印相关信息并抛出 ValueError 异常
            print(s, i, 'failed partition test')
            raise ValueError()

    # 断言检查第一类 Stirling 数 (9, i) 的值是否符合预期
    assert [stirling(9, i, kind=1) for i in range(11)] == [
        0, 40320, 109584, 118124, 67284, 22449, 4536, 546, 36, 1, 0]
    perms = list(permutations(range(4)))
    # 断言检查排列的循环结构数量是否与 Stirling 数 (4, i) 的计算结果一致
    assert [sum(1 for p in perms if Permutation(p).cycles == i)
            for i in range(5)] == [0, 6, 11, 6, 1] == [
            stirling(4, i, kind=1) for i in range(5)]
    # 断言检查带符号的第一类 Stirling 数 (n, k) 的计算结果是否符合预期
    assert [stirling(n, k, signed=1)
        for n in range(10) for k in range(1, n + 1)] == [
            1, -1,
            1, 2, -3,
            1, -6, 11, -6,
            1, 24, -50, 35, -10,
            1, -120, 274, -225, 85, -15,
            1, 720, -1764, 1624, -735, 175, -21,
            1, -5040, 13068, -13132, 6769, -1960, 322, -28,
            1, 40320, -109584, 118124, -67284, 22449, -4536, 546, -36, 1]
    # 断言检查第一类 Stirling 数 (n, k) 的计算结果是否符合预期
    assert  [stirling(n, k, kind=1)
        for n in range(10) for k in range(n+1)] == [
            1,
            0, 1,
            0, 1, 1,
            0, 2, 3, 1,
            0, 6, 11, 6, 1,
            0, 24, 50, 35, 10, 1,
            0, 120, 274, 225, 85, 15, 1,
            0, 720, 1764, 1624, 735, 175, 21, 1,
            0, 5040, 13068, 13132, 6769, 1960, 322, 28, 1,
            0, 40320, 109584, 118124, 67284, 22449, 4536, 546, 36, 1]
    # 断言检查第二类 Stirling 数 (n, k) 的计算结果是否符合预期
    assert [stirling(n, k, kind=2)
        for n in range(10) for k in range(n+1)] == [
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 0, 0, 0,
            0, 1, 3, 1, 0, 0, 0,
            0, 1, 7, 6, 1, 0,
            0, 1, 15, 25, 10, 1,
            0, 1, 31, 90, 65, 15, 1,
            0, 1, 63, 301, 350, 140, 21, 1,
            0, 1, 127, 966, 1701, 1050, 266, 28, 1]
    # 使用 stirling 函数计算第二类斯特林数，验证结果列表是否符合预期
    assert [stirling(n, k, kind=2)
        for n in range(10) for k in range(n+1)] == [
            1,
            0, 1,
            0, 1, 1,
            0, 1, 3, 1,
            0, 1, 7, 6, 1,
            0, 1, 15, 25, 10, 1,
            0, 1, 31, 90, 65, 15, 1,
            0, 1, 63, 301, 350, 140, 21, 1,
            0, 1, 127, 966, 1701, 1050, 266, 28, 1,
            0, 1, 255, 3025, 7770, 6951, 2646, 462, 36, 1]
    # 使用 stirling 函数计算第一类斯特林数，验证两个相同参数输入是否返回相同结果
    assert stirling(3, 4, kind=1) == stirling(3, 4, kind=1) == 0
    # 验证 stirling 函数对于负数参数是否引发 ValueError 异常
    raises(ValueError, lambda: stirling(-2, 2))

    # 断言 _stirling1 返回类型为 SymPy 的 Integer
    assert isinstance(_stirling1(6, 3), Integer)
    # 断言 _stirling2 返回类型为 SymPy 的 Integer
    assert isinstance(_stirling2(6, 3), Integer)

    # 定义 delta 函数，计算列表 p 的所有子集中两两元素差的最小值
    def delta(p):
        if len(p) == 1:
            return oo
        return min(abs(i[0] - i[1]) for i in subsets(p, 2))
    # 计算集合 {0, 1, 2, 3, 4} 的多重集分割，每个分割中的最小差值要大于等于 d
    parts = multiset_partitions(range(5), 3)
    d = 2
    # 断言符合条件的分割数量等于 stirling(5, 3, d=d)，并且值为 7
    assert (sum(1 for p in parts if all(delta(i) >= d for i in p)) ==
            stirling(5, 3, d=d) == 7)

    # 其他覆盖测试
    assert nC('abb', 2) == nC('aab', 2) == 2
    assert nP(3, 3, replacement=True) == nP('aabc', 3, replacement=True) == 27
    assert nP(3, 4) == 0
    assert nP('aabc', 5) == 0
    assert nC(4, 2, replacement=True) == nC('abcdd', 2, replacement=True) == \
        len(list(multiset_combinations('aabbccdd', 2))) == 10
    assert nC('abcdd') == sum(nC('abcdd', i) for i in range(6)) == 24
    assert nC(list('abcdd'), 4) == 4
    assert nT('aaaa') == nT(4) == len(list(partitions(4))) == 5
    assert nT('aaab') == len(list(multiset_partitions('aaab'))) == 7
    assert nC('aabb'*3, 3) == 4  # aaa, bbb, abb, baa
    assert dict(_AOP_product((4,1,1,1))) == {
        0: 1, 1: 4, 2: 7, 3: 8, 4: 8, 5: 7, 6: 4, 7: 1}
    # 下面的测试是早期版本的函数中显示问题的第一个 t，因此不是完全随机选择
    t = (3, 9, 4, 6, 6, 5, 5, 2, 10, 4)
    assert sum(_AOP_product(t)[i] for i in range(55)) == 58212000
    # 断言 _multiset_histogram 函数对包含非整数键的字典输入引发 ValueError 异常
    raises(ValueError, lambda: _multiset_histogram({1:'a'}))
# 定义测试函数 test_PR_14617，用于验证 sympy 库中 nT 函数的行为
def test_PR_14617():
    # 从 sympy 库中导入 nT 函数
    from sympy.functions.combinatorial.numbers import nT
    # 针对 n 和 k 的组合进行循环测试
    for n in (0, []):
        for k in (-1, 0, 1):
            # 当 k 等于 0 时，验证 nT 函数的返回值是否为 1
            if k == 0:
                assert nT(n, k) == 1
            # 当 k 不等于 0 时，验证 nT 函数的返回值是否为 0
            else:
                assert nT(n, k) == 0


# 定义测试函数 test_issue_8496，用于验证在指定条件下是否会引发 TypeError
def test_issue_8496():
    # 导入符号 n 和 k，用于测试
    n = Symbol("n")
    k = Symbol("k")
    # 使用 lambda 函数验证 catalan 函数在给定 n 和 k 下是否会引发 TypeError
    raises(TypeError, lambda: catalan(n, k))


# 定义测试函数 test_issue_8601，用于验证 catalan 函数在不同输入下的行为
def test_issue_8601():
    # 定义符号 n，确保其为负整数
    n = Symbol('n', integer=True, negative=True)

    # 验证 catalan 函数在不同输入下的返回值
    assert catalan(n - 1) is S.Zero
    assert catalan(Rational(-1, 2)) is S.ComplexInfinity
    assert catalan(-S.One) == Rational(-1, 2)
    # 对 catalan 函数在负浮点数输入下进行数值计算验证
    c1 = catalan(-5.6).evalf()
    assert str(c1) == '6.93334070531408e-5'
    c2 = catalan(-35.4).evalf()
    assert str(c2) == '-4.14189164517449e-24'


# 定义测试函数 test_motzkin，用于验证 motzkin 模块中的多个功能
def test_motzkin():
    # 验证 motzkin.is_motzkin 函数在指定输入下是否返回 True 或 False
    assert motzkin.is_motzkin(4) == True
    assert motzkin.is_motzkin(9) == True
    assert motzkin.is_motzkin(10) == False
    # 验证 motzkin.find_motzkin_numbers_in_range 函数在给定范围内的输出结果
    assert motzkin.find_motzkin_numbers_in_range(10,200) == [21, 51, 127]
    assert motzkin.find_motzkin_numbers_in_range(10,400) == [21, 51, 127, 323]
    assert motzkin.find_motzkin_numbers_in_range(10,1600) == [21, 51, 127, 323, 835]
    # 验证 motzkin.find_first_n_motzkins 函数在给定数目下的输出结果
    assert motzkin.find_first_n_motzkins(5) == [1, 1, 2, 4, 9]
    assert motzkin.find_first_n_motzkins(7) == [1, 1, 2, 4, 9, 21, 51]
    assert motzkin.find_first_n_motzkins(10) == [1, 1, 2, 4, 9, 21, 51, 127, 323, 835]
    # 使用 lambda 函数验证 motzkin 模块中的异常处理
    raises(ValueError, lambda: motzkin.eval(77.58))
    raises(ValueError, lambda: motzkin.eval(-8))
    raises(ValueError, lambda: motzkin.find_motzkin_numbers_in_range(-2,7))
    raises(ValueError, lambda: motzkin.find_motzkin_numbers_in_range(13,7))
    raises(ValueError, lambda: motzkin.find_first_n_motzkins(112.8))


# 定义测试函数 test_nD_derangements，用于验证 nD 函数及其相关函数的行为
def test_nD_derangements():
    # 从 sympy.utilities.iterables 导入多个函数和 nD 函数
    from sympy.utilities.iterables import (partitions, multiset,
        multiset_derangements, multiset_permutations)
    from sympy.functions.combinatorial.numbers import nD

    # 初始化一个空列表，用于存储每次迭代的结果
    got = []
    # 对 partitions(8, k=4) 产生的分区进行循环处理
    for i in partitions(8, k=4):
        s = []
        it = 0
        # 遍历分区中的元素及其出现次数，生成多重集 s
        for k, v in i.items():
            for _ in range(v):
                s.extend([it]*k)
                it += 1
        ms = multiset(s)
        # 计算 multiset_permutations(s) 中不同于 s 的排列数目，并验证 nD 函数的返回值
        c1 = sum(1 for i in multiset_permutations(s) if all(i != j for i, j in zip(i, s)))
        assert c1 == nD(ms) == nD(ms, 0) == nD(ms, 1)
        # 计算 multiset_derangements(s) 的结果，并验证其与 c1 的长度是否一致
        v = [tuple(i) for i in multiset_derangements(s)]
        c2 = len(v)
        assert c2 == len(set(v))
        assert c1 == c2
        # 将每次迭代的结果添加到 got 列表中
        got.append(c1)
    # 验证最终得到的 got 列表是否与预期一致
    assert got == [1, 4, 6, 12, 24, 24, 61, 126, 315, 780, 297, 772,
        2033, 5430, 14833]

    # 验证 nD 函数在不同输入下的行为
    assert nD('1112233456', brute=True) == nD('1112233456') == 16356
    assert nD('') == nD([]) == nD({}) == 0
    assert nD({1: 0}) == 0
    raises(ValueError, lambda: nD({1: -1}))
    assert nD('112') == 0
    assert nD(i='112') == 0
    assert [nD(n=i) for i in range(6)] == [0, 0, 1, 2, 9, 44]
    assert nD((i for i in range(4))) == nD('0123') == 9
    assert nD(m=(i for i in range(4))) == 3
    assert nD(m={0: 1, 1: 1, 2: 1, 3: 1}) == 3
    assert nD(m=[0, 1, 2, 3]) == 3
    # 调用 nD 函数并验证抛出 TypeError 异常，使用 lambda 匿名函数捕获异常
    raises(TypeError, lambda: nD(m=0))
    # 调用 nD 函数并验证抛出 TypeError 异常，使用 lambda 匿名函数捕获异常
    raises(TypeError, lambda: nD(-1))
    # 断言调用 nD 函数返回值为 1，传入字典作为参数
    assert nD({-1: 1, -2: 1}) == 1
    # 断言调用 nD 函数返回值为 0，传入字典作为参数
    assert nD(m={0: 3}) == 0
    # 调用 nD 函数并验证抛出 ValueError 异常，使用 lambda 匿名函数捕获异常
    raises(ValueError, lambda: nD(i='123', n=3))
    # 调用 nD 函数并验证抛出 ValueError 异常，使用 lambda 匿名函数捕获异常
    raises(ValueError, lambda: nD(i='123', m=(1,2)))
    # 调用 nD 函数并验证抛出 ValueError 异常，使用 lambda 匿名函数捕获异常
    raises(ValueError, lambda: nD(n=0, m=(1,2)))
    # 调用 nD 函数并验证抛出 ValueError 异常，使用 lambda 匿名函数捕获异常
    raises(ValueError, lambda: nD({1: -1}))
    # 调用 nD 函数并验证抛出 ValueError 异常，使用 lambda 匿名函数捕获异常
    raises(ValueError, lambda: nD(m={-1: 1, 2: 1}))
    # 调用 nD 函数并验证抛出 ValueError 异常，使用 lambda 匿名函数捕获异常
    raises(ValueError, lambda: nD(m={1: -1, 2: 1}))
    # 调用 nD 函数并验证抛出 ValueError 异常，使用 lambda 匿名函数捕获异常
    raises(ValueError, lambda: nD(m=[-1, 2]))
    # 调用 nD 函数并验证抛出 TypeError 异常，使用 lambda 匿名函数捕获异常
    raises(TypeError, lambda: nD({1: x}))
    # 调用 nD 函数并验证抛出 TypeError 异常，使用 lambda 匿名函数捕获异常
    raises(TypeError, lambda: nD(m={1: x}))
    # 调用 nD 函数并验证抛出 TypeError 异常，使用 lambda 匿名函数捕获异常
    raises(TypeError, lambda: nD(m={x: 1}))
# 定义一个测试函数，用于测试已弃用的数论符号函数的行为
def test_deprecated_ntheory_symbolic_functions():
    # 从 sympy.testing.pytest 模块导入 warns_deprecated_sympy 函数，用于捕获弃用警告
    from sympy.testing.pytest import warns_deprecated_sympy

    # 使用 warns_deprecated_sympy 上下文管理器捕获弃用警告，并执行以下断言
    with warns_deprecated_sympy():
        # 断言 carmichael.is_carmichael(3) 返回 False
        assert not carmichael.is_carmichael(3)
    # 使用 warns_deprecated_sympy 上下文管理器捕获弃用警告，并执行以下断言
    with warns_deprecated_sympy():
        # 断言 carmichael.find_carmichael_numbers_in_range(10, 20) 返回一个空列表
        assert carmichael.find_carmichael_numbers_in_range(10, 20) == []
    # 使用 warns_deprecated_sympy 上下文管理器捕获弃用警告，并执行以下断言
    with warns_deprecated_sympy():
        # 断言 carmichael.find_first_n_carmichaels(1) 返回 True 或非空结果
        assert carmichael.find_first_n_carmichaels(1)
```