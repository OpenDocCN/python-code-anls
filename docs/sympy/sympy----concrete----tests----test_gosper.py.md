# `D:\src\scipysrc\sympy\sympy\concrete\tests\test_gosper.py`

```
"""Tests for Gosper's algorithm for hypergeometric summation. """

# 导入所需的 SymPy 模块和函数
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.polytools import Poly
from sympy.simplify.simplify import simplify
from sympy.concrete.gosper import gosper_normal, gosper_sum, gosper_term
from sympy.abc import a, b, j, k, m, n, r, x

# 定义测试函数 test_gosper_normal，测试 gosper_normal 函数的返回值
def test_gosper_normal():
    eq = 4*n + 5, 2*(4*n + 1)*(2*n + 3), n
    # 验证带有 polys=True 参数的 gosper_normal 函数返回值
    assert gosper_normal(*eq) == \
        (Poly(Rational(1, 4), n), Poly(n + Rational(3, 2)), Poly(n + Rational(1, 4)))
    # 验证带有 polys=False 参数的 gosper_normal 函数返回值
    assert gosper_normal(*eq, polys=False) == \
        (Rational(1, 4), n + Rational(3, 2), n + Rational(1, 4))

# 定义测试函数 test_gosper_term，测试 gosper_term 函数的返回值
def test_gosper_term():
    # 验证 gosper_term 函数对给定表达式的求和结果
    assert gosper_term((4*k + 1)*factorial(
        k)/factorial(2*k + 1), k) == (-k - S.Half)/(k + Rational(1, 4))

# 定义测试函数 test_gosper_sum，测试 gosper_sum 函数的多种情况
def test_gosper_sum():
    # 验证 gosper_sum 函数对常数 1 的求和结果
    assert gosper_sum(1, (k, 0, n)) == 1 + n
    # 验证 gosper_sum 函数对 k 的求和结果
    assert gosper_sum(k, (k, 0, n)) == n*(1 + n)/2
    # 验证 gosper_sum 函数对 k^2 的求和结果
    assert gosper_sum(k**2, (k, 0, n)) == n*(1 + n)*(1 + 2*n)/6
    # 验证 gosper_sum 函数对 k^3 的求和结果
    assert gosper_sum(k**3, (k, 0, n)) == n**2*(1 + n)**2/4

    # 验证 gosper_sum 函数对指数函数 2^k 的求和结果
    assert gosper_sum(2**k, (k, 0, n)) == 2*2**n - 1

    # 验证 gosper_sum 函数对阶乘 factorial(k) 的求和结果
    assert gosper_sum(factorial(k), (k, 0, n)) is None
    # 验证 gosper_sum 函数对二项式系数 binomial(n, k) 的求和结果
    assert gosper_sum(binomial(n, k), (k, 0, n)) is None

    # 验证 gosper_sum 函数对 k*factorial(k)/k^2 的求和结果
    assert gosper_sum(factorial(k)/k**2, (k, 0, n)) is None
    # 验证 gosper_sum 函数对 (k - 3)*factorial(k) 的求和结果
    assert gosper_sum((k - 3)*factorial(k), (k, 0, n)) is None

    # 验证 gosper_sum 函数对 k*factorial(k) 的不定积分结果
    assert gosper_sum(k*factorial(k), k) == factorial(k)
    # 验证 gosper_sum 函数对 k*factorial(k) 的定积分结果
    assert gosper_sum(
        k*factorial(k), (k, 0, n)) == n*factorial(n) + factorial(n) - 1

    # 验证 gosper_sum 函数对 (-1)^k*binomial(n, k) 的求和结果
    assert gosper_sum((-1)**k*binomial(n, k), (k, 0, n)) == 0
    # 验证 gosper_sum 函数对 (-1)^k*binomial(n, k) 的一般求和结果
    assert gosper_sum((
        -1)**k*binomial(n, k), (k, 0, m)) == -(-1)**m*(m - n)*binomial(n, m)/n

    # 验证 gosper_sum 函数对 (4*k + 1)*factorial(k)/factorial(2*k + 1) 的求和结果
    assert gosper_sum((4*k + 1)*factorial(k)/factorial(2*k + 1), (k, 0, n)) == \
        (2*factorial(2*n + 1) - factorial(n))/factorial(2*n + 1)

    # 验证 gosper_sum 函数对复杂表达式的求和结果
    assert gosper_sum(
        n*(n + a + b)*a**n*b**n/(factorial(n + a)*factorial(n + b)), \
        (n, 0, m)).simplify() == -exp(m*log(a) + m*log(b))*gamma(a + 1) \
        *gamma(b + 1)/(gamma(a)*gamma(b)*gamma(a + m + 1)*gamma(b + m + 1)) \
        + 1/(gamma(a)*gamma(b))

# 定义测试函数 test_gosper_sum_indefinite，测试 gosper_sum 函数的不定积分情况
def test_gosper_sum_indefinite():
    # 验证 gosper_sum 函数对 k 的不定积分结果
    assert gosper_sum(k, k) == k*(k - 1)/2
    # 验证 gosper_sum 函数对 k^2 的不定积分结果
    assert gosper_sum(k**2, k) == k*(k - 1)*(2*k - 1)/6

    # 验证 gosper_sum 函数对 1/(k*(k + 1)) 的不定积分结果
    assert gosper_sum(1/(k*(k + 1)), k) == -1/k
    # 验证 gosper_sum 函数对复杂表达式的不定积分结果
    assert gosper_sum(-(27*k**4 + 158*k**3 + 430*k**2 + 678*k + 445)*gamma(2*k
                      + 4)/(3*(3*k + 7)*gamma(3*k + 6)), k) == \
        (3*k + 5)*(k**2 + 2*k + 5)*gamma(2*k + 4)/gamma(3*k + 6)


def test_gosper_sum_parametric():
    # 使用高斯-戈斯珀和（Gosper's algorithm）计算给定表达式的和
    assert gosper_sum(
        # 计算二项式系数 binomial(S.Half, m - j + 1) 和 binomial(S.Half, m + j)
        binomial(S.Half, m - j + 1) * binomial(S.Half, m + j),
        # 求和变量 j 的范围为 1 到 n
        (j, 1, n)
    ) == \
        # 计算表达式右侧的结果
        n * (1 + m - n) * (-1 + 2 * m + 2 * n) * binomial(S.Half, 1 + m - n) * \
        binomial(S.Half, m + n) / (m * (1 + 2 * m))
# 定义测试函数，测试 gosper_sum 函数在代数表达式上的计算结果是否正确
def test_gosper_sum_algebraic():
    # 断言：对 gosper_sum 函数应用代数表达式 n**2 + sqrt(2)，范围为 (n, 0, m)，期望结果等于 (m + 1)*(2*m**2 + m + 6*sqrt(2))/6
    assert gosper_sum(
        n**2 + sqrt(2), (n, 0, m)) == (m + 1)*(2*m**2 + m + 6*sqrt(2))/6


def test_gosper_sum_iterated():
    # 计算各个表达式
    f1 = binomial(2*k, k)/4**k
    f2 = (1 + 2*n)*binomial(2*n, n)/4**n
    f3 = (1 + 2*n)*(3 + 2*n)*binomial(2*n, n)/(3*4**n)
    f4 = (1 + 2*n)*(3 + 2*n)*(5 + 2*n)*binomial(2*n, n)/(15*4**n)
    f5 = (1 + 2*n)*(3 + 2*n)*(5 + 2*n)*(7 + 2*n)*binomial(2*n, n)/(105*4**n)

    # 逐一断言 gosper_sum 函数的计算结果是否符合预期
    assert gosper_sum(f1, (k, 0, n)) == f2
    assert gosper_sum(f2, (n, 0, n)) == f3
    assert gosper_sum(f3, (n, 0, n)) == f4
    assert gosper_sum(f4, (n, 0, n)) == f5


# AeqB 测试，测试表达式在 www.math.upenn.edu/~wilf/AeqB.pdf 中给出的表达式
def test_gosper_sum_AeqB_part1():
    # 定义各个表达式 f1a 到 f1h 和 g1a 到 g1h
    f1a = n**4
    f1b = n**3*2**n
    f1c = 1/(n**2 + sqrt(5)*n - 1)
    f1d = n**4*4**n/binomial(2*n, n)
    f1e = factorial(3*n)/(factorial(n)*factorial(n + 1)*factorial(n + 2)*27**n)
    f1f = binomial(2*n, n)**2/((n + 1)*4**(2*n))
    f1g = (4*n - 1)*binomial(2*n, n)**2/((2*n - 1)**2*4**(2*n))
    f1h = n*factorial(n - S.Half)**2/factorial(n + 1)**2

    g1a = m*(m + 1)*(2*m + 1)*(3*m**2 + 3*m - 1)/30
    g1b = 26 + 2**(m + 1)*(m**3 - 3*m**2 + 9*m - 13)
    g1c = (m + 1)*(m*(m**2 - 7*m + 3)*sqrt(5) - (
        3*m**3 - 7*m**2 + 19*m - 6))/(2*m**3*sqrt(5) + m**4 + 5*m**2 - 1)/6
    g1d = Rational(-2, 231) + 2*4**m*(m + 1)*(63*m**4 + 112*m**3 + 18*m**2 -
             22*m + 3)/(693*binomial(2*m, m))
    g1e = Rational(-9, 2) + (81*m**2 + 261*m + 200)*factorial(
        3*m + 2)/(40*27**m*factorial(m)*factorial(m + 1)*factorial(m + 2))
    g1f = (2*m + 1)**2*binomial(2*m, m)**2/(4**(2*m)*(m + 1))
    g1g = -binomial(2*m, m)**2/4**(2*m)
    g1h = 4*pi -(2*m + 1)**2*(3*m + 4)*factorial(m - S.Half)**2/factorial(m + 1)**2

    # 对每个表达式调用 gosper_sum 函数，断言计算结果是否为预期值
    g = gosper_sum(f1a, (n, 0, m))
    assert g is not None and simplify(g - g1a) == 0
    g = gosper_sum(f1b, (n, 0, m))
    assert g is not None and simplify(g - g1b) == 0
    g = gosper_sum(f1c, (n, 0, m))
    assert g is not None and simplify(g - g1c) == 0
    g = gosper_sum(f1d, (n, 0, m))
    assert g is not None and simplify(g - g1d) == 0
    g = gosper_sum(f1e, (n, 0, m))
    assert g is not None and simplify(g - g1e) == 0
    g = gosper_sum(f1f, (n, 0, m))
    assert g is not None and simplify(g - g1f) == 0
    g = gosper_sum(f1g, (n, 0, m))
    assert g is not None and simplify(g - g1g) == 0
    g = gosper_sum(f1h, (n, 0, m))
    # 需要在此处调用 rewrite(gamma)，因为涉及到阶乘(1/2)的项
    assert g is not None and simplify(g - g1h).rewrite(gamma) == 0


def test_gosper_sum_AeqB_part2():
    # 定义表达式 f2a 到 f2c 和对应的预期值 g2a 到 g2b
    f2a = n**2*a**n
    f2b = (n - r/2)*binomial(r, n)
    f2c = factorial(n - 1)**2/(factorial(n - x)*factorial(n + x))

    g2a = -a*(a + 1)/(a - 1)**3 + a**(
        m + 1)*(a**2*m**2 - 2*a*m**2 + m**2 - 2*a*m + 2*m + a + 1)/(a - 1)**3
    g2b = (m - r)*binomial(r, m)/2

    # ff 是一个未使用的表达式，没有相关断言
    ff = factorial(1 - x)*factorial(1 + x)
    # 计算 g2c 的值，这是一个复杂的表达式，涉及到 ff, x, m 的计算和阶乘操作
    g2c = 1/ff*(1 - 1/x**2) + factorial(m)**2/(x**2*factorial(m - x)*factorial(m + x))

    # 使用 gosper_sum 函数计算 f2a 的高斯-伯勒和，n 的范围是 0 到 m
    g = gosper_sum(f2a, (n, 0, m))
    # 断言 g 不为 None，并且简化后 g 和 g2a 相等
    assert g is not None and simplify(g - g2a) == 0

    # 使用 gosper_sum 函数计算 f2b 的高斯-伯勒和，n 的范围是 0 到 m
    g = gosper_sum(f2b, (n, 0, m))
    # 断言 g 不为 None，并且简化后 g 和 g2b 相等
    assert g is not None and simplify(g - g2b) == 0

    # 使用 gosper_sum 函数计算 f2c 的高斯-伯勒和，n 的范围是 1 到 m
    g = gosper_sum(f2c, (n, 1, m))
    # 断言 g 不为 None，并且简化后 g 和 g2c 相等
    assert g is not None and simplify(g - g2c) == 0


这段代码计算了几个表达式的值，并使用 `gosper_sum` 函数来计算高斯-伯勒和。每个表达式计算后，使用断言来验证计算的结果与预期的结果是否一致。
# 定义测试函数 test_gosper_nan，用于测试高斯-波斯特求和算法
def test_gosper_nan():
    # 定义符号变量 a，b，n，m，其中 a 和 b 是正整数，n 和 m 是整数
    a = Symbol('a', positive=True)
    b = Symbol('b', positive=True)
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True)
    
    # 定义二维函数 f2d，表示为 n*(n + a + b)*a**n*b**n/(factorial(n + a)*factorial(n + b))
    f2d = n*(n + a + b)*a**n*b**n/(factorial(n + a)*factorial(n + b))
    
    # 定义二维函数 g2d，表示为 1/(factorial(a - 1)*factorial(b - 1)) - a**(m + 1)*b**(m + 1)/(factorial(a + m)*factorial(b + m))
    g2d = 1/(factorial(a - 1)*factorial(b - 1)) - a**(m + 1)*b**(m + 1)/(factorial(a + m)*factorial(b + m))
    
    # 使用高斯-波斯特求和算法计算函数 f2d 在变量 n 从 0 到 m 的和，并将结果赋给变量 g
    g = gosper_sum(f2d, (n, 0, m))
    
    # 断言简化后的 g 与 g2d 相等
    assert simplify(g - g2d) == 0


# 定义测试函数 test_gosper_sum_AeqB_part3
def test_gosper_sum_AeqB_part3():
    # 定义各个函数 f3a 到 f3g，每个函数的具体含义如下：
    f3a = 1/n**4
    f3b = (6*n + 3)/(4*n**4 + 8*n**3 + 8*n**2 + 4*n + 3)
    f3c = 2**n*(n**2 - 2*n - 1)/(n**2*(n + 1)**2)
    f3d = n**2*4**n/((n + 1)*(n + 2))
    f3e = 2**n/(n + 1)
    f3f = 4*(n - 1)*(n**2 - 2*n - 1)/(n**2*(n + 1)**2*(n - 2)**2*(n - 3)**2)
    f3g = (n**4 - 14*n**2 - 24*n - 9)*2**n/(n**2*(n + 1)**2*(n + 2)**2*(n + 3)**2)
    
    # 定义 g3b 到 g3g，每个变量的含义如下：
    g3b = m*(m + 2)/(2*m**2 + 4*m + 3)
    g3c = 2**m/m**2 - 2
    g3d = Rational(2, 3) + 4**(m + 1)*(m - 1)/(m + 2)/3
    g3f = -(Rational(-1, 16) + 1/((m - 2)**2*(m + 1)**2))  # 这里的 AeqB 键值错误
    g3g = Rational(-2, 9) + 2**(m + 1)/((m + 1)**2*(m + 3)**2)
    
    # 使用高斯-波斯特求和算法计算函数 f3a 到 f3g 在指定区间内的和，并分别与 g3b 到 g3g 做断言比较
    g = gosper_sum(f3a, (n, 1, m))
    assert g is None
    g = gosper_sum(f3b, (n, 1, m))
    assert g is not None and simplify(g - g3b) == 0
    g = gosper_sum(f3c, (n, 1, m - 1))
    assert g is not None and simplify(g - g3c) == 0
    g = gosper_sum(f3d, (n, 1, m))
    assert g is not None and simplify(g - g3d) == 0
    g = gosper_sum(f3e, (n, 0, m - 1))
    assert g is None
    g = gosper_sum(f3f, (n, 4, m))
    assert g is not None and simplify(g - g3f) == 0
    g = gosper_sum(f3g, (n, 1, m))
    assert g is not None and simplify(g - g3g) == 0
```