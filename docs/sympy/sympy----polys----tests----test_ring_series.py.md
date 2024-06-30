# `D:\src\scipysrc\sympy\sympy\polys\tests\test_ring_series.py`

```
from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
    rs_trunc, rs_mul, rs_square, rs_pow, _has_constant_term, rs_hadamard_exp,
    rs_series_from_list, rs_exp, rs_log, rs_newton, rs_series_inversion,
    rs_compose_add, rs_asin, rs_atan, rs_atanh, rs_tan, rs_cot, rs_sin, rs_cos,
    rs_cos_sin, rs_sinh, rs_cosh, rs_tanh, _tan1, rs_fun, rs_nth_root,
    rs_LambertW, rs_series_reversion, rs_is_puiseux, rs_series)
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
    tanh, log, sqrt)
from sympy.core.numbers import Rational
from sympy.core import expand, S

# 自定义函数，用于检查两个数值是否足够接近
def is_close(a, b):
    tol = 10**(-10)
    assert abs(a - b) < tol

# 测试函数，测试 ring 和 ring_series 模块的一些功能
def test_ring_series1():
    # 创建一个有理数环 R，并定义符号 x
    R, x = ring('x', QQ)
    # 定义多项式 p
    p = x**4 + 2*x**3 + 3*x + 4
    # 测试 _invert_monoms 函数是否按预期工作
    assert _invert_monoms(p) == 4*x**4 + 3*x**3 + 2*x + 1
    # 测试 rs_hadamard_exp 函数是否按预期工作
    assert rs_hadamard_exp(p) == x**4/24 + x**3/3 + 3*x + 4
    
    # 重新定义有理数环 R 和符号 x，重复之前的测试
    R, x = ring('x', QQ)
    p = x**4 + 2*x**3 + 3*x + 4
    # 测试 rs_integrate 函数是否按预期工作
    assert rs_integrate(p, x) == x**5/5 + x**4/2 + 3*x**2/2 + 4*x
    
    # 重新定义有理数环 R，并定义符号 x 和 y
    R, x, y = ring('x, y', QQ)
    p = x**2*y**2 + x + 1
    # 测试 rs_integrate 函数在 x 和 y 上的表现
    assert rs_integrate(p, x) == x**3*y**2/3 + x**2/2 + x
    assert rs_integrate(p, y) == x**2*y**3/3 + x*y + y

# 测试函数，测试 rs_trunc 函数
def test_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = (y + t*x)**4
    # 测试 rs_trunc 函数是否按预期工作
    p1 = rs_trunc(p, x, 3)
    assert p1 == y**4 + 4*y**3*t*x + 6*y**2*t**2*x**2

# 测试函数，测试 rs_mul 函数和 rs_square 函数
def test_mul_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = 1 + t*x + t*y
    for i in range(2):
        p = rs_mul(p, p, t, 3)
    
    # 测试 rs_mul 函数是否按预期工作
    assert p == 6*x**2*t**2 + 12*x*y*t**2 + 6*y**2*t**2 + 4*x*t + 4*y*t + 1
    p = 1 + t*x + t*y + t**2*x*y
    p1 = rs_mul(p, p, t, 2)
    assert p1 == 1 + 2*t*x + 2*t*y
    
    # 创建有理数环 R1 和符号 z，测试 rs_mul 函数抛出异常的情况
    R1, z = ring('z', QQ)
    raises(ValueError, lambda: rs_mul(p, z, x, 2))
    
    p1 = 2 + 2*x + 3*x**2
    p2 = 3 + x**2
    # 测试 rs_mul 函数在 p1 和 p2 上的表现
    assert rs_mul(p1, p2, x, 4) == 2*x**3 + 11*x**2 + 6*x + 6

# 测试函数，测试 rs_square 函数
def test_square_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = (1 + t*x + t*y)*2
    p1 = rs_mul(p, p, x, 3)
    p2 = rs_square(p, x, 3)
    # 测试 rs_square 函数是否按预期工作
    assert p1 == p2
    p = 1 + x + x**2 + x**3
    assert rs_square(p, x, 4) == 4*x**3 + 3*x**2 + 2*x + 1

# 测试函数，测试 rs_pow 函数
def test_pow_trunc():
    R, x, y, z = ring('x, y, z', QQ)
    p0 = y + x*z
    p = p0**16
    for xx in (x, y, z):
        p1 = rs_trunc(p, xx, 8)
        p2 = rs_pow(p0, 16, xx, 8)
        # 测试 rs_pow 函数是否按预期工作
        assert p1 == p2
    
    p = 1 + x
    p1 = rs_pow(p, 3, x, 2)
    assert p1 == 1 + 3*x
    assert rs_pow(p, 0, x, 2) == 1
    assert rs_pow(p, -2, x, 2) == 1 - 2*x
    p = x + y
    assert rs_pow(p, 3, y, 3) == x**3 + 3*x**2*y + 3*x*y**2
    assert rs_pow(1 + x, Rational(2, 3), x, 4) == 4*x**3/81 - x**2/9 + x*Rational(2, 3) + 1

# 测试函数，测试 _has_constant_term 函数
def test_has_constant_term():
    R, x, y, z = ring('x, y, z', QQ)
    p = y + x*z
    # 测试 _has_constant_term 函数是否按预期工作
    assert _has_constant_term(p, x)
    p = x + x**4
    assert not _has_constant_term(p, x)
    p = 1 + x + x**4
    assert _has_constant_term(p, x)
    p = x + y + x*z
def test_inversion():
    # 定义一个有理数环 R 和一个变量 x
    R, x = ring('x', QQ)
    # 定义多项式 p = 2 + x + 2*x**2
    p = 2 + x + 2*x**2
    # 设置 n 的值为 5
    n = 5
    # 计算 p 的逆序列 p1
    p1 = rs_series_inversion(p, x, n)
    # 断言 p 乘以 p1 的截断系数与 1 相等
    assert rs_trunc(p*p1, x, n) == 1
    # 重新定义有理数环 R，添加变量 y
    R, x, y = ring('x, y', QQ)
    # 定义多项式 p = 2 + x + 2*x**2 + y*x + x**2*y
    p = 2 + x + 2*x**2 + y*x + x**2*y
    # 计算 p 的逆序列 p1
    p1 = rs_series_inversion(p, x, n)
    # 断言 p 乘以 p1 的截断系数与 1 相等
    assert rs_trunc(p*p1, x, n) == 1

    # 重新定义有理数环 R，添加变量 y
    R, x, y = ring('x, y', QQ)
    # 定义多项式 p = 1 + x + y
    p = 1 + x + y
    # 断言调用 rs_series_inversion(p, x, 4) 会引发 NotImplementedError
    raises(NotImplementedError, lambda: rs_series_inversion(p, x, 4))
    # 定义零多项式 p
    p = R.zero
    # 断言调用 rs_series_inversion(p, x, 3) 会引发 ZeroDivisionError
    raises(ZeroDivisionError, lambda: rs_series_inversion(p, x, 3))


def test_series_reversion():
    # 定义有理数环 R 和变量 x, y
    R, x, y = ring('x, y', QQ)

    # 计算 rs_tan(x, x, 10) 的逆序列，并断言其与 rs_atan(y, y, 8) 相等
    p = rs_tan(x, x, 10)
    assert rs_series_reversion(p, x, 8, y) == rs_atan(y, y, 8)

    # 计算 rs_sin(x, x, 10) 的逆序列，并断言其与 5*y**7/112 + 3*y**5/40 + y**3/6 + y 相等
    p = rs_sin(x, x, 10)
    assert rs_series_reversion(p, x, 8, y) == 5*y**7/112 + 3*y**5/40 + y**3/6 + y


def test_series_from_list():
    # 定义有理数环 R 和变量 x
    R, x = ring('x', QQ)
    # 定义多项式 p = 1 + 2*x + x**2 + 3*x**3
    p = 1 + 2*x + x**2 + 3*x**3
    # 定义系数列表 c = [1, 2, 0, 4, 4]
    c = [1, 2, 0, 4, 4]
    # 使用 rs_series_from_list 计算多项式 r，并断言其与 r1 相等
    r = rs_series_from_list(p, c, x, 5)
    pc = R.from_list(list(reversed(c)))
    r1 = rs_trunc(pc.compose(x, p), x, 5)
    assert r == r1
    # 重新定义有理数环 R，添加变量 y
    R, x, y = ring('x, y', QQ)
    # 定义系数列表 c = [1, 3, 5, 7]
    c = [1, 3, 5, 7]
    # 使用 rs_series_from_list 计算多项式 p1，并断言其与 p2 相等
    p1 = rs_series_from_list(x + y, c, x, 3, concur=0)
    p2 = rs_trunc((1 + 3*(x+y) + 5*(x+y)**2 + 7*(x+y)**3), x, 3)
    assert p1 == p2

    # 重新定义有理数环 R 和变量 x
    R, x = ring('x', QQ)
    # 定义 h = 25
    h = 25
    # 计算 rs_exp(x, x, h) - 1 的逆序列，并断言其与 p2 相等
    p = rs_exp(x, x, h) - 1
    p1 = rs_series_from_list(p, c, x, h)
    p2 = 0
    for i, cx in enumerate(c):
        p2 += cx*rs_pow(p, i, x, h)
    assert p1 == p2


def test_log():
    # 定义有理数环 R 和变量 x
    R, x = ring('x', QQ)
    # 定义多项式 p = 1 + x
    p = 1 + x
    # 计算 rs_log(p, x, 4)/x**2，并断言其与 Rational(1, 3)*x - S.Half + x**(-1) 相等
    p1 = rs_log(p, x, 4)/x**2
    assert p1 == Rational(1, 3)*x - S.Half + x**(-1)
    # 定义多项式 p = 1 + x + 2*x**2/3
    p = 1 + x + 2*x**2/3
    # 计算 rs_log(p, x, 9)，并断言其与给定表达式相等
    p1 = rs_log(p, x, 9)
    assert p1 == -17*x**8/648 + 13*x**7/189 - 11*x**6/162 - x**5/45 + 7*x**4/36 - x**3/3 + x**2/6 + x
    # 计算 p 的逆序列 p2，并计算 rs_log(p2, x, 9)，断言其与 -p1 相等

    R, x, y = ring('x, y', QQ)
    p = 1 + x + 2*y*x**2
    p1 = rs_log(p, x, 6)
    assert p1 == (4*x**5*y**2 - 2*x**5*y - 2*x**4*y**2 + x**5/5 + 2*x**4*y - x**4/4 - 2*x**3*y + x**3/3 + 2*x**2*y - x**2/2 + x)

    # 在系列中的常量项
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_log(x + a, x, 5) == -EX(1/(4*a**4))*x**4 + EX(1/(3*a**3))*x**3 - EX(1/(2*a**2))*x**2 + EX(1/a)*x + EX(log(a))
    assert rs_log(x + x**2*y + a, x, 4) == -EX(a**(-2))*x**3*y + EX(1/(3*a**3))*x**3 + EX(1/a)*x**2*y - EX(1/(2*a**2))*x**2 + EX(1/a)*x + EX(log(a))

    # 定义多项式 p = x + x**2 + 3
    p = x + x**2 + 3
    # 计算 rs_log(p, x, 10).compose(x, 5)，并断言其与给定表达式相等
    assert rs_log(p, x, 10).compose(x, 5) == EX(log(3) + Rational(19281291595, 9920232))


def test_exp():
    # 定义有理数环 R 和变量 x
    R, x = ring('x', QQ)
    # 定义多项式 p = x + x**4
    p = x + x**4
    # 循环计算不同精度 h 下的 rs_exp(p, x, h)，并断言其与 rs_log(p1, x, h) 相等
    for h in [10, 30]:
        q = rs_series_inversion(1 + p, x, h) - 1
        p1 = rs_exp(q, x, h)
    # 创建符号变量 'a'
    a = symbols('a')
    # 在有理数域 QQ[exp(a), a] 上创建多项式环 R，同时定义变量 x 和 y
    R, x, y = ring('x, y', QQ[exp(a), a])
    # 断言语句：验证 rs_exp 函数对于 x + a 的展开结果是否正确
    assert rs_exp(x + a, x, 5) == exp(a)*x**4/24 + exp(a)*x**3/6 + \
        exp(a)*x**2/2 + exp(a)*x + exp(a)
    # 断言语句：验证 rs_exp 函数对于 x + x**2*y + a 的展开结果是否正确
    assert rs_exp(x + x**2*y + a, x, 5) == exp(a)*x**4*y**2/2 + \
            exp(a)*x**4*y/2 + exp(a)*x**4/24 + exp(a)*x**3*y + \
            exp(a)*x**3/6 + exp(a)*x**2*y + exp(a)*x**2/2 + exp(a)*x + exp(a)

    # 在扩展域 EX 上重新定义多项式环 R，同时定义变量 x 和 y
    R, x, y = ring('x, y', EX)
    # 断言语句：验证 rs_exp 函数对于 x + a 的展开结果是否正确，使用扩展域 EX
    assert rs_exp(x + a, x, 5) ==  EX(exp(a)/24)*x**4 + EX(exp(a)/6)*x**3 + \
        EX(exp(a)/2)*x**2 + EX(exp(a))*x + EX(exp(a))
    # 断言语句：验证 rs_exp 函数对于 x + x**2*y + a 的展开结果是否正确，使用扩展域 EX
    assert rs_exp(x + x**2*y + a, x, 5) == EX(exp(a)/2)*x**4*y**2 + \
        EX(exp(a)/2)*x**4*y + EX(exp(a)/24)*x**4 + EX(exp(a))*x**3*y + \
        EX(exp(a)/6)*x**3 + EX(exp(a))*x**2*y + EX(exp(a)/2)*x**2 + \
        EX(exp(a))*x + EX(exp(a))
# 定义一个测试函数，测试牛顿迭代法在给定环境下的表现
def test_newton():
    # 创建有理数域 R 和变量 x
    R, x = ring('x', QQ)
    # 定义多项式 p = x^2 - 2
    p = x**2 - 2
    # 使用牛顿迭代法计算多项式 p 在 x = 4 处的结果
    r = rs_newton(p, x, 4)
    # 断言计算结果与预期值相等
    assert r == 8*x**4 + 4*x**2 + 2

# 定义一个测试函数，测试多项式的组合加法
def test_compose_add():
    # 创建有理数域 R 和变量 x
    R, x = ring('x', QQ)
    # 定义两个多项式 p1 = x^3 - 1 和 p2 = x^2 - 2
    p1 = x**3 - 1
    p2 = x**2 - 2
    # 断言两个多项式的组合加法结果与预期值相等
    assert rs_compose_add(p1, p2) == x**6 - 6*x**4 - 2*x**3 + 12*x**2 - 12*x - 7

# 定义一个测试函数，测试多元多项式在指定函数上的操作
def test_fun():
    # 创建有理数域 R 和变量 x, y
    R, x, y = ring('x, y', QQ)
    # 定义多元多项式 p = x*y + x^2*y^3 + x^5*y
    p = x*y + x**2*y**3 + x**5*y
    # 断言 rs_fun 和 rs_tan 在多项式 p, 函数 rs_tan, 变量 x, 和次数 10 下的结果相等
    assert rs_fun(p, rs_tan, x, 10) == rs_tan(p, x, 10)
    # 断言 rs_fun 和 _tan1 在多项式 p, 函数 _tan1, 变量 x, 和次数 10 下的结果相等
    assert rs_fun(p, _tan1, x, 10) == _tan1(p, x, 10)

# 定义一个测试函数，测试多元多项式的 n 次根操作
def test_nth_root():
    # 创建有理数域 R 和变量 x, y
    R, x, y = ring('x, y', QQ)
    # 断言 rs_nth_root 计算结果与预期值相等，指定参数为 (1 + x^2*y, 4, x, 10)
    assert rs_nth_root(1 + x**2*y, 4, x, 10) == -77*x**8*y**4/2048 + \
        7*x**6*y**3/128 - 3*x**4*y**2/32 + x**2*y/4 + 1
    # 断言 rs_nth_root 计算结果与预期值相等，指定参数为 (1 + x*y + x^2*y^3, 3, x, 5)
    assert rs_nth_root(1 + x*y + x**2*y**3, 3, x, 5) == -x**4*y**6/9 + \
        5*x**4*y**5/27 - 10*x**4*y**4/243 - 2*x**3*y**4/9 + 5*x**3*y**3/81 + \
        x**2*y**3/3 - x**2*y**2/9 + x*y/3 + 1
    # 断言 rs_nth_root 计算结果与预期值相等，指定参数为 (8*x, 3, x, 3)
    assert rs_nth_root(8*x, 3, x, 3) == 2*x**QQ(1, 3)
    # 断言 rs_nth_root 计算结果与预期值相等，指定参数为 (8*x + x^2 + x^3, 3, x, 3)
    assert rs_nth_root(8*x + x**2 + x**3, 3, x, 3) == x**QQ(4,3)/12 + 2*x**QQ(1,3)
    # 计算 rs_nth_root 结果赋值给变量 r，断言 r 与预期值相等，指定参数为 (8*x + x^2*y + x^3, 3, x, 4)
    r = rs_nth_root(8*x + x**2*y + x**3, 3, x, 4)
    assert r == -x**QQ(7,3)*y**2/288 + x**QQ(7,3)/12 + x**QQ(4,3)*y/12 + 2*x**QQ(1,3)

    # 在级数中的常数项
    a = symbols('a')
    # 创建表达式域 R 和变量 x, y
    R, x, y = ring('x, y', EX)
    # 断言 rs_nth_root 计算结果与预期值相等，指定参数为 (x + a, 3, x, 4)
    assert rs_nth_root(x + a, 3, x, 4) == EX(5/(81*a**QQ(8, 3)))*x**3 - \
        EX(1/(9*a**QQ(5, 3)))*x**2 + EX(1/(3*a**QQ(2, 3)))*x + EX(a**QQ(1, 3))
    # 断言 rs_nth_root 计算结果与预期值相等，指定参数为 (x^(2/3) + x^2*y + 5, 2, x, 3)
    assert rs_nth_root(x**QQ(2, 3) + x**2*y + 5, 2, x, 3) == -EX(sqrt(5)/100)*\
        x**QQ(8, 3)*y - EX(sqrt(5)/16000)*x**QQ(8, 3) + EX(sqrt(5)/10)*x**2*y + \
        EX(sqrt(5)/2000)*x**2 - EX(sqrt(5)/200)*x**QQ(4, 3) + \
        EX(sqrt(5)/10)*x**QQ(2, 3) + EX(sqrt(5))

# 定义一个测试函数，测试多元多项式的反正切操作
def test_atan():
    # 创建有理数域 R 和变量 x, y
    R, x, y = ring('x, y', QQ)
    # 断言 rs_atan 计算结果与预期值相等，指定参数为 (x, x, 9)
    assert rs_atan(x, x, 9) == -x**7/7 + x**5/5 - x**3/3 + x
    # 断言 rs_atan 计算结果与预期值相等，指定参数为 (x*y + x^2*y^3, x, 9)
    assert rs_atan(x*y + x**2*y**3, x, 9) == 2*x**8*y**11 - x**8*y**9 + \
        2*x**7*y**9 - x**7*y**7/7 - x**6*y**9/3 + x**6*y**7 - x**5*y**7 + \
        x**5*y**5/5 - x**4*y**5 - x**3*y**3/3 + x**2*y**3 + x*y

    # 在级数中的常数项
    a = symbols('a')
    # 创建表达式域 R 和变量 x, y
    R, x, y = ring('x, y', EX)
    # 断言 rs_atan 计算结果与预期值相等，指定参数为 (x + a, x, 5)
    assert rs_atan(x + a, x, 5) == -EX((a**3 - a)/(a**8 + 4*a**6 + 6*a**4 + \
        4*a**2 + 1))*x**4 + EX((3*a**2 - 1)/(3*a**6 + 9*a**4 + \
        9*a**2 + 3))*x**3 - EX(a/(a**4 + 2*a**2 + 1))*x**2 + \
        EX(1/(a**2 + 1))*x + EX(atan(a))
    # 断言 rs_atan 计算结果与预期值相等，指定参数为 (x + x^2*y + a, x,
    # 断言：计算 rs_tan(x, x, 9) / x**5 的值，并与给定的多项式表达式进行比较
    assert rs_tan(x, x, 9)/x**5 == \
        Rational(17, 315)*x**2 + Rational(2, 15) + Rational(1, 3)*x**(-2) + x**(-4)

    # 断言：计算 rs_tan(x*y + x**2*y**3, x, 9) 的值，并与给定的多项式表达式进行比较
    assert rs_tan(x*y + x**2*y**3, x, 9) == 4*x**8*y**11/3 + 17*x**8*y**9/45 + \
        4*x**7*y**9/3 + 17*x**7*y**7/315 + x**6*y**9/3 + 2*x**6*y**7/3 + \
        x**5*y**7 + 2*x**5*y**5/15 + x**4*y**5 + x**3*y**3/3 + x**2*y**3 + x*y

    # 定义符号 'a'
    a = symbols('a')
    # 创建多项式环 R，并定义变量 x, y，使用有理数域 QQ 中的 tan(a) 和 'a'
    R, x, y = ring('x, y', QQ[tan(a), a])
    # 断言：计算 rs_tan(x + a, x, 5) 的值，并与给定的多项式表达式进行比较
    assert rs_tan(x + a, x, 5) == (tan(a)**5 + 5*tan(a)**3/3 +
        2*tan(a)/3)*x**4 + (tan(a)**4 + 4*tan(a)**2/3 + Rational(1, 3))*x**3 + \
        (tan(a)**3 + tan(a))*x**2 + (tan(a)**2 + 1)*x + tan(a)

    # 断言：计算 rs_tan(x + x**2*y + a, x, 4) 的值，并与给定的多项式表达式进行比较
    assert rs_tan(x + x**2*y + a, x, 4) == (2*tan(a)**3 + 2*tan(a))*x**3*y + \
        (tan(a)**4 + Rational(4, 3)*tan(a)**2 + Rational(1, 3))*x**3 + (tan(a)**2 + 1)*x**2*y + \
        (tan(a)**3 + tan(a))*x**2 + (tan(a)**2 + 1)*x + tan(a)

    # 创建多项式环 R，并定义变量 x, y，使用扩展域 EX
    R, x, y = ring('x, y', EX)
    # 断言：计算 rs_tan(x + a, x, 5) 的值，并与给定的多项式表达式进行比较
    assert rs_tan(x + a, x, 5) == EX(tan(a)**5 + 5*tan(a)**3/3 +
        2*tan(a)/3)*x**4 + EX(tan(a)**4 + 4*tan(a)**2/3 + EX(1)/3)*x**3 + \
        EX(tan(a)**3 + tan(a))*x**2 + EX(tan(a)**2 + 1)*x + EX(tan(a))

    # 断言：计算 rs_tan(x + x**2*y + a, x, 4) 的值，并与给定的多项式表达式进行比较
    assert rs_tan(x + x**2*y + a, x, 4) == EX(2*tan(a)**3 +
        2*tan(a))*x**3*y + EX(tan(a)**4 + 4*tan(a)**2/3 + EX(1)/3)*x**3 + \
        EX(tan(a)**2 + 1)*x**2*y + EX(tan(a)**3 + tan(a))*x**2 + \
        EX(tan(a)**2 + 1)*x + EX(tan(a))

    # 定义多项式 p
    p = x + x**2 + 5
    # 断言：计算 rs_atan(p, x, 10).compose(x, 10) 的值，并与给定的表达式进行比较
    assert rs_atan(p, x, 10).compose(x, 10) == EX(atan(5) + S(67701870330562640) / \
        668083460499)
# 定义一个测试函数 test_cot，用于测试 rs_cot 函数的输出是否正确
def test_cot():
    # 在有理数环 QQ 中定义环 R，变量 x 和 y
    R, x, y = ring('x, y', QQ)
    # 断言：测试 rs_cot 函数对 x**6 + x**7 的输出是否等于预期结果
    assert rs_cot(x**6 + x**7, x, 8) == x**(-6) - x**(-5) + x**(-4) - \
        x**(-3) + x**(-2) - x**(-1) + 1 - x + x**2 - x**3 + x**4 - x**5 + \
        2*x**6/3 - 4*x**7/3
    # 断言：测试 rs_cot 函数对 x + x**2*y 的输出是否等于预期结果
    assert rs_cot(x + x**2*y, x, 5) == -x**4*y**5 - x**4*y/15 + x**3*y**4 - \
        x**3/45 - x**2*y**3 - x**2*y/3 + x*y**2 - x/3 - y + x**(-1)

# 定义一个测试函数 test_sin，用于测试 rs_sin 函数的输出是否正确
def test_sin():
    # 在有理数环 QQ 中定义环 R，变量 x 和 y
    R, x, y = ring('x, y', QQ)
    # 断言：测试 rs_sin 函数对 x 的输出是否等于预期结果
    assert rs_sin(x, x, 9)/x**5 == \
        Rational(-1, 5040)*x**2 + Rational(1, 120) - Rational(1, 6)*x**(-2) + x**(-4)
    # 断言：测试 rs_sin 函数对 x*y + x**2*y**3 的输出是否等于预期结果
    assert rs_sin(x*y + x**2*y**3, x, 9) == x**8*y**11/12 - \
        x**8*y**9/720 + x**7*y**9/12 - x**7*y**7/5040 - x**6*y**9/6 + \
        x**6*y**7/24 - x**5*y**7/2 + x**5*y**5/120 - x**4*y**5/2 - \
        x**3*y**3/6 + x**2*y**3 + x*y

    # 常数级数中的项
    a = symbols('a')
    # 在含有 sin(a), cos(a), a 的 QQ 环中定义环 R，变量 x 和 y
    R, x, y = ring('x, y', QQ[sin(a), cos(a), a])
    # 断言：测试 rs_sin 函数对 x + a 的输出是否等于预期结果
    assert rs_sin(x + a, x, 5) == sin(a)*x**4/24 - cos(a)*x**3/6 - \
        sin(a)*x**2/2 + cos(a)*x + sin(a)
    # 断言：测试 rs_sin 函数对 x + x**2*y + a 的输出是否等于预期结果
    assert rs_sin(x + x**2*y + a, x, 5) == -sin(a)*x**4*y**2/2 - \
        cos(a)*x**4*y/2 + sin(a)*x**4/24 - sin(a)*x**3*y - cos(a)*x**3/6 + \
        cos(a)*x**2*y - sin(a)*x**2/2 + cos(a)*x + sin(a)

    # 在 EX 环中定义环 R，变量 x 和 y
    R, x, y = ring('x, y', EX)
    # 断言：测试 rs_sin 函数对 x + a 的输出是否等于预期结果
    assert rs_sin(x + a, x, 5) == EX(sin(a)/24)*x**4 - EX(cos(a)/6)*x**3 - \
        EX(sin(a)/2)*x**2 + EX(cos(a))*x + EX(sin(a))
    # 断言：测试 rs_sin 函数对 x + x**2*y + a 的输出是否等于预期结果
    assert rs_sin(x + x**2*y + a, x, 5) == -EX(sin(a)/2)*x**4*y**2 - \
        EX(cos(a)/2)*x**4*y + EX(sin(a)/24)*x**4 - EX(sin(a))*x**3*y - \
        EX(cos(a)/6)*x**3 + EX(cos(a))*x**2*y - EX(sin(a)/2)*x**2 + \
        EX(cos(a))*x + EX(sin(a))

# 定义一个测试函数 test_cos，用于测试 rs_cos 函数的输出是否正确
def test_cos():
    # 在有理数环 QQ 中定义环 R，变量 x 和 y
    R, x, y = ring('x, y', QQ)
    # 断言：测试 rs_cos 函数对 x 的输出是否等于预期结果
    assert rs_cos(x, x, 9)/x**5 == \
        Rational(1, 40320)*x**3 - Rational(1, 720)*x + Rational(1, 24)*x**(-1) - S.Half*x**(-3) + x**(-5)
    # 断言：测试 rs_cos 函数对 x*y + x**2*y**3 的输出是否等于预期结果
    assert rs_cos(x*y + x**2*y**3, x, 9) == x**8*y**12/24 - \
        x**8*y**10/48 + x**8*y**8/40320 + x**7*y**10/6 - \
        x**7*y**8/120 + x**6*y**8/4 - x**6*y**6/720 + x**5*y**6/6 - \
        x**4*y**6/2 + x**4*y**4/24 - x**3*y**4 - x**2*y**2/2 + 1

    # 常数级数中的项
    a = symbols('a')
    # 在含有 sin(a), cos(a), a 的 QQ 环中定义环 R，变量 x 和 y
    R, x, y = ring('x, y', QQ[sin(a), cos(a), a])
    # 断言：测试 rs_cos 函数对 x + a 的输出是否等于预期结果
    assert rs_cos(x + a, x, 5) == cos(a)*x**4/24 + sin(a)*x**3/6 - \
        cos(a)*x**2/2 - sin(a)*x + cos(a)
    # 断言：测试 rs_cos 函数对 x + x**2*y + a 的输出是否等于预期结果
    assert rs_cos(x + x**2*y + a, x, 5) == -cos(a)*x**4*y**2/2 + \
        sin(a)*x**4*y/2 + cos(a)*x**4/24 - cos(a)*x**3*y + sin(a)*x**3/6 - \
        sin(a)*x**2*y - cos(a)*x**2/2 - sin(a)*x + cos(a)

    # 在 EX 环中定义环 R，变量 x 和 y
    R, x, y = ring('x, y', EX)
    # 断言：测试 rs_cos 函数对 x + a 的输出是否等于预期结果
    assert rs_cos(x + a, x, 5) == EX(cos(a)/24)*x**4 + EX(sin(a)/6)*x**3 - \
        EX(cos(a)/2)*x**2 - EX(sin(a))*x + EX(cos(a))
    # 断言：测试 rs_cos 函数对 x + x**2*y + a 的输出是否等于预期结果
    assert rs_cos(x + x**2*y + a, x, 5) == -EX(cos(a)/2)*x**4*y**2 + \
        EX(sin(a)/2)*x**4*y + EX(cos(a)/24)*x**4 - EX(cos(a))*x**3*y + \
        EX(sin(a)/6)*x**3 - EX(sin(a))*x**2*y - EX(cos(a)/2)*x**2 - \
        EX(sin(a))*x + EX(cos(a))

# 定义一个测试函数 test_cos_sin，用于测试 rs_cos_sin 函数的输出是否正确
def test_cos_sin():
    # 在有理数环 QQ 中定义环 R，变量 x 和 y
    R, x, y = ring('x, y', QQ)
    # 调用 rs_cos_sin 函数，获取返回的 cos 和 sin 的级数
    cos, sin = rs_cos_sin(x, x, 9)
    # 断言：验证 cos 的值是否等于调用 rs_cos 函数返回的结果
    assert cos == rs_cos(x, x, 9)
    # 断言：验证 sin 的值是否等于调用 rs_sin 函数返回的结果
    assert sin == rs_sin(x, x, 9)
    # 调用 rs_cos_sin 函数，计算 x + x*y 的余弦和正弦值，返回结果分别赋给 cos 和 sin
    cos, sin = rs_cos_sin(x + x*y, x, 5)
    # 断言：验证 cos 的值是否等于调用 rs_cos 函数返回的结果，参数为 x + x*y, x, 5
    assert cos == rs_cos(x + x*y, x, 5)
    # 断言：验证 sin 的值是否等于调用 rs_sin 函数返回的结果，参数为 x + x*y, x, 5
    assert sin == rs_sin(x + x*y, x, 5)
# 定义测试函数 test_atanh，用于测试 rs_atanh 函数的各种情况
def test_atanh():
    # 在有理数域 QQ 中创建环 R，并定义变量 x, y
    R, x, y = ring('x, y', QQ)
    # 断言：rs_atanh 函数的结果与给定的多项式表达式相等，验证高阶项和常数项的匹配性
    assert rs_atanh(x, x, 9)/x**5 == Rational(1, 7)*x**2 + Rational(1, 5) + Rational(1, 3)*x**(-2) + x**(-4)
    # 断言：rs_atanh 函数对复杂表达式的计算结果正确性验证，包含多项式和分数
    assert rs_atanh(x*y + x**2*y**3, x, 9) == 2*x**8*y**11 + x**8*y**9 + \
        2*x**7*y**9 + x**7*y**7/7 + x**6*y**9/3 + x**6*y**7 + x**5*y**7 + \
        x**5*y**5/5 + x**4*y**5 + x**3*y**3/3 + x**2*y**3 + x*y

    # 创建符号变量 a
    a = symbols('a')
    # 在表达式环 EX 中重新定义环 R，并重新定义变量 x, y
    R, x, y = ring('x, y', EX)
    # 断言：rs_atanh 函数在符号 a 加上 x 后的结果符合复杂表达式的展开形式
    assert rs_atanh(x + a, x, 5) == EX((a**3 + a)/(a**8 - 4*a**6 + 6*a**4 - \
        4*a**2 + 1))*x**4 - EX((3*a**2 + 1)/(3*a**6 - 9*a**4 + \
        9*a**2 - 3))*x**3 + EX(a/(a**4 - 2*a**2 + 1))*x**2 - EX(1/(a**2 - \
        1))*x + EX(atanh(a))
    # 断言：rs_atanh 函数在符号 a 加上 x**2*y 后的结果符合复杂表达式的展开形式
    assert rs_atanh(x + x**2*y + a, x, 4) == EX(2*a/(a**4 - 2*a**2 + \
        1))*x**3*y - EX((3*a**2 + 1)/(3*a**6 - 9*a**4 + 9*a**2 - 3))*x**3 - \
        EX(1/(a**2 - 1))*x**2*y + EX(a/(a**4 - 2*a**2 + 1))*x**2 - \
        EX(1/(a**2 - 1))*x + EX(atanh(a))

    # 定义多项式 p
    p = x + x**2 + 5
    # 断言：rs_atanh 函数对 p 的计算结果的复合函数形式正确性验证
    assert rs_atanh(p, x, 10).compose(x, 10) == EX(Rational(-733442653682135, 5079158784) \
        + atanh(5))

# 定义测试函数 test_sinh，用于测试 rs_sinh 函数的各种情况
def test_sinh():
    # 在有理数域 QQ 中创建环 R，并定义变量 x, y
    R, x, y = ring('x, y', QQ)
    # 断言：rs_sinh 函数的结果与给定的多项式表达式相等，验证高阶项和常数项的匹配性
    assert rs_sinh(x, x, 9)/x**5 == Rational(1, 5040)*x**2 + Rational(1, 120) + Rational(1, 6)*x**(-2) + x**(-4)
    # 断言：rs_sinh 函数对复杂表达式的计算结果正确性验证，包含多项式和分数
    assert rs_sinh(x*y + x**2*y**3, x, 9) == x**8*y**11/12 + \
        x**8*y**9/720 + x**7*y**9/12 + x**7*y**7/5040 + x**6*y**9/6 + \
        x**6*y**7/24 + x**5*y**7/2 + x**5*y**5/120 + x**4*y**5/2 + \
        x**3*y**3/6 + x**2*y**3 + x*y

# 定义测试函数 test_cosh，用于测试 rs_cosh 函数的各种情况
def test_cosh():
    # 在有理数域 QQ 中创建环 R，并定义变量 x, y
    R, x, y = ring('x, y', QQ)
    # 断言：rs_cosh 函数的结果与给定的多项式表达式相等，验证高阶项和常数项的匹配性
    assert rs_cosh(x, x, 9)/x**5 == Rational(1, 40320)*x**3 + Rational(1, 720)*x + Rational(1, 24)*x**(-1) + \
        S.Half*x**(-3) + x**(-5)
    # 断言：rs_cosh 函数对复杂表达式的计算结果正确性验证，包含多项式和分数
    assert rs_cosh(x*y + x**2*y**3, x, 9) == x**8*y**12/24 + \
        x**8*y**10/48 + x**8*y**8/40320 + x**7*y**10/6 + \
        x**7*y**8/120 + x**6*y**8/4 + x**6*y**6/720 + x**5*y**6/6 + \
        x**4*y**6/2 + x**4*y**4/24 + x**3*y**4 + x**2*y**2/2 + 1

# 定义测试函数 test_tanh，用于测试 rs_tanh 函数的各种情况
def test_tanh():
    # 在有理数域 QQ 中创建环 R，并定义变量 x, y
    R, x, y = ring('x, y', QQ)
    # 断言：rs_tanh 函数的结果与给定的多项式表达式相等，验证高阶项和常数项的匹配性
    assert rs_tanh(x, x, 9)/x**5 == Rational(-17, 315)*x**2 + Rational(2, 15) - Rational(1, 3)*x**(-2) + x**(-4)
    # 断言：rs_tanh 函数对复杂表达式的计算结果正确性验证，包含多项式和分数
    assert rs_tanh(x*y + x**2*y**3, x, 9) == 4*x**8*y**11/3 - \
        17*x**8*y**9/45 + 4*x**7*y**9/3 - 17*x**7*y**7/315 - x**6*y**9/3 + \
        2*x**6*y**7/3 - x**5*y**7 + 2*x**5*y**5/15 - x**4*y**5 - \
        x**3*y**3/3 + x**2*y**3 + x*y

    # 创建符号变量 a
    a = symbols('a')
    # 在表达式环 EX 中重新定义环 R，并重新定义变量 x, y
    R, x, y = ring('x, y', EX)
    # 断言：rs_tanh 函数在符号 a 加上 x 后的结果符合复杂表达式的展开形式
    assert rs_tanh(x + a, x, 5) == EX(tanh(a)**5 - 5*tanh(a)**3/3 +
        2*tanh(a)/3)*x**4 + EX(-tanh(a)**4 + 4*tanh(a)**2/3 - QQ(1, 3))*x**3 + \
        EX(tanh(a)**3 - tanh(a))*x**2 + EX(-tanh(a)**2 + 1)*x + EX(tanh(a))

    # 对 rs_tanh 函数的复合结果进行验证
    p = rs_tanh(x + x**2*y + a, x, 4)
    assert (p.compose(x,
    # 使用函数 `ring` 创建一个多项式环 R，其中包含变量 x 和 y，使用实数域 RR
    R, x, y = ring('x, y', RR)
    # 创建一个符号变量 a
    a = symbols('a')
    # 对于每一对 rs_func 和 sympy_func，分别执行以下操作：
    for rs_func, sympy_func in zip(rs_funcs, sympy_funcs):
        # 计算 rs_func 的结果，对 x 进行组合和代换
        p = rs_func(2 + x, x, 5).compose(x, 5)
        # 计算 sympy_func 的级数展开并移除高阶无穷小项
        q = sympy_func(2 + a).series(a, 0, 5).removeO()
        # 检查 p 和 q 是否在数值上相近
        is_close(p.as_expr(), q.subs(a, 5).n())

    # 计算 rs_nth_root 的结果，对 x 进行组合和代换
    p = rs_nth_root(2 + x, 5, x, 5).compose(x, 5)
    # 计算 (2 + a) 的 1/5 次幂的级数展开并移除高阶无穷小项
    q = ((2 + a)**QQ(1, 5)).series(a, 0, 5).removeO()
    # 检查 p 和 q 是否在数值上相近
    is_close(p.as_expr(), q.subs(a, 5).n())
# 定义测试函数 test_is_regular，用于测试 rs_is_puiseux 函数的正常情况
def test_is_regular():
    # 创建有理函数环 R，并定义变量 x, y
    R, x, y = ring('x, y', QQ)
    # 定义多项式 p = 1 + 2*x + x**2 + 3*x**3
    p = 1 + 2*x + x**2 + 3*x**3
    # 断言 rs_is_puiseux(p, x) 返回 False
    assert not rs_is_puiseux(p, x)

    # 重新赋值 p = x + x**(1/5)*y
    p = x + x**QQ(1,5)*y
    # 断言 rs_is_puiseux(p, x) 返回 True
    assert rs_is_puiseux(p, x)
    # 断言 rs_is_puiseux(p, y) 返回 False
    assert not rs_is_puiseux(p, y)

    # 重新赋值 p = x + x**2*y**(1/5)*y
    p = x + x**2*y**QQ(1,5)*y
    # 断言 rs_is_puiseux(p, x) 返回 False
    assert not rs_is_puiseux(p, x)

# 定义测试函数 test_puiseux，用于测试不同的 puiseux 函数
def test_puiseux():
    # 创建有理函数环 R，并定义变量 x, y
    R, x, y = ring('x, y', QQ)
    # 定义多项式 p = x**(2/5) + x**(2/3) + x
    p = x**QQ(2,5) + x**QQ(2,3) + x

    # 测试 rs_series_inversion 函数
    r = rs_series_inversion(p, x, 1)
    r1 = -x**QQ(14,15) + x**QQ(4,5) - 3*x**QQ(11,15) + x**QQ(2,3) + \
        2*x**QQ(7,15) - x**QQ(2,5) - x**QQ(1,5) + x**QQ(2,15) - x**QQ(-2,15) \
        + x**QQ(-2,5)
    # 断言 r 等于 r1
    assert r == r1

    # 测试 rs_nth_root 函数
    r = rs_nth_root(1 + p, 3, x, 1)
    assert r == -x**QQ(4,5)/9 + x**QQ(2,3)/3 + x**QQ(2,5)/3 + 1

    # 测试 rs_log 函数
    r = rs_log(1 + p, x, 1)
    assert r == -x**QQ(4,5)/2 + x**QQ(2,3) + x**QQ(2,5)

    # 测试 rs_LambertW 函数
    r = rs_LambertW(p, x, 1)
    assert r == -x**QQ(4,5) + x**QQ(2,3) + x**QQ(2,5)

    # 重新定义 p1 = x + x**(1/5)*y
    p1 = x + x**QQ(1,5)*y
    # 测试 rs_exp 函数
    r = rs_exp(p1, x, 1)
    assert r == x**QQ(4,5)*y**4/24 + x**QQ(3,5)*y**3/6 + x**QQ(2,5)*y**2/2 + \
        x**QQ(1,5)*y + 1

    # 测试 rs_atan 函数
    r = rs_atan(p, x, 2)
    assert r ==  -x**QQ(9,5) - x**QQ(26,15) - x**QQ(22,15) - x**QQ(6,5)/3 + \
        x + x**QQ(2,3) + x**QQ(2,5)

    # 测试 rs_atan 函数，传入 p1 = x + x**(1/5)*y
    r = rs_atan(p1, x, 2)
    assert r ==  x**QQ(9,5)*y**9/9 + x**QQ(9,5)*y**4 - x**QQ(7,5)*y**7/7 - \
        x**QQ(7,5)*y**2 + x*y**5/5 + x - x**QQ(3,5)*y**3/3 + x**QQ(1,5)*y

    # 测试 rs_asin 函数
    r = rs_asin(p, x, 2)
    assert r == x**QQ(9,5)/2 + x**QQ(26,15)/2 + x**QQ(22,15)/2 + \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    # 测试 rs_cot 函数
    r = rs_cot(p, x, 1)
    assert r == -x**QQ(14,15) + x**QQ(4,5) - 3*x**QQ(11,15) + \
        2*x**QQ(2,3)/3 + 2*x**QQ(7,15) - 4*x**QQ(2,5)/3 - x**QQ(1,5) + \
        x**QQ(2,15) - x**QQ(-2,15) + x**QQ(-2,5)

    # 测试 rs_cos_sin 函数
    r = rs_cos_sin(p, x, 2)
    assert r[0] == x**QQ(28,15)/6 - x**QQ(5,3) + x**QQ(8,5)/24 - x**QQ(7,5) - \
        x**QQ(4,3)/2 - x**QQ(16,15) - x**QQ(4,5)/2 + 1
    assert r[1] == -x**QQ(9,5)/2 - x**QQ(26,15)/2 - x**QQ(22,15)/2 - \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    # 测试 rs_atanh 函数
    r = rs_atanh(p, x, 2)
    assert r == x**QQ(9,5) + x**QQ(26,15) + x**QQ(22,15) + x**QQ(6,5)/3 + x + \
        x**QQ(2,3) + x**QQ(2,5)

    # 测试 rs_sinh 函数
    r = rs_sinh(p, x, 2)
    assert r == x**QQ(9,5)/2 + x**QQ(26,15)/2 + x**QQ(22,15)/2 + \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    # 测试 rs_cosh 函数
    r = rs_cosh(p, x, 2)
    assert r == x**QQ(28,15)/6 + x**QQ(5,3) + x**QQ(8,5)/24 + x**QQ(7,5) + \
        x**QQ(4,3)/2 + x**QQ(16,15) + x**QQ(4,5)/2 + 1

    # 测试 rs_tanh 函数
    r = rs_tanh(p, x, 2)
    assert r == -x**QQ(9,5) - x**QQ(26,15) - x**QQ(22,15) - x**QQ(6,5)/3 + \
        x + x**QQ(2,3) + x**QQ(2,5)

# 定义测试函数 test_puiseux_algebraic，测试包含代数元素的 puiseux 函数
def test_puiseux_algebraic(): # https://github.com/sympy/sympy/issues/24395
    # 创建有理函数环 K，包含 sqrt(2) 的代数扩域
    K = QQ.algebraic_field(sqrt(2))
    sqrt2 = K.from_sympy(sqrt(2))
    # 定义符号变量 x, y
    x, y = symbols('x, y')
    # 创建有理函数环 R，包含变量 x, y，并使用 K 作为系数环
    R, xr, yr = ring([x, y], K)
    # 定义多项式 p = (1+sqrt(2))*xr**(1/2) + (1-sqrt(2))*yr**(2/3)
    p = (1+sqrt2
    # 定义一个有理函数环 R，变量为 x
    R, x = ring('x', QQ)
    # 调用 rs_sin 函数计算 sin(x) 的近似级数，乘以 x**(-5)
    r = rs_sin(x, x, 15)*x**(-5)
    # 断言 r 的值，确保近似级数的准确性
    assert r == x**8/6227020800 - x**6/39916800 + x**4/362880 - x**2/5040 + \
        QQ(1,120) - x**-2/6 + x**-4

    # 调用 rs_sin 函数计算 sin(x) 的近似级数，使用 10 阶的多项式
    p = rs_sin(x, x, 10)
    # 调用 rs_nth_root 函数计算 p 的平方根，使用 2 阶的多项式
    r = rs_nth_root(p, 2, x, 10)
    # 断言 r 的值，确保平方根的准确性
    assert r == -67*x**QQ(17,2)/29030400 - x**QQ(13,2)/24192 + \
        x**QQ(9,2)/1440 - x**QQ(5,2)/12 + x**QQ(1,2)

    # 再次调用 rs_sin 函数计算 sin(x) 的近似级数，使用 10 阶的多项式
    p = rs_sin(x, x, 10)
    # 调用 rs_nth_root 函数计算 p 的7次根，使用 10 阶的多项式
    r = rs_nth_root(p, 7, x, 10)
    # 调用 rs_pow 函数计算 r 的5次方，使用 10 阶的多项式
    r = rs_pow(r, 5, x, 10)
    # 断言 r 的值，确保5次方的准确性
    assert r == -97*x**QQ(61,7)/124467840 - x**QQ(47,7)/16464 + \
        11*x**QQ(33,7)/3528 - 5*x**QQ(19,7)/42 + x**QQ(5,7)

    # 调用 rs_exp 函数计算 exp(x**(1/2)) 的近似级数，使用 10 阶的多项式
    r = rs_exp(x**QQ(1,2), x, 10)
    # 断言 r 的值，确保 exp(x**(1/2)) 的近似级数的准确性
    assert r == x**QQ(19,2)/121645100408832000 + x**9/6402373705728000 + \
        x**QQ(17,2)/355687428096000 + x**8/20922789888000 + \
        x**QQ(15,2)/1307674368000 + x**7/87178291200 + \
        x**QQ(13,2)/6227020800 + x**6/479001600 + x**QQ(11,2)/39916800 + \
        x**5/3628800 + x**QQ(9,2)/362880 + x**4/40320 + x**QQ(7,2)/5040 + \
        x**3/720 + x**QQ(5,2)/120 + x**2/24 + x**QQ(3,2)/6 + x/2 + \
        x**QQ(1,2) + 1
# 定义测试函数 test_puiseux2，该函数用于测试 Puiseux 级数相关的计算
def test_puiseux2():
    # 创建有理数域 QQ 上的带变量 'y' 的环 R 和变量 'y'
    R, y = ring('y', QQ)
    # 在环 R 上创建带变量 'x' 的环 S 和变量 'x'
    S, x = ring('x', R)

    # 定义多项式 p，包括 x 和 x 的五分之一次方乘以 y
    p = x + x**QQ(1,5)*y
    # 使用 rs_atan 函数计算 p 的 Puiseux 级数，取前三项
    r = rs_atan(p, x, 3)
    # 断言计算结果 r 符合预期值
    assert r == (y**13/13 + y**8 + 2*y**3)*x**QQ(13,5) - (y**11/11 + y**6 +
        y)*x**QQ(11,5) + (y**9/9 + y**4)*x**QQ(9,5) - (y**7/7 +
        y**2)*x**QQ(7,5) + (y**5/5 + 1)*x - y**3*x**QQ(3,5)/3 + y*x**QQ(1,5)


# 修饰符 @slow 表示以下测试函数较为耗时
@slow
# 定义测试函数 test_rs_series，用于测试 rs_series 函数的多项式级数展开功能
def test_rs_series():
    # 创建符号变量 x, a, b, c
    x, a, b, c = symbols('x, a, b, c')

    # 断言 rs_series(a, a, 5) 的表达式等于 a
    assert rs_series(a, a, 5).as_expr() == a
    # 断言 rs_series(sin(a), a, 5) 的表达式等于 sin(a) 在 a=0 处展开到五阶后去除高阶项
    assert rs_series(sin(a), a, 5).as_expr() == (sin(a).series(a, 0,
        5)).removeO()
    # 断言 rs_series(sin(a) + cos(a), a, 5) 的表达式等于 (sin(a) + cos(a)) 在 a=0 处展开到五阶后去除高阶项
    assert rs_series(sin(a) + cos(a), a, 5).as_expr() == ((sin(a) +
        cos(a)).series(a, 0, 5)).removeO()
    # 断言 rs_series(sin(a)*cos(a), a, 5) 的表达式等于 (sin(a)*cos(a)) 在 a=0 处展开到五阶后去除高阶项
    assert rs_series(sin(a)*cos(a), a, 5).as_expr() == ((sin(a)*
        cos(a)).series(a, 0, 5)).removeO()

    # 定义多项式 p1
    p = (sin(a) - a)*(cos(a**2) + a**4/2)
    # 断言 rs_series(p, a, 10) 的表达式等于 p 在 a=0 处展开到十阶后去除高阶项
    assert expand(rs_series(p, a, 10).as_expr()) == expand(p.series(a, 0,
        10).removeO())

    # 定义多项式 p2
    p = sin(a**2/2 + a/3) + cos(a/5)*sin(a/2)**3
    # 断言 rs_series(p, a, 5) 的表达式等于 p 在 a=0 处展开到五阶后去除高阶项
    assert expand(rs_series(p, a, 5).as_expr()) == expand(p.series(a, 0,
        5).removeO())

    # 定义多项式 p3
    p = sin(x**2 + a)*(cos(x**3 - 1) - a - a**2)
    # 断言 rs_series(p, a, 5) 的表达式等于 p 在 a=0 处展开到五阶后去除高阶项
    assert expand(rs_series(p, a, 5).as_expr()) == expand(p.series(a, 0,
        5).removeO())

    # 定义多项式 p4
    p = sin(a**2 - a/3 + 2)**5*exp(a**3 - a/2)
    # 断言 rs_series(p, a, 10) 的表达式等于 p 在 a=0 处展开到十阶后去除高阶项
    assert expand(rs_series(p, a, 10).as_expr()) == expand(p.series(a, 0,
        10).removeO())

    # 定义多项式 p5
    p = sin(a + b + c)
    # 断言 rs_series(p, a, 5) 的表达式等于 p 在 a=0 处展开到五阶后去除高阶项
    assert expand(rs_series(p, a, 5).as_expr()) == expand(p.series(a, 0,
        5).removeO())

    # 定义多项式 p6
    p = tan(sin(a**2 + 4) + b + c)
    # 断言 rs_series(p, a, 6) 的表达式等于 p 在 a=0 处展开到六阶后去除高阶项
    assert expand(rs_series(p, a, 6).as_expr()) == expand(p.series(a, 0,
        6).removeO())

    # 定义多项式 p7
    p = a**QQ(2,5) + a**QQ(2,3) + a
    # 使用 rs_series 函数计算 p 的级数展开，取前两项
    r = rs_series(tan(p), a, 2)
    # 断言计算结果 r 的表达式符合预期值
    assert r.as_expr() == a**QQ(9,5) + a**QQ(26,15) + a**QQ(22,15) + a**QQ(6,5)/3 + \
        a + a**QQ(2,3) + a**QQ(2,5)

    # 定义多项式 p8
    p = a**QQ(2,5) + a**QQ(2,3) + a
    # 使用 rs_series 函数计算 p 的指数函数展开，取前一项
    r = rs_series(exp(p), a, 1)
    # 断言计算结果 r 的表达式符合预期值
    assert r.as_expr() == a**QQ(4,5)/2 + a**QQ(2,3) + a**QQ(2,5) + 1

    # 定义多项式 p9
    p = a**QQ(2,5) + a**QQ(2,3) + a
    # 使用 rs_series 函数计算 p 的正弦函数展开，取前两项
    r = rs_series(sin(p), a, 2)
    # 断言计算结果 r 的表达式符合预期值
    assert r.as_expr() == -a**QQ(9,5)/2 - a**QQ(26,15)/2 - a**QQ(22,15)/2 - \
        a**QQ(6,5)/6 + a + a**QQ(2,3) + a**QQ(2,5)

    # 定义多项式 p10
    p = a**QQ(2,5) + a**QQ(2,3) + a
    # 使用 rs_series 函数计算 p 的余弦函数展开，取前两项
    r = rs_series(cos(p), a, 2)
    # 断言计算结果 r 的表达式符合预期值
    assert r.as_expr() == a**QQ(28,15)/6 - a**QQ(5,3) + a**QQ(8,5)/24 - a**QQ(7,5) - \
        a**QQ(4,3)/2 - a**QQ(16,15) - a**QQ(4,5)/2 + 1

    # 断言 rs_series(sin(a)/7, a, 5) 的表达式等于 sin(a)/7 在 a=0 处展开到五阶后去除高阶项
    assert rs_series(sin(a)/7, a, 5).as_expr() == (sin(a)/7).series(a, 0,
            5).removeO()

    #
```