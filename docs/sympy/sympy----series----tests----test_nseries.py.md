# `D:\src\scipysrc\sympy\sympy\series\tests\test_nseries.py`

```
# 导入累积边界的符号、导数计算和极点错误处理等工具函数
from sympy.calculus.util import AccumBounds
from sympy.core.function import (Derivative, PoleError)
# 导入数学常数 E、虚数单位 I、整数、有理数、圆周率等
from sympy.core.numbers import (E, I, Integer, Rational, pi)
# 导入单例 S，表示单个值
from sympy.core.singleton import S
# 导入符号变量及符号变量的集合
from sympy.core.symbol import (Symbol, symbols)
# 导入符号函数库中的符号函数，如符号函数、指数函数、双曲函数等
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, cot, sin, tan)
# 导入极限相关的函数和对象
from sympy.series.limits import limit
# 导入 O 表示的渐近展开对象
from sympy.series.order import O
# 导入预定义的符号变量 x, y, z
from sympy.abc import x, y, z

# 导入测试用的异常处理和测试标记
from sympy.testing.pytest import raises, XFAIL


# 定义一个名为 test_simple_1 的测试函数
def test_simple_1():
    assert x.nseries(x, n=5) == x
    assert y.nseries(x, n=5) == y
    assert (1/(x*y)).nseries(y, n=5) == 1/(x*y)
    assert Rational(3, 4).nseries(x, n=5) == Rational(3, 4)
    assert x.nseries() == x

# 定义一个名为 test_mul_0 的测试函数
def test_mul_0():
    assert (x*log(x)).nseries(x, n=5) == x*log(x)

# 定义一个名为 test_mul_1 的测试函数
def test_mul_1():
    assert (x*log(2 + x)).nseries(x, n=5) == x*log(2) + x**2/2 - x**3/8 + \
        x**4/24 + O(x**5)
    assert (x*log(1 + x)).nseries(
        x, n=5) == x**2 - x**3/2 + x**4/3 + O(x**5)

# 定义一个名为 test_pow_0 的测试函数
def test_pow_0():
    assert (x**2).nseries(x, n=5) == x**2
    assert (1/x).nseries(x, n=5) == 1/x
    assert (1/x**2).nseries(x, n=5) == 1/x**2
    assert (x**Rational(2, 3)).nseries(x, n=5) == (x**Rational(2, 3))
    assert (sqrt(x)**3).nseries(x, n=5) == (sqrt(x)**3)

# 定义一个名为 test_pow_1 的测试函数
def test_pow_1():
    assert ((1 + x)**2).nseries(x, n=5) == x**2 + 2*x + 1

    # https://github.com/sympy/sympy/issues/21075
    assert ((sqrt(x) + 1)**2).nseries(x) == 2*sqrt(x) + x + 1
    assert ((sqrt(x) + cbrt(x))**2).nseries(x) == 2*x**Rational(5, 6)\
        + x**Rational(2, 3) + x

# 定义一个名为 test_geometric_1 的测试函数
def test_geometric_1():
    assert (1/(1 - x)).nseries(x, n=5) == 1 + x + x**2 + x**3 + x**4 + O(x**5)
    assert (x/(1 - x)).nseries(x, n=6) == x + x**2 + x**3 + x**4 + x**5 + O(x**6)
    assert (x**3/(1 - x)).nseries(x, n=8) == x**3 + x**4 + x**5 + x**6 + \
        x**7 + O(x**8)

# 定义一个名为 test_sqrt_1 的测试函数
def test_sqrt_1():
    assert sqrt(1 + x).nseries(x, n=5) == 1 + x/2 - x**2/8 + x**3/16 - 5*x**4/128 + O(x**5)

# 定义一个名为 test_exp_1 的测试函数
def test_exp_1():
    assert exp(x).nseries(x, n=5) == 1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)
    assert exp(x).nseries(x, n=12) == 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 +  \
        x**6/720 + x**7/5040 + x**8/40320 + x**9/362880 + x**10/3628800 +  \
        x**11/39916800 + O(x**12)
    assert exp(1/x).nseries(x, n=5) == exp(1/x)
    assert exp(1/(1 + x)).nseries(x, n=4) ==  \
        (E*(1 - x - 13*x**3/6 + 3*x**2/2)).expand() + O(x**4)
    assert exp(2 + x).nseries(x, n=5) ==  \
        (exp(2)*(1 + x + x**2/2 + x**3/6 + x**4/24)).expand() + O(x**5)

# 定义一个名为 test_exp_sqrt_1 的测试函数
def test_exp_sqrt_1():
    # 断言：验证 exp(1 + sqrt(x)) 的 n 次级数展开是否等于 (exp(1) * (1 + sqrt(x) + x/2 + sqrt(x)*x/6)) 的展开加上 O(sqrt(x)**3)
    assert exp(1 + sqrt(x)).nseries(x, n=3) == \
        (exp(1)*(1 + sqrt(x) + x/2 + sqrt(x)*x/6)).expand() + O(sqrt(x)**3)
def test_power_x_x1():
    assert (exp(x*log(x))).nseries(x, n=4) == \
        1 + x*log(x) + x**2*log(x)**2/2 + x**3*log(x)**3/6 + O(x**4*log(x)**4)
    # 检验表达式 exp(x*log(x)) 的四阶近似级数是否与给定的级数相等

def test_power_x_x2():
    assert (x**x).nseries(x, n=4) == \
        1 + x*log(x) + x**2*log(x)**2/2 + x**3*log(x)**3/6 + O(x**4*log(x)**4)
    # 检验表达式 x**x 的四阶近似级数是否与给定的级数相等

def test_log_singular1():
    assert log(1 + 1/x).nseries(x, n=5) == x - log(x) - x**2/2 + x**3/3 - \
        x**4/4 + O(x**5)
    # 检验表达式 log(1 + 1/x) 的五阶近似级数是否与给定的级数相等

def test_log_power1():
    e = 1 / (1/x + x ** (log(3)/log(2)))
    assert e.nseries(x, n=5) == -x**(log(3)/log(2) + 2) + x + O(x**5)
    # 检验表达式 1 / (1/x + x ** (log(3)/log(2))) 的五阶近似级数是否与给定的级数相等

def test_log_series():
    l = Symbol('l')
    e = 1/(1 - log(x))
    assert e.nseries(x, n=5, logx=l) == 1/(1 - l)
    # 检验表达式 1/(1 - log(x)) 的五阶近似级数是否与给定的级数相等，其中 log(x) 用 l 表示

def test_log2():
    e = log(-1/x)
    assert e.nseries(x, n=5) == -log(x) + log(-1)
    # 检验表达式 log(-1/x) 的五阶近似级数是否与给定的级数相等

def test_log3():
    l = Symbol('l')
    e = 1/log(-1/x)
    assert e.nseries(x, n=4, logx=l) == 1/(-l + log(-1))
    # 检验表达式 1/log(-1/x) 的四阶近似级数是否与给定的级数相等，其中 log(-1/x) 用 l 表示

def test_series1():
    e = sin(x)
    assert e.nseries(x, 0, 0) != 0
    assert e.nseries(x, 0, 0) == O(1, x)
    assert e.nseries(x, 0, 1) == O(x, x)
    assert e.nseries(x, 0, 2) == x + O(x**2, x)
    assert e.nseries(x, 0, 3) == x + O(x**3, x)
    assert e.nseries(x, 0, 4) == x - x**3/6 + O(x**4, x)
    # 检验 sin(x) 在不同阶数下的近似级数是否符合预期

    e = (exp(x) - 1)/x
    assert e.nseries(x, 0, 3) == 1 + x/2 + x**2/6 + O(x**3)
    # 检验 (exp(x) - 1)/x 的三阶近似级数是否符合预期

    assert x.nseries(x, 0, 2) == x
    # 检验 x 的二阶近似级数是否符合预期

@XFAIL
def test_series1_failing():
    assert x.nseries(x, 0, 0) == O(1, x)
    assert x.nseries(x, 0, 1) == O(x, x)
    # 检验预期失败的情况下，x 的近似级数是否符合预期

def test_seriesbug1():
    assert (1/x).nseries(x, 0, 3) == 1/x
    assert (x + 1/x).nseries(x, 0, 3) == x + 1/x
    # 检验 (1/x) 和 (x + 1/x) 在三阶近似级数下是否符合预期

def test_series2x():
    assert ((x + 1)**(-2)).nseries(x, 0, 4) == 1 - 2*x + 3*x**2 - 4*x**3 + O(x**4, x)
    assert ((x + 1)**(-1)).nseries(x, 0, 4) == 1 - x + x**2 - x**3 + O(x**4, x)
    assert ((x + 1)**0).nseries(x, 0, 3) == 1
    assert ((x + 1)**1).nseries(x, 0, 3) == 1 + x
    assert ((x + 1)**2).nseries(x, 0, 3) == x**2 + 2*x + 1
    assert ((x + 1)**3).nseries(x, 0, 3) == 1 + 3*x + 3*x**2 + O(x**3)
    # 检验 (x + 1)**(-2)、(x + 1)**(-1)、(x + 1)**0、(x + 1)**1、(x + 1)**2、(x + 1)**3
    # 在不同阶数下的近似级数是否符合预期

    assert (1/(1 + x)).nseries(x, 0, 4) == 1 - x + x**2 - x**3 + O(x**4, x)
    assert (x + 3/(1 + 2*x)).nseries(x, 0, 4) == 3 - 5*x + 12*x**2 - 24*x**3 + O(x**4, x)
    # 检验 1/(1 + x) 和 x + 3/(1 + 2*x) 在四阶近似级数下是否符合预期

    assert ((1/x + 1)**3).nseries(x, 0, 3) == 1 + 3/x + 3/x**2 + x**(-3)
    assert (1/(1 + 1/x)).nseries(x, 0, 4) == x - x**2 + x**3 - O(x**4, x)
    assert (1/(1 + 1/x**2)).nseries(x, 0, 6) == x**2 - x**4 + O(x**6, x)
    # 检验 (1/x + 1)**3、1/(1 + 1/x) 和 1/(1 + 1/x**2) 在不同阶数下的近似级数是否符合预期

def test_bug2():  # 1/log(0)*log(0) problem
    w = Symbol("w")
    e = (w**(-1) + w**(
        -log(3)*log(2)**(-1)))**(-1)*(3*w**(-log(3)*log(2)**(-1)) + 2*w**(-1))
    e = e.expand()
    assert e.nseries(w, 0, 4).subs(w, 0) == 3
    # 检验特定表达式在 w 近似为 0 的情况下，其四阶近似级数是否符合预期，并验证其在 w=0 处的值是否为 3

def test_exp():
    e = (1 + x)**(1/x)
    assert e.nseries(x, n=3) == exp(1) - x*exp(1)/2 + 11*exp(1)*x**2/24 + O(x**3)
    # 检验表达式 (1 + x)**(1/x) 的三阶近似级数是否与给定的级数相等

def test_exp2():
    w = Symbol("w")
    e = w**(1 - log(x)/(log(2) + log(x)))
    logw = Symbol("logw")
    assert e.nseries(
        w, 0, 1, logx=logw) == exp(logw*log(2)/(log(x) + log(2)))
    # 检验表达式 w**(1 - log(x)/(log
    # 断言：验证调用对象 e 的 nseries 方法对于给定的参数 x 和 n=3 的结果是否等于 3 - x + x**2 + O(x**3)。
    assert e.nseries(x, n=3) == 3 - x + x**2 + O(x**3)
# 定义测试函数 test_generalexponent
def test_generalexponent():
    # 设置变量 p 为 2
    p = 2
    # 计算表达式 e，这里 x 是一个符号变量
    e = (2/x + 3/x**p)/(1/x + 1/x**p)
    # 断言 e 在 x 展开至 0 附近的前 3 阶项系数符合期望值
    assert e.nseries(x, 0, 3) == 3 - x + x**2 + O(x**3)
    
    # 将 p 设置为 S.Half（1/2）
    p = S.Half
    # 重新计算表达式 e，使用 S.Half 代替 2
    e = (2/x + 3/x**p)/(1/x + 1/x**p)
    # 断言 e 在 x 展开至 0 附近的前 2 阶项系数符合期望值
    assert e.nseries(x, 0, 2) == 2 - x + sqrt(x) + x**(S(3)/2) + O(x**2)
    
    # 设置 e 为一个简单的表达式 1 + sqrt(x)
    e = 1 + sqrt(x)
    # 断言 e 在 x 展开至 0 附近的前 4 阶项系数符合期望值
    assert e.nseries(x, 0, 4) == 1 + sqrt(x)


# 更复杂的例子


# 定义测试函数 test_genexp_x
def test_genexp_x():
    # 设置表达式 e 为 1/(1 + sqrt(x))
    e = 1/(1 + sqrt(x))
    # 断言 e 在 x 展开至 0 附近的前 2 阶项系数符合期望值
    assert e.nseries(x, 0, 2) == \
        1 + x - sqrt(x) - sqrt(x)**3 + O(x**2, x)


# 更复杂的例子


# 定义测试函数 test_genexp_x2
def test_genexp_x2():
    # 设置 p 为 Rational(3, 2)
    p = Rational(3, 2)
    # 计算表达式 e
    e = (2/x + 3/x**p)/(1/x + 1/x**p)
    # 断言 e 在 x 展开至 0 附近的前 3 阶项系数符合期望值
    assert e.nseries(x, 0, 3) == 3 + x + x**2 - sqrt(x) - x**(S(3)/2) - x**(S(5)/2) + O(x**3)


# 定义测试函数 test_seriesbug2
def test_seriesbug2():
    # 创建符号变量 w
    w = Symbol("w")
    # 简单情况 (1)
    e = ((2*w)/w)**(1 + w)
    # 断言 e 在 w 展开至 0 附近的前 1 阶项系数符合期望值
    assert e.nseries(w, 0, 1) == 2 + O(w, w)
    # 断言 e 在将 w 替换为 0 后的值符合期望值 2
    assert e.nseries(w, 0, 1).subs(w, 0) == 2


# 更复杂的例子


# 定义测试函数 test_seriesbug2b
def test_seriesbug2b():
    # 创建符号变量 w
    w = Symbol("w")
    # 测试 sin
    e = sin(2*w)/w
    # 断言 e 在 w 展开至 0 附近的前 3 阶项系数符合期望值
    assert e.nseries(w, 0, 3) == 2 - 4*w**2/3 + O(w**3)


# 定义测试函数 test_seriesbug2d
def test_seriesbug2d():
    # 创建实数符号变量 w
    w = Symbol("w", real=True)
    # 计算表达式 e
    e = log(sin(2*w)/w)
    # 断言 e 在 w 展开至 0 附近的前 5 阶项系数符合期望值
    assert e.series(w, n=5) == log(2) - 2*w**2/3 - 4*w**4/45 + O(w**5)


# 定义测试函数 test_seriesbug2c
def test_seriesbug2c():
    # 创建实数符号变量 w
    w = Symbol("w", real=True)
    # 更复杂的情况，但是 sin(x)~x，所以结果与 (1) 中的相同
    e = (sin(2*w)/w)**(1 + w)
    # 断言 e 在 w 展开至 0 附近的前 1 阶项系数符合期望值
    assert e.series(w, 0, 1) == 2 + O(w)
    # 断言 e 在 w 展开至 0 附近的前 3 阶项系数符合期望值
    assert e.series(w, 0, 3) == 2 + 2*w*log(2) + \
        w**2*(Rational(-4, 3) + log(2)**2) + O(w**3)
    # 断言 e 在将 w 替换为 0 后的值符合期望值 2
    assert e.series(w, 0, 2).subs(w, 0) == 2


# 定义测试函数 test_expbug4
def test_expbug4():
    # 创建实数符号变量 x
    x = Symbol("x", real=True)
    # 断言计算结果与期望值相等
    assert (log(
        sin(2*x)/x)*(1 + x)).series(x, 0, 2) == log(2) + x*log(2) + O(x**2, x)
    # 断言计算结果与期望值相等
    assert exp(
        log(sin(2*x)/x)*(1 + x)).series(x, 0, 2) == 2 + 2*x*log(2) + O(x**2)

    # 断言计算结果与期望值相等
    assert exp(log(2) + O(x)).nseries(x, 0, 2) == 2 + O(x)
    # 断言计算结果与期望值相等
    assert ((2 + O(x))**(1 + x)).nseries(x, 0, 2) == 2 + O(x)


# 定义测试函数 test_logbug4
def test_logbug4():
    # 断言计算结果与期望值相等
    assert log(2 + O(x)).nseries(x, 0, 2) == log(2) + O(x, x)


# 定义测试函数 test_expbug5
def test_expbug5():
    # 断言计算结果与期望值相等
    assert exp(log(1 + x)/x).nseries(x, n=3) == exp(1) + -exp(1)*x/2 + 11*exp(1)*x**2/24 + O(x**3)

    # 断言计算结果与期望值相等
    assert exp(O(x)).nseries(x, 0, 2) == 1 + O(x)


# 定义测试函数 test_sinsinbug
def test_sinsinbug():
    # 断言计算结果与期望值相等
    assert sin(sin(x)).nseries(x, 0, 8) == x - x**3/3 + x**5/10 - 8*x**7/315 + O(x**8)


# 定义测试函数 test_issue_3258
def test_issue_3258():
    # 创建表达式 a
    a = x/(exp(x) - 1)
    # 断言计算结果与期望值相等
    assert a.nseries(x, 0, 5) == 1 - x/2 - x**4/720 + x**2/12 + O(x**5)


# 定义测试函数 test_issue_3204
def test_issue_3204():
    # 创建非负实数符号变量 x
    x = Symbol("x", nonnegative=True)
    # 创建表达式 f
    f = sin(x**3)**Rational(1, 3)
    # 断言计算结果与期望
    # 断言：验证 sin(x + y) 在 x 的 n 阶级数展开结果是否等于 sin(y) + O(x)
    assert sin(x + y).nseries(x, n=1) == sin(y) + O(x)
    
    # 断言：验证 sin(x + y) 在 x 的 n=2 阶级数展开结果是否等于 sin(y) + cos(y)*x + O(x**2)
    assert sin(x + y).nseries(x, n=2) == sin(y) + cos(y)*x + O(x**2)
    
    # 断言：验证 sin(x + y) 在 x 的 n=5 阶级数展开结果是否等于 sin(y) + cos(y)*x - sin(y)*x**2/2 - \
    #        cos(y)*x**3/6 + sin(y)*x**4/24 + O(x**5)
    assert sin(x + y).nseries(x, n=5) == sin(y) + cos(y)*x - sin(y)*x**2/2 - \
        cos(y)*x**3/6 + sin(y)*x**4/24 + O(x**5)
def test_issue_3515():
    # 定义表达式 e
    e = sin(8*x)/x
    # 断言表达式 e 在 x = 0 处展开到 n=6 阶的结果
    assert e.nseries(x, n=6) == 8 - 256*x**2/3 + 4096*x**4/15 + O(x**6)


def test_issue_3505():
    # 定义表达式 e
    e = sin(x)**(-4)*(sqrt(cos(x))*sin(x)**2 - cos(x)**Rational(1, 3)*sin(x)**2)
    # 断言表达式 e 在 x = 0 处展开到 n=9 阶的结果
    assert e.nseries(x, n=9) == Rational(-1, 12) - 7*x**2/288 - \
        43*x**4/10368 - 1123*x**6/2488320 + 377*x**8/29859840 + O(x**9)


def test_issue_3501():
    # 定义符号变量 a
    a = Symbol("a")
    # 定义表达式 e1
    e = x**(-2)*(x*sin(a + x) - x*sin(a))
    # 断言表达式 e1 在 x = 0 处展开到 n=6 阶的结果
    assert e.nseries(x, n=6) == cos(a) - sin(a)*x/2 - cos(a)*x**2/6 + \
        x**3*sin(a)/24 + x**4*cos(a)/120 - x**5*sin(a)/720 + O(x**6)
    # 重新定义表达式 e2
    e = x**(-2)*(x*cos(a + x) - x*cos(a))
    # 断言表达式 e2 在 x = 0 处展开到 n=6 阶的结果
    assert e.nseries(x, n=6) == -sin(a) - cos(a)*x/2 + sin(a)*x**2/6 + \
        cos(a)*x**3/24 - x**4*sin(a)/120 - x**5*cos(a)/720 + O(x**6)


def test_issue_3502():
    # 定义表达式 e
    e = sin(5*x)/sin(2*x)
    # 断言表达式 e 在 x = 0 处展开到 n=2 阶的结果
    assert e.nseries(x, n=2) == Rational(5, 2) + O(x**2)
    # 断言表达式 e 在 x = 0 处展开到 n=6 阶的结果
    assert e.nseries(x, n=6) == \
        Rational(5, 2) - 35*x**2/4 + 329*x**4/48 + O(x**6)


def test_issue_3503():
    # 定义表达式 e
    e = sin(2 + x)/(2 + x)
    # 断言表达式 e 在 x = 0 处展开到 n=2 阶的结果
    assert e.nseries(x, n=2) == sin(2)/2 + x*cos(2)/2 - x*sin(2)/4 + O(x**2)


def test_issue_3506():
    # 定义表达式 e
    e = (x + sin(3*x))**(-2)*(x*(x + sin(3*x)) - (x + sin(3*x))*sin(2*x))
    # 断言表达式 e 在 x = 0 处展开到 n=7 阶的结果
    assert e.nseries(x, n=7) == \
        Rational(-1, 4) + 5*x**2/96 + 91*x**4/768 + 11117*x**6/129024 + O(x**7)


def test_issue_3508():
    # 定义符号变量 x
    x = Symbol("x", real=True)
    # 断言 log(sin(x)) 的级数展开结果
    assert log(sin(x)).series(x, n=5) == log(x) - x**2/6 - x**4/180 + O(x**5)
    # 定义表达式 e
    e = -log(x) + x*(-log(x) + log(sin(2*x))) + log(sin(2*x))
    # 断言表达式 e 在 x = 0 处展开到 n=5 阶的结果
    assert e.series(x, n=5) == \
        log(2) + log(2)*x - 2*x**2/3 - 2*x**3/3 - 4*x**4/45 + O(x**5)


def test_issue_3507():
    # 定义表达式 e
    e = x**(-4)*(x**2 - x**2*sqrt(cos(x)))
    # 断言表达式 e 在 x = 0 处展开到 n=9 阶的结果
    assert e.nseries(x, n=9) == \
        Rational(1, 4) + x**2/96 + 19*x**4/5760 + 559*x**6/645120 + 29161*x**8/116121600 + O(x**9)


def test_issue_3639():
    # 断言 sin(cos(x)) 的级数展开结果
    assert sin(cos(x)).nseries(x, n=5) == \
        sin(1) - x**2*cos(1)/2 - x**4*sin(1)/8 + x**4*cos(1)/24 + O(x**5)


def test_hyperbolic():
    # 断言双曲正弦函数的级数展开结果
    assert sinh(x).nseries(x, n=6) == x + x**3/6 + x**5/120 + O(x**6)
    # 断言双曲余弦函数的级数展开结果
    assert cosh(x).nseries(x, n=5) == 1 + x**2/2 + x**4/24 + O(x**5)
    # 断言双曲正切函数的级数展开结果
    assert tanh(x).nseries(x, n=6) == x - x**3/3 + 2*x**5/15 + O(x**6)
    # 断言双曲余切函数的级数展开结果
    assert coth(x).nseries(x, n=6) == \
        1/x - x**3/45 + x/3 + 2*x**5/945 + O(x**6)
    # 断言反双曲正弦函数的级数展开结果
    assert asinh(x).nseries(x, n=6) == x - x**3/6 + 3*x**5/40 + O(x**6)
    # 断言反双曲余弦函数的级数展开结果
    assert acosh(x).nseries(x, n=6) == \
        pi*I/2 - I*x - 3*I*x**5/40 - I*x**3/6 + O(x**6)
    # 断言反双曲正切函数的级数展开结果
    assert atanh(x).nseries(x, n=6) == x + x**3/3 + x**5/5 + O(x**6)
    # 断言反双曲余切函数的级数展开结果
    assert acoth(x).nseries(x, n=6) == -I*pi/2 + x + x**3/3 + x**5/5 + O(x**6)


def test_series2():
    # 定义符号变量 w 和 x
    w = Symbol("w", real=True)
    x = Symbol("x", real=True)
    # 定义表达式 e
    e = w**(-2)*(w*exp(1/x - w) - w*exp(1/x))
    # 断言表达式 e 在 w = 0 处展开到 n=4 阶的结果
    assert e.nseries(w, n=4) == -exp(1/x) + w*exp(1/x)/2 - w**2*exp(1/x)/6 + w**3*exp(1/x)/24 + O(w**4)


def test_series3():
    # 定义符号变量 w
    w = Symbol("w", real=True)
    # 定义表达式 e
    e = w**(-6)*(w**3*tan(w) - w**3*sin(w))
    # 断言：验证 e 对象调用 nseries 方法返回的结果是否等于指定的表达式
    assert e.nseries(w, n=8) == Integer(1)/2 + w**2/8 + 13*w**4/240 + 529*w**6/24192 + O(w**8)
def test_bug4():
    # 创建符号变量 w
    w = Symbol("w")
    # 定义表达式 e
    e = x/(w**4 + x**2*w**4 + 2*x*w**4)*w**4
    # 断言表达式 e 在 w 的 n 阶数列展开中移除 O 大O项，并展开后的结果在指定列表中
    assert e.nseries(w, n=2).removeO().expand() in [x/(1 + 2*x + x**2),
        1/(1 + x/2 + 1/x/2)/2, 1/x/(1 + 2/x + x**(-2))]


def test_bug5():
    # 创建符号变量 w 和 l
    w = Symbol("w")
    l = Symbol('l')
    # 定义表达式 e
    e = (-log(w) + log(1 + w*log(x)))**(-2)*w**(-2)*((-log(w) +
        log(1 + x*w))*(-log(w) + log(1 + w*log(x)))*w - x*(-log(w) +
        log(1 + w*log(x)))*w)
    # 断言表达式 e 在 w 的 n 阶数列展开中，带有对数项 logx=l 的结果
    assert e.nseries(w, n=0, logx=l) == x/w/l + 1/w + O(1, w)
    # 断言表达式 e 在 w 的 n=1 阶数列展开中，带有对数项 logx=l 的结果
    assert e.nseries(w, n=1, logx=l) == x/w/l + 1/w - x/l + 1/l*log(x) \
        + x*log(x)/l**2 + O(w)


def test_issue_4115():
    # 断言 sin(x)/(1 - cos(x)) 在 x=0 处的 n=1 阶数列展开结果
    assert (sin(x)/(1 - cos(x))).nseries(x, n=1) == 2/x + O(x)
    # 断言 sin(x)**2/(1 - cos(x)) 在 x=0 处的 n=1 阶数列展开结果
    assert (sin(x)**2/(1 - cos(x))).nseries(x, n=1) == 2 + O(x)


def test_pole():
    # 断言 sin(1/x) 的 x=0 处的数列展开结果会引发 PoleError
    raises(PoleError, lambda: sin(1/x).series(x, 0, 5))
    # 断言 sin(1 + 1/x) 的 x=0 处的数列展开结果会引发 PoleError
    raises(PoleError, lambda: sin(1 + 1/x).series(x, 0, 5))
    # 断言 x*sin(1/x) 的 x=0 处的数列展开结果会引发 PoleError
    raises(PoleError, lambda: (x*sin(1/x)).series(x, 0, 5))


def test_expsinbug():
    # 断言 exp(sin(x)) 在 x=0 处的 n=0 阶数列展开结果
    assert exp(sin(x)).series(x, 0, 0) == O(1, x)
    # 断言 exp(sin(x)) 在 x=0 处的 n=1 阶数列展开结果
    assert exp(sin(x)).series(x, 0, 1) == 1 + O(x)
    # 断言 exp(sin(x)) 在 x=0 处的 n=2 阶数列展开结果
    assert exp(sin(x)).series(x, 0, 2) == 1 + x + O(x**2)
    # 断言 exp(sin(x)) 在 x=0 处的 n=3 阶数列展开结果
    assert exp(sin(x)).series(x, 0, 3) == 1 + x + x**2/2 + O(x**3)
    # 断言 exp(sin(x)) 在 x=0 处的 n=4 阶数列展开结果
    assert exp(sin(x)).series(x, 0, 4) == 1 + x + x**2/2 + O(x**4)
    # 断言 exp(sin(x)) 在 x=0 处的 n=5 阶数列展开结果
    assert exp(sin(x)).series(x, 0, 5) == 1 + x + x**2/2 - x**4/8 + O(x**5)


def test_floor():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言 floor(x) 在 x=0 处的数列展开结果
    assert floor(x).series(x) == 0
    # 断言 floor(-x) 在 x=0 处的数列展开结果
    assert floor(-x).series(x) == -1
    # 断言 floor(sin(x)) 在 x=0 处的数列展开结果
    assert floor(sin(x)).series(x) == 0
    # 断言 floor(sin(-x)) 在 x=0 处的数列展开结果
    assert floor(sin(-x)).series(x) == -1
    # 断言 floor(x**3) 在 x=0 处的数列展开结果
    assert floor(x**3).series(x) == 0
    # 断言 floor(-x**3) 在 x=0 处的数列展开结果
    assert floor(-x**3).series(x) == -1
    # 断言 floor(cos(x)) 在 x=0 处的数列展开结果
    assert floor(cos(x)).series(x) == 0
    # 断言 floor(cos(-x)) 在 x=0 处的数列展开结果
    assert floor(cos(-x)).series(x) == 0
    # 断言 floor(5 + sin(x)) 在 x=0 处的数列展开结果
    assert floor(5 + sin(x)).series(x) == 5
    # 断言 floor(5 + sin(-x)) 在 x=0 处的数列展开结果
    assert floor(5 + sin(-x)).series(x) == 4

    # 断言 floor(x) 在 x=2 处的数列展开结果
    assert floor(x).series(x, 2) == 2
    # 断言 floor(-x) 在 x=2 处的数列展开结果
    assert floor(-x).series(x, 2) == -3

    # 创建符号变量 x，且为负数
    x = Symbol('x', negative=True)
    # 断言 floor(x + 1.5) 在 x=0 处的数列展开结果
    assert floor(x + 1.5).series(x) == 1


def test_frac():
    # 断言 frac(x) 在 x=0 处的数列展开结果，cdir=1
    assert frac(x).series(x, cdir=1) == x
    # 断言 frac(x) 在 x=0 处的数列展开结果，cdir=-1
    assert frac(x).series(x, cdir=-1) == 1 + x
    # 断言 frac(2*x + 1) 在 x=0 处的数列展开结果，cdir=1
    assert frac(2*x + 1).series(x, cdir=1) == 2*x
    # 断言 frac(2*x + 1) 在 x=0 处的数列展开结果，cdir=-1
    assert frac(2*x + 1).series(x, cdir=-1) == 1 + 2*x
    # 断言 frac(x**2) 在 x=0 处的数列展开结果，cdir=1
    assert frac(x**2).series(x, cdir=1) == x**2
    # 断言 frac(x**2) 在 x=0 处的数列展开结果，cdir=-1
    assert frac(x**2).series(x, cdir=-1) == x**2
    # 断言 frac(sin(x) + 5) 在 x=0 处的数列展开结果，cdir=1
    assert frac(sin(x) + 5).series(x, cdir=1) == x - x**3/6 + x**5/120 + O(x**6)
    # 断言 frac(sin(x) + 5) 在 x=0 处的数列展开结果，cdir=-1
    assert frac(sin(x) + 5).series(x, cdir=-1) == 1 + x - x**3/6 + x**5/120 + O(x**6)
    # 断言 frac(sin(x) + S
    # 创建一个符号对象 'a'
    a = Symbol('a')
    
    # 断言：计算 x 的 n 次级数展开，并验证其结果等于 x
    assert abs(x).nseries(x, n=4) == x
    
    # 断言：计算 -x 的 n 次级数展开，并验证其结果等于 x
    assert abs(-x).nseries(x, n=4) == x
    
    # 断言：计算 x + 1 的 n 次级数展开，并验证其结果等于 x + 1
    assert abs(x + 1).nseries(x, n=4) == x + 1
    
    # 断言：计算 sin(x) 的 n 次级数展开，并验证其结果等于 x - Rational(1, 6)*x**3 + O(x**4)
    assert abs(sin(x)).nseries(x, n=4) == x - Rational(1, 6)*x**3 + O(x**4)
    
    # 断言：计算 sin(-x) 的 n 次级数展开，并验证其结果等于 x - Rational(1, 6)*x**3 + O(x**4)
    assert abs(sin(-x)).nseries(x, n=4) == x - Rational(1, 6)*x**3 + O(x**4)
    
    # 断言：计算 x - a 的 1 次级数展开，并验证其结果等于 -a*sign(1 - a) + (x - 1)*sign(1 - a) + sign(1 - a)
    assert abs(x - a).nseries(x, 1) == -a*sign(1 - a) + (x - 1)*sign(1 - a) + sign(1 - a)
def test_dir():
    # 检查绝对值函数的级数展开，方向为正时应返回原始值
    assert abs(x).series(x, 0, dir="+") == x
    # 检查绝对值函数的级数展开，方向为负时应返回相反数
    assert abs(x).series(x, 0, dir="-") == -x
    # 检查向下取整函数的级数展开，增加2后方向为正时应返回2
    assert floor(x + 2).series(x, 0, dir='+') == 2
    # 检查向下取整函数的级数展开，增加2后方向为负时应返回1
    assert floor(x + 2).series(x, 0, dir='-') == 1
    # 检查向下取整函数的级数展开，增加2.2后方向为负时应返回2
    assert floor(x + 2.2).series(x, 0, dir='-') == 2
    # 检查向上取整函数的级数展开，增加2.2后方向为负时应返回3
    assert ceiling(x + 2.2).series(x, 0, dir='-') == 3
    # 检查正弦函数的级数展开，增加x + y后方向为负时应与方向为正时结果相同
    assert sin(x + y).series(x, 0, dir='-') == sin(x + y).series(x, 0, dir='+')


def test_cdir():
    # 检查绝对值函数的级数展开，cdir参数为1时应返回原始值
    assert abs(x).series(x, 0, cdir=1) == x
    # 检查绝对值函数的级数展开，cdir参数为-1时应返回相反数
    assert abs(x).series(x, 0, cdir=-1) == -x
    # 检查向下取整函数的级数展开，增加2后cdir参数为1时应返回2
    assert floor(x + 2).series(x, 0, cdir=1) == 2
    # 检查向下取整函数的级数展开，增加2后cdir参数为-1时应返回1
    assert floor(x + 2).series(x, 0, cdir=-1) == 1
    # 检查向下取整函数的级数展开，增加2.2后cdir参数为1时应返回2
    assert floor(x + 2.2).series(x, 0, cdir=1) == 2
    # 检查向上取整函数的级数展开，增加2.2后cdir参数为-1时应返回3
    assert ceiling(x + 2.2).series(x, 0, cdir=-1) == 3
    # 检查正弦函数的级数展开，增加x + y后cdir参数为-1时应与cdir参数为1时结果相同
    assert sin(x + y).series(x, 0, cdir=-1) == sin(x + y).series(x, 0, cdir=1)


def test_issue_3504():
    # 创建符号变量'a'
    a = Symbol("a")
    # 构造asin(a*x)/x表达式
    e = asin(a*x)/x
    # 检查该表达式在x=4时的二阶级数展开，并移除高阶项
    assert e.series(x, 4, n=2).removeO() == \
        (x - 4)*(a/(4*sqrt(-16*a**2 + 1)) - asin(4*a)/16) + asin(4*a)/4


def test_issue_4441():
    # 创建符号变量'a'和'b'
    a, b = symbols('a,b')
    # 构造1/(1 + a*x)表达式
    f = 1/(1 + a*x)
    # 检查该表达式在x=0时展开到五阶级数
    assert f.series(x, 0, 5) == 1 - a*x + a**2*x**2 - a**3*x**3 + \
        a**4*x**4 + O(x**5)
    # 构造1/(1 + (a + b)*x)表达式
    f = 1/(1 + (a + b)*x)
    # 检查该表达式在x=0时展开到三阶级数
    assert f.series(x, 0, 3) == 1 + x*(-a - b)\
        + x**2*(a + b)**2 + O(x**3)


def test_issue_4329():
    # 检查正切函数tan(x)在x=pi/2附近展开到三阶级数，并移除高阶项
    assert tan(x).series(x, pi/2, n=3).removeO() == \
        -pi/6 + x/3 - 1/(x - pi/2)
    # 检查余切函数cot(x)在x=pi附近展开到三阶级数，并移除高阶项
    assert cot(x).series(x, pi, n=3).removeO() == \
        -x/3 + pi/3 + 1/(x - pi)
    # 检查tan(x)^tan(2*x)函数在x=pi/4处的极限是否等于exp(-1)
    assert limit(tan(x)**tan(2*x), x, pi/4) == exp(-1)


def test_issue_5183():
    # 检查绝对值函数abs(x + x**2)在一阶级数展开中是否为O(x)
    assert abs(x + x**2).series(n=1) == O(x)
    # 检查绝对值函数abs(x + x**2)在二阶级数展开中是否为x + O(x**2)
    assert abs(x + x**2).series(n=2) == x + O(x**2)
    # 检查(1 + x)^2在x=0处展开到六阶级数是否为x^2 + 2*x + 1
    assert ((1 + x)**2).series(x, n=6) == x**2 + 2*x + 1
    # 检查(1 + 1/x)在默认x=0处展开是否为1 + 1/x
    assert (1 + 1/x).series() == 1 + 1/x
    # 检查exp(x)的级数展开，并对其求导后是否得到1 + x + x^2/2 + x^3/6 + x^4/24 + O(x^5)
    assert Derivative(exp(x).series(), x).doit() == \
        1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)


def test_issue_5654():
    # 创建符号变量'a'
    a = Symbol('a')
    # 检查(1/(x**2+a**2)**2)在x0=I*a处展开到零阶是否等于预期值
    assert (1/(x**2+a**2)**2).nseries(x, x0=I*a, n=0) == \
        -I/(4*a**3*(-I*a + x)) - 1/(4*a**2*(-I*a + x)**2) + O(1, (x, I*a))
    # 检查(1/(x**2+a**2)**2)在x0=I*a处展开到一阶是否等于预期值
    assert (1/(x**2+a**2)**2).nseries(x, x0=I*a, n=1) == 3/(16*a**4) \
        -I/(4*a**3*(-I*a + x)) - 1/(4*a**2*(-I*a + x)**2) + O(-I*a + x, (x, I*a))


def test_issue_5925():
    # 对sqrt(x + z)在z=0处展开一阶级数
    sx = sqrt(x + z).series(z, 0, 1)
    # 对sqrt(x + y + z)在z=0处展开一阶级数
    sxy = sqrt(x + y + z).series(z, 0, 1)
    # 将sx中的x替换为x + y后与sxy
```