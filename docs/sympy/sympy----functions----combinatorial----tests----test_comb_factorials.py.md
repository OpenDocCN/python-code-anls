# `D:\src\scipysrc\sympy\sympy\functions\combinatorial\tests\test_comb_factorials.py`

```
# 导入符号计算库中的特定模块和函数

from sympy.concrete.products import Product  # 导入产品（积）相关的函数
from sympy.core.function import expand_func  # 导入函数扩展相关的函数
from sympy.core.mod import Mod  # 导入模运算相关的函数
from sympy.core.mul import Mul  # 导入乘法相关的函数
from sympy.core import EulerGamma  # 导入欧拉常数
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)  # 导入各种数学常数
from sympy.core.relational import Eq  # 导入关系运算相关的函数
from sympy.core.singleton import S  # 导入单例相关的函数
from sympy.core.symbol import (Dummy, Symbol, symbols)  # 导入符号相关的函数
from sympy.functions.combinatorial.factorials import (ff, rf, binomial, factorial, factorial2)  # 导入组合数学函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.special.gamma_functions import (gamma, polygamma)  # 导入伽马函数及其变体
from sympy.polys.polytools import Poly  # 导入多项式相关的函数
from sympy.series.order import O  # 导入大O符号相关的函数
from sympy.simplify.simplify import simplify  # 导入简化函数
from sympy.core.expr import unchanged  # 导入未变表达式相关的函数
from sympy.core.function import ArgumentIndexError  # 导入参数索引错误相关的函数
from sympy.functions.combinatorial.factorials import subfactorial  # 导入子阶乘函数
from sympy.functions.special.gamma_functions import uppergamma  # 导入上伽马函数
from sympy.testing.pytest import XFAIL, raises, slow  # 导入测试相关的函数

# 解决和修复问题＃10388的测试更新 - 这是相同解决问题的更新测试
def test_rf_eval_apply():
    # 定义符号变量
    x, y = symbols('x,y')
    n, k = symbols('n k', integer=True)
    m = Symbol('m', integer=True, nonnegative=True)

    # 检查特殊情况下的rf函数返回结果
    assert rf(nan, y) is nan
    assert rf(x, nan) is nan

    # 检查rf函数在不变调用时的行为
    assert unchanged(rf, x, y)

    # 检查rf函数对无穷大和负无穷大的处理
    assert rf(oo, 0) == 1
    assert rf(-oo, 0) == 1

    assert rf(oo, 6) is oo
    assert rf(-oo, 7) is -oo
    assert rf(-oo, 6) is oo

    assert rf(oo, -6) is oo
    assert rf(-oo, -7) is oo

    # 检查rf函数对复数参数的处理
    assert rf(-1, pi) == 0
    assert rf(-5, 1 + I) == 0

    # 检查rf函数在某些参数下的不变性
    assert unchanged(rf, -3, k)
    assert unchanged(rf, x, Symbol('k', integer=False))
    assert rf(-3, Symbol('k', integer=False)) == 0
    assert rf(Symbol('x', negative=True, integer=True), Symbol('k', integer=False)) == 0

    # 检查rf函数在非负整数参数下的值
    assert rf(x, 0) == 1
    assert rf(x, 1) == x
    assert rf(x, 2) == x*(x + 1)
    assert rf(x, 3) == x*(x + 1)*(x + 2)
    assert rf(x, 5) == x*(x + 1)*(x + 2)*(x + 3)*(x + 4)

    # 检查rf函数在负整数参数下的值
    assert rf(x, -1) == 1/(x - 1)
    assert rf(x, -2) == 1/((x - 1)*(x - 2))
    assert rf(x, -3) == 1/((x - 1)*(x - 2)*(x - 3))

    # 检查rf函数在特定参数下的阶乘计算
    assert rf(1, 100) == factorial(100)

    # 检查rf函数在多项式参数下的值和类型
    assert rf(x**2 + 3*x, 2) == (x**2 + 3*x)*(x**2 + 3*x + 1)
    assert isinstance(rf(x**2 + 3*x, 2), Mul)
    assert rf(x**3 + x, -2) == 1/((x**3 + x - 1)*(x**3 + x - 2))

    # 检查rf函数在多项式参数下的多项式对象输出
    assert rf(Poly(x**2 + 3*x, x), 2) == Poly(x**4 + 8*x**3 + 19*x**2 + 12*x, x)
    assert isinstance(rf(Poly(x**2 + 3*x, x), 2), Poly)
    raises(ValueError, lambda: rf(Poly(x**2 + 3*x, x, y), 2))
    assert rf(Poly(x**3 + x, x), -2) == 1/(x**6 - 9*x**5 + 35*x**4 - 75*x**3 + 94*x**2 - 66*x + 20)
    raises(ValueError, lambda: rf(Poly(x**3 + x, x, y), -2))

    # 检查rf函数在混合参数类型下的整数性质
    assert rf(x, m).is_integer is None
    assert rf(n, k).is_integer is None
    assert rf(n, m).is_integer is True
    assert rf(n, k + pi).is_integer is False
    assert rf(n, m + pi).is_integer is False
    assert rf(pi, m).is_integer is False
    # 定义一个函数 check，用于验证给定的操作符 o 对于参数 x 和 k 的重写操作是否正确
    def check(x, k, o, n):
        # 创建两个虚拟对象 a 和 b
        a, b = Dummy(), Dummy()
        # 定义一个 lambda 函数 r，用于执行操作符 o 对 a 和 b 的重写操作，然后进行变量替换
        r = lambda x, k: o(a, b).rewrite(n).subs({a:x,b:k})
        # 循环遍历范围为 [-5, 5) 的 i 和 j 值
        for i in range(-5,5):
          for j in range(-5,5):
              # 断言 o(i, j) 应该等于 r(i, j)，否则抛出异常
              assert o(i, j) == r(i, j), (o, n, i, j)

    # 使用不同的参数调用 check 函数，用于验证不同的操作符和重写规则
    check(x, k, rf, ff)
    check(x, k, rf, binomial)
    check(n, k, rf, factorial)
    check(x, y, rf, factorial)
    check(x, y, rf, binomial)

    # 断言 rf(x, k) 重写为 ff 后的结果应该等于 ff(x + k - 1, k)
    assert rf(x, k).rewrite(ff) == ff(x + k - 1, k)

    # 断言 rf(x, k) 重写为 gamma 后的结果应该等于 Piecewise 对象
    assert rf(x, k).rewrite(gamma) == Piecewise(
        (gamma(k + x)/gamma(x), x > 0),
        ((-1)**k*gamma(1 - x)/gamma(-k - x + 1), True))

    # 断言 rf(5, k) 重写为 gamma 后的结果应该等于 gamma(k + 5)/24
    assert rf(5, k).rewrite(gamma) == gamma(k + 5)/24

    # 断言 rf(x, k) 重写为 binomial 后的结果应该等于 factorial(k)*binomial(x + k - 1, k)
    assert rf(x, k).rewrite(binomial) == factorial(k)*binomial(x + k - 1, k)

    # 断言 rf(n, k) 重写为 factorial 后的结果应该等于 Piecewise 对象
    assert rf(n, k).rewrite(factorial) == Piecewise(
        (factorial(k + n - 1)/factorial(n - 1), n > 0),
        ((-1)**k*factorial(-n)/factorial(-k - n), True))

    # 断言 rf(5, k) 重写为 factorial 后的结果应该等于 factorial(k + 4)/24
    assert rf(5, k).rewrite(factorial) == factorial(k + 4)/24

    # 断言 rf(x, y) 重写为 factorial 后的结果应该等于 rf(x, y) 自身
    assert rf(x, y).rewrite(factorial) == rf(x, y)

    # 断言 rf(x, y) 重写为 binomial 后的结果应该等于 rf(x, y) 自身
    assert rf(x, y).rewrite(binomial) == rf(x, y)

    # 导入随机模块和 mpmath 库的 rf 函数，并进行 100 次随机测试
    import random
    from mpmath import rf as mpmath_rf
    for i in range(100):
        # 随机生成 x 和 k 的值，范围为 [-500, 0) ∪ (0, 500]
        x = -500 + 500 * random.random()
        k = -500 + 500 * random.random()
        # 断言 mpmath 的 rf 函数计算结果与当前定义的 rf 函数结果的误差小于 10**(-15)
        assert (abs(mpmath_rf(x, k) - rf(x, k)) < 10**(-15))
# 定义一个测试函数 test_ff_eval_apply，用于测试 ff 函数的各种情况
def test_ff_eval_apply():
    # 创建符号变量 x 和 y
    x, y = symbols('x,y')
    # 创建整数符号变量 n 和 k
    n, k = symbols('n k', integer=True)
    # 创建非负整数符号变量 m
    m = Symbol('m', integer=True, nonnegative=True)

    # 断言 ff 函数对于 nan 返回 nan
    assert ff(nan, y) is nan
    # 断言 ff 函数对于 x 和 nan 返回 nan
    assert ff(x, nan) is nan

    # 断言 unchanged 函数可以保持 ff 函数不变
    assert unchanged(ff, x, y)

    # 测试 ff 函数对于无穷大和 0 的情况
    assert ff(oo, 0) == 1
    assert ff(-oo, 0) == 1

    # 测试 ff 函数对于无穷大和正整数的情况
    assert ff(oo, 6) is oo
    assert ff(-oo, 7) is -oo
    assert ff(-oo, 6) is oo

    # 测试 ff 函数对于无穷大和负整数的情况
    assert ff(oo, -6) is oo
    assert ff(-oo, -7) is oo

    # 测试 ff 函数对于正整数的情况
    assert ff(x, 0) == 1
    assert ff(x, 1) == x
    assert ff(x, 2) == x*(x - 1)
    assert ff(x, 3) == x*(x - 1)*(x - 2)
    assert ff(x, 5) == x*(x - 1)*(x - 2)*(x - 3)*(x - 4)

    # 测试 ff 函数对于负整数的情况
    assert ff(x, -1) == 1/(x + 1)
    assert ff(x, -2) == 1/((x + 1)*(x + 2))
    assert ff(x, -3) == 1/((x + 1)*(x + 2)*(x + 3))

    # 测试 ff 函数对于大整数的情况
    assert ff(100, 100) == factorial(100)

    # 测试 ff 函数对于多项式的情况
    assert ff(2*x**2 - 5*x, 2) == (2*x**2  - 5*x)*(2*x**2 - 5*x - 1)
    assert isinstance(ff(2*x**2 - 5*x, 2), Mul)
    assert ff(x**2 + 3*x, -2) == 1/((x**2 + 3*x + 1)*(x**2 + 3*x + 2))

    # 测试 ff 函数对于多项式对象的情况
    assert ff(Poly(2*x**2 - 5*x, x), 2) == Poly(4*x**4 - 28*x**3 + 59*x**2 - 35*x, x)
    assert isinstance(ff(Poly(2*x**2 - 5*x, x), 2), Poly)
    raises(ValueError, lambda: ff(Poly(2*x**2 - 5*x, x, y), 2))
    assert ff(Poly(x**2 + 3*x, x), -2) == 1/(x**4 + 12*x**3 + 49*x**2 + 78*x + 40)
    raises(ValueError, lambda: ff(Poly(x**2 + 3*x, x, y), -2))

    # 测试 ff 函数对于符号变量的情况
    assert ff(x, m).is_integer is None
    assert ff(n, k).is_integer is None
    assert ff(n, m).is_integer is True
    assert ff(n, k + pi).is_integer is False
    assert ff(n, m + pi).is_integer is False
    assert ff(pi, m).is_integer is False

    # 断言 ff 函数返回的类型是 ff
    assert isinstance(ff(x, x), ff)
    # 断言 ff 函数对于整数 n 返回阶乘 n!
    assert ff(n, n) == factorial(n)

    # 定义一个检查函数 check，用于检查不同参数和操作的结果
    def check(x, k, o, n):
        a, b = Dummy(), Dummy()
        r = lambda x, k: o(a, b).rewrite(n).subs({a:x,b:k})
        for i in range(-5,5):
          for j in range(-5,5):
              # 断言 ff 函数和重写后的结果相等
              assert o(i, j) == r(i, j), (o, n)
    # 使用不同的函数和参数调用 check 函数进行测试
    check(x, k, ff, rf)
    check(x, k, ff, gamma)
    check(n, k, ff, factorial)
    check(x, k, ff, binomial)
    check(x, y, ff, factorial)
    check(x, y, ff, binomial)

    # 断言 ff 函数重写为 rf 后的结果
    assert ff(x, k).rewrite(rf) == rf(x - k + 1, k)
    # 断言 ff 函数重写为 gamma 后的结果
    assert ff(x, k).rewrite(gamma) == Piecewise(
        (gamma(x + 1)/gamma(-k + x + 1), x >= 0),
        ((-1)**k*gamma(k - x)/gamma(-x), True))
    assert ff(5, k).rewrite(gamma) == 120/gamma(6 - k)
    # 断言 ff 函数重写为 factorial 后的结果
    assert ff(n, k).rewrite(factorial) == Piecewise(
        (factorial(n)/factorial(-k + n), n >= 0),
        ((-1)**k*factorial(k - n - 1)/factorial(-n - 1), True))
    assert ff(5, k).rewrite(factorial) == 120/factorial(5 - k)
    # 断言 ff 函数重写为 binomial 后的结果
    assert ff(x, k).rewrite(binomial) == factorial(k) * binomial(x, k)
    # 断言 ff 函数重写为 factorial 后和本身相等
    assert ff(x, y).rewrite(factorial) == ff(x, y)
    # 断言 ff 函数重写为 binomial 后和本身相等
    assert ff(x, y).rewrite(binomial) == ff(x, y)

    # 导入必要的模块和函数
    import random
    from mpmath import ff as mpmath_ff
    # 进行随机测试，比较 mpmath_ff 和 ff 函数的结果
    for i in range(100):
        x = -500 + 500 * random.random()
        k = -500 + 500 * random.random()
        a = mpmath_ff(x, k)
        b = ff(x, k)
        # 断言 mpmath_ff 和 ff 函数的结果的绝对差小于其绝对值乘以 10 的负 15 次方
        assert (abs(a - b) < abs(a) * 10**(-15))
# 定义一个测试函数，用于验证 rf 和 ff 函数在高精度下的计算正确性
def test_rf_ff_eval_hiprec():
    # 定义一个高精度的数值常量 maple
    maple = Float('6.9109401292234329956525265438452')
    # 计算 ff(18, 2/3) 的数值结果，精确到32位小数
    us = ff(18, Rational(2, 3)).evalf(32)
    # 断言计算结果的相对误差应该小于1e-31
    assert abs(us - maple)/us < 1e-31

    # 重新定义 maple 为另一个高精度的数值常量
    maple = Float('6.8261540131125511557924466355367')
    # 计算 rf(18, 2/3) 的数值结果，精确到32位小数
    us = rf(18, Rational(2, 3)).evalf(32)
    # 断言计算结果的相对误差应该小于1e-31
    assert abs(us - maple)/us < 1e-31

    # 重新定义 maple 为第三个高精度的数值常量
    maple = Float('34.007346127440197150854651814225')
    # 计算 rf(4.4, 2.2) 的数值结果，使用默认的精度
    us = rf(Float('4.4', 32), Float('2.2', 32));
    # 断言计算结果的相对误差应该小于1e-31
    assert abs(us - maple)/us < 1e-31


# 定义一个测试函数，用于验证 rf 函数在 mpmath 中的数值计算正确性
def test_rf_lambdify_mpmath():
    # 导入 lambdify 函数并使用 mpmath 作为后端
    from sympy.utilities.lambdify import lambdify
    # 定义符号变量 x 和 y
    x, y = symbols('x,y')
    # 将 rf(x, y) 转换为 mpmath 中的数值函数 f
    f = lambdify((x,y), rf(x, y), 'mpmath')
    # 定义一个高精度的数值常量 maple
    maple = Float('34.007346127440197')
    # 计算 f(4.4, 2.2) 的数值结果
    us = f(4.4, 2.2)
    # 断言计算结果的相对误差应该小于1e-15
    assert abs(us - maple)/us < 1e-15


# 定义一个测试函数，用于验证阶乘函数 factorial 的各种性质
def test_factorial():
    # 定义符号变量 x, n, k, r, s, t, u
    x = Symbol('x')
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True, nonnegative=True)
    r = Symbol('r', integer=False)
    s = Symbol('s', integer=False, negative=True)
    t = Symbol('t', nonnegative=True)
    u = Symbol('u', noninteger=True)

    # 断言阶乘的特定值
    assert factorial(-2) is zoo
    assert factorial(0) == 1
    assert factorial(7) == 5040
    assert factorial(19) == 121645100408832000
    assert factorial(31) == 8222838654177922817725562880000000
    # 断言阶乘函数的函数属性
    assert factorial(n).func == factorial
    assert factorial(2*n).func == factorial

    # 断言阶乘函数的整数性质
    assert factorial(x).is_integer is None
    assert factorial(n).is_integer is None
    assert factorial(k).is_integer
    assert factorial(r).is_integer is None

    # 断言阶乘函数的正数性质
    assert factorial(n).is_positive is None
    assert factorial(k).is_positive

    # 断言阶乘函数的实数性质
    assert factorial(x).is_real is None
    assert factorial(n).is_real is None
    assert factorial(k).is_real is True
    assert factorial(r).is_real is None
    assert factorial(s).is_real is True
    assert factorial(t).is_real is True
    assert factorial(u).is_real is True

    # 断言阶乘函数的合数性质
    assert factorial(x).is_composite is None
    assert factorial(n).is_composite is None
    assert factorial(k).is_composite is None
    assert factorial(k + 3).is_composite is True
    assert factorial(r).is_composite is None
    assert factorial(s).is_composite is None
    assert factorial(t).is_composite is None
    assert factorial(u).is_composite is None

    # 断言阶乘函数的正无穷性质
    assert factorial(oo) is oo


# 定义一个测试函数，用于验证阶乘函数在模运算中的性质
def test_factorial_Mod():
    # 定义素数符号 pr
    pr = Symbol('pr', prime=True)
    # 定义两个素数模数 p, q 和两个复合数模数 r, s
    p, q = 10**9 + 9, 10**9 + 33 # prime modulo
    r, s = 10**7 + 5, 33333333 # composite modulo
    # 断言阶乘模素数的性质
    assert Mod(factorial(pr - 1), pr) == pr - 1
    assert Mod(factorial(pr - 1), -pr) == -1
    # 断言阶乘模复合数的性质
    assert Mod(factorial(r - 1, evaluate=False), r) == 0
    assert Mod(factorial(s - 1, evaluate=False), s) == 0
    assert Mod(factorial(p - 1, evaluate=False), p) == p - 1
    assert Mod(factorial(q - 1, evaluate=False), q) == q - 1
    assert Mod(factorial(p - 50, evaluate=False), p) == 854928834
    assert Mod(factorial(q - 1800, evaluate=False), q) == 905504050
    assert Mod(factorial(153, evaluate=False), r) == Mod(factorial(153), r)
    assert Mod(factorial(255, evaluate=False), s) == Mod(factorial(255), s)
    # 断言：对阶乘函数的结果取模后应等于零
    assert Mod(factorial(4, evaluate=False), 3) == S.Zero
    # 断言：对阶乘函数的结果取模后应等于零
    assert Mod(factorial(5, evaluate=False), 6) == S.Zero
# 定义一个测试函数，用于测试阶乘的不同性质
def test_factorial_diff():
    # 定义符号变量 n，限定为整数类型
    n = Symbol('n', integer=True)

    # 断言阶乘函数的导数与 gamma 函数和 polygamma 函数的关系
    assert factorial(n).diff(n) == \
        gamma(1 + n)*polygamma(0, 1 + n)
    
    # 断言对 n 的平方阶乘的导数
    assert factorial(n**2).diff(n) == \
        2*n*gamma(1 + n**2)*polygamma(0, 1 + n**2)
    
    # 断言对阶乘函数进行二阶导数时会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: factorial(n**2).fdiff(2))


# 定义测试阶乘函数的级数展开功能
def test_factorial_series():
    # 定义符号变量 n，限定为整数类型
    n = Symbol('n', integer=True)

    # 断言阶乘函数在 n = 0 附近的级数展开结果
    assert factorial(n).series(n, 0, 3) == \
        1 - n*EulerGamma + n**2*(EulerGamma**2/2 + pi**2/12) + O(n**3)


# 定义测试阶乘函数的重写功能
def test_factorial_rewrite():
    # 定义符号变量 n，限定为整数类型
    n = Symbol('n', integer=True)
    # 定义符号变量 k，限定为整数类型且非负
    k = Symbol('k', integer=True, nonnegative=True)

    # 断言阶乘函数通过 gamma 函数进行重写后的结果
    assert factorial(n).rewrite(gamma) == gamma(n + 1)
    
    # 创建一个虚拟符号变量 _i
    _i = Dummy('i')
    # 断言阶乘函数通过 Product 函数进行重写的结果
    assert factorial(k).rewrite(Product).dummy_eq(Product(_i, (_i, 1, k)))
    
    # 断言阶乘函数通过 Product 函数重写的结果与其自身相等
    assert factorial(n).rewrite(Product) == factorial(n)


# 定义测试阶乘的其他性质
def test_factorial2():
    # 定义符号变量 n，限定为整数类型
    n = Symbol('n', integer=True)

    # 断言对于一些特定的输入，阶乘函数的计算结果
    assert factorial2(-1) == 1
    assert factorial2(0) == 1
    assert factorial2(7) == 105
    assert factorial2(8) == 384

    # 以下为详尽的测试条件
    tt = Symbol('tt', integer=True, nonnegative=True)
    tte = Symbol('tte', even=True, nonnegative=True)
    tpe = Symbol('tpe', even=True, positive=True)
    tto = Symbol('tto', odd=True, nonnegative=True)
    tf = Symbol('tf', integer=True, nonnegative=False)
    tfe = Symbol('tfe', even=True, nonnegative=False)
    tfo = Symbol('tfo', odd=True, nonnegative=False)
    ft = Symbol('ft', integer=False, nonnegative=True)
    ff = Symbol('ff', integer=False, nonnegative=False)
    fn = Symbol('fn', integer=False)
    nt = Symbol('nt', nonnegative=True)
    nf = Symbol('nf', nonnegative=False)
    nn = Symbol('nn')
    z = Symbol('z', zero=True)
    #Solves and Fixes Issue #10388 - This is the updated test for the same solved issue
    # 断言对于特定的输入会引发 ValueError 异常
    raises(ValueError, lambda: factorial2(oo))
    raises(ValueError, lambda: factorial2(Rational(5, 2)))
    raises(ValueError, lambda: factorial2(-4))
    # 断言阶乘函数的返回类型为整数
    assert factorial2(n).is_integer is None
    # 断言对整数类型符号变量的阶乘函数调用返回整数
    assert factorial2(tt - 1).is_integer
    assert factorial2(tte - 1).is_integer
    assert factorial2(tpe - 3).is_integer
    assert factorial2(tto - 4).is_integer
    assert factorial2(tto - 2).is_integer
    # 断言对于非整数类型符号变量的阶乘函数调用返回 None
    assert factorial2(tf).is_integer is None
    assert factorial2(tfe).is_integer is None
    assert factorial2(tfo).is_integer is None
    assert factorial2(ft).is_integer is None
    assert factorial2(ff).is_integer is None
    assert factorial2(fn).is_integer is None
    assert factorial2(nt).is_integer is None
    assert factorial2(nf).is_integer is None
    assert factorial2(nn).is_integer is None

    # 断言阶乘函数的返回类型为正数
    assert factorial2(n).is_positive is None
    assert factorial2(tt - 1).is_positive is True
    assert factorial2(tte - 1).is_positive is True
    assert factorial2(tpe - 3).is_positive is True
    assert factorial2(tpe - 1).is_positive is True
    assert factorial2(tto - 2).is_positive is True
    assert factorial2(tto - 1).is_positive is True
    # 断言对于非整数类型符号变量的阶乘函数调用返回 None
    assert factorial2(tf).is_positive is None
    # 检查阶乘结果是否为正数，预期结果为 None
    assert factorial2(tfe).is_positive is None
    # 检查阶乘结果是否为正数，预期结果为 None
    assert factorial2(tfo).is_positive is None
    # 检查阶乘结果是否为正数，预期结果为 None
    assert factorial2(ft).is_positive is None
    # 检查阶乘结果是否为正数，预期结果为 None
    assert factorial2(ff).is_positive is None
    # 检查阶乘结果是否为正数，预期结果为 None
    assert factorial2(fn).is_positive is None
    # 检查阶乘结果是否为正数，预期结果为 None
    assert factorial2(nt).is_positive is None
    # 检查阶乘结果是否为正数，预期结果为 None
    assert factorial2(nf).is_positive is None
    # 检查阶乘结果是否为正数，预期结果为 None
    assert factorial2(nn).is_positive is None
    
    # 检查阶乘结果是否为偶数，预期结果为 None
    assert factorial2(tt).is_even is None
    # 检查阶乘结果是否为奇数，预期结果为 None
    assert factorial2(tt).is_odd is None
    # 检查阶乘结果是否为偶数，预期结果为 None
    assert factorial2(tte).is_even is None
    # 检查阶乘结果是否为奇数，预期结果为 None
    assert factorial2(tte).is_odd is None
    # 检查阶乘结果加2后是否为偶数，预期结果为 True
    assert factorial2(tte + 2).is_even is True
    # 检查阶乘结果是否为偶数，预期结果为 True
    assert factorial2(tpe).is_even is True
    # 检查阶乘结果是否为奇数，预期结果为 False
    assert factorial2(tpe).is_odd is False
    # 检查阶乘结果是否为奇数，预期结果为 True
    assert factorial2(tto).is_odd is True
    # 检查阶乘结果是否为偶数，预期结果为 None
    assert factorial2(tf).is_even is None
    # 检查阶乘结果是否为奇数，预期结果为 None
    assert factorial2(tf).is_odd is None
    # 检查阶乘结果是否为偶数，预期结果为 None
    assert factorial2(tfe).is_even is None
    # 检查阶乘结果是否为奇数，预期结果为 None
    assert factorial2(tfe).is_odd is None
    # 检查阶乘结果是否为偶数，预期结果为 False
    assert factorial2(tfo).is_even is False
    # 检查阶乘结果是否为奇数，预期结果为 None
    assert factorial2(tfo).is_odd is None
    # 检查阶乘结果是否为偶数，预期结果为 False
    assert factorial2(z).is_even is False
    # 检查阶乘结果是否为奇数，预期结果为 True
    assert factorial2(z).is_odd is True
def test_factorial2_rewrite():
    # 定义符号变量 n，表示为整数
    n = Symbol('n', integer=True)
    # 断言计算 factorial2(n) 重写为 gamma 函数的表达式
    assert factorial2(n).rewrite(gamma) == \
        2**(n/2)*Piecewise((1, Eq(Mod(n, 2), 0)), (sqrt(2)/sqrt(pi), Eq(Mod(n, 2), 1)))*gamma(n/2 + 1)
    # 断言计算 factorial2(2*n) 重写为 gamma 函数的表达式
    assert factorial2(2*n).rewrite(gamma) == 2**n*gamma(n + 1)
    # 断言计算 factorial2(2*n + 1) 重写为 gamma 函数的表达式
    assert factorial2(2*n + 1).rewrite(gamma) == \
        sqrt(2)*2**(n + S.Half)*gamma(n + Rational(3, 2))/sqrt(pi)


def test_binomial():
    # 定义符号变量
    x = Symbol('x')
    n = Symbol('n', integer=True)
    nz = Symbol('nz', integer=True, nonzero=True)
    k = Symbol('k', integer=True)
    kp = Symbol('kp', integer=True, positive=True)
    kn = Symbol('kn', integer=True, negative=True)
    u = Symbol('u', negative=True)
    v = Symbol('v', nonnegative=True)
    p = Symbol('p', positive=True)
    z = Symbol('z', zero=True)
    nt = Symbol('nt', integer=False)
    kt = Symbol('kt', integer=False)
    a = Symbol('a', integer=True, nonnegative=True)
    b = Symbol('b', integer=True, nonnegative=True)

    # 断言计算二项式系数 binomial(n, k) 的值
    assert binomial(0, 0) == 1
    assert binomial(1, 1) == 1
    assert binomial(10, 10) == 1
    assert binomial(n, z) == 1
    assert binomial(1, 2) == 0
    assert binomial(-1, 2) == 1
    assert binomial(1, -1) == 0
    assert binomial(-1, 1) == -1
    assert binomial(-1, -1) == 0
    assert binomial(S.Half, S.Half) == 1
    assert binomial(-10, 1) == -10
    assert binomial(-10, 7) == -11440
    assert binomial(n, -1) == 0  # 对于所有整数（负数、零、正数）都成立
    assert binomial(kp, -1) == 0
    assert binomial(nz, 0) == 1
    # 展开 binomial(n, 1) 并断言其值
    assert expand_func(binomial(n, 1)) == n
    # 展开 binomial(n, 2) 并断言其值
    assert expand_func(binomial(n, 2)) == n*(n - 1)/2
    # 展开 binomial(n, n - 2) 并断言其值
    assert expand_func(binomial(n, n - 2)) == n*(n - 1)/2
    # 展开 binomial(n, n - 1) 并断言其值
    assert expand_func(binomial(n, n - 1)) == n
    # 断言 binomial(n, 3) 的类型是 binomial
    assert binomial(n, 3).func == binomial
    # 展开 binomial(n, 3) 并断言其值
    assert binomial(n, 3).expand(func=True) == n**3/6 - n**2/2 + n/3
    # 展开 binomial(n, 3) 并断言其值
    assert expand_func(binomial(n, 3)) == n*(n - 2)*(n - 1)/6
    # 断言 binomial(n, n) 的类型是 binomial
    assert binomial(n, n).func == binomial
    # 断言 binomial(n, n + 1) 的类型是 binomial
    assert binomial(n, n + 1).func == binomial
    # 断言 binomial(kp, kp + 1) 的值为 0
    assert binomial(kp, kp + 1) == 0
    # 断言 binomial(kn, kn) 的值为 0
    assert binomial(kn, kn) == 0  # issue #14529
    # 断言 binomial(n, u) 的类型是 binomial
    assert binomial(n, u).func == binomial
    # 断言 binomial(kp, u) 的类型是 binomial
    assert binomial(kp, u).func == binomial
    # 断言 binomial(n, p) 的类型是 binomial
    assert binomial(n, p).func == binomial
    # 断言 binomial(n, k) 的类型是 binomial
    assert binomial(n, k).func == binomial
    # 断言 binomial(n, n + p) 的类型是 binomial
    assert binomial(n, n + p).func == binomial
    # 断言 binomial(kp, kp + p) 的类型是 binomial

    assert expand_func(binomial(n, n - 3)) == n*(n - 2)*(n - 1)/6

    # 断言 binomial(n, k) 的 is_integer 属性为 True
    assert binomial(n, k).is_integer
    # 断言 binomial(nt, k) 的 is_integer 属性为 None
    assert binomial(nt, k).is_integer is None
    # 断言 binomial(x, nt) 的 is_integer 属性为 False
    assert binomial(x, nt).is_integer is False

    # 断言计算 gamma(25) 与 6 的二项式系数
    assert binomial(gamma(25), 6) == 79232165267303928292058750056084441948572511312165380965440075720159859792344339983120618959044048198214221915637090855535036339620413440000
    # 断言计算 1324 与 47 的二项式系数
    assert binomial(1324, 47) == 906266255662694632984994480774946083064699457235920708992926525848438478406790323869952
    # 检查二项式系数计算是否正确
    assert binomial(1735, 43) == 190910140420204130794758005450919715396159959034348676124678207874195064798202216379800
    assert binomial(2512, 53) == 213894469313832631145798303740098720367984955243020898718979538096223399813295457822575338958939834177325304000
    assert binomial(3383, 52) == 27922807788818096863529701501764372757272890613101645521813434902890007725667814813832027795881839396839287659777235
    assert binomial(4321, 51) == 124595639629264868916081001263541480185227731958274383287107643816863897851139048158022599533438936036467601690983780576

    # 检查二项式系数是否非负
    assert binomial(a, b).is_nonnegative is True
    assert binomial(-1, 2, evaluate=False).is_nonnegative is True
    assert binomial(10, 5, evaluate=False).is_nonnegative is True
    assert binomial(10, -3, evaluate=False).is_nonnegative is True
    assert binomial(-10, -3, evaluate=False).is_nonnegative is True
    assert binomial(-10, 2, evaluate=False).is_nonnegative is True
    assert binomial(-10, 1, evaluate=False).is_nonnegative is False
    assert binomial(-10, 7, evaluate=False).is_nonnegative is False

    # issue #14625：验证特定情况下的二项式系数
    for _ in (pi, -pi, nt, v, a):
        assert binomial(_, _) == 1
        assert binomial(_, _ - 1) == _

    # 验证返回值类型是否为 binomial 类型
    assert isinstance(binomial(u, u), binomial)
    assert isinstance(binomial(u, u - 1), binomial)
    assert isinstance(binomial(x, x), binomial)
    assert isinstance(binomial(x, x - 1), binomial)

    # issue #18802：验证二项式系数的展开结果
    assert expand_func(binomial(x + 1, x)) == x + 1
    assert expand_func(binomial(x, x - 1)) == x
    assert expand_func(binomial(x + 1, x - 1)) == x*(x + 1)/2
    assert expand_func(binomial(x**2 + 1, x**2)) == x**2 + 1

    # issue #13980 和 #13981：验证负数下的二项式系数为 0
    assert binomial(-7, -5) == 0
    assert binomial(-23, -12) == 0
    assert binomial(Rational(13, 2), -10) == 0
    assert binomial(-49, -51) == 0

    # 验证特定复数参数下的二项式系数计算
    assert binomial(19, Rational(-7, 2)) == S(-68719476736)/(911337863661225*pi)
    assert binomial(0, Rational(3, 2)) == S(-2)/(3*pi)
    assert binomial(-3, Rational(-7, 2)) is zoo
    assert binomial(kn, kt) is zoo

    # 验证 binomial 函数的 func 属性是否正确
    assert binomial(nt, kt).func == binomial

    # 验证特定有理数下的二项式系数计算
    assert binomial(nt, Rational(15, 6)) == 8*gamma(nt + 1)/(15*sqrt(pi)*gamma(nt - Rational(3, 2)))
    assert binomial(Rational(20, 3), Rational(-10, 8)) == gamma(Rational(23, 3))/(gamma(Rational(-1, 4))*gamma(Rational(107, 12)))
    assert binomial(Rational(19, 2), Rational(-7, 2)) == Rational(-1615, 8388608)
    assert binomial(Rational(-13, 5), Rational(-7, 8)) == gamma(Rational(-8, 5))/(gamma(Rational(-29, 40))*gamma(Rational(1, 8)))
    assert binomial(Rational(-19, 8), Rational(-13, 5)) == gamma(Rational(-11, 8))/(gamma(Rational(-8, 5))*gamma(Rational(49, 40)))

    # 验证复数参数下的二项式系数计算
    assert binomial(I, Rational(-89, 8)) == gamma(1 + I)/(gamma(Rational(-81, 8))*gamma(Rational(97, 8) + I))
    assert binomial(I, 2*I) == gamma(1 + I)/(gamma(1 - I)*gamma(1 + 2*I))
    assert binomial(-7, I) is zoo
    # 断言，验证二项式函数的计算结果是否正确
    assert binomial(Rational(-7, 6), I) == gamma(Rational(-1, 6))/(gamma(Rational(-1, 6) - I)*gamma(1 + I))
    
    # 断言，验证二项式函数的计算结果是否正确
    assert binomial((1+2*I), (1+3*I)) == gamma(2 + 2*I)/(gamma(1 - I)*gamma(2 + 3*I))
    
    # 断言，验证二项式函数的计算结果是否正确
    assert binomial(I, 5) == Rational(1, 3) - I/S(12)
    
    # 断言，验证二项式函数的计算结果是否正确
    assert binomial((2*I + 3), 7) == -13*I/S(63)
    
    # 断言，验证二项式函数返回的结果是否是 binomial 类型的实例
    assert isinstance(binomial(I, n), binomial)
    
    # 断言，验证展开二项式函数后的计算结果是否正确
    assert expand_func(binomial(3, 2, evaluate=False)) == 3
    
    # 断言，验证展开二项式函数后的计算结果是否正确
    assert expand_func(binomial(n, 0, evaluate=False)) == 1
    
    # 断言，验证展开二项式函数后的计算结果是否正确
    assert expand_func(binomial(n, -2, evaluate=False)) == 0
    
    # 断言，验证展开二项式函数后的计算结果是否正确
    assert expand_func(binomial(n, k)) == binomial(n, k)
# 定义测试函数 test_binomial_Mod，用于测试二项式系数的模运算
def test_binomial_Mod():
    # 设置两个大素数作为模数
    p, q = 10**5 + 3, 10**9 + 33
    # 设置两个合成数作为模数
    r = 10**7 + 5

    # Lucas 定理的测试
    assert Mod(binomial(156675, 4433, evaluate=False), p) == Mod(binomial(156675, 4433), p)

    # 阶乘模的测试
    assert Mod(binomial(1234, 432, evaluate=False), q) == Mod(binomial(1234, 432), q)

    # 二项式分解的测试
    assert Mod(binomial(253, 113, evaluate=False), r) == Mod(binomial(253, 113), r)

    # 使用 Granville 的 Lucas 定理推广的测试
    assert Mod(binomial(10**18, 10**12, evaluate=False), p*p) == 3744312326


# 用 @slow 装饰器标记的测试函数，执行速度较慢
@slow
def test_binomial_Mod_slow():
    # 设置两个大素数作为模数
    p, q = 10**5 + 3, 10**9 + 33
    # 设置两个合成数作为模数
    r, s = 10**7 + 5, 33333333

    # 符号变量声明
    n, k, m = symbols('n k m')

    # 模运算的测试
    assert (binomial(n, k) % q).subs({n: s, k: p}) == Mod(binomial(s, p), q)
    assert (binomial(n, k) % m).subs({n: 8, k: 5, m: 13}) == 4
    assert (binomial(9, k) % 7).subs(k, 2) == 1

    # Lucas 定理的测试
    assert Mod(binomial(123456, 43253, evaluate=False), p) == Mod(binomial(123456, 43253), p)
    assert Mod(binomial(-178911, 237, evaluate=False), p) == Mod(-binomial(178911 + 237 - 1, 237), p)
    assert Mod(binomial(-178911, 238, evaluate=False), p) == Mod(binomial(178911 + 238 - 1, 238), p)

    # 阶乘模的测试
    assert Mod(binomial(9734, 451, evaluate=False), q) == Mod(binomial(9734, 451), q)
    assert Mod(binomial(-10733, 4459, evaluate=False), q) == Mod(binomial(-10733, 4459), q)
    assert Mod(binomial(-15733, 4458, evaluate=False), q) == Mod(binomial(-15733, 4458), q)
    assert Mod(binomial(23, -38, evaluate=False), q) is S.Zero
    assert Mod(binomial(23, 38, evaluate=False), q) is S.Zero

    # 二项式分解的测试
    assert Mod(binomial(753, 119, evaluate=False), r) == Mod(binomial(753, 119), r)
    assert Mod(binomial(3781, 948, evaluate=False), s) == Mod(binomial(3781, 948), s)
    assert Mod(binomial(25773, 1793, evaluate=False), s) == Mod(binomial(25773, 1793), s)
    assert Mod(binomial(-753, 118, evaluate=False), r) == Mod(binomial(-753, 118), r)
    assert Mod(binomial(-25773, 1793, evaluate=False), s) == Mod(binomial(-25773, 1793), s)


# 定义测试函数 test_binomial_diff，用于测试二项式系数的微分
def test_binomial_diff():
    # 声明整数符号变量
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True)

    # 对 n 的微分测试
    assert binomial(n, k).diff(n) == \
        (-polygamma(0, 1 + n - k) + polygamma(0, 1 + n))*binomial(n, k)
    assert binomial(n**2, k**3).diff(n) == \
        2*n*(-polygamma(0, 1 + n**2 - k**3) + polygamma(0, 1 + n**2))*binomial(n**2, k**3)

    # 对 k 的微分测试
    assert binomial(n, k).diff(k) == \
        (-polygamma(0, 1 + k) + polygamma(0, 1 + n - k))*binomial(n, k)
    assert binomial(n**2, k**3).diff(k) == \
        3*k**2*(-polygamma(0, 1 + k**3) + polygamma(0, 1 + n**2 - k**3))*binomial(n**2, k**3)
    raises(ArgumentIndexError, lambda: binomial(n, k).fdiff(3))


# 定义测试函数 test_binomial_rewrite，用于测试二项式系数的重写
def test_binomial_rewrite():
    # 声明整数符号变量和实数符号变量
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True)
    x = Symbol('x')
    # 使用 assert 语句来检查 binomial 函数的重写结果是否符合预期
    assert binomial(n, k).rewrite(factorial) == factorial(n)/(factorial(k)*factorial(n - k))
    # 使用 assert 语句来检查 binomial 函数的重写结果是否符合预期，使用 gamma 函数
    assert binomial(n, k).rewrite(gamma) == gamma(n + 1)/(gamma(k + 1)*gamma(n - k + 1))
    # 使用 assert 语句来检查 binomial 函数的重写结果是否符合预期，使用 ff 函数和 factorial 函数
    assert binomial(n, k).rewrite(ff) == ff(n, k) / factorial(k)
    # 使用 assert 语句来检查 binomial 函数的重写结果是否符合预期，n 和 x 之间的二项式系数
    assert binomial(n, x).rewrite(ff) == binomial(n, x)
@XFAIL
# 定义一个被标记为预期失败的测试函数，用来测试阶乘简化失败的情况
def test_factorial_simplify_fail():
    # 对表达式 simplify(factorial(x + 1).diff(x) - ((x + 1)*factorial(x)).diff(x))) == 0 进行断言
    from sympy.abc import x
    assert simplify(x*polygamma(0, x + 1) - x*polygamma(0, x + 2) +
                    polygamma(0, x + 1) - polygamma(0, x + 2) + 1) == 0


# 定义一个测试函数来测试子阶乘函数 subfactorial 的不同用例
def test_subfactorial():
    # 断言所有的 subfactorial(i) 对应于预期的结果 ans，其中 i 从 0 开始递增
    assert all(subfactorial(i) == ans for i, ans in enumerate(
        [1, 0, 1, 2, 9, 44, 265, 1854, 14833, 133496]))
    # 断言 subfactorial(oo) 等于 oo（正无穷）
    assert subfactorial(oo) is oo
    # 断言 subfactorial(nan) 等于 nan（非数字）
    assert subfactorial(nan) is nan
    # 断言 subfactorial(23) 的值等于特定的长整型数值
    assert subfactorial(23) == 9510425471055777937262
    # 断言 unchanged(subfactorial, 2.2) 的结果为真

    # 创建符号 x
    x = Symbol('x')
    # 断言 subfactorial(x).rewrite(uppergamma) 等于 uppergamma(x + 1, -1)/S.Exp1

    # 创建整数和非负整数符号
    tt = Symbol('tt', integer=True, nonnegative=True)
    tf = Symbol('tf', integer=True, nonnegative=False)
    tn = Symbol('tf', integer=True)
    ft = Symbol('ft', integer=False, nonnegative=True)
    ff = Symbol('ff', integer=False, nonnegative=False)
    fn = Symbol('ff', integer=False)
    nt = Symbol('nt', nonnegative=True)
    nf = Symbol('nf', nonnegative=False)
    nn = Symbol('nf')
    te = Symbol('te', even=True, nonnegative=True)
    to = Symbol('to', odd=True, nonnegative=True)

    # 一系列断言来测试 subfactorial 函数返回的属性
    assert subfactorial(tt).is_integer
    assert subfactorial(tf).is_integer is None
    assert subfactorial(tn).is_integer is None
    assert subfactorial(ft).is_integer is None
    assert subfactorial(ff).is_integer is None
    assert subfactorial(fn).is_integer is None
    assert subfactorial(nt).is_integer is None
    assert subfactorial(nf).is_integer is None
    assert subfactorial(nn).is_integer is None
    assert subfactorial(tt).is_nonnegative
    assert subfactorial(tf).is_nonnegative is None
    assert subfactorial(tn).is_nonnegative is None
    assert subfactorial(ft).is_nonnegative is None
    assert subfactorial(ff).is_nonnegative is None
    assert subfactorial(fn).is_nonnegative is None
    assert subfactorial(nt).is_nonnegative is None
    assert subfactorial(nf).is_nonnegative is None
    assert subfactorial(nn).is_nonnegative is None
    assert subfactorial(tt).is_even is None
    assert subfactorial(tt).is_odd is None
    assert subfactorial(te).is_odd is True
    assert subfactorial(to).is_even is True
```