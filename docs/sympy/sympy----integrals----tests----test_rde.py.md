# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_rde.py`

```
"""Most of these tests come from the examples in Bronstein's book."""
# 导入必要的库和模块
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.integrals.risch import (DifferentialExtension,
    NonElementaryIntegralException)
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
    normal_denom, special_denom, bound_degree, spde, solve_poly_rde,
    no_cancel_equal, cancel_primitive, cancel_exp, rischDE)
# 导入用于测试的断言函数
from sympy.testing.pytest import raises
# 导入符号变量
from sympy.abc import x, t, z, n

# 定义符号变量
t0, t1, t2, k = symbols('t:3 k')

# 测试函数：计算多项式相对于另一个多项式的阶
def test_order_at():
    # 定义多项式
    a = Poly(t**4, t)
    b = Poly((t**2 + 1)**3*t, t)
    c = Poly((t**2 + 1)**6*t, t)
    d = Poly((t**2 + 1)**10*t**10, t)
    e = Poly((t**2 + 1)**100*t**37, t)
    p1 = Poly(t, t)
    p2 = Poly(1 + t**2, t)
    # 断言各种情况下的阶数
    assert order_at(a, p1, t) == 4
    assert order_at(b, p1, t) == 1
    assert order_at(c, p1, t) == 1
    assert order_at(d, p1, t) == 10
    assert order_at(e, p1, t) == 37
    assert order_at(a, p2, t) == 0
    assert order_at(b, p2, t) == 3
    assert order_at(c, p2, t) == 6
    assert order_at(d, p1, t) == 10  # 这里应该是 p1 而不是 p2
    assert order_at(e, p2, t) == 100
    assert order_at(Poly(0, t), Poly(t, t), t) is oo
    assert order_at_oo(Poly(t**2 - 1, t), Poly(t + 1), t) == \
        order_at_oo(Poly(t - 1, t), Poly(1, t), t) == -1
    assert order_at_oo(Poly(0, t), Poly(1, t), t) is oo

# 测试函数：弱标准化函数
def test_weak_normalizer():
    # 定义多项式和微分扩展对象
    a = Poly((1 + x)*t**5 + 4*t**4 + (-1 - 3*x)*t**3 - 4*t**2 + (-2 + 2*x)*t, t)
    d = Poly(t**4 - 3*t**2 + 2, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    # 调用弱标准化函数
    r = weak_normalizer(a, d, DE, z)
    # 断言结果
    assert r == (Poly(t**5 - t**4 - 4*t**3 + 4*t**2 + 4*t - 4, t, domain='ZZ[x]'),
        (Poly((1 + x)*t**2 + x*t, t, domain='ZZ[x]'),
         Poly(t + 1, t, domain='ZZ[x]')))
    assert weak_normalizer(r[1][0], r[1][1], DE) == (Poly(1, t), r[1])
    r = weak_normalizer(Poly(1 + t**2), Poly(t**2 - 1, t), DE, z)
    assert r == (Poly(t**4 - 2*t**2 + 1, t), (Poly(-3*t**2 + 1, t), Poly(t**2 - 1, t)))
    assert weak_normalizer(r[1][0], r[1][1], DE, z) == (Poly(1, t), r[1])
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2)]})
    r = weak_normalizer(Poly(1 + t**2), Poly(t, t), DE, z)
    assert r == (Poly(t, t), (Poly(0, t), Poly(1, t)))
    assert weak_normalizer(r[1][0], r[1][1], DE, z) == (Poly(1, t), r[1])

# 测试函数：普通分母函数
def test_normal_denom():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    raises(NonElementaryIntegralException, lambda: normal_denom(Poly(1, x), Poly(1, x),
    Poly(1, x), Poly(x, x), DE))
    fa, fd = Poly(t**2 + 1, t), Poly(1, t)
    ga, gd = Poly(1, t), Poly(t**2, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    # 断言异常情况
    assert normal_denom(fa, fd, ga, gd, DE) == \
        (Poly(t, t), (Poly(t**3 - t**2 + t - 1, t), Poly(1, t)), (Poly(1, t),
        Poly(1, t)), Poly(t, t))

# 测试函数：特殊分母函数
def test_special_denom():
    # TODO: add more tests here
    # 创建一个差分扩展对象 DE，使用指定的扩展字典
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    # 断言特殊分母函数 special_denom 的返回结果与预期结果相等
    assert special_denom(
        Poly(1, t),         # 参数1: 多项式 Poly(1, t)
        Poly(t**2, t),      # 参数2: 多项式 Poly(t**2, t)
        Poly(1, t),         # 参数3: 多项式 Poly(1, t)
        Poly(t**2 - 1, t),  # 参数4: 多项式 Poly(t**2 - 1, t)
        Poly(t, t),         # 参数5: 多项式 Poly(t, t)
        DE                  # 参数6: 差分扩展对象 DE
    ) == (
        Poly(1, t),         # 预期返回结果1: 多项式 Poly(1, t)
        Poly(t**2 - 1, t),  # 预期返回结果2: 多项式 Poly(t**2 - 1, t)
        Poly(t**2 - 1, t),  # 预期返回结果3: 多项式 Poly(t**2 - 1, t)
        Poly(t, t)          # 预期返回结果4: 多项式 Poly(t, t)
    )
    # issue 3940
    # 创建一个包含三个多项式的微分扩展对象 DE，用于测试特殊分母函数 special_denom
    # 注意，这不是一个很好的测试，因为分母只是 1，但至少可以测试指数取消的情况
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-2*x*t0, t0),
        Poly(I*k*t1, t1)]})
    # 减少 DE 的级别，可能是降低微分方程的复杂度或减少参数
    DE.decrement_level()
    # 断言特殊分母函数对给定参数的返回值
    assert special_denom(Poly(1, t0), Poly(I*k, t0), Poly(1, t0), Poly(t0, t0),
    Poly(1, t0), DE) == \
        (Poly(1, t0, domain='ZZ'), Poly(I*k, t0, domain='ZZ_I[k,x]'),
                Poly(t0, t0, domain='ZZ'), Poly(1, t0, domain='ZZ'))

    # 断言特殊分母函数在 'tan' 情况下的返回值
    assert special_denom(Poly(1, t), Poly(t**2, t), Poly(1, t), Poly(t**2 - 1, t),
    Poly(t, t), DE, case='tan') == \
           (Poly(1, t, t0, domain='ZZ'), Poly(t**2, t0, t, domain='ZZ[x]'),
            Poly(t, t, t0, domain='ZZ'), Poly(1, t0, domain='ZZ'))

    # 断言特殊分母函数在 'unrecognized_case' 情况下会引发 ValueError 异常
    raises(ValueError, lambda: special_denom(Poly(1, t), Poly(t**2, t), Poly(1, t), Poly(t**2 - 1, t),
    Poly(t, t), DE, case='unrecognized_case'))


def test_bound_degree_fail():
    # Primitive
    # 创建一个包含三个多项式的微分扩展对象 DE，用于测试 bound_degree 函数
    DE = DifferentialExtension(extension={'D': [Poly(1, x),
        Poly(t0/x**2, t0), Poly(1/x, t)]})
    # 断言 bound_degree 函数对给定参数的返回值
    assert bound_degree(Poly(t**2, t), Poly(-(1/x**2*t**2 + 1/x), t),
        Poly((2*x - 1)*t**4 + (t0 + x)/x*t**3 - (t0 + 4*x**2)/2*x*t**2 + x*t,
        t), DE) == 3


def test_bound_degree():
    # Base
    # 创建一个只包含一个多项式的微分扩展对象 DE，用于测试 bound_degree 函数
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    # 断言 bound_degree 函数对给定参数的返回值
    assert bound_degree(Poly(1, x), Poly(-2*x, x), Poly(1, x), DE) == 0

    # Primitive (see above test_bound_degree_fail)
    # TODO: Add test for when the degree bound becomes larger after limited_integrate
    # TODO: Add test for db == da - 1 case

    # Exp
    # TODO: Add tests
    # TODO: Add test for when the degree becomes larger after parametric_log_deriv()

    # Nonlinear
    # 创建一个包含两个多项式的微分扩展对象 DE，用于测试 bound_degree 函数
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    # 断言 bound_degree 函数对给定参数的返回值
    assert bound_degree(Poly(t, t), Poly((t - 1)*(t**2 + 1), t), Poly(1, t), DE) == 0


def test_spde():
    # 创建一个包含两个多项式的微分扩展对象 DE，用于测试 spde 函数
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    # 断言 spde 函数对给定参数的返回值，预期会引发 NonElementaryIntegralException 异常
    raises(NonElementaryIntegralException, lambda: spde(Poly(t, t), Poly((t - 1)*(t**2 + 1), t), Poly(1, t), 0, DE))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    # 断言 spde 函数对给定参数的返回值
    assert spde(Poly(t**2 + x*t*2 + x**2, t), Poly(t**2/x**2 + (2/x - 1)*t, t),
        Poly(t**2/x**2 + (2/x - 1)*t, t), 0, DE) == \
        (Poly(0, t), Poly(0, t), 0, Poly(0, t), Poly(1, t, domain='ZZ(x)'))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0/x**2, t0), Poly(1/x, t)]})
    # 断言 spde 函数对给定参数的返回值
    assert spde(Poly(t**2, t), Poly(-t**2/x**2 - 1/x, t),
    Poly((2*x - 1)*t**4 + (t0 + x)/x*t**3 - (t0 + 4*x**2)/(2*x)*t**2 + x*t, t), 3, DE) == \
        (Poly(0, t), Poly(0, t), 0, Poly(0, t),
        Poly(t0*t**2/2 + x**2*t**2 - x**2*t, t, domain='ZZ(x,t0)'))
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    # 断言 spde 函数对给定参数的返回值
    assert spde(Poly(x**2 + x + 1, x), Poly(-2*x - 1, x), Poly(x**5/2 + 
    # 对第一个等式进行断言，验证 spde 函数的返回结果是否符合预期
    assert spde(Poly(3*x**4/4 + x**3 - x**2 + 1, x), 4, DE) == \
        (Poly(0, x, domain='QQ'),                      # 零多项式，x 的系数域为有理数 QQ
         Poly(x/2 - Rational(1, 4), x),                # x/2 - 1/4 的多项式形式，x 的系数域为有理数 QQ
         2,                                            # 整数 2
         Poly(x**2 + x + 1, x),                        # x^2 + x + 1 的多项式形式，x 的系数域为整数 ZZ
         Poly(x*Rational(5, 4), x))                    # x * 5/4 的多项式形式，x 的系数域为有理数 QQ
    # 对第二个等式进行断言，验证 spde 函数的返回结果是否符合预期
    assert spde(Poly(x**2 + x + 1, x),                 # x^2 + x + 1 的多项式形式，x 的系数域为整数 ZZ
                Poly(-2*x - 1, x),                     # -2*x - 1 的多项式形式，x 的系数域为整数 ZZ
                Poly(3*x**4/4 + x**3 - x**2 + 1, x),    # 3*x**4/4 + x**3 - x**2 + 1 的多项式形式，x 的系数域为整数 ZZ
                n,                                     # 变量 n
                DE) == \
        (Poly(0, x, domain='QQ'),                      # 零多项式，x 的系数域为有理数 QQ
         Poly(x/2 - Rational(1, 4), x),                # x/2 - 1/4 的多项式形式，x 的系数域为有理数 QQ
         -2 + n,                                       # -2 + n 的表达式，n 是变量
         Poly(x**2 + x + 1, x),                        # x^2 + x + 1 的多项式形式，x 的系数域为整数 ZZ
         Poly(x*Rational(5, 4), x))                    # x * 5/4 的多项式形式，x 的系数域为有理数 QQ
    # 创建一个 DifferentialExtension 对象 DE，定义了一个扩展字典，包含关于 x 和 t 的多项式
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1, t)]})
    # 断言异常 NonElementaryIntegralException 被引发，lambda 表达式调用 spde 函数
    raises(NonElementaryIntegralException, lambda: spde(Poly((t - 1)*(t**2 + 1)**2, t),  # (t - 1)*(t**2 + 1)**2 的 t 的多项式形式
                                                       Poly((t - 1)*(t**2 + 1), t),       # (t - 1)*(t**2 + 1) 的 t 的多项式形式
                                                       Poly(1, t),                       # 1 的 t 的多项式形式
                                                       0,                                # 整数 0
                                                       DE))
    # 重新赋值 DE，创建一个 DifferentialExtension 对象，定义一个扩展字典，只包含关于 x 的多项式
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    # 对第三个等式进行断言，验证 spde 函数的返回结果是否符合预期
    assert spde(Poly(x**2 - x, x),                      # x^2 - x 的多项式形式，x 的系数域为整数 ZZ
                Poly(1, x),                             # 1 的 x 的多项式形式，x 的系数域为整数 ZZ
                Poly(9*x**4 - 10*x**3 + 2*x**2, x),      # 9*x**4 - 10*x**3 + 2*x**2 的多项式形式，x 的系数域为整数 ZZ
                4,                                       # 整数 4
                DE) == \
        (Poly(0, x, domain='ZZ'),                       # 零多项式，x 的系数域为整数 ZZ
         Poly(0, x),                                   # 零多项式，x 的系数域为整数 ZZ
         0,                                            # 整数 0
         Poly(0, x),                                   # 零多项式，x 的系数域为整数 ZZ
         Poly(3*x**3 - 2*x**2, x, domain='QQ'))         # 3*x**3 - 2*x**2 的多项式形式，x 的系数域为有理数 QQ
    # 对第四个等式进行断言，验证 spde 函数的返回结果是否符合预期
    assert spde(Poly(x**2 - x, x),                      # x^2 - x 的多项式形式，x 的系数域为整数 ZZ
                Poly(x**2 - 5*x + 3, x),                # x^2 - 5*x + 3 的多项式形式，x 的系数域为整数 ZZ
                Poly(x**7 - x**6 - 2*x**4 + 3*x**3 - x**2, x),  # x^7 - x^6 - 2*x^4 + 3*x^3 - x^2 的多项式形式，x 的系数域为整数 ZZ
                5,                                       # 整数 5
                DE) == \
        (Poly(1, x, domain='QQ'),                       # 1 的 x 的多项式形式，x 的系数域为有理数 QQ
         Poly(x + 1, x, domain='QQ'),                   # x + 1 的 x 的多项式形式，x 的系数域为有理数 QQ
         1,                                            # 整数 1
         Poly(x**4 - x**3, x),                         # x^4 - x^3 的多项式形式，x 的系数域为整数 ZZ
         Poly(x**3 - x**2, x, domain='QQ'))             # x^3 - x^2 的多项式形式，x 的系数域为有理数 QQ
def test_solve_poly_rde_no_cancel():
    # 定义一个测试函数，用于测试 solve_poly_rde 函数在无需取消的情况下的行为
    # 当 b 的次数较大时
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    # 断言 solve_poly_rde 函数的返回结果与预期的多项式相等
    assert solve_poly_rde(Poly(t**2 + 1, t), Poly(t**3 + (x + 1)*t**2 + t + x + 2, t),
                          oo, DE) == Poly(t + x, t)
    
    # 当 b 的次数较小时
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert solve_poly_rde(Poly(0, x), Poly(x/2 - Rational(1, 4), x), oo, DE) == \
        Poly(x**2/4 - x/4, x)
    
    # 当 b 的次数等于 D 的次数减1时
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert solve_poly_rde(Poly(2, t), Poly(t**2 + 2*t + 3, t), 1, DE) == \
        Poly(t + 1, t, x)
    
    # 测试 no_cancel_equal 函数
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert no_cancel_equal(Poly(1 - t, t),
                           Poly(t**3 + t**2 - 2*x*t - 2*x, t), oo, DE) == \
        (Poly(t**2, t), 1, Poly((-2 - 2*x)*t - 2*x, t))


def test_solve_poly_rde_cancel():
    # 测试带有取消的 solve_poly_rde 函数
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    # 测试指数部分的取消
    assert cancel_exp(Poly(2*x, t), Poly(2*x, t), 0, DE) == \
        Poly(1, t)
    assert cancel_exp(Poly(2*x, t), Poly((1 + 2*x)*t, t), 1, DE) == \
        Poly(t, t)
    
    # TODO: 添加更多的指数测试，包括需要使用 is_deriv_in_field() 的测试

    # 测试原始部分的取消
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    
    # 如果 DecrementLevel 上下文管理器正常工作，则此处不应引发任何问题
    raises(NonElementaryIntegralException, lambda: cancel_primitive(Poly(1, t), Poly(t, t), oo, DE))
    
    assert cancel_primitive(Poly(1, t), Poly(t + 1/x, t), 2, DE) == \
        Poly(t, t)
    assert cancel_primitive(Poly(4*x, t), Poly(4*x*t**2 + 2*t/x, t), 3, DE) == \
        Poly(t**2, t)
    
    # TODO: 添加更多原始测试，包括需要使用 is_deriv_in_field() 的测试


def test_rischDE():
    # TODO: 添加更多 rischDE 的测试，包括文本中的测试案例
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    DE.decrement_level()
    assert rischDE(Poly(-2*x, x), Poly(1, x), Poly(1 - 2*x - 2*x**2, x),
                   Poly(1, x), DE) == \
        (Poly(x + 1, x), Poly(1, x))
```