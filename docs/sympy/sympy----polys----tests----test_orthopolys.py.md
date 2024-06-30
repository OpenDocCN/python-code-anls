# `D:\src\scipysrc\sympy\sympy\polys\tests\test_orthopolys.py`

```
"""Tests for efficient functions for generating orthogonal polynomials. """

# 导入所需的模块和函数
from sympy.core.numbers import Rational as Q  # 导入 Rational 类并重命名为 Q
from sympy.core.singleton import S  # 导入 S 单例
from sympy.core.symbol import symbols  # 导入 symbols 函数
from sympy.polys.polytools import Poly  # 导入 Poly 类
from sympy.testing.pytest import raises  # 导入 raises 函数

# 导入正交多项式生成函数
from sympy.polys.orthopolys import (
    jacobi_poly,  # Jacobi 多项式函数
    gegenbauer_poly,  # Gegenbauer 多项式函数
    chebyshevt_poly,  # Chebyshev 第一类多项式函数
    chebyshevu_poly,  # Chebyshev 第二类多项式函数
    hermite_poly,  # Hermite 多项式函数
    hermite_prob_poly,  # Hermite 概率多项式函数
    legendre_poly,  # Legendre 多项式函数
    laguerre_poly,  # Laguerre 多项式函数
    spherical_bessel_fn,  # 球贝塞尔函数
)

# 导入变量 x, a, b
from sympy.abc import x, a, b


# 测试 Jacobi 多项式函数
def test_jacobi_poly():
    raises(ValueError, lambda: jacobi_poly(-1, a, b, x))  # 检查负的 n 值是否会引发 ValueError

    assert jacobi_poly(1, a, b, x, polys=True) == Poly(
        (a/2 + b/2 + 1)*x + a/2 - b/2, x, domain='ZZ(a,b)')  # 检查 Jacobi 多项式的计算结果是否正确

    assert jacobi_poly(0, a, b, x) == 1  # 检查 n=0 时 Jacobi 多项式的值是否为 1
    assert jacobi_poly(1, a, b, x) == a/2 - b/2 + x*(a/2 + b/2 + 1)  # 检查 n=1 时 Jacobi 多项式的计算结果是否正确
    assert jacobi_poly(2, a, b, x) == (a**2/8 - a*b/4 - a/8 + b**2/8 - b/8 +
                                       x**2*(a**2/8 + a*b/4 + a*Q(7, 8) + b**2/8 +
                                             b*Q(7, 8) + Q(3, 2)) + x*(a**2/4 +
                                            a*Q(3, 4) - b**2/4 - b*Q(3, 4)) - S.Half)  # 检查 n=2 时 Jacobi 多项式的计算结果是否正确

    assert jacobi_poly(1, a, b, polys=True) == Poly(
        (a/2 + b/2 + 1)*x + a/2 - b/2, x, domain='ZZ(a,b)')  # 检查 Jacobi 多项式的计算结果是否正确


# 测试 Gegenbauer 多项式函数
def test_gegenbauer_poly():
    raises(ValueError, lambda: gegenbauer_poly(-1, a, x))  # 检查负的 n 值是否会引发 ValueError

    assert gegenbauer_poly(
        1, a, x, polys=True) == Poly(2*a*x, x, domain='ZZ(a)')  # 检查 Gegenbauer 多项式的计算结果是否正确

    assert gegenbauer_poly(0, a, x) == 1  # 检查 n=0 时 Gegenbauer 多项式的值是否为 1
    assert gegenbauer_poly(1, a, x) == 2*a*x  # 检查 n=1 时 Gegenbauer 多项式的计算结果是否正确
    assert gegenbauer_poly(2, a, x) == -a + x**2*(2*a**2 + 2*a)  # 检查 n=2 时 Gegenbauer 多项式的计算结果是否正确
    assert gegenbauer_poly(
        3, a, x) == x**3*(4*a**3/3 + 4*a**2 + a*Q(8, 3)) + x*(-2*a**2 - 2*a)  # 检查 n=3 时 Gegenbauer 多项式的计算结果是否正确

    assert gegenbauer_poly(1, S.Half).dummy_eq(x)  # 检查当 a=S.Half 时，Gegenbauer 多项式是否等于 x
    assert gegenbauer_poly(1, a, polys=True) == Poly(2*a*x, x, domain='ZZ(a)')  # 检查 Gegenbauer 多项式的计算结果是否正确


# 测试 Chebyshev 第一类多项式函数
def test_chebyshevt_poly():
    raises(ValueError, lambda: chebyshevt_poly(-1, x))  # 检查负的 n 值是否会引发 ValueError

    assert chebyshevt_poly(1, x, polys=True) == Poly(x)  # 检查 Chebyshev 第一类多项式的计算结果是否正确

    assert chebyshevt_poly(0, x) == 1  # 检查 n=0 时 Chebyshev 第一类多项式的值是否为 1
    assert chebyshevt_poly(1, x) == x  # 检查 n=1 时 Chebyshev 第一类多项式的计算结果是否正确
    assert chebyshevt_poly(2, x) == 2*x**2 - 1  # 检查 n=2 时 Chebyshev 第一类多项式的计算结果是否正确
    assert chebyshevt_poly(3, x) == 4*x**3 - 3*x  # 检查 n=3 时 Chebyshev 第一类多项式的计算结果是否正确
    assert chebyshevt_poly(4, x) == 8*x**4 - 8*x**2 + 1  # 检查 n=4 时 Chebyshev 第一类多项式的计算结果是否正确
    assert chebyshevt_poly(5, x) == 16*x**5 - 20*x**3 + 5*x  # 检查 n=5 时 Chebyshev 第一类多项式的计算结果是否正确
    assert chebyshevt_poly(6, x) == 32*x**6 - 48*x**4 + 18*x**2 - 1  # 检查 n=6 时 Chebyshev 第一类多项式的计算结果是否正确
    assert chebyshevt_poly(75, x) == (2*chebyshevt_poly(37, x)*chebyshevt_poly(38, x) - x).expand()  # 检查 n=75 时 Chebyshev 第一类多项式的计算结果是否正确
    assert chebyshevt_poly(100, x) == (2*chebyshevt_poly(50, x)**2 - 1).expand()  # 检查 n=100 时 Chebyshev 第一类多项式的计算结果是否正确

    assert chebyshevt_poly(1).dummy_eq(x)  # 检查当 n=1 时，Chebyshev 第一类多项式是否等于 x
    assert chebyshevt_poly(1, polys=True) == Poly(x)  # 检查 Chebyshev 第一类多项式的计算结果是否正确


# 测试 Chebyshev 第二类多项式函数
def test_chebyshevu_poly():
    raises(ValueError, lambda: chebyshevu_poly(-1, x))  # 检查负的 n 值是否会引发 ValueError

    assert chebyshevu_poly(1, x, polys=True) == Poly(2*x)  # 检查 Chebyshev 第二类多项式的计算结果是否正确

    assert chebyshevu_poly(0, x) == 1  # 检查 n=0 时 Chebyshev 第二类多项式的值是否为 1
    assert chebyshevu_poly(
    # 断言：验证 Chebyshev 多项式的值是否符合预期
    assert chebyshevu_poly(5, x) == 32*x**5 - 32*x**3 + 6*x
    
    # 断言：验证 Chebyshev 多项式的值是否符合预期
    assert chebyshevu_poly(6, x) == 64*x**6 - 80*x**4 + 24*x**2 - 1
    
    # 断言：验证 Chebyshev 多项式的虚部是否与预期相等
    assert chebyshevu_poly(1).dummy_eq(2*x)
    
    # 断言：验证 Chebyshev 多项式在使用多项式标志时的返回结果是否符合预期
    assert chebyshevu_poly(1, polys=True) == Poly(2*x)
# 定义测试函数 test_hermite_poly，用于测试 Hermite 多项式的计算
def test_hermite_poly():
    # 检查当 n 为负数时是否引发 ValueError 异常，使用 lambda 表达式调用 hermite_poly 函数
    raises(ValueError, lambda: hermite_poly(-1, x))

    # 断言计算 Hermite 多项式 H_1(x) 是否等于 2*x，使用 polys=True 返回多项式对象
    assert hermite_poly(1, x, polys=True) == Poly(2*x)

    # 断言计算 Hermite 多项式 H_0(x) 是否等于 1
    assert hermite_poly(0, x) == 1
    # 断言计算 Hermite 多项式 H_1(x) 是否等于 2*x
    assert hermite_poly(1, x) == 2*x
    # 断言计算 Hermite 多项式 H_2(x) 是否等于 4*x**2 - 2
    assert hermite_poly(2, x) == 4*x**2 - 2
    # 断言计算 Hermite 多项式 H_3(x) 是否等于 8*x**3 - 12*x
    assert hermite_poly(3, x) == 8*x**3 - 12*x
    # 断言计算 Hermite 多项式 H_4(x) 是否等于 16*x**4 - 48*x**2 + 12
    assert hermite_poly(4, x) == 16*x**4 - 48*x**2 + 12
    # 断言计算 Hermite 多项式 H_5(x) 是否等于 32*x**5 - 160*x**3 + 120*x
    assert hermite_poly(5, x) == 32*x**5 - 160*x**3 + 120*x
    # 断言计算 Hermite 多项式 H_6(x) 是否等于 64*x**6 - 480*x**4 + 720*x**2 - 120
    assert hermite_poly(6, x) == 64*x**6 - 480*x**4 + 720*x**2 - 120

    # 断言计算 Hermite 多项式 H_1(x) 是否近似等于 2*x，使用 dummy_eq 进行比较
    assert hermite_poly(1).dummy_eq(2*x)
    # 断言计算 Hermite 多项式 H_1(x) 是否等于 2*x，使用 polys=True 返回多项式对象
    assert hermite_poly(1, polys=True) == Poly(2*x)


# 定义测试函数 test_hermite_prob_poly，用于测试物理 Hermite 多项式的计算
def test_hermite_prob_poly():
    # 检查当 n 为负数时是否引发 ValueError 异常，使用 lambda 表达式调用 hermite_prob_poly 函数
    raises(ValueError, lambda: hermite_prob_poly(-1, x))

    # 断言计算物理 Hermite 多项式 H'_1(x) 是否等于 x，使用 polys=True 返回多项式对象
    assert hermite_prob_poly(1, x, polys=True) == Poly(x)

    # 断言计算物理 Hermite 多项式 H'_0(x) 是否等于 1
    assert hermite_prob_poly(0, x) == 1
    # 断言计算物理 Hermite 多项式 H'_1(x) 是否等于 x
    assert hermite_prob_poly(1, x) == x
    # 断言计算物理 Hermite 多项式 H'_2(x) 是否等于 x**2 - 1
    assert hermite_prob_poly(2, x) == x**2 - 1
    # 断言计算物理 Hermite 多项式 H'_3(x) 是否等于 x**3 - 3*x
    assert hermite_prob_poly(3, x) == x**3 - 3*x
    # 断言计算物理 Hermite 多项式 H'_4(x) 是否等于 x**4 - 6*x**2 + 3
    assert hermite_prob_poly(4, x) == x**4 - 6*x**2 + 3
    # 断言计算物理 Hermite 多项式 H'_5(x) 是否等于 x**5 - 10*x**3 + 15*x
    assert hermite_prob_poly(5, x) == x**5 - 10*x**3 + 15*x
    # 断言计算物理 Hermite 多项式 H'_6(x) 是否等于 x**6 - 15*x**4 + 45*x**2 - 15
    assert hermite_prob_poly(6, x) == x**6 - 15*x**4 + 45*x**2 - 15

    # 断言计算物理 Hermite 多项式 H'_1(x) 是否近似等于 x，使用 dummy_eq 进行比较
    assert hermite_prob_poly(1).dummy_eq(x)
    # 断言计算物理 Hermite 多项式 H'_1(x) 是否等于 x，使用 polys=True 返回多项式对象
    assert hermite_prob_poly(1, polys=True) == Poly(x)


# 定义测试函数 test_legendre_poly，用于测试 Legendre 多项式的计算
def test_legendre_poly():
    # 检查当 n 为负数时是否引发 ValueError 异常，使用 lambda 表达式调用 legendre_poly 函数
    raises(ValueError, lambda: legendre_poly(-1, x))

    # 断言计算 Legendre 多项式 P_1(x) 是否等于 x，使用 polys=True 返回多项式对象
    assert legendre_poly(1, x, polys=True) == Poly(x, domain='QQ')

    # 断言计算 Legendre 多项式 P_0(x) 是否等于 1
    assert legendre_poly(0, x) == 1
    # 断言计算 Legendre 多项式 P_1(x) 是否等于 x
    assert legendre_poly(1, x) == x
    # 断言计算 Legendre 多项式 P_2(x) 是否等于 (3/2)*x**2 - 1/2
    assert legendre_poly(2, x) == Q(3, 2)*x**2 - Q(1, 2)
    # 断言计算 Legendre 多项式 P_3(x) 是否等于 (5/2)*x**3 - (3/2)*x
    assert legendre_poly(3, x) == Q(5, 2)*x**3 - Q(3, 2)*x
    # 断言计算 Legendre 多项式 P_4(x) 是否等于 (35/8)*x**4 - (30/8)*x**2 + (3/8)
    assert legendre_poly(4, x) == Q(35, 8)*x**4 - Q(30, 8)*x**2 + Q(3, 8)
    # 断言计算 Legendre 多项式 P_5(x) 是否等于 (63/8)*x**5 - (70/8)*x**3 + (15/8)*x
    assert legendre_poly(5, x) == Q(63, 8)*x**5 - Q(70, 8)*x**3 + Q(15, 8)*x
    # 断言计算 Legendre 多项式 P_6(x) 是否等于 (231/16)*x**6 - (315/16)*x**4 + (105/16)*x**2 - (5/16)
    assert legendre_poly(6, x) == Q(231, 16)*x**6 - Q(315, 16)*x**4 + Q(105, 16)*x**2 - Q(5, 16)

    # 断言计算 Legendre 多项式 P_1(x) 是否近似等于 x，使用 dummy_eq 进行比较
    assert legendre_poly(1).dummy_eq(x)
    # 断言计算 Legendre 多项式 P_1(x) 是否等于 x，使用 polys=True 返回多项式对象
    assert legendre_poly(1, polys=True) == Poly(x)


# 定义测试函数 test_laguerre_poly，用于测试 Laguerre 多项式的计算
def test_laguerre_poly():
    # 检查当 n 为负数时是否引发 ValueError 异常，使用 lambda 表达式调用 laguerre_poly 函数
    raises(ValueError, lambda: laguerre_poly(-1, x))

    # 断言计算 Laguerre 多
    # 使用 assert 断言来验证 laguerre_poly 函数在给定参数下的返回值是否符合预期
    assert laguerre_poly(1, polys=True) == Poly(-x + 1)
# 定义一个测试函数，用于测试球贝塞尔函数的特定情况
def test_spherical_bessel_fn():
    # 使用 sympy 符号 x 和 z 来表示数学符号变量
    x, z = symbols("x z")
    # 断言球贝塞尔函数的值为 1/z**2
    assert spherical_bessel_fn(1, z) == 1/z**2
    # 断言球贝塞尔函数的值为 -1/z + 3/z**3
    assert spherical_bessel_fn(2, z) == -1/z + 3/z**3
    # 断言球贝塞尔函数的值为 -6/z**2 + 15/z**4
    assert spherical_bessel_fn(3, z) == -6/z**2 + 15/z**4
    # 断言球贝塞尔函数的值为 1/z - 45/z**3 + 105/z**5
    assert spherical_bessel_fn(4, z) == 1/z - 45/z**3 + 105/z**5
```