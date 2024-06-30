# `D:\src\scipysrc\sympy\sympy\polys\tests\test_appellseqs.py`

```
# 导入所需模块和函数，包括Rational作为Q，Poly，raises，以及几个Appell序列相关的函数
from sympy.core.numbers import Rational as Q
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.appellseqs import (bernoulli_poly, bernoulli_c_poly,
    euler_poly, genocchi_poly, andre_poly)
# 导入符号变量 x
from sympy.abc import x

# 测试函数：测试 bernoulli_poly 函数
def test_bernoulli_poly():
    # 断言异常值处理：当 n 为负数时抛出 ValueError 异常
    raises(ValueError, lambda: bernoulli_poly(-1, x))
    # 断言对于 n=1 时，使用 polys=True 参数的返回结果应为一个多项式对象，等于 x - 1/2
    assert bernoulli_poly(1, x, polys=True) == Poly(x - Q(1,2))

    # 下面依次断言 bernoulli_poly 函数在不同 n 值下的计算结果是否正确
    assert bernoulli_poly(0, x) == 1
    assert bernoulli_poly(1, x) == x - Q(1,2)
    assert bernoulli_poly(2, x) == x**2 - x + Q(1,6)
    assert bernoulli_poly(3, x) == x**3 - Q(3,2)*x**2 + Q(1,2)*x
    assert bernoulli_poly(4, x) == x**4 - 2*x**3 + x**2 - Q(1,30)
    assert bernoulli_poly(5, x) == x**5 - Q(5,2)*x**4 + Q(5,3)*x**3 - Q(1,6)*x
    assert bernoulli_poly(6, x) == x**6 - 3*x**5 + Q(5,2)*x**4 - Q(1,2)*x**2 + Q(1,42)

    # 断言函数的 dummy_eq 方法和 polys=True 参数的正确使用
    assert bernoulli_poly(1).dummy_eq(x - Q(1,2))
    assert bernoulli_poly(1, polys=True) == Poly(x - Q(1,2))

# 测试函数：测试 bernoulli_c_poly 函数
def test_bernoulli_c_poly():
    raises(ValueError, lambda: bernoulli_c_poly(-1, x))
    assert bernoulli_c_poly(1, x, polys=True) == Poly(x, domain='QQ')

    assert bernoulli_c_poly(0, x) == 1
    assert bernoulli_c_poly(1, x) == x
    assert bernoulli_c_poly(2, x) == x**2 - Q(1,3)
    assert bernoulli_c_poly(3, x) == x**3 - x
    assert bernoulli_c_poly(4, x) == x**4 - 2*x**2 + Q(7,15)
    assert bernoulli_c_poly(5, x) == x**5 - Q(10,3)*x**3 + Q(7,3)*x
    assert bernoulli_c_poly(6, x) == x**6 - 5*x**4 + 7*x**2 - Q(31,21)

    assert bernoulli_c_poly(1).dummy_eq(x)
    assert bernoulli_c_poly(1, polys=True) == Poly(x, domain='QQ')

    # 断言与 bernoulli_poly 函数之间的关系
    assert 2**8 * bernoulli_poly(8, (x+1)/2).expand() == bernoulli_c_poly(8, x)
    assert 2**9 * bernoulli_poly(9, (x+1)/2).expand() == bernoulli_c_poly(9, x)

# 测试函数：测试 genocchi_poly 函数
def test_genocchi_poly():
    raises(ValueError, lambda: genocchi_poly(-1, x))
    assert genocchi_poly(2, x, polys=True) == Poly(-2*x + 1)

    assert genocchi_poly(0, x) == 0
    assert genocchi_poly(1, x) == -1
    assert genocchi_poly(2, x) == 1 - 2*x
    assert genocchi_poly(3, x) == 3*x - 3*x**2
    assert genocchi_poly(4, x) == -1 + 6*x**2 - 4*x**3
    assert genocchi_poly(5, x) == -5*x + 10*x**3 - 5*x**4
    assert genocchi_poly(6, x) == 3 - 15*x**2 + 15*x**4 - 6*x**5

    assert genocchi_poly(2).dummy_eq(-2*x + 1)
    assert genocchi_poly(2, polys=True) == Poly(-2*x + 1)

    # 断言与 bernoulli_poly 和 bernoulli_c_poly 函数之间的关系
    assert 2 * (bernoulli_poly(8, x) - bernoulli_c_poly(8, x)) == genocchi_poly(8, x)
    assert 2 * (bernoulli_poly(9, x) - bernoulli_c_poly(9, x)) == genocchi_poly(9, x)

# 测试函数：测试 euler_poly 函数
def test_euler_poly():
    raises(ValueError, lambda: euler_poly(-1, x))
    assert euler_poly(1, x, polys=True) == Poly(x - Q(1,2))

    assert euler_poly(0, x) == 1
    assert euler_poly(1, x) == x - Q(1,2)
    assert euler_poly(2, x) == x**2 - x
    assert euler_poly(3, x) == x**3 - Q(3,2)*x**2 + Q(1,4)
    assert euler_poly(4, x) == x**4 - 2*x**3 + x
    # 断言：验证欧拉多项式的计算结果是否正确
    assert euler_poly(5, x) == x**5 - Q(5,2)*x**4 + Q(5,2)*x**2 - Q(1,2)

    # 断言：验证欧拉多项式的计算结果是否正确
    assert euler_poly(6, x) == x**6 - 3*x**5 + 5*x**3 - 3*x

    # 断言：验证欧拉多项式（以多项式形式）与给定的表达式是否等价
    assert euler_poly(1).dummy_eq(x - Q(1,2))

    # 断言：验证以多项式形式返回的欧拉多项式是否正确
    assert euler_poly(1, polys=True) == Poly(x - Q(1,2))

    # 断言：验证格诺奇多项式的计算结果是否正确
    assert genocchi_poly(9, x) == euler_poly(8, x) * -9

    # 断言：验证格诺奇多项式的计算结果是否正确
    assert genocchi_poly(10, x) == euler_poly(9, x) * -10
# 定义测试函数 test_andre_poly
def test_andre_poly():
    # 测试调用 andre_poly 函数，预期会引发 ValueError，因为第一个参数为负数
    raises(ValueError, lambda: andre_poly(-1, x))
    # 断言调用 andre_poly 函数，当参数为 1 时返回 Poly(x)，并指定 polys=True
    assert andre_poly(1, x, polys=True) == Poly(x)

    # 断言调用 andre_poly 函数，当参数为 0 时返回 1
    assert andre_poly(0, x) == 1
    # 断言调用 andre_poly 函数，当参数为 1 时返回 x
    assert andre_poly(1, x) == x
    # 断言调用 andre_poly 函数，当参数为 2 时返回 x**2 - 1
    assert andre_poly(2, x) == x**2 - 1
    # 断言调用 andre_poly 函数，当参数为 3 时返回 x**3 - 3*x
    assert andre_poly(3, x) == x**3 - 3*x
    # 断言调用 andre_poly 函数，当参数为 4 时返回 x**4 - 6*x**2 + 5
    assert andre_poly(4, x) == x**4 - 6*x**2 + 5
    # 断言调用 andre_poly 函数，当参数为 5 时返回 x**5 - 10*x**3 + 25*x
    assert andre_poly(5, x) == x**5 - 10*x**3 + 25*x
    # 断言调用 andre_poly 函数，当参数为 6 时返回 x**6 - 15*x**4 + 75*x**2 - 61
    assert andre_poly(6, x) == x**6 - 15*x**4 + 75*x**2 - 61

    # 断言调用 andre_poly 函数，当参数为 1 时返回的结果 dummy_eq(x) 为 True
    assert andre_poly(1).dummy_eq(x)
    # 断言调用 andre_poly 函数，当参数为 1 时返回 Poly(x)，并指定 polys=True
    assert andre_poly(1, polys=True) == Poly(x)
```