# `D:\src\scipysrc\sympy\sympy\polys\tests\test_monomials.py`

```
"""Tests for tools and arithmetics for monomials of distributed polynomials. """

# 导入所需的符号和函数
from sympy.polys.monomials import (
    itermonomials, monomial_count,
    monomial_mul, monomial_div,
    monomial_gcd, monomial_lcm,
    monomial_max, monomial_min,
    monomial_divides, monomial_pow,
    Monomial,
)

# 导入异常处理类
from sympy.polys.polyerrors import ExactQuotientFailed

# 导入符号变量
from sympy.abc import a, b, c, x, y, z
# 导入符号变量创建函数
from sympy.core import S, symbols
# 导入测试工具中的异常断言
from sympy.testing.pytest import raises

# 定义测试函数
def test_monomials():

    # total_degree tests
    # 检查空列表情况下的单项式迭代，期望返回单项式1
    assert set(itermonomials([], 0)) == {S.One}
    assert set(itermonomials([], 1)) == {S.One}
    assert set(itermonomials([], 2)) == {S.One}

    # 检查空列表情况下，带指定排序阶数的单项式迭代，期望返回单项式1
    assert set(itermonomials([], 0, 0)) == {S.One}
    assert set(itermonomials([], 1, 0)) == {S.One}
    assert set(itermonomials([], 2, 0)) == {S.One}

    # 检查在空列表情况下请求超过列表长度的阶数，期望抛出StopIteration异常
    raises(StopIteration, lambda: next(itermonomials([], 0, 1)))
    raises(StopIteration, lambda: next(itermonomials([], 0, 2)))
    raises(StopIteration, lambda: next(itermonomials([], 0, 3)))

    # 检查空列表情况下，带指定排序阶数的单项式迭代，期望返回空集
    assert set(itermonomials([], 0, 1)) == set()
    assert set(itermonomials([], 0, 2)) == set()
    assert set(itermonomials([], 0, 3)) == set()

    # 检查对负阶数的单项式迭代，期望抛出ValueError异常
    raises(ValueError, lambda: set(itermonomials([], -1)))
    raises(ValueError, lambda: set(itermonomials([x], -1)))
    raises(ValueError, lambda: set(itermonomials([x, y], -1)))

    # 检查在含有一个变量的列表中不同阶数的单项式迭代，验证返回的单项式集合是否正确
    assert set(itermonomials([x], 0)) == {S.One}
    assert set(itermonomials([x], 1)) == {S.One, x}
    assert set(itermonomials([x], 2)) == {S.One, x, x**2}
    assert set(itermonomials([x], 3)) == {S.One, x, x**2, x**3}

    # 检查在含有两个变量的列表中不同阶数的单项式迭代，验证返回的单项式集合是否正确
    assert set(itermonomials([x, y], 0)) == {S.One}
    assert set(itermonomials([x, y], 1)) == {S.One, x, y}
    assert set(itermonomials([x, y], 2)) == {S.One, x, y, x**2, y**2, x*y}
    assert set(itermonomials([x, y], 3)) == \
            {S.One, x, y, x**2, x**3, y**2, y**3, x*y, x*y**2, y*x**2}

    # 检查在含有三个非交换变量的列表中不同阶数的单项式迭代，验证返回的单项式集合是否正确
    i, j, k = symbols('i j k', commutative=False)
    assert set(itermonomials([i, j, k], 0)) == {S.One}
    assert set(itermonomials([i, j, k], 1)) == {S.One, i, j, k}
    assert set(itermonomials([i, j, k], 2)) == \
           {S.One, i, j, k, i**2, j**2, k**2, i*j, i*k, j*i, j*k, k*i, k*j}

    # 检查在含有三个非交换变量的列表中更高阶数的单项式迭代，验证返回的单项式集合是否正确
    assert set(itermonomials([i, j, k], 3)) == \
            {S.One, i, j, k, i**2, j**2, k**2, i*j, i*k, j*i, j*k, k*i, k*j,
                    i**3, j**3, k**3,
                    i**2 * j, i**2 * k, j * i**2, k * i**2,
                    j**2 * i, j**2 * k, i * j**2, k * j**2,
                    k**2 * i, k**2 * j, i * k**2, j * k**2,
                    i*j*i, i*k*i, j*i*j, j*k*j, k*i*k, k*j*k,
                    i*j*k, i*k*j, j*i*k, j*k*i, k*i*j, k*j*i,
            }

    # 检查在含有混合变量的列表中不同阶数的单项式迭代，验证返回的单项式集合是否正确
    assert set(itermonomials([x, i, j], 0)) == {S.One}
    assert set(itermonomials([x, i, j], 1)) == {S.One, x, i, j}
    assert set(itermonomials([x, i, j], 2)) == {S.One, x, i, j, x*i, x*j, i*j, j*i, x**2, i**2, j**2}
    # 使用 itermonomials 函数生成一个生成器对象，生成的元素是关于变量 x, i, j 的所有三项式的集合
    assert set(itermonomials([x, i, j], 3)) == \
            {S.One, x, i, j, x*i, x*j, i*j, j*i, x**2, i**2, j**2,
                            x**3, i**3, j**3,
                            x**2 * i, x**2 * j,
                            x * i**2, j * i**2, i**2 * j, i*j*i,
                            x * j**2, i * j**2, j**2 * i, j*i*j,
                            x * i * j, x * j * i
            }

    # degree_list tests
    # 空列表作为参数，应该返回集合 {1}
    assert set(itermonomials([], [])) == {S.One}

    # 对于空的 degree_list 参数，应该抛出 ValueError 异常
    raises(ValueError, lambda: set(itermonomials([], [0])))
    raises(ValueError, lambda: set(itermonomials([], [1])))
    raises(ValueError, lambda: set(itermonomials([], [2])))

    # 对于多余的 degree_list 参数，应该抛出 ValueError 异常
    raises(ValueError, lambda: set(itermonomials([x], [1], [])))
    raises(ValueError, lambda: set(itermonomials([x], [1, 2], [])))
    raises(ValueError, lambda: set(itermonomials([x], [1, 2, 3], [])))

    raises(ValueError, lambda: set(itermonomials([x], [], [1])))
    raises(ValueError, lambda: set(itermonomials([x], [], [1, 2])))
    raises(ValueError, lambda: set(itermonomials([x], [], [1, 2, 3])))

    raises(ValueError, lambda: set(itermonomials([x, y], [1, 2], [1, 2, 3])))
    raises(ValueError, lambda: set(itermonomials([x, y, z], [1, 2, 3], [0, 1])))

    raises(ValueError, lambda: set(itermonomials([x], [1], [-1])))
    raises(ValueError, lambda: set(itermonomials([x, y], [1, 2], [1, -1])))

    raises(ValueError, lambda: set(itermonomials([], [], 1)))
    raises(ValueError, lambda: set(itermonomials([], [], 2)))
    raises(ValueError, lambda: set(itermonomials([], [], 3)))

    raises(ValueError, lambda: set(itermonomials([x, y], [0, 1], [1, 2])))
    raises(ValueError, lambda: set(itermonomials([x, y, z], [0, 0, 3], [0, 1, 2])))

    # 对于单一变量 x，不同的 degree_list 参数应该返回不同的集合
    assert set(itermonomials([x], [0])) == {S.One}
    assert set(itermonomials([x], [1])) == {S.One, x}
    assert set(itermonomials([x], [2])) == {S.One, x, x**2}
    assert set(itermonomials([x], [3])) == {S.One, x, x**2, x**3}

    # 对于单一变量 x，同时指定 degree_list 和 degree_multiset 参数应该返回不同的集合
    assert set(itermonomials([x], [3], [1])) == {x, x**3, x**2}
    assert set(itermonomials([x], [3], [2])) == {x**3, x**2}

    # 对于两个变量 x 和 y，不同的 degree_list 参数应该返回不同的集合
    assert set(itermonomials([x, y], 3, 3)) == {x**3, x**2*y, x*y**2, y**3}
    assert set(itermonomials([x, y], 3, 2)) == {x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3}

    assert set(itermonomials([x, y], [0, 0])) == {S.One}
    assert set(itermonomials([x, y], [0, 1])) == {S.One, y}
    assert set(itermonomials([x, y], [0, 2])) == {S.One, y, y**2}
    assert set(itermonomials([x, y], [0, 2], [0, 1])) == {y, y**2}
    assert set(itermonomials([x, y], [0, 2], [0, 2])) == {y**2}

    assert set(itermonomials([x, y], [1, 0])) == {S.One, x}
    assert set(itermonomials([x, y], [1, 1])) == {S.One, x, y, x*y}
    assert set(itermonomials([x, y], [1, 2])) == {S.One, x, y, x*y, y**2, x*y**2}
    assert set(itermonomials([x, y], [1, 2], [1, 1])) == {x*y, x*y**2}
    assert set(itermonomials([x, y], [1, 2], [1, 2])) == {x*y**2}

    assert set(itermonomials([x, y], [2, 0])) == {S.One, x, x**2}
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, y], [2, 1])) == {S.One, x, y, x*y, x**2, x**2*y}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, y], [2, 2])) == \
            {S.One, y**2, x*y**2, x, x*y, x**2, x**2*y**2, y, x**2*y}
    
    # 创建非交换符号变量 i, j, k
    i, j, k = symbols('i j k', commutative=False)
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], 2, 2)) == \
            {k*i, i**2, i*j, j*k, j*i, k**2, j**2, k*j, i*k}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], 3, 2)) == \
            {j*k**2, i*k**2, k*i*j, k*i**2, k**2, j*k*j, k*j**2, i*k*i, i*j,
                    j**2*k, i**2*j, j*i*k, j**3, i**3, k*j*i, j*k*i, j*i,
                    k**2*j, j*i**2, k*j, k*j*k, i*j*i, j*i*j, i*j**2, j**2,
                    k*i*k, i**2, j*k, i*k, i*k*j, k**3, i**2*k, j**2*i, k**2*i,
                    i*j*k, k*i
            }
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], [0, 0, 0])) == {S.One}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], [0, 0, 1])) == {1, k}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], [0, 1, 0])) == {1, j}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], [1, 0, 0])) == {i, 1}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], [0, 0, 2])) == {k**2, 1, k}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], [0, 2, 0])) == {1, j, j**2}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], [2, 0, 0])) == {i, 1, i**2}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], [1, 1, 1])) == {1, k, j, j*k, i*k, i, i*j, i*j*k}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([i, j, k], [2, 2, 2])) == \
            {1, k, i**2*k**2, j*k, j**2, i, i*k, j*k**2, i*j**2*k**2,
                    i**2*j, i**2*j**2, k**2, j**2*k, i*j**2*k,
                    j**2*k**2, i*j, i**2*k, i**2*j**2*k, j, i**2*j*k,
                    i*j**2, i*k**2, i*j*k, i**2*j**2*k**2, i*j*k**2, i**2, i**2*j*k**2
            }
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, j, k], [0, 0, 0])) == {S.One}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, j, k], [0, 0, 1])) == {1, k}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, j, k], [0, 1, 0])) == {1, j}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, j, k], [1, 0, 0])) == {x, 1}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, j, k], [0, 0, 2])) == {k**2, 1, k}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, j, k], [0, 2, 0])) == {1, j, j**2}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, j, k], [2, 0, 0])) == {x, 1, x**2}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, j, k], [1, 1, 1])) == {1, k, j, j*k, x*k, x, x*j, x*j*k}
    
    # 验证生成的多项式集合是否与预期的集合相等
    assert set(itermonomials([x, j, k], [2, 2, 2])) == \
            {1, k, x**2*k**2, j*k, j**2, x, x*k, j*k**2, x*j**2*k**2,
                    x**2*j, x**2*j**2, k**2, j**2*k, x*j**2*k,
                    j**2*k**2, x*j, x**2*k, x**2*j**2*k, j, x**2*j*k,
                    x*j**2, x*k**2, x*j*k, x**2*j**2*k**2, x*j*k**2, x**2, x**2*j*k**2
            }
# 测试函数，用于验证 monomial_count 函数的正确性
def test_monomial_count():
    assert monomial_count(2, 2) == 6
    assert monomial_count(2, 3) == 10

# 测试函数，用于验证 monomial_mul 函数的正确性
def test_monomial_mul():
    assert monomial_mul((3, 4, 1), (1, 2, 0)) == (4, 6, 1)

# 测试函数，用于验证 monomial_div 函数的正确性
def test_monomial_div():
    assert monomial_div((3, 4, 1), (1, 2, 0)) == (2, 2, 1)

# 测试函数，用于验证 monomial_gcd 函数的正确性
def test_monomial_gcd():
    assert monomial_gcd((3, 4, 1), (1, 2, 0)) == (1, 2, 0)

# 测试函数，用于验证 monomial_lcm 函数的正确性
def test_monomial_lcm():
    assert monomial_lcm((3, 4, 1), (1, 2, 0)) == (3, 4, 1)

# 测试函数，用于验证 monomial_max 函数的正确性
def test_monomial_max():
    assert monomial_max((3, 4, 5), (0, 5, 1), (6, 3, 9)) == (6, 5, 9)

# 测试函数，用于验证 monomial_pow 函数的正确性
def test_monomial_pow():
    assert monomial_pow((1, 2, 3), 3) == (3, 6, 9)

# 测试函数，用于验证 monomial_min 函数的正确性
def test_monomial_min():
    assert monomial_min((3, 4, 5), (0, 5, 1), (6, 3, 9)) == (0, 3, 1)

# 测试函数，用于验证 monomial_divides 函数的正确性
def test_monomial_divides():
    assert monomial_divides((1, 2, 3), (4, 5, 6)) is True
    assert monomial_divides((1, 2, 3), (0, 5, 6)) is False

# 测试函数，用于验证 Monomial 类的各种方法和属性
def test_Monomial():
    # 创建两个 Monomial 对象
    m = Monomial((3, 4, 1), (x, y, z))
    n = Monomial((1, 2, 0), (x, y, z))

    # 验证 as_expr 方法的返回值是否符合预期
    assert m.as_expr() == x**3*y**4*z
    assert n.as_expr() == x**1*y**2

    # 使用不同的变量验证 as_expr 方法的返回值是否符合预期
    assert m.as_expr(a, b, c) == a**3*b**4*c
    assert n.as_expr(a, b, c) == a**1*b**2

    # 验证 Monomial 对象的 exponents 和 gens 属性是否正确
    assert m.exponents == (3, 4, 1)
    assert m.gens == (x, y, z)
    assert n.exponents == (1, 2, 0)
    assert n.gens == (x, y, z)

    # 验证 Monomial 对象的相等性和索引访问是否正确
    assert m == (3, 4, 1)
    assert n != (3, 4, 1)
    assert m != (1, 2, 0)
    assert n == (1, 2, 0)
    assert (m == 1) is False
    assert m[0] == m[-3] == 3
    assert m[1] == m[-2] == 4
    assert m[2] == m[-1] == 1
    assert n[0] == n[-3] == 1
    assert n[1] == n[-2] == 2
    assert n[2] == n[-1] == 0

    # 验证切片操作是否正确
    assert m[:2] == (3, 4)
    assert n[:2] == (1, 2)

    # 验证乘法和除法操作是否正确
    assert m*n == Monomial((4, 6, 1))
    assert m/n == Monomial((2, 2, 1))
    assert m*(1, 2, 0) == Monomial((4, 6, 1))
    assert m/(1, 2, 0) == Monomial((2, 2, 1))

    # 验证 gcd 和 lcm 方法的正确性
    assert m.gcd(n) == Monomial((1, 2, 0))
    assert m.lcm(n) == Monomial((3, 4, 1))
    assert m.gcd((1, 2, 0)) == Monomial((1, 2, 0))
    assert m.lcm((1, 2, 0)) == Monomial((3, 4, 1))

    # 验证幂操作的正确性
    assert m**0 == Monomial((0, 0, 0))
    assert m**1 == m
    assert m**2 == Monomial((6, 8, 2))
    assert m**3 == Monomial((9, 12, 3))

    # 验证循环幂运算的正确性
    _a = Monomial((0, 0, 0))
    for n in range(10):
        assert _a == m**n
        _a *= m

    # 验证异常情况的处理
    raises(ExactQuotientFailed, lambda: m/Monomial((5, 2, 0)))
    mm = Monomial((1, 2, 3))
    raises(ValueError, lambda: mm.as_expr())
    assert str(mm) == 'Monomial((1, 2, 3))'
    assert str(m) == 'x**3*y**4*z**1'
    raises(NotImplementedError, lambda: m*1)
    raises(NotImplementedError, lambda: m/1)
    raises(ValueError, lambda: m**-1)
    raises(TypeError, lambda: m.gcd(3))
    raises(TypeError, lambda: m.lcm(3))
```