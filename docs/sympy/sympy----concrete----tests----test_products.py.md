# `D:\src\scipysrc\sympy\sympy\concrete\tests\test_products.py`

```
# 导入SymPy库中的具体模块和函数
from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import Sum
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (rf, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises

# 定义整数类型的符号变量
a, k, n, m, x = symbols('a,k,n,m,x', integer=True)
# 定义一个未指定类型的函数 f
f = Function('f')

# 定义一个测试函数，用于验证 Karr 的乘积约定
def test_karr_convention():
    # 测试我们希望满足的 Karr 乘积约定。
    # 参考他的论文 "Summation in Finite Terms" 中对此定义的详细推理。
    # 对于和的约定，可以在第309页和第1.4节中找到相关描述。
    # 对于乘积，我们可以类似地得到：
    #
    # \prod_{m <= i < n} f(i) '具有显而易见的意义'    当 m < n 时
    # \prod_{m <= i < n} f(i) = 0                    当 m = n 时
    # \prod_{m <= i < n} f(i) = 1 / \prod_{n <= i < m} f(i)  当 m > n 时
    #
    # 需要注意的是，他定义的所有乘积的上限都是*独占的*。
    # 相比之下，SymPy和通常的数学符号表示方式为：
    #
    # prod_{i = a}^b f(i) = f(a) * f(a+1) * ... * f(b-1) * f(b)
    #
    # 其中上限是*包含的*。因此，将两者之间进行转换，我们有：
    #
    # \prod_{m <= i < n} f(i) = \prod_{i = m}^{n-1} f(i)
    #
    # 这里有意使用了两种不同的方式来排版乘积及其上下限。

    # 定义整数类型的符号变量
    i = Symbol("i", integer=True)
    k = Symbol("k", integer=True)
    j = Symbol("j", integer=True, positive=True)

    # 使用具体因子和符号限制的简单示例。

    # 普通的乘积：m = k，n = k + j，因此 m < n：
    m = k
    n = k + j

    a = m
    b = n - 1
    S1 = Product(i**2, (i, a, b)).doit()

    # 反向的乘积：m = k + j，n = k，因此 m > n：
    m = k + j
    n = k

    a = m
    b = n - 1
    S2 = Product(i**2, (i, a, b)).doit()

    assert S1 * S2 == 1

    # 测试空乘积：m = k，n = k，因此 m = n：
    m = k
    n = k

    a = m
    b = n - 1
    Sz = Product(i**2, (i, a, b)).doit()

    assert Sz == 1

    # 另一个示例，这次使用未指定因子和数值限制。（我们不能在同一个示例中执行两个测试。）
    f = Function("f")

    # 普通乘积，m < n：
    m = 2
    n = 11

    a = m
    b = n - 1
    S1 = Product(f(i), (i, a, b)).doit()

    # 反向乘积，m > n：
    m = 11
    n = 2

    a = m
    b = n - 1
    # 计算积分 S2 = ∏ f(i) 对于变量 i 从 a 到 b 的乘积，并求解
    S2 = Product(f(i), (i, a, b)).doit()

    # 断言简化后的 S1 * S2 等于 1
    assert simplify(S1 * S2) == 1

    # 测试当 m = n 时空积的情况：
    m = 5
    n = 5

    # 设置变量 a 和 b，以便计算空积 Sz = ∏ f(i) 对于变量 i 从 a 到 b 的乘积，并求解
    a = m
    b = n - 1
    Sz = Product(f(i), (i, a, b)).doit()

    # 断言 Sz 的值应为 1
    assert Sz == 1
def test_karr_proposition_2a():
    # Test Karr, page 309, proposition 2, part a
    # 定义符号变量 i, u, v，均为整数类型
    i, u, v = symbols('i u v', integer=True)

    def test_the_product(m, n):
        # 计算函数 g(i) = i**3 + 2*i**2 - 3*i
        g = i**3 + 2*i**2 - 3*i
        # 计算 f(i) = Delta g(i) = g(i+1) / g(i)
        f = simplify(g.subs(i, i+1) / g)
        # 计算乘积 Product_{m <= i < n} f(i)
        a = m
        b = n - 1
        P = Product(f, (i, a, b)).doit()
        # 断言测试是否满足 Product_{m <= i < n} f(i) = g(n) / g(m)
        assert combsimp(P / (g.subs(i, n) / g.subs(i, m))) == 1

    # m < n
    test_the_product(u, u + v)
    # m = n
    test_the_product(u, u)
    # m > n
    test_the_product(u + v, u)


def test_karr_proposition_2b():
    # Test Karr, page 309, proposition 2, part b
    # 定义符号变量 i, u, v, w，均为整数类型
    i, u, v, w = symbols('i u v w', integer=True)

    def test_the_product(l, n, m):
        # 计算函数 s(i) = i**3
        s = i**3
        # 计算第一个乘积 S1 = Product_{l <= i < n} s(i)
        a = l
        b = n - 1
        S1 = Product(s, (i, a, b)).doit()
        # 计算第二个乘积 S2 = Product_{l <= i < m} s(i)
        a = l
        b = m - 1
        S2 = Product(s, (i, a, b)).doit()
        # 计算第三个乘积 S3 = Product_{m <= i < n} s(i)
        a = m
        b = n - 1
        S3 = Product(s, (i, a, b)).doit()
        # 断言测试是否满足 S1 = S2 * S3
        assert combsimp(S1 / (S2 * S3)) == 1

    # l < m < n
    test_the_product(u, u + v, u + v + w)
    # l < m = n
    test_the_product(u, u + v, u + v)
    # l < m > n
    test_the_product(u, u + v + w, v)
    # l = m < n
    test_the_product(u, u, u + v)
    # l = m = n
    test_the_product(u, u, u)
    # l = m > n
    test_the_product(u + v, u + v, u)
    # l > m < n
    test_the_product(u + v, u, u + w)
    # l > m = n
    test_the_product(u + v, u, u)
    # l > m > n
    test_the_product(u + v + w, u + v, u)


def test_simple_products():
    # 断言测试多个简单乘积的结果
    assert product(2, (k, a, n)) == 2**(n - a + 1)
    assert product(k, (k, 1, n)) == factorial(n)
    assert product(k**3, (k, 1, n)) == factorial(n)**3

    assert product(k + 1, (k, 0, n - 1)) == factorial(n)
    assert product(k + 1, (k, a, n - 1)) == rf(1 + a, n - a)

    assert product(cos(k), (k, 0, 5)) == cos(1)*cos(2)*cos(3)*cos(4)*cos(5)
    assert product(cos(k), (k, 3, 5)) == cos(3)*cos(4)*cos(5)
    assert product(cos(k), (k, 1, Rational(5, 2))) != cos(1)*cos(2)

    assert isinstance(product(k**k, (k, 1, n)), Product)

    assert Product(x**k, (k, 1, n)).variables == [k]

    raises(ValueError, lambda: Product(n))
    raises(ValueError, lambda: Product(n, k))
    raises(ValueError, lambda: Product(n, k, 1))
    raises(ValueError, lambda: Product(n, k, 1, 10))
    raises(ValueError, lambda: Product(n, (k, 1)))

    assert product(1, (n, 1, oo)) == 1  # issue 8301
    assert product(2, (n, 1, oo)) is oo
    assert product(-1, (n, 1, oo)).func is Product


def test_multiple_products():
    # 断言测试多重乘积的结果
    assert product(x, (n, 1, k), (k, 1, m)) == x**(m**2/2 + m/2)
    assert product(f(n), (
        n, 1, m), (m, 1, k)) == Product(f(n), (n, 1, m), (m, 1, k)).doit()
    # 断言：验证对于给定的 f(n) 函数和范围 (m, 1, k), (n, 1, k)，以下表达式成立
    # 检查条件1：Product(f(n), (m, 1, k), (n, 1, k)).doit() 等于
    # 检查条件2：Product(Product(f(n), (m, 1, k)), (n, 1, k)).doit() 等于
    # 检查条件3：product(f(n), (m, 1, k), (n, 1, k)) 等于
    # 检查条件4：product(product(f(n), (m, 1, k)), (n, 1, k)) 等于
    # 检查条件5：Product(f(n)**k, (n, 1, k))
    assert Product(f(n), (m, 1, k), (n, 1, k)).doit() == \
        Product(Product(f(n), (m, 1, k)), (n, 1, k)).doit() == \
        product(f(n), (m, 1, k), (n, 1, k)) == \
        product(product(f(n), (m, 1, k)), (n, 1, k)) == \
        Product(f(n)**k, (n, 1, k))
    
    # 断言：验证对于给定的 x 变量和范围 (x, 1, k), (k, 1, n)，以下表达式成立
    # 检查条件1：Product(x, (x, 1, k), (k, 1, n)).doit() 等于 Product(factorial(k), (k, 1, n))
    assert Product(
        x, (x, 1, k), (k, 1, n)).doit() == Product(factorial(k), (k, 1, n))
    
    # 断言：验证对于给定的 x 变量和范围 (n, 1, k), (k, 1, m)，variables 属性返回的列表
    assert Product(x**k, (n, 1, k), (k, 1, m)).variables == [n, k]
def test_rational_products():
    # 检查有理数乘积函数是否正确
    assert product(1 + 1/k, (k, 1, n)) == rf(2, n)/factorial(n)


def test_special_products():
    # 检查特殊乘积函数是否正确：Wallis乘积
    assert product((4*k)**2 / (4*k**2 - 1), (k, 1, n)) == \
        4**n*factorial(n)**2/rf(S.Half, n)/rf(Rational(3, 2), n)

    # 检查特殊乘积函数是否正确：Euler的正弦乘积公式
    assert product(1 + a/k**2, (k, 1, n)) == \
        rf(1 - sqrt(-a), n)*rf(1 + sqrt(-a), n)/factorial(n)**2


def test__eval_product():
    from sympy.abc import i, n
    # issue 4809：检查问题编号4809
    a = Function('a')
    assert product(2*a(i), (i, 1, n)) == 2**n * Product(a(i), (i, 1, n))
    # issue 4810：检查问题编号4810
    assert product(2**i, (i, 1, n)) == 2**(n*(n + 1)/2)
    k, m = symbols('k m', integer=True)
    assert product(2**i, (i, k, m)) == 2**(-k**2/2 + k/2 + m**2/2 + m/2)
    n = Symbol('n', negative=True, integer=True)
    p = Symbol('p', positive=True, integer=True)
    assert product(2**i, (i, n, p)) == 2**(-n**2/2 + n/2 + p**2/2 + p/2)
    assert product(2**i, (i, p, n)) == 2**(n**2/2 + n/2 - p**2/2 + p/2)


def test_product_pow():
    # issue 4817：检查问题编号4817
    assert product(2**f(k), (k, 1, n)) == 2**Sum(f(k), (k, 1, n))
    assert product(2**(2*f(k)), (k, 1, n)) == 2**Sum(2*f(k), (k, 1, n))


def test_infinite_product():
    # issue 5737：检查问题编号5737
    assert isinstance(Product(2**(1/factorial(n)), (n, 0, oo)), Product)


def test_conjugate_transpose():
    p = Product(x**k, (k, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()

    A, B = symbols("A B", commutative=False)
    p = Product(A*B**k, (k, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()

    p = Product(B**k*A, (k, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()


def test_simplify_prod():
    y, t, b, c, v, d = symbols('y, t, b, c, v, d', integer=True)

    _simplify = lambda e: simplify(e, doit=False)
    assert _simplify(Product(x*y, (x, n, m), (y, a, k)) * \
        Product(y, (x, n, m), (y, a, k))) == \
            Product(x*y**2, (x, n, m), (y, a, k))
    assert _simplify(3 * y* Product(x, (x, n, m)) * Product(x, (x, m + 1, a))) \
        == 3 * y * Product(x, (x, n, a))
    assert _simplify(Product(x, (x, k + 1, a)) * Product(x, (x, n, k))) == \
        Product(x, (x, n, a))
    assert _simplify(Product(x, (x, k + 1, a)) * Product(x + 1, (x, n, k))) == \
        Product(x, (x, k + 1, a)) * Product(x + 1, (x, n, k))
    assert _simplify(Product(x, (t, a, b)) * Product(y, (t, a, b)) * \
        Product(x, (t, b+1, c))) == Product(x*y, (t, a, b)) * \
            Product(x, (t, b+1, c))
    # 断言：简化左侧表达式 Product(x, (t, a, b)) * Product(x, (t, b+1, c)) * Product(y, (t, a, b))
    # 等于右侧表达式 Product(x*y, (t, a, b)) * Product(x, (t, b+1, c))
    assert _simplify(Product(x, (t, a, b)) * Product(x, (t, b+1, c)) * \
        Product(y, (t, a, b))) == Product(x*y, (t, a, b)) * \
            Product(x, (t, b+1, c))

    # 断言：简化表达式 Product(sin(t)**2 + cos(t)**2 + 1, (t, a, b))
    # 等于表达式 Product(2, (t, a, b))
    assert _simplify(Product(sin(t)**2 + cos(t)**2 + 1, (t, a, b))) == \
        Product(2, (t, a, b))

    # 断言：简化表达式 Product(sin(t)**2 + cos(t)**2 - 1, (t, a, b))
    # 等于表达式 Product(0, (t, a, b))
    assert _simplify(Product(sin(t)**2 + cos(t)**2 - 1, (t, a, b))) == \
           Product(0, (t, a, b))

    # 断言：简化表达式 Product(v*Product(sin(t)**2 + cos(t)**2, (t, a, b)), (v, c, d))
    # 等于表达式 Product(v*Product(1, (t, a, b)), (v, c, d))
    assert _simplify(Product(v*Product(sin(t)**2 + cos(t)**2, (t, a, b)),
                             (v, c, d))) == Product(v*Product(1, (t, a, b)), (v, c, d))
# 定义一个测试函数，用于测试 Product 类的 change_index 方法
def test_change_index():
    # 定义符号变量 b, y, c, d, z，均为整数类型
    b, y, c, d, z = symbols('b, y, c, d, z', integer=True)

    # 断言：测试 Product 对象的 change_index 方法
    assert Product(x, (x, a, b)).change_index(x, x + 1, y) == \
        Product(y - 1, (y, a + 1, b + 1))
    
    # 断言：测试 Product 对象的 change_index 方法
    assert Product(x**2, (x, a, b)).change_index(x, x - 1) == \
        Product((x + 1)**2, (x, a - 1, b - 1))
    
    # 断言：测试 Product 对象的 change_index 方法
    assert Product(x**2, (x, a, b)).change_index(x, -x, y) == \
        Product((-y)**2, (y, -b, -a))
    
    # 断言：测试 Product 对象的 change_index 方法
    assert Product(x, (x, a, b)).change_index(x, -x - 1) == \
        Product(-x - 1, (x, -b - 1, -a - 1))
    
    # 断言：测试 Product 对象的 change_index 方法
    assert Product(x*y, (x, a, b), (y, c, d)).change_index(x, x - 1, z) == \
        Product((z + 1)*y, (z, a - 1, b - 1), (y, c, d))


# 定义一个测试函数，用于测试 Product 类的 reorder 方法
def test_reorder():
    # 定义符号变量 b, y, c, d, z，均为整数类型
    b, y, c, d, z = symbols('b, y, c, d, z', integer=True)

    # 断言：测试 Product 对象的 reorder 方法
    assert Product(x*y, (x, a, b), (y, c, d)).reorder((0, 1)) == \
        Product(x*y, (y, c, d), (x, a, b))
    
    # 断言：测试 Product 对象的 reorder 方法
    assert Product(x, (x, a, b), (x, c, d)).reorder((0, 1)) == \
        Product(x, (x, c, d), (x, a, b))
    
    # 断言：测试 Product 对象的 reorder 方法
    assert Product(x*y + z, (x, a, b), (z, m, n), (y, c, d)).reorder(\
        (2, 0), (0, 1)) == Product(x*y + z, (z, m, n), (y, c, d), (x, a, b))
    
    # 断言：测试 Product 对象的 reorder 方法
    assert Product(x*y*z, (x, a, b), (y, c, d), (z, m, n)).reorder(\
        (0, 1), (1, 2), (0, 2)) == \
        Product(x*y*z, (x, a, b), (z, m, n), (y, c, d))
    
    # 断言：测试 Product 对象的 reorder 方法
    assert Product(x*y*z, (x, a, b), (y, c, d), (z, m, n)).reorder(\
        (x, y), (y, z), (x, z)) == \
        Product(x*y*z, (x, a, b), (z, m, n), (y, c, d))
    
    # 断言：测试 Product 对象的 reorder 方法
    assert Product(x*y, (x, a, b), (y, c, d)).reorder((x, 1)) == \
        Product(x*y, (y, c, d), (x, a, b))
    
    # 断言：测试 Product 对象的 reorder 方法
    assert Product(x*y, (x, a, b), (y, c, d)).reorder((y, x)) == \
        Product(x*y, (y, c, d), (x, a, b))


# 定义一个测试函数，用于测试 Product 类的 is_convergent 方法
def test_Product_is_convergent():
    # 断言：测试 Product 对象的 is_convergent 方法
    assert Product(1/n**2, (n, 1, oo)).is_convergent() is S.false
    
    # 断言：测试 Product 对象的 is_convergent 方法
    assert Product(exp(1/n**2), (n, 1, oo)).is_convergent() is S.true
    
    # 断言：测试 Product 对象的 is_convergent 方法
    assert Product(1/n, (n, 1, oo)).is_convergent() is S.false
    
    # 断言：测试 Product 对象的 is_convergent 方法
    assert Product(1 + 1/n, (n, 1, oo)).is_convergent() is S.false
    
    # 断言：测试 Product 对象的 is_convergent 方法
    assert Product(1 + 1/n**2, (n, 1, oo)).is_convergent() is S.true


# 定义一个测试函数，用于测试 Product 类的 reverse_order 方法
def test_reverse_order():
    # 定义符号变量 x, y, a, b, c, d，均为整数类型
    x, y, a, b, c, d = symbols('x, y, a, b, c, d', integer=True)

    # 断言：测试 Product 对象的 reverse_order 方法
    assert Product(x, (x, 0, 3)).reverse_order(0) == Product(1/x, (x, 4, -1))
    
    # 断言：测试 Product 对象的 reverse_order 方法
    assert Product(x*y, (x, 1, 5), (y, 0, 6)).reverse_order(0, 1) == \
           Product(x*y, (x, 6, 0), (y, 7, -1))
    
    # 断言：测试 Product 对象的 reverse_order 方法
    assert Product(x, (x, 1, 2)).reverse_order(0) == Product(1/x, (x, 3, 0))
    
    # 断言：测试 Product 对象的 reverse_order 方法
    assert Product(x, (x, 1, 3)).reverse_order(0) == Product(1/x, (x, 4, 0))
    
    # 断言：测试 Product 对象的 reverse_order 方法
    assert Product(x, (x, 1, a)).reverse_order(0) == Product(1/x, (x, a + 1, 0))
    
    # 断言：测试 Product 对象的 reverse_order 方法
    assert Product(x, (x, a, 5)).reverse_order(0) == Product(1/x, (x, 6, a - 1))
    
    # 断言：测试 Product 对象的 reverse_order 方法
    assert Product(x, (x, a + 1, a + 5)).reverse_order(0) == \
           Product(1/x, (x, a + 6, a))
    
    # 断言：测试 Product 对象的 reverse_order 方法
    assert Product(x, (x, a + 1, a + 2)).reverse_order(0) == \
           Product(1/x, (x, a + 3, a))
    
    # 断言：测试 Product 对象的 reverse_order 方法
    assert Product(x, (x, a + 1, a + 1)).reverse_order(0) == \
           Product(1/x, (x, a + 2, a))
    # 断言语句：验证 Product 类的 reverse_order 方法的预期行为
    
    # 验证当 x 是 0 时，调用 reverse_order 方法的结果是否等于给定的 Product 对象
    assert Product(x, (x, a, b)).reverse_order(0) == Product(1/x, (x, b + 1, a - 1))
    
    # 验证当 x 是变量时，调用 reverse_order 方法的结果是否等于给定的 Product 对象
    assert Product(x, (x, a, b)).reverse_order(x) == Product(1/x, (x, b + 1, a - 1))
    
    # 验证在多变量的情况下，调用 reverse_order 方法的结果是否等于给定的 Product 对象
    assert Product(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1) == \
           Product(x*y, (x, b + 1, a - 1), (y, 6, 1))
    
    # 验证在多变量的情况下，调用 reverse_order 方法的结果是否等于给定的 Product 对象
    assert Product(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x) == \
           Product(x*y, (x, b + 1, a - 1), (y, 6, 1))
# 定义一个测试函数，用于测试问题编号 9983
def test_issue_9983():
    # 创建一个整数且为正的符号变量 n
    n = Symbol('n', integer=True, positive=True)
    # 创建一个乘积表达式，包含无穷迭代范围，其中每项为 1 + 1/n**Rational(2, 3)
    p = Product(1 + 1/n**Rational(2, 3), (n, 1, oo))
    # 断言该乘积表达式不收敛
    assert p.is_convergent() is S.false
    # 断言 product 函数计算的结果与 p.doit() 相等
    assert product(1 + 1/n**Rational(2, 3), (n, 1, oo)) == p.doit()


# 定义一个测试函数，用于测试问题编号 13546
def test_issue_13546():
    # 创建符号变量 n 和 k
    n = Symbol('n')
    k = Symbol('k')
    # 创建一个乘积表达式，包含从 k=0 到 n-1 的迭代范围，每项为 n + 1 / 2**k，并求值
    p = Product(n + 1 / 2**k, (k, 0, n-1)).doit()
    # 断言当 n=2 时，乘积表达式的值为 15/2
    assert p.subs(n, 2).doit() == Rational(15, 2)


# 定义一个测试函数，用于测试问题编号 14036
def test_issue_14036():
    # 创建符号变量 a 和 n
    a, n = symbols('a n')
    # 断言无穷乘积的结果不等于 0
    assert product(1 - a**2 / (n*pi)**2, [n, 1, oo]) != 0


# 定义一个测试函数，用于测试重写求和为乘积的问题
def test_rewrite_Sum():
    # 断言乘积表达式被重写为对数求和的指数形式
    assert Product(1 - S.Half**2/k**2, (k, 1, oo)).rewrite(Sum) == \
        exp(Sum(log(1 - 1/(4*k**2)), (k, 1, oo)))


# 定义一个测试函数，用于测试 KroneckerDelta 函数与乘积的组合
def test_KroneckerDelta_Product():
    # 创建符号变量 y
    y = Symbol('y')
    # 断言在 x 范围从 0 到 1 的乘积中，每个项乘以 KroneckerDelta(x, y) 的结果为 0
    assert Product(x*KroneckerDelta(x, y), (x, 0, 1)).doit() == 0


# 定义一个测试函数，用于测试问题编号 20848
def test_issue_20848():
    # 创建一个虚拟变量 _i
    _i = Dummy('i')
    # 创建符号变量 t, y, z
    t, y, z = symbols('t y z')
    # 断言在关于 x 的乘积中求导数，得到的结果是乘积的求和形式
    assert diff(Product(x, (y, 1, z)), x).as_dummy() == \
        Sum(Product(x, (y, 1, _i - 1))*Product(x, (y, _i + 1, z)), (_i, 1, z)).as_dummy()
    # 断言在关于 x 的乘积中求导数，得到的结果是 x**(z - 1) * z
    assert diff(Product(x, (y, 1, z)), x).doit() == x**(z - 1)*z
    # 断言在关于 x 的乘积中求导数，得到的结果是 Derivative(Product(x, (y, x, z)), x)
    assert diff(Product(x, (y, x, z)), x) == Derivative(Product(x, (y, x, z)), x)
    # 断言在关于 t 的乘积中求导数，得到的结果是 0
    assert diff(Product(t, (x, 1, z)), x) == S(0)
    # 断言在关于 x 的乘积中，乘以 sin(n*x) 并对 x 求导后的结果为 0
    assert Product(sin(n*x), (n, -1, 1)).diff(x).doit() == S(0)
```