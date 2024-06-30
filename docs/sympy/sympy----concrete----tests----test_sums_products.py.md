# `D:\src\scipysrc\sympy\sympy\concrete\tests\test_sums_products.py`

```
from math import prod  # 导入 math 模块中的 prod 函数

from sympy.concrete.expr_with_intlimits import ReorderError  # 导入 Sympy 中的 ReorderError 异常类
from sympy.concrete.products import (Product, product)  # 导入 Sympy 中的 Product 类和 product 函数
from sympy.concrete.summations import (Sum, summation, telescopic,  # 导入 Sympy 中的 Sum 类和相关函数
     eval_sum_residue, _dummy_with_inherited_properties_concrete)
from sympy.core.function import (Derivative, Function)  # 导入 Sympy 中的 Derivative 类和 Function 类
from sympy.core import (Catalan, EulerGamma)  # 导入 Sympy 中的 Catalan 和 EulerGamma 常数
from sympy.core.facts import InconsistentAssumptions  # 导入 Sympy 中的 InconsistentAssumptions 异常类
from sympy.core.mod import Mod  # 导入 Sympy 中的 Mod 类
from sympy.core.numbers import (E, I, Rational, nan, oo, pi)  # 导入 Sympy 中的数学常数和对象
from sympy.core.relational import Eq  # 导入 Sympy 中的 Eq 类
from sympy.core.numbers import Float  # 导入 Sympy 中的 Float 类
from sympy.core.singleton import S  # 导入 Sympy 中的 S 单例对象
from sympy.core.symbol import (Dummy, Symbol, symbols)  # 导入 Sympy 中的 Dummy, Symbol 和 symbols
from sympy.core.sympify import sympify  # 导入 Sympy 中的 sympify 函数
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)  # 导入 Sympy 中的组合数学函数
from sympy.functions.combinatorial.numbers import harmonic  # 导入 Sympy 中的 harmonic 函数
from sympy.functions.elementary.complexes import Abs  # 导入 Sympy 中的 Abs 函数
from sympy.functions.elementary.exponential import (exp, log)  # 导入 Sympy 中的指数和对数函数
from sympy.functions.elementary.hyperbolic import (sinh, tanh)  # 导入 Sympy 中的双曲函数
from sympy.functions.elementary.integers import floor  # 导入 Sympy 中的 floor 函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 Sympy 中的平方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入 Sympy 中的分段函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入 Sympy 中的三角函数
from sympy.functions.special.gamma_functions import (gamma, lowergamma)  # 导入 Sympy 中的 Gamma 函数
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入 Sympy 中的 KroneckerDelta 函数
from sympy.functions.special.zeta_functions import zeta  # 导入 Sympy 中的 Zeta 函数
from sympy.integrals.integrals import Integral  # 导入 Sympy 中的 Integral 类
from sympy.logic.boolalg import And, Or  # 导入 Sympy 中的逻辑运算符 And 和 Or
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入 Sympy 中的 MatrixSymbol 类
from sympy.matrices.expressions.special import Identity  # 导入 Sympy 中的 Identity 特殊矩阵
from sympy.matrices import (Matrix, SparseMatrix,  # 导入 Sympy 中的矩阵类和相关函数
    ImmutableDenseMatrix, ImmutableSparseMatrix, diag)
from sympy.sets.fancysets import Range  # 导入 Sympy 中的 Range 集合
from sympy.sets.sets import Interval  # 导入 Sympy 中的区间 Interval 类
from sympy.simplify.combsimp import combsimp  # 导入 Sympy 中的 combsimp 函数
from sympy.simplify.simplify import simplify  # 导入 Sympy 中的 simplify 函数
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)  # 导入 Sympy 中的张量索引相关类
from sympy.testing.pytest import XFAIL, raises, slow  # 导入 Sympy 测试相关的装饰器
from sympy.abc import a, b, c, d, k, m, x, y, z  # 导入 Sympy 的符号变量

n = Symbol('n', integer=True)  # 定义一个名为 n 的符号变量，限定其为整数
f, g = symbols('f g', cls=Function)  # 定义 f 和 g 为函数符号变量

def test_karr_convention():
    # 测试 Karr 的求和约定是否满足
    # 参见他的论文 "Summation in Finite Terms" 中详细的推理和理由
    # 该约定在第309页以及1.4节中定义如下:
    #
    # \sum_{m <= i < n} f(i) 'has the obvious meaning'   for m < n
    # \sum_{m <= i < n} f(i) = 0                         for m = n
    # \sum_{m <= i < n} f(i) = - \sum_{n <= i < m} f(i)  for m > n
    #
    # 需要注意的是，他定义的所有求和的上限都是*排除的*。
    # 相比之下，SymPy 和通常的数学表示法有:
    #
    # sum_{i = a}^b f(i) = f(a) + f(a+1) + ... + f(b-1) + f(b)
    #
    # 其中上限是*包含的*。因此，在 SymPy 中转换求和时需要注意这一点。
    # 定义整数符号变量 i, k, j
    i = Symbol("i", integer=True)
    k = Symbol("k", integer=True)
    j = Symbol("j", integer=True)

    # 对具体的求和公式进行示例，以及使用符号化的上下限

    # 正常求和：m = k，n = k + j，因此 m < n：
    m = k
    n = k + j

    a = m
    b = n - 1
    # 计算求和 Sum(i**2, (i, a, b)) 并求其值
    S1 = Sum(i**2, (i, a, b)).doit()

    # 反向求和：m = k + j，n = k，因此 m > n：
    m = k + j
    n = k

    a = m
    b = n - 1
    # 计算求和 Sum(i**2, (i, a, b)) 并求其值
    S2 = Sum(i**2, (i, a, b)).doit()

    # 验证 S1 + S2 的简化结果为 0
    assert simplify(S1 + S2) == 0

    # 测试空求和：m = k，n = k，因此 m = n：
    m = k
    n = k

    a = m
    b = n - 1
    # 计算求和 Sum(i**2, (i, a, b)) 并求其值
    Sz = Sum(i**2, (i, a, b)).doit()

    # 验证 Sz 的值为 0
    assert Sz == 0

    # 另一个示例，这次使用未指定的求和项和数值上下限。（我们不能在同一个示例中同时进行这两个测试。）

    # 正常求和，其中 m < n：
    m = 2
    n = 11

    a = m
    b = n - 1
    # 计算求和 Sum(f(i), (i, a, b)) 并求其值
    S1 = Sum(f(i), (i, a, b)).doit()

    # 反向求和，其中 m > n：
    m = 11
    n = 2

    a = m
    b = n - 1
    # 计算求和 Sum(f(i), (i, a, b)) 并求其值
    S2 = Sum(f(i), (i, a, b)).doit()

    # 验证 S1 + S2 的简化结果为 0
    assert simplify(S1 + S2) == 0

    # 测试空求和，其中 m = n：
    m = 5
    n = 5

    a = m
    b = n - 1
    # 计算求和 Sum(f(i), (i, a, b)) 并求其值
    Sz = Sum(f(i), (i, a, b)).doit()

    # 验证 Sz 的值为 0
    assert Sz == 0

    # 定义一个分段函数 e，当 Mod(i, 2) > 0 时为 exp(-i)，否则为 0
    e = Piecewise((exp(-i), Mod(i, 2) > 0), (0, True))
    # 对分段函数 e 进行求和，上下限为 (i, 0, 11)
    s = Sum(e, (i, 0, 11))
    # 验证数值结果保留 3 位小数时与精确求值的结果相等
    assert s.n(3) == s.doit().n(3)
def test_karr_proposition_2a():
    # Test Karr, page 309, proposition 2, part a
    # 定义整数符号变量 i, u, v
    i = Symbol("i", integer=True)
    u = Symbol("u", integer=True)
    v = Symbol("v", integer=True)

    # 定义内部函数 test_the_sum，用于测试和验证数学命题
    def test_the_sum(m, n):
        # 计算函数 g(i) = i^3 + 2*i^2 - 3*i
        g = i**3 + 2*i**2 - 3*i
        # 计算 g 的差分 f = Delta g
        f = simplify(g.subs(i, i+1) - g)
        # 计算和 S = Sum_{i=m}^{n-1} f(i)
        a = m
        b = n - 1
        S = Sum(f, (i, a, b)).doit()
        # 验证数学命题 Sum_{m <= i < n} f(i) = g(n) - g(m)
        assert simplify(S - (g.subs(i, n) - g.subs(i, m))) == 0

    # 测试情况：m < n
    test_the_sum(u,   u+v)
    # 测试情况：m = n
    test_the_sum(u,   u  )
    # 测试情况：m > n
    test_the_sum(u+v, u  )


def test_karr_proposition_2b():
    # Test Karr, page 309, proposition 2, part b
    # 定义整数符号变量 i, u, v, w
    i = Symbol("i", integer=True)
    u = Symbol("u", integer=True)
    v = Symbol("v", integer=True)
    w = Symbol("w", integer=True)

    # 定义内部函数 test_the_sum，用于测试和验证数学命题
    def test_the_sum(l, n, m):
        # 计算求和项 s = i^3
        s = i**3
        # 第一个求和 S1 = Sum_{i=l}^{n-1} s
        a = l
        b = n - 1
        S1 = Sum(s, (i, a, b)).doit()
        # 第二个求和 S2 = Sum_{i=l}^{m-1} s
        a = l
        b = m - 1
        S2 = Sum(s, (i, a, b)).doit()
        # 第三个求和 S3 = Sum_{i=m}^{n-1} s
        a = m
        b = n - 1
        S3 = Sum(s, (i, a, b)).doit()
        # 验证数学命题 S1 = S2 + S3
        assert S1 - (S2 + S3) == 0

    # 测试情况：l < m < n
    test_the_sum(u,     u+v,   u+v+w)
    # 测试情况：l < m = n
    test_the_sum(u,     u+v,   u+v  )
    # 测试情况：l < m > n
    test_the_sum(u,     u+v+w, v    )
    # 测试情况：l = m < n
    test_the_sum(u,     u,     u+v  )
    # 测试情况：l = m = n
    test_the_sum(u,     u,     u    )
    # 测试情况：l = m > n
    test_the_sum(u+v,   u+v,   u    )
    # 测试情况：l > m < n
    test_the_sum(u+v,   u,     u+w  )
    # 测试情况：l > m = n
    test_the_sum(u+v,   u,     u    )
    # 测试情况：l > m > n
    test_the_sum(u+v+w, u+v,   u    )


def test_arithmetic_sums():
    # 测试数学求和函数的各种情况

    # 验证：Sum_{n=a}^{b} 1 = b - a + 1
    assert summation(1, (n, a, b)) == b - a + 1

    # 验证：Sum_{n=a}^{b} NaN = NaN
    assert Sum(S.NaN, (n, a, b)) is S.NaN

    # 验证：Sum_{n=a}^{a} x = x
    assert Sum(x, (n, a, a)).doit() == x

    # 验证：Sum_{x=a}^{a} x = a
    assert Sum(x, (x, a, a)).doit() == a

    # 验证：Sum_{n=1}^{a} x = a*x
    assert Sum(x, (n, 1, a)).doit() == a*x

    # 验证：Sum_{x=1}^{10} x = 55
    assert Sum(x, (x, Range(1, 11))).doit() == 55

    # 验证：Sum_{x=1}^{10, step=2} x = 25
    assert Sum(x, (x, Range(1, 11, 2))).doit() == 25

    # 验证：Sum_{x=1}^{10, step=2} x = Sum_{x=9}^{1, step=-2} x
    assert Sum(x, (x, Range(1, 10, 2))) == Sum(x, (x, Range(9, 0, -2)))

    # 定义变量 lo, hi
    lo, hi = 1, 2

    # 验证：Sum_{n=lo}^{hi} n != Sum_{n=hi}^{lo} n
    s1 = Sum(n, (n, lo, hi))
    s2 = Sum(n, (n, hi, lo))
    assert s1 != s2

    # 验证：Sum_{n=lo}^{hi} n = 3, Sum_{n=hi}^{lo} n = 0
    assert s1.doit() == 3 and s2.doit() == 0

    # 定义变量 lo, hi
    lo, hi = x, x + 1

    # 验证：Sum_{n=lo}^{hi} n != Sum_{n=hi}^{lo} n
    s1 = Sum(n, (n, lo, hi))
    s2 = Sum(n, (n, hi, lo))
    assert s1 != s2

    # 验证：Sum_{n=lo}^{hi} n = 2*x + 1, Sum_{n=hi}^{lo} n = 0
    assert s1.doit() == 2*x + 1 and s2.doit() == 0

    # 验证：Sum_{x=1}^{2} (Integral(x, (x, 1, y)) + x) = y^2 + 2
    assert Sum(Integral(x, (x, 1, y)) + x, (x, 1, 2)).doit() == y**2 + 2

    # 验证：Sum_{n=1}^{10} 1 = 10
    assert summation(1, (n, 1, 10)) == 10

    # 验证：Sum_{n=0}^{10^10} 2*n = 100000000010000000000
    assert summation(2*n, (n, 0, 10**10)) == 100000000010000000000

    # 验证：Sum_{n=a}^{1} Sum_{m=1}^{d} 4*n*m = 2*d + 2*d**2 + a*d + a*d**2 - d*a**2 - a**2*d**2
    assert summation(4*n*m, (n, a, 1), (m, 1, d)).expand() == \
        2*d + 2*d**2 + a*d + a*d**2 - d*a**2 - a**2*d**2

    # 验证：Sum_{n=-2}^{1} cos(n) = cos(-2) + cos(-1) + cos(0) + cos(1)
    assert summation(cos(n), (n, -2, 1)) == cos(-2) + cos(-1) + cos(0) + cos(1)

    # 验证：Sum_{n=x}^{x+2} cos(n) = cos(x) + cos(x+1) + cos(x+2)
    assert summation(cos(n), (n, x, x + 2)) == cos(x) + cos(x + 1) + cos(x + 2)
    # 确保 sum 函数返回的结果是 Sum 类型的对象
    assert isinstance(summation(cos(n), (n, x, x + S.Half)), Sum)
    
    # 确保对 k 从 0 到无穷大的求和结果是无穷大 (∞)
    assert summation(k, (k, 0, oo)) is oo
    
    # 确保对 k 从 1 到 11 的求和结果等于 55
    assert summation(k, (k, Range(1, 11))) == 55
# 定义测试函数 test_polynomial_sums，用于测试 summation 函数的多项式求和功能
def test_polynomial_sums():
    # 断言：求解 n^2 在 n 从 3 到 8 的和应为 199
    assert summation(n**2, (n, 3, 8)) == 199
    # 断言：求解 n 在 n 从 a 到 b 的和应为 ((a + b)*(b - a + 1)/2).expand()
    assert summation(n, (n, a, b)) == \
        ((a + b)*(b - a + 1)/2).expand()
    # 断言：求解 n^2 在 n 从 1 到 b 的和应为 ((2*b**3 + 3*b**2 + b)/6).expand()
    assert summation(n**2, (n, 1, b)) == \
        ((2*b**3 + 3*b**2 + b)/6).expand()
    # 断言：求解 n^3 在 n 从 1 到 b 的和应为 ((b**4 + 2*b**3 + b**2)/4).expand()
    assert summation(n**3, (n, 1, b)) == \
        ((b**4 + 2*b**3 + b**2)/4).expand()
    # 断言：求解 n^6 在 n 从 1 到 b 的和应为 ((6*b**7 + 21*b**6 + 21*b**5 - 7*b**3 + b)/42).expand()
    assert summation(n**6, (n, 1, b)) == \
        ((6*b**7 + 21*b**6 + 21*b**5 - 7*b**3 + b)/42).expand()


# 定义测试函数 test_geometric_sums，用于测试 summation 函数的几何级数求和功能
def test_geometric_sums():
    # 断言：求解 pi^n 在 n 从 0 到 b 的和应为 (1 - pi**(b + 1)) / (1 - pi)
    assert summation(pi**n, (n, 0, b)) == (1 - pi**(b + 1)) / (1 - pi)
    # 断言：求解 2 * 3^n 在 n 从 0 到 b 的和应为 3**(b + 1) - 1
    assert summation(2 * 3**n, (n, 0, b)) == 3**(b + 1) - 1
    # 断言：求解 (1/2)^n 在 n 从 1 到 oo 的和应为 1
    assert summation(S.Half**n, (n, 1, oo)) == 1
    # 断言：求解 2^n 在 n 从 0 到 b 的和应为 2**(b + 1) - 1
    assert summation(2**n, (n, 0, b)) == 2**(b + 1) - 1
    # 断言：求解 2^n 在 n 从 1 到 oo 的和应为 oo
    assert summation(2**n, (n, 1, oo)) is oo
    # 断言：求解 2^(-n) 在 n 从 1 到 oo 的和应为 1
    assert summation(2**(-n), (n, 1, oo)) == 1
    # 断言：求解 3^(-n) 在 n 从 4 到 oo 的和应为 1/54
    assert summation(3**(-n), (n, 4, oo)) == Rational(1, 54)
    # 断言：求解 2^(-4*n + 3) 在 n 从 1 到 oo 的和应为 8/15
    assert summation(2**(-4*n + 3), (n, 1, oo)) == Rational(8, 15)
    # 断言：求解 2^(n + 1) 在 n 从 1 到 b 的和展开后应为 4*(2**b - 1)
    assert summation(2**(n + 1), (n, 1, b)).expand() == 4*(2**b - 1)

    # issue 6664:
    # 断言：求解 x^n 在 n 从 0 到 oo 的和应为 Piecewise((1/(-x + 1), Abs(x) < 1), (Sum(x**n, (n, 0, oo)), True))
    assert summation(x**n, (n, 0, oo)) == \
        Piecewise((1/(-x + 1), Abs(x) < 1), (Sum(x**n, (n, 0, oo)), True))

    # 断言：求解 (-2)^n 在 n 从 0 到 oo 的和应为 -oo
    assert summation(-2**n, (n, 0, oo)) is -oo
    # 断言：求解 I^n 在 n 从 0 到 oo 的和应为 Sum(I**n, (n, 0, oo))
    assert summation(I**n, (n, 0, oo)) == Sum(I**n, (n, 0, oo))

    # issue 6802:
    # 断言：求解 (-1)^(2*x + 2) 在 x 从 0 到 n 的和应为 n + 1
    assert summation((-1)**(2*x + 2), (x, 0, n)) == n + 1
    # 断言：求解 (-2)^(2*x + 2) 在 x 从 0 到 n 的和应为 4*4**(n + 1)/3 - 4/3
    assert summation((-2)**(2*x + 2), (x, 0, n)) == 4*4**(n + 1)/S(3) - Rational(4, 3)
    # 断言：求解 (-1)^x 在 x 从 0 到 n 的和应为 -(-1)^(n + 1)/2 + 1/2
    assert summation((-1)**x, (x, 0, n)) == -(-1)**(n + 1)/S(2) + S.Half
    # 断言：求解 y^x 在 x 从 a 到 b 的和应为 Piecewise((-a + b + 1, Eq(y, 1)), ((y**a - y**(b + 1))/(-y + 1), True))
    assert summation(y**x, (x, a, b)) == \
        Piecewise((-a + b + 1, Eq(y, 1)), ((y**a - y**(b + 1))/(-y + 1), True))
    # 断言：求解 (-2)^(y*x + 2) 在 x 从 0 到 n 的和应为 4*Piecewise((n + 1, Eq((-2)**y, 1)), ((-(-2)**(y*(n + 1)) + 1)/(-(-2)**y + 1), True))
    assert summation((-2)**(y*x + 2), (x, 0, n)) == \
        4*Piecewise((n + 1, Eq((-2)**y, 1)),
                    ((-(-2)**(y*(n + 1)) + 1)/(-(-2)**y + 1), True))

    # issue 8251:
    # 断言：求解 (1/(n + 1)^2)*n^2 在 n 从 0 到 oo 的和应为 oo
    assert summation((1/(n + 1)**2)*n**2, (n, 0, oo)) is oo

    # issue 9908:
    # 断言：求解 1/(n^3 - 1) 在 n 从 -oo 到 -2 的和应等于 summation(1/(n^3 - 1), (n, -oo, -2))
    assert Sum(1/(n**3 - 1), (n, -oo, -2)).doit() == summation(1/(n**3 - 1), (n, -oo, -2))

    # issue 11642:
    # 求解 0.5^n 在 n 从 1 到 oo 的和，并验证结果为 1.0
    result = Sum(0.5**n, (n, 1, oo)).doit()
    assert result == 1.0
    # 断言：验证结果为浮点数
    assert result.is_Float

    # 求解 0.25^n 在 n 从 1 到 oo 的和，并验证结果
    # 断言：对于从 k=0 到 n 的 1/k 的求和结果应该等于符号表达式 Sum(1/k, (k, 0, n))
    assert summation(1/k, (k, 0, n)) == Sum(1/k, (k, 0, n))
    
    # 断言：对于从 k=1 到 n 的 1/k 的求和结果应该等于 harmonic(n)，即调和数的值
    assert summation(1/k, (k, 1, n)) == harmonic(n)
    
    # 断言：对于从 k=1 到 n 的 n/k 的求和结果应该等于 n 乘以 harmonic(n)
    assert summation(n/k, (k, 1, n)) == n * harmonic(n)
    
    # 断言：对于从 k=5 到 n 的 1/k 的求和结果应该等于 harmonic(n) 减去 harmonic(4)
    assert summation(1/k, (k, 5, n)) == harmonic(n) - harmonic(4)
def test_composite_sums():
    # 定义复合求和表达式 f
    f = S.Half*(7 - 6*n + Rational(1, 7)*n**3)
    # 对 f 进行从 a 到 b 的求和
    s = summation(f, (n, a, b))
    # 断言求和结果不是 Sum 对象
    assert not isinstance(s, Sum)
    # 初始化变量 A 为 0
    A = 0
    # 遍历范围 -3 到 4 的整数
    for i in range(-3, 5):
        # 将 f 在 n=i 处的值累加到 A
        A += f.subs(n, i)
    # 计算在 a=-3, b=4 处的求和结果，并赋给变量 B
    B = s.subs(a, -3).subs(b, 4)
    # 断言 A 等于 B
    assert A == B


def test_hypergeometric_sums():
    # 断言二项式求和 binomial(2*k, k)/4**k 在 k 从 0 到 n 的范围内等于给定表达式
    assert summation(binomial(2*k, k)/4**k, (k, 0, n)) == (1 + 2*n)*binomial(2*n, n)/4**n
    # 断言二项式求和 binomial(2*k, k)/5**k 在 k 从负无穷到正无穷的范围内等于给定表达式
    assert summation(binomial(2*k, k)/5**k, (k, -oo, oo)) == sqrt(5)


def test_other_sums():
    # 定义函数 f
    f = m**2 + m*exp(m)
    # 定义函数 g
    g = 3*exp(Rational(3, 2))/2 + exp(S.Half)/2 - exp(Rational(-1, 2))/2 - 3*exp(Rational(-3, 2))/2 + 5

    # 断言在 m 从 Rational(-3, 2) 到 Rational(3, 2) 的范围内对 f 求和结果等于 g
    assert summation(f, (m, Rational(-3, 2), Rational(3, 2))) == g
    # 断言在 m 从 -1.5 到 1.5 的范围内对 f 求和结果的数值近似等于 g 的数值近似，精度为 1e-10
    assert summation(f, (m, -1.5, 1.5)).evalf().epsilon_eq(g.evalf(), 1e-10)


fac = factorial


def NS(e, n=15, **options):
    # 返回表达式 e 的数值字符串表示，精度为 n
    return str(sympify(e).evalf(n, **options))


def test_evalf_fast_series():
    # 断言欧拉变换的级数求和结果与 sqrt(2) 的数值字符串表示相等，精度为 100
    assert NS(Sum(
        fac(2*n + 1)/fac(n)**2/2**(3*n + 1), (n, 0, oo)), 100) == NS(sqrt(2), 100)

    # 断言几个级数求和结果与数学常数的数值字符串表示相等，精度为 100
    estr = NS(E, 100)
    assert NS(Sum(1/fac(n), (n, 0, oo)), 100) == estr
    assert NS(1/Sum((1 - 2*n)/fac(2*n), (n, 0, oo)), 100) == estr
    assert NS(Sum((2*n + 1)/fac(2*n), (n, 0, oo)), 100) == estr
    assert NS(Sum((4*n + 3)/2**(2*n + 1)/fac(2*n + 1), (n, 0, oo))**2, 100) == estr

    pistr = NS(pi, 100)
    # 断言拉马努金级数求和结果与 pi 的数值字符串表示相等，精度为 100
    assert NS(9801/sqrt(8)/Sum(fac(
        4*n)*(1103 + 26390*n)/fac(n)**4/396**(4*n), (n, 0, oo)), 100) == pistr
    assert NS(1/Sum(
        binomial(2*n, n)**3 * (42*n + 5)/2**(12*n + 4), (n, 0, oo)), 100) == pistr
    # 断言马钦公式求和结果与 pi 的数值字符串表示相等，精度为 100
    assert NS(16*Sum((-1)**n/(2*n + 1)/5**(2*n + 1), (n, 0, oo)) -
        4*Sum((-1)**n/(2*n + 1)/239**(2*n + 1), (n, 0, oo)), 100) == pistr

    # 断言阿波利亚常数的求和结果与其数值字符串表示相等，精度为 100
    astr = NS(zeta(3), 100)
    P = 126392*n**5 + 412708*n**4 + 531578*n**3 + 336367*n**2 + 104000* \
        n + 12463
    assert NS(Sum((-1)**n * P / 24 * (fac(2*n + 1)*fac(2*n)*fac(
        n))**3 / fac(3*n + 2) / fac(4*n + 3)**3, (n, 0, oo)), 100) == astr
    assert NS(Sum((-1)**n * (205*n**2 + 250*n + 77)/64 * fac(n)**10 /
              fac(2*n + 1)**5, (n, 0, oo)), 100) == astr


def test_evalf_fast_series_issue_4021():
    # 断言卡塔兰常数的求和结果与其数值字符串表示相等，精度为 100
    assert NS(Sum((-1)**(n - 1)*2**(8*n)*(40*n**2 - 24*n + 3)*fac(2*n)**3*
        fac(n)**2/n**3/(2*n - 1)/fac(4*n)**2, (n, 1, oo))/64, 100) == \
        NS(Catalan, 100)
    astr = NS(zeta(3), 100)
    assert NS(5*Sum(
        (-1)**(n - 1)*fac(n)**2 / n**3 / fac(2*n), (n, 1, oo))/2, 100) == astr
    assert NS(Sum((-1)**(n - 1)*(56*n**2 - 32*n + 5) / (2*n - 1)**2 * fac(n - 1)
              **3 / fac(3*n), (n, 1, oo))/4, 100) == astr


def test_evalf_slow_series():
    # 断言级数求和 (-1)**n / n 的数值字符串表示精确到小数点后 15 位与 -log(2) 的数值字符串表示相等
    assert NS(Sum((-1)**n / n, (n, 1, oo)), 15) == NS(-log(2), 15)
    # 断言级数求和 (-1)**n / n 的数值字符串表示精确到小数点后 50 位与 -log(2) 的数值字符串表示相等
    assert NS(Sum((-1)**n / n, (n, 1, oo)), 50) == NS(-log(2), 50)
    # 断言级数求和 1 / n**2 的数值字符串表示精确到小数点后 15 位与 pi**2/6 的数值字符串表示相等
    assert NS(Sum(1/n**2, (n, 1, oo)), 15) == NS(pi**2/6, 15)
    # 断言：计算级数 1/n^2 的和，直到无穷大，并将结果保留 100 位小数，应该等于 pi^2/6 的值，也保留 100 位小数
    assert NS(Sum(1/n**2, (n, 1, oo)), 100) == NS(pi**2/6, 100)
    
    # 断言：计算级数 1/n^2 的和，直到无穷大，并将结果保留 500 位小数，应该等于 pi^2/6 的值，也保留 500 位小数
    assert NS(Sum(1/n**2, (n, 1, oo)), 500) == NS(pi**2/6, 500)
    
    # 断言：计算级数 (-1)^n / (2*n + 1)^3 的和，直到无穷大，并将结果保留 15 位小数，应该等于 pi^3/32 的值，也保留 15 位小数
    assert NS(Sum((-1)**n / (2*n + 1)**3, (n, 0, oo)), 15) == NS(pi**3/32, 15)
    
    # 断言：计算级数 (-1)^n / (2*n + 1)^3 的和，直到无穷大，并将结果保留 50 位小数，应该等于 pi^3/32 的值，也保留 50 位小数
    assert NS(Sum((-1)**n / (2*n + 1)**3, (n, 0, oo)), 50) == NS(pi**3/32, 50)
def test_evalf_oo_to_oo():
    # 这里曾经在某些情况下出现错误
    # 不进行数值计算，但至少不会抛出错误
    # 符号计算结果应为 0，这是不正确的
    assert Sum(1/(n**2+1), (n, -oo, oo)).evalf() == Sum(1/(n**2+1), (n, -oo, oo))
    # 对于从 1 到 oo 的符号计算
    assert Sum(1/(factorial(abs(n))), (n, -oo, -1)).evalf() == Sum(1/(factorial(abs(n))), (n, -oo, -1))


def test_euler_maclaurin():
    # 使用 E-M 精确计算多项式求和
    def check_exact(f, a, b, m, n):
        A = Sum(f, (k, a, b))
        s, e = A.euler_maclaurin(m, n)
        assert (e == 0) and (s.expand() == A.doit())
    check_exact(k**4, a, b, 0, 2)
    check_exact(k**4 + 2*k, a, b, 1, 2)
    check_exact(k**4 + k**2, a, b, 1, 5)
    check_exact(k**5, 2, 6, 1, 2)
    check_exact(k**5, 2, 6, 1, 3)
    assert Sum(x-1, (x, 0, 2)).euler_maclaurin(m=30, n=30, eps=2**-15) == (0, 0)
    # 非精确结果
    assert Sum(k**6, (k, a, b)).euler_maclaurin(0, 2)[1] != 0
    # 数值测试
    for mi, ni in [(2, 4), (2, 20), (10, 20), (18, 20)]:
        A = Sum(1/k**3, (k, 1, oo))
        s, e = A.euler_maclaurin(mi, ni)
        assert abs((s - zeta(3)).evalf()) < e.evalf()

    raises(ValueError, lambda: Sum(1, (x, 0, 1), (k, 0, 1)).euler_maclaurin())


@slow
def test_evalf_euler_maclaurin():
    assert NS(Sum(1/k**k, (k, 1, oo)), 15) == '1.29128599706266'
    assert NS(Sum(1/k**k, (k, 1, oo)),
              50) == '1.2912859970626635404072825905956005414986193682745'
    assert NS(Sum(1/k - log(1 + 1/k), (k, 1, oo)), 15) == NS(EulerGamma, 15)
    assert NS(Sum(1/k - log(1 + 1/k), (k, 1, oo)), 50) == NS(EulerGamma, 50)
    assert NS(Sum(log(k)/k**2, (k, 1, oo)), 15) == '0.937548254315844'
    assert NS(Sum(log(k)/k**2, (k, 1, oo)),
              50) == '0.93754825431584375370257409456786497789786028861483'
    assert NS(Sum(1/k, (k, 1000000, 2000000)), 15) == '0.693147930560008'
    assert NS(Sum(1/k, (k, 1000000, 2000000)),
              50) == '0.69314793056000780941723211364567656807940638436025'


def test_evalf_symbolic():
    # issue 6328
    expr = Sum(f(x), (x, 1, 3)) + Sum(g(x), (x, 1, 3))
    assert expr.evalf() == expr


def test_evalf_issue_3273():
    assert Sum(0, (k, 1, oo)).evalf() == 0


def test_simple_products():
    assert Product(S.NaN, (x, 1, 3)) is S.NaN
    assert product(S.NaN, (x, 1, 3)) is S.NaN
    assert Product(x, (n, a, a)).doit() == x
    assert Product(x, (x, a, a)).doit() == a
    assert Product(x, (y, 1, a)).doit() == x**a

    lo, hi = 1, 2
    s1 = Product(n, (n, lo, hi))
    s2 = Product(n, (n, hi, lo))
    assert s1 != s2
    # 根据 Karr 乘积约定，这是正确的
    assert s1.doit() == 2
    assert s2.doit() == 1

    lo, hi = x, x + 1
    s1 = Product(n, (n, lo, hi))
    s2 = Product(n, (n, hi, lo))
    s3 = 1 / Product(n, (n, hi + 1, lo - 1))
    assert s1 != s2
    # 根据 Karr 乘积约定，这是正确的
    assert s1.doit() == x*(x + 1)
    # 断言 s2 对象的计算结果应该等于 1
    assert s2.doit() == 1
    # 断言 s3 对象的计算结果应该等于 x*(x + 1)
    assert s3.doit() == x*(x + 1)

    # 断言积分对象和常数项的乘积的计算结果应该等于 (y**2 + 1)*(y**2 + 3)
    assert Product(Integral(2*x, (x, 1, y)) + 2*x, (x, 1, 2)).doit() == \
        (y**2 + 1)*(y**2 + 3)
    
    # 断言以指定范围内的数的乘积的计算结果应该等于 2**(b - a + 1)
    assert product(2, (n, a, b)) == 2**(b - a + 1)
    
    # 断言以指定范围内的数的乘积的计算结果应该等于 factorial(b)
    assert product(n, (n, 1, b)) == factorial(b)
    
    # 断言以指定范围内的数的乘积的计算结果应该等于 factorial(b)**3
    assert product(n**3, (n, 1, b)) == factorial(b)**3
    
    # 断言以指定范围内的数的乘积的计算结果应该等于 3**(2*(1 - a + b) + b/2 + (b**2)/2 + a/2 - (a**2)/2)
    assert product(3**(2 + n), (n, a, b)) \
        == 3**(2*(1 - a + b) + b/2 + (b**2)/2 + a/2 - (a**2)/2)
    
    # 断言余弦函数在指定范围内的乘积的计算结果应该等于 cos(3)*cos(4)*cos(5)
    assert product(cos(n), (n, 3, 5)) == cos(3)*cos(4)*cos(5)
    
    # 断言余弦函数在指定范围内的乘积的计算结果应该等于 cos(x)*cos(x + 1)*cos(x + 2)
    assert product(cos(n), (n, x, x + 2)) == cos(x)*cos(x + 1)*cos(x + 2)
    
    # 断言余弦函数在指定范围内的乘积的类型应该是 Product 类型
    assert isinstance(product(cos(n), (n, x, x + S.Half)), Product)
    
    # 断言以指定范围内的数的乘积的类型应该是 Product 类型，但是由于 Product 通常无法正确计算此例子，因此断言失败
    # 如果 Product 成功计算此例子，那么它很可能是错误的！
    assert isinstance(Product(n**n, (n, 1, b)), Product)
# 定义测试函数，用于测试有理数乘积
def test_rational_products():
    # 断言：对于表达式 product(1 + 1/n, (n, a, b)) 简化后应该等于 (1 + b)/a
    assert combsimp(product(1 + 1/n, (n, a, b))) == (1 + b)/a
    # 断言：对于表达式 product(n + 1, (n, a, b)) 简化后应该等于 gamma(2 + b)/gamma(1 + a)
    assert combsimp(product(n + 1, (n, a, b))) == gamma(2 + b)/gamma(1 + a)
    # 断言：对于表达式 product((n + 1)/(n - 1), (n, a, b)) 简化后应该等于 b*(1 + b)/(a*(a - 1))
    assert combsimp(product((n + 1)/(n - 1), (n, a, b))) == b*(1 + b)/(a*(a - 1))
    # 断言：对于表达式 product(n/(n + 1)/(n + 2), (n, a, b)) 简化后应该等于 a*gamma(a + 2)/(b + 1)/gamma(b + 3)
    assert combsimp(product(n/(n + 1)/(n + 2), (n, a, b))) == a*gamma(a + 2)/(b + 1)/gamma(b + 3)
    # 断言：对于表达式 product(n*(n + 1)/(n - 1)/(n - 2), (n, a, b)) 简化后应该等于 b**2*(b - 1)*(1 + b)/(a - 1)**2/(a*(a - 2))
    assert combsimp(product(n*(n + 1)/(n - 1)/(n - 2), (n, a, b))) == b**2*(b - 1)*(1 + b)/(a - 1)**2/(a*(a - 2))


# 定义测试函数，用于测试 Wallis 乘积
def test_wallis_product():
    # Wallis 乘积，以两种不同形式给出以确保 Product 能够因式化简单的有理表达式
    A = Product(4*n**2 / (4*n**2 - 1), (n, 1, b))
    B = Product((2*n)*(2*n)/(2*n - 1)/(2*n + 1), (n, 1, b))
    R = pi*gamma(b + 1)**2/(2*gamma(b + S.Half)*gamma(b + Rational(3, 2)))
    assert simplify(A.doit()) == R
    assert simplify(B.doit()) == R
    # 这个应该最终也能做到（sin 的 Euler 乘积公式）
    # assert Product(1+x/n**2, (n, 1, b)) == ...


# 定义测试函数，用于测试 Telescopic 和 Sums
def test_telescopic_sums():
    # 检查输入 2 的评论问题 4127
    assert Sum(1/k - 1/(k + 1), (k, 1, n)).doit() == 1 - 1/(1 + n)
    assert Sum(
        f(k) - f(k + 2), (k, m, n)).doit() == -f(1 + n) - f(2 + n) + f(m) + f(1 + m)
    assert Sum(cos(k) - cos(k + 3), (k, 1, n)).doit() == -cos(1 + n) - \
        cos(2 + n) - cos(3 + n) + cos(1) + cos(2) + cos(3)

    # 虚拟变量不应该影响结果
    assert telescopic(1/m, -m/(1 + m), (m, n - 1, n)) == \
        telescopic(1/k, -k/(1 + k), (k, n - 1, n))

    assert Sum(1/x/(x - 1), (x, a, b)).doit() == 1/(a - 1) - 1/b
    eq = 1/((5*n + 2)*(5*(n + 1) + 2))
    assert Sum(eq, (n, 0, oo)).doit() == S(1)/10
    nz = symbols('nz', nonzero=True)
    v = Sum(eq.subs(5, nz), (n, 0, oo)).doit()
    assert v.subs(nz, 5).simplify() == S(1)/10
    # 检查非符号情况下是否使用 apart
    s = Sum(eq, (n, 0, k)).doit()
    v = Sum(eq, (n, 0, 10**100)).doit()
    assert v == s.subs(k, 10**100)


# 定义测试函数，用于测试 Sum 重构
def test_sum_reconstruct():
    s = Sum(n**2, (n, -1, 1))
    assert s == Sum(*s.args)
    raises(ValueError, lambda: Sum(x, x))
    raises(ValueError, lambda: Sum(x, (x, 1, 2)))


# 定义测试函数，用于测试 limit subs
def test_limit_subs():
    for F in (Sum, Product, Integral):
        assert F(a*exp(a), (a, -2, 2)) == F(a*exp(a), (a, -b, b)).subs(b, 2)
        assert F(a, (a, F(b, (b, 1, 2)), 4)).subs(F(b, (b, 1, 2)), c) == \
            F(a, (a, c, 4))
        assert F(x, (x, 1, x + y)).subs(x, 1) == F(x, (x, 1, y + 1))


# 定义测试函数，用于测试 function subs
def test_function_subs():
    S = Sum(x*f(y),(x,0,oo),(y,0,oo))
    assert S.subs(f(y),y) == Sum(x*y,(x,0,oo),(y,0,oo))
    assert S.subs(f(x),x) == S
    raises(ValueError, lambda: S.subs(f(y),x+y) )
    S = Sum(x*log(y),(x,0,oo),(y,0,oo))
    assert S.subs(log(y),y) == S
    S = Sum(x*f(y),(x,0,oo),(y,0,oo))
    assert S.subs(f(y),y) == Sum(x*y,(x,0,oo),(y,0,oo))


# 定义测试函数，用于测试等式
def test_equality():
    # 如果失败则删除下面的特殊处理
    raises(ValueError, lambda: Sum(x, x))
    # 定义符号 x，限定为实数
    r = symbols('x', real=True)
    # 对于每个类 Sum、Product、Integral 进行迭代
    for F in (Sum, Product, Integral):
        try:
            # 断言不同的参数情况下不相等
            assert F(x, x) != F(y, y)
            assert F(x, (x, 1, 2)) != F(x, x)
            assert F(x, (x, x)) != F(x, x)  # 否则它们会打印相同的内容
            assert F(1, x) != F(1, y)
        except ValueError:
            # 如果抛出 ValueError 异常则跳过
            pass
        # 断言不同的极限值下不相等
        assert F(a, (x, 1, 2)) != F(a, (x, 1, 3))
        # 断言不同的变量名下不相等
        assert F(a, (x, 1, x)) != F(a, (y, 1, y))
        # 断言不同的表达式下不相等
        assert F(a, (x, 1, 2)) != F(b, (x, 1, 2))
        # 断言不同的假设下不相等
        assert F(x, (x, 1, 2)) != F(r, (r, 1, 2))
        # 断言只有虚拟变量不同的情况下相等
        assert F(1, (x, 1, x)).dummy_eq(F(1, (y, 1, x)))

    # issue 5265 的问题验证
    assert Sum(x, (x, 1, x)).subs(x, a) == Sum(x, (x, 1, a))
# 定义测试函数 test_Sum_doit，用于测试符号求和的计算结果

def test_Sum_doit():
    # 断言符号求和 Sum(n*Integral(a**2), (n, 0, 2)) 的计算结果等于 a**3
    assert Sum(n*Integral(a**2), (n, 0, 2)).doit() == a**3
    # 断言不进行深度计算的符号求和 Sum(n*Integral(a**2), (n, 0, 2)) 的计算结果等于 3*Integral(a**2)
    assert Sum(n*Integral(a**2), (n, 0, 2)).doit(deep=False) == 3*Integral(a**2)
    # 断言使用 summation 函数进行求和的结果等于 3*Integral(a**2)
    assert summation(n*Integral(a**2), (n, 0, 2)) == 3*Integral(a**2)

    # 测试嵌套求和的计算结果
    s = Sum( Sum( Sum(2,(z,1,n+1)), (y,x+1,n)), (x,1,n))
    # 断言 s.doit() 减去 n*(n+1)*(n-1) 的因式化结果等于 0
    assert 0 == (s.doit() - n*(n+1)*(n-1)).factor()

    # KroneckerDelta 函数在整数范围内假设为有限的
    # 断言 Sum(KroneckerDelta(x, y), (x, -oo, oo)).doit() 的结果等于 Piecewise((1, And(-oo < y, y < oo)), (0, True))
    assert Sum(KroneckerDelta(x, y), (x, -oo, oo)).doit() == Piecewise((1, And(-oo < y, y < oo)), (0, True))
    # 断言 Sum(KroneckerDelta(m, n), (m, -oo, oo)).doit() 的结果等于 1
    assert Sum(KroneckerDelta(m, n), (m, -oo, oo)).doit() == 1
    # 断言 Sum(m*KroneckerDelta(x, y), (x, -oo, oo)).doit() 的结果等于 Piecewise((m, And(-oo < y, y < oo)), (0, True))
    assert Sum(m*KroneckerDelta(x, y), (x, -oo, oo)).doit() == Piecewise((m, And(-oo < y, y < oo)), (0, True))
    # 断言 Sum(x*KroneckerDelta(m, n), (m, -oo, oo)).doit() 的结果等于 x
    assert Sum(x*KroneckerDelta(m, n), (m, -oo, oo)).doit() == x
    # 断言 Sum(Sum(KroneckerDelta(m, n), (m, 1, 3)), (n, 1, 3)).doit() 的结果等于 3
    assert Sum(Sum(KroneckerDelta(m, n), (m, 1, 3)), (n, 1, 3)).doit() == 3
    # 断言 Sum(Sum(KroneckerDelta(k, m), (m, 1, 3)), (n, 1, 3)).doit() 的结果等于 3 * Piecewise((1, And(1 <= k, k <= 3)), (0, True))
    assert Sum(Sum(KroneckerDelta(k, m), (m, 1, 3)), (n, 1, 3)).doit() == \
           3 * Piecewise((1, And(1 <= k, k <= 3)), (0, True))
    # 断言 Sum(f(n) * Sum(KroneckerDelta(m, n), (m, 0, oo)), (n, 1, 3)).doit() 的结果等于 f(1) + f(2) + f(3)
    assert Sum(f(n) * Sum(KroneckerDelta(m, n), (m, 0, oo)), (n, 1, 3)).doit() == \
           f(1) + f(2) + f(3)
    # 断言 Sum(f(n) * Sum(KroneckerDelta(m, n), (m, 0, oo)), (n, 1, oo)).doit() 的结果等于 Sum(f(n), (n, 1, oo))
    assert Sum(f(n) * Sum(KroneckerDelta(m, n), (m, 0, oo)), (n, 1, oo)).doit() == \
           Sum(f(n), (n, 1, oo))

    # issue 2597
    nmax = symbols('N', integer=True, positive=True)
    pw = Piecewise((1, And(1 <= n, n <= nmax)), (0, True))
    # 断言 Sum(pw, (n, 1, nmax)).doit() 的结果等于 Sum(Piecewise((1, nmax >= n),
    # (0, True)), (n, 1, nmax))
    assert Sum(pw, (n, 1, nmax)).doit() == Sum(Piecewise((1, nmax >= n),
                    (0, True)), (n, 1, nmax))

    q, s = symbols('q, s')
    # 断言 summation(1/n**(2*s), (n, 1, oo)) 的结果等于 Piecewise((zeta(2*s), 2*s > 1),
    # (Sum(n**(-2*s), (n, 1, oo)), True))
    assert summation(1/n**(2*s), (n, 1, oo)) == Piecewise((zeta(2*s), 2*s > 1),
        (Sum(n**(-2*s), (n, 1, oo)), True))
    # 断言 summation(1/(n+1)**s, (n, 0, oo)) 的结果等于 Piecewise((zeta(s), s > 1),
    # (Sum((n + 1)**(-s), (n, 0, oo)), True))
    assert summation(1/(n+1)**s, (n, 0, oo)) == Piecewise((zeta(s), s > 1),
        (Sum((n + 1)**(-s), (n, 0, oo)), True))
    # 断言 summation(1/(n+q)**s, (n, 0, oo)) 的结果等于 Piecewise(
    # (zeta(s, q), And(q > 0, s > 1)),
    # (Sum((n + q)**(-s), (n, 0, oo)), True))
    assert summation(1/(n+q)**s, (n, 0, oo)) == Piecewise(
        (zeta(s, q), And(q > 0, s > 1)),
        (Sum((n + q)**(-s), (n, 0, oo)), True))
    # 断言 summation(1/(n+q)**s, (n, q, oo)) 的结果等于 Piecewise(
    # (zeta(s, 2*q), And(2*q > 0, s > 1)),
    # (Sum((n + q)**(-s), (n, q, oo)), True))
    assert summation(1/(n+q)**s, (n, q, oo)) == Piecewise(
        (zeta(s, 2*q), And(2*q > 0, s > 1)),
        (Sum((n + q)**(-s), (n, q, oo)), True))
    # 断言 summation(1/n**2, (n, 1, oo)) 的结果等于 zeta(2)
    assert summation(1/n**2, (n, 1, oo)) == zeta(2)
    # 断言 summation(1/n**s, (n, 0, oo)) 的结果等于 Sum(n**(-s), (n, 0, oo))
    assert summation(1/n**s, (n, 0, oo)) == Sum(n**(-s), (n, 0, oo))


# 定义测试函数 test_Product_doit，用于测试符号乘积的计算结果

def test_Product_doit():
    # 断言符号乘积 Product(n*Integral(a**2), (n, 1, 3)) 的计算结果等于 2 * a**9 / 9
    assert Product(n*Integral(a**2), (n, 1, 3)).doit() == 2 * a**9 / 9
    # 断言不进行深度计算的符号乘积 Product(n*Integral(a**2), (n, 1, 3)) 的计算结果等于 6*Integral(a**2)**3
    assert Product(n*Integral(a
    # 断言：e 关于 a 的导数等于对 e 求关于 a 的导数
    assert e.diff(a) == Derivative(e, a)
    # 断言：对于求和表达式 Sum(x*y, (x, 1, 3), (a, 2, 5)) 求关于 y 的导数的结果等于以下两个表达式的结果相等：先对整个求和表达式求值再求 y 的导数，以及直接对求和表达式求 y 的导数，结果应为 24
    assert Sum(x*y, (x, 1, 3), (a, 2, 5)).diff(y).doit() == \
        Sum(x*y, (x, 1, 3), (a, 2, 5)).doit().diff(y) == 24
    # 断言：对于求和表达式 Sum(x, (x, 1, 2)) 求关于 y 的导数应该等于 0，因为 y 不在求和的变量范围内
    assert Sum(x, (x, 1, 2)).diff(y) == 0
def test_hypersum():
    # 检查简化后的级数求和是否等于指定表达式
    assert simplify(summation(x**n/fac(n), (n, 1, oo))) == -1 + exp(x)
    # 检查级数求和是否等于余弦函数
    assert summation((-1)**n * x**(2*n) / fac(2*n), (n, 0, oo)) == cos(x)
    # 检查简化后的级数求和是否等于指定表达式
    assert simplify(summation((-1)**n*x**(2*n + 1) /
        factorial(2*n + 1), (n, 3, oo))) == -x + sin(x) + x**3/6 - x**5/120

    # 检查级数求和是否等于有理数和 Zeta 函数的值
    assert summation(1/(n + 2)**3, (n, 1, oo)) == Rational(-9, 8) + zeta(3)
    # 检查级数求和是否等于 π 的四次幂除以 90
    assert summation(1/n**4, (n, 1, oo)) == pi**4/90

    # 计算负无穷到零的级数求和
    s = summation(x**n*n, (n, -oo, 0))
    # 断言结果为 Piecewise 对象
    assert s.is_Piecewise
    # 断言第一个条件为 -1/(x*(1 - 1/x)**2)
    assert s.args[0].args[0] == -1/(x*(1 - 1/x)**2)
    # 断言第二个条件为 |1/x| < 1
    assert s.args[0].args[1] == (abs(1/x) < 1)

    # 创建一个符号 m，其为正整数
    m = Symbol('n', integer=True, positive=True)
    # 检查二项式系数的级数求和是否等于 2 的 m 次方
    assert summation(binomial(m, k), (k, 0, m)) == 2**m


def test_issue_4170():
    # 检查级数求和是否等于自然常数 e
    assert summation(1/factorial(k), (k, 0, oo)) == E


def test_is_commutative():
    from sympy.physics.secondquant import NO, F, Fd
    m = Symbol('m', commutative=False)
    # 对于每个 f 函数（Sum, Product, Integral），检查是否可交换
    for f in (Sum, Product, Integral):
        assert f(z, (z, 1, 1)).is_commutative is True
        assert f(z*y, (z, 1, 6)).is_commutative is True
        assert f(m*x, (x, 1, 2)).is_commutative is False

        # 断言对于特定的 NO(Fd(x)*F(y))*z 表达式，是否可交换
        assert f(NO(Fd(x)*F(y))*z, (z, 1, 2)).is_commutative is False


def test_is_zero():
    # 检查对于每个函数（Sum, Product），是否为零
    for func in [Sum, Product]:
        assert func(0, (x, 1, 1)).is_zero is True
        assert func(x, (x, 1, 1)).is_zero is None

    assert Sum(0, (x, 1, 0)).is_zero is True
    assert Product(0, (x, 1, 0)).is_zero is False


def test_is_number():
    # is_number 不依赖于求值或假设，等价于 `not foo.free_symbols`
    assert Sum(1, (x, 1, 1)).is_number is True
    assert Sum(1, (x, 1, x)).is_number is False
    assert Sum(0, (x, y, z)).is_number is False
    assert Sum(x, (y, 1, 2)).is_number is False
    assert Sum(x, (y, 1, 1)).is_number is False
    assert Sum(x, (x, 1, 2)).is_number is True
    assert Sum(x*y, (x, 1, 2), (y, 1, 3)).is_number is True

    assert Product(2, (x, 1, 1)).is_number is True
    assert Product(2, (x, 1, y)).is_number is False
    assert Product(0, (x, y, z)).is_number is False
    assert Product(1, (x, y, z)).is_number is False
    assert Product(x, (y, 1, x)).is_number is False
    assert Product(x, (y, 1, 2)).is_number is False
    assert Product(x, (y, 1, 1)).is_number is False
    assert Product(x, (x, 1, 2)).is_number is True
    # 对于每个函数 func 在 Sum 和 Product 中进行迭代
    for func in [Sum, Product]:
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为空
        assert func(1, (x, 1, 2)).free_symbols == set()
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {y}
        assert func(0, (x, 1, y)).free_symbols == {y}
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {y}
        assert func(2, (x, 1, y)).free_symbols == {y}
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为空
        assert func(x, (x, 1, 2)).free_symbols == set()
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {y}
        assert func(x, (x, 1, y)).free_symbols == {y}
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {x, y}
        assert func(x, (y, 1, y)).free_symbols == {x, y}
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {x}
        assert func(x, (y, 1, 2)).free_symbols == {x}
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {x}
        assert func(x, (y, 1, 1)).free_symbols == {x}
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {x, z}
        assert func(x, (y, 1, z)).free_symbols == {x, z}
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为空
        assert func(x, (x, 1, y), (y, 1, 2)).free_symbols == set()
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {z}
        assert func(x, (x, 1, y), (y, 1, z)).free_symbols == {z}
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {y}
        assert func(x, (x, 1, y), (y, 1, y)).free_symbols == {y}
        # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {x, z}
        assert func(x, (y, 1, y), (y, 1, z)).free_symbols == {x, z}
    
    # 断言：调用 Sum 函数，并检查其返回的表达式中的自由符号集合是否为 {y}
    assert Sum(1, (x, 1, y)).free_symbols == {y}
    # 注：free_symbols 方法回答的是对象本身是否有自由符号，而不是其求值后是否有自由符号
    # 断言：调用 Product 函数，并检查其返回的表达式中的自由符号集合是否为 {y}
    assert Product(1, (x, 1, y)).free_symbols == {y}
    # 注：不计算不独立于积分变量的自由符号
    # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为空
    assert func(f(x), (f(x), 1, 2)).free_symbols == set()
    # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {x}
    assert func(f(x), (f(x), 1, x)).free_symbols == {x}
    # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {y}
    assert func(f(x), (f(x), 1, y)).free_symbols == {y}
    # 断言：调用 func 函数，并检查其返回的表达式中的自由符号集合是否为 {x, y}
    assert func(f(x), (z, 1, y)).free_symbols == {x, y}
def test_conjugate_transpose():
    # 定义非交换符号 A 和 B
    A, B = symbols("A B", commutative=False)
    # 创建求和表达式 p = A * B**n, n 从 1 到 3
    p = Sum(A*B**n, (n, 1, 3))
    # 断言共轭转置后的结果等于转置后再计算的结果
    assert p.adjoint().doit() == p.doit().adjoint()
    # 断言共轭后再计算的结果等于计算后再共轭的结果
    assert p.conjugate().doit() == p.doit().conjugate()
    # 断言转置后再计算的结果等于计算后再转置的结果
    assert p.transpose().doit() == p.doit().transpose()

    # 交换 A 和 B 的顺序后重复上述断言
    p = Sum(B**n*A, (n, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()


def test_noncommutativity_honoured():
    # 定义非交换符号 A 和 B
    A, B = symbols("A B", commutative=False)
    # 定义整数 M
    M = symbols('M', integer=True, positive=True)
    # 创建求和表达式 p = A * B**n, n 从 1 到 M
    p = Sum(A*B**n, (n, 1, M))
    # 断言计算结果等于给定的分段函数表达式
    assert p.doit() == A*Piecewise((M, Eq(B, 1)),
                                   ((B - B**(M + 1))*(1 - B)**(-1), True))

    # 交换 A 和 B 的顺序后重复上述断言
    p = Sum(B**n*A, (n, 1, M))
    assert p.doit() == Piecewise((M, Eq(B, 1)),
                                 ((B - B**(M + 1))*(1 - B)**(-1), True))*A

    # 创建求和表达式 p = A * B**n * A * B**n, n 从 1 到 M
    p = Sum(B**n*A*B**n, (n, 1, M))
    # 断言计算结果等于 p 本身
    assert p.doit() == p


def test_issue_4171():
    # 断言阶乘求和在 k 趋向无穷时等于无穷
    assert summation(factorial(2*k + 1)/factorial(2*k), (k, 0, oo)) is oo
    # 断言求和 2*k + 1 在 k 趋向无穷时等于无穷
    assert summation(2*k + 1, (k, 0, oo)) is oo


def test_issue_6273():
    # 断言对 x 的求和，保留两位小数且替换 n 为 1 的结果等于浮点数 1.00
    assert Sum(x, (x, 1, n)).n(2, subs={n: 1}) == Float(1, 2)


def test_issue_6274():
    # 断言对 x 的求和，从 1 到 0 的结果等于 0
    assert Sum(x, (x, 1, 0)).doit() == 0
    # 断言对 x 的求和，从 1 到 0 的数值化结果等于字符串 '0'
    assert NS(Sum(x, (x, 1, 0))) == '0'
    # 断言对 n 的求和，从 10 到 5 的结果等于 -30
    assert Sum(n, (n, 10, 5)).doit() == -30
    # 断言对 n 的求和，从 10 到 5 的数值化结果等于 '-30.0000000000000'
    assert NS(Sum(n, (n, 10, 5))) == '-30.0000000000000'


def test_simplify_sum():
    y, t, v = symbols('y, t, v')

    # 定义简化函数 _simplify
    _simplify = lambda e: simplify(e, doit=False)
    # 断言简化后的两个求和表达式相等
    assert _simplify(Sum(x*y, (x, n, m), (y, a, k)) + \
        Sum(y, (x, n, m), (y, a, k))) == Sum(y * (x + 1), (x, n, m), (y, a, k))
    # 断言简化后的两个求和表达式相等
    assert _simplify(Sum(x, (x, n, m)) + Sum(x, (x, m + 1, a))) == \
        Sum(x, (x, n, a))
    # 断言简化后的两个求和表达式相等
    assert _simplify(Sum(x, (x, k + 1, a)) + Sum(x, (x, n, k))) == \
        Sum(x, (x, n, a))
    # 断言简化后的两个求和表达式相等
    assert _simplify(Sum(x, (x, k + 1, a)) + Sum(x + 1, (x, n, k))) == \
        Sum(x, (x, n, a)) + Sum(1, (x, n, k))
    # 断言简化后的表达式相等
    assert _simplify(Sum(x, (x, 0, 3)) * 3 + 3 * Sum(x, (x, 4, 6)) + \
        4 * Sum(z, (z, 0, 1))) == 4*Sum(z, (z, 0, 1)) + 3*Sum(x, (x, 0, 6))
    # 断言简化后的表达式相等
    assert _simplify(3*Sum(x**2, (x, a, b)) + Sum(x, (x, a, b))) == \
        Sum(x*(3*x + 1), (x, a, b))
    # 断言简化后的表达式相等
    assert _simplify(Sum(x**3, (x, n, k)) * 3 + 3 * Sum(x, (x, n, k)) + \
        4 * y * Sum(z, (z, n, k))) + 1 == \
            4*y*Sum(z, (z, n, k)) + 3*Sum(x**3 + x, (x, n, k)) + 1
    # 断言简化后的表达式相等
    assert _simplify(Sum(x, (x, a, b)) + 1 + Sum(x, (x, b + 1, c))) == \
        1 + Sum(x, (x, a, c))
    # 断言简化后的表达式相等
    assert _simplify(Sum(x, (t, a, b)) + Sum(y, (t, a, b)) + \
        Sum(x, (t, b+1, c))) == x * Sum(1, (t, a, c)) + y * Sum(1, (t, a, b))
    # 断言简化后的表达式相等
    assert _simplify(Sum(x, (t, a, b)) + Sum(x, (t, b+1, c)) + \
        Sum(y, (t, a, b))) == x * Sum(1, (t, a, c)) + y * Sum(1, (t, a, b))
    # 断言简化后的表达式相等
    assert _simplify(Sum(x, (t, a, b)) + 2 * Sum(x, (t, b+1, c))) == \
        _simplify(Sum(x, (t, a, b)) + Sum(x, (t, b+1, c)) + Sum(x, (t, b+1, c)))
    # 断言：简化求和乘积的表达式等于分解后的乘积
    assert _simplify(Sum(x, (x, a, b))*Sum(x**2, (x, a, b))) == \
        Sum(x, (x, a, b)) * Sum(x**2, (x, a, b))

    # 断言：简化多个求和的加法等于求和内部元素的和乘以求和的数量
    assert _simplify(Sum(x, (t, a, b)) + Sum(y, (t, a, b)) + Sum(z, (t, a, b))) \
        == (x + y + z) * Sum(1, (t, a, b))          # issue 8596

    # 断言：简化多个求和的加法等于求和内部元素的和乘以求和的数量
    assert _simplify(Sum(x, (t, a, b)) + Sum(y, (t, a, b)) + Sum(z, (t, a, b)) + \
        Sum(v, (t, a, b))) == (x + y + z + v) * Sum(1, (t, a, b))  # issue 8596

    # 断言：简化求和乘积后除以常数等于将求和内部元素除以常数
    assert _simplify(Sum(x * y, (x, a, b)) / (3 * y)) == \
        (Sum(x, (x, a, b)) / 3)

    # 断言：简化求和乘积除以相同因子等于只保留函数 f(x) 在求和中的部分
    assert _simplify(Sum(f(x) * y * z, (x, a, b)) / (y * z)) \
        == Sum(f(x), (x, a, b))

    # 断言：简化常数乘以求和的差等于零
    assert _simplify(Sum(c * x, (x, a, b)) - c * Sum(x, (x, a, b))) == 0

    # 断言：简化常数乘以求和的和等于常数乘以求和的和
    assert _simplify(c * (Sum(x, (x, a, b))  + y)) == c * (y + Sum(x, (x, a, b)))

    # 断言：简化常数乘以求和的和等于常数乘以求和的和
    assert _simplify(c * (Sum(x, (x, a, b)) + y * Sum(x, (x, a, b)))) == \
        c * (y + 1) * Sum(x, (x, a, b))

    # 断言：简化求和内部包含求和后再次求和的表达式
    assert _simplify(Sum(Sum(c * x, (x, a, b)), (y, a, b))) == \
                c * Sum(x, (x, a, b), (y, a, b))

    # 断言：简化求和内部包含求和和乘积的表达式
    assert _simplify(Sum((3 + y) * Sum(c * x, (x, a, b)), (y, a, b))) == \
                c * Sum((3 + y), (y, a, b)) * Sum(x, (x, a, b))

    # 断言：简化求和内部包含求和和乘积的表达式
    assert _simplify(Sum((3 + t) * Sum(c * t, (x, a, b)), (y, a, b))) == \
                c*t*(t + 3)*Sum(1, (x, a, b))*Sum(1, (y, a, b))

    # 断言：简化求和内部包含两部分求和的表达式
    assert _simplify(Sum(Sum(d * t, (x, a, b - 1)) + \
                Sum(d * t, (x, b, c)), (t, a, b))) == \
                    d * Sum(1, (x, a, c)) * Sum(t, (t, a, b))

    # 断言：简化求和内部包含三角函数的平方和加一的表达式
    assert _simplify(Sum(sin(t)**2 + cos(t)**2 + 1, (t, a, b))) == \
        2 * Sum(1, (t, a, b))
# 定义一个测试函数，用于测试Sum类的change_index方法
def test_change_index():
    # 定义整数符号变量b, v, w
    b, v, w = symbols('b, v, w', integer=True)

    # 断言语句：测试Sum对象的change_index方法是否按预期修改索引变量x为x+1，结果应为Sum(y - 1, (y, a + 1, b + 1))
    assert Sum(x, (x, a, b)).change_index(x, x + 1, y) == \
        Sum(y - 1, (y, a + 1, b + 1))
    
    # 断言语句：测试Sum对象的change_index方法是否按预期修改索引变量x为x-1，结果应为Sum((x+1)**2, (x, a-1, b-1))
    assert Sum(x**2, (x, a, b)).change_index(x, x - 1) == \
        Sum((x + 1)**2, (x, a - 1, b - 1))
    
    # 断言语句：测试Sum对象的change_index方法是否按预期修改索引变量x为-x，结果应为Sum((-y)**2, (y, -b, -a))
    assert Sum(x**2, (x, a, b)).change_index(x, -x, y) == \
        Sum((-y)**2, (y, -b, -a))
    
    # 断言语句：测试Sum对象的change_index方法是否按预期修改索引变量x为-x-1，结果应为Sum(-x-1, (x, -b-1, -a-1))
    assert Sum(x, (x, a, b)).change_index(x, -x - 1) == \
        Sum(-x - 1, (x, -b - 1, -a - 1))
    
    # 断言语句：测试Sum对象的change_index方法是否按预期修改多个索引变量x为x-1, y为z+1，结果应为Sum((z+1)*y, (z, a-1, b-1), (y, c, d))
    assert Sum(x*y, (x, a, b), (y, c, d)).change_index(x, x - 1, z) == \
        Sum((z + 1)*y, (z, a - 1, b - 1), (y, c, d))
    
    # 断言语句：测试Sum对象的change_index方法是否按预期修改索引变量x为x+v，结果应为Sum(-v + x, (x, a+v, b+v))
    assert Sum(x, (x, a, b)).change_index(x, x + v) == \
        Sum(-v + x, (x, a + v, b + v))
    
    # 断言语句：测试Sum对象的change_index方法是否按预期修改索引变量x为-x-v，结果应为Sum(-v - x, (x, -b - v, -a - v))
    assert Sum(x, (x, a, b)).change_index(x, -x - v) == \
        Sum(-v - x, (x, -b - v, -a - v))
    
    # 断言语句：测试Sum对象的change_index方法是否按预期修改索引变量x为wx，结果应为Sum(v/w, (v, bw, aw))
    assert Sum(x, (x, a, b)).change_index(x, w*x, v) == \
        Sum(v/w, (v, b*w, a*w))
    
    # 断言语句：测试Sum对象的change_index方法是否能捕获到预期的异常，此处应引发ValueError
    raises(ValueError, lambda: Sum(x, (x, a, b)).change_index(x, 2*x))


# 定义一个测试函数，用于测试Sum类的reorder方法
def test_reorder():
    # 定义整数符号变量b, y, c, d, z
    b, y, c, d, z = symbols('b, y, c, d, z', integer=True)

    # 断言语句：测试Sum对象的reorder方法是否按预期重新排列索引顺序，结果应为Sum(x*y, (y, c, d), (x, a, b))
    assert Sum(x*y, (x, a, b), (y, c, d)).reorder((0, 1)) == \
        Sum(x*y, (y, c, d), (x, a, b))
    
    # 断言语句：测试Sum对象的reorder方法是否按预期重新排列索引顺序，结果应为Sum(x, (x, c, d), (x, a, b))
    assert Sum(x, (x, a, b), (x, c, d)).reorder((0, 1)) == \
        Sum(x, (x, c, d), (x, a, b))
    
    # 断言语句：测试Sum对象的reorder方法是否按预期重新排列多个索引顺序，结果应为Sum(x*y+z, (z, m, n), (y, c, d), (x, a, b))
    assert Sum(x*y + z, (x, a, b), (z, m, n), (y, c, d)).reorder(\
        (2, 0), (0, 1)) == Sum(x*y + z, (z, m, n), (y, c, d), (x, a, b))
    
    # 断言语句：测试Sum对象的reorder方法是否按预期重新排列多个索引顺序，结果应为Sum(x*y*z, (x, a, b), (z, m, n), (y, c, d))
    assert Sum(x*y*z, (x, a, b), (y, c, d), (z, m, n)).reorder(\
        (0, 1), (1, 2), (0, 2)) == Sum(x*y*z, (x, a, b), (z, m, n), (y, c, d))
    
    # 断言语句：测试Sum对象的reorder方法是否按预期重新排列多个索引顺序，结果应为Sum(x*y*z, (x, a, b), (z, m, n), (y, c, d))
    assert Sum(x*y*z, (x, a, b), (y, c, d), (z, m, n)).reorder(\
        (x, y), (y, z), (x, z)) == Sum(x*y*z, (x, a, b), (z, m, n), (y, c, d))
    
    # 断言语句：测试Sum对象的reorder方法是否按预期重新排列索引顺序，结果应为Sum(x*y, (y, c, d), (x, a, b))
    assert Sum(x*y, (x, a, b), (y, c, d)).reorder((x, 1)) == \
        Sum(x*y, (y, c, d), (x, a, b))
    
    # 断言语句：测试Sum对象的reorder方法是否按预期重新排列索引顺序，结果应为Sum(x*y, (y, c, d), (x, a, b))
    assert Sum(x*y, (x, a, b), (y, c, d)).reorder((y, x)) == \
        Sum(x*y, (y, c, d), (x, a, b))


# 定义一个测试函数，用于测试Sum类的reverse_order方法
def test_reverse_order():
    # 断言语句：测试Sum对象的reverse_order方法是否按预期反转索引顺序，结果应为Sum(-x, (x, 4, -1))
    assert Sum(x, (x, 0, 3)).reverse_order(0) == Sum(-x, (x, 4, -1))
    
    # 断言语句：测试Sum对象的reverse_order方法是否按预期反转多个索引顺序，结果应为Sum(x*y, (x, 6, 0), (y, 7, -1))
    assert Sum(x*y, (x, 1, 5), (y, 0, 6)).reverse_order(0, 1) == \
           Sum(x*y, (x, 6, 0), (y, 7, -1))
    
    # 断言语句：测试Sum对象的reverse_order方法是否按预期反转索引顺序，结果应为Sum(-x, (x, 3, 0))
    assert Sum(x,
    # 断言语句，验证两个表达式是否相等
    assert Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x) == \
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))
    # 上述断言的含义是：左侧的求和表达式按照 x 从 a 到 b，y 从 2 到 5 的顺序求和后，反转 y 和 x 的顺序，应该等于
    # 右侧的求和表达式按照 x 从 b + 1 到 a - 1，y 从 6 到 1 的顺序求和。
def test_issue_7097():
    # 检查数学表达式的相等性
    assert sum(x**n/n for n in range(1, 401)) == summation(x**n/n, (n, 1, 400))


def test_factor_expand_subs():
    # 测试因式分解
    assert Sum(4 * x, (x, 1, y)).factor() == 4 * Sum(x, (x, 1, y))
    assert Sum(x * a, (x, 1, y)).factor() == a * Sum(x, (x, 1, y))
    assert Sum(4 * x * a, (x, 1, y)).factor() == 4 * a * Sum(x, (x, 1, y))
    assert Sum(4 * x * y, (x, 1, y)).factor() == 4 * y * Sum(x, (x, 1, y))

    # 测试展开
    _x = Symbol('x', zero=False)
    assert Sum(x+1,(x,1,y)).expand() == Sum(x,(x,1,y)) + Sum(1,(x,1,y))
    assert Sum(x+a*x**2,(x,1,y)).expand() == Sum(x,(x,1,y)) + Sum(a*x**2,(x,1,y))
    assert Sum(_x**(n + 1)*(n + 1), (n, -1, oo)).expand() \
        == Sum(n*_x*_x**n + _x*_x**n, (n, -1, oo))
    assert Sum(x**(n + 1)*(n + 1), (n, -1, oo)).expand(power_exp=False) \
        == Sum(n*x**(n + 1) + x**(n + 1), (n, -1, oo))
    assert Sum(x**(n + 1)*(n + 1), (n, -1, oo)).expand(force=True) \
           == Sum(x*x**n, (n, -1, oo)) + Sum(n*x*x**n, (n, -1, oo))
    assert Sum(a*n+a*n**2,(n,0,4)).expand() \
        == Sum(a*n,(n,0,4)) + Sum(a*n**2,(n,0,4))
    assert Sum(_x**a*_x**n,(x,0,3)) \
        == Sum(_x**(a+n),(x,0,3)).expand(power_exp=True)
    _a, _n = symbols('a n', positive=True)
    assert Sum(x**(_a+_n),(x,0,3)).expand(power_exp=True) \
        == Sum(x**_a*x**_n, (x, 0, 3))
    assert Sum(x**(_a-_n),(x,0,3)).expand(power_exp=True) \
        == Sum(x**(_a-_n),(x,0,3)).expand(power_exp=False)

    # 测试替换
    assert Sum(1/(1+a*x**2),(x,0,3)).subs([(a,3)]) == Sum(1/(1+3*x**2),(x,0,3))
    assert Sum(x*y,(x,0,y),(y,0,x)).subs([(x,3)]) == Sum(x*y,(x,0,y),(y,0,3))
    assert Sum(x,(x,1,10)).subs([(x,y-2)]) == Sum(x,(x,1,10))
    assert Sum(1/x,(x,1,10)).subs([(x,(3+n)**3)]) == Sum(1/x,(x,1,10))
    assert Sum(1/x,(x,1,10)).subs([(x,3*x-2)]) == Sum(1/x,(x,1,10))


def test_distribution_over_equality():
    # 测试在等式上分布
    assert Product(Eq(x*2, f(x)), (x, 1, 3)).doit() == Eq(48, f(1)*f(2)*f(3))
    assert Sum(Eq(f(x), x**2), (x, 0, y)) == \
        Eq(Sum(f(x), (x, 0, y)), Sum(x**2, (x, 0, y)))


def test_issue_2787():
    n, k = symbols('n k', positive=True, integer=True)
    p = symbols('p', positive=True)
    binomial_dist = binomial(n, k)*p**k*(1 - p)**(n - k)
    s = Sum(binomial_dist*k, (k, 0, n))
    res = s.doit().simplify()
    ans = Piecewise(
        (n*p, x),
        (Sum(k*p**k*binomial(n, k)*(1 - p)**(n - k), (k, 0, n)),
        True)).subs(x, (Eq(n, 1) | (n > 1)) & (p/Abs(p - 1) <= 1))
    ans2 = Piecewise(
        (n*p, x),
        (factorial(n)*Sum(p**k*(1 - p)**(-k + n)/
        (factorial(-k + n)*factorial(k - 1)), (k, 0, n)),
        True)).subs(x, (Eq(n, 1) | (n > 1)) & (p/Abs(p - 1) <= 1))
    assert res in [ans, ans2]  # XXX system dependent
    # Issue #17165: make sure that another simplify does not complicate
    # the result by much. Why didn't first simplify replace
    # Eq(n, 1) | (n > 1) with True?
    assert res.simplify().count_ops() <= res.count_ops() + 2
def test_issue_4668():
    # Assert that the sum of 1/n from n=2 to infinity is infinity
    assert summation(1/n, (n, 2, oo)) is oo


def test_matrix_sum():
    # Create a 2x2 matrix A with specific elements including symbol n
    A = Matrix([[0, 1], [n, 0]])

    # Compute the sum of matrix A with n ranging from 0 to 3, and evaluate the result
    result = Sum(A, (n, 0, 3)).doit()
    assert result == Matrix([[0, 4], [6, 0]])
    # Assert that the type of result is ImmutableDenseMatrix
    assert result.__class__ == ImmutableDenseMatrix

    # Create a sparse matrix A with the same elements
    A = SparseMatrix([[0, 1], [n, 0]])

    # Compute the sum of sparse matrix A with n ranging from 0 to 3, and evaluate the result
    result = Sum(A, (n, 0, 3)).doit()
    # Assert that the type of result is ImmutableSparseMatrix
    assert result.__class__ == ImmutableSparseMatrix


def test_failing_matrix_sum():
    n = Symbol('n')
    # TODO Implement matrix geometric series summation.
    # Define matrix A with specific elements
    A = Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    # Assert that the sum of A**n from n=1 to 4 is a zero matrix
    assert Sum(A ** n, (n, 1, 4)).doit() == \
        Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # issue sympy/sympy#16989
    # Assert that the sum of A**n from n=1 to 1 is matrix A itself
    assert summation(A**n, (n, 1, 1)) == A


def test_indexed_idx_sum():
    # Define symbol i as an Indexed symbol
    i = symbols('i', cls=Idx)
    # Create an Indexed variable r based on i
    r = Indexed('r', i)
    # Assert that the sum of r over i from 0 to 3 evaluates to the sum of r with i replaced by j for j in range(4)
    assert Sum(r, (i, 0, 3)).doit() == sum(r.xreplace({i: j}) for j in range(4))
    # Assert that the product of r over i from 0 to 3 evaluates to the product of r with i replaced by j for j in range(4)
    assert Product(r, (i, 0, 3)).doit() == prod([r.xreplace({i: j}) for j in range(4)])

    # Define symbol j as an integer
    j = symbols('j', integer=True)
    # Assert that the sum of r over i from j to j+2 evaluates to the sum of r with i replaced by j+k for k in range(3)
    assert Sum(r, (i, j, j+2)).doit() == sum(r.xreplace({i: j+k}) for k in range(3))
    # Assert that the product of r over i from j to j+2 evaluates to the product of r with i replaced by j+k for k in range(3)
    assert Product(r, (i, j, j+2)).doit() == prod([r.xreplace({i: j+k}) for k in range(3)])

    # Define i as an Idx symbol with range (1, 3)
    k = Idx('k', range=(1, 3))
    # Define IndexedBase A
    A = IndexedBase('A')
    # Assert that the sum of A[k] over k evaluates to the sum of A with k replaced by Idx(j, (1, 3)) for j in range(1, 4)
    assert Sum(A[k], k).doit() == sum(A[Idx(j, (1, 3))] for j in range(1, 4))
    # Assert that the product of A[k] over k evaluates to the product of A with k replaced by Idx(j, (1, 3)) for j in range(1, 4)
    assert Product(A[k], k).doit() == prod([A[Idx(j, (1, 3))] for j in range(1, 4)])

    # Assert that attempting to sum A[k] from k=1 to 4 raises a ValueError
    raises(ValueError, lambda: Sum(A[k], (k, 1, 4)))
    # Assert that attempting to sum A[k] from k=0 to 3 raises a ValueError
    raises(ValueError, lambda: Sum(A[k], (k, 0, 3)))
    # Assert that attempting to sum A[k] from k=2 to infinity raises a ValueError
    raises(ValueError, lambda: Sum(A[k], (k, 2, oo)))

    # Assert that attempting to compute the product of A[k] from k=1 to 4 raises a ValueError
    raises(ValueError, lambda: Product(A[k], (k, 1, 4)))
    # Assert that attempting to compute the product of A[k] from k=0 to 3 raises a ValueError
    raises(ValueError, lambda: Product(A[k], (k, 0, 3)))
    # Assert that attempting to compute the product of A[k] from k=2 to infinity raises a ValueError
    raises(ValueError, lambda: Product(A[k], (k, 2, oo)))


@slow
def test_is_convergent():
    # divergence tests --
    # Assert that the sum of n/(2*n + 1) from n=1 to infinity is divergent
    assert Sum(n/(2*n + 1), (n, 1, oo)).is_convergent() is S.false
    # Assert that the sum of factorial(n)/5**n from n=1 to infinity is divergent
    assert Sum(factorial(n)/5**n, (n, 1, oo)).is_convergent() is S.false
    # Assert that the sum of 3**(-2*n - 1)*n**n from n=1 to infinity is divergent
    assert Sum(3**(-2*n - 1)*n**n, (n, 1, oo)).is_convergent() is S.false
    # Assert that the sum of (-1)**n*n from n=3 to infinity is divergent
    assert Sum((-1)**n*n, (n, 3, oo)).is_convergent() is S.false
    # Assert that the sum of (-1)**n from n=1 to infinity is divergent
    assert Sum((-1)**n, (n, 1, oo)).is_convergent() is S.false
    # Assert that the sum of log(1/n) from n=2 to infinity is divergent
    assert Sum(log(1/n), (n, 2, oo)).is_convergent() is S.false

    # Raabe's test --
    # Assert that the sum involving Raabe's test is convergent
    assert Sum(Product((3*m),(m,1,n))/Product((3*m+4),(m,1,n)),(n,1,oo)).is_convergent() is S.true

    # root test --
    # Assert that the sum of (-12)**n/n from n=1 to infinity is divergent
    assert Sum((-12)**n/n, (n, 1, oo)).is_convergent() is S.false

    # integral test --

    # p-series test --
    # Assert that the sum of 1/(n**2 + 1) from n=1 to infinity is convergent
    assert Sum(1/(n**2 + 1), (n, 1, oo)).is_convergent() is S.true
    # Assert that the sum of 1/n**(6/5) from n=1 to infinity is convergent
    assert Sum(1/n**Rational(6, 5), (n, 1, oo)).is_convergent() is S.true
    # Assert that the sum of 2/(n*sqrt(n - 1)) from n=2 to infinity is convergent
    assert Sum(2/(n*sqrt(n - 1)), (n, 2, oo)).is_convergent() is S.true
    # Assert that the sum of 1/(sqrt(n)*sqrt(n)) from n=2 to infinity is divergent
    assert Sum(1/(sqrt(n)*sqrt(n)), (n, 2, oo)).is_convergent() is S.false
    # Assert that the sum of factorial(n) / factorial(n+2) from n=1 to infinity is convergent
    assert Sum(factorial(n) / factorial(n+2), (n, 1, oo)).is_convergent() is S.true
    # Assert that the sum involving rising factorial and falling factorial is convergent
    assert Sum(rf(5,n)/rf(7,n),(n,1,oo)).is_convergent() is S.true
    # 使用 SymPy 的 Sum 函数计算数列的收敛性

    # 确定该级数不收敛
    assert Sum((rf(1, n)*rf(2, n))/(rf(3, n)*factorial(n)), (n, 1, oo)).is_convergent() is S.false

    # 比较测试 --
    # 以下各级数的收敛性断言
    assert Sum(1/(n + log(n)), (n, 1, oo)).is_convergent() is S.false
    assert Sum(1/(n**2*log(n)), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1/(n*log(n)), (n, 2, oo)).is_convergent() is S.false
    assert Sum(2/(n*log(n)*log(log(n))**2), (n, 5, oo)).is_convergent() is S.true
    assert Sum(2/(n*log(n)**2), (n, 2, oo)).is_convergent() is S.true
    assert Sum((n - 1)/(n**2*log(n)**3), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1/(n*log(n)*log(log(n))), (n, 5, oo)).is_convergent() is S.false
    assert Sum((n - 1)/(n*log(n)**3), (n, 3, oo)).is_convergent() is S.false
    assert Sum(2/(n**2*log(n)), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1/(n*sqrt(log(n))*log(log(n))), (n, 100, oo)).is_convergent() is S.false
    assert Sum(log(log(n))/(n*log(n)**2), (n, 100, oo)).is_convergent() is S.true
    assert Sum(log(n)/n**2, (n, 5, oo)).is_convergent() is S.true

    # 交替级数测试 --
    assert Sum((-1)**(n - 1)/(n**2 - 1), (n, 3, oo)).is_convergent() is S.true

    # 负无穷限制测试
    assert Sum(1/(n**2 + 1), (n, -oo, 1)).is_convergent() is S.true
    assert Sum(1/(n - 1), (n, -oo, -1)).is_convergent() is S.false
    assert Sum(1/(n**2 - 1), (n, -oo, -5)).is_convergent() is S.true
    assert Sum(1/(n**2 - 1), (n, -oo, 2)).is_convergent() is S.true
    assert Sum(1/(n**2 - 1), (n, -oo, oo)).is_convergent() is S.true

    # 分段函数
    f = Piecewise((n**(-2), n <= 1), (n**2, n > 1))
    assert Sum(f, (n, 1, oo)).is_convergent() is S.false
    assert Sum(f, (n, -oo, oo)).is_convergent() is S.false
    assert Sum(f, (n, 1, 100)).is_convergent() is S.true
    #assert Sum(f, (n, -oo, 1)).is_convergent() is S.true

    # 积分测试
    assert Sum(log(n)/n**3, (n, 1, oo)).is_convergent() is S.true
    assert Sum(-log(n)/n**3, (n, 1, oo)).is_convergent() is S.true
    # 下列函数在 (x, y) = (1.2, 0.43), (3.0, -0.25) 和 (6.8, 0.050) 处有极大值
    eq = (x - 2)*(x**2 - 6*x + 4)*exp(-x)
    assert Sum(eq, (x, 1, oo)).is_convergent() is S.true
    assert Sum(eq, (x, 1, 2)).is_convergent() is S.true
    assert Sum(1/(x**3), (x, 1, oo)).is_convergent() is S.true
    assert Sum(1/(x**S.Half), (x, 1, oo)).is_convergent() is S.false

    # 问题 19545
    assert Sum(1/n - 3/(3*n + 2), (n, 1, oo)).is_convergent() is S.true

    # 问题 19836
    assert Sum(4/(n + 2) - 5/(n + 1) + 1/n, (n, 7, oo)).is_convergent() is S.true
def test_is_absolutely_convergent():
    # 检验交替级数的绝对收敛性质
    assert Sum((-1)**n, (n, 1, oo)).is_absolutely_convergent() is S.false
    # 检验幂级数的绝对收敛性质
    assert Sum((-1)**n/n**2, (n, 1, oo)).is_absolutely_convergent() is S.true


@XFAIL
def test_convergent_failing():
    # 狄利克雷级数测试
    assert Sum(sin(n)/n, (n, 1, oo)).is_convergent() is S.true
    assert Sum(sin(2*n)/n, (n, 1, oo)).is_convergent() is S.true


def test_issue_6966():
    # 符号定义
    i, k, m = symbols('i k m', integer=True)
    z_i, q_i = symbols('z_i q_i')
    # 级数求和
    a_k = Sum(-q_i*z_i/k,(i,1,m))
    # 求导数
    b_k = a_k.diff(z_i)
    assert isinstance(b_k, Sum)
    assert b_k == Sum(-q_i/k,(i,1,m))


def test_issue_10156():
    # 级数求和
    cx = Sum(2*y**2*x, (x, 1,3))
    # 表达式
    e = 2*y*Sum(2*cx*x**2, (x, 1, 9))
    assert e.factor() == \
        8*y**3*Sum(x, (x, 1, 3))*Sum(x**2, (x, 1, 9))


def test_issue_10973():
    # 检验级数的收敛性质
    assert Sum((-n + (n**3 + 1)**(S(1)/3))/log(n), (n, 1, oo)).is_convergent() is S.true


def test_issue_14129():
    # 符号定义
    x = Symbol('x', zero=False)
    # 求和求值
    assert Sum( k*x**k, (k, 0, n-1)).doit() == \
        Piecewise((n**2/2 - n/2, Eq(x, 1)), ((n*x*x**n -
            n*x**n - x*x**n + x)/(x - 1)**2, True))
    assert Sum( x**k, (k, 0, n-1)).doit() == \
        Piecewise((n, Eq(x, 1)), ((-x**n + 1)/(-x + 1), True))
    assert Sum( k*(x/y+x)**k, (k, 0, n-1)).doit() == \
        Piecewise((n*(n - 1)/2, Eq(x, y/(y + 1))),
        (x*(y + 1)*(n*x*y*(x + x/y)**(n - 1) +
        n*x*(x + x/y)**(n - 1) - n*y*(x + x/y)**(n - 1) -
        x*y*(x + x/y)**(n - 1) - x*(x + x/y)**(n - 1) + y)/
        (x*y + x - y)**2, True))


def test_issue_14112():
    # 检验级数的绝对收敛性质
    assert Sum((-1)**n/sqrt(n), (n, 1, oo)).is_absolutely_convergent() is S.false
    assert Sum((-1)**(2*n)/n, (n, 1, oo)).is_convergent() is S.false
    assert Sum((-2)**n + (-3)**n, (n, 1, oo)).is_convergent() is S.false


def test_issue_14219():
    # 矩阵定义
    A = diag(0, 2, -3)
    res = diag(1, 15, -20)
    assert Sum(A**n, (n, 0, 3)).doit() == res


def test_sin_times_absolutely_convergent():
    # 检验正弦函数乘以级数的绝对收敛性质
    assert Sum(sin(n) / n**3, (n, 1, oo)).is_convergent() is S.true
    assert Sum(sin(n) * log(n) / n**3, (n, 1, oo)).is_convergent() is S.true


def test_issue_14111():
    # 检验级数的收敛性质
    assert Sum(1/log(log(n)), (n, 22, oo)).is_convergent() is S.false


def test_issue_14484():
    # 检验级数的收敛性质
    assert Sum(sin(n)/log(log(n)), (n, 22, oo)).is_convergent() is S.false


def test_issue_14640():
    # 符号定义
    i, n = symbols("i n", integer=True)
    a, b, c = symbols("a b c", zero=False)

    # 求和求值
    assert Sum(a**-i/(a - b), (i, 0, n)).doit() == Sum(
        1/(a*a**i - a**i*b), (i, 0, n)).doit() == Piecewise(
            (n + 1, Eq(1/a, 1)),
            ((-a**(-n - 1) + 1)/(1 - 1/a), True))/(a - b)

    assert Sum((b*a**i - c*a**i)**-2, (i, 0, n)).doit() == Piecewise(
        (n + 1, Eq(a**(-2), 1)),
        ((-a**(-2*n - 2) + 1)/(1 - 1/a**2), True))/(b - c)**2

    s = Sum(i*(a**(n - i) - b**(n - i))/(a - b), (i, 0, n)).doit()
    assert not s.has(Sum)
    assert s.subs({a: 2, b: 3, n: 5}) == 122


def test_issue_15943():
    # 待添加测试代码
    pass
    #`
# 计算二项式求和并重写为 gamma 函数形式
s = Sum(binomial(n, k)*factorial(n - k), (k, 0, n)).doit().rewrite(gamma)
# 断言计算结果是否符合特定表达式
assert s == -E*(n + 1)*gamma(n + 1)*lowergamma(n + 1, 1)/gamma(n + 2) + E*gamma(n + 1)
# 断言化简后的计算结果是否等于特定表达式
assert s.simplify() == E*(factorial(n) - lowergamma(n + 1, 1))
# 定义一个测试函数，用于测试 Sum 类的 dummy_eq 方法
def test_Sum_dummy_eq():
    # 断言：Sum(x, (x, a, b)) 的 dummy_eq 方法不等于 1
    assert not Sum(x, (x, a, b)).dummy_eq(1)
    # 断言：Sum(x, (x, a, b)) 的 dummy_eq 方法不等于 Sum(x, (x, a, b), (a, 1, 2))
    assert not Sum(x, (x, a, b)).dummy_eq(Sum(x, (x, a, b), (a, 1, 2)))
    # 断言：Sum(x, (x, a, b)) 的 dummy_eq 方法不等于 Sum(x, (x, a, c))
    assert not Sum(x, (x, a, b)).dummy_eq(Sum(x, (x, a, c)))
    # 断言：Sum(x, (x, a, b)) 的 dummy_eq 方法等于 Sum(x, (x, a, b))
    assert Sum(x, (x, a, b)).dummy_eq(Sum(x, (x, a, b)))
    # 创建一个 Dummy 符号 d
    d = Dummy()
    # 断言：Sum(x, (x, a, d)) 的 dummy_eq 方法等于 Sum(x, (x, a, c))，其中 c 是符号
    assert Sum(x, (x, a, d)).dummy_eq(Sum(x, (x, a, c)), c)
    # 断言：Sum(x, (x, a, d)) 的 dummy_eq 方法不等于 Sum(x, (x, a, c))
    assert not Sum(x, (x, a, d)).dummy_eq(Sum(x, (x, a, c)))
    # 断言：Sum(x, (x, a, c)) 的 dummy_eq 方法等于 Sum(y, (y, a, c))
    assert Sum(x, (x, a, c)).dummy_eq(Sum(y, (y, a, c)))
    # 断言：Sum(x, (x, a, d)) 的 dummy_eq 方法等于 Sum(y, (y, a, c))，其中 c 是符号
    assert Sum(x, (x, a, d)).dummy_eq(Sum(y, (y, a, c)), c)
    # 断言：Sum(x, (x, a, d)) 的 dummy_eq 方法不等于 Sum(y, (y, a, c))
    assert not Sum(x, (x, a, d)).dummy_eq(Sum(y, (y, a, c)))


# 定义一个测试函数，用于测试 summation 函数的特定问题
def test_issue_15852():
    # 断言：summation(x**y*y, (y, -oo, oo)) 的结果等于 Sum(x**y*y, (y, -oo, oo)) 的计算结果
    assert summation(x**y*y, (y, -oo, oo)).doit() == Sum(x**y*y, (y, -oo, oo))


# 定义一个测试函数，用于测试 Sum 类的异常情况
def test_exceptions():
    # 创建 Sum 对象 S = Sum(x, (x, a, b))
    S = Sum(x, (x, a, b))
    # 使用 lambda 函数检查调用 S.change_index(x, x**2, y) 时是否抛出 ValueError 异常
    raises(ValueError, lambda: S.change_index(x, x**2, y))
    # 创建 Sum 对象 S = Sum(x, (x, a, b), (x, 1, 4))
    S = Sum(x, (x, a, b), (x, 1, 4))
    # 使用 lambda 函数检查调用 S.index(x) 时是否抛出 ValueError 异常
    raises(ValueError, lambda: S.index(x))
    # 创建 Sum 对象 S = Sum(x, (x, a, b), (y, 1, 4))
    S = Sum(x, (x, a, b), (y, 1, 4))
    # 使用 lambda 函数检查调用 S.reorder([x]) 时是否抛出 ValueError 异常
    raises(ValueError, lambda: S.reorder([x]))
    # 创建 Sum 对象 S = Sum(x, (x, y, b), (y, 1, 4))
    S = Sum(x, (x, y, b), (y, 1, 4))
    # 使用 lambda 函数检查调用 S.reorder_limit(0, 1) 时是否抛出 ReorderError 异常
    raises(ReorderError, lambda: S.reorder_limit(0, 1))
    # 创建 Sum 对象 S = Sum(x*y, (x, a, b), (y, 1, 4))
    S = Sum(x*y, (x, a, b), (y, 1, 4))
    # 使用 lambda 函数检查调用 S.is_convergent() 时是否抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: S.is_convergent())


# 定义一个测试函数，用于测试 Sum 和 Product 类在特定假设下的行为
def test_sumproducts_assumptions():
    # 创建整数且为正的符号 M 和 m
    M = Symbol('M', integer=True, positive=True)
    m = Symbol('m', integer=True)
    # 对 Sum 和 Product 类进行迭代测试
    for func in [Sum, Product]:
        # 断言：func(m, (m, -M, M)) 的 is_positive 属性为 None
        assert func(m, (m, -M, M)).is_positive is None
        # 断言：func(m, (m, -M, M)) 的 is_nonpositive 属性为 None
        assert func(m, (m, -M, M)).is_nonpositive is None
        # 断言：func(m, (m, -M, M)) 的 is_negative 属性为 None
        assert func(m, (m, -M, M)).is_negative is None
        # 断言：func(m, (m, -M, M)) 的 is_nonnegative 属性为 None
        assert func(m, (m, -M, M)).is_nonnegative is None
        # 断言：func(m, (m, -M, M)) 的 is_finite 属性为 True
        assert func(m, (m, -M, M)).is_finite is True

    # 创建整数且为非负的符号 m
    m = Symbol('m', integer=True, nonnegative=True)
    # 对 Sum 和 Product 类进行迭代测试
    for func in [Sum, Product]:
        # 断言：func(m, (m, 0, M)) 的 is_positive 属性为 None
        assert func(m, (m, 0, M)).is_positive is None
        # 断言：func(m, (m, 0, M)) 的 is_nonpositive 属性为 None
        assert func(m, (m, 0, M)).is_nonpositive is None
        # 断言：func(m, (m, 0, M)) 的 is_negative 属性为 False
        assert func(m, (m, 0, M)).is_negative is False
        # 断言：func(m, (m, 0, M)) 的 is_nonnegative 属性为 True
        assert func(m, (m, 0, M)).is_nonnegative is True
        # 断言：func(m, (m, 0, M)) 的 is_finite 属性为 True
        assert func(m, (m, 0, M)).is_finite is True

    # 创建整数且为正的符号 m
    m = Symbol('m', integer=True, positive=True)
    # 对 Sum 和 Product 类进行迭代测试
    for func in [Sum, Product]:
        # 断言：func(m, (m, 1, M)) 的 is_positive 属性为 True
        assert func(m, (m, 1, M)).is_positive is True
        # 断言：func(m, (m, 1, M)) 的 is_nonpositive 属性为 False
        assert func(m, (m, 1, M)).is_nonpositive is False
        # 断言：func(m, (m, 1, M)) 的 is_negative 属性为 False
        assert func(m, (m, 1, M)).is_negative is False
        # 断言：func(m, (m, 1, M)) 的 is_nonnegative 属性为 True
        assert func(m, (m, 1, M)).is_nonnegative is True
        # 断言：func(m, (m, 1, M)) 的 is_finite 属性为 True
        assert func(m, (m, 1, M)).is_finite is True

    # 创建整数且为负的符号 m
    m = Symbol('m', integer=True, negative
    # 断言：对于从 -M 到 0 的 m 的和，应该是非正的
    assert Sum(m, (m, -M, 0)).is_nonpositive is True
    # 断言：对于从 -M 到 0 的 m 的和，应该是负数
    assert Sum(m, (m, -M, 0)).is_negative is None
    # 断言：对于从 -M 到 0 的 m 的和，应该是非负数
    assert Sum(m, (m, -M, 0)).is_nonnegative is None
    # 断言：对于从 -M 到 0 的 m 的和，应该是有限的
    assert Sum(m, (m, -M, 0)).is_finite is True
    
    # 断言：对于从 0 到正无穷大的 m 的和，应该是正数
    assert Sum(2, (m, 0, oo)).is_positive is None
    # 断言：对于从 0 到正无穷大的 m 的和，应该是非正数
    assert Sum(2, (m, 0, oo)).is_nonpositive is None
    # 断言：对于从 0 到正无穷大的 m 的和，应该是负数
    assert Sum(2, (m, 0, oo)).is_negative is None
    # 断言：对于从 0 到正无穷大的 m 的和，应该是非负数
    assert Sum(2, (m, 0, oo)).is_nonnegative is None
    # 断言：对于从 0 到正无穷大的 m 的和，应该是有限的
    assert Sum(2, (m, 0, oo)).is_finite is None
    
    # 断言：对于从 0 到正无穷大的 m 的乘积，应该是正数
    assert Product(2, (m, 0, oo)).is_positive is None
    # 断言：对于从 0 到正无穷大的 m 的乘积，应该是非正数
    assert Product(2, (m, 0, oo)).is_nonpositive is None
    # 断言：对于从 0 到正无穷大的 m 的乘积，应该是负数
    assert Product(2, (m, 0, oo)).is_negative is False
    # 断言：对于从 0 到正无穷大的 m 的乘积，应该是非负数
    assert Product(2, (m, 0, oo)).is_nonnegative is None
    # 断言：对于从 0 到正无穷大的 m 的乘积，应该是有限的
    assert Product(2, (m, 0, oo)).is_finite is None
    
    # 断言：对于从 M 到 M-1 的 x 的乘积，应该是正数
    assert Product(0, (x, M, M-1)).is_positive is True
    # 断言：对于从 M 到 M-1 的 x 的乘积，应该是有限的
    assert Product(0, (x, M, M-1)).is_finite is True
# 定义一个测试函数，用于测试一些关于符号和求和的数学表达式
def test_expand_with_assumptions():
    # 定义符号 M，x，m 分别为整数、正数和非负数符号
    M = Symbol('M', integer=True, positive=True)
    x = Symbol('x', positive=True)
    m = Symbol('m', nonnegative=True)
    # 断言对数函数应用在 Product 中的展开结果等于对数应用在 Sum 中的展开结果
    assert log(Product(x**m, (m, 0, M))).expand() == Sum(m*log(x), (m, 0, M))
    # 断言对数函数应用在 exp(Product) 中的展开结果等于对数应用在 Sum(x**m) 中的展开结果
    assert log(Product(exp(x**m), (m, 0, M))).expand() == Sum(x**m, (m, 0, M))
    # 断言对数函数应用在 Product 中并使用 rewrite 方法转换为 Sum 的展开结果等于对数应用在 Sum 中的展开结果
    assert log(Product(x**m, (m, 0, M))).rewrite(Sum).expand() == Sum(m*log(x), (m, 0, M))
    # 断言对数函数应用在 exp(Product) 中并使用 rewrite 方法转换为 Sum 的展开结果等于对数应用在 Sum(x**m) 中的展开结果
    assert log(Product(exp(x**m), (m, 0, M))).rewrite(Sum).expand() == Sum(x**m, (m, 0, M))

    # 定义符号 n，i，j，x，y 分别为非负整数、正整数，以及正数符号
    n = Symbol('n', nonnegative=True)
    i, j = symbols('i,j', positive=True, integer=True)
    x, y = symbols('x,y', positive=True)
    # 断言对数函数应用在 Product(x**i*y**j) 中的展开结果等于对数应用在 Sum(i*log(x) + j*log(y)) 中的展开结果
    assert log(Product(x**i*y**j, (i, 1, n), (j, 1, m))).expand() \
        == Sum(i*log(x) + j*log(y), (i, 1, n), (j, 1, m))

    # 定义符号 m，x
    m = Symbol('m', nonnegative=True, integer=True)
    # 创建一个 Sum 对象 s，表示 x**m 的和，范围是从 m=0 到 m=M
    s = Sum(x**m, (m, 0, M))
    # 使用 rewrite 方法将 Sum 转换为 Product，然后检查结果中是否包含 Product 符号
    s_as_product = s.rewrite(Product)
    assert s_as_product.has(Product)
    # 断言转换后的结果等于 log(Product(exp(x**m), (m, 0, M)))，即对数应用在 exp(Product) 中的展开结果
    assert s_as_product == log(Product(exp(x**m), (m, 0, M)))
    # 断言对转换后的结果应用 expand 后等于原始的 Sum 对象 s
    assert s_as_product.expand() == s
    # 将符号 M 替换为具体值 5，创建一个新的 Sum 对象 s5
    s5 = s.subs(M, 5)
    # 使用 rewrite 方法将 Sum 对象 s5 转换为 Product
    s5_as_product = s5.rewrite(Product)
    assert s5_as_product.has(Product)
    # 断言转换后的结果 doit 后应用 expand 等于原始的 Sum 对象 s5
    assert s5_as_product.doit().expand() == s5.doit()


# 定义一个测试函数，用于测试 Sum 对象的 has_finite_limits 方法
def test_has_finite_limits():
    x = Symbol('x')
    # 断言对 Sum(1, (x, 1, 9)) 的 has_finite_limits 结果为 True
    assert Sum(1, (x, 1, 9)).has_finite_limits is True
    # 断言对 Sum(1, (x, 1, oo)) 的 has_finite_limits 结果为 False
    assert Sum(1, (x, 1, oo)).has_finite_limits is False
    # 定义符号 M，断言对 Sum(1, (x, 1, M)) 的 has_finite_limits 结果为 None
    M = Symbol('M')
    assert Sum(1, (x, 1, M)).has_finite_limits is None
    # 定义符号 M 为正数，断言对 Sum(1, (x, 1, M)) 的 has_finite_limits 结果为 True
    M = Symbol('M', positive=True)
    assert Sum(1, (x, 1, M)).has_finite_limits is True
    # 定义符号 x 为正数，M 为符号，断言对 Sum(1, (x, 1, M)) 的 has_finite_limits 结果为 True
    x = Symbol('x', positive=True)
    M = Symbol('M')
    assert Sum(1, (x, 1, M)).has_finite_limits is True
    # 断言对 Sum(1, (x, 1, M), (y, -oo, oo)) 的 has_finite_limits 结果为 False
    assert Sum(1, (x, 1, M), (y, -oo, oo)).has_finite_limits is False


# 定义一个测试函数，用于测试 Sum 对象的 has_reversed_limits 方法
def test_has_reversed_limits():
    # 断言对 Sum(1, (x, 1, 1)) 的 has_reversed_limits 结果为 False
    assert Sum(1, (x, 1, 1)).has_reversed_limits is False
    # 断言对 Sum(1, (x, 1, 9)) 的 has_reversed_limits 结果为 False
    assert Sum(1, (x, 1, 9)).has_reversed_limits is False
    # 断言对 Sum(1, (x, 1, -9)) 的 has_reversed_limits 结果为 True
    assert Sum(1, (x, 1, -9)).has_reversed_limits is True
    # 断言对 Sum(1, (x, 1, 0)) 的 has_reversed_limits 结果为 True
    assert Sum(1, (x, 1, 0)).has_reversed_limits is True
    # 断言对 Sum(1, (x, 1, oo)) 的 has_reversed_limits 结果为 False
    assert Sum(1, (x, 1, oo)).has_reversed_limits is False
    # 定义符号 M，断言对 Sum(1, (x, 1, M)) 的 has_reversed_limits 结果为 None
    M = Symbol('M')
    assert Sum(1, (x, 1, M)).has_reversed_limits is None
    # 定义符号 M 为正整数，断言对 Sum(1, (x, 1, M)) 的 has_reversed_limits 结果为 False
    M = Symbol('M', positive=True, integer=True)
    assert Sum(1, (x, 1, M)).has_reversed_limits is False
    # 断言对 Sum(1, (x, 1, M), (y, -oo, oo)) 的 has_reversed_limits 结果为 False
    assert Sum(1, (x, 1, M), (y, -oo, oo)).has_reversed_limits is False
    # 定义符号 M 为负数，断言对 Sum(1, (x, 1, M)) 的 has_reversed_limits 结果为 True
    M = Symbol('M', negative=True)
    assert Sum(1, (x, 1, M)).has_reversed_limits is True
    # 断言对 Sum(1, (x, 1, M), (y, -oo, oo)) 的 has_reversed_limits 结果为 True
    assert Sum(1, (x, 1, M), (y, -oo, oo)).has_reversed_limits is True
    # 断言对 Sum(1, (x, oo, oo)) 的 has_reversed_limits 结果为 None
    assert Sum(1, (x, oo, oo)).has_reversed_limits is None


# 定义一个测试函数，用于测试 Sum 对象的 has_empty_sequence 方法
def test_has_empty_sequence():
    # 断言对 Sum(1, (x, 1, 1)) 的 has_empty_sequence 结果为 False
    assert Sum(1,
    # 断言语句，用于检查条件是否为真，如果条件为假则会触发 AssertionError
    assert Sum(1, (x, oo, oo)).has_empty_sequence is False
# 测试空序列情况下的乘积计算
def test_empty_sequence():
    # 断言：无穷大范围内两个符号 x 和 y 的乘积的结果等于 1
    assert Product(x*y, (x, -oo, oo), (y, 1, 0)).doit() == 1
    # 断言：y 从 1 到 0，x 从负无穷到正无穷的乘积的结果等于 1
    assert Product(x*y, (y, 1, 0), (x, -oo, oo)).doit() == 1
    # 断言：无穷大范围内 x 的和的结果等于 0，y 从 1 到 0
    assert Sum(x, (x, -oo, oo), (y, 1, 0)).doit() == 0
    # 断言：y 从 1 到 0，x 从负无穷到正无穷的和的结果等于 0
    assert Sum(x, (y, 1, 0), (x, -oo, oo)).doit() == 0


# 测试问题编号 8016
def test_issue_8016():
    k = Symbol('k', integer=True)
    n, m = symbols('n, m', integer=True, positive=True)
    # 定义求和表达式 s
    s = Sum(binomial(m, k)*binomial(m, n - k)*(-1)**k, (k, 0, n))
    # 断言：s 简化后的结果等于给定公式
    assert s.doit().simplify() == \
        cos(pi*n/2)*gamma(m + 1)/gamma(n/2 + 1)/gamma(m - n/2 + 1)


# 测试问题编号 14313
def test_issue_14313():
    # 断言：级数 Sum(Half**(floor(n/2)), (n, 1, oo)) 是收敛的
    assert Sum(S.Half**floor(n/2), (n, 1, oo)).is_convergent()


# 测试问题编号 14563
def test_issue_14563():
    # 断言：1 除以 Sum(1, (x, 0, 1)) 的余数等于 1
    assert 1 % Sum(1, (x, 0, 1)) == 1


# 测试问题编号 16735
def test_issue_16735():
    # 断言：级数 Sum(5**n/gamma(n+1), (n, 1, oo)) 是收敛的，并且为真值 S.true
    assert Sum(5**n/gamma(n+1), (n, 1, oo)).is_convergent() is S.true


# 测试问题编号 14871
def test_issue_14871():
    # 断言：级数 Sum((Rational(1, 10))**n*rf(0, n)/factorial(n), (n, 0, oo)) 经过重写并计算的结果等于 1
    assert Sum((Rational(1, 10))**n*rf(0, n)/factorial(n), (n, 0, oo)).rewrite(factorial).doit() == 1


# 测试问题编号 17165
def test_issue_17165():
    n = symbols("n", integer=True)
    x = symbols('x')
    s = (x*Sum(x**n, (n, -1, oo)))
    ssimp = s.doit().simplify()
    # 断言：经过简化的表达式 ssimp 等于给定的分段函数表达式
    assert ssimp == Piecewise((-1/(x - 1), (x > -1) & (x < 1)),
                              (x*Sum(x**n, (n, -1, oo)), True)), ssimp
    # 断言：经过简化的 ssimp 等于其自身
    assert ssimp.simplify() == ssimp


# 测试问题编号 19379
def test_issue_19379():
    # 断言：级数 Sum(factorial(n)/factorial(n + 2), (n, 1, oo)) 是收敛的，并且为真值 S.true
    assert Sum(factorial(n)/factorial(n + 2), (n, 1, oo)).is_convergent() is S.true


# 测试问题编号 20777
def test_issue_20777():
    # 断言：级数 Sum(exp(x*sin(n/m)), (n, 1, m)) 经过计算等于其自身
    assert Sum(exp(x*sin(n/m)), (n, 1, m)).doit() == Sum(exp(x*sin(n/m)), (n, 1, m))


# 测试具有继承属性的虚拟符号
def test__dummy_with_inherited_properties_concrete():
    x = Symbol('x')

    from sympy.core.containers import Tuple
    # 调用 _dummy_with_inherited_properties_concrete 函数
    d = _dummy_with_inherited_properties_concrete(Tuple(x, 0, 5))
    # 断言：d 具有属性 is_real、is_integer、is_nonnegative、is_extended_nonnegative
    assert d.is_real
    assert d.is_integer
    assert d.is_nonnegative
    assert d.is_extended_nonnegative

    d = _dummy_with_inherited_properties_concrete(Tuple(x, 1, 9))
    # 断言：d 具有属性 is_real、is_integer、is_positive，并且 is_odd 为 None
    assert d.is_real
    assert d.is_integer
    assert d.is_positive
    assert d.is_odd is None

    d = _dummy_with_inherited_properties_concrete(Tuple(x, -5, 5))
    # 断言：d 具有属性 is_real、is_integer，is_positive、is_extended_nonnegative、is_odd 都为 None
    assert d.is_real
    assert d.is_integer
    assert d.is_positive is None
    assert d.is_extended_nonnegative is None
    assert d.is_odd is None

    d = _dummy_with_inherited_properties_concrete(Tuple(x, -1.5, 1.5))
    # 断言：d 具有属性 is_real，is_integer 为 None，其余都为 None
    assert d.is_real
    assert d.is_integer is None
    assert d.is_positive is None
    assert d.is_extended_nonnegative is None

    N = Symbol('N', integer=True, positive=True)
    d = _dummy_with_inherited_properties_concrete(Tuple(x, 2, N))
    # 断言：d 具有属性 is_real、is_positive、is_integer
    assert d.is_real
    assert d.is_positive
    assert d.is_integer

    # 断言：如果没有添加任何假设，则返回 None
    N = Symbol('N', integer=True, positive=True)
    d = _dummy_with_inherited_properties_concrete(Tuple(N, 2, 4))
    assert d is None

    x = Symbol('x', negative=True)
    # 断言：_dummy_with_inherited_properties_concrete 函数在不一致的假设下会引发异常 InconsistentAssumptions
    raises(InconsistentAssumptions,
           lambda: _dummy_with_inherited_properties_concrete(Tuple(x, 1, 5)))
# 定义测试函数，用于测试矩阵符号的求和在数值限制下的结果
def test_matrixsymbol_summation_numerical_limits():
    # 定义一个3x3的矩阵符号A
    A = MatrixSymbol('A', 3, 3)
    # 定义一个整数符号n
    n = Symbol('n', integer=True)

    # 断言：对A的幂次求和，n从0到2，计算结果应该等于单位矩阵Identity(3)加上A加上A的平方
    assert Sum(A**n, (n, 0, 2)).doit() == Identity(3) + A + A**2
    # 断言：对A求和，n从0到2，计算结果应该等于3乘以A
    assert Sum(A, (n, 0, 2)).doit() == 3*A
    # 断言：对n乘以A求和，n从0到2，计算结果应该等于3乘以A
    assert Sum(n*A, (n, 0, 2)).doit() == 3*A

    # 定义一个3x3的矩阵B
    B = Matrix([[0, n, 0], [-1, 0, 0], [0, 0, 2]])
    # 预期的结果是A与B的和，n从0到3的求和结果应该等于ans
    ans = Matrix([[0, 6, 0], [-4, 0, 0], [0, 0, 8]]) + 4*A
    assert Sum(A+B, (n, 0, 3)).doit() == ans
    # 预期的结果是A乘以B，n从0到3的求和结果应该等于ans
    ans = A*Matrix([[0, 6, 0], [-4, 0, 0], [0, 0, 8]])
    assert Sum(A*B, (n, 0, 3)).doit() == ans

    # 预期的结果是A的n次方乘以B的n次方，n从1到3的求和结果应该等于ans
    ans = (A**2*Matrix([[-2, 0, 0], [0,-2, 0], [0, 0, 4]]) +
           A**3*Matrix([[0, -9, 0], [3, 0, 0], [0, 0, 8]]) +
           A*Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 2]]))
    assert Sum(A**n*B**n, (n, 1, 3)).doit() == ans


# 定义测试函数，用于测试GitHub上的Issue 21651
def test_issue_21651():
    # 定义一个整数符号i
    i = Symbol('i')
    # a是一个求和表达式，从1到2，对2*2**(-i)的整数部分求和
    a = Sum(floor(2*2**(-i)), (i, S.One, 2))
    # 断言：计算a的结果应该等于1
    assert a.doit() == S.One


# 定义测试函数，用于测试矩阵符号的求和在符号限制下的结果（预期失败）
@XFAIL
def test_matrixsymbol_summation_symbolic_limits():
    # 定义一个正整数符号N
    N = Symbol('N', integer=True, positive=True)

    # 定义一个3x3的矩阵符号A
    A = MatrixSymbol('A', 3, 3)
    # 定义一个整数符号n
    n = Symbol('n', integer=True)
    # 断言：对A求和，n从0到N，计算结果应该等于(N+1)*A
    assert Sum(A, (n, 0, N)).doit() == (N+1)*A
    # 断言：对n乘以A求和，n从0到N，计算结果应该等于(N**2/2 + N/2)*A


# 定义测试函数，用于测试通过残余求和的结果
def test_summation_by_residues():
    # 定义一个符号x
    x = Symbol('x')

    # 从Nakhle H. Asmar, Loukas Grafakos的《Complex Analysis with Applications》中选取的例子
    # 断言：计算1 / (x**2 + 1)，在x从负无穷到正无穷的残余求和应该等于pi/tanh(pi)
    assert eval_sum_residue(1 / (x**2 + 1), (x, -oo, oo)) == pi/tanh(pi)
    # 断言：计算1 / x**6，在x从1到正无穷的残余求和应该等于pi**6/945
    assert eval_sum_residue(1 / x**6, (x, S(1), oo)) == pi**6/945
    # 断言：计算1 / (x**2 + 9)，在x从负无穷到正无穷的残余求和应该等于pi/(3*tanh(3*pi))
    assert eval_sum_residue(1 / (x**2 + 9), (x, -oo, oo)) == pi/(3*tanh(3*pi))
    # 断言：计算1 / (x**2 + 1)**2，在x从负无穷到正无穷的残余求和应该等于指定的表达式
    assert eval_sum_residue(1 / (x**2 + 1)**2, (x, -oo, oo)).cancel() == \
        (-pi**2*tanh(pi)**2 + pi*tanh(pi) + pi**2)/(2*tanh(pi)**2)
    # 断言：计算x**2 / (x**2 + 1)**2，在x从负无穷到正无穷的残余求和应该等于指定的表达式
    assert eval_sum_residue(x**2 / (x**2 + 1)**2, (x, -oo, oo)).cancel() == \
        (-pi**2 + pi*tanh(pi) + pi**2*tanh(pi)**2)/(2*tanh(pi)**2)
    # 断言：计算1 / (4*x**2 - 1)，在x从负无穷到正无穷的残余求和应该等于0
    assert eval_sum_residue(1 / (4*x**2 - 1), (x, -oo, oo)) == 0
    # 断言：计算x**2 / (x**2 - S(1)/4)**2，在x从负无穷到正无穷的残余求和应该等于pi**2/2
    assert eval_sum_residue(x**2 / (x**2 - S(1)/4)**2, (x, -oo, oo)) == pi**2/2
    # 断言：计算1 / (4*x**2 - 1)**2，在x从负无穷到正无穷的残余求和应该等于pi**2/8
    assert eval_sum_residue(1 / (4*x**2 - 1)**2, (x, -oo, oo)) == pi**2/8
    # 断言：计算1 / ((x - S(1)/2)**2 + 1)，在x从负无穷到正无穷的残余求和应该等于pi*tanh(pi)
    assert eval_sum_residue(1 / ((x - S(1)/2)**2 + 1), (x, -oo, oo)) == pi*tanh(pi)
    # 断言：计算1 / x**2，在x从1到正无穷的残余求和应该等于pi**2/6
    assert eval_sum_residue(1 / x**2, (x, S(1), oo)) == pi**2/6
    # 断言：计算1 / x**4，在x从1到正无穷的残余求和应该等于pi**4/90
    assert eval_sum_residue(1 / x**4, (x, S(1), oo)) == pi**4/90
    # 断言：计算1 / x**2 / (x**2 + 4)，在x从1到正无穷的
    # 对于 (-1)**x / (x**2 + 1) 在区间 (1, ∞) 上的残余求和，期望结果为 -1/2 + π/(2*sinh(π))
    assert eval_sum_residue((-1)**x / (x**2 + 1), (x, S(1), oo)) == \
        -S(1)/2 + pi/(2*sinh(pi))
    
    # 对于 (-1)**x / (x**2 + 1) 在区间 (-1, ∞) 上的残余求和，期望结果为 π/(2*sinh(π))
    assert eval_sum_residue((-1)**x / (x**2 + 1), (x, S(-1), oo)) == \
        pi/(2*sinh(pi))
    
    # 对于 1 / (x**2 + 2*x + 2) 在区间 (-1, ∞) 上的残余求和，期望结果为 1/2 + π/(2*tanh(π))
    assert eval_sum_residue(1 / (x**2 + 2*x + 2), (x, S(-1), oo)) == S(1)/2 + pi/(2*tanh(pi))
    
    # 对于 1 / (x**2 + 4*x + 5) 在区间 (-2, ∞) 上的残余求和，期望结果为 1/2 + π/(2*tanh(π))
    assert eval_sum_residue(1 / (x**2 + 4*x + 5), (x, S(-2), oo)) == S(1)/2 + pi/(2*tanh(pi))
    
    # 对于 1 / (x**2 - 2*x + 2) 在区间 (1, ∞) 上的残余求和，期望结果为 1/2 + π/(2*tanh(π))
    assert eval_sum_residue(1 / (x**2 - 2*x + 2), (x, S(1), oo)) == S(1)/2 + pi/(2*tanh(pi))
    
    # 对于 1 / (x**2 - 4*x + 5) 在区间 (2, ∞) 上的残余求和，期望结果为 1/2 + π/(2*tanh(π))
    assert eval_sum_residue(1 / (x**2 - 4*x + 5), (x, S(2), oo)) == S(1)/2 + pi/(2*tanh(pi))
    
    # 对于 (-1)**x * -1 / (x**2 + 2*x + 2) 在区间 (-1, ∞) 上的残余求和，期望结果为 1/2 + π/(2*sinh(π))
    assert eval_sum_residue((-1)**x * -1 / (x**2 + 2*x + 2), (x, S(-1), oo)) ==  S(1)/2 + pi/(2*sinh(pi))
    
    # 对于 (-1)**x * -1 / (x**2 - 2*x + 2) 在区间 (1, ∞) 上的残余求和，期望结果为 1/2 + π/(2*sinh(π))
    assert eval_sum_residue((-1)**x * -1 / (x**2 - 2*x + 2), (x, S(1), oo)) == S(1)/2 + pi/(2*sinh(pi))
    
    # 对于 1 / x**2 在区间 (2, ∞) 上的残余求和，期望结果为 -1 + π**2/6
    assert eval_sum_residue(1 / x**2, (x, S(2), oo)) == -1 + pi**2/6
    
    # 对于 1 / x**2 在区间 (3, ∞) 上的残余求和，期望结果为 -5/4 + π**2/6
    assert eval_sum_residue(1 / x**2, (x, S(3), oo)) == -S(5)/4 + pi**2/6
    
    # 对于 (-1)**x / x**2 在区间 (1, ∞) 上的残余求和，期望结果为 -π**2/12
    assert eval_sum_residue((-1)**x / x**2, (x, S(1), oo)) == -pi**2/12
    
    # 对于 (-1)**x / x**2 在区间 (2, ∞) 上的残余求和，期望结果为 1 - π**2/12
    assert eval_sum_residue((-1)**x / x**2, (x, S(2), oo)) == 1 - pi**2/12
@slow
# 定义一个被标记为慢速的测试函数，通常用于执行较耗时的测试任务
def test_summation_by_residues_failing():
    # 创建一个符号变量 x
    x = Symbol('x')

    # 由于残余计算中的 bug，以下断言会失败
    assert eval_sum_residue(x**2 / (x**4 + 1), (x, S(1), oo))
    # 以下断言应该不等于 0，因为存在 bug
    assert eval_sum_residue(1 / ((x - 1)*(x - 2) + 1), (x, -oo, oo)) != 0


def test_process_limits():
    # 从 sympy.concrete.expr_with_limits 导入 _process_limits 函数
    from sympy.concrete.expr_with_limits import _process_limits

    # 下面两行应该抛出 ValueError，因为传入的不是一个 (x, ...) 的形式
    raises(ValueError, lambda: _process_limits(
        Range(3), discrete=True))
    raises(ValueError, lambda: _process_limits(
        Range(3), discrete=False))
    
    # 下面两行应该抛出 ValueError，因为应该传入一个 (x, union) 的形式
    union = Or(x < 1, x > 3).as_set()
    raises(ValueError, lambda: _process_limits(
        union, discrete=True))
    raises(ValueError, lambda: _process_limits(
        union, discrete=False))

    # 如果不需要触发错误，下面的断言应该通过
    assert _process_limits((x, 1, 2)) == ([(x, 1, 2)], 1)

    # 下面的断言用于检查 S.Reals 是否是 Interval 类的实例，以便在 _process_limits 中检测 Reals
    assert isinstance(S.Reals, Interval)

    C = Integral  # 连续限制
    # 下面的断言验证等价性，用于检测在 _process_limits 中是否存在 Reals
    assert C(x, x >= 5) == C(x, (x, 5, oo))
    assert C(x, x < 3) == C(x, (x, -oo, 3))
    ans = C(x, (x, 0, 3))
    assert C(x, And(x >= 0, x < 3)) == ans
    assert C(x, (x, Interval.Ropen(0, 3))) == ans
    raises(TypeError, lambda: C(x, (x, Range(3))))

    # 离散限制
    for D in (Sum, Product):
        r, ans = Range(3, 10, 2), D(2*x + 3, (x, 0, 3))
        assert D(x, (x, r)) == ans
        assert D(x, (x, r.reversed)) == ans
        r, ans = Range(3, oo, 2), D(2*x + 3, (x, 0, oo))
        assert D(x, (x, r)) == ans
        assert D(x, (x, r.reversed)) == ans
        r, ans = Range(-oo, 5, 2), D(3 - 2*x, (x, 0, oo))
        assert D(x, (x, r)) == ans
        assert D(x, (x, r.reversed)) == ans
        raises(TypeError, lambda: D(x, x > 0))
        raises(ValueError, lambda: D(x, Interval(1, 3)))
        raises(NotImplementedError, lambda: D(x, (x, union)))


def test_pr_22677():
    # 创建整数符号变量 b
    b = Symbol('b', integer=True, positive=True)
    # 验证在 p 为 0，q 为 1，n 为 3 时，求和结果为 3
    assert Sum(1/x**2,(x, 0, b)).doit() == Sum(x**(-2), (x, 0, b))
    assert Sum(1/(x - b)**2,(x, 0, b-1)).doit() == Sum(
        (-b + x)**(-2), (x, 0, b - 1))


def test_issue_23952():
    # 创建实数非负符号变量 p, q
    p, q = symbols("p q", real=True, nonnegative=True)
    # 创建整数非负符号变量 k1, k2，以及正整数符号变量 n
    k1, k2 = symbols("k1 k2", integer=True, nonnegative=True)
    n = Symbol("n", integer=True, positive=True)
    # 表达式 expr 用于计算绝对值求和
    expr = Sum(abs(k1 - k2)*p**k1 *(1 - q)**(n - k2),
        (k1, 0, n), (k2, 0, n))
    # 验证在 p 为 0，q 为 1，n 为 3 时，求和结果为 3
    assert expr.subs(p,0).subs(q,1).subs(n, 3).doit() == 3
```