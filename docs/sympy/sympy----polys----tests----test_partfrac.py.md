# `D:\src\scipysrc\sympy\sympy\polys\tests\test_partfrac.py`

```
# 导入符号计算相关的模块和函数
from sympy.polys.partfrac import (
    apart_undetermined_coeffs,
    apart,
    apart_list, assemble_partfrac_list
)

# 导入具体的数学表达式、函数和符号
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (E, I, Rational, pi, all_close)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)

# 导入数学函数，如平方根
from sympy.functions.elementary.miscellaneous import sqrt

# 导入矩阵相关函数
from sympy.matrices.dense import Matrix

# 导入多项式相关工具函数
from sympy.polys.polytools import (Poly, factor)

# 导入有理函数相关工具函数
from sympy.polys.rationaltools import together

# 导入根式和相关工具函数
from sympy.polys.rootoftools import RootSum

# 导入测试相关的函数和模块
from sympy.testing.pytest import raises, XFAIL

# 导入常见的符号变量
from sympy.abc import x, y, a, b, c


# 定义测试函数 test_apart，用于测试分式分解函数 apart
def test_apart():
    # 测试对常数 1 的分解，应返回其本身
    assert apart(1) == 1
    # 测试对常数 1 关于变量 x 的分解，应返回其本身
    assert apart(1, x) == 1

    # 定义两个表达式 f 和 g，测试分解 (x**2 + 1)/(x + 1)
    f, g = (x**2 + 1)/(x + 1), 2/(x + 1) + x - 1
    # 测试在不完全展开模式下的分解结果是否正确
    assert apart(f, full=False) == g
    # 测试在完全展开模式下的分解结果是否正确
    assert apart(f, full=True) == g

    # 定义两个表达式 f 和 g，测试分解 1/((x + 2)(x + 1))
    f, g = 1/(x + 2)/(x + 1), 1/(1 + x) - 1/(2 + x)
    # 测试在不完全展开模式下的分解结果是否正确
    assert apart(f, full=False) == g
    # 测试在完全展开模式下的分解结果是否正确
    assert apart(f, full=True) == g

    # 定义两个表达式 f 和 g，测试分解 1/((x + 1)(x + 5))
    f, g = 1/(x + 1)/(x + 5), -1/(5 + x)/4 + 1/(1 + x)/4
    # 测试在不完全展开模式下的分解结果是否正确
    assert apart(f, full=False) == g
    # 测试在完全展开模式下的分解结果是否正确
    assert apart(f, full=True) == g

    # 测试对表达式 (E*x + 2)/(x - pi)*(x - 1) 的分解
    assert apart((E*x + 2)/(x - pi)*(x - 1), x) == \
        2 - E + E*pi + E*x + (E*pi + 2)*(pi - 1)/(x - pi)

    # 测试对方程 (x**2 + 1)/(x + 1) = x 的分解
    assert apart(Eq((x**2 + 1)/(x + 1), x), x) == Eq(x - 1 + 2/(x + 1), x)

    # 测试对 x/2 关于变量 y 的分解，应返回 x/2 本身
    assert apart(x/2, y) == x/2

    # 定义两个表达式 f 和 g，测试分解 (x+y)/(2*x - y)
    f, g = (x+y)/(2*x - y), Rational(3, 2)*y/(2*x - y) + S.Half
    # 测试在不完全展开模式下的分解结果是否正确
    assert apart(f, x, full=False) == g
    # 测试在完全展开模式下的分解结果是否正确
    assert apart(f, x, full=True) == g

    # 定义两个表达式 f 和 g，测试分解 (x+y)/(2*x - y)
    f, g = (x+y)/(2*x - y), 3*x/(2*x - y) - 1
    # 测试在不完全展开模式下的分解结果是否正确
    assert apart(f, y, full=False) == g
    # 测试在完全展开模式下的分解结果是否正确
    assert apart(f, y, full=True) == g

    # 测试对未实现的情况抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: apart(1/(x + 1)/(y + 2)))


# 定义测试函数 test_apart_matrix，用于测试矩阵的分解
def test_apart_matrix():
    # 创建一个 2x2 的矩阵 M，其元素为 1/((x + i + 1)(x + j))
    M = Matrix(2, 2, lambda i, j: 1/(x + i + 1)/(x + j))

    # 断言分解矩阵 M 的结果是否正确
    assert apart(M) == Matrix([
        [1/x - 1/(x + 1), (x + 1)**(-2)],
        [1/(2*x) - (S.Half)/(x + 2), 1/(x + 1) - 1/(x + 2)],
    ])


# 定义测试函数 test_apart_symbolic，用于测试符号表达式的分解
def test_apart_symbolic():
    # 定义符号表达式 f 和 g
    f = a*x**4 + (2*b + 2*a*c)*x**3 + (4*b*c - a**2 + a*c**2)*x**2 + \
        (-2*a*b + 2*b*c**2)*x - b**2
    g = a**2*x**4 + (2*a*b + 2*c*a**2)*x**3 + (4*a*b*c + b**2 +
        a**2*c**2)*x**2 + (2*c*b**2 + 2*a*b*c**2)*x + b**2*c**2

    # 测试对符号表达式 f/g 关于变量 x 的分解结果是否正确
    assert apart(f/g, x) == 1/a - 1/(x + c)**2 - b**2/(a*(a*x + b)**2)

    # 测试对符号表达式 1/((x + a)(x + b)(x + c)) 关于变量 x 的分解结果是否正确
    assert apart(1/((x + a)*(x + b)*(x + c)), x) == \
        1/((a - c)*(b - c)*(c + x)) - 1/((a - b)*(b - c)*(b + x)) + \
        1/((a - b)*(a - c)*(a + x))


# 定义内部函数 _make_extension_example，用于示例扩展问题
def _make_extension_example():
    # 导入 Mul 类
    from sympy.core import Mul

    # 定义函数 mul2，用于乘法操作的特殊处理
    def mul2(expr):
        # 2-arg mul hack...
        return Mul(2, expr, evaluate=False)

    # 定义表达式 f 和 g，示例扩展问题中的具体表达式
    f = ((x**2 + 1)**3/((x - 1)**2*(x + 1)**2*(-x**2 + 2*x + 1)*(x**2 + 2*x - 1)))
    g = (1/mul2(x - sqrt(2) + 1)
       - 1/mul2(x - sqrt(2) - 1)
       + 1/mul2(x + 1 + sqrt(2))
       - 1/mul2(x - 1 + sqrt(2))
       + 1/mul2((x + 1)**2)
       + 1/mul2((x - 1)**2))
    # 返回函数 f 和 g 的结果
    return f, g
# 定义一个测试函数，用于测试 sympy 的 apart 函数在不同情况下的表现
def test_apart_extension():
    # 定义一个符号表达式 f
    f = 2/(x**2 + 1)
    # 定义一个符号表达式 g
    g = I/(x + I) - I/(x - I)

    # 断言使用 apart 函数对 f 进行分解，使用扩展 I 后应该得到 g
    assert apart(f, extension=I) == g
    # 断言使用 apart 函数对 f 进行高斯分解后应该得到 g
    assert apart(f, gaussian=True) == g

    # 定义一个符号表达式 f
    f = x/((x - 2)*(x + I))

    # 断言使用 apart 函数对 f 进行分解，然后再合并因式应该得到 f
    assert factor(together(apart(f)).expand()) == f

    # 调用 _make_extension_example 函数返回 f, g
    f, g = _make_extension_example()

    # 使用 dotprodsimp 上下文，断言使用 apart 函数对 f 进行分解，使用扩展 {sqrt(2)} 后应该得到 g
    with dotprodsimp(True):
        assert apart(f, x, extension={sqrt(2)}) == g


# 定义一个测试函数，用于测试 apart 函数在特定扩展情况下的表现
def test_apart_extension_xfail():
    # 调用 _make_extension_example 函数返回 f, g
    f, g = _make_extension_example()
    # 断言使用 apart 函数对 f 进行分解，使用扩展 {sqrt(2)} 后应该得到 g
    assert apart(f, x, extension={sqrt(2)}) == g


# 定义一个测试函数，用于测试 apart 函数在 full 模式下的表现
def test_apart_full():
    # 定义一个符号表达式 f
    f = 1/(x**2 + 1)

    # 断言使用 apart 函数对 f 进行部分分解（full=False）应该得到 f 本身
    assert apart(f, full=False) == f
    # 断言使用 apart 函数对 f 进行完全分解（full=True）应该得到特定的表达式
    assert apart(f, full=True).dummy_eq(
        -RootSum(x**2 + 1, Lambda(a, a/(x - a)), auto=False)/2)

    # 定义一个符号表达式 f
    f = 1/(x**3 + x + 1)

    # 断言使用 apart 函数对 f 进行部分分解（full=False）应该得到 f 本身
    assert apart(f, full=False) == f
    # 断言使用 apart 函数对 f 进行完全分解（full=True）应该得到特定的表达式
    assert apart(f, full=True).dummy_eq(
        RootSum(x**3 + x + 1,
        Lambda(a, (a**2*Rational(6, 31) - a*Rational(9, 31) + Rational(4, 31))/(x - a)), auto=False))

    # 定义一个符号表达式 f
    f = 1/(x**5 + 1)

    # 断言使用 apart 函数对 f 进行部分分解（full=False）应该得到特定的表达式
    assert apart(f, full=False) == \
        (Rational(-1, 5))*((x**3 - 2*x**2 + 3*x - 4)/(x**4 - x**3 + x**2 -
         x + 1)) + (Rational(1, 5))/(x + 1)
    # 断言使用 apart 函数对 f 进行完全分解（full=True）应该得到特定的表达式
    assert apart(f, full=True).dummy_eq(
        -RootSum(x**4 - x**3 + x**2 - x + 1,
        Lambda(a, a/(x - a)), auto=False)/5 + (Rational(1, 5))/(x + 1))


# 定义一个测试函数，用于测试 apart 函数在处理浮点数时的表现
def test_apart_full_floats():
    # 定义一个复杂的符号表达式 f
    f = (
        6.43369157032015e-9*x**3 + 1.35203404799555e-5*x**2
        + 0.00357538393743079*x + 0.085
        )/(
        4.74334912634438e-11*x**4 + 4.09576274286244e-6*x**3
        + 0.00334241812250921*x**2 + 0.15406018058983*x + 1.0
    )

    # 定义预期的分解结果 expected
    expected = (
        133.599202650992/(x + 85524.0054884464)
        + 1.07757928431867/(x + 774.88576677949)
        + 0.395006955518971/(x + 40.7977016133126)
        + 0.564264854137341/(x + 7.79746609204661)
    )

    # 使用 apart 函数对 f 进行完全分解并计算浮点数结果
    f_apart = apart(f, full=True).evalf()

    # 使用 all_close 函数断言 f_apart 和 expected 在一定误差范围内相等
    assert all_close(f_apart, expected, rtol=1e-3, atol=1e-5)


# 定义一个测试函数，用于测试 apart_undetermined_coeffs 函数
def test_apart_undetermined_coeffs():
    # 定义两个多项式 p, q
    p = Poly(2*x - 3)
    q = Poly(x**9 - x**8 - x**6 + x**5 - 2*x**2 + 3*x - 1)
    # 定义预期的分解结果 r
    r = (-x**7 - x**6 - x**5 + 4)/(x**8 - x**5 - 2*x + 1) + 1/(x - 1)

    # 断言使用 apart_undetermined_coeffs 函数对 p, q 进行分解应该得到 r
    assert apart_undetermined_coeffs(p, q) == r

    # 定义两个多项式 p, q，使用 ZZ[a,b] 域
    p = Poly(1, x, domain='ZZ[a,b]')
    q = Poly((x + a)*(x + b), x, domain='ZZ[a,b]')
    # 定义预期的分解结果 r
    r = 1/((a - b)*(b + x)) - 1/((a - b)*(a + x))

    # 断言使用 apart_undetermined_coeffs 函数对 p, q 进行分解应该得到 r
    assert apart_undetermined_coeffs(p, q) == r


# 定义一个测试函数，用于测试 apart_list 函数
def test_apart_list():
    # 导入 numbered_symbols 函数
    from sympy.utilities.iterables import numbered_symbols
    # 定义函数 dummy_eq
    def dummy_eq(i, j):
        if type(i) in (list, tuple):
            return all(dummy_eq(i, j) for i, j in zip(i, j))
        return i == j or i.dummy_eq(j)

    # 定义符号表达式 f
    w0, w1, w2 = Symbol("w0"), Symbol("w1"), Symbol("w2")
    _a = Dummy("a")

    # 定义一个符号表达式 f
    f = (-2*x - 2*x**2) / (3*x**2 - 6*x)
    # 调用 apart_list 函数对 f 进行分解，使用 numbered_symbols("w") 作为 dummies 参数
    got = apart_list(f, x, dummies=numbered_symbols("w"))
    # 初始化变量 `ans`，包含一个元组，其中包括一个整数 `-1`，一个多项式对象 `Poly(Rational(2, 3), x, domain='QQ')`，以及一个包含单个元组的列表。
    ans = (-1, Poly(Rational(2, 3), x, domain='QQ'),
        [(Poly(w0 - 2, w0, domain='ZZ'), Lambda(_a, 2), Lambda(_a, -_a + x), 1)])
    # 使用断言检查 `got` 是否等于 `ans`
    assert dummy_eq(got, ans)
    
    # 调用 `apart_list` 函数，对表达式 `2/(x**2-2)` 进行部分分式分解，使用 `numbered_symbols("w")` 生成虚拟变量。
    got = apart_list(2/(x**2-2), x, dummies=numbered_symbols("w"))
    # 初始化变量 `ans`，包含一个元组，其中包括一个整数 `1`，一个多项式对象 `Poly(0, x, domain='ZZ')`，以及一个包含单个元组的列表。
    ans = (1, Poly(0, x, domain='ZZ'), [(Poly(w0**2 - 2, w0, domain='ZZ'),
        Lambda(_a, _a/2),
        Lambda(_a, -_a + x), 1)])
    # 使用断言检查 `got` 是否等于 `ans`
    assert dummy_eq(got, ans)
    
    # 初始化变量 `f`，包含一个多项式表达式 `36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)`
    f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)
    # 调用 `apart_list` 函数，对表达式 `f` 进行部分分式分解，使用 `numbered_symbols("w")` 生成虚拟变量。
    got = apart_list(f, x, dummies=numbered_symbols("w"))
    # 初始化变量 `ans`，包含一个元组，其中包括一个整数 `1`，一个多项式对象 `Poly(0, x, domain='ZZ')`，以及一个包含多个元组的列表。
    ans = (1, Poly(0, x, domain='ZZ'),
        [(Poly(w0 - 2, w0, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1),
        (Poly(w1**2 - 1, w1, domain='ZZ'), Lambda(_a, -3*_a - 6), Lambda(_a, -_a + x), 2),
        (Poly(w2 + 1, w2, domain='ZZ'), Lambda(_a, -4), Lambda(_a, -_a + x), 1)])
    # 使用断言检查 `got` 是否等于 `ans`
    assert dummy_eq(got, ans)
# 定义测试函数，用于测试 assemble_partfrac_list 函数的功能
def test_assemble_partfrac_list():
    # 定义一个分式表达式 f
    f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)
    # 对 f 进行部分分式分解，得到部分分式列表 pfd
    pfd = apart_list(f)
    # 断言调用 assemble_partfrac_list 函数后的返回值，应与预期结果相等
    assert assemble_partfrac_list(pfd) == -4/(x + 1) - 3/(x + 1)**2 - 9/(x - 1)**2 + 4/(x - 2)

    # 定义一个 Dummy 变量 a
    a = Dummy("a")
    # 定义一个包含元组、多项式对象和列表的复合结构 pfd
    pfd = (1, Poly(0, x, domain='ZZ'), [([sqrt(2),-sqrt(2)], Lambda(a, a/2), Lambda(a, -a + x), 1)])
    # 断言调用 assemble_partfrac_list 函数后的返回值，应与预期结果相等
    assert assemble_partfrac_list(pfd) == -1/(sqrt(2)*(x + sqrt(2))) + 1/(sqrt(2)*(x - sqrt(2)))


@XFAIL
# 定义一个测试函数，用于测试非交换伪多变量表达式的处理
def test_noncommutative_pseudomultivariate():
    # 创建一个非交换类 foo 继承自 Expr
    class foo(Expr):
        is_commutative=False
    # 定义一个表达式 e
    e = x/(x + x*y)
    # 定义一个表达式 c
    c = 1/(1 + y)
    # 断言 apart 函数不会穿透非交换表达式，验证表达式相等性
    assert apart(e + foo(e)) == c + foo(c)
    assert apart(e*foo(e)) == c*foo(c)

# 定义一个测试函数，用于测试非交换表达式的处理
def test_noncommutative():
    # 创建一个非交换类 foo 继承自 Expr
    class foo(Expr):
        is_commutative=False
    # 定义一个表达式 e
    e = x/(x + x*y)
    # 定义一个表达式 c
    c = 1/(1 + y)
    # 断言调用 assemble_partfrac_list 函数后的返回值，应与预期结果相等
    assert assemble_partfrac_list(e + foo()) == c + foo()

# 定义一个测试函数，用于测试问题编号 5798 的处理
def test_issue_5798():
    # 断言 apart 函数对复合表达式的处理结果，应与预期结果相等
    assert apart(
        2*x/(x**2 + 1) - (x - 1)/(2*(x**2 + 1)) + 1/(2*(x + 1)) - 2/x) == \
        (3*x + 1)/(x**2 + 1)/2 + 1/(x + 1)/2 - 2/x
```