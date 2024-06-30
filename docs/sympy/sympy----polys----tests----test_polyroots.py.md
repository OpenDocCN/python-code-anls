# `D:\src\scipysrc\sympy\sympy\polys\tests\test_polyroots.py`

```
"""Tests for algorithms for computing symbolic roots of polynomials. """

# 导入需要的符号、函数和类
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import (conjugate, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.polys.domains.integerring import ZZ
from sympy.sets.sets import Interval
from sympy.simplify.powsimp import powsimp

# 导入需要的类和函数
from sympy.polys import Poly, cyclotomic_poly, intervals, nroots, rootof

# 导入多项式根的计算相关函数
from sympy.polys.polyroots import (root_factors, roots_linear,
    roots_quadratic, roots_cubic, roots_quartic, roots_quintic,
    roots_cyclotomic, roots_binomial, preprocess_roots, roots)

# 导入正交多项式相关函数
from sympy.polys.orthopolys import legendre_poly

# 导入多项式相关异常
from sympy.polys.polyerrors import PolynomialError, \
    UnsolvableFactorError

# 导入多项式工具函数
from sympy.polys.polyutils import _nsort

# 导入测试相关的函数和类
from sympy.testing.pytest import raises, slow

# 导入随机数验证模块
from sympy.core.random import verify_numerically

# 导入数学计算模块
import mpmath

# 导入迭代器模块
from itertools import product

# 定义符号变量
a, b, c, d, e, q, t, x, y, z = symbols('a,b,c,d,e,q,t,x,y,z')


def _check(roots):
    # this is the desired invariant for roots returned
    # by all_roots. It is trivially true for linear
    # polynomials.
    # 检查返回的根是否满足预期不变量，对于线性多项式来说这是平凡的
    nreal = sum(1 if i.is_real else 0 for i in roots)
    assert sorted(roots[:nreal]) == list(roots[:nreal])
    for ix in range(nreal, len(roots), 2):
        if not (
                roots[ix + 1] == roots[ix] or
                roots[ix + 1] == conjugate(roots[ix])):
            return False
    return True


def test_roots_linear():
    # 测试线性多项式的根计算
    assert roots_linear(Poly(2*x + 1, x)) == [Rational(-1, 2)]


def test_roots_quadratic():
    # 测试二次多项式的根计算
    assert roots_quadratic(Poly(2*x**2, x)) == [0, 0]
    assert roots_quadratic(Poly(2*x**2 + 3*x, x)) == [Rational(-3, 2), 0]
    assert roots_quadratic(Poly(2*x**2 + 3, x)) == [-I*sqrt(6)/2, I*sqrt(6)/2]
    assert roots_quadratic(Poly(2*x**2 + 4*x + 3, x)) == [-1 - I*sqrt(2)/2, -1 + I*sqrt(2)/2]
    _check(Poly(2*x**2 + 4*x + 3, x).all_roots())

    # 复杂二次多项式的根计算
    f = x**2 + (2*a*e + 2*c*e)/(a - c)*x + (d - b + a*e**2 - c*e**2)/(a - c)
    assert roots_quadratic(Poly(f, x)) == \
        [-e*(a + c)/(a - c) - sqrt(a*b + c*d - a*d - b*c + 4*a*c*e**2)/(a - c),
         -e*(a + c)/(a - c) + sqrt(a*b + c*d - a*d - b*c + 4*a*c*e**2)/(a - c)]

    # 检查简化
    f = Poly(y*x**2 - 2*x - 2*y, x)
    assert roots_quadratic(f) == \
        [-sqrt(2*y**2 + 1)/y + 1/y, sqrt(2*y**2 + 1)/y + 1/y]
    f = Poly(x**2 + (-y**2 - 2)*x + y**2 + 1, x)
    assert roots_quadratic(f) == \
        [1,y**2 + 1]

    # 复杂二次多项式的根计算
    f = Poly(sqrt(2)*x**2 - 1, x)
    r = roots_quadratic(f)
    assert r == _nsort(r)

    # 检查问题 #8255
    f = Poly(-24*x**2 - 180*x + 264)
    # 对于给定的多项式对象 f，计算其所有根的平方，并且保证根的排序
    assert [w.n(2) for w in f.all_roots(radicals=True)] == \
           [w.n(2) for w in f.all_roots(radicals=False)]
    
    # 使用 product 函数生成 (-2, 2), (-2, 2), (0, -1) 的所有排列组合作为系数 _a, _b, _c
    for _a, _b, _c in product((-2, 2), (-2, 2), (0, -1)):
        # 构造二次多项式 _a*x**2 + _b*x + _c
        f = Poly(_a*x**2 + _b*x + _c)
        # 计算多项式的根
        roots = roots_quadratic(f)
        # 断言多项式的根与排序后的根相同
        assert roots == _nsort(roots)
def test_issue_7724():
    # 创建一个四次多项式对象，eq = x^4*i + x^2 + i
    eq = Poly(x**4*I + x**2 + I, x)
    # 断言多项式的根与预期的字典相等
    assert roots(eq) == {
        sqrt(I/2 + sqrt(5)*I/2): 1,
        sqrt(-sqrt(5)*I/2 + I/2): 1,
        -sqrt(I/2 + sqrt(5)*I/2): 1,
        -sqrt(-sqrt(5)*I/2 + I/2): 1}


def test_issue_8438():
    # 创建一个多项式对象 p = x^3 + y*x^2 - 2*x - 3，并将其转换为表达式形式
    p = Poly([1, y, -2, -3], x).as_expr()
    # 计算 p 关于 x 的立方根
    roots = roots_cubic(Poly(p, x), x)
    # 定义复数 z = -3/2 - 7i/2，这个在给定代码中会失败
    z = Rational(-3, 2) - I*7/2  # this will fail in code given in commit msg
    # 将 roots 中的每个根 r 替换为 y=z 后的值，存入列表 post
    post = [r.subs(y, z) for r in roots]
    # 断言 post 的集合与使用 p.subs(y, z) 计算得到的根集合相等
    assert set(post) == \
    set(roots_cubic(Poly(p.subs(y, z), x)))
    # /!\ 如果 p 没有被转换为表达式形式，这个过程会非常慢
    assert all(p.subs({y: z, x: i}).n(2, chop=True) == 0 for i in post)


def test_issue_8285():
    # 计算一个多项式的所有根，该多项式为 (4*x^8 - 1) * (x^2 + 1)
    roots = (Poly(4*x**8 - 1, x)*Poly(x**2 + 1)).all_roots()
    # 断言这些根通过 _check 函数的检查
    assert _check(roots)
    # 创建一个四次多项式对象 f = x^4 + 5*x^2 + 6
    f = Poly(x**4 + 5*x**2 + 6, x)
    # ro 是 f 的根的列表
    ro = [rootof(f, i) for i in range(4)]
    # 断言 f 的所有根与 ro 相等
    assert roots == ro
    # 再次断言这些根通过 _check 函数的检查
    assert _check(roots)
    # 从中识别出超过 2 个复数根
    roots = Poly(2*x**8 - 1).all_roots()
    # 断言这些根通过 _check 函数的检查
    assert _check(roots)
    # 断言 Poly(2*x^10 - 1).all_roots() 的长度为 10，不会失败
    assert len(Poly(2*x**10 - 1).all_roots()) == 10  # doesn't fail


def test_issue_8289():
    # 计算一个多项式的所有根，该多项式为 (x^2 + 2) * (x^4 + 2)
    roots = (Poly(x**2 + 2)*Poly(x**4 + 2)).all_roots()
    # 断言这些根通过 _check 函数的检查
    assert _check(roots)
    # 计算一个多项式的所有根，该多项式为 x^6 + 3*x^3 + 2
    roots = Poly(x**6 + 3*x**3 + 2, x).all_roots()
    # 断言这些根通过 _check 函数的检查
    assert _check(roots)
    # 计算一个多项式的所有根，该多项式为 x^6 - x + 1
    roots = Poly(x**6 - x + 1).all_roots()
    # 断言这些根通过 _check 函数的检查
    assert _check(roots)
    # 计算一个多项式的所有根，该多项式为 x^4 + 4*x^2 + 4，所有根的虚部为 2 的倍数
    roots = Poly(x**4 + 4*x**2 + 4, x).all_roots()
    # 断言这些根通过 _check 函数的检查
    assert _check(roots)


def test_issue_14291():
    # 断言一个多项式的所有根，该多项式为 ((x - 1)^2 + 1) * ((x - 1)^2 + 2) * (x - 1)
    assert Poly(((x - 1)**2 + 1)*((x - 1)**2 + 2)*(x - 1)
        ).all_roots() == [1, 1 - I, 1 + I, 1 - sqrt(2)*I, 1 + sqrt(2)*I]
    # 创建一个四次多项式对象 p = x^4 + 10*x^2 + 1
    p = x**4 + 10*x**2 + 1
    # ans 是 p 的根的列表
    ans = [rootof(p, i) for i in range(4)]
    # 断言 p 的所有根与 ans 相等
    assert Poly(p).all_roots() == ans
    # 检查 ans 是否通过 _check 函数的检查
    _check(ans)


def test_issue_13340():
    # 创建一个多项式对象 eq = y^3 + exp(x)*y + x，y 属于 EX 域
    eq = Poly(y**3 + exp(x)*y + x, y, domain='EX')
    # 计算 eq 的根
    roots_d = roots(eq)
    # 断言根的数量为 3
    assert len(roots_d) == 3


def test_issue_14522():
    # 创建一个四次多项式对象 eq = x^4 + x^3*(16 + 32*I) + x^2*(-285 + 386*I) + x*(-2824 - 448*I) - 2058 - 6053*I
    eq = Poly(x**4 + x**3*(16 + 32*I) + x**2*(-285 + 386*I) + x*(-2824 - 448*I) - 2058 - 6053*I, x)
    # 计算 eq 的根
    roots_eq = roots(eq)
    # 断言所有根 r 是否满足 eq(r) == 0
    assert all(eq(r) == 0 for r in roots_eq)


def test_issue_15076():
    # 解一个四次多项式 t^4 - 6*t^2 + t/x - 3，并断言其第一个解是否包含变量 x
    sol = roots_quartic(Poly(t**4 - 6*t**2 + t/x - 3, t))
    assert sol[0].has(x)


def test_issue_16589():
    # 创建一个四次多项式对象 eq = x^4 - 8*sqrt(2)*x^3 + 4*x^3 - 64*sqrt(2)*x^2 + 1024*x
    eq = Poly(x**4 - 8*sqrt(2)*x**3 + 4*x**3 - 64*sqrt(2)*x**2 + 1024*x, x)
    # 计算 eq 的根
    roots_eq = roots(eq)
    # 断言 0 是否是 eq 的根
    assert 0 in roots_eq


def test_roots_cubic():
    # 断言一个三次多项式的根，该多项式为 2*x^3，应该为 [0, 0, 0]
    assert roots_cubic(Poly(2*x**3, x)) == [0, 0, 0]
    # 断言一个三次多项式的根，该多项式为 x^3 - 3*x^2 + 3*x - 1，应该为 [1, 1, 1]

    assert roots_cubic(Poly(x**3 - 3*x**2 + 3*x - 1, x)) == [1, 1, 1]

    # 当 y 是任意值时有效（问题 21263）
    r = root(y, 3)
    # 断言
    # 断言：验证多项式 2*x**3 - 3*x**2 - 3*x - 1 的一个根是否等于 S.Half + 3**Rational(1, 3)/2 + 3**Rational(2, 3)/2
    assert roots_cubic(Poly(2*x**3 - 3*x**2 - 3*x - 1, x))[0] == \
         S.Half + 3**Rational(1, 3)/2 + 3**Rational(2, 3)/2
    
    # 定义方程 eq = -x**3 + 2*x**2 + 3*x - 2
    eq = -x**3 + 2*x**2 + 3*x - 2
    
    # 断言：验证带有三角参数的方程的根是否与使用 roots_cubic 函数计算得到的结果相同
    assert roots(eq, trig=True, multiple=True) == \
           roots_cubic(Poly(eq, x), trig=True) == [
        Rational(2, 3) + 2*sqrt(13)*cos(acos(8*sqrt(13)/169)/3)/3,
        -2*sqrt(13)*sin(-acos(8*sqrt(13)/169)/3 + pi/6)/3 + Rational(2, 3),
        -2*sqrt(13)*cos(-acos(8*sqrt(13)/169)/3 + pi/3)/3 + Rational(2, 3),
        ]
# 测试四次方程的根的函数
def test_roots_quartic():
    # 断言第一个四次方程的根应为 [0, 0, 0, 0]
    assert roots_quartic(Poly(x**4, x)) == [0, 0, 0, 0]
    # 断言第二个四次方程的根在以下可能的结果中之一
    assert roots_quartic(Poly(x**4 + x**3, x)) in [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ]
    # 断言第三个四次方程的根在以下可能的结果中之一
    assert roots_quartic(Poly(x**4 - x**3, x)) in [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    # 计算左侧方程的根，并与预期的右侧结果进行排序比较
    lhs = roots_quartic(Poly(x**4 + x, x))
    rhs = [S.Half + I*sqrt(3)/2, S.Half - I*sqrt(3)/2, S.Zero, -S.One]
    assert sorted(lhs, key=hash) == sorted(rhs, key=hash)

    # 对所有四次方程根的可能情况进行测试
    for i, (a, b, c, d) in enumerate([(1, 2, 3, 0),
                                      (3, -7, -9, 9),
                                      (1, 2, 3, 4),
                                      (1, 2, 3, 4),
                                      (-7, -3, 3, -6),
                                      (-3, 5, -6, -4),
                                      (6, -5, -10, -3)]):
        if i == 2:
            c = -a*(a**2/S(8) - b/S(2))
        elif i == 3:
            d = a*(a*(a**2*Rational(3, 256) - b/S(16)) + c/S(4))
        eq = x**4 + a*x**3 + b*x**2 + c*x + d
        ans = roots_quartic(Poly(eq, x))
        # 断言所有根满足方程的数值近似为零
        assert all(eq.subs(x, ai).n(chop=True) == 0 for ai in ans)

    # 非所有符号四次方程都是不可解的
    eq = Poly(q*x + q/4 + x**4 + x**3 + 2*x**2 - Rational(1, 3), x)
    sol = roots_quartic(eq)
    # 断言所有根数值上符合方程的数值近似为零
    assert all(verify_numerically(eq.subs(x, i), 0) for i in sol)
    z = symbols('z', negative=True)
    eq = x**4 + 2*x**3 + 3*x**2 + x*(z + 11) + 5
    zans = roots_quartic(Poly(eq, x))
    # 断言所有根数值上符合方程的数值近似为零
    assert all(verify_numerically(eq.subs(((x, i), (z, -1))), 0) for i in zans)
    # 但是某些情况是可解的（参见问题编号 4989）
    # 如果解不是分段函数，也是可以的，但以下测试应该通过
    eq = Poly(y*x**4 + x**3 - x + z, x)
    ans = roots_quartic(eq)
    # 断言所有解都是分段函数
    assert all(type(i) == Piecewise for i in ans)
    reps = (
        {"y": Rational(-1, 3), "z": Rational(-1, 4)},  # 4个实根
        {"y": Rational(-1, 3), "z": Rational(-1, 2)},  # 2个实根
        {"y": Rational(-1, 3), "z": -2}  # 0个实根
    )
    for rep in reps:
        sol = roots_quartic(Poly(eq.subs(rep), x))
        # 断言所有根数值上符合方程的数值近似为零
        assert all(verify_numerically(w.subs(rep) - s, 0) for w, s in zip(ans, sol))


# 测试特定问题编号 21287
def test_issue_21287():
    # 断言四次方程的根不包含 Piecewise 类型的对象
    assert not any(isinstance(i, Piecewise) for i in roots_quartic(
        Poly(x**4 - x**2*(3 + 5*I) + 2*x*(-1 + I) - 1 + 3*I, x)))


# 测试五次方程的根
def test_roots_quintic():
    eqs = (x**5 - 2,
            (x/2 + 1)**5 - 5*(x/2 + 1) + 12,
            x**5 - 110*x**3 - 55*x**2 + 2310*x + 979)
    for eq in eqs:
        roots = roots_quintic(Poly(eq))
        # 断言每个五次方程有5个根
        assert len(roots) == 5
        # 断言所有根数值上符合方程的数值近似为零
        assert all(eq.subs(x, r.n(10)).n(chop = 1e-5) == 0 for r in roots)


# 测试周期多项式的根
def test_roots_cyclotomic():
    # 断言周期多项式的根为 [1]
    assert roots_cyclotomic(cyclotomic_poly(1, x, polys=True)) == [1]
    # 断言周期多项式的根为 [-1]
    assert roots_cyclotomic(cyclotomic_poly(2, x, polys=True)) == [-1]
    # 断言：计算3阶循环多项式的根是否等于预期值
    assert roots_cyclotomic(cyclotomic_poly(3, x, polys=True)) == [Rational(-1, 2) - I*sqrt(3)/2, Rational(-1, 2) + I*sqrt(3)/2]
    
    # 断言：计算4阶循环多项式的根是否等于预期值
    assert roots_cyclotomic(cyclotomic_poly(4, x, polys=True)) == [-I, I]
    
    # 断言：计算6阶循环多项式的根是否等于预期值
    assert roots_cyclotomic(cyclotomic_poly(6, x, polys=True)) == [S.Half - I*sqrt(3)/2, S.Half + I*sqrt(3)/2]

    # 断言：计算7阶循环多项式的根是否等于预期值
    assert roots_cyclotomic(cyclotomic_poly(7, x, polys=True)) == [
        -cos(pi/7) - I*sin(pi/7),
        -cos(pi/7) + I*sin(pi/7),
        -cos(pi*Rational(3, 7)) - I*sin(pi*Rational(3, 7)),
        -cos(pi*Rational(3, 7)) + I*sin(pi*Rational(3, 7)),
        cos(pi*Rational(2, 7)) - I*sin(pi*Rational(2, 7)),
        cos(pi*Rational(2, 7)) + I*sin(pi*Rational(2, 7)),
    ]

    # 断言：计算8阶循环多项式的根是否等于预期值
    assert roots_cyclotomic(cyclotomic_poly(8, x, polys=True)) == [
        -sqrt(2)/2 - I*sqrt(2)/2,
        -sqrt(2)/2 + I*sqrt(2)/2,
        sqrt(2)/2 - I*sqrt(2)/2,
        sqrt(2)/2 + I*sqrt(2)/2,
    ]

    # 断言：计算12阶循环多项式的根是否等于预期值
    assert roots_cyclotomic(cyclotomic_poly(12, x, polys=True)) == [
        -sqrt(3)/2 - I/2,
        -sqrt(3)/2 + I/2,
        sqrt(3)/2 - I/2,
        sqrt(3)/2 + I/2,
    ]

    # 断言：计算1阶循环多项式的根是否等于预期值，使用因式分解
    assert roots_cyclotomic(cyclotomic_poly(1, x, polys=True), factor=True) == [1]
    
    # 断言：计算2阶循环多项式的根是否等于预期值，使用因式分解
    assert roots_cyclotomic(cyclotomic_poly(2, x, polys=True), factor=True) == [-1]

    # 断言：计算3阶循环多项式的根是否等于预期值，使用因式分解
    assert roots_cyclotomic(cyclotomic_poly(3, x, polys=True), factor=True) == \
        [-root(-1, 3), -1 + root(-1, 3)]
    
    # 断言：计算4阶循环多项式的根是否等于预期值，使用因式分解
    assert roots_cyclotomic(cyclotomic_poly(4, x, polys=True), factor=True) == \
        [-I, I]

    # 断言：计算5阶循环多项式的根是否等于预期值，使用因式分解
    assert roots_cyclotomic(cyclotomic_poly(5, x, polys=True), factor=True) == \
        [-root(-1, 5), -root(-1, 5)**3, root(-1, 5)**2, -1 - root(-1, 5)**2 + root(-1, 5) + root(-1, 5)**3]

    # 断言：计算6阶循环多项式的根是否等于预期值，使用因式分解
    assert roots_cyclotomic(cyclotomic_poly(6, x, polys=True), factor=True) == \
        [1 - root(-1, 3), root(-1, 3)]
# 定义测试函数 test_roots_binomial，用于测试 roots_binomial 函数的功能
def test_roots_binomial():
    # 断言：对于 Poly(5*x, x)，roots_binomial 应返回 [0]
    assert roots_binomial(Poly(5*x, x)) == [0]
    # 断言：对于 Poly(5*x**4, x)，roots_binomial 应返回 [0, 0, 0, 0]
    assert roots_binomial(Poly(5*x**4, x)) == [0, 0, 0, 0]
    # 断言：对于 Poly(5*x + 2, x)，roots_binomial 应返回 [Rational(-2, 5)]
    assert roots_binomial(Poly(5*x + 2, x)) == [Rational(-2, 5)]

    # 计算 A 的值，A = 10**(3/4) / 10
    A = 10**Rational(3, 4)/10

    # 断言：对于 Poly(5*x**4 + 2, x)，roots_binomial 应返回四个复数根
    assert roots_binomial(Poly(5*x**4 + 2, x)) == \
        [-A - A*I, -A + A*I, A - A*I, A + A*I]

    # 调用 _check 函数检查 Poly(x**8 - 2) 的根
    _check(roots_binomial(Poly(x**8 - 2)))

    # 定义非负符号变量 a1 和 b1
    a1 = Symbol('a1', nonnegative=True)
    b1 = Symbol('b1', nonnegative=True)

    # 计算二次多项式 Poly(a1*x**2 + b1, x) 的根
    r0 = roots_quadratic(Poly(a1*x**2 + b1, x))
    r1 = roots_binomial(Poly(a1*x**2 + b1, x))

    # 断言：经过 powsimp 处理后，r0 和 r1 的第一个根相等
    assert powsimp(r0[0]) == powsimp(r1[0])
    # 断言：经过 powsimp 处理后，r0 和 r1 的第二个根相等
    assert powsimp(r0[1]) == powsimp(r1[1])

    # 使用 product 函数迭代组合 (a, b, s, n)
    for a, b, s, n in product((1, 2), (1, 2), (-1, 1), (2, 3, 4, 5)):
        # 如果 a == b 且 a != 1，则跳过当前迭代
        if a == b and a != 1:
            continue
        # 构造多项式 p = Poly(a*x**n + s*b)
        p = Poly(a*x**n + s*b)
        # 计算多项式 p 的根 ans
        ans = roots_binomial(p)
        # 断言：ans 经过 _nsort 排序后仍等于原始 ans
        assert ans == _nsort(ans)

    # issue 8813 的特定测试情况
    assert roots(Poly(2*x**3 - 16*y**3, x)) == {
        2*y*(Rational(-1, 2) - sqrt(3)*I/2): 1,
        2*y: 1,
        2*y*(Rational(-1, 2) + sqrt(3)*I/2): 1}


# 定义测试函数 test_roots_preprocessing，用于测试 preprocess_roots 函数的功能
def test_roots_preprocessing():
    # 定义多项式 f
    f = a*y*x**2 + y - b
    # 调用 preprocess_roots 处理 Poly(f, x)
    coeff, poly = preprocess_roots(Poly(f, x))
    # 断言：coeff 应为 1，poly 应为 Poly(a*y*x**2 + y - b, x)
    assert coeff == 1
    assert poly == Poly(a*y*x**2 + y - b, x)

    # 后续类似的测试用例省略，依次测试不同形式的多项式和预处理结果的正确性

    # 定义多项式 f
    f = Poly(-y**2 + x**2*exp(x), y, domain=ZZ[x, exp(x)])
    g = Poly(-y**2 + exp(x), y, domain=ZZ[exp(x)])

    # 断言：对 Poly(f, x) 的预处理结果应为 (x, g)
    assert preprocess_roots(f) == (x, g)


# 定义测试函数 test_roots0，用于测试 roots 函数的功能
def test_roots0():
    # 断言：roots(1, x) 应为空字典
    assert roots(1, x) == {}
    # 断言：roots(x, x) 应返回 {S.Zero: 1}
    assert roots(x, x) == {S.Zero: 1}
    # 断言：roots(x**9, x) 应返回 {S.Zero: 9}
    assert roots(x**9, x) == {S.Zero: 9}
    # 断言：对于展开后的 ((x - 2)*(x + 3)*(x - 4))，roots 应返回 {-3: 1, 2: 1, 4: 1}
    assert roots(((x - 2)*(x + 3)*(x - 4)).expand(), x) == {-S(3): 1, S(2): 1, S(4): 1}

    # 断言：roots(2*x + 1, x) 应返回 {Rational(-1, 2): 1}
    assert roots(2*x + 1, x) == {Rational(-1, 2): 1}
    # 断言：对给定表达式求根，期望得到 {Rational(-1, 2): 2} 的结果
    assert roots((2*x + 1)**2, x) == {Rational(-1, 2): 2}

    # 断言：对给定表达式求根，期望得到 {Rational(-1, 2): 5} 的结果
    assert roots((2*x + 1)**5, x) == {Rational(-1, 2): 5}

    # 断言：对给定表达式求根，期望得到 {Rational(-1, 2): 10} 的结果
    assert roots((2*x + 1)**10, x) == {Rational(-1, 2): 10}

    # 断言：对给定表达式求根，期望得到 {I: 1, S.One: 1, -S.One: 1, -I: 1} 的结果
    assert roots(x**4 - 1, x) == {I: 1, S.One: 1, -S.One: 1, -I: 1}

    # 断言：对给定表达式求根，期望得到 {I: 2, S.One: 2, -S.One: 2, -I: 2} 的结果
    assert roots((x**4 - 1)**2, x) == {I: 2, S.One: 2, -S.One: 2, -I: 2}

    # 断言：对给定表达式求根，期望得到 {Rational(3, 2): 2} 的结果
    assert roots(((2*x - 3)**2).expand(), x) == {Rational(3, 2): 2}

    # 断言：对给定表达式求根，期望得到 {Rational(-3, 2): 2} 的结果
    assert roots(((2*x + 3)**2).expand(), x) == {Rational(-3, 2): 2}

    # 断言：对给定表达式求根，期望得到 {Rational(3, 2): 3} 的结果
    assert roots(((2*x - 3)**3).expand(), x) == {Rational(3, 2): 3}

    # 断言：对给定表达式求根，期望得到 {Rational(-3, 2): 3} 的结果
    assert roots(((2*x + 3)**3).expand(), x) == {Rational(-3, 2): 3}

    # 断言：对给定表达式求根，期望得到 {Rational(3, 2): 5} 的结果
    assert roots(((2*x - 3)**5).expand(), x) == {Rational(3, 2): 5}

    # 断言：对给定表达式求根，期望得到 {Rational(-3, 2): 5} 的结果
    assert roots(((2*x + 3)**5).expand(), x) == {Rational(-3, 2): 5}

    # 断言：对给定表达式求根，期望得到 {b/a: 5} 的结果
    assert roots(((a*x - b)**5).expand(), x) == {b/a: 5}

    # 断言：对给定表达式求根，期望得到 {-b/a: 5} 的结果
    assert roots(((a*x + b)**5).expand(), x) == {-b/a: 5}

    # 断言：对给定表达式求根，期望得到 {a: 1, S.One: 1} 的结果
    assert roots(x**2 + (-a - 1)*x + a, x) == {a: 1, S.One: 1}

    # 断言：对给定表达式求根，期望得到 {S.One: 2, S.NegativeOne: 2} 的结果
    assert roots(x**4 - 2*x**2 + 1, x) == {S.One: 2, S.NegativeOne: 2}

    # 断言：对给定表达式求根，期望得到 {S.One: 2, -1 - sqrt(2): 1, S.Zero: 2, -1 + sqrt(2): 1} 的结果
    assert roots(x**6 - 4*x**4 + 4*x**3 - x**2, x) == \
        {S.One: 2, -1 - sqrt(2): 1, S.Zero: 2, -1 + sqrt(2): 1}

    # 断言：对给定表达式求根，期望得到指定的复数和实数解的结果字典
    assert roots(x**8 - 1, x) == {
        sqrt(2)/2 + I*sqrt(2)/2: 1,
        sqrt(2)/2 - I*sqrt(2)/2: 1,
        -sqrt(2)/2 + I*sqrt(2)/2: 1,
        -sqrt(2)/2 - I*sqrt(2)/2: 1,
        S.One: 1, -S.One: 1, I: 1, -I: 1
    }

    # 计算多项式 f 的定义
    f = -2016*x**2 - 5616*x**3 - 2056*x**4 + 3324*x**5 + 2176*x**6 - \
        224*x**7 - 384*x**8 - 64*x**9

    # 断言：对给定多项式求根，期望得到指定的解的结果字典
    assert roots(f) == {S.Zero: 2, -S(2): 2, S(2): 1, Rational(-7, 2): 1,
                Rational(-3, 2): 1, Rational(-1, 2): 1, Rational(3, 2): 1}

    # 断言：对给定表达式求根，期望得到 {(a + b + c + d)/(a + b + c): 1} 的结果
    assert roots((a + b + c)*x - (a + b + c + d), x) == {(a + b + c + d)/(a + b + c): 1}

    # 断言：对给定表达式求根，期望得到空的结果字典，因为 cubics=False
    assert roots(x**3 + x**2 - x + 1, x, cubics=False) == {}

    # 断言：对给定表达式求根，期望得到指定的实数解的结果字典，因为 cubics=False
    assert roots(((x - 2)*(x + 3)*(x - 4)).expand(), x, cubics=False) == {-S(3): 1, S(2): 1, S(4): 1}

    # 断言：对给定表达式求根，期望得到指定的实数解的结果字典，因为 cubics=False
    assert roots(((x - 2)*(x + 3)*(x - 4)*(x - 5)).expand(), x, cubics=False) == \
        {-S(3): 1, S(2): 1, S(4): 1, S(5): 1}

    # 断言：对给定表达式求根，期望得到指定的复数解的结果字典
    assert roots(x**3 + 2*x**2 + 4*x + 8, x) == {-S(2): 1, -2*I: 1, 2*I: 1}

    # 断言：对给定表达式求根，期望得到指定的复数解的结果字典，因为 cubics=True
    assert roots(x**3 + 2*x**2 + 4*x + 8, x, cubics=True) == \
        {-2*I: 1, 2*I: 1, -S(2): 1}

    # 断言：对给定表达式求根，期望得到指定的复数和实数解的结果字典
    assert roots((x**2 - x)*(x**3 + 2*x**2 + 4*x + 8), x ) == \
        {S.One: 1, S.Zero: 1, -S(2): 1, -2*I: 1, 2*I
    # 定义多项式 f = x**4 + x**3 + x**2 + x + 1
    f = x**4 + x**3 + x**2 + x + 1

    # 定义有理数 r1_4, r1_8, r5_8 分别为 1/4, 1/8 和 5/8
    r1_4, r1_8, r5_8 = [ Rational(*r) for r in ((1, 4), (1, 8), (5, 8)) ]

    # 使用 assert 检查多项式 f 在 x 的根的计算结果是否符合预期
    assert roots(f, x) == {
        # 第一个根
        -r1_4 + r1_4*5**r1_2 + I*(r5_8 + r1_8*5**r1_2)**r1_2: 1,
        # 第二个根
        -r1_4 + r1_4*5**r1_2 - I*(r5_8 + r1_8*5**r1_2)**r1_2: 1,
        # 第三个根
        -r1_4 - r1_4*5**r1_2 + I*(r5_8 - r1_8*5**r1_2)**r1_2: 1,
        # 第四个根
        -r1_4 - r1_4*5**r1_2 - I*(r5_8 - r1_8*5**r1_2)**r1_2: 1,
    }

    # 定义多项式 f = z**3 + (-2 - y)*z**2 + (1 + 2*y - 2*x**2)*z - y + 2*x**2
    f = z**3 + (-2 - y)*z**2 + (1 + 2*y - 2*x**2)*z - y + 2*x**2

    # 使用 assert 检查多项式 f 在 z 的根的计算结果是否符合预期
    assert roots(f, z) == {
        S.One: 1,
        S.Half + S.Half*y + S.Half*sqrt(1 - 2*y + y**2 + 8*x**2): 1,
        S.Half + S.Half*y - S.Half*sqrt(1 - 2*y + y**2 + 8*x**2): 1,
    }

    # 使用 assert 检查不包含立方项的多项式在 x 的根的计算结果是否为空字典
    assert roots(a*b*c*x**3 + 2*x**2 + 4*x + 8, x, cubics=False) == {}

    # 使用 assert 检查包含立方项的多项式在 x 的根的计算结果是否非空
    assert roots(a*b*c*x**3 + 2*x**2 + 4*x + 8, x, cubics=True) != {}

    # 使用 assert 检查多项式 x**4 - 1 在整数根 (filter='Z') 的计算结果是否符合预期
    assert roots(x**4 - 1, x, filter='Z') == {S.One: 1, -S.One: 1}

    # 使用 assert 检查多项式 x**4 - 1 在虚数根 (filter='I') 的计算结果是否符合预期
    assert roots(x**4 - 1, x, filter='I') == {I: 1, -I: 1}

    # 使用 assert 检查多项式 (x - 1)*(x + 1) 在 x 的根的计算结果是否符合预期
    assert roots((x - 1)*(x + 1), x) == {S.One: 1, -S.One: 1}

    # 使用 assert 检查多项式 (x - 1)*(x + 1) 在满足谓词条件的根的计算结果是否符合预期
    assert roots(
        (x - 1)*(x + 1), x, predicate=lambda r: r.is_positive) == {S.One: 1}

    # 使用 assert 检查多项式 x**4 - 1 在整数根 (filter='Z', multiple=True) 的计算结果是否符合预期
    assert roots(x**4 - 1, x, filter='Z', multiple=True) == [-S.One, S.One]

    # 使用 assert 检查多项式 x**4 - 1 在虚数根 (filter='I', multiple=True) 的计算结果是否符合预期
    assert roots(x**4 - 1, x, filter='I', multiple=True) == [I, -I]

    # 定义符号 ar, br 为实数
    ar, br = symbols('a, b', real=True)

    # 定义多项式 p = x**2*(ar-br)**2 + 2*x*(br-ar) + 1
    p = x**2*(ar-br)**2 + 2*x*(br-ar) + 1

    # 使用 assert 检查多项式 p 在实数根 (filter='R') 的计算结果是否符合预期
    assert roots(p, x, filter='R') == {1/(ar - br): 2}

    # 使用 assert 检查多项式 x**3 在 x 的多重根的计算结果是否符合预期
    assert roots(x**3, x, multiple=True) == [S.Zero, S.Zero, S.Zero]

    # 使用 assert 检查常数 1234 在 x 的多重根的计算结果是否为空列表
    assert roots(1234, x, multiple=True) == []

    # 定义多项式 f = x**6 - x**5 + x**4 - x**3 + x**2 - x + 1
    f = x**6 - x**5 + x**4 - x**3 + x**2 - x + 1

    # 使用 assert 检查多项式 f 在 x 的根的计算结果是否符合预期
    assert roots(f) == {
        -I*sin(pi/7) + cos(pi/7): 1,
        -I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 1,
        -I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 1,
        I*sin(pi/7) + cos(pi/7): 1,
        I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 1,
        I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 1,
    }

    # 定义多项式 g = ((x**2 + 1)*f**2).expand()
    g = ((x**2 + 1)*f**2).expand()

    # 使用 assert 检查多项式 g 在 x 的根的计算结果是否符合预期
    assert roots(g) == {
        -I*sin(pi/7) + cos(pi/7): 2,
        -I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 2,
        -I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 2,
        I*sin(pi/7) + cos(pi/7): 2,
        I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 2,
        I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 2,
        -I: 1, I: 1,
    }

    # 计算多项式 x**3 + 40*x + 64 的所有根
    r = roots(x**3 + 40*x + 64)

    # 从所有根中找到实数根
    real_root = [rx for rx in r if rx.is_real][0]

    # 计算实数根的表达式
    cr = 108 + 6*sqrt(1074)
    assert real_root == -2*root(cr, 3)/3 + 20/root(cr, 3)

    # 定义多项式 eq，并指定其域为 EX
    eq = Poly((7 + 5*sqrt(2))*x**3 + (-6 - 4*sqrt(2))*x**2 + (-sqrt(2) - 1)*x + 2, x, domain='EX')

    # 使用 assert 检查多项式 eq 在 x 的根的计算结果是否符合预期
    assert roots
    # 创建一个多项式对象 `eq`，表示 x 的三次方程，其系数为：
    # - x^3 - 2*x^2 + 6*sqrt(2)*x^2 - 8*sqrt(2)*x + 23*x - 14 + 14*sqrt(2)
    # 多项式对象的定义域为符号表达式 'EX'，即表示这是一个符号表达式。
    eq = Poly(x**3 - 2*x**2 + 6*sqrt(2)*x**2 - 8*sqrt(2)*x + 23*x - 14 +
            14*sqrt(2), x, domain='EX')
    
    # 使用 assert 语句验证多项式 `eq` 的根是否与预期的字典相匹配
    assert roots(eq) == {-2*sqrt(2) + 2: 1, -2*sqrt(2) + 1: 1, -2*sqrt(2) - 1: 1}
    
    # 使用 assert 语句验证另一个多项式的根是否与预期的字典相匹配
    assert roots(Poly((x + sqrt(2))**3 - 7, x, domain='EX')) == \
        {-sqrt(2) + root(7, 3)*(-S.Half - sqrt(3)*I/2): 1,
         -sqrt(2) + root(7, 3)*(-S.Half + sqrt(3)*I/2): 1,
         -sqrt(2) + root(7, 3): 1}
# 定义一个测试函数，用于测试根的计算是否会导致程序挂起
def test_roots_slow():
    """Just test that calculating these roots does not hang. """
    # 定义符号变量
    a, b, c, d, x = symbols("a,b,c,d,x")

    # 定义多项式表达式 f1 和 f2
    f1 = x**2*c + (a/b) + x*c*d - a
    f2 = x**2*(a + b*(c - d)*a) + x*a*b*c/(b*d - d) + (a*d - c/d)

    # 验证多项式 f1 和 f2 的根是否为 [1, 1]
    assert list(roots(f1, x).values()) == [1, 1]
    assert list(roots(f2, x).values()) == [1, 1]

    # 定义符号变量
    (zz, yy, xx, zy, zx, yx, k) = symbols("zz,yy,xx,zy,zx,yx,k")

    # 定义多项式表达式 e1 和 e2
    e1 = (zz - k)*(yy - k)*(xx - k) + zy*yx*zx + zx - zy - yx
    e2 = (zz - k)*yx*yx + zx*(yy - k)*zx + zy*zy*(xx - k)

    # 验证多项式 e1 - e2 的根是否为 [1, 1, 1]
    assert list(roots(e1 - e2, k).values()) == [1, 1, 1]

    # 定义多项式 f，并计算其根
    f = x**3 + 2*x**2 + 8
    R = list(roots(f).keys())

    # 验证 f 在其所有根处的值是否不接近于零
    assert not any(i for i in [f.subs(x, ri).n(chop=True) for ri in R])


# 定义一个测试函数，用于验证根的计算在数值精度上的准确性
def test_roots_inexact():
    R1 = roots(x**2 + x + 1, x, multiple=True)
    R2 = roots(x**2 + x + 1.0, x, multiple=True)

    # 验证两个多项式的根是否非常接近
    for r1, r2 in zip(R1, R2):
        assert abs(r1 - r2) < 1e-12

    # 定义复杂的多项式表达式 f
    f = x**4 + 3.0*sqrt(2.0)*x**3 - (78.0 + 24.0*sqrt(3.0))*x**2 \
        + 144.0*(2*sqrt(3.0) + 9.0)

    # 计算多项式 f 的根
    R1 = roots(f, multiple=True)
    R2 = (-12.7530479110482, -3.85012393732929,
          4.89897948556636, 7.46155167569183)

    # 验证计算出的根是否与预期根非常接近
    for r1, r2 in zip(R1, R2):
        assert abs(r1 - r2) < 1e-10


# 定义一个测试函数，用于验证复杂多项式表达式的根是否为零
def test_roots_preprocessed():
    E, F, J, L = symbols("E,F,J,L")

    # 定义复杂多项式表达式 f
    f = -21601054687500000000*E**8*J**8/L**16 + \
        508232812500000000*F*x*E**7*J**7/L**14 - \
        4269543750000000*E**6*F**2*J**6*x**2/L**12 + \
        16194716250000*E**5*F**3*J**5*x**3/L**10 - \
        27633173750*E**4*F**4*J**4*x**4/L**8 + \
        14840215*E**3*F**5*J**3*x**5/L**6 + \
        54794*E**2*F**6*J**2*x**6/(5*L**4) - \
        1153*E*J*F**7*x**7/(80*L**2) + \
        633*F**8*x**8/160000

    # 验证多项式 f 是否没有实根
    assert roots(f, x) == {}

    # 计算多项式 f 的数值近似根，并与预期根进行比较
    R1 = roots(f.evalf(), x, multiple=True)
    R2 = [-1304.88375606366, 97.1168816800648, 186.946430171876, 245.526792947065,
          503.441004174773, 791.549343830097, 1273.16678129348, 1850.10650616851]

    # 使用通配符定义 p
    w = Wild('w')
    p = w*E*J/(F*L**2)

    # 验证计算出的根是否与预期根非常接近，并且匹配通配符表达式 p
    assert len(R1) == len(R2)
    for r1, r2 in zip(R1, R2):
        match = r1.match(p)
        assert match is not None and abs(match[w] - r2) < 1e-10


# 定义一个测试函数，用于验证严格模式和非严格模式下的根计算结果
def test_roots_strict():
    # 验证非严格模式下多项式 x^2 - 2*x + 1 的根
    assert roots(x**2 - 2*x + 1, strict=False) == {1: 2}
    # 验证严格模式下多项式 x^2 - 2*x + 1 的根，预期引发 UnsolvableFactorError
    raises(UnsolvableFactorError, lambda: roots(x**2 - 2*x + 1, strict=True))

    # 验证非严格模式下复杂多项式的根
    assert roots(x**6 - 2*x**5 - x**2 + 3*x - 2, strict=False) == {2: 1}


# 定义一个测试函数，用于验证混合类型的根计算
def test_roots_mixed():
    # 定义多项式 f
    f = -1936 - 5056*x - 7592*x**2 + 2704*x**3 - 49*x**4

    # 计算 f 的实数区间和所有根
    _re, _im = intervals(f, all=True)
    _nroots = nroots(f)
    _sroots = roots(f, multiple=True)

    # 将区间转换为 Interval 对象
    _re = [ Interval(a, b) for (a, b), _ in _re ]
    _im = [ Interval(re(a), re(b))*Interval(im(a), im(b)) for (a, b),
            _ in _im ]

    # 合并所有的区间
    _intervals = _re + _im

    # 计算多项式 f 的数值近似根，并按排序键排序
    _sroots = [ r.evalf() for r in _sroots ]
    _nroots = sorted(_nroots, key=lambda x: x.sort_key())
    _sroots = sorted(_sroots, key=lambda x: x.sort_key())
    for _roots in (_nroots, _sroots):
        # 循环遍历 _nroots 和 _sroots 元组中的元素（假设它们是可迭代对象）
        for i, r in zip(_intervals, _roots):
            # 使用 zip 函数同时迭代 _intervals 和 _roots 中的元素，i 是 _intervals 中的元素，r 是 _roots 中的元素
            if r.is_real:
                # 如果 r 是实数
                assert r in i
                # 断言 r 在 i（即 _intervals 中的一个区间）中
            else:
                # 如果 r 是复数
                assert (re(r), im(r)) in i
                # 断言 (re(r), im(r)) 元组在 i 中（即复数 r 的实部和虚部分别在 _intervals 中的一个区间中）
def test_root_factors():
    # 测试单项式的因子分解，期望结果为单项式本身的列表
    assert root_factors(Poly(1, x)) == [Poly(1, x)]
    # 测试包含变量的单项式的因子分解，期望结果为单项式本身的列表
    assert root_factors(Poly(x, x)) == [Poly(x, x)]

    # 测试二次方程的因子分解，期望结果为二次方程的因子列表
    assert root_factors(x**2 - 1, x) == [x + 1, x - 1]
    # 测试含有变量的二次方程的因子分解，期望结果为含有变量表达式的因子列表
    assert root_factors(x**2 - y, x) == [x - sqrt(y), x + sqrt(y)]

    # 测试四次方程的平方的因子分解，期望结果为四次方程的根的重复列表
    assert root_factors((x**4 - 1)**2) == \
        [x + 1, x + 1, x - 1, x - 1, x - I, x - I, x + I, x + I]

    # 测试多项式对象的因子分解，期望结果为多项式对象的因子列表
    assert root_factors(Poly(x**4 - 1, x), filter='Z') == \
        [Poly(x + 1, x), Poly(x - 1, x), Poly(x**2 + 1, x)]
    # 测试多项式对象的因子分解，期望结果为多项式对象的因子列表
    assert root_factors(8*x**2 + 12*x**4 + 6*x**6 + x**8, x, filter='Q') == \
        [x, x, x**6 + 6*x**4 + 12*x**2 + 8]


@slow
def test_nroots1():
    n = 64
    # 创建 n 阶勒让德多项式
    p = legendre_poly(n, x, polys=True)

    # 测试多项式的数值根，期望抛出无收敛异常
    raises(mpmath.mp.NoConvergence, lambda: p.nroots(n=3, maxsteps=5))

    # 计算多项式的数值根
    roots = p.nroots(n=3)
    # 根据数值根的顺序进行断言，数值根从小到大排列
    assert [str(r) for r in roots] == \
            ['-0.999', '-0.996', '-0.991', '-0.983', '-0.973', '-0.961',
            '-0.946', '-0.930', '-0.911', '-0.889', '-0.866', '-0.841',
            '-0.813', '-0.784', '-0.753', '-0.720', '-0.685', '-0.649',
            '-0.611', '-0.572', '-0.531', '-0.489', '-0.446', '-0.402',
            '-0.357', '-0.311', '-0.265', '-0.217', '-0.170', '-0.121',
            '-0.0730', '-0.0243', '0.0243', '0.0730', '0.121', '0.170',
            '0.217', '0.265', '0.311', '0.357', '0.402', '0.446', '0.489',
            '0.531', '0.572', '0.611', '0.649', '0.685', '0.720', '0.753',
            '0.784', '0.813', '0.841', '0.866', '0.889', '0.911', '0.930',
            '0.946', '0.961', '0.973', '0.983', '0.991', '0.996', '0.999']

def test_nroots2():
    # 创建五次多项式对象
    p = Poly(x**5 + 3*x + 1, x)

    # 计算多项式的数值根
    roots = p.nroots(n=3)
    # 根据数值根的顺序进行断言，数值根按其实部排序，如果实部相同则按虚部排序，实部优先出现
    assert [str(r) for r in roots] == \
            ['-0.332', '-0.839 - 0.944*I', '-0.839 + 0.944*I',
                '1.01 - 0.937*I', '1.01 + 0.937*I']

    # 计算多项式的数值根
    roots = p.nroots(n=5)
    # 根据数值根的顺序进行断言，数值根从小到大排列
    assert [str(r) for r in roots] == \
            ['-0.33199', '-0.83907 - 0.94385*I', '-0.83907 + 0.94385*I',
              '1.0051 - 0.93726*I', '1.0051 + 0.93726*I']


def test_roots_composite():
    # 计算复合表达式的根，期望结果的长度为 3
    assert len(roots(Poly(y**3 + y**2*sqrt(x) + y + x, y, composite=True))) == 3


def test_issue_19113():
    # 创建余弦函数的三次幂表达式
    eq = cos(x)**3 - cos(x) + 1
    # 断言在多项式计算根时会抛出多项式错误异常
    raises(PolynomialError, lambda: roots(eq))


def test_issue_17454():
    # 断言根据给定的多项式列表进行多根计算，期望结果为多根列表
    assert roots([1, -3*(-4 - 4*I)**2/8 + 12*I, 0], multiple=True) == [0, 0]


def test_issue_20913():
    # 断言对给定的实系数多项式进行实根计算，期望结果为实根列表
    assert Poly(x + 9671406556917067856609794, x).real_roots() == [-9671406556917067856609794]
    # 断言对给定的多项式进行实根计算，期望结果为实根列表
    assert Poly(x**3 + 4, x).real_roots() == [-2**(S(2)/3)]


def test_issue_22768():
    e = Rational(1, 3)
    r = (-1/a)**e*(a + 1)**(5*e)
    # 断言对给定的多项式进行根计算，期望结果为包含根及其重数的字典
    assert roots(Poly(a*x**3 + (a + 1)**5, x)) == {
        r: 1,
        -r*(1 + sqrt(3)*I)/2: 1,
        r*(-1 + sqrt(3)*I)/2: 1}
```