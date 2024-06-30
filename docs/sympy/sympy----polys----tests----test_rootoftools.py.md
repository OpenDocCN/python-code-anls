# `D:\src\scipysrc\sympy\sympy\polys\tests\test_rootoftools.py`

```
# 导入必要的库和模块
from sympy.polys.polytools import Poly  # 导入 Poly 类，用于多项式操作
import sympy.polys.rootoftools as rootoftools  # 导入 rootoftools 模块
from sympy.polys.rootoftools import (rootof, RootOf, CRootOf, RootSum,
    _pure_key_dict as D)  # 导入 RootOf 相关的类和函数

from sympy.polys.polyerrors import (
    MultivariatePolynomialError,  # 多变量多项式错误
    GeneratorsNeeded,  # 需要生成器
    PolynomialError,  # 多项式错误
)

from sympy.core.function import (Function, Lambda)  # 导入 Function 和 Lambda 类
from sympy.core.numbers import (Float, I, Rational)  # 导入 Float, I 和 Rational 类
from sympy.core.relational import Eq  # 导入 Eq 类
from sympy.core.singleton import S  # 导入 S 单例
from sympy.functions.elementary.exponential import (exp, log)  # 导入指数和对数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import tan  # 导入正切函数
from sympy.integrals.integrals import Integral  # 导入积分类
from sympy.polys.orthopolys import legendre_poly  # 导入 Legendre 多项式
from sympy.solvers.solvers import solve  # 导入 solve 函数

from sympy.testing.pytest import raises, slow  # 导入 pytest 中的 raises 和 slow 装饰器
from sympy.core.expr import unchanged  # 导入 unchanged 函数

from sympy.abc import a, b, x, y, z, r  # 导入 sympy 中的常用符号变量

# 定义测试函数 test_CRootOf___new__
def test_CRootOf___new__():
    # 测试 rootof 函数的基本用法
    assert rootof(x, 0) == 0
    assert rootof(x, -1) == 0

    assert rootof(x, S.Zero) == 0

    # 测试 rootof 函数对于线性多项式的用法
    assert rootof(x - 1, 0) == 1
    assert rootof(x - 1, -1) == 1

    assert rootof(x + 1, 0) == -1
    assert rootof(x + 1, -1) == -1

    # 测试 rootof 函数对于二次方程的用法
    assert rootof(x**2 + 2*x + 3, 0) == -1 - I*sqrt(2)
    assert rootof(x**2 + 2*x + 3, 1) == -1 + I*sqrt(2)
    assert rootof(x**2 + 2*x + 3, -1) == -1 + I*sqrt(2)
    assert rootof(x**2 + 2*x + 3, -2) == -1 - I*sqrt(2)

    # 使用 radicals=False 参数测试 rootof 函数返回类型
    r = rootof(x**2 + 2*x + 3, 0, radicals=False)
    assert isinstance(r, RootOf) is True

    r = rootof(x**2 + 2*x + 3, 1, radicals=False)
    assert isinstance(r, RootOf) is True

    r = rootof(x**2 + 2*x + 3, -1, radicals=False)
    assert isinstance(r, RootOf) is True

    r = rootof(x**2 + 2*x + 3, -2, radicals=False)
    assert isinstance(r, RootOf) is True

    # 使用 radicals=True 参数测试 rootof 函数对于根式的处理
    assert rootof((x - 1)*(x + 1), 0, radicals=True) == -1
    assert rootof((x - 1)*(x + 1), 1, radicals=True) == 1
    assert rootof((x - 1)*(x + 1), -1, radicals=True) == 1
    assert rootof((x - 1)*(x + 1), -2, radicals=True) == -1

    # 测试 rootof 函数对于复杂多项式的用法
    assert rootof((x - 1)*(x**3 + x + 3), 0) == rootof(x**3 + x + 3, 0)
    assert rootof((x - 1)*(x**3 + x + 3), 1) == 1
    assert rootof((x - 1)*(x**3 + x + 3), 2) == rootof(x**3 + x + 3, 1)
    assert rootof((x - 1)*(x**3 + x + 3), 3) == rootof(x**3 + x + 3, 2)
    assert rootof((x - 1)*(x**3 + x + 3), -1) == rootof(x**3 + x + 3, 2)
    assert rootof((x - 1)*(x**3 + x + 3), -2) == rootof(x**3 + x + 3, 1)
    assert rootof((x - 1)*(x**3 + x + 3), -3) == 1
    assert rootof((x - 1)*(x**3 + x + 3), -4) == rootof(x**3 + x + 3, 0)

    # 测试 rootof 函数对于特定多项式的用法
    assert rootof(x**4 + 3*x**3, 0) == -3
    assert rootof(x**4 + 3*x**3, 1) == 0
    assert rootof(x**4 + 3*x**3, 2) == 0
    # 断言表达式，验证 x**4 + 3*x**3 在 x = 3 时的根为 0
    assert rootof(x**4 + 3*x**3, 3) == 0

    # 使用 lambda 函数检测 GeneratorsNeeded 异常是否被正确触发
    raises(GeneratorsNeeded, lambda: rootof(0, 0))
    raises(GeneratorsNeeded, lambda: rootof(1, 0))

    # 使用 lambda 函数检测 PolynomialError 异常是否被正确触发
    raises(PolynomialError, lambda: rootof(Poly(0, x), 0))
    raises(PolynomialError, lambda: rootof(Poly(1, x), 0))
    raises(PolynomialError, lambda: rootof(x - y, 0))
    # issue 8617
    raises(PolynomialError, lambda: rootof(exp(x), 0))

    # 使用 lambda 函数检测 NotImplementedError 异常是否被正确触发
    raises(NotImplementedError, lambda: rootof(x**3 - x + sqrt(2), 0))
    raises(NotImplementedError, lambda: rootof(x**3 - x + I, 0))

    # 使用 lambda 函数检测 IndexError 异常是否被正确触发
    raises(IndexError, lambda: rootof(x**2 - 1, -4))
    raises(IndexError, lambda: rootof(x**2 - 1, -3))
    raises(IndexError, lambda: rootof(x**2 - 1, 2))
    raises(IndexError, lambda: rootof(x**2 - 1, 3))
    # 使用 lambda 函数检测 ValueError 异常是否被正确触发
    raises(ValueError, lambda: rootof(x**2 - 1, x))

    # 断言表达式，验证 Poly(x - y, x) 的根在位置 0 处为 y
    assert rootof(Poly(x - y, x), 0) == y

    # 断言表达式，验证 Poly(x**2 - y, x) 的根在位置 0 处为 -sqrt(y)
    assert rootof(Poly(x**2 - y, x), 0) == -sqrt(y)
    # 断言表达式，验证 Poly(x**2 - y, x) 的根在位置 1 处为 sqrt(y)
    assert rootof(Poly(x**2 - y, x), 1) == sqrt(y)

    # 断言表达式，验证 Poly(x**3 - y, x) 的根在位置 0 处为 y**(1/3)
    assert rootof(Poly(x**3 - y, x), 0) == y**Rational(1, 3)

    # 断言表达式，验证 y*x**3 + y*x + 2*y 在 x = 0 时的根为 -1
    assert rootof(y*x**3 + y*x + 2*y, x, 0) == -1
    # 使用 lambda 函数检测 NotImplementedError 异常是否被正确触发
    raises(NotImplementedError, lambda: rootof(x**3 + x + 2*y, x, 0))

    # 断言表达式，验证 Poly(x**3 + x + 1, x) 的根在位置 0 处具有可交换性
    assert rootof(x**3 + x + 1, 0).is_commutative is True
# 定义一个测试函数，用于测试 CRootOf 类的属性
def test_CRootOf_attributes():
    # 计算方程 x**3 + x + 3 的根，取第 0 个根
    r = rootof(x**3 + x + 3, 0)
    # 断言 r 是否为数值
    assert r.is_number
    # 断言 r 的自由符号集合是否为空集
    assert r.free_symbols == set()
    # 如果以下断言失败，则多变量多项式显然是支持的，而 RootOf.free_symbols 函数
    # 应该被修改，以返回不是 PurePoly 虚拟符号的任何符号
    raises(NotImplementedError, lambda: rootof(Poly(x**3 + y*x + 1, x), 0))


# 定义一个测试函数，用于测试 CRootOf 类的 __eq__ 方法
def test_CRootOf___eq__():
    # 断言两个相同参数的 rootof 对象是否相等，应该返回 True
    assert (rootof(x**3 + x + 3, 0) == rootof(x**3 + x + 3, 0)) is True
    # 断言不同参数的 rootof 对象是否不相等，应该返回 False
    assert (rootof(x**3 + x + 3, 0) == rootof(x**3 + x + 3, 1)) is False
    # 断言两个相同参数的 rootof 对象是否相等，应该返回 True
    assert (rootof(x**3 + x + 3, 1) == rootof(x**3 + x + 3, 1)) is True
    # 断言不同参数的 rootof 对象是否不相等，应该返回 False
    assert (rootof(x**3 + x + 3, 1) == rootof(x**3 + x + 3, 2)) is False
    # 断言两个相同参数的 rootof 对象是否相等，应该返回 True
    assert (rootof(x**3 + x + 3, 2) == rootof(x**3 + x + 3, 2)) is True

    # 断言两个相同方程不同参数的 rootof 对象是否相等，应该返回 True
    assert (rootof(x**3 + x + 3, 0) == rootof(y**3 + y + 3, 0)) is True
    # 断言不同方程不同参数的 rootof 对象是否不相等，应该返回 False
    assert (rootof(x**3 + x + 3, 0) == rootof(y**3 + y + 3, 1)) is False
    # 断言两个相同方程不同参数的 rootof 对象是否相等，应该返回 True
    assert (rootof(x**3 + x + 3, 1) == rootof(y**3 + y + 3, 1)) is True
    # 断言不同方程不同参数的 rootof 对象是否不相等，应该返回 False
    assert (rootof(x**3 + x + 3, 1) == rootof(y**3 + y + 3, 2)) is False
    # 断言两个相同方程不同参数的 rootof 对象是否相等，应该返回 True
    assert (rootof(x**3 + x + 3, 2) == rootof(y**3 + y + 3, 2)) is True


# 定义一个测试函数，用于测试 CRootOf 类的 __eval_Eq__ 方法
def test_CRootOf___eval_Eq__():
    # 定义一个函数 f
    f = Function('f')
    # 定义方程 eq = x**3 + x + 3
    eq = x**3 + x + 3
    # 计算方程 eq 的第 2 个根
    r = rootof(eq, 2)
    # 计算方程 eq 的第 1 个根
    r1 = rootof(eq, 1)
    # 断言 r 和 r1 是否不相等，应该返回 S.false
    assert Eq(r, r1) is S.false
    # 断言 r 和 r 自身是否相等，应该返回 S.true
    assert Eq(r, r) is S.true
    # 断言未改变的 Eq 函数是否与给定参数 x 相等
    assert unchanged(Eq, r, x)
    # 断言 r 是否与 0 相等，应该返回 S.false
    assert Eq(r, 0) is S.false
    # 断言 r 是否与正无穷相等，应该返回 S.false
    assert Eq(r, S.Infinity) is S.false
    # 断言 r 是否与虚数单位 I 相等，应该返回 S.false
    assert Eq(r, I) is S.false
    # 断言未改变的 Eq 函数是否与给定参数 f(0) 相等
    assert unchanged(Eq, r, f(0))
    # 解方程 eq，对每个实数解 s 进行断言，使得 r 不等于 s，应该返回 S.false
    sol = solve(eq)
    for s in sol:
        if s.is_real:
            assert Eq(r, s) is S.false
    # 计算方程 eq 的第 0 个根
    r = rootof(eq, 0)
    # 对每个实数解 s 进行断言，使得 r 等于 s，应该返回 S.true
    for s in sol:
        if s.is_real:
            assert Eq(r, s) is S.true
    # 定义方程 eq = x**3 + x + 1
    eq = x**3 + x + 1
    # 解方程 eq，对每个根 i 和解 j 进行断言，使得 rootof(eq, i) 等于 j 的计数为 3
    assert [Eq(rootof(eq, i), j) for i in range(3) for j in sol].count(True) == 3
    # 断言 rootof(eq, 0) 是否与 1 + S.ImaginaryUnit 不相等，应返回 False


# 定义一个测试函数，用于测试 CRootOf 类的 is_real 方法
def test_CRootOf_is_real():
    # 断言方程 x**3 + x + 3 的第 0 个根是否为实数，应返回 True
    assert rootof(x**3 + x + 3, 0).is_real is True
    # 断言方程 x**3 + x + 3 的第 1 个根是否为实数，应返回 False
    assert rootof(x**3 + x + 3, 1).is_real is False
    # 断言方程 x**3 + x + 3 的第 2 个根是否为实数，应返回 False
    assert rootof(x**3 + x + 3, 2).is_real is False


# 定义一个测试函数，用于测试 CRootOf 类的 is_complex 方法
def test_CRootOf_is_complex():
    # 断言方程 x**3 + x + 3 的第 0 个根是否为复数，应返回 True
    assert rootof(x**3 + x + 3, 0).is_complex is True


# 定义一个测试函数，用于测试 CRootOf 类的 subs 方法
def test_CRootOf_subs():
    # 断言方程 x**3 + x + 1 的第 0 个根在替换 x 为 y 后，是否等于方程 y**3 + y + 1 的第 0 个根
    assert rootof(x**3 + x + 1, 0).subs(x, y) == rootof(y**3 + y + 1, 0)


# 定义一个测试函数，用于测试 CRootOf 类的 diff 方法
def test_CRootOf_diff():
    # 断言方程 x**3 + x + 1 的第 0 个根对 x 求导数是否等于 0
    assert rootof(x**3 + x + 1, 0).diff(x) == 0
    # 断言方程 x**3 + x + 1 的第 0 个根对 y 求导数是否等于 0
    assert rootof(x**3 + x + 1, 0).diff(y) == 0


# 定义一个慢速测试函数，用于测试 CRootOf 类的 evalf 方法
@slow
def test_CRootOf_evalf():
    #
    # 计算多项式的实根
    roots = [str(r.n(17)) for r in p.real_roots()]
    
    # 计算特定根的实部和虚部，并进行数值估算
    re = rootof(x**5 - 5*x + 12, 0).evalf(n=20)
    assert re.epsilon_eq(Float("-1.84208596619025438271"))

    # 第二个根的实部和虚部的数值估算
    re, im = rootof(x**5 - 5*x + 12, 1).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("-0.351854240827371999559"))
    assert im.epsilon_eq(Float("-1.709561043370328882010"))

    # 第三个根的实部和虚部的数值估算
    re, im = rootof(x**5 - 5*x + 12, 2).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("-0.351854240827371999559"))
    assert im.epsilon_eq(Float("+1.709561043370328882010"))

    # 第四个根的实部和虚部的数值估算
    re, im = rootof(x**5 - 5*x + 12, 3).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("+1.272897223922499190910"))
    assert im.epsilon_eq(Float("-0.719798681483861386681"))

    # 第五个根的实部和虚部的数值估算
    re, im = rootof(x**5 - 5*x + 12, 4).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("+1.272897223922499190910"))
    assert im.epsilon_eq(Float("+0.719798681483861386681"))

    # 验证多项式方程的特定根是否满足预期数值
    assert str(rootof(x**5 + 2*x**4 + x**3 - 68719476736, 0).n(3)) == '147.'

    # 计算多项式方程的根的实部和虚部，并进行数值估算，确保两个根的实部相等，虚部不等
    a, b = rootof(eq, 1).n(2).as_real_imag()
    c, d = rootof(eq, 2).n(2).as_real_imag()
    assert a == c
    assert b < d
    assert b == -d

    # 验证特定的 Legendre 多项式根是否在不同精度下一致
    r = rootof(legendre_poly(64, x), 7)
    assert r.n(2) == r.n(100).n(2)

    # 验证无理数根的数值近似，确保结果为预期的虚数单位乘以 -1
    r0 = rootof(x**2 + 1, 0, radicals=False)
    r1 = rootof(x**2 + 1, 1, radicals=False)
    assert r0.n(4) == Float(-1.0, 4) * I
    assert r1.n(4) == Float(1.0, 4) * I

    # 验证处理多项式根时避免 UnboundLocalError
    c = CRootOf(90720*x**6 - 4032*x**4 + 84*x**2 - 1, 0)
    assert c._eval_evalf(2)  # 确保不会失败

    # 验证处理不希望计算的虚数部分时的数值近似
    assert str(RootOf(x**16 + 32*x**14 + 508*x**12 + 5440*x**10 +
        39510*x**8 + 204320*x**6 + 755548*x**4 + 1434496*x**2 +
        877969, 10).n(2)) == '-3.4*I'
    assert abs(RootOf(x**4 + 10*x**2 + 1, 0).n(2)) < 0.4

    # 验证重置和参数的使用
    r = [RootOf(x**3 + x + 3, i) for i in range(3)]
    r[0]._reset()
    for ri in r:
        i = ri._get_interval()
        ri.n(2)
        assert i != ri._get_interval()
        ri._reset()
        assert i == ri._get_interval()
        assert i == i.func(*i.args)
def test_issue_24978():
    # 定义一个二次多项式，其导数已被提取（因为前导系数为负），作为 CRootOf.poly 存储
    f = -x**2 + 2
    # 计算多项式 f 的第一个根
    r = CRootOf(f, 0)
    # 断言通过表达式表示的多项式与预期相等
    assert r.poly.as_expr() == x**2 - 2
    # 执行一个操作，触发计算一个区间，将 r.poly 放入缓存
    r.n()
    # 断言 r.poly 存在于实根缓存中
    assert r.poly in rootoftools._reals_cache


def test_CRootOf_evalf_caching_bug():
    # 计算多项式 x^5 - 5x + 12 的第一个根
    r = rootof(x**5 - 5*x + 12, 1)
    # 计算其数值结果
    r.n()
    # 获取其区间
    a = r._get_interval()
    # 再次计算相同多项式的第一个根
    r = rootof(x**5 - 5*x + 12, 1)
    # 计算其数值结果
    r.n()
    # 获取其区间
    b = r._get_interval()
    # 断言两个区间相等
    assert a == b


def test_CRootOf_real_roots():
    # 断言多项式 x^5 + x + 1 的实根为 [rootof(x^3 - x^2 + 1, 0)]
    assert Poly(x**5 + x + 1).real_roots() == [rootof(x**3 - x**2 + 1, 0)]
    # 断言不使用根式的多项式 x^5 + x + 1 的实根为 [rootof(x^3 - x^2 + 1, 0)]
    assert Poly(x**5 + x + 1).real_roots(radicals=False) == [rootof(
        x**3 - x**2 + 1, 0)]

    # 解决 GitHub 上的问题 https://github.com/sympy/sympy/issues/20902
    # 创建多项式 -3*x^4 - 10*x^3 - 12*x^2 - 6*x - 1
    p = Poly(-3*x**4 - 10*x**3 - 12*x**2 - 6*x - 1, x, domain='ZZ')
    # 断言其实数根为 [-1, -1, -1, -1/3]
    assert CRootOf.real_roots(p) == [S(-1), S(-1), S(-1), S(-1)/3]


def test_CRootOf_all_roots():
    # 断言多项式 x^5 + x + 1 的所有根
    assert Poly(x**5 + x + 1).all_roots() == [
        rootof(x**3 - x**2 + 1, 0),
        Rational(-1, 2) - sqrt(3)*I/2,
        Rational(-1, 2) + sqrt(3)*I/2,
        rootof(x**3 - x**2 + 1, 1),
        rootof(x**3 - x**2 + 1, 2),
    ]

    # 断言不使用根式的多项式 x^5 + x + 1 的所有根
    assert Poly(x**5 + x + 1).all_roots(radicals=False) == [
        rootof(x**3 - x**2 + 1, 0),
        rootof(x**2 + x + 1, 0, radicals=False),
        rootof(x**2 + x + 1, 1, radicals=False),
        rootof(x**3 - x**2 + 1, 1),
        rootof(x**3 - x**2 + 1, 2),
    ]


def test_CRootOf_eval_rational():
    # 获取勒让德多项式 P_4(x)
    p = legendre_poly(4, x, polys=True)
    # 计算其实数根的有理数近似值
    roots = [r.eval_rational(n=18) for r in p.real_roots()]
    # 断言所有根都是有理数
    for root in roots:
        assert isinstance(root, Rational)
    # 将所有根转换为指定精度的字符串表示
    roots = [str(root.n(17)) for root in roots]
    # 断言计算结果与预期一致
    assert roots == [
            "-0.86113631159405258",
            "-0.33998104358485626",
             "0.33998104358485626",
             "0.86113631159405258",
             ]


def test_CRootOf_lazy():
    # 创建不可约多项式 x^3 + 2x + 2
    f = Poly(x**3 + 2*x + 2)

    # 实数根:
    CRootOf.clear_cache()
    r = CRootOf(f, 0)
    # 构造后尚未在缓存中:
    assert r.poly not in rootoftools._reals_cache
    assert r.poly not in rootoftools._complexes_cache
    # 计算数值结果后，加入缓存:
    r.evalf()
    assert r.poly in rootoftools._reals_cache
    assert r.poly not in rootoet in cache, after construction:
    assert r.poly not in rootoftools._reals_cache
    assert r.poly not in rootoftools._complexes_cache
    r.evalf()
    # In cache after evaluation:
    assert r.poly in rootoftools._reals_cache
    assert r.poly in rootoftools._complexes_cache

    # composite poly with both real and complex roots:
    f = Poly((x**2 - 2)*(x**2 + 1))

    # real root:
    CRootOf.clear_cache()
    r = CRootOf(f, 0)
    # In cache immediately after construction:
    ```
    # 断言 r.poly 在实数根缓存中
    assert r.poly in rootoftools._reals_cache
    # 断言 r.poly 不在复数根缓存中
    assert r.poly not in rootoftools._complexes_cache
    
    # 复数根:
    # 清除 CRootOf 类的缓存
    CRootOf.clear_cache()
    # 使用 f 和索引 2 构造一个复数根对象 r
    r = CRootOf(f, 2)
    # 构造完成后立即检查是否在缓存中：
    # 断言 r.poly 在实数根缓存中
    assert r.poly in rootoftools._reals_cache
    # 断言 r.poly 在复数根缓存中
    assert r.poly in rootoftools._complexes_cache
# 定义测试函数 test_RootSum___new__()
def test_RootSum___new__():
    # 定义多项式表达式 f(x)
    f = x**3 + x + 3

    # 定义 Lambda 函数 g(r) = log(r*x)
    g = Lambda(r, log(r*x))
    # 创建 RootSum 对象 s，计算 RootSum(f, g)
    s = RootSum(f, g)

    # 断言 s 是 RootSum 类的实例
    assert isinstance(s, RootSum) is True

    # 断言 RootSum(f**2, g) 等于 2*RootSum(f, g)
    assert RootSum(f**2, g) == 2*RootSum(f, g)
    # 断言 RootSum((x - 7)*f**3, g) 等于 log(7*x) + 3*RootSum(f, g)
    assert RootSum((x - 7)*f**3, g) == log(7*x) + 3*RootSum(f, g)

    # issue 5571
    # 断言 hash(RootSum((x - 7)*f**3, g)) 等于 hash(log(7*x) + 3*RootSum(f, g))
    assert hash(RootSum((x - 7)*f**3, g)) == hash(log(7*x) + 3*RootSum(f, g))

    # 断言调用 RootSum(x**3 + x + y) 抛出 MultivariatePolynomialError
    raises(MultivariatePolynomialError, lambda: RootSum(x**3 + x + y))
    # 断言调用 RootSum(x**2 + 3, lambda x: x) 抛出 ValueError
    raises(ValueError, lambda: RootSum(x**2 + 3, lambda x: x))

    # 断言 RootSum(f, exp) 等于 RootSum(f, Lambda(x, exp(x)))
    assert RootSum(f, exp) == RootSum(f, Lambda(x, exp(x)))
    # 断言 RootSum(f, log) 等于 RootSum(f, Lambda(x, log(x)))
    assert RootSum(f, log) == RootSum(f, Lambda(x, log(x)))

    # 断言调用 RootSum(f, auto=False) 返回的对象是 RootSum 类的实例
    assert isinstance(RootSum(f, auto=False), RootSum) is True

    # 断言 RootSum(f) 等于 0
    assert RootSum(f) == 0
    # 断言 RootSum(f, Lambda(x, x)) 等于 0
    assert RootSum(f, Lambda(x, x)) == 0
    # 断言 RootSum(f, Lambda(x, x**2)) 等于 -2
    assert RootSum(f, Lambda(x, x**2)) == -2

    # 断言 RootSum(f, Lambda(x, 1)) 等于 3
    assert RootSum(f, Lambda(x, 1)) == 3
    # 断言 RootSum(f, Lambda(x, 2)) 等于 6
    assert RootSum(f, Lambda(x, 2)) == 6

    # 断言调用 RootSum(f, auto=False).is_commutative 返回 True
    assert RootSum(f, auto=False).is_commutative is True

    # 断言 RootSum(f, Lambda(x, 1/(x + x**2))) 等于 Rational(11, 3)
    assert RootSum(f, Lambda(x, 1/(x + x**2))) == Rational(11, 3)
    # 断言 RootSum(f, Lambda(x, y/(x + x**2))) 等于 Rational(11, 3)*y
    assert RootSum(f, Lambda(x, y/(x + x**2))) == Rational(11, 3)*y

    # 断言 RootSum(x**2 - 1, Lambda(x, 3*x**2), x) 等于 6
    assert RootSum(x**2 - 1, Lambda(x, 3*x**2), x) == 6
    # 断言 RootSum(x**2 - y, Lambda(x, 3*x**2), x) 等于 6*y
    assert RootSum(x**2 - y, Lambda(x, 3*x**2), x) == 6*y

    # 断言 RootSum(x**2 - 1, Lambda(x, z*x**2), x) 等于 2*z
    assert RootSum(x**2 - 1, Lambda(x, z*x**2), x) == 2*z
    # 断言 RootSum(x**2 - y, Lambda(x, z*x**2), x) 等于 2*z*y
    assert RootSum(x**2 - y, Lambda(x, z*x**2), x) == 2*z*y

    # 断言 RootSum(x**2 - 1, Lambda(x, exp(x)), quadratic=True) 等于 exp(-1) + exp(1)
    assert RootSum(x**2 - 1, Lambda(x, exp(x)), quadratic=True) == exp(-1) + exp(1)

    # 断言 RootSum(x**3 + a*x + a**3, tan, x) 等于 RootSum(x**3 + x + 1, Lambda(x, tan(a*x)))
    assert RootSum(x**3 + a*x + a**3, tan, x) == RootSum(x**3 + x + 1, Lambda(x, tan(a*x)))
    # 断言 RootSum(a**3*x**3 + a*x + 1, tan, x) 等于 RootSum(x**3 + x + 1, Lambda(x, tan(x/a)))


# 定义测试函数 test_RootSum_free_symbols()
def test_RootSum_free_symbols():
    # 断言 RootSum(x**3 + x + 3, Lambda(r, exp(r))).free_symbols 返回空集合
    assert RootSum(x**3 + x + 3, Lambda(r, exp(r))).free_symbols == set()
    # 断言 RootSum(x**3 + x + 3, Lambda(r, exp(a*r))).free_symbols 返回 {a}
    assert RootSum(x**3 + x + 3, Lambda(r, exp(a*r))).free_symbols == {a}
    # 断言 RootSum(x**3 + x + y, Lambda(r, exp(a*r)), x).free_symbols 返回 {a, y}


# 定义测试函数 test_RootSum___eq__()
def test_RootSum___eq__():
    # 定义 Lambda 函数 f(x) = exp(x)
    f = Lambda(x, exp(x))

    # 断言 RootSum(x**3 + x + 1, f) 等于 RootSum(x**3 + x + 1, f)
    assert (RootSum(x**3 + x + 1, f) == RootSum(x**3 + x + 1, f)) is True
    # 断言 RootSum(x**3 + x + 1, f) 等于 RootSum(y**3 + y + 1, f)
    assert (RootSum(x**3 + x + 1, f) == RootSum(y**3 + y + 1, f)) is True

    # 断言 RootSum(x**3 + x + 1, f) 不等于 RootSum(x**3 + x + 2, f)
    assert (RootSum(x**3 + x + 1, f) == RootSum(x**3 + x + 2, f)) is False
    # 断言 RootSum(x**3 + x + 1, f) 不等于 RootSum(y**3 + y + 2, f)
    assert (RootSum(x**3 + x + 1, f) == RootSum(y**3 + y + 2, f)) is False


# 定义测试函数 test_RootSum_doit()
def test_RootSum_doit():
    # 定义 RootSum 对象 rs，计算 RootSum(x**2 + 1, exp)
    rs = RootSum(x**2 + 1, exp)

    # 断言 rs 是 RootSum 类的实例
    assert isinstance(rs, RootSum) is True
    # 断言 rs.doit() 等于 exp(-I) + exp(I)
    assert rs.doit() == exp(-I) + exp(I)

    # 定义 RootSum 对象 rs，计算 RootSum(x**2 + a, exp, x)
    rs = RootSum(x**2 + a, exp, x)

    # 断言 rs 是 RootSum 类的实例
    assert isinstance(rs, RootSum) is True
    # 断言 rs.doit() 等于 exp(-sqrt(-a)) + exp(sqrt(-a))
    assert rs.doit() == exp(-sqrt(-a)) + exp(sqrt(-a))


# 定义测试函数 test_RootSum_evalf()
def test_RootSum_evalf():
    # 定义 RootSum 对象 rs，
    # 创建一个 Lambda 函数 g，其参数为 r，表达式为 exp(r*x)
    g = Lambda(r, exp(r*x))

    # 定义多项式 F，其为 y 的三次方加上 y 加上 3
    F = y**3 + y + 3
    # 创建一个 Lambda 函数 G，其参数为 r，表达式为 exp(r*y)
    G = Lambda(r, exp(r*y))

    # 断言：将 y 替换为 1 后的 RootSum(f, g) 等于未替换的 RootSum(f, g)
    assert RootSum(f, g).subs(y, 1) == RootSum(f, g)
    # 断言：将 x 替换为 y 后的 RootSum(f, g) 等于将 F 和 G 代入 RootSum(f, g) 后的结果
    assert RootSum(f, g).subs(x, y) == RootSum(F, G)
# 测试 RootSum 对象的 rational 参数函数
def test_RootSum_rational():
    # 断言：RootSum 对象应满足指定的函数 Lambda(z, z/(x - z))，与预期结果相等
    assert RootSum(z**5 - z + 1, Lambda(z, z/(x - z))) == (4*x - 5)/(x**5 - x + 1)

    # 定义多项式 f 和 Lambda 函数 g
    f = 161*z**3 + 115*z**2 + 19*z + 1
    g = Lambda(z, z*log(-3381*z**4/4 - 3381*z**3/4 - 625*z**2/2 - z*Rational(125, 2) - 5 + exp(x)))

    # 断言：RootSum 对象应满足指定的函数 g，对其求导后与预期结果相等
    assert RootSum(f, g).diff(x) == -((5*exp(2*x) - 6*exp(x) + 4)*exp(x)/(exp(3*x) - exp(2*x) + 1))/7


# 测试 RootSum 对象的独立使用
def test_RootSum_independent():
    # 定义多项式 f 和 Lambda 函数 g、h
    f = (x**3 - a)**2 * (x**4 - b)**3
    g = Lambda(x, 5*tan(x) + 7)
    h = Lambda(x, tan(x))

    # 创建两个独立的 RootSum 对象 r0 和 r1
    r0 = RootSum(x**3 - a, h, x)
    r1 = RootSum(x**4 - b, h, x)

    # 断言：RootSum 对象应按顺序包含指定的项
    assert RootSum(f, g, x).as_ordered_terms() == [10*r0, 15*r1, 126]


# 测试 issue 7876
def test_issue_7876():
    # 获取多项式 x**6 - x + 1 的所有根并作为列表 l1
    l1 = Poly(x**6 - x + 1, x).all_roots()
    # 创建包含同一多项式的根的列表 l2
    l2 = [rootof(x**6 - x + 1, i) for i in range(6)]
    # 断言：l1 和 l2 的 frozenset 应相等
    assert frozenset(l1) == frozenset(l2)


# 测试 issue 8316
def test_issue_8316():
    # 创建多项式 7*x**8 - 9
    f = Poly(7*x**8 - 9)
    # 断言：多项式 f 的所有根数应为 8
    assert len(f.all_roots()) == 8
    # 创建多项式 7*x**8 - 10
    f = Poly(7*x**8 - 10)
    # 断言：多项式 f 的所有根数应为 8
    assert len(f.all_roots()) == 8


# 测试 _imag_count 函数
def test__imag_count():
    # 导入 _imag_count_of_factor 函数并定义 imag_count 函数
    from sympy.polys.rootoftools import _imag_count_of_factor
    def imag_count(p):
        return sum(_imag_count_of_factor(f) * m for f, m in p.factor_list()[1])

    # 一系列断言，验证 imag_count 函数的返回结果
    assert imag_count(Poly(x**6 + 10*x**2 + 1)) == 2
    assert imag_count(Poly(x**2)) == 0
    assert imag_count(Poly([1]*3 + [-1], x)) == 0
    assert imag_count(Poly(x**3 + 1)) == 0
    assert imag_count(Poly(x**2 + 1)) == 2
    assert imag_count(Poly(x**2 - 1)) == 0
    assert imag_count(Poly(x**4 - 1)) == 2
    assert imag_count(Poly(x**4 + 1)) == 0
    assert imag_count(Poly([1, 2, 3], x)) == 0
    assert imag_count(Poly(x**3 + x + 1)) == 0
    assert imag_count(Poly(x**4 + x + 1)) == 0
    def q(r1, r2, p):
        return Poly(((x - r1)*(x - r2)).subs(x, x**p), x)
    assert imag_count(q(-1, -2, 2)) == 4
    assert imag_count(q(-1, 2, 2)) == 2
    assert imag_count(q(1, 2, 2)) == 0
    assert imag_count(q(1, 2, 4)) == 4
    assert imag_count(q(-1, 2, 4)) == 2
    assert imag_count(q(-1, -2, 4)) == 0


# 测试 RootOf 对象的 is_imaginary 方法
def test_RootOf_is_imaginary():
    # 创建 RootOf 对象 r 和其对应的区间 i
    r = RootOf(x**4 + 4*x**2 + 1, 1)
    i = r._get_interval()
    # 断言：RootOf 对象 r 应为虚数且其区间 i 的上下限乘积小于等于 0
    assert r.is_imaginary and i.ax*i.bx <= 0


# 测试区间对象的 is_disjoint 方法
def test_is_disjoint():
    # 创建方程 x**3 + 5*x + 1
    eq = x**3 + 5*x + 1
    # 获取方程根的区间 ir 和 ii
    ir = rootof(eq, 0)._get_interval()
    ii = rootof(eq, 1)._get_interval()
    # 断言：ir 和 ii 的 is_disjoint 方法应返回 True
    assert ir.is_disjoint(ii)
    assert ii.is_disjoint(ir)


# 测试纯键字典的使用
def test_pure_key_dict():
    # 创建空的 D 对象 p
    p = D()
    # 断言：键 x 不在 p 中
    assert (x in p) is False
    # 断言：键 1 不在 p 中
    assert (1 in p) is False
    # 向 p 中添加键值对 (x, 1)
    p[x] = 1
    # 断言：键 x 现在在 p 中
    assert x in p
    # 断言：键 y 不在 p 中
    assert y in p
    # 断言：键 y 在 p 中的值应为 1
    assert p[y] == 1
    # 断言：访问键为 1 的值会抛出 KeyError
    raises(KeyError, lambda: p[1])
    # 定义函数 dont，试图向 p 中添加键为 k 的值为 2，预期会抛出 ValueError
    def dont(k):
        p[k] = 2
    raises(ValueError, lambda: dont(1))


# 带有 @slow 标记的近似相对计算测试
@slow
def test_eval_approx_relative():
    # 清除 CRootOf 对象的缓存
    CRootOf.clear_cache()
    # 创建 CRootOf 对象列表 t
    t = [CRootOf(x**3 + 10*x + 1, i) for i in range(3)]
    # 断言：t 中每个对象的 eval_rational(1e-1) 方法返回值应与预期列表中对应项相等
    assert [i.eval_rational(1e-1) for i in t] == [
        Rational(-21, 220), Rational(15, 256) - I*805/256,
        Rational(15, 256) + I*805/256]
    # 重置 t[0] 对象
    t[0]._reset()
    # 断言：验证 t 列表中每个元素的 eval_rational 方法在给定精度下的计算结果是否符合预期
    assert [i.eval_rational(1e-1, 1e-4) for i in t] == [
        Rational(-21, 220), Rational(3275, 65536) - I*414645/131072,
        Rational(3275, 65536) + I*414645/131072]
    
    # 断言：验证 t[0] 对象的 _get_interval 方法返回的对象的 dx 属性是否小于 1e-1
    assert S(t[0]._get_interval().dx) < 1e-1
    
    # 断言：验证 t[1] 对象的 _get_interval 方法返回的对象的 dx 属性是否小于 1e-1
    assert S(t[1]._get_interval().dx) < 1e-1
    
    # 断言：验证 t[1] 对象的 _get_interval 方法返回的对象的 dy 属性是否小于 1e-4
    assert S(t[1]._get_interval().dy) < 1e-4
    
    # 断言：验证 t[2] 对象的 _get_interval 方法返回的对象的 dx 属性是否小于 1e-1
    assert S(t[2]._get_interval().dx) < 1e-1
    
    # 断言：验证 t[2] 对象的 _get_interval 方法返回的对象的 dy 属性是否小于 1e-4
    assert S(t[2]._get_interval().dy) < 1e-4
    
    # 重置 t[0] 对象的状态
    t[0]._reset()
    
    # 断言：验证 t 列表中每个元素的 eval_rational 方法在给定精度下的计算结果是否符合预期
    assert [i.eval_rational(1e-4, 1e-4) for i in t] == [
        Rational(-2001, 20020), Rational(6545, 131072) - I*414645/131072,
        Rational(6545, 131072) + I*414645/131072]
    
    # 断言：验证 t[0] 对象的 _get_interval 方法返回的对象的 dx 属性是否小于 1e-4
    assert S(t[0]._get_interval().dx) < 1e-4
    
    # 断言：验证 t[1] 对象的 _get_interval 方法返回的对象的 dx 属性是否小于 1e-4
    assert S(t[1]._get_interval().dx) < 1e-4
    
    # 断言：验证 t[1] 对象的 _get_interval 方法返回的对象的 dy 属性是否小于 1e-4
    assert S(t[1]._get_interval().dy) < 1e-4
    
    # 断言：验证 t[2] 对象的 _get_interval 方法返回的对象的 dx 属性是否小于 1e-4
    assert S(t[2]._get_interval().dx) < 1e-4
    
    # 断言：验证 t[2] 对象的 _get_interval 方法返回的对象的 dy 属性是否小于 1e-4
    assert S(t[2]._get_interval().dy) < 1e-4
    
    # 重置 t[0] 对象的状态
    t[0]._reset()
    
    # 注释：在以下断言中，实际相对精度应小于所测试的值，但不应超过
    # 断言：验证 t 列表中每个元素的 eval_rational 方法在 n=2 的情况下的计算结果是否符合预期
    assert [i.eval_rational(n=2) for i in t] == [
        Rational(-202201, 2024022), Rational(104755, 2097152) - I*6634255/2097152,
        Rational(104755, 2097152) + I*6634255/2097152]
    
    # 断言：验证 t[0] 对象的 _get_interval 方法返回的对象的 dx 属性相对于 t[0] 的绝对值是否小于 1e-2
    assert abs(S(t[0]._get_interval().dx)/t[0]) < 1e-2
    
    # 断言：验证 t[1] 对象的 _get_interval 方法返回的对象的 dx 属性相对于 t[1] 的绝对值是否小于 1e-2
    assert abs(S(t[1]._get_interval().dx)/t[1]).n() < 1e-2
    
    # 断言：验证 t[1] 对象的 _get_interval 方法返回的对象的 dy 属性相对于 t[1] 的绝对值是否小于 1e-2
    assert abs(S(t[1]._get_interval().dy)/t[1]).n() < 1e-2
    
    # 断言：验证 t[2] 对象的 _get_interval 方法返回的对象的 dx 属性相对于 t[2] 的绝对值是否小于 1e-2
    assert abs(S(t[2]._get_interval().dx)/t[2]).n() < 1e-2
    
    # 断言：验证 t[2] 对象的 _get_interval 方法返回的对象的 dy 属性相对于 t[2] 的绝对值是否小于 1e-2
    assert abs(S(t[2]._get_interval().dy)/t[2]).n() < 1e-2
    
    # 重置 t[0] 对象的状态
    t[0]._reset()
    
    # 断言：验证 t 列表中每个元素的 eval_rational 方法在 n=3 的情况下的计算结果是否符合预期
    assert [i.eval_rational(n=3) for i in t] == [
        Rational(-202201, 2024022), Rational(1676045, 33554432) - I*106148135/33554432,
        Rational(1676045, 33554432) + I*106148135/33554432]
    
    # 断言：验证 t[0] 对象的 _get_interval 方法返回的对象的 dx 属性相对于 t[0] 的绝对值是否小于 1e-3
    assert abs(S(t[0]._get_interval().dx)/t[0]) < 1e-3
    
    # 断言：验证 t[1] 对象的 _get_interval 方法返回的对象的 dx 属性相对于 t[1] 的绝对值是否小于 1e-3
    assert abs(S(t[1]._get_interval().dx)/t[1]).n() < 1e-3
    
    # 断言：验证 t[1] 对象的 _get_interval 方法返回的对象的 dy 属性相对于 t[1] 的绝对值是否小于 1e-3
    assert abs(S(t[1]._get_interval().dy)/t[1]).n() < 1e-3
    
    # 断言：验证 t[2] 对象的 _get_interval 方法返回的对象的 dx 属性相对于 t[2] 的绝对值是否小于 1e-3
    assert abs(S(t[2]._get_interval().dx)/t[2]).n() < 1e-3
    
    # 断言：验证 t[2] 对象的 _get_interval 方法返回的对象的 dy 属性相对于 t[2] 的绝对值是否小于 1e-3
    assert abs(S(t[2]._get_interval().dy)/t[2]).n() < 1e-3
    
    # 重置 t[0] 对象的状态
    t[0]._reset()
    
    # 断言：验证 t 列表中每个元素的 eval_approx 方法在 n=2 的情况下的计算结果是否符合预期
    a = [i.eval_approx(2) for i in t]
    
    # 断言：验证 a 列表中每个元素转换为字符串后的结果是否与预期一致
    assert [str(i) for i in a] == [
        '-0.10', '0.05 - 3.2*I', '0.05 + 3.2*I']
    
    # 断言：验证所有 t 列表中元素与对应 a 列表中元素之间的相对误差是否均小于 1e-2
    assert all(abs(((a[i] - t[i])/t[i]).n()) < 1e-2 for i in range(len(a)))
# 定义一个测试函数，用于测试问题 15920
def test_issue_15920():
    # 计算 x**5 - x + 1 的零点，并取其中的第一个根
    r = rootof(x**5 - x + 1, 0)
    # 创建一个积分对象，积分表达式为 x 在 1 到 y 之间
    p = Integral(x, (x, 1, y))
    # 断言 r 和 p 是相等的
    assert unchanged(Eq, r, p)

# 定义一个测试函数，用于测试问题 19113
def test_issue_19113():
    # 定义方程式 y**3 - y + 1
    eq = y**3 - y + 1
    # 断言将方程式转换为多项式后，实部的根与指定字符串相等
    assert str(Poly(eq).real_roots()) == '[CRootOf(x**3 - x + 1, 0)]'
    # 断言将方程式中的 y 替换为 tan(y) 后，多项式的实根与指定字符串相等
    assert str(Poly(eq.subs(y, tan(y))).real_roots()) == '[CRootOf(x**3 - x + 1, 0)]'
    # 断言将方程式中的 y 替换为 tan(x) 后，多项式的实根与指定字符串相等
    assert str(Poly(eq.subs(y, tan(x))).real_roots()) == '[CRootOf(x**3 - x + 1, 0)]'
```