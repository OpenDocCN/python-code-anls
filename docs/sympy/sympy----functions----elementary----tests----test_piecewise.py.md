# `D:\src\scipysrc\sympy\sympy\functions\elementary\tests\test_piecewise.py`

```
# 从 sympy.concrete.summations 模块导入 Sum 类
from sympy.concrete.summations import Sum
# 从 sympy.core.add 模块导入 Add 类
from sympy.core.add import Add
# 从 sympy.core.basic 模块导入 Basic 类
from sympy.core.basic import Basic
# 从 sympy.core.containers 模块导入 Tuple 类
from sympy.core.containers import Tuple
# 从 sympy.core.expr 模块导入 unchanged 函数
from sympy.core.expr import unchanged
# 从 sympy.core.function 模块导入 Function 类、diff 函数和 expand 函数
from sympy.core.function import Function, diff, expand
# 从 sympy.core.mul 模块导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core.mod 模块导入 Mod 类
from sympy.core.mod import Mod
# 从 sympy.core.numbers 模块导入各种数学常数和类
from sympy.core.numbers import Float, I, Rational, oo, pi, zoo
# 从 sympy.core.relational 模块导入关系运算符类 Eq、Ge、Gt 和 Ne
from sympy.core.relational import Eq, Ge, Gt, Ne
# 从 sympy.core.singleton 模块导入 S 对象
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Symbol 类和 symbols 函数
from sympy.core.symbol import Symbol, symbols
# 从 sympy.functions.combinatorial.factorials 模块导入 factorial 函数
from sympy.functions.combinatorial.factorials import factorial
# 从 sympy.functions.elementary.complexes 模块导入各种复数相关函数
from sympy.functions.elementary.complexes import Abs, adjoint, arg, conjugate, im, re, transpose
# 从 sympy.functions.elementary.exponential 模块导入 exp 和 log 函数
from sympy.functions.elementary.exponential import exp, log
# 从 sympy.functions.elementary.miscellaneous 模块导入 Max、Min 和 sqrt 函数
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
# 从 sympy.functions.elementary.piecewise 模块导入 Piecewise 类和相关函数
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold, piecewise_exclusive, Undefined, ExprCondPair
# 从 sympy.functions.elementary.trigonometric 模块导入三角函数 cos 和 sin
from sympy.functions.elementary.trigonometric import cos, sin
# 从 sympy.functions.special.delta_functions 模块导入 DiracDelta 和 Heaviside 函数
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
# 从 sympy.functions.special.tensor_functions 模块导入 KroneckerDelta 函数
from sympy.functions.special.tensor_functions import KroneckerDelta
# 从 sympy.integrals.integrals 模块导入 Integral 和 integrate 函数
from sympy.integrals.integrals import Integral, integrate
# 从 sympy.logic.boolalg 模块导入逻辑运算函数 And、ITE、Not 和 Or
from sympy.logic.boolalg import And, ITE, Not, Or
# 从 sympy.matrices.expressions.matexpr 模块导入 MatrixSymbol 类
from sympy.matrices.expressions.matexpr import MatrixSymbol
# 从 sympy.printing 模块导入 srepr 函数
from sympy.printing import srepr
# 从 sympy.sets.contains 模块导入 Contains 类
from sympy.sets.contains import Contains
# 从 sympy.sets.sets 模块导入 Interval 类
from sympy.sets.sets import Interval
# 从 sympy.solvers.solvers 模块导入 solve 函数
from sympy.solvers.solvers import solve
# 从 sympy.testing.pytest 模块导入 raises 和 slow 函数
from sympy.testing.pytest import raises, slow
# 从 sympy.utilities.lambdify 模块导入 lambdify 函数
from sympy.utilities.lambdify import lambdify

# 定义符号变量 a, b, c, d, x, y
a, b, c, d, x, y = symbols('a:d, x, y')
# 定义非零的符号变量 z
z = symbols('z', nonzero=True)

# 定义测试函数 test_piecewise1
def test_piecewise1():
    # 测试分段函数的规范化
    assert Piecewise((x, x < 1.)).has(1.0)  # 不会被修改为 x < 1
    # 测试 unchanged 函数，期望不改变 Piecewise 的结果
    assert unchanged(Piecewise, ExprCondPair(x, x < 1), ExprCondPair(0, True))
    # 测试 Piecewise 函数构造和比较
    assert Piecewise((x, x < 1), (0, True)) == Piecewise(ExprCondPair(x, x < 1),
                                                         ExprCondPair(0, True))
    assert Piecewise((x, x < 1), (0, True), (1, True)) == \
        Piecewise((x, x < 1), (0, True))
    assert Piecewise((x, x < 1), (0, False), (-1, 1 > 2)) == \
        Piecewise((x, x < 1))
    assert Piecewise((x, x < 1), (0, x < 1), (0, True)) == \
        Piecewise((x, x < 1), (0, True))
    assert Piecewise((x, x < 1), (0, x < 2), (0, True)) == \
        Piecewise((x, x < 1), (0, True))
    assert Piecewise((x, x < 1), (x, x < 2), (0, True)) == \
        Piecewise((x, Or(x < 1, x < 2)), (0, True))
    assert Piecewise((x, x < 1), (x, x < 2), (x, True)) == x
    assert Piecewise((x, True)) == x
    # 明确构造的空 Piecewise 不被接受
    raises(TypeError, lambda: Piecewise())
    # False 条件从不保留
    assert Piecewise((2*x, x < 0), (x, False)) == \
        Piecewise((2*x, x < 0), (x, False), evaluate=False) == \
        Piecewise((2*x, x < 0))
    assert Piecewise((x, False)) == Undefined
    raises(TypeError, lambda: Piecewise(x))
    # 断言，验证 Piecewise((x, 1)) 等于 x，在这里 1 和 0 被接受为 True 和 False
    assert Piecewise((x, 1)) == x  

    # 断言，验证 Piecewise((x, 2)) 抛出 TypeError 异常
    raises(TypeError, lambda: Piecewise((x, 2)))

    # 断言，验证 Piecewise((x, x**2)) 抛出 TypeError 异常
    raises(TypeError, lambda: Piecewise((x, x**2)))

    # 断言，验证 Piecewise(([1], True)) 抛出 TypeError 异常
    raises(TypeError, lambda: Piecewise(([1], True)))

    # 断言，验证 Piecewise(((1, 2), True)) 等于 Tuple(1, 2)
    assert Piecewise(((1, 2), True)) == Tuple(1, 2)

    # 创建条件 cond，Piecewise((1, x < 0), (2, True)) < y
    cond = (Piecewise((1, x < 0), (2, True)) < y)

    # 断言，验证 Piecewise((1, cond)) 等于 Piecewise((1, ITE(x < 0, y > 1, y > 2)))
    assert Piecewise((1, cond)) == Piecewise((1, ITE(x < 0, y > 1, y > 2)))

    # 断言，验证 Piecewise((1, x > 0), (2, And(x <= 0, x > -1))) 等于 Piecewise((1, x > 0), (2, x > -1))
    assert Piecewise((1, x > 0), (2, And(x <= 0, x > -1))) == Piecewise((1, x > 0), (2, x > -1))

    # 断言，验证 Piecewise((1, x <= 0), (2, (x < 0) & (x > -1))) 等于 Piecewise((1, x <= 0))
    assert Piecewise((1, x <= 0), (2, (x < 0) & (x > -1))) == Piecewise((1, x <= 0))

    # 测试 Piecewise 中的 Contains 支持
    pwise = Piecewise((1, And(x <= 6, x > 1, Contains(x, S.Integers))),
                      (0, True))
    assert pwise.subs(x, pi) == 0
    assert pwise.subs(x, 2) == 1
    assert pwise.subs(x, 7) == 0

    # 测试 subs 方法
    p = Piecewise((-1, x < -1), (x**2, x < 0), (log(x), x >= 0))
    p_x2 = Piecewise((-1, x**2 < -1), (x**4, x**2 < 0), (log(x**2), x**2 >= 0))
    assert p.subs(x, x**2) == p_x2
    assert p.subs(x, -5) == -1
    assert p.subs(x, -1) == 1
    assert p.subs(x, 1) == log(1)

    # 更多 subs 方法的测试
    p2 = Piecewise((1, x < pi), (-1, x < 2*pi), (0, x > 2*pi))
    p3 = Piecewise((1, Eq(x, 0)), (1/x, True))
    p4 = Piecewise((1, Eq(x, 0)), (2, 1/x > 2))
    assert p2.subs(x, 2) == 1
    assert p2.subs(x, 4) == -1
    assert p2.subs(x, 10) == 0
    assert p3.subs(x, 0.0) == 1
    assert p4.subs(x, 0.0) == 1

    # 创建函数符号
    f, g, h = symbols('f,g,h', cls=Function)
    
    # 创建 Piecewise 对象 pf 和 pg，并验证 pg.subs(g, f) 等于 pf
    pf = Piecewise((f(x), x < -1), (f(x) + h(x) + 2, x <= 1))
    pg = Piecewise((g(x), x < -1), (g(x) + h(x) + 2, x <= 1))
    assert pg.subs(g, f) == pf

    # 断言，验证 Piecewise((1, Eq(x, 0)), (0, True)).subs(x, 0) 等于 1
    assert Piecewise((1, Eq(x, 0)), (0, True)).subs(x, 0) == 1

    # 断言，验证 Piecewise((1, Eq(x, 0)), (0, True)).subs(x, 1) 等于 0
    assert Piecewise((1, Eq(x, 0)), (0, True)).subs(x, 1) == 0

    # 断言，验证 Piecewise((1, Eq(x, y)), (0, True)).subs(x, y) 等于 1
    assert Piecewise((1, Eq(x, y)), (0, True)).subs(x, y) == 1

    # 断言，验证 Piecewise((1, Eq(x, z)), (0, True)).subs(x, z) 等于 1
    assert Piecewise((1, Eq(x, z)), (0, True)).subs(x, z) == 1

    # 断言，验证 Piecewise((1, Eq(exp(x), cos(z))), (0, True)).subs(x, z) 等于 Piecewise((1, Eq(exp(z), cos(z))), (0, True))
    assert Piecewise((1, Eq(exp(x), cos(z))), (0, True)).subs(x, z) == Piecewise((1, Eq(exp(z), cos(z))), (0, True))

    # 创建 Piecewise 对象 p5 和 p5.subs(y, 0)，验证结果
    p5 = Piecewise((0, Eq(cos(x) + y, 0)), (1, True))
    assert p5.subs(y, 0) == Piecewise((0, Eq(cos(x), 0)), (1, True))

    # 断言，验证 Piecewise((-1, y < 1), (0, x < 0), (1, Eq(x, 0)), (2, True)).subs(x, 1) 等于 Piecewise((-1, y < 1), (2, True))
    assert Piecewise((-1, y < 1), (0, x < 0), (1, Eq(x, 0)), (2, True)).subs(x, 1) == Piecewise((-1, y < 1), (2, True))

    # 断言，验证 Piecewise((1, Eq(x**2, -1)), (2, x < 0)).subs(x, I) 等于 1
    assert Piecewise((1, Eq(x**2, -1)), (2, x < 0)).subs(x, I) == 1

    # 创建 Piecewise 对象 p6，验证 p6.subs(x, n) 等于 Undefined
    p6 = Piecewise((x, x > 0))
    n = symbols('n', negative=True)
    assert p6.subs(x, n) == Undefined

    # 测试 evalf 方法
    assert p.evalf() == Piecewise((-1.0, x < -1), (x**2, x < 0), (log(x), True))
    assert p.evalf(subs={x: -2}) == -1.0
    assert p.evalf(subs={x: -1}) == 1.0
    assert p.evalf(subs={x: 1}) == log(1)
    assert p6.evalf(subs={x: -5}) == Undefined

    # 测试 doit 方法
    f_int = Piecewise((Integral(x, (x, 0, 1)), x < 1))
    assert f_int.doit() == Piecewise((S.Half, x < 1))

    # 测试 differentiation
    f = x
    fp = x*p
    # 定义 Piecewise 函数 dp，根据不同区间返回不同表达式的值
    dp = Piecewise((0, x < -1), (2*x, x < 0), (1/x, x >= 0))
    # 计算 fp_dx，为 x*dp + p 的值
    fp_dx = x*dp + p
    # 断言：p 对 x 的导数应等于 dp
    assert diff(p, x) == dp
    # 断言：f*p 对 x 的导数应等于 fp_dx
    assert diff(f*p, x) == fp_dx

    # 测试简单的算术操作
    assert x*p == fp
    assert x*p + p == p + x*p
    assert p + f == f + p
    assert p + dp == dp + p
    assert p - dp == -(dp - p)

    # 测试幂运算
    # 定义 Piecewise 函数 dp2，根据不同区间返回不同幂次的表达式
    dp2 = Piecewise((0, x < -1), (4*x**2, x < 0), (1/x**2, x >= 0))
    assert dp**2 == dp2

    # 测试 _eval_interval 方法
    # 定义两个表达式 f1 和 f2
    f1 = x*y + 2
    f2 = x*y**2 + 3
    # 定义 Piecewise 函数 peval，根据 x 的值返回不同表达式的值
    peval = Piecewise((f1, x < 0), (f2, x > 0))
    # 计算 peval._eval_interval(x, 0, 0)，应等于 0
    assert peval._eval_interval(x, 0, 0) == 0
    # 计算 peval._eval_interval(x, -1, 1)，应等于 peval_interval
    assert peval._eval_interval(x, -1, 1) == peval_interval
    # 定义 Piecewise 函数 peval2，根据 x 的值返回不同表达式的值
    peval2 = Piecewise((f1, x < 0), (f2, True))
    # 计算 peval2._eval_interval(x, 0, 0)，应等于 0
    assert peval2._eval_interval(x, 0, 0) == 0
    # 计算 peval2._eval_interval(x, 1, -1)，应等于 -peval_interval
    assert peval2._eval_interval(x, 1, -1) == -peval_interval
    # 计算 peval2._eval_interval(x, -1, -2)，应等于 f1.subs(x, -2) - f1.subs(x, -1)
    assert peval2._eval_interval(x, -1, -2) == f1.subs(x, -2) - f1.subs(x, -1)
    # 计算 peval2._eval_interval(x, -1, 1)，应等于 peval_interval
    assert peval2._eval_interval(x, -1, 1) == peval_interval
    # 计算 peval2._eval_interval(x, None, 0)，应等于 peval2.subs(x, 0)
    assert peval2._eval_interval(x, None, 0) == peval2.subs(x, 0)
    # 计算 peval2._eval_interval(x, -1, None)，应等于 -peval2.subs(x, -1)
    assert peval2._eval_interval(x, -1, None) == -peval2.subs(x, -1)

    # 测试积分
    # 断言：p 的积分应等于 Piecewise 对象
    assert p.integrate() == Piecewise(
        (-x, x < -1),
        (x**3/3 + Rational(4, 3), x < 0),
        (x*log(x) - x + Rational(4, 3), True))
    # 重新定义 p
    p = Piecewise((x, x < 1), (x**2, -1 <= x), (x, 3 < x))
    # 计算积分 integrate(p, (x, -2, 2))，应等于 Rational(5, 6)
    assert integrate(p, (x, -2, 2)) == Rational(5, 6)
    # 计算积分 integrate(p, (x, 2, -2))，应等于 Rational(-5, 6)
    assert integrate(p, (x, 2, -2)) == Rational(-5, 6)
    # 重新定义 p
    p = Piecewise((0, x < 0), (1, x < 1), (0, x < 2), (1, x < 3), (0, True))
    # 计算积分 integrate(p, (x, -oo, oo))，应等于 2
    assert integrate(p, (x, -oo, oo)) == 2
    # 重新定义 p
    p = Piecewise((x, x < -10), (x**2, x <= -1), (x, 1 < x))
    # 计算积分 integrate(p, (x, -2, 2))，应为 Undefined
    assert integrate(p, (x, -2, 2)) == Undefined

    # 测试可交换性
    # 断言：p 应为 Piecewise 对象，并且具有交换性质
    assert isinstance(p, Piecewise) and p.is_commutative is True
def test_piecewise_free_symbols():
    # 创建一个 Piecewise 函数对象，根据条件返回不同的表达式
    f = Piecewise((x, a < 0), (y, True))
    # 断言自由符号集合是否包含 x, y, a
    assert f.free_symbols == {x, y, a}


def test_piecewise_integrate1():
    # 定义符号变量 x, y 为实数
    x, y = symbols('x y', real=True)

    # 创建 Piecewise 函数对象 f，根据条件返回不同的表达式
    f = Piecewise(((x - 2)**2, x >= 0), (1, True))
    # 断言 f 在区间 [-2, 2] 上的积分结果为 14/3
    assert integrate(f, (x, -2, 2)) == Rational(14, 3)

    # 创建 Piecewise 函数对象 g，根据条件返回不同的表达式
    g = Piecewise(((x - 5)**5, x >= 4), (f, True))
    # 断言 g 在区间 [-2, 2] 上的积分结果为 14/3
    assert integrate(g, (x, -2, 2)) == Rational(14, 3)
    # 断言 g 在区间 [-2, 5] 上的积分结果为 43/6
    assert integrate(g, (x, -2, 5)) == Rational(43, 6)

    # 断言 g 是否等于给定的 Piecewise 对象
    assert g == Piecewise(((x - 5)**5, x >= 4), (f, x < 4))

    # 修改 Piecewise 对象 g，根据条件返回不同的表达式
    g = Piecewise(((x - 5)**5, 2 <= x), (f, x < 2))
    # 断言 g 在区间 [-2, 2] 上的积分结果为 14/3
    assert integrate(g, (x, -2, 2)) == Rational(14, 3)
    # 断言 g 在区间 [-2, 5] 上的积分结果为 -701/6
    assert integrate(g, (x, -2, 5)) == Rational(-701, 6)

    # 断言 g 是否等于给定的 Piecewise 对象
    assert g == Piecewise(((x - 5)**5, 2 <= x), (f, True))

    # 修改 Piecewise 对象 g，根据条件返回不同的表达式
    g = Piecewise(((x - 5)**5, 2 <= x), (2*f, True))
    # 断言 g 在区间 [-2, 2] 上的积分结果为 28/3
    assert integrate(g, (x, -2, 2)) == Rational(28, 3)
    # 断言 g 在区间 [-2, 5] 上的积分结果为 -673/6
    assert integrate(g, (x, -2, 5)) == Rational(-673, 6)


def test_piecewise_integrate1b():
    # 创建 Piecewise 对象 g，根据条件返回不同的表达式
    g = Piecewise((1, x > 0), (0, Eq(x, 0)), (-1, x < 0))
    # 断言 g 在区间 [-1, 1] 上的积分结果为 0
    assert integrate(g, (x, -1, 1)) == 0

    # 创建 Piecewise 对象 g，根据条件返回不同的表达式
    g = Piecewise((1, x - y < 0), (0, True))
    # 断言 g 在区间 [-oo, 0] 上的积分结果为 -Min(0, x)
    assert integrate(g, (y, -oo, 0)) == -Min(0, x)
    # 断言在 x=-3 时，g 在区间 [-oo, 0] 上的积分结果为 3
    assert g.subs(x, -3).integrate((y, -oo, 0)) == 3
    # 断言 g 在区间 [0, -oo] 上的积分结果为 Min(0, x)
    assert integrate(g, (y, 0, -oo)) == Min(0, x)
    # 断言 g 在区间 [0, oo] 上的积分结果为 -Max(0, x) + oo
    assert integrate(g, (y, 0, oo)) == -Max(0, x) + oo
    # 断言 g 在区间 [-oo, 42] 上的积分结果为 -Min(42, x) + 42
    assert integrate(g, (y, -oo, 42)) == -Min(42, x) + 42
    # 断言 g 在区间 [-oo, oo] 上的积分结果为 -x + oo
    assert integrate(g, (y, -oo, oo)) == -x + oo

    # 创建 Piecewise 对象 g，根据条件返回不同的表达式
    g = Piecewise((0, x < 0), (x, x <= 1), (1, True))
    # 计算 g 对 x 积分，然后对 y 积分的结果，赋值给 gy1 和 g1y
    gy1 = g.integrate((x, y, 1))
    g1y = g.integrate((x, 1, y))
    # 遍历不同的 yy 值，断言 g 对 x 积分在区间 [yy, 1] 上的结果是否等于 gy1 在 yy 处的值
    for yy in (-1, S.Half, 2):
        assert g.integrate((x, yy, 1)) == gy1.subs(y, yy)
        # 断言 g 对 x 积分在区间 [1, yy] 上的结果是否等于 g1y 在 yy 处的值
        assert g.integrate((x, 1, yy)) == g1y.subs(y, yy)
    # 断言 gy1 是否等于给定的 Piecewise 对象
    assert gy1 == Piecewise(
        (-Min(1, Max(0, y))**2/2 + S.Half, y < 1),
        (-y + 1, True))
    # 断言 g1y 是否等于给定的 Piecewise 对象
    assert g1y == Piecewise(
        (Min(1, Max(0, y))**2/2 - S.Half, y < 1),
        (y - 1, True))


@slow
def test_piecewise_integrate1ca():
    # 定义符号变量 y 为实数
    y = symbols('y', real=True)
    # 创建 Piecewise 对象 g，根据条件返回不同的表达式
    g = Piecewise(
        (1 - x, Interval(0, 1).contains(x)),
        (1 + x, Interval(-1, 0).contains(x)),
        (0, True)
        )
    # 计算 g 对 x 积分，然后对 y 积分的结果，赋值给 gy1 和 g1y
    gy1 = g.integrate((x, y, 1))
    g1y = g.integrate((x, 1, y))

    # 断言 g 对 x 积分在区间 [-2, 1] 上的结果是否等于 gy1 在 y=-2 处的值
    assert g.integrate((x, -2, 1)) == gy1.subs(y, -2)
    # 断言 g 对 x 积分在区间 [1, -2] 上的结果是否等于 g1y 在 y=-2 处的值
    assert g.integrate((x, 1, -2)) == g1y.subs(y, -2)
    # 断言 g 对 x 积分在区间 [0, 1] 上的结果是否等于 gy1 在 y=0 处的值
    assert g.integrate((x, 0, 1)) == gy1.subs(y, 0)
    # 断言 g 对 x 积分在区间 [1, 0] 上的结果是否等于 g1y 在 y=0 处的值
    assert g.integrate((x, 1, 0)) == g1y.subs(y, 0)
    # 断言 g 对 x 积分在区间 [2, 1] 上的结果是否等于 gy1 在 y=2 处的值
    assert g.integrate((x, 2, 1)) == gy1.subs(y, 2)
    # 断
    # 断言条件 gy1 等于 Piecewise 对象，根据条件分段定义函数
    assert gy1 == Piecewise(
        # 第一个分段：当 y < 1 时
        (
            # 计算第一个分段的表达式
            -Min(1, Max(-1, y))**2/2 - Min(1, Max(-1, y)) +
            Min(1, Max(0, y))**2 + S.Half, y < 1
        ),
        # 第二个分段：当 y >= 1 时
        (
            # 结果为 0
            0, True
        )
    )
    
    # 断言条件 g1y 等于 Piecewise 对象，根据条件分段定义函数
    assert g1y == Piecewise(
        # 第一个分段：当 y < 1 时
        (
            # 计算第一个分段的表达式
            Min(1, Max(-1, y))**2/2 + Min(1, Max(-1, y)) -
            Min(1, Max(0, y))**2 - S.Half, y < 1
        ),
        # 第二个分段：当 y >= 1 时
        (
            # 结果为 0
            0, True
        )
    )
@slow
# 定义一个测试函数，用于测试 piecewise_integrate1cb 函数
def test_piecewise_integrate1cb():
    # 定义实数符号 y
    y = symbols('y', real=True)
    # 定义一个分段函数 g，根据 x 的不同取值返回不同的表达式
    g = Piecewise(
        (0, Or(x <= -1, x >= 1)),  # 如果 x <= -1 或者 x >= 1，则返回 0
        (1 - x, x > 0),  # 如果 x > 0，则返回 1 - x
        (1 + x, True)  # 对于其他情况，返回 1 + x
        )
    # 对 g 进行关于 x 的积分，积分上限是 (x, y, 1)
    gy1 = g.integrate((x, y, 1))
    # 对 g 进行关于 x 的积分，积分下限是 (x, 1, y)
    g1y = g.integrate((x, 1, y))

    # 断言语句，验证积分结果与预期相符
    assert g.integrate((x, -2, 1)) == gy1.subs(y, -2)
    assert g.integrate((x, 1, -2)) == g1y.subs(y, -2)
    assert g.integrate((x, 0, 1)) == gy1.subs(y, 0)
    assert g.integrate((x, 1, 0)) == g1y.subs(y, 0)
    assert g.integrate((x, 2, 1)) == gy1.subs(y, 2)
    assert g.integrate((x, 1, 2)) == g1y.subs(y, 2)

    # 断言语句，验证积分并简化后与预期的 Piecewise 表达式相符
    assert piecewise_fold(gy1.rewrite(Piecewise)
        ).simplify() == Piecewise(
            (1, y <= -1),
            (-y**2/2 - y + S.Half, y <= 0),
            (y**2/2 - y + S.Half, y < 1),
            (0, True))
    assert piecewise_fold(g1y.rewrite(Piecewise)
        ).simplify() == Piecewise(
            (-1, y <= -1),
            (y**2/2 + y - S.Half, y <= 0),
            (-y**2/2 + y - S.Half, y < 1),
            (0, True))

    # 断言语句，验证 gy1 和 g1y 在条件 y < 1 下的简化结果
    # 例如，应用条件 Min(1, Max(-1, y)) --> Max(-1, y)
    assert gy1 == Piecewise(
        (
            -Min(1, Max(-1, y))**2/2 - Min(1, Max(-1, y)) +
            Min(1, Max(0, y))**2 + S.Half, y < 1),
        (0, True)
        )
    assert g1y == Piecewise(
        (
            Min(1, Max(-1, y))**2/2 + Min(1, Max(-1, y)) -
            Min(1, Max(0, y))**2 - S.Half, y < 1),
        (0, True))


# 定义一个测试函数，用于测试 piecewise_integrate2 函数
def test_piecewise_integrate2():
    # 导入 permutations 函数
    from itertools import permutations
    # 定义积分的变量限制为 (x, c, d)
    lim = Tuple(x, c, d)
    # 定义一个分段函数 p，根据 x 的不同取值返回不同的表达式
    p = Piecewise((1, x < a), (2, x > b), (3, True))
    # 对 p 进行关于 x 的积分，积分变量限制为 lim
    q = p.integrate(lim)
    # 断言语句，验证积分结果与预期的 Piecewise 表达式相符
    assert q == Piecewise(
        (-c + 2*d - 2*Min(d, Max(a, c)) + Min(d, Max(a, b, c)), c < d),
        (-2*c + d + 2*Min(c, Max(a, d)) - Min(c, Max(a, b, d)), True))
    # 遍历 permutations((1, 2, 3, 4)) 中的每个排列
    for v in permutations((1, 2, 3, 4)):
        # 创建字典 r，将 (a, b, c, d) 映射到当前排列 v
        r = dict(zip((a, b, c, d), v))
        # 断言语句，验证替换变量后的表达式结果相等
        assert p.subs(r).integrate(lim.subs(r)) == q.subs(r)


# 定义一个测试函数，用于测试 meijer_bypass 函数
def test_meijer_bypass():
    # 完全绕过 meijerg 机制处理 Piecewise 在积分中的情况
    assert Piecewise((1, x < 4), (0, True)).integrate((x, oo, 1)) == -3


# 定义一个测试函数，用于测试 piecewise_integrate3_inequality_conditions 函数
def test_piecewise_integrate3_inequality_conditions():
    # 导入 cartes 函数
    from sympy.utilities.iterables import cartes
    # 定义变量限制为 (x, 0, 5)
    lim = (x, 0, 5)
    # 定义一个分段函数 p，根据 x 的不同取值返回不同的表达式
    p = Piecewise((1, x > a), (2, x > b), (0, True))
    # 对 p 进行关于 x 的积分，积分结果赋值给 ans
    ans = p.integrate(lim)
    # 遍历 cartes(N, repeat=2) 中的每个组合 (i, j)
    for i, j in cartes(N, repeat=2):
        # 创建字典 reps，将 (a, b) 映射到当前组合 (i, j)
        reps = dict(zip((a, b), (i, j)))
        # 断言语句，验证替换变量后的表达式结果相等
        assert ans.subs(reps) == p.subs(reps).integrate(lim)
    # 断言语句，验证特定变量替换后的表达式结果
    assert ans.subs(a, 4).subs(b, 1) == 0 + 2*3 + 1

    # 定义一个分段函数 p，根据 x 的不同取值返回不同的表达式
    p = Piecewise((1, x > a), (2, x < b), (0, True))
    # 对 p 进行关于 x 的积分，积分结果赋值给 ans
    ans = p.integrate(lim)
    # 遍历 cartes(N, repeat=2) 中的每个组合 (i, j)
    for i, j in cartes(N, repeat=2):
        # 创建字典 reps，将 (a, b) 映射到当前组合 (i, j)
        reps = dict(zip((a, b), (i, j)))
        # 断言语句，验证替换变量后的表达式结果相等
        assert ans.subs(reps) == p.subs(reps).integrate(lim)

    # 删除涉及 c1 和 c2 的旧测试部分，因为那些
    # 将上述内容简化，不同的表达式中使用了值0，而上述代码使用了3个不同的值
# 标记此函数作为一个慢速测试的装饰器
@slow
# 定义一个测试函数，用于测试 Piecewise 对象在符号条件下的积分行为
def test_piecewise_integrate4_symbolic_conditions():
    # 定义实数符号变量 a, b, x, y
    a = Symbol('a', real=True)
    b = Symbol('b', real=True)
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    
    # 定义多个 Piecewise 对象 p0 到 p5，每个对象具有不同的条件和取值
    p0 = Piecewise((0, Or(x < a, x > b)), (1, True))
    p1 = Piecewise((0, x < a), (0, x > b), (1, True))
    p2 = Piecewise((0, x > b), (0, x < a), (1, True))
    p3 = Piecewise((0, x < a), (1, x < b), (0, True))
    p4 = Piecewise((0, x > b), (1, x > a), (0, True))
    p5 = Piecewise((1, And(a < x, x < b)), (0, True))

    # 在 a=1, b=3 以及 a=3, b=1 时，检查 y 取 0 到 4 的值
    lim = Tuple(x, -oo, y)
    for p in (p0, p1, p2, p3, p4, p5):
        ans = p.integrate(lim)
        for i in range(5):
            reps = {a:1, b:3, y:i}
            # 断言积分的结果与直接代入值后再积分的结果相等
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
            reps = {a: 3, b:1, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
    
    # 对 x 范围从 y 到正无穷的积分进行类似的检查
    lim = Tuple(x, y, oo)
    for p in (p0, p1, p2, p3, p4, p5):
        ans = p.integrate(lim)
        for i in range(5):
            reps = {a:1, b:3, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
            reps = {a:3, b:1, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
    
    # 定义一个 Piecewise 对象 ans，用于比较 p0, p1, p2, p4 的 x 范围积分结果
    ans = Piecewise(
        (0, x <= Min(a, b)),
        (x - Min(a, b), x <= b),
        (b - Min(a, b), True))
    for i in (p0, p1, p2, p4):
        # 断言 p0, p1, p2, p4 对象在 x 范围内积分的结果与 ans 相等
        assert i.integrate(x) == ans
    
    # 断言 p3 对象在 x 范围内积分的结果
    assert p3.integrate(x) == Piecewise(
        (0, x < a),
        (-a + x, x <= Max(a, b)),
        (-a + Max(a, b), True))
    
    # 断言 p5 对象在 x 范围内积分的结果
    assert p5.integrate(x) == Piecewise(
        (0, x <= a),
        (-a + x, x <= Max(a, b)),
        (-a + Max(a, b), True))
    
    # 重新定义 p1 到 p5，更新其条件和取值
    p1 = Piecewise((0, x < a), (S.Half, x > b), (1, True))
    p2 = Piecewise((S.Half, x > b), (0, x < a), (1, True))
    p3 = Piecewise((0, x < a), (1, x < b), (S.Half, True))
    p4 = Piecewise((S.Half, x > b), (1, x > a), (0, True))
    p5 = Piecewise((1, And(a < x, x < b)), (S.Half, x > b), (0, True))

    # 再次检查 a=1, b=3 以及 a=3, b=1 时，y 取 0 到 4 的值
    lim = Tuple(x, -oo, y)
    for p in (p1, p2, p3, p4, p5):
        ans = p.integrate(lim)
        for i in range(5):
            reps = {a:1, b:3, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
            reps = {a: 3, b:1, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))


# 定义一个测试函数，用于测试 Piecewise 对象在独立条件下的积分行为
def test_piecewise_integrate5_independent_conditions():
    # 定义一个 Piecewise 对象 p，其条件分为 y=0 和通用情况 x*y
    p = Piecewise((0, Eq(y, 0)), (x*y, True))
    # 断言在 x 范围从 1 到 3 的积分结果
    assert integrate(p, (x, 1, 3)) == Piecewise((0, Eq(y, 0)), (4*y, True))


# 定义一个测试函数，用于解决特定问题编号 22917
def test_issue_22917():
    p = (Piecewise((0, ITE((x - y > 1) | (2 * x - 2 * y > 1), False,
                           ITE(x - y > 1, 2 * y - 2 < -1, 2 * x - 2 * y > 1))),
                   (Piecewise((0, ITE(x - y > 1, True, 2 * x - 2 * y > 1)),
                              (2 * Piecewise((0, x - y > 1), (y, True)), True)), True))
         + 2 * Piecewise((1, ITE((x - y > 1) | (2 * x - 2 * y > 1), False,
                                 ITE(x - y > 1, 2 * y - 2 < -1, 2 * x - 2 * y > 1))),
                         (Piecewise((1, ITE(x - y > 1, True, 2 * x - 2 * y > 1)),
                                    (2 * Piecewise((1, x - y > 1), (x, True)), True)), True)))


    # 定义复杂的 Piecewise 表达式 p，包含多个条件分支和 ITE 函数调用


    assert piecewise_fold(p) == Piecewise((2, (x - y > S.Half) | (x - y > 1)),
                                          (2*y + 4, x - y > 1),
                                          (4*x + 2*y, True))


    # 断言验证 piecewise_fold 函数对 p 的计算结果是否等于给定的 Piecewise 结果


    assert piecewise_fold(p > 1).rewrite(ITE) == ITE((x - y > S.Half) | (x - y > 1), True,
                                                     ITE(x - y > 1, 2*y + 4 > 1, 4*x + 2*y > 1))


    # 断言验证 piecewise_fold 函数对 p 大于 1 的情况下重写 ITE 函数后的计算结果是否正确
# 定义测试函数 test_piecewise_simplify
def test_piecewise_simplify():
    # 创建 Piecewise 对象 p，根据条件选择不同的表达式
    p = Piecewise(((x**2 + 1)/x**2, Eq(x*(1 + x) - x**2, 0)),
                  ((-1)**x*(-1), True))
    # 断言简化后的 p 对象与预期结果相同
    assert p.simplify() == \
        Piecewise((zoo, Eq(x, 0)), ((-1)**(x + 1), True))

    # 在条件中存在 Eq 的情况下进行简化
    assert Piecewise(
        (a, And(Eq(a, 0), Eq(a + b, 0))), (1, True)).simplify(
        ) == Piecewise(
        (0, And(Eq(a, 0), Eq(b, 0))), (1, True))

    # 断言多个条件的 Piecewise 对象简化后与预期结果相同
    assert Piecewise((2*x*factorial(a)/(factorial(y)*factorial(-y + a)),
        Eq(y, 0) & Eq(-y + a, 0)), (2*factorial(a)/(factorial(y)*factorial(-y
        + a)), Eq(y, 0) & Eq(-y + a, 1)), (0, True)).simplify(
        ) == Piecewise(
            (2*x, And(Eq(a, 0), Eq(y, 0))),
            (2, And(Eq(a, 1), Eq(y, 0))),
            (0, True))

    # 检查 Piecewise 对象在特定条件下不进行简化
    args = (2, And(Eq(x, 2), Ge(y, 0))), (x, True)
    assert Piecewise(*args).simplify() == Piecewise(*args)

    # 断言 Piecewise 对象在给定条件下不进行简化
    args = (1, Eq(x, 0)), (sin(x)/x, True)
    assert Piecewise(*args).simplify() == Piecewise(*args)

    # 断言 Piecewise 对象简化后与预期结果相同
    assert Piecewise((2 + y, And(Eq(x, 2), Eq(y, 0))), (x, True)
        ).simplify() == x

    # 检查 x 或 f(x) 被识别为 lhs（左侧符号）时的情况
    args = Tuple((1, Eq(x, 0)), (sin(x) + 1 + x, True))
    ans = x + sin(x) + 1
    f = Function('f')
    assert Piecewise(*args).simplify() == ans
    assert Piecewise(*args.subs(x, f(x))).simplify() == ans.subs(x, f(x))

    # 检查 issue 18634 的情况
    d = Symbol("d", integer=True)
    n = Symbol("n", integer=True)
    t = Symbol("t", positive=True)
    # 创建 Piecewise 表达式，包含不同条件下的表达式
    expr = Piecewise((-d + 2*n, Eq(1/t, 1)), (t**(1 - 4*n)*t**(4*n - 1)*(-d + 2*n), True))
    # 断言简化后的 expr 对象与预期结果相同
    assert expr.simplify() == -d + 2*n

    # 检查 issue 22747
    # 定义一个 Piecewise 对象 p，包含多个条件和表达式的分段函数
    p = Piecewise(
        # 第一个条件：当 t 小于 -2 时，返回 0
        (0, (t < -2) & (t < -1) & (t < 0)),
        # 第二个条件：当 t 小于 -1 时，返回 (t/2 + 1)*(t + 1)*(t + 2)
        ((t/2 + 1)*(t + 1)*(t + 2), (t < -1) & (t < 0)),
        # 第三个条件：当 t 小于 1 时，返回 (S.Half - t/2)*(1 - t)*(t + 1)
        ((S.Half - t/2)*(1 - t)*(t + 1), (t < -2) & (t < -1) & (t < 1)),
        # 第四个条件：当 t 在 -2 到 0 之间时，返回 (t + 1)*(-t*(t/2 + 1) + (S.Half - t/2)*(1 - t))
        ((t + 1)*(-t*(t/2 + 1) + (S.Half - t/2)*(1 - t)), (t < -2) & (t < -1) & (t < 0) & (t < 1)),
        # 第五个条件：当 t 在 -1 到 1 之间时，返回 (t + 1)*((S.Half - t/2)*(1 - t) + (t/2 + 1)*(t + 2))
        ((t + 1)*((S.Half - t/2)*(1 - t) + (t/2 + 1)*(t + 2)), (t < -1) & (t < 1)),
        # 第六个条件：当 t 在 -1 到 0 之间时，返回 (t + 1)*(-t*(t/2 + 1) + (S.Half - t/2)*(1 - t))
        ((t + 1)*(-t*(t/2 + 1) + (S.Half - t/2)*(1 - t)), (t < -1) & (t < 0) & (t < 1)),
        # 第七个条件：当 t 小于 -2 时，返回 0
        (0, (t < -2) & (t < -1)),
        # 第八个条件：当 t 小于 -1 时，返回 (t/2 + 1)*(t + 1)*(t + 2)
        ((t/2 + 1)*(t + 1)*(t + 2), t < -1),
        # 第九个条件：当 t 在 0 到 1 之间时，返回 (t + 1)*(-t*(t/2 + 1) + (S.Half - t/2)*(t + 1))
        ((t + 1)*(-t*(t/2 + 1) + (S.Half - t/2)*(t + 1)), (t < 0) & ((t < -2) | (t < 0))),
        # 第十个条件：当 t 在 1 到 2 之间时，返回 (S.Half - t/2)*(1 - t)*(t + 1)
        ((S.Half - t/2)*(1 - t)*(t + 1), (t < 1) & ((t < -2) | (t < 1))),
        # 最后一个条件：其他情况返回 0
        (0, True)
    )

    # 断言简化后的 p 与给定的 Piecewise 对象相等
    assert p.simplify() == Piecewise(
        # 简化后的第一个条件：当 t 小于 -2 时，返回 0
        (0, t < -2),
        # 简化后的第二个条件：当 t 小于 -1 时，返回 (t + 1)*(t + 2)**2/2
        ((t + 1)*(t + 2)**2/2, t < -1),
        # 简化后的第三个条件：当 t 小于 0 时，返回 -3*t**3/2 - 5*t**2/2 + 1
        (-3*t**3/2 - 5*t**2/2 + 1, t < 0),
        # 简化后的第四个条件：当 t 小于 1 时，返回 3*t**3/2 - 5*t**2/2 + 1
        (3*t**3/2 - 5*t**2/2 + 1, t < 1),
        # 简化后的第五个条件：当 t 小于 2 时，返回 (1 - t)*(t - 2)**2/2
        ((1 - t)*(t - 2)**2/2, t < 2),
        # 最后一个条件：其他情况返回 0
        (0, True)
    )

    # coverage
    # 测试 Piecewise 对象的 simplify 方法
    nan = Undefined
    assert Piecewise((1, x > 3), (2, x < 2), (3, x > 1)).simplify() == Piecewise((1, x > 3), (2, x < 2), (3, True))
    assert Piecewise((1, x < 2), (2, x < 1), (3, True)).simplify() == Piecewise((1, x < 2), (3, True))
    assert Piecewise((1, x > 2)).simplify() == Piecewise((1, x > 2), (nan, True))
    assert Piecewise((1, (x >= 2) & (x < oo))).simplify() == Piecewise((1, (x >= 2) & (x < oo)), (nan, True))
    assert Piecewise((1, x < 2), (2, (x > 1) & (x < 3)), (3, True)).simplify() == Piecewise((1, x < 2), (2, x < 3), (3, True))
    assert Piecewise((1, x < 2), (2, (x <= 3) & (x > 1)), (3, True)).simplify() == Piecewise((1, x < 2), (2, x <= 3), (3, True))
    assert Piecewise((1, x < 2), (2, (x > 2) & (x < 3)), (3, True)).simplify() == Piecewise((1, x < 2), (2, (x > 2) & (x < 3)), (3, True))
    assert Piecewise((1, x < 2), (2, (x >= 1) & (x <= 3)), (3, True)).simplify() == Piecewise((1, x < 2), (2, x <= 3), (3, True))
    assert Piecewise((1, x < 1), (2, (x >= 2) & (x <= 3)), (3, True)).simplify() == Piecewise((1, x < 1), (2, (x >= 2) & (x <= 3)), (3, True))
    # https://github.com/sympy/sympy/issues/25603
    # 验证条件逻辑表达式，检查 Piecewise 对象在简化后是否与其自身相等
    assert Piecewise((log(x), (x <= 5) & (x > 3)), (x, True)
        ).simplify() == Piecewise((log(x), (x <= 5) & (x > 3)), (x, True))
    
    # 验证条件逻辑表达式，检查 Piecewise 对象在简化后是否与期望的结果相等
    assert Piecewise((1, (x >= 1) & (x < 3)), (2, (x > 2) & (x < 4))
        ).simplify() == Piecewise((1, (x >= 1) & (x < 3)), (
        2, (x >= 3) & (x < 4)), (nan, True))
    
    # 验证条件逻辑表达式，检查 Piecewise 对象在简化后是否与期望的结果相等
    assert Piecewise((1, (x >= 1) & (x <= 3)), (2, (x > 2) & (x < 4))
        ).simplify() == Piecewise((1, (x >= 1) & (x <= 3)), (
        2, (x > 3) & (x < 4)), (nan, True))
    
    # 定义符号 L，要求其为非负数
    L = Symbol('L', nonnegative=True)
    # 创建 Piecewise 对象 p，包含多个条件及其对应的表达式
    p = Piecewise((nan, x <= 0), (0, (x >= 0) & (L > x) & (L - x <= 0)),
        (x - L/2, (L > x) & (L - x <= 0)),
        (L/2 - x, (x >= 0) & (L > x)),
        (0, L > x), (nan, True))
    # 验证条件逻辑表达式，检查 Piecewise 对象在简化后是否与期望的结果相等
    assert p.simplify() == Piecewise(
        (nan, x <= 0), (L/2 - x, L > x), (nan, True))
    # 将符号 L 替换为符号 y，并验证 Piecewise 对象在简化后是否与期望的结果相等
    assert p.subs(L, y).simplify() == Piecewise(
        (nan, x <= 0), (-x + y/2, x < Max(0, y)), (0, x < y), (nan, True))
# 定义一个函数用于测试 Piecewise 对象的求解功能
def test_piecewise_solve():
    # 创建一个 Piecewise 对象，表示绝对值函数的定义
    abs2 = Piecewise((-x, x <= 0), (x, x > 0))
    # 替换 x 为 x - 2，得到新的函数 f
    f = abs2.subs(x, x - 2)
    # 断言求解 f(x) = 0 的结果为 [2]
    assert solve(f, x) == [2]
    # 断言求解 f(x) - 1 = 0 的结果为 [1, 3]
    assert solve(f - 1, x) == [1, 3]

    # 创建另一个 Piecewise 对象，包含二次函数和常数
    f = Piecewise(((x - 2)**2, x >= 0), (1, True))
    # 断言求解 f(x) = 0 的结果为 [2]
    assert solve(f, x) == [2]

    # 创建一个新的 Piecewise 对象 g，其定义中包含先前定义的 f
    g = Piecewise(((x - 5)**5, x >= 4), (f, True))
    # 断言求解 g(x) = 0 的结果为 [2, 5]
    assert solve(g, x) == [2, 5]

    # 修改 g 的定义，将 f 作为第二个条件
    g = Piecewise(((x - 5)**5, x >= 4), (f, x < 4))
    # 断言求解 g(x) = 0 的结果为 [2, 5]
    assert solve(g, x) == [2, 5]

    # 修改 g 的定义，将 f 作为第二个条件，并改变区间
    g = Piecewise(((x - 5)**5, x >= 2), (f, x < 2))
    # 断言求解 g(x) = 0 的结果为 [5]
    assert solve(g, x) == [5]

    # 修改 g 的定义，将 f 作为第二个条件，并且设为默认条件
    g = Piecewise(((x - 5)**5, x >= 2), (f, True))
    # 断言求解 g(x) = 0 的结果为 [5]
    assert solve(g, x) == [5]

    # 修改 g 的定义，增加一个不可能的条件
    g = Piecewise(((x - 5)**5, x >= 2), (f, True), (10, False))
    # 断言求解 g(x) = 0 的结果为 [5]
    assert solve(g, x) == [5]

    # 修改 g 的定义，包含更复杂的条件
    g = Piecewise(((x - 5)**5, x >= 2),
                  (-x + 2, x - 2 <= 0), (x - 2, x - 2 > 0))
    # 断言求解 g(x) = 0 的结果为 [5]
    assert solve(g, x) == [5]

    # 测试没有指定符号的 Piecewise 对象的求解
    assert solve(Piecewise((x - 2, x > 2), (2 - x, True)) - 3) == [-1, 5]

    # 创建一个包含未实现的条件的 Piecewise 对象
    f = Piecewise(((x - 2)**2, x >= 0), (0, True))
    # 断言调用 solve 函数时会抛出 NotImplementedError
    raises(NotImplementedError, lambda: solve(f, x))

    # 定义一个函数，用于过滤掉 NaN 值
    def nona(ans):
        return list(filter(lambda x: x is not S.NaN, ans))
    # 创建一个依赖于 y 的 Piecewise 对象 p
    p = Piecewise((x**2 - 4, x < y), (x - 2, True))
    # 求解 p(x) = 0 的结果并检查多个情况下的有效性
    ans = solve(p, x)
    assert nona([i.subs(y, -2) for i in ans]) == [2]
    assert nona([i.subs(y, 2) for i in ans]) == [-2, 2]
    assert nona([i.subs(y, 3) for i in ans]) == [-2, 2]
    # 断言结果与预期的 Piecewise 对象列表匹配
    assert ans == [
        Piecewise((-2, y > -2), (S.NaN, True)),
        Piecewise((2, y <= 2), (S.NaN, True)),
        Piecewise((2, y > 2), (S.NaN, True))]

    # 测试一个包含不同分支定义的 Piecewise 对象
    absxm3 = Piecewise(
        (x - 3, 0 <= x - 3),
        (3 - x, 0 > x - 3)
    )
    # 断言求解 absxm3 - y = 0 的结果与预期的 Piecewise 对象列表匹配
    assert solve(absxm3 - y, x) == [
        Piecewise((-y + 3, -y < 0), (S.NaN, True)),
        Piecewise((y + 3, y >= 0), (S.NaN, True))]
    
    # 创建一个依赖于正数符号 p 的 Piecewise 对象
    p = Symbol('p', positive=True)
    # 断言求解 absxm3 - p = 0 的结果与预期匹配
    assert solve(absxm3 - p, x) == [-p + 3, p + 3]

    # 测试一个包含函数调用的 Piecewise 对象
    f = Function('f')
    # 断言求解 -f(x) = Piecewise((1, x > 0), (0, True)) 的结果与预期匹配
    assert solve(Eq(-f(x), Piecewise((1, x > 0), (0, True))), f(x)) == \
        [Piecewise((-1, x > 0), (0, True))]

    # 测试一个包含条件与逻辑运算的 Piecewise 对象
    f = Piecewise((2*x**2, And(0 < x, x < 1)), (2, True))
    # 断言求解 f(x) - 1 = 0 的结果与预期匹配
    assert solve(f - 1) == [1/sqrt(2)]


# 定义一个函数用于测试 Piecewise 对象的折叠功能
def test_piecewise_fold():
    # 创建一个简单的 Piecewise 对象 p
    p = Piecewise((x, x < 1), (1, 1 <= x))

    # 断言对 p*x 进行折叠后结果与预期匹配
    assert piecewise_fold(x*p) == Piecewise((x**2, x < 1), (x, 1 <= x))
    
    # 断言对 p + p 进行折叠后结果与预期匹配
    assert piecewise_fold(p + p) == Piecewise((2*x, x < 1), (2, 1 <= x))
    
    # 断言对两个 Piecewise 对象相加后进行折叠结果与预期匹配
    assert piecewise_fold(Piecewise((1, x < 0), (2, True))
                          + Piecewise((10, x < 0), (-10, True))) == \
        Piecewise((11, x < 0), (-8, True))

    # 创建两个复杂的 Piecewise 对象 p1 和 p2
    p1 = Piecewise((0, x < 0), (x, x <= 1), (0, True))
    p2 = Piecewise((0, x < 0), (1 - x, x <= 1), (0, True))

    # 创建一个复杂的 Piecewise 对象 p，并进行折叠和积分的比较
    p = 4*p1 + 2*p2
    assert integrate(
        piecewise_fold(p), (x, -oo, oo)) == integrate(2*x + 2, (x, 0, 1))

    # 断言对复杂条件的 Piecewise 对象进行折叠后结果与预期匹配
    assert piecewise_fold(
        Piecewise((1, y <= 0), (-Piecewise((2, y >= 0)), True)
        )) == Piecewise((1, y <= 0), (-2, y >= 0))
    # 断言语句：验证 piecewise_fold 函数对于给定的输入是否产生预期的输出
    
    # 定义两个 Piecewise 对象 a 和 b，分别描述 x 和 y 的分段函数
    a, b = (Piecewise((2, Eq(x, 0)), (0, True)),
        Piecewise((x, Eq(-x + y, 0)), (1, Eq(-x + y, 1)), (0, True)))
    
    # 断言语句：验证 Mul 函数在 evaluate=False 的情况下对 a 和 b 的乘积是否满足交换律
    assert piecewise_fold(Mul(a, b, evaluate=False)
        ) == piecewise_fold(Mul(b, a, evaluate=False))
# 定义测试函数，用于验证 Piecewise 类的条件分段函数的行为
def test_piecewise_fold_piecewise_in_cond():
    # 定义第一个 Piecewise 对象 p1，根据 x 的值条件返回 cos(x) 或 0
    p1 = Piecewise((cos(x), x < 0), (0, True))
    # 定义第二个 Piecewise 对象 p2，根据 p1 是否为 0 返回 0 或 p1 / |p1|
    p2 = Piecewise((0, Eq(p1, 0)), (p1 / Abs(p1), True))
    # 断言验证特定值处 p2 的计算结果
    assert p2.subs(x, -pi/2) == 0
    assert p2.subs(x, 1) == 0
    assert p2.subs(x, -pi/4) == 1
    
    # 定义第三个 Piecewise 对象 p4，根据 p1 是否为 0 返回 0 或 1
    p4 = Piecewise((0, Eq(p1, 0)), (1, True))
    # 调用 piecewise_fold 函数对 p4 进行简化
    ans = piecewise_fold(p4)
    # 遍历特定范围内的值，验证简化后的结果与原始函数在这些点的计算结果一致
    for i in range(-1, 1):
        assert ans.subs(x, i) == p4.subs(x, i)

    # 定义关系表达式 r1，判断 1 是否小于第一个 Piecewise 对象的值
    r1 = 1 < Piecewise((1, x < 1), (3, True))
    # 调用 piecewise_fold 函数对 r1 进行简化
    ans = piecewise_fold(r1)
    # 遍历特定范围内的值，验证简化后的结果与原始函数在这些点的计算结果一致
    for i in range(2):
        assert ans.subs(x, i) == r1.subs(x, i)

    # 定义第四个 Piecewise 对象 p5 和 p6，根据条件返回不同的值
    p5 = Piecewise((1, x < 0), (3, True))
    p6 = Piecewise((1, x < 1), (3, True))
    # 定义第七个 Piecewise 对象 p7，根据 p5 是否小于 p6 返回 1 或 0
    p7 = Piecewise((1, p5 < p6), (0, True))
    # 调用 piecewise_fold 函数对 p7 进行简化
    ans = piecewise_fold(p7)
    # 遍历特定范围内的值，验证简化后的结果与原始函数在这些点的计算结果一致
    for i in range(-1, 2):
        assert ans.subs(x, i) == p7.subs(x, i)


# 定义另一个测试函数，用于验证 Piecewise 类的条件分段函数的行为
def test_piecewise_fold_piecewise_in_cond_2():
    # 定义第一个 Piecewise 对象 p1，根据 x 的值条件返回 cos(x) 或 0
    p1 = Piecewise((cos(x), x < 0), (0, True))
    # 定义第二个 Piecewise 对象 p2，根据 p1 是否为 0 返回 0 或 1/p1
    p2 = Piecewise((0, Eq(p1, 0)), (1 / p1, True))
    # 定义第三个 Piecewise 对象 p3，根据不同的条件返回不同的值
    p3 = Piecewise(
        (0, (x >= 0) | Eq(cos(x), 0)),
        (1/cos(x), x < 0),
        (zoo, True))  # redundant b/c all x are already covered
    # 断言验证 p2 的简化结果与 p3 相等
    assert(piecewise_fold(p2) == p3)


# 定义测试函数，验证 Piecewise 对象在 expand 函数下的行为
def test_piecewise_fold_expand():
    # 定义第一个 Piecewise 对象 p1，在区间 (0, 1] 内返回 1，否则返回 0
    p1 = Piecewise((1, Interval(0, 1, False, True).contains(x)), (0, True))

    # 对 expand((1 - x)*p1) 进行简化
    p2 = piecewise_fold(expand((1 - x)*p1))
    # 定义条件 cond，表示 x 属于 [0, 1) 区间
    cond = ((x >= 0) & (x < 1))
    # 断言验证简化结果与预期 Piecewise 对象相等，evaluate=False 表示不进行求值
    assert piecewise_fold(expand((1 - x)*p1), evaluate=False) == Piecewise((1 - x, cond), (-x, cond), (1, cond), (0, True), evaluate=False)
    # 断言验证简化结果与预期 Piecewise 对象相等，evaluate=None 表示只简化不求值
    assert piecewise_fold(expand((1 - x)*p1), evaluate=None) == Piecewise((1 - x, cond), (0, True))
    # 断言验证 p2 与预期 Piecewise 对象相等
    assert p2 == Piecewise((1 - x, cond), (0, True))
    # 断言验证 p2 与 expand((1 - x)*p1) 的结果相等
    assert p2 == expand(piecewise_fold((1 - x)*p1))


# 定义测试函数，验证 Piecewise 对象的重复性质
def test_piecewise_duplicate():
    # 定义 Piecewise 对象 p，根据不同条件返回不同的表达式
    p = Piecewise((x, x < -10), (x**2, x <= -1), (x, 1 < x))
    # 断言验证 p 与 Piecewise 对象 *p.args 相等
    assert p == Piecewise(*p.args)


# 定义测试函数，验证 Piecewise 对象在 doit 方法下的行为
def test_doit():
    # 定义第一个 Piecewise 对象 p1，根据不同的条件返回不同的表达式
    p1 = Piecewise((x, x < 1), (x**2, -1 <= x), (x, 3 < x))
    # 定义第二个 Piecewise 对象 p2，其中包含积分表达式
    p2 = Piecewise((x, x < 1), (Integral(2 * x), -1 <= x), (x, 3 < x))
    # 断言验证 p2.doit() 的计算结果与 p1 相等
    assert p2.doit() == p1
    # 断言验证 p2.doit(deep=False) 的计算结果与 p2 自身相等
    assert p2.doit(deep=False) == p2
    # issue 17165
    # 定义第三个 Piecewise 对象 p1，包含一个求和表达式，并对其进行 doit() 操作
    p1 = Sum(y**x, (x, -1, oo)).doit()
    # 断言验证 p1.doit() 的计算结果与 p1 自身相等
    assert p1.doit() == p1


# 定义测试函数，验证 Piecewise 对象在 Interval 区间条件下的行为
def test_piecewise_interval():
    # 定义第一个 Piecewise 对象 p1，在区间 (0, 1) 内返回 x，否则返回 0
    p1 = Piecewise((x, Interval(0, 1).contains(x)), (0, True))
    # 断言验证在特定点的计算结果
    assert p1.subs(x, -0.5) == 0
    assert p1.subs(x, 0.5) == 0.5
    # 断言验证 p1 对 x 的导数计算结果
    assert p1.diff(x) == Piecewise((1, Interval(0, 1).contains(x)), (0, True))
    # 断言验证 p1 在区间上的积分计算结果
    assert integrate(p1, x) == Piecewise(
        (0, x <= 0),
        (x**2/2, x <= 1),
        (S.Half, True))


# 定义测试函数，验证 piecewise_exclusive 函数对 Piecewise 对象的处理
def test
    # 断言语句：验证 piecewise_exclusive 函数对给定 Piecewise 对象的返回值是否符合预期

    # 第一个断言：验证包含嵌套 Piecewise 的 Piecewise 对象
    assert piecewise_exclusive(Piecewise((1, y <= 0),
                                         (-Piecewise((2, y >= 0)), True))) == \
        Piecewise((1, y <= 0),
                  (-Piecewise((2, y >= 0),
                              (S.NaN, y < 0), evaluate=False), y > 0), evaluate=False)

    # 第二个断言：验证简单的 Piecewise 对象
    assert piecewise_exclusive(Piecewise((1, x > y))) == Piecewise((1, x > y),
                                                                  (S.NaN, x <= y),
                                                                  evaluate=False)

    # 第三个断言：验证带有 skip_nan=True 参数的 Piecewise 对象
    assert piecewise_exclusive(Piecewise((1, x > y)),
                               skip_nan=True) == Piecewise((1, x > y))

    # 定义实数符号 xr, yr
    xr, yr = symbols('xr, yr', real=True)

    # 创建 Piecewise 对象 p1 和 p1x
    p1 = Piecewise((1, xr < 0), (2, True), evaluate=False)
    p1x = Piecewise((1, xr < 0), (2, xr >= 0), evaluate=False)

    # 创建 Piecewise 对象 p2, p2x, p2xx
    p2 = Piecewise((p1, yr < 0), (3, True), evaluate=False)
    p2x = Piecewise((p1, yr < 0), (3, yr >= 0), evaluate=False)
    p2xx = Piecewise((p1x, yr < 0), (3, yr >= 0), evaluate=False)

    # 断言语句：验证 piecewise_exclusive 函数对给定 Piecewise 对象的返回值是否符合预期
    assert piecewise_exclusive(p2) == p2xx

    # 断言语句：验证带有 deep=False 参数的 piecewise_exclusive 函数对给定 Piecewise 对象的返回值是否符合预期
    assert piecewise_exclusive(p2, deep=False) == p2x
# 定义一个测试函数，用于测试 Piecewise 对象的行为
def test_piecewise_collapse():
    # 断言 Piecewise((x, True)) 等于 x
    assert Piecewise((x, True)) == x
    
    # 设定条件 a = x < 1
    a = x < 1
    # 断言 Piecewise((x, a), (x + 1, a)) 等于 Piecewise((x, a))
    assert Piecewise((x, a), (x + 1, a)) == Piecewise((x, a))
    # 断言 Piecewise((x, a), (x + 1, a.reversed)) 等于 Piecewise((x, a))
    assert Piecewise((x, a), (x + 1, a.reversed)) == Piecewise((x, a))
    
    # 设定条件 b = x < 5
    b = x < 5
    
    # 定义函数 canonical(i)，用于处理 Piecewise 对象中的子表达式
    def canonical(i):
        if isinstance(i, Piecewise):
            return Piecewise(*i.args)
        return i
    
    # 循环测试不同参数下的 canonical 函数行为
    for args in [
        ((1, a), (Piecewise((2, a), (3, b)), b)),
        ((1, a), (Piecewise((2, a), (3, b.reversed)), b)),
        ((1, a), (Piecewise((2, a), (3, b)), b), (4, True)),
        ((1, a), (Piecewise((2, a), (3, b), (4, True)), b)),
        ((1, a), (Piecewise((2, a), (3, b), (4, True)), b), (5, True))
    ]:
        for i in (0, 2, 10):
            # 断言 canonical(Piecewise(*args, evaluate=False).subs(x, i)) 等于 canonical(Piecewise(*args).subs(x, i))
            assert canonical(
                Piecewise(*args, evaluate=False).subs(x, i)
            ) == canonical(Piecewise(*args).subs(x, i))
    
    # 定义符号变量 r1, r2, r3, r4
    r1, r2, r3, r4 = symbols('r1:5')
    
    # 根据不同的条件定义 a, b, c, d
    a = x < r1
    b = x < r2
    c = x < r3
    d = x < r4
    
    # 多个 Piecewise 对象的比较和简化
    assert Piecewise((1, a), (Piecewise((2, a), (3, b), (4, c)), b), (5, c)) == Piecewise((1, a), (3, b), (5, c))
    assert Piecewise((1, a), (Piecewise((2, a), (3, b), (4, c), (6, True)), c), (5, d)) == Piecewise((1, a), (Piecewise((3, b), (4, c)), c), (5, d))
    assert Piecewise((1, Or(a, d)), (Piecewise((2, d), (3, b), (4, c)), b), (5, c)) == Piecewise((1, Or(a, d)), (Piecewise((2, d), (3, b)), b), (5, c))
    assert Piecewise((1, c), (2, ~c), (3, S.true)) == Piecewise((1, c), (2, S.true))
    assert Piecewise((1, c), (2, And(~c, b)), (3, True)) == Piecewise((1, c), (2, b), (3, True))
    
    # 将变量替换为特定的值，然后进行比较
    assert Piecewise((1, c), (2, Or(~c, b)), (3, True)).subs(dict(zip((r1, r2, r3, r4, x), (1, 2, 3, 4, 3.5)))) == 2
    assert Piecewise((1, c), (2, ~c)) == Piecewise((1, c), (2, True))


# 定义一个测试函数，测试 Piecewise 对象的 lambdify 方法
def test_piecewise_lambdify():
    # 定义一个 Piecewise 对象 p
    p = Piecewise(
        (x**2, x < 0),
        (x, Interval(0, 1, False, True).contains(x)),
        (2 - x, x >= 1),
        (0, True)
    )
    
    # 将 p 转换为一个函数 f，并进行断言验证
    f = lambdify(x, p)
    assert f(-2.0) == 4.0
    assert f(0.0) == 0.0
    assert f(0.5) == 0.5
    assert f(2.0) == 0.0


# 定义一个测试函数，测试 Piecewise 对象的级数展开
def test_piecewise_series():
    # 导入 sympy 库中的级数展开相关模块
    from sympy.series.order import O
    
    # 定义两个 Piecewise 对象 p1 和 p2，验证它们的级数展开是否相等
    p1 = Piecewise((sin(x), x < 0), (cos(x), x > 0))
    p2 = Piecewise((x + O(x**2), x < 0), (1 + O(x**2), x > 0))
    assert p1.nseries(x, n=2) == p2


# 定义一个测试函数，测试 Piecewise 对象的主导项
def test_piecewise_as_leading_term():
    # 定义多个 Piecewise 对象，验证它们的主导项是否符合预期
    p1 = Piecewise((1/x, x > 1), (0, True))
    p2 = Piecewise((x, x > 1), (0, True))
    p3 = Piecewise((1/x, x > 1), (x, True))
    p4 = Piecewise((x, x > 1), (1/x, True))
    p5 = Piecewise((1/x, x > 1), (x, True))
    p6 = Piecewise((1/x, x < 1), (x, True))
    p7 = Piecewise((x, x < 1), (1/x, True))
    p8 = Piecewise((x, x > 1), (1/x, True))
    
    # 断言各个 Piecewise 对象的主导项是否正确
    assert p1.as_leading_term(x) == 0
    assert p2.as_leading_term(x) == 0
    assert p3.as_leading_term(x) == x
    assert p4.as_leading_term(x) == 1/x
    assert p5.as_leading_term(x) == x
    assert p6.as_leading_term(x) == 1/x
    # 断言语句：验证多项式 p7 的主导项是 x
    assert p7.as_leading_term(x) == x
    
    # 断言语句：验证多项式 p8 的主导项是 1/x
    assert p8.as_leading_term(x) == 1/x
# 定义测试函数，用于测试复杂分段函数的各种属性和方法
def test_piecewise_complex():
    # 创建四个复杂分段函数对象，每个对象由条件和表达式组成
    p1 = Piecewise((2, x < 0), (1, 0 <= x))
    p2 = Piecewise((2*I, x < 0), (I, 0 <= x))
    p3 = Piecewise((I*x, x > 1), (1 + I, True))
    p4 = Piecewise((-I*conjugate(x), x > 1), (1 - I, True))

    # 断言共轭函数应用于 p1 时等于 p1 本身
    assert conjugate(p1) == p1
    # 断言共轭函数应用于 p2 时等于对 p2 进行折叠（piecewise_fold）的结果
    assert conjugate(p2) == piecewise_fold(-p2)
    # 断言共轭函数应用于 p3 时等于 p4
    assert conjugate(p3) == p4

    # 断言 p1 的虚部属性为 False，实部属性为 True
    assert p1.is_imaginary is False
    assert p1.is_real is True
    # 断言 p2 的虚部属性为 True，实部属性为 False
    assert p2.is_imaginary is True
    assert p2.is_real is False
    # 断言 p3 的虚部和实部属性为 None
    assert p3.is_imaginary is None
    assert p3.is_real is None

    # 断言 p1 的实部和虚部分别为 (p1, 0)
    assert p1.as_real_imag() == (p1, 0)
    # 断言 p2 的实部和虚部分别为 (0, -I*p2)
    assert p2.as_real_imag() == (0, -I*p2)


# 定义测试函数，用于测试共轭转置等操作
def test_conjugate_transpose():
    # 声明符号 A 和 B，并确保它们不可交换
    A, B = symbols("A B", commutative=False)
    # 创建一个复杂分段函数 p
    p = Piecewise((A*B**2, x > 0), (A**2*B, True))
    
    # 断言 p 的伴随（adjoint）等于对条件和表达式分别求伴随得到的复杂分段函数
    assert p.adjoint() == Piecewise((adjoint(A*B**2), x > 0), (adjoint(A**2*B), True))
    # 断言 p 的共轭等于对条件和表达式分别求共轭得到的复杂分段函数
    assert p.conjugate() == Piecewise((conjugate(A*B**2), x > 0), (conjugate(A**2*B), True))
    # 断言 p 的转置等于对条件和表达式分别求转置得到的复杂分段函数
    assert p.transpose() == Piecewise((transpose(A*B**2), x > 0), (transpose(A**2*B), True))


# 定义测试函数，用于测试分段函数的求值
def test_piecewise_evaluate():
    # 断言单一条件为 True 的分段函数等于其表达式 x
    assert Piecewise((x, True)) == x
    # 断言带有 evaluate=True 参数的分段函数等于其表达式 x
    assert Piecewise((x, True), evaluate=True) == x
    # 断言带有不同条件的分段函数的 args 属性正确构造
    assert Piecewise((1, Eq(1, x))).args == ((1, Eq(x, 1)),)
    # 断言带有 evaluate=False 参数的分段函数的 args 属性正确构造
    assert Piecewise((1, Eq(1, x)), evaluate=False).args == ((1, Eq(1, x)),)
    # 创建一个带有 evaluate=False 参数的分段函数 p，并断言其等于表达式 x
    p = Piecewise((x, True), evaluate=False)
    assert p == x


# 定义测试函数，用于测试分段函数的表达式和集合对的转换
def test_as_expr_set_pairs():
    # 断言带有两个条件的分段函数的 as_expr_set_pairs 方法正确构造表达式和集合对
    assert Piecewise((x, x > 0), (-x, x <= 0)).as_expr_set_pairs() == [(x, Interval(0, oo, True, True)), (-x, Interval(-oo, 0))]

    # 断言带有两个条件的另一个分段函数的 as_expr_set_pairs 方法正确构造表达式和集合对
    assert Piecewise(((x - 2)**2, x >= 0), (0, True)).as_expr_set_pairs() == [((x - 2)**2, Interval(0, oo)), (0, Interval(-oo, 0, True, True))]


# 定义测试函数，用于测试分段函数的 srepr 方法是否为自恒等式
def test_S_srepr_is_identity():
    # 创建一个分段函数 p，并断言其与其自身的 srepr 生成的对象相等
    p = Piecewise((10, Eq(x, 0)), (12, True))
    q = S(srepr(p))
    assert p == q


# 定义测试函数，用于测试分段函数处理特定问题的集成
def test_issue_12587():
    # 创建一个包含三个条件的复杂分段函数 p，断言其在指定区间上的积分结果为 23
    p = Piecewise((1, x > 4), (2, Not((x <= 3) & (x > -1))), (3, True))
    assert p.integrate((x, -5, 5)) == 23
    
    # 创建一个包含三个条件的另一个复杂分段函数 p，并断言其在指定区间上的积分结果
    p = Piecewise((1, x > 1), (2, x < y), (3, True))
    lim = x, -3, 3
    ans = p.integrate(lim)
    for i in range(-1, 3):
        assert ans.subs(y, i) == p.subs(y, i).integrate(lim)


# 定义测试函数，用于测试分段函数处理特定问题的集成
def test_issue_11045():
    # 断言对给定函数的积分结果
    assert integrate(1/(x*sqrt(x**2 - 1)), (x, 1, 2)) == pi/3

    # 断言带有复杂 And 条件的分段函数的积分结果
    assert Piecewise((1, And(Or(x < 1, x > 3), x < 2)), (0, True)).integrate((x, 0, 3)) == 1

    # 断言带有 hidden false 条件的分段函数的积分结果
    assert Piecewise((1, x > 1), (2, x > x + 1), (3, True)).integrate((x, 0, 3)) == 5

    # 断言带有 targetcond 是 Eq 条件的分段函数的积分结果
    assert Piecewise((1, x > 1), (2, Eq(1, x)), (3, True)).integrate((x, 0, 4)) == 6

    # 断言带有复杂 And 条件的分段函数的积分结果
    assert Piecewise((1, And(2*x > x + 1, x < 2)), (0, True)).integrate((x, 0, 3)) == 1

    # 断言带有复杂 Or 条件的分段函数的积分结果
    # 断言条件：Piecewise 对象的 integrate 方法结果应该等于 2
    assert Piecewise((1, Or(2*x > x + 2, x < 1)), (0, True)
        ).integrate((x, 0, 3)) == 2
    
    # 断言条件：Piecewise 对象的 integrate 方法结果应该等于 5
    assert Piecewise((1, x > 1), (2, x > x + 1), (3, True)
        ).integrate((x, 0, 3)) == 5
    
    # 断言条件：Piecewise 对象的 integrate 方法结果应该等于 6
    assert Piecewise((2, Eq(1 - x, x*(1/x - 1))), (0, True)
        ).integrate((x, 0, 3)) == 6

    # 断言条件：Piecewise 对象的 integrate 方法结果应该等于 6
    assert Piecewise((1, Or(x < 1, x > 2)), (2, x > 3), (3, True)
        ).integrate((x, 0, 4)) == 6

    # 断言条件：Piecewise 对象的 integrate 方法结果应该等于 2
    assert Piecewise((1, Ne(x, 0)), (2, True)
        ).integrate((x, -1, 1)) == 2

    # 断言条件：Piecewise 对象的 integrate 方法结果应该等于 5
    assert Piecewise((x, (x > 1) & (x < 3)), (1, (x < 4))
        ).integrate((x, 1, 4)) == 5

    # 创建 Piecewise 对象并进行 integrate 方法的调用，检查结果是否符合预期
    p = Piecewise((x, (x > 1) & (x < 3)), (1, (x < 4)))
    nan = Undefined
    i = p.integrate((x, 1, y))
    assert i == Piecewise(
        (y - 1, y < 1),
        (Min(3, y)**2/2 - Min(3, y) + Min(4, y) - S.Half,
            y <= Min(4, y)),
        (nan, True))
    assert p.integrate((x, 1, -1)) == i.subs(y, -1)
    assert p.integrate((x, 1, 4)) == 5
    assert p.integrate((x, 1, 5)) is nan

    # 创建 Piecewise 对象并进行 integrate 方法的调用，检查结果是否符合预期
    p = Piecewise((1, x > 1), (2, Not(And(x > 1, x < 3))), (3, True))
    assert p.integrate((x, 0, 3)) == 4

    # 创建 Piecewise 对象并进行 integrate 方法的调用，检查结果是否符合预期
    p = Piecewise(
        (1, And(5 > x, x > 1)),
        (2, Or(x < 3, x > 7)),
        (4, x < 8))
    assert p.integrate((x, 0, 10)) == 20

    # 断言条件：Piecewise 对象的 integrate 方法结果应该为 NaN
    assert Piecewise((1, x < 1), (2, And(Eq(x, 3), x > 1))
        ).integrate((x, 0, 3)) is S.NaN
    
    # 断言条件：Piecewise 对象的 integrate 方法结果应该等于 7
    assert Piecewise((1, x < 1), (2, And(Eq(x, 3), x > 1)), (3, True)
        ).integrate((x, 0, 3)) == 7
    
    # 断言条件：Piecewise 对象的 integrate 方法结果应该等于 4
    assert Piecewise((1, x < 0), (2, And(Eq(x, 3), x < 1)), (3, True)
        ).integrate((x, -1, 1)) == 4
    
    # 断言条件：Piecewise 对象的 integrate 方法结果应该等于 7
    assert Piecewise((1, x < 1), (2, Eq(x, 3) & (y < x)), (3, True)
        ).integrate((x, 0, 3)) == 7
def test_holes():
    nan = Undefined  # 定义 Undefined 作为未定义值
    assert Piecewise((1, x < 2)).integrate(x) == Piecewise(
        (x, x < 2), (nan, True))  # 对 Piecewise 函数在 x < 2 条件下的积分进行断言
    assert Piecewise((1, And(x > 1, x < 2))).integrate(x) == Piecewise(
        (nan, x < 1), (x, x < 2), (nan, True))  # 对 Piecewise 函数在 And(x > 1, x < 2) 条件下的积分进行断言
    assert Piecewise((1, And(x > 1, x < 2))).integrate((x, 0, 3)) is nan  # 对 Piecewise 函数在区间 (x, 0, 3) 下的积分结果进行断言为 nan
    assert Piecewise((1, And(x > 0, x < 4))).integrate((x, 1, 3)) == 2  # 对 Piecewise 函数在区间 (x, 1, 3) 下的积分结果进行断言为 2

    # this also tests that the integrate method is used on non-Piecewise
    # arguments in _eval_integral
    A, B = symbols("A B")  # 定义符号 A 和 B
    a, b = symbols('a b', real=True)  # 定义实数符号 a 和 b
    assert Piecewise((A, And(x < 0, a < 1)), (B, Or(x < 1, a > 2))
        ).integrate(x) == Piecewise(
        (B*x, (a > 2)),  # 对 Piecewise 函数在 a > 2 条件下的积分结果进行断言为 B*x
        (Piecewise((A*x, x < 0), (B*x, x < 1), (nan, True)), a < 1),  # 对 Piecewise 函数在 a < 1 条件下的积分结果进行断言
        (Piecewise((B*x, x < 1), (nan, True)), True))  # 对 Piecewise 函数在其他条件下的积分结果进行断言


def test_issue_11922():
    def f(x):
        return Piecewise((0, x < -1), (1 - x**2, x < 1), (0, True))  # 定义 Piecewise 函数 f(x)
    autocorr = lambda k: (
        f(x) * f(x + k)).integrate((x, -1, 1))  # 定义自相关函数 autocorr，计算 f(x) * f(x + k) 在 x 范围 (-1, 1) 内的积分
    assert autocorr(1.9) > 0  # 对自相关函数在 k = 1.9 处的结果进行断言，应大于 0
    k = symbols('k')
    good_autocorr = lambda k: (
        (1 - x**2) * f(x + k)).integrate((x, -1, 1))  # 定义修正后的自相关函数 good_autocorr，计算 (1 - x**2) * f(x + k) 在 x 范围 (-1, 1) 内的积分
    a = good_autocorr(k)
    assert a.subs(k, 3) == 0  # 对修正后的自相关函数在 k = 3 处的结果进行断言，应为 0
    k = symbols('k', positive=True)
    a = good_autocorr(k)
    assert a.subs(k, 3) == 0  # 对修正后的自相关函数在 k = 3 处的结果进行断言，应为 0
    assert Piecewise((0, x < 1), (10, (x >= 1))
        ).integrate() == Piecewise((0, x < 1), (10*x - 10, True))  # 对 Piecewise 函数在全域内的积分结果进行断言


def test_issue_5227():
    f = 0.0032513612725229*Piecewise((0, x < -80.8461538461539),
        (-0.0160799238820171*x + 1.33215984776403, x < 2),
        (Piecewise((0.3, x > 123), (0.7, True)) +
        Piecewise((0.4, x > 2), (0.6, True)), x <=
        123), (-0.00817409766454352*x + 2.10541401273885, x <
        380.571428571429), (0, True))  # 定义复杂的 Piecewise 函数 f
    i = integrate(f, (x, -oo, oo))  # 对 f 在 x 范围 (-oo, oo) 内进行积分
    assert i == Integral(f, (x, -oo, oo)).doit()  # 对积分结果进行断言
    assert str(i) == '1.00195081676351'  # 对积分结果的字符串表示进行断言
    assert Piecewise((1, x - y < 0), (0, True)
        ).integrate(y) == Piecewise((0, y <= x), (-x + y, True))  # 对 Piecewise 函数在 y 范围内的积分结果进行断言


def test_issue_10137():
    a = Symbol('a', real=True)  # 定义实数符号 a
    b = Symbol('b', real=True)  # 定义实数符号 b
    x = Symbol('x', real=True)  # 定义实数符号 x
    y = Symbol('y', real=True)  # 定义实数符号 y
    p0 = Piecewise((0, Or(x < a, x > b)), (1, True))  # 定义 Piecewise 函数 p0
    p1 = Piecewise((0, Or(a > x, b < x)), (1, True))  # 定义 Piecewise 函数 p1
    assert integrate(p0, (x, y, oo)) == integrate(p1, (x, y, oo))  # 对 p0 和 p1 在 x 范围 (y, oo) 内的积分结果进行断言相等
    p3 = Piecewise((1, And(0 < x, x < a)), (0, True))  # 定义 Piecewise 函数 p3
    p4 = Piecewise((1, And(a > x, x > 0)), (0, True))  # 定义 Piecewise 函数 p4
    ip3 = integrate(p3, x)  # 对 p3 在 x 方向进行积分
    assert ip3 == Piecewise(
        (0, x <= 0),  # 对积分结果进行条件断言
        (x, x <= Max(0, a)),  # 对积分结果进行条件断言
        (Max(0, a), True))  # 对积分结果进行条件断言
    ip4 = integrate(p4, x)  # 对 p4 在 x 方向进行积分
    assert ip4 == ip3  # 对 p4 和 p3 的积分结果进行断言相等
    assert p3.integrate((x, 2, 4)) == Min(4, Max(2, a)) - 2  # 对 p3 在区间 (x, 2, 4) 的积分结果进行断言
    assert p4.integrate((x, 2, 4)) == Min(4, Max(2, a)) - 2  # 对 p4 在区间 (x, 2, 4) 的积分结果进行断言


def test_stackoverflow_43852159():
    f = lambda x: Piecewise((1, (x >= -1) & (x <= 1)), (0, True))  # 定义 Piecewise 函数 f
    Conv = lambda x: integrate(f(x - y)*f(y), (y, -oo, +oo))  # 定义卷积函数 Conv
    cx = Conv(x)  # 计算 Conv 在 x 处的值
    assert cx.subs(x, -1.5) == cx.subs(x, 1.5)  # 对 Conv 在 x = -1.5 和 x = 1.5 处的结果进行断言相等
    assert cx.subs(x, 3) == 0  # 对 Conv 在 x = 3 处的结果进行断言为 0
    # 使用断言检查 piecewise_fold 函数对 f(x - y)*f(y) 的返回结果是否等于 Piecewise 对象
    assert piecewise_fold(f(x - y)*f(y)) == Piecewise(
        # 如果 y 在 [-1, 1] 且 x - y 在 [-1, 1] 范围内，则返回 1
        (1, (y >= -1) & (y <= 1) & (x - y >= -1) & (x - y <= 1)),
        # 否则返回 0
        (0, True))
def test_issue_12557():
    '''
    # 导入 sympy 库，并声明符号变量 x, y, z, t, k
    import sympy as sym
    x,y,z,t = sym.symbols('x y z t')
    k = sym.symbols("k", integer=True)
    # 计算 sympy 中的傅里叶级数，cos(k*x)*sqrt(x**2)，在区间 [-pi, pi] 上
    fourier = sym.fourier_series(sym.cos(k*x)*sym.sqrt(x**2),
                                 (x, -sym.pi, sym.pi))
    # 断言计算结果是否与期望的 FourierSeries 对象相等
    assert fourier == FourierSeries(
    sqrt(x**2)*cos(k*x), (x, -pi, pi), (Piecewise((pi**2,
    Eq(k, 0)), (2*(-1)**k/k**2 - 2/k**2, True))/(2*pi),
    SeqFormula(Piecewise((pi**2, (Eq(_n, 0) & Eq(k, 0)) | (Eq(_n, 0) &
    Eq(_n, k) & Eq(k, 0)) | (Eq(_n, 0) & Eq(k, 0) & Eq(_n, -k)) | (Eq(_n,
    0) & Eq(_n, k) & Eq(k, 0) & Eq(_n, -k))), (pi**2/2, Eq(_n, k) | Eq(_n,
    -k) | (Eq(_n, 0) & Eq(_n, k)) | (Eq(_n, k) & Eq(k, 0)) | (Eq(_n, 0) &
    Eq(_n, -k)) | (Eq(_n, k) & Eq(_n, -k)) | (Eq(k, 0) & Eq(_n, -k)) |
    (Eq(_n, 0) & Eq(_n, k) & Eq(_n, -k)) | (Eq(_n, k) & Eq(k, 0) & Eq(_n,
    -k))), ((-1)**k*pi**2*_n**3*sin(pi*_n)/(pi*_n**4 - 2*pi*_n**2*k**2 +
    pi*k**4) - (-1)**k*pi**2*_n**3*sin(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2
    - pi*k**4) + (-1)**k*pi*_n**2*cos(pi*_n)/(pi*_n**4 - 2*pi*_n**2*k**2 +
    pi*k**4) - (-1)**k*pi*_n**2*cos(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2 -
    pi*k**4) - (-1)**k*pi**2*_n*k**2*sin(pi*_n)/(pi*_n**4 -
    2*pi*_n**2*k**2 + pi*k**4) +
    (-1)**k*pi**2*_n*k**2*sin(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2 -
    pi*k**4) + (-1)**k*pi*k**2*cos(pi*_n)/(pi*_n**4 - 2*pi*_n**2*k**2 +
    pi*k**4) - (-1)**k*pi*k**2*cos(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2 -
    pi*k**4) - (2*_n**2 + 2*k**2)/(_n**4 - 2*_n**2*k**2 + k**4),
    True))*cos(_n*x)/pi, (_n, 1, oo)), SeqFormula(0, (_k, 1, oo))))
    '''
    # 声明实数符号变量 x 和整数符号变量 k
    x = symbols("x", real=True)
    k = symbols('k', integer=True, finite=True)
    # 定义函数 abs2(x)，表示 x 的绝对值函数
    abs2 = lambda x: Piecewise((-x, x <= 0), (x, x > 0))
    # 断言 abs2(x) 在区间 [-pi, pi] 上的积分结果是否等于 pi^2
    assert integrate(abs2(x), (x, -pi, pi)) == pi**2
    # 定义函数 func = cos(k*x)*sqrt(x**2)
    func = cos(k*x)*sqrt(x**2)
    # 断言 func 在区间 [-pi, pi] 上的积分结果是否与预期的 Piecewise 对象相等
    assert integrate(func, (x, -pi, pi)) == Piecewise(
        (2*(-1)**k/k**2 - 2/k**2, Ne(k, 0)), (pi**2, True))

def test_issue_6900():
    # 导入 permutations 函数
    from itertools import permutations
    # 声明符号变量 t0, t1, T, t
    t0, t1, T, t = symbols('t0, t1 T t')
    # 定义 Piecewise 函数 f，表示分段函数
    f = Piecewise((0, t < t0), (x, And(t0 <= t, t < t1)), (0, t >= t1))
    # 计算 f 对 t 的积分，赋值给 g
    g = f.integrate(t)
    # 断言 g 的值与预期的 Piecewise 对象相等
    assert g == Piecewise(
        (0, t <= t0),
        (t*x - t0*x, t <= Max(t0, t1)),
        (-t0*x + x*Max(t0, t1), True))
    # 遍历 permutations(range(2)) 的所有排列
    for i in permutations(range(2)):
        # 构造替换字典 reps，并进行断言
        reps = dict(zip((t0,t1), i))
        for tt in range(-1,3):
            # 断言替换后的表达式是否相等
            assert (g.xreplace(reps).subs(t,tt) ==
                f.xreplace(reps).integrate(t).subs(t,tt))
    # 构造 Tuple 对象 lim = (t, t0, T)，计算 f 对 lim 的积分，赋值给 g
    lim = Tuple(t, t0, T)
    g = f.integrate(lim)
    # 定义预期的 Piecewise 对象 ans
    ans = Piecewise(
        (-t0*x + x*Min(T, Max(t0, t1)), T > t0),
        (0, True))
    # 遍历 permutations(range(3)) 的所有排列
    for i in permutations(range(3)):
        # 构造替换字典 reps，并进行断言
        reps = dict(zip((t0,t1,T), i))
        tru = f.xreplace(reps).integrate(lim.xreplace(reps))
        assert tru == ans.xreplace(reps)
    # 最终断言 g 的值与预期的 ans 对象相等
    assert g == ans


def test_issue_10122():
    # 断言求解 abs(x) + abs(x - 1) - 1 > 0 的结果是否等于 Or(And(-oo < x, x < S.Zero), And(S.One < x, x < oo))
    assert solve(abs(x) + abs(x - 1) - 1 > 0, x
        ) == Or(And(-oo < x, x < S.Zero), And(S.One < x, x < oo))
# 定义一个名为 test_issue_4313 的测试函数
def test_issue_4313():
    # 定义分段函数 u，根据不同条件返回不同的值
    u = Piecewise((0, x <= 0), (1, x >= a), (x/a, True))
    # 定义表达式 e，其中包含 u 的子表达式
    e = (u - u.subs(x, y))**2/(x - y)**2
    # 计算 Max(0, a) 的值并赋给 M
    M = Max(0, a)
    # 断言积分结果是否等于一个分段函数
    assert integrate(e, x).expand() == Piecewise(
        # 第一个分段
        (Piecewise(
            (0, x <= 0),
            # 在 x <= M 时的表达式
            (-y**2/(a**2*x - a**2*y) + x/a**2 - 2*y*log(-y)/a**2 +
                2*y*log(x - y)/a**2 - y/a**2, x <= M),
            # 默认情况下的表达式
            (-y**2/(-a**2*y + a**2*M) + 1/(-y + M) -
                1/(x - y) - 2*y*log(-y)/a**2 + 2*y*log(-y +
                M)/a**2 - y/a**2 + M/a**2, True)),
        # 第二个分段
        ((a <= y) & (y <= 0)) | ((y <= 0) & (y > -oo))),
        # 第三个分段
        (Piecewise(
            (-1/(x - y), x <= 0),
            (-a**2/(a**2*x - a**2*y) + 2*a*y/(a**2*x - a**2*y) -
                y**2/(a**2*x - a**2*y) + 2*log(-y)/a - 2*log(x - y)/a +
                2/a + x/a**2 - 2*y*log(-y)/a**2 + 2*y*log(x - y)/a**2 -
                y/a**2, x <= M),
            (-a**2/(-a**2*y + a**2*M) + 2*a*y/(-a**2*y +
                a**2*M) - y**2/(-a**2*y + a**2*M) +
                2*log(-y)/a - 2*log(-y + M)/a + 2/a -
                2*y*log(-y)/a**2 + 2*y*log(-y + M)/a**2 -
                y/a**2 + M/a**2, True)),
        a <= y),
        # 第四个分段
        (Piecewise(
            (-y**2/(a**2*x - a**2*y), x <= 0),
            (x/a**2 + y/a**2, x <= M),
            (a**2/(-a**2*y + a**2*M) -
                a**2/(a**2*x - a**2*y) - 2*a*y/(-a**2*y + a**2*M) +
                2*a*y/(a**2*x - a**2*y) + y**2/(-a**2*y + a**2*M) -
                y**2/(a**2*x - a**2*y) + y/a**2 + M/a**2, True)),
        True))


# 定义一个名为 test__intervals 的测试函数
def test__intervals():
    # 断言 Piecewise 对象的 _intervals 方法调用结果
    assert Piecewise((x + 2, Eq(x, 3)))._intervals(x) == (True, [])
    assert Piecewise(
        # 第一个分段
        (1, x > x + 1),
        # 第二个分段
        (Piecewise((1, x < x + 1)), 2*x < 2*x + 1),
        (1, True))._intervals(x) == (True, [(-oo, oo, 1, 1)])
    assert Piecewise((1, Ne(x, I)), (0, True))._intervals(x) == (True,
        [(-oo, oo, 1, 0)])
    assert Piecewise((-cos(x), sin(x) >= 0), (cos(x), True)
        )._intervals(x) == (True,
        [(0, pi, -cos(x), 0), (-oo, oo, cos(x), 1)])
    # 下面的断言测试去除重复值和零宽度非 Eq 生成的区间
    assert Piecewise((1, Abs(x**(-2)) > 1), (0, True)
        )._intervals(x) == (True,
        [(-1, 0, 1, 0), (0, 1, 1, 0), (-oo, oo, 0, 1)])


# 定义一个名为 test_containment 的测试函数
def test_containment():
    # 定义多个变量并赋值
    a, b, c, d, e = [1, 2, 3, 4, 5]
    # 定义 p 为两个 Piecewise 对象的乘积
    p = (Piecewise((d, x > 1), (e, True))*
        Piecewise((a, Abs(x - 1) < 1), (b, Abs(x - 2) < 2), (c, True)))
    # 断言 p 的积分再求导后的结果是否等于一个 Piecewise 对象
    assert p.integrate(x).diff(x) == Piecewise(
        (c*e, x <= 0),
        (a*e, x <= 1),
        (a*d, x < 2),  # 这里是我们希望正确得到的结果
        (b*d, x < 4),
        (c*d, True))


# 定义一个名为 test_piecewise_with_DiracDelta 的测试函数
def test_piecewise_with_DiracDelta():
    # 定义 DiracDelta 对象 d1
    d1 = DiracDelta(x - 1)
    # 断言在无穷区间 (-oo, oo) 上对 DiracDelta 对象的积分结果为 1
    assert integrate(d1, (x, -oo, oo)) == 1
    # 断言在区间 (0, 2) 上对 DiracDelta 对象的积分结果为 1
    assert integrate(d1, (x, 0, 2)) == 1
    # 断言 Piecewise 对象在积分后的结果
    assert Piecewise((d1, Eq(x, 2)), (0, True)).integrate(x) == 0
    assert Piecewise((d1, x < 2), (0, True)).integrate(x) == Piecewise(
        (Heaviside(x - 1), x < 2), (1, True))
    # 如果函数在积分限制处不连续，例如 integrate(d1, (x, -2, 1)) 或 Piecewise((d1, Eq(x, 1)))
    # TODO: 在这里需要添加代码来引发错误或者进行相应处理
def test_issue_10258():
    # 测试 Piecewise 对象的 is_zero 方法
    assert Piecewise((0, x < 1), (1, True)).is_zero is None
    # 测试 Piecewise 对象的 is_zero 方法
    assert Piecewise((-1, x < 1), (1, True)).is_zero is False
    # 创建一个带有零属性的符号对象 'a'
    a = Symbol('a', zero=True)
    # 测试 Piecewise 对象的 is_zero 方法
    assert Piecewise((0, x < 1), (a, True)).is_zero
    # 测试 Piecewise 对象的 is_zero 方法
    assert Piecewise((1, x < 1), (a, x < 3)).is_zero is None
    # 创建一个普通符号对象 'a'
    a = Symbol('a')
    # 测试 Piecewise 对象的 is_zero 方法
    assert Piecewise((0, x < 1), (a, True)).is_zero is None
    # 测试 Piecewise 对象的 is_nonzero 方法
    assert Piecewise((0, x < 1), (1, True)).is_nonzero is None
    # 测试 Piecewise 对象的 is_nonzero 方法
    assert Piecewise((1, x < 1), (2, True)).is_nonzero
    # 测试 Piecewise 对象的 is_finite 方法
    assert Piecewise((0, x < 1), (oo, True)).is_finite is None
    # 测试 Piecewise 对象的 is_finite 方法
    assert Piecewise((0, x < 1), (1, True)).is_finite
    # 创建一个基础符号对象 'b'
    b = Basic()
    # 测试 Piecewise 对象的 is_finite 方法
    assert Piecewise((b, x < 1)).is_finite is None

    # 10258
    # 创建一个比较对象 'c'
    c = Piecewise((1, x < 0), (2, True)) < 3
    # 测试比较结果是否不等于 True
    assert c != True
    # 测试 piecewise_fold 函数的输出结果
    assert piecewise_fold(c) == True


def test_issue_10087():
    # 创建两个 Piecewise 对象 'a' 和 'b'
    a, b = Piecewise((x, x > 1), (2, True)), Piecewise((x, x > 3), (3, True))
    # 将 a 和 b 相乘得到 m
    m = a*b
    # 对 m 进行 piecewise_fold 处理得到 f
    f = piecewise_fold(m)
    # 遍历指定点，验证 m 和 f 在这些点的取值是否相同
    for i in (0, 2, 4):
        assert m.subs(x, i) == f.subs(x, i)
    # 将 a 和 b 相加得到 m
    m = a + b
    # 对 m 进行 piecewise_fold 处理得到 f
    f = piecewise_fold(m)
    # 遍历指定点，验证 m 和 f 在这些点的取值是否相同
    for i in (0, 2, 4):
        assert m.subs(x, i) == f.subs(x, i)


def test_issue_8919():
    # 创建符号 'c' 的列表
    c = symbols('c:5')
    # 创建符号 'x'
    x = symbols("x")
    # 创建 Piecewise 对象 f1 和 f2
    f1 = Piecewise((c[1], x < 1), (c[2], True))
    f2 = Piecewise((c[3], x < Rational(1, 3)), (c[4], True))
    # 验证积分的计算结果是否符合预期
    assert integrate(f1*f2, (x, 0, 2)) == c[1]*c[3]/3 + 2*c[1]*c[4]/3 + c[2]*c[4]
    # 创建 Piecewise 对象 f1 和 f2
    f1 = Piecewise((0, x < 1), (2, True))
    f2 = Piecewise((3, x < 2), (0, True))
    # 验证积分的计算结果是否符合预期
    assert integrate(f1*f2, (x, 0, 3)) == 6

    # 创建正数符号 'y'
    y = symbols("y", positive=True)
    # 创建实数符号 'a, b, c, x, z'
    a, b, c, x, z = symbols("a,b,c,x,z", real=True)
    # 创建积分对象 I
    I = Integral(Piecewise(
        (0, (x >= y) | (x < 0) | (b > c)),
        (a, True)), (x, 0, z))
    # 计算积分并验证结果
    ans = I.doit()
    assert ans == Piecewise((0, b > c), (a*Min(y, z) - a*Min(0, z), True))
    # 对多种条件进行组合测试
    for cond in (True, False):
         for yy in range(1, 3):
             for zz in range(-yy, 0, yy):
                 reps = [(b > c, cond), (y, yy), (z, zz)]
                 # 验证代入后的结果是否符合预期
                 assert ans.subs(reps) == I.subs(reps).doit()


def test_unevaluated_integrals():
    # 创建函数 'f'
    f = Function('f')
    # 创建 Piecewise 对象 p
    p = Piecewise((1, Eq(f(x) - 1, 0)), (2, x - 10 < 0), (0, True))
    # 验证积分的结果是否是积分对象本身
    assert p.integrate(x) == Integral(p, x)
    # 验证积分的结果是否是积分对象本身
    assert p.integrate((x, 0, 5)) == Integral(p, (x, 0, 5))
    # 替换 f(x) 为 x%2 进行测试，预期结果是数值 10.0
    assert Integral(p, (x, 0, 5)).subs(f(x), x%2).n() == 10.0

    # 创建 Piecewise 对象 p
    # 测试使用 _solve_inequality 替代 solve_univariate_inequality 失败时的情况
    assert p.integrate(y) == Piecewise(
        (y, Eq(f(x), 1) | ((x < 10) & Eq(f(x), 1))),
        (2*y, (x > -oo) & (x < 10)), (0, True))


def test_conditions_as_alternate_booleans():
    # 创建符号 'a, b, c'
    a, b, c = symbols('a:c')
    # 验证 Piecewise 对象的重写结果是否符合预期
    assert Piecewise((x, Piecewise((y < 1, x > 0), (y > 1, True)))
        ) == Piecewise((x, ITE(x > 0, y < 1, y > 1)))


def test_Piecewise_rewrite_as_ITE():
    # 创建符号 'a, b, c, d'
    a, b, c, d = symbols('a:d')
    # 定义一个函数 `_ITE`，接受任意数量的参数作为条件-结果对，返回用 `ITE` 重写后的表达式
    def _ITE(*args):
        # 使用 `Piecewise` 类创建一个分段函数，然后用 `ITE` 进行重写
        return Piecewise(*args).rewrite(ITE)

    # 断言：测试 `_ITE` 函数是否正确处理不同的条件-结果对
    assert _ITE((a, x < 1), (b, x >= 1)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, x < oo)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, Or(y < 1, x < oo)), (c, y > 0)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, True)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, x < 2), (c, True)) == ITE(x < 1, a, ITE(x < 2, b, c))
    assert _ITE((a, x < 1), (b, y < 2), (c, True)) == ITE(x < 1, a, ITE(y < 2, b, c))
    assert _ITE((a, x < 1), (b, x < oo), (c, y < 1)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (c, y < 1), (b, x < oo), (d, True)) == ITE(x < 1, a, ITE(y < 1, c, b))
    assert _ITE((a, x < 0), (b, Or(x < oo, y < 1))) == ITE(x < 0, a, b)
    raises(TypeError, lambda: _ITE((x + 1, x < 1), (x, True)))
    # 测试：如果 `a` 被替换成 `y`，则代码覆盖率完整，但需要使用其他方法而不是 `as_set` 来检测这一点
    raises(NotImplementedError, lambda: _ITE((x, x < y), (y, x >= a)))
    raises(ValueError, lambda: _ITE((a, x < 2), (b, x > 3)))
# 测试函数，用于检查问题编号 14052 的集成情况
def test_issue_14052():
    # 断言集成 abs(sin(x)) 在区间 [0, 2*pi] 上的结果等于 4
    assert integrate(abs(sin(x)), (x, 0, 2*pi)) == 4


# 测试函数，用于检查问题编号 14240 的 piecewise_fold 函数
def test_issue_14240():
    # 测试两个 Piecewise 对象相加后折叠的结果
    assert piecewise_fold(
        Piecewise((1, a), (2, b), (4, True)) +
        Piecewise((8, a), (16, True))
        ) == Piecewise((9, a), (18, b), (20, True))
    
    # 测试两个 Piecewise 对象相乘后折叠的结果
    assert piecewise_fold(
        Piecewise((2, a), (3, b), (5, True)) *
        Piecewise((7, a), (11, True))
        ) == Piecewise((14, a), (33, b), (55, True))
    
    # 如果使用朴素折叠，以下断言将会挂起
    # 测试多个 Piecewise 对象相加后折叠的结果
    assert piecewise_fold(Add(*[
        Piecewise((i, a), (0, True)) for i in range(40)])
        ) == Piecewise((780, a), (0, True))
    
    # 测试多个 Piecewise 对象相乘后折叠的结果
    assert piecewise_fold(Mul(*[
        Piecewise((i, a), (0, True)) for i in range(1, 41)])
        ) == Piecewise((factorial(40), a), (0, True))


# 测试函数，用于检查问题编号 14787
def test_issue_14787():
    x = Symbol('x')
    # 创建一个 Piecewise 对象，并测试其 evalf 方法的字符串表示
    f = Piecewise((x, x < 1), ((S(58) / 7), True))
    assert str(f.evalf()) == "Piecewise((x, x < 1), (8.28571428571429, True))"


# 测试函数，用于检查问题编号 21481
def test_issue_21481():
    b, e = symbols('b e')
    # 创建复杂的 Piecewise 对象 C
    C = Piecewise(
        (2,
        ((b > 1) & (e > 0)) |
        ((b > 0) & (b < 1) & (e < 0)) |
        ((e >= 2) & (b < -1) & Eq(Mod(e, 2), 0)) |
        ((e <= -2) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 0))),
        (S.Half,
        ((b > 1) & (e < 0)) |
        ((b > 0) & (e > 0) & (b < 1)) |
        ((e <= -2) & (b < -1) & Eq(Mod(e, 2), 0)) |
        ((e >= 2) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 0))),
        (-S.Half,
        Eq(Mod(e, 2), 1) &
        (((e <= -1) & (b < -1)) | ((e >= 1) & (b > -1) & (b < 0)))),
        (-2,
        ((e >= 1) & (b < -1) & Eq(Mod(e, 2), 1)) |
        ((e <= -1) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 1)))
    )
    
    # 创建包含 Piecewise 对象 C 的新 Piecewise 对象 A
    A = Piecewise(
        (1, Eq(b, 1) | Eq(e, 0) | (Eq(b, -1) & Eq(Mod(e, 2), 0))),
        (0, Eq(b, 0) & (e > 0)),
        (-1, Eq(b, -1) & Eq(Mod(e, 2), 1)),
        (C, Eq(im(b), 0) & Eq(im(e), 0))
    )

    # 对 A 进行折叠操作，得到 B
    B = piecewise_fold(A)
    
    # 简化 A 和 B，并验证在不同的值下它们的一致性
    sa = A.simplify()
    sb = B.simplify()
    v = (-2, -1, -S.Half, 0, S.Half, 1, 2)
    for i in v:
        for j in v:
            r = {b:i, e:j}
            ok = [k.xreplace(r) for k in (A, B, sa, sb)]
            assert len(set(ok)) == 1


# 测试函数，用于检查问题编号 8458
def test_issue_8458():
    x, y = symbols('x y')
    # 创建 Piecewise 对象 p1，测试其简化结果
    p1 = Piecewise((0, Eq(x, 0)), (sin(x), True))
    assert p1.simplify() == sin(x)
    
    # 创建 Piecewise 对象 p2，测试其简化结果
    p2 = Piecewise((x, Eq(x, 0)), (4*x + (y-2)**4, Eq(x, 0) & Eq(x+y, 2)), (sin(x), True))
    assert p2.simplify() == sin(x)
    
    # 创建 Piecewise 对象 p3，测试其简化结果
    p3 = Piecewise((x+1, Eq(x, -1)), (4*x + (y-2)**4, Eq(x, 0) & Eq(x+y, 2)), (sin(x), True))
    assert p3.simplify() == Piecewise((0, Eq(x, -1)), (sin(x), True))


# 测试函数，用于检查问题编号 16417
def test_issue_16417():
    z = Symbol('z')
    # 验证 unchanged 函数对 Piecewise 的作用
    assert unchanged(Piecewise, (1, Or(Eq(im(z), 0), Gt(re(z), 0))), (2, True))

    x = Symbol('x')
    # 验证 unchanged 函数对 Piecewise 的作用
    assert unchanged(Piecewise, (S.Pi, re(x) < 0),
                 (0, Or(re(x) > 0, Ne(im(x), 0))),
                 (S.NaN, True))
    r = Symbol('r', real=True)
    # 创建一个 Piecewise 对象 p，根据条件分段定义函数：
    # 当 re(r) < 0 时，返回 S.Pi
    # 当 re(r) > 0 或者 im(r) 不等于 0 时，返回 0
    # 其他情况返回 S.NaN
    p = Piecewise((S.Pi, re(r) < 0),
                 (0, Or(re(r) > 0, Ne(im(r), 0))),
                 (S.NaN, True))
    
    # 使用断言检查 p 是否与给定的 Piecewise 对象相等，不进行评估
    assert p == Piecewise((S.Pi, r < 0),
                 (0, r > 0),
                 (S.NaN, True), evaluate=False)
    
    # 注释部分代码，因为虚部不等于 0 所以无法工作
    #i = Symbol('i', imaginary=True)
    #p = Piecewise((S.Pi, re(i) < 0),
    #              (0, Or(re(i) > 0, Ne(im(i), 0))),
    #              (S.NaN, True))
    #assert p == Piecewise((0, Ne(im(i), 0)),
    #                      (S.NaN, True), evaluate=False)
    
    # 计算复数 i = I * r
    i = I*r
    
    # 创建一个新的 Piecewise 对象 p，根据条件分段定义函数：
    # 当 re(i) < 0 时，返回 S.Pi
    # 当 re(i) > 0 或者 im(i) 不等于 0 时，返回 0
    # 其他情况返回 S.NaN
    p = Piecewise((S.Pi, re(i) < 0),
                  (0, Or(re(i) > 0, Ne(im(i), 0))),
                  (S.NaN, True))
    
    # 使用断言检查 p 是否与给定的 Piecewise 对象相等，不进行评估
    assert p == Piecewise((0, Ne(im(i), 0)),
                          (S.NaN, True), evaluate=False)
    
    # 使用断言检查 p 是否与给定的 Piecewise 对象相等，不进行评估
    assert p == Piecewise((0, Ne(r, 0)),
                          (S.NaN, True), evaluate=False)
# 定义一个测试函数，用于验证将表达式重写为克罗内克 δ 函数的正确性
def test_eval_rewrite_as_KroneckerDelta():
    # 定义符号变量 x, y, z, n, t, m
    x, y, z, n, t, m = symbols('x y z n t m')
    # K 是克罗内克 δ 函数的引用
    K = KroneckerDelta
    # 定义 lambda 函数 f，接受参数 p，将 p 重写为克罗内克 δ 函数后展开
    f = lambda p: expand(p.rewrite(K))

    # 定义 Piecewise 对象 p1，并验证 f(p1) 的值是否等于 1 - K(x, y)
    p1 = Piecewise((0, Eq(x, y)), (1, True))
    assert f(p1) == 1 - K(x, y)

    # 定义 Piecewise 对象 p2，并验证 f(p2) 的值
    p2 = Piecewise((x, Eq(y,0)), (z, Eq(t,0)), (n, True))
    assert f(p2) == n*K(0, t)*K(0, y) - n*K(0, t) - n*K(0, y) + n + \
           x*K(0, y) - z*K(0, t)*K(0, y) + z*K(0, t)

    # 定义 Piecewise 对象 p3，并验证 f(p3) 的值是否等于 1 - K(x, y)
    p3 = Piecewise((1, Ne(x, y)), (0, True))
    assert f(p3) == 1 - K(x, y)

    # 以下类似地定义 Piecewise 对象，并验证其克罗内克 δ 函数的重写是否正确
    p4 = Piecewise((1, Eq(x, 3)), (4, True))
    assert f(p4) == 4 - 3*K(3, x)

    p5 = Piecewise((3, Ne(x, 2)), (4, Eq(y, 2)), (5, True))
    assert f(p5) == -K(2, x)*K(2, y) + 2*K(2, x) + 3

    p6 = Piecewise((0, Ne(x, 1) & Ne(y, 4)), (1, True))
    assert f(p6) == -K(1, x)*K(4, y) + K(1, x) + K(4, y)

    p7 = Piecewise((2, Eq(y, 3) & Ne(x, 2)), (1, True))
    assert f(p7) == -K(2, x)*K(3, y) + K(3, y) + 1

    p8 = Piecewise((4, Eq(x, 3) & Ne(y, 2)), (1, True))
    assert f(p8) == -3*K(2, y)*K(3, x) + 3*K(3, x) + 1

    p9 = Piecewise((6, Eq(x, 4) & Eq(y, 1)), (1, True))
    assert f(p9) == 5 * K(1, y) * K(4, x) + 1

    p10 = Piecewise((4, Ne(x, -4) | Ne(y, 1)), (1, True))
    assert f(p10) == -3 * K(-4, x) * K(1, y) + 4

    p11 = Piecewise((1, Eq(y, 2) | Ne(x, -3)), (2, True))
    assert f(p11) == -K(-3, x)*K(2, y) + K(-3, x) + 1

    p12 = Piecewise((-1, Eq(x, 1) | Ne(y, 3)), (1, True))
    assert f(p12) == -2*K(1, x)*K(3, y) + 2*K(3, y) - 1

    p13 = Piecewise((3, Eq(x, 2) | Eq(y, 4)), (1, True))
    assert f(p13) == -2*K(2, x)*K(4, y) + 2*K(2, x) + 2*K(4, y) + 1

    p14 = Piecewise((1, Ne(x, 0) | Ne(y, 1)), (3, True))
    assert f(p14) == 2 * K(0, x) * K(1, y) + 1

    p15 = Piecewise((2, Eq(x, 3) | Ne(y, 2)), (3, Eq(x, 4) & Eq(y, 5)), (1, True))
    assert f(p15) == -2*K(2, y)*K(3, x)*K(4, x)*K(5, y) + K(2, y)*K(3, x) + \
           2*K(2, y)*K(4, x)*K(5, y) - K(2, y) + 2

    p16 = Piecewise((0, Ne(m, n)), (1, True))*Piecewise((0, Ne(n, t)), (1, True))\
          *Piecewise((0, Ne(n, x)), (1, True)) - Piecewise((0, Ne(t, x)), (1, True))
    assert f(p16) == K(m, n)*K(n, t)*K(n, x) - K(t, x)

    p17 = Piecewise((0, Ne(t, x) & (Ne(m, n) | Ne(n, t) | Ne(n, x))),
                    (1, Ne(t, x)), (-1, Ne(m, n) | Ne(n, t) | Ne(n, x)), (0, True))
    assert f(p17) == K(m, n)*K(n, t)*K(n, x) - K(t, x)

    p18 = Piecewise((-4, Eq(y, 1) | (Eq(x, -5) & Eq(x, z))), (4, True))
    assert f(p18) == 8*K(-5, x)*K(1, y)*K(x, z) - 8*K(-5, x)*K(x, z) - 8*K(1, y) + 4

    p19 = Piecewise((0, x > 2), (1, True))
    assert f(p19) == p19

    p20 = Piecewise((0, And(x < 2, x > -5)), (1, True))
    assert f(p20) == p20

    p21 = Piecewise((0, Or(x > 1, x < 0)), (1, True))
    assert f(p21) == p21

    p22 = Piecewise((0, ~((Eq(y, -1) | Ne(x, 0)) & (Ne(x, 1) | Ne(y, -1)))), (1, True))
    assert f(p22) == K(-1, y)*K(0, x) - K(-1, y)*K(1, x) - K(0, x) + 1
    # 创建一个名为 u1 的 Uniform 对象，范围在 [0, 1] 之间，用于生成均匀分布的随机数
    u1 = Uniform('u1', 0, 1)
    # 创建一个名为 u2 的 Uniform 对象，范围在 [0, 1] 之间，用于生成均匀分布的随机数
    u2 = Uniform('u2', 0, 1)
    # 计算 u1 + u2 的密度值。由于结果可能很大，这里不是特别重要（并且理想情况下应更简单）。
    # 但是不应该导致异常。
    density(u1 + u2)
def test_issue_7370():
    # 定义一个分段函数，当 x <= 2400 时返回 1
    f = Piecewise((1, x <= 2400))
    # 对 f 在区间 [0, 252.4] 上进行积分，返回积分结果
    v = integrate(f, (x, 0, Float("252.4", 30)))
    # 断言积分结果的字符串形式为 '252.400000000000000000000000000'
    assert str(v) == '252.400000000000000000000000000'


def test_issue_14933():
    # 定义符号变量 x 和 y
    x = Symbol('x')
    y = Symbol('y')

    # 定义一个 1x1 的矩阵符号 inp
    inp = MatrixSymbol('inp', 1, 1)
    # 构建替换字典，将 y 映射到 inp 矩阵的第 0 行第 0 列，将 x 映射到同一位置
    rep_dict = {y: inp[0, 0], x: inp[0, 0]}

    # 定义一个分段函数 p，当 y > 0 且 x < 0 时返回 1
    p = Piecewise((1, ITE(y > 0, x < 0, True)))
    # 断言 p 在替换字典 rep_dict 下的结果等于另一个分段函数的结果
    assert p.xreplace(rep_dict) == Piecewise((1, ITE(inp[0, 0] > 0, inp[0, 0] < 0, True)))


def test_issue_16715():
    # 断言调用 Piecewise 对象的 as_expr_set_pairs 方法会抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: Piecewise((x, x<0), (0, y>1)).as_expr_set_pairs())


def test_issue_20360():
    # 定义符号变量 t 和 tau
    t, tau = symbols("t tau", real=True)
    # 定义符号变量 n 为整数
    n = symbols("n", integer=True)
    # 定义 lambda 为 pi * (n - S.Half)
    lam = pi * (n - S.Half)
    # 计算指数函数的积分，返回积分结果
    eq = integrate(exp(lam * tau), (tau, 0, t))
    # 断言化简后的积分结果等于给定的表达式
    assert eq.simplify() == (2*exp(pi*t*(2*n - 1)/2) - 2)/(pi*(2*n - 1))


def test_piecewise_eval():
    # XXX 如果这个简化被移出 eval 并进入 boolalg 或 Piecewise 简化函数，则这些测试可能需要修改
    f = lambda x: x.args[0].cond
    # 未简化的情况下断言分段函数的条件部分
    assert f(Piecewise((x, (x > -oo) & (x < 3)))) == ((x > -oo) & (x < 3))
    assert f(Piecewise((x, (x > -oo) & (x < oo)))) == ((x > -oo) & (x < oo))
    assert f(Piecewise((x, (x > -3) & (x < 3)))) == ((x > -3) & (x < 3))
    assert f(Piecewise((x, (x > -3) & (x < oo)))) == ((x > -3) & (x < oo))
    assert f(Piecewise((x, (x <= 3) & (x > -oo)))) == ((x <= 3) & (x > -oo))
    assert f(Piecewise((x, (x <= 3) & (x > -3)))) == ((x <= 3) & (x > -3))
    assert f(Piecewise((x, (x >= -3) & (x < 3)))) == ((x >= -3) & (x < 3))
    assert f(Piecewise((x, (x >= -3) & (x < oo)))) == ((x >= -3) & (x < oo))
    assert f(Piecewise((x, (x >= -3) & (x <= 3)))) == ((x >= -3) & (x <= 3))
    assert f(Piecewise((x, (x <= oo) & (x > -oo)))) == (x > -oo) & (x <= oo)
    assert f(Piecewise((x, (x <= oo) & (x > -3)))) == (x > -3) & (x <= oo)
    assert f(Piecewise((x, (x >= -oo) & (x < 3)))) == (x < 3) & (x >= -oo)
    assert f(Piecewise((x, (x >= -oo) & (x < oo)))) == (x < oo) & (x >= -oo)
    assert f(Piecewise((x, (x >= -oo) & (x <= 3)))) == (x <= 3) & (x >= -oo)
    assert f(Piecewise((x, (x >= -oo) & (x <= oo)))) == (x <= oo) & (x >= -oo)
    assert f(Piecewise((x, (x >= -3) & (x <= oo)))) == (x >= -3) & (x <= oo)
    assert f(Piecewise((x, (Abs(arg(a)) <= 1) | (Abs(arg(a)) < 1)))) == (Abs(arg(a)) <= 1) | (Abs(arg(a)) < 1)


def test_issue_22533():
    # 定义实数符号变量 x
    x = Symbol('x', real=True)
    # 定义分段函数 f
    f = Piecewise((-1 / x, x <= 0), (1 / x, True))
    # 断言 f 对 x 的积分结果等于另一个分段函数
    assert integrate(f, x) == Piecewise((-log(x), x <= 0), (log(x), True))


def test_issue_24072():
    # 断言一个分段函数等于另一个分段函数
    assert Piecewise((1, x > 1), (2, x <= 1), (3, x <= 1)) == Piecewise((1, x > 1), (2, True))


def test_piecewise__eval_is_meromorphic():
    # 这个测试可能需要修改，如果这种简化被移出 eval 并进入 boolalg 或 Piecewise 简化函数
    pass
    """ Issue 24127: Tests eval_is_meromorphic auxiliary method """
    # 定义符号变量 x，并指定其为实数
    x = symbols('x', real=True)
    # 定义 Piecewise 函数 f，根据条件分段定义：当 x < 0 时为 1，否则为 sqrt(1 - x)
    f = Piecewise((1, x < 0), (sqrt(1 - x), True))
    # 断言：在点 x + I 处，f 是否亚解析（meromorphic）
    assert f.is_meromorphic(x, I) is None
    # 断言：在点 x - 1 处，f 是否亚解析（meromorphic）
    assert f.is_meromorphic(x, -1) == True
    # 断言：在点 x + 0 处，f 是否亚解析（meromorphic）
    assert f.is_meromorphic(x, 0) == None
    # 断言：在点 x + 1 处，f 是否亚解析（meromorphic）
    assert f.is_meromorphic(x, 1) == False
    # 断言：在点 x + 2 处，f 是否亚解析（meromorphic）
    assert f.is_meromorphic(x, 2) == True
    # 断言：在点 x + a 处，f 是否亚解析（meromorphic），其中 a 是一个符号变量
    assert f.is_meromorphic(x, Symbol('a')) == None
    # 断言：在点 x + a 处，f 是否亚解析（meromorphic），其中 a 是一个实数符号变量
    assert f.is_meromorphic(x, Symbol('a', real=True)) == None
```