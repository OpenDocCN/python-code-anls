# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_hyperexpand.py`

```
# 导入随机整数生成函数randrange
from sympy.core.random import randrange

# 导入sympy库中的各种特殊函数和类
from sympy.simplify.hyperexpand import (ShiftA, ShiftB, UnShiftA, UnShiftB,
                       MeijerShiftA, MeijerShiftB, MeijerShiftC, MeijerShiftD,
                       MeijerUnShiftA, MeijerUnShiftB, MeijerUnShiftC,
                       MeijerUnShiftD,
                       ReduceOrder, reduce_order, apply_operators,
                       devise_plan, make_derivative_operator, Formula,
                       hyperexpand, Hyper_Function, G_Function,
                       reduce_order_meijer,
                       build_hypergeometric_formula)

# 导入sympy库中的具体类和函数
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.abc import z, a, b, c
from sympy.testing.pytest import XFAIL, raises, slow, tooslow
from sympy.core.random import verify_numerically as tn

# 导入sympy库中的数学常数和特殊函数
from sympy.core.numbers import (Rational, pi)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.functions.special.bessel import besseli
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)

# 定义一个测试函数，用于测试特殊函数的展开
def test_branch_bug():
    # 断言：对给定的超几何函数进行展开，验证结果是否符合预期
    assert hyperexpand(hyper((Rational(-1, 3), S.Half), (Rational(2, 3), Rational(3, 2)), -z)) == \
        -z**S('1/3')*lowergamma(exp_polar(I*pi)/3, z)/5 \
        + sqrt(pi)*erf(sqrt(z))/(5*sqrt(z))
    # 断言：对给定的Meijer G 函数进行展开，验证结果是否符合预期
    assert hyperexpand(meijerg([Rational(7, 6), 1], [], [Rational(2, 3)], [Rational(1, 6), 0], z)) == \
        2*z**S('2/3')*(2*sqrt(pi)*erf(sqrt(z))/sqrt(z) - 2*lowergamma(
                       Rational(2, 3), z)/z**S('2/3'))*gamma(Rational(2, 3))/gamma(Rational(5, 3))

# 定义一个测试函数，用于测试超几何函数的展开
def test_hyperexpand():
    # 引用参考文献：Luke, Y. L. (1969), The Special Functions and Their Approximations,
    # Volume 1, section 6.2
    # 断言：空超几何函数的展开结果应为exp(z)
    assert hyperexpand(hyper([], [], z)) == exp(z)
    # 断言：带有参数的超几何函数的展开结果应为log(1 + z)
    assert hyperexpand(hyper([1, 1], [2], -z)*z) == log(1 + z)
    # 断言：带有半整数参数的超几何函数的展开结果应为cos(z)
    assert hyperexpand(hyper([], [S.Half], -z**2/4)) == cos(z)
    # 断言：带有半整数参数的超几何函数的展开结果应为sin(z)
    assert hyperexpand(z*hyper([], [S('3/2')], -z**2/4)) == sin(z)
    # 断言：带有半整数参数的超几何函数的展开结果应为asin(z)
    assert hyperexpand(hyper([S('1/2'), S('1/2')], [S('3/2')], z**2)*z) \
        == asin(z)
    # 断言：对二项式乘以z^2的求和结果进行展开，验证结果是否为表达式
    assert isinstance(Sum(binomial(2, z)*z**2, (z, 0, a)).doit(), Expr)

# 定义一个函数，用于判断是否可以对给定的超几何函数进行展开
def can_do(ap, bq, numerical=True, div=1, lowerplane=False):
    # 对给定的超几何函数进行展开
    r = hyperexpand(hyper(ap, bq, z))
    # 如果结果中仍包含超几何函数，则无法完全展开，返回False
    if r.has(hyper):
        return False
    # 如果不需要数值化的话，直接返回True
    if not numerical:
        return True
    # 准备用于替换的空字典
    repl = {}
    # 获取结果中除了z之外的随机符号集合
    randsyms = r.free_symbols - {z}
    while randsyms:
        # 只检查随机生成的参数。
        for n, ai in enumerate(randsyms):
            # 将随机符号对应的复数随机数除以 div 后存入 repl 字典
            repl[ai] = randcplx(n) / div
        # 如果在替换后的表达式中没有任何负整数的情况，退出循环
        if not any(b.is_Integer and b <= 0 for b in Tuple(*bq).subs(repl)):
            break
    [a, b, c, d] = [2, -1, 3, 1]
    # 如果 lowerplane 为真，重新赋值 a, b, c, d
    if lowerplane:
        [a, b, c, d] = [2, -2, 3, -1]
    # 返回使用替换后的参数计算得到的结果
    return tn(
        hyper(ap, bq, z).subs(repl),  # 计算超几何函数并替换其中的参数
        r.replace(exp_polar, exp).subs(repl),  # 替换 r 中的极坐标表达式为指数表达式并计算
        z, a=a, b=b, c=c, d=d)
# 定义一个名为 test_roach 的测试函数，用于测试 can_do 函数的多个断言
def test_roach():
    # 断言可以处理 [S.Half] 和 [Rational(9, 2)] 这两个参数
    assert can_do([S.Half], [Rational(9, 2)])
    # 断言可以处理空的第一个参数列表和包含数值和有理数的第二个参数列表
    assert can_do([], [1, Rational(5, 2), 4])
    # 断言可以处理包含有理数和整数的两个参数列表
    assert can_do([Rational(-1, 2), 1, 2], [3, 4])
    # 断言可以处理第一个参数列表包含一个有理数，第二个参数列表包含多个有理数和 S.Half
    assert can_do([Rational(1, 3)], [Rational(-2, 3), Rational(-1, 2), S.Half, 1])
    # 断言可以处理两个有理数的第一个参数列表和第二个参数列表包含一个整数和一个有理数
    assert can_do([Rational(-3, 2), Rational(-1, 2)], [Rational(-5, 2), 1])
    # 断言可以处理第一个参数列表包含一个有理数，第二个参数列表包含一个有理数和 S.Half
    assert can_do([Rational(-3, 2), ], [Rational(-1, 2), S.Half])  # shine-integral
    # 断言可以处理两个有理数的第一个参数列表和第二个参数列表包含一个整数
    assert can_do([Rational(-3, 2), Rational(-1, 2)], [2])  # elliptic integrals


# 标记为预期失败的测试函数
@XFAIL
def test_roach_fail():
    # 断言 can_do 函数无法处理包含有理数和 S.Half 的两个参数列表
    assert can_do([Rational(-1, 2), 1], [Rational(1, 4), S.Half, Rational(3, 4)])  # PFDD
    # 断言 can_do 函数无法处理包含一个有理数的第一个参数列表和包含两个整数的第二个参数列表
    assert can_do([Rational(3, 2)], [Rational(5, 2), 5])  # struve function
    # 断言 can_do 函数无法处理包含一个有理数和一个整数的第一个参数列表，和包含两个有理数的第二个参数列表
    assert can_do([Rational(-1, 2), S.Half, 1], [Rational(3, 2), Rational(5, 2)])  # polylog, pfdd
    # 断言 can_do 函数无法处理包含三个整数的第一个参数列表和包含 S.Half 和一个整数的第二个参数列表
    assert can_do([1, 2, 3], [S.Half, 4])  # XXX ?
    # 断言 can_do 函数无法处理 [S.Half] 和包含三个有理数的第二个参数列表
    assert can_do([S.Half], [Rational(-1, 3), Rational(-1, 2), Rational(-2, 3)])  # PFDD ?


# 测试多项式函数
def test_polynomial():
    # 导入 oo 和 hyper 函数，断言对特定的超几何函数调用返回无穷大
    from sympy.core.numbers import oo
    assert hyperexpand(hyper([], [-1], z)) is oo
    assert hyperexpand(hyper([-2], [-1], z)) is oo
    # 断言对特定的超几何函数调用返回 1
    assert hyperexpand(hyper([0, 0], [-1], z)) == 1
    # 断言 can_do 函数可以处理包含整数、负整数和随机复数的参数列表
    assert can_do([-5, -2, randcplx(), randcplx()], [-10, randcplx()])
    # 断言对特定的超几何函数调用返回特定值
    assert hyperexpand(hyper((-1, 1), (-2,), z)) == 1 + z/2


# 测试超级展开基函数
def test_hyperexpand_bases():
    # 断言对特定的超几何函数调用返回特定的表达式
    assert hyperexpand(hyper([2], [a], z)) == \
        a + z**(-a + 1)*(-a**2 + 3*a + z*(a - 1) - 2)*exp(z)* \
        lowergamma(a - 1, z) - 1
    # TODO [a+1, aRational(-1, 2)], [2*a]
    # 断言对特定的超几何函数调用返回特定的表达式
    assert hyperexpand(hyper([1, 2], [3], z)) == -2/z - 2*log(-z + 1)/z**2
    # 断言对特定的超几何函数调用返回特定的表达式
    assert hyperexpand(hyper([S.Half, 2], [Rational(3, 2)], z)) == \
        -1/(2*z - 2) + atanh(sqrt(z))/sqrt(z)/2
    # 断言对特定的超几何函数调用返回特定的表达式
    assert hyperexpand(hyper([S.Half, S.Half], [Rational(5, 2)], z)) == \
        (-3*z + 3)/4/(z*sqrt(-z + 1)) \
        + (6*z - 3)*asin(sqrt(z))/(4*z**Rational(3, 2))
    # 断言对特定的超几何函数调用返回特定的表达式
    assert hyperexpand(hyper([1, 2], [Rational(3, 2)], z)) == -1/(2*z - 2) \
        - asin(sqrt(z))/(sqrt(z)*(2*z - 2)*sqrt(-z + 1))
    # 断言对特定的超几何函数调用返回特定的表达式
    assert hyperexpand(hyper([Rational(-1, 2) - 1, 1, 2], [S.Half, 3], z)) == \
        sqrt(z)*(z*Rational(6, 7) - Rational(6, 5))*atanh(sqrt(z)) \
        + (-30*z**2 + 32*z - 6)/35/z - 6*log(-z + 1)/(35*z**2)
    # 断言对特定的超几何函数调用返回特定的表达式
    assert hyperexpand(hyper([1 + S.Half, 1, 1], [2, 2], z)) == \
        -4*log(sqrt(-z + 1)/2 + S.Half)/z
    # TODO hyperexpand(hyper([a], [2*a + 1], z))
    # TODO [S.Half, a], [Rational(3, 2), a+1]
    # 断言对特定的超几何函数调用返回特定的表达式
    assert hyperexpand(hyper([2], [b, 1], z)) == \
        z**(-b/2 + S.Half)*besseli(b - 1, 2*sqrt(z))*gamma(b) \
        + z**(-b/2 + 1)*besseli(b, 2*sqrt(z))*gamma(b)
    # TODO [a], [a - S.Half, 2*a]


# 测试超级展开的参数化函数
def test_hyperexpand_parametric():
    # 断言对特定的超几何函数调用返回特定的表达式
    assert hyperexpand(hyper([a, S.Half + a], [S.Half], z)) \
        == (1 + sqrt(z))**(-2*a)/2 + (1 - sqrt(z))**(-2*a)/2
    # 断言语句：验证超级函数的展开结果是否等于给定表达式
    assert hyperexpand(hyper([a, Rational(-1, 2) + a], [2*a], z)) \
        == 2**(2*a - 1)*((-z + 1)**S.Half + 1)**(-2*a + 1)
# 定义一个测试函数，用于验证 shifted_sum 函数的正确性
def test_shifted_sum():
    # 从 sympy 库中导入 simplify 函数
    from sympy.simplify.simplify import simplify
    # 断言：对给定的超几何级数进行展开和简化后，结果应该等于指定的表达式
    assert simplify(hyperexpand(z**4*hyper([2], [3, S('3/2')], -z**2))) \
        == z*sin(2*z) + (-z**2 + S.Half)*cos(2*z) - S.Half


# 定义一个内部函数 _randrat，返回一个有理数，避免使用整数
def _randrat():
    """ Steer clear of integers. """
    return S(randrange(25) + 10)/50


# 定义一个函数 randcplx，返回一个复数，用于多项式系数不是实数时
def randcplx(offset=-1):
    """ Polys is not good with real coefficients. """
    return _randrat() + I*_randrat() + I*(1 + offset)


# 用装饰器 @slow 标记的测试函数，用于测试公式
@slow
def test_formulae():
    # 从 sympy 库中导入 hyperexpand 函数和 FormulaCollection 类
    from sympy.simplify.hyperexpand import FormulaCollection
    # 获取 FormulaCollection 类的公式集合
    formulae = FormulaCollection().formulae
    # 遍历每个公式
    for formula in formulae:
        # 根据公式生成一个函数 h
        h = formula.func(formula.z)
        rep = {}
        # 枚举公式中的符号，替换为随机复数
        for n, sym in enumerate(formula.symbols):
            rep[sym] = randcplx(n)

        # 注意：hyperexpand 返回真正分支的函数。我们知道我们在主分支上，但数值评估仍可能出错
        # (例如，如果 exp_polar 无法 evalf)。
        # 可以尝试将所有 exp_polar 替换为 exp，通常这样可以解决问题。

        # 首先测试闭合形式是否正确
        h = h.subs(rep)
        closed_form = formula.closed_form.subs(rep).rewrite('nonrepsmall')
        z = formula.z
        assert tn(h, closed_form.replace(exp_polar, exp), z)

        # 现在测试计算得到的矩阵
        cl = (formula.C * formula.B)[0].subs(rep).rewrite('nonrepsmall')
        assert tn(closed_form.replace(
            exp_polar, exp), cl.replace(exp_polar, exp), z)
        deriv1 = z*formula.B.applyfunc(lambda t: t.rewrite(
            'nonrepsmall')).diff(z)
        deriv2 = formula.M * formula.B
        for d1, d2 in zip(deriv1, deriv2):
            assert tn(d1.subs(rep).replace(exp_polar, exp),
                      d2.subs(rep).rewrite('nonrepsmall').replace(exp_polar, exp), z)


# 定义一个测试函数，用于验证 Meijer 函数的公式
def test_meijerg_formulae():
    # 从 sympy 库中导入 MeijerFormulaCollection 类
    from sympy.simplify.hyperexpand import MeijerFormulaCollection
    # 获取 MeijerFormulaCollection 类的公式集合
    formulae = MeijerFormulaCollection().formulae
    # 遍历每个签名下的公式
    for sig in formulae:
        for formula in formulae[sig]:
            # 根据公式的参数生成一个 Meijer 函数 g
            g = meijerg(formula.func.an, formula.func.ap,
                        formula.func.bm, formula.func.bq,
                        formula.z)
            rep = {}
            # 遍历公式中的符号，替换为随机复数
            for sym in formula.symbols:
                rep[sym] = randcplx()

            # 首先测试闭合形式是否正确
            g = g.subs(rep)
            closed_form = formula.closed_form.subs(rep)
            z = formula.z
            assert tn(g, closed_form, z)

            # 现在测试计算得到的矩阵
            cl = (formula.C * formula.B)[0].subs(rep)
            assert tn(closed_form, cl, z)
            deriv1 = z*formula.B.diff(z)
            deriv2 = formula.M * formula.B
            for d1, d2 in zip(deriv1, deriv2):
                assert tn(d1.subs(rep), d2.subs(rep), z)


# 定义一个操作函数 op，返回 z 与给定函数的 z 偏导数的乘积
def op(f):
    return z*f.diff(z)


# 验证一个计划是否生成了正确的结果
def test_plan():
    # 断言：根据给定的超函数生成的计划与空列表相等
    assert devise_plan(Hyper_Function([0], ()),
            Hyper_Function([0], ()), z) == []
    # 断言异常是否被引发，确保在给定的条件下会抛出 ValueError 异常
    with raises(ValueError):
        # 调用 devise_plan 函数，传入 Hyper_Function 对象和 z 参数，期望抛出异常
        devise_plan(Hyper_Function([1], ()), Hyper_Function((), ()), z)

    # 断言异常是否被引发，确保在给定的条件下会抛出 ValueError 异常
    with raises(ValueError):
        # 调用 devise_plan 函数，传入 Hyper_Function 对象和 z 参数，期望抛出异常
        devise_plan(Hyper_Function([2], [1]), Hyper_Function([2], [2]), z)

    # 断言异常是否被引发，确保在给定的条件下会抛出 ValueError 异常
    with raises(ValueError):
        # 调用 devise_plan 函数，传入 Hyper_Function 对象和 z 参数，期望抛出异常
        devise_plan(Hyper_Function([2], []), Hyper_Function([S("1/2")], []), z)

    # 提示不能使用 pi/(10000 + n)，因为 polys 函数非常慢。
    # 生成三个复数 a1, a2, b1，分别用于后续计算
    a1, a2, b1 = (randcplx(n) for n in range(3))
    # 将 b1 加上 2*i，其中 i 是虚数单位
    b1 += 2*I
    # 计算超几何函数 h，传入参数 [a1, a2], [b1], z
    h = hyper([a1, a2], [b1], z)

    # 计算超几何函数 h2，传入参数 (a1 + 1, a2), [b1], z
    h2 = hyper((a1 + 1, a2), [b1], z)
    # 断言应用操作符后得到的结果与预期的 h2 相等
    assert tn(apply_operators(h,
        # 调用 devise_plan 函数，计算两个超函数的差异操作
        devise_plan(Hyper_Function((a1 + 1, a2), [b1]),
            Hyper_Function((a1, a2), [b1]), z), op),
        h2, z)

    # 计算超几何函数 h2，传入参数 (a1 + 1, a2 - 1), [b1], z
    h2 = hyper((a1 + 1, a2 - 1), [b1], z)
    # 断言应用操作符后得到的结果与预期的 h2 相等
    assert tn(apply_operators(h,
        # 调用 devise_plan 函数，计算两个超函数的差异操作
        devise_plan(Hyper_Function((a1 + 1, a2 - 1), [b1]),
            Hyper_Function((a1, a2), [b1]), z), op),
        h2, z)
def test_plan_derivatives():
    # 定义变量 a1, a2, a3 分别赋值为 1, 2, 和 '1/2'
    a1, a2, a3 = 1, 2, S('1/2')
    # 定义变量 b1, b2 分别赋值为 3 和 '5/2'
    b1, b2 = 3, S('5/2')
    # 使用给定的参数创建 Hyper_Function 对象 h
    h = Hyper_Function((a1, a2, a3), (b1, b2))
    # 使用稍作修改后的参数创建另一个 Hyper_Function 对象 h2
    h2 = Hyper_Function((a1 + 1, a2 + 1, a3 + 2), (b1 + 1, b2 + 1))
    # 设计操作计划 ops，以应用于 h2 和 h
    ops = devise_plan(h2, h, z)
    # 创建 Formula 对象 f，包含 h, z, h(z), 空列表
    f = Formula(h, z, h(z), [])
    # 制作导数操作符 deriv，并应用于 f.M 和 z，计算结果后与 h2(z) 比较，并断言结果
    deriv = make_derivative_operator(f.M, z)
    assert tn((apply_operators(f.C, ops, deriv)*f.B)[0], h2(z), z)

    # 使用稍作修改后的参数创建 h2
    h2 = Hyper_Function((a1, a2 - 1, a3 - 2), (b1 - 1, b2 - 1))
    # 重新计算操作计划 ops，并与 h2(z) 断言结果
    ops = devise_plan(h2, h, z)
    assert tn((apply_operators(f.C, ops, deriv)*f.B)[0], h2(z), z)


def test_reduction_operators():
    # 生成随机复数 a1, a2, b1，并用于创建 hyper 函数 h
    a1, a2, b1 = (randcplx(n) for n in range(3))
    h = hyper([a1], [b1], z)

    # 断言不同参数输入 ReduceOrder(2, 0)、ReduceOrder(2, -1)、ReduceOrder(1, '1/2') 的结果为 None
    assert ReduceOrder(2, 0) is None
    assert ReduceOrder(2, -1) is None
    assert ReduceOrder(1, S('1/2')) is None

    # 使用不同参数创建 h2
    h2 = hyper((a1, a2), (b1, a2), z)
    # 断言应用 ReduceOrder(a2, a2) 后的结果与 h2(z) 相等
    assert tn(ReduceOrder(a2, a2).apply(h, op), h2, z)

    # 使用不同参数创建 h2
    h2 = hyper((a1, a2 + 1), (b1, a2), z)
    # 断言应用 ReduceOrder(a2 + 1, a2) 后的结果与 h2(z) 相等
    assert tn(ReduceOrder(a2 + 1, a2).apply(h, op), h2, z)

    # 使用不同参数创建 h2
    h2 = hyper((a2 + 4, a1), (b1, a2), z)
    # 断言应用 ReduceOrder(a2 + 4, a2) 后的结果与 h2(z) 相等
    assert tn(ReduceOrder(a2 + 4, a2).apply(h, op), h2, z)

    # 测试多步骤的降阶操作
    ap = (a2 + 4, a1, b1 + 1)
    bq = (a2, b1, b1)
    # 减少 Hyper_Function(ap, bq) 的阶数，得到 func 和 ops
    func, ops = reduce_order(Hyper_Function(ap, bq))
    # 断言结果与 hyper(ap, bq, z) 相等
    assert func.ap == (a1,)
    assert func.bq == (b1,)
    assert tn(apply_operators(h, ops, op), hyper(ap, bq, z), z)


def test_shift_operators():
    # 生成随机复数 a1, a2, b1, b2, b3，并用于创建 hyper 函数 h
    a1, a2, b1, b2, b3 = (randcplx(n) for n in range(5))
    h = hyper((a1, a2), (b1, b2, b3), z)

    # 断言 ShiftA(0) 和 ShiftB(1) 引发 ValueError
    raises(ValueError, lambda: ShiftA(0))
    raises(ValueError, lambda: ShiftB(1))

    # 断言 ShiftA(a1) 的应用结果与预期的 hyper((a1 + 1, a2), (b1, b2, b3), z) 相等
    assert tn(ShiftA(a1).apply(h, op), hyper((a1 + 1, a2), (b1, b2, b3), z), z)
    # 断言 ShiftA(a2) 的应用结果与预期的 hyper((a1, a2 + 1), (b1, b2, b3), z) 相等
    assert tn(ShiftA(a2).apply(h, op), hyper((a1, a2 + 1), (b1, b2, b3), z), z)
    # 断言 ShiftB(b1) 的应用结果与预期的 hyper((a1, a2), (b1 - 1, b2, b3), z) 相等
    assert tn(ShiftB(b1).apply(h, op), hyper((a1, a2), (b1 - 1, b2, b3), z), z)
    # 断言 ShiftB(b2) 的应用结果与预期的 hyper((a1, a2), (b1, b2 - 1, b3), z) 相等
    assert tn(ShiftB(b2).apply(h, op), hyper((a1, a2), (b1, b2 - 1, b3), z), z)
    # 断言 ShiftB(b3) 的应用结果与预期的 hyper((a1, a2), (b1, b2, b3 - 1), z) 相等
    assert tn(ShiftB(b3).apply(h, op), hyper((a1, a2), (b1, b2, b3 - 1), z), z)


def test_ushift_operators():
    # 生成随机复数 a1, a2, b1, b2, b3，并用于创建 hyper 函数 h
    a1, a2, b1, b2, b3 = (randcplx(n) for n in range(5))
    h = hyper((a1, a2), (b1, b2, b3), z)

    # 断言 UnShiftA((1,), (), 0, z) 和 UnShiftB((), (-1,), 0, z) 引发 ValueError
    raises(ValueError, lambda: UnShiftA((1,), (), 0, z))
    raises(ValueError, lambda: UnShiftB((), (-1,), 0, z))
    # 断言 UnShiftA((1,), (0, -1, 1), 0, z) 和 UnShiftB((0, 1), (1,), 0, z) 引发 ValueError
    raises(ValueError, lambda: UnShiftA((1,), (0, -1, 1), 0, z))
    raises(ValueError, lambda: UnShiftB((0, 1), (1,), 0, z))

    # 创建 UnShiftA 对象 s，并断言其应用结果与预期的 hyper((a1 - 1, a2), (b1, b2, b3), z) 相等
    s = UnShiftA((a1, a2), (b1, b2, b3), 0, z)
    assert tn(s.apply(h, op), hyper((a1 - 1, a2), (b1, b2, b3), z), z)
    # 创建 UnShiftA 对象 s，并断言其应用结果与预期的 hyper((a1, a2 - 1), (b1, b2, b3), z) 相等
    s = UnShiftA((a1, a2), (b1, b2, b3), 1, z)
    assert tn(s.apply(h, op), hyper((a1, a2 - 1), (b1, b2, b3), z), z)

    # 创建 UnShiftB 对象
    """
    This helper function tries to hyperexpand() the meijer g-function
    corresponding to the parameters a1, a2, b1, b2.
    It returns False if this expansion still contains g-functions.
    If numeric is True, it also tests the so-obtained formula numerically
    (at random values) and returns False if the test fails.
    Else it returns True.
    """
    # 导入扩展函数和取消极化函数
    from sympy.core.function import expand
    from sympy.functions.elementary.complexes import unpolarify
    # 对 meijer g 函数进行超展开
    r = hyperexpand(meijerg(a1, a2, b1, b2, z))
    # 检查超展开结果是否仍包含 meijer g 函数
    if r.has(meijerg):
        return False
    # 注意事项：hyperexpand() 返回一个真正分支的函数，而数值评估只适用于主分支。
    # 由于我们在主分支上评估，这不应该是问题，但类似 exp_polar(I*pi/2*x)**a 的表达式会被错误评估。
    # 因此我们必须消除它们。expand 在这方面具有启发式作用...
    r = unpolarify(expand(r, force=True, power_base=True, power_exp=False,
                          mul=False, log=False, multinomial=False, basic=False))

    # 如果不需要数值测试，直接返回 True
    if not numeric:
        return True

    # 准备替换字典，用随机复数替换 meijer g 函数的自由符号
    repl = {}
    for n, ai in enumerate(meijerg(a1, a2, b1, b2, z).free_symbols - {z}):
        repl[ai] = randcplx(n)
    # 对 meijer g 函数及其展开结果进行数值测试
    return tn(meijerg(a1, a2, b1, b2, z).subs(repl), r.subs(repl), z)
@slow
# 定义一个名为 test_meijerg_expand 的测试函数，用于测试 meijerg 函数的扩展性
def test_meijerg_expand():
    # 导入所需的函数和模块
    from sympy.simplify.gammasimp import gammasimp
    from sympy.simplify.simplify import simplify
    # 根据 mpmath 文档的示例进行断言
    assert hyperexpand(meijerg([[], []], [[0], []], -z)) == exp(z)

    # 测试特定输入的超函函数展开结果是否符合预期
    assert hyperexpand(meijerg([[1, 1], []], [[1], [0]], z)) == \
        log(z + 1)
    assert hyperexpand(meijerg([[1, 1], []], [[1], [1]], z)) == \
        z/(z + 1)
    assert hyperexpand(meijerg([[], []], [[S.Half], [0]], (z/2)**2)) \
        == sin(z)/sqrt(pi)
    assert hyperexpand(meijerg([[], []], [[0], [S.Half]], (z/2)**2)) \
        == cos(z)/sqrt(pi)
    assert can_do_meijer([], [a], [a - 1, a - S.Half], [])
    assert can_do_meijer([], [], [a/2], [-a/2], False)  # branches...
    assert can_do_meijer([a], [b], [a], [b, a - 1])

    # 根据维基百科的示例进行断言
    assert hyperexpand(meijerg([1], [], [], [0], z)) == \
        Piecewise((0, abs(z) < 1), (1, abs(1/z) < 1),
                 (meijerg([1], [], [], [0], z), True))
    assert hyperexpand(meijerg([], [1], [0], [], z)) == \
        Piecewise((1, abs(z) < 1), (0, abs(1/z) < 1),
                 (meijerg([], [1], [0], [], z), True))

    # 根据《特殊函数及其逼近》进行断言
    assert can_do_meijer([], [], [a + b/2], [a, a - b/2, a + S.Half])
    assert can_do_meijer(
        [], [], [a], [b], False)  # branches only agree for small z
    assert can_do_meijer([], [S.Half], [a], [-a])
    assert can_do_meijer([], [], [a, b], [])
    assert can_do_meijer([], [], [a, b], [])
    assert can_do_meijer([], [], [a, a + S.Half], [b, b + S.Half])
    assert can_do_meijer([], [], [a, -a], [0, S.Half], False)  # dito
    assert can_do_meijer([], [], [a, a + S.Half, b, b + S.Half], [])
    assert can_do_meijer([S.Half], [], [0], [a, -a])
    assert can_do_meijer([S.Half], [], [a], [0, -a], False)  # dito
    assert can_do_meijer([], [a - S.Half], [a, b], [a - S.Half], False)
    assert can_do_meijer([], [a + S.Half], [a + b, a - b, a], [], False)
    assert can_do_meijer([a + S.Half], [], [b, 2*a - b, a], [], False)

    # 测试一个特定的情况是否为零
    assert can_do_meijer([], [], [], [a, b])

    # 测试一个已知 bug
    assert hyperexpand(meijerg([0, 2], [], [], [-1, 1], z)) == \
        Piecewise((0, abs(z) < 1),
                  (z*(1 - 1/z**2)/2, abs(1/z) < 1),
                  (meijerg([0, 2], [], [], [-1, 1], z), True))

    # 测试最简单的答案是否正确返回
    assert gammasimp(simplify(hyperexpand(
        meijerg([1], [1 - a], [-a/2, -a/2 + S.Half], [], 1/z)))) == \
        -2*sqrt(pi)*(sqrt(z + 1) + 1)**a/a

    # 测试超函数是否正确返回
    assert hyperexpand(meijerg([1], [], [a], [0, 0], z)) == hyper(
        (a,), (a + 1, a + 1), z*exp_polar(I*pi))*z**a*gamma(a)/gamma(a + 1)**2

    # 测试 place 选项
    f = meijerg(((0, 1), ()), ((S.Half,), (0,)), z**2)
    assert hyperexpand(f) == sqrt(pi)/sqrt(1 + z**(-2))
    assert hyperexpand(f, place=0) == sqrt(pi)*z/sqrt(z**2 + 1)


def test_meijerg_lookup():
    # 导入 sympy 库中的特殊函数和对象
    from sympy.functions.special.error_functions import (Ci, Si)
    from sympy.functions.special.gamma_functions import uppergamma
    
    # 断言：展开 Meijer G 函数的结果等于右侧的表达式
    assert hyperexpand(meijerg([a], [], [b, a], [], z)) == \
        z**b*exp(z)*gamma(-a + b + 1)*uppergamma(a - b, z)
    
    # 断言：展开 Meijer G 函数的结果等于右侧的表达式
    assert hyperexpand(meijerg([0], [], [0, 0], [], z)) == \
        exp(z)*uppergamma(0, z)
    
    # 断言：判断是否能计算 Meijer G 函数
    assert can_do_meijer([a], [], [b, a + 1], [])
    
    # 断言：判断是否能计算 Meijer G 函数
    assert can_do_meijer([a], [], [b + 2, a], [])
    
    # 断言：判断是否能计算 Meijer G 函数
    assert can_do_meijer([a], [], [b - 2, a], [])
    
    # 断言：展开 Meijer G 函数的结果等于右侧的表达式
    assert hyperexpand(meijerg([a], [], [a, a, a - S.Half], [], z)) == \
        -sqrt(pi)*z**(a - S.Half)*(2*cos(2*sqrt(z))*(Si(2*sqrt(z)) - pi/2)
                                   - 2*sin(2*sqrt(z))*Ci(2*sqrt(z)))
    
    # 断言：展开 Meijer G 函数的结果等于右侧的表达式，多个等价的 Meijer G 函数表达式
    assert hyperexpand(meijerg([a], [], [a, a - S.Half, a], [], z)) == \
        hyperexpand(meijerg([a], [], [a, a - S.Half, a], [], z)) == \
        hyperexpand(meijerg([a], [], [a - S.Half, a, a], [], z))
    
    # 断言：判断是否能计算 Meijer G 函数
    assert can_do_meijer([a - 1], [], [a + 2, a - Rational(3, 2), a + 1], [])
@XFAIL
def test_meijerg_expand_fail():
    # 这些测试基本上测试 hyper([], [1/2 - a, 1/2 + 1, 1/2], z)，
    # 这个函数非常混乱。但由于 meijer g 函数实际上产生贝塞尔函数的和，
    # 有时可以简化很多，并且会放入表格中...
    assert can_do_meijer([], [], [a + S.Half], [a, a - b/2, a + b/2])
    assert can_do_meijer([], [], [0, S.Half], [a, -a])
    assert can_do_meijer([], [], [3*a - S.Half, a, -a - S.Half], [a - S.Half])
    assert can_do_meijer([], [], [0, a - S.Half, -a - S.Half], [S.Half])
    assert can_do_meijer([], [], [a, b + S.Half, b], [2*b - a])
    assert can_do_meijer([], [], [a, b + S.Half, b, 2*b - a])
    assert can_do_meijer([S.Half], [], [-a, a], [0])


@slow
def test_meijerg():
    # 仔细设置参数。
    # 注意：这些测试以前有时会失败。我认为现在已经修复了，但如果你在这里遇到了难以解释的测试失败，请告诉我种子。
    a1, a2 = (randcplx(n) - 5*I - n*I for n in range(2))
    b1, b2 = (randcplx(n) + 5*I + n*I for n in range(2))
    b3, b4, b5, a3, a4, a5 = (randcplx() for n in range(6))
    g = meijerg([a1], [a3, a4], [b1], [b3, b4], z)

    assert ReduceOrder.meijer_minus(3, 4) is None
    assert ReduceOrder.meijer_plus(4, 3) is None

    g2 = meijerg([a1, a2], [a3, a4], [b1], [b3, b4, a2], z)
    assert tn(ReduceOrder.meijer_plus(a2, a2).apply(g, op), g2, z)

    g2 = meijerg([a1, a2], [a3, a4], [b1], [b3, b4, a2 + 1], z)
    assert tn(ReduceOrder.meijer_plus(a2, a2 + 1).apply(g, op), g2, z)

    g2 = meijerg([a1, a2 - 1], [a3, a4], [b1], [b3, b4, a2 + 2], z)
    assert tn(ReduceOrder.meijer_plus(a2 - 1, a2 + 2).apply(g, op), g2, z)

    g2 = meijerg([a1], [a3, a4, b2 - 1], [b1, b2 + 2], [b3, b4], z)
    assert tn(ReduceOrder.meijer_minus(
        b2 + 2, b2 - 1).apply(g, op), g2, z, tol=1e-6)

    # 测试多步减少
    an = [a1, a2]
    bq = [b3, b4, a2 + 1]
    ap = [a3, a4, b2 - 1]
    bm = [b1, b2 + 1]
    niq, ops = reduce_order_meijer(G_Function(an, ap, bm, bq))
    assert niq.an == (a1,)
    assert set(niq.ap) == {a3, a4}
    assert niq.bm == (b1,)
    assert set(niq.bq) == {b3, b4}
    assert tn(apply_operators(g, ops, op), meijerg(an, ap, bm, bq, z), z)


def test_meijerg_shift_operators():
    # 仔细设置参数。XXX 这个测试有时仍然会失败
    a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 = (randcplx(n) for n in range(10))
    g = meijerg([a1], [a3, a4], [b1], [b3, b4], z)

    assert tn(MeijerShiftA(b1).apply(g, op),
              meijerg([a1], [a3, a4], [b1 + 1], [b3, b4], z), z)
    assert tn(MeijerShiftB(a1).apply(g, op),
              meijerg([a1 - 1], [a3, a4], [b1], [b3, b4], z), z)
    assert tn(MeijerShiftC(b3).apply(g, op),
              meijerg([a1], [a3, a4], [b1], [b3 + 1, b4], z), z)
    assert tn(MeijerShiftD(a3).apply(g, op),
              meijerg([a1], [a3 - 1, a4], [b1], [b3, b4], z), z)

    s = MeijerUnShiftA([a1], [a3, a4], [b1], [b3, b4], 0, z)
    # 断言：应用操作函数 g 到 s 上，并与给定的 Meijer 函数进行比较
    assert tn(
        s.apply(g, op), meijerg([a1], [a3, a4], [b1 - 1], [b3, b4], z), z)

    # 创建 MeijerUnShiftC 类的实例 s，用给定参数初始化
    s = MeijerUnShiftC([a1], [a3, a4], [b1], [b3, b4], 0, z)
    # 断言：应用操作函数 g 到 s 上，并与给定的 Meijer 函数进行比较
    assert tn(
        s.apply(g, op), meijerg([a1], [a3, a4], [b1], [b3 - 1, b4], z), z)

    # 创建 MeijerUnShiftB 类的实例 s，用给定参数初始化
    s = MeijerUnShiftB([a1], [a3, a4], [b1], [b3, b4], 0, z)
    # 断言：应用操作函数 g 到 s 上，并与给定的 Meijer 函数进行比较
    assert tn(
        s.apply(g, op), meijerg([a1 + 1], [a3, a4], [b1], [b3, b4], z), z)

    # 创建 MeijerUnShiftD 类的实例 s，用给定参数初始化
    s = MeijerUnShiftD([a1], [a3, a4], [b1], [b3, b4], 0, z)
    # 断言：应用操作函数 g 到 s 上，并与给定的 Meijer 函数进行比较
    assert tn(
        s.apply(g, op), meijerg([a1], [a3 + 1, a4], [b1], [b3, b4], z), z)
# 声明一个装饰器 @slow，用于标记测试函数为较慢运行的测试用例
@slow
# 定义函数 test_meijerg_confluence，用于测试 Meijer G 函数的性质
def test_meijerg_confluence():
    # 声明一个内部函数 t，用于检查 Meijer G 函数的特定性质
    def t(m, a, b):
        # 导入 sympy 的 sympify 函数，用于将输入转换为 sympy 对象
        from sympy.core.sympify import sympify
        # 将 a 和 b 转换为 sympy 对象
        a, b = sympify([a, b])
        # 备份原始的 m 到 m_，并对 m 进行超级展开
        m_ = m
        m = hyperexpand(m)
        # 检查 m 是否符合 Piecewise((a, abs(z) < 1), (b, abs(1/z) < 1), (m_, True)) 的形式
        if not m == Piecewise((a, abs(z) < 1), (b, abs(1/z) < 1), (m_, True)):
            return False
        # 检查 m 的第一个参数和第二个参数是否分别等于 a 和 b
        if not (m.args[0].args[0] == a and m.args[1].args[0] == b):
            return False
        # 生成一个随机复数 z0，并检查 m(z0) 和 a(z0) 的差距是否小于 1e-10
        z0 = randcplx()/10
        if abs(m.subs(z, z0).n() - a.subs(z, z0).n()).n() > 1e-10:
            return False
        # 检查 m(1/z0) 和 b(1/z0) 的差距是否小于 1e-10
        if abs(m.subs(z, 1/z0).n() - b.subs(z, 1/z0).n()).n() > 1e-10:
            return False
        # 如果所有条件均满足，则返回 True
        return True

    # 断言不同的 Meijer G 函数及其参数，确保 t 函数返回 True
    assert t(meijerg([], [1, 1], [0, 0], [], z), -log(z), 0)
    assert t(meijerg([], [3, 1], [0, 0], [], z), -z**2/4 + z - log(z)/2 - Rational(3, 4), 0)
    assert t(meijerg([], [3, 1], [-1, 0], [], z), z**2/12 - z/2 + log(z)/2 + Rational(1, 4) + 1/(6*z), 0)
    assert t(meijerg([], [1, 1, 1, 1], [0, 0, 0, 0], [], z), -log(z)**3/6, 0)
    assert t(meijerg([1, 1], [], [], [0, 0], z), 0, -log(1/z))
    assert t(meijerg([1, 1], [2, 2], [1, 1], [0, 0], z), -z*log(z) + 2*z, -log(1/z) + 2)
    assert t(meijerg([S.Half], [1, 1], [0, 0], [Rational(3, 2)], z), log(z)/2 - 1, 0)

    # 定义函数 u，用于测试不同参数下的 Meijer G 函数
    def u(an, ap, bm, bq):
        # 计算 Meijer G 函数并进行超级展开
        m = meijerg(an, ap, bm, bq, z)
        m2 = hyperexpand(m, allow_hyper=True)
        # 如果 m2 包含 Meijer G 函数并且不是 Piecewise 且长度不为 3，则返回 False
        if m2.has(meijerg) and not (m2.is_Piecewise and len(m2.args) == 3):
            return False
        # 调用 t 函数检查 m 和 m2 的其他性质是否满足
        return tn(m, m2, z)
    assert u([], [1], [0, 0], [])
    assert u([1, 1], [], [], [0])
    assert u([1, 1], [2, 2, 5], [1, 1, 6], [0, 0])
    assert u([1, 1], [2, 2, 5], [1, 1, 6], [0])

# 定义函数 test_meijerg_with_Floats，用于测试带有浮点数的 Meijer G 函数
def test_meijerg_with_Floats():
    # 见问题 #10681，导入实数域 RR
    from sympy.polys.domains.realfield import RR
    # 计算 Meijer G 函数 f，并指定其值和表达式 g
    f = meijerg(((3.0, 1), ()), ((Rational(3, 2),), (0,)), z)
    a = -2.3632718012073
    g = a*z**Rational(3, 2)*hyper((-0.5, Rational(3, 2)), (Rational(5, 2),), z*exp_polar(I*pi))
    # 断言 f/g 的数值近似等于 1.0，精度为 1e-12
    assert RR.almosteq((hyperexpand(f)/g).n(), 1.0, 1e-12)

# 定义函数 test_lerchphi，用于测试 Lerch φ 函数
def test_lerchphi():
    # 导入 Lerch φ 函数和 polylog 函数
    from sympy.functions.special.zeta_functions import (lerchphi, polylog)
    # 导入 gamma 简化函数 gammasimp
    from sympy.simplify.gammasimp import gammasimp
    # 断言超级展开后的超几何函数等于 Lerch φ 函数
    assert hyperexpand(hyper([1, a], [a + 1], z)/a) == lerchphi(z, 1, a)
    assert hyperexpand(hyper([1, a, a], [a + 1, a + 1], z)/a**2) == lerchphi(z, 2, a)
    assert hyperexpand(hyper([1, a, a, a], [a + 1, a + 1, a + 1], z)/a**3) == lerchphi(z, 3, a)
    assert hyperexpand(hyper([1] + [a]*10, [a + 1]*10, z)/a**10) == lerchphi(z, 10, a)
    # 断言 gamma 简化超级展开的 Meijer G 函数等于 Lerch φ 函数
    assert gammasimp(hyperexpand(meijerg([0, 1 - a], [], [0], [-a], exp_polar(-I*pi)*z))) == lerchphi(z, 1, a)
    assert gammasimp(hyperexpand(meijerg([0, 1 - a, 1 - a], [], [0], [-a, -a], exp_polar(-I*pi)*z))) == lerchphi(z, 2, a)
    assert gammasimp(hyperexpand(meijerg([0, 1 - a, 1 - a, 1 - a], [], [0], [-a, -a, -a], exp_polar(-I*pi)*z))) == lerchphi(z, 3, a)
    # 断言超级展开的超几何函数等于 -log(1 + -z)
    assert hyperexpand(z*hyper([1, 1], [2], z)) == -log(1 + -z)
    # 断言：展开超几何函数 z * hyper([1, 1, 1], [2, 2], z) 应该等于 polylog(2, z)
    assert hyperexpand(z*hyper([1, 1, 1], [2, 2], z)) == polylog(2, z)

    # 断言：展开超几何函数 z * hyper([1, 1, 1, 1], [2, 2, 2], z) 应该等于 polylog(3, z)
    assert hyperexpand(z*hyper([1, 1, 1, 1], [2, 2, 2], z)) == polylog(3, z)

    # 断言：展开超几何函数 hyper([1, a, 1 + S.Half], [a + 1, S.Half], z) 应该等于计算得到的表达式
    assert hyperexpand(hyper([1, a, 1 + S.Half], [a + 1, S.Half], z)) == \
        -2*a/(z - 1) + (-2*a**2 + a)*lerchphi(z, 1, a)

    # 现在进行数值测试，确保正确执行了化简等操作

    # 断言：检查一个有理函数 (负整数次 polylog)
    assert can_do([2, 2, 2], [1, 1])

    # 注意：这些测试包含 log(1-x) 等函数... 确保 |z| < 1
    # 对 polylog 进行次数的减少
    assert can_do([1, 1, 1, b + 5], [2, 2, b], div=10)

    # 对 lerchphi 进行次数的减少
    # 注意：mpmath 中的 lerchphi 不稳定
    assert can_do(
        [1, a, a, a, b + 5], [a + 1, a + 1, a + 1, b], numerical=False)

    # 测试一个 bug
    from sympy.functions.elementary.complexes import Abs
    assert hyperexpand(hyper([S.Half, S.Half, S.Half, 1],
                             [Rational(3, 2), Rational(3, 2), Rational(3, 2)], Rational(1, 4))) == \
        Abs(-polylog(3, exp_polar(I*pi)/2) + polylog(3, S.Half))
# 定义测试函数 test_partial_simp
def test_partial_simp():
    # 首先测试超几何函数的公式是否有效
    a, b, c, d, e = (randcplx() for _ in range(5))
    # 对于给定的超几何函数列表，生成相应的函数对象
    for func in [Hyper_Function([a, b, c], [d, e]),
            Hyper_Function([], [a, b, c, d, e])]:
        # 构建超几何函数的公式
        f = build_hypergeometric_formula(func)
        # 获取公式的参数 z
        z = f.z
        # 断言公式的闭合形式等于原始超几何函数在 z 处的值
        assert f.closed_form == func(z)
        # 计算第一阶导数并比较
        deriv1 = f.B.diff(z)*z
        deriv2 = f.M*f.B
        for func1, func2 in zip(deriv1, deriv2):
            assert tn(func1, func2, z)

    # 现在测试公式是否部分简化
    a, b, z = symbols('a b z')
    # 断言对给定的超几何函数进行展开后的结果是否正确
    assert hyperexpand(hyper([3, a], [1, b], z)) == \
        (-a*b/2 + a*z/2 + 2*a)*hyper([a + 1], [b], z) \
        + (a*b/2 - 2*a + 1)*hyper([a], [b], z)
    # 断言展开后的超几何函数与原始函数在 z 处的值相等
    assert tn(
        hyperexpand(hyper([3, d], [1, e], z)), hyper([3, d], [1, e], z), z)
    assert hyperexpand(hyper([3], [1, a, b], z)) == \
        hyper((), (a, b), z) \
        + z*hyper((), (a + 1, b), z)/(2*a) \
        - z*(b - 4)*hyper((), (a + 1, b + 1), z)/(2*a*b)
    assert tn(
        hyperexpand(hyper([3], [1, d, e], z)), hyper([3], [1, d, e], z), z)


# 定义测试函数 test_hyperexpand_special
def test_hyperexpand_special():
    # 断言对特定的超几何函数进行展开后的结果是否正确
    assert hyperexpand(hyper([a, b], [c], 1)) == \
        gamma(c)*gamma(c - a - b)/gamma(c - a)/gamma(c - b)
    assert hyperexpand(hyper([a, b], [1 + a - b], -1)) == \
        gamma(1 + a/2)*gamma(1 + a - b)/gamma(1 + a)/gamma(1 + a/2 - b)
    assert hyperexpand(hyper([a, b], [1 + b - a], -1)) == \
        gamma(1 + b/2)*gamma(1 + b - a)/gamma(1 + b)/gamma(1 + b/2 - a)
    assert hyperexpand(meijerg([1 - z - a/2], [1 - z + a/2], [b/2], [-b/2], 1)) == \
        gamma(1 - 2*z)*gamma(z + a/2 + b/2)/gamma(1 - z + a/2 - b/2) \
        /gamma(1 - z - a/2 + b/2)/gamma(1 - z + a/2 + b/2)
    assert hyperexpand(hyper([a], [b], 0)) == 1
    assert hyper([a], [b], 0) != 0


# 定义测试函数 test_Mod1_behavior
def test_Mod1_behavior():
    from sympy.core.symbol import Symbol
    from sympy.simplify.simplify import simplify
    n = Symbol('n', integer=True)
    # 注意：这里不应该出现死循环
    assert simplify(hyperexpand(meijerg([1], [], [n + 1], [0], z))) == \
        lowergamma(n + 1, z)


# 定义测试函数 test_prudnikov_misc
@slow
def test_prudnikov_misc():
    # 断言对于给定的参数列表，can_do 函数能够处理
    assert can_do([1, (3 + I)/2, (3 - I)/2], [Rational(3, 2), 2])
    assert can_do([S.Half, a - 1], [Rational(3, 2), a + 1], lowerplane=True)
    assert can_do([], [b + 1])
    assert can_do([a], [a - 1, b + 1])

    assert can_do([a], [a - S.Half, 2*a])
    assert can_do([a], [a - S.Half, 2*a + 1])
    assert can_do([a], [a - S.Half, 2*a - 1])
    assert can_do([a], [a + S.Half, 2*a])
    assert can_do([a], [a + S.Half, 2*a + 1])
    assert can_do([a], [a + S.Half, 2*a - 1])
    assert can_do([S.Half], [b, 2 - b])
    assert can_do([S.Half], [b, 3 - b])
    assert can_do([1], [2, b])

    assert can_do([a, a + S.Half], [2*a, b, 2*a - b + 1])
    assert can_do([a, a + S.Half], [S.Half, 2*a, 2*a + S.Half])
    assert can_do([a], [a + 1], lowerplane=True)  # lowergamma


# 定义测试函数 test_prudnikov_1
def test_prudnikov_1():
    # A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev (1990).
    # 这个测试函数没有实际代码，仅作为文档存在
    # Integrals and Series: More Special Functions, Vol. 3,.
    # Gordon and Breach Science Publisher
    
    # 7.3.1
    # 使用 `can_do` 函数检查给定参数组合是否能计算特定特殊函数的值
    assert can_do([a, -a], [S.Half])
    assert can_do([a, 1 - a], [S.Half])
    assert can_do([a, 1 - a], [Rational(3, 2)])
    assert can_do([a, 2 - a], [S.Half])
    assert can_do([a, 2 - a], [Rational(3, 2)])
    assert can_do([a, 2 - a], [Rational(3, 2)])
    assert can_do([a, a + S.Half], [2*a - 1])
    assert can_do([a, a + S.Half], [2*a])
    assert can_do([a, a + S.Half], [2*a + 1])
    assert can_do([a, a + S.Half], [S.Half])
    assert can_do([a, a + S.Half], [Rational(3, 2)])
    assert can_do([a, a/2 + 1], [a/2])
    assert can_do([1, b], [2])
    # 使用 `can_do` 函数检查 Lerch Phi 函数对于给定参数组合是否能计算，其中 numerical 参数设置为 False 表示处理复杂的分支情况，当 |z| > 1 时
    assert can_do([1, b], [b + 1], numerical=False)  # Lerch Phi
             # NOTE: branches are complicated for |z| > 1
    
    # 使用 `can_do` 函数检查给定参数组合是否能计算特殊函数的值
    assert can_do([a], [2*a])
    assert can_do([a], [2*a + 1])
    assert can_do([a], [2*a - 1])
@slow
def test_prudnikov_2():
    # 设置变量 h 为 SymPy 中的 Half
    h = S.Half
    # 断言验证函数 can_do 的返回结果
    assert can_do([-h, -h], [h])
    assert can_do([-h, h], [3*h])
    assert can_do([-h, h], [5*h])
    assert can_do([-h, h], [7*h])
    assert can_do([-h, 1], [h])

    # 循环遍历不同的 p, n, m 值进行断言验证
    for p in [-h, h]:
        for n in [-h, h, 1, 3*h, 2, 5*h, 3, 7*h, 4]:
            for m in [-h, h, 3*h, 5*h, 7*h]:
                assert can_do([p, n], [m])
        for n in [1, 2, 3, 4]:
            for m in [1, 2, 3, 4]:
                assert can_do([p, n], [m])


def test_prudnikov_3():
    # 设置变量 h 为 SymPy 中的 Half
    h = S.Half
    # 断言验证函数 can_do 的返回结果，使用 SymPy 中的 Rational 创建有理数对象
    assert can_do([Rational(1, 4), Rational(3, 4)], [h])
    assert can_do([Rational(1, 4), Rational(3, 4)], [3*h])
    assert can_do([Rational(1, 3), Rational(2, 3)], [3*h])
    assert can_do([Rational(3, 4), Rational(5, 4)], [h])
    assert can_do([Rational(3, 4), Rational(5, 4)], [3*h])


@tooslow
def test_prudnikov_3_slow():
    # XXX: This is marked as tooslow and hence skipped in CI. None of the
    # individual cases below fails or hangs. Some cases are slow and the loops
    # below generate 280 different cases. Is it really necessary to test all
    # 280 cases here?
    # 设置变量 h 为 SymPy 中的 Half
    h = S.Half
    # 遍历多重循环，每个循环中进行断言验证函数 can_do 的返回结果
    for p in [1, 2, 3, 4]:
        for n in [-h, h, 1, 3*h, 2, 5*h, 3, 7*h, 4, 9*h]:
            for m in [1, 3*h, 2, 5*h, 3, 7*h, 4]:
                assert can_do([p, m], [n])


@slow
def test_prudnikov_4():
    # 设置变量 h 为 SymPy 中的 Half
    h = S.Half
    # 遍历多重循环，每个循环中进行断言验证函数 can_do 的返回结果
    for p in [3*h, 5*h, 7*h]:
        for n in [-h, h, 3*h, 5*h, 7*h]:
            for m in [3*h, 2, 5*h, 3, 7*h, 4]:
                assert can_do([p, m], [n])
        for n in [1, 2, 3, 4]:
            for m in [2, 3, 4]:
                assert can_do([p, m], [n])


@slow
def test_prudnikov_5():
    # 设置变量 h 为 SymPy 中的 Half
    h = S.Half

    # 多层嵌套循环进行断言验证函数 can_do 的返回结果
    for p in [1, 2, 3]:
        for q in range(p, 4):
            for r in [1, 2, 3]:
                for s in range(r, 4):
                    assert can_do([-h, p, q], [r, s])

    for p in [h, 1, 3*h, 2, 5*h, 3]:
        for q in [h, 3*h, 5*h]:
            for r in [h, 3*h, 5*h]:
                for s in [h, 3*h, 5*h]:
                    if s <= q and s <= r:
                        assert can_do([-h, p, q], [r, s])

    for p in [h, 1, 3*h, 2, 5*h, 3]:
        for q in [1, 2, 3]:
            for r in [h, 3*h, 5*h]:
                for s in [1, 2, 3]:
                    assert can_do([-h, p, q], [r, s])


@slow
def test_prudnikov_6():
    # 设置变量 h 为 SymPy 中的 Half
    h = S.Half

    # 多层嵌套循环进行断言验证函数 can_do 的返回结果
    for m in [3*h, 5*h]:
        for n in [1, 2, 3]:
            for q in [h, 1, 2]:
                for p in [1, 2, 3]:
                    assert can_do([h, q, p], [m, n])
            for q in [1, 2, 3]:
                for p in [3*h, 5*h]:
                    assert can_do([h, q, p], [m, n])

    for q in [1, 2]:
        for p in [1, 2, 3]:
            for m in [1, 2, 3]:
                for n in [1, 2, 3]:
                    assert can_do([h, q, p], [m, n])

    assert can_do([h, h, 5*h], [3*h, 3*h])
    assert can_do([h, 1, 5*h], [3*h, 3*h])
    assert can_do([h, 2, 2], [1, 3])

    # pages 435 to 457 contain more PFDD and stuff like this
def test_prudnikov_7():
    # 确定 can_do 函数对 [3], [6] 返回 True
    assert can_do([3], [6])

    h = S.Half
    # 对于每个 n 在 [1/2, 3/2, 5/2, 7/2] 中，确保 can_do 函数对 [-1/2], [n] 返回 True
    for n in [h, 3*h, 5*h, 7*h]:
        assert can_do([-h], [n])
    # 对于每个 m 在 [-1/2, 1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4] 中，和每个 n 在 [-1/2, 1/2, 3/2, 5/2, 7/2, 1, 2, 3, 4] 中，确保 can_do 函数返回 True
    for m in [-h, h, 1, 3*h, 2, 5*h, 3, 7*h, 4]:  # HERE
        for n in [-h, h, 3*h, 5*h, 7*h, 1, 2, 3, 4]:
            assert can_do([m], [n])


@slow
def test_prudnikov_8():
    h = S.Half

    # 7.12.2
    # 对于每个 ai 在 [1, 2, 3] 中，每个 bi 在 [1, 2, 3] 中，每个 ci 在 [1, ai] 中，每个 di 在 [-1/2, 1, 3/2, 2, 5/2, 3] 中，确保 can_do 函数返回 True
    for ai in [1, 2, 3]:
        for bi in [1, 2, 3]:
            for ci in range(1, ai + 1):
                for di in [h, 1, 3*h, 2, 5*h, 3]:
                    assert can_do([ai, bi], [ci, di])
        # 对于每个 bi 在 [-1/2, 1/2, 3/2, 5/2] 中，每个 ci 在 [-1/2, 1, 3/2, 2, 5/2, 3] 中，每个 di 在 [1, 2, 3] 中，确保 can_do 函数返回 True
        for bi in [3*h, 5*h]:
            for ci in [h, 1, 3*h, 2, 5*h, 3]:
                for di in [1, 2, 3]:
                    assert can_do([ai, bi], [ci, di])

    # 对于每个 ai 在 [-1/2, 1/2, 3/2, 5/2] 中，每个 bi 在 [1, 2, 3] 中，每个 ci 在 [1, -1/2, 1, 3/2, 5/2, 3] 中，每个 di 在 [1, 2, 3] 中，确保 can_do 函数返回 True
    for ai in [-h, h, 3*h, 5*h]:
        for bi in [1, 2, 3]:
            for ci in [h, 1, 3*h, 2, 5*h, 3]:
                for di in [1, 2, 3]:
                    assert can_do([ai, bi], [ci, di])
        # 对于每个 bi 在 [-1/2, 1/2, 3/2, 5/2] 中，每个 ci 在 [-1/2, 3/2, 5/2, 3] 中，每个 di 在 [-1/2, 1, 3/2, 2, 5/2, 3] 中，确保 can_do 函数在 ci <= bi 时返回 True
        for bi in [h, 3*h, 5*h]:
            for ci in [h, 3*h, 5*h, 3]:
                for di in [h, 1, 3*h, 2, 5*h, 3]:
                    if ci <= bi:
                        assert can_do([ai, bi], [ci, di])


def test_prudnikov_9():
    # 7.13.1 [we have a general formula ... so this is a bit pointless]
    # 对于每个 i 在 [0, 1, 2, 3, 4, 5, 6, 7, 8] 中，确保 can_do 函数对空列表 [] 和 [(i + 1)/2] 返回 True
    for i in range(9):
        assert can_do([], [(S(i) + 1)/2])
    # 对于每个 i 在 [0, 1, 2, 3, 4] 中，确保 can_do 函数对空列表 [] 和 [-(2*i + 1)/2] 返回 True
    for i in range(5):
        assert can_do([], [-(2*S(i) + 1)/2])


@slow
def test_prudnikov_10():
    # 7.14.2
    h = S.Half
    # 对于每个 p 在 [-1/2, 1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4] 中，每个 m 在 [1, 2, 3, 4] 中，每个 n 在 [m, 1, 2, 3, 4] 中，确保 can_do 函数返回 True
    for p in [-h, h, 1, 3*h, 2, 5*h, 3, 7*h, 4]:
        for m in [1, 2, 3, 4]:
            for n in range(m, 5):
                assert can_do([p], [m, n])

    # 对于每个 p 在 [1, 2, 3, 4] 中，每个 n 在 [-1/2, 1/2, 3/2, 5/2, 7/2] 中，每个 m 在 [1, 2, 3, 4] 中，确保 can_do 函数返回 True
    for p in [1, 2, 3, 4]:
        for n in [h, 3*h, 5*h, 7*h]:
            for m in [1, 2, 3, 4]:
                assert can_do([p], [n, m])

    # 对于每个 p 在 [-1/2, 1/2, 3/2, 5/2, 7/2] 中，每个 m 在 [-1/2, 1, 2, 5/2, 3, 7/2, 4] 中，确保 can_do 函数对 [7/2], [5/2, m] 和 [7/2], [3/2, m] 返回 True
    for p in [3*h, 5*h, 7*h]:
        for m in [h, 1, 2, 5*h, 3, 7*h, 4]:
            assert can_do([p], [h, m])
            assert can_do([p], [3*h, m])

    # 对于每个 m 在 [-1/2, 1, 2, 5/2, 3, 7/2, 4] 中，确保 can_do 函数对 [-1/2], [1/2, m] 返回 True
    for m in [h, 1, 2, 5*h, 3, 7*h, 4]:
        assert can_do([7*h], [5*h, m])

    # 确保 can_do 函数对 [-1/2], [1/2, 1/2] 返回 True
    assert can_do([Rational(-1, 2)], [S.Half, S.Half])  # shine-integral shi


def test_prudnikov_11():
    # 7.15
    # 确保 can_do 函数对 [a, a + 1/2], [2a, b, 2a - b] 返回 True
    assert can_do([a, a + S.Half], [2*a, b, 2*a - b])
    # 确保 can_do 函数对 [a, a + 1/2], [3/2, 2a, 2a - 1/2] 返回 True
    assert can_do([a, a + S.Half], [Rational(3, 2), 2*a, 2*a - S.Half])

    # 确保 can_do 函数对 [1/4, 3/4], [1/2, 1/2, 1] 返回 True
    assert can_do([R
    # 调用 can_do 函数，验证给定的参数组合是否满足特定条件
    assert can_do([], [S.Half, Rational(3, 2), 1])
    # 调用 can_do 函数，验证给定的参数组合是否满足特定条件
    assert can_do([], [Rational(3, 4), Rational(3, 2), Rational(5, 4)])
    # 调用 can_do 函数，验证给定的参数组合是否满足特定条件
    assert can_do([], [1, 1, Rational(3, 2)])
    # 调用 can_do 函数，验证给定的参数组合是否满足特定条件
    assert can_do([], [1, 2, Rational(3, 2)])
    # 调用 can_do 函数，验证给定的参数组合是否满足特定条件
    assert can_do([], [1, Rational(3, 2), Rational(3, 2)])
    # 调用 can_do 函数，验证给定的参数组合是否满足特定条件
    assert can_do([], [Rational(5, 4), Rational(3, 2), Rational(7, 4)])
    # 调用 can_do 函数，验证给定的参数组合是否满足特定条件
    assert can_do([], [2, Rational(3, 2), Rational(3, 2)])
@slow
def test_prudnikov_2F1():
    h = S.Half
    # 定义 S.Half 为 h，即分数 1/2
    # Elliptic integrals（椭圆积分）
    for p in [-h, h]:
        # 遍历 p 取值为 -1/2 和 1/2
        for m in [h, 3*h, 5*h, 7*h]:
            # 遍历 m 取值为 1/2, 3/2, 5/2, 7/2
            for n in [1, 2, 3, 4]:
                # 遍历 n 取值为 1, 2, 3, 4
                assert can_do([p, m], [n])
                # 调用 can_do 函数，检查是否能计算给定参数组合的函数值


@XFAIL
def test_prudnikov_fail_2F1():
    assert can_do([a, b], [b + 1])  # incomplete beta function（不完全贝塔函数）
    assert can_do([-1, b], [c])    # Poly. also -2, -3 etc（多项式，还有 -2，-3 等等）

    # TODO polys（多项式）

    # Legendre functions（Legendre 函数）:
    assert can_do([a, b], [a + b + S.Half])
    assert can_do([a, b], [a + b - S.Half])
    assert can_do([a, b], [a + b + Rational(3, 2)])
    assert can_do([a, b], [(a + b + 1)/2])
    assert can_do([a, b], [(a + b)/2 + 1])
    assert can_do([a, b], [a - b + 1])
    assert can_do([a, b], [a - b + 2])
    assert can_do([a, b], [2*b])
    assert can_do([a, b], [S.Half])
    assert can_do([a, b], [Rational(3, 2)])
    assert can_do([a, 1 - a], [c])
    assert can_do([a, 2 - a], [c])
    assert can_do([a, 3 - a], [c])
    assert can_do([a, a + S.Half], [c])
    assert can_do([1, b], [c])
    assert can_do([1, b], [Rational(3, 2)])

    assert can_do([Rational(1, 4), Rational(3, 4)], [1])

    # PFDD（可能是某种特定问题的标识）
    o = S.One
    assert can_do([o/8, 1], [o/8*9])
    assert can_do([o/6, 1], [o/6*7])
    assert can_do([o/6, 1], [o/6*13])
    assert can_do([o/5, 1], [o/5*6])
    assert can_do([o/5, 1], [o/5*11])
    assert can_do([o/4, 1], [o/4*5])
    assert can_do([o/4, 1], [o/4*9])
    assert can_do([o/3, 1], [o/3*4])
    assert can_do([o/3, 1], [o/3*7])
    assert can_do([o/8*3, 1], [o/8*11])
    assert can_do([o/5*2, 1], [o/5*7])
    assert can_do([o/5*2, 1], [o/5*12])
    assert can_do([o/5*3, 1], [o/5*8])
    assert can_do([o/5*3, 1], [o/5*13])
    assert can_do([o/8*5, 1], [o/8*13])
    assert can_do([o/4*3, 1], [o/4*7])
    assert can_do([o/4*3, 1], [o/4*11])
    assert can_do([o/3*2, 1], [o/3*5])
    assert can_do([o/3*2, 1], [o/3*8])
    assert can_do([o/5*4, 1], [o/5*9])
    assert can_do([o/5*4, 1], [o/5*14])
    assert can_do([o/6*5, 1], [o/6*11])
    assert can_do([o/6*5, 1], [o/6*17])


@XFAIL
def test_prudnikov_fail_3F2():
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [Rational(1, 3), Rational(2, 3)])
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [Rational(2, 3), Rational(4, 3)])
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [Rational(4, 3), Rational(5, 3)])

    # page 421
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [a*Rational(3, 2), (3*a + 1)/2])

    # pages 422 ...
    assert can_do([Rational(-1, 2), S.Half, S.Half], [1, 1])  # elliptic integrals（椭圆积分）
    assert can_do([Rational(-1, 2), S.Half, 1], [Rational(3, 2), Rational(3, 2)])
    # TODO LOTS more（更多的待办事项）

    # PFDD（可能是某种特定问题的标识）
    assert can_do([Rational(1, 8), Rational(3, 8), 1], [Rational(9, 8), Rational(11, 8)])
    assert can_do([Rational(1, 8), Rational(5, 8), 1], [Rational(9, 8), Rational(13, 8)])
    assert can_do([Rational(1, 8), Rational(7, 8), 1], [Rational(9, 8), Rational(15, 8)])
    # 使用 assert 语句来检查 can_do 函数对于给定参数的返回是否符合预期
    assert can_do([Rational(1, 6), Rational(1, 3), 1], [Rational(7, 6), Rational(4, 3)])
    assert can_do([Rational(1, 6), Rational(2, 3), 1], [Rational(7, 6), Rational(5, 3)])
    assert can_do([Rational(1, 6), Rational(2, 3), 1], [Rational(5, 3), Rational(13, 6)])
    assert can_do([S.Half, 1, 1], [Rational(1, 4), Rational(3, 4)])
    # 这里可能还会有更多的 assert 语句，用来验证更多的输入组合
@XFAIL
# 标记为测试失败，预期会失败

def test_prudnikov_fail_other():
    # 7.11.2
    # 7.12.1
    # 断言测试是否可以执行某些操作，待补充具体含义
    assert can_do([1, a], [b, 1 - 2*a + b])  # ???

    # 7.14.2
    # 断言测试不同参数组合下是否能执行某些操作（例如 struve 或 PFDD）
    assert can_do([Rational(-1, 2)], [S.Half, 1])  # struve
    assert can_do([1], [S.Half, S.Half])  # struve
    assert can_do([Rational(1, 4)], [S.Half, Rational(5, 4)])  # PFDD
    assert can_do([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)])  # PFDD
    assert can_do([1], [Rational(1, 4), Rational(3, 4)])  # PFDD
    assert can_do([1], [Rational(3, 4), Rational(5, 4)])  # PFDD
    assert can_do([1], [Rational(5, 4), Rational(7, 4)])  # PFDD
    # TODO LOTS more

    # 7.15.2
    # 断言测试不同参数组合下是否能执行某些操作（例如 PFDD）
    assert can_do([S.Half, 1], [Rational(3, 4), Rational(5, 4), Rational(3, 2)])  # PFDD
    assert can_do([S.Half, 1], [Rational(7, 4), Rational(5, 4), Rational(3, 2)])  # PFDD

    # 7.16.1
    # 断言测试不同参数组合下是否能执行某些操作（例如 PFDD）
    assert can_do([], [Rational(1, 3), S(2/3)])  # PFDD
    assert can_do([], [Rational(2, 3), S(4/3)])  # PFDD
    assert can_do([], [Rational(5, 3), S(4/3)])  # PFDD

    # XXX this does not *evaluate* right??
    # 断言测试是否能正确评估某个表达式
    assert can_do([], [a, a + S.Half, 2*a - 1])


def test_bug():
    # 创建一个超几何函数对象 h，用给定参数计算其超几何展开后的结果，断言是否相等
    h = hyper([-1, 1], [z], -1)
    assert hyperexpand(h) == (z + 1)/z


def test_omgissue_203():
    # 创建一个超几何函数对象 h，用给定参数计算其超几何展开后的结果，断言是否相等
    h = hyper((-5, -3, -4), (-6, -6), 1)
    assert hyperexpand(h) == Rational(1, 30)
    h = hyper((-6, -7, -5), (-6, -6), 1)
    assert hyperexpand(h) == Rational(-1, 6)
```