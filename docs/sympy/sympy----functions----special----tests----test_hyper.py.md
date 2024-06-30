# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_hyper.py`

```
from sympy.core.containers import Tuple  # 导入 Tuple 类，用于处理元组
from sympy.core.function import Derivative  # 导入 Derivative 类，用于处理导数相关功能
from sympy.core.numbers import (I, Rational, oo, pi)  # 导入一些常用的数学常数和符号
from sympy.core.singleton import S  # 导入 S，表示 sympy 的单例
from sympy.core.symbol import symbols  # 导入 symbols 函数，用于创建符号变量
from sympy.functions.elementary.exponential import (exp, log)  # 导入指数和对数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import cos  # 导入余弦函数
from sympy.functions.special.gamma_functions import gamma  # 导入 Gamma 函数
from sympy.functions.special.hyper import (appellf1, hyper, meijerg)  # 导入超几何函数相关
from sympy.series.order import O  # 导入 O，用于表示阶数的类
from sympy.abc import x, z, k  # 导入符号变量 x, z, k
from sympy.series.limits import limit  # 导入 limit 函数，用于计算极限
from sympy.testing.pytest import raises, slow  # 导入测试相关的函数和装饰器
from sympy.core.random import (
    random_complex_number as randcplx,  # 导入随机复数生成函数，并重命名为 randcplx
    verify_numerically as tn,  # 导入数值验证函数，并重命名为 tn
    test_derivative_numerically as td  # 导入数值导数测试函数，并重命名为 td
)


def test_TupleParametersBase():
    # 测试链式法则的实现是否正确
    p = hyper((), (), z**2)
    assert p.diff(z) == p*2*z


def test_hyper():
    raises(TypeError, lambda: hyper(1, 2, z))  # 测试在不支持的情况下是否会引发 TypeError 异常

    # 检查超几何函数的参数是否正确传递
    assert hyper((2, 1), (1,), z) == hyper(Tuple(1, 2), Tuple(1), z)
    assert hyper((2, 1, 2), (1, 2, 1, 3), z) == hyper((2,), (1, 3), z)

    # 测试 evaluate=False 时的超几何函数对象属性
    u = hyper((2, 1, 2), (1, 2, 1, 3), z, evaluate=False)
    assert u.ap == Tuple(1, 2, 2)
    assert u.bq == Tuple(1, 1, 2, 3)

    # 检查超几何函数对象的属性和特性
    h = hyper((1, 2), (3, 4, 5), z)
    assert h.ap == Tuple(1, 2)
    assert h.bq == Tuple(3, 4, 5)
    assert h.argument == z
    assert h.is_commutative is True
    h = hyper((2, 1), (4, 3, 5), z)
    assert h.ap == Tuple(1, 2)
    assert h.bq == Tuple(3, 4, 5)
    assert h.argument == z
    assert h.is_commutative is True

    # 确保所有参数被正确传递
    assert tn(hyper(Tuple(), Tuple(), z), exp(z), z)
    assert tn(z*hyper((1, 1), Tuple(2), -z), log(1 + z), z)

    # 对超几何函数进行导数测试
    h = hyper(
        (randcplx(), randcplx(), randcplx()), (randcplx(), randcplx()), z)
    assert td(h, z)

    a1, a2, b1, b2, b3 = symbols('a1:3, b1:4')
    # 检查超几何函数对 z 的导数计算是否正确
    assert hyper((a1, a2), (b1, b2, b3), z).diff(z) == \
        a1*a2/(b1*b2*b3) * hyper((a1 + 1, a2 + 1), (b1 + 1, b2 + 1, b3 + 1), z)

    # 不支持参数的导数计算
    assert hyper([z], [], z).diff(z) == Derivative(hyper([z], [], z), z)

    # 超几何函数对参数是无分支的
    from sympy.functions.elementary.complexes import polar_lift
    assert hyper([polar_lift(z)], [polar_lift(k)], polar_lift(x)) == \
        hyper([z], [k], polar_lift(x))

    # 超几何函数不会自动评估，但这里测试 evaluate 参数是否被正确接受
    assert hyper((1, 2), (1,), z, evaluate=False).func is hyper


def test_expand_func():
    # Gauss 超几何函数在 x=1 处的展开评估
    from sympy.abc import a, b, c
    from sympy.core.function import expand_func
    a1, b1, c1 = randcplx(), randcplx(), randcplx() + 5
    assert expand_func(hyper([a, b], [c], 1)) == \
        gamma(c)*gamma(-a - b + c)/(gamma(-a + c)*gamma(-b + c))
    # 对于超几何函数的扩展函数进行测试，验证其返回值与超几何函数本身的数值部分是否相近
    assert abs(expand_func(hyper([a1, b1], [c1], 1)).n()
               - hyper([a1, b1], [c1], 1).n()) < 1e-10

    # 测试空参数的超几何函数，期望其扩展后与指数函数 exp(z) 相等
    assert expand_func(hyper([], [], z)) == exp(z)

    # 测试带有参数的超几何函数，期望其扩展后保持不变
    assert expand_func(hyper([1, 2, 3], [], z)) == hyper([1, 2, 3], [], z)

    # 测试梅耶尔G函数的特定实例，期望其扩展后与 log(z + 1) 相等
    assert expand_func(meijerg([[1, 1], []], [[1], [0]], z)) == log(z + 1)

    # 测试另一个梅耶尔G函数的特定实例，期望其扩展后与原函数相等
    assert expand_func(meijerg([[1, 1], []], [[], []], z)) == \
        meijerg([[1, 1], []], [[], []], z)
# 定义一个函数，用于替换表达式中的虚拟符号为指定符号
def replace_dummy(expr, sym):
    # 导入需要的符号类
    from sympy.core.symbol import Dummy
    # 获取表达式中所有的虚拟符号
    dum = expr.atoms(Dummy)
    # 如果没有虚拟符号，则返回原表达式
    if not dum:
        return expr
    # 断言仅有一个虚拟符号
    assert len(dum) == 1
    # 将虚拟符号替换为指定的符号，并返回新表达式
    return expr.xreplace({dum.pop(): sym})


# 测试函数，用于验证超几何函数的重写求和表达式后的结果
def test_hyper_rewrite_sum():
    # 导入需要的符号和函数
    from sympy.concrete.summations import Sum
    from sympy.core.symbol import Dummy
    from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
    # 创建一个虚拟符号对象 _k
    _k = Dummy("k")
    # 验证超几何函数经过重写求和后的表达式是否符合预期
    assert replace_dummy(hyper((1, 2), (1, 3), x).rewrite(Sum), _k) == \
        Sum(x**_k / factorial(_k) * RisingFactorial(2, _k) /
            RisingFactorial(3, _k), (_k, 0, oo))

    # 验证不涉及重写求和的超几何函数保持不变
    assert hyper((1, 2, 3), (-1, 3), z).rewrite(Sum) == \
        hyper((1, 2, 3), (-1, 3), z)


# 测试函数，用于验证超几何函数的收敛半径
def test_radius_of_convergence():
    # 验证不同超几何函数的收敛半径是否符合预期
    assert hyper((1, 2), [3], z).radius_of_convergence == 1
    assert hyper((1, 2), [3, 4], z).radius_of_convergence is oo
    assert hyper((1, 2, 3), [4], z).radius_of_convergence == 0
    assert hyper((0, 1, 2), [4], z).radius_of_convergence is oo
    assert hyper((-1, 1, 2), [-4], z).radius_of_convergence == 0
    assert hyper((-1, -2, 2), [-1], z).radius_of_convergence is oo
    assert hyper((-1, 2), [-1, -2], z).radius_of_convergence == 0
    assert hyper([-1, 1, 3], [-2, 2], z).radius_of_convergence == 1
    assert hyper([-1, 1], [-2, 2], z).radius_of_convergence is oo
    assert hyper([-1, 1, 3], [-2], z).radius_of_convergence == 0
    assert hyper((-1, 2, 3, 4), [], z).radius_of_convergence is oo

    # 验证超几何函数的收敛性陈述是否正确
    assert hyper([1, 1], [3], 1).convergence_statement == True
    assert hyper([1, 1], [2], 1).convergence_statement == False
    assert hyper([1, 1], [2], -1).convergence_statement == True
    assert hyper([1, 1], [1], -1).convergence_statement == False


# 测试函数，用于验证梅杰尔函数的各项属性和方法
def test_meijer():
    # 验证梅杰尔函数的类型错误异常是否能正确抛出
    raises(TypeError, lambda: meijerg(1, z))
    raises(TypeError, lambda: meijerg(((1,), (2,)), (3,), (4,), z))

    # 验证梅杰尔函数通过简化参数后是否等价于指定形式
    assert meijerg(((1, 2), (3,)), ((4,), (5,)), z) == \
        meijerg(Tuple(1, 2), Tuple(3), Tuple(4), Tuple(5), z)

    # 创建一个复杂的梅杰尔函数对象 g，验证其各个属性是否正确设置
    g = meijerg((1, 2), (3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13, 14), z)
    assert g.an == Tuple(1, 2)
    assert g.ap == Tuple(1, 2, 3, 4, 5)
    assert g.aother == Tuple(3, 4, 5)
    assert g.bm == Tuple(6, 7, 8, 9)
    assert g.bq == Tuple(6, 7, 8, 9, 10, 11, 12, 13, 14)
    assert g.bother == Tuple(10, 11, 12, 13, 14)
    assert g.argument == z
    assert g.nu == 75
    assert g.delta == -1
    assert g.is_commutative is True
    assert g.is_number is False
    #issue 13071
    assert meijerg([[],[]], [[S.Half],[0]], 1).is_number is True

    # 验证特定条件下梅杰尔函数的 delta 参数是否正确设置为 S.Half
    assert meijerg([1, 2], [3], [4], [5], z).delta == S.Half

    # 验证一些具体参数下的梅杰尔函数是否符合预期
    # 确保所有参数都能正确传递到对应的位置
    assert tn(meijerg(Tuple(), Tuple(), Tuple(0), Tuple(), -z), exp(z), z)
    assert tn(sqrt(pi)*meijerg(Tuple(), Tuple(),
                               Tuple(0), Tuple(S.Half), z**2/4), cos(z), z)
    assert tn(meijerg(Tuple(1, 1), Tuple(), Tuple(1), Tuple(0), z),
              log(1 + z), z)

    # 测试异常情况
    # 调用 meijerg 函数，期望引发 ValueError 异常，参数为 (((3, 1), (2,)), ((oo,), (2, 0)), x)
    raises(ValueError, lambda: meijerg(((3, 1), (2,)), ((oo,), (2, 0)), x))
    # 调用 meijerg 函数，期望引发 ValueError 异常，参数为 (((3, 1), (2,)), ((1,), (2, 0)), x)
    raises(ValueError, lambda: meijerg(((3, 1), (2,)), ((1,), (2, 0)), x))

    # differentiation
    # 调用 meijerg 函数生成 g，参数为 ((randcplx(),), (randcplx() + 2*I,), Tuple(), (randcplx(), randcplx()), z)
    g = meijerg((randcplx(),), (randcplx() + 2*I,), Tuple(),
                (randcplx(), randcplx()), z)
    # 断言 td(g, z) 为真
    assert td(g, z)

    # 调用 meijerg 函数生成 g，参数为 (Tuple(), (randcplx(),), Tuple(), (randcplx(), randcplx()), z)
    g = meijerg(Tuple(), (randcplx(),), Tuple(),
                (randcplx(), randcplx()), z)
    # 断言 td(g, z) 为真
    assert td(g, z)

    # 调用 meijerg 函数生成 g，参数为 (Tuple(), Tuple(), Tuple(randcplx()), Tuple(randcplx(), randcplx()), z)
    g = meijerg(Tuple(), Tuple(), Tuple(randcplx()),
                Tuple(randcplx(), randcplx()), z)
    # 断言 td(g, z) 为真
    assert td(g, z)

    # 使用 symbols 函数定义符号变量 a1, a2, b1, b2, c1, c2, d1, d2
    a1, a2, b1, b2, c1, c2, d1, d2 = symbols('a1:3, b1:3, c1:3, d1:3')
    # 断言 meijerg 函数的 z 偏导数等于给定表达式
    assert meijerg((a1, a2), (b1, b2), (c1, c2), (d1, d2), z).diff(z) == \
        (meijerg((a1 - 1, a2), (b1, b2), (c1, c2), (d1, d2), z)
         + (a1 - 1)*meijerg((a1, a2), (b1, b2), (c1, c2), (d1, d2), z))/z

    # 断言 meijerg 函数对于参数 [z, z], [], [], [], z 的 z 偏导数等于其导数表达式
    assert meijerg([z, z], [], [], [], z).diff(z) == \
        Derivative(meijerg([z, z], [], [], [], z), z)

    # meijerg 函数对参数是参数是参数不变的
    from sympy.functions.elementary.complexes import polar_lift as pl
    # 断言 meijerg 函数使用 polar_lift 转换后的参数结果等于原始参数结果
    assert meijerg([pl(a1)], [pl(a2)], [pl(b1)], [pl(b2)], pl(z)) == \
        meijerg([a1], [a2], [b1], [b2], pl(z))

    # integrand
    # 从 sympy.abc 导入符号变量 a, b, c, d, s
    from sympy.abc import a, b, c, d, s
    # 断言 meijerg 函数的积分因子，对于给定的 s 参数等于给定的表达式
    assert meijerg([a], [b], [c], [d], z).integrand(s) == \
        z**s*gamma(c - s)*gamma(-a + s + 1)/(gamma(b - s)*gamma(-d + s + 1))
def test_meijerg_derivative():
    # 检查 Meijer G 函数的导数计算是否正确
    assert meijerg([], [1, 1], [0, 0, x], [], z).diff(x) == \
        log(z)*meijerg([], [1, 1], [0, 0, x], [], z) \
        + 2*meijerg([], [1, 1, 1], [0, 0, x, 0], [], z)

    y = randcplx()
    a = 5  # mpmath chokes with non-real numbers, and Mod1 with floats
    # 测试不同参数下 Meijer G 函数关于 x 的导数
    assert td(meijerg([x], [], [], [], y), x)
    assert td(meijerg([x**2], [], [], [], y), x)
    assert td(meijerg([], [x], [], [], y), x)
    assert td(meijerg([], [], [x], [], y), x)
    assert td(meijerg([], [], [], [x], y), x)
    assert td(meijerg([x], [a], [a + 1], [], y), x)
    assert td(meijerg([x], [a + 1], [a], [], y), x)
    assert td(meijerg([x, a], [], [], [a + 1], y), x)
    assert td(meijerg([x, a + 1], [], [], [a], y), x)
    b = Rational(3, 2)
    assert td(meijerg([a + 2], [b], [b - 3, x], [a], y), x)


def test_meijerg_period():
    # 测试 Meijer G 函数的周期性质
    assert meijerg([], [1], [0], [], x).get_period() == 2*pi
    assert meijerg([1], [], [], [0], x).get_period() == 2*pi
    assert meijerg([], [], [0], [], x).get_period() == 2*pi  # exp(x)
    assert meijerg(
        [], [], [0], [S.Half], x).get_period() == 2*pi  # cos(sqrt(x))
    assert meijerg(
        [], [], [S.Half], [0], x).get_period() == 4*pi  # sin(sqrt(x))
    assert meijerg([1, 1], [], [1], [0], x).get_period() is oo  # log(1 + x)


def test_hyper_unpolarify():
    from sympy.functions.elementary.exponential import exp_polar
    a = exp_polar(2*pi*I)*x
    b = x
    # 测试超几何函数的极坐标化简
    assert hyper([], [], a).argument == b
    assert hyper([0], [], a).argument == a
    assert hyper([0], [0], a).argument == b
    assert hyper([0, 1], [0], a).argument == a
    assert hyper([0, 1], [0], exp_polar(2*pi*I)).argument == 1


@slow
def test_hyperrep():
    from sympy.functions.special.hyper import (HyperRep, HyperRep_atanh,
        HyperRep_power1, HyperRep_power2, HyperRep_log1, HyperRep_asin1,
        HyperRep_asin2, HyperRep_sqrts1, HyperRep_sqrts2, HyperRep_log2,
        HyperRep_cosasin, HyperRep_sinasin)
    # First test the base class works.
    from sympy.functions.elementary.exponential import exp_polar
    from sympy.functions.elementary.piecewise import Piecewise
    a, b, c, d, z = symbols('a b c d z')

    class myrep(HyperRep):
        @classmethod
        def _expr_small(cls, x):
            return a

        @classmethod
        def _expr_small_minus(cls, x):
            return b

        @classmethod
        def _expr_big(cls, x, n):
            return c*n

        @classmethod
        def _expr_big_minus(cls, x, n):
            return d*n
    # 测试 HyperRep 类的重写方法
    assert myrep(z).rewrite('nonrep') == Piecewise((0, abs(z) > 1), (a, True))
    assert myrep(exp_polar(I*pi)*z).rewrite('nonrep') == \
        Piecewise((0, abs(z) > 1), (b, True))
    assert myrep(exp_polar(2*I*pi)*z).rewrite('nonrep') == \
        Piecewise((c, abs(z) > 1), (a, True))
    assert myrep(exp_polar(3*I*pi)*z).rewrite('nonrep') == \
        Piecewise((d, abs(z) > 1), (b, True))
    # 确认指定条件下，exp_polar(4*I*pi)*z 的非重复表示等于 Piecewise((2*c, abs(z) > 1), (a, True))
    assert myrep(exp_polar(4*I*pi)*z).rewrite('nonrep') == \
        Piecewise((2*c, abs(z) > 1), (a, True))

    # 确认指定条件下，exp_polar(5*I*pi)*z 的非重复表示等于 Piecewise((2*d, abs(z) > 1), (b, True))
    assert myrep(exp_polar(5*I*pi)*z).rewrite('nonrep') == \
        Piecewise((2*d, abs(z) > 1), (b, True))

    # 确认指定条件下，z 的非重复小表示等于 a
    assert myrep(z).rewrite('nonrepsmall') == a

    # 确认指定条件下，exp_polar(I*pi)*z 的非重复小表示等于 b
    assert myrep(exp_polar(I*pi)*z).rewrite('nonrepsmall') == b

    def t(func, hyp, z):
        """ 
        测试 func 是否是 hyp 的有效表示形式。
        
        首先测试 func 在小 z 条件下与 hyp 的一致性。
        """
        if not tn(func.rewrite('nonrepsmall'), hyp, z,
                  a=Rational(-1, 2), b=Rational(-1, 2), c=S.Half, d=S.Half):
            return False
        
        # 接下来检查两种小表示是否一致。
        if not tn(
            func.rewrite('nonrepsmall').subs(
                z, exp_polar(I*pi)*z).replace(exp_polar, exp),
            func.subs(z, exp_polar(I*pi)*z).rewrite('nonrepsmall'),
                z, a=Rational(-1, 2), b=Rational(-1, 2), c=S.Half, d=S.Half):
            return False
        
        # 接下来检查沿着 exp_polar(I*pi)*z 的表达式的连续性。
        expr = func.subs(z, exp_polar(I*pi)*z).rewrite('nonrep')
        if abs(expr.subs(z, 1 + 1e-15).n() - expr.subs(z, 1 - 1e-15).n()) > 1e-10:
            return False
        
        # 最后检查大表示的连续性。
        def dosubs(func, a, b):
            rv = func.subs(z, exp_polar(a)*z).rewrite('nonrep')
            return rv.subs(z, exp_polar(b)*z).replace(exp_polar, exp)
        
        for n in [0, 1, 2, 3, 4, -1, -2, -3, -4]:
            expr1 = dosubs(func, 2*I*pi*n, I*pi/2)
            expr2 = dosubs(func, 2*I*pi*n + I*pi, -I*pi/2)
            if not tn(expr1, expr2, z):
                return False
            expr1 = dosubs(func, 2*I*pi*(n + 1), -I*pi/2)
            expr2 = dosubs(func, 2*I*pi*n + I*pi, I*pi/2)
            if not tn(expr1, expr2, z):
                return False
        
        return True

    # 现在测试各种表示形式的有效性。
    a = Rational(1, 3)
    assert t(HyperRep_atanh(z), hyper([S.Half, 1], [Rational(3, 2)], z), z)
    assert t(HyperRep_power1(a, z), hyper([-a], [], z), z)
    assert t(HyperRep_power2(a, z), hyper([a, a - S.Half], [2*a], z), z)
    assert t(HyperRep_log1(z), -z*hyper([1, 1], [2], z), z)
    assert t(HyperRep_asin1(z), hyper([S.Half, S.Half], [Rational(3, 2)], z), z)
    assert t(HyperRep_asin2(z), hyper([1, 1], [Rational(3, 2)], z), z)
    assert t(HyperRep_sqrts1(a, z), hyper([-a, S.Half - a], [S.Half], z), z)
    assert t(HyperRep_sqrts2(a, z),
             -2*z/(2*a + 1)*hyper([-a - S.Half, -a], [S.Half], z).diff(z), z)
    assert t(HyperRep_log2(z), -z/4*hyper([Rational(3, 2), 1, 1], [2, 2], z), z)
    assert t(HyperRep_cosasin(a, z), hyper([-a, a], [S.Half], z), z)
    assert t(HyperRep_sinasin(a, z), 2*a*z*hyper([1 - a, 1 + a], [Rational(3, 2)], z), z)
@slow
# 定义一个名为 test_meijerg_eval 的测试函数，标记为慢速测试
def test_meijerg_eval():
    # 导入必要的模块和符号
    from sympy.functions.elementary.exponential import exp_polar
    from sympy.functions.special.bessel import besseli
    from sympy.abc import l
    # 生成一个随机复数
    a = randcplx()
    # 构造参数 arg
    arg = x*exp_polar(k*pi*I)
    # 计算第一个表达式 expr1
    expr1 = pi*meijerg([[], [(a + 1)/2]], [[a/2], [-a/2, (a + 1)/2]], arg**2/4)
    # 计算第二个表达式 expr2
    expr2 = besseli(a, arg)

    # 测试两个表达式在所有参数下是否一致
    for x_ in [0.5, 1.5]:
        for k_ in [0.0, 0.1, 0.3, 0.5, 0.8, 1, 5.751, 15.3]:
            # 断言两个表达式的数值之差小于给定精度
            assert abs((expr1 - expr2).n(subs={x: x_, k: k_})) < 1e-10
            assert abs((expr1 - expr2).n(subs={x: x_, k: -k_})) < 1e-10

    # 独立测试连续性
    eps = 1e-13
    expr2 = expr1.subs(k, l)
    for x_ in [0.5, 1.5]:
        for k_ in [0.5, Rational(1, 3), 0.25, 0.75, Rational(2, 3), 1.0, 1.5]:
            # 断言两个表达式在连续参数下的数值之差小于给定精度
            assert abs((expr1 - expr2).n(subs={x: x_, k: k_ + eps, l: k_ - eps})) < 1e-10
            assert abs((expr1 - expr2).n(subs={x: x_, k: -k_ + eps, l: -k_ - eps})) < 1e-10

    # 计算复合表达式 expr
    expr = (meijerg(((0.5,), ()), ((0.5, 0, 0.5), ()), exp_polar(-I*pi)/4)
            + meijerg(((0.5,), ()), ((0.5, 0, 0.5), ()), exp_polar(I*pi)/4)) \
        /(2*sqrt(pi))
    # 断言复合表达式与给定数值的差为零
    assert (expr - pi/exp(1)).n(chop=True) == 0


# 定义一个测试函数 test_limits
def test_limits():
    k, x = symbols('k, x')
    # 断言超几何函数的级数展开结果
    assert hyper((1,), (Rational(4, 3), Rational(5, 3)), k**2).series(k) == \
           1 + 9*k**2/20 + 81*k**4/1120 + O(k**6)  # issue 6350

    # 断言极限的计算结果
    assert limit(1/hyper((1, ), (1, ), x), x, 0) == 1


# 定义一个测试函数 test_appellf1
def test_appellf1():
    a, b1, b2, c, x, y = symbols('a b1 b2 c x y')
    # 断言 Appell F1 函数的对称性
    assert appellf1(a, b2, b1, c, y, x) == appellf1(a, b1, b2, c, x, y)
    assert appellf1(a, b1, b1, c, y, x) == appellf1(a, b1, b1, c, x, y)
    # 断言特定参数下 Appell F1 函数的值
    assert appellf1(a, b1, b2, c, S.Zero, S.Zero) is S.One

    # 验证 Appell F1 函数在延迟求值模式下的行为
    f = appellf1(a, b1, b2, c, S.Zero, S.Zero, evaluate=False)
    assert f.func is appellf1
    assert f.doit() is S.One


# 定义一个测试函数 test_derivative_appellf1
def test_derivative_appellf1():
    from sympy.core.function import diff
    a, b1, b2, c, x, y, z = symbols('a b1 b2 c x y z')
    # 验证 Appell F1 函数关于各个变量的偏导数
    assert diff(appellf1(a, b1, b2, c, x, y), x) == a*b1*appellf1(a + 1, b2, b1 + 1, c + 1, y, x)/c
    assert diff(appellf1(a, b1, b2, c, x, y), y) == a*b2*appellf1(a + 1, b1, b2 + 1, c + 1, x, y)/c
    assert diff(appellf1(a, b1, b2, c, x, y), z) == 0
    assert diff(appellf1(a, b1, b2, c, x, y), a) ==  Derivative(appellf1(a, b1, b2, c, x, y), a)


# 定义一个测试函数 test_eval_nseries
def test_eval_nseries():
    a1, b1, a2, b2 = symbols('a1 b1 a2 b2')
    # 断言超几何函数的数值级数展开
    assert hyper((1,2), (1,2,3), x**2)._eval_nseries(x, 7, None) == \
        1 + x**2/3 + x**4/24 + x**6/360 + O(x**7)
    # 断言指数函数的级数展开与超几何函数的关系
    assert exp(x)._eval_nseries(x,7,None) == \
        hyper((a1, b1), (a1, b1), x)._eval_nseries(x, 7, None)
    # 断言超几何函数的级数展开关于不同变量的情况
    assert hyper((a1, a2), (b1, b2), x)._eval_nseries(z, 7, None) ==\
        hyper((a1, a2), (b1, b2), x) + O(z**7)
    # 断言：使用超几何函数对表达式进行近似展开，并与预期值进行比较
    assert hyper((-S(1)/2, S(1)/2), (1,), 4*x/(x + 1)).nseries(x) == \
        1 - x + x**2/4 - 3*x**3/4 - 15*x**4/64 - 93*x**5/64 + O(x**6)
    
    # 断言：使用超几何函数对表达式进行近似展开，乘以 π/2，并与预期值进行比较
    assert (pi/2*hyper((-S(1)/2, S(1)/2), (1,), 4*x/(x + 1))).nseries(x) == \
        pi/2 - pi*x/2 + pi*x**2/8 - 3*pi*x**3/8 - 15*pi*x**4/128 - 93*pi*x**5/128 + O(x**6)
```