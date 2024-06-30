# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_meijerint.py`

```
# 导入 expand_func 函数，用于展开函数表达式
from sympy.core.function import expand_func
# 导入复数单位 I、有理数 Rational、无穷大 oo、圆周率 pi
from sympy.core.numbers import (I, Rational, oo, pi)
# 导入单例对象 S
from sympy.core.singleton import S
# 导入用于排序的默认排序键函数 default_sort_key
from sympy.core.sorting import default_sort_key
# 导入复数函数 Abs、arg、re、unpolarify
from sympy.functions.elementary.complexes import Abs, arg, re, unpolarify
# 导入指数函数 exp、exp_polar、log
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
# 导入双曲函数 cosh、acosh、sinh
from sympy.functions.elementary.hyperbolic import cosh, acosh, sinh
# 导入平方根函数 sqrt
from sympy.functions.elementary.miscellaneous import sqrt
# 导入分段函数 Piecewise、piecewise_fold
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
# 导入三角函数 cos、sin、sinc、asin
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, asin)
# 导入误差函数 erf、erfc
from sympy.functions.special.error_functions import (erf, erfc)
# 导入 Gamma 函数及其相关函数 gamma、polygamma
from sympy.functions.special.gamma_functions import (gamma, polygamma)
# 导入超函 hyper、meijerg
from sympy.functions.special.hyper import (hyper, meijerg)
# 导入积分相关函数 Integral、integrate
from sympy.integrals.integrals import (Integral, integrate)
# 导入超函数展开函数 hyperexpand
from sympy.simplify.hyperexpand import hyperexpand
# 导入简化函数 simplify
from sympy.simplify.simplify import simplify
# 导入梅杰尔积分相关函数 _rewrite_single、_rewrite1、meijerint_indefinite、_inflate_g、_create_lookup_table、meijerint_definite、meijerint_inversion
from sympy.integrals.meijerint import (_rewrite_single, _rewrite1,
    meijerint_indefinite, _inflate_g, _create_lookup_table,
    meijerint_definite, meijerint_inversion)
# 导入用于测试的慢速标记 slow
from sympy.testing.pytest import slow
# 导入数学符号 x、y、a、b、c、d、s、t、z
from sympy.abc import x, y, a, b, c, d, s, t, z


# 定义测试函数 test_rewrite_single
def test_rewrite_single():
    # 定义内部函数 t，用于测试单个 meijerg 函数重写
    def t(expr, c, m):
        # 对 meijerg 函数进行单个重写，并验证返回结果
        e = _rewrite_single(meijerg([a], [b], [c], [d], expr), x)
        assert e is not None
        assert isinstance(e[0][0][2], meijerg)
        assert e[0][0][2].argument.as_coeff_mul(x) == (c, (m,))
    
    # 定义内部函数 tn，用于测试无法重写的 meijerg 函数表达式
    def tn(expr):
        assert _rewrite_single(meijerg([a], [b], [c], [d], expr), x) is None
    
    # 调用 t 函数进行测试
    t(x, 1, x)
    t(x**2, 1, x**2)
    t(x**2 + y*x**2, y + 1, x**2)
    tn(x**2 + x)
    tn(x**y)
    
    # 定义内部函数 u，用于测试 meijerg 函数与指数函数 exp_polar 的数值验证
    def u(expr, x):
        from sympy.core.add import Add
        # 对表达式进行 meijerg 函数重写
        r = _rewrite_single(expr, x)
        # 替换表达式中的 exp_polar 为 exp，并进行数值验证
        e = Add(*[res[0]*res[2] for res in r[0]]).replace(
            exp_polar, exp)  # XXX Hack?
        assert verify_numerically(e, expr, x)
    
    # 调用 u 函数进行测试
    u(exp(-x)*sin(x), x)
    
    # 以下代码由于 hyperexpand 的变化，停止工作
    # 修复起来可能不值得
    #u(exp(-x)*sin(x)*cos(x), x)
    
    # 由于结果为 4*pi 的 g-函数，无法进行数值验证
    # 注意这也测试了反 Mellin 变换中的一个 bug
    # 以前会将 exp(4*pi*I*t) 转换成 exp(4*pi*I)**t 而不是 exp_polar
    #u(exp(x)*sin(x), x)
    assert _rewrite_single(exp(x)*sin(x), x) == \
        ([(-sqrt(2)/(2*sqrt(pi)), 0,
           meijerg(((Rational(-1, 2), 0, Rational(1, 4), S.Half, Rational(3, 4)), (1,)),
                   ((), (Rational(-1, 2), 0)), 64*exp_polar(-4*I*pi)/x**4))], True)


# 定义测试函数 test_rewrite1
def test_rewrite1():
    # 验证 _rewrite1 函数的返回结果
    assert _rewrite1(x**3*meijerg([a], [b], [c], [d], x**2 + y*x**2)*5, x) == \
        (5, x**3, [(1, 0, meijerg([a], [b], [c], [d], x**2*(y + 1)))], True)


# 定义测试函数 test_meijerint_indefinite_numerically，尚未实现
def test_meijerint_indefinite_numerically():
    pass
    # 定义一个函数 t，接受两个参数 fac 和 arg
    def t(fac, arg):
        # 使用 meijerg 函数生成一个广义超函数组合 g，乘以 fac
        g = meijerg([a], [b], [c], [d], arg)*fac
        # 随机生成复数并分别赋值给 a, b, c, d，创建一个替换字典 subs
        subs = {a: randcplx()/10, b: randcplx()/10 + I,
                c: randcplx(), d: randcplx()}
        # 对 g 在变量 x 上进行梅加函数不定积分
        integral = meijerint_indefinite(g, x)
        # 断言积分结果不为 None
        assert integral is not None
        # 断言通过数值验证 g 在 subs 变量替换下的值与积分的导数在 subs 变量替换下的值相等
        assert verify_numerically(g.subs(subs), integral.diff(x).subs(subs), x)
    
    # 调用 t 函数，分别传入不同的参数组合
    t(1, x)
    t(2, x)
    t(1, 2*x)
    t(1, x**2)
    t(5, x**S('3/2'))
    t(x**3, x)
    t(3*x**S('3/2'), 4*x**S('7/3'))
# 定义测试函数 test_meijerint_definite，用于测试 meijerint_definite 函数的输出
def test_meijerint_definite():
    # 调用 meijerint_definite 函数，并将结果分配给 v, b 变量
    v, b = meijerint_definite(x, x, 0, 0)
    # 断言 v 应为零且 b 应为 True
    assert v.is_zero and b is True
    # 再次调用 meijerint_definite 函数，并将结果分配给 v, b 变量
    v, b = meijerint_definite(x, x, oo, oo)
    # 断言 v 应为零且 b 应为 True
    assert v.is_zero and b is True


# 定义测试函数 test_inflate
def test_inflate():
    # 创建 subs 字典，包含符号变量和随机复数值
    subs = {a: randcplx()/10, b: randcplx()/10 + I, c: randcplx(),
            d: randcplx(), y: randcplx()/10}

    # 定义函数 t，接受四个参数，并引入 Mul 类和 _inflate_g 函数
    def t(a, b, arg, n):
        # 通过 meijerg 函数创建 m1 对象
        m1 = meijerg(a, b, arg)
        # 将 m1 对象通过 _inflate_g 函数扩展为 m2 对象
        m2 = Mul(*_inflate_g(m1, n))
        # NOTE: (the random number)**9 must still be on the principal sheet.
        # Thus make b&d small to create random numbers of small imaginary part.
        # 对 m1 和 m2 在 subs 变量替换后进行数值验证，并返回验证结果
        return verify_numerically(m1.subs(subs), m2.subs(subs), x, b=0.1, d=-0.1)
    
    # 断言调用 t 函数返回 True
    assert t([[a], [b]], [[c], [d]], x, 3)
    assert t([[a, y], [b]], [[c], [d]], x, 3)
    assert t([[a], [b]], [[c, y], [d]], 2*x**3, 3)


# 定义测试函数 test_recursive
def test_recursive():
    # 从 sympy.core.symbol 引入 symbols 函数，并声明 a, b, c 为正实数符号变量
    from sympy.core.symbol import symbols
    a, b, c = symbols('a b c', positive=True)
    # 创建 r 表达式，包含指数函数和符号变量 a, b
    r = exp(-(x - a)**2)*exp(-(x - b)**2)
    # 对 r 表达式进行积分，并使用 meijerg=True 选项
    e = integrate(r, (x, 0, oo), meijerg=True)
    # 断言简化后的 e 表达式等于给定表达式
    assert simplify(e.expand()) == (
        sqrt(2)*sqrt(pi)*(
        (erf(sqrt(2)*(a + b)/2) + 1)*exp(-a**2/2 + a*b - b**2/2))/4)
    
    # 创建包含指数函数和符号变量 a, b, c 的积分表达式
    e = integrate(exp(-(x - a)**2)*exp(-(x - b)**2)*exp(c*x), (x, 0, oo), meijerg=True)
    # 断言简化后的 e 表达式等于给定表达式
    assert simplify(e) == (
        sqrt(2)*sqrt(pi)*(erf(sqrt(2)*(2*a + 2*b + c)/4) + 1)*exp(-a**2 - b**2
        + (2*a + 2*b + c)**2/8)/4)
    
    # 对 exp(-(x - a - b - c)**2) 表达式进行积分，并使用 meijerg=True 选项
    assert simplify(integrate(exp(-(x - a - b - c)**2), (x, 0, oo), meijerg=True)) == \
        sqrt(pi)/2*(1 + erf(a + b + c))
    
    # 对 exp(-(x + a + b + c)**2) 表达式进行积分，并使用 meijerg=True 选项
    assert simplify(integrate(exp(-(x + a + b + c)**2), (x, 0, oo), meijerg=True)) == \
        sqrt(pi)/2*(1 - erf(a + b + c))


# 标记为慢速测试的 test_meijerint 函数
@slow
def test_meijerint():
    # 从 sympy.core.function 引入 expand 函数，从 sympy.core.symbol 引入 symbols 函数
    from sympy.core.function import expand
    from sympy.core.symbol import symbols
    s, t, mu = symbols('s t mu', real=True)
    # 断言积分表达式是否为 Piecewise 对象
    assert integrate(meijerg([], [], [0], [], s*t)
                     *meijerg([], [], [mu/2], [-mu/2], t**2/4),
                     (t, 0, oo)).is_Piecewise
    
    # 声明 s 符号变量为正实数
    s = symbols('s', positive=True)
    # 断言积分结果等于 gamma(s + 1)
    assert integrate(x**s*meijerg([[], []], [[0], []], x), (x, 0, oo)) == \
        gamma(s + 1)
    
    # 断言积分结果等于 gamma(s + 1)，并使用 meijerg=True 选项
    assert integrate(x**s*meijerg([[], []], [[0], []], x), (x, 0, oo),
                     meijerg=True) == gamma(s + 1)
    
    # 断言积分结果是 Integral 对象
    assert isinstance(integrate(x**s*meijerg([[], []], [[0], []], x),
                                (x, 0, oo), meijerg=False),
                      Integral)
    
    # 断言 meijerint_indefinite(exp(x), x) 等于 exp(x)
    assert meijerint_indefinite(exp(x), x) == exp(x)

    # TODO what simplifications should be done automatically?
    # This tests "extra case" for antecedents_1.
    a, b = symbols('a b', positive=True)
    # 断言简化后的 meijerint_definite(x**a, x, 0, b)[0] 等于 b**(a + 1)/(a + 1)
    assert simplify(meijerint_definite(x**a, x, 0, b)[0]) == \
        b**(a + 1)/(a + 1)

    # 断言 meijerint_definite((x + 1)**3*exp(-x), x, 0, oo) 结果为 (16, True)
    assert meijerint_definite((x + 1)**3*exp(-x), x, 0, oo) == (16, True)

    # Again, how about simplifications?
    sigma, mu = symbols('sigma mu', positive=True)
    # TODO: 继续在这里添加必要的注释
    i, c = meijerint_definite(exp(-((x - mu)/(2*sigma))**2), x, 0, oo)
    # 检查表达式是否简化为指定值
    assert simplify(i) == sqrt(pi)*sigma*(2 - erfc(mu/(2*sigma)))

    # 检查变量c是否为True
    assert c == True

    # 计算指定积分并进行检查
    i, _ = meijerint_definite(exp(-mu*x)*exp(sigma*x), x, 0, oo)
    # TODO it would be nice to test the condition
    assert simplify(i) == 1/(mu - sigma)

    # 测试替换限制的效果
    assert meijerint_definite(exp(x), x, -oo, 2) == (exp(2), True)
    # Note: causes a NaN in _check_antecedents
    assert expand(meijerint_definite(exp(x), x, 0, I)[0]) == exp(I) - 1
    assert expand(meijerint_definite(exp(-x), x, 0, x)[0]) == \
        1 - exp(-exp(I*arg(x))*abs(x))

    # 测试从负无穷到正无穷的积分
    assert meijerint_definite(exp(-x**2), x, -oo, oo) == (sqrt(pi), True)
    assert meijerint_definite(exp(-abs(x)), x, -oo, oo) == (2, True)
    assert meijerint_definite(exp(-(2*x - 3)**2), x, -oo, oo) == \
        (sqrt(pi)/2, True)
    assert meijerint_definite(exp(-abs(2*x - 3)), x, -oo, oo) == (1, True)
    assert meijerint_definite(exp(-((x - mu)/sigma)**2/2)/sqrt(2*pi*sigma**2),
                              x, -oo, oo) == (1, True)
    assert meijerint_definite(sinc(x)**2, x, -oo, oo) == (pi, True)

    # 测试两个 g 函数的额外条件之一
    assert meijerint_definite(exp(-x)*sin(x), x, 0, oo) == (S.Half, True)

    # 测试一个bug
    def res(n):
        return (1/(1 + x**2)).diff(x, n).subs(x, 1)*(-1)**n
    for n in range(6):
        assert integrate(exp(-x)*sin(x)*x**n, (x, 0, oo), meijerg=True) == \
            res(n)

    # 这个测试用于检查 trigexpand... 现在通过线性替换完成
    assert simplify(integrate(exp(-x)*sin(x + a), (x, 0, oo), meijerg=True)
                    ) == sqrt(2)*sin(a + pi/4)/2

    # 测试 prudnikov 的条件 14
    # （这实际上是 besselj*besselj 的变形，以防止其被识别为表格中的乘积。）
    a, b, s = symbols('a b s')
    assert meijerint_definite(meijerg([], [], [a/2], [-a/2], x/4)
                  *meijerg([], [], [b/2], [-b/2], x/4)*x**(s - 1), x, 0, oo
        ) == (
        (4*2**(2*s - 2)*gamma(-2*s + 1)*gamma(a/2 + b/2 + s)
         /(gamma(-a/2 + b/2 - s + 1)*gamma(a/2 - b/2 - s + 1)
           *gamma(a/2 + b/2 - s + 1)),
            (re(s) < 1) & (re(s) < S(1)/2) & (re(a)/2 + re(b)/2 + re(s) > 0)))

    # 测试一个bug
    assert integrate(sin(x**a)*sin(x**b), (x, 0, oo), meijerg=True) == \
        Integral(sin(x**a)*sin(x**b), (x, 0, oo))

    # 测试更好的 hyperexpand
    assert integrate(exp(-x**2)*log(x), (x, 0, oo), meijerg=True) == \
        (sqrt(pi)*polygamma(0, S.Half)/4).expand()

    # 测试 hyperexpand 的bug
    from sympy.functions.special.gamma_functions import lowergamma
    n = symbols('n', integer=True)
    assert simplify(integrate(exp(-x)*x**n, x, meijerg=True)) == \
        lowergamma(n + 1, x)

    # 测试带有参数 1/x 的bug
    alpha = symbols('alpha', positive=True)
    # 断言测试 meijerint_definite 函数的返回值是否等于预期值
    assert meijerint_definite((2 - x)**alpha*sin(alpha/x), x, 0, 2) == \
        (sqrt(pi)*alpha*gamma(alpha + 1)*meijerg(((), (alpha/2 + S.Half,
        alpha/2 + 1)), ((0, 0, S.Half), (Rational(-1, 2),)), alpha**2/16)/4, True)

    # 测试与 3016 相关的一个 bug
    # 定义符号变量 a 和 s，且它们均为正数
    a, s = symbols('a s', positive=True)
    # 断言简化后的积分结果是否等于预期值
    assert simplify(integrate(x**s*exp(-a*x**2), (x, -oo, oo))) == \
        a**(-s/2 - S.Half)*((-1)**s + 1)*gamma(s/2 + S.Half)/2
# 定义一个函数用于测试贝塞尔函数的性质
def test_bessel():
    # 从 sympy 库中导入贝塞尔函数 besseli 和 besselj
    from sympy.functions.special.bessel import (besseli, besselj)
    
    # 断言贝塞尔函数积分结果简化后应该等于特定的数学表达式
    assert simplify(integrate(besselj(a, z)*besselj(b, z)/z, (z, 0, oo),
                     meijerg=True, conds='none')) == \
        2*sin(pi*(a/2 - b/2))/(pi*(a - b)*(a + b))
    
    # 断言贝塞尔函数积分结果简化后应该等于特定的数学表达式
    assert simplify(integrate(besselj(a, z)*besselj(a, z)/z, (z, 0, oo),
                     meijerg=True, conds='none')) == 1/(2*a)

    # TODO 更多正交积分的实现

    # 断言一个特定积分结果与贝塞尔函数特定形式的关系成立
    assert simplify(integrate(sin(z*x)*(x**2 - 1)**(-(y + S.Half)),
                              (x, 1, oo), meijerg=True, conds='none')
                    *2/((z/2)**y*sqrt(pi)*gamma(S.Half - y))) == \
        besselj(y, z)

    # Werner Rosenheinrich 的注释
    # SOME INDEFINITE INTEGRALS OF BESSEL FUNCTIONS

    # 断言对贝塞尔函数进行积分的结果符合预期
    assert integrate(x*besselj(0, x), x, meijerg=True) == x*besselj(1, x)
    assert integrate(x*besseli(0, x), x, meijerg=True) == x*besseli(1, x)
    # TODO 可以进行更高阶的幂次积分，但是否应该减少到阶数 0 和 1？
    assert integrate(besselj(1, x), x, meijerg=True) == -besselj(0, x)
    assert integrate(besselj(1, x)**2/x, x, meijerg=True) == \
        -(besselj(0, x)**2 + besselj(1, x)**2)/2
    # TODO 更多的贝塞尔函数积分，当表格扩展或递归 Melin 函数实现时

    # 断言贝塞尔函数积分结果符合预期
    assert integrate(besselj(0, x)**2/x**2, x, meijerg=True) == \
        -2*x*besselj(0, x)**2 - 2*x*besselj(1, x)**2 \
        + 2*besselj(0, x)*besselj(1, x) - besselj(0, x)**2/x
    assert integrate(besselj(0, x)*besselj(1, x), x, meijerg=True) == \
        -besselj(0, x)**2/2
    assert integrate(x**2*besselj(0, x)*besselj(1, x), x, meijerg=True) == \
        x**2*besselj(1, x)**2/2
    assert integrate(besselj(0, x)*besselj(1, x)/x, x, meijerg=True) == \
        (x*besselj(0, x)**2 + x*besselj(1, x)**2 -
            besselj(0, x)*besselj(1, x))
    # TODO besselj(0, a*x)*besselj(0, b*x) 的工作原理？
    # TODO besselj(0, x)**2*besselj(1, x)**2 的工作原理？
    # TODO sin(x)*besselj(0, x) 等的实现问题
    # TODO 是否可以完成 x*log(x)*besselj(0, x)？
    # TODO besselj(1, x)*besselj(0, x+a) 的工作原理？
    # TODO 更多的不定积分，当 Struve 函数等被实现时

    # 测试一个替换的结果
    assert integrate(besselj(1, x**2)*x, x, meijerg=True) == \
        -besselj(0, x**2)/2


# 定义一个函数用于测试反演函数的条件输出
def test_inversion():
    # 从 sympy 库中导入贝塞尔函数 besselj 和 Heaviside 函数
    from sympy.functions.special.bessel import besselj
    from sympy.functions.special.delta_functions import Heaviside

    # 定义一个函数用于计算反演
    def inv(f):
        return piecewise_fold(meijerint_inversion(f, s, t))
    
    # 断言特定函数的反演结果应该等于预期的 Heaviside 和三角函数的乘积
    assert inv(1/(s**2 + 1)) == sin(t)*Heaviside(t)
    assert inv(s/(s**2 + 1)) == cos(t)*Heaviside(t)
    assert inv(exp(-s)/s) == Heaviside(t - 1)
    assert inv(1/sqrt(1 + s**2)) == besselj(0, t)*Heaviside(t)

    # 测试一些先前检查的前提条件
    assert meijerint_inversion(sqrt(s)/sqrt(1 + s**2), s, t) is None
    assert inv(exp(s**2)) is None
    assert meijerint_inversion(exp(-s**2), s, t) is None
    # 导入符号和逆拉普拉斯变换相关的类和函数
    from sympy.core.symbol import Symbol
    from sympy.integrals.transforms import InverseLaplaceTransform
    
    # 创建一个正数符号对象 a
    a = Symbol('a', positive=True)
    # 定义一个复杂函数表达式 F
    F = sqrt(pi/a)*exp(-2*sqrt(a)*sqrt(s))
    # 对 F 进行逆梅尔函数逆变换得到 f
    f = meijerint_inversion(F, s, t)
    # 断言 f 不是分段函数
    assert not f.is_Piecewise
    
    # 创建一个实数符号对象 b
    b = Symbol('b', real=True)
    # 在 F 中用 b 替换符号 a
    F = F.subs(a, b)
    # 对替换后的 F 进行逆梅尔函数逆变换得到 f2
    f2 = meijerint_inversion(F, s, t)
    # 断言 f2 是分段函数
    assert f2.is_Piecewise
    # 断言 f2 的第一个分段与 f 相同
    assert f2.args[0][0] == f.subs(a, b)
    # 断言 f2 的最后一个分段是一个未评估的变换
    assert f2.args[-1][1]
    # 创建逆拉普拉斯变换对象 ILT
    ILT = InverseLaplaceTransform(F, s, t, None)
    # 断言 f2 的最后一个分段是 ILT 或者 ILT 的积分形式
    assert f2.args[-1][0] == ILT or f2.args[-1][0] == ILT.as_integral
# 测试函数：检查指数函数在特定情况下的反演性质
def test_inversion_exp_real_nonreal_shift():
    # 导入符号操作相关库
    from sympy.core.symbol import Symbol
    # 导入 delta 函数
    from sympy.functions.special.delta_functions import DiracDelta
    # 定义实数符号 r
    r = Symbol('r', real=True)
    # 定义非扩展实数符号 c
    c = Symbol('c', extended_real=False)
    # 定义复数 a
    a = 1 + 2*I
    # 定义符号 z
    z = Symbol('z')
    # 断言：当 r 是实数时，指数函数的反演结果不是分段函数
    assert not meijerint_inversion(exp(r*s), s, t).is_Piecewise
    # 断言：指数函数 exp(a*s) 的反演结果为 None
    assert meijerint_inversion(exp(a*s), s, t) is None
    # 断言：指数函数 exp(c*s) 的反演结果为 None
    assert meijerint_inversion(exp(c*s), s, t) is None
    # 获取指数函数 exp(z*s) 的反演结果
    f = meijerint_inversion(exp(z*s), s, t)
    # 断言：指数函数 exp(z*s) 的反演结果是分段函数
    assert f.is_Piecewise
    # 断言：反演结果的第一个参数是 DiracDelta 函数
    assert isinstance(f.args[0][0], DiracDelta)


# 慢速测试：查找表测试
@slow
def test_lookup_table():
    # 导入随机数生成相关库
    from sympy.core.random import uniform, randrange
    # 导入加法相关库
    from sympy.core.add import Add
    # 导入 meijerint 模块的 z 符号
    from sympy.integrals.meijerint import z as z_dummy
    # 创建空的查找表
    table = {}
    # 创建查找表
    _create_lookup_table(table)
    # 遍历查找表中的所有值
    for l in table.values():
        # 遍历排序后的公式、项、条件和提示
        for formula, terms, cond, hint in sorted(l, key=default_sort_key):
            # 创建替代字典
            subs = {}
            # 遍历公式中的自由符号和 z_dummy
            for ai in list(formula.free_symbols) + [z_dummy]:
                # 如果 ai 有属性且属性不为空
                if hasattr(ai, 'properties') and ai.properties:
                    # 使用随机数生成 1 到 10 之间的正整数
                    subs[ai] = randrange(1, 10)
                else:
                    # 否则生成 1.5 到 2.0 之间的均匀分布随机数
                    subs[ai] = uniform(1.5, 2.0)
            # 如果 terms 不是列表，则用 subs 计算 terms
            if not isinstance(terms, list):
                terms = terms(subs)

            # 测试超级展开是否能完成此操作
            expanded = [hyperexpand(g) for (_, g) in terms]
            # 断言：所有展开后的结果要么是分段函数，要么不包含 meijerg 函数
            assert all(x.is_Piecewise or not x.has(meijerg) for x in expanded)

            # 测试 meijer g 函数的结果是否如预期
            expanded = Add(*[f*x for (f, x) in terms])
            a, b = formula.n(subs=subs), expanded.n(subs=subs)
            r = min(abs(a), abs(b))
            if r < 1:
                assert abs(a - b).n() <= 1e-10
            else:
                assert (abs(a - b)/r).n() <= 1e-10


# 测试分支 bug
def test_branch_bug():
    # 导入 gamma 函数相关库
    from sympy.functions.special.gamma_functions import lowergamma
    # 导入幂简化相关库
    from sympy.simplify.powsimp import powdenest
    # 断言：利用 erf(x**3) 积分后的导数在极坐标下等于特定值
    assert powdenest(integrate(erf(x**3), x, meijerg=True).diff(x),
           polar=True) == 2*erf(x**3)*gamma(Rational(2, 3))/3/gamma(Rational(5, 3))
    # 断言：利用 erf(x**3) 积分的结果
    assert integrate(erf(x**3), x, meijerg=True) == \
        2*x*erf(x**3)*gamma(Rational(2, 3))/(3*gamma(Rational(5, 3))) \
        - 2*gamma(Rational(2, 3))*lowergamma(Rational(2, 3), x**6)/(3*sqrt(pi)*gamma(Rational(5, 3)))


# 测试线性替换
def test_linear_subs():
    # 导入贝塞尔函数相关库
    from sympy.functions.special.bessel import besselj
    # 断言：对 sin(x - 1) 的积分使用 meijerg=True 选项
    assert integrate(sin(x - 1), x, meijerg=True) == -cos(1 - x)
    # 断言：对 besselj(1, x - 1) 的积分使用 meijerg=True 选项
    assert integrate(besselj(1, x - 1), x, meijerg=True) == -besselj(0, 1 - x)


# 慢速测试：概率相关积分
@slow
def test_probability():
    # 导入函数展开相关库
    from sympy.core.function import expand_mul
    # 导入符号操作相关库
    from sympy.core.symbol import (Symbol, symbols)
    # 导入 gamma 函数简化相关库
    from sympy.simplify.gammasimp import gammasimp
    # 导入幂简化相关库
    from sympy.simplify.powsimp import powsimp
    # 定义非零符号 mu1 和 mu2
    mu1, mu2 = symbols('mu1 mu2', nonzero=True)
    # 定义两个正数符号变量 sigma1 和 sigma2
    sigma1, sigma2 = symbols('sigma1 sigma2', positive=True)
    # 定义一个正数符号变量 rate
    rate = Symbol('lambda', positive=True)

    # 定义正态分布的概率密度函数
    def normal(x, mu, sigma):
        return 1/sqrt(2*pi*sigma**2)*exp(-(x - mu)**2/2/sigma**2)

    # 定义指数分布的概率密度函数
    def exponential(x, rate):
        return rate*exp(-rate*x)

    # 断言：正态分布的概率密度函数在整个实数轴上的积分为1
    assert integrate(normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) == 1
    # 断言：正态分布乘以 x 在整个实数轴上的积分为其均值 mu1
    assert integrate(x*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) == mu1
    # 断言：正态分布乘以 x^2 在整个实数轴上的积分为其均值的平方加上方差
    assert integrate(x**2*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) == mu1**2 + sigma1**2
    # 断言：正态分布乘以 x^3 在整个实数轴上的积分为其均值的立方加上三倍均值乘以方差的平方
    assert integrate(x**3*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) == mu1**3 + 3*mu1*sigma1**2
    # 断言：两个正态分布的乘积在整个二维实数平面上的积分为1
    assert integrate(normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == 1
    # 断言：正态分布乘以 x 和另一个正态分布在整个二维实数平面上的积分为第一个正态分布的均值 mu1
    assert integrate(x*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == mu1
    # 断言：正态分布乘以 y 和另一个正态分布在整个二维实数平面上的积分为第二个正态分布的均值 mu2
    assert integrate(y*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == mu2
    # 断言：正态分布乘以 x 和 y 和另一个正态分布在整个二维实数平面上的积分为两个正态分布的均值的乘积 mu1*mu2
    assert integrate(x*y*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == mu1*mu2
    # 断言：正态分布乘以 (x + y + 1) 在整个二维实数平面上的积分为 1 + 第一个正态分布的均值 mu1 + 第二个正态分布的均值 mu2
    assert integrate((x + y + 1)*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == 1 + mu1 + mu2
    # 断言：正态分布乘以 (x + y - 1) 在整个二维实数平面上的积分为 -1 + 第一个正态分布的均值 mu1 + 第二个正态分布的均值 mu2
    assert integrate((x + y - 1)*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == -1 + mu1 + mu2

    # 计算正态分布乘以 x^2 和另一个正态分布在整个二维实数平面上的积分
    i = integrate(x**2*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                  (x, -oo, oo), (y, -oo, oo), meijerg=True)
    # 断言：积分结果中不包含绝对值函数 Abs
    assert not i.has(Abs)
    # 断言：简化积分结果并与 mu1^2 + sigma1^2 进行比较
    assert simplify(i) == mu1**2 + sigma1**2
    # 断言：正态分布乘以 y^2 和另一个正态分布在整个二维实数平面上的积分为第二个正态分布的方差 sigma2^2 加上均值的平方 mu2^2
    assert integrate(y**2*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == sigma2**2 + mu2**2

    # 断言：指数分布在从 0 到正无穷的积分为 1
    assert integrate(exponential(x, rate), (x, 0, oo), meijerg=True) == 1
    # 断言：指数分布乘以 x 在从 0 到正无穷的积分为 1/rate
    assert integrate(x*exponential(x, rate), (x, 0, oo), meijerg=True) == 1/rate
    # 断言：指数分布乘以 x^2 在从 0 到正无穷的积分为 2/rate^2
    assert integrate(x**2*exponential(x, rate), (x, 0, oo), meijerg=True) == 2/rate**2

    # 定义期望函数 E，计算给定表达式的期望
    def E(expr):
        # 计算表达式乘以指数分布和正态分布在一定范围内的积分（x: 0 到正无穷，y: 全实数）
        res1 = integrate(expr*exponential(x, rate)*normal(y, mu1, sigma1),
                         (x, 0, oo), (y, -oo, oo), meijerg=True)
        # 计算表达式乘以指数分布和正态分布在一定范围内的积分（y: 全实数，x: 0 到正无穷）
        res2 = integrate(expr*exponential(x, rate)*normal(y, mu1, sigma1),
                        (y, -oo, oo), (x, 0, oo), meijerg=True)
        # 断言：两种积分结果在展开后相等
        assert expand_mul(res1) == expand_mul(res2)
        # 返回第一种积分结果
        return res1

    # 断言：期望函数 E 对于常数表达式 1 的结果为 1
    assert E(1) == 1
    # 断言：期望函数 E 对于表达式 x*y 的结果为 mu1/rate
    assert E(x*y) == mu1/rate
    # 断言：期望函数 E 对于表达式 x*y^2 的结果为 mu1^2/rate + sigma1^2/rate
    assert E(x*y**2) == mu1**2/rate + sigma1**2/rate
    # 计算方差的期望，即 sigma1^2 + 1/rate^2
    ans = sigma1**2 + 1/rate**2
    # 断言：简化期望函数 E 对于表达式 (
    # 计算 Beta 分布的概率密度函数值
    betadist = x**(alpha - 1)*(1 + x)**(-alpha - beta)*gamma(alpha + beta) \
        /gamma(alpha)/gamma(beta)
    # 确保 Beta 分布在全区间的积分等于 1
    assert integrate(betadist, (x, 0, oo), meijerg=True) == 1
    # 计算 Beta 分布的一阶矩
    i = integrate(x*betadist, (x, 0, oo), meijerg=True, conds='separate')
    # 确保一阶矩的简化结果符合预期
    assert (gammasimp(i[0]), i[1]) == (alpha/(beta - 1), 1 < beta)
    # 计算 Beta 分布的二阶矩
    j = integrate(x**2*betadist, (x, 0, oo), meijerg=True, conds='separate')
    # 确保二阶矩的条件部分符合预期
    assert j[1] == (beta > 2)
    # 确保 Beta 分布的方差计算正确
    assert gammasimp(j[0] - i[0]**2) == (alpha + beta - 1)*alpha \
        /(beta - 2)/(beta - 1)**2

    # Beta 分布
    # 注意：这里使用不定积分来计算 Beta 分布，同时也测试 meijerint_indefinite 返回的最简结果
    a, b = symbols('a b', positive=True)
    betadist = x**(a - 1)*(-x + 1)**(b - 1)*gamma(a + b)/(gamma(a)*gamma(b))
    # 确保 Beta 分布在区间 [0, 1] 的积分等于 1
    assert simplify(integrate(betadist, (x, 0, 1), meijerg=True)) == 1
    # 确保 Beta 分布在区间 [0, 1] 的一阶矩计算正确
    assert simplify(integrate(x*betadist, (x, 0, 1), meijerg=True)) == \
        a/(a + b)
    # 确保 Beta 分布在区间 [0, 1] 的二阶矩计算正确
    assert simplify(integrate(x**2*betadist, (x, 0, 1), meijerg=True)) == \
        a*(a + 1)/(a + b)/(a + b + 1)
    # 确保 Beta 分布在区间 [0, 1] 的一般幂次的积分计算正确
    assert simplify(integrate(x**y*betadist, (x, 0, 1), meijerg=True)) == \
        gamma(a + b)*gamma(a + y)/gamma(a)/gamma(a + b + y)

    # Chi 分布
    k = Symbol('k', integer=True, positive=True)
    chi = 2**(1 - k/2)*x**(k - 1)*exp(-x**2/2)/gamma(k/2)
    # 确保 Chi 分布在全区间的积分等于 1
    assert powsimp(integrate(chi, (x, 0, oo), meijerg=True)) == 1
    # 确保 Chi 分布在全区间的一阶矩计算正确
    assert simplify(integrate(x*chi, (x, 0, oo), meijerg=True)) == \
        sqrt(2)*gamma((k + 1)/2)/gamma(k/2)
    # 确保 Chi 分布在全区间的二阶矩计算正确
    assert simplify(integrate(x**2*chi, (x, 0, oo), meijerg=True)) == k

    # Chi^2 分布
    chisquared = 2**(-k/2)/gamma(k/2)*x**(k/2 - 1)*exp(-x/2)
    # 确保 Chi^2 分布在全区间的积分等于 1
    assert powsimp(integrate(chisquared, (x, 0, oo), meijerg=True)) == 1
    # 确保 Chi^2 分布在全区间的一阶矩计算正确
    assert simplify(integrate(x*chisquared, (x, 0, oo), meijerg=True)) == k
    # 确保 Chi^2 分布在全区间的二阶矩计算正确
    assert simplify(integrate(x**2*chisquared, (x, 0, oo), meijerg=True)) == \
        k*(k + 2)
    # 确保 Chi^2 分布在全区间的三阶矩计算正确
    assert gammasimp(integrate(((x - k)/sqrt(2*k))**3*chisquared, (x, 0, oo),
                    meijerg=True)) == 2*sqrt(2)/sqrt(k)

    # Dagum 分布
    a, b, p = symbols('a b p', positive=True)
    # XXX (x/b)**a does not work
    dagum = a*p/x*(x/b)**(a*p)/(1 + x**a/b**a)**(p + 1)
    # 确保 Dagum 分布在全区间的积分等于 1
    assert simplify(integrate(dagum, (x, 0, oo), meijerg=True)) == 1
    # XXX conditions are a mess
    # 计算 Dagum 分布的一阶矩
    arg = x*dagum
    # 确保 Dagum 分布的一阶矩计算正确
    assert simplify(integrate(arg, (x, 0, oo), meijerg=True, conds='none')
                    ) == a*b*gamma(1 - 1/a)*gamma(p + 1 + 1/a)/(
                    (a*p + 1)*gamma(p))
    # 确保 Dagum 分布的二阶矩计算正确
    assert simplify(integrate(x*arg, (x, 0, oo), meijerg=True, conds='none')
                    ) == a*b**2*gamma(1 - 2/a)*gamma(p + 1 + 2/a)/(
                    (a*p + 2)*gamma(p))

    # F 分布
    d1, d2 = symbols('d1 d2', positive=True)
    f = sqrt(((d1*x)**d1 * d2**d2)/(d1*x + d2)**(d1 + d2))/x \
        /gamma(d1/2)/gamma(d2/2)*gamma((d1 + d2)/2)
    # 确保 F 分布在全区间的积分等于 1
    assert simplify(integrate(f, (x, 0, oo), meijerg=True)) == 1
    # TODO conditions are a mess
    # 确定第一个积分结果是否等于 d2/(d2 - 2)，使用 MeijerG 函数和无条件
    assert simplify(integrate(x*f, (x, 0, oo), meijerg=True, conds='none')
                    ) == d2/(d2 - 2)
    
    # 确定第二个积分结果是否等于 d2**2*(d1 + 2)/d1/(d2 - 4)/(d2 - 2)，使用 MeijerG 函数和无条件
    assert simplify(integrate(x**2*f, (x, 0, oo), meijerg=True, conds='none')
                    ) == d2**2*(d1 + 2)/d1/(d2 - 4)/(d2 - 2)

    # TODO: gamma, rayleigh

    # 逆高斯分布的定义
    lamda, mu = symbols('lamda mu', positive=True)
    dist = sqrt(lamda/2/pi)*x**(Rational(-3, 2))*exp(-lamda*(x - mu)**2/x/2/mu**2)
    # 定义简化函数 mysimp，并确保逆高斯分布的积分结果为1
    mysimp = lambda expr: simplify(expr.rewrite(exp))
    assert mysimp(integrate(dist, (x, 0, oo))) == 1
    # 确定 x*逆高斯分布的积分结果是否等于 mu
    assert mysimp(integrate(x*dist, (x, 0, oo))) == mu
    # 确定 (x - mu)**2*逆高斯分布的积分结果是否等于 mu**3/lamda
    assert mysimp(integrate((x - mu)**2*dist, (x, 0, oo))) == mu**3/lamda
    # 确定 (x - mu)**3*逆高斯分布的积分结果是否等于 3*mu**5/lamda**2
    assert mysimp(integrate((x - mu)**3*dist, (x, 0, oo))) == 3*mu**5/lamda**2

    # Levi 分布的定义
    c = Symbol('c', positive=True)
    assert integrate(sqrt(c/2/pi)*exp(-c/2/(x - mu))/(x - mu)**S('3/2'),
                    (x, mu, oo)) == 1
    # oo 意味着高阶矩

    # 对数-逻辑分布的定义
    alpha, beta = symbols('alpha beta', positive=True)
    distn = (beta/alpha)*x**(beta - 1)/alpha**(beta - 1)/ \
        (1 + x**beta/alpha**beta)**2
    # FIXME: 如果 alpha, beta 未声明为有限值，则下面的代码会hang
    # 在 https://github.com/sympy/sympy/pull/16603 中的更改后
    assert simplify(integrate(distn, (x, 0, oo))) == 1
    # 注意条件混乱，但正确表明 beta > 1
    assert simplify(integrate(x*distn, (x, 0, oo), conds='none')) == \
        pi*alpha/beta/sin(pi/beta)
    # （对条件类似的评论适用）
    assert simplify(integrate(x**y*distn, (x, 0, oo), conds='none')) == \
        pi*alpha**y*y/beta/sin(pi*y/beta)

    # Weibull 分布的定义
    k = Symbol('k', positive=True)
    n = Symbol('n', positive=True)
    distn = k/lamda*(x/lamda)**(k - 1)*exp(-(x/lamda)**k)
    assert simplify(integrate(distn, (x, 0, oo))) == 1
    assert simplify(integrate(x**n*distn, (x, 0, oo))) == \
        lamda**n*gamma(1 + n/k)

    # Rice 分布的定义
    from sympy.functions.special.bessel import besseli
    nu, sigma = symbols('nu sigma', positive=True)
    rice = x/sigma**2*exp(-(x**2 + nu**2)/2/sigma**2)*besseli(0, x*nu/sigma**2)
    assert integrate(rice, (x, 0, oo), meijerg=True) == 1
    # 谁能验证更高的矩？

    # 拉普拉斯分布的定义
    mu = Symbol('mu', real=True)
    b = Symbol('b', positive=True)
    laplace = exp(-abs(x - mu)/b)/2/b
    assert integrate(laplace, (x, -oo, oo), meijerg=True) == 1
    assert integrate(x*laplace, (x, -oo, oo), meijerg=True) == mu
    assert integrate(x**2*laplace, (x, -oo, oo), meijerg=True) == \
        2*b**2 + mu**2

    # TODO: 是否还支持其他在 (-oo, oo) 上的分布？

    # 杂项测试
    k = Symbol('k', positive=True)
    assert gammasimp(expand_mul(integrate(log(x)*x**(k - 1)*exp(-x)/gamma(k),
                              (x, 0, oo)))) == polygamma(0, k)
@slow
def test_expint():
    """ Test various exponential integrals. """
    # 导入所需的符号和函数模块
    from sympy.core.symbol import Symbol
    from sympy.functions.elementary.hyperbolic import sinh
    from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, expint)
    
    # 断言积分结果与指定的指数积分函数值相等
    assert simplify(unpolarify(integrate(exp(-z*x)/x**y, (x, 1, oo),
                meijerg=True, conds='none'
                ).rewrite(expint).expand(func=True))) == expint(y, z)

    # 断言对指数衰减函数的积分结果等于特定的指数积分函数
    assert integrate(exp(-z*x)/x, (x, 1, oo), meijerg=True,
                     conds='none').rewrite(expint).expand() == \
        expint(1, z)
    
    # 断言对指数衰减函数的平方的积分结果等于对应的指数积分函数
    assert integrate(exp(-z*x)/x**2, (x, 1, oo), meijerg=True,
                     conds='none').rewrite(expint).expand() == \
        expint(2, z).rewrite(Ei).rewrite(expint)
    
    # 断言对指数衰减函数的立方的积分结果等于对应的指数积分函数
    assert integrate(exp(-z*x)/x**3, (x, 1, oo), meijerg=True,
                     conds='none').rewrite(expint).expand() == \
        expint(3, z).rewrite(Ei).rewrite(expint).expand()

    # 定义正数符号变量 t
    t = Symbol('t', positive=True)
    
    # 断言余弦函数除以自变量的积分结果等于柯西函数 Ci(t)
    assert integrate(-cos(x)/x, (x, t, oo), meijerg=True).expand() == Ci(t)
    
    # 断言正弦函数除以自变量的积分结果等于正弦积分函数 Si(t) 减去 pi/2
    assert integrate(-sin(x)/x, (x, t, oo), meijerg=True).expand() == \
        Si(t) - pi/2
    
    # 断言正弦函数除以自变量的积分结果等于正弦积分函数 Si(z)
    assert integrate(sin(x)/x, (x, 0, z), meijerg=True) == Si(z)
    
    # 断言双曲正弦函数除以自变量的积分结果等于双曲正弦积分函数 Shi(z)
    assert integrate(sinh(x)/x, (x, 0, z), meijerg=True) == Shi(z)
    
    # 断言指数函数除以自变量的积分结果等于 -i*pi 减去指数积分函数 Ei(x)
    assert integrate(exp(-x)/x, x, meijerg=True).expand().rewrite(expint) == \
        I*pi - expint(1, x)
    
    # 断言指数函数除以自变量的平方的积分结果等于指数积分函数减去其他项
    assert integrate(exp(-x)/x**2, x, meijerg=True).rewrite(expint).expand() \
        == expint(1, x) - exp(-x)/x - I*pi

    # 定义极坐标变量 u
    u = Symbol('u', polar=True)
    
    # 断言余弦函数除以自变量的积分结果的独立部分等于柯西函数 Ci(u)
    assert integrate(cos(u)/u, u, meijerg=True).expand().as_independent(u)[1] \
        == Ci(u)
    
    # 断言双曲余弦函数除以自变量的积分结果的独立部分等于科尔曼函数 Chi(u)
    assert integrate(cosh(u)/u, u, meijerg=True).expand().as_independent(u)[1] \
        == Chi(u)

    # 断言对指数积分函数的积分结果等于 x*expint(1, x) 减去 exp(-x)
    assert integrate(expint(1, x), x, meijerg=True
            ).rewrite(expint).expand() == x*expint(1, x) - exp(-x)
    
    # 断言对指数积分函数的平方的积分结果
    assert integrate(expint(2, x), x, meijerg=True
            ).rewrite(expint).expand() == \
        -x**2*expint(1, x)/2 + x*exp(-x)/2 - exp(-x)/2
    
    # 断言简化未极化的积分结果等于 -expint(y + 1, x)
    assert simplify(unpolarify(integrate(expint(y, x), x,
                 meijerg=True).rewrite(expint).expand(func=True))) == \
        -expint(y + 1, x)

    # 断言正弦积分函数的积分结果
    assert integrate(Si(x), x, meijerg=True) == x*Si(x) + cos(x)
    
    # 断言柯西函数的积分结果
    assert integrate(Ci(u), u, meijerg=True).expand() == u*Ci(u) - sin(u)
    
    # 断言双曲正弦积分函数的积分结果
    assert integrate(Shi(x), x, meijerg=True) == x*Shi(x) - cosh(x)
    
    # 断言科尔曼函数的积分结果
    assert integrate(Chi(u), u, meijerg=True).expand() == u*Chi(u) - sinh(u)

    # 断言正弦积分函数乘以指数函数的积分结果
    assert integrate(Si(x)*exp(-x), (x, 0, oo), meijerg=True) == pi/4
    
    # 断言指数积分函数乘以正弦函数的积分结果
    assert integrate(expint(1, x)*sin(x), (x, 0, oo), meijerg=True) == log(2)/2


def test_messy():
    # 导入所需的函数模块
    from sympy.functions.elementary.hyperbolic import (acosh, acoth)
    from sympy.functions.elementary.trigonometric import (asin, atan)
    from sympy.functions.special.bessel import besselj
    from sympy.functions.special.error_functions import (Chi, E1, Shi, Si)
    from sympy.integrals.transforms import (fourier_transform, laplace_transform)
    # 使用 Laplace 变换对 Si(x) 进行变换，期望结果简化后等于 ((-atan(s) + pi/2)/s, 0, True)
    assert (laplace_transform(Si(x), x, s, simplify=True) ==
            ((-atan(s) + pi/2)/s, 0, True))

    # 使用 Laplace 变换对 Shi(x) 进行变换，期望结果简化后等于 (acoth(s)/s, -oo, s**2 > 1)
    assert laplace_transform(Shi(x), x, s, simplify=True) == (
        acoth(s)/s, -oo, s**2 > 1)

    # 使用 Laplace 变换对 Chi(x) 进行变换，期望结果简化后等于 ((log(s**(-2)) - log(1 - 1/s**2))/(2*s), -oo, s**2 > 1)
    # 这里提醒可能需要简化对数中的不等式
    assert laplace_transform(Chi(x), x, s, simplify=True) == (
        (log(s**(-2)) - log(1 - 1/s**2))/(2*s), -oo, s**2 > 1)

    # 使用 Laplace 变换对 besselj(a, x) 进行变换，期望结果除第一个元素外等于 (0, (re(a) > -2) & (re(a) > -1))
    assert laplace_transform(besselj(a, x), x, s)[1:] == \
        (0, (re(a) > -2) & (re(a) > -1))

    # 对 besselj(1, x)/x 进行 Fourier 变换，期望结果简化后等于 Piecewise((0, (s > 1/(2*pi)) | (s < -1/(2*pi))),
    #                   (2*sqrt(-4*pi**2*s**2 + 1), True)), s > 0)
    # 这里提醒可能需要简化条件不等式
    ans = fourier_transform(besselj(1, x)/x, x, s, noconds=False)
    assert (ans[0].factor(deep=True).expand(), ans[1]) == \
        (Piecewise((0, (s > 1/(2*pi)) | (s < -1/(2*pi))),
                   (2*sqrt(-4*pi**2*s**2 + 1), True)), s > 0)
    # 这里提醒 Fourier 变换对于 besselj(0, x) 的情况比较混乱，但原因是可以接受的

    # 使用 Meijer G 函数计算 E1(x)*besselj(0, x) 在 (x, 0, oo) 区间上的积分，期望结果等于 log(1 + sqrt(2))
    assert integrate(E1(x)*besselj(0, x), (x, 0, oo), meijerg=True) == \
        log(1 + sqrt(2))

    # 使用 Meijer G 函数计算 E1(x)*besselj(1, x) 在 (x, 0, oo) 区间上的积分，期望结果等于 log(S.Half + sqrt(2)/2)
    assert integrate(E1(x)*besselj(1, x), (x, 0, oo), meijerg=True) == \
        log(S.Half + sqrt(2)/2)

    # 使用 Meijer G 函数计算 1/(x*sqrt(1 - x**2)) 在 x 上的积分，期望结果等于 Piecewise((-acosh(1/x), abs(x**(-2)) > 1), (I*asin(1/x), True))
    assert integrate(1/x/sqrt(1 - x**2), x, meijerg=True) == \
        Piecewise((-acosh(1/x), abs(x**(-2)) > 1), (I*asin(1/x), True))
def test_issue_6122():
    # 对于表达式 exp(-I*x**2)，在无穷区间 (-oo, oo) 上进行积分，使用 Meijer G 函数计算
    assert integrate(exp(-I*x**2), (x, -oo, oo), meijerg=True) == \
        -I*sqrt(pi)*exp(I*pi/4)


def test_issue_6252():
    # 创建一个复杂的表达式 expr，并使用 Meijer G 函数对其进行积分
    expr = 1/x/(a + b*x)**Rational(1, 3)
    anti = integrate(expr, x, meijerg=True)
    # 断言：anti 中不含有超函函数（hyper）
    assert not anti.has(hyper)
    # XXX 这个表达式有点乱，但实际上在微分和放入数值后似乎是有效的...


def test_issue_6348():
    # 对于表达式 exp(I*x)/(1 + x**2)，在无穷区间 (-oo, oo) 上进行积分，化简后重写为 exp 函数的形式
    assert integrate(exp(I*x)/(1 + x**2), (x, -oo, oo)).simplify().rewrite(exp) \
        == pi*exp(-1)


def test_fresnel():
    # 导入 Fresnel 函数并使用 expand_func 函数对 sin(pi*x**2/2) 和 cos(pi*x**2/2) 进行积分
    from sympy.functions.special.error_functions import (fresnelc, fresnels)

    assert expand_func(integrate(sin(pi*x**2/2), x)) == fresnels(x)
    assert expand_func(integrate(cos(pi*x**2/2), x)) == fresnelc(x)


def test_issue_6860():
    # 对于表达式 x**x**x，使用 Meijer G 函数进行不定积分
    assert meijerint_indefinite(x**x**x, x) is None


def test_issue_7337():
    # 对于表达式 x*sqrt(2*x + 3)，使用 Meijer G 函数进行不定积分并将结果整合在一起
    f = meijerint_indefinite(x*sqrt(2*x + 3), x).together()
    assert f == sqrt(2*x + 3)*(2*x**2 + x - 3)/5
    assert f._eval_interval(x, S.NegativeOne, S.One) == Rational(2, 5)


def test_issue_8368():
    # 对于表达式 cosh(x)*exp(-x*t)，使用 Meijer G 函数进行不定积分
    assert meijerint_indefinite(cosh(x)*exp(-x*t), x) == (
        (-t - 1)*exp(x) + (-t + 1)*exp(-x))*exp(-t*x)/2/(t**2 - 1)


def test_issue_10211():
    # 对于表达式 1/sqrt((y-x)**2 + h**2)**3，进行二重积分，其中 x 的范围是 [0, w]，y 的范围是 [0, w]
    assert integrate((1/sqrt((y-x)**2 + h**2)**3), (x,0,w), (y,0,w)) == \
        2*sqrt(1 + w**2/h**2)/h - 2/h


def test_issue_11806():
    # 对于表达式 1/sqrt(x**2 + y**2)**3，进行积分，其中 x 的范围是 [-L, L]
    from sympy.core.symbol import symbols
    y, L = symbols('y L', positive=True)
    assert integrate(1/sqrt(x**2 + y**2)**3, (x, -L, L)) == \
        2*L/(y**2*sqrt(L**2 + y**2))


def test_issue_10681():
    # 对于表达式 r**2*(R**2-r**2)**0.5，使用 Meijer G 函数进行不定积分
    from sympy.polys.domains.realfield import RR
    from sympy.abc import R, r
    f = integrate(r**2*(R**2-r**2)**0.5, r, meijerg=True)
    g = (1.0/3)*R**1.0*r**3*hyper((-0.5, Rational(3, 2)), (Rational(5, 2),),
                                  r**2*exp_polar(2*I*pi)/R**2)
    assert RR.almosteq((f/g).n(), 1.0, 1e-12)


def test_issue_13536():
    # 对于表达式 1/x**2，进行积分，其中 x 的范围是 [oo, a]
    from sympy.core.symbol import Symbol
    a = Symbol('a', positive=True)
    assert integrate(1/x**2, (x, oo, a)) == -1/a


def test_issue_6462():
    # 对于表达式 cos(x**n)/x**n，进行积分，其中 x 的范围是整个实数轴，n = 2 时比较两次积分结果
    from sympy.core.symbol import Symbol
    x = Symbol('x')
    n = Symbol('n')
    # 不是真正的问题，但 n = 1 时也是错误的答案，但没有异常
    assert integrate(cos(x**n)/x**n, x, meijerg=True).subs(n, 2).equals(
            integrate(cos(x**2)/x**2, x, meijerg=True))


def test_indefinite_1_bug():
    # 对于表达式 (b + t)**(-a)，使用 Meijer G 函数进行不定积分
    assert integrate((b + t)**(-a), t, meijerg=True) == -b*(1 + t/b)**(1 - a)/(a*b**a - b**a)


def test_pr_23583():
    # 这个结果是错误的。当这个测试失败时，检查新结果是否正确。
    assert integrate(1/sqrt((x - I)**2-1), meijerg=True) == \
           Piecewise((acosh(x - I), Abs((x - I)**2) > 1), (-I*asin(x - I), True))


def test_integrate_function_of_square_over_negatives():
    # 对于表达式 exp(-x**2)，在区间 [-5, 0] 上进行积分，使用 Meijer G 函数计算
    assert integrate(exp(-x**2), (x,-5,0), meijerg=True) == sqrt(pi)/2 * erf(5)


def test_issue_25949():
    # 导入符号并开始一个新的测试，未完成的测试案例
    from sympy.core.symbol import symbols
    # 创建符号变量 y，并确保其非零
    y = symbols("y", nonzero=True)
    # 使用 sympy 库中的 integrate 函数对 cosh(y*(x + 1)) 进行积分，积分区间为 x 从 -1 到 -0.25
    # meijerg=True 表示使用 Meijer G 函数进行积分计算
    # 断言计算结果应该等于 sinh(0.75*y) / y
    assert integrate(cosh(y*(x + 1)), (x, -1, -0.25), meijerg=True) == sinh(0.75*y) / y
```