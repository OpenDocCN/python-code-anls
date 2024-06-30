# `D:\src\scipysrc\sympy\sympy\stats\tests\test_discrete_rv.py`

```
# 导入SymPy库中具体的类和函数，用于具体的数学运算和统计分析
from sympy.concrete.summations import Sum
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.zeta_functions import zeta
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Eq, Ne
from sympy.logic.boolalg import Or
from sympy.sets.fancysets import Range
from sympy.stats import (P, E, variance, density, characteristic_function,
                         where, moment_generating_function, skewness, cdf,
                         kurtosis, coskewness)
from sympy.stats.drv_types import (PoissonDistribution, GeometricDistribution,
                                   FlorySchulz, Poisson, Geometric, Hermite, Logarithmic,
                                   NegativeBinomial, Skellam, YuleSimon, Zeta,
                                   DiscreteRV)
from sympy.testing.pytest import slow, nocache_fail, raises
from sympy.stats.symbolic_probability import Expectation

# 创建一个符号变量x，用于后续的符号运算
x = Symbol('x')

# 定义一个测试函数，用于测试Poisson分布
def test_PoissonDistribution():
    l = 3
    # 创建一个Poisson分布对象，参数为l
    p = PoissonDistribution(l)
    # 断言Poisson分布在10处的累积分布函数接近1
    assert abs(p.cdf(10).evalf() - 1) < .001
    # 断言Poisson分布在10.4处的累积分布函数接近1
    assert abs(p.cdf(10.4).evalf() - 1) < .001
    # 断言Poisson分布的期望值等于参数l
    assert p.expectation(x, x) == l
    # 断言Poisson分布的x^2的期望值减去x的期望值的平方等于参数l
    assert p.expectation(x**2, x) - p.expectation(x, x)**2 == l

# 定义一个测试函数，用于测试Poisson分布的随机变量对象
def test_Poisson():
    l = 3
    # 创建一个Poisson分布的随机变量，参数为l
    x = Poisson('x', l)
    # 断言随机变量的期望值等于参数l
    assert E(x) == l
    # 断言随机变量2*x的期望值等于2*l
    assert E(2*x) == 2*l
    # 断言随机变量的方差等于参数l
    assert variance(x) == l
    # 断言随机变量的概率密度函数为Poisson分布(l)
    assert density(x) == PoissonDistribution(l)
    # 断言E(x, evaluate=False)返回一个Expectation对象
    assert isinstance(E(x, evaluate=False), Expectation)
    # 断言E(2*x, evaluate=False)返回一个Expectation对象
    assert isinstance(E(2*x, evaluate=False), Expectation)
    # issue 8248的问题断言，随机变量的空间计算期望为1
    assert x.pspace.compute_expectation(1) == 1

# 定义一个测试函数，用于测试FlorySchulz分布
def test_FlorySchulz():
    a = Symbol("a")
    z = Symbol("z")
    # 创建一个FlorySchulz分布的随机变量，参数为a
    x = FlorySchulz('x', a)
    # 断言随机变量的期望值为(2 - a)/a
    assert E(x) == (2 - a)/a
    # 断言随机变量的方差减去2*(1 - a)/a**2等于0
    assert (variance(x) - 2*(1 - a)/a**2).simplify() == S(0)
    # 断言随机变量的概率密度函数为a**2*z*(1 - a)**(z - 1)
    assert density(x)(z) == a**2*z*(1 - a)**(z - 1)

# 定义一个测试函数，用于测试几何分布
@slow
def test_GeometricDistribution():
    p = S.One / 5
    # 创建一个几何分布的随机变量，参数为p
    d = GeometricDistribution(p)
    # 断言几何分布的期望值等于1/p
    assert d.expectation(x, x) == 1/p
    # 断言几何分布的x^2的期望值减去x的期望值的平方等于(1-p)/p**2
    assert d.expectation(x**2, x) - d.expectation(x, x)**2 == (1-p)/p**2
    # 断言几何分布在20000处的累积分布函数接近1
    assert abs(d.cdf(20000).evalf() - 1) < .001
    # 断言几何分布在20000.8处的累积分布函数接近1
    assert abs(d.cdf(20000.8).evalf() - 1) < .001
    # 创建一个几何分布的随机变量G，参数为p=1/4
    G = Geometric('G', p=S(1)/4)
    # 断言G随机变量的累积分布函数在7/2处等于P(G <= 7/2)
    assert cdf(G)(S(7)/2) == P(G <= S(7)/2)

    # 创建几何分布的随机变量X和Y，参数分别为1/5和3/10
    X = Geometric('X', Rational(1, 5))
    Y = Geometric('Y', Rational(3, 10))
    # 断言X, X+Y, X+2*Y的共偏度化简结果为sqrt(230)*Rational(81, 1150)
    assert coskewness(X, X + Y, X + 2*Y).simplify() == sqrt(230)*Rational(81, 1150)
    # 创建一个正的符号变量a1
    a1 = Symbol("a1", positive=True)
    # 创建一个负的符号变量a2
    a2 = Symbol("a2", negative=True)
    # 使用Hermite类创建一个Hermite多项式对象，并预期引发值错误(ValueError)异常
    raises(ValueError, lambda: Hermite("H", a1, a2))

    # 创建一个负的符号变量a1
    a1 = Symbol("a1", negative=True)
    # 创建一个正的符号变量a2
    a2 = Symbol("a2", positive=True)
    # 使用Hermite类创建一个Hermite多项式对象，并预期引发值错误(ValueError)异常
    raises(ValueError, lambda: Hermite("H", a1, a2))

    # 创建一个正的符号变量a1
    a1 = Symbol("a1", positive=True)
    # 创建一个未指定正负的符号变量x
    x = Symbol("x")
    # 使用Hermite类创建一个Hermite多项式对象H，其参数为a1和a2
    H = Hermite("H", a1, a2)
    # 断言H的矩生成函数在变量x处的值等于指定的数学表达式
    assert moment_generating_function(H)(x) == exp(a1*(exp(x) - 1)
                                            + a2*(exp(2*x) - 1))
    # 断言H的特征函数在变量x处的值等于指定的数学表达式
    assert characteristic_function(H)(x) == exp(a1*(exp(I*x) - 1)
                                                + a2*(exp(2*I*x) - 1))
    # 断言H的期望值等于指定的数学表达式
    assert E(H) == a1 + 2*a2

    # 使用Hermite类创建一个Hermite多项式对象H，指定参数a1=5和a2=4
    H = Hermite("H", a1=5, a2=4)
    # 断言H的概率密度函数在参数为2的点处的值等于指定的数学表达式
    assert density(H)(2) == 33*exp(-9)/2
    # 断言H的期望值等于指定的数学表达式
    assert E(H) == 13
    # 断言H的方差等于指定的数学表达式
    assert variance(H) == 21
    # 断言H的峰度等于指定的有理数
    assert kurtosis(H) == Rational(464,147)
    # 断言H的偏度等于指定的数学表达式
    assert skewness(H) == 37*sqrt(21)/441
# 定义一个测试函数 test_Logarithmic
def test_Logarithmic():
    # 设置参数 p 为 S.Half，即分数 1/2
    p = S.Half
    # 创建一个对数分布对象 x，命名为 'x'，参数为 p
    x = Logarithmic('x', p)
    # 断言对数分布 x 的期望 E(x) 的计算结果
    assert E(x) == -p / ((1 - p) * log(1 - p))
    # 断言对数分布 x 的方差 variance(x) 的计算结果
    assert variance(x) == -1/log(2)**2 + 2/log(2)
    # 断言对数分布 x 中表达式 2*x**2 + 3*x + 4 的期望 E 的计算结果
    assert E(2*x**2 + 3*x + 4) == 4 + 7 / log(2)
    # 断言对数分布 x 的期望 E(x, evaluate=False) 返回 Expectation 类型对象

# 被 @nocache_fail 装饰的测试函数 test_negative_binomial
@nocache_fail
def test_negative_binomial():
    # 设定负二项分布的参数 r 和 p
    r = 5
    p = S.One / 3
    # 创建一个负二项分布对象 x，命名为 'x'，参数为 r 和 p
    x = NegativeBinomial('x', r, p)
    # 断言负二项分布 x 的期望 E(x) 的计算结果
    assert E(x) == p*r / (1-p)
    # 断言负二项分布 x 的方差 variance(x) 的计算结果
    assert variance(x) == p*r / (1-p)**2
    # 断言负二项分布 x 中表达式 x**5 + 2*x + 3 的期望 E 的计算结果
    assert E(x**5 + 2*x + 3) == Rational(9207, 4)
    # 断言负二项分布 x 的期望 E(x, evaluate=False) 返回 Expectation 类型对象

# 定义一个测试函数 test_skellam
def test_skellam():
    # 创建符号变量 mu1, mu2, z
    mu1 = Symbol('mu1')
    mu2 = Symbol('mu2')
    z = Symbol('z')
    # 创建 Skellam 分布对象 X，命名为 'x'，参数为 mu1 和 mu2
    X = Skellam('x', mu1, mu2)
    # 断言 Skellam 分布 X 的密度函数 density(X)(z) 的计算结果
    assert density(X)(z) == (mu1/mu2)**(z/2) * \
        exp(-mu1 - mu2)*besseli(z, 2*sqrt(mu1*mu2))
    # 断言 Skellam 分布 X 的偏度 skewness(X) 展开后的计算结果
    assert skewness(X).expand() == mu1/(mu1*sqrt(mu1 + mu2) + mu2 *
                sqrt(mu1 + mu2)) - mu2/(mu1*sqrt(mu1 + mu2) + mu2*sqrt(mu1 + mu2))
    # 断言 Skellam 分布 X 的方差 variance(X) 展开后的计算结果
    assert variance(X).expand() == mu1 + mu2
    # 断言 Skellam 分布 X 的期望 E(X) 的计算结果
    assert E(X) == mu1 - mu2
    # 断言 Skellam 分布 X 的特征函数 characteristic_function(X)(z) 的计算结果
    assert characteristic_function(X)(z) == exp(
        mu1*exp(I*z) - mu1 - mu2 + mu2*exp(-I*z))
    # 断言 Skellam 分布 X 的矩生成函数 moment_generating_function(X)(z) 的计算结果
    assert moment_generating_function(X)(z) == exp(
        mu1*exp(z) - mu1 - mu2 + mu2*exp(-z))

# 定义一个测试函数 test_yule_simon
def test_yule_simon():
    # 导入符号 S 作为从 sympy.core.singleton 模块导入
    from sympy.core.singleton import S
    # 设置参数 rho 为 S(3)，即数值 3
    rho = S(3)
    # 创建 Yule-Simon 分布对象 x，命名为 'x'，参数为 rho
    x = YuleSimon('x', rho)
    # 断言简化后的 Yule-Simon 分布 x 的期望 E(x) 的计算结果
    assert simplify(E(x)) == rho / (rho - 1)
    # 断言简化后的 Yule-Simon 分布 x 的方差 variance(x) 的计算结果
    assert simplify(variance(x)) == rho**2 / ((rho - 1)**2 * (rho - 2))
    # 断言 Yule-Simon 分布 x 的期望 E(x, evaluate=False) 返回 Expectation 类型对象
    assert isinstance(E(x, evaluate=False), Expectation)
    # 断言用于测试累积分布函数 cdf(x) 的计算结果
    assert cdf(x)(x) == Piecewise((-beta(floor(x), 4)*floor(x) + 1, x >= 1), (0, True))

# 定义一个测试函数 test_zeta
def test_zeta():
    # 设置参数 s 为 S(5)，即数值 5
    s = S(5)
    # 创建 Zeta 分布对象 x，命名为 'x'，参数为 s
    x = Zeta('x', s)
    # 断言 Zeta 分布 x 的期望 E(x) 的计算结果
    assert E(x) == zeta(s-1) / zeta(s)
    # 断言简化后的 Zeta 分布 x 的方差 variance(x) 的计算结果
    assert simplify(variance(x)) == (
        zeta(s) * zeta(s-2) - zeta(s-1)**2) / zeta(s)**2

# 定义一个测试函数 test_discrete_probability
def test_discrete_probability():
    # 创建几何分布对象 X，命名为 'X'，参数为 Rational(1, 5)
    X = Geometric('X', Rational(1, 5))
    # 创建泊松分布对象 Y，命名为 'Y'，参数为 4
    Y = Poisson('Y', 4)
    # 创建几何分布对象 G，命名为 'e'，参数为 x
    G = Geometric('e', x)
    # 断言几何分布 X 满足等式 Eq(X, 3) 的概率 P(Eq(X, 3)) 的计算结果
    assert P(Eq(X, 3)) == Rational(16, 125)
    # 断言几何分布 X 满足 X < 3 的概率 P(X < 3) 的计算结果
    assert P(X < 3) == Rational(9, 25)
    # 断言几何分布 X 满足 X > 3 的概率 P(X > 3) 的计算结果
    assert P(X > 3) == Rational(64, 125)
    # 断言几何分布 X 满足 X >= 3 的概率 P(X >= 3) 的计算结果
    assert P(X >= 3) == Rational(16, 25)
    # 断言几何分布 X 满足 X <= 3 的概率 P(X <= 3) 的计算结果
    assert P(X <= 3) == Rational(61, 125)
    # 断言几何分布 X 满足不等式 Ne(X, 3) 的概率 P(Ne(X, 3)) 的计算结果
    assert P(Ne(X, 3)) == Rational(109, 125)
    # 断言泊松分布 Y 满足等式 Eq(Y, 3) 的概率 P(Eq(Y, 3)) 的计算结果
    assert P(Eq(Y, 3)) == 32*exp(-4)/3
    # 断言泊松分布 Y 满足 Y < 3 的概率 P(Y < 3) 的计算结果
    assert P(Y < 3) == 13*exp(-4)
    # 断言泊松分布 Y 满足 Y > 3 的
    # 断言概率变量 D 大于 3 的概率等于 1/8
    assert P(D > 3) == S(1)/8
    
    # 断言概率变量 D 的样本空间是自然数集 S.Naturals
    assert D.pspace.domain.set == S.Naturals
    
    # 使用无效的概率质量函数创建离散随机变量 X，但由于 check=False，不应该引发异常
    # 参见 test_drv_types.test_ContinuousRV 中的解释
    X = DiscreteRV(x, 1/x, S.Naturals)
    
    # 断言随机变量 X 小于 2 的概率等于 1
    assert P(X < 2) == 1
    
    # 断言随机变量 X 的期望值是无穷大
    assert E(X) == oo
# 定义一个测试函数，用于测试预计算的特征函数
def test_precomputed_characteristic_functions():
    # 导入 mpmath 库，用于数学计算
    import mpmath

    # 定义一个内部函数 test_cf，用于测试特征函数
    def test_cf(dist, support_lower_limit, support_upper_limit):
        # 获取分布的概率密度函数
        pdf = density(dist)
        # 定义符号变量 t 和 x
        t = S('t')
        x = S('x')

        # 第一个函数是分布的硬编码特征函数
        cf1 = lambdify([t], characteristic_function(dist)(t), 'mpmath')

        # 第二个函数是密度函数的傅立叶变换
        f = lambdify([x, t], pdf(x)*exp(I*x*t), 'mpmath')
        # 定义第二个特征函数为对密度函数傅立叶变换的求和
        cf2 = lambda t: mpmath.nsum(lambda x: f(x, t), [
            support_lower_limit, support_upper_limit], maxdegree=10)

        # 在几个测试点比较这两个函数的值
        for test_point in [2, 5, 8, 11]:
            n1 = cf1(test_point)
            n2 = cf2(test_point)

            # 断言两个函数在实部和虚部上的差距小于 1e-12
            assert abs(re(n1) - re(n2)) < 1e-12
            assert abs(im(n1) - im(n2)) < 1e-12

    # 使用不同的分布调用 test_cf 函数进行测试
    test_cf(Geometric('g', Rational(1, 3)), 1, mpmath.inf)
    test_cf(Logarithmic('l', Rational(1, 5)), 1, mpmath.inf)
    test_cf(NegativeBinomial('n', 5, Rational(1, 7)), 0, mpmath.inf)
    test_cf(Poisson('p', 5), 0, mpmath.inf)
    test_cf(YuleSimon('y', 5), 1, mpmath.inf)
    test_cf(Zeta('z', 5), 1, mpmath.inf)


# 测试动差生成函数的函数
def test_moment_generating_functions():
    # 定义符号变量 t
    t = S('t')

    # 计算几何分布的动差生成函数，并断言其一阶导数在 t=0 处为 2
    geometric_mgf = moment_generating_function(Geometric('g', S.Half))(t)
    assert geometric_mgf.diff(t).subs(t, 0) == 2

    # 计算对数分布的动差生成函数，并断言其一阶导数在 t=0 处为 1/log(2)
    logarithmic_mgf = moment_generating_function(Logarithmic('l', S.Half))(t)
    assert logarithmic_mgf.diff(t).subs(t, 0) == 1/log(2)

    # 计算负二项分布的动差生成函数，并断言其一阶导数在 t=0 处为 5/2
    negative_binomial_mgf = moment_generating_function(
        NegativeBinomial('n', 5, Rational(1, 3)))(t)
    assert negative_binomial_mgf.diff(t).subs(t, 0) == Rational(5, 2)

    # 计算泊松分布的动差生成函数，并断言其一阶导数在 t=0 处为 5
    poisson_mgf = moment_generating_function(Poisson('p', 5))(t)
    assert poisson_mgf.diff(t).subs(t, 0) == 5

    # 计算斯凯勒姆分布的动差生成函数，并断言其在 t=2 处的导数满足特定的条件
    skellam_mgf = moment_generating_function(Skellam('s', 1, 1))(t)
    assert skellam_mgf.diff(t).subs(
        t, 2) == (-exp(-2) + exp(2))*exp(-2 + exp(-2) + exp(2))

    # 计算尤尔-西蒙分布的动差生成函数，并简化其一阶导数在 t=0 处
    yule_simon_mgf = moment_generating_function(YuleSimon('y', 3))(t)
    assert simplify(yule_simon_mgf.diff(t).subs(t, 0)) == Rational(3, 2)

    # 计算默兹塔林分布的动差生成函数，并断言其一阶导数在 t=0 处为 pi**4/(90*zeta(5))
    zeta_mgf = moment_generating_function(Zeta('z', 5))(t)
    assert zeta_mgf.diff(t).subs(t, 0) == pi**4/(90*zeta(5))


# 测试 Or 运算符的函数
def test_Or():
    # 定义几何分布 X
    X = Geometric('X', S.Half)
    # 断言 P(Or(X < 3, X > 4)) 等于 13/16
    assert P(Or(X < 3, X > 4)) == Rational(13, 16)
    # 断言 P(Or(X > 2, X > 1)) 等于 P(X > 1)
    assert P(Or(X > 2, X > 1)) == P(X > 1)
    # 断言 P(Or(X >= 3, X < 3)) 等于 1
    assert P(Or(X >= 3, X < 3)) == 1


# 测试 where 函数的函数
def test_where():
    # 定义几何分布 X 和泊松分布 Y
    X = Geometric('X', Rational(1, 5))
    Y = Poisson('Y', 4)
    # 断言 where(X**2 > 4).set 的范围是 [3, ∞)
    assert where(X**2 > 4).set == Range(3, S.Infinity, 1)
    # 断言 where(X**2 >= 4).set 的范围是 [2, ∞)
    assert where(X**2 >= 4).set == Range(2, S.Infinity, 1)
    # 断言 where(Y**2 < 9).set 的范围是 [0, 3]
    assert where(Y**2 < 9).set == Range(0, 3, 1)
    # 断言 where(Y**2 <= 9).set 的范围是 [0, 4]
    assert where(Y**2 <= 9).set == Range(0, 4, 1)


# 测试条件概率的函数
def test_conditional():
    # 定义几何分布 X 和泊松分布 Y
    X = Geometric('X', Rational(2, 3))
    Y = Poisson('Y', 3)
    # 断言 P(X > 2, X > 3) 等于 1
    assert P(X > 2, X > 3) == 1
    # 断言 P(X > 3, X > 2) 等于 1/3
    assert P(X > 3, X > 2) == Rational(1, 3)
    # 断言 P(Y > 2, Y < 2) 等于 0
    assert P(Y > 2, Y < 2) == 0
    # 断言 P(Eq(Y, 3), Y >= 0) 等于 9*exp(-3)/2
    assert P(Eq(Y, 3), Y >= 0) == 9*exp(-3)/2
    # 断言 P(Eq(Y, 3), Eq(Y, 2)) 等于 0
    assert P(Eq(Y, 3), Eq(Y, 2)) == 0
    # 断言：检查条件概率 P(X < 2, Eq(X, 2)) 是否为 0
    assert P(X < 2, Eq(X, 2)) == 0
    # 断言：检查条件概率 P(X > 2, Eq(X, 3)) 是否为 1
    assert P(X > 2, Eq(X, 3)) == 1
# 定义一个测试函数，用于测试几何分布的随机变量和概率密度函数的相关功能
def test_product_spaces():
    # 创建一个名为 X1 的几何分布随机变量，成功概率为 1/2
    X1 = Geometric('X1', S.Half)
    # 创建一个名为 X2 的几何分布随机变量，成功概率为 1/3
    X2 = Geometric('X2', Rational(1, 3))
    # 断言验证 P(X1 + X2 < 3) 的字符串表示是否等于指定值
    assert str(P(X1 + X2 < 3).rewrite(Sum)) == (
        "Sum(Piecewise((1/(4*2**n), n >= -1), (0, True)), (n, -oo, -1))/3")
    # 断言验证 P(X1 + X2 > 3) 的字符串表示是否等于指定值
    assert str(P(X1 + X2 > 3).rewrite(Sum)) == (
        'Sum(Piecewise((2**(X2 - n - 2)*(3/2)**(1 - X2)/6, '
        'X2 - n <= 2), (0, True)), (X2, 1, oo), (n, 1, oo))')
    # 断言验证 P(X1 + X2 = 3) 是否等于 1/12
    assert P(Eq(X1 + X2, 3)) == Rational(1, 12)
```