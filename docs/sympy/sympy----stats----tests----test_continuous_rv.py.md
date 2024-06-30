# `D:\src\scipysrc\sympy\sympy\stats\tests\test_continuous_rv.py`

```
# 从 sympy.concrete.summations 模块导入 Sum 类
from sympy.concrete.summations import Sum
# 从 sympy.core.function 模块导入 Lambda、diff、expand_func 函数
from sympy.core.function import (Lambda, diff, expand_func)
# 从 sympy.core.mul 模块导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core 模块导入 EulerGamma 常数
from sympy.core import EulerGamma
# 从 sympy.core.numbers 模块导入 E 常数（自然对数的底 e）、I（虚数单位）、Rational（有理数）、pi（圆周率）
from sympy.core.numbers import (E as e, I, Rational, pi)
# 从 sympy.core.relational 模块导入 Eq（等式）和 Ne（不等式）类
from sympy.core.relational import (Eq, Ne)
# 从 sympy.core.singleton 模块导入 S（单例符号）
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Dummy、Symbol、symbols 函数
from sympy.core.symbol import (Dummy, Symbol, symbols)
# 从 sympy.functions.combinatorial.factorials 模块导入 binomial（二项式系数）、factorial（阶乘）函数
from sympy.functions.combinatorial.factorials import (binomial, factorial)
# 从 sympy.functions.elementary.complexes 模块导入 Abs（绝对值）、im（虚部）、re（实部）、sign（符号）函数
from sympy.functions.elementary.complexes import (Abs, im, re, sign)
# 从 sympy.functions.elementary.exponential 模块导入 exp（指数函数）、log（对数函数）函数
from sympy.functions.elementary.exponential import (exp, log)
# 从 sympy.functions.elementary.hyperbolic 模块导入 cosh（双曲余弦）、sinh（双曲正弦）函数
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
# 从 sympy.functions.elementary.integers 模块导入 floor（向下取整）函数
from sympy.functions.elementary.integers import floor
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt（平方根）函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.piecewise 模块导入 Piecewise（分段函数）类
from sympy.functions.elementary.piecewise import Piecewise
# 从 sympy.functions.elementary.trigonometric 模块导入 asin（反正弦）、atan（反正切）、cos（余弦）、sin（正弦）、tan（正切）函数
from sympy.functions.elementary.trigonometric import (asin, atan, cos, sin, tan)
# 从 sympy.functions.special.bessel 模块导入 besseli（修正贝塞尔函数）、besselj（贝塞尔函数）、besselk（第二类修正贝塞尔函数）函数
from sympy.functions.special.bessel import (besseli, besselj, besselk)
# 从 sympy.functions.special.beta_functions 模块导入 beta（贝塔函数）函数
from sympy.functions.special.beta_functions import beta
# 从 sympy.functions.special.error_functions 模块导入 erf（误差函数）、erfc（余误差函数）、erfi（反余误差函数）、expint（指数积分函数）函数
from sympy.functions.special.error_functions import (erf, erfc, erfi, expint)
# 从 sympy.functions.special.gamma_functions 模块导入 gamma（伽玛函数）、lowergamma（不完全伽玛函数）、uppergamma（上不完全伽玛函数）函数
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
# 从 sympy.functions.special.zeta_functions 模块导入 zeta（黎曼 zeta 函数）函数
from sympy.functions.special.zeta_functions import zeta
# 从 sympy.functions.special.hyper 模块导入 hyper（超几何函数）函数
from sympy.functions.special.hyper import hyper
# 从 sympy.integrals.integrals 模块导入 Integral（积分）类
from sympy.integrals.integrals import Integral
# 从 sympy.logic.boolalg 模块导入 And（逻辑与）、Or（逻辑或）函数
from sympy.logic.boolalg import (And, Or)
# 从 sympy.sets.sets 模块导入 Interval（区间）类
from sympy.sets.sets import Interval
# 从 sympy.simplify.simplify 模块导入 simplify（简化表达式）函数
from sympy.simplify.simplify import simplify
# 从 sympy.utilities.lambdify 模块导入 lambdify（生成函数的数值计算版本）函数
from sympy.utilities.lambdify import lambdify
# 从 sympy.functions.special.error_functions 模块导入 erfinv（逆误差函数）函数
from sympy.functions.special.error_functions import erfinv
# 从 sympy.functions.special.hyper 模块导入 meijerg（梅瑟函数）函数
from sympy.functions.special.hyper import meijerg
# 从 sympy.sets.sets 模块导入 FiniteSet（有限集合）、Complement（补集）、Intersection（交集）类
from sympy.sets.sets import FiniteSet, Complement, Intersection
# 从 sympy.stats 模块导入多个函数和类，包括概率统计相关的函数和分布
from sympy.stats import (P, E, where, density, variance, covariance, skewness, kurtosis, median,
                         given, pspace, cdf, characteristic_function, moment_generating_function,
                         ContinuousRV, Arcsin, Benini, Beta, BetaNoncentral, BetaPrime,
                         Cauchy, Chi, ChiSquared, ChiNoncentral, Dagum, Davis, Erlang, ExGaussian,
                         Exponential, ExponentialPower, FDistribution, FisherZ, Frechet, Gamma,
                         GammaInverse, Gompertz, Gumbel, Kumaraswamy, Laplace, Levy, Logistic, LogCauchy,
                         LogLogistic, LogitNormal, LogNormal, Maxwell, Moyal, Nakagami, Normal, GaussianInverse,
                         Pareto, PowerFunction, QuadraticU, RaisedCosine, Rayleigh, Reciprocal, ShiftedGompertz, StudentT,
                         Trapezoidal, Triangular, Uniform, UniformSum, VonMises, Weibull, coskewness,
                         WignerSemicircle, Wald, correlation, moment, cmoment, smoment, quantile,
                         Lomax, BoundedPareto)
# 从 sympy.stats.crv_types 模块导入 NormalDistribution（正态分布）、ExponentialDistribution（指数分布）、ContinuousDistributionHandmade（手工连续分布）类
from sympy.stats.crv_types import NormalDistribution, ExponentialDistribution, ContinuousDistributionHandmade
# 从 sympy.stats.joint_rv_types 模块导入 MultivariateLaplaceDistribution（多元拉普拉斯分布）、MultivariateNormalDistribution（多元正态分布）类
from sympy.stats.joint_rv_types import MultivariateLaplaceDistribution, MultivariateNormalDistribution
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDomain  # 导入单变量连续概率空间和域
from sympy.stats.compound_rv import CompoundPSpace  # 导入复合随机变量概率空间
from sympy.stats.symbolic_probability import Probability  # 导入符号概率
from sympy.testing.pytest import raises, XFAIL, slow, ignore_warnings  # 导入测试相关函数和装饰器
from sympy.core.random import verify_numerically as tn  # 导入数值验证函数

oo = S.Infinity  # 定义无穷大

x, y, z = map(Symbol, 'xyz')  # 定义符号变量 x, y, z

def test_single_normal():
    mu = Symbol('mu', real=True)  # 定义实数符号变量 mu
    sigma = Symbol('sigma', positive=True)  # 定义正数符号变量 sigma
    X = Normal('x', 0, 1)  # 创建均值为0，方差为1的正态分布随机变量 X
    Y = X*sigma + mu  # 定义随机变量 Y，为 X*sigma + mu

    assert E(Y) == mu  # 断言 Y 的期望值为 mu
    assert variance(Y) == sigma**2  # 断言 Y 的方差为 sigma 的平方
    pdf = density(Y)  # 计算 Y 的概率密度函数
    x = Symbol('x', real=True)  # 定义实数符号变量 x
    assert (pdf(x) ==
            2**S.Half*exp(-(x - mu)**2/(2*sigma**2))/(2*pi**S.Half*sigma))  # 断言 Y 的概率密度函数满足正态分布的定义

    assert P(X**2 < 1) == erf(2**S.Half/2)  # 断言 X^2 < 1 的概率为 erf(2^0.5 / 2)
    ans = quantile(Y)(x)  # 计算 Y 的分位数函数在 x 处的值
    assert ans == Complement(Intersection(FiniteSet(
        sqrt(2)*sigma*(sqrt(2)*mu/(2*sigma)+ erfinv(2*x - 1))),
        Interval(-oo, oo)), FiniteSet(mu))  # 断言 Y 的分位数函数的计算结果
    assert E(X, Eq(X, mu)) == mu  # 断言在条件 X = mu 下，X 的期望值为 mu

    assert median(X) == FiniteSet(0)  # 断言 X 的中位数为 0
    # issue 8248
    assert X.pspace.compute_expectation(1).doit() == 1  # 断言 X 的期望值为 1


def test_conditional_1d():
    X = Normal('x', 0, 1)  # 创建均值为0，方差为1的正态分布随机变量 X
    Y = given(X, X >= 0)  # 创建在 X >= 0 条件下的随机变量 Y
    z = Symbol('z')  # 定义符号变量 z

    assert density(Y)(z) == 2 * density(X)(z)  # 断言 Y 的概率密度函数等于 2 倍 X 的概率密度函数

    assert Y.pspace.domain.set == Interval(0, oo)  # 断言 Y 的概率空间的定义域为 [0, ∞)
    assert E(Y) == sqrt(2) / sqrt(pi)  # 断言 Y 的期望值为 sqrt(2) / sqrt(pi)

    assert E(X**2) == E(Y**2)  # 断言 X^2 的期望值等于 Y^2 的期望值


def test_ContinuousDomain():
    X = Normal('x', 0, 1)  # 创建均值为0，方差为1的正态分布随机变量 X
    assert where(X**2 <= 1).set == Interval(-1, 1)  # 断言 X^2 <= 1 的定义域为 [-1, 1]
    assert where(X**2 <= 1).symbol == X.symbol  # 断言 X^2 <= 1 的符号为 X.symbol
    assert where(And(X**2 <= 1, X >= 0)).set == Interval(0, 1)  # 断言 X^2 <= 1 且 X >= 0 的定义域为 [0, 1]
    raises(ValueError, lambda: where(sin(X) > 1))  # 断言当 sin(X) > 1 时引发 ValueError 异常

    Y = given(X, X >= 0)  # 创建在 X >= 0 条件下的随机变量 Y

    assert Y.pspace.domain.set == Interval(0, oo)  # 断言 Y 的概率空间的定义域为 [0, ∞)


def test_multiple_normal():
    X, Y = Normal('x', 0, 1), Normal('y', 0, 1)  # 创建均值为0，方差为1的正态分布随机变量 X, Y
    p = Symbol("p", positive=True)  # 定义正数符号变量 p

    assert E(X + Y) == 0  # 断言 X + Y 的期望值为 0
    assert variance(X + Y) == 2  # 断言 X + Y 的方差为 2
    assert variance(X + X) == 4  # 断言 X + X 的方差为 4
    assert covariance(X, Y) == 0  # 断言 X, Y 的协方差为 0
    assert covariance(2*X + Y, -X) == -2*variance(X)  # 断言 2*X + Y 和 -X 的协方差等于 -2*variance(X)
    assert skewness(X) == 0  # 断言 X 的偏度为 0
    assert skewness(X + Y) == 0  # 断言 X + Y 的偏度为 0
    assert kurtosis(X) == 3  # 断言 X 的峰度为 3
    assert kurtosis(X+Y) == 3  # 断言 X + Y 的峰度为 3
    assert correlation(X, Y) == 0  # 断言 X, Y 的相关系数为 0
    assert correlation(X, X + Y) == correlation(X, X - Y)  # 断言 X, X + Y 和 X, X - Y 的相关系数相等
    assert moment(X, 2) == 1  # 断言 X 的二阶矩为 1
    assert cmoment(X, 3) == 0  # 断言 X 的中心矩为 0
    assert moment(X + Y, 4) == 12  # 断言 X + Y 的四阶矩为 12
    assert cmoment(X, 2) == variance(X)  # 断言 X 的二阶中心矩等于方差
    assert smoment(X*X, 2) == 1  # 断言 X^2 的二阶标准化矩为 1
    assert smoment(X + Y, 3) == skewness(X + Y)  # 断言 X + Y 的三阶标准化矩等于偏度
    assert smoment(X + Y, 4) == kurtosis(X + Y)  # 断言 X + Y 的四阶标准化矩等于峰度
    assert E(X, Eq(X + Y, 0)) == 0  # 断言在条件 X + Y = 0 下，X 的期望值为 0
    assert variance(X, Eq(X + Y, 0)) == S.Half  # 断言在条件 X + Y = 0 下，X 的方差为 1/2
    assert quantile(X)(p) == sqrt(2)*erfinv(2*p - S.One)  # 断言 X 的分位数函数在 p 处的值为 sqrt(2)*erfinv(2*p - 1)


def test_symbolic():
    mu1, mu2 = symbols('mu1 mu2', real=True)  # 定义实数符号变量 mu1, mu2
    s1, s
    # 断言：验证方差函数计算结果是否等于 s1 的平方
    assert variance(X) == s1**2
    
    # 断言：验证方差函数对表达式 X + a*Y + b 的结果是否等于方差 X 的值加上 a^2 * 方差 Y 的值
    assert variance(X + a*Y + b) == variance(X) + a**2 * variance(Y)
    
    # 断言：验证随机变量 Z 的期望是否等于 1 除以速率 rate
    assert E(Z) == 1/rate
    
    # 断言：验证随机变量 a*Z + b 的期望是否等于 a 乘以 Z 的期望再加上 b
    assert E(a*Z + b) == a * E(Z) + b
    
    # 断言：验证随机变量 X + a*Z + b 的期望是否等于 mu1 加上 a 除以速率 rate 再加上 b
    assert E(X + a*Z + b) == mu1 + a/rate + b
    
    # 断言：验证随机变量 X 的中位数是否等于包含 mu1 的有限集合 FiniteSet(mu1)
    assert median(X) == FiniteSet(mu1)
# 定义一个测试函数 `test_cdf`
def test_cdf():
    # 创建一个正态分布随机变量 X，均值为 0，标准差为 1
    X = Normal('x', 0, 1)

    # 计算 X 的累积分布函数并赋给 d
    d = cdf(X)
    # 断言 P(X < 1) 等于 d(1) 重写后的值，使用 erfc 函数
    assert P(X < 1) == d(1).rewrite(erfc)
    # 断言 d(0) 等于 S.Half
    assert d(0) == S.Half

    # 给定条件 X > 0，计算 X 的累积分布函数并赋给 d
    d = cdf(X, X > 0)
    # 断言 d(0) 等于 0
    assert d(0) == 0

    # 创建一个参数为 10 的指数分布随机变量 Y
    Y = Exponential('y', 10)
    # 计算 Y 的累积分布函数并赋给 d
    d = cdf(Y)
    # 断言 d(-5) 等于 0
    assert d(-5) == 0
    # 断言 P(Y > 3) 等于 1 减去 d(3)
    assert P(Y > 3) == 1 - d(3)

    # 断言调用 cdf(X + Y) 时会引发 ValueError 异常
    raises(ValueError, lambda: cdf(X + Y))

    # 创建一个参数为 1 的指数分布随机变量 Z
    Z = Exponential('z', 1)
    # 计算 Z 的累积分布函数并赋给 f
    f = cdf(Z)
    # 断言 f(z) 的值为 Piecewise((1 - exp(-z), z >= 0), (0, True))
    assert f(z) == Piecewise((1 - exp(-z), z >= 0), (0, True))


# 定义一个测试函数 `test_characteristic_function`
def test_characteristic_function():
    # 创建一个在区间 [0, 1] 上的均匀分布随机变量 X
    X = Uniform('x', 0, 1)

    # 计算 X 的特征函数并赋给 cf
    cf = characteristic_function(X)
    # 断言 cf(1) 的值
    assert cf(1) == -I*(-1 + exp(I))

    # 创建一个均值为 1，标准差为 1 的正态分布随机变量 Y
    Y = Normal('y', 1, 1)
    # 计算 Y 的特征函数并赋给 cf
    cf = characteristic_function(Y)
    # 断言 cf(0) 的值为 1
    assert cf(0) == 1
    # 断言 cf(1) 的值为 exp(I - S.Half)
    assert cf(1) == exp(I - S.Half)

    # 创建一个参数为 5 的指数分布随机变量 Z
    Z = Exponential('z', 5)
    # 计算 Z 的特征函数并赋给 cf
    cf = characteristic_function(Z)
    # 断言 cf(0) 的值为 1
    assert cf(0) == 1
    # 断言 cf(1).expand() 的值为 Rational(25, 26) + I*5/26
    assert cf(1).expand() == Rational(25, 26) + I*5/26

    # 创建一个参数为 1 的反高斯分布随机变量 X
    X = GaussianInverse('x', 1, 1)
    # 计算 X 的特征函数并赋给 cf
    cf = characteristic_function(X)
    # 断言 cf(0) 的值为 1
    assert cf(0) == 1
    # 断言 cf(1) 的值为 exp(1 - sqrt(1 - 2*I))
    assert cf(1) == exp(1 - sqrt(1 - 2*I))

    # 创建一个参数为 0 和 1 的指数高斯分布随机变量 X
    X = ExGaussian('x', 0, 1, 1)
    # 计算 X 的特征函数并赋给 cf
    cf = characteristic_function(X)
    # 断言 cf(0) 的值为 1
    assert cf(0) == 1
    # 断言 cf(1) 的值为 (1 + I)*exp(Rational(-1, 2))/2
    assert cf(1) == (1 + I)*exp(Rational(-1, 2))/2

    # 创建一个参数为 0 和 1 的莱维分布随机变量 L
    L = Levy('x', 0, 1)
    # 计算 L 的特征函数并赋给 cf
    cf = characteristic_function(L)
    # 断言 cf(0) 的值为 1
    assert cf(0) == 1
    # 断言 cf(1) 的值为 exp(-sqrt(2)*sqrt(-I))


# 定义一个测试函数 `test_moment_generating_function`
def test_moment_generating_function():
    # 声明一个正符号的符号变量 t
    t = symbols('t', positive=True)

    # 声明符号变量 a, b, c
    a, b, c = symbols('a b c')

    # 计算 Beta 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(Beta('x', a, b))(t)
    # 断言 mgf 的值为超几何函数 hyper((a,), (a + b,), t)
    assert mgf == hyper((a,), (a + b,), t)

    # 计算 Chi 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(Chi('x', a))(t)
    # 断言 mgf 的值为复杂表达式

    # 计算 ChiSquared 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(ChiSquared('x', a))(t)
    # 断言 mgf 的值为复杂表达式

    # 计算 Erlang 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(Erlang('x', a, b))(t)
    # 断言 mgf 的值为简单表达式

    # 计算 ExGaussian 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(ExGaussian("x", a, b, c))(t)
    # 断言 mgf 的值为复杂表达式

    # 计算 Exponential 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(Exponential('x', a))(t)
    # 断言 mgf 的值为简单表达式

    # 计算 Gamma 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(Gamma('x', a, b))(t)
    # 断言 mgf 的值为简单表达式

    # 计算 Gumbel 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(Gumbel('x', a, b))(t)
    # 断言 mgf 的值为复杂表达式

    # 计算 Gompertz 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(Gompertz('x', a, b))(t)
    # 断言 mgf 的值为复杂表达式

    # 计算 Laplace 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(Laplace('x', a, b))(t)
    # 断言 mgf 的值为复杂表达式

    # 计算 Logistic 分布的矩生成函数并在 t 处求值，并赋给 mgf
    mgf = moment_generating_function(Log
    # 计算 RaisedCosine 分布的矩生成函数，并在 t 处求导
    mgf = moment_generating_function(RaisedCosine('x', 1, 1))(t)
    # 断言检查矩生成函数的值是否与预期相等
    assert mgf.diff(t).subs(t, 1) == -2*e*pi**2*sinh(1)/\
    (1 + pi**2)**2 + e*pi**2*cosh(1)/(1 + pi**2)

    # 计算 Rayleigh 分布的矩生成函数，并在 t 处求导
    mgf = moment_generating_function(Rayleigh('x', 1))(t)
    # 断言检查矩生成函数的值是否与预期相等
    assert mgf.diff(t).subs(t, 0) == sqrt(2)*sqrt(pi)/2

    # 计算 Triangular 分布的矩生成函数，返回其字符串表示形式
    mgf = moment_generating_function(Triangular('x', 1, 3, 2))(t)
    # 断言：计算 mgf 在 t = 1 处的导数并检查是否等于 -e + exp(3)
    assert mgf.diff(t).subs(t, 1) == -e + exp(3)

    # 计算均匀分布 Uniform('x', 0, 1) 的矩生成函数，并在 t = 1 处检查其导数是否等于 1
    mgf = moment_generating_function(Uniform('x', 0, 1))(t)
    assert mgf.diff(t).subs(t, 1) == 1

    # 计算均匀和分布 UniformSum('x', 1) 的矩生成函数，并在 t = 1 处检查其导数是否等于 1
    mgf = moment_generating_function(UniformSum('x', 1))(t)
    assert mgf.diff(t).subs(t, 1) == 1

    # 计算 Wigner半圆分布 WignerSemicircle('x', 1) 的矩生成函数，并在 t = 1 处检查其导数是否满足特定条件
    assert mgf.diff(t).subs(t, 1) == -2*besseli(1, 1) + besseli(2, 1) +\
        besseli(0, 1)


这些代码片段计算了不同概率分布（均匀分布、和分布、Wigner半圆分布）的矩生成函数在特定点 t = 1 处的导数，并使用断言来验证导数的准确性或特定的数值关系。
def test_ContinuousRV():
    # 正态分布的概率密度函数
    pdf = sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))  # Normal distribution
    # X 和 Y 应该是等价的
    X = ContinuousRV(x, pdf, check=True)
    # 创建一个标准正态分布对象 Y
    Y = Normal('y', 0, 1)

    # 断言 X 和 Y 的方差相等
    assert variance(X) == variance(Y)
    # 断言 X 大于 0 的概率等于 Y 大于 0 的概率
    assert P(X > 0) == P(Y > 0)
    
    # 创建一个指数分布对象 Z
    Z = ContinuousRV(z, exp(-z), set=Interval(0, oo))
    # 断言 Z 的取值范围为 [0, oo)
    assert Z.pspace.domain.set == Interval(0, oo)
    # 断言 Z 的期望值为 1
    assert E(Z) == 1
    # 断言 Z 大于 5 的概率为 exp(-5)
    assert P(Z > 5) == exp(-5)
    
    # 测试带有范围检查的 ContinuousRV 构造函数，应该引发 ValueError
    raises(ValueError, lambda: ContinuousRV(z, exp(-z), set=Interval(0, 10), check=True))

    # Gamma 分布的概率密度函数
    _x, k, theta = symbols("x k theta", positive=True)
    pdf = 1/(gamma(k)*theta**k)*_x**(k-1)*exp(-_x/theta)
    # 创建一个 Gamma 分布对象 X
    X = ContinuousRV(_x, pdf, set=Interval(0, oo))
    # 创建一个 Gamma 分布对象 Y
    Y = Gamma('y', k, theta)
    # 断言 X 的期望值与 Y 的期望值化简后相等
    assert (E(X) - E(Y)).simplify() == 0
    # 断言 X 的方差与 Y 的方差化简后相等
    assert (variance(X) - variance(Y)).simplify() == 0


def test_arcsin():
    # 创建两个实数符号对象 a 和 b
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)

    # 创建一个 Arcsin 分布对象 X
    X = Arcsin('x', a, b)
    # 断言 X 的概率密度函数
    assert density(X)(x) == 1/(pi*sqrt((-x + b)*(x - a)))
    # 断言 X 的累积分布函数
    assert cdf(X)(x) == Piecewise((0, a > x),
                            (2*asin(sqrt((-a + x)/(-a + b)))/pi, b >= x),
                            (1, True))
    # 断言 X 的取值范围为 [a, b]
    assert pspace(X).domain.set == Interval(a, b)


def test_benini():
    # 创建三个正数符号对象 alpha, beta, sigma
    alpha = Symbol("alpha", positive=True)
    beta = Symbol("beta", positive=True)
    sigma = Symbol("sigma", positive=True)
    # 创建一个 Benini 分布对象 X
    X = Benini('x', alpha, beta, sigma)

    # 断言 X 的概率密度函数
    assert density(X)(x) == ((alpha/x + 2*beta*log(x/sigma)/x)
                          *exp(-alpha*log(x/sigma) - beta*log(x/sigma)**2))

    # 断言 X 的取值范围为 [sigma, oo)
    assert pspace(X).domain.set == Interval(sigma, oo)
    # 测试未实现的函数，应该引发 NotImplementedError
    raises(NotImplementedError, lambda: moment_generating_function(X))

    # 测试构造函数参数错误，应该引发 ValueError
    alpha = Symbol("alpha", nonpositive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

    beta = Symbol("beta", nonpositive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

    alpha = Symbol("alpha", positive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

    beta = Symbol("beta", positive=True)
    sigma = Symbol("sigma", nonpositive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))


def test_beta():
    # 创建两个正数符号对象 a 和 b
    a, b = symbols('alpha beta', positive=True)
    # 创建一个 Beta 分布对象 B
    B = Beta('x', a, b)

    # 断言 B 的取值范围为 [0, 1]
    assert pspace(B).domain.set == Interval(0, 1)
    # 断言 B 的特征函数
    assert characteristic_function(B)(x) == hyper((a,), (a + b,), I*x)
    # 断言 B 的概率密度函数
    assert density(B)(x) == x**(a - 1)*(1 - x)**(b - 1)/beta(a, b)

    # 断言 B 的期望值化简后与 a / (a + b) 相等
    assert simplify(E(B)) == a / (a + b)
    # 断言 B 的方差化简后与 a*b / ((a + b)**2 * (a + b + 1)) 相等
    assert simplify(variance(B)) == (a*b) / ((a + b)**2 * (a + b + 1))

    # 使用数值版本进行测试
    a, b = 1, 2
    B = Beta('x', a, b)
    # 断言 B 的期望值的展开与数值版本相等
    assert expand_func(E(B)) == a / S(a + b)
    # 断言 B 的方差的展开与数值版本相等
    assert expand_func(variance(B)) == (a*b) / S((a + b)**2 * (a + b + 1))
    # 断言 B 的中位数为 FiniteSet(1 - 1/sqrt(2))

def test_beta_noncentral():
    # 这个函数未提供完整的代码，因此不需要进一步的注释
    # 定义两个正值符号 a 和 b
    a, b = symbols('a b', positive=True)
    # 定义一个非负值符号 c
    c = Symbol('c', nonnegative=True)
    # 创建一个虚拟符号 _k
    _k = Dummy('k')

    # 创建一个 Beta 非中心分布对象 X，参数为 a, b, c
    X = BetaNoncentral('x', a, b, c)

    # 断言 X 的样本空间为 [0, 1] 的区间
    assert pspace(X).domain.set == Interval(0, 1)

    # 计算 X 的概率密度函数
    dens = density(X)
    # 创建一个符号 z
    z = Symbol('z')

    # 计算 Beta 非中心分布的概率密度函数的表达式
    res = Sum( z**(_k + a - 1)*(c/2)**_k*(1 - z)**(b - 1)*exp(-c/2)/
               (beta(_k + a, b)*factorial(_k)), (_k, 0, oo))
    # 断言 dens(z) 等于 res
    assert dens(z).dummy_eq(res)

    # 断言如果无法确定符号的假设条件，BetaCentral 不应该抛出异常
    a, b, c = symbols('a b c')
    assert BetaNoncentral('x', a, b, c)

    # 创建一个非正值的符号 a，预期抛出 ValueError 异常
    a = Symbol('a', positive=False, real=True)
    raises(ValueError, lambda: BetaNoncentral('x', a, b, c))

    # 创建一个正值的符号 a 和一个非正值的符号 b，预期抛出 ValueError 异常
    a = Symbol('a', positive=True)
    b = Symbol('b', positive=False, real=True)
    raises(ValueError, lambda: BetaNoncentral('x', a, b, c))

    # 创建两个正值符号 a 和 b，以及一个非负值符号 c，预期抛出 ValueError 异常
    a = Symbol('a', positive=True)
    b = Symbol('b', positive=True)
    c = Symbol('c', nonnegative=False, real=True)
    raises(ValueError, lambda: BetaNoncentral('x', a, b, c))
# 定义测试函数 test_betaprime，用于测试 BetaPrime 分布的相关功能
def test_betaprime():
    # 声明一个正数符号变量 alpha
    alpha = Symbol("alpha", positive=True)

    # 声明一个正数符号变量 betap
    betap = Symbol("beta", positive=True)

    # 创建一个 BetaPrime 分布对象 X，以 alpha 和 betap 作为参数
    X = BetaPrime('x', alpha, betap)

    # 断言 BetaPrime 分布 X 的概率密度函数为指定的公式
    assert density(X)(x) == x**(alpha - 1)*(x + 1)**(-alpha - betap)/beta(alpha, betap)

    # 更改 alpha 为非正数，应该引发 ValueError 异常
    alpha = Symbol("alpha", nonpositive=True)
    raises(ValueError, lambda: BetaPrime('x', alpha, betap))

    # 将 alpha 重新设置为正数，但将 betap 设置为非正数，应该引发 ValueError 异常
    alpha = Symbol("alpha", positive=True)
    betap = Symbol("beta", nonpositive=True)
    raises(ValueError, lambda: BetaPrime('x', alpha, betap))

    # 创建一个 BetaPrime 分布对象 X，以 1 和 1 作为参数
    X = BetaPrime('x', 1, 1)

    # 断言 BetaPrime 分布 X 的中位数为有限集 {1}
    assert median(X) == FiniteSet(1)


# 定义测试函数 test_BoundedPareto，用于测试 BoundedPareto 分布的相关功能
def test_BoundedPareto():
    # 声明 L 和 H 符号变量，其值为负数，应该引发 ValueError 异常
    L, H = symbols('L, H', negative=True)
    raises(ValueError, lambda: BoundedPareto('X', 1, L, H))

    # 声明 L 和 H 符号变量，其值为非实数，应该引发 ValueError 异常
    L, H = symbols('L, H', real=False)
    raises(ValueError, lambda: BoundedPareto('X', 1, L, H))

    # 声明 L 和 H 符号变量，其值为正数，但参数 -1 不合法，应该引发 ValueError 异常
    L, H = symbols('L, H', positive=True)
    raises(ValueError, lambda: BoundedPareto('X', -1, L, H))

    # 创建一个 BoundedPareto 分布对象 X，以 2、L 和 H 作为参数
    X = BoundedPareto('X', 2, L, H)

    # 断言 BoundedPareto 分布 X 的概率空间域为指定的区间 [L, H]
    assert X.pspace.domain.set == Interval(L, H)

    # 断言 BoundedPareto 分布 X 的概率密度函数为指定的公式
    assert density(X)(x) == 2*L**2/(x**3*(1 - L**2/H**2))

    # 断言 BoundedPareto 分布 X 的累积分布函数为指定的分段函数
    assert cdf(X)(x) == Piecewise((-H**2*L**2/(x**2*(H**2 - L**2)) + H**2/(H**2 - L**2), L <= x), (0, True))

    # 断言 BoundedPareto 分布 X 的期望值经简化后与指定值相等
    assert E(X).simplify() == 2*H*L/(H + L)

    # 创建一个 BoundedPareto 分布对象 X，以 1、2 和 4 作为参数
    X = BoundedPareto('X', 1, 2, 4)

    # 断言 BoundedPareto 分布 X 的期望值经简化后与 log(16) 相等
    assert E(X).simplify() == log(16)

    # 断言 BoundedPareto 分布 X 的中位数为有限集 {8/3}
    assert median(X) == FiniteSet(Rational(8, 3))

    # 断言 BoundedPareto 分布 X 的方差经简化后与指定值相等
    assert variance(X).simplify() == 8 - 16*log(2)**2


# 定义测试函数 test_cauchy，用于测试 Cauchy 分布的相关功能
def test_cauchy():
    # 声明一个实数符号变量 x0
    x0 = Symbol("x0", real=True)

    # 声明一个正数符号变量 gamma
    gamma = Symbol("gamma", positive=True)

    # 声明一个正数符号变量 p
    p = Symbol("p", positive=True)

    # 创建一个 Cauchy 分布对象 X，以 x0 和 gamma 作为参数
    X = Cauchy('x', x0, gamma)

    # 断言 Cauchy 分布 X 的特征函数为指定的公式
    assert characteristic_function(X)(x) == exp(-gamma*Abs(x) + I*x*x0)

    # 测试未实现的生成函数，应该引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: moment_generating_function(X))

    # 断言 Cauchy 分布 X 的概率密度函数为指定的公式
    assert density(X)(x) == 1/(pi*gamma*(1 + (x - x0)**2/gamma**2))

    # 断言 Cauchy 分布 X 的累积分布函数的导数与概率密度函数相等
    assert diff(cdf(X)(x), x) == density(X)(x)

    # 断言 Cauchy 分布 X 的分位数函数为指定的公式
    assert quantile(X)(p) == gamma*tan(pi*(p - S.Half)) + x0

    # 声明一个非实数符号变量 x1，应该引发 ValueError 异常
    x1 = Symbol("x1", real=False)
    raises(ValueError, lambda: Cauchy('x', x1, gamma))

    # 更改 gamma 为非正数，应该引发 ValueError 异常
    gamma = Symbol("gamma", nonpositive=True)
    raises(ValueError, lambda: Cauchy('x', x0, gamma))

    # 断言 Cauchy 分布 X 的中位数为有限集 {x0}
    assert median(X) == FiniteSet(x0)


# 定义测试函数 test_chi，用于测试 Chi 分布的相关功能
def test_chi():
    # 导入符号 k，并声明其为整数
    from sympy.core.numbers import I
    k = Symbol("k", integer=True)

    # 创建一个 Chi 分布对象 X，以 k 作为参数
    X = Chi('x', k)

    # 断言 Chi 分布 X 的概率密度函数为指定的公式
    assert density(X)(x) == 2**(-k/2 + 1)*x**(k - 1)*exp(-x**2/2)/gamma(k/2)

    # 断言 Chi 分布 X 的特征函数为指定的公式
    assert characteristic_function(X)(x) == sqrt(2)*I*x*gamma(k/2 + S(1)/2)*hyper((k/2 + S(1)/2,),
                                            (S(3)/2,), -x**2/2)/gamma(k/2) + hyper((k/2,), (S(1)/2,), -x**2/2)

    # 断言 Chi 分布 X 的生成函数为指定的公式
    assert moment_generating_function(X)(x) == sqrt(2)*x*gamma(k/2 + S(1)/2)*hyper((k/2 + S(1)/2,),
                                                (S(3)/2,), x**2/2)/gamma(k/2) + hyper((k/2,), (S(1)/2,), x**2/2)

    # 更改 k 为非正数，应该引发 ValueError 异常
    k = Symbol("k", integer=True, positive=False)
    raises(ValueError, lambda: Chi('x', k))

    # 更改 k 为非整数，但为正数，应该引发 ValueError 异常
    k = Symbol("k", integer=False, positive=True)
    # 使用 raises 函数验证是否会抛出 ValueError 异常，预期调用 Chi 类的初始化方法并传入参数 'x' 和 k
    raises(ValueError, lambda: Chi('x', k))
def test_chi_noncentral():
    # 定义符号变量 k 为整数
    k = Symbol("k", integer=True)
    # 定义符号变量 l
    l = Symbol("l")

    # 创建 ChiNoncentral 分布对象 X，使用符号变量 k 和 l
    X = ChiNoncentral("x", k, l)
    # 断言：X 的概率密度函数与预期表达式相等
    assert density(X)(x) == (x**k * l * (x*l)**(-k/2) *
                             exp(-x**2/2 - l**2/2) * besseli(k/2 - 1, x*l))

    # 设置 k 为负整数，应该引发 ValueError 异常
    k = Symbol("k", integer=True, positive=False)
    raises(ValueError, lambda: ChiNoncentral('x', k, l))

    # 设置 k 为正整数，l 为非正数，应该引发 ValueError 异常
    k = Symbol("k", integer=True, positive=True)
    l = Symbol("l", nonpositive=True)
    raises(ValueError, lambda: ChiNoncentral('x', k, l))

    # 设置 k 为非整数，l 为正数，应该引发 ValueError 异常
    k = Symbol("k", integer=False)
    l = Symbol("l", positive=True)
    raises(ValueError, lambda: ChiNoncentral('x', k, l))


def test_chi_squared():
    # 定义符号变量 k 为整数
    k = Symbol("k", integer=True)
    # 创建 ChiSquared 分布对象 X，使用符号变量 k
    X = ChiSquared('x', k)

    # 断言：X 的特征函数与预期表达式相等
    assert characteristic_function(X)(x) == ((-2*I*x + 1)**(-k/2))

    # 断言：X 的概率密度函数与预期表达式相等
    assert density(X)(x) == 2**(-k/2) * x**(k/2 - 1) * exp(-x/2) / gamma(k/2)
    # 断言：X 的累积分布函数与预期表达式相等
    assert cdf(X)(x) == Piecewise((lowergamma(k/2, x/2) / gamma(k/2), x >= 0), (0, True))
    # 断言：X 的期望值与预期值相等
    assert E(X) == k
    # 断言：X 的方差与预期值相等
    assert variance(X) == 2*k

    # 设置 k 为负整数，应该引发 ValueError 异常
    k = Symbol("k", integer=True, positive=False)
    raises(ValueError, lambda: ChiSquared('x', k))

    # 设置 k 为非整数，但为正数，应该引发 ValueError 异常
    k = Symbol("k", integer=False, positive=True)
    raises(ValueError, lambda: ChiSquared('x', k))


def test_dagum():
    # 定义符号变量 p, a, b，均为正数
    p = Symbol("p", positive=True)
    b = Symbol("b", positive=True)
    a = Symbol("a", positive=True)

    # 创建 Dagum 分布对象 X，使用符号变量 p, a, b
    X = Dagum('x', p, a, b)
    # 断言：X 的概率密度函数与预期表达式相等
    assert density(X)(x) == a * p * (x/b)**(a*p) * ((x/b)**a + 1)**(-p - 1) / x
    # 断言：X 的累积分布函数与预期表达式相等
    assert cdf(X)(x) == Piecewise(((1 + (x/b)**(-a))**(-p), x >= 0),
                                  (0, True))

    # 设置 p 为非正数，应该引发 ValueError 异常
    p = Symbol("p", nonpositive=True)
    raises(ValueError, lambda: Dagum('x', p, a, b))

    # 设置 p 为正数，b 为非正数，应该引发 ValueError 异常
    p = Symbol("p", positive=True)
    b = Symbol("b", nonpositive=True)
    raises(ValueError, lambda: Dagum('x', p, a, b))

    # 设置 a 为非正数，应该引发 ValueError 异常
    b = Symbol("b", positive=True)
    a = Symbol("a", nonpositive=True)
    raises(ValueError, lambda: Dagum('x', p, a, b))
    
    # 创建 Dagum 分布对象 X，使用具体参数 1, 1, 1，断言中位数为有限集合 {1}
    X = Dagum('x', 1, 1, 1)
    assert median(X) == FiniteSet(1)


def test_davis():
    # 定义符号变量 b, n, mu，均为正数
    b = Symbol("b", positive=True)
    n = Symbol("n", positive=True)
    mu = Symbol("mu", positive=True)

    # 创建 Davis 分布对象 X，使用符号变量 b, n, mu
    X = Davis('x', b, n, mu)
    # 计算分子和分母，分别为概率密度函数的分子和分母
    dividend = b**n * (x - mu)**(-1 - n)
    divisor = (exp(b / (x - mu)) - 1) * (gamma(n) * zeta(n))
    # 断言：X 的概率密度函数与预期表达式相等
    assert density(X)(x) == dividend / divisor


def test_erlang():
    # 定义符号变量 k 为正整数，l 为正数
    k = Symbol("k", integer=True, positive=True)
    l = Symbol("l", positive=True)

    # 创建 Erlang 分布对象 X，使用符号变量 k, l
    X = Erlang("x", k, l)
    # 断言：X 的概率密度函数与预期表达式相等
    assert density(X)(x) == x**(k - 1) * l**k * exp(-x * l) / gamma(k)
    # 断言：X 的累积分布函数与预期表达式相等
    assert cdf(X)(x) == Piecewise((lowergamma(k, l * x) / gamma(k), x > 0),
                                  (0, True))


def test_exgaussian():
    # 定义符号变量 m, z，s, l 为正数
    m, z = symbols("m, z")
    s, l = symbols("s, l", positive=True)
    # 创建 ExGaussian 分布对象 X，使用符号变量 m, s, l
    X = ExGaussian("x", m, s, l)

    # 断言：X 的概率密度函数与预期表达式相等
    assert density(X)(z) == l * exp(l * (l * s**2 + 2 * m - 2 * z) / 2) * \
        erfc(sqrt(2) * (l * s**2 + m - z) / (2 * s)) / 2

    # 注意：actual_output simplifies to expected_output.
    # 计算 u 和 v 的值，用于后续的高斯累积分布函数计算
    u = l*(z - m)  # u 的计算
    v = l*s        # v 的计算

    # 计算高斯累积分布函数的值
    # GaussianCDF1 是正态分布 N(0, v) 的累积分布函数在 u 处的值
    GaussianCDF1 = cdf(Normal('x', 0, v))(u)
    # GaussianCDF2 是正态分布 N(v^2, v) 的累积分布函数在 u 处的值
    GaussianCDF2 = cdf(Normal('x', v**2, v))(u)

    # 计算实际输出值，这里是 cdf(X)(z) 的预期输出
    actual_output = GaussianCDF1 - exp(-u + (v**2/2) + log(GaussianCDF2))

    # 断言 cdf(X)(z) 等于实际计算的输出值
    assert cdf(X)(z) == actual_output

    # 断言方差的展开式是否等于 s^2 + l^(-2)
    assert variance(X).expand() == s**2 + l**(-2)

    # 断言偏度的展开式是否等于给定的表达式
    assert skewness(X).expand() == 2/(l**3*s**2*sqrt(s**2 + l**(-2)) + l *
                                      sqrt(s**2 + l**(-2)))
@slow
# 定义一个用于测试指数分布的测试函数，由于涉及数学统计，可能运行较慢

def test_exponential():
    # 定义一个正的符号lambda，用于指数分布的速率参数
    rate = Symbol('lambda', positive=True)
    # 创建一个指数分布随机变量X，其速率为rate
    X = Exponential('x', rate)
    # 定义一个正的实数符号p，用于后续概率计算

    p = Symbol("p", positive=True, real=True)

    # 断言指数分布的期望值为1/rate
    assert E(X) == 1/rate
    # 断言指数分布的方差为1/rate^2
    assert variance(X) == 1/rate**2
    # 断言指数分布的偏度为2
    assert skewness(X) == 2
    # 断言指数分布的偏度与三阶中心矩相同
    assert skewness(X) == smoment(X, 3)
    # 断言指数分布的峰度为9
    assert kurtosis(X) == 9
    # 断言指数分布的峰度与四阶中心矩相同
    assert kurtosis(X) == smoment(X, 4)
    # 断言2*X的四阶中心矩等于X的四阶中心矩
    assert smoment(2*X, 4) == smoment(X, 4)
    # 断言X的三阶原点矩等于3*2*1/rate^3
    assert moment(X, 3) == 3*2*1/rate**3
    # 断言X大于0的概率为1
    assert P(X > 0) is S.One
    # 断言X大于1的概率为exp(-rate)
    assert P(X > 1) == exp(-rate)
    # 断言X大于10的概率为exp(-10*rate)
    assert P(X > 10) == exp(-10*rate)
    # 断言X的分位数函数在概率p处的值为-log(1-p)/rate

    assert quantile(X)(p) == -log(1-p)/rate

    # 断言X小于等于1的区间
    assert where(X <= 1).set == Interval(0, 1)
    # 创建一个速率为1的指数分布随机变量Y
    Y = Exponential('y', 1)
    # 断言Y的中位数为{log(2)}的有限集
    assert median(Y) == FiniteSet(log(2))
    # 测试问题9970
    z = Dummy('z')
    # 断言X大于z的概率为exp(-z*rate)
    assert P(X > z) == exp(-z*rate)
    # 断言X小于z的概率为0
    assert P(X < z) == 0
    # 测试问题10076（具有区间(0,oo)的分布）
    x = Symbol('x')
    _z = Dummy('_z')
    # 创建一个以x为参数，速率为2的单连续概率空间b
    b = SingleContinuousPSpace(x, ExponentialDistribution(2))

    # 忽略警告，恢复测试一旦警告被移除
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        # 期望1：2*exp(-2*_z)在(_z, 3, oo)上的积分
        expected1 = Integral(2*exp(-2*_z), (_z, 3, oo))
        # 断言b大于3的概率（不评估）等同于重写的积分
        assert b.probability(x > 3, evaluate=False).rewrite(Integral).dummy_eq(expected1)

        # 期望2：2*exp(-2*_z)在(_z, 0, 4)上的积分
        expected2 = Integral(2*exp(-2*_z), (_z, 0, 4))
        # 断言b小于4的概率（不评估）等同于重写的积分
        assert b.probability(x < 4, evaluate=False).rewrite(Integral).dummy_eq(expected2)

    # 创建一个速率为2*rate的指数分布随机变量Y
    Y = Exponential('y', 2*rate)
    # 断言X与X和X的共偏度等于X的偏度
    assert coskewness(X, X, X) == skewness(X)
    # 断言coskewness(X, Y + rate*X, Y + 2*rate*X)等于一个复杂的表达式
    assert coskewness(X, Y + rate*X, Y + 2*rate*X) == \
                        4/(sqrt(1 + 1/(4*rate**2))*sqrt(4 + 1/(4*rate**2)))
    # 断言coskewness(X + 2*Y, Y + X, Y + 2*X, X > 3)等于一个复杂的根式表达式
    assert coskewness(X + 2*Y, Y + X, Y + 2*X, X > 3) == \
                        sqrt(170)*Rational(9, 85)

def test_exponential_power():
    # 定义符号mu，z，alpha和beta，alpha和beta为正
    mu = Symbol('mu')
    z = Symbol('z')
    alpha = Symbol('alpha', positive=True)
    beta = Symbol('beta', positive=True)

    # 创建指数幂分布随机变量X，其参数为mu，alpha和beta
    X = ExponentialPower('x', mu, alpha, beta)

    # 断言X的概率密度函数在z处的值
    assert density(X)(z) == beta*exp(-(Abs(mu - z)/alpha)
                                     ** beta)/(2*alpha*gamma(1/beta))
    # 断言X的累积分布函数在z处的值
    assert cdf(X)(z) == S.Half + lowergamma(1/beta,
                            (Abs(mu - z)/alpha)**beta)*sign(-mu + z)/\
                                (2*gamma(1/beta))


def test_f_distribution():
    # 定义正的符号d1和d2，用于F分布的参数
    d1 = Symbol("d1", positive=True)
    d2 = Symbol("d2", positive=True)

    # 创建F分布随机变量X，其参数为d1和d2
    X = FDistribution("x", d1, d2)

    # 断言X的概率密度函数在x处的值
    assert density(X)(x) == (d2**(d2/2)*sqrt((d1*x)**d1*(d1*x + d2)**(-d1 - d2))
                             /(x*beta(d1/2, d2/2)))

    # 断言未实现矩生成函数
    raises(NotImplementedError, lambda: moment_generating_function(X))

    # 断言d1为非正时抛出错误
    d1 = Symbol("d1", nonpositive=True)
    raises(ValueError, lambda: FDistribution('x', d1, d1))

    # 断言d1为非整数时抛出错误
    d1 = Symbol("d1", positive=True, integer=False)
    raises(ValueError, lambda: FDistribution('x', d1, d1))

    # 断言d1为正，d2为非正时抛出错误
    d1 = Symbol("d1", positive=True)
    d2 = Symbol("d2", nonpositive=True)
    raises(ValueError, lambda: FDistribution('x', d1, d2))

    # 断言d2为正，非整数时抛出错误
    d2 = Symbol("d2", positive=True, integer=False)
    raises(ValueError, lambda: FDistribution('x', d1, d2))
    # 创建一个符号对象 d1，其值为正数
    d1 = Symbol("d1", positive=True)
    # 创建一个符号对象 d2，其值为正数
    d2 = Symbol("d2", positive=True)
    
    # 创建一个 FisherZ 对象 X，使用符号 d1 和 d2 作为参数
    X = FisherZ("x", d1, d2)
    
    # 断言语句，验证概率密度函数 density(X)(x) 的计算结果是否与给定的表达式相等
    assert density(X)(x) == (2*d1**(d1/2)*d2**(d2/2)*(d1*exp(2*x) + d2)**(-d1/2 - d2/2)*exp(d1*x)/beta(d1/2, d2/2))
def test_frechet():
    # 定义正数符号的符号变量
    a = Symbol("a", positive=True)
    s = Symbol("s", positive=True)
    # 定义实数符号的符号变量
    m = Symbol("m", real=True)

    # 创建弗雷歇特分布对象 X
    X = Frechet("x", a, s=s, m=m)
    # 断言弗雷歇特分布的概率密度函数
    assert density(X)(x) == a*((x - m)/s)**(-a - 1)*exp(-((x - m)/s)**(-a))/s
    # 断言弗雷歇特分布的累积分布函数
    assert cdf(X)(x) == Piecewise((exp(-((-m + x)/s)**(-a)), m <= x), (0, True))

@slow
def test_gamma():
    # 定义正数符号的符号变量
    k = Symbol("k", positive=True)
    theta = Symbol("theta", positive=True)

    # 创建伽玛分布对象 X
    X = Gamma('x', k, theta)

    # 断言伽玛分布的特征函数
    assert characteristic_function(X)(x) == ((-I*theta*x + 1)**(-k))

    # 断言伽玛分布的概率密度函数
    assert density(X)(x) == x**(k - 1)*theta**(-k)*exp(-x/theta)/gamma(k)
    # 断言伽玛分布的累积分布函数，使用 Meijer G 函数
    assert cdf(X, meijerg=True)(z) == Piecewise(
            (-k*lowergamma(k, 0)/gamma(k + 1) +
                k*lowergamma(k, z/theta)/gamma(k + 1), z >= 0),
            (0, True))

    # 断言伽玛分布的期望
    assert E(X) == moment(X, 1)

    # 重新定义正数符号的符号变量
    k, theta = symbols('k theta', positive=True)
    X = Gamma('x', k, theta)
    # 断言伽玛分布的期望
    assert E(X) == k*theta
    # 断言伽玛分布的方差
    assert variance(X) == k*theta**2
    # 断言伽玛分布的偏度
    assert skewness(X).expand() == 2/sqrt(k)
    # 断言伽玛分布的峰度
    assert kurtosis(X).expand() == 3 + 6/k

    # 创建另一个伽玛分布对象 Y
    Y = Gamma('y', 2*k, 3*theta)
    # 断言两个伽玛分布的共偏
    assert coskewness(X, theta*X + Y, k*X + Y).simplify() == \
        2*531441**(-k)*sqrt(k)*theta*(3*3**(12*k) - 2*531441**k) \
        /(sqrt(k**2 + 18)*sqrt(theta**2 + 18))

def test_gamma_inverse():
    # 定义正数符号的符号变量
    a = Symbol("a", positive=True)
    b = Symbol("b", positive=True)
    # 创建伽玛逆分布对象 X
    X = GammaInverse("x", a, b)
    # 断言伽玛逆分布的概率密度函数
    assert density(X)(x) == x**(-a - 1)*b**a*exp(-b/x)/gamma(a)
    # 断言伽玛逆分布的累积分布函数
    assert cdf(X)(x) == Piecewise((uppergamma(a, b/x)/gamma(a), x > 0), (0, True))
    # 断言伽玛逆分布的特征函数，引发 NotImplementedError
    raises(NotImplementedError, lambda: moment_generating_function(X))

def test_gompertz():
    # 定义正数符号的符号变量
    b = Symbol("b", positive=True)
    eta = Symbol("eta", positive=True)

    # 创建戈姆佩尔茨分布对象 X
    X = Gompertz("x", b, eta)

    # 断言戈姆佩尔茨分布的概率密度函数
    assert density(X)(x) == b*eta*exp(eta)*exp(b*x)*exp(-eta*exp(b*x))
    # 断言戈姆佩尔茨分布的累积分布函数
    assert cdf(X)(x) == 1 - exp(eta)*exp(-eta*exp(b*x))
    # 断言戈姆佩尔茨分布的导数等于概率密度函数
    assert diff(cdf(X)(x), x) == density(X)(x)


def test_gumbel():
    # 定义正数符号的符号变量和无特定属性的符号变量
    beta = Symbol("beta", positive=True)
    mu = Symbol("mu")
    x = Symbol("x")
    y = Symbol("y")
    # 创建古贝尔分布对象 X 和 Y
    X = Gumbel("x", beta, mu)
    Y = Gumbel("y", beta, mu, minimum=True)
    # 断言古贝尔分布 X 的概率密度函数
    assert density(X)(x).expand() == \
    exp(mu/beta)*exp(-x/beta)*exp(-exp(mu/beta)*exp(-x/beta))/beta
    # 断言古贝尔分布 Y 的概率密度函数
    assert density(Y)(y).expand() == \
    exp(-mu/beta)*exp(y/beta)*exp(-exp(-mu/beta)*exp(y/beta))/beta
    # 断言古贝尔分布 X 的累积分布函数
    assert cdf(X)(x).expand() == \
    exp(-exp(mu/beta)*exp(-x/beta))
    # 断言古贝尔分布 X 的特征函数
    assert characteristic_function(X)(x) == exp(I*mu*x)*gamma(-I*beta*x + 1)

def test_kumaraswamy():
    # 定义正数符号的符号变量
    a = Symbol("a", positive=True)
    b = Symbol("b", positive=True)

    # 创建库马拉斯瓦米分布对象 X
    X = Kumaraswamy("x", a, b)
    # 断言库马拉斯瓦米分布的概率密度函数
    assert density(X)(x) == x**(a - 1)*a*b*(-x**a + 1)**(b - 1)
    # 断言语句，用于验证累积分布函数 cdf(X)(x) 的返回值是否符合预期
    assert cdf(X)(x) == Piecewise((0, x < 0),
                                  (-(-x**a + 1)**b + 1, x <= 1),
                                  (1, True))
    # cdf(X)(x): 调用函数 cdf(X) 并传入参数 x，返回累积分布函数的值
    # Piecewise((0, x < 0), ...): Piecewise 对象，根据 x 的取值范围返回不同的值
    # (-(-x**a + 1)**b + 1, x <= 1): 当 x 在 0 到 1 之间时的表达式
    # 1: 当 x 大于 1 时的默认返回值
    # True: 默认条件，表示对所有 x 的取值都成立
# 定义一个测试函数，用于测试 Laplace 分布的性质
def test_laplace():
    # 定义符号变量 mu 和 b，其中 b 是正数
    mu = Symbol("mu")
    b = Symbol("b", positive=True)

    # 创建 Laplace 分布对象 X，参数为 'x'，均值 mu，尺度参数 b
    X = Laplace('x', mu, b)

    # 测试特征函数 characteristic_function(X)，断言其返回值
    assert characteristic_function(X)(x) == (exp(I*mu*x)/(b**2*x**2 + 1))

    # 测试概率密度函数 density(X)，断言其返回值
    assert density(X)(x) == exp(-Abs(x - mu)/b)/(2*b)

    # 测试累积分布函数 cdf(X)，使用 Piecewise 表示
    assert cdf(X)(x) == Piecewise((exp((-mu + x)/b)/2, mu > x),
                            (-exp((mu - x)/b)/2 + 1, True))

    # 创建具有指定均值向量和协方差矩阵的 Laplace 分布对象 X
    X = Laplace('x', [1, 2], [[1, 0], [0, 1]])
    # 断言 X 的分布类型为 MultivariateLaplaceDistribution
    assert isinstance(pspace(X).distribution, MultivariateLaplaceDistribution)

# 定义一个测试函数，用于测试 Levy 分布的性质
def test_levy():
    # 定义符号变量 mu 和 c，其中 mu 是实数，c 是正数
    mu = Symbol("mu", real=True)
    c = Symbol("c", positive=True)

    # 创建 Levy 分布对象 X，参数为 'x'，均值 mu，尺度参数 c
    X = Levy('x', mu, c)

    # 断言 X 的定义域为区间 [mu, oo)
    assert X.pspace.domain.set == Interval(mu, oo)

    # 测试概率密度函数 density(X)，断言其返回值
    assert density(X)(x) == sqrt(c/(2*pi))*exp(-c/(2*(x - mu)))/((x - mu)**(S.One + S.Half))

    # 测试累积分布函数 cdf(X)，断言其返回值
    assert cdf(X)(x) == erfc(sqrt(c/(2*(x - mu))))

    # 使用 lambda 函数断言调用未实现的矩生成函数 moment_generating_function(X) 会引发 NotImplementedError
    raises(NotImplementedError, lambda: moment_generating_function(X))

    # 测试当 mu 是非实数时创建 Levy 分布对象会引发 ValueError
    mu = Symbol("mu", real=False)
    raises(ValueError, lambda: Levy('x',mu,c))

    # 测试当 c 是非正数时创建 Levy 分布对象会引发 ValueError
    c = Symbol("c", nonpositive=True)
    raises(ValueError, lambda: Levy('x',mu,c))

    # 测试当 mu 是实数且 c 是正数时创建 Levy 分布对象不会引发异常
    mu = Symbol("mu", real=True)
    raises(ValueError, lambda: Levy('x',mu,c))

# 定义一个测试函数，用于测试 LogCauchy 分布的性质
def test_logcauchy():
    # 定义符号变量 mu 和 sigma，其中 mu 和 sigma 都是正数
    mu = Symbol("mu", positive=True)
    sigma = Symbol("sigma", positive=True)

    # 创建 LogCauchy 分布对象 X，参数为 'x'，位置参数 mu，尺度参数 sigma
    X = LogCauchy("x", mu, sigma)

    # 测试概率密度函数 density(X)，断言其返回值
    assert density(X)(x) == sigma/(x*pi*(sigma**2 + (-mu + log(x))**2))

    # 测试累积分布函数 cdf(X)，断言其返回值
    assert cdf(X)(x) == atan((log(x) - mu)/sigma)/pi + S.Half

# 定义一个测试函数，用于测试 Logistic 分布的性质
def test_logistic():
    # 定义符号变量 mu，s，p，其中 mu 是实数，s 和 p 是正数
    mu = Symbol("mu", real=True)
    s = Symbol("s", positive=True)
    p = Symbol("p", positive=True)

    # 创建 Logistic 分布对象 X，参数为 'x'，均值 mu，尺度参数 s
    X = Logistic('x', mu, s)

    # 测试特征函数 characteristic_function(X)，断言其返回值
    assert characteristic_function(X)(x) == \
           (Piecewise((pi*s*x*exp(I*mu*x)/sinh(pi*s*x), Ne(x, 0)), (1, True)))

    # 测试概率密度函数 density(X)，断言其返回值
    assert density(X)(x) == exp((-x + mu)/s)/(s*(exp((-x + mu)/s) + 1)**2)

    # 测试累积分布函数 cdf(X)，断言其返回值
    assert cdf(X)(x) == 1/(exp((mu - x)/s) + 1)

    # 测试分位数函数 quantile(X)，断言其返回值
    assert quantile(X)(p) == mu - s*log(-S.One + 1/p)

# 定义一个测试函数，用于测试 LogLogistic 分布的性质
def test_loglogistic():
    # 定义符号变量 a 和 b，均为正数
    a, b = symbols('a b')
    assert LogLogistic('x', a, b)

    # 当 a 为负数时，创建 LogLogistic 分布对象会引发 ValueError
    a = Symbol('a', negative=True)
    b = Symbol('b', positive=True)
    raises(ValueError, lambda: LogLogistic('x', a, b))

    # 当 b 为负数时，创建 LogLogistic 分布对象会引发 ValueError
    a = Symbol('a', positive=True)
    b = Symbol('b', negative=True)
    raises(ValueError, lambda: LogLogistic('x', a, b))

    # 定义符号变量 a，b，z，p，均为正数
    a, b, z, p = symbols('a b z p', positive=True)
    # 创建 LogLogistic 分布对象 X，参数为 'x'，位置参数 a，尺度参数 b
    X = LogLogistic('x', a, b)
    # 测试概率密度函数 density(X)，断言其返回值
    assert density(X)(z) == b*(z/a)**(b - 1)/(a*((z/a)**b + 1)**2)
    # 测试累积分布函数 cdf(X)，断言其返回值
    assert cdf(X)(z) == 1/(1 + (z/a)**(-b))
    # 测试分位数函数 quantile(X)，断言其返回值
    assert quantile(X)(p) == a*(p/(1 - p))**(1/b)

    # 测试期望函数 E(X)，当 b <= 1 时返回 S.NaN，否则返回 pi*a/(b*sin(pi/b))
    assert E(X) == Piecewise((S.NaN, b <= 1), (pi*a/(b*sin(pi/b)), True))

    # 当 b > 1 时，创建 LogLogistic 分布对象 X，不会引发异常
    b = symbols('b', prime=True) # b > 1
    X = LogLogistic('x', a, b)
    # 测试期望函数 E(X)，断言其返回值
    assert E(X) == pi*a/(b*sin(pi/b))

    # 创建具有参数 a=1，b=2 的 LogLogistic 分布对象 X
    X = LogLogistic('x', 1, 2)
    # 测试中位数函数 median(X)，断言其返回值为 FiniteSet(1)
    assert median(X) == FiniteSet(1)

# 定义一个测试函数，用于测试 LogitNormal 分布的性质
def test_logitnormal():
    # 定义符号变量 mu 和 s，其中 mu 是实数，s 是正数
    mu = Symbol('mu', real=True)
    s = Symbol('s', positive=True)

    # 创建 LogitNormal 分布对象 X，参数为 'x'，位置参数 mu，尺度参数 s
    X = LogitNormal('x', mu, s)
    x = Symbol('x')
    # 断言：验证 density(X)(x) 的计算结果是否等于指定的数学表达式
    assert density(X)(x) == sqrt(2)*exp(-(-mu + log(x/(1 - x)))**2/(2*s**2))/(2*sqrt(pi)*s*x*(1 - x))
    
    # 断言：验证 cdf(X)(x) 的计算结果是否等于指定的数学表达式
    assert cdf(X)(x) == erf(sqrt(2)*(-mu + log(x/(1 - x)))/(2*s))/2 + S(1)/2
# 定义测试函数 test_lognormal
def test_lognormal():
    # 创建符号变量 mean 和 std，mean 是实数，std 是正数
    mean = Symbol('mu', real=True)
    std = Symbol('sigma', positive=True)
    # 创建 LogNormal 分布 X，均值为 mean，标准差为 std
    X = LogNormal('x', mean, std)
    
    # 以下两行代码被注释掉，因为 sympy 的积分器处理能力不佳
    # 断言期望值 E(X) 等于 exp(mean+std**2/2)
    # 断言方差 variance(X) 等于 (exp(std**2)-1) * exp(2*mean + std**2)

    # 断言无法计算矩生成函数，会抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: moment_generating_function(X))
    
    # 创建新的符号变量 mu 和 sigma，mu 是实数，sigma 是正数
    mu = Symbol("mu", real=True)
    sigma = Symbol("sigma", positive=True)
    
    # 创建 LogNormal 分布 X，均值为 mu，标准差为 sigma
    X = LogNormal('x', mu, sigma)
    # 断言 LogNormal 分布 X 的概率密度函数等于给定的表达式
    assert density(X)(x) == (sqrt(2)*exp(-(-mu + log(x))**2
                                    /(2*sigma**2))/(2*x*sqrt(pi)*sigma))
    
    # 断言 LogNormal 分布 X 的累积分布函数等于给定的 Piecewise 表达式
    assert cdf(X)(x) == Piecewise(
                        (erf(sqrt(2)*(-mu + log(x))/(2*sigma))/2
                        + S(1)/2, x > 0), (0, True))
    
    # 创建 LogNormal 分布 X，均值为 0，标准差为 1
    X = LogNormal('x', 0, 1)  # 均值为 0，标准差为 1
    # 断言 LogNormal 分布 X 的概率密度函数等于给定的表达式
    assert density(X)(x) == sqrt(2)*exp(-log(x)**2/2)/(2*x*sqrt(pi))


# 定义测试函数 test_Lomax
def test_Lomax():
    # 创建符号变量 a 和 l，a 是负数，l 也是负数，会引发 ValueError 异常
    a, l = symbols('a, l', negative=True)
    raises(ValueError, lambda: Lomax('X', a, l))
    
    # 创建符号变量 a 和 l，a 和 l 都不是实数，会引发 ValueError 异常
    a, l = symbols('a, l', real=False)
    raises(ValueError, lambda: Lomax('X', a, l))
    
    # 创建符号变量 a 和 l，a 是正数，l 是任意实数
    a, l = symbols('a, l', positive=True)
    # 创建 Lomax 分布 X，参数为 a 和 l
    X = Lomax('X', a, l)
    # 断言 Lomax 分布 X 的概率空间的定义域是区间 [0, oo)
    assert X.pspace.domain.set == Interval(0, oo)
    # 断言 Lomax 分布 X 的概率密度函数等于给定的表达式
    assert density(X)(x) == a*(1 + x/l)**(-a - 1)/l
    # 断言 Lomax 分布 X 的累积分布函数等于给定的 Piecewise 表达式
    assert cdf(X)(x) == Piecewise((1 - (1 + x/l)**(-a), x >= 0), (0, True))
    
    # 设置 a 为 3
    a = 3
    # 创建新的 Lomax 分布 X，参数为 a 和之前的 l
    X = Lomax('X', a, l)
    # 断言 Lomax 分布 X 的期望等于 l/2
    assert E(X) == l/2
    # 断言 Lomax 分布 X 的中位数等于给定的 FiniteSet 表达式
    assert median(X) == FiniteSet(l*(-1 + 2**Rational(1, 3)))
    # 断言 Lomax 分布 X 的方差等于给定的表达式
    assert variance(X) == 3*l**2/4


# 定义测试函数 test_maxwell
def test_maxwell():
    # 创建符号变量 a，a 是正数
    a = Symbol("a", positive=True)
    # 创建 Maxwell 分布 X，参数为 a
    X = Maxwell('x', a)

    # 断言 Maxwell 分布 X 的概率密度函数等于给定的表达式
    assert density(X)(x) == (sqrt(2)*x**2*exp(-x**2/(2*a**2))/
        (sqrt(pi)*a**3))
    # 断言 Maxwell 分布 X 的期望等于给定的表达式
    assert E(X) == 2*sqrt(2)*a/sqrt(pi)
    # 断言 Maxwell 分布 X 的方差等于给定的表达式
    assert variance(X) == -8*a**2/pi + 3*a**2
    # 断言 Maxwell 分布 X 的累积分布函数等于给定的表达式
    assert cdf(X)(x) == erf(sqrt(2)*x/(2*a)) - sqrt(2)*x*exp(-x**2/(2*a**2))/(sqrt(pi)*a)
    # 断言 Maxwell 分布 X 的累积分布函数对 x 的导数等于 Maxwell 分布 X 的概率密度函数
    assert diff(cdf(X)(x), x) == density(X)(x)


# 定义测试函数 test_Moyal，标记为 slow，可能需要较长时间运行
@slow
def test_Moyal():
    # 创建符号变量 mu，mu 是非实数
    mu = Symbol('mu',real=False)
    # 创建符号变量 sigma，sigma 是正数
    sigma = Symbol('sigma', positive=True)
    # 创建 Moyal 分布 M，参数为 mu 和 sigma，mu 不是实数，会引发 ValueError 异常
    raises(ValueError, lambda: Moyal('M',mu, sigma))

    # 创建符号变量 mu，mu 是实数
    mu = Symbol('mu', real=True)
    # 创建符号变量 sigma，sigma 是负数，会引发 ValueError 异常
    sigma = Symbol('sigma', negative=True)
    raises(ValueError, lambda: Moyal('M',mu, sigma))

    # 创建符号变量 sigma，sigma 是正数
    sigma = Symbol('sigma', positive=True)
    # 创建 Moyal 分布 M，参数为 mu 和 sigma
    M = Moyal('M', mu, sigma)
    # 断言 Moyal 分布 M 的概率密度函数等于给定的表达式
    assert density(M)(z) == sqrt(2)*exp(-exp((mu - z)/sigma)/2
                        - (-mu + z)/(2*sigma))/(2*sqrt(pi)*sigma)
    # 断言 Moyal 分布 M 的累积分布函数简化后等于给定的表达式
    assert cdf(M)(z).simplify() == 1 - erf(sqrt(2)*exp((mu - z)/(2*sigma))/2)
    # 断言 Moyal 分布 M 的特征函数等于给定的表达式
    assert characteristic_function(M)(z) == 2**(-I*sigma*z)*exp(I*mu*z) \
                        *gamma(-I*sigma*z + Rational(1, 2))/sqrt(pi)
    # 断言 Moyal 分布 M 的期望等于给定的表达式
    assert E(M) == mu + EulerGamma*sigma + sigma*log(2)
    # 断言 Moyal 分布 M 的矩生成函数等于给定的表达式
    assert moment_generating_function(M)(z) == 2**(-sigma*z)*exp(mu*z) \
                        *gamma(-sigma*z + Rational(1, 2))/sqrt(pi)


# 定义测试函数 test_nakagami
def test_nakagami():
    # 创建符号变量 mu 和 omega，mu 和 omega 都是正数
    mu = Symbol("mu", positive=True)
    omega = Symbol("omega", positive=True)
    # 创建一个 Nakagami 分布的实例 X，指定参数为 mu 和 omega
    X = Nakagami('x', mu, omega)
    # 断言：验证概率密度函数的计算结果是否符合预期
    assert density(X)(x) == (2*x**(2*mu - 1)*mu**mu*omega**(-mu)
                                *exp(-x**2*mu/omega)/gamma(mu))
    # 断言：验证期望值的简化结果是否符合预期
    assert simplify(E(X)) == (sqrt(mu)*sqrt(omega)
                                            *gamma(mu + S.Half)/gamma(mu + 1))
    # 断言：验证方差的简化结果是否符合预期
    assert simplify(variance(X)) == (
    omega - omega*gamma(mu + S.Half)**2/(gamma(mu)*gamma(mu + 1)))
    # 断言：验证累积分布函数的计算结果是否符合预期
    assert cdf(X)(x) == Piecewise(
                                (lowergamma(mu, mu*x**2/omega)/gamma(mu), x > 0),
                                (0, True))
    # 创建另一个 Nakagami 分布的实例 X，参数为 1 和 1
    X = Nakagami('x', 1, 1)
    # 断言：验证中位数的计算结果是否符合预期
    assert median(X) == FiniteSet(sqrt(log(2)))
def test_gaussian_inverse():
    # test for symbolic parameters
    # 定义符号变量 a 和 b
    a, b = symbols('a b')
    # 断言 GaussianInverse 类可以被实例化并传入 'x', a, b 作为参数
    assert GaussianInverse('x', a, b)

    # Inverse Gaussian distribution is also known as Wald distribution
    # `GaussianInverse` can also be referred by the name `Wald`
    # 定义符号变量 a, b, z
    a, b, z = symbols('a b z')
    # 创建一个 Wald 分布对象 X，使用参数 'x', a, b
    X = Wald('x', a, b)
    # 断言 X 的概率密度函数在 z 处的值符合给定的数学表达式
    assert density(X)(z) == sqrt(2)*sqrt(b/z**3)*exp(-b*(-a + z)**2/(2*a**2*z))/(2*sqrt(pi))

    # 定义正的符号变量 a 和 b
    a, b = symbols('a b', positive=True)
    # 定义正的符号变量 z
    z = Symbol('z', positive=True)

    # 创建一个 GaussianInverse 分布对象 X，使用参数 'x', a, b
    X = GaussianInverse('x', a, b)
    # 断言 X 的概率密度函数在 z 处的值符合给定的数学表达式
    assert density(X)(z) == sqrt(2)*sqrt(b)*sqrt(z**(-3))*exp(-b*(-a + z)**2/(2*a**2*z))/(2*sqrt(pi))
    # 断言 X 的期望值为 a
    assert E(X) == a
    # 断言 X 的方差展开后的值为 a 的立方除以 b
    assert variance(X).expand() == a**3/b
    # 断言 X 的累积分布函数在 z 处的值符合给定的数学表达式
    assert cdf(X)(z) == (S.Half - erf(sqrt(2)*sqrt(b)*(1 + z/a)/(2*sqrt(z)))/2)*exp(2*b/a) +\
         erf(sqrt(2)*sqrt(b)*(-1 + z/a)/(2*sqrt(z)))/2 + S.Half

    # 定义非正的符号变量 a
    a = symbols('a', nonpositive=True)
    # 断言当创建 GaussianInverse 分布对象时会引发 ValueError 异常
    raises(ValueError, lambda: GaussianInverse('x', a, b))

    # 定义正的符号变量 a 和非正的符号变量 b
    a = symbols('a', positive=True)
    b = symbols('b', nonpositive=True)
    # 断言当创建 GaussianInverse 分布对象时会引发 ValueError 异常
    raises(ValueError, lambda: GaussianInverse('x', a, b))


def test_pareto():
    # 定义正的符号变量 xm 和 beta
    xm, beta = symbols('xm beta', positive=True)
    # 计算 alpha
    alpha = beta + 5
    # 创建 Pareto 分布对象 X，使用参数 'x', xm, alpha
    X = Pareto('x', xm, alpha)

    # 获取 X 的概率密度函数
    dens = density(X)

    # 测试累积分布函数
    assert cdf(X)(x) == \
           Piecewise((-x**(-beta - 5)*xm**(beta + 5) + 1, x >= xm), (0, True))

    # 测试特征函数
    assert characteristic_function(X)(x) == \
           ((-I*x*xm)**(beta + 5)*(beta + 5)*uppergamma(-beta - 5, -I*x*xm))

    # 断言 X 的概率密度函数在 x 处的值符合给定的数学表达式
    assert dens(x) == x**(-(alpha + 1))*xm**(alpha)*(alpha)

    # 断言 X 的期望值简化后的结果符合给定的数学表达式
    assert simplify(E(X)) == alpha*xm/(alpha-1)

    # 计算 MGF 的泰勒级数仍然太慢
    #assert simplify(variance(X)) == xm**2*alpha / ((alpha-1)**2*(alpha-2))


def test_pareto_numeric():
    # 定义数值变量 xm 和 beta
    xm, beta = 3, 2
    # 计算 alpha
    alpha = beta + 5
    # 创建 Pareto 分布对象 X，使用参数 'x', xm, alpha
    X = Pareto('x', xm, alpha)

    # 断言 X 的期望值为给定的数学表达式
    assert E(X) == alpha*xm/S(alpha - 1)
    # 断言 X 的方差为给定的数学表达式
    assert variance(X) == xm**2*alpha / S((alpha - 1)**2*(alpha - 2))
    # 断言 X 的中位数为给定的数学表达式
    assert median(X) == FiniteSet(3*2**Rational(1, 7))
    # 斜度测试太慢。尝试简化函数？


def test_PowerFunction():
    # 定义非正的符号变量 alpha 和实数变量 a, b
    alpha = Symbol("alpha", nonpositive=True)
    a, b = symbols('a, b', real=True)
    # 断言当创建 PowerFunction 分布对象时会引发 ValueError 异常
    raises(ValueError, lambda: PowerFunction('x', alpha, a, b))

    # 定义实数变量 a, b
    a, b = symbols('a, b', real=False)
    # 断言当创建 PowerFunction 分布对象时会引发 ValueError 异常
    raises(ValueError, lambda: PowerFunction('x', alpha, a, b))

    # 定义正的符号变量 alpha 和实数变量 a, b
    alpha = Symbol("alpha", positive=True)
    a, b = symbols('a, b', real=True)
    # 断言创建 PowerFunction 分布对象 X，使用参数 'X', 2, a, b
    X = PowerFunction('X', 2, a, b)
    # 断言 X 的概率密度函数在 z 处的值符合给定的数学表达式
    assert density(X)(z) == (-2*a + 2*z)/(-a + b)**2
    # 断言 X 的累积分布函数在 z 处的值符合给定的数学表达式
    assert cdf(X)(z) == Piecewise((a**2/(a**2 - 2*a*b + b**2) -
        2*a*z/(a**2 - 2*a*b + b**2) + z**2/(a**2 - 2*a*b + b**2), a <= z), (0, True))

    # 创建 PowerFunction 分布对象 X，使用参数 'X', 2, 0, 1
    X = PowerFunction('X', 2, 0, 1)
    # 断言 X 的概率密度函数在 z 处的值符合给定的数学表达式
    assert density(X)(z) == 2*z
    # 断言 X 的累积分布函数在 z 处的值符合给定的数学表达式
    assert cdf(X)(z) == Piecewise((z**2, z >= 0), (0,True))
    # 断言 X 的期望值为给定的有理数
    assert E(X) == Rational(2,3)
    # 断言 P(X < 0) 的概率为 0
    assert P(X < 0) == 0
    # 断言 P(X < 1) 的概率为 1
    assert P(X < 1) == 1
    # 断言语句，用于检查条件是否为真，如果条件为假则触发 AssertionError 异常
    assert median(X) == FiniteSet(1/sqrt(2))
    # 断言：X 的中位数等于数学库中定义的无限集合中的 1/sqrt(2) 值
# 定义一个测试函数，用于测试 Raised Cosine 分布的功能
def test_raised_cosine():
    # 声明一个实数符号 mu
    mu = Symbol("mu", real=True)
    # 声明一个正数符号 s
    s = Symbol("s", positive=True)

    # 创建一个 RaisedCosine 随机变量 X
    X = RaisedCosine("x", mu, s)

    # 断言 Raised Cosine 随机变量 X 的概率空间为指定区间
    assert pspace(X).domain.set == Interval(mu - s, mu + s)
    
    # 断言特征函数的计算结果
    assert characteristic_function(X)(x) == \
           Piecewise((exp(-I*pi*mu/s)/2, Eq(x, -pi/s)), (exp(I*pi*mu/s)/2, Eq(x, pi/s)), (pi**2*exp(I*mu*x)*sin(s*x)/(s*x*(-s**2*x**2 + pi**2)), True))
    
    # 断言概率密度函数的计算结果
    assert density(X)(x) == (Piecewise(((cos(pi*(x - mu)/s) + 1)/(2*s),
                          And(x <= mu + s, mu - s <= x)), (0, True)))


# 定义一个测试函数，用于测试 Rayleigh 分布的功能
def test_rayleigh():
    # 声明一个正数符号 sigma
    sigma = Symbol("sigma", positive=True)

    # 创建一个 Rayleigh 随机变量 X
    X = Rayleigh('x', sigma)

    # 断言特征函数的计算结果
    assert characteristic_function(X)(x) == (-sqrt(2)*sqrt(pi)*sigma*x*(erfi(sqrt(2)*sigma*x/2) - I)*exp(-sigma**2*x**2/2)/2 + 1)

    # 断言概率密度函数的计算结果
    assert density(X)(x) ==  x*exp(-x**2/(2*sigma**2))/sigma**2
    
    # 断言期望值的计算结果
    assert E(X) == sqrt(2)*sqrt(pi)*sigma/2
    
    # 断言方差的计算结果
    assert variance(X) == -pi*sigma**2/2 + 2*sigma**2
    
    # 断言累积分布函数的计算结果
    assert cdf(X)(x) == 1 - exp(-x**2/(2*sigma**2))
    
    # 断言累积分布函数的导数与概率密度函数的关系
    assert diff(cdf(X)(x), x) == density(X)(x)


# 定义一个测试函数，用于测试 Reciprocal 分布的功能
def test_reciprocal():
    # 声明两个实数符号 a 和 b
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)

    # 创建一个 Reciprocal 随机变量 X
    X = Reciprocal('x', a, b)
    
    # 断言概率密度函数的计算结果
    assert density(X)(x) == 1/(x*(-log(a) + log(b)))
    
    # 断言累积分布函数的计算结果
    assert cdf(X)(x) == Piecewise((log(a)/(log(a) - log(b)) - log(x)/(log(a) - log(b)), a <= x), (0, True))
    
    # 创建一个具体数值的 Reciprocal 随机变量 X
    X = Reciprocal('x', 5, 30)
    
    # 断言期望值的计算结果
    assert E(X) == 25/(log(30) - log(5))
    
    # 断言小于 4 的概率为零
    assert P(X < 4) == S.Zero
    
    # 断言小于 20 的概率的计算结果
    assert P(X < 20) == log(20) / (log(30) - log(5)) - log(5) / (log(30) - log(5))
    
    # 断言在 x=10 处的累积分布函数的计算结果
    assert cdf(X)(10) == log(10) / (log(30) - log(5)) - log(5) / (log(30) - log(5))
    
    # 测试当 a 为非正数时，应引发 ValueError 异常
    a = symbols('a', nonpositive=True)
    raises(ValueError, lambda: Reciprocal('x', a, b))
    
    # 测试当 a 和 b 均为正数且 a+b 小于 a 时，应引发 ValueError 异常
    a = symbols('a', positive=True)
    b = symbols('b', positive=True)
    raises(ValueError, lambda: Reciprocal('x', a + b, a))


# 定义一个测试函数，用于测试 ShiftedGompertz 分布的功能
def test_shiftedgompertz():
    # 声明两个正数符号 b 和 eta
    b = Symbol("b", positive=True)
    eta = Symbol("eta", positive=True)
    
    # 创建一个 ShiftedGompertz 随机变量 X
    X = ShiftedGompertz("x", b, eta)
    
    # 断言概率密度函数的计算结果
    assert density(X)(x) == b*(eta*(1 - exp(-b*x)) + 1)*exp(-b*x)*exp(-eta*exp(-b*x))


# 定义一个测试函数，用于测试 StudentT 分布的功能
def test_studentt():
    # 声明一个正数符号 nu
    nu = Symbol("nu", positive=True)
    
    # 创建一个 StudentT 随机变量 X
    X = StudentT('x', nu)
    
    # 断言概率密度函数的计算结果
    assert density(X)(x) == (1 + x**2/nu)**(-nu/2 - S.Half)/(sqrt(nu)*beta(S.Half, nu/2))
    
    # 断言累积分布函数的计算结果
    assert cdf(X)(x) == S.Half + x*gamma(nu/2 + S.Half)*hyper((S.Half, nu/2 + S.Half),
                                (Rational(3, 2),), -x**2/nu)/(sqrt(pi)*sqrt(nu)*gamma(nu/2))
    
    # 测试未实现矩母函数的情况，应引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: moment_generating_function(X))


# 定义一个测试函数，用于测试 Trapezoidal 分布的功能
def test_trapezoidal():
    # 声明四个实数符号 a, b, c, d
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    c = Symbol("c", real=True)
    d = Symbol("d", real=True)
    
    # 创建一个 Trapezoidal 随机变量 X
    X = Trapezoidal('x', a, b, c, d)
    # 断言：验证密度函数 density(X)(x) 的返回值是否等于 Piecewise 对象，Piecewise 是一个条件分段函数
    assert density(X)(x) == Piecewise(((-2*a + 2*x)/((-a + b)*(-a - b + c + d)), (a <= x) & (x < b)),
                                      (2/(-a - b + c + d), (b <= x) & (x < c)),
                                      ((2*d - 2*x)/((-c + d)*(-a - b + c + d)), (c <= x) & (x <= d)),
                                      (0, True))
    
    # 定义一个 Trapezoidal 对象 X，表示一个梯形分布，定义在区间 [0, 3] 上
    X = Trapezoidal('x', 0, 1, 2, 3)
    
    # 断言：验证随机变量 X 的期望是否等于有理数 3/2
    assert E(X) == Rational(3, 2)
    
    # 断言：验证随机变量 X 的方差是否等于有理数 5/12
    assert variance(X) == Rational(5, 12)
    
    # 断言：验证随机变量 X 小于 2 的概率是否等于有理数 3/4
    assert P(X < 2) == Rational(3, 4)
    
    # 断言：验证随机变量 X 的中位数是否等于有限集 {3/2}
    assert median(X) == FiniteSet(Rational(3, 2))
def test_triangular():
    # 定义符号变量 a, b, c
    a = Symbol("a")
    b = Symbol("b")
    c = Symbol("c")

    # 创建一个三角分布随机变量 X，范围为 [a, b]，并断言其概率空间的定义域为 Interval(a, b)
    X = Triangular('x', a, b, c)
    assert pspace(X).domain.set == Interval(a, b)

    # 断言 X 的概率密度函数在某个变量 x 的字符串表示符合预期
    assert str(density(X)(x)) == ("Piecewise(((-2*a + 2*x)/((-a + b)*(-a + c)), (a <= x) & (c > x)), "
                                  "(2/(-a + b), Eq(c, x)), ((2*b - 2*x)/((-a + b)*(b - c)), (b >= x) & (c < x)), (0, True))")

    # 断言 X 的矩生成函数在某个变量 x 的展开形式符合预期
    assert moment_generating_function(X)(x).expand() == \
        ((-2*(-a + b)*exp(c*x) + 2*(-a + c)*exp(b*x) + 2*(b - c)*exp(a*x))/(x**2*(-a + b)*(-a + c)*(b - c))).expand()

    # 断言 X 的特征函数在某个变量 x 的字符串表示符合预期
    assert str(characteristic_function(X)(x)) == \
        '(2*(-a + b)*exp(I*c*x) - 2*(-a + c)*exp(I*b*x) - 2*(b - c)*exp(I*a*x))/(x**2*(-a + b)*(-a + c)*(b - c))'


def test_quadratic_u():
    # 定义实数符号变量 a, b
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)

    # 创建一个二次均匀分布随机变量 X，范围为 [a, b]
    X = QuadraticU("x", a, b)
    Y = QuadraticU("x", 1, 2)

    # 断言 X 的概率空间的定义域为 Interval(a, b)
    assert pspace(X).domain.set == Interval(a, b)

    # 断言 Y 的矩生成函数在 1 处的计算结果符合预期
    assert moment_generating_function(Y)(1)  == -15*exp(2) + 27*exp(1)
    # 断言 Y 的矩生成函数在 2 处的计算结果符合预期
    assert moment_generating_function(Y)(2) == -9*exp(4)/2 + 21*exp(2)/2

    # 断言 Y 的特征函数在 1 处的计算结果符合预期
    assert characteristic_function(Y)(1) == 3*I*(-1 + 4*I)*exp(I*exp(2*I))

    # 断言 X 的概率密度函数在变量 x 的定义域内的字符串表示符合预期
    assert density(X)(x) == (Piecewise((12*(x - a/2 - b/2)**2/(-a + b)**3,
                                        And(x <= b, a <= x)), (0, True)))


def test_uniform():
    # 定义实数符号变量 l, w（w 为正数）
    l = Symbol('l', real=True)
    w = Symbol('w', positive=True)

    # 创建一个均匀分布随机变量 X，范围为 [l, l + w]
    X = Uniform('x', l, l + w)

    # 断言 X 的期望值等于 l + w/2
    assert E(X) == l + w/2
    # 断言 X 的方差展开形式符合预期
    assert variance(X).expand() == w**2/12

    # 使用具体数值测试
    X = Uniform('x', 3, 5)
    assert P(X < 3) == 0 and P(X > 5) == 0
    assert P(X < 4) == P(X > 4) == S.Half
    assert median(X) == FiniteSet(4)

    # 定义符号变量 z，获取 X 的概率密度函数在不同点的取值，并断言符合预期
    z = Symbol('z')
    p = density(X)(z)
    assert p.subs(z, 3.7) == S.Half
    assert p.subs(z, -1) == 0
    assert p.subs(z, 6) == 0

    # 获取 X 的累积分布函数 cdf，并进行断言测试
    c = cdf(X)
    assert c(2) == 0 and c(3) == 0
    assert c(Rational(7, 2)) == Rational(1, 4)
    assert c(5) == 1 and c(6) == 1


@XFAIL
@slow
def test_uniform_P():
    """ This stopped working because SingleContinuousPSpace.compute_density no
    longer calls integrate on a DiracDelta but rather just solves directly.
    integrate used to call UniformDistribution.expectation which special-cased
    subsed out the Min and Max terms that Uniform produces

    I decided to regress on this class for general cleanliness (and I suspect
    speed) of the algorithm.
    """
    l = Symbol('l', real=True)
    w = Symbol('w', positive=True)
    X = Uniform('x', l, l + w)
    assert P(X < l) == 0 and P(X > l + w) == 0


def test_uniformsum():
    # 定义整数符号变量 n 和虚拟变量 _k
    n = Symbol("n", integer=True)
    _k = Dummy("k")
    x = Symbol("x")

    # 创建一个均匀和分布随机变量 X
    X = UniformSum('x', n)
    # 计算 X 的概率密度函数在变量 x 处的结果，并断言与预期一致
    res = Sum((-1)**_k*(-_k + x)**(n - 1)*binomial(n, _k), (_k, 0, floor(x)))/factorial(n - 1)
    assert density(X)(x).dummy_eq(res)

    # 断言 X 的概率空间的定义域为 Interval(0, n)

    # 测试特征函数
    assert X.pspace.domain.set == Interval(0, n)
    # 断言语句，测试特征函数 characteristic_function(X)(x) 是否等于 (-I*(exp(I*x) - 1)/x)**n
    assert characteristic_function(X)(x) == (-I*(exp(I*x) - 1)/x)**n
    
    # 断言语句，测试矩生成函数 moment_generating_function(X)(x) 是否等于 ((exp(x) - 1)/x)**n
    assert moment_generating_function(X)(x) == ((exp(x) - 1)/x)**n
def test_von_mises():
    # 定义符号变量 mu 和 k
    mu = Symbol("mu")
    k = Symbol("k", positive=True)

    # 创建 VonMises 分布对象 X，以 mu 和 k 作为参数
    X = VonMises("x", mu, k)
    # 断言 VonMises 分布的概率密度函数
    assert density(X)(x) == exp(k*cos(x - mu))/(2*pi*besseli(0, k))


def test_weibull():
    # 定义正的符号变量 a 和 b
    a, b = symbols('a b', positive=True)
    # FIXME: simplify(E(X)) 似乎在没有 extended_positive=True 时会卡住
    # 在 Linux 机器上这会导致内存迅速泄漏...
    # 创建 Weibull 分布对象 X，以 a 和 b 作为参数
    X = Weibull('x', a, b)

    # 断言 Weibull 分布的期望
    assert E(X).expand() == a * gamma(1 + 1/b)
    # 断言 Weibull 分布的方差
    assert variance(X).expand() == (a**2 * gamma(1 + 2/b) - E(X)**2).expand()
    # 断言 Weibull 分布的偏度
    assert simplify(skewness(X)) == (2*gamma(1 + 1/b)**3 - 3*gamma(1 + 1/b)*gamma(1 + 2/b) + gamma(1 + 3/b))/(-gamma(1 + 1/b)**2 + gamma(1 + 2/b))**Rational(3, 2)
    # 断言 Weibull 分布的峰度
    assert simplify(kurtosis(X)) == (-3*gamma(1 + 1/b)**4 +\
        6*gamma(1 + 1/b)**2*gamma(1 + 2/b) - 4*gamma(1 + 1/b)*gamma(1 + 3/b) + gamma(1 + 4/b))/(gamma(1 + 1/b)**2 - gamma(1 + 2/b))**2


def test_weibull_numeric():
    # 测试整数和有理数情况
    a = 1
    bvals = [S.Half, 1, Rational(3, 2), 5]
    for b in bvals:
        # 创建 Weibull 分布对象 X，以 a 和 b 作为参数
        X = Weibull('x', a, b)
        # 断言 Weibull 分布的期望
        assert simplify(E(X)) == expand_func(a * gamma(1 + 1/S(b)))
        # 断言 Weibull 分布的方差
        assert simplify(variance(X)) == simplify(
            a**2 * gamma(1 + 2/S(b)) - E(X)**2)
        # 不测试偏度... 在整数/分数值大于 3/2 时速度较慢


def test_wignersemicircle():
    # 定义正的符号变量 R
    R = Symbol("R", positive=True)

    # 创建 WignerSemicircle 分布对象 X，以 R 作为参数
    X = WignerSemicircle('x', R)
    # 断言 X 的概率空间的定义域
    assert pspace(X).domain.set == Interval(-R, R)
    # 断言 WignerSemicircle 分布的概率密度函数
    assert density(X)(x) == 2*sqrt(-x**2 + R**2)/(pi*R**2)
    # 断言 WignerSemicircle 分布的期望
    assert E(X) == 0

    # Tests ChiNoncentralDistribution
    # 断言 X 的特征函数
    assert characteristic_function(X)(x) == \
           Piecewise((2*besselj(1, R*x)/(R*x), Ne(x, 0)), (1, True))


def test_input_value_assertions():
    a, b = symbols('a b')
    p, q = symbols('p q', positive=True)
    m, n = symbols('m n', positive=False, real=True)

    # 断言 Normal 分布在参数不合适时引发 ValueError
    raises(ValueError, lambda: Normal('x', 3, 0))
    raises(ValueError, lambda: Normal('x', m, n))
    # 创建 Normal 分布对象，不引发错误
    Normal('X', a, p)
    # 断言指定分布在参数不合适时引发 ValueError
    raises(ValueError, lambda: Exponential('x', m))
    # 创建 Exponential 分布对象，不引发错误
    Exponential('Ex', p)
    # 对于 Pareto, Weibull, Beta, Gamma 四种分布，断言在参数不合适时引发 ValueError
    for fn in [Pareto, Weibull, Beta, Gamma]:
        raises(ValueError, lambda: fn('x', m, p))
        raises(ValueError, lambda: fn('x', p, n))
        # 创建指定分布对象，不引发错误
        fn('x', p, q)


def test_unevaluated():
    # 创建正态分布对象 X，以均值 0，标准差 1 作为参数
    X = Normal('x', 0, 1)
    # 创建虚拟变量 k
    k = Dummy('k')
    # 定义表达式 expr1 和 expr2
    expr1 = Integral(sqrt(2)*k*exp(-k**2/2)/(2*sqrt(pi)), (k, -oo, oo))
    expr2 = Integral(sqrt(2)*exp(-k**2/2)/(2*sqrt(pi)), (k, 0, oo))
    # 忽略警告后进行断言，测试期望值的表达式重写为积分形式
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert E(X, evaluate=False).rewrite(Integral).dummy_eq(expr1)
        assert E(X + 1, evaluate=False).rewrite(Integral).dummy_eq(expr1 + 1)
        assert P(X > 0, evaluate=False).rewrite(Integral).dummy_eq(expr2)

    # 断言在给定条件下的概率
    assert P(X > 0, X**2 < 1) == S.Half


def test_probability_unevaluated():
    # 创建正态分布对象 T，以均值 30，标准差 3 作为参数
    T = Normal('T', 30, 3)
    # 使用 ignore_warnings 上下文管理器，忽略 UserWarning 类型的警告
    with ignore_warnings(UserWarning): 
        # 断言语句，验证 P(T > 33, evaluate=False) 的类型为 Probability 类型
        assert type(P(T > 33, evaluate=False)) == Probability
# 定义测试函数，用于测试未评估状态下的密度计算
def test_density_unevaluated():
    # 定义两个正态分布随机变量 X 和 Y
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 2)
    # 断言 X + Y 的密度函数在未评估状态下返回积分对象
    assert isinstance(density(X+Y, evaluate=False)(z), Integral)


# 定义测试函数，用于测试正态分布对象的各种方法和功能
def test_NormalDistribution():
    # 创建一个均值为 0，方差为 1 的正态分布对象
    nd = NormalDistribution(0, 1)
    # 创建一个符号变量 x
    x = Symbol('x')
    # 断言正态分布对象的累积分布函数的计算结果
    assert nd.cdf(x) == erf(sqrt(2)*x/2)/2 + S.Half
    # 断言正态分布对象的期望值计算结果
    assert nd.expectation(1, x) == 1
    assert nd.expectation(x, x) == 0
    assert nd.expectation(x**2, x) == 1
    # 创建一个单变量连续概率空间对象，基于均值为 2，方差为 4 的正态分布
    a = SingleContinuousPSpace(x, NormalDistribution(2, 4))
    _z = Dummy('_z')

    # 预期1：计算概率密度函数小于 1 的积分表达式
    expected1 = Integral(sqrt(2)*exp(-(_z - 2)**2/32)/(8*sqrt(pi)), (_z, -oo, 1))
    assert a.probability(x < 1, evaluate=False).dummy_eq(expected1) is True

    # 预期2：计算概率密度函数大于 1 的积分表达式
    expected2 = Integral(sqrt(2)*exp(-(_z - 2)**2/32)/(8*sqrt(pi)), (_z, 1, oo))
    assert a.probability(x > 1, evaluate=False).dummy_eq(expected2) is True

    # 创建一个单变量连续概率空间对象，基于均值为 1，方差为 9 的正态分布
    b = SingleContinuousPSpace(x, NormalDistribution(1, 9))

    # 预期3：计算概率密度函数大于 6 的积分表达式
    expected3 = Integral(sqrt(2)*exp(-(_z - 1)**2/162)/(18*sqrt(pi)), (_z, 6, oo))
    assert b.probability(x > 6, evaluate=False).dummy_eq(expected3) is True

    # 预期4：计算概率密度函数小于 6 的积分表达式
    expected4 = Integral(sqrt(2)*exp(-(_z - 1)**2/162)/(18*sqrt(pi)), (_z, -oo, 6))
    assert b.probability(x < 6, evaluate=False).dummy_eq(expected4) is True


# 定义测试函数，用于测试随机参数生成和相关方法
def test_random_parameters():
    # 定义一个均值为 2，方差为 3 的正态分布变量 mu
    mu = Normal('mu', 2, 3)
    # 定义一个基于 mu 的正态分布变量 meas
    meas = Normal('T', mu, 1)
    # 断言在未评估状态下测度函数对 z 的计算结果
    assert density(meas, evaluate=False)(z)
    # 断言测度空间对象的类型是 CompoundPSpace
    assert isinstance(pspace(meas), CompoundPSpace)
    # 定义一个多元正态分布对象 X
    X = Normal('x', [1, 2], [[1, 0], [0, 1]])
    # 断言测度空间对象的分布类型是 MultivariateNormalDistribution
    assert isinstance(pspace(X).distribution, MultivariateNormalDistribution)
    # 断言在评估状态下测度函数对 z 的简化结果
    assert density(meas)(z).simplify() == sqrt(5)*exp(-z**2/20 + z/5 - S(1)/5)/(10*sqrt(pi))


# 定义测试函数，用于测试给定条件下的随机参数生成
def test_random_parameters_given():
    # 定义一个均值为 2，方差为 3 的正态分布变量 mu
    mu = Normal('mu', 2, 3)
    # 定义一个基于 mu 的正态分布变量 meas
    meas = Normal('T', mu, 1)
    # 断言在给定条件下生成的正态分布变量结果
    assert given(meas, Eq(mu, 5)) == Normal('T', 5, 1)


# 定义测试函数，用于测试共轭先验
def test_conjugate_priors():
    # 定义一个均值为 2，方差为 3 的正态分布变量 mu
    mu = Normal('mu', 2, 3)
    # 定义一个基于 mu 的正态分布变量 x
    x = Normal('x', mu, 1)
    # 断言在未评估状态下密度函数对 x, y 的计算结果
    assert isinstance(simplify(density(mu, Eq(x, y), evaluate=False)(z)),
                      Mul)


# 定义测试函数，用于测试复杂的单变量密度计算
def test_difficult_univariate():
    """ Since using solve in place of deltaintegrate we're able to perform
    substantially more complex density computations on single continuous random
    variables """
    # 定义一个均值为 0，方差为 1 的正态分布变量 x
    x = Normal('x', 0, 1)
    # 断言 x 的三次幂的密度函数计算结果
    assert density(x**3)
    # 断言 exp(x^2) 的密度函数计算结果
    assert density(exp(x**2))
    # 断言 log(x) 的密度函数计算结果
    assert density(log(x))


# 定义测试函数，用于测试问题 10003
def test_issue_10003():
    # 定义一个参数为 3 的指数分布变量 X
    X = Exponential('x', 3)
    # 定义一个参数为 1，形状参数为 2 的伽马分布变量 G
    G = Gamma('g', 1, 2)
    # 断言 X 小于 -1 的概率为 0
    assert P(X < -1) is S.Zero
    # 断言 G 小于 -1 的概率为 0
    assert P(G < -1) is S.Zero


# 定义测试函数，用于测试预计算的累积分布函数
def test_precomputed_cdf():
    # 定义实数符号变量 x 和 mu
    x = symbols("x", real=True)
    mu = symbols("mu", real=True)
    # 定义正数符号变量 sigma, xm, alpha, n
    sigma, xm, alpha = symbols("sigma xm alpha", positive=True)
    n = symbols("n", integer=True, positive=True)
    # 定义包含多个概率分布对象的列表 distribs
    distribs = [
            Normal("X", mu, sigma),
            Pareto("P", xm, alpha),
            ChiSquared("C", n),
            Exponential("E", sigma),
            # LogNormal("L", mu, sigma),
    ]
    # 遍历 distribs 列表中的每一个分布对象 X
    for X in distribs:
        # 计算 CDF(X) - X.pspace.density.compute_cdf()(x) 的差异
        compdiff = cdf(X)(x) - simplify(X.pspace.density.compute_cdf()(x))
        # 将差异表达式重新写为 erfc 的形式
        compdiff = simplify(compdiff.rewrite(erfc))
        # 断言差异为 0
        assert compdiff == 0
def test_precomputed_characteristic_functions():
    # 导入 mpmath 库
    import mpmath

    def test_cf(dist, support_lower_limit, support_upper_limit):
        # 计算概率密度函数
        pdf = density(dist)
        # 创建符号变量 t
        t = Symbol('t')

        # 第一个函数是分布的硬编码特征函数
        cf1 = lambdify([t], characteristic_function(dist)(t), 'mpmath')

        # 第二个函数是密度函数的傅里叶变换
        f = lambdify([x, t], pdf(x)*exp(I*x*t), 'mpmath')
        # 使用数值积分计算特征函数
        cf2 = lambda t: mpmath.quad(lambda x: f(x, t), [support_lower_limit, support_upper_limit], maxdegree=10)

        # 在各个点比较两个函数
        for test_point in [2, 5, 8, 11]:
            n1 = cf1(test_point)
            n2 = cf2(test_point)

            # 断言两个函数在实部和虚部的差异小于指定精度
            assert abs(re(n1) - re(n2)) < 1e-12
            assert abs(im(n1) - im(n2)) < 1e-12

    # 测试不同分布的特征函数
    test_cf(Beta('b', 1, 2), 0, 1)
    test_cf(Chi('c', 3), 0, mpmath.inf)
    test_cf(ChiSquared('c', 2), 0, mpmath.inf)
    test_cf(Exponential('e', 6), 0, mpmath.inf)
    test_cf(Logistic('l', 1, 2), -mpmath.inf, mpmath.inf)
    test_cf(Normal('n', -1, 5), -mpmath.inf, mpmath.inf)
    test_cf(RaisedCosine('r', 3, 1), 2, 4)
    test_cf(Rayleigh('r', 0.5), 0, mpmath.inf)
    test_cf(Uniform('u', -1, 1), -1, 1)
    test_cf(WignerSemicircle('w', 3), -3, 3)


def test_long_precomputed_cdf():
    # 创建实数符号变量 x
    x = symbols("x", real=True)
    # 定义多个分布
    distribs = [
            Arcsin("A", -5, 9),
            Dagum("D", 4, 10, 3),
            Erlang("E", 14, 5),
            Frechet("F", 2, 6, -3),
            Gamma("G", 2, 7),
            GammaInverse("GI", 3, 5),
            Kumaraswamy("K", 6, 8),
            Laplace("LA", -5, 4),
            Logistic("L", -6, 7),
            Nakagami("N", 2, 7),
            StudentT("S", 4)
            ]
    # 对每个分布进行测试
    for distr in distribs:
        # 对每个分布重复测试5次
        for _ in range(5):
            # 断言概率密度函数的导数等于密度函数，并在指定区间内进行测试
            assert tn(diff(cdf(distr)(x), x), density(distr)(x), x, a=0, b=0, c=1, d=0)

    # 定义均匀和分布对象
    US = UniformSum("US", 5)
    # 计算 pdf 在区间 (0, 1) 上的值
    pdf01 = density(US)(x).subs(floor(x), 0).doit()   # pdf on (0, 1)
    # 计算 cdf 在区间 (0, 1) 上的值
    cdf01 = cdf(US, evaluate=False)(x).subs(floor(x), 0).doit()   # cdf on (0, 1)
    # 断言 cdf 的导数等于 pdf
    assert tn(diff(cdf01, x), pdf01, x, a=0, b=0, c=1, d=0)


def test_issue_13324():
    # 创建均匀分布对象
    X = Uniform('X', 0, 1)
    # 断言条件期望值
    assert E(X, X > S.Half) == Rational(3, 4)
    assert E(X, X > 0) == S.Half

def test_issue_20756():
    # 创建两个均匀分布对象
    X = Uniform('X', -1, +1)
    Y = Uniform('Y', -1, +1)
    # 断言期望值为零
    assert E(X * Y) == S.Zero
    assert E(X * ((Y + 1) - 1)) == S.Zero
    assert E(Y * (X*(X + 1) - X*X)) == S.Zero

def test_FiniteSet_prob():
    # 创建指数分布和正态分布对象
    E = Exponential('E', 3)
    N = Normal('N', 5, 7)
    # 断言概率为零
    assert P(Eq(E, 1)) is S.Zero
    assert P(Eq(N, 2)) is S.Zero
    assert P(Eq(N, x)) is S.Zero

def test_prob_neq():
    # 创建指数分布和卡方分布对象
    E = Exponential('E', 4)
    X = ChiSquared('X', 4)
    # 断言不等概率为1
    assert P(Ne(E, 2)) == 1
    assert P(Ne(X, 4)) == 1
    assert P(Ne(X, 4)) == 1
    assert P(Ne(X, 5)) == 1
    assert P(Ne(E, x)) == 1

def test_union():
    # 创建正态分布对象
    N = Normal('N', 3, 2)
    # 断言：验证 P(N**2 - N > 2) 的简化结果是否等于 -erf(sqrt(2))/2 - erfc(sqrt(2)/4)/2 + Rational(3, 2)
    assert simplify(P(N**2 - N > 2)) == \
        -erf(sqrt(2))/2 - erfc(sqrt(2)/4)/2 + Rational(3, 2)
    
    # 断言：验证 P(N**2 - 4 > 0) 的简化结果是否等于 -erf(5*sqrt(2)/4)/2 - erfc(sqrt(2)/4)/2 + Rational(3, 2)
    assert simplify(P(N**2 - 4 > 0)) == \
        -erf(5*sqrt(2)/4)/2 - erfc(sqrt(2)/4)/2 + Rational(3, 2)
# 定义一个测试函数，用于测试 Or 运算符在概率计算中的行为
def test_Or():
    # 创建一个正态分布随机变量 N，均值为 0，标准差为 1
    N = Normal('N', 0, 1)
    # 断言简化表达式 P(Or(N > 2, N < 1)) 的结果
    assert simplify(P(Or(N > 2, N < 1))) == \
        -erf(sqrt(2))/2 - erfc(sqrt(2)/2)/2 + Rational(3, 2)
    # 断言条件 Or(N < 0, N < 1) 的概率等于条件 N < 1 的概率
    assert P(Or(N < 0, N < 1)) == P(N < 1)
    # 断言条件 Or(N > 0, N < 0) 总是为真，概率为 1
    assert P(Or(N > 0, N < 0)) == 1


# 定义一个测试函数，用于测试条件等式在概率计算中的行为
def test_conditional_eq():
    # 创建一个指数分布随机变量 E，参数为 1
    E = Exponential('E', 1)
    # 断言条件 Eq(E, 1) 在条件 Eq(E, 1) 下的概率为 1
    assert P(Eq(E, 1), Eq(E, 1)) == 1
    # 断言条件 Eq(E, 1) 在条件 Eq(E, 2) 下的概率为 0
    assert P(Eq(E, 1), Eq(E, 2)) == 0
    # 断言条件 E > 1 在条件 Eq(E, 2) 下的概率为 1
    assert P(E > 1, Eq(E, 2)) == 1
    # 断言条件 E < 1 在条件 Eq(E, 2) 下的概率为 0
    assert P(E < 1, Eq(E, 2)) == 0


# 定义一个测试函数，用于测试手动定义的连续分布的行为
def test_ContinuousDistributionHandmade():
    # 创建一个符号变量 x
    x = Symbol('x')
    # 创建一个虚拟变量 z
    z = Dummy('z')
    # 定义一个密度函数 dens，为一个分段函数
    dens = Lambda(x, Piecewise((S.Half, (0<=x)&(x<1)), (0, (x>=1)&(x<2)),
        (S.Half, (x>=2)&(x<3)), (0, True)))
    # 创建一个手工定义的连续分布 dens，并指定定义域为区间 [0, 3]
    dens = ContinuousDistributionHandmade(dens, set=Interval(0, 3))
    # 创建一个单一连续概率空间 space，以 z 为随机变量，dens 为密度函数
    space = SingleContinuousPSpace(z, dens)
    # 断言 dens 的概率密度函数与预期 Lambda 表达式相等
    assert dens.pdf == Lambda(x, Piecewise((S(1)/2, (x >= 0) & (x < 1)),
        (0, (x >= 1) & (x < 2)), (S(1)/2, (x >= 2) & (x < 3)), (0, True)))
    # 断言空间中值的中位数为区间 [1, 2]
    assert median(space.value) == Interval(1, 2)
    # 断言空间中值的期望为有理数 3/2
    assert E(space.value) == Rational(3, 2)
    # 断言空间中值的方差为有理数 13/12
    assert variance(space.value) == Rational(13, 12)


# 定义一个测试函数，用于测试 issue 16318 的问题
def test_issue_16318():
    # 测试 SingleContinuousDomain 的 compute_expectation 函数
    N = SingleContinuousDomain(x, Interval(0, 1))
    # 使用 lambda 函数测试 compute_expectation 函数是否会引发 ValueError 异常
    raises(ValueError, lambda: SingleContinuousDomain.compute_expectation(N, x+1, {x, y}))


# 定义一个测试函数，用于测试概率密度函数的计算行为
def test_compute_density():
    # 创建一个正态分布随机变量 X，均值为 0，方差为 sigma^2
    X = Normal('X', 0, Symbol("sigma")**2)
    # 使用 lambda 函数测试 density 函数是否会引发 ValueError 异常
    raises(ValueError, lambda: density(X**5 + X))
```