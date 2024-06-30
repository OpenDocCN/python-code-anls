# `D:\src\scipysrc\sympy\sympy\stats\tests\test_joint_rv.py`

```
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.elementary.complexes import polar_lift
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besselk
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices.dense import eye
from sympy.matrices.expressions.determinant import Determinant
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Interval, ProductSet)
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.numbers import comp
from sympy.integrals.integrals import integrate
from sympy.matrices import Matrix, MatrixSymbol
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.stats import density, median, marginal_distribution, Normal, Laplace, E, sample
from sympy.stats.joint_rv_types import (JointRV, MultivariateNormalDistribution,
                JointDistributionHandmade, MultivariateT, NormalGamma,
                GeneralizedMultivariateLogGammaOmega as GMVLGO, MultivariateBeta,
                GeneralizedMultivariateLogGamma as GMVLG, MultivariateEwens,
                Multinomial, NegativeMultinomial, MultivariateNormal,
                MultivariateLaplace)
from sympy.testing.pytest import raises, XFAIL, skip, slow
from sympy.external import import_module

from sympy.abc import x, y

# 定义测试函数，验证正态分布相关功能
def test_Normal():
    # 创建一个均值为 [1, 2]，协方差矩阵为 [[1, 0], [0, 1]] 的正态分布对象 m
    m = Normal('A', [1, 2], [[1, 0], [0, 1]])
    # 创建一个与 m 相同参数的多维正态分布对象 A
    A = MultivariateNormal('A', [1, 2], [[1, 0], [0, 1]])
    # 断言 m 与 A 对象相等
    assert m == A
    # 断言 m 在点 (1, 2) 处的概率密度函数值为 1/(2*pi)
    assert density(m)(1, 2) == 1/(2*pi)
    # 断言 m 的概率空间分布集合为实数的直积集合
    assert m.pspace.distribution.set == ProductSet(S.Reals, S.Reals)
    # 断言试图访问 m 的索引为 2 的值会引发 ValueError 异常
    raises(ValueError, lambda: m[2])
    
    # 创建一个均值为 [1, 2, 3]，协方差矩阵为 [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 的正态分布对象 n
    n = Normal('B', [1, 2, 3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # 创建一个均值为 Matrix([1, 2])，协方差矩阵为 Matrix([[1, 0], [0, 1]]) 的正态分布对象 p
    p = Normal('C',  Matrix([1, 2]), Matrix([[1, 0], [0, 1]]))
    # 断言 m 和 p 在点 (x, y) 处的概率密度函数值相等
    assert density(m)(x, y) == density(p)(x, y)
    # 断言正态分布 n 在维度 0 和 1 的边缘分布在点 (1, 2) 处的概率密度函数值为 1/(2*pi)
    assert marginal_distribution(n, 0, 1)(1, 2) == 1/(2*pi)
    # 断言试图计算 m 的边缘分布时会引发 ValueError 异常
    raises(ValueError, lambda: marginal_distribution(m))
    # 断言计算 m 的概率密度函数在整个实数范围内的积分结果为 1.0
    assert integrate(density(m)(x, y), (x, -oo, oo), (y, -oo, oo)).evalf() == 1.0
    
    # 创建一个均值为 [1, 2]，协方差矩阵的元素为 x 和 y 的正态分布对象 N
    N = Normal('N', [1, 2], [[x, 0], [0, y]])
    # 断言 N 在点 (0, 0) 处的概率密度函数值为 exp(-((4*x + y)/(2*x*y)))/(2*pi*sqrt(x*y))
    assert density(N)(0, 0) == exp(-((4*x + y)/(2*x*y)))/(2*pi*sqrt(x*y))

    # 断言创建具有不合法协方差矩阵的正态分布对象时会引发 ValueError 异常
    raises(ValueError, lambda: Normal('M', [1, 2], [[1, 1], [1, -1]]))
    
    # 符号计算部分
    # 声明一个整数且正数的符号变量 n
    n = symbols('n', integer=True, positive=True)
    # 声明一个 n x 1 的符号矩阵 mu
    mu = MatrixSymbol('mu', n, 1)
    # 声明一个 n x n 的符号矩阵 sigma
    sigma = MatrixSymbol('sigma', n, n)
    # 创建一个均值为 mu，协方差矩阵为 sigma 的正态分布对象 X
    X = Normal('X', mu, sigma)
    # 断言 X 的概率密度函数等于 MultivariateNormalDistribution(mu, sigma)
    assert density(X) == MultivariateNormalDistribution(mu, sigma)
    
    # 断言调用 median 函数时会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: median(m))
    # 断言期望值 E(X) 等于 mu
    assert E(X) == mu

    # 断言方差 variance(X) 等于 sigma
    assert variance(X) == sigma

    # 测试符号多元正态分布密度
    n = 3

    # 定义符号矩阵 Sg, mu, obs，分别表示协方差矩阵、均值向量和观测向量
    Sg = MatrixSymbol('Sg', n, n)
    mu = MatrixSymbol('mu', n, 1)
    obs = MatrixSymbol('obs', n, 1)

    # 创建多元正态分布随机变量 X，使用指定的均值 mu 和协方差矩阵 Sg
    X = MultivariateNormal('X', mu, Sg)

    # 获取 X 的概率密度函数
    density_X = density(X)

    # 计算密度函数在观测值 obs 处的值，并代入具体数值进行计算
    eval_a = density_X(obs).subs({Sg: eye(3), mu: Matrix([0, 0, 0]), obs: Matrix([0, 0, 0])}).doit()
    eval_b = density_X(0, 0, 0).subs({Sg: eye(3), mu: Matrix([0, 0, 0])}).doit()

    # 断言计算出的 eval_a 和 eval_b 分别等于预期值 sqrt(2)/(4*pi**(3/2))
    assert eval_a == sqrt(2)/(4*pi**Rational(3/2))
    assert eval_b == sqrt(2)/(4*pi**Rational(3/2))

    # 声明符号变量 n，表示多元正态分布的维度
    n = symbols('n', integer=True, positive=True)

    # 重新定义符号矩阵 Sg, mu, obs，n 为变量
    Sg = MatrixSymbol('Sg', n, n)
    mu = MatrixSymbol('mu', n, 1)
    obs = MatrixSymbol('obs', n, 1)

    # 创建多元正态分布随机变量 X，使用指定的均值 mu 和协方差矩阵 Sg
    X = MultivariateNormal('X', mu, Sg)

    # 获取 X 的概率密度函数，并计算在观测值 obs 处的密度
    density_X_at_obs = density(X)(obs)

    # 计算在观测值 obs 处的预期密度函数，以 MatrixElement 形式表示
    expected_density = MatrixElement(
        exp((S(1)/2) * (mu.T - obs.T) * Sg**(-1) * (-mu + obs)) /
        sqrt((2*pi)**n * Determinant(Sg)), 0, 0)

    # 断言计算出的 density_X_at_obs 等于预期的密度函数 expected_density
    assert density_X_at_obs == expected_density
# 定义一个函数用于测试 MultivariateT 分布
def test_MultivariateTDist():
    # 创建一个 MultivariateT 对象 t1，指定参数和自由度
    t1 = MultivariateT('T', [0, 0], [[1, 0], [0, 1]], 2)
    # 断言 t1 在点 (1, 1) 处的概率密度函数值为 1/(8*pi)
    assert(density(t1))(1, 1) == 1/(8*pi)
    # 断言 t1 的概率空间的分布集合为实数的笛卡尔积
    assert t1.pspace.distribution.set == ProductSet(S.Reals, S.Reals)
    # 断言 t1 的概率密度函数在整个实数范围内的二重积分结果为 1.0
    assert integrate(density(t1)(x, y), (x, -oo, oo), \
        (y, -oo, oo)).evalf() == 1.0
    # 使用 lambda 表达式验证是否会引发 ValueError 异常，参数设置不合法时应抛出异常
    raises(ValueError, lambda: MultivariateT('T', [1, 2], [[1, 1], [1, -1]], 1))
    # 创建一个 MultivariateT 对象 t2，其中协方差矩阵的元素使用变量 x 和 y
    t2 = MultivariateT('t2', [1, 2], [[x, 0], [0, y]], 1)
    # 断言 t2 在点 (1, 2) 处的概率密度函数值为 1/(2*pi*sqrt(x*y))
    assert density(t2)(1, 2) == 1/(2*pi*sqrt(x*y))


# 定义一个函数用于测试 MultivariateLaplace 和 Laplace 分布
def test_multivariate_laplace():
    # 使用 lambda 表达式验证是否会引发 ValueError 异常，参数设置不合法时应抛出异常
    raises(ValueError, lambda: Laplace('T', [1, 2], [[1, 2], [2, 1]]))
    # 创建一个 Laplace 对象 L，指定参数和协方差矩阵
    L = Laplace('L', [1, 0], [[1, 0], [0, 1]])
    # 创建一个 MultivariateLaplace 对象 L2，指定参数和协方差矩阵
    L2 = MultivariateLaplace('L2', [1, 0], [[1, 0], [0, 1]])
    # 断言 L 在点 (2, 3) 处的概率密度函数值为 exp(2)*besselk(0, sqrt(39))/pi
    assert density(L)(2, 3) == exp(2)*besselk(0, sqrt(39))/pi
    # 创建一个 Laplace 对象 L1，其中协方差矩阵的元素使用变量 x 和 y
    L1 = Laplace('L1', [1, 2], [[x, 0], [0, y]])
    # 断言 L1 在点 (0, 1) 处的概率密度函数值为 exp(2/y)*besselk(0, sqrt((2 + 4/y + 1/x)/y))/(pi*sqrt(x*y))
    assert density(L1)(0, 1) == \
        exp(2/y)*besselk(0, sqrt((2 + 4/y + 1/x)/y))/(pi*sqrt(x*y))
    # 断言 L 和 L2 的概率空间的分布应相等
    assert L.pspace.distribution.set == ProductSet(S.Reals, S.Reals)
    assert L.pspace.distribution == L2.pspace.distribution


# 定义一个函数用于测试 NormalGamma 分布
def test_NormalGamma():
    # 创建一个 NormalGamma 对象 ng，指定参数
    ng = NormalGamma('G', 1, 2, 3, 4)
    # 断言 ng 在点 (1, 1) 处的概率密度函数值为 32*exp(-4)/sqrt(pi)
    assert density(ng)(1, 1) == 32*exp(-4)/sqrt(pi)
    # 断言 ng 的概率空间的分布集合为实数的笛卡尔积和一个无限区间
    assert ng.pspace.distribution.set == ProductSet(S.Reals, Interval(0, oo))
    # 使用 lambda 表达式验证是否会引发 ValueError 异常，参数设置不合法时应抛出异常
    raises(ValueError, lambda:NormalGamma('G', 1, 2, 3, -1))
    # 断言 ng 在 x = 1 处的边缘分布函数值为 3*sqrt(10)*gamma(Rational(7, 4))/(10*sqrt(pi)*gamma(Rational(5, 4)))
    assert marginal_distribution(ng, 0)(1) == \
        3*sqrt(10)*gamma(Rational(7, 4))/(10*sqrt(pi)*gamma(Rational(5, 4)))
    # 断言 ng 在 y = 1 处的边缘分布函数值为 exp(-1/4)/128
    assert marginal_distribution(ng, y)(1) == exp(Rational(-1, 4))/128
    # 断言 ng 在 x = x 处的多维边缘分布函数值为 x**2*exp(-x/4)/128
    assert marginal_distribution(ng,[0,1])(x) == x**2*exp(-x/4)/128


# 定义一个函数用于测试 GeneralizedMultivariateLogGammaDistribution 分布
def test_GeneralizedMultivariateLogGammaDistribution():
    # 定义常数和变量
    h = S.Half
    omega = Matrix([[1, h, h, h],
                     [h, 1, h, h],
                     [h, h, 1, h],
                     [h, h, h, 1]])
    v, l, mu = (4, [1, 2, 3, 4], [1, 2, 3, 4])
    y_1, y_2, y_3, y_4 = symbols('y_1:5', real=True)
    delta = symbols('d', positive=True)
    # 创建 GeneralizedMultivariateLogGammaDistribution 对象 G，指定参数
    G = GMVLGO('G', omega, v, l, mu)
    # 创建 GeneralizedMultivariateLogGammaDistribution 对象 Gd，指定参数
    Gd = GMVLG('Gd', delta, v, l, mu)
    # 定义 Gd 的密度函数表达式字符串
    dend = ("d**4*Sum(4*24**(-n - 4)*(1 - d)**n*exp((n + 4)*(y_1 + 2*y_2 + 3*y_3 "
            "+ 4*y_4) - exp(y_1) - exp(2*y_2)/2 - exp(3*y_3)/3 - exp(4*y_4)/4)/"
            "(gamma(n + 1)*gamma(n + 4)**3), (n, 0, oo))")
    # 断言 Gd 在 (y_1, y_2, y_3, y_4) 处的概率密度函数的字符串表示与预期相符
    assert str(density(Gd)(y_1, y_2, y_3, y_4)) == dend
    # 定义 G 的密度函数表达式字符串
    den = ("5*2**(2/3)*5**(1/3)*Sum(4*24**(-n - 4)*(-2**(2/3)*5**(1/3)/4 + 1)**n*"
          "exp((n + 4)*(y_1 + 2*y_2 + 3*y_3 + 4*y_4) - exp(y_1) - exp(2*y_2)/2 - "
          "exp(3*y_3)/3 - exp(4*y_4)/4)/(gamma(n + 1)*gamma(n + 4)**3), (n, 0, oo))/64")
    # 断言 G 在 (y_1, y_2, y_3, y_4) 处的概率密度函数的字符串表示与预期相符
    assert str(density(G)(y_1, y_2, y_3, y_4)) == den
    marg = ("5*2**(2/3)*5**(1/3)*exp(4*y_1)*exp(-exp(y_1))*Integral(exp(-exp(4*G[3])"
            "/4)*exp(16*G[3])*Integral(exp(-exp(3*G[2])/3)*exp(12*G[2])*Integral(exp("
            "-exp(2*G[1])/2)*exp(8*G[1])*Sum((-1/4)**n*(-4 + 2**(2/3)*5**(1/3"
            "))**n*exp(n*y_1)*exp(2*n*G[1])*exp(3*n*G[2])*exp(4*n*G[3])/(24**n*gamma(n + 1)"
            "*gamma(n + 4)**3), (n, 0, oo)), (G[1], -oo, oo)), (G[2], -oo, oo)), (G[3]"
            ", -oo, oo))/5308416")
    # 定义 marg 字符串，表示一个数学表达式

    assert str(marginal_distribution(G, G[0])(y_1)) == marg
    # 使用 assert 语句检查 marg 分布函数的字符串表示是否与 marginal_distribution 函数计算结果相等

    omega_f1 = Matrix([[1, h, h]])
    # 定义一个 1x3 的矩阵 omega_f1

    omega_f2 = Matrix([[1, h, h, h],
                     [h, 1, 2, h],
                     [h, h, 1, h],
                     [h, h, h, 1]])
    # 定义一个 4x4 的矩阵 omega_f2

    omega_f3 = Matrix([[6, h, h, h],
                     [h, 1, 2, h],
                     [h, h, 1, h],
                     [h, h, h, 1]])
    # 定义一个 4x4 的矩阵 omega_f3

    v_f = symbols("v_f", positive=False, real=True)
    # 创建一个符号 v_f，它是一个非正实数

    l_f = [1, 2, v_f, 4]
    # 定义一个包含数值和符号 v_f 的列表 l_f

    m_f = [v_f, 2, 3, 4]
    # 定义一个包含数值和符号 v_f 的列表 m_f

    omega_f4 = Matrix([[1, h, h, h, h],
                     [h, 1, h, h, h],
                     [h, h, 1, h, h],
                     [h, h, h, 1, h],
                     [h, h, h, h, 1]])
    # 定义一个 5x5 的矩阵 omega_f4

    l_f1 = [1, 2, 3, 4, 5]
    # 定义一个包含整数的列表 l_f1

    omega_f5 = Matrix([[1]])
    # 定义一个 1x1 的矩阵 omega_f5

    mu_f5 = l_f5 = [1]
    # 定义两个列表 mu_f5 和 l_f5，它们都包含一个整数 1

    raises(ValueError, lambda: GMVLGO('G', omega_f1, v, l, mu))
    # 断言调用 GMVLGO 函数会引发 ValueError 异常，参数为 'G', omega_f1, v, l, mu

    raises(ValueError, lambda: GMVLGO('G', omega_f2, v, l, mu))
    # 断言调用 GMVLGO 函数会引发 ValueError 异常，参数为 'G', omega_f2, v, l, mu

    raises(ValueError, lambda: GMVLGO('G', omega_f3, v, l, mu))
    # 断言调用 GMVLGO 函数会引发 ValueError 异常，参数为 'G', omega_f3, v, l, mu

    raises(ValueError, lambda: GMVLGO('G', omega, v_f, l, mu))
    # 断言调用 GMVLGO 函数会引发 ValueError 异常，参数为 'G', omega, v_f, l, mu

    raises(ValueError, lambda: GMVLGO('G', omega, v, l_f, mu))
    # 断言调用 GMVLGO 函数会引发 ValueError 异常，参数为 'G', omega, v, l_f, mu

    raises(ValueError, lambda: GMVLGO('G', omega, v, l, m_f))
    # 断言调用 GMVLGO 函数会引发 ValueError 异常，参数为 'G', omega, v, l, m_f

    raises(ValueError, lambda: GMVLGO('G', omega_f4, v, l, mu))
    # 断言调用 GMVLGO 函数会引发 ValueError 异常，参数为 'G', omega_f4, v, l, mu

    raises(ValueError, lambda: GMVLGO('G', omega, v, l_f1, mu))
    # 断言调用 GMVLGO 函数会引发 ValueError 异常，参数为 'G', omega, v, l_f1, mu

    raises(ValueError, lambda: GMVLGO('G', omega_f5, v, l_f5, mu_f5))
    # 断言调用 GMVLGO 函数会引发 ValueError 异常，参数为 'G', omega_f5, v, l_f5, mu_f5

    raises(ValueError, lambda: GMVLG('G', Rational(3, 2), v, l, mu))
    # 断言调用 GMVLG 函数会引发 ValueError 异常，参数为 'G', Rational(3, 2), v, l, mu
def test_MultivariateBeta():
    # 定义正数符号变量 a1 和 a2
    a1, a2 = symbols('a1, a2', positive=True)
    # 定义非负实数符号变量 a1 和 a2
    a1_f, a2_f = symbols('a1, a2', positive=False, real=True)
    # 创建 MultivariateBeta 对象 mb，参数为符号变量 a1 和 a2
    mb = MultivariateBeta('B', [a1, a2])
    # 创建 MultivariateBeta 对象 mb_c，参数为符号变量 a1 和 a2
    mb_c = MultivariateBeta('C', a1, a2)
    # 断言 mb 的密度函数在 (1, 2) 处的值符合预期
    assert density(mb)(1, 2) == S(2)**(a2 - 1)*gamma(a1 + a2)/\
                                (gamma(a1)*gamma(a2))
    # 断言 mb_c 的边缘分布在第 0 维度上在 3 处的值符合预期
    assert marginal_distribution(mb_c, 0)(3) == S(3)**(a1 - 1)*gamma(a1 + a2)/\
                                                (a2*gamma(a1)*gamma(a2))
    # 测试是否会抛出 ValueError 异常，预期情况是 a1_f 和 a2 为正数，应该引发异常
    raises(ValueError, lambda: MultivariateBeta('b1', [a1_f, a2]))
    # 测试是否会抛出 ValueError 异常，预期情况是 a2_f 为正数，应该引发异常
    raises(ValueError, lambda: MultivariateBeta('b2', [a1, a2_f]))
    # 测试是否会抛出 ValueError 异常，预期情况是 a1 和 a2 均为正数，应该引发异常
    raises(ValueError, lambda: MultivariateBeta('b3', [0, 0]))
    # 测试是否会抛出 ValueError 异常，预期情况是 a1_f 和 a2_f 均为正数，应该引发异常
    raises(ValueError, lambda: MultivariateBeta('b4', [a1_f, a2_f]))
    # 断言 mb 的概率空间分布集合符合预期
    assert mb.pspace.distribution.set == ProductSet(Interval(0, 1), Interval(0, 1))


def test_MultivariateEwens():
    # 定义正数符号变量 n 和 theta
    n, theta, i = symbols('n theta i', positive=True)

    # 定义负数符号变量 theta_f
    theta_f = symbols('t_f', negative=True)
    # 定义整数符号变量 a1, a2, a3，均为正数和整数
    a = symbols('a_1:4', positive=True, integer=True)
    # 创建 MultivariateEwens 对象 ed，维度为 3，参数为 theta
    ed = MultivariateEwens('E', 3, theta)
    # 断言 ed 的密度函数在 (a[0], a[1], a[2]) 处的值符合预期
    assert density(ed)(a[0], a[1], a[2]) == Piecewise((6*2**(-a[1])*3**(-a[2])*
                                            theta**a[0]*theta**a[1]*theta**a[2]/
                                            (theta*(theta + 1)*(theta + 2)*
                                            factorial(a[0])*factorial(a[1])*
                                            factorial(a[2])), Eq(a[0] + 2*a[1] +
                                            3*a[2], 3)), (0, True))
    # 断言 ed 的边缘分布在维度 ed[1] 上在 a[1] 处的值符合预期
    assert marginal_distribution(ed, ed[1])(a[1]) == Piecewise((6*2**(-a[1])*
                                                    theta**a[1]/((theta + 1)*
                                                    (theta + 2)*factorial(a[1])),
                                                    Eq(2*a[1] + 1, 3)), (0, True))
    # 测试是否会抛出 ValueError 异常，预期情况是 theta_f 为负数，应该引发异常
    raises(ValueError, lambda: MultivariateEwens('e1', 5, theta_f))
    # 断言 ed 的概率空间分布集合符合预期
    assert ed.pspace.distribution.set == ProductSet(Range(0, 4, 1),
                                            Range(0, 2, 1), Range(0, 2, 1))

    # 定义符号变量 n 和 theta
    eds = MultivariateEwens('E', n, theta)
    # 定义 IndexedBase 对象 a
    a = IndexedBase('a')
    # 定义符号变量 j 和 k
    j, k = symbols('j, k')
    # 定义 den 为密度函数的预期形式
    den = Piecewise((factorial(n)*Product(theta**a[j]*(j + 1)**(-a[j])/
           factorial(a[j]), (j, 0, n - 1))/RisingFactorial(theta, n),
            Eq(n, Sum((k + 1)*a[k], (k, 0, n - 1)))), (0, True))
    # 断言 eds 的密度函数在符号变量 a 上的形式符合预期
    assert density(eds)(a).dummy_eq(den)


def test_Multinomial():
    # 定义非负整数符号变量 n, x1, x2, x3, x4
    n, x1, x2, x3, x4 = symbols('n, x1, x2, x3, x4', nonnegative=True, integer=True)
    # 定义正数符号变量 p1, p2, p3, p4
    p1, p2, p3, p4 = symbols('p1, p2, p3, p4', positive=True)
    # 定义负数符号变量 p1_f, n_f
    p1_f, n_f = symbols('p1_f, n_f', negative=True)
    # 创建 Multinomial 对象 M，参数为 n 和 [p1, p2, p3, p4]
    M = Multinomial('M', n, [p1, p2, p3, p4])
    # 创建 Multinomial 对象 C，参数为 3 和 [p1, p2, p3]
    C = Multinomial('C', 3, p1, p2, p3)
    # 断言：验证 density 函数对于给定的参数 x1, x2, x3, x4 是否返回与 Piecewise 表达式相等的结果
    assert density(M)(x1, x2, x3, x4) == Piecewise((p1**x1*p2**x2*p3**x3*p4**x4*
                                            f(n)/(f(x1)*f(x2)*f(x3)*f(x4)),
                                            Eq(n, x1 + x2 + x3 + x4)), (0, True))
    
    # 断言：验证在给定的条件 C[0] 下，marginal_distribution 函数对 x1 的计算结果是否等于指定的表达式
    assert marginal_distribution(C, C[0])(x1).subs(x1, 1) ==\
                                                            3*p1*p2**2 +\
                                                            6*p1*p2*p3 +\
                                                            3*p1*p3**2
    
    # 引发 ValueError 异常：测试 Multinomial 类是否能正确处理无效参数组合 'b1'
    raises(ValueError, lambda: Multinomial('b1', 5, [p1, p2, p3, p1_f]))
    
    # 引发 ValueError 异常：测试 Multinomial 类是否能正确处理无效参数组合 'b2'
    raises(ValueError, lambda: Multinomial('b2', n_f, [p1, p2, p3, p4]))
    
    # 引发 ValueError 异常：测试 Multinomial 类是否能正确处理无效参数组合 'b3'
    raises(ValueError, lambda: Multinomial('b3', n, 0.5, 0.4, 0.3, 0.1))
# 定义一个测试函数，测试负多项式分布的特性
def test_NegativeMultinomial():
    # 定义符号变量，要求为非负整数
    k0, x1, x2, x3, x4 = symbols('k0, x1, x2, x3, x4', nonnegative=True, integer=True)
    # 定义符号变量，要求为正数
    p1, p2, p3, p4 = symbols('p1, p2, p3, p4', positive=True)
    # 定义一个符号变量，要求为负数
    p1_f = symbols('p1_f', negative=True)
    # 创建一个负多项式分布对象 N，指定参数
    N = NegativeMultinomial('N', 4, [p1, p2, p3, p4])
    # 创建一个负多项式分布对象 C，指定参数
    C = NegativeMultinomial('C', 4, 0.1, 0.2, 0.3)
    # 导入 gamma 函数
    g = gamma
    # 导入 factorial 函数
    f = factorial
    # 断言：验证 N 的概率密度函数的简化结果是否为零
    assert simplify(density(N)(x1, x2, x3, x4) -
            p1**x1*p2**x2*p3**x3*p4**x4*(-p1 - p2 - p3 - p4 + 1)**4*g(x1 + x2 +
            x3 + x4 + 4)/(6*f(x1)*f(x2)*f(x3)*f(x4))) is S.Zero
    # 断言：验证 C 的边缘分布的计算结果是否接近于指定值
    assert comp(marginal_distribution(C, C[0])(1).evalf(), 0.33, .01)
    # 断言：验证创建具有无效参数的负多项式分布时是否引发 ValueError 异常
    raises(ValueError, lambda: NegativeMultinomial('b1', 5, [p1, p2, p3, p1_f]))
    # 断言：验证创建具有无效参数的负多项式分布时是否引发 ValueError 异常
    raises(ValueError, lambda: NegativeMultinomial('b2', k0, 0.5, 0.4, 0.3, 0.4))
    # 断言：验证 N 的概率空间分布的集合是否符合预期的乘积集合
    assert N.pspace.distribution.set == ProductSet(Range(0, oo, 1),
                    Range(0, oo, 1), Range(0, oo, 1), Range(0, oo, 1))


# 定义一个测试函数，测试联合概率空间和边缘分布的计算
@slow
def test_JointPSpace_marginal_distribution():
    # 创建一个多变量 t-分布对象 T，指定参数
    T = MultivariateT('T', [0, 0], [[1, 0], [0, 1]], 2)
    # 计算 T 的边缘分布函数，并用 x 表示
    got = marginal_distribution(T, T[1])(x)
    # 定义预期的边缘分布函数结果
    ans = sqrt(2)*(x**2/2 + 1)/(4*polar_lift(x**2/2 + 1)**(S(5)/2))
    # 断言：验证计算得到的边缘分布函数结果是否等于预期值
    assert got == ans, got
    # 断言：验证 T 的第一维的边缘分布函数积分结果是否为 1
    assert integrate(marginal_distribution(T, 1)(x), (x, -oo, oo)) == 1

    # 创建一个三维多变量 t-分布对象 t，指定参数
    t = MultivariateT('T', [0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)
    # 断言：验证 t 的第零维的边缘分布函数计算结果是否接近于指定值
    assert comp(marginal_distribution(t, 0)(1).evalf(), 0.2, .01)


# 定义一个测试函数，测试联合随机变量的期望计算
def test_JointRV():
    # 定义两个索引变量 x1 和 x2
    x1, x2 = (Indexed('x', i) for i in (1, 2))
    # 定义概率密度函数 pdf
    pdf = exp(-x1**2/2 + x1 - x2**2/2 - S.Half)/(2*pi)
    # 创建一个联合随机变量 X，指定概率密度函数
    X = JointRV('x', pdf)
    # 断言：验证 X 在给定点 (1, 2) 处的概率密度函数值是否等于预期值
    assert density(X)(1, 2) == exp(-2)/(2*pi)
    # 断言：验证 X 的概率空间分布是否为手工创建的联合分布对象
    assert isinstance(X.pspace.distribution, JointDistributionHandmade)
    # 断言：验证 X 的第零维的边缘分布函数在给定点 2 处的计算结果是否等于预期值
    assert marginal_distribution(X, 0)(2) == sqrt(2)*exp(Rational(-1, 2))/(2*sqrt(pi))


# 定义一个测试函数，测试正态分布的期望计算
def test_expectation():
    # 创建一个多元正态分布对象 m，指定均值和协方差矩阵
    m = Normal('A', [x, y], [[1, 0], [0, 1]])
    # 断言：验证 m 的第一维的期望是否简化为变量 y
    assert simplify(E(m[1])) == y


# 定义一个预期失败的测试函数，测试联合向量的期望计算
@XFAIL
def test_joint_vector_expectation():
    # 创建一个多元正态分布对象 m，指定均值和协方差矩阵
    m = Normal('A', [x, y], [[1, 0], [0, 1]])
    # 断言：验证 m 的期望是否为指定的向量 (x, y)
    assert E(m) == (x, y)


# 定义一个测试函数，测试使用 NumPy 库进行抽样
def test_sample_numpy():
    # 定义多个概率分布对象，包括多元正态分布、多元贝塔分布和多项式分布
    distribs_numpy = [
        MultivariateNormal("M", [3, 4], [[2, 1], [1, 2]]),
        MultivariateBeta("B", [0.4, 5, 15, 50, 203]),
        Multinomial("N", 50, [0.3, 0.2, 0.1, 0.25, 0.15])
    ]
    # 定义抽样的大小
    size = 3
    # 导入 NumPy 库
    numpy = import_module('numpy')
    # 如果没有安装 NumPy 库，则跳过测试
    if not numpy:
        skip('Numpy is not installed. Abort tests for _sample_numpy.')
    else:
        # 对每个概率分布对象进行抽样，使用 NumPy 库
        for X in distribs_numpy:
            samps = sample(X, size=size, library='numpy')
            # 对每个样本进行断言：验证样本是否属于概率空间分布的集合
            for sam in samps:
                assert tuple(sam) in X.pspace.distribution.set
        # 创建一个负多项式分布对象 N_c，指定参数
        N_c = NegativeMultinomial('N', 3, 0.1, 0.1, 0.1)
        # 断言：验证使用 NumPy 库抽样时创建 N_c 是否引发 NotImplementedError 异常
        raises(NotImplementedError, lambda: sample(N_c, library='numpy'))


# 定义一个测试函数，测试使用 SciPy 库进行抽样
    # 检查是否安装了 Scipy 库，如果没有则跳过测试并终止 _sample_scipy 的测试
    if not scipy:
        skip('Scipy not installed. Abort tests for _sample_scipy.')
    else:
        # 遍历 distribs_scipy 列表中的每个分布 X
        for X in distribs_scipy:
            # 从分布 X 中抽取指定大小的样本
            samps = sample(X, size=size)
            # 从分布 X 中抽取 2x2 大小的样本
            samps2 = sample(X, size=(2, 2))
            # 检查每个抽取的样本是否属于分布 X 的样本空间集合中
            for sam in samps:
                assert tuple(sam) in X.pspace.distribution.set
            # 检查 samps2 中每个元素是否属于分布 X 的样本空间集合中
            for i in range(2):
                for j in range(2):
                    assert tuple(samps2[i][j]) in X.pspace.distribution.set
        # 创建一个 NegativeMultinomial 对象 N_c，指定参数
        N_c = NegativeMultinomial('N', 3, 0.1, 0.1, 0.1)
        # 检查调用 sample(N_c) 是否会抛出 NotImplementedError 异常
        raises(NotImplementedError, lambda: sample(N_c))
# 定义一个测试函数，用于测试样本生成的 PyMC 库的函数
def test_sample_pymc():
    # 定义 PyMC 库中的三种概率分布
    distribs_pymc = [
        MultivariateNormal("M", [5, 2], [[1, 0], [0, 1]]),  # 多元正态分布
        MultivariateBeta("B", [0.4, 5, 15]),  # 多元贝塔分布
        Multinomial("N", 4, [0.3, 0.2, 0.1, 0.4])  # 多项分布
    ]
    size = 3  # 样本生成的大小
    pymc = import_module('pymc')  # 导入 PyMC 模块
    # 如果 PyMC 模块未安装，则跳过测试
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        # 遍历每种分布
        for X in distribs_pymc:
            # 从分布 X 中生成指定大小的样本，使用 PyMC 库
            samps = sample(X, size=size, library='pymc')
            # 对生成的每个样本进行断言，确保它们属于分布 X 的概率空间
            for sam in samps:
                assert tuple(sam.flatten()) in X.pspace.distribution.set
        # 创建一个负多项分布对象
        N_c = NegativeMultinomial('N', 3, 0.1, 0.1, 0.1)
        # 断言调用生成样本函数会抛出 NotImplementedError 异常
        raises(NotImplementedError, lambda: sample(N_c, library='pymc'))


# 定义一个测试函数，用于测试生成带有种子的样本
def test_sample_seed():
    # 定义两个索引变量 x1 和 x2
    x1, x2 = (Indexed('x', i) for i in (1, 2))
    # 定义一个数学表达式
    pdf = exp(-x1**2/2 + x1 - x2**2/2 - S.Half)/(2*pi)
    # 创建一个联合随机变量 X，其概率密度函数为 pdf
    X = JointRV('x', pdf)

    libraries = ['scipy', 'numpy', 'pymc']
    # 遍历每个库
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            # 如果成功导入该库
            if imported_lib:
                # 分别生成三组种子为 0 和 1 的样本，并进行断言检查
                s0, s1, s2 = [], [], []
                s0 = sample(X, size=10, library=lib, seed=0)
                s1 = sample(X, size=10, library=lib, seed=0)
                s2 = sample(X, size=10, library=lib, seed=1)
                assert all(s0 == s1)  # 断言 s0 和 s1 的样本相等
                assert all(s1 != s2)  # 断言 s1 和 s2 的样本不相等
        except NotImplementedError:
            continue

#
# XXX: This fails for pymc. Previously the test appeared to pass but that is
# just because the library argument was not passed so the test always used
# scipy.
#
# 定义一个测试函数，用于测试问题编号为 21057 的问题
def test_issue_21057():
    # 创建三个正态分布对象
    m = Normal("x", [0, 0], [[0, 0], [0, 0]])
    n = MultivariateNormal("x", [0, 0], [[0, 0], [0, 0]])
    p = Normal("x", [0, 0], [[0, 0], [0, 1]])
    # 断言 m 和 n 对象相等
    assert m == n
    # 定义三个库名称的元组
    libraries = ('scipy', 'numpy')  # , 'pymc')  # <-- pymc fails
    # 遍历每个库名称
    for library in libraries:
        try:
            imported_lib = import_module(library)
            # 如果成功导入该库
            if imported_lib:
                # 从 m, n, p 分布中生成指定大小的样本，使用当前库
                s1 = sample(m, size=8, library=library)
                s2 = sample(n, size=8, library=library)
                s3 = sample(p, size=8, library=library)
                # 断言 s1 和 s2 的样本相等
                assert tuple(s1.flatten()) == tuple(s2.flatten())
                # 对于 s3 中的每个样本，断言其属于分布 p 的概率空间
                for s in s3:
                    assert tuple(s.flatten()) in p.pspace.distribution.set
        except NotImplementedError:
            continue


#
# When this passes the pymc part can be uncommented in test_issue_21057 above
# and this can be deleted.
#
# 标记为预期失败的测试函数，用于测试问题编号为 21057 的 PyMC 问题
@XFAIL
def test_issue_21057_pymc():
    # 创建三个正态分布对象
    m = Normal("x", [0, 0], [[0, 0], [0, 0]])
    n = MultivariateNormal("x", [0, 0], [[0, 0], [0, 0]])
    p = Normal("x", [0, 0], [[0, 0], [0, 1]])
    # 断言 m 和 n 对象相等
    assert m == n
    # 定义一个包含唯一元素 'pymc' 的库名称元组
    libraries = ('pymc',)
    # 遍历给定的库列表
    for library in libraries:
        # 尝试导入当前遍历到的库
        try:
            imported_lib = import_module(library)
            # 如果成功导入库
            if imported_lib:
                # 从导入的库中抽取样本数据，每个样本大小为8
                s1 = sample(m, size=8, library=library)
                s2 = sample(n, size=8, library=library)
                s3 = sample(p, size=8, library=library)
                # 断言两个样本数据的展平后的元组相等
                assert tuple(s1.flatten()) == tuple(s2.flatten())
                # 遍历第三个样本集合
                for s in s3:
                    # 断言展平后的样本元组存在于概率空间分布的集合中
                    assert tuple(s.flatten()) in p.pspace.distribution.set
        # 如果抛出 NotImplementedError 异常，继续处理下一个库
        except NotImplementedError:
            continue
```