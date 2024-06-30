# `D:\src\scipysrc\sympy\sympy\stats\tests\test_compound_rv.py`

```
# 导入从 sympy.concrete.summations 模块中的 Sum 类
from sympy.concrete.summations import Sum
# 导入从 sympy.core.numbers 模块中的 oo（无穷大）和 pi（圆周率）常数
from sympy.core.numbers import (oo, pi)
# 导入从 sympy.core.relational 模块中的 Eq 类
from sympy.core.relational import Eq
# 导入从 sympy.core.singleton 模块中的 S（符号对象）类
from sympy.core.singleton import S
# 导入从 sympy.core.symbol 模块中的 symbols 函数
from sympy.core.symbol import symbols
# 导入从 sympy.functions.combinatorial.factorials 模块中的 factorial 函数
from sympy.functions.combinatorial.factorials import factorial
# 导入从 sympy.functions.elementary.exponential 模块中的 exp 函数
from sympy.functions.elementary.exponential import exp
# 导入从 sympy.functions.elementary.miscellaneous 模块中的 sqrt 函数
from sympy.functions.elementary.miscellaneous import sqrt
# 导入从 sympy.functions.elementary.piecewise 模块中的 Piecewise 类
from sympy.functions.elementary.piecewise import Piecewise
# 导入从 sympy.functions.special.beta_functions 模块中的 beta 函数
from sympy.functions.special.beta_functions import beta
# 导入从 sympy.functions.special.error_functions 模块中的 erf 函数
from sympy.functions.special.error_functions import erf
# 导入从 sympy.functions.special.gamma_functions 模块中的 gamma 函数
from sympy.functions.special.gamma_functions import gamma
# 导入从 sympy.integrals.integrals 模块中的 Integral 类
from sympy.integrals.integrals import Integral
# 导入从 sympy.sets.sets 模块中的 Interval 类
from sympy.sets.sets import Interval
# 导入从 sympy.stats 模块中的以下内容：Normal, P, E, density, Gamma, Poisson, Rayleigh,
# variance, Bernoulli, Beta, Uniform, cdf
from sympy.stats import (Normal, P, E, density, Gamma, Poisson, Rayleigh,
                        variance, Bernoulli, Beta, Uniform, cdf)
# 导入从 sympy.stats.compound_rv 模块中的 CompoundDistribution, CompoundPSpace 类
from sympy.stats.compound_rv import CompoundDistribution, CompoundPSpace
# 导入从 sympy.stats.crv_types 模块中的 NormalDistribution 类
from sympy.stats.crv_types import NormalDistribution
# 导入从 sympy.stats.drv_types 模块中的 PoissonDistribution 类
from sympy.stats.drv_types import PoissonDistribution
# 导入从 sympy.stats.frv_types 模块中的 BernoulliDistribution 类
from sympy.stats.frv_types import BernoulliDistribution
# 导入从 sympy.testing.pytest 模块中的 raises, ignore_warnings 函数
from sympy.testing.pytest import raises, ignore_warnings
# 导入从 sympy.stats.joint_rv_types 模块中的 MultivariateNormalDistribution 类
from sympy.stats.joint_rv_types import MultivariateNormalDistribution

# 导入从 sympy.abc 模块中的 x 符号
from sympy.abc import x


# 用于测试难以评估表达式的帮助函数
# 将输入的表达式展平成字符串后比较是否相等的 lambda 函数
flat = lambda s: ''.join(str(s).split())
# 检查多个表达式是否展平后相等的 lambda 函数
streq = lambda *a: len(set(map(flat, a))) == 1
# 断言检查 x 符号和 'x' 字符串是否相等，应为 True
assert streq(x, x)
# 断言检查 x 符号和 x + 1 表达式是否相等，应为 False
assert streq(x, 'x')
# 断言检查 x 符号和 'x' 字符串是否相等，应为 False
assert not streq(x, x + 1)


def test_normal_CompoundDist():
    # 创建均值为 1，标准差为 2 的正态分布随机变量 X
    X = Normal('X', 1, 2)
    # 创建均值为 X，标准差为 4 的正态分布随机变量 Y
    Y = Normal('X', X, 4)
    # 断言检查 Y 的概率密度函数在 x 处简化后是否等于指定表达式
    assert density(Y)(x).simplify() == sqrt(10)*exp(-x**2/40 + x/20 - S(1)/40)/(20*sqrt(pi))
    # 断言检查 Y 的期望是否等于 X 的均值
    assert E(Y) == 1  # it is always equal to mean of X
    # 断言检查 Y 大于 1 的概率是否等于 1/2
    assert P(Y > 1) == S(1)/2  # as 1 is the mean
    # 断言检查 Y 大于 5 的概率在简化后是否等于指定表达式
    assert P(Y > 5).simplify() ==  S(1)/2 - erf(sqrt(10)/5)/2
    # 断言检查 Y 的方差是否等于 X 的方差加上 4 的平方
    assert variance(Y) == variance(X) + 4**2  # 2**2 + 4**2
    # https://math.stackexchange.com/questions/1484451/
    # (Contains proof of E and variance computation)


def test_poisson_CompoundDist():
    # 定义正数和实数符号 k, t, y
    k, t, y = symbols('k t y', positive=True, real=True)
    # 创建参数为 k, t 的 Gamma 分布随机变量 G
    G = Gamma('G', k, t)
    # 创建以 G 为参数的泊松分布随机变量 D
    D = Poisson('P', G)
    # 断言检查 D 的概率密度函数在 y 处简化后是否等于指定表达式
    assert density(D)(y).simplify() == t**y*(t + 1)**(-k - y)*gamma(k + y)/(gamma(k)*gamma(y + 1))
    # https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture
    # 断言检查 D 的期望在简化后是否等于 k*t
    assert E(D).simplify() == k*t  # mean of NegativeBinomialDistribution


def test_bernoulli_CompoundDist():
    # 创建参数为 1, 2 的 Beta 分布随机变量 X
    X = Beta('X', 1, 2)
    # 创建以 X 为参数的伯努利分布随机变量 Y
    Y = Bernoulli('Y', X)
    # 断言检查 Y 的概率分布字典是否等于指定字典
    assert density(Y).dict == {0: S(2)/3, 1: S(1)/3}
    # 断言检查 Y 的期望是否等于 P(Y=1) 的概率，都应为 1/3
    assert E(Y) == P(Eq(Y, 1)) == S(1)/3
    # 断言检查 Y 的方差是否等于 2/9
    assert variance(Y) == S(2)/9
    # 断言检查 Y 的累积分布函数是否等于指定字典
    assert cdf(Y) == {0: S(2)/3, 1: 1}

    # 测试问题 8128
    # 创建参数为 1/2 的伯努利分布随机变量 a
    a = Bernoulli('a', S(1)/2)
    # 创建以 a 为参数的伯努利分布随机变量 b
    b = Bernoulli('b', a)
    # 断言检查 b 的概率分布字典是否等于指定字典
    assert density(b).dict == {0: S(1)/2, 1: S(1)/2}
    # 断言检查 b 大于 0.5 的概率是否等于 1/2
    # 创建 Rayleigh 分布对象 R，参数为 4
    R = Rayleigh('R', 4)
    # 创建以 R 为参数的正态分布对象 X，均值为 3
    X = Normal('X', 3, R)
    # 预期的答案表达式，包含分段函数和积分
    ans = '''
        Piecewise(((-sqrt(pi)*sinh(x/4 - 3/4) + sqrt(pi)*cosh(x/4 - 3/4))/(
        8*sqrt(pi)), Abs(arg(x - 3)) <= pi/4), (Integral(sqrt(2)*exp(-(x - 3)
        **2/(2*R**2))*exp(-R**2/32)/(32*sqrt(pi)), (R, 0, oo)), True))'''
    # 断言 X 的概率密度函数在 x 处的值等于预期答案 ans
    assert streq(density(X)(x), ans)

    # 表达式 expre，包含 X 和 R 的嵌套积分
    expre = '''
        Integral(X*Integral(sqrt(2)*exp(-(X-3)**2/(2*R**2))*exp(-R**2/32)/(32*
        sqrt(pi)),(R,0,oo)),(X,-oo,oo))'''
    # 忽略用户警告并断言 E(X, evaluate=False) 重写为积分的表达式等于 expre
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert streq(E(X, evaluate=False).rewrite(Integral), expre)

    # 创建参数为 1 的泊松分布对象 X
    X = Poisson('X', 1)
    # Y 是以 X 为参数的泊松分布对象
    Y = Poisson('Y', X)
    # Z 是以 Y 为参数的泊松分布对象
    Z = Poisson('Z', Y)
    # 表达式 exprd，包含 Y 和 X 的嵌套求和和积分
    exprd = Sum(exp(-Y)*Y**x*Sum(exp(-1)*exp(-X)*X**Y/(factorial(X)*factorial(Y)
                ), (X, 0, oo))/factorial(x), (Y, 0, oo))
    # 断言 Z 的概率密度函数在 x 处的值等于表达式 exprd
    assert density(Z)(x) == exprd

    # 创建均值为 1，标准差为 2 的正态分布对象 N
    N = Normal('N', 1, 2)
    # 创建均值为 3，标准差为 4 的正态分布对象 M
    M = Normal('M', 3, 4)
    # 创建以 M 和 N 为参数的正态分布对象 D
    D = Normal('D', M, N)
    # 表达式 exprd，包含 N 和 M 的嵌套积分
    exprd = '''
        Integral(sqrt(2)*exp(-(N-1)**2/8)*Integral(exp(-(x-M)**2/(2*N**2))*exp
        (-(M-3)**2/32)/(8*pi*N),(M,-oo,oo))/(4*sqrt(pi)),(N,-oo,oo))'''
    # 断言 D 的概率密度函数在 x 处的值（不进行评估）等于表达式 exprd
    assert streq(density(D, evaluate=False)(x), exprd)
# 定义测试函数 test_Compound_Distribution，用于测试复合分布的功能
def test_Compound_Distribution():
    # 创建一个均值为2，方差为4的正态分布随机变量 X
    X = Normal('X', 2, 4)
    # 使用 X 创建一个正态分布 N
    N = NormalDistribution(X, 4)
    # 创建一个复合分布 C，其基础分布为 N
    C = CompoundDistribution(N)
    # 断言 C 是连续分布
    assert C.is_Continuous
    # 断言 C 的取值范围是负无穷到正无穷
    assert C.set == Interval(-oo, oo)
    # 断言 C 对应的概率密度函数在给定 x 的情况下，简化后等于指定的表达式
    assert C.pdf(x, evaluate=True).simplify() == exp(-x**2/64 + x/16 - S(1)/16)/(8*sqrt(pi))

    # 断言使用正态分布创建的复合分布不是 CompoundDistribution 类的实例
    assert not isinstance(CompoundDistribution(NormalDistribution(2, 3)),
                            CompoundDistribution)
    # 创建一个均值为 [1, 2]，协方差矩阵为 [[2, 1], [1, 2]] 的多元正态分布 M
    M = MultivariateNormalDistribution([1, 2], [[2, 1], [1, 2]])
    # 使用 lambda 函数断言创建 M 的复合分布会引发 NotImplementedError
    raises(NotImplementedError, lambda: CompoundDistribution(M))

    # 创建一个参数为 2 和 4 的贝塔分布 X
    X = Beta('X', 2, 4)
    # 使用 X 创建一个伯努利分布 B
    B = BernoulliDistribution(X, 1, 0)
    # 创建一个复合分布 C，其基础分布为 B
    C = CompoundDistribution(B)
    # 断言 C 是有限分布
    assert C.is_Finite
    # 断言 C 的取值集合是 {0, 1}
    assert C.set == {0, 1}
    # 创建一个非负整数符号 y
    y = symbols('y', negative=False, integer=True)
    # 断言 C 的概率密度函数在给定 y 的情况下，等于指定的分段函数
    assert C.pdf(y, evaluate=True) == Piecewise((S(1)/(30*beta(2, 4)), Eq(y, 0)),
                (S(1)/(60*beta(2, 4)), Eq(y, 1)), (0, True))

    # 创建正数和实数符号 k, t, z
    k, t, z = symbols('k t z', positive=True, real=True)
    # 创建一个参数为 k 和 t 的 Gamma 分布 G
    G = Gamma('G', k, t)
    # 使用 G 创建一个泊松分布 X
    X = PoissonDistribution(G)
    # 创建一个复合分布 C，其基础分布为 X
    C = CompoundDistribution(X)
    # 断言 C 是离散分布
    assert C.is_Discrete
    # 断言 C 的取值集合是非负整数集合 S.Naturals0
    assert C.set == S.Naturals0
    # 断言 C 的概率密度函数在给定 z 的情况下，简化后等于指定的表达式
    assert C.pdf(z, evaluate=True).simplify() == t**z*(t + 1)**(-k - z)*gamma(k \
                    + z)/(gamma(k)*gamma(z + 1))


# 定义测试函数 test_compound_pspace，用于测试复合概率空间的功能
def test_compound_pspace():
    # 创建均值为 2，方差为 4 的正态分布随机变量 X
    X = Normal('X', 2, 4)
    # 创建均值为 3，方差为 6 的正态分布随机变量 Y
    Y = Normal('Y', 3, 6)
    # 断言 Y 的概率空间不是 CompoundPSpace 类的实例
    assert not isinstance(Y.pspace, CompoundPSpace)
    # 创建均值为 1，方差为 2 的正态分布 N
    N = NormalDistribution(1, 2)
    # 创建参数为 3 的泊松分布 D
    D = PoissonDistribution(3)
    # 创建参数为 0.2 的伯努利分布 B
    B = BernoulliDistribution(0.2, 1, 0)
    # 创建命名为 'N'，基础分布为 N 的复合概率空间 pspace1
    pspace1 = CompoundPSpace('N', N)
    # 创建命名为 'D'，基础分布为 D 的复合概率空间 pspace2
    pspace2 = CompoundPSpace('D', D)
    # 创建命名为 'B'，基础分布为 B 的复合概率空间 pspace3
    pspace3 = CompoundPSpace('B', B)
    # 断言 pspace1 不是 CompoundPSpace 类的实例
    assert not isinstance(pspace1, CompoundPSpace)
    # 断言 pspace2 不是 CompoundPSpace 类的实例
    assert not isinstance(pspace2, CompoundPSpace)
    # 断言 pspace3 不是 CompoundPSpace 类的实例
    assert not isinstance(pspace3, CompoundPSpace)
    # 创建均值为 [1, 2]，协方差矩阵为 [[2, 1], [1, 2]] 的多元正态分布 M
    M = MultivariateNormalDistribution([1, 2], [[2, 1], [1, 2]])
    # 使用 lambda 函数断言创建 M 的复合概率空间会引发 ValueError
    raises(ValueError, lambda: CompoundPSpace('M', M))
    # 创建均值为 X，方差为 6 的正态分布 Y
    Y = Normal('Y', X, 6)
    # 断言 Y 的概率空间是 CompoundPSpace 类的实例
    assert isinstance(Y.pspace, CompoundPSpace)
    # 断言 Y 的概率空间的分布等于使用 NormalDistribution(X, 6) 创建的复合分布
    assert Y.pspace.distribution == CompoundDistribution(NormalDistribution(X, 6))
    # 断言 Y 的概率空间的定义域集合是负无穷到正无穷的区间
    assert Y.pspace.domain.set == Interval(-oo, oo)
```