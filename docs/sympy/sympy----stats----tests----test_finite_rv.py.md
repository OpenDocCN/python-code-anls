# `D:\src\scipysrc\sympy\sympy\stats\tests\test_finite_rv.py`

```
# 导入从 sympy 模块中需要的具体类和函数
from sympy.concrete.summations import Sum
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.beta_functions import beta
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polytools import cancel
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.matrices import Matrix
from sympy.stats import (DiscreteUniform, Die, Bernoulli, Coin, Binomial, BetaBinomial,
                         Hypergeometric, Rademacher, IdealSoliton, RobustSoliton, P, E, variance,
                         covariance, skewness, density, where, FiniteRV, pspace, cdf,
                         correlation, moment, cmoment, smoment, characteristic_function,
                         moment_generating_function, quantile,  kurtosis, median, coskewness)
from sympy.stats.frv_types import DieDistribution, BinomialDistribution, \
    HypergeometricDistribution
from sympy.stats.rv import Density
from sympy.testing.pytest import raises

# 定义贝叶斯测试函数
def BayesTest(A, B):
    # 断言条件概率 P(A, B) 的定义
    assert P(A, B) == P(And(A, B)) / P(B)
    # 断言贝叶斯公式
    assert P(A, B) == P(B, A) * P(A) / P(B)

# 定义离散均匀分布的测试函数
def test_discreteuniform():
    # 符号变量的声明
    a, b, c, t = symbols('a b c t')
    # 创建离散均匀分布随机变量 X
    X = DiscreteUniform('X', [a, b, c])

    # 断言随机变量 X 的期望
    assert E(X) == (a + b + c)/3
    # 断言随机变量 X 的方差
    assert simplify(variance(X)
                    - ((a**2 + b**2 + c**2)/3 - (a/3 + b/3 + c/3)**2)) == 0
    # 断言随机变量 X 等于 a, b, c 的概率
    assert P(Eq(X, a)) == P(Eq(X, b)) == P(Eq(X, c)) == S('1/3')

    # 创建离散均匀分布随机变量 Y
    Y = DiscreteUniform('Y', range(-5, 5))

    # 断言随机变量 Y 的期望
    assert E(Y) == S('-1/2')
    # 断言随机变量 Y 的方差
    assert variance(Y) == S('33/4')
    # 断言随机变量 Y 的中位数
    assert median(Y) == FiniteSet(-1, 0)

    # 遍历随机变量 Y 的取值范围
    for x in range(-5, 5):
        # 断言随机变量 Y 等于 x 的概率
        assert P(Eq(Y, x)) == S('1/10')
        # 断言随机变量 Y 小于等于 x 的概率
        assert P(Y <= x) == S(x + 6)/10
        # 断言随机变量 Y 大于等于 x 的概率
        assert P(Y >= x) == S(5 - x)/10

    # 检查两个分布的密度函数是否一致
    assert dict(density(Die('D', 6)).items()) == \
           dict(density(DiscreteUniform('U', range(1, 7))).items())

    # 断言随机变量 X 的特征函数
    assert characteristic_function(X)(t) == exp(I*a*t)/3 + exp(I*b*t)/3 + exp(I*c*t)/3
    # 断言随机变量 X 的矩生成函数
    assert moment_generating_function(X)(t) == exp(a*t)/3 + exp(b*t)/3 + exp(c*t)/3
    # 检查问题 18611 的异常抛出
    raises(ValueError, lambda: DiscreteUniform('Z', [a, a, a, b, b, c]))

# 定义骰子的测试函数
def test_dice():
    # TODO: Make iid method!（待完成：创建独立同分布的方法）
    # 创建三个骰子 X, Y, Z
    X, Y, Z = Die('X', 6), Die('Y', 6), Die('Z', 6)
    # 符号变量的声明
    a, b, t, p = symbols('a b t p')

    # 断言骰子 X 的期望
    assert E(X) == 3 + S.Half
    # 断言骰子 X 的方差
    assert variance(X) == Rational(35, 12)
    # 断言两个骰子 X + Y 的期望
    assert E(X + Y) == 7
    # 断言两个骰子 X + X 的期望
    assert E(X + X) == 7


这段代码注释详细解释了每行代码的功能和作用，符合要求将注释和代码块结合在一起，不进行额外的总结或者注释代码块之外的内容。
    断言：期望值公式的线性性质验证
    assert E(a*X + b) == a*E(X) + b

    断言：方差的加法性质和二阶中心矩的关系验证
    assert variance(X + Y) == variance(X) + variance(Y) == cmoment(X + Y, 2)

    断言：方差的可加性和二阶中心矩的关系验证
    assert variance(X + X) == 4 * variance(X) == cmoment(X + X, 2)

    断言：零阶中心矩的定义验证
    assert cmoment(X, 0) == 1

    断言：四倍缩放后的三阶中心矩的关系验证
    assert cmoment(4*X, 3) == 64*cmoment(X, 3)

    断言：两个不相关随机变量的协方差为零
    assert covariance(X, Y) is S.Zero

    断言：随机变量与其自身和另一随机变量之和的协方差与方差的关系验证
    assert covariance(X, X + Y) == variance(X)

    断言：密度函数值为真的概率验证
    assert density(Eq(cos(X*S.Pi), 1))[True] == S.Half

    断言：两个随机变量的相关系数为零
    assert correlation(X, Y) == 0

    断言：相关系数的对称性验证
    assert correlation(X, Y) == correlation(Y, X)

    断言：混合矩的三阶中心矩和斜度的关系验证
    assert smoment(X + Y, 3) == skewness(X + Y)

    断言：混合矩的四阶中心矩和峰度的关系验证
    assert smoment(X + Y, 4) == kurtosis(X + Y)

    断言：零阶中心矩的定义验证
    assert smoment(X, 0) == 1

    断言：随机变量大于3的概率验证
    assert P(X > 3) == S.Half

    断言：随机变量大于3的概率验证（倍数情况）
    assert P(2*X > 6) == S.Half

    断言：随机变量X大于Y的概率验证
    assert P(X > Y) == Rational(5, 12)

    断言：随机变量X等于Y的概率验证
    assert P(Eq(X, Y)) == P(Eq(X, 1))

    断言：条件期望值的计算验证
    assert E(X, X > 3) == 5 == moment(X, 1, 0, X > 3)

    断言：条件期望值的计算验证（条件为另一随机变量）
    assert E(X, Y > 3) == E(X) == moment(X, 1, 0, Y > 3)

    断言：随机变量和自身与随机变量相等的期望值关系验证
    assert E(X + Y, Eq(X, Y)) == E(2*X)

    断言：零阶矩的定义验证
    assert moment(X, 0) == 1

    断言：缩放后随机变量二阶矩和原随机变量二阶矩的关系验证
    assert moment(5*X, 2) == 25*moment(X, 2)

    断言：分位数函数的定义验证
    assert quantile(X)(p) == Piecewise((nan, (p > 1) | (p < 0)),\
        (S.One, p <= Rational(1, 6)), (S(2), p <= Rational(1, 3)), (S(3), p <= S.Half),\
        (S(4), p <= Rational(2, 3)), (S(5), p <= Rational(5, 6)), (S(6), p <= 1))

    断言：条件概率的定义验证
    assert P(X > 3, X > 3) is S.One

    断言：条件概率的计算验证
    assert P(X > Y, Eq(Y, 6)) is S.Zero

    断言：同时满足两个方程的概率验证
    assert P(Eq(X + Y, 12)) == Rational(1, 36)

    断言：同时满足两个方程的概率验证（条件）
    assert P(Eq(X + Y, 12), Eq(X, 6)) == Rational(1, 6)

    断言：随机变量和其它随机变量的密度函数关系验证
    assert density(X + Y) == density(Y + Z) != density(X + X)

    断言：随机变量线性组合的密度函数值验证
    d = density(2*X + Y**Z)
    assert d[S(22)] == Rational(1, 108) and d[S(4100)] == Rational(1, 216) and S(3130) not in d

    断言：随机变量的概率空间验证
    assert pspace(X).domain.as_boolean() == Or(
        *[Eq(X.symbol, i) for i in [1, 2, 3, 4, 5, 6]])

    断言：满足条件的随机变量集合验证
    assert where(X > 3).set == FiniteSet(4, 5, 6)

    断言：特征函数的定义验证
    assert characteristic_function(X)(t) == exp(6*I*t)/6 + exp(5*I*t)/6 + exp(4*I*t)/6 + exp(3*I*t)/6 + exp(2*I*t)/6 + exp(I*t)/6

    断言：生成函数的定义验证
    assert moment_generating_function(X)(t) == exp(6*t)/6 + exp(5*t)/6 + exp(4*t)/6 + exp(3*t)/6 + exp(2*t)/6 + exp(t)/6

    断言：中位数的定义验证
    assert median(X) == FiniteSet(3, 4)

    断言：骰子对象的中位数验证
    D = Die('D', 7)
    assert median(D) == FiniteSet(4)

    断言：贝叶斯测试的执行验证
    BayesTest(X > 3, X + Y < 5)
    BayesTest(Eq(X - Y, Z), Z > Y)
    BayesTest(X > 3, X > 2)

    # 骰子对象的参数验证
    raises(ValueError, lambda: Die('X', -1))  # issue 8105: negative sides.
    raises(ValueError, lambda: Die('X', 0))
    raises(ValueError, lambda: Die('X', 1.5))  # issue 8103: non integer sides.

    # 骰子对象的符号验证
    n, k = symbols('n, k', positive=True)
    D = Die('D', n)
    dens = density(D).dict
    assert dens == Density(DieDistribution(n))
    assert set(dens.subs(n, 4).doit().keys()) == {1, 2, 3, 4}
    assert set(dens.subs(n, 4).doit().values()) == {Rational(1, 4)}
    k = Dummy('k', integer=True)
    assert E(D).dummy_eq(
        Sum(Piecewise((k/n, k <= n), (0, True)), (k, 1, n)))
    assert variance(D).subs(n, 6).doit() == Rational(35, 12)

    ki = Dummy('ki')
    cumuf = cdf(D)(k)
    assert cumuf.dummy_eq(
    # 计算和式，使用分段函数表示
    Sum(Piecewise((1/n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, k)))
    # 断言验证累积和的计算结果是否等于有理数 1/3，当 n=6, k=2 时
    assert cumuf.subs({n: 6, k: 2}).doit() == Rational(1, 3)

    # 创建一个虚拟变量 t
    t = Dummy('t')
    # 计算特征函数，使用虚拟变量 t
    cf = characteristic_function(D)(t)
    # 断言验证特征函数的表达式是否等价于给定的和式，当 n 是变量，ki 的范围是 1 到 n
    assert cf.dummy_eq(
    Sum(Piecewise((exp(ki*I*t)/n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, n)))
    # 断言验证特征函数在 n=3 时的具体计算结果
    assert cf.subs(n, 3).doit() == exp(3*I*t)/3 + exp(2*I*t)/3 + exp(I*t)/3

    # 计算矩生成函数，使用虚拟变量 t
    mgf = moment_generating_function(D)(t)
    # 断言验证矩生成函数的表达式是否等价于给定的和式，当 n 是变量，ki 的范围是 1 到 n
    assert mgf.dummy_eq(
    Sum(Piecewise((exp(ki*t)/n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, n)))
    # 断言验证矩生成函数在 n=3 时的具体计算结果
    assert mgf.subs(n, 3).doit() == exp(3*t)/3 + exp(2*t)/3 + exp(t)/3
def test_given():
    # 创建一个名为 X 的六面骰子对象
    X = Die('X', 6)
    # 断言当 X 大于 5 时，密度函数返回 {6: 1}
    assert density(X, X > 5) == {S(6): S.One}
    # 断言当 X 大于 2 时，where 函数返回的条件化对象转化为布尔表达式 X = 6
    assert where(X > 2, X > 5).as_boolean() == Eq(X.symbol, 6)


def test_domains():
    # 创建两个六面骰子对象 X 和 Y
    X, Y = Die('x', 6), Die('y', 6)
    x, y = X.symbol, Y.symbol
    # Domains
    # 创建条件对象 d，其中 X 大于 Y 的条件为 x > y
    d = where(X > Y)
    assert d.condition == (x > y)
    # 创建条件对象 d，其中同时满足 X > Y 和 Y > 3 的条件
    d = where(And(X > Y, Y > 3))
    # 断言条件对象 d 转化为布尔表达式的结果，等同于逻辑或的组合
    assert d.as_boolean() == Or(And(Eq(x, 5), Eq(y, 4)), And(Eq(x, 6),
        Eq(y, 5)), And(Eq(x, 6), Eq(y, 4)))
    # 断言条件对象 d 的元素个数为 3
    assert len(d.elements) == 3

    # 断言 X + Y 的概率空间的元素个数为 36
    assert len(pspace(X + Y).domain.elements) == 36

    # 创建一个四面骰子 Z，预期会抛出 ValueError 异常，因为与 X 的内部符号相同
    raises(ValueError, lambda: P(X > Z))  # Two domains with same internal symbol

    # 断言 X + Y 的概率空间的集合等于有序对 (1, 2, 3, 4, 5, 6) 的笛卡尔积
    assert pspace(X + Y).domain.set == FiniteSet(1, 2, 3, 4, 5, 6)**2

    # 断言 where(X > 3) 的集合等于 {4, 5, 6}
    assert where(X > 3).set == FiniteSet(4, 5, 6)
    # 断言 X 的概率空间的字典表示为 {1: X, 2: X, ..., 6: X}
    assert X.pspace.domain.dict == FiniteSet(
        *[Dict({X.symbol: i}) for i in range(1, 7)])

    # 断言 where(X > Y) 的字典表示为 {(i, j) | 1 <= i, j <= 6 and i > j}
    assert where(X > Y).dict == FiniteSet(*[Dict({X.symbol: i, Y.symbol: j})
            for i in range(1, 7) for j in range(1, 7) if i > j])

def test_bernoulli():
    p, a, b, t = symbols('p a b t')
    # 创建一个参数为 p 的伯努利随机变量 X
    X = Bernoulli('B', p, a, b)

    # 断言 X 的期望 E(X) 等于 a*p + b*(-p + 1)
    assert E(X) == a*p + b*(-p + 1)
    # 断言 X 的密度函数在 a 处的取值等于 p
    assert density(X)[a] == p
    # 断言 X 的密度函数在 b 处的取值等于 1 - p
    assert density(X)[b] == 1 - p
    # 断言 X 的特征函数在 t 处的取值等于 p * exp(I * a * t) + (-p + 1) * exp(I * b * t)
    assert characteristic_function(X)(t) == p * exp(I * a * t) + (-p + 1) * exp(I * b * t)
    # 断言 X 的矩生成函数在 t 处的取值等于 p * exp(a * t) + (-p + 1) * exp(b * t)

    assert moment_generating_function(X)(t) == p * exp(a * t) + (-p + 1) * exp(b * t)

    # 创建一个参数为 p，a=1，b=0 的伯努利随机变量 X
    X = Bernoulli('B', p, 1, 0)
    z = Symbol("z")

    # 断言 X 的期望 E(X) 等于 p
    assert E(X) == p
    # 断言 X 的方差简化后等于 p*(1 - p)
    assert simplify(variance(X)) == p*(1 - p)
    # 断言 E(a*X + b) 等于 a*E(X) + b
    assert E(a*X + b) == a*E(X) + b
    # 断言 a*X + b 的方差简化后等于 a^2 * variance(X)
    assert simplify(variance(a*X + b)) == simplify(a**2 * variance(X))
    # 断言 X 的分位数函数在 z 处的取值满足分段函数条件
    assert quantile(X)(z) == Piecewise((nan, (z > 1) | (z < 0)), (0, z <= 1 - p), (1, z <= 1))
    # 创建一个参数为 Rational(1, 2) 的伯努利随机变量 Y
    Y = Bernoulli('Y', Rational(1, 2))
    # 断言 Y 的中位数等于 {0, 1}
    assert median(Y) == FiniteSet(0, 1)
    # 创建一个参数为 Rational(2, 3) 的伯努利随机变量 Z
    Z = Bernoulli('Z', Rational(2, 3))
    # 断言 Z 的中位数等于 {1}
    assert median(Z) == FiniteSet(1)
    # 断言创建伯努利随机变量时出现的 ValueError 异常，参数超出合理范围
    raises(ValueError, lambda: Bernoulli('B', 1.5))
    raises(ValueError, lambda: Bernoulli('B', -0.5))

    # issue 8248
    # 断言 X 的概率空间计算期望为 1
    assert X.pspace.compute_expectation(1) == 1

    p = Rational(1, 5)
    # 创建一个参数为 p 的二项分布随机变量 X
    X = Binomial('X', 5, p)
    # 创建一个参数为 2*p 的二项分布随机变量 Y
    Y = Binomial('Y', 7, 2*p)
    # 创建一个参数为 3*p 的二项分布随机变量 Z
    Z = Binomial('Z', 9, 3*p)
    # 断言三个随机变量的共偏度函数计算简化后为 0
    assert coskewness(Y + Z, X + Y, X + Z).simplify() == 0
    # 断言四个随机变量的共偏度函数计算简化后满足给定的数学表达式
    assert coskewness(Y + 2*X + Z, X + 2*Y + Z, X + 2*Z + Y).simplify() == \
                        sqrt(1529)*Rational(12, 16819)
    # 断言四个随机变量的共偏度函数在给定条件下计算简化后满足给定的数学表达式
    assert coskewness(Y + 2*X + Z, X + 2*Y + Z, X + 2*Z + Y, X < 2).simplify() \
                        == -sqrt(357451121)*Rational(2812, 4646864573)

def test_cdf():
    # 创建一个六面骰子对象 D
    D = Die('D', 6)
    o = S.One

    # 断言 D 的累积分布函数 cdf 返回的字典等于给定的数学表达式
    assert cdf(
        D) == sympify({1: o/6, 2: o/3,
    # 断言，验证事件 F 等于事件 H 的概率是否等于有理数 1/10
    assert P(Eq(F, H)) == Rational(1, 10)
    
    # 获取随机变量 C 的概率空间的定义域
    d = pspace(C).domain
    
    # 断言，验证概率空间的定义域是否为条件 C 的符号等于 H 或者 C 的符号等于 T
    assert d.as_boolean() == Or(Eq(C.symbol, H), Eq(C.symbol, T))
    
    # 断言，验证当条件 C 大于条件 D 时是否会引发 ValueError 异常
    raises(ValueError, lambda: P(C > D))  # 无法智能地比较 H 和 T
# 验证二项分布参数的合法性
def test_binomial_verify_parameters():
    # 测试当参数 'n' 不合法时是否引发 ValueError 异常
    raises(ValueError, lambda: Binomial('b', .2, .5))
    # 测试当参数 'p' 不合法时是否引发 ValueError 异常
    raises(ValueError, lambda: Binomial('b', 3, 1.5))

# 测试二项分布在数值情况下的性质
def test_binomial_numeric():
    # 'n' 取值范围为 0 到 4
    nvals = range(5)
    # 'p' 取值范围包括 0, 1/4, 1/2, 3/4, 1
    pvals = [0, Rational(1, 4), S.Half, Rational(3, 4), 1]

    # 遍历所有 'n' 和 'p' 的组合
    for n in nvals:
        for p in pvals:
            # 创建二项分布随机变量 X
            X = Binomial('X', n, p)
            # 验证期望值 E(X) 的计算是否正确
            assert E(X) == n*p
            # 验证方差 variance(X) 的计算是否正确
            assert variance(X) == n*p*(1 - p)
            # 如果 n > 0 且 0 < p < 1，则验证偏度 skewness(X) 的计算是否正确
            if n > 0 and 0 < p < 1:
                assert skewness(X) == (1 - 2*p)/sqrt(n*p*(1 - p))
                # 验证峰度 kurtosis(X) 的计算是否正确
                assert kurtosis(X) == 3 + (1 - 6*p*(1 - p))/(n*p*(1 - p))
            # 遍历所有可能的成功次数 k，验证概率 P(X = k) 的计算是否正确
            for k in range(n + 1):
                assert P(Eq(X, k)) == binomial(n, k)*p**k*(1 - p)**(n - k)

# 测试二项分布的分位数计算
def test_binomial_quantile():
    # 创建二项分布随机变量 X，n = 50, p = 1/2
    X = Binomial('X', 50, S.Half)
    # 验证分位数的计算是否正确
    assert quantile(X)(0.95) == S(31)
    # 验证中位数的计算是否正确
    assert median(X) == FiniteSet(25)

    # 创建二项分布随机变量 X，n = 5, p = 1/2
    X = Binomial('X', 5, S.Half)
    # 创建符号 p 用于计算分位数
    p = Symbol("p", positive=True)
    # 验证分位数的计算是否正确
    assert quantile(X)(p) == Piecewise((nan, p > S.One), (S.Zero, p <= Rational(1, 32)),\
        (S.One, p <= Rational(3, 16)), (S(2), p <= S.Half), (S(3), p <= Rational(13, 16)),\
        (S(4), p <= Rational(31, 32)), (S(5), p <= S.One))
    # 验证中位数的计算是否正确
    assert median(X) == FiniteSet(2, 3)

# 测试二项分布的符号计算
def test_binomial_symbolic():
    # 固定 n = 2, p 为符号变量
    n = 2
    p = symbols('p', positive=True)
    # 创建二项分布随机变量 X
    X = Binomial('X', n, p)
    t = Symbol('t')

    # 验证期望值 E(X) 的简化是否正确
    assert simplify(E(X)) == n*p == simplify(moment(X, 1))
    # 验证方差 variance(X) 的简化是否正确
    assert simplify(variance(X)) == n*p*(1 - p) == simplify(cmoment(X, 2))
    # 验证偏度 skewness(X) 的计算是否正确
    assert cancel(skewness(X) - (1 - 2*p)/sqrt(n*p*(1 - p))) == 0
    # 验证峰度 kurtosis(X) 的计算是否正确
    assert cancel((kurtosis(X)) - (3 + (1 - 6*p*(1 - p))/(n*p*(1 - p)))) == 0
    # 验证特征函数 characteristic_function(X) 的计算是否正确
    assert characteristic_function(X)(t) == p ** 2 * exp(2 * I * t) + 2 * p * (-p + 1) * exp(I * t) + (-p + 1) ** 2
    # 验证矩母函数 moment_generating_function(X) 的计算是否正确
    assert moment_generating_function(X)(t) == p ** 2 * exp(2 * t) + 2 * p * (-p + 1) * exp(t) + (-p + 1) ** 2

    # 测试改变成功/失败赢得的能力
    H, T = symbols('H T')
    # 创建具有自定义成功/失败值的二项分布随机变量 Y
    Y = Binomial('Y', n, p, succ=H, fail=T)
    # 验证期望值 E(Y) 的计算是否正确
    assert simplify(E(Y) - (n*(H*p + T*(1 - p)))) == 0

    # 测试符号维度
    n = symbols('n')
    B = Binomial('B', n, p)
    # 验证大于2的概率的计算是否抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: P(B > 2))
    # 验证密度函数的字典表示是否正确
    assert density(B).dict == Density(BinomialDistribution(n, p, 1, 0))
    # 验证在 n = 4 时，密度函数的键集合是否正确
    assert set(density(B).dict.subs(n, 4).doit().keys()) == \
    {S.Zero, S.One, S(2), S(3), S(4)}
    # 验证在 n = 4 时，密度函数的值集合是否正确
    assert set(density(B).dict.subs(n, 4).doit().values()) == \
    {(1 - p)**4, 4*p*(1 - p)**3, 6*p**2*(1 - p)**2, 4*p**3*(1 - p), p**4}
    k = Dummy('k', integer=True)
    # 验证大于2的概率的计算是否正确
    assert E(B > 2).dummy_eq(
        Sum(Piecewise((k*p**k*(1 - p)**(-k + n)*binomial(n, k), (k >= 0)
        & (k <= n) & (k > 2)), (0, True)), (k, 0, n)))

# 测试贝塔二项分布的特性
def test_beta_binomial():
    # 验证贝塔二项分布参数的合法性
    raises(ValueError, lambda: BetaBinomial('b', .2, 1, 2))
    raises(ValueError, lambda: BetaBinomial('b', 2, -1, 2))
    raises(ValueError, lambda: BetaBinomial('b', 2, 1, -2))
    # 验证合法参数下是否创建成功
    assert BetaBinomial('b', 2, 1, 1)

    # 测试数值情况
    nvals = range(1,5)
    # 定义 alpha 和 beta 的可能取值列表，包括有理数和整数
    alphavals = [Rational(1, 4), S.Half, Rational(3, 4), 1, 10]
    betavals = [Rational(1, 4), S.Half, Rational(3, 4), 1, 10]

    # 遍历 nvals 中的每个 n
    for n in nvals:
        # 遍历 alphavals 中的每个 alpha
        for a in alphavals:
            # 遍历 betavals 中的每个 beta
            for b in betavals:
                # 创建一个 BetaBinomial 分布变量 X，其中参数为 n, a, b
                X = BetaBinomial('X', n, a, b)
                # 断言期望值 E(X) 等于一阶矩 moment(X, 1)
                assert E(X) == moment(X, 1)
                # 断言方差 variance(X) 等于中心矩 cmoment(X, 2)
                assert variance(X) == cmoment(X, 2)

    # 测试符号计算部分
    # 定义符号变量 n, a, b
    n, a, b = symbols('a b n')
    # 断言创建 BetaBinomial 分布变量 'x' 成功
    assert BetaBinomial('x', n, a, b)
    # 因为使用了循环，不能使用符号变量 n，所以将 n 设为 2
    n = 2
    # 定义 a 和 b 为正数符号变量
    a, b = symbols('a b', positive=True)
    # 创建 BetaBinomial 分布变量 X，参数为 n, a, b
    X = BetaBinomial('X', n, a, b)
    # 定义符号变量 t
    t = Symbol('t')

    # 断言期望值 E(X) 的展开等于一阶矩 moment(X, 1) 的展开
    assert E(X).expand() == moment(X, 1).expand()
    # 断言方差 variance(X) 的展开等于中心矩 cmoment(X, 2) 的展开
    assert variance(X).expand() == cmoment(X, 2).expand()
    # 断言偏度 skewness(X) 等于三阶矩 smoment(X, 3)
    assert skewness(X) == smoment(X, 3)
    # 断言特征函数 characteristic_function(X)(t) 的值等于指定表达式
    assert characteristic_function(X)(t) == exp(2*I*t)*beta(a + 2, b)/beta(a, b) +\
         2*exp(I*t)*beta(a + 1, b + 1)/beta(a, b) + beta(a, b + 2)/beta(a, b)
    # 断言动差生成函数 moment_generating_function(X)(t) 的值等于指定表达式
    assert moment_generating_function(X)(t) == exp(2*t)*beta(a + 2, b)/beta(a, b) +\
         2*exp(t)*beta(a + 1, b + 1)/beta(a, b) + beta(a, b + 2)/beta(a, b)
# 定义一个测试函数，用于测试超几何分布的数值计算
def test_hypergeometric_numeric():
    # N 在范围 [1, 5) 中循环，表示总体大小
    for N in range(1, 5):
        # m 在范围 [0, N] 中循环，表示样本中成功事件的数量
        for m in range(0, N + 1):
            # n 在范围 [1, N] 中循环，表示样本中抽取的总数量
            for n in range(1, N + 1):
                # 创建超几何分布对象 X，其中 N 是总体大小，m 是成功事件数，n 是样本中的抽取数量
                X = Hypergeometric('X', N, m, n)
                # 将 N, m, n 转换为符号化的对象
                N, m, n = map(sympify, (N, m, n))
                # 断言超几何分布 X 的概率密度函数所有值之和为 1
                assert sum(density(X).values()) == 1
                # 断言超几何分布 X 的期望值等于 n * m / N
                assert E(X) == n * m / N
                # 当 N > 1 时，断言超几何分布 X 的方差等于 n*(m/N)*(N - m)/N*(N - n)/(N - 1)
                if N > 1:
                    assert variance(X) == n*(m/N)*(N - m)/N*(N - n)/(N - 1)
                # 只有在 N > 2 且 0 < m < N 且 n < N 时，才测试偏度
                if N > 2 and 0 < m < N and n < N:
                    # 断言超几何分布 X 的偏度等于简化后的表达式
                    assert skewness(X) == simplify((N - 2*m)*sqrt(N - 1)*(N - 2*n)
                        / (sqrt(n*m*(N - m)*(N - n))*(N - 2)))

# 定义一个测试函数，用于测试超几何分布的符号计算
def test_hypergeometric_symbolic():
    # 定义符号变量 N, m, n
    N, m, n = symbols('N, m, n')
    # 创建超几何分布对象 H，其中 N 是总体大小，m 是成功事件数，n 是样本中的抽取数量
    H = Hypergeometric('H', N, m, n)
    # 获取超几何分布 H 的概率密度字典
    dens = density(H).dict
    # 获取超几何分布 H 大小为 N, m, n 的期望值
    expec = E(H > 2)
    # 断言超几何分布 H 的概率密度字典等于给定的超几何分布概率密度
    assert dens == Density(HypergeometricDistribution(N, m, n))
    # 断言将 N 替换为 5 后超几何分布 H 的概率密度字典等于给定的超几何分布概率密度
    assert dens.subs(N, 5).doit() == Density(HypergeometricDistribution(5, m, n))
    # 断言将 N 替换为 3，m 替换为 2，n 替换为 1 后超几何分布 H 的概率密度字典的键集合等于 {0, 1}
    assert set(dens.subs({N: 3, m: 2, n: 1}).doit().keys()) == {S.Zero, S.One}
    # 断言将 N 替换为 3，m 替换为 2，n 替换为 1 后超几何分布 H 的概率密度字典的值集合等于 {1/3, 2/3}
    assert set(dens.subs({N: 3, m: 2, n: 1}).doit().values()) == {Rational(1, 3), Rational(2, 3)}
    # 定义整数符号变量 k
    k = Dummy('k', integer=True)
    # 断言 expec 满足
    assert expec.dummy_eq(
        Sum(Piecewise((k*binomial(m, k)*binomial(N - m, -k + n)
        /binomial(N, n), k > 2), (0, True)), (k, 0, n)))

# 定义一个测试函数，用于测试 Rademacher 分布
def test_rademacher():
    # 创建 Rademacher 分布对象 X
    X = Rademacher('X')
    # 创建符号变量 t
    t = Symbol('t')

    # 断言 Rademacher 分布 X 的期望值为 0
    assert E(X) == 0
    # 断言 Rademacher 分布 X 的方差为 1
    assert variance(X) == 1
    # 断言 Rademacher 分布 X 的概率密度在 -1 处为 1/2
    assert density(X)[-1] == S.Half
    # 断言 Rademacher 分布 X 的概率密度在 1 处为 1/2
    assert density(X)[1] == S.Half
    # 断言 Rademacher 分布 X 的特征函数为 exp(I*t)/2 + exp(-I*t)/2
    assert characteristic_function(X)(t) == exp(I*t)/2 + exp(-I*t)/2
    # 断言 Rademacher 分布 X 的动差生成函数为 exp(t)/2 + exp(-t)/2
    assert moment_generating_function(X)(t) == exp(t) / 2 + exp(-t) / 2

# 定义一个测试函数，用于测试 Ideal Soliton 分布
def test_ideal_soliton():
    # 断言创建 Ideal Soliton 分布对象 'sol'，k 值为 -12 时引发 ValueError
    raises(ValueError, lambda : IdealSoliton('sol', -12))
    # 断言创建 Ideal Soliton 分布对象 'sol'，k 值为 13.2 时引发 ValueError
    raises(ValueError, lambda : IdealSoliton('sol', 13.2))
    # 断言创建 Ideal Soliton 分布对象 'sol'，k 值为 0 时引发 ValueError
    raises(ValueError, lambda : IdealSoliton('sol', 0))
    # 定义函数符号变量 f
    f = Function('f')
    # 断言 Ideal Soliton 分布 'sol' 的密度函数不支持传入函数 f
    raises(ValueError, lambda : density(IdealSoliton('sol', 10)).pmf(f))

    # 定义整数符号变量 k 和 x
    k = Symbol('k', integer=True, positive=True)
    x = Symbol('x', integer=True, positive=True)
    # 创建 Ideal Soliton 分布对象 sol，其中 k 是参数
    sol = IdealSoliton('sol', k)
    # 断言 Ideal Soliton 分布 sol 的密度函数低端为 1
    assert density(sol).low == S.One
    # 断言 Ideal Soliton 分布 sol 的密度函数高端为 k
    assert density(sol).high == k
    # 断言 Ideal Soliton 分布 sol 的密度函数字典等于 Ideal Soliton 分布 sol 的密度
    assert density(sol).dict == Density(density(sol))
    # 断言 Ideal Soliton 分布 sol 的概率质量函数在 x = 1 处为 1/k，在 k >= x 且 x > 1 时为 1/(x*(x - 1))
    assert density(sol).pmf(x) == Piecewise((1/k, Eq(x, 1)), (1/(x*(x - 1)), k >= x), (0, True))

    # 定义 k_vals 列表，包含多个整数值
    k_vals = [5, 20, 50, 100, 1000]
    for i in k_vals:
        # 断言 Ideal Soliton 分布 sol 在参数 k 替换为 i 后的期望值等于调和数 harmonic(i)
        assert E(sol.subs(k, i)) == harmonic(i) == moment(sol.subs(k, i), 1)
        # 断言 Ideal Soliton 分布 sol 在参数 k 替换为 i 后的方
# 定义一个测试函数，用于测试 RobustSoliton 类的各种情况下是否能正确触发 ValueError 异常
def test_robust_soliton():
    # 测试参数值为负数时是否触发异常
    raises(ValueError, lambda: RobustSoliton('robSol', -12, 0.1, 0.02))
    # 测试 delta 大于 1 时是否触发异常
    raises(ValueError, lambda: RobustSoliton('robSol', 13, 1.89, 0.1))
    # 测试 c 小于 0 时是否触发异常
    raises(ValueError, lambda: RobustSoliton('robSol', 15, 0.6, -2.31))
    # 创建一个 Function 对象 f
    f = Function('f')
    # 测试使用 RobustSoliton 分布的 pmf(f) 是否触发异常
    raises(ValueError, lambda: density(RobustSoliton('robSol', 15, 0.6, 0.1)).pmf(f))

    # 定义符号 k 为正整数
    k = Symbol('k', integer=True, positive=True)
    # 定义符号 delta 为正数
    delta = Symbol('delta', positive=True)
    # 定义符号 c 为正数
    c = Symbol('c', positive=True)
    # 创建 RobustSoliton 对象 robSol
    robSol = RobustSoliton('robSol', k, delta, c)
    # 断言 RobustSoliton 分布的 density.low 属性为 1
    assert density(robSol).low == 1
    # 断言 RobustSoliton 分布的 density.high 属性为 k
    assert density(robSol).high == k

    # 定义 k_vals、delta_vals、c_vals 三个列表
    k_vals = [10, 20, 50]
    delta_vals = [0.2, 0.4, 0.6]
    c_vals = [0.01, 0.03, 0.05]
    # 循环遍历 k_vals、delta_vals、c_vals 列表中的所有组合
    for x in k_vals:
        for y in delta_vals:
            for z in c_vals:
                # 断言使用给定参数 x、y、z 计算的期望值与一阶矩相等
                assert E(robSol.subs({k: x, delta: y, c: z})) == moment(robSol.subs({k: x, delta: y, c: z}), 1)
                # 断言使用给定参数 x、y、z 计算的方差与二阶中心矩相等
                assert variance(robSol.subs({k: x, delta: y, c: z})) == cmoment(robSol.subs({k: x, delta: y, c: z}), 2)
                # 断言使用给定参数 x、y、z 计算的偏度与三阶中心矩相等
                assert skewness(robSol.subs({k: x, delta: y, c: z})) == smoment(robSol.subs({k: x, delta: y, c: z}), 3)
                # 断言使用给定参数 x、y、z 计算的峰度与四阶中心矩相等
                assert kurtosis(robSol.subs({k: x, delta: y, c: z})) == smoment(robSol.subs({k: x, delta: y, c: z}), 4)

# 定义一个测试函数，测试 FiniteRV 类的各种功能
def test_FiniteRV():
    # 创建一个有限分布对象 F
    F = FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)}, check=True)
    # 创建符号 p，表示概率
    p = Symbol("p", positive=True)

    # 断言 F 的密度函数与给定的字典相等
    assert dict(density(F).items()) == {S.One: S.Half, S(2): Rational(1, 4), S(3): Rational(1, 4)}
    # 断言 P(F >= 2) 的概率为 S.Half
    assert P(F >= 2) == S.Half
    # 断言 quantile(F)(p) 的返回值在不同区间下的值符合预期
    assert quantile(F)(p) == Piecewise((nan, p > S.One), (S.One, p <= S.Half),\
        (S(2), p <= Rational(3, 4)),(S(3), True))

    # 断言 F 的 pspace 的域是一个逻辑或表达式，表示 F 可能取的值
    assert pspace(F).domain.as_boolean() == Or(
        *[Eq(F.symbol, i) for i in [1, 2, 3]])

    # 断言 F 的 pspace 的 set 属性为包含可能取值的有限集合
    assert F.pspace.domain.set == FiniteSet(1, 2, 3)
    # 测试创建时指定概率分布的值不合法是否会触发异常
    raises(ValueError, lambda: FiniteRV('F', {1: S.Half, 2: S.Half, 3: S.Half}, check=True))
    # 测试创建时指定概率分布值不合法是否会触发异常
    raises(ValueError, lambda: FiniteRV('F', {1: S.Half, 2: Rational(-1, 2), 3: S.One}, check=True))
    # 测试创建时指定概率分布值不合法是否会触发异常
    raises(ValueError, lambda: FiniteRV('F', {1: S.One, 2: Rational(3, 2), 3: S.Zero,\
        4: Rational(-1, 2), 5: Rational(-3, 4), 6: Rational(-1, 4)}, check=True))

    # 创建一个有限分布对象 X，指定一个不合法的概率分布，但因为 check=False，不会触发异常
    X = FiniteRV('X', {1: 1, 2: 2})
    # 断言 E(X) 的值为 5
    assert E(X) == 5
    # 断言 P(X <= 2) + P(X > 2) 不等于 1
    assert P(X <= 2) + P(X > 2) != 1

# 定义一个测试函数，测试 Bernoulli 分布的密度函数
def test_density_call():
    # 导入符号 p，表示 Bernoulli 分布的成功概率
    from sympy.abc import p
    # 创建一个 Bernoulli 分布对象 x
    x = Bernoulli('x', p)
    # 获取 x 的密度函数对象 d
    d = density(x)
    # 断言 d(0) 的值为 1 - p
    assert d(0) == 1 - p
    # 断言 d(S.Zero) 的值为 1 - p
    assert d(S.Zero) == 1 - p
    # 断言 d(5) 的值为 0
    assert d(5) == 0

    # 断言 0 在 d 的支持集合中
    assert 0 in d
    # 断言 5 不在 d 的支持集合中
    assert 5 not in d
    # 断言 d(S.Zero) 与 d[S.Zero] 相等
    assert d(S.Zero) == d[S.Zero]

# 定义一个测试函数，测试骰子分布的各种功能
def test_DieDistribution():
    # 导入符号 x，表示骰子的值
    from sympy.abc import x
    # 创建一个骰子分布对象 X，共有 6 个面
    X = DieDistribution(6)
    # 断言 X 在 S.Half 处的概率质量函数值为 S.Zero
    assert X.pmf(S.Half) is S.Zero
    # 断言 X 在 x=1 处的概率质量函数值为 1/6
    assert X.pmf(x).subs({x: 1}).doit() == Rational(1, 6)
    # 断言 X 在 x=7 处的概率质量函数值为 0
    assert X.pmf(x).subs({x: 7}).doit() == 0
    # 断言 X 在 x=-1 处的概率质量
    `
    # 期望抛出 ValueError 异常，调用 X 对象的 pmf 方法，传入 Matrix([0, 0]) 作为参数
    raises(ValueError, lambda: X.pmf(Matrix([0, 0])))
    # 期望抛出 ValueError 异常，调用 X 对象的 pmf 方法，传入 x**2 - 1 表达式作为参数
    raises(ValueError, lambda: X.pmf(x**2 - 1))
# 定义一个测试函数，用于测试有限概率空间的行为
def test_FinitePSpace():
    # 创建一个六面骰子 X
    X = Die('X', 6)
    # 根据 X 创建一个概率空间对象
    space = pspace(X)
    # 断言空间的密度等于一个六面骰子的分布
    assert space.density == DieDistribution(6)

# 定义一个测试函数，用于测试符号条件的行为
def test_symbolic_conditions():
    # 创建一个伯努利随机变量 B，概率为 1/4
    B = Bernoulli('B', Rational(1, 4))
    # 创建一个四面骰子 D
    D = Die('D', 4)
    # 创建符号变量 b 和 n
    b, n = symbols('b, n')
    # 创建符号条件 Y，表示 B 等于 b 的概率
    Y = P(Eq(B, b))
    # 创建符号条件 Z，表示 D 大于 n 的概率
    Z = E(D > n)
    # 断言条件 Y 的符号表达式
    assert Y == \
    Piecewise((Rational(1, 4), Eq(b, 1)), (0, True)) + \
    Piecewise((Rational(3, 4), Eq(b, 0)), (0, True))
    # 断言条件 Z 的符号表达式
    assert Z == \
    Piecewise((Rational(1, 4), n < 1), (0, True)) + Piecewise((S.Half, n < 2), (0, True)) + \
    Piecewise((Rational(3, 4), n < 3), (0, True)) + Piecewise((S.One, n < 4), (0, True))
```