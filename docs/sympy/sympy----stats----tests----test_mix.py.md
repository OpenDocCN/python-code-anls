# `D:\src\scipysrc\sympy\sympy\stats\tests\test_mix.py`

```
# 导入 Sympy 库中需要的模块和函数
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Ne)
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.stats import (Poisson, Beta, Exponential, P,
                        Multinomial, MultivariateBeta)
from sympy.stats.crv_types import Normal
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
from sympy.stats.joint_rv import MarginalDistribution
from sympy.stats.rv import pspace, density
from sympy.testing.pytest import ignore_warnings

# 定义一个测试函数，用于测试概率分布的密度函数计算
def test_density():
    # 定义符号 x 和 l，其中 l 是一个正数
    x = Symbol('x')
    l = Symbol('l', positive=True)
    # 用 Beta 分布定义一个泊松分布，X 是一个泊松随机变量
    rate = Beta(l, 2, 3)
    X = Poisson(x, rate)
    # 断言 X 的概率空间是一个 CompoundPSpace 对象
    assert isinstance(pspace(X), CompoundPSpace)
    # 断言在给定 Beta 分布率的条件下，X 的密度函数是一个泊松分布
    assert density(X, Eq(rate, rate.symbol)) == PoissonDistribution(l)
    
    # 定义两个正态分布 N1 和 N2
    N1 = Normal('N1', 0, 1)
    N2 = Normal('N2', N1, 2)
    # 断言 N2 在 x=0 处的密度函数值
    assert density(N2)(0).doit() == sqrt(10)/(10*sqrt(pi))
    # 断言在给定 N1=1 的条件下，N2 的密度函数
    assert simplify(density(N2, Eq(N1, 1))(x)) == \
        sqrt(2)*exp(-(x - 1)**2/8)/(4*sqrt(pi))
    # 断言简化后的 N2 的密度函数
    assert simplify(density(N2)(x)) == sqrt(10)*exp(-x**2/10)/(10*sqrt(pi))

# 定义一个测试函数，用于测试边缘分布的计算
def test_MarginalDistribution():
    # 定义几个正数符号：a1, p1, p2
    a1, p1, p2 = symbols('a1 p1 p2', positive=True)
    # 定义一个二项分布 C
    C = Multinomial('C', 2, p1, p2)
    # 用 C 定义一个多元贝塔分布 B
    B = MultivariateBeta('B', a1, C[0])
    # 计算 B 的边缘分布 MGR
    MGR = MarginalDistribution(B, (C[0],))
    # 定义 mgrc 变量，复杂表达式的乘积
    mgrc = Mul(Symbol('B'), Piecewise(ExprCondPair(Mul(Integer(2),
    Pow(Symbol('p1', positive=True), Indexed(IndexedBase(Symbol('C')),
    Integer(0))), Pow(Symbol('p2', positive=True),
    Indexed(IndexedBase(Symbol('C')), Integer(1))),
    Pow(factorial(Indexed(IndexedBase(Symbol('C')), Integer(0))), Integer(-1)),
    Pow(factorial(Indexed(IndexedBase(Symbol('C')), Integer(1))), Integer(-1))),
    Eq(Add(Indexed(IndexedBase(Symbol('C')), Integer(0)),
    Indexed(IndexedBase(Symbol('C')), Integer(1))), Integer(2))),
    ExprCondPair(Integer(0), True)), Pow(gamma(Symbol('a1', positive=True)),
    Integer(-1)), gamma(Add(Symbol('a1', positive=True),
    Indexed(IndexedBase(Symbol('C')), Integer(0)))),
    Pow(gamma(Indexed(IndexedBase(Symbol('C')), Integer(0))), Integer(-1)),
    Pow(Indexed(IndexedBase(Symbol('B')), Integer(0)),
    Add(Symbol('a1', positive=True), Integer(-1))),
    Pow(Indexed(IndexedBase(Symbol('B')), Integer(1)),


（注：因为剩余部分超出了单个注释块的长度限制，无法一次性完成，建议继续分批次注释。）
    # 创建一个符号 'C' 的 IndexedBase 对象，然后在其第一个元素上加上 -1，这是一个数学表达式
    Add(Indexed(IndexedBase(Symbol('C')), Integer(0)), Integer(-1))))
    # 断言调用 MGR 函数后返回的结果等于 mgrc，用于测试函数的正确性
    assert MGR(C) == mgrc
# 定义一个测试函数，用于测试复合分布的相关功能
def test_compound_distribution():
    # 创建一个参数为1的泊松分布随机变量 Y
    Y = Poisson('Y', 1)
    # 创建一个以 Y 为参数的泊松分布随机变量 Z
    Z = Poisson('Z', Y)
    # 断言 Z 的概率空间是 CompoundPSpace 类的实例
    assert isinstance(pspace(Z), CompoundPSpace)
    # 断言 Z 的概率空间的分布是 CompoundDistribution 类的实例
    assert isinstance(pspace(Z).distribution, CompoundDistribution)
    # 断言 Z 的概率密度函数在值为1时的计算结果
    assert Z.pspace.distribution.pdf(1).doit() == exp(-2)*exp(exp(-1))

# 定义一个测试函数，用于测试混合表达式的相关功能
def test_mix_expression():
    # 创建参数为1的泊松分布随机变量 Y 和参数为1的指数分布随机变量 E
    Y, E = Poisson('Y', 1), Exponential('E', 1)
    # 创建一个虚拟变量 k
    k = Dummy('k')
    # 定义表达式 expr1，是一个复杂的积分和求和表达式
    expr1 = Integral(Sum(exp(-1)*Integral(exp(-k)*DiracDelta(k - 2), (k, 0, oo)
    )/factorial(k), (k, 0, oo)), (k, -oo, 0))
    # 定义表达式 expr2，是另一个复杂的积分和求和表达式
    expr2 = Integral(Sum(exp(-1)*Integral(exp(-k)*DiracDelta(k - 2), (k, 0, oo)
    )/factorial(k), (k, 0, oo)), (k, 0, oo))
    # 断言两个随机变量 Y 和 E 的和等于1的概率为0
    assert P(Eq(Y + E, 1)) == 0
    # 断言两个随机变量 Y 和 E 的和不等于2的概率为1
    assert P(Ne(Y + E, 2)) == 1
    # 忽略用户警告后，断言 E + Y 小于2 的概率，重写为表达式 expr1
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert P(E + Y < 2, evaluate=False).rewrite(Integral).dummy_eq(expr1)
    # 忽略用户警告后，断言 E + Y 大于2 的概率，重写为表达式 expr2
        assert P(E + Y > 2, evaluate=False).rewrite(Integral).dummy_eq(expr2)
```