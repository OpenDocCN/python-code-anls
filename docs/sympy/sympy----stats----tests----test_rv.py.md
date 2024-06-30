# `D:\src\scipysrc\sympy\sympy\stats\tests\test_rv.py`

```
# 从 sympy.concrete.summations 模块导入 Sum 类
from sympy.concrete.summations import Sum
# 从 sympy.core.basic 模块导入 Basic 类
from sympy.core.basic import Basic
# 从 sympy.core.containers 模块导入 Tuple 类
from sympy.core.containers import Tuple
# 从 sympy.core.function 模块导入 Lambda 类
from sympy.core.function import Lambda
# 从 sympy.core.numbers 模块导入 Rational, nan, oo, pi 等类和常量
from sympy.core.numbers import (Rational, nan, oo, pi)
# 从 sympy.core.relational 模块导入 Eq 类
from sympy.core.relational import Eq
# 从 sympy.core.singleton 模块导入 S 单例对象
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Symbol, symbols 函数
from sympy.core.symbol import (Symbol, symbols)
# 从 sympy.functions.combinatorial.factorials 模块导入 FallingFactorial, binomial 函数
from sympy.functions.combinatorial.factorials import (FallingFactorial, binomial)
# 从 sympy.functions.elementary.exponential 模块导入 exp, log 函数
from sympy.functions.elementary.exponential import (exp, log)
# 从 sympy.functions.elementary.trigonometric 模块导入 cos, sin 函数
from sympy.functions.elementary.trigonometric import (cos, sin)
# 从 sympy.functions.special.delta_functions 模块导入 DiracDelta 函数
from sympy.functions.special.delta_functions import DiracDelta
# 从 sympy.integrals.integrals 模块导入 integrate 函数
from sympy.integrals.integrals import integrate
# 从 sympy.logic.boolalg 模块导入 And, Or 函数
from sympy.logic.boolalg import (And, Or)
# 从 sympy.matrices.dense 模块导入 Matrix 类
from sympy.matrices.dense import Matrix
# 从 sympy.sets.sets 模块导入 Interval 类
from sympy.sets.sets import Interval
# 从 sympy.tensor.indexed 模块导入 Indexed 类
from sympy.tensor.indexed import Indexed
# 从 sympy.stats 模块导入 Die, Normal, Exponential 等随机变量类和函数
from sympy.stats import (Die, Normal, Exponential, FiniteRV, P, E, H, variance,
        density, given, independent, dependent, where, pspace, GaussianUnitaryEnsemble,
        random_symbols, sample, Geometric, factorial_moment, Binomial, Hypergeometric,
        DiscreteUniform, Poisson, characteristic_function, moment_generating_function,
        BernoulliProcess, Variance, Expectation, Probability, Covariance, covariance, cmoment,
        moment, median)
# 从 sympy.stats.rv 模块导入各种随机变量相关类和函数
from sympy.stats.rv import (IndependentProductPSpace, rs_swap, Density, NamedArgsMixin,
        RandomSymbol, sample_iter, PSpace, is_random, RandomIndexedSymbol, RandomMatrixSymbol)
# 从 sympy.testing.pytest 模块导入测试相关的装饰器和函数
from sympy.testing.pytest import raises, skip, XFAIL, warns_deprecated_sympy
# 从 sympy.external 模块导入 import_module 函数
from sympy.external import import_module
# 从 sympy.core.numbers 模块导入 comp 函数
from sympy.core.numbers import comp
# 从 sympy.stats.frv_types 模块导入 BernoulliDistribution 类
from sympy.stats.frv_types import BernoulliDistribution
# 从 sympy.core.symbol 模块导入 Dummy 类
from sympy.core.symbol import Dummy
# 从 sympy.functions.elementary.piecewise 模块导入 Piecewise 类
from sympy.functions.elementary.piecewise import Piecewise

# 定义测试函数 test_where
def test_where():
    # 创建两个骰子随机变量 X 和 Y
    X, Y = Die('X'), Die('Y')
    # 创建一个正态分布随机变量 Z
    Z = Normal('Z', 0, 1)

    # 断言 Z^2 <= 1 的事件的结果集合是闭区间 [-1, 1]
    assert where(Z**2 <= 1).set == Interval(-1, 1)
    # 断言 Z^2 <= 1 的事件的布尔表达式等效于 [-1, 1] 关于 Z.symbol 的关系表达式
    assert where(Z**2 <= 1).as_boolean() == Interval(-1, 1).as_relational(Z.symbol)
    # 断言 X > Y 和 Y > 4 同时发生的事件的布尔表达式等效于 X.symbol == 6 和 Y.symbol == 5 的逻辑与
    assert where(And(X > Y, Y > 4)).as_boolean() == And(
        Eq(X.symbol, 6), Eq(Y.symbol, 5))

    # 断言 X < 3 的事件的结果集合长度为 2
    assert len(where(X < 3).set) == 2
    # 断言 1 在 X < 3 的事件的结果集合中
    assert 1 in where(X < 3).set

    # 重新定义 X 和 Y 为均值为 0，方差为 1 的正态分布随机变量
    X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    # 断言 X^2 <= 1 和 X >= 0 同时发生的事件的结果集合是闭区间 [0, 1]
    assert where(And(X**2 <= 1, X >= 0)).set == Interval(0, 1)
    # 创建给定条件下的随机变量 XX，其定义域是闭区间 [0, 1]
    XX = given(X, And(X**2 <= 1, X >= 0))
    # 断言 XX 的定义域是闭区间 [0, 1]
    assert XX.pspace.domain.set == Interval(0, 1)
    # 断言 XX 的定义域的布尔表达式等效于 0 <= X.symbol <= 1 的逻辑与
    assert XX.pspace.domain.as_boolean() == \
        And(0 <= X.symbol, X.symbol**2 <= 1, -oo < X.symbol, X.symbol < oo)

    # 使用 raises 断言捕获 TypeError 异常
    with raises(TypeError):
        XX = given(X, X + 3)


# 定义测试函数 test_random_symbols
def test_random_symbols():
    # 创建两个均值为 0，方差为 1 的正态分布随机变量 X 和 Y
    X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)

    # 断言表达式 2*X + 1 的随机符号集合是 {X}
    assert set(random_symbols(2*X + 1)) == {X}
    # 断言表达式 2*X + Y 的随机符号集合是 {X, Y}
    assert set(random_symbols(2*X + Y)) == {X, Y}
    # 断言表达式 2*X + Y.symbol 的随机符号集合是 {X}
    assert set(random_symbols(2*X + Y.symbol)) == {X}
    # 断言表达式 2 的随机符号集合是空集
    assert set(random_symbols(2)) == set()


# 定义测试函数 test_characteristic_function
def test_characteristic_function():
    # 从 sympy.core.numbers 模块导入 I 虚数单位
    from sympy.core.numbers import I
    # 创建均值为 0，方差为 1 的正态分布随机变量 X
    X = Normal('X',0,1)
    # 创建取值为 [1, 2, 7] 的离散均匀分布随机变量 Y
    Y = DiscreteUniform('Y', [1,2,7])
    # 创建均值为 2 的泊松分布随机变量 Z
    Z = Poisson('Z', 2)


这段代码包含了大量导入语句和三个测试函数的定义，每个函数的目的和用法在注释中有详细解释。
    # 定义符号变量 _t
    t = symbols('_t')
    # 定义函数 P，表示特征函数，其定义为 exp(-t**2/2)
    P = Lambda(t, exp(-t**2/2))
    # 定义函数 Q，表示特征函数，其定义为 exp(7*t*I)/3 + exp(2*t*I)/3 + exp(t*I)/3
    Q = Lambda(t, exp(7*t*I)/3 + exp(2*t*I)/3 + exp(t*I)/3)
    # 定义函数 R，表示特征函数，其定义为 exp(2 * exp(t*I) - 2)
    R = Lambda(t, exp(2 * exp(t*I) - 2))
    
    # 断言语句，用于验证特征函数 characteristic_function(X), characteristic_function(Y), characteristic_function(Z)
    # 分别与 P, Q, R 是否近似相等
    assert characteristic_function(X).dummy_eq(P)
    assert characteristic_function(Y).dummy_eq(Q)
    assert characteristic_function(Z).dummy_eq(R)
# 定义一个测试函数，用于测试生成函数的正确性
def test_moment_generating_function():

    # 定义正态分布随机变量X，均值为0，标准差为1
    X = Normal('X', 0, 1)
    # 定义离散均匀分布随机变量Y，取值为1、2、7
    Y = DiscreteUniform('Y', [1, 2, 7])
    # 定义泊松分布随机变量Z，参数为2
    Z = Poisson('Z', 2)
    # 定义符号变量t
    t = symbols('_t')
    # 定义函数P，Lambda表示匿名函数，表示exp(t**2/2)
    P = Lambda(t, exp(t**2/2))
    # 定义函数Q，Lambda表示匿名函数，表示(exp(7*t)/3 + exp(2*t)/3 + exp(t)/3)
    Q = Lambda(t, (exp(7*t)/3 + exp(2*t)/3 + exp(t)/3))
    # 定义函数R，Lambda表示匿名函数，表示exp(2 * exp(t) - 2)
    R = Lambda(t, exp(2 * exp(t) - 2))

    # 断言：moment_generating_function(X)的返回值与P相等
    assert moment_generating_function(X).dummy_eq(P)
    # 断言：moment_generating_function(Y)的返回值与Q相等
    assert moment_generating_function(Y).dummy_eq(Q)
    # 断言：moment_generating_function(Z)的返回值与R相等
    assert moment_generating_function(Z).dummy_eq(R)

# 定义一个测试函数，用于测试样本生成迭代器的正确性
def test_sample_iter():

    # 定义正态分布随机变量X，均值为0，标准差为1
    X = Normal('X', 0, 1)
    # 定义离散均匀分布随机变量Y，取值为1、2、7
    Y = DiscreteUniform('Y', [1, 2, 7])
    # 定义泊松分布随机变量Z，参数为2
    Z = Poisson('Z', 2)

    # 导入模块scipy
    scipy = import_module('scipy')
    # 如果没有安装scipy，跳过测试
    if not scipy:
        skip('Scipy is not installed. Abort tests')

    # 表达式expr为X的平方加3
    expr = X**2 + 3
    # 调用sample_iter生成expr的样本迭代器
    iterator = sample_iter(expr)

    # 表达式expr2为Y的平方加5乘以Y再加4
    expr2 = Y**2 + 5*Y + 4
    # 调用sample_iter生成expr2的样本迭代器
    iterator2 = sample_iter(expr2)

    # 表达式expr3为Z的立方加4
    expr3 = Z**3 + 4
    # 调用sample_iter生成expr3的样本迭代器
    iterator3 = sample_iter(expr3)

    # 定义函数is_iterator，用于判断对象是否为迭代器
    def is_iterator(obj):
        if (
            hasattr(obj, '__iter__') and
            (hasattr(obj, 'next') or
            hasattr(obj, '__next__')) and
            callable(obj.__iter__) and
            obj.__iter__() is obj
           ):
            return True
        else:
            return False

    # 断言：iterator是一个迭代器
    assert is_iterator(iterator)
    # 断言：iterator2是一个迭代器
    assert is_iterator(iterator2)
    # 断言：iterator3是一个迭代器
    assert is_iterator(iterator3)

# 定义一个测试函数，用于测试随机变量空间的处理函数pspace的正确性
def test_pspace():
    # 定义正态分布随机变量X，均值为0，标准差为1
    X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    # 定义符号变量x
    x = Symbol('x')

    # 断言：对于非随机变量，调用pspace会引发ValueError异常
    raises(ValueError, lambda: pspace(5 + 3))
    # 断言：对于不完整的布尔表达式，调用pspace会引发ValueError异常
    raises(ValueError, lambda: pspace(x < 1))
    # 断言：pspace(X)返回X的随机变量空间对象
    assert pspace(X) == X.pspace
    # 断言：pspace(2*X + 1)返回2*X + 1的随机变量空间对象
    assert pspace(2*X + 1) == X.pspace
    # 断言：pspace(2*X + Y)返回独立随机变量空间对象IndependentProductPSpace(Y.pspace, X.pspace)
    assert pspace(2*X + Y) == IndependentProductPSpace(Y.pspace, X.pspace)

# 定义一个测试函数，用于测试随机变量替换函数rs_swap的正确性
def test_rs_swap():
    # 定义正态分布随机变量X，均值为0，标准差为1
    X = Normal('x', 0, 1)
    # 定义指数分布随机变量Y，参数为1
    Y = Exponential('y', 1)

    # 定义均值为0，标准差为2的正态分布随机变量XX
    XX = Normal('x', 0, 2)
    # 定义均值为0，标准差为3的正态分布随机变量YY
    YY = Normal('y', 0, 3)

    # 表达式expr为2*X + Y
    expr = 2*X + Y
    # 断言：替换(X, Y)为(YY, XX)后，expr的结果为2*XX + YY
    assert expr.subs(rs_swap((X, Y), (YY, XX))) == 2*XX + YY

# 定义一个测试函数，用于测试随机变量符号对象RandomSymbol的正确性
def test_RandomSymbol():

    # 定义均值为0，标准差为1的正态分布随机变量X
    X = Normal('x', 0, 1)
    # 定义均值为0，标准差为2的正态分布随机变量Y
    Y = Normal('x', 0, 2)
    # 断言：X的符号与Y的符号相同
    assert X.symbol == Y.symbol
    # 断言：X不等于Y
    assert X != Y

    # 断言：X的名称与其符号的名称相同
    assert X.name == X.symbol.name

    # 定义均值为0，标准差为1的正态分布随机变量X，使用保留字lambda作为变量名
    X = Normal('lambda', 0, 1) # make sure we can use protected terms
    # 定义均值为0，标准差为1的正态分布随机变量X，使用SymPy保留字Lambda作为变量名
    X = Normal('Lambda', 0, 1) # make sure we can use SymPy terms

# 定义一个测试函数，用于测试随机变量的微分操作
def test_RandomSymbol_diff():
    # 定义均值为0，标准差为1的正态分布随机变量X
    X = Normal('x', 0, 1)
    # 断言：对2*X进行X的微分操作应成功
    assert (2*X).diff(X)

# 定义一个测试函数，用于测试随机变量符号对象RandomSymbol的空间属性pspace
def test_random_symbol_no_pspace():
    # 定义符号变量x的随机变量对象x
    x = RandomSymbol(Symbol('x'))
    # 断言：x的随机变量空间为PSpace()
    assert x.pspace == PSpace()

# 定义
    `
    # 定义一个名为 z 的符号变量，其被指定为整数类型
    z = Symbol('z', integer=True)
    
    # 导入 scipy 模块，并检查其是否成功导入
    scipy = import_module('scipy')
    if not scipy:
        # 如果导入失败，则跳过后续测试，并输出相应提示信息
        skip('Scipy is not installed. Abort tests')
    
    # 断言从样本 X 中取值属于集合 [1, 2, 3, 4, 5, 6]
    assert sample(X) in [1, 2, 3, 4, 5, 6]
    
    # 断言从样本 X + Y 中取值为浮点数类型
    assert isinstance(sample(X + Y), float)
    
    # 断言使用 P 函数计算 X + Y > 0 且 Y < 0 的概率结果为数值
    assert P(X + Y > 0, Y < 0, numsamples=10).is_number
    
    # 断言使用 E 函数计算 E(X + Y) 的期望值结果为数值
    assert E(X + Y, numsamples=10).is_number
    
    # 断言使用 E 函数计算 E(X**2 + Y) 的期望值结果为数值
    assert E(X**2 + Y, numsamples=10).is_number
    
    # 断言使用 E 函数计算 E((X + Y)**2) 的期望值结果为数值
    assert E((X + Y)**2, numsamples=10).is_number
    
    # 断言使用 variance 函数计算 X + Y 的方差结果为数值
    assert variance(X + Y, numsamples=10).is_number
    
    # 断言调用 P 函数时传入 Y > z 的参数会引发 TypeError 异常
    raises(TypeError, lambda: P(Y > z, numsamples=5))
    
    # 断言调用 P 函数计算 sin(Y) <= 1 的概率结果为 1.0
    assert P(sin(Y) <= 1, numsamples=10) == 1.0
    
    # 断言调用 P 函数同时计算 sin(Y) <= 1 和 cos(Y) < 1 的概率结果为 1.0
    assert P(sin(Y) <= 1, cos(Y) < 1, numsamples=10) == 1.0
    
    # 断言使用 density 函数生成的样本值全部在范围 [1, 7) 内
    assert all(i in range(1, 7) for i in density(X, numsamples=10))
    
    # 断言使用 density 函数生成的样本值全部在范围 [4, 7) 内，且满足条件 X > 3
    assert all(i in range(4, 7) for i in density(X, X > 3, numsamples=10))
    
    # 导入 numpy 模块，并检查其是否成功导入
    numpy = import_module('numpy')
    if not numpy:
        # 如果导入失败，则跳过后续测试，并输出相应提示信息
        skip('Numpy is not installed. Abort tests')
    
    # 断言从样本 X 中取值的类型为 numpy.int32 或 numpy.int64
    assert isinstance(sample(X), (numpy.int32, numpy.int64))
    
    # 断言从样本 Y 中取值的类型为 numpy.float64
    assert isinstance(sample(Y), numpy.float64)
    
    # 断言从样本 X 中取值大小为 2 的数组类型为 numpy.ndarray
    assert isinstance(sample(X, size=2), numpy.ndarray)
    
    # 使用 warns_deprecated_sympy 上下文管理器捕获对 sample 函数的过时警告
    with warns_deprecated_sympy():
        sample(X, numsamples=2)
@XFAIL
def test_samplingE():
    # 导入 import_module 函数
    scipy = import_module('scipy')
    # 如果未成功导入 scipy 模块，则跳过测试
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    # 定义正态分布随机变量 Y
    Y = Normal('Y', 0, 1)
    # 定义整数符号变量 z
    z = Symbol('z', integer=True)
    # 断言：计算 E(Sum(1/z**Y, (z, 1, oo))) 的期望值，并检查结果是否为数值
    assert E(Sum(1/z**Y, (z, 1, oo)), Y > 2, numsamples=3).is_number


def test_given():
    # 定义正态分布随机变量 X 和 Y
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 1)
    # 给定条件 X 为真
    A = given(X, True)
    # 给定条件 X 和 Y > 2
    B = given(X, Y > 2)

    # 断言：X、A、B 均相等
    assert X == A == B


def test_factorial_moment():
    # 定义泊松分布随机变量 X，二项分布随机变量 Y 和超几何分布随机变量 Z
    X = Poisson('X', 2)
    Y = Binomial('Y', 2, S.Half)
    Z = Hypergeometric('Z', 4, 2, 2)
    # 断言：计算 X、Y、Z 的二阶阶乘矩，并验证结果
    assert factorial_moment(X, 2) == 4
    assert factorial_moment(Y, 2) == S.Half
    assert factorial_moment(Z, 2) == Rational(1, 3)

    # 定义符号变量 x, y, z, l
    x, y, z, l = symbols('x y z l')
    # 重新定义二项分布随机变量 Y 和超几何分布随机变量 Z
    Y = Binomial('Y', 2, y)
    Z = Hypergeometric('Z', 10, 2, 3)
    # 断言：计算 Y、Z 的 l 阶阶乘矩，并验证结果
    assert factorial_moment(Y, l) == y**2*FallingFactorial(
        2, l) + 2*y*(1 - y)*FallingFactorial(1, l) + (1 - y)**2*\
            FallingFactorial(0, l)
    assert factorial_moment(Z, l) == 7*FallingFactorial(0, l)/\
        15 + 7*FallingFactorial(1, l)/15 + FallingFactorial(2, l)/15


def test_dependence():
    # 定义骰子随机变量 X 和 Y
    X, Y = Die('X'), Die('Y')
    # 断言：X 和 2*Y 为独立的
    assert independent(X, 2*Y)
    # 断言：X 和 2*Y 不相关
    assert not dependent(X, 2*Y)

    # 重新定义正态分布随机变量 X 和 Y
    X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    # 断言：X 和 Y 独立
    assert independent(X, Y)
    # 断言：X 和 2*X 相关
    assert dependent(X, 2*X)

    # 创建一个依赖关系
    XX, YY = given(Tuple(X, Y), Eq(X + Y, 3))
    # 断言：XX 和 YY 相关
    assert dependent(XX, YY)


def test_dependent_finite():
    # 定义骰子随机变量 X 和 Y
    X, Y = Die('X'), Die('Y')
    # 依赖测试需要符号条件，当前版本无法处理有限随机变量
    # 断言：X 和 Y+X 相关
    assert dependent(X, Y + X)

    # 创建一个依赖关系
    XX, YY = given(Tuple(X, Y), X + Y > 5)
    # 断言：XX 和 YY 相关
    assert dependent(XX, YY)


def test_normality():
    # 定义正态分布随机变量 X 和 Y
    X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    # 定义实数符号变量 x 和 z
    x = Symbol('x', real=True)
    z = Symbol('z', real=True)
    # 计算密度函数 dens，并验证其积分结果为1
    dens = density(X - Y, Eq(X + Y, z))

    assert integrate(dens(x), (x, -oo, oo)) == 1


def test_Density():
    # 定义六面骰子随机变量 X
    X = Die('X', 6)
    # 计算 X 的密度函数，并验证结果
    d = Density(X)
    assert d.doit() == density(X)

def test_NamedArgsMixin():
    # 定义 Foo 类，继承自 Basic 和 NamedArgsMixin
    class Foo(Basic, NamedArgsMixin):
        _argnames = 'foo', 'bar'

    # 创建 Foo 类的实例 a
    a = Foo(S(1), S(2))

    # 断言：验证实例 a 的属性 foo 和 bar 分别为 1 和 2
    assert a.foo == 1
    assert a.bar == 2

    # 断言：验证属性 baz 不存在于实例 a 中
    raises(AttributeError, lambda: a.baz)

    # 定义 Bar 类，继承自 Basic 和 NamedArgsMixin，但未指定 _argnames
    class Bar(Basic, NamedArgsMixin):
        pass

    # 断言：验证 Bar 类的实例化会引发 AttributeError
    raises(AttributeError, lambda: Bar(S(1), S(2)).foo)

def test_density_constant():
    # 断言：验证常数 3 的密度函数在参数 2 和 3 处的取值
    assert density(3)(2) == 0
    assert density(3)(3) == DiracDelta(0)

def test_cmoment_constant():
    # 断言：验证常数 3 的方差和 l 阶中心矩
    assert variance(3) == 0
    assert cmoment(3, 3) == 0
    assert cmoment(3, 4) == 0
    x = Symbol('x')
    # 断言：验证符号变量 x 的方差和 l 阶中心矩
    assert variance(x) == 0
    assert cmoment(x, 15) == 0
    assert cmoment(x, 0) == 1

def test_moment_constant():
    # 断言：验证常数 3 的矩
    assert moment(3, 0) == 1
    assert moment(3, 1) == 3
    assert moment(3, 2) == 9
    x = Symbol('x')
    # 断言：验证符号变量 x 的二阶矩
    assert moment(x, 2) == x**2

def test_median_constant():
    # 断言：验证常数 3 的中位数
    assert median(3) == 3
    x = Symbol('x')
    # 断言：验证符号变量 x 的中位数
    assert median(x) == x

def test_real():
    # 定义正态分布随机变量 x
    x = Normal('x', 0, 1)
    # 断言：验证 x 是否为实数
    assert x.is_real
# 定义一个测试函数，用于解决问题编号 10052
def test_issue_10052():
    # 创建一个指数分布随机变量 X，参数为 3
    X = Exponential('X', 3)
    # 断言 P(X < oo) 的概率为 1
    assert P(X < oo) == 1
    # 断言 P(X > oo) 的概率为 0
    assert P(X > oo) == 0
    # 断言 P(X < 2, X > oo) 的概率为 0
    assert P(X < 2, X > oo) == 0
    # 断言 P(X < oo, X > oo) 的概率为 0
    assert P(X < oo, X > oo) == 0
    # 断言 P(X < oo, X > 2) 的概率为 1
    assert P(X < oo, X > 2) == 1
    # 断言 P(X < 3, X == 2) 的概率为 0
    assert P(X < 3, X == 2) == 0
    # 使用 lambda 函数检查当输入为 1 时，是否会引发 ValueError 异常
    raises(ValueError, lambda: P(1))
    # 使用 lambda 函数检查当输入为 (X < 1, 2) 时，是否会引发 ValueError 异常
    raises(ValueError, lambda: P(X < 1, 2))

# 定义一个测试函数，用于解决问题编号 11934
def test_issue_11934():
    # 创建一个有限离散随机变量 X，密度分布为 {0: 0.5, 1: 0.5}
    density = {0: .5, 1: .5}
    X = FiniteRV('X', density)
    # 断言 E(X) 的期望为 0.5
    assert E(X) == 0.5
    # 断言 P(X >= 2) 的概率为 0
    assert P(X >= 2) == 0

# 定义一个测试函数，用于解决问题编号 8129
def test_issue_8129():
    # 创建一个指数分布随机变量 X，参数为 4
    X = Exponential('X', 4)
    # 断言 P(X >= X) 的概率为 1
    assert P(X >= X) == 1
    # 断言 P(X > X) 的概率为 0
    assert P(X > X) == 0
    # 断言 P(X > X+1) 的概率为 0
    assert P(X > X+1) == 0

# 定义一个测试函数，用于解决问题编号 12237
def test_issue_12237():
    # 创建两个正态分布随机变量 X 和 Y，均值为 0，标准差为 1
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 1)
    # 计算概率 P(X > 0, X) 赋值给变量 U
    U = P(X > 0, X)
    # 计算概率 P(Y < 0, X) 赋值给变量 V
    V = P(Y < 0, X)
    # 计算概率 P(X + Y > 0, X) 赋值给变量 W
    W = P(X + Y > 0, X)
    # 断言 W 的值等于 P(X + Y > 0, X)
    assert W == P(X + Y > 0, X)
    # 断言 U 的值等于 BernoulliDistribution(S.Half, S.Zero, S.One)
    assert U == BernoulliDistribution(S.Half, S.Zero, S.One)
    # 断言 V 的值等于 S.Half
    assert V == S.Half

# 定义一个测试函数，用于解决问题编号 12283
def test_issue_12283():
    # 创建一个正态分布随机变量 X，均值为 0，标准差为 1
    X = Normal('X', 0, 1)
    # 创建一个正态分布随机变量 Y，均值为 0，标准差为 1
    Y = Normal('Y', 0, 1)
    # 使用符号 a 和 b
    a, b = symbols('a, b')
    # 创建一个高斯单位集合随机过程 G，维度为 2
    G = GaussianUnitaryEnsemble('U', 2)
    # 创建一个伯努利过程随机变量 B，概率为 0.9
    B = BernoulliProcess('B', 0.9)
    # 断言 is_random(a) 返回 False
    assert not is_random(a)
    # 断言 is_random(a + b) 返回 False
    assert not is_random(a + b)
    # 断言 is_random(a * b) 返回 False
    assert not is_random(a * b)
    # 断言 is_random(Matrix([a**2, b**2])) 返回 False
    assert not is_random(Matrix([a**2, b**2]))
    # 断言 is_random(X) 返回 True
    assert is_random(X)
    # 断言 is_random(X**2 + Y) 返回 True
    assert is_random(X**2 + Y)
    # 断言 is_random(Y + b**2) 返回 True
    assert is_random(Y + b**2)
    # 断言 is_random(Y > 5) 返回 True
    assert is_random(Y > 5)
    # 断言 is_random(B[3] < 1) 返回 True
    assert is_random(B[3] < 1)
    # 断言 is_random(G) 返回 True
    assert is_random(G)
    # 断言 is_random(X * Y * B[1]) 返回 True
    assert is_random(X * Y * B[1])
    # 断言 is_random(Matrix([[X, B[2]], [G, Y]])) 返回 True
    assert is_random(Matrix([[X, B[2]], [G, Y]]))
    # 断言 is_random(Eq(X, 4)) 返回 True
    assert is_random(Eq(X, 4))

# 定义一个测试函数，用于解决问题编号 6810
def test_issue_6810():
    # 创建一个六面骰子随机变量 X
    X = Die('X', 6)
    # 创建一个正态分布随机变量 Y，均值为 0，标准差为 1
    Y = Normal('Y', 0, 1)
    # 断言 P(Eq(X, 2)) 的概率为 1/6
    assert P(Eq(X, 2)) == S(1)/6
    # 断言 P(Eq(Y, 0)) 的概率为 0
    assert P(Eq(Y, 0)) == 0
    # 断言 P(Or(X > 2, X < 3)) 的概率为 1
    assert P(Or(X > 2, X < 3)) == 1
    # 断言 P(And(X > 3, X > 2)) 的概率为 1/2
    assert P(And(X > 3, X > 2)) == S(1)/2

# 定义一个测试函数，用于解决问题编号 20286
def test_issue_20286():
    # 创建符号 n 和 p
    n, p = symbols('n p')
    # 创建一个二项分布随机变量 B，参数为 n 和 p
    B = Binomial('B', n, p)
    # 创建一个虚拟变量 k，限定为整数
    k = Dummy('k', integer=True)
    # 创建一个复杂的求和表达式 eq
    eq = Sum(
        Piecewise(
            (
                -p**k*(1 - p)**(-k + n)*log(p**k*(1 - p)**(-k + n)*binomial(n, k))*binomial(n, k),
                (k >= 0) & (k <= n)
            ),
            (nan, True)
        ),
        (k, 0, n)
    )
    # 断言 eq 与 H(B) 的虚拟等式
    assert eq.dummy_eq(H(B))
```