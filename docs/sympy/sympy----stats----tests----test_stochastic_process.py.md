# `D:\src\scipysrc\sympy\sympy\stats\tests\test_stochastic_process.py`

```
from sympy.concrete.summations import Sum  # 导入求和函数
from sympy.core.containers import Tuple  # 导入元组容器
from sympy.core.function import Lambda  # 导入 Lambda 函数
from sympy.core.numbers import (Float, Rational, oo, pi)  # 导入数值类型
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)  # 导入关系运算符
from sympy.core.singleton import S  # 导入符号常量
from sympy.core.symbol import (Symbol, symbols)  # 导入符号和符号集合
from sympy.functions.combinatorial.factorials import factorial  # 导入阶乘函数
from sympy.functions.elementary.exponential import exp  # 导入指数函数
from sympy.functions.elementary.integers import ceiling  # 导入天花板函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.special.error_functions import erf  # 导入误差函数
from sympy.functions.special.gamma_functions import (gamma, lowergamma)  # 导入伽马函数
from sympy.logic.boolalg import (And, Not)  # 导入逻辑运算符
from sympy.matrices.dense import Matrix  # 导入密集矩阵
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵符号
from sympy.matrices.immutable import ImmutableMatrix  # 导入不可变矩阵
from sympy.sets.contains import Contains  # 导入包含关系类
from sympy.sets.fancysets import Range  # 导入范围类集合
from sympy.sets.sets import (FiniteSet, Interval)  # 导入有限集合和区间
from sympy.stats import (DiscreteMarkovChain, P, TransitionMatrixOf, E,  # 导入概率统计相关函数和对象
                         StochasticStateSpaceOf, variance, ContinuousMarkovChain,
                         BernoulliProcess, PoissonProcess, WienerProcess,
                         GammaProcess, sample_stochastic_process)
from sympy.stats.joint_rv import JointDistribution  # 导入联合分布对象
from sympy.stats.joint_rv_types import JointDistributionHandmade  # 导入手工定义的联合分布类型
from sympy.stats.rv import RandomIndexedSymbol  # 导入随机变量符号
from sympy.stats.symbolic_probability import Probability, Expectation  # 导入概率和期望计算相关对象
from sympy.testing.pytest import (raises, skip, ignore_warnings,  # 导入测试相关函数和装饰器
                                  warns_deprecated_sympy)
from sympy.external import import_module  # 导入外部模块导入函数
from sympy.stats.frv_types import BernoulliDistribution  # 导入伯努利分布类型
from sympy.stats.drv_types import PoissonDistribution  # 导入泊松分布类型
from sympy.stats.crv_types import NormalDistribution, GammaDistribution  # 导入正态分布和伽马分布类型
from sympy.core.symbol import Str  # 导入字符串符号类型

# 定义测试函数，测试离散马尔可夫链的相关功能
def test_DiscreteMarkovChain():

    # 创建一个只有名称的离散马尔可夫链对象
    X = DiscreteMarkovChain("X")
    # 断言状态空间类型为范围集合
    assert isinstance(X.state_space, Range)
    # 断言索引集合为非负整数集合
    assert X.index_set == S.Naturals0
    # 断言转移概率矩阵为矩阵符号
    assert isinstance(X.transition_probabilities, MatrixSymbol)
    # 定义一个正整数符号 t
    t = symbols('t', positive=True, integer=True)
    # 断言 X[t] 是一个随机索引符号
    assert isinstance(X[t], RandomIndexedSymbol)
    # 断言 E(X[0]) 等于期望值对象 Expectation(X[0])
    assert E(X[0]) == Expectation(X[0])
    # 测试类型错误异常处理，传入整数 1，应抛出 TypeError
    raises(TypeError, lambda: DiscreteMarkovChain(1))
    # 测试未实现功能异常处理，调用 X(t)，应抛出 NotImplementedError
    raises(NotImplementedError, lambda: X(t))
    # 测试未实现功能异常处理，调用 communication_classes() 方法，应抛出 NotImplementedError
    raises(NotImplementedError, lambda: X.communication_classes())
    # 测试未实现功能异常处理，调用 canonical_form() 方法，应抛出 NotImplementedError
    raises(NotImplementedError, lambda: X.canonical_form())
    # 测试未实现功能异常处理，调用 decompose() 方法，应抛出 NotImplementedError
    raises(NotImplementedError, lambda: X.decompose())

    # 定义一个整数符号 nz
    nz = Symbol('n', integer=True)
    # 定义一个 nz x nz 的矩阵符号 TZ
    TZ = MatrixSymbol('M', nz, nz)
    # 定义一个 nz 范围集合 SZ
    SZ = Range(nz)
    # 创建一个具有名称、状态空间和转移概率矩阵的离散马尔可夫链对象 YZ
    YZ = DiscreteMarkovChain('Y', SZ, TZ)
    # 断言概率 P(YZ[2] == 1, YZ[1] == 0) 等于转移概率矩阵 TZ 的元素 TZ[0, 1]
    assert P(Eq(YZ[2], 1), Eq(YZ[1], 0)) == TZ[0, 1]

    # 测试值错误异常处理，调用 sample_stochastic_process(t)，应抛出 ValueError
    raises(ValueError, lambda: sample_stochastic_process(t))
    # 测试值错误异常处理，调用 next(sample_stochastic_process(X))，应抛出 ValueError
    raises(ValueError, lambda: next(sample_stochastic_process(X)))
    # 通过传入名称和状态空间创建离散马尔可夫链对象
    # pass name and state_space
    # 定义符号变量，分别表示状态符号、雨天、多云和晴天
    sym, rainy, cloudy, sunny = symbols('a Rainy Cloudy Sunny', real=True)
    # 定义状态空间列表，包含不同类型的状态表示方式
    state_spaces = [(1, 2, 3), [Str('Hello'), sym, DiscreteMarkovChain("Y", (1,2,3))],
                    Tuple(S(1), exp(sym), Str('World'), sympify=False), Range(-1, 5, 2),
                    [rainy, cloudy, sunny]]
    # 创建离散马尔可夫链对象列表，每个对象使用对应的状态空间初始化
    chains = [DiscreteMarkovChain("Y", state_space) for state_space in state_spaces]

    # 遍历马尔可夫链对象列表
    for i, Y in enumerate(chains):
        # 断言：转移概率矩阵类型为 MatrixSymbol
        assert isinstance(Y.transition_probabilities, MatrixSymbol)
        # 断言：状态空间与预期的列表或有限集合匹配
        assert Y.state_space == state_spaces[i] or Y.state_space == FiniteSet(*state_spaces[i])
        # 断言：状态数量为3
        assert Y.number_of_states == 3

        # 忽略用户警告的上下文
        with ignore_warnings(UserWarning):  # TODO: 一旦警告消除，恢复测试
            # 断言：条件概率的计算与期望值的创建
            assert P(Eq(Y[2], 1), Eq(Y[0], 2), evaluate=False) == Probability(Eq(Y[2], 1), Eq(Y[0], 2))
        # 断言：状态的期望值
        assert E(Y[0]) == Expectation(Y[0])

        # 断言：引发值错误，样本随机过程不可用
        raises(ValueError, lambda: next(sample_stochastic_process(Y)))

    # 断言：引发类型错误，非法的马尔可夫链参数
    raises(TypeError, lambda: DiscreteMarkovChain("Y", {1: 1}))
    # 创建马尔可夫链对象 Y，状态空间为指定范围
    Y = DiscreteMarkovChain("Y", Range(1, t, 2))
    # 断言：状态数量为上限 (t-1)/2 的天花板值
    assert Y.number_of_states == ceiling((t-1)/2)

    # 创建马尔可夫链对象列表，传递名称和转移概率矩阵
    chains = [DiscreteMarkovChain("Y", trans_probs=Matrix([[]])),
              DiscreteMarkovChain("Y", trans_probs=Matrix([[0, 1], [1, 0]])),
              DiscreteMarkovChain("Y", trans_probs=Matrix([[pi, 1-pi], [sym, 1-sym]]))]
    # 遍历马尔可夫链对象列表
    for Z in chains:
        # 断言：状态数量等于转移概率矩阵的行数
        assert Z.number_of_states == Z.transition_probabilities.shape[0]
        # 断言：转移概率矩阵类型为 ImmutableMatrix
        assert isinstance(Z.transition_probabilities, ImmutableMatrix)

    # 创建转移概率矩阵 T 和矩阵符号 TS
    T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    TS = MatrixSymbol('T', 3, 3)
    # 创建马尔可夫链对象 Y，指定名称、状态空间和转移概率矩阵
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    # 创建马尔可夫链对象 YS，指定名称、状态空间符号和转移概率矩阵符号
    YS = DiscreteMarkovChain("Y", ['One', 'Two', 3], TS)
    # 断言：联合分布计算结果等于联合分布对象
    assert Y.joint_distribution(1, Y[2], 3) == JointDistribution(Y[1], Y[2], Y[3])
    # 引发值错误，无法计算指定的联合分布
    raises(ValueError, lambda: Y.joint_distribution(Y[1].symbol, Y[2].symbol))
    # 断言：概率计算结果与预期值接近
    assert P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2) == Float(0.36, 2)
    # 断言：复杂的概率计算结果与预期表达式相等
    assert (P(Eq(YS[3], 2), Eq(YS[1], 1)) -
            (TS[0, 2]*TS[1, 0] + TS[1, 1]*TS[1, 2] + TS[1, 2]*TS[2, 2])).simplify() == 0
    # 断言：条件概率计算结果与预期概率对象相等
    assert P(Eq(YS[1], 1), Eq(YS[2], 2)) == Probability(Eq(YS[1], 1))
    # 断言：复杂条件概率计算结果与转移概率矩阵中对应项相等
    assert P(Eq(YS[3], 3), Eq(YS[1], 1)) == TS[0, 2]*TS[1, 0] + TS[1, 1]*TS[1, 2] + TS[1, 2]*TS[2, 2]
    # 创建转移概率矩阵 TO
    TO = Matrix([[0.25, 0.75, 0],[0, 0.25, 0.75],[0.75, 0, 0.25]])
    # 断言：带有转移矩阵条件的概率计算结果与预期值接近
    assert P(Eq(Y[3], 2), Eq(Y[1], 1) & TransitionMatrixOf(Y, TO)).round(3) == Float(0.375, 3)
    # 忽略用户警告的上下文
    with ignore_warnings(UserWarning): ### TODO: 一旦警告消除，恢复测试
        # 断言：条件期望值计算结果等于期望值对象
        assert E(Y[3], evaluate=False) == Expectation(Y[3])
        # 断言：条件期望值计算结果与预期值接近
        assert E(Y[3], Eq(Y[2], 1)).round(2) == Float(1.1, 3)
    # 创建矩阵符号 TSO
    TSO = MatrixSymbol('T', 4, 4)
    # 引发值错误，无法使用不匹配的转移矩阵进行概率计算
    raises(ValueError, lambda: str(P(Eq(YS[3], 2), Eq(YS[1], 1) & TransitionMatrixOf(YS, TSO))))
    raises(TypeError, lambda: DiscreteMarkovChain("Z", [0, 1, 2], symbols('M')))
    raises(ValueError, lambda: DiscreteMarkovChain("Z", [0, 1, 2], MatrixSymbol('T', 3, 4)))
    raises(ValueError, lambda: E(Y[3], Eq(Y[2], 6)))
    raises(ValueError, lambda: E(Y[2], Eq(Y[3], 1)))

    # 对概率查询进行扩展测试

    # 定义转移概率矩阵 TO1
    TO1 = Matrix([[Rational(1, 4), Rational(3, 4), 0],
                  [Rational(1, 3), Rational(1, 3), Rational(1, 3)],
                  [0, Rational(1, 4), Rational(3, 4)]])

    # 验证概率查询结果
    assert P(And(Eq(Y[2], 1), Eq(Y[1], 1), Eq(Y[0], 0)),
             Eq(Probability(Eq(Y[0], 0)), Rational(1, 4)) & TransitionMatrixOf(Y, TO1)) == Rational(1, 16)

    # 验证概率查询结果
    assert P(And(Eq(Y[2], 1), Eq(Y[1], 1), Eq(Y[0], 0)), TransitionMatrixOf(Y, TO1)) == Probability(Eq(Y[0], 0))/4

    # 验证概率查询结果
    assert P(Lt(X[1], 2) & Gt(X[1], 0), Eq(X[0], 2) &
             StochasticStateSpaceOf(X, [0, 1, 2]) & TransitionMatrixOf(X, TO1)) == Rational(1, 4)

    # 验证概率查询结果
    assert P(Lt(X[1], 2) & Gt(X[1], 0), Eq(X[0], 2) &
             StochasticStateSpaceOf(X, [S(0), '0', 1]) & TransitionMatrixOf(X, TO1)) == Rational(1, 4)

    # 验证概率查询结果
    assert P(Ne(X[1], 2) & Ne(X[1], 1), Eq(X[0], 2) &
             StochasticStateSpaceOf(X, [0, 1, 2]) & TransitionMatrixOf(X, TO1)) is S.Zero

    # 验证概率查询结果
    assert P(Ne(X[1], 2) & Ne(X[1], 1), Eq(X[0], 2) &
             StochasticStateSpaceOf(X, [S(0), '0', 1]) & TransitionMatrixOf(X, TO1)) is S.Zero

    # 验证概率查询结果
    assert P(And(Eq(Y[2], 1), Eq(Y[1], 1), Eq(Y[0], 0)), Eq(Y[1], 1)) == 0.1 * Probability(Eq(Y[0], 0))

    # 测试马尔可夫链的属性

    # 定义转移概率矩阵 TO2 和 TO3
    TO2 = Matrix([[S.One, 0, 0],
                  [Rational(1, 3), Rational(1, 3), Rational(1, 3)],
                  [0, Rational(1, 4), Rational(3, 4)]])
    TO3 = Matrix([[Rational(1, 4), Rational(3, 4), 0],
                  [Rational(1, 3), Rational(1, 3), Rational(1, 3)],
                  [0, Rational(1, 4), Rational(3, 4)]])

    # 创建马尔可夫链 Y2 和 Y3
    Y2 = DiscreteMarkovChain('Y', trans_probs=TO2)
    Y3 = DiscreteMarkovChain('Y', trans_probs=TO3)

    # 验证基本矩阵的计算结果
    assert Y3.fundamental_matrix() == ImmutableMatrix([[176, 81, -132], [36, 141, -52], [-44, -39, 208]]) / 125

    # 验证是否吸收链的属性
    assert Y2.is_absorbing_chain() == True
    assert Y3.is_absorbing_chain() == False

    # 验证规范形式的计算结果
    assert Y2.canonical_form() == ([0, 1, 2], TO2)
    assert Y3.canonical_form() == ([0, 1, 2], TO3)

    # 验证分解结果
    assert Y2.decompose() == ([0, 1, 2], TO2[0:1, 0:1], TO2[1:3, 0:1], TO2[1:3, 1:3])
    assert Y3.decompose() == ([0, 1, 2], TO3, Matrix(0, 3, []), Matrix(0, 0, []))

    # 定义转移概率矩阵 TO4 和马尔可夫链 Y4
    TO4 = Matrix([[Rational(1, 5), Rational(2, 5), Rational(2, 5)],
                  [Rational(1, 10), S.Half, Rational(2, 5)],
                  [Rational(3, 5), Rational(3, 10), Rational(1, 10)]])
    Y4 = DiscreteMarkovChain('Y', trans_probs=TO4)

    # 定义极限分布 w
    w = ImmutableMatrix([[Rational(11, 39), Rational(16, 39), Rational(4, 13)]])

    # 验证极限分布的计算结果
    assert Y4.limiting_distribution == w

    # 验证是否正则的属性
    assert Y4.is_regular() == True

    # 验证是否遍历的属性
    assert Y4.is_ergodic() == True

    # 定义转移矩阵符号 TS1 和马尔可夫链 Y5
    TS1 = MatrixSymbol('T', 3, 3)
    Y5 = DiscreteMarkovChain('Y', trans_probs=TS1)

    # 验证极限分布计算的结果
    assert Y5.limiting_distribution(w, TO4).doit() == True

    # 验证稳态分布的计算结果
    assert Y5.stationary_distribution(condition_set=True).subs(TS1, TO4).contains(w).doit() == S.true
    # 创建一个 5x5 的矩阵 TO6，表示马尔可夫链的状态转移概率
    TO6 = Matrix([[S.One, 0, 0, 0, 0],
                  [S.Half, 0, S.Half, 0, 0],
                  [0, S.Half, 0, S.Half, 0],
                  [0, 0, S.Half, 0, S.Half],
                  [0, 0, 0, 0, 1]])
    
    # 创建一个名为 Y6 的离散马尔可夫链对象，使用 TO6 作为状态转移概率矩阵
    Y6 = DiscreteMarkovChain('Y', trans_probs=TO6)
    
    # 断言 Y6 的基本矩阵（fundamental matrix）与给定的不可变矩阵相等
    assert Y6.fundamental_matrix() == ImmutableMatrix([[Rational(3, 2), S.One, S.Half],
                                                        [S.One, S(2), S.One],
                                                        [S.Half, S.One, Rational(3, 2)]])
    
    # 断言 Y6 的吸收概率矩阵（absorbing probabilities）与给定的不可变矩阵相等
    assert Y6.absorbing_probabilities() == ImmutableMatrix([[Rational(3, 4), Rational(1, 4)],
                                                            [S.Half, S.Half],
                                                            [Rational(1, 4), Rational(3, 4)]])
    
    # 使用 warns_deprecated_sympy 上下文，测试 Y6 的 absorbing_probabilites 方法（注：拼写错误，应为 absorbing_probabilities）
    with warns_deprecated_sympy():
        Y6.absorbing_probabilites()
    
    # 创建一个 3x3 的矩阵 TO7，表示另一个马尔可夫链的状态转移概率
    TO7 = Matrix([[Rational(1, 2), Rational(1, 4), Rational(1, 4)],
                  [Rational(1, 2), 0, Rational(1, 2)],
                  [Rational(1, 4), Rational(1, 4), Rational(1, 2)]])
    
    # 创建一个名为 Y7 的离散马尔可夫链对象，使用 TO7 作为状态转移概率矩阵
    Y7 = DiscreteMarkovChain('Y', trans_probs=TO7)
    
    # 断言 Y7 不是一个吸收链
    assert Y7.is_absorbing_chain() == False
    
    # 断言 Y7 的基本矩阵与给定的不可变矩阵相等
    assert Y7.fundamental_matrix() == ImmutableMatrix([[Rational(86, 75), Rational(1, 25), Rational(-14, 75)],
                                                        [Rational(2, 25), Rational(21, 25), Rational(2, 25)],
                                                        [Rational(-14, 75), Rational(1, 25), Rational(86, 75)]])
    
    # 创建一个名为 X 的离散马尔可夫链对象，使用空矩阵作为状态转移概率矩阵，测试零大小矩阵的功能
    X = DiscreteMarkovChain('X', trans_probs=Matrix([[]]))
    
    # 断言 X 的状态数为 0
    assert X.number_of_states == 0
    
    # 断言 X 的静止分布（stationary distribution）与空矩阵相等
    assert X.stationary_distribution() == Matrix([[]])
    
    # 断言 X 的通信类为空列表
    assert X.communication_classes() == []
    
    # 断言 X 的标准型（canonical form）是空列表和空矩阵
    assert X.canonical_form() == ([], Matrix([[]]))
    
    # 断言 X 的分解（decompose）结果是空列表和三个空矩阵
    assert X.decompose() == ([], Matrix([[]]), Matrix([[]]), Matrix([[]]))
    
    # 断言 X 不是正则链
    assert X.is_regular() == False
    
    # 断言 X 不是遍历链
    assert X.is_ergodic() == False
    
    # 创建一个 5x5 的矩阵 TO7，表示另一个马尔可夫链的状态转移概率，用于测试通信类
    TO7 = Matrix([[0, 5, 5, 0, 0],
                  [0, 0, 0, 10, 0],
                  [5, 0, 5, 0, 0],
                  [0, 10, 0, 0, 0],
                  [0, 3, 0, 3, 4]]) / 10
    
    # 创建一个名为 Y7 的离散马尔可夫链对象，使用 TO7 作为状态转移概率矩阵
    Y7 = DiscreteMarkovChain('Y', trans_probs=TO7)
    
    # 获取 Y7 的通信类信息，并解包到三个列表中
    tuples = Y7.communication_classes()
    classes, recurrence, periods = list(zip(*tuples))
    
    # 断言 Y7 的通信类、是否循环及周期与预期的值相符
    assert classes == ([1, 3], [0, 2], [4])
    assert recurrence == (True, False, False)
    assert periods == (2, 1, 1)
    
    # 创建一个 6x6 的矩阵 TO8，表示另一个马尔可夫链的状态转移概率，用于测试通信类
    TO8 = Matrix([[0, 0, 0, 10, 0, 0],
                  [5, 0, 5, 0, 0, 0],
                  [0, 4, 0, 0, 0, 6],
                  [10, 0, 0, 0, 0, 0],
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 0, 5, 5, 0]]) / 10
    
    # 创建一个名为 Y8 的离散马尔可夫链对象，使用 TO8 作为状态转移概率矩阵
    Y8 = DiscreteMarkovChain('Y', trans_probs=TO8)
    
    # 获取 Y8 的通信类信息，并解包到两个列表中
    tuples = Y8.communication_classes()
    classes, recurrence, periods = list(zip(*tuples))
    
    # 断言 Y8 的通信类、是否循环及周期与预期的值相符
    assert classes == ([0, 3], [1, 2, 5, 4])
    assert recurrence == (True, False)
    assert periods == (2, 2)
    # 定义转移概率矩阵 TO9
    TO9 = Matrix([[2, 0, 0, 3, 0, 0, 3, 2, 0, 0],
                  [0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 2, 2, 0, 0, 0, 0, 0, 3, 3],
                  [0, 0, 0, 3, 0, 0, 6, 1, 0, 0],
                  [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 10, 0, 0, 0, 0],
                  [4, 0, 0, 5, 0, 0, 1, 0, 0, 0],
                  [2, 0, 0, 4, 0, 0, 2, 2, 0, 0],
                  [3, 0, 1, 0, 0, 0, 0, 0, 4, 2],
                  [0, 0, 4, 0, 0, 0, 0, 0, 3, 3]]) / 10
    
    # 创建离散马尔可夫链对象 Y9，使用转移概率矩阵 TO9
    Y9 = DiscreteMarkovChain('Y', trans_probs=TO9)
    
    # 获取通讯类元组
    tuples = Y9.communication_classes()
    
    # 解压元组，得到通讯类、是否周期、周期长度
    classes, recurrence, periods = list(zip(*tuples))
    
    # 断言通讯类的预期值
    assert classes == ([0, 3, 6, 7], [1], [2, 8, 9], [5], [4])
    
    # 断言是否周期的预期值
    assert recurrence == (True, True, False, True, False)
    
    # 断言周期长度的预期值
    assert periods == (1, 1, 1, 1, 1)

    # 测试规范形式
    # 参考链接: https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    # 示例 11.13
    T = Matrix([[1, 0, 0, 0, 0],
                [S(1) / 2, 0, S(1) / 2, 0, 0],
                [0, S(1) / 2, 0, S(1) / 2, 0],
                [0, 0, S(1) / 2, 0, S(1) / 2],
                [0, 0, 0, 0, S(1)]])
    
    # 创建离散马尔可夫链对象 DW，使用转移概率矩阵 T 和状态列表 [0, 1, 2, 3, 4]
    DW = DiscreteMarkovChain('DW', [0, 1, 2, 3, 4], T)
    
    # 分解马尔可夫链，得到状态、矩阵 A、矩阵 B、矩阵 C
    states, A, B, C = DW.decompose()
    
    # 断言分解后的状态的预期值
    assert states == [0, 4, 1, 2, 3]
    
    # 断言矩阵 A 的预期值
    assert A == Matrix([[1, 0], [0, 1]])
    
    # 断言矩阵 B 的预期值
    assert B == Matrix([[S(1)/2, 0], [0, 0], [0, S(1)/2]])
    
    # 断言矩阵 C 的预期值
    assert C == Matrix([[0, S(1)/2, 0], [S(1)/2, 0, S(1)/2], [0, S(1)/2, 0]])
    
    # 获取规范形式后的状态列表和新矩阵
    states, new_matrix = DW.canonical_form()
    
    # 断言规范形式后的状态列表的预期值
    assert states == [0, 4, 1, 2, 3]
    
    # 断言规范形式后的新矩阵的预期值
    assert new_matrix == Matrix([[1, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0],
                                 [S(1)/2, 0, 0, S(1)/2, 0],
                                 [0, 0, S(1)/2, 0, S(1)/2],
                                 [0, S(1)/2, 0, S(1)/2, 0]])

    # 测试正则和遍历
    # 参考链接: https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    T = Matrix([[0, 4, 0, 0, 0],
                [1, 0, 3, 0, 0],
                [0, 2, 0, 2, 0],
                [0, 0, 3, 0, 1],
                [0, 0, 0, 4, 0]]) / 4
    
    # 创建离散马尔可夫链对象 X，使用转移概率矩阵 T
    X = DiscreteMarkovChain('X', trans_probs=T)
    
    # 断言 X 不是正则的
    assert not X.is_regular()
    
    # 断言 X 是遍历的
    assert X.is_ergodic()
    
    # 参考链接: http://www.math.wisc.edu/~valko/courses/331/MC2.pdf
    T = Matrix([[2, 1, 1],
                [2, 0, 2],
                [1, 1, 2]]) / 4
    
    # 创建离散马尔可夫链对象 X，使用转移概率矩阵 T
    X = DiscreteMarkovChain('X', trans_probs=T)
    
    # 断言 X 是正则的
    assert X.is_regular()
    
    # 断言 X 是遍历的
    assert X.is_ergodic()
    
    # 参考链接: https://docs.ufpr.br/~lucambio/CE222/1S2014/Kemeny-Snell1976.pdf
    T = Matrix([[1, 1], [1, 1]]) / 2
    
    # 创建离散马尔可夫链对象 X，使用转移概率矩阵 T
    X = DiscreteMarkovChain('X', trans_probs=T)
    
    # 断言 X 是正则的
    assert X.is_regular()
    
    # 断言 X 是遍历的
    assert X.is_ergodic()

    # 测试吸收链判断
    # 创建一个转移矩阵 T，表示马尔可夫链的转移概率
    T = Matrix([[0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]])
    # 创建一个名为 X 的离散马尔可夫链对象，使用上述转移矩阵 T
    X = DiscreteMarkovChain('X', trans_probs=T)
    # 断言此马尔可夫链不是吸收链
    assert not X.is_absorbing_chain()

    # 创建另一个转移矩阵 T，表示马尔可夫链的转移概率
    T = Matrix([[1, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 2]])/2
    # 使用新的转移矩阵 T 创建马尔可夫链对象 X
    X = DiscreteMarkovChain('X', trans_probs=T)
    # 断言此马尔可夫链是吸收链
    assert X.is_absorbing_chain()

    # 创建另一个转移矩阵 T，表示马尔可夫链的转移概率
    T = Matrix([[2, 0, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 0, 2]])/2
    # 使用新的转移矩阵 T 创建马尔可夫链对象 X
    X = DiscreteMarkovChain('X', trans_probs=T)
    # 断言此马尔可夫链是吸收链
    assert X.is_absorbing_chain()

    # 测试自定义状态空间
    # 创建一个名为 Y10 的离散马尔可夫链对象，指定状态空间和转移概率
    Y10 = DiscreteMarkovChain('Y', [1, 2, 3], TO2)
    # 调用 communication_classes 方法，获取通信类元组列表
    tuples = Y10.communication_classes()
    # 将通信类元组列表解压为各个组件
    classes, recurrence, periods = list(zip(*tuples))
    # 断言通信类列表与预期的相等
    assert classes == ([1], [2, 3])
    # 断言各通信类的周期性
    assert recurrence == (True, False)
    # 断言各通信类的周期
    assert periods == (1, 1)
    # 断言 Y10 的标准形式与预期相等
    assert Y10.canonical_form() == ([1, 2, 3], TO2)
    # 断言 Y10 的分解形式与预期相等
    assert Y10.decompose() == ([1, 2, 3], TO2[0:1, 0:1], TO2[1:3, 0:1], TO2[1:3, 1:3])

    # 测试各种其他查询
    # 创建一个转移矩阵 T，表示马尔可夫链的转移概率
    T = Matrix([[S.Half, Rational(1, 4), Rational(1, 4)],
                [Rational(1, 3), 0, Rational(2, 3)],
                [S.Half, S.Half, 0]])
    # 创建一个名为 X 的离散马尔可夫链对象，指定状态空间和转移概率
    X = DiscreteMarkovChain('X', [0, 1, 2], T)
    # 断言特定事件的概率等于预期值
    assert P(Eq(X[1], 2) & Eq(X[2], 1) & Eq(X[3], 0),
             Eq(P(Eq(X[1], 0)), Rational(1, 4)) & Eq(P(Eq(X[1], 1)), Rational(1, 4))) == Rational(1, 12)
    # 断言条件概率等于预期值
    assert P(Eq(X[2], 1) | Eq(X[2], 2), Eq(X[1], 1)) == Rational(2, 3)
    # 断言条件概率为零
    assert P(Eq(X[2], 1) & Eq(X[2], 2), Eq(X[1], 1)) is S.Zero
    # 断言条件概率等于预期值
    assert P(Ne(X[2], 2), Eq(X[1], 1)) == Rational(1, 3)
    # 断言在给定条件下的期望等于预期值
    assert E(X[1]**2, Eq(X[0], 1)) == Rational(8, 3)
    # 断言在给定条件下的方差等于预期值
    assert variance(X[1], Eq(X[0], 1)) == Rational(8, 9)
    # 断言会引发 ValueError 异常
    raises(ValueError, lambda: E(X[1], Eq(X[2], 1)))
    # 断言会引发 ValueError 异常
    raises(ValueError, lambda: DiscreteMarkovChain('X', [0, 1], T))

    # 测试具有不同状态空间的其他查询
    # 创建一个名为 X 的离散马尔可夫链对象，指定不同的状态空间和转移概率
    X = DiscreteMarkovChain('X', ['A', 'B', 'C'], T)
    # 断言特定事件的概率等于预期值
    assert P(Eq(X[1], 2) & Eq(X[2], 1) & Eq(X[3], 0),
             Eq(P(Eq(X[1], 0)), Rational(1, 4)) & Eq(P(Eq(X[1], 1)), Rational(1, 4))) == Rational(1, 12)
    # 断言条件概率等于预期值
    assert P(Eq(X[2], 1) | Eq(X[2], 2), Eq(X[1], 1)) == Rational(2, 3)
    # 断言条件概率为零
    assert P(Eq(X[2], 1) & Eq(X[2], 2), Eq(X[1], 1)) is S.Zero
    # 断言条件概率等于预期值
    assert P(Ne(X[2], 2), Eq(X[1], 1)) == Rational(1, 3)
    # 计算和断言特定的数学表达式
    a = X.state_space.args[0]
    c = X.state_space.args[2]
    assert (E(X[1] ** 2, Eq(X[0], 1)) - (a**2/3 + 2*c**2/3)).simplify() == 0
    assert (variance(X[1], Eq(X[0], 1)) - (2*(-a/3 + c/3)**2/3 + (2*a/3 - 2*c/3)**2/3)).simplify() == 0
    # 断言会引发 ValueError 异常
    raises(ValueError, lambda: E(X[1], Eq(X[2], 1)))

    # 测试具有多个 RandomIndexedSymbols 的查询
    # 创建一个转移矩阵 T，表示马尔可夫链的转移概率
    T = Matrix([[Rational(5, 10), Rational(3, 10), Rational(2, 10)],
                [Rational(2, 10), Rational(7, 10), Rational(1, 10)],
                [Rational(3, 10), Rational(3, 10), Rational(4, 10)]])
    # 创建一个名为 Y 的离散马尔可夫链对象，指定状态空间和转移概率
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    # 断言：计算条件概率，并四舍五入到小数点后五位，检查是否等于给定的浮点数
    assert P(Eq(Y[7], Y[5]), Eq(Y[2], 0)).round(5) == Float(0.44428, 5)
    
    # 断言：计算条件概率，并四舍五入到小数点后两位，检查是否等于给定的浮点数
    assert P(Gt(Y[3], Y[1]), Eq(Y[0], 0)).round(2) == Float(0.36, 2)
    
    # 断言：计算条件概率，并四舍五入到小数点后六位，检查是否等于给定的浮点数
    assert P(Le(Y[5], Y[10]), Eq(Y[4], 2)).round(6) == Float(0.583120, 6)
    
    # 断言：计算条件概率，并四舍五入到小数点后十四位，检查是否等于另一个条件概率的补数
    assert Float(P(Eq(Y[10], Y[5]), Eq(Y[4], 1)), 14) == Float(1 - P(Ne(Y[10], Y[5]), Eq(Y[4], 1)), 14)
    
    # 断言：计算条件概率，并四舍五入到小数点后十四位，检查是否等于另一个条件概率的补数
    assert Float(P(Gt(Y[8], Y[9]), Eq(Y[3], 2)), 14) == Float(1 - P(Le(Y[8], Y[9]), Eq(Y[3], 2)), 14)
    
    # 断言：计算条件概率，并四舍五入到小数点后十四位，检查是否等于另一个条件概率的补数
    assert Float(P(Lt(Y[1], Y[4]), Eq(Y[0], 0)), 14) == Float(1 - P(Ge(Y[1], Y[4]), Eq(Y[0], 0)), 14)
    
    # 断言：两个条件概率是否相等
    assert P(Eq(Y[5], Y[10]), Eq(Y[2], 1)) == P(Eq(Y[10], Y[5]), Eq(Y[2], 1))
    
    # 断言：两个条件概率是否相等
    assert P(Gt(Y[1], Y[2]), Eq(Y[0], 1)) == P(Lt(Y[2], Y[1]), Eq(Y[0], 1))
    
    # 断言：两个条件概率是否相等
    assert P(Ge(Y[7], Y[6]), Eq(Y[4], 1)) == P(Le(Y[6], Y[7]), Eq(Y[4], 1))

    # 测试符号查询
    # 定义符号变量
    a, b, c, d = symbols('a b c d')
    # 创建转移概率矩阵
    T = Matrix([[Rational(1, 10), Rational(4, 10), Rational(5, 10)], [Rational(3, 10), Rational(4, 10), Rational(3, 10)], [Rational(7, 10), Rational(2, 10), Rational(1, 10)]])
    # 创建离散马尔可夫链对象
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    # 创建查询条件
    query = P(Eq(Y[a], b), Eq(Y[c], d))
    # 断言：对查询进行符号替换和浮点化，四舍五入到小数点后四位，检查是否等于另一个条件概率的浮点化结果
    assert query.subs({a:10, b:2, c:5, d:1}).evalf().round(4) == P(Eq(Y[10], 2), Eq(Y[5], 1)).round(4)
    # 断言：对查询进行符号替换和浮点化，四舍五入到小数点后四位，检查是否等于另一个条件概率的浮点化结果
    assert query.subs({a:15, b:0, c:10, d:1}).evalf().round(4) == P(Eq(Y[15], 0), Eq(Y[10], 1)).round(4)
    # 创建大于查询条件
    query_gt = P(Gt(Y[a], b), Eq(Y[c], d))
    # 创建小于等于查询条件
    query_le = P(Le(Y[a], b), Eq(Y[c], d))
    # 断言：对大于查询条件和小于等于查询条件进行符号替换和浮点化后求和，检查是否等于1.0
    assert query_gt.subs({a:5, b:2, c:1, d:0}).evalf() + query_le.subs({a:5, b:2, c:1, d:0}).evalf() == 1.0
    # 创建大于等于查询条件
    query_ge = P(Ge(Y[a], b), Eq(Y[c], d))
    # 创建小于查询条件
    query_lt = P(Lt(Y[a], b), Eq(Y[c], d))
    # 断言：对大于等于查询条件和小于查询条件进行符号替换和浮点化后求和，检查是否等于1.0
    assert query_ge.subs({a:4, b:1, c:0, d:2}).evalf() + query_lt.subs({a:4, b:1, c:0, d:2}).evalf() == 1.0

    # 测试问题20078
    # 断言：测试表达式简化是否正确
    assert (2*Y[1] + 3*Y[1]).simplify() == 5*Y[1]
    # 断言：测试表达式简化是否正确
    assert (2*Y[1] - 3*Y[1]).simplify() == -Y[1]
    # 断言：测试表达式简化是否正确
    assert (2*(0.25*Y[1])).simplify() == 0.5*Y[1]
    # 断言：测试表达式简化是否正确
    assert ((2*Y[1]) * (0.25*Y[1])).simplify() == 0.5*Y[1]**2
    # 断言：测试表达式简化是否正确
    assert (Y[1]**2 + Y[1]**3).simplify() == (Y[1] + 1)*Y[1]**2
def test_sample_stochastic_process():
    # 检查是否导入了SciPy模块，如果没有则跳过采样测试
    if not import_module('scipy'):
        skip('SciPy Not installed. Skip sampling tests')
    import random
    random.seed(0)  # 设置随机数种子为0
    numpy = import_module('numpy')
    if numpy:
        numpy.random.seed(0)  # 如果导入了numpy，则设置其随机数种子为0，因为SciPy使用numpy进行采样
    T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    for samps in range(10):
        # 断言从随机过程Y的采样结果属于Y的状态空间
        assert next(sample_stochastic_process(Y)) in Y.state_space
    Z = DiscreteMarkovChain("Z", ['1', 1, 0], T)
    for samps in range(10):
        # 断言从随机过程Z的采样结果属于Z的状态空间
        assert next(sample_stochastic_process(Z)) in Z.state_space

    T = Matrix([[S.Half, Rational(1, 4), Rational(1, 4)],
                [Rational(1, 3), 0, Rational(2, 3)],
                [S.Half, S.Half, 0]])
    X = DiscreteMarkovChain('X', [0, 1, 2], T)
    for samps in range(10):
        # 断言从随机过程X的采样结果属于X的状态空间
        assert next(sample_stochastic_process(X)) in X.state_space
    W = DiscreteMarkovChain('W', [1, pi, oo], T)
    for samps in range(10):
        # 断言从随机过程W的采样结果属于W的状态空间
        assert next(sample_stochastic_process(W)) in W.state_space


def test_ContinuousMarkovChain():
    T1 = Matrix([[S(-2), S(2), S.Zero],
                 [S.Zero, S.NegativeOne, S.One],
                 [Rational(3, 2), Rational(3, 2), S(-3)]])
    C1 = ContinuousMarkovChain('C', [0, 1, 2], T1)
    # 断言连续马尔可夫链C1的极限分布为指定的不可变矩阵
    assert C1.limiting_distribution() == ImmutableMatrix([[Rational(3, 19), Rational(12, 19), Rational(4, 19)]])

    T2 = Matrix([[-S.One, S.One, S.Zero], [S.One, -S.One, S.Zero], [S.Zero, S.One, -S.One]])
    C2 = ContinuousMarkovChain('C', [0, 1, 2], T2)
    A, t = C2.generator_matrix, symbols('t', positive=True)
    # 断言连续马尔可夫链C2的过渡概率矩阵为指定的矩阵表达式
    assert C2.transition_probabilities(A)(t) == Matrix([[S.Half + exp(-2*t)/2, S.Half - exp(-2*t)/2, 0],
                                                       [S.Half - exp(-2*t)/2, S.Half + exp(-2*t)/2, 0],
                                                       [S.Half - exp(-t) + exp(-2*t)/2, S.Half - exp(-2*t)/2, exp(-t)]])
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        # 断言忽略用户警告后的概率表达式
        assert P(Eq(C2(1), 1), Eq(C2(0), 1), evaluate=False) == Probability(Eq(C2(1), 1), Eq(C2(0), 1))
    # 断言连续马尔可夫链C2的概率为指定表达式
    assert P(Eq(C2(1), 1), Eq(C2(0), 1)) == exp(-2)/2 + S.Half
    # 断言连续马尔可夫链C2的条件概率为指定表达式
    assert P(Eq(C2(1), 0) & Eq(C2(2), 1) & Eq(C2(3), 1),
                Eq(P(Eq(C2(1), 0)), S.Half)) == (Rational(1, 4) - exp(-2)/4)*(exp(-2)/2 + S.Half)
    # 断言连续马尔可夫链C2的条件概率为指定表达式
    assert P(Not(Eq(C2(1), 0) & Eq(C2(2), 1) & Eq(C2(3), 2)) |
                (Eq(C2(1), 0) & Eq(C2(2), 1) & Eq(C2(3), 2)),
                Eq(P(Eq(C2(1), 0)), Rational(1, 4)) & Eq(P(Eq(C2(1), 1)), Rational(1, 4))) is S.One
    # 断言连续马尔可夫链C2的期望为指定表达式
    assert E(C2(Rational(3, 2)), Eq(C2(0), 2)) == -exp(-3)/2 + 2*exp(Rational(-3, 2)) + S.Half
    # 断言连续马尔可夫链C2的方差为指定表达式
    assert variance(C2(Rational(3, 2)), Eq(C2(0), 1)) == ((S.Half - exp(-3)/2)**2*(exp(-3)/2 + S.Half)
                                                    + (Rational(-1, 2) - exp(-3)/2)**2*(S.Half - exp(-3)/2))
    # 断言引发指定的异常
    raises(KeyError, lambda: P(Eq(C2(1), 0), Eq(P(Eq(C2(1), 1)), S.Half)))
    # 使用 SymPy 的概率模块进行断言测试
    assert P(Eq(C2(1), 0), Eq(P(Eq(C2(5), 1)), S.Half)) == Probability(Eq(C2(1), 0))

    # 创建一个 3x3 的符号矩阵符号 'G'
    TS1 = MatrixSymbol('G', 3, 3)

    # 使用连续马尔可夫链创建对象 'CS1'，状态空间为 [0, 1, 2]，传递概率矩阵为 'TS1'
    CS1 = ContinuousMarkovChain('C', [0, 1, 2], TS1)

    # 获取 'CS1' 的生成矩阵 'A'
    A = CS1.generator_matrix

    # 断言：连续马尔可夫链 'CS1' 的状态转移概率函数为 exp(t*A)
    assert CS1.transition_probabilities(A)(t) == exp(t*A)

    # 创建一个带有符号状态空间的连续马尔可夫链 'C3'，状态空间为 [Symbol('0'), Symbol('1'), Symbol('2')]，传递概率矩阵为 'T2'
    C3 = ContinuousMarkovChain('C', [Symbol('0'), Symbol('1'), Symbol('2')], T2)

    # 断言：概率 P(C3(1)=1, C3(0)=1) 的计算结果为 exp(-2)/2 + S.Half
    assert P(Eq(C3(1), 1), Eq(C3(0), 1)) == exp(-2)/2 + S.Half

    # 断言：概率 P(C3(1)=Symbol('1'), C3(0)=Symbol('1')) 的计算结果为 exp(-2)/2 + S.Half
    assert P(Eq(C3(1), Symbol('1')), Eq(C3(0), Symbol('1'))) == exp(-2)/2 + S.Half

    # 测试概率查询
    # 创建一个 3x3 的符号矩阵 'G'
    G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])

    # 使用连续马尔可夫链创建对象 'C'，状态空间为 [0, 1, 2]，传递概率矩阵为 'G'
    C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)

    # 断言：概率 P(C(7.385)=C(3.19), C(0.862)=0) 的计算结果约为 0.35469（精确到小数点后五位）
    assert P(Eq(C(7.385), C(3.19)), Eq(C(0.862), 0)).round(5) == Float(0.35469, 5)

    # 断言：概率 P(C(98.715)>C(19.807), C(11.314)=2) 的计算结果约为 0.32452（精确到小数点后五位）
    assert P(Gt(C(98.715), C(19.807)), Eq(C(11.314), 2)).round(5) == Float(0.32452, 5)

    # 断言：概率 P(C(5.9)<=C(10.112), C(4)=1) 的计算结果约为 0.675214（精确到小数点后六位）
    assert P(Le(C(5.9), C(10.112)), Eq(C(4), 1)).round(6) == Float(0.675214, 6)

    # 断言：概率 P(C(7.32)=C(2.91), C(2.63)=1) 的计算结果约为 1 - P(C(7.32)!=C(2.91), C(2.63)=1)
    assert Float(P(Eq(C(7.32), C(2.91)), Eq(C(2.63), 1)), 14) == Float(1 - P(Ne(C(7.32), C(2.91)), Eq(C(2.63), 1)), 14)

    # 断言：概率 P(C(3.36)>C(1.101), C(0.8)=2) 的计算结果约为 1 - P(C(3.36)<=C(1.101), C(0.8)=2)
    assert Float(P(Gt(C(3.36), C(1.101)), Eq(C(0.8), 2)), 14) == Float(1 - P(Le(C(3.36), C(1.101)), Eq(C(0.8), 2)), 14)

    # 断言：概率 P(C(4.9)<C(2.79), C(1.61)=0) 的计算结果约为 1 - P(C(4.9)>=C(2.79), C(1.61)=0)
    assert Float(P(Lt(C(4.9), C(2.79)), Eq(C(1.61), 0)), 14) == Float(1 - P(Ge(C(4.9), C(2.79)), Eq(C(1.61), 0)), 14)

    # 断言：概率 P(C(5.243)=C(10.912), C(2.174)=1) 等于 P(C(10.912)=C(5.243), C(2.174)=1)
    assert P(Eq(C(5.243), C(10.912)), Eq(C(2.174), 1)) == P(Eq(C(10.912), C(5.243)), Eq(C(2.174), 1))

    # 断言：概率 P(C(2.344)>C(9.9), C(1.102)=1) 等于 P(C(9.9)<C(2.344), C(1.102)=1)
    assert P(Gt(C(2.344), C(9.9)), Eq(C(1.102), 1)) == P(Lt(C(9.9), C(2.344)), Eq(C(1.102), 1))

    # 断言：概率 P(C(7.87)>=C(1.008), C(0.153)=1) 等于 P(C(1.008)<=C(7.87), C(0.153)=1)
    assert P(Ge(C(7.87), C(1.008)), Eq(C(0.153), 1)) == P(Le(C(1.008), C(7.87)), Eq(C(0.153), 1))

    # 测试符号查询
    a, b, c, d = symbols('a b c d')

    # 创建概率查询表达式 'query'
    query = P(Eq(C(a), b), Eq(C(c), d))

    # 断言：使用具体值替换后，查询表达式的数值结果约等于原始查询表达式的数值结果
    assert query.subs({a:3.65, b:2, c:1.78, d:1}).evalf().round(10) == P(Eq(C(3.65), 2), Eq(C(1.78), 1)).round(10)

    # 创建大于查询 'query_gt'
    query_gt = P(Gt(C(a), b), Eq(C(c), d))

    # 创建小于等于查询 'query_le'
    query_le = P(Le(C(a), b), Eq(C(c), d))

    # 断言：大于查询和小于等于查询的结果之和等于 1.0
    assert query_gt.subs({a:13.2, b:0, c:3.29, d:2}).evalf() + query_le.subs({a:13.2, b:0, c:3.29, d:2}).evalf() == 1.0

    # 创建大于等于查询 'query_ge'
    query_ge = P(Ge(C(a), b), Eq(C(c), d))

    # 创建小于查询 'query_lt'
    query_lt = P(Lt(C(a), b), Eq(C(c), d))

    # 断言：大于等于查询和小于查询的结果之和等于 1.0
    assert query_ge.subs({a:7.43, b:1, c:1.45, d:0}).evalf() + query_lt
# 定义一个测试函数，用于测试伯努利过程类的各种属性和方法
def test_BernoulliProcess():

    # 创建一个伯努利过程对象B，指定成功概率为0.6，成功状态为1，失败状态为0
    B = BernoulliProcess("B", p=0.6, success=1, failure=0)
    # 断言状态空间为 {0, 1}
    assert B.state_space == FiniteSet(0, 1)
    # 断言索引集为自然数非负整数集合
    assert B.index_set == S.Naturals0
    # 断言成功状态为1
    assert B.success == 1
    # 断言失败状态为0
    assert B.failure == 0

    # 创建一个伯努利过程对象X，指定成功概率为1/3，成功状态为'H'，失败状态为'T'
    X = BernoulliProcess("X", p=Rational(1,3), success='H', failure='T')
    # 断言状态空间为 {'H', 'T'}
    assert X.state_space == FiniteSet('H', 'T')
    # 定义符号变量 H 和 T
    H, T = symbols("H,T")
    # 断言期望值 E(X[1]+X[2]*X[3]) 的计算结果
    assert E(X[1]+X[2]*X[3]) == H**2/9 + 4*H*T/9 + H/3 + 4*T**2/9 + 2*T/3

    # 定义正整数符号变量 t 和 x
    t, x = symbols('t, x', positive=True, integer=True)
    # 断言 B[t] 是随机索引符号对象的实例
    assert isinstance(B[t], RandomIndexedSymbol)

    # 测试伯努利过程对象参数异常处理
    raises(ValueError, lambda: BernoulliProcess("X", p=1.1, success=1, failure=0))
    # 测试 B(t) 方法未实现异常处理
    raises(NotImplementedError, lambda: B(t))

    # 测试索引越界异常处理
    raises(IndexError, lambda: B[-3])
    # 断言联合分布的手动定义，使用 lambda 表达式生成关于 B[3] 和 B[9] 的联合分布对象
    assert B.joint_distribution(B[3], B[9]) == JointDistributionHandmade(Lambda((B[3], B[9]),
                Piecewise((0.6, Eq(B[3], 1)), (0.4, Eq(B[3], 0)), (0, True))
                *Piecewise((0.6, Eq(B[9], 1)), (0.4, Eq(B[9], 0)), (0, True))))

    # 断言联合分布的手动定义，使用 lambda 表达式生成关于 B[2] 和 B[4] 的联合分布对象
    assert B.joint_distribution(2, B[4]) == JointDistributionHandmade(Lambda((B[2], B[4]),
                Piecewise((0.6, Eq(B[2], 1)), (0.4, Eq(B[2], 0)), (0, True))
                *Piecewise((0.6, Eq(B[4], 1)), (0.4, Eq(B[4], 0)), (0, True))))

    # 测试伯努利过程随机变量之和的分布
    Y = B[1] + B[2] + B[3]
    assert P(Eq(Y, 0)).round(2) == Float(0.06, 1)
    assert P(Eq(Y, 2)).round(2) == Float(0.43, 2)
    assert P(Eq(Y, 4)).round(2) == 0
    assert P(Gt(Y, 1)).round(2) == Float(0.65, 2)
    # 测试各随机索引变量的独立性
    assert P(Eq(B[1], 0) & Eq(B[2], 1) & Eq(B[3], 0) & Eq(B[4], 1)).round(2) == Float(0.06, 1)

    # 断言 E(2 * B[1] + B[2]) 的期望值计算结果
    assert E(2 * B[1] + B[2]).round(2) == Float(1.80, 3)
    # 断言 E(2 * B[1] + B[2] + 5) 的期望值计算结果
    assert E(2 * B[1] + B[2] + 5).round(2) == Float(6.80, 3)
    # 断言 E(B[2] * B[4] + B[10]) 的期望值计算结果
    assert E(B[2] * B[4] + B[10]).round(2) == Float(0.96, 2)
    # 断言 E(B[2] > 0, Eq(B[1],1) & Eq(B[2],1)) 的期望值计算结果
    assert E(B[2] > 0, Eq(B[1],1) & Eq(B[2],1)).round(2) == Float(0.60,2)
    # 断言 E(B[1]) 的期望值为0.6
    assert E(B[1]) == 0.6
    # 断言 P(B[1] > 0) 的概率计算结果
    assert P(B[1] > 0).round(2) == Float(0.60, 2)
    # 断言 P(B[1] < 1) 的概率计算结果
    assert P(B[1] < 1).round(2) == Float(0.40, 2)
    # 断言 P(B[1] > 0, B[2] <= 1) 的联合概率计算结果
    assert P(B[1] > 0, B[2] <= 1).round(2) == Float(0.60, 2)
    # 断言 P(B[12] * B[5] > 0) 的概率计算结果
    assert P(B[12] * B[5] > 0).round(2) == Float(0.36, 2)
    # 断言 P(B[12] * B[5] > 0, B[4] < 1) 的联合概率计算结果
    assert P(B[12] * B[5] > 0, B[4] < 1).round(2) == Float(0.36, 2)
    # 断言 P(Eq(B[2], 1), B[2] > 0) 的条件概率计算结果
    assert P(Eq(B[2], 1), B[2] > 0) == 1.0
    # 断言 P(Eq(B[5], 3)) 的概率为0
    assert P(Eq(B[5], 3)) == 0
    # 断言 P(Eq(B[1], 1), B[1] < 0) 的概率为0
    assert P(Eq(B[1], 1), B[1] < 0) == 0
    # 断言 P(B[2] > 0, Eq(B[2], 1)) 的概率为1
    assert P(B[2] > 0, Eq(B[2], 1)) == 1
    # 断言 P(B[2] < 0, Eq(B[2], 1)) 的概率为0
    assert P(B[2] < 0, Eq(B[2], 1)) == 0
    # 断言 P(B[2] > 0, B[2]==7) 的概率为0
    assert P(B[2] > 0, B[2]==7) == 0
    # 断言 P(B[5] > 0, B[5]) 的概率分布对象为伯努利分布
    assert P(B[5] > 0, B[5]) == BernoulliDistribution(0.6, 0, 1)
    # 测试异常处理：不能传入数字作为概率事件
    raises(ValueError, lambda: P(3))
    # 测试异常处理：不能在概率事件中传入数字
    raises(ValueError, lambda: P(B[3] > 0, 3))

    # 测试问题 19456
    # 定义 B[t] 的求
    # 确定表达式 B[x*t] 的自由符号集合是否包含 {B[x*t], x, t}
    assert B[x*t].free_symbols == {B[x*t], x, t}

    # 测试问题编号 20078
    # 简化后的表达式比较
    assert (2*B[t] + 3*B[t]).simplify() == 5*B[t]
    assert (2*B[t] - 3*B[t]).simplify() == -B[t]
    assert (2*(0.25*B[t])).simplify() == 0.5*B[t]
    assert (2*B[t] * 0.25*B[t]).simplify() == 0.5*B[t]**2
    assert (B[t]**2 + B[t]**3).simplify() == (B[t] + 1)*B[t]**2
def test_PoissonProcess():
    # 创建一个 Poisson 过程实例 X，参数为 "X" 和 λ=3
    X = PoissonProcess("X", 3)
    # 断言状态空间为非负整数集合
    assert X.state_space == S.Naturals0
    # 断言索引集合为区间 [0, ∞)
    assert X.index_set == Interval(0, oo)
    # 断言 λ 值为 3
    assert X.lamda == 3

    # 定义符号变量 t, d, x, y，均为正数
    t, d, x, y = symbols('t d x y', positive=True)
    # 断言 X(t) 返回一个随机索引符号的实例
    assert isinstance(X(t), RandomIndexedSymbol)
    # 断言 X(t) 的分布为 Poisson 分布，参数为 3*t
    assert X.distribution(t) == PoissonDistribution(3*t)
    # 使用 warns_deprecated_sympy() 上下文管理器检查 X.distribution(X(t)) 是否发出过时警告
    with warns_deprecated_sympy():
        X.distribution(X(t))
    # 断言创建 PoissonProcess 对象时，如果 λ 为负数会引发 ValueError 异常
    raises(ValueError, lambda: PoissonProcess("X", -1))
    # 断言对于不支持的操作 X[t] 会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: X[t])
    # 断言对于索引为负数时访问 X(-5) 会引发 IndexError 异常
    raises(IndexError, lambda: X(-5))

    # 断言 X(2) 和 X(3) 的联合分布为手动创建的 JointDistributionHandmade 对象
    assert X.joint_distribution(X(2), X(3)) == JointDistributionHandmade(Lambda((X(2), X(3)),
                6**X(2)*9**X(3)*exp(-15)/(factorial(X(2))*factorial(X(3)))))

    # 断言 X(4) 和 X(6) 的联合分布为手动创建的 JointDistributionHandmade 对象
    assert X.joint_distribution(4, 6) == JointDistributionHandmade(Lambda((X(4), X(6)),
                12**X(4)*18**X(6)*exp(-30)/(factorial(X(4))*factorial(X(6)))))

    # 断言事件 P(X(t) < 1) 的概率为 exp(-3*t)
    assert P(X(t) < 1) == exp(-3*t)
    # 断言事件 P(Eq(X(t), 0)) 在区间 [3, 5) 中的概率为 exp(-6)
    assert P(Eq(X(t), 0), Contains(t, Interval.Lopen(3, 5))) == exp(-6)  # exp(-2*lamda)
    # 断言事件 P(Eq(X(t), 1)) 在区间 [3, 4) 中的概率为 3*exp(-3)
    res = P(Eq(X(t), 1), Contains(t, Interval.Lopen(3, 4)))
    assert res == 3*exp(-3)

    # 断言事件 P(Eq(X(t), 1) & Eq(X(d), 1) & Eq(X(x), 1) & Eq(X(y), 1)) 的概率，
    # 因为区间不重叠，等价于 res 的四次方
    assert P(Eq(X(t), 1) & Eq(X(d), 1) & Eq(X(x), 1) & Eq(X(y), 1), Contains(t, Interval.Lopen(0, 1))
    & Contains(d, Interval.Lopen(1, 2)) & Contains(x, Interval.Lopen(2, 3))
    & Contains(y, Interval.Lopen(3, 4))) == res**4

    # 断言事件 P(Eq(X(t), 2) & Eq(X(d), 3)) 在区间 [0, 2) 和 [2, 4) 中的概率相等，
    # 返回 Probability 对象
    assert P(Eq(X(t), 2) & Eq(X(d), 3), Contains(t, Interval.Lopen(0, 2))
    & Contains(d, Interval.Ropen(2, 4))) == \
                Probability(Eq(X(d), 3) & Eq(X(t), 2), Contains(t, Interval.Lopen(0, 2))
                & Contains(d, Interval.Ropen(2, 4)))

    # 断言事件 P(Eq(X(t), 2) & Eq(X(d), 3)) 在区间 [0, 4) 中，引发 ValueError 异常（d 无界）
    raises(ValueError, lambda: P(Eq(X(t), 2) & Eq(X(d), 3),
    Contains(t, Interval.Lopen(0, 4)) & Contains(d, Interval.Lopen(3, oo)))) # no bound on d
    # 断言事件 P(Eq(X(3), 2)) 的概率为 81*exp(-9)/2
    assert P(Eq(X(3), 2)) == 81*exp(-9)/2
    # 断言事件 P(Eq(X(t), 2)) 在区间 [0, 5) 中的概率为 225*exp(-15)/2
    assert P(Eq(X(t), 2), Contains(t, Interval.Lopen(0, 5))) == 225*exp(-15)/2

    # 检查概率计算的正确性，验证其和为 1
    res1 = P(X(t) <= 3, Contains(t, Interval.Lopen(0, 5)))
    res2 = P(X(t) > 3, Contains(t, Interval.Lopen(0, 5)))
    assert res1 == 691*exp(-15)
    assert (res1 + res2).simplify() == 1

    # 检查 Not 和 Or 的事件概率计算
    assert P(Not(Eq(X(t), 2) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) & \
            Contains(d, Interval.Lopen(7, 8))).simplify() == -18*exp(-6) + 234*exp(-9) + 1
    # 断言事件 P(Eq(X(t), 2) | Ne(X(t), 4)) 在区间 (2, 4] 中的概率为 1 - 36*exp(-6)
    assert P(Eq(X(t), 2) | Ne(X(t), 4), Contains(t, Interval.Ropen(2, 4))) == 1 - 36*exp(-6)
    # 断言事件 P(X(t) > 2, X(t) + X(d)) 引发 ValueError 异常
    raises(ValueError, lambda: P(X(t) > 2, X(t) + X(d)))
    # 断言 E(X(t)) 的期望为 3*t，即给定时间戳 t 的分布性质
    assert E(X(t)) == 3*t
    # 断言 E(X(t)**2 + X(d)*2 + X(y)**3) 在区间 [0, 1), [1, 2), [3, 4) 中的期望为 75
    assert E(X(t)**2 + X(d)*2 + X(y)**3, Contains(t, Interval.Lopen(0, 1))
        & Contains(d, Interval.Lopen(1, 2)) & Contains(y, Interval.Ropen(3, 4))) == 75
    # 断言 E(X(t)**2) 在区间 [0, 1) 中的期望为 12
    assert E(X(t)**2, Contains(t, Interval.Lopen(0, 1))) == 12
    # 断言 E(x*(X(t) + X(d))*(X(t)**2+X(d)**2)) 在区间 [0, 1) 中的期望
    assert E(x*(X(t) + X(d))*(X(t)**2+X(d)**2), Contains(t, Interval.Lopen(0, 1))
    & Contains(d, Interval.Ropen(1, 2))) == \
            Expectation(x*(X(d) + X(t))*(X(d)**2 + X(t)**2), Contains(t, Interval.Lopen(0, 1))
            & Contains(d, Interval.Ropen(1, 2)))

    # 检查条件中的时间上限无限，会引发值错误
    raises(ValueError, lambda: E(X(t)**3, Contains(t, Interval.Lopen(1, oo))))

    # 计算期望值，验证等式是否成立
    assert E((X(t) + X(d))*(X(t) - X(d)), Contains(t, Interval.Lopen(0, 1))
        & Contains(d, Interval.Lopen(1, 2))) == 0

    # 计算期望值，验证等式是否成立
    assert E(X(2) + x*E(X(5))) == 15*x + 6

    # 计算期望值，验证等式是否成立
    assert E(x*X(1) + y) == 3*x + y

    # 计算概率，验证等式是否成立
    assert P(Eq(X(1), 2) & Eq(X(t), 3), Contains(t, Interval.Lopen(1, 2))) == 81*exp(-6)/4

    # 创建一个泊松过程对象 "Y"，参数为 6
    Y = PoissonProcess("Y", 6)

    # 将 X 和 Y 合并成 Z
    Z = X + Y

    # 断言 Z 的参数 lamda 等于 X 和 Y 的参数 lamda 之和，即 9
    assert Z.lamda == X.lamda + Y.lamda == 9

    # 检查是否会引发值错误，因为只能将 PoissonProcess 实例添加到 Z 中
    raises(ValueError, lambda: X + 5)

    # 将 Z 拆分为两个新的泊松过程 N 和 M，参数分别为 4 和 5
    N, M = Z.split(4, 5)

    # 断言 N 的参数 lamda 等于 4
    assert N.lamda == 4

    # 断言 M 的参数 lamda 等于 5
    assert M.lamda == 5

    # 检查是否会引发值错误，因为拆分的参数之和不等于 Z 的参数 lamda
    raises(ValueError, lambda: Z.split(3, 2))

    # 检查是否会引发值错误，验证等式是否成立
    raises(ValueError, lambda :P(Eq(X(t), 0), Contains(t, Interval.Lopen(1, 3)) & Eq(X(1), 0)))

    # 检查处理同时包含两个随机变量的查询
    res1 = P(Eq(N(3), N(5)))
    assert res1 == P(Eq(N(t), 0), Contains(t, Interval(3, 5)))

    # 检查处理同时包含两个随机变量的查询
    res2 = P(N(3) > N(1))
    assert res2 == P((N(t) > 0), Contains(t, Interval(1, 3)))

    # 检查条件不可能发生的情况
    assert P(N(3) < N(1)) == 0

    # 检查处理同时包含两个随机变量的查询
    res3 = P(N(3) <= N(1))
    assert res3 == P(Eq(N(t), 0), Contains(t, Interval(1, 3)))

    # 测试来自指定网址的例子
    X = PoissonProcess('X', 10) # 11.1
    assert P(Eq(X(S(1)/3), 3) & Eq(X(1), 10)) == exp(-10)*Rational(8000000000, 11160261)
    assert P(Eq(X(1), 1), Eq(X(S(1)/3), 3)) == 0
    assert P(Eq(X(1), 10), Eq(X(S(1)/3), 3)) == P(Eq(X(S(2)/3), 7))

    X = PoissonProcess('X', 2) # 11.2
    assert P(X(S(1)/2) < 1) == exp(-1)
    assert P(X(3) < 1, Eq(X(1), 0)) == exp(-4)
    assert P(Eq(X(4), 3), Eq(X(2), 3)) == exp(-4)

    X = PoissonProcess('X', 3)
    assert P(Eq(X(2), 5) & Eq(X(1), 2)) == Rational(81, 4)*exp(-6)

    # 检查几个属性
    assert P(X(2) <= 3, X(1)>=1) == 3*P(Eq(X(1), 0)) + 2*P(Eq(X(1), 1)) + P(Eq(X(1), 2))
    assert P(X(2) <= 3, X(1) > 1) == 2*P(Eq(X(1), 0)) + 1*P(Eq(X(1), 1))
    assert P(Eq(X(2), 5) & Eq(X(1), 2)) == P(Eq(X(1), 3))*P(Eq(X(1), 2))
    assert P(Eq(X(3), 4), Eq(X(1), 3)) == P(Eq(X(2), 1))

    # 测试问题编号 20078
    assert (2*X(t) + 3*X(t)).simplify() == 5*X(t)
    assert (2*X(t) - 3*X(t)).simplify() == -X(t)
    assert (2*(0.25*X(t))).simplify() == 0.5*X(t)
    assert (2*X(t) * 0.25*X(t)).simplify() == 0.5*X(t)**2
    assert (X(t)**2 + X(t)**3).simplify() == (X(t) + 1)*X(t)**2
# 定义测试函数 `test_WienerProcess`
def test_WienerProcess():
    # 创建一个 Wiener 过程对象 X，命名为 "X"
    X = WienerProcess("X")
    # 断言状态空间为实数集
    assert X.state_space == S.Reals
    # 断言索引集为半开区间 [0, oo)
    assert X.index_set == Interval(0, oo)

    # 声明符号变量 t, d, x, y，均为正数
    t, d, x, y = symbols('t d x y', positive=True)
    # 断言 X(t) 返回一个 RandomIndexedSymbol 对象
    assert isinstance(X(t), RandomIndexedSymbol)
    # 断言 X(t) 的分布为均值为 0，标准差为 sqrt(t) 的正态分布
    assert X.distribution(t) == NormalDistribution(0, sqrt(t))
    # 使用 warns_deprecated_sympy 上下文，断言 X.distribution(X(t)) 会产生弃用警告
    with warns_deprecated_sympy():
        X.distribution(X(t))
    # 断言尝试创建负参数的泊松过程会引发 ValueError 异常
    raises(ValueError, lambda: PoissonProcess("X", -1))
    # 断言尝试访问 X[t] 会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: X[t])
    # 断言尝试访问负索引 X(-2) 会引发 IndexError 异常
    raises(IndexError, lambda: X(-2))

    # 断言联合分布 X(2), X(3) 为手工制作的联合分布对象
    assert X.joint_distribution(X(2), X(3)) == JointDistributionHandmade(
        Lambda((X(2), X(3)), sqrt(6)*exp(-X(2)**2/4)*exp(-X(3)**2/6)/(12*pi)))
    # 断言联合分布 4, 6 为手工制作的联合分布对象
    assert X.joint_distribution(4, 6) == JointDistributionHandmade(
        Lambda((X(4), X(6)), sqrt(6)*exp(-X(4)**2/8)*exp(-X(6)**2/12)/(24*pi)))

    # 断言事件 P(X(t) < 3) 的简化结果为误差函数 erf 的计算结果
    assert P(X(t) < 3).simplify() == erf(3*sqrt(2)/(2*sqrt(t)))/2 + S(1)/2
    # 断言条件概率 P(X(t) > 2, Contains(t, Interval.Lopen(3, 7))) 的简化结果
    assert P(X(t) > 2, Contains(t, Interval.Lopen(3, 7))).simplify() == S(1)/2 - erf(sqrt(2)/2)/2

    # 断言事件 P((X(t) > 4) & (X(d) > 3) & (X(x) > 2) & (X(y) > 1), ...) 的简化结果
    assert P((X(t) > 4) & (X(d) > 3) & (X(x) > 2) & (X(y) > 1),
             Contains(t, Interval.Lopen(0, 1)) & Contains(d, Interval.Lopen(1, 2))
             & Contains(x, Interval.Lopen(2, 3)) & Contains(y, Interval.Lopen(3, 4))).simplify() == \
             (1 - erf(sqrt(2)/2))*(1 - erf(sqrt(2)))*(1 - erf(3*sqrt(2)/2))*(1 - erf(2*sqrt(2)))/16

    # 断言事件 P((X(t)< 2) & (X(d)> 3), Contains(t, Interval.Lopen(0, 2)) & ...) 的结果
    assert P((X(t)< 2) & (X(d)> 3), Contains(t, Interval.Lopen(0, 2))
             & Contains(d, Interval.Ropen(2, 4))) == Probability((X(d) > 3) & (X(t) < 2),
             Contains(d, Interval.Ropen(2, 4)) & Contains(t, Interval.Lopen(0, 2)))

    # 断言事件 P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) & ...) 的简化结果的字符串形式
    assert str(P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
                Contains(d, Interval.Lopen(7, 8))).simplify()) == \
                '-(1 - erf(3*sqrt(2)/2))*(2 - erfc(5/2))/4 + 1'
    # 断言期望 E(X(t)) 等于 0
    assert E(X(t)) == 0
    # 断言期望 E(x*(X(t) + X(d))*(X(t)**2+X(d)**2), ...) 等于期望 E(x*(X(d) + X(t))*(X(d)**2 + X(t)**2), ...)
    assert E(x*(X(t) + X(d))*(X(t)**2+X(d)**2), Contains(t, Interval.Lopen(0, 1))
             & Contains(d, Interval.Ropen(1, 2))) == Expectation(x*(X(d) + X(t))*(X(d)**2 + X(t)**2),
             Contains(d, Interval.Ropen(1, 2)) & Contains(t, Interval.Lopen(0, 1)))
    # 断言期望 E(X(t) + x*E(X(3))) 等于 0
    assert E(X(t) + x*E(X(3))) == 0

    # 测试问题 20078
    # 断言 (2*X(t) + 3*X(t)) 的简化结果等于 5*X(t)
    assert (2*X(t) + 3*X(t)).simplify() == 5*X(t)
    # 断言 (2*X(t) - 3*X(t)) 的简化结果等于 -X(t)
    assert (2*X(t) - 3*X(t)).simplify() == -X(t)
    # 断言 (2*(0.25*X(t))) 的简化结果等于 0.5*X(t)
    assert (2*(0.25*X(t))).simplify() == 0.5*X(t)
    # 断言 (2*X(t) * 0.25*X(t)) 的简化结果等于 0.5*X(t)**2
    assert (2*X(t) * 0.25*X(t)).simplify() == 0.5*X(t)**2


# 定义测试函数 `test_GammaProcess_symbolic`
def test_GammaProcess_symbolic():
    # 声明符号变量 t, d, x, y, g, l，均为正数
    t, d, x, y, g, l = symbols('t d x y g l', positive=True)
    # 创建一个 Gamma 过程对象 X，命名为 "X"，参数为 l 和 g
    X = GammaProcess("X", l, g)

    # 断言尝试访问 X[t] 会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: X[t])
    # 断言尝试访问负索引 X(-1) 会引发 IndexError 异常
    raises(IndexError, lambda: X(-1))
    # 断言 X(t) 返回一个 RandomIndexedSymbol 对象
    assert isinstance(X(t), RandomIndexedSymbol)
    # 断言状态空间为 [0, oo) 的区间
    assert X.state_space == Interval(0, oo)
    # 断言 X(t) 的分布为形状参数为 g*t，尺度参数为 1/l 的 Gamma 分布
    assert X.distribution(t) == GammaDistribution(g*t, 1/l)
    # 使用 warns_deprecated_sympy 上下文，断言 X.distribution(X(t)) 会产生弃用警告
    with warns_deprecated_sympy():
        X.distribution(X(t))
    # 使用自定义函数 `joint_distribution` 对象 X(3) 的联合分布，期望结果是由手工制作的 `JointDistributionHandmade` 对象给出的结果
    assert X.joint_distribution(5, X(3)) == JointDistributionHandmade(Lambda(
        (X(5), X(3)), l**(8*g)*exp(-l*X(3))*exp(-l*X(5))*X(3)**(3*g - 1)*X(5)**(5*g - 1)/(gamma(3*g)*gamma(5*g))))
    
    # 验证随机变量 X(t) 的期望是否等于 g*t/l
    assert E(X(t)) == g*t/l
    
    # 验证随机变量 X(t) 的方差是否简化后等于 g*t/l**2
    assert variance(X(t)).simplify() == g*t/l**2
    
    # 等同于 E(2*X(1)) + E(X(1)**2) + E(X(1)**3)，其中 E(X(1)) == g/l
    # 需要在时间间隔 (0, 1)、(1, 2) 和 (3, 4) 内对随机变量 X(t)**2 + X(d)*2 + X(y)**3 的期望进行计算
    assert E(X(t)**2 + X(d)*2 + X(y)**3, Contains(t, Interval.Lopen(0, 1))
        & Contains(d, Interval.Lopen(1, 2)) & Contains(y, Interval.Ropen(3, 4))) == \
        2*g/l + (g**2 + g)/l**2 + (g**3 + 3*g**2 + 2*g)/l**3
    
    # 验证随机变量 X(t) 在时间间隔 (3, 4) 内大于 3 的概率是否简化后等于 1 - lowergamma(g, 3*l)/gamma(g)
    assert P(X(t) > 3, Contains(t, Interval.Lopen(3, 4))).simplify() == \
        1 - lowergamma(g, 3*l)/gamma(g)
    
    # 测试问题编号 20078
    # 验证数学表达式是否简化后等于其期望结果
    assert (2*X(t) + 3*X(t)).simplify() == 5*X(t)
    assert (2*X(t) - 3*X(t)).simplify() == -X(t)
    assert (2*(0.25*X(t))).simplify() == 0.5*X(t)
    assert (2*X(t) * 0.25*X(t)).simplify() == 0.5*X(t)**2
    assert (X(t)**2 + X(t)**3).simplify() == (X(t) + 1)*X(t)**2
# 定义 Gamma 过程的数值测试函数
def test_GammaProcess_numeric():
    # 声明符号变量 t, d, x, y，均为正数
    t, d, x, y = symbols('t d x y', positive=True)
    # 创建 Gamma 过程 X，参数为 ("X", 1, 2)，lamda = 1, gamma = 2
    X = GammaProcess("X", 1, 2)
    # 断言状态空间为 [0, ∞)
    assert X.state_space == Interval(0, oo)
    # 断言索引集合为 [0, ∞)
    assert X.index_set == Interval(0, oo)
    # 断言 lamda 属性为 1
    assert X.lamda == 1
    # 断言 gamma 属性为 2
    assert X.gamma == 2

    # 期望抛出 ValueError 异常：lamda 不能为负数
    raises(ValueError, lambda: GammaProcess("X", -1, 2))
    # 期望抛出 ValueError 异常：gamma 不能为负数
    raises(ValueError, lambda: GammaProcess("X", 0, -2))
    # 期望抛出 ValueError 异常：lamda 和 gamma 都不能为负数
    raises(ValueError, lambda: GammaProcess("X", -1, -2))

    # 使用 Contains 对象进行事件的概率计算，并简化结果
    assert P((X(t) > 4) & (X(d) > 3) & (X(x) > 2) & (X(y) > 1), Contains(t,
        Interval.Lopen(0, 1)) & Contains(d, Interval.Lopen(1, 2)) & Contains(x,
        Interval.Lopen(2, 3)) & Contains(y, Interval.Lopen(3, 4))).simplify() == \
                                                            120*exp(-10)

    # 使用 Not 和 Or 进行事件的概率计算，并简化结果
    assert P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
        Contains(d, Interval.Lopen(7, 8))).simplify() == -4*exp(-3) + 472*exp(-8)/3 + 1
    assert P((X(t) > 2) | (X(t) < 4), Contains(t, Interval.Ropen(1, 4))).simplify() == \
                                            -643*exp(-4)/15 + 109*exp(-2)/15 + 1

    # 断言期望值 E(X(t)) 等于 2*t
    assert E(X(t)) == 2*t
    # 断言期望值 E(X(2) + x*E(X(5))) 等于 10*x + 4
    assert E(X(2) + x*E(X(5))) == 10*x + 4
```