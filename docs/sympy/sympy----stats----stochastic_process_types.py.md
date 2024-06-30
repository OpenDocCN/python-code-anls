# `D:\src\scipysrc\sympy\sympy\stats\stochastic_process_types.py`

```
# 导入随机模块
import random
# 导入 itertools 模块，用于高效循环和迭代操作
import itertools
# 导入 typing 模块中的类型别名，用于类型提示
from typing import (Sequence as tSequence, Union as tUnion, List as tList,
        Tuple as tTuple, Set as tSet)
# 导入 sympy 模块中的具体类和函数

# 导入具体的求和类 Sum
from sympy.concrete.summations import Sum
# 导入核心加法类 Add
from sympy.core.add import Add
# 导入核心基础类 Basic
from sympy.core.basic import Basic
# 导入缓存装饰器 cacheit
from sympy.core.cache import cacheit
# 导入核心容器类 Tuple
from sympy.core.containers import Tuple
# 导入表达式类 Expr
from sympy.core.expr import Expr
# 导入函数类 Function 和 Lambda
from sympy.core.function import (Function, Lambda)
# 导入乘法类 Mul
from sympy.core.mul import Mul
# 导入整数最大公约数函数 igcd
from sympy.core.intfunc import igcd
# 导入数值类 Integer, Rational, oo (无穷大), pi (圆周率)
from sympy.core.numbers import (Integer, Rational, oo, pi)
# 导入关系运算类 Eq (等于), Ge (大于等于), Gt (大于), Le (小于等于), Lt (小于), Ne (不等于)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
# 导入单例类 S
from sympy.core.singleton import S
# 导入符号类 Dummy 和 Symbol
from sympy.core.symbol import (Dummy, Symbol)
# 导入组合数学中的阶乘函数 factorial
from sympy.functions.combinatorial.factorials import factorial
# 导入指数函数 exp
from sympy.functions.elementary.exponential import exp
# 导入取天花板函数 ceiling
from sympy.functions.elementary.integers import ceiling
# 导入杂项函数 sqrt
from sympy.functions.elementary.miscellaneous import sqrt
# 导入分段函数 Piecewise
from sympy.functions.elementary.piecewise import Piecewise
# 导入伽玛函数 gamma
from sympy.functions.special.gamma_functions import gamma
# 导入布尔逻辑运算类 And, Not, Or
from sympy.logic.boolalg import (And, Not, Or)
# 导入矩阵异常类 NonSquareMatrixError
from sympy.matrices.exceptions import NonSquareMatrixError
# 导入密集矩阵类 Matrix, eye (单位矩阵), ones (全1矩阵), zeros (全0矩阵)
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
# 导入块矩阵类 BlockMatrix
from sympy.matrices.expressions.blockmatrix import BlockMatrix
# 导入矩阵符号类 MatrixSymbol
from sympy.matrices.expressions.matexpr import MatrixSymbol
# 导入特殊矩阵类 Identity
from sympy.matrices.expressions.special import Identity
# 导入不可变矩阵类 ImmutableMatrix
from sympy.matrices.immutable import ImmutableMatrix
# 导入条件集类 ConditionSet
from sympy.sets.conditionset import ConditionSet
# 导入包含关系类 Contains
from sympy.sets.contains import Contains
# 导入集合类 Range (范围)
from sympy.sets.fancysets import Range
# 导入集合类 FiniteSet (有限集合), Intersection (交集), Interval (区间), Set (集合), Union (并集)
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
# 导入线性方程组求解函数 linsolve
from sympy.solvers.solveset import linsolve
# 导入张量索引类 Indexed, IndexedBase
from sympy.tensor.indexed import (Indexed, IndexedBase)
# 导入关系运算类 Relational
from sympy.core.relational import Relational
# 导入布尔类型类 Boolean
from sympy.logic.boolalg import Boolean
# 导入 sympy 的警告模块
from sympy.utilities.exceptions import sympy_deprecation_warning
# 导入可迭代工具函数 strongly_connected_components
from sympy.utilities.iterables import strongly_connected_components
# 导入联合随机变量类 JointDistribution
from sympy.stats.joint_rv import JointDistribution
# 导入手动创建联合随机变量类 JointDistributionHandmade
from sympy.stats.joint_rv_types import JointDistributionHandmade
# 导入随机变量类 RandomIndexedSymbol, random_symbols, RandomSymbol,
# _symbol_converter, _value_check, pspace, given, dependent, is_random,
# sample_iter, Distribution, Density
from sympy.stats.rv import (RandomIndexedSymbol, random_symbols, RandomSymbol,
                            _symbol_converter, _value_check, pspace, given,
                            dependent, is_random, sample_iter, Distribution,
                            Density)
# 导入随机过程空间类 StochasticPSpace
from sympy.stats.stochastic_process import StochasticPSpace
# 导入符号概率类 Probability, Expectation
from sympy.stats.symbolic_probability import Probability, Expectation
# 导入伯努利分布类型类 Bernoulli, BernoulliDistribution, FiniteRV
from sympy.stats.frv_types import Bernoulli, BernoulliDistribution, FiniteRV
# 导入泊松分布类型类 Poisson, PoissonDistribution
from sympy.stats.drv_types import Poisson, PoissonDistribution
# 导入正态分布类型类 Normal, NormalDistribution, Gamma, GammaDistribution
from sympy.stats.crv_types import Normal, NormalDistribution, Gamma, GammaDistribution
# 导入 sympify 函数
from sympy.core.sympify import _sympify, sympify

# 定义 EmptySet 为 S.EmptySet
EmptySet = S.EmptySet

# 导出的全部模块名列表
__all__ = [
    'StochasticProcess',
    'DiscreteTimeStochasticProcess',
    'DiscreteMarkovChain',
    'TransitionMatrixOf',
    'StochasticStateSpaceOf',
    'GeneratorMatrixOf',
    'ContinuousMarkovChain',
    'BernoulliProcess',
    'PoissonProcess',  # 字符串 'PoissonProcess'，用作后续代码中的一个项
    'WienerProcess',   # 字符串 'WienerProcess'，用作后续代码中的一个项
    'GammaProcess'     # 字符串 'GammaProcess'，用作后续代码中的一个项
# 用于注册处理 Indexed 类型的函数 is_random
@is_random.register(Indexed)
def _(x):
    return is_random(x.base)

# 用于注册处理 RandomIndexedSymbol 类型的函数 is_random，并忽略类型检查
@is_random.register(RandomIndexedSymbol)  # type: ignore
def _(x):
    return True

# 辅助函数，用于将 list/tuple/set 转换为 Set
def _set_converter(itr):
    """
    Helper function for converting list/tuple/set to Set.
    If parameter is not an instance of list/tuple/set then
    no operation is performed.

    Returns
    =======

    Set
        The argument converted to Set.

    Raises
    ======

    TypeError
        If the argument is not an instance of list/tuple/set.
    """
    if isinstance(itr, (list, tuple, set)):
        itr = FiniteSet(*itr)
    if not isinstance(itr, Set):
        raise TypeError("%s is not an instance of list/tuple/set."%(itr))
    return itr

# 辅助函数，用于将 list/tuple/set/Range/Tuple/FiniteSet 转换为 tuple/Range
def _state_converter(itr: tSequence) -> tUnion[Tuple, Range]:
    """
    Helper function for converting list/tuple/set/Range/Tuple/FiniteSet
    to tuple/Range.
    """
    itr_ret: tUnion[Tuple, Range]

    if isinstance(itr, (Tuple, set, FiniteSet)):
        itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))

    elif isinstance(itr, (list, tuple)):
        # 检查状态是否唯一
        if len(set(itr)) != len(itr):
            raise ValueError('The state space must have unique elements.')
        itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))

    elif isinstance(itr, Range):
        # SymPy 中唯一的有序集合类型
        # 尝试转换为 tuple
        try:
            itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))
        except (TypeError, ValueError):
            itr_ret = itr

    else:
        raise TypeError("%s is not an instance of list/tuple/set/Range/Tuple/FiniteSet." % (itr))
    return itr_ret

# 辅助函数，将任意表达式转换为 SymPy 中可用的类型
def _sym_sympify(arg):
    """
    Converts an arbitrary expression to a type that can be used inside SymPy.
    As generally strings are unwise to use in the expressions,
    it returns the Symbol of argument if the string type argument is passed.

    Parameters
    ==========

    arg: The parameter to be converted to be used in SymPy.

    Returns
    =======

    The converted parameter.
    """
    if isinstance(arg, str):
        return Symbol(arg)
    else:
        return _sympify(arg)

# 辅助函数，用于检查 matrix 是否为 Matrix、MatrixSymbol 或 ImmutableMatrix 类型
def _matrix_checks(matrix):
    if not isinstance(matrix, (Matrix, MatrixSymbol, ImmutableMatrix)):
        raise TypeError("Transition probabilities either should "
                            "be a Matrix or a MatrixSymbol.")
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError("%s is not a square matrix"%(matrix))
    if isinstance(matrix, Matrix):
        matrix = ImmutableMatrix(matrix.tolist())
    return matrix

# 表示随机过程的基类，可以是离散或连续的
class StochasticProcess(Basic):
    """
    Base class for all the stochastic processes whether
    discrete or continuous.

    Parameters
    ==========

    sym: Symbol or str
    """
    # 状态空间的定义，通常为随机过程的可能取值范围，默认为实数集 S.Reals。
    # 对于离散集合，索引从零开始。
    state_space: Set
        The state space of the stochastic process, by default S.Reals.
        For discrete sets it is zero indexed.

    # 相关类
    # ========

    # 关联的离散时间随机过程类
    # DiscreteTimeStochasticProcess
    """

    # 索引集默认为实数集 S.Reals
    index_set = S.Reals

    # 初始化方法，创建一个新的随机索引符号对象
    def __new__(cls, sym, state_space=S.Reals, **kwargs):
        # 符号转换为内部格式
        sym = _symbol_converter(sym)
        # 状态空间转换为内部格式
        state_space = _set_converter(state_space)
        # 调用父类的构造方法创建对象
        return Basic.__new__(cls, sym, state_space)

    # 获取符号属性
    @property
    def symbol(self):
        return self.args[0]

    # 获取状态空间属性
    @property
    def state_space(self) -> tUnion[FiniteSet, Range]:
        # 如果状态空间不是有限集或范围，则将元组转换为有限集
        if not isinstance(self.args[1], (FiniteSet, Range)):
            assert isinstance(self.args[1], Tuple)
            return FiniteSet(*self.args[1])
        return self.args[1]

    # 发出关于分布方法的弃用警告
    def _deprecation_warn_distribution(self):
        sympy_deprecation_warning(
            """
            Calling the distribution method with a RandomIndexedSymbol
            argument, like X.distribution(X(t)) is deprecated. Instead, call
            distribution() with the given timestamp, like

            X.distribution(t)
            """,
            deprecated_since_version="1.7.1",
            active_deprecations_target="deprecated-distribution-randomindexedsymbol",
            stacklevel=4,
        )

    # 获取分布对象
    def distribution(self, key=None):
        # 如果没有指定关键字参数，则发出分布方法弃用警告
        if key is None:
            self._deprecation_warn_distribution()
        return Distribution()

    # 获取密度对象
    def density(self, x):
        return Density()

    # 调用方法，用于连续时间随机过程的索引
    def __call__(self, time):
        """
        Overridden in ContinuousTimeStochasticProcess.
        """
        # 抛出未实现错误，提示使用连续时间随机过程的索引方式
        raise NotImplementedError("Use [] for indexing discrete time stochastic process.")

    # 获取项目方法，用于离散时间随机过程的索引
    def __getitem__(self, time):
        """
        Overridden in DiscreteTimeStochasticProcess.
        """
        # 抛出未实现错误，提示使用离散时间随机过程的索引方式
        raise NotImplementedError("Use () for indexing continuous time stochastic process.")

    # 计算概率方法，抛出未实现错误
    def probability(self, condition):
        raise NotImplementedError()
    def joint_distribution(self, *args):
        """
        Computes the joint distribution of the random indexed variables.

        Parameters
        ==========

        args: iterable
            The finite list of random indexed variables/the key of a stochastic
            process whose joint distribution has to be computed.

        Returns
        =======

        JointDistribution
            The joint distribution of the list of random indexed variables.
            An unevaluated object is returned if it is not possible to
            compute the joint distribution.

        Raises
        ======

        ValueError: When the arguments passed are not of type RandomIndexSymbol
        or Number.
        """
        # Convert args to a mutable list
        args = list(args)
        
        # Iterate through each argument
        for i, arg in enumerate(args):
            # Check if the argument is a number
            if S(arg).is_Number:
                # If the index set is integers, retrieve the corresponding value
                if self.index_set.is_subset(S.Integers):
                    args[i] = self.__getitem__(arg)
                else:
                    # Otherwise, call the stochastic process with the argument
                    args[i] = self.__call__(arg)
            elif not isinstance(arg, RandomIndexedSymbol):
                # Raise an error if the argument is not a RandomIndexedSymbol
                raise ValueError("Expected a RandomIndexedSymbol or "
                                "key not  %s"%(type(arg)))

        # Check if the distribution of the first argument is empty
        if args[0].pspace.distribution == Distribution():
            return JointDistribution(*args)
        
        # Create a density function using a Lambda expression
        density = Lambda(tuple(args),
                         expr=Mul.fromiter(arg.pspace.process.density(arg) for arg in args))
        
        # Return a handmade JointDistribution using the computed density
        return JointDistributionHandmade(density)

    def expectation(self, condition, given_condition):
        raise NotImplementedError("Abstract method for expectation queries.")

    def sample(self):
        raise NotImplementedError("Abstract method for sampling queries.")


注释完成。这段代码定义了一个 `joint_distribution` 方法，用于计算随机索引变量的联合分布。它包含了参数说明、返回值说明和可能引发的异常。
class DiscreteTimeStochasticProcess(StochasticProcess):
    """
    Base class for all discrete stochastic processes.
    """

    def __getitem__(self, time):
        """
        For indexing discrete time stochastic processes.

        Returns
        =======
        
        RandomIndexedSymbol
            A symbol randomly indexed based on the provided time.
        """
        # 将时间符号化
        time = sympify(time)
        # 如果时间不是符号并且不在索引集合中，则引发索引错误
        if not time.is_symbol and time not in self.index_set:
            raise IndexError("%s is not in the index set of %s"%(time, self.symbol))
        # 创建索引对象
        idx_obj = Indexed(self.symbol, time)
        # 创建随机符号空间对象
        pspace_obj = StochasticPSpace(self.symbol, self, self.distribution(time))
        # 返回随机索引符号对象
        return RandomIndexedSymbol(idx_obj, pspace_obj)


class ContinuousTimeStochasticProcess(StochasticProcess):
    """
    Base class for all continuous time stochastic process.
    """

    def __call__(self, time):
        """
        For indexing continuous time stochastic processes.

        Returns
        =======
        
        RandomIndexedSymbol
            A symbol randomly indexed based on the provided time.
        """
        # 将时间符号化
        time = sympify(time)
        # 如果时间不是符号并且不在索引集合中，则引发索引错误
        if not time.is_symbol and time not in self.index_set:
            raise IndexError("%s is not in the index set of %s"%(time, self.symbol))
        # 创建函数对象
        func_obj = Function(self.symbol)(time)
        # 创建随机符号空间对象
        pspace_obj = StochasticPSpace(self.symbol, self, self.distribution(time))
        # 返回随机索引符号对象
        return RandomIndexedSymbol(func_obj, pspace_obj)


class TransitionMatrixOf(Boolean):
    """
    Assumes that the matrix is the transition matrix
    of the process.
    """

    def __new__(cls, process, matrix):
        """
        Creates a new instance of TransitionMatrixOf.

        Parameters
        ==========

        process : DiscreteMarkovChain
            The discrete Markov chain process.
        matrix : Matrix
            The matrix representing transition probabilities.

        Returns
        =======
        
        TransitionMatrixOf
            Instance of TransitionMatrixOf initialized with process and matrix.
        """
        # 如果进程不是离散马尔可夫链，则引发值错误
        if not isinstance(process, DiscreteMarkovChain):
            raise ValueError("Currently only DiscreteMarkovChain "
                                "support TransitionMatrixOf.")
        # 检查并处理矩阵
        matrix = _matrix_checks(matrix)
        # 返回基础类的新实例
        return Basic.__new__(cls, process, matrix)

    process = property(lambda self: self.args[0])
    matrix = property(lambda self: self.args[1])


class GeneratorMatrixOf(TransitionMatrixOf):
    """
    Assumes that the matrix is the generator matrix
    of the process.
    """

    def __new__(cls, process, matrix):
        """
        Creates a new instance of GeneratorMatrixOf.

        Parameters
        ==========

        process : ContinuousMarkovChain
            The continuous Markov chain process.
        matrix : Matrix
            The matrix representing generator rates.

        Returns
        =======
        
        GeneratorMatrixOf
            Instance of GeneratorMatrixOf initialized with process and matrix.
        """
        # 如果进程不是连续马尔可夫链，则引发值错误
        if not isinstance(process, ContinuousMarkovChain):
            raise ValueError("Currently only ContinuousMarkovChain "
                                "support GeneratorMatrixOf.")
        # 检查并处理矩阵
        matrix = _matrix_checks(matrix)
        # 返回基础类的新实例
        return Basic.__new__(cls, process, matrix)


class StochasticStateSpaceOf(Boolean):

    def __new__(cls, process, state_space):
        """
        Creates a new instance of StochasticStateSpaceOf.

        Parameters
        ==========

        process : DiscreteMarkovChain or ContinuousMarkovChain
            The discrete or continuous Markov chain process.
        state_space : Set or Range
            The set or range representing the state space.

        Returns
        =======
        
        StochasticStateSpaceOf
            Instance of StochasticStateSpaceOf initialized with process and state_space.
        """
        # 如果进程不是离散或连续马尔可夫链，则引发值错误
        if not isinstance(process, (DiscreteMarkovChain, ContinuousMarkovChain)):
            raise ValueError("Currently only DiscreteMarkovChain and ContinuousMarkovChain "
                                "support StochasticStateSpaceOf.")
        # 转换状态空间为适当的格式
        state_space = _state_converter(state_space)
        # 计算状态空间的大小
        if isinstance(state_space, Range):
            ss_size = ceiling((state_space.stop - state_space.start) / state_space.step)
        else:
            ss_size = len(state_space)
        # 创建状态索引对象
        state_index = Range(ss_size)
        # 返回基础类的新实例
        return Basic.__new__(cls, process, state_index)
    # 创建一个名为 process 的 property，返回 self.args[0] 的值
    process = property(lambda self: self.args[0])
    # 创建一个名为 state_index 的 property，返回 self.args[1] 的值
    state_index = property(lambda self: self.args[1])
class MarkovProcess(StochasticProcess):
    """
    Contains methods that handle queries
    common to Markov processes.
    """

    @property
    def number_of_states(self) -> tUnion[Integer, Symbol]:
        """
        The number of states in the Markov Chain.
        """
        # 返回 Markov 链中的状态数量，使用第三个参数的形状的第一个维度
        return _sympify(self.args[2].shape[0])  # type: ignore

    @property
    def _state_index(self):
        """
        Returns state index as Range.
        """
        # 返回状态索引作为 Range 对象
        return self.args[1]

    @classmethod
    def _sanity_checks(cls, state_space, trans_probs):
        # 尽量避免 state_space 或 trans_probs 为 None。
        # 这在开始时处理将有很大帮助。

        if (state_space is None) and (trans_probs is None):
            # 如果 state_space 和 trans_probs 都是 None，则创建一个整数的 Dummy 变量 _n
            _n = Dummy('n', integer=True, nonnegative=True)
            # 将 state_space 转换为 Range 对象
            state_space = _state_converter(Range(_n))
            # 对 trans_probs 执行矩阵检查
            trans_probs = _matrix_checks(MatrixSymbol('_T', _n, _n))

        elif state_space is None:
            # 如果 state_space 是 None，则对 trans_probs 执行矩阵检查
            trans_probs = _matrix_checks(trans_probs)
            # 将 trans_probs 的形状的第一个维度作为 state_space
            state_space = _state_converter(Range(trans_probs.shape[0]))

        elif trans_probs is None:
            # 如果 trans_probs 是 None，则将 state_space 转换为合适的格式
            state_space = _state_converter(state_space)
            # 如果 state_space 是 Range 对象，则计算其大小
            if isinstance(state_space, Range):
                _n = ceiling((state_space.stop - state_space.start) / state_space.step)
            else:
                _n = len(state_space)
            # 创建一个 _n x _n 的符号矩阵作为 trans_probs
            trans_probs = MatrixSymbol('_T', _n, _n)

        else:
            # 如果 state_space 和 trans_probs 都不为 None，则分别进行转换和检查
            state_space = _state_converter(state_space)
            trans_probs = _matrix_checks(trans_probs)
            # 如果 state_space 是 Range 对象，则计算其大小
            if isinstance(state_space, Range):
                ss_size = ceiling((state_space.stop - state_space.start) / state_space.step)
            else:
                ss_size = len(state_space)
            # 检查 state_space 和 trans_probs 的行数是否一致
            if ss_size != trans_probs.shape[0]:
                raise ValueError('The size of the state space and the number of '
                                 'rows of the transition matrix must be the same.')

        # 返回经过检查和转换后的 state_space 和 trans_probs
        return state_space, trans_probs
    def _extract_information(self, given_condition):
        """
        Helper function to extract information, like,
        transition matrix/generator matrix, state space, etc.
        """
        # 检查当前对象是否为离散马尔可夫链，获取相应的转移概率和状态索引
        if isinstance(self, DiscreteMarkovChain):
            trans_probs = self.transition_probabilities
            state_index = self._state_index
        # 检查当前对象是否为连续马尔可夫链，获取相应的生成矩阵和状态索引
        elif isinstance(self, ContinuousMarkovChain):
            trans_probs = self.generator_matrix
            state_index = self._state_index

        # 如果给定条件是逻辑与的实例，解析具体条件
        if isinstance(given_condition, And):
            gcs = given_condition.args
            given_condition = S.true
            for gc in gcs:
                # 如果条件是转移矩阵的实例，则更新转移概率
                if isinstance(gc, TransitionMatrixOf):
                    trans_probs = gc.matrix
                # 如果条件是随机状态空间的实例，则更新状态索引
                if isinstance(gc, StochasticStateSpaceOf):
                    state_index = gc.state_index
                # 如果条件是关系型条件，则与之前的条件进行逻辑与操作
                if isinstance(gc, Relational):
                    given_condition = given_condition & gc

        # 如果给定条件是转移矩阵的实例，更新转移概率并重置条件为真
        if isinstance(given_condition, TransitionMatrixOf):
            trans_probs = given_condition.matrix
            given_condition = S.true

        # 如果给定条件是随机状态空间的实例，更新状态索引并重置条件为真
        if isinstance(given_condition, StochasticStateSpaceOf):
            state_index = given_condition.state_index
            given_condition = S.true

        # 返回提取到的转移概率、状态索引以及处理后的条件
        return trans_probs, state_index, given_condition

    def _check_trans_probs(self, trans_probs, row_sum=1):
        """
        Helper function for checking the validity of transition
        probabilities.
        """
        # 如果转移概率不是矩阵符号，则将其转换为列表形式的行，并检查各行的总和是否等于指定值
        if not isinstance(trans_probs, MatrixSymbol):
            rows = trans_probs.tolist()
            for row in rows:
                if (sum(row) - row_sum) != 0:
                    raise ValueError("Values in a row must sum to %s. "
                                     "If you are using Float or floats then please use Rational."%(row_sum))

    def _work_out_state_index(self, state_index, given_condition, trans_probs):
        """
        Helper function to extract state space if there
        is a random symbol in the given condition.
        """
        # 如果给定条件不是 None，则检查其中是否包含随机符号，如果只有一个随机符号，则更新状态索引为其可能空间
        if given_condition is not None:
            rand_var = list(given_condition.atoms(RandomSymbol) -
                            given_condition.atoms(RandomIndexedSymbol))
            if len(rand_var) == 1:
                state_index = rand_var[0].pspace.set

        # 检查状态空间的符号性质，如果不是整数并且状态索引长度与转移概率行数不一致，则抛出错误
        sym_cond = not self.number_of_states.is_Integer
        cond1 = not sym_cond and len(state_index) != trans_probs.shape[0]
        if cond1:
            raise ValueError("state space is not compatible with the transition probabilities.")

        # 如果转移概率行数不是符号类型，则将状态索引设置为一个有限集合，范围为 0 到转移概率行数的上界
        if not isinstance(trans_probs.shape[0], Symbol):
            state_index = FiniteSet(*range(trans_probs.shape[0]))

        # 返回计算后的状态索引
        return state_index

    @cacheit
    # 定义一个辅助函数，用于预处理信息
    def _preprocess(self, given_condition, evaluate):
        """
        Helper function for pre-processing the information.
        """
        # 初始化是否信息不足的标志
        is_insufficient = False

        # 如果不需要评估结果，则避免进行预处理，直接返回预定义的值
        if not evaluate:
            return (True, None, None, None)

        # 提取转移矩阵和状态空间
        trans_probs, state_index, given_condition = self._extract_information(given_condition)

        # 如果提取的转移概率或给定条件为空，则认为信息不足
        if trans_probs is None or given_condition is None:
            is_insufficient = True
        else:
            # 根据对象类型检查转移概率
            if isinstance(self, DiscreteMarkovChain):
                self._check_trans_probs(trans_probs, row_sum=1)
            elif isinstance(self, ContinuousMarkovChain):
                self._check_trans_probs(trans_probs, row_sum=0)

            # 计算状态空间
            state_index = self._work_out_state_index(state_index, given_condition, trans_probs)

        # 返回处理结果：信息是否不足、转移概率、状态空间、给定条件
        return is_insufficient, trans_probs, state_index, given_condition

    # 将条件对象中的符号替换为索引值
    def replace_with_index(self, condition):
        if isinstance(condition, Relational):
            lhs, rhs = condition.lhs, condition.rhs
            # 如果左侧不是随机索引符号，则交换左右操作数
            if not isinstance(lhs, RandomIndexedSymbol):
                lhs, rhs = rhs, lhs
            # 使用符号的索引值替换条件对象中的符号
            condition = type(condition)(self.index_of.get(lhs, lhs),
                                        self.index_of.get(rhs, rhs))
        return condition
    def _symbolic_probability(self, condition, new_given_condition, rv, min_key_rv):
        # Function to calculate probability for queries with symbols

        # Check if condition is an instance of Relational
        if isinstance(condition, Relational):
            # Determine current state based on the structure of new_given_condition and condition
            curr_state = new_given_condition.rhs if isinstance(new_given_condition.lhs, RandomIndexedSymbol) \
                else new_given_condition.lhs
            next_state = condition.rhs if isinstance(condition.lhs, RandomIndexedSymbol) \
                else condition.lhs

            # Handle equality and inequality conditions differently
            if isinstance(condition, (Eq, Ne)):
                # Calculate probability based on the type of Markov Chain
                if isinstance(self, DiscreteMarkovChain):
                    P = self.transition_probabilities ** (rv[0].key - min_key_rv.key)
                else:
                    P = exp(self.generator_matrix * (rv[0].key - min_key_rv.key))
                
                # Determine the probability value based on the condition type (Eq or Ne)
                prob = P[curr_state, next_state] if isinstance(condition, Eq) else 1 - P[curr_state, next_state]
                
                # Return a Piecewise function representing the calculated probability
                return Piecewise((prob, rv[0].key > min_key_rv.key), (Probability(condition), True))
            else:
                # Handle conditions Ge, Lt, Gt, Le
                upper = 1
                greater = False
                if isinstance(condition, (Ge, Lt)):
                    upper = 0
                if isinstance(condition, (Ge, Gt)):
                    greater = True
                
                # Create a Dummy variable k and redefine condition based on RandomIndexedSymbol
                k = Dummy('k')
                condition = Eq(condition.lhs, k) if isinstance(condition.lhs, RandomIndexedSymbol) \
                    else Eq(condition.rhs, k)
                
                # Calculate total probability using Sum over state space
                total = Sum(self.probability(condition, new_given_condition), (k, next_state + upper, self.state_space._sup))
                
                # Return a Piecewise function considering greater than or less than conditions
                return Piecewise((total, rv[0].key > min_key_rv.key), (Probability(condition), True)) if greater \
                    else Piecewise((1 - total, rv[0].key > min_key_rv.key), (Probability(condition), True))
        else:
            # Return probability if condition is not Relational
            return Probability(condition, new_given_condition)
    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Handles expectation queries for markov process.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation
            Unevaluated object if computations cannot be done due to
            insufficient information.
        Expr
            In all other cases when the computations are successful.

        Note
        ====

        Any information passed at the time of query overrides
        any information passed at the time of object creation like
        transition probabilities, state space.

        Pass the transition matrix using TransitionMatrixOf,
        generator matrix using GeneratorMatrixOf and state space
        using StochasticStateSpaceOf in given_condition using & or And.
        """

        # 预处理条件和状态索引
        check, mat, state_index, condition = \
            self._preprocess(condition, evaluate)

        # 如果预处理成功，返回一个期望对象
        if check:
            return Expectation(expr, condition)

        # 提取表达式中的随机符号
        rvs = random_symbols(expr)

        # 处理特定情况：表达式为表达式类型且条件为等式且仅包含一个随机符号
        if isinstance(expr, Expr) and isinstance(condition, Eq) \
            and len(rvs) == 1:

            # 替换条件和状态索引中的索引
            condition = self.replace_with_index(condition)
            state_index = self.replace_with_index(state_index)

            # 获取随机符号
            rv = list(rvs)[0]

            # 获取条件的左右部分
            lhsg, rhsg = condition.lhs, condition.rhs

            # 确保左部是随机索引符号
            if not isinstance(lhsg, RandomIndexedSymbol):
                lhsg, rhsg = (rhsg, lhsg)

            # 检查条件中的状态是否在状态空间中
            if rhsg not in state_index:
                raise ValueError("%s state is not in the state space."%(rhsg))

            # 检查随机符号的索引是否正确
            if rv.key < lhsg.key:
                raise ValueError("Incorrect given condition is given, expectation "
                    "time %s < time %s"%(rv.key, rv.key))

            # 根据对象类型创建状态转移矩阵或生成器矩阵
            mat_of = TransitionMatrixOf(self, mat) if isinstance(self, DiscreteMarkovChain) else GeneratorMatrixOf(self, mat)

            # 构建新的条件
            cond = condition & mat_of & \
                    StochasticStateSpaceOf(self, state_index)

            # 定义求和函数，计算期望
            func = lambda s: self.probability(Eq(rv, s), cond) * expr.subs(rv, self._state_index[s])
            return sum(func(s) for s in state_index)

        # 抛出未实现错误，因为尚未处理给定的查询类型
        raise NotImplementedError("Mechanism for handling (%s, %s) queries hasn't been "
                                "implemented yet."%(expr, condition))
# 定义一个类 DiscreteMarkovChain，继承自 DiscreteTimeStochasticProcess 和 MarkovProcess
class DiscreteMarkovChain(DiscreteTimeStochasticProcess, MarkovProcess):
    """
    表示一个有限离散时间同质马尔可夫链。

    这种类型的马尔可夫链可以通过其（有序的）状态空间和其一步转移概率矩阵来唯一确定。

    Parameters
    ==========

    sym:
        给定的马尔可夫链的名称
    state_space:
        可选的，默认为 Range(n)
    trans_probs:
        可选的，默认为 MatrixSymbol('_T', n, n)

    Examples
    ========

    >>> from sympy.stats import DiscreteMarkovChain, TransitionMatrixOf, P, E
    >>> from sympy import Matrix, MatrixSymbol, Eq, symbols
    >>> T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> YS = DiscreteMarkovChain("Y")

    >>> Y.state_space
    {0, 1, 2}
    >>> Y.transition_probabilities
    Matrix([
    [0.5, 0.2, 0.3],
    [0.2, 0.5, 0.3],
    [0.2, 0.3, 0.5]])
    >>> TS = MatrixSymbol('T', 3, 3)
    >>> P(Eq(YS[3], 2), Eq(YS[1], 1) & TransitionMatrixOf(YS, TS))
    T[0, 2]*T[1, 0] + T[1, 1]*T[1, 2] + T[1, 2]*T[2, 2]
    >>> P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2)
    0.36

    Probabilities will be calculated based on indexes rather
    than state names. For example, with the Sunny-Cloudy-Rainy
    model with string state names:

    >>> from sympy.core.symbol import Str
    >>> Y = DiscreteMarkovChain("Y", [Str('Sunny'), Str('Cloudy'), Str('Rainy')], T)
    >>> P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2)
    0.36

    This gives the same answer as the ``[0, 1, 2]`` state space.
    Currently, there is no support for state names within probability
    and expectation statements. Here is a work-around using ``Str``:

    >>> P(Eq(Str('Rainy'), Y[3]), Eq(Y[1], Str('Cloudy'))).round(2)
    0.36

    Symbol state names can also be used:

    >>> sunny, cloudy, rainy = symbols('Sunny, Cloudy, Rainy')
    >>> Y = DiscreteMarkovChain("Y", [sunny, cloudy, rainy], T)
    >>> P(Eq(Y[3], rainy), Eq(Y[1], cloudy)).round(2)
    0.36

    Expectations will be calculated as follows:

    >>> E(Y[3], Eq(Y[1], cloudy))
    0.38*Cloudy + 0.36*Rainy + 0.26*Sunny

    Probability of expressions with multiple RandomIndexedSymbols
    can also be calculated provided there is only 1 RandomIndexedSymbol
    in the given condition. It is always better to use Rational instead
    of floating point numbers for the probabilities in the
    transition matrix to avoid errors.

    >>> from sympy import Gt, Le, Rational
    >>> T = Matrix([[Rational(5, 10), Rational(3, 10), Rational(2, 10)], [Rational(2, 10), Rational(7, 10), Rational(1, 10)], [Rational(3, 10), Rational(3, 10), Rational(4, 10)]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> P(Eq(Y[3], Y[1]), Eq(Y[0], 0)).round(3)
    0.409
    >>> P(Gt(Y[3], Y[1]), Eq(Y[0], 0)).round(2)
    0.36
    >>> P(Le(Y[15], Y[10]), Eq(Y[8], 2)).round(7)
    0.6963328

    Symbolic probability queries are also supported
    """
    >>> a, b, c, d = symbols('a b c d')
    >>> T = Matrix([[Rational(1, 10), Rational(4, 10), Rational(5, 10)], [Rational(3, 10), Rational(4, 10), Rational(3, 10)], [Rational(7, 10), Rational(2, 10), Rational(1, 10)]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> query = P(Eq(Y[a], b), Eq(Y[c], d))
    >>> query.subs({a:10, b:2, c:5, d:1}).round(4)
    0.3096
    >>> P(Eq(Y[10], 2), Eq(Y[5], 1)).evalf().round(4)
    0.3096
    >>> query_gt = P(Gt(Y[a], b), Eq(Y[c], d))
    >>> query_gt.subs({a:21, b:0, c:5, d:0}).evalf().round(5)
    0.64705
    >>> P(Gt(Y[21], 0), Eq(Y[5], 0)).round(5)
    0.64705
    
    There is limited support for arbitrarily sized states:
    
    >>> n = symbols('n', nonnegative=True, integer=True)
    >>> T = MatrixSymbol('T', n, n)
    >>> Y = DiscreteMarkovChain("Y", trans_probs=T)
    >>> Y.state_space
    Range(0, n, 1)
    >>> query = P(Eq(Y[a], b), Eq(Y[c], d))
    >>> query.subs({a:10, b:2, c:5, d:1})
    (T**5)[1, 2]
    
    References
    ==========
    
    .. [1] https://en.wikipedia.org/wiki/Markov_chain#Discrete-time_Markov_chain
    .. [2] https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    """
    index_set = S.Naturals0
    
    # 定义一个新的类，表示离散马尔可夫链
    def __new__(cls, sym, state_space=None, trans_probs=None):
        sym = _symbol_converter(sym)
    
        # 检查并设置状态空间和过渡概率
        state_space, trans_probs = MarkovProcess._sanity_checks(state_space, trans_probs)
    
        # 创建新的对象实例
        obj = Basic.__new__(cls, sym, state_space, trans_probs) # type: ignore
        indices = {}
        if isinstance(obj.number_of_states, Integer):
            # 如果状态数是整数，为每个状态设置索引
            for index, state in enumerate(obj._state_index):
                indices[state] = index
        obj.index_of = indices
        return obj
    
    @property
    # 返回过渡概率的属性
    def transition_probabilities(self):
        """
        Transition probabilities of discrete Markov chain,
        either an instance of Matrix or MatrixSymbol.
        """
        return self.args[2]
    
    def fundamental_matrix(self):
        """
        Each entry fundamental matrix can be interpreted as
        the expected number of times the chains is in state j
        if it started in state i.
    
        References
        ==========
    
        .. [1] https://lips.cs.princeton.edu/the-fundamental-matrix-of-a-finite-markov-chain/
    
        """
        _, _, _, Q = self.decompose()
    
        if Q.shape[0] > 0:  # if non-ergodic
            I = eye(Q.shape[0])
            if (I - Q).det() == 0:
                raise ValueError("The fundamental matrix doesn't exist.")
            return (I - Q).inv().as_immutable()
        else:  # if ergodic
            P = self.transition_probabilities
            I = eye(P.shape[0])
            w = self.fixed_row_vector()
            W = Matrix([list(w) for i in range(0, P.shape[0])])
            if (I - P + W).det() == 0:
                raise ValueError("The fundamental matrix doesn't exist.")
            return (I - P + W).inv().as_immutable()
    def absorbing_probabilities(self):
        """
        Computes the absorbing probabilities, i.e.
        the ij-th entry of the matrix denotes the
        probability of Markov chain being absorbed
        in state j starting from state i.
        """
        # Decompose the Markov chain into its components
        _, _, R, _ = self.decompose()
        # Compute the fundamental matrix N
        N = self.fundamental_matrix()
        # If either R or N is None, return None
        if R is None or N is None:
            return None
        # Compute and return the absorbing probabilities matrix
        return N * R

    def absorbing_probabilites(self):
        sympy_deprecation_warning(
            """
            DiscreteMarkovChain.absorbing_probabilites() is deprecated. Use
            absorbing_probabilities() instead (note the spelling difference).
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-absorbing_probabilites",
        )
        # Call the corrected method absorbing_probabilities()
        return self.absorbing_probabilities()

    def is_regular(self):
        # Get communication classes of the Markov chain
        tuples = self.communication_classes()
        # If there are no communication classes, return S.false (false symbol)
        if len(tuples) == 0:
            return S.false  # not defined for a 0x0 matrix
        # Unpack classes, _, periods from tuples
        classes, _, periods = list(zip(*tuples))
        # Return true if there is exactly one class and its period is 1
        return And(len(classes) == 1, periods[0] == 1)

    def is_ergodic(self):
        # Get communication classes of the Markov chain
        tuples = self.communication_classes()
        # If there are no communication classes, return S.false (false symbol)
        if len(tuples) == 0:
            return S.false  # not defined for a 0x0 matrix
        # Unpack classes, _, _ from tuples
        classes, _, _ = list(zip(*tuples))
        # Return true if there is exactly one communication class
        return S(len(classes) == 1)

    def is_absorbing_state(self, state):
        # Get transition probabilities matrix
        trans_probs = self.transition_probabilities
        # Check if trans_probs is an instance of ImmutableMatrix and state is within its range
        if isinstance(trans_probs, ImmutableMatrix) and \
            state < trans_probs.shape[0]:
            # Return true if the diagonal element at (state, state) is 1
            return S(trans_probs[state, state]) is S.One

    def is_absorbing_chain(self):
        # Decompose the Markov chain into its components
        states, A, B, C = self.decompose()
        r = A.shape[0]
        # Return true if the number of states r is greater than 0 and A is an identity matrix of size r
        return And(r > 0, A == Identity(r).as_explicit())

    def fixed_row_vector(self):
        """
        A wrapper for ``stationary_distribution()``.
        """
        # Call stationary_distribution() method to get the fixed row vector
        return self.stationary_distribution()

    @property
    def limiting_distribution(self):
        """
        The fixed row vector is the limiting
        distribution of a discrete Markov chain.
        """
        # The limiting distribution is equivalent to the fixed row vector
        return self.fixed_row_vector()

    def sample(self):
        """
        Returns
        =======

        sample: iterator object
            iterator object containing the sample

        """
        # Check if transition_probabilities is a valid matrix type
        if not isinstance(self.transition_probabilities, (Matrix, ImmutableMatrix)):
            raise ValueError("Transition Matrix must be provided for sampling")
        # Convert transition_probabilities to list
        Tlist = self.transition_probabilities.tolist()
        # Initialize samples list with a random initial state
        samps = [random.choice(list(self.state_space))]
        # Yield the first sample
        yield samps[0]
        # Initialize time counter
        time = 1
        # Initialize densities dictionary
        densities = {}
        # Iterate over each state in state_space
        for state in self.state_space:
            states = list(self.state_space)
            # Populate densities dictionary with transition probabilities
            densities[state] = {states[i]: Tlist[state][i]
                        for i in range(len(states))}
        # Infinite loop to generate samples
        while time < S.Infinity:
            # Generate next sample based on current state's density
            samps.append((next(sample_iter(FiniteRV("_", densities[samps[time - 1]])))))
            # Yield the current sample
            yield samps[time]
            # Increment time counter
            time += 1
class ContinuousMarkovChain(ContinuousTimeStochasticProcess, MarkovProcess):
    """
    Represents continuous time Markov chain.

    Parameters
    ==========

    sym : Symbol/str
        符号或字符串，表示这个连续时间马尔可夫链的名称
    state_space : Set
        可选参数，默认为 S.Reals，状态空间的集合
    gen_mat : Matrix/ImmutableMatrix/MatrixSymbol
        可选参数，默认为 None，生成矩阵，描述状态转移概率

    Examples
    ========

    >>> from sympy.stats import ContinuousMarkovChain, P
    >>> from sympy import Matrix, S, Eq, Gt
    >>> G = Matrix([[-S(1), S(1)], [S(1), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1], gen_mat=G)
    >>> C.limiting_distribution()
    Matrix([[1/2, 1/2]])
    >>> C.state_space
    {0, 1}
    >>> C.generator_matrix
    Matrix([
    [-1,  1],
    [ 1, -1]])

    Probability queries are supported

    >>> P(Eq(C(1.96), 0), Eq(C(0.78), 1)).round(5)
    0.45279
    >>> P(Gt(C(1.7), 0), Eq(C(0.82), 1)).round(5)
    0.58602

    Probability of expressions with multiple RandomIndexedSymbols
    can also be calculated provided there is only 1 RandomIndexedSymbol
    in the given condition. It is always better to use Rational instead
    of floating point numbers for the probabilities in the
    generator matrix to avoid errors.

    >>> from sympy import Gt, Le, Rational
    >>> G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    >>> P(Eq(C(3.92), C(1.75)), Eq(C(0.46), 0)).round(5)
    0.37933
    >>> P(Gt(C(3.92), C(1.75)), Eq(C(0.46), 0)).round(5)
    0.34211
    >>> P(Le(C(1.57), C(3.14)), Eq(C(1.22), 1)).round(4)
    0.7143

    Symbolic probability queries are also supported

    >>> from sympy import symbols
    >>> a,b,c,d = symbols('a b c d')
    >>> G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    >>> query = P(Eq(C(a), b), Eq(C(c), d))
    >>> query.subs({a:3.65, b:2, c:1.78, d:1}).evalf().round(10)
    0.4002723175
    >>> P(Eq(C(3.65), 2), Eq(C(1.78), 1)).round(10)
    0.4002723175
    >>> query_gt = P(Gt(C(a), b), Eq(C(c), d))
    >>> query_gt.subs({a:43.2, b:0, c:3.29, d:2}).evalf().round(10)
    0.6832579186
    >>> P(Gt(C(43.2), 0), Eq(C(3.29), 2)).round(10)
    0.6832579186

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Markov_chain#Continuous-time_Markov_chain
    .. [2] https://u.math.biu.ac.il/~amirgi/CTMCnotes.pdf
    """
    index_set = S.Reals  # 状态空间的索引集合为实数集合
    def __new__(cls, sym, state_space=None, gen_mat=None):
        sym = _symbol_converter(sym)
        # 调用 MarkovProcess 类的 _sanity_checks 方法进行状态空间和生成矩阵的合法性检查
        state_space, gen_mat = MarkovProcess._sanity_checks(state_space, gen_mat)
        # 创建一个新的实例对象
        obj = Basic.__new__(cls, sym, state_space, gen_mat)
        indices = {}
        # 如果状态数是整数型，则为状态空间中的每个状态分配一个索引
        if isinstance(obj.number_of_states, Integer):
            for index, state in enumerate(obj.state_space):
                indices[state] = index
        obj.index_of = indices
        return obj

    @property
    def generator_matrix(self):
        # 返回对象的第三个参数，即生成矩阵
        return self.args[2]

    @cacheit
    def transition_probabilities(self, gen_mat=None):
        t = Dummy('t')
        if isinstance(gen_mat, (Matrix, ImmutableMatrix)) and \
                gen_mat.is_diagonalizable():
            # 如果生成矩阵是可对角化的，使用对角化后的矩阵加速计算转移概率
            Q, D = gen_mat.diagonalize()
            return Lambda(t, Q*exp(t*D)*Q.inv())
        if gen_mat != None:
            # 否则使用原始的生成矩阵计算转移概率
            return Lambda(t, exp(t*gen_mat))

    def limiting_distribution(self):
        gen_mat = self.generator_matrix
        if gen_mat is None:
            return None
        if isinstance(gen_mat, MatrixSymbol):
            # 如果生成矩阵是符号矩阵，返回一个符号向量和限制条件的 Lambda 表达式
            wm = MatrixSymbol('wm', 1, gen_mat.shape[0])
            return Lambda((wm, gen_mat), Eq(wm*gen_mat, wm))
        w = IndexedBase('w')
        wi = [w[i] for i in range(gen_mat.shape[0])]
        wm = Matrix([wi])
        # 构造线性方程组列表，求解限制分布
        eqs = (wm*gen_mat).tolist()[0]
        eqs.append(sum(wi) - 1)
        soln = list(linsolve(eqs, wi))[0]
        return ImmutableMatrix([soln])
# 定义 BernoulliProcess 类，继承自 DiscreteTimeStochasticProcess 类
class BernoulliProcess(DiscreteTimeStochasticProcess):
    """
    The Bernoulli process consists of repeated
    independent Bernoulli process trials with the same parameter `p`.
    It's assumed that the probability `p` applies to every
    trial and that the outcomes of each trial
    are independent of all the rest. Therefore Bernoulli Process
    is Discrete State and Discrete Time Stochastic Process.

    Parameters
    ==========

    sym : Symbol/str
        符号或字符串，代表该随机过程的符号表示
    success : Integer/str
        认为是成功事件的整数或字符串。默认为 1。
    failure: Integer/str
        认为是失败事件的整数或字符串。默认为 0。
    p : Real Number between 0 and 1
        介于 0 到 1 之间的实数，表示成功事件发生的概率。

    Examples
    ========

    >>> from sympy.stats import BernoulliProcess, P, E
    >>> from sympy import Eq, Gt
    >>> B = BernoulliProcess("B", p=0.7, success=1, failure=0)
    >>> B.state_space
    {0, 1}
    >>> B.p.round(2)
    0.70
    >>> B.success
    1
    >>> B.failure
    0
    >>> X = B[1] + B[2] + B[3]
    >>> P(Eq(X, 0)).round(2)
    0.03
    >>> P(Eq(X, 2)).round(2)
    0.44
    >>> P(Eq(X, 4)).round(2)
    0
    >>> P(Gt(X, 1)).round(2)
    0.78
    >>> P(Eq(B[1], 0) & Eq(B[2], 1) & Eq(B[3], 0) & Eq(B[4], 1)).round(2)
    0.04
    >>> B.joint_distribution(B[1], B[2])
    JointDistributionHandmade(Lambda((B[1], B[2]), Piecewise((0.7, Eq(B[1], 1)),
    (0.3, Eq(B[1], 0)), (0, True))*Piecewise((0.7, Eq(B[2], 1)), (0.3, Eq(B[2], 0)),
    (0, True))))
    >>> E(2*B[1] + B[2]).round(2)
    2.10
    >>> P(B[1] < 1).round(2)
    0.30

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_process
    .. [2] https://mathcs.clarku.edu/~djoyce/ma217/bernoulli.pdf

    """

    # 索引集合为自然数的非负数集合
    index_set = S.Naturals0

    # 新建 BernoulliProcess 实例的构造函数，验证参数有效性，并返回实例
    def __new__(cls, sym, p, success=1, failure=0):
        _value_check(p >= 0 and p <= 1, 'Value of p must be between 0 and 1.')
        sym = _symbol_converter(sym)
        p = _sympify(p)
        success = _sym_sympify(success)
        failure = _sym_sympify(failure)
        return Basic.__new__(cls, sym, p, success, failure)

    # 返回符号属性
    @property
    def symbol(self):
        return self.args[0]

    # 返回概率 p 属性
    @property
    def p(self):
        return self.args[1]

    # 返回成功事件属性
    @property
    def success(self):
        return self.args[2]

    # 返回失败事件属性
    @property
    def failure(self):
        return self.args[3]

    # 返回状态空间，成功和失败事件的集合
    @property
    def state_space(self):
        return _set_converter([self.success, self.failure])

    # 返回分布函数，如果未指定键值则发出分布警告并返回 BernoulliDistribution(p)
    def distribution(self, key=None):
        if key is None:
            self._deprecation_warn_distribution()
            return BernoulliDistribution(self.p)
        return BernoulliDistribution(self.p, self.success, self.failure)

    # 返回简单随机变量，基于给定的随机变量名称和参数 p、succ、fail
    def simple_rv(self, rv):
        return Bernoulli(rv.name, p=self.p,
                succ=self.success, fail=self.failure)
    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Computes expectation.

        Parameters
        ==========

        expr : RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition : Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation of the RandomIndexedSymbol.

        """
        # 调用 _SubstituteRV 类的静态方法 _expectation 计算期望
        return _SubstituteRV._expectation(expr, condition, evaluate, **kwargs)

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Computes probability.

        Parameters
        ==========

        condition : Relational
            Condition for which probability has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        given_condition : Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Probability of the condition.

        """
        # 调用 _SubstituteRV 类的静态方法 _probability 计算概率
        return _SubstituteRV._probability(condition, given_condition, evaluate, **kwargs)

    def density(self, x):
        """
        Computes density function.

        Parameters
        ==========

        x : Symbol
            The symbol for which density needs to be computed.

        Returns
        =======

        Piecewise function representing the density.

        """
        # 根据输入的 x 计算密度函数，返回一个 Piecewise 对象
        return Piecewise((self.p, Eq(x, self.success)),
                         (1 - self.p, Eq(x, self.failure)),
                         (S.Zero, True))
class _SubstituteRV:
    """
    Internal class to handle the queries of expectation and probability
    by substitution.
    """

    @staticmethod
    def _rvindexed_subs(expr, condition=None):
        """
        Substitutes the RandomIndexedSymbol with the RandomSymbol with
        same name, distribution and probability as RandomIndexedSymbol.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        """

        # 获取所有随机符号
        rvs_expr = random_symbols(expr)
        # 如果表达式中存在随机符号
        if len(rvs_expr) != 0:
            swapdict_expr = {}
            # 遍历每个随机符号
            for rv in rvs_expr:
                if isinstance(rv, RandomIndexedSymbol):
                    # 用等效的简单随机变量替换随机索引符号
                    newrv = rv.pspace.process.simple_rv(rv)
                    swapdict_expr[rv] = newrv
            # 使用替换字典替换表达式中的随机索引符号
            expr = expr.subs(swapdict_expr)

        # 获取所有条件中的随机符号
        rvs_cond = random_symbols(condition)
        # 如果条件中存在随机符号
        if len(rvs_cond) != 0:
            swapdict_cond = {}
            # 遍历每个条件中的随机符号
            for rv in rvs_cond:
                if isinstance(rv, RandomIndexedSymbol):
                    # 用等效的简单随机变量替换随机索引符号
                    newrv = rv.pspace.process.simple_rv(rv)
                    swapdict_cond[rv] = newrv
            # 使用替换字典替换条件中的随机索引符号
            condition = condition.subs(swapdict_cond)

        # 返回替换后的表达式和条件
        return expr, condition

    @classmethod
    def _expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Internal method for computing expectation of indexed RV.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation of the RandomIndexedSymbol.

        """
        # 进行随机变量的替换
        new_expr, new_condition = self._rvindexed_subs(expr, condition)

        # 如果表达式不是随机变量，直接返回表达式本身
        if not is_random(new_expr):
            return new_expr

        # 获取新表达式的概率空间
        new_pspace = pspace(new_expr)

        # 如果存在条件，将条件应用于表达式
        if new_condition is not None:
            new_expr = given(new_expr, new_condition)

        # 如果表达式是加法，则根据线性期望的性质逐个计算
        if new_expr.is_Add:
            return Add(*[new_pspace.compute_expectation(
                        expr=arg, evaluate=evaluate, **kwargs)
                        for arg in new_expr.args])

        # 否则，直接计算表达式的期望
        return new_pspace.compute_expectation(
                new_expr, evaluate=evaluate, **kwargs)
    def _probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Internal method for computing probability of indexed RV
        
        Parameters
        ==========
        
        condition: Relational
            Condition for which probability has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        
        given_condition: Relational/And
            The given conditions under which computations should be done.
        
        Returns
        =======
        
        Probability of the condition.
        
        """
        # 对条件和给定条件进行符号替换
        new_condition, new_givencondition = self._rvindexed_subs(condition, given_condition)
        
        # 如果给定条件是随机符号，处理特殊情况
        if isinstance(new_givencondition, RandomSymbol):
            # 获取新条件中的随机符号
            condrv = random_symbols(new_condition)
            # 如果条件中只有一个随机符号且与给定条件相同，返回伯努利分布
            if len(condrv) == 1 and condrv[0] == new_givencondition:
                return BernoulliDistribution(self._probability(new_condition), 0, 1)
            
            # 如果新条件中有依赖于给定条件的随机符号，返回条件概率
            if any(dependent(rv, new_givencondition) for rv in condrv):
                return Probability(new_condition, new_givencondition)
            else:
                return self._probability(new_condition)
        
        # 如果给定条件不是关系型或布尔类型，引发值错误
        if new_givencondition is not None and \
                not isinstance(new_givencondition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (new_givencondition))
        
        # 如果条件或给定条件为 False，返回零
        if new_givencondition == False or new_condition == False:
            return S.Zero
        
        # 如果条件为 True，返回一
        if new_condition == True:
            return S.One
        
        # 如果条件不是关系型或布尔类型，引发值错误
        if not isinstance(new_condition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (new_condition))
        
        # 如果有给定条件，则在新的条件表达式上重新计算概率
        if new_givencondition is not None:
            return self._probability(given(new_condition, new_givencondition, **kwargs), **kwargs)
        
        # 计算条件空间中的概率，并根据需要进行评估
        result = pspace(new_condition).probability(new_condition, **kwargs)
        if evaluate and hasattr(result, 'doit'):
            return result.doit()
        else:
            return result
def get_timerv_swaps(expr, condition):
    """
    Finds the appropriate interval for each time stamp in expr by parsing
    the given condition and returns intervals for each timestamp and
    dictionary that maps variable time-stamped Random Indexed Symbol to its
    corresponding Random Indexed variable with fixed time stamp.

    Parameters
    ==========

    expr: SymPy Expression
        表达式，包含具有可变时间戳的随机索引符号
    condition: Relational/Boolean Expression
        包含表达式中变量时间戳时间边界的条件表达式

    Examples
    ========

    >>> from sympy.stats.stochastic_process_types import get_timerv_swaps, PoissonProcess
    >>> from sympy import symbols, Contains, Interval
    >>> x, t, d = symbols('x t d', positive=True)
    >>> X = PoissonProcess("X", 3)
    >>> get_timerv_swaps(x*X(t), Contains(t, Interval.Lopen(0, 1)))
    ([Interval.Lopen(0, 1)], {X(t): X(1)})
    >>> get_timerv_swaps((X(t)**2 + X(d)**2), Contains(t, Interval.Lopen(0, 1))
    ... & Contains(d, Interval.Ropen(1, 4))) # doctest: +SKIP
    ([Interval.Ropen(1, 4), Interval.Lopen(0, 1)], {X(d): X(3), X(t): X(1)})

    Returns
    =======

    intervals: list
        每个时间戳定义的区间列表（Interval/FiniteSet）
    rv_swap: dict
        将变量时间随机索引符号映射到固定时间随机索引变量的字典

    """

    # 检查条件是否为关系型或布尔表达式，否则抛出错误
    if not isinstance(condition, (Relational, Boolean)):
        raise ValueError("%s is not a relational or combination of relationals"
            % (condition))
    
    # 获取表达式中的所有随机索引符号
    expr_syms = list(expr.atoms(RandomIndexedSymbol))
    
    # 根据条件的类型，提取出条件列表
    if isinstance(condition, (And, Or)):
        given_cond_args = condition.args
    else: # 单一条件
        given_cond_args = (condition, )
    
    rv_swap = {}  # 用于存储随机变量交换的字典
    intervals = []  # 用于存储时间戳定义的区间列表
    
    # 遍历所有表达式中的随机索引符号
    for expr_sym in expr_syms:
        # 遍历所有给定的条件
        for arg in given_cond_args:
            # 如果条件中包含当前随机索引符号的关键字，并且关键字是符号类型
            if arg.has(expr_sym.key) and isinstance(expr_sym.key, Symbol):
                # 将条件转换为区间对象
                intv = _set_converter(arg.args[1])
                diff_key = intv._sup - intv._inf
                # 检查区间是否有无限大的边界
                if diff_key == oo:
                    raise ValueError("%s should have finite bounds" % str(expr_sym.name))
                elif diff_key == S.Zero:  # 如果区间只包含一个元素
                    diff_key = intv._sup
                # 将随机索引符号映射到固定时间随机索引变量
                rv_swap[expr_sym] = expr_sym.subs({expr_sym.key: diff_key})
                # 将区间对象添加到区间列表中
                intervals.append(intv)
    
    # 返回时间戳定义的区间列表和随机变量交换的字典
    return intervals, rv_swap


class CountingProcess(ContinuousTimeStochasticProcess):
    """
    This class handles the common methods of the Counting Processes
    such as Poisson, Wiener and Gamma Processes
    """
    # 设置索引集为 [0, ∞) 的区间对象
    index_set = _set_converter(Interval(0, oo))

    @property
    def symbol(self):
        return self.args[0]
    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Computes expectation

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Boolean
            The given conditions under which computations should be done, i.e,
            the intervals on which each variable time stamp in expr is defined

        Returns
        =======

        Expectation of the given expr

        """
        if condition is not None:
            # 获取时间戳交换和时间间隔
            intervals, rv_swap = get_timerv_swaps(expr, condition)
            # 当时间间隔不重叠时，认为它们是独立的
            if len(intervals) == 1 or all(Intersection(*intv_comb) == EmptySet
                                          for intv_comb in itertools.combinations(intervals, 2)):
                if expr.is_Add:
                    # 如果表达式是加法，则逐个计算期望
                    return Add.fromiter(self.expectation(arg, condition)
                                        for arg in expr.args)
                # 使用时间戳交换来替换表达式中的随机变量
                expr = expr.subs(rv_swap)
            else:
                # 如果时间间隔有重叠，则返回表达式的期望
                return Expectation(expr, condition)

        # 调用内部方法计算表达式的期望
        return _SubstituteRV._expectation(expr, evaluate=evaluate, **kwargs)


    def _solve_argwith_tworvs(self, arg):
        # 检查两个随机变量的时间戳，如果顺序不正确或者是等式类型，则进行调整
        if arg.args[0].key >= arg.args[1].key or isinstance(arg, Eq):
            # 计算时间戳的差异
            diff_key = abs(arg.args[0].key - arg.args[1].key)
            rv = arg.args[0]
            # 用差异创建新的参数
            arg = arg.__class__(rv.pspace.process(diff_key), 0)
        else:
            # 计算时间戳的差异
            diff_key = arg.args[1].key - arg.args[0].key
            rv = arg.args[1]
            # 用差异创建新的参数
            arg = arg.__class__(rv.pspace.process(diff_key), 0)
        # 返回调整后的参数
        return arg
class PoissonProcess(CountingProcess):
    """
    The Poisson process is a counting process. It is usually used in scenarios
    where we are counting the occurrences of certain events that appear
    to happen at a certain rate, but completely at random.

    Parameters
    ==========

    sym : Symbol/str
        Symbol or string representing the process.
    lamda : Positive number
        Rate of the process, `lamda > 0`.

    Examples
    ========

    >>> from sympy.stats import PoissonProcess, P, E
    >>> from sympy import symbols, Eq, Ne, Contains, Interval
    >>> X = PoissonProcess("X", lamda=3)
    >>> X.state_space
    Naturals0
    >>> X.lamda
    3
    >>> t1, t2 = symbols('t1 t2', positive=True)
    >>> P(X(t1) < 4)
    (9*t1**3/2 + 9*t1**2/2 + 3*t1 + 1)*exp(-3*t1)
    >>> P(Eq(X(t1), 2) | Ne(X(t1), 4), Contains(t1, Interval.Ropen(2, 4)))
    1 - 36*exp(-6)
    >>> P(Eq(X(t1), 2) & Eq(X(t2), 3), Contains(t1, Interval.Lopen(0, 2))
    ... & Contains(t2, Interval.Lopen(2, 4)))
    648*exp(-12)
    >>> E(X(t1))
    3*t1
    >>> E(X(t1)**2 + 2*X(t2),  Contains(t1, Interval.Lopen(0, 1))
    ... & Contains(t2, Interval.Lopen(1, 2)))
    18
    >>> P(X(3) < 1, Eq(X(1), 0))
    exp(-6)
    >>> P(Eq(X(4), 3), Eq(X(2), 3))
    exp(-6)
    >>> P(X(2) <= 3, X(1) > 1)
    5*exp(-3)

    Merging two Poisson Processes

    >>> Y = PoissonProcess("Y", lamda=4)
    >>> Z = X + Y
    >>> Z.lamda
    7

    Splitting a Poisson Process into two independent Poisson Processes

    >>> N, M = Z.split(l1=2, l2=5)
    >>> N.lamda, M.lamda
    (2, 5)

    References
    ==========

    .. [1] https://www.probabilitycourse.com/chapter11/11_0_0_intro.php
    .. [2] https://en.wikipedia.org/wiki/Poisson_point_process

    """

    def __new__(cls, sym, lamda):
        # 检查参数lamda必须为正数
        _value_check(lamda > 0, 'lamda should be a positive number.')
        # 转换sym为符号对象
        sym = _symbol_converter(sym)
        # 将lamda转换为符号表达式
        lamda = _sympify(lamda)
        # 调用基类的构造函数创建新的实例
        return Basic.__new__(cls, sym, lamda)

    @property
    def lamda(self):
        # 返回对象的lamda属性值
        return self.args[1]

    @property
    def state_space(self):
        # 返回状态空间为非负整数集合
        return S.Naturals0

    def distribution(self, key):
        # 如果key是RandomIndexedSymbol类型，则发出分布警告
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return PoissonDistribution(self.lamda*key.key)
        return PoissonDistribution(self.lamda*key)

    def density(self, x):
        # 返回泊松过程的密度函数
        return (self.lamda*x.key)**x / factorial(x) * exp(-(self.lamda*x.key))

    def simple_rv(self, rv):
        # 返回一个简单随机变量的泊松分布
        return Poisson(rv.name, lamda=self.lamda*rv.key)

    def __add__(self, other):
        # 合并两个泊松过程实例
        if not isinstance(other, PoissonProcess):
            raise ValueError("Only instances of Poisson Process can be merged")
        return PoissonProcess(Dummy(self.symbol.name + other.symbol.name),
                self.lamda + other.lamda)

    def split(self, l1, l2):
        # 将泊松过程分割为两个独立的泊松过程
        if _sympify(l1 + l2) != self.lamda:
            raise ValueError("Sum of l1 and l2 should be %s" % str(self.lamda))
        return PoissonProcess(Dummy("l1"), l1), PoissonProcess(Dummy("l2"), l2)
class WienerProcess(CountingProcess):
    """
    The Wiener process is a real valued continuous-time stochastic process.
    In physics it is used to study Brownian motion and it is often also called
    Brownian motion due to its historical connection with physical process of the
    same name originally observed by Scottish botanist Robert Brown.

    Parameters
    ==========

    sym : Symbol/str
        Symbol representing the Wiener process.

    Examples
    ========

    >>> from sympy.stats import WienerProcess, P, E
    >>> from sympy import symbols, Contains, Interval
    >>> X = WienerProcess("X")
    >>> X.state_space
    Reals
    >>> t1, t2 = symbols('t1 t2', positive=True)
    >>> P(X(t1) < 7).simplify()
    erf(7*sqrt(2)/(2*sqrt(t1)))/2 + 1/2
    >>> P((X(t1) > 2) | (X(t1) < 4), Contains(t1, Interval.Ropen(2, 4))).simplify()
    -erf(1)/2 + erf(2)/2 + 1
    >>> E(X(t1))
    0
    >>> E(X(t1) + 2*X(t2),  Contains(t1, Interval.Lopen(0, 1))
    ... & Contains(t2, Interval.Lopen(1, 2)))
    0

    References
    ==========

    .. [1] https://www.probabilitycourse.com/chapter11/11_4_0_brownian_motion_wiener_process.php
    .. [2] https://en.wikipedia.org/wiki/Wiener_process

    """
    def __new__(cls, sym):
        # 调用 _symbol_converter 方法转换符号类型
        sym = _symbol_converter(sym)
        # 调用基类 Basic 的构造方法创建实例
        return Basic.__new__(cls, sym)

    @property
    def state_space(self):
        # 返回状态空间为实数集
        return S.Reals

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            # 如果 key 是随机索引符号类型，则发出分布警告
            self._deprecation_warn_distribution()
            # 返回正态分布对象，均值为 0，标准差为 sqrt(key.key)
            return NormalDistribution(0, sqrt(key.key))
        # 返回正态分布对象，均值为 0，标准差为 sqrt(key)
        return NormalDistribution(0, sqrt(key))

    def density(self, x):
        # 返回密度函数 exp(-x**2/(2*x.key)) / (sqrt(2*pi)*sqrt(x.key))
        return exp(-x**2/(2*x.key)) / (sqrt(2*pi)*sqrt(x.key))

    def simple_rv(self, rv):
        # 返回以 rv.name 为均值，标准差为 sqrt(rv.key) 的正态分布随机变量
        return Normal(rv.name, 0, sqrt(rv.key))


class GammaProcess(CountingProcess):
    r"""
    A Gamma process is a random process with independent gamma distributed
    increments. It is a pure-jump increasing Levy process.

    Parameters
    ==========

    sym : Symbol/str
        Symbol representing the Gamma process.
    lamda : Positive number
        Jump size of the process, ``lamda > 0``
    gamma : Positive number
        Rate of jump arrivals, `\gamma > 0`

    Examples
    ========

    >>> from sympy.stats import GammaProcess, E, P, variance
    >>> from sympy import symbols, Contains, Interval, Not
    >>> t, d, x, l, g = symbols('t d x l g', positive=True)
    >>> X = GammaProcess("X", l, g)
    >>> E(X(t))
    g*t/l
    >>> variance(X(t)).simplify()
    g*t/l**2
    >>> X = GammaProcess('X', 1, 2)
    >>> P(X(t) < 1).simplify()
    lowergamma(2*t, 1)/gamma(2*t)
    >>> P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
    ... Contains(d, Interval.Lopen(7, 8))).simplify()
    -4*exp(-3) + 472*exp(-8)/3 + 1
    >>> E(X(2) + x*E(X(5)))
    10*x + 4

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_process

    """
    # 重载构造方法，初始化一个新的对象
    def __new__(cls, sym, lamda, gamma):
        # 检查 lamda 是否为正数，否则抛出异常
        _value_check(lamda > 0, 'lamda should be a positive number')
        # 检查 gamma 是否为正数，否则抛出异常
        _value_check(gamma > 0, 'gamma should be a positive number')
        # 将 sym 转换为符号对象
        sym = _symbol_converter(sym)
        # 将 gamma 转换为符号表达式对象
        gamma = _sympify(gamma)
        # 将 lamda 转换为符号表达式对象
        lamda = _sympify(lamda)
        # 调用基类 Basic 的构造方法创建对象
        return Basic.__new__(cls, sym, lamda, gamma)

    # 获取对象的 lamda 属性值
    @property
    def lamda(self):
        return self.args[1]

    # 获取对象的 gamma 属性值
    @property
    def gamma(self):
        return self.args[2]

    # 返回状态空间，即闭区间 [0, ∞)
    @property
    def state_space(self):
        return _set_converter(Interval(0, oo))

    # 计算分布函数，如果 key 是随机索引符号对象，则给出警告后返回 Gamma 分布对象
    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return GammaDistribution(self.gamma*key.key, 1/self.lamda)
        # 否则，返回以 key 为参数的 Gamma 分布对象
        return GammaDistribution(self.gamma*key, 1/self.lamda)

    # 计算密度函数，使用 Gamma 分布的参数计算
    def density(self, x):
        k = self.gamma*x.key
        theta = 1/self.lamda
        # 返回 Gamma 分布的密度函数值
        return x**(k - 1) * exp(-x/theta) / (gamma(k)*theta**k)

    # 创建简单随机变量，使用 Gamma 分布的参数
    def simple_rv(self, rv):
        return Gamma(rv.name, self.gamma*rv.key, 1/self.lamda)
```