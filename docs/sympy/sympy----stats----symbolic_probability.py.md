# `D:\src\scipysrc\sympy\sympy\stats\symbolic_probability.py`

```
# 导入 itertools 模块，用于生成迭代器的函数和类
import itertools
# 导入 SymPy 中的相关模块和类
from sympy.concrete.summations import Sum  # 导入 Sum 类，用于表示求和表达式
from sympy.core.add import Add  # 导入 Add 类，用于表示加法表达式
from sympy.core.expr import Expr  # 导入 Expr 类，所有表达式的基类
from sympy.core.function import expand as _expand  # 导入 expand 函数，并命名为 _expand
from sympy.core.mul import Mul  # 导入 Mul 类，用于表示乘法表达式
from sympy.core.relational import Eq  # 导入 Eq 类，用于表示等式
from sympy.core.singleton import S  # 导入 S 单例，用于表示符号
from sympy.core.symbol import Symbol  # 导入 Symbol 类，用于表示符号变量
from sympy.integrals.integrals import Integral  # 导入 Integral 类，用于表示积分
from sympy.logic.boolalg import Not  # 导入 Not 类，用于逻辑非运算
from sympy.core.parameters import global_parameters  # 导入 global_parameters 函数，用于全局参数设置
from sympy.core.sorting import default_sort_key  # 导入 default_sort_key 函数，用于默认排序键
from sympy.core.sympify import _sympify  # 导入 _sympify 函数，用于将输入转换为 SymPy 表达式
from sympy.core.relational import Relational  # 导入 Relational 类，用于表示关系式
from sympy.logic.boolalg import Boolean  # 导入 Boolean 类，用于布尔代数运算
from sympy.stats import variance, covariance  # 导入方差和协方差函数
from sympy.stats.rv import (  # 导入随机变量相关的类和函数
    RandomSymbol, pspace, dependent, given, sampling_E, RandomIndexedSymbol, is_random,
    PSpace, sampling_P, random_symbols
)

# 设置模块的公开接口，只包含 'Probability', 'Expectation', 'Variance', 'Covariance' 这四个名称
__all__ = ['Probability', 'Expectation', 'Variance', 'Covariance']

# 定义一个装饰器，用于判断表达式是否包含随机变量
@is_random.register(Expr)
def _(x):
    atoms = x.free_symbols
    if len(atoms) == 1 and next(iter(atoms)) == x:
        return False
    return any(is_random(i) for i in atoms)

# 定义一个特殊情况下的装饰器，用于判断 RandomSymbol 类型的对象一定是随机变量
@is_random.register(RandomSymbol)  # type: ignore
def _(x):
    return True

# Probability 类继承自 Expr 类，用于表示概率的符号表达式
class Probability(Expr):
    """
    Symbolic expression for the probability.

    Examples
    ========

    >>> from sympy.stats import Probability, Normal
    >>> from sympy import Integral
    >>> X = Normal("X", 0, 1)
    >>> prob = Probability(X > 1)
    >>> prob
    Probability(X > 1)

    Integral representation:

    >>> prob.rewrite(Integral)
    Integral(sqrt(2)*exp(-_z**2/2)/(2*sqrt(pi)), (_z, 1, oo))

    Evaluation of the integral:

    >>> prob.evaluate_integral()
    sqrt(2)*(-sqrt(2)*sqrt(pi)*erf(sqrt(2)/2) + sqrt(2)*sqrt(pi))/(4*sqrt(pi))
    """

    is_commutative = True  # 指示该类的对象是可交换的

    # 构造函数，接受 prob 和 condition 参数，并初始化一个符号表达式对象
    def __new__(cls, prob, condition=None, **kwargs):
        prob = _sympify(prob)  # 将 prob 参数转换为 SymPy 表达式
        if condition is None:
            obj = Expr.__new__(cls, prob)
        else:
            condition = _sympify(condition)  # 将 condition 参数转换为 SymPy 表达式
            obj = Expr.__new__(cls, prob, condition)
        obj._condition = condition  # 存储 condition 参数到对象的 _condition 属性
        return obj
    # 定义一个方法 `doit`，接受关键字参数 `hints`
    def doit(self, **hints):
        # 获取条件表达式
        condition = self.args[0]
        # 获取给定的条件
        given_condition = self._condition
        # 从 `hints` 参数中获取 `numsamples`，默认为 False
        numsamples = hints.get('numsamples', False)
        # 从 `hints` 参数中获取 `evaluate`，默认为 True
        evaluate = hints.get('evaluate', True)

        # 如果条件是 `Not` 类型的对象
        if isinstance(condition, Not):
            # 返回 `1 - self.func(condition.args[0], given_condition, evaluate=evaluate).doit(**hints)` 的结果
            return S.One - self.func(condition.args[0], given_condition,
                                     evaluate=evaluate).doit(**hints)

        # 如果条件中包含 `RandomIndexedSymbol` 对象
        if condition.has(RandomIndexedSymbol):
            # 使用概率空间来计算概率
            return pspace(condition).probability(condition, given_condition,
                                                 evaluate=evaluate)

        # 如果给定条件是 `RandomSymbol` 类型的对象
        if isinstance(given_condition, RandomSymbol):
            # 获取条件中的随机符号
            condrv = random_symbols(condition)
            # 如果只有一个随机符号且与给定条件相同，返回伯努利分布
            if len(condrv) == 1 and condrv[0] == given_condition:
                from sympy.stats.frv_types import BernoulliDistribution
                return BernoulliDistribution(self.func(condition).doit(**hints), 0, 1)
            # 如果任何一个随机变量依赖于给定条件，返回条件概率
            if any(dependent(rv, given_condition) for rv in condrv):
                return Probability(condition, given_condition)
            else:
                # 否则，返回条件概率的结果
                return Probability(condition).doit()

        # 如果给定条件不为 None 且不是关系表达式或布尔表达式
        if given_condition is not None and \
                not isinstance(given_condition, (Relational, Boolean)):
            # 抛出值错误异常
            raise ValueError("%s is not a relational or combination of relationals"
                             % (given_condition))

        # 如果给定条件为 False 或条件为 `S.false`
        if given_condition == False or condition is S.false:
            # 返回 `S.Zero`
            return S.Zero
        # 如果条件不是关系表达式或布尔表达式
        if not isinstance(condition, (Relational, Boolean)):
            # 抛出值错误异常
            raise ValueError("%s is not a relational or combination of relationals"
                             % (condition))
        # 如果条件为 `S.true`
        if condition is S.true:
            # 返回 `S.One`
            return S.One

        # 如果存在 `numsamples`
        if numsamples:
            # 进行采样操作
            return sampling_P(condition, given_condition, numsamples=numsamples)
        # 如果存在给定条件
        if given_condition is not None:  # If there is a condition
            # 在新的条件表达式上重新计算概率
            return Probability(given(condition, given_condition)).doit()

        # 否则将工作交给概率空间来处理
        if pspace(condition) == PSpace():
            return Probability(condition, given_condition)

        # 计算条件概率
        result = pspace(condition).probability(condition)
        # 如果结果具有 `doit` 方法并且 `evaluate` 为真
        if hasattr(result, 'doit') and evaluate:
            # 返回结果的计算值
            return result.doit()
        else:
            # 否则返回结果本身
            return result

    # 将 `_eval_rewrite_as_Integral` 方法重写为 `Integral` 的计算
    def _eval_rewrite_as_Integral(self, arg, condition=None, **kwargs):
        return self.func(arg, condition=condition).doit(evaluate=False)

    # 将 `_eval_rewrite_as_Sum` 方法重写为 `Integral` 的计算
    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    # 计算积分的评估
    def evaluate_integral(self):
        # 将重写为 `Integral` 的结果进行计算
        return self.rewrite(Integral).doit()
# 定义一个符号表达式，表示期望值。
class Expectation(Expr):
    """
    Symbolic expression for the expectation.

    Examples
    ========

    >>> from sympy.stats import Expectation, Normal, Probability, Poisson
    >>> from sympy import symbols, Integral, Sum
    >>> mu = symbols("mu")
    >>> sigma = symbols("sigma", positive=True)
    >>> X = Normal("X", mu, sigma)
    >>> Expectation(X)
    Expectation(X)
    >>> Expectation(X).evaluate_integral().simplify()
    mu

    To get the integral expression of the expectation:

    >>> Expectation(X).rewrite(Integral)
    Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    The same integral expression, in more abstract terms:

    >>> Expectation(X).rewrite(Probability)
    Integral(x*Probability(Eq(X, x)), (x, -oo, oo))

    To get the Summation expression of the expectation for discrete random variables:

    >>> lamda = symbols('lamda', positive=True)
    >>> Z = Poisson('Z', lamda)
    >>> Expectation(Z).rewrite(Sum)
    Sum(Z*lamda**Z*exp(-lamda)/factorial(Z), (Z, 0, oo))

    This class is aware of some properties of the expectation:

    >>> from sympy.abc import a
    >>> Expectation(a*X)
    Expectation(a*X)
    >>> Y = Normal("Y", 1, 2)
    >>> Expectation(X + Y)
    Expectation(X + Y)

    To expand the ``Expectation`` into its expression, use ``expand()``:

    >>> Expectation(X + Y).expand()
    Expectation(X) + Expectation(Y)
    >>> Expectation(a*X + Y).expand()
    a*Expectation(X) + Expectation(Y)
    >>> Expectation(a*X + Y)
    Expectation(a*X + Y)
    >>> Expectation((X + Y)*(X - Y)).expand()
    Expectation(X**2) - Expectation(Y**2)

    To evaluate the ``Expectation``, use ``doit()``:

    >>> Expectation(X + Y).doit()
    mu + 1
    >>> Expectation(X + Expectation(Y + Expectation(2*X))).doit()
    3*mu + 1

    To prevent evaluating nested ``Expectation``, use ``doit(deep=False)``

    >>> Expectation(X + Expectation(Y)).doit(deep=False)
    mu + Expectation(Expectation(Y))
    >>> Expectation(X + Expectation(Y + Expectation(2*X))).doit(deep=False)
    mu + Expectation(Expectation(Expectation(2*X) + Y))

    """

    # 定义类的构造函数，接受表达式和条件参数
    def __new__(cls, expr, condition=None, **kwargs):
        # 对表达式进行符号化处理
        expr = _sympify(expr)
        # 如果表达式是矩阵，则使用多变量概率期望值类
        if expr.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import ExpectationMatrix
            return ExpectationMatrix(expr, condition)
        # 如果没有条件参数且表达式不是随机变量，则直接返回表达式
        if condition is None:
            if not is_random(expr):
                return expr
            # 否则创建一个新的 Expectation 对象
            obj = Expr.__new__(cls, expr)
        else:
            # 如果有条件参数，则对条件参数进行符号化处理
            condition = _sympify(condition)
            # 创建一个带条件的 Expectation 对象
            obj = Expr.__new__(cls, expr, condition)
        # 将条件参数存储在对象的 _condition 属性中
        obj._condition = condition
        return obj

    # 返回对象是否可交换的评估结果
    def _eval_is_commutative(self):
        return(self.args[0].is_commutative)
    # 扩展期望表达式的内容，根据提示获取第一个参数表达式和条件
    expr = self.args[0]
    condition = self._condition

    # 如果表达式不是随机表达式，则直接返回该表达式
    if not is_random(expr):
        return expr

    # 如果表达式是加法表达式，则对其每个参数进行期望计算并返回新的加法表达式
    if isinstance(expr, Add):
        return Add.fromiter(Expectation(a, condition=condition).expand()
                for a in expr.args)

    # 对表达式进行扩展
    expand_expr = _expand(expr)

    # 如果扩展后的表达式是加法表达式，则对其每个参数进行期望计算并返回新的加法表达式
    if isinstance(expand_expr, Add):
        return Add.fromiter(Expectation(a, condition=condition).expand()
                for a in expand_expr.args)

    # 如果表达式是乘法表达式，则分离其中的随机部分和非随机部分，并返回相应的乘法结果
    elif isinstance(expr, Mul):
        rv = []
        nonrv = []
        for a in expr.args:
            if is_random(a):
                rv.append(a)
            else:
                nonrv.append(a)
        return Mul.fromiter(nonrv) * Expectation(Mul.fromiter(rv), condition=condition)

    # 如果以上条件均不满足，则返回自身
    return self

``````
    # 执行期望操作，考虑到深度和条件
    deep = hints.get('deep', True)
    condition = self._condition
    expr = self.args[0]
    numsamples = hints.get('numsamples', False)
    evaluate = hints.get('evaluate', True)

    # 如果深度标志为真，则递归执行表达式的期望操作
    if deep:
        expr = expr.doit(**hints)

    # 如果表达式不是随机表达式或者已经是期望表达式，则直接返回表达式本身
    if not is_random(expr) or isinstance(expr, Expectation):  # expr isn't random?
        return expr

    # 如果需要进行蒙特卡洛采样计算
    if numsamples:
        evalf = hints.get('evalf', True)
        return sampling_E(expr, condition, numsamples=numsamples, evalf=evalf)

    # 如果表达式中包含随机索引符号
    if expr.has(RandomIndexedSymbol):
        return pspace(expr).compute_expectation(expr, condition)

    # 如果存在条件，则创建新的表达式并重新计算期望值
    if condition is not None:  # If there is a condition
        return self.func(given(expr, condition)).doit(**hints)

    # 一些已知的效率语句

    # 如果表达式是加法表达式，则对其每个参数进行期望计算并返回新的加法表达式
    if expr.is_Add:  # We know that E is Linear
        return Add(*[self.func(arg, condition).doit(**hints)
                if not isinstance(arg, Expectation) else self.func(arg, condition)
                     for arg in expr.args])

    # 如果表达式是乘法表达式，并且其中包含期望符号，则直接返回表达式
    if expr.is_Mul:
        if expr.atoms(Expectation):
            return expr

    # 如果表达式的概率空间等于空，则直接返回表达式
    if pspace(expr) == PSpace():
        return self.func(expr)

    # 否则，将工作委托给概率空间来计算期望值，并根据需要评估结果
    result = pspace(expr).compute_expectation(expr, evaluate=evaluate)
    if hasattr(result, 'doit') and evaluate:
        return result.doit(**hints)
    else:
        return result
    # 根据概率分布参数，将表达式重写为概率的形式
    def _eval_rewrite_as_Probability(self, arg, condition=None, **kwargs):
        # 找出参数中的随机符号集合
        rvs = arg.atoms(RandomSymbol)
        # 如果随机符号数量大于1，目前不支持多个随机符号的处理
        if len(rvs) > 1:
            raise NotImplementedError()
        # 如果没有随机符号，则直接返回参数
        if len(rvs) == 0:
            return arg

        # 取出唯一的随机符号
        rv = rvs.pop()
        # 如果随机符号的概率空间未知，则无法处理
        if rv.pspace is None:
            raise ValueError("Probability space not known")

        # 获取随机符号的符号对象，并根据命名规则处理
        symbol = rv.symbol
        if symbol.name[0].isupper():
            symbol = Symbol(symbol.name.lower())
        else:
            symbol = Symbol(symbol.name + "_1")

        # 根据随机符号的概率空间类型，构建对应的积分或求和表达式
        if rv.pspace.is_Continuous:
            # 连续型随机变量的情况，返回积分表达式
            return Integral(arg.replace(rv, symbol) * Probability(Eq(rv, symbol), condition), 
                            (symbol, rv.pspace.domain.set.inf, rv.pspace.domain.set.sup))
        else:
            if rv.pspace.is_Finite:
                # 有限型随机变量暂未实现
                raise NotImplementedError
            else:
                # 离散型随机变量的情况，返回求和表达式
                return Sum(arg.replace(rv, symbol) * Probability(Eq(rv, symbol), condition), 
                           (symbol, rv.pspace.domain.set.inf, rv.pspace.set.sup))

    # 将表达式重写为积分的形式，并进行计算
    def _eval_rewrite_as_Integral(self, arg, condition=None, evaluate=False, **kwargs):
        return self.func(arg, condition=condition).doit(deep=False, evaluate=evaluate)

    # 将 Sum 方法重写为 Integral 方法，用于处理离散型情况下的求和
    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral # For discrete this will be Sum

    # 对积分进行评估计算
    def evaluate_integral(self):
        return self.rewrite(Integral).doit()

    # 对求和进行评估计算，实际上调用了 evaluate_integral 方法
    evaluate_sum = evaluate_integral
# 定义一个 Variance 类，继承自 Expr 类，用于表示方差的符号表达式。
class Variance(Expr):
    """
    Symbolic expression for the variance.

    Examples
    ========

    >>> from sympy import symbols, Integral
    >>> from sympy.stats import Normal, Expectation, Variance, Probability
    >>> mu = symbols("mu", positive=True)
    >>> sigma = symbols("sigma", positive=True)
    >>> X = Normal("X", mu, sigma)
    >>> Variance(X)
    Variance(X)
    >>> Variance(X).evaluate_integral()
    sigma**2

    Integral representation of the underlying calculations:

    >>> Variance(X).rewrite(Integral)
    Integral(sqrt(2)*(X - Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo)))**2*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    Integral representation, without expanding the PDF:

    >>> Variance(X).rewrite(Probability)
    -Integral(x*Probability(Eq(X, x)), (x, -oo, oo))**2 + Integral(x**2*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the variance in terms of the expectation

    >>> Variance(X).rewrite(Expectation)
    -Expectation(X)**2 + Expectation(X**2)

    Some transformations based on the properties of the variance may happen:

    >>> from sympy.abc import a
    >>> Y = Normal("Y", 0, 1)
    >>> Variance(a*X)
    Variance(a*X)

    To expand the variance in its expression, use ``expand()``:

    >>> Variance(a*X).expand()
    a**2*Variance(X)
    >>> Variance(X + Y)
    Variance(X + Y)
    >>> Variance(X + Y).expand()
    2*Covariance(X, Y) + Variance(X) + Variance(Y)

    """

    # 定义类的构造方法，接受参数 arg 和可选的 condition，以及其他关键字参数
    def __new__(cls, arg, condition=None, **kwargs):
        # 将参数 arg 转换为符号表达式
        arg = _sympify(arg)

        # 如果参数是一个矩阵，引入多元统计概率中的 VarianceMatrix 类来处理
        if arg.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import VarianceMatrix
            return VarianceMatrix(arg, condition)
        
        # 如果没有提供 condition 参数，则创建一个新的 Expr 对象
        if condition is None:
            obj = Expr.__new__(cls, arg)
        else:
            # 否则，将 condition 参数也转换为符号表达式，并创建一个带有条件的 Expr 对象
            condition = _sympify(condition)
            obj = Expr.__new__(cls, arg, condition)
        
        # 将 condition 存储在对象的 _condition 属性中
        obj._condition = condition
        return obj

    # 定义一个方法用于判断对象是否可交换
    def _eval_is_commutative(self):
        return self.args[0].is_commutative
    # 根据给定的提示参数来展开表达式
    def expand(self, **hints):
        # 获取第一个参数
        arg = self.args[0]
        # 获取条件
        condition = self._condition

        # 如果第一个参数不是随机符号，则返回零
        if not is_random(arg):
            return S.Zero

        # 如果第一个参数是 RandomSymbol 类型，则直接返回自身
        if isinstance(arg, RandomSymbol):
            return self
        # 如果第一个参数是 Add 类型
        elif isinstance(arg, Add):
            # 初始化一个空列表
            rv = []
            # 遍历 Add 类型参数的子参数
            for a in arg.args:
                # 如果子参数是随机符号，则加入列表
                if is_random(a):
                    rv.append(a)
            # 计算方差项
            variances = Add(*(Variance(xv, condition).expand() for xv in rv))
            # 定义映射到协方差的 lambda 函数
            map_to_covar = lambda x: 2*Covariance(*x, condition=condition).expand()
            # 计算协方差项
            covariances = Add(*map(map_to_covar, itertools.combinations(rv, 2)))
            # 返回方差项和协方差项之和
            return variances + covariances
        # 如果第一个参数是 Mul 类型
        elif isinstance(arg, Mul):
            # 初始化非随机变量和随机变量的列表
            nonrv = []
            rv = []
            # 遍历 Mul 类型参数的子参数
            for a in arg.args:
                # 如果子参数是随机符号，则加入随机变量列表；否则将其平方后加入非随机变量列表
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a**2)
            # 如果随机变量列表为空，则返回零
            if len(rv) == 0:
                return S.Zero
            # 返回非随机变量乘积乘以随机变量的方差
            return Mul.fromiter(nonrv)*Variance(Mul.fromiter(rv), condition)

        # 如果表达式中包含 RandomSymbol，则返回自身
        # 这里应该是默认情况，假设其他情况未覆盖到的情况
        return self

    # 重写为期望值表达式
    def _eval_rewrite_as_Expectation(self, arg, condition=None, **kwargs):
        # 计算参数的平方的期望
        e1 = Expectation(arg**2, condition)
        # 计算参数的期望的平方
        e2 = Expectation(arg, condition)**2
        # 返回期望值表达式的重写结果
        return e1 - e2

    # 重写为概率表达式
    def _eval_rewrite_as_Probability(self, arg, condition=None, **kwargs):
        # 使用期望值重写为概率表达式
        return self.rewrite(Expectation).rewrite(Probability)

    # 重写为积分表达式
    def _eval_rewrite_as_Integral(self, arg, condition=None, **kwargs):
        # 返回参数的方差表达式
        return variance(self.args[0], self._condition, evaluate=False)

    # 使用积分表达式重写为求和表达式
    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    # 计算积分的值
    def evaluate_integral(self):
        # 将表达式重写为积分，然后求值
        return self.rewrite(Integral).doit()
    """
    符号表达式，表示协方差。

    Examples
    ========

    >>> from sympy.stats import Covariance
    >>> from sympy.stats import Normal
    >>> X = Normal("X", 3, 2)
    >>> Y = Normal("Y", 0, 1)
    >>> Z = Normal("Z", 0, 1)
    >>> W = Normal("W", 0, 1)
    >>> cexpr = Covariance(X, Y)
    >>> cexpr
    Covariance(X, Y)

    评估协方差，`X` 和 `Y` 是独立的，因此结果为零：

    >>> cexpr.evaluate_integral()
    0

    将协方差表达式重写为期望的形式：

    >>> from sympy.stats import Expectation
    >>> cexpr.rewrite(Expectation)
    Expectation(X*Y) - Expectation(X)*Expectation(Y)

    若要展开参数，请使用 ``expand()``：

    >>> from sympy.abc import a, b, c, d
    >>> Covariance(a*X + b*Y, c*Z + d*W)
    Covariance(a*X + b*Y, c*Z + d*W)
    >>> Covariance(a*X + b*Y, c*Z + d*W).expand()
    a*c*Covariance(X, Z) + a*d*Covariance(W, X) + b*c*Covariance(Y, Z) + b*d*Covariance(W, Y)

    此类意识到协方差的一些性质：

    >>> Covariance(X, X).expand()
    Variance(X)
    >>> Covariance(a*X, b*Y).expand()
    a*b*Covariance(X, Y)
    """

    def __new__(cls, arg1, arg2, condition=None, **kwargs):
        # 将参数转换为符号表达式
        arg1 = _sympify(arg1)
        arg2 = _sympify(arg2)

        # 如果其中一个参数是矩阵，则调用 CrossCovarianceMatrix 处理
        if arg1.is_Matrix or arg2.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import CrossCovarianceMatrix
            return CrossCovarianceMatrix(arg1, arg2, condition)

        # 根据 evaluate 参数或全局设置，对参数进行排序
        if kwargs.pop('evaluate', global_parameters.evaluate):
            arg1, arg2 = sorted([arg1, arg2], key=default_sort_key)

        # 根据是否存在 condition 参数，创建新的表达式对象
        if condition is None:
            obj = Expr.__new__(cls, arg1, arg2)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, arg1, arg2, condition)
        obj._condition = condition
        return obj

    def _eval_is_commutative(self):
        # 检查第一个参数是否是可交换的
        return self.args[0].is_commutative

    def expand(self, **hints):
        # 获取参数和条件
        arg1 = self.args[0]
        arg2 = self.args[1]
        condition = self._condition

        # 如果参数相同，返回对应参数的方差展开
        if arg1 == arg2:
            return Variance(arg1, condition).expand()

        # 如果任一参数不是随机变量，则返回零
        if not is_random(arg1):
            return S.Zero
        if not is_random(arg2):
            return S.Zero

        # 对参数进行排序，保证一致性
        arg1, arg2 = sorted([arg1, arg2], key=default_sort_key)

        # 如果参数都是 RandomSymbol 类型，则返回它们的协方差
        if isinstance(arg1, RandomSymbol) and isinstance(arg2, RandomSymbol):
            return Covariance(arg1, arg2, condition)

        # 对每个参数的扩展进行处理
        coeff_rv_list1 = self._expand_single_argument(arg1.expand())
        coeff_rv_list2 = self._expand_single_argument(arg2.expand())

        # 计算每对扩展参数的协方差，并生成加法项
        addends = [a*b*Covariance(*sorted([r1, r2], key=default_sort_key), condition=condition)
                   for (a, r1) in coeff_rv_list1 for (b, r2) in coeff_rv_list2]
        return Add.fromiter(addends)

    @classmethod
    # 对单个参数表达式进行展开，返回 (系数, 随机符号) 的对列表：
    def _expand_single_argument(cls, expr):
        # 如果表达式是 RandomSymbol 类型，直接返回包含其本身的元组列表
        if isinstance(expr, RandomSymbol):
            return [(S.One, expr)]
        # 如果表达式是 Add 类型，则处理其 args 列表
        elif isinstance(expr, Add):
            outval = []
            # 遍历 Add 类型表达式的每一个参数
            for a in expr.args:
                # 如果参数是 Mul 类型，则调用 _get_mul_nonrv_rv_tuple 处理
                if isinstance(a, Mul):
                    outval.append(cls._get_mul_nonrv_rv_tuple(a))
                # 如果参数是随机变量，则将其系数视为 1，加入结果列表
                elif is_random(a):
                    outval.append((S.One, a))

            return outval
        # 如果表达式是 Mul 类型，则调用 _get_mul_nonrv_rv_tuple 处理
        elif isinstance(expr, Mul):
            return [cls._get_mul_nonrv_rv_tuple(expr)]
        # 如果表达式是随机变量，则将其系数视为 1，加入结果列表
        elif is_random(expr):
            return [(S.One, expr)]

    @classmethod
    # 获取 Mul 类型表达式中的非随机变量和随机变量的乘积对
    def _get_mul_nonrv_rv_tuple(cls, m):
        rv = []
        nonrv = []
        # 遍历 Mul 类型表达式的每一个因子
        for a in m.args:
            # 如果因子是随机变量，则加入随机变量列表 rv
            if is_random(a):
                rv.append(a)
            # 否则加入非随机变量列表 nonrv
            else:
                nonrv.append(a)
        # 返回非随机变量和随机变量的乘积对
        return (Mul.fromiter(nonrv), Mul.fromiter(rv))

    # 将当前对象重写为对应的期望值表达式
    def _eval_rewrite_as_Expectation(self, arg1, arg2, condition=None, **kwargs):
        # 构造 arg1*arg2 的期望值表达式 e1
        e1 = Expectation(arg1*arg2, condition)
        # 构造 arg1 和 arg2 分别求期望后相乘的表达式 e2
        e2 = Expectation(arg1, condition)*Expectation(arg2, condition)
        # 返回两者差，即重写后的结果
        return e1 - e2

    # 将当前对象重写为对应的概率表达式
    def _eval_rewrite_as_Probability(self, arg1, arg2, condition=None, **kwargs):
        # 先重写为期望值表达式，再重写为概率表达式
        return self.rewrite(Expectation).rewrite(Probability)

    # 将当前对象重写为对应的积分表达式
    def _eval_rewrite_as_Integral(self, arg1, arg2, condition=None, **kwargs):
        # 调用 covariance 函数计算两个参数的协方差，并返回不进行求值的表达式
        return covariance(self.args[0], self.args[1], self._condition, evaluate=False)

    # 将 _eval_rewrite_as_Sum 重写为 _eval_rewrite_as_Integral
    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    # 对当前对象进行积分求解，并返回结果
    def evaluate_integral(self):
        return self.rewrite(Integral).doit()
# 定义一个 Moment 类，继承自 Expr 类，用于表示符号化的 Moment
class Moment(Expr):
    """
    Symbolic class for Moment

    Examples
    ========

    >>> from sympy import Symbol, Integral
    >>> from sympy.stats import Normal, Expectation, Probability, Moment
    >>> mu = Symbol('mu', real=True)
    >>> sigma = Symbol('sigma', positive=True)
    >>> X = Normal('X', mu, sigma)
    >>> M = Moment(X, 3, 1)

    To evaluate the result of Moment use `doit`:

    >>> M.doit()
    mu**3 - 3*mu**2 + 3*mu*sigma**2 + 3*mu - 3*sigma**2 - 1

    Rewrite the Moment expression in terms of Expectation:

    >>> M.rewrite(Expectation)
    Expectation((X - 1)**3)

    Rewrite the Moment expression in terms of Probability:

    >>> M.rewrite(Probability)
    Integral((x - 1)**3*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the Moment expression in terms of Integral:

    >>> M.rewrite(Integral)
    Integral(sqrt(2)*(X - 1)**3*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    """
    
    def __new__(cls, X, n, c=0, condition=None, **kwargs):
        # 符号化 X, n, c
        X = _sympify(X)
        n = _sympify(n)
        c = _sympify(c)
        # 如果有条件，符号化条件并返回用条件构造的 Moment 实例
        if condition is not None:
            condition = _sympify(condition)
            return super().__new__(cls, X, n, c, condition)
        else:
            # 否则返回用 X, n, c 构造的 Moment 实例
            return super().__new__(cls, X, n, c)

    def doit(self, **hints):
        # 使用 Expectation 重写并计算 Moment 表达式的值
        return self.rewrite(Expectation).doit(**hints)

    def _eval_rewrite_as_Expectation(self, X, n, c=0, condition=None, **kwargs):
        # 重写 Moment 表达式为 Expectation 表达式
        return Expectation((X - c)**n, condition)

    def _eval_rewrite_as_Probability(self, X, n, c=0, condition=None, **kwargs):
        # 重写 Moment 表达式为 Probability 表达式
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, X, n, c=0, condition=None, **kwargs):
        # 重写 Moment 表达式为 Integral 表达式
        return self.rewrite(Expectation).rewrite(Integral)


# 定义一个 CentralMoment 类，继承自 Expr 类，用于表示符号化的 Central Moment
class CentralMoment(Expr):
    """
    Symbolic class Central Moment

    Examples
    ========

    >>> from sympy import Symbol, Integral
    >>> from sympy.stats import Normal, Expectation, Probability, CentralMoment
    >>> mu = Symbol('mu', real=True)
    >>> sigma = Symbol('sigma', positive=True)
    >>> X = Normal('X', mu, sigma)
    >>> CM = CentralMoment(X, 4)

    To evaluate the result of CentralMoment use `doit`:

    >>> CM.doit().simplify()
    3*sigma**4

    Rewrite the CentralMoment expression in terms of Expectation:

    >>> CM.rewrite(Expectation)
    Expectation((-Expectation(X) + X)**4)

    Rewrite the CentralMoment expression in terms of Probability:

    >>> CM.rewrite(Probability)
    Integral((x - Integral(x*Probability(True), (x, -oo, oo)))**4*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the CentralMoment expression in terms of Integral:

    >>> CM.rewrite(Integral)
    Integral(sqrt(2)*(X - Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo)))**4*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    """
    # 定义一个特殊方法 __new__，用于创建新对象
    def __new__(cls, X, n, condition=None, **kwargs):
        # 将 X 和 n 转换为符号表达式
        X = _sympify(X)
        n = _sympify(n)
        # 如果有条件参数，则将其也转换为符号表达式，并使用父类的 __new__ 方法创建对象
        if condition is not None:
            condition = _sympify(condition)
            return super().__new__(cls, X, n, condition)
        else:
            # 如果没有条件参数，则使用父类的 __new__ 方法创建对象
            return super().__new__(cls, X, n)

    # 定义方法 doit，执行对象的重写为期望值，并执行 doit 方法
    def doit(self, **hints):
        return self.rewrite(Expectation).doit(**hints)

    # 定义方法 _eval_rewrite_as_Expectation，将对象重写为期望值
    def _eval_rewrite_as_Expectation(self, X, n, condition=None, **kwargs):
        # 计算 X 的期望值 mu
        mu = Expectation(X, condition, **kwargs)
        # 返回 Moment(X, n, mu, condition, **kwargs) 对象重写为期望值的结果
        return Moment(X, n, mu, condition, **kwargs).rewrite(Expectation)

    # 定义方法 _eval_rewrite_as_Probability，将对象重写为概率
    def _eval_rewrite_as_Probability(self, X, n, condition=None, **kwargs):
        # 先将对象重写为期望值，再将期望值对象重写为概率对象
        return self.rewrite(Expectation).rewrite(Probability)

    # 定义方法 _eval_rewrite_as_Integral，将对象重写为积分
    def _eval_rewrite_as_Integral(self, X, n, condition=None, **kwargs):
        # 先将对象重写为期望值，再将期望值对象重写为积分对象
        return self.rewrite(Expectation).rewrite(Integral)
```