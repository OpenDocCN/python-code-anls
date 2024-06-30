# `D:\src\scipysrc\sympy\sympy\concrete\expr_with_limits.py`

```
# 从 sympy.core.add 模块中导入 Add 类
# 从 sympy.core.containers 模块中导入 Tuple 类
# 从 sympy.core.expr 模块中导入 Expr 类
# 从 sympy.core.function 模块中导入 AppliedUndef, UndefinedFunction 类
# 从 sympy.core.mul 模块中导入 Mul 类
# 从 sympy.core.relational 模块中导入 Equality, Relational 类
# 从 sympy.core.singleton 模块中导入 S 类
# 从 sympy.core.symbol 模块中导入 Symbol, Dummy 类
# 从 sympy.core.sympify 模块中导入 sympify 函数
# 从 sympy.functions.elementary.piecewise 模块中导入 piecewise_fold, Piecewise 类
# 从 sympy.logic.boolalg 模块中导入 BooleanFunction 类
# 从 sympy.matrices.matrixbase 模块中导入 MatrixBase 类
# 从 sympy.sets.sets 模块中导入 Interval, Set 类
# 从 sympy.sets.fancysets 模块中导入 Range 类
# 从 sympy.tensor.indexed 模块中导入 Idx 类
# 从 sympy.utilities 模块中导入 flatten 函数
# 从 sympy.utilities.iterables 模块中导入 sift, is_sequence 函数
# 从 sympy.utilities.exceptions 模块中导入 sympy_deprecation_warning 函数
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import (
    piecewise_fold,
    Piecewise
)
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrixbase import MatrixBase
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.exceptions import sympy_deprecation_warning


def _common_new(cls, function, *symbols, discrete, **assumptions):
    """Return either a special return value or the tuple,
    (function, limits, orientation). This code is common to
    both ExprWithLimits and AddWithLimits."""

    # 将 function 转换为 SymPy 表达式
    function = sympify(function)

    # 如果 function 是 Equality 类的实例
    if isinstance(function, Equality):
        # 处理限制条件并返回等式表达式
        limits, orientation = _process_limits(*symbols, discrete=discrete)
        # 如果限制条件不全为定积分形式，则发出警告
        if not (limits and all(len(limit) == 3 for limit in limits)):
            sympy_deprecation_warning(
                """
                Creating a indefinite integral with an Eq() argument is
                deprecated.

                This is because indefinite integrals do not preserve equality
                due to the arbitrary constants. If you want an equality of
                indefinite integrals, use Eq(Integral(a, x), Integral(b, x))
                explicitly.
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-indefinite-integral-eq",
                stacklevel=5,
            )

        # 提取等式左右两边，并返回相应的表达式
        lhs = function.lhs
        rhs = function.rhs
        return Equality(cls(lhs, *symbols, **assumptions), \
                        cls(rhs, *symbols, **assumptions))

    # 如果 function 是 S.NaN，直接返回 S.NaN
    if function is S.NaN:
        return S.NaN

    # 如果 symbols 非空，处理限制条件并替换 function 中的符号
    if symbols:
        limits, orientation = _process_limits(*symbols, discrete=discrete)
        for i, li in enumerate(limits):
            if len(li) == 4:
                function = function.subs(li[0], li[-1])
                limits[i] = Tuple(*li[:-1])
    else:
        # 如果未提供符号，尝试计算一般形式
        free = function.free_symbols
        if len(free) != 1:
            raise ValueError(
                "specify dummy variables for %s" % function)
        limits, orientation = [Tuple(s) for s in free], 1

    # 将嵌套调用的限制条件和 function 展开
    while cls == type(function):
        limits = list(function.limits) + limits
        function = function.function
    # 初始化一个空字典，用于存储替换规则
    reps = {}
    
    # 根据限制条件中的积分变量集合，创建一个集合，表示函数中的积分变量
    symbols_of_integration = {i[0] for i in limits}
    
    # 遍历函数中所有的 Piecewise 函数对象
    for p in function.atoms(Piecewise):
        # 如果某个 Piecewise 函数不包含任何积分变量
        if not p.has(*symbols_of_integration):
            # 为该 Piecewise 函数创建一个虚拟符号，将其加入替换字典中
            reps[p] = Dummy()
    
    # 使用替换字典，将函数中不含积分变量的 Piecewise 函数替换为虚拟符号
    function = function.xreplace(reps)
    
    # 对函数进行 Piecewise 折叠（合并 Piecewise 函数中相同条件的分支）
    function = piecewise_fold(function)
    
    # 移除替换过程中添加的虚拟符号，恢复原始的函数表达式
    function = function.xreplace({v: k for k, v in reps.items()})
    
    # 返回处理后的函数、限制条件和方向性
    return function, limits, orientation
def _process_limits(*symbols, discrete=None):
    """处理符号列表并将其转换为规范限制，存储为元组(symbol, lower, upper)。当上限缺失时，改变函数的方向。
    如果限制指定为(symbol, Range)，则可能返回长度为4的列表，如果需要变量的变换；在表达式中应替换符号的表达式是列表中的第四个元素。
    """
    limits = []  # 初始化一个空列表，用于存储限制条件
    orientation = 1  # 初始化方向变量为1，表示正向
    if discrete is None:
        err_msg = 'discrete must be True or False'  # 如果discrete参数为None，提示错误信息
    elif discrete:
        err_msg = 'use Range, not Interval or Relational'  # 如果discrete为True，提示使用Range而不是Interval或Relational
    else:
        err_msg = 'use Interval or Relational, not Range'  # 如果discrete为False，提示使用Interval或Relational而不是Range
    return limits, orientation  # 返回处理后的限制列表和方向信息


class ExprWithLimits(Expr):
    __slots__ = ('is_commutative',)

    def __new__(cls, function, *symbols, **assumptions):
        from sympy.concrete.products import Product
        pre = _common_new(cls, function, *symbols,
            discrete=issubclass(cls, Product), **assumptions)
        if isinstance(pre, tuple):
            function, limits, _ = pre  # 如果返回的预处理结果是元组，则解包赋值给function和limits
        else:
            return pre  # 如果不是元组，则直接返回预处理结果

        # limits must have upper and lower bounds; the indefinite form
        # is not supported. This restriction does not apply to AddWithLimits
        if any(len(l) != 3 or None in l for l in limits):
            raise ValueError('ExprWithLimits requires values for lower and upper bounds.')
            # 如果任何一个限制条件的长度不为3或者有None值，抛出数值错误异常

        obj = Expr.__new__(cls, **assumptions)  # 创建Expr类的新实例对象
        arglist = [function]
        arglist.extend(limits)  # 将限制条件添加到参数列表中
        obj._args = tuple(arglist)  # 将参数列表转换为元组并赋值给对象的_args属性
        obj.is_commutative = function.is_commutative  # 设置对象的is_commutative属性为函数是否可交换，已经在限制条件中检查过

        return obj  # 返回创建的对象实例

    @property
    def function(self):
        """返回应用于限制条件的函数。

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x
        >>> Integral(x**2, (x,)).function
        x**2

        See Also
        ========

        limits, variables, free_symbols
        """
        return self._args[0]  # 返回对象的第一个参数，即应用于限制条件的函数

    @property
    def kind(self):
        return self.function.kind  # 返回函数的类型

    @property
    def limits(self):
        """返回表达式的限制条件。

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i
        >>> Integral(x**i, (i, 1, 3)).limits
        ((i, 1, 3),)

        See Also
        ========

        function, variables, free_symbols
        """
        return self._args[1:]  # 返回对象的剩余参数，即表达式的限制条件

    @property
    def variables(self):
        """
        Return a list of the limit variables.

        >>> from sympy import Sum
        >>> from sympy.abc import x, i
        >>> Sum(x**i, (i, 1, 3)).variables
        [i]

        See Also
        ========

        function, limits, free_symbols
        as_dummy : Rename dummy variables
        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable
        """
        # 返回限制变量列表，从每个限制元组中取第一个元素（即变量）
        return [l[0] for l in self.limits]

    @property
    def bound_symbols(self):
        """
        Return only variables that are dummy variables.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i, j, k
        >>> Integral(x**i, (i, 1, 3), (j, 2), k).bound_symbols
        [i, j]

        See Also
        ========

        function, limits, free_symbols
        as_dummy : Rename dummy variables
        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable
        """
        # 返回仅为虚拟变量的变量列表，排除长度不为1的限制元组
        return [l[0] for l in self.limits if len(l) != 1]

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).

        Examples
        ========

        >>> from sympy import Sum
        >>> from sympy.abc import x, y
        >>> Sum(x, (x, y, 1)).free_symbols
        {y}
        """
        # 返回对象中的符号集合，不包括那些具有特定值（即虚拟符号）
        # 首先替换所有非符号的积分变量为虚拟变量，然后获取剩余的自由符号
        function, limits = self.function, self.limits
        reps = {i[0]: i[0] if i[0].free_symbols == {i[0]} else Dummy() for i in self.limits}
        function = function.xreplace(reps)
        isyms = function.free_symbols
        for xab in limits:
            v = reps[xab[0]]
            if len(xab) == 1:
                isyms.add(v)
                continue
            if v in isyms:
                isyms.remove(v)
            for i in xab[1:]:
                isyms.update(i.free_symbols)
        reps = {v: k for k, v in reps.items()}
        return {reps.get(_, _) for _ in isyms}

    @property
    def is_number(self):
        """
        Return True if the Sum has no free symbols, else False.
        """
        # 如果 Sum 没有自由符号则返回 True，否则返回 False
        return not self.free_symbols

    def _eval_interval(self, x, a, b):
        """
        Evaluate the Sum over the interval [a, b] for variable x.
        """
        # 替换限制列表中与 x 对应的元组，计算函数的积分
        limits = [(i if i[0] != x else (x, a, b)) for i in self.limits]
        integrand = self.function
        return self.func(integrand, *limits)
    # 检查限制是否被确定为有限的函数，根据以下几种情况：
    # - 显式的边界
    # - 对边界的假设
    # - 对变量的假设
    # 如果能确定边界是有限的，则返回 True；如果能确定边界是无限的，则返回 False；如果信息不足以确定，则返回 None。

    ret_None = False  # 初始化一个标志位，用于指示是否有限信息不足的情况
    for lim in self.limits:  # 遍历限制条件列表
        if len(lim) == 3:  # 如果限制条件包含上下界
            if any(l.is_infinite for l in lim[1:]):
                # 如果任何一个边界是无限的 (+/-oo)
                return False
            elif any(l.is_infinite is None for l in lim[1:]):
                # 或者可能存在对变量的假设
                if lim[0].is_infinite is None:
                    ret_None = True  # 如果变量的有限性未知，则设定标志位为 True
        else:
            if lim[0].is_infinite is None:
                ret_None = True  # 单个限制条件，且变量的有限性未知，则设定标志位为 True

    if ret_None:
        return None  # 如果存在有限性信息不足的情况，则返回 None
    return True  # 所有限制条件都确定为有限，则返回 True
    def has_reversed_limits(self):
        """
        Returns True if the limits are known to be in reversed order, either
        by the explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be in normal order, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 8, 1)).has_reversed_limits
        True

        >>> Sum(x, (x, 1, oo)).has_reversed_limits
        False

        >>> M = Symbol('M')
        >>> Integral(x, (x, 1, M)).has_reversed_limits

        >>> N = Symbol('N', integer=True, positive=True)
        >>> Sum(x, (x, 1, N)).has_reversed_limits
        False

        >>> Product(x, (x, 2, N)).has_reversed_limits

        >>> Product(x, (x, 2, N)).subs(N, N + 2).has_reversed_limits
        False

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.has_empty_sequence

        """
        ret_None = False  # 初始化一个变量 ret_None，用来表示是否存在不确定情况
        for lim in self.limits:  # 遍历 self.limits 列表中的每一个限制条件 lim
            if len(lim) == 3:  # 如果限制条件由三个部分组成
                var, a, b = lim  # 将限制条件解包为变量 var，下限 a，上限 b
                dif = b - a  # 计算上限和下限的差值 dif
                if dif.is_extended_negative:  # 如果 dif 是负数
                    return True  # 返回 True，表示限制条件是反向的
                elif dif.is_extended_nonnegative:  # 如果 dif 是非负数
                    continue  # 继续循环，继续检查下一个限制条件
                else:  # 如果 dif 不能确定是否为负数或非负数
                    ret_None = True  # 设置 ret_None 为 True，表示无法确定限制条件的顺序
            else:  # 如果限制条件不是由三个部分组成
                return None  # 返回 None，表示无法确定限制条件的顺序
        if ret_None:  # 如果存在不确定情况
            return None  # 返回 None，表示无法确定限制条件的顺序
        return False  # 如果所有限制条件均为正常顺序，则返回 False
class AddWithLimits(ExprWithLimits):
    r"""Represents unevaluated oriented additions.
        Parent class for Integral and Sum.
    """

    __slots__ = ()  # 使用空的 __slots__ 集合来优化内存空间，这个类不需要额外的实例属性

    def __new__(cls, function, *symbols, **assumptions):
        from sympy.concrete.summations import Sum
        # 调用 _common_new 函数处理参数，根据是否为 Sum 类进行离散性检查
        pre = _common_new(cls, function, *symbols,
            discrete=issubclass(cls, Sum), **assumptions)
        if isinstance(pre, tuple):
            # 如果返回的是元组，则解构出 function, limits, orientation
            function, limits, orientation = pre
        else:
            # 否则直接返回 pre
            return pre

        obj = Expr.__new__(cls, **assumptions)
        # 根据 orientation*function 创建参数列表 arglist，ExprWithLimits 中不使用 orientation
        arglist = [orientation * function]
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative  # 根据 function 是否可交换来设置 is_commutative

        return obj

    def _eval_adjoint(self):
        # 如果所有限制条件中的变量都是实数，则返回 self.func 的共轭转置
        if all(x.is_real for x in flatten(self.limits)):
            return self.func(self.function.adjoint(), *self.limits)
        return None

    def _eval_conjugate(self):
        # 如果所有限制条件中的变量都是实数，则返回 self.func 的共轭
        if all(x.is_real for x in flatten(self.limits)):
            return self.func(self.function.conjugate(), *self.limits)
        return None

    def _eval_transpose(self):
        # 如果所有限制条件中的变量都是实数，则返回 self.func 的转置
        if all(x.is_real for x in flatten(self.limits)):
            return self.func(self.function.transpose(), *self.limits)
        return None

    def _eval_factor(self, **hints):
        # 如果只有一个限制条件
        if 1 == len(self.limits):
            # 对 function 进行因式分解
            summand = self.function.factor(**hints)
            if summand.is_Mul:
                # 将 summand.args 按照是否可交换分组，并且不含有变量交集的部分作为 Mul 的输出
                out = sift(summand.args, lambda w: w.is_commutative \
                    and not set(self.variables) & w.free_symbols)
                return Mul(*out[True])*self.func(Mul(*out[False]), \
                    *self.limits)
        else:
            # 对 function 的前面 len(limits)-1 个限制条件进行递归调用 _eval_factor
            summand = self.func(self.function, *self.limits[0:-1]).factor()
            if not summand.has(self.variables[-1]):
                # 如果 summand 不含有最后一个变量，则返回乘以 self.func(1, [self.limits[-1]]) 的结果
                return self.func(1, [self.limits[-1]]).doit()*summand
            elif isinstance(summand, Mul):
                # 如果 summand 是 Mul 类型，则对其进行因式分解
                return self.func(summand, self.limits[-1]).factor()
        return self

    def _eval_expand_basic(self, **hints):
        # 对 function 进行展开操作
        summand = self.function.expand(**hints)
        force = hints.get('force', False)
        if (summand.is_Add and (force or summand.is_commutative and
                 self.has_finite_limits is not False)):
            # 如果 summand 是 Add 类型，并且满足条件，返回其每个元素作为 self.func 的结果
            return Add(*[self.func(i, *self.limits) for i in summand.args])
        elif isinstance(summand, MatrixBase):
            # 如果 summand 是 MatrixBase 类型，则对其每个元素应用 lambda 函数，并返回结果
            return summand.applyfunc(lambda x: self.func(x, *self.limits))
        elif summand != self.function:
            # 如果 summand 不等于原 function，则返回 self.func(summand, *self.limits)
            return self.func(summand, *self.limits)
        return self
```