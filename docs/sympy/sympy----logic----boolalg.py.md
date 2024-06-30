# `D:\src\scipysrc\sympy\sympy\logic\boolalg.py`

```
"""
Boolean algebra module for SymPy
"""

# 导入必要的库
from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent

# 定义一个函数 as_Boolean，用于获取表达式 e 的布尔值
def as_Boolean(e):
    """Like ``bool``, return the Boolean value of an expression, e,
    which can be any instance of :py:class:`~.Boolean` or ``bool``.

    Examples
    ========

    >>> from sympy import true, false, nan
    >>> from sympy.logic.boolalg import as_Boolean
    >>> from sympy.abc import x
    >>> as_Boolean(0) is false
    True
    >>> as_Boolean(1) is true
    True
    >>> as_Boolean(x)
    x
    >>> as_Boolean(2)
    Traceback (most recent call last):
    ...
    TypeError: expecting bool or Boolean, not `2`.
    >>> as_Boolean(nan)
    Traceback (most recent call last):
    ...
    TypeError: expecting bool or Boolean, not `nan`.

    """
    from sympy.core.symbol import Symbol
    # 如果 e 是 True，则返回 sympy 中的 true
    if e == True:
        return true
    # 如果 e 是 False，则返回 sympy 中的 false
    if e == False:
        return false
    # 如果 e 是符号变量 Symbol，并且其 is_zero 方法返回 None，则返回原符号变量
    if isinstance(e, Symbol):
        z = e.is_zero
        if z is None:
            return e
        # 如果 is_zero 方法返回 True，则返回 false；否则返回 true
        return false if z else true
    # 如果 e 是 Boolean 类型的实例，则直接返回 e
    if isinstance(e, Boolean):
        return e
    # 若上述条件均不满足，则抛出类型错误异常
    raise TypeError('expecting bool or Boolean, not `%s`.' % e)


# 定义一个类 Boolean，表示布尔对象，继承自 Basic 类
@sympify_method_args
class Boolean(Basic):
    """A Boolean object is an object for which logic operations make sense."""

    __slots__ = ()

    kind = BooleanKind  # 设置类属性 kind 为 BooleanKind

    # 定义按位与运算符 & 的重载方法
    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __and__(self, other):
        return And(self, other)

    __rand__ = __and__  # 对 & 运算符的反向重载为与运算

    # 定义按位或运算符 | 的重载方法
    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __or__(self, other):
        return Or(self, other)

    __ror__ = __or__  # 对 | 运算符的反向重载为或运算

    # 定义按位取反运算符 ~ 的重载方法
    def __invert__(self):
        """Overloading for ~"""
        return Not(self)

    # 定义按位右移运算符 >> 的重载方法
    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __rshift__(self, other):
        return Implies(self, other)

    # 定义按位左移运算符 << 的重载方法
    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __lshift__(self, other):
        return Implies(other, self)

    __rrshift__ = __lshift__  # 对 >> 运算符的反向重载为左移运算
    __rlshift__ = __rshift__  # 对 << 运算符的反向重载为右移运算

    # 定义按位异或运算符 ^ 的重载方法
    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __xor__(self, other):
        return Xor(self, other)

    __rxor__ = __xor__  # 对 ^ 运算符的反向重载为异或运算
    def equals(self, other):
        """
        Returns ``True`` if the given formulas have the same truth table.
        For two formulas to be equal they must have the same literals.

        Examples
        ========

        >>> from sympy.abc import A, B, C
        >>> from sympy import And, Or, Not
        >>> (A >> B).equals(~B >> ~A)
        True
        >>> Not(And(A, B, C)).equals(And(Not(A), Not(B), Not(C)))
        False
        >>> Not(And(A, Not(A))).equals(Or(B, Not(B)))
        False

        """
        from sympy.logic.inference import satisfiable  # 导入判断逻辑公式可满足性的函数
        from sympy.core.relational import Relational  # 导入关系运算相关模块

        if self.has(Relational) or other.has(Relational):  # 如果公式中含有关系运算，抛出未实现错误
            raise NotImplementedError('handling of relationals')
        return self.atoms() == other.atoms() and \
            not satisfiable(Not(Equivalent(self, other)))  # 返回两个公式的原子集合是否相同，并且它们的等价否定不可满足

    def to_nnf(self, simplify=True):
        # override where necessary
        return self  # 如果需要，覆盖此方法

    def as_set(self):
        """
        Rewrites Boolean expression in terms of real sets.

        Examples
        ========

        >>> from sympy import Symbol, Eq, Or, And
        >>> x = Symbol('x', real=True)
        >>> Eq(x, 0).as_set()
        {0}
        >>> (x > 0).as_set()
        Interval.open(0, oo)
        >>> And(-2 < x, x < 2).as_set()
        Interval.open(-2, 2)
        >>> Or(x < -2, 2 < x).as_set()
        Union(Interval.open(-oo, -2), Interval.open(2, oo))

        """
        from sympy.calculus.util import periodicity  # 导入周期性相关模块
        from sympy.core.relational import Relational  # 导入关系运算相关模块

        free = self.free_symbols  # 获取表达式中的自由符号
        if len(free) == 1:  # 如果自由符号数量为1
            x = free.pop()  # 弹出自由符号
            if x.kind is NumberKind:  # 如果符号类型是数字类型
                reps = {}  # 创建替换字典
                for r in self.atoms(Relational):  # 遍历表达式中的关系运算
                    if periodicity(r, x) not in (0, None):  # 如果关系运算具有周期性解
                        s = r._eval_as_set()  # 计算关系运算的集合形式
                        if s in (S.EmptySet, S.UniversalSet, S.Reals):
                            reps[r] = s.as_relational(x)  # 将结果作为关系运算的替换
                            continue
                        raise NotImplementedError(filldedent('''
                            as_set is not implemented for relationals
                            with periodic solutions
                            '''))
                new = self.subs(reps)  # 使用替换字典替换原表达式中的关系运算
                if new.func != self.func:
                    return new.as_set()  # 如果替换后的对象类型不同，则重新开始
                else:
                    return new._eval_as_set()  # 否则评估替换后的对象的集合形式

            return self._eval_as_set()  # 否则评估原始表达式的集合形式
        else:
            raise NotImplementedError("Sorry, as_set has not yet been"
                                      " implemented for multivariate"
                                      " expressions")  # 抛出未实现错误，多变量表达式尚未实现

    @property
    def binary_symbols(self):
        from sympy.core.relational import Eq, Ne  # 导入关系运算等模块
        return set().union(*[i.binary_symbols for i in self.args
                           if i.is_Boolean or i.is_Symbol
                           or isinstance(i, (Eq, Ne))])  # 返回表达式中的布尔符号和关系运算符号集合
    # 定义一个方法 `_eval_refine`，用于在对象实例上评估与给定假设相关的条件
    def _eval_refine(self, assumptions):
        # 从 sympy.assumptions 模块中导入 ask 函数
        from sympy.assumptions import ask
        # 使用 ask 函数评估对象实例 self 在给定假设 assumptions 下的条件
        ret = ask(self, assumptions)
        # 如果评估结果为 True，则返回全局符号 true
        if ret is True:
            return true
        # 如果评估结果为 False，则返回全局符号 false
        elif ret is False:
            return false
        # 如果评估结果为 None，则返回空值
        return None
class BooleanAtom(Boolean):
    """
    Base class of :py:class:`~.BooleanTrue` and :py:class:`~.BooleanFalse`.
    """
    # 标识这是一个布尔原子类
    is_Boolean = True
    # 标识这是一个原子类
    is_Atom = True
    # 操作优先级，比 Expr 高
    _op_priority = 11  # higher than Expr

    def simplify(self, *a, **kw):
        # 简化操作，返回自身
        return self

    def expand(self, *a, **kw):
        # 展开操作，返回自身
        return self

    @property
    def canonical(self):
        # 返回自身的规范形式
        return self

    def _noop(self, other=None):
        # 不支持的操作，抛出 TypeError
        raise TypeError('BooleanAtom not allowed in this context.')

    __add__ = _noop  # 加法操作为不支持
    __radd__ = _noop  # 右加法操作为不支持
    __sub__ = _noop  # 减法操作为不支持
    __rsub__ = _noop  # 右减法操作为不支持
    __mul__ = _noop  # 乘法操作为不支持
    __rmul__ = _noop  # 右乘法操作为不支持
    __pow__ = _noop  # 幂运算为不支持
    __rpow__ = _noop  # 右幂运算为不支持
    __truediv__ = _noop  # 真除法为不支持
    __rtruediv__ = _noop  # 右真除法为不支持
    __mod__ = _noop  # 取模操作为不支持
    __rmod__ = _noop  # 右取模操作为不支持
    _eval_power = _noop  # 幂运算求值为不支持

    # /// drop when Py2 is no longer supported
    def __lt__(self, other):
        # 小于比较操作为不支持，抛出 TypeError
        raise TypeError(filldedent('''
            A Boolean argument can only be used in
            Eq and Ne; all other relationals expect
            real expressions.
        '''))

    __le__ = __lt__  # 小于等于比较操作也为不支持
    __gt__ = __lt__  # 大于比较操作为不支持
    __ge__ = __lt__  # 大于等于比较操作为不支持
    # \\\

    def _eval_simplify(self, **kwargs):
        # 简化求值操作，返回自身
        return self


class BooleanTrue(BooleanAtom, metaclass=Singleton):
    """
    SymPy version of ``True``, a singleton that can be accessed via ``S.true``.

    This is the SymPy version of ``True``, for use in the logic module. The
    primary advantage of using ``true`` instead of ``True`` is that shorthand Boolean
    operations like ``~`` and ``>>`` will work as expected on this class, whereas with
    True they act bitwise on 1. Functions in the logic module will return this
    class when they evaluate to true.

    Notes
    =====

    There is liable to be some confusion as to when ``True`` should
    be used and when ``S.true`` should be used in various contexts
    throughout SymPy. An important thing to remember is that
    ``sympify(True)`` returns ``S.true``. This means that for the most
    part, you can just use ``True`` and it will automatically be converted
    to ``S.true`` when necessary, similar to how you can generally use 1
    instead of ``S.One``.

    The rule of thumb is:

    "If the boolean in question can be replaced by an arbitrary symbolic
    ``Boolean``, like ``Or(x, y)`` or ``x > 1``, use ``S.true``.
    Otherwise, use ``True``"

    In other words, use ``S.true`` only on those contexts where the
    boolean is being used as a symbolic representation of truth.
    For example, if the object ends up in the ``.args`` of any expression,
    then it must necessarily be ``S.true`` instead of ``True``, as
    elements of ``.args`` must be ``Basic``. On the other hand,
    ``==`` is not a symbolic operation in SymPy, since it always returns
    ``True`` or ``False``, and does so in terms of structural equality
    rather than mathematical, so it should return ``True``. The assumptions
    system should use ``True`` and ``False``. Aside from not satisfying
    """
    # SymPy 中的 True 单例类，可通过 S.true 访问

    def __lt__(self, other):
        # 小于比较操作为不支持，抛出 TypeError
        raise TypeError(filldedent('''
            A Boolean argument can only be used in
            Eq and Ne; all other relationals expect
            real expressions.
        '''))

    __le__ = __lt__  # 小于等于比较操作也为不支持
    __gt__ = __lt__  # 大于比较操作为不支持
    __ge__ = __lt__  # 大于等于比较操作为不支持

    def _eval_simplify(self, **kwargs):
        # 简化求值操作，返回自身
        return self
    """
    定义一个自定义的逻辑值类，实现了一些逻辑运算和方法。

    """

    def __bool__(self):
        """
        当对象被用于布尔上下文时调用，始终返回 True。
        """
        return True

    def __hash__(self):
        """
        返回对象的哈希值，始终返回 True 对应的哈希值。
        """
        return hash(True)

    def __eq__(self, other):
        """
        比较对象与其他对象的相等性，如果其他对象是 True，则返回 True，否则调用父类的比较方法。
        """
        if other is True:
            return True
        if other is False:
            return False
        return super().__eq__(other)

    @property
    def negated(self):
        """
        返回逻辑值 False 对象。
        """
        return false

    def as_set(self):
        """
        将逻辑值转换为相应的集合表示。
        """
        return S.UniversalSet
class BooleanFalse(BooleanAtom, metaclass=Singleton):
    """
    SymPy version of ``False``, a singleton that can be accessed via ``S.false``.

    This is the SymPy version of ``False``, for use in the logic module. The
    primary advantage of using ``false`` instead of ``False`` is that shorthand
    Boolean operations like ``~`` and ``>>`` will work as expected on this class,
    whereas with ``False`` they act bitwise on 0. Functions in the logic module
    will return this class when they evaluate to false.

    Notes
    ======

    See the notes section in :py:class:`sympy.logic.boolalg.BooleanTrue`

    Examples
    ========

    >>> from sympy import sympify, true, false, Or
    >>> sympify(False)
    False
    >>> _ is False, _ is false
    (False, True)

    >>> Or(true, false)
    True
    >>> _ is true
    True

    Python operators give a boolean result for false but a
    bitwise result for False

    >>> ~false, ~False  # doctest: +SKIP
    (True, -1)
    >>> false >> false, False >> False
    (True, 0)

    See Also
    ========

    sympy.logic.boolalg.BooleanTrue

    """
    def __bool__(self):
        # 返回 False，以确保该类实例在布尔上下文中被视为假
        return False

    def __hash__(self):
        # 返回 False 的哈希值，确保该类实例在散列集合中被正确处理
        return hash(False)

    def __eq__(self, other):
        # 检查是否与另一个对象相等，如果另一个对象是 True 则返回 False，如果是 False 则返回 True，否则调用父类的相等方法
        if other is True:
            return False
        if other is False:
            return True
        return super().__eq__(other)

    @property
    def negated(self):
        # 返回 S.true，表示该类实例的否定是 S.true
        return true

    def as_set(self):
        """
        Rewrite logic operators and relationals in terms of real sets.

        Examples
        ========

        >>> from sympy import false
        >>> false.as_set()
        EmptySet
        """
        # 返回 S.EmptySet，表示该类实例作为集合表示为空集
        return S.EmptySet


true = BooleanTrue()
false = BooleanFalse()
# 将 true 和 false 分别赋值为 BooleanTrue 和 BooleanFalse 的实例，以便可以通过 S.true 和 S.false 访问

# We want S.true and S.false to work, rather than S.BooleanTrue and
# S.BooleanFalse, but making the class and instance names the same causes some
# major issues (like the inability to import the class directly from this
# file).
S.true = true
S.false = false

_sympy_converter[bool] = lambda x: true if x else false
# 将一个 lambda 函数添加到 _sympy_converter 中，用于将 Python 中的 bool 类型转换为 true 或 false
    # 对输入的参数列表中的每个参数应用 as_Boolean 函数，返回一个布尔值列表
    def binary_check_and_simplify(self, *args):
        return [as_Boolean(i) for i in args]

    # 转换当前逻辑表达式为否定范式（NNF），默认进行简化操作
    def to_nnf(self, simplify=True):
        # 调用 _to_nnf 方法，传递所有参数和简化选项
        return self._to_nnf(*self.args, simplify=simplify)

    # 转换当前逻辑表达式为代数正常形式（ANF），默认进行深度转换
    def to_anf(self, deep=True):
        # 调用 _to_anf 方法，传递所有参数和深度转换选项
        return self._to_anf(*self.args, deep=deep)

    @classmethod
    # 类方法：将输入参数转换为否定范式（NNF）
    def _to_nnf(cls, *args, **kwargs):
        # 获取简化选项，默认为 True
        simplify = kwargs.get('simplify', True)
        # 用于存储唯一参数的集合
        argset = set()
        # 遍历输入的每个参数
        for arg in args:
            # 如果参数不是文字（literal），将其转换为否定范式（NNF）
            if not is_literal(arg):
                arg = arg.to_nnf(simplify)
            # 如果需要简化
            if simplify:
                # 如果参数是当前类的实例，则获取其参数
                if isinstance(arg, cls):
                    arg = arg.args
                else:
                    arg = (arg,)
                # 遍历参数的每个元素
                for a in arg:
                    # 如果 Not(a) 已经在 argset 中，则返回零元素（表示假）
                    if Not(a) in argset:
                        return cls.zero
                    # 将 a 添加到 argset 中
                    argset.add(a)
            else:
                # 将参数添加到 argset 中（不进行简化）
                argset.add(arg)
        # 使用 argset 中的元素创建并返回一个新的逻辑表达式
        return cls(*argset)

    @classmethod
    # 类方法：将输入参数转换为代数正常形式（ANF）
    def _to_anf(cls, *args, **kwargs):
        # 获取深度转换选项，默认为 True
        deep = kwargs.get('deep', True)
        # 用于存储新参数的列表
        new_args = []
        # 遍历输入的每个参数
        for arg in args:
            # 如果需要深度转换，并且参数不是文字或是 Not 类的实例，则进行深度转换
            if deep:
                if not is_literal(arg) or isinstance(arg, Not):
                    arg = arg.to_anf(deep=deep)
            # 将处理后的参数添加到 new_args 列表中
            new_args.append(arg)
        # 使用 new_args 中的元素创建并返回一个新的逻辑表达式
        return cls(*new_args, remove_true=False)

    # 从 Expr 类中复制的差分方法
    def diff(self, *symbols, **assumptions):
        # 设置默认的评估选项为 True
        assumptions.setdefault("evaluate", True)
        # 返回 Derivative 对象，对当前对象进行符号的导数计算
        return Derivative(self, *symbols, **assumptions)

    # 对当前对象进行导数的评估
    def _eval_derivative(self, x):
        # 如果 x 在二元符号集合中
        if x in self.binary_symbols:
            from sympy.core.relational import Eq
            from sympy.functions.elementary.piecewise import Piecewise
            # 返回一个分段函数，根据等式的情况返回 0 或 1
            return Piecewise(
                (0, Eq(self.subs(x, 0), self.subs(x, 1))),
                (1, True))
        # 如果 x 在自由符号集合中，则提示未实现
        elif x in self.free_symbols:
            # 未实现，参考文档：https://www.encyclopediaofmath.org/index.php/Boolean_differential_calculus
            pass
        else:
            # 返回零元素（表示假）
            return S.Zero
class And(LatticeOp, BooleanFunction):
    """
    Logical AND function.

    It evaluates its arguments in order, returning false immediately
    when an argument is false and true if they are all true.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import And
    >>> x & y
    x & y

    Notes
    =====

    The ``&`` operator is provided as a convenience, but note that its use
    here is different from its normal use in Python, which is bitwise
    and. Hence, ``And(a, b)`` and ``a & b`` will produce different results if
    ``a`` and ``b`` are integers.

    >>> And(x, y).subs(x, 1)
    y

    """
    
    # Zero element of logical AND
    zero = false
    # Identity element of logical AND
    identity = true

    # Number of arguments, initially unknown
    nargs = None

    @classmethod
    def _new_args_filter(cls, args):
        # Simplify arguments to their boolean forms
        args = BooleanFunction.binary_check_and_simplify(*args)
        # Further filter arguments based on lattice operations for AND
        args = LatticeOp._new_args_filter(args, And)
        newargs = []
        rel = set()
        # Iterate over ordered arguments
        for x in ordered(args):
            if x.is_Relational:
                c = x.canonical
                # Check for duplicate relational forms
                if c in rel:
                    continue
                elif c.negated.canonical in rel:
                    # If a negation of a relational form is found, return false
                    return [false]
                else:
                    rel.add(c)
            # Append non-relational arguments to newargs
            newargs.append(x)
        return newargs

    def _eval_subs(self, old, new):
        args = []
        bad = None
        # Iterate over current object's arguments
        for i in self.args:
            try:
                # Attempt substitution of old with new in each argument
                i = i.subs(old, new)
            except TypeError:
                # Store TypeError encountered during substitution
                if bad is None:
                    bad = i
                continue
            # If substitution results in false, return false
            if i == False:
                return false
            # If substitution results in something other than true, add it to args
            elif i != True:
                args.append(i)
        if bad is not None:
            # Raise TypeError encountered during substitution
            bad.subs(old, new)
        # If old is an instance of And, replace parts of arguments with new if all are present
        if isinstance(old, And):
            old_set = set(old.args)
            if old_set.issubset(args):
                args = set(args) - old_set
                args.add(new)

        # Return result of the AND function with updated arguments
        return self.func(*args)
    # 定义一个名为 _eval_simplify 的方法，用于简化表达式
    def _eval_simplify(self, **kwargs):
        # 导入需要的模块和函数
        from sympy.core.relational import Equality, Relational
        from sympy.solvers.solveset import linear_coeffs
        
        # 调用父类的 _eval_simplify 方法进行标准的简化操作
        rv = super()._eval_simplify(**kwargs)
        
        # 如果简化后的结果不是 And 类型，则直接返回结果
        if not isinstance(rv, And):
            return rv

        # 将 rv.args 中的 Relational 和非 Relational 的部分分开
        Rel, nonRel = sift(rv.args, lambda i: isinstance(i, Relational),
                           binary=True)
        # 如果没有 Relational 类型的元素，则直接返回结果
        if not Rel:
            return rv
        
        # 将 Rel 中的元素分为 Equality 和其他类型
        eqs, other = sift(Rel, lambda i: isinstance(i, Equality), binary=True)

        # 从 kwargs 中获取 measure 的值
        measure = kwargs['measure']
        
        # 如果有 Equality 类型的表达式存在
        if eqs:
            # 从 kwargs 中获取 ratio 的值
            ratio = kwargs['ratio']
            # 初始化两个空字典
            reps = {}
            sifted = {}
            
            # 按照自由符号的长度进行分组
            sifted = sift(ordered([
                (i.free_symbols, i) for i in eqs]),
                lambda x: len(x[0]))
            eqs = []
            nonlineqs = []
            
            # 当 sifted 中存在长度为 1 的分组时循环执行以下操作
            while 1 in sifted:
                for free, e in sifted.pop(1):
                    x = free.pop()
                    # 如果 e 的左侧不等于 x 或者 x 在 e 的右侧的自由符号中，且 x 不在 reps 中
                    # 尝试线性系数化简操作
                    if (e.lhs != x or x in e.rhs.free_symbols) and x not in reps:
                        try:
                            m, b = linear_coeffs(
                                Add(e.lhs, -e.rhs, evaluate=False), x)
                            enew = e.func(x, -b/m)
                            # 根据 measure 的比例判断是否采用新的简化结果
                            if measure(enew) <= ratio * measure(e):
                                e = enew
                            else:
                                eqs.append(e)
                                continue
                        except ValueError:
                            pass
                    # 如果 x 已经在 reps 中
                    if x in reps:
                        eqs.append(e.subs(x, reps[x]))
                    # 如果 e 的左侧等于 x 且 x 不在 e 的右侧自由符号中
                    elif e.lhs == x and x not in e.rhs.free_symbols:
                        reps[x] = e.rhs
                        eqs.append(e)
                    else:
                        # x 目前尚未确定，但可能稍后确定
                        nonlineqs.append(e)
                
                # 重新组织 sifted 字典
                resifted = defaultdict(list)
                for k in sifted:
                    for f, e in sifted[k]:
                        e = e.xreplace(reps)
                        f = e.free_symbols
                        resifted[len(f)].append((f, e))
                sifted = resifted
            
            # 将 resifted 中的内容添加到 eqs 中
            for k in sifted:
                eqs.extend([e for f, e in sifted[k]])
            
            # 将 nonlineqs 和 other 中的元素经过 reps 替换后添加到对应列表中
            nonlineqs = [ei.subs(reps) for ei in nonlineqs]
            other = [ei.subs(reps) for ei in other]
            
            # 更新 rv 的内容，将 eqs、nonlineqs、other 拼接为一个新的 And 对象
            rv = rv.func(*([i.canonical for i in (eqs + nonlineqs + other)] + nonRel))
        
        # 获取简化模式的列表
        patterns = _simplify_patterns_and()
        threeterm_patterns = _simplify_patterns_and3()
        
        # 应用基于模式的简化操作，返回简化后的结果
        return _apply_patternbased_simplification(rv, patterns,
                                                  measure, false,
                                                  threeterm_patterns=threeterm_patterns)
    # 计算表达式的交集，将结果作为集合返回
    def _eval_as_set(self):
        # 导入 Intersection 类用于计算集合的交集
        from sympy.sets.sets import Intersection
        # 对表达式中每个参数调用 as_set() 方法，得到其对应的集合表示
        return Intersection(*[arg.as_set() for arg in self.args])

    # 将逻辑非操作重写为逻辑 NOR（逻辑或非）操作
    def _eval_rewrite_as_Nor(self, *args, **kwargs):
        # 对每个参数取逻辑非操作，并将结果作为 NOR 操作的参数
        return Nor(*[Not(arg) for arg in self.args])

    # 将逻辑表达式转换为 Algebraic Normal Form (ANF)，可选择是否递归转换子表达式
    def to_anf(self, deep=True):
        # 如果 deep=True，则递归将每个子表达式转换为 ANF，并将结果进行 AND 连接
        if deep:
            result = And._to_anf(*self.args, deep=deep)
            # 对 ANF 表达式中的 XOR 进行分配律变换
            return distribute_xor_over_and(result)
        # 如果 deep=False，则直接返回当前表达式
        return self
class Or(LatticeOp, BooleanFunction):
    """
    Logical OR function

    It evaluates its arguments in order, returning true immediately
    when an  argument is true, and false if they are all false.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Or
    >>> x | y
    x | y

    Notes
    =====

    The ``|`` operator is provided as a convenience, but note that its use
    here is different from its normal use in Python, which is bitwise
    or. Hence, ``Or(a, b)`` and ``a | b`` will return different things if
    ``a`` and ``b`` are integers.

    >>> Or(x, y).subs(x, 0)
    y

    """

    # Define zero and identity elements for the OR operation
    zero = true       # Zero element for OR is equivalent to true
    identity = false  # Identity element for OR is equivalent to false

    @classmethod
    def _new_args_filter(cls, args):
        """
        Filter and simplify arguments for the OR operation.

        Args:
        - args: List of arguments to be filtered and simplified.

        Returns:
        - List of filtered arguments for OR operation.
        """
        newargs = []
        rel = []
        # Simplify and filter binary Boolean arguments
        args = BooleanFunction.binary_check_and_simplify(*args)
        for x in args:
            if x.is_Relational:
                c = x.canonical
                # Handle canonical forms of relational expressions
                if c in rel:
                    continue
                nc = c.negated.canonical
                if any(r == nc for r in rel):
                    return [true]
                rel.append(c)
            newargs.append(x)
        # Apply filter for lattice operations
        return LatticeOp._new_args_filter(newargs, Or)

    def _eval_subs(self, old, new):
        """
        Substitute old expressions with new ones in the OR operation.

        Args:
        - old: Expression to be replaced.
        - new: New expression to replace old.

        Returns:
        - Simplified OR expression after substitution.
        """
        args = []
        bad = None
        for i in self.args:
            try:
                i = i.subs(old, new)
            except TypeError:
                # Handle TypeError during substitution
                if bad is None:
                    bad = i
                continue
            if i == True:
                return true
            elif i != False:
                args.append(i)
        if bad is not None:
            # Raise TypeError if encountered during substitution
            bad.subs(old, new)
        # Replace parts of the arguments with new if all are present
        if isinstance(old, Or):
            old_set = set(old.args)
            if old_set.issubset(args):
                args = set(args) - old_set
                args.add(new)

        return self.func(*args)

    def _eval_as_set(self):
        """
        Evaluate the OR operation as a set union.

        Returns:
        - Union of sets corresponding to the OR operation arguments.
        """
        from sympy.sets.sets import Union
        return Union(*[arg.as_set() for arg in self.args])

    def _eval_rewrite_as_Nand(self, *args, **kwargs):
        """
        Rewrite OR operation as a NAND operation.

        Args:
        - args: Arguments to rewrite as NAND.

        Returns:
        - Equivalent NAND expression.
        """
        return Nand(*[Not(arg) for arg in self.args])

    def _eval_simplify(self, **kwargs):
        """
        Simplify the OR operation.

        Args:
        - kwargs: Additional arguments for simplification.

        Returns:
        - Simplified OR expression.
        """
        from sympy.core.relational import Le, Ge, Eq
        lege = self.atoms(Le, Ge)
        if lege:
            reps = {i: self.func(
                Eq(i.lhs, i.rhs), i.strict) for i in lege}
            return self.xreplace(reps)._eval_simplify(**kwargs)
        # Standard simplify using predefined patterns
        rv = super()._eval_simplify(**kwargs)
        if not isinstance(rv, Or):
            return rv
        patterns = _simplify_patterns_or()
        return _apply_patternbased_simplification(rv, patterns,
                                                  kwargs['measure'], true)
    # 定义一个方法 `to_anf`，接受一个布尔值参数 `deep`，默认为 True
    def to_anf(self, deep=True):
        # 生成一个从 1 到 self.args 的长度加 1 的范围对象，并赋值给 args
        args = range(1, len(self.args) + 1)
        # 对 args 中的每个元素 j，生成 self.args 的长度为 j 的组合，返回生成器对象
        args = (combinations(self.args, j) for j in args)
        # 将多个生成器对象合并成一个，返回一个迭代器，即 args 是 self.args 所有可能的子集（幂集）
        args = chain.from_iterable(args)  # powerset
        # 对 args 中的每个元素 arg，生成一个 And 对象
        args = (And(*arg) for arg in args)
        # 如果 deep 为 True，则对 args 中的每个元素 x，递归调用 to_anf 方法；否则保持不变
        args = (to_anf(x, deep=deep) if deep else x for x in args)
        # 返回一个 Xor 对象，包含 args 中所有元素，并保留 True 值
        return Xor(*list(args), remove_true=False)
class Not(BooleanFunction):
    """
    Logical Not function (negation)

    Returns ``true`` if the statement is ``false`` or ``False``.
    Returns ``false`` if the statement is ``true`` or ``True``.

    Examples
    ========

    >>> from sympy import Not, And, Or
    >>> from sympy.abc import x, A, B
    >>> Not(True)
    False
    >>> Not(False)
    True
    >>> Not(And(True, False))
    True
    >>> Not(Or(True, False))
    False
    >>> Not(And(And(True, x), Or(x, False)))
    ~x
    >>> ~x
    ~x
    >>> Not(And(Or(A, B), Or(~A, ~B)))
    ~((A | B) & (~A | ~B))

    Notes
    =====

    - The ``~`` operator is provided as a convenience, but note that its use
      here is different from its normal use in Python, which is bitwise
      not. In particular, ``~a`` and ``Not(a)`` will be different if ``a`` is
      an integer. Furthermore, since bools in Python subclass from ``int``,
      ``~True`` is the same as ``~1`` which is ``-2``, which has a boolean
      value of True.  To avoid this issue, use the SymPy boolean types
      ``true`` and ``false``.

    - As of Python 3.12, the bitwise not operator ``~`` used on a
      Python ``bool`` is deprecated and will emit a warning.

    >>> from sympy import true
    >>> ~True  # doctest: +SKIP
    -2
    >>> ~true
    False

    """

    is_Not = True  # 设置类属性 is_Not 为 True，表示这是一个 Not 类型的对象

    @classmethod
    def eval(cls, arg):
        # 如果参数是一个数字或者 True/False 值，返回相应的逻辑非结果
        if isinstance(arg, Number) or arg in (True, False):
            return false if arg else true
        # 如果参数是 Not 类型的对象，则返回它的参数，实现逻辑非的简化
        if arg.is_Not:
            return arg.args[0]
        # 如果参数是关系表达式 (Relational)，返回其否定形式
        # 简化关系对象
        if arg.is_Relational:
            return arg.negated

    def _eval_as_set(self):
        """
        Rewrite logic operators and relationals in terms of real sets.

        Examples
        ========

        >>> from sympy import Not, Symbol
        >>> x = Symbol('x')
        >>> Not(x > 0).as_set()
        Interval(-oo, 0)
        """
        # 将逻辑运算符和关系表达式重写为实数集合的形式
        return self.args[0].as_set().complement(S.Reals)
    # 将逻辑表达式转换为否范式（NNF），默认简化结果
    def to_nnf(self, simplify=True):
        # 如果当前表达式是文字（变量或其否定），直接返回自身
        if is_literal(self):
            return self

        # 获取表达式的第一个参数，通常是表达式中的子表达式
        expr = self.args[0]

        # 获取子表达式的函数和参数
        func, args = expr.func, expr.args

        # 如果子表达式是合取（And）操作
        if func == And:
            # 对子表达式中的每个参数取否，然后转换为析取（Or）操作
            return Or._to_nnf(*[Not(arg) for arg in args], simplify=simplify)

        # 如果子表达式是析取（Or）操作
        if func == Or:
            # 对子表达式中的每个参数取否，然后转换为合取（And）操作
            return And._to_nnf(*[Not(arg) for arg in args], simplify=simplify)

        # 如果子表达式是蕴含（Implies）操作
        if func == Implies:
            # 将蕴含转换为合取形式的否范式
            a, b = args
            return And._to_nnf(a, Not(b), simplify=simplify)

        # 如果子表达式是等价（Equivalent）操作
        if func == Equivalent:
            # 将等价转换为合取形式的否范式
            return And._to_nnf(Or(*args), Or(*[Not(arg) for arg in args]),
                               simplify=simplify)

        # 如果子表达式是异或（Xor）操作
        if func == Xor:
            # 对每对参数取否，生成所有可能的析取子句，并转换为合取形式的否范式
            result = []
            for i in range(1, len(args)+1, 2):
                for neg in combinations(args, i):
                    clause = [Not(s) if s in neg else s for s in args]
                    result.append(Or(*clause))
            return And._to_nnf(*result, simplify=simplify)

        # 如果子表达式是条件表达式（ITE）操作
        if func == ITE:
            # 将条件表达式转换为合取形式的否范式
            a, b, c = args
            return And._to_nnf(Or(a, Not(c)), Or(Not(a), Not(b)), simplify=simplify)

        # 若子表达式的操作不合法，抛出异常
        raise ValueError("Illegal operator %s in expression" % func)

    # 将逻辑表达式转换为代数正范式（ANF），默认深度转换
    def to_anf(self, deep=True):
        return Xor._to_anf(true, self.args[0], deep=deep)
# 定义一个名为 Xor 的类，继承自 BooleanFunction
class Xor(BooleanFunction):
    """
    Logical XOR (exclusive OR) function.

    Returns True if an odd number of the arguments are True and the rest are
    False.

    Returns False if an even number of the arguments are True and the rest are
    False.

    Examples
    ========

    >>> from sympy.logic.boolalg import Xor
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> Xor(True, False)
    True
    >>> Xor(True, True)
    False
    >>> Xor(True, False, True, True, False)
    True
    >>> Xor(True, False, True, False)
    False
    >>> x ^ y
    x ^ y

    Notes
    =====

    The ``^`` operator is provided as a convenience, but note that its use
    here is different from its normal use in Python, which is bitwise xor. In
    particular, ``a ^ b`` and ``Xor(a, b)`` will be different if ``a`` and
    ``b`` are integers.

    >>> Xor(x, y).subs(y, 0)
    x

    """
    
    # 定义类的构造函数 __new__，处理逻辑 XOR 的计算和优化
    def __new__(cls, *args, remove_true=True, **kwargs):
        # 初始化一个空的集合 argset
        argset = set()
        # 调用父类的构造方法，创建新的对象 obj
        obj = super().__new__(cls, *args, **kwargs)
        # 遍历 obj 的参数列表 _args
        for arg in obj._args:
            # 如果参数是数值或者 True/False
            if isinstance(arg, Number) or arg in (True, False):
                # 如果参数是 True，则替换为 true 对象；否则跳过
                if arg:
                    arg = true
                else:
                    continue
            # 如果参数是 Xor 类型
            if isinstance(arg, Xor):
                # 将 arg 中的每个参数加入或移除 argset
                for a in arg.args:
                    argset.remove(a) if a in argset else argset.add(a)
            # 如果参数已经在 argset 中，则移除；否则添加
            elif arg in argset:
                argset.remove(arg)
            else:
                argset.add(arg)
        # 初始化一个空列表 rel，用来存储关系式的元组
        rel = [(r, r.canonical, r.negated.canonical)
               for r in argset if r.is_Relational]
        # 初始化 odd 为 False，用来记录补集对的数量是否为奇数
        odd = False  # is number of complimentary pairs odd? start 0 -> False
        # 初始化一个空列表 remove，用来存储需要移除的补集对
        remove = []
        # 遍历 rel 列表
        for i, (r, c, nc) in enumerate(rel):
            # 从当前位置 i 开始遍历后面的元素
            for j in range(i + 1, len(rel)):
                rj, cj = rel[j][:2]
                # 如果找到补集对，则将 odd 取反，并退出内循环
                if cj == nc:
                    odd = not odd
                    break
                # 如果找到相同的关系式，则退出内循环
                elif cj == c:
                    break
            else:
                continue
            # 将找到的补集对加入 remove 列表
            remove.append((r, rj))
        # 如果 odd 为 True，则在 argset 中移除 true；否则添加 true 到 argset
        if odd:
            argset.remove(true) if true in argset else argset.add(true)
        # 遍历 remove 列表，移除 argset 中对应的元素对
        for a, b in remove:
            argset.remove(a)
            argset.remove(b)
        # 根据 argset 的长度返回不同的结果
        if len(argset) == 0:
            return false
        elif len(argset) == 1:
            return argset.pop()
        elif True in argset and remove_true:
            argset.remove(True)
            return Not(Xor(*argset))
        else:
            obj._args = tuple(ordered(argset))
            obj._argset = frozenset(argset)
            return obj

    # 定义一个属性 args，用来返回按顺序排列的参数列表
    # XXX: This should be cached on the object rather than using cacheit
    # Maybe it can be computed in __new__?
    @property  # type: ignore
    @cacheit
    def args(self):
        return tuple(ordered(self._argset))
    # 将表达式转换为否定标准形（NNF），并可选地进行简化
    def to_nnf(self, simplify=True):
        # 初始化空列表存放转换后的表达式
        args = []
        # 从0开始，每两个元素为一组迭代self.args
        for i in range(0, len(self.args)+1, 2):
            # 生成所有self.args中取反的组合
            for neg in combinations(self.args, i):
                # 根据取反组合创建新的子句列表
                clause = [Not(s) if s in neg else s for s in self.args]
                # 将子句转换为Or表达式，并添加到args列表中
                args.append(Or(*clause))
        # 调用And类的静态方法_to_nnf，传入args列表，并可选地进行简化
        return And._to_nnf(*args, simplify=simplify)

    # 将表达式重写为Or的形式
    def _eval_rewrite_as_Or(self, *args, **kwargs):
        # 获取当前对象的参数列表
        a = self.args
        # 调用_convert_to_varsSOP函数，将获取的奇校验项转换为变量
        return Or(*[_convert_to_varsSOP(x, self.args)
                    for x in _get_odd_parity_terms(len(a))])

    # 将表达式重写为And的形式
    def _eval_rewrite_as_And(self, *args, **kwargs):
        # 获取当前对象的参数列表
        a = self.args
        # 调用_convert_to_varsPOS函数，将获取的偶校验项转换为变量
        return And(*[_convert_to_varsPOS(x, self.args)
                     for x in _get_even_parity_terms(len(a))])

    # 对表达式进行简化操作
    def _eval_simplify(self, **kwargs):
        # 根据标准的简化逻辑（simplify_logic）简化每个参数的部分表达式
        rv = self.func(*[a.simplify(**kwargs) for a in self.args])
        # 将简化后的表达式转换为布尔代数的标准形式（ANF）
        rv = rv.to_anf()
        # 如果rv不是Xor类型，这通常不应该发生
        if not isinstance(rv, Xor):  # This shouldn't really happen here
            return rv
        # 获取简化模式_patterns_xor
        patterns = _simplify_patterns_xor()
        # 应用基于模式的简化函数_apply_patternbased_simplification，传入参数rv、patterns、kwargs['measure']和None
        return _apply_patternbased_simplification(rv, patterns,
                                                  kwargs['measure'], None)

    # 替换表达式中的旧部分为新部分
    def _eval_subs(self, old, new):
        # 如果old是Xor类型，则替换参数中的相应部分为新的参数
        if isinstance(old, Xor):
            # 将参数转换为集合old_set，检查是否所有旧部分都在self.args中
            old_set = set(old.args)
            if old_set.issubset(self.args):
                # 从self.args中去除旧部分，加入新部分，并调用self.func创建新的表达式
                args = set(self.args) - old_set
                args.add(new)
                return self.func(*args)
class Nand(BooleanFunction):
    """
    Logical NAND function.

    It evaluates its arguments in order, giving True immediately if any
    of them are False, and False if they are all True.

    Returns True if any of the arguments are False
    Returns False if all arguments are True

    Examples
    ========

    >>> from sympy.logic.boolalg import Nand
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> Nand(False, True)
    True
    >>> Nand(True, True)
    False
    >>> Nand(x, y)
    ~(x & y)

    """
    @classmethod
    def eval(cls, *args):
        # 返回逻辑 NAND 的结果，使用逻辑 AND 和 NOT 来实现
        return Not(And(*args))


class Nor(BooleanFunction):
    """
    Logical NOR function.

    It evaluates its arguments in order, giving False immediately if any
    of them are True, and True if they are all False.

    Returns False if any argument is True
    Returns True if all arguments are False

    Examples
    ========

    >>> from sympy.logic.boolalg import Nor
    >>> from sympy import symbols
    >>> x, y = symbols('x y')

    >>> Nor(True, False)
    False
    >>> Nor(True, True)
    False
    >>> Nor(False, True)
    False
    >>> Nor(False, False)
    True
    >>> Nor(x, y)
    ~(x | y)

    """
    @classmethod
    def eval(cls, *args):
        # 返回逻辑 NOR 的结果，使用逻辑 OR 和 NOT 来实现
        return Not(Or(*args))


class Xnor(BooleanFunction):
    """
    Logical XNOR function.

    Returns False if an odd number of the arguments are True and the rest are
    False.

    Returns True if an even number of the arguments are True and the rest are
    False.

    Examples
    ========

    >>> from sympy.logic.boolalg import Xnor
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> Xnor(True, False)
    False
    >>> Xnor(True, True)
    True
    >>> Xnor(True, False, True, True, False)
    False
    >>> Xnor(True, False, True, False)
    True

    """
    @classmethod
    def eval(cls, *args):
        # 返回逻辑 XNOR 的结果，使用逻辑 XOR 和 NOT 来实现
        return Not(Xor(*args))


class Implies(BooleanFunction):
    r"""
    Logical implication.

    A implies B is equivalent to if A then B. Mathematically, it is written
    as `A \Rightarrow B` and is equivalent to `\neg A \vee B` or ``~A | B``.

    Accepts two Boolean arguments; A and B.
    Returns False if A is True and B is False
    Returns True otherwise.

    Examples
    ========

    >>> from sympy.logic.boolalg import Implies
    >>> from sympy import symbols
    >>> x, y = symbols('x y')

    >>> Implies(True, False)
    False
    >>> Implies(False, False)
    True
    >>> Implies(True, True)
    True
    >>> Implies(False, True)
    True
    >>> x >> y
    Implies(x, y)
    >>> y << x
    Implies(x, y)

    Notes
    =====

    The ``>>`` and ``<<`` operators are provided as a convenience, but note
    that their use here is different from their normal use in Python, which is
    bit shifts. Hence, ``Implies(a, b)`` and ``a >> b`` will return different
    things if ``a`` and ``b`` are integers.  In particular, since Python
    """

    @classmethod
    def eval(cls, *args):
        # 返回逻辑 IMPLIES 的结果，使用逻辑 NOT、OR 和逻辑操作符来实现
        return Or(Not(args[0]), args[1])
    """
    considers ``True`` and ``False`` to be integers, ``True >> True`` will be
    the same as ``1 >> 1``, i.e., 0, which has a truth value of False.  To
    avoid this issue, use the SymPy objects ``true`` and ``false``.

    >>> from sympy import true, false
    >>> True >> False
    1
    >>> true >> false
    False
    """

    @classmethod
    def eval(cls, *args):
        try:
            newargs = []
            # 遍历参数列表，将整数或者 0、1 转换为布尔值
            for x in args:
                if isinstance(x, Number) or x in (0, 1):
                    newargs.append(bool(x))
                else:
                    newargs.append(x)
            A, B = newargs  # 将处理后的参数列表解构为 A 和 B
        except ValueError:
            # 抛出值错误异常，指示错误的操作数数量
            raise ValueError(
                "%d operand(s) used for an Implies "
                "(pairs are required): %s" % (len(args), str(args)))
        # 根据 A 和 B 的类型进行逻辑推导
        if A in (True, False) or B in (True, False):
            # 如果 A 或 B 是布尔值，返回逻辑或的结果
            return Or(Not(A), B)
        elif A == B:
            # 如果 A 等于 B，返回 true 对象
            return true
        elif A.is_Relational and B.is_Relational:
            # 如果 A 和 B 都是关系表达式，检查其规范化形式
            if A.canonical == B.canonical:
                return true
            if A.negated.canonical == B.canonical:
                return B
        else:
            # 对于其他情况，调用基类的构造方法返回新对象
            return Basic.__new__(cls, *args)

    def to_nnf(self, simplify=True):
        # 获取当前逻辑表达式的参数
        a, b = self.args
        # 将当前表达式转换为否定范式
        return Or._to_nnf(Not(a), b, simplify=simplify)

    def to_anf(self, deep=True):
        # 获取当前逻辑表达式的参数
        a, b = self.args
        # 将当前表达式转换为布尔代数的析取范式
        return Xor._to_anf(true, a, And(a, b), deep=deep)
class Equivalent(BooleanFunction):
    """
    Equivalence relation.

    ``Equivalent(A, B)`` is True iff A and B are both True or both False.

    Returns True if all of the arguments are logically equivalent.
    Returns False otherwise.

    For two arguments, this is equivalent to :py:class:`~.Xnor`.

    Examples
    ========

    >>> from sympy.logic.boolalg import Equivalent, And
    >>> from sympy.abc import x
    >>> Equivalent(False, False, False)
    True
    >>> Equivalent(True, False, False)
    False
    >>> Equivalent(x, And(x, True))
    True

    """
    
    def __new__(cls, *args, **options):
        # Import necessary components from sympy
        from sympy.core.relational import Relational
        # Sympify all arguments (convert to SymPy expressions)
        args = [_sympify(arg) for arg in args]

        # Create a set of arguments to handle duplicates and simplify
        argset = set(args)
        for x in args:
            # Convert Numbers and True/False into bools (0/1)
            if isinstance(x, Number) or x in [True, False]:  # Includes 0, 1
                argset.discard(x)
                argset.add(bool(x))

        # List to store relational expressions
        rel = []
        for r in argset:
            if isinstance(r, Relational):
                # Store canonical and negated canonical forms of relations
                rel.append((r, r.canonical, r.negated.canonical))
        
        # List to remove redundant relations
        remove = []
        for i, (r, c, nc) in enumerate(rel):
            for j in range(i + 1, len(rel)):
                rj, cj = rel[j][:2]
                # If canonical forms match, mark for removal
                if cj == nc:
                    return false  # If redundant relation found, return False
                elif cj == c:
                    remove.append((r, rj))
                    break
        
        # Remove redundant relations
        for a, b in remove:
            argset.remove(a)
            argset.remove(b)
            argset.add(True)  # Add True to handle the found equivalence
        
        # If only one argument or all are True, return True
        if len(argset) <= 1:
            return true
        # If True present, return conjunction of remaining arguments
        if True in argset:
            argset.discard(True)
            return And(*argset)
        # If False present, return conjunction of negated arguments
        if False in argset:
            argset.discard(False)
            return And(*[Not(arg) for arg in argset])
        
        # Create a frozenset of remaining arguments and instantiate the object
        _args = frozenset(argset)
        obj = super().__new__(cls, _args)
        obj._argset = _args
        return obj

    # Property to return ordered tuple of arguments
    @property  # type: ignore
    @cacheit
    def args(self):
        return tuple(ordered(self._argset))

    # Method to convert to Negation Normal Form (NNF)
    def to_nnf(self, simplify=True):
        args = []
        # Create NNF clauses using De Morgan's laws
        for a, b in zip(self.args, self.args[1:]):
            args.append(Or(Not(a), b))
        args.append(Or(Not(self.args[-1]), self.args[0]))
        return And._to_nnf(*args, simplify=simplify)

    # Method to convert to Algebraic Normal Form (ANF)
    def to_anf(self, deep=True):
        # Create conjunction and its negation for ANF transformation
        a = And(*self.args)
        b = And(*[to_anf(Not(arg), deep=False) for arg in self.args])
        b = distribute_xor_over_and(b)
        return Xor._to_anf(a, b, deep=deep)
    >>> ITE(True, False, True)
    False
    >>> ITE(Or(True, False), And(True, True), Xor(True, True))
    True
    >>> ITE(x, y, z)
    ITE(x, y, z)
    >>> ITE(True, x, y)
    x
    >>> ITE(False, x, y)
    y
    >>> ITE(x, y, y)
    y

    Trying to use non-Boolean args will generate a TypeError:

    >>> ITE(True, [], ())
    Traceback (most recent call last):
    ...
    TypeError: expecting bool, Boolean or ITE, not `[]`

"""
# 定义一个名为 ITE 的类，继承自 BooleanFunction
def __new__(cls, *args, **kwargs):
    # 导入需要的符号和关系运算
    from sympy.core.relational import Eq, Ne
    # 检查参数数量，必须为 3 个
    if len(args) != 3:
        raise ValueError('expecting exactly 3 args')
    # 将参数解包
    a, b, c = args
    # 检查是否使用了二元符号
    if isinstance(a, (Eq, Ne)):
        # 在这个上下文中，我们可以评估 Eq/Ne
        # 如果一个参数是二元符号，另一个是 true/false
        from sympy.logic.boolalg import as_Boolean
        b, c = map(as_Boolean, (b, c))
        # 获取所有二元符号的集合
        bin_syms = set().union(*[i.binary_symbols for i in (b, c)])
        # 如果在 a 的参数中找不到 bin_syms，就修改 a
        if len(set(a.args) - bin_syms) == 1:
            _a = a
            if a.lhs is true:
                a = a.rhs
            elif a.rhs is true:
                a = a.lhs
            elif a.lhs is false:
                a = Not(a.rhs)
            elif a.rhs is false:
                a = Not(a.lhs)
            else:
                # 二元运算只能等于 True 或 False
                a = false
            # 如果是 Ne 类型，则取反
            if isinstance(_a, Ne):
                a = Not(a)
    else:
        # 对非二元符号的情况进行简化和检查
        from sympy.logic.boolalg import BooleanFunction
        a, b, c = BooleanFunction.binary_check_and_simplify(
            a, b, c)
    # 初始化返回值
    rv = None
    # 如果 evaluate 参数为 True，则评估 ITE 的结果
    if kwargs.get('evaluate', True):
        rv = cls.eval(a, b, c)
    # 如果 rv 仍为 None，则创建一个新的 ITE 对象
    if rv is None:
        rv = BooleanFunction.__new__(cls, a, b, c, evaluate=False)
    return rv

@classmethod
def eval(cls, *args):
    # 导入需要的符号和关系运算
    from sympy.core.relational import Eq, Ne
    # 检查参数是否给出了唯一的结果
    a, b, c = args
    if isinstance(a, (Ne, Eq)):
        _a = a
        # 如果 true 在 a 的参数中，则取 a 的 lhs 或 rhs
        if true in a.args:
            a = a.lhs if a.rhs is true else a.rhs
        # 如果 false 在 a 的参数中，则取 a 的 lhs 或 rhs 的非
        elif false in a.args:
            a = Not(a.lhs) if a.rhs is false else Not(a.rhs)
        else:
            _a = None
        # 如果 _a 不为 None 并且是 Ne 类型，则取 a 的非
        if _a is not None and isinstance(_a, Ne):
            a = Not(a)
    # 根据 a 的值返回 b 或 c
    if a is true:
        return b
    if a is false:
        return c
    # 如果 b 等于 c，则返回 b
    if b == c:
        return b
    else:
        # 否则根据条件可能返回表达式的结果
        if b is true and c is false:
            return a
        if b is false and c is true:
            return Not(a)
    # 如果结果不是 args 的顺序，则创建一个新的 ITE 对象
    if [a, b, c] != args:
        return cls(a, b, c, evaluate=False)

def to_nnf(self, simplify=True):
    # 获取 ITE 的参数
    a, b, c = self.args
    # 将 ITE 转换为否定范式（NNF）
    return And._to_nnf(Or(Not(a), b), Or(a, c), simplify=simplify)
    # 定义一个方法 `_eval_as_set`，用于计算对象经过 NNF（Negation Normal Form）处理后的集合表示
    def _eval_as_set(self):
        # 调用对象的 `to_nnf` 方法，将其转换为 NNF
        return self.to_nnf().as_set()

    # 定义一个方法 `_eval_rewrite_as_Piecewise`，将对象重写为 Piecewise 形式
    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        # 导入 Piecewise 类
        from sympy.functions.elementary.piecewise import Piecewise
        # 返回一个 Piecewise 对象，根据条件 args[0] 决定选择 args[1] 或者 args[2]
        return Piecewise((args[1], args[0]), (args[2], True))
class Exclusive(BooleanFunction):
    """
    True if only one or no argument is true.

    ``Exclusive(A, B, C)`` is equivalent to ``~(A & B) & ~(A & C) & ~(B & C)``.

    For two arguments, this is equivalent to :py:class:`~.Xor`.

    Examples
    ========

    >>> from sympy.logic.boolalg import Exclusive
    >>> Exclusive(False, False, False)
    True
    >>> Exclusive(False, True, False)
    True
    >>> Exclusive(False, True, True)
    False

    """
    @classmethod
    def eval(cls, *args):
        # 创建一个空列表来存储生成的逻辑表达式
        and_args = []
        # 使用itertools.combinations生成每对参数组合，并对每对参数a和b执行以下操作：
        for a, b in combinations(args, 2):
            # 生成逻辑表达式Not(And(a, b))，并添加到and_args列表中
            and_args.append(Not(And(a, b)))
        # 使用And函数将所有生成的逻辑表达式连接起来，并返回结果
        return And(*and_args)


# end class definitions. Some useful methods


def conjuncts(expr):
    """Return a list of the conjuncts in ``expr``.

    Examples
    ========

    >>> from sympy.logic.boolalg import conjuncts
    >>> from sympy.abc import A, B
    >>> conjuncts(A & B)
    frozenset({A, B})
    >>> conjuncts(A | B)
    frozenset({A | B})

    """
    # 使用And.make_args方法从逻辑表达式expr中提取所有的合取项（conjuncts）并返回
    return And.make_args(expr)


def disjuncts(expr):
    """Return a list of the disjuncts in ``expr``.

    Examples
    ========

    >>> from sympy.logic.boolalg import disjuncts
    >>> from sympy.abc import A, B
    >>> disjuncts(A | B)
    frozenset({A, B})
    >>> disjuncts(A & B)
    frozenset({A & B})

    """
    # 使用Or.make_args方法从逻辑表达式expr中提取所有的析取项（disjuncts）并返回
    return Or.make_args(expr)


def distribute_and_over_or(expr):
    """
    Given a sentence ``expr`` consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in CNF.

    Examples
    ========

    >>> from sympy.logic.boolalg import distribute_and_over_or, And, Or, Not
    >>> from sympy.abc import A, B, C
    >>> distribute_and_over_or(Or(A, And(Not(B), Not(C))))
    (A | ~B) & (A | ~C)

    """
    # 调用_distribute函数来将逻辑表达式expr中的合取项分配到析取项上，并返回结果
    return _distribute((expr, And, Or))


def distribute_or_over_and(expr):
    """
    Given a sentence ``expr`` consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in DNF.

    Note that the output is NOT simplified.

    Examples
    ========

    >>> from sympy.logic.boolalg import distribute_or_over_and, And, Or, Not
    >>> from sympy.abc import A, B, C
    >>> distribute_or_over_and(And(Or(Not(A), B), C))
    (B & C) | (C & ~A)

    """
    # 调用_distribute函数来将逻辑表达式expr中的析取项分配到合取项上，并返回结果
    return _distribute((expr, Or, And))


def distribute_xor_over_and(expr):
    """
    Given a sentence ``expr`` consisting of conjunction and
    exclusive disjunctions of literals, return an
    equivalent exclusive disjunction.

    Note that the output is NOT simplified.

    Examples
    ========

    >>> from sympy.logic.boolalg import distribute_xor_over_and, And, Xor, Not
    >>> from sympy.abc import A, B, C
    >>> distribute_xor_over_and(And(Xor(Not(A), B), C))
    (B & C) ^ (C & ~A)
    """
    # 调用_distribute函数来将逻辑表达式expr中的异或运算分配到合取项上，并返回结果
    return _distribute((expr, Xor, And))


def _distribute(info):
    """
    Distributes ``info[1]`` over ``info[2]`` with respect to ``info[0]``.
    """
    # 内部函数，根据info中的信息将逻辑运算符分配到相应的合取项或析取项上，并返回结果
    # 如果 info[0] 是 info[2] 的实例
    if isinstance(info[0], info[2]):
        # 遍历 info[0] 的参数
        for arg in info[0].args:
            # 如果参数是 info[1] 的实例，则将其赋值给 conj 并中断循环
            if isinstance(arg, info[1]):
                conj = arg
                break
        else:
            # 如果没有找到符合条件的参数，则返回 info[0]
            return info[0]
        # 从 info[0].args 中除了 conj 外的参数构成 rest
        rest = info[2](*[a for a in info[0].args if a is not conj])
        # 对 conj 的每个参数应用 _distribute 函数，并用结果构成列表，传递给 info[1] 构造新对象，remove_true 参数为 False
        return info[1](*list(map(_distribute,
                                 [(info[2](c, rest), info[1], info[2])
                                  for c in conj.args])), remove_true=False)
    # 如果 info[0] 是 info[1] 的实例
    elif isinstance(info[0], info[1]):
        # 对 info[0].args 中的每个元素应用 _distribute 函数，用结果构成列表，传递给 info[1] 构造新对象，remove_true 参数为 False
        return info[1](*list(map(_distribute,
                                 [(x, info[1], info[2])
                                  for x in info[0].args])),
                       remove_true=False)
    else:
        # 如果 info[0] 不是 info[2] 或 info[1] 的实例，则直接返回 info[0]
        return info[0]
# 将逻辑表达式转换为代数正常形式（ANF）。
def to_anf(expr, deep=True):
    r"""
    Converts expr to Algebraic Normal Form (ANF).

    ANF is a canonical normal form, which means that two
    equivalent formulas will convert to the same ANF.

    A logical expression is in ANF if it has the form

    .. math:: 1 \oplus a \oplus b \oplus ab \oplus abc

    i.e. it can be:
        - purely true,
        - purely false,
        - conjunction of variables,
        - exclusive disjunction.

    The exclusive disjunction can only contain true, variables
    or conjunction of variables. No negations are permitted.

    If ``deep`` is ``False``, arguments of the boolean
    expression are considered variables, i.e. only the
    top-level expression is converted to ANF.

    Examples
    ========
    >>> from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent
    >>> from sympy.logic.boolalg import to_anf
    >>> from sympy.abc import A, B, C
    >>> to_anf(Not(A))
    A ^ True
    >>> to_anf(And(Or(A, B), Not(C)))
    A ^ B ^ (A & B) ^ (A & C) ^ (B & C) ^ (A & B & C)
    >>> to_anf(Implies(Not(A), Equivalent(B, C)), deep=False)
    True ^ ~A ^ (~A & (Equivalent(B, C)))

    """
    # 将表达式转换为符号表示（如果尚未是）
    expr = sympify(expr)

    # 如果表达式已经是代数正常形式（ANF），直接返回
    if is_anf(expr):
        return expr
    # 否则调用表达式的方法将其转换为 ANF 形式
    return expr.to_anf(deep=deep)


# 将逻辑表达式转换为否定正常形式（NNF）。
def to_nnf(expr, simplify=True):
    """
    Converts ``expr`` to Negation Normal Form (NNF).

    A logical expression is in NNF if it
    contains only :py:class:`~.And`, :py:class:`~.Or` and :py:class:`~.Not`,
    and :py:class:`~.Not` is applied only to literals.
    If ``simplify`` is ``True``, the result contains no redundant clauses.

    Examples
    ========

    >>> from sympy.abc import A, B, C, D
    >>> from sympy.logic.boolalg import Not, Equivalent, to_nnf
    >>> to_nnf(Not((~A & ~B) | (C & D)))
    (A | B) & (~C | ~D)
    >>> to_nnf(Equivalent(A >> B, B >> A))
    (A | ~B | (A & ~B)) & (B | ~A | (B & ~A))

    """
    # 如果表达式已经是否定正常形式（NNF），直接返回
    if is_nnf(expr, simplify):
        return expr
    # 否则调用表达式的方法将其转换为 NNF 形式
    return expr.to_nnf(simplify)


# 将命题逻辑句子转换为合取范式（CNF）。
def to_cnf(expr, simplify=False, force=False):
    """
    Convert a propositional logical sentence ``expr`` to conjunctive normal
    form: ``((A | ~B | ...) & (B | C | ...) & ...)``.
    If ``simplify`` is ``True``, ``expr`` is evaluated to its simplest CNF
    form using the Quine-McCluskey algorithm; this may take a long
    time. If there are more than 8 variables the ``force`` flag must be set
    to ``True`` to simplify (default is ``False``).

    Examples
    ========

    >>> from sympy.logic.boolalg import to_cnf
    >>> from sympy.abc import A, B, D
    >>> to_cnf(~(A | B) | D)
    (D | ~A) & (D | ~B)
    >>> to_cnf((A | B) & (A | ~A), True)
    A | B

    """
    # 将表达式转换为符号表示（如果尚未是）
    expr = sympify(expr)
    # 如果表达式不是布尔函数，则直接返回
    if not isinstance(expr, BooleanFunction):
        return expr
    # 如果 simplify 参数为真，则执行下面的条件语句块
    if simplify:
        # 如果不强制简化且表达式中谓词的数量超过8个，则引发值错误异常
        if not force and len(_find_predicates(expr)) > 8:
            raise ValueError(filldedent('''
            To simplify a logical expression with more
            than 8 variables may take a long time and requires
            the use of `force=True`.
            '''))
        # 使用 simplify_logic 函数简化逻辑表达式为合取范式（CNF），强制为真
        return simplify_logic(expr, 'cnf', True, force=force)

    # 如果表达式已经是合取范式（CNF），则直接返回表达式本身
    if is_cnf(expr):
        return expr

    # 消除表达式中的蕴含符号
    expr = eliminate_implications(expr)
    # 对表达式进行分配律的转换
    res = distribute_and_over_or(expr)

    # 返回转换后的结果
    return res
# 将命题逻辑表达式 ``expr`` 转换为析取范式（DNF）：``((A & ~B & ...) | (B & C & ...) | ...)``。
# 如果 ``simplify`` 为 ``True``，则使用 Quine-McCluskey 算法将 ``expr`` 简化到最简DNF形式；
# 这可能需要很长时间。如果变量数超过8个，则必须将 ``force`` 标志设置为 ``True`` 才能简化（默认为 ``False``）。

def to_dnf(expr, simplify=False, force=False):
    """
    Convert a propositional logical sentence ``expr`` to disjunctive normal
    form: ``((A & ~B & ...) | (B & C & ...) | ...)``.
    If ``simplify`` is ``True``, ``expr`` is evaluated to its simplest DNF form using
    the Quine-McCluskey algorithm; this may take a long
    time. If there are more than 8 variables, the ``force`` flag must be set to
    ``True`` to simplify (default is ``False``).

    Examples
    ========

    >>> from sympy.logic.boolalg import to_dnf
    >>> from sympy.abc import A, B, C
    >>> to_dnf(B & (A | C))
    (A & B) | (B & C)
    >>> to_dnf((A & B) | (A & ~B) | (B & C) | (~B & C), True)
    A | C

    """
    # 将表达式转换为符号对象
    expr = sympify(expr)
    # 如果表达式不是布尔函数，则直接返回
    if not isinstance(expr, BooleanFunction):
        return expr

    # 如果需要简化
    if simplify:
        # 如果变量数超过8个且未设置force标志，则抛出异常
        if not force and len(_find_predicates(expr)) > 8:
            raise ValueError(filldedent('''
            To simplify a logical expression with more
            than 8 variables may take a long time and requires
            the use of `force=True`.'''))
        # 使用简化逻辑函数将表达式转换为DNF形式
        return simplify_logic(expr, 'dnf', True, force=force)

    # 如果已经是DNF形式，则直接返回
    if is_dnf(expr):
        return expr

    # 消除蕴含
    expr = eliminate_implications(expr)
    # 分配析取到合取上
    return distribute_or_over_and(expr)


# 检查 ``expr`` 是否处于代数正常形式（ANF）中。
# 逻辑表达式在ANF中，如果其形式是 ``1 \oplus a \oplus b \oplus ab \oplus abc``
# 即纯真、纯假、变量的合取或异或的合取。异或的合取只能包含真、变量或变量的合取。不允许有否定。
def is_anf(expr):
    r"""
    Checks if ``expr``  is in Algebraic Normal Form (ANF).

    A logical expression is in ANF if it has the form

    .. math:: 1 \oplus a \oplus b \oplus ab \oplus abc

    i.e. it is purely true, purely false, conjunction of
    variables or exclusive disjunction. The exclusive
    disjunction can only contain true, variables or
    conjunction of variables. No negations are permitted.

    Examples
    ========

    >>> from sympy.logic.boolalg import And, Not, Xor, true, is_anf
    >>> from sympy.abc import A, B, C
    >>> is_anf(true)
    True
    >>> is_anf(A)
    True
    >>> is_anf(And(A, B, C))
    True
    >>> is_anf(Xor(A, Not(B)))
    False

    """
    # 将表达式转换为符号对象
    expr = sympify(expr)

    # 如果是文字且不是Not类型，则返回True
    if is_literal(expr) and not isinstance(expr, Not):
        return True

    # 如果是合取
    if isinstance(expr, And):
        # 检查每个参数是否是符号
        for arg in expr.args:
            if not arg.is_Symbol:
                return False
        return True

    # 如果是异或
    elif isinstance(expr, Xor):
        # 检查每个参数是否是符号或符号的合取
        for arg in expr.args:
            if isinstance(arg, And):
                for a in arg.args:
                    if not a.is_Symbol:
                        return False
            elif is_literal(arg):
                if isinstance(arg, Not):
                    return False
            else:
                return False
        return True

    else:
        return False


# 检查 ``expr`` 是否在否定正常形式（NNF）中。
# 逻辑表达式在NNF中，如果仅包含：And、Or 和 Not，并且 Not 仅应用于文字。
def is_nnf(expr, simplified=True):
    """
    Checks if ``expr`` is in Negation Normal Form (NNF).

    A logical expression is in NNF if it
    contains only :py:class:`~.And`, :py:class:`~.Or` and :py:class:`~.Not`,
    and :py:class:`~.Not` is applied only to literals.

    """
    # 将表达式转换为符号对象
    expr = sympify(expr)

    # 如果表达式是文字且未简化，则返回True
    if is_literal(expr) and not simplified:
        return True

    # 如果是合取或析取
    if isinstance(expr, (And, Or)):
        # 检查每个参数是否是文字或否定
        for arg in expr.args:
            if not (is_literal(arg) or isinstance(arg, Not)):
                return False
        return True

    # 如果是否定
    elif isinstance(expr, Not):
        # 检查否定是否只应用于文字
        if is_literal(expr.args[0]):
            return True
        else:
            return False

    else:
        return False
    # 将输入的表达式转换为 SymPy 表达式对象
    expr = sympify(expr)

    # 如果表达式是一个字面值（literal），直接返回 True
    if is_literal(expr):
        return True

    # 使用堆栈来追踪表达式的子表达式
    stack = [expr]

    # 当堆栈不为空时，持续处理表达式
    while stack:
        # 弹出堆栈顶部的表达式
        expr = stack.pop()

        # 如果表达式是 And 或者 Or 运算
        if expr.func in (And, Or):
            # 如果要求简化（simplified=True）
            if simplified:
                # 获取当前表达式的所有子表达式
                args = expr.args
                # 遍历每个子表达式
                for arg in args:
                    # 检查是否存在其否定形式在同一层级中
                    if Not(arg) in args:
                        return False
            # 将当前表达式的所有子表达式加入堆栈继续处理
            stack.extend(expr.args)

        # 如果表达式不是字面值，并且不是 And 或 Or 运算，则返回 False
        elif not is_literal(expr):
            return False

    # 如果所有子表达式都通过检查，则返回 True
    return True
# 检查表达式是否为合取范式（CNF）
def is_cnf(expr):
    """
    Test whether or not an expression is in conjunctive normal form.

    Examples
    ========

    >>> from sympy.logic.boolalg import is_cnf
    >>> from sympy.abc import A, B, C
    >>> is_cnf(A | B | C)
    True
    >>> is_cnf(A & B & C)
    True
    >>> is_cnf((A & B) | C)
    False

    """
    # 调用 _is_form 函数，检查表达式是否为 And 连接的 Or 表达式
    return _is_form(expr, And, Or)


# 检查表达式是否为析取范式（DNF）
def is_dnf(expr):
    """
    Test whether or not an expression is in disjunctive normal form.

    Examples
    ========

    >>> from sympy.logic.boolalg import is_dnf
    >>> from sympy.abc import A, B, C
    >>> is_dnf(A | B | C)
    True
    >>> is_dnf(A & B & C)
    True
    >>> is_dnf((A & B) | C)
    True
    >>> is_dnf(A & (B | C))
    False

    """
    # 调用 _is_form 函数，检查表达式是否为 Or 连接的 And 表达式
    return _is_form(expr, Or, And)


# 检查表达式是否符合指定形式的内部函数
def _is_form(expr, function1, function2):
    """
    Test whether or not an expression is of the required form.

    """
    # 将表达式转化为 Sympy 可处理的形式
    expr = sympify(expr)

    # 如果表达式是 function1 类型，则提取其中的变量列表
    vals = function1.make_args(expr) if isinstance(expr, function1) else [expr]
    for lit in vals:
        # 如果某个元素是 function2 类型，进一步检查其内部元素是否都是 literal
        if isinstance(lit, function2):
            vals2 = function2.make_args(lit) if isinstance(lit, function2) else [lit]
            for l in vals2:
                # 如果内部元素不是 literal，则返回 False
                if is_literal(l) is False:
                    return False
        # 如果元素本身不是 literal，则返回 False
        elif is_literal(lit) is False:
            return False

    # 如果所有条件都满足，则返回 True
    return True


# 将含有蕴含和等价关系的表达式转化为仅含有 And、Or 和 Not 的表达式
def eliminate_implications(expr):
    """
    Change :py:class:`~.Implies` and :py:class:`~.Equivalent` into
    :py:class:`~.And`, :py:class:`~.Or`, and :py:class:`~.Not`.
    That is, return an expression that is equivalent to ``expr``, but has only
    ``&``, ``|``, and ``~`` as logical
    operators.

    Examples
    ========

    >>> from sympy.logic.boolalg import Implies, Equivalent, \
         eliminate_implications
    >>> from sympy.abc import A, B, C
    >>> eliminate_implications(Implies(A, B))
    B | ~A
    >>> eliminate_implications(Equivalent(A, B))
    (A | ~B) & (B | ~A)
    >>> eliminate_implications(Equivalent(A, B, C))
    (A | ~C) & (B | ~A) & (C | ~B)

    """
    # 调用 to_nnf 函数将表达式转化为否定范式，并禁用简化操作
    return to_nnf(expr, simplify=False)


# 检查表达式是否为 literal
def is_literal(expr):
    """
    Returns True if expr is a literal, else False.

    Examples
    ========

    >>> from sympy import Or, Q
    >>> from sympy.abc import A, B
    >>> from sympy.logic.boolalg import is_literal
    >>> is_literal(A)
    True
    >>> is_literal(~A)
    True
    >>> is_literal(Q.zero(A))
    True
    >>> is_literal(A + B)
    True
    >>> is_literal(Or(A, B))
    False

    """
    from sympy.assumptions import AppliedPredicate

    # 如果表达式是 Not 类型，则递归检查其参数是否为 literal
    if isinstance(expr, Not):
        return is_literal(expr.args[0])
    # 如果表达式是 True 或 False，或者是应用的谓词或原子表达式，则为 literal
    elif expr in (True, False) or isinstance(expr, AppliedPredicate) or expr.is_Atom:
        return True
    # 如果表达式不是布尔函数，且所有参数都是应用的谓词或原子表达式，则为 literal
    elif not isinstance(expr, BooleanFunction) and all(
            (isinstance(expr, AppliedPredicate) or a.is_Atom) for a in expr.args):
        return True
    # 否则不是 literal
    return False


def to_int_repr(clauses, symbols):
    """
    Takes clauses in CNF format and puts them into an integer representation.

    """
    # 将符号列表转换为字典，其中符号作为键，从1开始的递增数字作为值
    symbols = dict(zip(symbols, range(1, len(symbols) + 1)))
    
    # 定义一个函数，用于将逻辑表达式中的符号转换为整数表示
    def append_symbol(arg, symbols):
        # 如果参数是 Not 类型的逻辑否定操作，则返回对应符号的负值
        if isinstance(arg, Not):
            return -symbols[arg.args[0]]
        else:
            # 否则，返回对应符号在 symbols 字典中的值
            return symbols[arg]
    
    # 返回一个列表的列表，其中每个子列表表示一个逻辑子句的整数表示
    return [{append_symbol(arg, symbols) for arg in Or.make_args(c)}
            for c in clauses]
def term_to_integer(term):
    """
    Return an integer corresponding to the base-2 digits given by *term*.

    Parameters
    ==========

    term : a string or list of ones and zeros
        Input representing binary digits.

    Examples
    ========

    >>> from sympy.logic.boolalg import term_to_integer
    >>> term_to_integer([1, 0, 0])
    4
    >>> term_to_integer('100')
    4

    """
    # Convert the input term (either list or string) to a binary string representation and then to integer
    return int(''.join(list(map(str, list(term)))), 2)


integer_to_term = ibin  # XXX could delete?


def truth_table(expr, variables, input=True):
    """
    Return a generator of all possible configurations of the input variables,
    and the result of the boolean expression for those values.

    Parameters
    ==========

    expr : Boolean expression
        The boolean expression to evaluate.

    variables : list of variables
        The list of variables involved in the boolean expression.

    input : bool (default ``True``)
        Indicates whether to return the input combinations along with the results.

    Examples
    ========

    >>> from sympy.logic.boolalg import truth_table
    >>> from sympy.abc import x,y
    >>> table = truth_table(x >> y, [x, y])
    >>> for t in table:
    ...     print('{0} -> {1}'.format(*t))
    [0, 0] -> True
    [0, 1] -> True
    [1, 0] -> False
    [1, 1] -> True

    >>> table = truth_table(x | y, [x, y])
    >>> list(table)
    [([0, 0], False), ([0, 1], True), ([1, 0], True), ([1, 1], True)]

    If ``input`` is ``False``, ``truth_table`` returns only a list of truth values.
    In this case, the corresponding input values of variables can be
    deduced from the index of a given output.

    >>> from sympy.utilities.iterables import ibin
    >>> vars = [y, x]
    >>> values = truth_table(x >> y, vars, input=False)
    >>> values = list(values)
    >>> values
    [True, False, True, True]

    >>> for i, value in enumerate(values):
    ...     print('{0} -> {1}'.format(list(zip(
    ...     vars, ibin(i, len(vars)))), value))
    [(y, 0), (x, 0)] -> True
    [(y, 0), (x, 1)] -> False
    [(y, 1), (x, 0)] -> True
    [(y, 1), (x, 1)] -> True

    """
    # Ensure variables are sympy expressions
    variables = [sympify(v) for v in variables]

    # Convert expr to a sympy expression if it is not already
    expr = sympify(expr)
    if not isinstance(expr, BooleanFunction) and not is_literal(expr):
        return

    # Generate all combinations of 0s and 1s for given variables
    table = product((0, 1), repeat=len(variables))
    for term in table:
        # Replace variables with current combination in the expression
        value = expr.xreplace(dict(zip(variables, term)))

        if input:
            yield list(term), value
        else:
            yield value


def _check_pair(minterm1, minterm2):
    """
    Checks if a pair of minterms differs by only one bit. If yes, returns
    index, else returns `-1`.
    """
    # Check if two minterms differ by only one bit position
    index = -1
    for x, i in enumerate(minterm1):  # zip(minterm1, minterm2) is slower
        if i != minterm2[x]:
            if index == -1:
                index = x
            else:
                return -1
    return index


def _convert_to_varsSOP(minterm, variables):
    """
    Converts a term in the expansion of a function from binary to its
    variable form (for SOP).

    """
    # Function description missing in the comments
    """
    根据布尔表达式 minterm 构建对应的布尔变量列表，其中:
    - 如果 minterm 的值为 1，则列表中对应位置的变量取值为 True
    - 如果 minterm 的值不为 1，则列表中对应位置的变量取值为 False
    - 如果 minterm 的值为 3，则对应位置的变量不包含在列表中

    返回一个包含构建的布尔变量列表的逻辑与运算结果
    """
    temp = [variables[n] if val == 1 else Not(variables[n])
            for n, val in enumerate(minterm) if val != 3]
    # 返回一个将 temp 列表中所有元素作为参数传递给 And 函数的结果
    return And(*temp)
# 将最大项（minterm）转换为其变量形式（用于POS），根据二进制值选择变量或其非
def _convert_to_varsPOS(maxterm, variables):
    temp = [variables[n] if val == 0 else Not(variables[n])
            for n, val in enumerate(maxterm) if val != 3]
    return Or(*temp)


# 将项（term）转换为其变量形式（用于ANF），根据二进制值选择变量
def _convert_to_varsANF(term, variables):
    temp = [variables[n] for n, t in enumerate(term) if t == 1]

    if not temp:
        return true  # 如果没有选择任何变量，返回逻辑真值

    return And(*temp)


# 获取所有包含奇数个1的n位二进制数的列表
def _get_odd_parity_terms(n):
    return [e for e in [ibin(i, n) for i in range(2**n)] if sum(e) % 2 == 1]


# 获取所有包含偶数个1的n位二进制数的列表
def _get_even_parity_terms(n):
    return [e for e in [ibin(i, n) for i in range(2**n)] if sum(e) % 2 == 0]


# 使用QM方法将一组最小项（minterms）尽可能简化为一个变量少的最小项集合
def _simplified_pairs(terms):
    if not terms:
        return []

    simplified_terms = []
    todo = list(range(len(terms)))

    # 按照1的个数分类，构建字典
    termdict = defaultdict(list)
    for n, term in enumerate(terms):
        ones = sum(1 for t in term if t == 1)
        termdict[ones].append(n)

    variables = len(terms[0])
    for k in range(variables):
        for i in termdict[k]:
            for j in termdict[k+1]:
                # 检查两个项是否可以通过只有一个位置不同来匹配
                index = _check_pair(terms[i], terms[j])
                if index != -1:
                    # 标记这两个项已处理
                    todo[i] = todo[j] = None
                    # 复制旧项
                    newterm = terms[i][:]
                    # 将不同的位置设为“don't care”（值为3）
                    newterm[index] = 3
                    # 如果新项不在已简化的集合中，则添加
                    if newterm not in simplified_terms:
                        simplified_terms.append(newterm)

    if simplified_terms:
        # 对新的项继续进行进一步简化
        simplified_terms = _simplified_pairs(simplified_terms)

    # 添加剩余的未简化项
    simplified_terms.extend([terms[i] for i in todo if i is not None])
    return simplified_terms


# 在真值表已经被足够简化后，使用主蕴含表方法识别和消除冗余对，并返回必要的参数
def _rem_redundancy(l1, terms):
    if not terms:
        return []

    # terms中项的数量
    nterms = len(terms)
    # l1中的项的数量
    nl1 = len(l1)
    #`
# 创建支配矩阵
dommatrix = [[0]*nl1 for n in range(nterms)]
# 列计数器，用于记录每列的支配项数
colcount = [0]*nl1
# 行计数器，用于记录每行的支配项数
rowcount = [0]*nterms
for primei, prime in enumerate(l1):
    for termi, term in enumerate(terms):
        # 检查主要主项是否覆盖了当前项
        if all(t == 3 or t == mt for t, mt in zip(prime, term)):
            # 将支配矩阵中相应位置设为1
            dommatrix[termi][primei] = 1
            # 更新列计数器和行计数器
            colcount[primei] += 1
            rowcount[termi] += 1

# 记录是否有任何变化
anythingchanged = True
# 返回所有列计数器不为0的主要主项列表
return [l1[i] for i in range(nl1) if colcount[i]]
def SOPform(variables, minterms, dontcares=None):
    """
    The SOPform function uses simplified_pairs and a redundant group-
    eliminating algorithm to convert the list of all input combos that
    generate '1' (the minterms) into the smallest sum-of-products form.

    The variables must be given as the first argument.

    Return a logical :py:class:`~.Or` function (i.e., the "sum of products" or
    "SOP" form) that gives the desired outcome. If there are inputs that can
    be ignored, pass them as a list, too.

    The result will be one of the (perhaps many) functions that satisfy
    the conditions.

    Examples
    ========

    >>> from sympy.logic import SOPform
    >>> from sympy import symbols
    >>> w, x, y, z = symbols('w x y z')
    >>> minterms = [[0, 0, 0, 1], [0, 0, 1, 1],
    ...             [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
    >>> dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    >>> SOPform([w, x, y, z], minterms, dontcares)
    (y & z) | (~w & ~x)

    The terms can also be represented as integers:

    >>> minterms = [1, 3, 7, 11, 15]
    >>> dontcares = [0, 2, 5]
    >>> SOPform([w, x, y, z], minterms, dontcares)
    (y & z) | (~w & ~x)

    They can also be specified using dicts, which does not have to be fully
    specified:

    >>> minterms = [{w: 0, x: 1}, {y: 1, z: 1, x: 0}]
    >>> SOPform([w, x, y, z], minterms)
    (x & ~w) | (y & z & ~x)

    Or a combination:

    >>> minterms = [4, 7, 11, [1, 1, 1, 1]]
    >>> dontcares = [{w : 0, x : 0, y: 0}, 5]
    >>> SOPform([w, x, y, z], minterms, dontcares)
    (w & y & z) | (~w & ~y) | (x & z & ~w)

    See also
    ========

    POSform

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Quine-McCluskey_algorithm
    .. [2] https://en.wikipedia.org/wiki/Don%27t-care_term

    """

    # 如果没有给定 minterms，则返回 false
    if not minterms:
        return false

    # 将 variables 转换为符号表达式
    variables = tuple(map(sympify, variables))

    # 调用 _input_to_binlist 函数，将 minterms 和 variables 转换为二进制列表
    minterms = _input_to_binlist(minterms, variables)
    # 将 dontcares 转换为二进制列表，如果没有提供则使用空列表，并根据变量列表进行处理
    dontcares = _input_to_binlist((dontcares or []), variables)
    # 检查每个 dontcares 是否已经存在于 minterms 中，如果存在则引发 ValueError 异常
    for d in dontcares:
        if d in minterms:
            raise ValueError('%s in minterms is also in dontcares' % d)

    # 调用 _sop_form 函数生成最小项表达式的标准或简化表达式
    return _sop_form(variables, minterms, dontcares)
# 将给定的变量、最小项和不关心项，利用简化对算法和冗余组消除算法，转换为最小的析取范式（POS 表达式）
def _sop_form(variables, minterms, dontcares):
    # 利用简化对算法对最小项和不关心项进行简化处理
    new = _simplified_pairs(minterms + dontcares)
    # 从简化后的表达式中去除冗余组，得到必需的最小项
    essential = _rem_redundancy(new, minterms)
    # 转换为对应变量的析取范式，并返回逻辑或运算的结果
    return Or(*[_convert_to_varsSOP(x, variables) for x in essential])


def POSform(variables, minterms, dontcares=None):
    """
    POSform函数使用简化对算法和冗余组消除算法，将生成 '1' 的所有输入组合（最小项）列表转换为最小的析取范式。

    第一个参数必须是变量列表。

    返回逻辑 :py:class:`~.And` 函数（即“积之和”或“POS”形式），以给出所需的结果。如果有可以忽略的输入，
    也将其作为列表传递。

    结果将是满足条件的（可能有多个）函数之一。

    Examples
    ========

    >>> from sympy.logic import POSform
    >>> from sympy import symbols
    >>> w, x, y, z = symbols('w x y z')
    >>> minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1],
    ...             [1, 0, 1, 1], [1, 1, 1, 1]]
    >>> dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    >>> POSform([w, x, y, z], minterms, dontcares)
    z & (y | ~w)

    可以将项表示为整数：

    >>> minterms = [1, 3, 7, 11, 15]
    >>> dontcares = [0, 2, 5]
    >>> POSform([w, x, y, z], minterms, dontcares)
    z & (y | ~w)

    也可以使用字典指定，不必全部指定：

    >>> minterms = [{w: 0, x: 1}, {y: 1, z: 1, x: 0}]
    >>> POSform([w, x, y, z], minterms)
    (x | y) & (x | z) & (~w | ~x)

    或者是混合形式：

    >>> minterms = [4, 7, 11, [1, 1, 1, 1]]
    >>> dontcares = [{w : 0, x : 0, y: 0}, 5]
    >>> POSform([w, x, y, z], minterms, dontcares)
    (w | x) & (y | ~w) & (z | ~y)

    See also
    ========

    SOPform

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Quine-McCluskey_algorithm
    .. [2] https://en.wikipedia.org/wiki/Don%27t-care_term

    """
    # 如果最小项为空列表，则返回false
    if not minterms:
        return false

    # 将变量列表映射为符号
    variables = tuple(map(sympify, variables))
    # 将最小项列表转换为二进制形式
    minterms = _input_to_binlist(minterms, variables)
    # 将不关心项列表转换为二进制形式，如果不关心项在最小项中，则引发错误
    dontcares = _input_to_binlist((dontcares or []), variables)
    for d in dontcares:
        if d in minterms:
            raise ValueError('%s in minterms is also in dontcares' % d)

    maxterms = []
    # 对每个可能的二进制组合进行遍历
    for t in product((0, 1), repeat=len(variables)):
        t = list(t)
        # 如果当前组合既不在最小项中，也不在不关心项中，则加入最大项列表
        if (t not in minterms) and (t not in dontcares):
            maxterms.append(t)

    # 对最大项和不关心项使用简化对算法进行简化
    new = _simplified_pairs(maxterms + dontcares)
    # 从简化后的表达式中去除冗余组，得到必需的最小项
    essential = _rem_redundancy(new, maxterms)
    # 转换为对应变量的析取范式，并返回逻辑与运算的结果
    return And(*[_convert_to_varsPOS(x, variables) for x in essential])


def ANFform(variables, truthvalues):
    """
    ANFform函数将真值列表转换为代数正常形式（ANF）。

    第一个参数必须是变量列表。

    返回True、False或逻辑 :py:class:`~.And` 函数（即代数正常形式）。

    """
    # 获取变量列表的长度（变量的数量）
    n_vars = len(variables)
    # 获取真值表的长度（真值的数量）
    n_values = len(truthvalues)

    # 如果真值表的长度不等于2的变量数量次方，则抛出数值错误异常
    if n_values != 2 ** n_vars:
        raise ValueError("The number of truth values must be equal to 2^%d, "
                         "got %d" % (n_vars, n_values))

    # 将变量列表中的每个变量名称转换为 sympy 的符号对象
    variables = tuple(map(sympify, variables))

    # 计算真值表对应的布尔函数的 ANF (Algebraic Normal Form) 系数
    coeffs = anf_coeffs(truthvalues)
    # 初始化空列表用于存储 ANF 表达式的项
    terms = []

    # 遍历所有可能的布尔函数输入组合
    for i, t in enumerate(product((0, 1), repeat=n_vars)):
        # 如果 ANF 系数为 1，则将当前布尔函数输入组合添加到 ANF 表达式的项中
        if coeffs[i] == 1:
            terms.append(t)

    # 构造 ANF 表达式，将每个布尔函数输入组合转换为 ANF 格式，并使用逻辑异或连接
    # remove_true=False 表示保留 True 的变量，即不进行化简
    return Xor(*[_convert_to_varsANF(x, variables) for x in terms],
               remove_true=False)
# 返回一个布尔表达式的按位与标准（ANF）的系数列表，用于表示布尔表达式的多项式（Zhegalkin polynomial）
def anf_coeffs(truthvalues):
    """
    Convert a list of truth values of some boolean expression
    to the list of coefficients of the polynomial mod 2 (exclusive
    disjunction) representing the boolean expression in ANF
    (i.e., the "Zhegalkin polynomial").

    There are `2^n` possible Zhegalkin monomials in `n` variables, since
    each monomial is fully specified by the presence or absence of
    each variable.

    We can enumerate all the monomials. For example, boolean
    function with four variables ``(a, b, c, d)`` can contain
    up to `2^4 = 16` monomials. The 13-th monomial is the
    product ``a & b & d``, because 13 in binary is 1, 1, 0, 1.

    A given monomial's presence or absence in a polynomial corresponds
    to that monomial's coefficient being 1 or 0 respectively.

    Examples
    ========
    >>> from sympy.logic.boolalg import anf_coeffs, bool_monomial, Xor
    >>> from sympy.abc import a, b, c
    >>> truthvalues = [0, 1, 1, 0, 0, 1, 0, 1]
    >>> coeffs = anf_coeffs(truthvalues)
    >>> coeffs
    [0, 1, 1, 0, 0, 0, 1, 0]
    >>> polynomial = Xor(*[
    ...     bool_monomial(k, [a, b, c])
    ...     for k, coeff in enumerate(coeffs) if coeff == 1
    ... ])
    >>> polynomial
    b ^ c ^ (a & b)
    """

    # 将列表长度转换为二进制字符串，确定变量数n
    s = '{:b}'.format(len(truthvalues))
    n = len(s) - 1

    # 如果给定的真值列表长度不是2的幂次，则抛出值错误
    if len(truthvalues) != 2**n:
        raise ValueError("The number of truth values must be a power of two, "
                         "got %d" % len(truthvalues))

    # 初始化系数列表，每个元素都是一个包含单个真值的列表
    coeffs = [[v] for v in truthvalues]

    # 枚举Zhegalkin多项式中的每个单项式
    for i in range(n):
        tmp = []
        # 合并相邻的系数列表，并通过异或操作生成新的系数列表
        for j in range(2 ** (n-i-1)):
            tmp.append(coeffs[2*j] +
                list(map(lambda x, y: x^y, coeffs[2*j], coeffs[2*j+1])))
        coeffs = tmp

    # 返回生成的Zhegalkin多项式的系数列表的第一个元素
    return coeffs[0]


# 返回第k个最小项（minterm）
def bool_minterm(k, variables):
    """
    Return the k-th minterm.

    Minterms are numbered by a binary encoding of the complementation
    pattern of the variables. This convention assigns the value 1 to
    the direct form and 0 to the complemented form.

    Parameters
    ==========

    k : int or list of 1's and 0's (complementation pattern)
    variables : list of variables

    Examples
    ========

    >>> from sympy.logic.boolalg import bool_minterm
    >>> from sympy.abc import x, y, z
    >>> bool_minterm([1, 0, 1], [x, y, z])
    x & z & ~y
    >>> bool_minterm(6, [x, y, z])
    x & y & ~z

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Canonical_normal_form#Indexing_minterms

    """
    if isinstance(k, int):
        k = ibin(k, len(variables))
    variables = tuple(map(sympify, variables))
    return _convert_to_varsSOP(k, variables)


# 返回第k个最大项（maxterm）
def bool_maxterm(k, variables):
    """
    Return the k-th maxterm.

    Each maxterm is assigned an index based on the opposite
    conventional binary encoding used for minterms. The maxterm
    convention assigns the value 0 to the direct form and 1 to
    the complemented form.

    Parameters
    ==========

    k : int or list of 1's and 0's (complementation pattern)
    variables : list of variables
    """
    # 未完成的部分将在下一步中继续添加
    k : int or list of 1's and 0's (complementation pattern)
    variables : list of variables

    Examples
    ========
    >>> from sympy.logic.boolalg import bool_maxterm
    >>> from sympy.abc import x, y, z
    >>> bool_maxterm([1, 0, 1], [x, y, z])
    y | ~x | ~z
    >>> bool_maxterm(6, [x, y, z])
    z | ~x | ~y

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Canonical_normal_form#Indexing_maxterms

    """
    # 如果 k 是整数，则转换成对应位数的二进制列表
    if isinstance(k, int):
        k = ibin(k, len(variables))
    # 将 variables 列表中的每个变量符号化（转换成符号对象）
    variables = tuple(map(sympify, variables))
    # 调用 _convert_to_varsPOS 函数将 k 和 variables 转换为其对应的变量的正范式（positive normal form）
    return _convert_to_varsPOS(k, variables)
def bool_monomial(k, variables):
    """
    Return the k-th monomial.

    Monomials are numbered by a binary encoding of the presence and
    absences of the variables. This convention assigns the value
    1 to the presence of variable and 0 to the absence of variable.

    Each boolean function can be uniquely represented by a
    Zhegalkin Polynomial (Algebraic Normal Form). The Zhegalkin
    Polynomial of the boolean function with `n` variables can contain
    up to `2^n` monomials. We can enumerate all the monomials.
    Each monomial is fully specified by the presence or absence
    of each variable.

    For example, boolean function with four variables ``(a, b, c, d)``
    can contain up to `2^4 = 16` monomials. The 13-th monomial is the
    product ``a & b & d``, because 13 in binary is 1, 1, 0, 1.

    Parameters
    ==========

    k : int or list of 1's and 0's
        Either an integer representing the index of the monomial in
        binary form or a list where each element is 1 or 0 indicating
        the presence or absence of corresponding variables.

    variables : list of variables
        A list of SymPy symbols representing the variables in the
        boolean function.

    Examples
    ========
    >>> from sympy.logic.boolalg import bool_monomial
    >>> from sympy.abc import x, y, z
    >>> bool_monomial([1, 0, 1], [x, y, z])
    x & z
    >>> bool_monomial(6, [x, y, z])
    x & y

    """
    # Convert integer index to binary list representation if necessary
    if isinstance(k, int):
        k = ibin(k, len(variables))
    # Convert each variable name to SymPy symbol
    variables = tuple(map(sympify, variables))
    # Return the boolean function in its Algebraic Normal Form (ANF)
    return _convert_to_varsANF(k, variables)


def _find_predicates(expr):
    """
    Helper to find logical predicates in BooleanFunctions.

    A logical predicate is defined here as anything within a BooleanFunction
    that is not a BooleanFunction itself.

    Parameters
    ==========

    expr : BooleanFunction or SymPy expression
        The expression in which logical predicates are to be found.

    Returns
    =======

    set
        A set containing all logical predicates found within the expression.

    """
    if not isinstance(expr, BooleanFunction):
        # If expr is not a BooleanFunction, return it as a set
        return {expr}
    # Recursively find predicates within each argument of the BooleanFunction
    return set().union(*(map(_find_predicates, expr.args)))


def simplify_logic(expr, form=None, deep=True, force=False, dontcare=None):
    """
    This function simplifies a boolean function to its simplified version
    in SOP or POS form. The return type is an :py:class:`~.Or` or
    :py:class:`~.And` object in SymPy.

    Parameters
    ==========

    expr : Boolean
        The boolean expression to be simplified.

    form : string (``'cnf'`` or ``'dnf'``) or ``None`` (default).
        If ``'cnf'`` or ``'dnf'``, the simplest expression in the corresponding
        normal form is returned; if ``None``, the answer is returned
        according to the form with fewest args (in CNF by default).

    deep : bool (default ``True``)
        Indicates whether to recursively simplify any
        non-boolean functions contained within the input.

    force : bool (default ``False``)
        By default, simplification is limited to expressions with
        up to 8 variables to avoid excessive computation time. Setting
        this to ``True`` removes that limit.

    dontcare : list or set, optional
        Specifies the list or set of minterms that are treated as
        don't cares during simplification.

    """
    # Function to simplify boolean expressions and return the result
    pass  # Placeholder for actual implementation
    # 如果指定的 form 不是 None、'cnf' 或 'dnf'，则抛出数值错误异常
    if form not in (None, 'cnf', 'dnf'):
        raise ValueError("form can be cnf or dnf only")
    # 将表达式转换为 SymPy 表达式对象
    expr = sympify(expr)
    # 如果指定了 form，则检查是否为正确的形式（cnf 或 dnf），并且所有参数都是文字变量且不涉及 Not 操作
    if form:
        form_ok = False
        # 检查是否为 CNF 形式
        if form == 'cnf':
            form_ok = is_cnf(expr)
        # 检查是否为 DNF 形式
        elif form == 'dnf':
            form_ok = is_dnf(expr)

        # 如果形式正确且所有参数都是文字变量，则直接返回表达式
        if form_ok and all(is_literal(a) for a in expr.args):
            return expr

    # 导入 SymPy 的关系运算模块
    from sympy.core.relational import Relational
    # 如果 deep 参数为真，则获取表达式中所有的关系运算符
    if deep:
        variables = expr.atoms(Relational)
        # 导入简化函数 simplify，并对所有关系运算符进行简化
        from sympy.simplify.simplify import simplify
        s = tuple(map(simplify, variables))
        # 使用简化后的结果替换原表达式中的变量
        expr = expr.xreplace(dict(zip(variables, s)))

    # 如果表达式不是布尔函数，则直接返回表达式
    if not isinstance(expr, BooleanFunction):
        return expr

    # 将 Relationals 替换为 Dummys 以减少变量数量
    repl = {}
    undo = {}
    from sympy.core.symbol import Dummy
    variables = expr.atoms(Relational)
    # 如果指定了 dontcare，则将其也加入变量集合中
    if dontcare is not None:
        dontcare = sympify(dontcare)
        variables.update(dontcare.atoms(Relational))

    # 替换关系运算符为虚拟变量，并处理其否定形式
    while variables:
        var = variables.pop()
        if var.is_Relational:
            d = Dummy()
            undo[d] = var
            repl[var] = d
            nvar = var.negated
            if nvar in variables:
                repl[nvar] = Not(d)
                variables.remove(nvar)

    # 使用替换后的表达式进行替换
    expr = expr.xreplace(repl)

    # 如果指定了 dontcare，则也对其进行替换
    if dontcare is not None:
        dontcare = dontcare.xreplace(repl)

    # 获取替换后的新变量集合
    variables = _find_predicates(expr)

    # 如果不强制使用并且变量数量超过 8，则恢复到替换前的状态
    if not force and len(variables) > 8:
        return expr.xreplace(undo)

    # 如果指定了 dontcare
    if dontcare is not None:
        # 获取 dontcare 中的变量
        dcvariables = _find_predicates(dontcare)
        # 将其添加到变量集合中
        variables.update(dcvariables)
        # 如果变量数量过多，则恢复到替换前的状态
        if not force and len(variables) > 8:
            variables = _find_predicates(expr)
            dontcare = None

    # 将变量分组为常数和变量值
    c, v = sift(ordered(variables), lambda x: x in (True, False), binary=True)
    variables = c + v
    # 将常量列表 c 中的 True 转换为 1，False 转换为 0，以符合真值表的标准化要求
    c = [1 if i == True else 0 for i in c]
    # 根据变量 v、逻辑表达式 expr 和常量列表 c 获取真值表
    truthtable = _get_truthtable(v, expr, c)
    # 如果存在不考虑的情况（dontcare），获取其对应的真值表
    if dontcare is not None:
        dctruthtable = _get_truthtable(v, dontcare, c)
        # 从主真值表中移除不考虑的真值
        truthtable = [t for t in truthtable if t not in dctruthtable]
    else:
        dctruthtable = []
    # 判断主真值表的长度是否大于或等于 2 的 (变量数 - 1) 次方
    big = len(truthtable) >= (2 ** (len(variables) - 1))
    # 如果形式为 'dnf' 或者 form 为 None 且主真值表很大，则返回最小和范式形式
    if form == 'dnf' or form is None and big:
        return _sop_form(variables, truthtable, dctruthtable).xreplace(undo)
    # 否则返回主析取范式形式
    return POSform(variables, truthtable, dctruthtable).xreplace(undo)
def _get_truthtable(variables, expr, const):
    """ Return a list of all combinations leading to a True result for ``expr``.
    """
    _variables = variables.copy()  # 复制变量列表，以避免修改原始输入

    def _get_tt(inputs):
        if _variables:
            v = _variables.pop()  # 弹出当前处理的变量
            # 处理变量为 false 的情况
            tab = [[i[0].xreplace({v: false}), [0] + i[1]] for i in inputs if i[0] is not false]
            # 处理变量为 true 的情况
            tab.extend([[i[0].xreplace({v: true}), [1] + i[1]] for i in inputs if i[0] is not false])
            return _get_tt(tab)  # 递归调用，处理下一个变量
        return inputs  # 返回处理后的所有组合结果

    # 构建结果列表，包含符合条件的组合
    res = [const + k[1] for k in _get_tt([[expr, []]]) if k[0]]
    if res == [[]]:
        return []
    else:
        return res


def _finger(eq):
    """
    Assign a 5-item fingerprint to each symbol in the equation:
    [
    # of times it appeared as a Symbol;
    # of times it appeared as a Not(symbol);
    # of times it appeared as a Symbol in an And or Or;
    # of times it appeared as a Not(Symbol) in an And or Or;
    a sorted tuple of tuples, (i, j, k), where i is the number of arguments
    in an And or Or with which it appeared as a Symbol, and j is
    the number of arguments that were Not(Symbol); k is the number
    of times that (i, j) was seen.
    ]

    Examples
    ========

    >>> from sympy.logic.boolalg import _finger as finger
    >>> from sympy import And, Or, Not, Xor, to_cnf, symbols
    >>> from sympy.abc import a, b, x, y
    >>> eq = Or(And(Not(y), a), And(Not(y), b), And(x, y))
    >>> dict(finger(eq))
    {(0, 0, 1, 0, ((2, 0, 1),)): [x],
    (0, 0, 1, 0, ((2, 1, 1),)): [a, b],
    (0, 0, 1, 2, ((2, 0, 1),)): [y]}
    >>> dict(finger(x & ~y))
    {(0, 1, 0, 0, ()): [y], (1, 0, 0, 0, ()): [x]}

    In the following, the (5, 2, 6) means that there were 6 Or
    functions in which a symbol appeared as itself amongst 5 arguments in
    which there were also 2 negated symbols, e.g. ``(a0 | a1 | a2 | ~a3 | ~a4)``
    is counted once for a0, a1 and a2.

    >>> dict(finger(to_cnf(Xor(*symbols('a:5')))))
    {(0, 0, 8, 8, ((5, 0, 1), (5, 2, 6), (5, 4, 1))): [a0, a1, a2, a3, a4]}

    The equation must not have more than one level of nesting:

    >>> dict(finger(And(Or(x, y), y)))
    {(0, 0, 1, 0, ((2, 0, 1),)): [x], (1, 0, 1, 0, ((2, 0, 1),)): [y]}
    >>> dict(finger(And(Or(x, And(a, x)), y)))
    Traceback (most recent call last):
    ...
    NotImplementedError: unexpected level of nesting

    So y and x have unique fingerprints, but a and b do not.
    """
    f = eq.free_symbols  # 提取方程式中的自由符号（未赋值的符号变量）
    # 使用字典初始化每个符号的指纹
    d = dict(list(zip(f, [[0]*4 + [defaultdict(int)] for fi in f])))
    # 对每个等式的参数进行迭代处理
    for a in eq.args:
        # 如果参数是一个符号（Symbol），增加其在字典中的计数
        if a.is_Symbol:
            d[a][0] += 1
        # 如果参数是一个逻辑非（Not），处理其子参数在字典中的计数
        elif a.is_Not:
            d[a.args[0]][1] += 1
        else:
            # 否则，参数是复合结构，统计其参数个数及子参数中逻辑非的个数
            o = len(a.args), sum(isinstance(ai, Not) for ai in a.args)
            # 遍历复合结构的每个子参数
            for ai in a.args:
                # 如果子参数是符号（Symbol），增加其在字典中的计数
                if ai.is_Symbol:
                    d[ai][2] += 1
                    # 更新子参数在不同结构（参数个数及逻辑非个数）下的统计信息
                    d[ai][-1][o] += 1
                # 如果子参数是逻辑非（Not），处理其子参数在字典中的计数
                elif ai.is_Not:
                    d[ai.args[0]][3] += 1
                else:
                    # 如果子参数是未预期的嵌套结构，抛出未实现的错误
                    raise NotImplementedError('unexpected level of nesting')
    
    # 创建一个默认值为列表的 defaultdict，用于反转字典 d 的键值对
    inv = defaultdict(list)
    # 对字典 d 的项进行排序后进行迭代
    for k, v in ordered(iter(d.items())):
        # 对每个值列表中的最后一个元素进行排序，并转换为元组
        v[-1] = tuple(sorted([i + (j,) for i, j in v[-1].items()]))
        # 将排序后的值作为键，原始键作为值，添加到 inv 字典中
        inv[tuple(v)].append(k)
    
    # 返回反转后的字典 inv
    return inv
# 定义布尔函数匹配的功能，返回一个简化版本的 *bool1* 和一个变量映射字典，
# 这个字典使得两个布尔表达式 *bool1* 和 *bool2* 在某些变量的对应下具有相同的逻辑行为。
def bool_map(bool1, bool2):
    # 定义一个内部函数，用于比较两个简化的布尔表达式，返回它们之间的变量映射字典（如果可能）。
    def match(function1, function2):
        # 如果两个表达式的类型不同，则返回空值，可能通过简化它们可以使它们相同。
        if function1.__class__ != function2.__class__:
            return None
        # 如果两个表达式的参数数量不同，则返回空值，可能通过简化它们可以使它们相同。
        if len(function1.args) != len(function2.args):
            return None
        # 如果第一个表达式是符号，则返回一个字典，将第一个表达式映射到第二个表达式。
        if function1.is_Symbol:
            return {function1: function2}

        # 获取表达式的指纹（fingerprint）字典
        f1 = _finger(function1)
        f2 = _finger(function2)

        # 如果两个指纹字典的长度不同，则返回 False
        if len(f1) != len(f2):
            return False

        # 如果可能，组装匹配字典
        matchdict = {}
        for k in f1.keys():
            if k not in f2:
                return False
            if len(f1[k]) != len(f2[k]):
                return False
            for i, x in enumerate(f1[k]):
                matchdict[x] = f2[k][i]
        return matchdict

    # 简化输入的布尔表达式 bool1 和 bool2
    a = simplify_logic(bool1)
    b = simplify_logic(bool2)
    # 使用 match 函数查找变量映射
    m = match(a, b)
    # 如果找到映射，则返回简化后的表达式 a 和映射 m
    if m:
        return a, m
    # 如果没有找到映射，则返回 None
    return m
# 定义一个函数，用于应用基于模式的简化到关系表达式中
def _apply_patternbased_simplification(rv, patterns, measure,
                                       dominatingvalue,
                                       replacementvalue=None,
                                       threeterm_patterns=None):
    """
    Replace patterns of Relational

    Parameters
    ==========

    rv : Expr
        Boolean expression

    patterns : tuple
        Tuple of tuples, with (pattern to simplify, simplified pattern) with
        two terms.

    measure : function
        Simplification measure.

    dominatingvalue : Boolean or ``None``
        The dominating value for the function of consideration.
        For example, for :py:class:`~.And` ``S.false`` is dominating.
        As soon as one expression is ``S.false`` in :py:class:`~.And`,
        the whole expression is ``S.false``.

    replacementvalue : Boolean or ``None``, optional
        The resulting value for the whole expression if one argument
        evaluates to ``dominatingvalue``.
        For example, for :py:class:`~.Nand` ``S.false`` is dominating, but
        in this case the resulting value is ``S.true``. Default is ``None``.
        If ``replacementvalue`` is ``None`` and ``dominatingvalue`` is not
        ``None``, ``replacementvalue = dominatingvalue``.

    threeterm_patterns : tuple, optional
        Tuple of tuples, with (pattern to simplify, simplified pattern) with
        three terms.

    """
    # 导入必要的模块和函数
    from sympy.core.relational import Relational, _canonical

    # 如果未指定替换值但指定了主导值，则将替换值设置为主导值
    if replacementvalue is None and dominatingvalue is not None:
        replacementvalue = dominatingvalue

    # 使用 Relational 模式进行替换
    Rel, nonRel = sift(rv.args, lambda i: isinstance(i, Relational),
                       binary=True)

    # 如果只有一个 Relational，直接返回原始表达式
    if len(Rel) <= 1:
        return rv

    # 过滤掉不包含非实数符号的 Relational
    Rel, nonRealRel = sift(Rel, lambda i: not any(s.is_real is False
                                                  for s in i.free_symbols),
                           binary=True)

    # 将 Relational 对象转换为它们的规范形式
    Rel = [i.canonical for i in Rel]

    # 如果提供了三项模式并且 Relational 的数量大于等于3，应用基于三项模式的简化
    if threeterm_patterns and len(Rel) >= 3:
        Rel = _apply_patternbased_threeterm_simplification(Rel,
                            threeterm_patterns, rv.func, dominatingvalue,
                            replacementvalue, measure)

    # 应用基于两项模式的简化
    Rel = _apply_patternbased_twoterm_simplification(Rel, patterns,
                    rv.func, dominatingvalue, replacementvalue, measure)

    # 重构原始表达式并返回
    rv = rv.func(*([_canonical(i) for i in ordered(Rel)]
                 + nonRel + nonRealRel))
    return rv


def _apply_patternbased_twoterm_simplification(Rel, patterns, func,
                                               dominatingvalue,
                                               replacementvalue,
                                               measure):
    """ Apply pattern-based two-term simplification."""
    # 导入必要的函数和模块
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.core.relational import Ge, Gt, _Inequality

    # 设定一个变量用于记录是否发生了变化
    changed = True
    # 当存在变化且关系列表 Rel 的长度大于等于 2 时，执行循环
    while changed and len(Rel) >= 2:
        changed = False
        # 仅使用 < 或 <= 的关系进行处理
        Rel = [r.reversed if isinstance(r, (Ge, Gt)) else r for r in Rel]
        # 根据 ordered 函数对 Rel 进行排序
        Rel = list(ordered(Rel))
        # 对于 Eq 和 Ne，也需要反转测试
        rtmp = [(r, ) if isinstance(r, _Inequality) else (r, r.reversed) for r in Rel]
        # 创建可能替换的结果列表
        results = []
        # 尝试所有可能的反转关系的组合
        for ((i, pi), (j, pj)) in combinations(enumerate(rtmp), 2):
            for pattern, simp in patterns:
                res = []
                for p1, p2 in product(pi, pj):
                    # 使用 SymPy 进行匹配
                    oldexpr = Tuple(p1, p2)
                    tmpres = oldexpr.match(pattern)
                    if tmpres:
                        res.append((tmpres, oldexpr))

                if res:
                    for tmpres, oldexpr in res:
                        # 存在匹配，计算替换结果
                        np = simp.xreplace(tmpres)
                        if np == dominatingvalue:
                            # 如果 np 等于 dominatingvalue，则整个表达式将替换为 replacementvalue
                            return [replacementvalue]
                        # 添加替换结果
                        if not isinstance(np, ITE) and not np.has(Min, Max):
                            # 只有当它们简化为关系时，才使用 ITE 和 Min/Max 替换
                            costsaving = measure(func(*oldexpr.args)) - measure(np)
                            if costsaving > 0:
                                results.append((costsaving, ([i, j], np)))
        if results:
            # 根据复杂性对结果进行排序
            results = sorted(results, key=lambda pair: pair[0], reverse=True)
            # 选择提供最大简化的替换
            replacement = results[0][1]
            idx, newrel = replacement
            idx.sort()
            # 删除旧的关系
            for index in reversed(idx):
                del Rel[index]
            if dominatingvalue is None or newrel != Not(dominatingvalue):
                # 插入新的关系（不需要插入不影响结果的值）
                if newrel.func == func:
                    for a in newrel.args:
                        Rel.append(a)
                else:
                    Rel.append(newrel)
            # 发生了变化，需要再次尝试
            changed = True
    # 返回处理后的关系列表 Rel
    return Rel
# 定义一个函数 `_apply_patternbased_threeterm_simplification`，用于应用基于模式的三项简化。
# 函数参数包括：
# - Rel: 一个关系表达式
# - patterns: 用于匹配的模式
# - func: 用于应用的函数
# - dominatingvalue: 主导值，被替换的值需要比它小
# - replacementvalue: 替换值，用于替换被简化的表达式
# - measure: 用于度量的对象或方法

""" Apply pattern-based three-term simplification."""
# 导入 sympy 库中的 Min 和 Max 函数，用于比较取最小值和最大值
from sympy.functions.elementary.miscellaneous import Min, Max
# 导入 sympy 库中的关系运算模块，包括 Le（小于等于）、Lt（小于）和 _Inequality（不等式）
from sympy.core.relational import Le, Lt, _Inequality
# 初始化变量 changed 为 True，表示开始时默认发生了变化
changed = True
    # While loop continues as long as 'changed' is True and the length of Rel is 3 or more
    while changed and len(Rel) >= 3:
        changed = False
        # Replace instances of Le/Lt with their reversed counterparts if they exist
        Rel = [r.reversed if isinstance(r, (Le, Lt)) else r for r in Rel]
        # Sort Rel based on custom ordering defined by 'ordered' function
        Rel = list(ordered(Rel))
        # Initialize an empty list to store results
        results = []
        # Create pairs for each element in Rel, considering reversed instances for Eq/Ne
        rtmp = [(r, ) if isinstance(r, _Inequality) else (r, r.reversed) for r in Rel]
        # Iterate over permutations of three elements from rtmp and patterns
        for ((i, pi), (j, pj), (k, pk)) in permutations(enumerate(rtmp), 3):
            for pattern, simp in patterns:
                res = []
                # Iterate over all combinations of pi, pj, pk
                for p1, p2, p3 in product(pi, pj, pk):
                    # Use SymPy's matching to find substitutions
                    oldexpr = Tuple(p1, p2, p3)
                    tmpres = oldexpr.match(pattern)
                    if tmpres:
                        res.append((tmpres, oldexpr))
                if res:
                    for tmpres, oldexpr in res:
                        # Calculate the replacement expression np
                        np = simp.xreplace(tmpres)
                        # Check if np matches dominatingvalue
                        if np == dominatingvalue:
                            # If np matches dominatingvalue, return replacementvalue as a list
                            return [replacementvalue]
                        # Check if np is not an ITE and does not have Min or Max
                        if not isinstance(np, ITE) and not np.has(Min, Max):
                            # Measure the cost saving by replacing oldexpr with np
                            costsaving = measure(func(*oldexpr.args)) - measure(np)
                            if costsaving > 0:
                                # If cost saving is positive, add the replacement to results
                                results.append((costsaving, ([i, j, k], np)))
        if results:
            # Sort results based on complexity in descending order
            results = sorted(results, key=lambda pair: pair[0], reverse=True)
            # Select the replacement providing the most simplification
            replacement = results[0][1]
            idx, newrel = replacement
            idx.sort()
            # Remove old relationals from Rel
            for index in reversed(idx):
                del Rel[index]
            # Check if dominatingvalue is None or newrel is not Not(dominatingvalue)
            if dominatingvalue is None or newrel != Not(dominatingvalue):
                # Insert newrel into Rel
                if newrel.func == func:
                    for a in newrel.args:
                        Rel.append(a)
                else:
                    Rel.append(newrel)
            # Set changed to True indicating a change was made
            changed = True
    # Return the modified Rel after all transformations
    return Rel
# 定义一个装饰器函数 `cacheit`，用于缓存函数的返回值
@cacheit
# 定义一个私有函数 `_simplify_patterns_and`，用于处理两个项的逻辑与模式
def _simplify_patterns_and():
    """ Two-term patterns for And."""
    
    # 从 sympy 库中导入必要的模块和函数
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.miscellaneous import Min, Max
    
    # 创建 Wild 对象 a, b, c，用于模式匹配
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    
    # 定义关系模式按字母顺序排列的元组，以及简化后的结果
    # 不使用 Ge, Gt
    _matchers_and = (
        (Tuple(Eq(a, b), Lt(a, b)), false),
        #(Tuple(Eq(a, b), Lt(b, a)), S.false),
        #(Tuple(Le(b, a), Lt(a, b)), S.false),
        #(Tuple(Lt(b, a), Le(a, b)), S.false),
        (Tuple(Lt(b, a), Lt(a, b)), false),
        (Tuple(Eq(a, b), Le(b, a)), Eq(a, b)),
        #(Tuple(Eq(a, b), Le(a, b)), Eq(a, b)),
        #(Tuple(Le(b, a), Lt(b, a)), Gt(a, b)),
        (Tuple(Le(b, a), Le(a, b)), Eq(a, b)),
        #(Tuple(Le(b, a), Ne(a, b)), Gt(a, b)),
        #(Tuple(Lt(b, a), Ne(a, b)), Gt(a, b)),
        (Tuple(Le(a, b), Lt(a, b)), Lt(a, b)),
        (Tuple(Le(a, b), Ne(a, b)), Lt(a, b)),
        (Tuple(Lt(a, b), Ne(a, b)), Lt(a, b)),
        # Sign
        (Tuple(Eq(a, b), Eq(a, -b)), And(Eq(a, S.Zero), Eq(b, S.Zero))),
        # Min/Max/ITE
        (Tuple(Le(b, a), Le(c, a)), Ge(a, Max(b, c))),
        (Tuple(Le(b, a), Lt(c, a)), ITE(b > c, Ge(a, b), Gt(a, c))),
        (Tuple(Lt(b, a), Lt(c, a)), Gt(a, Max(b, c))),
        (Tuple(Le(a, b), Le(a, c)), Le(a, Min(b, c))),
        (Tuple(Le(a, b), Lt(a, c)), ITE(b < c, Le(a, b), Lt(a, c))),
        (Tuple(Lt(a, b), Lt(a, c)), Lt(a, Min(b, c))),
        (Tuple(Le(a, b), Le(c, a)), ITE(Eq(b, c), Eq(a, b), ITE(b < c, false, And(Le(a, b), Ge(a, c))))),
        (Tuple(Le(c, a), Le(a, b)), ITE(Eq(b, c), Eq(a, b), ITE(b < c, false, And(Le(a, b), Ge(a, c))))),
        (Tuple(Lt(a, b), Lt(c, a)), ITE(b < c, false, And(Lt(a, b), Gt(a, c)))),
        (Tuple(Lt(c, a), Lt(a, b)), ITE(b < c, false, And(Lt(a, b), Gt(a, c)))),
        (Tuple(Le(a, b), Lt(c, a)), ITE(b <= c, false, And(Le(a, b), Gt(a, c)))),
        (Tuple(Le(c, a), Lt(a, b)), ITE(b <= c, false, And(Lt(a, b), Ge(a, c)))),
        (Tuple(Eq(a, b), Eq(a, c)), ITE(Eq(b, c), Eq(a, b), false)),
        (Tuple(Lt(a, b), Lt(-b, a)), ITE(b > 0, Lt(Abs(a), b), false)),
        (Tuple(Le(a, b), Le(-b, a)), ITE(b >= 0, Le(Abs(a), b), false)),
    )
    
    # 返回处理后的模式匹配结果
    return _matchers_and


# 定义一个装饰器函数 `cacheit`，用于缓存函数的返回值
@cacheit
# 定义一个私有函数 `_simplify_patterns_and3`，用于处理三个项的逻辑与模式
def _simplify_patterns_and3():
    """ Three-term patterns for And."""
    
    # 从 sympy 库中导入必要的模块和函数
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ge, Gt
    
    # 创建 Wild 对象 a，用于模式匹配
    a = Wild('a')
    # 创建一个名为 b 的 Wild 对象，用于在模式匹配中捕获任意的表达式部分
    b = Wild('b')
    
    # 创建一个名为 c 的 Wild 对象，用于在模式匹配中捕获另一个任意的表达式部分
    c = Wild('c')
    
    # 定义了一组关系模式及其简化形式，按照字母顺序排列，用于模式匹配
    # 注意，这些模式是用于逻辑关系判断的，不包括 Le, Lt 关系
    _matchers_and = (
        # 第一个模式: Ge(a, b) ∧ Ge(b, c) ∧ Gt(c, a) -> false
        (Tuple(Ge(a, b), Ge(b, c), Gt(c, a)), false),
    
        # 第二个模式: Ge(a, b) ∧ Gt(b, c) ∧ Gt(c, a) -> false
        (Tuple(Ge(a, b), Gt(b, c), Gt(c, a)), false),
    
        # 第三个模式: Gt(a, b) ∧ Gt(b, c) ∧ Gt(c, a) -> false
        (Tuple(Gt(a, b), Gt(b, c), Gt(c, a)), false),
    
        # 第四个模式: Ge(a, b) ∧ Ge(a, c) ∧ Ge(b, c) -> Ge(a, b) ∧ Ge(b, c)
        # 当满足 Ge(a, b) ∧ Ge(b, c) 条件时简化为 And(Ge(a, b), Ge(b, c))
        (Tuple(Ge(a, b), Ge(a, c), Ge(b, c)), And(Ge(a, b), Ge(b, c))),
    
        # 第五个模式: Ge(a, b) ∧ Ge(a, c) ∧ Gt(b, c) -> Ge(a, b) ∧ Gt(b, c)
        # 当满足 Ge(a, b) ∧ Gt(b, c) 条件时简化为 And(Ge(a, b), Gt(b, c))
        (Tuple(Ge(a, b), Ge(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))),
    
        # 第六个模式: Ge(a, b) ∧ Gt(a, c) ∧ Gt(b, c) -> Ge(a, b) ∧ Gt(b, c)
        # 当满足 Ge(a, b) ∧ Gt(b, c) 条件时简化为 And(Ge(a, b), Gt(b, c))
        (Tuple(Ge(a, b), Gt(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))),
    
        # 第七个模式: Ge(a, b) ∧ Gt(a, c) ∧ Gt(b, c) -> Ge(a, b) ∧ Gt(b, c)
        # 当满足 Ge(a, b) ∧ Gt(b, c) 条件时简化为 And(Ge(a, b), Gt(b, c))
        (Tuple(Ge(a, b), Gt(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))),
    
        # 第八个模式: Ge(a, c) ∧ Gt(a, b) ∧ Gt(b, c) -> Gt(a, b) ∧ Gt(b, c)
        # 当满足 Gt(a, b) ∧ Gt(b, c) 条件时简化为 And(Gt(a, b), Gt(b, c))
        (Tuple(Ge(a, c), Gt(a, b), Gt(b, c)), And(Gt(a, b), Gt(b, c))),
    
        # 第九个模式: Ge(b, c) ∧ Gt(a, b) ∧ Gt(a, c) -> Gt(a, b) ∧ Ge(b, c)
        # 当满足 Gt(a, b) ∧ Ge(b, c) 条件时简化为 And(Gt(a, b), Ge(b, c))
        (Tuple(Ge(b, c), Gt(a, b), Gt(a, c)), And(Gt(a, b), Ge(b, c))),
    
        # 第十个模式: Gt(a, b) ∧ Gt(a, c) ∧ Gt(b, c) -> Gt(a, b) ∧ Gt(b, c)
        # 当满足 Gt(a, b) ∧ Gt(b, c) 条件时简化为 And(Gt(a, b), Gt(b, c))
        (Tuple(Gt(a, b), Gt(a, c), Gt(b, c)), And(Gt(a, b), Gt(b, c))),
    
        # 第十一个模式: Ge(b, a) ∧ Ge(c, a) ∧ Ge(b, c) -> Ge(c, a) ∧ Ge(b, c)
        # 当满足 Ge(c, a) ∧ Ge(b, c) 条件时简化为 And(Ge(c, a), Ge(b, c))
        (Tuple(Ge(b, a), Ge(c, a), Ge(b, c)), And(Ge(c, a), Ge(b, c))),
    
        # 第十二个模式: Ge(b, a) ∧ Ge(c, a) ∧ Gt(b, c) -> Ge(c, a) ∧ Gt(b, c)
        # 当满足 Ge(c, a) ∧ Gt(b, c) 条件时简化为 And(Ge(c, a), Gt(b, c))
        (Tuple(Ge(b, a), Ge(c, a), Gt(b, c)), And(Ge(c, a), Gt(b, c))),
    
        # 第十三个模式: Ge(b, a) ∧ Gt(c, a) ∧ Gt(b, c) -> Gt(c, a) ∧ Gt(b, c)
        # 当满足 Gt(c, a) ∧ Gt(b, c) 条件时简化为 And(Gt(c, a), Gt(b, c))
        (Tuple(Ge(b, a), Gt(c, a), Gt(b, c)), And(Gt(c, a), Gt(b, c))),
    
        # 第十四个模式: Ge(c, a) ∧ Gt(b, a) ∧ Gt(b, c) -> Ge(c, a) ∧ Gt(b, c)
        # 当满足 Ge(c, a) ∧ Gt(b, c) 条件时简化为 And(Ge(c, a), Gt(b, c))
        (Tuple(Ge(c, a), Gt(b, a), Gt(b, c)), And(Ge(c, a), Gt(b, c))),
    
        # 第十五个模式: Ge(b, c) ∧ Gt(b, a) ∧ Gt(c, a) -> Gt(c, a) ∧ Ge(b, c)
        # 当满足 Gt(c, a) ∧ Ge(b, c) 条件时简化为 And(Ge(c, a), Gt(b, c))
        (Tuple(Ge(b, c), Gt(b, a), Gt(c, a)), And(Ge(c, a), Gt(b, c))),
    
        # 第十六个模式: Gt(b, a) ∧ Gt(c, a) ∧ Gt(b, c) -> Gt(c, a) ∧ Gt(b, c)
        # 当满足 Gt(c, a) ∧ Gt(b, c) 条件时简化为 And(Gt(c, a), Gt(b, c))
        (Tuple(Gt(b, a), Gt(c, a), Gt(b, c)), And(Gt(c, a), Gt(b, c))),
    
        # 第十七个模式: Ge(a, b) ∧ Ge(b, c) ∧ Ge(c, a) -> Eq(a, b) ∧ Eq(b, c)
        # 当满足 Eq(a, b) ∧ Eq(b, c) 条件时简化为 And(Eq(a, b), Eq(b, c))
        (Tuple(Ge(a, b), Ge(b, c), Ge(c, a)), And(Eq(a, b), Eq(b, c))),
    )
    
    # 返回模式匹配结果集合 _matchers_and
    return _matchers_and
# 使用装饰器 @cacheit 将函数标记为缓存函数，以便在调用时缓存结果
def _simplify_patterns_or():
    """ Two-term patterns for Or."""

    # 从 sympy.core 模块导入 Wild 类，用于匹配未知模式
    from sympy.core import Wild
    # 从 sympy.core.relational 模块导入关系运算符 Eq, Ne, Ge, Gt, Le, Lt
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    # 从 sympy.functions.elementary.complexes 模块导入 Abs 函数
    from sympy.functions.elementary.complexes import Abs
    # 从 sympy.functions.elementary.miscellaneous 模块导入 Min, Max 函数
    from sympy.functions.elementary.miscellaneous import Min, Max
    
    # 创建 Wild 类的实例 a, b, c 用于匹配模式中的未知项
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    
    # 定义用于匹配 Or 表达式的模式元组 _matchers_or
    # 每个模式元组包含两个模式和一个简化后的结果
    # 注释中提到关系运算符应按字母顺序排列
    _matchers_or = (
        (Tuple(Le(b, a), Le(a, b)), true),  # 匹配 (b <= a) & (a <= b) -> true
        # (Tuple(Le(b, a), Lt(a, b)), true),  # 注释掉的模式
        (Tuple(Le(b, a), Ne(a, b)), true),  # 匹配 (b <= a) & (a != b) -> true
        # (Tuple(Le(a, b), Lt(b, a)), true),  # 注释掉的模式
        # (Tuple(Le(a, b), Ne(a, b)), true),  # 注释掉的模式
        # (Tuple(Eq(a, b), Le(b, a)), Ge(a, b)),  # 注释掉的模式
        # (Tuple(Eq(a, b), Lt(b, a)), Ge(a, b)),  # 注释掉的模式
        (Tuple(Eq(a, b), Le(a, b)), Le(a, b)),  # 匹配 (a == b) & (a <= b) -> (a <= b)
        (Tuple(Eq(a, b), Lt(a, b)), Le(a, b)),  # 匹配 (a == b) & (a < b) -> (a <= b)
        # (Tuple(Le(b, a), Lt(b, a)), Ge(a, b)),  # 注释掉的模式
        (Tuple(Lt(b, a), Lt(a, b)), Ne(a, b)),  # 匹配 (b < a) & (a < b) -> (a != b)
        (Tuple(Lt(b, a), Ne(a, b)), Ne(a, b)),  # 匹配 (b < a) & (a != b) -> (a != b)
        (Tuple(Le(a, b), Lt(a, b)), Le(a, b)),  # 匹配 (a <= b) & (a < b) -> (a <= b)
        # (Tuple(Lt(a, b), Ne(a, b)), Ne(a, b)),  # 注释掉的模式
        (Tuple(Eq(a, b), Ne(a, c)), ITE(Eq(b, c), true, Ne(a, c))),  # 匹配 (a == b) & (a != c) -> ITE(a == c, true, a != c)
        (Tuple(Ne(a, b), Ne(a, c)), ITE(Eq(b, c), Ne(a, b), true)),  # 匹配 (a != b) & (a != c) -> ITE(a == c, a != b, true)
        # Min/Max/ITE
        (Tuple(Le(b, a), Le(c, a)), Ge(a, Min(b, c))),  # 匹配 (b <= a) & (c <= a) -> Ge(a, Min(b, c))
        # (Tuple(Ge(b, a), Ge(c, a)), Ge(Min(b, c), a)),  # 注释掉的模式
        (Tuple(Le(b, a), Lt(c, a)), ITE(b > c, Lt(c, a), Le(b, a))),  # 匹配 (b <= a) & (c < a) -> ITE(b > c, Lt(c, a), Le(b, a))
        (Tuple(Lt(b, a), Lt(c, a)), Gt(a, Min(b, c))),  # 匹配 (b < a) & (c < a) -> Gt(a, Min(b, c))
        # (Tuple(Gt(b, a), Gt(c, a)), Gt(Min(b, c), a)),  # 注释掉的模式
        (Tuple(Le(a, b), Le(a, c)), Le(a, Max(b, c))),  # 匹配 (a <= b) & (a <= c) -> Le(a, Max(b, c))
        # (Tuple(Le(b, a), Le(c, a)), Le(Max(b, c), a)),  # 注释掉的模式
        (Tuple(Le(a, b), Lt(a, c)), ITE(b >= c, Le(a, b), Lt(a, c))),  # 匹配 (a <= b) & (a < c) -> ITE(b >= c, Le(a, b), Lt(a, c))
        (Tuple(Lt(a, b), Lt(a, c)), Lt(a, Max(b, c))),  # 匹配 (a < b) & (a < c) -> Lt(a, Max(b, c))
        # (Tuple(Lt(b, a), Lt(c, a)), Lt(Max(b, c), a)),  # 注释掉的模式
        (Tuple(Le(a, b), Le(c, a)), ITE(b >= c, true, Or(Le(a, b), Ge(a, c)))),  # 匹配 (a <= b) & (c <= a) -> ITE(b >= c, true, (a <= b) | (a >= c))
        (Tuple(Le(c, a), Le(a, b)), ITE(b >= c, true, Or(Le(a, b), Ge(a, c)))),  # 匹配 (c <= a) & (a <= b) -> ITE(b >= c, true, (a <= b) | (a >= c))
        (Tuple(Lt(a, b), Lt(c, a)), ITE(b > c, true, Or(Lt(a, b), Gt(a, c)))),  # 匹配 (a < b) & (c < a) -> ITE(b > c, true, (a < b) | (a > c))
        (Tuple(Lt(c, a), Lt(a, b)), ITE(b > c, true, Or(Lt(a, b), Gt(a, c)))),  # 匹配 (c < a) & (a < b) -> ITE(b > c, true, (a < b) | (a > c))
        (Tuple(Le(a, b), Lt(c, a)), ITE(b >= c, true, Or(Le(a, b), Gt(a, c)))),  # 匹配 (a <= b) & (c < a) -> ITE(b >= c, true, (a <= b) | (a > c))
        (Tuple(Le(c, a), Lt(a, b)), ITE(b >= c, true, Or(Lt(a, b), Ge(a, c)))),  # 匹配 (c <= a) & (a < b) -> ITE(b >= c, true, (a < b) | (a >= c))
        (Tuple(Lt(b, a), Lt(a, -b)), ITE(b >= 0, Gt(Abs(a), b), true)),  # 匹配 (b < a) & (a < -b) -> ITE(b >= 0, Gt(Abs(a), b), true)
        (Tuple(Le(b, a), Le(a, -b)), ITE(b > 0, Ge(Abs(a), b), true)),  # 匹配 (b <= a) & (a <= -b) -> ITE(b > 0, Ge(Abs(a), b), true)
    )
    
    # 返回匹配模式的元组 _matchers_or
    return _matchers_or
    """ Two-term patterns for Xor."""

    # 导入必要的模块和函数
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    
    # 创建三个 Wild 对象，用于模式匹配
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    
    # 定义用于 XOR 模式匹配的元组列表
    # 每个元组包含两个模式和一个简化后的表达式
    # 注意：关系运算符应按字母顺序排列
    # 不使用 Ge, Gt
    _matchers_xor = (
        # (Tuple(Le(b, a), Lt(a, b)), true),
        # (Tuple(Lt(b, a), Le(a, b)), true),
        # (Tuple(Eq(a, b), Le(b, a)), Gt(a, b)),
        # (Tuple(Eq(a, b), Lt(b, a)), Ge(a, b)),
        
        # XOR 模式匹配规则
        (Tuple(Eq(a, b), Le(a, b)), Lt(a, b)),
        (Tuple(Eq(a, b), Lt(a, b)), Le(a, b)),
        (Tuple(Le(a, b), Lt(a, b)), Eq(a, b)),
        (Tuple(Le(a, b), Le(b, a)), Ne(a, b)),
        (Tuple(Le(b, a), Ne(a, b)), Le(a, b)),
        # (Tuple(Lt(b, a), Lt(a, b)), Ne(a, b)),
        (Tuple(Lt(b, a), Ne(a, b)), Lt(a, b)),
        # (Tuple(Le(a, b), Lt(a, b)), Eq(a, b)),
        # (Tuple(Le(a, b), Ne(a, b)), Ge(a, b)),
        # (Tuple(Lt(a, b), Ne(a, b)), Gt(a, b)),
        
        # Min/Max/ITE 模式匹配规则
        (Tuple(Le(b, a), Le(c, a)), And(Ge(a, Min(b, c)), Lt(a, Max(b, c)))),
        (Tuple(Le(b, a), Lt(c, a)), ITE(b > c, And(Gt(a, c), Lt(a, b)), And(Ge(a, b), Le(a, c)))),
        (Tuple(Lt(b, a), Lt(c, a)), And(Gt(a, Min(b, c)), Le(a, Max(b, c)))),
        (Tuple(Le(a, b), Le(a, c)), And(Le(a, Max(b, c)), Gt(a, Min(b, c)))),
        (Tuple(Le(a, b), Lt(a, c)), ITE(b < c, And(Lt(a, c), Gt(a, b)), And(Le(a, b), Ge(a, c)))),
        (Tuple(Lt(a, b), Lt(a, c)), And(Lt(a, Max(b, c)), Ge(a, Min(b, c))))
    )
    
    # 返回 XOR 模式匹配的元组列表
    return _matchers_xor
# 返回简化后的一元布尔表达式，否则返回原始表达式
def simplify_univariate(expr):
    # 导入所需的类和函数
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.core.relational import Eq, Ne

    # 如果表达式不是布尔函数，直接返回原始表达式
    if not isinstance(expr, BooleanFunction):
        return expr
    
    # 如果表达式包含 Eq 或 Ne，也直接返回原始表达式
    if expr.atoms(Eq, Ne):
        return expr
    
    # 复制表达式
    c = expr
    
    # 获取表达式中的自由符号
    free = c.free_symbols
    
    # 如果自由符号数量不等于1，返回原始表达式
    if len(free) != 1:
        return c
    
    # 取出唯一的自由符号
    x = free.pop()
    
    # 使用 Piecewise 对象处理表达式，获取表达式的区间
    ok, i = Piecewise((0, c), evaluate=False)._intervals(x, err_on_Eq=True)
    
    # 如果处理失败，返回原始表达式
    if not ok:
        return c
    
    # 如果没有区间，返回 false
    if not i:
        return false
    
    # 存储处理后的子表达式
    args = []
    
    # 遍历每个区间
    for a, b, _, _ in i:
        if a is S.NegativeInfinity:
            # 处理左无穷大的情况
            if b is S.Infinity:
                c = true
            else:
                # 如果表达式在 b 处为 true，则返回 (x <= b)，否则返回 (x < b)
                if c.subs(x, b) == True:
                    c = (x <= b)
                else:
                    c = (x < b)
        else:
            # 处理一般情况
            incl_a = (c.subs(x, a) == True)
            incl_b = (c.subs(x, b) == True)
            
            if incl_a and incl_b:
                # 如果 a 和 b 都包含在表达式中，则根据 b 是否无穷大返回不同的表达式
                if b.is_infinite:
                    c = (x >= a)
                else:
                    c = And(a <= x, x <= b)
            elif incl_a:
                # 如果只有 a 包含在表达式中，则返回 And(a <= x, x < b)
                c = And(a <= x, x < b)
            elif incl_b:
                # 如果只有 b 包含在表达式中，则根据 b 是否无穷大返回不同的表达式
                if b.is_infinite:
                    c = (x > a)
                else:
                    c = And(a < x, x <= b)
            else:
                # 如果 a 和 b 都不包含在表达式中，则返回 And(a < x, x < b)
                c = And(a < x, x < b)
        
        # 将处理后的子表达式添加到列表中
        args.append(c)
    
    # 返回所有子表达式的或逻辑
    return Or(*args)


# 逻辑门对应的类
# 在 gateinputcount 方法中使用
BooleanGates = (And, Or, Xor, Nand, Nor, Not, Xnor, ITE)

def gateinputcount(expr):
    """
    返回实现布尔表达式的逻辑门的总输入数量。

    Returns
    =======

    int
        门的输入数量

    Note
    ====

    不是所有布尔函数都视为门，只有被视为标准门的那些函数才算，包括：
    :py:class:`~.And`, :py:class:`~.Or`, :py:class:`~.Xor`,
    :py:class:`~.Not` 和 :py:class:`~.ITE`（多路选择器）。
    :py:class:`~.Nand`, :py:class:`~.Nor` 和 :py:class:`~.Xnor` 将被评估为
    ``Not(And())`` 等。

    Examples
    ========

    >>> from sympy.logic import And, Or, Nand, Not, gateinputcount
    >>> from sympy.abc import x, y, z
    >>> expr = And(x, y)
    >>> gateinputcount(expr)
    2
    >>> gateinputcount(Or(expr, z))
    4

    注意，``Nand`` 自动评估为 ``Not(And())``，因此

    >>> gateinputcount(Nand(x, y, z))
    4
    >>> gateinputcount(Not(And(x, y, z)))
    4

    可以通过使用 ``evaluate=False`` 避免这种情况

    >>> gateinputcount(Nand(x, y, z, evaluate=False))
    3

    还要注意，比较将计算为布尔变量：

    >>> gateinputcount(And(x > z, y >= 2))
    2

    符号也将计算为布尔变量：

    >>> gateinputcount(x)
    0

    """
    # 如果表达式不是布尔类型，抛出类型错误异常
    if not isinstance(expr, Boolean):
        raise TypeError("Expression must be Boolean")
    # 如果表达式是 BooleanGates 类型的实例，则执行以下逻辑
    if isinstance(expr, BooleanGates):
        # 返回表达式参数的数量加上每个参数递归调用 gateinputcount 函数后的总和
        return len(expr.args) + sum(gateinputcount(x) for x in expr.args)
    # 如果表达式不是 BooleanGates 类型的实例，则返回 0
    return 0
```