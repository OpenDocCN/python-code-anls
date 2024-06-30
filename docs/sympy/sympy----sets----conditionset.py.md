# `D:\src\scipysrc\sympy\sympy\sets\conditionset.py`

```
# 导入 SymPy 库中的特定模块和类
from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Lambda, BadSignatureError
from sympy.core.logic import fuzzy_bool
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import And, as_Boolean
from sympy.utilities.iterables import sift, flatten, has_dups
from sympy.utilities.exceptions import sympy_deprecation_warning
from .contains import Contains  # 导入当前目录下的 contains 模块中的 Contains 类
from .sets import Set, Union, FiniteSet, SetKind  # 导入当前目录下的 sets 模块中的一些类和函数

adummy = Dummy('conditionset')  # 创建一个名为 'conditionset' 的虚拟符号对象 adummy

class ConditionSet(Set):  # 定义 ConditionSet 类，继承自 Set 类
    r"""
    满足给定条件的元素集合。

    .. math:: \{x \mid \textrm{condition}(x) = \texttt{True}, x \in S\}

    Examples
    ========

    >>> from sympy import Symbol, S, ConditionSet, pi, Eq, sin, Interval
    >>> from sympy.abc import x, y, z

    >>> sin_sols = ConditionSet(x, Eq(sin(x), 0), Interval(0, 2*pi))
    >>> 2*pi in sin_sols
    True
    >>> pi/2 in sin_sols
    False
    >>> 3*pi in sin_sols
    False
    >>> 5 in ConditionSet(x, x**2 > 4, S.Reals)
    True

    如果值不在基本集合中，则结果为 false：

    >>> 5 in ConditionSet(x, x**2 > 4, Interval(2, 4))
    False

    Notes
    =====

    应避免使用具有假设的符号，否则条件可能会在不考虑集合的情况下评估：

    >>> n = Symbol('n', negative=True)
    >>> cond = (n > 0); cond
    False
    >>> ConditionSet(n, cond, S.Integers)
    EmptySet

    只有自由符号可以通过使用 `subs` 进行更改：

    >>> c = ConditionSet(x, x < 1, {x, z})
    >>> c.subs(x, y)
    ConditionSet(x, x < 1, {y, z})

    要检查 `pi` 是否在 `c` 中，请使用：

    >>> pi in c
    False

    如果没有指定基本集合，则默认为全集：

    >>> ConditionSet(x, x < 1).base_set
    UniversalSet

    只能使用符号或类似符号的表达式：

    >>> ConditionSet(x + 1, x + 1 < 1, S.Integers)
    Traceback (most recent call last):
    ...
    ValueError: non-symbol dummy not recognized in condition

    当基本集合是 ConditionSet 时，符号将尽可能统一，并优先考虑最外层的符号：

    >>> ConditionSet(x, x < y, ConditionSet(z, z + y < 2, S.Integers))
    ConditionSet(x, (x < y) & (x + y < 2), Integers)

    """
    # 定义一个特殊方法 __new__，用于创建一个新的对象实例
    def __new__(cls, sym, condition, base_set=S.UniversalSet):
        # 将输入的符号 sym 转换为符号表达式
        sym = _sympify(sym)
        # 将符号列表扁平化
        flat = flatten([sym])
        # 检查扁平化后的符号列表是否存在重复项，若有则抛出异常
        if has_dups(flat):
            raise BadSignatureError("Duplicate symbols detected")
        # 将 base_set 转换为符号表达式
        base_set = _sympify(base_set)
        # 检查 base_set 是否为 Set 对象，若不是则抛出类型错误异常
        if not isinstance(base_set, Set):
            raise TypeError(
                'base set should be a Set object, not %s' % base_set)
        # 将条件 condition 转换为符号表达式
        condition = _sympify(condition)

        # 如果条件 condition 是有限集（FiniteSet）类型
        if isinstance(condition, FiniteSet):
            # 保存原始条件
            condition_orig = condition
            # 构造一个生成器，将每个条件等式 lhs == 0 转换为 sympy 的 And 表达式
            temp = (Eq(lhs, 0) for lhs in condition)
            # 将生成器转换为 And 表达式
            condition = And(*temp)
            # 发出 sympy 废弃警告，提醒用户使用新的条件格式
            sympy_deprecation_warning(
                f"""
    Using a set for the condition in ConditionSet is deprecated. Use a boolean
    instead.

    In this case, replace

        {condition_orig}

    with

        {condition}
    """,
                deprecated_since_version='1.5',
                active_deprecations_target="deprecated-conditionset-set",
                )

        # Convert condition to a Boolean expression
        condition = as_Boolean(condition)

        # Return base_set if condition evaluates to true
        if condition is S.true:
            return base_set

        # Return EmptySet if condition evaluates to false
        if condition is S.false:
            return S.EmptySet

        # If base_set is EmptySet, return EmptySet
        if base_set is S.EmptySet:
            return S.EmptySet

        # Check if any symbols need to be processed
        for i in flat:
            # Ensure each item in flat has a '_diff_wrt' attribute
            if not getattr(i, '_diff_wrt', False):
                raise ValueError('`%s` is not symbol-like' % i)

        # Raise TypeError if sym is not in base_set
        if base_set.contains(sym) is S.false:
            raise TypeError('sym `%s` is not in base_set `%s`' % (sym, base_set))

        know = None
        # Handle FiniteSet instances in base_set
        if isinstance(base_set, FiniteSet):
            # Separate base_set into True and None based on condition evaluation
            sifted = sift(
                base_set, lambda _: fuzzy_bool(condition.subs(sym, _)))
            if sifted[None]:
                know = FiniteSet(*sifted[True])
                base_set = FiniteSet(*sifted[None])
            else:
                return FiniteSet(*sifted[True])

        # Handle instances of cls (likely ConditionSet)
        if isinstance(base_set, cls):
            s, c, b = base_set.args

            # Define a function to generate a signature based on sym
            def sig(s):
                return cls(s, Eq(adummy, 0)).as_dummy().sym

            # Generate signatures for sym and s
            sa, sb = map(sig, (sym, s))

            # Raise error if signatures of sym and s do not match
            if sa != sb:
                raise BadSignatureError('sym does not match sym of base set')

            reps = dict(zip(flatten([sym]), flatten([s])))

            # Handle different cases based on conditions and symbols
            if s == sym:
                condition = And(condition, c)
                base_set = b
            elif not c.free_symbols & sym.free_symbols:
                reps = {v: k for k, v in reps.items()}
                condition = And(condition, c.xreplace(reps))
                base_set = b
            elif not condition.free_symbols & s.free_symbols:
                sym = sym.xreplace(reps)
                condition = And(condition.xreplace(reps), c)
                base_set = b

        # Flatten ConditionSet(Contains(ConditionSet())) expressions if applicable
        if isinstance(condition, Contains) and (sym == condition.args[0]):
            if isinstance(condition.args[1], Set):
                return condition.args[1].intersect(base_set)

        # Create a new instance of cls with sym, condition, and base_set
        rv = Basic.__new__(cls, sym, condition, base_set)
        return rv if know is None else Union(know, rv)

    # Define properties for sym, condition, and base_set
    sym = property(lambda self: self.args[0])
    condition = property(lambda self: self.args[1])
    base_set = property(lambda self: self.args[2])

    # Define property to retrieve free symbols in the condition set
    @property
    def free_symbols(self):
        cond_syms = self.condition.free_symbols - self.sym.free_symbols
        return cond_syms | self.base_set.free_symbols

    # Define property to retrieve bound symbols in the condition set
    @property
    def bound_symbols(self):
        return flatten([self.sym])
    def _contains(self, other):
        # 定义用于检查符号类型是否匹配的函数
        def ok_sig(a, b):
            # 检查 a 和 b 是否是 Tuple 类型的实例
            tuples = [isinstance(i, Tuple) for i in (a, b)]
            # 统计其中一个是 Tuple 的个数
            c = tuples.count(True)
            # 如果只有一个是 Tuple，则返回 False
            if c == 1:
                return False
            # 如果两个都不是 Tuple，则返回 True
            if c == 0:
                return True
            # 如果两个都是 Tuple，检查它们的长度是否相同，并递归检查其中的元素
            return len(a) == len(b) and all(
                ok_sig(i, j) for i, j in zip(a, b))
        
        # 如果符号类型不匹配，则返回 S.false
        if not ok_sig(self.sym, other):
            return S.false

        # 尝试先对 base_set 执行 Contains 操作，并在结果为 False 时立即返回 S.false
        base_cond = Contains(other, self.base_set)
        if base_cond is S.false:
            return S.false

        # 将 other 替换到条件表达式中，可能会引发异常，例如 ConditionSet(x, 1/x >= 0, Reals).contains(0)
        lamda = Lambda((self.sym,), self.condition)
        try:
            lambda_cond = lamda(other)
        except TypeError:
            return None
        else:
            # 返回 base_cond 和 lambda_cond 的逻辑与结果
            return And(base_cond, lambda_cond)

    def as_relational(self, other):
        # 创建 Lambda 函数对象 f，以符号 self.sym 和条件 self.condition 初始化
        f = Lambda(self.sym, self.condition)
        # 如果 self.sym 是 Tuple 类型，则调用 f(*other)，否则调用 f(other)
        if isinstance(self.sym, Tuple):
            f = f(*other)
        else:
            f = f(other)
        # 返回 f 与 self.base_set.contains(other) 的逻辑与结果
        return And(f, self.base_set.contains(other))

    def _eval_subs(self, old, new):
        # 解构参数中的符号、条件和基础集合
        sym, cond, base = self.args
        # 将符号 sym 中的 old 替换为 adummy，生成新的 dsym
        dsym = sym.subs(old, adummy)
        # 检查 dsym 中是否包含 adummy
        insym = dsym.has(adummy)
        
        # 将 base 中的 old 替换为 new，生成新的 newbase
        newbase = base.subs(old, new)
        # 如果 newbase 与 base 不相同，则根据情况更新 cond
        if newbase != base:
            if not insym:
                cond = cond.subs(old, new)
            return self.func(sym, cond, newbase)
        
        # 如果 insym 为 True，则表示没有通过 subs 更改绑定的符号
        if insym:
            pass  # 不对通过 subs 更改绑定符号进行处理
        # 如果 new 具有 '_diff_wrt' 属性，则尝试更新 cond
        elif getattr(new, '_diff_wrt', False):
            cond = cond.subs(old, new)
        else:
            pass  # 让关于符号的错误在 __new__ 中引发
        # 返回更新后的对象
        return self.func(sym, cond, base)

    def _kind(self):
        # 返回符号 self.sym 的 SetKind 类型
        return SetKind(self.sym.kind)
```