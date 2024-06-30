# `D:\src\scipysrc\sympy\sympy\functions\elementary\piecewise.py`

```
# 从 sympy.core 模块中导入各种类和函数
from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
    _canonical, _canonical_coeff)
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
    true, false, Or, ITE, simplify_logic, to_cnf, distribute_or_over_and)
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name

# 从 itertools 模块中导入 product 函数
from itertools import product

# 定义 Undefined 为 S.NaN，表示未定义的值
Undefined = S.NaN  # Piecewise()

class ExprCondPair(Tuple):
    """Represents an expression, condition pair."""

    def __new__(cls, expr, cond):
        # 将 expr 转换为基本表达式
        expr = as_Basic(expr)
        if cond == True:
            # 如果条件为 True，则返回表达式和 true 的 Tuple
            return Tuple.__new__(cls, expr, true)
        elif cond == False:
            # 如果条件为 False，则返回表达式和 false 的 Tuple
            return Tuple.__new__(cls, expr, false)
        elif isinstance(cond, Basic) and cond.has(Piecewise):
            # 如果条件是基本表达式且包含 Piecewise，则进行处理
            cond = piecewise_fold(cond)  # 对 Piecewise 条件进行折叠
            if isinstance(cond, Piecewise):
                cond = cond.rewrite(ITE)  # 将 Piecewise 重写为 ITE 形式

        if not isinstance(cond, Boolean):
            # 如果条件不是 Boolean 类型，则抛出类型错误
            raise TypeError(filldedent('''
                Second argument must be a Boolean,
                not `%s`''' % func_name(cond)))
        # 返回表达式和条件的 Tuple
        return Tuple.__new__(cls, expr, cond)

    @property
    def expr(self):
        """
        Returns the expression of this pair.
        """
        return self.args[0]

    @property
    def cond(self):
        """
        Returns the condition of this pair.
        """
        return self.args[1]

    @property
    def is_commutative(self):
        # 返回表达式是否是可交换的
        return self.expr.is_commutative

    def __iter__(self):
        # 迭代器方法，返回表达式和条件
        yield self.expr
        yield self.cond

    def _eval_simplify(self, **kwargs):
        # 简化方法，返回简化后的表达式和条件的 Tuple
        return self.func(*[a.simplify(**kwargs) for a in self.args])


class Piecewise(Function):
    """
    Represents a piecewise function.

    Usage:

      Piecewise( (expr,cond), (expr,cond), ... )
        - Each argument is a 2-tuple defining an expression and condition
        - The conds are evaluated in turn returning the first that is True.
          If any of the evaluated conds are not explicitly False,
          e.g. ``x < 1``, the function is returned in symbolic form.
        - If the function is evaluated at a place where all conditions are False,
          nan will be returned.
        - Pairs where the cond is explicitly False, will be removed and no pair
          appearing after a True condition will ever be retained. If a single
          pair with a True condition remains, it will be returned, even when
          evaluation is False.

    Examples
    ========

    >>> from sympy import Piecewise, log, piecewise_fold
    >>> from sympy.abc import x, y
    >>> f = x**2
    >>> g = log(x)
    """
    # nargs 是 Piecewise 类的一个类属性，用于表示此类的参数数量，这里初始化为 None
    nargs = None

    # is_Piecewise 是 Piecewise 类的一个类属性，用于标识该类是否为 Piecewise 类型，这里初始化为 True
    is_Piecewise = True

    # __new__ 方法是 Python 类中的特殊方法，用于创建新的实例对象
    def __new__(cls, *args, **options):
        # 如果传入的参数数量为 0，则抛出 TypeError 异常
        if len(args) == 0:
            raise TypeError("At least one (expr, cond) pair expected.")

        # 创建一个空列表，用于存储经过 sympify 处理后的参数
        newargs = []
        # 遍历传入的参数列表 args
        for ec in args:
            # 如果 ec 是 ExprCondPair 类型，则调用它的 args 属性，否则直接使用 ec 作为 tuple
            pair = ExprCondPair(*getattr(ec, 'args', ec))
            # 获取条件 cond
            cond = pair.cond
            # 如果 cond 是 false，则跳过当前循环，不添加到 newargs 中
            if cond is false:
                continue
            # 添加处理后的 pair 到 newargs 中
            newargs.append(pair)
            # 如果 cond 是 true，则跳出循环，不再继续添加后续参数
            if cond is true:
                break

        # 获取 options 字典中的 evaluate 键的值，如果不存在则使用全局参数中的 evaluate 值
        eval = options.pop('evaluate', global_parameters.evaluate)
        
        # 如果 eval 为 True，尝试对 newargs 调用 cls.eval 方法进行求值
        if eval:
            r = cls.eval(*newargs)
            # 如果求值结果不为 None，则返回求值结果
            if r is not None:
                return r
        # 如果 newargs 中只有一个元素且其条件为 True，则返回该元素的表达式部分
        elif len(newargs) == 1 and newargs[0].cond == True:
            return newargs[0].expr
        
        # 调用 Basic 类的 __new__ 方法创建新的实例对象，传入 newargs 和 options
        return Basic.__new__(cls, *newargs, **options)

    # @classmethod 装饰器用于声明下面的方法为类方法
    def eval(cls, *_args):
        """评估 Piecewise 函数的条件，并返回相应的表达式或者 None。

        在这里进行的修改有：

        1. 使关系条件变得规范化
        2. 去除任何 False 的条件
        3. 忽略重复的前置条件
        4. 去除任何在首个 True 条件之后的条件

        如果没有剩余的条件，返回 Undefined。
        如果只有一个条件为 True，返回其对应的表达式。

        EXAMPLES
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> cond = -x < -1
        >>> args = [(1, cond), (4, cond), (3, False), (2, True), (5, x < 1)]
        >>> Piecewise(*args, evaluate=False)
        Piecewise((1, -x < -1), (4, -x < -1), (2, True))
        >>> Piecewise(*args)
        Piecewise((1, x > 1), (2, True))
        """
        if not _args:
            return Undefined

        if len(_args) == 1 and _args[0][-1] == True:
            return _args[0][0]

        # 对参数进行简化和合并
        newargs = _piecewise_collapse_arguments(_args)

        # 检查是否有冗余的条件
        missing = len(newargs) != len(_args)
        # 检查条件是否有改变
        same = all(a == b for a, b in zip(newargs, _args))
        # 如果有任何改变，返回更新后的表达式
        if missing or not same:
            return cls(*newargs)

    def doit(self, **hints):
        """
        对这个 Piecewise 函数进行求值。
        """
        newargs = []
        for e, c in self.args:
            if hints.get('deep', True):
                if isinstance(e, Basic):
                    # 如果 e 是 Basic 类型，则递归调用 doit 方法
                    newe = e.doit(**hints)
                    if newe != self:
                        e = newe
                if isinstance(c, Basic):
                    # 如果 c 是 Basic 类型，则递归调用 doit 方法
                    c = c.doit(**hints)
            newargs.append((e, c))
        return self.func(*newargs)

    def _eval_simplify(self, **kwargs):
        # 调用 piecewise_simplify 函数对 Piecewise 表达式进行简化
        return piecewise_simplify(self, **kwargs)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        for e, c in self.args:
            if c == True or c.subs(x, 0) == True:
                # 返回首项表达式的主导项
                return e.as_leading_term(x)

    def _eval_adjoint(self):
        # 返回每个元素的伴随操作结果
        return self.func(*[(e.adjoint(), c) for e, c in self.args])

    def _eval_conjugate(self):
        # 返回每个元素的共轭操作结果
        return self.func(*[(e.conjugate(), c) for e, c in self.args])

    def _eval_derivative(self, x):
        # 返回对 x 求导后的 Piecewise 表达式
        return self.func(*[(diff(e, x), c) for e, c in self.args])

    def _eval_evalf(self, prec):
        # 返回对每个元素进行数值评估后的结果
        return self.func(*[(e._evalf(prec), c) for e, c in self.args])
    def _eval_is_meromorphic(self, x, a):
        # 条件通常隐含假设参数是实数。
        # 因此，需要检查 as_set 是否存在。
        if not a.is_real:
            return None

        # 逐个扫描给定顺序中的 ExprCondPairs，找到包含 a 的片段，
        # 可能是作为边界点。
        for e, c in self.args:
            # 将 x 替换为 a，得到条件 cond。
            cond = c.subs(x, a)

            # 如果 cond 是关系式，则返回 None。
            if cond.is_Relational:
                return None

            # 如果 a 在 c 的 as_set 的边界上，则返回 None。
            if a in c.as_set().boundary:
                return None

            # 如果 a 是表达式 e 的定义域的内部点，则应用表达式。
            if cond:
                return e._eval_is_meromorphic(x, a)

    def piecewise_integrate(self, x, **kwargs):
        """返回 Piecewise 对象，其中每个表达式都替换为其不定积分。
        若要获得连续的不定积分，请使用 integrate 函数或方法。

        Examples
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> p = Piecewise((0, x < 0), (1, x < 1), (2, True))
        >>> p.piecewise_integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x, True))

        注意，这不会得到一个连续的函数，例如在 x = 1 处，第三个条件适用，
        其不定积分为 2*x，因此在那里的不定积分值为 2：

        >>> anti = _
        >>> anti.subs(x, 1)
        2

        连续的导数考虑到了到兴趣点的积分 *上限*，然而：

        >>> p.integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x - 1, True))
        >>> _.subs(x, 1)
        1

        See Also
        ========
        Piecewise._eval_integral
        """
        from sympy.integrals import integrate
        return self.func(*[(integrate(e, x, **kwargs), c) for e, c in self.args])

    def _eval_nseries(self, x, n, logx, cdir=0):
        # 对每个 ExprCondPairs 中的表达式求 n 阶级数展开，得到一个新的 Piecewise 对象。
        args = [(ec.expr._eval_nseries(x, n, logx), ec.cond) for ec in self.args]
        return self.func(*args)

    def _eval_power(self, s):
        # 对每个 ExprCondPairs 中的表达式求幂运算，得到一个新的 Piecewise 对象。
        return self.func(*[(e**s, c) for e, c in self.args])

    def _eval_subs(self, old, new):
        # 这不是严格必要的，但我们可以跟踪是否出现了 True 或 False 的条件，
        # 通过避免其他替换和避免出现在 True 条件之后的无效条件，可以更有效率。
        args = list(self.args)
        args_exist = False
        for i, (e, c) in enumerate(args):
            # 替换条件 c 中的 old 为 new。
            c = c._subs(old, new)
            if c != False:
                args_exist = True
                # 替换表达式 e 中的 old 为 new。
                e = e._subs(old, new)
            args[i] = (e, c)
            # 如果 c 变为 True，则终止循环。
            if c == True:
                break
        # 如果没有有效的 args，则设为 ((Undefined, True))。
        if not args_exist:
            args = ((Undefined, True),)
        return self.func(*args)

    def _eval_transpose(self):
        # 对每个 ExprCondPairs 中的表达式求转置，得到一个新的 Piecewise 对象。
        return self.func(*[(e.transpose(), c) for e, c in self.args])
    # 判断模板是否具有特定的属性（is_attr），并返回第一个非空属性值
    def _eval_template_is_attr(self, is_attr):
        b = None
        # 遍历表达式列表中的每个表达式及其相关参数
        for expr, _ in self.args:
            # 获取表达式对象的指定属性（is_attr）
            a = getattr(expr, is_attr)
            # 如果属性值为空，直接返回空值
            if a is None:
                return
            # 如果 b 为空，则将其设为当前属性值 a
            if b is None:
                b = a
            # 如果 b 不等于当前属性值 a，则返回空值
            elif b is not a:
                return
        # 返回第一个非空属性值 b
        return b

    # 使用 _eval_template_is_attr 方法判断是否具有 'is_finite' 属性
    _eval_is_finite = lambda self: self._eval_template_is_attr('is_finite')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_complex' 属性
    _eval_is_complex = lambda self: self._eval_template_is_attr('is_complex')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_even' 属性
    _eval_is_even = lambda self: self._eval_template_is_attr('is_even')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_imaginary' 属性
    _eval_is_imaginary = lambda self: self._eval_template_is_attr('is_imaginary')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_integer' 属性
    _eval_is_integer = lambda self: self._eval_template_is_attr('is_integer')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_irrational' 属性
    _eval_is_irrational = lambda self: self._eval_template_is_attr('is_irrational')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_negative' 属性
    _eval_is_negative = lambda self: self._eval_template_is_attr('is_negative')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_nonnegative' 属性
    _eval_is_nonnegative = lambda self: self._eval_template_is_attr('is_nonnegative')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_nonpositive' 属性
    _eval_is_nonpositive = lambda self: self._eval_template_is_attr('is_nonpositive')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_nonzero' 属性
    _eval_is_nonzero = lambda self: self._eval_template_is_attr('is_nonzero')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_odd' 属性
    _eval_is_odd = lambda self: self._eval_template_is_attr('is_odd')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_polar' 属性
    _eval_is_polar = lambda self: self._eval_template_is_attr('is_polar')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_positive' 属性
    _eval_is_positive = lambda self: self._eval_template_is_attr('is_positive')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_extended_real' 属性
    _eval_is_extended_real = lambda self: self._eval_template_is_attr('is_extended_real')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_extended_positive' 属性
    _eval_is_extended_positive = lambda self: self._eval_template_is_attr('is_extended_positive')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_extended_negative' 属性
    _eval_is_extended_negative = lambda self: self._eval_template_is_attr('is_extended_negative')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_extended_nonzero' 属性
    _eval_is_extended_nonzero = lambda self: self._eval_template_is_attr('is_extended_nonzero')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_extended_nonpositive' 属性
    _eval_is_extended_nonpositive = lambda self: self._eval_template_is_attr('is_extended_nonpositive')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_extended_nonnegative' 属性
    _eval_is_extended_nonnegative = lambda self: self._eval_template_is_attr('is_extended_nonnegative')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_real' 属性
    _eval_is_real = lambda self: self._eval_template_is_attr('is_real')
    # 使用 _eval_template_is_attr 方法判断是否具有 'is_zero' 属性
    _eval_is_zero = lambda self: self._eval_template_is_attr('is_zero')

    @classmethod
    # 类方法，用于评估条件的真假
    def __eval_cond(cls, cond):
        """Return the truth value of the condition."""
        # 如果条件为真，则返回 True
        if cond == True:
            return True
        # 如果条件为等式对象实例，则尝试计算其左右操作数的差值
        if isinstance(cond, Eq):
            try:
                diff = cond.lhs - cond.rhs
                # 如果差值具有交换性（commutative），则返回其是否为零的结果
                if diff.is_commutative:
                    return diff.is_zero
            except TypeError:
                pass
    def as_expr_set_pairs(self, domain=None):
        """Return tuples for each argument of self that give
        the expression and the interval in which it is valid
        which is contained within the given domain.
        If a condition cannot be converted to a set, an error
        will be raised. The variable of the conditions is
        assumed to be real; sets of real values are returned.

        Examples
        ========

        >>> from sympy import Piecewise, Interval
        >>> from sympy.abc import x
        >>> p = Piecewise(
        ...     (1, x < 2),
        ...     (2,(x > 0) & (x < 4)),
        ...     (3, True))
        >>> p.as_expr_set_pairs()
        [(1, Interval.open(-oo, 2)),
         (2, Interval.Ropen(2, 4)),
         (3, Interval(4, oo))]
        >>> p.as_expr_set_pairs(Interval(0, 3))
        [(1, Interval.Ropen(0, 2)),
         (2, Interval(2, 3))]
        """
        # 如果没有指定域，将域默认为实数集
        if domain is None:
            domain = S.Reals
        # 初始化表达式和集合的空列表
        exp_sets = []
        # 初始域为给定的 domain
        U = domain
        # 判断域是否复杂（即非实数集）
        complex = not domain.is_subset(S.Reals)
        # 用于存放条件中的自由符号
        cond_free = set()
        # 遍历分段函数的每个表达式和条件
        for expr, cond in self.args:
            # 将条件中的自由符号添加到集合中
            cond_free |= cond.free_symbols
            # 如果自由符号数量大于1，抛出未实现的错误
            if len(cond_free) > 1:
                raise NotImplementedError(filldedent('''
                    multivariate conditions are not handled.'''))
            # 如果域为复杂域
            if complex:
                # 检查条件中是否包含关系运算符
                for i in cond.atoms(Relational):
                    # 如果不是等式或不等式，抛出值错误
                    if not isinstance(i, (Eq, Ne)):
                        raise ValueError(filldedent('''
                            Inequalities in the complex domain are
                            not supported. Try the real domain by
                            setting domain=S.Reals'''))
            # 计算条件与当前域的交集
            cond_int = U.intersect(cond.as_set())
            # 更新当前域为原域减去条件与域的交集
            U = U - cond_int
            # 如果条件与域的交集不为空集，将表达式和交集加入结果列表
            if cond_int != S.EmptySet:
                exp_sets.append((expr, cond_int))
        # 返回表达式和集合的列表
        return exp_sets
    # 将参数中的条件重写为条件表达式ITE的形式
    def _eval_rewrite_as_ITE(self, *args, **kwargs):
        # 创建一个空字典以存储自由变量的条件
        byfree = {}
        # 将参数转换为列表
        args = list(args)
        # 判断是否存在默认条件
        default = any(c == True for b, c in args)
        # 遍历参数列表
        for i, (b, c) in enumerate(args):
            # 检查条件b是否为布尔类型或True
            if not isinstance(b, Boolean) and b != True:
                raise TypeError(filldedent('''
                    Expecting Boolean or bool but got `%s`
                    ''' % func_name(b)))
            # 如果条件c为True，则跳出循环
            if c == True:
                break
            # 遍历条件c的独立子条件（如果c是Or类型则展开）
            for c in c.args if isinstance(c, Or) else [c]:
                # 获取条件c中的自由符号
                free = c.free_symbols
                # 弹出一个自由符号x
                x = free.pop()
                try:
                    # 更新byfree字典，将x对应的条件转换为集合并存储
                    byfree[x] = byfree.setdefault(
                        x, S.EmptySet).union(c.as_set())
                except NotImplementedError:
                    # 如果没有默认条件并且无法判断条件是否覆盖所有变量，则抛出错误
                    if not default:
                        raise NotImplementedError(filldedent('''
                            A method to determine whether a multivariate
                            conditional is consistent with a complete coverage
                            of all variables has not been implemented so the
                            rewrite is being stopped after encountering `%s`.
                            This error would not occur if a default expression
                            like `(foo, True)` were given.
                            ''' % c))
                # 如果byfree[x]包含全集或实数集，则将第i个条件折叠为True并跳出内部循环
                if byfree[x] in (S.UniversalSet, S.Reals):
                    args[i] = list(args[i])
                    c = args[i][1] = True
                    break
            # 如果条件c为True，则跳出外部循环
            if c == True:
                break
        # 如果条件c不为True，则抛出值错误
        if c != True:
            raise ValueError(filldedent('''
                Conditions must cover all reals or a final default
                condition `(foo, True)` must be given.
                '''))
        # 初始化最终结果为args列表中的最后一个元素的第一个元素
        last, _ = args[i]  # ignore all past ith arg
        # 逆序遍历args列表的前i个元素
        for a, c in reversed(args[:i]):
            # 使用ITE表达式重写last
            last = ITE(c, a, last)
        # 返回规范化的最终结果
        return _canonical(last)
    # 将当前函数重写为 KroneckerDelta 的形式
    def _eval_rewrite_as_KroneckerDelta(self, *args, **kwargs):
        # 导入 KroneckerDelta 函数
        from sympy.functions.special.tensor_functions import KroneckerDelta

        # 规则字典，指定逻辑运算符对应的替换规则
        rules = {
            And: [False, False],
            Or: [True, True],
            Not: [True, False],
            Eq: [None, None],
            Ne: [None, None]
        }

        # 未识别条件的异常类
        class UnrecognizedCondition(Exception):
            pass

        # 条件重写函数
        def rewrite(cond):
            # 如果条件是等式 Eq 类型，则使用 KroneckerDelta 函数重写
            if isinstance(cond, Eq):
                return KroneckerDelta(*cond.args)
            # 如果条件是不等式 Ne 类型，则使用 1 减去 KroneckerDelta 函数重写
            if isinstance(cond, Ne):
                return 1 - KroneckerDelta(*cond.args)

            # 获取条件类型和参数
            cls, args = type(cond), cond.args
            # 如果条件类型不在规则字典中，则抛出未识别条件异常
            if cls not in rules:
                raise UnrecognizedCondition(cls)

            # 根据规则进行条件重写
            b1, b2 = rules[cls]
            k = Mul(*[1 - rewrite(c) for c in args]) if b1 else Mul(*[rewrite(c) for c in args])

            # 如果 b2 为 True，则返回 1 减去 k，否则返回 k
            if b2:
                return 1 - k
            return k

        # 初始化条件列表和真值
        conditions = []
        true_value = None
        # 遍历参数中的值和条件
        for value, cond in args:
            # 如果条件类型在规则字典中，则加入条件列表
            if type(cond) in rules:
                conditions.append((value, cond))
            # 如果条件为真 S.true，则设置真值为当前值
            elif cond is S.true:
                if true_value is None:
                    true_value = value
            else:
                return

        # 如果存在真值，则结果初始化为真值
        if true_value is not None:
            result = true_value

            # 对条件列表进行逆序遍历
            for value, cond in conditions[::-1]:
                try:
                    # 尝试使用 rewrite 函数对条件进行重写，并根据结果更新 result
                    k = rewrite(cond)
                    result = k * value + (1 - k) * result
                except UnrecognizedCondition:
                    return

            # 返回最终计算结果
            return result
def piecewise_fold(expr, evaluate=True):
    """
    Takes an expression containing a piecewise function and returns the
    expression in piecewise form. In addition, any ITE conditions are
    rewritten in negation normal form and simplified.

    The final Piecewise is evaluated (default) but if the raw form
    is desired, send ``evaluate=False``; if trivial evaluation is
    desired, send ``evaluate=None`` and duplicate conditions and
    processing of True and False will be handled.

    Examples
    ========

    >>> from sympy import Piecewise, piecewise_fold, S
    >>> from sympy.abc import x
    >>> p = Piecewise((x, x < 1), (1, S(1) <= x))
    >>> piecewise_fold(x*p)
    Piecewise((x**2, x < 1), (x, True))

    See Also
    ========

    Piecewise
    piecewise_exclusive
    """
    # 如果表达式不是 Basic 类型或者不包含 Piecewise 函数，直接返回表达式本身
    if not isinstance(expr, Basic) or not expr.has(Piecewise):
        return expr

    new_args = []
    # 如果表达式是 ExprCondPair 或者 Piecewise 类型，则遍历其参数
    if isinstance(expr, (ExprCondPair, Piecewise)):
        for e, c in expr.args:
            # 如果 e 不是 Piecewise 类型，则将 e 调用 piecewise_fold 递归处理
            if not isinstance(e, Piecewise):
                e = piecewise_fold(e)
            # 检查条件 c 是否包含 Piecewise，这在理论上应该不会发生
            assert not c.has(Piecewise)  # pragma: no cover
            # 如果条件 c 是 ITE 类型，则转换成否定正常形式并简化逻辑
            if isinstance(c, ITE):
                c = c.to_nnf()
                c = simplify_logic(c, form='cnf')
            # 如果 e 是 Piecewise 类型，则递归处理每个分支
            if isinstance(e, Piecewise):
                new_args.extend([(piecewise_fold(ei), And(ci, c))
                    for ei, ci in e.args])
            else:
                # 否则直接添加处理好的 e 和 c 到 new_args 中
                new_args.append((e, c))
    else:
        # 如果表达式不是 Add 或者 Mul，并且不是交换的，直接使用原始参数列表
        args = expr.args
        # 对每个参数进行 piecewise_fold 操作
        folded = list(map(piecewise_fold, args))
        # 对每个 folded 的组合进行迭代
        for ec in product(*[
                (i.args if isinstance(i, Piecewise) else
                 [(i, true)]) for i in folded]):
            e, c = zip(*ec)
            # 将每个组合的表达式和条件添加到 new_args 中
            new_args.append((expr.func(*e), And(*c)))

    if evaluate is None:
        # 如果 evaluate 是 None，则对 new_args 做一些处理以避免重复条件
        new_args = list(reversed([(e, c) for c, e in {
            c: e for e, c in reversed(new_args)}.items()]))
    # 创建新的 Piecewise 对象并返回
    rv = Piecewise(*new_args, evaluate=evaluate)
    # 如果 evaluate 是 None，并且 rv 只有一个参数并且条件为 True，则返回该参数的表达式部分
    if evaluate is None and len(rv.args) == 1 and rv.args[0].cond == True:
        return rv.args[0].expr
    # 如果 rv 中的任何子表达式包含 Piecewise，则再次进行 piecewise_fold 操作
    if any(s.expr.has(Piecewise) for p in rv.atoms(Piecewise) for s in p.args):
        return piecewise_fold(rv)
    # 否则返回 rv
    return rv
# 定义一个函数 `_clip`，用于处理两个区间 A 和 B 的交集
def _clip(A, B, k):
    """Return interval B as intervals that are covered by A (keyed
    to k) and all other intervals of B not covered by A keyed to -1.

    The reference point of each interval is the rhs; if the lhs is
    greater than the rhs then an interval of zero width interval will
    result, e.g. (4, 1) is treated like (1, 1).

    Examples
    ========

    >>> from sympy.functions.elementary.piecewise import _clip
    >>> from sympy import Tuple
    >>> A = Tuple(1, 3)
    >>> B = Tuple(2, 4)
    >>> _clip(A, B, 0)
    [(2, 3, 0), (3, 4, -1)]

    Interpretation: interval portion (2, 3) of interval (2, 4) is
    covered by interval (1, 3) and is keyed to 0 as requested;
    interval (3, 4) was not covered by (1, 3) and is keyed to -1.
    """
    # 解包区间 B 的起止点
    a, b = B
    # 解包区间 A 的起止点
    c, d = A
    # 通过 Min 和 Max 函数计算交集的起止点，确保交集的有效性
    c, d = Min(Max(c, a), b), Min(Max(d, a), b)
    # 计算区间 A 的左端点
    a = Min(a, b)
    # 初始化结果列表
    p = []
    # 检查并添加不在交集内的部分到结果列表，标记为 -1
    if a != c:
        p.append((a, c, -1))
    else:
        pass
    # 添加交集部分到结果列表，并标记为 k
    if c != d:
        p.append((c, d, k))
    else:
        pass
    # 检查并添加不在交集内的部分到结果列表，标记为 -1
    if b != d:
        if d == c and p and p[-1][-1] == -1:
            p[-1] = p[-1][0], b, -1
        else:
            p.append((d, b, -1))
    else:
        pass

    # 返回处理后的结果列表
    return p


def piecewise_simplify_arguments(expr, **kwargs):
    # 导入 simplify 函数
    from sympy.simplify.simplify import simplify

    # 简化表达式 expr 的条件
    f1 = expr.args[0].cond.free_symbols
    # 初始化参数为 None
    args = None
    # 如果表达式的项数为1且不包含等式，则执行以下操作
    if len(f1) == 1 and not expr.atoms(Eq):
        # 弹出集合中的唯一元素
        x = f1.pop()
        
        # 对表达式进行区间求解，排除包含等式的情况
        # 不处理作为布尔值处理的符号
        ok, abe_ = expr._intervals(x, err_on_Eq=True)
        
        # 定义一个函数，检查替换后条件是否为True
        def include(c, x, a):
            "return True if c.subs(x, a) is True, else False"
            try:
                return c.subs(x, a) == True
            except TypeError:
                return False
        
        # 如果区间求解成功
        if ok:
            args = []  # 初始化一个空列表，用于存储最终的参数
            covered = S.EmptySet  # 初始化一个空集合，表示已覆盖的区间
            
            # 导入Interval类
            from sympy.sets.sets import Interval
            
            # 遍历区间求解结果
            for a, b, e, i in abe_:
                # 获取子表达式的条件
                c = expr.args[i].cond
                
                # 判断a和b是否满足条件c
                incl_a = include(c, x, a)
                incl_b = include(c, x, b)
                
                # 创建一个区间，表示[a, b]，根据条件设置是否包含边界
                iv = Interval(a, b, not incl_a, not incl_b)
                
                # 计算差集，即去除已覆盖的部分
                cset = iv - covered
                
                # 如果差集为空，则继续下一轮循环
                if not cset:
                    continue
                
                try:
                    a = cset.inf  # 尝试获取差集的下界
                except NotImplementedError:
                    pass  # 如果未实现，则继续使用给定的a
                else:
                    # 如果成功获取下界，再次检查是否满足条件c
                    incl_a = include(c, x, a)
                
                # 根据incl_a和incl_b的情况，确定最终的条件c
                if incl_a and incl_b:
                    if a.is_infinite and b.is_infinite:
                        c = S.true
                    elif b.is_infinite:
                        c = (x > a) if a in covered else (x >= a)
                    elif a.is_infinite:
                        c = (x <= b)
                    elif a in covered:
                        c = And(a < x, x <= b)
                    else:
                        c = And(a <= x, x <= b)
                elif incl_a:
                    if a.is_infinite:
                        c = (x < b)
                    elif a in covered:
                        c = And(a < x, x < b)
                    else:
                        c = And(a <= x, x < b)
                elif incl_b:
                    if b.is_infinite:
                        c = (x > a)
                    else:
                        c = And(a < x, x <= b)
                else:
                    if a in covered:
                        c = (x < b)
                    else:
                        c = And(a < x, x < b)
                
                # 更新已覆盖的区间
                covered |= iv
                
                # 如果a是负无穷且incl_a为True，则更新已覆盖的负无穷区间
                if a is S.NegativeInfinity and incl_a:
                    covered |= {S.NegativeInfinity}
                
                # 如果b是正无穷且incl_b为True，则更新已覆盖的正无穷区间
                if b is S.Infinity and incl_b:
                    covered |= {S.Infinity}
                
                # 将最终的表达式e和条件c添加到参数列表args中
                args.append((e, c))
            
            # 如果未完全覆盖实数集，则添加一个未定义的标志True到参数列表args中
            if not S.Reals.is_subset(covered):
                args.append((Undefined, True))
    
    # 如果参数args为None，则初始化为表达式的所有参数列表
    if args is None:
        args = list(expr.args)
        
        # 遍历参数列表，简化条件c
        for i in range(len(args)):
            e, c = args[i]
            if isinstance(c, Basic):
                c = simplify(c, **kwargs)
            args[i] = (e, c)
    
    # 简化表达式
    doit = kwargs.pop('doit', None)
    # 遍历参数列表中的每个元素
    for i in range(len(args)):
        # 从元素中解包出表达式 e 和条件 c
        e, c  = args[i]
        # 检查 e 是否为 Basic 类型的对象
        if isinstance(e, Basic):
            # 对 e 进行简化，但不执行可能导致每次调用时增长的操作
            # 详情见 sympy/sympy#17165
            newe = simplify(e, doit=False, **kwargs)
            # 如果简化后的结果 newe 不等于原始表达式 e，则更新 e
            if newe != e:
                e = newe
        # 更新参数列表中的元素为简化后的表达式 e 和原始条件 c
        args[i] = (e, c)

    # 恢复 kwargs 中的 doit 标志位
    if doit is not None:
        kwargs['doit'] = doit

    # 返回一个 Piecewise 对象，其中包含经过处理后的参数列表
    return Piecewise(*args)
def _piecewise_collapse_arguments(_args):
    newargs = []  # 存储尚未评估的条件
    current_cond = set()  # 存储到给定 e, c 对的条件
    return newargs


_blessed = lambda e: getattr(e.lhs, '_diff_wrt', False) and (
    getattr(e.rhs, '_diff_wrt', None) or
    isinstance(e.rhs, (Rational, NumberSymbol)))


def piecewise_simplify(expr, **kwargs):
    expr = piecewise_simplify_arguments(expr, **kwargs)  # 简化参数表达式
    if not isinstance(expr, Piecewise):
        return expr
    args = list(expr.args)

    args = _piecewise_simplify_eq_and(args)  # 简化相等条件部分
    args = _piecewise_simplify_equal_to_next_segment(args)  # 简化与下一分段相等的表达式
    return Piecewise(*args)


def _piecewise_simplify_equal_to_next_segment(args):
    """
    检查表达式是否在一个相等条件下等同于下一段分段函数的表达式，参见：
    https://github.com/sympy/sympy/issues/8458
    """
    prevexpr = None
    for i, (expr, cond) in reversed(list(enumerate(args))):
        if prevexpr is not None:
            if isinstance(cond, And):
                eqs, other = sift(cond.args,
                                  lambda i: isinstance(i, Eq), binary=True)
            elif isinstance(cond, Eq):
                eqs, other = [cond], []
            else:
                eqs = other = []
            _prevexpr = prevexpr
            _expr = expr
            if eqs and not other:
                eqs = list(ordered(eqs))
                for e in eqs:
                    # 允许两个参数合并为一个，对于任意 e
                    # 否则仅限于简单参数的相等实例进行简化
                    if len(args) == 2 or _blessed(e):
                        _prevexpr = _prevexpr.subs(*e.args)
                        _expr = _expr.subs(*e.args)
            # 是否评估为相同？
            if _prevexpr == _expr:
                # 设置不相等部分的表达式为与下一段相同，
                # 在创建新的 Piecewise 时这些将合并
                args[i] = args[i].func(args[i + 1][0], cond)
            else:
                # 更新用于比较的表达式
                prevexpr = expr
        else:
            prevexpr = expr
    return args


def _piecewise_simplify_eq_and(args):
    """
    尝试简化条件和表达式中的相等性，例如：
    Piecewise((n, And(Eq(n,0), Eq(n + m, 0))), (1, True))
    -> Piecewise((0, And(Eq(n, 0), Eq(m, 0))), (1, True))
    """
    # 遍历参数列表 `args` 中的每个元素 `(expr, cond)`，同时获取索引 `i`
    for i, (expr, cond) in enumerate(args):
        # 检查条件 `cond` 是否为 `And` 类型
        if isinstance(cond, And):
            # 将 `cond` 中的子表达式分离成等式 (`Eq`) 和其他表达式
            eqs, other = sift(cond.args,
                              lambda i: isinstance(i, Eq), binary=True)
        # 如果条件 `cond` 是 `Eq` 类型
        elif isinstance(cond, Eq):
            # 将 `cond` 包装成列表形式，作为等式 (`Eq`)
            eqs, other = [cond], []
        else:
            # 否则，设定等式 (`Eq`) 和其他表达式为空列表
            eqs = other = []
        
        # 如果存在等式 (`eqs`)
        if eqs:
            # 将等式列表按顺序排列
            eqs = list(ordered(eqs))
            # 遍历等式列表 `eqs` 中的每个元素 `(j, e)`
            for j, e in enumerate(eqs):
                # 检查对象 `e` 是否为特定类型 `_blessed`
                # 并将表达式 `expr` 中的参数替换为 `e` 中的参数
                if _blessed(e):
                    expr = expr.subs(*e.args)
                    # 替换等式列表 `eqs` 中的后续等式
                    eqs[j + 1:] = [ei.subs(*e.args) for ei in eqs[j + 1:]]
                    # 替换其他表达式 `other` 中的参数
                    other = [ei.subs(*e.args) for ei in other]
            
            # 重新构建条件 `cond`，包括更新后的等式列表和其他表达式
            cond = And(*(eqs + other))
            # 将经过参数替换后的表达式 `expr` 和更新后的条件 `cond` 应用于原函数
            args[i] = args[i].func(expr, cond)
    
    # 返回更新后的参数列表 `args`
    return args
# 定义函数 piecewise_exclusive，用于将 Piecewise 对象的条件重写为互斥条件
def piecewise_exclusive(expr, *, skip_nan=False, deep=True):
    """
    Rewrite :class:`Piecewise` with mutually exclusive conditions.

    Explanation
    ===========

    SymPy represents the conditions of a :class:`Piecewise` in an
    "if-elif"-fashion, allowing more than one condition to be simultaneously
    True. The interpretation is that the first condition that is True is the
    case that holds. While this is a useful representation computationally it
    is not how a piecewise formula is typically shown in a mathematical text.
    The :func:`piecewise_exclusive` function can be used to rewrite any
    :class:`Piecewise` with more typical mutually exclusive conditions.

    Note that further manipulation of the resulting :class:`Piecewise`, e.g.
    simplifying it, will most likely make it non-exclusive. Hence, this is
    primarily a function to be used in conjunction with printing the Piecewise
    or if one would like to reorder the expression-condition pairs.

    If it is not possible to determine that all possibilities are covered by
    the different cases of the :class:`Piecewise` then a final
    :class:`~sympy.core.numbers.NaN` case will be included explicitly. This
    can be prevented by passing ``skip_nan=True``.

    Examples
    ========

    >>> from sympy import piecewise_exclusive, Symbol, Piecewise, S
    >>> x = Symbol('x', real=True)
    >>> p = Piecewise((0, x < 0), (S.Half, x <= 0), (1, True))
    >>> piecewise_exclusive(p)
    Piecewise((0, x < 0), (1/2, Eq(x, 0)), (1, x > 0))
    >>> piecewise_exclusive(Piecewise((2, x > 1)))
    Piecewise((2, x > 1), (nan, x <= 1))
    >>> piecewise_exclusive(Piecewise((2, x > 1)), skip_nan=True)
    Piecewise((2, x > 1))

    Parameters
    ==========

    expr: a SymPy expression.
        Any :class:`Piecewise` in the expression will be rewritten.
    skip_nan: ``bool`` (default ``False``)
        If ``skip_nan`` is set to ``True`` then a final
        :class:`~sympy.core.numbers.NaN` case will not be included.
    deep:  ``bool`` (default ``True``)
        If ``deep`` is ``True`` then :func:`piecewise_exclusive` will rewrite
        any :class:`Piecewise` subexpressions in ``expr`` rather than just
        rewriting ``expr`` itself.

    Returns
    =======

    An expression equivalent to ``expr`` but where all :class:`Piecewise` have
    been rewritten with mutually exclusive conditions.

    See Also
    ========

    Piecewise
    piecewise_fold
    """
    # 定义一个函数 make_exclusive，接受可变数量的位置参数 pwargs
    def make_exclusive(*pwargs):

        # 初始化累积条件为 False
        cumcond = false
        # 初始化新参数列表为空
        newargs = []

        # 处理前 n-1 个条件情况
        for expr_i, cond_i in pwargs[:-1]:
            # 计算当前条件的可行条件
            cancond = And(cond_i, Not(cumcond)).simplify()
            # 更新累积条件
            cumcond = Or(cond_i, cumcond).simplify()
            # 将处理后的表达式和条件添加到新参数列表中
            newargs.append((expr_i, cancond))

        # 处理第 n 个条件，延迟简化累积条件的处理
        expr_n, cond_n = pwargs[-1]
        # 计算第 n 个条件的可行条件
        cancond_n = And(cond_n, Not(cumcond)).simplify()
        # 将处理后的表达式和条件添加到新参数列表中
        newargs.append((expr_n, cancond_n))

        # 如果不跳过 NaN 的情况
        if not skip_nan:
            # 更新累积条件
            cumcond = Or(cond_n, cumcond).simplify()
            # 如果累积条件不是 True，则添加未定义的情况到新参数列表中
            if cumcond is not true:
                newargs.append((Undefined, Not(cumcond).simplify()))

        # 返回一个 Piecewise 对象，不进行评估
        return Piecewise(*newargs, evaluate=False)

    # 如果设置了 deep 参数，则对表达式中的 Piecewise 进行替换
    if deep:
        return expr.replace(Piecewise, make_exclusive)
    # 如果表达式是 Piecewise 类型，则调用 make_exclusive 处理它的参数
    elif isinstance(expr, Piecewise):
        return make_exclusive(*expr.args)
    # 否则直接返回原始表达式
    else:
        return expr
```