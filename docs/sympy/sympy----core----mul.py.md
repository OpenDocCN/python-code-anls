# `D:\src\scipysrc\sympy\sympy\core\mul.py`

```
from typing import Tuple as tTuple
from collections import defaultdict
from functools import reduce
from itertools import product
import operator

from .sympify import sympify
from .basic import Basic, _args_sortkey
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .intfunc import integer_nthroot, trailing
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up

from sympy.utilities.iterables import sift

# internal marker to indicate:
#   "there are still non-commutative objects -- don't forget to process them"
class NC_Marker:
    is_Order = False
    is_Mul = False
    is_Number = False
    is_Poly = False

    is_commutative = False


def _mulsort(args):
    # in-place sorting of args
    args.sort(key=_args_sortkey)


def _unevaluated_Mul(*args):
    """Return a well-formed unevaluated Mul: Numbers are collected and
    put in slot 0, any arguments that are Muls will be flattened, and args
    are sorted. Use this when args have changed but you still want to return
    an unevaluated Mul.

    Examples
    ========

    >>> from sympy.core.mul import _unevaluated_Mul as uMul
    >>> from sympy import S, sqrt, Mul
    >>> from sympy.abc import x
    >>> a = uMul(*[S(3.0), x, S(2)])
    >>> a.args[0]
    6.00000000000000
    >>> a.args[1]
    x

    Two unevaluated Muls with the same arguments will
    always compare as equal during testing:

    >>> m = uMul(sqrt(2), sqrt(3))
    >>> m == uMul(sqrt(3), sqrt(2))
    True
    >>> u = Mul(sqrt(3), sqrt(2), evaluate=False)
    >>> m == uMul(u)
    True
    >>> m == Mul(*m.args)
    False

    """
    args = list(args)
    newargs = []
    ncargs = []
    co = S.One
    while args:
        a = args.pop()
        if a.is_Mul:
            # Split the argument into commutative and non-commutative parts
            c, nc = a.args_cnc()
            args.extend(c)
            if nc:
                # Store non-commutative part as a Mul object
                ncargs.append(Mul._from_args(nc))
        elif a.is_Number:
            # Accumulate numerical coefficients
            co *= a
        else:
            # Collect regular arguments
            newargs.append(a)
    # Sort regular arguments
    _mulsort(newargs)
    # Insert accumulated coefficient if not unity
    if co is not S.One:
        newargs.insert(0, co)
    # Append non-commutative part if exists
    if ncargs:
        newargs.append(Mul._from_args(ncargs))
    # Return a new Mul object from sorted and structured arguments
    return Mul._from_args(newargs)


class Mul(Expr, AssocOp):
    """
    Expression representing multiplication operation for algebraic field.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Mul()`` must be ``Expr``. Infix operator ``*``
    on most scalar objects in SymPy calls this class.

    Another use of ``Mul()`` is to represent the structure of abstract
    multiplication so that its arguments can be substituted to return
    different class. Refer to examples section for this.
    ``Mul()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Mul(x, Mul(y, z))`` -> ``Mul(x, y, z)``

    2. Identity removing
        ``Mul(x, 1, y)`` -> ``Mul(x, y)``

    3. Exponent collecting by ``.as_base_exp()``
        ``Mul(x, x**2)`` -> ``Pow(x, 3)``

    4. Term sorting
        ``Mul(y, x, 2)`` -> ``Mul(2, x, y)``

    Since multiplication can be vector space operation, arguments may
    have the different :obj:`sympy.core.kind.Kind()`. Kind of the
    resulting object is automatically inferred.

    Examples
    ========

    >>> from sympy import Mul
    >>> from sympy.abc import x, y
    >>> Mul(x, 1)
    x
    >>> Mul(x, x)
    x**2

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Mul(1, 2, evaluate=False)
    1*2
    >>> Mul(x, x, evaluate=False)
    x*x

    ``Mul()`` also represents the general structure of multiplication
    operation.

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 2,2)
    >>> expr = Mul(x,y).subs({y:A})
    >>> expr
    x*A
    >>> type(expr)
    <class 'sympy.matrices.expressions.matmul.MatMul'>

    See Also
    ========

    MatMul

    """
    __slots__ = ()  # Define an empty slot tuple for instances of Mul

    args: tTuple[Expr, ...]  # Type hint indicating that args is a tuple of Expr objects

    is_Mul = True  # Flag indicating that this object is a multiplication expression

    _args_type = Expr  # Type of arguments expected (Expr objects)
    _kind_dispatcher = KindDispatcher("Mul_kind_dispatcher", commutative=True)
    # Kind dispatcher for determining the resulting kind of the multiplication expression

    @property
    def kind(self):
        arg_kinds = (a.kind for a in self.args)  # Iterate over argument kinds
        return self._kind_dispatcher(*arg_kinds)  # Determine and return the resulting kind

    def could_extract_minus_sign(self):
        if self == (-self):
            return False  # Check if the expression is its own negation
        c = self.args[0]  # Get the first argument
        return c.is_Number and c.is_extended_negative  # Check if it's a negative number

    def __neg__(self):
        c, args = self.as_coeff_mul()  # Get coefficient and remaining factors
        if args[0] is not S.ComplexInfinity:  # Ensure not dividing by ComplexInfinity
            c = -c  # Negate the coefficient
        if c is not S.One:  # If coefficient is not 1
            if args[0].is_Number:  # If the first argument is a number
                args = list(args)  # Convert arguments to list
                if c is S.NegativeOne:
                    args[0] = -args[0]  # Negate the first argument
                else:
                    args[0] *= c  # Multiply the first argument by the coefficient
            else:
                args = (c,) + args  # Include the coefficient as the first argument
        return self._from_args(args, self.is_commutative)  # Construct and return negated expression

    @classmethod
    def _eval_power(self, e):
        # 将表达式分为可交换项和不可交换项
        cargs, nc = self.args_cnc(split_1=False)

        # 如果指数 e 是整数
        if e.is_Integer:
            # 返回 Mul 类型对象，对可交换项 cargs 中的每个项求指数 e，不进行评估
            # 然后乘以不可交换项 nc 的 e 次方，同样不进行评估
            return Mul(*[Pow(b, e, evaluate=False) for b in cargs]) * \
                Pow(Mul._from_args(nc), e, evaluate=False)
        
        # 如果指数 e 是有理数且分母为 2
        if e.is_Rational and e.q == 2:
            # 如果当前对象是虚数
            if self.is_imaginary:
                # 取出虚部
                a = self.as_real_imag()[1]
                # 如果虚部也是有理数
                if a.is_Rational:
                    # 计算虚部的绝对值的平方根，并将其分解为整数部分和根号部分
                    n, d = abs(a/2).as_numer_denom()
                    n, t = integer_nthroot(n, 2)
                    # 如果能够完全开方
                    if t:
                        d, t = integer_nthroot(d, 2)
                        # 如果能够完全开方
                        if t:
                            # 导入符号函数
                            from sympy.functions.elementary.complexes import sign
                            # 计算结果 r
                            r = sympify(n)/d
                            # 返回一个未评估的 Mul 对象，其内容为 r 的 e 次方乘以 (1 + 符号函数值 * 虚数单位) 的 e 次方
                            return _unevaluated_Mul(r**e.p, (1 + sign(a)*S.ImaginaryUnit)**e.p)

        # 否则创建 Pow 对象
        p = Pow(self, e, evaluate=False)

        # 如果指数 e 是有理数或浮点数，则展开底数
        if e.is_Rational or e.is_Float:
            return p._eval_expand_power_base()

        # 返回 Pow 对象
        return p

    @classmethod
    def class_key(cls):
        # 返回一个元组，用于类的排序
        return 3, 0, cls.__name__

    def _eval_evalf(self, prec):
        # 将自身分解为系数和乘积的形式
        c, m = self.as_coeff_Mul()
        
        # 如果系数是 -1
        if c is S.NegativeOne:
            # 如果乘积 m 是 Mul 类型
            if m.is_Mul:
                # 对 m 调用 _eval_evalf 方法
                rv = -AssocOp._eval_evalf(m, prec)
            else:
                # 否则对 m 进行浮点数评估
                mnew = m._eval_evalf(prec)
                if mnew is not None:
                    m = mnew
                rv = -m
        else:
            # 否则对自身调用 _eval_evalf 方法
            rv = AssocOp._eval_evalf(self, prec)
        
        # 如果 rv 是数值类型，则展开其表达式
        if rv.is_number:
            return rv.expand()
        
        # 返回 rv
        return rv

    @property
    def _mpc_(self):
        """
        Convert self to an mpmath mpc if possible
        """
        # 导入 Float 类
        from .numbers import Float
        # 将自身分解为实部和虚部的乘积
        im_part, imag_unit = self.as_coeff_Mul()
        # 如果虚部不是虚数单位
        if imag_unit is not S.ImaginaryUnit:
            # 抛出属性错误，不能将 Mul 转换为 mpc，必须是形如 Number*I 的形式
            raise AttributeError("Cannot convert Mul to mpc. Must be of the form Number*I")

        # 返回一个元组，表示复数的实部和虚部
        return (Float(0)._mpf_, Float(im_part)._mpf_)

    @cacheit
    def as_two_terms(self):
        """Return head and tail of self.

        This is the most efficient way to get the head and tail of an
        expression.

        - if you want only the head, use self.args[0];
        - if you want to process the arguments of the tail then use
          self.as_coef_mul() which gives the head and a tuple containing
          the arguments of the tail when treated as a Mul.
        - if you want the coefficient when self is treated as an Add
          then use self.as_coeff_add()[0]

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> (3*x*y).as_two_terms()
        (3, x*y)
        """
        # 获取当前对象的参数列表
        args = self.args

        # 如果参数列表长度为1，返回 S.One 和当前对象本身
        if len(args) == 1:
            return S.One, self
        # 如果参数列表长度为2，直接返回参数列表
        elif len(args) == 2:
            return args
        # 否则，返回第一个参数和剩余参数组成的新对象
        else:
            return args[0], self._new_rawargs(*args[1:])

    @cacheit
    def as_coeff_mul(self, *deps, rational=True, **kwargs):
        # 如果指定了依赖项，则根据依赖项对参数进行筛选
        if deps:
            l1, l2 = sift(self.args, lambda x: x.has(*deps), binary=True)
            return self._new_rawargs(*l2), tuple(l1)
        # 否则，直接获取参数列表
        args = self.args
        # 如果第一个参数是数值类型
        if args[0].is_Number:
            # 如果不要求有理数或者第一个参数是有理数，则返回第一个参数和剩余参数
            if not rational or args[0].is_Rational:
                return args[0], args[1:]
            # 如果第一个参数是扩展负数，则返回 S.NegativeOne 和负的第一个参数及剩余参数
            elif args[0].is_extended_negative:
                return S.NegativeOne, (-args[0],) + args[1:]
        # 默认返回 S.One 和参数列表
        return S.One, args

    def as_coeff_Mul(self, rational=False):
        """
        Efficiently extract the coefficient of a product.
        """
        # 获取第一个参数作为系数，剩余参数作为列表
        coeff, args = self.args[0], self.args[1:]

        # 如果系数是数值类型
        if coeff.is_Number:
            # 如果不要求有理数或者系数是有理数
            if not rational or coeff.is_Rational:
                # 如果参数列表长度为1，返回系数和第一个参数
                if len(args) == 1:
                    return coeff, args[0]
                # 否则，返回系数和剩余参数组成的新对象
                else:
                    return coeff, self._new_rawargs(*args)
            # 如果系数是扩展负数，则返回 S.NegativeOne 和负的系数及剩余参数组成的新对象
            elif coeff.is_extended_negative:
                return S.NegativeOne, self._new_rawargs(*((-coeff,) + args))
        # 默认返回 S.One 和参数列表
        return S.One, self
    # 将对象转换为实部和虚部表示
    def as_real_imag(self, deep=True, **hints):
        # 导入必要的符号计算库函数
        from sympy.functions.elementary.complexes import Abs, im, re
        other = []  # 存储未处理的项
        coeffr = []  # 存储实系数的列表
        coeffi = []  # 存储虚系数的列表
        addterms = S.One  # 初始加法项为单位元
        for a in self.args:
            r, i = a.as_real_imag()  # 获取当前项的实部和虚部
            if i.is_zero:
                coeffr.append(r)  # 如果虚部为零，则将实部加入实系数列表
            elif r.is_zero:
                coeffi.append(i*S.ImaginaryUnit)  # 如果实部为零，则将虚部乘以虚数单位加入虚系数列表
            elif a.is_commutative:
                aconj = a.conjugate() if other else None
                # 搜索复共轭对：
                for idx, x in enumerate(other):
                    if x == aconj:
                        coeffr.append(Abs(x)**2)  # 找到复共轭对时，添加其模的平方到实系数列表
                        del other[idx]
                        break
                else:
                    if a.is_Add:
                        addterms *= a  # 如果当前项是加法，则更新加法项
                    else:
                        other.append(a)  # 否则将当前项加入未处理列表
            else:
                other.append(a)  # 非可交换项直接加入未处理列表
        m = self.func(*other)  # 用剩余的项重新构造对象
        if hints.get('ignore') == m:
            return  # 如果忽略标记等于当前对象，直接返回
        if len(coeffi) % 2:
            imco = im(coeffi.pop(0))  # 弹出虚系数列表的第一个元素作为纯虚部分
            # 其他偶数对应的项会形成一个实数因子，它们将被放在下面的 reco 中
        else:
            imco = S.Zero  # 如果虚系数数量为偶数，则纯虚部分为零
        reco = self.func(*(coeffr + coeffi))  # 构造实部和虚部对象
        r, i = (reco*re(m), reco*im(m))  # 计算实部和虚部的乘积
        if addterms == 1:
            if m == 1:
                if imco.is_zero:
                    return (reco, S.Zero)  # 如果 m 是 1，且纯虚部分为零，则返回实部和零
                else:
                    return (S.Zero, reco*imco)  # 否则返回零和纯虚部的乘积
            if imco is S.Zero:
                return (r, i)  # 如果纯虚部为零，则返回实部和虚部
            return (-imco*i, imco*r)  # 否则返回纯虚部和实部的乘积
        from .function import expand_mul
        addre, addim = expand_mul(addterms, deep=False).as_real_imag()  # 扩展加法项并转换为实部和虚部表示
        if imco is S.Zero:
            return (r*addre - i*addim, i*addre + r*addim)  # 如果纯虚部为零，则返回乘积结果
        else:
            r, i = -imco*i, imco*r  # 否则更新实部和虚部
            return (r*addre - i*addim, r*addim + i*addre)  # 返回乘积结果

    @staticmethod
    def _expandsums(sums):
        """
        _eval_expand_mul 的辅助函数。

        sums 必须是 Basic 实例的列表。
        """

        L = len(sums)
        if L == 1:
            return sums[0].args  # 如果列表长度为1，直接返回第一个元素的参数列表
        terms = []
        left = Mul._expandsums(sums[:L//2])  # 递归地扩展左半部分
        right = Mul._expandsums(sums[L//2:])  # 递归地扩展右半部分

        terms = [Mul(a, b) for a in left for b in right]  # 对左右两部分进行乘法组合
        added = Add(*terms)  # 将所有乘积组合加起来
        return Add.make_args(added)  # 返回加法结果的参数列表（可能会被合并为一个项）
    @cacheit
    def _eval_derivative(self, s):
        # 将表达式的参数列表复制一份
        args = list(self.args)
        # 用于存放求导后的项
        terms = []
        # 遍历参数列表
        for i in range(len(args)):
            # 对第 i 个参数求关于 s 的偏导数
            d = args[i].diff(s)
            # 如果求导结果不为空
            if d:
                # 构造新的乘积表达式，将其它参数与求导结果乘积起来
                # 使用 reduce 函数是为了在乘法运算中处理优先级和子类型问题
                terms.append(reduce(lambda x, y: x*y, (args[:i] + [d] + args[i + 1:]), S.One))
        # 返回所有项的和作为最终的导数表达式
        return Add.fromiter(terms)
    def _eval_derivative_n_times(self, s, n):
        # 导入必要的模块和类
        from .function import AppliedUndef
        from .symbol import Symbol, symbols, Dummy
        
        # 如果 s 不是 AppliedUndef 或 Symbol 类型，则调用父类方法处理
        if not isinstance(s, (AppliedUndef, Symbol)):
            # 其他类型的 s 可能表现不佳，例如 (cos(x)*sin(y)).diff([[x, y, z]])
            return super()._eval_derivative_n_times(s, n)
        
        # 导入整数和 Integer 类
        from .numbers import Integer
        # 获取自身的参数列表
        args = self.args
        m = len(args)
        
        # 如果 n 是整数或 Integer 类型
        if isinstance(n, (int, Integer)):
            # 使用 multinomial_coefficients_iterator 生成多项式系数
            from sympy.ntheory.multinomial import multinomial_coefficients_iterator
            terms = []
            # 遍历多项式系数生成器的结果
            for kvals, c in multinomial_coefficients_iterator(m, n):
                # 计算偏导数乘积
                p = Mul(*[arg.diff((s, k)) for k, arg in zip(kvals, args)])
                terms.append(c * p)
            # 返回乘积求和结果
            return Add(*terms)
        
        # 导入求和函数和阶乘函数
        from sympy.concrete.summations import Sum
        from sympy.functions.combinatorial.factorials import factorial
        from sympy.functions.elementary.miscellaneous import Max
        
        # 生成虚拟符号作为求和的指标
        kvals = symbols("k1:%i" % m, cls=Dummy)
        klast = n - sum(kvals)
        nfact = factorial(n)
        
        # 使用 multinomial 计算
        e, l = (
            nfact / prod(map(factorial, kvals)) / factorial(klast) *
            Mul(*[args[t].diff((s, kvals[t])) for t in range(m - 1)]) *
            args[-1].diff((s, Max(0, klast))),
            [(k, 0, n) for k in kvals]
        )
        # 返回求和表达式
        return Sum(e, *l)

    def _eval_difference_delta(self, n, step):
        # 导入差分函数
        from sympy.series.limitseq import difference_delta as dd
        arg0 = self.args[0]
        rest = Mul(*self.args[1:])
        
        # 返回差分公式的结果
        return (arg0.subs(n, n + step) * dd(rest, n, step) + dd(arg0, n, step) *
                rest)

    def _matches_simple(self, expr, repl_dict):
        # 处理单个项的匹配
        # 获取乘法表达式的系数和因子
        coeff, terms = self.as_coeff_Mul()
        terms = Mul.make_args(terms)
        
        # 如果只有一个因子
        if len(terms) == 1:
            # 将表达式和系数组合为新表达式
            newexpr = self.__class__._combine_inverse(expr, coeff)
            # 调用因子的匹配方法
            return terms[0].matches(newexpr, repl_dict)
        
        # 如果有多个因子则返回空
        return
    def matches(self, expr, repl_dict=None, old=False):
        # 将表达式转换为符号表达式
        expr = sympify(expr)
        # 如果当前对象和表达式都是可交换的
        if self.is_commutative and expr.is_commutative:
            # 调用可交换表达式匹配函数
            return self._matches_commutative(expr, repl_dict, old)
        # 如果一个是可交换的，一个不是，则返回 None
        elif self.is_commutative is not expr.is_commutative:
            return None

        # 只有当两个表达式都是非可交换的时候才继续
        # 分解自身和表达式的可交换和非可交换部分
        c1, nc1 = self.args_cnc()
        c2, nc2 = expr.args_cnc()
        # 如果某个部分为空，则用默认值 [1] 替代
        c1, c2 = [c or [1] for c in [c1, c2]]

        # 创建当前对象和表达式的可交换部分乘积
        comm_mul_self = Mul(*c1)
        comm_mul_expr = Mul(*c2)

        # 调用可交换部分匹配函数
        repl_dict = comm_mul_self.matches(comm_mul_expr, repl_dict, old)

        # 如果可交换部分没有匹配，并且两者不相等，则返回 None
        if not repl_dict and c1 != c2:
            return None

        # 现在匹配非可交换部分，将指数展开成乘积
        nc1 = Mul._matches_expand_pows(nc1)
        nc2 = Mul._matches_expand_pows(nc2)

        # 调用非可交换部分匹配函数
        repl_dict = Mul._matches_noncomm(nc1, nc2, repl_dict)

        # 如果 repl_dict 不为空，则返回 repl_dict，否则返回 None
        return repl_dict or None

    @staticmethod
    def _matches_expand_pows(arg_list):
        """展开指数为乘积的静态方法。

        Args:
            arg_list: 待处理的参数列表。

        Returns:
            list: 处理后的参数列表，指数大于 0 的符号展开为多个同一基数。
        """
        new_args = []
        for arg in arg_list:
            if arg.is_Pow and arg.exp > 0:
                new_args.extend([arg.base] * arg.exp)
            else:
                new_args.append(arg)
        return new_args

    @staticmethod
    def _matches_noncomm(nodes, targets, repl_dict=None):
        """非可交换乘法匹配器的静态方法。

        Args:
            nodes (list): 匹配器乘法表达式中的符号列表。
            targets (list): 被匹配乘法表达式中的参数列表。
            repl_dict (dict, optional): 可选的替换字典。默认为 None。

        Returns:
            dict: 成功匹配的替换字典，或者 None。
        """
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        # 记录可能的未来状态的列表
        agenda = []
        # 当前匹配状态，存储 nodes 和 targets 中的索引
        state = (0, 0)
        node_ind, target_ind = state
        # 通配符索引与其匹配的索引范围之间的映射
        wildcard_dict = {}

        while target_ind < len(targets) and node_ind < len(nodes):
            node = nodes[node_ind]

            if node.is_Wild:
                Mul._matches_add_wildcard(wildcard_dict, state)

            states_matches = Mul._matches_new_states(wildcard_dict, state,
                                                     nodes, targets)
            if states_matches:
                new_states, new_matches = states_matches
                agenda.extend(new_states)
                if new_matches:
                    for match in new_matches:
                        repl_dict[match] = new_matches[match]
            if not agenda:
                return None
            else:
                state = agenda.pop()
                node_ind, target_ind = state

        return repl_dict

    @staticmethod
    @staticmethod
    def _matches_add_wildcard(dictionary, state):
        # 将状态字典中的节点索引和目标索引解构赋值给变量
        node_ind, target_ind = state
        # 如果节点索引已存在于字典中，则更新其起始和结束位置；否则将其添加到字典中并设定起始和结束位置为目标索引
        if node_ind in dictionary:
            begin, end = dictionary[node_ind]
            dictionary[node_ind] = (begin, target_ind)
        else:
            dictionary[node_ind] = (target_ind, target_ind)

    @staticmethod
    def _matches_new_states(dictionary, state, nodes, targets):
        # 将状态字典中的节点索引和目标索引解构赋值给变量
        node_ind, target_ind = state
        # 获取节点列表中的当前节点和目标列表中的当前目标
        node = nodes[node_ind]
        target = targets[target_ind]

        # 如果已经用尽目标但未用尽节点，则不进行任何前进操作
        if target_ind >= len(targets) - 1 and node_ind < len(nodes) - 1:
            return None

        # 如果当前节点是通配符
        if node.is_Wild:
            # 尝试匹配通配符节点到目标列表中的当前位置
            match_attempt = Mul._matches_match_wilds(dictionary, node_ind,
                                                     nodes, targets)
            if match_attempt:
                # 如果当前节点之前已经有匹配，则检查当前匹配是否与之前的匹配不同
                other_node_inds = Mul._matches_get_other_nodes(dictionary,
                                                               nodes, node_ind)
                for ind in other_node_inds:
                    other_begin, other_end = dictionary[ind]
                    curr_begin, curr_end = dictionary[node_ind]

                    other_targets = targets[other_begin:other_end + 1]
                    current_targets = targets[curr_begin:curr_end + 1]

                    # 逐个比较当前匹配和之前匹配的目标
                    for curr, other in zip(current_targets, other_targets):
                        if curr != other:
                            return None

                # 通配符节点可以匹配多个目标，因此仅将目标索引前进一步
                new_state = [(node_ind, target_ind + 1)]
                # 如果存在下一个节点，则同时前进节点索引和目标索引
                if node_ind < len(nodes) - 1:
                    new_state.append((node_ind + 1, target_ind + 1))
                return new_state, match_attempt
        else:
            # 如果当前节点不是通配符，则确保未用尽节点但用尽目标，因为此时一个节点只能匹配一个目标
            if node_ind >= len(nodes) - 1 and target_ind < len(targets) - 1:
                return None

            # 尝试将当前节点与目标进行匹配
            match_attempt = node.matches(target)

            if match_attempt:
                return [(node_ind + 1, target_ind + 1)], match_attempt
            elif node == target:
                return [(node_ind + 1, target_ind + 1)], None
            else:
                return None
    def _matches_match_wilds(dictionary, wildcard_ind, nodes, targets):
        """Determine matches of a wildcard with sub-expression in `targets`."""
        # 获取通配符
        wildcard = nodes[wildcard_ind]
        # 获取通配符在字典中的起始和结束位置
        begin, end = dictionary[wildcard_ind]
        # 从目标列表中获取与通配符匹配的子表达式
        terms = targets[begin:end + 1]
        # 如果子表达式数量大于1，则将它们组合成乘积
        mult = Mul(*terms) if len(terms) > 1 else terms[0]
        # 返回通配符是否与乘积匹配的结果
        return wildcard.matches(mult)

    @staticmethod
    def _matches_get_other_nodes(dictionary, nodes, node_ind):
        """Find other wildcards that may have already been matched."""
        # 获取指定节点处的通配符
        ind_node = nodes[node_ind]
        # 返回与该节点处通配符相同的所有其他节点的索引列表
        return [ind for ind in dictionary if nodes[ind] == ind_node]

    @staticmethod
    def _combine_inverse(lhs, rhs):
        """
        Returns lhs/rhs, but treats arguments like symbols, so things
        like oo/oo return 1 (instead of a nan) and ``I`` behaves like
        a symbol instead of sqrt(-1).
        """
        # 导入必要的库和模块
        from sympy.simplify.simplify import signsimp
        from .symbol import Dummy
        # 如果左右操作数相同，则直接返回1
        if lhs == rhs:
            return S.One

        def check(l, r):
            # 检查特定条件下左右操作数是否相等
            if l.is_Float and r.is_comparable:
                # 如果两个对象都可以加上0并且结果相等，则返回True
                return l.__add__(0) == r.evalf().__add__(0)
            return False
        # 若满足特定条件，则返回1
        if check(lhs, rhs) or check(rhs, lhs):
            return S.One
        # 如果左右操作数中有任何一个是指数或乘积，则进行处理
        if any(i.is_Pow or i.is_Mul for i in (lhs, rhs)):
            # 定义虚拟符号'I'
            d = Dummy('I')
            _i = {S.ImaginaryUnit: d}
            i_ = {d: S.ImaginaryUnit}
            # 将左右操作数中的'I'符号替换为虚拟符号
            a = lhs.xreplace(_i).as_powers_dict()
            b = rhs.xreplace(_i).as_powers_dict()
            blen = len(b)
            # 对于右操作数中的每个项，如果左操作数中也存在，则进行处理
            for bi in tuple(b.keys()):
                if bi in a:
                    a[bi] -= b.pop(bi)
                    if not a[bi]:
                        a.pop(bi)
            # 如果右操作数中的项数发生变化，则重新构建左右操作数
            if len(b) != blen:
                lhs = Mul(*[k**v for k, v in a.items()]).xreplace(i_)
                rhs = Mul(*[k**v for k, v in b.items()]).xreplace(i_)
        # 返回左操作数除以右操作数的结果，经过符号简化
        rv = lhs/rhs
        srv = signsimp(rv)
        return srv if srv.is_Number else rv

    def as_powers_dict(self):
        """Convert expression into a dictionary of base to exponent."""
        # 创建默认字典
        d = defaultdict(int)
        # 遍历表达式中的每个项，将其转换为基数到指数的字典
        for term in self.args:
            for b, e in term.as_powers_dict().items():
                d[b] += e
        return d

    def as_numer_denom(self):
        """Separate the expression into numerator and denominator."""
        # 不使用_from_args重新构建分子和分母，因为它们的顺序不能保证分开后仍然相同
        # 分别获取所有子表达式的分子和分母，并返回重新构建的结果
        numers, denoms = list(zip(*[f.as_numer_denom() for f in self.args]))
        return self.func(*numers), self.func(*denoms)
    # 返回自身表达式的基底和指数
    def as_base_exp(self):
        e1 = None  # 初始化指数为 None
        bases = []  # 初始化基底列表
        nc = 0  # 初始化非交换项计数

        # 遍历自身的所有参数
        for m in self.args:
            b, e = m.as_base_exp()  # 获取参数 m 的基底和指数
            if not b.is_commutative:  # 如果基底不可交换
                nc += 1  # 非交换项计数加一
            if e1 is None:  # 如果 e1 仍为 None
                e1 = e  # 设置 e1 为当前参数的指数
            elif e != e1 or nc > 1:  # 如果当前参数的指数与 e1 不同或者非交换项超过一个
                return self, S.One  # 返回自身和单位元素 S.One

            bases.append(b)  # 将参数的基底添加到基底列表中

        # 返回新的表达式，其基底为 bases 组成的元组，指数为 e1
        return self.func(*bases), e1

    # 判断自身是否为多项式
    def _eval_is_polynomial(self, syms):
        return all(term._eval_is_polynomial(syms) for term in self.args)

    # 判断自身是否为有理函数
    def _eval_is_rational_function(self, syms):
        return all(term._eval_is_rational_function(syms) for term in self.args)

    # 判断自身是否为亚析函数
    def _eval_is_meromorphic(self, x, a):
        return _fuzzy_group((arg.is_meromorphic(x, a) for arg in self.args),
                            quick_exit=True)

    # 判断自身是否为代数表达式
    def _eval_is_algebraic_expr(self, syms):
        return all(term._eval_is_algebraic_expr(syms) for term in self.args)

    # 判断自身是否可交换
    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)

    # 判断自身是否为复数
    def _eval_is_complex(self):
        comp = _fuzzy_group(a.is_complex for a in self.args)
        if comp is False:
            if any(a.is_infinite for a in self.args):
                if any(a.is_zero is not False for a in self.args):
                    return None
                return False
        return comp

    # 判断自身是否为零
    def _eval_is_zero(self):
        # 如果有任意一个参数为零且没有参数为无穷大，但需要谨慎处理三值逻辑
        seen_zero, seen_infinite = self._eval_is_zero_infinite_helper()

        if seen_zero is False:
            return False
        elif seen_zero is True and seen_infinite is False:
            return True
        else:
            return None

    # 判断自身是否为无穷大
    def _eval_is_infinite(self):
        # 如果有任意一个参数为无穷大且没有参数为零，但需要谨慎处理三值逻辑
        seen_zero, seen_infinite = self._eval_is_zero_infinite_helper()

        if seen_infinite is True and seen_zero is False:
            return True
        elif seen_infinite is False:
            return False
        else:
            return None

    # 不需要实现 _eval_is_finite，因为假设系统可以根据有限性推断出来

    # 判断自身是否为有理数
    def _eval_is_rational(self):
        r = _fuzzy_group((a.is_rational for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            # 所有参数中除了一个之外都是有理数
            if all(a.is_zero is False for a in self.args):
                return False

    # 判断自身是否为代数数
    def _eval_is_algebraic(self):
        r = _fuzzy_group((a.is_algebraic for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            # 所有参数中除了一个之外都是代数数
            if all(a.is_zero is False for a in self.args):
                return False

    # 没有涉及奇偶检查时，这段代码足够使用
    # 检查是否所有参数中至少有一个是极坐标形式的
    def _eval_is_polar(self):
        # 检查是否存在至少一个参数是极坐标形式的
        has_polar = any(arg.is_polar for arg in self.args)
        # 返回结果，要求所有参数要么是极坐标形式，要么是正实数形式
        return has_polar and \
            all(arg.is_polar or arg.is_positive for arg in self.args)

    # 评估对象是否是扩展实数
    def _eval_is_extended_real(self):
        # 调用内部方法，根据参数指示返回实部或虚部
        return self._eval_real_imag(True)

    # 评估对象的实部或虚部
    def _eval_real_imag(self, real):
        zero = False  # 初始化零标志
        t_not_re_im = None  # 初始化非实部或虚部标志

        for t in self.args:
            if (t.is_complex or t.is_infinite) is False and t.is_extended_real is False:
                return False  # 如果参数既不是复数也不是无穷大，并且不是扩展实数，则返回False
            elif t.is_imaginary:  # 如果参数是虚数
                real = not real  # 切换到虚部
            elif t.is_extended_real:  # 如果参数是实数
                if not zero:  # 如果零标志为假
                    z = t.is_zero  # 检查是否为零
                    if not z and zero is False:
                        zero = z  # 更新零标志
                    elif z:
                        if all(a.is_finite for a in self.args):
                            return True  # 如果所有参数都是有限的，则返回True
                        return  # 否则返回None
            elif t.is_extended_real is False:
                # 符号或类似于`2 + I`的文本或符号虚数
                if t_not_re_im:
                    return  # 复杂项可能会取消
                t_not_re_im = t  # 更新非实部或虚部标志
            elif t.is_imaginary is False:  # 符号如`2`或`2 + I`
                if t_not_re_im:
                    return  # 复杂项可能会取消
                t_not_re_im = t  # 更新非实部或虚部标志
            else:
                return

        if t_not_re_im:
            if t_not_re_im.is_extended_real is False:
                if real:  # 如3
                    return zero  # 3 * (类似于2 + I或I的东西)不是实数
            if t_not_re_im.is_imaginary is False:  # 符号2或2 + I
                if not real:  # 如I
                    return zero  # I * (类似于2或2 + I的东西)不是实数
        elif zero is False:
            return real  # 不能被0压制
        elif real:
            return real  # 无论零是什么

    # 评估对象是否是虚数
    def _eval_is_imaginary(self):
        # 如果所有参数都不是零且是有限的，则评估实部或虚部为False
        if all(a.is_zero is False and a.is_finite for a in self.args):
            return self._eval_real_imag(False)

    # 评估对象是否是厄米特矩阵
    def _eval_is_hermitian(self):
        # 根据参数指示评估厄米特或反厄米特
        return self._eval_herm_antiherm(True)

    # 评估对象是否是反厄米特矩阵
    def _eval_is_antihermitian(self):
        # 根据参数指示评估厄米特或反厄米特
        return self._eval_herm_antiherm(False)

    # 评估对象是否是厄米特或反厄米特矩阵
    def _eval_herm_antiherm(self, herm):
        for t in self.args:
            if t.is_hermitian is None or t.is_antihermitian is None:
                return  # 如果参数是厄米特或反厄米特，则返回
            if t.is_hermitian:
                continue  # 如果参数是厄米特，则继续下一个参数
            elif t.is_antihermitian:
                herm = not herm  # 如果参数是反厄米特，则切换herm标志
            else:
                return

        if herm is not False:
            return herm  # 如果herm不是False，则返回herm

        is_zero = self._eval_is_zero()
        if is_zero:
            return True  # 如果评估为零，则返回True
        elif is_zero is False:
            return herm  # 如果评估不为零，则返回herm
    # 判断表达式是否为无理数
    def _eval_is_irrational(self):
        # 遍历表达式中的每个项
        for t in self.args:
            # 获取当前项是否为无理数的布尔值
            a = t.is_irrational
            # 如果当前项是无理数
            if a:
                # 复制表达式的所有项到变量 others，并移除当前项 t
                others = list(self.args)
                others.remove(t)
                # 检查剩余项是否全部为有理数且非零
                if all((x.is_rational and fuzzy_not(x.is_zero)) is True for x in others):
                    return True  # 如果满足条件，返回 True
                return  # 如果不满足条件，返回 None
            # 如果当前项不是无理数
            if a is None:
                return  # 返回 None
        # 如果所有项都是实数，则返回 False
        if all(x.is_real for x in self.args):
            return False

    # 判断表达式是否为扩展正数
    def _eval_is_extended_positive(self):
        """Return True if self is positive, False if not, and None if it
        cannot be determined.

        Explanation
        ===========

        This algorithm is non-recursive and works by keeping track of the
        sign which changes when a negative or nonpositive is encountered.
        Whether a nonpositive or nonnegative is seen is also tracked since
        the presence of these makes it impossible to return True, but
        possible to return False if the end result is nonpositive. e.g.

            pos * neg * nonpositive -> pos or zero -> None is returned
            pos * neg * nonnegative -> neg or zero -> False is returned
        """
        return self._eval_pos_neg(1)

    # 辅助函数，用于判断表达式的正负性
    def _eval_pos_neg(self, sign):
        # 初始标志变量
        saw_NON = saw_NOT = False
        # 遍历表达式中的每个项
        for t in self.args:
            # 如果当前项是扩展正数，则继续下一个循环
            if t.is_extended_positive:
                continue
            # 如果当前项是扩展负数，改变标志变量 sign 的符号
            elif t.is_extended_negative:
                sign = -sign
            # 如果当前项是零
            elif t.is_zero:
                # 如果所有项都是有限数，则返回 False；否则返回 None
                if all(a.is_finite for a in self.args):
                    return False
                return
            # 如果当前项是扩展非正数，改变标志变量 sign 的符号，并设置 saw_NON 为 True
            elif t.is_extended_nonpositive:
                sign = -sign
                saw_NON = True
            # 如果当前项是扩展非负数，设置 saw_NON 为 True
            elif t.is_extended_nonnegative:
                saw_NON = True
            # 如果当前项的 is_positive 属性为 False
            elif t.is_positive is False:
                sign = -sign
                # 如果已经出现过 is_positive 属性为 False 的项，则返回 None
                if saw_NOT:
                    return
                saw_NOT = True
            # 如果当前项的 is_negative 属性为 False
            elif t.is_negative is False:
                # 如果已经出现过 is_negative 属性为 False 的项，则返回 None
                if saw_NOT:
                    return
                saw_NOT = True
            else:
                return
        # 如果最终标志变量 sign 为 1，且未出现扩展非正数或 is_negative 为 False 的情况，则返回 True
        if sign == 1 and saw_NON is False and saw_NOT is False:
            return True
        # 如果标志变量 sign 小于 0，则返回 False
        if sign < 0:
            return False

    # 判断表达式是否为扩展负数
    def _eval_is_extended_negative(self):
        return self._eval_pos_neg(-1)
    def _eval_is_odd(self):
        # 检查是否为整数
        is_integer = self._eval_is_integer()
        if is_integer is not True:
            return is_integer

        # 导入分数函数
        from sympy.simplify.radsimp import fraction
        # 将表达式转换为分数形式
        n, d = fraction(self)
        
        # 如果分母为整数且为偶数
        if d.is_Integer and d.is_even:
            # 计算分子和分母中最小的2的幂次数之差是否为正，用于判断是否为偶数
            if (Add(*[i.as_base_exp()[1] for i in
                    Mul.make_args(n) if i.is_even]) - trailing(d.p)
                    ).is_positive:
                return False
            return
        
        # 遍历表达式中的每个项
        r, acc = True, 1
        for t in self.args:
            # 跳过绝对值为1的项
            if abs(t) is S.One:
                continue
            # 如果当前项为偶数，则整个表达式为偶数
            if t.is_even:
                return False
            # 如果r为假，则跳过
            if r is False:
                pass
            # 如果acc不为1且与当前项相加为奇数，则r为假
            elif acc != 1 and (acc + t).is_odd:
                r = False
            # 如果当前项不能判断为偶数，则r为未知
            elif t.is_even is None:
                r = None
            acc = t
        return r

    def _eval_is_even(self):
        # 导入分数函数
        from sympy.simplify.radsimp import fraction
        # 将表达式转换为分数形式
        n, d = fraction(self)
        
        # 如果分子为整数且为偶数
        if n.is_Integer and n.is_even:
            # 计算分母和分子中最小的2的幂次数之差是否为非负，用于判断是否为整数且为偶数
            if (Add(*[i.as_base_exp()[1] for i in
                    Mul.make_args(d) if i.is_even]) - trailing(n.p)
                    ).is_nonnegative:
                return False

    def _eval_is_composite(self):
        """
        Here we count the number of arguments that have a minimum value
        greater than two.
        If there are more than one of such a symbol then the result is composite.
        Else, the result cannot be determined.
        """
        number_of_args = 0 # count of symbols with minimum value greater than one
        # 遍历表达式中的每个项
        for arg in self.args:
            # 如果不是正整数，返回未知
            if not (arg.is_integer and arg.is_positive):
                return None
            # 如果arg-1为正，计数器加一
            if (arg-1).is_positive:
                number_of_args += 1

        # 如果计数器大于1，返回复合数（即有多个大于1的最小值）
        if number_of_args > 1:
            return True

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 返回表达式中每个项的主导项
        return self.func(*[t.as_leading_term(x, logx=logx, cdir=cdir) for t in self.args])

    def _eval_conjugate(self):
        # 返回表达式的共轭
        return self.func(*[t.conjugate() for t in self.args])

    def _eval_transpose(self):
        # 返回表达式的转置
        return self.func(*[t.transpose() for t in self.args[::-1]])

    def _eval_adjoint(self):
        # 返回表达式的伴随
        return self.func(*[t.adjoint() for t in self.args[::-1]])
    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import sqrt
        >>> (-3*sqrt(2)*(2 - 2*sqrt(2))).as_content_primitive()
        (6, -sqrt(2)*(1 - sqrt(2)))

        See docstring of Expr.as_content_primitive for more examples.
        """

        # 初始化系数为1
        coef = S.One
        # 初始化空列表存放处理后的表达式部分
        args = []
        # 遍历当前表达式的每个子表达式
        for a in self.args:
            # 获取子表达式的内容主因子和非内容主因子
            c, p = a.as_content_primitive(radical=radical, clear=clear)
            # 累乘内容主因子到总系数
            coef *= c
            # 如果非内容主因子不是1，则加入到args列表中
            if p is not S.One:
                args.append(p)
        # 不使用 self._from_args 重建 args，因为可能会有相同的 args 需要合并
        # 例如 (2+2*x)*(3+3*x) 应该是 (6, (1 + x)**2) 而不是 (6, (1+x)*(1+x))
        # 返回最终的结果，包括总系数和重新构建的表达式
        return coef, self.func(*args)

    def as_ordered_factors(self, order=None):
        """Transform an expression into an ordered list of factors.

        Examples
        ========

        >>> from sympy import sin, cos
        >>> from sympy.abc import x, y

        >>> (2*x*y*sin(x)*cos(x)).as_ordered_factors()
        [2, x, y, sin(x), cos(x)]

        """

        # 将表达式分解为内容部分和非内容部分
        cpart, ncpart = self.args_cnc()
        # 根据指定的排序顺序对内容部分进行排序
        cpart.sort(key=lambda expr: expr.sort_key(order=order))
        # 返回排序后的内容部分加上非内容部分的列表
        return cpart + ncpart

    @property
    def _sorted_args(self):
        # 将表达式转换为有序因子列表
        return tuple(self.as_ordered_factors())
# 创建一个名为 mul 的 AssocOpDispatcher 对象
mul = AssocOpDispatcher('mul')


# 定义一个函数 prod，计算给定列表 a 中元素的乘积，可以指定起始值，默认为 1
def prod(a, start=1):
    """Return product of elements of a. Start with int 1 so if only
       ints are included then an int result is returned.

    Examples
    ========

    >>> from sympy import prod, S
    >>> prod(range(3))
    0
    >>> type(_) is int
    True
    >>> prod([S(2), 3])
    6
    >>> _.is_Integer
    True

    You can start the product at something other than 1:

    >>> prod([1, 2], 3)
    6

    """
    # 使用 reduce 函数计算列表 a 中所有元素的乘积，起始值为 start
    return reduce(operator.mul, a, start)


# 定义一个内部函数 _keep_coeff，根据条件返回未评估的表达式 coeff*factors
def _keep_coeff(coeff, factors, clear=True, sign=False):
    """Return ``coeff*factors`` unevaluated if necessary.

    If ``clear`` is False, do not keep the coefficient as a factor
    if it can be distributed on a single factor such that one or
    more terms will still have integer coefficients.

    If ``sign`` is True, allow a coefficient of -1 to remain factored out.

    Examples
    ========

    >>> from sympy.core.mul import _keep_coeff
    >>> from sympy.abc import x, y
    >>> from sympy import S

    >>> _keep_coeff(S.Half, x + 2)
    (x + 2)/2
    >>> _keep_coeff(S.Half, x + 2, clear=False)
    x/2 + 1
    >>> _keep_coeff(S.Half, (x + 2)*y, clear=False)
    y*(x + 2)/2
    >>> _keep_coeff(S(-1), x + y)
    -x - y
    >>> _keep_coeff(S(-1), x + y, sign=True)
    -(x + y)
    """
    # 如果 coeff 不是数值类型，则返回 coeff*factors
    if not coeff.is_Number:
        if factors.is_Number:
            factors, coeff = coeff, factors
        else:
            return coeff*factors
    # 如果 factors 是 S.One，则返回 coeff
    if factors is S.One:
        return coeff
    # 如果 coeff 是 S.One，则返回 factors
    if coeff is S.One:
        return factors
    # 如果 coeff 是 S.NegativeOne 并且 sign 为 False，则返回 -factors
    elif coeff is S.NegativeOne and not sign:
        return -factors
    # 如果 factors 是 Add 类型
    elif factors.is_Add:
        # 如果 clear 为 False，coeff 是 Rational 类型，并且 coeff 的分母不为 1
        if not clear and coeff.is_Rational and coeff.q != 1:
            # 将 factors 中的每个元素都转换为 (coeff, m) 形式的元组
            args = [i.as_coeff_Mul() for i in factors.args]
            args = [(_keep_coeff(c, coeff), m) for c, m in args]
            # 如果 args 中存在整数系数的元素，则返回重新组合后的 Add 对象
            if any(c.is_Integer for c, _ in args):
                return Add._from_args([Mul._from_args(
                    i[1:] if i[0] == 1 else i) for i in args])
        # 否则返回 Mul(coeff, factors, evaluate=False)
        return Mul(coeff, factors, evaluate=False)
    # 如果 factors 是 Mul 类型
    elif factors.is_Mul:
        margs = list(factors.args)
        # 如果 margs[0] 是数值类型，则将 coeff 乘以 margs[0]
        if margs[0].is_Number:
            margs[0] *= coeff
            # 如果乘积结果为 1，则移除 margs[0]
            if margs[0] == 1:
                margs.pop(0)
        else:
            # 否则将 coeff 插入到 margs 的开头
            margs.insert(0, coeff)
        # 返回重新组合后的 Mul 对象
        return Mul._from_args(margs)
    else:
        # 计算 coeff*factors，并且如果结果是数值且 factors 不是数值，则返回 Mul(coeff, factors)
        m = coeff*factors
        if m.is_Number and not factors.is_Number:
            m = Mul._from_args((coeff, factors))
        return m


# 定义一个函数 expand_2arg，接受一个参数 e，并返回一个函数 bottom_up(e, do)
def expand_2arg(e):
    def do(e):
        # 如果 e 是 Mul 类型
        if e.is_Mul:
            # 将 e 分解为系数 c 和剩余部分 r
            c, r = e.as_coeff_Mul()
            # 如果 c 是数值类型，并且 r 是 Add 类型
            if c.is_Number and r.is_Add:
                # 返回未评估的 Add 对象，其中每个 ri 都乘以 c
                return _unevaluated_Add(*[c*ri for ri in r.args])
        # 否则直接返回 e
        return e
    # 返回函数 do
    return bottom_up(e, do)


# 导入所需的模块和类
from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add
```