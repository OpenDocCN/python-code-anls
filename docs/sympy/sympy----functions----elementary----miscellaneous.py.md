# `D:\src\scipysrc\sympy\sympy\functions\elementary\miscellaneous.py`

```
from sympy.core import Function, S, sympify, NumberKind  # 导入 Function, S, sympify 和 NumberKind 模块
from sympy.utilities.iterables import sift  # 导入 sift 函数
from sympy.core.add import Add  # 导入 Add 类
from sympy.core.containers import Tuple  # 导入 Tuple 类
from sympy.core.operations import LatticeOp, ShortCircuit  # 导入 LatticeOp 和 ShortCircuit 类
from sympy.core.function import (Application, Lambda,  # 导入 Application, Lambda 和 ArgumentIndexError 类
    ArgumentIndexError)
from sympy.core.expr import Expr  # 导入 Expr 类
from sympy.core.exprtools import factor_terms  # 导入 factor_terms 函数
from sympy.core.mod import Mod  # 导入 Mod 类
from sympy.core.mul import Mul  # 导入 Mul 类
from sympy.core.numbers import Rational  # 导入 Rational 类
from sympy.core.power import Pow  # 导入 Pow 类
from sympy.core.relational import Eq, Relational  # 导入 Eq 和 Relational 类
from sympy.core.singleton import Singleton  # 导入 Singleton 类
from sympy.core.sorting import ordered  # 导入 ordered 函数
from sympy.core.symbol import Dummy  # 导入 Dummy 类
from sympy.core.rules import Transform  # 导入 Transform 类
from sympy.core.logic import fuzzy_and, fuzzy_or, _torf  # 导入 fuzzy_and, fuzzy_or 和 _torf 函数
from sympy.core.traversal import walk  # 导入 walk 函数
from sympy.core.numbers import Integer  # 导入 Integer 类
from sympy.logic.boolalg import And, Or  # 导入 And 和 Or 类


def _minmax_as_Piecewise(op, *args):
    # 将 Min/Max 重写为 Piecewise 的辅助函数
    from sympy.functions.elementary.piecewise import Piecewise  # 导入 Piecewise 类
    ec = []
    for i, a in enumerate(args):
        c = [Relational(a, args[j], op) for j in range(i + 1, len(args))]
        ec.append((a, And(*c)))
    return Piecewise(*ec)  # 返回 Piecewise 对象


class IdentityFunction(Lambda, metaclass=Singleton):
    """
    The identity function

    Examples
    ========

    >>> from sympy import Id, Symbol
    >>> x = Symbol('x')
    >>> Id(x)
    x

    """

    _symbol = Dummy('x')  # 创建一个 Dummy 符号对象 '_symbol'

    @property
    def signature(self):
        return Tuple(self._symbol)  # 返回一个 Tuple，包含 '_symbol' 这个符号对象

    @property
    def expr(self):
        return self._symbol  # 返回 '_symbol' 这个符号对象本身


Id = S.IdentityFunction  # 将 S.IdentityFunction 赋值给 Id 变量


###############################################################################
############################# ROOT and SQUARE ROOT FUNCTION ###################
###############################################################################


def sqrt(arg, evaluate=None):
    """Returns the principal square root.

    Parameters
    ==========

    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.

    Examples
    ========

    >>> from sympy import sqrt, Symbol, S
    >>> x = Symbol('x')

    >>> sqrt(x)
    sqrt(x)

    >>> sqrt(x)**2
    x

    Note that sqrt(x**2) does not simplify to x.

    >>> sqrt(x**2)
    sqrt(x**2)

    This is because the two are not equal to each other in general.
    For example, consider x == -1:

    >>> from sympy import Eq
    >>> Eq(sqrt(x**2), x).subs(x, -1)
    False

    This is because sqrt computes the principal square root, so the square may
    put the argument in a different branch.  This identity does hold if x is
    positive:

    >>> y = Symbol('y', positive=True)
    >>> sqrt(y**2)
    y

    You can force this simplification by using the powdenest() function with
    the force option set to True:

    >>> from sympy import powdenest
    >>> sqrt(x**2)
    sqrt(x**2)

    """
    # 计算平方根的嵌套，使用 powdenest 函数，强制进行处理
    >>> powdenest(sqrt(x**2), force=True)
    # 返回 x，表示对 x**2 开根号后得到 x

    # 要获取平方根的两个分支，可以使用 rootof 函数：
    >>> from sympy import rootof

    # 对 x**2-3 使用 rootof 函数，分别获取其两个根
    >>> [rootof(x**2-3,i) for i in (0,1)]
    # 结果为 [-sqrt(3), sqrt(3)]

    # 虽然 ``sqrt`` 被打印出来，但实际上并没有 ``sqrt`` 函数，因此在表达式中查找 ``sqrt`` 会失败：
    >>> from sympy.utilities.misc import func_name
    >>> func_name(sqrt(x))
    # 返回 'Pow'

    # 对于找到 ``sqrt``，需要查找指数为 ``1/2`` 的 ``Pow`` 对象：
    >>> sqrt(x).has(sqrt)
    # 返回 False

    # 通过查找具有指数为 ``1/2`` 的 ``Pow`` 对象，可以找到 ``sqrt``：
    >>> (x + 1/sqrt(x)).find(lambda i: i.is_Pow and abs(i.exp) is S.Half)
    # 返回 {1/sqrt(x)}

    # 参见
    # =====
    # sympy.polys.rootoftools.rootof, root, real_root

    # 参考文献
    # ==========
    # .. [1] https://en.wikipedia.org/wiki/Square_root
    # .. [2] https://en.wikipedia.org/wiki/Principal_value
    """
    # 将参数 sympify(arg) 处理为 Pow 对象
    return Pow(arg, S.Half, evaluate=evaluate)
def root(arg, n, k=0, evaluate=None):
    r"""Returns the *k*-th *n*-th root of ``arg``.

    Parameters
    ==========

    arg : number or symbolic expression
        The base number or expression whose root is to be computed.

    n : int or Rational
        The degree of the root, which must be positive.

    k : int, optional
        Specifies which of the n roots to return. It should be an integer
        in the range {0, 1, ..., n-1}. Default is 0, which returns the
        principal root.

    evaluate : bool, optional
        Determines if the expression should be evaluated. If None, the value
        is taken from global_parameters.evaluate.

    Examples
    ========

    >>> from sympy import root, Rational
    >>> from sympy.abc import x, n

    >>> root(x, 2)
    sqrt(x)
        Returns the principal square root of x.

    >>> root(x, 3)
    x**(1/3)
        Returns the principal cube root of x.

    >>> root(x, n)
    x**(1/n)
        Returns the principal nth root of x.

    >>> root(x, -Rational(2, 3))
    x**(-3/2)
        Returns the principal -(3/2)th root of x.

    To obtain a specific k-th nth root, specify k:

    >>> root(-2, 3, 2)
    -(-1)**(2/3)*2**(1/3)
        Returns the 2nd cube root of -2.

    To obtain all n nth roots, use the rootof function. The following examples
    show the roots of unity for n equal to 2, 3, and 4:

    >>> from sympy import rootof

    >>> [rootof(x**2 - 1, i) for i in range(2)]
    [-1, 1]
        Returns the two roots of unity.

    >>> [rootof(x**3 - 1, i) for i in range(3)]
    [1, -1/2 - sqrt(3)*I/2, -1/2 + sqrt(3)*I/2]
        Returns the three roots of unity.

    >>> [rootof(x**4 - 1, i) for i in range(4)]
    [-1, 1, -I, I]
        Returns the four roots of unity.

    SymPy, like other symbolic algebra systems, returns the complex root of
    negative numbers. This is the principal root and differs from the
    textbook result that one might be expecting. For example, the cube root
    of -8 does not come back as -2:

    >>> root(-8, 3)
    2*(-1)**(1/3)
        Returns the principal cube root of -8.

    The real_root function can be used to either make the principal result
    real (or simply to return the real root directly):

    >>> from sympy import real_root
    >>> real_root(_)
    -2
        Returns the real principal cube root of -8.

    >>> real_root(-32, 5)
    -2
        Returns the real principal 5th root of -32.

    Alternatively, the n//2-th n-th root of a negative number can be computed
    with root:

    >>> root(-32, 5, 5//2)
        Returns the principal 5th root of -32.

    See Also
    ========

    sympy.polys.rootoftools.rootof, root, real_root

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Nth_root
    .. [2] https://en.wikipedia.org/wiki/Principal_value

    """
    return Pow(arg, Rational(1, 3), evaluate=evaluate)
    # 使用 sympify 函数将 n 转换为 SymPy 表达式
    n = sympify(n)
    # 如果 k 不为零，返回表达式 Mul(Pow(arg, S.One/n, evaluate=evaluate), S.NegativeOne**(2*k/n), evaluate=evaluate)
    if k:
        return Mul(Pow(arg, S.One/n, evaluate=evaluate), S.NegativeOne**(2*k/n), evaluate=evaluate)
    # 如果 k 为零，返回表达式 Pow(arg, 1/n, evaluate=evaluate)
    return Pow(arg, 1/n, evaluate=evaluate)
def real_root(arg, n=None, evaluate=None):
    r"""Return the real *n*'th-root of *arg* if possible.
    
    Parameters
    ==========

    n : int or None, optional
        If *n* is ``None``, then all instances of
        $(-n)^{1/\text{odd}}$ will be changed to $-n^{1/\text{odd}}$.
        This will only create a real root of a principal root.
        The presence of other factors may cause the result to not be
        real.
        
    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.
        
    Examples
    ========

    >>> from sympy import root, real_root

    >>> real_root(-8, 3)
    -2
    >>> root(-8, 3)
    2*(-1)**(1/3)
    >>> real_root(_)
    -2

    If one creates a non-principal root and applies real_root, the
    result will not be real (so use with caution):

    >>> root(-8, 3, 2)
    -2*(-1)**(2/3)
    >>> real_root(_)
    -2*(-1)**(2/3)

    See Also
    ========

    sympy.polys.rootoftools.rootof
    sympy.core.intfunc.integer_nthroot
    root, sqrt
    """
    # 导入必要的函数和类
    from sympy.functions.elementary.complexes import Abs, im, sign
    from sympy.functions.elementary.piecewise import Piecewise
    
    # 如果参数 n 不为 None，则根据不同情况返回不同的 Piecewise 表达式
    if n is not None:
        return Piecewise(
            # 当 n 等于 1 或者 -1 时，返回原始的根表达式
            (root(arg, n, evaluate=evaluate), Or(Eq(n, S.One), Eq(n, S.NegativeOne))),
            # 当 arg 为负数且 n 为奇数时，返回带有符号的根表达式
            (Mul(sign(arg), root(Abs(arg), n, evaluate=evaluate), evaluate=evaluate),
             And(Eq(im(arg), S.Zero), Eq(Mod(n, 2), S.One))),
            # 其他情况下，返回原始的根表达式
            (root(arg, n, evaluate=evaluate), True))
    
    # 对参数进行符号化处理
    rv = sympify(arg)
    
    # 定义一个转换函数，将 -(-x.base)**x.exp 转换为 x
    n1pow = Transform(lambda x: -(-x.base)**x.exp,
                      lambda x:
                      x.is_Pow and
                      x.base.is_negative and
                      x.exp.is_Rational and
                      x.exp.p == 1 and x.exp.q % 2)
    
    # 使用定义好的转换函数对 rv 进行替换操作
    return rv.xreplace(n1pow)
    def __new__(cls, *args, **assumptions):
        from sympy.core.parameters import global_parameters
        evaluate = assumptions.pop('evaluate', global_parameters.evaluate)
        args = (sympify(arg) for arg in args)

        # first standard filter, for cls.zero and cls.identity
        # also reshape Max(a, Max(b, c)) to Max(a, b, c)
        
        # 如果需要进行求值，则进行以下操作
        if evaluate:
            try:
                # 对参数进行第一次标准过滤，处理 cls.zero 和 cls.identity
                args = frozenset(cls._new_args_filter(args))
            except ShortCircuit:
                # 如果捕获到 ShortCircuit 异常，则返回 cls.zero
                return cls.zero
            # 移除可识别的冗余参数
            args = cls._collapse_arguments(args, **assumptions)
            # 查找本地的零值
            args = cls._find_localzeros(args, **assumptions)
        # 将参数集合冻结
        args = frozenset(args)

        # 如果参数为空集合，则返回 cls.identity
        if not args:
            return cls.identity

        # 如果参数集合长度为1，则返回集合中的唯一元素
        if len(args) == 1:
            return list(args).pop()

        # 创建基本对象
        obj = Expr.__new__(cls, *ordered(args), **assumptions)
        # 设置对象的参数集合属性
        obj._argset = args
        return obj

    @classmethod
    @classmethod
    def _new_args_filter(cls, arg_sequence):
        """
        Generator filtering args.

        first standard filter, for cls.zero and cls.identity.
        Also reshape ``Max(a, Max(b, c))`` to ``Max(a, b, c)``,
        and check arguments for comparability
        """
        # 遍历参数序列进行过滤
        for arg in arg_sequence:
            # 预先过滤，检查参数的可比性
            if not isinstance(arg, Expr) or arg.is_extended_real is False or (
                    arg.is_number and
                    not arg.is_comparable):
                raise ValueError("The argument '%s' is not comparable." % arg)

            # 如果参数等于 cls.zero，则引发 ShortCircuit 异常
            if arg == cls.zero:
                raise ShortCircuit(arg)
            # 如果参数等于 cls.identity，则继续下一个参数
            elif arg == cls.identity:
                continue
            # 如果参数的函数类型与 cls 相同，则展开参数
            elif arg.func == cls:
                yield from arg.args
            else:
                yield arg

    @classmethod
    def _find_localzeros(cls, values, **options):
        """
        Sequentially allocate values to localzeros.

        When a value is identified as being more extreme than another member it
        replaces that member; if this is never true, then the value is simply
        appended to the localzeros.
        """
        # 初始化本地零值集合
        localzeros = set()
        # 遍历传入的值集合
        for v in values:
            is_newzero = True
            localzeros_ = list(localzeros)
            # 遍历当前本地零值集合
            for z in localzeros_:
                # 如果当前值和已有零值相同，则不是新的零值
                if id(v) == id(z):
                    is_newzero = False
                else:
                    # 检查当前值是否与已有零值连接
                    con = cls._is_connected(v, z)
                    if con:
                        is_newzero = False
                        # 如果连接为真或者与 cls 相同，则替换已有的零值
                        if con is True or con == cls:
                            localzeros.remove(z)
                            localzeros.update([v])
            # 如果是新的零值，则添加到本地零值集合中
            if is_newzero:
                localzeros.update([v])
        return localzeros

    @classmethod
    def _is_connected(cls, x, y):
        """
        Check if x and y are connected somehow.
        """
        # 迭代两次来检查连接性
        for i in range(2):
            # 如果 x 和 y 相等，则它们连接
            if x == y:
                return True
            # 定义 t 和 f 分别为 Max 和 Min 函数
            t, f = Max, Min
            # 遍历操作符 ">" 和 "<"
            for op in "><":
                # 迭代两次来尝试比较 x 和 y
                for j in range(2):
                    try:
                        # 根据操作符比较 x 和 y
                        if op == ">":
                            v = x >= y
                        else:
                            v = x <= y
                    except TypeError:
                        return False  # 如果出现非实数的参数，则返回 False
                    # 如果比较结果不是关系表达式，则返回 t 或 f
                    if not v.is_Relational:
                        return t if v else f
                    # 交换 t 和 f 的值
                    t, f = f, t
                    # 交换 x 和 y 的值
                    x, y = y, x
                # 再次交换 x 和 y 的值，以便下一次迭代时相对于起始位置反向运行
                x, y = y, x
            # 简化可能是昂贵的，因此在尝试中要保守
            x = factor_terms(x - y)
            y = S.Zero

        # 如果两次迭代后仍未连接，则返回 False
        return False

    def _eval_derivative(self, s):
        # 计算函数对 s 的导数
        i = 0
        l = []
        # 遍历函数的参数
        for a in self.args:
            i += 1
            # 计算参数 a 对 s 的导数
            da = a.diff(s)
            # 如果导数为零，则继续下一个参数
            if da.is_zero:
                continue
            try:
                # 获取函数的 i 阶导数
                df = self.fdiff(i)
            except ArgumentIndexError:
                # 如果获取失败，则调用 Function 类的 fdiff 方法获取
                df = Function.fdiff(self, i)
            # 将导数和 i 阶导数的乘积加入列表
            l.append(df * da)
        # 返回所有乘积的和
        return Add(*l)

    def _eval_rewrite_as_Abs(self, *args, **kwargs):
        # 将函数重写为绝对值的形式
        from sympy.functions.elementary.complexes import Abs
        # 计算 s 和 d
        s = (args[0] + self.func(*args[1:]))/2
        d = abs(args[0] - self.func(*args[1:]))/2
        # 根据 self 的类型选择返回 s + d 或者 s - d 的重写绝对值形式
        return (s + d if isinstance(self, Max) else s - d).rewrite(Abs)

    def evalf(self, n=15, **options):
        # 对函数进行数值估计，返回结果精度为 n
        return self.func(*[a.evalf(n, **options) for a in self.args])

    def n(self, *args, **kwargs):
        # 对函数进行数值估计，接受与 evalf 相同的参数
        return self.evalf(*args, **kwargs)

    _eval_is_algebraic = lambda s: _torf(i.is_algebraic for i in s.args)
    _eval_is_antihermitian = lambda s: _torf(i.is_antihermitian for i in s.args)
    _eval_is_commutative = lambda s: _torf(i.is_commutative for i in s.args)
    _eval_is_complex = lambda s: _torf(i.is_complex for i in s.args)
    _eval_is_composite = lambda s: _torf(i.is_composite for i in s.args)
    _eval_is_even = lambda s: _torf(i.is_even for i in s.args)
    _eval_is_finite = lambda s: _torf(i.is_finite for i in s.args)
    _eval_is_hermitian = lambda s: _torf(i.is_hermitian for i in s.args)
    _eval_is_imaginary = lambda s: _torf(i.is_imaginary for i in s.args)
    _eval_is_infinite = lambda s: _torf(i.is_infinite for i in s.args)
    _eval_is_integer = lambda s: _torf(i.is_integer for i in s.args)
    _eval_is_irrational = lambda s: _torf(i.is_irrational for i in s.args)
    _eval_is_negative = lambda s: _torf(i.is_negative for i in s.args)
    _eval_is_noninteger = lambda s: _torf(i.is_noninteger for i in s.args)
    _eval_is_nonnegative = lambda s: _torf(i.is_nonnegative for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素的某种特性，并返回相应的布尔值结果
    _eval_is_nonpositive = lambda s: _torf(i.is_nonpositive for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素是否非零，并返回相应的布尔值结果
    _eval_is_nonzero = lambda s: _torf(i.is_nonzero for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素是否为奇数，并返回相应的布尔值结果
    _eval_is_odd = lambda s: _torf(i.is_odd for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素是否极坐标形式，并返回相应的布尔值结果
    _eval_is_polar = lambda s: _torf(i.is_polar for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素是否为正数，并返回相应的布尔值结果
    _eval_is_positive = lambda s: _torf(i.is_positive for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素是否为质数，并返回相应的布尔值结果
    _eval_is_prime = lambda s: _torf(i.is_prime for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素是否为有理数，并返回相应的布尔值结果
    _eval_is_rational = lambda s: _torf(i.is_rational for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素是否为实数，并返回相应的布尔值结果
    _eval_is_real = lambda s: _torf(i.is_real for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素是否为扩展实数，并返回相应的布尔值结果
    _eval_is_extended_real = lambda s: _torf(i.is_extended_real for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素是否为超越数，并返回相应的布尔值结果
    _eval_is_transcendental = lambda s: _torf(i.is_transcendental for i in s.args)
    # 定义 lambda 函数，用于检查对象集合中所有元素是否为零，并返回相应的布尔值结果
    _eval_is_zero = lambda s: _torf(i.is_zero for i in s.args)
# 定义一个名为 Max 的类，继承自 MinMaxBase 和 Application
class Max(MinMaxBase, Application):
    r"""
    如果可能的话，返回列表的最大值。

    当参数数量为一个时，返回该参数。

    当参数数量为两个时，返回 (a, b) 中较大的值。

    一般情况下，当列表长度大于2时，任务更为复杂。
    如果可能确定方向关系，则返回大于其他值的参数。

    如果无法确定这样的关系，则返回部分评估的结果。

    假设也被用来做决策。

    仅允许比较可比较的参数。

    它被命名为 ``Max`` 而不是 ``max`` 是为了避免与内置函数 ``max`` 冲突。

    Examples
    ========

    >>> from sympy import Max, Symbol, oo
    >>> from sympy.abc import x, y, z
    >>> p = Symbol('p', positive=True)
    >>> n = Symbol('n', negative=True)

    >>> Max(x, -2)
    Max(-2, x)
    >>> Max(x, -2).subs(x, 3)
    3
    >>> Max(p, -2)
    p
    >>> Max(x, y)
    Max(x, y)
    >>> Max(x, y) == Max(y, x)
    True
    >>> Max(x, Max(y, z))
    Max(x, y, z)
    >>> Max(n, 8, p, 7, -oo)
    Max(8, p)
    >>> Max (1, x, oo)
    oo

    * Algorithm

    任务可以被看作是在有向完全偏序集合 [1]_ 中搜索上确界。

    源数值按独立子集顺序分配，其中搜索上确界，并作为 Max 参数返回。

    如果结果上确界是唯一的，则返回它。

    独立子集是当前集合中仅相互可比较的值的集合。例如，自然数彼此可比较，但不能与符号 `x` 可比较。
    另一个例子是：带有负假设的符号 `x` 可以与自然数比较。

    还有一些“最小”元素，它们与所有其他元素可比较，并具有零属性（所有元素的最大或最小值）。
    例如，在无穷大的情况下，分配操作终止，并且仅返回该值。

    假设：
       - 如果 $A > B > C$，则 $A > C$
       - 如果 $A = B$，则可以删除 $B$

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Directed_complete_partial_order
    .. [2] https://en.wikipedia.org/wiki/Lattice_%28order%29

    See Also
    ========

    Min : 查找最小值
    """
    # 定义零元素为正无穷
    zero = S.Infinity
    # 定义单位元素为负无穷
    identity = S.NegativeInfinity
    # 计算函数差分（fdiff），接受参数 argindex 作为函数的一个索引
    def fdiff( self, argindex ):
        # 导入 Heaviside 函数
        from sympy.functions.special.delta_functions import Heaviside
        # 获取参数列表长度
        n = len(self.args)
        # 检查参数索引是否有效
        if 0 < argindex and argindex <= n:
            # 将索引转换为从零开始
            argindex -= 1
            # 如果参数个数为 2，则返回两个参数之差的 Heaviside 函数
            if n == 2:
                return Heaviside(self.args[argindex] - self.args[1 - argindex])
            # 否则，创建一个新的参数列表，排除掉 argindex 对应的参数
            newargs = tuple([self.args[i] for i in range(n) if i != argindex])
            # 返回 argindex 对应参数与其余参数的最大值的 Heaviside 函数
            return Heaviside(self.args[argindex] - Max(*newargs))
        else:
            # 如果参数索引无效，则抛出 ArgumentIndexError 异常
            raise ArgumentIndexError(self, argindex)

    # 将函数重写为 Heaviside 函数的形式
    def _eval_rewrite_as_Heaviside(self, *args, **kwargs):
        # 导入 Heaviside 函数
        from sympy.functions.special.delta_functions import Heaviside
        # 返回表达式中每个参数 j 对应的 Heaviside 函数乘积的加和
        return Add(*[j*Mul(*[Heaviside(j - i) for i in args if i!=j]) \
                for j in args])

    # 将函数重写为 Piecewise 函数的形式
    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        # 使用 _minmax_as_Piecewise 将表达式重写为 Piecewise 函数
        return _minmax_as_Piecewise('>=', *args)

    # 判断函数表达式是否为正数
    def _eval_is_positive(self):
        # 返回所有参数中是否存在某个参数为正数的模糊逻辑或
        return fuzzy_or(a.is_positive for a in self.args)

    # 判断函数表达式是否为非负数
    def _eval_is_nonnegative(self):
        # 返回所有参数中是否存在某个参数为非负数的模糊逻辑或
        return fuzzy_or(a.is_nonnegative for a in self.args)

    # 判断函数表达式是否为负数
    def _eval_is_negative(self):
        # 返回所有参数中是否存在某个参数为负数的模糊逻辑与
        return fuzzy_and(a.is_negative for a in self.args)
class Min(MinMaxBase, Application):
    """
    Return, if possible, the minimum value of the list.
    It is named ``Min`` and not ``min`` to avoid conflicts
    with the built-in function ``min``.

    Examples
    ========

    >>> from sympy import Min, Symbol, oo
    >>> from sympy.abc import x, y
    >>> p = Symbol('p', positive=True)
    >>> n = Symbol('n', negative=True)

    >>> Min(x, -2)
    Min(-2, x)
    >>> Min(x, -2).subs(x, 3)
    -2
    >>> Min(p, -3)
    -3
    >>> Min(x, y)
    Min(x, y)
    >>> Min(n, 8, p, -7, p, oo)
    Min(-7, n)

    See Also
    ========

    Max : find maximum values
    """
    zero = S.NegativeInfinity  # Class attribute representing negative infinity
    identity = S.Infinity  # Class attribute representing infinity

    def fdiff(self, argindex):
        """
        Method to compute the derivative with respect to a specific argument index.

        Parameters
        ----------
        argindex : int
            Index of the argument to differentiate with respect to.

        Returns
        -------
        Expr
            The derivative expression.

        Raises
        ------
        ArgumentIndexError
            If the provided `argindex` is out of range for the number of arguments.
        """
        from sympy.functions.special.delta_functions import Heaviside
        n = len(self.args)
        if 0 < argindex and argindex <= n:
            argindex -= 1
            if n == 2:
                return Heaviside(self.args[1-argindex] - self.args[argindex])
            newargs = tuple([self.args[i] for i in range(n) if i != argindex])
            return Heaviside(Min(*newargs) - self.args[argindex])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Heaviside(self, *args, **kwargs):
        """
        Rewrite method using Heaviside functions.

        Returns
        -------
        Expr
            The rewritten expression using Heaviside functions.
        """
        from sympy.functions.special.delta_functions import Heaviside
        return Add(*[j * Mul(*[Heaviside(i - j) for i in args if i != j]) \
                for j in args])

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        """
        Rewrite method using Piecewise functions.

        Returns
        -------
        Expr
            The rewritten expression using Piecewise functions.
        """
        return _minmax_as_Piecewise('<=', *args)

    def _eval_is_positive(self):
        """
        Evaluation method to determine if the expression is positive.

        Returns
        -------
        bool
            True if all arguments are positive, False otherwise.
        """
        return fuzzy_and(a.is_positive for a in self.args)

    def _eval_is_nonnegative(self):
        """
        Evaluation method to determine if the expression is nonnegative.

        Returns
        -------
        bool
            True if all arguments are nonnegative, False otherwise.
        """
        return fuzzy_and(a.is_nonnegative for a in self.args)

    def _eval_is_negative(self):
        """
        Evaluation method to determine if the expression is negative.

        Returns
        -------
        bool
            True if any argument is negative, False otherwise.
        """
        return fuzzy_or(a.is_negative for a in self.args)


class Rem(Function):
    """
    Returns the remainder when ``p`` is divided by ``q`` where ``p`` is finite
    and ``q`` is not equal to zero. The result, ``p - int(p/q)*q``, has the same sign
    as the divisor.

    Parameters
    ==========

    p : Expr
        Dividend.

    q : Expr
        Divisor.

    Notes
    =====

    ``Rem`` corresponds to the ``%`` operator in C.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Rem
    >>> Rem(x**3, y)
    Rem(x**3, y)
    >>> Rem(x**3, y).subs({x: -5, y: 3})
    -2

    See Also
    ========

    Mod
    """
    kind = NumberKind  # Class attribute representing the kind of number

    @classmethod
    def eval(cls, p, q):
        """
        Evaluation method to compute the remainder when p is divided by q.

        Parameters
        ----------
        p : Expr
            Dividend.
        q : Expr
            Divisor.

        Returns
        -------
        Expr
            The remainder of p divided by q.

        Raises
        ------
        ZeroDivisionError
            If q is zero.
        """
        if q.is_zero:
            raise ZeroDivisionError("Division by zero")
        if p is S.NaN or q is S.NaN or p.is_finite is False or q.is_finite is False:
            return S.NaN
        if p is S.Zero or p in (q, -q) or (p.is_integer and q == 1):
            return S.Zero

        if q.is_Number:
            if p.is_Number:
                return p - Integer(p/q)*q
```