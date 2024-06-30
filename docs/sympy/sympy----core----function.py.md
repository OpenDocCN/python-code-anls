# `D:\src\scipysrc\sympy\sympy\core\function.py`

```
"""
There are three types of functions implemented in SymPy:

    1) defined functions (in the sense that they can be evaluated) like
       exp or sin; they have a name and a body:
           f = exp
    2) undefined function which have a name but no body. Undefined
       functions can be defined using a Function class as follows:
           f = Function('f')
       (the result will be a Function instance)
    3) anonymous function (or lambda function) which have a body (defined
       with dummy variables) but have no name:
           f = Lambda(x, exp(x)*x)
           f = Lambda((x, y), exp(x)*y)
    The fourth type of functions are composites, like (sin + cos)(x); these work in
    SymPy core, but are not yet part of SymPy.

    Examples
    ========

    >>> import sympy
    >>> f = sympy.Function("f")
    >>> from sympy.abc import x
    >>> f(x)
    f(x)
    >>> print(sympy.srepr(f(x).func))
    Function('f')
    >>> f(x).args
    (x,)
"""

from __future__ import annotations
from typing import Any
from collections.abc import Iterable

from .add import Add
from .basic import Basic, _atomic
from .cache import cacheit
from .containers import Tuple, Dict
from .decorators import _sympifyit
from .evalf import pure_complex
from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .mul import Mul
from .numbers import Rational, Float, Integer
from .operations import LatticeOp
from .parameters import global_parameters
from .rules import Transform
from .singleton import S
from .sympify import sympify, _sympify

from .sorting import default_sort_key, ordered
from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning, ignore_warnings)
from sympy.utilities.iterables import (has_dups, sift, iterable,
    is_sequence, uniq, topological_sort)
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from sympy.utilities.misc import as_int, filldedent, func_name

import mpmath
from mpmath.libmp.libmpf import prec_to_dps

import inspect
from collections import Counter

def _coeff_isneg(a):
    """
    Return True if the leading Number is negative.

    Examples
    ========

    >>> from sympy.core.function import _coeff_isneg
    >>> from sympy import S, Symbol, oo, pi
    >>> _coeff_isneg(-3*pi)
    True
    >>> _coeff_isneg(S(3))
    False
    >>> _coeff_isneg(-oo)
    True
    >>> _coeff_isneg(Symbol('n', negative=True)) # coeff is 1
    False

    For matrix expressions:

    >>> from sympy import MatrixSymbol, sqrt
    >>> A = MatrixSymbol("A", 3, 3)
    >>> _coeff_isneg(-sqrt(2)*A)
    True
    >>> _coeff_isneg(sqrt(2)*A)
    False
    """
    
    if a.is_MatMul:
        a = a.args[0]
    if a.is_Mul:
        a = a.args[0]
    return a.is_Number and a.is_extended_negative


class PoleError(Exception):
    pass


class ArgumentIndexError(ValueError):
    pass


注释：

"""
SymPy 中实现了三种类型的函数：

1) defined functions（可以求值的函数），例如 exp 或 sin；它们有一个名称和一个主体：
       f = exp
2) undefined function（有名称但没有主体）。可以使用 Function 类定义未定义的函数，如下所示：
       f = Function('f')
   （结果将是一个 Function 实例）
3) anonymous function（或 lambda 函数），它们有一个主体（用虚拟变量定义），但没有名称：
       f = Lambda(x, exp(x)*x)
       f = Lambda((x, y), exp(x)*y)
第四种函数类型是复合函数，如 (sin + cos)(x)；这些在 SymPy 核心中可以工作，但尚不是 SymPy 的一部分。

示例
========

>>> import sympy
>>> f = sympy.Function("f")
>>> from sympy.abc import x
>>> f(x)
f(x)
>>> print(sympy.srepr(f(x).func))
Function('f')
>>> f(x).args
(x,)
"""
    # 定义类的特殊方法 __str__，用于返回对象的字符串表示
    def __str__(self):
        # 返回格式化的字符串，指示出现的无效操作
        return ("Invalid operation with argument number %s for Function %s" %
               (self.args[1], self.args[0]))
# 自定义异常类，用于表示 Lambda 函数签名无效时引发的异常
class BadSignatureError(TypeError):
    '''Raised when a Lambda is created with an invalid signature'''
    pass


# 自定义异常类，用于表示 Lambda 函数调用参数数量不正确时引发的异常
class BadArgumentsError(TypeError):
    '''Raised when a Lambda is called with an incorrect number of arguments'''
    pass


# Python 3 版本的函数装饰器，用于获取函数的参数个数信息
def arity(cls):
    """Return the arity of the function if it is known, else None.

    Explanation
    ===========

    When default values are specified for some arguments, they are
    optional and the arity is reported as a tuple of possible values.

    Examples
    ========

    >>> from sympy import arity, log
    >>> arity(lambda x: x)
    1
    >>> arity(log)
    (1, 2)
    >>> arity(lambda *x: sum(x)) is None
    True
    """
    # 获取函数的 eval 方法或函数本身（如果没有 eval 方法）
    eval_ = getattr(cls, 'eval', cls)

    # 使用 inspect 模块获取函数签名信息
    parameters = inspect.signature(eval_).parameters.items()

    # 如果函数有可变位置参数，则返回 None
    if [p for _, p in parameters if p.kind == p.VAR_POSITIONAL]:
        return

    # 分离出所有的位置或关键字参数
    p_or_k = [p for _, p in parameters if p.kind == p.POSITIONAL_OR_KEYWORD]
    
    # 统计没有默认值的参数个数和有默认值的参数个数
    no, yes = map(len, sift(p_or_k,
        lambda p: p.default == p.empty, binary=True))
    
    # 如果没有参数有默认值，则返回没有默认值的参数个数；否则返回可能的参数个数范围（元组）
    return no if not yes else tuple(range(no, no + yes + 1))


class FunctionClass(type):
    """
    Base class for function classes. FunctionClass is a subclass of type.

    Use Function('<function name>' [ , signature ]) to create
    undefined function classes.
    """
    # 重写 type 类的 __new__ 方法，保存原始的 __new__ 方法
    _new = type.__new__
    def __init__(cls, *args, **kwargs):
        """
        初始化方法，用于类的实例化。
        """
        # 优先使用关键字参数中的 nargs 值，或者类定义中的 nargs 值，否则使用 arity 函数计算的参数个数
        nargs = kwargs.pop('nargs', cls.__dict__.get('nargs', arity(cls)))
        # 如果 nargs 为 None 并且 cls.__dict__ 中没有 'nargs' 键
        if nargs is None and 'nargs' not in cls.__dict__:
            # 遍历类的方法解析顺序（Method Resolution Order，MRO），寻找定义了 '_nargs' 属性的超类
            for supcls in cls.__mro__:
                if hasattr(supcls, '_nargs'):
                    nargs = supcls._nargs
                    break
                else:
                    continue

        # 在这里规范化 nargs；将 nargs 转换为集合中的元组。
        if is_sequence(nargs):
            if not nargs:
                # 如果 nargs 被错误地指定为空序列，则引发 ValueError 异常
                raise ValueError(filldedent('''
                    Incorrectly specified nargs as %s:
                    if there are no arguments, it should be
                    `nargs = 0`;
                    if there are any number of arguments,
                    it should be
                    `nargs = None`''' % str(nargs)))
            nargs = tuple(ordered(set(nargs)))
        elif nargs is not None:
            # 如果 nargs 不为 None，则将其转换为整数类型的元组
            nargs = (as_int(nargs),)
        # 将计算得到的 nargs 赋值给类属性 _nargs
        cls._nargs = nargs

        # 当 __init__ 从 UndefinedFunction 被调用时，它只有一个参数；
        # 当从子类化 Function 被调用时，它通常带有 (name, bases, namespace) 的参数签名。
        if len(args) == 3:
            # 提取第三个参数 namespace
            namespace = args[2]
            # 如果 namespace 中有 'eval' 并且其不是类方法（不是 classmethod 类型）
            if 'eval' in namespace and not isinstance(namespace['eval'], classmethod):
                # 抛出 TypeError 异常，要求 Function 的子类中的 eval 必须是类方法（使用 @classmethod 定义）
                raise TypeError("eval on Function subclasses should be a class method (defined with @classmethod)")

    @property
    def __signature__(self):
        """
        提供 Python 3 中 inspect.signature 用于 Function 子类的有用签名。
        """
        # 尝试导入 inspect 中的 signature 函数
        try:
            from inspect import signature
        except ImportError:
            return None

        # TODO: 查看 nargs
        # 返回 self.eval 的签名
        return signature(self.eval)

    @property
    def free_symbols(self):
        """
        返回一个空集合。
        """
        return set()

    @property
    def xreplace(self):
        """
        返回一个 lambda 函数，用于替换规则中的符号。
        """
        # Function 需要参数，因此定义一个返回函数的属性，
        # 该函数接受参数... 然后使用该函数返回正确的值。
        return lambda rule, **_: rule.get(self, self)
    def nargs(self):
        """Return a set of the allowed number of arguments for the function.

        Examples
        ========

        >>> from sympy import Function
        >>> f = Function('f')

        If the function can take any number of arguments, the set of whole
        numbers is returned:

        >>> Function('f').nargs
        Naturals0

        If the function was initialized to accept one or more arguments, a
        corresponding set will be returned:

        >>> Function('f', nargs=1).nargs
        {1}
        >>> Function('f', nargs=(2, 1)).nargs
        {1, 2}

        The undefined function, after application, also has the nargs
        attribute; the actual number of arguments is always available by
        checking the ``args`` attribute:

        >>> f = Function('f')
        >>> f(1).nargs
        Naturals0
        >>> len(f(1).args)
        1
        """
        from sympy.sets.sets import FiniteSet
        # XXX it would be nice to handle this in __init__ but there are import
        # problems with trying to import FiniteSet there
        # 根据 self._nargs 属性的情况返回合法的参数数量集合
        return FiniteSet(*self._nargs) if self._nargs else S.Naturals0

    def _valid_nargs(self, n : int) -> bool:
        """ Return True if the specified integer is a valid number of arguments

        The number of arguments n is guaranteed to be an integer and positive

        """
        # 如果 self._nargs 存在，则检查 n 是否在 self._nargs 中
        if self._nargs:
            return n in self._nargs

        # 否则获取当前对象的 nargs 属性
        nargs = self.nargs
        # 检查 nargs 是否为 Naturals0，或者 n 是否在 nargs 中
        return nargs is S.Naturals0 or n in nargs

    def __repr__(cls):
        # 返回类的名称作为字符串表示
        return cls.__name__
# 定义一个应用函数的基类，继承自 Basic 类，并使用 FunctionClass 作为元类
class Application(Basic, metaclass=FunctionClass):
    """
    应用函数的基类。

    Explanation
    ===========

    Application 的实例表示将任何类型的应用应用于任何对象的结果。
    """

    # 类属性，表示这是一个函数类
    is_Function = True

    # 使用缓存装饰器对 __new__ 方法进行装饰
    @cacheit
    def __new__(cls, *args, **options):
        # 导入必要的模块
        from sympy.sets.fancysets import Naturals0
        from sympy.sets.sets import FiniteSet

        # 将参数 args 映射为 sympy 的表达式对象
        args = list(map(sympify, args))
        # 从 options 中取出 evaluate 参数，默认使用全局参数的 evaluate 设置
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        # 移除 options 中的 nargs 参数，如果存在的话
        # WildFunction（以及类似的对象）可能会定义 nargs，这里我们忽略这个值
        options.pop('nargs', None)

        # 如果 options 不为空，抛出 ValueError 异常
        if options:
            raise ValueError("Unknown options: %s" % options)

        # 如果需要进行求值
        if evaluate:
            # 调用 eval 方法对参数进行求值
            evaluated = cls.eval(*args)
            # 如果求值结果不为 None，则返回求值结果
            if evaluated is not None:
                return evaluated

        # 调用父类的 __new__ 方法创建对象
        obj = super().__new__(cls, *args, **options)

        # 统一 nargs 的处理
        sentinel = object()
        objnargs = getattr(obj, "nargs", sentinel)
        if objnargs is not sentinel:
            # 对于有序的序列，去重并排序 nargs
            if is_sequence(objnargs):
                nargs = tuple(ordered(set(objnargs)))
            elif objnargs is not None:
                nargs = (as_int(objnargs),)
            else:
                nargs = None
        else:
            # 对于没有指定 nargs 的情况，使用 obj._nargs 属性
            nargs = obj._nargs  # 注意这里的下划线

        # 将 nargs 转换为 FiniteSet
        obj.nargs = FiniteSet(*nargs) if nargs else Naturals0()
        return obj

    @classmethod
    def eval(cls, *args):
        """
        返回应用 cls 到参数 args 的规范形式。

        Explanation
        ===========

        当类 cls 即将实例化时调用 eval() 方法，它应返回一个简化的实例（可能是另一个类的实例），
        或者如果类 cls 不需要修改，则返回 None。

        以 "sign" 函数的 eval() 方法为例

        .. code-block:: python

            @classmethod
            def eval(cls, arg):
                if arg is S.NaN:
                    return S.NaN
                if arg.is_zero: return S.Zero
                if arg.is_positive: return S.One
                if arg.is_negative: return S.NegativeOne
                if isinstance(arg, Mul):
                    coeff, terms = arg.as_coeff_Mul(rational=True)
                    if coeff is not S.One:
                        return cls(coeff) * cls(terms)

        """
        return
    # 定义一个方法 func，返回其所属类的类对象
    def func(self):
        return self.__class__

    # 定义一个方法 _eval_subs，用于执行子表达式的替换操作
    def _eval_subs(self, old, new):
        # 检查旧表达式和新表达式是否都是函数，并且可调用，并且旧表达式等于 self.func，且新表达式的参数个数符合要求
        if (old.is_Function and new.is_Function and
            callable(old) and callable(new) and
            old == self.func and len(self.args) in new.nargs):
            # 对 self.args 中的每个元素应用旧表达式到新表达式的替换，然后调用新表达式生成新的表达式对象
            return new(*[i._subs(old, new) for i in self.args])
# 定义一个继承自 Application 和 Expr 的基类，用于表示数学函数。

# 它也可以作为未定义函数类的构造器。

# 详见“custom-functions”指南，了解如何对“Function”进行子类化以及可以定义哪些方法。

class Function(Application, Expr):
    r"""
    Base class for applied mathematical functions.

    It also serves as a constructor for undefined function classes.

    See the :ref:`custom-functions` guide for details on how to subclass
    ``Function`` and what methods can be defined.

    Examples
    ========

    **Undefined Functions**

    To create an undefined function, pass a string of the function name to
    ``Function``.

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> g = Function('g')(x)
    >>> f
    f
    >>> f(x)
    f(x)
    >>> g
    g(x)
    >>> f(x).diff(x)
    Derivative(f(x), x)
    >>> g.diff(x)
    Derivative(g(x), x)

    Assumptions can be passed to ``Function`` the same as with a
    :class:`~.Symbol`. Alternatively, you can use a ``Symbol`` with
    assumptions for the function name and the function will inherit the name
    and assumptions associated with the ``Symbol``:

    >>> f_real = Function('f', real=True)
    >>> f_real(x).is_real
    True
    >>> f_real_inherit = Function(Symbol('f', real=True))
    >>> f_real_inherit(x).is_real
    True

    Note that assumptions on a function are unrelated to the assumptions on
    the variables it is called on. If you want to add a relationship, subclass
    ``Function`` and define custom assumptions handler methods. See the
    :ref:`custom-functions-assumptions` section of the :ref:`custom-functions`
    guide for more details.

    **Custom Function Subclasses**

    The :ref:`custom-functions` guide has several
    :ref:`custom-functions-complete-examples` of how to subclass ``Function``
    to create a custom function.

    """

    @property
    def _diff_wrt(self):
        # 声明该属性返回 False，用于指示函数对象不支持与它进行微分。
        return False

    @cacheit
    def __new__(cls, *args, **options):
        # 处理类似 Function('f') 这样的调用
        if cls is Function:
            # 返回一个 UndefinedFunction 的实例，传递原始参数和选项
            return UndefinedFunction(*args, **options)

        # 获取参数个数
        n = len(args)

        # 如果参数个数不符合预期，则抛出 TypeError 异常
        if not cls._valid_nargs(n):
            # XXX: 异常消息必须按照这个格式，以便与 NumPy 的函数兼容，例如 vectorize()。详见 https://github.com/numpy/numpy/issues/1697.
            # 理想的解决方案是给异常附加元数据，并修改 NumPy 以利用这些信息。
            temp = ('%(name)s takes %(qual)s %(args)s '
                   'argument%(plural)s (%(given)s given)')
            raise TypeError(temp % {
                'name': cls,
                'qual': 'exactly' if len(cls.nargs) == 1 else 'at least',
                'args': min(cls.nargs),
                'plural': 's'*(min(cls.nargs) != 1),
                'given': n})

        # 获取选项中的 evaluate 参数，或者使用全局参数中的 evaluate 设置
        evaluate = options.get('evaluate', global_parameters.evaluate)
        # 调用父类的 __new__ 方法创建实例
        result = super().__new__(cls, *args, **options)
        # 如果 evaluate 为真，并且 result 是 cls 的实例且具有 args 属性
        if evaluate and isinstance(result, cls) and result.args:
            # 检查每个参数是否应该进行 evalf() 操作
            _should_evalf = [cls._should_evalf(a) for a in result.args]
            # 计算需要 evalf() 的最小精度
            pr2 = min(_should_evalf)
            if pr2 > 0:
                # 计算应用于结果的最大精度
                pr = max(_should_evalf)
                # 对结果进行 evalf() 操作，使用指定的精度转换为位数
                result = result.evalf(prec_to_dps(pr))

        # 将结果转换为 sympy 对象并返回
        return _sympify(result)

    @classmethod
    def _should_evalf(cls, arg):
        """
        决定是否应自动进行 evalf() 操作。

        解释
        ============

        默认情况下（在此实现中），只有当 ARG 是浮点数（包括复数）时才会进行该操作。
        此函数被 __new__ 方法使用。

        返回应该 evalf() 的精度，如果不应该 evalf() 则返回 -1。
        """
        if arg.is_Float:
            return arg._prec
        if not arg.is_Add:
            return -1
        m = pure_complex(arg)
        if m is None:
            return -1
        # m 的元素类型为 Number，因此具有 _prec 属性
        return max(m[0]._prec, m[1]._prec)

    @classmethod
    def class_key(cls):
        from sympy.sets.fancysets import Naturals0
        # 函数名称与对应的优先级映射关系
        funcs = {
            'exp': 10,
            'log': 11,
            'sin': 20,
            'cos': 21,
            'tan': 22,
            'cot': 23,
            'sinh': 30,
            'cosh': 31,
            'tanh': 32,
            'coth': 33,
            'conjugate': 40,
            're': 41,
            'im': 42,
            'arg': 43,
        }
        # 获取当前类名
        name = cls.__name__

        # 尝试从映射中获取函数名称对应的优先级
        try:
            i = funcs[name]
        # 如果不存在对应的函数名称，则设置默认值，若 cls.nargs 是 Naturals0 类型则设为 0，否则设为 10000
        except KeyError:
            i = 0 if isinstance(cls.nargs, Naturals0) else 10000

        # 返回一个元组，其中包含类别，优先级和名称信息
        return 4, i, name
    # 定义一个内部方法，用于评估表达式的数值计算
    def _eval_evalf(self, prec):

        # 定义一个内部函数，根据函数名查找对应的 mpmath 函数
        def _get_mpmath_func(fname):
            """Lookup mpmath function based on name"""
            # 如果是应用未定义的函数，则不应该在 mpmath 中查找，但可能有 ._imp_
            if isinstance(self, AppliedUndef):
                return None

            # 如果 mpmath 中不存在该函数名，则尝试查找其翻译或返回 None
            if not hasattr(mpmath, fname):
                fname = MPMATH_TRANSLATIONS.get(fname, None)
                if fname is None:
                    return None
            return getattr(mpmath, fname)

        # 获取对象自身的 _eval_mpmath 属性，如果不存在则设置为 None
        _eval_mpmath = getattr(self, '_eval_mpmath', None)
        if _eval_mpmath is None:
            # 如果 _eval_mpmath 不存在，则根据函数名获取 mpmath 函数和参数列表
            func = _get_mpmath_func(self.func.__name__)
            args = self.args
        else:
            # 否则调用 _eval_mpmath 方法获取函数和参数
            func, args = _eval_mpmath()

        # 如果未找到可用的函数，则回退到其他评估方式
        if func is None:
            # 获取对象的 _imp_ 属性，如果不存在则返回 None
            imp = getattr(self, '_imp_', None)
            if imp is None:
                return None
            try:
                # 尝试使用 _imp_ 属性计算表达式中每个参数的数值评估，并返回结果
                return Float(imp(*[i.evalf(prec) for i in self.args]), prec)
            except (TypeError, ValueError):
                return None

        # 将所有参数转换为 mpf 或 mpc 类型，并比请求的精度更高地转换为 mpmath 数据类型
        try:
            args = [arg._to_mpmath(prec + 5) for arg in args]

            # 定义一个函数，用于检查参数是否无法正确计算到有意义的精度
            def bad(m):
                from mpmath import mpf, mpc
                # 对于 mpf 值，精度是最后一个元素，如果为 1 则表示评估失败
                if isinstance(m, mpf):
                    m = m._mpf_
                    return m[1] != 1 and m[-1] == 1
                # 对于 mpc 值，检查实部和虚部的精度是否都正确
                elif isinstance(m, mpc):
                    m, n = m._mpc_
                    return m[1] != 1 and m[-1] == 1 and \
                        n[1] != 1 and n[-1] == 1
                else:
                    return False

            # 如果任何参数无法正确计算到有意义的精度，则引发 ValueError
            if any(bad(a) for a in args):
                raise ValueError

        except ValueError:
            return

        # 使用指定精度进行计算
        with mpmath.workprec(prec):
            v = func(*args)

        # 将 mpmath 计算结果转换为 SymPy 表达式
        return Expr._from_mpmath(v, prec)

    # 定义一个内部方法，用于计算表达式关于给定变量 s 的导数
    def _eval_derivative(self, s):
        # f(x).diff(s) -> x.diff(s) * f.fdiff(1)(s)
        i = 0
        l = []
        for a in self.args:
            i += 1
            # 计算当前参数关于 s 的导数
            da = a.diff(s)
            # 如果导数为零，则跳过当前参数
            if da.is_zero:
                continue
            try:
                # 尝试获取当前参数在表达式中的导数
                df = self.fdiff(i)
            except ArgumentIndexError:
                # 如果获取失败，则调用 Function.fdiff 方法获取导数
                df = Function.fdiff(self, i)
            # 将当前参数的导数乘以相应的表达式导数，并加入到列表中
            l.append(df * da)
        # 返回所有项的和，即表达式关于 s 的导数
        return Add(*l)
    # 返回一个布尔值，指示表达式是否是可交换的
    def _eval_is_commutative(self):
        return fuzzy_and(a.is_commutative for a in self.args)

    # 判断函数是否是亚分析函数
    def _eval_is_meromorphic(self, x, a):
        if not self.args:
            return True
        # 如果除了第一个参数外还有其他参数涉及到变量 x，则不是亚分析函数
        if any(arg.has(x) for arg in self.args[1:]):
            return False

        arg = self.args[0]
        # 检查第一个参数是否是亚分析函数
        if not arg._eval_is_meromorphic(x, a):
            return None

        # 判断参数是否是奇异点
        return fuzzy_not(type(self).is_singular(arg.subs(x, a)))

    # 声明一个属性 _singularities，类型可以是 FuzzyBool 或者元组，初始值为 None
    _singularities: FuzzyBool | tuple[Expr, ...] = None

    # 类方法，判断参数是否是本质奇异点、分支点或者非全纯函数
    @classmethod
    def is_singular(cls, a):
        """
        Tests whether the argument is an essential singularity
        or a branch point, or the functions is non-holomorphic.
        """
        ss = cls._singularities
        # 如果 _singularities 是 True、None 或 False 中的一个，直接返回其值
        if ss in (True, None, False):
            return ss

        # 判断参数 a 是否是无穷大，或者是否与 _singularities 中的某个元素之差为零
        return fuzzy_or(a.is_infinite if s is S.ComplexInfinity
                        else (a - s).is_zero for s in ss)

    # 返回对象自身作为 (base, exponent) 的 2 元组
    def as_base_exp(self):
        """
        Returns the method as the 2-tuple (base, exponent).
        """
        return self, S.One

    # 计算关于 self.args 的参数 args0 在 x 处的渐近级数展开
    # 此函数仅被 _eval_nseries 内部使用，不应直接调用；派生类可以重写此函数实现渐近级数展开
    def _eval_aseries(self, n, args0, x, logx):
        """
        Compute an asymptotic expansion around args0, in terms of self.args.
        This function is only used internally by _eval_nseries and should not
        be called directly; derived classes can overwrite this to implement
        asymptotic expansions.
        """
        raise PoleError(filldedent('''
            Asymptotic expansion of %s around %s is
            not implemented.''' % (type(self), args0)))

    # 返回函数的第一个导数
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of the function.
        """
        # 检查参数索引是否在有效范围内
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        ix = argindex - 1
        A = self.args[ix]
        # 如果参数 A 是可以区分的
        if A._diff_wrt:
            # 如果函数只有一个参数，或者 A 不是符号，则调用 _derivative_dispatch 函数
            if len(self.args) == 1 or not A.is_Symbol:
                return _derivative_dispatch(self, A)
            # 否则，检查是否有其他参数的自由符号中包含 A
            for i, v in enumerate(self.args):
                if i != ix and A in v.free_symbols:
                    # 不能在其他参数的自由符号中出现，参考问题 8510
                    break
            else:
                return _derivative_dispatch(self, A)

        # 参考问题 4624、4719、5600 和 8510
        # 创建一个虚拟符号 D，用于替换 A
        D = Dummy('xi_%i' % argindex, dummy_index=hash(A))
        args = self.args[:ix] + (D,) + self.args[ix + 1:]
        # 返回关于 D 的导数的代数表达式，其中 D 替换了 A
        return Subs(Derivative(self.func(*args), D), D, A)
    # 定义一个名为 _eval_as_leading_term 的方法，用于子类重写，返回在 x -> 0 时系列的第一个非零项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        """Stub that should be overridden by new Functions to return
        the first non-zero term in a series if ever an x-dependent
        argument whose leading term vanishes as x -> 0 might be encountered.
        See, for example, cos._eval_as_leading_term.
        """
        # 导入 Order 类用于计算序列
        from sympy.series.order import Order
        # 遍历当前对象的所有参数，计算每个参数关于 x 的主导项
        args = [a.as_leading_term(x, logx=logx) for a in self.args]
        # 创建一个 Order 对象表示阶数为 1 关于 x 的序列
        o = Order(1, x)
        # 如果任意参数中包含 x，并且该参数在 O(1, x) 的范围内
        if any(x in a.free_symbols and o.contains(a) for a in args):
            # 对于 x 和任何有限数都包含在 O(1, x) 中，
            # 但是表达式如 1/x 不包含在其中。如果任何参数简化为在 x -> 0 时消失的表达式
            # （如 x 或 x**2，但不包括 3、1/x 等），则需要 _eval_as_leading_term 方法
            # 来提供系列的第一个非零项。
            #
            # 例如，表达式           主导项
            #      ----------    ------------
            #      cos(1/x)      cos(1/x)
            #      cos(cos(x))   cos(1)
            #      cos(x)        1        <- 需要 _eval_as_leading_term
            #      sin(x)        x        <- 需要 _eval_as_leading_term
            #
            # 抛出未实现错误，指出该函数没有 _eval_as_leading_term 的实现
            raise NotImplementedError(
                '%s has no _eval_as_leading_term routine' % self.func)
        else:
            # 如果所有参数的主导项均不会在 x -> 0 时消失，则返回当前对象自身
            return self
class AppliedUndef(Function):
    """
    Base class for expressions resulting from the application of an undefined
    function.
    """

    is_number = False  # 设置实例属性 is_number 为 False

    def __new__(cls, *args, **options):
        # 将所有参数转换为 sympy 的表达式对象
        args = list(map(sympify, args))
        # 找出参数中的 UndefinedFunction 实例的名称
        u = [a.name for a in args if isinstance(a, UndefinedFunction)]
        # 如果存在 UndefinedFunction 实例，则抛出类型错误
        if u:
            raise TypeError('Invalid argument: expecting an expression, not UndefinedFunction%s: %s' % (
                's'*(len(u) > 1), ', '.join(u)))
        # 调用父类的 __new__ 方法创建对象
        obj = super().__new__(cls, *args, **options)
        return obj

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 返回自身作为主导项
        return self

    @property
    def _diff_wrt(self):
        """
        Allow derivatives wrt to undefined functions.

        Examples
        ========

        >>> from sympy import Function, Symbol
        >>> f = Function('f')
        >>> x = Symbol('x')
        >>> f(x)._diff_wrt
        True
        >>> f(x).diff(x)
        Derivative(f(x), x)
        """
        return True  # 允许对未定义函数求导


class UndefSageHelper:
    """
    Helper to facilitate Sage conversion.
    """
    def __get__(self, ins, typ):
        import sage.all as sage
        # 如果未绑定实例，返回一个函数，以 typ 的名称创建 Sage 函数
        if ins is None:
            return lambda: sage.function(typ.__name__)
        else:
            # 否则，将实例的参数转换为 Sage 可识别的对象，并创建对应的 Sage 函数
            args = [arg._sage_() for arg in ins.args]
            return lambda : sage.function(ins.__class__.__name__)(*args)

_undef_sage_helper = UndefSageHelper()  # 创建 UndefSageHelper 实例

class UndefinedFunction(FunctionClass):
    """
    The (meta)class of undefined functions.
    """
    # 定义一个元类的特殊方法 `__new__`，用于创建新的类实例
    def __new__(mcl, name, bases=(AppliedUndef,), __dict__=None, **kwargs):
        # 导入符号模块中的 _filter_assumptions 函数
        from .symbol import _filter_assumptions
        # 允许通过 Function('f', real=True) 或 Function(Symbol('f', real=True)) 方式设置假设
        assumptions, kwargs = _filter_assumptions(kwargs)
        
        # 如果 name 是 Symbol 类型，则合并假设，并将 name 转换为其名称
        if isinstance(name, Symbol):
            assumptions = name._merge(assumptions)
            name = name.name
        # 如果 name 不是 str 或 Symbol 类型，则抛出 TypeError 异常
        elif not isinstance(name, str):
            raise TypeError('expecting string or Symbol for name')
        else:
            # 检查假设中是否存在 'commutative'，若不存在则删除
            commutative = assumptions.get('commutative', None)
            assumptions = Symbol(name, **assumptions).assumptions0
            if commutative is None:
                assumptions.pop('commutative')
        
        # 如果 __dict__ 为空，则初始化为空字典
        __dict__ = __dict__ or {}
        
        # 将 assumptions 中的键值对以 'is_*' 的形式加入到 __dict__ 中
        __dict__.update({'is_%s' % k: v for k, v in assumptions.items()})
        
        # 将 kwargs 中的其他属性加入到 __dict__ 中
        __dict__.update(kwargs)
        
        # 将去除 'is_' 前缀后的假设重新加入到 kwargs 中
        kwargs.update(assumptions)
        
        # 将 _kwargs 属性加入到 __dict__ 中
        __dict__.update({'_kwargs': kwargs})
        
        # 为了正确进行 pickle 操作，设置 __module__ 为 None
        __dict__['__module__'] = None
        
        # 使用父类的 __new__ 方法创建类的新实例
        obj = super().__new__(mcl, name, bases, __dict__)
        
        # 设置实例的 name 和 _sage_ 属性
        obj.name = name
        obj._sage_ = _undef_sage_helper
        
        # 返回创建的类实例
        return obj

    # 定义一个类方法 `__instancecheck__`，用于检查类是否是实例的一部分
    def __instancecheck__(cls, instance):
        return cls in type(instance).__mro__

    # 类的类属性 _kwargs，类型为字典，键为字符串，值为布尔值或 None 类型
    _kwargs: dict[str, bool | None] = {}

    # 定义对象的哈希方法 `__hash__`
    def __hash__(self):
        # 返回对象哈希值，包括 class_key() 方法和 _kwargs 的不可变集合
        return hash((self.class_key(), frozenset(self._kwargs.items())))

    # 定义对象的相等比较方法 `__eq__`
    def __eq__(self, other):
        # 检查对象是否是同一类，并比较 class_key() 方法和 _kwargs 属性是否相等
        return (isinstance(other, self.__class__) and
                self.class_key() == other.class_key() and
                self._kwargs == other._kwargs)

    # 定义对象的不等比较方法 `__ne__`
    def __ne__(self, other):
        # 判断对象是否不相等
        return not self == other

    # 定义对象的只读属性 `_diff_wrt`
    @property
    def _diff_wrt(self):
        # 返回 False，表示对象不支持微分
        return False
# XXX: The type: ignore on WildFunction is because mypy complains:
#
# sympy/core/function.py:939: error: Cannot determine type of 'sort_key' in
# base class 'Expr'
#
# Somehow this is because of the @cacheit decorator but it is not clear how to
# fix it.

class WildFunction(Function, AtomicExpr):  # type: ignore
    """
    A WildFunction function matches any function (with its arguments).

    Examples
    ========

    >>> from sympy import WildFunction, Function, cos
    >>> from sympy.abc import x, y
    >>> F = WildFunction('F')
    >>> f = Function('f')
    >>> F.nargs
    Naturals0
    >>> x.match(F)
    >>> F.match(F)
    {F_: F_}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> cos(x).match(F)
    {F_: cos(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a given number of arguments, set ``nargs`` to the
    desired value at instantiation:

    >>> F = WildFunction('F', nargs=2)
    >>> F.nargs
    {2}
    >>> f(x).match(F)
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a range of arguments, set ``nargs`` to a tuple
    containing the desired number of arguments, e.g. if ``nargs = (1, 2)``
    then functions with 1 or 2 arguments will be matched.

    >>> F = WildFunction('F', nargs=(1, 2))
    >>> F.nargs
    {1, 2}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}
    >>> f(x, y, 1).match(F)

    """

    # XXX: What is this class attribute used for?
    include: set[Any] = set()

    def __init__(cls, name, **assumptions):
        from sympy.sets.sets import Set, FiniteSet
        cls.name = name
        nargs = assumptions.pop('nargs', S.Naturals0)
        if not isinstance(nargs, Set):
            # Canonicalize nargs here.  See also FunctionClass.
            if is_sequence(nargs):
                nargs = tuple(ordered(set(nargs)))
            elif nargs is not None:
                nargs = (as_int(nargs),)
            nargs = FiniteSet(*nargs)
        cls.nargs = nargs

    def matches(self, expr, repl_dict=None, old=False):
        if not isinstance(expr, (AppliedUndef, Function)):
            return None
        if len(expr.args) not in self.nargs:
            return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        repl_dict[self] = expr
        return repl_dict


这段代码定义了一个 `WildFunction` 类和其相关的功能和属性。
    the symbol and the count:

        >>> Derivative(f(x), x, x, y, x)
        Derivative(f(x), (x, 2), y, x)

# 对函数 f(x) 进行偏导数计算，指定变量 x 进行两次，变量 y 和再次变量 x 一次

    If the derivative cannot be performed, and evaluate is True, the
    order of the variables of differentiation will be made canonical:

        >>> Derivative(f(x, y), y, x, evaluate=True)
        Derivative(f(x, y), x, y)

# 如果无法计算导数，并且 evaluate 参数设置为 True，将按照字典序对变量进行排序以进行求导

    Derivatives with respect to undefined functions can be calculated:

        >>> Derivative(f(x)**2, f(x), evaluate=True)
        2*f(x)

# 可以对未定义函数进行导数计算，此处示例对 f(x)**2 求关于 f(x) 的导数

    Such derivatives will show up when the chain rule is used to
    evalulate a derivative:

        >>> f(g(x)).diff(x)
        Derivative(f(g(x)), g(x))*Derivative(g(x), x)

# 当使用链式法则计算导数时，将显示此类导数的形式

    Substitution is used to represent derivatives of functions with
    arguments that are not symbols or functions:

        >>> f(2*x + 3).diff(x) == 2*Subs(f(y).diff(y), y, 2*x + 3)
        True

# 使用替换来表示对具有非符号或函数参数的函数的导数

    Notes
    =====

    Simplification of high-order derivatives:

    Because there can be a significant amount of simplification that can be
    done when multiple differentiations are performed, results will be
    automatically simplified in a fairly conservative fashion unless the
    keyword ``simplify`` is set to False.

        >>> from sympy import sqrt, diff, Function, symbols
        >>> from sympy.abc import x, y, z
        >>> f, g = symbols('f,g', cls=Function)

        >>> e = sqrt((x + 1)**2 + x)
        >>> diff(e, (x, 5), simplify=False).count_ops()
        136
        >>> diff(e, (x, 5)).count_ops()
        30

# 高阶导数的简化说明

    Ordering of variables:

    If evaluate is set to True and the expression cannot be evaluated, the
    list of differentiation symbols will be sorted, that is, the expression is
    assumed to have continuous derivatives up to the order asked.

# 变量的排序说明

    Derivative wrt non-Symbols:

    For the most part, one may not differentiate wrt non-symbols.
    For example, we do not allow differentiation wrt `x*y` because
    there are multiple ways of structurally defining where x*y appears
    in an expression: a very strict definition would make
    (x*y*z).diff(x*y) == 0. Derivatives wrt defined functions (like
    cos(x)) are not allowed, either:

        >>> (x*y*z).diff(x*y)
        Traceback (most recent call last):
        ...
        ValueError: Can't calculate derivative wrt x*y.

# 非符号的导数说明

    To make it easier to work with variational calculus, however,
    derivatives wrt AppliedUndef and Derivatives are allowed.
    For example, in the Euler-Lagrange method one may write
    F(t, u, v) where u = f(t) and v = f'(t). These variables can be
    written explicitly as functions of time::

        >>> from sympy.abc import t
        >>> F = Function('F')
        >>> U = f(t)
        >>> V = U.diff(t)

    The derivative wrt f(t) can be obtained directly:

        >>> direct = F(t, U, V).diff(U)

# 允许针对 AppliedUndef 和 Derivatives 进行导数计算

    When differentiation wrt a non-Symbol is attempted, the non-Symbol
    is temporarily converted to a Symbol while the differentiation

# 尝试对非符号进行导数计算时，临时将非符号转换为符号进行计算
    is_Derivative = True



    # 设置属性 is_Derivative 为 True，表示这个对象是一个 Derivative（导数）对象
    @property



        # This attribute is for holding derivatives.
    # 返回表达式是否可以针对 Derivative 进行微分
    def _diff_wrt(self):
        """An expression may be differentiated wrt a Derivative if
        it is in elementary form.

        Examples
        ========

        >>> from sympy import Function, Derivative, cos
        >>> from sympy.abc import x
        >>> f = Function('f')

        >>> Derivative(f(x), x)._diff_wrt
        True
        >>> Derivative(cos(x), x)._diff_wrt
        False
        >>> Derivative(x + 1, x)._diff_wrt
        False

        A Derivative might be an unevaluated form of what will not be
        a valid variable of differentiation if evaluated. For example,

        >>> Derivative(f(f(x)), x).doit()
        Derivative(f(x), x)*Derivative(f(f(x)), f(x))

        Such an expression will present the same ambiguities as arise
        when dealing with any other product, like ``2*x``, so ``_diff_wrt``
        is False:

        >>> Derivative(f(f(x)), x)._diff_wrt
        False
        """
        return self.expr._diff_wrt and isinstance(self.doit(), Derivative)

    # 返回规范化的 Derivative 对象
    @property
    def canonical(cls):
        return cls.func(cls.expr,
            *Derivative._sort_variable_count(cls.variable_count))

    # 返回是否表达式是可交换的
    @classmethod
    def _eval_is_commutative(self):
        return self.expr.is_commutative

    # 计算对变量 v 的导数
    def _eval_derivative(self, v):
        # 如果 v（微分变量）不在 self._wrt_variables 中，则尝试计算导数
        if v not in self._wrt_variables:
            dedv = self.expr.diff(v)
            if isinstance(dedv, Derivative):
                return dedv.func(dedv.expr, *(self.variable_count + dedv.variable_count))
            # 如果 dedv（d(self.expr)/dv）可以简化，使得对 self.variables 中的变量的导数现在可以计算，则设置 evaluate=True
            # 看看是否还有其他可以进行的导数计算。最常见的情况是 dedv 是一个简单的数值，因此对任何其他变量的导数将消失。
            return self.func(dedv, *self.variables, evaluate=True)
        # 如果 v 在 self.variables 中，则已经尝试过对 v 的导数计算，并且原始时 evaluate=False 或者计算失败。
        variable_count = list(self.variable_count)
        variable_count.append((v, 1))
        return self.func(self.expr, *variable_count, evaluate=False)

    # 对表达式进行求值，根据 hints 参数决定是否深度求值
    def doit(self, **hints):
        expr = self.expr
        if hints.get('deep', True):
            expr = expr.doit(**hints)
        hints['evaluate'] = True
        rv = self.func(expr, *self.variable_count, **hints)
        # 如果结果 rv 不等于 self，并且 rv 中包含 Derivative，则继续深度求值
        if rv != self and rv.has(Derivative):
            rv = rv.doit(**hints)
        return rv

    # 用于实现的装饰器，将参数 z0 转换为 Sympy 对象，否则引发 NotImplementedError
    @_sympifyit('z0', NotImplementedError)
    # 定义一个方法，用于在数值上评估在 z0 处的导数。
    def doit_numerically(self, z0):
        """
        Evaluate the derivative at z numerically.

        When we can represent derivatives at a point, this should be folded
        into the normal evalf. For now, we need a special method.
        """
        # 如果自由符号数不为1或变量数不为1，则抛出未实现错误，暂不支持偏导数和高阶导数
        if len(self.free_symbols) != 1 or len(self.variables) != 1:
            raise NotImplementedError('partials and higher order derivatives')
        z = list(self.free_symbols)[0]

        def eval(x):
            # 使用 mpmath 精度创建表达式，并进行 evalf 计算
            f0 = self.expr.subs(z, Expr._from_mpmath(x, prec=mpmath.mp.prec))
            f0 = f0.evalf(prec_to_dps(mpmath.mp.prec))
            return f0._to_mpmath(mpmath.mp.prec)
        # 使用 mpmath 计算导数并返回结果
        return Expr._from_mpmath(mpmath.diff(eval,
                                             z0._to_mpmath(mpmath.mp.prec)),
                                 mpmath.mp.prec)

    @property
    # 返回表达式对象的属性
    def expr(self):
        return self._args[0]

    @property
    # 返回用于微分的变量列表，不考虑类型计数（整数或符号）
    def _wrt_variables(self):
        return [i[0] for i in self.variable_count]

    @property
    # 返回微分变量的属性
    def variables(self):
        # TODO: 废弃？是的，将此改为 'enumerated_variables'，并将 _wrt_variables 命名为 variables
        # TODO: 支持 `d^n`？
        rv = []
        for v, count in self.variable_count:
            if not count.is_Integer:
                raise TypeError(filldedent('''
                Cannot give expansion for symbolic count. If you just
                want a list of all variables of differentiation, use
                _wrt_variables.'''))
            rv.extend([v]*count)
        return tuple(rv)

    @property
    # 返回变量计数的属性
    def variable_count(self):
        return self._args[1:]

    @property
    # 返回导数计数的属性
    def derivative_count(self):
        return sum([count for _, count in self.variable_count], 0)

    @property
    # 返回表达式的自由符号集合，包括符号计数
    def free_symbols(self):
        ret = self.expr.free_symbols
        # 将符号计数添加到自由符号集合中
        for _, count in self.variable_count:
            ret.update(count.free_symbols)
        return ret

    @property
    # 返回表达式对象的种类
    def kind(self):
        return self.args[0].kind

    # 定义一个方法，计算表达式的级数展开，并以变量的形式返回
    def _eval_lseries(self, x, logx, cdir=0):
        dx = self.variables
        for term in self.expr.lseries(x, logx=logx, cdir=cdir):
            yield self.func(term, *dx)

    # 定义一个方法，计算表达式的 n 级数展开，并以变量的形式返回
    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.expr.nseries(x, n=n, logx=logx)
        o = arg.getO()
        dx = self.variables
        rv = [self.func(a, *dx) for a in Add.make_args(arg.removeO())]
        if o:
            rv.append(o/x)
        return Add(*rv)

    # 定义一个方法，计算表达式的主导项，并返回相应的微分结果
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        series_gen = self.expr.lseries(x)
        d = S.Zero
        for leading_term in series_gen:
            d = diff(leading_term, *self.variables)
            if d != 0:
                break
        return d

    @classmethod
    # 类方法：返回一个与给定表达式形状相同的零值
    def _get_zero_with_shape_like(cls, expr):
        return S.Zero
    @classmethod
    # 类方法装饰器，指示下面的函数是一个类方法，可以通过类调用而不是实例调用
    def _dispatch_eval_derivative_n_times(cls, expr, v, count):
        # _dispatch_eval_derivative_n_times 方法用于多次求导。
        # 如果当前对象没有覆盖 _eval_derivative_n_times 方法，
        # 在 Basic 类中的默认实现将调用一个循环来执行 _eval_derivative 方法：
        return expr._eval_derivative_n_times(v, count)
def _derivative_dispatch(expr, *variables, **kwargs):
    # 导入必要的类和函数
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.matrices.expressions.matexpr import MatrixExpr
    from sympy.tensor.array import NDimArray
    array_types = (MatrixBase, MatrixExpr, NDimArray, list, tuple, Tuple)
    # 检查表达式和变量的类型，确定是否需要使用 ArrayDerivative 处理
    if isinstance(expr, array_types) or any(isinstance(i[0], array_types) if isinstance(i, (tuple, list, Tuple)) else isinstance(i, array_types) for i in variables):
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        # 返回 ArrayDerivative 处理后的结果
        return ArrayDerivative(expr, *variables, **kwargs)
    # 否则使用普通的 Derivative 处理
    return Derivative(expr, *variables, **kwargs)


class Lambda(Expr):
    """
    Lambda(x, expr) represents a lambda function similar to Python's
    'lambda x: expr'. A function of several variables is written as
    Lambda((x, y, ...), expr).

    Examples
    ========

    A simple example:

    >>> from sympy import Lambda
    >>> from sympy.abc import x
    >>> f = Lambda(x, x**2)
    >>> f(4)
    16

    For multivariate functions, use:

    >>> from sympy.abc import y, z, t
    >>> f2 = Lambda((x, y, z, t), x + y**z + t**z)
    >>> f2(1, 2, 3, 4)
    73

    It is also possible to unpack tuple arguments:

    >>> f = Lambda(((x, y), z), x + y + z)
    >>> f((1, 2), 3)
    6

    A handy shortcut for lots of arguments:

    >>> p = x, y, z
    >>> f = Lambda(p, x + y*z)
    >>> f(*p)
    x + y*z

    """
    is_Function = True

    def __new__(cls, signature, expr):
        if iterable(signature) and not isinstance(signature, (tuple, Tuple)):
            sympy_deprecation_warning(
                """
                Using a non-tuple iterable as the first argument to Lambda
                is deprecated. Use Lambda(tuple(args), expr) instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-non-tuple-lambda",
            )
            signature = tuple(signature)
        sig = signature if iterable(signature) else (signature,)
        sig = sympify(sig)
        # 检查函数签名的合法性
        cls._check_signature(sig)

        if len(sig) == 1 and sig[0] == expr:
            return S.IdentityFunction

        return Expr.__new__(cls, sig, sympify(expr))

    @classmethod
    def _check_signature(cls, sig):
        syms = set()

        def rcheck(args):
            for a in args:
                if a.is_symbol:
                    if a in syms:
                        raise BadSignatureError("Duplicate symbol %s" % a)
                    syms.add(a)
                elif isinstance(a, Tuple):
                    rcheck(a)
                else:
                    raise BadSignatureError("Lambda signature should be only tuples"
                        " and symbols, not %s" % a)

        if not isinstance(sig, Tuple):
            raise BadSignatureError("Lambda signature should be a tuple not %s" % sig)
        # 递归检查函数签名
        rcheck(sig)

    @property
    # 返回函数的第一个参数作为签名
    def signature(self):
        """The expected form of the arguments to be unpacked into variables"""
        return self._args[0]

    # 返回函数的第二个参数作为表达式的返回值
    @property
    def expr(self):
        """The return value of the function"""
        return self._args[1]

    # 返回函数内部表示中使用的变量集合
    @property
    def variables(self):
        """The variables used in the internal representation of the function"""
        def _variables(args):
            if isinstance(args, Tuple):
                for arg in args:
                    yield from _variables(arg)
            else:
                yield args
        return tuple(_variables(self.signature))

    # 返回函数的参数数量，使用 FiniteSet 表示
    @property
    def nargs(self):
        from sympy.sets.sets import FiniteSet
        return FiniteSet(len(self.signature))

    # 将函数的自由符号集合返回，排除已知变量
    @property
    def free_symbols(self):
        return self.expr.free_symbols - set(self.variables)

    # 调用函数对象，根据参数匹配签名并返回表达式的计算结果
    def __call__(self, *args):
        n = len(args)
        if n not in self.nargs:  # Lambda 只能有 nargs 中的 1 个参数
            # XXX: 异常消息必须严格按照这种格式，以使其与 NumPy 的函数如 vectorize() 兼容。
            # 详见 https://github.com/numpy/numpy/issues/1697.
            # 理想的解决方案是仅附加元数据到异常，并修改 NumPy 以利用此功能。
            ## XXX Lambda 适用此规则吗？如果不适用，请移除此注释。
            temp = ('%(name)s takes exactly %(args)s '
                   'argument%(plural)s (%(given)s given)')
            raise BadArgumentsError(temp % {
                'name': self,
                'args': list(self.nargs)[0],
                'plural': 's'*(list(self.nargs)[0] != 1),
                'given': n})

        # 匹配函数签名和参数，返回符号参数映射
        d = self._match_signature(self.signature, args)

        # 替换表达式中的符号参数并返回结果
        return self.expr.xreplace(d)

    # 将函数签名与参数进行匹配，并返回符号参数映射
    def _match_signature(self, sig, args):

        symargmap = {}

        def rmatch(pars, args):
            for par, arg in zip(pars, args):
                if par.is_symbol:
                    symargmap[par] = arg
                elif isinstance(par, Tuple):
                    if not isinstance(arg, (tuple, Tuple)) or len(args) != len(pars):
                        raise BadArgumentsError("Can't match %s and %s" % (args, pars))
                    rmatch(par, arg)

        rmatch(sig, args)

        return symargmap

    # 如果 Lambda 是一个恒等函数，则返回 True
    @property
    def is_identity(self):
        """Return ``True`` if this ``Lambda`` is an identity function. """
        return self.signature == self.expr

    # 对表达式进行数值估算，返回结果
    def _eval_evalf(self, prec):
        return self.func(self.args[0], self.args[1].evalf(n=prec_to_dps(prec)))
class Subs(Expr):
    """
    Represents unevaluated substitutions of an expression.

    ``Subs(expr, x, x0)`` represents the expression resulting
    from substituting x with x0 in expr.

    Parameters
    ==========

    expr : Expr
        An expression.

    x : tuple, variable
        A variable or list of distinct variables.

    x0 : tuple or list of tuples
        A point or list of evaluation points
        corresponding to those variables.

    Examples
    ========

    >>> from sympy import Subs, Function, sin, cos
    >>> from sympy.abc import x, y, z
    >>> f = Function('f')

    Subs are created when a particular substitution cannot be made. The
    x in the derivative cannot be replaced with 0 because 0 is not a
    valid variables of differentiation:

    >>> f(x).diff(x).subs(x, 0)
    Subs(Derivative(f(x), x), x, 0)

    Once f is known, the derivative and evaluation at 0 can be done:

    >>> _.subs(f, sin).doit() == sin(x).diff(x).subs(x, 0) == cos(0)
    True

    Subs can also be created directly with one or more variables:

    >>> Subs(f(x)*sin(y) + z, (x, y), (0, 1))
    Subs(z + f(x)*sin(y), (x, y), (0, 1))
    >>> _.doit()
    z + f(0)*sin(1)

    Notes
    =====

    ``Subs`` objects
    # 定义一个新的类构造函数，用于创建一个符号表达式对象
    def __new__(cls, expr, variables, point, **assumptions):
        # 如果变量不是元组，将其转换为列表
        if not is_sequence(variables, Tuple):
            variables = [variables]
        # 将变量转换为元组
        variables = Tuple(*variables)

        # 检查变量中是否有重复项
        if has_dups(variables):
            # 找出重复的变量名并组成字符串
            repeated = [str(v) for v, i in Counter(variables).items() if i > 1]
            __ = ', '.join(repeated)
            # 抛出值错误，指明哪些表达式出现了多次
            raise ValueError(filldedent('''
                The following expressions appear more than once: %s
                ''' % __))

        # 将点坐标转换为元组，确保其与变量个数相同
        point = Tuple(*(point if is_sequence(point, Tuple) else [point]))

        # 如果点坐标数量与变量数量不同，抛出值错误
        if len(point) != len(variables):
            raise ValueError('Number of point values must be the same as '
                             'the number of variables.')

        # 如果点坐标为空，直接返回符号化的表达式
        if not point:
            return sympify(expr)

        # 如果表达式是一个 Subs 对象，将其变量和点坐标合并
        if isinstance(expr, Subs):
            variables = expr.variables + variables
            point = expr.point + point
            expr = expr.expr
        else:
            # 否则，将表达式符号化
            expr = sympify(expr)

        # 定义一个自定义的字符串打印器类，用于处理符号表达式的打印
        # 使用与点坐标值相等的符号名称（带下划线前缀）
        pre = "_"
        pts = sorted(set(point), key=default_sort_key)
        from sympy.printing.str import StrPrinter
        class CustomStrPrinter(StrPrinter):
            def _print_Dummy(self, expr):
                return str(expr) + str(expr.dummy_index)
        def mystr(expr, **settings):
            p = CustomStrPrinter(settings)
            return p.doprint(expr)
        
        # 循环直到找到符合条件的符号名称
        while 1:
            s_pts = {p: Symbol(pre + mystr(p)) for p in pts}
            reps = [(v, s_pts[p]) for v, p in zip(variables, point)]
            # 如果任何带下划线前缀的符号已经是自由符号，并且其与变量的点坐标不同，
            # 则表示有冲突，需要重新选择符号名称
            if any(r in expr.free_symbols and
                   r in variables and
                   Symbol(pre + mystr(point[variables.index(r)])) != r
                   for _, r in reps):
                pre += "_"
                continue
            break

        # 创建一个新的表达式对象，并进行变量替换
        obj = Expr.__new__(cls, expr, Tuple(*variables), point)
        obj._expr = expr.xreplace(dict(reps))
        return obj

    # 定义一个方法，用于判断表达式是否可交换
    def _eval_is_commutative(self):
        return self.expr.is_commutative
    def doit(self, **hints):
        # 从 self.args 中获取表达式、变量列表和替换点列表
        e, v, p = self.args

        # 移除自身映射
        for i, (vi, pi) in enumerate(zip(v, p)):
            # 如果变量和替换点相等，则从列表中移除对应位置的变量和替换点
            if vi == pi:
                v = v[:i] + v[i + 1:]
                p = p[:i] + p[i + 1:]
        # 如果变量列表为空，则直接返回原始表达式
        if not v:
            return self.expr

        # 如果表达式是 Derivative 类型
        if isinstance(e, Derivative):
            # 首先应用函数替换，例如将函数 vi 替换为相应的 pi
            undone = []
            for i, vi in enumerate(v):
                if isinstance(vi, FunctionClass):
                    e = e.subs(vi, p[i])
                else:
                    undone.append((vi, p[i]))
            # 如果替换后的表达式不再是 Derivative 类型，则继续求导
            if not isinstance(e, Derivative):
                e = e.doit()
            # 如果仍然是 Derivative 类型
            if isinstance(e, Derivative):
                # 对于不涉及微分的替换进行处理
                undone2 = []
                D = Dummy()
                arg = e.args[0]
                for vi, pi in undone:
                    # 如果 vi 不在表达式的自由变量中，则进行替换
                    if D not in e.xreplace({vi: D}).free_symbols:
                        if arg.has(vi):
                            e = e.subs(vi, pi)
                    else:
                        undone2.append((vi, pi))
                undone = undone2
                # 对于存在的变量进行微分操作
                wrt = []
                D = Dummy()
                expr = e.expr
                free = expr.free_symbols
                for vi, ci in e.variable_count:
                    if isinstance(vi, Symbol) and vi in free:
                        expr = expr.diff((vi, ci))
                    elif D in expr.subs(vi, D).free_symbols:
                        expr = expr.diff((vi, ci))
                    else:
                        wrt.append((vi, ci))
                # 注入剩余的替换
                rv = expr.subs(undone)
                # 按给定顺序进行剩余的微分操作
                for vc in wrt:
                    rv = rv.diff(vc)
            else:
                # 注入剩余的替换
                rv = e.subs(undone)
        else:
            # 如果表达式不是 Derivative 类型，则直接进行替换
            rv = e.doit(**hints).subs(list(zip(v, p)))

        # 如果 hints 中包含 'deep' 键，并且 rv 不等于 self，则继续递归调用 doit 方法
        if hints.get('deep', True) and rv != self:
            rv = rv.doit(**hints)
        return rv

    def evalf(self, prec=None, **options):
        # 调用 doit 方法并返回其 evalf 结果
        return self.doit().evalf(prec, **options)

    n = evalf  # type:ignore

    @property
    def variables(self):
        """要进行评估的变量列表"""
        return self._args[1]

    bound_symbols = variables

    @property
    def expr(self):
        """进行替换操作的表达式"""
        return self._args[0]

    @property
    def point(self):
        """变量要替换的值"""
        return self._args[2]

    @property
    def free_symbols(self):
        # 返回表达式的自由符号集合，排除变量列表中的符号，加上替换点的自由符号
        return (self.expr.free_symbols - set(self.variables) |
            set(self.point.free_symbols))
    def expr_free_symbols(self):
        # 发出 SymPy 弃用警告，指出 expr_free_symbols 属性已弃用，请使用 free_symbols 获取表达式的自由符号集合
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        
        # 忽略 SymPy 弃用警告的递归调用
        with ignore_warnings(SymPyDeprecationWarning):
            # 返回表达式的自由符号集合，通过减去变量集合和加上点集合的自由符号集合实现
            return (self.expr.expr_free_symbols - set(self.variables) |
                    set(self.point.expr_free_symbols))

    def __eq__(self, other):
        # 检查是否与另一个 Subs 对象相等，如果不是同一类型则返回 False
        if not isinstance(other, Subs):
            return False
        # 比较两个对象的可散列内容是否相等
        return self._hashable_content() == other._hashable_content()

    def __ne__(self, other):
        # 实现不等于运算符，基于相等运算符的结果取反
        return not(self == other)

    def __hash__(self):
        # 返回对象的哈希值，通过调用父类的哈希方法实现
        return super().__hash__()

    def _hashable_content(self):
        # 返回对象的可散列内容，用于在哈希比较中使用
        return (self._expr.xreplace(self.canonical_variables),
            ) + tuple(ordered([(v, p) for v, p in
            zip(self.variables, self.point) if not self.expr.has(v)]))

    def _eval_subs(self, old, new):
        # 在 Subs 对象中执行变量替换操作
        # Subs 对象的替换操作必须按照变量的顺序进行；Subs 对象的 subs 方法有以下不变条件:
        #    foo.doit().subs(reps) == foo.subs(reps).doit()

        # 将点集合转换为列表
        pt = list(self.point)
        
        # 如果旧变量在当前变量集合中
        if old in self.variables:
            # 如果新值是原子并且不在任何参数中出现，则替换是中性的
            if _atomic(new) == {new} and not any(
                    i.has(new) for i in self.args):
                # 执行单个变量替换操作
                return self.xreplace({old: new})
            
            # 找到旧变量在变量列表中的位置
            i = self.variables.index(old)
            
            # 对于从该位置到变量列表末尾的每个变量，将其点集合中的旧值替换为新值
            for j in range(i, len(self.variables)):
                pt[j] = pt[j]._subs(old, new)
            
            # 创建并返回新的 Subs 对象，其中包含替换后的点集合
            return self.func(self.expr, self.variables, pt)
        
        # 如果旧变量不在当前变量集合中
        v = [i._subs(old, new) for i in self.variables]
        
        # 如果变量列表发生变化，则创建新的 Subs 对象
        if v != list(self.variables):
            return self.func(self.expr, self.variables + (old,), pt + [new])
        
        # 替换表达式中的旧变量为新值，并更新点集合中的旧值为新值
        expr = self.expr._subs(old, new)
        pt = [i._subs(old, new) for i in self.point]
        
        # 创建并返回新的 Subs 对象，其中包含替换后的表达式和点集合
        return self.func(expr, v, pt)
    def _eval_derivative(self, s):
        # 在替换变量上应用导数的链式法则：
        f = self.expr  # 获取表达式
        vp = V, P = self.variables, self.point  # 获取变量和点的元组
        # 计算导数的乘积和替换表达式的偏导数
        val = Add.fromiter(p.diff(s)*Subs(f.diff(v), *vp).doit()
            for v, p in zip(V, P))

        # 获取表达式中的所有自由符号
        efree = f.free_symbols
        # 一些符号如IndexedBase会包括自身和参数作为自由符号
        # 筛选出包含多个自由符号的复合符号
        compound = {i for i in efree if len(i.free_symbols) > 1}
        # 使用虚拟符号隐藏它们，并确定独立的自由符号
        dums = {Dummy() for i in compound}
        masked = f.xreplace(dict(zip(compound, dums)))
        ifree = masked.free_symbols - dums
        # 包括复合符号
        free = ifree | compound
        # 移除已处理的变量
        free -= set(V)
        # 添加剩余复合符号的任何自由符号
        free |= {i for j in free & compound for i in j.free_symbols}
        # 如果s的符号在自由符号中，则还需进一步处理
        if free & s.free_symbols:
            val += Subs(f.diff(s), self.variables, self.point).doit()
        return val

    def _eval_nseries(self, x, n, logx, cdir=0):
        if x in self.point:
            # x是要替换的变量
            apos = self.point.index(x)
            other = self.variables[apos]
        else:
            other = x
        # 计算表达式的n阶级数展开
        arg = self.expr.nseries(other, n=n, logx=logx)
        o = arg.getO()
        terms = Add.make_args(arg.removeO())
        rv = Add(*[self.func(a, *self.args[1:]) for a in terms])
        if o:
            rv += o.subs(other, x)
        return rv

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        if x in self.point:
            # 如果x在点中，找出它的位置并获取对应的变量
            ipos = self.point.index(x)
            xvar = self.variables[ipos]
            return self.expr.as_leading_term(xvar)
        if x in self.variables:
            # 如果x是虚拟变量，表明在替换后将不再存在：
            return self
        # 变量与替换独立：
        return self.expr.as_leading_term(x)
def expand(e, deep=True, modulus=None, power_base=True, power_exp=True,
        mul=True, log=True, multinomial=True, basic=True, **hints):
    r"""
    Expand an expression using methods given as hints.

    Explanation
    ===========

    Hints evaluated unless explicitly set to False are:  ``basic``, ``log``,
    ``multinomial``, ``mul``, ``power_base``, and ``power_exp`` The following
    hints are supported but not applied unless set to True:  ``complex``,
    ``func``, and ``trig``.  In addition, the following meta-hints are
    supported by some or all of the other hints:  ``frac``, ``numer``,
    ``denom``, ``modulus``, and ``force``.  ``deep`` is supported by all
    hints.  Additionally, subclasses of Expr may define their own hints or
    meta-hints.

    The ``basic`` hint is used for any special rewriting of an object that
    should be done automatically (along with the other hints like ``mul``)
    when expand is called. This is a catch-all hint to handle any sort of
    simplification or special function expansion.

    Examples
    ========

    >>> from sympy import symbols, expand, sin, cos
    >>> x, y = symbols('x y')

    >>> expand(sin(x + y))
    sin(x)*cos(y) + cos(x)*sin(y)

    >>> expand((x + y)**3)
    x**3 + 3*x**2*y + 3*x*y**2 + y**3

    >>> expand(cos(x + y), trig=True)
    cos(x + y)

    >>> expand(cos(x + y), deep=False)
    cos(x + y)

    Notes
    =====

    - ``expand`` is not a generic simplification function; it's designed
      specifically for expanding expressions.
    - Use caution when setting deep=False as it may not fully expand nested
      expressions.

    See Also
    ========

    sympy.simplify.simplify
    sympy.simplify.fu
    """
    # 如果输入表达式 e 具有 diff() 方法，直接调用其 diff() 方法进行求导
    if hasattr(e, 'expand'):
        return e.expand(deep=deep, modulus=modulus, power_base=power_base,
                        power_exp=power_exp, mul=mul, log=log,
                        multinomial=multinomial, basic=basic, **hints)
    # 否则，使用 _eval_expand 方法来进行表达式的展开操作
    return e._eval_expand(deep=deep, modulus=modulus, power_base=power_base,
                          power_exp=power_exp, mul=mul, log=log,
                          multinomial=multinomial, basic=basic, **hints)
    expansion that may not be described by the existing hint names. To use
    this hint an object should override the ``_eval_expand_basic`` method.
    Objects may also define their own expand methods, which are not run by
    default.  See the API section below.

    If ``deep`` is set to ``True`` (the default), things like arguments of
    functions are recursively expanded.  Use ``deep=False`` to only expand on
    the top level.

    If the ``force`` hint is used, assumptions about variables will be ignored
    in making the expansion.

    Hints
    =====

    These hints are run by default

    mul
    ---

    Distributes multiplication over addition:

    >>> from sympy import cos, exp, sin
    >>> from sympy.abc import x, y, z
    >>> (y*(x + z)).expand(mul=True)
    x*y + y*z

    multinomial
    -----------

    Expand (x + y + ...)**n where n is a positive integer.

    >>> ((x + y + z)**2).expand(multinomial=True)
    x**2 + 2*x*y + 2*x*z + y**2 + 2*y*z + z**2

    power_exp
    ---------

    Expand addition in exponents into multiplied bases.

    >>> exp(x + y).expand(power_exp=True)
    exp(x)*exp(y)
    >>> (2**(x + y)).expand(power_exp=True)
    2**x*2**y

    power_base
    ----------

    Split powers of multiplied bases.

    This only happens by default if assumptions allow, or if the
    ``force`` meta-hint is used:

    >>> ((x*y)**z).expand(power_base=True)
    (x*y)**z
    >>> ((x*y)**z).expand(power_base=True, force=True)
    x**z*y**z
    >>> ((2*y)**z).expand(power_base=True)
    2**z*y**z

    Note that in some cases where this expansion always holds, SymPy performs
    it automatically:

    >>> (x*y)**2
    x**2*y**2

    log
    ---

    Pull out power of an argument as a coefficient and split logs products
    into sums of logs.

    Note that these only work if the arguments of the log function have the
    proper assumptions--the arguments must be positive and the exponents must
    be real--or else the ``force`` hint must be True:

    >>> from sympy import log, symbols
    >>> log(x**2*y).expand(log=True)
    log(x**2*y)
    >>> log(x**2*y).expand(log=True, force=True)
    2*log(x) + log(y)
    >>> x, y = symbols('x,y', positive=True)
    >>> log(x**2*y).expand(log=True)
    2*log(x) + log(y)

    basic
    -----

    This hint is intended primarily as a way for custom subclasses to enable
    expansion by default.

    These hints are not run by default:

    complex
    -------

    Split an expression into real and imaginary parts.

    >>> x, y = symbols('x,y')
    >>> (x + y).expand(complex=True)
    re(x) + re(y) + I*im(x) + I*im(y)
    >>> cos(x).expand(complex=True)
    -I*sin(re(x))*sinh(im(x)) + cos(re(x))*cosh(im(x))

    Note that this is just a wrapper around ``as_real_imag()``.  Most objects
    that wish to redefine ``_eval_expand_complex()`` should consider
    redefining ``as_real_imag()`` instead.

    func
    ----

    Expand other functions.

    >>> from sympy import gamma
    # 计算 gamma 函数在 x+1 处的展开式，并返回展开结果
    >>> gamma(x + 1).expand(func=True)
    x*gamma(x)
    
    # 执行三角函数的展开操作
    
    # 对 cos(x + y) 进行三角函数展开
    >>> cos(x + y).expand(trig=True)
    -sin(x)*sin(y) + cos(x)*cos(y)
    
    # 对 sin(2*x) 进行三角函数展开
    >>> sin(2*x).expand(trig=True)
    2*sin(x)*cos(x)
    
    # 注意，“sin(n*x)” 和 “cos(n*x)” 可以用 “sin(x)” 和 “cos(x)” 的形式表示，但由于恒等式 sin^2(x) + cos^2(x) = 1 的存在，这些形式不唯一。
    # 当前的实现使用了 Chebyshev 多项式的形式，但这可能会改变。参见 MathWorld 的文章 <https://mathworld.wolfram.com/Multiple-AngleFormulas.html> 获取更多信息。
    
    # 注释结束
    # 提示的应用顺序是任意的，但是一致的（在当前实现中，它们按字母顺序应用，除了 multinomial 比 mul 在前，但这可能会改变）。
    # 由于这个原因，一些提示可能会阻止其他提示的扩展，如果它们首先应用的话。
    # 例如，`mul` 可能会分配乘法并阻止 `log` 和 `power_base` 扩展它们。
    # 另外，如果 `mul` 在 `multinomial` 之前应用，表达式可能不会完全分布。
    # 解决方法是使用各种 `expand_hint` 辅助函数，或者在调用此函数时使用 `hint=False` 微调控制应用哪些提示。
    # 这里有一些示例：

    # 导入 sympy 中的扩展函数
    >>> from sympy import expand, expand_mul, expand_power_base
    # 定义正数符号
    >>> x, y, z = symbols('x,y,z', positive=True)

    # 扩展 `log` 函数被应用于 `mul` 之前的示例
    >>> expand(log(x*(y + z)))
    log(x) + log(y + z)

    # 这里我们看到 `log` 在 `mul` 之前被应用。要获取分配乘法后的表达形式，可以使用以下任一方法：
    >>> expand_mul(log(x*(y + z)))
    log(x*y + x*z)
    >>> expand(log(x*(y + z)), log=False)
    log(x*y + x*z)

    # 类似的情况也可能发生在 `power_base` 提示中：
    >>> expand((x*(y + z))**x)
    (x*y + x*z)**x

    # 要获取 `power_base` 扩展后的形式，可以使用以下任一方法：
    >>> expand((x*(y + z))**x, mul=False)
    x**x*(y + z)**x
    >>> expand_power_base((x*(y + z))**x)
    x**x*(y + z)**x

    # 分子表达式的部分可以被目标化：
    >>> expand((x + y)*y/x)
    y + y**2/x

    # 使用 `frac=True` 可以处理有理表达式的分子部分：
    >>> expand((x + y)*y/x/(x + 1), frac=True)
    (x*y + y**2)/(x**2 + x)
    # 使用 `numer=True` 可以处理有理表达式的分子部分：
    >>> expand((x + y)*y/x/(x + 1), numer=True)
    (x*y + y**2)/(x*(x + 1))
    # 使用 `denom=True` 可以处理有理表达式的分母部分：
    >>> expand((x + y)*y/x/(x + 1), denom=True)
    y*(x + y)/(x**2 + x)

    # `modulus` 元提示可以用来减少表达式扩展后的系数：
    >>> expand((3*x + 1)**2)
    9*x**2 + 6*x + 1
    >>> expand((3*x + 1)**2, modulus=5)
    4*x**2 + x + 1

    # `expand()` 函数和 `.expand()` 方法都可以使用，两者是等效的：
    >>> expand((x + 1)**2)
    x**2 + 2*x + 1
    >>> ((x + 1)**2).expand()
    x**2 + 2*x + 1

    # API
    # ===

    # 对象可以通过定义 `_eval_expand_hint()` 来定义它们自己的扩展提示。
    # 函数应该采用以下形式：
    # def _eval_expand_hint(self, **hints):
    # 只将方法应用于顶层表达式
    # ...

    # 请参阅下面的示例。只有当 `hint` 适用于特定对象时，对象才应该定义 `_eval_expand_hint()` 方法。
    # 在 Expr 中定义的通用 `_eval_expand_hint()` 方法将处理无操作的情况。

    # 每个提示应负责扩展该提示而不是其他内容。
    '''
    扩展给定的对象表达式，使用给定的扩展提示。
    
    'power_base'、'power_exp' 和 'mul' 是扩展方法的元提示，控制如何应用不同的扩展方法。
    '''
    # 不要修改此处；修改 Expr.expand 方法
    hints['power_base'] = power_base
    hints['power_exp'] = power_exp
    hints['mul'] = mul
    # 将 'log' 键映射到 log 变量，'multinomial' 键映射到 multinomial 变量，'basic' 键映射到 basic 变量，
    hints['log'] = log
    hints['multinomial'] = multinomial
    hints['basic'] = basic
    # 将表达式 e 转换为符号表达式，并进行展开操作，可选地使用深度展开、取模等参数
    return sympify(e).expand(deep=deep, modulus=modulus, **hints)
# This is a special application of two hints
# 这是两个提示的特殊应用示例

def _mexpand(expr, recursive=False):
    # expand multinomials and then expand products; this may not always
    # be sufficient to give a fully expanded expression (see
    # test_issue_8247_8354 in test_arit)
    # 展开多项式，然后展开乘积；这可能不足以给出完全展开的表达式（参见测试 test_issue_8247_8354 在 test_arit 中）
    if expr is None:
        return
    # 记录上一次的表达式
    was = None
    # 循环直到表达式不再改变
    while was != expr:
        # 记录当前表达式，并展开乘法和多项式
        was, expr = expr, expand_mul(expand_multinomial(expr))
        # 如果不递归展开，则退出循环
        if not recursive:
            break
    # 返回展开后的表达式
    return expr


# These are simple wrappers around single hints.
# 下面是几个简单的单一提示的包装器函数

def expand_mul(expr, deep=True):
    """
    Wrapper around expand that only uses the mul hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_mul, exp, log
    >>> x, y = symbols('x,y', positive=True)
    >>> expand_mul(exp(x+y)*(x+y)*log(x*y**2))
    x*exp(x + y)*log(x*y**2) + y*exp(x + y)*log(x*y**2)

    """
    # 将表达式转换为 sympy 的表达式对象，并调用 expand 方法进行展开
    return sympify(expr).expand(deep=deep, mul=True, power_exp=False,
    power_base=False, basic=False, multinomial=False, log=False)


def expand_multinomial(expr, deep=True):
    """
    Wrapper around expand that only uses the multinomial hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_multinomial, exp
    >>> x, y = symbols('x y', positive=True)
    >>> expand_multinomial((x + exp(x + 1))**2)
    x**2 + 2*x*exp(x + 1) + exp(2*x + 2)

    """
    # 将表达式转换为 sympy 的表达式对象，并调用 expand 方法进行展开
    return sympify(expr).expand(deep=deep, mul=False, power_exp=False,
    power_base=False, basic=False, multinomial=True, log=False)


def expand_log(expr, deep=True, force=False, factor=False):
    """
    Wrapper around expand that only uses the log hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_log, exp, log
    >>> x, y = symbols('x,y', positive=True)
    >>> expand_log(exp(x+y)*(x+y)*log(x*y**2))
    (x + y)*(log(x) + 2*log(y))*exp(x + y)

    """
    # 导入 log 函数和 fraction 函数
    from sympy.functions.elementary.exponential import log
    from sympy.simplify.radsimp import fraction
    # 如果 factor 是 False，则执行以下代码块
    if factor is False:
        # 定义内部函数 _handleMul，用于处理表达式中的乘法项
        def _handleMul(x):
            # 分数化简，获取分子和分母
            n, d = fraction(x)
            # 从分子中提取所有是对数的简单情况，且对数的底数是整数
            n = [i for i in n.atoms(log) if i.args[0].is_Integer]
            # 从分母中提取所有是对数的简单情况，且对数的底数是整数
            d = [i for i in d.atoms(log) if i.args[0].is_Integer]
            # 如果找到一个分子和一个分母
            if len(n) == 1 and len(d) == 1:
                n = n[0]
                d = d[0]
                # 导入 multiplicity 函数
                from sympy import multiplicity
                # 计算底数 d.args[0] 在 n.args[0] 中的重数
                m = multiplicity(d.args[0], n.args[0])
                # 如果重数 m 存在
                if m:
                    # 计算结果 r
                    r = m + log(n.args[0] // d.args[0]**m) / d
                    # 替换 n 为 d*r
                    x = x.subs(n, d*r)
            # 对表达式进行展开并化简对数
            x1 = expand_mul(expand_log(x, deep=deep, force=force, factor=True))
            # 如果展开后的表达式中对数的个数不超过原始表达式中对数的个数，则返回展开后的表达式
            if x1.count(log) <= x.count(log):
                return x1
            # 否则返回原始表达式
            return x

        # 使用 _handleMul 函数处理表达式中的乘法项
        expr = expr.replace(
            # 匿名函数，检查是否是乘法并且所有乘法项中至少有一个是有理数对数
            lambda x: x.is_Mul and all(any(isinstance(i, log) and i.args[0].is_Rational
                                           for i in Mul.make_args(j)) for j in x.as_numer_denom()),
            _handleMul)

    # 将 expr 转换为符号表达式，并展开表达式中的对数项
    return sympify(expr).expand(deep=deep, log=True, mul=False,
                                power_exp=False, power_base=False, multinomial=False,
                                basic=False, force=force, factor=factor)
# 使用 sympy 库对表达式进行符号化处理
def expand_func(expr, deep=True):
    """
    Wrapper around expand that only uses the func hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_func, gamma
    >>> from sympy.abc import x
    >>> expand_func(gamma(x + 2))
    x*(x + 1)*gamma(x)

    """
    return sympify(expr).expand(deep=deep, func=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


# 使用 sympy 库对表达式进行符号化处理，仅使用三角函数提示进行展开
def expand_trig(expr, deep=True):
    """
    Wrapper around expand that only uses the trig hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_trig, sin
    >>> from sympy.abc import x, y
    >>> expand_trig(sin(x+y)*(x+y))
    (x + y)*(sin(x)*cos(y) + sin(y)*cos(x))

    """
    return sympify(expr).expand(deep=deep, trig=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


# 使用 sympy 库对表达式进行符号化处理，仅使用复数提示进行展开
def expand_complex(expr, deep=True):
    """
    Wrapper around expand that only uses the complex hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_complex, exp, sqrt, I
    >>> from sympy.abc import z
    >>> expand_complex(exp(z))
    I*exp(re(z))*sin(im(z)) + exp(re(z))*cos(im(z))
    >>> expand_complex(sqrt(I))
    sqrt(2)/2 + sqrt(2)*I/2

    See Also
    ========

    sympy.core.expr.Expr.as_real_imag
    """
    return sympify(expr).expand(deep=deep, complex=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


# 使用 sympy 库对表达式进行符号化处理，仅使用底数幂指数提示进行展开
def expand_power_base(expr, deep=True, force=False):
    """
    Wrapper around expand that only uses the power_base hint.

    A wrapper to expand(power_base=True) which separates a power with a base
    that is a Mul into a product of powers, without performing any other
    expansions, provided that assumptions about the power's base and exponent
    allow.

    deep=False (default is True) will only apply to the top-level expression.

    force=True (default is False) will cause the expansion to ignore
    assumptions about the base and exponent. When False, the expansion will
    only happen if the base is non-negative or the exponent is an integer.

    >>> from sympy.abc import x, y, z
    >>> from sympy import expand_power_base, sin, cos, exp, Symbol

    >>> (x*y)**2
    x**2*y**2

    >>> (2*x)**y
    (2*x)**y
    >>> expand_power_base(_)
    2**y*x**y

    >>> expand_power_base((x*y)**z)
    (x*y)**z
    >>> expand_power_base((x*y)**z, force=True)
    x**z*y**z
    >>> expand_power_base(sin((x*y)**z), deep=False)
    sin((x*y)**z)
    >>> expand_power_base(sin((x*y)**z), force=True)
    sin(x**z*y**z)

    >>> expand_power_base((2*sin(x))**y + (2*cos(x))**y)
    2**y*sin(x)**y + 2**y*cos(x)**y

    >>> expand_power_base((2*exp(y))**x)
    2**x*exp(y)**x

    >>> expand_power_base((2*cos(x))**y)
    2**y*cos(x)**y

    """
    return sympify(expr).expand(deep=deep, power_base=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)
    # 将输入的表达式转换为符号表达式，然后进行展开
    return sympify(expr).expand(
        deep=deep,  # 是否深度展开，这里设为 False
        log=False,  # 是否展开对数函数，这里设为 False
        mul=False,  # 是否展开乘法，这里设为 False
        power_exp=False,  # 是否展开指数函数的指数部分，这里设为 False
        power_base=True,  # 是否展开指数函数的底数部分，这里设为 True
        multinomial=False,  # 是否展开多项式，这里设为 False
        basic=False,  # 是否展开基本的表达式，这里设为 False
        force=force  # 是否强制展开，根据参数 force 决定
    )
# 定义一个函数，用于展开表达式中的幂次表达式，只使用 power_exp 提示。
# 详细说明请参见 expand 函数的文档字符串。

def expand_power_exp(expr, deep=True):
    """
    Wrapper around expand that only uses the power_exp hint.

    See the expand docstring for more information.

    Examples
    ========

    >>> from sympy import expand_power_exp, Symbol
    >>> from sympy.abc import x, y
    >>> expand_power_exp(3**(y + 2))
    9*3**y
    >>> expand_power_exp(x**(y + 2))
    x**(y + 2)

    If ``x = 0`` the value of the expression depends on the
    value of ``y``; if the expression were expanded the result
    would be 0. So expansion is only done if ``x != 0``:

    >>> expand_power_exp(Symbol('x', zero=False)**(y + 2))
    x**2*x**y
    """
    return sympify(expr).expand(deep=deep, complex=False, basic=False,
                                log=False, mul=False, power_exp=True, power_base=False, multinomial=False)



# 定义一个函数，用于统计表达式中的操作次数或操作类型。

def count_ops(expr, visual=False):
    """
    Return a representation (integer or expression) of the operations in expr.

    Parameters
    ==========

    expr : Expr
        If expr is an iterable, the sum of the op counts of the
        items will be returned.

    visual : bool, optional
        If ``False`` (default) then the sum of the coefficients of the
        visual expression will be returned.
        If ``True`` then the number of each type of operation is shown
        with the core class types (or their virtual equivalent) multiplied by the
        number of times they occur.

    Examples
    ========

    >>> from sympy.abc import a, b, x, y
    >>> from sympy import sin, count_ops

    Although there is not a SUB object, minus signs are interpreted as
    either negations or subtractions:

    >>> (x - y).count_ops(visual=True)
    SUB
    >>> (-x).count_ops(visual=True)
    NEG

    Here, there are two Adds and a Pow:

    >>> (1 + a + b**2).count_ops(visual=True)
    2*ADD + POW

    In the following, an Add, Mul, Pow and two functions:

    >>> (sin(x)*x + sin(x)**2).count_ops(visual=True)
    ADD + MUL + POW + 2*SIN

    for a total of 5:

    >>> (sin(x)*x + sin(x)**2).count_ops(visual=False)
    5

    Note that "what you type" is not always what you get. The expression
    1/x/y is translated by sympy into 1/(x*y) so it gives a DIV and MUL rather
    than two DIVs:

    >>> (1/x/y).count_ops(visual=True)
    DIV + MUL

    The visual option can be used to demonstrate the difference in
    operations for expressions in different forms. Here, the Horner
    representation is compared with the expanded form of a polynomial:

    >>> eq=x*(1 + x*(2 + x*(3 + x)))
    >>> count_ops(eq.expand(), visual=True) - count_ops(eq, visual=True)
    -MUL + 3*POW

    The count_ops function also handles iterables:

    >>> count_ops([x, sin(x), None, True, x + 2], visual=False)
    2
    >>> count_ops([x, sin(x), None, True, x + 2], visual=True)
    ADD + SIN
    >>> count_ops({x: sin(x), x + 2: y + 1}, visual=True)
    2*ADD + SIN

    """
    from .relational import Relational
    from sympy.concrete.summations import Sum
    # 从sympy.integrals.integrals模块中导入Integral类
    from sympy.integrals.integrals import Integral
    # 从sympy.logic.boolalg模块中导入BooleanFunction类
    from sympy.logic.boolalg import BooleanFunction
    # 从sympy.simplify.radsimp模块中导入fraction函数
    from sympy.simplify.radsimp import fraction

    # 将表达式expr转换为sympy表达式对象
    expr = sympify(expr)
    # 如果expr是字典类型
    elif isinstance(expr, Dict):
        # 递归计算字典中每对键值对应的操作数
        ops = [count_ops(k, visual=visual) +
               count_ops(v, visual=visual) for k, v in expr.items()]
    # 如果expr是可迭代对象
    elif iterable(expr):
        # 对expr中的每个元素递归计算操作数
        ops = [count_ops(i, visual=visual) for i in expr]
    # 如果expr是关系表达式或布尔函数
    elif isinstance(expr, (Relational, BooleanFunction)):
        ops = []
        # 遍历expr的每个参数，递归计算操作数
        for arg in expr.args:
            ops.append(count_ops(arg, visual=True))
        # 创建一个符号对象o，其名字为func_name(expr, short=True)的大写形式
        o = Symbol(func_name(expr, short=True).upper())
        ops.append(o)
    # 如果expr不是Basic类的实例
    elif not isinstance(expr, Basic):
        ops = []
    else:  # it's Basic not isinstance(expr, Expr):
        # 如果expr是Basic类的实例，但不是Expr类的实例，则抛出类型错误
        if not isinstance(expr, Basic):
            raise TypeError("Invalid type of expr")
        else:
            ops = []
            args = [expr]
            # 使用深度优先搜索遍历expr的每个参数，递归计算操作数
            while args:
                a = args.pop()

                if a.args:
                    # 创建一个符号对象o，其名字为a的类型名称的大写形式
                    o = Symbol(type(a).__name__.upper())
                    # 如果a是布尔类型，将o乘以a的参数个数-1添加到操作数列表中
                    if a.is_Boolean:
                        ops.append(o*(len(a.args)-1))
                    else:
                        ops.append(o)
                    args.extend(a.args)

    # 如果ops为空列表
    if not ops:
        # 如果visual为True，返回sympy的零常量S.Zero
        if visual:
            return S.Zero
        # 否则返回整数0
        return 0

    # 将ops中的所有操作数求和，得到一个加法表达式ops
    ops = Add(*ops)

    # 如果visual为True，返回加法表达式ops
    if visual:
        return ops

    # 如果ops是一个数字
    if ops.is_Number:
        # 返回ops的整数形式
        return int(ops)

    # 否则，将加法表达式ops的所有参数中的第一个（如果存在）转换为整数并求和，返回结果
    return sum(int((a.args or [1])[0]) for a in Add.make_args(ops))
def nfloat(expr, n=15, exponent=False, dkeys=False):
    """Make all Rationals in expr Floats except those in exponents
    (unless the exponents flag is set to True) and those in undefined
    functions. When processing dictionaries, do not modify the keys
    unless ``dkeys=True``.

    Examples
    ========

    >>> from sympy import nfloat, cos, pi, sqrt
    >>> from sympy.abc import x, y
    >>> nfloat(x**4 + x/2 + cos(pi/3) + 1 + sqrt(y))
    x**4 + 0.5*x + sqrt(y) + 1.5
    >>> nfloat(x**4 + sqrt(y), exponent=True)
    x**4.0 + y**0.5

    Container types are not modified:

    >>> type(nfloat((1, 2))) is tuple
    True
    """
    from sympy.matrices.matrixbase import MatrixBase  # 导入 MatrixBase 类

    kw = {"n": n, "exponent": exponent, "dkeys": dkeys}  # 定义关键字参数字典

    if isinstance(expr, MatrixBase):  # 如果表达式是 MatrixBase 类型
        return expr.applyfunc(lambda e: nfloat(e, **kw))  # 对表达式中的每个元素应用 nfloat 函数

    # handling of iterable containers
    if iterable(expr, exclude=str):  # 如果表达式是可迭代的容器（排除字符串）
        if isinstance(expr, (dict, Dict)):  # 如果表达式是字典类型
            if dkeys:  # 如果 dkeys=True
                args = [tuple((nfloat(i, **kw) for i in a))  # 将字典的值转换为浮点数，保持键不变
                        for a in expr.items()]
            else:  # 如果 dkeys=False
                args = [(k, nfloat(v, **kw)) for k, v in expr.items()]  # 将字典的键和值分别转换为浮点数
            if isinstance(expr, dict):
                return type(expr)(args)  # 返回相同类型的字典
            else:
                return expr.func(*args)  # 返回使用浮点数处理后的函数对象
        elif isinstance(expr, Basic):  # 如果表达式是 Basic 类型
            return expr.func(*[nfloat(a, **kw) for a in expr.args])  # 返回使用浮点数处理后的函数对象
        return type(expr)([nfloat(a, **kw) for a in expr])  # 返回使用浮点数处理后的相同类型的容器

    rv = sympify(expr)  # 将表达式转换为 Sympy 的表达式对象

    if rv.is_Number:  # 如果 rv 是数值类型
        return Float(rv, n)  # 返回指定精度的浮点数对象
    elif rv.is_number:  # 如果 rv 是数值
        # evalf 不总是设置精度
        rv = rv.n(n)
        if rv.is_Number:
            rv = Float(rv.n(n), n)
        else:
            pass  # pure_complex(rv) is likely True
        return rv
    elif rv.is_Atom:  # 如果 rv 是原子类型
        return rv
    elif rv.is_Relational:  # 如果 rv 是关系类型
        args_nfloat = (nfloat(arg, **kw) for arg in rv.args)
        return rv.func(*args_nfloat)

    # watch out for RootOf instances that don't like to have
    # their exponents replaced with Dummies and also sometimes have
    # problems with evaluating at low precision (issue 6393)
    from sympy.polys.rootoftools import RootOf  # 导入 RootOf 类
    rv = rv.xreplace({ro: ro.n(n) for ro in rv.atoms(RootOf)})  # 将 RootOf 实例中的表达式替换为指定精度的浮点数

    from .power import Pow  # 导入 Pow 类
    if not exponent:  # 如果不处理指数
        reps = [(p, Pow(p.base, Dummy())) for p in rv.atoms(Pow)]  # 将 Pow 实例的指数替换为 Dummy()
        rv = rv.xreplace(dict(reps))  # 替换表达式中的 Pow 实例
    rv = rv.n(n)  # 将表达式的数值部分转换为指定精度的浮点数
    if not exponent:  # 如果不处理指数
        rv = rv.xreplace({d.exp: p.exp for p, d in reps})  # 恢复替换的指数
    else:
        # Pow._eval_evalf special cases Integer exponents so if
        # exponent is suppose to be handled we have to do so here
        rv = rv.xreplace(Transform(
            lambda x: Pow(x.base, Float(x.exp, n)),
            lambda x: x.is_Pow and x.exp.is_Integer))  # 处理 Pow 实例的整数指数

    return rv.xreplace(Transform(
        lambda x: x.func(*nfloat(x.args, n, exponent)),
        lambda x: isinstance(x, Function) and not isinstance(x, AppliedUndef)))  # 处理函数对象，不处理未定义的函数


from .symbol import Dummy, Symbol  # 导入 Dummy 和 Symbol 类
```