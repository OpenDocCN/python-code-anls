# `D:\src\scipysrc\sympy\sympy\core\operations.py`

```
from __future__ import annotations
from operator import attrgetter  # 导入attrgetter函数，用于获取对象的属性
from collections import defaultdict  # 导入defaultdict类，用于创建默认值为列表的字典

from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入sympy_deprecation_warning异常，用于发出SymPy的弃用警告

from .sympify import _sympify as _sympify_, sympify  # 导入_sympify_和sympify函数，用于将对象转换为SymPy表达式
from .basic import Basic  # 导入Basic类，作为基本类的基类
from .cache import cacheit  # 导入cacheit装饰器函数，用于缓存函数调用结果
from .sorting import ordered  # 导入ordered函数，用于对对象进行排序
from .logic import fuzzy_and  # 导入fuzzy_and函数，用于模糊逻辑的与操作
from .parameters import global_parameters  # 导入global_parameters对象，用于存储全局参数
from sympy.utilities.iterables import sift  # 导入sift函数，用于按条件分离对象
from sympy.multipledispatch.dispatcher import (Dispatcher,  # 导入Dispatcher类，用于多分派方法的调度
    ambiguity_register_error_ignore_dup,  # 导入ambiguity_register_error_ignore_dup函数，用于注册重复方法调用的错误
    str_signature, RaiseNotImplementedError)  # 导入str_signature函数和RaiseNotImplementedError异常

class AssocOp(Basic):
    """ Associative operations, can separate noncommutative and
    commutative parts.

    (a op b) op c == a op (b op c) == a op b op c.

    Base class for Add and Mul.

    This is an abstract base class, concrete derived classes must define
    the attribute `identity`.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Parameters
    ==========

    *args :
        Arguments which are operated

    evaluate : bool, optional
        Evaluate the operation. If not passed, refer to ``global_parameters.evaluate``.
    """

    # for performance reason, we don't let is_commutative go to assumptions,
    # and keep it right here
    __slots__: tuple[str, ...] = ('is_commutative',)  # 定义__slots__属性，用于限制类的实例能够拥有的属性，以提升性能

    _args_type: type[Basic] | None = None  # 定义_args_type类变量，指示操作数类型为Basic类或其子类，若为None则无特定要求

    @cacheit
    def __new__(cls, *args, evaluate=None, _sympify=True):
        # Allow faster processing by passing ``_sympify=False``, if all arguments
        # are already sympified.
        if _sympify:
            args = list(map(_sympify_, args))  # 如果_sympify为True，则将args中的每个元素转换为SymPy表达式

        # Disallow non-Expr args in Add/Mul
        typ = cls._args_type
        if typ is not None:
            from .relational import Relational
            if any(isinstance(arg, Relational) for arg in args):  # 如果args中存在Relational类的实例，则抛出TypeError异常
                raise TypeError("Relational cannot be used in %s" % cls.__name__)

            # This should raise TypeError once deprecation period is over:
            for arg in args:
                if not isinstance(arg, typ):  # 如果args中有任何一个元素不是typ类型，则发出SymPy的弃用警告
                    sympy_deprecation_warning(
                        f"""

Using non-Expr arguments in {cls.__name__} is deprecated (in this case, one of
the arguments has type {type(arg).__name__!r}).

If you really did intend to use a multiplication or addition operation with
    @deprecated(
                        useinstead="Use the '*' or '+' operators for multiplication or addition, respectively.",
                        reason="Use of positional arguments without an enclosing expression is deprecated.",
                        deprecated_since_version="1.7",
                        active_deprecations_target="non-expr-args-deprecated",
                        stacklevel=4,
                    )
    # 标记这个函数为被废弃的，提供了替代方法和废弃原因等信息
    def __new__(cls, *args, **kwargs):
        # 如果 evaluate 参数为 None，则使用全局参数中的 evaluate 值
        if evaluate is None:
            evaluate = global_parameters.evaluate
        # 如果 evaluate 为 False，则按照参数创建对象并执行构造后处理器
        if not evaluate:
            obj = cls._from_args(args)
            obj = cls._exec_constructor_postprocessors(obj)
            return obj

        # 过滤掉参数列表中等于类的 identity 属性的元素
        args = [a for a in args if a is not cls.identity]

        # 如果参数个数为 0，则返回类的 identity 属性
        if len(args) == 0:
            return cls.identity
        # 如果参数个数为 1，则直接返回第一个参数
        if len(args) == 1:
            return args[0]

        # 将参数列表按照 canonical 和 non-canonical 部分分离，并获取 order_symbols
        c_part, nc_part, order_symbols = cls.flatten(args)
        is_commutative = not nc_part
        # 使用 c_part 和 nc_part 创建一个新的对象，并设置其是否可交换属性
        obj = cls._from_args(c_part + nc_part, is_commutative)
        # 执行构造后处理器
        obj = cls._exec_constructor_postprocessors(obj)

        # 如果 order_symbols 不为 None，则创建 Order 对象并返回
        if order_symbols is not None:
            from sympy.series.order import Order
            return Order(obj, *order_symbols)
        # 否则直接返回创建的对象
        return obj

    @classmethod
    # 从已处理的参数 args 创建新的类实例
    def _from_args(cls, args, is_commutative=None):
        """Create new instance with already-processed args.
        If the args are not in canonical order, then a non-canonical
        result will be returned, so use with caution. The order of
        args may change if the sign of the args is changed."""
        # 如果参数列表长度为 0，则返回类的 identity 属性
        if len(args) == 0:
            return cls.identity
        # 如果参数列表长度为 1，则直接返回第一个参数
        elif len(args) == 1:
            return args[0]

        # 使用父类的 __new__ 方法创建一个新的对象实例
        obj = super().__new__(cls, *args)
        # 如果未提供 is_commutative 参数，则根据 args 中各元素的 is_commutative 属性决定
        if is_commutative is None:
            is_commutative = fuzzy_and(a.is_commutative for a in args)
        # 设置新创建对象的 is_commutative 属性
        obj.is_commutative = is_commutative
        return obj
    # 创建一个新的实例，使用调用者提供的参数（args）和关键字参数（kwargs），如果args为空则返回自身类的标识
    def _new_rawargs(self, *args, reeval=True, **kwargs):
        """Create new instance of own class with args exactly as provided by
        caller but returning the self class identity if args is empty.

        Examples
        ========

           This is handy when we want to optimize things, e.g.

               >>> from sympy import Mul, S
               >>> from sympy.abc import x, y
               >>> e = Mul(3, x, y)
               >>> e.args
               (3, x, y)
               >>> Mul(*e.args[1:])
               x*y
               >>> e._new_rawargs(*e.args[1:])  # the same as above, but faster
               x*y

           Note: use this with caution. There is no checking of arguments at
           all. This is best used when you are rebuilding an Add or Mul after
           simply removing one or more args. If, for example, modifications,
           result in extra 1s being inserted they will show up in the result:

               >>> m = (x*y)._new_rawargs(S.One, x); m
               1*x
               >>> m == x
               False
               >>> m.is_Mul
               True

           Another issue to be aware of is that the commutativity of the result
           is based on the commutativity of self. If you are rebuilding the
           terms that came from a commutative object then there will be no
           problem, but if self was non-commutative then what you are
           rebuilding may now be commutative.

           Although this routine tries to do as little as possible with the
           input, getting the commutativity right is important, so this level
           of safety is enforced: commutativity will always be recomputed if
           self is non-commutative and kwarg `reeval=False` has not been
           passed.
        """
        # 如果 reeval 为 True 并且 self 的 commutative 属性为 False，则设置 is_commutative 为 None，否则使用 self 的 commutative 属性
        if reeval and self.is_commutative is False:
            is_commutative = None
        else:
            is_commutative = self.is_commutative
        # 使用给定的 args 和计算得到的 is_commutative 调用 _from_args 方法，返回新实例
        return self._from_args(args, is_commutative)

    @classmethod
    def flatten(cls, seq):
        """Return seq so that none of the elements are of type `cls`. This is
        the vanilla routine that will be used if a class derived from AssocOp
        does not define its own flatten routine."""
        # 应用关联性，不使用交换性属性
        new_seq = []
        # 反向遍历 seq 列表
        while seq:
            o = seq.pop()  # 弹出列表最后一个元素
            # 如果 o 的类与 cls 类型相同，则将 o 的 args 扩展到 seq 中
            if o.__class__ is cls:  # classes must match exactly
                seq.extend(o.args)
            else:
                new_seq.append(o)  # 否则将 o 添加到 new_seq 中
        new_seq.reverse()  # 反转 new_seq 列表

        # 返回三个空列表（c_part, nc_part, order_symbols）和反转后的 new_seq 列表
        return [], new_seq, None
    # 返回一个帮助函数，用于检查表达式中是否包含子表达式的函数 .has()。
    # 它通过对类似节点的参数集合进行检查来判断表达式中是否包含子表达式。
    # 例如，对于表达式 x + 1 in x + y + 1，检查 {x, 1} 是否是 {x, y, 1} 的子集。
    def _has_matcher(self):
        def _ncsplit(expr):
            # 这里的函数与 args_cnc 不同，不假设 expr 是乘法操作，因此处理参数并始终返回一个集合。
            # 将表达式的参数分为可交换部分和非可交换部分。
            cpart, ncpart = sift(expr.args,
                lambda arg: arg.is_commutative is True, binary=True)
            return set(cpart), ncpart

        # 调用 _ncsplit 函数，分别获取可交换部分 c 和非可交换部分 nc。
        c, nc = _ncsplit(self)
        # 获取 self 对象的类
        cls = self.__class__

        def is_in(expr):
            # 如果 expr 是当前类的实例
            if isinstance(expr, cls):
                # 如果 expr 等于 self，返回 True
                if expr == self:
                    return True
                # 否则，调用 _ncsplit 函数获取 expr 的可交换部分 _c 和非可交换部分 _nc。
                _c, _nc = _ncsplit(expr)
                # 检查当前对象的可交换部分 c 是否是 expr 的可交换部分 _c 的子集
                if (c & _c) == c:
                    # 如果当前对象没有非可交换部分 nc，则返回 True
                    if not nc:
                        return True
                    # 如果当前对象的非可交换部分 nc 长度小于等于 expr 的非可交换部分 _nc 的长度
                    elif len(nc) <= len(_nc):
                        # 遍历 _nc 中长度为 len(nc) 的子序列，查找是否存在与 nc 相等的子序列
                        for i in range(len(_nc) - len(nc) + 1):
                            if _nc[i:i + len(nc)] == nc:
                                return True
            # 如果以上条件都不满足，则返回 False
            return False
        # 返回内部定义的 is_in 函数
        return is_in
    # 定义一个方法 `_eval_evalf`，用于对表达式进行评估
    def _eval_evalf(self, prec):
        """
        Evaluate the parts of self that are numbers; if the whole thing
        was a number with no functions it would have been evaluated, but
        it wasn't so we must judiciously extract the numbers and reconstruct
        the object. This is *not* simply replacing numbers with evaluated numbers.
        Numbers should be handled in the largest pure-number expression as possible.
        So the code below separates ``self`` into number and non-number parts and
        evaluates the number parts and walks the args of the non-number part recursively
        (doing the same thing).
        """
        # 导入需要的类
        from .add import Add
        from .mul import Mul
        from .symbol import Symbol
        from .function import AppliedUndef
        
        # 如果 self 是 Mul 或 Add 类的实例
        if isinstance(self, (Mul, Add)):
            # 将 self 分解为一个符号和一个尾部部分
            x, tail = self.as_independent(Symbol, AppliedUndef)
            # 如果 x 是一个关联操作函数，并且这里的 _evalf 将会调用 _eval_evalf（当前方法）
            # 所以我们必须打破递归
            if not (tail is self.identity or
                    isinstance(x, AssocOp) and x.is_Function or
                    x is self.identity and isinstance(tail, AssocOp)):
                # 这里，我们有一个数字，因此我们用给定的精度调用 _evalf
                x = x._evalf(prec) if x is not self.identity else self.identity
                args = []
                tail_args = tuple(self.func.make_args(tail))
                for a in tail_args:
                    # 在这里调用 _eval_evalf，因为我们不知道我们正在处理什么，
                    # 所有其他 _eval_evalf 应该做同样的事情（即，采用二进制精度，并找到可评估的参数）
                    newa = a._eval_evalf(prec)
                    if newa is None:
                        args.append(a)
                    else:
                        args.append(newa)
                return self.func(x, *args)

        # 这里与上面相同，但是没有纯数字参数要处理
        args = []
        for a in self.args:
            newa = a._eval_evalf(prec)
            if newa is None:
                args.append(a)
            else:
                args.append(newa)
        return self.func(*args)
    # 返回一个元素序列 `args`，使得 cls(*args) == expr
    def make_args(cls, expr):
        """
        Return a sequence of elements `args` such that cls(*args) == expr

        Examples
        ========

        >>> from sympy import Symbol, Mul, Add
        >>> x, y = map(Symbol, 'xy')

        >>> Mul.make_args(x*y)
        (x, y)
        >>> Add.make_args(x*y)
        (x*y,)
        >>> set(Add.make_args(x*y + y)) == set([y, x*y])
        True

        """
        # 如果 expr 是 cls 的实例，则返回 expr 的 args 属性（元素序列）
        if isinstance(expr, cls):
            return expr.args
        else:
            # 否则，将 expr 转换为 sympy 符号并返回其单元素元组
            return (sympify(expr),)

    # 对象方法，根据给定的提示参数进行计算
    def doit(self, **hints):
        # 如果提示参数中包含 'deep'，并且其值为 True，则深度处理每个项
        if hints.get('deep', True):
            # 对 self.args 中的每个项递归调用 doit 方法，并构建处理后的 terms 列表
            terms = [term.doit(**hints) for term in self.args]
        else:
            # 否则，直接使用 self.args 作为 terms
            terms = self.args
        # 将处理后的 terms 应用于当前对象的函数，并强制评估结果
        return self.func(*terms, evaluate=True)
class ShortCircuit(Exception):
    pass



class LatticeOp(AssocOp):
    """
    Join/meet operations of an algebraic lattice[1].

    Explanation
    ===========

    These binary operations are associative (op(op(a, b), c) = op(a, op(b, c))),
    commutative (op(a, b) = op(b, a)) and idempotent (op(a, a) = op(a) = a).
    Common examples are AND, OR, Union, Intersection, max or min. They have an
    identity element (op(identity, a) = a) and an absorbing element
    conventionally called zero (op(zero, a) = zero).

    This is an abstract base class, concrete derived classes must declare
    attributes zero and identity. All defining properties are then respected.

    Examples
    ========

    >>> from sympy import Integer
    >>> from sympy.core.operations import LatticeOp
    >>> class my_join(LatticeOp):
    ...     zero = Integer(0)
    ...     identity = Integer(1)
    >>> my_join(2, 3) == my_join(3, 2)
    True
    >>> my_join(2, my_join(3, 4)) == my_join(2, 3, 4)
    True
    >>> my_join(0, 1, 4, 2, 3, 4)
    0
    >>> my_join(1, 2)
    2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lattice_%28order%29
    """

    is_commutative = True

    def __new__(cls, *args, **options):
        # Sympify each argument in args
        args = (_sympify_(arg) for arg in args)

        try:
            # Attempt to create a frozenset of filtered arguments using _new_args_filter
            _args = frozenset(cls._new_args_filter(args))
        except ShortCircuit:
            # If ShortCircuit exception is raised, return sympified zero
            return sympify(cls.zero)
        if not _args:
            # If _args is empty, return sympified identity
            return sympify(cls.identity)
        elif len(_args) == 1:
            # If _args has only one element, return that element
            return set(_args).pop()
        else:
            # For other cases, create an instance of the class with ordered arguments
            # (using super to call the superclass AssocOp's __new__)
            obj = super(AssocOp, cls).__new__(cls, *ordered(_args))
            obj._argset = _args
            return obj



    @classmethod
    def _new_args_filter(cls, arg_sequence, call_cls=None):
        """Generator filtering args"""
        # Use call_cls or cls as ncls
        ncls = call_cls or cls
        for arg in arg_sequence:
            if arg == ncls.zero:
                # If argument matches zero, raise ShortCircuit exception
                raise ShortCircuit(arg)
            elif arg == ncls.identity:
                # If argument matches identity, skip it
                continue
            elif arg.func == ncls:
                # If argument is of the same class type, yield its arguments
                yield from arg.args
            else:
                # Otherwise, yield the argument itself
                yield arg



    @classmethod
    def make_args(cls, expr):
        """
        Return a set of args such that cls(*arg_set) == expr.
        """
        if isinstance(expr, cls):
            # If expr is already an instance of cls, return its argument set
            return expr._argset
        else:
            # Otherwise, return a frozenset with the sympified version of expr
            return frozenset([sympify(expr)])



class AssocOpDispatcher:
    """
    Handler dispatcher for associative operators

    .. notes::
       This approach is experimental, and can be replaced or deleted in the future.
       See https://github.com/sympy/sympy/pull/19463.

    Explanation
    """
    def __init__(self, name, doc=None):
        """
        初始化方法，设置名称和文档字符串，并生成处理器属性名。
        
        Parameters
        ==========
        name : str
            分发器的名称
        doc : str or None, optional
            分发器的文档字符串，默认为None
        """
        self.name = name
        self.doc = doc
        self.handlerattr = "_%s_handler" % name
        self._handlergetter = attrgetter(self.handlerattr)
        self._dispatcher = Dispatcher(name)

    def __repr__(self):
        """
        返回对象的字符串表示形式。
        """
        return "<dispatched %s>" % self.name

    def register_handlerclass(self, classes, typ, on_ambiguity=ambiguity_register_error_ignore_dup):
        """
        注册两个类的处理器类，同时注册它们的反向关系。

        Parameters
        ==========
        classes : tuple of two types
            需要进行比较的两个类
        typ : type
            用于表示 *cls1* 和 *cls2* 的类，必须在该类中实现 *self* 的处理方法。
        on_ambiguity : function, optional
            处理重复注册的处理方法，默认为 ambiguity_register_error_ignore_dup。
        """
        if not len(classes) == 2:
            raise RuntimeError(
                "Only binary dispatch is supported, but got %s types: <%s>." % (
                len(classes), str_signature(classes)
            ))
        if len(set(classes)) == 1:
            raise RuntimeError(
                "Duplicate types <%s> cannot be dispatched." % str_signature(classes)
            )
        self._dispatcher.add(tuple(classes), typ, on_ambiguity=on_ambiguity)
        self._dispatcher.add(tuple(reversed(classes)), typ, on_ambiguity=on_ambiguity)

    @cacheit
    def __call__(self, *args, _sympify=True, **kwargs):
        """
        调用对象实例时执行的方法，根据参数调度相应的处理函数。

        Parameters
        ==========
        *args :
            需要处理的参数
        _sympify : bool, optional
            是否应用 sympify 转换，默认为True
        **kwargs :
            其他关键字参数
        """
        if _sympify:
            args = tuple(map(_sympify_, args))
        handlers = frozenset(map(self._handlergetter, args))

        # no need to sympify again
        return self.dispatch(handlers)(*args, _sympify=False, **kwargs)

    @cacheit
    def dispatch(self, handlers):
        """
        Select the handler class, and return its handler method.
        """

        # 快速退出，如果只有一个处理程序
        if len(handlers) == 1:
            h, = handlers
            if not isinstance(h, type):
                raise RuntimeError("Handler {!r} is not a type.".format(h))
            return h

        # 递归选择注册的处理程序优先级
        for i, typ in enumerate(handlers):

            if not isinstance(typ, type):
                raise RuntimeError("Handler {!r} is not a type.".format(typ))

            # 第一次循环，初始化处理程序
            if i == 0:
                handler = typ
            else:
                prev_handler = handler
                # 调用内部的_dispatch方法，选择合适的处理程序
                handler = self._dispatcher.dispatch(prev_handler, typ)

                if not isinstance(handler, type):
                    raise RuntimeError(
                        "Dispatcher for {!r} and {!r} must return a type, but got {!r}".format(
                        prev_handler, typ, handler
                    ))

        # 返回最终选择的处理程序类
        return handler

    @property
    def __doc__(self):
        docs = [
            "Multiply dispatched associative operator: %s" % self.name,
            "Note that support for this is experimental, see the docs for :class:`AssocOpDispatcher` for details"
        ]

        if self.doc:
            docs.append(self.doc)

        s = "Registered handler classes\n"
        s += '=' * len(s)
        docs.append(s)

        amb_sigs = []

        typ_sigs = defaultdict(list)
        # 根据优先级反序列化函数签名
        for sigs in self._dispatcher.ordering[::-1]:
            key = self._dispatcher.funcs[sigs]
            typ_sigs[key].append(sigs)

        # 根据类型和签名构建文档字符串
        for typ, sigs in typ_sigs.items():

            sigs_str = ', '.join('<%s>' % str_signature(sig) for sig in sigs)

            if isinstance(typ, RaiseNotImplementedError):
                amb_sigs.append(sigs_str)
                continue

            s = 'Inputs: %s\n' % sigs_str
            s += '-' * len(s) + '\n'
            s += typ.__name__
            docs.append(s)

        if amb_sigs:
            s = "Ambiguous handler classes\n"
            s += '=' * len(s)
            docs.append(s)

            s = '\n'.join(amb_sigs)
            docs.append(s)

        # 返回格式化后的文档字符串
        return '\n\n'.join(docs)
```