# `D:\src\scipysrc\sympy\sympy\core\kind.py`

```
"""
Module to efficiently partition SymPy objects.

This system is introduced because class of SymPy object does not always
represent the mathematical classification of the entity. For example,
``Integral(1, x)`` and ``Integral(Matrix([1,2]), x)`` are both instance
of ``Integral`` class. However the former is number and the latter is
matrix.

One way to resolve this is defining subclass for each mathematical type,
such as ``MatAdd`` for the addition between matrices. Basic algebraic
operation such as addition or multiplication take this approach, but
defining every class for every mathematical object is not scalable.

Therefore, we define the "kind" of the object and let the expression
infer the kind of itself from its arguments. Function and class can
filter the arguments by their kind, and behave differently according to
the type of itself.

This module defines basic kinds for core objects. Other kinds such as
``ArrayKind`` or ``MatrixKind`` can be found in corresponding modules.

.. notes::
       This approach is experimental, and can be replaced or deleted in the future.
       See https://github.com/sympy/sympy/pull/20549.
"""

from collections import defaultdict  # 导入 defaultdict 类来创建默认字典

from .cache import cacheit  # 导入 cacheit 函数，用于缓存装饰器
from sympy.multipledispatch.dispatcher import (Dispatcher,  # 导入多重分派模块的相关函数和类
    ambiguity_warn, ambiguity_register_error_ignore_dup,
    str_signature, RaiseNotImplementedError)


class KindMeta(type):
    """
    Metaclass for ``Kind``.

    Assigns empty ``dict`` as class attribute ``_inst`` for every class,
    in order to endow singleton-like behavior.
    """
    def __new__(cls, clsname, bases, dct):
        dct['_inst'] = {}  # 创建空字典 _inst 作为类属性，实现类似单例的行为
        return super().__new__(cls, clsname, bases, dct)


class Kind(object, metaclass=KindMeta):
    """
    Base class for kinds.

    Kind of the object represents the mathematical classification that
    the entity falls into. It is expected that functions and classes
    recognize and filter the argument by its kind.

    Kind of every object must be carefully selected so that it shows the
    intention of design. Expressions may have different kind according
    to the kind of its arguments. For example, arguments of ``Add``
    must have common kind since addition is group operator, and the
    resulting ``Add()`` has the same kind.

    For the performance, each kind is as broad as possible and is not
    based on set theory. For example, ``NumberKind`` includes not only
    complex number but expression containing ``S.Infinity`` or ``S.NaN``
    which are not strictly number.

    Kind may have arguments as parameter. For example, ``MatrixKind()``
    may be constructed with one element which represents the kind of its
    elements.

    ``Kind`` behaves in singleton-like fashion. Same signature will
    return the same object.

    """
    # 定义一个特殊方法 __new__，用于创建类的新实例
    def __new__(cls, *args):
        # 检查参数 args 是否已经存在于类变量 _inst 中
        if args in cls._inst:
            # 如果存在，直接返回之前保存的实例
            inst = cls._inst[args]
        else:
            # 如果不存在，调用父类的 __new__ 方法创建新的实例
            inst = super().__new__(cls)
            # 将新创建的实例保存到类变量 _inst 中，以便下次复用
            cls._inst[args] = inst
        # 返回实例对象
        return inst
class _UndefinedKind(Kind):
    """
    Default kind for all SymPy objects. If the kind is not defined for
    the object, or if the object cannot infer the kind from its
    arguments, this will be returned.

    Examples
    ========

    >>> from sympy import Expr
    >>> Expr().kind
    UndefinedKind
    """
    # 定义一个默认的 SymPy 对象的种类，默认情况下返回此种类
    def __new__(cls):
        return super().__new__(cls)

    # 返回对象的字符串表示为 "UndefinedKind"
    def __repr__(self):
        return "UndefinedKind"

# 将 _UndefinedKind 类的实例赋值给 UndefinedKind 变量
UndefinedKind = _UndefinedKind()


class _NumberKind(Kind):
    """
    Kind for all numeric objects.

    This kind represents every number, including complex numbers,
    infinity and ``S.NaN``. Other objects such as quaternions do not
    have this kind.

    Most ``Expr`` are initially designed to represent the number, so
    this will be the most common kind in SymPy core. For example
    ``Symbol()``, which represents a scalar, has this kind as long as it
    is commutative.

    Numbers form a field. Any operation between number-kind objects will
    result this kind as well.

    Examples
    ========

    >>> from sympy import S, oo, Symbol
    >>> S.One.kind
    NumberKind
    >>> (-oo).kind
    NumberKind
    >>> S.NaN.kind
    NumberKind

    Commutative symbols are treated as numbers.

    >>> x = Symbol('x')
    >>> x.kind
    NumberKind
    >>> Symbol('y', commutative=False).kind
    UndefinedKind

    Operations between numbers result in the number kind.

    >>> (x+1).kind
    NumberKind

    See Also
    ========

    sympy.core.expr.Expr.is_Number : check if the object is strictly
    subclass of ``Number`` class.

    sympy.core.expr.Expr.is_number : check if the object is a number
    without any free symbols.
    """
    # 定义一个用于所有数值对象的种类
    def __new__(cls):
        return super().__new__(cls)

    # 返回对象的字符串表示为 "NumberKind"
    def __repr__(self):
        return "NumberKind"

# 将 _NumberKind 类的实例赋值给 NumberKind 变量
NumberKind = _NumberKind()


class _BooleanKind(Kind):
    """
    Kind for boolean objects.

    SymPy's ``S.true``, ``S.false``, and built-in ``True`` and ``False``
    have this kind. Boolean numbers ``1`` and ``0`` are not relevant.

    Examples
    ========

    >>> from sympy import S, Q
    >>> S.true.kind
    BooleanKind
    >>> Q.even(3).kind
    BooleanKind
    """
    # 定义一个用于布尔对象的种类
    def __new__(cls):
        return super().__new__(cls)

    # 返回对象的字符串表示为 "BooleanKind"
    def __repr__(self):
        return "BooleanKind"

# 将 _BooleanKind 类的实例赋值给 BooleanKind 变量
BooleanKind = _BooleanKind()


class KindDispatcher:
    """
    Dispatcher to select a kind from multiple kinds by binary dispatching.

    .. notes::
       This approach is experimental, and can be replaced or deleted in
       the future.

    Explanation
    ===========

    SymPy objects :obj:`sympy.core.kind.Kind()` vaguely represent the
    algebraic structure where the object belongs. Therefore, with
    given operations, we can always find a dominating kind among the
    different kinds. This class selects the kind by recursive binary
    dispatching. If the result cannot be determined, ``UndefinedKind``
    is returned.

    Examples
    ========

    Multiplication between numbers returns the number kind.
    """
    # 选择多个种类中的一种进行分发的调度器
    def __init__(self):
        pass

    # 这个调度器使用二进制分发来选择种类
    def select_kind(self):
        pass
    >>> from sympy import NumberKind, Mul
    >>> Mul._kind_dispatcher(NumberKind, NumberKind)
    NumberKind


# 调用 Mul 类的 _kind_dispatcher 方法，传入两个 NumberKind 类型作为参数，返回结果为 NumberKind。
# 这里展示了乘法操作中两个相同种类的数值对象相乘的情况。

Multiplication between number and unknown-kind object returns unknown kind.


    >>> from sympy import UndefinedKind
    >>> Mul._kind_dispatcher(NumberKind, UndefinedKind)
    UndefinedKind


# 调用 Mul 类的 _kind_dispatcher 方法，传入一个 NumberKind 和一个 UndefinedKind 类型作为参数，返回结果为 UndefinedKind。
# 这里展示了乘法操作中数值对象和未定义种类对象相乘的情况。

Any number and order of kinds is allowed.


    >>> Mul._kind_dispatcher(UndefinedKind, NumberKind)
    UndefinedKind
    >>> Mul._kind_dispatcher(NumberKind, UndefinedKind, NumberKind)
    UndefinedKind


# 对于任意数量和顺序的种类对象都是允许的。
# - 第一个示例：传入 UndefinedKind 和 NumberKind，返回 UndefinedKind。
# - 第二个示例：传入 NumberKind、UndefinedKind 和 NumberKind，返回 UndefinedKind。
# 这两个示例展示了乘法操作中不同种类对象组合的情况。

Since matrix forms a vector space over scalar field, multiplication
between matrix with numeric element and number returns matrix with
numeric element.


    >>> from sympy.matrices import MatrixKind
    >>> Mul._kind_dispatcher(MatrixKind(NumberKind), NumberKind)
    MatrixKind(NumberKind)


# 调用 Mul 类的 _kind_dispatcher 方法，传入一个具有数值元素的 MatrixKind(NumberKind) 和一个 NumberKind 类型作为参数，返回结果为 MatrixKind(NumberKind)。
# 这里展示了乘法操作中矩阵对象（具有数值元素）与数值对象相乘的情况。

If a matrix with number element and another matrix with unknown-kind
element are multiplied, we know that the result is matrix but the
kind of its elements is unknown.


    >>> Mul._kind_dispatcher(MatrixKind(NumberKind), MatrixKind(UndefinedKind))
    MatrixKind(UndefinedKind)


# 调用 Mul 类的 _kind_dispatcher 方法，传入一个具有数值元素的 MatrixKind(NumberKind) 和一个具有未定义种类元素的 MatrixKind(UndefinedKind) 作为参数，返回结果为 MatrixKind(UndefinedKind)。
# 这里展示了乘法操作中两个不同种类的矩阵对象相乘的情况。

Parameters
==========

name : str

commutative : bool, optional
    If True, binary dispatch will be automatically registered in
    reversed order as well.

doc : str, optional


# 这部分是函数 __init__ 的参数说明，描述了初始化方法的参数。


"""
    def __init__(self, name, commutative=False, doc=None):
        self.name = name
        self.doc = doc
        self.commutative = commutative
        self._dispatcher = Dispatcher(name)


# 初始化方法 __init__，接受 name、commutative 和 doc 作为参数，分别表示名称、是否交换性和文档字符串。
# - self.name：保存名称
# - self.doc：保存文档字符串
# - self.commutative：保存是否交换性的布尔值
# - self._dispatcher：初始化一个 Dispatcher 对象

    def __repr__(self):
        return "<dispatched %s>" % self.name


# 定义了 __repr__ 方法，返回一个格式化字符串，表示被调度的名称。

    def register(self, *types, **kwargs):
        """
        Register the binary dispatcher for two kind classes.

        If *self.commutative* is ``True``, signature in reversed order is
        automatically registered as well.
        """


# 定义了 register 方法，用于注册两个种类类别的二进制调度器。
# 如果 self.commutative 为 True，则会自动注册反向顺序的签名。

        on_ambiguity = kwargs.pop("on_ambiguity", None)
        if not on_ambiguity:
            if self.commutative:
                on_ambiguity = ambiguity_register_error_ignore_dup
            else:
                on_ambiguity = ambiguity_warn
        kwargs.update(on_ambiguity=on_ambiguity)

        if not len(types) == 2:
            raise RuntimeError(
                "Only binary dispatch is supported, but got %s types: <%s>." % (
                len(types), str_signature(types)
            ))

        def _(func):
            self._dispatcher.add(types, func, **kwargs)
            if self.commutative:
                self._dispatcher.add(tuple(reversed(types)), func, **kwargs)
        return _


# 在 register 方法中：
# - 处理 on_ambiguity 参数，如果未提供则根据 commutative 设置默认值。
# - 检查 types 参数长度是否为 2，否则抛出 RuntimeError 异常。
# - 返回一个闭包函数，用于向 _dispatcher 中添加新的函数调度规则。

    def __call__(self, *args, **kwargs):
        if self.commutative:
            kinds = frozenset(args)
        else:
            kinds = []
            prev = None
            for a in args:
                if prev is not a:
                    kinds.append(a)
                    prev = a
        return self.dispatch_kinds(kinds, **kwargs)


# 定义了 __call__ 方法，允许实例对象被调用，根据 commutative 属性创建 kinds 集合或列表，并调用 dispatch_kinds 方法进行分发。

    @cacheit


# 使用装饰器 @cacheit 对接下来的方法进行缓存处理。
    # 根据给定的 kinds 列表进行分发处理，支持多种类型的处理，返回最终的类型结果
    def dispatch_kinds(self, kinds, **kwargs):
        # 如果 kinds 列表长度为 1，直接返回唯一的 Kind 类型结果
        if len(kinds) == 1:
            result, = kinds
            if not isinstance(result, Kind):
                raise RuntimeError("%s is not a kind." % result)
            return result

        # 遍历 kinds 列表，确保每个元素都是 Kind 类型，否则抛出异常
        for i, kind in enumerate(kinds):
            if not isinstance(kind, Kind):
                raise RuntimeError("%s is not a kind." % kind)

            # 第一个元素直接赋值给 result，后续进行类型分发处理
            if i == 0:
                result = kind
            else:
                prev_kind = result

                # 获取前一种类和当前种类的类型
                t1, t2 = type(prev_kind), type(kind)
                k1, k2 = prev_kind, kind
                # 使用类型分发器获取对应的处理函数
                func = self._dispatcher.dispatch(t1, t2)

                # 如果未注册且支持交换顺序，尝试反向顺序
                if func is None and self.commutative:
                    func = self._dispatcher.dispatch(t2, t1)
                    k1, k2 = k2, k1

                # 如果仍然没有找到合适的处理函数，则结果为 UndefinedKind
                if func is None:
                    result = UndefinedKind
                else:
                    # 使用找到的处理函数处理 k1 和 k2，结果必须是 Kind 类型，否则抛出异常
                    result = func(k1, k2)
                    if not isinstance(result, Kind):
                        raise RuntimeError(
                            "Dispatcher for {!r} and {!r} must return a Kind, but got {!r}".format(
                            prev_kind, kind, result
                        ))

        return result

    @property
    def __doc__(self):
        # 构造对象的文档字符串
        docs = [
            "Kind dispatcher : %s" % self.name,
            "Note that support for this is experimental. See the docs for :class:`KindDispatcher` for details"
        ]

        # 如果对象有额外的文档说明，则添加到文档字符串中
        if self.doc:
            docs.append(self.doc)

        # 构造已注册的种类类别部分的文档内容
        s = "Registered kind classes\n"
        s += '=' * len(s)
        docs.append(s)

        # 使用 defaultdict 整理类型签名信息
        typ_sigs = defaultdict(list)
        for sigs in self._dispatcher.ordering[::-1]:
            key = self._dispatcher.funcs[sigs]
            typ_sigs[key].append(sigs)

        # 遍历整理后的类型签名信息，生成文档内容
        for func, sigs in typ_sigs.items():

            sigs_str = ', '.join('<%s>' % str_signature(sig) for sig in sigs)

            if isinstance(func, RaiseNotImplementedError):
                amb_sigs.append(sigs_str)
                continue

            s = 'Inputs: %s\n' % sigs_str
            s += '-' * len(s) + '\n'
            if func.__doc__:
                s += func.__doc__.strip()
            else:
                s += func.__name__
            docs.append(s)

        # 如果存在模糊的种类类别，生成相应的文档内容
        if amb_sigs:
            s = "Ambiguous kind classes\n"
            s += '=' * len(s)
            docs.append(s)

            s = '\n'.join(amb_sigs)
            docs.append(s)

        # 返回最终构造的文档字符串
        return '\n\n'.join(docs)
```