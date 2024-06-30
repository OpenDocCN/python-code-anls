# `D:\src\scipysrc\sympy\sympy\assumptions\assume.py`

```
"""A module which implements predicates and assumption context."""

# 导入必要的模块和函数
from contextlib import contextmanager
import inspect
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import Boolean, false, true
from sympy.multipledispatch.dispatcher import Dispatcher, str_signature
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.source import get_class

# 定义一个特殊的集合类，表示默认的断言，用于 `ask()` 函数
class AssumptionsContext(set):
    """
    Set containing default assumptions which are applied to the ``ask()``
    function.

    Explanation
    ===========

    This is used to represent global assumptions, but you can also use this
    class to create your own local assumptions contexts. It is basically a thin
    wrapper to Python's set, so see its documentation for advanced usage.

    Examples
    ========

    The default assumption context is ``global_assumptions``, which is initially empty:

    >>> from sympy import ask, Q
    >>> from sympy.assumptions import global_assumptions
    >>> global_assumptions
    AssumptionsContext()

    You can add default assumptions:

    >>> from sympy.abc import x
    >>> global_assumptions.add(Q.real(x))
    >>> global_assumptions
    AssumptionsContext({Q.real(x)})
    >>> ask(Q.real(x))
    True

    And remove them:

    >>> global_assumptions.remove(Q.real(x))
    >>> print(ask(Q.real(x)))
    None

    The ``clear()`` method removes every assumption:

    >>> global_assumptions.add(Q.positive(x))
    >>> global_assumptions
    AssumptionsContext({Q.positive(x)})
    >>> global_assumptions.clear()
    >>> global_assumptions
    AssumptionsContext()

    See Also
    ========

    assuming

    """

    def add(self, *assumptions):
        """Add assumptions."""
        for a in assumptions:
            super().add(a)

    def _sympystr(self, printer):
        if not self:
            return "%s()" % self.__class__.__name__
        return "{}({})".format(self.__class__.__name__, printer._print_set(self))

# 创建全局的断言上下文对象
global_assumptions = AssumptionsContext()


class AppliedPredicate(Boolean):
    """
    The class of expressions resulting from applying ``Predicate`` to
    the arguments. ``AppliedPredicate`` merely wraps its argument and
    remain unevaluated. To evaluate it, use the ``ask()`` function.

    Examples
    ========

    >>> from sympy import Q, ask
    >>> Q.integer(1)
    Q.integer(1)

    The ``function`` attribute returns the predicate, and the ``arguments``
    attribute returns the tuple of arguments.

    >>> type(Q.integer(1))
    <class 'sympy.assumptions.assume.AppliedPredicate'>
    >>> Q.integer(1).function
    Q.integer
    >>> Q.integer(1).arguments
    (1,)

    Applied predicates can be evaluated to a boolean value with ``ask``:

    >>> ask(Q.integer(1))
    True

    """
    __slots__ = ()  # 限制此类的实例不允许新增额外的实例属性
    # 创建一个新的实例，接受一个谓词和额外的参数
    def __new__(cls, predicate, *args):
        # 检查传入的谓词是否为 Predicate 类型，否则抛出类型错误异常
        if not isinstance(predicate, Predicate):
            raise TypeError("%s is not a Predicate." % predicate)
        # 对参数列表中的每个参数应用 _sympify 函数
        args = map(_sympify, args)
        # 调用父类的 __new__ 方法创建实例
        return super().__new__(cls, predicate, *args)

    @property
    def arg(self):
        """
        返回此假设使用的表达式。

        示例
        ========

        >>> from sympy import Q, Symbol
        >>> x = Symbol('x')
        >>> a = Q.integer(x + 1)
        >>> a.arg
        x + 1

        """
        # 将属性 _args 赋值给 args 变量
        args = self._args
        # 如果 args 列表长度为 2，则返回第二个元素，用于向后兼容
        if len(args) == 2:
            # 向后兼容
            return args[1]
        # 如果长度不为 2，抛出类型错误，'arg' 属性仅允许一元谓词使用
        raise TypeError("'arg' property is allowed only for unary predicates.")

    @property
    def function(self):
        """
        返回谓词函数。
        """
        # 将谓词函数作为 _args 的第一个元素返回
        return self._args[0]

    @property
    def arguments(self):
        """
        返回应用于谓词的参数。
        """
        # 将应用于谓词的参数作为 _args 的剩余部分返回
        return self._args[1:]

    def _eval_ask(self, assumptions):
        # 调用谓词函数的 eval 方法，传入参数和假设条件
        return self.function.eval(self.arguments, assumptions)

    @property
    def binary_symbols(self):
        # 从 ask 模块导入 Q
        from .ask import Q
        # 如果谓词函数是 Q.is_true
        if self.function == Q.is_true:
            # 取第一个参数
            i = self.arguments[0]
            # 如果参数是布尔类型或符号类型，则返回其二进制符号集合
            if i.is_Boolean or i.is_Symbol:
                return i.binary_symbols
        # 如果谓词函数是 Q.eq 或 Q.ne
        if self.function in (Q.eq, Q.ne):
            # 如果 true 或 false 存在于参数中
            if true in self.arguments or false in self.arguments:
                # 如果第一个参数是符号类型，则返回包含该参数的集合
                if self.arguments[0].is_Symbol:
                    return {self.arguments[0]}
                # 如果第二个参数是符号类型，则返回包含该参数的集合
                elif self.arguments[1].is_Symbol:
                    return {self.arguments[1]}
        # 默认返回空集合
        return set()
class PredicateMeta(type):
    # 定义元类 PredicateMeta，用于创建谓词类
    def __new__(cls, clsname, bases, dct):
        # 创建新的类对象
        # 如果在类属性字典中没有定义 "handler"，则分配一个空的调度器
        if "handler" not in dct:
            name = f"Ask{clsname.capitalize()}Handler"
            # 创建一个名为 name 的调度器对象，用于处理关键字 %s 的请求
            handler = Dispatcher(name, doc="Handler for key %s" % name)
            dct["handler"] = handler

        # 备份类文档字符串，如果不存在，则使用空字符串
        dct["_orig_doc"] = dct.get("__doc__", "")

        # 调用父类的 __new__ 方法来创建新的类对象
        return super().__new__(cls, clsname, bases, dct)

    @property
    def __doc__(cls):
        # 类文档字符串的属性方法
        handler = cls.handler
        doc = cls._orig_doc
        # 如果当前类不是 Predicate 类且 handler 不为空
        if cls is not Predicate and handler is not None:
            # 添加文档字符串描述 Handler
            doc += "Handler\n"
            doc += "    =======\n\n"

            # 追加调度器的文档，保持与 Sphinx 文档的一致性
            docs = ["    Multiply dispatched method: %s" % handler.name]
            if handler.doc:
                # 将调度器的文档每行添加到 docs 列表中
                for line in handler.doc.splitlines():
                    if not line:
                        continue
                    docs.append("    %s" % line)
            other = []
            # 遍历调度器函数的排序（逆序）
            for sig in handler.ordering[::-1]:
                func = handler.funcs[sig]
                if func.__doc__:
                    s = '    Inputs: <%s>' % str_signature(sig)
                    lines = []
                    # 将函数文档的每行添加到 lines 列表中
                    for line in func.__doc__.splitlines():
                        lines.append("    %s" % line)
                    s += "\n".join(lines)
                    docs.append(s)
                else:
                    other.append(str_signature(sig))
            if other:
                othersig = "    Other signatures:"
                for line in other:
                    othersig += "\n        * %s" % line
                docs.append(othersig)

            # 将所有文档字符串片段连接起来
            doc += '\n\n'.join(docs)

        # 返回最终的文档字符串
        return doc


class Predicate(Boolean, metaclass=PredicateMeta):
    """
    Base class for mathematical predicates. It also serves as a
    constructor for undefined predicate objects.

    Explanation
    ===========

    Predicate is a function that returns a boolean value [1].

    Predicate function is object, and it is instance of predicate class.
    When a predicate is applied to arguments, ``AppliedPredicate``
    instance is returned. This merely wraps the argument and remain
    unevaluated. To obtain the truth value of applied predicate, use the
    function ``ask``.

    Evaluation of predicate is done by multiple dispatching. You can
    register new handler to the predicate to support new types.

    Every predicate in SymPy can be accessed via the property of ``Q``.
    For example, ``Q.even`` returns the predicate which checks if the
    argument is even number.

    To define a predicate which can be evaluated, you must subclass this
    class, make an instance of it, and register it to ``Q``. After then,
    dispatch the handler by argument types.

    If you directly construct predicate using this class, you will get
    ``UndefinedPredicate`` which cannot be dispatched. This is useful
    """
    # 数学谓词的基类，也用作未定义谓词对象的构造函数
    is_Atom = True



    # 设置变量 is_Atom 为 True
    is_Atom = True



    def __new__(cls, *args, **kwargs):
        if cls is Predicate:
            return UndefinedPredicate(*args, **kwargs)
        obj = super().__new__(cls, *args)
        return obj



    # 定义一个特殊的方法 __new__，用于创建类的新实例
    def __new__(cls, *args, **kwargs):
        # 如果当前类是 Predicate，则返回 UndefinedPredicate 类的实例
        if cls is Predicate:
            return UndefinedPredicate(*args, **kwargs)
        # 否则调用父类的 __new__ 方法创建实例
        obj = super().__new__(cls, *args)
        return obj



    @property
    def name(self):
        # May be overridden
        return type(self).__name__



    # 定义一个属性方法 name，返回当前对象的类名
    @property
    def name(self):
        # 可能会被子类重写
        return type(self).__name__



    @classmethod
    def register(cls, *types, **kwargs):
        """
        Register the signature to the handler.
        """
        if cls.handler is None:
            raise TypeError("%s cannot be dispatched." % type(cls))
        return cls.handler.register(*types, **kwargs)



    # 定义一个类方法 register，用于将签名注册到处理器（handler）中
    @classmethod
    def register(cls, *types, **kwargs):
        """
        Register the signature to the handler.
        """
        # 如果处理器为 None，则抛出类型错误
        if cls.handler is None:
            raise TypeError("%s cannot be dispatched." % type(cls))
        # 调用处理器的 register 方法注册签名
        return cls.handler.register(*types, **kwargs)



    @classmethod
    def register_many(cls, *types, **kwargs):
        """
        Register multiple signatures to same handler.
        """
        def _(func):
            for t in types:
                if not is_sequence(t):
                    t = (t,)  # for convenience, allow passing `type` to mean `(type,)`
                cls.register(*t, **kwargs)(func)
        return _



    # 定义一个类方法 register_many，用于将多个签名注册到同一处理器（handler）中
    @classmethod
    def register_many(cls, *types, **kwargs):
        """
        Register multiple signatures to same handler.
        """
        def _(func):
            for t in types:
                if not is_sequence(t):
                    t = (t,)  # 为了方便起见，允许传递 type 来表示 (type,)
                # 调用 register 方法注册每一个签名
                cls.register(*t, **kwargs)(func)
        return _



    def __call__(self, *args):
        return AppliedPredicate(self, *args)



    # 定义一个特殊方法 __call__，使对象可调用
    def __call__(self, *args):
        # 返回一个 AppliedPredicate 对象，传入当前对象和参数 args
        return AppliedPredicate(self, *args)



    def eval(self, args, assumptions=True):
        """
        Evaluate ``self(*args)`` under the given assumptions.

        This uses only direct resolution methods, not logical inference.
        """
        result = None
        try:
            # 调用 handler 处理器处理参数 args 和假设 assumptions
            result = self.handler(*args, assumptions=assumptions)
        except NotImplementedError:
            pass
        return result



    # 定义方法 eval，用于在给定假设条件下评估 self(*args)
    def eval(self, args, assumptions=True):
        """
        Evaluate ``self(*args)`` under the given assumptions.

        This uses only direct resolution methods, not logical inference.
        """
        result = None
        try:
            # 调用 handler 处理器处理参数 args 和假设 assumptions
            result = self.handler(*args, assumptions=assumptions)
        except NotImplementedError:
            pass
        return result
    # 定义一个名为 _eval_refine 的方法，它接受一个名为 assumptions 的参数
    def _eval_refine(self, assumptions):
        # 当 Predicate 不再是布尔值时，删除这个方法
        # 目前该方法只是返回自身对象
        return self
class UndefinedPredicate(Predicate):
    """
    Predicate without handler.

    Explanation
    ===========

    This predicate is generated by using ``Predicate`` directly for
    construction. It does not have a handler, and evaluating this with
    arguments is done by SAT solver.

    Examples
    ========

    >>> from sympy import Predicate, Q
    >>> Q.P = Predicate('P')
    >>> Q.P.func
    <class 'sympy.assumptions.assume.UndefinedPredicate'>
    >>> Q.P.name
    Str('P')

    """

    handler = None  # 定义类变量 handler，初始化为 None

    def __new__(cls, name, handlers=None):
        # "handlers" 参数支持旧设计
        if not isinstance(name, Str):
            name = Str(name)  # 如果 name 不是 Str 类型，转换为 Str 类型
        obj = super(Boolean, cls).__new__(cls, name)  # 调用父类 Predicate 的 __new__ 方法创建对象
        obj.handlers = handlers or []  # 初始化对象的 handlers 属性，若 handlers 为 None 则设为空列表
        return obj

    @property
    def name(self):
        return self.args[0]  # 返回对象的第一个参数作为名称

    def _hashable_content(self):
        return (self.name,)  # 返回对象内容的可哈希版本，仅包含名称

    def __getnewargs__(self):
        return (self.name,)  # 返回用于对象重建的参数元组，仅包含名称

    def __call__(self, expr):
        return AppliedPredicate(self, expr)  # 调用对象实例返回 AppliedPredicate 对象

    def add_handler(self, handler):
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. Predicate.add_handler()
            should be replaced with the multipledispatch handler of Predicate.
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
        )
        self.handlers.append(handler)  # 添加处理程序到 handlers 列表

    def remove_handler(self, handler):
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. Predicate.remove_handler()
            should be replaced with the multipledispatch handler of Predicate.
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
        )
        self.handlers.remove(handler)  # 从 handlers 列表中移除指定的处理程序
    def eval(self, args, assumptions=True):
        # 支持已废弃设计
        # 当旧设计被移除时，这将始终返回 None
        sympy_deprecation_warning(
            """
            AskHandler 系统已被弃用。对未定义断言对象的评估应替换为 Predicate 的 multipledispatch 处理程序。
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
            stacklevel=5,
        )
        # 从参数中获取表达式
        expr, = args
        res, _res = None, None
        # 获取表达式类型的方法解析顺序
        mro = inspect.getmro(type(expr))
        # 遍历每个处理程序
        for handler in self.handlers:
            # 获取处理程序的类
            cls = get_class(handler)
            # 遍历表达式类型的方法解析顺序
            for subclass in mro:
                # 获取类中对应的评估方法
                eval_ = getattr(cls, subclass.__name__, None)
                # 如果评估方法为空，继续下一个子类
                if eval_ is None:
                    continue
                # 对表达式进行评估
                res = eval_(expr, assumptions)
                # 如果评估结果为 None，则继续尝试更高级别的类
                if res is None:
                    continue
                # 如果 _res 为 None，则将当前结果赋给 _res
                if _res is None:
                    _res = res
                else:
                    # 只有在两个解析器都得出结论时才检查一致性
                    if _res != res:
                        raise ValueError('incompatible resolutors')
                break
        # 返回评估结果
        return res
@contextmanager
def assuming(*assumptions):
    """
    上下文管理器，用于设定假设条件。

    示例
    ========

    >>> from sympy import assuming, Q, ask
    >>> from sympy.abc import x, y
    >>> print(ask(Q.integer(x + y)))
    None
    >>> with assuming(Q.integer(x), Q.integer(y)):
    ...     print(ask(Q.integer(x + y)))
    True
    """
    # 备份当前全局假设条件
    old_global_assumptions = global_assumptions.copy()
    # 更新全局假设条件为传入的假设
    global_assumptions.update(assumptions)
    try:
        # 进入上下文，执行相关代码块
        yield
    finally:
        # 无论如何，恢复原始的全局假设条件
        global_assumptions.clear()
        global_assumptions.update(old_global_assumptions)
```