# `D:\src\scipysrc\sympy\sympy\multipledispatch\dispatcher.py`

```
# 从未来模块导入 annotations，用于支持类型注解
from __future__ import annotations

# 导入警告模块中的 warn 函数
from warnings import warn
# 导入 inspect 模块，用于获取对象信息
import inspect
# 导入当前包中的 conflict 模块中的特定内容
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
# 导入当前包中的 utils 模块中的 expand_tuples 函数
from .utils import expand_tuples
# 导入 itertools 模块并使用 as 语法指定别名 itl
import itertools as itl

# 定义一个自定义的 NotImplementedError 类，用于多重分派时抛出未实现错误
class MDNotImplementedError(NotImplementedError):
    """ A NotImplementedError for multiple dispatch """

### Functions for on_ambiguity

# 定义函数 ambiguity_warn，用于在检测到模糊性时发出警告
def ambiguity_warn(dispatcher, ambiguities):
    """ Raise warning when ambiguity is detected

    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher

    See Also:
        Dispatcher.add
        warning_text
    """
    # 调用 warn 函数发出警告，警告内容由 warning_text 函数生成
    warn(warning_text(dispatcher.name, ambiguities), AmbiguityWarning)

# 定义类 RaiseNotImplementedError，用于在调用时抛出 NotImplementedError
class RaiseNotImplementedError:
    """Raise ``NotImplementedError`` when called."""

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def __call__(self, *args, **kwargs):
        types = tuple(type(a) for a in args)
        # 抛出 NotImplementedError 异常，指示模糊的类型签名
        raise NotImplementedError(
            "Ambiguous signature for %s: <%s>" % (
            self.dispatcher.name, str_signature(types)
        ))

# 定义函数 ambiguity_register_error_ignore_dup，根据不同情况注册或忽略错误处理
def ambiguity_register_error_ignore_dup(dispatcher, ambiguities):
    """
    If super signature for ambiguous types is duplicate types, ignore it.
    Else, register instance of ``RaiseNotImplementedError`` for ambiguous types.

    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher

    See Also:
        Dispatcher.add
        ambiguity_warn
    """
    # 遍历模糊类型签名对
    for amb in ambiguities:
        signature = tuple(super_signature(amb))
        # 如果超级签名中的类型是重复的，则忽略此模糊情况
        if len(set(signature)) == 1:
            continue
        # 否则，为模糊的类型签名注册 RaiseNotImplementedError 实例
        dispatcher.add(
            signature, RaiseNotImplementedError(dispatcher),
            on_ambiguity=ambiguity_register_error_ignore_dup
        )

###


# 定义全局变量 _unresolved_dispatchers，表示未解决的分派器集合，初始化为空集合
_unresolved_dispatchers: set[Dispatcher] = set()
# 定义全局变量 _resolve，表示解析标志列表，初始化为包含 True 的列表
_resolve = [True]

# 定义函数 halt_ordering，用于停止排序
def halt_ordering():
    _resolve[0] = False

# 定义函数 restart_ordering，用于重新启动排序
def restart_ordering(on_ambiguity=ambiguity_warn):
    _resolve[0] = True
    # 循环处理未解决的分派器集合
    while _unresolved_dispatchers:
        dispatcher = _unresolved_dispatchers.pop()
        dispatcher.reorder(on_ambiguity=on_ambiguity)

# 定义类 Dispatcher，用于基于类型签名分派方法
class Dispatcher:
    """ Dispatch methods based on type signature

    Use ``dispatch`` to add implementations

    Examples
    --------

    >>> from sympy.multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1

    >>> @dispatch(float)
    ... def f(x): # noqa: F811
    ...     return x - 1

    >>> f(3)
    4
    >>> f(3.0)
    2.0
    """
    # 使用 __slots__ 定义仅允许的实例属性，节省内存空间
    __slots__ = '__name__', 'name', 'funcs', 'ordering', '_cache', 'doc'

    def __init__(self, name, doc=None):
        self.name = self.__name__ = name
        self.funcs = {}
        self._cache = {}
        self.ordering = []
        self.doc = doc
    def register(self, *types, **kwargs):
        """ 注册新实现的调度器

        >>> from sympy.multipledispatch.dispatcher import Dispatcher
        >>> f = Dispatcher('f')
        >>> @f.register(int)
        ... def inc(x):
        ...     return x + 1

        >>> @f.register(float)
        ... def dec(x):
        ...     return x - 1

        >>> @f.register(list)
        ... @f.register(tuple)
        ... def reverse(x):
        ...     return x[::-1]

        >>> f(1)
        2

        >>> f(1.0)
        0.0

        >>> f([1, 2, 3])
        [3, 2, 1]
        """
        def _(func):
            # 向调度器添加类型和对应的函数实现
            self.add(types, func, **kwargs)
            return func
        return _

    @classmethod
    def get_func_params(cls, func):
        """ 获取函数的参数信息 """
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return sig.parameters.values()

    @classmethod
    def get_func_annotations(cls, func):
        """ 获取函数位置参数的注解信息 """
        params = cls.get_func_params(func)
        if params:
            Parameter = inspect.Parameter

            # 仅保留位置参数和位置或关键字参数
            params = (param for param in params
                      if param.kind in
                      (Parameter.POSITIONAL_ONLY,
                       Parameter.POSITIONAL_OR_KEYWORD))

            # 获取参数的注解元组
            annotations = tuple(
                param.annotation
                for param in params)

            # 如果所有注解都存在且不为默认空值，则返回注解元组
            if not any(ann is Parameter.empty for ann in annotations):
                return annotations
    def add(self, signature, func, on_ambiguity=ambiguity_warn):
        """ Add new types/method pair to dispatcher

        >>> from sympy.multipledispatch import Dispatcher
        >>> D = Dispatcher('add')
        >>> D.add((int, int), lambda x, y: x + y)
        >>> D.add((float, float), lambda x, y: x + y)

        >>> D(1, 2)
        3
        >>> D(1, 2.0)
        Traceback (most recent call last):
        ...
        NotImplementedError: Could not find signature for add: <int, float>

        When ``add`` detects a warning it calls the ``on_ambiguity`` callback
        with a dispatcher/itself, and a set of ambiguous type signature pairs
        as inputs.  See ``ambiguity_warn`` for an example.
        """
        # Handle annotations
        if not signature:
            # If signature is not provided, attempt to retrieve it from function annotations
            annotations = self.get_func_annotations(func)
            if annotations:
                signature = annotations

        # Handle union types
        if any(isinstance(typ, tuple) for typ in signature):
            # If signature contains union types (tuples), expand and recursively add each type pair
            for typs in expand_tuples(signature):
                self.add(typs, func, on_ambiguity)
            return

        # Validate types in the signature
        for typ in signature:
            if not isinstance(typ, type):
                # Raise TypeError if any type in signature is not a valid Python type
                str_sig = ', '.join(c.__name__ if isinstance(c, type)
                                    else str(c) for c in signature)
                raise TypeError("Tried to dispatch on non-type: %s\n"
                                "In signature: <%s>\n"
                                "In function: %s" %
                                (typ, str_sig, self.name))

        # Store the function associated with the provided signature
        self.funcs[signature] = func

        # Reorder the dispatcher functions based on current ambiguity resolution strategy
        self.reorder(on_ambiguity=on_ambiguity)

        # Clear the cache of previously resolved function mappings
        self._cache.clear()

    def reorder(self, on_ambiguity=ambiguity_warn):
        """ Reorder the dispatcher's functions based on resolved ordering

        If resolution of dispatch function ordering is successful, updates the ordering
        and handles any ambiguities by invoking the on_ambiguity callback.

        If resolution is unresolved, marks the dispatcher as unresolved in the _unresolved_dispatchers set.
        """
        if _resolve[0]:
            # If resolution is successful, update ordering and handle ambiguities
            self.ordering = ordering(self.funcs)
            amb = ambiguities(self.funcs)
            if amb:
                on_ambiguity(self, amb)
        else:
            # If resolution is unresolved, mark the dispatcher as unresolved
            _unresolved_dispatchers.add(self)

    def __call__(self, *args, **kwargs):
        """ Call method to dispatch based on argument types

        Retrieves types of arguments, resolves the appropriate function using the cache,
        or dispatches and resolves it if not cached. Handles errors if no matching function
        is found or if matching functions fail.

        Returns the result of the dispatched function.
        """
        types = tuple([type(arg) for arg in args])
        try:
            # Attempt to retrieve cached function based on argument types
            func = self._cache[types]
        except KeyError:
            # If no cached function found, attempt to dispatch and resolve function
            func = self.dispatch(*types)
            if not func:
                raise NotImplementedError(
                    'Could not find signature for %s: <%s>' %
                    (self.name, str_signature(types)))
            self._cache[types] = func
        try:
            # Execute resolved function with provided arguments and keyword arguments
            return func(*args, **kwargs)

        except MDNotImplementedError:
            # If function raises MDNotImplementedError, attempt to find and execute alternative functions
            funcs = self.dispatch_iter(*types)
            next(funcs)  # burn first
            for func in funcs:
                try:
                    return func(*args, **kwargs)
                except MDNotImplementedError:
                    pass
            # If no alternative functions succeed, raise NotImplementedError
            raise NotImplementedError("Matching functions for "
                                      "%s: <%s> found, but none completed successfully"
                                      % (self.name, str_signature(types)))
    def __str__(self):
        return "<dispatched %s>" % self.name
    __repr__ = __str__

    # 定义 dispatch 方法，用于确定给定类型签名的适当实现
    def dispatch(self, *types):
        """ Deterimine appropriate implementation for this type signature

        This method is internal.  Users should call this object as a function.
        Implementation resolution occurs within the ``__call__`` method.

        >>> from sympy.multipledispatch import dispatch
        >>> @dispatch(int)
        ... def inc(x):
        ...     return x + 1

        >>> implementation = inc.dispatch(int)
        >>> implementation(3)
        4

        >>> print(inc.dispatch(float))
        None

        See Also:
            ``sympy.multipledispatch.conflict`` - module to determine resolution order
        """

        # 如果 types 在 funcs 中，则直接返回对应的函数实现
        if types in self.funcs:
            return self.funcs[types]

        # 否则，尝试从 dispatch_iter 中获取下一个适合的函数实现
        try:
            return next(self.dispatch_iter(*types))
        except StopIteration:
            return None

    # 定义 dispatch_iter 方法，用于返回适合给定类型签名的函数实现生成器
    def dispatch_iter(self, *types):
        n = len(types)
        for signature in self.ordering:
            # 如果 signature 的长度与 types 相等，并且 types 中的每个类型都是 signature 中对应位置的子类，则返回对应函数实现
            if len(signature) == n and all(map(issubclass, types, signature)):
                result = self.funcs[signature]
                yield result

    # 定义 resolve 方法，用于确定给定类型签名的适当实现（已弃用，推荐使用 dispatch 方法）
    def resolve(self, types):
        """ Deterimine appropriate implementation for this type signature

        .. deprecated:: 0.4.4
            Use ``dispatch(*types)`` instead
        """
        # 发出警告，resolve 方法已弃用
        warn("resolve() is deprecated, use dispatch(*types)",
             DeprecationWarning)

        # 返回 dispatch 方法的结果
        return self.dispatch(*types)

    # 定义 __getstate__ 方法，用于获取对象的序列化状态
    def __getstate__(self):
        return {'name': self.name,
                'funcs': self.funcs}

    # 定义 __setstate__ 方法，用于设置对象的序列化状态
    def __setstate__(self, d):
        self.name = d['name']
        self.funcs = d['funcs']
        self.ordering = ordering(self.funcs)
        self._cache = {}

    # 定义 __doc__ 属性，用于返回对象的文档字符串
    @property
    def __doc__(self):
        docs = ["Multiply dispatched method: %s" % self.name]

        if self.doc:
            docs.append(self.doc)

        other = []
        for sig in self.ordering[::-1]:
            func = self.funcs[sig]
            if func.__doc__:
                s = 'Inputs: <%s>\n' % str_signature(sig)
                s += '-' * len(s) + '\n'
                s += func.__doc__.strip()
                docs.append(s)
            else:
                other.append(str_signature(sig))

        if other:
            docs.append('Other signatures:\n    ' + '\n    '.join(other))

        return '\n\n'.join(docs)

    # 定义 _help 方法，用于返回给定输入对应函数的文档字符串
    def _help(self, *args):
        return self.dispatch(*map(type, args)).__doc__

    # 定义 help 方法，用于打印给定输入对应函数的文档字符串
    def help(self, *args, **kwargs):
        """ Print docstring for the function corresponding to inputs """
        print(self._help(*args))

    # 定义 _source 方法，用于返回给定输入对应函数的源代码
    def _source(self, *args):
        func = self.dispatch(*map(type, args))
        if not func:
            raise TypeError("No function found")
        return source(func)

    # 定义 source 方法，用于打印给定输入对应函数的源代码
    def source(self, *args, **kwargs):
        """ Print source code for the function corresponding to inputs """
        print(self._source(*args))
# 定义一个函数 `source`，用于获取给定函数的源代码字符串
def source(func):
    # 获取函数所在文件路径，并生成文件信息字符串
    s = 'File: %s\n\n' % inspect.getsourcefile(func)
    # 将函数的源代码字符串追加到文件信息字符串中
    s = s + inspect.getsource(func)
    # 返回生成的完整字符串
    return s


# 定义一个类 `MethodDispatcher`，继承自 `Dispatcher` 类
class MethodDispatcher(Dispatcher):
    """ Dispatch methods based on type signature

    See Also:
        Dispatcher
    """

    # 类方法，用于获取函数的参数列表
    @classmethod
    def get_func_params(cls, func):
        # 检查是否有 `signature` 属性可用
        if hasattr(inspect, "signature"):
            # 获取函数的签名对象
            sig = inspect.signature(func)
            # 返回除第一个参数外的所有参数列表
            return itl.islice(sig.parameters.values(), 1, None)

    # 实例方法，用于描述实例的获取行为
    def __get__(self, instance, owner):
        # 将实例对象和所有者类保存到当前实例中
        self.obj = instance
        self.cls = owner
        return self

    # 实例可调用对象的执行方法
    def __call__(self, *args, **kwargs):
        # 获取传入参数的类型元组
        types = tuple([type(arg) for arg in args])
        # 根据参数类型分派对应的函数
        func = self.dispatch(*types)
        # 如果未找到对应的函数，则抛出未实现错误
        if not func:
            raise NotImplementedError('Could not find signature for %s: <%s>' %
                                      (self.name, str_signature(types)))
        # 调用匹配到的函数并返回结果
        return func(self.obj, *args, **kwargs)


# 函数，返回参数类型的字符串表示
def str_signature(sig):
    """ String representation of type signature

    >>> from sympy.multipledispatch.dispatcher import str_signature
    >>> str_signature((int, float))
    'int, float'
    """
    # 使用类型类名创建类型签名的字符串表示
    return ', '.join(cls.__name__ for cls in sig)


# 函数，返回分派函数中的歧义警告文本
def warning_text(name, amb):
    """ The text for ambiguity warnings """
    # 创建包含警告信息的文本
    text = "\nAmbiguities exist in dispatched function %s\n\n" % (name)
    text += "The following signatures may result in ambiguous behavior:\n"
    # 遍历并添加每对签名对应的歧义信息
    for pair in amb:
        text += "\t" + \
            ', '.join('[' + str_signature(s) + ']' for s in pair) + "\n"
    text += "\n\nConsider making the following additions:\n\n"
    # 添加建议的函数签名列表文本
    text += '\n\n'.join(['@dispatch(' + str_signature(super_signature(s))
                         + ')\ndef %s(...)' % name for s in amb])
    # 返回完整的警告文本
    return text
```