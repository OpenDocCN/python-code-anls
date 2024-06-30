# `D:\src\scipysrc\scipy\scipy\_lib\_uarray\_backend.py`

```
import typing  # 导入 typing 模块，用于类型提示
import types  # 导入 types 模块，用于操作类型信息
import inspect  # 导入 inspect 模块，用于解析 Python 对象的结构
import functools  # 导入 functools 模块，用于高阶函数（Higher-order functions）
from . import _uarray  # 从当前包中导入 _uarray 模块
import copyreg  # 导入 copyreg 模块，用于注册自定义的 Pickle 函数
import pickle  # 导入 pickle 模块，用于对象的序列化和反序列化
import contextlib  # 导入 contextlib 模块，用于创建上下文管理器

from ._uarray import (  # type: ignore
    BackendNotImplementedError,  # 导入 _uarray 模块中的 BackendNotImplementedError 异常
    _Function,  # 导入 _uarray 模块中的 _Function 类
    _SkipBackendContext,  # 导入 _uarray 模块中的 _SkipBackendContext 类
    _SetBackendContext,  # 导入 _uarray 模块中的 _SetBackendContext 类
    _BackendState,  # 导入 _uarray 模块中的 _BackendState 类
)

__all__ = [  # 将下面列出的符号导出给 from module import *
    "set_backend",  # 设置后端的函数名
    "set_global_backend",  # 设置全局后端的函数名
    "skip_backend",  # 跳过后端的函数名
    "register_backend",  # 注册后端的函数名
    "determine_backend",  # 确定后端的函数名
    "determine_backend_multi",  # 多次确定后端的函数名
    "clear_backends",  # 清除所有后端的函数名
    "create_multimethod",  # 创建多方法的函数名
    "generate_multimethod",  # 生成多方法的函数名
    "_Function",  # _uarray 模块中的 _Function 类
    "BackendNotImplementedError",  # 后端未实现的异常
    "Dispatchable",  # 分发对象的类型
    "wrap_single_convertor",  # 包装单个转换器的函数名
    "wrap_single_convertor_instance",  # 包装单个转换器实例的函数名
    "all_of_type",  # 指定类型的对象
    "mark_as",  # 标记为某一类的对象
    "set_state",  # 设置状态的函数名
    "get_state",  # 获取状态的函数名
    "reset_state",  # 重置状态的函数名
    "_BackendState",  # _uarray 模块中的 _BackendState 类
    "_SkipBackendContext",  # _uarray 模块中的 _SkipBackendContext 类
    "_SetBackendContext",  # _uarray 模块中的 _SetBackendContext 类
]

ArgumentExtractorType = typing.Callable[..., tuple["Dispatchable", ...]]  # 定义一个类型别名，表示参数提取器的类型
ArgumentReplacerType = typing.Callable[  # 定义一个类型别名，表示参数替换器的类型
    [tuple, dict, tuple], tuple[tuple, dict]
]

def unpickle_function(mod_name, qname, self_):
    import importlib  # 导入 importlib 模块，用于动态导入模块

    try:
        module = importlib.import_module(mod_name)  # 根据模块名导入模块
        qname = qname.split(".")  # 将限定名称按点号分割成列表
        func = module
        for q in qname:
            func = getattr(func, q)  # 逐级获取模块中的属性或子模块

        if self_ is not None:
            func = types.MethodType(func, self_)  # 如果 self_ 存在，将函数转换为方法

        return func  # 返回获取到的函数或方法对象
    except (ImportError, AttributeError) as e:
        from pickle import UnpicklingError  # 导入 UnpicklingError 异常类

        raise UnpicklingError from e  # 抛出反序列化时的异常

def pickle_function(func):
    mod_name = getattr(func, "__module__", None)  # 获取函数的模块名
    qname = getattr(func, "__qualname__", None)  # 获取函数的限定名称
    self_ = getattr(func, "__self__", None)  # 获取函数的 self 对象

    try:
        test = unpickle_function(mod_name, qname, self_)  # 尝试反序列化函数对象
    except pickle.UnpicklingError:
        test = None

    if test is not func:  # 检查反序列化后的函数对象是否与原函数相同
        raise pickle.PicklingError(
            f"Can't pickle {func}: it's not the same object as {test}"
        )

    return unpickle_function, (mod_name, qname, self_)  # 返回反序列化函数所需的信息元组

def pickle_state(state):
    return _uarray._BackendState._unpickle, state._pickle()  # 返回状态对象的反序列化函数和其 Pickle 表示

def pickle_set_backend_context(ctx):
    return _SetBackendContext, ctx._pickle()  # 返回设置后端上下文对象的反序列化函数和其 Pickle 表示

def pickle_skip_backend_context(ctx):
    return _SkipBackendContext, ctx._pickle()  # 返回跳过后端上下文对象的反序列化函数和其 Pickle 表示

copyreg.pickle(_Function, pickle_function)  # 使用 copyreg 注册 _Function 类的序列化函数
copyreg.pickle(_uarray._BackendState, pickle_state)  # 使用 copyreg 注册 _BackendState 类的序列化函数
copyreg.pickle(_SetBackendContext, pickle_set_backend_context)  # 使用 copyreg 注册 _SetBackendContext 类的序列化函数
copyreg.pickle(_SkipBackendContext, pickle_skip_backend_context)  # 使用 copyreg 注册 _SkipBackendContext 类的序列化函数

def get_state():
    """
    Returns an opaque object containing the current state of all the backends.

    Can be used for synchronization between threads/processes.

    See Also
    --------
    set_state
        Sets the state returned by this function.
    """
    return _uarray.get_state()  # 调用 _uarray 模块中的 get_state 函数，返回当前所有后端的状态信息

@contextlib.contextmanager
def reset_state():
    """
    Returns a context manager that resets all state once exited.

    See Also
    --------
    set_state
        Context manager that sets the backend state.
    """
    yield  # 使用生成器实现上下文管理器，当退出时重置所有状态
    # 定义一个生成器函数，用作上下文管理器，没有参数
    get_state
        Gets a state to be set by this context manager.
    """
    # 使用 set_state 函数设定由 get_state 函数返回的状态
    with set_state(get_state()):
        # 生成器函数的返回值
        yield
@contextlib.contextmanager
def set_state(state):
    """
    A context manager that sets the state of the backends to one returned by :obj:`get_state`.

    See Also
    --------
    get_state
        Gets a state to be set by this context manager.
    """  # noqa: E501
    # 获取当前的状态，用于备份
    old_state = get_state()
    # 设置后端状态为传入的状态
    _uarray.set_state(state)
    try:
        # yield 控制权返回给调用方
        yield
    finally:
        # 恢复原始的状态
        _uarray.set_state(old_state, True)


def create_multimethod(*args, **kwargs):
    """
    Creates a decorator for generating multimethods.

    This function creates a decorator that can be used with an argument
    extractor in order to generate a multimethod. Other than for the
    argument extractor, all arguments are passed on to
    :obj:`generate_multimethod`.

    See Also
    --------
    generate_multimethod
        Generates a multimethod.
    """
    # 返回一个装饰器，用于生成多方法
    def wrapper(a):
        return generate_multimethod(a, *args, **kwargs)

    return wrapper


def generate_multimethod(
    argument_extractor: ArgumentExtractorType,
    argument_replacer: ArgumentReplacerType,
    domain: str,
    default: typing.Optional[typing.Callable] = None,
):
    """
    Generates a multimethod.

    Parameters
    ----------
    argument_extractor : ArgumentExtractorType
        A callable which extracts the dispatchable arguments. Extracted arguments
        should be marked by the :obj:`Dispatchable` class. It has the same signature
        as the desired multimethod.
    argument_replacer : ArgumentReplacerType
        A callable with the signature (args, kwargs, dispatchables), which should also
        return an (args, kwargs) pair with the dispatchables replaced inside the
        args/kwargs.
    domain : str
        A string value indicating the domain of this multimethod.
    default: Optional[Callable], optional
        The default implementation of this multimethod, where ``None`` (the default)
        specifies there is no default implementation.

    Examples
    --------
    In this example, ``a`` is to be dispatched over, so we return it, while marking it
    as an ``int``.
    The trailing comma is needed because the args have to be returned as an iterable.

    >>> def override_me(a, b):
    ...   return Dispatchable(a, int),

    Next, we define the argument replacer that replaces the dispatchables inside
    args/kwargs with the supplied ones.

    >>> def override_replacer(args, kwargs, dispatchables):
    ...     return (dispatchables[0], args[1]), {}

    Next, we define the multimethod.

    >>> overridden_me = generate_multimethod(
    ...     override_me, override_replacer, "ua_examples"
    ... )

    Notice that there's no default implementation, unless you supply one.

    >>> overridden_me(1, "a")
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...

    >>> overridden_me2 = generate_multimethod(
    ...     override_me, override_replacer, "ua_examples", default=lambda x, y: (x, y)
    ... )
    >>> overridden_me2(1, "a")
    """
    # 生成一个多方法

    # 返回生成的多方法装饰器
    return generate_multimethod
    # 创建一个包含两个元素的元组 (1, 'a')
    (1, 'a')

    # 查看相关内容
    See Also
    --------
    uarray
        查看模块文档，了解如何通过创建后端来覆盖该方法。

    # 使用给定的参数提取器获取默认参数、参数默认值和选项
    kw_defaults, arg_defaults, opts = get_defaults(argument_extractor)

    # 创建一个 _Function 对象 ua_func，用于处理参数提取、替换、领域、参数默认值、关键字默认值和默认值
    ua_func = _Function(
        argument_extractor,
        argument_replacer,
        domain,
        arg_defaults,
        kw_defaults,
        default,
    )

    # 使用 functools.update_wrapper() 更新 ua_func 的包装器，以便它看起来像 argument_extractor 函数
    return functools.update_wrapper(ua_func, argument_extractor)
# 设置首选后端的上下文管理器
def set_backend(backend, coerce=False, only=False):
    # 尝试从缓存中获取已设置的上下文
    try:
        return backend.__ua_cache__["set", coerce, only]
    except AttributeError:
        # 如果对象没有 __ua_cache__ 属性，则创建一个空字典
        backend.__ua_cache__ = {}
    except KeyError:
        pass

    # 创建一个设置后端的上下文对象
    ctx = _SetBackendContext(backend, coerce, only)
    # 将该上下文对象存入 backend.__ua_cache__ 中
    backend.__ua_cache__["set", coerce, only] = ctx
    # 返回上下文对象
    return ctx


# 跳过特定后端的上下文管理器
def skip_backend(backend):
    # 尝试从缓存中获取已设置的跳过上下文
    try:
        return backend.__ua_cache__["skip"]
    except AttributeError:
        # 如果对象没有 __ua_cache__ 属性，则创建一个空字典
        backend.__ua_cache__ = {}
    except KeyError:
        pass

    # 创建一个跳过特定后端的上下文对象
    ctx = _SkipBackendContext(backend)
    # 将该上下文对象存入 backend.__ua_cache__ 中
    backend.__ua_cache__["skip"] = ctx
    # 返回上下文对象
    return ctx


# 获取函数的默认参数值信息
def get_defaults(f):
    # 获取函数的参数签名信息
    sig = inspect.signature(f)
    kw_defaults = {}
    arg_defaults = []
    opts = set()
    # 遍历函数的参数签名
    for k, v in sig.parameters.items():
        # 如果参数有默认值，则将其添加到关键字参数默认值字典中
        if v.default is not inspect.Parameter.empty:
            kw_defaults[k] = v.default
        # 如果参数类型是位置参数或者位置或关键字参数，则将其默认值添加到位置参数默认值列表中
        if v.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            arg_defaults.append(v.default)
        # 将参数名添加到选项集合中
        opts.add(k)

    # 返回关键字参数默认值字典、位置参数默认值元组和选项集合
    return kw_defaults, tuple(arg_defaults), opts


# 设置全局后端的方法
def set_global_backend(backend, coerce=False, only=False, *, try_last=False):
    """
    This utility method replaces the default backend for permanent use. It
    will be tried in the list of backends automatically, unless the
    ``only`` flag is set on a backend. This will be the first tried
    backend outside the :obj:`set_backend` context manager.

    Note that this method is not thread-safe.

    .. warning::
        We caution library authors against using this function in
        their code. We do *not* support this use-case. This function
        is meant to be used only by users themselves, or by a reference
        implementation, if one exists.

    Parameters
    ----------
    backend
        The backend to register.
    coerce : bool
        Whether to coerce input types when trying this backend.
    only : bool
        If ``True``, no more backends will be tried if this fails.
        Implied by ``coerce=True``.
    """
    pass  # 此函数目前没有实现任何具体的操作，只是文档说明
    # 尝试设置全局后端到指定的 `backend`
    def _uarray.set_global_backend(backend, coerce, only, try_last):
        # 使用指定的参数设置全局后端，`coerce` 和 `only` 是额外的参数选项
        _uarray.set_global_backend(backend, coerce, only, try_last)
# 注册一个后端以供永久使用的实用方法。除非后端上设置了 ``only`` 标志，否则它将自动在后端列表中尝试。
# 注意，此方法不是线程安全的。
def register_backend(backend):
    _uarray.register_backend(backend)


# 清除已注册的后端的实用方法。
#
# .. warning::
#     我们警告库作者不要在其代码中使用此函数。我们不支持这种用例。此函数仅供用户自己使用。
#
# .. warning::
#     不要在多方法调用内部使用此方法，否则程序很可能会崩溃。
#
# Parameters
# ----------
# domain : Optional[str]
#     要取消注册后端的域。``None`` 表示为所有域取消注册。
# registered : bool
#     是否清除已注册的后端。参见 :obj:`register_backend`。
# globals : bool
#     是否清除全局后端。参见 :obj:`set_global_backend`。
#
# See Also
# --------
# register_backend : 全局注册后端。
# set_global_backend : 设置全局后端。
def clear_backends(domain, registered=True, globals=False):
    _uarray.clear_backends(domain, registered, globals)


class Dispatchable:
    """
    一个将参数标记为特定调度类型的实用类。

    Attributes
    ----------
    value
        Dispatchable 的值。

    type
        Dispatchable 的类型。

    Examples
    --------
    >>> x = Dispatchable(1, str)
    >>> x
    <Dispatchable: type=<class 'str'>, value=1>

    See Also
    --------
    all_of_type
        标记函数的所有未标记参数。

    mark_as
        允许创建一个将值标记为特定类型的实用函数。
    """

    def __init__(self, value, dispatch_type, coercible=True):
        self.value = value
        self.type = dispatch_type
        self.coercible = coercible

    def __getitem__(self, index):
        return (self.type, self.value)[index]

    def __str__(self):
        return f"<{type(self).__name__}: type={self.type!r}, value={self.value!r}>"

    __repr__ = __str__


def mark_as(dispatch_type):
    """
    创建一个将值标记为特定类型的实用函数。

    Examples
    --------
    >>> mark_int = mark_as(int)
    >>> mark_int(1)
    <Dispatchable: type=<class 'int'>, value=1>
    """
    return functools.partial(Dispatchable, dispatch_type=dispatch_type)


def all_of_type(arg_type):
    """
    将所有未标记的参数标记为给定类型。

    Examples
    --------
    >>> @all_of_type(str)
    ... def f(a, b):
    ...     return a, Dispatchable(b, int)
    >>> f('a', 1)
    (<Dispatchable: type=<class 'str'>, value='a'>,
     <Dispatchable: type=<class 'int'>, value=1>)
    """
    # 定义一个装饰器函数 outer，接受一个函数 func 作为参数
    def outer(func):
        # 使用 functools 模块的 wraps 装饰器，将 inner 函数的元数据与 func 保持一致
        @functools.wraps(func)
        # 定义内部函数 inner，接受任意数量的位置参数和关键字参数
        def inner(*args, **kwargs):
            # 调用被装饰的函数 func，获取其返回的结果
            extracted_args = func(*args, **kwargs)
            # 对返回结果中的每个元素进行处理，如果不是 Dispatchable 类型，则包装成 Dispatchable 对象
            return tuple(
                Dispatchable(arg, arg_type)
                if not isinstance(arg, Dispatchable)
                else arg
                for arg in extracted_args
            )

        # 返回内部函数 inner，这是装饰器的实际作用对象
        return inner

    # 返回装饰器函数 outer，使其可以作为一个装饰器使用
    return outer
# 定义函数 wrap_single_convertor，用于将单个元素的转换函数 convert_single 包装为适用于多个元素的转换函数
def wrap_single_convertor(convert_single):
    """
    Wraps a ``__ua_convert__`` defined for a single element to all elements.
    If any of them return ``NotImplemented``, the operation is assumed to be
    undefined.

    Accepts a signature of (value, type, coerce).
    """

    # 内部函数 __ua_convert__，用 functools.wraps 包装，保留原始 convert_single 的元信息
    @functools.wraps(convert_single)
    def __ua_convert__(dispatchables, coerce):
        # 初始化一个空列表用于存储转换后的结果
        converted = []
        # 遍历 dispatchables 中的每个元素
        for d in dispatchables:
            # 调用 convert_single 函数进行转换，传入 value, type, 和 coerce 的值
            c = convert_single(d.value, d.type, coerce and d.coercible)

            # 如果 convert_single 返回 NotImplemented，则整个操作返回 NotImplemented
            if c is NotImplemented:
                return NotImplemented

            # 将转换后的结果添加到 converted 列表中
            converted.append(c)

        # 返回所有转换后的结果
        return converted

    # 返回内部函数 __ua_convert__，即包装后的多元素转换函数
    return __ua_convert__


# 定义函数 wrap_single_convertor_instance，用于将单个元素的转换函数 convert_single 包装为适用于实例方法的转换函数
def wrap_single_convertor_instance(convert_single):
    """
    Wraps a ``__ua_convert__`` defined for a single element to all elements.
    If any of them return ``NotImplemented``, the operation is assumed to be
    undefined.

    Accepts a signature of (value, type, coerce).
    """

    # 内部函数 __ua_convert__，用 functools.wraps 包装，保留原始 convert_single 的元信息
    @functools.wraps(convert_single)
    def __ua_convert__(self, dispatchables, coerce):
        # 初始化一个空列表用于存储转换后的结果
        converted = []
        # 遍历 dispatchables 中的每个元素
        for d in dispatchables:
            # 调用 convert_single 函数进行转换，传入 self, value, type, 和 coerce 的值
            c = convert_single(self, d.value, d.type, coerce and d.coercible)

            # 如果 convert_single 返回 NotImplemented，则整个操作返回 NotImplemented
            if c is NotImplemented:
                return NotImplemented

            # 将转换后的结果添加到 converted 列表中
            converted.append(c)

        # 返回所有转换后的结果
        return converted

    # 返回内部函数 __ua_convert__，即包装后的多元素转换函数
    return __ua_convert__


# 定义函数 determine_backend，用于设置支持给定值的第一个活动后端
def determine_backend(value, dispatch_type, *, domain, only=True, coerce=False):
    """Set the backend to the first active backend that supports ``value``

    This is useful for functions that call multimethods without any dispatchable
    arguments. You can use :func:`determine_backend` to ensure the same backend
    is used everywhere in a block of multimethod calls.

    Parameters
    ----------
    value
        The value being tested
    dispatch_type
        The dispatch type associated with ``value``, aka
        ":ref:`marking <MarkingGlossary>`".
    domain: string
        The domain to query for backends and set.
    coerce: bool
        Whether or not to allow coercion to the backend's types. Implies ``only``.
    only: bool
        Whether or not this should be the last backend to try.

    See Also
    --------
    set_backend: For when you know which backend to set

    Notes
    -----

    Support is determined by the ``__ua_convert__`` protocol. Backends not
    supporting the type must return ``NotImplemented`` from their
    ``__ua_convert__`` if they don't support input of that type.

    Examples
    --------

    Suppose we have two backends ``BackendA`` and ``BackendB`` each supporting
    different types, ``TypeA`` and ``TypeB``. Neither supporting the other type:

    >>> with ua.set_backend(ex.BackendA):
    ...     ex.call_multimethod(ex.TypeB(), ex.TypeB())
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...

    Now consider a multimethod that creates a new object of ``TypeA``, or
    ``TypeB`` depending on the active backend.
    """
    >>> with ua.set_backend(ex.BackendA), ua.set_backend(ex.BackendB):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, ex.TypeA())
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...
    
    ``res`` is an object of ``TypeB`` because ``BackendB`` is set in the
    innermost with statement. So, ``call_multimethod`` fails since the types
    don't match.
    
    Instead, we need to first find a backend suitable for all of our objects.
    
    >>> with ua.set_backend(ex.BackendA), ua.set_backend(ex.BackendB):
    ...     x = ex.TypeA()
    ...     with ua.determine_backend(x, "mark", domain="ua_examples"):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, x)
    TypeA
    
    
    注释：
    
    
    # 在多重上下文管理器中设置两个不同的后端（BackendA 和 BackendB），进行多方法创建和调用的示例
    # 在第一个示例中，由于最内层的 with 语句设置了 BackendB，因此 res 是 TypeB 的对象。
    # 因此，调用 call_multimethod 失败，因为对象类型不匹配。
    
    # 需要先找到适合所有对象的后端。
    
    # 在第二个示例中，首先创建一个 TypeA 的对象 x，并在一个新的上下文管理器中确定 x 的后端。
    # 然后再创建多方法并调用，这次可以成功，因为调用的对象和返回的对象类型匹配。
# 定义一个函数，用于确定支持所有“dispatchables”的后端

"""
This is useful for functions that call multimethods without any dispatchable
arguments. You can use :func:`determine_backend_multi` to ensure the same
backend is used everywhere in a block of multimethod calls involving
multiple arrays.
此函数适用于调用多方法的函数，其中没有分派参数。您可以使用 :func:`determine_backend_multi` 确保在涉及多个数组的多方法调用块中处处使用相同的后端。

Parameters
----------
dispatchables: Sequence[Union[uarray.Dispatchable, Any]]
    必须支持的分派对象列表
domain: string
    用于查询后端和设置的域。
coerce: bool
    是否允许强制转换到后端的类型。意味着 ``only`` 参数为真。
only: bool
    是否应该是最后一个尝试的后端。
dispatch_type: Optional[Any]
    与 ``dispatchables`` 关联的默认分派类型，即 ":ref:`marking <MarkingGlossary>`"。

See Also
--------
determine_backend: 用于单个分派值
set_backend: 确定要设置哪个后端时

Notes
-----
支持由 ``__ua_convert__`` 协议决定。不支持该类型输入的后端必须从其 ``__ua_convert__`` 中返回 ``NotImplemented`` 。

Examples
--------

:func:`determine_backend` 允许从单个对象设置后端。:func:`determine_backend_multi` 允许同时检查多个对象在后端中的支持情况。假设我们有一个支持在同一调用中的 ``TypeA`` 和 ``TypeB`` 的 ``BackendAB`` ，以及一个不支持 ``TypeA`` 的 ``BackendBC`` 。

>>> with ua.set_backend(ex.BackendAB), ua.set_backend(ex.BackendBC):
...     a, b = ex.TypeA(), ex.TypeB()
...     with ua.determine_backend_multi(
...         [ua.Dispatchable(a, "mark"), ua.Dispatchable(b, "mark")],
...         domain="ua_examples"
...     ):
...         res = ex.creation_multimethod()
...         ex.call_multimethod(res, a, b)
TypeA

这不会调用 ``BackendBC`` ，因为它不支持 ``TypeA`` 。

我们还可以在指定 ``dispatchables`` 参数的默认 ``dispatch_type`` 时，不使用 ``ua.Dispatchable`` 。

>>> with ua.set_backend(ex.BackendAB), ua.set_backend(ex.BackendBC):
...     a, b = ex.TypeA(), ex.TypeB()
...     with ua.determine_backend_multi(
...         [a, b], dispatch_type="mark", domain="ua_examples"
...     ):
...         res = ex.creation_multimethod()
...         ex.call_multimethod(res, a, b)
TypeA

"""
# 如果 kwargs 中包含 "dispatch_type" 键，则将其弹出并为 dispatchables 参数中的非 Dispatchable 对象创建 Dispatchable 对象
if "dispatch_type" in kwargs:
    disp_type = kwargs.pop("dispatch_type")
    dispatchables = tuple(
        d if isinstance(d, Dispatchable) else Dispatchable(d, disp_type)
        for d in dispatchables
    )
    else:
        # 如果不是单个 Dispatchable 对象，则将 dispatchables 转换为元组
        dispatchables = tuple(dispatchables)
        # 检查所有 dispatchables 是否都是 Dispatchable 类型的实例
        if not all(isinstance(d, Dispatchable) for d in dispatchables):
            # 如果有任何一个不是 Dispatchable 类型的实例，则抛出类型错误
            raise TypeError("dispatchables must be instances of uarray.Dispatchable")

    # 检查是否有额外的关键字参数传入
    if len(kwargs) != 0:
        # 如果有额外的关键字参数，则抛出类型错误，显示具体的参数内容
        raise TypeError(f"Received unexpected keyword arguments: {kwargs}")

    # 确定后端引擎，根据给定的 domain、dispatchables 和 coerce 参数
    backend = _uarray.determine_backend(domain, dispatchables, coerce)

    # 将确定的后端引擎设置为当前的后端，可以选择性地进行 coerce 和 only 参数设置
    return set_backend(backend, coerce=coerce, only=only)
```