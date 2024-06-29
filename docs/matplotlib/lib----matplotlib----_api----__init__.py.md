# `D:\src\scipysrc\matplotlib\lib\matplotlib\_api\__init__.py`

```py
"""
Helper functions for managing the Matplotlib API.

This documentation is only relevant for Matplotlib developers, not for users.

.. warning::

    This module and its submodules are for internal use only.  Do not use them
    in your own code.  We may change the API at any time with no warning.

"""

import functools
import itertools
import re
import sys
import warnings

from .deprecation import (  # noqa: F401
    deprecated, warn_deprecated,
    rename_parameter, delete_parameter, make_keyword_only,
    deprecate_method_override, deprecate_privatize_attribute,
    suppress_matplotlib_deprecation_warning,
    MatplotlibDeprecationWarning)


class classproperty:
    """
    Like `property`, but also triggers on access via the class, and it is the
    *class* that's passed as argument.

    Examples
    --------
    ::

        class C:
            @classproperty
            def foo(cls):
                return cls.__name__

        assert C.foo == "C"
    """

    def __init__(self, fget, fset=None, fdel=None, doc=None):
        self._fget = fget
        if fset is not None or fdel is not None:
            raise ValueError('classproperty only implements fget.')
        self.fset = fset
        self.fdel = fdel
        # docs are ignored for now
        self._doc = doc

    def __get__(self, instance, owner):
        return self._fget(owner)

    @property
    def fget(self):
        return self._fget


# In the following check_foo() functions, the first parameter is positional-only to make
# e.g. `_api.check_isinstance([...], types=foo)` work.

def check_isinstance(types, /, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is an instance
    of one of *types*; if not, raise an appropriate TypeError.

    As a special case, a ``None`` entry in *types* is treated as NoneType.

    Examples
    --------
    >>> _api.check_isinstance((SomeClass, None), arg=arg)
    """
    # Determine the type for None and convert types into a tuple
    none_type = type(None)
    types = ((types,) if isinstance(types, type) else
             (none_type,) if types is None else
             tuple(none_type if tp is None else tp for tp in types))

    # Helper function to get the type name
    def type_name(tp):
        return ("None" if tp is none_type
                else tp.__qualname__ if tp.__module__ == "builtins"
                else f"{tp.__module__}.{tp.__qualname__}")

    # Iterate over key-value pairs in kwargs and check types
    for k, v in kwargs.items():
        if not isinstance(v, types):
            names = [*map(type_name, types)]
            if "None" in names:  # Move it to the end for better wording.
                names.remove("None")
                names.append("None")
            raise TypeError(
                "{!r} must be an instance of {}, not a {}".format(
                    k,
                    ", ".join(names[:-1]) + " or " + names[-1]
                    if len(names) > 1 else names[0],
                    type_name(type(v))))


def check_in_list(values, /, *, _print_supported_values=True, **kwargs):
    """
    Check if values in kwargs are in the specified list of values.

    """
    For each *key, value* pair in *kwargs*, check that *value* is in *values*;
    if not, raise an appropriate ValueError.

    Parameters
    ----------
    values : iterable
        Sequence of values to check on.
    _print_supported_values : bool, default: True
        Whether to print *values* when raising ValueError.
    **kwargs : dict
        *key, value* pairs as keyword arguments to find in *values*.

    Raises
    ------
    ValueError
        If any *value* in *kwargs* is not found in *values*.

    Examples
    --------
    >>> _api.check_in_list(["foo", "bar"], arg=arg, other_arg=other_arg)
    """
    # 如果 kwargs 为空，则抛出类型错误异常
    if not kwargs:
        raise TypeError("No argument to check!")
    
    # 遍历 kwargs 中的每个键值对
    for key, val in kwargs.items():
        # 检查 val 是否在 values 中
        if val not in values:
            # 准备错误消息，指出 val 不是 key 的有效值
            msg = f"{val!r} is not a valid value for {key}"
            # 如果 _print_supported_values 为真，则附加支持的值列表到错误消息中
            if _print_supported_values:
                msg += f"; supported values are {', '.join(map(repr, values))}"
            # 抛出 ValueError 异常，带上错误消息
            raise ValueError(msg)
# 定义一个函数 check_shape，接受位置参数 shape 和关键字参数 kwargs
def check_shape(shape, /, **kwargs):
    """
    对于 kwargs 中的每个键值对，检查值的形状是否与 shape 相符合；
    如果不符合，抛出适当的 ValueError 异常。

    shape 中的 None 表示自由尺寸，可以具有任意长度。
    例如 (None, 2) -> (N, 2)

    要检查的值必须是 numpy 数组。

    Examples
    --------
    要检查形状为 (N, 2) 的数组

    >>> _api.check_shape((None, 2), arg=arg, other_arg=other_arg)
    """
    # 遍历 kwargs 中的键值对
    for k, v in kwargs.items():
        # 获取当前值 v 的形状
        data_shape = v.shape

        # 检查形状长度是否相同，以及每个维度是否符合 shape 中的要求
        if (len(data_shape) != len(shape)
                or any(s != t and t is not None for s, t in zip(data_shape, shape))):
            # 准备维度标签和形状描述文本
            dim_labels = iter(itertools.chain(
                'NMLKJIH',
                (f"D{i}" for i in itertools.count())))
            text_shape = ", ".join([str(n) if n is not None else next(dim_labels)
                                    for n in shape[::-1]][::-1])
            if len(shape) == 1:
                text_shape += ","

            # 抛出值错误，说明期望的形状与实际形状不符
            raise ValueError(
                f"{k!r} must be {len(shape)}D with shape ({text_shape}), "
                f"but your input has shape {v.shape}"
            )


# 定义一个函数 check_getitem，接受位置参数 mapping 和关键字参数 kwargs
def check_getitem(mapping, /, **kwargs):
    """
    kwargs 必须包含单个键值对。如果 key 存在于 mapping 中，返回 mapping[value]；
    否则，抛出适当的 ValueError 异常。

    Examples
    --------
    >>> _api.check_getitem({"foo": "bar"}, arg=arg)
    """
    # 如果 kwargs 中不是单个键值对，抛出值错误
    if len(kwargs) != 1:
        raise ValueError("check_getitem takes a single keyword argument")
    
    # 获取唯一的键值对 (k, v)
    (k, v), = kwargs.items()
    
    # 尝试从 mapping 中获取 v 对应的值
    try:
        return mapping[v]
    except KeyError:
        # 如果 KeyError，抛出值错误，说明 v 不是有效的值
        raise ValueError(
            f"{v!r} is not a valid value for {k}; supported values are "
            f"{', '.join(map(repr, mapping))}") from None


# 定义一个装饰器函数 caching_module_getattr，接受一个类 cls 作为参数
def caching_module_getattr(cls):
    """
    用于实现模块级别 __getattr__ 的辅助装饰器，作为一个类来使用。

    此装饰器必须在模块的顶层使用如下::

        @caching_module_getattr
        class __getattr__:  # 类名必须为 ``__getattr__``
            @property  # 只有属性会被考虑
            def name(self): ...

    ``__getattr__`` 类将被替换为一个 ``__getattr__`` 函数，尝试访问模块上的 ``name``
    将解析为相应的属性（可能会被诸如 ``_api.deprecated`` 之类的装饰器标记为废弃）。
    所有属性都会隐式缓存。如果找不到相应名称的属性，将生成并抛出适当的 AttributeError。
    """

    # 断言 cls 的名称必须为 "__getattr__"
    assert cls.__name__ == "__getattr__"
    
    # 收集 cls 中所有的属性，过滤出属性类型为 property 的项
    props = {name: prop for name, prop in vars(cls).items()
             if isinstance(prop, property)}
    
    # 创建 cls 的实例
    instance = cls()

    # 返回一个被缓存的函数
    @functools.cache
    # 定义一个特殊方法 __getattr__，用于在属性不存在时动态获取属性值
    def __getattr__(name):
        # 检查属性名是否在 props 字典中
        if name in props:
            # 如果存在，返回该属性对应的值，通过调用其 __get__ 方法获取属性值
            return props[name].__get__(instance)
        # 如果属性名不存在于 props 字典中，抛出 AttributeError 异常
        raise AttributeError(
            # 异常消息，指明模块名和未找到的属性名
            f"module {cls.__module__!r} has no attribute {name!r}")
    
    # 返回定义的 __getattr__ 方法，使其可以被调用
    return __getattr__
# 定义一个装饰器函数，用于为类定义属性的别名。
def define_aliases(alias_d, cls=None):
    """
    Class decorator for defining property aliases.

    Use as ::

        @_api.define_aliases({"property": ["alias", ...], ...})
        class C: ...

    For each property, if the corresponding ``get_property`` is defined in the
    class so far, an alias named ``get_alias`` will be defined; the same will
    be done for setters.  If neither the getter nor the setter exists, an
    exception will be raised.

    The alias map is stored as the ``_alias_map`` attribute on the class and
    can be used by `.normalize_kwargs` (which assumes that higher priority
    aliases come last).
    """
    if cls is None:  # 如果没有提供类参数，返回实际的类装饰器函数。
        return functools.partial(define_aliases, alias_d)

    # 内部函数，用于创建属性别名的方法。
    def make_alias(name):  # 在 *name* 上强制使用闭包。
        @functools.wraps(getattr(cls, name))
        def method(self, *args, **kwargs):
            return getattr(self, name)(*args, **kwargs)
        return method

    # 遍历传入的 alias_d 字典，为每个属性设置别名。
    for prop, aliases in alias_d.items():
        exists = False
        for prefix in ["get_", "set_"]:
            # 检查类的变量中是否存在以 get_ 或 set_ 开头的方法名。
            if prefix + prop in vars(cls):
                exists = True
                for alias in aliases:
                    # 创建别名方法。
                    method = make_alias(prefix + prop)
                    method.__name__ = prefix + alias
                    method.__doc__ = f"Alias for `{prefix + prop}`."
                    setattr(cls, prefix + alias, method)
        # 如果找不到对应的 getter 或 setter 方法，则抛出 ValueError 异常。
        if not exists:
            raise ValueError(
                f"Neither getter nor setter exists for {prop!r}")

    # 返回当前类的所有别名及其映射关系。
    def get_aliased_and_aliases(d):
        return {*d, *(alias for aliases in d.values() for alias in aliases)}

    # 检查现有的 _alias_map 属性，防止与新的 alias_d 中的别名发生冲突。
    preexisting_aliases = getattr(cls, "_alias_map", {})
    conflicting = (get_aliased_and_aliases(preexisting_aliases)
                   & get_aliased_and_aliases(alias_d))
    if conflicting:
        # 如果发生冲突，则抛出 NotImplementedError 异常。
        raise NotImplementedError(
            f"Parent class already defines conflicting aliases: {conflicting}")
    
    # 将新的别名映射添加到 _alias_map 属性中，并返回更新后的类。
    cls._alias_map = {**preexisting_aliases, **alias_d}
    return cls
    # 选择匹配的签名函数以执行
    # 
    # 注意
    # -----
    # `select_matching_signature` 旨在帮助实现具有多重签名的函数。一般情况下，应避免使用这种函数，除非出于向后兼容的考虑。
    # 典型的使用模式如下：
    # ::
    # 
    #     def my_func(*args, **kwargs):
    #         params = select_matching_signature(
    #             [lambda old1, old2: locals(), lambda new: locals()],
    #             *args, **kwargs)
    #         if "old1" in params:
    #             warn_deprecated(...)
    #             old1, old2 = params.values()  # 注意 locals() 是有序的。
    #         else:
    #             new, = params.values()
    #         # 使用 params 执行操作
    # 
    #     这使得 *my_func* 可以被调用，要么传入两个参数 (*old1* 和 *old2*)，要么传入一个参数 (*new*)。
    #     注意新签名应该放在最后，这样如果调用者传入的参数与任何签名都不匹配，将会抛出 `TypeError`。
    """
    # 与其依赖于 locals() 的顺序，可以使用 func 的签名来实现（``bound = inspect.signature(func).bind(*args, **kwargs);
    # bound.apply_defaults(); return bound``），但这种方法明显较慢。
    # 遍历函数列表，尝试每个函数来执行，直到找到匹配的签名。
    for i, func in enumerate(funcs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            # 如果当前函数是最后一个，并且仍然没有找到匹配的签名，则抛出 TypeError。
            if i == len(funcs) - 1:
                raise
def nargs_error(name, takes, given):
    """Generate a TypeError to be raised by function calls with wrong arity."""
    # 构造一个 TypeError，指示函数调用的参数数量错误
    return TypeError(f"{name}() takes {takes} positional arguments but "
                     f"{given} were given")


def kwarg_error(name, kw):
    """
    Generate a TypeError to be raised by function calls with wrong kwarg.

    Parameters
    ----------
    name : str
        The name of the calling function.
    kw : str or Iterable[str]
        Either the invalid keyword argument name, or an iterable yielding
        invalid keyword arguments (e.g., a ``kwargs`` dict).
    """
    # 如果 kw 不是字符串，取其迭代器的第一个元素作为错误的关键字参数名
    if not isinstance(kw, str):
        kw = next(iter(kw))
    # 构造一个 TypeError，指示函数调用有意外的关键字参数
    return TypeError(f"{name}() got an unexpected keyword argument '{kw}'")


def recursive_subclasses(cls):
    """Yield *cls* and direct and indirect subclasses of *cls*."""
    # 生成器函数，递归地生成 cls 的直接和间接子类
    yield cls
    for subcls in cls.__subclasses__():
        yield from recursive_subclasses(subcls)


def warn_external(message, category=None):
    """
    `warnings.warn` wrapper that sets *stacklevel* to "outside Matplotlib".

    The original emitter of the warning can be obtained by patching this
    function back to `warnings.warn`, i.e. ``_api.warn_external =
    warnings.warn`` (or ``functools.partial(warnings.warn, stacklevel=2)``,
    etc.).
    """
    # 检查调用此函数的栈帧，以确定警告的来源是否在 Matplotlib 之外
    frame = sys._getframe()
    for stacklevel in itertools.count(1):
        if frame is None:
            # 在嵌入式上下文中可能会遇到 frame 为 None 的情况
            break
        if not re.match(r"\A(matplotlib|mpl_toolkits)(\Z|\.(?!tests\.))",
                        # 解决 sphinx-gallery 未设置 __name__ 的问题
                        frame.f_globals.get("__name__", "")):
            break
        frame = frame.f_back
    # 预先断开局部变量和栈帧之间的引用循环
    del frame
    # 发出警告，指定消息和警告类别，堆栈级别设置为在 Matplotlib 之外
    warnings.warn(message, category, stacklevel)
```