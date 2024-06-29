# `D:\src\scipysrc\pandas\pandas\util\_decorators.py`

```
# 引入使用`annotations`特性的导入语句
from __future__ import annotations

# 导入装饰函数相关的模块和函数
from functools import wraps
import inspect
# 导入文本处理相关模块，用于缩进处理
from textwrap import dedent
# 引入类型提示相关模块和类型
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
# 导入警告相关模块，用于发出警告
import warnings

# 导入Pandas内部模块，用于缓存只读属性
from pandas._libs.properties import cache_readonly
# 导入Pandas内部模块，用于类型相关定义
from pandas._typing import (
    F,
    T,
)
# 导入Pandas内部模块，用于处理异常的函数
from pandas.util._exceptions import find_stack_level

# 如果在类型检查环境下，导入特定的集合抽象类
if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Mapping,
    )


# 装饰器函数，用于向已弃用函数添加警告和修改文档字符串以标明弃用信息
def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: str | None = None,
    klass: type[Warning] | None = None,
    stacklevel: int = 2,
    msg: str | None = None,
) -> Callable[[F], F]:
    """
    Return a new function that emits a deprecation warning on use.

    To use this method for a deprecated function, another function
    `alternative` with the same signature must exist. The deprecated
    function will emit a deprecation warning, and in the docstring
    it will contain the deprecation directive with the provided version
    so it can be detected for future removal.

    Parameters
    ----------
    name : str
        Name of function to deprecate.
    alternative : func
        Function to use instead.
    version : str
        Version of pandas in which the method has been deprecated.
    alt_name : str, optional
        Name to use in preference of alternative.__name__.
    klass : Warning, default FutureWarning
        Class of warning to emit.
    stacklevel : int, default 2
        How many levels deep in the stack to issue the warning.
    msg : str
        The message to display in the warning.
        Default is '{name} is deprecated. Use {alt_name} instead.'
    """
    # 如果未提供替代名称，则使用替代函数的名称
    alt_name = alt_name or alternative.__name__
    # 如果未指定警告类，则默认使用FutureWarning
    klass = klass or FutureWarning
    # 如果未提供警告消息，则使用默认的弃用消息格式
    warning_msg = msg or f"{name} is deprecated, use {alt_name} instead."

    # 包装函数，用于发出弃用警告并调用替代函数
    @wraps(alternative)
    def wrapper(*args, **kwargs) -> Callable[..., Any]:
        warnings.warn(warning_msg, klass, stacklevel=stacklevel)
        return alternative(*args, **kwargs)

    # 向文档字符串添加弃用指示
    # 检查函数是否具有正确格式的文档字符串，包括一行简短摘要和正文
    msg = msg or f"Use `{alt_name}` instead."
    doc_error_msg = (
        "deprecate needs a correctly formatted docstring in "
        "the target function (should have a one liner short "
        "summary, and opening quotes should be in their own "
        f"line). Found:\n{alternative.__doc__}"
    )

    # 在优化模式下，Python会移除文档字符串，因此我们检查是否存在正确格式的文档字符串
    # 允许文档字符串为空
    if alternative.__doc__:
        if alternative.__doc__.count("\n") < 3:
            raise AssertionError(doc_error_msg)
        empty1, summary, empty2, doc_string = alternative.__doc__.split("\n", 3)
        if empty1 or empty2 and not summary:
            raise AssertionError(doc_error_msg)
        # 格式化新的文档字符串，添加弃用信息和版本号
        wrapper.__doc__ = dedent(
            f"""
        {summary.strip()}

        .. deprecated:: {version}
            {msg}

        {dedent(doc_string)}"""
        )
    # 返回一个函数装饰器 wrapper，用于修饰其他函数
    return wrapper  # type: ignore[return-value]
# 定义一个装饰器函数，用于标记函数中已弃用的关键字参数
def deprecate_kwarg(
    # 被弃用的参数名称
    old_arg_name: str,
    # 新参数的名称，如果为 None，则表示该关键字参数已被弃用
    new_arg_name: str | None,
    # 用于将旧参数映射到新参数的映射表或可调用对象，可选
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = None,
    # 控制警告信息显示的堆栈级别，默认为 2
    stacklevel: int = 2,
) -> Callable[[F], F]:
    """
    Decorator to deprecate a keyword argument of a function.

    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise warning that
        ``old_arg_name`` keyword is deprecated.
    mapping : dict or callable
        If mapping is present, use it to translate old arguments to
        new arguments. A callable must do its own value checking;
        values not found in a dict will be forwarded unchanged.

    Examples
    --------
    The following deprecates 'cols', using 'columns' instead

    >>> @deprecate_kwarg(old_arg_name="cols", new_arg_name="columns")
    ... def f(columns=""):
    ...     print(columns)
    >>> f(columns="should work ok")
    should work ok

    >>> f(cols="should raise warning")  # doctest: +SKIP
    FutureWarning: cols is deprecated, use columns instead
      warnings.warn(msg, FutureWarning)
    should raise warning

    >>> f(cols="should error", columns="can't pass do both")  # doctest: +SKIP
    TypeError: Can only specify 'cols' or 'columns', not both

    >>> @deprecate_kwarg("old", "new", {"yes": True, "no": False})
    ... def f(new=False):
    ...     print("yes!" if new else "no!")
    >>> f(old="yes")  # doctest: +SKIP
    FutureWarning: old='yes' is deprecated, use new=True instead
      warnings.warn(msg, FutureWarning)
    yes!

    To raise a warning that a keyword will be removed entirely in the future

    >>> @deprecate_kwarg(old_arg_name="cols", new_arg_name=None)
    ... def f(cols="", another_param=""):
    ...     print(cols)
    >>> f(cols="should raise warning")  # doctest: +SKIP
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version please takes steps to stop use of 'cols'
    should raise warning
    >>> f(another_param="should not raise warning")  # doctest: +SKIP
    should not raise warning

    >>> f(cols="should raise warning", another_param="")  # doctest: +SKIP
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version please takes steps to stop use of 'cols'
    should raise warning
    """
    # 如果 mapping 存在但不是 dict 且不可调用，则引发 TypeError
    if mapping is not None and not hasattr(mapping, "get") and not callable(mapping):
        raise TypeError(
            "mapping from old to new argument values must be dict or callable!"
        )
    # 定义一个装饰器函数，用于处理关键字参数的废弃警告
    def _deprecate_kwarg(func: F) -> F:
        # 使用 functools.wraps 装饰器，保留原始函数的元数据
        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable[..., Any]:
            # 弹出旧的关键字参数值，如果存在的话
            old_arg_value = kwargs.pop(old_arg_name, None)

            # 如果旧的关键字参数值不为 None，则进行处理
            if old_arg_value is not None:
                # 如果新参数名为 None，则发出警告并返回原函数调用
                if new_arg_name is None:
                    msg = (
                        f"the {old_arg_name!r} keyword is deprecated and "
                        "will be removed in a future version. Please take "
                        f"steps to stop the use of {old_arg_name!r}"
                    )
                    warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                    kwargs[old_arg_name] = old_arg_value
                    return func(*args, **kwargs)

                # 如果提供了映射函数，则根据映射处理旧的参数值
                elif mapping is not None:
                    if callable(mapping):
                        new_arg_value = mapping(old_arg_value)
                    else:
                        new_arg_value = mapping.get(old_arg_value, old_arg_value)
                    msg = (
                        f"the {old_arg_name}={old_arg_value!r} keyword is "
                        "deprecated, use "
                        f"{new_arg_name}={new_arg_value!r} instead."
                    )
                # 否则直接使用旧的参数值作为新的参数值
                else:
                    new_arg_value = old_arg_value
                    msg = (
                        f"the {old_arg_name!r} keyword is deprecated, "
                        f"use {new_arg_name!r} instead."
                    )

                # 发出警告，指示参数已废弃
                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                # 如果同时指定了新旧两个参数，则引发类型错误
                if kwargs.get(new_arg_name) is not None:
                    msg = (
                        f"Can only specify {old_arg_name!r} "
                        f"or {new_arg_name!r}, not both."
                    )
                    raise TypeError(msg)
                # 设置新的关键字参数值
                kwargs[new_arg_name] = new_arg_value
            # 返回原函数调用结果
            return func(*args, **kwargs)

        # 返回装饰后的函数
        return cast(F, wrapper)

    # 返回装饰器函数 _deprecate_kwarg 本身
    return _deprecate_kwarg
def _format_argument_list(allow_args: list[str]) -> str:
    """
    Convert the allow_args argument (either string or integer) of
    `deprecate_nonkeyword_arguments` function to a string describing
    it to be inserted into warning message.

    Parameters
    ----------
    allow_args : list of str
        The list of argument names for `deprecate_nonkeyword_arguments`,
        excluding 'self'.

    Returns
    -------
    str
        A string describing the argument list in the warning message.

    Examples
    --------
    `format_argument_list([])` -> ''
    `format_argument_list(['a'])` -> " except for the argument 'a'"
    `format_argument_list(['a', 'b'])` -> " except for the arguments 'a' and 'b'"
    `format_argument_list(['a', 'b', 'c'])` ->
        " except for the arguments 'a', 'b' and 'c'"
    """
    if "self" in allow_args:
        allow_args.remove("self")
    if not allow_args:
        return ""
    elif len(allow_args) == 1:
        return f" except for the argument '{allow_args[0]}'"
    else:
        last = allow_args[-1]
        args = ", ".join(["'" + x + "'" for x in allow_args[:-1]])
        return f" except for the arguments {args} and '{last}'"


def future_version_msg(version: str | None) -> str:
    """Specify which version of pandas the deprecation will take place in."""
    if version is None:
        return "In a future version of pandas"
    else:
        return f"Starting with pandas version {version}"


def deprecate_nonkeyword_arguments(
    version: str | None,
    allowed_args: list[str] | None = None,
    name: str | None = None,
) -> Callable[[F], F]:
    """
    Decorator to deprecate a use of non-keyword arguments of a function.

    Parameters
    ----------
    version : str or None, optional
        The version in which positional arguments will become
        keyword-only. If None, then the warning message won't
        specify any particular version.

    allowed_args : list of str or None, optional
        If specified, it lists the names of the first arguments
        of decorated functions allowed as positional arguments.

    name : str or None, optional
        The name of the function to display in the warning message.
        If None, the Qualified name of the function is used.
    """
    `
    # 定义一个装饰器函数，接受一个函数作为参数，并返回一个装饰后的函数
    def decorate(func):
        # 获取被装饰函数的旧参数签名
        old_sig = inspect.signature(func)
    
        # 如果有指定允许的参数列表，则使用该列表，否则根据旧参数签名确定允许的参数
        if allowed_args is not None:
            allow_args = allowed_args
        else:
            allow_args = [
                p.name
                for p in old_sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]
    
        # 将旧参数签名中不在允许列表内的参数改为仅限关键字参数
        new_params = [
            p.replace(kind=p.KEYWORD_ONLY)
            if (
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.name not in allow_args
            )
            else p
            for p in old_sig.parameters.values()
        ]
    
        # 根据修改后的参数列表创建新的函数签名
        new_params.sort(key=lambda p: p.kind)
        new_sig = old_sig.replace(parameters=new_params)
    
        # 提示将来版本中该函数所有参数都将要求使用关键字参数
        num_allow_args = len(allow_args)
        msg = (
            f"{future_version_msg(version)} all arguments of "
            f"{name or func.__qualname__}{{arguments}} will be keyword-only."
        )
    
        # 定义装饰后的函数的包装器
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 如果传入的位置参数数量超过允许的参数数量，发出警告
            if len(args) > num_allow_args:
                warnings.warn(
                    msg.format(arguments=_format_argument_list(allow_args)),
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            # 调用原始函数并返回其结果
            return func(*args, **kwargs)
    
        # 设置包装器函数的签名为修改后的新签名
        wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
        return wrapper
def doc(*docstrings: None | str | Callable, **params: object) -> Callable[[F], F]:
    """
    A decorator to take docstring templates, concatenate them and perform string
    substitution on them.

    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : None, str, or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: F) -> F:
        # collecting docstring and docstring templates
        docstring_components: list[str | Callable] = []
        # 如果被装饰函数已有文档字符串，则将其添加到组件列表中
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        # 遍历传入的各个文档字符串参数
        for docstring in docstrings:
            if docstring is None:
                continue
            # 如果参数是一个带有 "_docstring_components" 属性的对象，则扩展组件列表
            if hasattr(docstring, "_docstring_components"):
                docstring_components.extend(
                    docstring._docstring_components  # pyright: ignore[reportAttributeAccessIssue]
                )
            # 否则，如果参数是字符串或带有文档字符串，则直接添加到组件列表
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        # 对参数进行格式化替换
        params_applied = [
            component.format(**params)
            if isinstance(component, str) and len(params) > 0
            else component
            for component in docstring_components
        ]

        # 将所有组件合并成一个完整的文档字符串并赋值给被装饰函数的 __doc__ 属性
        decorated.__doc__ = "".join(
            [
                component
                if isinstance(component, str)
                else dedent(component.__doc__ or "")
                for component in params_applied
            ]
        )

        # 错误： "F" has no attribute "_docstring_components"
        # 添加一个属性 "_docstring_components" 到被装饰函数，用于存储文档字符串组件
        decorated._docstring_components = (  # type: ignore[attr-defined]
            docstring_components
        )
        return decorated

    return decorator
    One can also use positional arguments.

    sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    @sub_first_last_names
    def some_function(x):
        "%s %s wrote the Raven"
    """



    # 定义一个接受任意参数的初始化方法
    def __init__(self, *args, **kwargs) -> None:
        # 如果既有位置参数又有关键字参数，则抛出断言错误
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")

        # 使用传入的参数来初始化 self.params
        self.params = args or kwargs

    # 定义一个装饰器，用于替换函数的文档字符串中的占位符
    def __call__(self, func: F) -> F:
        # 如果函数有文档字符串，将其格式化替换为参数化后的字符串
        func.__doc__ = func.__doc__ and func.__doc__ % self.params
        return func

    # 定义一个方法，用于更新 self.params 的值
    def update(self, *args, **kwargs) -> None:
        """
        Update self.params with supplied args.
        """
        # 如果 self.params 是字典类型，则更新其值
        if isinstance(self.params, dict):
            self.params.update(*args, **kwargs)
class Appender:
    """
    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    add_copyright = Appender("Copyright (c) 2009", join='\\n')

    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """

    addendum: str | None

    def __init__(self, addendum: str | None, join: str = "", indents: int = 0) -> None:
        # 如果需要缩进，对附加文本进行缩进处理
        if indents > 0:
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func: T) -> T:
        # 确保函数文档字符串不为 None
        func.__doc__ = func.__doc__ if func.__doc__ else ""
        self.addendum = self.addendum if self.addendum else ""
        # 将函数原始文档字符串和附加文本按照指定的连接符连接起来，并进行缩进处理
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = dedent(self.join.join(docitems))
        return func


def indent(text: str | None, indents: int = 1) -> str:
    """
    Indents the given text by adding spaces to the beginning of each line.

    Args:
        text: The input text to be indented.
        indents: Number of spaces to add as indentation.

    Returns:
        The indented text.
    """
    if not text or not isinstance(text, str):
        return ""
    jointext = "".join(["\\n"] + ["    "] * indents)
    return jointext.join(text.split("\\n"))


__all__ = [
    "Appender",
    "cache_readonly",
    "deprecate",
    "deprecate_kwarg",
    "deprecate_nonkeyword_arguments",
    "doc",
    "future_version_msg",
    "Substitution",
]


def set_module(module) -> Callable[[F], F]:
    """Private decorator for overriding __module__ on a function or class.

    Example usage::

        @set_module("pandas")
        def example():
            pass


        assert example.__module__ == "pandas"
    """
    # 返回一个装饰器函数，用于设置函数或类的 __module__ 属性
    def decorator(func: F) -> F:
        if module is not None:
            func.__module__ = module
        return func

    return decorator
```