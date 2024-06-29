# `.\numpy\numpy\_core\overrides.py`

```py
# 导入必要的模块：collections, functools, os
# 导入自定义模块中的 set_module 和 getargspec 函数
"""Implementation of __array_function__ overrides from NEP-18."""
# 导入 numpy 的核心模块中的 _ArrayFunctionDispatcher 类
from numpy._core._multiarray_umath import (
    add_docstring, _get_implementing_args, _ArrayFunctionDispatcher
)

# 定义一个空集合 ARRAY_FUNCTIONS
ARRAY_FUNCTIONS = set()

# 描述了 like 参数的文档字符串模板
array_function_like_doc = (
    """like : array_like, optional
        Reference object to allow the creation of arrays which are not
        NumPy arrays. If an array-like passed in as ``like`` supports
        the ``__array_function__`` protocol, the result will be defined
        by it. In this case, it ensures the creation of an array object
        compatible with that passed in via this argument."""
)

# 替换公共 API 文档中的 ${ARRAY_FUNCTION_LIKE} 字符串为 array_function_like_doc
def set_array_function_like_doc(public_api):
    if public_api.__doc__ is not None:
        public_api.__doc__ = public_api.__doc__.replace(
            "${ARRAY_FUNCTION_LIKE}",
            array_function_like_doc,
        )
    return public_api

# 为 _ArrayFunctionDispatcher 类添加文档字符串
add_docstring(
    _ArrayFunctionDispatcher,
    """
    Class to wrap functions with checks for __array_function__ overrides.

    All arguments are required, and can only be passed by position.

    Parameters
    ----------
    dispatcher : function or None
        The dispatcher function that returns a single sequence-like object
        of all arguments relevant.  It must have the same signature (except
        the default values) as the actual implementation.
        If ``None``, this is a ``like=`` dispatcher and the
        ``_ArrayFunctionDispatcher`` must be called with ``like`` as the
        first (additional and positional) argument.
    implementation : function
        Function that implements the operation on NumPy arrays without
        overrides.  Arguments passed calling the ``_ArrayFunctionDispatcher``
        will be forwarded to this (and the ``dispatcher``) as if using
        ``*args, **kwargs``.

    Attributes
    ----------
    _implementation : function
        The original implementation passed in.
    """
)

# 为 _get_implementing_args 函数添加文档字符串
add_docstring(
    _get_implementing_args,
    """
    Collect arguments on which to call __array_function__.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of possibly array-like arguments to check for
        __array_function__ methods.

    Returns
    -------
    Sequence of arguments with __array_function__ methods, in the order in
    which they should be called.
    """
)

# 定义一个命名元组 ArgSpec，用于描述函数的参数信息
ArgSpec = collections.namedtuple('ArgSpec', 'args varargs keywords defaults')

# 定义一个函数 verify_matching_signatures 用于验证调度函数的签名是否匹配
def verify_matching_signatures(implementation, dispatcher):
    """Verify that a dispatcher function has the right signature."""
    # 获取实现函数和调度函数的参数规范
    implementation_spec = ArgSpec(*getargspec(implementation))
    dispatcher_spec = ArgSpec(*getargspec(dispatcher))
    # 检查实现和调度器函数的参数规范是否一致
    if (implementation_spec.args != dispatcher_spec.args or
            implementation_spec.varargs != dispatcher_spec.varargs or
            implementation_spec.keywords != dispatcher_spec.keywords or
            (bool(implementation_spec.defaults) !=
             bool(dispatcher_spec.defaults)) or
            (implementation_spec.defaults is not None and
             len(implementation_spec.defaults) !=
             len(dispatcher_spec.defaults))):
        # 如果存在参数规范不一致，则抛出运行时错误，指明实现和调度器函数的函数签名不同
        raise RuntimeError('implementation and dispatcher for %s have '
                           'different function signatures' % implementation)
    
    # 检查实现函数是否有默认参数
    if implementation_spec.defaults is not None:
        # 如果实现函数有默认参数，则进一步检查调度器函数的默认参数是否全为 None
        if dispatcher_spec.defaults != (None,) * len(dispatcher_spec.defaults):
            # 如果调度器函数的默认参数不全为 None，则抛出运行时错误，指明调度器函数只能使用 None 作为默认参数值
            raise RuntimeError('dispatcher functions can only use None for '
                               'default argument values')
# 定义一个装饰器函数，用于实现 __array_function__ 协议的分发
def array_function_dispatch(dispatcher=None, module=None, verify=True,
                            docs_from_dispatcher=False):
    """Decorator for adding dispatch with the __array_function__ protocol.

    See NEP-18 for example usage.

    Parameters
    ----------
    dispatcher : callable or None
        Function that when called like ``dispatcher(*args, **kwargs)`` with
        arguments from the NumPy function call returns an iterable of
        array-like arguments to check for ``__array_function__``.

        If `None`, the first argument is used as the single `like=` argument
        and not passed on.  A function implementing `like=` must call its
        dispatcher with `like` as the first non-keyword argument.
    module : str, optional
        __module__ attribute to set on new function, e.g., ``module='numpy'``.
        By default, module is copied from the decorated function.
    verify : bool, optional
        If True, verify the that the signature of the dispatcher and decorated
        function signatures match exactly: all required and optional arguments
        should appear in order with the same names, but the default values for
        all optional arguments should be ``None``. Only disable verification
        if the dispatcher's signature needs to deviate for some particular
        reason, e.g., because the function has a signature like
        ``func(*args, **kwargs)``.
    docs_from_dispatcher : bool, optional
        If True, copy docs from the dispatcher function onto the dispatched
        function, rather than from the implementation. This is useful for
        functions defined in C, which otherwise don't have docstrings.

    Returns
    -------
    Function suitable for decorating the implementation of a NumPy function.

    """
    # 内部装饰器函数，用于实际装饰待分发函数的实现
    def decorator(implementation):
        # 如果 verify 为 True，则验证分发器和被装饰函数的签名是否匹配
        if verify:
            if dispatcher is not None:
                verify_matching_signatures(implementation, dispatcher)
            else:
                # 直接使用 __code__ 来验证签名类似于 verify_matching_signature
                co = implementation.__code__
                last_arg = co.co_argcount + co.co_kwonlyargcount - 1
                last_arg = co.co_varnames[last_arg]
                # 检查最后一个参数是否为 "like"，且是否为关键字参数
                if last_arg != "like" or co.co_kwonlyargcount == 0:
                    raise RuntimeError(
                        "__array_function__ expects `like=` to be the last "
                        "argument and a keyword-only argument. "
                        f"{implementation} does not seem to comply.")

        # 如果 docs_from_dispatcher 为 True，则从分发器函数复制文档字符串到被分发函数
        if docs_from_dispatcher:
            add_docstring(implementation, dispatcher.__doc__)

        # 创建 _ArrayFunctionDispatcher 对象，将分发器和实现函数封装起来
        public_api = _ArrayFunctionDispatcher(dispatcher, implementation)
        # 使用 functools.wraps 将装饰器函数的属性复制到 public_api 上
        public_api = functools.wraps(implementation)(public_api)

        # 如果指定了 module，则设置 public_api 的 __module__ 属性
        if module is not None:
            public_api.__module__ = module

        # 将 public_api 添加到全局集合 ARRAY_FUNCTIONS 中
        ARRAY_FUNCTIONS.add(public_api)

        # 返回装饰后的 public_api 函数
        return public_api

    # 返回内部的装饰器函数
    return decorator
# 定义一个函数，用于生成特定的装饰器，其参数和 array_function_dispatcher 函数的顺序相反
def array_function_from_dispatcher(
        implementation, module=None, verify=True, docs_from_dispatcher=True):
    """Like array_function_dispatcher, but with function arguments flipped."""

    # 定义一个装饰器函数，接受一个调度函数作为参数
    def decorator(dispatcher):
        # 返回调用 array_function_dispatch 函数的结果，使用装饰器的参数和给定的实现函数
        return array_function_dispatch(
            dispatcher, module, verify=verify,
            docs_from_dispatcher=docs_from_dispatcher)(implementation)
    
    # 返回装饰器函数
    return decorator
```