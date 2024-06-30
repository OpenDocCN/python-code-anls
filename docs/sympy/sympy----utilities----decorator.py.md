# `D:\src\scipysrc\sympy\sympy\utilities\decorator.py`

```
# 导入必要的模块和函数
import sys
import types
import inspect
from functools import wraps, update_wrapper

# 导入 sympy 库中的异常处理模块
from sympy.utilities.exceptions import sympy_deprecation_warning

# 定义一个工厂函数，用于创建“threaded”装饰器
def threaded_factory(func, use_add):
    # 导入 sympy 中的一些模块和函数
    from sympy.core import sympify
    from sympy.matrices import MatrixBase
    from sympy.utilities.iterables import iterable
    
    # 定义被装饰后的函数，使用 @wraps(func) 来保留原函数的元数据
    @wraps(func)
    def threaded_func(expr, *args, **kwargs):
        # 如果表达式是 MatrixBase 类型，则逐个应用 func 到其中的元素
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(lambda f: func(f, *args, **kwargs))
        # 如果表达式是可迭代的，则尝试将 func 应用到每个元素
        elif iterable(expr):
            try:
                return expr.__class__([func(f, *args, **kwargs) for f in expr])
            except TypeError:
                return expr
        # 否则，将表达式转换为 sympy 的表达式
        else:
            expr = sympify(expr)

            # 如果 use_add 为 True，并且表达式是加法类型，则应用 func 到每个元素
            if use_add and expr.is_Add:
                return expr.__class__(*[ func(f, *args, **kwargs) for f in expr.args ])
            # 如果表达式是关系类型，则分别应用 func 到左右两边
            elif expr.is_Relational:
                return expr.__class__(func(expr.lhs, *args, **kwargs),
                                      func(expr.rhs, *args, **kwargs))
            # 否则，直接应用 func 到整个表达式
            else:
                return func(expr, *args, **kwargs)

    return threaded_func


# 定义 threaded 装饰器，用于处理对象的子元素，包括 Add 类型
def threaded(func):
    """Apply ``func`` to sub--elements of an object, including :class:`~.Add`.

    This decorator is intended to make it uniformly possible to apply a
    function to all elements of composite objects, e.g. matrices, lists, tuples
    and other iterable containers, or just expressions.

    This version of :func:`threaded` decorator allows threading over
    elements of :class:`~.Add` class. If this behavior is not desirable
    use :func:`xthreaded` decorator.

    Functions using this decorator must have the following signature::

      @threaded
      def function(expr, *args, **kwargs):

    """
    return threaded_factory(func, True)


# 定义 xthreaded 装饰器，用于处理对象的子元素，但不包括 Add 类型
def xthreaded(func):
    """Apply ``func`` to sub--elements of an object, excluding :class:`~.Add`.

    This decorator is intended to make it uniformly possible to apply a
    function to all elements of composite objects, e.g. matrices, lists, tuples
    and other iterable containers, or just expressions.

    This version of :func:`threaded` decorator disallows threading over
    elements of :class:`~.Add` class. If this behavior is not desirable
    use :func:`threaded` decorator.

    Functions using this decorator must have the following signature::

      @xthreaded
      def function(expr, *args, **kwargs):

    """
    return threaded_factory(func, False)


# 定义 conserve_mpmath_dps 装饰器，用于在函数执行后恢复 mpmath.mp.dps 的值
def conserve_mpmath_dps(func):
    """After the function finishes, resets the value of ``mpmath.mp.dps`` to
    the value it had before the function was run."""
    import mpmath
    
    # 定义函数包装器 func_wrapper，保存当前 mpmath.mp.dps 的值，并在执行后恢复
    def func_wrapper(*args, **kwargs):
        dps = mpmath.mp.dps
        try:
            return func(*args, **kwargs)
        finally:
            mpmath.mp.dps = dps

    # 使用 update_wrapper 将 func_wrapper 的元数据更新为 func 的元数据
    func_wrapper = update_wrapper(func_wrapper, func)
    return func_wrapper
class no_attrs_in_subclass:
    """Don't 'inherit' certain attributes from a base class

    >>> from sympy.utilities.decorator import no_attrs_in_subclass

    >>> class A(object):
    ...     x = 'test'

    >>> A.x = no_attrs_in_subclass(A, A.x)

    >>> class B(A):
    ...     pass

    >>> hasattr(A, 'x')
    True
    >>> hasattr(B, 'x')
    False

    """
    # 初始化方法，接受类和属性作为参数
    def __init__(self, cls, f):
        self.cls = cls
        self.f = f

    # 获取属性的方法，用于检查是否在子类中继承
    def __get__(self, instance, owner=None):
        if owner == self.cls:
            # 如果属性属于当前类，则返回属性本身或其绑定方法
            if hasattr(self.f, '__get__'):
                return self.f.__get__(instance, owner)
            return self.f
        # 如果属性不属于当前类，则引发属性错误
        raise AttributeError


def doctest_depends_on(exe=None, modules=None, disable_viewers=None,
                       python_version=None, ground_types=None):
    """
    Adds metadata about the dependencies which need to be met for doctesting
    the docstrings of the decorated objects.

    ``exe`` should be a list of executables

    ``modules`` should be a list of modules

    ``disable_viewers`` should be a list of viewers for :func:`~sympy.printing.preview.preview` to disable

    ``python_version`` should be the minimum Python version required, as a tuple
    (like ``(3, 0)``)
    """
    # 初始化依赖字典
    dependencies = {}
    # 如果有可执行文件列表，添加到依赖字典中
    if exe is not None:
        dependencies['executables'] = exe
    # 如果有模块列表，添加到依赖字典中
    if modules is not None:
        dependencies['modules'] = modules
    # 如果有需要禁用的预览视图列表，添加到依赖字典中
    if disable_viewers is not None:
        dependencies['disable_viewers'] = disable_viewers
    # 如果有最低要求的 Python 版本，添加到依赖字典中
    if python_version is not None:
        dependencies['python_version'] = python_version
    # 如果有地面类型列表，添加到依赖字典中
    if ground_types is not None:
        dependencies['ground_types'] = ground_types

    # 定义一个用于跳过测试的函数
    def skiptests():
        from sympy.testing.runtests import DependencyError, SymPyDocTests, PyTestReporter  # 延迟导入
        r = PyTestReporter()
        t = SymPyDocTests(r, None)
        try:
            # 检查依赖项
            t._check_dependencies(**dependencies)
        except DependencyError:
            return True  # 跳过 doctest 测试
        else:
            return False  # 运行 doctest 测试

    # 定义一个依赖装饰器函数
    def depends_on_deco(fn):
        # 将依赖项元数据添加到函数对象
        fn._doctest_depends_on = dependencies
        fn.__doctest_skip__ = skiptests

        # 如果被装饰的对象是类，则使用 no_attrs_in_subclass 防止继承指定属性
        if inspect.isclass(fn):
            fn._doctest_depdends_on = no_attrs_in_subclass(
                fn, fn._doctest_depends_on)
            fn.__doctest_skip__ = no_attrs_in_subclass(
                fn, fn.__doctest_skip__)
        return fn

    return depends_on_deco


def public(obj):
    """
    Append ``obj``'s name to global ``__all__`` variable (call site).

    By using this decorator on functions or classes you achieve the same goal
    as by filling ``__all__`` variables manually, you just do not have to repeat
    yourself (object's name). You also know if object is public at definition
    site, not at some random location (where ``__all__`` was set).

    Note that in multiple decorator setup (in almost all cases) ``@public``

    """
    # 将对象的名称追加到全局 __all__ 变量中（调用位置）
    pass  # 该注释为了保持与原代码结构一致，这里无实际代码
    # 如果对象是函数类型，则获取其全局命名空间（global namespace）和对象的名称
    if isinstance(obj, types.FunctionType):
        ns = obj.__globals__
        name = obj.__name__
    # 如果对象是类类型或者元类（metaclass），则获取其所在模块的命名空间和对象的名称
    elif isinstance(obj, (type(type), type)):
        ns = sys.modules[obj.__module__].__dict__
        name = obj.__name__
    # 如果对象类型不是函数或类，则抛出类型错误异常
    else:
        raise TypeError("expected a function or a class, got %s" % obj)
    
    # 如果全局命名空间中不存在 "__all__"，则创建一个新的列表，其中包含当前对象的名称
    if "__all__" not in ns:
        ns["__all__"] = [name]
    # 如果 "__all__" 已经存在于全局命名空间中，则将当前对象的名称添加到列表中
    else:
        ns["__all__"].append(name)
    
    # 返回修饰后的对象
    return obj
def memoize_property(propfunc):
    """定义一个装饰器函数，用于缓存可能昂贵的属性计算结果。

    Args:
        propfunc: 要装饰的属性计算函数。

    Returns:
        property: 返回一个带缓存功能的属性访问器。
    """
    # 构造属性名，在函数名前加上下划线
    attrname = '_' + propfunc.__name__
    # 唯一性标识符，用于判断属性是否已经计算过
    sentinel = object()

    @wraps(propfunc)
    def accessor(self):
        # 尝试获取已缓存的属性值，如果没有则为唯一性标识符
        val = getattr(self, attrname, sentinel)
        # 如果属性值为唯一性标识符，说明还未计算过，计算并缓存结果
        if val is sentinel:
            val = propfunc(self)
            setattr(self, attrname, val)
        return val

    # 返回带缓存功能的属性访问器
    return property(accessor)


def deprecated(message, *, deprecated_since_version,
               active_deprecations_target, stacklevel=3):
    '''
    将一个函数标记为已废弃。

    该装饰器用于标记整个函数或类已废弃。如果只是某个功能已废弃，应直接使用
    :func:`~.warns_deprecated_sympy`。此装饰器只是一个方便的封装。
    在功能上，使用此装饰器与在函数顶部调用 ``warns_deprecated_sympy()`` 没有功能差异。

    Args:
        message (str): 废弃警告信息。
        deprecated_since_version (str): 废弃版本号。
        active_deprecations_target (str): 活跃废弃目标。
        stacklevel (int, optional): 警告栈级别，默认为 3。

    Examples
    ========

    >>> from sympy.utilities.decorator import deprecated
    >>> from sympy import simplify
    >>> @deprecated("""\
    ... The simplify_this(expr) function is deprecated. Use simplify(expr)
    ... instead.""", deprecated_since_version="1.1",
    ... active_deprecations_target='simplify-this-deprecation')
    ... def simplify_this(expr):
    ...     """
    ...     Simplify ``expr``.
    ...
    ...     .. deprecated:: 1.1
    ...
    ...        The ``simplify_this`` function is deprecated. Use :func:`simplify`
    ...        instead. See its documentation for more information. See
    ...        :ref:`simplify-this-deprecation` for details.
    ...
    ...     """
    ...     return simplify(expr)
    >>> from sympy.abc import x
    >>> simplify_this(x*(x + 1) - x**2) # doctest: +SKIP
    <stdin>:1: SymPyDeprecationWarning:
    <BLANKLINE>
    The simplify_this(expr) function is deprecated. Use simplify(expr)
    instead.
    <BLANKLINE>
    See https://docs.sympy.org/latest/explanation/active-deprecations.html#simplify-this-deprecation
    for details.
    <BLANKLINE>
    This has been deprecated since SymPy version 1.1. It
    will be removed in a future version of SymPy.
    <BLANKLINE>
      simplify_this(x)
    x

    See Also
    ========
    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.exceptions.ignore_warnings
    sympy.testing.pytest.warns_deprecated_sympy

    '''
    # 创建一个包含装饰器参数的字典，用于设置函数或类的版本信息和目标
    decorator_kwargs = {"deprecated_since_version": deprecated_since_version,
                        "active_deprecations_target": active_deprecations_target}
    
    # 定义一个装饰器函数，用于标记过时的函数或类
    def deprecated_decorator(wrapped):
        # 如果被装饰的对象是一个类
        if hasattr(wrapped, '__mro__'):
            # 定义一个新的类 wrapper，继承自 wrapped 类
            class wrapper(wrapped):
                # 复制原类的文档字符串和模块信息
                __doc__ = wrapped.__doc__
                __module__ = wrapped.__module__
                # 记录原始的被过时函数或类
                _sympy_deprecated_func = wrapped
                # 如果原始类有 '__new__' 方法
                if '__new__' in wrapped.__dict__:
                    # 定义一个新的 '__new__' 方法
                    def __new__(cls, *args, **kwargs):
                        # 发出 Sympy 过时警告
                        sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                        return super().__new__(cls, *args, **kwargs)
                else:
                    # 否则，定义一个新的 '__init__' 方法
                    def __init__(self, *args, **kwargs):
                        # 发出 Sympy 过时警告
                        sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                        super().__init__(*args, **kwargs)
            # 设置新类的名称与原类相同
            wrapper.__name__ = wrapped.__name__
        else:
            # 如果被装饰的是函数而不是类
            @wraps(wrapped)
            def wrapper(*args, **kwargs):
                # 发出 Sympy 过时警告
                sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                return wrapped(*args, **kwargs)
            # 记录原始的被过时函数
            wrapper._sympy_deprecated_func = wrapped
    
        # 返回装饰后的函数或类
        return wrapper
    
    # 返回装饰器函数 deprecated_decorator，用于标记过时的函数或类
    return deprecated_decorator
```