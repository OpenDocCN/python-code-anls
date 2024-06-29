# `D:\src\scipysrc\matplotlib\lib\matplotlib\_api\deprecation.py`

```
# 导入上下文管理、函数装饰器、检查函数结构、数学运算和警告模块
import contextlib
import functools
import inspect
import math
import warnings


class MatplotlibDeprecationWarning(DeprecationWarning):
    """用于发出 Matplotlib 用户使用的过时警告的类."""


def _generate_deprecation_warning(
        since, message='', name='', alternative='', pending=False, obj_type='',
        addendum='', *, removal=''):
    # 如果是待弃用状态，确保没有预定的移除时间
    if pending:
        if removal:
            raise ValueError(
                "A pending deprecation cannot have a scheduled removal")
    else:
        # 如果不是待弃用状态，计算默认的移除版本
        if not removal:
            macro, meso, *_ = since.split('.')
            removal = f'{macro}.{int(meso) + 2}'
        removal = f"in {removal}"
    
    # 如果没有指定消息，生成默认的过时警告消息
    if not message:
        message = (
            ("The %(name)s %(obj_type)s" if obj_type else "%(name)s")
            + (" will be deprecated in a future version"
               if pending else
               " was deprecated in Matplotlib %(since)s and will be removed %(removal)s"
               )
            + "."
            + (" Use %(alternative)s instead." if alternative else "")
            + (" %(addendum)s" if addendum else ""))
    
    # 根据是否是待弃用状态选择警告类型
    warning_cls = (PendingDeprecationWarning if pending
                   else MatplotlibDeprecationWarning)
    
    # 返回生成的警告实例
    return warning_cls(message % dict(
        func=name, name=name, obj_type=obj_type, since=since, removal=removal,
        alternative=alternative, addendum=addendum))


def warn_deprecated(
        since, *, message='', name='', alternative='', pending=False,
        obj_type='', addendum='', removal=''):
    """
    显示标准化的过时警告信息.

    Parameters
    ----------
    since : str
        API 弃用的版本信息.
    message : str, optional
        覆盖默认的过时警告消息.
        ``%(since)s``, ``%(name)s``, ``%(alternative)s``, ``%(obj_type)s``, ``%(addendum)s``,
        和 ``%(removal)s`` 格式说明符将会被传递给函数的相应参数值所替换.
    name : str, optional
        被弃用对象的名称.
    alternative : str, optional
        用户可以替代被弃用 API 的另一个 API.
        如果提供了，过时警告将会告知用户此替代选择.
    pending : bool, optional
        如果为 True，则使用 PendingDeprecationWarning 而不是 DeprecationWarning.
        不能与 *removal* 一起使用.
    obj_type : str, optional
        被弃用对象的类型.
    addendum : str, optional
        直接附加到最终消息的附加文本.
    removal : str, optional
        API 弃用的移除时间信息.
    """
    # removal参数用于指定预期的移除版本。默认情况下（空字符串），移除版本将自动从since参数计算得出。
    # 设置为其他假值将不会安排移除日期。不能与pending参数同时使用。
    Examples
    --------
    ::
    
        # 警告"matplotlib.name_of_module"模块即将被弃用
        warn_deprecated('1.4.0', name='matplotlib.name_of_module',
                        obj_type='module')
# 定义一个装饰器函数，用于标记函数、类或属性已被弃用
def deprecated(since, *, message='', name='', alternative='', pending=False,
               obj_type=None, addendum='', removal=''):
    """
    Decorator to mark a function, a class, or a property as deprecated.

    When deprecating a classmethod, a staticmethod, or a property, the
    ``@deprecated`` decorator should go *under* ``@classmethod`` and
    ``@staticmethod`` (i.e., `deprecated` should directly decorate the
    underlying callable), but *over* ``@property``.

    When deprecating a class ``C`` intended to be used as a base class in a
    multiple inheritance hierarchy, ``C`` *must* define an ``__init__`` method
    (if ``C`` instead inherited its ``__init__`` from its own base class, then
    ``@deprecated`` would mess up ``__init__`` inheritance when installing its
    own (deprecation-emitting) ``C.__init__``).

    Parameters are the same as for `warn_deprecated`, except that *obj_type*
    defaults to 'class' if decorating a class, 'attribute' if decorating a
    property, and 'function' otherwise.

    Examples
    --------
    ::

        @deprecated('1.4.0')
        def the_function_to_deprecate():
            pass
    """
    
    # 返回一个名为deprecate的函数对象，用于实际执行标记弃用的操作
    return deprecate


class deprecate_privatize_attribute:
    """
    Helper to deprecate public access to an attribute (or method).

    This helper should only be used at class scope, as follows::

        class Foo:
            attr = _deprecate_privatize_attribute(*args, **kwargs)

    where *all* parameters are forwarded to `deprecated`.  This form makes
    ``attr`` a property which forwards read and write access to ``self._attr``
    (same name but with a leading underscore), with a deprecation warning.
    Note that the attribute name is derived from *the name this helper is
    assigned to*.  This helper also works for deprecating methods.
    """

    # 初始化方法，将所有参数传递给deprecated函数，并保存返回的函数对象于self.deprecator
    def __init__(self, *args, **kwargs):
        self.deprecator = deprecated(*args, **kwargs)

    # 设置名称方法，将属性名设置为self.deprecator的返回值，该返回值是一个装饰过的属性对象
    def __set_name__(self, owner, name):
        setattr(owner, name, self.deprecator(
            # 使用lambda函数创建一个property对象，用于访问以"_"开头的属性
            property(lambda self: getattr(self, f"_{name}"),
                     lambda self, value: setattr(self, f"_{name}", value)),
            name=name))


# Used by _copy_docstring_and_deprecators to redecorate pyplot wrappers and
# boilerplate.py to retrieve original signatures.  It may seem natural to store
# this information as an attribute on the wrapper, but if the wrapper gets
# itself functools.wraps()ed, then such attributes are silently propagated to
# the outer wrapper, which is not desired.
# 定义一个空字典，用于存储装饰器信息，供其他函数使用
DECORATORS = {}


def rename_parameter(since, old, new, func=None):
    """
    Decorator indicating that parameter *old* of *func* is renamed to *new*.

    The actual implementation of *func* should use *new*, not *old*.  If *old*
    is passed to *func*, a DeprecationWarning is emitted, and its value is
    used, even if *new* is also passed by keyword (this is to simplify pyplot
    wrapper functions, which always pass *new* explicitly to the Axes method).
    """
    # 创建一个部分函数，用于重命名参数的装饰器，固定了since, old, new三个参数
    decorator = functools.partial(rename_parameter, since, old, new)

    # 如果函数为空，则直接返回装饰器本身
    if func is None:
        return decorator

    # 获取函数的签名信息
    signature = inspect.signature(func)
    
    # 断言旧参数名不在函数签名中，否则抛出异常
    assert old not in signature.parameters, (
        f"Matplotlib internal error: {old!r} cannot be a parameter for "
        f"{func.__name__}()")
    
    # 断言新参数名在函数签名中，否则抛出异常
    assert new in signature.parameters, (
        f"Matplotlib internal error: {new!r} must be a parameter for "
        f"{func.__name__()}")

    # 定义一个装饰后的函数，用于处理参数重命名的逻辑
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 如果旧参数名在关键字参数中，则发出警告，并将其改为新参数名
        if old in kwargs:
            warn_deprecated(
                since, message=f"The {old!r} parameter of {func.__name__}() "
                f"has been renamed {new!r} since Matplotlib {since}; support "
                f"for the old name will be dropped %(removal)s.")
            kwargs[new] = kwargs.pop(old)
        # 调用原始函数，传入更新后的参数
        return func(*args, **kwargs)

    # 将装饰后的函数记录到全局的装饰器字典中
    DECORATORS[wrapper] = decorator
    
    # 返回装饰后的函数
    return wrapper
class _deprecated_parameter_class:
    # 一个用于表示被废弃参数的类的定义
    def __repr__(self):
        # 返回一个描述被废弃参数的字符串
        return "<deprecated parameter>"


# 创建一个实例化的被废弃参数对象
_deprecated_parameter = _deprecated_parameter_class()


def delete_parameter(since, name, func=None, **kwargs):
    """
    Decorator indicating that parameter *name* of *func* is being deprecated.

    The actual implementation of *func* should keep the *name* parameter in its
    signature, or accept a ``**kwargs`` argument (through which *name* would be
    passed).

    Parameters that come after the deprecated parameter effectively become
    keyword-only (as they cannot be passed positionally without triggering the
    DeprecationWarning on the deprecated parameter), and should be marked as
    such after the deprecation period has passed and the deprecated parameter
    is removed.

    Parameters other than *since*, *name*, and *func* are keyword-only and
    forwarded to `.warn_deprecated`.

    Examples
    --------
    ::

        @_api.delete_parameter("3.1", "unused")
        def func(used_arg, other_arg, unused, more_args): ...
    """

    # 创建一个偏函数装饰器，固定部分参数以便传递给delete_parameter函数
    decorator = functools.partial(delete_parameter, since, name, **kwargs)

    # 如果没有提供func，则返回偏函数装饰器
    if func is None:
        return decorator

    # 获取函数func的签名
    signature = inspect.signature(func)
    # 获取装饰后函数中用于接收**kwargs参数的参数名，默认为"kwargs"
    kwargs_name = next((param.name for param in signature.parameters.values()
                        if param.kind == inspect.Parameter.VAR_KEYWORD), None)
    # 如果被废弃参数name在函数签名中
    if name in signature.parameters:
        # 获取参数的类型
        kind = signature.parameters[name].kind
        is_varargs = kind is inspect.Parameter.VAR_POSITIONAL
        is_varkwargs = kind is inspect.Parameter.VAR_KEYWORD
        # 如果被废弃参数不是可变位置参数或可变关键字参数
        if not is_varargs and not is_varkwargs:
            # 设置被废弃参数的默认值为_deprecated_parameter对象
            name_idx = (
                # 如果参数类型是仅关键字参数，则被废弃参数不能通过位置传递
                math.inf if kind is inspect.Parameter.KEYWORD_ONLY
                # 如果调用方的参数个数不超过这个索引，则被废弃参数不能通过位置传递
                else [*signature.parameters].index(name))
            # 替换函数签名，将被废弃参数的默认值替换为_deprecated_parameter对象
            func.__signature__ = signature = signature.replace(parameters=[
                param.replace(default=_deprecated_parameter)
                if param.name == name else param
                for param in signature.parameters.values()])
        else:
            # 被废弃参数可以始终通过位置传递
            name_idx = -1
    else:
        is_varargs = is_varkwargs = False
        # 被废弃参数不能通过位置传递
        name_idx = math.inf
        # 断言检查kwargs_name是否存在，如果不存在则抛出错误
        assert kwargs_name, (
            f"Matplotlib internal error: {name!r} must be a parameter for "
            f"{func.__name__}()")

    # 从kwargs中弹出'addendum'键的值，如果没有则为None
    addendum = kwargs.pop('addendum', None)

    @functools.wraps(func)
    def wrapper(*inner_args, **inner_kwargs):
        # 检查是否参数个数小于等于 name_idx，并且 inner_kwargs 中没有 name 对应的键
        if len(inner_args) <= name_idx and name not in inner_kwargs:
            # 在简单且非过时的情况下，早期返回（比调用 bind() 快得多）。
            return func(*inner_args, **inner_kwargs)
        
        # 使用传入的参数和关键字参数创建一个参数绑定对象
        arguments = signature.bind(*inner_args, **inner_kwargs).arguments
        
        # 如果是可变位置参数，并且 arguments 中包含 name 对应的参数
        if is_varargs and arguments.get(name):
            # 发出警告，说明位置参数已被弃用
            warn_deprecated(
                since, message=f"Additional positional arguments to "
                f"{func.__name__}() are deprecated since %(since)s and "
                f"support for them will be removed %(removal)s.")
        
        # 如果是可变关键字参数，并且 arguments 中包含 name 对应的参数
        elif is_varkwargs and arguments.get(name):
            # 发出警告，说明关键字参数已被弃用
            warn_deprecated(
                since, message=f"Additional keyword arguments to "
                f"{func.__name__}() are deprecated since %(since)s and "
                f"support for them will be removed %(removal)s.")
        
        # 对于 pyplot 包装器，不能仅仅检查 `name not in arguments`，因为它总是显式地传递所有参数。
        elif any(name in d and d[name] != _deprecated_parameter
                 for d in [arguments, arguments.get(kwargs_name, {})]):
            # 如果任何参数跟在 name 后面且不是 _deprecated_parameter，它们应该作为关键字而不是位置参数传递。
            deprecation_addendum = (
                f"If any parameter follows {name!r}, they should be passed as "
                f"keyword, not positionally.")
            # 发出警告，说明参数已被弃用
            warn_deprecated(
                since,
                name=repr(name),
                obj_type=f"parameter of {func.__name__}()",
                addendum=(addendum + " " + deprecation_addendum) if addendum
                         else deprecation_addendum,
                **kwargs)
        
        # 调用原始函数并返回其结果
        return func(*inner_args, **inner_kwargs)

    # 将 wrapper 函数与对应的 decorator 关联存储到 DECORATORS 字典中
    DECORATORS[wrapper] = decorator
    # 返回 wrapper 函数作为装饰器的结果
    return wrapper
# 创建一个装饰器函数，用于指示将参数 *name*（或其后的任何参数）作为位置参数传递给 *func* 是不推荐的做法。
# 当应用于具有 pyplot 包装器的方法时，应将其作为最外层装饰器使用，以便 :file:`boilerplate.py` 可以访问原始签名。
def make_keyword_only(since, name, func=None):
    # 创建一个部分应用了 make_keyword_only 函数的 decorator 函数对象
    decorator = functools.partial(make_keyword_only, since, name)

    # 如果 func 为 None，则返回 decorator 函数
    if func is None:
        return decorator

    # 获取 func 函数的签名信息
    signature = inspect.signature(func)
    POK = inspect.Parameter.POSITIONAL_OR_KEYWORD
    KWO = inspect.Parameter.KEYWORD_ONLY

    # 断言参数 name 存在于函数签名中且其类型为 POSITIONAL_OR_KEYWORD
    assert (name in signature.parameters
            and signature.parameters[name].kind == POK), (
        f"Matplotlib internal error: {name!r} must be a positional-or-keyword "
        f"parameter for {func.__name__}(). If this error happens on a function with a "
        f"pyplot wrapper, make sure make_keyword_only() is the outermost decorator.")

    # 获取函数参数列表
    names = [*signature.parameters]
    name_idx = names.index(name)

    # 获取参数列表中从 name 开始到末尾的参数，且其类型为 POSITIONAL_OR_KEYWORD 的参数名列表
    kwonly = [name for name in names[name_idx:]
              if signature.parameters[name].kind == POK]

    # 创建一个装饰后的函数 wrapper，用于替代原始函数 func
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 在这里不使用 signature.bind，因为当与 rename_parameter 一起使用时可能会失败，
        # 如果传递了“旧”参数名（signature.bind 会失败，但实际调用将成功）。
        if len(args) > name_idx:
            # 如果传入的位置参数个数大于 name 的索引位置，则发出废弃警告
            warn_deprecated(
                since, message="Passing the %(name)s %(obj_type)s "
                "positionally is deprecated since Matplotlib %(since)s; the "
                "parameter will become keyword-only %(removal)s.",
                name=name, obj_type=f"parameter of {func.__name__}()")
        # 调用原始函数 func，并将所有参数传递给它
        return func(*args, **kwargs)

    # 不修改 func 的签名，因为 boilerplate.py 需要它
    # 替换 wrapper 函数的签名参数列表，将原始参数列表中的某些参数改为 KEYWORD_ONLY 类型的参数
    wrapper.__signature__ = signature.replace(parameters=[
        param.replace(kind=KWO) if param.name in kwonly else param
        for param in signature.parameters.values()])
    
    # 将装饰后的 wrapper 函数和 decorator 函数关联存储在 DECORATORS 字典中
    DECORATORS[wrapper] = decorator

    # 返回装饰后的 wrapper 函数
    return wrapper


# 返回带有过时警告的 obj.method，如果它被重写则返回 None
# Parameters 参数列表中至少包含一个“since”键
def deprecate_method_override(method, obj, *, allow_empty=False, **kwargs):
    # 定义一个空函数
    def empty(): pass
    # 定义一个没有实现内容的函数，并且带有文档字符串 "doc"
    def empty_with_docstring(): """doc"""

    # 获取方法的名称
    name = method.__name__

    # 获取对象 obj 中的名为 name 的属性
    bound_child = getattr(obj, name)

    # 如果 obj 是一个类，并且 bound_child 是一个类型为 type(empty) 的实例，
    # 那么我们需要使用未绑定的方法（unbound methods）。
    # 否则，我们使用 method.__get__(obj) 来获取绑定的方法。
    bound_base = (
        method  # 如果 obj 是一个类，那么我们需要使用未绑定的方法。
        if isinstance(bound_child, type(empty)) and isinstance(obj, type)
        else method.__get__(obj))

    # 如果 bound_child 不等于 bound_base，并且不允许返回空（allow_empty 为 False），
    # 或者 bound_child 的代码不是空函数或带有文档字符串的空函数，
    # 则发出弃用警告。
    if (bound_child != bound_base
            and (not allow_empty
                 or (getattr(getattr(bound_child, "__code__", None),
                             "co_code", None)
                     not in [empty.__code__.co_code,
                             empty_with_docstring.__code__.co_code]))):
        warn_deprecated(**{"name": name, "obj_type": "method", **kwargs})
        # 返回 bound_child，因为它被标记为弃用
        return bound_child

    # 否则返回 None，表示未找到弃用的情况
    return None
# 定义一个上下文管理器函数，用于抑制 Matplotlib 废弃警告
@contextlib.contextmanager
def suppress_matplotlib_deprecation_warning():
    # 使用 warnings 模块捕获警告
    with warnings.catch_warnings():
        # 设置简单过滤器，忽略特定的 Matplotlib 废弃警告
        warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
        # 通过 yield 关键字将控制权传递给调用者，允许执行被保护的代码块
        yield
```