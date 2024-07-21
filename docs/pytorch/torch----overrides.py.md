# `.\pytorch\torch\overrides.py`

```
# 装饰器，用于临时禁用给定模块中特定消息模式的 UserWarning 警告
def _disable_user_warnings(
    func: Callable,
    regex: str = ".*is deprecated, please use.*",
    module: str = "torch",
) -> Callable:
    # 包装函数，捕获警告并临时忽略特定模块中匹配给定正则表达式模式的 UserWarning 消息
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            # 过滤特定模块中的 UserWarning 类别和消息模式
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=regex, module=module
            )
            return func(*args, **kwargs)

    return wrapper


# 使用 functools.lru_cache(None) 装饰器缓存函数调用结果，避免重复计算
@functools.lru_cache(None)
# 使用上面定义的 _disable_user_warnings 装饰器，临时禁用特定模块中特定消息模式的 UserWarning
def get_ignored_functions() -> Set[Callable]:
    # 返回不能被 __torch_function__ 覆盖的公共函数集合
    """
    Return public functions that cannot be overridden by ``__torch_function__``.

    Returns
    -------
    set
        A set of functions.
    """
    ...
    # Set[Callable]
    # 定义一个类型提示为 Set[Callable] 的变量，表示这是一个包含函数的元组，这些函数是在 torch API 中公开的，但不能使用 `__torch_function__` 进行重写。
    # 大多数情况下，这是因为这些函数的参数既不是张量也不是类张量。

    # Examples
    # --------
    # 下面的示例展示了如何检查特定函数是否在 get_ignored_functions() 返回的集合中：
    # >>> torch.Tensor.as_subclass in torch.overrides.get_ignored_functions()
    # True
    # >>> torch.add in torch.overrides.get_ignored_functions()
    # False
    """
    # Tensor = torch.Tensor
    # 将 torch.Tensor 类型赋值给变量 Tensor
    }
# 使用 functools 模块中的 lru_cache 装饰器，对函数进行结果缓存，参数为 None 表示缓存无大小限制
@functools.lru_cache(None)
# 使用 _disable_user_warnings 装饰器修饰函数，可能用于在运行时禁用用户警告
def get_testing_overrides() -> Dict[Callable, Callable]:
    """Return a dict containing dummy overrides for all overridable functions

    返回包含所有可重写函数的虚拟重写的字典

    Returns
    -------
    Dict[Callable, Callable]
        返回一个字典，将 PyTorch API 中可重写的函数映射到具有相同签名的 lambda 函数，
        这些 lambda 函数无条件地返回 -1。这些 lambda 函数对于测试定义了 ``__torch_function__`` 的类型的 API 覆盖率非常有用。

    Examples
    --------
    >>> import inspect
    >>> my_add = torch.overrides.get_testing_overrides()[torch.add]
    >>> inspect.signature(my_add)
    <Signature (input, other, out=None)>
    """
    # 创建一个空的字典，用于存储函数的虚拟重写
    ret = {}

    # 获取当前 Tensor 类型的引用
    Tensor = torch.Tensor

    # 检查是否存在私有使用的后端名称，如果存在，则加入虚拟重写字典中
    privateuse1_backend_name = (
        torch.utils.backend_registration._privateuse1_backend_name
    )
    if hasattr(Tensor, privateuse1_backend_name):
        ret[
            getattr(Tensor, privateuse1_backend_name)
        ] = lambda self, device=None, non_blocking=False, **kwargs: -1
        ret[
            getattr(Tensor, f"is_{privateuse1_backend_name}").__get__
        ] = lambda self: -1  # noqa: B009

    # 创建另一个空字典，用于存储被忽略的函数
    ret2 = {}

    # 调用 get_ignored_functions 函数，获取被忽略的函数列表
    ignored = get_ignored_functions()
    for k, v in ret.items():
        # 遍历返回字典中的键值对，其中 k 是方法，v 是方法对应的值

        # 生成默认方法（如 __add__）及其就地操作方法（如 add_）的方法名列表
        names = [
            k.__name__,            # 默认方法名
            k.__name__ + "_",      # 就地操作方法名
            "__" + k.__name__ + "__",      # 双下划线方法名（dunder method）
            "__i" + k.__name__ + "__",     # 就地双下划线方法名
            "__r" + k.__name__ + "__",     # 反向双下划线方法名
        ]

        if k.__name__.startswith("bitwise_"):
            # 如果方法名以 "bitwise_" 开头，则生成位操作的双下划线方法名
            subname = k.__name__[len("bitwise_") :]
            names.extend(
                ["__" + subname + "__", "__i" + subname + "__", "__r" + subname + "__"]
            )

        # 遍历方法名列表，获取对应的方法对象
        for name in names:
            func = getattr(Tensor, name, None)
            # 如果获取到的方法是可调用的，并且不在 ret 中，也不在 ignored 集合中
            if callable(func) and func not in ret and func not in ignored:
                # 将方法及其值加入到 ret2 字典中
                ret2[func] = v

    # 将 ret2 字典中的内容更新到 ret 字典中
    ret.update(ret2)
    # 返回更新后的 ret 字典
    return ret
# 定义一个装饰器函数，用于包装传入的函数以支持 __torch_function__ 功能
def wrap_torch_function(dispatcher: Callable):
    """Wraps a given function with ``__torch_function__`` -related functionality.

    Parameters
    ----------
    dispatcher: Callable
        A callable that returns an iterable of Tensor-likes passed into the function.

    Note
    ----
    This decorator may reduce the performance of your code. Generally, it's enough to express
    your code as a series of functions that, themselves, support __torch_function__. If you
    find yourself in the rare situation where this is not the case, e.g. if you're wrapping a
    low-level library and you also need it to work for Tensor-likes, then this function is available.

    Examples
    --------
    >>> def dispatcher(a): # Must have the same signature as func
    ...     return (a,)
    >>> @torch.overrides.wrap_torch_function(dispatcher)
    >>> def func(a): # This will make func dispatchable by __torch_function__
    ...     return a + 0
    """

    # 内部函数 inner，实际上是一个装饰器，接受被装饰的函数 func
    def inner(func):
        @functools.wraps(func)
        # 包装函数 wrapped，处理 __torch_function__ 相关逻辑
        def wrapped(*args, **kwargs):
            # 调用 dispatcher 函数获取与被装饰函数相关的参数
            relevant_args = dispatcher(*args, **kwargs)
            # 检查相关参数是否支持 __torch_function__
            if has_torch_function(relevant_args):
                # 调用 handle_torch_function 处理 __torch_function__ 相关逻辑
                return handle_torch_function(wrapped, relevant_args, *args, **kwargs)

            # 如果相关参数不支持 __torch_function__，则直接调用原始函数
            return func(*args, **kwargs)

        return wrapped

    return inner


# 定义一个函数，用于获取需要调用 __torch_function__ 的参数列表
def _get_overloaded_args(
    relevant_args: Iterable[Any],
    get_type_fn: Callable[[Any], Type] = None,
) -> List[Any]:
    """Returns a list of arguments on which to call __torch_function__.

    Checks arguments in relevant_args for __torch_function__ implementations,
    storing references to the arguments and their types in overloaded_args and
    overloaded_types in order of calling precedence. Only distinct types are
    considered. If a type is a subclass of another type it will have higher
    precedence, otherwise the precedence order is the same as the order of
    arguments in relevant_args, that is, from left-to-right in the argument list.

    The precedence-determining algorithm implemented in this function is
    described in `NEP-0018`_.

    See torch::append_overloaded_arg for the equivalent function in the C++
    implementation.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of array-like arguments to check for __torch_function__
        methods.

    get_type_fn : callable, optional
        Function to call on each argument in relevant_args to get its type.

    Returns
    -------
    overloaded_args : list
        Arguments from relevant_args on which to call __torch_function__
        methods, in the order in which they should be called.

    .. _NEP-0018:
       https://numpy.org/neps/nep-0018-array-function-protocol.html
    """
    # 如果没有提供 get_type_fn，则默认使用 type 函数获取类型信息

    if get_type_fn is None:
        get_type_fn = type

    # 如果未启用 torch function，则返回空列表，不执行后续逻辑
    if not torch._C._is_torch_function_enabled():
        return []

    # 运行时复杂度为 O(num_arguments * num_unique_types)
    # 初始化一个空的集合，用于存储重载函数的类型
    overloaded_types: Set[Type] = set()
    # 初始化一个空的列表，用于存储重载的参数
    overloaded_args: List[Any] = []
    # 遍历相关参数列表
    for arg in relevant_args:
        # 获取参数的类型
        arg_type = get_type_fn(arg)
        
        # 只收集具有唯一类型的参数，这可以确保即使有很长的可能重载的参数列表，也能保持合理的性能。
        #
        # 注意：重要的是要排除 _disabled_torch_function_impl，否则可能会出现问题。
        if (
            arg_type not in overloaded_types
            and hasattr(arg_type, "__torch_function__")
            and arg_type.__torch_function__ != torch._C._disabled_torch_function_impl
        ):
            # 如果已经有了重载类型，添加新的类型到集合中
            if overloaded_types:
                overloaded_types.add(arg_type)
                # 默认将参数插入到列表末尾，但如果它是另一个参数的子类，则插入到该参数之前。
                # 这确保了“子类在超类之前”的顺序。
                index = len(overloaded_args)
                for i, old_arg in enumerate(overloaded_args):
                    if issubclass(arg_type, get_type_fn(old_arg)):
                        index = i
                        break
                overloaded_args.insert(index, arg)
            else:
                # 如果是第一个类型，则初始化类型集合和参数列表
                overloaded_types = {arg_type}
                overloaded_args = [arg]
    
    # 返回收集到的重载参数列表
    return overloaded_args
def handle_torch_function(
    public_api: Callable,
    relevant_args: Iterable[Any],
    *args,
    **kwargs,
) -> Any:
    """Implement a function with checks for ``__torch_function__`` overrides.

    See torch::autograd::handle_torch_function for the equivalent of this
    function in the C++ implementation.

    Arguments
    ---------
    public_api : function
        Function exposed by the public torch API originally called like
        ``public_api(*args, **kwargs)`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __torch_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    object
        Result from calling ``implementation`` or an ``__torch_function__``
        method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.

    Example
    -------
    >>> def func(a):
    ...     if has_torch_function_unary(a):
    ...         return handle_torch_function(func, (a,), a)
    ...     return a + 0
    """
    # Check for __torch_function__ methods in the given arguments.
    overloaded_args = _get_overloaded_args(relevant_args)
    # Extract types of overloaded_args.
    types = tuple(map(type, overloaded_args))

    # Check if __torch_function__ mode is enabled.
    if _is_torch_function_mode_enabled():
        # Temporarily pop the mode and call __torch_function__.
        with _pop_mode_temporarily() as mode:
            result = mode.__torch_function__(public_api, types, args, kwargs)
        # If __torch_function__ returned a result other than NotImplemented, return it.
        if result is not NotImplemented:
            return result

    # Call __torch_function__ overrides for each overloaded argument.
    for overloaded_arg in overloaded_args:
        # Retrieve the __torch_function__ method for the current argument.
        torch_func_method = overloaded_arg.__torch_function__
        # Check if it's a plain method and issue a deprecation warning if so.
        if (
            hasattr(torch_func_method, "__self__")
            and torch_func_method.__self__ is overloaded_arg
            and torch_func_method is not torch._C._disabled_torch_function_impl
        ):
            warnings.warn(
                "Defining your `__torch_function__ as a plain method is deprecated and "
                "will be an error in future, please define it as a classmethod.",
                DeprecationWarning,
            )

        # Call the __torch_function__ method with public_api, types, args, kwargs.
        result = torch_func_method(public_api, types, args, kwargs)

        # If __torch_function__ returned a result other than NotImplemented, return it.
        if result is not NotImplemented:
            return result

    # Construct the function name string for the public_api.
    func_name = f"{public_api.__module__}.{public_api.__name__}"
    # 构建错误消息，指出未找到在实现__torch_function__的类型中处理指定函数的方法
    msg = (
        f"no implementation found for '{func_name}' on types that implement "
        f"__torch_function__: {[type(arg) for arg in overloaded_args]}"
    )
    # 如果当前处于torch function模式，则添加当前函数模式到错误消息中
    if _is_torch_function_mode_enabled():
        msg += f" nor in mode {_get_current_function_mode()}"
    # 抛出类型错误，包含构建好的错误消息
    raise TypeError(msg)
# 将 _has_torch_function 函数的文档字符串添加到 has_torch_function 变量中
has_torch_function = _add_docstr(
    _has_torch_function,
    r"""Check for __torch_function__ implementations in the elements of an iterable
    or if a __torch_function__ mode is enabled.  Considers exact ``Tensor`` s
    and ``Parameter`` s non-dispatchable.  Use this to guard a call to
    :func:`handle_torch_function`; don't use it to test if something
    is Tensor-like, use :func:`is_tensor_like` instead.
    Arguments
    ---------
    relevant_args : iterable
        Iterable or arguments to check for __torch_function__ methods.
    Returns
    -------
    bool
        True if any of the elements of relevant_args have __torch_function__
        implementations, False otherwise.
    See Also
    ________
    torch.is_tensor_like
        Checks if something is a Tensor-like, including an exact ``Tensor``.
    """
)

# 将 _has_torch_function_unary 函数的文档字符串添加到 has_torch_function_unary 变量中
has_torch_function_unary = _add_docstr(
    _has_torch_function_unary,
    r"""Special case of `has_torch_function` for single inputs.
    Instead of:
      `has_torch_function((t,))`
    call:
      `has_torch_function_unary(t)`
    which skips unnecessary packing and unpacking work.
    """
)

# 将 _has_torch_function_variadic 函数的文档字符串添加到 has_torch_function_variadic 变量中
has_torch_function_variadic = _add_docstr(
    _has_torch_function_variadic,
    r"""Special case of `has_torch_function` that skips tuple creation.

    This uses the METH_FASTCALL protocol introduced in Python 3.7

    Instead of:
      `has_torch_function((a, b))`
    call:
      `has_torch_function_variadic(a, b)`
    which skips unnecessary packing and unpacking work.
    """
)

# 定义一个装饰器函数，将其结果缓存起来，用于获取可重载函数和它们的索引
@functools.lru_cache(None)
def _get_overridable_functions() -> (
    Tuple[Dict[Any, List[Callable]], Dict[Callable, str]]
):
    # 创建一个默认字典，用于存储各个命名空间下的可重载函数
    overridable_funcs = collections.defaultdict(list)
    # 创建一个空字典，用于存储被测试函数的索引
    index = {}
    # 定义要测试的命名空间及其内容
    tested_namespaces = [
        ("torch", torch, torch.__all__),
        ("torch.functional", torch.functional, torch.functional.__all__),
        ("torch.nn.functional", torch.nn.functional, dir(torch.nn.functional)),
        ("torch.nn.init", torch.nn.init, dir(torch.nn.init)),
        ("torch.Tensor", torch.Tensor, dir(torch.Tensor)),
        ("torch.linalg", torch.linalg, dir(torch.linalg)),
        ("torch.fft", torch.fft, dir(torch.fft)),
        ("torch.special", torch.special, dir(torch.special)),
    ]
    for namespace_str, namespace, ns_funcs in tested_namespaces:
        # 遍历测试过的命名空间及其函数列表
        for func_name in ns_funcs:
            # 遍历命名空间中的函数名

            ignore = False
            # 初始化忽略标志为False

            # 忽略私有函数或在torch.__init__中删除的函数
            if namespace is not torch.Tensor:
                # 如果命名空间不是torch.Tensor
                if func_name.startswith("__"):
                    # 如果函数名以双下划线开头，则跳过
                    continue
                elif func_name.startswith("_"):
                    # 如果函数名以下划线开头，则标记为忽略
                    ignore = True
                elif func_name.endswith("_"):
                    # 如果函数名以下划线结尾，则标记为忽略
                    ignore = True
                elif not func_name[0].islower():
                    # 如果函数名不以小写字母开头，则标记为忽略
                    ignore = True
                elif func_name == "unique_dim":
                    # 如果函数名为"unique_dim"，则跳过
                    continue
            else:
                # 如果命名空间是torch.Tensor
                func = getattr(namespace, func_name)
                if getattr(object, func_name, None) == func:
                    # 如果命名空间中的函数与全局对象中同名函数相同，则跳过
                    continue
                if func_name == "__weakref__":
                    # 如果函数名为"__weakref__"，则跳过
                    continue

            func = getattr(namespace, func_name)
            if namespace is torch.Tensor and getattr(object, func_name, None) == func:
                # 如果命名空间是torch.Tensor且全局对象中同名函数与命名空间中的函数相同，则跳过
                continue

            # 忽略重新导出的模块
            if isinstance(func, types.ModuleType):
                # 如果函数是模块类型，则跳过
                continue

            # 忽略__future__导入
            if isinstance(func, __future__._Feature):
                # 如果函数是__future__._Feature类型，则跳过
                continue

            if not callable(func) and hasattr(func, "__get__"):
                # 如果函数不可调用且具有__get__属性
                index[func.__get__] = f"{namespace_str}.{func_name}.__get__"
                index[func.__set__] = f"{namespace_str}.{func_name}.__set__"
                if ignore:
                    # 如果标记为忽略，则继续下一次循环
                    continue
                if func.__get__ in get_ignored_functions():
                    # 如果func.__get__在get_ignored_functions()返回的元组中，则验证不通过
                    msg = (
                        "{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                        "but still has an explicit override"
                    )
                    assert func.__get__ not in get_testing_overrides(), msg.format(
                        namespace, func.__name__
                    )
                    continue
                else:
                    # 将可重写的函数及其__get__添加到overridable_funcs字典中
                    overridable_funcs[func].append(func.__get__)
                    continue

            if not callable(func):
                # 如果函数不可调用，则继续下一次循环
                continue

            # 将函数及其完整命名添加到index字典中
            index[func] = f"{namespace_str}.{func_name}"

            if ignore:
                # 如果标记为忽略，则继续下一次循环
                continue

            # 不能被__torch_function__重写的函数，验证不通过
            if func in get_ignored_functions():
                msg = (
                    "{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                    "but still has an explicit override"
                )
                assert func not in get_testing_overrides(), msg.format(
                    namespace, func.__name__
                )
                continue

            # 将可重写的函数及其命名空间添加到overridable_funcs字典中
            overridable_funcs[namespace].append(func)

    # 返回可重写的函数字典和完整命名字典
    return overridable_funcs, index
# 禁用用户警告的装饰器，用于包装函数
@_disable_user_warnings
# 获取可以通过 __torch_function__ 覆盖的函数列表
def get_overridable_functions() -> Dict[Any, List[Callable]]:
    """List functions that are overridable via __torch_function__

    Returns
    -------
    Dict[Any, List[Callable]]
        A dictionary that maps namespaces that contain overridable functions
        to functions in that namespace that can be overridden.
    """
    return _get_overridable_functions()[0]


# 禁用用户警告的装饰器，用于包装函数
@_disable_user_warnings
# 解析传递给 __torch_function__ 的函数的人类可读字符串名称
def resolve_name(f):
    """Get a human readable string name for a function passed to
    __torch_function__

    Arguments
    ---------
    f : Callable
        Function to resolve the name of.

    Returns
    -------
    str
        Name of the function; if eval'ed it should give back the input
        function.
    """
    # 如果 f 是 torch._ops.OpOverload 或 torch._ops.OpOverloadPacket 类型，返回其字符串表示
    if isinstance(f, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return str(f)
    # 否则从 _get_overridable_functions() 返回的字典中获取函数 f 的名称
    return _get_overridable_functions()[1].get(f)


# 使用 functools.lru_cache 进行缓存装饰，不限制缓存大小
@functools.lru_cache(None)
# 返回可以在 torch.Tensor 上重写的方法集合
def _get_tensor_methods() -> Set[Callable]:
    """Returns a set of the overridable methods on ``torch.Tensor``"""
    # 获取可以重写的函数集合
    overridable_funcs = get_overridable_functions()
    # 获取 torch.Tensor 上的方法集合，并转为集合返回
    methods = set(overridable_funcs[torch.Tensor])
    return methods


# 禁用用户警告的装饰器，用于包装函数
@_disable_user_warnings
# 判断传入的函数是否是属于或者是 torch.Tensor 的方法或属性
def is_tensor_method_or_property(func: Callable) -> bool:
    """
    Returns True if the function passed in is a handler for a
    method or property belonging to ``torch.Tensor``, as passed
    into ``__torch_function__``.

    .. note::
       For properties, their ``__get__`` method must be passed in.

    This may be needed, in particular, for the following reasons:

    1. Methods/properties sometimes don't contain a `__module__` slot.
    2. They require that the first passed-in argument is an instance
       of ``torch.Tensor``.

    Examples
    --------
    >>> is_tensor_method_or_property(torch.Tensor.add)
    True
    >>> is_tensor_method_or_property(torch.add)
    False
    """
    # 判断 func 是否在 _get_tensor_methods() 返回的方法集合中，或者其名称是否为 "__get__"
    return func in _get_tensor_methods() or func.__name__ == "__get__"


# 检查传入的对象是否类似于 Tensor
def is_tensor_like(inp):
    """
    Returns ``True`` if the passed-in input is a Tensor-like.

    Currently, this occurs whenever there's a ``__torch_function__``
    attribute on the type of the input.

    Examples
    --------
    A subclass of tensor is generally a Tensor-like.

    >>> class SubTensor(torch.Tensor): ...
    >>> is_tensor_like(SubTensor([0]))
    True

    Built-in or user types aren't usually Tensor-like.

    >>> is_tensor_like(6)
    False
    >>> is_tensor_like(None)
    False
    >>> class NotATensor: ...
    >>> is_tensor_like(NotATensor())
    False

    But, they can be made Tensor-like by implementing __torch_function__.

    >>> class TensorLike:
    ...     @classmethod
    ...     def __torch_function__(cls, func, types, args, kwargs):
    ...         return -1
    >>> is_tensor_like(TensorLike())
    True
    """
    # 如果 inp 的类型是 torch.Tensor 或者其具有 "__torch_function__" 属性，则返回 True
    return type(inp) is torch.Tensor or hasattr(inp, "__torch_function__")


# TorchFunctionMode 类的简要说明
class TorchFunctionMode:
    """
    This class serves as a placeholder for defining the mode of torch function behavior.

    It is intended to provide a structured way to manage and control how torch functions behave
    when overridden via `__torch_function__`. This could involve defining certain modes or states
    that affect the behavior of torch operations or tensors when specific conditions are met.

    The exact implementation details and usage scenarios for this class may vary depending on
    specific needs related to torch function customization and behavior modification.
    """
    """
    A ``TorchFunctionMode`` allows you to override the meaning of all
    ``__torch_function__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchFunctionMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_function__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_function__`` implementation, either explicitly
    invoke ``self.__torch_function__(...)``, or use the context manager
    ``enable_torch_function_mode(self, replace=self.inner)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """

    inner: "TorchFunctionMode"

    # Force metaclass to generate constructor at the base of the hierarchy
    def __init__(self):
        pass

    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError

    def __enter__(self):
        _push_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_mode()

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn(
            "`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`"
        )
        instance = cls(*args, **kwargs)
        return instance
# 获取当前 Torch 函数模式的状态，返回最顶层的函数模式对象
def _get_current_function_mode():
    # 获取当前 Torch 函数调用栈的长度
    stack_len = _len_torch_function_stack()
    # 如果栈长度大于 0，则返回栈顶的函数模式对象；否则返回 None
    return _get_function_stack_at(stack_len - 1) if stack_len > 0 else None


# 获取当前 Torch 函数模式调用栈中所有函数模式对象的列表
def _get_current_function_mode_stack():
    # 获取当前 Torch 函数调用栈的长度
    stack_len = _len_torch_function_stack()
    # 返回所有函数模式对象组成的列表
    return [_get_function_stack_at(i) for i in range(stack_len)]


# 将指定的模式推入 Torch 函数调用栈
def _push_mode(mode):
    _push_on_torch_function_stack(mode)


# 弹出 Torch 函数调用栈的顶部模式，并返回被弹出的模式对象
def _pop_mode():
    old = _pop_torch_function_stack()
    return old


# 临时暂时弹出 Torch 函数调用栈的顶部模式，使用 contextlib 管理上下文
@contextlib.contextmanager
def _pop_mode_temporarily():
    # 先弹出当前的模式，并保存到 old 变量中
    old = _pop_mode()
    try:
        # 在上下文中返回旧模式对象，使用 yield
        yield old
    finally:
        # 在上下文结束后，将旧模式对象重新推入 Torch 函数调用栈
        _push_mode(old)


# 定义一个基于 Torch 函数模式的基类，继承自 TorchFunctionMode
class BaseTorchFunctionMode(TorchFunctionMode):
    # 定义 __torch_function__ 方法，用于处理 Torch 函数的调用
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则初始化为空字典
        if kwargs is None:
            kwargs = {}
        # 调用 func 函数，传入 args 和 kwargs 参数
        return func(*args, **kwargs)


# 使用 contextlib 定义一个上下文管理器，用于临时启用可重入分派
@contextlib.contextmanager
def enable_reentrant_dispatch():
    # 使用 torch._C._RestorePythonTLSSnapshot 上下文管理器
    # 注意：不能简单地使用 `enable_reentrant_dispatch = torch._C._RestorePythonTLSSnapshot`
    # 原因：
    # 1. 在导入此文件时，torch._C._RestorePythonTLSSnapshot 可能不可用，可能是导入顺序问题。
    # 2. enable_reentrant_dispatch 技术上是公共 API；赋值会改变 __module__ 看起来像是私有的。
    with torch._C._RestorePythonTLSSnapshot():
        try:
            # 在上下文中使用 yield，使其成为上下文管理器
            yield
        finally:
            # 最终在上下文结束后执行的操作，此处为空操作
            pass
```