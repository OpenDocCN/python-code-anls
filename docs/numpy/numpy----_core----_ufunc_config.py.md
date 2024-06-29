# `.\numpy\numpy\_core\_ufunc_config.py`

```
"""
Functions for changing global ufunc configuration

This provides helpers which wrap `_get_extobj_dict` and `_make_extobj`, and
`_extobj_contextvar` from umath.
"""
import collections.abc  # 导入 collections.abc 模块
import contextlib  # 导入 contextlib 模块
import contextvars  # 导入 contextvars 模块
import functools  # 导入 functools 模块

from .._utils import set_module  # 从上层模块的 _utils 中导入 set_module 函数
from .umath import _make_extobj, _get_extobj_dict, _extobj_contextvar  # 从 umath 模块导入 _make_extobj, _get_extobj_dict, _extobj_contextvar

__all__ = [
    "seterr", "geterr", "setbufsize", "getbufsize", "seterrcall", "geterrcall",
    "errstate", '_no_nep50_warning'
]  # 公开的模块接口列表

@set_module('numpy')
def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """
    Set how floating-point errors are handled.

    Note that operations on integer scalar types (such as `int16`) are
    handled like floating point, and are affected by these settings.

    Parameters
    ----------
    all : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Set treatment for all types of floating-point errors at once:

        - ignore: Take no action when the exception occurs.
        - warn: Print a :exc:`RuntimeWarning` (via the Python `warnings`
          module).
        - raise: Raise a :exc:`FloatingPointError`.
        - call: Call a function specified using the `seterrcall` function.
        - print: Print a warning directly to ``stdout``.
        - log: Record error in a Log object specified by `seterrcall`.

        The default is not to change the current behavior.
    divide : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for division by zero.
    over : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for floating-point overflow.
    under : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for floating-point underflow.
    invalid : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for invalid floating-point operation.

    Returns
    -------
    old_settings : dict
        Dictionary containing the old settings.

    See also
    --------
    seterrcall : Set a callback function for the 'call' mode.
    geterr, geterrcall, errstate

    Notes
    -----
    The floating-point exceptions are defined in the IEEE 754 standard [1]_:

    - Division by zero: infinite result obtained from finite numbers.
    - Overflow: result too large to be expressed.
    - Underflow: result so close to zero that some precision
      was lost.
    - Invalid operation: result is not an expressible number, typically
      indicates that a NaN was produced.

    .. [1] https://en.wikipedia.org/wiki/IEEE_754

    Examples
    --------
    >>> orig_settings = np.seterr(all='ignore')  # seterr to known value
    >>> np.int16(32000) * np.int16(3)
    30464
    >>> np.seterr(over='raise')
    {'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
    >>> old_settings = np.seterr(all='warn', over='raise')
    >>> np.int16(32000) * np.int16(3)
    """
    # 根据传入的参数设置浮点数错误的处理方式，并返回旧的设置
    old_settings = _get_extobj_dict()  # 获取当前的异常处理设置
    new_settings = {}  # 初始化一个空字典，用于存储新的异常处理设置

    if all is not None:
        new_settings['all'] = _make_extobj(all)  # 根据 all 参数设置所有类型异常处理方式

    if divide is not None:
        new_settings['divide'] = _make_extobj(divide)  # 根据 divide 参数设置除法异常处理方式

    if over is not None:
        new_settings['over'] = _make_extobj(over)  # 根据 over 参数设置溢出异常处理方式

    if under is not None:
        new_settings['under'] = _make_extobj(under)  # 根据 under 参数设置下溢异常处理方式

    if invalid is not None:
        new_settings['invalid'] = _make_extobj(invalid)  # 根据 invalid 参数设置无效操作异常处理方式

    _extobj_contextvar.set(new_settings)  # 使用 contextvar 设置新的异常处理方式

    return old_settings  # 返回旧的异常处理设置
    # 引发的异常通常是由于浮点数溢出而导致的乘法操作错误
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    FloatingPointError: overflow encountered in scalar multiply
    
    # 设置 NumPy 库的错误处理方式为打印所有错误
    >>> old_settings = np.seterr(all='print')
    # 获取当前的 NumPy 错误处理设置
    >>> np.geterr()
    {'divide': 'print', 'over': 'print', 'under': 'print', 'invalid': 'print'}
    # 对两个 int16 类型的数进行乘法运算，结果为 30464
    >>> np.int16(32000) * np.int16(3)
    30464
    # 恢复原始的 NumPy 错误处理设置
    >>> np.seterr(**orig_settings)  # restore original
    {'divide': 'print', 'over': 'print', 'under': 'print', 'invalid': 'print'}
    
    """
    # 获取当前的扩展对象字典
    old = _get_extobj_dict()
    # 从字典中移除键为 "call" 和 "bufsize" 的项，如果存在的话
    # 这些项在错误状态对象中并不存在，因此需要移除
    old.pop("call", None)
    old.pop("bufsize", None)
    
    # 使用指定的错误处理设置创建扩展对象
    extobj = _make_extobj(
            all=all, divide=divide, over=over, under=under, invalid=invalid)
    # 将新创建的扩展对象设置为当前的上下文变量
    _extobj_contextvar.set(extobj)
    # 返回修改后的扩展对象字典
    return old
@set_module('numpy')
def geterr():
    """
    Get the current way of handling floating-point errors.

    Returns
    -------
    res : dict
        A dictionary with keys "divide", "over", "under", and "invalid",
        whose values are from the strings "ignore", "print", "log", "warn",
        "raise", and "call". The keys represent possible floating-point
        exceptions, and the values define how these exceptions are handled.

    See Also
    --------
    geterrcall, seterr, seterrcall

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> np.geterr()
    {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
    >>> np.arange(3.) / np.arange(3.)  # doctest: +SKIP
    array([nan,  1.,  1.])
    RuntimeWarning: invalid value encountered in divide

    >>> oldsettings = np.seterr(all='warn', invalid='raise')
    >>> np.geterr()
    {'divide': 'warn', 'over': 'warn', 'under': 'warn', 'invalid': 'raise'}
    >>> np.arange(3.) / np.arange(3.)
    Traceback (most recent call last):
      ...
    FloatingPointError: invalid value encountered in divide
    >>> oldsettings = np.seterr(**oldsettings)  # restore original

    """
    # Retrieve the current error handling settings as a dictionary
    res = _get_extobj_dict()
    # Remove the entries for "call" and "bufsize" from the dictionary
    res.pop("call", None)
    res.pop("bufsize", None)
    return res


@set_module('numpy')
def setbufsize(size):
    """
    Set the size of the buffer used in ufuncs.

    .. versionchanged:: 2.0
        The scope of setting the buffer is tied to the `numpy.errstate`
        context.  Exiting a ``with errstate():`` will also restore the bufsize.

    Parameters
    ----------
    size : int
        Size of buffer.

    Returns
    -------
    bufsize : int
        Previous size of ufunc buffer in bytes.

    Examples
    --------
    When exiting a `numpy.errstate` context manager the bufsize is restored:

    >>> with np.errstate():
    ...     np.setbufsize(4096)
    ...     print(np.getbufsize())
    ...
    8192
    4096
    >>> np.getbufsize()
    8192

    """
    # Retrieve the current bufsize setting
    old = _get_extobj_dict()["bufsize"]
    # Create a new extended object with the specified bufsize
    extobj = _make_extobj(bufsize=size)
    # Set the new extended object using context variables
    _extobj_contextvar.set(extobj)
    return old


@set_module('numpy')
def getbufsize():
    """
    Return the size of the buffer used in ufuncs.

    Returns
    -------
    getbufsize : int
        Size of ufunc buffer in bytes.

    Examples
    --------
    >>> np.getbufsize()
    8192

    """
    # Retrieve and return the current bufsize from the extended object
    return _get_extobj_dict()["bufsize"]


@set_module('numpy')
def seterrcall(func):
    """
    Set the floating-point error callback function or log object.

    There are two ways to capture floating-point error messages.  The first
    is to set the error-handler to 'call', using `seterr`.  Then, set
    the function to call using this function.

    The second is to set the error-handler to 'log', using `seterr`.

    """
    # This function is intended to set the error callback function or log object,
    # but the actual implementation details are not provided in this excerpt.
    """
    Floating-point errors then trigger a call to the 'write' method of
    the provided object.
    
    Parameters
    ----------
    func : callable f(err, flag) or object with write method
        Function to call upon floating-point errors ('call'-mode) or
        object whose 'write' method is used to log such message ('log'-mode).
    
        The call function takes two arguments. The first is a string describing
        the type of error (such as "divide by zero", "overflow", "underflow",
        or "invalid value"), and the second is the status flag.  The flag is a
        byte, whose four least-significant bits indicate the type of error, one
        of "divide", "over", "under", "invalid"::
    
          [0 0 0 0 divide over under invalid]
    
        In other words, ``flags = divide + 2*over + 4*under + 8*invalid``.
    
        If an object is provided, its write method should take one argument,
        a string.
    
    Returns
    -------
    h : callable, log instance or None
        The old error handler.
    
    See Also
    --------
    seterr, geterr, geterrcall
    
    Examples
    --------
    Callback upon error:
    
    >>> def err_handler(type, flag):
    ...     print("Floating point error (%s), with flag %s" % (type, flag))
    ...
    
    >>> orig_handler = np.seterrcall(err_handler)
    >>> orig_err = np.seterr(all='call')
    
    >>> np.array([1, 2, 3]) / 0.0
    Floating point error (divide by zero), with flag 1
    array([inf, inf, inf])
    
    >>> np.seterrcall(orig_handler)
    <function err_handler at 0x...>
    >>> np.seterr(**orig_err)
    {'divide': 'call', 'over': 'call', 'under': 'call', 'invalid': 'call'}
    
    Log error message:
    
    >>> class Log:
    ...     def write(self, msg):
    ...         print("LOG: %s" % msg)
    ...
    
    >>> log = Log()
    >>> saved_handler = np.seterrcall(log)
    >>> save_err = np.seterr(all='log')
    
    >>> np.array([1, 2, 3]) / 0.0
    LOG: Warning: divide by zero encountered in divide
    array([inf, inf, inf])
    
    >>> np.seterrcall(orig_handler)
    <numpy.Log object at 0x...>
    >>> np.seterr(**orig_err)
    {'divide': 'log', 'over': 'log', 'under': 'log', 'invalid': 'log'}
    
    """
    old = _get_extobj_dict()["call"]
    extobj = _make_extobj(call=func)
    _extobj_contextvar.set(extobj)
    return old
# 设置 numpy 模块的默认模块名为 'numpy'
@set_module('numpy')
# 定义函数 geterrcall，返回当前用于处理浮点错误的回调函数
def geterrcall():
    """
    Return the current callback function used on floating-point errors.

    When the error handling for a floating-point error (one of "divide",
    "over", "under", or "invalid") is set to 'call' or 'log', the function
    that is called or the log instance that is written to is returned by
    `geterrcall`. This function or log instance has been set with
    `seterrcall`.

    Returns
    -------
    errobj : callable, log instance or None
        The current error handler. If no handler was set through `seterrcall`,
        ``None`` is returned.

    See Also
    --------
    seterrcall, seterr, geterr

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> np.geterrcall()  # we did not yet set a handler, returns None

    >>> orig_settings = np.seterr(all='call')
    >>> def err_handler(type, flag):
    ...     print("Floating point error (%s), with flag %s" % (type, flag))
    >>> old_handler = np.seterrcall(err_handler)
    >>> np.array([1, 2, 3]) / 0.0
    Floating point error (divide by zero), with flag 1
    array([inf, inf, inf])

    >>> cur_handler = np.geterrcall()
    >>> cur_handler is err_handler
    True
    >>> old_settings = np.seterr(**orig_settings)  # restore original
    >>> old_handler = np.seterrcall(None)  # restore original

    """
    return _get_extobj_dict()["call"]


# 定义一个未指定类型的类 _unspecified
class _unspecified:
    pass


# 将 _unspecified 类型的实例赋值给 _Unspecified 变量
_Unspecified = _unspecified()


# 设置 numpy 模块的默认模块名为 'numpy'，定义一个类 errstate
@set_module('numpy')
class errstate:
    """
    errstate(**kwargs)

    Context manager for floating-point error handling.

    Using an instance of `errstate` as a context manager allows statements in
    that context to execute with a known error handling behavior. Upon entering
    the context the error handling is set with `seterr` and `seterrcall`, and
    upon exiting it is reset to what it was before.

    ..  versionchanged:: 1.17.0
        `errstate` is also usable as a function decorator, saving
        a level of indentation if an entire function is wrapped.

    .. versionchanged:: 2.0
        `errstate` is now fully thread and asyncio safe, but may not be
        entered more than once.
        It is not safe to decorate async functions using ``errstate``.

    Parameters
    ----------
    kwargs : {divide, over, under, invalid}
        Keyword arguments. The valid keywords are the possible floating-point
        exceptions. Each keyword should have a string value that defines the
        treatment for the particular error. Possible values are
        {'ignore', 'warn', 'raise', 'call', 'print', 'log'}.

    See Also
    --------
    seterr, geterr, seterrcall, geterrcall

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> olderr = np.seterr(all='ignore')  # Set error handling to known state.


    """
    >>> np.arange(3) / 0.
    array([nan, inf, inf])
    
    
    # 在NumPy中，对一个数组进行除以零操作会产生特定的浮点数错误，结果为NaN和inf。
    >>> with np.errstate(divide='ignore'):
    ...     np.arange(3) / 0.
    array([nan, inf, inf])
    
    
    # 使用`np.errstate`上下文管理器来临时忽略浮点数除以零的错误。
    >>> np.sqrt(-1)
    np.float64(nan)
    
    
    # 调用`np.sqrt`函数对负数求平方根会引发浮点数无效的错误，结果为NaN。
    >>> with np.errstate(invalid='raise'):
    ...     np.sqrt(-1)
    Traceback (most recent call last):
      File "<stdin>", line 2, in <module>
    FloatingPointError: invalid value encountered in sqrt
    
    
    # 使用`np.errstate`上下文管理器来设置对浮点数无效操作（如求负数的平方根）抛出异常。
    Outside the context the error handling behavior has not changed:
    >>> np.geterr()
    {'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
    
    
    # 在`np.errstate`上下文管理器外，NumPy错误处理状态未发生改变。
    >>> olderr = np.seterr(**olderr)  # restore original state
    
    
    # 使用`np.seterr`恢复先前保存的错误处理状态。
    """
    __slots__ = (
        "_call", "_all", "_divide", "_over", "_under", "_invalid", "_token")
    
    
    # 定义类的`__slots__`属性，限制实例属性的创建，优化内存使用。
    def __init__(self, *, call=_Unspecified,
                 all=None, divide=None, over=None, under=None, invalid=None):
        self._token = None
        self._call = call
        self._all = all
        self._divide = divide
        self._over = over
        self._under = under
        self._invalid = invalid
    
    
    # 初始化`np.errstate`类的实例，设置各种错误处理状态。
    def __enter__(self):
        # Note that __call__ duplicates much of this logic
        if self._token is not None:
            raise TypeError("Cannot enter `np.errstate` twice.")
        if self._call is _Unspecified:
            extobj = _make_extobj(
                    all=self._all, divide=self._divide, over=self._over,
                    under=self._under, invalid=self._invalid)
        else:
            extobj = _make_extobj(
                    call=self._call,
                    all=self._all, divide=self._divide, over=self._over,
                    under=self._under, invalid=self._invalid)
    
        self._token = _extobj_contextvar.set(extobj)
    
    
    # 实现`__enter__`方法，用于进入`np.errstate`上下文管理器，设置错误处理状态。
    def __exit__(self, *exc_info):
        _extobj_contextvar.reset(self._token)
    
    
    # 实现`__exit__`方法，用于退出`np.errstate`上下文管理器，重置错误处理状态。
    def __call__(self, func):
        # We need to customize `__call__` compared to `ContextDecorator`
        # because we must store the token per-thread so cannot store it on
        # the instance (we could create a new instance for this).
        # This duplicates the code from `__enter__`.
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if self._call is _Unspecified:
                extobj = _make_extobj(
                        all=self._all, divide=self._divide, over=self._over,
                        under=self._under, invalid=self._invalid)
            else:
                extobj = _make_extobj(
                        call=self._call,
                        all=self._all, divide=self._divide, over=self._over,
                        under=self._under, invalid=self._invalid)
    
            _token = _extobj_contextvar.set(extobj)
            try:
                # Call the original, decorated, function:
                return func(*args, **kwargs)
            finally:
                _extobj_contextvar.reset(_token)
    
        return inner
    
    
    # 实现`__call__`方法，允许`np.errstate`实例作为装饰器使用，设置错误处理状态。
# 创建一个上下文变量 NO_NEP50_WARNING，用于跟踪是否禁用了 NEP 50 警告
NO_NEP50_WARNING = contextvars.ContextVar("_no_nep50_warning", default=False)

# 将修饰器应用于下面的函数，指定其模块为 'numpy'
@set_module('numpy')
# 定义一个上下文管理器函数 _no_nep50_warning
@contextlib.contextmanager
def _no_nep50_warning():
    """
    上下文管理器，用于禁用 NEP 50 警告。仅在全局启用 NEP 50 警告时才相关
    （这种情况下不是线程/上下文安全）。

    此警告上下文管理器本身是完全安全的。
    """
    # 设置 NO_NEP50_WARNING 上下文变量为 True，并返回一个标记 token
    token = NO_NEP50_WARNING.set(True)
    try:
        # 执行 yield，进入上下文管理器的主体部分
        yield
    finally:
        # 在上下文管理器结束后，重置 NO_NEP50_WARNING 上下文变量为原来的 token
        NO_NEP50_WARNING.reset(token)
```