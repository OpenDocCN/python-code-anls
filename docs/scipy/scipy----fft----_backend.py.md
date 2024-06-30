# `D:\src\scipysrc\scipy\scipy\fft\_backend.py`

```
import scipy._lib.uarray as ua
from . import _basic_backend
from . import _realtransforms_backend
from . import _fftlog_backend

class _ScipyBackend:
    """The default backend for fft calculations
    
    Notes
    -----
    We use the domain ``numpy.scipy`` rather than ``scipy`` because ``uarray``
    treats the domain as a hierarchy. This means the user can install a single
    backend for ``numpy`` and have it implement ``numpy.scipy.fft`` as well.
    """
    __ua_domain__ = "numpy.scipy.fft"

    @staticmethod
    def __ua_function__(method, args, kwargs):
        # Try to find the method in basic backend
        fn = getattr(_basic_backend, method.__name__, None)
        if fn is None:
            # If not found, try in realtransforms backend
            fn = getattr(_realtransforms_backend, method.__name__, None)
        if fn is None:
            # If still not found, try in fftlog backend
            fn = getattr(_fftlog_backend, method.__name__, None)
        if fn is None:
            # If method is not found in any backend, return NotImplemented
            return NotImplemented
        # Call the found function with given arguments and keyword arguments
        return fn(*args, **kwargs)

_named_backends = {
    'scipy': _ScipyBackend,
}

def _backend_from_arg(backend):
    """Maps strings to known backends and validates the backend
    
    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.
        
    Returns
    -------
    object
        The validated backend object.
        
    Raises
    ------
    ValueError
        If the backend is not recognized or does not implement "numpy.scipy.fft".
    """
    if isinstance(backend, str):
        try:
            # Retrieve the backend class from the named backends dictionary
            backend = _named_backends[backend]
        except KeyError as e:
            # Raise ValueError if the backend name is not found
            raise ValueError(f'Unknown backend {backend}') from e
    
    # Check if the backend implements the required domain
    if backend.__ua_domain__ != 'numpy.scipy.fft':
        # Raise ValueError if the backend does not implement "numpy.scipy.fft"
        raise ValueError('Backend does not implement "numpy.scipy.fft"')
    
    return backend

def set_global_backend(backend, coerce=False, only=False, try_last=False):
    """Sets the global fft backend
    
    This utility method replaces the default backend for permanent use. It
    will be tried in the list of backends automatically, unless the
    ``only`` flag is set on a backend. This will be the first tried
    backend outside the :obj:`set_backend` context manager.
    
    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.
    coerce : bool
        Whether to coerce input types when trying this backend.
    only : bool
        If ``True``, no more backends will be tried if this fails.
        Implied by ``coerce=True``.
    try_last : bool
        If ``True``, the global backend is tried after registered backends.
        
    Raises
    ------
    ValueError
        If the backend does not implement ``numpy.scipy.fft``.
        
    Notes
    -----
    This will overwrite the previously set global backend, which, by default, is
    the SciPy implementation.
    
    Examples
    --------
    We can set the global fft backend:
    
    >>> from scipy.fft import fft, set_global_backend
    >>> set_global_backend("scipy")  # Sets global backend (default is "scipy").
    >>> fft([1])  # Calls the global backend
    array([1.+0.j])
    """
    # Validate and retrieve the backend object
    backend = _backend_from_arg(backend)
    # Set the global backend using the validated backend object
    ua.set_global_backend(backend, coerce=coerce, only=only, try_last=try_last)

def register_backend(backend):
    """
    Placeholder function for registering additional backends.
    """
    pass
    # 注册一个后端以供永久使用。

    # 注册的后端具有最低优先级，在全局后端尝试之后使用。

    # 参数
    # ------
    # backend : {object, 'scipy'}
    #     要使用的后端。
    #     可以是包含已知后端名称 {'scipy'} 的字符串，
    #     或者是实现 uarray 协议的对象。

    # 引发
    # ------
    # ValueError: 如果后端没有实现 ``numpy.scipy.fft``。

    # 示例
    # --------
    # 我们可以注册一个新的 fft 后端：

    # >>> from scipy.fft import fft, register_backend, set_global_backend
    # >>> class NoopBackend:  # 定义一个无效的后端
    # ...     __ua_domain__ = "numpy.scipy.fft"
    # ...     def __ua_function__(self, func, args, kwargs):
    # ...          return NotImplemented
    # >>> set_global_backend(NoopBackend())  # 将无效的后端设置为全局
    # >>> register_backend("scipy")  # 注册一个新的后端
    # # 由于全局后端返回 `NotImplemented`，因此调用注册的后端
    # >>> fft([1])
    # array([1.+0.j])
    # >>> set_global_backend("scipy")  # 将全局后端恢复为默认值

    """
    backend = _backend_from_arg(backend)
    # 根据传入的参数获取后端对象

    ua.register_backend(backend)
    # 使用 uarray 模块注册指定的后端
    ```
# 定义一个上下文管理器，用于在固定作用域内设置后端。

def set_backend(backend, coerce=False, only=False):
    """Context manager to set the backend within a fixed scope.

    Upon entering the ``with`` statement, the given backend will be added to
    the list of available backends with the highest priority. Upon exit, the
    backend is reset to the state before entering the scope.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.
    coerce : bool, optional
        Whether to allow expensive conversions for the ``x`` parameter. e.g.,
        copying a NumPy array to the GPU for a CuPy backend. Implies ``only``.
    only : bool, optional
        If only is ``True`` and this backend returns ``NotImplemented``, then a
        BackendNotImplemented error will be raised immediately. Ignoring any
        lower priority backends.

    Examples
    --------
    >>> import scipy.fft as fft
    >>> with fft.set_backend('scipy', only=True):
    ...     fft.fft([1])  # Always calls the scipy implementation
    array([1.+0.j])
    """
    # 根据参数获取后端对象
    backend = _backend_from_arg(backend)
    # 调用 ua 模块的 set_backend 方法，设置当前作用域内的后端
    return ua.set_backend(backend, coerce=coerce, only=only)


def skip_backend(backend):
    """Context manager to skip a backend within a fixed scope.

    Within the context of a ``with`` statement, the given backend will not be
    called. This covers backends registered both locally and globally. Upon
    exit, the backend will again be considered.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to skip.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.

    Examples
    --------
    >>> import scipy.fft as fft
    >>> fft.fft([1])  # Calls default SciPy backend
    array([1.+0.j])
    >>> with fft.skip_backend('scipy'):  # We explicitly skip the SciPy backend
    ...     fft.fft([1])                 # leaving no implementation available
    Traceback (most recent call last):
        ...
    BackendNotImplementedError: No selected backends had an implementation ...
    """
    # 根据参数获取后端对象
    backend = _backend_from_arg(backend)
    # 调用 ua 模块的 skip_backend 方法，跳过当前作用域内的指定后端
    return ua.skip_backend(backend)


# 设置全局默认后端为 'scipy'，并尝试将其置于最低优先级
set_global_backend('scipy', try_last=True)
```