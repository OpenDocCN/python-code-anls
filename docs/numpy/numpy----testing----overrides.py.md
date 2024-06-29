# `.\numpy\numpy\testing\overrides.py`

```
# 导入测试 `__array_function__` 和 ufunc 重写实现的工具

from numpy._core.overrides import ARRAY_FUNCTIONS as _array_functions
# 导入 numpy 库中的 ufunc 别名 `_ufunc`
from numpy import ufunc as _ufunc
# 导入 numpy 库中的 umath 模块别名 `_umath`
import numpy._core.umath as _umath

def get_overridable_numpy_ufuncs():
    """列出所有可以通过 `__array_ufunc__` 被重写的 numpy ufunc 函数

    Parameters
    ----------
    None

    Returns
    -------
    set
        包含所有可以在公共 numpy API 中通过 `__array_ufunc__` 被重写的 ufunc 集合。
    """
    # 从 `_umath.__dict__.values()` 中选择所有的 ufunc 对象，并存入集合 `ufuncs`
    ufuncs = {obj for obj in _umath.__dict__.values()
              if isinstance(obj, _ufunc)}
    return ufuncs

def allows_array_ufunc_override(func):
    """确定一个函数是否可以通过 `__array_ufunc__` 被重写

    Parameters
    ----------
    func : callable
        可能可以通过 `__array_ufunc__` 被重写的函数

    Returns
    -------
    bool
        如果 `func` 可以通过 `__array_ufunc__` 被重写则返回 `True`，否则返回 `False`。

    Notes
    -----
    该函数等价于 ``isinstance(func, np.ufunc)`` 并且对于在 Numpy 之外定义的 ufuncs 也能正确工作。
    """
    return isinstance(func, np.ufunc)

def get_overridable_numpy_array_functions():
    """列出所有可以通过 `__array_function__` 被重写的 numpy 函数

    Parameters
    ----------
    None

    Returns
    -------
    set
        包含所有可以在公共 numpy API 中通过 `__array_function__` 被重写的函数集合。

    """
    # 'import numpy' 并没有导入 `recfunctions`，因此确保它被导入，这样定义在那里的 ufunc 才会出现在 ufunc 列表中
    from numpy.lib import recfunctions
    return _array_functions.copy()

def allows_array_function_override(func):
    """确定一个 Numpy 函数是否可以通过 `__array_function__` 被重写

    Parameters
    ----------
    func : callable
        可能可以通过 `__array_function__` 被重写的函数

    Returns
    -------
    bool
        如果 `func` 是可以在 Numpy API 中通过 `__array_function__` 被重写的函数则返回 `True`，否则返回 `False`。
    """
    return func in _array_functions
```