# `D:\src\scipysrc\scipy\scipy\_lib\uarray.py`

```
"""
`uarray` provides functions for generating multimethods that dispatch to
multiple different backends

This should be imported, rather than `_uarray` so that an installed version could
be used instead, if available. This means that users can call
`uarray.set_backend` directly instead of going through SciPy.
"""


# 尝试导入 `uarray` 库，如果导入失败则标记 `_has_uarray` 为 False
try:
    import uarray as _uarray
except ImportError:
    _has_uarray = False
else:
    # 导入成功后，从 SciPy 库中导入版本号比较模块 `Version`
    from scipy._lib._pep440 import Version as _Version

    # 检查 `uarray` 版本是否大于等于 0.8
    _has_uarray = _Version(_uarray.__version__) >= _Version("0.8")
    # 清理 `_uarray` 和 `_Version` 的引用，以便后续使用
    del _uarray
    del _Version


# 如果 `_has_uarray` 为真，则从 `uarray` 中导入所有内容和 `_Function` 类
if _has_uarray:
    from uarray import *  # noqa: F403
    from uarray import _Function
else:
    # 否则，从当前模块的 `_uarray` 中导入所有内容和 `_Function` 类
    from ._uarray import *  # noqa: F403
    from ._uarray import _Function  # noqa: F401

# 清理 `_has_uarray` 的引用，结束条件分支
del _has_uarray
```