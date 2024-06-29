# `.\numpy\numpy\dtypes.py`

```py
"""
This module is home to specific dtypes related functionality and their classes.
For more general information about dtypes, also see `numpy.dtype` and
:ref:`arrays.dtypes`.

Similar to the builtin ``types`` module, this submodule defines types (classes)
that are not widely used directly.

.. versionadded:: NumPy 1.25

    The dtypes module is new in NumPy 1.25.  Previously DType classes were
    only accessible indirectly.


DType classes
-------------

The following are the classes of the corresponding NumPy dtype instances and
NumPy scalar types.  The classes can be used in ``isinstance`` checks and can
also be instantiated or used directly.  Direct use of these classes is not
typical, since their scalar counterparts (e.g. ``np.float64``) or strings
like ``"float64"`` can be used.
"""

# See doc/source/reference/routines.dtypes.rst for module-level docs

# 初始化 __all__ 为空列表，用于存放模块中公开的对象名称
__all__ = []

# 定义函数 _add_dtype_helper，用于向 dtypes 模块中添加 DType 类和别名
def _add_dtype_helper(DType, alias):
    # 从 numpy 模块导入 dtypes
    from numpy import dtypes
    
    # 将 DType 类添加到 dtypes 模块中，并将其名称添加到 __all__ 列表中
    setattr(dtypes, DType.__name__, DType)
    __all__.append(DType.__name__)

    # 如果存在别名，则去除前缀后添加到 dtypes 模块中，并将其名称添加到 __all__ 列表中
    if alias:
        alias = alias.removeprefix("numpy.dtypes.")
        setattr(dtypes, alias, DType)
        __all__.append(alias)
```