# `D:\src\scipysrc\sympy\sympy\core\compatibility.py`

```
"""
.. deprecated:: 1.10

   ``sympy.core.compatibility`` is deprecated. See
   :ref:`sympy-core-compatibility`.

Reimplementations of constructs introduced in later versions of Python than
we support. Also some functions that are needed SymPy-wide and are located
here for easy import.

"""


# 从 sympy.utilities.exceptions 导入 sympy_deprecation_warning 函数
from sympy.utilities.exceptions import sympy_deprecation_warning

# 发出 SymPy 弃用警告，提醒用户不再使用 sympy.core.compatibility 模块
sympy_deprecation_warning("""
The sympy.core.compatibility submodule is deprecated.

This module was only ever intended for internal use. Some of the functions
that were in this module are available from the top-level SymPy namespace,
i.e.,

    from sympy import ordered, default_sort_key

The remaining were only intended for internal SymPy use and should not be used
by user code.
""",
                          deprecated_since_version="1.10",
                          active_deprecations_target="deprecated-sympy-core-compatibility",
                          )


# 从 .sorting 模块中导入 ordered, _nodes, default_sort_key 函数，忽略 F401 错误
from .sorting import ordered, _nodes, default_sort_key # noqa:F401

# 从 sympy.utilities.misc 导入 as_int 函数，忽略 F401 错误
from sympy.utilities.misc import as_int as _as_int # noqa:F401

# 从 sympy.utilities.iterables 导入 iterable, is_sequence, NotIterable 函数，忽略 F401 错误
from sympy.utilities.iterables import iterable, is_sequence, NotIterable # noqa:F401
```