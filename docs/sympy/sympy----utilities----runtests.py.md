# `D:\src\scipysrc\sympy\sympy\utilities\runtests.py`

```
"""
.. deprecated:: 1.6

   sympy.utilities.runtests has been renamed to sympy.testing.runtests.
"""

# 导入警告函数用于向用户发出 sympy.utilities.runtests 模块已弃用的警告
from sympy.utilities.exceptions import sympy_deprecation_warning

# 发出警告，提示用户 sympy.utilities.runtests 模块已弃用，建议使用 sympy.testing.runtests 替代
sympy_deprecation_warning("The sympy.utilities.runtests submodule is deprecated. Use sympy.testing.runtests instead.",
    deprecated_since_version="1.6",
    active_deprecations_target="deprecated-sympy-utilities-submodules")

# 导入 sympy.testing.runtests 中的所有内容，忽略 F401 错误（未使用的导入）
from sympy.testing.runtests import *  # noqa:F401
```