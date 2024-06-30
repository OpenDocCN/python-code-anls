# `D:\src\scipysrc\sympy\sympy\utilities\randtest.py`

```
"""
`
"""
.. deprecated:: 1.6

   sympy.utilities.randtest has been renamed to sympy.core.random.
"""
# 从 sympy.utilities.exceptions 模块导入 sympy_deprecation_warning 函数，用于显示弃用警告
from sympy.utilities.exceptions import sympy_deprecation_warning

# 显示弃用警告，提醒用户 sympy.utilities.randtest 模块已被弃用，建议使用 sympy.core.random 模块
sympy_deprecation_warning("The sympy.utilities.randtest submodule is deprecated. Use sympy.core.random instead.",
    deprecated_since_version="1.6",  # 指定弃用版本为 1.6
    active_deprecations_target="deprecated-sympy-utilities-submodules")  # 指定目标为已弃用的 sympy.utilities 子模块

# 从 sympy.core.random 模块导入所有内容，忽略 F401 警告（未使用的导入）
from sympy.core.random import *  # noqa:F401
```