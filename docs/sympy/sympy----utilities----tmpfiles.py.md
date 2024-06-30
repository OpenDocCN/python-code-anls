# `D:\src\scipysrc\sympy\sympy\utilities\tmpfiles.py`

```
"""
.. deprecated:: 1.6

   sympy.utilities.tmpfiles has been renamed to sympy.testing.tmpfiles.
"""
# 从 sympy.utilities.exceptions 模块中导入 sympy_deprecation_warning 函数
from sympy.utilities.exceptions import sympy_deprecation_warning

# 发出 SymPy 弃用警告，提醒用户 sympy.utilities.tmpfiles 模块已被重命名为 sympy.testing.tmpfiles
sympy_deprecation_warning("The sympy.utilities.tmpfiles submodule is deprecated. Use sympy.testing.tmpfiles instead.",
    deprecated_since_version="1.6",  # 模块从版本 1.6 开始被弃用
    active_deprecations_target="deprecated-sympy-utilities-submodules")  # 弃用目标是 sympy.utilities.tmpfiles 的子模块

# 从 sympy.testing.tmpfiles 模块导入所有内容，忽略 F401 类型的未使用导入警告
from sympy.testing.tmpfiles import *  # noqa:F401
```