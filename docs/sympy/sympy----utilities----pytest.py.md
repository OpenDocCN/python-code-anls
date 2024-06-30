# `D:\src\scipysrc\sympy\sympy\utilities\pytest.py`

```
"""
.. deprecated:: 1.6

   sympy.utilities.pytest has been renamed to sympy.testing.pytest.
"""
# 导入警告函数，用于显示 SymPy 废弃警告信息
from sympy.utilities.exceptions import sympy_deprecation_warning

# 发出 SymPy 废弃警告，提示用户停止使用 sympy.utilities.pytest 模块，改用 sympy.testing.pytest
sympy_deprecation_warning("The sympy.utilities.pytest submodule is deprecated. Use sympy.testing.pytest instead.",
    deprecated_since_version="1.6",
    active_deprecations_target="deprecated-sympy-utilities-submodules")

# 导入 sympy.testing.pytest 所有内容，忽略 F401 错误（未使用的导入）
from sympy.testing.pytest import *  # noqa:F401
```