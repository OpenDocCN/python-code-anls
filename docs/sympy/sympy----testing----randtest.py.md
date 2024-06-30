# `D:\src\scipysrc\sympy\sympy\testing\randtest.py`

```
"""
.. deprecated:: 1.10

   ``sympy.testing.randtest`` functions have been moved to
   :mod:`sympy.core.random`.

"""
# 引入 sympy 库中的异常处理函数 sympy_deprecation_warning
from sympy.utilities.exceptions import sympy_deprecation_warning

# 发出关于 sympy.testing.randtest 模块已废弃的警告，并建议使用 sympy.core.random 替代
sympy_deprecation_warning("The sympy.testing.randtest submodule is deprecated. Use sympy.core.random instead.",
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-sympy-testing-randtest")

# 从 sympy.core.random 模块导入以下函数，不检查未使用的导入警告
from sympy.core.random import (  # noqa:F401
    random_complex_number,         # 导入随机复数生成函数
    verify_numerically,            # 导入数值验证函数
    test_derivative_numerically,   # 导入数值求导验证函数
    _randrange,                    # 导入随机范围生成函数
    _randint                       # 导入随机整数生成函数
)
```