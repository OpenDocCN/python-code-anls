# `D:\src\scipysrc\sympy\sympy\core\trace.py`

```
# 导入 sympy_deprecation_warning 函数，用于发出 Sympy 废弃警告消息
from sympy.utilities.exceptions import sympy_deprecation_warning

# 发出 Sympy 废弃警告消息，指示 sympy.core.trace 已废弃，建议使用 sympy.physics.quantum.trace 替代
sympy_deprecation_warning(
    """
    sympy.core.trace is deprecated. Use sympy.physics.quantum.trace
    instead.
    """,
    deprecated_since_version="1.10",  # 指定从版本 1.10 开始废弃
    active_deprecations_target="sympy-core-trace-deprecated",  # 指定警告的目标
)

# 从 sympy.physics.quantum.trace 模块导入 Tr 类，使用 noqa:F401 禁止未使用的导入警告
from sympy.physics.quantum.trace import Tr # noqa:F401
```