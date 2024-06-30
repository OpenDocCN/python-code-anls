# `D:\src\scipysrc\sympy\sympy\simplify\traversaltools.py`

```
# 从 sympy.core.traversal 模块中导入 use 别名为 _use
from sympy.core.traversal import use as _use
# 从 sympy.utilities.decorator 模块中导入 deprecated 装饰器函数
from sympy.utilities.decorator import deprecated

# 使用 deprecated 装饰器将 _use 函数包装起来，并设置警告信息和版本信息
use = deprecated(
    """
    Using use from the sympy.simplify.traversaltools submodule is
    deprecated.

    Instead, use use from the top-level sympy namespace, like

        sympy.use
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved"
)(_use)
```