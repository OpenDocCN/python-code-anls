# `D:\src\scipysrc\sympy\sympy\core\tests\test_compatibility.py`

```
from sympy.testing.pytest import warns_deprecated_sympy

# 定义一个测试函数，用于测试 sympy.core.compatibility 的兼容性子模块
def test_compatibility_submodule():
    # 使用 warns_deprecated_sympy 上下文管理器来捕获 sympy.core.compatibility 的弃用警告
    with warns_deprecated_sympy():
        # 导入 sympy.core.compatibility 模块，并忽略 F401 警告（未使用的导入）
        import sympy.core.compatibility # noqa:F401
```