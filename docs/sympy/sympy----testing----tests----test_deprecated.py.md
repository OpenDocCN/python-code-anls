# `D:\src\scipysrc\sympy\sympy\testing\tests\test_deprecated.py`

```
# 从 sympy.testing.pytest 模块中导入 warns_deprecated_sympy 函数
from sympy.testing.pytest import warns_deprecated_sympy

# 定义一个测试函数，用于测试 sympy.testing.randtest 模块中的过时功能
def test_deprecated_testing_randtest():
    # 使用 warns_deprecated_sympy 上下文管理器捕获已弃用警告
    with warns_deprecated_sympy():
        # 导入 sympy.testing.randtest 模块，并忽略未使用的导入警告
        import sympy.testing.randtest  # noqa:F401
```