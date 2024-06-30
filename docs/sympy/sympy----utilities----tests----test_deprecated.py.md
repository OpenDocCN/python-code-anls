# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_deprecated.py`

```
# 导入 warns_deprecated_sympy 函数，用于捕获 Sympy 库中的已弃用警告
from sympy.testing.pytest import warns_deprecated_sympy

# 以下代码是为了测试 Sympy 库中已弃用的实用工具模块的功能
def test_deprecated_utilities():
    # 在执行以下代码块期间，捕获 sympy 库中已弃用的警告
    with warns_deprecated_sympy():
        # 导入 sympy.utilities.pytest 模块，并忽略 F401 类型的未使用变量警告
        import sympy.utilities.pytest  # noqa:F401
    with warns_deprecated_sympy():
        # 导入 sympy.utilities.runtests 模块，并忽略 F401 类型的未使用变量警告
        import sympy.utilities.runtests  # noqa:F401
    with warns_deprecated_sympy():
        # 导入 sympy.utilities.randtest 模块，并忽略 F401 类型的未使用变量警告
        import sympy.utilities.randtest  # noqa:F401
    with warns_deprecated_sympy():
        # 导入 sympy.utilities.tmpfiles 模块，并忽略 F401 类型的未使用变量警告
        import sympy.utilities.tmpfiles  # noqa:F401
```