# `D:\src\scipysrc\scipy\scipy\integrate\lsoda.py`

```
# 此文件不适用于公共使用，并将在 SciPy v2.0.0 中移除。

# 导入 _sub_module_deprecation 函数，用于处理子模块的废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 模块级别变量，指定在 from module import * 时可导出的符号，禁止 flake8 F822 错误检查
__all__ = ['lsoda']  # noqa: F822


# 定义 __dir__() 函数，返回当前模块可导出的所有符号
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，用于动态获取属性，处理废弃警告和属性访问
def __getattr__(name):
    return _sub_module_deprecation(sub_package="integrate", module="lsoda",
                                   private_modules=["_lsoda"], all=__all__,
                                   attribute=name)
```