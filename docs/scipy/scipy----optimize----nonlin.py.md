# `D:\src\scipysrc\scipy\scipy\optimize\nonlin.py`

```
# 引入 `_sub_module_deprecation` 函数，用于向用户发出关于子模块已废弃的警告信息。
# 声明 `__all__` 变量，包含将在 `scipy.optimize` 命名空间中公开的函数列表，用于限制 `from module import *` 语法的导入范围。
from scipy._lib.deprecation import _sub_module_deprecation

# 设置 `__all__` 变量，指定了当前模块中将被公开的函数名称列表。`noqa: F822` 标志告知 linter 忽略未使用的 `__all__` 变量的警告。
__all__ = [  # noqa: F822
    'BroydenFirst',
    'InverseJacobian',
    'KrylovJacobian',
    'anderson',
    'broyden1',
    'broyden2',
    'diagbroyden',
    'excitingmixing',
    'linearmixing',
    'newton_krylov',
]

# 定义 `__dir__()` 函数，返回当前模块中公开的所有函数的名称列表。
def __dir__():
    return __all__

# 定义 `__getattr__(name)` 函数，用于在模块中动态获取属性。
def __getattr__(name):
    # 调用 `_sub_module_deprecation()` 函数，发出关于废弃子模块的警告信息，参数指定了相关的子包、模块和属性名称。
    return _sub_module_deprecation(sub_package="optimize", module="nonlin",
                                   private_modules=["_nonlin"], all=__all__,
                                   attribute=name)
```