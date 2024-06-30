# `D:\src\scipysrc\scipy\scipy\optimize\zeros.py`

```
# 导入 `_sub_module_deprecation` 函数，用于处理子模块废弃警告
# 声明 `__all__` 列表，包含此模块中公开的类和函数名，用于 `from module import *` 语法
from scipy._lib.deprecation import _sub_module_deprecation

# 以下是将来在 SciPy v2.0.0 版本中将删除的文件，不建议公共使用
# 请使用 `scipy.optimize` 命名空间来导入以下函数

# 定义模块的公开接口列表，避免 F822 类型的 Flake8 检查
__all__ = [  # noqa: F822
    'RootResults',
    'bisect',
    'brenth',
    'brentq',
    'newton',
    'ridder',
    'toms748',
]

# 返回当前模块的公开接口列表
def __dir__():
    return __all__

# 当尝试访问当前模块中不存在的属性时，调用 `_sub_module_deprecation` 函数
# 传递参数指定废弃的子模块、模块名、私有模块列表、所有公开属性的列表和尝试访问的属性名
def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="zeros",
                                   private_modules=["_zeros_py"], all=__all__,
                                   attribute=name)
```