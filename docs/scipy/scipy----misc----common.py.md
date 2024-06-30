# `D:\src\scipysrc\scipy\scipy\misc\common.py`

```
# 引入警告：此文件不适合公共使用，将在 SciPy v2.0.0 中删除。
# 使用 `scipy.datasets` 命名空间来导入下面包含的数据集函数。

# 从 scipy._lib.deprecation 模块中引入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，指定了可以通过 `from module import *` 导入的公共符号
__all__ = [
    'central_diff_weights', 'derivative', 'ascent', 'face',
    'electrocardiogram'
]


# 定义 __dir__() 函数，返回模块中定义的公共符号列表 __all__
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，用于在属性不存在时调用，返回 _sub_module_deprecation 函数的调用结果
def __getattr__(name):
    return _sub_module_deprecation(sub_package="misc", module="common",
                                   private_modules=["_common"], all=__all__,
                                   attribute=name)
```