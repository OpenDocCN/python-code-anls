# `D:\src\scipysrc\scipy\scipy\interpolate\ndgriddata.py`

```
# 这个文件不是用于公共使用的，将在 SciPy v2.0.0 中移除。
# 使用 `scipy.interpolate` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含了几个字符串元素，用于模块的导入
__all__ = [  # noqa: F822
    'CloughTocher2DInterpolator',
    'LinearNDInterpolator',
    'NearestNDInterpolator',
    'griddata',
]

# 定义 __dir__() 函数，返回模块的公开成员列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，当访问不存在的属性时调用
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，传递一些参数来处理子模块的弃用提示
    return _sub_module_deprecation(sub_package="interpolate", module="ndgriddata",
                                   private_modules=["_ndgriddata"], all=__all__,
                                   attribute=name)
```