# `D:\src\scipysrc\scipy\scipy\signal\lti_conversion.py`

```
# 该文件不适用于公共使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.signal` 命名空间来导入下面包含的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# __all__ 列表定义了模块中的公共接口（被导出的函数名）
__all__ = [  # noqa: F822
    'tf2ss', 'abcd_normalize', 'ss2tf', 'zpk2ss', 'ss2zpk',
    'cont2discrete', 'tf2zpk', 'zpk2tf', 'normalize'
]


# 定义 __dir__() 函数，返回模块中公共接口的列表 __all__
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，用于动态获取模块中的属性（函数）
def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="lti_conversion",
                                   private_modules=["_lti_conversion"], all=__all__,
                                   attribute=name)
```