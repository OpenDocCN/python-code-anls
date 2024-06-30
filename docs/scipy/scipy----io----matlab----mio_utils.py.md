# `D:\src\scipysrc\scipy\scipy\io\matlab\mio_utils.py`

```
# 导入_scipy._lib.deprecation模块中的_sub_module_deprecation函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个空列表__all__
__all__: list[str] = []

# 定义一个特殊方法__dir__()，返回__all__列表
def __dir__():
    return __all__

# 定义一个特殊方法__getattr__(name)，返回_sub_module_deprecation函数的调用结果
def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio_utils",
                                   private_modules=["_mio_utils"], all=__all__,
                                   attribute=name)
```