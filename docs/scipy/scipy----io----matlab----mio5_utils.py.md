# `D:\src\scipysrc\scipy\scipy\io\matlab\mio5_utils.py`

```
# 此文件不适合公共使用，并将在 SciPy v2.0.0 中删除。
# 使用 `scipy.io.matlab` 命名空间来导入以下包含的函数。

# 导入用于处理子模块废弃警告的函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个空列表，用于存储导出的模块名称
__all__: list[str] = []


# 定义一个特殊函数 __dir__()，返回模块导出的所有名称
def __dir__():
    return __all__


# 定义一个特殊函数 __getattr__()，处理动态属性访问
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，发送关于子模块废弃的警告
    return _sub_module_deprecation(sub_package="io.matlab", module="mio5_utils",
                                   private_modules=["_mio5_utils"], all=__all__,
                                   attribute=name)
```