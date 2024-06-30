# `D:\src\scipysrc\scipy\scipy\io\netcdf.py`

```
# 以下代码不适合公开使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.io` 命名空间来导入下面包含的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含 "netcdf_file" 和 "netcdf_variable"，并忽略 F822 错误
__all__ = ["netcdf_file", "netcdf_variable"]  # noqa: F822


# 定义特殊方法 __dir__()，返回模块可导出的所有名称列表
def __dir__():
    return __all__


# 定义特殊方法 __getattr__(name)，用于获取指定名称的属性
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，传递以下参数：
    # sub_package="io"，module="netcdf"，private_modules=["_netcdf"]，all=__all__，attribute=name
    return _sub_module_deprecation(sub_package="io", module="netcdf",
                                   private_modules=["_netcdf"], all=__all__,
                                   attribute=name)
```