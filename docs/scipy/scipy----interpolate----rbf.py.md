# `D:\src\scipysrc\scipy\scipy\interpolate\rbf.py`

```
# 这个文件不适用于公共使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.interpolate` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，指定可导出的模块成员，F822 告诉 linter 忽略未定义的名称警告
__all__ = ["Rbf"]  # noqa: F822

# 定义 __dir__() 函数，返回可导出的模块成员列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，处理动态属性访问
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，生成有关模块废弃的警告信息
    return _sub_module_deprecation(sub_package="interpolate", module="rbf",
                                   private_modules=["_rbf"], all=__all__,
                                   attribute=name)
```