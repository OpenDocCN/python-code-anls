# `D:\src\scipysrc\scipy\scipy\stats\mvn.py`

```
# 该文件不适用于公共使用，并将在 SciPy v2.0.0 中删除。
# 使用 `scipy.stats` 命名空间来导入下面包含的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 初始化空列表 __all__，用于存储模块中应被导入的公共成员
__all__: list[str] = []


# 定义 __dir__() 函数，返回模块中公共成员列表
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，用于在访问不存在的属性时触发
# 它调用 _sub_module_deprecation 函数来处理对 "mvn" 子模块的访问，
# 根据给定参数进行转发和警告处理
def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="mvn",
                                   private_modules=["_mvn"], all=__all__,
                                   attribute=name)
```