# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools.py`

```
# 该文件不适用于公共使用，并且将在 SciPy v2.0.0 中删除。
# 使用 `scipy.sparse` 命名空间来导入下面包含的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个空列表，用于存储模块中应导出的所有对象名
__all__: list[str] = []


# 定义 __dir__() 函数，当使用 dir() 函数时返回 __all__ 列表内容
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，处理对模块中不存在的属性的访问
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，以警告用户有关模块转移和弃用的信息
    return _sub_module_deprecation(sub_package="sparse", module="sparsetools",
                                   private_modules=["_sparsetools"], all=__all__,
                                   attribute=name)
```