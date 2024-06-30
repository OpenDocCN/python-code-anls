# `D:\src\scipysrc\scipy\scipy\integrate\quadpack.py`

```
# 这个文件不适合公共使用，并且将在 SciPy v2.0.0 中删除。
# 使用 `scipy.integrate` 命名空间来导入以下所包含的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 指定模块的公开接口列表
__all__ = [  # noqa: F822
    "quad",         # 导出 quad 函数
    "dblquad",      # 导出 dblquad 函数
    "tplquad",      # 导出 tplquad 函数
    "nquad",        # 导出 nquad 函数
    "IntegrationWarning",   # 导出 IntegrationWarning 类
]


# 定义一个 __dir__ 函数，返回当前模块的公开接口列表
def __dir__():
    return __all__


# 定义一个 __getattr__ 函数，用于在动态获取属性时进行模块的逐步弃用处理
def __getattr__(name):
    return _sub_module_deprecation(sub_package="integrate", module="quadpack",
                                   private_modules=["_quadpack_py"], all=__all__,
                                   attribute=name)
```