# `D:\src\scipysrc\scipy\scipy\integrate\odepack.py`

```
# 此文件不适用于公共使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.integrate` 命名空间来导入下面包含的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# __all__ 列表指定了模块导出的公共接口，这里包括 'odeint' 和 'ODEintWarning'
__all__ = ['odeint', 'ODEintWarning']  # noqa: F822


# 定义一个特殊方法 __dir__()，返回模块的公共接口列表 __all__
def __dir__():
    return __all__


# 定义一个特殊方法 __getattr__(name)，用于处理动态属性访问
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，传递参数说明废弃的子模块是 "integrate"，主模块是 "odepack"
    # 还传递了私有子模块列表 ["_odepack_py"]、全部公共接口列表 __all__，以及要访问的属性名 name
    return _sub_module_deprecation(sub_package="integrate", module="odepack",
                                   private_modules=["_odepack_py"], all=__all__,
                                   attribute=name)
```