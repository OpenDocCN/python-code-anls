# `D:\src\scipysrc\scipy\scipy\optimize\cobyla.py`

```
# 这个文件不适合公共使用，并且将在 SciPy v2.0.0 版本中移除。
# 使用 `scipy.optimize` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义模块的公开接口列表，用于控制模块中哪些内容可以被导入
__all__ = [  # noqa: F822
    'OptimizeResult',  # 导出 OptimizeResult 类
    'fmin_cobyla',      # 导出 fmin_cobyla 函数
]

# 定义一个特殊方法 __dir__()，返回模块的公开接口列表 __all__
def __dir__():
    return __all__

# 定义一个特殊方法 __getattr__(name)，用于动态获取模块属性
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，标记子模块的过时使用
    return _sub_module_deprecation(sub_package="optimize", module="cobyla",
                                   private_modules=["_cobyla_py"], all=__all__,
                                   attribute=name)
```