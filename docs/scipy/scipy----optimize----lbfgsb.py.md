# `D:\src\scipysrc\scipy\scipy\optimize\lbfgsb.py`

```
# 导入 `_sub_module_deprecation` 函数，用于处理子模块的废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个列表，包含了模块中公开的类和函数名
__all__ = [  # noqa: F822
    'LbfgsInvHessProduct',  # 将 `LbfgsInvHessProduct` 添加到公开接口中
    'OptimizeResult',        # 将 `OptimizeResult` 添加到公开接口中
    'fmin_l_bfgs_b',         # 将 `fmin_l_bfgs_b` 添加到公开接口中
    'zeros',                 # 将 `zeros` 添加到公开接口中
]

# 定义一个特殊方法 `__dir__()`，返回公开接口的列表 `__all__`
def __dir__():
    return __all__

# 定义一个特殊方法 `__getattr__(name)`，当属性被访问时执行以下操作：
# 使用 `_sub_module_deprecation` 处理子模块 "optimize" 的 "lbfgsb" 模块的属性访问，
# 同时标记 `private_modules=["_lbfgsb_py"]` 为废弃，`all=__all__` 表示支持的全部属性名
def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="lbfgsb",
                                   private_modules=["_lbfgsb_py"], all=__all__,
                                   attribute=name)
```