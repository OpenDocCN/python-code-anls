# `D:\src\scipysrc\scipy\scipy\sparse\linalg\isolve.py`

```
# 导入 _sub_module_deprecation 函数，用于处理模块废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含当前模块中公开的函数名称，用于控制导出的内容
__all__ = [  # noqa: F822
    'bicg', 'bicgstab', 'cg', 'cgs', 'gcrotmk', 'gmres',
    'lgmres', 'lsmr', 'lsqr',
    'minres', 'qmr', 'tfqmr', 'test'
]

# 定义 __dir__() 函数，返回当前模块公开的函数列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，当模块属性未找到时调用，用于处理废弃的子模块或函数
def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse.linalg", module="isolve",
                                   private_modules=["_isolve"], all=__all__,
                                   attribute=name)
```