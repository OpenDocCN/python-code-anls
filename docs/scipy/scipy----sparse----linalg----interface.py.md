# `D:\src\scipysrc\scipy\scipy\sparse\linalg\interface.py`

```
# 导入 _sub_module_deprecation 函数，该函数用于处理子模块的废弃警告和提示信息
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，用于指定模块中公开的接口名称
__all__ = [
    'LinearOperator', 'aslinearoperator',
]


# 定义特殊方法 __dir__()，返回模块中公开的所有接口名称列表
def __dir__():
    return __all__


# 定义特殊方法 __getattr__(name)，在当前模块中未找到属性时调用
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，生成废弃警告消息，指定子模块为 sparse.linalg，
    # 主模块为 interface，私有模块为 _interface，公开接口为 __all__ 中列出的接口名称
    return _sub_module_deprecation(sub_package="sparse.linalg", module="interface",
                                   private_modules=["_interface"], all=__all__,
                                   attribute=name)
```