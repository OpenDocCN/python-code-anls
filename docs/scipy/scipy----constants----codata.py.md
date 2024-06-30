# `D:\src\scipysrc\scipy\scipy\constants\codata.py`

```
# 引入 `_sub_module_deprecation` 函数，用于处理子模块的废弃警告
# 导入以下函数到当前命名空间，这些函数将被公开访问
# 被忽略的 F822 错误是为了禁止检查 PEP-8 的规则
from scipy._lib.deprecation import _sub_module_deprecation

# __all__ 列表定义了模块中公开的接口，这些接口可以通过 `from module import *` 形式导入
__all__ = [
    'physical_constants', 'value', 'unit', 'precision', 'find',
    'ConstantWarning', 'k', 'c',
]


# 定义一个特殊方法 `__dir__()`，返回模块中公开的接口列表
def __dir__():
    return __all__


# 定义一个特殊方法 `__getattr__(name)`，用于在访问不存在的属性时触发
# 调用 `_sub_module_deprecation()` 函数，传入参数以处理废弃警告
# 具体地，将子模块指定为 "constants"，模块指定为 "codata"
# 私有模块 "_codata" 也指定为私有模块列表
# `all` 参数指定当前模块中所有可用的公共接口，`attribute` 参数传入请求的属性名
def __getattr__(name):
    return _sub_module_deprecation(sub_package="constants", module="codata",
                                   private_modules=["_codata"], all=__all__,
                                   attribute=name)
```