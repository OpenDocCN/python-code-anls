# `.\numpy\numpy\exceptions.pyi`

```py
# 导入typing模块中的overload函数，用于支持函数重载
from typing import overload

# 定义__all__列表，声明在模块中公开的成员
__all__: list[str]

# 定义复数警告类，继承自RuntimeWarning
class ComplexWarning(RuntimeWarning): ...

# 定义模块废弃警告类，继承自DeprecationWarning
class ModuleDeprecationWarning(DeprecationWarning): ...

# 定义可见废弃警告类，继承自UserWarning
class VisibleDeprecationWarning(UserWarning): ...

# 定义秩警告类，继承自RuntimeWarning
class RankWarning(RuntimeWarning): ...

# 定义过于困难错误类，继承自RuntimeError
class TooHardError(RuntimeError): ...

# 定义数据类型提升错误类，继承自TypeError
class DTypePromotionError(TypeError): ...

# 定义轴错误类，继承自ValueError和IndexError
class AxisError(ValueError, IndexError):
    # 轴的值，可能是None或整数
    axis: None | int
    # 数组的维度，可能是None或整数
    ndim: None | int
    
    # 重载构造函数，支持字符串轴的初始化
    @overload
    def __init__(self, axis: str, ndim: None = ..., msg_prefix: None = ...) -> None: ...
    
    # 重载构造函数，支持整数轴和维度的初始化
    @overload
    def __init__(self, axis: int, ndim: int, msg_prefix: None | str = ...) -> None: ...
    
    # 返回轴错误的字符串描述
    def __str__(self) -> str: ...
```