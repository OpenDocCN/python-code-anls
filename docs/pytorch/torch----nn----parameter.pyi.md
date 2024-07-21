# `.\pytorch\torch\nn\parameter.pyi`

```
# mypy: allow-untyped-defs
# 导入 TypeGuard 类型守卫，用于类型检查
from typing_extensions import TypeGuard

# 导入 Tensor 类型，device 类型，dtype 类型
from torch import device, dtype, Tensor

# Parameter 类，继承自 Tensor 类
class Parameter(Tensor):
    # 参数初始化方法
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...

# 判断是否为延迟加载参数的函数 is_lazy
def is_lazy(
    param: Tensor,  # 参数 param 的类型为 Tensor
) -> TypeGuard[UninitializedParameter | UninitializedBuffer]: ...

# UninitializedParameter 类，继承自 Tensor 类
class UninitializedParameter(Tensor):
    # 参数初始化方法
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...
    # materialize 方法，用于实例化未初始化的参数
    def materialize(
        self,
        shape: tuple[int, ...],  # 形状参数为元组类型，包含整数
        device: device | None = None,  # 设备参数，可以为 None 或 device 类型
        dtype: dtype | None = None,  # 数据类型参数，可以为 None 或 dtype 类型
    ) -> None: ...

# UninitializedBuffer 类，继承自 Tensor 类
class UninitializedBuffer(Tensor):
    # 参数初始化方法
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...
    # materialize 方法，用于实例化未初始化的缓冲区
    def materialize(
        self,
        shape: tuple[int, ...],  # 形状参数为元组类型，包含整数
        device: device | None = None,  # 设备参数，可以为 None 或 device 类型
        dtype: dtype | None = None,  # 数据类型参数，可以为 None 或 dtype 类型
    ) -> None: ...
```