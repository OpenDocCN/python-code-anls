# `.\pytorch\torch\_C\_functions.pyi`

```
from typing import AnyStr

from torch import Tensor

# 定义一个名为UndefinedGrad的类，表示未定义的梯度
class UndefinedGrad:
    # 初始化方法，无需额外操作，使用Ellipsis占位
    def __init__(self) -> None: ...

    # 调用实例时的方法，接受多个Tensor类型的输入，并返回Tensor类型的列表
    def __call__(self, *inputs: Tensor) -> list[Tensor]: ...

# 定义一个名为DelayedError的类，表示延迟错误
class DelayedError:
    # 初始化方法，接受一个消息msg和一个整数num_inputs作为参数
    def __init__(self, msg: AnyStr, num_inputs: int) -> None: ...

    # 调用实例时的方法，接受一个Tensor类型的列表作为输入，返回一个Tensor类型的列表
    def __call__(self, inputs: list[Tensor]) -> list[Tensor]: ...
```