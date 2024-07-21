# `.\pytorch\test\package\package_a\fake_interface.py`

```
# 导入PyTorch库
import torch
# 从torch库中导入Tensor类
from torch import Tensor

# 定义一个接口ModuleInterface，继承自torch.nn.Module
@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    # 定义接口方法one，接收两个Tensor类型参数，返回一个Tensor类型结果
    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        pass

# 定义一个类OrigModule，继承自torch.nn.Module，实现了ModuleInterface接口
class OrigModule(torch.nn.Module):
    """A module that implements ModuleInterface."""

    # 实现ModuleInterface接口中的one方法，计算inp1和inp2的和再加1
    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        return inp1 + inp2 + 1

    # 定义一个方法two，接收一个Tensor类型参数，返回一个Tensor类型结果，加2
    def two(self, input: Tensor) -> Tensor:
        return input + 2

    # 重写父类的forward方法，接收一个Tensor类型参数，返回一个Tensor类型结果
    def forward(self, input: Tensor) -> Tensor:
        # 计算input与self.one(input, input)的和再加1，并返回结果
        return input + self.one(input, input) + 1

# 定义一个类NewModule，继承自torch.nn.Module，实现了ModuleInterface接口
class NewModule(torch.nn.Module):
    """A *different* module that implements ModuleInterface."""

    # 实现ModuleInterface接口中的one方法，计算inp1和(inp2+1)的乘积再加1
    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        return inp1 * inp2 + 1

    # 重写父类的forward方法，接收一个Tensor类型参数，返回一个Tensor类型结果
    def forward(self, input: Tensor) -> Tensor:
        # 调用self.one(input, input + 1)，并返回结果
        return self.one(input, input + 1)

# 定义一个类UsesInterface，继承自torch.nn.Module
class UsesInterface(torch.nn.Module):
    # 声明一个proxy_mod属性，类型为ModuleInterface接口
    proxy_mod: ModuleInterface

    # 构造方法，初始化时创建一个OrigModule对象赋值给proxy_mod属性
    def __init__(self):
        super().__init__()
        self.proxy_mod = OrigModule()

    # 重写父类的forward方法，接收一个Tensor类型参数，返回一个Tensor类型结果
    def forward(self, input: Tensor) -> Tensor:
        # 调用self.proxy_mod的one方法，传入input和input作为参数，并返回结果
        return self.proxy_mod.one(input, input)
```