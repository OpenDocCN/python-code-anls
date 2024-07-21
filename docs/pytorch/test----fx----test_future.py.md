# `.\pytorch\test\fx\test_future.py`

```
# Owner(s): ["module: fx"]

# 导入未来的注解特性，用于支持类型定义时的前向引用
from __future__ import annotations  # type: ignore[attr-defined]

# 导入类型相关的模块
import typing

# 导入 PyTorch 库
import torch
# 从 torch.fx 模块中导入 symbolic_trace 函数
from torch.fx import symbolic_trace


# 定义类 A，实现了 __call__ 方法，接受一个 torch.Tensor 类型的参数 x，并返回 torch.Tensor 类型的结果
class A:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.add(x, x)


# 定义类 M1，继承自 torch.nn.Module，无前向引用
class M1(torch.nn.Module):
    # 实现 forward 方法，接受两个参数：torch.Tensor 类型的 x 和类 A 的实例 a，并返回 torch.Tensor 类型的结果
    def forward(self, x: torch.Tensor, a: A) -> torch.Tensor:
        return a(x)


# 定义类 M2，继承自 torch.nn.Module，有前向引用
class M2(torch.nn.Module):
    # 实现 forward 方法，接受两个参数：torch.Tensor 类型的 x 和类 A 的实例 a，并返回 torch.Tensor 类型的结果
    def forward(self, x: torch.Tensor, a: A) -> torch.Tensor:
        return a(x)


# 定义类 M3，继承自 torch.nn.Module，使用 typing.List[torch.Tensor] 类型的参数，无前向引用
class M3(torch.nn.Module):
    # 实现 forward 方法，接受两个参数：列表，其中元素为 torch.Tensor 类型，和类 A 的实例 a，并返回 torch.Tensor 类型的结果
    def forward(self, x: typing.List[torch.Tensor], a: A) -> torch.Tensor:
        return a(x[0])


# 定义类 M4，继承自 torch.nn.Module，使用 typing.List[torch.Tensor] 类型的参数，有前向引用
class M4(torch.nn.Module):
    # 实现 forward 方法，接受两个参数：列表，其中元素为 torch.Tensor 类型，和类 A 的实例 a，并返回 torch.Tensor 类型的结果
    def forward(self, x: typing.List[torch.Tensor], a: A) -> torch.Tensor:
        return a(x[0])


# 创建一个形状为 (2, 3) 的随机张量 x
x = torch.rand(2, 3)

# 计算 x + x 的结果，用于后续的比较
ref = torch.add(x, x)

# 对 M1 类进行符号化追踪
traced1 = symbolic_trace(M1())
# 使用追踪后的 M1 类对象调用 forward 方法，传入 x 和 A 类的实例，获取结果 res1
res1 = traced1(x, A())
# 断言追踪后的结果与预期的 ref 相等
assert torch.all(torch.eq(ref, res1))

# 对 M2 类进行符号化追踪
traced2 = symbolic_trace(M2())
# 使用追踪后的 M2 类对象调用 forward 方法，传入 x 和 A 类的实例，获取结果 res2
res2 = traced2(x, A())
# 断言追踪后的结果与预期的 ref 相等
assert torch.all(torch.eq(ref, res2))

# 对 M3 类进行符号化追踪
traced3 = symbolic_trace(M3())
# 使用追踪后的 M3 类对象调用 forward 方法，传入 [x] 列表和 A 类的实例，获取结果 res3
res3 = traced3([x], A())
# 断言追踪后的结果与预期的 ref 相等
assert torch.all(torch.eq(ref, res3))

# 对 M4 类进行符号化追踪
traced4 = symbolic_trace(M4())
# 使用追踪后的 M4 类对象调用 forward 方法，传入 [x] 列表和 A 类的实例，获取结果 res4
res4 = traced4([x], A())
# 断言追踪后的结果与预期的 ref 相等
assert torch.all(torch.eq(ref, res4))
```