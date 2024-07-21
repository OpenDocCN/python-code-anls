# `.\pytorch\test\jit\fixtures_srcs\fixtures_src.py`

```
# 导入 Union 类型以支持多种可能的输入类型
from typing import Union

# 导入 PyTorch 库
import torch


# 定义一个继承自 nn.Module 的类 TestVersionedDivTensorExampleV7
class TestVersionedDivTensorExampleV7(torch.nn.Module):
    # 定义 forward 方法，接受两个参数 a 和 b
    def forward(self, a, b):
        # 使用 "/" 操作符进行张量的除法运算
        result_0 = a / b
        # 使用 torch.div() 方法进行张量的除法运算
        result_1 = torch.div(a, b)
        # 使用张量对象的 .div() 方法进行除法运算
        result_2 = a.div(b)
        # 返回三种不同除法运算结果的元组
        return result_0, result_1, result_2


# 定义一个继承自 nn.Module 的类 TestVersionedLinspaceV7
class TestVersionedLinspaceV7(torch.nn.Module):
    # 定义 forward 方法，接受两个参数 a 和 b，类型可以是 int、float 或 complex
    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
        # 使用 torch.linspace() 方法生成在 [a, b] 区间内的均匀间隔的数值，默认步数为 5
        c = torch.linspace(a, b, steps=5)
        # 使用 torch.linspace() 方法生成在 [a, b] 区间内的均匀间隔的数值，步数由参数自动确定
        d = torch.linspace(a, b)
        # 返回生成的两个张量 c 和 d
        return c, d


# 定义一个继承自 nn.Module 的类 TestVersionedLinspaceOutV7
class TestVersionedLinspaceOutV7(torch.nn.Module):
    # 定义 forward 方法，接受三个参数 a、b 和 out，其中 out 是一个预先分配内存的张量
    def forward(
        self,
        a: Union[int, float, complex],
        b: Union[int, float, complex],
        out: torch.Tensor,
    ):
        # 使用 torch.linspace() 方法生成在 [a, b] 区间内的均匀间隔的数值，并将结果存储在预先分配的张量 out 中
        return torch.linspace(a, b, out=out)


# 定义一个继承自 nn.Module 的类 TestVersionedLogspaceV8
class TestVersionedLogspaceV8(torch.nn.Module):
    # 定义 forward 方法，接受两个参数 a 和 b，类型可以是 int、float 或 complex
    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
        # 使用 torch.logspace() 方法生成在 [a, b] 区间内的对数间隔的数值，默认步数为 5
        c = torch.logspace(a, b, steps=5)
        # 使用 torch.logspace() 方法生成在 [a, b] 区间内的对数间隔的数值，步数由参数自动确定
        d = torch.logspace(a, b)
        # 返回生成的两个张量 c 和 d
        return c, d


# 定义一个继承自 nn.Module 的类 TestVersionedLogspaceOutV8
class TestVersionedLogspaceOutV8(torch.nn.Module):
    # 定义 forward 方法，接受三个参数 a、b 和 out，其中 out 是一个预先分配内存的张量
    def forward(
        self,
        a: Union[int, float, complex],
        b: Union[int, float, complex],
        out: torch.Tensor,
    ):
        # 使用 torch.logspace() 方法生成在 [a, b] 区间内的对数间隔的数值，并将结果存储在预先分配的张量 out 中
        return torch.logspace(a, b, out=out)


# 定义一个继承自 nn.Module 的类 TestVersionedGeluV9
class TestVersionedGeluV9(torch.nn.Module):
    # 定义 forward 方法，接受一个参数 x
    def forward(self, x):
        # 使用 torch._C._nn.gelu() 方法对输入张量 x 进行 GELU 激活函数计算
        return torch._C._nn.gelu(x)


# 定义一个继承自 nn.Module 的类 TestVersionedGeluOutV9
class TestVersionedGeluOutV9(torch.nn.Module):
    # 定义 forward 方法，接受一个参数 x
    def forward(self, x):
        # 创建一个与输入张量 x 相同大小的全零张量 out
        out = torch.zeros_like(x)
        # 使用 torch._C._nn.gelu() 方法对输入张量 x 进行 GELU 激活函数计算，并将结果存储在预先分配的张量 out 中
        return torch._C._nn.gelu(x, out=out)


# 定义一个继承自 nn.Module 的类 TestVersionedRandomV10
class TestVersionedRandomV10(torch.nn.Module):
    # 定义 forward 方法，接受一个参数 x
    def forward(self, x):
        # 创建一个与输入张量 x 相同大小的全零张量 out
        out = torch.zeros_like(x)
        # 在张量 out 中生成随机数，范围为 [0, 10)
        return out.random_(0, 10)


# 定义一个继承自 nn.Module 的类 TestVersionedRandomFuncV10
class TestVersionedRandomFuncV10(torch.nn.Module):
    # 定义 forward 方法，接受一个参数 x
    def forward(self, x):
        # 创建一个与输入张量 x 相同大小的全零张量 out
        out = torch.zeros_like(x)
        # 使用 .random() 方法在张量 out 中生成随机数，范围为 [0, 10)
        return out.random(0, 10)


# 定义一个继承自 nn.Module 的类 TestVersionedRandomOutV10
class TestVersionedRandomOutV10(torch.nn.Module):
    # 定义 forward 方法，接受一个参数 x
    def forward(self, x):
        # 创建一个与输入张量 x 相同大小的全零张量 x
        x = torch.zeros_like(x)
        # 创建一个与输入张量 x 相同大小的全零张量 out
        out = torch.zeros_like(x)
        # 使用 .random() 方法在张量 x 中生成随机数，范围为 [0, 10)，并将结果存储在预先分配的张量 out 中
        x.random(0, 10, out=out)
        # 返回生成的随机数张量 out
        return out
```