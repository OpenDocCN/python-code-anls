# `.\pytorch\test\linear.py`

```
import torch  # 导入PyTorch库


class LinearMod(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类(torch.nn.Linear)的初始化方法

    def forward(self, input):
        return torch._C._nn.linear(input, self.weight, self.bias)  # 使用底层C++实现的线性函数进行前向传播计算


print(torch.jit.trace(LinearMod(20, 20), torch.rand([20, 20])).graph)  # 对定义的LinearMod模型进行追踪，并输出其计算图
```