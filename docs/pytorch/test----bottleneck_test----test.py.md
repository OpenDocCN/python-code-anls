# `.\pytorch\test\bottleneck_test\test.py`

```
# 导入PyTorch库
import torch

# 创建一个3x3的张量x，所有元素为1，设置requires_grad=True表示我们想要计算梯度
x = torch.ones((3, 3), requires_grad=True)

# 对表达式3 * x求和，并进行反向传播
(3 * x).sum().backward()
```