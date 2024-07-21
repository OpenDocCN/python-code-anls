# `.\pytorch\test\typing\reveal\size.py`

```
# 导入 PyTorch 库
import torch

# 创建一个空列表来存放 PyTorch 张量
input = []

# 向列表中添加一个一维张量，包含四个元素：1.0, 2.0, 3.0, 4.0
input.append(torch.tensor([1.0, 2.0, 3.0, 4.0]))

# 向列表中添加一个二维张量，包含一个行向量：[1.0, 2.0, 3.0, 4.0]
input.append(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

# 向列表中添加一个三维张量，包含一个深度为一的张量：[[1.0, 2.0, 3.0, 4.0]]
input.append(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))

# 输出第一个张量的第一维度的类型信息，预期是一个整数 (E: int)
reveal_type(input[0].shape[0])

# 输出第二个张量的第二维度的类型信息，预期是一个整数 (E: int)
reveal_type(input[1].shape[1])

# 输出第三个张量的第三维度的类型信息，预期是一个整数 (E: int)
reveal_type(input[2].shape[2])
```