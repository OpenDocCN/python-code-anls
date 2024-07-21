# `.\pytorch\torch\_export\db\examples\scalar_output.py`

```
# 引入 torch 库
import torch

# 从 torch.export 模块导入 Dim 类
from torch.export import Dim

# 创建一个形状为 (3, 2) 的随机张量 x
x = torch.randn(3, 2)

# 创建一个名为 dim1_x 的 Dim 对象
dim1_x = Dim("dim1_x")

# 定义一个名为 ScalarOutput 的 Torch 模块
class ScalarOutput(torch.nn.Module):
    """
    从图中返回标量值是支持的，除了张量输出。符号形状被捕获，排名被专门化。
    """
    def __init__(self):
        super().__init__()

    # 前向传播函数，返回输入张量 x 的第二维度大小加 1
    def forward(self, x):
        return x.shape[1] + 1

# 创建一个 ScalarOutput 类的实例，命名为 model
model = ScalarOutput()

# 定义一个示例输入元组，包含之前创建的张量 x
example_inputs = (x,)

# 创建一个包含 "torch.dynamic-shape" 标签的字典
tags = {"torch.dynamic-shape"}

# 创建一个动态形状信息的嵌套字典，其中 "x" 键对应的值是 {1: dim1_x}
dynamic_shapes = {"x": {1: dim1_x}}
```