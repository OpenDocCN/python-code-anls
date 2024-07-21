# `.\pytorch\torch\_export\db\examples\dynamic_shape_map.py`

```py
# 引入 torch 库
import torch

# 从 functorch 库中引入 map 函数
from functorch.experimental.control_flow import map

# 定义一个继承自 torch.nn.Module 的类 DynamicShapeMap
class DynamicShapeMap(torch.nn.Module):
    """
    functorch map() maps a function over the first tensor dimension.
    """

    # 定义模型的前向传播方法
    def forward(self, xs, y):
        # 定义一个内部函数 body，对输入的两个张量元素逐元素相加
        def body(x, y):
            return x + y

        # 使用 functorch 的 map 函数，将 body 函数映射到 xs 和 y 上
        return map(body, xs, y)

# 创建一个示例输入 example_inputs，包括两个张量：一个形状为 (3, 2)，另一个形状为 (2,)
example_inputs = (torch.randn(3, 2), torch.randn(2))

# 定义一个包含字符串标签的集合 tags
tags = {"torch.dynamic-shape", "torch.map"}

# 创建一个 DynamicShapeMap 类的实例 model
model = DynamicShapeMap()
```