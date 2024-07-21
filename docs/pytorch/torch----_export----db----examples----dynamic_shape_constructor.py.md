# `.\pytorch\torch\_export\db\examples\dynamic_shape_constructor.py`

```
# 引入 torch 库
import torch

# 定义一个继承自 torch.nn.Module 的类 DynamicShapeConstructor
class DynamicShapeConstructor(torch.nn.Module):
    """
    Tensor constructors should be captured with dynamic shape inputs rather
    than being baked in with static shape.
    """
    
    # 定义 forward 方法，处理输入 x
    def forward(self, x):
        # 返回一个形状为 x.shape[0] * 2 的全零张量
        return torch.zeros(x.shape[0] * 2)

# 示例输入，包含一个形状为 (3, 2) 的随机张量
example_inputs = (torch.randn(3, 2),)

# 标签集合，包含字符串 "torch.dynamic-shape"
tags = {"torch.dynamic-shape"}

# 创建 DynamicShapeConstructor 类的实例，用于后续的模型调用
model = DynamicShapeConstructor()
```