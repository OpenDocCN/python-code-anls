# `.\pytorch\torch\_export\db\examples\dynamic_shape_if_guard.py`

```py
# 导入torch库，用于深度学习模型和张量操作
import torch

# 定义一个继承自torch.nn.Module的类DynamicShapeIfGuard，表示一个动态形状条件判断的模块
class DynamicShapeIfGuard(torch.nn.Module):
    """
    `if` statement with backed dynamic shape predicate will be specialized into
    one particular branch and generate a guard. However, export will fail if the
    the dimension is marked as dynamic shape from higher level API.
    """
    # 定义前向传播函数，接收输入参数x
    def forward(self, x):
        # 如果输入张量x的第一个维度的大小为3，则执行以下操作
        if x.shape[0] == 3:
            # 返回x的余弦值张量
            return x.cos()
        
        # 如果条件不满足，返回x的正弦值张量
        return x.sin()

# 定义一个示例输入，包含一个大小为(3, 2, 2)的随机张量元组
example_inputs = (torch.randn(3, 2, 2),)

# 定义一个标签集合，用于标记该模型相关特性，包括"torch.dynamic-shape"和"python.control-flow"
tags = {"torch.dynamic-shape", "python.control-flow"}

# 创建一个DynamicShapeIfGuard类的实例，表示一个动态形状条件判断的模型
model = DynamicShapeIfGuard()
```