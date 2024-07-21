# `.\pytorch\torch\_export\db\examples\dynamic_shape_view.py`

```py
# 导入torch模块，用于构建和操作神经网络
import torch

# 定义一个继承自torch.nn.Module的类DynamicShapeView，表示动态形状应该传递到视图参数，而不应该固定在导出的图中
class DynamicShapeView(torch.nn.Module):
    """
    Dynamic shapes should be propagated to view arguments instead of being
    baked into the exported graph.
    """

    # 定义模型的前向传播方法，接受输入张量x，并返回变换后的张量
    def forward(self, x):
        # 计算新的张量形状，保留原始张量除了最后一个维度外的所有维度，最后一个维度替换为(2, 5)
        new_x_shape = x.size()[:-1] + (2, 5)
        # 使用新的形状对输入张量x进行视图变换
        x = x.view(*new_x_shape)
        # 对变换后的张量进行维度置换，将第1和第3个维度交换位置
        return x.permute(0, 2, 1)

# 定义一个示例输入example_inputs，包含一个形状为(10, 10)的随机张量
example_inputs = (torch.randn(10, 10),)
# 定义一个标签tags，用于标识此模型支持动态形状
tags = {"torch.dynamic-shape"}
# 创建DynamicShapeView类的一个实例model，用于后续的模型操作
model = DynamicShapeView()
```