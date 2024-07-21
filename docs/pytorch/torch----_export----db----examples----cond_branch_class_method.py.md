# `.\pytorch\torch\_export\db\examples\cond_branch_class_method.py`

```py
# 引入torch模块，这是PyTorch深度学习框架的主要模块之一
import torch

# 从functorch.experimental.control_flow模块中导入cond函数，用于条件控制流
from functorch.experimental.control_flow import cond

# 定义一个继承自torch.nn.Module的子模块MySubModule
class MySubModule(torch.nn.Module):
    def foo(self, x):
        # 返回输入张量x的余弦值
        return x.cos()

    def forward(self, x):
        # 调用foo方法对输入x进行处理
        return self.foo(x)

# 定义一个继承自torch.nn.Module的模块CondBranchClassMethod
class CondBranchClassMethod(torch.nn.Module):
    """
    cond()函数中传递的分支函数（true_fn和false_fn）必须遵循以下规则:
      - 两个分支必须接受相同的参数，这些参数也必须与传递给cond的分支参数匹配。
      - 两个分支必须返回一个单一的张量。
      - 返回的张量必须具有相同的张量元数据，例如形状和数据类型。
      - 分支函数可以是自由函数、嵌套函数、lambda函数或类方法。
      - 分支函数不能有闭包变量。
      - 输入或全局变量上不允许原地变异。
    
    此示例演示了在cond()中使用类方法的情况。

    注意: 如果pred在批次大小小于2的维度上进行测试，它将被特化。
    """

    def __init__(self):
        super().__init__()
        # 创建MySubModule实例作为CondBranchClassMethod的子模块
        self.subm = MySubModule()

    def bar(self, x):
        # 返回输入张量x的正弦值
        return x.sin()

    def forward(self, x):
        # 调用cond()函数，根据条件x.shape[0] <= 2选择执行self.subm.forward还是self.bar
        return cond(x.shape[0] <= 2, self.subm.forward, self.bar, [x])

# 创建一个示例输入，包含一个形状为(3,)的随机张量
example_inputs = (torch.randn(3),)

# 定义一个标签集合，包括"torch.cond"和"torch.dynamic-shape"
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}

# 创建CondBranchClassMethod的实例，即模型对象
model = CondBranchClassMethod()
```