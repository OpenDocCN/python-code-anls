# `.\pytorch\torch\_export\db\examples\class_method.py`

```
# 引入 torch 模块，用于深度学习任务
import torch

# 定义一个继承自 torch.nn.Module 的类 ClassMethod，表示一个神经网络模型
class ClassMethod(torch.nn.Module):
    """
    Class methods are inlined during tracing.
    """

    # 类方法修饰符 @classmethod，用于定义一个可以通过类调用而非实例调用的方法
    @classmethod
    def method(cls, x):
        # 返回输入 x 加 1 的结果
        return x + 1

    # 类的初始化方法，初始化神经网络的结构
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一个线性层，输入维度为 4，输出维度为 2
        self.linear = torch.nn.Linear(4, 2)

    # 前向传播方法，定义数据从输入到输出的流程
    def forward(self, x):
        # 将输入 x 经过线性层处理
        x = self.linear(x)
        # 返回经过类方法计算的结果，乘以当前类和类方法的结果
        return self.method(x) * self.__class__.method(x) * type(self).method(x)

# 示例输入，一个大小为 (3, 4) 的张量
example_inputs = (torch.randn(3, 4),)

# 创建一个 ClassMethod 类的实例，即一个神经网络模型
model = ClassMethod()
```