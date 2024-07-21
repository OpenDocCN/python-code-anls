# `.\pytorch\torch\_export\db\examples\autograd_function.py`

```
# 引入torch模块
import torch

# 定义一个自定义的自动求导函数类 MyAutogradFunction，继承自 torch.autograd.Function
class MyAutogradFunction(torch.autograd.Function):
    # 静态方法：前向传播函数，接收上下文对象 ctx 和输入张量 x
    @staticmethod
    def forward(ctx, x):
        # 返回输入张量 x 的克隆
        return x.clone()

    # 静态方法：反向传播函数，接收上下文对象 ctx 和梯度输出 grad_output
    @staticmethod
    def backward(ctx, grad_output):
        # 返回梯度输出 grad_output 加上常数 1
        return grad_output + 1

# 定义一个继承自 torch.nn.Module 的自动求导函数类 AutogradFunction
class AutogradFunction(torch.nn.Module):
    """
    TorchDynamo does not keep track of backward() on autograd functions. We recommend to
    use `allow_in_graph` to mitigate this problem.
    """

    # 前向传播函数，接收输入张量 x
    def forward(self, x):
        # 调用 MyAutogradFunction 中定义的 apply 方法进行前向传播
        return MyAutogradFunction.apply(x)

# 示例输入为一个元组，包含一个大小为 (3, 2) 的随机张量
example_inputs = (torch.randn(3, 2),)

# 创建 AutogradFunction 类的实例作为模型
model = AutogradFunction()
```