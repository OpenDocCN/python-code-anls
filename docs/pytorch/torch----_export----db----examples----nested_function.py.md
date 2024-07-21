# `.\pytorch\torch\_export\db\examples\nested_function.py`

```py
# 引入torch模块，用于神经网络相关操作
# mypy: allow-untyped-defs 允许不带类型注释的函数定义
import torch

# 定义一个继承自torch.nn.Module的类NestedFunction，表示一个嵌套函数模块
class NestedFunction(torch.nn.Module):
    """
    Nested functions are traced through. Side effects on global captures
    are not supported though.
    嵌套函数会被追踪。全局捕获的副作用是不被支持的。
    """

    # 定义该模块的前向传播函数
    def forward(self, a, b):
        # 计算a和b的和，并赋值给x
        x = a + b
        # 计算a和b的差，并赋值给z
        z = a - b

        # 定义一个闭包函数closure，接受参数y
        def closure(y):
            # 声明x为非局部变量，以便在闭包中修改外层函数中的x
            nonlocal x
            # 对外层函数中的x进行自增操作
            x += 1
            # 返回计算结果x * y + z
            return x * y + z

        # 调用闭包函数closure，并将x作为参数传递给它，返回闭包函数的结果
        return closure(x)

# 定义一个示例输入example_inputs，包含两个随机生成的torch张量
example_inputs = (torch.randn(3, 2), torch.randn(2))
# 定义一个标签tags，包含字符串"python.closure"
tags = {"python.closure"}
# 创建一个NestedFunction类的实例model，表示一个嵌套函数模块的模型
model = NestedFunction()
```