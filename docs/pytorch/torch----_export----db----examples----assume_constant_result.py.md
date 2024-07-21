# `.\pytorch\torch\_export\db\examples\assume_constant_result.py`

```py
# 引入 torch 库
import torch
# 引入 torch._dynamo 库，用于假定常量结果
import torch._dynamo as torchdynamo

# 定义一个继承自 torch.nn.Module 的类 AssumeConstantResult
class AssumeConstantResult(torch.nn.Module):
    """
    Applying `assume_constant_result` decorator to burn make non-tracable code as constant.
    应用 `assume_constant_result` 装饰器以将不可追踪的代码变为常量。
    """

    # 使用 torchdynamo.assume_constant_result 装饰器修饰 get_item 方法
    @torchdynamo.assume_constant_result
    def get_item(self, y):
        # 返回 y 的整数值
        return y.int().item()

    # 前向传播方法，接受输入 x 和 y
    def forward(self, x, y):
        # 返回 x 的前 get_item(y) 个元素
        return x[: self.get_item(y)]

# 示例输入
example_inputs = (torch.randn(3, 2), torch.tensor(4))
# 标签
tags = {"torch.escape-hatch"}
# 创建 AssumeConstantResult 类的实例 model
model = AssumeConstantResult()
```