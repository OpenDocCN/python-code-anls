# `.\pytorch\torch\_export\db\examples\list_contains.py`

```py
# 引入torch模块，用于神经网络和张量操作
import torch

# 创建一个继承自torch.nn.Module的类ListContains，用于定义神经网络模型
class ListContains(torch.nn.Module):
    """
    List containment relation can be checked on a dynamic shape or constants.
    """

    # 定义神经网络模型的前向传播方法
    def forward(self, x):
        # 断言输入张量x的最后一个维度大小为6或2
        assert x.size(-1) in [6, 2]
        # 断言输入张量x的第一个维度大小不在4、5、6中
        assert x.size(0) not in [4, 5, 6]
        # 断言字符串"monkey"不在列表["cow", "pig"]中
        assert "monkey" not in ["cow", "pig"]
        # 返回输入张量x加上自身的结果
        return x + x

# 定义一个示例输入example_inputs，包含一个形状为(3, 2)的随机张量
example_inputs = (torch.randn(3, 2),)

# 定义一个标签tags，用于标识模型的特性，包括"torch.dynamic-shape"、"python.data-structure"、"python.assert"
tags = {"torch.dynamic-shape", "python.data-structure", "python.assert"}

# 创建一个ListContains类的实例model，用于后续模型训练或推断
model = ListContains()
```