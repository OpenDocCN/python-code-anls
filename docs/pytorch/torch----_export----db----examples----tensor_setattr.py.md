# `.\pytorch\torch\_export\db\examples\tensor_setattr.py`

```py
# 声明允许未标注类型定义的标志
# mypy: allow-untyped-defs
# 导入PyTorch库
import torch

# 定义一个继承自torch.nn.Module的类TensorSetattr
class TensorSetattr(torch.nn.Module):
    """
    setattr() call onto tensors is not supported.
    setattr()调用不支持在张量上进行。
    """

    # 定义类的前向传播方法
    def forward(self, x, attr):
        # 使用setattr()方法在张量x上设置名为attr的属性，并赋予随机生成的3x2张量值
        setattr(x, attr, torch.randn(3, 2))
        # 返回设置属性后的张量x加上4的结果
        return x + 4

# 创建一个示例输入元组example_inputs，包含一个形状为3x2的随机张量和一个字符串"attr"
example_inputs = (torch.randn(3, 2), "attr")

# 创建一个标签字典tags，包含一个"python.builtin"的标签
tags = {"python.builtin"}

# 创建一个TensorSetattr类的实例model
model = TensorSetattr()
```