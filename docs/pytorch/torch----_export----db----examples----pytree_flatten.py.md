# `.\pytorch\torch\_export\db\examples\pytree_flatten.py`

```
# 引入 torch 库
import torch

# 从 torch.utils 中引入 _pytree 模块，并重命名为 pytree
from torch.utils import _pytree as pytree

# 定义一个继承自 torch.nn.Module 的类 PytreeFlatten
class PytreeFlatten(torch.nn.Module):
    """
    Pytree from PyTorch can be captured by TorchDynamo.
    """

    # 定义类的前向传播方法
    def forward(self, x):
        # 使用 pytree 模块中的 tree_flatten 函数对输入 x 进行展平操作，
        # 返回展平后的张量列表 y 和展平规范 spec
        y, spec = pytree.tree_flatten(x)
        # 返回展平后张量列表 y 中第一个张量加 1 的结果
        return y[0] + 1

# 定义一个示例输入 example_inputs，包含一个元组，元组中包含一个字典，
# 字典的键为 1，对应的值为形状为 (3, 2) 的随机张量，字典的键为 2，对应的值为形状为 (3, 2) 的随机张量
example_inputs = ({1: torch.randn(3, 2), 2: torch.randn(3, 2)},)

# 创建 PytreeFlatten 类的一个实例，命名为 model
model = PytreeFlatten()
```