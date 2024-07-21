# `.\pytorch\torch\_export\db\examples\torch_sym_min.py`

```
# 导入torch模块
import torch

# 从torch._export.db.case模块导入SupportLevel类
from torch._export.db.case import SupportLevel

# 定义一个继承自torch.nn.Module的类TorchSymMin，用于自定义torch.sym_min操作符
class TorchSymMin(torch.nn.Module):
    """
    torch.sym_min operator is not supported in export.
    """
    
    # 定义forward方法，接受输入x，并返回其求和与torch.sym_min的结果
    def forward(self, x):
        return x.sum() + torch.sym_min(x.size(0), 100)

# 定义一个示例输入example_inputs，包含一个形状为(3, 2)的随机张量
example_inputs = (torch.randn(3, 2),)

# 定义一个标签tags，用于标识torch.operator
tags = {"torch.operator"}

# 定义一个支持级别support_level，表示当前torch.sym_min操作符尚未被支持
support_level = SupportLevel.NOT_SUPPORTED_YET

# 创建一个TorchSymMin类的实例model
model = TorchSymMin()
```