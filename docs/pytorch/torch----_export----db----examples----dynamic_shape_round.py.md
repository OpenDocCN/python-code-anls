# `.\pytorch\torch\_export\db\examples\dynamic_shape_round.py`

```
# 导入 torch 库，这是一个用于深度学习的主要库
# mypy: allow-untyped-defs 用于指示 mypy 在检查时允许未类型化的定义
import torch

# 从 torch._export.db.case 模块导入 SupportLevel 类
from torch._export.db.case import SupportLevel

# 从 torch.export 模块导入 Dim 类
from torch.export import Dim

# 定义一个继承自 torch.nn.Module 的类 DynamicShapeRound，用于处理动态形状的情况
class DynamicShapeRound(torch.nn.Module):
    """
    Calling round on dynamic shapes is not supported.
    动态形状上调用 round 不受支持。
    """

    # 定义 forward 方法，处理模型的前向传播逻辑
    def forward(self, x):
        # 返回张量 x 的前一半（向下取整）
        return x[: round(x.shape[0] / 2)]

# 创建一个形状为 (3, 2) 的随机张量 x
x = torch.randn(3, 2)

# 创建一个 Dim 对象 dim0_x，代表维度 "dim0_x"
dim0_x = Dim("dim0_x")

# 将张量 x 放入元组 example_inputs 中
example_inputs = (x,)

# 创建一个标签集合 tags，包含 "torch.dynamic-shape" 和 "python.builtin"
tags = {"torch.dynamic-shape", "python.builtin"}

# 设置支持级别 support_level 为 SupportLevel.NOT_SUPPORTED_YET，表示尚未支持
support_level = SupportLevel.NOT_SUPPORTED_YET

# 创建动态形状字典 dynamic_shapes，包含键 "x" 和对应的形状映射 {0: dim0_x}
dynamic_shapes = {"x": {0: dim0_x}}

# 创建 DynamicShapeRound 类的实例 model
model = DynamicShapeRound()
```