# `.\lucidrains\gradnorm-pytorch\gradnorm_pytorch\mocks.py`

```
# 导入 torch 中的 nn 模块
from torch import nn

# 定义一个带有多个损失函数的模拟网络类
class MockNetworkWithMultipleLosses(nn.Module):
    # 初始化函数，接受维度和损失函数数量作为参数
    def __init__(
        self,
        dim,
        num_losses = 2
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 定义网络的主干部分，包括线性层、SiLU 激活函数和另一个线性层
        self.backbone = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # 定义多个判别器，每个判别器都是一个线性层，数量由参数 num_losses 决定
        self.discriminators = nn.ModuleList([
            nn.Linear(dim, 1) for _ in range(num_losses)
        ])

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 将输入 x 通过主干部分得到输出
        backbone_output = self.backbone(x)

        # 初始化损失列表
        losses = []

        # 遍历每个判别器
        for discr in self.discriminators:
            # 计算判别器的输出作为损失
            loss = discr(backbone_output)
            # 将损失的均值添加到损失列表中
            losses.append(loss.mean())

        # 返回损失列表和主干部分的输出
        return losses, backbone_output
```