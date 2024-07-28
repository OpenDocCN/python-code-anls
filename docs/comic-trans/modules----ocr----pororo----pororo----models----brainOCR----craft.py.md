# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\craft.py`

```py
# 导入PyTorch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 导入自定义模块：Vgg16BN和init_weights
from ._modules import Vgg16BN, init_weights

# 定义双重卷积模块
class DoubleConv(nn.Module):

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super(DoubleConv, self).__init__()
        # 定义双重卷积层序列
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),  # 1x1卷积层
            nn.BatchNorm2d(mid_ch),  # 批量归一化层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),  # 3x3卷积层（带填充）
            nn.BatchNorm2d(out_ch),  # 批量归一化层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )

    def forward(self, x: Tensor):
        x = self.conv(x)
        return x

# 定义CRAFT模型
class CRAFT(nn.Module):

    def __init__(self, pretrained: bool = False, freeze: bool = False) -> None:
        super(CRAFT, self).__init__()

        # 基础网络部分
        self.basenet = Vgg16BN(pretrained, freeze)

        # U型网络部分
        self.upconv1 = DoubleConv(1024, 512, 256)  # 第一层双重卷积
        self.upconv2 = DoubleConv(512, 256, 128)   # 第二层双重卷积
        self.upconv3 = DoubleConv(256, 128, 64)    # 第三层双重卷积
        self.upconv4 = DoubleConv(128, 64, 32)     # 第四层双重卷积

        num_class = 2
        # 分类卷积层序列
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 3x3卷积层（带填充）
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 3x3卷积层（带填充）
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 3x3卷积层（带填充）
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(16, 16, kernel_size=1),  # 1x1卷积层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(16, num_class, kernel_size=1),  # 1x1卷积层，最终分类层
        )

        # 初始化权重
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x: Tensor):
        # 基础网络部分的前向传播
        sources = self.basenet(x)

        # U型网络部分的前向传播
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(
            y,
            size=sources[2].size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(
            y,
            size=sources[3].size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(
            y,
            size=sources[4].size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature
```