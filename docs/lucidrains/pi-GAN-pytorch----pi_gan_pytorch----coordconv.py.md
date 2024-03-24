# `.\lucidrains\pi-GAN-pytorch\pi_gan_pytorch\coordconv.py`

```
# 从给定链接中导入所需的库
# https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
import torch
import torch.nn as nn

# 定义一个名为AddCoords的类，继承自nn.Module
class AddCoords(nn.Module):

    # 初始化函数，接受一个布尔值参数with_r，默认为False
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    # 前向传播函数，接受一个输入张量input_tensor
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        # 获取输入张量的维度信息
        batch_size, _, x_dim, y_dim = input_tensor.size()

        # 创建xx_channel和yy_channel张量，用于表示坐标信息
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        # 对坐标信息进行归一化处理
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        # 将坐标信息映射到[-1, 1]范围内
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        # 将坐标信息扩展到batch维度，并转置维度
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        # 将坐标信息与输入张量拼接在一起
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        # 如果with_r为True，则计算距离信息并拼接到结果中
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

# 定义一个名为CoordConv的类，继承自nn.Module
class CoordConv(nn.Module):

    # 初始化函数，接受输入通道数in_channels、输出通道数out_channels和其他关键字参数
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        # 创建AddCoords对象，传入with_r参数
        self.addcoords = AddCoords(with_r=with_r)
        # 计算输入尺寸大小
        in_size = in_channels+2
        if with_r:
            in_size += 1
        # 创建卷积层对象
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    # 前向传播函数，接受输入张量x
    def forward(self, x):
        # 将输入张量经过AddCoords处理后再经过卷积层处理
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
```