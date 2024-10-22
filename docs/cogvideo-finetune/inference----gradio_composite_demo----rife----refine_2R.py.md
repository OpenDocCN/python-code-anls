# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\refine_2R.py`

```py
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从本地模块导入 warp 函数
from .warplayer import warp
# 导入 PyTorch 的功能性操作
import torch.nn.functional as F

# 检查 CUDA 是否可用，设置设备为 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义卷积层构造函数
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    # 返回一个顺序容器，包括卷积层和 PReLU 激活函数
    return nn.Sequential(
        nn.Conv2d(
            in_planes,  # 输入通道数
            out_planes,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            stride=stride,  # 步幅
            padding=padding,  # 填充
            dilation=dilation,  # 膨胀率
            bias=True,  # 使用偏置
        ),
        nn.PReLU(out_planes),  # PReLU 激活函数
    )


# 定义反卷积层构造函数
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    # 返回一个顺序容器，包括反卷积层和 PReLU 激活函数
    return nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_channels=in_planes,  # 输入通道数
            out_channels=out_planes,  # 输出通道数
            kernel_size=4,  # 反卷积核大小
            stride=2,  # 步幅
            padding=1,  # 填充
            bias=True  # 使用偏置
        ),
        nn.PReLU(out_planes),  # PReLU 激活函数
    )


# 定义卷积块类
class Conv2(nn.Module):
    # 初始化函数，设置输入输出通道及步幅
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()  # 调用父类初始化
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)  # 第一层卷积
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)  # 第二层卷积

    # 前向传播函数
    def forward(self, x):
        x = self.conv1(x)  # 通过第一层卷积
        x = self.conv2(x)  # 通过第二层卷积
        return x  # 返回结果


c = 16  # 定义通道数常量


# 定义上下文网络类
class Contextnet(nn.Module):
    # 初始化函数
    def __init__(self):
        super(Contextnet, self).__init__()  # 调用父类初始化
        self.conv1 = Conv2(3, c, 1)  # 第一层卷积块，输入通道为 3
        self.conv2 = Conv2(c, 2 * c)  # 第二层卷积块
        self.conv3 = Conv2(2 * c, 4 * c)  # 第三层卷积块
        self.conv4 = Conv2(4 * c, 8 * c)  # 第四层卷积块

    # 前向传播函数
    def forward(self, x, flow):
        x = self.conv1(x)  # 通过第一层卷积块
        # 对 flow 进行双线性插值缩放
        f1 = warp(x, flow)  # 使用 warp 函数处理流
        x = self.conv2(x)  # 通过第二层卷积块
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            * 0.5  # 对 flow 进行缩放
        )
        f2 = warp(x, flow)  # 使用 warp 函数处理流
        x = self.conv3(x)  # 通过第三层卷积块
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            * 0.5  # 对 flow 进行缩放
        )
        f3 = warp(x, flow)  # 使用 warp 函数处理流
        x = self.conv4(x)  # 通过第四层卷积块
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            * 0.5  # 对 flow 进行缩放
        )
        f4 = warp(x, flow)  # 使用 warp 函数处理流
        return [f1, f2, f3, f4]  # 返回所有特征图


# 定义 U-Net 类
class Unet(nn.Module):
    # 初始化函数
    def __init__(self):
        super(Unet, self).__init__()  # 调用父类初始化
        self.down0 = Conv2(17, 2 * c, 1)  # 下采样块，输入通道为 17
        self.down1 = Conv2(4 * c, 4 * c)  # 下采样块
        self.down2 = Conv2(8 * c, 8 * c)  # 下采样块
        self.down3 = Conv2(16 * c, 16 * c)  # 下采样块
        self.up0 = deconv(32 * c, 8 * c)  # 上采样块
        self.up1 = deconv(16 * c, 4 * c)  # 上采样块
        self.up2 = deconv(8 * c, 2 * c)  # 上采样块
        self.up3 = deconv(4 * c, c)  # 上采样块
        self.conv = nn.Conv2d(c, 3, 3, 2, 1)  # 输出层，转换到 3 个通道
    # 定义前向传播函数，接收多张图像及相关特征作为输入
    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        # 将输入图像及其变换结果和其他特征拼接，传入下采样层 down0 进行处理
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        # 将 s0 与 c0 和 c1 的第一个通道拼接，传入下采样层 down1 进行处理
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        # 将 s1 与 c0 和 c1 的第二个通道拼接，传入下采样层 down2 进行处理
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        # 将 s2 与 c0 和 c1 的第三个通道拼接，传入下采样层 down3 进行处理
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        # 将 s3 与 c0 和 c1 的第四个通道拼接，传入上采样层 up0 进行处理
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        # 将 x 与 s2 拼接，传入上采样层 up1 进行处理
        x = self.up1(torch.cat((x, s2), 1))
        # 将 x 与 s1 拼接，传入上采样层 up2 进行处理
        x = self.up2(torch.cat((x, s1), 1))
        # 将 x 与 s0 拼接，传入上采样层 up3 进行处理
        x = self.up3(torch.cat((x, s0), 1))
        # 对最终输出进行卷积操作
        x = self.conv(x)
        # 返回经过 sigmoid 激活函数处理后的输出
        return torch.sigmoid(x)
```