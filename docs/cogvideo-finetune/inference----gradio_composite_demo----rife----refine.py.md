# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\refine.py`

```py
# 导入 PyTorch 库及其神经网络模块
import torch
import torch.nn as nn
# 从当前模块导入 warp 函数
from .warplayer import warp
# 导入 PyTorch 的功能性模块
import torch.nn.functional as F

# 根据是否可用 CUDA 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义卷积层的构造函数
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    # 返回一个包含卷积层和 PReLU 激活函数的序列
    return nn.Sequential(
        nn.Conv2d(
            in_planes,  # 输入通道数
            out_planes,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            stride=stride,  # 步幅
            padding=padding,  # 填充
            dilation=dilation,  # 膨胀
            bias=True,  # 是否使用偏置
        ),
        nn.PReLU(out_planes),  # 添加 PReLU 激活函数
    )


# 定义反卷积层的构造函数
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    # 返回一个包含反卷积层和 PReLU 激活函数的序列
    return nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True
        ),
        nn.PReLU(out_planes),  # 添加 PReLU 激活函数
    )


# 定义 Conv2 模块
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        # 初始化第一个卷积层
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        # 初始化第二个卷积层
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        # 经过第一个卷积层
        x = self.conv1(x)
        # 经过第二个卷积层
        x = self.conv2(x)
        # 返回输出
        return x


c = 16  # 定义通道数的常量


# 定义 Contextnet 模块
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        # 初始化四个 Conv2 模块
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x, flow):
        # 经过第一个卷积层
        x = self.conv1(x)
        # 对光流进行双线性插值缩放并乘以0.5
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            * 0.5
        )
        # 应用 warp 函数处理 x 和 flow
        f1 = warp(x, flow)
        # 经过第二个卷积层
        x = self.conv2(x)
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            * 0.5
        )
        f2 = warp(x, flow)
        # 经过第三个卷积层
        x = self.conv3(x)
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            * 0.5
        )
        f3 = warp(x, flow)
        # 经过第四个卷积层
        x = self.conv4(x)
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            * 0.5
        )
        f4 = warp(x, flow)
        # 返回四个处理后的特征
        return [f1, f2, f3, f4]


# 定义 Unet 模块
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # 初始化四个下采样 Conv2 模块
        self.down0 = Conv2(17, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        # 初始化四个上采样 deconv 模块
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        # 初始化最后的卷积层以生成输出
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)
    # 定义前向传播方法，接收多张图像和特征
        def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
            # 将输入图像和特征拼接，并通过下采样层 down0 处理
            s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
            # 将 s0 与 c0 和 c1 的第一层特征拼接，并通过下采样层 down1 处理
            s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
            # 将 s1 与 c0 和 c1 的第二层特征拼接，并通过下采样层 down2 处理
            s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
            # 将 s2 与 c0 和 c1 的第三层特征拼接，并通过下采样层 down3 处理
            s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
            # 将 s3 与 c0 和 c1 的第四层特征拼接，并通过上采样层 up0 处理
            x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
            # 将上一步的结果与 s2 拼接，并通过上采样层 up1 处理
            x = self.up1(torch.cat((x, s2), 1))
            # 将上一步的结果与 s1 拼接，并通过上采样层 up2 处理
            x = self.up2(torch.cat((x, s1), 1))
            # 将上一步的结果与 s0 拼接，并通过上采样层 up3 处理
            x = self.up3(torch.cat((x, s0), 1))
            # 通过卷积层处理最终结果
            x = self.conv(x)
            # 返回经过 sigmoid 激活函数处理的输出
            return torch.sigmoid(x)
```