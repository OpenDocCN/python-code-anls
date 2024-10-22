# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\IFNet_HDv3.py`

```py
# 导入 PyTorch 和其神经网络模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 从同一目录导入 warp 函数
from .warplayer import warp

# 检查是否可用 CUDA，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义卷积层的构造函数
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        # 创建一个卷积层
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        # 使用 PReLU 激活函数
        nn.PReLU(out_planes),
    )


# 定义卷积层加批量归一化的构造函数
def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        # 创建卷积层，不使用偏置
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        # 添加批量归一化层
        nn.BatchNorm2d(out_planes),
        # 使用 PReLU 激活函数
        nn.PReLU(out_planes),
    )


# 定义 IFBlock 类，继承自 nn.Module
class IFBlock(nn.Module):
    # 初始化方法，接受输入通道数和常数 c
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        # 定义第一个卷积序列
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        # 定义多个卷积块
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))
        # 定义反卷积层，恢复特征图尺寸
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1),
        )
        # 定义另一个反卷积层，输出单通道特征
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1),
        )

    # 前向传播方法
    def forward(self, x, flow, scale=1):
        # 调整输入图像尺寸
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False
        )
        # 调整光流尺寸并缩放
        flow = (
            F.interpolate(
                flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False
            )
            * 1.0
            / scale
        )
        # 连接 x 和 flow，经过卷积处理
        feat = self.conv0(torch.cat((x, flow), 1))
        # 通过多个卷积块进行特征增强
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        # 处理 flow
        flow = self.conv1(feat)
        # 处理 mask
        mask = self.conv2(feat)
        # 恢复 flow 的尺寸并缩放
        flow = (
            F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            * scale
        )
        # 恢复 mask 的尺寸
        mask = F.interpolate(
            mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False
        )
        # 返回光流和掩码
        return flow, mask


# 定义 IFNet 类，继承自 nn.Module
class IFNet(nn.Module):
    # 初始化 IFNet 类的构造函数
        def __init__(self):
            # 调用父类的构造函数
            super(IFNet, self).__init__()
            # 创建第一个 IFBlock，输入通道为 7 + 4，参数 c 设置为 90
            self.block0 = IFBlock(7 + 4, c=90)
            # 创建第二个 IFBlock，输入通道为 7 + 4，参数 c 设置为 90
            self.block1 = IFBlock(7 + 4, c=90)
            # 创建第三个 IFBlock，输入通道为 7 + 4，参数 c 设置为 90
            self.block2 = IFBlock(7 + 4, c=90)
            # 创建第四个 IFBlock，输入通道为 10 + 4，参数 c 设置为 90
            self.block_tea = IFBlock(10 + 4, c=90)
            # 上下文网络的实例化（被注释掉）
            # self.contextnet = Contextnet()
            # UNet 的实例化（被注释掉）
            # self.unet = Unet()
    
        # 前向传播函数，处理输入 x 和缩放列表，训练标志为 False
        def forward(self, x, scale_list=[4, 2, 1], training=False):
            # 如果不是训练模式
            if training == False:
                # 获取通道数，假设输入有两个部分
                channel = x.shape[1] // 2
                # 将前半部分赋值给 img0
                img0 = x[:, :channel]
                # 将后半部分赋值给 img1
                img1 = x[:, channel:]
            # 初始化流列表
            flow_list = []
            # 初始化合并列表
            merged = []
            # 初始化掩码列表
            mask_list = []
            # 将 img0 赋值给 warped_img0
            warped_img0 = img0
            # 将 img1 赋值给 warped_img1
            warped_img1 = img1
            # 创建一个与 x 的前四个通道相同的流，初始化为零
            flow = (x[:, :4]).detach() * 0
            # 创建一个与 x 的第一个通道相同的掩码，初始化为零
            mask = (x[:, :1]).detach() * 0
            # 初始化约束损失为零
            loss_cons = 0
            # 创建一个包含 block0、block1 和 block2 的列表
            block = [self.block0, self.block1, self.block2]
            # 循环三次，处理三个块
            for i in range(3):
                # 通过块处理图像和流，获取 f0 和 m0
                f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
                # 通过块处理逆向图像和流，获取 f1 和 m1
                f1, m1 = block[i](
                    torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1),
                    torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                    scale=scale_list[i],
                )
                # 更新流，添加平均值
                flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                # 更新掩码，添加平均值
                mask = mask + (m0 + (-m1)) / 2
                # 将掩码添加到掩码列表
                mask_list.append(mask)
                # 将流添加到流列表
                flow_list.append(flow)
                # 对 img0 进行光流变形，更新 warped_img0
                warped_img0 = warp(img0, flow[:, :2])
                # 对 img1 进行光流变形，更新 warped_img1
                warped_img1 = warp(img1, flow[:, 2:4])
                # 将变形后的图像添加到合并列表
                merged.append((warped_img0, warped_img1))
            """
            # 计算上下文特征 c0（被注释掉）
            c0 = self.contextnet(img0, flow[:, :2])
            # 计算上下文特征 c1（被注释掉）
            c1 = self.contextnet(img1, flow[:, 2:4])
            # 通过 UNet 计算临时结果（被注释掉）
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            # 处理临时结果，得到最终结果 res（被注释掉）
            res = tmp[:, 1:4] * 2 - 1
            """
            # 对每个掩码应用 sigmoid 函数
            for i in range(3):
                mask_list[i] = torch.sigmoid(mask_list[i])
                # 合并 warped_img0 和 warped_img1，应用当前掩码
                merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
                # 进行范围限制（被注释掉）
                # merged[i] = torch.clamp(merged[i] + res, 0, 1)
            # 返回流列表、最后的掩码和合并后的图像
            return flow_list, mask_list[2], merged
```