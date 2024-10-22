# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\laplacian.py`

```py
# 导入 PyTorch 库
import torch
# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的函数性模块
import torch.nn.functional as F

# 设置设备为 GPU，如果可用则使用 CUDA，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入 PyTorch 库（重复导入，可省略）
import torch


# 定义生成高斯核的函数
def gauss_kernel(size=5, channels=3):
    # 创建一个 5x5 的高斯核张量
    kernel = torch.tensor(
        [
            [1.0, 4.0, 6.0, 4.0, 1],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ]
    )
    # 将高斯核的值归一化
    kernel /= 256.0
    # 重复高斯核以适应指定的通道数
    kernel = kernel.repeat(channels, 1, 1, 1)
    # 将高斯核移动到指定的设备
    kernel = kernel.to(device)
    # 返回生成的高斯核
    return kernel


# 定义下采样函数
def downsample(x):
    # 对输入张量进行下采样，步幅为2
    return x[:, :, ::2, ::2]


# 定义上采样函数
def upsample(x):
    # 在输入的最后一个维度上拼接一个与输入形状相同的零张量
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    # 调整张量形状，以便进行上采样
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    # 转置张量的维度
    cc = cc.permute(0, 1, 3, 2)
    # 在转置后的张量的最后一个维度上拼接一个零张量
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2).to(device)], dim=3)
    # 调整张量形状以适应上采样
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    # 转置回原来的维度顺序
    x_up = cc.permute(0, 1, 3, 2)
    # 使用高斯卷积对上采样后的图像进行卷积处理并返回结果
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1]))


# 定义高斯卷积函数
def conv_gauss(img, kernel):
    # 使用反射模式对图像进行填充
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode="reflect")
    # 对填充后的图像应用卷积操作
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    # 返回卷积结果
    return out


# 定义拉普拉斯金字塔函数
def laplacian_pyramid(img, kernel, max_levels=3):
    # 当前图像初始化为输入图像
    current = img
    # 初始化金字塔列表
    pyr = []
    # 进行多级拉普拉斯金字塔计算
    for level in range(max_levels):
        # 对当前图像进行高斯卷积
        filtered = conv_gauss(current, kernel)
        # 对卷积结果进行下采样
        down = downsample(filtered)
        # 对下采样结果进行上采样
        up = upsample(down)
        # 计算当前图像与上采样结果之间的差异
        diff = current - up
        # 将差异加入金字塔列表
        pyr.append(diff)
        # 更新当前图像为下采样结果
        current = down
    # 返回构建的拉普拉斯金字塔
    return pyr


# 定义拉普拉斯损失类
class LapLoss(torch.nn.Module):
    # 初始化方法
    def __init__(self, max_levels=5, channels=3):
        # 调用父类初始化方法
        super(LapLoss, self).__init__()
        # 设置金字塔的最大层数
        self.max_levels = max_levels
        # 生成高斯核
        self.gauss_kernel = gauss_kernel(channels=channels)

    # 前向传播方法
    def forward(self, input, target):
        # 计算输入图像的拉普拉斯金字塔
        pyr_input = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        # 计算目标图像的拉普拉斯金字塔
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        # 计算输入和目标金字塔之间的 L1 损失之和并返回
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
```