# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\pytorch_msssim\__init__.py`

```py
# 导入 PyTorch 和其功能模块，数学库和 NumPy 库
import torch
import torch.nn.functional as F
from math import exp
import numpy as np

# 检查是否有可用的 GPU，选择计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 创建高斯窗口
def gaussian(window_size, sigma):
    # 生成一个一维高斯分布，x 在 [0, window_size) 范围内
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    # 返回归一化的高斯窗口
    return gauss / gauss.sum()


# 创建二维高斯窗口
def create_window(window_size, channel=1):
    # 生成一维高斯窗口并增加维度以适配二维
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # 计算二维窗口，通过矩阵乘法生成
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).to(device)
    # 扩展窗口以适配多个通道
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 创建三维高斯窗口
def create_window_3d(window_size, channel=1):
    # 生成一维高斯窗口并增加维度以适配二维
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # 计算二维窗口
    _2D_window = _1D_window.mm(_1D_window.t())
    # 通过矩阵乘法创建三维窗口
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    # 扩展窗口以适配多个通道和维度
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().to(device)
    return window


# 计算结构相似性指数（SSIM）
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # 判断值范围，常见的范围为255, 1, 或2
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        # 计算动态范围
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    # 获取输入图像的通道数和尺寸
    (_, channel, height, width) = img1.size()
    if window is None:
        # 确定实际窗口大小并创建窗口
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    # 计算加权平均值，使用填充以保持尺寸
    mu1 = F.conv2d(F.pad(img1, (5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=channel)
    mu2 = F.conv2d(F.pad(img2, (5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=channel)

    # 计算均值的平方和均值乘积
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(F.pad(img1 * img1, (5, 5, 5, 5), "replicate"), window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(img2 * img2, (5, 5, 5, 5), "replicate"), window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(F.pad(img1 * img2, (5, 5, 5, 5), "replicate"), window, padding=padd, groups=channel) - mu1_mu2

    # 设置常量 C1 和 C2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # 计算对比度灵敏度
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # 计算对比度灵敏度

    # 计算 SSIM 映射
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    # 根据需要返回平均 SSIM 或逐通道的平均 SSIM
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    # 根据需要返回完整的结果或仅返回 SSIM
    if full:
        return ret, cs
    return ret


# 计算与 MATLAB 版本兼容的 SSIM（未实现）
def ssim_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # 值范围可能不同于 255。其他常见范围是 1（sigmoid）和 2（tanh）。
    if val_range is None:
        # 检查 img1 的最大值是否大于 128，以确定最大值
        if torch.max(img1) > 128:
            max_val = 255  # 设置最大值为 255
        else:
            max_val = 1  # 设置最大值为 1

        # 检查 img1 的最小值是否小于 -0.5，以确定最小值
        if torch.min(img1) < -0.5:
            min_val = -1  # 设置最小值为 -1
        else:
            min_val = 0  # 设置最小值为 0
        L = max_val - min_val  # 计算值范围 L
    else:
        L = val_range  # 如果 val_range 已定义，直接使用它

    padd = 0  # 初始化填充大小为 0
    (_, _, height, width) = img1.size()  # 获取 img1 的尺寸信息
    if window is None:
        # 计算真实窗口大小，限制在 window_size、height 和 width 之内
        real_size = min(window_size, height, width)
        # 创建一个 3D 窗口，并将其移动到 img1 的设备和数据类型
        window = create_window_3d(real_size, channel=1).to(img1.device, dtype=img1.dtype)
        # 因为考虑彩色图像为体积图像，所以通道设置为 1

    # 在 img1 中增加一个维度
    img1 = img1.unsqueeze(1)
    # 在 img2 中增加一个维度
    img2 = img2.unsqueeze(1)

    # 对 img1 进行卷积操作，并添加边界填充
    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1)
    # 对 img2 进行卷积操作，并添加边界填充
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1)

    # 计算 mu1 的平方
    mu1_sq = mu1.pow(2)
    # 计算 mu2 的平方
    mu2_sq = mu2.pow(2)
    # 计算 mu1 和 mu2 的乘积
    mu1_mu2 = mu1 * mu2

    # 计算 img1 的方差
    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padd, groups=1) - mu1_sq
    # 计算 img2 的方差
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padd, groups=1) - mu2_sq
    # 计算 img1 和 img2 的协方差
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padd, groups=1) - mu1_mu2

    # 计算 SSIM 常数 C1
    C1 = (0.01 * L) ** 2
    # 计算 SSIM 常数 C2
    C2 = (0.03 * L) ** 2

    # 计算 v1，用于 SSIM 公式
    v1 = 2.0 * sigma12 + C2
    # 计算 v2，用于 SSIM 公式
    v2 = sigma1_sq + sigma2_sq + C2
    # 计算对比度敏感性
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    # 计算 SSIM 映射
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    # 根据 size_average 决定返回的结果
    if size_average:
        ret = ssim_map.mean()  # 返回 SSIM 的平均值
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)  # 返回 SSIM 的均值按维度计算

    # 如果 full 为 True，返回 SSIM 和对比度敏感性
    if full:
        return ret, cs
    return ret  # 返回 SSIM
# 计算多尺度结构相似性指数（MSSSIM）函数，比较两张图片的相似度
def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    # 获取第一张图片的设备信息
    device = img1.device
    # 设置权重，用于不同尺度的相似性计算
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    # 权重的数量，表示层级数
    levels = weights.size()[0]
    # 初始化 MSSSIM 和 MCS 的列表
    mssim = []
    mcs = []
    # 对于每一个层级，计算相似性和对比度
    for _ in range(levels):
        # 计算结构相似性（SSIM）和对比度（CS）
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        # 将结果添加到列表中
        mssim.append(sim)
        mcs.append(cs)

        # 对输入图像进行平均池化，以减小图像尺寸
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    # 将结果列表转换为张量
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # 如果需要，则对结果进行归一化处理
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    # 根据权重计算 MCS 和 MSSIM 的幂
    pow1 = mcs**weights
    pow2 = mssim**weights
    # 从 Matlab 实现中引用的最终输出计算
    output = torch.prod(pow1[:-1] * pow2[-1])
    # 返回最终的相似性结果
    return output


# 用于重用窗口的 SSIM 类
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        # 初始化 SSIM 模块并设置参数
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # 假设使用三通道图像
        self.channel = 3
        # 创建用于计算 SSIM 的窗口
        self.window = create_window(window_size, channel=self.channel)

    def forward(self, img1, img2):
        # 获取第一张图像的尺寸信息
        (_, channel, _, _) = img1.size()

        # 如果通道数和数据类型匹配，则使用现有窗口
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            # 否则重新创建窗口并更新通道数
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        # 计算 SSIM 值
        _ssim = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        # 计算 DSSIM 值
        dssim = (1 - _ssim) / 2
        # 返回 DSSIM 值
        return dssim


# 用于多尺度结构相似性计算的 MSSSIM 类
class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        # 初始化 MSSSIM 模块并设置参数
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # 调用 msssim 函数计算相似性
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
```