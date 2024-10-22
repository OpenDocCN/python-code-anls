# CogVideo & CogVideoX 微调代码源码解析（三）



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

# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\loss.py`

```py
# 导入 PyTorch 库及其他必要库
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 设置设备为 GPU（如果可用），否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义一个名为 EPE 的神经网络模块
class EPE(nn.Module):
    # 初始化方法
    def __init__(self):
        super(EPE, self).__init__()

    # 前向传播方法，计算损失图
    def forward(self, flow, gt, loss_mask):
        # 计算流和真实值的平方差
        loss_map = (flow - gt.detach()) ** 2
        # 对损失图进行归一化处理
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        # 返回加权后的损失图
        return loss_map * loss_mask


# 定义一个名为 Ternary 的神经网络模块
class Ternary(nn.Module):
    # 初始化方法
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7  # 定义补丁大小
        out_channels = patch_size * patch_size  # 输出通道数为补丁大小的平方
        # 创建一个单位矩阵并重塑为补丁形状
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        # 转置以适应卷积操作
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        # 将其转换为张量并移动到适当的设备
        self.w = torch.tensor(self.w).float().to(device)

    # 变换图像的方法
    def transform(self, img):
        # 使用卷积操作提取补丁
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        # 计算变换结果
        transf = patches - img
        # 进行归一化处理
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    # RGB 转灰度的方法
    def rgb2gray(self, rgb):
        # 分离 RGB 通道并计算灰度值
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    # 计算汉明距离的方法
    def hamming(self, t1, t2):
        # 计算平方差
        dist = (t1 - t2) ** 2
        # 对距离进行归一化
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    # 生成有效掩码的方法
    def valid_mask(self, t, padding):
        # 获取输入的尺寸
        n, _, h, w = t.size()
        # 创建内掩码
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        # 填充掩码以适应边界
        mask = F.pad(inner, [padding] * 4)
        return mask

    # 前向传播方法，计算损失
    def forward(self, img0, img1):
        # 将图像转换为灰度并进行变换
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        # 返回汉明距离加有效掩码
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


# 定义一个名为 SOBEL 的神经网络模块
class SOBEL(nn.Module):
    # 初始化方法
    def __init__(self):
        super(SOBEL, self).__init__()
        # 定义 Sobel X 卷积核
        self.kernelX = torch.tensor(
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1],
            ]
        ).float()
        # Sobel Y 卷积核为 X 卷积核的转置
        self.kernelY = self.kernelX.clone().T
        # 扩展卷积核以适应输入通道
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(device)

    # 前向传播方法，计算损失
    def forward(self, pred, gt):
        # 获取输入的维度
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        # 将预测和真实值堆叠以便于卷积
        img_stack = torch.cat([pred.reshape(N * C, 1, H, W), gt.reshape(N * C, 1, H, W)], 0)
        # 对堆叠图像应用 Sobel X 卷积
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        # 对堆叠图像应用 Sobel Y 卷积
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        # 提取预测和真实值的 Sobel 特征
        pred_X, gt_X = sobel_stack_x[: N * C], sobel_stack_x[N * C :]
        pred_Y, gt_Y = sobel_stack_y[: N * C], sobel_stack_y[N * C :]

        # 计算 L1 损失
        L1X, L1Y = torch.abs(pred_X - gt_X), torch.abs(pred_Y - gt_Y)
        loss = L1X + L1Y  # 总损失为两个方向的损失之和
        return loss


# 定义一个 MeanShift 模块，继承自卷积层
class MeanShift(nn.Conv2d):
    # 初始化 MeanShift 类，设置数据的均值、标准差和范围
        def __init__(self, data_mean, data_std, data_range=1, norm=True):
            # 获取数据均值的数量
            c = len(data_mean)
            # 调用父类构造函数，初始化 kernel_size 为 1
            super(MeanShift, self).__init__(c, c, kernel_size=1)
            # 将标准差转换为张量
            std = torch.Tensor(data_std)
            # 初始化权重为单位矩阵，形状为 (c, c, 1, 1)
            self.weight.data = torch.eye(c).view(c, c, 1, 1)
            # 如果需要标准化
            if norm:
                # 按标准差对权重进行归一化
                self.weight.data.div_(std.view(c, 1, 1, 1))
                # 计算偏置为负的均值乘以范围
                self.bias.data = -1 * data_range * torch.Tensor(data_mean)
                # 对偏置进行标准差归一化
                self.bias.data.div_(std)
            else:
                # 如果不标准化，直接按标准差调整权重
                self.weight.data.mul_(std.view(c, 1, 1, 1))
                # 计算偏置为均值乘以范围
                self.bias.data = data_range * torch.Tensor(data_mean)
            # 不需要计算梯度
            self.requires_grad = False
# 定义一个名为 VGGPerceptualLoss 的类，继承自 PyTorch 的 nn.Module
class VGGPerceptualLoss(torch.nn.Module):
    # 初始化方法，接受一个可选的 rank 参数，默认为 0
    def __init__(self, rank=0):
        # 调用父类的初始化方法
        super(VGGPerceptualLoss, self).__init__()
        # 创建一个空列表，用于存储网络块
        blocks = []
        # 设定预训练模型的标志
        pretrained = True
        # 加载 VGG19 模型的特征提取部分，并存储到类属性中
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        # 初始化 MeanShift 对象，用于图像标准化，并将其移至 GPU
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        # 将 VGG 模型的所有参数设置为不需要梯度更新
        for param in self.parameters():
            param.requires_grad = False

    # 定义前向传播方法，接受两个输入图像 X 和 Y 以及可选的 indices
    def forward(self, X, Y, indices=None):
        # 对输入图像 X 进行标准化
        X = self.normalize(X)
        # 对输入图像 Y 进行标准化
        Y = self.normalize(Y)
        # 指定要提取特征的层的索引
        indices = [2, 7, 12, 21, 30]
        # 定义对应于各层的权重
        weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        # 初始化权重索引 k 和损失值 loss
        k = 0
        loss = 0
        # 遍历 VGG 特征提取层的索引
        for i in range(indices[-1]):
            # 通过 VGG 模型提取 X 的特征
            X = self.vgg_pretrained_features[i](X)
            # 通过 VGG 模型提取 Y 的特征
            Y = self.vgg_pretrained_features[i](Y)
            # 如果当前层索引在指定的 indices 中
            if (i + 1) in indices:
                # 计算当前层的感知损失，并累加到总损失中
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                # 更新权重索引 k
                k += 1
        # 返回计算得到的总损失
        return loss


# 主程序入口
if __name__ == "__main__":
    # 创建一个全零的张量，形状为 (3, 3, 256, 256)，并移至指定设备
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    # 创建一个随机正态分布的张量，形状与 img0 相同，并移至指定设备
    img1 = torch.tensor(np.random.normal(0, 1, (3, 3, 256, 256))).float().to(device)
    # 实例化 Ternary 类的对象
    ternary_loss = Ternary()
    # 输出 ternary_loss 计算 img0 和 img1 的结果的形状
    print(ternary_loss(img0, img1).shape)
```

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

# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\RIFE.py`

```py
# 从 PyTorch 的优化器导入 AdamW
from torch.optim import AdamW
# 从 PyTorch 导入分布式数据并行支持
from torch.nn.parallel import DistributedDataParallel as DDP
# 导入 IFNet 模型相关模块
from .IFNet import *
from .IFNet_m import *
from .loss import *
from .laplacian import *
from .refine import *

# 根据 CUDA 可用性设置设备为 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义模型类
class Model:
    # 初始化模型，接收本地进程标识和是否使用任意流网络的标志
    def __init__(self, local_rank=-1, arbitrary=False):
        # 如果使用任意流网络，则初始化为 IFNet_m
        if arbitrary == True:
            self.flownet = IFNet_m()
        # 否则初始化为 IFNet
        else:
            self.flownet = IFNet()
        # 将模型移动到指定设备
        self.device()
        # 使用 AdamW 优化器，设置学习率和权重衰减
        self.optimG = AdamW(
            self.flownet.parameters(), lr=1e-6, weight_decay=1e-3
        )  # 使用较大的权重衰减可能避免 NaN 损失
        # 初始化 EPE 损失计算
        self.epe = EPE()
        # 初始化拉普拉斯损失计算
        self.lap = LapLoss()
        # 初始化 SOBEL 操作
        self.sobel = SOBEL()
        # 如果指定本地进程标识，使用分布式数据并行包装流网络
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    # 定义训练模式
    def train(self):
        # 将流网络设置为训练模式
        self.flownet.train()

    # 定义评估模式
    def eval(self):
        # 将流网络设置为评估模式
        self.flownet.eval()

    # 将模型移动到指定设备
    def device(self):
        self.flownet.to(device)

    # 加载模型参数
    def load_model(self, path, rank=0):
        # 定义转换函数，去除参数名中的 "module."
        def convert(param):
            return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

        # 如果当前进程是主进程，加载流网络的状态字典
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load("{}/flownet.pkl".format(path))))

    # 保存模型参数
    def save_model(self, path, rank=0):
        # 如果当前进程是主进程，保存流网络的状态字典
        if rank == 0:
            torch.save(self.flownet.state_dict(), "{}/flownet.pkl".format(path))

    # 推断函数
    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        # 根据缩放比例调整缩放列表
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        # 将输入图像在通道维度上拼接
        imgs = torch.cat((img0, img1), 1)
        # 调用流网络进行推断，获取流、掩膜、合成图等
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(
            imgs, scale_list, timestep=timestep
        )
        # 如果不使用测试时间增强，返回合成图的第三个版本
        if TTA == False:
            return merged[2]
        else:
            # 使用翻转图像进行推断，获取第二个合成图
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(
                imgs.flip(2).flip(3), scale_list, timestep=timestep
            )
            # 返回两个合成图的平均值
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    # 更新模型参数，进行图像处理和损失计算
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        # 设置优化器的学习率
        for param_group in self.optimG.param_groups:
            param_group["lr"] = learning_rate
        # 将输入图像分为两部分，前3个通道和后面的通道
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        # 根据训练状态设置模型为训练模式或评估模式
        if training:
            self.train()  # 设置为训练模式
        else:
            self.eval()   # 设置为评估模式
        # 通过流网模型计算流和相关输出
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(
            torch.cat((imgs, gt), 1), scale=[4, 2, 1]  # 将图像和真实标签拼接并传入网络
        )
        # 计算合并图像与真实标签的L1损失
        loss_l1 = (self.lap(merged[2], gt)).mean()
        # 计算教师网络合并图像与真实标签的L1损失
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        # 如果是训练状态，进行反向传播和优化步骤
        if training:
            self.optimG.zero_grad()  # 清空优化器的梯度
            # 计算总损失，结合各个损失项
            loss_G = (
                loss_l1 + loss_tea + loss_distill * 0.01  # 在训练 RIFEm 时，loss_distill 的权重应该设置为 0.005 或 0.002
            )
            # 进行反向传播以计算梯度
            loss_G.backward()
            # 更新优化器中的参数
            self.optimG.step()
        else:
            # 在评估模式下获取教师网络的流
            flow_teacher = flow[2]
        # 返回合并后的图像和损失的详细信息
        return merged[2], {
            "merged_tea": merged_teacher,  # 教师网络的合并结果
            "mask": mask,                   # 流的掩码
            "mask_tea": mask,               # 教师网络的掩码
            "flow": flow[2][:, :2],         # 当前流的前两个通道
            "flow_tea": flow_teacher,       # 教师网络的流
            "loss_l1": loss_l1,             # 当前L1损失
            "loss_tea": loss_tea,           # 教师网络的L1损失
            "loss_distill": loss_distill,   # 蒸馏损失
        }
```

# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\RIFE_HDv3.py`

```py
# 导入 PyTorch 及其子模块和其他必要的库
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from .warplayer import warp  # 导入自定义模块 warp
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块
from .IFNet_HDv3 import *  # 导入 IFNet_HDv3 模块中的所有内容
import torch.nn.functional as F  # 导入功能性激活函数
from .loss import *  # 导入自定义损失函数模块

# 检查是否有可用的 GPU，并选择相应的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    # 初始化模型，设置本地进程的排名（默认值为 -1）
    def __init__(self, local_rank=-1):
        # 实例化 IFNet 网络
        self.flownet = IFNet()
        # 将模型移动到指定的设备
        self.device()
        # 使用 AdamW 优化器，设置学习率和权重衰减
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        # 实例化 EPE 损失对象
        self.epe = EPE()
        # self.vgg = VGGPerceptualLoss().to(device)  # （注释掉的）实例化 VGG 感知损失对象并移动到设备
        # 实例化 SOBEL 边缘检测对象
        self.sobel = SOBEL()
        # 如果 local_rank 不为 -1，则使用分布式数据并行
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    # 设置模型为训练模式
    def train(self):
        self.flownet.train()

    # 设置模型为评估模式
    def eval(self):
        self.flownet.eval()

    # 将模型移动到指定的设备
    def device(self):
        self.flownet.to(device)

    # 从指定路径加载模型参数
    def load_model(self, path, rank=0):
        # 内部函数用于转换参数名称
        def convert(param):
            # 如果 rank 为 -1，移除参数名称中的 "module." 前缀
            if rank == -1:
                return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}
            else:
                return param

        # 如果 rank 小于等于 0，则加载模型
        if rank <= 0:
            # 如果有可用的 GPU，加载到 GPU
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load("{}/flownet.pkl".format(path))))
            # 否则加载到 CPU
            else:
                self.flownet.load_state_dict(convert(torch.load("{}/flownet.pkl".format(path), map_location="cpu")))

    # 保存模型参数到指定路径
    def save_model(self, path, rank=0):
        # 如果 rank 为 0，保存模型状态字典
        if rank == 0:
            torch.save(self.flownet.state_dict(), "{}/flownet.pkl".format(path))

    # 进行推理，合并两幅图像并返回结果
    def inference(self, img0, img1, scale=1.0):
        # 将 img0 和 img1 在通道维度上进行拼接
        imgs = torch.cat((img0, img1), 1)
        # 根据输入缩放比例生成缩放列表
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        # 调用 flownet 进行推理，获取光流、掩膜和合并结果
        flow, mask, merged = self.flownet(imgs, scale_list)
        # 返回合并结果的第三个输出
        return merged[2]

    # 更新模型参数，计算损失并进行优化
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        # 更新优化器中的学习率
        for param_group in self.optimG.param_groups:
            param_group["lr"] = learning_rate
        # 从输入图像中分离出 img0 和 img1
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        # 根据训练状态设置模型模式
        if training:
            self.train()
        else:
            self.eval()
        # 定义缩放比例
        scale = [4, 2, 1]
        # 调用 flownet 进行前向计算
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        # 计算 L1 损失
        loss_l1 = (merged[2] - gt).abs().mean()
        # 计算平滑损失
        loss_smooth = self.sobel(flow[2], flow[2] * 0).mean()
        # loss_vgg = self.vgg(merged[2], gt)  # （注释掉的）计算 VGG 感知损失
        # 如果处于训练状态，进行反向传播和优化步骤
        if training:
            self.optimG.zero_grad()  # 清空梯度
            loss_G = loss_cons + loss_smooth * 0.1  # 计算总损失
            loss_G.backward()  # 反向传播
            self.optimG.step()  # 更新参数
        else:
            flow_teacher = flow[2]  # 获取教师模型的光流
        # 返回合并结果和各项损失
        return merged[2], {
            "mask": mask,
            "flow": flow[2][:, :2],
            "loss_l1": loss_l1,
            "loss_cons": loss_cons,
            "loss_smooth": loss_smooth,
        }
```

# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\warplayer.py`

```py
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn

# 设置计算设备为 CUDA（如果可用），否则为 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化一个空字典，用于存储反向扭曲的网格
backwarp_tenGrid = {}

# 定义一个函数，用于根据输入和光流进行扭曲
def warp(tenInput, tenFlow):
    # 将光流的设备和尺寸转换为字符串，作为字典的键
    k = (str(tenFlow.device), str(tenFlow.size()))
    # 如果该键不在字典中
    if k not in backwarp_tenGrid:
        # 创建水平网格，从 -1 到 1 线性分布，尺寸与光流的宽度相同
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)  # 生成水平坐标
            .view(1, 1, 1, tenFlow.shape[3])  # 重塑为适合的形状
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)  # 扩展到输入的批次大小
        )
        # 创建垂直网格，从 -1 到 1 线性分布，尺寸与光流的高度相同
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)  # 生成垂直坐标
            .view(1, 1, tenFlow.shape[2], 1)  # 重塑为适合的形状
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])  # 扩展到输入的批次大小
        )
        # 将水平和垂直网格合并并存储到字典中
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    # 将光流进行归一化处理，以适应输入尺寸
    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),  # 归一化宽度
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),  # 归一化高度
        ],
        1,  # 在通道维度上连接
    )

    # 将网格与光流相加，并调整维度顺序以适应 grid_sample 函数
    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    # 使用 grid_sample 函数进行图像扭曲，返回扭曲后的图像
    return torch.nn.functional.grid_sample(
        input=tenInput, grid=g, mode="bilinear", padding_mode="border", align_corners=True
    )
```

# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\__init__.py`

```py
请提供需要注释的代码，我会为每个语句添加解释。
```

# `.\cogvideo-finetune\inference\gradio_composite_demo\rife_model.py`

```py
# 导入 PyTorch 库，用于深度学习操作
import torch
# 从 diffusers 库导入 VaeImageProcessor，用于处理图像
from diffusers.image_processor import VaeImageProcessor
# 导入 PyTorch 的函数式 API，主要用于张量操作
from torch.nn import functional as F
# 导入 OpenCV 库，用于图像处理
import cv2
# 导入自定义的 utils 模块，可能包含一些实用函数
import utils
# 从 rife.pytorch_msssim 导入 ssim_matlab，用于计算结构相似性
from rife.pytorch_msssim import ssim_matlab
# 导入 NumPy 库，用于数组操作
import numpy as np
# 导入 logging 模块，用于记录日志
import logging
# 从 skvideo.io 导入用于视频输入输出的库
import skvideo.io
# 从 rife.RIFE_HDv3 导入 Model 类，可能用于帧插值模型
from rife.RIFE_HDv3 import Model
# 从 huggingface_hub 导入下载模型和快照的功能
from huggingface_hub import hf_hub_download, snapshot_download
# 创建一个日志记录器，使用当前模块的名称
logger = logging.getLogger(__name__)

# 检查是否可以使用 GPU，如果可以则设为 'cuda'，否则设为 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义图像填充函数，接受图像和缩放比例作为参数
def pad_image(img, scale):
    # 解构图像形状，获取通道数、高度和宽度
    _, _, h, w = img.shape
    # 计算填充大小，确保是 32 的倍数
    tmp = max(32, int(32 / scale))
    # 计算填充后的高度和宽度
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    # 计算所需的填充边界
    padding = (0,  pw - w, 0, ph - h)
    # 返回填充后的图像和填充参数
    return F.pad(img, padding), padding

# 定义推理函数，接受模型、两帧图像、放大比例和分割数量作为参数
def make_inference(model, I0, I1, upscale_amount, n):
    # 调用模型进行推理，生成中间帧
    middle = model.inference(I0, I1, upscale_amount)
    # 如果分割数量为 1，返回中间帧
    if n == 1:
        return [middle]
    # 递归调用，生成前半部分插值帧
    first_half = make_inference(model, I0, middle, upscale_amount, n=n // 2)
    # 递归调用，生成后半部分插值帧
    second_half = make_inference(model, middle, I1, upscale_amount, n=n // 2)
    # 如果分割数量为奇数，合并结果
    if n % 2:
        return [*first_half, middle, *second_half]
    # 否则直接合并前后两部分
    else:
        return [*first_half, *second_half]

# 使用 PyTorch 的推理模式进行插值操作
@torch.inference_mode()
def ssim_interpolation_rife(model, samples, exp=1, upscale_amount=1, output_device="cpu"):
    # 打印样本数据类型
    print(f"samples dtype:{samples.dtype}")
    # 打印样本形状
    print(f"samples shape:{samples.shape}")
    # 初始化输出列表
    output = []
    # 创建进度条，用于显示推理进度
    pbar = utils.ProgressBar(samples.shape[0], desc="RIFE inference")
    # 样本形状为 [帧数, 通道数, 高度, 宽度]
    # 遍历样本的每一帧
    for b in range(samples.shape[0]):
        # 选取当前帧并增加维度
        frame = samples[b : b + 1]
        # 获取当前帧的高度和宽度
        _, _, h, w = frame.shape
        
        # 将当前帧赋值给 I0
        I0 = samples[b : b + 1]
        # 如果有下一帧，则赋值给 I1，否则使用最后一帧
        I1 = samples[b + 1 : b + 2] if b + 2 < samples.shape[0] else samples[-1:]
         
        # 对 I0 进行填充并返回填充后的图像和填充信息
        I0, padding = pad_image(I0, upscale_amount)
        # 将 I0 转换为浮点数类型
        I0 = I0.to(torch.float)
        # 对 I1 进行填充，第二个返回值不需要
        I1, _ = pad_image(I1, upscale_amount)
        # 将 I1 转换为浮点数类型
        I1 = I1.to(torch.float)
         
        # 将 I0 和 I1 进行双线性插值，缩放至 (32, 32)
        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)

        # 计算 I0_small 和 I1_small 之间的 SSIM 值
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        # 如果 SSIM 值大于 0.996，进行以下操作
        if ssim > 0.996:
            # 将当前帧重新赋值给 I1
            I1 = samples[b : b + 1]
            # print(f'upscale_amount:{upscale_amount}')  # 输出放大倍数（注释掉）
            # print(f'ssim:{upscale_amount}')  # 输出 SSIM 值（注释掉）
            # print(f'I0 shape:{I0.shape}')  # 输出 I0 的形状（注释掉）
            # print(f'I1 shape:{I1.shape}')  # 输出 I1 的形状（注释掉）
            # 对 I1 进行填充并返回填充信息
            I1, padding = pad_image(I1, upscale_amount)
            # print(f'I0 shape:{I0.shape}')  # 输出 I0 的形状（注释掉）
            # print(f'I1 shape:{I1.shape}')  # 输出 I1 的形状（注释掉）
            # 进行推理，使用 I0 和 I1 以及放大倍数
            I1 = make_inference(model, I0, I1, upscale_amount, 1)
            
            # print(f'I0 shape:{I0.shape}')  # 输出 I0 的形状（注释掉）
            # print(f'I1[0] shape:{I1[0].shape}')  # 输出 I1[0] 的形状（注释掉）
            # 取出推理结果的第一张图像
            I1 = I1[0]
            
            # print(f'I1[0] unpadded shape:{I1.shape}')  # 输出 I1 的去填充形状（注释掉） 
            # 将 I1 进行双线性插值，缩放至 (32, 32)
            I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
            # 重新计算 SSIM 值
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            # 根据填充信息确定如何裁剪 frame
            if padding[3] > 0 and padding[1] > 0:
                frame = I1[:, :, : -padding[3],:-padding[1]]
            elif padding[3] > 0:
                frame = I1[:, :, : -padding[3],:]
            elif padding[1] > 0:
                frame = I1[:, :, :,:-padding[1]]
            else:
                frame = I1

        # 初始化临时输出列表
        tmp_output = []
        # 如果 SSIM 值小于 0.2，进行以下操作
        if ssim < 0.2:
            # 根据指数生成多张 I0
            for i in range((2**exp) - 1):
                tmp_output.append(I0)

        else:
            # 如果指数不为零，则进行推理并生成输出
            tmp_output = make_inference(model, I0, I1, upscale_amount, 2**exp - 1) if exp else []

        # 对 frame 进行填充
        frame, _ = pad_image(frame, upscale_amount)
        # print(f'frame shape:{frame.shape}')  # 输出 frame 的形状（注释掉）

        # 将 frame 进行插值，缩放至原始的高度和宽度
        frame = F.interpolate(frame, size=(h, w))
        # 将处理后的 frame 加入输出列表
        output.append(frame.to(output_device))
        # 遍历临时输出并处理
        for i, tmp_frame in enumerate(tmp_output): 
            # tmp_frame, _ = pad_image(tmp_frame, upscale_amount)  # 对 tmp_frame 进行填充（注释掉）
            # 将 tmp_frame 进行插值，缩放至原始的高度和宽度
            tmp_frame = F.interpolate(tmp_frame, size=(h, w))
            # 将处理后的 tmp_frame 加入输出列表
            output.append(tmp_frame.to(output_device))
        # 更新进度条
        pbar.update(1)
    # 返回最终输出
    return output
# 加载 RIFE 模型并返回模型实例
def load_rife_model(model_path):
    # 创建模型实例
    model = Model()
    # 从指定路径加载模型，第二个参数为 -1（表示不使用特定的版本）
    model.load_model(model_path, -1)
    # 将模型设置为评估模式
    model.eval()
    # 返回加载的模型
    return model


# 创建一个生成器，逐帧输出视频帧，类似于 cv2.VideoCapture
def frame_generator(video_capture):
    # 无限循环，直到读取完所有帧
    while True:
        # 从视频捕捉对象中读取一帧，ret 为读取成功标志，frame 为当前帧
        ret, frame = video_capture.read()
        # 如果没有读取到帧，退出循环
        if not ret:
            break
        # 生成当前帧
        yield frame
    # 释放视频捕捉对象
    video_capture.release()


def rife_inference_with_path(model, video_path):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    # 获取视频的帧率（每秒多少帧）
    fps = video_capture.get(cv2.CAP_PROP_FPS)  
    # 获取视频的总帧数
    tot_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  
    # 存储处理后帧的列表
    pt_frame_data = []
    # 使用 skvideo.io 逐帧读取视频
    pt_frame = skvideo.io.vreader(video_path)
    # 循环读取视频帧
    while video_capture.isOpened():
        # 读取一帧
        ret, frame = video_capture.read()

        # 如果没有读取到帧，退出循环
        if not ret:
            break

        # 将 BGR 格式的帧转换为 RGB 格式
        frame_rgb = frame[..., ::-1]
        # 创建帧的副本
        frame_rgb = frame_rgb.copy()
        # 将 RGB 帧转换为 tensor，并归一化到 [0, 1] 之间
        tensor = torch.from_numpy(frame_rgb).float().to("cpu", non_blocking=True).float() / 255.0
        # 将处理后的 tensor 按 [c, h, w] 格式添加到列表
        pt_frame_data.append(
            tensor.permute(2, 0, 1)
        )  # to [c, h, w,]

    # 将帧数据堆叠为一个 tensor
    pt_frame = torch.from_numpy(np.stack(pt_frame_data))
    # 将 tensor 移动到指定设备
    pt_frame = pt_frame.to(device)
    # 创建进度条，显示处理进度
    pbar = utils.ProgressBar(tot_frame, desc="RIFE inference")
    # 使用 RIFE 模型进行帧插值
    frames = ssim_interpolation_rife(model, pt_frame)
    # 堆叠生成的帧为一个 tensor
    pt_image = torch.stack([frames[i].squeeze(0) for i in range(len(frames))])
    # 将处理后的 tensor 转换为 NumPy 数组
    image_np = VaeImageProcessor.pt_to_numpy(pt_image)  # (to [49, 512, 480, 3])
    # 将 NumPy 数组转换为 PIL 图像
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
    # 保存处理后的视频，并设置帧率
    video_path = utils.save_video(image_pil, fps=16)
    # 更新进度条
    if pbar:
        pbar.update(1)
    # 返回保存的视频路径
    return video_path


def rife_inference_with_latents(model, latents):
    # 存储 RIFE 处理结果的列表
    rife_results = []
    # 将潜在变量移动到指定设备
    latents = latents.to(device)
    # 遍历每个潜在变量
    for i in range(latents.size(0)):
        # 取出当前的潜在变量
        latent = latents[i]

        # 使用 RIFE 模型进行帧插值
        frames = ssim_interpolation_rife(model, latent)
        # 堆叠生成的帧为一个 tensor
        pt_image = torch.stack([frames[i].squeeze(0) for i in range(len(frames))])  # (to [f, c, w, h])
        # 将处理结果添加到列表中
        rife_results.append(pt_image)

    # 返回所有处理结果的堆叠 tensor
    return torch.stack(rife_results)


# if __name__ == "__main__":
#     # 下载 RIFE 模型快照到指定目录
#     snapshot_download(repo_id="AlexWortega/RIFE", local_dir="model_rife")
#     # 加载 RIFE 模型
#     model = load_rife_model("model_rife")
 
#     # 使用指定视频路径进行 RIFE 推理
#     video_path = rife_inference_with_path(model, "/mnt/ceph/develop/jiawei/CogVideo/output/20241003_130720.mp4")
#     # 打印保存的视频路径
#     print(video_path)
```

# `.\cogvideo-finetune\inference\gradio_composite_demo\utils.py`

```py
# 导入数学库
import math
# 从 typing 模块导入 Union 和 List 类型注解
from typing import Union, List

# 导入 PyTorch 库
import torch
# 导入操作系统相关功能
import os
# 导入日期和时间处理功能
from datetime import datetime
# 导入 NumPy 库
import numpy as np
# 导入 itertools 库
import itertools
# 导入图像处理库 PIL
import PIL.Image
# 导入 safetensors 库用于处理张量
import safetensors.torch
# 导入进度条显示库 tqdm
import tqdm
# 导入日志记录库
import logging
# 从 diffusers.utils 导入视频导出功能
from diffusers.utils import export_to_video
# 导入模型加载器
from spandrel import ModelLoader

# 创建一个记录器，命名为当前文件名
logger = logging.getLogger(__file__)


# 定义加载 PyTorch 文件的函数
def load_torch_file(ckpt, device=None, dtype=torch.float16):
    # 如果未指定设备，则默认使用 CPU
    if device is None:
        device = torch.device("cpu")
    # 检查文件扩展名，判断是否为 safetensors 格式
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        # 加载 safetensors 文件
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        # 检查当前 PyTorch 版本是否支持 weights_only 参数
        if not "weights_only" in torch.load.__code__.co_varnames:
            logger.warning(
                "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
            )

        # 加载普通 PyTorch 文件
        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        # 如果包含 global_step，记录调试信息
        if "global_step" in pl_sd:
            logger.debug(f"Global Step: {pl_sd['global_step']}")
        # 根据不同的键获取模型状态字典
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        elif "params_ema" in pl_sd:
            sd = pl_sd["params_ema"]
        else:
            sd = pl_sd

    # 将加载的张量转换为指定数据类型
    sd = {k: v.to(dtype) for k, v in sd.items()}
    # 返回状态字典
    return sd


# 定义替换状态字典前缀的函数
def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    # 如果需要过滤键，则初始化输出字典
    if filter_keys:
        out = {}
    else:
        out = state_dict
    # 遍历所有要替换的前缀
    for rp in replace_prefix:
        # 找到以指定前缀开头的所有键，并生成新键
        replace = list(
            map(
                lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])),
                filter(lambda a: a.startswith(rp), state_dict.keys()),
            )
        )
        # 遍历需要替换的键值对
        for x in replace:
            # 从状态字典中移除旧键，添加新键
            w = state_dict.pop(x[0])
            out[x[1]] = w
    # 返回更新后的字典
    return out


# 定义计算模块大小的函数
def module_size(module):
    module_mem = 0
    # 获取模块的状态字典
    sd = module.state_dict()
    # 遍历状态字典中的每个键
    for k in sd:
        t = sd[k]
        # 计算模块内所有张量的元素总数乘以元素大小，累加到模块内存大小
        module_mem += t.nelement() * t.element_size()
    # 返回模块的内存大小
    return module_mem


# 定义计算平铺缩放步骤的函数
def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    # 计算平铺所需的步骤数
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


# 使用无梯度模式定义平铺缩放多维函数
@torch.inference_mode()
def tiled_scale_multidim(
    samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", pbar=None
):
    # 获取平铺的维度数量
    dims = len(tile)
    # 打印样本的数据类型
    print(f"samples dtype:{samples.dtype}")
    # 初始化输出张量，形状为样本数量和调整后的通道数
    output = torch.empty(
        [samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:])),
        device=output_device,
    )
    # 遍历样本的每个元素
        for b in range(samples.shape[0]):
            # 获取当前样本的切片
            s = samples[b : b + 1]
            # 初始化输出张量，大小根据上采样比例计算
            out = torch.zeros(
                [s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])),
                device=output_device,
            )
            # 初始化输出分母张量，用于后续归一化
            out_div = torch.zeros(
                [s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])),
                device=output_device,
            )
    
            # 生成所有可能的切片位置
            for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(s.shape[2:], tile))):
                # 设置输入为当前样本
                s_in = s
                # 用于存储上采样的位置
                upscaled = []
    
                # 遍历每个维度
                for d in range(dims):
                    # 计算当前切片的位置，确保不越界
                    pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                    # 确定当前切片的长度
                    l = min(tile[d], s.shape[d + 2] - pos)
                    # 从样本中提取相应的切片
                    s_in = s_in.narrow(d + 2, pos, l)
                    # 记录上采样位置
                    upscaled.append(round(pos * upscale_amount))
    
                # 对输入进行处理，得到上采样的结果
                ps = function(s_in).to(output_device)
                # 创建与 ps 相同形状的全一掩码
                mask = torch.ones_like(ps)
                # 计算羽化的大小
                feather = round(overlap * upscale_amount)
                # 为每个维度应用羽化处理
                for t in range(feather):
                    for d in range(2, dims + 2):
                        # 处理掩码的前端羽化
                        m = mask.narrow(d, t, 1)
                        m *= (1.0 / feather) * (t + 1)
                        # 处理掩码的后端羽化
                        m = mask.narrow(d, mask.shape[d] - 1 - t, 1)
                        m *= (1.0 / feather) * (t + 1)
    
                # 定义输出张量
                o = out
                o_d = out_div
                # 将上采样结果添加到输出张量中
                for d in range(dims):
                    o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                    o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])
    
                # 更新输出张量和分母张量
                o += ps * mask
                o_d += mask
    
                # 如果有进度条，更新进度
                if pbar is not None:
                    pbar.update(1)
    
            # 将结果存回输出张量，进行归一化处理
            output[b : b + 1] = out / out_div
        # 返回最终输出
        return output
# 定义一个函数，用于对样本进行分块缩放
def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    pbar=None,
):
    # 调用 tiled_scale_multidim 函数，传递参数以执行缩放
    return tiled_scale_multidim(
        samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device, pbar
    )


# 定义一个函数，从检查点加载上采样模型
def load_sd_upscale(ckpt, inf_device):
    # 从指定设备加载模型权重文件
    sd = load_torch_file(ckpt, device=inf_device)
    # 检查权重字典中是否存在特定键
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        # 替换字典中键的前缀
        sd = state_dict_prefix_replace(sd, {"module.": ""})
    # 加载模型并将其转换为半精度
    out = ModelLoader().load_from_state_dict(sd).half()
    # 返回加载的模型
    return out


# 定义一个函数，用于对给定的张量进行上采样
def upscale(upscale_model, tensor: torch.Tensor, inf_device, output_device="cpu") -> torch.Tensor:
    # 计算上采样模型所需的内存
    memory_required = module_size(upscale_model.model)
    memory_required += (
        (512 * 512 * 3) * tensor.element_size() * max(upscale_model.scale, 1.0) * 384.0
    )  # 384.0 是模型内存占用的估算值，TODO: 需要更准确
    memory_required += tensor.nelement() * tensor.element_size()
    # 打印所需内存的大小
    print(f"UPScaleMemory required: {memory_required / 1024 / 1024 / 1024} GB")

    # 将上采样模型移至指定的设备
    upscale_model.to(inf_device)
    # 定义分块的大小和重叠量
    tile = 512
    overlap = 32

    # 计算总的处理步骤
    steps = tensor.shape[0] * get_tiled_scale_steps(
        tensor.shape[3], tensor.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
    )

    # 初始化进度条
    pbar = ProgressBar(steps, desc="Tiling and Upscaling")

    # 调用 tiled_scale 进行上采样
    s = tiled_scale(
        samples=tensor.to(torch.float16),
        function=lambda a: upscale_model(a),
        tile_x=tile,
        tile_y=tile,
        overlap=overlap,
        upscale_amount=upscale_model.scale,
        pbar=pbar,
    )

    # 将模型移回输出设备
    upscale_model.to(output_device)
    # 返回上采样后的结果
    return s


# 定义一个函数，用于对批量的潜变量进行上采样并拼接
def upscale_batch_and_concatenate(upscale_model, latents, inf_device, output_device="cpu") -> torch.Tensor:
    # 初始化一个空列表以存储上采样的潜变量
    upscaled_latents = []
    # 遍历每个潜变量
    for i in range(latents.size(0)):
        latent = latents[i]
        # 对当前潜变量进行上采样
        upscaled_latent = upscale(upscale_model, latent, inf_device, output_device)
        # 将上采样结果添加到列表中
        upscaled_latents.append(upscaled_latent)
    # 返回拼接后的张量
    return torch.stack(upscaled_latents)


# 定义一个函数，用于保存视频文件
def save_video(tensor: Union[List[np.ndarray], List[PIL.Image.Image]], fps: int = 8):
    # 获取当前时间戳并格式化为字符串
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 定义视频文件的保存路径
    video_path = f"./output/{timestamp}.mp4"
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # 将张量导出为视频文件
    export_to_video(tensor, video_path, fps=fps)
    # 返回视频文件路径
    return video_path


# 定义一个进度条类
class ProgressBar:
    def __init__(self, total, desc=None):
        # 初始化总步骤数和当前步骤数
        self.total = total
        self.current = 0
        # 创建进度条对象
        self.b_unit = tqdm.tqdm(total=total, desc="ProgressBar context index: 0" if desc is None else desc)

    # 更新进度条的方法
    def update(self, value):
        # 如果传入值超过总数，则设置为总数
        if value > self.total:
            value = self.total
        # 更新当前进度
        self.current = value
        # 刷新进度条显示
        if self.b_unit is not None:
            self.b_unit.set_description("ProgressBar context index: {}".format(self.current))
            self.b_unit.refresh()

            # 更新进度
            self.b_unit.update(self.current)
```

# `.\cogvideo-finetune\inference\gradio_web_demo.py`

```
"""
# 主文件用于 Gradio 网络演示，使用 CogVideoX-2B 模型生成视频
# 设置环境变量 OPENAI_API_KEY 使用 OpenAI API 增强提示

# 此演示仅支持文本到视频的生成模型。
# 如果希望使用图像到视频或视频到视频生成模型，
# 请使用 gradio_composite_demo 实现完整的 GUI 功能。

# 使用方法：
# OpenAI_API_KEY=your_openai_api_key OpenAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py
"""

# 导入操作系统相关功能
import os
# 导入多线程功能
import threading
# 导入时间功能
import time

# 导入 Gradio 库以构建 Web 应用
import gradio as gr
# 导入 PyTorch 库进行深度学习
import torch
# 导入 CogVideoXPipeline 模型
from diffusers import CogVideoXPipeline
# 导入导出视频功能
from diffusers.utils import export_to_video
# 导入日期时间处理功能
from datetime import datetime, timedelta
# 导入 OpenAI 库以使用其 API
from openai import OpenAI
# 导入 MoviePy 库进行视频编辑
import moviepy.editor as mp

# 从预训练模型加载 CogVideoXPipeline，指定数据类型为 bfloat16，并移动到 GPU
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")

# 启用 VAE 的切片功能
pipe.vae.enable_slicing()
# 启用 VAE 的平铺功能
pipe.vae.enable_tiling()

# 创建输出目录，如果已存在则不报错
os.makedirs("./output", exist_ok=True)
# 创建临时目录，如果已存在则不报错
os.makedirs("./gradio_tmp", exist_ok=True)

# 定义系统提示，指导视频生成的描述
sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

# 定义转换提示的函数，接受提示和重试次数作为参数
def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    # 如果没有设置 OpenAI API 密钥，返回原始提示
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt

    # 创建 OpenAI 客户端
    client = OpenAI()
    # 去除提示两端的空白
    text = prompt.strip()

    # 返回原始提示
    return prompt

# 定义推断函数，接受提示、推断步骤和引导尺度
def infer(prompt: str, num_inference_steps: int, guidance_scale: float, progress=gr.Progress(track_tqdm=True)):
    # 清空 GPU 缓存
    torch.cuda.empty_cache()
    # 使用模型生成视频，指定相关参数
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        guidance_scale=guidance_scale,
    ).frames[0]

    # 返回生成的视频
    return video

# 定义保存视频的函数，接受张量作为参数
def save_video(tensor):
    # 获取当前时间戳，用于生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 定义视频保存路径
    video_path = f"./output/{timestamp}.mp4"
    # 创建视频保存目录，如果已存在则不报错
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # 将张量导出为视频文件
    export_to_video(tensor, video_path)
    # 返回视频文件路径
    return video_path

# 定义将视频转换为 GIF 的函数，接受视频路径作为参数
def convert_to_gif(video_path):
    # 使用 MoviePy 加载视频文件
    clip = mp.VideoFileClip(video_path)
    # 设置视频的帧率为 8
    clip = clip.set_fps(8)
    # 调整剪辑的高度为 240 像素，保持宽高比
        clip = clip.resize(height=240)
        # 将视频路径中的 ".mp4" 后缀替换为 ".gif" 后缀，生成 GIF 文件路径
        gif_path = video_path.replace(".mp4", ".gif")
        # 将剪辑写入 GIF 文件，设置每秒帧数为 8
        clip.write_gif(gif_path, fps=8)
        # 返回生成的 GIF 文件路径
        return gif_path
# 定义删除旧文件的函数
def delete_old_files():
    # 无限循环，持续执行删除旧文件的任务
    while True:
        # 获取当前时间
        now = datetime.now()
        # 计算10分钟前的时间，用于判断文件是否过期
        cutoff = now - timedelta(minutes=10)
        # 定义需要清理的目录列表
        directories = ["./output", "./gradio_tmp"]

        # 遍历每个目录
        for directory in directories:
            # 遍历目录中的每个文件
            for filename in os.listdir(directory):
                # 构建文件的完整路径
                file_path = os.path.join(directory, filename)
                # 检查该路径是否为文件
                if os.path.isfile(file_path):
                    # 获取文件的最后修改时间
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    # 判断文件是否早于截止时间
                    if file_mtime < cutoff:
                        # 删除该文件
                        os.remove(file_path)
        # 每600秒（10分钟）暂停一次
        time.sleep(600)

# 启动一个线程来执行删除旧文件的函数，设置为守护线程
threading.Thread(target=delete_old_files, daemon=True).start()

# 使用 Gradio 创建用户界面
with gr.Blocks() as demo:
    # 创建 Markdown 组件，显示标题
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               CogVideoX Gradio Simple Space🤗
            """)

    # 创建一行布局
    with gr.Row():
        # 创建一列布局
        with gr.Column():
            # 创建文本框用于输入提示
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)

            # 创建一个行布局
            with gr.Row():
                # 创建 Markdown 组件，说明增强提示按钮的功能
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one."
                )
                # 创建增强提示的按钮
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")

            # 创建另一列布局
            with gr.Column():
                # 创建 Markdown 组件，描述可选参数
                gr.Markdown(
                    "**Optional Parameters** (default values are recommended)<br>"
                    "Increasing the number of inference steps will produce more detailed videos, but it will slow down the process.<br>"
                    "50 steps are recommended for most cases.<br>"
                )
                # 创建一行布局，包含推理步数和引导比例输入框
                with gr.Row():
                    num_inference_steps = gr.Number(label="Inference Steps", value=50)
                    guidance_scale = gr.Number(label="Guidance Scale", value=6.0)
                # 创建生成视频的按钮
                generate_button = gr.Button("🎬 Generate Video")

        # 创建另一列布局
        with gr.Column():
            # 创建视频输出组件
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            # 创建一行布局，包含下载按钮
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)

    # 定义生成视频的函数
    def generate(prompt, num_inference_steps, guidance_scale, model_choice, progress=gr.Progress(track_tqdm=True)):
        # 调用推理函数生成张量
        tensor = infer(prompt, num_inference_steps, guidance_scale, progress=progress)
        # 保存生成的视频并获取其路径
        video_path = save_video(tensor)
        # 更新视频输出组件为可见，并设置视频路径
        video_update = gr.update(visible=True, value=video_path)
        # 将视频转换为 GIF 并获取其路径
        gif_path = convert_to_gif(video_path)
        # 更新 GIF 下载按钮为可见，并设置 GIF 路径
        gif_update = gr.update(visible=True, value=gif_path)

        # 返回视频路径和更新信息
        return video_path, video_update, gif_update

    # 定义增强提示的函数
    def enhance_prompt_func(prompt):
        # 转换提示并允许重试一次
        return convert_prompt(prompt, retry_times=1)
    # 为生成按钮添加点击事件，触发生成函数
        generate_button.click(
            # 绑定生成函数到点击事件
            generate,
            # 定义输入组件，包括提示文本、推理步骤数和引导尺度
            inputs=[prompt, num_inference_steps, guidance_scale],
            # 定义输出组件，包括视频输出和下载按钮
            outputs=[video_output, download_video_button, download_gif_button],
        )
    
    # 为增强按钮添加点击事件，触发增强提示函数
        enhance_button.click(enhance_prompt_func, 
            # 定义输入组件，包括提示文本
            inputs=[prompt], 
            # 定义输出组件，更新提示文本
            outputs=[prompt]
        )
# 检查当前模块是否为主程序入口
if __name__ == "__main__":
    # 调用 demo 对象的 launch 方法
    demo.launch()
```

# CogVideo & CogVideoX

[Read this in English](./README_zh.md)

[中文阅读](./README_zh.md)

<div align="center">
<img src=resources/logo.svg width="50%"/>
</div>
<p align="center">
<a href="https://huggingface.co/spaces/THUDM/CogVideoX-5B" target="_blank"> 🤗 Huggingface Space</a> または <a href="https://modelscope.cn/studios/ZhipuAI/CogVideoX-5b-demo" target="_blank"> 🤖 ModelScope Space</a> で CogVideoX-5B モデルをオンラインで体験してください
</p>
<p align="center">
📚 <a href="https://arxiv.org/abs/2408.06072" target="_blank">論文</a>と<a href="https://zhipu-ai.feishu.cn/wiki/DHCjw1TrJiTyeukfc9RceoSRnCh" target="_blank">使用ドキュメント</a>を表示します。
</p>
<p align="center">
    👋 <a href="resources/WECHAT.md" target="_blank">WeChat</a> と <a href="https://discord.gg/dCGfUsagrD" target="_blank">Discord</a> に参加
</p>
<p align="center">
📍 <a href="https://chatglm.cn/video?lang=en?fr=osm_cogvideo">清影</a> と <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">APIプラットフォーム</a> を訪問して、より大規模な商用ビデオ生成モデルを体験.
</p>

## 更新とニュース

- 🔥🔥 **ニュース**: ```py/10/13```: コスト削減のため、単一の4090 GPUで`CogVideoX-5B`を微調整できるフレームワーク [cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory) がリリースされました。複数の解像度での微調整に対応しています。ぜひご利用ください！- 🔥**ニュース**: ```py/10/10```: 技術報告書を更新し、より詳細なトレーニング情報とデモを追加しました。
- 🔥 **ニュース**: ```py/10/10```: 技術報告書を更新しました。[こちら](https://arxiv.org/pdf/2408.06072) をクリックしてご覧ください。さらにトレーニングの詳細とデモを追加しました。デモを見るには[こちら](https://yzy-thu.github.io/CogVideoX-demo/)をクリックしてください。
- 🔥**ニュース**: ```py/10/09```: 飛書の[技術ドキュメント](https://zhipu-ai.feishu.cn/wiki/DHCjw1TrJiTyeukfc9RceoSRnCh)でCogVideoXの微調整ガイドを公開しています。分配の自由度をさらに高めるため、公開されているドキュメント内のすべての例が完全に再現可能です。
- 🔥**ニュース**: ```py/9/19```: CogVideoXシリーズの画像生成ビデオモデル **CogVideoX-5B-I2V**
  をオープンソース化しました。このモデルは、画像を背景入力として使用し、プロンプトワードと組み合わせてビデオを生成することができ、より高い制御性を提供します。これにより、CogVideoXシリーズのモデルは、テキストからビデオ生成、ビデオの継続、画像からビデオ生成の3つのタスクをサポートするようになりました。オンラインでの[体験](https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space)
  をお楽しみください。
- 🔥🔥 **ニュース**: ```py/9/19```:
  CogVideoXのトレーニングプロセスでビデオデータをテキスト記述に変換するために使用されるキャプションモデル [CogVLM2-Caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption)
  をオープンソース化しました。ダウンロードしてご利用ください。
- 🔥 ```py/8/27```: CogVideoXシリーズのより大きなモデル **CogVideoX-5B**
  をオープンソース化しました。モデルの推論性能を大幅に最適化し、推論のハードルを大幅に下げました。`GTX 1080TI` などの旧型GPUで
  **CogVideoX-2B** を、`RTX 3060` などのデスクトップGPUで **CogVideoX-5B**
  モデルを実行できます。依存関係を更新・インストールするために、[要件](requirements.txt)
  を厳守し、推論コードは [cli_demo](inference/cli_demo.py) を参照してください。さらに、**CogVideoX-2B** モデルのオープンソースライセンスが
  **Apache 2.0 ライセンス** に変更されました。
- 🔥 ```py/8/6```: **CogVideoX-2B** 用の **3D Causal VAE** をオープンソース化しました。これにより、ビデオをほぼ無損失で再構築することができます。
- 🔥 ```py/8/6```: CogVideoXシリーズのビデオ生成モデルの最初のモデル、**CogVideoX-2B** をオープンソース化しました。
- 🌱 **ソース**: ```py/5/19```: CogVideoビデオ生成モデルをオープンソース化しました（現在、`CogVideo`
  ブランチで確認できます）。これは、トランスフォーマーに基づく初のオープンソース大規模テキスト生成ビデオモデルです。技術的な詳細については、[ICLR'23論文](https://arxiv.org/abs/2205.15868)
  をご覧ください。

**より強力なモデルが、より大きなパラメータサイズで登場予定です。お楽しみに！**

## 目次

特定のセクションにジャンプ：

- [クイックスタート](#クイックスタート)
    - [SAT](#sat)
    - [Diffusers](#Diffusers)
- [CogVideoX-2B ギャラリー](#CogVideoX-2B-ギャラリー)
- [モデル紹介](#モデル紹介)
- [プロジェクト構造](#プロジェクト構造)
    - [推論](#推論)
    - [sat](#sat)
    - [ツール](#ツール)
- [プロジェクト計画](#プロジェクト計画)
- [モデルライセンス](#モデルライセンス)
- [CogVideo(ICLR'23)モデル紹介](#CogVideoICLR23)
- [引用](#引用)

## クイックスタート

### プロンプトの最適化

モデルを実行する前に、[こちら](inference/convert_demo.py)
を参考にして、GLM-4（または同等の製品、例えばGPT-4）の大規模モデルを使用してどのようにモデルを最適化するかをご確認ください。これは非常に重要です。モデルは長いプロンプトでトレーニングされているため、良いプロンプトがビデオ生成の品質に直接影響を与えます。

### SAT

[sat_demo](sat/README.md) の指示に従ってください:
SATウェイトの推論コードと微調整コードが含まれています。CogVideoXモデル構造に基づいて改善することをお勧めします。革新的な研究者は、このコードを使用して迅速なスタッキングと開発を行うことができます。

### Diffusers

```py
pip install -r requirements.txt
```

次に [diffusers_demo](inference/cli_demo.py) を参照してください: 推論コードの詳細な説明が含まれており、一般的なパラメータの意味についても言及しています。

量子化推論の詳細については、[diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao/) を参照してください。Diffusers
と TorchAO を使用することで、量子化推論も可能となり、メモリ効率の良い推論や、コンパイル時に場合によっては速度の向上が期待できます。A100
および H100
上でのさまざまな設定におけるメモリおよび時間のベンチマークの完全なリストは、[diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao)
に公開されています。

## Gallery

### CogVideoX-5B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/cf5953ea-96d3-48fd-9907-c4708752c714" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fe0a78e6-b669-4800-8cf0-b5f9b5145b52" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/c182f606-8f8c-421d-b414-8487070fcfcb" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7db2bbce-194d-434d-a605-350254b6c298" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/62b01046-8cab-44cc-bd45-4d965bb615ec" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d78e552a-4b3f-4b81-ac3f-3898079554f6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/30894f12-c741-44a2-9e6e-ddcacc231e5b" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/926575ca-7150-435b-a0ff-4900a963297b" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-2B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9de41efd-d4d1-4095-aeda-246dd834e91d" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/941d6661-6a8d-4a1b-b912-59606f0b2841" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/938529c4-91ae-4f60-b96b-3c3947fa63cb" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

ギャラリーの対応するプロンプトワードを表示するには、[こちら](resources/galary_prompt.md)をクリックしてください

## モデル紹介

CogVideoXは、[清影](https://chatglm.cn/video?fr=osm_cogvideox) と同源のオープンソース版ビデオ生成モデルです。
以下の表に、提供しているビデオ生成モデルの基本情報を示します:

<table  style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">モデル名</th>
    <th style="text-align: center;">CogVideoX-2B</th>
    <th style="text-align: center;">CogVideoX-5B</th>
    <th style="text-align: center;">CogVideoX-5B-I2V </th>
  </tr>
  <tr>
    <td style="text-align: center;">推論精度</td>
    <td style="text-align: center;"><b>FP16*(推奨)</b>, BF16, FP32, FP8*, INT8, INT4は非対応</td>
    <td colspan="2" style="text-align: center;"><b>BF16(推奨)</b>, FP16, FP32, FP8*, INT8, INT4は非対応</td>
  </tr>
  <tr>
    <td style="text-align: center;">単一GPUのメモリ消費<br></td>
    <td style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> FP16: 18GB <br><b>diffusers FP16: 4GBから* </b><br><b>diffusers INT8(torchao): 3.6GBから*</b></td>
    <td colspan="2" style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> BF16: 26GB <br><b>diffusers BF16 : 5GBから* </b><br><b>diffusers INT8(torchao): 4.4GBから* </b></td>
  </tr>
  <tr>
    <td style="text-align: center;">マルチGPUのメモリ消費</td>
    <td style="text-align: center;"><b>FP16: 10GB* using diffusers</b><br></td>
    <td colspan="2" style="text-align: center;"><b>BF16: 15GB* using diffusers</b><br></td>
  </tr>
  <tr>
    <td style="text-align: center;">推論速度<br>(ステップ = 50, FP/BF16)</td>
    <td style="text-align: center;">単一A100: 約90秒<br>単一H100: 約45秒</td>
    <td colspan="2" style="text-align: center;">単一A100: 約180秒<br>単一H100: 約90秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">ファインチューニング精度</td>
    <td style="text-align: center;"><b>FP16</b></td>
    <td colspan="2" style="text-align: center;"><b>BF16</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">ファインチューニング時のメモリ消費</td>
    <td style="text-align: center;">47 GB (bs=1, LORA)<br> 61 GB (bs=2, LORA)<br> 62GB (bs=1, SFT)</td>
    <td style="text-align: center;">63 GB (bs=1, LORA)<br> 80 GB (bs=2, LORA)<br> 75GB (bs=1, SFT)<br></td>
    <td style="text-align: center;">78 GB (bs=1, LORA)<br> 75GB (bs=1, SFT, 16GPU)<br></td>
  </tr>
  <tr>
    <td style="text-align: center;">プロンプト言語</td>
    <td colspan="3" style="text-align: center;">英語*</td>
  </tr>
  <tr>
    <td style="text-align: center;">プロンプトの最大トークン数</td>
    <td colspan="3" style="text-align: center;">226トークン</td>
  </tr>
  <tr>
    <td style="text-align: center;">ビデオの長さ</td>
    <td colspan="3" style="text-align: center;">6秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">フレームレート</td>
    <td colspan="3" style="text-align: center;">8フレーム/秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">ビデオ解像度</td>
    <td colspan="3" style="text-align: center;">720 * 480、他の解像度は非対応(ファインチューニング含む)</td>
  </tr>
  <tr>
    <td style="text-align: center;">位置エンコーディング</td>
    <td style="text-align: center;">3d_sincos_pos_embed</td>
    <td style="text-align: center;">3d_sincos_pos_embed</td>
    <td style="text-align: center;">3d_rope_pos_embed + learnable_pos_embed</td>
  </tr>
  <tr>
    <td style="text-align: center;">ダウンロードリンク (Diffusers)</td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-2b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-2b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-2b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b-I2V">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b-I2V">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b-I2V">🟣 WiseModel</a></td>
  </tr>
  <tr>
    <td style="text-align: center;">ダウンロードリンク (SAT)</td>
    <td colspan="3" style="text-align: center;"><a href="./sat/README_ja.md">SAT</a></td>
  </tr>
</table>

**データ解説**

+ diffusersライブラリを使用してテストする際には、`diffusers`ライブラリが提供する全ての最適化が有効になっています。この方法は
  **NVIDIA A100 / H100**以外のデバイスでのメモリ/メモリ消費のテストは行っていません。通常、この方法は**NVIDIA
  Ampereアーキテクチャ**
  以上の全てのデバイスに適応できます。最適化を無効にすると、メモリ消費は倍増し、ピークメモリ使用量は表の3倍になりますが、速度は約3〜4倍向上します。以下の最適化を部分的に無効にすることが可能です:

```py
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

+ マルチGPUで推論する場合、`enable_sequential_cpu_offload()`最適化を無効にする必要があります。
+ INT8モデルを使用すると推論速度が低下しますが、これはメモリの少ないGPUで正常に推論を行い、ビデオ品質の損失を最小限に抑えるための措置です。推論速度は大幅に低下します。
+ CogVideoX-2Bモデルは`FP16`精度でトレーニングされており、CogVideoX-5Bモデルは`BF16`
  精度でトレーニングされています。推論時にはモデルがトレーニングされた精度を使用することをお勧めします。
+ [PytorchAO](https://github.com/pytorch/ao)および[Optimum-quanto](https://github.com/huggingface/optimum-quanto/)
  は、CogVideoXのメモリ要件を削減するためにテキストエンコーダ、トランスフォーマ、およびVAEモジュールを量子化するために使用できます。これにより、無料のT4
  Colabやより少ないメモリのGPUでモデルを実行することが可能になります。同様に重要なのは、TorchAOの量子化は`torch.compile`
  と完全に互換性があり、推論速度を大幅に向上させることができる点です。`NVIDIA H100`およびそれ以上のデバイスでは`FP8`
  精度を使用する必要があります。これには、`torch`、`torchao`、`diffusers`、`accelerate`
  Pythonパッケージのソースコードからのインストールが必要です。`CUDA 12.4`の使用をお勧めします。
+ 推論速度テストも同様に、上記のメモリ最適化方法を使用しています。メモリ最適化を使用しない場合、推論速度は約10％向上します。
  `diffusers`バージョンのモデルのみが量子化をサポートしています。
+ モデルは英語入力のみをサポートしており、他の言語は大規模モデルの改善を通じて英語に翻訳できます。
+ モデルのファインチューニングに使用されるメモリは`8 * H100`環境でテストされています。プログラムは自動的に`Zero 2`
  最適化を使用しています。表に具体的なGPU数が記載されている場合、ファインチューニングにはその数以上のGPUが必要です。

## 友好的リンク

コミュニティからの貢献を大歓迎し、私たちもオープンソースコミュニティに積極的に貢献しています。以下の作品はすでにCogVideoXに対応しており、ぜひご利用ください：

+ [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun): CogVideoX-Funは、CogVideoXアーキテクチャを基にした改良パイプラインで、自由な解像度と複数の起動方法をサポートしています。
+ [CogStudio](https://github.com/pinokiofactory/cogstudio): CogVideo の Gradio Web UI の別のリポジトリ。より高機能な Web UI をサポートします。
+ [Xorbits Inference](https://github.com/xorbitsai/inference):
  強力で包括的な分散推論フレームワークであり、ワンクリックで独自のモデルや最新のオープンソースモデルを簡単にデプロイできます。
+ [ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper)
  ComfyUIフレームワークを使用して、CogVideoXをワークフローに統合します。
+ [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys): VideoSysは、使いやすく高性能なビデオ生成インフラを提供し、最新のモデルや技術を継続的に統合しています。
+ [AutoDLイメージ](https://www.codewithgpu.com/i/THUDM/CogVideo/CogVideoX-5b-demo): コミュニティメンバーが提供するHuggingface
  Spaceイメージのワンクリックデプロイメント。
+ [インテリアデザイン微調整モデル](https://huggingface.co/collections/bertjiazheng/koolcogvideox-66e4762f53287b7f39f8f3ba): は、CogVideoXを基盤にした微調整モデルで、インテリアデザイン専用に設計されています。
+ [xDiT](https://github.com/xdit-project/xDiT): xDiTは、複数のGPUクラスター上でDiTsを並列推論するためのエンジンです。xDiTはリアルタイムの画像およびビデオ生成サービスをサポートしています。

## プロジェクト構造

このオープンソースリポジトリは、**CogVideoX** オープンソースモデルの基本的な使用方法と微調整の例を迅速に開始するためのガイドです。

### Colabでのクイックスタート

無料のColab T4上で直接実行できる3つのプロジェクトを提供しています。

+ [CogVideoX-5B-T2V-Colab.ipynb](https://colab.research.google.com/drive/1pCe5s0bC_xuXbBlpvIH1z0kfdTLQPzCS?usp=sharing):
  CogVideoX-5B テキストからビデオへの生成用Colabコード。
+ [CogVideoX-5B-T2V-Int8-Colab.ipynb](https://colab.research.google.com/drive/1DUffhcjrU-uz7_cpuJO3E_D4BaJT7OPa?usp=sharing):
  CogVideoX-5B テキストからビデオへの量子化推論用Colabコード。1回の実行に約30分かかります。
+ [CogVideoX-5B-I2V-Colab.ipynb](https://colab.research.google.com/drive/17CqYCqSwz39nZAX2YyonDxosVKUZGzcX?usp=sharing):
  CogVideoX-5B 画像からビデオへの生成用Colabコード。
+ [CogVideoX-5B-V2V-Colab.ipynb](https://colab.research.google.com/drive/1comfGAUJnChl5NwPuO8Ox5_6WCy4kbNN?usp=sharing):
  CogVideoX-5B ビデオからビデオへの生成用Colabコード。

### Inference

+ [cli_demo](inference/cli_demo.py): 推論コードの詳細な説明が含まれており、一般的なパラメータの意味についても言及しています。
+ [cli_demo_quantization](inference/cli_demo_quantization.py):
  量子化モデル推論コードで、低メモリのデバイスでも実行可能です。また、このコードを変更して、FP8 精度の CogVideoX
  モデルの実行をサポートすることもできます。
+ [diffusers_vae_demo](inference/cli_vae_demo.py): VAE推論コードの実行には現在71GBのメモリが必要ですが、将来的には最適化される予定です。
+ [space demo](inference/gradio_composite_demo): Huggingface Spaceと同じGUIコードで、フレーム補間や超解像ツールが組み込まれています。

<div style="text-align: center;">
    <img src="resources/web_demo.png" style="width: 100%; height: auto;" />
</div>

+ [convert_demo](inference/convert_demo.py):
  ユーザー入力をCogVideoXに適した形式に変換する方法。CogVideoXは長いキャプションでトレーニングされているため、入力テキストをLLMを使用してトレーニング分布と一致させる必要があります。デフォルトではGLM-4を使用しますが、GPT、Geminiなどの他のLLMに置き換えることもできます。
+ [gradio_web_demo](inference/gradio_web_demo.py): CogVideoX-2B / 5B モデルを使用して動画を生成する方法を示す、シンプルな
  Gradio Web UI デモです。私たちの Huggingface Space と同様に、このスクリプトを使用して Web デモを起動することができます。

### finetune

+ [train_cogvideox_lora](finetune/README_ja.md): CogVideoX diffusers 微調整方法の詳細な説明が含まれています。このコードを使用して、自分のデータセットで
  CogVideoX を微調整することができます。

### sat

+ [sat_demo](sat/README.md):
  SATウェイトの推論コードと微調整コードが含まれています。CogVideoXモデル構造に基づいて改善することをお勧めします。革新的な研究者は、このコードを使用して迅速なスタッキングと開発を行うことができます。

### ツール

このフォルダには、モデル変換/キャプション生成などのツールが含まれています。

+ [convert_weight_sat2hf](tools/convert_weight_sat2hf.py): SAT モデルの重みを Huggingface モデルの重みに変換します。
+ [caption_demo](tools/caption/README_ja.md): Caption ツール、ビデオを理解してテキストで出力するモデル。
+ [export_sat_lora_weight](tools/export_sat_lora_weight.py): SAT ファインチューニングモデルのエクスポートツール、SAT Lora
  Adapter を diffusers 形式でエクスポートします。
+ [load_cogvideox_lora](tools/load_cogvideox_lora.py): diffusers 版のファインチューニングされた Lora Adapter
  をロードするためのツールコード。
+ [llm_flux_cogvideox](tools/llm_flux_cogvideox/llm_flux_cogvideox.py): オープンソースのローカル大規模言語モデル +
  Flux + CogVideoX を使用して自動的に動画を生成します。
+ [parallel_inference_xdit](tools/parallel_inference/parallel_inference_xdit.py)：
[xDiT](https://github.com/xdit-project/xDiT)
  によってサポートされ、ビデオ生成プロセスを複数の GPU で並列化します。
+ [cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory): CogVideoXの低コスト微調整フレームワークで、
`diffusers`バージョンのモデルに適応しています。より多くの解像度に対応し、単一の4090 GPUでCogVideoX-5Bの微調整が可能です。

## CogVideo(ICLR'23)

論文の公式リポジトリ: [CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)
は [CogVideo branch](https://github.com/THUDM/CogVideo/tree/CogVideo) にあります。

**CogVideoは比較的高フレームレートのビデオを生成することができます。**
32フレームの4秒間のクリップが以下に示されています。

![High-frame-rate sample](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/appendix-sample-highframerate.png)

![Intro images](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/intro-image.png)
<div align="center">
  <video src="https://github.com/user-attachments/assets/2fa19651-e925-4a2a-b8d6-b3f216d490ba" width="80%" controls autoplay></video>
</div>


CogVideoのデモは [https://models.aminer.cn/cogvideo](https://models.aminer.cn/cogvideo/) で体験できます。
*元の入力は中国語です。*

## 引用

🌟 私たちの仕事が役立つと思われた場合、ぜひスターを付けていただき、論文を引用してください。

```py
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
@article{hong2022cogvideo,
  title={CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers},
  author={Hong, Wenyi and Ding, Ming and Zheng, Wendi and Liu, Xinghan and Tang, Jie},
  journal={arXiv preprint arXiv:2205.15868},
  year={2022}
}
```

あなたの貢献をお待ちしています！詳細は[こちら](resources/contribute_ja.md)をクリックしてください。

## ライセンス契約

このリポジトリのコードは [Apache 2.0 License](LICENSE) の下で公開されています。

CogVideoX-2B モデル (対応するTransformersモジュールやVAEモジュールを含む) は
[Apache 2.0 License](LICENSE) の下で公開されています。

CogVideoX-5B モデル（Transformers モジュール、画像生成ビデオとテキスト生成ビデオのバージョンを含む） は
[CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) の下で公開されています。


# CogVideo & CogVideoX

[Read this in English](./README_zh.md)

[日本語で読む](./README_ja.md)


<div align="center">
<img src=resources/logo.svg width="50%"/>
</div>
<p align="center">
在 <a href="https://huggingface.co/spaces/THUDM/CogVideoX-5B" target="_blank"> 🤗 Huggingface Space</a> 或 <a href="https://modelscope.cn/studios/ZhipuAI/CogVideoX-5b-demo" target="_blank"> 🤖 ModelScope Space</a> 在线体验 CogVideoX-5B 模型
</p>
<p align="center">
📚 查看 <a href="https://arxiv.org/abs/2408.06072" target="_blank">论文</a> 和 <a href="https://zhipu-ai.feishu.cn/wiki/DHCjw1TrJiTyeukfc9RceoSRnCh" target="_blank">使用文档</a>
</p>
<p align="center">
    👋 加入我们的 <a href="resources/WECHAT.md" target="_blank">微信</a> 和  <a href="https://discord.gg/dCGfUsagrD" target="_blank">Discord</a> 
</p>
<p align="center">
📍 前往<a href="https://chatglm.cn/video?fr=osm_cogvideox"> 清影</a> 和 <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9"> API平台</a> 体验更大规模的商业版视频生成模型。
</p>

## 项目更新

- 🔥🔥 **News**: ```py/10/13```: 成本更低，单卡4090可微调`CogVideoX-5B`的微调框架[cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory)已经推出，多种分辨率微调，欢迎使用。
- 🔥 **News**: ```py/10/10```: 我们更新了我们的技术报告,请点击 [这里](https://arxiv.org/pdf/2408.06072) 查看，附上了更多的训练细节和demo，关于demo，点击[这里](https://yzy-thu.github.io/CogVideoX-demo/) 查看。
- 🔥 **News**: ```py/10/09```: 我们在飞书[技术文档](https://zhipu-ai.feishu.cn/wiki/DHCjw1TrJiTyeukfc9RceoSRnCh")公开CogVideoX微调指导，以进一步增加分发自由度，公开文档中所有示例可以完全复现
- 🔥 **News**: ```py/9/19```: 我们开源 CogVideoX 系列图生视频模型 **CogVideoX-5B-I2V**
  。该模型可以将一张图像作为背景输入，结合提示词一起生成视频，具有更强的可控性。
  至此，CogVideoX系列模型已经支持文本生成视频，视频续写，图片生成视频三种任务。欢迎前往在线[体验](https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space)。
- 🔥 **News**: ```py/9/19```: CogVideoX 训练过程中用于将视频数据转换为文本描述的 Caption
  模型 [CogVLM2-Caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption)
  已经开源。欢迎前往下载并使用。
- 🔥 ```py/8/27```:  我们开源 CogVideoX 系列更大的模型 **CogVideoX-5B**
  。我们大幅度优化了模型的推理性能，推理门槛大幅降低，您可以在 `GTX 1080TI` 等早期显卡运行 **CogVideoX-2B**，在 `RTX 3060`
  等桌面端甜品卡运行 **CogVideoX-5B** 模型。 请严格按照[要求](requirements.txt)
  更新安装依赖，推理代码请查看 [cli_demo](inference/cli_demo.py)。同时，**CogVideoX-2B** 模型开源协议已经修改为**Apache 2.0
  协议**。
- 🔥 ```py/8/6```: 我们开源 **3D Causal VAE**，用于 **CogVideoX-2B**，可以几乎无损地重构视频。
- 🔥 ```py/8/6```: 我们开源 CogVideoX 系列视频生成模型的第一个模型, **CogVideoX-2B**。
- 🌱 **Source**: ```py/5/19```: 我们开源了 CogVideo 视频生成模型（现在你可以在 `CogVideo` 分支中看到），这是首个开源的基于
  Transformer 的大型文本生成视频模型，您可以访问 [ICLR'23 论文](https://arxiv.org/abs/2205.15868) 查看技术细节。

## 目录

跳转到指定部分：

- [快速开始](#快速开始)
    - [SAT](#sat)
    - [Diffusers](#Diffusers)
- [CogVideoX-2B 视频作品](#cogvideox-2b-视频作品)
- [CogVideoX模型介绍](#模型介绍)
- [完整项目代码结构](#完整项目代码结构)
    - [Inference](#inference)
    - [SAT](#sat)
    - [Tools](#tools)
- [开源项目规划](#开源项目规划)
- [模型协议](#模型协议)
- [CogVideo(ICLR'23)模型介绍](#cogvideoiclr23)
- [引用](#引用)

## 快速开始

### 提示词优化

在开始运行模型之前，请参考 [这里](inference/convert_demo.py) 查看我们是怎么使用GLM-4(或者同级别的其他产品，例如GPT-4)
大模型对模型进行优化的，这很重要，
由于模型是在长提示词下训练的，一个好的提示词直接影响了视频生成的质量。

### SAT

查看sat文件夹下的 [sat_demo](sat/README.md)：包含了 SAT 权重的推理代码和微调代码，推荐基于此代码进行 CogVideoX
模型结构的改进，研究者使用该代码可以更好的进行快速的迭代和开发。

### Diffusers

```py
pip install -r requirements.txt
```

查看[diffusers_demo](inference/cli_demo.py)：包含对推理代码更详细的解释，包括各种关键的参数。

欲了解更多关于量化推理的细节，请参考 [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao/)。使用 Diffusers
和 TorchAO，量化推理也是可能的，这可以实现内存高效的推理，并且在某些情况下编译后速度有所提升。有关在 A100 和 H100
上使用各种设置的内存和时间基准测试的完整列表，已发布在 [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao)
上。

## 视频作品

### CogVideoX-5B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/cf5953ea-96d3-48fd-9907-c4708752c714" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fe0a78e6-b669-4800-8cf0-b5f9b5145b52" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/c182f606-8f8c-421d-b414-8487070fcfcb" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7db2bbce-194d-434d-a605-350254b6c298" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/62b01046-8cab-44cc-bd45-4d965bb615ec" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d78e552a-4b3f-4b81-ac3f-3898079554f6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/30894f12-c741-44a2-9e6e-ddcacc231e5b" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/926575ca-7150-435b-a0ff-4900a963297b" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-2B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9de41efd-d4d1-4095-aeda-246dd834e91d" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/941d6661-6a8d-4a1b-b912-59606f0b2841" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/938529c4-91ae-4f60-b96b-3c3947fa63cb" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>


查看画廊的对应提示词，请点击[这里](resources/galary_prompt.md)

## 模型介绍

CogVideoX是 [清影](https://chatglm.cn/video?fr=osm_cogvideox) 同源的开源版本视频生成模型。
下表展示我们提供的视频生成模型相关基础信息:

<table  style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">模型名</th>
    <th style="text-align: center;">CogVideoX-2B</th>
    <th style="text-align: center;">CogVideoX-5B</th>
    <th style="text-align: center;">CogVideoX-5B-I2V </th>
  </tr>
  <tr>
    <td style="text-align: center;">推理精度</td>
    <td style="text-align: center;"><b>FP16*(推荐)</b>, BF16, FP32，FP8*，INT8，不支持INT4</td>
    <td colspan="2" style="text-align: center;"><b>BF16(推荐)</b>, FP16, FP32，FP8*，INT8，不支持INT4</td>
  </tr>
  <tr>
    <td style="text-align: center;">单GPU显存消耗<br></td>
    <td style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> FP16: 18GB <br><b>diffusers FP16: 4GB起* </b><br><b>diffusers INT8(torchao): 3.6G起*</b></td>
    <td colspan="2" style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> BF16: 26GB <br><b>diffusers BF16 : 5GB起* </b><br><b>diffusers INT8(torchao): 4.4G起* </b></td>
  </tr>
  <tr>
    <td style="text-align: center;">多GPU推理显存消耗</td>
    <td style="text-align: center;"><b>FP16: 10GB* using diffusers</b><br></td>
    <td colspan="2" style="text-align: center;"><b>BF16: 15GB* using diffusers</b><br></td>
  </tr>
  <tr>
    <td style="text-align: center;">推理速度<br>(Step = 50, FP/BF16)</td>
    <td style="text-align: center;">单卡A100: ~90秒<br>单卡H100: ~45秒</td>
    <td colspan="2" style="text-align: center;">单卡A100: ~180秒<br>单卡H100: ~90秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">微调精度</td>
    <td style="text-align: center;"><b>FP16</b></td>
    <td colspan="2" style="text-align: center;"><b>BF16</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">微调显存消耗</td>
    <td style="text-align: center;">47 GB (bs=1, LORA)<br> 61 GB (bs=2, LORA)<br> 62GB (bs=1, SFT)</td>
    <td style="text-align: center;">63 GB (bs=1, LORA)<br> 80 GB (bs=2, LORA)<br> 75GB (bs=1, SFT)<br></td>
    <td style="text-align: center;">78 GB (bs=1, LORA)<br> 75GB (bs=1, SFT, 16GPU)<br></td>
  </tr>
  <tr>
    <td style="text-align: center;">提示词语言</td>
    <td colspan="3" style="text-align: center;">English*</td>
  </tr>
  <tr>
    <td style="text-align: center;">提示词长度上限</td>
    <td colspan="3" style="text-align: center;">226 Tokens</td>
  </tr>
  <tr>
    <td style="text-align: center;">视频长度</td>
    <td colspan="3" style="text-align: center;">6 秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">帧率</td>
    <td colspan="3" style="text-align: center;">8 帧 / 秒 </td>
  </tr>
  <tr>
    <td style="text-align: center;">视频分辨率</td>
    <td colspan="3" style="text-align: center;">720 * 480，不支持其他分辨率(含微调)</td>
  </tr>
    <tr>
    <td style="text-align: center;">位置编码</td>
    <td style="text-align: center;">3d_sincos_pos_embed</td>
   <td style="text-align: center;">3d_sincos_pos_embed</td>
    <td style="text-align: center;">3d_rope_pos_embed + learnable_pos_embed</td>
  </tr>
  <tr>
    <td style="text-align: center;">下载链接 (Diffusers)</td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-2b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-2b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-2b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b-I2V">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b-I2V">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b-I2V">🟣 WiseModel</a></td>
  </tr>
  <tr>
    <td style="text-align: center;">下载链接 (SAT)</td>
    <td colspan="3" style="text-align: center;"><a href="./sat/README_zh.md">SAT</a></td>
  </tr>
</table>

**数据解释**

+ 使用 diffusers 库进行测试时，启用了全部`diffusers`库自带的优化，该方案未测试在非**NVIDIA A100 / H100**
  外的设备上的实际显存 / 内存占用。通常，该方案可以适配于所有 **NVIDIA 安培架构**
  以上的设备。若关闭优化，显存占用会成倍增加，峰值显存约为表格的3倍。但速度提升3-4倍左右。你可以选择性的关闭部分优化，这些优化包括:

```py
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

+ 多GPU推理时，需要关闭 `enable_sequential_cpu_offload()` 优化。
+ 使用 INT8 模型会导致推理速度降低，此举是为了满足显存较低的显卡能正常推理并保持较少的视频质量损失，推理速度大幅降低。
+ CogVideoX-2B 模型采用 `FP16` 精度训练， 搜有 CogVideoX-5B 模型采用 `BF16` 精度训练。我们推荐使用模型训练的精度进行推理。
+ [PytorchAO](https://github.com/pytorch/ao) 和 [Optimum-quanto](https://github.com/huggingface/optimum-quanto/)
  可以用于量化文本编码器、Transformer 和 VAE 模块，以降低 CogVideoX 的内存需求。这使得在免费的 T4 Colab 或更小显存的 GPU
  上运行模型成为可能！同样值得注意的是，TorchAO 量化完全兼容 `torch.compile`，这可以显著提高推理速度。在 `NVIDIA H100`
  及以上设备上必须使用 `FP8` 精度，这需要源码安装 `torch`、`torchao`、`diffusers` 和 `accelerate` Python
  包。建议使用 `CUDA 12.4`。
+ 推理速度测试同样采用了上述显存优化方案，不采用显存优化的情况下，推理速度提升约10%。 只有`diffusers`版本模型支持量化。
+ 模型仅支持英语输入，其他语言可以通过大模型润色时翻译为英语。
+ 模型微调所占用的显存是在 `8 * H100` 环境下进行测试，程序已经自动使用`Zero 2` 优化。表格中若有标注具体GPU数量则必须使用大于等于该数量的GPU进行微调。

## 友情链接

我们非常欢迎来自社区的贡献，并积极的贡献开源社区。以下作品已经对CogVideoX进行了适配，欢迎大家使用:
+ [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun): CogVideoX-Fun是一个基于CogVideoX结构修改后的的pipeline，支持自由的分辨率，多种启动方式。
+ [CogStudio](https://github.com/pinokiofactory/cogstudio): CogVideo 的 Gradio Web UI单独实现仓库，支持更多功能的 Web UI。
+ [Xorbits Inference](https://github.com/xorbitsai/inference): 性能强大且功能全面的分布式推理框架，轻松一键部署你自己的模型或内置的前沿开源模型。
+ [ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper) 使用ComfyUI框架，将CogVideoX加入到你的工作流中。
+ [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys): VideoSys 提供了易用且高性能的视频生成基础设施，支持完整的管道，并持续集成最新的模型和技术。
+ [AutoDL镜像](https://www.codewithgpu.com/i/THUDM/CogVideo/CogVideoX-5b-demo): 由社区成员提供的一键部署Huggingface
  Space镜像。
+ [室内设计微调模型](https://huggingface.co/collections/bertjiazheng/koolcogvideox-66e4762f53287b7f39f8f3ba) 基于 CogVideoX的微调模型，它专为室内设计而设计
+ [xDiT](https://github.com/xdit-project/xDiT): xDiT是一个用于在多GPU集群上对DiTs并行推理的引擎。xDiT支持实时图像和视频生成服务。


## 完整项目代码结构

本开源仓库将带领开发者快速上手 **CogVideoX** 开源模型的基础调用方式、微调示例。

### Colab 快速使用

这里提供了三个能直接在免费的 Colab T4上 运行的项目

+ [CogVideoX-5B-T2V-Colab.ipynb](https://colab.research.google.com/drive/1pCe5s0bC_xuXbBlpvIH1z0kfdTLQPzCS?usp=sharing):
  CogVideoX-5B 文字生成视频 Colab 代码。
+ [CogVideoX-5B-T2V-Int8-Colab.ipynb](https://colab.research.google.com/drive/1DUffhcjrU-uz7_cpuJO3E_D4BaJT7OPa?usp=sharing):
  CogVideoX-5B 文字生成视频量化推理 Colab 代码，运行一次大约需要30分钟。
+ [CogVideoX-5B-I2V-Colab.ipynb](https://colab.research.google.com/drive/17CqYCqSwz39nZAX2YyonDxosVKUZGzcX?usp=sharing):
  CogVideoX-5B 图片生成视频 Colab 代码。
+ [CogVideoX-5B-V2V-Colab.ipynb](https://colab.research.google.com/drive/1comfGAUJnChl5NwPuO8Ox5_6WCy4kbNN?usp=sharing):
  CogVideoX-5B 视频生成视频 Colab 代码。

### inference

+ [cli_demo](inference/cli_demo.py): 更详细的推理代码讲解，常见参数的意义，在这里都会提及。
+ [cli_demo_quantization](inference/cli_demo_quantization.py):
  量化模型推理代码，可以在显存较低的设备上运行，也可以基于此代码修改，以支持运行FP8等精度的CogVideoX模型。请注意，FP8
  仅测试通过，且必须将 `torch-nightly`,`torchao`源代码安装，不建议在生产环境中使用。
+ [diffusers_vae_demo](inference/cli_vae_demo.py): 单独执行VAE的推理代码。
+ [space demo](inference/gradio_composite_demo): Huggingface Space同款的 GUI 代码，植入了插帧，超分工具。

<div style="text-align: center;">
    <img src="resources/web_demo.png" style="width: 100%; height: auto;" />
</div>

+ [convert_demo](inference/convert_demo.py): 如何将用户的输入转换成适合
  CogVideoX的长输入。因为CogVideoX是在长文本上训练的，所以我们需要把输入文本的分布通过LLM转换为和训练一致的长文本。脚本中默认使用GLM-4，也可以替换为GPT、Gemini等任意大语言模型。
+ [gradio_web_demo](inference/gradio_composite_demo/app.py): 与 Huggingface Space 完全相同的代码实现，快速部署 CogVideoX
  GUI体验。

### finetune

+ [train_cogvideox_lora](finetune/README_zh.md): diffusers版本 CogVideoX 模型微调方案和细节。

### sat

+ [sat_demo](sat/README_zh.md): 包含了 SAT 权重的推理代码和微调代码，推荐基于 CogVideoX
  模型结构进行改进，创新的研究者使用改代码以更好的进行快速的堆叠和开发。

### tools

本文件夹包含了一些工具，用于模型的转换 / Caption 等工作。

+ [convert_weight_sat2hf](tools/convert_weight_sat2hf.py): 将 SAT 模型权重转换为 Huggingface 模型权重。
+ [caption_demo](tools/caption/README_zh.md):  Caption 工具，对视频理解并用文字输出的模型。
+ [export_sat_lora_weight](tools/export_sat_lora_weight.py):  SAT微调模型导出工具，将
  SAT Lora Adapter 导出为 diffusers 格式。
+ [load_cogvideox_lora](tools/load_cogvideox_lora.py): 载入diffusers版微调Lora Adapter的工具代码。
+ [llm_flux_cogvideox](tools/llm_flux_cogvideox/llm_flux_cogvideox.py): 使用开源本地大语言模型 + Flux +
  CogVideoX实现自动化生成视频。
+ [parallel_inference_xdit](tools/parallel_inference/parallel_inference_xdit.py):
在多个 GPU 上并行化视频生成过程，
  由[xDiT](https://github.com/xdit-project/xDiT)提供支持。
+ [cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory): CogVideoX低成文微调框架，适配`diffusers`版本模型。支持更多分辨率，单卡4090即可微调 CogVideoX-5B 。

## CogVideo(ICLR'23)

[CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)
的官方repo位于[CogVideo branch](https://github.com/THUDM/CogVideo/tree/CogVideo)。

**CogVideo可以生成高帧率视频，下面展示了一个32帧的4秒视频。**

![High-frame-rate sample](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/appendix-sample-highframerate.png)

![Intro images](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/intro-image.png)


<div align="center">
  <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="80%" controls autoplay></video>
</div>

CogVideo的demo网站在[https://models.aminer.cn/cogvideo](https://models.aminer.cn/cogvideo/)。您可以在这里体验文本到视频生成。
*原始输入为中文。*

## 引用

🌟 如果您发现我们的工作有所帮助，欢迎引用我们的文章，留下宝贵的stars

```py
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
@article{hong2022cogvideo,
  title={CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers},
  author={Hong, Wenyi and Ding, Ming and Zheng, Wendi and Liu, Xinghan and Tang, Jie},
  journal={arXiv preprint arXiv:2205.15868},
  year={2022}
}
```

我们欢迎您的贡献，您可以点击[这里](resources/contribute_zh.md)查看更多信息。

## 模型协议

本仓库代码使用 [Apache 2.0 协议](LICENSE) 发布。

CogVideoX-2B 模型 (包括其对应的Transformers模块，VAE模块) 根据 [Apache 2.0 协议](LICENSE) 许可证发布。

CogVideoX-5B 模型 (Transformers 模块，包括图生视频，文生视频版本)
根据 [CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE)
许可证发布。
