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