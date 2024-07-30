# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\modules\transformation.py`

```py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TpsSpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, F, I_size, I_r_size, I_channel_num: int = 1):
        """Based on RARE TPS
        input:
            F : number of fiducial points (3 or 4)
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            None
        """
        super(TpsSpatialTransformerNetwork, self).__init__()
        self.F = F  # 设置 fiducial points 的数量
        self.I_size = I_size  # 输入图像 I 的尺寸 (height, width)
        self.I_r_size = I_r_size  # 矫正后图像 I_r 的尺寸 (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num  # 输入图像 I 的通道数
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # 通过 Localization Network 获取 batch 输入图像的控制点 C'，大小为 batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # 使用 GridGenerator 构建 P'，大小为 batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])

        # 使用 grid_sample 函数对 batch 输入图像进行采样，得到矫正后的图像 batch_I_r
        batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode="border")

        return batch_I_r


class LocalizationNetwork(nn.Module):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """
    # 初始化函数，用于创建一个本地化网络对象
    def __init__(self, F, I_channel_num: int):
        # 调用父类的初始化方法
        super(LocalizationNetwork, self).__init__()
        # 设定本地化网络的参数
        self.F = F  # F 是 fiducial 点的数量
        self.I_channel_num = I_channel_num  # 输入图像的通道数
        # 定义卷积神经网络的结构
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.I_channel_num,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # 第一层卷积：输入通道数为 I_channel_num，输出通道数为 64，卷积核大小为 3x3
            nn.BatchNorm2d(64),  # 批量归一化层，归一化通道数为 64 的特征图
            nn.ReLU(True),  # ReLU 激活函数，in-place 模式
            nn.MaxPool2d(2, 2),  # 最大池化层，池化窗口大小为 2x2，步长为 2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # 第二层卷积：输入通道数为 64，输出通道数为 128
            nn.BatchNorm2d(128),  # 批量归一化层，归一化通道数为 128 的特征图
            nn.ReLU(True),  # ReLU 激活函数，in-place 模式
            nn.MaxPool2d(2, 2),  # 最大池化层，池化窗口大小为 2x2，步长为 2
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # 第三层卷积：输入通道数为 128，输出通道数为 256
            nn.BatchNorm2d(256),  # 批量归一化层，归一化通道数为 256 的特征图
            nn.ReLU(True),  # ReLU 激活函数，in-place 模式
            nn.MaxPool2d(2, 2),  # 最大池化层，池化窗口大小为 2x2，步长为 2
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),  # 第四层卷积：输入通道数为 256，输出通道数为 512
            nn.BatchNorm2d(512),  # 批量归一化层，归一化通道数为 512 的特征图
            nn.ReLU(True),  # ReLU 激活函数，in-place 模式
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化层，将特征图池化成大小为 1x1
        )

        # 本地化网络的第一个全连接层，输出维度为 256
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256),
                                              nn.ReLU(True))
        # 本地化网络的第二个全连接层，输出维度为 F*2，用于预测 fiducial 点的坐标
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # 初始化本地化网络的第二个全连接层的权重为 0
        self.localization_fc2.weight.data.fill_(0)

        # 初始化本地化网络的第二个全连接层的偏置，参考 RARE 论文中的 Fig. 6 (a)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = (
            torch.from_numpy(initial_bias).float().view(-1))

    # 前向传播函数，接收输入批量图像 batch_I，返回预测的 fiducial 点坐标 batch_C_prime
    def forward(self, batch_I):
        """
        :param batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        :return: batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)  # 获取批量大小
        features = self.conv(batch_I).view(batch_size, -1)  # 将卷积后的特征图展平
        batch_C_prime = self.localization_fc2(
            self.localization_fc1(features)).view(batch_size, self.F, 2)  # 通过两个全连接层预测 fiducial 点的坐标
        return batch_C_prime  # 返回预测的 fiducial 点坐标
class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)

        # for multi-gpu, you need register buffer
        self.register_buffer(
            "inv_delta_C",
            torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # F+3 x F+3
        self.register_buffer("P_hat",
                             torch.tensor(
                                 self._build_P_hat(self.F, self.C, self.P)).float())  # n x F+3

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1),  # 1 x F+3
            ],
            axis=0,
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        """ Return P, a grid of coordinates in I_r """
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        P = np.stack(
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2)
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2
    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # 获取点集 P 的数量，即 n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1),
                         (1, F, 1))  # 将点集 P 沿第二维复制 F 次，形成 P_tile：n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 将控制点集 C 扩展为 1 x F x 2
        P_diff = P_tile - C_tile  # 计算 P_tile 和 C_tile 的差值，形成 P_diff：n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2,
                                  keepdims=False)  # 计算 P_diff 沿第二维度的 L2 范数，得到 rbf_norm：n x F
        rbf = np.multiply(np.square(rbf_norm),
                          np.log(rbf_norm + self.eps))  # 计算 RBF 插值权重，形成 rbf：n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)  # 将偏置项 1、点集 P 和 RBF 插值权重拼接，形成 P_hat：n x F+3
        return P_hat  # 返回 P_hat：n x F+3

    def build_P_prime(self, batch_C_prime):
        """ 从 batch_C_prime [batch_size x F x 2] 生成网格 """
        batch_size = batch_C_prime.size(0)  # 获取 batch 的大小
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)  # 将 inv_delta_C 在第一维度上复制 batch_size 次，形成 batch_inv_delta_C：batch_size x F+3 x 2
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)  # 将 P_hat 在第一维度上复制 batch_size 次，形成 batch_P_hat：batch_size x n x F+3
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
            dim=1)  # 在 batch_C_prime 后面拼接全零张量，形成 batch_C_prime_with_zeros：batch_size x F+3 x 2
        batch_T = torch.bmm(batch_inv_delta_C,
                            batch_C_prime_with_zeros)  # 执行批量矩阵乘法，得到 batch_T：batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # 执行批量矩阵乘法，得到 batch_P_prime：batch_size x n x 2
        return batch_P_prime  # 返回 batch_P_prime：batch_size x n x 2
```