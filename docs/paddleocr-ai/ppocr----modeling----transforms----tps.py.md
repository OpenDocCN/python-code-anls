# `.\PaddleOCR\ppocr\modeling\transforms\tps.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用来源
# https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/transformation.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np

# 定义 ConvBNLayer 类，包含卷积和批归一化操作
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        # 创建卷积层
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        bn_name = "bn_" + name
        # 创建批归一化层
        self.bn = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
    # 定义一个前向传播函数，接收输入 x
    def forward(self, x):
        # 对输入 x 进行卷积操作
        x = self.conv(x)
        # 对卷积结果 x 进行批量归一化操作
        x = self.bn(x)
        # 返回处理后的结果 x
        return x
class LocalizationNetwork(nn.Layer):
    def forward(self, x):
        """
           估计几何变换参数
           参数:
               image: 输入
           返回:
               batch_C_prime: 几何变换的矩阵
        """
        B = x.shape[0]  # 获取输入张量的批量大小
        i = 0  # 初始化计数器
        for block in self.block_list:  # 遍历网络中的每个块
            x = block(x)  # 对输入张量进行块的前向传播
        x = x.squeeze(axis=2).squeeze(axis=2)  # 压缩张量的维度
        x = self.fc1(x)  # 使用全连接层进行计算

        x = F.relu(x)  # 使用ReLU激活函数
        x = self.fc2(x)  # 使用另一个全连接层进行计算
        x = x.reshape(shape=[-1, self.F, 2])  # 重新调整张量的形状
        return x  # 返回结果张量

    def get_initial_fiducials(self):
        """ 查看 RARE 论文中的图 6 (a) """
        F = self.F  # 获取特征点的数量
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))  # 在指定范围内生成均匀间隔的数字序列
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))  # 在指定范围内生成均匀间隔的数字序列
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))  # 在指定范围内生成均匀间隔的数字序列
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)  # 沿指定轴堆叠数组
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)  # 沿指定轴堆叠数组
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)  # 沿指定轴连接数组
        return initial_bias  # 返回初始偏置数组


class GridGenerator(nn.Layer):
    def __init__(self, in_channels, num_fiducial):
        super(GridGenerator, self).__init__()  # 调用父类的构造函数
        self.eps = 1e-6  # 设置一个小的常数
        self.F = num_fiducial  # 设置特征点的数量

        name = "ex_fc"  # 设置名称
        initializer = nn.initializer.Constant(value=0.0)  # 使用常数初始化器
        param_attr = ParamAttr(
            learning_rate=0.0, initializer=initializer, name=name + "_w")  # 设置参数属性
        bias_attr = ParamAttr(
            learning_rate=0.0, initializer=initializer, name=name + "_b")  # 设置偏置属性
        self.fc = nn.Linear(
            in_channels,
            6,
            weight_attr=param_attr,
            bias_attr=bias_attr,
            name=name)  # 创建线性层
    def forward(self, batch_C_prime, I_r_size):
        """
        生成用于 grid_sampler 的网格。
        Args:
            batch_C_prime: 几何变换的矩阵
            I_r_size: 输入图像的形状
        Return:
            batch_P_prime: grid_sampler 的网格
        """
        # 构建 C 矩阵
        C = self.build_C_paddle()
        # 构建 P 矩阵
        P = self.build_P_paddle(I_r_size)

        # 构建 inv_delta_C 张量
        inv_delta_C_tensor = self.build_inv_delta_C_paddle(C).astype('float32')
        # 构建 P_hat 张量
        P_hat_tensor = self.build_P_hat_paddle(
            C, paddle.to_tensor(P)).astype('float32')

        # 设置不计算梯度
        inv_delta_C_tensor.stop_gradient = True
        P_hat_tensor.stop_gradient = True

        # 获取扩展的 batch_C_prime 张量
        batch_C_ex_part_tensor = self.get_expand_tensor(batch_C_prime)

        batch_C_ex_part_tensor.stop_gradient = True

        # 在 axis=1 上连接 batch_C_prime 和 batch_C_ex_part_tensor，得到 batch_C_prime_with_zeros
        batch_C_prime_with_zeros = paddle.concat(
            [batch_C_prime, batch_C_ex_part_tensor], axis=1)
        # 计算 batch_T
        batch_T = paddle.matmul(inv_delta_C_tensor, batch_C_prime_with_zeros)
        # 计算 batch_P_prime
        batch_P_prime = paddle.matmul(P_hat_tensor, batch_T)
        return batch_P_prime

    def build_C_paddle(self):
        """ 返回 I_r 中基准点的坐标; C """
        F = self.F
        # 在 -1.0 到 1.0 之间生成 F/2 个均匀间隔的值，作为控制点的 x 坐标
        ctrl_pts_x = paddle.linspace(-1.0, 1.0, int(F / 2), dtype='float64')
        # 创建 F/2 个值为 -1 的张量，作为控制点的 y 坐标（顶部）
        ctrl_pts_y_top = -1 * paddle.ones([int(F / 2)], dtype='float64')
        # 创建 F/2 个值为 1 的张量，作为控制点的 y 坐标（底部）
        ctrl_pts_y_bottom = paddle.ones([int(F / 2)], dtype='float64')
        # 将控制点的 x 和 y 坐标堆叠在一起，得到控制点的坐标
        ctrl_pts_top = paddle.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = paddle.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = paddle.concat([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2
    # 构建 P 矩阵，用于存储网格点的坐标信息
    def build_P_paddle(self, I_r_size):
        I_r_height, I_r_width = I_r_size
        # 生成 x 轴方向的网格点坐标
        I_r_grid_x = (paddle.arange(
            -I_r_width, I_r_width, 2, dtype='float64') + 1.0
                      ) / paddle.to_tensor(np.array([I_r_width]))

        # 生成 y 轴方向的网格点坐标
        I_r_grid_y = (paddle.arange(
            -I_r_height, I_r_height, 2, dtype='float64') + 1.0
                      ) / paddle.to_tensor(np.array([I_r_height]))

        # 构建 P 矩阵，存储网格点的坐标信息
        P = paddle.stack(paddle.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
        P = paddle.transpose(P, perm=[1, 0, 2])
        # 将 P 矩阵 reshape 成 n x 2 的形状
        return P.reshape([-1, 2])

    # 构建 inv_delta_C 矩阵，用于计算 T
    def build_inv_delta_C_paddle(self, C):
        """ Return inv_delta_C which is needed to calculate T """
        F = self.F
        # 构建单位矩阵 hat_eye
        hat_eye = paddle.eye(F, dtype='float64')  # F x F
        # 计算 hat_C 矩阵
        hat_C = paddle.norm(
            C.reshape([1, F, 2]) - C.reshape([F, 1, 2]), axis=2) + hat_eye
        hat_C = (hat_C**2) * paddle.log(hat_C)
        # 构建 delta_C 矩阵
        delta_C = paddle.concat(  # F+3 x F+3
            [
                paddle.concat(
                    [paddle.ones(
                        (F, 1), dtype='float64'), C, hat_C], axis=1),  # F x F+3
                paddle.concat(
                    [
                        paddle.zeros(
                            (2, 3), dtype='float64'), paddle.transpose(
                                C, perm=[1, 0])
                    ],
                    axis=1),  # 2 x F+3
                paddle.concat(
                    [
                        paddle.zeros(
                            (1, 3), dtype='float64'), paddle.ones(
                                (1, F), dtype='float64')
                    ],
                    axis=1)  # 1 x F+3
            ],
            axis=0)
        # 计算 inv_delta_C 矩阵
        inv_delta_C = paddle.inverse(delta_C)
        return inv_delta_C  # F+3 x F+3
    # 构建 P_hat 矩阵，用于存储特征点 P 到中心点 C 的距离信息
    def build_P_hat_paddle(self, C, P):
        # 获取特征点的数量
        F = self.F
        # 设置一个很小的值，用于防止除零错误
        eps = self.eps
        # 获取特征点的数量
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        # 将特征点 P 进行复制，形成 n x F x 2 的矩阵
        P_tile = paddle.tile(paddle.unsqueeze(P, axis=1), (1, F, 1))
        # 将中心点 C 进行复制，形成 1 x F x 2 的矩阵
        C_tile = paddle.unsqueeze(C, axis=0)  # 1 x F x 2
        # 计算特征点 P 与中心点 C 之间的差值
        P_diff = P_tile - C_tile  # n x F x 2
        # 计算特征点 P 与中心点 C 之间的欧氏距离
        rbf_norm = paddle.norm(P_diff, p=2, axis=2, keepdim=False)

        # 计算径向基函数的值
        rbf = paddle.multiply(
            paddle.square(rbf_norm), paddle.log(rbf_norm + eps))
        # 将特征点 P、径向基函数值和一个全为 1 的列向量拼接在一起，形成 P_hat 矩阵
        P_hat = paddle.concat(
            [paddle.ones(
                (n, 1), dtype='float64'), P, rbf], axis=1)
        return P_hat  # n x F+3

    # 获取扩展的张量，用于存储中心点的扩展信息
    def get_expand_tensor(self, batch_C_prime):
        # 获取批量数据的维度信息
        B, H, C = batch_C_prime.shape
        # 将批量数据进行形状重塑，变为 B x (H*C) 的矩阵
        batch_C_prime = batch_C_prime.reshape([B, H * C])
        # 将重塑后的数据通过全连接层进行处理
        batch_C_ex_part_tensor = self.fc(batch_C_prime)
        # 将处理后的数据再次进行形状重塑，变为 B x 3 x 2 的矩阵
        batch_C_ex_part_tensor = batch_C_ex_part_tensor.reshape([-1, 3, 2])
        return batch_C_ex_part_tensor
# 定义 TPS 类，继承自 nn.Layer 类
class TPS(nn.Layer):
    # 初始化函数，接受输入通道数、关键点数量、定位网络学习率和模型名称作为参数
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(TPS, self).__init__()
        # 初始化定位网络
        self.loc_net = LocalizationNetwork(in_channels, num_fiducial, loc_lr, model_name)
        # 初始化网格生成器
        self.grid_generator = GridGenerator(self.loc_net.out_channels, num_fiducial)
        # 输出通道数等于输入通道数
        self.out_channels = in_channels

    # 前向传播函数，接受图像作为输入
    def forward(self, image):
        # 设置图像的梯度为可计算
        image.stop_gradient = False
        # 通过定位网络获取 batch_C_prime
        batch_C_prime = self.loc_net(image)
        # 通过网格生成器生成 batch_P_prime
        batch_P_prime = self.grid_generator(batch_C_prime, image.shape[2:])
        # 重塑 batch_P_prime 的形状
        batch_P_prime = batch_P_prime.reshape([-1, image.shape[2], image.shape[3], 2])
        # 初始化是否为 fp16 类型的标志
        is_fp16 = False
        # 如果 batch_P_prime 的数据类型不是 float32
        if batch_P_prime.dtype != paddle.float32:
            # 保存原始数据类型
            data_type = batch_P_prime.dtype
            # 将图像和 batch_P_prime 转换为 float32 类型
            image = image.cast(paddle.float32)
            batch_P_prime = batch_P_prime.cast(paddle.float32)
            # 设置为 fp16 类型
            is_fp16 = True
        # 通过 F.grid_sample 函数对图像进行采样，得到 batch_I_r
        batch_I_r = F.grid_sample(x=image, grid=batch_P_prime)
        # 如果是 fp16 类型，则将 batch_I_r 转换为原始数据类型
        if is_fp16:
            batch_I_r = batch_I_r.cast(data_type)
        
        # 返回 batch_I_r
        return batch_I_r
```