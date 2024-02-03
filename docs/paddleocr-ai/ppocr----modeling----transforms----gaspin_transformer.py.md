# `.\PaddleOCR\ppocr\modeling\transforms\gaspin_transformer.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np
import functools
from .tps import GridGenerator

'''This code is refer from:
https://github.com/hikopensource/DAVAR-Lab-OCR/davarocr/davar_rcg/models/transformations/gaspin_transformation.py
'''
# 导入必要的库和模块

class SP_TransformerNetwork(nn.Layer):
    """
    Sturture-Preserving Transformation (SPT) as Equa. (2) in Ref. [1]
    Ref: [1] SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition. AAAI-2021.
    """
    # 定义 SP_TransformerNetwork 类，实现结构保持变换

    def __init__(self, nc=1, default_type=5):
        """ Based on SPIN
        Args:
            nc (int): number of input channels (usually in 1 or 3)
            default_type (int): the complexity of transformation intensities (by default set to 6 as the paper)
        """
        # 初始化函数，接受输入通道数和默认变换复杂度
        super(SP_TransformerNetwork, self).__init__()
        # 调用父类构造函数
        self.power_list = self.cal_K(default_type)
        # 计算变换强度的列表
        self.sigmoid = nn.Sigmoid()
        # 创建 Sigmoid 激活函数
        self.bn = nn.InstanceNorm2D(nc)
        # 创建 2D 实例归一化层
    def cal_K(self, k=5):
        """

        Args:
            k (int): the complexity of transformation intensities (by default set to 6 as the paper)

        Returns:
            List: the normalized intensity of each pixel in [0,1], denoted as \beta [1x(2K+1)]

        """
        # 导入 math 模块中的 log 函数
        from math import log
        # 初始化空列表 x
        x = []
        # 如果 k 不等于 0
        if k != 0:
            # 遍历范围为 [1, k] 的整数
            for i in range(1, k+1):
                # 计算下限值
                lower = round(log(1-(0.5/(k+1))*i)/log((0.5/(k+1))*i), 2)
                # 计算上限值
                upper = round(1/lower, 2)
                # 将下限值和上限值添加到列表 x 中
                x.append(lower)
                x.append(upper)
        # 添加 1.00 到列表 x 中
        x.append(1.00)
        # 返回列表 x
        return x

    def forward(self, batch_I, weights, offsets, lambda_color=None):
        """

        Args:
            batch_I (Tensor): batch of input images [batch_size x nc x I_height x I_width]
            weights:
            offsets: the predicted offset by AIN, a scalar
            lambda_color: the learnable update gate \alpha in Equa. (5) as
                          g(x) = (1 - \alpha) \odot x + \alpha \odot x_{offsets}

        Returns:
            Tensor: transformed images by SPN as Equa. (4) in Ref. [1]
                        [batch_size x I_channel_num x I_r_height x I_r_width]

        """
        # 将 batch_I 归一化到 [0,1] 范围
        batch_I = (batch_I + 1) * 0.5
        # 如果 offsets 不为 None
        if offsets is not None:
            # 根据 lambda_color 对 batch_I 进行线性插值
            batch_I = batch_I*(1-lambda_color) + offsets*lambda_color
        # 将 weights 转换为 Tensor，并在最后两个维度上添加维度
        batch_weight_params = paddle.unsqueeze(paddle.unsqueeze(weights, -1), -1)
        # 计算 batch_I 的各次幂，并在通道维度上堆叠
        batch_I_power = paddle.stack([batch_I.pow(p) for p in self.power_list], axis=1)

        # 计算加权和
        batch_weight_sum = paddle.sum(batch_I_power * batch_weight_params, axis=1)
        # 对加权和进行批归一化
        batch_weight_sum = self.bn(batch_weight_sum)
        # 对加权和进行 sigmoid 激活
        batch_weight_sum = self.sigmoid(batch_weight_sum)
        # 将加权和映射到 [-1,1] 范围
        batch_weight_sum = batch_weight_sum * 2 - 1
        # 返回处理后的图像
        return batch_weight_sum
class GA_SPIN_Transformer(nn.Layer):
    """
    Geometric-Absorbed SPIN Transformation (GA-SPIN) proposed in Ref. [1]


    Ref: [1] SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition. AAAI-2021.
    """

    def init_spin(self, nz):
        """
        初始化 SPIN 变换

        Args:
            nz (int): paired \betas 指数的数量，即 K x 2 的值

        """
        # 初始化 ID 列表，包含 nz 个 0.00 和一个 5.00
        init_id = [0.00]*nz+[5.00]
        # 如果存在偏移量
        if self.offsets:
            # 添加一个 -5.00 到初始化 ID 列表
            init_id += [-5.00]
            # 将 init_id 重复 3 次
            # init_id *=3
        # 将初始化 ID 列表转换为 numpy 数组
        init = np.array(init_id)

        # 如果存在空间变换网络
        if self.stn:
            # 获取 F 的值
            F = self.F
            # 在 x 轴上均匀分布控制点
            ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
            # 在顶部 y 轴上均匀分布控制点
            ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
            # 在底部 y 轴上均匀分布控制点
            ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
            # 组合顶部控制点坐标
            ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
            # 组合底部控制点坐标
            ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
            # 将顶部和底部控制点连接起来
            initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
            # 将 initial_bias 转换为一维数组
            initial_bias = initial_bias.reshape(-1)
            # 将 initial_bias 和 init 连接起来
            init = np.concatenate([init, initial_bias], axis=0)
        # 返回初始化结果
        return init
```