# `.\PaddleOCR\ppocr\modeling\heads\det_db_head.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer

# 定义一个函数，用于获取偏置属性
def get_bias_attr(k):
    # 计算标准差
    stdv = 1.0 / math.sqrt(k * 1.0)
    # 初始化器为均匀分布
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    # 设置偏置属性
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr

# 定义一个类 Head，继承自 nn.Layer
class Head(nn.Layer):
    # 初始化函数，定义了一个头部网络模块，包含卷积层和批归一化层
    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        # 调用父类的初始化函数
        super(Head, self).__init__()

        # 第一个卷积层，输入通道数为in_channels，输出通道数为in_channels // 4，卷积核大小为kernel_list[0]
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            weight_attr=ParamAttr(),
            bias_attr=False)
        
        # 第一个批归一化层，输入通道数为in_channels // 4
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act='relu')

        # 第二个反卷积层，输入通道数为in_channels // 4，输出通道数为in_channels // 4，卷积核大小为kernel_list[1]，步长为2
        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        
        # 第二个批归一化层，输入通道数为in_channels // 4
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu")
        
        # 第三个反卷积层，输入通道数为in_channels // 4，输出通道数为1，卷积核大小为kernel_list[2]，步长为2
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4), )
    # 定义前向传播函数，接受输入 x 和是否返回中间特征的标志 return_f
    def forward(self, x, return_f=False):
        # 使用第一个卷积层对输入 x 进行卷积操作
        x = self.conv1(x)
        # 对卷积结果进行批归一化操作
        x = self.conv_bn1(x)
        # 使用第二个卷积层对结果进行卷积操作
        x = self.conv2(x)
        # 对卷积结果进行批归一化操作
        x = self.conv_bn2(x)
        # 如果需要返回中间特征，则将当前特征保存到 f 中
        if return_f is True:
            f = x
        # 使用第三个卷积层对结果进行卷积操作
        x = self.conv3(x)
        # 对卷积结果进行 sigmoid 激活函数操作
        x = F.sigmoid(x)
        # 如果需要返回中间特征，则返回当前结果和中间特征
        if return_f is True:
            return x, f
        # 否则只返回当前结果
        return x
class DBHead(nn.Layer):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        # 初始化 DBHead 类
        super(DBHead, self).__init__()
        # 设置超参数 k
        self.k = k
        # 创建 Head 对象用于二值化
        self.binarize = Head(in_channels, **kwargs)
        # 创建 Head 对象用于阈值
        self.thresh = Head(in_channels, **kwargs)

    def step_function(self, x, y):
        # 定义阶跃函数
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        # 获取收缩图
        shrink_maps = self.binarize(x)
        # 如果不是训练阶段，直接返回收缩图
        if not self.training:
            return {'maps': shrink_maps}

        # 获取阈值图
        threshold_maps = self.thresh(x)
        # 通过阶跃函数得到二值图
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        # 拼接收缩图、阈值图和二值图
        y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
        return {'maps': y}


class LocalModule(nn.Layer):
    def __init__(self, in_c, mid_c, use_distance=True):
        # 初始化 LocalModule 类
        super(self.__class__, self).__init__()
        # 创建 3x3 卷积层和 BN 层
        self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act='relu')
        # 创建 1x1 卷积层
        self.last_1 = nn.Conv2D(mid_c, 1, 1, 1, 0)

    def forward(self, x, init_map, distance_map):
        # 拼接输入特征图和初始化图
        outf = paddle.concat([init_map, x], axis=1)
        # 最后的卷积操作
        out = self.last_1(self.last_3(outf))
        return out


class PFHeadLocal(DBHead):
    def __init__(self, in_channels, k=50, mode='small', **kwargs):
        # 初始化 PFHeadLocal 类
        super(PFHeadLocal, self).__init__(in_channels, k, **kwargs)
        # 设置模式
        self.mode = mode

        # 上采样层
        self.up_conv = nn.Upsample(scale_factor=2, mode="nearest", align_mode=1)
        # 根据模式选择不同的 LocalModule
        if self.mode == 'large':
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 4)
        elif self.mode == 'small':
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 8)
    # 前向传播函数，接受输入 x 和目标 targets
    def forward(self, x, targets=None):
        # 对输入 x 进行二值化处理，同时返回缩小后的特征图和特征图
        shrink_maps, f = self.binarize(x, return_f=True)
        # 将缩小后的特征图作为基础特征图
        base_maps = shrink_maps
        # 使用上采样后的特征图和缩小后的特征图作为输入，进行条件二值化
        cbn_maps = self.cbn_layer(self.up_conv(f), shrink_maps, None)
        # 对条件二值化后的特征图进行 sigmoid 激活函数处理
        cbn_maps = F.sigmoid(cbn_maps)
        # 如果不是训练阶段，则返回基础特征图和条件二值化后的特征图的平均值
        if not self.training:
            return {'maps': 0.5 * (base_maps + cbn_maps), 'cbn_maps': cbn_maps}

        # 使用阈值网络对输入 x 进行阈值处理
        threshold_maps = self.thresh(x)
        # 使用阈值特征图和缩小后的特征图进行阈值函数处理，得到二值化特征图
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        # 将条件二值化特征图、阈值特征图和二值化特征图在通道维度上拼接
        y = paddle.concat([cbn_maps, threshold_maps, binary_maps], axis=1)
        # 返回结果字典，包括拼接后的特征图、条件二值化特征图和二值化特征图
        return {'maps': y, 'distance_maps': cbn_maps, 'cbn_maps': binary_maps}
```