# `.\PaddleOCR\ppocr\modeling\heads\det_sast_head.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何明示或暗示的保证或条件，无论是明示的还是暗示的
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

# 定义一个卷积和批归一化层的类
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        # 创建卷积层
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        # 创建批归一化层
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name="bn_" + name + "_scale"),
            bias_attr=ParamAttr(name="bn_" + name + "_offset"),
            moving_mean_name="bn_" + name + "_mean",
            moving_variance_name="bn_" + name + "_variance")

    # 前向传播函数
    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        # 批归一化操作
        x = self.bn(x)
        return x

# 定义 SAST_Header1 类
class SAST_Header1(nn.Layer):
    # 初始化函数，接受输入通道数和其他参数
    def __init__(self, in_channels, **kwargs):
        # 调用父类的初始化函数
        super(SAST_Header1, self).__init__()
        # 定义输出通道数列表
        out_channels = [64, 64, 128]
        # 定义得分分支的卷积层序列
        self.score_conv = nn.Sequential(
            # 第一个卷积层
            ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_score1'),
            # 第二个卷积层
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_score2'),
            # 第三个卷积层
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_score3'),
            # 第四个卷积层
            ConvBNLayer(out_channels[2], 1, 3, 1, act=None, name='f_score4')
        )
        # 定义边界分支的卷积层序列
        self.border_conv = nn.Sequential(
            # 第一个卷积层
            ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_border1'),
            # 第二个卷积层
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_border2'),
            # 第三个卷积层
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_border3'),
            # 第四个卷积层
            ConvBNLayer(out_channels[2], 4, 3, 1, act=None, name='f_border4')            
        )

    # 前向传播函数，接受输入数据 x
    def forward(self, x):
        # 得分分支的前向传播
        f_score = self.score_conv(x)
        # 对得分进行 sigmoid 激活
        f_score = F.sigmoid(f_score)
        # 边界分支的前向传播
        f_border = self.border_conv(x)
        # 返回得分和边界结果
        return f_score, f_border
class SAST_Header2(nn.Layer):
    # 定义 SAST 模型的第二个头部
    def __init__(self, in_channels, **kwargs):
        # 初始化函数
        super(SAST_Header2, self).__init__()
        out_channels = [64, 64, 128]
        # 定义输出通道数
        self.tvo_conv = nn.Sequential(
            # 定义 tvo_conv 为包含多个卷积层的序列
            ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_tvo1'),
            # 第一个卷积层
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_tvo2'),
            # 第二个卷积层
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_tvo3'),
            # 第三个卷积层
            ConvBNLayer(out_channels[2], 8, 3, 1, act=None, name='f_tvo4')
            # 第四个卷积层
        )
        self.tco_conv = nn.Sequential(
            # 定义 tco_conv 为包含多个卷积层的序列
            ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_tco1'),
            # 第一个卷积层
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_tco2'),
            # 第二个卷积层
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_tco3'),
            # 第三个卷积层
            ConvBNLayer(out_channels[2], 2, 3, 1, act=None, name='f_tco4')            
            # 第四个卷积层
        )

    def forward(self, x):
        # 前向传播函数
        f_tvo = self.tvo_conv(x)
        f_tco = self.tco_conv(x)
        return f_tvo, f_tco


class SASTHead(nn.Layer):
    """
    """
    # 定义 SAST 模型的头部
    def __init__(self, in_channels, **kwargs):
        # 初始化函数
        super(SASTHead, self).__init__()

        self.head1 = SAST_Header1(in_channels)
        # 创建 SAST_Header1 实例
        self.head2 = SAST_Header2(in_channels)
        # 创建 SAST_Header2 实例

    def forward(self, x, targets=None):
        # 前向传播函数
        f_score, f_border = self.head1(x)
        # 获取 SAST_Header1 的输出
        f_tvo, f_tco = self.head2(x)
        # 获取 SAST_Header2 的输出

        predicts = {}
        predicts['f_score'] = f_score
        predicts['f_border'] = f_border
        predicts['f_tvo'] = f_tvo
        predicts['f_tco'] = f_tco
        return predicts
```