# `.\PaddleOCR\ppocr\modeling\necks\rf_adaptor.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
这段代码参考自：
https://github.com/hikopensource/DAVAR-Lab-OCR/blob/main/davarocr/davar_rcg/models/connects/single_block/RFAdaptor.py
"""

import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal, KaimingNormal

kaiming_init_ = KaimingNormal()
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


class S2VAdaptor(nn.Layer):
    """ 语义到视觉适配模块"""

    def __init__(self, in_channels=512):
        super(S2VAdaptor, self).__init__()

        self.in_channels = in_channels  # 输入通道数为 512

        # 特征增强模块，通道注意力
        self.channel_inter = nn.Linear(
            self.in_channels, self.in_channels, bias_attr=False)  # 线性变换层
        self.channel_bn = nn.BatchNorm1D(self.in_channels)  # 一维批归一化层
        self.channel_act = nn.ReLU()  # 激活函数为 ReLU
        self.apply(self.init_weights)  # 初始化权重

    def init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            kaiming_init_(m.weight)  # 使用 Kaiming 初始化卷积层权重
            if isinstance(m, nn.Conv2D) and m.bias is not None:
                zeros_(m.bias)  # 初始化偏置为 0
        elif isinstance(m, (nn.BatchNorm, nn.BatchNorm2D, nn.BatchNorm1D)):
            zeros_(m.bias)  # 初始化批归一化层偏置为 0
            ones_(m.weight)  # 初始化批归一化层权重为 1
    # 前向传播函数，接收语义信息作为输入
    def forward(self, semantic):
        # 复制一份输入的语义信息
        semantic_source = semantic  # batch, channel, height, width

        # 特征变换
        # 压缩掉高度维度，然后转置，得到 batch, width, channel 的形状
        semantic = semantic.squeeze(2).transpose([0, 2, 1])  
        # 对转置后的语义信息进行通道交互操作，得到 batch, width, channel 的形状
        channel_att = self.channel_inter(semantic)  
        # 再次转置，得到 batch, channel, width 的形状
        channel_att = channel_att.transpose([0, 2, 1])  
        # 对通道交互后的结果进行通道归一化操作，得到 batch, channel, width 的形状
        channel_bn = self.channel_bn(channel_att)  
        # 对通道归一化后的结果进行激活函数操作，得到 batch, channel, width 的形状
        channel_att = self.channel_act(channel_bn)  

        # 特征增强
        # 将原始语义信息与通道注意力权重相乘，增强特征，得到 batch, channel, 1, width 的形状
        channel_output = semantic_source * channel_att.unsqueeze(-2)  

        # 返回增强后的特征
        return channel_output
class V2SAdaptor(nn.Layer):
    """ Visual to Semantic adaptation module"""

    def __init__(self, in_channels=512, return_mask=False):
        # 初始化 V2SAdaptor 类
        super(V2SAdaptor, self).__init__()

        # 参数初始化
        self.in_channels = in_channels
        self.return_mask = return_mask

        # 输出转换
        self.channel_inter = nn.Linear(
            self.in_channels, self.in_channels, bias_attr=False)
        self.channel_bn = nn.BatchNorm1D(self.in_channels)
        self.channel_act = nn.ReLU()

    def forward(self, visual):
        # 特征增强
        visual = visual.squeeze(2).transpose([0, 2, 1])  # 将维度为2的维度压缩，然后转置
        channel_att = self.channel_inter(visual)  # 通过线性变换增强特征
        channel_att = channel_att.transpose([0, 2, 1])  # 再次转置
        channel_bn = self.channel_bn(channel_att)  # 批量归一化
        channel_att = self.channel_act(channel_bn)  # 激活函数

        # 尺寸对齐
        channel_output = channel_att.unsqueeze(-2)  # 在倒数第二个维度上增加一个维度

        if self.return_mask:
            return channel_output, channel_att
        return channel_output


class RFAdaptor(nn.Layer):
    def __init__(self, in_channels=512, use_v2s=True, use_s2v=True, **kwargs):
        # 初始化 RFAdaptor 类
        super(RFAdaptor, self).__init__()
        if use_v2s is True:
            self.neck_v2s = V2SAdaptor(in_channels=in_channels, **kwargs)
        else:
            self.neck_v2s = None
        if use_s2v is True:
            self.neck_s2v = S2VAdaptor(in_channels=in_channels, **kwargs)
        else:
            self.neck_s2v = None
        self.out_channels = in_channels
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 将输入 x 拆分为视觉特征和识别特征
        visual_feature, rcg_feature = x
        # 如果视觉特征不为空
        if visual_feature is not None:
            # 获取视觉特征的形状信息
            batch, source_channels, v_source_height, v_source_width = visual_feature.shape
            # 重塑视觉特征的形状
            visual_feature = visual_feature.reshape(
                [batch, source_channels, 1, v_source_height * v_source_width])

        # 如果存在从视觉到识别的连接
        if self.neck_v2s is not None:
            # 计算视觉到识别的特征
            v_rcg_feature = rcg_feature * self.neck_v2s(visual_feature)
        else:
            # 否则直接使用识别特征
            v_rcg_feature = rcg_feature

        # 如果存在从识别到视觉的连接
        if self.neck_s2v is not None:
            # 计算识别到视觉的特征
            v_visual_feature = visual_feature + self.neck_s2v(rcg_feature)
        else:
            # 否则直接使用视觉特征
            v_visual_feature = visual_feature
        
        # 如果视觉到识别的特征不为空
        if v_rcg_feature is not None:
            # 获取视觉到识别的特征的形状信息
            batch, source_channels, source_height, source_width = v_rcg_feature.shape
            # 重塑视觉到识别的特征的形状
            v_rcg_feature = v_rcg_feature.reshape(
                [batch, source_channels, 1, source_height * source_width])
            # 去除多余的维度并转置
            v_rcg_feature = v_rcg_feature.squeeze(2).transpose([0, 2, 1])
        
        # 返回处理后的视觉特征和识别特征
        return v_visual_feature, v_rcg_feature
```