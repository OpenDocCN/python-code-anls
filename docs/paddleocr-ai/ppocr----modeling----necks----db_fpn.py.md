# `.\PaddleOCR\ppocr\modeling\necks\db_fpn.py`

```py
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

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
import os
import sys
from ppocr.modeling.necks.intracl import IntraCLBlock

# 获取当前文件所在目录路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录路径添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上级目录路径添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

# 导入模型结构定义
from ppocr.modeling.backbones.det_mobilenet_v3 import SEModule

# 定义 DSConv 类，继承自 nn.Layer
class DSConv(nn.Layer):
    # 初始化 DSConv 类，设置输入通道数、输出通道数、卷积核大小、填充、步长等参数
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 stride=1,
                 groups=None,
                 if_act=True,
                 act="relu",
                 **kwargs):
        # 调用父类的初始化方法
        super(DSConv, self).__init__()
        # 如果未指定分组数，则将分组数设置为输入通道数
        if groups == None:
            groups = in_channels
        # 设置是否使用激活函数和激活函数类型
        self.if_act = if_act
        self.act = act
        # 创建第一个卷积层，设置输入通道数、输出通道数、卷积核大小、步长、填充和分组数
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        # 创建第一个批归一化层，设置通道数为输入通道数
        self.bn1 = nn.BatchNorm(num_channels=in_channels, act=None)

        # 创建第二个卷积层，设置输入通道数、输出通道数为输入通道数的四倍，卷积核大小为1
        self.conv2 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias_attr=False)

        # 创建第二个批归一化层，设置通道数为输入通道数的四倍
        self.bn2 = nn.BatchNorm(num_channels=int(in_channels * 4), act=None)

        # 创建第三个卷积层，设置输入通道数为输入通道数的四倍，输出通道数为指定的输出通道数，卷积核大小为1
        self.conv3 = nn.Conv2D(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias_attr=False)
        # 保存输入通道数和输出通道数
        self._c = [in_channels, out_channels]
        # 如果输入通道数不等于输出通道数，则创建额外的卷积层，将输入通道数调整为输出通道数
        if in_channels != out_channels:
            self.conv_end = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias_attr=False)
    # 定义前向传播函数，接收输入数据
    def forward(self, inputs):
    
        # 第一层卷积操作
        x = self.conv1(inputs)
        # 第一层批归一化操作
        x = self.bn1(x)

        # 第二层卷积操作
        x = self.conv2(x)
        # 第二层批归一化操作
        x = self.bn2(x)
        
        # 如果需要激活函数
        if self.if_act:
            # 如果激活函数为 relu
            if self.act == "relu":
                x = F.relu(x)
            # 如果激活函数为 hardswish
            elif self.act == "hardswish":
                x = F.hardswish(x)
            # 如果选择了不支持的激活函数，打印错误信息并退出程序
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()

        # 第三层卷积操作
        x = self.conv3(x)
        # 如果输入通道数和输出通道数不相等，进行残差连接
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        # 返回处理后的数据
        return x
# 定义一个继承自 nn.Layer 的 DBFPN 类
class DBFPN(nn.Layer):
    # 前向传播函数，接收输入 x
    def forward(self, x):
        # 将输入 x 拆分为 c2, c3, c4, c5 四个部分
        c2, c3, c4, c5 = x

        # 对 c5 进行卷积操作得到 in5
        in5 = self.in5_conv(c5)
        # 对 c4 进行卷积操作得到 in4
        in4 = self.in4_conv(c4)
        # 对 c3 进行卷积操作得到 in3
        in3 = self.in3_conv(c3)
        # 对 c2 进行卷积操作得到 in2
        in2 = self.in2_conv(c2)

        # 将 in5 上采样并与 in4 相加得到 out4
        out4 = in4 + F.upsample(in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16
        # 将 out4 上采样并与 in3 相加得到 out3
        out3 = in3 + F.upsample(out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8
        # 将 out3 上采样并与 in2 相加得到 out2
        out2 = in2 + F.upsample(out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4

        # 对 in5 进行卷积操作得到 p5
        p5 = self.p5_conv(in5)
        # 对 out4 进行卷积操作得到 p4
        p4 = self.p4_conv(out4)
        # 对 out3 进行卷积操作得到 p3
        p3 = self.p3_conv(out3)
        # 对 out2 进行卷积操作得到 p2
        p2 = self.p2_conv(out2)
        
        # 将 p5, p4, p3, p2 沿着 axis=1 进行拼接
        fuse = paddle.concat([p5, p4, p3, p2], axis=1)

        # 如果使用 ASF 模块，则对 fuse 进行 ASF 操作
        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])

        # 返回融合后的特征图
        return fuse


# 定义一个继承自 nn.Layer 的 RSELayer 类
class RSELayer(nn.Layer):
    # 初始化函数，接收输入通道数、输出通道数、卷积核大小和是否使用 shortcut
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        # 使用 KaimingUniform 初始化权重
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.out_channels = out_channels
        # 定义输入卷积层
        self.in_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        # 定义 SE 模块
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    # 前向传播函数，接收输入 ins
    def forward(self, ins):
        # 对输入 ins 进行卷积操作得到 x
        x = self.in_conv(ins)
        # 如果使用 shortcut，则将 x 与 SE 模块的输出相加
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        # 返回输出结果
        return out


# 定义一个继承自 nn.Layer 的 RSEFPN 类
class RSEFPN(nn.Layer):
    # 初始化函数，定义了 RSEFPN 类的构造方法，接受输入通道数、输出通道数和是否使用 shortcut 作为参数
    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        # 调用父类的构造方法
        super(RSEFPN, self).__init__()
        # 设置输出通道数
        self.out_channels = out_channels
        # 初始化输入通道卷积和输出通道卷积为 LayerList
        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()
        # 初始化是否使用 intracl 为 False
        self.intracl = False
        # 如果 kwargs 中包含 intracl 并且其值为 True
        if 'intracl' in kwargs.keys() and kwargs['intracl'] is True:
            # 将 intracl 设置为 True
            self.intracl = kwargs['intracl']
            # 初始化四个 IntraCLBlock 实例，每个实例的输出通道数为 out_channels 的四分之一，reduce_factor 为 2
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

        # 遍历输入通道列表
        for i in range(len(in_channels)):
            # 向输入通道卷积列表中添加 RSELayer 实例，输入通道数为 in_channels[i]，输出通道数为 out_channels，卷积核大小为 1，是否使用 shortcut
            self.ins_conv.append(
                RSELayer(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    shortcut=shortcut))
            # 向输出通道卷积列表中添加 RSELayer 实例，输入通道数为 out_channels，输出通道数为 out_channels 的四分之一，卷积核大小为 3，是否使用 shortcut
            self.inp_conv.append(
                RSELayer(
                    out_channels,
                    out_channels // 4,
                    kernel_size=3,
                    shortcut=shortcut))
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 将输入 x 拆分为 c2, c3, c4, c5 四个部分
        c2, c3, c4, c5 = x

        # 对 c5 进行卷积操作得到 in5
        in5 = self.ins_conv[3](c5)
        # 对 c4 进行卷积操作得到 in4
        in4 = self.ins_conv[2](c4)
        # 对 c3 进行卷积操作得到 in3
        in3 = self.ins_conv[1](c3)
        # 对 c2 进行卷积操作得到 in2
        in2 = self.ins_conv[0](c2)

        # 将 in5 上采样并与 in4 相加得到 out4
        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16
        # 将 out4 上采样并与 in3 相加得到 out3
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8
        # 将 out3 上采样并与 in2 相加得到 out2
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4

        # 对 in5 进行卷积操作得到 p5
        p5 = self.inp_conv[3](in5)
        # 对 out4 进行卷积操作得到 p4
        p4 = self.inp_conv[2](out4)
        # 对 out3 进行卷积操作得到 p3
        p3 = self.inp_conv[1](out3)
        # 对 out2 进行卷积操作得到 p2
        p2 = self.inp_conv[0](out2)

        # 如果开启了 intracl 标志，则对 p5, p4, p3, p2 分别进行额外的卷积操作
        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        # 将 p5 上采样 8 倍
        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        # 将 p4 上采样 4 倍
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        # 将 p3 上采样 2 倍
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        # 将 p5, p4, p3, p2 沿着通道维度拼接
        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        # 返回拼接后的结果
        return fuse
class LKPAN(nn.Layer):
    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)  # 使用第四个卷积层处理 c5
        in4 = self.ins_conv[2](c4)  # 使用第三个卷积层处理 c4
        in3 = self.ins_conv[1](c3)  # 使用第二个卷积层处理 c3
        in2 = self.ins_conv[0](c2)  # 使用第一个卷积层处理 c2

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 将 in5 上采样并与 in4 相加，得到 out4，1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 将 out4 上采样并与 in3 相加，得到 out3，1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 将 out3 上采样并与 in2 相加，得到 out2，1/4

        f5 = self.inp_conv[3](in5)  # 使用第四个输入卷积层处理 in5
        f4 = self.inp_conv[2](out4)  # 使用第三个输入卷积层处理 out4
        f3 = self.inp_conv[1](out3)  # 使用第二个输入卷积层处理 out3
        f2 = self.inp_conv[0](out2)  # 使用第一个输入卷积层处理 out2

        pan3 = f3 + self.pan_head_conv[0](f2)  # 将 f2 经过第一个 PAN 头卷积层处理并与 f3 相加，得到 pan3
        pan4 = f4 + self.pan_head_conv[1](pan3)  # 将 pan3 经过第二个 PAN 头卷积层处理并与 f4 相加，得到 pan4
        pan5 = f5 + self.pan_head_conv[2](pan4)  # 将 pan4 经过第三个 PAN 头卷积层处理并与 f5 相加，得到 pan5

        p2 = self.pan_lat_conv[0](f2)  # 使用第一个 PAN 横向卷积层处理 f2
        p3 = self.pan_lat_conv[1](pan3)  # 使用第二个 PAN 横向卷积层处理 pan3
        p4 = self.pan_lat_conv[2](pan4)  # 使用第三个 PAN 横向卷积层处理 pan4
        p5 = self.pan_lat_conv[3](pan5)  # 使用第四个 PAN 横向卷积层处理 pan5

        if self.intracl is True:
            p5 = self.incl4(p5)  # 如果 intracl 为真，则使用 incl4 处理 p5
            p4 = self.incl3(p4)  # 如果 intracl 为真，则使用 incl3 处理 p4
            p3 = self.incl2(p3)  # 如果 intracl 为真，则使用 incl2 处理 p3
            p2 = self.incl1(p2)  # 如果 intracl 为真，则使用 incl1 处理 p2

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)  # 将 p5 上采样，比例为 8，模式为 nearest
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)  # 将 p4 上采样，比例为 4，模式为 nearest
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)  # 将 p3 上采样，比例为 2，模式为 nearest

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)  # 沿着 axis=1 连接 p5, p4, p3, p2
        return fuse


class ASFBlock(nn.Layer):
    """
    This code is refered from:
        https://github.com/MhLiao/DB/blob/master/decoders/feature_attention.py
    """
    # 定义 Adaptive Scale Fusion (ASF) block 的类，继承自 nn.Layer
    def __init__(self, in_channels, inter_channels, out_features_num=4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: 输入数据的通道数
            inter_channels: 中间通道数
            out_features_num: 融合阶段的数量，默认为4
        """
        # 调用父类的构造函数
        super(ASFBlock, self).__init__()
        # 使用 KaimingUniform 初始化器初始化权重
        weight_attr = paddle.nn.initializer.KaimingUniform()
        # 设置输入通道数、中间通道数和融合阶段数量
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        # 创建一个卷积层，输入通道数为 in_channels，输出通道数为 inter_channels，卷积核大小为 3x3，填充为1
        self.conv = nn.Conv2D(in_channels, inter_channels, 3, padding=1)

        # 空间尺度变换模块
        self.spatial_scale = nn.Sequential(
            #Nx1xHxW
            # 创建一个卷积层，输入通道数为1，输出通道数为1，卷积核大小为 3x3，无偏置，填充为1，权重初始化为 weight_attr
            nn.Conv2D(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                bias_attr=False,
                padding=1,
                weight_attr=ParamAttr(initializer=weight_attr)),
            # ReLU 激活函数
            nn.ReLU(),
            # 创建一个卷积层，输入通道数为1，输出通道数为1，卷积核大小为 1x1，无偏置，权重初始化为 weight_attr
            nn.Conv2D(
                in_channels=1,
                out_channels=1,
                kernel_size=1,
                bias_attr=False,
                weight_attr=ParamAttr(initializer=weight_attr)),
            # Sigmoid 激活函数
            nn.Sigmoid())

        # 通道尺度变换模块
        self.channel_scale = nn.Sequential(
            # 创建一个卷积层，输入通道数为 inter_channels，输出通道数为 out_features_num，卷积核大小为 1x1，无偏置，权重初始化为 weight_attr
            nn.Conv2D(
                in_channels=inter_channels,
                out_channels=out_features_num,
                kernel_size=1,
                bias_attr=False,
                weight_attr=ParamAttr(initializer=weight_attr)),
            # Sigmoid 激活函数
            nn.Sigmoid())
    # 前向传播函数，接收融合特征和特征列表作为输入
    def forward(self, fuse_features, features_list):
        # 使用卷积层处理融合特征
        fuse_features = self.conv(fuse_features)
        # 在第1个维度上对融合特征进行平均，保持维度
        spatial_x = paddle.mean(fuse_features, axis=1, keepdim=True)
        # 计算空间尺度
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        # 计算通道尺度
        attention_scores = self.channel_scale(attention_scores)
        # 断言特征列表的长度等于输出特征数量
        assert len(features_list) == self.out_features_num

        # 初始化输出列表
        out_list = []
        # 遍历输出特征数量
        for i in range(self.out_features_num):
            # 将每个输出特征乘以对应的注意力分数，并添加到输出列表中
            out_list.append(attention_scores[:, i:i + 1] * features_list[i])
        # 沿着第1个维度拼接输出列表中的数据
        return paddle.concat(out_list, axis=1)
```