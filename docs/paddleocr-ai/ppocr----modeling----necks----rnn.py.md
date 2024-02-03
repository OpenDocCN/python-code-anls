# `.\PaddleOCR\ppocr\modeling\necks\rnn.py`

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

# 导入自定义模块
from ppocr.modeling.heads.rec_ctc_head import get_para_bias_attr
from ppocr.modeling.backbones.rec_svtrnet import Block, ConvBNLayer, trunc_normal_, zeros_, ones_

# 定义一个类，用于将图像转换为序列
class Im2Seq(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        # 获取输入张量的形状信息
        B, C, H, W = x.shape
        # 断言输入张量的高度为1
        assert H == 1
        # 压缩输入张量的高度维度
        x = x.squeeze(axis=2)
        # 转置输入张量，将通道维度放在最后
        x = x.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        return x

# 定义一个带有 RNN 的编码器类
class EncoderWithRNN(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        # 创建一个双向 LSTM 层
        self.lstm = nn.LSTM(
            in_channels, hidden_size, direction='bidirectional', num_layers=2)

    def forward(self, x):
        # 将输入张量传入 LSTM 层
        x, _ = self.lstm(x)
        return x

# 定义一个双向 LSTM 类
class BidirectionalLSTM(nn.Layer):
    # 定义双向 LSTM 模型类，继承自 nn.Module
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 num_layers=1,
                 dropout=0,
                 direction=False,
                 time_major=False,
                 with_linear=False):
        # 调用父类的初始化方法
        super(BidirectionalLSTM, self).__init__()
        # 设置是否包含线性层的标志
        self.with_linear = with_linear
        # 创建 LSTM 层
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            direction=direction,
            time_major=time_major)
    
        # 如果包含线性层，则创建线性层
        if self.with_linear:
            self.linear = nn.Linear(hidden_size * 2, output_size)
    
    # 前向传播函数
    def forward(self, input_feature):
        # LSTM 前向传播
        recurrent, _ = self.rnn(
            input_feature
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        # 如果包含线性层，则对 LSTM 输出进行线性变换
        if self.with_linear:
            output = self.linear(recurrent)  # batch_size x T x output_size
            return output
        # 否则直接返回 LSTM 输出
        return recurrent
# 定义一个带有级联RNN的编码器类
class EncoderWithCascadeRNN(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 out_channels,
                 num_layers=2,
                 with_linear=False):
        super(EncoderWithCascadeRNN, self).__init__()
        # 设置输出通道数为最后一个通道数
        self.out_channels = out_channels[-1]
        # 创建一个包含多个双向LSTM层的编码器
        self.encoder = nn.LayerList([
            BidirectionalLSTM(
                in_channels if i == 0 else out_channels[i - 1],
                hidden_size,
                output_size=out_channels[i],
                num_layers=1,
                direction='bidirectional',
                with_linear=with_linear) for i in range(num_layers)
        ])

    # 前向传播函数
    def forward(self, x):
        # 遍历编码器中的每一层，并对输入进行处理
        for i, l in enumerate(self.encoder):
            x = l(x)
        return x


# 定义一个带有全连接层的编码器类
class EncoderWithFC(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        # 设置输出通道数为隐藏层大小
        self.out_channels = hidden_size
        # 获取全连接层的参数和偏置属性
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=0.00001, k=in_channels)
        # 创建一个全连接层
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name='reduce_encoder_fea')

    # 前向传播函数
    def forward(self, x):
        # 对输入进行全连接层处理
        x = self.fc(x)
        return x


# 定义一个带有SVTR的编码器类
class EncoderWithSVTR(nn.Layer):
    # 初始化权重函数
    def _init_weights(self, m):
        # 如果是线性层，使用截断正态分布初始化权重
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            # 如果是线性层且有偏置，初始化偏置为0
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        # 如果是LayerNorm层，初始化偏置为0，权重为1
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 如果使用引导，则克隆输入 x，并设置梯度为不计算
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        # 将 z 赋值给 h，用于后续的快捷方式
        h = z
        # 降低维度
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR 全局块
        B, C, H, W = z.shape
        z = z.flatten(2).transpose([0, 2, 1])
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        # 最后阶段
        z = z.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        z = self.conv3(z)
        z = paddle.concat((h, z), axis=1)
        z = self.conv1x1(self.conv4(z))
        # 返回 z
        return z
# 定义一个名为SequenceEncoder的类，继承自nn.Layer
class SequenceEncoder(nn.Layer):
    # 初始化函数，接受输入通道数、编码器类型、隐藏层大小和其他关键字参数
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        # 调用父类的初始化函数
        super(SequenceEncoder, self).__init__()
        # 创建一个Im2Seq对象，用于将输入数据转换为序列
        self.encoder_reshape = Im2Seq(in_channels)
        # 获取Im2Seq对象的输出通道数
        self.out_channels = self.encoder_reshape.out_channels
        # 存储编码器类型
        self.encoder_type = encoder_type
        # 如果编码器类型为'reshape'，则只进行形状重塑操作
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            # 定义支持的编码器类型及对应的类
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': EncoderWithRNN,
                'svtr': EncoderWithSVTR,
                'cascadernn': EncoderWithCascadeRNN
            }
            # 断言编码器类型在支持的编码器字典中
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())
            # 根据不同的编码器类型选择对应的编码器类进行初始化
            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, **kwargs)
            elif encoder_type == 'cascadernn':
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size, **kwargs)
            else:
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size)
            # 更新输出通道数为编码器的输出通道数
            self.out_channels = self.encoder.out_channels
            # 标记不仅进行形状重塑操作
            self.only_reshape = False

    # 前向传播函数
    def forward(self, x):
        # 如果编码器类型不是'svtr'，则先进行形状重塑操作，然后根据编码器类型进行编码
        if self.encoder_type != 'svtr':
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            # 如果编码器类型是'svtr'，则先进行'svtr'编码，再进行形状重塑操作
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x
```