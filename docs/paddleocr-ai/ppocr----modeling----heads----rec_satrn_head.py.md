# `.\PaddleOCR\ppocr\modeling\heads\rec_satrn_head.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
"""
# 代码参考自:
# https://github.com/open-mmlab/mmocr/blob/1.x/mmocr/models/textrecog/encoders/satrn_encoder.py
# https://github.com/open-mmlab/mmocr/blob/1.x/mmocr/models/textrecog/decoders/nrtr_decoder.py

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr, reshape, transpose
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import KaimingNormal, Uniform, Constant

# 定义 ConvBNLayer 类，包含卷积、批归一化和激活函数
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 num_groups=1):
        super(ConvBNLayer, self).__init__()

        # 创建卷积层，设置输入通道数、输出通道数、卷积核大小、步长、填充和分组数
        self.conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)

        # 创建批归一化层，设置输出通道数和权重、偏置的初始化方式
        self.bn = nn.BatchNorm2D(
            num_filters,
            weight_attr=ParamAttr(initializer=Constant(1)),
            bias_attr=ParamAttr(initializer=Constant(0)))
        # 创建 ReLU 激活函数层
        self.relu = nn.ReLU()
    # 定义一个前向传播函数，接收输入并返回输出
    def forward(self, inputs):
        # 将输入数据通过卷积层处理得到输出
        y = self.conv(inputs)
        # 将卷积层的输出通过批量归一化层处理
        y = self.bn(y)
        # 将批量归一化层的输出通过激活函数处理
        y = self.relu(y)
        # 返回处理后的输出
        return y
# 定义一个 SATRNEncoderLayer 类，用于实现 Transformer 编码器层
class SATRNEncoderLayer(nn.Layer):
    # 初始化函数，设置各种参数
    def __init__(self,
                 d_model=512,
                 d_inner=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False):
        super().__init__()
        # LayerNorm 层，用于归一化输入数据
        self.norm1 = nn.LayerNorm(d_model)
        # 多头注意力机制，用于提取输入数据的关键信息
        self.attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        # LayerNorm 层，用于归一化输入数据
        self.norm2 = nn.LayerNorm(d_model)
        # 自定义的前馈神经网络，用于对数据进行非线性变换
        self.feed_forward = LocalityAwareFeedforward(
            d_model, d_inner, dropout=dropout)

    # 前向传播函数，处理输入数据并返回处理后的结果
    def forward(self, x, h, w, mask=None):
        # 获取输入数据的形状信息
        n, hw, c = x.shape
        # 保存输入数据的残差连接
        residual = x
        # 对输入数据进行归一化处理
        x = self.norm1(x)
        # 使用多头注意力机制处理输入数据
        x = residual + self.attn(x, x, x, mask)
        # 保存处理后的数据的残差连接
        residual = x
        # 对处理后的数据再次进行归一化处理
        x = self.norm2(x)
        # 调整数据的形状
        x = x.transpose([0, 2, 1]).reshape([n, c, h, w])
        # 使用自定义的前馈神经网络处理数据
        x = self.feed_forward(x)
        # 调整数据的形状
        x = x.reshape([n, c, hw]).transpose([0, 2, 1])
        # 将残差连接与处理后的数据相加并返回结果
        x = residual + x
        return x

# 定义一个 LocalityAwareFeedforward 类，用于实现自定义的前馈神经网络
class LocalityAwareFeedforward(nn.Layer):
    # 初始化函数，设置各种参数
    def __init__(
            self,
            d_in,
            d_hid,
            dropout=0.1, ):
        super().__init__()
        # 第一个卷积层，用于对输入数据进行卷积操作
        self.conv1 = ConvBNLayer(d_in, 1, d_hid, stride=1, padding=0)
        # 深度可分离卷积层，用于对输入数据进行深度可分离卷积操作
        self.depthwise_conv = ConvBNLayer(
            d_hid, 3, d_hid, stride=1, padding=1, num_groups=d_hid)
        # 第二个卷积层，用于对输入数据进行卷积操作
        self.conv2 = ConvBNLayer(d_hid, 1, d_in, stride=1, padding=0)

    # 前向传播函数，处理输入数据并返回处理后的结果
    def forward(self, x):
        # 使用第一个卷积层处理输入数据
        x = self.conv1(x)
        # 使用深度可分离卷积层处理数据
        x = self.depthwise_conv(x)
        # 使用第二个卷积层处理数据
        x = self.conv2(x)

        return x

# 定义一个 Adaptive2DPositionalEncoding 类
class Adaptive2DPositionalEncoding(nn.Layer):
    # 初始化函数，设置默认隐藏层维度、高度、宽度和dropout率
    def __init__(self, d_hid=512, n_height=100, n_width=100, dropout=0.1):
        # 调用父类的初始化函数
        super().__init__()

        # 获取高度位置编码表
        h_position_encoder = self._get_sinusoid_encoding_table(n_height, d_hid)
        # 转置高度位置编码表
        h_position_encoder = h_position_encoder.transpose([1, 0])
        # 重塑高度位置编码表
        h_position_encoder = h_position_encoder.reshape([1, d_hid, n_height, 1])

        # 获取宽度位置编码表
        w_position_encoder = self._get_sinusoid_encoding_table(n_width, d_hid)
        # 转置宽度位置编码表
        w_position_encoder = w_position_encoder.transpose([1, 0])
        # 重塑宽度位置编码表
        w_position_encoder = w_position_encoder.reshape([1, d_hid, 1, n_width])

        # 注册高度位置编码表为缓冲区
        self.register_buffer('h_position_encoder', h_position_encoder)
        # 注册宽度位置编码表为缓冲区
        self.register_buffer('w_position_encoder', w_position_encoder)

        # 生成高度缩放因子
        self.h_scale = self.scale_factor_generate(d_hid)
        # 生成宽度缩放因子
        self.w_scale = self.scale_factor_generate(d_hid)
        # 创建自适应平均池化层
        self.pool = nn.AdaptiveAvgPool2D(1)
        # 创建dropout层
        self.dropout = nn.Dropout(p=dropout)

    # 获取正弦位置编码表
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        # 计算分母
        denominator = paddle.to_tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.reshape([1, -1])
        # 创建位置张量
        pos_tensor = paddle.cast(
            paddle.arange(n_position).unsqueeze(-1), 'float32')
        # 计算正弦编码表
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = paddle.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = paddle.cos(sinusoid_table[:, 1::2])

        return sinusoid_table

    # 生成缩放因子
    def scale_factor_generate(self, d_hid):
        # 创建缩放因子模型
        scale_factor = nn.Sequential(
            nn.Conv2D(d_hid, d_hid, 1),
            nn.ReLU(), nn.Conv2D(d_hid, d_hid, 1), nn.Sigmoid())

        return scale_factor
    # 定义一个前向传播函数，接受输入张量 x
    def forward(self, x):
        # 获取输入张量 x 的形状信息，分别为批大小 b、通道数 c、高度 h、宽度 w
        b, c, h, w = x.shape

        # 对输入张量 x 进行平均池化操作
        avg_pool = self.pool(x)

        # 计算高度位置编码，通过对平均池化结果进行高度缩放和高度位置编码矩阵的切片操作得到
        h_pos_encoding = \
            self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        
        # 计算宽度位置编码，通过对平均池化结果进行宽度缩放和宽度位置编码矩阵的切片操作得到
        w_pos_encoding = \
            self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]

        # 将输入张量 x、高度位置编码和宽度位置编码相加得到输出
        out = x + h_pos_encoding + w_pos_encoding

        # 对输出进行 dropout 操作
        out = self.dropout(out)

        # 返回处理后的输出
        return out
class ScaledDotProductAttention(nn.Layer):
    # 定义缩放点积注意力机制类
    def __init__(self, temperature, attn_dropout=0.1):
        # 初始化函数，传入温度参数和注意力机制的dropout率
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # 前向传播函数，传入查询、键、值以及可选的掩码
        def masked_fill(x, mask, value):
            # 定义填充掩码的函数
            y = paddle.full(x.shape, value, x.dtype)
            return paddle.where(mask, y, x)

        # 计算注意力分数
        attn = paddle.matmul(q / self.temperature, k.transpose([0, 1, 3, 2]))
        if mask is not None:
            # 如果存在掩码，则进行填充
            attn = masked_fill(attn, mask == 0, -1e9)
            # attn = attn.masked_fill(mask == 0, float('-inf'))
            # attn += mask

        # 对注意力分数进行softmax归一化
        attn = self.dropout(F.softmax(attn, axis=-1))
        # 计算输出
        output = paddle.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Layer):
    # 定义多头注意力机制类
    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False):
        # 初始化函数，传入头数、模型维度、键和值的维度、dropout率以及是否使用偏置
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        # 定义线性变换层
        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias_attr=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias_attr=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias_attr=qkv_bias)

        # 使用缩放点积注意力机制
        self.attention = ScaledDotProductAttention(d_k**0.5, dropout)

        # 定义全连接层和dropout层
        self.fc = nn.Linear(self.dim_v, d_model, bias_attr=qkv_bias)
        self.proj_drop = nn.Dropout(dropout)
    # 定义一个前向传播函数，接受查询(q)、键(k)、值(v)和掩码(mask)作为输入
    def forward(self, q, k, v, mask=None):
        # 获取输入张量的形状信息
        batch_size, len_q, _ = q.shape
        _, len_k, _ = k.shape

        # 将查询(q)、键(k)、值(v)分别通过线性变换并重塑形状
        q = self.linear_q(q).reshape([batch_size, len_q, self.n_head, self.d_k])
        k = self.linear_k(k).reshape([batch_size, len_k, self.n_head, self.d_k])
        v = self.linear_v(v).reshape([batch_size, len_k, self.n_head, self.d_v])

        # 将查询(q)、键(k)、值(v)的维度进行转置
        q, k, v = q.transpose([0, 2, 1, 3]), k.transpose(
            [0, 2, 1, 3]), v.transpose([0, 2, 1, 3])

        # 如果存在掩码(mask)，则根据不同维度进行扩展
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)

        # 调用注意力机制函数，计算注意力输出和注意力权重
        attn_out, _ = self.attention(q, k, v, mask=mask)

        # 将注意力输出的维度进行转置和重塑
        attn_out = attn_out.transpose([0, 2, 1, 3]).reshape(
            [batch_size, len_q, self.dim_v])

        # 通过全连接层进行特征映射
        attn_out = self.fc(attn_out)
        # 对映射后的特征进行丢弃(dropout)
        attn_out = self.proj_drop(attn_out)

        # 返回处理后的注意力输出
        return attn_out
# 定义一个名为 SATRNEncoder 的类，继承自 nn.Layer 类
class SATRNEncoder(nn.Layer):
    # 初始化函数，设置模型的各种参数
    def __init__(self,
                 n_layers=12,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 n_position=100,
                 d_inner=256,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # 创建一个自适应的二维位置编码对象
        self.position_enc = Adaptive2DPositionalEncoding(
            d_hid=d_model,
            n_height=n_position,
            n_width=n_position,
            dropout=dropout)
        # 创建一个包含多个 SATRNEncoderLayer 的列表
        self.layer_stack = nn.LayerList([
            SATRNEncoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        # 创建一个 LayerNorm 层
        self.layer_norm = nn.LayerNorm(d_model)

    # 前向传播函数
    def forward(self, feat, valid_ratios=None):
        """
        Args:
            feat (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: A tensor of shape :math:`(N, T, D_m)`.
        """
        # 如果未提供有效比例，则默认为 1.0
        if valid_ratios is None:
            valid_ratios = [1.0 for _ in range(feat.shape[0])]
        # 对输入特征进行位置编码
        feat = self.position_enc(feat)
        n, c, h, w = feat.shape

        # 创建一个全零的掩码张量
        mask = paddle.zeros((n, h, w))
        # 根据有效比例设置掩码值
        for i, valid_ratio in enumerate(valid_ratios):
            valid_width = min(w, math.ceil(w * valid_ratio))
            mask[i, :, :valid_width] = 1

        # 重塑掩码张量和特征张量的形状
        mask = mask.reshape([n, h * w])
        feat = feat.reshape([n, c, h * w])

        # 调整特征张量的维度顺序
        output = feat.transpose([0, 2, 1])
        # 遍历每个编码层并进行前向传播
        for enc_layer in self.layer_stack:
            output = enc_layer(output, h, w, mask)
        # 对输出进行 LayerNorm 处理
        output = self.layer_norm(output)

        return output


class PositionwiseFeedForward(nn.Layer):
    # 初始化神经网络模型的参数和层
    def __init__(self, d_in, d_hid, dropout=0.1):
        # 调用父类的初始化方法
        super().__init__()
        # 定义输入层到隐藏层的线性变换
        self.w_1 = nn.Linear(d_in, d_hid)
        # 定义隐藏层到输出层的线性变换
        self.w_2 = nn.Linear(d_hid, d_in)
        # 定义激活函数为 GELU
        self.act = nn.GELU()
        # 定义 dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    # 定义前向传播过程
    def forward(self, x):
        # 输入层到隐藏层的线性变换
        x = self.w_1(x)
        # 使用 GELU 激活函数
        x = self.act(x)
        # 隐藏层到输出层的线性变换
        x = self.w_2(x)
        # 使用 dropout 层进行正则化
        x = self.dropout(x)

        # 返回输出结果
        return x
# 定义一个继承自 nn.Layer 的 PositionalEncoding 类
class PositionalEncoding(nn.Layer):
    # 初始化函数，设置隐藏层维度、位置编码长度和 dropout 概率
    def __init__(self, d_hid=512, n_position=200, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 不是一个参数，注册一个缓冲区，存储位置编码表，形状为 (1, n_position, d_hid)
        self.register_buffer(
            'position_table',
            self._get_sinusoid_encoding_table(n_position, d_hid))

    # 获取正弦位置编码表的函数
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        # 计算分母
        denominator = paddle.to_tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.reshape([1, -1])
        # 生成位置张量
        pos_tensor = paddle.cast(
            paddle.arange(n_position).unsqueeze(-1), 'float32')
        # 计算正弦和余弦值
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = paddle.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = paddle.cos(sinusoid_table[:, 1::2])

        return sinusoid_table.unsqueeze(0)

    # 前向传播函数
    def forward(self, x):
        # 对输入张量 x 加上位置编码表，并进行克隆和分离
        x = x + self.position_table[:, :x.shape[1]].clone().detach()
        return self.dropout(x)


class TFDecoderLayer(nn.Layer):
    # 初始化 TransformerEncoderLayer 类
    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 operation_order=None):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化 LayerNorm 层对象，用于归一化输入数据
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # 初始化自注意力机制对象
        self.self_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        # 初始化编码器-解码器注意力机制对象
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        # 初始化前馈神经网络对象
        self.mlp = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        # 设置操作顺序，默认为 None
        self.operation_order = operation_order
        # 如果操作顺序为 None，则设置默认操作顺序
        if self.operation_order is None:
            self.operation_order = ('norm', 'self_attn', 'norm', 'enc_dec_attn',
                                    'norm', 'ffn')
        # 断言操作顺序在指定的范围内
        assert self.operation_order in [
            ('norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'),
            ('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn', 'norm')
        ]
    # 定义前向传播函数，接受解码器输入、编码器输出，以及自注意力掩码和编码-解码注意力掩码
    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None):
        # 如果操作顺序为('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn', 'norm')
        if self.operation_order == ('self_attn', 'norm', 'enc_dec_attn', 'norm',
                                    'ffn', 'norm'):
            # 使用解码器输入进行自注意力计算
            dec_attn_out = self.self_attn(dec_input, dec_input, dec_input,
                                          self_attn_mask)
            # 将自注意力输出与解码器输入相加
            dec_attn_out += dec_input
            # 对相加后的结果进行归一化
            dec_attn_out = self.norm1(dec_attn_out)

            # 使用解码器自注意力输出和编码器输出进行编码-解码注意力计算
            enc_dec_attn_out = self.enc_attn(dec_attn_out, enc_output,
                                             enc_output, dec_enc_attn_mask)
            # 将编码-解码注意力输出与解码器自注意力输出相加
            enc_dec_attn_out += dec_attn_out
            # 对相加后的结果进行归一化
            enc_dec_attn_out = self.norm2(enc_dec_attn_out)

            # 使用多层感知机处理编码-解码注意力输出
            mlp_out = self.mlp(enc_dec_attn_out)
            # 将多层感知机输出与编码-解码注意力输出相加
            mlp_out += enc_dec_attn_out
            # 对相加后的结果进行归一化
            mlp_out = self.norm3(mlp_out)
        # 如果操作顺序为('norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn')
        elif self.operation_order == ('norm', 'self_attn', 'norm',
                                      'enc_dec_attn', 'norm', 'ffn'):
            # 对解码器输入进行归一化
            dec_input_norm = self.norm1(dec_input)
            # 使用归一化后的解码器输入进行自注意力计算
            dec_attn_out = self.self_attn(dec_input_norm, dec_input_norm,
                                          dec_input_norm, self_attn_mask)
            # 将自注意力输出与解码器输入相加
            dec_attn_out += dec_input

            # 对自注意力输出进行归一化
            enc_dec_attn_in = self.norm2(dec_attn_out)
            # 使用归一化后的自注意力输出和编码器输出进行编码-解码注意力计算
            enc_dec_attn_out = self.enc_attn(enc_dec_attn_in, enc_output,
                                             enc_output, dec_enc_attn_mask)
            # 将编码-解码注意力输出与自注意力输出相加
            enc_dec_attn_out += dec_attn_out

            # 使用多层感知机处理归一化后的编码-解码注意力输出
            mlp_out = self.mlp(self.norm3(enc_dec_attn_out))
            # 将多层感知机输出与编码-解码注意力输出相加
            mlp_out += enc_dec_attn_out

        # 返回多层感知机输出
        return mlp_out
class SATRNDecoder(nn.Layer):
    # 定义 SATRN 解码器类
    def __init__(self,
                 n_layers=6,
                 d_embedding=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=256,
                 n_position=200,
                 dropout=0.1,
                 num_classes=93,
                 max_seq_len=40,
                 start_idx=1,
                 padding_idx=92):
        # 初始化函数，设置解码器的各种参数
        super().__init__()

        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len

        self.trg_word_emb = nn.Embedding(
            num_classes, d_embedding, padding_idx=padding_idx)
        # 定义目标词嵌入层

        self.position_enc = PositionalEncoding(
            d_embedding, n_position=n_position)
        # 定义位置编码层
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.LayerList([
            TFDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        # 创建多层 Transformer 解码器层
        self.layer_norm = nn.LayerNorm(d_model, epsilon=1e-6)

        pred_num_class = num_classes - 1  # ignore padding_idx
        self.classifier = nn.Linear(d_model, pred_num_class)
        # 定义线性分类器

    @staticmethod
    def get_pad_mask(seq, pad_idx):
        # 静态方法，用于生成填充掩码
        return (seq != pad_idx).unsqueeze(-2)

    @staticmethod
    def get_subsequent_mask(seq):
        """For masking out the subsequent info."""
        # 用于屏蔽后续信息的掩码
        len_s = seq.shape[1]
        subsequent_mask = 1 - paddle.triu(
            paddle.ones((len_s, len_s)), diagonal=1)
        # 生成上三角矩阵
        subsequent_mask = paddle.cast(subsequent_mask.unsqueeze(0), 'bool')

        return subsequent_mask
    # 定义注意力函数，接收目标序列、源数据、源数据掩码作为输入
    def _attention(self, trg_seq, src, src_mask=None):
        # 将目标序列转换为词嵌入
        trg_embedding = self.trg_word_emb(trg_seq)
        # 对目标序列进行位置编码
        trg_pos_encoded = self.position_enc(trg_embedding)
        # 对目标序列进行dropout处理
        tgt = self.dropout(trg_pos_encoded)

        # 生成目标序列的掩码，包括填充掩码和后续掩码
        trg_mask = self.get_pad_mask(
            trg_seq,
            pad_idx=self.padding_idx) & self.get_subsequent_mask(trg_seq)
        output = tgt
        # 遍历每个解码层
        for dec_layer in self.layer_stack:
            # 对输出进行解码层处理
            output = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
        # 对输出进行层归一化处理
        output = self.layer_norm(output)

        return output

    # 定义获取掩码的函数，接收logit和有效比例作为输入
    def _get_mask(self, logit, valid_ratios):
        N, T, _ = logit.shape
        mask = None
        # 如果有效比例不为空
        if valid_ratios is not None:
            mask = paddle.zeros((N, T))
            # 遍历每个样本和对应的有效比例
            for i, valid_ratio in enumerate(valid_ratios):
                # 计算有效宽度
                valid_width = min(T, math.ceil(T * valid_ratio))
                mask[i, :valid_width] = 1

        return mask

    # 定义训练前向传播函数，接收特征、编码器输出、目标序列和有效比例作为输入
    def forward_train(self, feat, out_enc, targets, valid_ratio):
        # 获取源数据掩码
        src_mask = self._get_mask(out_enc, valid_ratio)
        # 对目标序列进行注意力计算
        attn_output = self._attention(targets, out_enc, src_mask=src_mask)
        # 对注意力输出进行分类器处理
        outputs = self.classifier(attn_output)

        return outputs
    # 在前向传播中进行测试，生成模型输出
    def forward_test(self, feat, out_enc, valid_ratio):
        # 获取源序列的掩码
        src_mask = self._get_mask(out_enc, valid_ratio)
        # 获取输出编码的样本数量
        N = out_enc.shape[0]
        # 初始化目标序列，填充为padding_idx
        init_target_seq = paddle.full(
            (N, self.max_seq_len + 1), self.padding_idx, dtype='int64')
        # 将起始标记填充到目标序列的第一列
        init_target_seq[:, 0] = self.start_idx

        outputs = []
        # 遍历最大序列长度
        for step in range(0, paddle.to_tensor(self.max_seq_len)):
            # 使用注意力机制生成解码器输出
            decoder_output = self._attention(
                init_target_seq, out_enc, src_mask=src_mask)
            # 对解码器输出进行softmax操作
            step_result = F.softmax(
                self.classifier(decoder_output[:, step, :]), axis=-1)
            # 将当前步骤的结果添加到输出列表中
            outputs.append(step_result)
            # 获取当前步骤的最大概率索引
            step_max_index = paddle.argmax(step_result, axis=-1)
            # 将最大概率索引填充到目标序列的下一列
            init_target_seq[:, step + 1] = step_max_index

        # 在第二维度上堆叠输出结果
        outputs = paddle.stack(outputs, axis=1)

        return outputs

    # 在前向传播中根据训练状态选择执行训练或测试
    def forward(self, feat, out_enc, targets=None, valid_ratio=None):
        # 如果处于训练状态，则执行训练前向传播
        if self.training:
            return self.forward_train(feat, out_enc, targets, valid_ratio)
        # 否则执行测试前向传播
        else:
            return self.forward_test(feat, out_enc, valid_ratio)
class SATRNHead(nn.Layer):
    # 定义 SATRNHead 类，继承自 nn.Layer 类
    def __init__(self, enc_cfg, dec_cfg, **kwargs):
        # 初始化函数，接受 encoder 和 decoder 的配置参数以及其他关键字参数
        super(SATRNHead, self).__init__()
        # 调用父类的初始化函数

        # encoder 模块
        self.encoder = SATRNEncoder(**enc_cfg)
        # 创建 SATRNEncoder 实例并赋值给 self.encoder

        # decoder 模块
        self.decoder = SATRNDecoder(**dec_cfg)
        # 创建 SATRNDecoder 实例并赋值给 self.decoder

    def forward(self, feat, targets=None):
        # 前向传播函数，接受特征 feat 和目标 targets（可选）

        if targets is not None:
            # 如果目标不为空
            targets, valid_ratio = targets
            # 将目标和有效比例分别赋值给 targets 和 valid_ratio
        else:
            targets, valid_ratio = None, None
            # 否则将 targets 和 valid_ratio 设置为 None

        holistic_feat = self.encoder(feat, valid_ratio)  # bsz c
        # 使用 encoder 处理特征 feat 和有效比例 valid_ratio，得到 holistic_feat

        final_out = self.decoder(feat, holistic_feat, targets, valid_ratio)
        # 使用 decoder 处理特征 feat、holistic_feat、目标 targets 和有效比例 valid_ratio，得到 final_out

        return final_out
        # 返回 final_out
```