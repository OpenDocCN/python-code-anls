# `.\PaddleOCR\ppocr\modeling\heads\rec_can_head.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何担保或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 本代码参考自:
# https://github.com/LBH1024/CAN/models/can.py
# https://github.com/LBH1024/CAN/models/counting.py
# https://github.com/LBH1024/CAN/models/decoder.py
# https://github.com/LBH1024/CAN/models/attention.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle.nn 和 paddle 模块
import paddle.nn as nn
import paddle
import math
'''
Counting Module
'''


# 定义 ChannelAtt 类，继承自 nn.Layer
class ChannelAtt(nn.Layer):
    # 初始化方法
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        # 创建自适应平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        # 创建线性层序列
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    # 前向传播方法
    def forward(self, x):
        # 获取输入张量的形状信息
        b, c, _, _ = x.shape
        # 对输入张量进行平均池化并重塑形状
        y = paddle.reshape(self.avg_pool(x), [b, c])
        # 通过线性层计算注意力权重并重塑形状
        y = paddle.reshape(self.fc(y), [b, c, 1, 1])
        # 返回加权后的张量
        return x * y


# 定义 CountingDecoder 类，继承自 nn.Layer
class CountingDecoder(nn.Layer):
    # 初始化 CountingDecoder 类，设置输入通道数、输出通道数和卷积核大小
    def __init__(self, in_channel, out_channel, kernel_size):
        # 调用父类的初始化方法
        super(CountingDecoder, self).__init__()
        # 保存输入通道数和输出通道数
        self.in_channel = in_channel
        self.out_channel = out_channel

        # 定义转换层，包括卷积和批归一化操作
        self.trans_layer = nn.Sequential(
            nn.Conv2D(
                self.in_channel,
                512,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias_attr=False),
            nn.BatchNorm2D(512))

        # 定义通道注意力机制
        self.channel_att = ChannelAtt(512, 16)

        # 定义预测层，包括卷积和 Sigmoid 激活函数
        self.pred_layer = nn.Sequential(
            nn.Conv2D(
                512, self.out_channel, kernel_size=1, bias_attr=False),
            nn.Sigmoid())

    # 前向传播函数，接收输入 x 和掩码 mask
    def forward(self, x, mask):
        # 获取输入 x 的形状信息
        b, _, h, w = x.shape
        # 经过转换层处理输入 x
        x = self.trans_layer(x)
        # 经过通道注意力机制处理 x
        x = self.channel_att(x)
        # 经过预测层处理 x
        x = self.pred_layer(x)

        # 如果存在掩码 mask，则将 x 与 mask 相乘
        if mask is not None:
            x = x * mask
        # 重新调整 x 的形状
        x = paddle.reshape(x, [b, self.out_channel, -1])
        # 沿着最后一个维度求和，得到 x1
        x1 = paddle.sum(x, axis=-1)

        # 返回 x1 和重新调整后的 x
        return x1, paddle.reshape(x, [b, self.out_channel, h, w])
'''
Attention Decoder
'''

# 定义一个位置编码器，使用正弦函数进行位置编码
class PositionEmbeddingSine(nn.Layer):
    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        # 如果传入了 scale 参数但未设置 normalize 为 True，则抛出数值错误
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        # 如果未传入 scale 参数，则默认设置为 2π
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # 前向传播函数，用于计算位置编码
    def forward(self, x, mask):
        # 计算 y 方向的位置编码
        y_embed = paddle.cumsum(mask, 1, dtype='float32')
        # 计算 x 方向的位置编码
        x_embed = paddle.cumsum(mask, 2, dtype='float32')

        # 如果需要归一化
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        # 生成维度参数
        dim_t = paddle.arange(self.num_pos_feats, dtype='float32')
        dim_d = paddle.expand(paddle.to_tensor(2), dim_t.shape)
        dim_t = self.temperature**(2 * (dim_t / dim_d).astype('int64') /
                                   self.num_pos_feats)

        # 计算 x 方向的位置编码
        pos_x = paddle.unsqueeze(x_embed, [3]) / dim_t
        # 计算 y 方向的位置编码
        pos_y = paddle.unsqueeze(y_embed, [3]) / dim_t

        # 对 x 方向的位置编码进行正弦和余弦函数处理
        pos_x = paddle.flatten(
            paddle.stack(
                [
                    paddle.sin(pos_x[:, :, :, 0::2]),
                    paddle.cos(pos_x[:, :, :, 1::2])
                ],
                axis=4),
            3)
        # 对 y 方向的位置编码进行正弦和余弦函数处理
        pos_y = paddle.flatten(
            paddle.stack(
                [
                    paddle.sin(pos_y[:, :, :, 0::2]),
                    paddle.cos(pos_y[:, :, :, 1::2])
                ],
                axis=4),
            3)

        # 合并 x 和 y 方向的位置编码，并进行转置
        pos = paddle.transpose(
            paddle.concat(
                [pos_y, pos_x], axis=3), [0, 3, 1, 2])

        return pos


# 定义一个注意力解码器
class AttDecoder(nn.Layer):
    # 初始化注意力解码器的参数
    def __init__(self, ratio, is_train, input_size, hidden_size,
                 encoder_out_channel, dropout, dropout_ratio, word_num,
                 counting_decoder_out_channel, attention):
        # 调用父类的初始化方法
        super(AttDecoder, self).__init__()
        # 设置输入大小、隐藏大小、编码器输出通道、注意力维度、dropout概率、比率、词汇数量等参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_channel = encoder_out_channel
        self.attention_dim = attention['attention_dim']
        self.dropout_prob = dropout
        self.ratio = ratio
        self.word_num = word_num

        self.counting_num = counting_decoder_out_channel
        self.is_train = is_train

        # 初始化权重
        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)
        # 初始化词嵌入层
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        # 初始化词输入GRU单元
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        # 初始化词注意力层
        self.word_attention = Attention(hidden_size, attention['attention_dim'])

        # 初始化编码器特征卷积层
        self.encoder_feature_conv = nn.Conv2D(
            self.out_channel,
            self.attention_dim,
            kernel_size=attention['word_conv_kernel'],
            padding=attention['word_conv_kernel'] // 2)

        # 初始化词状态权重
        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        # 初始化词嵌入权重
        self.word_embedding_weight = nn.Linear(self.input_size,
                                               self.hidden_size)
        # 初始化词上下文权重
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        # 初始化计数上下文权重
        self.counting_context_weight = nn.Linear(self.counting_num,
                                                 self.hidden_size)
        # 初始化词转换层
        self.word_convert = nn.Linear(self.hidden_size, self.word_num)

        # 如果有dropout，则初始化dropout层
        if dropout:
            self.dropout = nn.Dropout(dropout_ratio)
    # 初始化隐藏状态，根据输入的特征和特征掩码计算平均值
    average = paddle.sum(paddle.sum(features * feature_mask, axis=-1),
                         axis=-1) / paddle.sum(
                             (paddle.sum(feature_mask, axis=-1)), axis=-1)
    # 使用初始化权重函数对平均值进行处理
    average = self.init_weight(average)
    # 对处理后的平均值应用双曲正切函数，返回隐藏状态
    return paddle.tanh(average)
# 定义注意力机制模块
class Attention(nn.Layer):
    # 初始化函数，接受隐藏层大小和注意力维度作为参数
    def __init__(self, hidden_size, attention_dim):
        super(Attention, self).__init__()
        self.hidden = hidden_size
        self.attention_dim = attention_dim
        # 定义隐藏层到注意力维度的线性变换
        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        # 定义注意力卷积层
        self.attention_conv = nn.Conv2D(
            1, 512, kernel_size=11, padding=5, bias_attr=False)
        # 定义注意力权重线性变换
        self.attention_weight = nn.Linear(
            512, self.attention_dim, bias_attr=False)
        # 定义将注意力维度转换为1维的线性变换
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

    # 前向传播函数，接受CNN特征，CNN特征转置，隐藏状态，alpha和图像掩码作为输入
    def forward(self,
                cnn_features,
                cnn_features_trans,
                hidden,
                alpha_sum,
                image_mask=None):
        # 计算查询向量
        query = self.hidden_weight(hidden)
        # 对alpha_sum进行卷积操作
        alpha_sum_trans = self.attention_conv(alpha_sum)
        # 计算覆盖注意力
        coverage_alpha = self.attention_weight(
            paddle.transpose(alpha_sum_trans, [0, 2, 3, 1]))
        # 计算alpha分数
        alpha_score = paddle.tanh(
            paddle.unsqueeze(query, [1, 2]) + coverage_alpha + paddle.transpose(
                cnn_features_trans, [0, 2, 3, 1]))
        # 计算能量
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = paddle.exp(paddle.squeeze(energy, -1))

        # 如果存在图像掩码，则将能量乘以图像掩码
        if image_mask is not None:
            energy_exp = energy_exp * paddle.squeeze(image_mask, 1)
        # 计算alpha值
        alpha = energy_exp / (paddle.unsqueeze(
            paddle.sum(paddle.sum(energy_exp, -1), -1), [1, 2]) + 1e-10)
        alpha_sum = paddle.unsqueeze(alpha, 1) + alpha_sum
        # 计算上下文向量
        context_vector = paddle.sum(
            paddle.sum((paddle.unsqueeze(alpha, 1) * cnn_features), -1), -1)

        return context_vector, alpha, alpha_sum

# 定义CANHead类
class CANHead(nn.Layer):
    # 初始化 CANHead 类，传入输入通道数、输出通道数、比率、attdecoder 参数
    def __init__(self, in_channel, out_channel, ratio, attdecoder, **kwargs):
        # 调用父类的初始化方法
        super(CANHead, self).__init__()

        # 设置输入通道数和输出通道数
        self.in_channel = in_channel
        self.out_channel = out_channel

        # 创建一个 CountingDecoder 对象，用于计数预测，传入输入通道数、输出通道数和固定值 3
        self.counting_decoder1 = CountingDecoder(self.in_channel,
                                                 self.out_channel, 3)  # mscm
        # 创建另一个 CountingDecoder 对象，用于计数预测，传入输入通道数、输出通道数和固定值 5
        self.counting_decoder2 = CountingDecoder(self.in_channel,
                                                 self.out_channel, 5)

        # 创建一个 AttDecoder 对象，用于注意力解码，传入比率和额外的 attdecoder 参数
        self.decoder = AttDecoder(ratio, **attdecoder)

        # 设置比率
        self.ratio = ratio

    # 前向传播方法，接收输入和目标数据
    def forward(self, inputs, targets=None):
        # 解包输入数据，包括 CNN 特征、图像掩码和标签
        cnn_features, images_mask, labels = inputs

        # 对图像掩码进行下采样，步长为比率，得到计数掩码
        counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        # 使用第一个计数解码器进行计数预测
        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        # 使用第二个计数解码器进行计数预测
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        # 将两个计数预测结果取平均
        counting_preds = (counting_preds1 + counting_preds2) / 2

        # 使用注意力解码器进行词概率预测，传入 CNN 特征、标签、计数预测和图像掩码
        word_probs = self.decoder(cnn_features, labels, counting_preds,
                                  images_mask)
        # 返回词概率、总计数预测、第一个计数预测、第二个计数预测
        return word_probs, counting_preds, counting_preds1, counting_preds2
```