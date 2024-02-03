# `.\PaddleOCR\ppocr\modeling\heads\rec_aster_head.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用来源
# https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/attention_recognition_head.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import paddle
from paddle import nn
from paddle.nn import functional as F

# 定义 AsterHead 类，继承自 nn.Layer
class AsterHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 sDim,
                 attDim,
                 max_len_labels,
                 time_step=25,
                 beam_width=5,
                 **kwargs):
        super(AsterHead, self).__init__()
        # 输出通道数
        self.num_classes = out_channels
        # 输入通道数
        self.in_planes = in_channels
        # sDim 参数
        self.sDim = sDim
        # attDim 参数
        self.attDim = attDim
        # 最大标签长度
        self.max_len_labels = max_len_labels
        # 初始化 AttentionRecognitionHead 类
        self.decoder = AttentionRecognitionHead(in_channels, out_channels, sDim,
                                                attDim, max_len_labels)
        # 时间步数
        self.time_step = time_step
        # 初始化 Embedding 类
        self.embeder = Embedding(self.time_step, in_channels)
        # Beam 搜索宽度
        self.beam_width = beam_width
        # 结束标记
        self.eos = self.num_classes - 3
    # 定义一个前向传播函数，接受输入 x，目标 targets 和嵌入 embed
    def forward(self, x, targets=None, embed=None):
        # 初始化返回结果字典
        return_dict = {}
        # 使用嵌入层处理输入 x，得到嵌入向量
        embedding_vectors = self.embeder(x)

        # 如果处于训练状态
        if self.training:
            # 解包目标 targets，获取目标序列、序列长度和其他信息
            rec_targets, rec_lengths, _ = targets
            # 使用解码器处理输入 x、目标序列和序列长度，得到重构预测结果
            rec_pred = self.decoder([x, rec_targets, rec_lengths], embedding_vectors)
            # 将重构预测结果和嵌入向量添加到返回结果字典中
            return_dict['rec_pred'] = rec_pred
            return_dict['embedding_vectors'] = embedding_vectors
        else:
            # 使用束搜索算法生成重构预测结果和对应的分数
            rec_pred, rec_pred_scores = self.decoder.beam_search(
                x, self.beam_width, self.eos, embedding_vectors)
            # 将重构预测结果、分数和嵌入向量添加到返回结果字典中
            return_dict['rec_pred'] = rec_pred
            return_dict['rec_pred_scores'] = rec_pred_scores
            return_dict['embedding_vectors'] = embedding_vectors

        # 返回结果字典
        return return_dict
class Embedding(nn.Layer):
    # 定义嵌入层类，用于将输入数据编码成词嵌入形式
    def __init__(self, in_timestep, in_planes, mid_dim=4096, embed_dim=300):
        super(Embedding, self).__init__()
        self.in_timestep = in_timestep
        self.in_planes = in_planes
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        # 创建线性层，将输入数据编码成词嵌入形式
        self.eEmbed = nn.Linear(
            in_timestep * in_planes,
            self.embed_dim)  # Embed encoder output to a word-embedding like

    def forward(self, x):
        # 重塑输入数据的形状
        x = paddle.reshape(x, [paddle.shape(x)[0], -1])
        # 将输入数据通过嵌入层进行编码
        x = self.eEmbed(x)
        return x


class AttentionRecognitionHead(nn.Layer):
    """
  input: [b x 16 x 64 x in_planes]
  output: probability sequence: [b x T x num_classes]
  """

    def __init__(self, in_channels, out_channels, sDim, attDim, max_len_labels):
        super(AttentionRecognitionHead, self).__init__()
        self.num_classes = out_channels  # this is the output classes. So it includes the <EOS>.
        self.in_planes = in_channels
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels

        # 创建解码器单元
        self.decoder = DecoderUnit(
            sDim=sDim, xDim=in_channels, yDim=self.num_classes, attDim=attDim)

    def forward(self, x, embed):
        x, targets, lengths = x
        batch_size = paddle.shape(x)[0]
        # Decoder
        # 获取解码器的初始状态
        state = self.decoder.get_initial_state(embed)
        outputs = []
        for i in range(max(lengths)):
            if i == 0:
                y_prev = paddle.full(
                    shape=[batch_size], fill_value=self.num_classes)
            else:
                y_prev = targets[:, i - 1]
            # 解码器进行解码
            output, state = self.decoder(x, state, y_prev)
            outputs.append(output)
        # 将输出结果拼接在一起
        outputs = paddle.concat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs

    # inference stage.
    # 定义一个方法，用于对输入数据进行采样
    def sample(self, x):
        # 解包输入数据，获取 x 的值
        x, _, _ = x
        # 获取批量大小
        batch_size = x.size(0)
        
        # 初始化解码器状态
        state = paddle.zeros([1, batch_size, self.sDim])

        # 初始化预测的标识和分数列表
        predicted_ids, predicted_scores = [], []
        
        # 循环生成预测结果
        for i in range(self.max_len_labels):
            if i == 0:
                # 如果是第一次迭代，将 y_prev 初始化为全为 num_classes 的张量
                y_prev = paddle.full(
                    shape=[batch_size], fill_value=self.num_classes)
            else:
                # 否则将 y_prev 设置为上一次的预测结果
                y_prev = predicted

            # 使用解码器进行解码，得到输出和新的状态
            output, state = self.decoder(x, state, y_prev)
            # 对输出进行 softmax 处理
            output = F.softmax(output, axis=1)
            # 获取最大值和对应的索引
            score, predicted = output.max(1)
            # 将预测的标识和分数添加到列表中
            predicted_ids.append(predicted.unsqueeze(1))
            predicted_scores.append(score.unsqueeze(1))
        
        # 将预测的标识和分数拼接成张量
        predicted_ids = paddle.concat([predicted_ids, 1])
        predicted_scores = paddle.concat([predicted_scores, 1])
        
        # 返回预测的标识和分数
        return predicted_ids, predicted_scores
# 定义注意力机制单元的类
class AttentionUnit(nn.Layer):
    # 初始化函数，接受输入维度sDim、xDim和注意力维度attDim
    def __init__(self, sDim, xDim, attDim):
        super(AttentionUnit, self).__init__()

        # 初始化类的属性
        self.sDim = sDim
        self.xDim = xDim
        self.attDim = attDim

        # 创建线性层，用于将输入sDim映射到attDim维度
        self.sEmbed = nn.Linear(sDim, attDim)
        # 创建线性层，用于将输入xDim映射到attDim维度
        self.xEmbed = nn.Linear(xDim, attDim)
        # 创建线性层，用于将attDim维度映射到1维度
        self.wEmbed = nn.Linear(attDim, 1)

    # 前向传播函数，接受输入x和sPrev
    def forward(self, x, sPrev):
        # 获取输入x的形状信息，batch_size为批量大小，T为时间步，_为xDim
        batch_size, T, _ = x.shape  # [b x T x xDim]
        # 将输入x重塑为二维张量，形状为[(b x T) x xDim]
        x = paddle.reshape(x, [-1, self.xDim])  # [(b x T) x xDim]
        # 将重塑后的x通过xEmbed层映射到attDim维度，形状为[(b x T) x attDim]
        xProj = self.xEmbed(x)  # [(b x T) x attDim]
        # 将xProj再次重塑为三维张量，形状为[b x T x attDim]
        xProj = paddle.reshape(xProj, [batch_size, T, -1])  # [b x T x attDim]

        # 压缩输入sPrev的第一个维度
        sPrev = sPrev.squeeze(0)
        # 将压缩后的sPrev通过sEmbed层映射到attDim维度，形状为[b x attDim]
        sProj = self.sEmbed(sPrev)  # [b x attDim]
        # 在第二个维度上增加一个维度，形状变为[b x 1 x attDim]
        sProj = paddle.unsqueeze(sProj, 1)  # [b x 1 x attDim]
        # 在第二个维度上复制sProj，使其形状为[b x T x attDim]
        sProj = paddle.expand(sProj, [batch_size, T, self.attDim])  # [b x T x attDim]

        # 计算tanh(sProj + xProj)，形状为[b x T x attDim]
        sumTanh = paddle.tanh(sProj + xProj)
        # 将sumTanh重塑为二维张量，形状为[(b x T) x attDim]
        sumTanh = paddle.reshape(sumTanh, [-1, self.attDim])

        # 将sumTanh通过wEmbed层映射到1维度，形状为[(b x T) x 1]
        vProj = self.wEmbed(sumTanh)  # [(b x T) x 1]
        # 将vProj重塑为二维张量，形状为[b x T]
        vProj = paddle.reshape(vProj, [batch_size, T])
        # 对vProj进行softmax操作，沿着第二个维度计算，得到注意力权重alpha
        alpha = F.softmax(vProj, axis=1)  # attention weights for each sample in the minibatch
        # 返回注意力权重alpha
        return alpha


class DecoderUnit(nn.Layer):
    # 初始化解码器单元，设置输入维度、输出维度、注意力维度和注意力机制的维度
    def __init__(self, sDim, xDim, yDim, attDim):
        # 调用父类的初始化方法
        super(DecoderUnit, self).__init__()
        # 初始化解码器单元的各个维度参数
        self.sDim = sDim
        self.xDim = xDim
        self.yDim = yDim
        self.attDim = attDim
        self.emdDim = attDim

        # 初始化注意力单元
        self.attention_unit = AttentionUnit(sDim, xDim, attDim)
        # 初始化目标词嵌入层
        self.tgt_embedding = nn.Embedding(
            yDim + 1, self.emdDim, weight_attr=nn.initializer.Normal(
                std=0.01))  # 最后一个用于<BOS>
        # 初始化GRU单元
        self.gru = nn.GRUCell(input_size=xDim + self.emdDim, hidden_size=sDim)
        # 初始化全连接层
        self.fc = nn.Linear(
            sDim,
            yDim,
            weight_attr=nn.initializer.Normal(std=0.01),
            bias_attr=nn.initializer.Constant(value=0))
        # 初始化嵌入层全连接层
        self.embed_fc = nn.Linear(300, self.sDim)

    # 获取初始状态
    def get_initial_state(self, embed, tile_times=1):
        # 断言嵌入层的维度为300
        assert embed.shape[1] == 300
        # 将嵌入层通过全连接层映射到sDim维度，得到状态
        state = self.embed_fc(embed)  # N * sDim
        # 如果需要复制多次状态
        if tile_times != 1:
            # 在第二维度上增加一个维度
            state = state.unsqueeze(1)
            # 转置状态
            trans_state = paddle.transpose(state, perm=[1, 0, 2])
            # 在第二维度上复制状态
            state = paddle.tile(trans_state, repeat_times=[tile_times, 1, 1])
            # 再次转置状态
            trans_state = paddle.transpose(state, perm=[1, 0, 2])
            # 重新调整状态形状
            state = paddle.reshape(trans_state, shape=[-1, self.sDim])
        # 在第一维度上增加一个维度
        state = state.unsqueeze(0)  # 1 * N * sDim
        # 返回状态
        return state
    # 前向传播函数，接受输入特征 x、上一个隐藏状态 sPrev 和上一个输出 yPrev
    def forward(self, x, sPrev, yPrev):
        # x: 图像解码器输出的特征序列
        batch_size, T, _ = x.shape
        # 使用注意力机制计算注意力权重 alpha
        alpha = self.attention_unit(x, sPrev)
        # 根据注意力权重计算上下文向量
        context = paddle.squeeze(paddle.matmul(alpha.unsqueeze(1), x), axis=1)
        # 将上一个输出 yPrev 转换为 int64 类型
        yPrev = paddle.cast(yPrev, dtype="int64")
        # 使用目标词嵌入层将上一个输出 yPrev 映射为词嵌入向量 yProj
        yProj = self.tgt_embedding(yPrev)

        # 将词嵌入向量 yProj 和上下文向量 context 连接起来
        concat_context = paddle.concat([yProj, context], 1)
        concat_context = paddle.squeeze(concat_context, 1)
        # 压缩上一个隐藏状态 sPrev 的维度
        sPrev = paddle.squeeze(sPrev, 0)
        # 使用 GRU 网络计算输出和新的隐藏状态
        output, state = self.gru(concat_context, sPrev)
        output = paddle.squeeze(output, axis=1)
        # 使用全连接层将输出转换为最终输出
        output = self.fc(output)
        # 返回输出和新的隐藏状态
        return output, state
```