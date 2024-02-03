# `.\PaddleOCR\ppocr\modeling\heads\rec_spin_att_head.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

"""
此代码参考自：
https://github.com/hikopensource/DAVAR-Lab-OCR/davarocr/davar_rcg/models/sequence_heads/att_head.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 定义 SPINAttentionHead 类，继承自 nn.Layer
class SPINAttentionHead(nn.Layer):
    # 初始化函数，接受输入通道数、输出通道数和隐藏层大小等参数
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(SPINAttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        # 初始化注意力机制的 LSTM 单元
        self.attention_cell = AttentionLSTMCell(
            in_channels, hidden_size, out_channels, use_gru=False)
        # 初始化生成器，将隐藏层输出映射到输出通道数
        self.generator = nn.Linear(hidden_size, out_channels)

    # 将字符转换为 one-hot 编码
    def _char_to_onehot(self, input_char, onehot_dim):
        # 使用 paddle 的 one_hot 函数将字符转换为 one-hot 编码
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot
    # 前向传播函数，接受输入和目标序列，返回预测概率
    def forward(self, inputs, targets=None, batch_max_length=25):
        # 获取输入的批量大小
        batch_size = paddle.shape(inputs)[0]
        # 设置序列的最大长度
        num_steps = batch_max_length + 1 # +1 for [sos] at end of sentence

        # 初始化隐藏状态
        hidden = (paddle.zeros((batch_size, self.hidden_size)),
                    paddle.zeros((batch_size, self.hidden_size)))
        # 存储每个时间步的输出隐藏状态
        output_hiddens = []
        
        # 训练模式下
        if self.training: # for train
            # 获取目标序列
            targets = targets[0]
            # 遍历每个时间步
            for i in range(num_steps):
                # 将字符转换为独热编码
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes)
                # 使用注意力机制计算输出和隐藏状态
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs,
                                                               char_onehots)
                # 将输出隐藏状态添加到列表中
                output_hiddens.append(paddle.unsqueeze(outputs, axis=1))
            # 拼接所有时间步的输出隐藏状态
            output = paddle.concat(output_hiddens, axis=1)
            # 生成预测概率
            probs = self.generator(output)        
        else:
            # 推理模式下，初始化目标序列、概率、字符独热编码、输出和注意力权重
            targets = paddle.zeros(shape=[batch_size], dtype="int32")
            probs = None
            char_onehots = None
            outputs = None
            alpha = None

            # 遍历每个时间步
            for i in range(num_steps):
                # 将字符转换为独热编码
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                # 使用注意力机制计算输出和隐藏状态
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs,
                                                               char_onehots)
                # 生成当前时间步的预测概率
                probs_step = self.generator(outputs)
                # 拼接预测概率
                if probs is None:
                    probs = paddle.unsqueeze(probs_step, axis=1)
                else:
                    probs = paddle.concat(
                        [probs, paddle.unsqueeze(
                            probs_step, axis=1)], axis=1)
                # 获取下一个时间步的输入
                next_input = probs_step.argmax(axis=1)
                targets = next_input
        # 如果不是训练模式，对预测概率进行 softmax 处理
        if not self.training:
            probs = paddle.nn.functional.softmax(probs, axis=2)
        # 返回预测概率
        return probs
class AttentionLSTMCell(nn.Layer):
    # 定义注意力机制的 LSTM 单元
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        # 初始化函数
        super(AttentionLSTMCell, self).__init__()
        # 输入到隐藏层的线性变换
        self.i2h = nn.Linear(input_size, hidden_size, bias_attr=False)
        # 隐藏层到隐藏层的线性变换
        self.h2h = nn.Linear(hidden_size, hidden_size)
        # 计算注意力分数的线性变换
        self.score = nn.Linear(hidden_size, 1, bias_attr=False)
        # 根据是否使用 GRU 初始化 LSTM 或 GRU 单元
        if not use_gru:
            self.rnn = nn.LSTMCell(
                input_size=input_size + num_embeddings, hidden_size=hidden_size)
        else:
            self.rnn = nn.GRUCell(
                input_size=input_size + num_embeddings, hidden_size=hidden_size)

        # 隐藏层的大小
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # 前一个隐藏状态经过线性变换
        batch_H_proj = self.i2h(batch_H)
        # 前一个隐藏状态经过线性变换并增加维度
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden[0]), axis=1)
        # 将两个线性变换结果相加并经过激活函数
        res = paddle.add(batch_H_proj, prev_hidden_proj)
        res = paddle.tanh(res)
        # 计算注意力分数
        e = self.score(res)

        # 对注意力分数进行 softmax 归一化
        alpha = F.softmax(e, axis=1)
        # 调整 alpha 的维度
        alpha = paddle.transpose(alpha, [0, 2, 1])
        # 计算上下文向量
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        # 将上下文向量和字符的 one-hot 编码拼接起来
        concat_context = paddle.concat([context, char_onehots], 1)
        # 更新当前隐藏状态
        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha
```