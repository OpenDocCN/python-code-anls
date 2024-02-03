# `.\PaddleOCR\ppocr\modeling\heads\rec_att_head.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以了解具体语言规定的权限和限制。

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

# 定义一个注意力头部的类
class AttentionHead(nn.Layer):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        # 初始化注意力机制的 GRU 单元
        self.attention_cell = AttentionGRUCell(
            in_channels, hidden_size, out_channels, use_gru=False)
        # 初始化生成器，用于将隐藏状态映射到输出类别
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
        # 设置最大步长为 batch_max_length
        num_steps = batch_max_length

        # 初始化隐藏状态为全零
        hidden = paddle.zeros((batch_size, self.hidden_size))
        # 存储每个时间步的输出隐藏状态
        output_hiddens = []

        # 如果存在目标序列
        if targets is not None:
            # 遍历每个时间步
            for i in range(num_steps):
                # 将目标序列转换为 one-hot 编码
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes)
                # 使用注意力机制计算输出和隐藏状态
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs,
                                                               char_onehots)
                # 将输出隐藏状态添加到列表中
                output_hiddens.append(paddle.unsqueeze(outputs, axis=1))
            # 拼接所有时间步的输出隐藏状态
            output = paddle.concat(output_hiddens, axis=1)
            # 通过生成器获取预测概率
            probs = self.generator(output)
        else:
            # 如果不存在目标序列，则初始化相关变量
            targets = paddle.zeros(shape=[batch_size], dtype="int32")
            probs = None
            char_onehots = None
            outputs = None
            alpha = None

            # 遍历每个时间步
            for i in range(num_steps):
                # 将目标序列转换为 one-hot 编码
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                # 使用注意力机制计算输出和隐藏状态
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs,
                                                               char_onehots)
                # 通过生成器获取当前时间步的预测概率
                probs_step = self.generator(outputs)
                # 如果预测概率为空，则初始化为当前时间步的概率
                if probs is None:
                    probs = paddle.unsqueeze(probs_step, axis=1)
                else:
                    # 否则拼接当前时间步的概率到之前的概率中
                    probs = paddle.concat(
                        [probs, paddle.unsqueeze(
                            probs_step, axis=1)], axis=1)
                # 获取下一个时间步的输入
                next_input = probs_step.argmax(axis=1)
                targets = next_input
        # 如果不处于训练模式，则对预测概率进行 softmax 处理
        if not self.training:
            probs = paddle.nn.functional.softmax(probs, axis=2)
        # 返回预测概率
        return probs
class AttentionGRUCell(nn.Layer):
    # 定义注意力机制的 GRU 单元
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        # 初始化函数
        super(AttentionGRUCell, self).__init__()
        # 输入到隐藏层的线性变换
        self.i2h = nn.Linear(input_size, hidden_size, bias_attr=False)
        # 隐藏层到隐藏层的线性变换
        self.h2h = nn.Linear(hidden_size, hidden_size)
        # 用于计算注意力分数的线性变换
        self.score = nn.Linear(hidden_size, 1, bias_attr=False)

        # 创建 GRU 单元
        self.rnn = nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size)

        # 隐藏层的大小
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # 前一个隐藏状态的线性变换
        batch_H_proj = self.i2h(batch_H)
        # 前一个隐藏状态的线性变换并增加一个维度
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden), axis=1)

        # 将两个线性变换结果相加并经过激活函数
        res = paddle.add(batch_H_proj, prev_hidden_proj)
        res = paddle.tanh(res)
        # 计算注意力分数
        e = self.score(res)

        # 计算注意力权重
        alpha = F.softmax(e, axis=1)
        alpha = paddle.transpose(alpha, [0, 2, 1])
        # 计算上下文向量
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        concat_context = paddle.concat([context, char_onehots], 1)

        # 更新当前隐藏状态
        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha


class AttentionLSTM(nn.Layer):
    # 定义注意力机制的 LSTM 模型
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        # 初始化函数
        super(AttentionLSTM, self).__init__()
        # 输入通道数
        self.input_size = in_channels
        # 隐藏状态大小
        self.hidden_size = hidden_size
        # 输出类别数
        self.num_classes = out_channels

        # 创建注意力机制的 LSTM 单元
        self.attention_cell = AttentionLSTMCell(
            in_channels, hidden_size, out_channels, use_gru=False)
        # 生成器，用于输出预测结果
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        # 将字符转换为 one-hot 编码
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot
    # 前向传播函数，接受输入和目标序列，返回预测概率
    def forward(self, inputs, targets=None, batch_max_length=25):
        # 获取输入的批量大小
        batch_size = inputs.shape[0]
        # 设置最大步长
        num_steps = batch_max_length

        # 初始化隐藏状态
        hidden = (paddle.zeros((batch_size, self.hidden_size)), paddle.zeros(
            (batch_size, self.hidden_size)))
        # 存储每个时间步的隐藏状态
        output_hiddens = []

        # 如果存在目标序列
        if targets is not None:
            # 遍历每个时间步
            for i in range(num_steps):
                # 将目标字符转换为独热向量
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes)
                # 使用注意力机制计算隐藏状态和注意力权重
                hidden, alpha = self.attention_cell(hidden, inputs,
                                                    char_onehots)

                # 更新隐藏状态
                hidden = (hidden[1][0], hidden[1][1])
                # 将隐藏状态添加到输出列表中
                output_hiddens.append(paddle.unsqueeze(hidden[0], axis=1))
            # 拼接所有隐藏状态
            output = paddle.concat(output_hiddens, axis=1)
            # 生成预测概率
            probs = self.generator(output)

        # 如果不存在目标序列
        else:
            # 初始化目标序列为全零
            targets = paddle.zeros(shape=[batch_size], dtype="int32")
            probs = None
            char_onehots = None
            alpha = None

            # 遍历每个时间步
            for i in range(num_steps):
                # 将目标字符转换为独热向量
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                # 使用注意力机制计算隐藏状态和注意力权重
                hidden, alpha = self.attention_cell(hidden, inputs,
                                                    char_onehots)
                # 生成当前时间步的预测概率
                probs_step = self.generator(hidden[0])
                # 更新隐藏状态
                hidden = (hidden[1][0], hidden[1][1])
                # 将当前时间步的预测概率添加到总概率中
                if probs is None:
                    probs = paddle.unsqueeze(probs_step, axis=1)
                else:
                    probs = paddle.concat(
                        [probs, paddle.unsqueeze(
                            probs_step, axis=1)], axis=1)

                # 获取下一个时间步的输入
                next_input = probs_step.argmax(axis=1)

                targets = next_input
        # 如果不处于训练模式，则对概率进行softmax处理
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