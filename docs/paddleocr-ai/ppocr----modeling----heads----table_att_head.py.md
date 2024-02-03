# `.\PaddleOCR\ppocr\modeling\heads\table_att_head.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
from paddle import ParamAttr
import paddle.nn.functional as F
import numpy as np

# 导入自定义的 AttentionGRUCell 模块
from .rec_att_head import AttentionGRUCell

# 定义函数，根据 L2 正则化系数和输入维度 k 获取参数和偏置属性
def get_para_bias_attr(l2_decay, k):
    if l2_decay > 0:
        # 如果 L2 正则化系数大于 0，则使用 L2 正则化
        regularizer = paddle.regularizer.L2Decay(l2_decay)
        stdv = 1.0 / math.sqrt(k * 1.0)
        initializer = nn.initializer.Uniform(-stdv, stdv)
    else:
        regularizer = None
        initializer = None
    # 定义权重属性和偏置属性
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]

# 定义 TableAttentionHead 类，继承自 nn.Layer
class TableAttentionHead(nn.Layer):
    # 初始化函数，设置各种参数
    def __init__(self,
                 in_channels,
                 hidden_size,
                 in_max_len=488,
                 max_text_length=800,
                 out_channels=30,
                 loc_reg_num=4,
                 **kwargs):
        # 调用父类的初始化函数
        super(TableAttentionHead, self).__init__()
        # 设置输入通道数
        self.input_size = in_channels[-1]
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置输出通道数
        self.out_channels = out_channels
        # 设置最大文本长度
        self.max_text_length = max_text_length

        # 创建结构注意力单元
        self.structure_attention_cell = AttentionGRUCell(
            self.input_size, hidden_size, self.out_channels, use_gru=False)
        # 创建结构生成器
        self.structure_generator = nn.Linear(hidden_size, self.out_channels)
        # 设置输入最大长度
        self.in_max_len = in_max_len

        # 根据输入最大长度选择不同的线性转换层
        if self.in_max_len == 640:
            self.loc_fea_trans = nn.Linear(400, self.max_text_length + 1)
        elif self.in_max_len == 800:
            self.loc_fea_trans = nn.Linear(625, self.max_text_length + 1)
        else:
            self.loc_fea_trans = nn.Linear(256, self.max_text_length + 1)
        # 创建位置生成器
        self.loc_generator = nn.Linear(self.input_size + hidden_size,
                                       loc_reg_num)

    # 将字符转换为独热编码
    def _char_to_onehot(self, input_char, onehot_dim):
        # 使用 PyTorch 的 F.one_hot 函数将字符转换为独热编码
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot
class SLAHead(nn.Layer):
    # 定义 SLAHead 类，继承自 nn.Layer

    def forward(self, inputs, targets=None):
        # 定义 forward 方法，接受输入和目标参数

        fea = inputs[-1]
        # 获取输入的最后一个元素作为特征

        batch_size = fea.shape[0]
        # 获取特征的批量大小

        # reshape
        fea = paddle.reshape(fea, [fea.shape[0], fea.shape[1], -1])
        # 重新塑形特征张量

        fea = fea.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        # 转置特征张量的维度顺序

        hidden = paddle.zeros((batch_size, self.hidden_size))
        # 创建全零张量作为隐藏状态

        structure_preds = paddle.zeros(
            (batch_size, self.max_text_length + 1, self.num_embeddings))
        # 创建全零张量作为结构预测

        loc_preds = paddle.zeros(
            (batch_size, self.max_text_length + 1, self.loc_reg_num))
        # 创建全零张量作为位置预测

        structure_preds.stop_gradient = True
        loc_preds.stop_gradient = True
        # 设置结构预测和位置预测不需要梯度

        if self.training and targets is not None:
            # 如果是训练模式且目标不为空

            structure = targets[0]
            # 获取目标的第一个元素作为结构

            for i in range(self.max_text_length + 1):
                # 遍历最大文本长度加一次

                hidden, structure_step, loc_step = self._decode(structure[:, i],
                                                                fea, hidden)
                # 解码结构和位置信息

                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step
                # 更新结构和位置预测

        else:
            pre_chars = paddle.zeros(shape=[batch_size], dtype="int32")
            # 创建全零张量作为先前字符

            max_text_length = paddle.to_tensor(self.max_text_length)
            # 将最大文本长度转换为张量

            # for export
            loc_step, structure_step = None, None
            # 初始化位置步骤和结构步骤为 None

            for i in range(max_text_length + 1):
                # 遍历最大文本长度加一次

                hidden, structure_step, loc_step = self._decode(pre_chars, fea,
                                                                hidden)
                # 解码先前字符、特征和隐藏状态

                pre_chars = structure_step.argmax(axis=1, dtype="int32")
                # 更新先前字符为结构步骤的最大值索引

                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step
                # 更新结构和位置预测

        if not self.training:
            # 如果不是训练模式

            structure_preds = F.softmax(structure_preds)
            # 对结构预测进行 softmax 操作

        return {'structure_probs': structure_preds, 'loc_preds': loc_preds}
        # 返回结构概率和位置预测
    # 解码器，用于预测每一步的表格标签和坐标
    def _decode(self, pre_chars, features, hidden):
        """
        Predict table label and coordinates for each step
        @param pre_chars: Table label in previous step
        @param features:
        @param hidden: hidden status in previous step
        @return:
        """
        # 将前一个字符的表格标签转换为嵌入特征
        emb_feature = self.emb(pre_chars)
        # 使用结构注意力单元预测输出和隐藏状态
        # 输出形状为 b * self.hidden_size
        (output, hidden), alpha = self.structure_attention_cell(
            hidden, features, emb_feature)

        # 生成结构信息
        structure_step = self.structure_generator(output)
        # 生成位置信息
        loc_step = self.loc_generator(output)
        return hidden, structure_step, loc_step

    # 将字符转换为 one-hot 编码
    def _char_to_onehot(self, input_char):
        # 使用 F.one_hot 将输入字符转换为 one-hot 编码
        input_ont_hot = F.one_hot(input_char, self.num_embeddings)
        return input_ont_hot
```