# `.\PaddleOCR\ppocr\losses\rec_can_loss.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码参考自:
# https://github.com/LBH1024/CAN/models/can.py

import paddle
import paddle.nn as nn
import numpy as np

# 定义 CANLoss 类，包含两部分:
# word_average_loss: 符号的平均准确性
# counting_loss: 每个符号的计数损失
class CANLoss(nn.Layer):

    def __init__(self):
        super(CANLoss, self).__init__()

        # 是否使用标签掩码
        self.use_label_mask = False
        # 输出通道数
        self.out_channel = 111
        # 交叉熵损失函数，如果使用标签掩码则设置为 'none'，否则设置为默认值
        self.cross = nn.CrossEntropyLoss(
            reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        # 平滑 L1 损失函数，计算均值
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')
        # 比率设为 16
        self.ratio = 16
    # 前向传播函数，接收模型预测结果和批次数据作为输入
    def forward(self, preds, batch):
        # 获取模型预测的单词概率
        word_probs = preds[0]
        # 获取模型预测的计数结果
        counting_preds = preds[1]
        counting_preds1 = preds[2]
        counting_preds2 = preds[3]
        # 获取批次数据中的标签
        labels = batch[2]
        # 获取批次数据中的标签掩码
        labels_mask = batch[3]
        # 生成计数标签
        counting_labels = gen_counting_label(labels, self.out_channel, True)
        # 计算计数损失
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) \
                        + self.counting_loss(counting_preds, counting_labels)

        # 计算单词损失
        word_loss = self.cross(
            paddle.reshape(word_probs, [-1, word_probs.shape[-1]]),
            paddle.reshape(labels, [-1]))
        # 计算单词平均损失
        word_average_loss = paddle.sum(
            paddle.reshape(word_loss * labels_mask, [-1])) / (
                paddle.sum(labels_mask) + 1e-10
            ) if self.use_label_mask else word_loss
        # 计算总损失
        loss = word_average_loss + counting_loss
        # 返回损失值
        return {'loss': loss}
# 生成计数标签，统计每个样本中每个类别出现的次数
def gen_counting_label(labels, channel, tag):
    # 获取标签的形状
    b, t = labels.shape
    # 创建一个全零数组，用于存储计数结果
    counting_labels = np.zeros([b, channel])

    # 根据标志位选择需要忽略的类别
    if tag:
        ignore = [0, 1, 107, 108, 109, 110]
    else:
        ignore = []
    # 遍历每个样本和每个时间步
    for i in range(b):
        for j in range(t):
            # 获取当前标签值
            k = labels[i][j]
            # 如果标签值在忽略列表中，则跳过
            if k in ignore:
                continue
            else:
                # 否则在计数数组中对应位置加一
                counting_labels[i][k] += 1
    # 将计数数组转换为张量，并指定数据类型为 float32
    counting_labels = paddle.to_tensor(counting_labels, dtype='float32')
    # 返回计数结果张量
    return counting_labels
```