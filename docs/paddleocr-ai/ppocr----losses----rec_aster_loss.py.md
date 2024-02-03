# `.\PaddleOCR\ppocr\losses\rec_aster_loss.py`

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
# 均按"原样"分发，不附带任何担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn

# 定义余弦嵌入损失函数类
class CosineEmbeddingLoss(nn.Layer):
    def __init__(self, margin=0.):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.epsilon = 1e-12

    # 前向传播函数
    def forward(self, x1, x2, target):
        # 计算余弦相似度
        similarity = paddle.sum(
            x1 * x2, axis=-1) / (paddle.norm(
                x1, axis=-1) * paddle.norm(
                    x2, axis=-1) + self.epsilon)
        # 创建一个全为1的张量
        one_list = paddle.full_like(target, fill_value=1)
        # 计算损失
        out = paddle.mean(
            paddle.where(
                paddle.equal(target, one_list), 1. - similarity,
                paddle.maximum(
                    paddle.zeros_like(similarity), similarity - self.margin)))

        return out

# 定义 AsterLoss 类
class AsterLoss(nn.Layer):
    # 初始化 AsterLoss 类的实例
    def __init__(self,
                 weight=None,  # 权重参数，默认为 None
                 size_average=True,  # 是否对损失进行平均，默认为 True
                 ignore_index=-100,  # 忽略的索引，默认为 -100
                 sequence_normalize=False,  # 是否对序列进行归一化，默认为 False
                 sample_normalize=True,  # 是否对样本进行归一化，默认为 True
                 **kwargs):  # 允许接收额外的关键字参数
        # 调用父类的构造函数
        super(AsterLoss, self).__init__()
        # 初始化权重
        self.weight = weight
        # 初始化是否对损失进行平均
        self.size_average = size_average
        # 初始化忽略的索引
        self.ignore_index = ignore_index
        # 初始化是否对序列进行归一化
        self.sequence_normalize = sequence_normalize
        # 初始化是否对样本进行归一化
        self.sample_normalize = sample_normalize
        # 初始化语义损失
        self.loss_sem = CosineEmbeddingLoss()
        # 是否使用余弦损失
        self.is_cosin_loss = True
        # 初始化重建损失函数
        self.loss_func_rec = nn.CrossEntropyLoss(weight=None, reduction='none')
    # 前向传播函数，计算损失并返回
    def forward(self, predicts, batch):
        # 获取目标值和标签长度
        targets = batch[1].astype("int64")
        label_lengths = batch[2].astype('int64')
        sem_target = batch[3].astype('float32')
        embedding_vectors = predicts['embedding_vectors']
        rec_pred = predicts['rec_pred']

        # 计算语义损失
        if not self.is_cosin_loss:
            sem_loss = paddle.sum(self.loss_sem(embedding_vectors, sem_target))
        else:
            label_target = paddle.ones([embedding_vectors.shape[0]])
            sem_loss = paddle.sum(
                self.loss_sem(embedding_vectors, sem_target, label_target))

        # 重构损失
        batch_size, def_max_length = targets.shape[0], targets.shape[1]

        # 创建掩码
        mask = paddle.zeros([batch_size, def_max_length])
        for i in range(batch_size):
            mask[i, :label_lengths[i]] = 1
        mask = paddle.cast(mask, "float32")
        max_length = max(label_lengths)
        assert max_length == rec_pred.shape[1]
        targets = targets[:, :max_length]
        mask = mask[:, :max_length]
        rec_pred = paddle.reshape(rec_pred, [-1, rec_pred.shape[2])
        input = nn.functional.log_softmax(rec_pred, axis=1)
        targets = paddle.reshape(targets, [-1, 1])
        mask = paddle.reshape(mask, [-1, 1])
        output = -paddle.index_sample(input, index=targets) * mask
        output = paddle.sum(output)
        
        # 序列归一化
        if self.sequence_normalize:
            output = output / paddle.sum(mask)
        
        # 样本归一化
        if self.sample_normalize:
            output = output / batch_size

        # 计算总损失
        loss = output + sem_loss * 0.1
        return {'loss': loss}
```