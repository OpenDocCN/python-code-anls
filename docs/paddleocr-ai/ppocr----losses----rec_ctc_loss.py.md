# `.\PaddleOCR\ppocr\losses\rec_ctc_loss.py`

```
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

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn

# 定义一个 CTCLoss 类，继承自 nn.Layer
class CTCLoss(nn.Layer):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        # 初始化 CTC 损失函数，设置空白标签为 0，不进行缩减
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    # 前向传播函数
    def forward(self, predicts, batch):
        # 如果 predicts 是列表或元组，则取最后一个元素
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        # 调整预测数据的维度顺序
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        # 创建预测长度张量
        preds_lengths = paddle.to_tensor(
            [N] * B, dtype='int64', place=paddle.CPUPlace())
        # 获取标签数据并转换为 int32 类型
        labels = batch[1].astype("int32")
        # 获取标签长度并转换为 int64 类型
        label_lengths = batch[2].astype('int64')
        # 计算损失值
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        # 如果使用焦点损失
        if self.use_focal_loss:
            # 计算权重
            weight = paddle.exp(-loss)
            weight = paddle.subtract(paddle.to_tensor([1.0]), weight)
            weight = paddle.square(weight)
            loss = paddle.multiply(loss, weight)
        # 计算平均损失值
        loss = loss.mean()
        # 返回损失值
        return {'loss': loss}
```