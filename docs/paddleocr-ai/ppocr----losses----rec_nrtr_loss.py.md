# `.\PaddleOCR\ppocr\losses\rec_nrtr_loss.py`

```py
import paddle
from paddle import nn
import paddle.nn.functional as F

# 定义一个自定义的损失函数类 NRTRLoss，继承自 nn.Layer
class NRTRLoss(nn.Layer):
    # 初始化函数，接受平滑参数 smoothing 和忽略索引 ignore_index
    def __init__(self, smoothing=True, ignore_index=0, **kwargs):
        super(NRTRLoss, self).__init__()
        # 如果忽略索引大于等于0且不使用平滑，则使用交叉熵损失函数
        if ignore_index >= 0 and not smoothing:
            self.loss_func = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=ignore_index)
        self.smoothing = smoothing

    # 前向传播函数，接受预测值 pred 和批量数据 batch
    def forward(self, pred, batch):
        # 获取批量数据中第二个元素的最大值
        max_len = batch[2].max()
        # 从批量数据中获取目标值 tgt
        tgt = batch[1][:, 1:2 + max_len]
        # 将预测值和目标值 reshape 成二维数组
        pred = pred.reshape([-1, pred.shape[2])
        tgt = tgt.reshape([-1])
        # 如果使用平滑
        if self.smoothing:
            # 设置平滑参数和类别数
            eps = 0.1
            n_class = pred.shape[1]
            # 将目标值转换为 one-hot 编码
            one_hot = F.one_hot(tgt, pred.shape[1])
            # 对 one-hot 编码进行平滑处理
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            # 计算对数概率和非填充掩码
            log_prb = F.log_softmax(pred, axis=1)
            non_pad_mask = paddle.not_equal(
                tgt, paddle.zeros(
                    tgt.shape, dtype=tgt.dtype))
            # 计算损失值
            loss = -(one_hot * log_prb).sum(axis=1)
            loss = loss.masked_select(non_pad_mask).mean()
        else:
            # 如果不使用平滑，则直接计算损失值
            loss = self.loss_func(pred, tgt)
        # 返回损失值
        return {'loss': loss}
```