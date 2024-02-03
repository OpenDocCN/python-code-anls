# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\losses\DB_loss.py`

```
import paddle
from models.losses.basic_loss import BalanceCrossEntropyLoss, MaskL1Loss, DiceLoss

# 定义 DBLoss 类，继承自 paddle.nn.Layer
class DBLoss(paddle.nn.Layer):
    def __init__(self,
                 alpha=1.0,
                 beta=10,
                 ohem_ratio=3,
                 reduction='mean',
                 eps=1e-06):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        # 确保 reduction 参数只能是 'mean' 或 'sum'
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        # 初始化 BalanceCrossEntropyLoss 对象，negative_ratio 参数为 ohem_ratio
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        # 初始化 DiceLoss 对象，eps 参数为 eps
        self.dice_loss = DiceLoss(eps=eps)
        # 初始化 MaskL1Loss 对象，eps 参数为 eps
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    # 前向传播函数
    def forward(self, pred, batch):
        # 获取预测结果中的 shrink_maps、threshold_maps、binary_maps
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]
        # 计算 shrink_maps 的损失
        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'],
                                         batch['shrink_mask'])
        # 计算 threshold_maps 的损失
        loss_threshold_maps = self.l1_loss(
            threshold_maps, batch['threshold_map'], batch['threshold_mask'])
        # 初始化 metrics 字典，记录损失值
        metrics = dict(
            loss_shrink_maps=loss_shrink_maps,
            loss_threshold_maps=loss_threshold_maps)
        # 如果预测结果中有多于两个通道的数据
        if pred.shape[1] > 2:
            # 计算 binary_maps 的损失
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'],
                                              batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps
            # 计算总损失值
            loss_all = (self.alpha * loss_shrink_maps + self.beta *
                        loss_threshold_maps + loss_binary_maps)
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics
```