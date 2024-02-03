# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\losses\basic_loss.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 14:39
# @Author  : zhoujun
# 导入 paddle 库
import paddle
# 导入 paddle 中的神经网络模块
import paddle.nn as nn

# 定义 BalancedCrossEntropyLoss 类，继承自 nn.Layer
class BalanceCrossEntropyLoss(nn.Layer):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    '''
    # 初始化函数
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    # 前向传播函数
    def forward(self,
                pred: paddle.Tensor,
                gt: paddle.Tensor,
                mask: paddle.Tensor,
                return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        # 计算正样本
        positive = (gt * mask)
        # 计算负样本
        negative = ((1 - gt) * mask)
        # 计算正样本数量
        positive_count = int(positive.sum())
        # 计算负样本数量，限制最大数量为正样本数量乘以负样本比例
        negative_count = min(
            int(negative.sum()), int(positive_count * self.negative_ratio))
        # 计算二分类交叉熵损失
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        # 计算正样本损失
        positive_loss = loss * positive
        # 计算负样本损失
        negative_loss = loss * negative
        # 选取负样本损失中的前 negative_count 个值
        negative_loss, _ = negative_loss.reshape([-1]).topk(negative_count)

        # 计算平衡损失
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_count + negative_count + self.eps)

        # 如果需要返回原始损失，同时返回平衡损失和原始损失
        if return_origin:
            return balance_loss, loss
        return balance_loss

# 定义 DiceLoss 类，继承自 nn.Layer
class DiceLoss(nn.Layer):
    '''
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    '''
    # 初始化函数
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps
    # 定义前向传播函数，接受预测值、真实值、掩码和权重作为输入
    def forward(self, pred: paddle.Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        # 调用_compute函数计算损失
        return self._compute(pred, gt, mask, weights)

    # 定义内部计算函数，接受预测值、真实值、掩码和权重作为输入
    def _compute(self, pred, gt, mask, weights):
        # 如果预测值的维度为4，则取第一个通道的值
        if len(pred.shape) == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        # 断言预测值、真实值和掩码的形状相同
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        # 如果存在权重，则将掩码乘以权重
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        # 计算交集
        intersection = (pred * gt * mask).sum()
        # 计算并集
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        # 计算损失
        loss = 1 - 2.0 * intersection / union
        # 断言损失值小于等于1
        assert loss <= 1
        # 返回损失值
        return loss
# 定义一个自定义的损失函数类 MaskL1Loss，继承自 nn.Layer
class MaskL1Loss(nn.Layer):
    # 初始化函数，设置损失函数的参数 eps，默认值为 1e-6
    def __init__(self, eps=1e-6):
        # 调用父类的初始化函数
        super(MaskL1Loss, self).__init__()
        # 将参数 eps 赋给实例变量 self.eps
        self.eps = eps

    # 前向传播函数，接收预测值 pred、真实值 gt 和掩码 mask 作为输入
    def forward(self, pred: paddle.Tensor, gt, mask):
        # 计算 L1 损失，乘以掩码 mask，然后求和并除以掩码 mask 的和再加上一个很小的值 eps，防止除零错误
        loss = (paddle.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        # 返回计算得到的损失值
        return loss
```