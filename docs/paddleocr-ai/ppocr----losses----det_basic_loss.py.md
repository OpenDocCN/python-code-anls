# `.\PaddleOCR\ppocr\losses\det_basic_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以了解具体语言规定的权限和限制
"""
# 引用来源
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/losses/basic_loss.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle
from paddle import nn
import paddle.nn.functional as F

# 定义一个名为 BalanceLoss 的类，继承自 nn.Layer
class BalanceLoss(nn.Layer):
    # 初始化 BalanceLoss 类，设置默认参数和可选参数
    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 negative_ratio=3,
                 return_origin=False,
                 eps=1e-6,
                 **kwargs):
        """
               The BalanceLoss for Differentiable Binarization text detection
               args:
                   balance_loss (bool): whether balance loss or not, default is True
                   main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
                       'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
                   negative_ratio (int|float): float, default is 3.
                   return_origin (bool): whether return unbalanced loss or not, default is False.
                   eps (float): default is 1e-6.
               """
        # 调用父类的构造函数
        super(BalanceLoss, self).__init__()
        # 设置类的属性值
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

        # 根据主要损失类型选择相应的损失函数
        if self.main_loss_type == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.main_loss_type == "Euclidean":
            self.loss = nn.MSELoss()
        elif self.main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == "BCELoss":
            self.loss = BCELoss(reduction='none')
        elif self.main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(self.eps)
        else:
            # 如果主要损失类型不在指定范围内，抛出异常
            loss_type = [
                'CrossEntropy', 'DiceLoss', 'Euclidean', 'BCELoss', 'MaskL1Loss'
            ]
            raise Exception(
                "main_loss_type in BalanceLoss() can only be one of {}".format(
                    loss_type))
    def forward(self, pred, gt, mask=None):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            pred (variable): predicted feature maps.
            gt (variable): ground truth feature maps.
            mask (variable): masked maps.
        return: (variable) balanced loss
        """
        # 计算正样本：预测值和掩码相乘
        positive = gt * mask
        # 计算负样本：1减去真实值再和掩码相乘
        negative = (1 - gt) * mask

        # 计算正样本数量
        positive_count = int(positive.sum())
        # 计算负样本数量，限制最大数量为正样本数量乘以负样本比例
        negative_count = int(
            min(negative.sum(), positive_count * self.negative_ratio))
        # 计算损失值
        loss = self.loss(pred, gt, mask=mask)

        # 如果不需要平衡损失，则直接返回损失值
        if not self.balance_loss:
            return loss

        # 计算正样本损失
        positive_loss = positive * loss
        # 计算负样本损失
        negative_loss = negative * loss
        # 将负样本损失重塑为一维数组
        negative_loss = paddle.reshape(negative_loss, shape=[-1])
        # 如果存在负样本，则按损失值降序排序并取前负样本数量个
        if negative_count > 0:
            sort_loss = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            # negative_loss, _ = paddle.topk(negative_loss, k=negative_count_int)
            # 计算平衡损失
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps)
        else:
            # 如果不存在负样本，则只计算正样本损失
            balance_loss = positive_loss.sum() / (positive_count + self.eps)
        # 如果需要返回原始损失值，则返回平衡损失和原始损失
        if self.return_origin:
            return balance_loss, loss

        # 返回平衡损失
        return balance_loss
class DiceLoss(nn.Layer):
    def __init__(self, eps=1e-6):
        # 初始化 DiceLoss 类，设置 eps 参数
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        """
        DiceLoss function.
        """

        # 断言预测值、真实值和掩码的形状相同
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            # 如果权重不为 None，则断言权重和掩码的形状相同
            assert weights.shape == mask.shape
            mask = weights * mask
        # 计算交集
        intersection = paddle.sum(pred * gt * mask)

        # 计算并集
        union = paddle.sum(pred * mask) + paddle.sum(gt * mask) + self.eps
        # 计算 Dice Loss
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Layer):
    def __init__(self, eps=1e-6):
        # 初始化 MaskL1Loss 类，设置 eps 参数
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        Mask L1 Loss
        """
        # 计算 Mask L1 Loss
        loss = (paddle.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        loss = paddle.mean(loss)
        return loss


class BCELoss(nn.Layer):
    def __init__(self, reduction='mean'):
        # 初始化 BCELoss 类，设置 reduction 参数
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        # 计算二元交叉熵损失
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss
```