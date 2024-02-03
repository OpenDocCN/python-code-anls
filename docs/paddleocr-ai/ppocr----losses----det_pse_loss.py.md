# `.\PaddleOCR\ppocr\losses\det_pse_loss.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本:
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以了解具体语言规定的权限和限制。
"""
# 代码来源于:
# https://github.com/whai362/PSENet/blob/python3/models/head/psenet_head.py

import paddle
from paddle import nn
from paddle.nn import functional as F
import numpy as np
from ppocr.utils.iou import iou

# 定义 PSE Loss 类
class PSELoss(nn.Layer):
    def __init__(self,
                 alpha,
                 ohem_ratio=3,
                 kernel_sample_mask='pred',
                 reduction='sum',
                 eps=1e-6,
                 **kwargs):
        """实现 PSE Loss.
        """
        super(PSELoss, self).__init__()
        # 断言 reduction 参数只能是 'sum', 'mean', 'none' 中的一个
        assert reduction in ['sum', 'mean', 'none']
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.kernel_sample_mask = kernel_sample_mask
        self.reduction = reduction
        self.eps = eps
    # 前向传播函数，计算损失值
    def forward(self, outputs, labels):
        # 获取模型输出中的预测值
        predicts = outputs['maps']
        # 对预测值进行插值，将尺寸放大4倍
        predicts = F.interpolate(predicts, scale_factor=4)

        # 获取文本预测值和内核预测值
        texts = predicts[:, 0, :, :]
        kernels = predicts[:, 1:, :, :]
        # 获取标签中的文本真值、内核真值和训练掩码
        gt_texts, gt_kernels, training_masks = labels[1:]

        # 计算文本损失
        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.dice_loss(texts, gt_texts, selected_masks)
        iou_text = iou((texts > 0).astype('int64'), gt_texts, training_masks, reduce=False)
        losses = dict(loss_text=loss_text, iou_text=iou_text)

        # 计算内核损失
        loss_kernels = []
        if self.kernel_sample_mask == 'gt':
            selected_masks = gt_texts * training_masks
        elif self.kernel_sample_mask == 'pred':
            selected_masks = (F.sigmoid(texts) > 0.5).astype('float32') * training_masks

        for i in range(kernels.shape[1]):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = paddle.mean(paddle.stack(loss_kernels, axis=1), axis=1)
        iou_kernel = iou((kernels[:, -1, :, :] > 0).astype('int64'), gt_kernels[:, -1, :, :], training_masks * gt_texts, reduce=False)
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))
        loss = self.alpha * loss_text + (1 - self.alpha) * loss_kernels
        losses['loss'] = loss
        # 根据reduction参数对损失进行汇总
        if self.reduction == 'sum':
            losses = {x: paddle.sum(v) for x, v in losses.items()}
        elif self.reduction == 'mean':
            losses = {x: paddle.mean(v) for x, v in losses.items()}
        return losses
    # 计算 Dice Loss，用于评估模型预测结果与真实标签之间的相似度
    def dice_loss(self, input, target, mask):
        # 对输入进行 sigmoid 激活函数处理
        input = F.sigmoid(input)

        # 将输入、目标和掩码重塑为二维数组
        input = input.reshape([input.shape[0], -1])
        target = target.reshape([target.shape[0], -1])
        mask = mask.reshape([mask.shape[0], -1])

        # 将输入、目标按掩码进行元素级相乘
        input = input * mask
        target = target * mask

        # 计算 Dice Loss 中的各项指标
        a = paddle.sum(input * target, 1)
        b = paddle.sum(input * input, 1) + self.eps
        c = paddle.sum(target * target, 1) + self.eps
        d = (2 * a) / (b + c)

        # 返回 Dice Loss 结果
        return 1 - d

    # 使用 Online Hard Example Mining (OHEM) 策略选择样本
    def ohem_single(self, score, gt_text, training_mask, ohem_ratio=3):
        # 计算正样本数量
        pos_num = int(paddle.sum((gt_text > 0.5).astype('float32'))) - int(
            paddle.sum(
                paddle.logical_and((gt_text > 0.5), (training_mask <= 0.5))
                .astype('float32')))

        # 如果没有正样本，则直接返回训练掩码
        if pos_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(
                [1, selected_mask.shape[0], selected_mask.shape[1]]).astype(
                    'float32')
            return selected_mask

        # 计算负样本数量，并根据 OHEM 比例选择样本
        neg_num = int(paddle.sum((gt_text <= 0.5).astype('float32')))
        neg_num = int(min(pos_num * ohem_ratio, neg_num))

        # 如果没有负样本，则直接返回训练掩码
        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(
                [1, selected_mask.shape[0], selected_mask.shape[1]]).astype(
                    'float32')
            return selected_mask

        # 计算负样本得分，并根据阈值选择样本
        neg_score = paddle.masked_select(score, gt_text <= 0.5)
        neg_score_sorted = paddle.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]

        # 根据阈值和条件选择样本，并返回结果
        selected_mask = paddle.logical_and(
            paddle.logical_or((score >= threshold), (gt_text > 0.5)),
            (training_mask > 0.5))
        selected_mask = selected_mask.reshape(
            [1, selected_mask.shape[0], selected_mask.shape[1]]).astype(
                'float32')
        return selected_mask
    # 在训练过程中使用在线困难样本挖掘（OHEM）方法，选择困难样本的标签
    def ohem_batch(self, scores, gt_texts, training_masks, ohem_ratio=3):
        # 存储选定的困难样本的掩码
        selected_masks = []
        # 遍历每个样本
        for i in range(scores.shape[0]):
            # 对每个样本应用 OHEM 方法，选择困难样本的标签
            selected_masks.append(
                self.ohem_single(scores[i, :, :], gt_texts[i, :, :],
                                 training_masks[i, :, :], ohem_ratio))

        # 将选定的困难样本的掩码连接在一起，并转换为 float32 类型
        selected_masks = paddle.concat(selected_masks, 0).astype('float32')
        # 返回选定的困难样本的掩码
        return selected_masks
```