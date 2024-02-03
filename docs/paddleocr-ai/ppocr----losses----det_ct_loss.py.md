# `.\PaddleOCR\ppocr\losses\det_ct_loss.py`

```py
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
"""
# 代码来源：
# https://github.com/shengtao96/CentripetalText/tree/main/models/loss

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np

# 定义一个函数，用于在线困难样本挖掘
def ohem_single(score, gt_text, training_mask):
    # 在线困难样本挖掘

    # 计算正样本数量
    pos_num = int(paddle.sum(gt_text > 0.5)) - int(
        paddle.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # 如果没有正样本，则选择全部训练样本
        selected_mask = training_mask
        selected_mask = paddle.cast(
            selected_mask.reshape(
                (1, selected_mask.shape[0], selected_mask.shape[1])), "float32")
        return selected_mask

    # 计算负样本数量
    neg_num = int(paddle.sum((gt_text <= 0.5) & (training_mask > 0.5)))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        # 如果没有负样本，则选择全部训练样本
        selected_mask = training_mask
        selected_mask = paddle.cast(
            selected_mask.reshape(
                (1, selected_mask.shape[0], selected_mask.shape[1])), "float32")
        return selected_mask

    # 获取负样本的分数，并按照分数排序
    neg_score = score[(gt_text <= 0.5) & (training_mask > 0.5)]
    neg_score_sorted = paddle.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]
    # 创建一个布尔掩码，选择得分大于阈值或者真实文本大于0.5且训练掩码大于0.5的部分
    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    # 将选择的掩码重塑为形状为(1, selected_mask.shape[0], selected_mask.shape[1])的张量，并转换为float32类型
    selected_mask = paddle.cast(
        selected_mask.reshape((1, selected_mask.shape[0], selected_mask.shape[1])), "float32")
    # 返回选择的掩码
    return selected_mask
# 根据输入的分数、真实文本和训练掩码计算OHEM（Online Hard Example Mining）的选定掩码
def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(
            ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[
                i, :, :]))

    # 将选定掩码连接并转换为float32类型
    selected_masks = paddle.cast(paddle.concat(selected_masks, 0), "float32")
    return selected_masks


# 计算单个样本的IoU（Intersection over Union）
def iou_single(a, b, mask, n_class):
    EPS = 1e-6
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []

    # 计算每个类别的IoU
    for i in range(n_class):
        inter = paddle.cast(((a == i) & (b == i)), "float32")
        union = paddle.cast(((a == i) | (b == i)), "float32")

        miou.append(paddle.sum(inter) / (paddle.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


# 计算整个批次的IoU
def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = a.reshape((batch_size, -1))
    b = b.reshape((batch_size, -1))
    mask = mask.reshape((batch_size, -1))

    iou = paddle.zeros((batch_size, ), dtype="float32")
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = paddle.mean(iou)
    return iou


# 定义Dice Loss类
class DiceLoss(nn.Layer):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight
    # 定义一个前向传播函数，计算损失值
    def forward(self, input, target, mask, reduce=True):
        # 获取输入数据的批量大小
        batch_size = input.shape[0]
        # 对输入数据进行 sigmoid 激活函数处理，将数值缩放到 0-1 之间
        input = F.sigmoid(input)  # scale to 0-1

        # 将输入数据 reshape 成二维数组
        input = input.reshape((batch_size, -1))
        # 将目标数据 reshape 成二维数组，并转换为 float32 类型
        target = paddle.cast(target.reshape((batch_size, -1)), "float32")
        # 将掩码数据 reshape 成二维数组，并转换为 float32 类型
        mask = paddle.cast(mask.reshape((batch_size, -1)), "float32")

        # 对输入数据和目标数据进行掩码处理
        input = input * mask
        target = target * mask

        # 计算输入数据和目标数据的点积和
        a = paddle.sum(input * target, axis=1)
        # 计算输入数据的平方和，并加上一个小值 0.001 防止除零错误
        b = paddle.sum(input * input, axis=1) + 0.001
        # 计算目标数据的平方和，并加上一个小值 0.001 防止除零错误
        c = paddle.sum(target * target, axis=1) + 0.001
        # 计算损失值
        d = (2 * a) / (b + c)
        loss = 1 - d

        # 根据损失权重调整损失值
        loss = self.loss_weight * loss

        # 如果需要对损失值进行降维处理
        if reduce:
            loss = paddle.mean(loss)

        # 返回计算得到的损失值
        return loss
# 定义 SmoothL1Loss 类，继承自 nn.Layer
class SmoothL1Loss(nn.Layer):
    # 初始化函数，设置参数 beta 和 loss_weight 的默认值
    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        # 初始化 beta 和 loss_weight
        self.beta = beta
        self.loss_weight = loss_weight

        # 创建一个 640x640x2 的全零数组 np_coord
        np_coord = np.zeros(shape=[640, 640, 2], dtype=np.int64)
        # 遍历数组，为每个元素赋值坐标值
        for i in range(640):
            for j in range(640):
                np_coord[i, j, 0] = j
                np_coord[i, j, 1] = i
        # 重塑数组形状为 (-1, 2)
        np_coord = np_coord.reshape((-1, 2))

        # 使用 np_coord 创建一个形状为 (640*640, 2) 的参数 self.coord
        self.coord = self.create_parameter(
            shape=[640 * 640, 2],
            dtype="int32",  # 注意：在 Paddle 2.3.1 之前不支持 "int64"
            default_initializer=nn.initializer.Assign(value=np_coord))
        # 设置参数 self.coord 不参与梯度计算
        self.coord.stop_gradient = True

    # 定义 forward_single 方法，计算损失
    def forward_single(self, input, target, mask, beta=1.0, eps=1e-6):
        # 获取输入的 batch_size
        batch_size = input.shape[0]

        # 计算输入和目标的绝对差值，并乘以 mask
        diff = paddle.abs(input - target) * mask.unsqueeze(1)
        # 根据 Smooth L1 Loss 公式计算损失
        loss = paddle.where(diff < beta, 0.5 * diff * diff / beta,
                            diff - 0.5 * beta)
        # 将损失转换为 float32 类型，并重塑形状
        loss = paddle.cast(loss.reshape((batch_size, -1)), "float32")
        mask = paddle.cast(mask.reshape((batch_size, -1)), "float32")
        # 按最后一个维度求和
        loss = paddle.sum(loss, axis=-1)
        # 计算平均损失
        loss = loss / (mask.sum(axis=-1) + eps)

        return loss
    # 定义一个方法，用于选择单个样本
    def select_single(self, distance, gt_instance, gt_kernel_instance,
                      training_mask):
        
        # 进入无梯度计算环境
        with paddle.no_grad():
            # paddle 2.3.1, paddle.slice 不支持:
            # distance[:, self.coord[:, 1], self.coord[:, 0]]
            # 初始化一个空列表，用于存储选择的距离值
            select_distance_list = []
            # 遍历两次
            for i in range(2):
                # 选择第i个维度的距离值
                tmp1 = distance[i, :]
                # 根据坐标选择对应的距离值
                tmp2 = tmp1[self.coord[:, 1], self.coord[:, 0]]
                # 将选择的距离值添加到列表中
                select_distance_list.append(tmp2.unsqueeze(0))
            # 沿指定维度拼接列表中的距离值
            select_distance = paddle.concat(select_distance_list, axis=0)

            # 计算偏移点的坐标
            off_points = paddle.cast(
                self.coord, "float32") + 10 * select_distance.transpose((1, 0))

            # 将偏移点坐标转换为整型
            off_points = paddle.cast(off_points, "int64")
            # 将偏移点坐标限制在合理范围内
            off_points = paddle.clip(off_points, 0, distance.shape[-1] - 1)

            # 根据条件选择对应的掩码
            selected_mask = (
                gt_instance[self.coord[:, 1], self.coord[:, 0]] !=
                gt_kernel_instance[off_points[:, 1], off_points[:, 0]])
            # 将选择的掩码转换为整型
            selected_mask = paddle.cast(
                selected_mask.reshape((1, -1, distance.shape[-1])), "int64")
            # 将选择的训练掩码与原始训练掩码相乘
            selected_training_mask = selected_mask * training_mask

            # 返回选择后的训练掩码
            return selected_training_mask
    # 定义前向传播函数，接收距离、真实实例、真实核实例、训练掩码、真实距离等参数
    def forward(self,
                distances,
                gt_instances,
                gt_kernel_instances,
                training_masks,
                gt_distances,
                reduce=True):

        # 选取训练掩码
        selected_training_masks = []
        # 遍历距离的第一维度
        for i in range(distances.shape[0]):
            # 选取单个样本的训练掩码
            selected_training_masks.append(
                self.select_single(distances[i, :, :, :], gt_instances[i, :, :],
                                   gt_kernel_instances[i, :, :], training_masks[
                                       i, :, :]))
        # 将选取的训练掩码拼接并转换为 float32 类型
        selected_training_masks = paddle.cast(
            paddle.concat(selected_training_masks, 0), "float32")

        # 计算损失
        loss = self.forward_single(distances, gt_distances,
                                   selected_training_masks, self.beta)
        # 根据损失权重调整损失值
        loss = self.loss_weight * loss

        # 在不计算梯度的情况下进行以下操作
        with paddle.no_grad():
            # 获取批次大小
            batch_size = distances.shape[0]
            # 将选取的训练掩码重塑为二维数组
            false_num = selected_training_masks.reshape((batch_size, -1))
            # 按最后一个维度求和
            false_num = false_num.sum(axis=-1)
            # 将训练掩码重塑为二维数组
            total_num = paddle.cast(
                training_masks.reshape((batch_size, -1)), "float32")
            # 按最后一个维度求和
            total_num = total_num.sum(axis=-1)
            # 计算 IOU
            iou_text = (total_num - false_num) / (total_num + 1e-6)

        # 如果需要缩减损失
        if reduce:
            # 计算平均损失
            loss = paddle.mean(loss)

        # 返回损失和 IOU
        return loss, iou_text
class CTLoss(nn.Layer):
    # 定义自定义的损失函数类，继承自 nn.Layer
    def __init__(self):
        # 初始化函数
        super(CTLoss, self).__init__()
        # 创建 DiceLoss 实例作为 kernel_loss
        self.kernel_loss = DiceLoss()
        # 创建 SmoothL1Loss 实例作为 loc_loss，设置参数 beta=0.1, loss_weight=0.05

    def forward(self, preds, batch):
        # 前向传播函数，接收模型预测结果 preds 和数据 batch
        imgs = batch[0]
        # 获取数据中的图像
        out = preds['maps']
        # 获取模型预测结果中的 maps

        gt_kernels, training_masks, gt_instances, gt_kernel_instances, training_mask_distances, gt_distances = batch[1:]
        # 从数据中获取多个变量

        kernels = out[:, 0, :, :]
        # 从模型预测结果中获取 kernels

        distances = out[:, 1:, :, :]
        # 从模型预测结果中获取 distances

        # kernel loss
        selected_masks = ohem_batch(kernels, gt_kernels, training_masks)
        # 使用 ohem_batch 函数获取选定的 masks

        loss_kernel = self.kernel_loss(
            kernels, gt_kernels, selected_masks, reduce=False)
        # 计算 kernel loss

        iou_kernel = iou(paddle.cast((kernels > 0), "int64"),
                         gt_kernels,
                         training_masks,
                         reduce=False)
        # 计算 IOU

        losses = dict(loss_kernels=loss_kernel, )
        # 将 kernel loss 存入 losses 字典

        # loc loss
        loss_loc, iou_text = self.loc_loss(
            distances,
            gt_instances,
            gt_kernel_instances,
            training_mask_distances,
            gt_distances,
            reduce=False)
        # 计算 loc loss 和 IOU

        losses.update(dict(loss_loc=loss_loc, ))
        # 将 loc loss 存入 losses 字典

        loss_all = loss_kernel + loss_loc
        # 计算总的损失

        losses = {'loss': loss_all}
        # 将总的损失存入 losses 字典

        return losses
        # 返回 losses 字典作为输出结果
```