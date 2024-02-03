# `.\PaddleOCR\ppocr\losses\det_drrg_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码来源于：
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/losses/drrg_loss.py

# 导入 PaddlePaddle 框架
import paddle
# 导入 PaddlePaddle 中的函数模块
import paddle.nn.functional as F
# 从 PaddlePaddle 中导入 nn 模块
from paddle import nn

# 定义 DRRGLoss 类，继承自 nn.Layer
class DRRGLoss(nn.Layer):
    # 初始化函数，接受 ohem_ratio 参数，默认值为 3.0
    def __init__(self, ohem_ratio=3.0):
        # 调用父类的初始化函数
        super().__init__()
        # 设置 ohem_ratio 属性为传入的 ohem_ratio 参数值
        self.ohem_ratio = ohem_ratio
        # 设置 downsample_ratio 属性为 1.0
        self.downsample_ratio = 1.0
    def balance_bce_loss(self, pred, gt, mask):
        """Balanced Binary-CrossEntropy Loss.

        Args:
            pred (Tensor): Shape of :math:`(1, H, W)`.
            gt (Tensor): Shape of :math:`(1, H, W)`.
            mask (Tensor): Shape of :math:`(1, H, W)`.

        Returns:
            Tensor: Balanced bce loss.
        """
        # 断言预测值、真实值和掩码的形状相同
        assert pred.shape == gt.shape == mask.shape
        # 断言预测值在 [0, 1] 范围内
        assert paddle.all(pred >= 0) and paddle.all(pred <= 1)
        # 断言真实值在 [0, 1] 范围内
        assert paddle.all(gt >= 0) and paddle.all(gt <= 1)
        # 计算正样本（真实值为1）的数量
        positive = gt * mask
        # 计算负样本（真实值为0）的数量
        negative = (1 - gt) * mask
        positive_count = int(positive.sum())

        if positive_count > 0:
            # 计算二元交叉熵损失
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            positive_loss = paddle.sum(loss * positive)
            negative_loss = loss * negative
            # 计算负样本数量，最大为正样本数量乘以过采样比例
            negative_count = min(
                int(negative.sum()), int(positive_count * self.ohem_ratio))
        else:
            positive_loss = paddle.to_tensor(0.0)
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            negative_loss = loss * negative
            negative_count = 100
        # 选取负样本损失中的前 k 个值
        negative_loss, _ = paddle.topk(
            negative_loss.reshape([-1]), negative_count)

        # 计算平衡损失
        balance_loss = (positive_loss + paddle.sum(negative_loss)) / (
            float(positive_count + negative_count) + 1e-5)

        return balance_loss

    def gcn_loss(self, gcn_data):
        """CrossEntropy Loss from gcn module.

        Args:
            gcn_data (tuple(Tensor, Tensor)): The first is the
                prediction with shape :math:`(N, 2)` and the
                second is the gt label with shape :math:`(m, n)`
                where :math:`m * n = N`.

        Returns:
            Tensor: CrossEntropy loss.
        """
        # 从 gcn 模块中计算交叉熵损失
        gcn_pred, gt_labels = gcn_data
        gt_labels = gt_labels.reshape([-1])
        # 计算交叉熵损失
        loss = F.cross_entropy(gcn_pred, gt_labels)

        return loss
    # 将位掩码转换为张量
    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        # 获取批处理大小
        batch_size = len(bitmasks)
        # 初始化结果列表
        results = []

        # 初始化内核列表
        kernel = []
        # 遍历每个批次
        for batch_inx in range(batch_size):
            # 获取当前批次的掩码
            mask = bitmasks[batch_inx]
            # 获取掩码的大小
            mask_sz = mask.shape
            # 计算填充值，使掩码与目标大小匹配
            pad = [0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]]
            # 使用常数值0填充掩码
            mask = F.pad(mask, pad, mode='constant', value=0)
            # 将处理后的掩码添加到内核列表中
            kernel.append(mask)
        # 将内核列表转换为张量
        kernel = paddle.stack(kernel)
        # 将内核列表添加到结果列表中
        results.append(kernel)

        # 返回结果列表
        return results
```