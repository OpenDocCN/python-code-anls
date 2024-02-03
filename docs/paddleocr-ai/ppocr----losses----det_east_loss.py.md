# `.\PaddleOCR\ppocr\losses\det_east_loss.py`

```py
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
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
from paddle import nn
# 导入 DiceLoss 类
from .det_basic_loss import DiceLoss

# 定义 EASTLoss 类，继承自 nn.Layer
class EASTLoss(nn.Layer):
    """
    """

    # 初始化方法
    def __init__(self,
                 eps=1e-6,
                 **kwargs):
        # 调用父类的初始化方法
        super(EASTLoss, self).__init__()
        # 创建 DiceLoss 对象，传入 eps 参数
        self.dice_loss = DiceLoss(eps=eps)
    # 前向传播函数，计算预测结果和标签之间的损失
    def forward(self, predicts, labels):
        # 从标签中获取得分、几何信息和掩码信息
        l_score, l_geo, l_mask = labels[1:]
        # 从预测结果中获取得分和几何信息
        f_score = predicts['f_score']
        f_geo = predicts['f_geo']

        # 计算 Dice 损失
        dice_loss = self.dice_loss(f_score, l_score, l_mask)

        # 计算 smooth L1 损失
        channels = 8
        # 将几何信息按通道数拆分
        l_geo_split = paddle.split(
            l_geo, num_or_sections=channels + 1, axis=1)
        f_geo_split = paddle.split(f_geo, num_or_sections=channels, axis=1)
        smooth_l1 = 0
        for i in range(0, channels):
            # 计算几何信息的差异
            geo_diff = l_geo_split[i] - f_geo_split[i]
            abs_geo_diff = paddle.abs(geo_diff)
            smooth_l1_sign = paddle.less_than(abs_geo_diff, l_score)
            smooth_l1_sign = paddle.cast(smooth_l1_sign, dtype='float32')
            in_loss = abs_geo_diff * abs_geo_diff * smooth_l1_sign + \
                (abs_geo_diff - 0.5) * (1.0 - smooth_l1_sign)
            out_loss = l_geo_split[-1] / channels * in_loss * l_score
            smooth_l1 += out_loss
        smooth_l1_loss = paddle.mean(smooth_l1 * l_score)

        # 将 Dice 损失乘以一个系数
        dice_loss = dice_loss * 0.01
        # 计算总损失
        total_loss = dice_loss + smooth_l1_loss
        # 汇总损失信息
        losses = {"loss":total_loss, \
                  "dice_loss":dice_loss,\
                  "smooth_l1_loss":smooth_l1_loss}
        return losses
```