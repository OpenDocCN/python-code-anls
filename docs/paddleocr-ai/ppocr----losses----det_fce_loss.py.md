# `.\PaddleOCR\ppocr\losses\det_fce_loss.py`

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
# 均按“原样”分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码参考自：
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/losses/fce_loss.py

# 导入必要的库
import numpy as np
from paddle import nn
import paddle
import paddle.nn.functional as F
from functools import partial

# 定义一个函数，用于对多个输入参数执行相同的函数操作
def multi_apply(func, *args, **kwargs):
    # 如果有额外的关键字参数，使用偏函数进行处理
    pfunc = partial(func, **kwargs) if kwargs else func
    # 对输入参数执行函数操作，并将结果转换为元组返回
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

# 定义一个类，实现 FCENet 损失函数
class FCELoss(nn.Layer):
    """用于实现 FCENet 损失函数
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped
        Text Detection

    [https://arxiv.org/abs/2104.10442]

    Args:
        fourier_degree (int) : 最大傅里叶变换度 k
        num_sample (int) : 回归损失的采样点数
            如果太小，FCENet 很容易过拟合
        ohem_ratio (float): OHEM 中的负/正比例
    """

    def __init__(self, fourier_degree, num_sample, ohem_ratio=3.):
        super().__init__()
        self.fourier_degree = fourier_degree
        self.num_sample = num_sample
        self.ohem_ratio = ohem_ratio
    # 前向传播函数，接收预测值和标签值作为参数
    def forward(self, preds, labels):
        # 断言预测值为字典类型
        assert isinstance(preds, dict)
        # 从预测值中获取'levels'键对应的值
        preds = preds['levels']

        # 从标签中获取p3_maps, p4_maps, p5_maps
        p3_maps, p4_maps, p5_maps = labels[1:]
        # 断言p3_maps的第一个元素的行数等于4倍的傅里叶级数加5
        assert p3_maps[0].shape[0] == 4 * self.fourier_degree + 5, 'fourier degree not equal in FCEhead and FCEtarget'

        # 将p3_maps, p4_maps, p5_maps转换为张量
        gts = [p3_maps, p4_maps, p5_maps]
        for idx, maps in enumerate(gts):
            gts[idx] = paddle.to_tensor(np.stack(maps))

        # 调用multi_apply函数计算损失
        losses = multi_apply(self.forward_single, preds, gts)

        # 初始化各种损失为0的张量
        loss_tr = paddle.to_tensor(0.).astype('float32')
        loss_tcl = paddle.to_tensor(0.).astype('float32')
        loss_reg_x = paddle.to_tensor(0.).astype('float32')
        loss_reg_y = paddle.to_tensor(0.).astype('float32')
        loss_all = paddle.to_tensor(0.).astype('float32')

        # 遍历损失列表，计算总损失和各个部分的损失
        for idx, loss in enumerate(losses):
            loss_all += sum(loss)
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            elif idx == 2:
                loss_reg_x += sum(loss)
            else:
                loss_reg_y += sum(loss)

        # 将各个损失值存储在字典中
        results = dict(
            loss=loss_all,
            loss_text=loss_tr,
            loss_center=loss_tcl,
            loss_reg_x=loss_reg_x,
            loss_reg_y=loss_reg_y, )
        # 返回结果字典
        return results
    # 实现在线困难样本挖掘（OHEM）的函数，输入为预测值、目标值和训练掩码
    def ohem(self, predict, target, train_mask):

        # 计算正样本的掩码
        pos = (target * train_mask).astype('bool')
        # 计算负样本的掩码
        neg = ((1 - target) * train_mask).astype('bool')

        # 将正样本掩码扩展为两个维度
        pos2 = paddle.concat([pos.unsqueeze(1), pos.unsqueeze(1)], axis=1)
        # 将负样本掩码扩展为两个维度
        neg2 = paddle.concat([neg.unsqueeze(1), neg.unsqueeze(1)], axis=1)

        # 计算正样本数量
        n_pos = pos.astype('float32').sum()

        # 如果存在正样本
        if n_pos.item() > 0:
            # 计算正样本的交叉熵损失
            loss_pos = F.cross_entropy(
                predict.masked_select(pos2).reshape([-1, 2]),
                target.masked_select(pos).astype('int64'),
                reduction='sum')
            # 计算负样本的交叉熵损失
            loss_neg = F.cross_entropy(
                predict.masked_select(neg2).reshape([-1, 2]),
                target.masked_select(neg).astype('int64'),
                reduction='none')
            # 计算负样本数量，取最小值
            n_neg = min(
                int(neg.astype('float32').sum().item()),
                int(self.ohem_ratio * n_pos.astype('float32')))
        else:
            # 如果不存在正样本，正样本损失为0
            loss_pos = paddle.to_tensor(0.)
            # 计算负样本的交叉熵损失
            loss_neg = F.cross_entropy(
                predict.masked_select(neg2).reshape([-1, 2]),
                target.masked_select(neg).astype('int64'),
                reduction='none')
            # 设置负样本数量为100
            n_neg = 100
        # 如果负样本损失数量大于负样本数量，只保留前n_neg个损失
        if len(loss_neg) > n_neg:
            loss_neg, _ = paddle.topk(loss_neg, n_neg)

        # 返回正负样本损失之和除以正负样本数量之和
        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).astype('float32')
    # 将傅立叶系数映射转换为多边形映射
    def fourier2poly(self, real_maps, imag_maps):
        """Transform Fourier coefficient maps to polygon maps.

        Args:
            real_maps (tensor): A map composed of the real parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
            imag_maps (tensor):A map composed of the imag parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)

        Returns
            x_maps (tensor): A map composed of the x value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
            y_maps (tensor): A map composed of the y value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
        """

        # 生成包含傅立叶系数的 k 向量
        k_vect = paddle.arange(
            -self.fourier_degree, self.fourier_degree + 1,
            dtype='float32').reshape([-1, 1])
        # 生成包含采样点的 i 向量
        i_vect = paddle.arange(
            0, self.num_sample, dtype='float32').reshape([1, -1])

        # 计算变换矩阵
        transform_matrix = 2 * np.pi / self.num_sample * paddle.matmul(k_vect,
                                                                       i_vect)

        # 计算 x1
        x1 = paddle.einsum('ak, kn-> an', real_maps,
                           paddle.cos(transform_matrix))
        # 计算 x2
        x2 = paddle.einsum('ak, kn-> an', imag_maps,
                           paddle.sin(transform_matrix))
        # 计算 y1
        y1 = paddle.einsum('ak, kn-> an', real_maps,
                           paddle.sin(transform_matrix))
        # 计算 y2
        y2 = paddle.einsum('ak, kn-> an', imag_maps,
                           paddle.cos(transform_matrix))

        # 计算 x_maps
        x_maps = x1 - x2
        # 计算 y_maps
        y_maps = y1 + y2

        # 返回 x_maps 和 y_maps
        return x_maps, y_maps
```