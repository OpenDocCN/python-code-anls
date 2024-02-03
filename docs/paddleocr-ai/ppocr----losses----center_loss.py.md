# `.\PaddleOCR\ppocr\losses\center_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何明示或暗示的保证或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
#
# 该代码参考自: https://github.com/KaiyangZhou/pytorch-center-loss

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 定义 CenterLoss 类，用于实现 Wen 等人在 ECCV 2016 中提出的深度人脸识别的判别特征学习方法
class CenterLoss(nn.Layer):

    def __init__(self, num_classes=6625, feat_dim=96, center_file_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # 初始化中心矩阵，形状为 [类别数, 特征维度]
        self.centers = paddle.randn(
            shape=[self.num_classes, self.feat_dim]).astype("float64")

        # 如果指定了中心文件路径
        if center_file_path is not None:
            # 确保中心文件路径存在
            assert os.path.exists(
                center_file_path
            ), f"center path({center_file_path}) must exist when it is not None."
            # 从中心文件中加载中心数据
            with open(center_file_path, 'rb') as f:
                char_dict = pickle.load(f)
                # 将加载的中心数据转换为张量，并更新到中心矩阵中
                for key in char_dict.keys():
                    self.centers[key] = paddle.to_tensor(char_dict[key])
    # 定义一个类的方法，用于计算损失函数
    def __call__(self, predicts, batch):
        # 断言 predicts 是一个列表或元组
        assert isinstance(predicts, (list, tuple))
        # 将 predicts 拆分为 features 和 predicts 两部分
        features, predicts = predicts

        # 将 features 重塑为二维数组，并转换为 float64 类型
        feats_reshape = paddle.reshape(
            features, [-1, features.shape[-1]]).astype("float64")
        # 根据 axis=2 求 predicts 的最大值索引
        label = paddle.argmax(predicts, axis=2)
        # 将 label 重塑为一维数组
        label = paddle.reshape(label, [label.shape[0] * label.shape[1]])

        # 获取 feats_reshape 的 batch_size
        batch_size = feats_reshape.shape[0]

        # 计算 feats_reshape 和 centers 之间的 L2 距离
        square_feat = paddle.sum(paddle.square(feats_reshape),
                                 axis=1,
                                 keepdim=True)
        square_feat = paddle.expand(square_feat, [batch_size, self.num_classes])

        # 计算 centers 的平方和
        square_center = paddle.sum(paddle.square(self.centers),
                                   axis=1,
                                   keepdim=True)
        square_center = paddle.expand(
            square_center, [self.num_classes, batch_size]).astype("float64")
        square_center = paddle.transpose(square_center, [1, 0])

        # 计算距离矩阵 distmat
        distmat = paddle.add(square_feat, square_center)
        feat_dot_center = paddle.matmul(feats_reshape,
                                        paddle.transpose(self.centers, [1, 0]))
        distmat = distmat - 2.0 * feat_dot_center

        # 生成掩码 mask
        classes = paddle.arange(self.num_classes).astype("int64")
        label = paddle.expand(
            paddle.unsqueeze(label, 1), (batch_size, self.num_classes))
        mask = paddle.equal(
            paddle.expand(classes, [batch_size, self.num_classes]),
            label).astype("float64")
        dist = paddle.multiply(distmat, mask)

        # 计算损失值 loss
        loss = paddle.sum(paddle.clip(dist, min=1e-12, max=1e+12)) / batch_size
        # 返回包含损失值的字典
        return {'loss_center': loss}
```