# `.\PaddleOCR\ppocr\modeling\heads\gcn.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发，基于“原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 许可证下的限制
"""
此代码参考自：
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/modules/gcn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BatchNorm1D(nn.BatchNorm1D):
    # 一维批量归一化层
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        # 计算动量
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            # 如果不进行仿射变换，设置权重和偏置的学习率为0
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)


class MeanAggregator(nn.Layer):
    # 均值聚合器
    def forward(self, features, A):
        # 矩阵乘法，计算 A 和 features 的乘积
        x = paddle.bmm(A, features)
        return x


class GraphConv(nn.Layer):
    # 初始化函数，定义输入维度和输出维度
    def __init__(self, in_dim, out_dim):
        # 调用父类的初始化函数
        super().__init__()
        # 保存输入维度和输出维度
        self.in_dim = in_dim
        self.out_dim = out_dim
        # 创建权重参数，形状为[in_dim * 2, out_dim]，使用 XavierUniform 初始化器
        self.weight = self.create_parameter(
            [in_dim * 2, out_dim],
            default_initializer=nn.initializer.XavierUniform())
        # 创建偏置参数，形状为[out_dim]，设置为偏置参数，使用 Assign 初始化器，初始值为[0] * out_dim
        self.bias = self.create_parameter(
            [out_dim],
            is_bias=True,
            default_initializer=nn.initializer.Assign([0] * out_dim))

        # 创建 MeanAggregator 实例
        self.aggregator = MeanAggregator()

    # 前向传播函数，接收特征和邻接矩阵作为输入
    def forward(self, features, A):
        # 获取特征的维度信息
        b, n, d = features.shape
        # 断言特征维度与输入维度相同
        assert d == self.in_dim
        # 聚合特征信息
        agg_feats = self.aggregator(features, A)
        # 拼接原始特征和聚合后的特征
        cat_feats = paddle.concat([features, agg_feats], axis=2)
        # 矩阵乘法操作，计算输出
        out = paddle.einsum('bnd,df->bnf', cat_feats, self.weight)
        # 使用 ReLU 激活函数
        out = F.relu(out + self.bias)
        # 返回输出结果
        return out
class GCN(nn.Layer):
    # 定义 GCN 类，继承自 nn.Layer
    def __init__(self, feat_len):
        # 初始化函数，接受特征长度作为参数
        super(GCN, self).__init__()
        # 调用父类的初始化函数
        self.bn0 = BatchNorm1D(feat_len, affine=False)
        # 初始化 BatchNorm1D 层，用于对输入数据进行批量归一化
        self.conv1 = GraphConv(feat_len, 512)
        # 初始化第一层 GraphConv，输入特征长度为 feat_len，输出特征长度为 512
        self.conv2 = GraphConv(512, 256)
        # 初始化第二层 GraphConv，输入特征长度为 512，输出特征长度为 256
        self.conv3 = GraphConv(256, 128)
        # 初始化第三层 GraphConv，输入特征长度为 256，输出特征长度为 128
        self.conv4 = GraphConv(128, 64)
        # 初始化第四层 GraphConv，输入特征长度为 128，输出特征长度为 64
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.PReLU(32), nn.Linear(32, 2))
        # 初始化分类器，包含两个线性层和一个 PReLU 激活函数

    def forward(self, x, A, knn_inds):
        # 前向传播函数，接受输入 x、邻接矩阵 A 和 knn_inds

        num_local_graphs, num_max_nodes, feat_len = x.shape
        # 获取输入 x 的形状信息

        x = x.reshape([-1, feat_len])
        # 将输入 x 展平为二维张量
        x = self.bn0(x)
        # 对输入 x 进行批量归一化
        x = x.reshape([num_local_graphs, num_max_nodes, feat_len])
        # 将归一化后的 x 重新恢复为原始形状

        x = self.conv1(x, A)
        # 经过第一层 GraphConv 运算
        x = self.conv2(x, A)
        # 经过第二层 GraphConv 运算
        x = self.conv3(x, A)
        # 经过第三层 GraphConv 运算
        x = self.conv4(x, A)
        # 经过第四层 GraphConv 运算

        k = knn_inds.shape[-1]
        # 获取 knn_inds 的最后一个维度大小
        mid_feat_len = x.shape[-1]
        # 获取 x 的最后一个维度大小
        edge_feat = paddle.zeros([num_local_graphs, k, mid_feat_len])
        # 创建全零张量 edge_feat，形状为 [num_local_graphs, k, mid_feat_len]
        for graph_ind in range(num_local_graphs):
            # 遍历每个局部图
            edge_feat[graph_ind, :, :] = x[graph_ind][paddle.to_tensor(knn_inds[graph_ind])]
            # 根据 knn_inds 选择对应的特征，填充到 edge_feat 中
        edge_feat = edge_feat.reshape([-1, mid_feat_len])
        # 将 edge_feat 展平为二维张量
        pred = self.classifier(edge_feat)
        # 使用分类器进行预测

        return pred
        # 返回预测结果
```