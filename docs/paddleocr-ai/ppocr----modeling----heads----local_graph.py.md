# `.\PaddleOCR\ppocr\modeling\heads\local_graph.py`

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
# 均基于“原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用来源
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/modules/local_graph.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
from ppocr.ext_op import RoIAlignRotated

# 标准化邻接矩阵
def normalize_adjacent_matrix(A):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    # 将对角线元素设为1
    A = A + np.eye(A.shape[0])
    # 计算每列的和
    d = np.sum(A, axis=0)
    # 将和限制在0以上
    d = np.clip(d, 0, None)
    # 计算每个元素的倒数再取平方根
    d_inv = np.power(d, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_inv = np.diag(d_inv)
    # 计算标准化后的邻接矩阵
    G = A.dot(d_inv).transpose().dot(d_inv)
    return G

# 计算欧几里得距离矩阵
def euclidean_distance_matrix(A, B):
    """Calculate the Euclidean distance matrix.

    Args:
        A (ndarray): The point sequence.
        B (ndarray): The point sequence with the same dimensions as A.

    returns:
        D (ndarray): The Euclidean distance matrix.
    """
    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape[1] == B.shape[1]

    m = A.shape[0]
    n = B.shape[0]

    # 计算 A 和 B 的点积
    A_dots = (A * A).sum(axis=1).reshape((m, 1)) * np.ones(shape=(1, n))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(m, 1))
    # 计算欧几里得距离的平方
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    zero_mask = np.less(D_squared, 0.0)
    D_squared[zero_mask] = 0.0
    # 计算欧几里得距离
    D = np.sqrt(D_squared)
    return D
def feature_embedding(input_feats, out_feat_len):
    """Embed features. This code was partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    Args:
        input_feats (ndarray): The input features of shape (N, d), where N is
            the number of nodes in graph, d is the input feature vector length.
        out_feat_len (int): The length of output feature vector.

    Returns:
        embedded_feats (ndarray): The embedded features.
    """
    # 检查输入特征的维度是否为2
    assert input_feats.ndim == 2
    # 检查输出特征向量长度是否为整数
    assert isinstance(out_feat_len, int)
    # 检查输出特征向量长度是否大于等于输入特征向量长度
    assert out_feat_len >= input_feats.shape[1]

    # 获取节点数量
    num_nodes = input_feats.shape[0]
    # 获取特征向量维度
    feat_dim = input_feats.shape[1]
    # 计算特征向量重复次数
    feat_repeat_times = out_feat_len // feat_dim
    # 计算余数维度
    residue_dim = out_feat_len % feat_dim

    # 如果余数维度大于0
    if residue_dim > 0:
        # 计算嵌入波
        embed_wave = np.array([
            np.power(1000, 2.0 * (j // 2) / feat_repeat_times + 1)
            for j in range(feat_repeat_times + 1)
        ]).reshape((feat_repeat_times + 1, 1, 1))
        # 重复特征向量
        repeat_feats = np.repeat(
            np.expand_dims(
                input_feats, axis=0), feat_repeat_times, axis=0)
        # 计算余数特征向量
        residue_feats = np.hstack([
            input_feats[:, 0:residue_dim], np.zeros(
                (num_nodes, feat_dim - residue_dim))
        ])
        residue_feats = np.expand_dims(residue_feats, axis=0)
        repeat_feats = np.concatenate([repeat_feats, residue_feats], axis=0)
        # 计算嵌入特征向量
        embedded_feats = repeat_feats / embed_wave
        embedded_feats[:, 0::2] = np.sin(embedded_feats[:, 0::2])
        embedded_feats[:, 1::2] = np.cos(embedded_feats[:, 1::2])
        embedded_feats = np.transpose(embedded_feats, (1, 0, 2)).reshape(
            (num_nodes, -1))[:, 0:out_feat_len]
    # 如果条件不满足，则执行以下操作
    else:
        # 创建一个嵌入的波形数组，根据特征重复次数计算每个元素的值
        embed_wave = np.array([
            np.power(1000, 2.0 * (j // 2) / feat_repeat_times)
            for j in range(feat_repeat_times)
        ]).reshape((feat_repeat_times, 1, 1))
        # 将输入特征扩展并重复多次
        repeat_feats = np.repeat(
            np.expand_dims(
                input_feats, axis=0), feat_repeat_times, axis=0)
        # 将重复的特征除以嵌入的波形
        embedded_feats = repeat_feats / embed_wave
        # 对偶数索引位置的元素应用正弦函数
        embedded_feats[:, 0::2] = np.sin(embedded_feats[:, 0::2])
        # 对奇数索引位置的元素应用余弦函数
        embedded_feats[:, 1::2] = np.cos(embedded_feats[:, 1::2])
        # 转置嵌入的特征，并重新组织形状
        embedded_feats = np.transpose(embedded_feats, (1, 0, 2)).reshape(
            (num_nodes, -1)).astype(np.float32)
    
    # 返回嵌入的特征
    return embedded_feats
# 定义一个名为 LocalGraphs 的类
class LocalGraphs:
    # 初始化方法，接受多个参数
    def __init__(self, k_at_hops, num_adjacent_linkages, node_geo_feat_len,
                 pooling_scale, pooling_output_size, local_graph_thr):
        # 断言 k_at_hops 参数长度为2
        assert len(k_at_hops) == 2
        # 断言 k_at_hops 参数中的元素都为整数
        assert all(isinstance(n, int) for n in k_at_hops)
        # 断言 num_adjacent_linkages 参数为整数
        assert isinstance(num_adjacent_linkages, int)
        # 断言 node_geo_feat_len 参数为整数
        assert isinstance(node_geo_feat_len, int)
        # 断言 pooling_scale 参数为浮点数
        assert isinstance(pooling_scale, float)
        # 断言 pooling_output_size 参数中的元素都为整数
        assert all(isinstance(n, int) for n in pooling_output_size)
        # 断言 local_graph_thr 参数为浮点数
        assert isinstance(local_graph_thr, float)

        # 将参数赋值给类的属性
        self.k_at_hops = k_at_hops
        self.num_adjacent_linkages = num_adjacent_linkages
        self.node_geo_feat_dim = node_geo_feat_len
        # 创建 RoIAlignRotated 对象并赋值给 pooling 属性
        self.pooling = RoIAlignRotated(pooling_output_size, pooling_scale)
        self.local_graph_thr = local_graph_thr
```