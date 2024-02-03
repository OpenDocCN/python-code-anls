# `.\PaddleOCR\ppocr\modeling\heads\det_drrg_head.py`

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
# 均基于“原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码来源于：
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/dense_heads/drrg_head.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 引入警告模块
import warnings
# 引入 OpenCV 模块
import cv2
# 引入 NumPy 模块
import numpy as np
# 引入 Paddle 模块
import paddle
# 引入 Paddle 中的神经网络模块
import paddle.nn as nn
# 引入 Paddle 中的函数模块
import paddle.nn.functional as F
# 引入 GCN 模块
from .gcn import GCN
# 引入 LocalGraphs 模块
from .local_graph import LocalGraphs
# 引入 ProposalLocalGraphs 模块
from .proposal_local_graph import ProposalLocalGraphs

# 定义 DRRGHead 类，继承自 nn.Layer
class DRRGHead(nn.Layer):
    # 定义前向传播函数，接受输入和目标参数
    def forward(self, inputs, targets=None):
        """
        Args:
            inputs (Tensor): Shape of :math:`(N, C, H, W)`.
            gt_comp_attribs (list[ndarray]): The padded text component
                attributes. Shape: (num_component, 8).

        Returns:
            tuple: Returns (pred_maps, (gcn_pred, gt_labels)).

                - | pred_maps (Tensor): Prediction map with shape
                    :math:`(N, C_{out}, H, W)`.
                - | gcn_pred (Tensor): Prediction from GCN module, with
                    shape :math:`(N, 2)`.
                - | gt_labels (Tensor): Ground-truth label with shape
                    :math:`(N, 8)`.
        """
        # 如果处于训练模式
        if self.training:
            # 断言目标参数不为空
            assert targets is not None
            # 获取目标文本组件属性
            gt_comp_attribs = targets[7]
            # 使用输出卷积层处理输入数据
            pred_maps = self.out_conv(inputs)
            # 将输入数据和预测地图在通道维度上拼接
            feat_maps = paddle.concat([inputs, pred_maps], axis=1)
            # 调用图卷积网络的训练方法，获取节点特征、邻接矩阵、最近邻索引和目标标签
            node_feats, adjacent_matrices, knn_inds, gt_labels = self.graph_train(
                feat_maps, np.stack(gt_comp_attribs))

            # 使用图卷积网络进行预测
            gcn_pred = self.gcn(node_feats, adjacent_matrices, knn_inds)

            # 返回预测地图和预测结果以及目标标签
            return pred_maps, (gcn_pred, gt_labels)
        # 如果处于非训练模式
        else:
            # 调用单个测试方法
            return self.single_test(inputs)
    def single_test(self, feat_maps):
        r"""
        Args:
            feat_maps (Tensor): Shape of :math:`(N, C, H, W)`.

        Returns:
            tuple: Returns (edge, score, text_comps).

                - | edge (ndarray): The edge array of shape :math:`(N, 2)`
                    where each row is a pair of text component indices
                    that makes up an edge in graph.
                - | score (ndarray): The score array of shape :math:`(N,)`,
                    corresponding to the edge above.
                - | text_comps (ndarray): The text components of shape
                    :math:`(N, 9)` where each row corresponds to one box and
                    its score: (x1, y1, x2, y2, x3, y3, x4, y4, score).
        """
        # 使用输出卷积层处理特征图
        pred_maps = self.out_conv(feat_maps)
        # 将原始特征图和预测特征图拼接在一起
        feat_maps = paddle.concat([feat_maps, pred_maps], axis=1)

        # 调用图测试函数，获取图数据
        none_flag, graph_data = self.graph_test(pred_maps, feat_maps)

        # 解析图数据
        (local_graphs_node_feat, adjacent_matrices, pivots_knn_inds,
         pivot_local_graphs, text_comps) = graph_data

        # 如果标志为None，则返回空
        if none_flag:
            return None, None, None
        # 使用GCN进行预测
        gcn_pred = self.gcn(local_graphs_node_feat, adjacent_matrices,
                            pivots_knn_inds)
        # 对预测结果进行softmax处理
        pred_labels = F.softmax(gcn_pred, axis=1)

        edges = []
        scores = []
        pivot_local_graphs = pivot_local_graphs.squeeze().numpy()

        # 遍历邻居节点，构建边和分数
        for pivot_ind, pivot_local_graph in enumerate(pivot_local_graphs):
            pivot = pivot_local_graph[0]
            for k_ind, neighbor_ind in enumerate(pivots_knn_inds[pivot_ind]):
                neighbor = pivot_local_graph[neighbor_ind.item()]
                edges.append([pivot, neighbor])
                scores.append(pred_labels[pivot_ind * pivots_knn_inds.shape[1] +
                                          k_ind, 1].item())

        edges = np.asarray(edges)
        scores = np.asarray(scores)

        return edges, scores, text_comps
```