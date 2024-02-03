# `.\PaddleOCR\ppocr\losses\kie_sdmgr_loss.py`

```py
# 版权声明
# 版权所有 (c) 2022 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言管理权限和限制。

# 引用来源：https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/kie/losses/sdmgr_loss.py

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import nn
import paddle

# 定义 SDMGRLoss 类，继承自 nn.Layer
class SDMGRLoss(nn.Layer):
    def __init__(self, node_weight=1.0, edge_weight=1.0, ignore=0):
        super().__init__()
        # 定义节点损失函数为交叉熵损失，设置忽略索引为 ignore
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore)
        # 定义边损失函数为交叉熵损失，设置忽略索引为 -1
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=-1)
        # 设置节点权重、边权重和忽略索引
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.ignore = ignore

    # 预处理函数，将输入转换为张量
    def pre_process(self, gts, tag):
        # 将输入转换为 numpy 数组
        gts, tag = gts.numpy(), tag.numpy().tolist()
        temp_gts = []
        batch = len(tag)
        for i in range(batch):
            num, recoder_len = tag[i][0], tag[i][1]
            # 将数据转换为张量，数据类型为 int64
            temp_gts.append(
                paddle.to_tensor(
                    gts[i, :num, :num + 1], dtype='int64'))
        return temp_gts
    # 前向传播函数，接收预测结果和批处理数据作为输入
    def forward(self, pred, batch):
        # 将预测结果分为节点预测和边预测
        node_preds, edge_preds = pred
        # 从批处理数据中获取真实标签和标签类型
        gts, tag = batch[4], batch[5]
        # 对真实标签进行预处理
        gts = self.pre_process(gts, tag)
        # 将节点标签和边标签分别提取出来
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].reshape([-1]))
        # 将节点标签和边标签合并
        node_gts = paddle.concat(node_gts)
        edge_gts = paddle.concat(edge_gts)

        # 获取有效的节点和边的索引
        node_valids = paddle.nonzero(node_gts != self.ignore).reshape([-1])
        edge_valids = paddle.nonzero(edge_gts != -1).reshape([-1])
        
        # 计算节点损失和边损失
        loss_node = self.loss_node(node_preds, node_gts)
        loss_edge = self.loss_edge(edge_preds, edge_gts)
        
        # 计算总损失
        loss = self.node_weight * loss_node + self.edge_weight * loss_edge
        
        # 返回损失和准确率信息
        return dict(
            loss=loss,
            loss_node=loss_node,
            loss_edge=loss_edge,
            acc_node=self.accuracy(
                paddle.gather(node_preds, node_valids),
                paddle.gather(node_gts, node_valids)),
            acc_edge=self.accuracy(
                paddle.gather(edge_preds, edge_valids),
                paddle.gather(edge_gts, edge_valids)))
```