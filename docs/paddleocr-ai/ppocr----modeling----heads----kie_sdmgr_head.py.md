# `.\PaddleOCR\ppocr\modeling\heads\kie_sdmgr_head.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按"原样"分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
# 参考来源：https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/kie/heads/sdmgr_head.py

# 导入必要的库和模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

# 定义 SDMGRHead 类，继承自 nn.Layer
class SDMGRHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_chars=92,
                 visual_dim=16,
                 fusion_dim=1024,
                 node_input=32,
                 node_embed=256,
                 edge_input=5,
                 edge_embed=256,
                 num_gnn=2,
                 num_classes=26,
                 bidirectional=False):
        super().__init__()

        # 定义融合层，输入维度为 [visual_dim, node_embed]，输出维度为 node_embed，隐藏层维度为 fusion_dim
        self.fusion = Block([visual_dim, node_embed], node_embed, fusion_dim)
        # 定义节点嵌入层，字符数量为 num_chars，输入维度为 node_input，padding_idx 为 0
        self.node_embed = nn.Embedding(num_chars, node_input, 0)
        hidden = node_embed // 2 if bidirectional else node_embed
        # 定义 LSTM 层，输入维度为 node_input，隐藏层维度为 hidden，层数为 1
        self.rnn = nn.LSTM(
            input_size=node_input, hidden_size=hidden, num_layers=1)
        # 定义边嵌入层，输入维度为 edge_input，输出维度为 edge_embed
        self.edge_embed = nn.Linear(edge_input, edge_embed)
        # 定义 GNN 层列表，包含 num_gnn 个 GNNLayer 层
        self.gnn_layers = nn.LayerList(
            [GNNLayer(node_embed, edge_embed) for _ in range(num_gnn)])
        # 定义节点分类层，输入维度为 node_embed，输出维度为 num_classes
        self.node_cls = nn.Linear(node_embed, num_classes)
        # 定义边分类层，输入维度为 edge_embed，输出维度为 2
        self.edge_cls = nn.Linear(edge_embed, 2)

# 定义 GNNLayer 类，继承自 nn.Layer
class GNNLayer(nn.Layer):
    # 初始化函数，设置节点维度和边维度，默认为256
    def __init__(self, node_dim=256, edge_dim=256):
        # 调用父类的初始化函数
        super().__init__()
        # 输入全连接层，输入维度为节点维度*2+边维度，输出维度为节点维度
        self.in_fc = nn.Linear(node_dim * 2 + edge_dim, node_dim)
        # 系数全连接层，输入维度为节点维度，输出维度为1
        self.coef_fc = nn.Linear(node_dim, 1)
        # 输出全连接层，输入维度为节点维度，输出维度为节点维度
        self.out_fc = nn.Linear(node_dim, node_dim)
        # ReLU激活函数
        self.relu = nn.ReLU()

    # 前向传播函数，接收节点、边和节点数量列表作为输入
    def forward(self, nodes, edges, nums):
        # 初始化起始索引和拼接节点列表
        start, cat_nodes = 0, []
        # 遍历节点数量列表
        for num in nums:
            # 根据节点数量切片节点列表
            sample_nodes = nodes[start:start + num]
            # 将节点列表进行拼接并重塑形状，得到拼接后的节点列表
            cat_nodes.append(
                paddle.concat([
                    paddle.expand(sample_nodes.unsqueeze(1), [-1, num, -1]),
                    paddle.expand(sample_nodes.unsqueeze(0), [num, -1, -1])
                ], -1).reshape([num**2, -1]))
            # 更新起始索引
            start += num
        # 将拼接后的节点列表和边进行拼接
        cat_nodes = paddle.concat([paddle.concat(cat_nodes), edges], -1)
        # 对拼接后的节点列表进行ReLU激活函数处理
        cat_nodes = self.relu(self.in_fc(cat_nodes))
        # 使用系数全连接层计算系数
        coefs = self.coef_fc(cat_nodes)

        # 重新初始化起始索引和残差列表
        start, residuals = 0, []
        # 再次遍历节点数量列表
        for num in nums:
            # 计算残差，包括softmax处理和残差计算
            residual = F.softmax(
                -paddle.eye(num).unsqueeze(-1) * 1e9 +
                coefs[start:start + num**2].reshape([num, num, -1]), 1)
            # 将残差与拼接后的节点列表相乘并按维度求和，得到残差列表
            residuals.append((residual * cat_nodes[start:start + num**2]
                              .reshape([num, num, -1])).sum(1))
            # 更新起始索引
            start += num**2

        # 更新节点列表，包括残差处理和输出全连接层处理
        nodes += self.relu(self.out_fc(paddle.concat(residuals)))
        # 返回更新后的节点列表和拼接后的节点列表
        return [nodes, cat_nodes]
class Block(nn.Layer):
    # 定义一个名为 Block 的类，继承自 nn.Layer 类
    def __init__(self,
                 input_dims,
                 output_dim,
                 mm_dim=1600,
                 chunks=20,
                 rank=15,
                 shared=False,
                 dropout_input=0.,
                 dropout_pre_lin=0.,
                 dropout_output=0.,
                 pos_norm='before_cat'):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数
        self.rank = rank
        # 设置类属性 rank
        self.dropout_input = dropout_input
        # 设置类属性 dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        # 设置类属性 dropout_pre_lin
        self.dropout_output = dropout_output
        # 设置类属性 dropout_output
        assert (pos_norm in ['before_cat', 'after_cat'])
        # 断言 pos_norm 的取值只能是 'before_cat' 或 'after_cat'
        self.pos_norm = pos_norm
        # 设置类属性 pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        # 创建一个线性层，输入维度为 input_dims[0]，输出维度为 mm_dim
        self.linear1 = (self.linear0
                        if shared else nn.Linear(input_dims[1], mm_dim))
        # 如果 shared 为 True，则 self.linear1 和 self.linear0 共享参数，否则创建一个新的线性层
        self.merge_linears0 = nn.LayerList()
        # 创建一个空的 LayerList 对象 merge_linears0
        self.merge_linears1 = nn.LayerList()
        # 创建一个空的 LayerList 对象 merge_linears1
        self.chunks = self.chunk_sizes(mm_dim, chunks)
        # 计算 mm_dim 在 chunks 个块中的大小
        for size in self.chunks:
            # 遍历 chunks 中的每个大小
            ml0 = nn.Linear(size, size * rank)
            # 创建一个线性层，输入维度为 size，输出维度为 size * rank
            self.merge_linears0.append(ml0)
            # 将 ml0 添加到 merge_linears0 中
            ml1 = ml0 if shared else nn.Linear(size, size * rank)
            # 如果 shared 为 True，则 ml1 和 ml0 共享参数，否则创建一个新的线性层
            self.merge_linears1.append(ml1)
            # 将 ml1 添加到 merge_linears1 中
        self.linear_out = nn.Linear(mm_dim, output_dim)
        # 创建一个线性层，输入维度为 mm_dim，输出维度为 output_dim
    # 前向传播函数，接收输入 x
    def forward(self, x):
        # 对输入 x 的第一个元素进行线性变换
        x0 = self.linear0(x[0])
        # 对输入 x 的第二个元素进行线性变换
        x1 = self.linear1(x[1])
        # 获取 x1 的 batch size
        bs = x1.shape[0]
        # 如果有输入层的 dropout
        if self.dropout_input > 0:
            # 对 x0 和 x1 进行 dropout
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        # 将 x0 按照指定维度分割成多个块
        x0_chunks = paddle.split(x0, self.chunks, -1)
        # 将 x1 按照指定维度分割成多个块
        x1_chunks = paddle.split(x1, self.chunks, -1)
        # 初始化结果列表
        zs = []
        # 遍历 x0 和 x1 的块以及对应的 merge_linears0 和 merge_linears1
        for x0_c, x1_c, m0, m1 in zip(x0_chunks, x1_chunks, self.merge_linears0, self.merge_linears1):
            # 对 x0_c 和 x1_c 进行线性变换并相乘
            m = m0(x0_c) * m1(x1_c)  # bs x split_size*rank
            # 重塑形状
            m = m.reshape([bs, self.rank, -1])
            # 沿着指定维度求和
            z = paddle.sum(m, 1)
            # 如果位置归一化在拼接之前
            if self.pos_norm == 'before_cat':
                # 对 z 进行非线性变换
                z = paddle.sqrt(F.relu(z)) - paddle.sqrt(F.relu(-z))
                # 对 z 进行归一化
                z = F.normalize(z)
            # 将 z 添加到结果列表中
            zs.append(z)
        # 拼接所有的 z
        z = paddle.concat(zs, 1)
        # 如果位置归一化在拼接之后
        if self.pos_norm == 'after_cat':
            # 对 z 进行非线性变换
            z = paddle.sqrt(F.relu(z)) - paddle.sqrt(F.relu(-z))
            # 对 z 进行归一化
            z = F.normalize(z)

        # 如果有输出层的 dropout
        if self.dropout_pre_lin > 0:
            # 对 z 进行 dropout
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        # 对 z 进行线性变换
        z = self.linear_out(z)
        # 如果有输出层的 dropout
        if self.dropout_output > 0:
            # 对 z 进行 dropout
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        # 返回 z
        return z

    # 计算每个块的大小
    def chunk_sizes(self, dim, chunks):
        # 计算每个块的大小
        split_size = (dim + chunks - 1) // chunks
        # 初始化大小列表
        sizes_list = [split_size] * chunks
        # 调整最后一个块的大小
        sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
        # 返回大小列表
        return sizes_list
```