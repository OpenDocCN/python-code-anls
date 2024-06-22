# `.\models\graphormer\modeling_graphormer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2022 年 Microsoft、clefourrier 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”基础分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
""" PyTorch Graphormer 模型。"""

import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "graphormer-base-pcqm4mv1"
_CONFIG_FOR_DOC = "GraphormerConfig"

# 预训练模型存档列表
GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "clefourrier/graphormer-base-pcqm4mv1",
    "clefourrier/graphormer-base-pcqm4mv2",
    # 查看所有 Graphormer 模型，请访问 https://huggingface.co/models?filter=graphormer
]

def quant_noise(module: nn.Module, p: float, block_size: int):
    """
    从以下链接获取：
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/quant_noise.py

    包装模块并对权重应用量化噪声，以便后续使用迭代乘积量化进行量化，如“使用量化噪声进行极端模型压缩”中所述

    Args:
        - module: nn.Module
        - p: 量化噪声的数量
        - block_size: 用于后续使用 iPQ 进行量化的块的大小

    Remarks:
        - 模块的权重必须符合块大小的要求
        - 目前仅支持线性、嵌入和 Conv2d 模块
        - 有关如何通过卷积权重按块进行量化的更多详细信息，请参阅“位的变化：重新审视神经网络的量化”
        - 我们在此实现了论文中所述的最简单形式的噪声，即随机丢弃块
    """

    # 如果没有量化噪声，则不注册钩子
    if p <= 0:
        return module

    # 支持的模块
    if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        raise NotImplementedError("Module unsupported for quant_noise.")

    # 测试模块的权重是否符合块大小的要求
    is_conv = module.weight.ndim == 4

    # 2D 矩阵
    # 如果不是卷积层
    if not is_conv:
        # 检查输入特征是否是块大小的倍数
        if module.weight.size(1) % block_size != 0:
            raise AssertionError("Input features must be a multiple of block sizes")

    # 如果是卷积层
    else:
        # 对于1x1卷积
        if module.kernel_size == (1, 1):
            # 检查输入通道是否是块大小的倍数
            if module.in_channels % block_size != 0:
                raise AssertionError("Input channels must be a multiple of block sizes")
        # 对于普通卷积
        else:
            # 计算卷积核大小
            k = module.kernel_size[0] * module.kernel_size[1]
            # 检查卷积核大小是否是块大小的倍数
            if k % block_size != 0:
                raise AssertionError("Kernel size must be a multiple of block size")

    # 定义前向预处理钩子函数
    def _forward_pre_hook(mod, input):
        # 在训练时不添加噪声
        if mod.training:
            # 如果不是卷积层
            if not is_conv:
                # 获取权重和大小
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # 将权重矩阵分成块并随机丢弃选定的块
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # 获取权重和大小
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # 将权重矩阵分成块并随机丢弃选定的块
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])

            # 缩放权重并应用掩码
            mask = mask.to(torch.bool)  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    # 注册前向预处理钩子函数
    module.register_forward_pre_hook(_forward_pre_hook)
    # 返回模块
    return module
class LayerDropModuleList(nn.ModuleList):
    """
    从 https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/layer_drop.py
    中实现的基于 [`torch.nn.ModuleList`] 的 LayerDrop 实现。LayerDrop 描述在 https://arxiv.org/abs/1909.11556。

    每次迭代 LayerDropModuleList 实例时，我们都会重新选择要丢弃的层。在评估期间，我们始终会迭代所有层。

    用法:

    ```python
    layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
    for layer in layers:  # 这可能会迭代层 1 和 3
        x = layer(x)
    for layer in layers:  # 这可能会迭代所有层
        x = layer(x)
    for layer in layers:  # 这可能不会迭代任何层
        x = layer(x)
    ```py

    Args:
        p (float): 每个层丢弃的概率
        modules (iterable, optional): 要添加的模块的可迭代对象
    """

    def __init__(self, p: float, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.p = p

    def __iter__(self) -> Iterator[nn.Module]:
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m


class GraphormerGraphNodeFeature(nn.Module):
    """
    为图中的每个节点计算节点特征。
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_atoms = config.num_atoms

        self.atom_encoder = nn.Embedding(config.num_atoms + 1, config.hidden_size, padding_idx=config.pad_token_id)
        self.in_degree_encoder = nn.Embedding(
            config.num_in_degree, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.out_degree_encoder = nn.Embedding(
            config.num_out_degree, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.graph_token = nn.Embedding(1, config.hidden_size)

    def forward(
        self,
        input_nodes: torch.LongTensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
    ) -> torch.Tensor:
        n_graph, n_node = input_nodes.size()[:2]

        node_feature = (  # 节点特征 + 图标记
            self.atom_encoder(input_nodes).sum(dim=-2)  # [n_graph, n_node, n_hidden]
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphormerGraphAttnBias(nn.Module):
    """
    为每个头计算注意力偏置。
    """
    # 初始化函数，接受一个 GraphormerConfig 对象作为参数
    def __init__(self, config: GraphormerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置头数为配置中的注意力头数
        self.num_heads = config.num_attention_heads
        # 设置多跳最大距离为配置中的多跳最大距离
        self.multi_hop_max_dist = config.multi_hop_max_dist

        # 不改变边特征嵌入学习，因为边嵌入表示为原始特征和最短路径的组合
        # 创建边编码器，使用 nn.Embedding，边数为配置中的边数加1，维度为注意力头数，填充索引为0
        self.edge_encoder = nn.Embedding(config.num_edges + 1, config.num_attention_heads, padding_idx=0)

        # 设置边类型为配置中的边类型
        self.edge_type = config.edge_type
        # 如果边类型为 "multi_hop"
        if self.edge_type == "multi_hop":
            # 创建边距离编码器，使用 nn.Embedding，边距离数乘以注意力头数的平方，维度为1
            self.edge_dis_encoder = nn.Embedding(
                config.num_edge_dis * config.num_attention_heads * config.num_attention_heads,
                1,
            )

        # 创建空间位置编码器，使用 nn.Embedding，空间位置数为配置中的空间位置数，维度为注意力头数，填充索引为0
        self.spatial_pos_encoder = nn.Embedding(config.num_spatial, config.num_attention_heads, padding_idx=0)

        # 创建图标记虚拟距离编码器，使用 nn.Embedding，虚拟距离数为1，维度为注意力头数
        self.graph_token_virtual_distance = nn.Embedding(1, config.num_attention_heads)

    # 前向传播函数
    def forward(
        self,
        input_nodes: torch.LongTensor,
        attn_bias: torch.Tensor,
        spatial_pos: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
    # 定义函数的输入和输出类型
    ) -> torch.Tensor:
        # 获取输入节点的数量和图的数量
        n_graph, n_node = input_nodes.size()[:2]
        # 复制注意力偏置
        graph_attn_bias = attn_bias.clone()
        # 在第1维度上添加维度，然后重复多次，扩展为 [n_graph, n_head, n_node+1, n_node+1]
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )

        # 空间位置偏置
        # 将 [n_graph, n_node, n_node, n_head] 转置为 [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        # 更新图的注意力偏置
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # 重置空间位置
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # 边特征
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()

            spatial_pos_[spatial_pos_ == 0] = 1  # 将填充值设为1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                input_edges = input_edges[:, :, :, : self.multi_hop_max_dist, :]
            
            # 对边特征进行编码和平均
            input_edges = self.edge_encoder(input_edges).mean(-2)
            max_dist = input_edges.size(-2)
            edge_input_flat = input_edges.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :],
            )
            input_edges = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(
                1, 2, 3, 0, 4
            )
            input_edges = (input_edges.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # 对注意力边类型进行编码和平均，然后转置
            input_edges = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        # 更新图的注意力偏置
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + input_edges
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # 重置

        # 返回更新后的图的注意力偏置
        return graph_attn_bias
class GraphormerMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, config: GraphormerConfig):
        # 初始化函数，接受一个配置参数
        super().__init__()
        # 设置嵌入维度
        self.embedding_dim = config.embedding_dim
        # 如果配置中指定了kdim，则使用配置中的值，否则使用嵌入维度
        self.kdim = config.kdim if config.kdim is not None else config.embedding_dim
        # 如果配置中指定了vdim，则使用配置中的值，否则使用嵌入维度
        self.vdim = config.vdim if config.vdim is not None else config.embedding_dim
        # 检查是否query、key和value的维度相同
        self.qkv_same_dim = self.kdim == config.embedding_dim and self.vdim == config.embedding_dim

        # 设置注意力头的数量
        self.num_heads = config.num_attention_heads
        # 创建一个dropout模块用于注意力
        self.attention_dropout_module = torch.nn.Dropout(p=config.attention_dropout, inplace=False)

        # 计算每个头的维度
        self.head_dim = config.embedding_dim // config.num_attention_heads
        # 检查嵌入维度是否可以被头的数量整除
        if not (self.head_dim * config.num_attention_heads == self.embedding_dim):
            raise AssertionError("The embedding_dim must be divisible by num_heads.")
        # 设置缩放因子
        self.scaling = self.head_dim**-0.5

        # 是否使用自注意力机制
        self.self_attention = True  # config.self_attention
        # 如果不使用自注意力机制，则抛出异常
        if not (self.self_attention):
            raise NotImplementedError("The Graphormer model only supports self attention for now.")
        # 如果使用自注意力机制且query、key和value的维度不相同，则抛出异常
        if self.self_attention and not self.qkv_same_dim:
            raise AssertionError("Self-attention requires query, key and value to be of the same size.")

        # 创建线性层用于k的投影
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        # 创建线性层用于v的投影
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        # 创建线性层用于q的投影
        self.q_proj = quant_noise(
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        # 创建线性层用于输出的投影
        self.out_proj = quant_noise(
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        # 是否进行ONNX跟踪
        self.onnx_trace = False

    def reset_parameters(self):
        # 重置参数
        if self.qkv_same_dim:
            # 如果query、key和value的维度相同，则使用缩放初始化
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            # 否则使用普通初始化
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        # 初始化输出投影层的权重
        nn.init.xavier_uniform_(self.out_proj.weight)
        # 如果输出投影层有偏置，则初始化为0
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
    # 定义一个方法用于实现注意力机制的前向传播
    def forward(
        self,
        query: torch.LongTensor,  # 查询向量，类型为 LongTensor
        key: Optional[torch.Tensor],  # 键向量，可选的 Tensor 类型
        value: Optional[torch.Tensor],  # 值向量，可选的 Tensor 类型
        attn_bias: Optional[torch.Tensor],  # 注意力偏置，可选的 Tensor 类型
        key_padding_mask: Optional[torch.Tensor] = None,  # 键的填充掩码，可选的 Tensor 类型，默认为 None
        need_weights: bool = True,  # 是否需要权重，默认为 True
        attn_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选的 Tensor 类型，默认为 None
        before_softmax: bool = False,  # 是否在 softmax 之前，默认为 False
        need_head_weights: bool = False,  # 是否需要头部权重，默认为 False
    # 定义一个方法用于应用稀疏掩码到注意力权重
    def apply_sparse_mask(self, attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int) -> torch.Tensor:
        # 返回注意力权重
        return attn_weights
# 定义一个名为GraphormerGraphEncoderLayer的类，继承自nn.Module
class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(self, config: GraphormerConfig) -> None:
        super().__init__()

        # 初始化参数
        self.embedding_dim = config.embedding_dim
        self.num_attention_heads = config.num_attention_heads
        self.q_noise = config.q_noise
        self.qn_block_size = config.qn_block_size
        self.pre_layernorm = config.pre_layernorm

        # 创建一个Dropout模块，用于在训练过程中随机丢弃一定比例的输入数据
        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)

        # 创建一个Dropout模块，用于在激活函数中随机丢弃一定比例的数据
        self.activation_dropout_module = torch.nn.Dropout(p=config.activation_dropout, inplace=False)

        # 初始化激活函数
        self.activation_fn = ACT2FN[config.activation_fn]
        # 创建一个GraphormerMultiheadAttention对象，用于自注意力机制
        self.self_attn = GraphormerMultiheadAttention(config)

        # 与自注意力层相关联的LayerNorm层
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        # 创建一个全连接层，用于处理输入数据
        self.fc1 = self.build_fc(
            self.embedding_dim,
            config.ffn_embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )
        # 创建一个全连接层，用于处理中间数据
        self.fc2 = self.build_fc(
            config.ffn_embedding_dim,
            self.embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )

        # 与位置智能前馈神经网络相关联的LayerNorm层
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    # 构建全连接层
    def build_fc(
        self, input_dim: int, output_dim: int, q_noise: float, qn_block_size: int
    ) -> Union[nn.Module, nn.Linear, nn.Embedding, nn.Conv2d]:
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    # 前向传播函数
    def forward(
        self,
        input_nodes: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original
        Transformer implementation.
        """
        # 保存输入节点作为残差连接的一部分
        residual = input_nodes
        # 如果设置了 pre_layernorm 标志，则在 self-attention 模块之前应用 LayerNorm
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        # 使用 self-attention 模块处理输入节点
        input_nodes, attn = self.self_attn(
            query=input_nodes,
            key=input_nodes,
            value=input_nodes,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        # 对输出结果应用 dropout 模块
        input_nodes = self.dropout_module(input_nodes)
        # 将残差连接添加到输出结果中
        input_nodes = residual + input_nodes
        # 如果未设置 pre_layernorm 标志，则在 self-attention 模块之后应用 LayerNorm
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        # 保存输入节点作为残差连接的一部分
        residual = input_nodes
        # 如果设置了 pre_layernorm 标志，则在最终层之前应用 LayerNorm
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        # 应用激活函数到第一个全连接层
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        # 对输出结果应用激活函数的 dropout 模块
        input_nodes = self.activation_dropout_module(input_nodes)
        # 应用第二个全连接层
        input_nodes = self.fc2(input_nodes)
        # 对输出结果应用 dropout 模块
        input_nodes = self.dropout_module(input_nodes)
        # 将残差连接添加到输出结果中
        input_nodes = residual + input_nodes
        # 如果未设置 pre_layernorm 标志，则在最终层之后应用 LayerNorm
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)

        # 返回处理后的输入节点和注意力权重
        return input_nodes, attn
class GraphormerGraphEncoder(nn.Module):
    # 定义 Graphormer 图编码器类，继承自 nn.Module
    def __init__(self, config: GraphormerConfig):
        # 初始化函数，接受一个 GraphormerConfig 类型的参数 config
        super().__init__()
        # 调用父类的初始化函数

        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        # 创建一个 Dropout 模块，使用配置中的 dropout 参数
        self.layerdrop = config.layerdrop
        # 从配置中获取 layerdrop 参数
        self.embedding_dim = config.embedding_dim
        # 从配置中获取 embedding_dim 参数
        self.apply_graphormer_init = config.apply_graphormer_init
        # 从配置中获取 apply_graphormer_init 参数
        self.traceable = config.traceable
        # 从配置中获取 traceable 参数

        self.graph_node_feature = GraphormerGraphNodeFeature(config)
        # 创建一个 GraphormerGraphNodeFeature 对象，使用配置参数
        self.graph_attn_bias = GraphormerGraphAttnBias(config)
        # 创建一个 GraphormerGraphAttnBias 对象，使用配置参数

        self.embed_scale = config.embed_scale
        # 从配置中获取 embed_scale 参数

        if config.q_noise > 0:
            # 如果配置中的 q_noise 大于 0
            self.quant_noise = quant_noise(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                config.q_noise,
                config.qn_block_size,
            )
            # 创建一个量化噪声模块，使用配置中的参数
        else:
            self.quant_noise = None
            # 否则将量化噪声模块设为 None

        if config.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
            # 如果配置中的 encoder_normalize_before 为 True，则创建一个 LayerNorm 模块
        else:
            self.emb_layer_norm = None
            # 否则将 LayerNorm 模块设为 None

        if config.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
            # 如果配置中的 pre_layernorm 为 True，则创建一个 LayerNorm 模块

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
            # 如果 layerdrop 大于 0.0，则创建一个 LayerDropModuleList 对象
        else:
            self.layers = nn.ModuleList([])
            # 否则创建一个空的 ModuleList 对象
        self.layers.extend([GraphormerGraphEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 将 GraphormerGraphEncoderLayer 对象添加到 layers 中，数量为 num_hidden_layers

        # 在构建模型后应用模型参数的初始化
        if config.freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")
            # 如果配置中的 freeze_embeddings 为 True，则抛出未实现的错误

        for layer in range(config.num_trans_layers_to_freeze):
            # 遍历需要冻结的转换层
            m = self.layers[layer]
            # 获取当前层
            if m is not None:
                # 如果当前层不为 None
                for p in m.parameters():
                    p.requires_grad = False
                    # 将当前层的参数设置为不需要梯度计算

    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_bias: torch.Tensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        # 前向传播函数，接受多个输入参数
        ) -> Tuple[Union[torch.Tensor, List[torch.LongTensor]], torch.Tensor]:
        # 定义函数的输入参数和返回值类型，包括一个张量或长整型张量的列表和一个张量
        # 计算填充掩码，用于多头注意力
        data_x = input_nodes
        # 获取输入节点的图和节点数量
        n_graph, n_node = data_x.size()[:2]
        # 创建填充掩码，将节点序列中值为0的位置标记为True
        padding_mask = (data_x[:, :, 0]).eq(0)
        # 创建一个与填充掩码相同形状的全零张量，用于标记类别节点
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        # 将类别节点填充掩码与普通节点填充掩码拼接在一起
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        # 计算注意力偏置
        attn_bias = self.graph_attn_bias(input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type)

        # 如果存在令牌嵌入，则使用令牌嵌入替换输入节点
        if token_embeddings is not None:
            input_nodes = token_embeddings
        else:
            # 否则使用图节点特征函数处理输入节点
            input_nodes = self.graph_node_feature(input_nodes, in_degree, out_degree)

        # 如果存在扰动，则将扰动添加到输入节点的非类别节点上
        if perturb is not None:
            input_nodes[:, 1:, :] += perturb

        # 如果存在嵌入缩放因子，则将输入节点乘以该因子
        if self.embed_scale is not None:
            input_nodes = input_nodes * self.embed_scale

        # 如果存在量化噪声，则对输入节点应用量化噪声
        if self.quant_noise is not None:
            input_nodes = self.quant_noise(input_nodes)

        # 如果存在嵌入层归一化，则对输入节点进行归一化
        if self.emb_layer_norm is not None:
            input_nodes = self.emb_layer_norm(input_nodes)

        # 对输入节点应用丢弃模块
        input_nodes = self.dropout_module(input_nodes)

        # 转置输入节点的维度
        input_nodes = input_nodes.transpose(0, 1)

        # 初始化内部状态列表
        inner_states = []
        # 如果不仅返回最后一个状态
        if not last_state_only:
            inner_states.append(input_nodes)

        # 遍历所有层
        for layer in self.layers:
            # 对每一层进行前向传播
            input_nodes, _ = layer(
                input_nodes,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            # 如果不仅返回最后一个状态
            if not last_state_only:
                inner_states.append(input_nodes)

        # 获取图表示
        graph_rep = input_nodes[0, :, :]

        # 如果仅返回最后一个状态
        if last_state_only:
            inner_states = [input_nodes]

        # 如果可追踪，则返回内部状态和图表示的堆叠张量
        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            # 否则返回内部状态列表和图表示
            return inner_states, graph_rep
class GraphormerDecoderHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        """num_classes should be 1 for regression, or the number of classes for classification"""
        # 初始化学习偏置参数
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        # 创建线性分类器
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.num_classes = num_classes

    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
        # 使用分类器进行前向传播
        input_nodes = self.classifier(input_nodes)
        # 添加学习偏置参数
        input_nodes = input_nodes + self.lm_output_learned_bias
        return input_nodes


class GraphormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GraphormerConfig
    base_model_prefix = "graphormer"
    main_input_name_nodes = "input_nodes"
    main_input_name_edges = "input_edges"

    def normal_(self, data: torch.Tensor):
        # 使用正态分布初始化数据，并将参数转移到 CPU 上以保持 RNG 一致性
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    def init_graphormer_params(self, module: Union[nn.Linear, nn.Embedding, GraphormerMultiheadAttention]):
        """
        Initialize the weights specific to the Graphormer Model.
        """
        if isinstance(module, nn.Linear):
            # 初始化线性层的权重和偏置
            self.normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            # 初始化嵌入层的权重
            self.normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, GraphormerMultiheadAttention):
            # 初始化多头注意力机制的权重
            self.normal_(module.q_proj.weight.data)
            self.normal_(module.k_proj.weight.data)
            self.normal_(module.v_proj.weight.data)

    def _init_weights(
        self,
        module: Union[
            nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, GraphormerMultiheadAttention, GraphormerGraphEncoder
        ],
        ):
        """
        Initialize the weights
        """
        # 检查模块是否为线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 如果存在偏置项，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 检查模块是否为嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 如果存在填充索引，则将对应权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 检查模块是否为自定义的多头注意力层
        elif isinstance(module, GraphormerMultiheadAttention):
            # 分别初始化查询、键、值的投影权重为正态分布
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            # 重置参数
            module.reset_parameters()
        # 检查模块是否为 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为零
            module.bias.data.zero_()
            # 初始化权重为1
            module.weight.data.fill_(1.0)
        # 检查模块是否为自定义的图编码器
        elif isinstance(module, GraphormerGraphEncoder):
            # 如果需要应用初始化，则调用初始化函数
            if module.apply_graphormer_init:
                module.apply(self.init_graphormer_params)

        # 再次检查模块是否为 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为零
            module.bias.data.zero_()
            # 初始化权重为1
            module.weight.data.fill_(1.0)
class GraphormerModel(GraphormerPreTrainedModel):
    """The Graphormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    GraphormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.
    """

    def __init__(self, config: GraphormerConfig):
        # 调用父类的构造函数，初始化模型配置
        super().__init__(config)
        # 设置最大节点数
        self.max_nodes = config.max_nodes

        # 初始化图编码器
        self.graph_encoder = GraphormerGraphEncoder(config)

        # 是否共享输入输出嵌入
        self.share_input_output_embed = config.share_input_output_embed
        self.lm_output_learned_bias = None

        # 在微调期间设置为True
        self.load_softmax = not getattr(config, "remove_head", False)

        # 线性变换层，用于LM头部
        self.lm_head_transform_weight = nn.Linear(config.embedding_dim, config.embedding_dim)
        # 激活函数
        self.activation_fn = ACT2FN[config.activation_fn]
        # LayerNorm层
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

        # 执行后续初始化操作
        self.post_init()

    def reset_output_layer_parameters(self):
        # 重置输出层参数
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_bias: torch.Tensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        perturb: Optional[torch.FloatTensor] = None,
        masked_tokens: None = None,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.LongTensor], BaseModelOutputWithNoAttention]:
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取内部状态和图表示
        inner_states, graph_rep = self.graph_encoder(
            input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos, attn_edge_type, perturb=perturb
        )

        # 获取最后一个内部状态，然后反转Batch和Graph长度
        input_nodes = inner_states[-1].transpose(0, 1)

        # 仅投影掩码标记
        if masked_tokens is not None:
            raise NotImplementedError

        # LayerNorm和激活函数
        input_nodes = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(input_nodes)))

        # 投影回词汇表大小
        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, "weight"):
            input_nodes = torch.nn.functional.linear(input_nodes, self.graph_encoder.embed_tokens.weight)

        # 如果不返回字典，则返回元组
        if not return_dict:
            return tuple(x for x in [input_nodes, inner_states] if x is not None)
        return BaseModelOutputWithNoAttention(last_hidden_state=input_nodes, hidden_states=inner_states)

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        # 返回编码器支持的最大输出长度
        return self.max_nodes
class GraphormerForGraphClassification(GraphormerPreTrainedModel):
    """
    This model can be used for graph-level classification or regression tasks.

    It can be trained on
    - regression (by setting config.num_classes to 1); there should be one float-type label per graph
    - one task classification (by setting config.num_classes to the number of classes); there should be one integer
      label per graph
    - binary multi-task classification (by setting config.num_classes to the number of labels); there should be a list
      of integer labels for each graph.
    """

    def __init__(self, config: GraphormerConfig):
        # 调用父类的构造函数，初始化模型
        super().__init__(config)
        # 创建 GraphormerModel 对象作为编码器
        self.encoder = GraphormerModel(config)
        # 从配置中获取嵌入维度
        self.embedding_dim = config.embedding_dim
        # 从配置中获取类别数量
        self.num_classes = config.num_classes
        # 创建 GraphormerDecoderHead 对象作为分类器
        self.classifier = GraphormerDecoderHead(self.embedding_dim, self.num_classes)
        # 设置为编码器-解码器模型
        self.is_encoder_decoder = True

        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_bias: torch.Tensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # 如果 return_dict 为 None，则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 编码器的前向传播
        encoder_outputs = self.encoder(
            input_nodes,
            input_edges,
            attn_bias,
            in_degree,
            out_degree,
            spatial_pos,
            attn_edge_type,
            return_dict=True,
        )
        # 获取编码器输出和隐藏状态
        outputs, hidden_states = encoder_outputs["last_hidden_state"], encoder_outputs["hidden_states"]

        # 通过分类器获取头部输出
        head_outputs = self.classifier(outputs)
        # 获取 logits
        logits = head_outputs[:, 0, :].contiguous()

        loss = None
        if labels is not None:
            # 创建标签的掩码，排除 NaN 值
            mask = ~torch.isnan(labels)

            if self.num_classes == 1:  # regression
                # 使用均方误差损失函数
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif self.num_classes > 1 and len(labels.shape) == 1:  # One task classification
                # 使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[mask].view(-1, self.num_classes), labels[mask].view(-1))
            else:  # Binary multi-task classification
                # 使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss(reduction="sum")
                loss = loss_fct(logits[mask], labels[mask])

        if not return_dict:
            # 如果不返回字典，则返回损失、logits 和隐藏状态
            return tuple(x for x in [loss, logits, hidden_states] if x is not None)
        # 返回序列分类器输出
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=None)
```